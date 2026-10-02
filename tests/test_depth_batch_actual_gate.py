"""Tiny writer/reference stubs; no native full-resolution or model execution."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_depth_actual_test", infra/"depth_batch_actual_gate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


@dataclass
class Record:
    index: int
    raw_depth_m: np.ndarray
    aligned_depth_m: np.ndarray
    scale: float
    shift: float
    valid_count: int


def predicted_inputs(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 2); monkeypatch.setattr(gate, "WIDTH", 3)
    base = tmp_path/"outputs/episode_000015"; frames = []
    (base/"depth_full").mkdir(parents=True); (base/"scale_smoke").mkdir(); (base/"cari_inputs").mkdir()
    for i in range(9):
        d = np.array([[0., np.nan, 2.], [3., 4., 5.]], np.float32)
        valid = np.array([[True, False, True], [True, True, False]], bool)
        path = base/f"depth_full/{i:06d}.npz"
        np.savez(path, depth=d, mask=valid, intrinsics=np.eye(3), frame_index=np.array(i))
        frames.append({"frame_index": i, "output_sha256": gate.sha256(path)})
    frames += [{"frame_index": i} for i in range(9, 501)]
    flags = {"status": "pass", "input_track": "track_1", "input_sha256": "a"*64, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    depth = flags | {"stage": "monocular_moge2_full_video", "script_sha256": gate.LEGACY_DEPTH_SHA,
        "total_video_frames": 501, "input_dataset_revision": gate.DATASET_REVISION, "frames": frames}
    dp = base/"depth_full/report.json"; dp.write_text(json.dumps(depth)); monkeypatch.setattr(gate, "DEPTH_REPORT_SHA", gate.sha256(dp))
    alignment = flags | {"stage": "predicted_human_anchored_moge2_pointmaps", "episode_index": 15,
        "script_sha256": gate.ALIGNMENT_SCRIPT_SHA, "frame_indices": [0, 250, 500], "depth_alignment": {"shared_scale": .91321}}
    ap = base/"scale_smoke/report.json"; ap.write_text(json.dumps(alignment)); monkeypatch.setattr(gate, "ALIGNMENT_REPORT_SHA", gate.sha256(ap))
    hp = base/"cari_inputs/aligned_depth.h5"; hp.write_bytes(b"fakefrozenH5"); monkeypatch.setattr(gate, "REFERENCE_H5_SHA", gate.sha256(hp))
    prep = flags | {"stage": "world_reward_native_cari_inputs", "producer_revision": gate.LEGACY_PREP_REVISION,
        "script_sha256": gate.LEGACY_PREP_SHA, "frames": 501, "original_frame_coverage_verified": True,
        "input_report_sha256": {"depth": gate.sha256(dp), "alignment": gate.sha256(ap)}, "depth_h5": str(hp),
        "file_sha256": {"depth_h5": gate.sha256(hp)}, "depth_validation": {"validation_mode": "exhaustive", "frame_counts": {"cam": 501}, "frame_shapes": {"cam": [2, 3]}}}
    pp = base/"cari_inputs/report.json"; pp.write_text(json.dumps(prep))
    return pp, dp, ap, hp


def test_actual_operations_preserve_valid_zero_count_dtype_and_original_frame(gate, tmp_path, monkeypatch):
    predicted_inputs(gate, tmp_path, monkeypatch)
    reports, hashes, records, reference, sources, legacy = gate.load_inputs(tmp_path, SimpleNamespace(DepthFrameRecord=Record))
    assert legacy is True and len(records) == 9 and len(sources) == 13
    for i, record in enumerate(records):
        assert record.index == i and record.valid_count == 4  # Not count_positive ==3!
        assert record.raw_depth_m.dtype == np.float32 and np.isfinite(record.raw_depth_m).all()
        assert np.array_equal(record.raw_depth_m, [[0., 0., 2.], [3., 4., 0.]])
        assert np.array_equal(record.aligned_depth_m, record.raw_depth_m*.91321) and record.shift == 0.
    assert hashes["reference_H5"] == gate.sha256(reference)


@pytest.mark.parametrize("fault", ["other_legacy_rev", "other_legacy_source", "depth_missing_source", "depth_episode_wrong", "video", "full_count", "duplicate", "scale", "h5sha", "npzsha", "oracle"])
def test_actual_source_bindings_fail_closed(gate, tmp_path, monkeypatch, fault):
    pp, dp, ap, hp = predicted_inputs(gate, tmp_path, monkeypatch)
    p, d, a = [json.loads(x.read_text()) for x in (pp, dp, ap)]
    if fault == "other_legacy_rev": p["producer_revision"] = "b"*40
    elif fault == "other_legacy_source": p["script_sha256"] = "b"*64
    elif fault == "depth_missing_source": d.pop("script_sha256")
    elif fault == "depth_episode_wrong": d["episode_index"] = 0
    elif fault == "video": a["input_sha256"] = "b"*64
    elif fault == "full_count": p["frames"] = 500
    elif fault == "duplicate": d["frames"][9]["frame_index"] = 8
    elif fault == "scale": a["depth_alignment"]["shared_scale"] = -1.
    elif fault == "h5sha": p["file_sha256"]["depth_h5"] = "b"*64
    elif fault == "npzsha": d["frames"][0]["output_sha256"] = "b"*64
    else: p["ground_truth_used"] = True
    for path, value in ((pp, p), (dp, d), (ap, a)): path.write_text(json.dumps(value))
    with pytest.raises(ValueError): gate.load_inputs(tmp_path, SimpleNamespace(DepthFrameRecord=Record))


@pytest.mark.parametrize("mode,sizes", [("single", [1]*9), ("batch8", [8, 1])])
def test_native_writer_actual_camera_identity_and_tail(gate, tmp_path, mode, sizes):
    records = [Record(i, np.ones((2, 3), np.float32), np.ones((2, 3), np.float32)*.91, .91, 0., 6) for i in range(9)]
    seen = []; events = []; identity = {"actual": "source", "scale": .91}
    class Writer:
        def __init__(self, path, names, **kwargs):
            assert names == {"actualcam": [f"{i:06d}" for i in range(9)]}
            assert kwargs == {"alignment_method": gate.METHOD, "alignment_input_identity": identity, "encoding_workers": 8}
        def __enter__(self): return self
        def __exit__(self, *args): events.append("close")
        def write_frame(self, camera, index, raw, aligned, **kwargs):
            assert camera == "actualcam" and raw is records[index].raw_depth_m
            assert kwargs == {"scale": .91, "shift": 0., "valid_count": 6}; seen.append([index])
        def write_frames(self, camera, values): seen.append([r.index for r in values])
        def mark_complete(self): events.append("complete")
    gate.run_write(SimpleNamespace(MHRDepthH5Writer=Writer), tmp_path/"unused", records, mode, "actualcam", identity)
    assert [len(x) for x in seen] == sizes and sum(seen, []) == list(range(9)) and events == ["complete", "close"]


class Dataset:
    def __init__(self, value): self.value = value; self.dtype = getattr(value, "dtype", None); self.shape = getattr(value, "shape", None)
    def __getitem__(self, key): return self.value[key]


def reference_fixture(gate, monkeypatch):
    records = [Record(i, np.array([[0., 2.]], np.float32), np.array([[0., 1.82]], np.float32), .91, 0., 2) for i in range(9)]
    identity = {"alignment": "actual", "ground_truth_used": False}; data = {"frame_names": {"cam": None}, "frame_names/cam": Dataset(np.array([f"{i:06d}" for i in range(501)]))}
    for key, dtype, value in (("scale", np.float32, .91), ("shift", np.float32, 0.), ("valid_count", np.int32, 2)):
        data[f"alignment/cam/{key}"] = Dataset(np.full(501, value, dtype))
    for kind in ("raw", "aligned"):
        values = np.empty(501, object)
        for r in records:
            original = r.raw_depth_m if kind == "raw" else r.aligned_depth_m
            values[r.index] = np.frombuffer((original*1000.).astype(np.uint16).tobytes(), np.uint8)
        data[f"{kind}/cam"] = Dataset(values)
    attrs = {"complete": True, "depth_alignment_method": gate.METHOD, "identity_attr": json.dumps(identity)}
    class File:
        def __init__(self): self.attrs = attrs
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, name): return data[name]
    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda *a: File()))
    native = SimpleNamespace(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE="identity_attr", decode_depth_png_uint16=lambda payload: np.frombuffer(payload, np.uint16).reshape(1, 2))
    return records, identity, data, attrs, native


def test_full_original_names_but_only_nine_reference_payloads(gate, monkeypatch):
    records, identity, _, _, native = reference_fixture(gate, monkeypatch)
    camera, actual, receipt = gate.read_receipt(native, "unused", records, total=501)
    assert camera == "cam" and actual == identity and receipt["frame_names"] == [f"{i:06d}" for i in range(9)]
    assert len(receipt["encoded_payloads"]["raw"]) == 9 and receipt["alignment_metadata"]["valid_count"]["values"] == [2]*9


@pytest.mark.parametrize("fault", ["names", "dtype", "count", "method", "unsafe_identity", "identity", "payload", "incomplete"])
def test_reference_schema_or_byte_contract_fails(gate, monkeypatch, fault):
    records, identity, data, attrs, native = reference_fixture(gate, monkeypatch)
    if fault == "names": data["frame_names/cam"].value[8] = "000007"
    elif fault == "dtype": data["alignment/cam/scale"] = Dataset(np.full(501, .91, np.float64))
    elif fault == "count": data["alignment/cam/valid_count"].value[0] = 1
    elif fault == "method": attrs["depth_alignment_method"] = "other"
    elif fault == "unsafe_identity": attrs["identity_attr"] = '__import__("os")'
    elif fault == "identity": attrs["identity_attr"] = '["notdict"]'
    elif fault == "payload": data["raw/cam"].value[0] = np.array([0, 0, 0, 0], np.uint8)
    else: attrs["complete"] = False
    with pytest.raises(ValueError): gate.read_receipt(native, "unused", records, total=501)


def test_exhaustive_validator_all_selected_frames_no_skip(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 2); monkeypatch.setattr(gate, "WIDTH", 3)
    calls = []; records = [None]*9
    def validate(path, **kwargs):
        calls.append(kwargs); return {"validation_mode": "exhaustive", "frame_counts": {"cam": 9}, "frame_shapes": {"cam": (2, 3)}}
    monkeypatch.setattr(gate, "read_receipt", lambda *a, **k: ("cam", {}, {"reference": "identical"}))
    _, receipt = gate.validate_receipt(SimpleNamespace(validate_depth_h5=validate), tmp_path/"unused", records, "cam", {"actual": True})
    assert receipt == {"reference": "identical"} and calls == [{"expected_cameras": ["cam"], "expected_alignment_method": gate.METHOD,
        "expected_alignment_input_identity": {"actual": True}, "validation_workers": 8}]


@pytest.mark.parametrize("fault", [None, "reference", "speed", "mutation"])
def test_balanced_actual_trials_reference_parity_tail_cleanup_and_gate(gate, tmp_path, monkeypatch, fault):
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setattr(gate, "HEIGHT", 2); monkeypatch.setattr(gate, "WIDTH", 3)
    identity = {"depth_backend": "moge2", "depth_model_id": "Ruicheng/moge-2-vitl-normal",
        "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340", "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
        "alignment_report_sha256": "a"*64, "monocular_depth": {"backend": "moge2", "model_id": "Ruicheng/moge-2-vitl-normal",
        "model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340", "source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030"},
        "ground_truth_used": False, "alignment": "one_predicted_human_anchored_clip_scalar_no_offset"}
    records = [Record(i, np.ones((2, 3), np.float32), np.ones((2, 3), np.float32), 1., 0., 6) for i in range(9)]
    reports = {"prep": {"depth_validation": {"validation_mode": "exhaustive", "frame_counts": {"cam": 501}, "frame_shapes": {"cam": [2, 3]}}}}
    monkeypatch.setattr(gate, "load_inputs", lambda *a: (reports, {"alignment": "a"*64, "reference_H5": "b"*64}, records, tmp_path/"reference", [], True))
    def receipt(count, changed=False):
        return {"frame_names": [f"{i:06d}" for i in range(count)], "encoded_payloads": {"raw": ["changed" if changed else "reference"]*count},
            "alignment_metadata": {"scale": {"dtype": "float32", "values": [1.]*count,
                "sha256": hashlib.sha256(np.ones(count, np.float32).tobytes()).hexdigest()}}}
    monkeypatch.setattr(gate, "read_receipt", lambda *a, **k: ("cam", identity, receipt(9)))
    calls = []
    def write(native, target, values, mode, camera, actual_identity):
        assert camera == "cam" and actual_identity == identity
        target.write_bytes(b"stubH5"); calls.append((mode, len(values)))
        if fault == "mutation": values[0].raw_depth_m[0, 0] += .00001
        return (1.24 if fault == "speed" else 2.) if mode == "single" else 1.
    def validate(native, target, values, camera, actual_identity):
        assert actual_identity == identity
        return .5, receipt(len(values), changed=fault == "reference" and target.name.startswith("repeat"))
    monkeypatch.setattr(gate, "run_write", write); monkeypatch.setattr(gate, "validate_receipt", validate)
    report = {"trials": []}
    if fault:
        with pytest.raises(ValueError): gate.run_gate(None, tmp_path, report, tmp_path/"report.json")
        assert report.get("status") != "pass"
    else:
        gate.run_gate(None, tmp_path, report, tmp_path/"report.json")
        assert report["status"] == "pass" and report["reference_byte_parity"] is True and report["median_write_speedup"] == 2.
        assert calls == [("single", 2), ("batch8", 2), ("single", 9), ("batch8", 9), ("batch8", 9), ("single", 9)]
        assert [t["order"] for t in report["trials"]] == [["single", "batch8"], ["batch8", "single"]]
    assert report["temporary_H5_removed"] is True and not list(tmp_path.glob("depth-batch-actual-*"))


def test_main_early_fail_freezes_report_no_models_or_outputs_altered(gate, tmp_path, monkeypatch):
    out = tmp_path/"results/depth-batch-actual-v1"; out.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); iterdir = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else iterdir(p))
    with pytest.raises(ValueError, match="source SHA"): gate.main([])
    path = out/"report.json"; frozen = path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["challenge_inputs_used"] is True and report["ground_truth_used"] is False
    assert report["tail_frames"] == 1 and report["selected_frame_indices"] == list(range(9)) and report["budget_seconds"] == 240
    assert report["adoption_performed"] is False and report["whole_preparation_throughput_verified"] is False
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_wrapper_specific_readonly_sources_no_video_GTorGPU(gate):
    wrapper = Path(gate.__file__).with_name("run_depth_batch_actual_gate.sh").read_text(); source = Path(gate.__file__).read_text()
    assert "243s docker run" in wrapper and "--network none --memory 16g --cpus 4" in wrapper and "--gpus" not in wrapper
    assert 'for INDEX in {0..8}' in wrapper and 'readonly")' in wrapper and 'chown -R' not in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper and all(w not in wrapper for w in ("weights", "data/", "body_full", "automatic_masks"))
    assert "int(valid.sum())" in source and "raw = np.where(valid, d, 0.); aligned = raw*scale" in source
    assert "validate_payloads=False" not in source and "pickle" not in source.replace("allow_pickle=False", "")
    with pytest.raises(SystemExit): gate.main(["--episode", "0"])

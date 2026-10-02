"""Tiny orchestration stubs and procedural contracts, no native full-grid work."""
from dataclasses import dataclass
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/depth_batch_gate.py"
    spec = importlib.util.spec_from_file_location("wr_depth_batch_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


@dataclass
class Record:
    index: int
    raw_depth_m: np.ndarray
    aligned_depth_m: np.ndarray
    scale: float
    shift: float
    valid_count: int


def test_procedural_depth_seed_positive_float32_no_inputs_modified(gate):
    a = gate.procedural_depths(32, 24, 4); b = gate.procedural_depths(32, 24, 4)
    assert all(x.dtype == np.float32 and x.shape == (24, 32) and np.min(x) > 0 and np.max(x) < 65.535 for x in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and not np.array_equal(a[0], a[1])
    copies = [x.copy() for x in a]; records = gate.records_for(SimpleNamespace(DepthFrameRecord=Record), a)
    for index, record in enumerate(records):
        assert record.index == index and record.valid_count == 24*32 and record.shift == 0. and record.scale == .83
        assert np.array_equal(record.aligned_depth_m, a[index]*np.float32(.83)) and np.array_equal(a[index], copies[index])


@pytest.mark.parametrize("mode,sizes", [("single", [1]*17), ("batch8", [8, 8, 1])])
def test_identical_records_writer_semantics_workers8_mark_complete(gate, tmp_path, mode, sizes):
    records = gate.records_for(SimpleNamespace(DepthFrameRecord=Record), [np.ones((2, 3), np.float32) for _ in range(17)])
    calls = []; seen = []
    class Writer:
        def __init__(self, path, names, **kwargs):
            assert names == {gate.CAMERA: [f"{i:06d}" for i in range(17)]}
            assert kwargs["encoding_workers"] == 8 and kwargs["alignment_input_identity"] == gate.IDENTITY
        def __enter__(self): return self
        def __exit__(self, *args): calls.append("exit")
        def write_frame(self, camera, index, raw, aligned, **kwargs):
            assert camera == gate.CAMERA and raw is records[index].raw_depth_m and aligned is records[index].aligned_depth_m
            assert kwargs == {"scale": .83, "shift": 0., "valid_count": 6}; seen.append([index])
        def write_frames(self, camera, values):
            assert camera == gate.CAMERA; seen.append([v.index for v in values])
            assert all(v is records[v.index] for v in values)
        def mark_complete(self): calls.append("mark_complete")
    assert gate.run_write(SimpleNamespace(MHRDepthH5Writer=Writer), tmp_path/"unused.h5", records, mode) >= 0
    assert [len(x) for x in seen] == sizes and sum(seen, []) == list(range(17)) and calls == ["mark_complete", "exit"]


def stub_gate(gate, monkeypatch, output, *, mismatch=False, speedup=2.):
    monkeypatch.setattr(gate, "procedural_depths", lambda: [np.ones((2, 3), np.float32) for _ in range(16)])
    native = SimpleNamespace(DepthFrameRecord=Record); calls = []
    def write(native, path, records, mode):
        path.write_bytes(b"temporaryH5stub"); calls.append((path.name, mode, len(records)))
        return speedup if mode == "single" else 1.
    def validate(native, path, records):
        data = {"frame_names": [f"{i:06d}" for i in range(len(records))], "encoded_payloads": {"raw": "same", "aligned": "same"}}
        if mismatch and "batch8" in path.name and "repeat" in path.name: data["encoded_payloads"]["aligned"] = "changed"
        return .5, data
    monkeypatch.setattr(gate, "run_write", write); monkeypatch.setattr(gate, "validate_and_receipt", validate)
    report = {"trials": []}; return native, calls, report


def test_balanced_AB_BA_parity_timing_tempcleanup(gate, tmp_path, monkeypatch):
    native, calls, report = stub_gate(gate, monkeypatch, tmp_path)
    gate.run_gate(native, tmp_path, report, tmp_path/"report.json")
    assert [(m, n) for _, m, n in calls] == [("single", 2), ("batch8", 2), ("single", 16), ("batch8", 16), ("batch8", 16), ("single", 16)]
    assert report["status"] == "pass" and report["median_write_speedup"] == 2. and report["payload_byte_parity"] is True
    assert report["temporary_H5_removed"] is True and list(tmp_path.iterdir()) == [tmp_path/"report.json"]


def test_mismatched_payload_aborts_preserves_partial_and_removes_temp(gate, tmp_path, monkeypatch):
    native, calls, report = stub_gate(gate, monkeypatch, tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="parity failed"): gate.run_gate(native, tmp_path, report, tmp_path/"report.json")
    assert len(report["trials"]) == 1 and report["trials"][0]["payload_byte_parity"] is False
    assert not list(tmp_path.glob("depth-batch-*")) and len(calls) == 4


def test_speed_fail_does_not_relax_gate_or_discard_parity(gate, tmp_path, monkeypatch):
    native, _, report = stub_gate(gate, monkeypatch, tmp_path, speedup=1.24)
    with pytest.raises(ValueError, match="below1.25"): gate.run_gate(native, tmp_path, report, tmp_path/"report.json")
    assert report["payload_byte_parity"] is True and report["median_write_speedup"] == 1.24


def test_native_source_missing_fails_before_dependency_import(gate, tmp_path):
    with pytest.raises(ValueError, match="source SHA"): gate.load_native(tmp_path)


@pytest.mark.parametrize("invalid_background", [False, True])
def test_exact_payload_order_metadata_and_native_quantization_receipt(gate, tmp_path, monkeypatch, invalid_background):
    import sys
    depths = [np.full((2, 3), 2.1234+i*.03, np.float32) for i in range(3)]
    if invalid_background:
        for depth in depths: depth[0, 0] = 0.
    records = gate.records_for(SimpleNamespace(DepthFrameRecord=Record), depths, count_positive=invalid_background)
    class Dataset:
        def __init__(self, value): self.value = value; self.shape = np.shape(value)
        def __getitem__(self, key): return self.value[key]
    data = {f"frame_names/{gate.CAMERA}": Dataset(np.array([f"{i:06d}" for i in range(3)]))}
    for key, dtype, values in (("scale", np.float32, [.83]*3), ("shift", np.float32, [0.]*3), ("valid_count", np.int32, [5 if invalid_background else 6]*3)):
        data[f"alignment/{gate.CAMERA}/{key}"] = Dataset(np.array(values, dtype=dtype))
    for kind in ("raw", "aligned"):
        payloads = []
        for r in records:
            depth = r.raw_depth_m if kind == "raw" else r.aligned_depth_m
            payloads.append(np.frombuffer((depth*1000.).astype(np.uint16).tobytes(), dtype=np.uint8))
        data[f"{kind}/{gate.CAMERA}"] = Dataset(np.array(payloads, dtype=object))
        data[f"{kind}/{gate.CAMERA}"].shape = (3,)
    class File:
        attrs = {"complete": True}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key): return data[key]
    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda *args: File()))
    calls = []
    native = SimpleNamespace(validate_depth_h5=lambda path, **kwargs: (calls.append(kwargs) or
        {"validation_mode": "exhaustive", "frame_counts": {gate.CAMERA: 3}, "frame_shapes": {gate.CAMERA: (2, 3)}}),
        decode_depth_png_uint16=lambda payload: np.frombuffer(np.asarray(payload, np.uint8).tobytes(), dtype=np.uint16).reshape(2, 3))
    elapsed, receipt = gate.validate_and_receipt(native, tmp_path/"unused", records)
    assert elapsed >= 0 and receipt["frame_names"] == ["000000", "000001", "000002"]
    assert calls[0]["validation_workers"] == 8 and "validate_payloads" not in calls[0]
    assert len(receipt["encoded_payloads"]["raw"]) == 3 and receipt["alignment_metadata"]["valid_count"]["dtype"] == "int32"
    data[f"alignment/{gate.CAMERA}/shift"] = Dataset(np.ones(3, np.float32))
    with pytest.raises(ValueError, match="metadata changed"): gate.validate_and_receipt(native, tmp_path/"unused", records)


def test_main_failure_freezes_exclusive_report_no_job(gate, tmp_path, monkeypatch):
    out = tmp_path/"results/depth-batch-v1"; out.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else original(p))
    with pytest.raises(ValueError, match="source SHA"): gate.main([])
    p = out/"report.json"; frozen = p.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["adoption_performed"] is False and report["budget_seconds"] == 240
    with pytest.raises(FileExistsError): gate.main([])
    assert p.read_bytes() == frozen


def test_wrapper_CPUonly_no_assets_data_validation_skip(gate):
    wrapper = Path(gate.__file__).with_name("run_depth_batch_gate.sh").read_text(); source = Path(gate.__file__).read_text()
    assert "--gpus" not in wrapper and "243s docker run" in wrapper and "--network none" in wrapper
    assert "--memory 16g --cpus 4" in wrapper and "--entrypoint python" in wrapper and "chown -R" not in wrapper
    assert not any(f"src=$ROOT/{x}" in wrapper for x in ("data", "outputs", "weights", "validation", "results,dst"))
    assert "validate_payloads=False" not in source and "validate_depth_h5(" in source and "validation_workers=WORKERS" in source
    assert "DepthFrameRecord" in source and "writer.write_frames" in source and "TemporaryDirectory" in source
    with pytest.raises(SystemExit): gate.main(["--frames", "2"])


def test_original_generator_byte_frozen_on_tiny_reference(gate):
    expected = ["152feca0bc8c61f192826affa62059c706f7e271eb923628897a56cf25cf24f7",
        "4c9f1a24f1a2ea3eab1dcfef389c47a6ce7fc892808b625fecfc8efe321b6621",
        "15e1654958cc9f17d3d3da4a4c982ed19b29e4aaa8f9d6b274c0a9d820a72c3f",
        "2c3b2902ee7ff7061dc01b294e1efbe537b78f3816a6408ed8f40784d9a8481d"]
    assert [hashlib.sha256(d.tobytes()).hexdigest() for d in gate.procedural_depths(32, 24, 4)] == expected


def test_compact_depths_new_seed_and_true_zero_background_valid_counts(gate):
    depths = gate.procedural_depths(width=32, height=24, frames=8, seed=1711, invalid_background=True)
    repeat = gate.procedural_depths(width=32, height=24, frames=8, seed=1711, invalid_background=True)
    original = gate.procedural_depths(width=32, height=24, frames=8)
    records = gate.records_for(SimpleNamespace(DepthFrameRecord=Record), depths, count_positive=True)
    for depth, again, old, record in zip(depths, repeat, original, records):
        assert depth.dtype == np.float32 and depth.shape == (24, 32) and np.min(depth) == 0.
        assert np.isfinite(depth).all() and np.max(depth) < 65.535 and np.any(depth > 0)
        assert np.array_equal(depth, again) and not np.array_equal(depth, old)
        assert record.valid_count == np.count_nonzero(depth > 0) < depth.size
        assert np.array_equal(record.aligned_depth_m, depth*np.float32(.83))
        assert (record.aligned_depth_m[depth == 0] == 0).all() and record.shift == 0.


def test_compact_eight_fullbatch_AB_BA_exhaustive_and_parity_unchanged(gate, tmp_path, monkeypatch):
    native, calls, report = stub_gate(gate, monkeypatch, tmp_path)
    depths = [np.array([[0., 2., 2.], [0., 2., 0.]], np.float32) for _ in range(8)]
    # Explicit injected inputs must never use the original 16-frame generator.
    def forbidden(): raise AssertionError("Original generator must not run")
    monkeypatch.setattr(gate, "procedural_depths", forbidden)
    gate.run_gate(native, tmp_path, report, tmp_path/"report.json", depths=depths, count_positive=True)
    assert [(m, n) for _, m, n in calls] == [("single", 2), ("batch8", 2), ("single", 8), ("batch8", 8), ("batch8", 8), ("single", 8)]
    assert report["valid_count_per_frame"] == [3]*8 and report["invalid_zero_count_per_frame"] == [3]*8
    assert report["median_write_speedup"] == 2. and report["payload_byte_parity"] is True and report["temporary_H5_removed"] is True
    assert [t["order"] for t in report["trials"]] == [["single", "batch8"], ["batch8", "single"]]


def test_compact_does_not_relax_speed_or_payload_failure(gate, tmp_path, monkeypatch):
    native, _, report = stub_gate(gate, monkeypatch, tmp_path, speedup=1.24)
    depths = [np.array([[0., 2.]], np.float32) for _ in range(8)]
    with pytest.raises(ValueError, match="below1.25"):
        gate.run_gate(native, tmp_path, report, tmp_path/"report.json", depths=depths, count_positive=True)
    assert report["median_write_speedup"] == 1.24 and report["payload_byte_parity"] is True


def test_compact_main_fail_preserves_old_failure_and_scoped_report(gate, tmp_path, monkeypatch):
    results = tmp_path/"results"; original = results/"depth-batch-v1"; original.mkdir(parents=True)
    old = original/"report.json"; old.write_bytes(b"immutable original240s failed")
    out = results/"depth-batch-compact-v1"; out.mkdir()
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); iterdir = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else iterdir(p))
    with pytest.raises(ValueError, match="source SHA"): gate.main(["--protocol", "compact"])
    report = json.loads((out/"report.json").read_text())
    assert report["stage"] == gate.COMPACT_STAGE and report["protocol"] == "compact"
    assert report["frames_per_trial"] == 8 and report["procedural_seed"] == 1711 and report["budget_seconds"] == 240
    assert report["procedural_invalid_zero_background"] is True and report["adoption_performed"] is False
    assert report["timed_partial_tail_batch_tested"] is False and report["whole_preparation_throughput_verified"] is False
    assert old.read_bytes() == b"immutable original240s failed"
    with pytest.raises(FileExistsError): gate.main(["--protocol", "compact"])


def test_compact_main_explicit_parameters_no_global_mutation(gate, tmp_path, monkeypatch):
    out = tmp_path/"results/depth-batch-compact-v1"; out.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); iterdir = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else iterdir(p))
    marker = [np.zeros((2, 3), np.float32) for _ in range(8)]; calls = []
    def generator(**kwargs): calls.append(kwargs); return marker
    def run(native, output, report, path, **kwargs):
        assert kwargs == {"depths": marker, "count_positive": True}
        report.update(status="pass", median_write_speedup=2.)
    monkeypatch.setattr(gate, "load_native", lambda root: None); monkeypatch.setattr(gate, "procedural_depths", generator)
    monkeypatch.setattr(gate, "run_gate", run); gate.main(["--protocol", "compact"])
    assert calls == [{"width": 1536, "height": 1152, "frames": 8, "seed": 1711, "invalid_background": True}]
    assert gate.FRAMES == 16 and gate.SCALE == .83 and gate.BUDGET == 240


def test_compact_wrapper_fixed_scope_and_only_two_protocols(gate):
    wrapper = Path(gate.__file__).with_name("run_depth_batch_compact_gate.sh").read_text()
    assert 'OUT="$ROOT/results/depth-batch-compact-v1"' in wrapper and '--protocol compact' in wrapper
    assert "243s docker run" in wrapper and "--memory 16g --cpus 4" in wrapper and "--network none" in wrapper
    assert "--gpus" not in wrapper and "chown -R" not in wrapper
    assert not any(f"src=$ROOT/{x}" in wrapper for x in ("data", "outputs", "weights", "validation", "results,dst"))
    with pytest.raises(SystemExit): gate.main(["--protocol", "other"])
    with pytest.raises(SystemExit): gate.main(["--proto", "compact"])

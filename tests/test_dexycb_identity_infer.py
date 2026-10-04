"""Tiny original-grid synthetic JPEGs and stub native callbacks, never models."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from world_reward.automatic_candidate_bank import build_automatic_candidate_bank

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "infra"), str(REPO / "src")]
spec = importlib.util.spec_from_file_location("wr_dex_identity_test", REPO / "infra/dexycb_identity_infer.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def sealed(path, raw):
    path.write_bytes(raw); path.chmod(0o444)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def cohort(tmp_path, frames=3):
    directory = tmp_path / "inputs"; directory.mkdir()
    buffer = io.BytesIO(); Image.new("RGB", (640, 480), (30, 40, 50)).save(buffer, format="JPEG")
    raw = buffer.getvalue(); sequences, rows = [], []
    for clip in range(12):
        index = driver.INDICES[clip % 6]
        seq = dict(subject=driver.SUBJECTS[clip // 6], sequence=f"20200813_{clip:06d}", sequence_lex_index=index,
            camera=driver.CAMERA, frames=frames, partition="blind_evaluation" if clip < 6 else ("fit" if index in (0, 32, 64) else "decision"))
        sequences.append(seq)
        for frame in range(frames):
            name = f"subject_{clip // 6 + 1:02d}_sequence_{index:03d}_frame_{frame:06d}.jpg"
            pin = sealed(directory / name, raw)
            rows.append(dict(file=name, **pin, width=640, height=480, subject=seq["subject"], sequence=seq["sequence"],
                camera=driver.CAMERA, frame_position=frame, source_frame_id=frame))
    manifest = dict(schema="world-reward-dexycb-identity-rgb-v1", license="CC-BY-NC-4.0", sequences=sequences, images=rows,
        timestamps_available=False, frame_count_evidence="all_RGB_and_label_member_headers_no_meta_values",
        source_archives={s: dict(bytes=100, sha256="a" * 64) for s in driver.SUBJECTS},
        training_overlap_verified=False, challenge_overlap_verified=False)
    pin = sealed(directory / "manifest.json", json.dumps(manifest).encode())
    return directory, pin, manifest


class Predictor:
    def __init__(self, empty=()):
        self.images, self.calls, self.empty = [], [], set(empty)

    def set_image(self, rgb): self.images.append(rgb)

    def predict(self, *, box, multimask_output):
        self.calls.append((box.copy(), multimask_output))
        n = len(box); masks = np.zeros((n, 1, 480, 640), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(box.astype(int)):
            if i not in self.empty: masks[i, 0, y0:y1, x0:x1] = 1
        score = np.arange(n, dtype=np.float32) - .2
        return (masks[0], score.reshape(n, 1)[0], None) if n == 1 else (masks, score[:, None], None)


def detector(rgb, query):
    assert rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3) and not rgb.flags.writeable
    boxes = ((10, 10, 110, 110), (210, 10, 310, 110)) if query == "hand." else ((120, 140, 220, 240), (330, 140, 430, 240))
    return np.array(boxes, np.float32), np.full(len(boxes), .8, np.float32)


def bank(frames=3, empty=()):
    built = build_automatic_candidate_bank(np.zeros((480, 640, 3), np.uint8), detector, detector, Predictor(empty))
    arrays, candidates = driver.bank_arrays(built, frames)
    row = dict(candidates=candidates, background_query_count=len(built.background_query_points))
    return arrays, row


def native(data, *, motion=True, invisible=False):
    queries = data["query_points"]; n, t = len(queries), len(data["video"])
    static = np.broadcast_to(queries[:, [2, 1]][:, None], (n, t, 2)).copy()
    tracks = static.astype(np.float32)
    if motion: tracks[:, :, 0] += np.arange(t, dtype=np.float32)[None, :] * 5
    # Match the native multiplication/division order exactly, including rounding.
    tracks256 = tracks * np.array([256 / 640, 256 / 480], np.float32)
    tracks = (tracks256 * np.array([640, 480], np.float32)) / np.array([256, 256], np.float32)
    visible = np.full((n, t), not invisible, bool)
    return dict(tracks=tracks, tracks_256=tracks256, visible=visible, occlusion=np.full((n, t), -2, np.float32),
        expected_dist=np.full((n, t), -2, np.float32), query_points=queries.copy(), point_indices=data["point_indices"].copy(),
        frame_index=np.arange(t, dtype=np.int64), static_tracks=static, static_visible=np.ones((n, t), bool))


def test_full_public_reader_uses_only_public_rgb(tmp_path, monkeypatch):
    directory, pin, _ = cohort(tmp_path)
    original = Path.open
    def public_only(path, *args, **kwargs):
        assert path.parent == directory
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_only)
    clips, frozen, manifest = driver.public_inputs(directory, pin)
    assert len(clips) == 12 and len(frozen) == 37 and manifest["timestamps_available"] is False
    assert [c["sequence_lex_index"] for c in clips] == list(driver.INDICES) * 2
    assert all([r["source_frame_id"] for r in c["records"]] == [0, 1, 2] for c in clips)
    assert np.array_equal(driver.read_rgb(clips[0]["records"][0]), np.array(Image.open(clips[0]["records"][0]["path"])))


@pytest.mark.parametrize("fault", ["timestamp", "private", "frame", "sequence", "partition", "boolframe", "extra", "bytechange"])
def test_invalid_or_changed_public_inputs_fail_closed(tmp_path, fault):
    directory, pin, manifest = cohort(tmp_path)
    if fault == "timestamp": manifest["timestamps_available"] = True
    if fault == "private": manifest["images"][0]["private_label"] = "labels.npz"
    if fault == "frame": manifest["images"][0]["source_frame_id"] = 1
    if fault == "sequence": manifest["sequences"][0]["sequence_lex_index"] = 1
    if fault == "partition": manifest["sequences"][6]["partition"] = "blind_evaluation"
    if fault == "boolframe": manifest["images"][0]["frame_position"] = False
    if fault == "extra": sealed(directory / "meta.yml", b"no private values")
    if fault == "bytechange":
        p = directory / manifest["images"][0]["file"]; p.chmod(0o600); p.write_bytes(b"changed"); p.chmod(0o444)
    p = directory / "manifest.json"; p.chmod(0o600); pin = sealed(p, json.dumps(manifest).encode())
    with pytest.raises(ValueError): driver.public_inputs(directory, pin)


def test_all_twelve_banks_one_encoder_each_and_all_candidates(tmp_path):
    directory, pin, _ = cohort(tmp_path); clips = driver.public_inputs(directory, pin)[0]
    out = tmp_path / "banks"; out.mkdir(); report = dict(clips=[]); predictor = Predictor(empty=(2,))
    calls = []
    def detect(rgb, query): calls.append(query); return detector(rgb, query)
    driver.observe_masks(clips, out, report, lambda: None, detect=detect, predictor=predictor)
    assert calls == ["hand.", "object."] * 12
    assert len(predictor.images) == len(predictor.calls) == len(report["clips"]) == 12
    assert all(boxes.shape == (4, 4) and multimask is False for boxes, multimask in predictor.calls)
    for row in report["clips"]:
        assert [c["kind"] for c in row["candidates"]] == ["hand", "hand", "object", "object"]
        assert row["candidates"][2]["query_count"] == row["candidates"][2]["mask_pixels"] == 0
        with np.load(out / row["file"], allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}; driver.validate_bank(arrays, row, 3)
            assert arrays["query_offsets"][3] == arrays["query_offsets"][2]
            assert arrays["raw_hand_boxes"].dtype == np.float32
        assert (out / row["file"]).stat().st_mode & 0o777 == 0o444


def test_missing_class_preserves_failed_diagnostic_and_aborts_no_clip_mining(tmp_path):
    directory, pin, _ = cohort(tmp_path); clips = driver.public_inputs(directory, pin)[0]
    out = tmp_path / "banks"; out.mkdir(); report = dict(clips=[]); calls = []
    def detect(rgb, query):
        calls.append(query)
        return (np.empty((0, 4)), np.empty(0)) if query == "hand." else detector(rgb, query)
    with pytest.raises(ValueError, match="coverage failed"):
        driver.observe_masks(clips, out, report, lambda: None, detect=detect, predictor=Predictor())
    assert calls == ["hand.", "object."] and len(report["clips"]) == 1
    assert {p.name for p in out.iterdir()} == {"clip_000.npz"}
    assert [c["kind"] for c in report["clips"][0]["candidates"]] == ["object", "object"]


@pytest.mark.parametrize("fault", ["dropped", "timeline", "floatoffset", "queryoffset", "mask", "background", "raw", "querycount"])
def test_serialized_bank_whole_geometry_and_query_contract(tmp_path, fault):
    arrays, row = bank(); arrays = {k: v.copy() for k, v in arrays.items()}; row = copy.deepcopy(row)
    if fault == "dropped": arrays.pop("raw_hand_scores")
    if fault == "timeline": arrays["frame_index"][-1] = 1
    if fault == "floatoffset": arrays["query_offsets"] = arrays["query_offsets"].astype(float)
    if fault == "queryoffset": arrays["query_points"][0, 1:] -= .5
    if fault == "mask": arrays["initial_masks"][0] = False
    if fault == "background": arrays["query_points"][-1] = arrays["query_points"][0]
    if fault == "raw": arrays["raw_hand_scores"][0] = np.nan
    if fault == "querycount": row["candidates"][0]["query_count"] -= 1
    with pytest.raises(ValueError): driver.validate_bank(arrays, row, 3)


def test_one_full_native_call_all_queries_q0_instances_and_no_static_controls():
    arrays, row = bank(frames=5, empty=(0, 2)); calls = []
    video = np.zeros((5, 480, 640, 3), np.uint8)
    def predict(data): calls.append(data); return native(data)
    result = driver.track_clip(video, arrays, row, predict)
    assert len(calls) == 1 and calls[0]["video"] is video
    np.testing.assert_array_equal(calls[0]["query_points"], arrays["query_points"])
    np.testing.assert_array_equal(result["frame_index"], np.arange(5, dtype=np.int64))
    assert "static_tracks" not in result and "static_visible" not in result
    assert result["motion_features"].shape == (4, 2, 2, 3)
    assert not result["motion_hand_supported"][:, 0].any() and not result["motion_object_supported"][:, 0].any()
    assert not result["motion_pair_supported"][:, 0].any()
    assert result["motion_camera_supported"].all()
    np.testing.assert_array_equal(result["motion_time_intervals"], np.ones(4))
    np.testing.assert_allclose(result["motion_features"][:, 1, 1], 0, atol=2e-14)


def test_background_missing_never_becomes_confident_zero_motion():
    arrays, row = bank(); video = np.zeros((3, 480, 640, 3), np.uint8)
    result = driver.track_clip(video, arrays, row, lambda d: native(d, invisible=True))
    assert not result["motion_camera_supported"].any() and not result["motion_pair_supported"].any()
    assert np.isnan(result["motion_features"]).all()
    assert result["tracks"].shape[1] == 3  # Full coordinates retained even when invisible.


@pytest.mark.parametrize("fault", ["nativeframes", "nativequery", "nativefinite", "static", "nativecoords"])
def test_native_full_t_source_math_validation_not_bypassed(fault):
    arrays, row = bank(); video = np.zeros((3, 480, 640, 3), np.uint8)
    def predict(data):
        result = native(data)
        if fault == "nativeframes": result["frame_index"][-1] = 1
        if fault == "nativequery": result["query_points"][0, 2] += 1
        if fault == "nativefinite": result["tracks"][0, 1, 0] = np.nan
        if fault == "static": result["static_tracks"][0, 0, 0] += 1
        if fault == "nativecoords": result["tracks"][0, 0, 0] += 1
        return result
    with pytest.raises(ValueError): driver.track_clip(video, arrays, row, predict)


def test_full_twelve_tracking_calls_and_sealed_raw_features(tmp_path):
    directory, pin, _ = cohort(tmp_path); clips = driver.public_inputs(directory, pin)[0]
    banks = [bank() for _ in clips]; out = tmp_path / "tracks"; out.mkdir(); report = dict(clips=[]); calls = []
    def predict(data): calls.append((len(data["video"]), len(data["query_points"]))); return native(data)
    driver.observe_tracks(clips, banks, out, report, lambda: None, predict=predict)
    assert len(calls) == len(report["clips"]) == 12 and all(t == 3 and q > 32 for t, q in calls)
    with np.load(out / "clip_011.npz", allow_pickle=False) as saved:
        assert all(k in saved for k in ("tracks", "visible", "occlusion", "expected_dist", "motion_features", "motion_pair_supported"))
        assert not any("static" in k for k in saved.files)
        assert saved["occlusion"].dtype == saved["expected_dist"].dtype == np.float32


def test_saved_outputs_refuse_overwrite_or_pickle(tmp_path):
    path = tmp_path / "arrays.npz"
    driver.save_arrays(path, {"q0": np.empty((0, 3), np.float64)})
    with pytest.raises(FileExistsError): driver.save_arrays(path, {"q0": np.empty((0, 3))})
    with pytest.raises(ValueError): driver.save_arrays(tmp_path / "bad.npz", {"unsafe": np.array([{}], object)})
    assert not (tmp_path / "bad.npz").exists()


def test_stage_cli_exact_pins_and_no_budget_retuning():
    args = ["--stage", "masks", "--manifest-sha256", "a" * 64, "--manifest-bytes", "100", "--acquisition-report-sha256", "b" * 64,
        "--acquisition-report-bytes", "200"]
    assert driver.parser().parse_args(args).stage == "masks"
    assert driver.BUDGETS == {"masks": 300, "tracks": 7200} and driver.IMAGES["masks"] != driver.IMAGES["tracks"]
    with pytest.raises(SystemExit): driver.parser().parse_args(args + ["--budget", "10000"])


def test_host_imports_no_numpy_torch_or_pillow_under_isolated_python():
    script = """import importlib.util,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'infra'),str(root/'src')]
spec=importlib.util.spec_from_file_location('pure_host',root/'infra/dexycb_identity_infer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert not any(name in sys.modules for name in ('numpy','torch','PIL','transformers','sam2'))
print('stdlib_only')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(REPO)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "stdlib_only"


def test_host_banks_hashes_without_decoding_or_scientific_dependencies(tmp_path, monkeypatch):
    directory = tmp_path / "masks"; directory.mkdir(); sealed(directory / ".container.cid", b"a" * 64)
    clips, rows = [], []
    for i in range(12):
        name = f"clip_{i:03d}.npz"; pin = sealed(directory / name, b"not decoded by host")
        clips.append(dict(clip_index=i, frames=3)); rows.append(dict(clip_index=i, frames=3, file=name, **pin))
    source = dict(closure="manufactured")
    manifest_pin = dict(bytes=12, sha256="b" * 64)
    receipt = dict(schema="world-reward-dexycb-identity-infer-v1", stage="masks", status="pass", phase="complete",
        producer_revision="c" * 40, image_id=driver.binding.IMAGE, public_manifest=manifest_pin, original_rehashed_after=True,
        private_annotations_read=False, challenge_inputs_used=False, oracle_modes=[], quality_verified=False,
        detector_calls=24, sam2_encoder_calls=12, sam2_batch_calls=12, budget_seconds=300, clips=rows,
        source_binding=source, script_sha256="d" * 64)
    pin = sealed(directory / "report.json", json.dumps(receipt).encode())
    original = driver.binding.identity
    monkeypatch.setattr(driver.binding, "identity", lambda p, *a: dict(bytes=2, sha256="d" * 64)
                        if Path(p) == Path(driver.__file__) else original(p, *a))
    monkeypatch.setattr(driver, "source_binding", lambda *a: source)
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No host NP decode")))
    banks, frozen = driver.load_banks(directory, pin, clips, "c" * 40, manifest_pin, decode=False)
    assert banks == [] and len(frozen) == 13
    p = directory / rows[0]["file"]; p.chmod(0o600); p.write_bytes(b"changed"); p.chmod(0o444)
    with pytest.raises(ValueError, match="bank/clip bytes"): driver.load_banks(directory, pin, clips, "c" * 40, manifest_pin, decode=False)


def test_acquisition_real_source_map_accepts_empty_init_and_never_opens_labels(tmp_path, monkeypatch):
    root = tmp_path / "root"; code = root / "newcode"; code.mkdir(parents=True)
    revision = "e" * 40; original = root / "jobs" / revision / "run_dexycb_acquire/code"; original.mkdir(parents=True)
    script_pin = sealed(original / "infra.py", b"source")
    (original / "infra").mkdir(); sealed(original / "infra/dexycb_acquire.py", b"source")
    (original / "configs").mkdir(); (code / "configs").mkdir()
    protocol_pin = sealed(original / "configs/dexycb_identity_protocol.json", b"protocol")
    sealed(code / "configs/dexycb_identity_protocol.json", b"protocol")
    empty_pin = sealed(original / "empty_init.py", b"")
    marker_pin = sealed(original.parent / "revision", (revision + "\n").encode())
    archive = {s: dict(bytes=100, sha256="a" * 64) for s in driver.SUBJECTS}
    manifest_pin = dict(bytes=100, sha256="a" * 64)
    image = dict(file="frame.jpg", bytes=20, sha256="b" * 64)
    sources = {str(original / "infra/dexycb_acquire.py"): script_pin,
        str(original / "configs/dexycb_identity_protocol.json"): protocol_pin,
        str(original / "empty_init.py"): empty_pin, str(original.parent / "revision"): marker_pin}
    receipt = dict(stage="external_dexycb_identity_rgb_private_byte_acquisition", status="pass", phase="complete",
        sequences=12, annotation_values_parsed=False, private_values_interpreted=False, inference_performed=False,
        gpu_used=False, source_rehashed_after=True, public_manifest=manifest_pin, archive_proofs=archive,
        source_before=sources, source_after=sources, producer_revision=revision, script_sha256=script_pin["sha256"],
        frames=1, retained_files={"frame.jpg": {k: image[k] for k in ("bytes", "sha256")}, "private.npz": dict(bytes=4, sha256="c" * 64)})
    folder = root / driver.BASE; folder.mkdir(parents=True); pin = sealed(folder / "report.json", json.dumps(receipt).encode())
    monkeypatch.setattr(driver, "ROOT", root)
    def marker_read(path, maximum, readonly):
        raw, got = driver.binding.selected.read(path, maximum, readonly=readonly)
        return raw, {k: got[k] for k in ("bytes", "sha256")}
    monkeypatch.setattr(driver.binding, "kernel_helper", lambda: SimpleNamespace(read=marker_read))
    manifest = dict(source_archives=archive, images=[image])
    assert driver.acquisition_proof(pin, manifest_pin, manifest, code) == pin
    # A private path injected into purported source evidence fails BEFORE opening it.
    receipt["source_before"][str(folder / "eval_private/labels.npz")] = dict(bytes=4, sha256="f" * 64)
    path = folder / "report.json"; path.chmod(0o600); pin = sealed(path, json.dumps(receipt).encode())
    with pytest.raises(ValueError, match="never private labels"): driver.acquisition_proof(pin, manifest_pin, manifest, code)


def test_readonly_mount_whitelist_excludes_private_and_old_oracle_data(tmp_path, monkeypatch):
    code = tmp_path / "code"
    # canonical does not require existence; only byte authentication does, in host-proof first.
    masks = driver.host_mounts("masks", code, "a" * 40)
    assert code in masks and driver.ROOT / driver.BASE / "inputs" in masks
    assert driver.binding.BUILD_CODE in masks and driver.binding.KERNEL_CODE in masks
    assert all(not any(s in str(p) for s in ("eval_private", "public_v2", "robotap.zip")) for p in masks)
    assert driver.ROOT not in masks and driver.binding.DEST not in masks
    native_files = {"tapnet/__init__.py": {}, "tapnet/torch/tapir_model.py": {}}
    monkeypatch.setattr(driver.binding, "pinned", lambda *a: dict(source_helpers={"infra/run_bootstapir_runtime_verify.sh": {}}))
    monkeypatch.setattr(driver.boots.source, "read_protocol", lambda *a: dict(source=dict(files=native_files), checkpoint=dict(file="checkpoint.pt")))
    tracks = driver.host_mounts("tracks", code, "a" * 40)
    assert driver.output_path("masks", "a" * 40) in tracks
    assert all(not any(s in str(p) for s in ("eval_private", "public_v2", "data/")) for p in tracks)
    assert driver.ROOT / driver.BASE / "report.json" not in tracks
    assert driver.ROOT / driver.boots.BASE / "assets/tapnet_source/tapnet/torch/tapir_model.py" in tracks


@pytest.mark.parametrize("fault", ["source", "kernel", "image", "operator"])
def test_actual_frontend_ancestry_rejected_before_models(monkeypatch, fault):
    cfg = dict(bytes=1, sha256="a" * 64)
    old = dict(closure_sha256="b" * 64, helpers={"infra/frontend_grounding_build.py": dict(sha256=driver.binding.BUILD_SHA), driver.binding.CONFIG: cfg})
    gate = dict(helpers={"infra/frontend_sam2_kernel_gate.py": driver.binding.KERNEL_SOURCE_PIN, driver.binding.CONFIG: cfg})
    child = dict(Id=driver.binding.IMAGE, Architecture="amd64", Os="linux", RootFS=dict(Layers=["layer"] * 44))
    parent = dict(Id="base", RootFS=dict(Layers=["layer"] * 44))
    build = dict(schema="world_reward.frontend_grounding_build.v6", stage="frontend_grounding_build", status="pass", phase="complete",
        producer_revision=driver.binding.BUILD_REV, offline_build_exit_code=0, child_probe_exit_code=0,
        parent_unchanged_verified=True, source_rechecked_before_and_after=True, extension_import_verified=True,
        CUDA_execution_verified=False, replica_ready=False, license_eligibility_verified=False, training_overlap_verified=False,
        source_binding=old, child_image=child, parent_image=parent,
        owner=hashlib.sha256((driver.binding.BUILD_REV + old["closure_sha256"]).encode()).hexdigest())
    receipt = dict(schema="world_reward.frontend_sam2_kernel_gate.v1", stage="frontend_sam2_kernel_gate", status="pass",
        producer_revision=driver.binding.KERNEL_REV, script_sha256=driver.binding.KERNEL_SOURCE_PIN["sha256"], image_id=driver.binding.IMAGE,
        models_loaded=False, challenge_data_read=False, CUDA_operator_execution_verified=True, build_report_identity=driver.binding.BUILD_PIN,
        replica_ready=False, operator_source_identities=dict(extension=driver.binding.EXTENSION_PIN), source_binding=gate)
    if fault == "source": build["source_binding"] = {}
    if fault == "kernel": receipt["status"] = "fail"
    if fault == "image": child["Id"] = "different"
    if fault == "operator": receipt["operator_source_identities"] = {}
    kernel = SimpleNamespace(BASE="base", BUILD_HELPERS=(), closure=lambda code, *a: old if code == driver.binding.BUILD_CODE else gate)
    monkeypatch.setattr(driver.binding, "kernel_helper", lambda: kernel)
    monkeypatch.setattr(driver.binding, "pinned", lambda p, *a: build if p == driver.binding.BUILD_REPORT else receipt)
    monkeypatch.setattr(driver.binding, "identity", lambda *a: cfg)
    monkeypatch.setattr(driver.binding.selected, "load_contract", lambda *a: (_ for _ in ()).throw(AssertionError("No models after invalid ancestry")))
    with pytest.raises(ValueError): driver.frontend_proof(Path("/new/code"))


def test_no_oracle_adapter_contact_or_personcrop_paths():
    source = (REPO / "infra/dexycb_identity_infer.py").read_text(); tree = ast.parse(source)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "boots.native_prediction" in calls and "boots.validate_prediction" in calls
    assert not any(c in calls for c in ("boots.bindings", "boots.public_records", "boots.validate_video", "binding.authenticate"))
    assert all(x not in source for x in ("ycbv_point_depth", "DWPose", "robotap_boots_public", "ycb_grasp_ind", "np.interp"))
    assert "original_frame_index_not_seconds" in source and "identity_accepted=False" in source
    builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "build_sam2"]
    assert len(builds) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in builds[0].keywords}["apply_postprocessing"] is True
    assert source.startswith('"""Azure-only GroundingDINO/')

"""Data-free native-feed/133 replay contracts, not an actual ORT inference claim."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

INFRA = Path(__file__).parents[1] / "infra"
sys.path.insert(0, str(INFRA))
spec = importlib.util.spec_from_file_location("dwpose_smoke", INFRA / "dwpose_smoke.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def metadata_session(*, provider=None, input_shape=None, outputs=None):
    inputs = [SimpleNamespace(name="image", shape=input_shape or [1, 3, 384, 288], type="tensor(float)")]
    out = outputs or [SimpleNamespace(name="simcc_x", shape=[1, 133, 576], type="tensor(float)"),
                      SimpleNamespace(name="simcc_y", shape=[1, 133, 768], type="tensor(float)")]
    return SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: out, get_providers=lambda: provider or ["CPUExecutionProvider"])


def test_graph_metadata_actual_names_cpu_only_symbolic_batch():
    for batch in (1, "batch", None):
        session = metadata_session(input_shape=[batch, 3, 384, 288])
        result = gate.validate_session(session)
        assert result["input"]["name"] == "image" and result["outputs"][0]["shape"][-1] == 576


@pytest.mark.parametrize("kwargs", [{"provider": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
    {"input_shape": [2, 3, 384, 288]}, {"input_shape": [True, 3, 384, 288]}, {"input_shape": [1, 3, 288, 384]},
    {"outputs": [SimpleNamespace(name="simcc_x", shape=[1, 134, 576], type="tensor(float)"),
                 SimpleNamespace(name="simcc_y", shape=[1, 134, 768], type="tensor(float)")]}])
def test_graph_metadata_failclosed(kwargs):
    with pytest.raises(ValueError): gate.validate_session(metadata_session(**kwargs))


def test_no_134_or_extra_output_names_or_nonfloat_graph():
    session = metadata_session(); session.get_inputs()[0].type = "tensor(double)"
    with pytest.raises(ValueError): gate.validate_session(session)
    session = metadata_session(); session.get_outputs()[1].name = "simcc_x"
    with pytest.raises(ValueError): gate.validate_session(session)


def test_automatic_bbox_exclusive_edges_full_original_grid(monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 6)
    mask = np.zeros((6, 8), np.uint8); mask[1:4, 2:7] = 255; frozen = mask.copy()
    assert np.array_equal(gate.actor_bbox(mask), np.array([[2, 1, 7, 4]], np.float32))
    assert np.array_equal(mask, frozen)


@pytest.mark.parametrize("kind", ["empty", "dtype", "shape", "gray", "masked"])
def test_mask_not_fabricated_or_empty_fallback(monkeypatch, kind):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 6)
    mask = np.zeros((6, 8), np.uint8); mask[0, 0] = 255
    if kind == "empty": mask[:] = 0
    elif kind == "dtype": mask = mask.astype(bool)
    elif kind == "shape": mask = mask[:5]
    elif kind == "gray": mask[1, 1] = 1
    elif kind == "masked": mask = np.ma.array(mask, mask=np.zeros_like(mask, bool))
    with pytest.raises(ValueError): gate.actor_bbox(mask)


def test_pip_only_exact_two_pinned_wheels_private_noindex_nodeps(tmp_path):
    argv = gate.pip_argv(tmp_path / "prefix", tmp_path / "root")
    assert argv[:4] == [sys.executable, "-I", "-m", "pip"]
    assert "--isolated" in argv and "--no-index" in argv and "--no-deps" in argv and "--no-compile" in argv
    assert argv[argv.index("--target") + 1] == str(tmp_path / "prefix")
    assert len([x for x in argv if x.endswith(".whl")]) == 2
    assert argv[-2].endswith("onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl")
    assert argv[-1].endswith("flatbuffers-25.12.19-py2.py3-none-any.whl")
    assert not any(x.startswith("http") or x in ("--upgrade", "--user", "--extra-index-url") for x in argv)


def test_proxy_delegates_same_native_float64_list_objects_no_cast():
    session = metadata_session(); seen = []; records = []; persisted = []
    outputs = [np.ones((1, 133, 576), np.float32), np.ones((1, 133, 768), np.float32)]
    session.run = lambda names, feed: (seen.append((names, feed)), outputs)[1]
    value = [np.arange(3 * 384 * 288, dtype=np.float64).reshape(3, 384, 288)]
    feed = {"image": value}; proxy = gate.SessionProxy(session, records, lambda: persisted.append(True))
    assert proxy.run(["simcc_x", "simcc_y"], feed) is outputs
    assert seen[0][1] is feed and seen[0][1]["image"] is value and seen[0][1]["image"][0] is value[0]
    assert records[0]["supplied_array"]["dtype"] == "float64"
    assert records[0]["delegated_unmodified"] is True and records[0]["effective_runtime_conversion_observed"] is False
    assert records[0]["run_completed"] is True and len(persisted) == 2


@pytest.mark.parametrize("kind", ["f32", "ndarray", "shape", "nan", "names", "extra", "outputs"])
def test_proxy_bad_feed_does_not_soft_retry(kind):
    session = metadata_session(); calls = []
    session.run = lambda *_: calls.append(1) or [np.zeros((1, 133, 576), np.float32), np.zeros((1, 133, 768), np.float32)]
    names = ["simcc_x", "simcc_y"]; feed = {"image": [np.zeros((3, 384, 288), np.float64)]}
    if kind == "f32": feed["image"][0] = feed["image"][0].astype(np.float32)
    elif kind == "ndarray": feed["image"] = np.array(feed["image"])
    elif kind == "shape": feed["image"][0] = np.zeros((3, 288, 384), np.float64)
    elif kind == "nan": feed["image"][0][0, 0, 0] = np.nan
    elif kind == "names": names.reverse()
    elif kind == "extra": feed["extra"] = feed["image"]
    elif kind == "outputs": session.run = lambda *_: [np.zeros((1, 134, 576), np.float32), np.zeros((1, 134, 768), np.float32)]
    with pytest.raises(ValueError): gate.SessionProxy(session, [], lambda: None).run(names, feed)
    if kind != "outputs": assert calls == []


def test_native_feed_backend_failure_preserves_attempt_no_cast_or_retry():
    session = metadata_session(); count = []
    def run(*_): count.append(1); raise RuntimeError("native backend cannot consume float64 list")
    session.run = run; records = []
    with pytest.raises(RuntimeError, match="cannot consume"):
        gate.SessionProxy(session, records, lambda: None).run(["simcc_x", "simcc_y"], {"image": [np.zeros((3, 384, 288), np.float64)]})
    assert count == [1] and len(records) == 1 and records[0]["run_completed"] is False
    assert records[0]["supplied_array"]["dtype"] == "float64"


def test_native_pipeline_fixture_preserves_float64_normalization_and_decode():
    """Fixture executes native-style preprocessing/delegation, not actual ORT."""
    session = metadata_session(); records = []
    def backend(names, feed):
        assert isinstance(feed["image"], list) and feed["image"][0].dtype == np.float64
        x = np.zeros((1, 133, 576), np.float32); y = np.zeros((1, 133, 768), np.float32)
        x[..., 20] = 2.; y[..., 30] = 3.
        return [x, y]
    session.run = backend
    normalized = (np.full((384, 288, 3), 128, np.uint8) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    outputs = gate.SessionProxy(session, records, lambda: None).run(["simcc_x", "simcc_y"], {"image": [normalized.transpose(2, 0, 1)]})
    points = np.stack([outputs[0].argmax(-1), outputs[1].argmax(-1)], -1).astype(np.float32) / 2.
    scores = np.minimum(outputs[0].max(-1), outputs[1].max(-1))
    assert gate.validate_prediction(points, scores).all()
    assert np.array_equal(points[0, 0], [10., 15.]) and np.array_equal(scores, np.full((1, 133), 2., np.float32))


def prediction_row(file="clip_00_frame_000.png", score=0.):
    points = np.full((1, 133, 2), 42., np.float64); scores = np.full((1, 133), score, np.float32)
    valid = gate.validate_prediction(points, scores)
    return {"file": file, "keypoints": gate.array_identity(points), "scores": gate.array_identity(scores),
            "validity": gate.array_identity(valid), "raw_simcc": [{"sha256": "a" * 64}, {"sha256": "b" * 64}]}


def test_scores_not_probabilities_invalid_positions_not_sentinel_allinvalid_abi():
    points = np.full((1, 133, 2), 100., np.float64); scores = np.zeros((1, 133), np.float32)
    scores[0, :3] = [-2., 0., 8.]
    valid = gate.validate_prediction(points, scores)
    assert valid.sum() == 1 and scores[0, 2] == 8. and (points == 100).all()
    rows = [prediction_row(), prediction_row("clip_01_frame_000.png")]
    gate.validate_replay([{"predictions": copy.deepcopy(rows), "calls": [{"run_completed": True}] * 2} for _ in range(2)])


@pytest.mark.parametrize("kind", ["nan", "points134", "scores134", "scores64", "masked"])
def test_prediction_strict_133_numeric_contract(kind):
    points = np.zeros((1, 133, 2), np.float64); scores = np.zeros((1, 133), np.float32)
    if kind == "nan": points[0, 0, 0] = np.nan
    elif kind == "points134": points = np.zeros((1, 134, 2), np.float64)
    elif kind == "scores134": scores = np.zeros((1, 134), np.float32)
    elif kind == "scores64": scores = scores.astype(np.float64)
    elif kind == "masked": points = np.ma.array(points, mask=np.zeros_like(points, bool))
    with pytest.raises(ValueError): gate.validate_prediction(points, scores)


@pytest.mark.parametrize("kind", ["keypoints", "scores", "validity", "raw_simcc", "order", "callfail", "missing"])
def test_replay_byte_identity_no_average_or_selection(kind):
    rows = [prediction_row(), prediction_row("clip_01_frame_000.png")]
    sessions = [{"predictions": copy.deepcopy(rows), "calls": [{"run_completed": True}] * 2} for _ in range(2)]
    if kind in ("keypoints", "scores", "validity"): sessions[1]["predictions"][0][kind]["sha256"] = "0" * 64
    elif kind == "raw_simcc": sessions[1]["predictions"][0][kind][0]["sha256"] = "0" * 64
    elif kind == "order": sessions[1]["predictions"].reverse()
    elif kind == "callfail": sessions[1]["calls"][0] = {"run_completed": False}
    elif kind == "missing": sessions[1]["predictions"].pop()
    with pytest.raises(ValueError): gate.validate_replay(sessions)


def public_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 8)
    base = tmp_path / gate.PUBLIC; (base / "inputs").mkdir(parents=True); (base / "automatic_masks").mkdir()
    images, rows, selected = [], [], []
    for index in range(15):
        clip, frame = divmod(index, 5); name = f"clip_{clip:02d}_frame_{frame:03d}.png"
        rgb_sha = f"{index + 1:064x}"; mask_sha = f"{index + 20:064x}"
        if index in (0, 5):
            rgb = base / "inputs" / name; rgb.write_bytes(b"own-public-rgb-" + str(index).encode()); rgb_sha = gate.acquisition.digest(rgb)
            human = base / "automatic_masks" / (Path(name).stem + "_human.png"); human.write_bytes(b"own-automatic-mask-" + str(index).encode()); mask_sha = gate.acquisition.digest(human)
            selected.append((index, name, rgb_sha, mask_sha, human.stat().st_size, 7))
        images.append({"file": name, "sha256": rgb_sha, "width": 8, "height": 8})
        row = {"file": name, "clip_index": clip, "frame_index": frame, "rgb_sha256": rgb_sha}
        for label, query in (("human", "person."), ("object", "bottle.")):
            row.update({label + "_query": query, label + "_mask_file": Path(name).stem + "_" + label + ".png",
                        label + "_mask_sha256": mask_sha, label + "_mask_bytes": selected[-1][4] if index in (0, 5) else 1,
                        label + "_mask_pixels": 7})
        rows.append(row)
    manifest = {"schema": "world-reward-identity-rgb-v1", "images": images}
    mask = {"stage": "public_identity_rgb_automatic_masks", "status": "pass", "phase": "complete", "frames": 15,
        "producer_revision": gate.MASK_REVISION, "network": "none", "private_truth_read": False, "ground_truth_used": False,
        "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [], "human_query": "person.", "object_query": "bottle.",
        "actual_detector_calls": 30, "actual_sam2_calls": 30, "actual_sam2_image_encoder_calls": 15,
        "actual_automatic_inference_verified": True, "all_cases_retained": True, "input_manifest_bytes": 2199, "records": rows}
    manifest_path = base / "inputs/manifest.json"; mask_path = base / "automatic_masks/report.json"
    def write():
        manifest_path.write_text(json.dumps(manifest)); monkeypatch.setattr(gate, "MANIFEST_SHA", gate.acquisition.digest(manifest_path))
        mask["input_manifest_sha256"] = gate.MANIFEST_SHA; mask_path.write_text(json.dumps(mask)); monkeypatch.setattr(gate, "MASK_SHA", gate.acquisition.digest(mask_path))
    monkeypatch.setattr(gate, "SELECTED", tuple(selected)); write()
    # Only historical receipt byte counts are injected: payload hashes and safety guards remain real.
    original = gate.identity
    def identity(path, *, sha=None, size=None):
        return original(path, sha=sha, size=None if Path(path) in (manifest_path, mask_path) else size)
    monkeypatch.setattr(gate, "identity", identity)
    return manifest, mask, write, base


def test_real_manifest15_metadata_only_two_public_file_mounts(tmp_path, monkeypatch):
    _, _, _, base = public_fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    receipt = gate.validate_public(tmp_path)
    assert len(receipt["selected"]) == 2 and len(list((base / "inputs").glob("*.png"))) == 2
    assert before == {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["private_field", "missing", "order", "dims", "rgb_binding", "mask_name", "gt", "calls", "boolcount", "oracle", "empty"])
def test_public_metadata_before_images_fail_closed(tmp_path, monkeypatch, kind):
    manifest, mask, write, _ = public_fixture(tmp_path, monkeypatch)
    if kind == "private_field": manifest["camera_K"] = [[1]]
    elif kind == "missing": manifest["images"].pop()
    elif kind == "order": manifest["images"].reverse()
    elif kind == "dims": manifest["images"][0]["width"] = 9
    elif kind == "rgb_binding": mask["records"][0]["rgb_sha256"] = "0" * 64
    elif kind == "mask_name": mask["records"][0]["human_mask_file"] = "../../private.png"
    elif kind == "gt": mask["ground_truth_used"] = True
    elif kind == "calls": mask["actual_detector_calls"] = 29
    elif kind == "boolcount": mask["records"][0]["human_mask_pixels"] = True
    elif kind == "oracle": mask["oracle_modes"] = ["synthetic_camera"]
    elif kind == "empty": mask["records"][0]["human_mask_pixels"] = 0
    write()
    with pytest.raises(ValueError): gate.validate_public(tmp_path)


@pytest.mark.parametrize("kind", ["rgb", "mask", "manifest", "symlink"])
def test_exact_public_hashes_and_regular_files_not_repaired(tmp_path, monkeypatch, kind):
    _, _, _, base = public_fixture(tmp_path, monkeypatch)
    name = gate.SELECTED[0][1]
    if kind == "rgb": (base / "inputs" / name).write_bytes(b"changed")
    elif kind == "mask": (base / "automatic_masks" / (Path(name).stem + "_human.png")).write_bytes(b"changed")
    elif kind == "manifest": (base / "inputs/manifest.json").write_text("{}")
    else:
        path = base / "inputs" / name; path.rename(path.with_suffix(".bin")); path.symlink_to(path.with_suffix(".bin").name)
    with pytest.raises(ValueError): gate.validate_public(tmp_path)


def audit_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.audit, "validate_previous", lambda *a: gate.audit.PREVIOUS_SHA)
    monkeypatch.setattr(gate.audit, "validate_failed", lambda *a: {"files": [{"file": "own-tiny-fixture"}]})
    directory = tmp_path / gate.audit.OUT; directory.mkdir(parents=True)
    data = {"stage": "pinned_dwpose_wheels_notice_audit_v3", "status": "pass", "phase": "complete",
        "producer_revision": gate.AUDIT_REVISION, "script_sha256": gate.AUDIT_SOURCE_SHA, "image_id": gate.audit.IMAGE,
        "original_receipt_sha256": gate.audit.FAILED_SHA, "original_producer_revision": gate.audit.FAILED_REVISION,
        "previous_failed_audit_sha256": gate.audit.PREVIOUS_SHA, "original_acquisition_script_sha256": gate.audit.SOURCE_SHA,
        "legacy_wheel_audits_omitted": True, "final_assets_receipt_source_rehashed": True, "network": "none", "device": "cpu", "oracle_modes": []}
    for key in ("gpu_used", "assets_redownloaded", "original_assets_modified", "original_failure_rewritten", "packages_installed", "upstream_source_executed", "inference_performed", "runtime_verified", "accuracy_verified", "license_clearance_verified", "training_data_rights_verified", "training_overlap_excluded", "challenge_eligibility_verified", "adoption_authorized", "private_truth_read", "challenge_inputs_used", "hand_labeled_test"): data[key] = False
    rows = []
    for package, count in (("onnxruntime", 353), ("flatbuffers", 14)):
        spec = gate.acquisition.WHEELS[package]
        row = {"package": package, "version": spec["version"], "member_count": count, "tags": spec["tags"], "requires_dist": spec["requires"],
               "safe_member_paths_verified": True, "selected_text_crc_verified": True, "symlinks_present": False, "archive_expanded": False,
               "embedded_license_present": package == "onnxruntime", "external_primary_license_verified": package == "flatbuffers", "retained_texts": []}
        names = ["METADATA", "WHEEL", "LICENSE"] if package == "onnxruntime" else ["METADATA", "WHEEL", "external-primary/flatbuffers-LICENSE"]
        for name in names:
            path = directory / package / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"own-notice-" + name.encode())
            row["retained_texts"].append({"file": name, "member": None if name.startswith("external-primary/") else name,
                                          "bytes": path.stat().st_size, "sha256": gate.acquisition.digest(path)})
        if package == "flatbuffers":
            old = next(r for r in gate.acquisition.ASSETS if r[0] == "licenses/flatbuffers-LICENSE")
            record = row["retained_texts"][-1]
            monkeypatch.setattr(gate.acquisition, "ASSETS", tuple((n, record["bytes"], record["sha256"], u) if n == old[0] else (n, s, h, u) for n, s, h, u in gate.acquisition.ASSETS))
            row.update(external_primary_license_source_revision=gate.acquisition.FLATBUFFERS_REV, external_primary_license_url=old[3])
        rows.append(row)
    data["wheel_audits"] = rows; path = directory / "report.json"
    def write(): path.write_text(json.dumps(data)); monkeypatch.setattr(gate, "AUDIT_SHA", gate.acquisition.digest(path))
    write(); return data, write, directory


def test_exact_v3_receipt_real_notice_hashes_and_old_failure_readonly(tmp_path, monkeypatch):
    _, _, directory = audit_fixture(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = gate.validate_assets(tmp_path)
    assert len(result["retained_notices"]) == 6
    assert result["original_receipt_sha256"] == gate.audit.FAILED_SHA
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["status", "source", "old_sha", "packages_installed", "overlap", "count", "version", "external", "url", "traversal", "duplicate", "text_sha"])
def test_notice_metadata_primarylicense_and_hashes_strict(tmp_path, monkeypatch, kind):
    data, write, _ = audit_fixture(tmp_path, monkeypatch)
    if kind == "status": data["status"] = "fail"
    elif kind == "source": data["script_sha256"] = "0" * 64
    elif kind == "old_sha": data["original_receipt_sha256"] = "0" * 64
    elif kind in ("packages_installed", "overlap"): data["packages_installed" if kind == "packages_installed" else "training_overlap_excluded"] = True
    elif kind == "count": data["wheel_audits"].pop()
    elif kind == "version": data["wheel_audits"][0]["version"] = "0"
    elif kind == "external": data["wheel_audits"][1]["retained_texts"][-1]["member"] = "pretend-embedded"
    elif kind == "url": data["wheel_audits"][1]["external_primary_license_url"] = "https://example.com/LICENSE"
    elif kind == "traversal": data["wheel_audits"][0]["retained_texts"][0]["file"] = "../private.json"
    elif kind == "duplicate": data["wheel_audits"][0]["retained_texts"].append(data["wheel_audits"][0]["retained_texts"][0].copy())
    elif kind == "text_sha": data["wheel_audits"][0]["retained_texts"][0]["sha256"] = "0" * 64
    write()
    with pytest.raises(ValueError): gate.validate_assets(tmp_path)


def test_source_pins_and_no_old_source_edits():
    data = gate.source_identity()
    assert data["dwpose_acquire.py"]["sha256"] == gate.audit.SOURCE_SHA
    assert data["dwpose_wheel_audit.py"]["sha256"] == gate.AUDIT_SOURCE_SHA


@pytest.mark.parametrize("failure", [False, True])
def test_private_prefix_removed_and_reported_even_on_failure(failure):
    report = {}; snapshots = []
    with pytest.raises(RuntimeError) if failure else __import__("contextlib").nullcontext():
        with gate.private_prefix(report, lambda: snapshots.append(report.copy())) as prefix:
            assert prefix.exists() and prefix.is_relative_to("/tmp") and report["private_prefix_removed"] is False
            (prefix / "own-disposable.txt").write_text("temporary fixture")
            if failure: raise RuntimeError("native operation failed")
    assert not prefix.exists() and report["private_prefix_removed"] is True and snapshots[-1]["private_prefix_removed"] is True


@pytest.mark.parametrize("values", [(4, 1, "sequential"), (0, 1, "sequential"), (4, 0, "sequential"), (4, 1, "parallel")])
def test_actual_session_options_not_just_requested(values):
    session = SimpleNamespace(get_session_options=lambda: SimpleNamespace(intra_op_num_threads=values[0], inter_op_num_threads=values[1], execution_mode=values[2]))
    ort = SimpleNamespace(ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"))
    if values == (4, 1, "sequential"): gate.validate_options(session, ort)
    else:
        with pytest.raises(ValueError): gate.validate_options(session, ort)


@pytest.mark.parametrize("kind", ["intact", "changed", "missing", "extra", "duplicate"])
def test_all_four_output_artifacts_rehashed_before_pass(tmp_path, kind):
    (tmp_path / "report.json").write_text("own receipt"); sessions = []
    for s in range(2):
        rows = []
        for f in range(2):
            name = f"session_{s}_frame_{f}.npz"; path = tmp_path / name
            with path.open("xb") as stream: np.savez(stream, own_fixture=np.array([s, f]))
            rows.append({"prediction_file": name, "prediction": gate.identity(path)})
        sessions.append({"predictions": rows})
    if kind == "intact": gate.validate_artifacts(tmp_path, sessions); return
    if kind == "changed": (tmp_path / sessions[0]["predictions"][0]["prediction_file"]).write_bytes(b"tampered")
    elif kind == "missing": (tmp_path / sessions[1]["predictions"][1]["prediction_file"]).unlink()
    elif kind == "extra": (tmp_path / "extra.npz").write_bytes(b"extra")
    elif kind == "duplicate": sessions[1]["predictions"][1] = sessions[0]["predictions"][0].copy()
    with pytest.raises(ValueError): gate.validate_artifacts(tmp_path, sessions)


@pytest.mark.parametrize("fail_native", [False, True])
def test_full_session_orchestration_native_style_list_pipeline_and_cleanup(tmp_path, monkeypatch, fail_native):
    """Own Python fixture + fake ORT exercises real driver; no actual model claim."""
    from PIL import Image
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 8)
    base = tmp_path / gate.PUBLIC; (base / "inputs").mkdir(parents=True); (base / "automatic_masks").mkdir()
    selected = []
    for clip in range(2):
        name = f"clip_{clip:02d}_frame_000.png"; human = Path(name).stem + "_human.png"
        Image.fromarray(np.full((8, 8, 3), 100 + clip, np.uint8)).save(base / "inputs" / name)
        mask = np.zeros((8, 8), np.uint8); mask[2:7, 1:6] = 255
        Image.fromarray(mask).save(base / "automatic_masks" / human)
        selected.append({"file": name, "human_mask_file": human, "human_mask_pixels": 25})
    source = tmp_path / gate.acquisition.BASE / "source/onnxpose.py"; source.parent.mkdir(parents=True)
    source.write_text('''import numpy as np
def inference_pose(session, out_bbox, oriImg):
    assert out_bbox.shape == (1,4) and oriImg.dtype == np.uint8
    # Own minimal fixture reproduces float64 normalization and list delegation.
    normalized=(np.full((384,288,3),oriImg[0,0,0],np.uint8)-np.array([123.675,116.28,103.53]))/np.array([58.395,57.12,57.375])
    outputs=session.run([o.name for o in session.get_outputs()], {session.get_inputs()[0].name:[normalized.transpose(2,0,1)]})
    points=np.stack([outputs[0].argmax(-1),outputs[1].argmax(-1)],-1).astype(np.float64)/2.
    return points,np.minimum(outputs[0].max(-1),outputs[1].max(-1))
''')
    constructed, supplied, pipcalls = [], [], []
    class Session:
        def __init__(self, path, sess_options, providers):
            self.metadata = metadata_session(); self.options = sess_options
            assert providers == ["CPUExecutionProvider"] and Path(path).name == gate.acquisition.ASSETS[0][0]
            constructed.append(self)
        def get_inputs(self): return self.metadata.get_inputs()
        def get_outputs(self): return self.metadata.get_outputs()
        def get_providers(self): return self.metadata.get_providers()
        def get_session_options(self): return self.options
        def disable_fallback(self): self.fallback_disabled = True
        def run(self, names, feed):
            assert self.fallback_disabled and type(feed["image"]) is list and feed["image"][0].dtype == np.float64
            supplied.append(feed["image"][0].copy())
            if fail_native: raise RuntimeError("own fixture native feed failure")
            x = np.zeros((1, 133, 576), np.float32); y = np.zeros((1, 133, 768), np.float32)
            x[..., 100] = 4.; y[..., 200] = 2.
            return [x, y]
    ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"), InferenceSession=Session)
    monkeypatch.setattr(gate, "source_identity", lambda: {"ownfixture": "stable"})
    monkeypatch.setattr(gate, "validate_assets", lambda root: {"ownfixture": "stable"})
    monkeypatch.setattr(gate, "validate_public", lambda root: {"selected": selected})
    monkeypatch.setattr(gate, "dependency_identity", lambda: {"ownfixture": "not actual runtime"})
    monkeypatch.setattr(gate, "import_runtime", lambda prefix: (ort, {}))
    monkeypatch.setattr(gate.util, "find_spec", lambda name: None)
    monkeypatch.setattr(gate.subprocess, "run", lambda argv, **kwargs: pipcalls.append(argv))
    out = tmp_path / "own-output"; out.mkdir(); (out / "report.json").write_text("own mutable receipt fixture")
    report = {"sessions": []}; snapshots = []
    if fail_native:
        with pytest.raises(RuntimeError, match="own fixture native feed failure"):
            gate.perform(tmp_path, out, report, lambda: snapshots.append(copy.deepcopy(report)), time.perf_counter())
        assert len(constructed) == len(supplied) == 1 and len(report["sessions"][0]["calls"]) == 1
        assert report["sessions"][0]["calls"][0]["run_completed"] is False and "native_cpu_abi_verified" not in report
    else:
        gate.perform(tmp_path, out, report, lambda: snapshots.append(copy.deepcopy(report)), time.perf_counter())
        assert len(constructed) == 2 and constructed[0] is not constructed[1] and len(supplied) == 4
        assert report["status"] == "pass" and report["native_cpu_abi_verified"] is True
        assert len(list(out.glob("*.npz"))) == 4 and report["two_session_byte_replay_verified"] is True
        assert np.array_equal(supplied[0], supplied[2]) and np.array_equal(supplied[1], supplied[3])
    assert len(pipcalls) == 1 and report["private_prefix_removed"] is True and not Path(report["private_prefix"]).exists()


def test_wrapper_syntax_exact_cpu_firewall_and_unknown_args():
    wrapper = INFRA / "run_dwpose_smoke.sh"; text = wrapper.read_text()
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    failed = subprocess.run(["bash", str(wrapper), "--cast-feed"], capture_output=True)
    assert failed.returncode == 2
    assert "--network none --memory 8g --cpus 4" in text and "--gpus" not in text and "CUDA_VISIBLE_DEVICES=" in text
    assert "eval_private" not in text and "mhr" not in text and "checkpoints" not in text
    assert text.count("/inputs/clip_") == 6 and text.count("_human.png") == 6
    assert "src=$BASE/inputs,dst=" not in text and "src=$BASE,dst=" not in text
    assert "--kill-after=10s 183s" in text and "PYTHONDONTWRITEBYTECODE=1" in text
    assert gate.audit.IMAGE in text

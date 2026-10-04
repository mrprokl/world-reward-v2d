"""Own tiny byte fixtures and analytic point maps; no dataset, models or GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
spec = importlib.util.spec_from_file_location("wr_ycbv_depth_test", ROOT / "infra/ycbv_point_depth.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def cohort(tmp_path):
    directory = tmp_path / gate.BASE / "inputs"; directory.mkdir(parents=True)
    rows = []
    for scene in gate.SCENES:
        for frame in range(96):
            name = f"scene_{scene:06d}_frame_{frame:06d}.png"; data = f"undecoded own bytes {scene}:{frame}".encode()
            path = directory / name; path.write_bytes(data); path.chmod(0o444)
            rows.append({"scene_id": scene, "frame_id": frame, "file": name, "sha256": hashlib.sha256(data).hexdigest(), "width": 640, "height": 480})
    manifest = {"schema": gate.SCHEMA, "revision": gate.REVISION, "license": "MIT", "selection": gate.SELECTION, "attribution": gate.ATTRIBUTION, "images": rows}
    path = directory / "manifest.json"; path.write_text(json.dumps(manifest)); path.chmod(0o444)
    pins = {"schema": gate.PINS_SCHEMA, "manifest": gate.files.identity(path),
        "acquisition_report": {"bytes": 123, "sha256": "a" * 64, "producer_revision": "b" * 40, "script_sha256": "c" * 64}}
    return directory, pins, manifest


def rewrite_manifest(directory, pins, manifest):
    path = directory / "manifest.json"; path.chmod(0o644); path.write_text(json.dumps(manifest)); path.chmod(0o444)
    pins["manifest"] = gate.files.identity(path)


def test_exact_reader_all288_no_decode_no_private(monkeypatch, tmp_path):
    directory, pins, manifest = cohort(tmp_path); original = Path.open
    def public_only(path, *args, **kwargs):
        assert path.parent == directory
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_only)
    records, proof = gate.public_inputs(directory, pins)
    assert len(records) == len(proof["RGB_identities"]) == 288
    assert [r["file"] for r in records] == list(gate.filenames())
    assert [r["scene_id"] for r in records[::96]] == [48, 49, 50]
    assert [{k: v for k, v in r.items() if k != "path"} for r in records] == manifest["images"]
    assert proof["manifest"] == pins["manifest"]


@pytest.mark.parametrize("fault", ["schema", "extra", "missing", "bytes_bool", "bytes_zero", "sha_short", "sha_upper", "producer", "script", "receipt_extra"])
def test_pin_firewall_before_any_file(tmp_path, monkeypatch, fault):
    directory, pins, _ = cohort(tmp_path)
    if fault == "schema": pins["schema"] = "other"
    elif fault == "extra": pins["depth"] = {}
    elif fault == "missing": pins.pop("manifest")
    elif fault == "bytes_bool": pins["manifest"]["bytes"] = True
    elif fault == "bytes_zero": pins["manifest"]["bytes"] = 0
    elif fault == "sha_short": pins["manifest"]["sha256"] = "a"
    elif fault == "sha_upper": pins["manifest"]["sha256"] = "A" * 64
    elif fault == "producer": pins["acquisition_report"]["producer_revision"] = "main"
    elif fault == "script": pins["acquisition_report"]["script_sha256"] = False
    else: pins["acquisition_report"]["private"] = True
    def forbidden(*args, **kwargs): pytest.fail("Invalid pins reached a file")
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


@pytest.mark.parametrize("fault", ["order", "duplicate", "short", "license", "revision", "selection", "attribution", "private", "row_private", "scene_bool", "frame_float", "grid", "name", "sha"])
def test_manifest_exact_cohort_not_adaptive(tmp_path, fault):
    directory, pins, manifest = cohort(tmp_path)
    if fault == "order": manifest["images"][0], manifest["images"][1] = manifest["images"][1], manifest["images"][0]
    elif fault == "duplicate": manifest["images"][1] = copy.deepcopy(manifest["images"][0])
    elif fault == "short": manifest["images"].pop()
    elif fault in ("license", "revision", "selection", "attribution"): manifest[fault] = "changed"
    elif fault == "private": manifest["K"] = [800]
    elif fault == "row_private": manifest["images"][0]["depth"] = "private"
    elif fault == "scene_bool": manifest["images"][0]["scene_id"] = True
    elif fault == "frame_float": manifest["images"][0]["frame_id"] = 0.
    elif fault == "grid": manifest["images"][0]["width"] = 480
    elif fault == "name": manifest["images"][0]["file"] = "../private.png"
    else: manifest["images"][-1]["sha256"] = "0" * 64
    rewrite_manifest(directory, pins, manifest)
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


@pytest.mark.parametrize("fault", ["missing", "extra", "mutate_unselected", "writable", "alias", "symlink", "manifest_sha"])
def test_all_original_hashes_and_public_file_inventory(tmp_path, fault):
    directory, pins, _ = cohort(tmp_path); target = directory / gate.filenames()[-1]
    if fault == "missing": target.unlink()
    elif fault == "extra": (directory / "private.json").write_text("{}")
    elif fault == "mutate_unselected": target.chmod(0o644); target.write_bytes(b"different"); target.chmod(0o444)
    elif fault == "writable": target.chmod(0o644)
    elif fault == "alias": os.link(target, tmp_path / "alias")
    elif fault == "symlink": target.unlink(); target.symlink_to(directory / gate.filenames()[0])
    else: pins["manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


def analytic_arrays(monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 4); monkeypatch.setattr(gate, "HEIGHT", 3)
    y, x = np.mgrid[:3, :4]; focal = 5.
    points = np.stack(((x + .5 - 2) / focal * 2, (y + .5 - 1.5) / focal * 2, np.full_like(x, 2.)), axis=-1).astype(np.float32)
    K = np.array([[focal / 4, 0, .5], [0, focal / 3, .5], [0, 0, 1]], dtype=np.float32)
    return {"depth": np.full((3, 4), 2., dtype=np.float32), "points": points,
        "mask": np.ones((3, 4), dtype=bool), "intrinsics": K, "frame_index": np.array(0, dtype=np.int64)}


def test_camera_native_dtype_full_grid_and_invalid_pixels_unmodified(monkeypatch):
    arrays = analytic_arrays(monkeypatch); arrays["mask"][0, 0] = False
    arrays["depth"][0, 0] = np.nan; arrays["points"][0, 0] = np.inf
    before = gate.array_identities(arrays); check = gate.validate_prediction_arrays(arrays)
    assert check["valid_pixels"] == 11 and check["excluded_pixels"] == 1
    assert check["metric_scale_accuracy_verified"] is False and gate.array_identities(arrays) == before


@pytest.mark.parametrize("fault", ["dtype", "mask", "shape", "frame", "extra", "K", "nonfinite_valid", "wrong_Z", "wrong_ray", "no_valid"])
def test_camera_prediction_contract_fails_without_repair(monkeypatch, fault):
    arrays = analytic_arrays(monkeypatch)
    if fault == "dtype": arrays["depth"] = arrays["depth"].astype(np.float64)
    elif fault == "mask": arrays["mask"] = arrays["mask"].astype(np.uint8)
    elif fault == "shape": arrays["points"] = arrays["points"][:2]
    elif fault == "frame": arrays["frame_index"] = np.array(1, dtype=np.int64)
    elif fault == "extra": arrays["K"] = arrays["intrinsics"]
    elif fault == "K": arrays["intrinsics"][0, 0] *= 2
    elif fault == "nonfinite_valid": arrays["depth"][0, 0] = np.nan
    elif fault == "wrong_Z": arrays["points"][0, 0, 2] += .1
    elif fault == "wrong_ray": arrays["points"][0, 0, 0] += .1
    else: arrays["mask"][:] = False
    with pytest.raises(ValueError): gate.validate_prediction_arrays(arrays)


def test_output_modes_at_creation_under_umask_zero_and_no_overwrite(tmp_path):
    old = os.umask(0)
    try:
        for mode in (0o400, 0o444):
            path = tmp_path / str(mode)
            with gate.exclusive(path, mode) as stream:
                assert path.stat().st_mode & 0o777 == mode; stream.write(b"tiny")
            with pytest.raises(FileExistsError): gate.exclusive(path, mode)
    finally: os.umask(old)


def test_marker_preserves_original_644_and_rejects_links(tmp_path):
    path = tmp_path / "revision"; path.write_bytes(b"a" * 40 + b"\n"); path.chmod(0o644)
    assert gate.marker_identity(path)["bytes"] == 41 and path.stat().st_mode & 0o777 == 0o644
    link = tmp_path / "link"; link.symlink_to(path)
    with pytest.raises(ValueError): gate.marker_identity(link)


def host_fixture(tmp_path, monkeypatch):
    directory, pins, _ = cohort(tmp_path); code = tmp_path / "current-code"; code.mkdir()
    (code / gate.PIN_FILE).parent.mkdir(); (code / gate.PIN_FILE).write_text(json.dumps(pins)); (code / gate.PIN_FILE).chmod(0o444)
    producer = pins["acquisition_report"]["producer_revision"]
    original = tmp_path / "jobs" / producer / "run_ycbv_point_acquire/code"
    for name in gate.ACQUISITION_FILES:
        path = original / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"own producer source " + name.encode()); path.chmod(0o444)
    for name, data in (("revision", (producer + "\n").encode()), ("source-sha256", b"d" * 64 + b"\n")):
        path = original.parent / name; path.write_bytes(data)
    pins["acquisition_report"]["script_sha256"] = gate.files.identity(original / gate.ACQUISITION_FILES[0])["sha256"]
    sources = {"files": {n: gate.files.identity(original / n) for n in gate.ACQUISITION_FILES},
        "markers": {n: gate.marker_identity(original.parent / n) for n in ("revision", "source-sha256")}}
    report = {"stage": "external_ycbv_contiguous_rgb_only_acquisition", "status": "pass", "phase": "complete",
        "producer_revision": producer, "script_sha256": pins["acquisition_report"]["script_sha256"],
        "dataset_revision": gate.REVISION, "license": "MIT", "image_id": gate.IMAGE, "device": "cpu",
        "gpu_used": False, "inference_performed": False, "challenge_inputs_used": False, "models_downloaded": False,
        "train_downloaded": False, "sparse_test_downloaded": False, "private_annotations_exported_as_inference_inputs": False,
        "selection_before_private_annotation_values": True, "selected_frames": 288, "all_instances_retained": True,
        "disposable_archives_removed": True, "source_rehashed_after": True, "public_manifest": pins["manifest"], "source_helpers": sources}
    path = directory.parent / "report.json"; path.write_text(json.dumps(report)); path.chmod(0o400)
    pins["acquisition_report"].update(gate.files.identity(path))
    pin_path = code / gate.PIN_FILE; pin_path.chmod(0o644); pin_path.write_text(json.dumps(pins)); pin_path.chmod(0o444)
    monkeypatch.setattr(gate, "bound_source", lambda *a, **k: {"own_tiny_source": True})
    return code, path, pins, report


def test_host_binds_actual_receipt_producer_only_without_annotations(tmp_path, monkeypatch):
    code, path, _, _ = host_fixture(tmp_path, monkeypatch)
    original_open = Path.open
    def no_private(path, *args, **kwargs):
        assert "eval_private" not in path.parts
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_private)
    assert len(gate.host_proof(tmp_path, code, "e" * 40)) == 64
    path.chmod(0o600); path.write_text("invalid unpinned JSON"); path.chmod(0o400)
    called = False; original_json = gate.files.strict_json
    def parse(raw):
        nonlocal called
        if raw == b"invalid unpinned JSON": called = True
        return original_json(raw)
    monkeypatch.setattr(gate.files, "strict_json", parse)
    with pytest.raises(ValueError): gate.host_proof(tmp_path, code, "e" * 40)
    assert not called


def test_wrapper_native_scope_and_syntax_and_runtime_closure():
    wrapper = ROOT / "infra/run_ycbv_point_depth.sh"; text = wrapper.read_text()
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    for literal in ("--gpus all", "--network none", "--user 1000:1000", "--read-only", "--cap-drop ALL", "flock --nonblock 9", "330s", "--cidfile", "--entrypoint /usr/bin/env", "WR_YCBV_HOST_PROOF_SHA256"):
        assert literal in text
    assert 'src=$BASE/eval_private' not in text and 'src=$BASE/report.json' not in text and 'src=$CODE,dst=$CODE' not in text
    assert 'exec 9<"$ROOT/jobs/.world-reward-h100.lock"' in text and 'exec 9>' not in text
    assert "if not lock.exists():" in text and 'lock_fd_identity' in text and 'before.st_nlink!=1' in text
    source = (ROOT / "infra/ycbv_point_depth.py").read_text()
    assert "network.infer(tensor[None], fov_x=fov)" in source
    assert "force_projection=" not in source and "apply_mask=" not in source and "native.seed" not in source
    launcher_spec = importlib.util.spec_from_file_location("wr_ycbv_bundle_test", ROOT / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(launcher_spec); launcher_spec.loader.exec_module(launcher)
    code = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ("infra", "src", "configs") for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".json")}
    closure = launcher.runtime_bundle_paths(code, "infra/run_ycbv_point_depth.sh")
    assert set(gate.SOURCE_FILES) <= set(closure)
    assert set(gate.ACQUISITION_FILES) <= set(closure)


def test_top_level_stdlib_host_import_does_not_load_numerical_runtime():
    script = "import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2]);assert not {'numpy','torch','moge','object_synthetic_observations'}&sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "infra"), str(ROOT / "infra/ycbv_point_depth.py")], check=True)

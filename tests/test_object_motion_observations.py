"""Tiny CPU contracts only; no RGB challenge data, weights or actual inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("wr_test_motion_observations", directory/"object_motion_observations.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inputs(gate, tmp_path):
    folder = tmp_path/"inputs"; folder.mkdir(); records = []
    for obj in range(gate.OBJECTS):
        for frame in range(gate.FRAMES):
            name = f"object_{obj:02d}_frame_{frame:03d}.png"; value = bytes([obj, frame])+b"ownRGBfixture"
            (folder/name).write_bytes(value)
            records.append({"file": name, "sha256": hashlib.sha256(value).hexdigest(), "width": 512, "height": 384})
    (folder/"manifest.json").write_text(json.dumps({"schema": gate.SCHEMA, "images": records}))
    return folder, records


def test_all_24_rgb_inputs_exact_order_and_no_private_metadata(gate, tmp_path):
    folder, records = inputs(gate, tmp_path); selected, receipt = gate.public_inputs(folder)
    assert [(r["object_index"], r["frame_index"]) for r in selected] == [(o, f) for o in range(3) for f in range(8)]
    assert [r["file"] for r in selected] == [r["file"] for r in records]
    assert all(r["bytes"] > 0 and r["path"].exists() for r in selected)
    assert receipt == gate.observations_helper.identity(folder/"manifest.json")


@pytest.mark.parametrize("fault", ["private", "record_private", "order", "count", "size_bool", "size_wrong", "sha", "sha_type", "empty", "missing", "extra", "symlink", "manifest_symlink", "old_schema", "traversal"])
def test_manifest_files_fail_closed(gate, tmp_path, fault):
    folder, records = inputs(gate, tmp_path); path = folder/"manifest.json"; manifest = json.loads(path.read_text())
    if fault == "private": manifest["ground_truth"] = []
    elif fault == "record_private": manifest["images"][0]["K"] = []
    elif fault == "order": manifest["images"][0], manifest["images"][1] = manifest["images"][1], manifest["images"][0]
    elif fault == "count": manifest["images"].pop()
    elif fault == "size_bool": manifest["images"][0]["width"] = True
    elif fault == "size_wrong": manifest["images"][0]["height"] = 512
    elif fault == "sha": manifest["images"][0]["sha256"] = "a"*64
    elif fault == "sha_type": manifest["images"][0]["sha256"] = 3
    elif fault == "empty": (folder/records[0]["file"]).write_bytes(b""); manifest["images"][0]["sha256"] = hashlib.sha256(b"").hexdigest()
    elif fault == "missing": (folder/records[0]["file"]).unlink()
    elif fault == "extra": (folder/"camera.json").write_text("{}")
    elif fault == "old_schema": manifest["schema"] = "world-reward-objects-rgb-inputs-v1"
    elif fault == "traversal": manifest["images"][0]["file"] = "../outside.png"
    elif fault == "symlink":
        target = folder/records[0]["file"]; saved = tmp_path/"outside"; target.rename(saved); target.symlink_to(saved)
    path.write_text(json.dumps(manifest))
    if fault == "manifest_symlink":
        saved = tmp_path/"outside_manifest"; path.rename(saved); path.symlink_to(saved)
    with pytest.raises((ValueError, FileNotFoundError)): gate.public_inputs(folder)


def test_reuses_fixed_mask_and_camera_validation_without_geometry_fill(gate):
    h, w = gate.HEIGHT, gate.WIDTH; yy, xx = np.mgrid[:h, :w]; f = 640.
    rgb = np.full((h, w, 3), 200, np.uint8); rgb[20:40, 20:40] = 50
    foreground, _ = gate.observations_helper.foreground_mask(rgb)
    depth = np.full((h, w), 2, np.float32)
    points = np.stack(((xx+.5-w/2)/f*depth, (yy+.5-h/2)/f*depth, depth), axis=-1).astype(np.float32)
    normalized = np.array([[f/w, 0., .5], [0., f/h, .5], [0., 0., 1.]])
    valid = np.ones((h, w), np.bool_); valid[20:22, 20:40] = False
    arrays, checks = gate.observations_helper.observations(rgb, foreground, depth, points, valid, normalized)
    assert arrays["mask"].sum() == 360 and arrays["pointmap"].shape == (3, h, w)
    assert np.array_equal(arrays["pointmap"].transpose(1, 2, 0), points) and arrays["K"][0, 0] == 640
    assert checks["status"] == "pass" and checks["metric_scale_accuracy_verified"] is False
    points[0, 0, 0] = np.nan
    with pytest.raises(ValueError): gate.observations_helper.observations(rgb, foreground, depth, points, valid, normalized)


def test_array_receipts_cover_indices_and_exact_c_order(gate):
    arrays = {"pointmap": np.arange(24, dtype=np.float32).reshape(3, 2, 4), "mask": np.ones((2, 4), bool),
              "object_index": np.array(2, np.int64), "frame_index": np.array(7, np.int64)}
    actual = gate.array_identities(arrays)
    assert set(actual) == set(arrays)
    for name, value in arrays.items():
        assert actual[name] == {"sha256": hashlib.sha256(value.tobytes()).hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}


def test_failed_prerequisite_preserves_exclusive_partial_before_heavy_imports(gate, tmp_path, monkeypatch):
    output = tmp_path/"validation/object_motion_v1/observations"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="regular canonical"): gate.main([])
    path = output/"report.json"; frozen = path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["MoGe_forward_calls"] == 0 and report["private_truth_read"] is False
    assert report["budget_seconds"] == 180 and report["helper_sha256"] == gate.sha256(Path(gate.observations_helper.__file__))
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_wrapper_only_public_rgb_and_exact_existing_weights(gate):
    with pytest.raises(SystemExit): gate.main(["--episode", "0"])
    wrapper = Path(gate.__file__).with_name("run_object_motion_observations.sh").read_text(); source = Path(gate.__file__).read_text()
    assert "for object in 00 01 02" in wrapper and "for frame in 000 001 002 003 004 005 006 007" in wrapper
    assert "183s docker run" in wrapper and "--network none" in wrapper and "--memory 32g --cpus 4" in wrapper
    assert '--entrypoint python' in wrapper and 'chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"' in wrapper
    assert "apply_mask=False" in source and "observations_helper.observations" in source
    assert "eval_private" not in wrapper and "chown -R" not in wrapper and "src=$BASE,dst=$BASE" not in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor", "validation,dst"))
    assert wrapper.count("hub/blobs/9f/") == 2 and "weights-acquisition.json" in wrapper

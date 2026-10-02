"""Tiny pure RGB/pinhole/provenance tests; no model, torch, GPU or private GT."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/object_synthetic_observations.py"
    spec = importlib.util.spec_from_file_location("wr_test_object_observations", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inputs(gate, tmp_path):
    folder = tmp_path/"inputs"; folder.mkdir(); records = []
    for obj in range(2):
        for view in range(6):
            file = f"object_{obj:02d}_view_{view:02d}.png"; value = bytes([obj, view])+b"tinyRGBfixture"
            if view in gate.TRAIN_VIEWS: (folder/file).write_bytes(value)
            records.append({"file": file, "sha256": gate.hashlib.sha256(value).hexdigest(), "width": 512, "height": 384})
    (folder/"manifest.json").write_text(json.dumps({"schema": gate.SCHEMA, "images": records}))
    return folder, records


def test_only_six_training_pixels_manifest_twelve_identities(gate, tmp_path):
    folder, records = inputs(gate, tmp_path)
    selected, receipt, all_images = gate.public_inputs(folder)
    assert [(r["object_index"], r["view_index"]) for r in selected] == [(o, v) for o in range(2) for v in (0, 2, 4)]
    assert len(all_images) == 12 and receipt["sha256"] == gate.sha256(folder/"manifest.json")
    assert all(not (folder/r["file"]).exists() for r in all_images if r["view_index"] in (1, 3, 5))


@pytest.mark.parametrize("fault", ["private", "wrong_order", "width", "sha", "missing", "heldout_exposed", "symlink"])
def test_public_whitelist_fail_closed(gate, tmp_path, fault):
    folder, records = inputs(gate, tmp_path); path = folder/"manifest.json"; manifest = json.loads(path.read_text())
    if fault == "private": manifest["camera_K"] = []
    elif fault == "wrong_order": manifest["images"][0], manifest["images"][1] = manifest["images"][1], manifest["images"][0]
    elif fault == "width": manifest["images"][0]["width"] = True
    elif fault == "sha": manifest["images"][0]["sha256"] = "a"*64
    elif fault == "missing": (folder/records[0]["file"]).unlink()
    elif fault == "heldout_exposed": (folder/records[1]["file"]).write_bytes(b"heldout")
    else:
        target = folder/records[0]["file"]; saved = tmp_path/"outside"; target.rename(saved); target.symlink_to(saved)
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)): gate.public_inputs(folder)


def test_generic_border_median_preserves_object_hole_no_gt(gate):
    rgb = np.full((24, 28, 3), 200, np.uint8); rgb[5:19, 7:21] = [50, 90, 120]; rgb[10:14, 12:16] = 200
    mask, evidence = gate.foreground_mask(rgb)
    assert mask.sum() == 180 and not mask[10:14, 12:16].any()
    assert evidence["distance_threshold"] == .08 and evidence["minimum_component_pixels"] == 64
    assert evidence["connectivity"] == 4 and evidence["background_median_RGB_unit"] == [200/255]*3


@pytest.mark.parametrize("fault", ["absent", "small", "ambiguous", "float", "channels"])
def test_absence_ambiguity_never_fullimage_fallback(gate, fault):
    rgb = np.full((32, 32, 3), 200, np.uint8)
    if fault == "small": rgb[4:9, 4:9] = 0
    elif fault == "ambiguous": rgb[2:10, 2:10] = 0; rgb[20:28, 20:28] = 0
    elif fault == "float": rgb = rgb.astype(float)
    elif fault == "channels": rgb = rgb[..., 0]
    with pytest.raises(ValueError): gate.foreground_mask(rgb)


def prediction_fixture(gate):
    h, w = gate.HEIGHT, gate.WIDTH; f = np.hypot(w, h); y, x = np.mgrid[:h, :w]
    depth = np.full((h, w), 2, np.float32)
    points = np.stack(((x+.5-w/2)/f*depth, (y+.5-h/2)/f*depth, depth), axis=-1).astype(np.float32)
    normalized = np.array([[f/w, 0., .5], [0., f/h, .5], [0., 0., 1.]])
    rgb = np.zeros((h, w, 3), np.uint8); mask = np.zeros((h, w), np.bool_); mask[8:20, 8:20] = True
    return [rgb, mask, depth, points, np.ones((h, w), np.bool_), normalized]


def test_native_fullgrid_points_retained_exactly_and_foreground_intersected(gate):
    args = prediction_fixture(gate); args[4][8:10, 8:20] = False
    arrays, checks = gate.observations(*args)
    assert arrays["mask"].sum() == 120 and arrays["pointmap"].shape == (3, 384, 512)
    assert np.array_equal(arrays["pointmap"].transpose(1, 2, 0), args[3])
    assert arrays["pointmap"].dtype == np.float32 and checks["status"] == "pass"
    assert checks["metric_scale_accuracy_verified"] is False and arrays["K"][0, 0] == 640


@pytest.mark.parametrize("fault", ["background_nan", "foreground_inf", "negative", "wrong_K", "too_few", "validity_float"])
def test_invalid_points_not_filled_or_hidden_behind_mask(gate, fault):
    args = prediction_fixture(gate)
    if fault == "background_nan": args[3][0, 0, 0] = np.nan
    elif fault == "foreground_inf": args[3][9, 9, 0] = np.inf
    elif fault == "negative": args[2][0, 0] = -1; args[3][0, 0, 2] = -1
    elif fault == "wrong_K": args[5][0, 0] *= 1.1
    elif fault == "too_few": args[4][:] = False
    else: args[4] = args[4].astype(float)
    with pytest.raises(ValueError): gate.observations(*args)


def test_installed_source_pin_sha_are_not_arbitrary_metadata(gate, tmp_path, monkeypatch):
    model = tmp_path/"moge/model"; model.mkdir(parents=True); (model/"v2.py").write_text("tiny own fixture")
    monkeypatch.setattr(gate, "SOURCE_V2_SHA", gate.sha256(model/"v2.py"))
    direct = {"url": "https://github.com/microsoft/MoGe.git", "vcs_info": {"vcs": "git", "commit_id": gate.SOURCE_REVISION}}
    assert gate.installed_source(model.parent, direct)["v2_source_sha256"] == gate.SOURCE_V2_SHA
    direct["vcs_info"]["commit_id"] = "a"*40
    with pytest.raises(ValueError): gate.installed_source(model.parent, direct)


def test_snapshot_only_exact_hf_sha_blob_link_permitted(gate, tmp_path, monkeypatch):
    cache = tmp_path/"weights/cari4d/hf_home/hub"; model = cache/"models--Ruicheng--moge-2-vitl-normal"
    blob = model/"blobs"/gate.MODEL_SHA; blob.parent.mkdir(parents=True); blob.write_bytes(b"tinyweightfixture")
    snapshot = model/f"snapshots/{gate.MODEL_REVISION}/model.pt"; snapshot.parent.mkdir(parents=True); snapshot.symlink_to(blob)
    (tmp_path/"results").mkdir(); (tmp_path/"results/weights-acquisition.json").write_text(json.dumps({"assets": [
        {"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": gate.MODEL_REVISION, "cache_dir": str(cache)}]}))
    old = gate.identity
    monkeypatch.setattr(gate, "identity", lambda p: {"sha256": gate.MODEL_SHA, "bytes": gate.MODEL_BYTES} if Path(p) == blob else old(p))
    assert gate.model_asset(tmp_path)[0] == blob
    snapshot.unlink(); snapshot.symlink_to(tmp_path/"other")
    with pytest.raises(ValueError): gate.model_asset(tmp_path)


def test_failed_prerequisite_preserves_exclusive_partial(gate, tmp_path, monkeypatch):
    output = tmp_path/"validation/objects_rgb_v1/observations"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="regular canonical"): gate.main([])
    path = output/"report.json"; frozen = path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["MoGe_forward_calls"] == 0 and report["private_truth_read"] is False
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_wrapper_no_heldout_private_parent_or_challenge_mounts(gate):
    with pytest.raises(SystemExit): gate.main(["--object", "0"])
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_object_synthetic_observations.sh").read_text()
    assert "apply_mask=False" in source and "validate_camera_pointmap" in source
    assert "for view in 00 02 04" in wrapper and "for object in 00 01" in wrapper
    assert "src=$BASE/inputs,dst=$BASE/inputs" not in wrapper and "src=$BASE,dst=$BASE" not in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor", "validation,dst"))
    assert "eval_private" not in wrapper and "chown -R" not in wrapper
    assert "123s docker run" in wrapper and "--network none" in wrapper and "--memory 16g" in wrapper

"""Pure tiny J1 schema/grounding contracts; no model, private truth or GPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_joint_infer_test", infra/"joint_rgb_infer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def public(gate, tmp_path):
    base = tmp_path/"validation/joint_rgb_v1"; inputs = base/"inputs"; masks = base/"automatic_masks"
    inputs.mkdir(parents=True); masks.mkdir(); images = []; records = []
    for clip in range(3):
        for frame in range(3):
            filename = f"clip_{clip:02d}_frame_{frame:03d}.png"; stem = Path(filename).stem
            p = inputs/filename; p.write_bytes(bytes([clip, frame])+b"own fixture")
            images.append({"file": filename, "sha256": gate.sha256(p), "width": 1024, "height": 768})
            record = {"file": filename, "rgb_sha256": gate.sha256(p), "clip_index": clip, "frame_index": frame}
            for kind in ("human", "object"):
                name = f"{stem}_{kind}.png"; p = masks/name; p.write_bytes(kind.encode()+bytes([clip, frame]))
                record[kind+"_mask_file"] = name; record[kind+"_mask_sha256"] = gate.sha256(p)
            records.append(record)
    mp = inputs/"manifest.json"; mp.write_text(json.dumps({"schema": "world-reward-joint-rgb-v1", "images": images}))
    rp = masks/"report.json"; rp.write_text(json.dumps({"stage": "public_joint_rgb_automatic_masks", "status": "pass",
        "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "frames": 9,
        "input_manifest_sha256": gate.sha256(mp), "records": records}))
    return mp, rp


def test_public_nine_exact_order_hashes_no_private(gate, tmp_path):
    public(gate, tmp_path); records, hashes = gate.public_inputs(tmp_path)
    assert [(r["clip_index"], r["frame_index"]) for r in records] == [(c, f) for c in range(3) for f in range(3)]
    assert set(hashes) == {"public_inputs_sha", "mask_report_sha"} and len(records) == 9


@pytest.mark.parametrize("field", ["ground_truth_used", "hand_labeled_test"])
@pytest.mark.parametrize("bad", [True, 0, None, "False", "missing"])
def test_explicit_no_gt_no_hand_labels_required_in_mask_receipt(gate, tmp_path, field, bad):
    _, rp = public(gate, tmp_path); report = json.loads(rp.read_text())
    if bad == "missing": del report[field]
    else: report[field] = bad
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError): gate.public_inputs(tmp_path)


@pytest.mark.parametrize("fault", ["private", "width_bool", "mask_order", "frames_list", "frames_bool", "wrong_stage", "sha", "mask_file", "missing", "extra", "symlink", "private_flag", "old_manifest"])
def test_public_inputs_fail_closed(gate, tmp_path, fault):
    mp, rp = public(gate, tmp_path); m = json.loads(mp.read_text()); r = json.loads(rp.read_text())
    if fault == "private": m["K"] = []
    elif fault == "width_bool": m["images"][0]["width"] = True
    elif fault == "mask_order": r["records"][0], r["records"][1] = r["records"][1], r["records"][0]
    elif fault == "frames_list": r["frames"] = r["records"]
    elif fault == "frames_bool": r["frames"] = True
    elif fault == "wrong_stage": r["stage"] = "joint_rgb_automatic_person_object_masks"
    elif fault == "sha": r["records"][0]["human_mask_sha256"] = "a"*64
    elif fault == "mask_file": r["records"][0]["object_mask_file"] = "../outside.png"
    elif fault == "missing": (mp.parent/m["images"][0]["file"]).unlink()
    elif fault == "extra": (mp.parent/"private.npz").write_bytes(b"no")
    elif fault == "symlink":
        p = mp.parent/m["images"][0]["file"]; saved = tmp_path/"outside"; p.rename(saved); p.symlink_to(saved)
    elif fault == "private_flag": r["private_truth_read"] = True
    elif fault == "old_manifest": m["schema"] = "world-reward-hands-rgb-inputs-v1"
    mp.write_text(json.dumps(m)); r["input_manifest_sha256"] = gate.sha256(mp); rp.write_text(json.dumps(r))
    with pytest.raises(ValueError): gate.public_inputs(tmp_path)


def tiny_grids(gate, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 8); monkeypatch.setattr(gate, "WIDTH", 8)
    depth = np.full((9, 8, 8), 2., np.float32); points = np.ones((9, 8, 8, 3), np.float32); points[..., 2] = depth
    render = depth.copy(); render[:3] *= 2; render[3:6] *= 3; render[6:] *= 4
    mask = np.ones((9, 8, 8), bool); objects = np.zeros_like(mask); objects[:, 0, :8] = True
    return points, depth, render, mask.copy(), mask.copy(), objects, mask.copy()


def test_shared_clip_alpha_excludes_object_and_changes_xyz_once_not_body(gate, monkeypatch):
    args = tiny_grids(gate, monkeypatch); args[2][:, 0] *= 1000; before = args[0].copy()
    aligned, scales, diagnostics = gate.align_clips(*args)
    assert np.allclose(scales, [2., 3., 4.]) and len(diagnostics) == 3
    assert np.array_equal(args[0], before)
    assert np.array_equal(aligned, before*np.repeat(scales, 3).astype(np.float32)[:, None, None, None])
    assert all(r["supported_frames"] == 3 and r["correspondences"] == 3*56 for r in diagnostics)


@pytest.mark.parametrize("fault", ["unsupported", "object_all", "validity_float", "nan_valid", "bad_Z"])
def test_scale_support_or_invalid_valid_points_fail_without_fallback(gate, monkeypatch, fault):
    args = list(tiny_grids(gate, monkeypatch))
    if fault == "unsupported": args[3][0] = False
    elif fault == "object_all": args[5][:] = True
    elif fault == "validity_float": args[6] = args[6].astype(float)
    elif fault == "nan_valid": args[0][0, 1, 1, 0] = np.nan
    elif fault == "bad_Z": args[0][0, 1, 1, 2] = -1
    with pytest.raises(ValueError): gate.align_clips(*args)


def test_invalid_unselected_values_not_fabricated(gate, monkeypatch):
    args = tiny_grids(gate, monkeypatch); args[6][:, 0] = False; args[0][:, 0] = np.nan
    result, _, _ = gate.align_clips(*args)
    assert np.isnan(result[:, 0]).all()


def test_fixed_prior_pointmap_rays_no_camera_fit(gate, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 6); monkeypatch.setattr(gate, "WIDTH", 8)
    y, x = np.mgrid[:6, :8]; K = np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]])
    monkeypatch.setattr(gate, "CAMERA_K", K); depth = np.full((6, 8), 2., np.float32)
    points = np.stack(((x+.5-4)/10*depth, (y+.5-3)/10*depth, depth), axis=-1).astype(np.float32)
    valid = np.ones((6, 8), bool); norm = np.diag([1/8, 1/6, 1.])@K
    assert gate.pointmap_contract(depth, points, valid, norm)["status"] == "pass"
    norm[0, 0] *= 1.1
    with pytest.raises(ValueError): gate.pointmap_contract(depth, points, valid, norm)


def test_early_failure_report_immutable_before_heavy_import(gate, tmp_path, monkeypatch):
    output = tmp_path/"validation/joint_rgb_v1/predictions"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="regular canonical"): gate.main([])
    p = output/"report.json"; frozen = p.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["body_calls_completed"] == 0 and report["private_truth_read"] is False
    assert report["camera_prior"] == [[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]]
    with pytest.raises(FileExistsError): gate.main([])
    assert p.read_bytes() == frozen


def test_wrapper_mount_scope_and_unchanged_loaders(gate):
    wrapper = Path(gate.__file__).with_name("run_joint_rgb_infer.sh").read_text(); source = Path(gate.__file__).read_text()
    assert "303s docker run" in wrapper and "--network none" in wrapper and "--memory 32g --cpus 4" in wrapper
    assert '--entrypoint python' in wrapper and 'chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"' in wrapper
    assert "eval_private" not in wrapper and "chown -R" not in wrapper and "src=$BASE,dst=$BASE" not in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "validation,dst"))
    assert "human.load_model(root, torch)" in source and "human.decode_prediction" in source
    assert 'inference_type="body"' in source and "cam_int=camera" in source and "apply_mask=False" in source
    assert "fit_shared_depth_scale" in source and "~object_masks" in source
    with pytest.raises(SystemExit): gate.main(["--episode", "0"])

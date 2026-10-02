"""Pure tiny J1 schema/grounding contracts; no model, private truth or GPU."""
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("focal", [1280., 960., 1600., 23., 12000.])
def test_learned_camera_uses_actual_normalized_intrinsics_without_clamp(gate, focal):
    K = np.array([[focal, 0., 512.], [0., focal, 384.], [0., 0., 1.]])
    assert np.allclose(gate.camera_candidate(np.diag([1/1024, 1/768, 1.])@K), K)


@pytest.mark.parametrize("fault", ["negative", "zero", "nan", "offcenter", "anisotropic", "skew", "projective", "dtype", "shape"])
def test_native_learned_camera_fail_closed(gate, fault):
    K = np.diag([1/1024, 1/768, 1.])@gate.CAMERA_K
    if fault == "negative": K[0, 0] = -1.
    elif fault == "zero": K[0, 0] = 0.
    elif fault == "nan": K[0, 0] = np.nan
    elif fault == "offcenter": K[0, 2] += .01
    elif fault == "anisotropic": K[1, 1] *= 1.1
    elif fault == "skew": K[0, 1] = .01
    elif fault == "projective": K[2, 0] = .01
    elif fault == "dtype": K = K.astype(int)
    else: K = K[0]
    with pytest.raises(ValueError): gate.camera_candidate(K)


def test_three_frame_median_shared_clip_camera_no_truth_or_clamp(gate):
    focals = np.array([2300., 1700., 2100., 15., 11., 16., 900., 750., 800.])
    K = gate.clip_cameras(focals)
    assert K.shape == (3, 3, 3) and np.array_equal(K[:, 0, 0], [2100., 15., 800.])
    assert np.array_equal(K[:, 1, 1], K[:, 0, 0])
    assert np.array_equal(K[:, :2, 2], [[512., 384.]]*3)
    assert np.array_equal(focals, [2300., 1700., 2100., 15., 11., 16., 900., 750., 800.])
    for bad in (focals[:8], focals.astype(int), np.full(9, np.nan), np.zeros(9)):
        with pytest.raises(ValueError): gate.clip_cameras(bad)


class TinyTensor(np.ndarray):
    def float(self): return self.astype(np.float32)
    def unsqueeze(self, dim): return np.expand_dims(self, dim)
    def squeeze(self, dim): return np.ndarray.squeeze(self, axis=dim)


def solver_fixture(gate, monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 128); monkeypatch.setattr(gate, "HEIGHT", 128)
    def nearest(x, size, mode):
        assert size == (64, 64) and mode == "nearest"
        return x[..., ::2, ::2]
    torch = SimpleNamespace(is_tensor=lambda x: isinstance(x, TinyTensor), bool=np.bool_,
        inference_mode=nullcontext, nn=SimpleNamespace(functional=SimpleNamespace(interpolate=nearest)))
    points = np.ones((1, 128, 128, 3), np.float32).view(TinyTensor)
    mask = np.zeros((1, 128, 128), bool).view(TinyTensor); mask[0, 0, 0] = mask[0, 2, 2] = True
    calls = []
    def original(p, m, *, focal, downsample_size):
        calls.append((p, m, focal, downsample_size)); return "original_unchanged"
    module = SimpleNamespace(recover_focal_shift=original)
    def infer(image, *, fov_x, apply_mask):
        assert apply_mask is False
        return module.recover_focal_shift(points, mask, focal=None if fov_x is None else 1.)
    return torch, module, SimpleNamespace(infer=infer), points, mask, calls, original


def test_both_camera_passes_call_original_solver_and_restore(gate, monkeypatch):
    torch, module, net, points, _, calls, original = solver_fixture(gate, monkeypatch); diagnostics = []
    for i in range(18):
        result = gate.checked_depth_infer(torch, module, net, points, None if i < 9 else 45., diagnostics, {"index": i})
        assert result == "original_unchanged" and module.recover_focal_shift is original
    assert len(calls) == len(diagnostics) == 18
    assert all(r["native_nearest64_valid_pixels"] == 2 and r["original_solver_returned"] for r in diagnostics)
    assert [r["focal_prior_supplied"] for r in diagnostics] == [False]*9+[True]*9


def test_unsampled_valid_pixels_do_not_hide_native_fallback(gate, monkeypatch):
    torch, module, net, points, mask, calls, original = solver_fixture(gate, monkeypatch)
    mask[:] = False; mask[0, 1::2, 1::2] = True; mask[0, 0, 0] = True; diagnostics = []
    with pytest.raises(ValueError, match="default-camera fallback"):
        gate.checked_depth_infer(torch, module, net, points, None, diagnostics, {})
    assert not calls and module.recover_focal_shift is original
    assert diagnostics == [{"native_nearest64_valid_pixels": 1, "focal_prior_supplied": False, "original_solver_returned": False}]


@pytest.mark.parametrize("fault", ["mask_dtype", "mask_shape", "backend", "no_solver"])
def test_solver_abi_or_backend_errors_restore_hook(gate, monkeypatch, fault):
    torch, module, net, points, mask, _, original = solver_fixture(gate, monkeypatch)
    if fault == "mask_dtype": net.infer = lambda *a, **k: module.recover_focal_shift(points, mask.astype(float))
    elif fault == "mask_shape": net.infer = lambda *a, **k: module.recover_focal_shift(points, mask[0])
    elif fault == "backend":
        def fail(*a, **k): raise RuntimeError("backend failure")
        net.infer = fail
    else: net.infer = lambda *a, **k: None
    with pytest.raises(RuntimeError): gate.checked_depth_infer(torch, module, net, points, None, [], {})
    assert module.recover_focal_shift is original


def test_explicit_learned_K_pointmap_validation(gate, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 6); monkeypatch.setattr(gate, "WIDTH", 8)
    K = np.array([[11., 0., 4.], [0., 11., 3.], [0., 0., 1.]])
    y, x = np.mgrid[:6, :8]; depth = np.full((6, 8), 2., np.float32)
    p = np.stack(((x+.5-4)/11*depth, (y+.5-3)/11*depth, depth), axis=-1).astype(np.float32)
    assert gate.pointmap_contract(depth, p, np.ones((6, 8), bool), np.diag([1/8, 1/6, 1.])@K, K)["status"] == "pass"
    with pytest.raises(ValueError): gate.pointmap_contract(depth, p, np.ones((6, 8), bool), np.diag([1/8, 1/6, 1.])@K, K*2)


def test_learned_namespace_failure_never_overwrites_fixed(gate, tmp_path, monkeypatch):
    base = tmp_path/"validation/joint_rgb_v1"; fixed = base/"predictions"; fixed.mkdir(parents=True)
    (fixed/"report.json").write_bytes(b"immutable J1")
    out = base/"predictions_camera_v1"; out.mkdir()
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); iterdir = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else iterdir(p))
    with pytest.raises(ValueError): gate.main(["--camera-source", "learned"])
    report = json.loads((out/"report.json").read_text())
    assert report["stage"] == gate.CAMERA_STAGE and report["budget_seconds"] == 600
    assert report["camera_source"] == "learned" and report["focal_fitted"] is True and report["camera_prior"] is None
    assert report["body_calls_completed"] == report["MoGe_calls_completed"] == 0
    assert (fixed/"report.json").read_bytes() == b"immutable J1"


def test_learned_wrapper_scope_and_network_call_contract(gate):
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_joint_rgb_camera_infer.sh").read_text()
    assert "603s docker run" in wrapper and '--camera-source learned' in wrapper and 'OUT="$BASE/predictions_camera_v1"' in wrapper
    assert '--memory 32g --cpus 4' in wrapper and '--network none' in wrapper and 'eval_private' not in wrapper
    assert '"camera_K": cameras if learned else CAMERA_K.copy()' in source
    assert 'camera_candidate(normalized)' in source and 'focal_candidates_pixels=focals.tolist()' in source
    assert 'chown -R' not in wrapper

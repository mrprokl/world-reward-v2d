"""Tiny CPU-only public inference contracts; no sensor truth, model or GPU."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_tudl_infer_test", infra/"tudl_infer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def inputs(gate, tmp_path):
    folder = tmp_path/"inputs"; folder.mkdir(); images = []
    for scene, frames in enumerate(gate.FRAME_IDS, 1):
        for frame in frames:
            filename = f"scene_{scene:06d}_frame_{frame:06d}.png"; path = folder/filename
            path.write_bytes(filename.encode()+b"own tiny RGB fixture")
            images.append(dict(scene_id=scene, frame_id=frame, file=filename, sha256=gate.sha256(path), width=640, height=480))
    manifest = dict(schema=gate.SCHEMA, revision=gate.REVISION, license=gate.LICENSE, selection=gate.SELECTION, images=images)
    (folder/"manifest.json").write_text(json.dumps(manifest))
    return folder, manifest


def test_exact_nine_public_original_scene_frame_order(gate, tmp_path):
    folder, manifest = inputs(gate, tmp_path); records, receipt = gate.public_inputs(folder)
    assert [(r["scene_id"], r["frame_id"]) for r in records] == [(s, f) for s, fs in enumerate(gate.FRAME_IDS, 1) for f in fs]
    assert [{k: v for k, v in r.items() if k != "path"} for r in records] == manifest["images"]
    assert all(r["path"] == folder/r["file"] for r in records)
    assert receipt == gate.helper.identity(folder/"manifest.json")


@pytest.mark.parametrize("fault", ["private", "record_private", "revision", "license", "selection", "schema", "order", "count",
    "scene_bool", "frame_bool", "width_bool", "grid", "filename", "sha", "sha_type", "empty", "missing", "extra", "symlink", "manifest_symlink"])
def test_public_input_scope_fail_closed(gate, tmp_path, fault):
    folder, manifest = inputs(gate, tmp_path); path = folder/"manifest.json"; first = manifest["images"][0]
    if fault == "private": manifest["K"] = []
    elif fault == "record_private": first["depth"] = "private.png"
    elif fault in ("revision", "license", "selection", "schema"): manifest[fault] = "wrong"
    elif fault == "order": manifest["images"][0], manifest["images"][1] = manifest["images"][1], manifest["images"][0]
    elif fault == "count": manifest["images"].pop()
    elif fault == "scene_bool": first["scene_id"] = True
    elif fault == "frame_bool": first["frame_id"] = False
    elif fault == "width_bool": first["width"] = True
    elif fault == "grid": first["height"] = 384
    elif fault == "filename": first["file"] = "../outside.png"
    elif fault == "sha": first["sha256"] = "a"*64
    elif fault == "sha_type": first["sha256"] = 1
    elif fault == "empty":
        (folder/first["file"]).write_bytes(b""); first["sha256"] = gate.sha256(folder/first["file"])
    elif fault == "missing": (folder/first["file"]).unlink()
    elif fault == "extra": (folder/"camera.json").write_text("{}")
    elif fault == "symlink":
        p = folder/first["file"]; saved = tmp_path/"outside"; p.rename(saved); p.symlink_to(saved)
    path.write_text(json.dumps(manifest))
    if fault == "manifest_symlink":
        saved = tmp_path/"outside_manifest"; path.rename(saved); path.symlink_to(saved)
    with pytest.raises(ValueError): gate.public_inputs(folder)


@pytest.mark.parametrize("focal", [800., 580., 23., 12000.])
def test_native_camera_pixel_focal_not_clipped_or_scaled(gate, focal):
    K = np.array([[focal, 0., 320.], [0., focal, 240.], [0., 0., 1.]])
    assert np.allclose(gate.camera_candidate(np.diag([1/640, 1/480, 1.])@K), K)


@pytest.mark.parametrize("fault", ["zero", "negative", "nan", "offcenter", "anisotropic", "skew", "projective", "shape", "dtype", "masked"])
def test_camera_candidate_rejects_invalid_not_repairs(gate, fault):
    K = np.diag([1/640, 1/480, 1.])@gate.FIXED_K
    if fault == "zero": K[0, 0] = 0
    elif fault == "negative": K[0, 0] = -1
    elif fault == "nan": K[0, 0] = np.nan
    elif fault == "offcenter": K[0, 2] += .01
    elif fault == "anisotropic": K[1, 1] *= 1.1
    elif fault == "skew": K[0, 1] = .01
    elif fault == "projective": K[2, 0] = .01
    elif fault == "shape": K = K[0]
    elif fault == "dtype": K = K.astype(int)
    else: K = np.ma.array(K, mask=False)
    with pytest.raises(ValueError): gate.camera_candidate(K)


def test_all_three_frames_per_scene_median_no_oracle_no_clamp(gate):
    focals = np.array([1500., 1300., 1700., 9., 7., 11., 700., 650., 750.]); original = focals.copy()
    cameras = gate.scene_cameras(focals)
    assert np.array_equal(cameras[:, 0, 0], [1500., 9., 700.])
    assert np.array_equal(cameras[:, 1, 1], cameras[:, 0, 0])
    assert np.array_equal(cameras[:, :2, 2], [[320., 240.]]*3) and np.array_equal(focals, original)
    for bad in (focals[:8], focals.astype(int), np.full(9, np.nan), np.zeros(9), np.ma.array(focals)):
        with pytest.raises(ValueError): gate.scene_cameras(bad)


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
    calls = []; sentinel = object()
    def original(p, m, *, focal, downsample_size):
        calls.append((p, m, focal, downsample_size)); return sentinel
    module = SimpleNamespace(recover_focal_shift=original)
    def infer(image, *, fov_x, apply_mask, force_projection):
        assert apply_mask is False and force_projection is True
        return module.recover_focal_shift(points, mask, focal=None if fov_x is None else 1.)
    return torch, module, SimpleNamespace(infer=infer), points, mask, calls, original, sentinel


def test_all27_solver_calls_have_declared_order_and_return_original(gate, monkeypatch):
    torch, module, net, points, _, calls, original, sentinel = solver_fixture(gate, monkeypatch); diagnostics = []
    for phase in ("fixed_prior", "camera_candidate", "shared_camera"):
        for i in range(9):
            result = gate.checked_infer(torch, module, net, points, None if phase == "camera_candidate" else 45., diagnostics,
                                        {"pass": phase, "file_index": i})
            assert result is sentinel and module.recover_focal_shift is original
    assert len(calls) == len(diagnostics) == 27
    assert [(r["pass"], r["file_index"]) for r in diagnostics] == [(p, i) for p in ("fixed_prior", "camera_candidate", "shared_camera") for i in range(9)]
    assert [r["focal_prior_supplied"] for r in diagnostics] == [True]*9+[False]*9+[True]*9
    assert all(r["native_nearest64_valid_pixels"] == 2 and r["original_solver_returned"] is True for r in diagnostics)
    assert all(c[0] is points and c[3] == (64, 64) for c in calls)


def test_native_nearest64_support_not_fullgrid_count(gate, monkeypatch):
    torch, module, net, points, mask, calls, original, _ = solver_fixture(gate, monkeypatch)
    mask[:] = False; mask[0, 1::2, 1::2] = True; mask[0, 0, 0] = True; diagnostics = []
    with pytest.raises(ValueError, match="default-camera fallback"):
        gate.checked_infer(torch, module, net, points, None, diagnostics, {})
    assert not calls and module.recover_focal_shift is original
    assert diagnostics == [{"native_nearest64_valid_pixels": 1, "focal_prior_supplied": False, "original_solver_returned": False}]


@pytest.mark.parametrize("fault", ["mask_dtype", "mask_shape", "points_shape", "backend", "no_solver", "two_calls", "sampling"])
def test_solver_failures_restore_hook_and_propagate(gate, monkeypatch, fault):
    torch, module, net, points, mask, _, original, _ = solver_fixture(gate, monkeypatch)
    if fault == "mask_dtype": net.infer = lambda *a, **k: module.recover_focal_shift(points, mask.astype(float))
    elif fault == "mask_shape": net.infer = lambda *a, **k: module.recover_focal_shift(points, mask[0])
    elif fault == "points_shape": net.infer = lambda *a, **k: module.recover_focal_shift(points[0], mask)
    elif fault == "sampling": net.infer = lambda *a, **k: module.recover_focal_shift(points, mask, downsample_size=(32, 32))
    elif fault == "no_solver": net.infer = lambda *a, **k: None
    elif fault == "two_calls":
        def twice(*a, **k):
            module.recover_focal_shift(points, mask); return module.recover_focal_shift(points, mask)
        net.infer = twice
    else:
        def fail(*a, **k): raise RuntimeError("actual backend failure")
        net.infer = fail
    with pytest.raises(RuntimeError): gate.checked_infer(torch, module, net, points, None, [], {})
    assert module.recover_focal_shift is original


def tiny_arrays(gate, monkeypatch):
    monkeypatch.setattr(gate, "HEIGHT", 6); monkeypatch.setattr(gate, "WIDTH", 8)
    fixed_K = np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]])
    learned_K = np.array([[11., 0., 4.], [0., 11., 3.], [0., 0., 1.]])
    monkeypatch.setattr(gate, "FIXED_K", fixed_K)
    def arrays(K):
        y, x = np.mgrid[:6, :8]; depth = np.full((6, 8), 2., np.float32)
        points = np.stack(((x+.5-4)/K[0, 0]*depth, (y+.5-3)/K[1, 1]*depth, depth), -1).astype(np.float32)
        return dict(points=points, depth=depth, validity=np.ones((6, 8), bool), K=K)
    return gate.prediction_arrays(arrays(fixed_K), arrays(learned_K), dict(scene_id=1, frame_id=0)), learned_K


def test_cpu_npz_contract_original_dtypes_rays_and_no_mutation(gate, monkeypatch, tmp_path):
    data, learned = tiny_arrays(gate, monkeypatch); before = {k: v.copy() for k, v in data.items()}
    checks = gate.validate_prediction_arrays(data, 1, 0, learned)
    assert set(checks) == {"fixed", "learned"} and all(c["status"] == "pass" for c in checks.values())
    assert all(np.array_equal(data[k], v) for k, v in before.items())
    path = tmp_path/"tiny.npz"; np.savez_compressed(path, **data)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(data); gate.validate_prediction_arrays(archive, 1, 0, learned)


@pytest.mark.parametrize("fault", ["extra", "missing", "scene", "frame", "scalar_dtype", "scalar_shape", "depth_dtype", "points_dtype",
    "validity_dtype", "shape", "camera", "ray", "Z", "nan", "no_valid", "masked", "camera_masked"])
def test_prediction_contract_fail_closed(gate, monkeypatch, fault):
    data, learned = tiny_arrays(gate, monkeypatch)
    if fault == "extra": data["truth"] = np.array(0)
    elif fault == "missing": del data["fixed_depth"]
    elif fault == "scene": data["scene_id"] = np.array(2, np.int64)
    elif fault == "frame": data["frame_id"] = np.array(3, np.int64)
    elif fault == "scalar_dtype": data["scene_id"] = np.array(1, np.int32)
    elif fault == "scalar_shape": data["scene_id"] = np.array([1], np.int64)
    elif fault == "depth_dtype": data["fixed_depth"] = data["fixed_depth"].astype(np.float64)
    elif fault == "points_dtype": data["fixed_points"] = data["fixed_points"].astype(np.float64)
    elif fault == "validity_dtype": data["fixed_validity"] = data["fixed_validity"].astype(float)
    elif fault == "shape": data["fixed_points"] = data["fixed_points"][0]
    elif fault == "camera": data["learned_K"][0, 0] *= 1.1
    elif fault == "ray": data["fixed_points"][0, 0, 0] += .1
    elif fault == "Z": data["fixed_points"][0, 0, 2] *= 1.1
    elif fault == "nan": data["fixed_points"][0, 0] = np.nan
    elif fault == "no_valid": data["fixed_validity"][:] = False
    elif fault == "masked": data["fixed_depth"] = np.ma.array(data["fixed_depth"], mask=False)
    else: learned = np.ma.array(learned, mask=False)
    with pytest.raises(ValueError): gate.validate_prediction_arrays(data, 1, 0, learned)


def test_invalid_excluded_geometry_preserved_never_filled(gate, monkeypatch):
    data, learned = tiny_arrays(gate, monkeypatch)
    for mode in ("fixed", "learned"):
        data[mode+"_validity"][0] = False; data[mode+"_points"][0] = np.nan; data[mode+"_depth"][0] = -np.inf
    checks = gate.validate_prediction_arrays(data, 1, 0, learned)
    assert all(c["excluded_pixels"] == 8 for c in checks.values())
    assert all(np.isnan(data[m+"_points"][0]).all() and np.isneginf(data[m+"_depth"][0]).all() for m in ("fixed", "learned"))


def test_early_failure_is_frozen_before_heavy_import(gate, tmp_path, monkeypatch):
    output = tmp_path/"validation/tudl_rgb_v1/predictions_v1"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda p: [Path("lo")] if str(p) == "/sys/class/net" else original(p))
    with pytest.raises(ValueError, match="regular canonical"): gate.main([])
    path = output/"report.json"; frozen = path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["MoGe_calls_completed"] == 0 and report["private_truth_read"] is False
    assert report["budget_seconds"] == 600 and report["network"] == "none"
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_wrapper_scope_and_unchanged_native_flags(gate):
    wrapper = Path(gate.__file__).with_name("run_tudl_infer.sh").read_text(); source = Path(gate.__file__).read_text()
    assert "603s docker run" in wrapper and "--network none" in wrapper and "--memory 32g --cpus 4" in wrapper
    assert '--entrypoint python' in wrapper and 'chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"' in wrapper
    assert "eval_private" not in wrapper and "chown -R" not in wrapper and "src=$BASE,dst=$BASE" not in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor", "validation,dst"))
    assert wrapper.count("hub/blobs/9f/") == 2 and "weights-acquisition.json" in wrapper
    assert "apply_mask=False, force_projection=True" in source and "GEOMETRY_SHA" in source
    with pytest.raises(SystemExit): gate.main(["--episode", "0"])

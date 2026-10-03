"""Small pure camera/ratio fixtures and fake Torch ABI; no native inference."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src")); monkeypatch.syspath_prepend(str(root / "infra"))
    spec = importlib.util.spec_from_file_location("wr_rgbd_anchor_test", root / "infra/rgbd_anchor_contract.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module); spec.loader.exec_module(module)
    return module


def grid(contract, width=100, height=100, focal=800.):
    return contract.Grid(width, height, [[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])


def baseline(g, depth_value=2., invalid=False):
    yy, xx = np.indices(g.shape, dtype=np.float64); k = g.K
    depth = np.full(g.shape, depth_value, np.float32)
    points = np.stack(((xx + .5 - k[0, 2]) / k[0, 0] * depth,
        (yy + .5 - k[1, 2]) / k[1, 1] * depth, depth), axis=-1).astype(np.float32)
    valid = np.ones(g.shape, np.bool_)
    if invalid: valid[30:32, 30:32] = False; depth[~valid] = np.nan; points[~valid] = np.nan
    return dict(depth=depth, points=points, validity=valid, K=g.K)


def test_grid_immutable_input_copy_and_border_fullnative_stop(contract):
    camera = np.array([[800., 0., 50.], [0., 800., 50.], [0., 0., 1.]])
    g = contract.Grid(100, 100, camera); camera[0, 0] = 3.
    assert g.K[0, 0] == 800.
    returned = g.K; returned[0, 0] = 1.; assert g.K[0, 0] == 800.
    assert contract.border_mask(g).sum() == 3600
    full = grid(contract, 1536, 1152, 1920.)
    assert np.allclose(contract.expected_processed_camera(full), [[647.5, 0., 259.], [0., 653.3333333333, 196.], [0., 0., 1.]])
    with pytest.raises(ValueError, match="divisible"): contract.border_mask(full)
    tiny = grid(contract, 20, 20)
    with pytest.raises(ValueError, match="support"):
        contract.border_ratio(tiny, np.ones(tiny.shape, np.float32), np.ones(tiny.shape, np.bool_), np.ones(tiny.shape, np.float32))


@pytest.mark.parametrize("fault", ["bool", "size", "center", "anisotropic", "negative", "nan", "masked"])
def test_camera_grid_invalid_no_private_calibration_fallback(contract, fault):
    width = True if fault == "bool" else 100; height = 1 if fault == "size" else 100
    camera = np.array([[800., 0., 50.], [0., 800., 50.], [0., 0., 1.]])
    if fault == "center": camera[0, 2] = 49.
    elif fault == "anisotropic": camera[1, 1] = 810.
    elif fault == "negative": camera[0, 0] = -1.
    elif fault == "nan": camera[0, 0] = np.nan
    elif fault == "masked": camera = np.ma.array(camera)
    with pytest.raises(ValueError): contract.Grid(width, height, camera)


def test_ratio_support_and_median4_explicit_caller_order(contract):
    g = grid(contract); raw = np.ones(g.shape, np.float32); diagnostics = []
    order = [(7, 4), (7, 9), (7, 20), (7, 32), (2, 8), (2, 15), (2, 40), (2, 60)]
    for pair, value in zip(order, [1., 3., 8., 20., 2., 4., 6., 10.]):
        data = baseline(g, value); row = contract.border_ratio(g, data["depth"], data["validity"], raw)
        row.update(scene_id=pair[0], frame_id=pair[1]); diagnostics.append(row)
    assert contract.scene_anchors(g, diagnostics, order) == {7: 5.5, 2: 5.}
    invalid = np.ones(g.shape, np.bool_); ids = np.flatnonzero(contract.border_mask(g)); invalid.reshape(-1)[ids[3400:]] = False
    with pytest.raises(ValueError, match="support"): contract.border_ratio(g, raw, invalid, raw)
    with pytest.raises(ValueError): contract.scene_anchors(g, diagnostics, list(reversed(order)))
    with pytest.raises(ValueError): contract.scene_anchors(g, diagnostics[:-1], order)
    with pytest.raises(ValueError): contract.scene_anchors(g, diagnostics, order[:4] * 2)


def test_candidate_exact11arrays_invalids_preserved_and_byte_replay(contract):
    g = grid(contract); original = baseline(g, invalid=True); before = contract.array_identities(original)
    raw = np.full(g.shape, 1.5, np.float32)
    data = contract.candidate_arrays(g, original, raw, 2., 7, 4)
    assert set(data) == contract.ARRAY_KEYS and len(data) == 11
    assert data["moge_depth"].tobytes() == original["depth"].tobytes()
    assert data["moge_points"].tobytes() == original["points"].tobytes()
    assert contract.array_identities(original) == before
    assert np.array_equal(data["anchored_validity"], original["validity"])
    assert np.all(data["anchored_depth"] == 3.) and np.isfinite(data["anchored_points"]).all()
    checks = contract.validate_prediction_arrays(g, data, 7, 4, 2.)
    assert checks["moge"]["excluded_pixels"] == 4 and checks["anchored"]["valid_pixels"] == 10000


@pytest.mark.parametrize("fault", ["drop", "expand", "offset", "tiny_ray", "dtype", "anchor", "id", "raw_nan", "baseline_ray", "extra"])
def test_replay_rejects_dropping_offsets_wrongrays_and_changed_ids(contract, fault):
    g = grid(contract); data = contract.candidate_arrays(g, baseline(g, invalid=True), np.ones(g.shape, np.float32), 2., 7, 4)
    if fault == "drop": data["anchored_validity"][0, 0] = False
    elif fault == "expand": data["anchored_validity"][30, 30] = True
    elif fault == "offset": data["anchored_depth"] += .1
    elif fault == "tiny_ray": data["anchored_points"][0, 0, 0] = np.nextafter(data["anchored_points"][0, 0, 0], np.float32(np.inf))
    elif fault == "dtype": data["da3_depth"] = data["da3_depth"].astype(np.float64)
    elif fault == "anchor": data["scene_anchor"] = np.array(3., np.float64)
    elif fault == "id": data["scene_id"] = np.array(8, np.int64)
    elif fault == "raw_nan": data["da3_depth"][40, 40] = np.nan
    elif fault == "baseline_ray": data["moge_points"][0, 0, 0] += .1
    else: data["private_K"] = np.eye(3)
    with pytest.raises(ValueError): contract.validate_prediction_arrays(g, data, 7, 4, 2.)


@pytest.mark.parametrize("coefficient", [True, 2, 0., -1., float("nan"), 1e100, 1e-100])
def test_coefficient_types_and_numeric_saturation_fail_closed(contract, coefficient):
    g = grid(contract)
    with pytest.raises(ValueError): contract.candidate_arrays(g, baseline(g), np.ones(g.shape, np.float32), coefficient, 7, 4)


def test_da3_camera_generic720_not_old640_constant(contract):
    g = grid(contract, 720, 540); expected = contract.expected_processed_camera(g)
    assert np.allclose(expected, [[575.5555555556, 0., 259.], [0., 580.7407407407, 196.], [0., 0., 1.]])
    factor = contract.metric_factor(g, expected.astype(np.float32))
    assert factor == (float(np.float32(expected[0, 0])) + float(np.float32(expected[1, 1]))) / 600
    with pytest.raises(ValueError): contract.metric_factor(g, contract.expected_processed_camera(grid(contract, 640, 480)))


def test_tudl_pure_regression_arrays_ratio_and_aggregate_byte_identical(contract):
    import tudl_anchor_infer as old
    g = grid(contract, 640, 480); source = baseline(g, invalid=True); raw = np.full(g.shape, 1.4, np.float32)
    new = contract.candidate_arrays(g, source, raw, 1.23, 1, 1788); prior = old.candidate_arrays(source, raw, 1.23, 1, 1788)
    assert contract.array_identities(new) == old.array_identities(prior)
    assert contract.border_ratio(g, source["depth"], source["validity"], raw) == old.border_ratio(source["depth"], source["validity"], raw)
    order = [(s, f) for s, frames in enumerate(old.inputs.FRAME_IDS, 1) for f in frames]; rows = []
    for scene, frame in order:
        row = contract.border_ratio(g, source["depth"], source["validity"], raw); row.update(scene_id=scene, frame_id=frame); rows.append(row)
    assert list(contract.scene_anchors(g, rows, order).values()) == old.scene_anchors(rows)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value); self.shape = self.value.shape; self.dtype = self.value.dtype
    def __getitem__(self, index): return Tensor(self.value[index])
    def float(self): return Tensor(self.value.astype(np.float32))
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.value, axis))
    def squeeze(self, axis): return Tensor(np.squeeze(self.value, axis))
    def __gt__(self, other): return Tensor(self.value > other)
    def sum(self): return Tensor(self.value.sum())
    def item(self): return self.value.item()
    def detach(self): return self
    def cpu(self): return self
    def cuda(self): return self
    def numpy(self): return self.value
    def __mul__(self, other): return Tensor(self.value * other)


class InferenceMode:
    def __enter__(self): return None
    def __exit__(self, *_): return False


def fake_torch(sample_count=4096):
    seen = []
    def interpolate(tensor, size, mode, **kwargs):
        seen.append((size, mode, kwargs))
        if mode == "nearest":
            values = np.zeros((1, 1, 64, 64), np.float32); values.reshape(-1)[:sample_count] = 1.; return Tensor(values)
        return Tensor(np.full((1, 1, *size), tensor.value.flat[0], np.float32))
    return SimpleNamespace(float32=np.dtype(np.float32), bool=np.dtype(np.bool_), is_tensor=lambda value: isinstance(value, Tensor),
        inference_mode=InferenceMode, nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)),
        from_numpy=Tensor), seen


@pytest.mark.parametrize("fault", [None, "fallback", "wrong_alias", "no_callback", "callback_exception", "wrong_camera"])
def test_checked_native_moge_exact_call_alias_restore_and_fullgrid_camera(contract, fault):
    g = grid(contract); torch, seen = fake_torch(1 if fault == "fallback" else 4096); calls = []
    def original(points, mask, **kwargs): calls.append(kwargs); return "original result"
    module = SimpleNamespace(recover_focal_shift=original)
    data = baseline(g); normalized = np.diag([.01, .01, 1.]) @ g.K
    class Network:
        def infer(self, tensor, **kwargs):
            assert tensor.shape == (1, 3, 100, 100)
            assert kwargs == dict(fov_x=float(np.degrees(2 * np.arctan(100 / 1600))), force_projection=True, apply_mask=False)
            if fault != "no_callback":
                module.recover_focal_shift(Tensor(data["points"][None]), Tensor(data["validity"][None]), focal=800., downsample_size=(64, 64))
            if fault == "callback_exception": raise RuntimeError("native exception")
            camera = normalized.copy()
            if fault == "wrong_camera": camera[0, 0] *= 2.
            return {key: Tensor(value[None]) for key, value in dict(depth=data["depth"], points=data["points"], mask=data["validity"], intrinsics=camera).items()}
    alias = (lambda *_: None) if fault == "wrong_alias" else original; diagnostics = []
    if fault is None:
        result, checks = contract.checked_moge_infer(torch, g, module, alias, Network(), Tensor(np.zeros((3, 100, 100), np.float32)), diagnostics, {"frame_id": 4})
        assert checks["valid_pixels"] == 10000 and diagnostics[0]["original_solver_returned"] is True
        assert calls == [dict(focal=800., downsample_size=(64, 64))] and seen == [((64, 64), "nearest", {})]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            contract.checked_moge_infer(torch, g, module, alias, Network(), Tensor(np.zeros((3, 100, 100), np.float32)), diagnostics, {})
    assert module.recover_focal_shift is original


def test_native_da3_resize_once_original_grid_and_bilinear_arguments(contract):
    torch, seen = fake_torch(); g = grid(contract, 720, 540)
    native = np.ones((1, 392, 518), np.float32); camera = contract.expected_processed_camera(g)
    result = contract.resize_metric_depth(torch, g, native, camera)
    assert result.shape == (1, 1, 540, 720)
    assert seen == [((540, 720), "bilinear", {"align_corners": False})]
    assert np.all(result.value == np.float32(contract.metric_factor(g, camera)))

"""Data-free source parity and native callback support; no Torch/model import."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_depth_camera_support", infra/"depth_camera_support.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_frozen_original_function_ast_and_geometry_pin_identical(module):
    infra = Path(module.__file__).parent
    trees = [ast.parse((infra/name).read_text()) for name in ("joint_rgb_infer.py", "depth_camera_support.py")]
    functions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "checked_depth_infer") for tree in trees]
    assert ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False)
    pin = next(n.value.value for n in trees[0].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GEOMETRY_SHA" for t in n.targets))
    assert module.GEOMETRY_SHA == pin
    assert (module.WIDTH, module.HEIGHT) == (1024, 768)


class Tensor:
    def __init__(self, values): self.values = np.asarray(values)
    @property
    def shape(self): return self.values.shape
    @property
    def dtype(self): return self.values.dtype
    def float(self): return Tensor(self.values.astype(np.float32))
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.values, axis))
    def squeeze(self, axis): return Tensor(np.squeeze(self.values, axis))
    def sum(self): return Tensor(self.values.sum())
    def item(self): return self.values.item()
    def __gt__(self, value): return Tensor(self.values > value)
    def __getitem__(self, index): return Tensor(self.values[index])


@pytest.fixture
def native(module, monkeypatch):
    monkeypatch.setattr(module, "WIDTH", 8); monkeypatch.setattr(module, "HEIGHT", 6)
    state = SimpleNamespace(count=2, fault=None, calls=[], interpolations=[])
    points = Tensor(np.zeros((1, 6, 8, 3), np.float32)); mask = Tensor(np.ones((1, 6, 8), bool))
    output = {"native_points": points}; solver_result = object()
    def interpolate(value, size, mode):
        state.interpolations.append((value.shape, value.dtype, size, mode))
        sampled = np.zeros((1, 1, 64, 64), np.float32); sampled.ravel()[:state.count] = 1.
        return Tensor(sampled)
    def original(p, m, *, focal, downsample_size):
        state.calls.append((p, m, focal, downsample_size))
        if state.fault == "solver": raise RuntimeError("native solver failed")
        return solver_result
    native_module = SimpleNamespace(recover_focal_shift=original)
    def infer(value, **kwargs):
        assert value.shape == (1, 3, 6, 8) and kwargs == {"fov_x": 50., "apply_mask": False}
        if state.fault == "network": raise RuntimeError("native network failed")
        actual_mask = None if state.fault == "none_mask" else mask
        if state.fault == "mask_dtype": actual_mask = Tensor(mask.values.astype(np.float32))
        if state.fault == "mask_shape": actual_mask = Tensor(np.ones((6, 8), bool))
        actual_points = Tensor(np.zeros((1, 6, 8, 2), np.float32)) if state.fault == "points_shape" else points
        size = (32, 32) if state.fault == "sampling" else (64, 64)
        for _ in range(0 if state.fault == "zero_calls" else 2 if state.fault == "two_calls" else 1):
            result = native_module.recover_focal_shift(actual_points, actual_mask, focal=11., downsample_size=size)
            assert result is solver_result
        return output
    torch = SimpleNamespace(bool=np.dtype(bool), is_tensor=lambda v: isinstance(v, Tensor), inference_mode=nullcontext,
                            nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)))
    return SimpleNamespace(state=state, torch=torch, module=native_module, network=SimpleNamespace(infer=infer),
                           tensor=Tensor(np.zeros((3, 6, 8), np.float32)), original=original, output=output, points=points, mask=mask)


def test_original_solver_arguments_results_and_one_call_preserved(module, native):
    diagnostics = [{"prior_entry": True}]
    result = module.checked_depth_infer(native.torch, native.module, native.network, native.tensor, 50., diagnostics, {"frame": 4})
    assert result is native.output and native.module.recover_focal_shift is native.original
    assert native.state.calls == [(native.points, native.mask, 11., (64, 64))]
    assert native.state.interpolations == [((1, 1, 6, 8), np.dtype(np.float32), (64, 64), "nearest")]
    assert diagnostics == [{"prior_entry": True}, {"frame": 4, "native_nearest64_valid_pixels": 2,
                                                  "focal_prior_supplied": True, "original_solver_returned": True}]


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_sample_support_never_calls_default_camera_solver(module, native, count):
    native.state.count = count; diagnostics = []
    with pytest.raises(ValueError, match="default-camera fallback"):
        module.checked_depth_infer(native.torch, native.module, native.network, native.tensor, 50., diagnostics, {})
    assert native.state.calls == [] and not diagnostics[0]["original_solver_returned"]
    assert native.module.recover_focal_shift is native.original


@pytest.mark.parametrize("fault", ["none_mask", "mask_dtype", "mask_shape", "points_shape", "sampling", "zero_calls", "two_calls", "solver", "network"])
def test_native_abi_call_count_or_failure_restores_instrumentation(module, native, fault):
    native.state.fault = fault
    with pytest.raises(RuntimeError):
        module.checked_depth_infer(native.torch, native.module, native.network, native.tensor, 50., [], {})
    assert native.module.recover_focal_shift is native.original

"""Data-free identity-composition tests: fake Torch, actual NumPy, no native inference."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("cari96_native_test", ROOT / "infra/cari96_native.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


NP_TORCH = SimpleNamespace(is_tensor=lambda value: False)


def inputs(scale_delta=True):
    batch = {
        "mhr_shape_init": np.broadcast_to(np.arange(45, dtype=np.float32) / 100, (1, 96, 45)).copy(),
        "mhr_scale_init": np.broadcast_to(np.arange(28, dtype=np.float32) / 100, (1, 96, 28)).copy(),
        "mhr_body_pose_cont_init": np.zeros((1, 96, 260), np.float32),
    }
    pred = dict(delta_mhr_shape=np.full((1, 96, 45), .03, np.float32),
                delta_mhr_body_pose_cont=np.full((1, 96, 260), .1, np.float32),
                delta_mhr_global_rot6d=np.full((1, 96, 6), .02, np.float32),
                delta_mhr_hand=np.full((1, 96, 108), .04, np.float32),
                rot=np.full((1, 96, 6), .2, np.float32), trans=np.full((1, 96, 3), .3, np.float32))
    if scale_delta: pred["delta_mhr_scale"] = np.full((1, 96, 28), .04, np.float32)
    return pred, batch


def compose(pred, batch):
    result = {key[:-5]:value + pred.get("delta_" + key[:-5], np.zeros_like(value)) for key,value in batch.items() if key.endswith("_init")}
    result.update(rot=pred["rot"], trans=pred["trans"], hand=pred["delta_mhr_hand"], root=pred["delta_mhr_global_rot6d"])
    return result


@pytest.mark.parametrize("scale_delta", [False, True])
def test_only_present_identity_deltas_zeroed_original_raw_and_all_other_references_untouched(gate, scale_delta):
    pred, batch = inputs(scale_delta); before = {key:value.tobytes() for key,value in pred.items()}; seen = []; report = {}
    def delegate(effective, original_batch):
        assert effective is not pred and original_batch is batch
        for key,value in pred.items():
            if key in ("delta_mhr_shape", "delta_mhr_scale"):
                assert not np.shares_memory(effective[key], value) and not np.any(effective[key])
            else: assert effective[key] is value
        seen.append(effective)
        return compose(effective, original_batch)
    output = gate.constrain_composition(NP_TORCH, delegate, pred, batch, report)
    assert len(seen) == 1 and ("delta_mhr_scale" in seen[0]) == scale_delta
    for key in gate.IDENTITY_DIMS: assert output[key].tobytes() == batch[key + "_init"].tobytes()
    assert output["rot"] is pred["rot"] and output["trans"] is pred["trans"]
    assert output["hand"] is pred["delta_mhr_hand"] and output["root"] is pred["delta_mhr_global_rot6d"]
    assert np.array_equal(output["mhr_body_pose_cont"], pred["delta_mhr_body_pose_cont"])
    assert {key:value.tobytes() for key,value in pred.items()} == before
    assert all(report[name] == 1 for name in gate.COUNTERS)
    assert report["raw_prediction_bytes_preserved"] and not report["original_network_modified"]
    assert not report["unchanged_original_CARI_method"] and not report["quality_verified"]


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.data
    def data_ptr(self): return self.data.ctypes.data


def test_fake_torch_zeros_like_keeps_dtype_shape_without_tensor_alias(gate):
    pred, batch = inputs(); pred = {k:Tensor(v) for k,v in pred.items()}; batch = {k:Tensor(v) for k,v in batch.items()}
    torch = SimpleNamespace(is_tensor=lambda v:isinstance(v, Tensor), zeros_like=lambda v:Tensor(np.zeros_like(v.data)))
    def delegate(effective, original):
        assert effective["delta_mhr_shape"].data_ptr() != pred["delta_mhr_shape"].data_ptr()
        assert not effective["delta_mhr_scale"].data.any()
        return {"mhr_shape":Tensor(original["mhr_shape_init"].data + effective["delta_mhr_shape"].data),
                "mhr_scale":Tensor(original["mhr_scale_init"].data + effective["delta_mhr_scale"].data)}
    output = gate.constrain_composition(torch, delegate, pred, batch, {})
    assert output["mhr_shape"].data.dtype == np.float32


@pytest.mark.parametrize("fault", ["missing_shape", "missing_init", "short_window", "multiple_batch", "wrong_shape_dim", "wrong_scale_dim",
                                  "f64", "masked", "nan", "variable_shape", "variable_scale", "delta_f64", "delta_nan"])
def test_native_identity_contract_fails_before_original_delegate(gate, fault):
    pred, batch = inputs(); calls = []
    if fault == "missing_shape": pred.pop("delta_mhr_shape")
    elif fault == "missing_init": batch.pop("mhr_scale_init")
    elif fault == "short_window": batch["mhr_shape_init"] = batch["mhr_shape_init"][:, :95]
    elif fault == "multiple_batch": batch["mhr_shape_init"] = np.tile(batch["mhr_shape_init"], (2, 1, 1))
    elif fault == "wrong_shape_dim": pred["delta_mhr_shape"] = np.zeros((1, 96, 44), np.float32)
    elif fault == "wrong_scale_dim": pred["delta_mhr_scale"] = np.zeros((1, 96, 68), np.float32)
    elif fault == "f64": batch["mhr_shape_init"] = batch["mhr_shape_init"].astype(np.float64)
    elif fault == "masked": pred["delta_mhr_shape"] = np.ma.array(pred["delta_mhr_shape"], mask=False)
    elif fault == "nan": batch["mhr_scale_init"][0, 0, 0] = np.nan
    elif fault == "variable_shape": batch["mhr_shape_init"][0, 5, 0] += .1
    elif fault == "variable_scale": batch["mhr_scale_init"][0, 8, 0] += .1
    elif fault == "delta_f64": pred["delta_mhr_scale"] = pred["delta_mhr_scale"].astype(np.float64)
    elif fault == "delta_nan": pred["delta_mhr_shape"][0, 0, 0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        gate.constrain_composition(NP_TORCH, lambda *args:calls.append(args), pred, batch, {})
    assert not calls


@pytest.mark.parametrize("fault", ["raw_array", "raw_key", "raw_reference", "initializer", "output_identity", "output_missing", "raises"])
def test_delegate_failure_or_mutation_never_marked_verified(gate, fault):
    pred, batch = inputs(); report = {}
    def delegate(effective, original):
        output = compose(effective, original)
        if fault == "raw_array": effective["trans"][0, 0, 0] += .1
        elif fault == "raw_key": pred["new"] = np.zeros((1,), np.float32)
        elif fault == "raw_reference": pred["rot"] = pred["rot"].copy()
        elif fault == "initializer": original["mhr_body_pose_cont_init"][0, 0, 0] = .1
        elif fault == "output_identity": output["mhr_shape"][0, 0, 0] = .1
        elif fault == "output_missing": output.pop("mhr_scale")
        else: raise RuntimeError("native compose failed")
        return output
    with pytest.raises((ValueError, RuntimeError)):
        gate.constrain_composition(NP_TORCH, delegate, pred, batch, report)
    assert report["composition_hook_calls"] == report["composition_delegate_calls"] == 1
    assert report.get("composition_verified_calls", 0) == 0


def clone_fixture(gate, monkeypatch):
    """Explicit test-only pinned toy module; production6020 hash remains checked separately."""
    source = b'''from __future__ import annotations
def run_mhr_wild_inference(prediction, batch, *, stride=96):
    result = compose_mhr_output(prediction, batch)
    return result, prediction, object_pose_from_relative(prediction), stride
'''
    filename = "/own/7c0d/tools/run_mhr_wild_inference.py"
    composition_source = b'def compose_mhr_output(prediction, batch):\n    return implementation(prediction, batch)\n'
    delegate_globals = {"implementation":compose}
    exec(compile(composition_source, "/own/7c0d/learning/training/mhr_losses.py", "exec"), delegate_globals)
    globals_ = dict(__file__=filename, compose_mhr_output=delegate_globals["compose_mhr_output"],
                    object_pose_from_relative=lambda prediction:prediction["rot"])
    exec(compile(source, filename, "exec"), globals_)
    monkeypatch.setattr(gate, "NATIVE_SOURCE_BYTES", len(source)); monkeypatch.setattr(gate, "NATIVE_SOURCE_SHA256", hashlib.sha256(source).hexdigest())
    return globals_["run_mhr_wild_inference"], source


def test_functiontype_clone_same_bytecode_defaults_only_composer_global_differs(gate, monkeypatch):
    original, source = clone_fixture(gate, monkeypatch); report = {}; globals_before = dict(original.__globals__)
    clone = gate.clone_native_forward(NP_TORCH, original, report, source_bytes=source)
    assert clone is not original and clone.__code__ is original.__code__ and clone.__defaults__ is original.__defaults__
    assert clone.__kwdefaults__ == original.__kwdefaults__ and clone.__kwdefaults__ is not original.__kwdefaults__
    assert clone.__globals__ is not original.__globals__
    assert [key for key in original.__globals__ if clone.__globals__[key] is not original.__globals__[key]] == ["compose_mhr_output"]
    pred, batch = inputs(False); before = {key:value.tobytes() for key,value in pred.items()}
    output, raw, object_pose, stride = clone(pred, batch)
    assert raw is pred and object_pose is pred["rot"] and stride == 96
    assert {key:value.tobytes() for key,value in pred.items()} == before
    assert original.__globals__ == globals_before and report["native_compose_global_unchanged"]
    assert report["overridden_globals"] == ["compose_mhr_output"] and not report["native_global_replacement_performed"]
    assert report["composition_verified_calls"] == 1 and not report["native_global_restoration_required"]
    np.testing.assert_array_equal(output["mhr_shape"], batch["mhr_shape_init"])


@pytest.mark.parametrize("fault", ["hash", "size", "wrong_function", "modified_code", "wrong_file", "global_file", "delegate_file", "nonfresh"])
def test_clone_source_and_actual_callable_binding_fail_closed(gate, monkeypatch, fault):
    original, source = clone_fixture(gate, monkeypatch); report = {}
    if fault == "hash": source = source[:-1] + b" "
    elif fault == "size": source += b"\n"
    elif fault == "wrong_function": original.__name__ = "different"
    elif fault == "modified_code": original = FunctionType(original.__code__.replace(co_consts=(*original.__code__.co_consts, "tampered")), original.__globals__, original.__name__)
    elif fault == "wrong_file": original = FunctionType(original.__code__.replace(co_filename="/own/tools/other.py"), original.__globals__, original.__name__)
    elif fault == "global_file": original.__globals__["__file__"] = "/other/tools/run_mhr_wild_inference.py"
    elif fault == "delegate_file": original.__globals__["compose_mhr_output"].__code__ = original.__globals__["compose_mhr_output"].__code__.replace(co_filename="/other.py")
    else: report["composition_hook_calls"] = 1
    with pytest.raises((ValueError, TypeError)):
        gate.clone_native_forward(NP_TORCH, original, report, source_bytes=source)


def test_original_delegate_exception_propagates_without_global_patch(gate, monkeypatch):
    original, source = clone_fixture(gate, monkeypatch)
    def fails(*_): raise RuntimeError("native source error")
    original.__globals__["compose_mhr_output"].__globals__["implementation"] = fails
    clone = gate.clone_native_forward(NP_TORCH, original, {}, source_bytes=source)
    pred, batch = inputs(); delegate = original.__globals__["compose_mhr_output"]
    with pytest.raises(RuntimeError, match="native source error"): clone(pred, batch)
    assert original.__globals__["compose_mhr_output"] is delegate


def test_source_constants_and_module_firewall_no_model_io_or_torch_import(gate):
    assert gate.NATIVE_SOURCE_SHA256 == "6020ecb82f292183def6342f7bc357894620180f20ab0865b2d9c51c597b5b6f"
    assert gate.NATIVE_SOURCE_BYTES == 50990 and gate.WINDOW == 96 and gate.IDENTITY_DIMS == {"mhr_shape":45,"mhr_scale":28}
    tree = ast.parse(Path(gate.__file__).read_text())
    assert not any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names) for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"open","exec","eval"} for n in ast.walk(tree))

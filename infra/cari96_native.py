"""H102: isolate identity-delta composition, not the native network or its raw output.

The caller verifies the checkpoint/config/assets and supplies the pinned entrypoint
bytes. This module performs no I/O, model load, inference, initialization or fit.
The commercial config predicts shape45 but NOT scale28. An absent scale delta is
therefore retained as absent; native composition already supplies its zero fallback.
"""
from collections.abc import Mapping, MutableMapping
import hashlib
from pathlib import PurePosixPath
from types import CodeType, FunctionType
import sys

import numpy as np

UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
NATIVE_SOURCE_SHA256 = "6020ecb82f292183def6342f7bc357894620180f20ab0865b2d9c51c597b5b6f"
NATIVE_SOURCE_BYTES = 50990
NATIVE_RELATIVE_PATH = "tools/run_mhr_wild_inference.py"
COMPOSITION_RELATIVE_PATH = "learning/training/mhr_losses.py"
IDENTITY_DIMS = {"mhr_shape": 45, "mhr_scale": 28}
WINDOW = 96
COUNTERS = ("composition_hook_calls", "composition_delegate_calls", "composition_delegate_returns", "composition_verified_calls")


def _numpy(torch, value):
    if np.ma.isMaskedArray(value): raise ValueError("Masked native arrays are prohibited")
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().numpy()
    if type(value) is not np.ndarray: raise TypeError("Native float32 tensor/array required")
    return value


def _bytes(torch, value):
    array = _numpy(torch, value)
    if array.dtype != np.float32 or not np.isfinite(array).all(): raise ValueError("Finite original FP32 predictions required")
    return array.shape, array.dtype.str, array.tobytes(order="C")


def _snapshot(torch, values):
    return {key: _bytes(torch, value) for key, value in values.items() if torch.is_tensor(value) or isinstance(value, np.ndarray)}


def _counter(report, name):
    value = report.setdefault(name, 0)
    if type(value) is not int or value < 0: raise ValueError("Exact nonnegative hook counter required")
    report[name] = value + 1


def constrain_composition(torch, original_compose, prediction, batch, report):
    """One original compose call; only present shape/scale deltas get fresh zeros.

    The shared initializer must already be present before native neutral-height,
    crop/render/cache materialization. Its replacement/re-decoding is NOT done here.
    No returned non-identity block, object pose or raw prediction is postprocessed.
    """
    if not isinstance(report, MutableMapping): raise TypeError("Mutable hook evidence required")
    _counter(report, "composition_hook_calls")
    if not callable(original_compose) or not isinstance(prediction, Mapping) or not isinstance(batch, Mapping):
        raise TypeError("Original compose and native prediction/batch mappings required")
    raw_keys = tuple(prediction); raw_values = dict(prediction)
    raw_before = _snapshot(torch, prediction)
    init = {}; expected = {}
    for key, dimension in IDENTITY_DIMS.items():
        name = key + "_init"
        if name not in batch: raise ValueError("Shared native initializer identity is missing")
        init[key] = batch[name]; array = _numpy(torch, init[key])
        if array.shape != (1, WINDOW, dimension): raise ValueError("Exact one96-frame native identity window required")
        expected[key] = _bytes(torch, init[key])
        if any(row.tobytes() != array[0, 0].tobytes() for row in array[0]):
            raise ValueError("Initializer identity must be byte-constant before composition")
    batch_before = _snapshot(torch, {key:value for key,value in batch.items() if key.startswith("mhr_") and key.endswith("_init")})
    if "delta_mhr_shape" not in prediction: raise ValueError("Pinned commercial shape head must produce delta_mhr_shape")
    effective = dict(prediction)
    for key, dimension in IDENTITY_DIMS.items():
        delta = "delta_" + key
        if delta not in prediction: continue  # pred_mhr_scale=false, not a fabricated head output.
        value = prediction[delta]; array = _numpy(torch, value)
        if array.shape != (1, WINDOW, dimension): raise ValueError("Unexpected native identity-delta shape")
        _bytes(torch, value)
        zero = torch.zeros_like(value) if torch.is_tensor(value) else np.zeros_like(value)
        if (torch.is_tensor(value) and zero.data_ptr() == value.data_ptr()) or (not torch.is_tensor(value) and np.shares_memory(zero, value)):
            raise ValueError("Identity zero-deltas must not alias original raw predictions")
        effective[delta] = zero
    for key in prediction:
        if key not in ("delta_mhr_shape", "delta_mhr_scale") and effective[key] is not prediction[key]:
            raise ValueError("Non-identity prediction reference changed")
    _counter(report, "composition_delegate_calls")
    try:
        result = original_compose(effective, batch)
        _counter(report, "composition_delegate_returns")
    finally:
        if (tuple(prediction) != raw_keys or any(prediction[key] is not value for key,value in raw_values.items())
                or _snapshot(torch, prediction) != raw_before
                or _snapshot(torch, {key:value for key,value in batch.items() if key.startswith("mhr_") and key.endswith("_init")}) != batch_before):
            raise ValueError("Original raw predictions or native initializer changed during composition")
        report.update(raw_prediction_bytes_preserved=True, initializer_bytes_preserved=True)
    if not isinstance(result, Mapping): raise TypeError("Original native composed parameter mapping required")
    for key in IDENTITY_DIMS:
        if key not in result or _bytes(torch, result[key]) != expected[key]:
            raise ValueError("Original composition did not retain exact shared native identity")
    _counter(report, "composition_verified_calls")
    report.update(constrained_identity_composition=True, scale_delta_present="delta_mhr_scale" in prediction,
                  nonidentity_output_postprocessing=False, original_network_modified=False,
                  unchanged_original_CARI_method=False, quality_verified=False, adoption_authorized=False)
    return result


def clone_native_forward(torch, native_function, report, *, source_bytes):
    """Same pinned code/defaults/closure, shallow globals COPY with one hook.

    Hashing supplied source plus comparing its compiled function code binds the
    actual callable without filesystem reads or executing/importing source code.
    The driver independently pins the imported native composition/source closure.
    """
    if not isinstance(report, MutableMapping) or type(native_function) is not FunctionType:
        raise TypeError("Actual native Python function and mutable report required")
    if type(source_bytes) is not bytes or len(source_bytes) != NATIVE_SOURCE_BYTES or hashlib.sha256(source_bytes).hexdigest() != NATIVE_SOURCE_SHA256:
        raise ValueError("Pinned unchanged7c0d native entrypoint bytes required")
    code = native_function.__code__; filename = PurePosixPath(code.co_filename)
    if (native_function.__name__ != "run_mhr_wild_inference" or not filename.is_absolute() or ".." in filename.parts
            or not str(filename).endswith("/" + NATIVE_RELATIVE_PATH)
            or str(native_function.__globals__.get("__file__")) != str(filename)):
        raise ValueError("Native source filename/function identity differs")
    compiled = compile(source_bytes, str(filename), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    originals = [value for value in compiled.co_consts if isinstance(value, CodeType) and value.co_name == native_function.__name__]
    if len(originals) != 1 or originals[0] != code or originals[0].co_firstlineno != code.co_firstlineno:
        raise ValueError("Callable code is not the exact pinned native source function")
    delegate = native_function.__globals__.get("compose_mhr_output")
    if (type(delegate) is not FunctionType or delegate.__name__ != "compose_mhr_output"
            or not str(PurePosixPath(delegate.__code__.co_filename)).endswith("/" + COMPOSITION_RELATIVE_PATH)):
        raise ValueError("Original imported native composition function required")
    globals_copy = dict(native_function.__globals__)
    def constrained(prediction, batch):
        if native_function.__globals__.get("compose_mhr_output") is not delegate:
            raise ValueError("Original native composition global changed")
        try:
            return constrain_composition(torch, delegate, prediction, batch, report)
        finally:
            if native_function.__globals__.get("compose_mhr_output") is not delegate:
                raise ValueError("Original native composition global changed")
            report.update(native_compose_global_unchanged=True, native_global_replacement_performed=False,
                          native_global_restoration_required=False)
    globals_copy["compose_mhr_output"] = constrained
    clone = FunctionType(code, globals_copy, native_function.__name__, native_function.__defaults__, native_function.__closure__)
    clone.__kwdefaults__ = None if native_function.__kwdefaults__ is None else dict(native_function.__kwdefaults__)
    clone.__annotations__ = dict(native_function.__annotations__); clone.__module__ = native_function.__module__; clone.__qualname__ = native_function.__qualname__
    if any(globals_copy[key] is not value for key,value in native_function.__globals__.items() if key != "compose_mhr_output"):
        raise ValueError("Only composition may differ in isolated native function globals")
    report.update(native_source_sha256=NATIVE_SOURCE_SHA256, upstream_revision=UPSTREAM_REVISION,
                  native_function_code_unchanged=True, isolated_function_globals=True,
                  overridden_globals=["compose_mhr_output"], native_compose_global_unchanged=True,
                  native_global_replacement_performed=False, native_global_restoration_required=False)
    for name in COUNTERS:
        if name in report and (type(report[name]) is not int or report[name] != 0): raise ValueError("Fresh zero-call hook report required")
        report[name] = 0
    return clone

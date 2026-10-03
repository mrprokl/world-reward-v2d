"""Pure original-timeline checks for the pinned native CARI sliding policy.

The audited 7c0d3b94 entrypoint uses full windows of 96, stride 96, and an
additional terminal window when needed. Overlapping predictions belong to
their first occurrence; there is no padding, interpolation or frame dropping.
These helpers perform no source/model I/O and establish no reconstruction
quality. The caller must independently bind the actual native code to its
audited source before supplying its code object or runtime frame.
"""

from __future__ import annotations

from numbers import Integral
from types import CodeType, FrameType

import numpy as np


NATIVE_WINDOW = 96
NATIVE_STRIDE = 96


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def native_window_starts(total_frames: int) -> tuple[int, ...]:
    """Exact native ``sliding_window_starts(N, 96, 96)`` without coercion.

    Unlike the upstream convenience helper, invalid noninteger inputs are
    rejected rather than truncated. At least one full original window is
    required. For 501 frames the final start is 405, not a padded start of 480.
    """
    count = _integer(total_frames, "total_frames", minimum=NATIVE_WINDOW)
    starts = list(range(0, count - NATIVE_WINDOW + 1, NATIVE_STRIDE))
    terminal = count - NATIVE_WINDOW
    if starts[-1] != terminal:
        starts.append(terminal)
    return tuple(starts)


def finite_chunks(total_frames: int, chunk_size: int = 16) -> tuple[slice, ...]:
    """Ordered finite slices covering every original exactly once.

    Each actual batch count is ``selection.stop - selection.start``; the last
    batch is not assumed to have ``chunk_size`` entries. No data is read or
    transformed. Single-frame consumers are allowed even though native
    CoCoNet itself requires at least 96 frames.
    """
    count = _integer(total_frames, "total_frames", minimum=1)
    size = _integer(chunk_size, "chunk_size", minimum=1)
    return tuple(slice(start, min(start + size, count)) for start in range(0, count, size))


def _window_indices(value: object, expected_start: int) -> np.ndarray:
    if (type(value) is not np.ndarray or value.dtype != np.dtype("int64")
            or value.shape != (NATIVE_WINDOW,)
            or not np.array_equal(value, np.arange(expected_start, expected_start + NATIVE_WINDOW, dtype=np.int64))):
        raise ValueError("Exact ordered original int64 indices of a native96 window required")
    return value


def first_occurrence_ownership(
    total_frames: int, actual_window_indices: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return readonly ``owner_window[N]`` and ``owner_local[N]`` arrays.

    Only the entire exact native schedule is accepted, including its terminal
    overlap. Missing, repeated, reordered, padded or altered windows fail.
    Inputs are never modified. The returned indices let a consumer compare
    its stored full trajectory to the untouched raw output of each actual
    window without averaging or reassigning the overlap.
    """
    starts = native_window_starts(total_frames)
    count = int(total_frames)
    if type(actual_window_indices) not in (list, tuple) or len(actual_window_indices) != len(starts):
        raise ValueError("Complete exact native window schedule required")
    windows = [_window_indices(value, start) for value, start in zip(actual_window_indices, starts)]
    owner_window = np.full(count, -1, dtype=np.int64)
    owner_local = np.full(count, -1, dtype=np.int64)
    for ordinal, indices in enumerate(windows):
        keep = owner_window[indices] == -1
        selected = indices[keep]
        owner_window[selected] = ordinal
        owner_local[selected] = np.arange(NATIVE_WINDOW, dtype=np.int64)[keep]
    if np.any(owner_window < 0) or np.any(owner_local < 0):
        raise ValueError("Native windows left an original frame without an owner")
    owner_window.flags.writeable = False
    owner_local.flags.writeable = False
    return owner_window, owner_local


def native_caller_window_indices(
    caller: FrameType, *, native_code: CodeType, total_frames: int,
    window_ordinal: int, batch: object,
) -> np.ndarray:
    """Read actual indices from the immediate unchanged native caller frame.

    The driver passes its already source-bound native ``__code__`` and the
    immediate composition caller (obtained at the hook, not inside this
    helper). Native batches contain no explicit frame identities. Reading
    the real caller's locals avoids guessing identities from hook ordinal or
    matching equal poses. Ordinal checks require the expected native order;
    actual frame/code/batch/indices/names must also independently agree.

    This function never writes frame locals, the batch, arrays or native
    globals and does not retain a frame reference. It returns an owned,
    readonly copy. Source hash validation and comparison of every initialized
    parameter block with these indices remain the driver's responsibility.
    """
    starts = native_window_starts(total_frames)
    ordinal = _integer(window_ordinal, "window_ordinal", minimum=0)
    if ordinal >= len(starts):
        raise ValueError("Native window ordinal exceeds the exact original schedule")
    if (type(caller) is not FrameType or type(native_code) is not CodeType
            or native_code.co_name != "run_mhr_wild_inference" or caller.f_code is not native_code):
        raise ValueError("Actual immediate source-bound native caller frame required")
    values = caller.f_locals
    if values.get("batch") is not batch or "batch" not in values:
        raise ValueError("Hook batch must be the exact unchanged native caller batch")
    if (type(values.get("start")) is not int or values["start"] != starts[ordinal]
            or type(values.get("clip_len")) is not int or values["clip_len"] != NATIVE_WINDOW):
        raise ValueError("Actual native start/window length differs from the exact schedule")
    names = values.get("names")
    if type(names) is not list or names != [f"{index:06d}" for index in range(int(total_frames))]:
        raise ValueError("Actual native names must retain every original frame in order")
    indices = _window_indices(values.get("indices"), starts[ordinal]).copy()
    indices.flags.writeable = False
    return indices

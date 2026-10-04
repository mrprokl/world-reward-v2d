"""Streaming full-timeline hand evidence; no models, I/O, association or scoring.

The caller authenticates sources and configures the native IMAGE callback using
the declared protocol. This library cannot verify native detector settings. Its
deadline covers only this scan, not caller source hashing or serialization.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time

import numpy as np

from world_reward.hand_observations import (
    HandObservations, _copy, _size, _text, normalized_hand_instances,
)


@dataclass(frozen=True)
class ImageHandProtocol:
    running_mode: str = "IMAGE"
    num_hands: int = 4
    min_hand_detection_confidence: float = .5
    min_hand_presence_confidence: float = .5
    min_tracking_confidence: float = .5

    def __post_init__(self):
        if (type(self.running_mode) is not str or self.running_mode != "IMAGE"
                or type(self.num_hands) is not int or self.num_hands != 4
                or any(type(v) is not float or v != .5 for v in (
                    self.min_hand_detection_confidence, self.min_hand_presence_confidence,
                    self.min_tracking_confidence))):
            raise ValueError("Frozen IMAGE/4-hands/.5-threshold protocol required")


PROTOCOL = ImageHandProtocol()


@dataclass(frozen=True, eq=False)
class NativeHandResult:
    """All native proposals, not persistent IDs or per-joint visibility."""
    normalized_xyz: np.ndarray
    native_world_xyz: np.ndarray | None = None
    native_handedness_labels: tuple[str, ...] | None = None
    native_handedness_scores: np.ndarray | None = None

    def __post_init__(self):
        xyz = _copy(self.normalized_xyz, "native normalized XYZ", kind="f")
        if xyz.ndim != 3 or xyz.shape[1:] != (21, 3):
            raise ValueError("Native results require (N,21,3), including empty N=0")
        object.__setattr__(self, "normalized_xyz", xyz)
        if self.native_world_xyz is not None:
            object.__setattr__(self, "native_world_xyz", _copy(
                self.native_world_xyz, "native hand-centred world XYZ", shape=xyz.shape, kind="f"))
        labels = self.native_handedness_labels
        if labels is not None:
            if type(labels) is not tuple or len(labels) != len(xyz):
                raise ValueError("Native labels must preserve every proposal slot")
            for label in labels:
                _text(label)
        scores = self.native_handedness_scores
        if scores is not None:
            scores = _copy(scores, "native handedness scores", shape=(len(xyz),), kind="f")
            if labels is None or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Scores describe native handedness only, not detection confidence")
            object.__setattr__(self, "native_handedness_scores", scores)


@dataclass(frozen=True, eq=False)
class HandScanResult:
    observations: HandObservations
    native_results: tuple[NativeHandResult, ...]
    elapsed_seconds: float
    protocol: ImageHandProtocol = PROTOCOL

    def __post_init__(self):
        if (type(self.observations) is not HandObservations or type(self.native_results) is not tuple
                or len(self.native_results) != len(self.observations.frame_index)
                or any(type(item) is not NativeHandResult for item in self.native_results)
                or type(self.protocol) is not ImageHandProtocol
                or type(self.elapsed_seconds) is not float or not math.isfinite(self.elapsed_seconds)
                or self.elapsed_seconds < 0):
            raise ValueError("Complete typed native/observation timeline and finite elapsed time required")
        for raw, frame in zip(self.native_results, self.observations.frames):
            converted = normalized_hand_instances(raw.normalized_xyz, self.observations.image_size,
                native_world_xyz=raw.native_world_xyz, native_handedness_labels=raw.native_handedness_labels,
                native_handedness_scores=raw.native_handedness_scores)
            for name in ("original_xy", "image_z", "hand_centred_world_xyz"):
                left, right = getattr(converted, name), getattr(frame, name)
                if (left is None) != (right is None):
                    raise ValueError("Native/converted coordinate fields differ")
                if left is not None and (left.values.dtype != right.values.dtype
                        or left.values.shape != right.values.shape or left.values.tobytes() != right.values.tobytes()
                        or not np.array_equal(left.supported, right.supported)):
                    raise ValueError("Native/converted bytes or numerical support differ")
            if frame.wrist_relative_xyz is not None or frame.native_handedness_labels != raw.native_handedness_labels:
                raise ValueError("Scan cannot invent local coordinates or change native labels")
            left, right = raw.native_handedness_scores, frame.native_handedness_scores
            if ((left is None) != (right is None) or (left is not None and (
                    left.dtype != right.dtype or left.tobytes() != right.tobytes()))):
                raise ValueError("Native classification scores differ")

    @property
    def capacity_saturation(self):
        """Count equals configured capacity; not proof of missing or accepted hands."""
        flags = np.array([len(r.normalized_xyz) == self.protocol.num_hands for r in self.native_results], bool)
        flags.flags.writeable = False
        return flags


def scan_hands(frames, callback, *, total_frames: int, image_size: tuple[int, int],
               method: str, source_refs: Mapping, budget_seconds: float,
               timestamps_seconds=None, protocol=PROTOCOL, monotonic=time.monotonic) -> HandScanResult:
    """Call exactly once on each unchanged original RGB, then freeze all evidence.

    ``frames`` yields ``(original_index, uint8 HxWx3 RGB)`` in full arange order.
    No RGB stack/copy is retained. Errors propagate: no retry, implicit empty,
    stride, crop, mirror, fallback or FPS. The optional clock is for tiny tests.
    """
    if (type(total_frames) is not int or total_frames <= 0 or type(protocol) is not ImageHandProtocol
            or type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds)
            or budget_seconds <= 0 or not callable(callback) or not callable(monotonic)):
        raise ValueError("Positive declared frame count/scan budget and callable required")
    height, width = _size(image_size)
    start = monotonic()
    if type(start) not in (int, float) or not math.isfinite(start):
        raise ValueError("Finite monotonic clock required")
    last = start

    def check():
        nonlocal last
        now = monotonic()
        if type(now) not in (int, float) or not math.isfinite(now) or now < last:
            raise ValueError("Monotonic clock must remain finite and nondecreasing")
        last = now
        if now - start > budget_seconds:
            raise TimeoutError("Hand scan exceeded its declared core budget")
        return float(now - start)

    indices = np.arange(total_frames, dtype=np.int64)
    empty = normalized_hand_instances(np.empty((0, 21, 3), np.float32), image_size)
    # Validate/freeze metadata and real timestamps before any callback.
    declaration = HandObservations(indices, image_size, (empty,) * total_frames,
                                   method, source_refs, timestamps_seconds)
    iterator = iter(frames); native = []; observations = []
    for index in range(total_frames):
        check()
        try:
            item = next(iterator)
        except StopIteration as error:
            raise ValueError("Original timeline is missing a frame") from error
        check()
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not int or item[0] != index:
            raise ValueError("Original frame index must equal each declared arange position")
        rgb = item[1]
        if (type(rgb) is not np.ndarray or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3)):
            raise ValueError("Original uint8 RGB grid required; no implicit image conversion")
        result = callback(index, rgb)
        check()
        if type(result) is not NativeHandResult:
            raise ValueError("Callback must return typed native evidence, not an implicit empty")
        raw = NativeHandResult(result.normalized_xyz, result.native_world_xyz,
                               result.native_handedness_labels, result.native_handedness_scores)
        native.append(raw)
        observations.append(normalized_hand_instances(raw.normalized_xyz, image_size,
            native_world_xyz=raw.native_world_xyz, native_handedness_labels=raw.native_handedness_labels,
            native_handedness_scores=raw.native_handedness_scores))
        del rgb, item, result
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("Original timeline contains an extra frame")
    check()
    evidence = HandObservations(indices, image_size, tuple(observations), method,
                                declaration.source_refs, declaration.timestamps_seconds)
    result = HandScanResult(evidence, tuple(native), check(), protocol)
    object.__setattr__(result, "elapsed_seconds", check())  # includes final evidence construction
    return result

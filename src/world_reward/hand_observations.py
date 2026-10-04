"""Readonly automatic hand evidence, not identity, visibility or shared 3D pose.

Support means producer-declared numerical availability, not observed visibility.
Unsupported native values are retained without filling, clipping or mirroring.
Local/world hand coordinates never establish a global human/object metric gauge.
No I/O, models, crop inference, environment lookup or temporal association.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np

SCHEMA = "world_reward.hand_observations.v1"
# Native HandLandmark enum: MediaPipe cad7f3ab99ebf175947e40c5252c642612aae927,
# mediapipe/python/solutions/hands.py, SHA256
# 4eea13e0f63eae5cf2dc21448ea78df38f4c90a2a36f43f1db8d8797f788b51a.
LANDMARK_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def _text(value):
    if (type(value) is not str or not value.strip() or len(value) > 256
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("Explicit bounded text required")
    return value


def _copy(value, name, *, shape=None, kind=None, dtype=None):
    if np.ma.isMaskedArray(value):
        raise ValueError(name + ": masked arrays are not native evidence")
    array = np.asarray(value)
    if ((shape is not None and array.shape != shape)
            or (kind is not None and array.dtype.kind not in kind)
            or (dtype is not None and array.dtype != np.dtype(dtype))):
        raise ValueError(name + ": declared shape and dtype required")
    owned = np.array(array, copy=True, order="C", subok=False)
    owned.flags.writeable = False
    return owned


def _metadata(value):
    """Freeze tiny JSON references; these declarations do not verify their source."""
    def freeze(item):
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise ValueError("Source reference keys must be strings")
            return {key: freeze(v) for key, v in item.items()}
        if type(item) in (list, tuple):
            return [freeze(v) for v in item]
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise ValueError("Only tiny JSON source metadata, not arrays or models")

    if not isinstance(value, Mapping) or not value:
        raise ValueError("Explicit method/source reference mapping required")
    copied = freeze(value)
    if len(json.dumps(copied, allow_nan=False).encode()) > 8192:
        raise ValueError("Source references must be at most 8KB")

    def readonly(item):
        if type(item) is dict:
            return MappingProxyType({k: readonly(v) for k, v in item.items()})
        if type(item) is list:
            return tuple(readonly(v) for v in item)
        return item
    return readonly(copied)


def _size(value):
    if (type(value) is not tuple or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError("Constant positive integer (height, width) required")
    return value


@dataclass(frozen=True, eq=False)
class LandmarkEvidence:
    values: np.ndarray  # (instances, 21, dimensions); raw floating-point bytes
    supported: np.ndarray  # (instances, 21), numerical availability only
    coordinate_frame: str
    unit: str

    def __post_init__(self):
        values = _copy(self.values, "landmarks", kind="f")
        if values.ndim != 3 or values.shape[1] != 21 or values.shape[2] not in (1, 2, 3):
            raise ValueError("Ragged instances each require 21 semantic landmarks")
        supported = _copy(self.supported, "support", shape=values.shape[:2], dtype=np.bool_)
        if np.any(supported & ~np.isfinite(values).all(axis=-1)):
            raise ValueError("Nonfinite coordinates cannot be supported")
        _text(self.coordinate_frame); _text(self.unit)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "supported", supported)


@dataclass(frozen=True, eq=False)
class HandInstances:
    """Native proposal order; handedness is classification, never detection confidence."""
    original_xy: LandmarkEvidence
    image_z: LandmarkEvidence | None = None
    wrist_relative_xyz: LandmarkEvidence | None = None
    hand_centred_world_xyz: LandmarkEvidence | None = None
    native_handedness_labels: tuple[str, ...] | None = None
    native_handedness_scores: np.ndarray | None = None  # not per-joint visibility/presence

    def __post_init__(self):
        if type(self.original_xy) is not LandmarkEvidence:
            raise ValueError("Explicit original-image XY evidence required")
        count = len(self.original_xy.values)
        fields = (("original_xy", 2, "original_image", "pixel"),
                  ("image_z", 1, "wrist_relative_image_z", None),
                  ("wrist_relative_xyz", 3, "wrist_relative", None),
                  ("hand_centred_world_xyz", 3, "hand_centred_world", None))
        for name, dims, frame, unit in fields:
            item = getattr(self, name)
            if item is None:
                continue
            if (type(item) is not LandmarkEvidence or item.values.shape != (count, 21, dims)
                    or item.coordinate_frame != frame or (unit is not None and item.unit != unit)):
                raise ValueError(name + ": separate coordinate convention/count required")
            object.__setattr__(self, name, LandmarkEvidence(
                item.values, item.supported, item.coordinate_frame, item.unit))
        labels = self.native_handedness_labels
        if labels is not None:
            if type(labels) is not tuple or len(labels) != count:
                raise ValueError("Native classification labels must match all instances")
            for label in labels:
                _text(label)
        scores = self.native_handedness_scores
        if scores is not None:
            scores = _copy(scores, "handedness scores", shape=(count,), kind="f")
            if labels is None or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Finite native classification scores require matching labels")
            object.__setattr__(self, "native_handedness_scores", scores)

    @property
    def count(self):
        return len(self.original_xy.values)


@dataclass(frozen=True, eq=False)
class HandObservations:
    frame_index: np.ndarray
    image_size: tuple[int, int]  # original (height, width), constant for all T
    frames: tuple[HandInstances, ...]
    method: str
    source_refs: Mapping
    timestamps_seconds: np.ndarray | None = None  # real supplied times, not a FPS assumption
    landmark_names: tuple[str, ...] = LANDMARK_NAMES

    def __post_init__(self):
        indices = _copy(self.frame_index, "original frame indices", dtype=np.int64)
        if (indices.ndim != 1 or not len(indices)
                or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
            raise ValueError("Complete original int64 arange(T) required")
        _size(self.image_size); _text(self.method)
        if (type(self.frames) is not tuple or len(self.frames) != len(indices)
                or any(type(frame) is not HandInstances for frame in self.frames)):
            raise ValueError("Every original frame needs a ragged instance record, even empty")
        if type(self.landmark_names) is not tuple or self.landmark_names != LANDMARK_NAMES:
            raise ValueError("Verified standard 21-landmark semantic order required")
        frames = tuple(HandInstances(**{k: getattr(frame, k) for k in
            ("original_xy", "image_z", "wrist_relative_xyz", "hand_centred_world_xyz",
             "native_handedness_labels", "native_handedness_scores")}) for frame in self.frames)
        times = self.timestamps_seconds
        if times is not None:
            times = _copy(times, "timestamps", shape=indices.shape, kind="f")
            if not np.isfinite(times).all() or np.any(times[1:] <= times[:-1]):
                raise ValueError("Supplied real timestamps must be finite and strictly increasing")
        object.__setattr__(self, "frame_index", indices)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "source_refs", _metadata(self.source_refs))
        object.__setattr__(self, "timestamps_seconds", times)

    @property
    def outside_image(self):
        """Numerically finite XY beyond [0,W) x [0,H); retained, not visibility."""
        height, width = self.image_size
        flags = []
        for frame in self.frames:
            xy = frame.original_xy.values
            flag = np.isfinite(xy).all(axis=-1) & (
                (xy[..., 0] < 0) | (xy[..., 0] >= width)
                | (xy[..., 1] < 0) | (xy[..., 1] >= height))
            flag.flags.writeable = False
            flags.append(flag)
        return tuple(flags)


def normalized_hand_instances(normalized_xyz, image_size, *, native_handedness_labels=None,
                              native_handedness_scores=None, native_world_xyz=None):
    """MediaPipe-style native XYZ: XY * [W,H] once, unchanged wrist-relative Z.

    No clipping, half-pixel offset, mirror or camera inference. World coordinates
    remain hand-centred native metre predictions, never a global camera pose.
    Empty (0,21,3) arrays and numerically unsupported outputs are preserved.
    """
    height, width = _size(image_size)
    native = _copy(normalized_xyz, "normalized landmarks", kind="f")
    if native.ndim != 3 or native.shape[1:] != (21, 3):
        raise ValueError("Native normalized landmarks require (N,21,3)")
    with np.errstate(over="ignore", invalid="ignore"):
        xy = native[..., :2] * np.asarray([width, height], dtype=native.dtype)
    original = LandmarkEvidence(xy, np.isfinite(xy).all(axis=-1), "original_image", "pixel")
    z = native[..., 2:3]
    image_z = LandmarkEvidence(z, np.isfinite(z).all(axis=-1),
                              "wrist_relative_image_z", "native_image_scale")
    world = None
    if native_world_xyz is not None:
        values = _copy(native_world_xyz, "native hand-centred world XYZ", kind="f")
        if values.shape != native.shape:
            raise ValueError("Native world instances/semantic landmarks differ")
        world = LandmarkEvidence(values, np.isfinite(values).all(axis=-1),
                                 "hand_centred_world", "metre")
    return HandInstances(original, image_z=image_z, hand_centred_world_xyz=world,
                         native_handedness_labels=native_handedness_labels,
                         native_handedness_scores=native_handedness_scores)

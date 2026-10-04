"""Readonly Track 1 scene adapter, not a fitter, provenance verifier or scorer.

The first migration primitive carries already-converted MHR model parameters.
It copies bytes without interpreting native PCA/continuous controls, estimating
camera calibration, rebaking object scale or filling absent observations. Scene
metadata declares one shared camera frame; proving that a decoder actually uses
that frame remains the producer's responsibility. No I/O or models enter here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType

import numpy as np

from world_reward.contracts import Reconstruction, require_rigid_transforms
from world_reward.submission import MHR_PARAMETER_FORMAT, Track1Episode

SCHEMA = "world_reward.shared_scene.v1"
PIXEL_CONVENTIONS = ("integer_pixel_centres", "half_pixel_centres")


def _array(value, name, *, shape=None, dtype=None):
    original = np.asarray(value)
    if (np.ma.isMaskedArray(value) or original.dtype.kind not in "fiu"
            or (dtype is not None and original.dtype != np.dtype(dtype))
            or (shape is not None and original.shape != shape) or not np.isfinite(original).all()):
        raise ValueError(f"{name}: finite unmasked array with declared shape/dtype required")
    owned = np.array(original, copy=True, order="C", subok=False)
    owned.flags.writeable = False
    return owned


def _text(value, name):
    if type(value) is not str or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name}: explicit bounded text required")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Provenance requires JSON string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Provenance accepts tiny JSON metadata only, not arrays or models")


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _indices(value):
    result = _array(value, "frame_index", dtype=np.int64)
    if result.ndim != 1 or not len(result) or not np.array_equal(result, np.arange(len(result), dtype=np.int64)):
        raise ValueError("Complete original int64 arange(T) required; no dropping or reindexing")
    return result


@dataclass(frozen=True)
class ArtifactRef:
    """Metadata reference only; constructing it does not authenticate its file."""
    path: str
    bytes: int
    sha256: str

    def __post_init__(self):
        path = PurePosixPath(_text(self.path, "artifact path"))
        if (path.is_absolute() or path.as_posix() != self.path or ".." in path.parts
                or not path.parts or "\\" in self.path):
            raise ValueError("Artifact path must be canonical root-relative POSIX metadata")
        if (type(self.bytes) is not int or self.bytes <= 0 or type(self.sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None):
            raise ValueError("Artifact requires positive bytes and SHA256")


@dataclass(frozen=True, eq=False)
class SceneCamera:
    intrinsics: np.ndarray
    image_size: tuple[int, int]  # (height, width)
    gauge_convention: str
    pixel_convention: str
    camera_policy: str = "rgb_inferred"
    coordinate_frame: str = "shared_camera"
    linear_unit: str = "metre"

    def __post_init__(self):
        K = _array(self.intrinsics, "K", shape=(3, 3))
        if (K.dtype.kind != "f" or K[0, 0] <= 0 or K[1, 1] <= 0
                or K[1, 0] != 0 or not np.array_equal(K[2], [0, 0, 1])):
            raise ValueError("One positive supplied pixel K required; no normalization or estimation")
        if (type(self.image_size) is not tuple or len(self.image_size) != 2
                or any(type(size) is not int or size <= 0 for size in self.image_size)):
            raise ValueError("Explicit positive integer (height, width) required")
        _text(self.gauge_convention, "gauge convention")
        if (self.pixel_convention not in PIXEL_CONVENTIONS or self.camera_policy != "rgb_inferred"
                or self.coordinate_frame != "shared_camera" or self.linear_unit != "metre"):
            raise ValueError("Declare RGB-inferred pixel convention and shared camera-frame metres")
        object.__setattr__(self, "intrinsics", K)


@dataclass(frozen=True, eq=False)
class ObservationRef:
    """Full-T support for an automatic artifact, not a calibrated confidence.

    Unsupported entries stay false, even if a downstream initializer fills a
    pose. An entirely absent observation needs no artifact and is never filled.
    """
    name: str
    frame_index: np.ndarray
    supported: np.ndarray
    artifact: ArtifactRef | None = None

    def __post_init__(self):
        _text(self.name, "observation name")
        indices = _indices(self.frame_index)
        supported = np.asarray(self.supported)
        if np.ma.isMaskedArray(self.supported) or supported.dtype != np.bool_ or supported.shape != indices.shape:
            raise ValueError("Explicit bool support for every original frame required")
        if (self.artifact is not None and type(self.artifact) is not ArtifactRef) or (supported.any() and self.artifact is None):
            raise ValueError("Supported observations need a declared ArtifactRef")
        owned = supported.copy(); owned.flags.writeable = False
        object.__setattr__(self, "frame_index", indices)
        object.__setattr__(self, "supported", owned)


@dataclass(frozen=True, eq=False)
class SharedScene:
    clip_id: str
    frame_index: np.ndarray
    camera: SceneCamera
    reconstruction: Reconstruction
    object_vertices: np.ndarray
    object_faces: np.ndarray
    human_expression: np.ndarray
    provenance: Mapping
    observations: tuple[ObservationRef, ...] = ()
    mhr_parameter_format: str = MHR_PARAMETER_FORMAT

    def __post_init__(self):
        _text(self.clip_id, "clip identity")
        object.__setattr__(self, "frame_index", _indices(self.frame_index))
        if type(self.camera) is not SceneCamera or type(self.reconstruction) is not Reconstruction:
            raise ValueError("Declared SceneCamera and converted Reconstruction required")
        c = self.camera
        object.__setattr__(self, "camera", SceneCamera(c.intrinsics, c.image_size, c.gauge_convention,
            c.pixel_convention, c.camera_policy, c.coordinate_frame, c.linear_unit))
        r = self.reconstruction
        if (np.ma.isMaskedArray(r.object_scale) or np.asarray(r.object_scale).shape != ()
                or np.asarray(r.object_scale).dtype.kind not in "fiu"):
            raise ValueError("One numeric object scale required, never a per-frame scale")
        object.__setattr__(self, "reconstruction", Reconstruction(
            **{key: _array(getattr(r, key), key) for key in ("pose", "scales", "shape", "object_rotation", "object_translation")},
            object_scale=np.asarray(r.object_scale)[()]))
        for key in ("object_vertices", "object_faces", "human_expression"):
            object.__setattr__(self, key, _array(getattr(self, key), key))
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Explicit video-only provenance mapping required")
        provenance = _freeze(self.provenance)
        if len(json.dumps(_thaw(provenance), allow_nan=False).encode()) > 16384:
            raise ValueError("Provenance must remain tiny metadata (<=16KB)")
        object.__setattr__(self, "provenance", provenance)
        if type(self.observations) is not tuple or any(type(item) is not ObservationRef for item in self.observations):
            raise ValueError("Observations require a tuple of full-timeline references")
        if len({item.name for item in self.observations}) != len(self.observations):
            raise ValueError("Duplicate observation names")
        observations = tuple(ObservationRef(item.name, item.frame_index, item.supported, item.artifact) for item in self.observations)
        if any(not np.array_equal(item.frame_index, self.frame_index) for item in observations):
            raise ValueError("Observation timeline differs from the complete scene")
        object.__setattr__(self, "observations", observations)
        self.validate()

    @property
    def total_video_frames(self):
        return len(self.frame_index)

    def validate(self):
        """Existing ABI/mesh/SE3 gates, not geometry-forward or accuracy proof."""
        _indices(self.frame_index)
        c = self.camera
        SceneCamera(c.intrinsics, c.image_size, c.gauge_convention, c.pixel_convention,
            c.camera_policy, c.coordinate_frame, c.linear_unit)
        for item in self.observations:
            ObservationRef(item.name, item.frame_index, item.supported, item.artifact)
            if not np.array_equal(item.frame_index, self.frame_index):
                raise ValueError("Observation timeline changed")
        episode = Track1Episode(self.reconstruction, self.object_vertices, self.object_faces,
            self.human_expression, _thaw(self.provenance), self.total_video_frames, self.mhr_parameter_format)
        episode.validate()
        transforms = np.broadcast_to(np.eye(4), (self.total_video_frames, 4, 4)).copy()
        transforms[:, :3, :3] = self.reconstruction.object_rotation
        transforms[:, :3, 3] = self.reconstruction.object_translation
        require_rigid_transforms(transforms, self.total_video_frames)

    @classmethod
    def from_track1_episode(cls, episode, *, clip_id, frame_index, camera, observations=()):
        if type(episode) is not Track1Episode:
            raise ValueError("A validated converted Track1Episode is required")
        episode.validate()
        if len(frame_index) != episode.total_video_frames:
            raise ValueError("Original input frame count differs from scene indices")
        return cls(clip_id, frame_index, camera, episode.reconstruction, episode.object_vertices,
            episode.object_faces, episode.human_expression, episode.provenance, observations, episode.mhr_parameter_format)

    @classmethod
    def from_reconstruction(cls, reconstruction, *, object_vertices, object_faces,
                            human_expression, provenance, total_video_frames, **metadata):
        episode = Track1Episode(reconstruction, object_vertices, object_faces, human_expression, provenance, total_video_frames)
        return cls.from_track1_episode(episode, **metadata)

    def to_track1_episode(self):
        """Fresh byte-preserving arrays; scale/canonical mesh are not rebaked."""
        self.validate()
        r = self.reconstruction
        copied = Reconstruction(*(_array(getattr(r, key), key) for key in
            ("pose", "scales", "shape", "object_rotation", "object_translation")), r.object_scale)
        return Track1Episode(copied, _array(self.object_vertices, "vertices"), _array(self.object_faces, "faces"),
            _array(self.human_expression, "expression"), _thaw(self.provenance), self.total_video_frames, self.mhr_parameter_format)

"""Strict numerical contract at the model-to-official-packer boundary."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numbers import Integral


def require_rigid_transforms(transforms: np.ndarray, total_frames: int) -> None:
    """Reject scale, shear, reflections and perspective in a camera pose batch.

    No projection onto SO(3): invalid native output must fail rather than be
    silently repaired. Tolerances account for float32 roundoff only.
    """
    if isinstance(total_frames, (bool, np.bool_)) or not isinstance(total_frames, Integral) or total_frames < 1:
        raise ValueError("total_frames must be a positive integer")
    if np.ma.isMaskedArray(transforms):
        raise ValueError("Rigid transforms cannot hide invalid entries behind a mask")
    value = np.asarray(transforms)
    if (value.shape != (total_frames, 4, 4)
            or not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating))
            or not np.isfinite(value).all()):
        raise ValueError("Require finite real [T,4,4] rigid transforms")
    value = value.astype(np.float64)
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("Rigid transforms must have homogeneous bottom row [0,0,0,1]")
    rotation = value[:, :3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-4, rtol=0):
            raise ValueError("Rigid transforms must contain orthonormal rotations")
        if not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0):
            raise ValueError("Rigid transforms must contain proper rotations, not reflections")


@dataclass(frozen=True)
class Reconstruction:
    pose: np.ndarray
    scales: np.ndarray
    shape: np.ndarray
    object_rotation: np.ndarray
    object_translation: np.ndarray
    object_scale: float

    def validate(self, total_frames: int) -> None:
        expected = {"pose":(total_frames,136),"scales":(68,),"shape":(45,),
                    "object_rotation":(total_frames,3,3),"object_translation":(total_frames,3)}
        for name, size in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != size or not np.isfinite(value).all():
                raise ValueError(f"{name}: expected finite {size}, got {value.shape}")
        rotation = np.asarray(self.object_rotation)
        if not np.allclose(rotation @ rotation.swapaxes(-1,-2), np.eye(3), atol=1e-4, rtol=0):
            raise ValueError("Object rotations must be orthonormal")
        if not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0):
            raise ValueError("Object rotations must not contain reflections")
        if not np.isfinite(self.object_scale) or self.object_scale <= 0:
            raise ValueError("Object scale must be finite and positive")


def require_video_only_provenance(provenance: dict) -> None:
    # Missing/ambiguous provenance is not an assurance of no oracle use.
    if provenance.get("ground_truth_used") is not False:
        raise ValueError("Explicit ground_truth_used=False is required")
    if provenance.get("input_track") != "track_1":
        raise ValueError("Reconstruction must derive from Track 1")
    if provenance.get("oracle_modes") != []:
        raise ValueError("All oracle modes must be disabled explicitly")

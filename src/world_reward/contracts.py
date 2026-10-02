"""Strict numerical contract at the model-to-official-packer boundary."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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

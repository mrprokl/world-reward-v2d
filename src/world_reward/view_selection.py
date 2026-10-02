"""Experimental automatic views from predicted object poses and quality proxies.

No images, ground truth or model inference are consumed. Quality is lexicographic
(mask visibility, depth support, sharpness, lowest original frame index), not a
calibrated accuracy score. Greedy angular coverage is not a globally optimal set.
Object-to-camera poses must use the same fixed object origin and coordinate frame.
Canonical *rotation* preserves angles; moving the object origin need not do so.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ViewSelection:
    frame_indices: tuple[int, ...]
    positions: tuple[int, ...]
    directions: tuple[tuple[float, float, float], ...]
    minimum_angles_rad: tuple[float, ...]
    coverage_valid: bool
    report: dict[str, object]


def _array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must not be a masked array")
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array with shape {shape}")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite, including in invalid frames")
    return array


def select_object_views(
    frame_indices: object, rotations: object, translations: object,
    mask_visibility_fractions: object, depth_valid_fractions: object,
    sharpness: object, valid: object, *, num_views: int = 3,
) -> ViewSelection:
    """Return anchor first, then max-min angular views; never fabricate a pose.

    For column-vector poses ``p_camera = R @ p_object + t``, viewing direction
    is ``normalize(-R.T @ t)``. Explicit valid flags plus strictly positive mask
    and depth support determine eligibility; no occlusion/blur cutoff is invented.
    Every supplied pose is validated, even when ``valid`` is false. Insufficient
    eligible views raise; insufficient distinct directions return an abstention.
    The fixed 1e-6-radian coverage guard is numerical, not a validated quality
    threshold. Angles tied within 1e-12 radians use quality and original frame ID.
    A successful selection is only an experimental input proposal, not adoption.
    """
    if isinstance(num_views, (bool, np.bool_)) or not isinstance(num_views, Integral) or num_views < 1:
        raise ValueError("num_views must be a positive integer")
    if np.ma.isMaskedArray(frame_indices):
        raise ValueError("frame_indices must not be a masked array")
    if isinstance(frame_indices, (list, tuple)) and any(isinstance(i, (bool, np.bool_)) for i in frame_indices):
        raise ValueError("frame_indices must not contain boolean values")
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
        raise ValueError("frame_indices must be a nonempty original integer vector")
    if (indices < 0).any() or np.unique(indices).size != indices.size:
        raise ValueError("frame_indices must be nonnegative and unique")
    count = len(indices)
    r = _array(rotations, "rotations", (count, 3, 3))
    t = _array(translations, "translations", (count, 3))
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), rtol=0, atol=1e-6) or not np.allclose(
        np.linalg.det(r), 1, rtol=0, atol=1e-6
    ):
        raise ValueError("rotations must be proper SO(3); no repair is performed")
    magnitude = np.max(np.abs(t), axis=1)
    if (magnitude == 0).any():
        raise ValueError("translations must be nonzero to define a viewing direction")
    # Normalize before rotation to avoid overflow, without altering the direction.
    directions = -np.einsum("nji,nj->ni", r, t / magnitude[:, None])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    visibility = _array(mask_visibility_fractions, "mask_visibility_fractions", (count,))
    depth = _array(depth_valid_fractions, "depth_valid_fractions", (count,))
    blur = _array(sharpness, "sharpness", (count,))
    if ((visibility < 0) | (visibility > 1)).any() or ((depth < 0) | (depth > 1)).any():
        raise ValueError("visibility and depth fractions must be in [0, 1]")
    if (blur < 0).any():
        raise ValueError("sharpness must be nonnegative")
    if np.ma.isMaskedArray(valid):
        raise ValueError("valid must not be a masked array")
    flags = np.asarray(valid)
    if flags.shape != (count,) or flags.dtype.kind != "b":
        raise ValueError("valid must be an explicit boolean vector")
    eligible = np.flatnonzero(flags & (visibility > 0) & (depth > 0)).tolist()
    if num_views > len(eligible):
        raise ValueError(f"Requested {num_views} views but only {len(eligible)} are eligible")

    def quality(position: int) -> tuple[float, float, float, int]:
        return float(visibility[position]), float(depth[position]), float(blur[position]), -int(indices[position])

    selected = [max(eligible, key=quality)]
    minimum_angles = [0.0]  # The anchor has no preceding direction.
    while len(selected) < num_views:
        remaining = [i for i in eligible if i not in selected]
        selected_directions = directions[selected]
        angles = {}
        for i in remaining:
            dot = selected_directions @ directions[i]
            cross = np.linalg.norm(np.cross(selected_directions, directions[i]), axis=1)
            angles[i] = float(np.min(np.arctan2(cross, dot)))
        best_angle = max(angles.values())
        winner = max((i for i in remaining if best_angle - angles[i] <= 1e-12), key=quality)
        selected.append(winner)
        minimum_angles.append(angles[winner])
    coverage_valid = all(angle > 1e-6 for angle in minimum_angles[1:])
    chosen_indices = tuple(int(indices[i]) for i in selected)
    report = {
        "status": "selected" if coverage_valid else "abstain",
        "reason": None if coverage_valid else "insufficient_distinct_view_directions",
        "input_count": count, "eligible_count": len(eligible),
        "requested_views": int(num_views), "selected_frame_indices": list(chosen_indices),
        "minimum_angles_rad": minimum_angles,
        "quality_order": ["mask_visibility_fraction", "depth_valid_fraction", "sharpness", "lowest_frame_index"],
        "coverage_valid": coverage_valid, "ground_truth_used": False,
        "pose_predictions_unverified": True, "quality_is_proxy": True,
        "experimental_hypothesis": "camera_local_multi_view_shape_fusion",
        "adoption_authorized": False,
    }
    return ViewSelection(
        chosen_indices, tuple(selected),
        tuple(tuple(float(x) for x in directions[i]) for i in selected),
        tuple(minimum_angles), coverage_valid, report,
    )

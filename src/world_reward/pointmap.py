"""Strict pinhole/linear-camera-Z consistency of an estimated point map.

Only numpy and tiny JSON diagnostics: no resampling, camera fitting, depth
repair, metric-scale inference or challenge ground truth. Callers supply the
same-camera intrinsics they expect, never source calibration from another track.
"""

from __future__ import annotations

import numpy as np


def _array(value, name: str, *, boolean: bool = False) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name}: masked arrays require an explicit validity array")
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid array") from exc
    if boolean:
        if result.dtype != np.dtype(bool):
            raise ValueError(f"{name}: explicit boolean array required")
    elif result.dtype.kind not in "iuf":
        raise ValueError(f"{name}: real numeric array required")
    return result


def _intrinsics(value, name: str) -> np.ndarray:
    matrix = _array(value, name).astype(np.float64, copy=False)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: finite (3, 3) pinhole intrinsics required")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not np.array_equal(matrix[2], [0, 0, 1])):
        raise ValueError(f"{name}: positive focal lengths and zero-skew pinhole intrinsics required")
    return matrix


def validate_camera_pointmap(depth, points, validity, intrinsics_normalized, expected_K) -> dict:
    """Verify same-camera depth/point-map geometry without modifying any input.

    Shapes are depth [H,W], points [H,W,3], validity bool [H,W], and each K
    [3,3]. Convert normalized K with diag(W,H,1) @ K, then require expected
    pixel K using atol=1e-3 pixels, rtol=1e-6. Every validity=True record must
    be finite with positive depth and Z. Excluded records may contain arbitrary
    invalid values; no valid record is silently discarded or repaired.

    Valid camera Z must equal depth (atol=1e-5 in input depth units, rtol=1e-6).
    Reprojection uses X right/Y down/Z forward and must be within 1e-3 pixels
    of each raster pixel cell's centre (x+.5,y+.5). Depth units remain caller-
    supplied: these checks cannot establish absolute physical scale or accuracy.
    """
    depths, pointmap = _array(depth, "depth"), _array(points, "points")
    mask = _array(validity, "validity", boolean=True)
    if depths.ndim != 2 or min(depths.shape) < 1:
        raise ValueError("depth: nonempty shape (H, W) required")
    height, width = depths.shape
    if pointmap.shape != (height, width, 3) or mask.shape != depths.shape:
        raise ValueError("depth, points and validity must share the exact original (H, W) grid")
    normalized = _intrinsics(intrinsics_normalized, "intrinsics_normalized")
    expected = _intrinsics(expected_K, "expected_K")
    with np.errstate(over="ignore", invalid="ignore"):
        pixel_K = np.diag([width, height, 1.]) @ normalized
    if not np.isfinite(pixel_K).all() or not np.allclose(pixel_K, expected, atol=1e-3, rtol=1e-6):
        raise ValueError("Normalized intrinsics do not match expected pixel camera K")
    count = int(np.count_nonzero(mask))
    if not count:
        raise ValueError("Point map has no explicitly valid records")
    valid_depth = depths[mask].astype(np.float64, copy=False)
    valid_points = pointmap[mask].astype(np.float64, copy=False)
    if (not np.isfinite(valid_depth).all() or not np.isfinite(valid_points).all()
            or (valid_depth <= 0).any() or (valid_points[:, 2] <= 0).any()):
        raise ValueError("Every validity=True record requires finite points and positive depth/camera Z")
    if not np.allclose(valid_points[:, 2], valid_depth, atol=1e-5, rtol=1e-6):
        raise ValueError("Point-map camera Z differs from depth; ray range is not camera-Z depth")
    yy, xx = np.nonzero(mask)
    target_pixels = np.column_stack((xx + .5, yy + .5))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = valid_points[:, :2] / valid_points[:, 2, None]
        projected = projected * [pixel_K[0, 0], pixel_K[1, 1]] + pixel_K[:2, 2]
        delta = projected - target_pixels
        projection_errors = np.hypot(delta[:, 0], delta[:, 1])
    if not np.isfinite(projection_errors).all() or (projection_errors > 1e-3).any():
        raise ValueError("Point-map reprojection does not match (x+0.5, y+0.5) pixel centres")
    return {
        "schema": "world-reward-camera-pointmap-v1", "status": "pass",
        "image_size_hw": [height, width], "pixel_K": pixel_K.tolist(),
        "valid_pixels": count, "excluded_pixels": int(mask.size - count),
        "intrinsics_max_absolute_error_pixels": float(np.max(np.abs(pixel_K - expected))),
        "camera_z_max_absolute_error": float(np.max(np.abs(valid_points[:, 2] - valid_depth))),
        "reprojection_max_error_pixels": float(np.max(projection_errors)),
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "depth_kind": "linear_camera_z", "pixel_sample": "x_plus_0.5_y_plus_0.5",
        "metric_scale_accuracy_verified": False,
    }

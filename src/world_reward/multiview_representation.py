"""Pure experimental camera/pixel contracts, not a native MV-SAM3D quality gate.

Camera-local XYZ stays registered to its RGB pixels. A coordinate-only object or
world warp is not an aligned image. No model, normalization estimator or resize
algorithm is implemented here; supplied native SSI results can be checked, while
joint registration below covers only explicit crop/nearest source-pixel gathers.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class CameraPointmap:
    frame_index: int
    points_chw: np.ndarray
    valid_pixels: np.ndarray
    intrinsics: np.ndarray
    convention: str
    report: dict[str, object]


def _array(value: object, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must not be a masked array")
    array = np.asarray(value)
    if (shape is not None and array.shape != shape) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric with shape {shape}")
    return array.astype(np.float64, copy=True)


def _tolerance(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def validate_camera_pointmap(
    pointmap_chw: object, intrinsics: object, valid_pixels: object, *,
    frame_index: int = 0, reprojection_tolerance_px: float = 1e-3,
) -> CameraPointmap:
    """Check OpenCV CHW XYZ against pixel centers (x+.5,y+.5), not depth truth.

    Invalid pixels are explicitly supplied, never inferred/repaired. Nonfinite
    coordinates outside support remain untouched and counted. Positive scaling
    along each ray passes by design: this cannot establish metric depth accuracy.
    """
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, Integral) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative original integer")
    tolerance = _tolerance(reprojection_tolerance_px, "reprojection_tolerance_px")
    points = _array(pointmap_chw, "pointmap_chw")
    if points.ndim != 3 or points.shape[0] != 3 or min(points.shape[1:]) < 1:
        raise ValueError("pointmap_chw must have shape (3, H, W), with positive H/W")
    k = _array(intrinsics, "intrinsics", (3, 3))
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1]):
        raise ValueError("intrinsics must be finite positive-focal pinhole K")
    if k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("intrinsics skew is unsupported, not silently discarded")
    if np.ma.isMaskedArray(valid_pixels):
        raise ValueError("valid_pixels must not be a masked array")
    valid = np.asarray(valid_pixels)
    if valid.shape != points.shape[1:] or valid.dtype.kind != "b" or not valid.any():
        raise ValueError("valid_pixels must be a nonempty-support boolean H/W array")
    valid = valid.copy()
    supported = points[:, valid]
    if not np.isfinite(supported).all() or (supported[2] <= 0).any():
        raise ValueError("Supported points must be finite positive-depth OpenCV XYZ")
    rows, columns = np.nonzero(valid)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        uv = supported[:2] / supported[2] * np.diag(k)[:2, None] + k[:2, 2, None]
    error = np.linalg.norm(uv - np.stack((columns + .5, rows + .5)), axis=0)
    if not np.isfinite(error).all() or float(error.max()) > tolerance:
        raise ValueError("Pointmap is not camera-local RGB pixel-center registered; no warp/flip repair")
    report = {"frame_index": int(frame_index), "supported_pixels": int(valid.sum()),
              "nonfinite_pixels_outside_support": int((~np.isfinite(points).all(0) & ~valid).sum()),
              "max_reprojection_error_px": float(error.max()), "pixel_center_offset": .5,
              "camera_local_registration_verified": True, "native_model_verified": False,
              "ground_truth_used": False, "metric_depth_accuracy_verified": False}
    return CameraPointmap(int(frame_index), _readonly(points), _readonly(valid), _readonly(k),
                          "OpenCV_x_right_y_down_z_forward", report)


def to_pytorch3d_once(camera: CameraPointmap) -> CameraPointmap:
    """Illustrate the native XY flip; never feed this result to run_multi_view.

    That API performs the flip itself. A second flip request rejects the tag.
    This is not an object/world canonical transform or a new image reprojection.
    """
    if not isinstance(camera, CameraPointmap) or camera.convention != "OpenCV_x_right_y_down_z_forward":
        raise ValueError("Require validated OpenCV camera pointmap; reject a second flip")
    points = camera.points_chw.copy()
    points[:2] *= -1
    return CameraPointmap(camera.frame_index, _readonly(points), camera.valid_pixels, camera.intrinsics,
                          "PyTorch3D_x_left_y_up_z_forward",
                          dict(camera.report, coordinate_flip_count=1))


def validate_ssi_roundtrip(
    metric_chw: object, normalized_chw: object, scale: object, shift: object,
    valid_pixels: object, *, tolerance_m: float = 1e-6,
) -> dict[str, object]:
    """Check supplied native SSI: metric = normalized * scale + shift.

    Scale is a positive scalar or three-vector; shift a finite three-vector. This
    does not estimate those parameters, prove isotropic physical scale, or assert
    that the checkpoint normalizes its pointmap (some configurations do not).
    """
    tolerance = _tolerance(tolerance_m, "tolerance_m")
    metric = _array(metric_chw, "metric_chw")
    normalized = _array(normalized_chw, "normalized_chw", metric.shape)
    if metric.ndim != 3 or metric.shape[0] != 3 or min(metric.shape[1:]) < 1:
        raise ValueError("metric_chw must have shape (3, H, W)")
    s = _array(scale, "scale")
    offset = _array(shift, "shift", (3,))
    if s.shape not in ((), (3,)) or not np.isfinite(s).all() or (s <= 0).any() or not np.isfinite(offset).all():
        raise ValueError("SSI needs positive finite scalar/three-vector scale and finite shift")
    if np.ma.isMaskedArray(valid_pixels):
        raise ValueError("valid_pixels must not be a masked array")
    valid = np.asarray(valid_pixels)
    if valid.shape != metric.shape[1:] or valid.dtype.kind != "b" or not valid.any():
        raise ValueError("valid_pixels must be a nonempty-support boolean H/W array")
    if not np.isfinite(metric[:, valid]).all() or not np.isfinite(normalized[:, valid]).all():
        raise ValueError("SSI supported points must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        restored = normalized[:, valid] * np.broadcast_to(s, (3,))[:, None] + offset[:, None]
        error = np.linalg.norm(restored - metric[:, valid], axis=0)
    if not np.isfinite(error).all() or float(error.max()) > tolerance:
        raise ValueError("SSI roundtrip differs; do not repair with an extra scale/rotation")
    return {"supported_pixels": int(valid.sum()), "max_roundtrip_error_m": float(error.max()),
            "native_normalization_estimator_verified": False, "ground_truth_used": False}


def validate_joint_pixel_registration(
    original_rgb: object, original_mask: object, original_pointmap_chw: object,
    transformed_rgb: object, transformed_mask: object, transformed_pointmap_chw: object,
    source_pixels_yx: object,
) -> dict[str, object]:
    """Check all three arrays use an explicit identical crop/nearest pixel gather.

    Does not validate bilinear resize, antialiasing, RGB normalization or the
    native preprocessor. Pass pointmaps in the same representation on both sides.
    """
    values = (original_rgb, original_mask, transformed_rgb, transformed_mask, source_pixels_yx)
    if any(np.ma.isMaskedArray(value) for value in values):
        raise ValueError("Registration arrays must not be masked arrays")
    rgb, mask, target_rgb, target_mask, pixels = map(np.asarray, values)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError("original_rgb must be uint8 H/W/3")
    if target_rgb.ndim != 3 or target_rgb.shape[-1] != 3 or target_rgb.dtype != np.uint8 or min(target_rgb.shape[:2]) < 1:
        raise ValueError("transformed_rgb must be uint8 H/W/3")
    if mask.shape != rgb.shape[:2] or mask.dtype.kind != "b" or target_mask.shape != target_rgb.shape[:2] or target_mask.dtype.kind != "b":
        raise ValueError("Masks must be boolean and match their RGB grids")
    points = _array(original_pointmap_chw, "original_pointmap_chw", (3, *rgb.shape[:2]))
    target_points = _array(transformed_pointmap_chw, "transformed_pointmap_chw", (3, *target_rgb.shape[:2]))
    if pixels.shape != (*target_rgb.shape[:2], 2) or pixels.dtype.kind not in "iu":
        raise ValueError("source_pixels_yx must be an integer target-H/W/2 array")
    y, x = pixels[..., 0], pixels[..., 1]
    if (y < 0).any() or (y >= rgb.shape[0]).any() or (x < 0).any() or (x >= rgb.shape[1]).any():
        raise ValueError("Source pixel indices must be in original bounds")
    if not np.array_equal(target_rgb, rgb[y, x]) or not np.array_equal(target_mask, mask[y, x]):
        raise ValueError("RGB/mask do not share the declared source-pixel registration")
    if not np.array_equal(target_points, points[:, y, x], equal_nan=True):
        raise ValueError("Pointmap is warped/resampled differently from RGB/mask")
    return {"target_pixels": int(y.size), "joint_nearest_or_crop_registration_verified": True,
            "native_preprocessor_verified": False, "ground_truth_used": False}

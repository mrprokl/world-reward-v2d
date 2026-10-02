"""Data-free adapter from original SAM 3D Body blocks to CARI4D MHR inputs.

The caller supplies a decoder using the *same* MHR asset and learned
``head_pose`` buffers as Body inference, and configures imports from the
verified video_to_data checkout at EXPECTED_UPSTREAM_REVISION. Numerical
round-trip checks do not establish source identity, licensing, metric truth,
or final submission eligibility. No model, media, filesystem or network I/O.

Pinned contracts: cari4d/lib_mhr/{body_pose,camera_conventions,mhr_layer}.py;
SAM's 204 model controls are not a 260D continuous body-pose representation.
The original 133D body and 108D hand predictions must remain available.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np


EXPECTED_UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
MINIMUM_COCONET_WINDOW_FRAMES = 96
ROUNDTRIP_TOLERANCE_M = 1e-5
PROJECTION_TOLERANCE_PX = 0.05
PARAMETER_DIMS = {
    "mhr_global_rot6d": 6,
    "mhr_trans": 3,
    "mhr_body_pose_cont": 260,
    "mhr_hand": 108,
    "mhr_shape": 45,
    "mhr_scale": 28,
    "mhr_face": 72,
}
_RAW_DIMS = {
    "global_rot": 3,
    "pred_cam_t": 3,
    "body_pose_params": 133,
    "hand_pose_params": 108,
    "shape_params": 45,
    "scale_params": 28,
    "expr_params": 72,
}


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _floating_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} must be a floating array of shape {shape}, got {array.shape} {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    result = array.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} cannot be represented as finite float32")
    return result


def _geometry_array(value: Any, frames: int, label: str, *, points: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != frames or array.shape[2] != 3 or array.shape[1] == 0:
        raise ValueError(f"{label} must have shape [T,N>0,3]")
    if points is not None and array.shape[1] != points:
        raise ValueError(f"{label} must have {points} points")
    return _floating_array(array, tuple(array.shape), label)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _native_helpers():
    # Lazy imports: no Torch/model dependency when merely importing this file.
    # Body 6D packs column0 followed by column1; root 6D packs the first two
    # matrix columns interleaved. Do not substitute a shared generic encoder.
    from lib_mhr.body_pose import compact_model_params_to_cont_body_np
    from lib_mhr.camera_conventions import (
        mhr_init_parameter_frame_metadata,
        sam3d_root_camera_to_world_rot6d,
    )

    return compact_model_params_to_cont_body_np, sam3d_root_camera_to_world_rot6d, mhr_init_parameter_frame_metadata


def _native_decode(mhr_layer: Any, parameters: Mapping[str, np.ndarray]):
    import torch

    with torch.inference_mode():
        return mhr_layer.mhr_forward(parameters)


def _maximum_point_error(actual: Any, expected: np.ndarray, label: str) -> float:
    array = _to_numpy(actual)
    if array.shape != expected.shape or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"Decoded {label} shape/finite contract failed: {array.shape}, expected {expected.shape}")
    error = float(np.linalg.norm(array.astype(np.float64) - expected.astype(np.float64), axis=-1).max())
    if not np.isfinite(error) or error >= ROUNDTRIP_TOLERANCE_M:
        raise ValueError(f"Native CARI MHR {label} round-trip failed: {error:.9g} m (requires < {ROUNDTRIP_TOLERANCE_M:g})")
    return error


def candidate_mhr_parameters(
    arrays: Mapping[str, Any],
    intrinsics: Any,
    image_size: tuple[int, int],
    *,
    total_frames: int,
    mhr_layer: Any,
    decode_batch_size: int = 8,
) -> dict[str, Any]:
    """Return a canonical, pickle-ready CoCoNet *initializer*, never a submission.

    ``arrays`` is the unmodified full Body NPZ mapping. ``image_size`` is the
    original (height, width), ``intrinsics`` its fixed RGB-size-only K. Neither
    K nor any caller array is changed. Frames must exactly cover the declared
    original video. The downstream CoCoNet runner additionally requires >=96
    frames; this format/decoder adapter also supports tiny offline regressions.

    Mandatory native MHRLayer decoding must reproduce all provided camera-space
    vertices, joints and 70 keypoints to <1e-5 m, with identical mesh topology.
    The original expression vector must be exactly zero. Original shape/scale
    variation is reported, not silently averaged: clip-constant identity remains
    required before final submission. Missing raw blocks never get inferred
    from the 204 controls or from pre-zeroing ``pred_pose_raw``.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of original Body predictions")
    count = _positive_integer(total_frames, "total_frames")
    batch_size = _positive_integer(decode_batch_size, "decode_batch_size")
    if mhr_layer is None or not callable(getattr(mhr_layer, "mhr_forward", None)):
        raise ValueError("A native MHRLayer decoder is required")
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be original (height, width)")
    height = _positive_integer(image_size[0], "image height")
    width = _positive_integer(image_size[1], "image width")
    required = set(_RAW_DIMS) | {
        "frame_index", "vertices_root_camera_m", "vertices_camera_m", "pred_joint_coords",
        "pred_keypoints_3d", "pred_keypoints_2d", "focal_length", "faces",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Original Body blocks are missing: {missing}; no inverse-PCA or invented mapping")
    indices = np.asarray(arrays["frame_index"])
    if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer) or not np.array_equal(indices, np.arange(count)):
        raise ValueError("frame_index must exactly cover all declared original frames from zero")
    raw = {key: _floating_array(arrays[key], (count, dim), key) for key, dim in _RAW_DIMS.items()}
    if np.count_nonzero(np.asarray(arrays["expr_params"])) != 0:
        raise ValueError("Original Body expressions must be exactly zero, not overwritten by the adapter")
    if np.any(raw["pred_cam_t"][:, 2] <= 0):
        raise ValueError("Original Body camera translation must have positive depth")
    root_vertices = _geometry_array(arrays["vertices_root_camera_m"], count, "vertices_root_camera_m")
    camera_vertices = _geometry_array(arrays["vertices_camera_m"], count, "vertices_camera_m", points=root_vertices.shape[1])
    root_joints = _geometry_array(arrays["pred_joint_coords"], count, "pred_joint_coords")
    if root_joints.shape[1] < 2:
        raise ValueError("Original joints must include MHR root joint 1")
    root_keypoints = _geometry_array(arrays["pred_keypoints_3d"], count, "pred_keypoints_3d", points=70)
    image_keypoints = _floating_array(arrays["pred_keypoints_2d"], (count, 70, 2), "pred_keypoints_2d")
    focal = _floating_array(arrays["focal_length"], (count,), "focal_length")
    faces = np.asarray(arrays["faces"])
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must be a nonempty integer [F,3] topology")
    if faces.min() < 0 or faces.max() >= root_vertices.shape[1]:
        raise ValueError("faces index outside original Body vertices")
    faces = faces.astype(np.int64, copy=True)
    K = _floating_array(intrinsics, (3, 3), "Body intrinsics")
    expected_focal = float(np.hypot(height, width))
    expected_K = np.array([[expected_focal, 0, width / 2], [0, expected_focal, height / 2], [0, 0, 1]], dtype=np.float64)
    if not np.allclose(K, expected_K, rtol=1e-6, atol=1e-4) or not np.allclose(focal, expected_focal, rtol=1e-6, atol=1e-4):
        raise ValueError("Immutable K must match the original Body RGB-size default FOV; no camera substitution")
    translation = raw["pred_cam_t"][:, None, :]
    translation_error = _maximum_point_error(root_vertices + translation, camera_vertices, "translation exactly once")
    camera_joints = root_joints + translation
    camera_keypoints = root_keypoints + translation
    if np.any(camera_keypoints[..., 2] <= 0):
        raise ValueError("Original Body keypoints must project in front of the camera")
    projected = np.stack((K[0, 0] * camera_keypoints[..., 0] / camera_keypoints[..., 2] + K[0, 2], K[1, 1] * camera_keypoints[..., 1] / camera_keypoints[..., 2] + K[1, 2]), axis=-1)
    projection_error = float(np.linalg.norm(projected.astype(np.float64) - image_keypoints, axis=-1).max())
    if not np.isfinite(projection_error) or projection_error > PROJECTION_TOLERANCE_PX:
        raise ValueError(f"Immutable Body K/keypoint projection failed: {projection_error:.9g} px")
    encode_body, encode_root, frame_metadata = _native_helpers()
    parameters = {
        # For world=camera I, this stores the original global Euler triple
        # through CARI's ZYX root-rotation convention; do not flip Euler signs.
        "mhr_global_rot6d": encode_root(raw["global_rot"].copy(), np.eye(4, dtype=np.float32)),
        "mhr_trans": raw["pred_cam_t"].copy(),
        "mhr_body_pose_cont": encode_body(raw["body_pose_params"].copy()),
        "mhr_hand": raw["hand_pose_params"].copy(),
        "mhr_shape": raw["shape_params"].copy(),
        "mhr_scale": raw["scale_params"].copy(),
        "mhr_face": raw["expr_params"].copy(),
    }
    parameters = {key: _floating_array(value, (count, PARAMETER_DIMS[key]), key) for key, value in parameters.items()}
    errors = {"vertices": 0.0, "joints": 0.0, "keypoints": 0.0}
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        # Native decoding receives owned copies, never aliases of inputs or
        # returned parameters. It adds mhr_trans after its /100 and Y/Z flip.
        decoded = _native_decode(mhr_layer, {key: value[start:stop].copy() for key, value in parameters.items()})
        for name, actual, expected in (
            ("vertices", decoded.vertices, camera_vertices[start:stop]),
            ("joints", decoded.joints, camera_joints[start:stop]),
            ("keypoints", decoded.keypoints, camera_keypoints[start:stop]),
        ):
            errors[name] = max(errors[name], _maximum_point_error(actual, expected, name))
        decoded_faces = _to_numpy(decoded.faces)
        if not np.issubdtype(decoded_faces.dtype, np.integer) or not np.array_equal(decoded_faces, faces):
            raise ValueError("Native CARI MHR decoder topology differs from original Body faces")
    identity_constant = all(np.array_equal(np.asarray(arrays[key]), np.broadcast_to(np.asarray(arrays[key])[0], np.asarray(arrays[key]).shape)) for key in ("shape_params", "scale_params"))
    metadata = {
        **frame_metadata(),
        "source": "sam3d_body_rgb_world_reward_canonical_adapter",
        "adapter_expected_upstream_revision": EXPECTED_UPSTREAM_REVISION,
        "ground_truth_used": False,
        "hand_labeled_test": False,
        "oracle_modes": [],
        "geometry_units": "metres",
        "geometry_frame": "camera_x_right_y_down_z_forward_world_identity",
        "camera_intrinsics": K.tolist(),
        "camera_policy": "immutable_original_body_RGB_size_default_FOV",
        "mhr_geometry_forward_verified": True,
        "native_roundtrip_max_error_m": errors,
        "translation_once_max_error_m": translation_error,
        "projection_max_error_px": projection_error,
        "human_identity_clip_constant": bool(identity_constant),
        "submission_eligible": False,
        "challenge_performance_verified": False,
        "minimum_coconet_window_frames": MINIMUM_COCONET_WINDOW_FRAMES,
        "identity_policy": "original_per_frame_initializer_unchanged_not_final_submission",
    }
    return {
        "body_model": "mhr",
        **parameters,
        "mhr_joints": camera_joints.copy(),
        "mhr_keypoints": camera_keypoints.copy(),
        "frames": [f"{index:06d}" for index in range(count)],
        "kids": [0],
        "metadata": metadata,
    }

"""Authored full-T local-chart polynomial rigid-pose reference, not a native fit.

This DEV proposal is deliberately named ``local_chart_SG_REFERENCE``: it is not
the exact geometric Savitzky--Golay solver of any cited author, nor validated HOI
accuracy. Smooth the world-space fixed-mesh centroid and proper SO(3) logarithms
in a separate chart at each frame. Never average Euler angles or quaternion
components, unwrap globally, drop frames, change shape/scale, or freeze motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.signal import savgol_coeffs
from scipy.spatial.transform import Rotation


WINDOW_SECONDS = 0.3
POLYNOMIAL_DEGREE = 3
CHART_LIMIT_RADIANS = np.pi - 1e-3


@dataclass(frozen=True)
class SmoothedPose:
    rotation: np.ndarray
    translation: np.ndarray
    diagnostics: dict


def _finite(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{name}: masked arrays are not complete observations')
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name}: exact finite real shape {shape} required')
    return array.astype(np.float64, copy=True)


def local_chart_sg_reference(rotation, translation, *, frame_indices, fps,
        canonical_centroid, preserve_time_zero=True) -> SmoothedPose:
    """Frozen 9-frame/cubic reference at 30 Hz, rescaled by physical duration.

    ``R[t]`` maps canonical columns to the shared world/camera frame; hence the
    fixed-mesh centroid is ``R[t] @ canonical_centroid + T[t]``. Its polynomial
    estimate and ``exp(poly(log(R[j] @ R[t].T))) @ R[t]`` determine the output.
    The nearest odd count to ``0.3 * fps`` is used (ties round upward), without
    changing the declared degree or silently shortening a clip's window. Edges
    use full one-sided windows, polynomial evaluation at the original timestamp.
    Exactly contiguous original indices beginning at zero are required. No
    occlusion masks, interpolation or truth labels are accepted. Preserve the
    original first R/T by default; this is a boundary gauge constraint, not a
    per-frame evaluation alignment. Local angles approaching pi fail explicitly.
    """
    if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, Real) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps: positive finite original frame rate required')
    if type(preserve_time_zero) is not bool:
        raise ValueError('preserve_time_zero: explicit boolean required')
    if np.ma.isMaskedArray(frame_indices):
        raise ValueError('frame_indices: full original indices required, not masked')
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError('frame_indices: exact contiguous full-T indices starting at zero required')
    count = len(indices)
    physical_count = WINDOW_SECONDS * float(fps)
    if not np.isfinite(physical_count) or physical_count > count + 1:
        raise ValueError('Clip is shorter than the frozen physical smoothing window')
    window = 2 * int(np.floor(physical_count / 2)) + 1
    if window <= POLYNOMIAL_DEGREE or window > count:
        raise ValueError('Original clip/frame rate cannot support the frozen cubic odd window')
    r = _finite(rotation, (count, 3, 3), 'rotation')
    t = _finite(translation, (count, 3), 'translation')
    centroid = _finite(canonical_centroid, (3,), 'canonical_centroid')
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
            np.linalg.det(r), 1, atol=1e-6, rtol=0):
        raise ValueError('rotation: proper SO(3) required, never repair/project invalid matrices')
    half = window // 2
    starts = np.clip(np.arange(count) - half, 0, count - window)
    neighbours = starts[:, None] + np.arange(window)
    positions = np.arange(count) - starts
    coefficients = np.stack([savgol_coeffs(window, POLYNOMIAL_DEGREE, pos=p, use='dot')
        for p in range(window)])[positions]
    relative = r[neighbours] @ r[:, None].transpose(0, 1, 3, 2)
    logs = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(count, window, 3)
    maximum_angle = float(np.linalg.norm(logs, axis=-1).max())
    if maximum_angle >= CHART_LIMIT_RADIANS:
        raise ValueError('Local SO(3) chart approaches pi; reject rather than silently unwrap or alter the filter')
    correction = np.einsum('tw,twk->tk', coefficients, logs)
    maximum_correction = float(np.linalg.norm(correction, axis=-1).max())
    if maximum_correction >= CHART_LIMIT_RADIANS:
        raise ValueError('Polynomial SO(3) correction leaves the trusted local chart')
    filtered_r = Rotation.from_rotvec(correction).as_matrix() @ r
    with np.errstate(over='ignore', invalid='ignore'):
        world_centroid = np.einsum('tij,j->ti', r, centroid) + t
        filtered_centroid = np.einsum('tw,twk->tk', coefficients, world_centroid[neighbours])
        filtered_t = filtered_centroid - np.einsum('tij,j->ti', filtered_r, centroid)
    if not np.isfinite(filtered_t).all():
        raise ValueError('Centroid polynomial exceeded finite numeric range')
    if preserve_time_zero:
        filtered_r[0], filtered_t[0] = r[0], t[0]
    return SmoothedPose(filtered_r, filtered_t, dict(
        schema='world_reward.pose_smoothing_reference.v1', method='local_chart_SG_REFERENCE',
        interpretation='authored DEV proposal, not exact author geometric-SG solver or validated HOI accuracy',
        frames=count, original_frame_indices_preserved=True, dropped_frames=0, fps=float(fps),
        proposed_window_seconds=WINDOW_SECONDS, window_frames=window, window_count_seconds=window/float(fps),
        window_timestamp_span_seconds=(window-1)/float(fps),
        polynomial_degree=POLYNOMIAL_DEGREE, endpoints='one_sided_full_window_polynomial',
        time_zero_gauge_preserved=preserve_time_zero, canonical_centroid=centroid.tolist(),
        local_chart_limit_radians=float(CHART_LIMIT_RADIANS), maximum_local_angle_radians=maximum_angle,
        maximum_correction_radians=maximum_correction, shape_changed=False, scale_changed=False,
        global_unwrap=False, ground_truth_used=False, accuracy_validated=False, production_adopted=False))

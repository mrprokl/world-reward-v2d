"""Frozen automatic point-association unary; not depth, contact or pose accuracy.

The caller supplies only RGB-derived camera/pointmaps, automatic masks, native
Boots visibility and one fixed reconstructed surface. Numeric arrays cannot
establish provenance or metre units. No GT, fitting, raster backend or fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

GRID_COLUMNS, GRID_ROWS = 16, 12
MAX_QUERIES, MIN_QUERIES = 32, 8
TRACK_GRID_SIZE, ERROR_SCALE_PIXELS, COST_WEIGHT = 256, 8.0, 0.1


def _real(value, name):
    a = np.asarray(value)
    if (np.ma.isMaskedArray(value) or a.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(a).all()):
        raise ValueError(f'{name}: finite unmasked float32/float64 required')
    return a.astype(np.float64, copy=False)


def _bool(value, shape, name):
    a = np.asarray(value)
    if np.ma.isMaskedArray(value) or a.dtype != np.bool_ or a.shape != shape:
        raise ValueError(f'{name}: exact boolean shape {shape} required')
    return a


def _size(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name}: integer >= {minimum} required')
    return int(value)


def select_query_indices(automatic_mask, depth_valid, surface_visible, predicted_pointmap):
    """First32 admissible centre-grid pixels as int64[Q,2] (y,x), raster order.

    Grid centres are floor((i+.5)*size/count); queries use (0,y+.5,x+.5)
    continuous pixel coordinates downstream, matching centre-ray pointmaps.
    surface_visible must denote original predicted-surface first ray hits; the
    caller, not this selector, raycasts. No easiest-track or subsequent refill.
    Only explicitly unsupported pointmap pixels may contain NaN/invalid depth.
    """
    mask = np.asarray(automatic_mask)
    if mask.ndim != 2:
        raise ValueError('automatic_mask: original two-dimensional grid required')
    h, w = mask.shape
    _size(h, 'height', GRID_ROWS); _size(w, 'width', GRID_COLUMNS)
    mask = _bool(automatic_mask, (h, w), 'automatic_mask')
    depth = _bool(depth_valid, (h, w), 'depth_valid')
    surface = _bool(surface_visible, (h, w), 'surface_visible')
    points = np.asarray(predicted_pointmap)
    if (np.ma.isMaskedArray(predicted_pointmap) or points.shape != (h, w, 3)
            or points.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        raise ValueError('predicted_pointmap: original float32/float64[H,W,3] required')
    support = mask & depth & surface
    if not np.isfinite(points[support]).all() or np.any(points[support, 2] <= 0):
        raise ValueError('Declared predicted-surface support must be finite and in front')
    yy = ((2 * np.arange(GRID_ROWS) + 1) * h) // (2 * GRID_ROWS)
    xx = ((2 * np.arange(GRID_COLUMNS) + 1) * w) // (2 * GRID_COLUMNS)
    y, x = np.meshgrid(yy, xx, indexing='ij')
    indices = np.column_stack((y.ravel(), x.ravel())).astype(np.int64)
    indices = indices[support[indices[:, 0], indices[:, 1]]][:MAX_QUERIES].copy()
    if len(indices) < MIN_QUERIES:
        raise ValueError('Fewer than8 fixed automatic queries; no refill or fallback')
    indices.flags.writeable = False
    return indices


@dataclass(frozen=True)
class PointPoseCost:
    """Invalid slots have placeholder0 and MUST retain the explicit mask.

    No-visible evidence gives zero even for a moving pose, never an invented
    track. Positive-depth witnesses are necessary, not whole-mesh visibility or
    embedding certification. This result never asserts reconstruction quality.
    """
    costs: np.ndarray
    valid_candidates: np.ndarray
    visible_count: np.ndarray
    no_visible_evidence: np.ndarray
    nonfront_witness_count: np.ndarray


def point_pose_cost(canonical_points, tracks_256, native_visible, rotations,
                    translations, K, frame_index, *, image_width, image_height,
                    valid_candidates):
    """Return fixed .1*mean(min(1, error256/8)) per original frame/candidate.

    Shapes: points[Q,3] once per clip; tracks[T,Q,2] xy on native256 grid;
    visibility[T,Q] bool; R[T,C,3,3], t[T,C,3] in the same metre gauge;
    K[3,3] fixed zero-skew OpenCV intrinsics; frame_index=int64 arange(T).
    Finite masked candidates remain proper SE(3). Nonpositive-Z witnesses
    invalidate the whole candidate explicitly, without witness deletion. A
    frame with no geometrically admissible candidate fails, including occlusion.
    No transforms, scales, time indices, native visibility or costs are fitted.
    """
    w, h = _size(image_width, 'image_width'), _size(image_height, 'image_height')
    points, tracks = _real(canonical_points, 'canonical_points'), _real(tracks_256, 'tracks_256')
    r, t, camera = _real(rotations, 'rotations'), _real(translations, 'translations'), _real(K, 'K')
    if points.ndim != 2 or points.shape[1:] != (3,) or not MIN_QUERIES <= len(points) <= MAX_QUERIES:
        raise ValueError('One clip-constant [8..32,3] canonical point array required')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Canonical witnesses must be distinct; no duplicated query weighting')
    if r.ndim != 4 or r.shape[2:] != (3, 3) or not r.shape[0] or not r.shape[1]:
        raise ValueError('rotations: nonempty [T,C,3,3] required')
    frames, candidates = r.shape[:2]
    if t.shape != (frames, candidates, 3) or tracks.shape != (frames, len(points), 2):
        raise ValueError('Full original frame/candidate/query shapes required')
    visible = _bool(native_visible, (frames, len(points)), 'native_visible')
    valid = _bool(valid_candidates, (frames, candidates), 'valid_candidates').copy()
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64
            or not np.array_equal(indices, np.arange(frames, dtype=np.int64))):
        raise ValueError('frame_index: exact full original int64 arange(T) required')
    if (camera.shape != (3, 3) or camera[0, 0] <= 0 or camera[1, 1] <= 0
            or camera[0, 1] != 0 or camera[1, 0] != 0 or not np.array_equal(camera[2], [0, 0, 1])):
        raise ValueError('One positive-focal zero-skew OpenCV K required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if (not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-6, rtol=0)
                    or not np.allclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
                raise ValueError('Every candidate must be a proper SO(3) rotation')
            camera_points = np.einsum('tcij,qj->tcqi', r, points) + t[:, :, None]
            if not np.isfinite(camera_points).all():
                raise ValueError('Canonical transformation must remain finite')
            nonfront = np.count_nonzero(camera_points[..., 2] <= 0, axis=-1)
            valid &= nonfront == 0
            if not valid.any(axis=1).all():
                raise ValueError('Original frame has no admissible candidate; no pose fabrication')
            xyz = camera_points[valid]
            xy = xyz[..., :2] / xyz[..., 2, None]
            xy = (xy * [camera[0, 0], camera[1, 1]] + camera[:2, 2]) * [TRACK_GRID_SIZE/w, TRACK_GRID_SIZE/h]
            if not np.isfinite(xy).all():
                raise ValueError('Every admissible witness projection must remain finite')
            projected = np.zeros(camera_points.shape[:-1] + (2,), dtype=np.float64)
            projected[valid] = xy
            counts = np.count_nonzero(visible, axis=1)
            costs = np.zeros((frames, candidates), dtype=np.float64)
            for frame in range(frames):
                if not counts[frame]:
                    continue
                delta = projected[frame, valid[frame]][:, visible[frame]] - tracks[frame, visible[frame]]
                error = np.hypot(delta[..., 0], delta[..., 1])
                if not np.isfinite(error).all():
                    raise ValueError('Original continuous projection must remain finite')
                costs[frame, valid[frame]] = COST_WEIGHT * np.minimum(1, error / ERROR_SCALE_PIXELS).mean(axis=-1)
    except FloatingPointError as exc:
        raise ValueError('Point projection exceeds finite numeric range') from exc
    arrays = (costs, valid, counts.astype(np.int64), counts == 0, nonfront.astype(np.int64))
    for array in arrays:
        array.flags.writeable = False
    return PointPoseCost(*arrays)

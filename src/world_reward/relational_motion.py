"""Automatic-track 2D relational features, not contact or identity decisions.

One ordinary least-squares affine nuisance is fit to observed background tracks
only, shared by every hand/object pair. It is not a physical camera estimate:
moving background, outliers, parallax and affine model error can bias every pair.
No mesh, calibration sensor, target identity, cost weight or likelihood enters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(value):
    owned = np.array(value, copy=True)
    owned.flags.writeable = False
    return owned


def _tracks(value, supported, frames, name):
    original, support = np.asarray(value), np.asarray(supported)
    if (np.ma.isMaskedArray(value) or np.ma.isMaskedArray(supported)
            or original.dtype.kind != "f" or original.ndim != 3
            or original.shape[1:] != (frames, 2)
            or support.dtype != np.bool_ or support.shape != original.shape[:2]):
        raise ValueError(f"{name}: floating [Q,T,2] and boolean [Q,T] required")
    points = np.array(original, dtype=np.float64, copy=True)
    if not np.isfinite(points[support]).all():
        raise ValueError(f"{name}: observed points must be finite; absent points stay unsupported")
    return _readonly(points), _readonly(support)


def _instances(tracks, support, frames, name):
    if (type(tracks) is not tuple or not tracks or type(support) is not tuple
            or len(tracks) != len(support)):
        raise ValueError(f"{name}: nonempty tuples of per-instance tracks/support required")
    return tuple(_tracks(p, s, frames, name) for p, s in zip(tracks, support))


@dataclass(frozen=True, eq=False)
class RelationalMotionEvidence:
    """Readonly full-adjacent-timeline arrays; NaN means unsupported, not zero.

    features[T-1,H,O,3] = (norm(v_hand-v_object), norm(v_hand)^2,
    norm(v_object)^2), with pair_supported[T-1,H,O]. Velocities divide pixel
    residuals by image diagonal and actual timestamp interval. Thus speed units
    are image-diagonal fractions per timestamp unit, energies their squares.
    No statistical confidence, contact regime or identity is inferred.

    camera_affine[T-1,2,3] maps original pixels (x,y,1) to next-frame pixels.
    background_rmse is positional error/image diagonal, not calibrated noise;
    singular values/condition/rank diagnose the centered source design. Rank
    uses NumPy's standard numerical least-squares rank, not a tuned threshold.
    Fewer than three paired background points or rank<2 leave camera unsupported.

    hand/object_velocity[T-1,instances,2] is the componentwise median residual
    velocity over same-query observations at both ends. correspondence_count
    keeps every such query. dispersion is per-component median absolute deviation
    from that median. A median is robust for instance translation, not general
    rotation/deformation; symmetric rotation may have zero coherent velocity
    despite substantial dispersion. Componentwise medians are not rotation
    equivariant. All instances remain present even if entirely unsupported or
    represented by zero queries; no placeholder track is manufactured.
    """
    frame_index: np.ndarray
    time_intervals: np.ndarray
    features: np.ndarray
    pair_supported: np.ndarray
    camera_affine: np.ndarray
    camera_supported: np.ndarray
    background_correspondence_count: np.ndarray
    background_design_rank: np.ndarray
    background_singular_values: np.ndarray
    background_condition: np.ndarray
    background_rmse: np.ndarray
    hand_velocity: np.ndarray
    object_velocity: np.ndarray
    hand_supported: np.ndarray
    object_supported: np.ndarray
    hand_correspondence_count: np.ndarray
    object_correspondence_count: np.ndarray
    hand_dispersion: np.ndarray
    object_dispersion: np.ndarray


def relational_motion_features(frame_index, timestamps, hand_tracks, hand_observed,
                               object_tracks, object_observed, background_tracks,
                               background_observed, *, image_width, image_height):
    """Extract raw 2D movement evidence from supplied automatic point tracks.

    Each hand/object entry is [Q,T,2], Q>=0, with a separate [Q,T] boolean
    observation mask. Instance tuples remain nonempty, but Q=0 preserves an
    instance without tracked evidence, rather than inserting a fake query.
    Background is one [Qb,T,2] bank, Qb may be zero. These masks
    declare automatic observed support, not calibrated visibility. Unsupported
    coordinates may be NaN and are never used or interpolated. Full original
    frame_index must be int64 arange(T), T>=1; finite timestamps[T] strictly
    increase in caller-declared units, never an assumed FPS or sparse reindexing.

    Fit OLS to ALL paired background observations, not a pair-specific subset or
    robust-fit inlier selection. Report residuals without a goodness threshold.
    Absent/degenerate nuisance leaves every pair unsupported for that interval,
    never a confident identity/zero-motion fallback. Finite numeric overflow is
    an explicit integration error. This is only a raw feature producer; learning
    a relational likelihood and abstention requires fresh external calibration.
    """
    indices, times = np.asarray(frame_index), np.asarray(timestamps)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64
            or indices.ndim != 1 or not len(indices)
            or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
        raise ValueError("Complete original int64 arange(T) required")
    frames = len(indices)
    if (np.ma.isMaskedArray(timestamps) or times.shape != (frames,)
            or times.dtype.kind not in "fiu" or not np.isfinite(times).all()):
        raise ValueError("Finite real timestamps[T] required")
    for size in (image_width, image_height):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("Positive integer original image width/height required")
    diagonal = float(np.hypot(image_width, image_height))
    if not np.isfinite(diagonal):
        raise ValueError("Finite image diagonal required")
    hands = _instances(hand_tracks, hand_observed, frames, "hand")
    objects = _instances(object_tracks, object_observed, frames, "object")
    background, bg_support = _tracks(background_tracks, background_observed, frames, "background")
    steps, nh, no = frames - 1, len(hands), len(objects)
    affine = np.full((steps, 2, 3), np.nan)
    camera_ok = np.zeros(steps, dtype=bool)
    bg_count, ranks = np.zeros(steps, dtype=np.int64), np.zeros(steps, dtype=np.int64)
    singular = np.full((steps, 2), np.nan)
    condition, rmse = np.full(steps, np.nan), np.full(steps, np.nan)
    velocities = [np.full((steps, n, 2), np.nan) for n in (nh, no)]
    dispersions = [np.full((steps, n, 2), np.nan) for n in (nh, no)]
    counts = [np.zeros((steps, n), dtype=np.int64) for n in (nh, no)]
    supported = [np.zeros((steps, n), dtype=bool) for n in (nh, no)]
    features = np.full((steps, nh, no, 3), np.nan)
    pair_ok = np.zeros((steps, nh, no), dtype=bool)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            intervals = np.diff(np.array(times, dtype=np.float64, copy=True))
            if np.any(intervals <= 0):
                raise ValueError("Timestamps must strictly increase in their original units")
            for t, dt in enumerate(intervals):
                for bank, count in zip((hands, objects), counts):
                    count[t] = [np.count_nonzero(obs[:, t] & obs[:, t + 1]) for _, obs in bank]
                good = bg_support[:, t] & bg_support[:, t + 1]
                bg_count[t] = np.count_nonzero(good)
                if bg_count[t] < 3:
                    continue
                source, target = background[good, t], background[good, t + 1]
                source_mean, target_mean = source.mean(0), target.mean(0)
                x, y = (source - source_mean) / diagonal, (target - target_mean) / diagonal
                try:
                    coefficient, _, rank, values = np.linalg.lstsq(x, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue  # No affine nuisance is certified after solver failure.
                ranks[t], singular[t] = rank, values
                condition[t] = float(values[0] / values[1]) if values[1] > 0 else np.inf
                if rank < 2 or not np.isfinite(coefficient).all():
                    continue
                affine[t, :, :2] = coefficient.T
                affine[t, :, 2] = target_mean - source_mean @ coefficient
                rmse[t] = np.sqrt(np.mean(np.sum((y - x @ coefficient) ** 2, axis=1)))
                camera_ok[t] = True
                for bank, velocity, dispersion, observed in zip((hands, objects), velocities, dispersions, supported):
                    for j, (points, obs) in enumerate(bank):
                        paired = obs[:, t] & obs[:, t + 1]
                        if not paired.any():
                            continue
                        residual = ((points[paired, t + 1] - target_mean)
                                    - (points[paired, t] - source_mean) @ coefficient) / diagonal / dt
                        velocity[t, j] = np.median(residual, axis=0)
                        dispersion[t, j] = np.median(np.abs(residual - velocity[t, j]), axis=0)
                        observed[t, j] = True
                pair_ok[t] = supported[0][t, :, None] & supported[1][t, None, :]
                vh, vo = velocities[0][t], velocities[1][t]
                for h, o in np.argwhere(pair_ok[t]):
                    features[t, h, o] = (np.linalg.norm(vh[h] - vo[o]), vh[h] @ vh[h], vo[o] @ vo[o])
    except FloatingPointError as error:
        raise ValueError("Relational movement overflowed; check supplied coordinates/timestamp units") from error
    # BLAS-backed dot products need not honor NumPy's floating-point errstate.
    if (not np.isfinite(features[pair_ok]).all() or not np.isfinite(affine[camera_ok]).all()
            or not np.isfinite(rmse[camera_ok]).all()
            or any(not np.isfinite(value[mask]).all() for values in (velocities, dispersions)
                   for value, mask in zip(values, supported))):
        raise ValueError("Relational movement overflowed; supported evidence must remain finite")
    return RelationalMotionEvidence(*map(_readonly, (
        indices, intervals, features, pair_ok, affine, camera_ok, bg_count, ranks,
        singular, condition, rmse, *velocities, *supported, *counts, *dispersions)))

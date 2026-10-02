"""Image-evidence-guided temporal selection of supplied object-pose hypotheses.

This module has no ground-truth, reference-mesh, or evaluation-alignment input.
It selects a pose at every input frame; it neither drops occlusions nor fills
them with a separately manufactured static trajectory. An unobserved interval
is nevertheless ambiguous if the supplied hypotheses contain no useful cues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NoFeasiblePathError(ValueError):
    """Supplied valid hypotheses and optional speed bounds admit no full path."""


@dataclass(frozen=True)
class PosePath:
    """Selected original candidates and their discrete symmetry representatives.

    Arrays are new copies, indexed exactly like the input timeline. Rotations
    are ``input_rotation @ symmetry``; translations are unchanged. No SO(3)
    averaging, scale change, pose interpolation, or evaluation alignment occurs.
    Costs describe this selection objective, not reconstruction accuracy.
    """

    candidate_indices: np.ndarray
    symmetry_indices: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    unary_cost: float
    transition_cost: float
    total_cost: float


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite; mask unused candidates explicitly")
    return array


def _check_rotations(rotations: np.ndarray, name: str) -> None:
    if not np.allclose(
        rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0):
        raise ValueError(f"{name} must contain proper SO(3) rotations")


def _symmetry_group(symmetries: np.ndarray | None) -> np.ndarray:
    if symmetries is None:
        return np.eye(3)[None]
    group = _finite_array(symmetries, "symmetries")
    if group.ndim != 3 or group.shape[1:] != (3, 3) or not len(group):
        raise ValueError("symmetries must have nonempty shape [S,3,3]")
    _check_rotations(group, "symmetries")
    if not np.any(np.max(np.abs(group - np.eye(3)), axis=(1, 2)) <= 1e-5):
        raise ValueError("symmetries must include the identity")
    distances = np.max(np.abs(group[:, None] - group[None]), axis=(-1, -2))
    np.fill_diagonal(distances, np.inf)
    if np.any(distances <= 1e-5):
        raise ValueError("symmetries must not contain duplicates")
    for first in group:
        products = first @ group
        residual = np.max(np.abs(products[:, None] - group[None]), axis=(-1, -2))
        if np.any(residual.min(axis=1) > 1e-5):
            raise ValueError("symmetries must be a closed discrete rotation group")
    return group


def _nonnegative(value: float, name: str) -> float:
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from error
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return number


def _rotation_distances(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = np.einsum("aji,bjk->abik", previous, current)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    skew = np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    # atan2 is stable at both identity and pi; acos(trace) can turn an
    # identical rotation's rounding error into a spurious nonzero speed.
    sine = np.linalg.norm(skew, axis=-1) / 2
    return np.arctan2(sine, cosine)


def select_pose_path(
    rotations: np.ndarray,
    translations: np.ndarray,
    image_costs: np.ndarray,
    *,
    observation_confidence: np.ndarray | None = None,
    valid_candidates: np.ndarray | None = None,
    symmetries: np.ndarray | None = None,
    frame_times: np.ndarray | None = None,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    max_translation_speed: float | None = None,
    max_rotation_speed: float | None = None,
) -> PosePath:
    """Find the globally minimum-cost full pose path by first-order Viterbi.

    ``rotations[T,K,3,3]`` and ``translations[T,K,3]`` are supplied poses in
    one fixed camera/world frame, with translation in metres. ``image_costs``
    has shape [T,K]; lower is better, e.g. a negative log image likelihood.
    Finite negative energies are allowed. ``observation_confidence[T]`` lies
    in [0,1] and weights the frame's image costs: zero denotes missing image
    evidence, NOT a valid estimate of no movement. Candidate-specific model
    confidence may instead be incorporated in the caller's image costs.

    A transition costs ``(translation_weight * distance_m**2 +
    rotation_weight * SO3_angle_rad**2) / dt``. Thus it penalizes integrated
    squared speed, not merely large absolute poses. Both optional speed bounds
    default to absent: real fast motion must remain possible when image evidence
    warrants it. ``frame_times[T]`` must increase strictly; units may be seconds
    or original video frame indices (default 0..T-1). Bounds use those same time
    units. There are no transitions or smoothing across a fabricated frame.

    ``symmetries[S,3,3]`` must be a finite SO(3) group including identity.
    The caller must verify these are actual geometry symmetries **about the
    supplied mesh origin**, using only permitted reconstructed inputs. No
    category-name or hidden-reference symmetry is inferred here. Viterbi states
    are (candidate, symmetry), with pose ``R @ S``. This produces a consistent
    discrete representative rather than averaging incompatible orientations;
    an arbitrary equivalent global symmetry gauge can remain unidentifiable.

    Invalid candidates use an explicit boolean [T,K] mask. Every array must
    still be finite (put a harmless pose in masked slots). Missing candidates,
    malformed arrays, nonrigid rotations and infeasible speed bounds fail
    closed. Ties follow candidate/symmetry input order. Runtime is O(T(KS)^2),
    memory O(TKS + (KS)^2); retain a small ranked hypothesis pool upstream.
    """
    rotations = _finite_array(rotations, "rotations")
    translations = _finite_array(translations, "translations")
    image_costs = _finite_array(image_costs, "image_costs")
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [T,K,3,3]")
    frames, candidates = rotations.shape[:2]
    if not frames or not candidates:
        raise ValueError("T and K must both be positive")
    if translations.shape != (frames, candidates, 3):
        raise ValueError("translations must have shape [T,K,3]")
    if image_costs.shape != (frames, candidates):
        raise ValueError("image_costs must have shape [T,K]")
    _check_rotations(rotations, "rotations")
    group = _symmetry_group(symmetries)
    translation_weight = _nonnegative(translation_weight, "translation_weight")
    rotation_weight = _nonnegative(rotation_weight, "rotation_weight")
    if max_translation_speed is not None:
        max_translation_speed = _nonnegative(max_translation_speed, "max_translation_speed")
    if max_rotation_speed is not None:
        max_rotation_speed = _nonnegative(max_rotation_speed, "max_rotation_speed")

    confidence = (
        np.ones(frames)
        if observation_confidence is None
        else _finite_array(observation_confidence, "observation_confidence")
    )
    if confidence.shape != (frames,) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("observation_confidence must have shape [T] and lie in [0,1]")
    if valid_candidates is None:
        valid = np.ones((frames, candidates), dtype=bool)
    else:
        valid = np.asarray(valid_candidates)
        if valid.dtype != np.bool_ or valid.shape != (frames, candidates):
            raise ValueError("valid_candidates must be boolean with shape [T,K]")
    missing = np.flatnonzero(~valid.any(axis=1))
    if len(missing):
        raise NoFeasiblePathError(f"frame {missing[0]} has no valid candidate")
    times = (
        np.arange(frames, dtype=np.float64)
        if frame_times is None
        else _finite_array(frame_times, "frame_times")
    )
    if times.shape != (frames,):
        raise ValueError("frame_times must have shape [T] and strictly increase")
    with np.errstate(over="raise", invalid="raise"):
        try:
            intervals = np.diff(times)
        except FloatingPointError as error:
            raise ValueError("frame-time intervals overflowed; check time units") from error
    if np.any(intervals <= 0):
        raise ValueError("frame_times must have shape [T] and strictly increase")

    symmetry_count = len(group)
    states = candidates * symmetry_count
    state_rotations = (rotations[:, :, None] @ group[None, None]).reshape(
        frames, states, 3, 3
    )
    state_translations = np.repeat(translations, symmetry_count, axis=1)
    unary = np.repeat(image_costs * confidence[:, None], symmetry_count, axis=1)
    unary[~np.repeat(valid, symmetry_count, axis=1)] = np.inf
    backpointers = np.full((frames, states), -1, dtype=np.int64)
    costs = unary[0].copy()
    selected_transition_costs = np.zeros((frames, states))

    for frame in range(1, frames):
        dt = intervals[frame - 1]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                delta = state_translations[frame - 1, :, None] - state_translations[frame, None]
                distance = np.linalg.norm(delta, axis=-1)
                angles = _rotation_distances(state_rotations[frame - 1], state_rotations[frame])
                transition = (translation_weight * distance**2 + rotation_weight * angles**2) / dt
            except FloatingPointError as error:
                raise ValueError("transition costs overflowed; check units and numeric magnitudes") from error
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                if max_translation_speed is not None:
                    transition[distance / dt > max_translation_speed + 1e-12] = np.inf
                if max_rotation_speed is not None:
                    transition[angles / dt > max_rotation_speed + 1e-12] = np.inf
            except FloatingPointError as error:
                raise ValueError("transition speeds overflowed; check frame-time units") from error
        with np.errstate(over="raise", invalid="raise"):
            try:
                cumulative = costs[:, None] + transition
            except FloatingPointError as error:
                raise ValueError("path costs overflowed; check image-cost magnitudes") from error
        best_previous = np.argmin(cumulative, axis=0)
        best_costs = cumulative[best_previous, np.arange(states)]
        with np.errstate(over="raise", invalid="raise"):
            try:
                costs = best_costs + unary[frame]
            except FloatingPointError as error:
                raise ValueError("path costs overflowed; check image-cost magnitudes") from error
        if not np.isfinite(costs).any():
            raise NoFeasiblePathError(f"frame {frame} has no feasible full-path predecessor")
        backpointers[frame] = best_previous
        selected_transition_costs[frame] = transition[best_previous, np.arange(states)]

    path = np.empty(frames, dtype=np.int64)
    path[-1] = np.argmin(costs)
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    frame_indices = np.arange(frames)
    with np.errstate(over="raise", invalid="raise"):
        try:
            unary_cost = float(unary[frame_indices, path].sum())
            transition_cost = float(selected_transition_costs[frame_indices, path].sum())
        except FloatingPointError as error:
            raise ValueError("selected path costs overflowed; check cost magnitudes") from error
    total_cost = unary_cost + transition_cost
    if not np.isfinite(total_cost):
        raise ValueError("selected total cost overflowed; check cost magnitudes")
    return PosePath(
        candidate_indices=path // symmetry_count,
        symmetry_indices=path % symmetry_count,
        rotations=state_rotations[frame_indices, path].copy(),
        translations=state_translations[frame_indices, path].copy(),
        unary_cost=unary_cost,
        transition_cost=transition_cost,
        total_cost=total_cost,
    )

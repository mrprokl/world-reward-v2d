"""Bounded shared-shape proposals from predicted depth points and fixed poses.

This local five-DOF fit is not a full mesh/RGB optimizer. Observations query
only their nearest deformed surface sample: invisible surface points are never
pulled onto visible evidence. One canonical shape is shared across the clip;
supplied camera poses, centroid, metric scale, and sample are immutable inputs.
Sparse/partial depth or biased predicted poses can leave shape ambiguous. A
lower fitting residual is not evidence of accuracy against held-out ground
truth; no learned model, challenge record, score, or GT is read here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from world_reward.contracts import require_rigid_transforms
from world_reward.shape_model import DEFAULT_MAX_ABS_LOG_STRETCH, apply_fixed_shape, deformation


@dataclass(frozen=True)
class ShapeFitProposal:
    """A candidate, never an automatic mesh mutation; check ``accepted`` first."""

    params5: np.ndarray
    matrix: np.ndarray
    accepted: bool
    report: dict


def _array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide invalid entries behind a mask")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.float64, copy=True)


def _points(value, name: str, *, rank: int) -> np.ndarray:
    points = _array(value, name)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < rank + 1:
        raise ValueError(f"{name} must have nonempty [N,3] support of rank >= {rank}")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = points - points.mean(axis=0)
    if not np.isfinite(centered).all():
        raise ValueError(f"{name} numeric range overflow")
    singular = np.linalg.svd(centered, compute_uv=False)
    if (not np.isfinite(singular).all() or singular[0] <= 0
            or singular[rank - 1] <= singular[0] * 1e-8):
        raise ValueError(f"{name} is underconstrained (rank < {rank})")
    return points


def _equal_frame_soft_l1(point_counts: list[int], scale_m: float):
    """Robustify metric coordinates without changing scale with point count.

    Residual coordinates are delta/sqrt(N_i). For SciPy ``f_scale=1``, each
    coordinate uses soft-L1 scale ``scale_m/sqrt(N_i)``. Thus the cost equals
    ``sum_i mean_j soft_l1(delta_ij; scale_m)`` up to SciPy's factor 1/2.
    The five trailing shape-prior residuals retain their linear squared loss.
    """
    coordinate_scales_squared = np.concatenate([
        np.full(3 * count, scale_m * scale_m / count) for count in point_counts
    ])
    size = len(coordinate_scales_squared)

    def loss(squared_residuals):
        result = np.empty((3, len(squared_residuals)), dtype=np.float64)
        scaled = squared_residuals[:size] / coordinate_scales_squared
        root = np.sqrt(1 + scaled)
        # Equivalent to 2*sigma**2*(sqrt(1+z/sigma**2)-1), without cancellation.
        result[0, :size] = 2 * squared_residuals[:size] / (root + 1)
        result[1, :size] = 1 / root
        result[2, :size] = -.5 / coordinate_scales_squared / root**3
        result[0, size:] = squared_residuals[size:]
        result[1, size:] = 1
        result[2, size:] = 0
        return result

    return loss


def fit_shared_shape(
    canonical_surface_points,
    centroid,
    observed_frames: Sequence,
    rotations,
    translations,
    *,
    shape_prior: float = .01,
    f_scale_m: float = .01,
    max_nfev: int = 100,
    max_points_per_frame: int = 2048,
) -> ShapeFitProposal:
    """Fit one determinant-one SPD shape, starting at identity, with fixed poses.

    Surface samples must have 3D rank; each visible observation may be planar
    (rank >=2), which does not guarantee identification of all shape controls.
    Inputs must already share metric camera units. At most 2048 observations
    per frame are retained by deterministic equally spaced original indices.
    Each frame contributes equal weight regardless of its retained point count.

    Residuals are observed-minus-nearest-deformed-surface XYZ coordinates,
    divided by sqrt(N_i), with a soft-L1 transition at 0.01 m in the unweighted
    coordinates. ``shape_prior * ||params5||**2`` is the squared log-stretch
    prior (its weight has metric-squared units), not a hidden scale/offset fit.
    The loss is coordinatewise, not a rotation-invariant robust vector norm.

    Every parameter lies in +/-ln(1.5)/4. Gershgorin bounds the symmetric
    trace-free matrix's spectral radius by ln(1.5), so the model's spectral
    gate is satisfied without clipping or projecting any candidate. The
    conservative box covers less than the complete allowed SPD family.

    Optimization is bounded to <=100 objective evaluations. Nonconvergence,
    unchanged fit, or worsened raw equal-frame observed RMSE produces
    ``accepted=False``; the candidate is still reported for diagnostics and
    must not be adopted. No input arrays or caller mesh are ever modified.
    """
    surface = _points(canonical_surface_points, "canonical_surface_points", rank=3)
    pivot = _array(centroid, "centroid", (3,))
    if not isinstance(observed_frames, Sequence) or isinstance(observed_frames, (str, bytes)) or not observed_frames:
        raise ValueError("observed_frames must be a nonempty sequence of visible point arrays")
    if (isinstance(shape_prior, (bool, np.bool_)) or not isinstance(shape_prior, Real)
            or not np.isfinite(shape_prior) or shape_prior < 0):
        raise ValueError("shape_prior must be finite and nonnegative")
    if (isinstance(f_scale_m, (bool, np.bool_)) or not isinstance(f_scale_m, Real)
            or not np.isfinite(f_scale_m) or f_scale_m <= 0 or f_scale_m * f_scale_m == 0):
        raise ValueError("f_scale_m must be finite and positive at float64 squared precision")
    for name, value, maximum in (("max_nfev", max_nfev, 100), ("max_points_per_frame", max_points_per_frame, 2048)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    if max_points_per_frame < 3:
        raise ValueError("max_points_per_frame must retain at least three visible points")
    count = len(observed_frames)
    rotation = _array(rotations, "rotations", (count, 3, 3))
    translation = _array(translations, "translations", (count, 3))
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    poses[:, :3, :3], poses[:, :3, 3] = rotation, translation
    require_rigid_transforms(poses, count)
    observations, selected_indices, original_counts = [], [], []
    for index, frame in enumerate(observed_frames):
        points = _points(frame, f"observed_frames[{index}]", rank=2)
        original_counts.append(len(points))
        indices = (np.linspace(0, len(points) - 1, max_points_per_frame, dtype=np.int64)
                   if len(points) > max_points_per_frame else np.arange(len(points), dtype=np.int64))
        # Recheck support after subsampling: never repair an unlucky sparse set.
        observations.append(_points(points[indices], f"subsampled_observed_frames[{index}]", rank=2))
        selected_indices.append(indices)
    point_counts = [len(points) for points in observations]
    prior_sqrt = float(np.sqrt(shape_prior))
    robust_loss = _equal_frame_soft_l1(point_counts, float(f_scale_m))
    limit = DEFAULT_MAX_ABS_LOG_STRETCH / 4

    def coordinate_errors(parameters):
        shaped = apply_fixed_shape(surface, parameters, centroid=pivot)
        errors = []
        for points, r, t in zip(observations, rotation, translation, strict=True):
            with np.errstate(over="ignore", invalid="ignore"):
                posed = shaped @ r.T + t
            if not np.isfinite(posed).all():
                raise ValueError("Fixed-pose deformed surface exceeds finite numeric range")
            distances, indices = cKDTree(posed).query(points, k=1, workers=1)
            if not np.isfinite(distances).all():
                raise ValueError("Observed nearest-surface residual exceeds finite numeric range")
            errors.append(points - posed[indices])
        return errors

    def residual(parameters):
        coordinates = coordinate_errors(parameters)
        return np.concatenate([*(error.ravel() / np.sqrt(len(error)) for error in coordinates),
                               prior_sqrt * parameters])

    def raw_rmse(parameters):
        coordinates = coordinate_errors(parameters)
        with np.errstate(over="ignore", invalid="ignore"):
            frame_mse = np.asarray([np.mean(np.sum(error * error, axis=1)) for error in coordinates])
        if not np.isfinite(frame_mse).all():
            raise ValueError("Raw observed residual exceeds finite numeric range")
        return float(np.sqrt(frame_mse.mean())), np.sqrt(frame_mse)

    initial = np.zeros(5, dtype=np.float64)
    before_rmse, before_frames = raw_rmse(initial)
    before_residual = residual(initial)
    before_cost = float(robust_loss(before_residual**2)[0].sum() / 2)
    optimized = least_squares(residual, initial, bounds=(-np.full(5, limit), np.full(5, limit)),
                              method="trf", loss=robust_loss, f_scale=1., max_nfev=int(max_nfev),
                              ftol=1e-8, xtol=1e-8, gtol=1e-8)
    parameters = _array(optimized.x, "optimized_params5", (5,))
    if np.any(np.abs(parameters) > limit):
        raise ValueError("Optimizer returned a proposal outside the conservative parameter bounds")
    matrix = deformation(parameters)
    after_rmse, after_frames = raw_rmse(parameters)
    after_residual = residual(parameters)
    after_cost = float(robust_loss(after_residual**2)[0].sum() / 2)
    if not np.isfinite(after_cost):
        raise ValueError("Optimizer returned a nonfinite shape objective")
    accepted = bool(optimized.success and after_rmse < before_rmse and after_cost <= before_cost)
    status = ("improved" if accepted else "optimizer_failed" if not optimized.success
              else "raw_residual_worse" if after_rmse > before_rmse else "unchanged_or_not_improved")
    report = {
        "schema": "world-reward-fixed-pose-shared-shape-fit-v1", "status": status, "accepted": accepted,
        "interpretation": "visible_observed_to_surface_fitting_only_not_reconstruction_accuracy",
        "params5": parameters.tolist(), "deformation": matrix.tolist(), "determinant": float(np.linalg.det(matrix)),
        "centroid": pivot.tolist(), "shared_shape": True, "poses_fitted": False, "scale_fitted": False,
        "translation_fitted": False, "projected_or_clipped": False, "input_arrays_modified": False,
        "raw_equal_frame_rmse_before_m": before_rmse, "raw_equal_frame_rmse_after_m": after_rmse,
        "raw_per_frame_rmse_before_m": before_frames.tolist(), "raw_per_frame_rmse_after_m": after_frames.tolist(),
        "robust_prior_cost_before": before_cost, "robust_prior_cost_after": after_cost,
        "optimizer_success": bool(optimized.success), "optimizer_status": int(optimized.status),
        "optimizer_message": str(optimized.message), "optimizer_nfev": int(optimized.nfev),
        "max_nfev": int(max_nfev), "shape_prior": float(shape_prior), "robust_loss": "coordinatewise_soft_l1",
        "robust_transition_unweighted_m": float(f_scale_m), "equal_frame_weight": True,
        "parameter_absolute_bound": limit, "max_abs_log_stretch": DEFAULT_MAX_ABS_LOG_STRETCH,
        "original_observation_counts": original_counts, "sampled_observation_counts": point_counts,
        "subsample_method": "equally_spaced_original_indices_no_random_state",
        "sampled_original_indices": [indices.tolist() for indices in selected_indices],
        "nearest_neighbour_direction": "observed_to_deformed_surface_only",
        "challenge_performance_verified": False,
    }
    return ShapeFitProposal(parameters, matrix, accepted, report)

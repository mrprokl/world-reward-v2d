"""Experimental fixed-pose SPD5 proposal isolating surface discretization.

No pose uncertainty, RGB loss, silhouette validation or model-selection gate is
implemented. Lower observed residual is not shape accuracy or adoption evidence.
One fixed-volume shape/pivot is shared; supplied metric camera poses never move.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from world_reward.continuous_surface import observed_to_triangle_distance_squared, validate_numpy_geometry
from world_reward.shape_model import DEFAULT_MAX_ABS_LOG_STRETCH, deformation


PARAMETER_BOUND = DEFAULT_MAX_ABS_LOG_STRETCH / 4


@dataclass(frozen=True)
class ContinuousShapeProposal:
    params5: np.ndarray
    matrix: np.ndarray
    accepted: bool
    report: dict


def _array(value, name, shape=None):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype.kind not in "iuf" or not np.isfinite(array).all()
            or (shape is not None and array.shape != shape)):
        raise ValueError(f"{name} requires finite real values with shape {shape}")
    return array.astype(np.float64, copy=True)


def prepare_inputs(vertices, faces, centroid, observed_frames, rotations, translations):
    """Pure immutable geometry/pose guards, before any CUDA/Torch import."""
    v = _array(vertices, "canonical_vertices")
    if v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3:
        raise ValueError("Require canonical vertices [V,3]")
    f = np.asarray(faces)
    if (np.ma.isMaskedArray(faces) or f.dtype.kind not in "iu" or f.ndim != 2 or f.shape[1:] != (3,)
            or not len(f) or (f < 0).any() or (f >= len(v)).any()):
        raise ValueError("Require nonempty valid integer active faces, no padding")
    f = f.astype(np.int64, copy=True)
    validate_numpy_geometry(v[:1], v[f])  # Repeated, collapsed and padded faces fail.
    pivot = _array(centroid, "centroid", (3,))
    if (not isinstance(observed_frames, Sequence) or isinstance(observed_frames, (str, bytes))
            or not len(observed_frames)):
        raise ValueError("Require nonempty sequence of observed frame arrays")
    observations, selected, counts = [], [], []
    for frame in observed_frames:
        p = _array(frame, "observed_frame")
        if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 3:
            raise ValueError("Observed frames require at least three finite [N,3] points")
        ids = np.linspace(0, len(p) - 1, 2048, dtype=np.int64) if len(p) > 2048 else np.arange(len(p))
        sampled = p[ids]
        with np.errstate(over="ignore", invalid="ignore"):
            centered = sampled - sampled.mean(0)
        if not np.isfinite(centered).all():
            raise ValueError("Observed support exceeds finite numeric range")
        singular = np.linalg.svd(centered, compute_uv=False)
        if not np.isfinite(singular).all() or singular[0] <= 0 or singular[1] <= singular[0] * 1e-8:
            raise ValueError("Observed frame is underconstrained (rank < 2)")
        observations.append(sampled.copy()); selected.append(ids); counts.append(len(p))
    r = _array(rotations, "rotations", (len(observations), 3, 3))
    t = _array(translations, "translations", (len(observations), 3))
    if (not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)):
        raise ValueError("Supplied rotations must be proper SO(3), never projected")
    return v, f, pivot, observations, r, t, selected, counts


def _controls(shape_prior, robust_transition_m, max_evaluations, point_chunk_size):
    for value, name, zero_allowed in ((shape_prior, "shape_prior", True), (robust_transition_m, "robust_transition_m", False)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
                or value < 0 or (not zero_allowed and value == 0)
                or (not zero_allowed and (value < np.sqrt(np.finfo(float).tiny) or value > np.sqrt(np.finfo(float).max)))):
            raise ValueError(f"{name} must be finite {'nonnegative' if zero_allowed else 'positive'}")
    for value, name in ((max_evaluations, "max_evaluations"), (point_chunk_size, "point_chunk_size")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_evaluations > 100:
        raise ValueError("max_evaluations cannot exceed 100")


def isotropic_robust_cost_numpy(squared_distances, transition_m=.01):
    """Pure scalar-vector pseudo-Huber reference, not coordinatewise loss."""
    _controls(.01, transition_m, 100, 64)
    d2 = _array(squared_distances, "squared_distances")
    if d2.ndim != 1 or not len(d2) or (d2 < 0).any():
        raise ValueError("Require nonempty nonnegative squared distances")
    with np.errstate(over="ignore", invalid="ignore"):
        values = 2 * d2 / (np.sqrt(1 + d2 / transition_m**2) + 1)
    if not np.isfinite(values).all():
        raise ValueError("Robust costs exceed finite numeric range")
    return float(values.mean())


class _EvaluationLimit(RuntimeError):
    pass


def fit_shared_shape_continuous(canonical_vertices, faces, centroid, observed_frames, rotations, translations,
                                *, shape_prior=.01, robust_transition_m=.01, max_evaluations=100, point_chunk_size=64):
    """Return a non-applied proposal; CUDA only, fixed poses, <=100 evaluations.

    Same conservative +/-ln(1.5)/4 parameter box as the sampled fitter. It
    bounds every eigenvalue of trace-free symmetric log-stretch by ln(1.5)
    without projection. Pose/scale/volume never enter the optimizer parameters.
    ``accepted`` denotes only strict training objective/RMSE improvement and
    optimizer success; external untouched-frame falsification remains required.
    """
    _controls(shape_prior, robust_transition_m, max_evaluations, point_chunk_size)
    v, f, pivot, observations, r, t, ids, counts = prepare_inputs(
        canonical_vertices, faces, centroid, observed_frames, rotations, translations)
    import torch
    from scipy.optimize import minimize
    if not torch.cuda.is_available():
        raise RuntimeError("Continuous shape experiment requires Azure CUDA, no local CPU fallback")
    vertex, pivot_t = torch.tensor(v, device="cuda"), torch.tensor(pivot, device="cuda")
    face_t, rotation, translation = torch.tensor(f, device="cuda"), torch.tensor(r, device="cuda"), torch.tensor(t, device="cuda")
    targets = [torch.tensor(p, device="cuda") for p in observations]
    evaluations, history = 0, []

    def objective(parameters):
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise _EvaluationLimit("Hard objective evaluation budget reached")
        evaluations += 1
        parameters = _array(parameters, "optimizer parameters", (5,))
        if (np.abs(parameters) > PARAMETER_BOUND).any():
            raise ValueError("Optimizer parameters outside fixed bounds; no projection")
        theta = torch.tensor(parameters, device="cuda", requires_grad=True)
        a, b, xy, xz, yz = theta.unbind()
        log_stretch = torch.stack((a, xy, xz, xy, b, yz, xz, yz, -a - b)).reshape(3, 3)
        shaped = (vertex - pivot_t) @ torch.matrix_exp(log_stretch).T + pivot_t
        robust, mse = [], []
        for points, R, T in zip(targets, rotation, translation, strict=True):
            d2 = observed_to_triangle_distance_squared(points, (shaped @ R.T + T)[face_t], point_chunk_size=point_chunk_size)
            robust.append((2 * d2 / (torch.sqrt(1 + d2 / robust_transition_m**2) + 1)).mean())
            mse.append(d2.mean())
        cost = torch.stack(robust).mean() + shape_prior * theta.square().sum()
        cost.backward()
        value, gradient = float(cost.detach()), theta.grad.detach().cpu().numpy().copy()
        raw = float(torch.sqrt(torch.stack(mse).mean()).detach())
        if not np.isfinite(value) or not np.isfinite(raw) or not np.isfinite(gradient).all():
            raise ValueError("Continuous objective/gradient produced nonfinite output")
        history.append((parameters.copy(), value, raw))
        return value, gradient

    initial = np.zeros(5)
    objective(initial)
    # Initial gradient is recomputed by SciPy within the same hard budget; no
    # mutable surface/nearest-face state is cached between parameter proposals.
    success, message, chosen = False, "", initial
    try:
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * 5,
                          options={"maxfun": int(max_evaluations), "maxiter": int(max_evaluations), "ftol": 1e-12, "gtol": 1e-8})
        success, message, chosen = bool(result.success), str(result.message), _array(result.x, "optimized parameters", (5,))
    except _EvaluationLimit as exc:
        message = str(exc)
    matching = [record for record in history if np.array_equal(record[0], chosen)]
    if not matching:
        raise RuntimeError("Optimizer final parameters lack an evaluated finite objective")
    before, after = history[0], matching[-1]
    matrix = deformation(chosen)
    if evaluations >= max_evaluations:
        success, message = False, "Hard objective evaluation budget reached; convergence not certified"
    accepted = bool(success and after[2] < before[2] and after[1] <= before[1])
    report = {"schema": "world-reward-fixed-pose-continuous-shape-v1", "status": "improved" if accepted else "abstained",
              "accepted_training_proposal": accepted, "params5": chosen.tolist(), "matrix": matrix.tolist(),
              "optimizer_success": success, "optimizer_message": message, "objective_evaluations": evaluations,
              "max_evaluations": int(max_evaluations), "raw_equal_frame_rmse_before_m": before[2], "raw_equal_frame_rmse_after_m": after[2],
              "robust_prior_cost_before": before[1], "robust_prior_cost_after": after[1], "shape_prior": float(shape_prior),
              "robust_transition_m": float(robust_transition_m), "robust_loss": "isotropic_pseudo_huber", "equal_frame_weight": True,
              "point_chunk_size": int(point_chunk_size), "parameter_absolute_bound": PARAMETER_BOUND,
              "original_observation_counts": counts, "sampled_original_indices": [index.tolist() for index in ids],
              "shared_shape": True, "poses_fitted": False, "pose_uncertainty_not_modeled": True,
              "fixedposes_not_truthvalidated": True, "scale_fitted": False, "fixed_volume": True,
              "projected_or_clipped": False, "external_reserved_frame_adoption_gate_required": True,
              "adoption_authorized": False, "challenge_performance_verified": False}
    return ContinuousShapeProposal(chosen.copy(), matrix, accepted, report)

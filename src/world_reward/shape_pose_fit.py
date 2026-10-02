"""Experimental nested continuous-surface M0 pose / M1 shared-shape+pose fit.

Supplied views each get a bounded rigid nuisance correction; unseen-view pose
transfer is not modeled. Equal-view metric pseudo-Huber data and pose starts are
identical in both models. Only M1 has the declared SPD prior. No registration to
truth, priors in the data Schur, pseudo-inverse, hidden vertex attraction, mesh
mutation, production adoption or calibrated identification is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from world_reward.shape_fit_continuous import prepare_inputs
from world_reward.shape_model import DEFAULT_MAX_ABS_LOG_STRETCH, deformation
from world_reward.shape_selection import shape_schur_complement


SHAPE_UNIT, ROTATION_UNIT_RAD, TRANSLATION_UNIT_M = .05, .01, .01
POSE_BOUND_NORMALIZED = 5.
SHAPE_BOUND_NORMALIZED = DEFAULT_MAX_ABS_LOG_STRETCH / 4 / SHAPE_UNIT
MAX_OPTIMIZER_CALLS, MAX_MODEL_CALLS, JACOBIAN_STEP = 50, 100, 1e-4
ROBUST_TRANSITION_M, SHAPE_PRIOR = .01, .01


@dataclass(frozen=True)
class PoseShapeModel:
    params5: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    heldout_losses_m2: np.ndarray
    report: dict


def _check_budget(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError("Nested shape/pose experiment exceeded its fixed wall-clock budget")


def transform_numpy(q, vertices, pivot, rotations, translations, joint):
    """Exact SO(3) exponential and shared SPD; normalized coordinates explicit."""
    from scipy.spatial.transform import Rotation
    if type(joint) is not bool or np.ma.isMaskedArray(q):
        raise ValueError("Joint model must be explicit boolean and parameters cannot be masked")
    source = np.asarray(q)
    if source.dtype.kind not in "iuf":
        raise ValueError("Normalized parameters must be finite real values")
    q = source.astype(np.float64, copy=True)
    count, offset = len(rotations), 5 if joint else 0
    if q.shape != (offset + 6 * count,) or not np.isfinite(q).all():
        raise ValueError("Require finite normalized parameters matching every supplied view")
    theta = q[:5] * SHAPE_UNIT if joint else np.zeros(5)
    matrix = deformation(theta)
    shaped = (np.asarray(vertices) - pivot) @ matrix.T + pivot if joint else np.asarray(vertices).copy()
    pose = q[offset:].reshape(count, 6)
    R = Rotation.from_rotvec(pose[:, :3] * ROTATION_UNIT_RAD).as_matrix() @ rotations
    t = translations + pose[:, 3:] * TRANSLATION_UNIT_M
    return shaped, R, t, theta


def fixed_feature_residuals(posed_vertices, faces, observations, assignments):
    """Local vector residual with closest-face/region assignments frozen.

    Face IDs come from the exact continuous primitive, not vertex/sample KNN.
    Region0=interior plane;1..3=closed edge with its endpoint/interior regime
    fixed. This diagnostic excludes correspondence switching and is not an
    exact nonsmooth Hessian or a statistical-information certificate.
    """
    residuals = []
    for vertices, points, (face_ids, regions, endpoint_regimes) in zip(posed_vertices, observations, assignments, strict=True):
        triangles = vertices[faces[face_ids]]
        a, b, c = triangles.transpose(1, 0, 2)
        normal = np.cross(b - a, c - a)
        norm2 = np.einsum("ij,ij->i", normal, normal)
        closest = points - normal * (np.einsum("ij,ij->i", points - a, normal) / norm2)[:, None]
        for edge_id, (start, end) in enumerate(((a, b), (b, c), (c, a)), 1):
            chosen = regions == edge_id
            edge = end - start
            parameter = np.einsum("ij,ij->i", points - start, edge) / np.einsum("ij,ij->i", edge, edge)
            parameter = np.where(endpoint_regimes < 0, 0., np.where(endpoint_regimes > 0, 1., parameter))
            closest[chosen] = (start + parameter[:, None] * edge)[chosen]
        residuals.append((points - closest).ravel() / np.sqrt(len(points) * len(observations)))
    result = np.concatenate(residuals)
    if not np.isfinite(result).all():
        raise ValueError("Local fixed-feature data residual is nonfinite")
    return result


def data_schur_from_jacobian(jacobian):
    """Unweighted equal-view data JᵀJ only; no robust prior/damping/PSD repair."""
    J = np.asarray(jacobian)
    if (np.ma.isMaskedArray(jacobian) or J.dtype.kind not in "iuf" or J.ndim != 2
            or J.shape[1] < 11 or (J.shape[1] - 5) % 6 or not len(J) or not np.isfinite(J).all()):
        raise ValueError("Require finite data Jacobian with SPD5 plus per-view SE3 columns")
    with np.errstate(over="ignore", invalid="ignore"):
        H = J.T @ J
    if not np.isfinite(H).all():
        raise ValueError("Data Hessian exceeds finite numeric range")
    return shape_schur_complement(H[:5, :5], H[5:, 5:], H[:5, 5:])


def _assignments(torch, posed, faces, observations):
    from world_reward.continuous_surface import _torch_distances
    assignments, ambiguous = [], False
    for vertices, p_np in zip(posed, observations, strict=True):
        p, triangles = torch.tensor(p_np, device="cuda"), vertices[faces]
        ids = []
        with torch.no_grad():
            for start in range(0, len(p), 64):
                distances = _torch_distances(torch, p[start:start + 64, None], triangles[None])
                if len(triangles) > 1:
                    best = distances.topk(2, largest=False, dim=1)
                    ambiguous |= bool((best.values[:, 1] - best.values[:, 0] <= 1e-12).any())
                ids.append(distances.argmin(dim=1))
        ids = torch.cat(ids).cpu().numpy()
        tri = triangles.detach().cpu().numpy()[ids]
        a, b, c = tri.transpose(1, 0, 2)
        normal = np.cross(b - a, c - a); norm2 = np.einsum("ij,ij->i", normal, normal)
        delta = p_np - a
        bary_v = np.einsum("ij,ij->i", np.cross(delta, c - a), normal) / norm2
        bary_w = np.einsum("ij,ij->i", np.cross(b - a, delta), normal) / norm2
        inside = (bary_v >= 0) & (bary_w >= 0) & (bary_v + bary_w <= 1)
        edge_costs, parameters = [], []
        for s, e in ((a, b), (b, c), (c, a)):
            edge = e - s
            parameter = np.einsum("ij,ij->i", p_np - s, edge) / np.einsum("ij,ij->i", edge, edge)
            d = p_np - (s + np.clip(parameter, 0, 1)[:, None] * edge)
            edge_costs.append(np.einsum("ij,ij->i", d, d)); parameters.append(parameter)
        edge = np.argmin(np.array(edge_costs), axis=0)
        selected_t = np.array(parameters)[edge, np.arange(len(p_np))]
        regions = np.where(inside, 0, edge + 1)
        regimes = np.where(selected_t <= 0, -1, np.where(selected_t >= 1, 1, 0))
        ambiguous |= bool(np.any(inside & (np.minimum.reduce((np.abs(bary_v), np.abs(bary_w), np.abs(1 - bary_v - bary_w))) <= 1e-7)))
        assignments.append((ids, regions, regimes))
    return assignments, ambiguous


class _EvaluationLimit(RuntimeError):
    pass


def fit_nested_shape_pose(vertices, faces, pivot, training_frames, heldout_frames, rotations, translations,
                          *, heldout_pixels_disjoint: bool, deadline_monotonic=None):
    """Fit both models, with <=50 objective and <=100 total calls per model.

    Hard-call accounting includes the finite-difference Schur and held-out
    evaluations. Both start independently from exactly the supplied poses and
    zero shape; neither consumes held-out pixels for fitting. Fits that do not
    certify convergence return their initial parameters, never a hidden best
    trial. View-pixel holdout is a proxy, not unseen-view pose generalization.
    """
    if heldout_pixels_disjoint is not True:
        raise ValueError("Explicit heldout_pixels_disjoint=True required")
    v, f, center, observations, R, T, _, _ = prepare_inputs(vertices, faces, pivot, training_frames, rotations, translations)
    _, _, _, heldout, _, _, _, _ = prepare_inputs(vertices, faces, pivot, heldout_frames, rotations, translations)
    if len(observations) != 3 or len(heldout) != 3 or any(len(p) > 128 for p in (*observations, *heldout)):
        raise ValueError("Frozen experiment requires exactly three views and <=128 points per split/view")
    deadline = time.perf_counter() + 120 if deadline_monotonic is None else deadline_monotonic
    if not isinstance(deadline, (float, int)) or isinstance(deadline, bool) or not np.isfinite(deadline):
        raise ValueError("Require a finite absolute monotonic deadline")
    _check_budget(deadline)
    import torch
    from scipy.optimize import minimize
    from world_reward.continuous_surface import observed_to_triangle_distance_squared
    if not torch.cuda.is_available():
        raise RuntimeError("Nested shape/pose experiment requires Azure CUDA; no CPU/local fallback")
    vertex, center_t, rotation, translation = (torch.tensor(value, device="cuda") for value in (v, center, R, T))
    face_t = torch.tensor(f, device="cuda")
    targets = [torch.tensor(p, device="cuda") for p in observations]
    models = []
    for joint in (False, True):
        offset, count = 5 if joint else 0, 0
        initial = np.zeros(offset + 18); history = []
        def torch_geometry(q):
            theta = q[:5] * SHAPE_UNIT if joint else q.new_zeros(5)
            a, b, xy, xz, yz = theta.unbind()
            log = torch.stack((a, xy, xz, xy, b, yz, xz, yz, -a - b)).reshape(3, 3)
            shaped = (vertex - center_t) @ torch.matrix_exp(log).T + center_t if joint else vertex
            pose = q[offset:].reshape(3, 6)
            x, y, z = (pose[:, :3] * ROTATION_UNIT_RAD).unbind(dim=1)
            zero = x * 0
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=1).reshape(3, 3, 3)
            corrected_R = torch.matrix_exp(skew) @ rotation
            corrected_t = translation + pose[:, 3:] * TRANSLATION_UNIT_M
            return shaped[None] @ corrected_R.transpose(1, 2) + corrected_t[:, None], theta
        def objective(parameters):
            nonlocal count
            _check_budget(deadline)
            if count >= MAX_OPTIMIZER_CALLS: raise _EvaluationLimit("Frozen50 objective-call budget")
            count += 1
            parameters = np.asarray(parameters, dtype=np.float64)
            limits = np.r_[np.full(offset, SHAPE_BOUND_NORMALIZED), np.full(18, POSE_BOUND_NORMALIZED)]
            if parameters.shape != initial.shape or not np.isfinite(parameters).all() or np.any(np.abs(parameters) > limits):
                raise ValueError("Optimizer parameters exceed finite declared bounds; no clipping")
            q = torch.tensor(parameters, device="cuda", requires_grad=True)
            posed, theta = torch_geometry(q)
            d2 = [observed_to_triangle_distance_squared(p, mesh[face_t]) for p, mesh in zip(targets, posed, strict=True)]
            data = torch.stack([(2 * d / (torch.sqrt(1 + d / ROBUST_TRANSITION_M**2) + 1)).mean() for d in d2]).mean()
            cost = data + (SHAPE_PRIOR * theta.square().sum() if joint else 0)
            cost.backward()
            value, gradient = float(cost.detach()), q.grad.detach().cpu().numpy().copy()
            if not np.isfinite(value) or not np.isfinite(gradient).all(): raise ValueError("Nonfinite nuisance fit objective/gradient")
            history.append((np.asarray(parameters).copy(), value))
            return value, gradient
        bounds = [(-SHAPE_BOUND_NORMALIZED, SHAPE_BOUND_NORMALIZED)] * offset + [(-POSE_BOUND_NORMALIZED, POSE_BOUND_NORMALIZED)] * 18
        success, chosen, message, iterations = False, initial, "", 0
        try:
            result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxfun": MAX_OPTIMIZER_CALLS, "maxiter": MAX_OPTIMIZER_CALLS, "ftol": 1e-12, "gtol": 1e-8})
            success, message, iterations = bool(result.success), str(result.message), int(result.nit)
            if success and count < MAX_OPTIMIZER_CALLS: chosen = np.asarray(result.x).copy()
            else: success = False
        except _EvaluationLimit as exc: message = str(exc)
        _check_budget(deadline)
        shaped, fit_R, fit_t, theta = transform_numpy(chosen, v, center, R, T, joint)
        posed, _ = torch_geometry(torch.tensor(chosen, device="cuda"))
        objective_calls, schur, schur_error, assignment_ambiguous = count, None, None, False
        if joint:
            assignments, assignment_ambiguous = _assignments(torch, posed, face_t, observations)
            count += 1
            columns = []
            for column in range(len(chosen)):
                _check_budget(deadline)
                errors = []
                for sign in (-1, 1):
                    shifted = chosen.copy(); shifted[column] += sign * JACOBIAN_STEP
                    # Bounds are optimizer constraints, not derivative clipping.
                    sv, sr, st, _ = transform_numpy(shifted, v, center, R, T, joint)
                    errors.append(fixed_feature_residuals(sv[None] @ sr.transpose(0, 2, 1) + st[:, None], f, observations, assignments))
                    count += 1
                columns.append((errors[1] - errors[0]) / (2 * JACOBIAN_STEP))
            try: schur = data_schur_from_jacobian(np.array(columns).T)
            except ValueError as exc: schur_error = str(exc)
        count += 1
        _check_budget(deadline)
        held_losses = np.array([float(observed_to_triangle_distance_squared(torch.tensor(p, device="cuda"), mesh[face_t]).mean().detach())
                                for p, mesh in zip(heldout, posed, strict=True)])
        if count > MAX_MODEL_CALLS: raise RuntimeError("Total nested-model evaluation budget exceeded100")
        report = {"model": "M1_shape_pose" if joint else "M0_pose_only", "optimizer_success": success,
                  "optimizer_message": message, "optimizer_iterations": iterations, "objective_calls": objective_calls,
                  "total_model_calls": count, "max_objective_calls": MAX_OPTIMIZER_CALLS, "max_total_model_calls": MAX_MODEL_CALLS,
                  "initial_objective_m2": history[0][1] if history else None,
                  "final_objective_m2": next((cost for q, cost in reversed(history) if np.array_equal(q, chosen)), None),
                  "convergence_failure_kept_initial": not success, "shape_bounds_saturated": bool(joint and np.any(np.abs(chosen[:5]) >= SHAPE_BOUND_NORMALIZED - 1e-8)),
                  "pose_bounds_saturated": bool(np.any(np.abs(chosen[offset:]) >= POSE_BOUND_NORMALIZED - 1e-8)),
                  "data_schur_eigenvalues": np.linalg.eigvalsh(schur).tolist() if schur is not None else None,
                  "data_schur_failure": schur_error, "assignment_boundary_or_tie_detected": assignment_ambiguous,
                  "schur_data_only": True, "schur_fixed_face_and_region": True, "schur_jacobian_central_step_normalized": JACOBIAN_STEP,
                  "schur_statistical_identification_verified": False, "pose_modes_checked": False,
                  "normalized_shape_unit_log": SHAPE_UNIT, "normalized_rotation_unit_rad": ROTATION_UNIT_RAD,
                  "normalized_translation_unit_m": TRANSLATION_UNIT_M, "shape_prior": SHAPE_PRIOR if joint else 0.,
                  "robust_transition_m": ROBUST_TRANSITION_M, "input_poses_oracle_verified": False,
                  "heldout_view_pixels_only": True, "unseen_view_pose_transfer_verified": False, "adoption_authorized": False}
        models.append(PoseShapeModel(theta.copy(), fit_R.copy(), fit_t.copy(), held_losses, report))
    return tuple(models)

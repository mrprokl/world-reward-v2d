"""CPU feasible-start rigid object projection toward a frozen full-T proposal.

The fixed human and unchanged original object surface define SAME anatomical
witness bounds, not physical contact or nonpenetration truth. Optimize six
object DOF about its physical mesh centroid; the full-vertex displacement to the
fixed target has an exact mean/covariance form, without a tuned rotation weight.
Whole-clip acceptance uses all exact witness bounds and decreasing objective.
RGB, floor, rotation/motion retention and PEN remain independent caller gates.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_feasible_placement import _NearestSurface
from .native_pose_sqp import PoseSQPConfig, QPProjectionFailure, project_diagonal_qp, STATE_DIM


@dataclass(frozen=True)
class RigidContactProjectionConfig:
    max_steps: int = 20
    budget_seconds: float = 120.
    max_restorations: int = 4
    rotation_trust_rad: float = .05
    translation_trust_m: float = .01
    numerical_slack_m: float = 1e-7
    development_reference: str = 'authored_two_seed_rigid_feasible_start_not_HOI_calibration'

    def __post_init__(self):
        fixed = dict(max_steps=20, max_restorations=4, rotation_trust_rad=.05,
            translation_trust_m=.01, numerical_slack_m=1e-7,
            development_reference='authored_two_seed_rigid_feasible_start_not_HOI_calibration')
        if (any(type(getattr(self, k)) is not type(v) or getattr(self, k) != v for k, v in fixed.items())
                or type(self.budget_seconds) not in (float, int) or not np.isfinite(self.budget_seconds)
                or not 0 < self.budget_seconds <= 120):
            raise ValueError('Fixed authored numerical policy and bounded CPU runtime required')


@dataclass(frozen=True)
class PoseProjection:
    rotation: np.ndarray
    translation: np.ndarray
    baseline_gaps_m: np.ndarray
    gaps_m: np.ndarray
    diagnostics: dict


class _BudgetExhausted(RuntimeError):
    pass


def _float(value, shape, name):
    if np.ma.isMaskedArray(value): raise ValueError(name + ': masked geometry forbidden')
    a = np.asarray(value)
    if a.dtype not in (np.dtype('float32'), np.dtype('float64')) or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + ': exact finite FP32/64 shape required')
    return a.astype(np.float64, copy=True)


def _pose(r, t, count, name):
    r = _float(r, (count, 3, 3), name + ' rotation'); t = _float(t, (count, 3), name + ' translation')
    if (not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0)):
        raise ValueError(name + ': original native near-SO3 ABI required, no rotation repair')
    return r, t


def _moments(vertices):
    centre = vertices.mean(0); u = vertices - centre
    return centre, u.T @ u / len(u)


def _retract(r, t, step, centre):
    q = Rotation.from_rotvec(step[:, :3]).as_matrix()
    c = np.einsum('tij,j->ti', r, centre) + t
    rotated = q @ r
    translated = c + step[:, 3:] - np.einsum('tij,j->ti', rotated, centre)
    rotated[0], translated[0] = r[0], t[0]
    return rotated, translated


def _object_in_front(vertices, r, t):
    """Every ORIGINAL vertex at every ORIGINAL timestamp, no clipping/deletion."""
    depth = np.einsum('tj,pj->tp', r[:, 2], vertices) + t[:, 2, None]
    return bool(np.isfinite(depth).all() and np.all(depth > 0))


def _objective(centre, covariance, r, t, target_r, target_t):
    """Exactly .5*SUM_t MEAN_v ||R v+t - R_target v-t_target||²."""
    error = r - target_r
    c = np.einsum('tij,j->ti', r, centre) + t
    target_c = np.einsum('tij,j->ti', target_r, centre) + target_t
    value = .5 * (np.einsum('tij,jk,tik->', error, covariance, error) + np.square(c - target_c).sum())
    cross = r @ covariance @ target_r.transpose(0, 2, 1)
    rotation_gradient = np.stack((cross[:, 2, 1] - cross[:, 1, 2],
        cross[:, 0, 2] - cross[:, 2, 0], cross[:, 1, 0] - cross[:, 0, 1]), axis=1)
    gradient = np.c_[rotation_gradient, c - target_c]
    world_cov = r @ covariance @ r.transpose(0, 2, 1)
    diagonal = np.trace(world_cov, axis1=1, axis2=2)[:, None] - np.diagonal(world_cov, axis1=1, axis2=2)
    # A geometry-null rotation has zero gradient. Machine-epsilon conditioning
    # makes the proximal metric positive, without changing the objective.
    diagonal = np.maximum(diagonal, np.finfo(np.float64).eps * max(1., float(np.trace(covariance))))
    metric = np.c_[diagonal, np.ones((len(r), 3))]
    gradient[0] = 0.  # Fixed original time-zero gauge, not frame-wise alignment.
    return float(value), gradient, metric


def _contacts(surface, hands, active, r, t, check):
    gaps = np.full(active.shape, np.nan); anchors = np.full((*active.shape, 3), np.nan)
    faces = np.full(active.shape, -1, np.int64)
    for f, side in zip(*np.nonzero(active)):
        check(); local = (hands[f, side] - t[f]) @ r[f]
        gap, anchor, face = surface.closest(local); check()
        gaps[f, side], anchors[f, side], faces[f, side] = gap, anchor, face
    return gaps, anchors, faces


def _rows(surface, hands, active, r, t, centre, gaps, anchors, faces, radii):
    rows = [[] for _ in r]; bounds = [[] for _ in r]
    c = np.einsum('tij,j->ti', r, centre) + t
    for f, side in zip(*np.nonzero(active)):
        local = (hands[f, side] - t[f]) @ r[f]
        gap = gaps[f, side]
        if gap > 1e-12:
            normals = [(local - anchors[f, side]) / gap]
        else:
            triangle = surface.triangles[faces[f, side]]
            n = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]); length = np.linalg.norm(n)
            if length <= 1e-15:
                raise QPProjectionFailure('Degenerate zero-gap surface cusp; no contact dropped or signed-normal invented')
            normals = [n / length, -n / length]
        for n in normals:
            camera_n = r[f] @ n
            # d p_local = R^T [h-c]_x d_omega - R^T d_centroid.
            # Left rotations preserve the original near-SO3 Gram matrix exactly
            # in real arithmetic; do not independently normalize old predictions.
            rows[f].append(np.r_[np.cross(camera_n, hands[f, side] - c[f]), -camera_n])
            bounds[f].append(radii[f, side] - gap)
    return [np.asarray(a, np.float64).reshape(-1, 6) for a in rows], [np.asarray(b, np.float64) for b in bounds]


def _qp(gradient, metric, rows, bounds, trust, config):
    """Reuse the unchanged numerical dual solver with zero dummy dimensions.

    This is a SIX-DOF rigid QP, not a 133-DOF native-body fit. Dummy columns have
    zero objective/constraint coefficients and their exact minimizer is zero.
    """
    count = len(gradient); g = np.zeros((count, STATE_DIM)); g[:, :6] = gradient
    d = np.ones_like(g); d[:, :6] = metric
    limits = np.ones_like(g)
    limits[:, :6] = trust * np.r_[np.full(3, config.rotation_trust_rad), np.full(3, config.translation_trust_m)]
    embedded = [np.pad(a, ((0, 0), (0, STATE_DIM - 6))) for a in rows]
    cfg = PoseSQPConfig(budget_seconds=config.budget_seconds)
    step, receipt = project_diagonal_qp(g, d, embedded, bounds, limits, config=cfg)
    if np.any(step[:, 6:] != 0): raise ValueError('Zero dummy rigid-QP dimensions unexpectedly moved')
    step[0] = 0.
    return step[:, :6], dict(**receipt, optimized_state_dimensions=6,
        numerical_solver_embedding_dimensions=STATE_DIM, native_body_fitted=False)


def project_rigid_pose_to_contacts(vertices, faces, R_A, t_A, R_target, t_target,
        hand_points_camera, activations, frame_index, *, config=RigidContactProjectionConfig()):
    """Feasible dynamic A -> target descent, exact frozen witness tubes.

    Active hand points are automatically selected OUTSIDE this primitive from A
    and held fixed. Inactive rows are NaN; they do not imply real contact absence.
    No frame is deleted or individually selected for acceptance. Every step and
    restoration is a WHOLE full-T proposal; failed proposals retain the last
    fully feasible dynamic clip. Caller seals then applies independent RGB/QA.
    """
    if type(config) is not RigidContactProjectionConfig: raise ValueError('Explicit rigid projection policy required')
    started = time.monotonic()
    def check():
        if time.monotonic() - started >= config.budget_seconds:
            raise _BudgetExhausted('Bounded rigid projection runtime exhausted; no partial clip accepted')
    if np.ma.isMaskedArray(frame_index) or np.ma.isMaskedArray(activations) or np.ma.isMaskedArray(hand_points_camera):
        raise ValueError('Full unmasked original timeline/activity/witness geometry required')
    frames = np.asarray(frame_index); active = np.asarray(activations)
    if (frames.dtype != np.int64 or frames.ndim != 1 or len(frames) < 3
            or not np.array_equal(frames, np.arange(len(frames)))
            or active.dtype != bool or active.shape != (len(frames), 2)):
        raise ValueError('Full original int64 timeline starting at zero and native boolean activations required')
    count = len(frames); hands = np.asarray(hand_points_camera)
    if (hands.dtype not in (np.dtype('float32'), np.dtype('float64')) or hands.shape != (count, 2, 3)
            or not np.isfinite(hands[active]).all() or not np.isnan(hands[~active]).all()):
        raise ValueError('Same finite active hand witnesses / NaN inactive rows required, never self-disable')
    hands = hands.astype(np.float64, copy=True); active = active.copy()
    original_r, original_t = _pose(R_A, t_A, count, 'Original A')
    target_r, target_t = _pose(R_target, t_target, count, 'Frozen target')
    v = np.asarray(vertices)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v): raise ValueError('Full nonempty original mesh required')
    v = _float(v, v.shape, 'Fixed vertices'); f = np.asarray(faces)
    if (np.ma.isMaskedArray(faces) or f.dtype.kind not in 'iu' or f.ndim != 2 or f.shape[1:] != (3,)
            or not len(f) or np.any(f < 0) or np.any(f >= len(v))): raise ValueError('All original valid mesh triangles required')
    if not _object_in_front(v, original_r, original_t):
        raise ValueError('Original whole object must remain positive camera-depth, no clipped vertices')
    centre, covariance = _moments(v); surface = _NearestSurface(v, f)
    check(); baseline = _contacts(surface, hands, active, original_r, original_t, check)
    radii = baseline[0] + config.numerical_slack_m
    r, t = original_r.copy(), original_t.copy(); current = baseline
    value, _, _ = _objective(centre, covariance, r, t, target_r, target_t); initial_value = value
    attempts = []; accepted = 0; trust = 1.; exhausted = False; surface_passes = 1
    first_reference_seconds = time.monotonic() - started
    for iteration in range(config.max_steps):
        row = dict(iteration=iteration, status='rejected', trust_fraction=trust); attempts.append(row)
        try:
            check(); _, gradient, metric = _objective(centre, covariance, r, t, target_r, target_t)
            rows, bounds = _rows(surface, hands, active, r, t, centre, *current, radii)
            step, row['QP'] = _qp(gradient, metric, rows, bounds, trust, config); check()
            predicted = -float((gradient * step + .5 * metric * step ** 2).sum()); row['predicted_decrease'] = predicted
            if predicted <= 1e-12 * (1 + value): row['status'] = 'no_verified_local_descent'; break
            proposal_r, proposal_t = _retract(r, t, step, centre)
            proposed = _contacts(surface, hands, active, proposal_r, proposal_t, check); surface_passes += 1
            restorations = 0
            while not np.all(proposed[0][active] <= radii[active]) and restorations < config.max_restorations:
                check(); _, _, repair_metric = _objective(centre, covariance, proposal_r, proposal_t, target_r, target_t)
                rows, bounds = _rows(surface, hands, active, proposal_r, proposal_t, centre, *proposed, radii)
                repair, _ = _qp(np.zeros_like(gradient), repair_metric, rows, bounds, trust, config); check()
                proposal_r, proposal_t = _retract(proposal_r, proposal_t, repair, centre)
                proposed = _contacts(surface, hands, active, proposal_r, proposal_t, check); surface_passes += 1; restorations += 1
            row.update(restoration_steps=restorations, maximum_contact_violation_m=
                max(0., float((proposed[0] - radii)[active].max())) if active.any() else 0.)
            if not _object_in_front(v, proposal_r, proposal_t):
                row['status'] = 'whole_object_camera_depth_rejected'; trust *= .5; continue
            if row['maximum_contact_violation_m'] > 0:
                row['status'] = 'exact_contact_rejected'; trust *= .5; continue
            proposal_value, _, _ = _objective(centre, covariance, proposal_r, proposal_t, target_r, target_t)
            row['actual_objective'] = proposal_value
            if proposal_value >= value - 1e-12 * (1 + value): row['status'] = 'objective_rejected'; trust *= .5; continue
            ratio = (value - proposal_value) / predicted
            r, t, current, value = proposal_r, proposal_t, proposed, proposal_value; accepted += 1
            row.update(status='accepted_whole_clip', actual_predicted_ratio=ratio)
            trust = min(1., trust * 2) if ratio > .75 else trust * .5 if ratio < .25 else trust
        except QPProjectionFailure as exc:
            row.update(status='local_QP_unverified', reason=str(exc)[:240], physical_infeasibility_claimed=False); trust *= .5
        except _BudgetExhausted:
            row['status'] = 'budget_exhausted_partial_proposal_discarded'; exhausted = True; break
    if not np.all(current[0][active] <= radii[active]): raise ValueError('Final full-clip witness bounds violated')
    if not _object_in_front(v, r, t): raise ValueError('Final whole object crossed the camera plane')
    if not accepted: r, t = np.asarray(R_A).copy(), np.asarray(t_A).copy()
    return PoseProjection(r, t, baseline[0].copy(), current[0].copy(), dict(
        schema='world_reward.rigid_pose_contact_projection.v1',
        status='accepted_rigid_projection_pending_independent_QA' if accepted else 'dynamic_A_fallback_no_improvement',
        objective_before=initial_value, objective_after=value, objective='half_full_mesh_vertex_mean_square_displacement_to_frozen_target',
        accepted_whole_clip_steps=accepted, attempts=attempts, full_original_frames=count,
        original_active_witness_rows=int(active.sum()), original_frame_indices_preserved=True,
        source_mesh_vertices=len(v), source_mesh_faces=len(f), object_geometry_scale_changed=False,
        exact_full_surface_witness_bounds_preserved=True, original_near_SO3_not_repaired=True,
        first_pose_exactly_preserved=True, witness_selection='caller_frozen_automatic_A_IDs_not_reselected',
        first_full_reference_seconds=first_reference_seconds, surface_passes_completed=surface_passes,
        elapsed_seconds=time.monotonic() - started, budget_exhausted=exhausted, config=asdict(config),
        human_fixed=True, native_fits=0, model_calls=0, GPU_used=False,
        all_original_object_vertices_positive_camera_depth=True,
        ground_truth_used=False, private_truth_read=False, production_adopted=False,
        physical_contact_verified=False, penetration_evaluated=False, whole_hand_minimum_claimed=False,
        RGB_motion_floor_QA_pending=True, heldout_accuracy_verified=False, static_trajectory_substitute=False))


def authored_DEV():
    """Two fixed seeds: moving narrow box, bimanual rotation, noise and release."""
    rows = []
    for seed in (20261010, 20261011):
        rng = np.random.default_rng(seed); count = 33; phase = np.arange(count)
        vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.08, .08) for z in (-.15, .15)], float)
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
        r = Rotation.from_rotvec(np.c_[phase * 0., phase * 0., .07 * np.sin(phase / 9)]).as_matrix()
        t = np.c_[phase * .004, .05 * np.sin(phase / 7), 2. + .015 * np.cos(phase / 6)]
        hands = np.einsum('tij,pj->tpi', r, np.array([[-.52, .015, 0], [.52, -.015, 0]])) + t[:, None]
        active = np.ones((count, 2), bool); active[12:16] = False; hands[~active] = np.nan
        target_r = Rotation.from_rotvec(np.c_[phase * 0., phase * 0., .25 + rng.normal(0., .002, count)]).as_matrix() @ r
        target_t = t + rng.normal(0., .002, (count, 3)); target_r[0], target_t[0] = r[0], t[0]
        result = project_rigid_pose_to_contacts(vertices, faces, r, t, target_r, target_t, hands, active, phase.astype(np.int64))
        d = result.diagnostics
        if (not d['accepted_whole_clip_steps'] or not d['objective_after'] < .4 * d['objective_before']
                or not np.all(result.gaps_m[active] <= result.baseline_gaps_m[active] + 1e-7)
                or np.linalg.norm(result.translation[-1] - result.translation[0]) < .08
                or not np.array_equal(result.rotation[0], r[0]) or not np.array_equal(result.translation[0], t[0])):
            raise ValueError('Authored moving bimanual rigid DEV failed; not a real reconstruction gain')
        rows.append(dict(seed=seed, objective_before=d['objective_before'], objective_after=d['objective_after'],
            accepted_whole_clip_steps=d['accepted_whole_clip_steps'], full_frames=count,
            contact_bounds_passed=True, fixed_human=True, inactive_rows_retained=int((~active).sum())))
    return dict(passed=True, known_authored_geometry=True, rows=rows, ground_truth_challenge_used=False,
        scope='numerical_capacity_not_HOI_calibration_or_actual_accuracy')

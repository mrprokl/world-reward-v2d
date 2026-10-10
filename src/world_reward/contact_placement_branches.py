"""Bounded original/current surface-branch multistart for joint placement.

The closest triangles of two proposed hands select just ONE component of a
nonconvex union. Their translated triangle-radius tubes may be disjoint even
when another component admits a shared correction. Raising Dykstra's iteration
cap cannot fix a disjoint branch. Try the Cartesian product of original and
current triangles (at most four), preserving the existing tolerances/work cap,
then choose the smallest verified camera correction among converged branches.

Original anchors identify triangles, not fixed barycentric points: sliding on a
face and transitions remain allowed. All complete-surface/emitted-translation
checks are retained. Search is bounded, not globally complete; changed proposed
articulation may make even the original branches incompatible. Failure never
proves physical/global infeasibility. No GT, pose/shape/rotation changes, contact
relabeling, smoothing, object deletion, frozen motion or quality claim occurs.
"""
from dataclasses import dataclass
import hashlib
from itertools import product

import numpy as np

from . import contact_feasible_placement as single


def _evidence_identity(evidence):
    if not isinstance(evidence, single.ContactFeasibilityEvidence):
        raise ValueError('Frozen original contact evidence required')
    h = hashlib.sha256()
    for name, value in vars(evidence).items():
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            h.update(str((value.dtype.str, value.shape)).encode()); h.update(value.tobytes())
        else: h.update(repr(value).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class OriginalContactBranches:
    """Caller authenticates A inputs; factory independently rechecks saved gaps."""
    evidence_sha256: str
    face_indices: np.ndarray
    anchor_points_object: np.ndarray

    def __post_init__(self):
        for name in ('face_indices', 'anchor_points_object'):
            object.__setattr__(self, name, single._immutable(getattr(self, name)))


class BranchPlacementFailure(single.ContactFeasibilityFailure):
    def __init__(self, frame_index, maximum_violation_m, branch_attempts):
        super().__init__(frame_index, 'bounded original/current branch search found no converged verified correction '
            '(not a global or physical infeasibility proof)', maximum_violation_m)
        self.branch_attempts = branch_attempts


def freeze_original_branches(evidence, original_witness_points_camera, original_object_translation):
    """Recompute ORIGINAL complete-surface closest triangles, never proposal IDs.

    Inputs are the same independently source-bound A witnesses/T used to create
    evidence. Inactive witnesses remain NaN. Native FP32 rotations are neither
    repaired nor replaced. Factory verification does not authenticate sources.
    """
    digest = _evidence_identity(evidence); count = len(evidence.frame_index)
    raw = np.asarray(original_witness_points_camera)
    if (np.ma.isMaskedArray(original_witness_points_camera) or raw.shape != (count, 2, 3)
            or raw.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(raw[evidence.activations]).all()
            or not np.isnan(raw[~evidence.activations]).all()):
        raise ValueError('Original finite active witnesses and explicitly unknown inactive witnesses required')
    translation = single._float_array(original_object_translation, (count, 3), 'Original object translation')
    surface = single._NearestSurface(evidence.object_vertices, evidence.object_faces)
    faces = np.full((count, 2), -1, np.int64); anchors = np.full((count, 2, 3), np.nan)
    points = raw.astype(float)
    for frame, side in zip(*np.nonzero(evidence.activations)):
        local = (points[frame, side]-translation[frame])@evidence.object_rotation[frame]
        gap, anchor, face = surface.closest(local)
        if gap != evidence.baseline_gap_m[frame, side]:
            raise ValueError('Original witness/surface/gauge differs from frozen baseline gap')
        faces[frame, side] = face; anchors[frame, side] = anchor
    return OriginalContactBranches(digest, faces, anchors)


def _branch_projection(points, bounds, surface, config, frame, initial_faces):
    faces = np.asarray(initial_faces, np.int64); previous = None; total_cycles = 0
    maximum_violation = 0.
    radii = bounds-2*config.convergence_tolerance_m
    for update in range(1, config.max_face_updates+1):
        triangles = surface.triangles[faces]
        if len(points) == 2:
            shifted = triangles-points[:, None]
            lo, hi = shifted.min(1), shifted.max(1)
            difference = np.maximum(np.maximum(lo[0]-hi[1], lo[1]-hi[0]), 0.)
            lower = float(np.linalg.norm(difference)-radii.sum())
            guard = 128*np.finfo(float).eps*max(1., float(np.abs(shifted).max()), float(bounds.max()))
            if lower > guard:
                return None, dict(initial_faces=list(map(int, initial_faces)), final_faces=faces.tolist(),
                    status='convex_branch_disjoint_AABB_certificate', projection_cycles=total_cycles,
                    face_updates=update, maximum_violation_m=lower,
                    branch_separation_lower_bound_m=lower-guard, physical_infeasibility_claimed=False)
        correction, count, converged, maximum_violation = single._dykstra(points, triangles, radii, config)
        total_cycles += count
        measured = [surface.closest(point+correction) for point in points]
        after = np.asarray([r[0] for r in measured]); next_faces = np.asarray([r[2] for r in measured])
        unchanged = np.array_equal(next_faces, faces)
        fixed_point = previous is not None and np.linalg.norm(correction-previous) <= config.convergence_tolerance_m
        if converged and np.all(after <= bounds) and (unchanged or fixed_point):
            return correction, dict(initial_faces=list(map(int, initial_faces)), final_faces=next_faces.tolist(),
                status='converged_complete_surface_verified', projection_cycles=total_cycles, face_updates=update,
                maximum_violation_m=0., physical_infeasibility_claimed=False)
        if unchanged and not converged: break
        previous = correction.copy(); faces = next_faces
    return None, dict(initial_faces=list(map(int, initial_faces)), final_faces=faces.tolist(),
        status='bounded_branch_projection_not_converged', projection_cycles=total_cycles, face_updates=update,
        maximum_violation_m=float(max(0., maximum_violation)), physical_infeasibility_claimed=False)


def project_joint_translations_branches(evidence, proposal_hand_points_camera,
        proposal_human_translation, proposal_object_translation, *, original_branches):
    """Same full-T/equal-split ABI as the original primitive; <=4 branch starts."""
    digest = _evidence_identity(evidence); count = len(evidence.frame_index)
    if (not isinstance(original_branches, OriginalContactBranches)
            or original_branches.evidence_sha256 != digest
            or original_branches.face_indices.shape != (count, 2)
            or original_branches.face_indices.dtype != np.int64
            or (original_branches.face_indices[evidence.activations] < 0).any()
            or (original_branches.face_indices[evidence.activations] >= len(evidence.object_faces)).any()
            or (original_branches.face_indices[~evidence.activations] != -1).any()):
        raise ValueError('Original source/evidence-bound full-T branch inventory required')
    nhand = evidence.source_indices.shape[1]
    points = single._float_array(proposal_hand_points_camera, (count, 2, nhand, 3), 'Proposed complete anatomy')
    human = single._float_array(proposal_human_translation, (count, 3), 'Proposed human translation')
    obj = single._float_array(proposal_object_translation, (count, 3), 'Proposed object translation')
    single._precision_check(evidence.config, points, human, obj, evidence.object_vertices)
    surface = single._NearestSurface(evidence.object_vertices, evidence.object_faces)
    bounds = evidence.baseline_gap_m+evidence.config.numerical_slack_m
    relative = np.zeros((count, 3)); before = np.full((count, 2), np.nan); after = before.copy()
    cycles = np.zeros(count, np.int64); updates = cycles.copy(); selected = np.full((count, 2), -1, np.int64)
    diagnostics = []
    for frame in range(count):
        sides = np.flatnonzero(evidence.activations[frame])
        if not len(sides): continue
        witnesses = points[frame, sides, evidence.witness_columns[frame, sides]]
        rotation = evidence.object_rotation[frame]
        local = (witnesses-obj[frame])@rotation
        measured = [surface.closest(point) for point in local]
        before[frame, sides] = [r[0] for r in measured]
        if np.all(before[frame, sides] <= bounds[frame, sides]):
            after[frame, sides] = before[frame, sides]; continue
        choices = [tuple(dict.fromkeys((measured[i][2], int(original_branches.face_indices[frame, side]))))
            for i, side in enumerate(sides)]
        attempts = []; candidates = []
        for initial in product(*choices):
            correction, row = _branch_projection(local, bounds[frame, sides], surface,
                evidence.config, int(evidence.frame_index[frame]), initial)
            attempts.append(row); cycles[frame] += row['projection_cycles']; updates[frame] += row['face_updates']
            if correction is None: continue
            camera = np.linalg.solve(rotation.T, correction)
            fitted_h = human[frame]+camera/2; fitted_o = obj[frame]-camera/2
            actual = (witnesses+(fitted_h-human[frame])-fitted_o)@rotation
            gaps = np.asarray([surface.closest(point)[0] for point in actual])
            if np.any(gaps > bounds[frame, sides]):
                row.update(status='emitted_translation_bound_failed', maximum_violation_m=float(
                    np.maximum(gaps-bounds[frame, sides], 0.).max())); continue
            row['camera_correction_norm_m'] = float(np.linalg.norm(camera))
            candidates.append((row['camera_correction_norm_m'], tuple(initial), camera, gaps, row))
        if not candidates:
            raise BranchPlacementFailure(evidence.frame_index[frame],
                min(r['maximum_violation_m'] for r in attempts), attempts)
        _, initial, camera, gaps, chosen = min(candidates, key=lambda c: (c[0], c[1]))
        relative[frame] = camera; after[frame, sides] = gaps; selected[frame, sides] = initial
        diagnostics.append(dict(frame_index=int(evidence.frame_index[frame]), active_hands=sides.tolist(),
            branches_attempted=len(attempts), branches_feasible=len(candidates), selected_initial_faces=list(initial),
            current_nearest_faces=[r[2] for r in measured], attempts=attempts))
    hd, od = relative/2, -relative/2
    arrays = dict(frame_index=evidence.frame_index, activations=evidence.activations,
        human_translation_camera=human+hd, object_translation_camera=obj+od,
        human_correction_camera=hd, object_correction_camera=od, relative_correction_camera=relative,
        witness_gap_before_m=before, witness_gap_after_m=after, frozen_gap_bound_m=bounds,
        projection_cycles=cycles, face_updates=updates, selected_initial_face_indices=selected)
    result = {key: single._immutable(value) for key, value in arrays.items()}
    result.update(schema='world_reward.contact_feasible_placement.v1',
        branch_solver_schema='world_reward.contact_placement_branches.v1', max_initial_branches=4,
        original_branch_evidence_sha256=digest, branch_diagnostics=diagnostics,
        constraint_scope='original_frozen_anatomical_witness_not_whole_hand_score',
        minimum_change_scope='smallest_verified_camera_correction_among_converged_bounded_branch_starts',
        object_rotation_orthogonality_error=float(np.abs(evidence.object_rotation@
            evidence.object_rotation.transpose(0, 2, 1)-np.eye(3)).max()), fixed_object_rotation_repaired=False,
        numerical_interior_margin_m=2*evidence.config.convergence_tolerance_m,
        global_minimum_verified=False, physical_contact_verified=False, temporal_quality_verified=False,
        whole_hand_min_evaluated=False, source_reference=evidence.source_reference,
        witness_selection_reference=evidence.witness_selection_reference,
        original_whole_hand_minimum_recomputed=evidence.original_whole_hand_minimum_recomputed,
        original_whole_hand_minimum_attested=evidence.original_whole_hand_minimum_attested,
        original_whole_hand_minimum_nondegradation_bound=evidence.original_whole_hand_minimum_attested,
        development_reference=evidence.config.development_reference)
    return result

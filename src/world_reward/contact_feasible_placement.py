"""Local minimum-change joint translation subject to frozen contact-gap bounds.

This is an experimental numerical primitive, NOT a physical contact estimator,
native CARI4D parity, or a validated reconstruction improvement. Original native
activations must be qualified automatically *before* a proposal is fitted. For
each active hand, an ORIGINAL automatic anatomical witness is frozen. Exact
whole-hand/continuous-triangle minimum selection is optional and must be marked
separately from approximate original QA selection. Its original continuous gap
plus a declared numerical slack is a hard bound; neither its ID nor its bound
is selected from the proposal. An arbitrary preserved witness bounds only its
own gap, NOT nondegradation of the original whole-hand minimum.

For a relative correction ``d = dh - do``, the equal Euclidean minimum-change
split is ``dh = d/2, do = -d/2``. Each current triangle plus a distance-radius
ball is convex. Dykstra projects the origin onto the intersection of at most
two such translated tubes, then updates nearest-face branches. Barycentric
anchors are NOT fixed: tangential sliding and transitions between triangles
remain possible. Final distances use the complete original triangle surface.

The union of triangle tubes is nonconvex: branch-local convergence does not
establish a global minimum, and failure does not prove physical infeasibility.
Branch projection minimizes object-frame Euclidean correction. For a truly
orthonormal rotation that is also the camera-frame Euclidean metric. Native
float32 near-rotations are accepted at the existing 1e-5 ABI tolerance without
repair; their exact inverse transports the correction, and the metric mismatch
is reported rather than pretending exact world-space optimality.
An unchanged original anatomical witness can overconstrain articulation/regrasp.
Unsigned proximity does not establish contact, friction or nonpenetration. RGB,
reserved evidence, full-hand distance, penetration and temporal/motion quality
need independent external gates before a producer may adopt this primitive.
There is no temporal smoothing, static substitute, rotation/shape/scale fitting,
GT, manual label, object deletion or oracle in this module.

Joint contact refinement is motivated by Cao et al., arXiv:2012.09856, and
Open-CHOIR, arXiv:2605.20992v4. The *hard* nondegradation projection here is an
own hypothesis, not an algorithm or guarantee attributed to either paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .sequence_pose import _ContactTriangleSurface


def _immutable(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class ContactPlacementConfig:
    """Externally declared numerical tolerances/work caps, not contact weights.

    ``activation_distance_m`` must be the upstream original native gate (5 cm
    in the audited optimizer), not adjusted to enable additional contacts.
    Slack is numerical, not a claim about measurement accuracy. Projection uses
    a conservative interior margin of twice the convergence tolerance so the
    final *complete-surface* verifier never silently adds a second tolerance.
    """
    activation_distance_m: float
    numerical_slack_m: float
    convergence_tolerance_m: float
    max_projection_cycles: int
    max_face_updates: int
    development_reference: str

    def __post_init__(self):
        values = (self.activation_distance_m, self.numerical_slack_m,
                  self.convergence_tolerance_m)
        if (any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in values)
                or self.convergence_tolerance_m > self.numerical_slack_m/8
                or type(self.max_projection_cycles) is not int
                or not 1 <= self.max_projection_cycles <= 10000
                or type(self.max_face_updates) is not int or not 1 <= self.max_face_updates <= 32
                or type(self.development_reference) is not str
                or not self.development_reference.strip()):
            raise ValueError('Positive declared tolerances, bounded work and external reference required')


@dataclass(frozen=True)
class ContactFeasibilityEvidence:
    """Factory-produced immutable original evidence; no source authentication.

    Camera/world inputs use one metre gauge; object vertices are already in the
    fixed clip-scale object frame. The caller must establish automatic native
    provenance, no GT, and that the supplied chronological range is complete.
    Inactive hands have witness -1 and gap NaN, not an inferred contact absence.
    """
    frame_index: np.ndarray
    activations: np.ndarray
    source_indices: np.ndarray
    witness_columns: np.ndarray
    witness_source_indices: np.ndarray
    baseline_gap_m: np.ndarray
    object_rotation: np.ndarray
    object_vertices: np.ndarray
    object_faces: np.ndarray
    config: ContactPlacementConfig
    source_reference: str
    witness_selection_reference: str
    original_whole_hand_minimum_recomputed: bool
    original_whole_hand_minimum_attested: bool

    def __post_init__(self):
        for name in ('frame_index', 'activations', 'source_indices', 'witness_columns',
                     'witness_source_indices', 'baseline_gap_m', 'object_rotation',
                     'object_vertices', 'object_faces'):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


class ContactFeasibilityFailure(RuntimeError):
    """Reject the complete proposal on a numerical/local work-cap failure.

    This is NOT a certificate that the actual interaction or even the full
    nonconvex translation problem is infeasible. Never omit that frame/hand.
    """
    def __init__(self, frame_index, reason, maximum_violation_m):
        self.frame_index = int(frame_index)
        self.reason = reason
        self.maximum_violation_m = float(maximum_violation_m)
        super().__init__(f'Frame {self.frame_index}: {reason}; '
                         f'maximum local gap violation {self.maximum_violation_m:.9g} m')


def _float_array(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{name}: masked geometry forbidden')
    a = np.asarray(value)
    if a.dtype not in (np.dtype('float32'), np.dtype('float64')) or a.shape != shape:
        raise ValueError(f'{name}: exact shape and float32/64 geometry required')
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: full chronological finite predicted geometry required')
    return a.astype(np.float64, copy=True)


def _precision_check(config, *values):
    magnitude = max(1., *(float(np.abs(v).max()) for v in values))
    if 128*np.finfo(float).eps*magnitude > config.convergence_tolerance_m/8:
        raise ValueError('Coordinate magnitude is insufficiently conditioned for the declared tolerance')


def _closest_triangle_points(point, triangles):
    """Closest points on every closed triangle, retaining degenerate segments.

    Matches the audited native unsigned distance semantics: contained planar
    projection or a closed-edge projection, not nearest vertices or sampled
    surface points. Arithmetic is float64, not an exact-arithmetic certificate.
    """
    edges = np.roll(triangles, -1, axis=1)-triangles
    delta = point[None, None]-triangles
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        edge2 = np.einsum('fki,fki->fk', edges, edges)
        normal = np.cross(edges[:, 0], -edges[:, 2])
        normal2 = np.einsum('fi,fi->f', normal, normal)
        parameter = np.divide(np.einsum('fki,fki->fk', delta, edges), edge2,
                              out=np.zeros_like(edge2), where=edge2 > 0)
        segment = triangles+np.clip(parameter, 0., 1.)[..., None]*edges
        segment_delta = point[None, None]-segment
        segment2 = np.einsum('fki,fki->fk', segment_delta, segment_delta)
        winner = segment2.argmin(axis=1)
        closest = segment[np.arange(len(triangles)), winner].copy()
        squared = segment2[np.arange(len(triangles)), winner].copy()
        height = np.einsum('fi,fi->f', point[None]-triangles[:, 0], normal)
        ratio = np.divide(height, normal2, out=np.zeros_like(height), where=normal2 > 0)
        planar = point[None]-ratio[:, None]*normal
        signs = np.einsum('fki,fi->fk', np.cross(edges, planar[:, None]-triangles), normal)
        inside = (signs >= 0).all(axis=1) & (normal2 > 0)
        planar_delta = point[None]-planar
        planar2 = np.einsum('fi,fi->f', planar_delta, planar_delta)
        use = inside & (planar2 < squared)
        closest[use] = planar[use]; squared[use] = planar2[use]
    if (not np.isfinite(closest).all() or not np.isfinite(squared).all()
            or np.any(squared < 0) or not np.isfinite(normal2).all()
            or np.any(np.any(normal != 0, axis=1) & (normal2 == 0))):
        raise ValueError('Continuous contact projection exceeds finite representable arithmetic')
    return closest, squared


class _NearestSurface:
    """Reuse the qualified conservative broadphase; no stale nearest-face cache."""
    def __init__(self, vertices, faces):
        self.distance_surface = _ContactTriangleSurface(vertices, faces)
        self.triangles = self.distance_surface.starts

    def closest(self, point, *, broadphase=True):
        candidates = (self.distance_surface._candidate_faces(point) if broadphase
                      else self.distance_surface._all_faces)
        best, best_point, best_face = np.inf, None, -1
        for first in range(0, len(candidates), 2048):
            ids = candidates[first:first+2048]
            closest, squared = _closest_triangle_points(point, self.triangles[ids])
            index = int(squared.argmin())
            # Sorted original face IDs and strict comparison give deterministic
            # equal-distance ties. No triangle/vertex is deleted or reduced.
            if squared[index] < best:
                best, best_point, best_face = squared[index], closest[index].copy(), int(ids[index])
        if best_point is None:
            raise ValueError('The complete fixed object surface must be nonempty')
        return float(np.sqrt(best)), best_point, best_face


def _validated_layout(source_indices, original_object_rotation, original_object_translation,
                      object_vertices, object_faces, native_activations, frame_index,
                      count, joints, config, source_reference):
    if not isinstance(config, ContactPlacementConfig):
        raise ValueError('A frozen contact-placement configuration is required')
    if type(source_reference) is not str or not source_reference.strip():
        raise ValueError('Original automatic native evidence provenance is required')
    raw = (source_indices, object_vertices, object_faces, native_activations, frame_index)
    if any(np.ma.isMaskedArray(value) for value in raw):
        raise ValueError('Explicit unmasked native geometry, gates and chronology required')
    ids, vertices, faces, active, frames = map(np.asarray, raw)
    rotation = _float_array(original_object_rotation, (count, 3, 3), 'Original object rotation')
    translation = _float_array(original_object_translation, (count, 3), 'Original object translation')
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)):
        raise ValueError('Complete fixed object vertices required')
    vertices = _float_array(vertices, vertices.shape, 'Original object vertices')
    if (faces.dtype.kind not in 'iu' or faces.ndim != 2 or faces.shape[1:] != (3,)
            or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices))
            or active.dtype != np.bool_ or active.shape != (count, 2)
            or ids.dtype.kind not in 'iu' or ids.shape != (2, joints) or np.any(ids < 0)
            or np.max(ids) > np.iinfo(np.int64).max
            or any(len(np.unique(side)) != joints for side in ids)
            or frames.dtype.kind not in 'iu' or frames.shape != (count,)
            or np.any(frames < 0) or np.max(frames) > np.iinfo(np.int64).max
            or not np.all(np.diff(frames.astype(np.int64)) == 1)):
        raise ValueError('Exact native boolean gates, unique anatomical IDs, fixed faces and full chronology required')
    if (not np.allclose(rotation@rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError('Fixed object rotations must be proper near-orthonormal native matrices')
    _precision_check(config, vertices, translation)
    return ids, vertices, faces.astype(np.int64), active, frames, rotation, translation


def freeze_contact_evidence(original_hand_points_camera, source_indices,
                            original_object_rotation, original_object_translation,
                            object_vertices, object_faces, native_activations,
                            frame_index, *, config, source_reference):
    """Audit path: freeze original exact whole-hand minimum before proposals.

    This visits every supplied native hand vertex with a conservative exact
    triangle broadphase. It is NOT required in the producer hot path: callers
    with already-qualified native minimum witnesses should instead use
    ``freeze_contact_witnesses``. Activations must be original positive native
    evidence AND original proximity, not a distance-only gate invented here.
    The per-frame anatomical ID is frozen; its fitted position is NOT frozen.
    """
    if np.ma.isMaskedArray(original_hand_points_camera):
        raise ValueError('Explicit unmasked original hand geometry required')
    points = np.asarray(original_hand_points_camera)
    if (points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 3
            or points.shape[0] < 1 or points.shape[2] < 1):
        raise ValueError('Complete chronological hand anatomy must have shape [T,2,J,3]')
    count, _, joints, _ = points.shape
    points = _float_array(points, points.shape, 'Original hand points')
    ids, vertices, faces, active, frames, rotation, translation = _validated_layout(
        source_indices, original_object_rotation, original_object_translation,
        object_vertices, object_faces, native_activations, frame_index,
        count, joints, config, source_reference)
    _precision_check(config, points)
    surface = _NearestSurface(vertices, faces.astype(np.int64))
    columns = np.full((count, 2), -1, np.int64)
    witnesses = np.full((count, 2), -1, np.int64)
    baseline = np.full((count, 2), np.nan)
    for frame in range(count):
        if not active[frame].any():
            continue
        local = (points[frame]-translation[frame])@rotation[frame]
        distances = surface.distance_surface.distances(local.reshape(-1, 3), batch_size=32)
        distances = distances.reshape(2, joints)
        for side in np.flatnonzero(active[frame]):
            column = int(np.lexsort((ids[side], distances[side]))[0])
            gap = float(distances[side, column])
            if gap >= config.activation_distance_m:
                raise ValueError('An active original native hand fails its original proximity gate; do not self-disable')
            columns[frame, side] = column; witnesses[frame, side] = ids[side, column]
            baseline[frame, side] = gap
    return ContactFeasibilityEvidence(frames.astype(np.int64), active, ids.astype(np.int64),
        columns, witnesses, baseline, rotation, vertices, faces.astype(np.int64), config,
        source_reference, 'recomputed_original_exact_whole_native_hand_to_all_triangles', True, True)


def freeze_contact_witnesses(original_witness_points_camera, source_indices,
                            original_witness_source_indices, original_object_rotation,
                            original_object_translation, object_vertices, object_faces,
                            native_activations, frame_index, *, config, source_reference,
                            witness_selection_reference,
                            original_selection_is_whole_hand_minimum=False):
    """Hot path: freeze original automatic witness IDs, query <=2 points/T.

    IDs must come from ORIGINAL PRE-FIT automatic native/QA selection, never a
    proposal minimum or manual label. An original nearest-object-vertex QA
    witness is allowed but is NOT relabeled as a continuous whole-hand minimum.
    Only set ``original_selection_is_whole_hand_minimum`` for an independently
    authenticated original native whole-hand/ALL-triangle argmin. This function
    cannot verify that selection/provenance and marks it caller-attested.
    It DOES independently recompute each preserved witness's original distance
    on the complete fixed surface, so supplied gap numbers are never trusted.
    Arbitrary QA witnesses can lie beyond the native proximity gate even when
    another original hand vertex passes that gate: their own gap remains the
    constraint, never proof of native activation or whole-hand nondegradation.
    Inactive witness IDs must be -1 and positions NaN; no frames are skipped.
    Full proposed anatomy still uses the original complete source-ID ordering.
    """
    raw = (original_witness_points_camera, original_witness_source_indices, source_indices)
    if any(np.ma.isMaskedArray(value) for value in raw):
        raise ValueError('Explicit unmasked original automatic native witnesses required')
    points, witnesses, source_ids = map(np.asarray, raw)
    if (points.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or points.ndim != 3 or points.shape[1:] != (2, 3) or len(points) < 1
            or source_ids.ndim != 2 or source_ids.shape[0] != 2 or source_ids.shape[1] < 1
            or type(witness_selection_reference) is not str or not witness_selection_reference.strip()
            or type(original_selection_is_whole_hand_minimum) is not bool):
        raise ValueError('Original [T,2,3] witnesses and authenticated native minimum selection reference required')
    count = len(points); joints = source_ids.shape[1]
    ids, vertices, faces, active, frames, rotation, translation = _validated_layout(
        source_ids, original_object_rotation, original_object_translation, object_vertices,
        object_faces, native_activations, frame_index, count, joints, config, source_reference)
    if (witnesses.dtype.kind != 'i' or witnesses.shape != (count, 2)
            or not np.isfinite(points[active]).all() or not np.isnan(points[~active]).all()
            or np.any(witnesses[active] < 0) or np.any(witnesses[~active] != -1)):
        raise ValueError('Every active original witness needs a known ID/finite point; inactive is explicitly unknown')
    points = points.astype(np.float64, copy=True)
    if active.any():
        _precision_check(config, points[active])
    columns = np.full((count, 2), -1, np.int64); baseline = np.full((count, 2), np.nan)
    surface = _NearestSurface(vertices, faces)
    for frame in range(count):
        for side in np.flatnonzero(active[frame]):
            matched = np.flatnonzero(ids[side] == witnesses[frame, side])
            if len(matched) != 1:
                raise ValueError('Original witness ID must belong to that exact native hand pool')
            gap = surface.closest((points[frame, side]-translation[frame])@rotation[frame])[0]
            if original_selection_is_whole_hand_minimum and gap >= config.activation_distance_m:
                raise ValueError('An active original witness fails its original native proximity gate; do not self-disable')
            columns[frame, side] = matched[0]; baseline[frame, side] = gap
    return ContactFeasibilityEvidence(frames.astype(np.int64), active, ids.astype(np.int64),
        columns, witnesses.astype(np.int64), baseline, rotation, vertices, faces, config,
        source_reference, witness_selection_reference, False,
        original_selection_is_whole_hand_minimum)


def _triangle_gap(point, triangle):
    closest, squared = _closest_triangle_points(point, triangle[None])
    return float(np.sqrt(squared[0])), closest[0]


def _dykstra(points, triangles, radii, config):
    """Origin projection on <=2 convex translated triangle-radius tubes."""
    correction = np.zeros(3); dual = np.zeros_like(points)
    tolerance = config.convergence_tolerance_m
    for cycle in range(1, config.max_projection_cycles+1):
        maximum_change = 0.
        for index, (point, triangle, radius) in enumerate(zip(points, triangles, radii)):
            value = correction+dual[index]
            gap, closest = _triangle_gap(point+value, triangle)
            projected = value.copy()
            if gap > radius:
                projected -= (1.-radius/gap)*(point+value-closest)
            dual[index] = value-projected
            maximum_change = max(maximum_change, float(np.linalg.norm(projected-correction)))
            correction = projected
        violations = np.array([_triangle_gap(point+correction, triangle)[0]-radius
                               for point, triangle, radius in zip(points, triangles, radii)])
        if not np.isfinite(correction).all() or not np.isfinite(violations).all():
            raise ValueError('Local contact projection is not finitely representable')
        if maximum_change <= tolerance and violations.max() <= tolerance:
            return correction, cycle, True, float(max(0., violations.max()))
    return correction, config.max_projection_cycles, False, float(max(0., violations.max()))


def _relative_projection(points, bounds, surface, config, original_frame):
    distances, _, faces = zip(*(surface.closest(point) for point in points))
    before = np.asarray(distances)
    if np.all(before <= bounds):
        return np.zeros(3), before, before.copy(), 0, 0
    faces = np.asarray(faces, np.int64)
    radii = bounds-2*config.convergence_tolerance_m
    previous = None; cycles = 0; maximum_violation = np.inf
    for update in range(1, config.max_face_updates+1):
        correction, count, converged, maximum_violation = _dykstra(
            points, surface.triangles[faces], radii, config)
        cycles += count
        distances, _, next_faces = zip(*(surface.closest(point+correction) for point in points))
        after = np.asarray(distances); next_faces = np.asarray(next_faces, np.int64)
        # A converged convex-branch projection must pass the complete-surface
        # bounds, WITHOUT adding tolerance again. Failure rejects the full clip.
        if converged and np.all(after <= bounds):
            unchanged = np.array_equal(next_faces, faces)
            fixed_point = previous is not None and np.linalg.norm(correction-previous) <= config.convergence_tolerance_m
            if unchanged or fixed_point:
                return correction, before, after, cycles, update
        if np.array_equal(next_faces, faces) and not converged:
            raise ContactFeasibilityFailure(original_frame,
                'local convex projection did not converge (not an infeasibility proof)', maximum_violation)
        previous = correction.copy(); faces = next_faces
    raise ContactFeasibilityFailure(original_frame,
        'nearest-face work cap reached without verified branch-local convergence', maximum_violation)


def project_joint_translations(evidence, proposal_hand_points_camera,
                               proposal_human_translation, proposal_object_translation):
    """Place BOTH actors without increasing frozen active original gap bounds.

    Hand positions must be the actual proposed native anatomical geometry, in
    the same ordering as original IDs. An added human correction translates the
    entire actor, not individual fingers. Object rotations and full geometry are
    those frozen in ``evidence``; do not pass a changed rotation/shape/scale.
    All original frames are returned. Inactive release frames are exactly the
    proposed translations. Per-frame corrections may worsen temporal/image or
    penetration quality; a downstream producer MUST independently reject that.

    Bounding a preserved witness upper-bounds the final whole-hand minimum by
    that witness's bound. This is original whole-hand NONDEGRADATION only if the
    witness was independently attested/recomputed as its ORIGINAL exact minimum.
    Otherwise it guarantees only the same original QA witness, not minimum
    nondegradation. Returned gaps never constitute a whole-hand contact score.
    The function does not authenticate data sources or report a metric gain.
    """
    if not isinstance(evidence, ContactFeasibilityEvidence):
        raise ValueError('Frozen original contact evidence required')
    count, _ = evidence.activations.shape
    joints = evidence.source_indices.shape[1]
    points = _float_array(proposal_hand_points_camera, (count, 2, joints, 3), 'Proposed hand points')
    human = _float_array(proposal_human_translation, (count, 3), 'Proposed human translation')
    obj = _float_array(proposal_object_translation, (count, 3), 'Proposed object translation')
    _precision_check(evidence.config, points, human, obj, evidence.object_vertices)
    surface = _NearestSurface(evidence.object_vertices, evidence.object_faces)
    relative = np.zeros((count, 3)); before = np.full((count, 2), np.nan)
    after = np.full((count, 2), np.nan); cycles = np.zeros(count, np.int64)
    updates = np.zeros(count, np.int64)
    bounds = evidence.baseline_gap_m+evidence.config.numerical_slack_m
    for frame in range(count):
        active_sides = np.flatnonzero(evidence.activations[frame])
        if not len(active_sides):
            continue
        columns = evidence.witness_columns[frame, active_sides]
        witnesses = points[frame, active_sides, columns]
        local = (witnesses-obj[frame])@evidence.object_rotation[frame]
        change, initial, final, ncycles, nupdates = _relative_projection(local,
            bounds[frame, active_sides], surface, evidence.config, evidence.frame_index[frame])
        # Preserve the ORIGINAL native matrix, including FP32 roundoff. A
        # transpose would not exactly invert that matrix and could defeat a
        # tighter numerical contact bound. No polar projection/repair here.
        relative[frame] = np.linalg.solve(evidence.object_rotation[frame].T, change)
        before[frame, active_sides] = initial; after[frame, active_sides] = final
        cycles[frame] = ncycles; updates[frame] = nupdates
    human_delta = relative/2; object_delta = -relative/2
    fitted_human, fitted_obj = human+human_delta, obj+object_delta
    if not np.isfinite(fitted_human).all() or not np.isfinite(fitted_obj).all():
        raise ValueError('Joint translations exceed finite representable arithmetic')
    # Re-evaluate the ACTUAL emitted actor translations, not merely local d:
    # finite-precision cancellation between actor coordinates can differ.
    for frame in range(count):
        for side in np.flatnonzero(evidence.activations[frame]):
            column = evidence.witness_columns[frame, side]
            actual_human_shift = fitted_human[frame]-human[frame]
            local = (points[frame, side, column]+actual_human_shift-fitted_obj[frame])@evidence.object_rotation[frame]
            gap = surface.closest(local)[0]
            if gap > bounds[frame, side]:
                raise ContactFeasibilityFailure(evidence.frame_index[frame],
                    'emitted joint translations violate the frozen complete-surface bound', gap-bounds[frame, side])
            after[frame, side] = gap
    arrays = dict(frame_index=evidence.frame_index, activations=evidence.activations,
        human_translation_camera=fitted_human, object_translation_camera=fitted_obj,
        human_correction_camera=human_delta, object_correction_camera=object_delta,
        relative_correction_camera=relative, witness_gap_before_m=before,
        witness_gap_after_m=after, frozen_gap_bound_m=bounds,
        projection_cycles=cycles, face_updates=updates)
    result = {key: _immutable(value) for key, value in arrays.items()}
    result.update(schema='world_reward.contact_feasible_placement.v1',
        constraint_scope='original_frozen_anatomical_witness_not_whole_hand_score',
        minimum_change_scope='equal_joint_split_for_branch_local_object_frame_projection',
        object_rotation_orthogonality_error=float(np.abs(
            evidence.object_rotation@evidence.object_rotation.transpose(0, 2, 1)-np.eye(3)).max()),
        fixed_object_rotation_repaired=False,
        numerical_interior_margin_m=2*evidence.config.convergence_tolerance_m,
        global_minimum_verified=False, physical_contact_verified=False,
        temporal_quality_verified=False, whole_hand_min_evaluated=False,
        source_reference=evidence.source_reference,
        witness_selection_reference=evidence.witness_selection_reference,
        original_whole_hand_minimum_recomputed=evidence.original_whole_hand_minimum_recomputed,
        original_whole_hand_minimum_attested=evidence.original_whole_hand_minimum_attested,
        original_whole_hand_minimum_nondegradation_bound=evidence.original_whole_hand_minimum_attested,
        development_reference=evidence.config.development_reference)
    return result

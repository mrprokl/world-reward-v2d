"""Fixed-shape RGB, or explicit same-gauge RGB/depth, bundle adjustment.

No physics snapping, scale/camera/shape fitting, mask filling or zero-velocity
prior. Pose estimates are correlated proposals, not calibrated measurements.
All frames participate; hidden motion remains inferred and is not identifiable
without evidence. Parameters must be frozen on non-challenge development data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp
from .depth_covariance import CommonModeDepthConfig, FrameDepthWhitening
from .contact_patch import ContactPatchConfig, smooth_patch_distances


@dataclass(frozen=True)
class SequencePoseConfig:
    pixel_sigma: float
    prior_centroid_sigma_diameter: float
    prior_rotation_sigma_rad: float
    acceleration_sigma_diameter_s2: float
    angular_acceleration_sigma_rad_s2: float
    prior_weight: float
    temporal_weight: float
    max_nfev: int
    development_reference: str

    def __post_init__(self):
        values = [getattr(self, k) for k in asdict(self) if k not in
                  ('max_nfev', 'development_reference')]
        if (any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in values)
                or type(self.max_nfev) is not int or not 1 <= self.max_nfev <= 300
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Positive externally frozen scales/weights/budget required')


@dataclass(frozen=True)
class RGBDepthConfig:
    """Global depth scale frozen externally; not learned-depth calibration.

    Depth is camera-axis z in the same metre gauge as the fixed mesh/poses.
    A caller must authenticate its RGB-derived source and gauge. No source
    calibration or reference trajectories may be supplied to challenge runs.
    """
    depth_sigma_diameter: float
    development_reference: str

    def __post_init__(self):
        if (type(self.depth_sigma_diameter) not in (int, float)
                or not np.isfinite(self.depth_sigma_diameter) or self.depth_sigma_diameter <= 0
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Positive externally frozen depth scale and reference required')


@dataclass(frozen=True)
class SequenceContactConfig:
    """Externally frozen anatomical contact scale and CPU work cap."""
    contact_sigma_diameter: float
    max_points_per_hand: int
    max_point_triangle_pairs: int
    development_reference: str

    def __post_init__(self):
        if (type(self.contact_sigma_diameter) not in (int, float)
                or not np.isfinite(self.contact_sigma_diameter) or self.contact_sigma_diameter <= 0
                or type(self.max_points_per_hand) is not int or not 1 <= self.max_points_per_hand <= 8
                or type(self.max_point_triangle_pairs) is not int or self.max_point_triangle_pairs <= 0
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Positive externally frozen contact scale, <=8 witnesses and work cap required')


@dataclass(frozen=True)
class SequenceContactEvidence:
    """Caller-qualified automatic activations, frozen before pose optimization.

    True means the caller qualified contact from independent automatic evidence
    (for example positive native logit AND original anatomical proximity). False
    is no active contact constraint, not proof of physical absence. Hand points
    are predicted anatomical vertices in the same camera/metre frame as poses.
    This module cannot authenticate the source or certify contact/gauge accuracy.
    """
    activations: np.ndarray
    hand_points_camera: np.ndarray
    hand_visible: np.ndarray
    object_faces: np.ndarray
    source_reference: str

    def __post_init__(self):
        activations, points, visible, faces = map(np.asarray,
            (self.activations, self.hand_points_camera, self.hand_visible, self.object_faces))
        if (any(np.ma.isMaskedArray(value) for value in
                (self.activations, self.hand_points_camera, self.hand_visible, self.object_faces))
                or activations.dtype != np.bool_ or activations.ndim != 2 or activations.shape[1:] != (2,)
                or points.dtype.kind != 'f' or points.ndim != 4 or points.shape[:2] != activations.shape
                or points.shape[-1] != 3 or points.shape[2] < 1
                or visible.dtype != np.bool_ or visible.shape != points.shape[:-1]
                or not np.isfinite(points[visible]).all() or not np.isnan(points[~visible]).all()
                or faces.dtype.kind not in 'iu' or faces.ndim != 2 or faces.shape[1:] != (3,)
                or not len(faces) or np.any(faces < 0)
                or type(self.source_reference) is not str or not self.source_reference.strip()):
            raise ValueError('Frozen boolean activations, exact supported anatomical points, faces and source required')
        for name, value in (('activations', activations), ('hand_points_camera', points),
                            ('hand_visible', visible), ('object_faces', faces)):
            value = np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, value)


class _ContactTriangleSurface:
    """Continuous unsigned surface distance with conservative exact broadphase.

    Enumerates contained planar projections and every closed edge. Degenerate
    triangles contribute their segments/vertices, never disappear. Closest-face
    ties are nonsmooth; sparse numerical differentiation does not change that.
    Triangle balls and a nearest referenced vertex bound prune only faces that
    cannot beat the upper bound. All faces remain stored; unsafe bounds use the
    complete surface. This is float64 geometry, not an exact-arithmetic proof.
    """
    def __init__(self, vertices, faces):
        if np.any(faces >= len(vertices)):
            raise ValueError('Contact faces must index the same complete fixed mesh')
        triangles = vertices[faces]
        self.starts = triangles
        self.edges = np.roll(triangles, -1, axis=1)-triangles
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            self.edge2 = np.einsum('fki,fki->fk', self.edges, self.edges)
            self.normal = np.cross(self.edges[:, 0], -self.edges[:, 2])
            self.normal2 = np.einsum('fi,fi->f', self.normal, self.normal)
        if (not np.isfinite(self.edges).all() or not np.isfinite(self.edge2).all()
                or not np.isfinite(self.normal2).all()
                or np.any(np.any(self.normal != 0, axis=1) & (self.normal2 == 0))):
            raise ValueError('Fixed contact triangle arithmetic exceeds finite representable range')
        self._all_faces = np.arange(len(triangles), dtype=np.int64)
        self._centre_tree = self._vertex_tree = None
        self._maximum_radius = np.inf
        self._coordinate_magnitude = float(np.abs(triangles).max())
        # Unreferenced vertices are NOT on the surface and cannot bound its
        # distance. Index only vertices appearing in at least one original face.
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            centres = (triangles/3).sum(axis=1)
            radius = np.linalg.norm(triangles-centres[:, None], axis=2).max(axis=1)
        if np.isfinite(centres).all() and np.isfinite(radius).all():
            try:
                self._centre_tree = cKDTree(centres, copy_data=True)
                self._vertex_tree = cKDTree(vertices[np.unique(faces)], copy_data=True)
                self._maximum_radius = float(radius.max())
            except (ValueError, OverflowError):
                self._centre_tree = self._vertex_tree = None

    def _candidate_faces(self, point):
        """Every possibly winning face, or full mesh on any invalid bound."""
        if self._centre_tree is None or self._vertex_tree is None:
            return self._all_faces
        try:
            upper, _ = self._vertex_tree.query(point, k=1, eps=0)
            magnitude = max(self._coordinate_magnitude, float(np.abs(point).max()),
                            float(upper), self._maximum_radius, np.finfo(float).tiny)
            guard = 128*np.finfo(float).eps*magnitude
            bound = np.nextafter(float(upper)+self._maximum_radius+guard, np.inf)
            if not np.isfinite(bound) or bound < 0:
                return self._all_faces
            candidates = self._centre_tree.query_ball_point(point, bound, eps=0)
            if not candidates: return self._all_faces
            return np.sort(np.asarray(candidates, dtype=np.int64))
        except (ValueError, OverflowError):
            return self._all_faces

    def distances(self, points, *, broadphase=True, batch_size=1):
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('Finite supported object-frame contact points required')
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError('Exact surface execution batches must contain 1..32 points')
        result = np.full(len(points), np.inf)
        if broadphase and batch_size == 1:
            batches = ((slice(index,index+1), points[index:index+1], self._candidate_faces(point))
                       for index, point in enumerate(points))
        elif broadphase:
            # Batch ONLY identical conservative face sets: no approximate
            # distance, reduced surface, union inflation or candidate reuse
            # across optimizer calls. Preserve every original output row.
            def prepared_batches():
                # Bound temporary face-set storage independently of full T.
                for window in range(0,len(points),batch_size):
                    grouped = {}
                    for index in range(window,min(window+batch_size,len(points))):
                        candidates = self._candidate_faces(points[index])
                        key = candidates.tobytes()
                        if key not in grouped: grouped[key] = (candidates, [])
                        grouped[key][1].append(index)
                    for candidates, indices in grouped.values():
                        selected_points = np.asarray(indices,np.int64)
                        yield selected_points,points[selected_points],candidates
            batches = prepared_batches()
        else:
            batches = ((slice(begin,begin+32), points[begin:begin+32], self._all_faces)
                       for begin in range(0, len(points), 32))
        for rows, p, candidates in batches:
            best = np.full(len(p), np.inf)
            for first in range(0, len(candidates), 2048):
                selected = candidates[first:first+2048]
                starts = self.starts[selected]; edges = self.edges[selected]
                edge2 = self.edge2[selected]
                normal = self.normal[selected]; normal2 = self.normal2[selected]
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    delta = p[:, None, None, :]-starts[None]
                    parameter = np.divide(np.einsum('pfki,fki->pfk', delta, edges), edge2[None],
                        out=np.zeros(delta.shape[:-1]), where=edge2[None] > 0)
                    segment_delta = delta-np.clip(parameter, 0., 1.)[..., None]*edges[None]
                    squared = np.einsum('pfki,pfki->pfk', segment_delta, segment_delta).min(axis=2)
                    height = np.einsum('pfi,fi->pf', p[:, None]-starts[None, :, 0], normal)
                    ratio = np.divide(height, normal2[None], out=np.zeros_like(height), where=normal2[None] > 0)
                    projection = p[:, None]-ratio[..., None]*normal[None]
                    signs = np.einsum('pfki,fi->pfk',
                        np.cross(edges[None], projection[:, :, None]-starts[None]), normal)
                    inside = (signs >= 0).all(axis=2) & (normal2[None] > 0)
                    planar_delta = p[:, None]-projection
                    planar = np.einsum('pfi,pfi->pf', planar_delta, planar_delta)
                    squared = np.minimum(squared, np.where(inside, planar, np.inf))
                if not np.isfinite(squared).all() or np.any(squared < 0):
                    raise ValueError('Continuous contact distances exceed finite representable range')
                best = np.minimum(best, squared.min(axis=1))
            result[rows] = np.sqrt(best)
        return result


def _contact_rows(evidence, config, vertices, count, patch_config=None):
    if type(evidence) is not SequenceContactEvidence or type(config) is not SequenceContactConfig:
        raise ValueError('Explicit frozen SequenceContactEvidence and SequenceContactConfig required')
    if evidence.activations.shape != (count, 2):
        raise ValueError('Anatomical contact must preserve the full original timeline')
    surface = _ContactTriangleSurface(vertices, evidence.object_faces)
    if patch_config is None:
        samples = np.linspace(0, evidence.hand_points_camera.shape[2]-1,
            min(config.max_points_per_hand, evidence.hand_points_camera.shape[2]), dtype=np.int64)
    else:
        # Do not silently discard preselected candidates with legacy sampling.
        size = evidence.hand_points_camera.shape[2]
        if size > min(config.max_points_per_hand, patch_config.max_candidates):
            raise ValueError('Full frozen candidate pool exceeds explicit contact work cap')
        samples = np.arange(size, dtype=np.int64)
    frames, sides = np.nonzero(evidence.activations & (np.arange(count)[:, None] > 0))
    points, point_frames, groups = [], [], []
    for group, (frame, side) in enumerate(zip(frames, sides)):
        supported = samples[evidence.hand_visible[frame, side, samples]]
        if not len(supported):
            raise ValueError('Active contact lacks a supported fixed anatomical witness; do not self-disable')
        points.extend(evidence.hand_points_camera[frame, side, supported])
        point_frames.extend([frame]*len(supported)); groups.extend([group]*len(supported))
    pairs = len(points)*len(evidence.object_faces)
    if pairs > config.max_point_triangle_pairs:
        raise ValueError('Exact contact point-triangle work cap exceeded before fitting')
    return (surface, frames, np.asarray(points, float).reshape(-1, 3),
            np.asarray(point_frames, np.int64), np.asarray(groups, np.int64), samples, pairs)


def _real(value, shape, name, finite=True):
    x = np.asarray(value)
    if (np.ma.isMaskedArray(value) or x.dtype.kind != 'f' or x.shape != shape
            or finite and not np.isfinite(x).all()):
        raise ValueError(name + ': exact floating shape and finite supported values required')
    return x.astype(np.float64, copy=True)


def _rigid(r, t):
    if (not np.isfinite(r).all() or not np.isfinite(t).all()
            or not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
        raise ValueError('Proper finite SE(3), no repair, required')


def initialize_missing_poses(rotations, translations, observed):
    """Bilateral endpoints initialize latent poses; observations stay unchanged.

    Edge gaps are unresolved, not stationary extrapolations. This initialization
    is never exposed as a measured pose or converted into a segmentation mask.
    """
    flags = np.asarray(observed)
    if flags.dtype != np.bool_ or flags.ndim != 1 or len(flags) < 3:
        raise ValueError('Explicit full-T boolean observation flags required')
    n = len(flags)
    r = _real(rotations, (n, 3, 3), 'rotations', False)
    t = _real(translations, (n, 3), 'translations', False)
    _rigid(r[flags], t[flags])
    if not np.isnan(r[~flags]).all() or not np.isnan(t[~flags]).all():
        raise ValueError('Missing observations must be NaN, not manufactured measurements')
    if not flags[0] or not flags[-1]:
        raise ValueError('Leading/trailing unknown poses require an RGB recovery anchor')
    anchors = np.flatnonzero(flags)
    r[~flags] = Slerp(anchors, Rotation.from_matrix(r[flags]))(np.flatnonzero(~flags)).as_matrix()
    for axis in range(3):
        t[~flags, axis] = np.interp(np.flatnonzero(~flags), anchors, t[flags, axis])
    return r, t


@dataclass(frozen=True)
class SequencePoseResult:
    rotations: np.ndarray
    translations: np.ndarray
    frame_index: np.ndarray
    observed_rgb: np.ndarray
    diagnostics: dict


def refine_sequence(vertices, canonical_points, tracks_xy, track_visible,
                    rotations, translations, pose_observed, K, frame_index, fps,
                    config: SequencePoseConfig, *, tracks_depth_m=None,
                    depth_visible=None, depth_config: RGBDepthConfig | None = None,
                    depth_covariance_config: CommonModeDepthConfig | None = None,
                    contact_evidence: SequenceContactEvidence | None = None,
                    contact_config: SequenceContactConfig | None = None,
                    contact_patch_config: ContactPatchConfig | None = None):
    """One sparse full-T robust SE(3) fit; time-zero gauge stays exact.

    Canonical material points must be attached once, before tracking. Visibility
    is RGB support, not ground-truth absence. NaN coordinates are permitted ONLY
    for unsupported observations and have exactly zero observation residual.
    Rotations use local SO(3) increments; angular acceleration compares adjacent
    spatial angular velocities, avoiding global axis-angle wraparound. Centroid
    acceleration is canonical-origin invariant. No zero-velocity loss is used.

    Optional depth must measure these same fixed tracked material points, not
    per-frame pose-projected locations or radial ranges. Supported scalar z is
    finite and positive; every unknown is NaN. Depth support is a subset of RGB
    support. Only R/T are fitted; depth cannot change shape, scale, K or gauge.
    Without all three explicit depth arguments, the RGB-only route is unchanged.
    Optional covariance changes within-frame measurement weighting only, not
    observed depths or a per-frame camera/scale. Zero preserves the legacy route.

    Optional anatomical contact adds distance to the complete original triangle
    surface for caller-frozen active hand/frame entries only. Each active hand
    uses the closest of a fixed regular anatomical subset, not attraction of
    every hand vertex. Activation never depends on an optimized candidate pose.
    Optional soft patch aggregation uses a caller-frozen candidate pool without
    requiring all candidates to touch or fixing tangential motion.
    """
    if type(config) is not SequencePoseConfig:
        raise ValueError('Explicit externally frozen config required')
    v, p = np.asarray(vertices), np.asarray(canonical_points)
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3 or p.ndim != 2
            or p.shape[1:] != (3,) or len(p) < 4):
        raise ValueError('One fixed mesh and at least four fixed material witnesses required')
    v, p = _real(v, v.shape, 'vertices'), _real(p, p.shape, 'canonical_points')
    ids = np.asarray(frame_index)
    if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(ids) < 3 or not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError('Full original contiguous timeline required')
    n, q = len(ids), len(p)
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive actual frame rate in seconds required')
    visible, flags = np.asarray(track_visible), np.asarray(pose_observed)
    if visible.dtype != np.bool_ or visible.shape != (n, q) or flags.dtype != np.bool_ or flags.shape != (n,):
        raise ValueError('Original RGB/pose support flags required')
    xy = _real(tracks_xy, (n, q, 2), 'tracks', False)
    if not np.isfinite(xy[visible]).all() or not np.isnan(xy[~visible]).all():
        raise ValueError('Unsupported coordinates must be explicit NaN')
    if np.any(visible[1:].sum(0) == 0):
        raise ValueError('Every retained witness needs post-anchor RGB evidence')
    use_depth = any(value is not None for value in (tracks_depth_m, depth_visible, depth_config))
    if depth_covariance_config is not None and (not use_depth
            or type(depth_covariance_config) is not CommonModeDepthConfig):
        raise ValueError('Depth covariance requires explicit inferred depth and frozen config')
    if use_depth:
        if tracks_depth_m is None or depth_visible is None or type(depth_config) is not RGBDepthConfig:
            raise ValueError('Explicit depth values, support and externally frozen RGBDepthConfig required')
        depth_flags = np.asarray(depth_visible)
        if (np.ma.isMaskedArray(depth_visible) or depth_flags.dtype != np.bool_
                or depth_flags.shape != (n, q) or np.any(depth_flags & ~visible)):
            raise ValueError('Exact boolean depth support must be a subset of RGB support')
        depth = _real(tracks_depth_m, (n, q), 'tracks_depth_m', False)
        if (not np.isfinite(depth[depth_flags]).all() or np.any(depth[depth_flags] <= 0)
                or not np.isnan(depth[~depth_flags]).all()):
            raise ValueError('Supported depth must be positive finite axial z; unknowns must be NaN')
        depth_frames, depth_queries = np.nonzero(depth_flags & (ids[:, None] > 0))
        if not len(depth_frames):
            raise ValueError('Explicit RGB/depth route needs post-anchor measured depth evidence')
    r0 = _real(rotations, (n, 3, 3), 'rotations')
    t0 = _real(translations, (n, 3), 'translations'); _rigid(r0, t0)
    camera = _real(K, (3, 3), 'K')
    if (not np.array_equal(camera[2], [0, 0, 1]) or camera[0, 0] <= 0
            or camera[1, 1] <= 0 or camera[0, 1] != 0 or camera[1, 0] != 0):
        raise ValueError('One fixed positive focal, zero-skew camera required')
    centre = v.mean(0)
    diameter = 2 * np.sqrt(np.mean(np.sum((v-centre)**2, axis=1)))
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('Nondegenerate fixed geometry required')
    contact_supplied = contact_evidence is not None or contact_config is not None
    if contact_patch_config is not None:
        if not contact_supplied or type(contact_patch_config) is not ContactPatchConfig:
            raise ValueError('Contact patch requires explicit frozen anatomical evidence/config')
    if contact_supplied:
        contact_surface, contact_frames, contact_points, contact_point_frames, contact_groups, contact_samples, contact_pairs = _contact_rows(
            contact_evidence, contact_config, v, n, contact_patch_config)
    use_contact = contact_supplied and len(contact_frames) > 0
    use_covariance = use_depth and depth_covariance_config is not None and depth_covariance_config.sigma_common_diameter > 0
    if use_covariance:
        depth_whitening = FrameDepthWhitening(depth_frames,
            float(diameter*depth_config.depth_sigma_diameter),
            float(diameter*depth_covariance_config.sigma_common_diameter))
    # Spatial support must constrain more than an arbitrary axis.
    singular = np.linalg.svd(p-p.mean(0), compute_uv=False)
    if singular[1] <= singular[0]*1e-8:
        raise ValueError('Canonical RGB points are rank deficient')
    frame_rows, query_rows = np.nonzero(visible & (ids[:, None] > 0))
    pose_rows = np.flatnonzero(flags & (ids > 0))
    base_centres = np.einsum('tij,j->ti', r0, centre)+t0
    counts = visible[1:].sum(0)
    # Quadratic-regime track normalization, not calibrated robust track balance.
    track_weights = np.sqrt((n-1)/counts[query_rows])
    near = max(np.finfo(float).eps*diameter*64, np.finfo(float).tiny)

    def poses(x):
        delta = x.reshape(n-1, 6)
        rr = r0.copy(); tt = t0.copy()
        rr[1:] = Rotation.from_rotvec(delta[:, :3]).as_matrix() @ r0[1:]
        tt[1:] += delta[:, 3:]*diameter
        return rr, tt

    def residual(x):
        rr, tt = poses(x)
        xyz = np.einsum('nij,nj->ni', rr[frame_rows], p[query_rows])+tt[frame_rows]
        # Barrier remains explicit; behind-camera points are not deleted.
        z = np.maximum(xyz[:, 2], near)
        predicted = xyz[:, :2]/z[:, None]*camera.diagonal()[:2]+camera[:2, 2]
        rgb = (predicted-xy[frame_rows, query_rows])/config.pixel_sigma
        rgb *= track_weights[:, None]
        barrier = np.minimum(xyz[:, 2]-near, 0)/diameter
        centres = np.einsum('tij,j->ti', rr, centre)+tt
        prior_c = (centres[pose_rows]-base_centres[pose_rows])/(diameter*config.prior_centroid_sigma_diameter)
        prior_r = Rotation.from_matrix(rr[pose_rows] @ r0[pose_rows].swapaxes(-1, -2)).as_rotvec()/config.prior_rotation_sigma_rad
        acceleration = np.diff(centres, n=2, axis=0)*fps**2/(diameter*config.acceleration_sigma_diameter_s2)
        velocity_r = Rotation.from_matrix(rr[1:] @ rr[:-1].swapaxes(-1, -2)).as_rotvec()*fps
        acceleration_r = np.diff(velocity_r, axis=0)*fps/config.angular_acceleration_sigma_rad_s2
        residuals = (rgb.ravel(), barrier, np.sqrt(config.prior_weight)*prior_c.ravel(),
            np.sqrt(config.prior_weight)*prior_r.ravel(), np.sqrt(config.temporal_weight)*acceleration.ravel(),
            np.sqrt(config.temporal_weight)*acceleration_r.ravel())
        if use_depth:
            depth_xyz = np.einsum('nij,nj->ni', rr[depth_frames], p[depth_queries])+tt[depth_frames]
            depth_residual = depth_xyz[:, 2]-depth[depth_frames, depth_queries]
            measured_z = (depth_whitening.apply(depth_residual) if use_covariance else
                depth_residual/(diameter*depth_config.depth_sigma_diameter))
            residuals += (measured_z,)
        if use_contact:
            object_points = np.einsum('qi,qij->qj', contact_points-tt[contact_point_frames], rr[contact_point_frames])
            distances = contact_surface.distances(object_points,
                batch_size=32 if contact_patch_config is not None else 1)
            if contact_patch_config is None:
                hand_distances = np.full(len(contact_frames), np.inf)
                np.minimum.at(hand_distances, contact_groups, distances)
            else:
                hand_distances = smooth_patch_distances(distances, contact_groups,
                    len(contact_frames), float(diameter*contact_patch_config.temperature_diameter))
            residuals += (hand_distances/(diameter*contact_config.contact_sigma_diameter),)
        return np.concatenate(residuals)

    rows = 3*len(frame_rows)+6*len(pose_rows)+6*(n-2)
    if use_depth: rows += len(depth_frames)
    if use_contact: rows += len(contact_frames)
    pattern = lil_matrix((rows, (n-1)*6), dtype=np.int8)
    offset = 0
    for width in (2, 1):
        for frame in frame_rows:
            pattern[offset:offset+width, (frame-1)*6:frame*6] = 1; offset += width
    for _ in range(2):
        for frame in pose_rows:
            pattern[offset:offset+3, (frame-1)*6:frame*6] = 1; offset += 3
    for _ in range(2):
        for frame in range(n-2):
            for other in range(max(frame, 1), frame+3):
                pattern[offset:offset+3, (other-1)*6:other*6] = 1
            offset += 3
    if use_depth:
        for frame in depth_frames:
            pattern[offset, (frame-1)*6:frame*6] = 1; offset += 1
    if use_contact:
        for frame in contact_frames:
            pattern[offset, (frame-1)*6:frame*6] = 1; offset += 1
    start = np.zeros((n-1)*6)
    initial_residual = residual(start)
    result = least_squares(residual, start, jac_sparsity=pattern.tocsr(), loss='soft_l1',
        f_scale=1., max_nfev=config.max_nfev, x_scale='jac', ftol=1e-6, xtol=1e-6, gtol=1e-6)
    rr, tt = poses(result.x); _rigid(rr, tt)
    xyz = p[None] @ rr.swapaxes(-1, -2)+tt[:, None]
    if not np.isfinite(xyz).all() or (xyz[..., 2] <= near).any():
        raise ValueError('Result places canonical witnesses behind camera')
    # Unqueried parts of the fixed mesh must remain renderable too. Chunk only
    # the check, never clip/delete triangles or alter the reconstructed geometry.
    for start in range(0, len(v), 4096):
        full = v[None, start:start+4096] @ rr.swapaxes(-1, -2)+tt[:, None]
        if not np.isfinite(full).all() or (full[..., 2] <= near).any():
            raise ValueError('Result places fixed mesh vertices behind camera')
    final_residual = residual(result.x)
    cost = lambda a: float(np.sum(np.sqrt(1+a*a)-1))
    if cost(final_residual) > cost(initial_residual)*(1+1e-8):
        raise ValueError('Objective did not improve; do not replace original predictions')
    diagnostics = dict(method='fixed_shape_full_T_RGB_SE3_bundle_v1', evaluations=int(result.nfev),
        converged=bool(result.success), termination=int(result.status), initial_cost=cost(initial_residual),
        final_cost=cost(final_residual), RGB_observations=len(frame_rows),
        latent_frames=int(np.count_nonzero(~visible.any(1))), geometry_camera_scale_unchanged=True,
        time_zero_unchanged=True, static_constraint=False, contact_constraint=False,
        full_mesh_in_front_verified=True, RGB_weighting='quadratic_track_normalized_soft_l1',
        config=asdict(config), quality_verified=False)
    if use_depth:
        diagnostics.update(method='fixed_shape_full_T_RGB_depth_SE3_bundle_v1',
            depth_observations=len(depth_frames), depth_supported_frames=int(depth_flags.any(1).sum()),
            depth_config=asdict(depth_config), depth_units='same_camera_axial_z_metres',
            depth_source_authenticated_by_module=False, depth_accuracy_verified=False,
            depth_weighting='one_scalar_soft_l1_per_measured_point_diameter_normalized')
        if use_covariance:
            diagnostics.update(depth_covariance_config=asdict(depth_covariance_config),
                depth_weighting='symmetric_rank_one_withinframe_whitened_soft_l1',
                depth_temporal_covariance_modeled=False, depth_noise_calibration_verified=False)
    if use_contact:
        diagnostics.update(method='fixed_shape_full_T_RGB'+('_depth' if use_depth else '')+'_contact_SE3_bundle_v1',
            contact_constraint=True, contact_active_entries=len(contact_frames),
            contact_selected_point_indices=contact_samples.tolist(), contact_config=asdict(contact_config),
            contact_source_reference=contact_evidence.source_reference,
            contact_activation_policy='caller_qualified_prefrozen_no_candidate_reactivation',
            contact_activation_verified_by_module=False, contact_accuracy_verified=False,
            contact_distance='minimum_anatomical_point_to_full_continuous_triangle_surface',
            contact_point_triangle_pairs_per_residual=contact_pairs,
            contact_mesh_faces=len(contact_evidence.object_faces), contact_degenerate_faces_retained=True)
        diagnostics['contact_broadphase'] = 'conservative_triangle_balls_referenced_vertex_upper_bound_full_fallback'
        if contact_patch_config is not None:
            diagnostics.update(contact_patch_config=asdict(contact_patch_config),
                contact_distance='softmax_squared_distance_frozen_anatomical_pool_to_full_surface',
                contact_candidate_pool_frozen=True, contact_tangential_lock=False,
                contact_normals_used=False, contact_memory_model_used=False)
    return SequencePoseResult(rr, tt, ids.copy(), visible.any(1), diagnostics)

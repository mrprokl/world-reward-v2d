"""Fixed-shape RGB bundle adjustment with explicit missing observations.

No physics snapping, scale/camera/shape fitting, mask filling or zero-velocity
prior. Pose estimates are correlated proposals, not calibrated measurements.
All frames participate; hidden motion remains inferred and is not identifiable
without evidence. Parameters must be frozen on non-challenge development data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp


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
                or not self.development_reference.strip()):
            raise ValueError('Positive externally frozen scales/weights/budget required')


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
                    config: SequencePoseConfig):
    """One sparse full-T robust SE(3) fit; time-zero gauge stays exact.

    Canonical material points must be attached once, before tracking. Visibility
    is RGB support, not ground-truth absence. NaN coordinates are permitted ONLY
    for unsupported observations and have exactly zero observation residual.
    Rotations use local SO(3) increments; angular acceleration compares adjacent
    spatial angular velocities, avoiding global axis-angle wraparound. Centroid
    acceleration is canonical-origin invariant. No zero-velocity loss is used.
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
    # Spatial support must constrain more than an arbitrary axis.
    singular = np.linalg.svd(p-p.mean(0), compute_uv=False)
    if singular[1] <= singular[0]*1e-8:
        raise ValueError('Canonical RGB points are rank deficient')
    frame_rows, query_rows = np.nonzero(visible & (ids[:, None] > 0))
    pose_rows = np.flatnonzero(flags & (ids > 0))
    base_centres = np.einsum('tij,j->ti', r0, centre)+t0
    counts = visible[1:].sum(0)
    # Track-balanced residual: long-lived tracks cannot silently dominate.
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
        return np.concatenate((rgb.ravel(), barrier, np.sqrt(config.prior_weight)*prior_c.ravel(),
            np.sqrt(config.prior_weight)*prior_r.ravel(), np.sqrt(config.temporal_weight)*acceleration.ravel(),
            np.sqrt(config.temporal_weight)*acceleration_r.ravel()))

    rows = 3*len(frame_rows)+6*len(pose_rows)+6*(n-2)
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
    start = np.zeros((n-1)*6)
    initial_residual = residual(start)
    result = least_squares(residual, start, jac_sparsity=pattern.tocsr(), loss='soft_l1',
        f_scale=1., max_nfev=config.max_nfev, x_scale='jac', ftol=1e-6, xtol=1e-6, gtol=1e-6)
    rr, tt = poses(result.x); _rigid(rr, tt)
    xyz = p[None] @ rr.swapaxes(-1, -2)+tt[:, None]
    if not np.isfinite(xyz).all() or (xyz[..., 2] <= near).any():
        raise ValueError('Result places canonical witnesses behind camera')
    final_residual = residual(result.x)
    cost = lambda a: float(np.sum(np.sqrt(1+a*a)-1))
    if cost(final_residual) > cost(initial_residual)*(1+1e-8):
        raise ValueError('Objective did not improve; do not replace original predictions')
    diagnostics = dict(method='fixed_shape_full_T_RGB_SE3_bundle_v1', evaluations=int(result.nfev),
        converged=bool(result.success), termination=int(result.status), initial_cost=cost(initial_residual),
        final_cost=cost(final_residual), RGB_observations=len(frame_rows),
        latent_frames=int(np.count_nonzero(~visible.any(1))), geometry_camera_scale_unchanged=True,
        time_zero_unchanged=True, static_constraint=False, contact_constraint=False,
        config=asdict(config), quality_verified=False)
    return SequencePoseResult(rr, tt, ids.copy(), visible.any(1), diagnostics)

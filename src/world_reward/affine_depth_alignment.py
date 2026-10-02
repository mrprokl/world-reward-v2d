"""Experimental clip-shared slope/per-frame Z offset against predicted human Z.

Neither input is truth. Checkerboard held-out pixels test local consistency,
not independent temporal validation, metric accuracy or challenge adoption.
Every supplied human pixel/frame must be finite, positive and supported. Only
within-frame depth variation identifies the slope; motion between frames does
not. No camera, human geometry, object size, mask or outlier is fitted/dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from .metric_alignment import fit_shared_depth_scale
from .pointmap import validate_camera_pointmap

MINIMUM_PAIRS, MINIMUM_SPLIT_PAIRS = 64, 32
MINIMUM_VARIATION_RATIO, IRLS_STEPS, HUBER_K = .005, 10, 1.345


class UnderconstrainedDepthAlignment(ValueError):
    """Only slope-information/rank failure; malformed inputs remain fatal."""


def spatial_split_mask(height, width):
    """Immutable-grid even 8x8 block parity for training; odd is held out."""
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1 for v in (height, width)):
        raise ValueError("Spatial split dimensions must be positive integers")
    yy, xx = np.indices((int(height), int(width)))
    return ((yy//8+xx//8) % 2) == 0


@dataclass(frozen=True)
class AffineFrameSupport:
    frame_index: int
    visible_pixels: int
    train_pixels: int
    heldout_pixels: int
    heldout_alpha_only_median_relative_error: float
    heldout_affine_median_relative_error: float


@dataclass(frozen=True)
class AffineDepthAlignment:
    selection: Literal["alpha_only", "affine"]
    shared_scale: float
    frame_offsets: tuple[float, ...]
    alpha_only_scale: float
    affine_proposal_scale: float
    affine_proposal_offsets: tuple[float, ...]
    frames: tuple[AffineFrameSupport, ...]
    image_size_hw: tuple[int, int]
    within_frame_depth_variation_ratio: float
    normalized_design_rank: int
    normalized_design_condition_number: float
    heldout_alpha_only_median_relative_error: float
    heldout_affine_median_relative_error: float

    def to_dict(self):
        return {"schema": "world-reward-shared-affine-depth-alignment-v1", **asdict(self),
                "model": "predicted_human_Z = clip_shared_positive_alpha * predicted_depth_Z + frame_beta",
                "depth_units": "metres", "fit_pixels": "even parity 8x8 raster blocks",
                "heldout_pixels": "odd parity 8x8 raster blocks; never used in fit",
                "heldout_aggregation": "median of equally weighted per-frame median relative errors",
                "weighting": "equal total weight per frame, Huber IRLS within each frame",
                "irls_steps": IRLS_STEPS, "huber_k": HUBER_K, "MAD_scale_floor_m": 1e-6,
                "minimum_visible_pixels_per_frame": MINIMUM_PAIRS, "minimum_pixels_per_split": MINIMUM_SPLIT_PAIRS,
                "minimum_within_frame_depth_variation_ratio": MINIMUM_VARIATION_RATIO,
                "heldout_minimum_median_gain": .05, "heldout_maximum_frame_regression": .05,
                "heldout_pixels_disjoint": True, "block_independence_verified": False,
                "policy_thresholds_validated": False, "metric_accuracy_verified": False,
                "challenge_performance_verified": False, "adoption_authorized": False}


def _array(value, name, dimensions, *, boolean=False):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.ndim != dimensions or min(array.shape) < 1
            or (array.dtype != np.bool_ if boolean else array.dtype.kind not in "iuf")):
        raise ValueError(f"{name} requires a nonempty explicit {'bool' if boolean else 'real'} {dimensions}D array")
    return array.copy() if boolean else array.astype(np.float64, copy=True)


def _indices(values, count):
    if (not isinstance(values, (list, tuple, np.ndarray)) or len(values) != count
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError("Original frame indices must be strictly increasing nonnegative integers")
    return tuple(int(i) for i in values)


def _solve(x, y, weights):
    means = [(float(w@a), float(w@b)) for a, b, w in zip(x, y, weights)]
    variance = float(np.mean([w@((a-mx)**2) for a, w, (mx, _) in zip(x, weights, means)]))
    covariance = float(np.mean([w@((a-mx)*(b-my)) for a, b, w, (mx, my) in zip(x, y, weights, means)]))
    if not np.isfinite([variance, covariance]).all():
        raise ValueError("Weighted affine statistics exceed finite numeric range")
    if variance <= np.finfo(float).eps:
        raise UnderconstrainedDepthAlignment("Weighted within-frame slope is numerically rank-deficient; no damping or fallback")
    alpha = covariance/variance
    return alpha, np.array([my-alpha*mx for mx, my in means])


def fit_shared_affine_depth_alignment(depths, human_render_depths, visible_human_masks,
                                     frame_indices) -> AffineDepthAlignment:
    """Train once; retain alpha-only unless untouched spatial pixels favor affine.

    Arrays are [T,H,W], masks are caller-supplied automatic visible human support
    on the same immutable raster. At least three frames, 64 selected valid pairs
    and 32 in EACH split are required for EVERY frame. Bad selected pairs cause
    rejection, never a changed mask. All inputs are copied; external mutation is
    not a permissible source of new support. Different visibility across frames
    is legitimate, but different raster grids or frame counts are not.
    """
    z = _array(depths, "depths", 3); h = _array(human_render_depths, "human_render_depths", 3)
    mask = _array(visible_human_masks, "visible_human_masks", 3, boolean=True)
    if z.shape != h.shape or z.shape != mask.shape or len(z) < 3:
        raise ValueError("Require three or more paired frames with fixed identical depth/render/mask grids")
    indices = _indices(frame_indices, len(z))
    if not np.isfinite(z[mask]).all() or not np.isfinite(h[mask]).all() or np.any(z[mask] <= 0) or np.any(h[mask] <= 0):
        raise ValueError("Every selected human camera-Z pair must be finite and positive; no dropped support")
    parity = spatial_split_mask(*z.shape[1:])
    train, heldout = mask & parity, mask & ~parity
    counts = [(int(m.sum()), int(a.sum()), int(b.sum())) for m, a, b in zip(mask, train, heldout)]
    if any(n < MINIMUM_PAIRS or min(a, b) < MINIMUM_SPLIT_PAIRS for n, a, b in counts):
        raise ValueError("Every frame requires64 visible pairs and32 train/heldout pairs; no unsupported frame drop")
    x, y = [a[m] for a, m in zip(z, train)], [a[m] for a, m in zip(h, train)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sx, sy = float(np.mean([a.mean() for a in x])), float(np.mean([a.mean() for a in y]))
        xn, yn = [a/sx for a in x], [a/sy for a in y]
        variation = float(np.sqrt(np.mean([np.var(a) for a in xn])))
    if not np.isfinite([sx, sy, variation]).all() or min(sx, sy) <= 0:
        raise ValueError("Normalized depth statistics exceed finite numeric range")
    if variation < MINIMUM_VARIATION_RATIO:
        raise UnderconstrainedDepthAlignment("Slope is underconstrained by within-frame depth variation; between-frame motion does not identify it")
    # Within-frame centering makes the normalized design Gram diagonal: slope
    # variance plus one equally weighted intercept/frame. This is numerical rank
    # evidence only, not a statistically calibrated information measure.
    eigenvalues = np.r_[variation**2, np.full(len(z), 1/len(z))]
    if eigenvalues.min() <= np.finfo(float).eps*eigenvalues.max()*len(eigenvalues):
        raise UnderconstrainedDepthAlignment("Normalized affine design is numerically rank-deficient")
    baseline = fit_shared_depth_scale(z, h, train, indices,
        min_correspondences_per_frame=MINIMUM_SPLIT_PAIRS, min_supported_frames=len(z)).shared_scale
    weights = [np.full(len(a), 1/len(a)) for a in xn]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        alpha, beta = _solve(xn, yn, weights)
        for _ in range(IRLS_STEPS):
            next_weights = []
            for a, b, offset in zip(xn, yn, beta):
                residual = (b-alpha*a-offset)*sy
                if not np.isfinite(residual).all(): raise ValueError("Huber residual exceeds finite numeric range")
                scale = max(1e-6, 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
                robust = np.minimum(1., HUBER_K*scale/np.maximum(np.abs(residual), np.finfo(float).tiny))
                next_weights.append(robust/robust.sum())  # every finite pixel keeps positive weight
            alpha, beta = _solve(xn, yn, next_weights)
        alpha, beta = float(alpha*sy/sx), beta*sy
    if not np.isfinite([alpha, *beta]).all() or alpha <= 0:
        raise ValueError("Affine slope/offsets must be finite with strictly positive shared slope")
    errors0, errors1 = [], []
    for estimated, rendered, support, offset in zip(z, h, heldout, beta):
        proposal = alpha*estimated[support]+offset
        if not np.isfinite(proposal).all() or np.any(proposal <= 0):
            raise ValueError("Affine held-out camera-Z must remain finite and positive; no clipping")
        errors0.append(float(np.median(np.abs(baseline*estimated[support]-rendered[support])/rendered[support])))
        errors1.append(float(np.median(np.abs(proposal-rendered[support])/rendered[support])))
    median0, median1 = float(np.median(errors0)), float(np.median(errors1))
    if not np.isfinite([*errors0, *errors1]).all(): raise ValueError("Held-out relative residual exceeds finite numeric range")
    selected = median0 > 0 and median1 <= .95*median0 and np.all(np.asarray(errors1) <= 1.05*np.asarray(errors0))
    frames = tuple(AffineFrameSupport(i, *n, a, b) for i, n, a, b in zip(indices, counts, errors0, errors1))
    return AffineDepthAlignment("affine" if selected else "alpha_only", alpha if selected else baseline,
        tuple(float(b) for b in beta) if selected else (0.,)*len(z), baseline, alpha,
        tuple(float(b) for b in beta), frames, z.shape[1:], variation, len(eigenvalues),
        float(eigenvalues.max()/eigenvalues.min()), median0, median1)


def apply_affine_depth_alignment(depth, validity, camera_K, alignment: AffineDepthAlignment, *, frame_index):
    """Apply selected Z transform and rebuild unchanged-K +.5 rays, without repair.

    Returns new float64 depth/XYZ; validity and camera are NEVER changed. Invalid
    input records remain explicitly invalid (including their NaNs), not replaced
    with plausible points. Any selected-valid nonfinite/nonpositive result fails.
    Offsets change camera Z/rays, not human geometry or object canonical scale.
    """
    z = _array(depth, "depth", 2); valid = _array(validity, "validity", 2, boolean=True)
    K = _array(camera_K, "camera_K", 2)
    if (z.shape != valid.shape or not valid.any() or K.shape != (3, 3) or not np.isfinite(K).all()
            or min(K[0, 0], K[1, 1]) <= 0 or K[0, 1] != 0 or K[1, 0] != 0 or not np.array_equal(K[2], [0, 0, 1])):
        raise ValueError("Require unchanged positive zero-skew camera and same nonempty valid raster")
    if not np.isfinite(z[valid]).all() or np.any(z[valid] <= 0):
        raise ValueError("Every original valid depth must be finite and positive before affine application")
    if not isinstance(alignment, AffineDepthAlignment): raise ValueError("Require explicit frozen affine selection")
    if z.shape != alignment.image_size_hw: raise ValueError("Camera-Z raster must retain the fitted original image grid")
    _indices([frame_index], 1)
    positions = [i for i, support in enumerate(alignment.frames) if support.frame_index == frame_index]
    if len(positions) != 1: raise ValueError("Selected original frame index must be present exactly once")
    with np.errstate(over="ignore", invalid="ignore"):
        aligned = alignment.shared_scale*z+alignment.frame_offsets[positions[0]]
        yy, xx = np.indices(z.shape)
        points = np.stack(((xx+.5-K[0, 2])*aligned/K[0, 0], (yy+.5-K[1, 2])*aligned/K[1, 1], aligned), axis=-1)
    validate_camera_pointmap(aligned, points, valid, np.diag([1/z.shape[1], 1/z.shape[0], 1.])@K, K)
    return aligned, points

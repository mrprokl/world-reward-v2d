"""Frozen D96 root-only numerical policy; no observations, models or private IO.

Euler increments are native ZYX controls, not externally composed rotations.
The independent 2D objective cannot certify 3D, interaction or challenge quality.
"""
from __future__ import annotations

import numpy as np

TRAIN_COCO = (5, 6, 7, 8, 11, 12, 13, 14, 15, 16)
COCO_TO_MHR = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
TRAIN_MHR = tuple(COCO_TO_MHR[j] for j in TRAIN_COCO)
HELDOUT_COCO = (0, 1, 2, 3, 4, 9, 10)
EVALUATED_STATES = 60
XY_BOUND_M = .30
LOG_Z_BOUND = float(np.log(1.25))
EULER_AXIS_BOUND = .30 / np.sqrt(3.)
PRIOR_SIGMA = .15
HUBER_PIXELS = 5.
MIN_TRAIN_POINTS = 6
MIN_EXTENT_PIXELS = 32.
MIN_JACOBIAN_RATIO = 1e-5
PHYSICAL_BOUNDS = (XY_BOUND_M, XY_BOUND_M, LOG_Z_BOUND,
                   EULER_AXIS_BOUND, EULER_AXIS_BOUND, EULER_AXIS_BOUND)


def numeric(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError("Hidden validity is forbidden: " + name)
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("Finite floating shape required: " + name)
    return array.astype(np.float64)


def bounded_delta(latent):
    """XY metres, log depth ratio, then three native Euler radians."""
    u = numeric(latent, (6,), "six latent coordinates")
    return np.tanh(u) * np.array(PHYSICAL_BOUNDS)


def physical_delta(delta):
    d = numeric(delta, (6,), "bounded physical delta")
    if np.any(np.abs(d) > np.array(PHYSICAL_BOUNDS)):
        raise ValueError("Frozen physical bounds exceeded")
    return d


def camera_translation(original, delta):
    origin = numeric(original, (3,), "original camera translation")
    d = physical_delta(delta)
    if origin[2] <= 0:
        raise ValueError("Positive original camera Z required")
    with np.errstate(over="ignore", invalid="ignore"):
        translated = np.array([origin[0]+d[0], origin[1]+d[1], origin[2]*np.exp(d[2])])
    if not np.isfinite(translated).all() or translated[2] <= 0:
        raise ValueError("Camera translation overflow or invalid depth")
    return translated


def training_observations(keypoints, scores):
    xy = numeric(keypoints, (133, 2), "native image keypoints")
    raw = numeric(scores, (133,), "unclamped native scores")
    valid = raw[list(TRAIN_COCO)] > 0
    selected = xy[list(TRAIN_COCO)][valid]
    with np.errstate(over="ignore", invalid="ignore"):
        extent = np.ptp(selected, axis=0) if len(selected) else np.zeros(2)
    if len(selected) < MIN_TRAIN_POINTS or not np.isfinite(extent).all() or np.any(extent < MIN_EXTENT_PIXELS):
        raise ValueError("Training observations insufficient; retain frame and fail trial")
    return selected, np.array(TRAIN_MHR, dtype=np.int64)[valid], valid


def objective(projected, observed, delta):
    if np.ma.isMaskedArray(observed):
        raise ValueError("Hidden observation validity is forbidden")
    target = np.asarray(observed)
    if target.ndim != 2 or target.shape[1:] != (2,) or not MIN_TRAIN_POINTS <= len(target) <= len(TRAIN_COCO):
        raise ValueError("Six to ten fixed training observations required")
    truth = numeric(target, target.shape, "training observations")
    prediction = numeric(projected, truth.shape, "projected native training keypoints")
    d = physical_delta(delta)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.linalg.norm(prediction-truth, axis=1) / HUBER_PIXELS
        data = float(np.where(q <= 1, .5*q*q, q-.5).mean())
    prior = float(.5*np.square(d/PRIOR_SIGMA).sum())
    if not np.isfinite([data, prior, data+prior]).all():
        raise ValueError("Nonfinite objective arithmetic; no loss clipping")
    return {"data": data, "prior": prior, "total": data+prior}


def jacobian_evidence(jacobian, points):
    """Only 2D observation rows; caller must not append prior residuals."""
    if type(points) is not int or not MIN_TRAIN_POINTS <= points <= len(TRAIN_COCO):
        raise ValueError("Exact fixed observed point count required")
    j = numeric(jacobian, (2*points, 6), "unregularized observation Jacobian")
    singular = np.linalg.svd(j, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("Nonfinite observation singular values")
    ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    if ratio < MIN_JACOBIAN_RATIO:
        raise ValueError("Degenerate native observation Jacobian; no prior rank rescue")
    return {"singular_values": singular.tolist(), "minimum_maximum_ratio": ratio,
            "observation_rows": 2*points, "prior_rows_included": False}


def best_evaluated(losses):
    loss = numeric(losses, (EVALUATED_STATES,), "sixty evaluated objectives")
    if np.any(loss < 0):
        raise ValueError("Nonnegative total objectives required")
    index = int(np.argmin(loss))  # First tie; no unevaluated final update.
    if loss[index] > loss[0]+1e-6:
        raise ValueError("Selected evaluated state regressed")
    return index


def quality_decision(human_errors, per_hand_errors):
    """Three clips, two branches (baseline/refit), two hands; no alignment."""
    human = numeric(human_errors, (3, 2), "per-clip camera PVE")
    hands = numeric(per_hand_errors, (3, 2, 2), "per-clip/branch/hand relative object errors")
    if np.any(human < 0) or np.any(hands < 0):
        raise ValueError("Nonnegative complete private errors required")
    defined = bool(np.all(human[:, 0] > 0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = (human[:, 0]-human[:, 1])/human[:, 0] if defined else None
        hand_limits = hands[:, 0]*1.05
    if (defined and not np.isfinite(gain).all()) or not np.isfinite(hand_limits).all():
        raise ValueError("Nonfinite private relative-gain arithmetic")
    interaction_pass = bool(np.all(hands[:, 1] <= hand_limits))
    gates = {"median_camera_pve_gain_5pct": defined and float(np.median(gain)) >= .05,
             "no_clip_camera_pve_regression_over_5pct": defined and bool(np.all(gain >= -.05)),
             "no_per_hand_clip_relative_object_regression_over_5pct": interaction_pass}
    return {"gates": gates, "zero_human_baseline_relative_gain_undefined": not defined,
            "per_clip_camera_pve_relative_gain": gain.tolist() if defined else None,
            "synthetic_root_refit_hypothesis_supported": all(gates.values()),
            "adoption_authorized": False, "full_HOI_verified": False,
            "real_domain_verified": False, "challenge_performance_verified": False}

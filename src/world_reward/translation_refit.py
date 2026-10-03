"""H97 external-camera translation policy, frozen before any efficacy fit.

Learned native geometry/identity/rotations are unchanged. This three-coordinate
follow-up is not a repeat or relaxation of the failed six-coordinate D96 trial.
Only independent RGB observations enter fitting; metric quality is separate.
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
PHYSICAL_BOUNDS = (XY_BOUND_M, XY_BOUND_M, LOG_Z_BOUND)
PRIOR_SIGMA = .15
HUBER_PIXELS = 5.
MIN_TRAIN_POINTS = 6
MIN_EXTENT_PIXELS = 32.
MIN_JACOBIAN_RATIO = 1e-5
LEARNING_RATE = .01
BETAS = (.9, .999)
EPS = 1e-8


def numeric(value, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError("Hidden validity is forbidden: " + name)
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("Finite floating shape required: " + name)
    return array.astype(np.float64)


def bounded_delta(latent):
    """Camera XY metres and log camera-translation Z ratio, not vertex scaling."""
    u = numeric(latent, (3,), "three latent coordinates")
    return np.tanh(u) * np.array(PHYSICAL_BOUNDS)


def physical_delta(delta):
    d = numeric(delta, (3,), "bounded physical delta")
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


def project_and_jacobian(baseline_points, original_translation, latent, camera_K):
    """Exact pinhole projection and its 2N x 3 observation-only latent Jacobian.

    Baseline points already include the original external camera translation.
    Every point receives one common additive deltaT. The Z derivative is of
    T0_z exp(delta_log_z), never of the individual vertex depth.
    """
    if np.ma.isMaskedArray(baseline_points):
        raise ValueError("Hidden point validity is forbidden")
    shape = np.shape(baseline_points)
    if len(shape) != 2 or shape[1:] != (3,) or shape[0] < 1:
        raise ValueError("Complete nonempty camera points required")
    points = numeric(baseline_points, shape, "frozen camera points")
    origin = numeric(original_translation, (3,), "original camera translation")
    u = numeric(latent, (3,), "three latent coordinates")
    k = numeric(camera_K, (3, 3), "fixed pinhole intrinsics")
    if (not np.array_equal(k[2], [0., 0., 1.]) or k[0, 1] != 0 or k[1, 0] != 0
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError("Positive zero-skew pinhole intrinsics required")
    d = bounded_delta(u)
    translation = camera_translation(origin, d)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        p = points + (translation-origin)
        x, y, z = p.T
        projected = np.c_[k[0, 0]*x/z+k[0, 2], k[1, 1]*y/z+k[1, 2]]
        chain = np.array(PHYSICAL_BOUNDS)*(1-np.tanh(u)**2)
        chain[2] *= translation[2]
        jacobian = np.zeros((len(points), 2, 3), np.float64)
        jacobian[:, 0, 0] = k[0, 0]/z*chain[0]
        jacobian[:, 1, 1] = k[1, 1]/z*chain[1]
        jacobian[:, 0, 2] = -k[0, 0]*(x/z)/z*chain[2]
        jacobian[:, 1, 2] = -k[1, 1]*(y/z)/z*chain[2]
    if (not np.isfinite(p).all() or np.any(z <= 0) or not np.isfinite(projected).all()
            or not np.isfinite(jacobian).all()):
        raise ValueError("Translated projection crossed camera plane or overflowed")
    return projected, jacobian.reshape(-1, 3)


def objective_and_gradient(projected, observed, delta, jacobian, latent):
    """Equal-weight radial Huber/5 plus the same physical-coordinate prior."""
    if np.ma.isMaskedArray(observed):
        raise ValueError("Hidden observation validity is forbidden")
    shape = np.shape(observed)
    if len(shape) != 2 or shape[1:] != (2,) or not MIN_TRAIN_POINTS <= shape[0] <= len(TRAIN_COCO):
        raise ValueError("Six to ten fixed training observations required")
    target = numeric(observed, shape, "training observations")
    prediction = numeric(projected, shape, "projected training keypoints")
    d = physical_delta(delta)
    u = numeric(latent, (3,), "three latent coordinates")
    if not np.array_equal(d, bounded_delta(u)):
        raise ValueError("Physical delta differs from its actual evaluated latent")
    j = numeric(jacobian, (2*shape[0], 3), "observation-only Jacobian")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = prediction-target
        norm = np.linalg.norm(residual, axis=1)
        q = norm/HUBER_PIXELS
        data = float(np.where(q <= 1, .5*q*q, q-.5).mean())
        # At zero residual the quadratic branch has gradient exactly zero.
        factors = np.ones(len(target))/HUBER_PIXELS**2
        large = norm > HUBER_PIXELS
        factors[large] = 1/(HUBER_PIXELS*norm[large])
        data_gradient = j.T @ (residual*factors[:, None]/len(target)).reshape(-1)
        prior = float(.5*np.square(d/PRIOR_SIGMA).sum())
        gradient = data_gradient+d/PRIOR_SIGMA**2*np.array(PHYSICAL_BOUNDS)*(1-np.tanh(u)**2)
    if not np.isfinite([data, prior, data+prior]).all() or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite objective arithmetic; no clipping or loss repair")
    return {"data": data, "prior": prior, "total": data+prior}, gradient


def objective(projected, observed, delta):
    """Value-only diagnostic equivalent to the unchanged D96 three-term subset."""
    if np.ma.isMaskedArray(observed):
        raise ValueError("Hidden observation validity is forbidden")
    shape = np.shape(observed)
    if len(shape) != 2 or shape[1:] != (2,) or not MIN_TRAIN_POINTS <= shape[0] <= len(TRAIN_COCO):
        raise ValueError("Six to ten fixed training observations required")
    target = numeric(observed, shape, "training observations")
    prediction = numeric(projected, shape, "projected training keypoints")
    d = physical_delta(delta)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.linalg.norm(prediction-target, axis=1)/HUBER_PIXELS
        data = float(np.where(q <= 1, .5*q*q, q-.5).mean())
        prior = float(.5*np.square(d/PRIOR_SIGMA).sum())
    if not np.isfinite([data, prior, data+prior]).all():
        raise ValueError("Nonfinite objective arithmetic; no clipping or loss repair")
    return {"data": data, "prior": prior, "total": data+prior}


def adam_step(latent, gradient, first_moment, second_moment, update_index):
    """Manual float64 CPU Adam, one-based update index, exact fixed constants."""
    if type(update_index) is not int or not 1 <= update_index < EVALUATED_STATES:
        raise ValueError("Only the59 observed Adam updates are allowed")
    u = numeric(latent, (3,), "latent")
    g = numeric(gradient, (3,), "gradient")
    m = numeric(first_moment, (3,), "first moment")
    v = numeric(second_moment, (3,), "second moment")
    if np.any(v < 0):
        raise ValueError("Nonnegative Adam second moment required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = BETAS[0]*m+(1-BETAS[0])*g
        v = BETAS[1]*v+(1-BETAS[1])*g*g
        next_u = u-LEARNING_RATE*(m/(1-BETAS[0]**update_index))/(np.sqrt(v/(1-BETAS[1]**update_index))+EPS)
    if not np.isfinite(next_u).all() or not np.isfinite(m).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite Adam update; no optimizer repair")
    return next_u, m, v


def jacobian_evidence(jacobian, points):
    if type(points) is not int or not MIN_TRAIN_POINTS <= points <= len(TRAIN_COCO):
        raise ValueError("Exact observed point count required")
    j = numeric(jacobian, (2*points, 3), "unregularized observation Jacobian")
    singular = np.linalg.svd(j, compute_uv=False)
    ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    if not np.isfinite(singular).all() or ratio < MIN_JACOBIAN_RATIO:
        raise ValueError("Degenerate observation Jacobian; no prior rank rescue")
    return {"singular_values": singular.tolist(), "minimum_maximum_ratio": ratio,
            "observation_rows": 2*points, "prior_rows_included": False}


def best_evaluated(losses):
    loss = numeric(losses, (EVALUATED_STATES,), "sixty evaluated objectives")
    if np.any(loss < 0):
        raise ValueError("Nonnegative total objectives required")
    return int(np.argmin(loss))  # First tie, never an unseen final update.


def quality_decision(human_errors, per_hand_errors):
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
    gates = {"median_camera_pve_gain_5pct": defined and float(np.median(gain)) >= .05,
             "no_clip_camera_pve_regression_over_5pct": defined and bool(np.all(gain >= -.05)),
             "no_per_hand_clip_relative_object_regression_over_5pct": bool(np.all(hands[:, 1] <= hand_limits))}
    return {"gates": gates, "zero_human_baseline_relative_gain_undefined": not defined,
            "per_clip_camera_pve_relative_gain": gain.tolist() if defined else None,
            "synthetic_translation_hypothesis_supported": all(gates.values()),
            "cohort_reused_from_unscored_D96": True, "independent_replication_verified": False,
            "adoption_authorized": False, "full_HOI_verified": False,
            "real_domain_verified": False, "challenge_performance_verified": False}

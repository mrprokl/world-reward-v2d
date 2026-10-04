"""Private calibration math for already-frozen automatic candidate features.

No I/O, tracking, temporal MAP, contact or 3D model is implemented here. Labels
enter training or post-selection evaluation only; inference proposal keys are
automatic hypotheses, never certified dataset identities. Three fit/decision
clips are a mechanism micro-test, not statistical or generalization confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InsufficientCalibrationError(ValueError):
    """The frozen calibration bank cannot identify a binary relation model."""


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name}: integer in [{low}, {high}] required")
    return int(value)


def _keys(value, count, name):
    if (type(value) is not tuple or len(value) != count
            or any(type(key) is not str or not key.strip()
                   or any(ord(c) < 32 or ord(c) == 127 for c in key) for key in value)):
        raise ValueError(f"{name}: tuple of nonempty textual keys required")
    return value


def target_object_id(ycb_ids, ycb_grasp_ind):
    """Sequence grasp target, not framewise contact/active-hand ground truth."""
    ids = np.asarray(ycb_ids)
    if (np.ma.isMaskedArray(ycb_ids) or ids.ndim != 1 or not len(ids)
            or ids.dtype.kind not in "iu" or np.any((ids < 1) | (ids > 21))
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Unique YCB class IDs 1..21 required, not segmentation row indices")
    return int(ids[_integer(ycb_grasp_ind, "ycb_grasp_ind", 0, len(ids) - 1)])


def _segmentation(value):
    seg = np.asarray(value)
    if (np.ma.isMaskedArray(value) or seg.ndim != 2 or 0 in seg.shape
            or seg.dtype.kind not in "iu" or not np.isin(seg, (*range(22), 255)).all()):
        raise ValueError("Integer segmentation: 0 background, 255 merged hands, 1..21 objects")
    return seg


@dataclass(frozen=True)
class MaskIdentity:
    object_id: int | None
    mask_pixels: int
    majority_fraction: float
    target_recall: float
    reciprocal_coverage: bool


def majority_identity(mask, segmentation):
    """Private strict >1/2 mask majority; reciprocal coverage is a separate gate.

    A clean fragment can identify an object without covering most of it. Empty,
    mixed, background or merged-hand masks supply no object label. This helper
    must not select/rewrite a predictor proposal using private segmentation.
    """
    seg, selected = _segmentation(segmentation), np.asarray(mask)
    if np.ma.isMaskedArray(mask) or selected.dtype != np.bool_ or selected.shape != seg.shape:
        raise ValueError("Original-grid boolean candidate mask required")
    size = int(np.count_nonzero(selected))
    counts = np.bincount(seg[selected].astype(np.int64), minlength=256)[1:22]
    j = int(np.argmax(counts))
    fraction = float(counts[j] / size) if size else 0.
    identity = j + 1 if 2 * int(counts[j]) > size else None
    recall = float(counts[j] / np.count_nonzero(seg == identity)) if identity else 0.
    return MaskIdentity(identity, size, fraction, recall, bool(identity and recall > .5))


def _features(value, supported):
    x, ok = np.asarray(value), np.asarray(supported)
    if (np.ma.isMaskedArray(value) or np.ma.isMaskedArray(supported)
            or x.ndim != 2 or not x.shape[1] or x.dtype.kind != "f"
            or ok.dtype != np.bool_ or ok.shape != (len(x),)):
        raise ValueError("Floating [N,D] features and boolean [N] support required")
    if not np.isfinite(x[ok]).all():
        raise ValueError("Supported features must be finite; unsupported rows never become zeros")
    return np.array(x, dtype=np.float64, copy=True), ok


def _fit_weights(x, usable, labels, clips, identities):
    weights = np.zeros(len(x))
    for clip in sorted(set(clips)):
        slots = np.array([key == clip for key in clips]) & usable
        if set(labels[slots].tolist()) != {0, 1}:
            raise InsufficientCalibrationError("Every fit clip needs supported positive and negative identities")
        groups = sorted({identities[i] for i in np.flatnonzero(slots)})
        for identity in groups:
            idx = np.array([i for i in np.flatnonzero(slots) if identities[i] == identity])
            if len(np.unique(labels[idx])) != 1:
                raise ValueError("One private training identity cannot have contradictory labels")
            _, inverse, counts = np.unique(x[idx], axis=0, return_inverse=True, return_counts=True)
            # Exact nuisance duplicates do not change a clip/identity's mass.
            weights[idx] = 1. / (len(set(clips)) * len(groups) * len(counts) * counts[inverse])
    return weights


def _sigmoid(value):
    result = np.empty_like(value)
    good = value >= 0
    result[good] = 1. / (1. + np.exp(-value[good]))
    exponential = np.exp(value[~good])
    result[~good] = exponential / (1. + exponential)
    return result


def _objective(theta, design, labels, weights):
    scores = design @ theta
    loss = float(weights @ (np.logaddexp(0., scores) - labels * scores)
                 + .5 * (theta[1:] @ theta[1:]))
    if not np.isfinite(scores).all() or not np.isfinite(loss):
        raise ValueError("Logistic objective overflowed")
    return loss


@dataclass(frozen=True, eq=False)
class IdentityLogistic:
    mean: np.ndarray
    scale: np.ndarray
    constant_features: np.ndarray
    coefficients: np.ndarray
    intercept: float
    sample_weights: np.ndarray
    objective: float
    gradient_inf_norm: float
    iterations: int
    fit_clips: int
    l2: float = 1.


def fit_identity_logistic(features, supported, labels, clip_ids, identity_ids):
    """Fixed L2=1 logistic loss; intercept unpenalized, weighted loss normalized.

    All clips weigh equally, then private identities equally within each clip,
    then distinct feature variants equally. Duplicate masks are not independent
    positive examples. Labels are {-1 unknown,0 wrong object,1 target}; unknown
    or unsupported rows have zero fit mass, but an uninformative clip aborts.
    Weighted standardization uses fit rows ONLY; constant features have scale1.
    Deterministic Newton/Armijo uses 100 iterations, 50 backtracks, gradient
    tolerance1e-9. Nonconvergence raises, never returns a usable fallback model.
    Caller freezes split/features before giving any private labels to this API.
    """
    x, ok = _features(features, supported)
    y = np.asarray(labels)
    clips, identities = _keys(clip_ids, len(x), "clip_ids"), _keys(identity_ids, len(x), "identity_ids")
    if (not len(x) or np.ma.isMaskedArray(labels) or y.shape != (len(x),)
            or y.dtype.kind not in "iu" or not np.isin(y, (-1, 0, 1)).all()):
        raise ValueError("Nonempty integer binary/unknown labels required")
    usable = ok & (y >= 0)
    weights = _fit_weights(x, usable, y, clips, identities)
    xx, yy, ww = x[usable], y[usable].astype(np.float64), weights[usable]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = ww @ xx
            variance = ww @ ((xx - mean) ** 2)
            constant = np.ptp(xx, axis=0) == 0
            scale = np.where(constant, 1., np.sqrt(variance))
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Fit standardization is numerically invalid")
            design = np.column_stack((np.ones(len(xx)), (xx - mean) / scale))
            theta, penalty = np.zeros(design.shape[1]), np.ones(design.shape[1])
            penalty[0] = 0.
            for iteration in range(101):
                loss = _objective(theta, design, yy, ww)
                p = _sigmoid(design @ theta)
                gradient = design.T @ (ww * (p - yy)) + penalty * theta
                norm = float(np.max(np.abs(gradient)))
                if not np.isfinite(gradient).all():
                    raise ValueError("Logistic gradient overflowed")
                if norm <= 1e-9:
                    break
                if iteration == 100:
                    raise RuntimeError("Fixed logistic solver did not converge")
                hessian = design.T @ ((ww * p * (1. - p))[:, None] * design) + np.diag(penalty)
                step = np.linalg.solve(hessian, gradient)
                descent = float(gradient @ step)
                if not np.isfinite(step).all() or not descent > 0:
                    raise RuntimeError("Logistic Newton step is not a finite descent")
                for backtrack in range(50):
                    alpha = .5 ** backtrack
                    trial = theta - alpha * step
                    if _objective(trial, design, yy, ww) <= loss - 1e-4 * alpha * descent:
                        theta = trial
                        break
                else:
                    raise RuntimeError("Fixed logistic line search failed")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Logistic calibration is numerically invalid") from error
    return IdentityLogistic(*map(_readonly, (mean, scale, constant, theta[1:])),
                            float(theta[0]), _readonly(weights), loss, norm, iteration, len(set(clips)))


def score_identity_candidates(model, features, supported):
    """Raw logits, NOT calibrated probabilities; unsupported scores remain NaN."""
    if type(model) is not IdentityLogistic:
        raise ValueError("A completed IdentityLogistic model is required")
    x, ok = _features(features, supported)
    if x.shape[1] != len(model.mean):
        raise ValueError("Feature dimension differs from fit")
    result = np.full(len(x), np.nan)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result[ok] = ((x[ok] - model.mean) / model.scale) @ model.coefficients + model.intercept
        except FloatingPointError as error:
            raise ValueError("Candidate scoring overflowed") from error
    if not np.isfinite(result[ok]).all():
        raise ValueError("Supported candidate scores must be finite")
    return _readonly(result)


@dataclass(frozen=True)
class IdentityDecision:
    proposal_id: str | None
    reason: str
    raw_best_score: float | None
    raw_score_gap: float | None
    ranked_scores: tuple[tuple[str, float], ...]


def decide_identity(scores, supported, proposal_ids, *, minimum_gap):
    """Mean distinct scores per AUTOMATIC proposal group; never private class IDs.

    Strict best score>0 and gap>minimum_gap are explicit decision rules, not
    confidence/probabilities. Freeze minimum_gap on an external decision split
    before evaluation; no selection/threshold search is supplied here. A missing
    competitor, tie, nonpositive winner or single-identity bank abstains.
    """
    values, ok = np.asarray(scores), np.asarray(supported)
    keys = _keys(proposal_ids, len(values) if values.ndim == 1 else 0, "proposal_ids")
    if (np.ma.isMaskedArray(scores) or np.ma.isMaskedArray(supported) or values.ndim != 1
            or values.dtype.kind != "f" or ok.dtype != np.bool_ or ok.shape != values.shape
            or not np.isfinite(values[ok]).all()):
        raise ValueError("Floating scores and explicit boolean support required")
    if (isinstance(minimum_gap, (bool, np.bool_))
            or not isinstance(minimum_gap, (int, float, np.integer, np.floating))
            or not np.isfinite(minimum_gap) or minimum_gap < 0):
        raise ValueError("Explicit finite nonnegative raw-score gap required")
    if not len(values) or not ok.any():
        return IdentityDecision(None, "NO_EVIDENCE", None, None, ())
    if not ok.all():
        return IdentityDecision(None, "UNSUPPORTED_COMPETITOR", None, None, ())
    ranked = tuple(sorted(((key, float(np.mean(np.unique(values[np.array(keys) == key]))))
                           for key in set(keys)), key=lambda row: (-row[1], row[0])))
    best = ranked[0][1]
    gap = float(best - ranked[1][1]) if len(ranked) > 1 else None
    if not np.isfinite(best) or (gap is not None and not np.isfinite(gap)):
        raise ValueError("Grouped raw scores overflowed")
    reason = ("NO_ALTERNATIVE" if gap is None else "NONPOSITIVE_SCORE" if best <= 0
              else "INSUFFICIENT_GAP" if gap <= minimum_gap else "STRICT_WINNER")
    return IdentityDecision(ranked[0][0] if reason == "STRICT_WINNER" else None, reason, best, gap, ranked)


@dataclass(frozen=True, eq=False)
class IdentityTimeline:
    frame_index: np.ndarray
    correct_identity: np.ndarray
    reciprocal_coverage: np.ndarray
    wrong_id: np.ndarray
    unknown_mask: np.ndarray
    missing: np.ndarray
    target_visible: np.ndarray
    longest_failure_gap: int


def evaluate_identity_timeline(frame_index, selected_masks, segmentations, target_id):
    """Private matching AFTER selections freeze; every original frame stays scored.

    None/empty masks are missing; mixed/hand/background masks are unknown, not
    silently dropped wrong predictions. Known nontarget objects are wrong-ID.
    Even invisible targets remain in denominators (visibility is diagnostic).
    Coverage requires both target identity and reciprocal >1/2 target recall.
    No GT is allowed back into selection, grouping, feature fitting or pose.
    """
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64 or indices.ndim != 1
            or not len(indices) or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
        raise ValueError("Complete original int64 arange(T) required")
    target = _integer(target_id, "target_id", 1, 21)
    if (type(selected_masks) is not tuple or type(segmentations) is not tuple
            or len(selected_masks) != len(indices) or len(segmentations) != len(indices)):
        raise ValueError("One frozen selection and segmentation per original frame required")
    correct, coverage, wrong, unknown, missing, visible = np.zeros((6, len(indices)), dtype=bool)
    shape = None
    for t, (mask, value) in enumerate(zip(selected_masks, segmentations)):
        seg = _segmentation(value)
        if shape is not None and seg.shape != shape:
            raise ValueError("Original image grid must remain constant through the clip")
        shape, visible[t] = seg.shape, bool(np.any(seg == target))
        if mask is None:
            missing[t] = True
            continue
        assignment = majority_identity(mask, seg)
        missing[t] = assignment.mask_pixels == 0
        unknown[t] = not missing[t] and assignment.object_id is None
        correct[t] = assignment.object_id == target
        wrong[t] = assignment.object_id is not None and not correct[t]
        coverage[t] = correct[t] and assignment.reciprocal_coverage
    gap = longest = 0
    for failure in ~coverage:
        gap = gap + 1 if failure else 0
        longest = max(longest, gap)
    return IdentityTimeline(*map(_readonly, (indices, correct, coverage, wrong, unknown, missing, visible)), longest)


def macro_identity_metrics(clips):
    """Equal clip weights, all full timelines; no successful-only denominator."""
    if type(clips) is not tuple or not clips or any(type(clip) is not IdentityTimeline for clip in clips):
        raise ValueError("Nonempty tuple of completed IdentityTimeline required")
    names = ("correct_identity", "reciprocal_coverage", "wrong_id", "unknown_mask", "missing", "target_visible")
    return {"clips": len(clips), "frames": sum(len(clip.frame_index) for clip in clips),
            **{name + "_rate": float(np.mean([np.mean(getattr(clip, name)) for clip in clips])) for name in names},
            "max_failure_gap": max(clip.longest_failure_gap for clip in clips)}

"""Private, conditional 2D hand diagnostics from already-frozen observations.

No file loading, prediction changes, fitted matching, visibility inference or
accuracy threshold. The verified subset is wrist plus native indices 5..20;
thumbs remain in boxes/raw evidence but never enter joint-error evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hand_observations import HandObservations

VERIFIED_JOINT_INDICES = (0, *range(5, 21))


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _mean(values):
    values = tuple(values)
    if not values:
        return None
    result = math.fsum(value / len(values) for value in values)
    if not math.isfinite(result):
        raise ValueError("Finite diagnostic errors required; never publish infinity")
    return result


@dataclass(frozen=True)
class HandEvaluationCounts:
    frames: int
    returned_predictions: int
    nonempty_prediction_frames: int
    annotated_positive_frames: int
    uniquely_associated_frames: int
    missed_frames: int
    ambiguous_frames: int
    unlabelled_frames: int
    annotation_without_mask_frames: int
    positive_frames_without_valid_joints: int
    valid_gt_joints: int
    scored_joints: int
    scored_frames: int
    other_predictions: int  # Not certified false positives/bystanders.
    unavailable_prediction_boxes: int
    outside_prediction_joints: int


@dataclass(frozen=True, eq=False)
class HandClipEvaluation:
    clip: str
    frame_index: np.ndarray
    annotated_positive: np.ndarray
    unlabelled: np.ndarray
    annotation_without_mask: np.ndarray
    unique_association: np.ndarray
    missed: np.ndarray
    ambiguous: np.ndarray
    matched_prediction: np.ndarray  # Native slot, -1 unless exactly one overlap.
    prediction_count: np.ndarray
    bbox_available_count: np.ndarray
    other_prediction_count: np.ndarray
    outside_joint_count: np.ndarray
    valid_gt_joints: np.ndarray  # [T,17], only annotated-positive frames.
    joint_error_pixels: tuple[tuple[float | None, ...], ...]  # [T,17]
    frame_epe_pixels: tuple[float | None, ...]
    counts: HandEvaluationCounts
    joint_weighted_epe_pixels: float | None
    frame_mean_epe_pixels: float | None
    diagnostic_status: str  # pass/fail/inconclusive, never method adoption.


@dataclass(frozen=True)
class HandPooledEvaluation:
    clips: tuple[HandClipEvaluation, ...]
    counts: HandEvaluationCounts
    joint_weighted_epe_pixels: float | None
    frame_mean_epe_pixels: float | None
    diagnostic_status: str


def evaluate_hand_clip(clip, observations, gt_frame_index, gt_seg, gt_joint_2d):
    """Evaluate full original T, with canonical GT joint_2d [T,21,2].

    ``gt_seg`` yields exactly T uint8 original HxW arrays, consumed one at a time.
    GT hand pixels (255) form pixel-cell bounds [minx,miny,maxx+1,maxy+1]. A native
    box is min/max of all 21 XY only if every point is numerically supported and
    finite; no clipping/padding/mirroring. Exactly one positive-area overlap
    associates a slot. Multiple overlaps stay ambiguous, even if one has lower
    joint error. Empty seg plus all -1 joints is UNLABELLED, not hand absence.

    GT verified joints are independently valid iff finite and inside [0,W)x[0,H).
    EPE is conditional on unique association AND valid GT, with no PCK threshold,
    normalization or alignment. Out-of-grid finite predictions remain errors.
    Caller seals ALL predictions before providing any private annotation values.
    """
    if (type(clip) is not str or not clip.strip() or len(clip) > 256
            or any(ord(c) < 32 or ord(c) == 127 for c in clip)
            or type(observations) is not HandObservations):
        raise ValueError("Explicit clip and frozen HandObservations required")
    frames = len(observations.frames); height, width = observations.image_size
    indices = np.asarray(gt_frame_index)
    if (np.ma.isMaskedArray(gt_frame_index) or indices.dtype != np.int64
            or indices.shape != (frames,) or not np.array_equal(indices, np.arange(frames))
            or not np.array_equal(indices, observations.frame_index)):
        raise ValueError("Matching complete original int64 arange(T) required")
    joints = np.asarray(gt_joint_2d)
    if (np.ma.isMaskedArray(gt_joint_2d) or joints.dtype.kind != "f"
            or joints.shape != (frames, 21, 2)):
        raise ValueError("Canonical floating GT joint_2d [T,21,2] required")
    positive, unlabelled, no_mask, unique, missed, ambiguous = (np.zeros(frames, bool) for _ in range(6))
    matched = np.full(frames, -1, np.int64)
    counts, boxes, others, outside = (np.zeros(frames, np.int64) for _ in range(4))
    valid = np.zeros((frames, 17), bool); errors, means = [], []
    segments = iter(gt_seg)
    for t, native in enumerate(observations.frames):
        try:
            raw_seg = next(segments)
        except StopIteration:
            raise ValueError("Missing original GT segmentation frame") from None
        seg = np.asarray(raw_seg)
        if np.ma.isMaskedArray(raw_seg) or seg.dtype != np.uint8 or seg.shape != (height, width):
            raise ValueError("Original HxW uint8 segmentation required")
        xy = native.original_xy.values; counts[t] = native.count
        available = native.original_xy.supported.all(axis=1) & np.isfinite(xy).all(axis=(1, 2))
        boxes[t] = int(available.sum())
        outside[t] = int(np.count_nonzero(np.isfinite(xy).all(axis=-1) & (
            (xy[..., 0] < 0) | (xy[..., 0] >= width) | (xy[..., 1] < 0) | (xy[..., 1] >= height))))
        ys, xs = np.nonzero(seg == 255); positive[t] = bool(len(xs))
        unlabelled[t] = not positive[t] and bool(np.all(joints[t] == -1))
        no_mask[t] = not positive[t] and not unlabelled[t]
        row_errors = [None] * 17
        if positive[t]:
            gt = joints[t, VERIFIED_JOINT_INDICES, :].astype(np.float64)
            valid[t] = np.isfinite(gt).all(axis=1) & (gt[:, 0] >= 0) & (gt[:, 0] < width) & (gt[:, 1] >= 0) & (gt[:, 1] < height)
            lo = xy[available].min(axis=1); hi = xy[available].max(axis=1)
            intersection = np.minimum(hi, (xs.max()+1, ys.max()+1)) - np.maximum(lo, (xs.min(), ys.min()))
            hits = np.flatnonzero(available)[np.all(intersection > 0, axis=1)]
            unique[t], missed[t], ambiguous[t] = len(hits) == 1, len(hits) == 0, len(hits) > 1
            if unique[t]:
                matched[t] = int(hits[0])
                for j in np.flatnonzero(valid[t]):
                    predicted = xy[matched[t], VERIFIED_JOINT_INDICES[j]].astype(np.float64)
                    value = math.hypot(float(predicted[0])-float(gt[j, 0]), float(predicted[1])-float(gt[j, 1]))
                    if not math.isfinite(value):
                        raise ValueError("Joint error overflowed; no finite-error fallback")
                    row_errors[j] = value
        others[t] = counts[t] - int(unique[t])
        errors.append(tuple(row_errors)); means.append(_mean(v for v in row_errors if v is not None))
    try:
        next(segments)
    except StopIteration:
        pass
    else:
        raise ValueError("Extra segmentation frame would change original timeline")
    scored = tuple(v for row in errors for v in row if v is not None)
    scored_frames = tuple(v for v in means if v is not None)
    summary = HandEvaluationCounts(frames, int(counts.sum()), int(np.count_nonzero(counts)), int(positive.sum()),
        int(unique.sum()), int(missed.sum()), int(ambiguous.sum()), int(unlabelled.sum()), int(no_mask.sum()),
        int(np.count_nonzero(positive & ~valid.any(axis=1))), int(valid.sum()), len(scored), len(scored_frames),
        int(others.sum()), int((counts-boxes).sum()), int(outside.sum()))
    status = "inconclusive" if not positive.any() else ("pass" if unique.any() else "fail")
    return HandClipEvaluation(clip, *map(_readonly, (indices, positive, unlabelled, no_mask, unique, missed,
        ambiguous, matched, counts, boxes, others, outside, valid)), tuple(errors), tuple(means), summary,
        _mean(scored), _mean(scored_frames), status)


def pool_hand_evaluations(clips):
    """Pool any nonempty frozen cohort (including the preregistered three clips).

    Joint-weighted EPE pools scored joints; frame mean weighs each scorable
    original frame equally, not each clip. Fail if ANY clip fails; a no-positive
    clip keeps the cohort inconclusive. No acceptance or accuracy claim follows.
    """
    if (type(clips) is not tuple or not clips or any(type(c) is not HandClipEvaluation for c in clips)
            or len({c.clip for c in clips}) != len(clips)):
        raise ValueError("Nonempty tuple of distinct clip evaluations required")
    fields = HandEvaluationCounts.__dataclass_fields__
    counts = HandEvaluationCounts(**{name: sum(getattr(c.counts, name) for c in clips) for name in fields})
    errors = (v for c in clips for row in c.joint_error_pixels for v in row if v is not None)
    means = (v for c in clips for v in c.frame_epe_pixels if v is not None)
    status = "fail" if any(c.diagnostic_status == "fail" for c in clips) else (
        "inconclusive" if any(c.diagnostic_status == "inconclusive" for c in clips) else "pass")
    return HandPooledEvaluation(clips, counts, _mean(errors), _mean(means), status)

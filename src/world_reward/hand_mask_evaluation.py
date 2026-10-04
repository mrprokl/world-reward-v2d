"""Paired union-mask diagnostics, never FP/identity or model-adoption decisions.

All annotated-positive frames count, including missing proposals and empty masks
with IoU/Dice zero. No hand pixels is unlabelled for this diagnostic, not a
certified negative. Contamination counts are pixel labels, not semantic FP truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hand_mask_proposals import HandMaskProposals
from .hand_temporal_masks import TemporalMaskFrame


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _mean(values):
    values = tuple(v for v in values if v is not None)
    return math.fsum(values) / len(values) if values else None


@dataclass(frozen=True, eq=False)
class MaskBranchEvaluation:
    predicted_pixels: np.ndarray  # Full T union, no duplicated instance pixels.
    intersection_pixels: np.ndarray
    union_pixels: np.ndarray
    object_label_pixels: np.ndarray  # seg in 1..21 intersect predicted union.
    background_label_pixels: np.ndarray  # seg == 0 intersect predicted union.
    supported_slot_count: np.ndarray
    unsupported_slot_count: np.ndarray
    supported_empty_slot_count: np.ndarray
    empty_union: np.ndarray
    iou: tuple[float | None, ...]
    dice: tuple[float | None, ...]
    mean_positive_iou: float | None
    mean_positive_dice: float | None
    positive_empty_union_frames: int
    total_predicted_pixels: int
    total_object_label_pixels: int
    total_background_label_pixels: int
    positive_predicted_pixels: int
    positive_object_label_pixels: int
    positive_background_label_pixels: int


@dataclass(frozen=True, eq=False)
class HandMaskEvaluation:
    frame_index: np.ndarray
    annotated_positive: np.ndarray
    unlabelled: np.ndarray
    gt_hand_pixels: np.ndarray
    proposal_count: np.ndarray
    b_intervention_slot_count: np.ndarray
    b_reuse_slot_count: np.ndarray
    positive_frames: int
    no_proposal_frames: int
    positive_no_proposal_frames: int
    b_intervention_frames: int
    positive_b_intervention_frames: int
    a: MaskBranchEvaluation
    b: MaskBranchEvaluation
    paired_dice_delta: tuple[float | None, ...]
    mean_positive_paired_dice_delta: float | None


@dataclass(frozen=True, eq=False)
class TemporalHandMaskEvaluation:
    frame_index: np.ndarray
    annotated_positive: np.ndarray
    unlabelled: np.ndarray
    gt_hand_pixels: np.ndarray
    proposal_count_a: np.ndarray  # IMAGE slots vary per frame.
    proposal_count_b: np.ndarray  # VIDEO anchor slots, not physical identities.
    positive_frames: int
    anchor_position: int | None
    seeded_proposal_ids: tuple[int, ...]
    a: MaskBranchEvaluation
    b: MaskBranchEvaluation
    paired_dice_delta: tuple[float | None, ...]
    mean_positive_paired_dice_delta: float | None


def _indices(frame_index):
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64 or indices.ndim != 1
            or not len(indices) or not np.array_equal(indices, np.arange(len(indices)))):
        raise ValueError("Full original nonempty int64 arange(T) required")
    return indices


def _segmentation(raw_seg, grid):
    seg = np.asarray(raw_seg)
    if (np.ma.isMaskedArray(raw_seg) or seg.dtype != np.uint8 or seg.shape != grid
            or not np.isin(seg, (*range(22), 255)).all()):
        raise ValueError("Original uint8 segmentation labels 0,1..21,255 required")
    return seg


def _branch_row(masks, supported, seg):
    n = len(supported); grid = seg.shape
    if (masks.dtype != np.bool_ or masks.shape != (n, *grid) or supported.dtype != np.bool_
            or supported.shape != (n,) or masks[~supported].any()):
        raise ValueError("Typed original masks/support; unsupported masks must stay empty")
    gt = seg == 255; g = int(gt.sum())
    union = masks.any(axis=0); size = int(union.sum()); intersection = int(np.count_nonzero(union & gt))
    total_union = size + g - intersection
    return (size, intersection, total_union, int(np.count_nonzero(union & (seg > 0) & (seg < 22))),
        int(np.count_nonzero(union & (seg == 0))), int(supported.sum()), n-int(supported.sum()),
        int(np.count_nonzero(supported & ~masks.any(axis=(1, 2)))), size == 0,
        intersection/total_union if g else None, 2*intersection/(size+g) if g else None)


def _branch_result(branch, positive):
    columns = list(zip(*branch)); numeric = tuple(_readonly(np.asarray(v, dtype=np.bool_ if i == 8 else np.int64)) for i, v in enumerate(columns[:9]))
    iou, dice = tuple(columns[9]), tuple(columns[10])
    return MaskBranchEvaluation(*numeric, iou, dice, _mean(iou), _mean(dice),
        int(np.count_nonzero(positive & numeric[8])), int(numeric[0].sum()), int(numeric[3].sum()), int(numeric[4].sum()),
        int(numeric[0][positive].sum()), int(numeric[3][positive].sum()), int(numeric[4][positive].sum()))


def evaluate_hand_masks(frame_index, proposals, gt_seg):
    """Stream frozen full-T HandMaskProposals and uint8 original-grid segmentation.

    Every slot enters A/B unions regardless of native score/support. Unsupported
    slots must be empty, as emitted by the producer; malformed ones fail rather
    than being silently filtered. Scores never select masks. A/B denominators
    are identical annotated-positive frames; no255 yields nullable IoU/Dice.
    B interventions count supplied non-reuse slots, even if masks happen equal.
    Means weight each positive original frame equally. Pool clips externally by
    these per-frame diagnostics, not by averaging unequal-size clip means.
    """
    indices = _indices(frame_index)
    p_iter, s_iter = iter(proposals), iter(gt_seg); rows = []; branches = [[], []]; grid = None
    for t in indices:
        try:
            p, raw_seg = next(p_iter), next(s_iter)
        except StopIteration:
            raise ValueError("Every original frame needs both proposal and GT records") from None
        if type(p) is not HandMaskProposals or type(p.frame_index) is not int or p.frame_index != t:
            raise ValueError("Matching original proposal frame required")
        arrays = [v for v in p.__dict__.values() if isinstance(v, np.ndarray)]
        if any(v.flags.writeable for v in arrays):
            raise ValueError("Already-frozen readonly proposals required")
        if grid is None:
            grid = p.image_size
        if (p.image_size != grid or type(grid) is not tuple or len(grid) != 2
                or any(type(v) is not int or v <= 0 for v in grid)):
            raise ValueError("Constant original image grid required")
        seg = _segmentation(raw_seg, grid)
        n = len(p.local_ids)
        if any(m.dtype != np.bool_ or m.shape != (n, *grid) for m in (p.masks_a, p.masks_b)):
            raise ValueError("Typed original masks for every retained slot required")
        if (p.b_reuses_a.dtype != np.bool_ or p.b_reuses_a.shape != (n,)
                or not np.array_equal(p.masks_a[p.b_reuses_a], p.masks_b[p.b_reuses_a])):
            raise ValueError("Explicit B reuse must preserve exact A masks")
        gt = seg == 255; g = int(gt.sum()); rows.append((g, n, int((~p.b_reuses_a).sum()), int(p.b_reuses_a.sum())))
        for j, (masks, supported) in enumerate(((p.masks_a, p.mask_supported_a), (p.masks_b, p.mask_supported_b))):
            branches[j].append(_branch_row(masks, supported, seg))
    end = object()
    for iterator in (p_iter, s_iter):
        if next(iterator, end) is not end:
            raise ValueError("Extra records may not extend/reindex the original timeline")
    gt_pixels, counts, interventions, reuses = map(np.asarray, zip(*rows)); positive = gt_pixels > 0
    output = [_branch_result(branch, positive) for branch in branches]
    delta = tuple(b-a if a is not None else None for a, b in zip(output[0].dice, output[1].dice))
    return HandMaskEvaluation(*map(_readonly, (indices, positive, ~positive, gt_pixels, counts, interventions, reuses)),
        int(positive.sum()), int(np.count_nonzero(counts == 0)), int(np.count_nonzero(positive & (counts == 0))),
        int(np.count_nonzero(interventions)), int(np.count_nonzero(positive & (interventions > 0))), *output, delta, _mean(delta))


def _temporal_frame(p, branch, t, grid):
    if (type(p) is not TemporalMaskFrame or p.branch != branch or type(p.position) is not int
            or type(p.original_frame_id) is not int or p.position != t or p.original_frame_id != t):
        raise ValueError("Paired chronological original TemporalMaskFrame records required")
    arrays = (p.raw_boxes, p.native_boxes, p.box_usable, p.masks, p.supported)
    if any(type(v) is not np.ndarray or v.flags.writeable for v in arrays):
        raise ValueError("Already-frozen readonly temporal arrays required")
    n = len(p.proposal_ids)
    if (type(p.proposal_ids) is not tuple or any(type(i) is not int for i in p.proposal_ids)
            or p.proposal_ids != tuple(range(n)) or p.raw_boxes.dtype != np.float64 or p.raw_boxes.shape != (n, 4)
            or p.native_boxes.dtype != np.float32 or p.native_boxes.shape != (n, 4)
            or p.box_usable.dtype != np.bool_ or p.box_usable.shape != (n,)
            or p.supported.dtype != np.bool_ or p.supported.shape != (n,)
            or p.masks.dtype != np.bool_ or p.masks.shape != (n, *grid) or p.masks[~p.supported].any()
            or not np.array_equal(p.supported, p.box_usable)):
        raise ValueError("Exact typed producer slots/boxes/support required; IDs are not actors")
    expected = np.clip(p.raw_boxes, [0, 0, 0, 0], [grid[1], grid[0], grid[1], grid[0]]).astype(np.float32)
    finite = np.isfinite(p.raw_boxes).all(axis=1)
    if (not np.all(finite | np.isnan(p.raw_boxes).all(axis=1))
            or not np.all(~finite | (p.raw_boxes[:, :2] <= p.raw_boxes[:, 2:]).all(axis=1))
            or not np.array_equal(expected, p.native_boxes, equal_nan=True)
            or np.any(p.box_usable & (~np.isfinite(p.raw_boxes).all(axis=1)
                | (p.native_boxes[:, 0] >= p.native_boxes[:, 2]) | (p.native_boxes[:, 1] >= p.native_boxes[:, 3])))):
        raise ValueError("Original clipped native box evidence differs")
    if branch == "A":
        scores = p.raw_scores
        if (type(scores) is not np.ndarray or scores.flags.writeable or scores.dtype != np.float64 or scores.shape != (n,)
                or not np.isfinite(scores[p.box_usable]).all() or not np.isnan(scores[~p.box_usable]).all()
                or p.evidence != "image_prompt"):
            raise ValueError("Raw IMAGE scores/evidence must retain exact supported slots")
    elif p.raw_scores is not None or p.evidence not in ("no_anchor_abstention", "anchor_prompt", "video_inferred"):
        raise ValueError("VIDEO evidence may not invent IMAGE scores or identities")
    return n


def _temporal_pairs(indices, paired_frames, metadata):
    iterator = iter(paired_frames); grid = None; anchor = None; signature = None; b_evidence = []
    for t in indices:
        try:
            pair = next(iterator)
        except StopIteration:
            raise ValueError("Every original frame needs both branches and GT") from None
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not TemporalMaskFrame:
            raise ValueError("Explicit typed (A, B) records required")
        a, b = pair
        if grid is None:
            if type(a.masks) is not np.ndarray or a.masks.ndim != 3 or min(a.masks.shape[1:]) <= 0:
                raise ValueError("Original positive image grid required")
            grid = a.masks.shape[1:]
        for p, name in zip(pair, ("A", "B")): _temporal_frame(p, name, int(t), grid)
        fixed = (b.proposal_ids, b.raw_boxes.tobytes(), b.native_boxes.tobytes(), b.box_usable.tobytes())
        if signature is None: signature = fixed
        if fixed != signature: raise ValueError("All VIDEO frames must preserve the same original anchor bank")
        if anchor is None and a.box_usable.any():
            anchor = int(t)
            if fixed != (a.proposal_ids, a.raw_boxes.tobytes(), a.native_boxes.tobytes(), a.box_usable.tobytes()):
                raise ValueError("VIDEO slots must be ALL original slots of the first usable IMAGE frame")
        b_evidence.append(b.evidence)
        yield pair
    end = object()
    if next(iterator, end) is not end:
        raise ValueError("Extra records may not extend/reindex the original timeline")
    expected = ["no_anchor_abstention"]*len(indices) if anchor is None else ["anchor_prompt" if t == anchor else "video_inferred" for t in indices]
    if b_evidence != expected or (anchor is None and signature[0]):
        raise ValueError("First-automatic-anchor or whole-clip no-anchor provenance differs")
    metadata.update(anchor_position=anchor, seeded_proposal_ids=tuple(map(int, np.flatnonzero(pair[1].box_usable))))


def validate_temporal_hand_frames(frame_index, paired_frames):
    """Full-T source-only lossless/provenance prepass; no private pixels or metric."""
    metadata = {}
    for _ in _temporal_pairs(_indices(frame_index), paired_frames, metadata): pass
    return metadata


def evaluate_temporal_hand_masks(frame_index, paired_frames, gt_seg):
    """Union diagnostics for IMAGE versus fixed-anchor VIDEO, different N allowed.

    ``paired_frames`` yields chronological (A, B) TemporalMaskFrame tuples. The
    producer's retrospective reverse masks remain at their original indices.
    All slots enter each union; missing/empty positive frames score zero. IDs
    certify anchor-slot provenance only, never physical identity or visibility.
    """
    indices = _indices(frame_index); segments = iter(gt_seg); metadata = {}; rows = []; branches = [[], []]
    for pair in _temporal_pairs(indices, paired_frames, metadata):
        try: raw_seg = next(segments)
        except StopIteration: raise ValueError("Every original frame needs GT") from None
        seg = _segmentation(raw_seg, pair[0].masks.shape[1:])
        rows.append((int(np.count_nonzero(seg == 255)), *(len(p.proposal_ids) for p in pair)))
        for j, p in enumerate(pair): branches[j].append(_branch_row(p.masks, p.supported, seg))
    end = object()
    if next(segments, end) is not end: raise ValueError("Extra GT may not extend/reindex the original timeline")
    gt_pixels, counts_a, counts_b = map(np.asarray, zip(*rows)); positive = gt_pixels > 0
    output = [_branch_result(branch, positive) for branch in branches]
    delta = tuple(b-a if a is not None else None for a, b in zip(output[0].dice, output[1].dice))
    # No GT chooses a slot, anchor, branch or a physical identity.
    return TemporalHandMaskEvaluation(*map(_readonly, (indices, positive, ~positive, gt_pixels, counts_a, counts_b)),
        int(positive.sum()), metadata['anchor_position'], metadata['seeded_proposal_ids'], *output, delta, _mean(delta))

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
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64 or indices.ndim != 1
            or not len(indices) or not np.array_equal(indices, np.arange(len(indices)))):
        raise ValueError("Full original nonempty int64 arange(T) required")
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
        seg = np.asarray(raw_seg)
        if (np.ma.isMaskedArray(raw_seg) or seg.dtype != np.uint8 or seg.shape != grid
                or not np.isin(seg, (*range(22), 255)).all()):
            raise ValueError("Original uint8 segmentation labels 0,1..21,255 required")
        n = len(p.local_ids)
        if any(m.dtype != np.bool_ or m.shape != (n, *grid) for m in (p.masks_a, p.masks_b)):
            raise ValueError("Typed original masks for every retained slot required")
        if (p.b_reuses_a.dtype != np.bool_ or p.b_reuses_a.shape != (n,)
                or not np.array_equal(p.masks_a[p.b_reuses_a], p.masks_b[p.b_reuses_a])):
            raise ValueError("Explicit B reuse must preserve exact A masks")
        gt = seg == 255; g = int(gt.sum()); rows.append((g, n, int((~p.b_reuses_a).sum()), int(p.b_reuses_a.sum())))
        for j, (masks, supported) in enumerate(((p.masks_a, p.mask_supported_a), (p.masks_b, p.mask_supported_b))):
            if (masks.dtype != np.bool_ or masks.shape != (n, *grid) or supported.dtype != np.bool_
                    or supported.shape != (n,) or masks[~supported].any()):
                raise ValueError("Typed original masks/support; unsupported masks must stay empty")
            union = masks.any(axis=0); size = int(union.sum()); intersection = int(np.count_nonzero(union & gt))
            total_union = size + g - intersection
            branches[j].append((size, intersection, total_union, int(np.count_nonzero(union & (seg > 0) & (seg < 22))),
                int(np.count_nonzero(union & (seg == 0))), int(supported.sum()), n-int(supported.sum()),
                int(np.count_nonzero(supported & ~masks.any(axis=(1, 2)))), size == 0,
                intersection/total_union if g else None, 2*intersection/(size+g) if g else None))
    end = object()
    for iterator in (p_iter, s_iter):
        if next(iterator, end) is not end:
            raise ValueError("Extra records may not extend/reindex the original timeline")
    gt_pixels, counts, interventions, reuses = map(np.asarray, zip(*rows)); positive = gt_pixels > 0
    output = []
    for branch in branches:
        columns = list(zip(*branch)); numeric = tuple(_readonly(np.asarray(v, dtype=np.bool_ if i == 8 else np.int64)) for i, v in enumerate(columns[:9]))
        iou, dice = tuple(columns[9]), tuple(columns[10])
        output.append(MaskBranchEvaluation(*numeric, iou, dice, _mean(iou), _mean(dice),
            int(np.count_nonzero(positive & numeric[8])), int(numeric[0].sum()), int(numeric[3].sum()), int(numeric[4].sum()),
            int(numeric[0][positive].sum()), int(numeric[3][positive].sum()), int(numeric[4][positive].sum())))
    delta = tuple(b-a if a is not None else None for a, b in zip(output[0].dice, output[1].dice))
    return HandMaskEvaluation(*map(_readonly, (indices, positive, ~positive, gt_pixels, counts, interventions, reuses)),
        int(positive.sum()), int(np.count_nonzero(counts == 0)), int(np.count_nonzero(positive & (counts == 0))),
        int(np.count_nonzero(interventions)), int(np.count_nonzero(positive & (interventions > 0))), *output, delta, _mean(delta))

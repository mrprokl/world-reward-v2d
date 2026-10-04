"""Paired native SAM2 prompts on every local hand slot; no actor/GT selection.

This callback operator does not load models or qualify sources, licences or
accuracy. Missing support is not an empty observed mask. Native scores are raw
predicted IoUs, never calibrated confidence or a target-selection signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_reward.automatic_candidate_bank import _sam2_outputs
from world_reward.hand_observations import HandInstances, _copy

POINT_INDICES = (0, *range(5, 21))


@dataclass(frozen=True, eq=False)
class HandMaskProposals:
    frame_index: int
    image_size: tuple[int, int]
    local_ids: tuple[str, ...]  # Frame-local proposal slots, not physical identity.
    original_xy: np.ndarray
    original_supported: np.ndarray
    raw_boxes: np.ndarray
    clipped_boxes: np.ndarray
    native_boxes: np.ndarray  # Explicit FP32 model prompt; never replaces raw box.
    box_usable: np.ndarray
    points17: np.ndarray
    native_points17: np.ndarray  # Explicit FP32 model prompt, no clipping/refill.
    points_usable: np.ndarray
    points_outside: np.ndarray
    masks_a: np.ndarray
    masks_b: np.ndarray
    mask_supported_a: np.ndarray
    mask_supported_b: np.ndarray
    raw_scores_a: np.ndarray
    raw_scores_b: np.ndarray
    b_reuses_a: np.ndarray
    diagnostics: tuple[str, ...]
    point_indices: tuple[int, ...] = POINT_INDICES


def propose_hand_masks(rgb, hands, sam2_predictor, *, frame_index):
    """A=box-only; B=same box+17 positive points, otherwise explicitly reuse A.

    Original RGB and HandInstances XY are never resized, cropped, mirrored or
    refilled. Boxes use min/max of ALL21 finite numerically available XY, then
    clip to [0,W]x[0,H]. Degenerate clipped/FP32-native boxes keep their original
    slot with mask support=False. Missing raw boxes are NaN diagnostics, not a
    manufactured zero box. The 17 native points [0,5..20] must ALL be available
    and inside [0,W)x[0,H) to use B, including after native FP32 conversion.
    Original and actual FP32 prompts are retained separately, never snapped.

    For any usable boxes, set_image ONCE; predict A ONCE with float32[N,4],
    multimask_output=False. Predict B ONCE for the eligible subbatch with the
    SAME boxes, float32[M,17,2] positives and int32[M,17] labels=1. Genuine SAM2
    N=1 squeeze vs N>1 native output shapes are validated without own thresholds.
    Empty native masks and every rejected/empty slot remain in original order.
    Empty/all-unusable frames make no model call. Errors propagate without retry.
    """
    image = np.asarray(rgb)
    if (np.ma.isMaskedArray(rgb) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1
            or type(hands) is not HandInstances or type(frame_index) is not int or frame_index < 0):
        raise ValueError("Original uint8 RGB grid, HandInstances and original integer index required")
    height, width = image.shape[:2]
    # Copy producer-owned evidence before callbacks; finite availability is not visibility.
    xy = _copy(hands.original_xy.values, "original XY", kind="f")
    support = _copy(hands.original_xy.supported, "original support", dtype=np.bool_)
    count = len(xy)
    points = _copy(xy[:, POINT_INDICES], "17 native original points", kind="f")
    finite = np.isfinite(points).all(axis=-1)
    outside = finite & ((points[..., 0] < 0) | (points[..., 0] >= width)
                        | (points[..., 1] < 0) | (points[..., 1] >= height))
    with np.errstate(over="ignore", invalid="ignore"):
        native_points = points.astype(np.float32)
    native_in_grid = np.isfinite(native_points).all(axis=-1) & (native_points[..., 0] >= 0) & (native_points[..., 0] < width) \
                     & (native_points[..., 1] >= 0) & (native_points[..., 1] < height)
    usable_points = finite & support[:, POINT_INDICES] & ~outside & native_in_grid
    boxes = np.full((count, 4), np.nan, np.float64)
    clipped = np.full_like(boxes, np.nan)
    full_finite = np.isfinite(xy).all(axis=(1, 2))
    boxes[full_finite, :2] = xy[full_finite].min(axis=1)
    boxes[full_finite, 2:] = xy[full_finite].max(axis=1)
    clipped[full_finite] = np.clip(boxes[full_finite], [0, 0, 0, 0], [width, height, width, height])
    native_boxes = clipped.astype(np.float32)
    box_usable = full_finite & support.all(axis=1) & (native_boxes[:, 0] < native_boxes[:, 2]) & (native_boxes[:, 1] < native_boxes[:, 3])
    eligible_b = box_usable & usable_points.all(axis=1)
    masks_a = np.zeros((count, height, width), np.bool_); masks_b = np.zeros_like(masks_a)
    observed_a = np.zeros(count, np.bool_); observed_b = np.zeros_like(observed_a)
    scores_a = np.full(count, np.nan, np.float64); scores_b = np.full_like(scores_a, np.nan)
    active = np.flatnonzero(box_usable); positive = np.flatnonzero(eligible_b)
    if len(active):
        if (not callable(getattr(sam2_predictor, "set_image", None))
                or not callable(getattr(sam2_predictor, "predict", None))):
            raise ValueError("Native SAM2 set_image/predict callbacks required")
        sam2_predictor.set_image(_copy(image, "original RGB", dtype=np.uint8))
        masks, scores = _sam2_outputs(sam2_predictor.predict(
            box=native_boxes[active].copy(), multimask_output=False), len(active), height, width)
        masks_a[active] = masks; scores_a[active] = scores; observed_a[active] = True
        masks_b[:] = masks_a; scores_b[:] = scores_a; observed_b[:] = observed_a
        if len(positive):
            masks, scores = _sam2_outputs(sam2_predictor.predict(
                box=native_boxes[positive].copy(), point_coords=native_points[positive].copy(),
                point_labels=np.ones((len(positive), 17), np.int32), multimask_output=False),
                len(positive), height, width)
            masks_b[positive] = masks; scores_b[positive] = scores; observed_b[positive] = True
    reasons = tuple("nonfinite_xy" if not full_finite[i] else "unavailable_xy" if not support[i].all()
                    else "degenerate_clipped_native_box" if not box_usable[i]
                    else "b_reuses_a_unusable_points" if not eligible_b[i] else "box_and_17_points"
                    for i in range(count))
    values = (xy, support, boxes, clipped, native_boxes, box_usable, points, native_points, usable_points, outside, masks_a, masks_b,
              observed_a, observed_b, scores_a, scores_b, ~eligible_b)
    owned = tuple(_copy(value, "owned proposal evidence") for value in values)
    return HandMaskProposals(frame_index, (height, width), tuple(f"hand:{i:06d}" for i in range(count)),
                             *owned, reasons)

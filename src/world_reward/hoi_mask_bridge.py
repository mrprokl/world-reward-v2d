"""Geometric HOI-box/mask correspondences, never identity or contact evidence.

Original FP32 xyxy boxes intersect binary pixel cells [x,x+1] x [y,y+1].
Areas are analytic cell intersections calculated in FP64, not centre sampling.
Outside-image box area remains in the IoU denominator. All positive overlaps
and one independent NULL route per detection survive; no mask is a NULL mask.
This module does not choose an association, assign weights or calibrate scores.
"""
from dataclasses import dataclass

import numpy as np

from .hoi_detr_observations import HOIDetrObservations
from .point_mask_association import MaskAnchor


def _owned(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class HOIMaskBridge:
    original_frame_index: int
    image_size: tuple[int, int]
    proposal_slots: np.ndarray
    retained_nms_positions: np.ndarray
    query_ids: np.ndarray
    class_ids: np.ndarray
    boxes_original_xyxy: np.ndarray
    raw_scores: np.ndarray
    decayed_scores: np.ndarray
    mask_ids: tuple[str, ...]
    masks: np.ndarray
    detector_box_area: np.ndarray       # [N], full box, including outside image
    mask_area: np.ndarray               # [M], binary pixel-cell area
    intersection_area: np.ndarray       # [N,M], FP64 analytic cell intersection
    box_iou: np.ndarray                 # [N,M], zero for empty union
    mask_coverage: np.ndarray           # [N,M], zero for empty mask
    supported: np.ndarray               # [N,M], strictly positive intersection
    detection_indices: np.ndarray       # [R], original retained detection slots
    mask_indices: np.ndarray            # [R], -1 is NULL, NEVER a mask index


def bridge_hoi_masks(observations: HOIDetrObservations, anchor: MaskAnchor):
    """Keep all rows/masks and all positive-overlap routes plus explicit NULL.

    Route order is detection-slot order, positive mask-slot order, then NULL.
    Degenerate (zero-area) boxes are retained with only NULL; inverted/nonfinite
    boxes fail. The frame and original (height,width) grid must match exactly.
    Input contracts supply automatic provenance; this math cannot certify it.
    """
    if (type(observations) is not HOIDetrObservations or type(anchor) is not MaskAnchor
            or observations.original_frame_index != anchor.frame_index
            or observations.image_size != anchor.masks.shape[1:]):
        raise ValueError("Matching original-frame HOI observations and mask grid required")
    boxes = observations.boxes_original_xyxy
    n, m = len(observations.query_ids), len(anchor.mask_ids)
    if (np.ma.isMaskedArray(boxes) or boxes.dtype != np.float32
            or boxes.shape != (n, 4) or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] < boxes[:, :2])):
        raise ValueError("Finite native FP32 xyxy boxes, not inverted, required")
    b = boxes.astype(np.float64)
    box_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    mask_area = anchor.masks.sum(axis=(1, 2), dtype=np.int64).astype(np.float64)
    intersection = np.zeros((n, m), np.float64)
    h, w = observations.image_size
    xs, ys = np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
    for i, (x0, y0, x1, y1) in enumerate(b):
        wx = np.maximum(0., np.minimum(xs + 1., x1) - np.maximum(xs, x0))
        wy = np.maximum(0., np.minimum(ys + 1., y1) - np.maximum(ys, y0))
        ix, iy = np.flatnonzero(wx), np.flatnonzero(wy)
        if m and len(ix) and len(iy):
            # Restrict integration to nonzero cell widths, never modify the box.
            xx, yy = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)
            intersection[i] = np.einsum("mhw,h,w->m", anchor.masks[:, yy, xx],
                                        wy[yy], wx[xx], dtype=np.float64)
    union = box_area[:, None] + mask_area[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    coverage = np.divide(intersection, mask_area[None, :],
                         out=np.zeros_like(intersection), where=mask_area[None, :] > 0)
    support = intersection > 0
    rows, columns = [], []
    for i in range(n):
        positive = np.flatnonzero(support[i])
        rows.extend([i] * (len(positive) + 1))
        columns.extend(positive.tolist() + [-1])
    arrays = (np.arange(n, dtype=np.int64), observations.retained_nms_positions,
              observations.query_ids, observations.class_ids, boxes,
              observations.raw_scores, observations.decayed_scores)
    return HOIMaskBridge(observations.original_frame_index, observations.image_size,
                         *map(_owned, arrays), anchor.mask_ids, _owned(anchor.masks),
                         *map(_owned, (box_area, mask_area, intersection, iou, coverage, support,
                                      np.asarray(rows, np.int64), np.asarray(columns, np.int64))))

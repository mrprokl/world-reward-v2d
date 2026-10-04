"""All automatic frame-zero hand/object candidates; no target selection or I/O.

Model callbacks are supplied by the caller. This module neither loads models
nor verifies source, licence, training overlap or inference accuracy. Stable
candidate IDs label proposals, never accepted physical identities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_reward.prompt_selection import BoxDetection, non_maximum_suppression

CONFIDENCE, NMS_IOU = .3, .7
QUERIES = (("hand", "hand."), ("object", "object."))


def _owned(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _real(value, shape, name, *, boolean=False):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.shape != shape
            or array.dtype.kind not in ("fiub" if boolean else "fiu") or not np.isfinite(array).all()):
        raise ValueError(f"{name}: finite real array with shape {shape} required")
    return _owned(array)


@dataclass(frozen=True, eq=False)
class DetectorRecords:
    """All raw finite box/score records, before bounds/confidence/NMS filtering.

    Original array dtypes/order are preserved. Finite invalid boxes/scores stay
    in this diagnostic record, but the unchanged existing NMS rejects them.
    Neither detector scores nor SAM2 predicted IoUs are calibrated confidence.
    """
    kind: str
    query: str
    boxes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, eq=False)
class AutomaticCandidate:
    kind: str
    stable_id: str
    box: np.ndarray
    detector_score: float
    sam2_score: float
    mask: np.ndarray
    query_points: np.ndarray


@dataclass(frozen=True, eq=False)
class AutomaticCandidateBank:
    """One original RGB grid, all retained candidates and outside-union queries.

    query_points are readonly float64[Q,3] (t=0,y,x) in original continuous
    pixel coordinates, using centers (column+.5,row+.5). Floor maps each query
    back to its actual mask pixel. This convention is explicit, not an inferred
    offset correction to native BootsTAPIR. The RGB/image grid is never resized
    here. Empty masks/candidates are retained with Q=0, no placeholder point.
    Background outside the union of ALL initial masks is a proxy, not certified
    stationary background or camera evidence. No later masks are manufactured.
    """
    image_size: tuple[int, int]  # height, width
    detector_records: tuple[DetectorRecords, ...]
    candidates: tuple[AutomaticCandidate, ...]
    background_query_points: np.ndarray


def _grid(box, count, mask, *, inside):
    """Floor fixed continuous bbox grid centers; unique pixel row-major, no refill."""
    x0, y0, x1, y1 = box
    xx = np.floor(x0 + (np.arange(count) + .5) * (x1 - x0) / count).astype(np.int64)
    yy = np.floor(y0 + (np.arange(count) + .5) * (y1 - y0) / count).astype(np.int64)
    x, y = np.meshgrid(xx, yy)
    yx = np.unique(np.column_stack((y.ravel(), x.ravel())), axis=0)
    selected = mask[yx[:, 0], yx[:, 1]]
    yx = yx[selected if inside else ~selected]
    return _owned(np.column_stack((np.zeros(len(yx)), yx.astype(np.float64) + .5)), np.float64)


def _sam2_outputs(outputs, count, height, width):
    if type(outputs) is not tuple or len(outputs) != 3:
        raise ValueError("SAM2 predict must return its native (masks, scores, low_res) tuple")
    masks, scores, low_res = outputs
    # Native predict uses squeeze(0), removing the batch axis ONLY when N=1.
    mask_shape = (1, height, width) if count == 1 else (count, 1, height, width)
    score_shape = (1,) if count == 1 else (count, 1)
    masks = _real(masks, mask_shape, "SAM2 masks", boolean=True)
    scores = _real(scores, score_shape, "SAM2 scores")
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("Native SAM2 binary masks required; no own logit threshold")
    if low_res is not None:
        low_shape = (1, 256, 256) if count == 1 else (count, 1, 256, 256)
        _real(low_res, low_shape, "SAM2 low-resolution logits")
    return masks.reshape(count, height, width), scores.reshape(count)


def build_automatic_candidate_bank(rgb, hand_detector, object_detector, sam2_predictor,
                                   *, hand_query="hand.", object_query="object."):
    """Run the fixed all-candidate frame-zero procedure through supplied models.

    Separate detector callbacks receive (readonly original uint8 RGB, exact
    query), returning (boxes[N,4], raw_scores[N]). Reuse existing classwise NMS
    with confidence .3/IoU .7; no hand/object ambiguity gate, target lookup,
    nearest selection, candidate cap or adaptive prompt exists. Candidates sort
    by (class, original xyxy) independent of detector input order.

    For N>0 call SAM2 set_image ONCE, then predict(box=float32[N,4],
    multimask_output=False) ONCE; preserve native single-mask postprocessing.
    Native predicted scores are retained without gating or probability claims.
    Q<=16 queries use a 4x4 grid in each original box and actual mask membership;
    Q<=64 background queries use an 8x8 whole-image grid outside ALL masks.
    No new point replaces a missed grid point. N=0 skips SAM2 and still returns
    both detector records and background queries. Missing classes/empty masks
    are diagnostic data for the caller's separately specified coverage gate.
    """
    if (type(hand_query) is not str or type(object_query) is not str
            or (hand_query, object_query) != ("hand.", "object.")):
        raise ValueError("Exact fixed hand. and object. query contracts required")
    image = np.asarray(rgb)
    if (np.ma.isMaskedArray(rgb) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Original uint8 RGB[H,W,3] required")
    height, width = image.shape[:2]
    image = _owned(image)
    if not callable(hand_detector) or not callable(object_detector):
        raise ValueError("Two automatic detector callbacks required")
    records, retained = [], []
    for (kind, query), detector in zip(QUERIES, (hand_detector, object_detector)):
        output = detector(image, query)
        if type(output) is not tuple or len(output) != 2:
            raise ValueError("Detector must return (raw_boxes, raw_scores)")
        boxes = np.asarray(output[0])
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("Detector boxes must have shape [N,4], including N=0")
        boxes = _real(output[0], boxes.shape, "Detector boxes")
        scores = _real(output[1], (len(boxes),), "Detector scores")
        records.append(DetectorRecords(kind, query, boxes, scores))
        detections = tuple(BoxDetection(tuple(map(float, box)), float(score)) for box, score in zip(boxes, scores))
        retained.extend((kind, item) for item in non_maximum_suppression(detections, width, height, CONFIDENCE, NMS_IOU))
    retained.sort(key=lambda pair: (pair[0], pair[1].box))
    union = np.zeros((height, width), dtype=bool)
    candidates = []
    if retained:
        if not callable(getattr(sam2_predictor, "set_image", None)) or not callable(getattr(sam2_predictor, "predict", None)):
            raise ValueError("Native SAM2 predictor set_image/predict API required")
        sam2_predictor.set_image(image)
        boxes = np.array([item.box for _, item in retained], dtype=np.float32)
        masks, scores = _sam2_outputs(sam2_predictor.predict(box=boxes, multimask_output=False), len(retained), height, width)
        for i, ((kind, detection), mask, score) in enumerate(zip(retained, masks, scores)):
            mask = _owned(mask, np.bool_)
            union |= mask
            candidates.append(AutomaticCandidate(kind, f"{kind}:{i:06d}", _owned(detection.box, np.float64),
                                                 detection.score, float(score), mask,
                                                 _grid(detection.box, 4, mask, inside=True)))
    return AutomaticCandidateBank((height, width), tuple(records), tuple(candidates),
                                  _grid((0, 0, width, height), 8, union, inside=False))

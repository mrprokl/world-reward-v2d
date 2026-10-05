"""Paired raw object-ranking scores on one complete native HOI proposal bank.

A is the lexicographically best negative (hand-centre to object-box distance,
hand-centre to object-centre distance) / original image diagonal. Larger is
better, with no scalar weighting or independently chosen distance minima.
B is the largest ORIGINAL interaction logit[1]-logit[0] over the SAME hands.
These are scores, not calibrated probabilities, contact or physical identity.
No object winner or object tie breaker, box matching, threshold or label is
produced. The anchor aggregation retains a tuple, not a selected hand identity.
"""
from dataclasses import dataclass
import math

import numpy as np

from .hoi_detr_observations import HOIDetrObservations


def _owned(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class HOIObjectRanking:
    original_frame_index: int
    image_size: tuple[int, int]
    object_slots: np.ndarray
    object_query_ids: np.ndarray
    object_class_ids: np.ndarray
    object_nms_positions: np.ndarray
    eligible_hand_slots: np.ndarray
    eligible_hand_query_ids: np.ndarray
    scores_a: np.ndarray              # [objects,2], lexicographic larger is better
    scores_b: np.ndarray
    supported: np.ndarray


def score_hoi_objects(observations: HOIDetrObservations, *, eligible_hand_slots=None):
    """Retain every role1 detection, including aliases, in original slot order.

    Explicit eligible slots must be a unique int64 vector of role0 retained
    detection indices (NOT query IDs/NMS indices). None uses all native hands;
    empty uses none. Supplied slot order is immaterial. No eligible hand means
    NaN/unsupported for every object, not a fake low-score ranking. Input source
    authenticity and any external eligibility semantics remain caller concerns.
    """
    if type(observations) is not HOIDetrObservations:
        raise ValueError("Complete native HOIDetrObservations required")
    o = observations; roles, boxes, queries = o.class_ids, o.boxes_original_xyxy, o.query_ids
    n = len(queries)
    if (roles.dtype != np.int64 or roles.shape != (n,) or np.any((roles < 0) | (roles > 2))
            or queries.dtype != np.int64 or queries.shape != (n,)
            or np.any((queries < 0) | (queries >= 1500))
            or boxes.dtype != np.float32 or boxes.shape != (n, 4)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2])):
        raise ValueError("Valid original roles/query IDs and finite FP32 xyxy boxes required")
    hands, objects = np.flatnonzero(roles == 0), np.flatnonzero(roles == 1)
    expected = np.column_stack((np.repeat(hands, len(objects)), np.tile(objects, len(hands))))
    pairs, logits = o.hand_object_pairs, o.hand_object_logits
    if (pairs.dtype != np.int64 or not np.array_equal(pairs, expected)
            or logits.dtype != np.float32 or logits.shape != (len(expected), 2)
            or not np.isfinite(logits).all()):
        raise ValueError("Complete original hand-object pairs and finite raw two-class logits required")
    selected = hands if eligible_hand_slots is None else np.asarray(eligible_hand_slots)
    if (np.ma.isMaskedArray(eligible_hand_slots) or selected.dtype != np.int64 or selected.ndim != 1
            or np.any((selected < 0) | (selected >= n))
            or len(np.unique(selected)) != len(selected) or np.any(roles[selected] != 0)):
        raise ValueError("Unique eligible native role0 int64 detection slots required")
    selected = np.sort(selected)
    diagonal = math.hypot(*o.image_size)
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Finite positive original image diagonal required")
    a, b = np.full((len(objects), 2), np.nan, np.float64), np.full(len(objects), np.nan, np.float64)
    support = np.full(len(objects), bool(len(selected)), bool)
    if len(selected) and len(objects):
        xy = boxes.astype(np.float64)
        centres = (xy[:, :2] + xy[:, 2:]) / 2
        delta = centres[selected, None, :] - centres[None, objects, :]
        distance_centre = np.hypot(delta[..., 0], delta[..., 1]) / diagonal
        hands_xy = centres[selected, None, :]
        distance_xy = np.maximum(np.maximum(xy[None, objects, :2] - hands_xy,
                                            hands_xy - xy[None, objects, 2:]), 0.)
        distance_box = np.hypot(distance_xy[..., 0], distance_xy[..., 1]) / diagonal
        for j in range(len(objects)):
            best = np.lexsort((distance_centre[:, j], distance_box[:, j]))[0]
            a[j] = -np.array([distance_box[best, j], distance_centre[best, j]])
        # FP64 subtraction avoids FP32 overflow and preserves the native logits.
        differences = (logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64))
        b[:] = differences.reshape(len(hands), len(objects))[np.searchsorted(hands, selected)].max(axis=0)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite paired object score")
    return HOIObjectRanking(o.original_frame_index, o.image_size,
        *map(_owned, (objects, queries[objects], roles[objects], o.retained_nms_positions[objects],
                      selected, queries[selected], a, b, support)))

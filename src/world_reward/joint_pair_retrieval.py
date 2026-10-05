"""Positive-only retrieval of a frozen automatic person/object pair bank.

This evaluator never selects inference crops, hands, targets or candidate rows.
Class-agnostic IoU matching happens AFTER caller-frozen scores. All reference
classes/boxes survive, including hierarchy duplicates and groups: ambiguity is
a miss. Unannotated winners mean positive-not-retrieved, NOT false relations,
contact/OFF labels, hand ownership, task identity or exhaustive precision.

Exact automatic (person box, object box) aliases count once in score ties and
must have identical scores. Pool real native route scores upstream WITHOUT GT;
this evaluator neither averages aliases nor collapses reference identities.
"""
from dataclasses import dataclass

import numpy as np

from .openimages_holds_reference import OpenImagesHoldsReference


def _snapshot(value, shape, name):
    if (type(value) is not np.ndarray or value.dtype not in (np.float32, np.float64)
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(name + ": finite plain FP32/FP64 array required")
    return np.array(value, dtype=np.float64, copy=True, order="C")


def _boxes(value, name):
    if type(value) is not np.ndarray or value.ndim != 2 or value.shape[1:] != (4,):
        raise ValueError(name + ": original normalized [N,4] xyxy required")
    boxes = _snapshot(value, value.shape, name)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = boxes[:, 2:] - boxes[:, :2]
        area = extent[:, 0] * extent[:, 1]
    if np.any(extent < 0) or not np.isfinite(extent).all() or not np.isfinite(area).all():
        raise ValueError(name + ": noninverted finite represented areas required")
    # Zero-area/off-grid native candidates remain competitors, never clipped.
    return boxes


def _sealed(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _aliases(boxes):
    first, inverse, seen = [], [], {}
    for i, box in enumerate(boxes):
        key = tuple(box)
        if key not in seen:
            seen[key] = len(first)
            first.append(i)
        inverse.append(seen[key])
    return np.asarray(first, np.int64), np.asarray(inverse, np.int64)


def _match(boxes, regions):
    reference = regions.boxes_xyxy
    if (reference.dtype != np.float64 or not np.isfinite(reference).all()
            or np.any((reference < 0) | (reference > 1))):
        raise ValueError("Original normalized parser reference boxes required")
    area_reference = np.prod(reference[:, 2:] - reference[:, :2], axis=1)
    result = np.full(len(boxes), -1, np.int64)
    for i, box in enumerate(boxes):
        width = np.maximum(0., np.minimum(box[2:], reference[:, 2:])
                           - np.maximum(box[:2], reference[:, :2]))
        intersection = width[:, 0] * width[:, 1]
        union = np.prod(box[2:] - box[:2]) + area_reference - intersection
        iou = np.divide(intersection, union, out=np.zeros(len(reference)), where=union > 0)
        matched = np.flatnonzero(iou >= .5)
        # Include ALL classes/groups in matching BEFORE the group censor.
        if len(matched) == 1 and not regions.group_of[matched[0]]:
            result[i] = matched[0]
    return result


@dataclass(frozen=True, eq=False)
class JointPairRetrieval:
    image_positive_retrieval: float
    top_score: float | None
    top_unique_pair_count: int
    top_annotated_positive_pair_count: int
    raw_person_proposal_count: int
    raw_object_proposal_count: int
    unique_person_box_count: int
    unique_object_box_count: int
    unique_pair_count: int
    person_reference_indices: np.ndarray  # original automatic row order; -1 unknown
    object_reference_indices: np.ndarray
    resolved_positive_pair_count: int
    unbound_positive_relation_count: int
    reference_positive_pair_slots: int  # unique resolved pairs + unbound relations
    retrieved_positive_pair_count: int
    positive_pair_proposal_recall: float
    resolved_pair_person_count: int
    retrieved_resolved_pair_person_count: int
    resolved_pair_person_proposal_recall: float | None
    resolved_pair_object_count: int
    retrieved_resolved_pair_object_count: int
    resolved_pair_object_proposal_recall: float | None
    total_positive_person_slots: int
    unscorable_positive_person_slots: int
    unbound_person_endpoint_count: int
    unbound_object_endpoint_count: int


def evaluate_joint_pair_retrieval(person_boxes, object_boxes, scores, reference):
    """Score ALL dense N×M pairs once; exact numeric ties get fractional credit.

    Caller authenticates/freezes predictions. Scores may be negative but MUST
    all be finite; unsupported evidence requires caller-level abstention, not a
    selective row drop here. Boxes are original-grid normalized, with no image
    alignment/rotation/clipping. FP32 inputs are copied exactly into FP64.

    Reference must be the original parse_holds_reference result with >=1 manual
    positive. Missing banks/unresolved reference positives cannot create hits.
    Pair proposal recall retains unbound relations in its denominator. Endpoint
    recalls explicitly concern unique endpoints of RESOLVED pairs only: this
    parser does not retain every otherwise-bound positive object endpoint.
    """
    p, o = _boxes(person_boxes, "Automatic person boxes"), _boxes(object_boxes, "Automatic object boxes")
    bank = _snapshot(scores, (len(p), len(o)), "Complete joint score bank")
    pi, pa = _aliases(p)
    oi, oa = _aliases(o)
    unique_scores = bank[np.ix_(pi, oi)]
    if not np.array_equal(bank, unique_scores[np.ix_(pa, oa)]):
        raise ValueError("Exact automatic box-pair aliases require equal frozen scores")
    if type(reference) is not OpenImagesHoldsReference:
        raise ValueError("Original parse_holds_reference result required")
    links = reference.holds
    diagnostic = reference.binding_diagnostics
    counts = [diagnostic.get(k) for k in ("unbound_positive_relations", "unbound_person_endpoints", "unbound_object_endpoints")]
    if (any(type(x) is not int or x < 0 for x in counts)
            or type(reference.total_positive_person_slots) is not int or reference.total_positive_person_slots < 1
            or links.dtype != np.int64 or links.ndim != 2 or links.shape[1:] != (2,)
            or np.any(links < 0) or np.any(links[:, 0] >= len(reference.persons.instance_ids))
            or np.any(links[:, 1] >= len(reference.objects.instance_ids))
            or len(np.unique(links, axis=0)) != len(links) or len(links) + counts[0] < 1):
        raise ValueError("Original positive scope and nonnegative parser binding counts required")
    pm, om = _match(p, reference.persons), _match(o, reference.objects)
    positive = {tuple(map(int, pair)) for pair in links}
    persons, objects = {int(x) for x in pm if x >= 0}, {int(x) for x in om if x >= 0}
    rp, ro = {a for a, _ in positive}, {b for _, b in positive}
    retrieved = sum(a in persons and b in objects for a, b in positive)
    top, tied, hit = None, 0, 0
    if unique_scores.size:
        top = float(unique_scores.max())
        rows, columns = np.nonzero(unique_scores == top)
        tied = len(rows)
        hit = sum((int(pm[pi[a]]), int(om[oi[b]])) in positive for a, b in zip(rows, columns))
    slots = len(positive) + counts[0]
    return JointPairRetrieval(
        hit / tied if tied else 0., top, tied, hit, len(p), len(o), len(pi), len(oi), len(pi) * len(oi),
        _sealed(pm), _sealed(om), len(positive), counts[0], slots, retrieved, retrieved / slots,
        len(rp), len(rp & persons), len(rp & persons) / len(rp) if rp else None,
        len(ro), len(ro & objects), len(ro & objects) / len(ro) if ro else None,
        reference.total_positive_person_slots, reference.unbound_positive_person_count, counts[1], counts[2],
    )

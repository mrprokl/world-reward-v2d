"""Separate person/object proposal ceilings, never actor or interaction truth.

All OWLv2 observations survive. Fixed rank budgets are evaluator-only views,
not a production selector; native objectness logits are not class confidence.
One proposal can recover at most one annotation in the primary matching metric.
COCO boxes are continuous XYWH, NOT V-COCO's inclusive official-v3 AP convention.
"""
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np

from .owlv2_object_observations import Owlv2ObjectObservations
from .proposal_bbox_recall import evaluate_bbox_proposals
from .proposal_recall import THRESHOLDS, _require, _readonly

BUDGETS = (32, 128, 3600)
PRIMARY_BUDGET = 128
GATE = .70


def _xywh(boxes):
    _require(type(boxes) is np.ndarray and boxes.ndim == 2 and boxes.shape[1] == 4
             and boxes.dtype.kind in 'fiu' and np.isfinite(boxes).all(),
             'Complete finite original-pixel XYXY proposal array required')
    value = boxes.astype(np.float64, copy=True)
    _require(np.all(value[:, 2:] >= value[:, :2]), 'Inverted proposals fail, not clipped or repaired')
    value[:, 2:] -= value[:, :2]
    return value


@dataclass(frozen=True)
class EndpointProposalRecall:
    person: object
    objects: object  # fixed budget -> complete BBoxProposalRecall view
    ranked_native_patch_ids: np.ndarray
    full_native_object_count: int
    original_frame_index: int
    image_size: tuple[int, int]
    scope: str = 'separate_endpoint_localization_only_not_pair_or_ownership'


def evaluate_endpoint_proposals(person_boxes_xyxy, objects, truth_boxes_xywh,
                                entity_kinds, *, entity_ids):
    """Use private references only in evaluation, after full banks are frozen.

    Rank descending native FP32 logit, exact ties by original row-major patch
    ID. No GT-derived ranking, clipping, deduplication, best-budget gate or NMS.
    Every original countable P/O instance contributes, including duplicate boxes
    with distinct annotation IDs. Missing/inaccurate endpoints are not injected.
    """
    _require(type(objects) is Owlv2ObjectObservations, 'Full native object bank required')
    objects = Owlv2ObjectObservations(**{f.name: getattr(objects, f.name) for f in fields(objects)})
    person = _xywh(person_boxes_xyxy)
    _require(type(truth_boxes_xywh) is np.ndarray and truth_boxes_xywh.ndim == 2
             and truth_boxes_xywh.shape[1] == 4 and truth_boxes_xywh.dtype.kind in 'fiu'
             and np.isfinite(truth_boxes_xywh).all() and np.all(truth_boxes_xywh[:, 2:] > 0),
             'All original countable continuous-XYWH references required')
    count = len(truth_boxes_xywh)
    _require(type(entity_kinds) is tuple and len(entity_kinds) == count
             and all(x in ('human', 'object') for x in entity_kinds)
             and entity_kinds.count('human') >= 2 and entity_kinds.count('object') >= 2,
             'Frozen multiperson/multiobject reference population required')
    _require(type(entity_ids) is tuple and len(entity_ids) == len(set(entity_ids)) == count
             and all(type(x) is str and x for x in entity_ids), 'Distinct original annotation IDs required')
    p = np.asarray(entity_kinds) == 'human'; o = ~p
    def score(boxes, chosen, kind):
        return evaluate_bbox_proposals(boxes, truth_boxes_xywh[chosen],
            (kind,)*int(chosen.sum()), entity_ids=tuple(x for x, yes in zip(entity_ids, chosen) if yes))
    rank = np.lexsort((objects.patch_ids, -objects.objectness_logits))
    native = _xywh(objects.boxes_original_xyxy)
    result = {k: score(native[rank[:k]], o, 'object') for k in BUDGETS}
    return EndpointProposalRecall(score(person, p, 'human'), MappingProxyType(result),
        _readonly(objects.patch_ids[rank]), len(native), objects.original_frame_index, objects.image_size)


def _aggregate(rows, fixed_slots):
    recalls = {}
    for threshold in THRESHOLDS:
        values = [r.metrics['recalls'][str(threshold)]['all'] for r in rows]
        recalls[str(threshold)] = dict(
            acquired_instances=sum(r['eligible'] for r in values),
            best_iou_recovered_acquired_instances=sum(r['recovered'] for r in values),
            one_to_one_recovered_acquired_instances=sum(r['one_to_one_recovered'] for r in values),
            macro_fixed_slot_best_iou_recall=sum(r['recall'] for r in values)/fixed_slots,
            macro_fixed_slot_one_to_one_recall=sum(r['one_to_one_recall'] for r in values)/fixed_slots,
        )
    # A matched-proposal fraction is diagnostic capacity, not exhaustive COCO
    # foreground precision: unannotated objects/parts can remain legitimate.
    return dict(recalls=recalls, native_boxes=sum(len(r.native_boxes_xywh) for r in rows),
        matched_proposal_fraction_at_half=(
            sum(r.metrics['recalls']['0.5']['all']['one_to_one_recovered'] for r in rows)
            /sum(len(r.native_boxes_xywh) for r in rows)
            if sum(len(r.native_boxes_xywh) for r in rows) else None))


def aggregate_endpoint_fixed_slots(results):
    """Every frozen slot contributes: unavailable RGB/banks score zero.

    The fixed conjunction tests persons and O@128 independently. No compensation
    via objects/persons, O@3600, pooled recall or successful-only denominator.
    This function does not open references or authorize reserved access itself.
    """
    _require(type(results) in (list, tuple) and len(results) > 0
             and all(r is None or type(r) is EndpointProposalRecall for r in results),
             'Complete fixed ordered endpoint slot list required')
    present = [r for r in results if r is not None]; n = len(results)
    for r in present:
        _require(tuple(r.objects) == BUDGETS and r.full_native_object_count == 3600
                 and np.array_equal(np.sort(r.ranked_native_patch_ids), np.arange(3600)),
                 'Full original object bank and exact declared budgets required')
    person = _aggregate([r.person for r in present], n)
    objects = {str(k): _aggregate([r.objects[k] for r in present], n) for k in BUDGETS}
    primary_p = person['recalls']['0.5']['macro_fixed_slot_one_to_one_recall']
    primary_o = objects[str(PRIMARY_BUDGET)]['recalls']['0.5']['macro_fixed_slot_one_to_one_recall']
    return dict(slots=n, acquired_slots=len(present), missing_slots=n-len(present),
        persons=person, objects=objects, primary=dict(iou=.5, object_budget=PRIMARY_BUDGET,
            person_gate=GATE, object_gate=GATE, person_recall=primary_p, object_recall=primary_o,
            both_endpoint_gates_pass=primary_p >= GATE and primary_o >= GATE),
        missing_slot_recall=0., all_slot_instance_micro_available=False,
        rank_views_evaluation_only=True, whole_object_banks_retained=True,
        scope='endpoint_capacity_not_relation_or_ownership', quality_verified=False,
        training_overlap_verified=False, adoption=False)

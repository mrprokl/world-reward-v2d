"""Complete native box proposal capacity, not semantics or target selection.

XYWH is interpreted exactly as serialized, with no clipping, +1 pixel repair,
deduplication or area-based selection. Private annotations are evaluator-only.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .proposal_recall import THRESHOLDS, _matching, _readonly, _require


def _boxes(value, *, positive, name):
    _require(type(value) is np.ndarray and value.ndim == 2 and value.shape[1] == 4
             and value.dtype.kind in 'fiu' and np.isfinite(value).all(),
             f'Finite original {name} ndarray XYWH required')
    result = value.astype(np.float64, copy=True)
    _require(np.all(result[:, 2:] > 0 if positive else result[:, 2:] >= 0),
             f'Original {name} extents invalid')
    _require(np.isfinite(result[:, :2] + result[:, 2:]).all()
             and np.isfinite(result[:, 2] * result[:, 3]).all(), 'Box arithmetic overflow')
    return result


@dataclass(frozen=True)
class BBoxProposalRecall:
    entity_ids: tuple[str, ...]
    entity_kinds: tuple[str, ...]
    iou: np.ndarray
    best_iou: np.ndarray
    matching_slots: np.ndarray
    native_boxes_xywh: np.ndarray
    generate_seconds: float | None
    metrics: object


def evaluate_bbox_proposals(native_boxes_xywh, truth_boxes_xywh, entity_kinds, *,
                            entity_ids, generate_seconds=None):
    """Retain every native slot, including duplicate and zero-extent boxes.

    Truth instances must be distinct countable annotation IDs. Identical boxes
    can still describe different physical instances; they are not deduplicated.
    Matching is diagnostic capacity, never an output interaction assignment.
    """
    native = _boxes(native_boxes_xywh, positive=False, name='native')
    truth = _boxes(truth_boxes_xywh, positive=True, name='truth')
    g, n = len(truth), len(native)
    _require(g > 0 and type(entity_ids) is tuple and len(entity_ids) == len(set(entity_ids)) == g
             and all(type(s) is str and s for s in entity_ids), 'Distinct private annotation IDs required')
    _require(type(entity_kinds) is tuple and len(entity_kinds) == g
             and all(k in ('human', 'object') for k in entity_kinds), 'Original instance kinds required')
    _require(generate_seconds is None or (type(generate_seconds) in (int, float)
             and math.isfinite(generate_seconds) and generate_seconds >= 0), 'Finite native generation cost required')
    lo = np.maximum(truth[:, None, :2], native[None, :, :2])
    hi = np.minimum(truth[:, None, :2] + truth[:, None, 2:], native[None, :, :2] + native[None, :, 2:])
    wh = np.maximum(hi - lo, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = truth[:, 2, None] * truth[:, 3, None] + native[None, :, 2] * native[None, :, 3] - intersection
    _require(np.isfinite(intersection).all() and np.isfinite(union).all() and np.all(union > 0),
             'Finite positive box union required')
    iou = intersection / union
    best = iou.max(1) if n else np.zeros(g, np.float64)
    kinds = np.array(entity_kinds); matching = []; recalls = {}
    for threshold in THRESHOLDS:
        edges = iou >= threshold; matching.append(_matching(edges)); groups = {}
        for kind, chosen in (('all', np.ones(g, bool)), ('human', kinds == 'human'), ('object', kinds == 'object')):
            count = int(chosen.sum()); recovered = int(np.count_nonzero(best[chosen] >= threshold))
            unique = int(np.count_nonzero(_matching(edges[chosen]) >= 0))
            groups[kind] = MappingProxyType(dict(eligible=count, recovered=recovered, one_to_one_recovered=unique,
                recall=recovered/count if count else None, one_to_one_recall=unique/count if count else None))
        recalls[str(threshold)] = MappingProxyType(groups)
    metrics = MappingProxyType(dict(native_boxes=n, truth_instances=g, recalls=MappingProxyType(recalls),
        kind_matching_scope='independent_within_kind_nonadditive',
        scope='bbox_proposal_localization_only', ownership_verified=False, accuracy_verified=False, adoption=False))
    return BBoxProposalRecall(entity_ids, entity_kinds, *map(_readonly, (iou, best, np.stack(matching), native)),
        None if generate_seconds is None else float(generate_seconds), metrics)


def aggregate_fixed_slot_bbox_recall(results):
    """Image-balanced scores over every frozen slot; missing images score zero.

    Missing slots have unknown instance counts here. Report no misleading
    all-slot instance-micro statistic or successful-acquisitions-only gate.
    """
    _require(type(results) in (tuple, list) and len(results) > 0
             and all(r is None or type(r) is BBoxProposalRecall for r in results), 'Complete fixed slot list required')
    present = [r for r in results if r is not None]
    recalls = {}
    for threshold in THRESHOLDS:
        groups = {}
        for kind in ('all', 'human', 'object'):
            rows = [r.metrics['recalls'][str(threshold)][kind] for r in present]
            _require(all(row['eligible'] > 0 for row in rows), 'Each acquired crowd slot must support both kinds')
            groups[kind] = dict(fixed_slots=len(results), acquired_instances=sum(x['eligible'] for x in rows),
                recovered_acquired_instances=sum(x['recovered'] for x in rows),
                macro_fixed_slot_recall=sum(x['recall'] for x in rows)/len(results),
                macro_fixed_slot_one_to_one_recall=sum(x['one_to_one_recall'] for x in rows)/len(results))
        recalls[str(threshold)] = groups
    costs = [r.generate_seconds for r in present if r.generate_seconds is not None]
    return dict(slots=len(results), acquired_slots=len(present), missing_slots=len(results)-len(present),
        native_boxes=sum(len(r.native_boxes_xywh) for r in present), recalls=recalls,
        native_generate_seconds=sum(costs) if costs else None, cost_images=len(costs),
        missing_slot_recall=0., all_slot_instance_micro_available=False,
        scope='bbox_proposal_localization_only', ownership_verified=False, accuracy_verified=False, adoption=False)

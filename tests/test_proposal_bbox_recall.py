"""Two-pixel fixtures and exhaustive small matching; no media or labels."""
import itertools

import numpy as np
import pytest

from world_reward.proposal_bbox_recall import evaluate_bbox_proposals as score, aggregate_fixed_slot_bbox_recall as aggregate
from world_reward.proposal_recall import _matching


def fixture():
    boxes = np.array([[0, 0, 2, 2], [4, 0, 2, 2]], np.float64)
    return boxes, boxes.copy(), ('human', 'object'), ('p', 'o')


def test_exact_boxes_original_order_readonly_no_selection():
    p, t, kinds, ids = fixture(); r = score(p, t, kinds, entity_ids=ids, generate_seconds=.1)
    assert r.iou.tolist() == [[1, 0], [0, 1]]
    assert r.matching_slots.tolist() == [[0, 1]] * 3
    assert np.array_equal(r.native_boxes_xywh, p)
    p[:] = 99; t[:] = 99
    assert r.best_iou.tolist() == [1, 1]
    for a in (r.iou, r.best_iou, r.matching_slots, r.native_boxes_xywh):
        assert not a.flags.writeable
    with pytest.raises(TypeError): r.metrics['adoption'] = True


def test_duplicate_physical_instances_and_empty_native_bank():
    box = np.array([[0., 0., 2., 2.]])
    r = score(box, np.repeat(box, 2, axis=0), ('human', 'object'), entity_ids=('p', 'o'))
    assert r.metrics['recalls']['0.5']['all']['recall'] == 1
    assert r.metrics['recalls']['0.5']['all']['one_to_one_recall'] == .5
    assert r.metrics['recalls']['0.5']['human']['one_to_one_recall'] == 1
    assert r.metrics['recalls']['0.5']['object']['one_to_one_recall'] == 1
    empty = score(np.empty((0, 4)), box, ('human',), entity_ids=('p',))
    assert empty.best_iou.tolist() == [0] and empty.matching_slots.tolist() == [[-1]] * 3


def test_exact_xywh_no_inclusive_pixel_repair_clipping_or_dedup():
    t = np.array([[0., 0., 2., 2.]])
    p = np.array([[0., 0., 1., 2.], [0., 0., 1., 2.], [-2., 0., 2., 2.], [0., 0., 0., 2.]])
    r = score(p, t, ('human',), entity_ids=('p',))
    assert r.iou.tolist() == [[.5, .5, 0., 0.]]
    assert len(r.native_boxes_xywh) == 4
    assert r.metrics['recalls']['0.5']['all']['recovered'] == 1
    assert r.metrics['recalls']['0.75']['all']['recovered'] == 0


def test_fixed_missing_slots_are_misses_not_dropped_from_gate():
    p, t, kinds, ids = fixture(); r = score(p, t, kinds, entity_ids=ids)
    a = aggregate([r, None, None, r])
    assert a['recalls']['0.5']['all']['macro_fixed_slot_recall'] == .5
    assert a['missing_slots'] == 2 and not a['all_slot_instance_micro_available']
    assert aggregate([None, None])['recalls']['0.5']['all']['macro_fixed_slot_recall'] == 0


@pytest.mark.parametrize('bad', ['list', 'mask', 'bool', 'shape', 'nan', 'inf', 'negative', 'truth_zero', 'overflow',
                                    'kinds_list', 'bad_kind', 'ids_list', 'duplicate_id', 'empty_id', 'cost_bool', 'cost_nan'])
def test_malformed_arrays_or_ledger_rejected_without_repair(bad):
    p, t, kinds, ids = fixture(); cost = None
    if bad == 'list': p = p.tolist()
    elif bad == 'mask': p = np.ma.array(p)
    elif bad == 'bool': p = p.astype(bool)
    elif bad == 'shape': p = p[:, :3]
    elif bad == 'nan': p[0, 0] = np.nan
    elif bad == 'inf': p[0, 0] = np.inf
    elif bad == 'negative': p[0, 2] = -1
    elif bad == 'truth_zero': t[0, 2] = 0
    elif bad == 'overflow': p[:] = 1e308
    elif bad == 'kinds_list': kinds = list(kinds)
    elif bad == 'bad_kind': kinds = ('hand', 'object')
    elif bad == 'ids_list': ids = list(ids)
    elif bad == 'duplicate_id': ids = ('p', 'p')
    elif bad == 'empty_id': ids = ('', 'o')
    elif bad == 'cost_bool': cost = True
    else: cost = float('nan')
    with np.errstate(over='ignore'), pytest.raises(ValueError): score(p, t, kinds, entity_ids=ids, generate_seconds=cost)


@pytest.mark.parametrize('value', [[], {}, [False], [np.array([1])]])
def test_aggregate_needs_complete_typed_slots(value):
    with pytest.raises(ValueError): aggregate(value)


def test_aggregate_rejects_unsupported_present_categories():
    box = np.array([[0., 0., 1., 1.]])
    r = score(box, box, ('human',), entity_ids=('p',))
    with pytest.raises(ValueError): aggregate([r])


def test_exact_maxcardinality_for_every_three_by_three_graph():
    for bits in itertools.product((False, True), repeat=9):
        edges = np.array(bits).reshape(3, 3); expected = 0
        for columns in itertools.product((-1, 0, 1, 2), repeat=3):
            used = [c for c in columns if c >= 0]
            if len(used) == len(set(used)) and all(c < 0 or edges[r, c] for r, c in enumerate(columns)):
                expected = max(expected, len(used))
        got = _matching(edges)
        assert np.count_nonzero(got >= 0) == expected
        assert len(set(got[got >= 0])) == expected

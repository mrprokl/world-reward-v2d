"""Tiny authored boxes/full-grid arrays only; no COCO/GT/model/RGB access."""
from dataclasses import replace

import numpy as np
import pytest

from world_reward import endpoint_proposal_recall as p
from world_reward.owlv2_object_observations import Owlv2ObjectObservations, GRID


def fixture():
    truth = np.array([[0, 0, 2, 2], [4, 0, 2, 2],
                      [0, 4, 2, 2], [4, 4, 2, 2]], np.float64)
    normalized = np.zeros((3600, 4), np.float32)
    normalized[:2] = [[.125, .625, .25, .25], [.625, .625, .25, .25]]
    corners = np.concatenate((normalized[:, :2]-normalized[:, 2:]/2,
                              normalized[:, :2]+normalized[:, 2:]/2), axis=1)*np.float32(8)
    objects = Owlv2ObjectObservations(5, (8, 8), GRID, np.arange(3600, dtype=np.int64),
        normalized, np.zeros(3600, np.float32), corners)
    person = np.array([[0, 0, 2, 2], [4, 0, 6, 2]], np.float64)
    return person, objects, truth, ('human', 'human', 'object', 'object'), ('p0', 'p1', 'o0', 'o1')


def score(values=None):
    person, objects, truth, kinds, ids = fixture() if values is None else values
    return p.evaluate_endpoint_proposals(person, objects, truth, kinds, entity_ids=ids)


def test_separate_primary_full_bank_stable_ties_one_to_one():
    r = score(); a = p.aggregate_endpoint_fixed_slots([r])
    assert a['primary']['both_endpoint_gates_pass']
    assert a['primary']['person_recall'] == a['primary']['object_recall'] == 1.
    assert tuple(r.objects) == (32, 128, 3600)
    assert r.full_native_object_count == 3600
    assert np.array_equal(r.ranked_native_patch_ids, np.arange(3600))
    assert a['objects']['128']['matched_proposal_fraction_at_half'] == 2/128
    assert not r.ranked_native_patch_ids.flags.writeable


def test_missing_person_never_compensated_by_dense_object_bank_or_omitted():
    values = list(fixture()); values[0] = np.empty((0, 4), np.float64)
    a = p.aggregate_endpoint_fixed_slots([score(tuple(values))])
    assert a['primary']['person_recall'] == 0 and a['primary']['object_recall'] == 1
    assert not a['primary']['both_endpoint_gates_pass']
    a = p.aggregate_endpoint_fixed_slots([score(), None])
    assert a['primary']['person_recall'] == a['primary']['object_recall'] == .5
    assert a['missing_slots'] == 1 and not a['all_slot_instance_micro_available']
    assert not a['primary']['both_endpoint_gates_pass']


def test_budget_failure_not_rescued_by_all_patches_or_gt_best_rank():
    values = list(fixture()); b = values[1]
    logits = np.ones(3600, np.float32); logits[:2] = -999
    values[1] = replace(b, objectness_logits=logits)
    r = score(tuple(values)); a = p.aggregate_endpoint_fixed_slots([r])
    assert r.ranked_native_patch_ids[-2:].tolist() == [0, 1]
    assert a['objects']['3600']['recalls']['0.5']['macro_fixed_slot_one_to_one_recall'] == 1
    assert a['primary']['object_recall'] == 0 and not a['primary']['both_endpoint_gates_pass']
    assert np.array_equal(values[1].objectness_logits, logits)


def test_duplicate_annotations_are_distinct_and_one_proposal_gets_only_one_vote():
    values = list(fixture()); values[2][3] = values[2][2]
    a = p.aggregate_endpoint_fixed_slots([score(tuple(values))])
    v = a['objects']['128']['recalls']['0.5']
    assert v['macro_fixed_slot_best_iou_recall'] == 1
    assert v['macro_fixed_slot_one_to_one_recall'] == .5
    assert not a['primary']['both_endpoint_gates_pass']


def test_no_clipping_outside_image_box_zero_area_diagnostics_retained():
    values = list(fixture()); values[0] = np.array([[-2, 0, 2, 2], [4, 0, 6, 2], [0, 0, 0, 0]], np.float64)
    r = score(tuple(values)); assert len(r.person.native_boxes_xywh) == 3
    assert r.person.iou[0, 0] == .5 and not r.person.iou[:, 2].any()


@pytest.mark.parametrize('bad', ['person_list', 'person_nan', 'person_inverted', 'object_type',
                                'truth_zero', 'truth_nan', 'one_person', 'duplicate_ids', 'kind_list'])
def test_bad_protocol_inputs_fail_without_repair(bad):
    values = list(fixture())
    if bad == 'person_list': values[0] = values[0].tolist()
    elif bad == 'person_nan': values[0][0, 0] = np.nan
    elif bad == 'person_inverted': values[0][0, 2] = -5
    elif bad == 'object_type': values[1] = {}
    elif bad == 'truth_zero': values[2][0, 2] = 0
    elif bad == 'truth_nan': values[2][0, 0] = np.nan
    elif bad == 'one_person': values[3] = ('human', 'object', 'object', 'object')
    elif bad == 'duplicate_ids': values[4] = ('p0', 'p0', 'o0', 'o1')
    else: values[3] = list(values[3])
    with pytest.raises(ValueError): score(tuple(values))


def test_all_missing_slots_zero_without_fake_instance_micro():
    a = p.aggregate_endpoint_fixed_slots([None]*32)
    assert a['primary']['person_recall'] == a['primary']['object_recall'] == 0
    assert a['slots'] == a['missing_slots'] == 32 and a['acquired_slots'] == 0
    assert a['persons']['matched_proposal_fraction_at_half'] is None


@pytest.mark.parametrize('bad', [[], {}, [False], [np.array([1])]])
def test_complete_typed_slot_ledger_required(bad):
    with pytest.raises(ValueError): p.aggregate_endpoint_fixed_slots(bad)

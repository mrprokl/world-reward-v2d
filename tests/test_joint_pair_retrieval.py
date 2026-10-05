"""Manufactured controls only; no images, model outputs or quality claims."""
import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from world_reward.joint_pair_retrieval import evaluate_joint_pair_retrieval
from world_reward.openimages_holds_reference import parse_holds_reference

MAN, WOMAN, PERSON = "/m/04yx4", "/m/03bt1vf", "/m/01g317"
CUP, BOTTLE = "/m/02p5f1q", "/m/01xs3r"
P = ((0., 0., .4, 1.), (.6, 0., 1., 1.))
O = ((.1, .3, .2, .4), (.7, .3, .8, .4))


def bbox(cls, box, group="0"):
    return dict(LabelName=cls, IsGroupOf=group,
                **dict(zip(("XMin", "YMin", "XMax", "YMax"), map(str, box))))


def holds(person=0, obj=0, *, cls=MAN, object_class=CUP, person_box=None, object_box=None):
    row = dict(RelationshipLabel="holds", LabelName1=cls, LabelName2=object_class)
    for suffix, box in (("1", P[person] if person_box is None else person_box),
                        ("2", O[obj] if object_box is None else object_box)):
        row.update(dict(zip((k + suffix for k in ("XMin", "YMin", "XMax", "YMax")), map(str, box))))
    return row


def reference(extra_boxes=(), rows=None):
    return parse_holds_reference([bbox(MAN, P[0]), bbox(WOMAN, P[1]),
        bbox(CUP, O[0]), bbox(BOTTLE, O[1]), *extra_boxes], [holds()] if rows is None else rows)


def evaluate(scores=((-1., -3.), (-4., -2.)), *, persons=P, objects=O, ref=None):
    return evaluate_joint_pair_retrieval(np.asarray(persons, np.float64).reshape(-1, 4),
        np.asarray(objects, np.float64).reshape(-1, 4), np.asarray(scores, np.float64), reference() if ref is None else ref)


def test_joint_endpoints_and_raw_negative_scores_not_thresholded():
    r = evaluate()
    assert r.image_positive_retrieval == 1 and r.top_score == -1
    assert r.top_unique_pair_count == r.top_annotated_positive_pair_count == 1
    assert r.unique_pair_count == 4 and r.positive_pair_proposal_recall == 1
    np.testing.assert_array_equal(r.person_reference_indices, [0, 1])
    np.testing.assert_array_equal(r.object_reference_indices, [0, 1])
    assert r.resolved_pair_person_proposal_recall == r.resolved_pair_object_proposal_recall == 1


@pytest.mark.parametrize("scores", [((-2., -3.), (0., -4.)), ((-2., 0.), (-3., -4.))])
def test_correct_object_wrong_person_or_correct_person_wrong_object_is_not_retrieved(scores):
    r = evaluate(scores)
    assert r.image_positive_retrieval == 0 and r.positive_pair_proposal_recall == 1
    assert not hasattr(r, "false_positive_count")


def test_exact_tie_fraction_includes_unknown_and_unannotated_competitors():
    assert evaluate(np.zeros((2, 2))).image_positive_retrieval == .25
    r = evaluate(((1., 1.), (0., 0.)), objects=(O[0], (.4, .4, .5, .5)))
    assert r.image_positive_retrieval == .5 and r.top_unique_pair_count == 2
    np.testing.assert_array_equal(r.object_reference_indices, [0, -1])
    # No tolerance or rounded margin: the next representable number really wins.
    r = evaluate(((1., np.nextafter(1., 2.)), (0., 0.)))
    assert r.image_positive_retrieval == 0


def test_all_annotated_positive_pairs_remain_and_any_top_positive_can_hit():
    ref = reference(rows=[holds(), holds(1, 1, cls=WOMAN, object_class=BOTTLE)])
    r = evaluate(np.zeros((2, 2)), ref=ref)
    assert r.image_positive_retrieval == .5 and r.resolved_positive_pair_count == 2
    assert r.reference_positive_pair_slots == r.retrieved_positive_pair_count == 2
    assert r.resolved_pair_person_count == r.resolved_pair_object_count == 2


@pytest.mark.parametrize("persons,objects", [((), O), (P, ()), ((), ())])
def test_missing_banks_keep_positive_denominator_and_zero_retrieval(persons, objects):
    r = evaluate(np.empty((len(persons), len(objects))), persons=persons, objects=objects)
    assert r.image_positive_retrieval == r.positive_pair_proposal_recall == 0
    assert r.top_score is None and r.top_unique_pair_count == r.unique_pair_count == 0
    assert r.reference_positive_pair_slots == 1


@pytest.mark.parametrize("endpoint,cls", [("person", PERSON), ("object", BOTTLE)])
def test_reference_hierarchy_different_class_same_box_remains_ambiguous(endpoint, cls):
    box = P[0] if endpoint == "person" else O[0]
    ref = reference([bbox(cls, box)])
    assert len(ref.holds) == 1  # Original parser binds same-class only.
    r = evaluate(ref=ref)
    assert r.image_positive_retrieval == r.positive_pair_proposal_recall == 0
    mapping = r.person_reference_indices if endpoint == "person" else r.object_reference_indices
    assert mapping[0] == -1  # Evaluator is class-agnostic, not a hierarchy aliaser.


@pytest.mark.parametrize("group", ["0", "1", "-1"])
@pytest.mark.parametrize("endpoint", ["person", "object"])
def test_exact_match_does_not_bypass_second_near_box_or_group(endpoint, group):
    box = (.01, 0., .4, 1.) if endpoint == "person" else (.101, .3, .2, .4)
    cls = PERSON if endpoint == "person" else BOTTLE
    r = evaluate(ref=reference([bbox(cls, box, group)]))
    assert r.image_positive_retrieval == r.positive_pair_proposal_recall == 0


@pytest.mark.parametrize("group", ["1", "-1"])
@pytest.mark.parametrize("endpoint", ["person", "object"])
def test_unique_group_or_unknown_group_censored_and_unbound_retained(endpoint, group):
    boxes = [bbox(MAN, P[0]), bbox(WOMAN, P[1]), bbox(CUP, O[0]), bbox(BOTTLE, O[1])]
    boxes[0 if endpoint == "person" else 2]["IsGroupOf"] = group
    ref = parse_holds_reference(boxes, [holds()])
    r = evaluate(ref=ref)
    assert r.image_positive_retrieval == r.positive_pair_proposal_recall == 0
    assert r.unbound_positive_relation_count == r.reference_positive_pair_slots == 1
    assert r.resolved_pair_person_proposal_recall is r.resolved_pair_object_proposal_recall is None
    assert r.unscorable_positive_person_slots == 1


def test_partially_unbound_person_preserves_resolved_hit_but_not_pair_recall_denominator():
    ref = reference(rows=[holds(), holds(object_class="/m/missing", object_box=(.4, .4, .5, .5))])
    r = evaluate(ref=ref)
    assert r.image_positive_retrieval == 1  # A known positive is still a hit.
    assert r.resolved_positive_pair_count == r.unbound_positive_relation_count == 1
    assert r.reference_positive_pair_slots == 2 and r.positive_pair_proposal_recall == .5
    assert r.total_positive_person_slots == r.unscorable_positive_person_slots == 1
    assert r.unbound_object_endpoint_count == 1 and r.unbound_person_endpoint_count == 0


def test_all_unbound_and_empty_inventory_not_fake_hits_or_fake_endpoint_recalls():
    r = evaluate(ref=parse_holds_reference([], [holds()]))
    assert r.image_positive_retrieval == r.positive_pair_proposal_recall == 0
    assert r.resolved_positive_pair_count == 0 and r.unbound_positive_relation_count == 1
    assert r.unbound_person_endpoint_count == r.unbound_object_endpoint_count == 1
    assert r.resolved_pair_person_proposal_recall is r.resolved_pair_object_proposal_recall is None


def test_geometry_not_class_prediction_and_iou_half_inclusive():
    # Exact predicted classes are not inputs: independently predicted endpoint boxes are.
    # Dyadic geometry gives an exactly represented IoU=.5, no tolerance needed.
    box = (0., 0., .25, 1.)
    ref = parse_holds_reference([bbox(MAN, box), bbox(CUP, O[0])], [holds(person_box=box)])
    r = evaluate(persons=((0., 0., .5, 1.), P[1]), ref=ref)
    assert r.person_reference_indices[0] == 0 and r.image_positive_retrieval == 1
    r = evaluate(persons=((0., 0., np.nextafter(.5, 1.), 1.), P[1]), ref=ref)
    assert r.person_reference_indices[0] == -1 and r.image_positive_retrieval == 0


def test_zero_area_and_off_grid_competitors_preserved_without_clipping():
    r = evaluate(((0., 1.), (0., 0.)), objects=(O[0], (-1., -1., 0., 0.)))
    assert r.raw_object_proposal_count == 2 and r.image_positive_retrieval == 0
    assert r.object_reference_indices[1] == -1
    r = evaluate(((0., 1.), (0., 0.)), objects=(O[0], (.7, .3, .7, .4)))
    assert r.top_score == 1 and r.image_positive_retrieval == 0


def test_exact_automatic_aliases_count_once_in_ties_not_reference_collapse():
    r = evaluate(np.zeros((3, 3)), persons=(P[0], P[0], P[1]), objects=(O[0], O[0], O[1]))
    assert r.raw_person_proposal_count == r.raw_object_proposal_count == 3
    assert r.unique_person_box_count == r.unique_object_box_count == 2
    assert r.unique_pair_count == r.top_unique_pair_count == 4
    assert r.top_annotated_positive_pair_count == 1 and r.image_positive_retrieval == .25
    assert r.retrieved_positive_pair_count == 1


@pytest.mark.parametrize("axis", ["person", "object"])
def test_unequal_alias_scores_rejected_not_mean_max_or_lucky_duplicate(axis):
    persons = (P[0], P[0], P[1]) if axis == "person" else P
    objects = (O[0], O[0], O[1]) if axis == "object" else O
    scores = np.zeros((len(persons), len(objects)))
    scores[1, 0] = 1. if axis == "person" else 0.
    scores[0, 1] = 1. if axis == "object" else 0.
    with pytest.raises(ValueError, match="aliases require equal"):
        evaluate(scores, persons=persons, objects=objects)


def test_unscorable_no_positive_scope_raises_not_zero_nonpositive_record():
    with pytest.raises(ValueError, match="positive scope"):
        evaluate(ref=parse_holds_reference([], []))
    with pytest.raises(ValueError, match="parse_holds_reference"):
        evaluate(ref=object())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["person", "object", "scores"])
def test_any_nonfinite_value_even_losing_competitor_rejected(target, value):
    p, o, s = np.array(P), np.array(O), np.zeros((2, 2))
    {"person": p, "object": o, "scores": s}[target].flat[-1] = value
    with pytest.raises(ValueError): evaluate_joint_pair_retrieval(p, o, s, reference())


@pytest.mark.parametrize("target", ["person", "object", "scores"])
def test_plain_float_shape_dtype_required_no_masked_arrays_or_lists(target):
    values = {"person": np.array(P), "object": np.array(O), "scores": np.zeros((2, 2))}
    for invalid in (values[target].tolist(), np.ma.array(values[target]), values[target].astype(np.int64), np.zeros((1, 1))):
        current = dict(values); current[target] = invalid
        with pytest.raises(ValueError): evaluate_joint_pair_retrieval(current["person"], current["object"], current["scores"], reference())


def test_inversion_and_represented_area_overflow_rejected():
    for box in ((.4, 0., .3, 1.), (-1e308, 0., 1e308, 1.)):
        with pytest.raises(ValueError): evaluate(persons=(box, P[1]))


def test_inputs_not_mutated_outputs_immutable_and_native_fp32_accepted():
    p, o = np.array(P, np.float32), np.array(O, np.float32)
    s = np.array(((-1., -3.), (-4., -2.)), np.float32)
    ref = reference(); before = copy.deepcopy((p, o, s))
    r = evaluate_joint_pair_retrieval(p, o, s, ref)
    for a, b in zip((p, o, s), before): np.testing.assert_array_equal(a, b)
    assert r.image_positive_retrieval == 1
    for a in (r.person_reference_indices, r.object_reference_indices):
        assert not a.flags.writeable
        with pytest.raises(ValueError): a.setflags(write=True)
    p[:] = o[:] = s[:] = 0
    np.testing.assert_array_equal(r.person_reference_indices, [0, 1])
    with pytest.raises(FrozenInstanceError): r.image_positive_retrieval = 0


def test_order_permutation_and_alias_repetition_do_not_change_scalar_metric():
    r = evaluate()
    permuted = evaluate(((-2., -4.), (-3., -1.)), persons=P[::-1], objects=O[::-1])
    assert r.image_positive_retrieval == permuted.image_positive_retrieval
    assert r.positive_pair_proposal_recall == permuted.positive_pair_proposal_recall
    repeated = evaluate(((-1., -1., -3.), (-1., -1., -3.), (-4., -4., -2.)),
                        persons=(P[0], P[0], P[1]), objects=(O[0], O[0], O[1]))
    assert r.image_positive_retrieval == repeated.image_positive_retrieval
    assert r.top_score == repeated.top_score and r.unique_pair_count == repeated.unique_pair_count

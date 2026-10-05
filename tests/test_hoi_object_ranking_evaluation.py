"""Tiny external-reference fixtures, no datasets/models or actual outcomes."""
from dataclasses import FrozenInstanceError, fields
import itertools
from fractions import Fraction

import numpy as np
import pytest

from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.hoi_object_ranking_evaluation import (
    ReferenceRegions, evaluate_hoi_object_ranking, exact_image_signflip, exact_image_sign_test,
)


def observation(boxes, roles, margins=None):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4); roles = np.asarray(roles, np.int64); n = len(roles)
    pair_args = []
    for a, b in ((0, 1), (1, 2)):
        left, right = np.flatnonzero(roles == a), np.flatnonzero(roles == b)
        pairs = np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64)
        logits = np.zeros((len(pairs), 2), np.float32)
        if a == 0 and margins is not None: logits[:, 1] = np.asarray(margins, np.float32)
        pair_args.extend((pairs, logits))
    scores = np.full(n, .75, np.float32)
    return HOIDetrObservations(0, (20, 20), np.zeros((1500, 3), np.float32),
        np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
        np.column_stack((boxes, scores)), np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64), roles, boxes, scores, scores, *pair_args)


def regions(boxes, classes=None, groups=None, prefix="r"):
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4); n = len(boxes)
    return ReferenceRegions(tuple(prefix+str(i) for i in range(n)),
        tuple("obj" for _ in range(n)) if classes is None else tuple(classes), boxes,
        np.zeros(n, bool) if groups is None else np.asarray(groups, bool))


def evaluate(o=None, persons=None, objects=None, holds=None, vocabulary=(("person", "obj"),), **kwargs):
    persons = regions([[0, 0, 10, 10]], ["person"], prefix="p") if persons is None else persons
    objects = regions([[5, 0, 7, 2], [1, 0, 3, 2]], prefix="o") if objects is None else objects
    o = observation([[1, 1, 1, 1], [5, 0, 7, 2], [1, 0, 3, 2]], [0, 1, 1], [4, 1]) if o is None else o
    holds = np.array([[0, 0]], np.int64) if holds is None else holds
    return evaluate_hoi_object_ranking(o, persons, objects, holds, vocabulary, **kwargs)


def test_shared_proxy_mapping_concordance_and_full_bank_retrieval():
    r = evaluate()
    np.testing.assert_array_equal(r.hand_person_indices, [0, -1, -1])
    np.testing.assert_array_equal(r.object_reference_indices, [-1, 0, 1])
    assert (r.image_concordance_a, r.image_concordance_b) == (0., 1.)
    assert (r.image_retrieval_a, r.image_retrieval_b) == (0., 1.)
    assert r.positive_person_count == r.informative_person_count == 1
    assert r.comparison_count.tolist() == [1]


def test_aliases_aggregate_best_once_not_extra_retrieval_votes():
    # Two positive aliases vs one tied negative: should be 1/2, NOT 2/3.
    objects = regions([[5, 0, 7, 2], [5, 4, 7, 6]])
    o = observation([[0, 3, 0, 3], [5, 0, 7, 2], [5, 0, 7, 2], [5, 4, 7, 6]],
                    [0, 1, 1, 1], [3, 3, 3])
    r = evaluate(o=o, objects=objects)
    assert r.known_candidate_count.tolist() == [2] and r.comparison_count.tolist() == [1]
    assert r.image_retrieval_b == .5 and r.image_concordance_b == .5
    assert r.image_retrieval_a == .5 and r.image_concordance_a == .5
    assert len(r.score_banks[0].object_slots) == 3


def test_unknown_winner_retained_as_miss_and_not_proven_negative():
    o = observation([[1, 1, 1, 1], [5, 0, 7, 2], [1, 0, 3, 2], [14, 14, 16, 16]],
                    [0, 1, 1, 1], [4, 1, 10])
    r = evaluate(o=o)
    assert r.image_concordance_b == 1 and r.image_retrieval_b == 0
    assert r.negative_count.tolist() == [1] and r.unknown_candidate_count.tolist() == [1]


def test_off_vocabulary_unannotated_objects_are_unknown_not_negative():
    objs = regions([[5, 0, 7, 2], [1, 0, 3, 2]], ["obj", "unverified"])
    r = evaluate(objects=objs)
    assert r.negative_count.tolist() == [0] and r.comparison_count.tolist() == [0]
    assert r.image_concordance_a is None and r.image_concordance_b is None
    np.testing.assert_array_equal(r.person_object_reference_indices, [[0, -1]])
    assert r.image_retrieval_a == 0 and r.image_retrieval_b == 1


@pytest.mark.parametrize("kind", ["no_hand", "no_object", "both"])
def test_missing_support_retains_every_positive_person_retrieval_denominator(kind):
    o = {"no_hand": observation([[5, 0, 7, 2]], [1]),
         "no_object": observation([[1, 1, 1, 1]], [0]),
         "both": observation([], [])}[kind]
    r = evaluate(o=o)
    assert r.positive_person_count == 1 and r.positive_count.tolist() == [1]
    assert r.image_retrieval_a == r.image_retrieval_b == 0
    assert r.image_concordance_a is None and r.image_concordance_b is None


def test_all_positive_objects_and_all_people_not_selected_target():
    persons = regions([[0, 0, 10, 10], [10, 0, 20, 10]], ["person", "person"], prefix="p")
    objs = regions([[5, 0, 7, 2], [1, 0, 3, 2], [12, 0, 14, 2]])
    o = observation([[1, 1, 1, 1], [5, 0, 7, 2], [1, 0, 3, 2], [12, 0, 14, 2]],
                    [0, 1, 1, 1], [4, 4, 1])
    r = evaluate(o=o, persons=persons, objects=objs, holds=np.array([[0, 0], [0, 1], [1, 2]], np.int64))
    assert r.positive_person_count == 2 and r.positive_count.tolist() == [2, 1]
    assert r.comparison_count.tolist() == [2, 0]
    assert r.image_retrieval_b == .5  # supported first person hit; second person miss
    assert r.image_concordance_b == 1 and r.informative_person_count == 1


def test_person_conditional_eligible_anchors_not_global_max():
    persons = regions([[0, 0, 10, 10], [10, 0, 20, 10]], ["person", "person"])
    objs = regions([[5, 0, 7, 2], [12, 0, 14, 2]])
    o = observation([[1, 1, 1, 1], [11, 1, 11, 1], [5, 0, 7, 2], [12, 0, 14, 2]],
                    [0, 0, 1, 1], [5, 1, 0, 8])
    r = evaluate(o=o, persons=persons, objects=objs, holds=np.array([[0, 0], [1, 1]], np.int64))
    assert r.image_concordance_b == r.image_retrieval_b == 1
    np.testing.assert_array_equal(r.score_banks[0].eligible_hand_slots, [0])
    np.testing.assert_array_equal(r.score_banks[1].eligible_hand_slots, [1])
    np.testing.assert_array_equal(r.score_banks[0].scores_b, [5, 1])
    np.testing.assert_array_equal(r.score_banks[1].scores_b, [0, 8])


def test_known_and_unknown_exact_top_ties_fractional_without_dropped_slots():
    objs = regions([[5, 0, 7, 2], [5, 4, 7, 6]])
    o = observation([[0, 3, 0, 3], [5, 0, 7, 2], [5, 4, 7, 6], [14, 14, 16, 16]],
                    [0, 1, 1, 1], [3, 3, 3])
    r = evaluate(o=o, objects=objs)
    assert r.image_retrieval_b == 1/3
    assert r.image_concordance_b == .5 and r.unknown_candidate_count.tolist() == [1]


def test_reference_and_native_permutations_preserve_per_instance_metrics():
    persons = regions([[0, 0, 10, 10], [10, 0, 20, 10]], ["person", "person"])
    objs = regions([[5, 0, 7, 2], [12, 0, 14, 2]])
    boxes = np.array([[1, 1, 1, 1], [11, 1, 11, 1], [5, 0, 7, 2], [12, 0, 14, 2]])
    o = observation(boxes, [0, 0, 1, 1], [5, 1, 0, 8])
    r = evaluate(o=o, persons=persons, objects=objs, holds=np.array([[0, 0], [1, 1]], np.int64))
    persons = ReferenceRegions(persons.instance_ids[::-1], persons.class_ids[::-1], persons.boxes_xyxy[::-1], persons.group_of[::-1])
    objs = ReferenceRegions(objs.instance_ids[::-1], objs.class_ids[::-1], objs.boxes_xyxy[::-1], objs.group_of[::-1])
    o = observation(boxes[::-1], [1, 1, 0, 0], [8, 0, 1, 5])
    s = evaluate(o=o, persons=persons, objects=objs, holds=np.array([[1, 1], [0, 0]], np.int64))
    assert r.image_concordance_a == s.image_concordance_a and r.image_concordance_b == s.image_concordance_b
    np.testing.assert_array_equal(r.retrieval_b, s.retrieval_b[::-1])


@pytest.mark.parametrize("group", [False, True])
def test_multiple_or_group_person_containment_censors_even_unique_nongroup(group):
    people = regions([[0, 0, 10, 10], [0, 0, 2, 2]], ["person", "person"], [False, group])
    r = evaluate(persons=people)
    assert r.hand_person_indices[0] == -1 and r.eligible_hand_count.tolist() == [0, 0]
    assert r.image_retrieval_b == 0


@pytest.mark.parametrize("group", [False, True])
def test_multiple_or_group_object_overlap_censors_unique_nongroup(group):
    objs = regions([[5, 0, 7, 2], [5, 0, 7, 2], [1, 0, 3, 2]], groups=[False, group, False])
    r = evaluate(objects=objs)
    assert r.object_reference_indices[1] == -1 and r.matched_positive_count.tolist() == [0]
    assert r.image_retrieval_b == 0 and r.image_concordance_b is None


def test_group_positive_person_and_object_unscorable_not_dropped():
    r = evaluate(persons=regions([[0, 0, 10, 10]], ["person"], [True]))
    assert r.reference_unscorable.tolist() == [True] and r.positive_person_count == 1
    assert r.image_retrieval_a == r.image_retrieval_b == 0 and not r.reliable_negative_pairs.any()
    r = evaluate(objects=regions([[5, 0, 7, 2], [1, 0, 3, 2]], groups=[True, False]))
    assert r.reference_unscorable.tolist() == [True] and r.image_retrieval_b == 0


def test_iou_exact_half_is_eligible_and_normalized_reference_equivalent():
    o = observation([[1, 1, 1, 1], [5, 0, 9, 2], [1, 0, 3, 2]], [0, 1, 1], [4, 1])
    r = evaluate(o=o)
    assert r.object_reference_indices[1] == 0  # 4 / 8 == .5
    s = evaluate(o=o, persons=regions([[0, 0, .5, .5]], ["person"]),
                 objects=regions([[.25, 0, .35, .1], [.05, 0, .15, .1]]), reference_coordinates="normalized")
    np.testing.assert_array_equal(r.object_reference_indices, s.object_reference_indices)
    assert r.image_concordance_b == s.image_concordance_b


def test_closed_person_boundaries_and_outside_boxes_no_clipping():
    o = observation([[10, 10, 10, 10], [5, 0, 7, 2], [-5, 0, 3, 2]], [0, 1, 1], [3, 4])
    r = evaluate(o=o)
    assert r.hand_person_indices[0] == 0
    assert r.object_reference_indices[2] == -1  # outside area stays in denominator
    assert r.image_retrieval_b == 0


def test_person_class_specific_vocabulary_no_parent_alias_inference():
    persons = regions([[0, 0, 10, 10]], ["human_subclass"])
    r = evaluate(persons=persons)
    assert not r.reliable_negative_pairs.any() and r.unknown_candidate_count.tolist() == [2]
    assert r.reference_unscorable.tolist() == [True] and r.image_retrieval_b == 0


def test_no_positive_person_inconclusive_image_metrics_null_not_success():
    r = evaluate(holds=np.empty((0, 2), np.int64))
    assert r.positive_person_count == 0 and r.image_retrieval_a is None and r.image_concordance_b is None


def test_outputs_and_external_source_arrays_are_owned_immutable():
    o = observation([[1, 1, 1, 1], [5, 0, 7, 2], [1, 0, 3, 2]], [0, 1, 1], [4, 1])
    persons, objects = regions([[0, 0, 10, 10]], ["person"]), regions([[5, 0, 7, 2], [1, 0, 3, 2]])
    before = o.boxes_original_xyxy.tobytes(), persons.boxes_xyxy.tobytes(), objects.boxes_xyxy.tobytes()
    r = evaluate(o, persons, objects)
    for f in fields(r):
        a = getattr(r, f.name)
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable
            with pytest.raises(ValueError): a.flags.writeable = True
    with pytest.raises(FrozenInstanceError): r.positive_person_count = 0
    assert before == (o.boxes_original_xyxy.tobytes(), persons.boxes_xyxy.tobytes(), objects.boxes_xyxy.tobytes())


def test_empty_reference_banks_are_not_fabricated_positive_or_comparisons():
    r = evaluate(persons=regions([], []), objects=regions([], []), holds=np.empty((0, 2), np.int64))
    assert r.positive_pairs.shape == (0, 0) and r.person_object_reference_indices.shape == (0, 2)
    assert r.positive_person_count == r.informative_person_count == 0
    assert r.image_retrieval_a is None and r.image_concordance_b is None


@pytest.mark.parametrize("kind", ["nan", "inverted", "zero", "masked", "duplicate_id"])
def test_invalid_boxes_and_instance_identity_contract(kind):
    boxes = np.array([[0, 0, 1, 1]], np.float64)
    ids, classes, group = ("x",), ("obj",), np.zeros(1, bool)
    if kind == "nan": boxes[0, 0] = np.nan
    elif kind == "inverted": boxes[0, 2] = -1
    elif kind == "zero": boxes[0, 2] = 0
    elif kind == "masked": boxes = np.ma.array(boxes, mask=False)
    else: boxes = np.repeat(boxes, 2, axis=0); ids, classes, group = ("x", "x"), ("obj", "obj"), np.zeros(2, bool)
    with pytest.raises(ValueError): ReferenceRegions(ids, classes, boxes, group)


@pytest.mark.parametrize("bad", ["links_dtype", "links_duplicate", "links_range", "vocab_duplicate", "vocab_shape", "normalized", "region_dtype", "group_dtype"])
def test_invalid_reference_contracts(bad):
    args = {}
    if bad == "links_dtype": args["holds"] = np.array([[0, 0]], np.int32)
    elif bad == "links_duplicate": args["holds"] = np.array([[0, 0], [0, 0]], np.int64)
    elif bad == "links_range": args["holds"] = np.array([[1, 0]], np.int64)
    elif bad == "vocab_duplicate": args["vocabulary"] = (("person", "obj"),) * 2
    elif bad == "vocab_shape": args["vocabulary"] = (("person",),)
    elif bad == "normalized": args["reference_coordinates"] = "normalized"
    elif bad == "region_dtype":
        with pytest.raises(ValueError): ReferenceRegions(("x",), ("obj",), np.ones((1, 4), np.float32), np.zeros(1, bool))
        return
    else:
        with pytest.raises(ValueError): ReferenceRegions(("x",), ("obj",), np.array([[0, 0, 1, 1]], np.float64), np.zeros(1, np.int64))
        return
    with pytest.raises(ValueError): evaluate(**args)


def test_exact_signflip_matches_independent_rational_enumeration_and_keeps_zeros():
    d = np.array([.5, -.25, 0., .125, .25, .5], np.float64)
    target = sum(map(Fraction, d)); outcomes = []
    for signs in itertools.product((-1, 1), repeat=len(d)):
        outcomes.append(sum(Fraction(x) * sign for x, sign in zip(d, signs)))
    r = exact_image_signflip(d)
    assert r.p_one_sided == sum(x >= target for x in outcomes) / len(outcomes)
    assert r.zero_count == 1 and r.informative_count == 6 and r.mean_delta == d.mean()
    assert exact_image_signflip(np.zeros(6, np.float64)).p_one_sided == 1
    assert exact_image_signflip(np.ones(5, np.float64)).p_one_sided is None
    with pytest.raises(ValueError): exact_image_signflip(np.ones(21, np.float64))


def test_distinct_exact_sign_test_big_cohort_zero_ties_and_eligibility():
    r = exact_image_sign_test(np.ones(128, np.float64))
    assert r.p_one_sided == 2**-128 and r.informative_count == r.image_count == 128
    r = exact_image_sign_test(np.array([1, 1, 1, 1, 1, -1, 0, 0], np.float64))
    assert r.p_one_sided == 7 / 64 and r.zero_count == 2 and r.mean_delta == .5
    assert exact_image_sign_test(np.array([1, 1, 1, 1, 1, 0], np.float64)).p_one_sided is None
    assert r.test != exact_image_signflip(np.ones(6, np.float64)).test


def test_statistical_mean_no_overflow_or_subnormal_loss():
    for test in (exact_image_signflip, exact_image_sign_test):
        assert test(np.full(6, np.finfo(np.float64).max)).mean_delta == np.finfo(np.float64).max
        tiny = np.nextafter(0., 1.)
        assert test(np.full(6, tiny)).mean_delta == tiny
        assert test(np.array([1., -1., 1., -1., 0., 0.], np.float64)).mean_delta == 0


@pytest.mark.parametrize("values", [np.ones(6, np.float32), np.array([np.nan], np.float64),
                                  np.array([np.inf], np.float64), np.empty(0, np.float64),
                                  np.ones((2, 3), np.float64), np.ma.array(np.ones(6), mask=False)])
def test_statistical_tests_reject_incomplete_nonfinite_foreign_vectors(values):
    for test in (exact_image_signflip, exact_image_sign_test):
        with pytest.raises(ValueError): test(values)

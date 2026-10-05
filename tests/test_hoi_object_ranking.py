"""Manufactured paired scoring checks, not native or empirical qualification."""
from dataclasses import FrozenInstanceError, fields, replace
import itertools

import numpy as np
import pytest

from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.hoi_object_ranking import score_hoi_objects


def observation(boxes, roles, logits=None, *, queries=None, grid=(3, 4)):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    roles = np.asarray(roles, np.int64); n = len(roles)
    pairs = []
    for a, b in ((0, 1), (1, 2)):
        left, right = np.flatnonzero(roles == a), np.flatnonzero(roles == b)
        p = np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64)
        values = np.zeros((len(p), 2), np.float32)
        if a == 0 and logits is not None: values = np.asarray(logits, np.float32).reshape(len(p), 2)
        pairs.extend((p, values))
    scores = np.full(n, .75, np.float32)
    return HOIDetrObservations(8, grid, np.zeros((1500, 3), np.float32),
        np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
        np.column_stack((boxes, scores)), np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64), np.asarray(range(n) if queries is None else queries, np.int64),
        roles, boxes, scores, scores, *pairs)


def scene():
    # Pair order (hand0,obj1),(hand0,obj3),(hand2,obj1),(hand2,obj3).
    return observation([[0, 0, 2, 2], [2, 0, 4, 2], [8, 0, 10, 2], [6, 0, 8, 2]],
                       [0, 1, 0, 1], [[1, 2], [2, -1], [-2, 3], [2, 3]])


def test_same_bank_exact_scores_raw_logit1_minus0_and_native_slots():
    o = scene(); r = score_hoi_objects(o)
    np.testing.assert_array_equal(r.object_slots, [1, 3])
    np.testing.assert_array_equal(r.eligible_hand_slots, [0, 2])
    np.testing.assert_array_equal(r.scores_a, [[-.2, -.4], [-.2, -.4]])
    np.testing.assert_array_equal(r.scores_b, [5, 1])
    np.testing.assert_array_equal(r.object_nms_positions, o.retained_nms_positions[[1, 3]])
    assert r.supported.all() and r.original_frame_index == 8


def test_explicit_anchor_subset_shared_between_both_arms_and_order_invariant():
    o = scene(); r = score_hoi_objects(o, eligible_hand_slots=np.array([2], np.int64))
    np.testing.assert_array_equal(r.scores_a, [[-1., -1.2], [-.2, -.4]])
    np.testing.assert_array_equal(r.scores_b, [5, 1])
    s = score_hoi_objects(o, eligible_hand_slots=np.array([2, 0], np.int64))
    all_hands = score_hoi_objects(o)
    for f in fields(s):
        a, b = getattr(s, f.name), getattr(all_hands, f.name)
        if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
        else: assert a == b


@pytest.mark.parametrize("mode", ["no_hands", "none_eligible", "no_objects", "empty"])
def test_absence_has_no_fabricated_rank(mode):
    if mode == "none_eligible": o, slots = scene(), np.empty(0, np.int64)
    else:
        roles = {"no_hands": [1, 2, 1], "no_objects": [0, 2], "empty": []}[mode]
        o, slots = observation([[0, 0, 0, 0]] * len(roles), roles), None
    r = score_hoi_objects(o, eligible_hand_slots=slots)
    assert len(r.object_slots) == int(np.count_nonzero(o.class_ids == 1))
    assert not r.supported.any() and np.isnan(r.scores_a).all() and np.isnan(r.scores_b).all()


def test_aliases_and_ties_retained_without_winner():
    o = observation([[0, 0, 0, 0], [-1, 0, -1, 0], [1, 0, 1, 0]],
                    [0, 1, 1], [[-3, 2], [-3, 2]], queries=[9, 9, 9])
    r = score_hoi_objects(o)
    np.testing.assert_array_equal(r.object_query_ids, [9, 9])
    np.testing.assert_array_equal(r.object_slots, [1, 2])
    np.testing.assert_array_equal(r.scores_a, [[-.2, -.2], [-.2, -.2]])
    np.testing.assert_array_equal(r.scores_b, [5, 5])
    assert r.supported.all() and not hasattr(r, "accepted_object")


def test_outside_original_image_and_extreme_f32_are_not_clipped_or_overflowed():
    f = np.finfo(np.float32).max
    o = observation([[-f, -f, -f, -f], [f, f, f, f]], [0, 1], [[-f, f]])
    r = score_hoi_objects(o)
    assert np.isfinite(r.scores_a).all() and np.isfinite(r.scores_b).all()
    np.testing.assert_array_equal(r.scores_a[0], [-np.hypot(2*float(f), 2*float(f))/5] * 2)
    assert r.scores_b[0] == 2*float(f)


def test_independent_scalar_reference_and_proposal_permutation():
    o = scene(); rng = np.random.default_rng(8401)
    raw = rng.integers(-100, 100, (len(o.query_ids), 4)).astype(np.float32)/8
    boxes = np.column_stack((np.minimum(raw[:, 0], raw[:, 2]), np.minimum(raw[:, 1], raw[:, 3]),
                             np.maximum(raw[:, 0], raw[:, 2]), np.maximum(raw[:, 1], raw[:, 3])))
    o = replace(o, boxes_original_xyxy=boxes)
    r = score_hoi_objects(o)
    for j, obj in enumerate(r.object_slots):
        c = [(float(o.boxes_original_xyxy[obj, k])+float(o.boxes_original_xyxy[obj, k+2]))/2 for k in (0, 1)]
        distances = []
        gains = []
        for hand in r.eligible_hand_slots:
            h = [(float(o.boxes_original_xyxy[hand, k])+float(o.boxes_original_xyxy[hand, k+2]))/2 for k in (0, 1)]
            x0, y0, x1, y1 = map(float, o.boxes_original_xyxy[obj])
            dx, dy = max(x0-h[0], h[0]-x1, 0), max(y0-h[1], h[1]-y1, 0)
            distances.append((np.hypot(dx, dy)/5, np.hypot(h[0]-c[0], h[1]-c[1])/5))
            row = np.flatnonzero(np.all(o.hand_object_pairs == [hand, obj], axis=1))[0]
            gains.append(float(o.hand_object_logits[row, 1])-float(o.hand_object_logits[row, 0]))
        np.testing.assert_array_equal(r.scores_a[j], -np.asarray(min(distances)))
        assert r.scores_b[j] == max(gains)
    p = np.array([3, 2, 1, 0]); roles = o.class_ids[p]
    hands, objects = np.flatnonzero(roles == 0), np.flatnonzero(roles == 1)
    logits = [o.hand_object_logits[np.flatnonzero(np.all(o.hand_object_pairs == [p[h], p[z]], axis=1))[0]]
              for h, z in itertools.product(hands, objects)]
    s = score_hoi_objects(observation(boxes[p], roles, logits, queries=o.query_ids[p]))
    np.testing.assert_array_equal(r.object_query_ids, s.object_query_ids[::-1])
    np.testing.assert_array_equal(r.scores_a, s.scores_a[::-1])
    np.testing.assert_array_equal(r.scores_b, s.scores_b[::-1])


@pytest.mark.parametrize("slots", [[1], [4], [-1], [0, 0], [3], [0., 2.]])
def test_invalid_eligible_slots(slots):
    with pytest.raises(ValueError): score_hoi_objects(scene(), eligible_hand_slots=np.asarray(slots))


def test_spatial_lexicographic_priority_not_independent_minima_or_weights():
    # Elongated box: inside hand farther from centre beats outside closer hand.
    o = observation([[-9, 0, -9, 0], [0, 2, 0, 2], [-10, -1, 10, 1]],
                    [0, 0, 1], [[0, 1], [0, 2]])
    r = score_hoi_objects(o)
    np.testing.assert_array_equal(r.scores_a, [[-0., -1.8]])
    assert r.scores_b[0] == 2  # B independently aggregates over the SAME two hands.
    # Equal primary distance: secondary centre distance is the only tiebreaker.
    o = observation([[-9, 0, -9, 0], [0, 0, 0, 0], [-10, -1, 10, 1]],
                    [0, 0, 1], [[0, 1], [0, 2]])
    np.testing.assert_array_equal(score_hoi_objects(o).scores_a, [[-0., -0.]])


@pytest.mark.parametrize("slots", [np.array([0, 2], np.int32), np.array([False, True]),
                                  np.ma.array([0, 2], mask=[False, False])])
def test_foreign_or_masked_slot_vectors_rejected(slots):
    with pytest.raises(ValueError): score_hoi_objects(scene(), eligible_hand_slots=slots)


def test_spatial_scores_keep_original_diagonal_and_translation_invariance():
    o = scene(); r = score_hoi_objects(o)
    # Shared image-coordinate translation changes no spatial relationship.
    s = score_hoi_objects(replace(o, boxes_original_xyxy=o.boxes_original_xyxy + np.float32(64)))
    np.testing.assert_array_equal(r.scores_a, s.scores_a)
    np.testing.assert_array_equal(r.scores_b, s.scores_b)
    # Doubling geometry AND image dimensions cancels in normalized distances.
    s = score_hoi_objects(replace(o, image_size=(6, 8), boxes_original_xyxy=o.boxes_original_xyxy * np.float32(2)))
    np.testing.assert_array_equal(r.scores_a, s.scores_a)
    np.testing.assert_array_equal(r.scores_b, s.scores_b)


@pytest.mark.parametrize("bad", ["role", "query", "inverted", "nan", "inf_logit", "pairs", "dtype", "type"])
def test_malformed_native_bank_fails(bad):
    o = scene()
    if bad == "type": o = object()
    elif bad == "role": object.__setattr__(o, "class_ids", np.array([0, 1, 3, 1], np.int64))
    elif bad == "query": object.__setattr__(o, "query_ids", np.array([0, 1, 1500, 3], np.int64))
    elif bad in ("inverted", "nan", "dtype"):
        b = o.boxes_original_xyxy.copy().astype(np.float64 if bad == "dtype" else np.float32)
        if bad == "inverted": b[0] = [1, 0, 0, 1]
        elif bad == "nan": b[0, 0] = np.nan
        object.__setattr__(o, "boxes_original_xyxy", b)
    elif bad == "inf_logit": object.__setattr__(o, "hand_object_logits", np.full((4, 2), np.inf, np.float32))
    else: object.__setattr__(o, "hand_object_pairs", o.hand_object_pairs[::-1])
    with pytest.raises(ValueError): score_hoi_objects(o)


def test_immutable_outputs_no_alias_source_unchanged():
    o = scene(); before = {f.name: getattr(o, f.name).tobytes() for f in fields(o) if isinstance(getattr(o, f.name), np.ndarray)}
    r = score_hoi_objects(o)
    for f in fields(r):
        a = getattr(r, f.name)
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable
            with pytest.raises(ValueError): a.flags.writeable = True
            assert all(not np.shares_memory(a, getattr(o, z.name)) for z in fields(o) if isinstance(getattr(o, z.name), np.ndarray))
    with pytest.raises(FrozenInstanceError): r.original_frame_index = 0
    assert before == {f.name: getattr(o, f.name).tobytes() for f in fields(o) if isinstance(getattr(o, f.name), np.ndarray)}
    o.hand_object_logits.flags.writeable = True; o.hand_object_logits[:] = 0
    np.testing.assert_array_equal(r.scores_b, [5, 1])

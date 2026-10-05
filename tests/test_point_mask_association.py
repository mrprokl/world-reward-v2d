"""Procedural operator checks only; no models, real observations or empirical win."""
import itertools

import numpy as np
import pytest

from world_reward import point_mask_association as p


def anchor(t, ids=("left", "right")):
    masks = np.zeros((2, 4, 8), bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 1:3, 5:7] = True
    return p.MaskAnchor(t, ids, masks)


def crossing():
    anchors = (anchor(0), anchor(2))
    seeds = (p.HypothesisSeed("A", 0, "left"), p.HypothesisSeed("B", 0, "right"))
    initial = np.array([[1.5, 1.5], [2.5, 2.5], [5.5, 1.5], [6.5, 2.5]])
    tracks = np.array([[[1.5, 1.5], [3.5, 1.5], [5.5, 1.5]],
                       [[2.5, 2.5], [4.5, 2.5], [6.5, 2.5]],
                       [[5.5, 1.5], [3.5, 1.5], [1.5, 1.5]],
                       [[6.5, 2.5], [4.5, 2.5], [2.5, 2.5]]])
    return dict(frame_index=np.arange(3, dtype=np.int64), anchors=anchors, seeds=seeds,
                tracks=tracks, visibility=np.ones((4, 3), bool),
                query_ownership=("A", "A", "B", "B"),
                query_birth_frame_indices=np.zeros(4, np.int64), query_points_xy=initial)


def assign(rows, cols, scores, support=None, *, null=.5, tolerance=0.):
    scores = np.asarray(scores, dtype=np.float64).reshape(len(rows), len(cols))
    if support is None:
        support = np.ones(scores.shape, bool)
    return p.partial_mask_assignment(rows, cols, scores, support,
                                     unmatched_score=null, tie_tolerance=tolerance)


def test_same_bank_crossing_individual_correspondences_versus_iou():
    args = crossing()
    evidence = p.point_mask_evidence(**args)
    b = evidence.anchors[-1]
    reference = p.MaskAnchor(0, ("A", "B"), args["anchors"][0].masks)
    a = p.mask_iou_evidence(reference, args["anchors"][-1])
    left = assign(a.hypothesis_ids, a.candidate_ids, a.fraction_iou, a.supported)
    right = assign(b.hypothesis_ids, b.candidate_ids, b.fraction_inside, b.supported)
    assert left.unique_candidate_ids == ("left", "right")
    assert right.unique_candidate_ids == ("right", "left")
    assert left.unmatched_score == right.unmatched_score == .5
    np.testing.assert_array_equal(b.inside_count, [[0, 2], [2, 0]])
    np.testing.assert_array_equal(evidence.frame_index, [0, 1, 2])
    assert evidence.query_ownership == args["query_ownership"]
    assert left.unique and right.unique  # Algebra, not accepted real identities.
    seeded_a = p.seed_mask_iou_evidence(args["frame_index"], args["anchors"], args["seeds"])[-1]
    assert seeded_a.hypothesis_ids == b.hypothesis_ids and seeded_a.candidate_ids == b.candidate_ids
    np.testing.assert_array_equal(seeded_a.fraction_iou, a.fraction_iou)


def test_rotation_does_not_reduce_correspondences_to_zero_median_velocity():
    masks = np.zeros((1, 8, 8), bool)
    masks[0, 2:6, 2:6] = True
    anchors = tuple(p.MaskAnchor(t, ("object",), masks) for t in (0, 1))
    initial = np.array([[2.5, 2.5], [5.5, 2.5], [5.5, 5.5], [2.5, 5.5]])
    tracks = np.stack((initial, 8. - initial), axis=1)
    np.testing.assert_array_equal(np.median(tracks[:, 1] - tracks[:, 0], axis=0), [0., 0.])
    result = p.point_mask_evidence(np.arange(2, dtype=np.int64), anchors,
        (p.HypothesisSeed("rigid", 0, "object"),), tracks, np.ones((4, 2), bool),
        ("rigid",) * 4, np.zeros(4, np.int64), initial).anchors[-1]
    assert result.inside_count[0, 0] == result.native_visible_count[0] == 4
    assert result.fraction_inside[0, 0] == 1.


def test_native_visible_denominator_includes_off_image_and_on_image_misses():
    args = crossing()
    args["tracks"][:, 2] = [[1.5, 1.5], [8., 1.5], [4., 1.5], [np.nan, np.inf]]
    args["visibility"][3, 2] = False
    args["query_ownership"] = ("A",) * 4
    args["query_points_xy"][:] = [1.5, 1.5]
    result = p.point_mask_evidence(**args).anchors[-1]
    assert result.query_count.tolist() == [4, 0]
    assert result.native_visible_count.tolist() == [3, 0]
    assert result.on_image_count.tolist() == [2, 0]
    assert result.off_image_count.tolist() == [1, 0]
    assert result.hidden_count.tolist() == [1, 0]
    assert result.inside_count.tolist() == [[1, 0], [0, 0]]
    assert result.fraction_inside[0, 0] == 1 / 3
    assert result.supported[0].all() and not result.supported[1].any()
    assert np.isfinite(result.fraction_inside).all()


@pytest.mark.parametrize("mode", ["off_image", "occluded", "no_queries"])
def test_unknown_is_not_perfect_and_all_hypotheses_remain(mode):
    args = crossing()
    if mode == "off_image":
        args["tracks"][:, 2] = [1e30, -1e30]
    elif mode == "occluded":
        args["visibility"][:, 2] = False
        args["tracks"][:, 2] = np.nan
    else:
        args.update(tracks=np.empty((0, 3, 2)), visibility=np.empty((0, 3), bool),
                    query_ownership=(), query_birth_frame_indices=np.empty(0, np.int64),
                    query_points_xy=np.empty((0, 2)))
    evidence = p.point_mask_evidence(**args)
    result = evidence.anchors[-1]
    assert result.hypothesis_ids == ("A", "B") and result.candidate_ids == ("left", "right")
    assert not result.supported.any() and not result.fraction_inside.any()
    answer = assign(result.hypothesis_ids, result.candidate_ids, result.fraction_inside, result.supported)
    assert answer.row_status == ("UNIQUE_UNMATCHED",) * 2
    assert answer.unique_candidate_ids == (None, None)
    assert answer.optimal_candidate_unmatched.all()
    np.testing.assert_array_equal(evidence.frame_index, np.arange(3))


def test_query_seed_time_does_not_censor_backward_native_tracking():
    anchors = (anchor(0), anchor(4))
    tracks = np.array([[[1.5, 1.5], [2., 1.5], [3., 1.5], [4., 1.5], [99., 99.]]])
    result = p.point_mask_evidence(np.arange(5, dtype=np.int64), anchors,
        (p.HypothesisSeed("late", 4, "right"),), tracks, np.ones((1, 5), bool),
        ("late",), np.array([4], np.int64), np.array([[5.5, 1.5]]))
    assert result.anchors[0].fraction_inside.tolist() == [[1., 0.]]
    assert result.anchors[0].supported.all()
    assert result.anchors[-1].off_image_count.tolist() == [1]
    assert not result.anchors[-1].supported.any()
    assert result.query_birth_frame_indices.tolist() == [4]
    # Original query coordinates, not the drifting predicted point at birth,
    # establish seed-mask membership.
    np.testing.assert_array_equal(result.query_points_xy, [[5.5, 1.5]])


def test_mixed_seed_times_share_exact_a_b_hypothesis_and_candidate_bank():
    anchors = (anchor(0), anchor(4))
    seeds = (p.HypothesisSeed("late", 4, "right"), p.HypothesisSeed("early", 0, "left"))
    tracks = np.array([[[5.5, 1.5]] * 5, [[1.5, 1.5]] * 5])
    b = p.point_mask_evidence(np.arange(5, dtype=np.int64), anchors, seeds, tracks,
        np.ones((2, 5), bool), ("late", "early"), np.array([4, 0], np.int64),
        np.array([[5.5, 1.5], [1.5, 1.5]]))
    a = p.seed_mask_iou_evidence(b.frame_index, anchors, seeds)
    for aa, bb in zip(a, b.anchors):
        assert aa.hypothesis_ids == bb.hypothesis_ids == ("early", "late")
        assert aa.candidate_ids == bb.candidate_ids
        assert aa.reference_frame_indices.tolist() == [0, 4]
        np.testing.assert_array_equal(aa.fraction_iou, bb.fraction_inside)
        for value in vars(aa).values():
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable
                with pytest.raises(ValueError): value.setflags(write=True)


def test_overlapping_masks_and_original_duplicate_queries_remain_independent():
    masks = np.zeros((2, 3, 3), bool)
    masks[:, 1, 1] = True
    anchors = (p.MaskAnchor(0, ("a", "b"), masks),)
    result = p.point_mask_evidence(np.array([0], np.int64), anchors,
        (p.HypothesisSeed("A", 0, "a"), p.HypothesisSeed("B", 0, "b")),
        np.full((3, 1, 2), 1.5), np.ones((3, 1), bool), ("A", "A", "B"),
        np.zeros(3, np.int64), np.full((3, 2), 1.5)).anchors[0]
    assert result.inside_count.tolist() == [[2, 2], [1, 1]]
    assert result.mask_area.tolist() == [1, 1]
    assert result.fraction_inside.tolist() == [[1., 1.], [1., 1.]]
    answer = assign(result.hypothesis_ids, result.candidate_ids, result.fraction_inside, result.supported)
    assert not answer.unique and answer.optimal_routes.all()
    assert answer.unique_candidate_ids == (None, None)


def test_floor_xy_on_original_grid_without_offset_correction():
    masks = np.zeros((1, 2, 3), bool)
    masks[0, 0, 0] = True
    result = p.point_mask_evidence(np.array([0], np.int64),
        (p.MaskAnchor(0, ("pixel",), masks),), (p.HypothesisSeed("A", 0, "pixel"),),
        np.array([[[.999, .999]], [[1., .999]], [[-.001, 0.]]]), np.ones((3, 1), bool),
        ("A",) * 3, np.zeros(3, np.int64), np.full((3, 2), .25)).anchors[0]
    assert result.inside_count.tolist() == [[1]]
    assert result.on_image_count.tolist() == [2] and result.off_image_count.tolist() == [1]
    assert result.fraction_inside[0, 0] == 1 / 3


def test_empty_masks_iou_and_points_are_zero_unsupported_not_perfect():
    empty = p.MaskAnchor(0, ("empty",), np.zeros((1, 2, 2), bool))
    iou = p.mask_iou_evidence(empty, empty)
    assert iou.intersection.tolist() == iou.union.tolist() == [[0]]
    assert not iou.fraction_iou.any() and not iou.supported.any()
    points = p.point_mask_evidence(np.array([0], np.int64), (empty,),
        (p.HypothesisSeed("A", 0, "empty"),), np.empty((0, 1, 2)), np.empty((0, 1), bool),
        (), np.empty(0, np.int64), np.empty((0, 2))).anchors[0]
    assert points.candidate_ids == ("empty",) and not points.supported.any()


def test_missing_candidate_anchor_keeps_full_t_and_hypotheses():
    args = crossing()
    args["anchors"] = args["anchors"][:1] + (p.MaskAnchor(2, (), np.empty((0, 4, 8), bool)),)
    result = p.point_mask_evidence(**args).anchors[-1]
    assert result.fraction_inside.shape == result.supported.shape == (2, 0)
    assert result.native_visible_count.tolist() == [2, 2]
    answer = assign(result.hypothesis_ids, (), result.fraction_inside, result.supported)
    assert answer.row_status == ("UNIQUE_UNMATCHED",) * 2


def test_hungarian_is_global_not_independent_nearest_maximum():
    result = assign(("A", "B"), ("a", "b"), [[1., .75], [1., .25]], null=.1)
    assert result.unique_candidate_ids == ("b", "a")
    assert result.optimum_score == 1.75 and result.unique
    assert result.solver_calls == 1 + 4 + 2 + 2


def test_every_global_tied_route_is_exposed_without_arbitrary_identity():
    result = assign(("A", "B"), ("a", "b"), np.ones((2, 2)))
    assert result.optimal_routes.all() and not result.optimal_unmatched.any()
    assert not result.optimal_candidate_unmatched.any()
    assert result.row_status == ("AMBIGUOUS", "AMBIGUOUS")
    assert result.unique_candidate_ids == (None, None) and not result.unique
    assert not result.edge_regrets.any()


def test_null_tie_not_a_chosen_match_and_null_cost_not_doubled():
    tie = assign(("A",), ("a",), [[.5]])
    assert tie.optimal_routes[0, 0] and tie.optimal_unmatched[0]
    assert tie.optimal_candidate_unmatched[0] and not tie.unique
    assert tie.unique_candidate_ids == (None,)
    match = assign(("A",), ("a",), [[.75]])
    assert match.unique_candidate_ids == ("a",) and match.optimum_score == .75
    unmatched = assign(("A",), ("a",), [[.25]])
    assert unmatched.row_status == ("UNIQUE_UNMATCHED",) and unmatched.optimum_score == .5


def test_competing_hypotheses_can_each_have_match_and_unmatched_routes():
    result = assign(("A", "B"), ("a",), [[1.], [1.]])
    assert result.optimum_score == 1.5 and result.optimal_routes.all()
    assert result.optimal_unmatched.all() and not result.optimal_candidate_unmatched.any()
    assert result.row_status == ("AMBIGUOUS", "AMBIGUOUS")


def test_unsupported_high_score_never_selects_or_deletes_a_hypothesis():
    result = assign(("A", "B"), ("a", "b"), np.ones((2, 2)),
                    np.array([[False, False], [True, False]]))
    assert result.unique_candidate_ids == (None, "a")
    assert np.isinf(result.edge_regrets[0]).all()
    assert result.row_status == ("UNIQUE_UNMATCHED", "UNIQUE_MATCH")


def test_explicit_numeric_tolerance_reports_alternatives_not_learned_confidence():
    exact = assign(("A",), ("a",), [[.5000000001]])
    tolerant = assign(("A",), ("a",), [[.5000000001]], tolerance=1e-9)
    assert exact.unique and not tolerant.unique
    assert tolerant.optimal_unmatched[0]
    assert tolerant.unmatched_regrets[0] > 0


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_empty_assignment_banks_are_explicit(shape):
    n, m = shape
    result = assign(tuple(f"h{i}" for i in range(n)), tuple(f"c{j}" for j in range(m)), np.zeros(shape))
    assert result.unique and result.optimum_score == n * .5
    assert result.optimal_routes.shape == shape
    assert result.optimal_unmatched.all() and result.optimal_candidate_unmatched.all()


def test_detection_hypothesis_query_order_invariance():
    args = crossing()
    baseline = p.point_mask_evidence(**args)
    args["anchors"] = tuple(p.MaskAnchor(a.frame_index, a.mask_ids[::-1], a.masks[::-1]) for a in args["anchors"])
    args["seeds"] = args["seeds"][::-1]
    order = np.array([3, 1, 0, 2])
    for key in ("tracks", "visibility", "query_birth_frame_indices", "query_points_xy"):
        args[key] = args[key][order]
    args["query_ownership"] = tuple(args["query_ownership"][i] for i in order)
    result = p.point_mask_evidence(**args)
    assert result.seeds == baseline.seeds
    for a, b in zip(result.anchors, baseline.anchors):
        assert a.hypothesis_ids == b.hypothesis_ids and a.candidate_ids == b.candidate_ids
        for field in ("inside_count", "native_visible_count", "fraction_inside", "supported"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    scores = np.array([[.25, 1.], [1., .25]])
    a = assign(("A", "B"), ("left", "right"), scores)
    b = assign(("B", "A"), ("right", "left"), scores[::-1, ::-1])
    assert a.unique_candidate_ids == b.unique_candidate_ids
    np.testing.assert_array_equal(a.optimal_routes, b.optimal_routes)
    np.testing.assert_array_equal(a.edge_regrets, b.edge_regrets)


def test_owned_immutable_outputs_and_no_mutation_of_caller_arrays():
    args = crossing()
    original = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    evidence = p.point_mask_evidence(**args)
    for key, value in original.items():
        np.testing.assert_array_equal(args[key], value)
    arrays = [evidence.frame_index, evidence.query_points_xy, evidence.query_birth_frame_indices]
    arrays.extend(v for a in evidence.anchors for v in vars(a).values() if isinstance(v, np.ndarray))
    iou = p.mask_iou_evidence(args["anchors"][0], args["anchors"][-1])
    arrays.extend(v for v in vars(iou).values() if isinstance(v, np.ndarray))
    answer = assign(("A", "B"), ("a", "b"), [[1., 0.], [0., 1.]])
    arrays.extend(v for v in vars(answer).values() if isinstance(v, np.ndarray))
    arrays.extend(a.masks for a in args["anchors"])
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    args["frame_index"][:] = 9
    args["query_points_xy"][:] = 99
    assert evidence.frame_index.tolist() == [0, 1, 2]
    assert evidence.query_points_xy[0].tolist() == [1.5, 1.5]


def test_global_route_sets_match_exhaustive_tiny_partial_assignments():
    # Independent discrete oracle for ambiguity/null semantics, not production DP.
    rng = np.random.default_rng(17)
    for n, m in ((1, 2), (2, 1), (2, 2), (3, 2)):
        for _ in range(8):
            scores = rng.integers(0, 5, size=(n, m)) / 4
            support = rng.integers(0, 2, size=(n, m)).astype(bool)
            result = assign(tuple(f"h{i}" for i in range(n)), tuple(f"c{j}" for j in range(m)), scores, support)
            valid = []
            for choices in itertools.product(range(-1, m), repeat=n):
                selected = [j for j in choices if j >= 0]
                if len(set(selected)) != len(selected) or any(j >= 0 and not support[i, j] for i, j in enumerate(choices)):
                    continue
                value = sum(.5 if j == -1 else scores[i, j] for i, j in enumerate(choices))
                valid.append((value, choices))
            optimum = max(value for value, _ in valid)
            best = [choices for value, choices in valid if value == optimum]
            assert result.optimum_score == optimum
            for i in range(n):
                assert result.optimal_unmatched[i] == any(path[i] == -1 for path in best)
                for j in range(m):
                    assert result.optimal_routes[i, j] == any(path[i] == j for path in best)
            for j in range(m):
                assert result.optimal_candidate_unmatched[j] == any(j not in path for path in best)


@pytest.mark.parametrize("bad", ["duplicate", "dtype", "masked", "shape", "empty_grid", "index", "keys"])
def test_invalid_mask_anchor_rejected(bad):
    keys = ("a", "b")
    masks = np.ones((2, 3, 3), bool)
    index = 0
    if bad == "duplicate": keys = ("a", "a")
    elif bad == "dtype": masks = masks.astype(np.uint8)
    elif bad == "masked": masks = np.ma.array(masks, mask=False)
    elif bad == "shape": masks = masks[0]
    elif bad == "empty_grid": masks = np.empty((2, 0, 3), bool)
    elif bad == "index": index = True
    elif bad == "keys": keys = ("a", "\n")
    with pytest.raises(ValueError): p.MaskAnchor(index, keys, masks)


@pytest.mark.parametrize("bad", ["reindex", "indexdtype", "trackdtype", "trackshape", "visdtype", "maskedtracks",
    "maskedvis", "visible_nan", "query_nan", "query_offimage", "query_wrongmask", "query_birth", "birthdtype",
    "owner", "duplicate_seed", "duplicate_source", "seed_missing", "unordered_anchors", "grid", "seedframe"])
def test_invalid_point_provenance_inputs_rejected(bad):
    args = crossing()
    if bad == "reindex": args["frame_index"][1] = 2
    elif bad == "indexdtype": args["frame_index"] = args["frame_index"].astype(np.int32)
    elif bad == "trackdtype": args["tracks"] = args["tracks"].astype(np.int64)
    elif bad == "trackshape": args["tracks"] = args["tracks"][:, :2]
    elif bad == "visdtype": args["visibility"] = args["visibility"].astype(np.uint8)
    elif bad == "maskedtracks": args["tracks"] = np.ma.array(args["tracks"], mask=False)
    elif bad == "maskedvis": args["visibility"] = np.ma.array(args["visibility"], mask=False)
    elif bad == "visible_nan": args["tracks"][0, 1, 0] = np.nan
    elif bad == "query_nan": args["query_points_xy"][0, 0] = np.nan
    elif bad == "query_offimage": args["query_points_xy"][0, 0] = 8.
    elif bad == "query_wrongmask": args["query_points_xy"][0] = [5.5, 1.5]
    elif bad == "query_birth": args["query_birth_frame_indices"][0] = 1
    elif bad == "birthdtype": args["query_birth_frame_indices"] = args["query_birth_frame_indices"].astype(np.int32)
    elif bad == "owner": args["query_ownership"] = ("missing", "A", "B", "B")
    elif bad == "duplicate_seed": args["seeds"] = (args["seeds"][0],) * 2
    elif bad == "duplicate_source": args["seeds"] = (args["seeds"][0], p.HypothesisSeed("B", 0, "left"))
    elif bad == "seed_missing": args["seeds"] = (p.HypothesisSeed("A", 0, "missing"), args["seeds"][1])
    elif bad == "unordered_anchors": args["anchors"] = args["anchors"][::-1]
    elif bad == "grid": args["anchors"] = args["anchors"][:1] + (p.MaskAnchor(2, ("a",), np.ones((1, 2, 2), bool)),)
    elif bad == "seedframe": args["seeds"] = (p.HypothesisSeed("A", 1, "left"), args["seeds"][1])
    with pytest.raises(ValueError): p.point_mask_evidence(**args)


@pytest.mark.parametrize("bad", ["duplicate_h", "duplicate_c", "score_nan", "score_inf", "score_range", "score_dtype",
    "shape", "support_dtype", "masked", "null_bool", "null_nan", "null_range", "tol_bool", "tol_neg", "tol_inf"])
def test_invalid_assignment_inputs_rejected(bad):
    rows, cols = ("A", "B"), ("a", "b")
    scores = np.eye(2)
    support = np.ones((2, 2), bool)
    null, tolerance = .5, 0.
    if bad == "duplicate_h": rows = ("A", "A")
    elif bad == "duplicate_c": cols = ("a", "a")
    elif bad == "score_nan": scores[0, 0] = np.nan
    elif bad == "score_inf": scores[0, 0] = np.inf
    elif bad == "score_range": scores[0, 0] = 1.1
    elif bad == "score_dtype": scores = scores.astype(np.int64)
    elif bad == "shape": scores = scores[:1]
    elif bad == "support_dtype": support = support.astype(np.uint8)
    elif bad == "masked": scores = np.ma.array(scores, mask=False)
    elif bad == "null_bool": null = True
    elif bad == "null_nan": null = np.nan
    elif bad == "null_range": null = -.1
    elif bad == "tol_bool": tolerance = False
    elif bad == "tol_neg": tolerance = -1.
    elif bad == "tol_inf": tolerance = np.inf
    with pytest.raises(ValueError):
        p.partial_mask_assignment(rows, cols, scores, support, unmatched_score=null, tie_tolerance=tolerance)

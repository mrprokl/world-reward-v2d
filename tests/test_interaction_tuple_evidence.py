"""Procedural observation plumbing checks; NOT automatic ownership validation."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import ast

import numpy as np
import pytest

from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.interaction_tuple_evidence import (
    FEATURE_NAMES, POSE_POINT_NAMES, SIDE_NAMES, build_interaction_tuple_evidence,
)


def person(count=2):
    points = np.full((count, 133, 2), 3., np.float64)
    points[:, 9] = [4., 4.]
    points[:, 91] = [6., 4.]
    points[:, 10] = [18., 12.]
    points[:, 112] = [20., 12.]
    return PersonPoseObservations(11, (16, 24), tuple(f"automatic-proposal-{i}" for i in range(count)),
                                  np.tile(np.array([[0., 0., 24., 16.]], np.float32), (count, 1)),
                                  np.arange(count, dtype=np.float32)-.75, points,
                                  np.full((count, 133), .25, np.float32))


def hoi(classes=(0, 1, 0, 1, 2)):
    classes = np.array(classes, np.int64)
    count = len(classes)
    boxes = np.tile(np.array([[5., 3., 9., 7.]], np.float32), (count, 1))
    decayed = np.linspace(-.4, .7, count, dtype=np.float32)
    nms = np.column_stack((boxes, decayed)).astype(np.float32)
    pairs = []
    for roles in ((0, 1), (1, 2)):
        left, right = (np.flatnonzero(classes == c) for c in roles)
        pairs.append(np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64))
    logits = [(np.arange(len(p)*2).reshape(len(p), 2)-5).astype(np.float32) for p in pairs]
    return HOIDetrObservations(11, (16, 24), np.zeros((1500, 3), np.float32),
                              np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
                              nms, np.arange(count, dtype=np.int64)[::-1].copy(),
                              np.arange(count, dtype=np.int64), np.full(count, 7, np.int64), classes,
                              boxes, np.full(count, .5, np.float32), decayed,
                              pairs[0], logits[0], pairs[1], logits[1])


def feature(result, name):
    index = result.feature_names.index(name)
    return result.features[:, index], result.feature_supported[:, index]


def test_all_person_side_pairs_in_original_order_with_duplicates_and_source_slots():
    p, h = person(3), hoi()
    r = build_interaction_tuple_evidence(p, h)
    assert len(h.hand_object_pairs) == 4 and r.features.shape == (24, 15)
    assert r.source_person_ids == p.person_ids
    a = r.arrays
    np.testing.assert_array_equal(a["person_slots"], np.repeat(np.arange(3), 8))
    np.testing.assert_array_equal(a["side_indices"], np.tile([0]*4+[1]*4, 3))
    np.testing.assert_array_equal(a["native_pair_slots"], np.tile(np.arange(4), 6))
    np.testing.assert_array_equal(a["detection_slots"], np.tile(h.hand_object_pairs, (6, 1)))
    np.testing.assert_array_equal(a["query_ids"], np.full((24, 2), 7))
    np.testing.assert_array_equal(a["native_flat_keep"], h.native_nms_keep[h.retained_nms_positions[a["detection_slots"]]])
    np.testing.assert_array_equal(a["retained_nms_positions"], h.retained_nms_positions[a["detection_slots"]])
    assert r.anatomical_ownership_verified is r.selection_performed is False
    assert len(r.source_observation_references) == 2
    assert all(len(sha) == 64 for _, sha in r.source_observation_references)


def test_native_body_wrist_and_hand_root_are_distinct_anatomical_indices():
    r = build_interaction_tuple_evidence(person(1), hoi((0, 1)))
    assert SIDE_NAMES == ("left", "right") and POSE_POINT_NAMES == ("body_wrist", "hand_root")
    np.testing.assert_array_equal(r.arrays["pose_keypoint_indices"], [[9, 91], [10, 112]])
    np.testing.assert_array_equal(r.arrays["pose_original_xy"], [[[4, 4], [6, 4]], [[18, 12], [20, 12]]])
    assert r.arrays["pose_original_xy"].dtype == np.float64
    assert r.arrays["pose_raw_scores"].dtype == np.float32
    assert r.arrays["hoi_logits"].dtype == np.float32


def test_geometric_distances_use_original_diagonal_without_clipping_or_probability():
    r = build_interaction_tuple_evidence(person(1), hoi((0, 1)))
    d = np.hypot(16, 24)
    expected = [1/d, np.hypot(3, 1)/d, 0., np.hypot(1, 1)/d, 2/d, 1., 16/384, 1.]
    np.testing.assert_allclose(r.features[0, :8], expected, rtol=0, atol=1e-15)
    assert r.feature_supported[0].all()
    np.testing.assert_array_equal(r.arrays["hoi_logits"][0], [-5., -4.])
    assert r.features[0, 8] == .25 and r.features[0, 10] == -.75
    assert r.features[0, 13] == np.float32(-.4)


@pytest.mark.parametrize("count,classes", [(0, (0, 1)), (2, ()), (2, (0,)), (2, (1,)), (0, ())])
def test_genuine_empty_person_or_pair_bank_has_no_dummy_null_predictions(count, classes):
    r = build_interaction_tuple_evidence(person(count), hoi(classes))
    assert r.features.shape == r.feature_supported.shape == (0, len(FEATURE_NAMES))
    assert r.arrays["pose_original_xy"].shape == (0, 2, 2)
    assert r.arrays["hoi_logits"].shape == (0, 2)
    assert r.arrays["detection_slots"].shape == (0, 2)
    assert r.source_person_ids == person(count).person_ids


@pytest.mark.parametrize("score", [0., -1., -100.])
def test_native_miss_is_nan_unsupported_but_raw_scores_and_coords_preserved(score):
    p = person(1)
    raw = p.raw_scores.copy(); raw[0, 9] = score
    p = replace(p, raw_scores=raw)
    r = build_interaction_tuple_evidence(p, hoi((0, 1)))
    assert np.isnan(r.features[0, [0, 1, 4]]).all()
    assert not r.feature_supported[0, [0, 1, 4]].any()
    assert r.feature_supported[0, [2, 3, 8]].all()
    assert r.features[0, 8] == score
    np.testing.assert_array_equal(r.arrays["pose_original_xy"][0, 0], [4., 4.])
    assert not r.arrays["pose_native_valid"][0, 0]
    assert r.arrays["pose_in_original_image"][0, 0]


@pytest.mark.parametrize("xy", [[-100., 4.], [24., 4.], [4., 16.], [4., -1.]])
def test_positive_native_offgrid_point_remains_finite_diagnostic_separate_coord_flag(xy):
    p = person(1); points = p.keypoints_original_xy.copy(); points[0, 9] = xy
    r = build_interaction_tuple_evidence(replace(p, keypoints_original_xy=points), hoi((0, 1)))
    assert r.feature_supported[0, [0, 1, 4]].all()
    assert np.isfinite(r.features[0, [0, 1, 4]]).all()
    assert r.arrays["pose_native_valid"][0, 0]
    assert not r.arrays["pose_in_original_image"][0, 0]
    np.testing.assert_array_equal(r.arrays["pose_original_xy"][0, 0], xy)


@pytest.mark.parametrize("box,positive,inside", [([-2., 3., 9., 7.], True, False),
                                               ([5., 3., 25., 7.], True, False),
                                               ([5., 3., 5., 7.], False, False),
                                               ([9., 3., 5., 7.], False, False)])
def test_native_offgrid_or_degenerate_hand_boxes_not_repaired_or_rows_dropped(box, positive, inside):
    h = hoi((0, 1)); boxes = h.boxes_original_xyxy.copy(); boxes[0] = box
    r = build_interaction_tuple_evidence(person(1), replace(h, boxes_original_xyxy=boxes))
    assert len(r.features) == 2
    assert bool(r.arrays["hoi_box_positive_area"][0, 0]) is positive
    assert bool(r.arrays["hoi_box_in_original_image"][0, 0]) is inside
    np.testing.assert_array_equal(r.arrays["hoi_boxes_original_xyxy"][0, 0], box)
    assert bool(r.feature_supported[0, 0]) is positive
    assert bool(r.feature_supported[0, 5]) is positive
    assert r.feature_supported[0, 7:].all()
    if not positive:
        assert np.isnan(r.features[0, [0, 1, 2, 3, 5, 6]]).all()
        assert r.feature_supported[0, 4]


def test_permutations_do_not_resolve_identity_or_deduplicate_competing_proposals():
    p = person(2); ids = (p.person_ids[1], p.person_ids[0])
    permuted = replace(p, person_ids=ids, detector_scores=p.detector_scores[::-1].copy())
    h = hoi((1, 0, 1, 0))
    r = build_interaction_tuple_evidence(permuted, h)
    assert r.source_person_ids == ids
    np.testing.assert_array_equal(r.arrays["detection_slots"][:4], [[1, 0], [1, 2], [3, 0], [3, 2]])
    np.testing.assert_array_equal(r.arrays["person_detector_scores"][:8], np.full(8, .25, np.float32))
    assert len(r.features) == 16


def test_raw_two_logits_and_native_score_banks_never_softmaxed_or_clamped():
    h = hoi(); logits = np.array([[-100, -120], [20, -10], [-2, 60], [0, 0]], np.float32)
    h = replace(h, hand_object_logits=logits)
    r = build_interaction_tuple_evidence(person(1), h)
    np.testing.assert_array_equal(r.arrays["hoi_logits"][:4], logits)
    np.testing.assert_array_equal(r.features[:4, 7], [-20, -30, 62, 0])
    np.testing.assert_array_equal(r.arrays["hoi_raw_scores"][:4], h.raw_scores[h.hand_object_pairs])
    np.testing.assert_array_equal(r.arrays["hoi_decayed_scores"][:4], h.decayed_scores[h.hand_object_pairs])


@pytest.mark.parametrize("change", [dict(original_frame_index=12), dict(image_size=(17, 24)), dict(image_size=(16, 25))])
def test_different_actual_frame_or_original_image_size_rejected(change):
    with pytest.raises(ValueError, match="Same original frame/image"):
        build_interaction_tuple_evidence(person(1), replace(hoi((0, 1)), **change))


@pytest.mark.parametrize("left,right", [(None, hoi(())), (person(0), None), ({}, {})])
def test_untyped_transport_or_null_bank_rejected(left, right):
    with pytest.raises(ValueError, match="observation types"):
        build_interaction_tuple_evidence(left, right)


def test_arrays_and_mapping_are_immutable_and_inputs_not_changed():
    p, h = person(), hoi()
    before = [(a, a.tobytes(), a.flags.writeable) for x in (p, h) for a in vars(x).values() if type(a) is np.ndarray]
    r = build_interaction_tuple_evidence(p, h)
    for array in (*r.arrays.values(), r.features, r.feature_supported):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        r.arrays["invented"] = np.array([1])
    with pytest.raises(FrozenInstanceError):
        r.original_frame_index = 0
    assert all(a.tobytes() == raw and a.flags.writeable == writable for a, raw, writable in before)


def test_references_bind_raw_unselected_bank_changes_not_only_selected_tuple():
    p, h = person(1), hoi((0, 1, 2))
    first = build_interaction_tuple_evidence(p, h)
    changed = h.query_logits.copy(); changed[1400, 2] = 1.
    second = build_interaction_tuple_evidence(p, replace(h, query_logits=changed))
    assert first.source_observation_references[0] == second.source_observation_references[0]
    assert first.source_observation_references[1] != second.source_observation_references[1]
    np.testing.assert_array_equal(first.features, second.features)


def test_numerical_overflow_is_unsupported_nan_not_a_false_finite_measurement():
    p = person(1); points = p.keypoints_original_xy.copy(); points[0, 9] = [np.finfo(np.float64).max]*2
    points[0, 91] = [-np.finfo(np.float64).max]*2
    r = build_interaction_tuple_evidence(replace(p, keypoints_original_xy=points), hoi((0, 1)))
    assert not r.feature_supported[0, 4] and np.isnan(r.features[0, 4])
    assert not np.isinf(r.features).any()
    assert r.arrays["pose_native_valid"][0].all()
    assert not r.arrays["pose_in_original_image"][0].any()


def test_pure_source_has_no_runtime_media_model_selection_or_gt_imports():
    path = Path(__file__).parents[1]/"src/world_reward/interaction_tuple_evidence.py"
    tree = ast.parse(path.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"torch", "cv2", "PIL", "subprocess", "os", "scipy"}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called & {"argmin", "argmax", "softmax", "clip", "sort", "open", "load"}

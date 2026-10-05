"""Tiny automatic-bank numerical contracts; no quality or ownership labels."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import ast

import numpy as np
import pytest

import world_reward.interaction_candidate_evidence as module
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.interaction_candidate_evidence import (
    GenericObjectObservations, build_interaction_candidate_evidence,
)


def person(n=2):
    xy = np.full((n, 133, 2), 3., np.float64)
    xy[:, 9], xy[:, 91], xy[:, 10], xy[:, 112] = [4., 4.], [6., 4.], [18., 12.], [20., 12.]
    return PersonPoseObservations(11, (16, 24), tuple(f'person-{i}' for i in range(n)),
        np.tile(np.array([[0., 0., 24., 16.]], np.float32), (n, 1)),
        np.arange(n, dtype=np.float32)-.75, xy, np.full((n, 133), .25, np.float32))


def objects(n=3, dtype=np.float32):
    return GenericObjectObservations(11, (16, 24), tuple(f'object-{i}' for i in range(n)),
        np.tile(np.array([[5., 3., 9., 7.]], dtype), (n, 1)),
        np.arange(n, dtype=np.float64)-.4, (('automatic-source-receipt', 'a'*64),))


def hoi(classes=(0, 1, 0, 1)):
    c = np.array(classes, np.int64); n = len(c)
    box = np.tile(np.array([[5., 3., 9., 7.]], np.float32), (n, 1))
    score = np.linspace(-.4, .7, n, dtype=np.float32)
    pairs = []
    for roles in ((0, 1), (1, 2)):
        a, b = (np.flatnonzero(c == role) for role in roles)
        pairs.append(np.column_stack((np.repeat(a, len(b)), np.tile(b, len(a)))).astype(np.int64))
    return HOIDetrObservations(11, (16, 24), np.zeros((1500, 3), np.float32),
        np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
        np.column_stack((box, score)).astype(np.float32), np.arange(n, dtype=np.int64)[::-1].copy(),
        np.arange(n, dtype=np.int64), np.full(n, 7, np.int64), c, box,
        np.full(n, .5, np.float32), score, pairs[0], np.full((len(pairs[0]), 2), -3., np.float32),
        pairs[1], np.full((len(pairs[1]), 2), -2., np.float32))


def test_complete_base_order_duplicates_without_any_hoi():
    p, o = person(2), objects(3)
    r = build_interaction_candidate_evidence(p, o)
    assert r.features.shape == (12, 10) and r.hoi_evidence is None
    a = r.arrays
    np.testing.assert_array_equal(a['person_slots'], [0]*6+[1]*6)
    np.testing.assert_array_equal(a['side_indices'], [0]*3+[1]*3+[0]*3+[1]*3)
    np.testing.assert_array_equal(a['generic_object_slots'], [0, 1, 2]*4)
    assert r.objects.object_ids == ('object-0', 'object-1', 'object-2')
    assert r.source_person_ids == p.person_ids and len(r.source_observation_references) == 2
    assert r.scope['hoi_provided'] is False and r.scope['hoi_native_pairs'] == 0
    assert r.scope['OFF'] != r.scope['UNKNOWN'] and r.scope['absence_predictions_generated'] is False
    assert r.selection_performed is r.anatomical_ownership_verified is False


@pytest.mark.parametrize('classes', [(), (0,), (1,), (2,), (0, 2)])
def test_empty_native_pair_bank_never_deletes_generic_candidates(classes):
    h = hoi(classes); r = build_interaction_candidate_evidence(person(), objects(), h)
    assert r.features.shape == (12, 10) and r.hoi_evidence.features.shape == (0, 15)
    assert r.route_features.shape == r.route_supported.shape == (0, 2)
    assert r.hoi_routes['native_pair_slots'].shape == (0,)
    assert r.scope['hoi_provided'] is True and r.scope['hoi_native_pairs'] == 0


@pytest.mark.parametrize('n,o', [(0, 0), (0, 3), (2, 0)])
def test_genuine_empty_base_has_no_null_prediction(n, o):
    r = build_interaction_candidate_evidence(person(n), objects(o), hoi())
    assert r.features.shape == (0, 10) and r.arrays['pose_original_xy'].shape == (0, 2, 2)
    assert r.scope['person_bank_empty'] == (n == 0) and r.scope['object_bank_empty'] == (o == 0)
    assert r.scope['absence_predictions_generated'] is False
    assert r.objects.boxes_original_xyxy.shape == (o, 4)


def test_separate_full_hoi_pair_object_routes_have_no_nearest_or_iou_filter():
    h = hoi(); r = build_interaction_candidate_evidence(person(), objects(3), h)
    assert r.hoi_evidence.features.shape == (16, 15) and r.route_features.shape == (12, 2)
    a = r.hoi_routes
    np.testing.assert_array_equal(a['native_pair_slots'], np.repeat(np.arange(4), 3))
    np.testing.assert_array_equal(a['generic_object_slots'], [0, 1, 2]*4)
    np.testing.assert_array_equal(a['detection_slots'], np.repeat(h.hand_object_pairs, 3, axis=0))
    np.testing.assert_array_equal(a['query_ids'], np.full((12, 2), 7))
    np.testing.assert_array_equal(a['raw_logits'], np.full((12, 2), -3., np.float32))
    np.testing.assert_array_equal(r.route_features, np.tile([1., 0.], (12, 1)))
    assert r.route_supported.all()


def test_disjoint_generic_object_route_remains_supported_with_zero_iou():
    o = replace(objects(2), boxes_original_xyxy=np.array([[5., 3., 9., 7.], [20., 10., 23., 15.]], np.float32))
    h = hoi((0, 1)); without = build_interaction_candidate_evidence(person(), o)
    r = build_interaction_candidate_evidence(person(), o, h)
    np.testing.assert_array_equal(r.features, without.features)
    assert r.route_features.shape == (2, 2) and r.route_features[1, 0] == 0
    assert r.route_supported[1].all() and r.route_features[1, 1] > 0
    np.testing.assert_array_equal(r.hoi_routes['generic_object_slots'], [0, 1])


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16, np.uint16])
def test_raw_dtype_scores_coordinates_preserved(dtype):
    o = objects(dtype=dtype); r = build_interaction_candidate_evidence(person(), o)
    assert r.arrays['object_boxes_original_xyxy'].dtype == dtype
    assert r.arrays['object_raw_scores'].dtype == np.float64
    assert r.arrays['pose_original_xy'].dtype == np.float64
    assert r.arrays['pose_raw_scores'].dtype == np.float32
    np.testing.assert_array_equal(r.arrays['object_raw_scores'][:3], o.raw_scores)
    assert r.features[0, 8] == np.float32(-.75) and r.features[0, 9] == -.4


def test_diagnostics_use_actual_grid_and_native_anatomical_indices():
    r = build_interaction_candidate_evidence(person(1), objects(1))
    d = np.hypot(16, 24)
    np.testing.assert_array_equal(r.arrays['pose_keypoint_indices'], [[9, 91], [10, 112]])
    np.testing.assert_allclose(r.features[0, :6], [1/d, np.hypot(3, 1)/d, 0., np.hypot(1, 1)/d, 2/d, 16/384],
                               rtol=0, atol=1e-15)
    assert r.feature_supported.all()


@pytest.mark.parametrize('score', [-2., 0.])
def test_missing_native_pose_keeps_raw_points_scores_but_no_distance(score):
    p = person(1); s = p.raw_scores.copy(); s[0, 9] = score; p = replace(p, raw_scores=s)
    r = build_interaction_candidate_evidence(p, objects(1))
    assert np.isnan(r.features[0, [0, 1, 4]]).all() and not r.feature_supported[0, [0, 1, 4]].any()
    assert r.features[0, 6] == score and r.feature_supported[0, 6]
    np.testing.assert_array_equal(r.arrays['pose_original_xy'][0, 0], [4., 4.])
    assert r.arrays['pose_native_valid'][0, 0] == False


def test_offgrid_positive_native_pose_is_diagnostic_not_visibility():
    p = person(1); xy = p.keypoints_original_xy.copy(); xy[0, 9] = [-8., 35.]; p = replace(p, keypoints_original_xy=xy)
    r = build_interaction_candidate_evidence(p, objects(1))
    assert np.isfinite(r.features[0, [0, 1, 4]]).all() and r.feature_supported[0, [0, 1, 4]].all()
    assert r.arrays['pose_native_valid'][0, 0] and not r.arrays['pose_in_original_image'][0, 0]


@pytest.mark.parametrize('box,positive', [([-2., 2., 9., 7.], True), ([20., 2., 30., 7.], True),
                                        ([5., 3., 5., 7.], False), ([9., 7., 5., 3.], False)])
def test_offgrid_or_degenerate_generic_object_kept_without_clip(box, positive):
    o = replace(objects(1), boxes_original_xyxy=np.array([box], np.float64))
    r = build_interaction_candidate_evidence(person(1), o, hoi((0, 1)))
    assert r.features.shape == (2, 10) and r.route_features.shape == (1, 2)
    np.testing.assert_array_equal(r.arrays['object_boxes_original_xyxy'][0], box)
    assert o.box_positive_area[0] == positive and not o.box_in_original_image[0]
    assert r.feature_supported[0, 0] == positive
    assert r.route_supported[0, 0] == positive


def test_permutation_reorders_slots_without_score_or_identity_choice():
    o = objects(3); permutation = [2, 0, 1]
    q = replace(o, object_ids=tuple(o.object_ids[i] for i in permutation),
                boxes_original_xyxy=o.boxes_original_xyxy[permutation], raw_scores=o.raw_scores[permutation])
    a, b = (build_interaction_candidate_evidence(person(), x) for x in (o, q))
    np.testing.assert_array_equal(a.features.reshape(4, 3, 10)[:, permutation], b.features.reshape(4, 3, 10))
    assert a.source_observation_references != b.source_observation_references


@pytest.mark.parametrize('changes', [dict(original_frame_index=12), dict(image_size=(17, 24))])
def test_any_frame_grid_mismatch_rejected(changes):
    with pytest.raises(ValueError, match='Same original'):
        build_interaction_candidate_evidence(person(), replace(objects(), **changes))
    with pytest.raises(ValueError, match='HOI must use'):
        build_interaction_candidate_evidence(person(), objects(), replace(hoi(), **changes))


@pytest.mark.parametrize('changes', [dict(object_ids=('same',)*3), dict(object_ids=['a', 'b', 'c']),
    dict(object_ids=('a', 'b', 'bad\n')), dict(raw_scores=np.array([0., np.nan, 1.])),
    dict(raw_scores=np.zeros(3, bool)), dict(boxes_original_xyxy=np.zeros((3, 3))),
    dict(boxes_original_xyxy=np.ma.array(np.zeros((3, 4)))), dict(original_frame_index=True),
    dict(image_size=(16, True)), dict(provenance_references=(('unverified', 'bad'),))])
def test_malformed_raw_bank_fails_without_repair(changes):
    with pytest.raises(ValueError):
        replace(objects(), **changes)


def test_all_returned_arrays_immutable_and_input_copies_nonaliasing():
    b = np.array([[5., 3., 9., 7.]], np.float32); s = np.array([-3.], np.float32)
    o = GenericObjectObservations(11, (16, 24), ('object',), b, s)
    b[:] = 0; s[:] = 10
    r = build_interaction_candidate_evidence(person(), o, hoi())
    np.testing.assert_array_equal(o.boxes_original_xyxy, [[5., 3., 9., 7.]])
    for x in (o.boxes_original_xyxy, o.raw_scores, o.box_positive_area, o.box_in_original_image,
              r.features, r.feature_supported, r.route_features, r.route_supported,
              *r.arrays.values(), *r.hoi_routes.values()):
        assert not x.flags.writeable
        with pytest.raises(ValueError):
            x.flags.writeable = True
    with pytest.raises(TypeError):
        r.scope['OFF'] = 'prediction'
    with pytest.raises(FrozenInstanceError):
        o.object_ids = ('replacement',)


def test_mutation_during_optional_hoi_build_rejected(monkeypatch):
    p = person(); original = module.build_interaction_tuple_evidence
    def changed(a, b):
        a.raw_scores.flags.writeable = True; a.raw_scores[0, 0] += .5
        return original(a, b)
    monkeypatch.setattr(module, 'build_interaction_tuple_evidence', changed)
    with pytest.raises(ValueError, match='Original banks changed'):
        build_interaction_candidate_evidence(p, objects(), hoi())


def test_finite_overflow_becomes_unsupported_not_fake_finite_or_filtered():
    o = replace(objects(1), boxes_original_xyxy=np.array([[-1e308, -1e308, 1e308, 1e308]]))
    r = build_interaction_candidate_evidence(person(), o, hoi())
    assert r.features.shape == (4, 10) and not r.feature_supported[:, 5].any()
    assert np.isnan(r.features[:, 5]).all() and r.arrays['object_box_positive_area'].all()


def test_wrong_types_rejected_and_source_has_no_io_models_or_private_truth():
    with pytest.raises(ValueError):
        build_interaction_candidate_evidence(person(), {}, hoi())
    with pytest.raises(ValueError):
        build_interaction_candidate_evidence(person(), objects(), {})
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(x and ('identity_calibration' in x or 'renderer' in x) for x in imports)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not {'open', 'eval', 'exec', 'select_interacting_actor', 'fit_identity_logistic'} & calls

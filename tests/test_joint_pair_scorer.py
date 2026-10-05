"""Manufactured automatic evidence/positive sets; no datasets or model adoption."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import ast
import json
from types import MappingProxyType

import numpy as np
import pytest

import world_reward.joint_pair_scorer as module
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_candidate_evidence import (
    GenericObjectObservations, build_interaction_candidate_evidence,
)
from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.joint_pair_scorer import (
    FEATURE_NAMES, JointPairFeatures, InsufficientPairFitError,
    build_joint_pair_features, fit_joint_pair_linear, score_baseline_pairs, score_joint_pairs,
)


def person(boxes=((0., 0., 10., 20.), (10., 0., 20., 20.))):
    n = len(boxes)
    xy = np.full((n, 133, 2), 3., np.float64)
    xy[:, 9], xy[:, 91], xy[:, 10], xy[:, 112] = [4., 4.], [6., 4.], [18., 12.], [20., 12.]
    return PersonPoseObservations(11, (20, 30), tuple(f'person-{i}' for i in range(n)),
        np.array(boxes, np.float32).reshape(-1, 4), np.full(n, .8, np.float32), xy,
        np.full((n, 133), .25, np.float32))


def objects(boxes=((5., 3., 9., 7.), (20., 10., 23., 15.))):
    n = len(boxes)
    return GenericObjectObservations(11, (20, 30), tuple(f'object-{i}' for i in range(n)),
        np.array(boxes, np.float32).reshape(-1, 4), np.full(n, .6, np.float32))


def hoi(classes=(0, 1), *, logits=None, boxes=None, raw=None):
    c = np.array(classes, np.int64); n = len(c)
    box = np.tile(np.array([[5., 3., 9., 7.]], np.float32), (n, 1)) if boxes is None else np.array(boxes, np.float32).reshape(-1, 4)
    pairs = []
    for a, b in ((0, 1), (1, 2)):
        left, right = (np.flatnonzero(c == role) for role in (a, b))
        pairs.append(np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64))
    scores = np.full(n, .5, np.float32) if raw is None else np.asarray(raw, np.float32)
    interaction = np.tile(np.array([[-2., 1.]], np.float32), (len(pairs[0]), 1)) if logits is None else np.asarray(logits, np.float32)
    return HOIDetrObservations(11, (20, 30), np.zeros((1500, 3), np.float32),
        np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
        np.column_stack((box, scores)).astype(np.float32), np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), c, box, scores, scores,
        pairs[0], interaction, pairs[1], np.zeros((len(pairs[1]), 2), np.float32))


def bank(p=None, o=None, h=None):
    p = person() if p is None else p
    o = objects() if o is None else o
    return build_joint_pair_features(p, build_interaction_candidate_evidence(p, o, h))


def toy_bank(*, sign=1., offset=0., available=True):
    values = np.zeros((1, 2, len(FEATURE_NAMES)))
    values[0, :, 0] = np.array([-1., 1.]) * sign + offset if available else 0.
    ok = np.zeros(values.shape, bool); ok[..., 0] = available
    return JointPairFeatures(0, (10, 10), np.array([[0., 0., 1., 1.]]),
        np.array([[0., 0., .2, .2], [.8, .8, 1., 1.]]), np.array([0], np.int64),
        np.array([0, 1], np.int64), values, ok, np.zeros((1, 2, 2)), ())


def fit_inputs():
    banks = tuple(toy_bank(sign=1. if i % 2 else -1.) for i in range(24))
    labels = tuple(np.array([[i % 2 == 0, i % 2 == 1]]) for i in range(24))
    keys = dict(image_ids=tuple(f'image-{i:02d}' for i in range(24)),
                author_groups=tuple(f'author-{i:02d}' for i in range(24)))
    return banks, labels, keys


def test_exact_feature_schema_base_side_pool_and_original_normalized_grid():
    r = bank()
    assert len(FEATURE_NAMES) == 30 and r.values.shape == r.available.shape == (2, 2, 30)
    np.testing.assert_allclose(r.person_boxes_normalized, [[0, 0, 1/3, 1], [1/3, 0, 2/3, 1]], rtol=0, atol=0)
    np.testing.assert_array_equal(r.raw_person_to_unique, [0, 1])
    d = np.hypot(20, 30)
    side0, side1 = 1/d, np.hypot(9, 5)/d
    np.testing.assert_allclose(r.values[0, 0, :3], [side0, side1, (side0+side1)/2], rtol=0, atol=1e-15)
    assert r.available[..., :22].all()
    assert np.all(r.values[..., 22:] == 0) and not r.available[..., 22:].any()


def test_baseline_is_original_person_diagonal_and_lexicographic_exact_ties():
    r = bank(person(((0., 0., 10., 20.),)), objects(((5., 3., 9., 7.), (20., 10., 23., 15.))))
    diagonal = np.hypot(10, 20)
    np.testing.assert_allclose(r.baseline_a[0, 0], [0., -np.hypot(2, 5)/diagonal], rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.baseline_a[0, 1], [-11.5/diagonal, -np.hypot(16.5, 2.5)/diagonal], rtol=0, atol=1e-15)
    # Primary component dominates without a weighted combination or tiny margin.
    custom = replace(r, baseline_a=np.array([[[-1., -100.], [-1., -1.]]]))
    np.testing.assert_array_equal(score_baseline_pairs(custom), [[0., 1.]])
    custom = replace(r, baseline_a=np.array([[[-1., -100.], [-2., 1000.]]]))
    np.testing.assert_array_equal(score_baseline_pairs(custom), [[1., 0.]])
    custom = replace(r, baseline_a=np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(score_baseline_pairs(custom), [[0., 0.]])


def test_exact_coordinate_aliases_keep_all_raw_maps_but_do_not_weight_copies():
    single = bank(person(((0., 0., 10., 20.),)), objects(((5., 3., 9., 7.),)), hoi())
    duplicated = bank(person(((0., 0., 10., 20.),)*3), objects(((5., 3., 9., 7.),)*4), hoi((0, 1, 0, 1)))
    assert duplicated.values.shape == (1, 1, 30)
    np.testing.assert_array_equal(duplicated.raw_person_to_unique, [0, 0, 0])
    np.testing.assert_array_equal(duplicated.raw_object_to_unique, [0, 0, 0, 0])
    np.testing.assert_array_equal(single.values, duplicated.values)
    np.testing.assert_array_equal(single.available, duplicated.available)
    np.testing.assert_array_equal(single.baseline_a, duplicated.baseline_a)


def test_aliases_with_different_automatic_values_pool_not_error_or_gt_choice():
    p = person(((0., 0., 10., 20.),)*3)
    p = replace(p, detector_scores=np.array([.3, .6, .6]))
    r = bank(p, objects(((5., 3., 9., 7.),)))
    assert r.values[0, 0, FEATURE_NAMES.index('person_detector_raw_score_mean')] == pytest.approx(.45)
    assert r.raw_person_to_unique.tolist() == [0, 0, 0]


def test_side_swap_and_raw_proposal_order_do_not_change_grouped_features():
    p, o = person(), objects()
    xy, scores = p.keypoints_original_xy.copy(), p.raw_scores.copy()
    xy[:, [9, 10, 91, 112]] = xy[:, [10, 9, 112, 91]]
    scores[:, [9, 10, 91, 112]] = scores[:, [10, 9, 112, 91]]
    swapped = replace(p, keypoints_original_xy=xy, raw_scores=scores)
    permuted = replace(swapped, person_ids=swapped.person_ids[::-1], boxes_original_xyxy=swapped.boxes_original_xyxy[::-1],
        detector_scores=swapped.detector_scores[::-1], keypoints_original_xy=swapped.keypoints_original_xy[::-1], raw_scores=swapped.raw_scores[::-1])
    reverse_objects = replace(o, object_ids=o.object_ids[::-1], boxes_original_xyxy=o.boxes_original_xyxy[::-1], raw_scores=o.raw_scores[::-1])
    a, b = bank(p, o, hoi()), bank(permuted, reverse_objects, hoi())
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.available, b.available)
    np.testing.assert_array_equal(a.baseline_a, b.baseline_a)


def test_missing_points_are_zero_plus_false_availability_not_fake_anatomy():
    p = person(((0., 0., 10., 20.),)); scores = p.raw_scores.copy()
    scores[:, [9, 10, 91, 112]] = 0
    r = bank(replace(p, raw_scores=scores), objects(((5., 3., 9., 7.),)), hoi())
    assert np.all(r.values[..., :15] == 0) and not r.available[..., :15].any()
    assert r.available[..., 15:22].all()
    assert not r.available[..., 22:24].any() and not r.available[..., 27:30].any()
    assert r.available[..., 24:27].all()  # Native direct bridge/margin still exists.


@pytest.mark.parametrize('classes', [None, (), (0,), (1,), (2,), (0, 2)])
def test_optional_or_empty_hoi_has_no_dummy_route_or_base_loss(classes):
    h = None if classes is None else hoi(classes)
    a, b = bank(), bank(h=h)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.available, b.available)
    assert np.all(b.values[..., 22:] == 0) and not b.available[..., 22:].any()


def test_hoi_original_margin_and_geometry_weighted_raw_scores():
    r = bank(person(((0., 0., 10., 20.),)), objects(((5., 3., 9., 7.),)), hoi(raw=(.4, .8)))
    j = len(FEATURE_NAMES)-8
    assert r.values[0, 0, j+2] == 1 and r.values[0, 0, j+3] == 0
    assert r.values[0, 0, j+4] == r.values[0, 0, j+5] == 3
    assert r.values[0, 0, j+6] == pytest.approx(np.float32(.4))
    assert r.values[0, 0, j+7] == pytest.approx(np.float32(.8))
    assert r.available[0, 0, j:].all()


def test_zero_iou_route_still_has_raw_margin_but_no_weighted_evidence():
    r = bank(person(((0., 0., 10., 20.),)), objects(((20., 10., 23., 15.),)), hoi())
    assert r.values[0, 0, 24] == 0 and r.available[0, 0, 24]
    assert r.values[0, 0, 26] == 3 and r.available[0, 0, 26]
    assert np.all(r.values[0, 0, 27:] == 0) and not r.available[0, 0, 27:].any()


def test_all_hoi_routes_including_disjoint_high_margin_preserved():
    h = hoi((0, 1, 1), logits=((-2., 1.), (-10., 10.)),
            boxes=((5., 3., 9., 7.), (5., 3., 9., 7.), (20., 10., 23., 15.)))
    r = bank(person(((0., 0., 10., 20.),)), objects(((5., 3., 9., 7.),)), h)
    assert r.values[0, 0, 26] == 20  # No IoU gate on raw margin max.
    assert r.values[0, 0, 27] == 3  # Fixed IoU weights are a separate descriptor.


@pytest.mark.parametrize('n,m', [(0, 0), (0, 2), (2, 0)])
def test_empty_banks_shape_preservation_no_full_image_fallback(n, m):
    r = bank(person(((0., 0., 10., 20.),)*n), objects(((5., 3., 9., 7.),)*m), hoi())
    assert r.values.shape == (int(n > 0), int(m > 0), 30)
    assert r.baseline_a.shape == (*r.values.shape[:2], 2)
    assert score_baseline_pairs(r).shape == r.values.shape[:2]


def test_person_fingerprint_or_incomplete_base_or_hoi_rejected():
    p, o = person(), objects(); e = build_interaction_candidate_evidence(p, o, hoi())
    changed = replace(p, detector_scores=p.detector_scores + .1)
    with pytest.raises(ValueError, match='must match'): build_joint_pair_features(changed, e)
    with pytest.raises(ValueError, match='Complete base'): build_joint_pair_features(p, replace(e, features=e.features[:-1]))
    with pytest.raises(ValueError, match='Complete original HOI'):
        build_joint_pair_features(p, replace(e, route_features=e.route_features[:-1]))


def test_hoi_native_slot_relabel_not_silently_reshape_joined():
    p, o = person(), objects(); e = build_interaction_candidate_evidence(p, o, hoi((0, 1, 1)))
    routes = dict(e.hoi_routes); routes['native_pair_slots'] = routes['native_pair_slots'][::-1]
    with pytest.raises(ValueError, match='slot order'):
        build_joint_pair_features(p, replace(e, hoi_routes=MappingProxyType(routes)))
    arrays = dict(e.hoi_evidence.arrays); arrays['side_indices'] = 1-arrays['side_indices']
    h = replace(e.hoi_evidence, arrays=MappingProxyType(arrays))
    with pytest.raises(ValueError, match='slot order'):
        build_joint_pair_features(p, replace(e, hoi_evidence=h))


def test_negative_width_object_rejected_not_clipped_and_zero_area_kept():
    with pytest.raises(ValueError, match='noninverted'):
        bank(o=objects(((9., 7., 5., 3.),)))
    r = bank(o=objects(((5., 3., 5., 7.),)))
    assert r.values.shape == (2, 1, 30) and not r.available[..., :12].any()


def test_feature_output_immutable_and_no_input_mutation():
    p, o = person(), objects(); e = build_interaction_candidate_evidence(p, o, hoi())
    snapshots = (p.keypoints_original_xy.tobytes(), o.boxes_original_xyxy.tobytes(), e.features.tobytes())
    r = build_joint_pair_features(p, e)
    assert snapshots == (p.keypoints_original_xy.tobytes(), o.boxes_original_xyxy.tobytes(), e.features.tobytes())
    for a in (r.values, r.available, r.baseline_a, r.person_boxes_normalized, r.object_boxes_normalized,
              r.raw_person_to_unique, r.raw_object_to_unique):
        assert not a.flags.writeable
        with pytest.raises(ValueError): a.setflags(write=True)
    with pytest.raises(FrozenInstanceError): r.image_size = (1, 1)


def test_fit_single_zero_start_fixed_loss_and_dense_all_pair_scores():
    banks, labels, keys = fit_inputs(); model = fit_joint_pair_linear(banks, labels, **keys)
    assert model.fit_record_count == model.usable_fit_count == 24
    assert model.missing_positive_fit_count == model.no_alternative_fit_count == 0
    assert model.gradient_inf_norm <= module.GRADIENT_TOLERANCE and model.objective < np.log(2)
    assert model.coefficients.shape == (60,) and model.coefficients[0] > 0
    assert np.all(model.coefficients[1:] == 0)
    for b, mask in zip(banks, labels):
        scores = score_joint_pairs(model, b)
        assert scores.shape == (1, 2) and scores[mask].min() > scores[~mask].max()
        assert not scores.flags.writeable
    repeated = fit_joint_pair_linear(banks, labels, **keys)
    np.testing.assert_array_equal(model.coefficients, repeated.coefficients)
    for a in (model.mean, model.scale, model.coefficients):
        with pytest.raises(ValueError): a.setflags(write=True)


def test_standardization_fit_only_absence_not_zero_measurement_and_score_no_refit():
    banks, labels, keys = fit_inputs()
    altered = list(banks)
    # Feature1 only observed in12 FIT records with value10; unavailable others0.
    for i, b in enumerate(altered):
        x, a = b.values.copy(), b.available.copy()
        if i < 12: x[..., 1] = 10.; a[..., 1] = True
        altered[i] = replace(b, values=x, available=a)
    model = fit_joint_pair_linear(tuple(altered), labels, **keys)
    assert model.mean[1] == pytest.approx(10.) and model.scale[1] == 1
    assert model.mean[2] == 0 and model.scale[2] == 1
    before = model.mean.tobytes(), model.scale.tobytes(), model.coefficients.tobytes()
    score_joint_pairs(model, toy_bank(offset=10000.))
    assert before == (model.mean.tobytes(), model.scale.tobytes(), model.coefficients.tobytes())


def test_exact_non_dyadic_constant_has_scale_one_not_tiny_rounding_std():
    banks, labels, keys = fit_inputs(); updated = []
    for b in banks:
        x, ok = b.values.copy(), b.available.copy(); x[..., 1] = .3; ok[..., 1] = True
        updated.append(replace(b, values=x, available=ok))
    model = fit_joint_pair_linear(tuple(updated), labels, **keys)
    assert model.scale[1] == 1 and model.mean[1] == pytest.approx(.3)


def test_pairs_per_image_not_annotation_count_define_equal_image_mass():
    banks, labels, keys = fit_inputs(); updated, masks = list(banks), list(labels)
    for i, b in enumerate(updated):
        x, ok = b.values.copy(), b.available.copy()
        if i == 0:
            x = np.concatenate((x, x), axis=1); ok = np.concatenate((ok, ok), axis=1)
            x[..., 1] = 24.; ok[..., 1] = True
            updated[i] = replace(b, object_boxes_normalized=np.array([[0, 0, .1, .1], [.2, .2, .3, .3],
                [.4, .4, .5, .5], [.8, .8, 1, 1]], np.float64), values=x, available=ok, baseline_a=np.zeros((1, 4, 2)))
            masks[i] = np.concatenate((masks[i], masks[i]), axis=1)
        else:
            x[..., 1] = 0.; ok[..., 1] = True; updated[i] = replace(b, values=x, available=ok)
    model = fit_joint_pair_linear(tuple(updated), tuple(masks), **keys)
    assert model.mean[1] == pytest.approx(1.)  # Each image1/24, not each pair1/50.


def test_fit_image_order_permutation_is_deterministic_and_balanced():
    banks, labels, keys = fit_inputs(); first = fit_joint_pair_linear(banks, labels, **keys)
    permutation = list(range(24))[::-1]
    second = fit_joint_pair_linear(tuple(banks[i] for i in permutation), tuple(labels[i] for i in permutation),
        image_ids=tuple(keys['image_ids'][i] for i in permutation), author_groups=tuple(keys['author_groups'][i] for i in permutation))
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.fit_image_ids == second.fit_image_ids


def test_missing_positive_and_no_alternative_counted_not_repaired():
    banks, labels, keys = fit_inputs(); masks = list(labels)
    masks[:6] = [np.zeros((1, 2), bool)]*6
    masks[6:12] = [np.ones((1, 2), bool)]*6
    model = fit_joint_pair_linear(banks, tuple(masks), **keys)
    assert model.usable_fit_count == 12 and model.missing_positive_fit_count == model.no_alternative_fit_count == 6
    assert len(model.fit_image_ids) == 12
    masks[12] = np.zeros((1, 2), bool)
    with pytest.raises(InsufficientPairFitError): fit_joint_pair_linear(banks, tuple(masks), **keys)


def test_empty_fit_banks_keep_slots_and_count_missing_positive():
    banks, labels, keys = fit_inputs(); bb, mm = list(banks), list(labels)
    bb[:2] = [bank(person(()), objects(()))]*2
    mm[:2] = [np.empty((0, 0), bool)]*2
    model = fit_joint_pair_linear(tuple(bb), tuple(mm), **keys)
    assert model.usable_fit_count == 22 and model.missing_positive_fit_count == 2
    assert score_joint_pairs(model, bb[0]).shape == (0, 0)


def test_positive_set_multi_positive_gradient_matches_finite_difference():
    design = np.array([[1., 0.], [0., 1.], [-1., -.5]])
    mask = np.array([True, True, False]); theta = np.array([.2, -.3])
    loss, gradient = module._loss_gradient(theta, [(design, mask)])
    epsilon = 1e-6
    numerical = []
    for i in range(2):
        d = np.eye(2)[i]*epsilon
        numerical.append((module._loss_gradient(theta+d, [(design, mask)])[0]
                          - module._loss_gradient(theta-d, [(design, mask)])[0])/(2*epsilon))
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)


def test_stable_listwise_values_translation_and_pair_order_invariance():
    a, weights = module._softmax(np.array([1000., 1001., 1002.]))
    b, shifted = module._softmax(np.array([-1000., -999., -998.]))
    assert a-b == pytest.approx(2000.)
    np.testing.assert_array_equal(weights, shifted)
    design = np.array([[1., 0.], [0., 1.], [-1., -.5]])
    mask = np.array([True, True, False]); theta = np.array([.2, -.3])
    original = module._loss_gradient(theta, [(design, mask)])
    reordered = module._loss_gradient(theta, [(design[::-1], mask[::-1])])
    assert original[0] == pytest.approx(reordered[0])
    np.testing.assert_allclose(original[1], reordered[1], atol=1e-15, rtol=0)


def test_unknown_alternative_is_not_binary_label_but_denominator_gradient_suppresses_it():
    design = np.eye(3); theta = np.zeros(3); mask = np.array([True, False, False])
    _, gradient = module._loss_gradient(theta, [(design, mask)])
    np.testing.assert_allclose(gradient, [-2/3, 1/3, 1/3], rtol=0, atol=1e-15)


@pytest.mark.parametrize('change', ['count', 'duplicate_author', 'duplicate_image', 'non_boolean', 'bad_shape'])
def test_fit_split_and_positive_schema_fail_closed(change):
    banks, labels, keys = fit_inputs()
    if change == 'count': banks, labels = banks[:-1], labels[:-1]
    elif change == 'duplicate_author': keys['author_groups'] = ('same',)*24
    elif change == 'duplicate_image': keys['image_ids'] = ('same',)*24
    elif change == 'non_boolean': labels = (labels[0].astype(np.int64), *labels[1:])
    else: labels = (np.zeros((1, 1), bool), *labels[1:])
    with pytest.raises(ValueError): fit_joint_pair_linear(banks, labels, **keys)


@pytest.mark.parametrize('kind', ['nan', 'missing_without_zero', 'masked', 'wrong_features'])
def test_malformed_feature_bank_rejected_before_fit(kind):
    banks, labels, keys = fit_inputs(); b = banks[0]
    if kind in ('nan', 'missing_without_zero'):
        x = b.values.copy(); x[..., -1] = np.nan if kind == 'nan' else 1.; b = replace(b, values=x)
    elif kind == 'masked': b = replace(b, values=np.ma.array(b.values))
    else: b = replace(b, feature_names=('not_original',))
    with pytest.raises(ValueError): fit_joint_pair_linear((b, *banks[1:]), labels, **keys)


def test_unpooled_exact_coordinate_aliases_rejected_not_extra_fit_mass():
    banks, labels, keys = fit_inputs()
    b = replace(banks[0], object_boxes_normalized=np.tile(banks[0].object_boxes_normalized[:1], (2, 1)))
    with pytest.raises(ValueError, match='already be pooled'):
        fit_joint_pair_linear((b, *banks[1:]), labels, **keys)


def test_deadline_and_nonconvergence_fail_no_partial_model(monkeypatch):
    banks, labels, keys = fit_inputs()
    clock = iter([0., 0., module.FIT_SECONDS+1])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    with pytest.raises(RuntimeError, match='deadline'): fit_joint_pair_linear(banks, labels, **keys)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    monkeypatch.setattr(module, 'MAX_ITERATIONS', 0)
    with pytest.raises(RuntimeError, match='did not converge'): fit_joint_pair_linear(banks, labels, **keys)


def test_line_search_failure_does_not_return_partial_model(monkeypatch):
    banks, labels, keys = fit_inputs()
    monkeypatch.setattr(module, 'MAX_BACKTRACKS', 0)
    with pytest.raises(RuntimeError, match='line search failed'): fit_joint_pair_linear(banks, labels, **keys)


def test_scoring_invalid_fitted_arrays_and_nonfinite_fit_numerics_fail():
    banks, labels, keys = fit_inputs(); model = fit_joint_pair_linear(banks, labels, **keys)
    with pytest.raises(ValueError): score_joint_pairs(replace(model, scale=np.zeros(30)), banks[0])
    with pytest.raises(ValueError): score_joint_pairs(replace(model, coefficients=np.full(60, np.nan)), banks[0])
    with pytest.raises(ValueError): score_joint_pairs(replace(model, feature_names=('wrong',)), banks[0])
    x = banks[0].values.copy(); x[..., 0] = [-1e308, 1e308]
    with pytest.raises((ValueError, FloatingPointError)):
        fit_joint_pair_linear((replace(banks[0], values=x), *banks[1:]), labels, **keys)


def test_config_matches_fixed_recipe_and_no_data_model_or_private_parser_calls():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/openimages_joint_pair_scorer_v1.json').read_text())
    assert cfg['fit']['image_slots'] == module.FIT_SLOTS == 24
    assert cfg['fit']['minimum_usable_distinct_authors'] == module.MIN_USABLE_FIT == 12
    assert cfg['fit']['L2'] == module.L2 == 1
    assert cfg['fit']['maximum_iterations'] == module.MAX_ITERATIONS
    assert cfg['fit']['maximum_backtracks'] == module.MAX_BACKTRACKS
    assert cfg['fit']['budget_seconds'] == module.FIT_SECONDS
    assert cfg['fit']['gradient_inf_tolerance'] == module.GRADIENT_TOLERANCE
    tree = ast.parse(Path(module.__file__).read_text())
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not {'open', 'eval', 'exec', 'parse_holds_reference', 'default_rng', 'minimize'} & calls
    assert cfg['decision']['use'] == 'diagnostic_only_no_threshold_hyperparameter_or_grid_selection'
    assert cfg['test']['opens'] == 1 and cfg['implementation_PASS_is_quality_PASS'] is False

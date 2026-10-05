"""Small manufactured numerical controls; no datasets, runtime or ownership truth."""
from dataclasses import FrozenInstanceError, replace
import inspect

import numpy as np
import pytest

import world_reward.coherent_pair_learning as m
from world_reward.coherent_route_scorer import score_coherent_routes
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_candidate_evidence import GenericObjectObservations, build_interaction_candidate_evidence
from world_reward.person_pose_observations import PersonPoseObservations


def person(n=1):
    xy = np.full((n, 133, 2), 5., np.float64)
    xy[:, [9, 91, 10, 112]] = [[3., 4.], [4., 4.], [15., 10.], [16., 10.]]
    return PersonPoseObservations(2, (20, 30), tuple(f'p{i}' for i in range(n)),
        np.tile([[0., 0., 25., 18.]], (n, 1)), np.full(n, .9, np.float32), xy, np.full((n, 133), .7, np.float32))


def objects(n=2):
    boxes = np.column_stack((np.arange(n)*5.+5., np.full(n, 3.), np.arange(n)*5.+8., np.full(n, 7.)))
    return GenericObjectObservations(2, (20, 30), tuple(f'o{i}' for i in range(n)), boxes, np.zeros(n))


def hoi(copies=1):
    boxes = np.tile([[2., 2., 5., 6.], [5., 3., 8., 7.]], (copies, 1)).astype(np.float32)
    n = len(boxes); cls = np.tile([0, 1], copies).astype(np.int64)
    hand, obj = np.flatnonzero(cls == 0), np.flatnonzero(cls == 1)
    pairs = np.column_stack((np.repeat(hand, len(obj)), np.tile(obj, len(hand)))).astype(np.int64)
    scores = np.full(n, .5, np.float32)
    return HOIDetrObservations(2, (20, 30), np.zeros((1500, 3), np.float32), np.zeros((1500, 4), np.float32),
        np.zeros((1500, 256), np.float32), np.column_stack((boxes, scores)).astype(np.float32),
        np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), cls,
        boxes, scores, scores, pairs, np.tile([[-2., 1.]], (len(pairs), 1)).astype(np.float32),
        np.empty((0, 2), np.int64), np.empty((0, 2), np.float32))


def bank(*, n=1, o=2, h=True):
    p = person(n)
    return m.pair_route_bank(p, build_interaction_candidate_evidence(p, objects(o), hoi() if h else None))


def scales():
    return m.PairScale(np.arange(1., 13.), np.ones(12, bool))


def brute_vectors(b, scale):
    """Independent scalar enumeration of all whole geometry+margin branches."""
    e = b.evidence; n, o, k = len(e.source_person_ids), len(e.objects.object_ids), e.scope['hoi_native_pairs']
    out = [[[] for _ in b.object_boxes] for _ in b.person_boxes]
    for p in range(n):
        for s in range(2):
            for j in range(o):
                row = (p*2+s)*o+j; pb, ob = e.arrays['person_boxes_original_xyxy'][row], e.objects.boxes_original_xyxy[j]
                if not (pb[2] > pb[0] and pb[3] > pb[1] and ob[2] > ob[0] and ob[3] > ob[1]): continue
                vector = np.zeros(18)
                for f in range(6):
                    if e.feature_supported[row, f] and scale.variable[f]: vector[f] = e.features[row, f]/scale.scale[f]
                for f, c in enumerate((0, 2, 4), 12): vector[f] = e.feature_supported[row, c]
                rows = []
                for r in range(k):
                    hrow = (p*2+s)*k+r; h = e.hoi_evidence; boxes = h.arrays['hoi_boxes_original_xyxy'][hrow]
                    if not all(box[2] > box[0] and box[3] > box[1] for box in boxes) or not h.feature_supported[hrow, 7]: continue
                    v = vector.copy()
                    for f in range(4):
                        if h.feature_supported[hrow, f] and scale.variable[6+f]: v[6+f] = h.features[hrow, f]/scale.scale[6+f]
                    for f in range(2):
                        rrow = r*o+j
                        if e.route_supported[rrow, f] and scale.variable[10+f]: v[10+f] = e.route_features[rrow, f]/scale.scale[10+f]
                    v[15], v[16], v[17] = h.feature_supported[hrow, 0], h.feature_supported[hrow, 2], h.features[hrow, 7]
                    rows.append(v)
                out[b.person_to_group[p]][b.object_to_group[j]].extend(rows if rows else [vector])
    return out


def test_fullbank_scores_match_original_masked_core_and_brute_complete_vectors():
    b = bank(n=2); scale = scales(); theta = np.linspace(-.8, .9, 17)
    result = m.score_pair_groups(b, theta, scale, alpha=.4)
    native = score_coherent_routes(b.evidence, m.masked_model(theta, scale, alpha=.4))
    np.testing.assert_array_equal(result.native.scores_b, native.scores_b)
    vectors = brute_vectors(b, scale)
    for p, row in enumerate(vectors):
        for j, v in enumerate(row):
            values = np.asarray(v) @ np.r_[theta, .4]
            assert result.scores[p, j] == pytest.approx(values.max(), abs=1e-14)
    assert result.native.scores_a.shape == (2, 2, 2)
    assert result.native.source_person_ids == b.evidence.source_person_ids


def test_alpha_zero_same_support_scores_and_all_native_ids_preserved():
    b = bank(); t = np.arange(17.)/10
    a, zero = m.score_pair_groups(b, t, scales()), m.score_pair_groups(b, t, scales(), alpha=0.)
    np.testing.assert_array_equal(a.scores, zero.scores)
    np.testing.assert_array_equal(a.supported, zero.supported)
    np.testing.assert_array_equal(zero.native.native_pair_slots, [0])
    assert zero.derivatives.shape == (*zero.scores.shape, 1)


def test_loss_gradient_independent_finite_difference_at_untied_branches():
    b = bank(); t = np.linspace(-.71, .84, 17); mask = (np.array([[True, False]]),)
    value, gradient, counts = m.loss_gradient(t, (b,), mask, scales())
    eps = 1e-6; numerical = []
    for j in range(17):
        step = np.zeros(17); step[j] = eps
        plus = m.loss_gradient(t+step, (b,), mask, scales())[0]
        minus = m.loss_gradient(t-step, (b,), mask, scales())[0]
        numerical.append((plus-minus)/(2*eps))
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-7)
    alpha = .37
    _, ag, _ = m.loss_gradient(t, (b,), mask, scales(), alpha=alpha)
    fd = (m.loss_gradient(t, (b,), mask, scales(), alpha=alpha+eps)[0]
          -m.loss_gradient(t, (b,), mask, scales(), alpha=alpha-eps)[0])/(2*eps)
    np.testing.assert_allclose(ag, [fd], atol=2e-8)
    assert value > 0 and counts == dict(fit_records=1, used=1, missing_positive=0, no_alternative=0)


def test_relational_alpha_has_actual_pair_specific_loss_gradient_and_improvement():
    p = person(); h = hoi(copies=2); logits = h.hand_object_logits.copy(); logits[:, 1] = [-2., 0., -2., -2.]
    e = build_interaction_candidate_evidence(p, objects(), replace(h, hand_object_logits=logits))
    bridge = e.route_features.copy(); bridge[:, 0] = np.array([[0., -4.], [-4., 0.], [-10., -10.], [-10., -10.]]).ravel()
    b = m.pair_route_bank(p, replace(e, route_features=bridge))
    theta = np.zeros(17); theta[10] = 1.; unit = m.PairScale(np.ones(12), np.ones(12, bool))
    masks = (np.array([[False, True]]),); alpha = .2; eps = 1e-6
    value, gradient, _ = m.loss_gradient(theta, (b,), masks, unit, alpha=alpha)
    fd = (m.loss_gradient(theta, (b,), masks, unit, alpha=alpha+eps)[0]
          -m.loss_gradient(theta, (b,), masks, unit, alpha=alpha-eps)[0])/(2*eps)
    assert gradient[0] < 0
    np.testing.assert_allclose(gradient, [fd], atol=1e-9)
    initial, _, _ = m.loss_gradient(theta, (b,), masks, unit, alpha=0.)
    assert value < initial


def test_positive_set_gradient_retains_multiple_positives_and_equal_image_mass():
    b = bank(o=3); theta = np.linspace(-.2, .3, 17); unit = scales()
    multi, single = np.array([[True, True, False]]), np.array([[False, False, True]])
    l1, g1, _ = m.loss_gradient(theta, (b,), (multi,), unit)
    l2, g2, _ = m.loss_gradient(theta, (b,), (single,), unit)
    loss, gradient, counts = m.loss_gradient(theta, (b, b), (multi, single), unit)
    assert loss == pytest.approx((l1+l2)/2)
    np.testing.assert_allclose(gradient, (g1+g2)/2, atol=1e-15)
    assert counts['used'] == 2
    _, _, counts = m.loss_gradient(theta, (b, b), (multi, np.ones((1, 3), bool)), unit)
    assert counts['used'] == 1 and counts['no_alternative'] == 1


def test_zero_ties_average_distinct_vectors_and_directional_max_not_claimed_gradient():
    b = bank(); scale = scales(); t = np.zeros(17); direction = np.linspace(-1., 1., 17)
    result = m.score_pair_groups(b, t, scale); vectors = brute_vectors(b, scale)
    eps = 1e-7; moved = m.score_pair_groups(b, eps*direction, scale)
    for p, row in enumerate(vectors):
        for j, values in enumerate(row):
            unique = np.unique(np.array(values)[:, :17], axis=0)
            np.testing.assert_allclose(result.derivatives[p, j], unique.mean(axis=0), atol=1e-15)
            assert moved.scores[p, j]/eps == pytest.approx(np.max(unique @ direction), abs=1e-12)
            # A selected generalized gradient is not a directional derivative.
            assert result.derivatives[p, j] @ direction <= np.max(unique @ direction)+1e-14


def test_near_floating_ties_follow_native_route_order_before_adding_large_base():
    p = person(); e = build_interaction_candidate_evidence(p, objects(1), hoi(copies=2))
    base, local = e.features.copy(), e.hoi_evidence.features.copy()
    base[:, 0] = 1e16
    local[:, 0] = np.tile([1., np.nextafter(1., np.inf), 0., 0.], 2)
    e = replace(e, features=base, hoi_evidence=replace(e.hoi_evidence, features=local))
    b = m.pair_route_bank(p, e); t = np.zeros(17); t[0] = t[6] = 1.
    unit = m.PairScale(np.ones(12), np.ones(12, bool)); result = m.score_pair_groups(b, t, unit)
    native = score_coherent_routes(e, m.masked_model(t, unit))
    assert result.scores[0, 0] == native.scores_a.max()
    # Large base rounds final scores alike, but native max occurs BEFORE base.
    assert result.derivatives[0, 0, 6] == np.nextafter(1., np.inf)
    base[:, 0] = 0.; local[:, 0] = 1e16
    bridge = e.route_features.copy(); bridge[:, 0] = [1., 0., 0., 0.]
    e = replace(e, features=base, route_features=bridge, hoi_evidence=replace(e.hoi_evidence, features=local))
    t[10] = 1.; result = m.score_pair_groups(m.pair_route_bank(p, e), t, unit)
    # Both distinct complete vectors tie after native local+bridge addition.
    assert result.derivatives[0, 0, 10] == .5


def test_a_tie_gradient_does_not_depend_on_raw_logits():
    p = person(); h = hoi(copies=2); e = build_interaction_candidate_evidence(p, objects(), h)
    b = m.pair_route_bank(p, e)
    logits = h.hand_object_logits.copy(); logits[:, 1] = [1., 2., 3., 4.]
    other = m.pair_route_bank(p, build_interaction_candidate_evidence(p, objects(), replace(h, hand_object_logits=logits)))
    a, aa = m.score_pair_groups(b, np.zeros(17), scales()), m.score_pair_groups(other, np.zeros(17), scales())
    np.testing.assert_array_equal(a.scores, aa.scores); np.testing.assert_array_equal(a.derivatives, aa.derivatives)
    assert not np.array_equal(m.score_pair_groups(b, np.zeros(17), scales(), alpha=0.).derivatives,
                              m.score_pair_groups(other, np.zeros(17), scales(), alpha=0.).derivatives)


def test_alias_copies_and_source_permutations_do_not_add_score_gradient_or_scale_mass():
    p, obj, h = person(), objects(), hoi(); b = m.pair_route_bank(p, build_interaction_candidate_evidence(p, obj, h))
    pp = replace(p, person_ids=('p0', 'copy'), boxes_original_xyxy=np.repeat(p.boxes_original_xyxy, 2, axis=0),
        detector_scores=np.repeat(p.detector_scores, 2), keypoints_original_xy=np.repeat(p.keypoints_original_xy, 2, axis=0),
        raw_scores=np.repeat(p.raw_scores, 2, axis=0))
    oo = replace(obj, object_ids=('o1', 'o0', 'copy'), boxes_original_xyxy=obj.boxes_original_xyxy[[1, 0, 0]], raw_scores=np.zeros(3))
    copy = m.pair_route_bank(pp, build_interaction_candidate_evidence(pp, oo, hoi(copies=2)))
    scale, duplicate_scale = m.fit_scales((b,)), m.fit_scales((copy,))
    np.testing.assert_array_equal(scale.variable, duplicate_scale.variable)
    np.testing.assert_array_equal(scale.scale, duplicate_scale.scale)
    for alpha in (None, 0., .3):
        a = m.score_pair_groups(b, np.zeros(17), scales(), alpha=alpha)
        d = m.score_pair_groups(copy, np.zeros(17), scales(), alpha=alpha)
        np.testing.assert_array_equal(a.scores, d.scores); np.testing.assert_array_equal(a.derivatives, d.derivatives)
        assert d.native.scores_a.shape == (2, 2, 3)


def test_fit_scale_equal_image_then_distinct_tuple_mass_and_no_centering():
    b = bank(h=False); e = b.evidence; x = e.features.copy(); ok = e.feature_supported.copy()
    x[:, :6] = np.array([0., 2., 0., 2.])[:, None]; b = replace(b, evidence=replace(e, features=x))
    x2 = x.copy(); x2[:, :6] = 10.; c = replace(b, evidence=replace(e, features=x2))
    scale = m.fit_scales((b, c))
    # Equal image mass: values0,2 receive.25 each, value10 receives.5.
    expected = np.sqrt(.25*(0.-5.5)**2+.25*(2.-5.5)**2+.5*(10.-5.5)**2)
    np.testing.assert_allclose(scale.scale[:6], expected)
    assert not scale.variable[6:].any() and np.all(scale.scale[6:] == 1.)
    model = m.masked_model(np.ones(17), scale)
    np.testing.assert_allclose(model.base_weights[:6], 1/expected)
    assert model.bias == 0. and not hasattr(scale, 'mean')


def test_partial_anatomy_unavailable_contributes_only_explicit_indicator_no_off():
    p = person(); raw = p.raw_scores.copy(); raw[0, [9, 91]] = 0.
    p = replace(p, raw_scores=raw); b = m.pair_route_bank(p, build_interaction_candidate_evidence(p, objects(), hoi()))
    t = np.ones(17); result = m.score_pair_groups(b, t, scales())
    assert result.supported.all() and result.native.supported.all()
    assert np.isnan(b.evidence.features[:2, :5]).all()
    assert np.isfinite(result.scores).all() and not hasattr(result, 'OFF')


def test_absent_routes_is_base_and_relational_derivative_zero():
    b = bank(h=False); t = np.arange(17.)/10
    a, other = m.score_pair_groups(b, t, scales()), m.score_pair_groups(b, t, scales(), alpha=100.)
    np.testing.assert_array_equal(a.scores, other.scores)
    np.testing.assert_array_equal(other.derivatives, 0.)
    assert not other.native.route_supported.any()


@pytest.mark.parametrize('n,o', [(0, 0), (0, 2), (1, 0)])
def test_empty_bank_retained_no_fallback(n, o):
    b = bank(n=n, o=o); result = m.score_pair_groups(b, np.zeros(17), scales())
    assert result.scores.shape == (0 if not n else 1, o) and not result.scores.size
    with pytest.raises(ValueError, match='No informative'):
        m.loss_gradient(np.zeros(17), (b,), (np.zeros(result.scores.shape, bool),), scales())


def test_zero_area_anchor_keeps_nan_group_and_missing_reference_loss_count():
    p = person(); obj = objects(); boxes = obj.boxes_original_xyxy.copy(); boxes[0, 2:] = boxes[0, :2]
    b = m.pair_route_bank(p, build_interaction_candidate_evidence(p, replace(obj, boxes_original_xyxy=boxes), hoi()))
    result = m.score_pair_groups(b, np.zeros(17), scales())
    assert not result.supported[0, 0] and np.isnan(result.scores[0, 0]) and result.scores.shape == (1, 2)
    good = bank(); masks = (np.array([[True, False]]), np.array([[True, False]]))
    _, _, counts = m.loss_gradient(np.zeros(17), (b, good), masks, scales())
    assert counts['missing_positive'] == 1 and counts['used'] == 1


def test_constant_and_allmissing_values_fixed_zero_not_indicator_zeroing():
    b = bank(h=False); scale = m.fit_scales((b,)); assert not scale.variable[6:].any()
    model = m.masked_model(np.ones(17), scale)
    np.testing.assert_array_equal(model.tuple_weights, 0.)
    np.testing.assert_array_equal(model.tuple_availability_weights[[0, 2]], 1.)
    assert not scale.scale.flags.writeable and not scale.variable.flags.writeable


def test_real_small_fixed512_fit_both_arms_frozen_geometry_no_convergence_claim():
    b = bank(h=False); masks = (np.array([[True, False]]),)
    fit = m.fit_coherent_pairs((b,), masks)
    assert fit.iterations_per_arm == 512 and not fit.converged
    assert fit.loss_a_best < fit.loss_a_initial and fit.loss_b_best == fit.loss_b_initial and fit.alpha == 0.
    assert fit.record_counts['used'] == 1 and not fit.theta.flags.writeable
    with pytest.raises(FrozenInstanceError): fit.alpha = 99.
    assert 'reserved' not in inspect.signature(m.fit_coherent_pairs).parameters


def test_optimizer_reuses_postupdate_gradient_exactly_512_updates_each_arm(monkeypatch):
    b = bank(h=False); masks = (np.array([[True, False]]),); original = m.loss_gradient
    calls = []
    def counted(*args, **kwargs):
        calls.append((args[0].copy(), kwargs.get('alpha')))
        return original(*args, **kwargs)
    monkeypatch.setattr(m, 'loss_gradient', counted)
    fit = m.fit_coherent_pairs((b,), masks)
    assert len(calls) == 2*(512+1)
    assert sum(alpha is None for _, alpha in calls) == 513
    for theta, alpha in calls[513:]:
        np.testing.assert_array_equal(theta, fit.theta)
    with pytest.raises(TypeError): fit.record_counts['used'] = 99


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_parameters_rejected_before_scores(value):
    t = np.zeros(17); t[2] = value
    with pytest.raises(ValueError): m.score_pair_groups(bank(), t, scales())


def test_input_bytes_stable_and_exclusive_mapping_validation():
    b = bank(); before = m.core._fingerprint(b.evidence); t = np.arange(17.)
    m.score_pair_groups(b, t, scales(), alpha=.3)
    assert before == m.core._fingerprint(b.evidence)
    np.testing.assert_array_equal(t, np.arange(17.))
    for value in (b.person_boxes, b.object_boxes, b.person_to_group, b.object_to_group):
        with pytest.raises(ValueError): value.flags.writeable = True
    with pytest.raises(ValueError): replace(b, object_to_group=np.zeros(2, np.int64))
    with pytest.raises(ValueError): m.pair_route_bank(replace(person(), person_ids=('wrong',)), b.evidence)
    with pytest.raises(ValueError): m.masked_model(t, scales(), alpha=-1.)

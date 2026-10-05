"""Tiny manufactured reference controls; no FIT, RGB, runtime or held-out labels."""
from dataclasses import replace

import numpy as np
import pytest

from world_reward import coherent_pair_learning as old
from world_reward import coherent_pair_marginal as marginal
from world_reward.coherent_pair_cache import prepare_pair_cache
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_candidate_evidence import GenericObjectObservations, build_interaction_candidate_evidence
from world_reward.person_pose_observations import PersonPoseObservations


def bank(n=2, o=3, copies=2, *, no_hoi=False, missing=False, aliases=False):
    xy = np.full((n, 133, 2), 5., np.float64)
    xy[:, [9, 91, 10, 112]] = [[3., 4.], [4., 4.], [15., 10.], [16., 10.]]
    scores = np.full((n, 133), .7, np.float32)
    if missing and n:
        scores[0, [9, 91, 10, 112]] = 0.
    pb = np.array([[i*2., 0., i*2.+25., 18.] for i in range(n)]).reshape(n, 4)
    if aliases and n:
        pb[:] = [0., 0., 25., 18.]
    person = PersonPoseObservations(2, (20, 30), tuple(f'p{i}' for i in range(n)), pb,
        np.full(n, .9, np.float32), xy, scores)
    ob = np.column_stack((np.arange(o)*5.+5., np.full(o, 3.), np.arange(o)*5.+8., np.full(o, 7.)))
    if aliases and o > 1:
        ob[-1] = ob[0]
    obj = GenericObjectObservations(2, (20, 30), tuple(f'o{i}' for i in range(o)), ob, np.zeros(o))
    h = None
    if not no_hoi:
        boxes = np.tile([[2., 2., 5., 6.], [5., 3., 8., 7.]], (copies, 1)).astype(np.float32)
        count = len(boxes); cls = np.tile([0, 1], copies).astype(np.int64)
        hands, objects = np.flatnonzero(cls == 0), np.flatnonzero(cls == 1)
        pairs = np.column_stack((np.repeat(hands, len(objects)), np.tile(objects, len(hands)))).astype(np.int64)
        score = np.full(count, .5, np.float32); ids = np.arange(count, dtype=np.int64)
        h = HOIDetrObservations(2, (20, 30), np.zeros((1500, 3), np.float32), np.zeros((1500, 4), np.float32),
            np.zeros((1500, 256), np.float32), np.column_stack((boxes, score)).astype(np.float32), ids, ids, ids,
            cls, boxes, score, score, pairs, np.tile([[-2., 1.]], (len(pairs), 1)).astype(np.float32),
            np.empty((0, 2), np.int64), np.empty((0, 2), np.float32))
    return old.pair_route_bank(person, build_interaction_candidate_evidence(person, obj, h))


def reference(b, *, variable=True):
    scale = old.PairScale(np.arange(1., 13.), np.full(12, variable, bool))
    return marginal.MarginalPairReference(prepare_pair_cache(b, scale))


def with_margins(b, margins):
    """Authored native logits remain route-consistent across every person/side/object."""
    e = b.evidence; h = e.hoi_evidence; n, o, k = old._bank(b)
    logits = np.column_stack((np.zeros(k), margins)).astype(np.float64)
    arrays = dict(h.arrays); arrays['hoi_logits'] = np.tile(logits, (n*2, 1))
    routes = dict(e.hoi_routes); routes['raw_logits'] = np.repeat(logits, o, axis=0)
    features = h.features.copy(); features[:, 7] = np.tile(margins, n*2)
    return replace(b, evidence=replace(e, hoi_evidence=replace(h, arrays=arrays, features=features), hoi_routes=routes))


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def score(r, theta=None, *, temperature=.7, alpha=0.):
    if theta is None:
        theta = np.linspace(-.2, .3, 17, dtype=np.float64)
    return marginal.score_pair_marginal(r, theta, temperature=temperature, alpha=alpha)


@pytest.mark.parametrize('alpha', [0., .2, 1.])
def test_all_original_slots_support_identity_and_immutable_arrays(alpha):
    r = reference(bank(aliases=True, missing=True)); before = old.core._fingerprint(r.cache)
    result = score(r, alpha=alpha); n, o, k = r.cache.counts
    assert result.native_scores_a.shape == result.native_scores_b.shape == (n, 2, o)
    assert result.scores_a.shape == (len(r.cache.person_members), len(r.cache.object_members))
    bits(result.native_supported, r.cache.factors['good'])
    for name in ('native_pair_slots', 'native_detection_slots', 'native_query_ids', 'native_retained_nms_positions', 'native_flat_keep'):
        bits(result.identity[name], getattr(r.cache.native_template, name))
    assert result.identity['native_pair_slots'].shape == (k,)
    assert result.identity['source_person_ids'] == r.cache.native_template.source_person_ids
    for name in ('native_scores_a', 'native_scores_b', 'scores_a', 'scores_b', 'geometry_derivatives_a', 'geometry_derivatives_b', 'alpha_derivatives_b'):
        x = getattr(result, name)
        assert not x.flags.writeable
        with pytest.raises(ValueError):
            x.flags.writeable = True
    with pytest.raises(TypeError):
        result.identity['source_person_ids'] = ()
    assert old.core._fingerprint(r.cache) == before


@pytest.mark.parametrize('temperature', [.1, .7, 3.])
def test_alpha_zero_shared_path_bitexact_and_a_independent_of_native_margin(temperature):
    b = bank(); changed = with_margins(b, np.arange(4)*.7-9.)
    original, other = score(reference(b), temperature=temperature), score(reference(changed), temperature=temperature)
    for result in (original, other):
        bits(result.scores_a, result.scores_b); bits(result.native_scores_a, result.native_scores_b)
        bits(result.geometry_derivatives_a, result.geometry_derivatives_b)
    bits(original.scores_a, other.scores_a)
    assert np.any(original.alpha_derivatives_b != other.alpha_derivatives_b)


def test_full_hierarchical_uniform_prior_against_direct_independent_enumeration():
    b = bank(n=1, o=1); e = b.evidence; h = e.hoi_evidence; values = h.features.copy()
    values[:, 0] = np.tile([.1, .1, .3, .3], 2)
    b = replace(b, evidence=replace(e, hoi_evidence=replace(h, features=values)))
    b = with_margins(b, np.array([-2., 3., 4., 4.]))
    r = reference(b); theta = np.linspace(-.2, .3, 17); tau, alpha = .8, .3
    result = score(r, theta, temperature=tau, alpha=alpha)
    mass, all_x, all_m, all_g = [], [], [], []
    sides = [states for states in r.group_sides[0] if states]
    for states in sides:
        for state in states:
            for m in state.margins:
                mass.append(1./len(sides)/len(states)/len(state.margins)); all_x.append(state.values); all_m.append(m)
                all_g.append(float(theta @ state.values)+alpha*m)
    x, m, g, prior = np.array(all_x), np.array(all_m), np.array(all_g), np.array(mass)
    offset = g.max(); weighted = prior*np.exp((g-offset)/tau); posterior = weighted/weighted.sum()
    expected = offset+tau*np.log(weighted.sum())
    np.testing.assert_allclose(result.scores_b[0, 0], expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(result.geometry_derivatives_b[0, 0], posterior @ x, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(result.alpha_derivatives_b[0, 0], posterior @ m, atol=2e-15, rtol=2e-15)
    assert len(sides[0]) == 2 and sorted(len(s.margins) for s in sides[0]) == [1, 2]


@pytest.mark.parametrize('alpha', [0., .37])
def test_fp64_geometry_and_margin_derivatives_finite_differences(alpha):
    r = reference(bank(n=1, o=2, missing=True)); theta = np.linspace(-.2, .3, 17); step = 1e-6
    result = score(r, theta, alpha=alpha)
    for j in range(17):
        delta = np.zeros(17); delta[j] = step
        plus, minus = score(r, theta+delta, alpha=alpha), score(r, theta-delta, alpha=alpha)
        for name, derivative in (('scores_a', result.geometry_derivatives_a), ('scores_b', result.geometry_derivatives_b)):
            numerical = (getattr(plus, name)-getattr(minus, name))/(2*step)
            np.testing.assert_allclose(numerical, derivative[..., j], atol=2e-9, rtol=2e-7)
    if alpha:
        numerical = (score(r, theta, alpha=alpha+step).scores_b-score(r, theta, alpha=alpha-step).scores_b)/(2*step)
    else:
        numerical = (score(r, theta, alpha=step).scores_b-result.scores_b)/step
    np.testing.assert_allclose(numerical, result.alpha_derivatives_b, atol=2e-8, rtol=2e-7)


def test_duplicate_native_routes_and_person_object_aliases_have_no_extra_mass():
    one, many = reference(bank(n=1, o=1, copies=1)), reference(bank(n=3, o=2, copies=3, aliases=True))
    for alpha in (0., .3):
        a, b = score(one, alpha=alpha), score(many, alpha=alpha)
        for name in ('scores_a', 'scores_b', 'geometry_derivatives_a', 'geometry_derivatives_b', 'alpha_derivatives_b'):
            bits(getattr(a, name), getattr(b, name))
    assert len(one.group_sides[0][0][0].native_route_refs) < len(many.group_sides[0][0][0].native_route_refs)
    assert many.cache.counts == (3, 2, 9)


def test_route_permutation_signedzero_and_raw_not_normalized_geometry_key():
    b = bank(n=1, o=1); e = b.evidence; h = e.hoi_evidence; x = h.features.copy()
    x[:, 0] = np.tile([0., -0., .25, .25], 2)
    b = replace(b, evidence=replace(e, hoi_evidence=replace(h, features=x)))
    r = reference(b, variable=False)
    assert len(r.group_sides[0][0]) == 2  # Distinct RAW values survive disabled FIT value scales.
    assert all(np.array_equal(s.values[:12], np.zeros(12)) for s in r.group_sides[0][0])
    order = np.arange(len(x)//2)[::-1]
    values = x.reshape(1, 2, -1, 15)[:, :, order].reshape(x.shape)
    flags = h.feature_supported.reshape(1, 2, -1, 15)[:, :, order].reshape(h.feature_supported.shape)
    other = reference(replace(b, evidence=replace(b.evidence, hoi_evidence=replace(h, features=values, feature_supported=flags))), variable=False)
    for alpha in (0., .3):
        a, z = score(r, alpha=alpha), score(other, alpha=alpha)
        bits(a.scores_a, z.scores_a); bits(a.scores_b, z.scores_b); bits(a.geometry_derivatives_b, z.geometry_derivatives_b)


@pytest.mark.parametrize('n,o', [(0, 0), (0, 2), (1, 0)])
def test_empty_populations_keep_all_native_identity_slots(n, o):
    r = reference(bank(n=n, o=o)); result = score(r)
    assert result.native_scores_a.shape == (n, 2, o)
    assert result.scores_a.shape == (n, o)
    assert result.identity['native_pair_slots'].shape == (4,)
    assert not result.supported.any()
    bits(result.scores_a, result.scores_b)


def test_base_only_no_hoi_missing_joints_and_unsupported_anchor_not_off():
    b = bank(n=1, no_hoi=True, missing=True); r = reference(b); result = score(r, alpha=.7)
    assert result.supported.all() and not result.native_route_supported.any()
    assert all(len(states) == 1 and not states[0].has_route for states in r.group_sides[0])
    assert all(states[0].native_route_refs[0][-1] == -1 for states in r.group_sides[0])
    bits(result.scores_a, result.scores_b); assert np.all(result.alpha_derivatives_b == 0.)
    e = b.evidence; object_boxes = e.objects.boxes_original_xyxy.copy(); object_boxes[0, 2:] = object_boxes[0, :2]
    p = PersonPoseObservations(2, (20, 30), ('p0',), np.array([[0., 0., 25., 18.]]), np.array([.9]),
        np.full((1, 133, 2), 5.), np.zeros((1, 133), np.float32))
    bad = old.pair_route_bank(p, build_interaction_candidate_evidence(p, replace(e.objects, boxes_original_xyxy=object_boxes)))
    output = score(reference(bad)); assert not output.supported[0, 0] and np.isnan(output.scores_a[0, 0])
    assert np.all(output.geometry_derivatives_a[0, 0] == 0.) and np.isnan(output.native_scores_a[:, :, 0]).all()


def test_state_route_presence_and_support_flags_are_part_of_raw_key():
    r = reference(bank(n=1, o=1, missing=True)); c = r.cache
    base = marginal._state(c, 0, 0, 0, -1)[0]; route = marginal._state(c, 0, 0, 0, 0)[0]
    assert base != route and base[-1] == 0 and route[-1] == 1
    assert len(route) == 12*8+12+5+1


def test_equal_supported_side_prior_not_geometry_count_prior_and_absent_side():
    b = bank(n=1, o=1); e = b.evidence; h = e.hoi_evidence; x = h.features.copy()
    x[:4, 0] = [.1, .2, .3, .4]  # Four distinct left geometries, one right geometry.
    b = replace(b, evidence=replace(e, hoi_evidence=replace(h, features=x)))
    r = reference(b); assert [len(states) for states in r.group_sides[0]] == [4, 1]
    theta = np.zeros(17); theta[6] = 1.; tau = .8; result = score(r, theta, temperature=tau)
    side_values = [np.array([state.values @ theta for state in states]) for states in r.group_sides[0]]
    expected = tau*np.log(sum(np.exp(g/tau).mean() for g in side_values)/2.)
    np.testing.assert_allclose(result.scores_a[0, 0], expected, atol=2e-15, rtol=2e-15)
    naive = tau*np.log(np.exp(np.concatenate(side_values)/tau).mean())
    assert abs(expected-naive) > 1e-4
    # The numerical side mixture also handles an absent side without a phantom zero state.
    single = marginal._pair((r.group_sides[0][0], ()), theta, tau, 0.)
    expected_single = tau*np.log(np.exp(side_values[0]/tau).mean())
    np.testing.assert_allclose(single[0], expected_single, atol=2e-15, rtol=2e-15)
    assert single[-1] is True
    # Actual box support must stay consistent across anatomical sides; no invented missing anchor.
    a = dict(b.evidence.arrays); boxes = a['person_boxes_original_xyxy'].copy(); boxes[1, 2:] = boxes[1, :2]
    a['person_boxes_original_xyxy'] = boxes
    bad = replace(b, evidence=replace(b.evidence, arrays=a))
    # Validation keeps each person anchor identical across both sides, so this corruption cannot pass.
    with pytest.raises(ValueError):
        reference(bad)


def test_no_geometry_reweight_from_duplicating_one_of_distinct_native_margins():
    b = bank(n=1, o=1, copies=2)
    b = with_margins(b, np.array([-2., 3., 3., 3.]))
    r = reference(b); assert len(r.group_sides[0][0]) == 1
    state = r.group_sides[0][0][0]
    assert len(state.margins) == 2 and len(state.native_route_refs) == 4
    result = score(r, np.zeros(17), temperature=1., alpha=.5)
    expected = np.log((np.exp(-1.)+np.exp(1.5))/2.)
    np.testing.assert_allclose(result.scores_b[0, 0], expected, atol=2e-15, rtol=2e-15)
    assert abs(result.scores_b[0, 0]-1.5) > .1  # Marginal arithmetic mixture, not max B.


def test_complete_routes_do_not_mix_local_and_bridge_from_different_route_ids():
    b = bank(n=1, o=1); e = b.evidence; h = e.hoi_evidence
    x = h.features.copy(); x[:, 0] = np.tile([0., 10., 0., 10.], 2)
    bridge = e.route_features.copy(); bridge[:, 0] = [10., 0., 10., 0.]
    b = replace(b, evidence=replace(e, route_features=bridge, hoi_evidence=replace(h, features=x)))
    r = reference(b); theta = np.zeros(17); theta[6], theta[10] = r.cache.scales.scale[6], r.cache.scales.scale[10]
    result = score(r, theta, temperature=1.)
    np.testing.assert_allclose(result.scores_a, [[10.]], atol=2e-15, rtol=2e-15)
    assert result.scores_a[0, 0] != 20.  # Independent maxima would invent the incompatible route.


def test_parameters_copied_source_fingerprints_stable_and_overflow_fails():
    r = reference(bank(n=1)); theta = np.linspace(-.2, .3, 17); before = old.core._fingerprint(r)
    result = score(r, theta); saved = result.scores_a.copy(); theta[:] = 123.
    bits(result.scores_a, saved); assert old.core._fingerprint(r) == before
    with pytest.raises((FloatingPointError, ValueError)):
        score(r, np.full(17, np.finfo(np.float64).max))


@pytest.mark.parametrize('field,value', [('temperature', 0.), ('temperature', np.nan), ('temperature', True), ('alpha', -1.), ('alpha', np.inf), ('alpha', True)])
def test_invalid_scalars_and_nonfinite_coefficients_fail_without_repair(field, value):
    r = reference(bank(n=1)); kwargs = dict(temperature=.7, alpha=0.); kwargs[field] = value
    with pytest.raises(ValueError):
        marginal.score_pair_marginal(r, np.zeros(17), **kwargs)
    for theta in (np.zeros(16), np.zeros(17, np.float32), np.full(17, np.nan)):
        with pytest.raises(ValueError):
            marginal.score_pair_marginal(r, theta, temperature=.7)

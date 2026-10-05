"""Manufactured exact equivalence only; no real FIT/reference/runtime inputs."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from world_reward import coherent_pair_cache as cache
from world_reward import coherent_pair_learning as old
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_candidate_evidence import GenericObjectObservations, build_interaction_candidate_evidence
from world_reward.person_pose_observations import PersonPoseObservations


def bank(n=2, o=3, copies=2, *, no_hoi=False, missing=False, aliases=False):
    xy = np.full((n, 133, 2), 5., np.float64)
    xy[:, [9, 91, 10, 112]] = [[3., 4.], [4., 4.], [15., 10.], [16., 10.]]
    scores = np.full((n, 133), .7, np.float32)
    if missing and n: scores[0, [9, 91, 112]] = 0.
    pb = np.array([[i*2., 0., i*2.+25., 18.] for i in range(n)]).reshape(n, 4)
    if aliases and n: pb[:] = [0., 0., 25., 18.]
    person = PersonPoseObservations(2, (20, 30), tuple(f'p{i}' for i in range(n)), pb,
        np.full(n, .9, np.float32), xy, scores)
    ob = np.column_stack((np.arange(o)*5.+5., np.full(o, 3.), np.arange(o)*5.+8., np.full(o, 7.)))
    if aliases and o > 1: ob[-1] = ob[0]
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


def scales(variable=True):
    return old.PairScale(np.arange(1., 13.), np.full(12, variable, bool))


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def identical(a, b):
    for name in ('scores', 'supported', 'derivatives'): bits(getattr(a, name), getattr(b, name))
    for name in a.native.__dataclass_fields__:
        x, y = getattr(a.native, name), getattr(b.native, name)
        if type(x) is np.ndarray: bits(x, y)
        elif name == 'native_identity_available':
            assert x.keys() == y.keys()
            for key in x: bits(x[key], y[key])
        else: assert x == y


@pytest.mark.parametrize('alpha', [None, 0., .25, 3.])
@pytest.mark.parametrize('theta', [np.zeros(17), np.arange(17.)/32., np.linspace(-.71, .84, 17)])
def test_exact_native_group_scores_derivatives_loss_all_blocks(alpha, theta):
    b = bank(missing=True, aliases=True); s = scales(); c = cache.prepare_pair_cache(b, s)
    reference = old.score_pair_groups(b, theta, s, alpha=alpha)
    for block in (1, 2, 128, 3600):
        identical(reference, cache.score_pair_groups(c, theta, alpha=alpha, object_block_size=block))
    positive = np.zeros(reference.scores.shape, bool); positive[0, 0] = True
    a = old.loss_gradient(theta, (b,), (positive,), s, alpha=alpha)
    z = cache.loss_gradient(theta, (c,), (positive,), alpha=alpha)
    assert np.float64(a[0]).tobytes() == np.float64(z[0]).tobytes(); bits(a[1], z[1]); assert a[2] == z[2]


@pytest.mark.parametrize('n,o', [(0, 0), (0, 2), (1, 0)])
def test_zero_population_no_hoi_and_empty_native_pairs(n, o):
    for no_hoi in (False, True):
        b = bank(n, o, no_hoi=no_hoi); s = scales(); c = cache.prepare_pair_cache(b, s)
        for alpha in (None, 0., .25):
            identical(old.score_pair_groups(b, np.zeros(17), s, alpha=alpha),
                      cache.score_pair_groups(c, np.zeros(17), alpha=alpha))
        with pytest.raises(ValueError, match='No informative'):
            cache.loss_gradient(np.zeros(17), (c,), (np.zeros((len(c.person_members), len(c.object_members)), bool),))
    b = bank(n, o, copies=0); c = cache.prepare_pair_cache(b, scales())
    identical(old.score_pair_groups(b, np.zeros(17), scales()), cache.score_pair_groups(c, np.zeros(17)))


def test_near_ties_preserve_max_before_large_base_and_local_bridge_rounding():
    b = bank(n=1, o=1); e = b.evidence; base, local = e.features.copy(), e.hoi_evidence.features.copy()
    base[:, 0] = 1e16; local[:, 0] = np.tile([1., np.nextafter(1., np.inf), 0., -0.], 2)
    b = replace(b, evidence=replace(e, features=base, hoi_evidence=replace(e.hoi_evidence, features=local)))
    t = np.zeros(17); t[0] = t[6] = 1.; unit = old.PairScale(np.ones(12), np.ones(12, bool))
    a = old.score_pair_groups(b, t, unit); z = cache.score_pair_groups(cache.prepare_pair_cache(b, unit), t)
    identical(a, z); assert z.derivatives[0, 0, 6] == np.nextafter(1., np.inf)
    base[:, 0] = -0.; local[:, 0] = 1e16; bridge = e.route_features.copy(); bridge[:, 0] = [1., 0., -0., 0.]
    b = replace(b, evidence=replace(e, features=base, route_features=bridge, hoi_evidence=replace(e.hoi_evidence, features=local)))
    t[10] = 1.
    a = old.score_pair_groups(b, t, unit); z = cache.score_pair_groups(cache.prepare_pair_cache(b, unit), t)
    identical(a, z); assert z.derivatives[0, 0, 10] == .5


def test_distinct_byte_mean_exact_signedzero_duplicates_and_permutations():
    x = np.array([[1., -0., 3.], [-1., 0., 1.], [1., 0., 3.], [1e16, 2., 0.], [-1e16, 2., -0.]])
    x[x == 0.] = 0.
    expected = np.mean(np.stack([v for _, v in sorted({v.tobytes(): v for v in x}.items())]), axis=0)
    for permutation in ([0, 1, 2, 3, 4], [4, 1, 0, 3, 2], [0, 0, 1, 4, 3]):
        bits(expected, cache._distinct_mean([x[permutation[:2]], x[permutation[2:]]]))


def test_independent_scalar_complete_routes_and_exact_distinct_winner_vectors():
    b = bank(n=2, aliases=True); e = b.evidence; n, o, k = old._bank(b)
    unit = old.PairScale(np.ones(12), np.ones(12, bool)); c = cache.prepare_pair_cache(b, unit)
    # Binary-dyadic features keep scalar and vector ufunc arithmetic comparable.
    theta = np.arange(17.)/32.
    for alpha in (None, 0., .25):
        result = cache.score_pair_groups(c, theta, alpha=alpha)
        for p, persons in enumerate(c.person_members):
            for target, objects in enumerate(c.object_members):
                final = []
                for i in persons:
                    for side in range(2):
                        for j in objects:
                            row = (i*2+side)*o+j; base = np.float64(0.)
                            v = np.zeros(18 if alpha is not None else 17)
                            for col in range(10):
                                if col < 6:
                                    x = e.features[row, col] if e.feature_supported[row, col] else 0.
                                    v[col] = x
                                    if theta[col] != 0: base += x*theta[col]
                                if col in (0, 2, 4):
                                    idx = 12+(0, 2, 4).index(col); v[idx] = e.feature_supported[row, col]
                                    if theta[idx] != 0: base += v[idx]*theta[idx]
                            routes = []
                            for r in range(k):
                                hr = (i*2+side)*k+r; h = e.hoi_evidence; local = np.float64(0.); bridge = np.float64(0.); complete = v.copy()
                                for col in range(4):
                                    x = h.features[hr, col] if h.feature_supported[hr, col] else 0.
                                    complete[6+col] = x
                                    if theta[6+col] != 0: local += x*theta[6+col]
                                    if col in (0, 2):
                                        idx = 15+(0, 2).index(col); complete[idx] = h.feature_supported[hr, col]
                                        if theta[idx] != 0: local += complete[idx]*theta[idx]
                                for col in range(2):
                                    x = e.route_features[r*o+j, col] if e.route_supported[r*o+j, col] else 0.
                                    complete[10+col] = x
                                    if theta[10+col] != 0: bridge += x*theta[10+col]
                                score = local+bridge
                                if alpha is not None:
                                    complete[17] = h.features[hr, 7]; score = score+complete[17]*alpha
                                routes.append((score, complete))
                            maximum = max(x[0] for x in routes)
                            final.append((base+maximum, [x[1] for x in routes if x[0] == maximum]))
                best = max(x[0] for x in final)
                vectors = [v for score, vv in final if score == best for v in vv]
                for v in vectors: v[v == 0.] = 0.
                unique = {v.tobytes(): v for v in vectors}
                mean = np.mean(np.stack([unique[key] for key in sorted(unique)]), axis=0)
                assert np.float64(result.scores[p, target]).tobytes() == np.float64(best).tobytes()
                bits(result.derivatives[p, target], mean[17:] if alpha is not None else mean)


def test_original_alias_route_permutations_and_duplicate_rows_add_no_mass():
    b = bank(n=2, o=3, copies=2, aliases=True); e = b.evidence
    # Native detection aliases are unchanged; reverse all Cartesian route rows.
    order = np.arange(e.scope['hoi_native_pairs'])[::-1]
    h = e.hoi_evidence; n, o, k = old._bank(b)
    arrays = dict(h.arrays)
    for name, x in arrays.items():
        if x.shape[:1] == (n*2*k,): arrays[name] = x.reshape(n, 2, k, *x.shape[1:])[:, :, order].reshape(x.shape)
    arrays['native_pair_slots'] = h.arrays['native_pair_slots']
    routes = {name: x.reshape(k, o, *x.shape[1:])[order].reshape(x.shape) for name, x in e.hoi_routes.items()}
    routes['native_pair_slots'] = e.hoi_routes['native_pair_slots']
    other = replace(b, evidence=replace(e, hoi_evidence=replace(h, arrays=arrays,
        features=h.features.reshape(n, 2, k, 15)[:, :, order].reshape(-1, 15),
        feature_supported=h.feature_supported.reshape(n, 2, k, 15)[:, :, order].reshape(-1, 15)),
        hoi_routes=routes, route_features=e.route_features.reshape(k, o, 2)[order].reshape(-1, 2),
        route_supported=e.route_supported.reshape(k, o, 2)[order].reshape(-1, 2)))
    for alpha in (None, 0., .25):
        a = cache.score_pair_groups(cache.prepare_pair_cache(b, scales()), np.zeros(17), alpha=alpha)
        z = cache.score_pair_groups(cache.prepare_pair_cache(other, scales()), np.zeros(17), alpha=alpha)
        for name in ('scores', 'supported', 'derivatives'): bits(getattr(a, name), getattr(z, name))


def test_finite_difference_untied_and_tie_direction_with_independent_enumeration():
    b = bank(n=1); c = cache.prepare_pair_cache(b, scales()); t = np.linspace(-.71, .84, 17)
    positive = np.array([[True, False, False]]); epsilon = 1e-6
    _, gradient, _ = cache.loss_gradient(t, (c,), (positive,))
    numerical = []
    for j in range(17):
        step = np.zeros(17); step[j] = epsilon
        numerical.append((cache.loss_gradient(t+step, (c,), (positive,))[0]-cache.loss_gradient(t-step, (c,), (positive,))[0])/(2*epsilon))
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-7)
    # Independent complete branch coordinates; ties average distinct byte keys,
    # while one-sided moved score takes the max, not the selected mean gradient.
    t = np.zeros(17); d = np.linspace(-1., 1., 17); z = cache.score_pair_groups(c, t)
    moved = cache.score_pair_groups(c, epsilon*d)
    assert np.all((z.derivatives @ d) <= moved.scores/epsilon+1e-12)
    a = .37; _, g, _ = cache.loss_gradient(np.linspace(-.71, .84, 17), (c,), (positive,), alpha=a)
    fd = (cache.loss_gradient(np.linspace(-.71, .84, 17), (c,), (positive,), alpha=a+epsilon)[0]
          -cache.loss_gradient(np.linspace(-.71, .84, 17), (c,), (positive,), alpha=a-epsilon)[0])/(2*epsilon)
    np.testing.assert_allclose(g, [fd], atol=2e-8)


def test_constant_missing_zeroarea_no_routes_and_missed_positive_counts():
    b = bank(n=1, no_hoi=True, missing=True); c = cache.prepare_pair_cache(b, scales(False)); t = np.ones(17)
    for alpha in (None, .25): identical(old.score_pair_groups(b, t, scales(False), alpha=alpha), cache.score_pair_groups(c, t, alpha=alpha))
    e = b.evidence; boxes = e.objects.boxes_original_xyxy.copy(); boxes[0, 2:] = boxes[0, :2]
    obj = replace(e.objects, boxes_original_xyxy=boxes)
    # Rebuild through the original observation constructor rather than repair flags.
    p = PersonPoseObservations(2, (20, 30), ('p0',), np.array([[0., 0., 25., 18.]]), np.array([.9]),
        np.full((1, 133, 2), 5.), np.zeros((1, 133), np.float32))
    bad = old.pair_route_bank(p, build_interaction_candidate_evidence(p, obj))
    cb = cache.prepare_pair_cache(bad, scales(False)); positive = np.array([[True, False, False]])
    a = old.loss_gradient(t, (bad, b), (positive, positive), scales(False))
    z = cache.loss_gradient(t, (cb, c), (positive, positive))
    assert a[0] == z[0]; bits(a[1], z[1]); assert a[2] == z[2] and z[2]['missing_positive'] == 1


def test_exact_equal_image_positive_set_loss_multiple_positives_and_no_alternative():
    b = bank(n=1, copies=1); c = cache.prepare_pair_cache(b, scales()); theta = np.linspace(-.2, .3, 17)
    masks = (np.array([[True, True, False]]), np.array([[False, False, True]]), np.ones((1, 3), bool))
    for alpha in (None, .25):
        a = old.loss_gradient(theta, (b, b, b), masks, scales(), alpha=alpha)
        z = cache.loss_gradient(theta, (c, c, c), masks, alpha=alpha)
        assert np.float64(a[0]).tobytes() == np.float64(z[0]).tobytes(); bits(a[1], z[1]); assert a[2] == z[2]
        assert z[2]['used'] == 2 and z[2]['no_alternative'] == 1


def test_invalid_source_mapping_and_finite_overflow_fail_without_repairs():
    b = bank(); maps = b.object_to_group.copy(); maps[:] = 0
    with pytest.raises(ValueError): cache.prepare_pair_cache(replace(b, object_to_group=maps), scales())
    factors = b.evidence.features.copy(); factors[:, 0] = np.finfo(np.float64).max
    b = replace(b, evidence=replace(b.evidence, features=factors)); c = cache.prepare_pair_cache(b, scales())
    theta = np.zeros(17); theta[0] = np.finfo(np.float64).max
    with pytest.raises((ValueError, FloatingPointError)): old.score_pair_groups(b, theta, scales())
    with pytest.raises((ValueError, FloatingPointError)): cache.score_pair_groups(c, theta)


def test_constructor_authenticates_derived_factors_support_ids_and_source_once():
    b = bank(); c = cache.prepare_pair_cache(b, scales())
    factors = dict(c.factors); factors['base_values'] = factors['base_values']+100.
    with pytest.raises(ValueError,match='must match original source'): replace(c,factors=factors)
    factors = dict(c.factors); factors['good'] = np.zeros_like(factors['good'])
    with pytest.raises(ValueError,match='must match original source'): replace(c,factors=factors)
    factors = dict(c.factors); factors['usable'] = np.zeros_like(factors['usable'])
    with pytest.raises(ValueError,match='must match original source'): replace(c,factors=factors)
    foreign = replace(c.native_template,source_person_ids=('unrelated',),source_evidence_fingerprint='fake')
    with pytest.raises(ValueError,match='native provenance'): replace(c,native_template=foreign)
    with pytest.raises(ValueError,match='native provenance'): replace(c,source_bank_fingerprint='0'*64)
    with pytest.raises(ValueError): replace(c,source_bank=bank(n=1))
    supplied_ids = c.native_template.native_query_ids.copy()
    caller_template = replace(c.native_template,native_query_ids=supplied_ids)
    z = replace(c,native_template=caller_template)
    supplied_ids[:] = 123456
    result = cache.score_pair_groups(z,np.zeros(17))
    bits(result.native.native_query_ids,c.native_template.native_query_ids)
    assert not result.native.native_query_ids.flags.writeable
    with pytest.raises(ValueError): result.native.native_query_ids.flags.writeable = True
    # Replacements that retain identical authenticated contents remain valid.
    identical(cache.score_pair_groups(c,np.zeros(17)),cache.score_pair_groups(replace(c),np.zeros(17)))


def test_cache_byte_backing_source_stability_bad_inputs_and_no_recurring_source_validation(monkeypatch):
    b = bank(); s = scales(); digest = core_fingerprint = old.core._fingerprint(b)
    c = cache.prepare_pair_cache(b, s)
    assert c.source_bank_fingerprint == digest
    assert c.factor_fingerprint == old.core._fingerprint((c.factors, c.scales, c.person_members, c.object_members))
    for x in (*c.factors.values(), *c.person_members, *c.object_members, c.scales.scale, c.scales.variable):
        assert not x.flags.writeable
        with pytest.raises(ValueError): x.flags.writeable = True
    with pytest.raises(TypeError): c.factors['good'] = np.ones(1)
    with pytest.raises(FrozenInstanceError): c.counts = (0, 0, 0)
    factors = dict(c.factors); factors['good'] = factors['good'].astype(np.int64)
    with pytest.raises(ValueError): replace(c, factors=factors)
    monkeypatch.setattr(old.core, '_validate', lambda _: (_ for _ in ()).throw(AssertionError('recurring evidence validation')))
    monkeypatch.setattr(old, '_bank', lambda _: (_ for _ in ()).throw(AssertionError('recurring alias scans')))
    cache.score_pair_groups(c, np.zeros(17))
    assert old.core._fingerprint(b) == core_fingerprint
    for t in (np.full(17, np.nan), np.zeros(16)):
        with pytest.raises(ValueError): cache.score_pair_groups(c, t)
    with pytest.raises(ValueError): cache.score_pair_groups(c, np.zeros(17), alpha=-1.)
    with pytest.raises(ValueError): cache.score_pair_groups(c, np.zeros(17), object_block_size=0)

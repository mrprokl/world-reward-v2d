"""Manufactured banks only: coherence/ABI falsifications, not ownership quality."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
import ast

import numpy as np
import pytest

import world_reward.coherent_route_scorer as module
from world_reward.coherent_route_scorer import CoherentRouteLinear, CoherentRouteMaskedLinear, score_coherent_routes
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_candidate_evidence import GenericObjectObservations, build_interaction_candidate_evidence
from world_reward.person_pose_observations import PersonPoseObservations


def person(n=2):
    xy = np.full((n, 133, 2), 4., np.float64)
    xy[:, [9, 91, 10, 112]] = [[3., 4.], [4., 4.], [15., 10.], [16., 10.]]
    return PersonPoseObservations(7, (20, 30), tuple(f'person-{i}' for i in range(n)),
        np.tile([[0., 0., 25., 18.]], (n, 1)), np.arange(n, dtype=np.float64),
        xy, np.full((n, 133), .7, np.float32))


def objects(n=3):
    boxes = np.column_stack((np.arange(n)+5., np.full(n, 3.), np.arange(n)+8., np.full(n, 7.)))
    return GenericObjectObservations(7, (20, 30), tuple(f'object-{i}' for i in range(n)), boxes,
                                     np.arange(n, dtype=np.float64)-.3)


def hoi(*, copies=1, logits=None):
    box = np.tile([[2., 2., 5., 6.], [5., 3., 8., 7.]], (copies, 1)).astype(np.float32)
    n = len(box);classes = np.tile([0, 1], copies).astype(np.int64)
    hands, targets = np.flatnonzero(classes == 0), np.flatnonzero(classes == 1)
    pairs = np.column_stack((np.repeat(hands, len(targets)), np.tile(targets, len(hands)))).astype(np.int64)
    log = np.tile([[-2., 1.]], (len(pairs), 1)).astype(np.float32) if logits is None else np.asarray(logits, np.float32)
    scores = np.full(n, .5, np.float32)
    return HOIDetrObservations(7, (20, 30), np.zeros((1500, 3), np.float32), np.zeros((1500, 4), np.float32),
        np.zeros((1500, 256), np.float32), np.column_stack((box, scores)).astype(np.float32),
        np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64),
        classes, box, scores, scores, pairs, log, np.empty((0, 2), np.int64), np.empty((0, 2), np.float32))


def weights(*, base=None, local=None, bridge=None, logit=0., bias=0.):
    return CoherentRouteLinear(np.zeros(10) if base is None else np.asarray(base),
        np.zeros(14) if local is None else np.asarray(local), np.zeros(2) if bridge is None else np.asarray(bridge),
        logit, bias)


def evidence(n=2, o=3, h=None):
    return build_interaction_candidate_evidence(person(n), objects(o), h)


def test_no_hoi_keeps_every_person_side_object_base_without_off():
    e = evidence();w = np.zeros(10);w[9] = 2
    r = score_coherent_routes(e, weights(base=w, bias=-1, logit=20))
    assert r.scores_a.shape == r.scores_b.shape == (2, 2, 3)
    np.testing.assert_array_equal(r.scores_a, r.base_scores)
    np.testing.assert_array_equal(r.scores_a, r.scores_b)
    assert r.supported.all() and not r.route_supported.any()
    assert r.source_person_ids == e.source_person_ids and r.source_object_ids == e.objects.object_ids
    assert r.source_observation_references == e.source_observation_references
    assert r.native_pair_slots.shape == (0,) and r.native_detection_slots.shape == (0, 2)
    assert not hasattr(r, 'winner') and not hasattr(r, 'OFF')


@pytest.mark.parametrize('n,o', [(0, 0), (0, 3), (2, 0)])
def test_empty_original_banks_not_fallback(n, o):
    e = evidence(n, o, hoi());r = score_coherent_routes(e, weights())
    assert r.scores_a.shape == (n, 2, o) and not r.supported.size
    assert r.source_person_ids == e.source_person_ids and r.source_object_ids == e.objects.object_ids


@pytest.mark.parametrize('o', [0, 3])
def test_empty_person_native_identities_aligned_unavailable_not_invented(o):
    h = hoi(copies=2);r = score_coherent_routes(evidence(0, o, h), weights())
    np.testing.assert_array_equal(r.native_pair_slots, np.arange(4))
    for field, name in (('native_detection_slots', 'detection_slots'), ('native_query_ids', 'query_ids'),
                        ('native_retained_nms_positions', 'retained_nms_positions'), ('native_flat_keep', 'native_flat_keep')):
        values, available = getattr(r, field), r.native_identity_available[name]
        assert values.shape == available.shape == (4, 2)
        assert not values.flags.writeable and not available.flags.writeable
        if o and name in ('detection_slots', 'query_ids'):
            assert available.all()
            np.testing.assert_array_equal(values, h.hand_object_pairs if name == 'detection_slots' else h.query_ids[h.hand_object_pairs])
        else:
            assert not available.any()
            np.testing.assert_array_equal(values, -1)
    with pytest.raises(TypeError):r.native_identity_available['query_ids'] = np.ones((4, 2), bool)


@pytest.mark.parametrize('name', ['detection_slots', 'query_ids'])
def test_zero_objects_rejects_negative_original_tuple_identity(name):
    e = evidence(2, 0, hoi());a = dict(e.hoi_evidence.arrays);a[name] = a[name].copy();a[name][0, 0] = -1
    with pytest.raises(ValueError, match='Nonnegative original native tuple'):
        score_coherent_routes(replace(e, hoi_evidence=replace(e.hoi_evidence, arrays=MappingProxyType(a))), weights())


def test_logit_added_to_own_route_before_reduction():
    e = evidence(1, 1, hoi(copies=2, logits=[[0., 0.], [0., 10.], [0., -2.], [0., 1.]]))
    local = np.zeros(14);local[0] = -1
    x = e.hoi_evidence.features.copy();x[:, 0] = np.tile([0., 9., 50., 30.], 2)
    e = replace(e, hoi_evidence=replace(e.hoi_evidence, features=x))
    r = score_coherent_routes(e, weights(local=local, logit=1))
    np.testing.assert_array_equal(r.scores_a, 0.)
    np.testing.assert_array_equal(r.scores_b, 1.)  # route1: -9+10, NOT max(logit)+max(geometry)=10
    assert r.route_supported.all()


def test_counterexample_independent_minima_invent_impossible_route():
    e = evidence(1, 1, hoi(copies=2));x = e.hoi_evidence.features.copy();bridge = e.route_features.copy()
    x[:, 0] = np.tile([0., 10., 40., 40.], 2);bridge[:, 1] = [10., 0., 40., 40.]
    e = replace(e, hoi_evidence=replace(e.hoi_evidence, features=x), route_features=bridge)
    local = np.zeros(14);local[0] = -1
    r = score_coherent_routes(e, weights(local=local, bridge=[0., -1.]))
    np.testing.assert_array_equal(r.scores_a, -10.)
    np.testing.assert_array_equal(r.scores_b, -10.)
    assert -min(x[:4, 0])-min(bridge[:, 1]) == 0.  # invalid old independent feature pooling


def test_dense_object_block_equivalence_and_all3600_objects():
    e = evidence(2, 3600, hoi(copies=2));rng = np.random.default_rng(4)
    m = weights(base=rng.normal(size=10), local=rng.normal(size=14), bridge=rng.normal(size=2), logit=.5)
    a = score_coherent_routes(e, m, object_block_size=3600)
    for block in (1, 7, 128, 4000):
        b = score_coherent_routes(e, m, object_block_size=block)
        for key in ('scores_a', 'scores_b', 'supported', 'route_supported'):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    assert a.scores_a.shape == (2, 2, 3600)


def test_duplicate_exact_routes_do_not_add_mass_or_change_original_scores():
    m = weights(local=np.arange(14, dtype=np.float64), bridge=[1., -2.], logit=.3)
    a = score_coherent_routes(evidence(h=hoi()), m)
    b = score_coherent_routes(evidence(h=hoi(copies=2)), m)
    np.testing.assert_array_equal(a.scores_a, b.scores_a);np.testing.assert_array_equal(a.scores_b, b.scores_b)
    assert b.native_pair_slots.tolist() == [0, 1, 2, 3]
    assert b.native_detection_slots.tolist() == [[0, 1], [0, 3], [2, 1], [2, 3]]
    np.testing.assert_array_equal(b.native_retained_nms_positions, b.native_detection_slots)
    np.testing.assert_array_equal(b.native_flat_keep, b.native_detection_slots)


def test_duplicate_object_preserves_original_score_no_identity_fusion():
    p, o, h = person(), objects(), hoi()
    duplicate = replace(o, object_ids=o.object_ids+('object-copy',),
        boxes_original_xyxy=np.concatenate((o.boxes_original_xyxy, o.boxes_original_xyxy[:1])),
        raw_scores=np.r_[o.raw_scores, o.raw_scores[0]])
    m = weights(base=np.arange(10.), local=np.arange(14.), bridge=[1., -1.], logit=.2)
    a = score_coherent_routes(build_interaction_candidate_evidence(p, o, h), m)
    b = score_coherent_routes(build_interaction_candidate_evidence(p, duplicate, h), m)
    np.testing.assert_array_equal(a.scores_a, b.scores_a[:, :, :3]);np.testing.assert_array_equal(a.scores_b, b.scores_b[:, :, :3])
    np.testing.assert_array_equal(b.scores_b[:, :, 0], b.scores_b[:, :, 3])
    assert len(b.source_object_ids) == 4


def test_person_object_side_permutation_equivariance():
    p, o, h = person(), objects(), hoi(copies=2);m = weights(base=np.arange(10.), local=np.arange(14.), bridge=[1., -1.], logit=.2)
    a = score_coherent_routes(build_interaction_candidate_evidence(p, o, h), m)
    xy, scores = p.keypoints_original_xy.copy(), p.raw_scores.copy()
    xy[:, [9, 91, 10, 112]] = xy[:, [10, 112, 9, 91]];scores[:, [9, 91, 10, 112]] = scores[:, [10, 112, 9, 91]]
    pp = replace(p, person_ids=p.person_ids[::-1], boxes_original_xyxy=p.boxes_original_xyxy[::-1],
                 detector_scores=p.detector_scores[::-1], keypoints_original_xy=xy[::-1], raw_scores=scores[::-1])
    order = [2, 0, 1];oo = replace(o, object_ids=tuple(o.object_ids[i] for i in order),
                                boxes_original_xyxy=o.boxes_original_xyxy[order], raw_scores=o.raw_scores[order])
    b = score_coherent_routes(build_interaction_candidate_evidence(pp, oo, h), m)
    np.testing.assert_array_equal(a.scores_a[::-1, ::-1][:, :, order], b.scores_a)
    np.testing.assert_array_equal(a.scores_b[::-1, ::-1][:, :, order], b.scores_b)


def test_missing_active_features_unsupported_not_zero_or_off():
    p = person(1);scores = p.raw_scores.copy();scores[0, 9] = 0.;p = replace(p, raw_scores=scores)
    e = build_interaction_candidate_evidence(p, objects(), hoi());base = np.zeros(10);base[0] = -1
    r = score_coherent_routes(e, weights(base=base, logit=1))
    assert np.isnan(r.scores_a[0, 0]).all() and not r.supported[0, 0].any()
    assert r.supported[0, 1].all() and r.route_supported[0, 1].all()
    local = np.zeros(14);local[0] = -1
    r = score_coherent_routes(e, weights(local=local, logit=1))
    assert r.supported.all() and not r.route_supported[0, 0].any()
    np.testing.assert_array_equal(r.scores_b[0, 0], r.base_scores[0, 0])


def test_source_bytes_model_copies_and_output_seals():
    e = evidence(h=hoi());original = module._fingerprint(e);b = np.arange(10.);m = weights(base=b)
    b[:] = 999.;r = score_coherent_routes(e, m)
    assert module._fingerprint(e) == original == r.source_evidence_fingerprint
    np.testing.assert_array_equal(m.base_weights, np.arange(10.))
    for a in (m.base_weights, m.tuple_weights, m.bridge_weights, r.base_scores, r.scores_a, r.scores_b,
              r.supported, r.route_supported, r.native_pair_slots, r.native_detection_slots, r.native_query_ids):
        assert not a.flags.writeable
        with pytest.raises(ValueError):a.flags.writeable = True
    with pytest.raises(FrozenInstanceError):m.bias = 4.


def test_mutation_during_scoring_fails_without_rewriting_sources(monkeypatch):
    e = evidence(h=hoi());x = e.features.copy();e = replace(e, features=x)
    original = module._linear
    changed = False
    def mutated(values, support, weights):
        nonlocal changed
        answer = original(values, support, weights)
        if not changed:
            x[0, 0] += 1.;changed = True
        return answer
    monkeypatch.setattr(module, '_linear', mutated)
    with pytest.raises(ValueError, match='evidence or parameters changed'):
        score_coherent_routes(e, globals()['weights']())


def test_native_route_permutation_preserves_scores_and_original_ids():
    e = evidence(h=hoi(copies=2));k, o, n = 4, 3, 2;order = [2, 0, 3, 1]
    h = e.hoi_evidence;ha = {}
    for key, value in h.arrays.items():
        if key == 'native_pair_slots':ha[key] = value.copy()
        else:ha[key] = value.reshape(n*2, k, *value.shape[1:])[:, order].reshape(value.shape)
    hh = replace(h, arrays=MappingProxyType(ha),
        features=h.features.reshape(n*2, k, 15)[:, order].reshape(h.features.shape),
        feature_supported=h.feature_supported.reshape(n*2, k, 15)[:, order].reshape(h.feature_supported.shape))
    ra = {}
    for key, value in e.hoi_routes.items():
        if key == 'native_pair_slots':ra[key] = value.copy()
        else:ra[key] = value.reshape(k, o, *value.shape[1:])[order].reshape(value.shape)
    q = replace(e, hoi_evidence=hh, hoi_routes=MappingProxyType(ra),
        route_features=e.route_features.reshape(k, o, 2)[order].reshape(e.route_features.shape),
        route_supported=e.route_supported.reshape(k, o, 2)[order].reshape(e.route_supported.shape))
    m = weights(base=np.arange(10.), local=np.arange(14.), bridge=[1., -2.], logit=.3)
    a, b = score_coherent_routes(e, m), score_coherent_routes(q, m)
    np.testing.assert_array_equal(a.scores_a, b.scores_a);np.testing.assert_array_equal(a.scores_b, b.scores_b)
    np.testing.assert_array_equal(a.native_detection_slots[order], b.native_detection_slots)
    np.testing.assert_array_equal(a.native_flat_keep[order], b.native_flat_keep)


def test_arithmetic_overflow_is_rejected_not_clipped_or_repaired():
    e = evidence();base = np.zeros(10);base[8] = 1e308
    with pytest.raises((ValueError, FloatingPointError)):
        score_coherent_routes(e, weights(base=base, bias=1e308))


@pytest.mark.parametrize('change', [dict(original_frame_index=True), dict(original_frame_index=8),
    dict(image_size=(20, 31)), dict(feature_names=('bad',)), dict(source_person_ids=('same', 'same')),
    dict(features=np.zeros((1, 10))), dict(feature_supported=np.zeros((12, 10), np.int64))])
def test_invalid_frame_grid_features_rejected(change):
    with pytest.raises(ValueError):score_coherent_routes(replace(evidence(h=hoi()), **change), weights())


@pytest.mark.parametrize('field', ['person_slots', 'side_indices', 'generic_object_slots', 'pose_keypoint_indices'])
def test_invalid_candidate_slots_rejected(field):
    e = evidence(h=hoi());a = dict(e.arrays);a[field] = a[field].copy();a[field].flat[0] += 1
    with pytest.raises(ValueError):score_coherent_routes(replace(e, arrays=MappingProxyType(a)), weights())


@pytest.mark.parametrize('field', ['native_pair_slots', 'generic_object_slots', 'detection_slots', 'query_ids', 'raw_logits'])
def test_invalid_bridge_identity_rejected(field):
    e = evidence(h=hoi());a = dict(e.hoi_routes);a[field] = a[field].copy();a[field].flat[0] += 1
    with pytest.raises(ValueError):score_coherent_routes(replace(e, hoi_routes=MappingProxyType(a)), weights())


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_bad_block_size_rejected(value):
    with pytest.raises(ValueError):score_coherent_routes(evidence(), weights(), object_block_size=value)


@pytest.mark.parametrize('field', ['person_boxes_original_xyxy', 'pose_original_xy', 'pose_native_valid', 'object_box_in_original_image'])
def test_invalid_original_array_shape_or_type_rejected(field):
    e = evidence();a = dict(e.arrays);a[field] = np.zeros((1,))
    with pytest.raises(ValueError):score_coherent_routes(replace(e, arrays=MappingProxyType(a)), weights())


@pytest.mark.parametrize('change', [dict(base_weights=np.zeros(9)), dict(tuple_weights=np.zeros(15)),
    dict(bridge_weights=np.array([0., np.nan])), dict(base_weights=np.ma.array(np.zeros(10))),
    dict(logit_weight=np.inf), dict(bias=True)])
def test_bad_model_parameters_rejected(change):
    fields=dict(base_weights=np.zeros(10), tuple_weights=np.zeros(14), bridge_weights=np.zeros(2), logit_weight=0., bias=0.)
    fields.update(change)
    with pytest.raises(ValueError):CoherentRouteLinear(**fields)


def test_no_io_training_thresholds_or_private_labels():
    tree = ast.parse(Path(module.__file__).read_text())
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not {'open', 'exec', 'eval', 'fit', 'select_interacting_actor'} & calls
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(x and ('torch' in x or 'identity_calibration' in x) for x in imports)


def masked(*, base=None, local=None, bridge=None, ba=None, ta=None, ra=None, logit=0., bias=0.):
    raw = weights(base=base, local=local, bridge=bridge, logit=logit, bias=bias)
    return CoherentRouteMaskedLinear(raw.base_weights, raw.tuple_weights, raw.bridge_weights, raw.logit_weight, raw.bias,
        base_availability_weights=np.zeros(10) if ba is None else np.asarray(ba),
        tuple_availability_weights=np.zeros(14) if ta is None else np.asarray(ta),
        bridge_availability_weights=np.zeros(2) if ra is None else np.asarray(ra))


def missing_anatomy(n=1):
    p = person(n); scores = p.raw_scores.copy(); scores[:, [9, 91, 10, 112]] = 0.
    return replace(p, raw_scores=scores)


def test_masked_missing_root_preserves_box_proxy_rawstrict_unchanged():
    p = person(1); scores = p.raw_scores.copy(); scores[:, 91] = 0.; p = replace(p, raw_scores=scores)
    e = build_interaction_candidate_evidence(p, objects(), hoi()); before = module._fingerprint(e)
    w = np.zeros(10); w[2] = -2.; beta = np.zeros(10); beta[2] = 3.
    raw = score_coherent_routes(e, weights(base=w))
    m = score_coherent_routes(e, masked(base=w, ba=beta))
    assert np.isnan(raw.scores_a[0, 0]).all() and not raw.supported[0, 0].any()
    assert m.supported.all() and np.isfinite(m.scores_a).all()
    np.testing.assert_array_equal(m.base_scores[0, 0], 0.)
    expected = e.features.reshape(1, 2, 3, 10)[0, 1, :, 2]*-2.+3.
    np.testing.assert_array_equal(m.base_scores[0, 1], expected)
    assert module._fingerprint(e) == before and np.isnan(e.features[:, 2]).any()


def test_masked_all_missing_anatomy_retains_anchored_proxy_not_anatomic_owner():
    e = build_interaction_candidate_evidence(missing_anatomy(), objects(), hoi())
    m = masked(base=np.ones(10), local=np.ones(14), bridge=np.ones(2), ba=np.ones(10),ta=np.ones(14),ra=np.ones(2),logit=.2)
    r = score_coherent_routes(e,m)
    assert r.supported.all() and r.route_supported.all() and np.isfinite(r.scores_b).all()
    assert not e.anatomical_ownership_verified and not hasattr(r,'owner')


@pytest.mark.parametrize('anchor',['person','object'])
def test_masked_no_positive_base_anchor_unsupported_even_large_availability_bias(anchor):
    p, o = person(1), objects()
    if anchor == 'object':
        boxes=o.boxes_original_xyxy.copy(); boxes[:, 2]=boxes[:, 0]
        o=replace(o,boxes_original_xyxy=boxes)
    e=build_interaction_candidate_evidence(p,o,hoi())
    if anchor == 'person':
        arrays=dict(e.arrays); arrays['person_boxes_original_xyxy']=arrays['person_boxes_original_xyxy'].copy()
        arrays['person_boxes_original_xyxy'][:, 2]=arrays['person_boxes_original_xyxy'][:, 0]
        e=replace(e,arrays=MappingProxyType(arrays),hoi_evidence=None,hoi_routes=MappingProxyType({}),
            route_features=np.empty((0,2)),route_supported=np.empty((0,2),bool),
            source_observation_references=e.source_observation_references[:2],scope=MappingProxyType(dict(e.scope,hoi_native_pairs=0)))
    r=score_coherent_routes(e,masked(ba=np.full(10,100.),ta=np.full(14,100.),ra=np.full(2,100.),bias=1000.,logit=10.))
    assert not r.supported.any() and not r.route_supported.any() and np.isnan(r.scores_a).all() and np.isnan(r.scores_b).all()


@pytest.mark.parametrize('role',[0,1])
def test_masked_zero_area_native_hand_or_direct_box_cannot_form_bias_only_route(role):
    h=hoi(); boxes=h.boxes_original_xyxy.copy(); boxes[role,2]=boxes[role,0]; h=replace(h,boxes_original_xyxy=boxes)
    e=build_interaction_candidate_evidence(missing_anatomy(),objects(),h)
    r=score_coherent_routes(e,masked(ba=np.ones(10),ta=np.ones(14)*100,ra=np.ones(2)*100,logit=10.))
    assert r.supported.all() and not r.route_supported.any()
    np.testing.assert_array_equal(r.scores_a,r.base_scores);np.testing.assert_array_equal(r.scores_b,r.base_scores)


def test_masked_structural_support_independent_of_all_coefficients_and_same_arms():
    e=build_interaction_candidate_evidence(missing_anatomy(),objects(),hoi())
    a=score_coherent_routes(e,masked()); b=score_coherent_routes(e,masked(base=np.arange(10.),local=np.arange(14.),
        bridge=np.arange(2.),ba=np.arange(10.),ta=np.arange(14.),ra=np.arange(2.),logit=-3.,bias=7.))
    np.testing.assert_array_equal(a.supported,b.supported);np.testing.assert_array_equal(a.route_supported,b.route_supported)
    assert a.route_supported.all()


def test_masked_zero_logit_exact_arm_parity_and_absent_hoi_no_fake_off():
    for h in (None,hoi(copies=2)):
        e=build_interaction_candidate_evidence(missing_anatomy(),objects(),h)
        r=score_coherent_routes(e,masked(base=np.arange(10.),local=np.arange(14.),bridge=[2.,-1.],ba=np.arange(10.),
            ta=np.arange(14.),ra=[3.,-2.],logit=0.))
        np.testing.assert_array_equal(r.scores_a,r.scores_b)
        if h is None:np.testing.assert_array_equal(r.scores_a,r.base_scores);assert not r.route_supported.any()


def test_masked_zero_availability_matches_raw_on_fully_available_same_anchors():
    e=evidence(h=hoi()); rng=np.random.default_rng(5)
    params=dict(base=rng.normal(size=10),local=rng.normal(size=14),bridge=rng.normal(size=2),logit=.4,bias=3.)
    a=score_coherent_routes(e,weights(**params));b=score_coherent_routes(e,masked(**params))
    for key in ('base_scores','scores_a','scores_b','supported','route_supported'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


def test_masked_dense_block_equivalence_all3600_and_original_duplicates():
    p=missing_anatomy(2);o=objects(3600);m=masked(base=np.arange(10.),local=np.arange(14.),bridge=[1.,-2.],
        ba=np.arange(10.),ta=np.arange(14.),ra=[2.,-1.],logit=.3)
    e=build_interaction_candidate_evidence(p,o,hoi());a=score_coherent_routes(e,m,object_block_size=3600)
    for block in (1,7,128,4000):
        b=score_coherent_routes(e,m,object_block_size=block)
        for key in ('scores_a','scores_b','supported','route_supported'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    duplicate=score_coherent_routes(build_interaction_candidate_evidence(p,o,hoi(copies=2)),m)
    np.testing.assert_array_equal(a.scores_b,duplicate.scores_b)
    oo=replace(o,object_ids=o.object_ids+('copy',),boxes_original_xyxy=np.concatenate((o.boxes_original_xyxy,o.boxes_original_xyxy[:1])),
        raw_scores=np.r_[o.raw_scores,o.raw_scores[0]])
    b=score_coherent_routes(build_interaction_candidate_evidence(p,oo,hoi()),m)
    np.testing.assert_array_equal(a.scores_b,b.scores_b[:,:,:3600]);np.testing.assert_array_equal(b.scores_b[:,:,0],b.scores_b[:,:,-1])


def test_masked_person_object_permutation_and_no_visibility_filter():
    p,o=missing_anatomy(2),objects();boxes=o.boxes_original_xyxy.copy();boxes[0]=[-30.,-20.,-10.,-5.]
    o=replace(o,boxes_original_xyxy=boxes);order=[2,0,1]
    pp=replace(p,person_ids=p.person_ids[::-1],boxes_original_xyxy=p.boxes_original_xyxy[::-1],detector_scores=p.detector_scores[::-1],
        keypoints_original_xy=p.keypoints_original_xy[::-1],raw_scores=p.raw_scores[::-1])
    oo=replace(o,object_ids=tuple(o.object_ids[i] for i in order),boxes_original_xyxy=o.boxes_original_xyxy[order],raw_scores=o.raw_scores[order])
    m=masked(base=np.arange(10.),local=np.arange(14.),bridge=[1.,-2.],ba=np.arange(10.),ta=np.arange(14.),ra=[1.,2.],logit=.4)
    a=score_coherent_routes(build_interaction_candidate_evidence(p,o,hoi()),m)
    b=score_coherent_routes(build_interaction_candidate_evidence(pp,oo,hoi()),m)
    np.testing.assert_array_equal(a.scores_b[::-1,:,order],b.scores_b)
    assert a.supported[:,:,0].all()  # finite positive off-grid object remains, not claimed visible


@pytest.mark.parametrize('n,o',[(0,0),(0,3),(2,0)])
def test_masked_empty_banks_keep_original_slots_and_unavailable_identity_flags(n,o):
    e=evidence(n,o,hoi()); r=score_coherent_routes(e,masked(ba=np.ones(10),ta=np.ones(14),ra=np.ones(2)))
    assert r.scores_a.shape==(n,2,o) and not r.supported.size and r.native_pair_slots.shape==(1,)


def test_masked_anchors_validate_original_flag_box_coherence_without_repair():
    e=evidence(h=hoi());arrays=dict(e.arrays);arrays['object_box_positive_area']=~arrays['object_box_positive_area']
    with pytest.raises(ValueError,match='positive-area'):score_coherent_routes(replace(e,arrays=MappingProxyType(arrays)),masked())
    h=e.hoi_evidence;arrays=dict(h.arrays);arrays['hoi_box_positive_area']=~arrays['hoi_box_positive_area']
    with pytest.raises(ValueError,match='positive-area'):score_coherent_routes(replace(e,hoi_evidence=replace(h,arrays=MappingProxyType(arrays))),masked())


@pytest.mark.parametrize('name,n',[('base_availability_weights',10),('tuple_availability_weights',14),('bridge_availability_weights',2)])
def test_masked_availability_coefficients_are_finite_copied_and_sealed(name,n):
    params=dict(base_availability_weights=np.zeros(10),tuple_availability_weights=np.zeros(14),bridge_availability_weights=np.zeros(2))
    values=np.arange(n,dtype=np.float64);params[name]=values
    m=CoherentRouteMaskedLinear(np.zeros(10),np.zeros(14),np.zeros(2),0.,**params);values[:]=999.
    np.testing.assert_array_equal(getattr(m,name),np.arange(n));assert not getattr(m,name).flags.writeable
    with pytest.raises(ValueError):getattr(m,name).flags.writeable=True
    params[name]=np.full(n,np.nan)
    with pytest.raises(ValueError):CoherentRouteMaskedLinear(np.zeros(10),np.zeros(14),np.zeros(2),0.,**params)
    params[name]=np.zeros(n+1)
    with pytest.raises(ValueError):CoherentRouteMaskedLinear(np.zeros(10),np.zeros(14),np.zeros(2),0.,**params)

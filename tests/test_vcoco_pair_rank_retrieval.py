"""Manufactured role/box arrays only; no actual annotations or prediction banks."""
from copy import deepcopy
from itertools import combinations
import math

import numpy as np
import pytest

from world_reward import vcoco_pair_rank_retrieval as m
from world_reward import vcoco_pair_retrieval as old
from world_reward.vcoco_role_reference import parse_vcoco_role_reference


def fixture():
    images = [dict(id=501, width=100, height=80)]
    instances = [dict(id=11, image_id=501, category_id=1, iscrowd=0, bbox=[0, 0, 10, 20], area=200),
                 dict(id=12, image_id=501, category_id=1, iscrowd=0, bbox=[60, 0, 10, 20], area=200),
                 dict(id=31, image_id=501, category_id=17, iscrowd=0, bbox=[10, 30, 10, 10], area=100),
                 dict(id=32, image_id=501, category_id=44, iscrowd=0, bbox=[60, 30, 10, 10], area=100),
                 dict(id=40, image_id=501, category_id=7, iscrowd=0, bbox=[80, 50, 5, 5], area=25)]
    actions = [dict(action_name='role_A', role_name=['agent', 'obj', 'instr'], ann_id=[11, 12], label=[1, 1],
                    image_id=[501, 501], role_object_id=[11, 12, 31, 32, 0, 0]),
               dict(action_name='duplicate_role', role_name=['agent', 'obj'], ann_id=[11], label=[1],
                    image_id=[501], role_object_id=[11, 31])]
    person = np.array([[0, 0, 10, 20], [60, 0, 70, 20]], np.float64)
    obj = np.array([[10, 30, 20, 40], [60, 30, 70, 40], [80, 50, 85, 55]], np.float64)
    score = np.array([[5., -3., -8.], [-4., 2., -7.]], np.float64)
    return actions, instances, images, person, obj, score


def evaluate(f=None, *, support=None):
    actions, instances, images, person, obj, score = fixture() if f is None else f
    return m.evaluate_vcoco_pair_rank_retrieval(person, obj, score,
        np.ones(score.shape, bool) if support is None else support,
        parse_vcoco_role_reference(actions, instances, images), instances, 501, (80, 100),
        tuple(f'p{i}' for i in range(len(person))), tuple(f'o{i}' for i in range(len(obj))))


def test_all_group_positive_mask_duplicate_actions_and_rank_views():
    f = fixture(); before = deepcopy(f[:3]); raw = [a.tobytes() for a in f[3:]]
    r = evaluate(f)
    assert r.budgets == (1, 3, 5) and r.native_positive_pairs == ((11, 31), (12, 32))
    assert r.positive_mask.tolist() == [[True, False, False], [False, True, False]]
    assert r.recalls == (.5, 1., 1.) and r.expected_positive_hits == (1., 2., 2.)
    assert r.any_positive_at1 == r.validation.image_positive_retrieval == 1.
    assert r.status == 'scorable_localized_positive' and f[:3] == before and raw == [a.tobytes() for a in f[3:]]
    assert r.validation.role_diagnostics['missing_positive_role_slots'] == 2
    assert not r.scope['unknown_pairs_verified_negative'] and not r.scope['source_authenticated']


def test_uniform_tie_probabilities_without_id_priorities():
    f = list(fixture()); f[-1] = np.zeros((2, 3), np.float64)
    r = evaluate(tuple(f))
    np.testing.assert_allclose(r.positive_hit_probabilities, [[1/6, 1/6], [.5, .5], [5/6, 5/6]], rtol=0, atol=3e-16)
    np.testing.assert_allclose(r.recalls, [1/6, .5, 5/6], rtol=0, atol=3e-16)
    assert r.any_positive_at1 == r.validation.image_positive_retrieval == 2/6
    f[-1][0, 2] = np.nextafter(0., 1.)
    r = evaluate(tuple(f)); assert r.any_positive_at1 == r.recalls[0] == 0.


def test_multiple_distinct_proposal_geometries_same_native_gt_hit_only_once():
    a, i, im, p, o, s = fixture()
    o = np.vstack((o, [10, 30, 19, 40]))
    s = np.array([[10., 0., 0., 9.], [0., 8., 0., 0.]], np.float64)
    r = evaluate((a, i, im, p, o, s))
    assert r.positive_mask.sum() == 3 and r.native_positive_pairs == ((11, 31), (12, 32))
    assert r.expected_positive_hits == (1., 2., 2.) and r.recalls == (.5, 1., 1.)
    s[:] = 0.; r = evaluate((a, i, im, p, o, s))
    assert r.any_positive_at1 == 3/8
    expected_a = 1-math.comb(8-2, 3)/math.comb(8, 3)
    expected_b = 3/8
    assert r.expected_positive_hits[1] == pytest.approx(expected_a+expected_b, abs=1e-15)
    assert r.recalls[1] == pytest.approx((expected_a+expected_b)/2, abs=1e-15)


def test_above_cut_gt_does_not_gain_mass_from_same_gt_cutoff_tie():
    a, i, im, p, o, s = fixture(); o = np.vstack((o, [10, 30, 19, 40]))
    s = np.zeros((2, 4), np.float64); s[0, 0] = 1.
    r = evaluate((a, i, im, p, o, s))
    assert r.positive_hit_probabilities[:, 0].tolist() == [1., 1., 1.]
    assert r.positive_hit_probabilities[1, 1] == pytest.approx(2/7)
    assert r.expected_positive_hits[1] == pytest.approx(1+2/7)


def test_complete_rank_expectation_against_explicit_small_tie_selections():
    a, i, im, p, o, _ = fixture(); o = np.vstack((o, [10, 30, 19, 40]))
    # Independent original proposal -> native positive association, not metric helpers.
    association = np.array([0, -1, -1, 0, -1, 1, -1, -1])
    for values in (np.zeros(8), np.array([2., 1., 1., 1., 0., 1., 0., -1.]),
                   np.array([0., 2., 2., 1., 0., 1., 2., 0.])):
        r = evaluate((a, i, im, p, o, values.reshape(2, 4)))
        for j, budget in enumerate(m.BUDGETS):
            cut = sorted(values, reverse=True)[budget-1]
            above = {int(v) for v in np.flatnonzero(values > cut)}
            tied = tuple(int(v) for v in np.flatnonzero(values == cut))
            samples = tuple(combinations(tied, budget-len(above)))
            hits = [len({int(association[v]) for v in above | set(sample) if association[v] >= 0})
                    for sample in samples]
            assert r.expected_positive_hits[j] == pytest.approx(sum(hits)/len(hits), rel=0, abs=3e-16)


def test_coordinate_alias_duplication_and_permutation_invariant_canonical_maps():
    a, i, im, p, o, s = fixture()
    base = evaluate((a, i, im, p, o, np.zeros_like(s)))
    pp, oo = [1, 0, 0], [2, 1, 0, 0]
    r = evaluate((a, i, im, p[pp], o[oo], np.zeros((3, 4), np.float64)))
    assert r.validation.raw_proposal_counts == (3, 4) and r.validation.unique_proposal_counts == (2, 3)
    assert r.person_to_group.tolist() == [1, 0, 0] and r.object_to_group.tolist() == [2, 1, 0, 0]
    assert [v.tolist() for v in r.person_members] == [[1, 2], [0]]
    assert [v.tolist() for v in r.object_members] == [[2, 3], [1], [0]]
    for name in ('person_boxes', 'object_boxes', 'positive_mask', 'scores', 'supported', 'positive_hit_probabilities'):
        assert np.array_equal(getattr(r, name), getattr(base, name), equal_nan=True)
    assert r.recalls == base.recalls and r.any_positive_at1 == base.any_positive_at1


@pytest.mark.parametrize('fault', ['score', 'support'])
def test_original_alias_validator_not_replaced_by_max_or_mean(fault):
    a, i, im, p, o, s = fixture(); p = np.vstack((p, p[:1])); s = s[[0, 1, 0]].copy()
    support = np.ones(s.shape, bool)
    if fault == 'score': s[2, 1] = 99.
    else: support[2, 1] = False; s[2, 1] = np.nan
    with pytest.raises(ValueError, match='aliases'): evaluate((a, i, im, p, o, s), support=support)


@pytest.mark.parametrize('which', ['person', 'object', 'both', 'unsupported'])
def test_empty_unsupported_and_missing_proposals_keep_reference_denominator(which):
    a, i, im, p, o, s = fixture()
    if which in ('person', 'both'): p = np.empty((0, 4), np.float64)
    if which in ('object', 'both'): o = np.empty((0, 4), np.float64)
    support = np.zeros((len(p), len(o)), bool); s = np.full(support.shape, np.nan)
    r = evaluate((a, i, im, p, o, s), support=support)
    assert r.recalls == (0., 0., 0.) and r.any_positive_at1 == 0.
    assert r.positive_mask.shape == (len(p), len(o)) and len(r.native_positive_pairs) == 2
    if which == 'unsupported': assert r.positive_mask.sum() == 2


def test_partial_support_preserves_true_positive_mask_and_miss():
    f = list(fixture()); support = np.ones((2, 3), bool); support[0, 0] = False; f[-1][0, 0] = np.nan
    r = evaluate(tuple(f), support=support)
    assert r.positive_mask.sum() == 2 and r.supported.sum() == 5
    assert r.positive_mask[0, 0] and not r.supported[0, 0]
    assert r.recalls == (.5, .5, .5)


@pytest.mark.parametrize('kind', ['crowd', 'wrong_class', 'same_class', 'context'])
def test_all_context_matching_censors_ambiguous_person_or_object(kind):
    a, i, im, p, o, s = fixture(); row = deepcopy(i[2]); row['id'] = 99
    if kind == 'crowd': row['iscrowd'] = 1
    elif kind == 'wrong_class': row['category_id'] = 1
    elif kind == 'same_class': row = deepcopy(i[0]); row['id'] = 99
    i.append(row); r = evaluate((a, i, im, p, o, s))
    assert r.positive_mask.sum() == 1 and r.recalls == (0., .5, .5)
    assert r.any_positive_at1 == r.validation.image_positive_retrieval == 0.


def test_no_localized_reference_returns_none_not_drop_image_or_fake_zero_truth():
    a, i, im, p, o, s = fixture()
    for action in a:
        action['role_object_id'] = [*action['ann_id'], *([0]*(len(action['role_object_id'])-len(action['ann_id'])))]
    r = evaluate((a, i, im, p, o, s))
    assert r.status == 'no_localized_positive' and r.recalls == (None, None, None)
    assert r.expected_positive_hits == (None, None, None) and r.any_positive_at1 is None
    assert not r.positive_mask.any() and r.positive_hit_probabilities.shape == (3, 0)
    assert r.validation.role_diagnostics['missing_positive_role_slots'] == 5


def test_mask_is_independent_scores_and_support_but_not_fake_negative_truth():
    f = list(fixture()); base = evaluate(tuple(f)); f[-1] = np.full((2, 3), np.nan)
    changed = evaluate(tuple(f), support=np.zeros((2, 3), bool))
    assert np.array_equal(changed.positive_mask, base.positive_mask) and changed.recalls == (0., 0., 0.)
    assert not changed.scope['unknown_pairs_verified_negative']


def test_full3600_unique_candidates_unknown_rank_winners_not_deleted():
    a, i, im, p, o, s = fixture(); x = np.arange(3597, dtype=np.float64)+1000
    o = np.vstack((o, np.column_stack((x, x, x+1, x+1))))
    s = np.zeros((2, 3600), np.float64); s[:, -3:] = 9.
    r = evaluate((a, i, im, p, o, s))
    assert r.positive_mask.shape == (2, 3600) and r.positive_mask.sum() == 2
    assert r.validation.raw_proposal_counts == r.validation.unique_proposal_counts == (2, 3600)
    assert r.recalls == (0., 0., 0.)


def test_immutable_outputs_and_no_input_or_reference_mutation():
    f = fixture(); raw = [a.tobytes() for a in f[3:]]; before = deepcopy(f[:3]); r = evaluate(f)
    for a in (r.positive_mask, r.scores, r.supported, r.person_boxes, r.object_boxes, r.person_to_group,
              r.object_to_group, r.positive_hit_probabilities, *r.person_members, *r.object_members):
        with pytest.raises(ValueError): a.flags.writeable = True
    with pytest.raises(TypeError): r.scope['ownership_verified'] = True
    assert raw == [a.tobytes() for a in f[3:]] and f[:3] == before


def test_wrapper_reuses_original_validation_and_detects_later_input_mutation(monkeypatch):
    f = fixture(); original_unique = m.np.unique; changed = [False]
    def unique(*args, **kwargs):
        value = original_unique(*args, **kwargs)
        if not changed[0]: f[-1][0, 0] = 99.; changed[0] = True
        return value
    monkeypatch.setattr(m.np, 'unique', unique)
    with pytest.raises(ValueError, match='mutated'): evaluate(f)


@pytest.mark.parametrize('n', range(1, 11))
def test_cutoff_formula_against_all_tiny_subsets(n):
    for positive in range(n+1):
        for draws in range(min(n, 5)+1):
            subsets = tuple(combinations(range(n), draws))
            exact = sum(any(v < positive for v in selected) for selected in subsets)/len(subsets)
            assert m._hit_probability(n, positive, draws) == pytest.approx(exact, rel=0, abs=3e-16)


def test_bounded_product_large_tie_not_unstable_comb_overflow():
    n = 1_000_000_000_000
    assert m._hit_probability(n, 1, 5) == pytest.approx(5/n, rel=1e-15)
    for args in ((0, 1, 0), (3, -1, 1), (3, 2, 4), (10, 2, 6), (True, 1, 1)):
        with pytest.raises(ValueError): m._hit_probability(*args)


@pytest.mark.parametrize('fault', ['infinite', 'wrong_mask', 'other_image', 'missing_context', 'inverted', 'duplicate_ids'])
def test_invalid_inputs_fail_via_unchanged_evaluator(fault):
    a, i, im, p, o, s = fixture(); support = np.ones(s.shape, bool)
    if fault == 'infinite': s[0, 0] = np.inf
    elif fault == 'wrong_mask': support = support.astype(np.int64)
    elif fault == 'other_image': i[0]['image_id'] = 999
    elif fault == 'missing_context': i.pop(2)
    elif fault == 'inverted': p[0, 2] = -1
    reference = parse_vcoco_role_reference(a, i, im) if fault not in ('other_image', 'missing_context') else parse_vcoco_role_reference(a, fixture()[1], im)
    ids = ('p', 'p') if fault == 'duplicate_ids' else ('p0', 'p1')
    with pytest.raises(ValueError):
        m.evaluate_vcoco_pair_rank_retrieval(p, o, s, support, reference, i, 501, (80, 100), ids, ('o0', 'o1', 'o2'))

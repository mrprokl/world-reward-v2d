"""Manufactured contact primitive contracts, not challenge quality validation."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from world_reward.contact_patch import (ContactPatchConfig, frozen_interval_candidates,
                                       smooth_patch_distances)


def fixture(count=6, candidates=3):
    points = np.arange(count*2*candidates*3, dtype=float).reshape(count, 2, candidates, 3)/100
    ids = np.vstack((np.arange(candidates)+100, np.arange(candidates)+200))
    distances = np.broadcast_to(np.arange(candidates, dtype=float)[None, None]/100,
                               points.shape[:-1]).copy()
    active = np.zeros((count, 2), bool); active[:, 0] = True
    supported = np.ones(points.shape[:-1], bool)
    return points, ids, distances, active, supported


def test_config_is_frozen_and_requires_external_reference():
    config = ContactPatchConfig(.01, 8, 'manufactured:nonchallenge_external_dev')
    with pytest.raises(FrozenInstanceError):
        config.max_candidates = 1
    for args in ((0., 8, 'dev'), (np.inf, 8, 'dev'), (.01, 9, 'dev'),
                 (.01, True, 'dev'), (.01, 1, ' ')):
        with pytest.raises(ValueError):
            ContactPatchConfig(*args)


def test_interval_pool_keeps_ids_not_positions_when_nearest_witness_changes():
    f = fixture(candidates=2)
    f[2][::2, 0] = [.001, .02]; f[2][1::2, 0] = [.02, .001]
    ids, points, visible = frozen_interval_candidates(*f, 2)
    np.testing.assert_array_equal(ids[:, 0], np.tile([100, 101], (6, 1)))
    np.testing.assert_array_equal(points[:, 0], f[0][:, 0])
    assert not np.array_equal(points[0, 0], points[-1, 0])
    assert visible[:, 0].all()
    assert (ids[:, 1] == -1).all() and np.isnan(points[:, 1]).all()


def test_inactive_gap_starts_an_independent_pool():
    f = fixture()
    f[3][2:4, 0] = False
    f[2][4:, 0] = [.3, .2, .01]
    ids, points, visible = frozen_interval_candidates(*f, 1)
    np.testing.assert_array_equal(ids[:, 0, 0], [100, 100, -1, -1, 102, 102])
    assert np.isnan(points[2:4]).all() and not visible[2:4].any()


def test_source_column_permutation_and_equal_score_ties_are_deterministic():
    f = fixture()
    f[2][:] = .01
    original = frozen_interval_candidates(*f, 2)
    permutation = [2, 0, 1]
    reordered = frozen_interval_candidates(f[0][:, :, permutation], f[1][:, permutation],
        f[2][:, :, permutation], f[3], f[4][:, :, permutation], 2)
    for actual, expected in zip(reordered, original):
        np.testing.assert_array_equal(actual, expected)


def test_unsupported_geometry_stays_nan_and_known_id_is_not_erased():
    f = fixture(candidates=2)
    f[4][1, 0, 0] = False; f[0][1, 0, 0] = np.nan; f[2][1, 0, 0] = np.nan
    ids, points, visible = frozen_interval_candidates(*f, 2)
    assert ids[1, 0, 0] == 100 and not visible[1, 0, 0]
    assert np.isnan(points[1, 0, 0]).all() and visible[1, 0, 1]


def test_bounded_pool_coverage_failure_cannot_self_disable_an_active_contact():
    f = fixture(candidates=2)
    f[4][1, 0, 0] = False; f[0][1, 0, 0] = np.nan; f[2][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        frozen_interval_candidates(*f, 1)


def test_small_pool_and_no_contact_are_explicit():
    f = fixture(candidates=1)
    assert frozen_interval_candidates(*f, 8)[0].shape == (6, 2, 1)
    f[3][:] = False
    ids, points, visible = frozen_interval_candidates(*f, 8)
    assert (ids == -1).all() and np.isnan(points).all() and not visible.any()


def test_candidate_outputs_have_immutable_owned_storage():
    f = fixture()
    expected = frozen_interval_candidates(*f, 2)
    snapshot = tuple(value.copy() for value in expected)
    f[0][:] = 100.; f[1][:] = 1000; f[2][:] = 100.; f[3][:] = False; f[4][:] = False
    for value, copy in zip(expected, snapshot):
        np.testing.assert_array_equal(value, copy)
        with pytest.raises(ValueError):
            value.setflags(write=True)


@pytest.mark.parametrize('field', range(5))
def test_masked_anatomical_inputs_rejected(field):
    f = list(fixture()); f[field] = np.ma.array(f[field])
    with pytest.raises(ValueError):
        frozen_interval_candidates(*f, 2)


@pytest.mark.parametrize('mutation', ['shape', 'duplicate', 'negative', 'nan_supported',
    'finite_unsupported', 'bad_distance', 'missing_active', 'unsigned_overflow', 'float_active'])
def test_malformed_anatomical_inputs_rejected(mutation):
    f = list(fixture())
    if mutation == 'shape': f[0] = f[0][..., :2]
    if mutation == 'duplicate': f[1][0, 1] = f[1][0, 0]
    if mutation == 'negative': f[1][0, 0] = -1
    if mutation == 'nan_supported': f[0][0, 0, 0] = np.nan
    if mutation == 'finite_unsupported': f[4][0, 0, 0] = False
    if mutation == 'bad_distance': f[2][0, 0, 0] = -.01
    if mutation == 'missing_active':
        f[4][0, 0] = False; f[0][0, 0] = np.nan; f[2][0, 0] = np.nan
    if mutation == 'unsigned_overflow': f[1] = f[1].astype(np.uint64); f[1][0, 0] = 2**63
    if mutation == 'float_active': f[3] = f[3].astype(float)
    with pytest.raises(ValueError):
        frozen_interval_candidates(*f, 2)


def test_singletons_are_exact_legacy_distances_even_at_extreme_scales():
    distances = np.array([0., 1e-300, .02, 1e300])
    result = smooth_patch_distances(distances, np.arange(4), 4, .01)
    np.testing.assert_array_equal(result, distances)
    assert not result.flags.writeable


def test_soft_patch_matches_independent_direct_formula_and_interleaved_groups():
    d = np.array([.01, .02, .03, .04, .05]); groups = np.array([1, 0, 1, 0, 1])
    result = smooth_patch_distances(d, groups, 2, .025)
    expected = []
    for group in range(2):
        values = d[groups == group]; weights = np.exp(-values**2/.025**2)
        expected.append(np.sqrt(np.dot(weights, values**2)/weights.sum()))
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    order = [4, 1, 0, 3, 2]
    np.testing.assert_allclose(smooth_patch_distances(d[order], groups[order], 2, .025), result)


def test_soft_patch_continuity_and_derivative_symmetry_when_minimum_changes():
    def call(x):
        return smooth_patch_distances(np.array([.02+x, .02-x]), np.zeros(2, np.int64), 1, .01)[0]
    h = 1e-6
    assert call(h) == pytest.approx(call(-h), abs=1e-15)
    assert abs(call(h)-call(0)) < 1e-8
    assert abs((call(h)-call(-h))/(2*h)) < 1e-8


def test_real_tangential_slide_changes_positions_without_contact_identity_freeze():
    f = fixture(candidates=2)
    f[0][:, 0, :, 0] = np.arange(6)[:, None]/10
    f[0][:, 0, :, 2] = [.0, .01]
    f[2][:, 0] = np.abs(f[0][:, 0, :, 2])  # Independent planar surface fixture.
    ids, points, visible = frozen_interval_candidates(*f, 2)
    distances = np.abs(points[:, 0, :, 2]).reshape(-1)
    residual = smooth_patch_distances(distances, np.repeat(np.arange(6), 2), 6, .002)
    assert visible[:, 0].all() and np.all(np.diff(points[:, 0, 0, 0]) > 0)
    np.testing.assert_allclose(residual, residual[0], atol=1e-15)
    assert (ids[:, 0, 0] == 100).all()  # ID persistence does not freeze its trajectory.


def test_far_candidate_negligible_not_all_points_forced_to_touch():
    residual = smooth_patch_distances(np.array([.01, 1.]), np.zeros(2, np.int64), 1, .02)[0]
    assert residual == pytest.approx(.01, rel=1e-12)
    assert residual < np.sqrt((.01**2+1.)/2)/10


def test_true_contact_bias_is_positive_not_hidden_or_claimed_zero():
    residual = smooth_patch_distances(np.array([0., .01]), np.zeros(2, np.int64), 1, .01)[0]
    expected = .01*np.sqrt(np.exp(-1)/(1+np.exp(-1)))
    assert residual > 0
    assert residual == pytest.approx(expected, rel=1e-14)


def test_supported_positive_near_distance_is_not_zeroed_by_a_huge_far_candidate():
    residual = smooth_patch_distances(np.array([1e-200, 1e200]), np.zeros(2, np.int64), 1, .01)[0]
    assert residual > 0
    assert residual == pytest.approx(1e-200, rel=1e-12, abs=0)


def test_exponentially_tiny_weights_retain_representable_weighted_energy():
    # exp(-800) underflows, but sqrt(exp(-800)*1e400) is representable.
    d = np.array([0., 1e200]); temperature = float(1e200/np.sqrt(800))
    residual = smooth_patch_distances(d, np.zeros(2, np.int64), 1, temperature)[0]
    assert residual > 0
    assert np.log(residual) == pytest.approx(np.log(1e200)-400, abs=2e-12)


@pytest.mark.parametrize('scale', [1e-300, 1e300, np.finfo(float).max])
def test_equal_positive_candidates_are_finite_and_never_fake_zero(scale):
    residual = smooth_patch_distances(np.full(3, scale), np.zeros(3, np.int64), 1, .01)[0]
    assert np.isfinite(residual) and residual > 0
    assert residual == pytest.approx(scale, rel=1e-12, abs=0)


@pytest.mark.parametrize('d,groups,n,temp', [
    (np.array([-.01]), np.array([0]), 1, .01),
    (np.array([np.nan]), np.array([0]), 1, .01),
    (np.array([.01]), np.array([0.]), 1, .01),
    (np.array([.01]), np.array([1]), 2, .01),
    (np.array([.01]), np.array([-1]), 1, .01),
    (np.array([.01]), np.array([0]), 1, 0.),
    (np.array([.01]), np.array([0]), True, .01),
    (np.array([[.01]]), np.array([[0]]), 1, .01),
    (np.ma.array([.01]), np.array([0]), 1, .01),
])
def test_invalid_or_empty_supported_group_rejected(d, groups, n, temp):
    with pytest.raises(ValueError):
        smooth_patch_distances(d, groups, n, temp)


def test_empty_group_set_and_all_zero_physical_contact_are_explicit():
    assert smooth_patch_distances(np.empty(0), np.empty(0, np.int64), 0, .01).shape == (0,)
    np.testing.assert_array_equal(smooth_patch_distances(np.zeros(4), np.array([0, 1, 1, 0]), 2, .01), [0., 0.])

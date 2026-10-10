"""Tiny manufactured covariance contracts; no data, models or quality claims."""
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from world_reward.depth_covariance import CommonModeDepthConfig, FrameDepthWhitening


def test_config_preserves_explicit_external_reference_and_zero_legacy_mode():
    config = CommonModeDepthConfig(0., 'manufactured:common_mode_unit_contract')
    assert asdict(config) == dict(sigma_common_diameter=0.,
                                 development_reference='manufactured:common_mode_unit_contract')
    with pytest.raises(FrozenInstanceError):
        config.sigma_common_diameter = .1


@pytest.mark.parametrize('value', [-1., np.inf, -np.inf, np.nan, True, '1', None])
def test_config_rejects_invalid_common_scale(value):
    with pytest.raises(ValueError):
        CommonModeDepthConfig(value, 'external:unit')


@pytest.mark.parametrize('reference', ['', ' ', None, 3])
def test_config_requires_development_reference(reference):
    with pytest.raises(ValueError):
        CommonModeDepthConfig(.03, reference)


def test_quadratic_and_symmetric_operator_equal_dense_inverse_covariance():
    rng = np.random.default_rng(31415)
    ids = np.array([7, 1, 7, 12, 1, 1, 7, 5, 1], np.int64)
    iid, common = .007, .06
    operator = FrameDepthWhitening(ids, iid, common)
    covariance = iid**2*np.eye(len(ids))+common**2*(ids[:, None] == ids[None])
    eigvals, eigvecs = np.linalg.eigh(covariance)
    reference = (eigvecs/np.sqrt(eigvals))@eigvecs.T
    for _ in range(5):
        residual = rng.normal(0., .04, len(ids))
        actual = operator.apply(residual)
        np.testing.assert_allclose(actual, reference@residual, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual@actual,
            residual@np.linalg.solve(covariance, residual), rtol=1e-12)


def test_permutation_equivariance_without_requiring_sorted_or_contiguous_ids():
    ids = np.array([100, 2, 100, 2, 100, 999, 2])
    residual = np.array([.05, -.01, .09, -.07, -.04, .3, .02])
    permutation = np.array([6, 4, 0, 5, 2, 3, 1])
    original = FrameDepthWhitening(ids, .01, .04).apply(residual)
    permuted = FrameDepthWhitening(ids[permutation], .01, .04).apply(residual[permutation])
    np.testing.assert_allclose(permuted, original[permutation], atol=1e-14)


def test_actual_support_counts_control_common_mode_information_not_nominal_tracks():
    ids = np.array([1, 3, 3, 3, 3, 40, 40])
    operator = FrameDepthWhitening(ids, .02, .08)
    result = operator.apply(np.full(7, .03))
    np.testing.assert_allclose(result,
        .03/np.sqrt(.02**2+np.array([1, 4, 4, 4, 4, 2, 2])*.08**2))
    assert operator.group_count == 3 and operator.observation_count == 7
    assert not np.any(operator.frame_ids == 2)  # No fabricated missing frame.


def test_relative_geometry_contrasts_keep_independent_weight():
    residual = np.array([-.02, .04, -.06, .04])  # Zero-mean within one frame.
    result = FrameDepthWhitening(np.ones(4, int), .01, .3).apply(residual)
    np.testing.assert_allclose(result, residual/.01, atol=1e-15)


def test_duplicate_common_evidence_does_not_gain_unbounded_information():
    energies = []
    for n in (1, 4, 100):
        result = FrameDepthWhitening(np.zeros(n, int), .01, .1).apply(np.full(n, .03))
        energies.append(result@result)
    assert energies[2] < 1.01*energies[0]
    assert energies[2] < (.03/.1)**2


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_common_zero_is_exact_legacy_division(dtype):
    residual = np.array([.1, -.2, .3, .123456789], dtype=dtype)
    operator = FrameDepthWhitening(np.array([0, 3, 0, 3]), .013, 0.)
    np.testing.assert_array_equal(operator.apply(residual), residual.astype(np.float64)/.013)


def test_operator_owns_immutable_arrays_and_never_mutates_inputs():
    ids = np.array([1, 1, 2])
    residual = np.array([.04, -.01, .02]); saved = residual.copy()
    operator = FrameDepthWhitening(ids, .01, .03)
    expected = operator.apply(residual)
    ids[:] = 100
    np.testing.assert_array_equal(operator.frame_ids, [1, 1, 2])
    np.testing.assert_array_equal(operator.apply(residual), expected)
    np.testing.assert_array_equal(residual, saved)
    with pytest.raises(ValueError):
        operator.frame_ids.setflags(write=True)
    with pytest.raises(ValueError):
        operator._groups[0] = 10
    with pytest.raises(FrozenInstanceError):
        operator.sigma_common_m = 1.


@pytest.mark.parametrize('ids', [[], np.array([1.]), np.array([True]), np.array([-1]),
                                 np.array([[1]]), np.ma.array([1], mask=[False])])
def test_invalid_frame_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        FrameDepthWhitening(ids, .01, .03)


@pytest.mark.parametrize('iid,common', [(0., .1), (-1., .1), (np.inf, .1), (np.nan, .1),
    (True, .1), (.01, -1.), (.01, np.inf), (.01, np.nan), (.01, False)])
def test_invalid_operator_scales_are_rejected(iid, common):
    with pytest.raises(ValueError):
        FrameDepthWhitening(np.array([1]), iid, common)


@pytest.mark.parametrize('residual', [np.array([1]), np.array([np.nan]), np.array([np.inf]),
    np.array([[.1]]), np.array([.1, .2]), np.ma.array([.1], mask=[False])])
def test_invalid_supported_residuals_are_rejected(residual):
    with pytest.raises(ValueError):
        FrameDepthWhitening(np.array([1]), .01, .03).apply(residual)


def test_large_mean_avoids_naive_sum_overflow():
    result = FrameDepthWhitening(np.ones(2, int), 1., 1e308).apply(np.full(2, 1e308))
    np.testing.assert_allclose(result, np.full(2, 1/np.sqrt(2)), rtol=1e-15)


def test_invalid_covariance_arithmetic_and_residual_overflow_fail_fast():
    with pytest.raises(ValueError, match='Covariance scales'):
        FrameDepthWhitening(np.array([1]), float(np.nextafter(0., 1.)), .1)
    with pytest.raises(ValueError, match='Covariance scales'):
        FrameDepthWhitening(np.ones(4, int), 1., 1e308)
    with pytest.raises(ValueError, match='Whitened residuals'):
        FrameDepthWhitening(np.array([1]), 1e-300, 0.).apply(np.array([1e300]))


def test_single_observation_is_total_variance_not_independent_variance():
    result = FrameDepthWhitening(np.array([7]), .03, .04).apply(np.array([.1]))
    np.testing.assert_array_equal(result, [2.])

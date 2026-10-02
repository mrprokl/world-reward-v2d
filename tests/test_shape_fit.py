"""Procedural CPU tests of proposal contracts, not reconstruction validation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import world_reward.shape_fit as shape_fit
from world_reward.shape_model import apply_fixed_shape, deformation


@pytest.fixture
def procedural_case():
    rng = np.random.default_rng(17)
    # Uneven fixed sample, asymmetric canonical dimensions, distinct surfaces.
    surface = rng.uniform(-1, 1, (64, 3)) * [.7, .43, .28] + [.08, -.04, .11]
    surface[:24, 0] = -.62
    surface[24:48, 1] = .39
    surface[48:, 2] = .39
    pivot = surface.mean(axis=0)
    known = np.array([.06, -.05, .02, -.015, .025])
    shaped = apply_fixed_shape(surface, known, centroid=pivot)
    rotations = Rotation.from_euler("xyz", [[.2, -.1, .4], [-.1, .3, -.5], [.4, -.2, .1]]).as_matrix()
    translations = np.array([[.2, -.1, 2.3], [-.3, .1, 2.8], [.1, .2, 2.6]])
    visible_indices = [np.arange(0, 30), np.arange(24, 58), np.r_[np.arange(5, 14), np.arange(42, 64)]]
    # Known shape is test-fixture synthesis only, never a fitting input or correspondence.
    observed = [shaped[indices] @ rotation.T + translation
                for indices, rotation, translation in zip(visible_indices, rotations, translations)]
    return surface, pivot, observed, rotations, translations, known


def fit_case(case, **kwargs):
    return shape_fit.fit_shared_shape(*case[:5], **kwargs)


def test_partial_multiview_shared_shape_improves_fixed_pose_observed_fit(procedural_case):
    result = fit_case(procedural_case)
    assert result.accepted
    assert result.report["optimizer_success"]
    assert result.report["raw_equal_frame_rmse_after_m"] < result.report["raw_equal_frame_rmse_before_m"] * .15
    assert result.report["robust_prior_cost_after"] < result.report["robust_prior_cost_before"]
    assert np.linalg.norm(result.params5 - procedural_case[-1]) < .012
    assert np.linalg.det(result.matrix) == pytest.approx(1., rel=1e-13)
    np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-15)
    assert np.linalg.eigvalsh(result.matrix).min() > 0
    assert result.report["poses_fitted"] is False and result.report["scale_fitted"] is False
    assert result.report["projected_or_clipped"] is False
    assert result.report["challenge_performance_verified"] is False
    assert result.report["nearest_neighbour_direction"] == "observed_to_deformed_surface_only"
    json.dumps(result.report, allow_nan=False)


def test_fit_does_not_mutate_samples_pivot_observations_or_fixed_poses(procedural_case):
    before = copy.deepcopy(procedural_case)
    result = fit_case(procedural_case)
    for original, old in zip(procedural_case, before):
        if isinstance(original, list):
            for value, copy_ in zip(original, old): np.testing.assert_array_equal(value, copy_)
        else: np.testing.assert_array_equal(original, old)
    result.params5[:] = 0
    np.testing.assert_array_equal(procedural_case[-1], before[-1])


def test_identity_perfect_evidence_unchanged_without_hidden_surface_pull(procedural_case):
    surface, pivot, _, rotations, translations, _ = procedural_case
    # Only one small visible subset per frame; no reverse Chamfer obligation.
    observed = [surface[:8] @ rotation.T + translation for rotation, translation in zip(rotations, translations)]
    result = shape_fit.fit_shared_shape(surface, pivot, observed, rotations, translations)
    assert not result.accepted
    np.testing.assert_array_equal(result.params5, np.zeros(5))
    assert result.report["raw_equal_frame_rmse_before_m"] == pytest.approx(0, abs=1e-15)
    assert result.report["raw_equal_frame_rmse_after_m"] == pytest.approx(0, abs=1e-15)


def test_equal_frame_weight_and_robust_physical_transition_independent_of_count():
    counts = [4, 20]
    loss = shape_fit._equal_frame_soft_l1(counts, .01)
    # Equal physical coordinate errors, but residuals are divided by sqrt(N_i).
    delta = .03
    residuals = np.r_[np.full(12, delta / np.sqrt(4)), np.full(60, delta / np.sqrt(20)), np.zeros(5)]
    rho = loss(residuals**2)
    assert rho[0, :12].sum() == pytest.approx(rho[0, 12:72].sum(), rel=1e-14)
    assert rho[1, 0] == pytest.approx(rho[1, 12])
    assert rho[1, 0] == pytest.approx(1 / np.sqrt(1 + (delta / .01)**2))
    # Shape prior stays squared, not robustly truncated.
    residuals[-5:] = 1
    prior = loss(residuals**2)
    np.testing.assert_array_equal(prior[:, -5:], np.tile([1., 1., 0.], (5, 1)).T)


def test_observation_replication_does_not_reweight_a_frame(procedural_case):
    surface, pivot, observations, rotations, translations, _ = procedural_case
    first = fit_case(procedural_case, shape_prior=0)
    repeated = [np.repeat(observations[0], 5, axis=0), *observations[1:]]
    second = shape_fit.fit_shared_shape(surface, pivot, repeated, rotations, translations, shape_prior=0)
    np.testing.assert_allclose(second.params5, first.params5, atol=2e-6, rtol=0)
    assert second.report["raw_equal_frame_rmse_before_m"] == pytest.approx(first.report["raw_equal_frame_rmse_before_m"], rel=1e-13)


def test_deterministic_capped_subsample_records_exact_indices(procedural_case):
    surface, pivot, observations, rotations, translations, _ = procedural_case
    many = [np.tile(points, (40, 1)) for points in observations]
    result = shape_fit.fit_shared_shape(surface, pivot, many, rotations, translations,
                                        max_points_per_frame=48, max_nfev=50)
    repeated = shape_fit.fit_shared_shape(surface, pivot, many, rotations, translations,
                                          max_points_per_frame=48, max_nfev=50)
    np.testing.assert_array_equal(result.params5, repeated.params5)
    assert result.report["sampled_observation_counts"] == [48] * 3
    for original, selected in zip(many, result.report["sampled_original_indices"]):
        np.testing.assert_array_equal(selected, np.linspace(0, len(original) - 1, 48, dtype=np.int64))


def test_nonconvergence_candidate_is_not_accepted(procedural_case):
    result = fit_case(procedural_case, max_nfev=1)
    assert not result.accepted
    assert result.report["status"] == "optimizer_failed"
    assert result.report["optimizer_nfev"] == 1
    assert result.report["optimizer_status"] == 0
    assert result.report["input_arrays_modified"] is False


def test_raw_residual_worse_candidate_not_adopted_even_if_solver_says_success(procedural_case, monkeypatch):
    def false_success(*args, **kwargs):
        return SimpleNamespace(x=np.array([-.07, .08, -.06, .06, -.06]), success=True,
                               status=1, message="synthetic solver status", nfev=2)
    monkeypatch.setattr(shape_fit, "least_squares", false_success)
    result = fit_case(procedural_case)
    assert not result.accepted
    assert result.report["raw_equal_frame_rmse_after_m"] > result.report["raw_equal_frame_rmse_before_m"]
    assert result.report["status"] == "raw_residual_worse"


def test_optimizer_cannot_bypass_conservative_bounds_or_finiteness(procedural_case, monkeypatch):
    for invalid in (np.array([.2, 0, 0, 0, 0]), np.array([np.nan, 0, 0, 0, 0])):
        monkeypatch.setattr(shape_fit, "least_squares", lambda *args, **kwargs: SimpleNamespace(x=invalid))
        with pytest.raises(ValueError): fit_case(procedural_case)


def test_conservative_parameter_box_satisfies_original_spectral_gate():
    bound = np.log(1.5) / 4
    for code in range(32):
        params = np.array([bound if code & (1 << index) else -bound for index in range(5)])
        matrix = deformation(params)
        assert np.linalg.eigvalsh(matrix).min() >= 1 / 1.5 - 1e-14
        assert np.linalg.eigvalsh(matrix).max() <= 1.5 + 1e-14


@pytest.mark.parametrize("field,bad", [("surface", np.empty((0, 3))), ("surface", np.zeros((4, 3))),
                                      ("surface", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])),
                                      ("surface", np.array([[np.nan, 0, 0]])), ("surface", np.ma.array(np.ones((4, 3)), mask=False)),
                                      ("pivot", [0, 0, np.inf]), ("pivot", np.ma.array(np.zeros(3), mask=False)),
                                      ("observations", []), ("observations", [np.empty((0, 3))]),
                                      ("observations", [np.zeros((4, 3))]), ("observations", "points")])
def test_invalid_arrays_and_rank_fail_before_optimization(procedural_case, field, bad):
    arguments = list(procedural_case[:5])
    arguments[{"surface": 0, "pivot": 1, "observations": 2}[field]] = bad
    with pytest.raises(ValueError): shape_fit.fit_shared_shape(*arguments)


@pytest.mark.parametrize("field,bad", [("rotations", np.tile(np.diag([-1., 1, 1]), (3, 1, 1))),
                                      ("rotations", np.tile(np.eye(3) * 1.01, (3, 1, 1))),
                                      ("rotations", np.zeros((1, 3, 3))),
                                      ("translations", np.zeros((1, 3))), ("translations", np.full((3, 3), np.nan)),
                                      ("translations", np.ma.array(np.zeros((3, 3)), mask=False))])
def test_invalid_or_non_so3_fixed_poses_fail(procedural_case, field, bad):
    arguments = list(procedural_case[:5])
    arguments[3 if field == "rotations" else 4] = bad
    with pytest.raises(ValueError): shape_fit.fit_shared_shape(*arguments)


@pytest.mark.parametrize("name,value", [("max_nfev", 0), ("max_nfev", 101), ("max_nfev", True),
                                       ("max_points_per_frame", 2), ("max_points_per_frame", 2049),
                                       ("max_points_per_frame", 3.), ("shape_prior", -.01),
                                       ("shape_prior", np.nan), ("shape_prior", True), ("f_scale_m", 0),
                                       ("f_scale_m", np.inf), ("f_scale_m", np.nan), ("f_scale_m", True),
                                       ("f_scale_m", 1e-300)])
def test_predeclared_control_bounds_fail_closed(procedural_case, name, value):
    with pytest.raises(ValueError): fit_case(procedural_case, **{name: value})

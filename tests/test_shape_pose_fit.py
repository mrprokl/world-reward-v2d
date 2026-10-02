"""Tiny exact geometry/numerical guards, never local CUDA fitting."""

import numpy as np
import pytest

from world_reward.shape_pose_fit import (
    JACOBIAN_STEP, MAX_MODEL_CALLS, MAX_OPTIMIZER_CALLS, ROTATION_UNIT_RAD,
    SHAPE_UNIT, TRANSLATION_UNIT_M, _check_budget, data_schur_from_jacobian,
    fit_nested_shape_pose, fixed_feature_residuals, transform_numpy,
)


def geometry():
    vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
    faces = np.array([[0, 1, 2]])
    return vertices, faces


def test_frozen_normalized_units_and_total_budget_include_diagnostics():
    assert SHAPE_UNIT == .05 and ROTATION_UNIT_RAD == .01 and TRANSLATION_UNIT_M == .01
    assert MAX_OPTIMIZER_CALLS == 50 and MAX_MODEL_CALLS == 100
    assert MAX_OPTIMIZER_CALLS + 1 + 2 * (5 + 3 * 6) + 1 == 98
    assert JACOBIAN_STEP == 1e-4


def test_transform_is_exact_rigid_pose_not_linearized_vertex_deformation():
    from scipy.spatial.transform import Rotation
    vertices, _ = geometry()
    R, t, pivot = np.eye(3)[None], np.zeros((1, 3)), np.zeros(3)
    q = np.array([1., -2., 3., 4., -3., 2.])
    shaped, rotation, translation, theta = transform_numpy(q, vertices, pivot, R, t, False)
    np.testing.assert_array_equal(shaped, vertices)
    np.testing.assert_allclose(rotation[0], Rotation.from_rotvec(q[:3] * .01).as_matrix(), atol=1e-15)
    np.testing.assert_allclose(rotation[0] @ rotation[0].T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation[0]) == pytest.approx(1.)
    np.testing.assert_array_equal(translation[0], q[3:] * .01)
    np.testing.assert_array_equal(theta, np.zeros(5))
    assert shaped is not vertices


def test_joint_transform_shared_spd_volume_and_pivot_invariant():
    vertices, _ = geometry()
    pivot = np.array([.02, -.03, .01])
    vertices = np.vstack((vertices, pivot))
    q = np.r_[.4, -.2, .1, -.15, .05, np.zeros(12)]
    R, t = np.repeat(np.eye(3)[None], 2, axis=0), np.zeros((2, 3))
    shaped, rotation, translation, theta = transform_numpy(q, vertices, pivot, R, t, True)
    np.testing.assert_allclose(shaped[-1], pivot, atol=1e-15)
    np.testing.assert_array_equal(theta, q[:5] * .05)
    np.testing.assert_allclose(rotation, R, atol=1e-15)
    np.testing.assert_array_equal(translation, t)


@pytest.mark.parametrize("q", [np.ones(5), np.full(6, np.nan), np.ones((1, 6))])
def test_invalid_transform_dimensions_nonfinite_rejected(q):
    vertices, _ = geometry()
    with pytest.raises(ValueError): transform_numpy(q, vertices, np.zeros(3), np.eye(3)[None], np.zeros((1, 3)), False)


@pytest.mark.parametrize("q,joint", [(np.ma.array(np.zeros(6), mask=False), False), (np.zeros(6), 1),
                                      (np.full(6, "0"), False), (np.zeros(6), np.bool_(False))])
def test_ambiguous_model_or_hidden_parameters_rejected(q, joint):
    vertices, _ = geometry()
    with pytest.raises(ValueError): transform_numpy(q, vertices, np.zeros(3), np.eye(3)[None], np.zeros((1, 3)), joint)


def test_fixed_plane_and_edge_vector_residual_analytic_small_triangle():
    vertices, faces = geometry()
    points = np.array([[.0025, .0025, .01], [.001, -.02, .015], [-.02, -.02, .02]])
    assignments = [(np.zeros(3, dtype=int), np.array([0, 1, 1]), np.array([0, 0, -1]))]
    result = fixed_feature_residuals(vertices[None], faces, [points], assignments).reshape(-1, 3) * np.sqrt(3)
    np.testing.assert_allclose(result, [[0., 0., .01], [0., -.02, .015], [-.02, -.02, .02]], atol=1e-15)
    expected_squared = np.sum(result**2, axis=1)
    from world_reward.continuous_surface import numpy_reference_distance_squared
    np.testing.assert_allclose(expected_squared, numpy_reference_distance_squared(points, vertices[faces]), atol=1e-15)


def test_fixed_feature_residual_zero_surface_is_nonsingular_vector_not_sqrt():
    vertices, faces = geometry()
    points = np.array([[.0025, .0025, 0.]])
    assignments = [(np.array([0]), np.array([0]), np.array([0]))]
    zero = fixed_feature_residuals(vertices[None], faces, [points], assignments)
    np.testing.assert_array_equal(zero, np.zeros(3))
    plus = fixed_feature_residuals((vertices + [0., 0., 1e-6])[None], faces, [points], assignments)
    minus = fixed_feature_residuals((vertices - [0., 0., 1e-6])[None], faces, [points], assignments)
    np.testing.assert_allclose((plus - minus) / 2e-6, [0., 0., -1.], atol=1e-15)


def test_data_schur_uses_no_prior_or_damping_and_exposes_gauge():
    J = np.eye(11)
    np.testing.assert_array_equal(data_schur_from_jacobian(J), np.eye(5))
    # Shape columns equal the first five pose columns: full shape/pose gauge.
    J = np.zeros((6, 11)); J[:, 5:] = np.eye(6); J[:, :5] = J[:, 5:10]
    np.testing.assert_array_equal(data_schur_from_jacobian(J), np.zeros((5, 5)))
    with pytest.raises(ValueError, match="rank-deficient"):
        data_schur_from_jacobian(np.zeros((6, 11)))


@pytest.mark.parametrize("jacobian", [np.ones((5, 10)), np.ones((5, 12)), np.full((11, 11), np.nan),
                                      np.ma.array(np.eye(11), mask=False)])
def test_bad_or_hidden_data_jacobian_not_repaired(jacobian):
    with pytest.raises(ValueError): data_schur_from_jacobian(jacobian)


def test_overflow_data_hessian_fails_instead_of_fake_conditioning():
    with pytest.raises(ValueError, match="finite numeric range"):
        data_schur_from_jacobian(np.full((11, 11), 1e200))


@pytest.mark.parametrize("flag", [False, None, 1, "true", np.bool_(True)])
def test_fit_requires_exact_disjoint_holdout_before_any_gpu_import(flag):
    with pytest.raises(ValueError, match="disjoint"):
        fit_nested_shape_pose(None, None, None, None, None, None, None, heldout_pixels_disjoint=flag)


def test_expired_deadline_fails_fast_without_cuda(monkeypatch):
    import world_reward.shape_pose_fit as module
    monkeypatch.setattr(module.time, "perf_counter", lambda: 12.)
    with pytest.raises(TimeoutError): _check_budget(11.)
    _check_budget(12.)


def test_underconstrained_or_wrong_view_count_rejected_before_cuda():
    v, f = geometry()
    points = np.array([[0., 0., .01], [.001, 0., .01], [0., .001, .01]])
    with pytest.raises(ValueError, match="three views"):
        fit_nested_shape_pose(v, f, np.zeros(3), [points], [points], np.eye(3)[None], np.zeros((1, 3)), heldout_pixels_disjoint=True)
    with pytest.raises(ValueError, match="128"):
        fit_nested_shape_pose(v, f, np.zeros(3), [np.tile(points, (44, 1))] * 3, [points] * 3,
                              np.repeat(np.eye(3)[None], 3, 0), np.zeros((3, 3)), heldout_pixels_disjoint=True)

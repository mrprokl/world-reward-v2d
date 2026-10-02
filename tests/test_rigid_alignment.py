import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.rigid_alignment import align_observed_points


def shape():
    rng = np.random.default_rng(217)
    points = rng.uniform(-1, 1, (512, 3))
    # Distinct axis scales and curved offset avoid artificial rotational symmetry.
    points[:, 0] *= 1.3
    points[:, 1] = 0.7 * points[:, 1] + 0.15 * points[:, 0] ** 2
    points[:, 2] *= 0.4
    return points


def truth():
    return Rotation.from_rotvec([0.07, -0.05, 0.04]).as_matrix(), np.array([0.2, -0.1, 2.0])


def test_partial_asymmetric_observations_recover_rigid_pose_without_hidden_surface_penalty():
    source = shape()
    rotation, translation = truth()
    observed = source[source[:, 0] > -0.3] @ rotation.T + translation
    initial_r = Rotation.from_rotvec([0.01, -0.01, 0.01]).as_matrix() @ rotation
    initial_t = translation + [0.02, -0.01, 0.015]
    result = align_observed_points(source, observed, initial_r, initial_t)
    assert result.status == "improved"
    assert result.final_residual < result.initial_residual
    assert np.allclose(result.rotation, rotation, atol=1e-8)
    assert np.allclose(result.translation, translation, atol=1e-8)
    assert result.final_residual < 1e-10
    assert result.inliers == int(0.8 * len(observed))
    assert 1 <= result.iterations <= 30
    json.dumps(result.to_dict(), allow_nan=False)
    assert result.to_dict()["scale_fitted"] is False


def test_occlusion_and_outliers_within_trimming_budget_are_not_forced_into_fit():
    source = shape()
    rotation, translation = truth()
    visible = source[:256] @ rotation.T + translation
    outliers = np.full((40, 3), [30.0, -50.0, 70.0])
    observed = np.concatenate([visible, outliers])
    result = align_observed_points(source, observed, rotation, translation + [0.01, 0.01, -0.02])
    assert result.status == "improved"
    assert result.final_residual < 1e-10
    assert np.allclose(result.translation, translation, atol=1e-8)
    assert np.linalg.det(result.rotation) == pytest.approx(1)


def test_noisy_partial_evidence_improves_residual_with_fixed_geometry_scale():
    source = shape()
    rotation, translation = truth()
    rng = np.random.default_rng(52)
    observed = source[:256] @ rotation.T + translation + rng.normal(0, 0.001, (256, 3))
    result = align_observed_points(source, observed, rotation, translation + [0.025, -0.02, 0.015])
    assert result.final_residual < result.initial_residual
    assert result.final_residual < 0.003
    assert np.linalg.det(result.rotation) == pytest.approx(1)
    assert np.allclose(result.rotation.T @ result.rotation, np.eye(3), atol=1e-12)


def test_exact_zero_initial_fit_keeps_original_arrays_if_no_improvement():
    source = shape()
    rotation, translation = truth()
    observed = source @ rotation.T + translation
    r_before, t_before = rotation.copy(), translation.copy()
    result = align_observed_points(source, observed, rotation, translation)
    assert result.status == "unchanged"
    assert result.initial_residual == result.final_residual == 0
    assert np.array_equal(result.rotation, r_before)
    assert np.array_equal(result.translation, t_before)
    assert np.array_equal(rotation, r_before) and np.array_equal(translation, t_before)


def test_large_absolute_move_allowed_with_useful_initialization_no_temporal_or_speed_gate():
    source = shape()
    rotation, _ = truth()
    translation = np.array([300.0, -200.0, 900.0])
    observed = source[:256] @ rotation.T + translation
    result = align_observed_points(source, observed, rotation, translation + [0.015, 0.01, -0.01])
    assert result.final_residual < 1e-9
    assert np.allclose(result.translation, translation, atol=1e-8)


def test_planar_rank_two_correspondences_are_supported_with_proper_rotation():
    source = shape()
    source[:, 2] = 0
    rotation, translation = truth()
    observed = source[:256] @ rotation.T + translation
    result = align_observed_points(source, observed, rotation, translation + [0.01, -0.01, 0.01])
    assert result.final_residual < 1e-9
    assert np.linalg.det(result.rotation) == pytest.approx(1)


@pytest.mark.parametrize("points", [np.zeros((64, 3)), np.column_stack([np.arange(64), np.zeros(64), np.zeros(64)])])
def test_point_or_line_support_is_underconstrained_and_not_guessed(points):
    with pytest.raises(ValueError, match="rank < 2"):
        align_observed_points(points, points, np.eye(3), np.zeros(3))


def test_initial_pose_nearest_neighbours_collapsed_to_one_surface_point_fail_underconstraint():
    source = shape()
    observed = source[:64] + [1e5, 1e5, 1e5]
    with pytest.raises(ValueError, match="rank < 2"):
        align_observed_points(source, observed, np.eye(3), np.zeros(3))


def test_no_all_pairs_distance_matrix_uses_bounded_ckdtree(monkeypatch):
    import world_reward.rigid_alignment as module
    real_tree = module.cKDTree
    queries = []

    class CheckedTree:
        def __init__(self, points):
            assert points.ndim == 2 and points.shape[1] == 3
            self.tree = real_tree(points)

        def query(self, points, k):
            queries.append((points.shape, k))
            return self.tree.query(points, k=k)

    monkeypatch.setattr(module, "cKDTree", CheckedTree)
    source = shape()
    result = module.align_observed_points(source, source[:100], np.eye(3), np.zeros(3))
    assert result.status == "unchanged"
    assert queries and all(k == 1 for _, k in queries)


@pytest.mark.parametrize("bad", [
    np.empty((0, 3)), np.zeros((20, 2)), np.zeros(3),
    np.full((64, 3), np.nan), np.full((64, 3), np.inf),
    np.zeros((64, 3), dtype=bool), np.zeros((64, 3), dtype=complex),
    np.full((64, 3), "0"), np.ma.array(np.ones((64, 3)), mask=False),
])
@pytest.mark.parametrize("which", ["source", "observed"])
def test_malformed_points_fail_explicitly_no_silent_filter(bad, which):
    source = shape()
    a, b = (bad, source) if which == "source" else (source, bad)
    with pytest.raises(ValueError):
        align_observed_points(a, b, np.eye(3), np.zeros(3))


@pytest.mark.parametrize("rotation", [
    np.ones((3, 3)), np.diag([-1.0, 1.0, 1.0]), np.eye(2), np.full((3, 3), np.nan),
])
def test_invalid_initial_rotation_fails_not_projected_to_so3(rotation):
    source = shape()
    with pytest.raises(ValueError, match="proper SO"):
        align_observed_points(source, source, rotation, np.zeros(3))


@pytest.mark.parametrize("translation", [np.zeros(2), np.full(3, np.inf), np.full(3, np.nan)])
def test_invalid_initial_translation_fails(translation):
    source = shape()
    with pytest.raises(ValueError, match="translation"):
        align_observed_points(source, source, np.eye(3), translation)


@pytest.mark.parametrize("kwargs", [
    {"trim_fraction": 0}, {"trim_fraction": 1.1}, {"trim_fraction": np.nan},
    {"trim_fraction": True}, {"min_correspondences": 0}, {"min_correspondences": True},
    {"max_iterations": 0}, {"max_iterations": 1.5}, {"tolerance": -1}, {"tolerance": np.inf},
])
def test_invalid_preregistered_controls_fail(kwargs):
    source = shape()
    with pytest.raises(ValueError):
        align_observed_points(source, source, np.eye(3), np.zeros(3), **kwargs)


def test_insufficient_observed_trimmed_support_fails_no_padding():
    source = shape()
    with pytest.raises(ValueError, match="Insufficient observed"):
        align_observed_points(source, source[:39], np.eye(3), np.zeros(3))


def test_input_source_observed_geometry_and_pose_arrays_never_mutated():
    source = shape()
    rotation, translation = truth()
    observed = source[:256] @ rotation.T + translation
    copies = [value.copy() for value in (source, observed, rotation, translation)]
    align_observed_points(source, observed, rotation, translation)
    assert all(np.array_equal(value, copy) for value, copy in zip((source, observed, rotation, translation), copies))


def test_kabsch_reflection_correction_produces_proper_so3_not_mirrored_geometry():
    from world_reward.rigid_alignment import _kabsch

    source = shape()[:64]
    target = source * [-1, 1, 1]
    rotation, translation = _kabsch(source, target)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isfinite(translation).all()


def test_worsening_proposal_keeps_original_transform_exactly(monkeypatch):
    import world_reward.rigid_alignment as module

    source = shape()
    rotation = np.eye(3)
    translation = np.array([0.01, -0.01, 0.01])
    monkeypatch.setattr(module, "_kabsch", lambda *_: (np.eye(3), np.array([100, 100, 100])))
    result = module.align_observed_points(source, source, rotation, translation)
    assert result.status == "unchanged"
    assert result.final_residual == result.initial_residual
    assert np.array_equal(result.rotation, rotation)
    assert np.array_equal(result.translation, translation)

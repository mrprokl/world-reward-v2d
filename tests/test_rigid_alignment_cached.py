"""Tiny own procedural tests; no full benchmark, challenge data, GPU or network."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import world_reward.rigid_alignment_cached as cached_module
from world_reward.rigid_alignment import RigidAlignment, _correspondences, align_observed_points
from world_reward.rigid_alignment_cached import _cached_correspondences, align_observed_points_cached


def procedural():
    rng = np.random.default_rng(307)
    mesh = rng.uniform(-1, 1, (192, 3)) * [.5, .3, .2]
    mesh[:, 1] += .15 * mesh[:, 0] ** 2
    rotation = Rotation.from_rotvec([.07, -.05, .04]).as_matrix()
    translation = np.array([.2, -.1, 2.])
    visible = mesh[:96] @ rotation.T + translation + rng.normal(0, .001, (96, 3))
    observations = np.concatenate([visible, rng.uniform(5, 10, (12, 3))])
    initial_R = Rotation.from_rotvec([.01, -.01, .01]).as_matrix() @ rotation
    initial_t = translation + [.01, -.005, .008]
    return mesh, observations, initial_R, initial_t


def test_noisy_partial_evidence_exact_discrete_abi_and_numerical_pose_parity():
    inputs = procedural()
    before = [value.copy() for value in inputs]
    original = align_observed_points(*inputs)
    cached = align_observed_points_cached(*inputs)
    assert original.status == cached.status == "improved"
    assert original.inliers == cached.inliers and original.iterations == cached.iterations
    np.testing.assert_allclose(cached.rotation, original.rotation, atol=1e-12, rtol=0)
    np.testing.assert_allclose(cached.translation, original.translation, atol=1e-12, rtol=0)
    assert abs(cached.initial_residual - original.initial_residual) < 1e-12
    assert abs(cached.final_residual - original.final_residual) < 1e-12
    assert cached.final_residual < cached.initial_residual
    assert set(cached.to_dict()) == set(original.to_dict())
    assert cached.to_dict()["scale_fitted"] is False
    json.dumps(cached.to_dict(), allow_nan=False)
    for value, old in zip(inputs, before, strict=True): assert value.tobytes() == old.tobytes()


def test_inverse_rigid_query_same_exact_nontied_correspondence_indices_and_metres():
    mesh, observed, rotation, translation = procedural()
    tree = cKDTree(mesh)
    source, targets, residual = _correspondences(mesh, observed, rotation, translation, 70)
    csource, ctargets, cresidual = _cached_correspondences(tree, mesh, observed, rotation, translation, 70)
    np.testing.assert_array_equal(source, csource)
    np.testing.assert_array_equal(targets, ctargets)
    assert abs(cresidual - residual) < 1e-12
    wrong_query = cKDTree(mesh).query(observed @ rotation - translation, k=1)[0]
    assert np.max(wrong_query) > .05  # Wrong inverse-translation order is detectable.


def test_only_one_tree_built_per_alignment_and_queries_are_bounded(monkeypatch):
    tree_builds, queries = [], []
    real_tree = cached_module.cKDTree
    class CheckedTree:
        def __init__(self, points):
            tree_builds.append(points.copy())
            self.tree = real_tree(points)
        def query(self, points, k):
            queries.append((points.shape, k))
            return self.tree.query(points, k=k)
    monkeypatch.setattr(cached_module, "cKDTree", CheckedTree)
    inputs = procedural()
    result = align_observed_points_cached(*inputs)
    assert len(tree_builds) == 1
    np.testing.assert_array_equal(tree_builds[0], inputs[0])
    assert len(queries) == result.iterations + 1
    assert all(shape == inputs[1].shape and k == 1 for shape, k in queries)


def test_absolute_kabsch_pose_not_composed_with_initializer(monkeypatch):
    mesh, observed, R, t = procedural()
    expected = align_observed_points(mesh, observed, R, t)
    proposals = []
    original_kabsch = cached_module._kabsch
    def audited(source, target):
        result = original_kabsch(source, target)
        proposals.append(result)
        return result
    monkeypatch.setattr(cached_module, "_kabsch", audited)
    actual = align_observed_points_cached(mesh, observed, R, t)
    assert proposals
    np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-12)
    np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-12)
    assert np.linalg.norm(actual.translation - t) > 1e-4


def test_worsening_proposal_returns_initial_pose_exactly(monkeypatch):
    inputs = procedural()
    monkeypatch.setattr(cached_module, "_kabsch", lambda *_: (np.eye(3), np.array([100., 100., 100.])))
    result = align_observed_points_cached(*inputs)
    assert result.status == "unchanged" and result.initial_residual == result.final_residual
    assert result.rotation.tobytes() == inputs[2].tobytes()
    assert result.translation.tobytes() == inputs[3].tobytes()


def test_exact_zero_identity_case_unchanged_without_invented_bit_parity_for_arbitrary_rotations():
    mesh = procedural()[0]
    result = align_observed_points_cached(mesh, mesh[:96], np.eye(3), np.zeros(3))
    assert result.status == "unchanged" and result.initial_residual == result.final_residual == 0
    # Arbitrary rotation followed by its inverse can introduce roundoff. We do
    # not force the new solver's status/iteration parity for exact-zero fits.
    R, t = procedural()[2:]
    observed = mesh[:96] @ R.T + t
    new = align_observed_points_cached(mesh, observed, R, t)
    old = align_observed_points(mesh, observed, R, t)
    assert old.status == "unchanged"
    assert new.initial_residual < 1e-12 and new.final_residual < 1e-12
    np.testing.assert_allclose(new.rotation, old.rotation, atol=1e-12)
    np.testing.assert_allclose(new.translation, old.translation, atol=1e-12)


def test_stable_trimming_retains_observation_order_at_exact_identity_ties():
    mesh = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    observed = np.tile([.5, 0., 0.], (4, 1))
    source, targets, residual = _cached_correspondences(cKDTree(mesh), mesh, observed, np.eye(3), np.zeros(3), 2)
    np.testing.assert_array_equal(targets, [0, 1])
    assert source[0] == source[1] and residual == .5


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("bad", [np.empty((0, 3)), np.zeros((64, 2)), np.full((64, 3), np.nan),
                                  np.zeros((64, 3), dtype=bool), np.ma.array(np.ones((64, 3)), mask=False)])
def test_original_point_validation_reused_without_filter_or_repair(which, bad):
    inputs = list(procedural())
    inputs[which] = bad
    with pytest.raises(ValueError): align_observed_points_cached(*inputs)


@pytest.mark.parametrize("kwargs", [{"trim_fraction": 0}, {"trim_fraction": True}, {"trim_fraction": np.inf},
                                    {"min_correspondences": 0}, {"min_correspondences": True},
                                    {"max_iterations": 0}, {"max_iterations": 1.5},
                                    {"tolerance": -1}, {"tolerance": np.nan}])
def test_unchanged_declared_optimizer_controls_reject_invalid(kwargs):
    with pytest.raises(ValueError): align_observed_points_cached(*procedural(), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.diag([-1., 1., 1.]), np.eye(2), np.full((3, 3), np.nan)])
def test_no_rotation_projection_or_reflection_accepted(bad):
    mesh, observations, _, t = procedural()
    with pytest.raises(ValueError, match="proper SO"):
        align_observed_points_cached(mesh, observations, bad, t)


def test_rank_and_trim_support_fail_no_padding():
    mesh, observed, R, t = procedural()
    with pytest.raises(ValueError, match="rank < 2"):
        align_observed_points_cached(np.zeros((64, 3)), observed, R, t)
    with pytest.raises(ValueError, match="Insufficient"):
        align_observed_points_cached(mesh, observed[:39], R, t)
    with pytest.raises(ValueError, match="translation"):
        align_observed_points_cached(mesh, observed, R, np.zeros(2))


def test_inverse_query_overflow_fails_not_silently_clipped():
    mesh = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    observations = np.full((2, 3), 1e308)
    translation = np.full(3, -1e308)
    with pytest.raises(ValueError, match="Inverse-transformed"):
        _cached_correspondences(cKDTree(mesh), mesh, observations, np.eye(3), translation, 1)


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/rigid_cache_gate.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_rigid_cache_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_procedural_generator_is_deterministic_and_preserves24_hypotheses(gate):
    mesh, views = gate.procedural_cases(surface_count=64, observation_count=40)
    mesh_again, views_again = gate.procedural_cases(surface_count=64, observation_count=40)
    assert mesh.shape == (64, 3) and mesh.tobytes() == mesh_again.tobytes()
    assert len(views) == 3 and all(len(seeds) == 24 and points.shape == (40, 3) for points, seeds in views)
    for (points, seeds), (points_again, seeds_again) in zip(views, views_again, strict=True):
        assert points.tobytes() == points_again.tobytes()
        for (R, t), (R2, t2) in zip(seeds, seeds_again, strict=True):
            assert R.tobytes() == R2.tobytes() and t.tobytes() == t2.tobytes()


@pytest.mark.parametrize("name,value", [("status", "unchanged"), ("inliers", 63), ("iterations", 4)])
def test_gate_never_relaxes_discrete_status_iterations_or_inlier_parity(gate, name, value):
    original = RigidAlignment(np.eye(3), np.zeros(3), "improved", .1, .01, 64, 3)
    fields = dict(original.__dict__)
    fields[name] = value
    with pytest.raises(ValueError, match="exact"):
        gate.compare_results(original, RigidAlignment(**fields))


def test_gate_numerical_limits_and_exact_correspondence_indices(gate):
    mesh, observed, R, t = procedural()
    old, new = align_observed_points(mesh, observed, R, t), align_observed_points_cached(mesh, observed, R, t)
    assert all(error < 1e-6 for error in gate.compare_results(old, new).values())
    query = gate.compare_queries(mesh, observed, R, t, cKDTree(mesh))
    assert query["nearest_indices_exact"] and query["trim_indices_exact"]
    changed = RigidAlignment(new.rotation, new.translation + [.000002, 0, 0], new.status,
                             new.initial_residual, new.final_residual, new.inliers, new.iterations)
    with pytest.raises(ValueError, match="1e-6"):
        gate.compare_results(old, changed)
    class WrongTree:
        def query(self, points, k):
            distances, indices = cKDTree(mesh).query(points, k=k)
            indices[0] = (indices[0] + 1) % len(mesh)
            return distances, indices
    with pytest.raises(ValueError, match="nearest-neighbour index"):
        gate.compare_queries(mesh, observed, R, t, WrongTree())


def test_gate_is_cpu_offline_frozen_and_no_challenge_mounts():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "infra/run_rigid_cache_gate.sh").read_text()
    assert "--network none" in wrapper and "--cpus 4" in wrapper and "--gpus" not in wrapper
    assert "src=$CODE,dst=$CODE,readonly" in wrapper
    assert "src=$ROOT/results,dst=$ROOT/results" in wrapper
    for directory in ("data", "outputs", "weights", "vendor"):
        assert f"src=$ROOT/{directory}" not in wrapper
    assert "no configurable" not in wrapper or "configurable controls" in wrapper


def test_failed_gate_preserves_partial_comparison_and_explicit_synthetic_oracle(gate, monkeypatch):
    mesh, observed, R, t = procedural()
    monkeypatch.setattr(gate, "procedural_cases", lambda: (mesh, [(observed, [(R, t)])]))
    def fail(*_): raise ValueError("strict parity failure")
    monkeypatch.setattr(gate, "compare_results", fail)
    diagnostics = {}
    with pytest.raises(ValueError, match="strict parity"):
        gate.run_gate(diagnostics)
    assert diagnostics["synthetic_oracle_initialization"] is True
    assert diagnostics["challenge_ground_truth_used"] is False
    assert diagnostics["throughput_hypothesis_accepted"] is False
    assert diagnostics["exact_status_inlier_iteration_parity"] is False
    assert len(diagnostics["comparisons"]) == 1
    assert "original" in diagnostics["comparisons"][0] and "cached" in diagnostics["comparisons"][0]
    assert diagnostics["timings_seconds"] == {"original_rebuild": [], "canonical_cached": []}


def test_failed_timing_retains_completed_trials_and_current_progress(gate, monkeypatch):
    mesh, observed, R, t = procedural()
    monkeypatch.setattr(gate, "procedural_cases", lambda: (mesh, [(observed, [(R, t)])]))
    monkeypatch.setattr(gate, "compare_results", lambda *_: {})
    monkeypatch.setattr(gate, "compare_queries", lambda *_: {"max_nearest_distance_error_m": 0.})
    calls = []
    original = gate.align_observed_points
    def fail_after_first_trial(*args):
        calls.append(1)
        if len(calls) == 3: raise RuntimeError("timing interruption")
        return original(*args)
    monkeypatch.setattr(gate, "align_observed_points", fail_after_first_trial)
    diagnostics = {}
    with pytest.raises(RuntimeError, match="timing interruption"):
        gate.run_gate(diagnostics)
    assert len(diagnostics["timings_seconds"]["original_rebuild"]) == 1
    assert len(diagnostics["timings_seconds"]["canonical_cached"]) == 2
    assert diagnostics["current_timing_trial"] == {"trial": 1, "mode": "original_rebuild", "alignments_completed": 0}
    assert diagnostics["throughput_hypothesis_accepted"] is False

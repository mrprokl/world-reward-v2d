"""Pure planning/geometry/loss tests only; no Torch or CUDA fitter execution."""

import builtins
import inspect

import numpy as np
import pytest

import world_reward.shape_fit_continuous as fitter
from world_reward.shape_model import deformation


@pytest.fixture
def inputs():
    vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [0., 0., .01]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    observations = [np.array([[0., 0., 3.], [.002, 0., 3.], [0., .002, 3.]])]
    return vertices, faces, vertices.mean(0), observations, np.eye(3)[None], np.array([[0., 0., 3.]])


def test_input_copies_fixed_poses_topology_pivot_and_observation_indices(inputs):
    prepared = fitter.prepare_inputs(*inputs)
    for old, new in zip(inputs[:3], prepared[:3], strict=True):
        np.testing.assert_array_equal(old, new)
        assert not np.shares_memory(old, new)
    np.testing.assert_array_equal(prepared[3][0], inputs[3][0])
    np.testing.assert_array_equal(prepared[4], inputs[4])
    np.testing.assert_array_equal(prepared[5], inputs[5])
    np.testing.assert_array_equal(prepared[6][0], [0, 1, 2])
    assert prepared[7] == [3]


def test_observations_uniform_deterministic_cap2048_no_hidden_filling(inputs):
    x = np.arange(3000, dtype=float)
    observed = np.column_stack((x, x % 17, np.full(3000, 3.)))
    prepared = fitter.prepare_inputs(*inputs[:3], [observed], *inputs[4:])
    expected = np.linspace(0, 2999, 2048, dtype=np.int64)
    np.testing.assert_array_equal(prepared[6][0], expected)
    np.testing.assert_array_equal(prepared[3][0], observed[expected])
    assert prepared[7] == [3000]


@pytest.mark.parametrize("failure", ["padding", "negative", "outside", "float", "zeroarea", "masked"])
def test_bad_active_face_geometry_fails_before_torch(inputs, failure):
    values = list(inputs)
    faces = values[1].copy()
    if failure == "padding": faces[0] = 0
    if failure == "negative": faces[0, 0] = -1
    if failure == "outside": faces[0, 0] = 4
    if failure == "float": faces = faces.astype(float)
    if failure == "zeroarea": values[0] = values[0].copy(); values[0][2] = values[0][1]
    if failure == "masked": faces = np.ma.array(faces, mask=False)
    values[1] = faces
    with pytest.raises(ValueError): fitter.prepare_inputs(*values)


@pytest.mark.parametrize("failure", ["rotation_scale", "reflection", "nanpose", "badcount", "maskedpose", "emptyobs", "rank1", "nanobs"])
def test_bad_observations_or_supplied_poses_never_repaired(inputs, failure):
    values = list(inputs)
    if failure == "rotation_scale": values[4] = np.eye(3)[None] * 2
    if failure == "reflection": values[4] = np.diag([1., 1., -1.])[None]
    if failure == "nanpose": values[5] = np.array([[0., 0., np.nan]])
    if failure == "badcount": values[4] = np.stack([np.eye(3)] * 2)
    if failure == "maskedpose": values[4] = np.ma.array(values[4], mask=False)
    if failure == "emptyobs": values[3] = []
    if failure == "rank1": values[3] = [np.array([[0., 0., 3.], [1., 0., 3.], [2., 0., 3.]])]
    if failure == "nanobs": values[3] = [values[3][0].copy()]; values[3][0][0, 0] = np.nan
    with pytest.raises(ValueError): fitter.prepare_inputs(*values)


def test_isotropic_robust_cost_equal_frame_weights_and_vector_derivative():
    error = np.array([.03, -.02, .01])
    squared = error @ error
    first = fitter.isotropic_robust_cost_numpy(np.full(3, squared))
    second = fitter.isotropic_robust_cost_numpy(np.full(31, squared))
    assert first == pytest.approx(second)
    rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert fitter.isotropic_robust_cost_numpy(np.array([(rotation @ error) @ (rotation @ error)])) == pytest.approx(first)
    expected = 2 * error / np.sqrt(1 + squared / .01**2)
    for axis in range(3):
        delta = np.eye(3)[axis] * 1e-7
        fd = (fitter.isotropic_robust_cost_numpy(np.array([(error + delta) @ (error + delta)]))
              - fitter.isotropic_robust_cost_numpy(np.array([(error - delta) @ (error - delta)]))) / 2e-7
        assert fd == pytest.approx(expected[axis], abs=1e-12)
    assert fitter.isotropic_robust_cost_numpy(np.zeros(4)) == 0.


def test_conservative_parameter_box_proves_spd_fixed_volume_spectral_bound():
    assert fitter.PARAMETER_BOUND == np.log(1.5) / 4
    import itertools
    for signs in itertools.product((-1, 1), repeat=5):
        params = np.array(signs) * fitter.PARAMETER_BOUND
        matrix = deformation(params)
        eig = np.linalg.eigvalsh(matrix)
        assert eig.min() >= 1 / 1.5 - 1e-12 and eig.max() <= 1.5 + 1e-12
        assert np.linalg.det(matrix) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize("kwargs", [{"shape_prior": True}, {"shape_prior": -1}, {"shape_prior": np.nan},
                                     {"robust_transition_m": False}, {"robust_transition_m": 0}, {"robust_transition_m": np.inf},
                                     {"robust_transition_m": 1e-200}, {"max_evaluations": True}, {"max_evaluations": 0},
                                     {"max_evaluations": 101}, {"max_evaluations": 1.5}, {"point_chunk_size": False},
                                     {"point_chunk_size": 0}, {"point_chunk_size": "64"}])
def test_invalid_controls_fail_before_any_torch_import(inputs, kwargs, monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **other):
        if name == "torch": pytest.fail("Heavy import before pure validation")
        return original(name, *args, **other)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError): fitter.fit_shared_shape_continuous(*inputs, **kwargs)


def test_defaults_and_hard_budget_architecture_no_pose_optimizer_or_autoapply():
    signature = inspect.signature(fitter.fit_shared_shape_continuous)
    assert signature.parameters["shape_prior"].default == .01
    assert signature.parameters["robust_transition_m"].default == .01
    assert signature.parameters["max_evaluations"].default == 100
    source = inspect.getsource(fitter.fit_shared_shape_continuous)
    assert "raise _EvaluationLimit" in source and "torch.cuda.is_available()" in source
    assert "external_reserved_frame_adoption_gate_required" in source and '"adoption_authorized": False' in source
    assert '"pose_uncertainty_not_modeled": True' in source

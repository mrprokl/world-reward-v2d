"""Tiny analytic H97 tests; no images/models/private observations."""
import numpy as np
import pytest

from world_reward import root_refit as prior, translation_refit as p


def fixture():
    points = np.c_[np.linspace(-.4, .6, 10), np.linspace(.2, -.5, 10)**2-.3, np.linspace(2.5, 3.5, 10)]
    origin = np.array([.1, -.2, 3.])
    k = np.array([[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]])
    return points, origin, k


def test_additive_translation_not_vertex_depth_scaling():
    points, origin, k = fixture(); u = np.array([.2, -.3, .5])
    xy, j = p.project_and_jacobian(points, origin, u, k)
    translated = points+p.camera_translation(origin, p.bounded_delta(u))-origin
    expected = translated[:, :2]/translated[:, 2:]*[1280, 1280]+[512, 384]
    assert np.allclose(xy, expected, atol=1e-12, rtol=0) and j.shape == (20, 3)
    assert not np.allclose(translated[:, 2], points[:, 2]*np.exp(p.bounded_delta(u)[2]))


@pytest.mark.parametrize("u", [np.zeros(3), np.array([.2, -.3, .5]), np.array([-1., 1., -.7])])
def test_analytic_jacobian_finite_difference(u):
    points, origin, k = fixture(); xy, j = p.project_and_jacobian(points, origin, u, k)
    eps = 1e-6
    fd = np.column_stack([(p.project_and_jacobian(points, origin, u+np.eye(3)[i]*eps, k)[0]
                           -p.project_and_jacobian(points, origin, u-np.eye(3)[i]*eps, k)[0]).reshape(-1)/(2*eps) for i in range(3)])
    assert np.allclose(j, fd, atol=1e-7, rtol=1e-7)
    evidence = p.jacobian_evidence(j, 10)
    assert evidence["prior_rows_included"] is False and len(evidence["singular_values"]) == 3


@pytest.mark.parametrize("residual", [0., .2, 20.])
def test_analytic_full_objective_gradient(residual):
    points, origin, k = fixture(); u = np.array([.2, -.3, .5]); xy, j = p.project_and_jacobian(points, origin, u, k)
    target = xy+residual
    value, grad = p.objective_and_gradient(xy, target, p.bounded_delta(u), j, u)
    assert value == p.objective(xy, target, p.bounded_delta(u))
    def loss(z):
        proj, jac = p.project_and_jacobian(points, origin, z, k)
        return p.objective_and_gradient(proj, target, p.bounded_delta(z), jac, z)[0]["total"]
    eps = 1e-6
    fd = np.array([(loss(u+np.eye(3)[i]*eps)-loss(u-np.eye(3)[i]*eps))/(2*eps) for i in range(3)])
    assert np.allclose(grad, fd, atol=1e-5, rtol=1e-6)


def test_prior_loss_observation_gates_unchanged():
    xy = np.arange(266, dtype=float).reshape(133, 2)*3; scores = np.ones(133)
    assert all(np.array_equal(a, b) for a, b in zip(p.training_observations(xy, scores), prior.training_observations(xy, scores)))
    projected, target = xy[:10], xy[10:20]; d = np.array([.03, -.06, .02])
    assert p.objective(projected, target, d) == prior.objective(projected, target, np.r_[d, np.zeros(3)])


def test_manual_adam_constants_and_observed_update_count():
    u = np.zeros(3); m = u.copy(); v = u.copy(); g = np.array([1., -2., 0.])
    first, m, v = p.adam_step(u, g, m, v, 1)
    assert np.allclose(first, -.01*g/(np.abs(g)+1e-8), atol=1e-14, rtol=0)
    for i in range(2, 60): first, m, v = p.adam_step(first, g, m, v, i)
    assert np.allclose(first, -.59*g/(np.abs(g)+1e-8), atol=1e-12, rtol=0)
    with pytest.raises(ValueError): p.adam_step(first, g, m, v, 60)
    with pytest.raises(ValueError): p.adam_step(first, g, m, -v-1, 1)


def test_steady_replay_zero_start_and_best_first_tie():
    points, origin, k = fixture(); target = p.project_and_jacobian(points, origin, np.array([.3, -.2, .1]), k)[0]
    def run():
        u = np.zeros(3); m = u.copy(); v = u.copy(); losses = []
        for state in range(60):
            xy, j = p.project_and_jacobian(points, origin, u, k)
            val, grad = p.objective_and_gradient(xy, target, p.bounded_delta(u), j, u)
            losses.append(val["total"])
            if state < 59: u, m, v = p.adam_step(u, grad, m, v, state+1)
        return np.array(losses)
    a, b = run(), run(); assert np.array_equal(a, b) and a[p.best_evaluated(a)] < a[0]
    ties = np.ones(60); ties[5] = ties[9] = 0.; assert p.best_evaluated(ties) == 5


@pytest.mark.parametrize("fault", ["masked", "nan", "shape", "z", "skew", "overflow", "behind"])
def test_projection_invalid_fails_without_repair(fault):
    points, origin, k = fixture(); u = np.zeros(3)
    if fault == "masked": points = np.ma.array(points, mask=False)
    elif fault == "nan": points[0, 0] = np.nan
    elif fault == "shape": u = np.zeros(6)
    elif fault == "z": origin[2] = 0
    elif fault == "skew": k[0, 1] = 1
    elif fault == "overflow": points[0, 0] = np.finfo(float).max
    else: points[0, 2] = -1
    with pytest.raises(ValueError): p.project_and_jacobian(points, origin, u, k)


@pytest.mark.parametrize("fault", ["masked", "few", "extent", "overflow"])
def test_observation_rejection(fault):
    xy = np.arange(266, dtype=float).reshape(133, 2); scores = np.ones(133)
    if fault == "masked": xy = np.ma.array(xy, mask=False)
    elif fault == "few": scores[:] = 0
    elif fault == "extent": xy[:] = 1
    else: xy[5] = -np.finfo(float).max; xy[6] = np.finfo(float).max
    with pytest.raises(ValueError): p.training_observations(xy, scores)


def test_rank_without_prior_and_delta_integrity():
    with pytest.raises(ValueError): p.jacobian_evidence(np.zeros((20, 3)), 10)
    points, origin, k = fixture(); u = np.zeros(3); xy, j = p.project_and_jacobian(points, origin, u, k)
    with pytest.raises(ValueError): p.objective_and_gradient(xy, xy, np.array([.1, 0., 0.]), j, u)
    with pytest.raises(ValueError): p.physical_delta(np.array([.31, 0., 0.]))


def test_quality_gates_identical_to_failed_six_dof_policy_not_fresh_replication():
    for human in (np.array([[10., 9.], [20., 18.], [30., 29.]]), np.array([[0., 0.], [1., 1.], [2., 2.]])):
        hands = np.ones((3, 2, 2)); hands[1, 1, 0] = 1.06
        a, b = p.quality_decision(human, hands), prior.quality_decision(human, hands)
        assert a["gates"] == b["gates"] and a["per_clip_camera_pve_relative_gain"] == b["per_clip_camera_pve_relative_gain"]
        assert not a["independent_replication_verified"] and not a["adoption_authorized"]

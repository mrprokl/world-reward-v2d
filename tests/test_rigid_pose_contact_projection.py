"""Data-free exact geometry/derivative/whole-clip rigid projection contracts."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import rigid_pose_contact_projection as core


def tiny(count=17, *, near=False):
    vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.08, .08) for z in (-.15, .15)], float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    phase = np.arange(count)
    r = Rotation.from_rotvec(np.c_[phase * 0., phase * 0., .01 * phase]).as_matrix()
    if near: r[:, :, 0] *= 1 + 2e-6
    t = np.c_[.004 * phase, .01 * np.sin(phase / 3), phase * 0. + 2]
    hands = np.einsum('tij,pj->tpi', r, np.array([[-.52, .02, 0], [.52, -.02, 0]])) + t[:, None]
    active = np.ones((count, 2), bool); active[6:9] = False; hands[~active] = np.nan
    sr = Rotation.from_rotvec(np.tile([0., 0., .2], (count, 1))).as_matrix() @ r
    st = t + [.002, .002, 0]; sr[0], st[0] = r[0], t[0]
    return vertices, faces, r, t, sr, st, hands, active, phase.astype(np.int64)


def test_runtime_authored_two_seeds_moving_bimanual_release_capacity_not_actual_gain():
    result = core.authored_DEV()
    assert result['passed'] and not result['ground_truth_challenge_used']
    assert [r['seed'] for r in result['rows']] == [20261010, 20261011]
    assert all(r['objective_after'] < .4 * r['objective_before'] for r in result['rows'])
    assert all(r['inactive_rows_retained'] == 8 for r in result['rows'])


def test_closed_form_full_mesh_objective_and_gradients_equal_direct_vertices_finite_differences():
    v, _, r, t, sr, st, *_ = tiny(near=True); v += [.3, -.4, .2]
    centre, covariance = core._moments(v)
    value, gradient, metric = core._objective(centre, covariance, r, t, sr, st)
    full = np.einsum('tij,pj->tpi', r, v) + t[:, None]
    target = np.einsum('tij,pj->tpi', sr, v) + st[:, None]
    assert value == pytest.approx(.5 * np.square(full - target).sum() / len(v), abs=1e-14)
    assert np.all(metric > 0)
    for f in (1, 9, 16):
        for axis in range(6):
            step = np.zeros((len(r), 6)); step[f, axis] = 1e-6
            pr, pt = core._retract(r, t, step, centre)
            mr, mt = core._retract(r, t, -step, centre)
            plus = core._objective(centre, covariance, pr, pt, sr, st)[0]
            minus = core._objective(centre, covariance, mr, mt, sr, st)[0]
            assert (plus - minus) / 2e-6 == pytest.approx(gradient[f, axis], abs=2e-10)
    np.testing.assert_array_equal(gradient[0], np.zeros(6))


def test_contact_rotation_and_translation_normal_gradient_matches_full_surface_FD_near_SO3():
    v, f, r, t, _, _, h, a, _ = tiny(near=True)
    surface = core._NearestSurface(v, f); centre, _ = core._moments(v)
    contacts = core._contacts(surface, h, a, r, t, lambda: None)
    rows, bounds = core._rows(surface, h, a, r, t, centre, *contacts, contacts[0] + 1e-7)
    frame = 4
    for axis in range(6):
        step = np.zeros((len(r), 6)); step[frame, axis] = 1e-6
        pr, pt = core._retract(r, t, step, centre); mr, mt = core._retract(r, t, -step, centre)
        for side in (0, 1):
            plus = surface.closest((h[frame, side] - pt[frame]) @ pr[frame])[0]
            minus = surface.closest((h[frame, side] - mt[frame]) @ mr[frame])[0]
            assert (plus - minus) / 2e-6 == pytest.approx(rows[frame][side, axis], abs=2e-10)
    assert np.allclose(bounds[frame], 1e-7)


def test_zero_gap_cusp_gets_two_unsigned_rows_and_never_invents_signed_penetration():
    v, f, r, t, _, _, h, a, _ = tiny()
    h[1, 0] = r[1] @ [.5, .01, 0.] + t[1]
    surface = core._NearestSurface(v, f); centre, _ = core._moments(v)
    contacts = core._contacts(surface, h, a, r, t, lambda: None)
    rows, _ = core._rows(surface, h, a, r, t, centre, *contacts, contacts[0] + 1e-7)
    assert rows[1].shape == (3, 6)
    np.testing.assert_allclose(rows[1][0], -rows[1][1], atol=0)


def test_six_dimension_embedding_exact_unconstrained_box_QP_dummy_never_native_fit():
    rng = np.random.default_rng(43); count = 17
    g = rng.normal(size=(count, 6)); metric = rng.uniform(.1, 2., (count, 6))
    g[0] = 0; rows = [np.empty((0, 6)) for _ in range(count)]; bounds = [np.empty(0) for _ in range(count)]
    step, receipt = core._qp(g, metric, rows, bounds, .5, core.RigidContactProjectionConfig())
    expected = np.clip(-g / metric, -.5 * np.r_[np.full(3, .05), np.full(3, .01)], .5 * np.r_[np.full(3, .05), np.full(3, .01)])
    np.testing.assert_array_equal(step, expected)
    assert receipt['optimized_state_dimensions'] == 6 and not receipt['native_body_fitted']


@pytest.mark.parametrize('near', [False, True])
def test_all_original_frames_gauge_geometry_and_contact_bounds_preserved_no_input_mutation(near):
    inputs = tiny(near=near); originals = [v.copy() for v in inputs]
    result = core.project_rigid_pose_to_contacts(*inputs)
    assert result.diagnostics['accepted_whole_clip_steps'] > 0
    assert result.diagnostics['objective_after'] < result.diagnostics['objective_before']
    assert np.all(result.gaps_m[inputs[7]] <= result.baseline_gaps_m[inputs[7]] + 1e-7)
    assert np.isnan(result.gaps_m[~inputs[7]]).all()
    assert np.array_equal(result.rotation[0], inputs[2][0]) and np.array_equal(result.translation[0], inputs[3][0])
    assert len(result.translation) == len(inputs[-1]) and np.linalg.norm(result.translation[-1] - result.translation[0]) > .04
    np.testing.assert_allclose(result.rotation.transpose(0, 2, 1) @ result.rotation,
        inputs[2].transpose(0, 2, 1) @ inputs[2], atol=2e-15)
    for v, old in zip(inputs, originals): np.testing.assert_array_equal(v, old)
    assert not result.diagnostics['production_adopted'] and not result.diagnostics['physical_contact_verified']
    assert result.diagnostics['human_fixed'] and result.diagnostics['RGB_motion_floor_QA_pending']


def test_whole_clip_failed_QP_keeps_original_dynamic_A_byte_exact_no_perframe_selection(monkeypatch):
    inputs = list(tiny()); inputs[2] = inputs[2].astype(np.float32); inputs[3] = inputs[3].astype(np.float32)
    def fail(*_): raise core.QPProjectionFailure('manufactured local failure, not physical infeasibility')
    monkeypatch.setattr(core, '_qp', fail)
    result = core.project_rigid_pose_to_contacts(*inputs)
    assert result.diagnostics['status'] == 'dynamic_A_fallback_no_improvement'
    np.testing.assert_array_equal(result.rotation, inputs[2]); np.testing.assert_array_equal(result.translation, inputs[3])
    assert result.rotation.dtype == inputs[2].dtype and result.translation.dtype == inputs[3].dtype
    assert len(result.diagnostics['attempts']) == 20
    assert all(a['physical_infeasibility_claimed'] is False for a in result.diagnostics['attempts'])


def test_over_budget_proposal_is_discarded_retaining_whole_dynamic_A(monkeypatch):
    inputs = tiny(); clock = [0.]; original = core._qp
    monkeypatch.setattr(core.time, 'monotonic', lambda: clock[0])
    def slow(*args):
        answer = original(*args); clock[0] = 121.; return answer
    monkeypatch.setattr(core, '_qp', slow)
    result = core.project_rigid_pose_to_contacts(*inputs)
    assert result.diagnostics['budget_exhausted'] and result.diagnostics['accepted_whole_clip_steps'] == 0
    np.testing.assert_array_equal(result.rotation, inputs[2]); np.testing.assert_array_equal(result.translation, inputs[3])
    assert result.diagnostics['attempts'][-1]['status'] == 'budget_exhausted_partial_proposal_discarded'


def test_no_active_contact_still_retains_full_moving_chronology_not_static_or_deleted():
    inputs = list(tiny()); inputs[7][:] = False; inputs[6][:] = np.nan
    result = core.project_rigid_pose_to_contacts(*inputs)
    assert result.diagnostics['accepted_whole_clip_steps'] > 0
    assert result.diagnostics['original_active_witness_rows'] == 0
    assert np.isnan(result.gaps_m).all() and len(result.translation) == 17
    assert np.linalg.norm(result.translation[-1] - result.translation[0]) > .04


def test_whole_object_camera_depth_is_checked_before_original_access_and_candidate_acceptance(monkeypatch):
    bad = list(tiny()); bad[3][:, 2] = .1
    with pytest.raises(ValueError, match='positive camera-depth'): core.project_rigid_pose_to_contacts(*bad)
    inputs = list(tiny()); inputs[7][:] = False; inputs[6][:] = np.nan
    original = core._object_in_front; calls = []
    def reject_candidate(*args):
        calls.append(1)
        if len(calls) > 1 and not np.array_equal(args[2], inputs[3]): return False
        return original(*args)
    monkeypatch.setattr(core, '_object_in_front', reject_candidate)
    result = core.project_rigid_pose_to_contacts(*inputs)
    assert result.diagnostics['accepted_whole_clip_steps'] == 0
    assert any(row['status'] == 'whole_object_camera_depth_rejected' for row in result.diagnostics['attempts'])
    np.testing.assert_array_equal(result.translation, inputs[3])


@pytest.mark.parametrize('fault', ['frames', 'floatframes', 'active', 'masked', 'inactivefinite', 'activenan',
    'Rinvalid', 'Rreflected', 'nonfiniteT', 'faceindex', 'vertices', 'wrongtarget'])
def test_invalid_observations_fail_closed_without_repair_or_contact_drop(fault):
    values = list(tiny())
    if fault == 'frames': values[-1][4] += 1
    elif fault == 'floatframes': values[-1] = values[-1].astype(float)
    elif fault == 'active': values[7] = values[7].astype(np.int64)
    elif fault == 'masked': values[6] = np.ma.array(values[6])
    elif fault == 'inactivefinite': values[6][~values[7]] = 0
    elif fault == 'activenan': values[6][1, 0] = np.nan
    elif fault == 'Rinvalid': values[2][1, 0, 0] += .01
    elif fault == 'Rreflected': values[2][:, :, 0] *= -1
    elif fault == 'nonfiniteT': values[3][1, 0] = np.nan
    elif fault == 'faceindex': values[1][0, 0] = len(values[0])
    elif fault == 'vertices': values[0][0, 0] = np.nan
    elif fault == 'wrongtarget': values[4] = values[4][:-1]
    with pytest.raises(ValueError): core.project_rigid_pose_to_contacts(*values)


def test_policy_not_replaced_with_video_weight_or_loosened_contact_slack():
    cfg = core.RigidContactProjectionConfig()
    for change in (dict(max_steps=21), dict(rotation_trust_rad=.06), dict(translation_trust_m=.02),
            dict(numerical_slack_m=1e-6), dict(max_restorations=5), dict(budget_seconds=121)):
        with pytest.raises(ValueError): replace(cfg, **change)

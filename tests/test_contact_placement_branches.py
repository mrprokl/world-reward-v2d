"""Analytic multicomponent tests; no real data or claimed physical truth."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import contact_feasible_placement as single
from world_reward import contact_placement_branches as branch
from world_reward.continuous_surface import numpy_reference_distance_squared


def config():
    return single.ContactPlacementConfig(.05, 1e-7, 1e-10, 400, 8, 'analytic_branch_unit_not_quality')


def three_parts(count=1):
    yz = np.array([[-2., -2], [2., -2], [2., 2], [-2., 2]])
    v = np.concatenate([np.c_[np.full(4, x), yz] for x in (0., 1., 3.)])
    f = np.array([[4*i, 4*i+1, 4*i+2] for i in range(3)]+[[4*i, 4*i+2, 4*i+3] for i in range(3)])
    R = np.tile(np.eye(3), (count, 1, 1)); t = np.zeros((count, 3)); ids = np.array([[0], [1]], np.int64)
    points = np.tile(np.array([[[1.01, 0., 0.]], [[3.01, 0., 0.]]]), (count, 1, 1, 1))
    active = np.ones((count, 2), bool); frames = np.arange(count, dtype=np.int64)
    witness = points[:, :, 0].copy()
    evidence = single.freeze_contact_witnesses(witness, ids, np.tile(ids[:, 0], (count, 1)),
        R, t, v, f, active, frames, config=config(), source_reference='analytic_original_only',
        witness_selection_reference='analytic_saved_original_anatomical_QA', original_selection_is_whole_hand_minimum=False)
    original = branch.freeze_original_branches(evidence, witness, t)
    proposal = points.copy(); proposal[..., 0] -= 1.61
    return evidence, original, proposal, t


def test_disjoint_current_nearest_parts_fail_but_original_branches_recover():
    e, original, proposed, t = three_parts()
    with pytest.raises(single.ContactFeasibilityFailure): single.project_joint_translations(e, proposed, t, t)
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    correction = 1.6-e.baseline_gap_m[0, 0]-config().numerical_slack_m+2*config().convergence_tolerance_m
    np.testing.assert_allclose(result['relative_correction_camera'], [[correction, 0., 0.]], atol=2e-15)
    row = result['branch_diagnostics'][0]
    assert row['branches_attempted'] == 4 and row['branches_feasible'] == 1
    assert row['attempts'][0]['status'] == 'convex_branch_disjoint_AABB_certificate'
    assert row['attempts'][0]['branch_separation_lower_bound_m'] > .97
    assert row['selected_initial_faces'] != row['current_nearest_faces']
    local = proposed[0, :, 0]+result['relative_correction_camera'][0]
    exact = np.sqrt(numpy_reference_distance_squared(local, e.object_vertices[e.object_faces]))
    assert np.all(exact <= result['frozen_gap_bound_m'][0])
    assert not result['global_minimum_verified'] and not result['physical_contact_verified']
    assert not result['original_whole_hand_minimum_nondegradation_bound']
    assert result['max_initial_branches'] == 4


def test_smallest_verified_correction_selected_over_all_converged_branch_candidates():
    e, original, proposed, t = three_parts()
    proposed[0, 0, 0, 0] = .4; proposed[0, 1, 0, 0] = 1.4
    # Current branches0/1 admit correction-.39; original1/3 are incompatible.
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    expected = -.4+e.baseline_gap_m[0, 0]+config().numerical_slack_m-2*config().convergence_tolerance_m
    assert result['relative_correction_camera'][0, 0] == pytest.approx(expected, abs=2e-15)
    row = result['branch_diagnostics'][0]
    assert row['branches_attempted'] == 4
    verified = [r['camera_correction_norm_m'] for r in row['attempts'] if r['status'] == 'converged_complete_surface_verified']
    assert np.linalg.norm(result['relative_correction_camera'][0]) == min(verified)


def test_original_gap_bound_not_target_anchor_point_or_fixed_triangle_minimum():
    e, original, proposed, t = three_parts()
    proposed[0, 0, 0] = [1.005, .7, -.8]; proposed[0, 1, 0] = [3.005, -.9, .6]
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    assert not result['relative_correction_camera'].any()
    assert result['branch_diagnostics'] == []
    assert not result['projection_cycles'].any()
    assert original.anchor_points_object[0, 0, 1] == 0.
    assert proposed[0, 0, 0, 1] == .7


def test_full_original_common_motion_inactive_release_and_inputs_immutable():
    e, original, proposed, t = three_parts(count=5)
    active = e.activations.copy(); active[2] = False
    witness = np.tile(np.array([[1.01, 0., 0.], [3.01, 0., 0.]]), (5, 1, 1)); witness[2] = np.nan
    ids = e.witness_source_indices.copy(); ids[2] = -1
    e = single.freeze_contact_witnesses(witness, e.source_indices, ids, e.object_rotation, t,
        e.object_vertices, e.object_faces, active, e.frame_index, config=config(),
        source_reference='analytic_original_only', witness_selection_reference='analytic_saved_QA')
    original = branch.freeze_original_branches(e, witness, t)
    common = np.c_[np.arange(5)*.2, np.arange(5)**2*.02, np.arange(5)*.03]
    saved = proposed.copy(); proposed += common[:, None, None]
    result = branch.project_joint_translations_branches(e, proposed, common, common, original_branches=original)
    np.testing.assert_allclose((result['human_translation_camera']+result['object_translation_camera'])/2, common, atol=1e-16)
    np.testing.assert_array_equal(result['human_translation_camera'][2], common[2])
    np.testing.assert_array_equal(result['object_translation_camera'][2], common[2])
    assert np.isnan(result['witness_gap_after_m'][2]).all()
    assert np.linalg.norm(result['object_translation_camera'][-1]-result['object_translation_camera'][0]) > .8
    np.testing.assert_array_equal(e.activations, active)
    np.testing.assert_array_equal(result['human_correction_camera'], -result['object_correction_camera'])
    for value in result.values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError): value.setflags(write=True)
    proposed[:] = 88.
    assert result['witness_gap_before_m'][0, 0] == pytest.approx(.6)


def test_failed_all_branches_explicit_no_global_or_physical_infeasibility_no_disabling():
    e, original, proposed, t = three_parts()
    proposed[0, 0, 0, 0] = .4; proposed[0, 1, 0, 0] = 5.4
    stored = proposed.copy()
    with pytest.raises(branch.BranchPlacementFailure) as caught:
        branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    assert caught.value.frame_index == 0 and caught.value.maximum_violation_m > 1.
    assert len(caught.value.branch_attempts) <= 4
    assert 'not a global or physical infeasibility proof' in caught.value.reason
    assert all(r['physical_infeasibility_claimed'] is False for r in caught.value.branch_attempts)
    np.testing.assert_array_equal(proposed, stored); assert e.activations.all()


def test_original_anchors_source_bindings_reject_mismatched_inputs_or_proposal_anchor():
    e, original, proposed, t = three_parts()
    bad = np.array([[[.4, 0., 0.], [2.4, 0., 0.]]])
    with pytest.raises(ValueError): branch.freeze_original_branches(e, bad, t)
    with pytest.raises(ValueError): branch.project_joint_translations_branches(
        e, proposed, t, t, original_branches=replace(original, evidence_sha256='0'*64))
    for value in (original.face_indices, original.anchor_points_object):
        with pytest.raises(ValueError): value.setflags(write=True)
    assert original.face_indices.dtype == np.int64


def test_rigid_camera_frame_equivariance_no_original_matrix_repair():
    e, original, proposed, t = three_parts()
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    R = Rotation.from_rotvec([.2, -.3, .1]).as_matrix(); offset = np.array([1., 2., 3.])
    witness = np.array([[[1.01, 0., 0.], [3.01, 0., 0.]]])@R.T+offset
    transformed = single.freeze_contact_witnesses(witness, e.source_indices, e.witness_source_indices,
        R[None], offset[None], e.object_vertices, e.object_faces, e.activations, e.frame_index,
        config=config(), source_reference='analytic_original_only', witness_selection_reference='analytic_saved_QA')
    anchors = branch.freeze_original_branches(transformed, witness, offset[None])
    placed = branch.project_joint_translations_branches(transformed, proposed@R.T+offset,
        t@R.T+offset, t@R.T+offset, original_branches=anchors)
    np.testing.assert_allclose(placed['relative_correction_camera'], result['relative_correction_camera']@R.T, atol=2e-14)
    assert placed['fixed_object_rotation_repaired'] is False


def test_no_contact_frames_all_retained_without_distance_or_projection_work(monkeypatch):
    e, original, proposed, t = three_parts(count=3)
    e = replace(e, activations=np.zeros((3, 2), bool), witness_columns=np.full((3, 2), -1),
        baseline_gap_m=np.full((3, 2), np.nan), witness_source_indices=np.full((3, 2), -1))
    original = branch.freeze_original_branches(e, np.full((3, 2, 3), np.nan), t)
    def forbidden(*args, **kwargs): raise AssertionError('No active hand: no contact query permitted')
    monkeypatch.setattr(single._NearestSurface, 'closest', forbidden)
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    np.testing.assert_array_equal(result['frame_index'], np.arange(3))
    np.testing.assert_array_equal(result['human_translation_camera'], t)
    assert not result['projection_cycles'].any() and not result['activations'].any()


def test_same_numerical_caps_bound_every_branch_and_no_parameter_retry():
    e, original, proposed, t = three_parts()
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    for frame in result['branch_diagnostics']:
        assert frame['branches_attempted'] <= 4
        for row in frame['attempts']:
            assert row['projection_cycles'] <= config().max_projection_cycles*config().max_face_updates
            assert row['face_updates'] <= config().max_face_updates
    assert result['numerical_interior_margin_m'] == 2*config().convergence_tolerance_m
    np.testing.assert_array_equal(result['frozen_gap_bound_m'], e.baseline_gap_m+config().numerical_slack_m)


def test_native_near_rotation_no_polar_repair_actual_camera_emit_passes():
    e, original, proposed, t = three_parts()
    rotation = np.eye(3, dtype=np.float32); rotation[0, 0] += np.float32(1e-6)
    witness = np.array([[[1.01, 0., 0.], [3.01, 0., 0.]]])
    e = single.freeze_contact_witnesses(witness, e.source_indices, e.witness_source_indices,
        rotation[None], t, e.object_vertices, e.object_faces, e.activations, e.frame_index,
        config=config(), source_reference='analytic_native_FP32_only', witness_selection_reference='original_QA')
    original = branch.freeze_original_branches(e, witness, t)
    result = branch.project_joint_translations_branches(e, proposed, t, t, original_branches=original)
    actual = (proposed[0, :, 0]+result['human_translation_camera'][0]-result['object_translation_camera'][0])@rotation
    gaps = np.sqrt(numpy_reference_distance_squared(actual, e.object_vertices[e.object_faces]))
    assert np.all(gaps <= result['frozen_gap_bound_m'][0])
    assert result['object_rotation_orthogonality_error'] > 0 and not result['fixed_object_rotation_repaired']


def test_degenerate_segments_retained_in_original_current_search_fullsurface():
    vertices = np.array([[0., -1, 0], [0, 1, 0], [0, 1, 0], [1., -1, 0], [1, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    active = np.array([[True, False]]); ids = np.array([[0], [1]], np.int64)
    witness = np.array([[[.01, 0., 0.], [np.nan, np.nan, np.nan]]]); T = np.zeros((1, 3))
    e = single.freeze_contact_witnesses(witness, ids, np.array([[0, -1]]), np.eye(3)[None], T,
        vertices, faces, active, np.array([0]), config=config(), source_reference='analytic_segment_only',
        witness_selection_reference='original_QA')
    original = branch.freeze_original_branches(e, witness, T)
    proposed = np.array([[[[.3, 0., 0.]], [[.3, 0., 0.]]]])
    result = branch.project_joint_translations_branches(e, proposed, T, T, original_branches=original)
    assert result['witness_gap_after_m'][0, 0] <= result['frozen_gap_bound_m'][0, 0]
    assert e.object_faces.shape == (2, 3)

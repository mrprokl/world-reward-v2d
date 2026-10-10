"""Manufactured numerical contracts, NOT challenge or external quality scores."""
from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.contact_feasible_placement import (ContactPlacementConfig,
    ContactFeasibilityFailure, freeze_contact_evidence, freeze_contact_witnesses,
    project_joint_translations,
    _NearestSurface, _closest_triangle_points)
from world_reward.continuous_surface import numpy_reference_distance_squared


def config():
    return ContactPlacementConfig(.05, 1e-7, 1e-10, 400, 8,
                                  'manufactured:unit_contract_not_quality')


def inputs(count=4, joints=3):
    vertices = np.array([[-10., -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    points = np.zeros((count, 2, joints, 3)); points[..., 2] = .02
    points[:, :, 0, 2] = .01
    ids = np.arange(2*joints).reshape(2, joints)
    active = np.zeros((count, 2), bool); active[:, 0] = True
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    translation = np.zeros((count, 3)); frames = np.arange(count, dtype=np.int64)
    return [points, ids, rotation, translation, vertices, faces, active, frames]


def freeze(args):
    return freeze_contact_evidence(*args, config=config(),
        source_reference='manufactured:automatic_native_gates_not_a_real_run')


def test_config_explicit_immutable_and_numerical_not_optimized_weights():
    c = config()
    with pytest.raises(FrozenInstanceError): c.max_face_updates = 2
    for change in ({'activation_distance_m': 0}, {'numerical_slack_m': np.nan},
                   {'convergence_tolerance_m': 1e-5}, {'max_projection_cycles': True},
                   {'max_face_updates': 33}, {'development_reference': ' '}):
        with pytest.raises(ValueError): replace(c, **change)


def test_original_whole_hand_witness_frozen_owned_and_ties_use_anatomical_id():
    args = inputs(); args[0][:, 0, 1, 2] = .01
    args[1][0] = [30, 7, 20]
    evidence = freeze(args)
    assert (evidence.witness_columns[:, 0] == 1).all()
    assert (evidence.witness_source_indices[:, 0] == 7).all()
    assert (evidence.witness_columns[:, 1] == -1).all()
    assert np.isnan(evidence.baseline_gap_m[:, 1]).all()
    assert (evidence.baseline_gap_m[:, 0] == .01).all()
    args[1][:] = 999; args[0][:] = 888; args[4][:] = 777; args[6][:] = False
    assert (evidence.witness_source_indices[:, 0] == 7).all()
    assert evidence.activations[:, 0].all() and evidence.object_vertices.max() == 10
    for value in vars(evidence).values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError): value.setflags(write=True)


def test_plane_exact_equal_joint_minimum_change_with_declared_interior_margin():
    args = inputs(count=1); evidence = freeze(args)
    proposal = args[0].copy(); proposal[0, 0, :, 2] += .05
    human = np.zeros((1, 3)); obj = human.copy()
    result = project_joint_translations(evidence, proposal, human, obj)
    correction = -.05+config().numerical_slack_m-2*config().convergence_tolerance_m
    np.testing.assert_allclose(result['relative_correction_camera'], [[0, 0, correction]], atol=1e-14)
    np.testing.assert_allclose(result['human_correction_camera'], [[0, 0, correction/2]], atol=1e-14)
    np.testing.assert_array_equal(result['human_correction_camera'], -result['object_correction_camera'])
    assert result['witness_gap_after_m'][0, 0] <= result['frozen_gap_bound_m'][0, 0]
    assert result['projection_cycles'][0] == 2
    assert not result['global_minimum_verified'] and not result['physical_contact_verified']
    assert not result['whole_hand_min_evaluated'] and not result['temporal_quality_verified']
    # Independent scalar closed-triangle oracle on the ACTUALLY placed witness.
    actual = (proposal[0, 0, 0]+result['human_translation_camera'][0]
              -result['object_translation_camera'][0])
    oracle = np.sqrt(numpy_reference_distance_squared(actual[None], args[4][args[5]])[0])
    assert oracle <= .01+config().numerical_slack_m


def test_already_feasible_translation_and_tangential_sliding_untouched():
    args = inputs(); evidence = freeze(args)
    proposal = args[0].copy()
    proposal[:, 0, :, 0] += np.linspace(-.7, .7, 4)[:, None]
    proposal[:, 0, :, 1] += np.linspace(.6, -.6, 4)[:, None]
    human = np.array([[.1, .2, 0], [.2, .1, 0], [.3, 0, 0], [.4, -.1, 0]])
    obj = np.zeros((4, 3))
    result = project_joint_translations(evidence, proposal, human, obj)
    np.testing.assert_array_equal(result['human_translation_camera'], human)
    np.testing.assert_array_equal(result['object_translation_camera'], obj)
    assert not result['relative_correction_camera'].any()
    assert not result['projection_cycles'].any()


def test_proposal_minimum_does_not_replace_original_anatomical_witness():
    args = inputs(count=1); evidence = freeze(args)
    proposal = args[0].copy(); proposal[0, 0, 0, 2] = .07
    proposal[0, 0, 1, 2] = .001  # A new nearer vertex cannot hide the old gap.
    result = project_joint_translations(evidence, proposal, np.zeros((1, 3)), np.zeros((1, 3)))
    assert result['witness_gap_before_m'][0, 0] == .07
    assert result['relative_correction_camera'][0, 2] < -.059
    assert result['witness_gap_after_m'][0, 0] <= result['frozen_gap_bound_m'][0, 0]
    # Native unsigned feasibility alone may push the other finger through the
    # plane. This deliberate limitation needs an INDEPENDENT penetration gate.
    assert proposal[0, 0, 1, 2]+result['relative_correction_camera'][0, 2] < 0


def test_common_motion_preserved_release_free_full_chronology_no_static_replacement():
    args = inputs(count=6); args[6][2:4] = False; evidence = freeze(args)
    common = np.array([[i*.2, .1*i*i, .03*i] for i in range(6)])
    proposal = args[0]+common[:, None, None]
    proposal[:, 0, :, 2] += .04
    result = project_joint_translations(evidence, proposal, common.copy(), common.copy())
    np.testing.assert_allclose((result['human_translation_camera']+result['object_translation_camera'])/2,
                               common, atol=1e-16)
    np.testing.assert_array_equal(result['human_translation_camera'][2:4], common[2:4])
    np.testing.assert_array_equal(result['object_translation_camera'][2:4], common[2:4])
    np.testing.assert_array_equal(result['frame_index'], np.arange(6))
    np.testing.assert_array_equal(result['activations'], args[6])
    assert np.isnan(result['witness_gap_after_m'][2:4]).all()
    assert not result['relative_correction_camera'][2:4].any()
    assert np.linalg.norm(result['object_translation_camera'][-1]-result['object_translation_camera'][0]) > 2


def test_rigid_shared_frame_equivariance_and_nearest_face_not_barycentric_freeze():
    args = inputs(); args[0][..., :2] = [-.6, .6]
    evidence = freeze(args); proposal = args[0].copy()
    proposal[:, 0, :, :2] = [.6, -.6]  # Cross the original square diagonal.
    proposal[:, 0, :, 2] += .03
    human = np.zeros((4, 3)); obj = human.copy()
    result = project_joint_translations(evidence, proposal, human, obj)
    assert not result['relative_correction_camera'][:, :2].any()
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    offset = np.array([1., 2., 3.])
    args[0] = args[0]@rotation.T+offset
    args[2] = rotation[None]@args[2]; args[3] = args[3]@rotation.T+offset
    transformed = project_joint_translations(freeze(args), proposal@rotation.T+offset,
        human@rotation.T+offset, obj@rotation.T+offset)
    np.testing.assert_allclose(transformed['relative_correction_camera'],
                               result['relative_correction_camera']@rotation.T, atol=1e-14)
    np.testing.assert_allclose(transformed['object_translation_camera'],
                               result['object_translation_camera']@rotation.T+offset, atol=1e-14)


def test_two_active_orthogonal_surface_constraints_jointly_feasible():
    args = inputs(count=1, joints=1)
    extra = np.array([[0., -10, -10], [0, 10, -10], [0, 10, 10], [0, -10, 10]])
    args[4] = np.r_[args[4], extra]; args[5] = np.r_[args[5], np.array([[4, 5, 6], [4, 6, 7]])]
    args[0][0, 0, 0] = [2., 0, .01]; args[0][0, 1, 0] = [.01, 0, 2.]
    args[6][:] = True; evidence = freeze(args)
    proposal = args[0].copy(); proposal[0, 0, 0, 2] += .03; proposal[0, 1, 0, 0] += .03
    result = project_joint_translations(evidence, proposal, np.zeros((1, 3)), np.zeros((1, 3)))
    delta = -.03+config().numerical_slack_m-2*config().convergence_tolerance_m
    np.testing.assert_allclose(result['relative_correction_camera'], [[delta, 0, delta]], atol=1e-14)
    assert np.all(result['witness_gap_after_m'] <= result['frozen_gap_bound_m'])


def test_conflicting_fixed_articulation_rejects_full_proposal_without_dropping_hand():
    args = inputs(count=1, joints=1)
    yz = np.array([[-10., -10], [10, -10], [10, 10], [-10, 10]])
    args[4] = np.r_[np.c_[np.full(4, -1.), yz], np.c_[np.full(4, 1.), yz]]
    args[5] = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    args[0][0, 0, 0] = [-1.01, 0, 0]; args[0][0, 1, 0] = [1.01, 0, 0]
    args[6][:] = True; evidence = freeze(args); proposal = args[0].copy()
    proposal[0, 0, 0, 0] -= .2; proposal[0, 1, 0, 0] += .2
    saved = proposal.copy()
    with pytest.raises(ContactFeasibilityFailure) as caught:
        project_joint_translations(evidence, proposal, np.zeros((1, 3)), np.zeros((1, 3)))
    assert caught.value.frame_index == 0 and caught.value.maximum_violation_m > .1
    assert 'not an infeasibility proof' in caught.value.reason
    np.testing.assert_array_equal(proposal, saved)
    assert evidence.activations.all()


def test_closest_closed_triangles_match_independent_full_surface_oracle_and_broadphase():
    rng = np.random.default_rng(5321)
    vertices = rng.normal(size=(36, 3)); faces = np.arange(36).reshape(12, 3)
    surface = _NearestSurface(vertices, faces)
    for point in rng.normal(size=(25, 3)):
        gap, anchor, face = surface.closest(point)
        full_gap, full_anchor, full_face = surface.closest(point, broadphase=False)
        oracle = np.sqrt(numpy_reference_distance_squared(point[None], vertices[faces])[0])
        np.testing.assert_allclose([gap, full_gap, np.linalg.norm(point-anchor)], oracle, atol=2e-15)
        np.testing.assert_array_equal(anchor, full_anchor)
        assert face == full_face


def test_degenerate_faces_retained_as_closed_segments_points_not_deleted():
    triangles = np.array([[[0., 0, 0], [1, 0, 0], [1, 0, 0]],
                          [[0., 2, 0], [0, 2, 0], [0, 2, 0]]])
    closest, squared = _closest_triangle_points(np.array([.5, 1., 0]), triangles)
    np.testing.assert_array_equal(closest, [[.5, 0, 0], [0, 2, 0]])
    np.testing.assert_array_equal(squared, [1, 1.25])


def test_original_per_frame_witness_can_change_but_not_become_static_world_anchor():
    args = inputs(); args[0][0:2, 0, 0, 2] = .02; args[0][0:2, 0, 1, 2] = .01
    evidence = freeze(args)
    np.testing.assert_array_equal(evidence.witness_columns[:, 0], [1, 1, 0, 0])
    proposal = args[0].copy(); proposal[:, 0, :, 2] += .04
    proposal[:, 0, :, 0] += np.arange(4)[:, None]*.2
    result = project_joint_translations(evidence, proposal, np.zeros((4, 3)), np.zeros((4, 3)))
    assert not result['relative_correction_camera'][:, 0].any()
    assert np.all(result['witness_gap_after_m'][:, 0] <= result['frozen_gap_bound_m'][:, 0])


def test_result_owns_all_arrays_and_does_not_modify_inputs():
    args = inputs(); evidence = freeze(args); proposal = args[0].copy()
    proposal[:, 0, :, 2] += .03; saved = proposal.copy()
    human = np.zeros((4, 3)); obj = human.copy()
    result = project_joint_translations(evidence, proposal, human, obj)
    np.testing.assert_array_equal(proposal, saved)
    assert not human.any() and not obj.any()
    stored = result['human_translation_camera'].copy(); proposal[:] = 88; human[:] = 77
    np.testing.assert_array_equal(result['human_translation_camera'], stored)
    for value in result.values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError): value.setflags(write=True)


def fast_inputs(args, evidence):
    points = np.full((len(args[0]), 2, 3), np.nan)
    for frame, side in zip(*np.nonzero(evidence.activations)):
        points[frame, side] = args[0][frame, side, evidence.witness_columns[frame, side]]
    return [points, args[1], evidence.witness_source_indices.copy(), args[2], args[3],
            args[4], args[5], args[6], args[7]]


def fast_freeze(args):
    return freeze_contact_witnesses(*args, config=config(),
        source_reference='manufactured:automatic_native_gates_not_a_real_run',
        witness_selection_reference='manufactured:original_native_complete_hand_triangle_argmin',
        original_selection_is_whole_hand_minimum=True)


def test_fast_original_witness_path_byte_parity_and_no_complete_hand_query(monkeypatch):
    args = inputs(); evidence = freeze(args)
    selected = fast_inputs(args, evidence)
    # No full-hand distances may be recomputed in the hot path. Safe closest
    # surface queries for <=2 original points/frame are still required.
    def forbidden(*_args, **_kwargs): raise AssertionError('hot path queried complete hand')
    monkeypatch.setattr('world_reward.sequence_pose._ContactTriangleSurface.distances', forbidden)
    fast = fast_freeze(selected)
    np.testing.assert_array_equal(fast.baseline_gap_m, evidence.baseline_gap_m)
    np.testing.assert_array_equal(fast.witness_columns, evidence.witness_columns)
    assert not fast.original_whole_hand_minimum_recomputed
    assert fast.original_whole_hand_minimum_attested
    assert evidence.original_whole_hand_minimum_recomputed
    proposal = args[0].copy(); proposal[:, 0, :, 2] += .04
    result = project_joint_translations(fast, proposal, np.zeros((4, 3)), np.zeros((4, 3)))
    assert not result['original_whole_hand_minimum_recomputed']
    assert 'original_native_complete_hand_triangle_argmin' in result['witness_selection_reference']
    assert np.all(result['witness_gap_after_m'][:, 0] <= result['frozen_gap_bound_m'][:, 0])


def test_original_approximate_qa_witness_allowed_without_claiming_whole_hand_nondegradation():
    args = inputs(); full_min = freeze(args); selected = fast_inputs(args, full_min)
    # Original approximate QA may select a far vertex while another vertex
    # passes the ORIGINAL native hand gate. Do not invent a new contact gate.
    selected[2][:, 0] = args[1][0, 1]; selected[0][:, 0] = args[0][:, 0, 1]
    selected[0][:, 0, 2] = .08
    evidence = freeze_contact_witnesses(*selected, config=config(),
        source_reference='manufactured:original_positive_native_contact_and_minimum_proximity',
        witness_selection_reference='provided_baseline_witness_not_whole_hand_min')
    assert not evidence.original_whole_hand_minimum_attested
    assert not evidence.original_whole_hand_minimum_recomputed
    assert (evidence.baseline_gap_m[:, 0] == .08).all()
    proposal = args[0].copy(); proposal[:, 0, 1, 2] = .1
    result = project_joint_translations(evidence, proposal, np.zeros((4, 3)), np.zeros((4, 3)))
    assert not result['original_whole_hand_minimum_nondegradation_bound']
    assert result['witness_selection_reference'] == 'provided_baseline_witness_not_whole_hand_min'
    assert np.all(result['witness_gap_after_m'][:, 0] <= .08+config().numerical_slack_m)


@pytest.mark.parametrize('bad', ['missing_id', 'inactive_id', 'inactive_point', 'active_nan',
    'far_active', 'unsigned_unknown', 'reference', 'masked'])
def test_fast_original_witness_path_rejects_unqualified_missing_evidence(bad):
    args = inputs(); selected = fast_inputs(args, freeze(args))
    if bad == 'missing_id': selected[2][0, 0] = 999
    if bad == 'inactive_id': selected[2][0, 1] = 0
    if bad == 'inactive_point': selected[0][0, 1] = 0
    if bad == 'active_nan': selected[0][0, 0] = np.nan
    if bad == 'far_active': selected[0][0, 0, 2] = .05
    if bad == 'unsigned_unknown': selected[2] = selected[2].astype(np.uint64)
    if bad == 'masked': selected[0] = np.ma.array(selected[0], mask=False)
    with pytest.raises(ValueError):
        if bad == 'reference':
            freeze_contact_witnesses(*selected, config=config(), source_reference='toy',
                                    witness_selection_reference=' ')
        else: fast_freeze(selected)


def test_native_float32_rotation_roundoff_preserved_not_silently_repaired():
    args = inputs(); rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix().astype(np.float32)
    args[0] = args[0]@rotation.T
    args[2] = np.broadcast_to(rotation, (4, 3, 3)).copy()
    evidence = freeze(args)
    proposal = args[0].copy(); proposal[:, 0] += np.array([0., 0, .04])@rotation.T
    result = project_joint_translations(evidence, proposal, np.zeros((4, 3)), np.zeros((4, 3)))
    np.testing.assert_array_equal(evidence.object_rotation, args[2])
    assert 0 < result['object_rotation_orthogonality_error'] < 1e-5
    assert not result['fixed_object_rotation_repaired']
    assert np.all(result['witness_gap_after_m'][:, 0] <= result['frozen_gap_bound_m'][:, 0])


@pytest.mark.parametrize('bad', ['missing_frame', 'bad_gate', 'far_active', 'wrong_rotation',
    'wrong_faces', 'duplicate_id', 'nan', 'integer_points', 'masked', 'precision'])
def test_malformed_evidence_fails_without_self_disabling(bad):
    args = inputs()
    if bad == 'missing_frame': args[7] = np.array([0, 1, 3, 4])
    if bad == 'bad_gate': args[6] = args[6].astype(float)
    if bad == 'far_active': args[0][:, 0, :, 2] = .05
    if bad == 'wrong_rotation': args[2][0, 0, 0] = 2
    if bad == 'wrong_faces': args[5][0, 0] = 100
    if bad == 'duplicate_id': args[1][0, 0] = args[1][0, 1]
    if bad == 'nan': args[0][0, 1, 0, 0] = np.nan  # Inactive anatomy remains required.
    if bad == 'integer_points': args[0] = args[0].astype(int)
    if bad == 'masked': args[0] = np.ma.array(args[0], mask=False)
    if bad == 'precision': args[0][..., 0] += 1e12
    with pytest.raises(ValueError): freeze(args)


@pytest.mark.parametrize('bad', ['shape', 'nan', 'wrong_dtype', 'masked', 'chronology'])
def test_malformed_proposal_fails_not_copies_last_frame_or_omits_contact(bad):
    args = inputs(); evidence = freeze(args)
    proposal = args[0].copy(); human = np.zeros((4, 3)); obj = human.copy()
    if bad == 'shape': proposal = proposal[:3]
    if bad == 'nan': proposal[0, 0, 0] = np.nan
    if bad == 'wrong_dtype': obj = obj.astype(int)
    if bad == 'masked': human = np.ma.array(human, mask=False)
    if bad == 'chronology': obj = obj[:3]
    with pytest.raises(ValueError): project_joint_translations(evidence, proposal, human, obj)

"""Tiny real-run integration contracts: no datasets, models, GT or cloud."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import contact_feasible_real as real
from world_reward.contact_feasible_placement import ContactFeasibilityFailure
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS


def tiny(count=96):
    ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    vertices = np.array([[-2., -2, 0], [2., -2, 0], [2., 2, 0], [-2., 2, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    rotation = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    translation = np.zeros((count, 3), np.float32); translation[:, 2] = 2.
    grid = np.arange(count, dtype=np.int64)
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
    bank = dict(vertices=vertices, faces=faces, rotations=rotation, translations=translation,
        object_scale=np.array(1., np.float32), original_K=K.copy(), K=K.copy(), frame_index=grid,
        fps=30., points=np.empty((0, 3)), xy=np.empty((count, 0, 2)), visible=np.empty((count, 0), bool))
    params = {k: np.zeros((count, d), np.float32) for k, d in NATIVE_PARAMETER_DIMS.items()}
    params['mhr_global_rot6d'][:] = [1, 0, 0, 1, 0, 0]
    params['mhr_trans'][:, 2] = 2.
    params.update(mhr_joints=np.full((count, 2, 3), [0., 0., 2.01], np.float32),
        mhr_keypoints=np.full((count, 3, 3), [0., 0., 2.01], np.float32),
        human_faces=faces.copy(), frame_index=grid.copy())
    original = np.full((count, 6, 3), [0., 0., 2.01], np.float32)
    original[:, :, 0] = np.linspace(-.1, .1, 6)
    spec = SimpleNamespace(width=640, height=480, episode_index=14)
    src = dict(human=original.copy(), native=deepcopy(params), QA_hand_ids=ids, spec=spec)
    trajectory = dict(pose=np.zeros((count, 136), np.float32), scales=np.zeros(68, np.float32),
        shape=params['mhr_shape'][0].copy(), expression=np.zeros(72, np.float32),
        object_rotation=rotation.copy(), object_translation=translation.copy(),
        object_scale=bank['object_scale'].copy(), object_vertices=vertices.copy(), object_faces=faces.copy(),
        camera_K=K.copy(), frame_index=grid.copy())
    active = np.zeros((count, 2), bool); active[1:3, 0] = True; active[4, 1] = True
    witnesses = np.full((count, 2), -1, np.int64); witnesses[1:3, 0] = 0; witnesses[4, 1] = 3
    qa = dict(activations=active, hand_vertex_ids=witnesses, frame_index=grid.copy())
    src['QA_witness_ids'] = witnesses.copy()
    human = original.copy(); human[:, :, 2] += .06
    params['mhr_joints'][..., 2] += .06; params['mhr_keypoints'][..., 2] += .06
    params['mhr_trans'][:, 2] += .06
    saved = dict(human=human, params=params, trajectory=trajectory, activation=active)
    return bank, src, saved, qa


def validate(bank, src, saved, qa):
    return real.validate_saved_arrays(bank, src, saved['trajectory'], saved['params'], saved['human'], qa,
        body_count=6, joint_count=2, keypoint_count=3)


def test_real_saved_schema_and_hot_projection_cover_all_frames_without_native_decode():
    bank, src, saved, qa = tiny()
    active, ids = validate(bank, src, saved, qa)
    target, params, trajectory, result, verification = real.project_saved(bank, src, saved)
    assert len(target) == 96 and verification['active_witnesses'] == 3
    assert target.dtype == params['mhr_trans'].dtype == np.float64
    assert result['original_whole_hand_minimum_attested'] is False
    assert result['original_whole_hand_minimum_recomputed'] is False
    assert result['original_whole_hand_minimum_nondegradation_bound'] is False
    assert verification['physical_contact_claimed'] is False
    assert verification['all_original_active_bounds_verified'] is True
    np.testing.assert_array_equal(result['human_correction_camera'], -result['object_correction_camera'])
    assert 'pose' not in trajectory
    assert 'pose' in saved['trajectory']
    assert np.max(result['witness_gap_after_m'][active]) <= .0100001
    np.testing.assert_array_equal(target[~active.any(1)], saved['human'][~active.any(1)].astype(float))
    for name in ('mhr_joints', 'mhr_keypoints'):
        np.testing.assert_allclose(params[name]-saved['params'][name],
            np.broadcast_to(result['human_correction_camera'][:, None], params[name].shape), atol=1e-15)
    np.testing.assert_allclose(target-saved['human'],
        np.broadcast_to(result['human_correction_camera'][:, None], target.shape), atol=1e-15)
    for key in saved['params'].keys()-{'mhr_trans', 'mhr_joints', 'mhr_keypoints'}:
        assert real.same_bytes(saved['params'][key], params[key])


@pytest.mark.parametrize('case', ['rotation', 'hands', 'shape', 'camera', 'frame_tail', 'activation_id', 'wrong_side', 'nonfinite', 'native_dtype'])
def test_malformed_or_changed_source_never_self_disables_or_repairs(case):
    bank, src, saved, qa = tiny()
    if case == 'rotation': saved['trajectory']['object_rotation'][0, 0, 0] += .01
    if case == 'hands': saved['params']['mhr_hand'][0, 0] += .01
    if case == 'shape': saved['params']['mhr_shape'][:] += .1
    if case == 'camera': saved['trajectory']['camera_K'][0, 0] += 1
    if case == 'frame_tail': qa['frame_index'][-1] -= 1
    if case == 'activation_id': qa['hand_vertex_ids'][1, 0] = -1
    if case == 'wrong_side': qa['hand_vertex_ids'][1, 0] = 3
    if case == 'nonfinite': saved['human'][0, 0, 0] = np.nan
    if case == 'native_dtype': saved['params']['mhr_trans'] = saved['params']['mhr_trans'].astype(float)
    with pytest.raises(ValueError): validate(bank, src, saved, qa)


def test_verify_actual_emitted_full_geometry_not_only_primitive_gap():
    bank, src, saved, _ = tiny()
    target, params, trajectory, result, _ = real.project_saved(bank, src, saved)
    target[1, src['QA_witness_ids'][1, 0], 2] += .001
    with pytest.raises(ContactFeasibilityFailure):
        real.verify_emitted_witnesses(bank, src, target, trajectory, result)


def dwpose(bank):
    count = len(bank['frame_index'])
    xy = np.full((count, 133, 2), [15., 25.]); scores = np.ones((count, 133), np.float32)
    return dict(original_xy=xy, raw_scores=scores, boxes_original_xyxy=np.zeros((count, 4), np.float32),
        actor_present=np.ones(count, bool), frame_index=bank['frame_index'].copy(), fps=np.array(30.))


def test_saved_original_grid_dwpose_conversion_matches_native_operator_exactly():
    bank, src, _, _ = tiny(); obs = dwpose(bank)
    result = real.validate_dwpose(obs, bank, src['spec'])
    np.testing.assert_array_equal(result.human_xy, obs['original_xy']-.5)
    for fps in (24., 29., 31.):
        obs['fps'] = np.array(fps)
        with pytest.raises(ValueError): real.validate_dwpose(obs, bank, src['spec'])


def test_existing_A_B_quality_reproduction_strict_and_none_not_fabricated():
    row = dict(RGB_mean_px=None, observations=12, heldout_accuracy_verified=False, gap=.002)
    real.preserve_reported_quality(row, deepcopy(row))
    wrong = dict(row, gap=.00201)
    with pytest.raises(ValueError): real.preserve_reported_quality(wrong, row)
    assert real.unsupported(1)['rerolled'] is False and real.unsupported(7)['fabricated_predictions'] is False
    assert real.COHORT == (9, 1, 14, 7)


def test_C_failure_keeps_A_B_sealed_and_all_cohort_results(monkeypatch, tmp_path):
    bank, src, saved, qa = tiny()
    saved.update(report=dict(baseline_binding={}, candidate_outputs={}, metrics={real.A_NAME: {}, real.B_NAME: {}}),
        evidence=SimpleNamespace(), cfg={})
    monkeypatch.setenv('WR_CODE_REVISION', 'a'*40)
    monkeypatch.setattr(real, 'FRAMES', {9: 96, 14: 96})
    monkeypatch.setattr(real.native, 'original_sources', lambda *_: (bank, src))
    monkeypatch.setattr(real.real.saved, 'load_npz', lambda *_: {})
    monkeypatch.setattr(real.real.contact, 'hand_indices', lambda *_: src['QA_hand_ids'])
    monkeypatch.setattr(real, 'saved_B', lambda *_: saved)
    monkeypatch.setattr(real.native, 'quality', lambda *_: {})
    def fail(*_): raise ContactFeasibilityFailure(4, 'numerical not physical infeasibility', .001)
    monkeypatch.setattr(real, 'project_saved', fail)
    out = tmp_path/'episode_000009'
    report = real.run_episode(9, out, SimpleNamespace(), {})
    assert report['status'] == 'complete_A_B_C_projection_rejected'
    assert set(report['metrics']) == {real.A_NAME, real.B_NAME}
    assert (out/'A_B_QA.json').is_file() and (out/'report.json').is_file()
    assert not (out/'target.npy').exists()
    assert report['C_rejection']['physical_infeasibility_claimed'] is False


def test_wrapper_is_offline_cpu_no_gpu_lock_and_all_inputs_readonly():
    wrapper = (Path(__file__).parents[1]/'infra/run_contact_feasible_real.sh').read_text()
    assert '--gpus' not in wrapper and 'nvidia-smi' not in wrapper and '.world-reward-h100.lock' not in wrapper
    assert '--network none' in wrapper and '--cpus 4' in wrapper and '--read-only' in wrapper
    for route in ('results/native-joint-real-$B_SOURCE', 'jobs/$B_SOURCE/run_native_joint_real', 'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4', 'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000014.mp4'):
        assert f'src=$ROOT/{route}' in wrapper
    assert 'WR_IMAGE_ID="$IMAGE"' in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper
    assert all((Path(__file__).parents[1]/name).is_file() for name in real.HELPERS)

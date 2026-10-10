"""Tiny CPU-only saved ablation contracts; no real datasets/models/cloud."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1] / 'infra'))
import pose_smoothing_real as runner
from world_reward.native_joint_refinement import JointImageEvidence


def tiny(count=41, *, near_rotation=False):
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    phase = np.arange(count)
    r = Rotation.from_rotvec(np.c_[phase * 0, phase * 0, .01 * phase + .003 * np.sin(phase * 2)]).as_matrix().astype(np.float32)
    if near_rotation: r[:, :, 0] *= 1. + 2e-6
    t = np.c_[.002 * phase + .003 * np.sin(phase * 2), phase * 0, np.ones(count) * 2].astype(np.float32)
    human = np.tile([[-.2, 0, 2.01], [.2, 0, 2.01]], (count, 1, 1)).astype(np.float32)
    keys = np.tile([.1, .1, 2.1], (count, 70, 1)).astype(np.float32)
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    points = np.array([[-.2, 0, 0], [.2, 0, 0]])
    from sequence_pose_probe import project
    xy = project(points, r, t, K)
    bank = dict(vertices=vertices, faces=faces, rotations=r, translations=t, K=K.copy(), original_K=K.copy(),
        frame_index=np.arange(count, dtype=np.int64), fps=30., object_scale=np.array(1., np.float32),
        points=points, xy=xy, visible=np.ones((count, 2), bool))
    active = np.ones((count, 2), bool); ids = np.tile([0, 1], (count, 1)).astype(np.int64)
    src = dict(human=human, native=dict(mhr_joints=keys[:, :2].copy(), mhr_keypoints=keys.copy(), human_faces=faces), QA_witness_ids=ids)
    evidence = JointImageEvidence(points, xy, bank['visible'], np.ones((count, 133, 2)) * [330, 260],
        np.ones((count, 133), np.float32), K, bank['frame_index'])
    return bank, src, active, evidence


def test_runtime_authored_noise_and_moving_DEV_is_finite_not_real_gain_claim():
    row = runner.authored_DEV()
    assert row['passed'] and row['full_frames'] == 121
    assert row['centre_MSE_after'] < .15 * row['centre_MSE_before']
    assert row['angular_MSE_after'] < .15 * row['angular_MSE_before']
    assert not row['challenge_ground_truth_used']


@pytest.mark.parametrize('near_rotation', [False, True])
def test_common_scene_rigid_motion_preserves_all_material_geometry_including_near_SO3(near_rotation):
    bank, src, active, _ = tiny(near_rotation=near_rotation)
    before = deepcopy((bank, src))
    method, proposals = runner.proposals(bank, src)
    assert set(proposals) == set(runner.VARIANTS)
    assert method['window_frames'] == 9
    common = proposals['common_SE3_diagnostic']; C = common['common_rotation']; u = common['common_translation']
    original_objects = np.einsum('tij,pj->tpi', bank['rotations'], bank['vertices']) + bank['translations'][:, None]
    actual_objects = np.einsum('tij,pj->tpi', common['rotation'], bank['vertices']) + common['translation'][:, None]
    expected_objects = np.einsum('tij,tpj->tpi', C, original_objects.astype(np.float64)) + u[:, None]
    # Original float32 forward construction is deliberately not used as a
    # precision oracle for FP64 affine-coordinate geometry.
    np.testing.assert_allclose(actual_objects, expected_objects, atol=2e-7)
    np.testing.assert_allclose(C @ C.transpose(0, 2, 1), np.tile(np.eye(3), (len(C), 1, 1)), atol=2e-15)
    np.testing.assert_array_equal(common['human'][0], src['human'][0])
    np.testing.assert_array_equal(common['rotation'][0], bank['rotations'][0])
    np.testing.assert_array_equal(common['translation'][0], bank['translations'][0])
    for field in ('human', 'joints', 'keypoints'):
        assert common[field].shape[0] == len(bank['frame_index'])
    np.testing.assert_allclose(np.linalg.norm(np.diff(common['human'], axis=1), axis=2),
        np.linalg.norm(np.diff(src['human'].astype(float), axis=1), axis=2), atol=2e-15)
    a = runner.contact_gaps(bank, src, src['human'], bank['rotations'], bank['translations'], active)
    b = runner.contact_gaps(bank, src, common['human'], common['rotation'], common['translation'], active)
    np.testing.assert_allclose(a['gap_m'], b['gap_m'], atol=5e-15)
    assert len(b['gap_m']) == active.sum()
    np.testing.assert_array_equal(bank['rotations'], before[0]['rotations'])
    np.testing.assert_array_equal(src['human'], before[1]['human'])
    assert proposals['object_only']['human'] is src['human']


def test_exact_continuous_surface_witness_distance_not_mesh_vertex_or_wholehand_claim():
    bank, src, active, _ = tiny()
    bank['rotations'][:] = np.eye(3); bank['translations'][:] = [0., 0., 2.]
    gap = runner.contact_gaps(bank, src, src['human'], bank['rotations'], bank['translations'], active)
    np.testing.assert_allclose(gap['gap_m'], .009999990463256836, atol=1e-15)
    assert gap['gap_m'].max() < .02


def test_original_native_QA_operators_execute_with_full_automatic_evidence_and_common_human_change():
    bank, src, active, evidence = tiny()
    _, proposed = runner.proposals(bank, src)
    baseline = runner.native.quality(bank, src, src['human'], src['native']['mhr_keypoints'],
        bank['rotations'], bank['translations'], evidence, active)
    p = proposed['common_SE3_diagnostic']
    after = runner.native.quality(bank, src, p['human'], p['keypoints'], p['rotation'], p['translation'], evidence, active)
    assert baseline['RGB_mean_px'] == 0.
    assert after['RGB_mean_px'] > 0. and after['RGB_observations'] == baseline['RGB_observations']
    assert after['reserved_human_observations'] == baseline['reserved_human_observations']
    assert after['reserved_human_RGB_mean_px'] != baseline['reserved_human_RGB_mean_px']
    # Historical native QA intentionally retains its original FP32 local
    # transform; mathematical invariant diagnostics above use FP64 in both arms.
    assert after['same_anatomical_contact_mean_m'] == pytest.approx(baseline['same_anatomical_contact_mean_m'], abs=1e-9)
    assert after['original_native_active_hand_frames'] == active.sum()


def test_motion_and_fixed_floor_diagnostics_preserve_moving_signal_and_use_one_original_plane():
    bank, src, _, _ = tiny(); floor = .5
    _, proposals = runner.proposals(bank, src)
    p = proposals['common_SE3_diagnostic']
    row = runner.motion_and_floor(bank, p['human'], p['rotation'], p['translation'], floor)
    assert row['centroid_path_length_m'] > .05 and row['rotation_net_change_rad'] > .3
    assert row['full_frames'] == len(bank['frame_index']) and row['fixed_virtual_floor_camera_y_m'] == floor
    assert row['physical_floor_claimed'] is False and row['motion_retention_verified_against_truth'] is False


def test_full_pose_and_common_geometry_seals_atomic_before_QA_without_native_controls(tmp_path):
    bank, src, _, _ = tiny(); _, proposals = runner.proposals(bank, src)
    for name, p in proposals.items():
        out = tmp_path / name; out.mkdir()
        pins = runner.seal_proposal(out, bank, p)
        with np.load(out / 'object_pose.npz', allow_pickle=False) as data:
            assert len(data['object_rotation']) == len(data['frame_index']) == 41
            for key, reference in [('object_vertices', bank['vertices']), ('object_faces', bank['faces']),
                    ('camera_K', bank['original_K']), ('object_scale', bank['object_scale'])]:
                np.testing.assert_array_equal(data[key], reference)
            assert 'pose' not in data and 'mhr_trans' not in data
        assert all(not p.stat().st_mode & 0o222 for p in out.iterdir())
        assert 'target' in pins if name == 'common_SE3_diagnostic' else 'target' not in pins
        with pytest.raises(FileExistsError): runner.seal_proposal(out, bank, p)


def test_episode_seals_all_proposals_before_any_candidate_QA_and_keeps_failed_metrics(monkeypatch, tmp_path):
    bank, src, active, evidence = tiny()
    qa = dict(metric=1.)
    b = dict(report=dict(baseline_binding={}, metrics={runner.saved.A_NAME: qa, runner.saved.B_NAME: qa}),
        human=src['human'], params=src['native'], trajectory=dict(object_rotation=bank['rotations'], object_translation=bank['translations']),
        evidence=evidence, activation=active, cfg={})
    monkeypatch.setenv('WR_CODE_REVISION', 'a' * 40)
    monkeypatch.setattr(runner.saved, 'FRAMES', {9: len(bank['frame_index']), 14: 442})
    monkeypatch.setattr(runner.native, 'original_sources', lambda *_: (bank, src))
    monkeypatch.setattr(runner.real.saved, 'load_npz', lambda *_: {})
    monkeypatch.setattr(runner.real.contact, 'hand_indices', lambda *_: np.array([[0], [1]]))
    monkeypatch.setattr(runner.saved, 'saved_B', lambda *_: b)
    monkeypatch.setattr(runner.native, 'quality_decision', lambda *_: dict(passed=False))
    calls = []
    out = tmp_path / 'episode_000009'
    def quality(*_):
        if len(calls) >= 2:
            assert all((out / name / 'geometry.json').is_file() for name in runner.VARIANTS)
            assert all((out / name / 'object_pose.npz').is_file() for name in runner.VARIANTS)
        calls.append(1); return qa.copy()
    monkeypatch.setattr(runner.native, 'quality', quality)
    row = runner.run_episode(9, out, SimpleNamespace(), {}, float('inf'))
    assert row['status'] == 'complete_saved_full_T_pose_smoothing_diagnostic', row.get('error')
    assert all(not r['decision_vs_A']['passed'] for r in row['variants'].values())
    assert row['native_control_export_available'] is False and not row['penetration_evaluated']
    assert runner.saved.unsupported(1)['rerolled'] is False and runner.saved.unsupported(7)['fabricated_predictions'] is False


def test_wrapper_offline_CPU_only_bound_inputs_no_GPU_lease_and_source_closure():
    root = Path(__file__).parents[1]
    wrapper = (root / 'infra/run_pose_smoothing_real.sh').read_text()
    assert '--gpus' not in wrapper and 'nvidia-smi' not in wrapper and '.world-reward-h100.lock' not in wrapper
    assert '--network none' in wrapper and '--read-only' in wrapper and '--cpus 4' in wrapper
    assert '303s docker' in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper
    assert 'WR_IMAGE_ID="$IMAGE"' in wrapper
    for route in ('results/native-joint-real-$B_SOURCE', 'jobs/$B_SOURCE/run_native_joint_real',
            'weights/cari4d/refinement/mhr_hand_surface_spec.npz'):
        assert f'src=$ROOT/{route},dst=$ROOT/{route},readonly' in wrapper
    assert all((root / helper).is_file() for helper in runner.HELPERS)
    from azure_job import runtime_bundle_paths
    files = {str(p.relative_to(root)): p.read_bytes() for folder in ('infra', 'src', 'configs')
        for p in (root / folder).rglob('*') if p.is_file() and ('__pycache__' not in p.parts)}
    files['pyproject.toml'] = (root / 'pyproject.toml').read_bytes()
    selected = runtime_bundle_paths(files, 'infra/run_pose_smoothing_real.sh')
    assert set(runner.HELPERS) <= set(selected)
    import inspect
    run_source = inspect.getsource(runner.run)
    assert run_source.index("report['authored_DEV'] = authored_DEV()") < run_source.index('for episode in saved.COHORT')


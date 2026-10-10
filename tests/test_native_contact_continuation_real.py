"""Tiny integration/ABI sealing tests; no models, RGB, GT or GPU locally."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'infra'))
import native_contact_continuation_real as run
from test_contact_feasible_real import tiny
from test_native_contact_continuation import inputs, make_geometry


def cfg():
    return {"gates": {v: 1.00000001 for v in run.METRIC_GATE_KEYS.values()}}


def test_gates_are_original_native_QA_ratios_not_new_contact_weights():
    bank = dict(points=np.empty((0, 3)))
    gates = run.gates(cfg(), bank)
    assert all(g.maximum_ratio == 1.00000001 and g.absolute_numerical_slack == 1e-12 for g in gates)
    assert [g.name for g in gates if g.allow_missing] == ['RGB_mean_px', 'RGB_motion_increment_error_mean_px']
    full = run.gates(cfg(), dict(points=np.ones((33, 3))))
    assert not any(g.allow_missing for g in full)


def test_frozen_real_witness_factory_keeps_ids_activity_original_gauge_and_all_frames():
    bank, src, b, _ = tiny(); evidence, branches = run.frozen_evidence(bank, src, b)
    np.testing.assert_array_equal(evidence.activations, b['activation'])
    np.testing.assert_array_equal(evidence.witness_source_indices, src['QA_witness_ids'])
    assert not evidence.original_whole_hand_minimum_attested
    assert branches.face_indices.shape == (96, 2) and not branches.face_indices.flags.writeable
    assert np.isnan(evidence.baseline_gap_m[~b['activation']]).all()


def test_final_seal_has_real_native_direct_pose_not_stale_B_controls(tmp_path, monkeypatch):
    a, b, oa, ob, e = inputs(angle=0.); g = make_geometry(a)
    bank = dict(vertices=e.object_vertices.astype(np.float32), faces=e.object_faces,
        rotations=e.object_rotation.astype(np.float32), original_K=np.eye(3))
    selected = dict(parameters=a, geometry=g, object_translation=oa,
        original_activations=e.activations, original_witness_ids=e.witness_source_indices,
        witness_gaps_m=e.baseline_gap_m)
    actual = {}
    def trajectory(controls, params, poses, vertices, faces, K, spec):
        actual.update(controls=controls.copy(), poses=poses.copy(), params=deepcopy(params))
        return dict(pose=controls[:, :136], scales=controls[0, 136:],
            object_rotation=poses[:, :3, :3], object_translation=poses[:, :3, 3], frame_index=np.arange(96))
    monkeypatch.setattr(run.native.export, 'trajectory', trajectory)
    monkeypatch.setattr(run.native.export, 'object_roundtrip', lambda *_: None)
    pins = run.seal_selected(tmp_path, bank, dict(spec=SimpleNamespace()), selected)
    assert set(pins) == {'target', 'trajectory', 'native_parameters', 'frozen_witnesses'}
    np.testing.assert_array_equal(actual['controls'][:, :136], g['pose'])
    np.testing.assert_array_equal(actual['controls'][:, 136:], np.broadcast_to(g['scales'], (96, 68)))
    with np.load(tmp_path/'native_parameters.npz', allow_pickle=False) as saved:
        assert saved['mhr_trans'].tobytes() == a['mhr_trans'].tobytes()
        np.testing.assert_array_equal(saved['mhr_joints'], g['human_joints'])
    with np.load(tmp_path/'QA_witnesses.npz', allow_pickle=False) as qa:
        np.testing.assert_array_equal(qa['activations'], e.activations)
        np.testing.assert_array_equal(qa['hand_vertex_ids'], e.witness_source_indices)
    assert (tmp_path/'target.npy').stat().st_mode & 0o222 == 0


def test_export_helper_accepts_actual_native_F64_object_translations_and_F32_controls():
    a, _, oa, _, e = inputs(angle=0.); g = make_geometry(a)
    spec = run.native.full.inputs.PublicClipSpec(episode_index=9, total_frames=96,
        camera_name='exo_camera', height=480, width=640)
    K = run.native.export.public.inferred_camera(spec)
    poses = np.tile(np.eye(4), (96, 1, 1)); poses[:, :3, 3] = oa
    data = run.native.export.trajectory(np.c_[g['pose'], np.broadcast_to(g['scales'], (96, 68))],
        a, poses, e.object_vertices.astype(np.float32), e.object_faces, K, spec)
    assert data['object_translation'].dtype == np.float64
    assert data['pose'].dtype == np.float32 and data['frame_index'].dtype == np.int64
    assert set(data) == run.native.export.TRAJECTORY_KEYS


def test_failclosed_native_A_replay_keeps_original_A_B_QA_report_no_fabrication(tmp_path, monkeypatch):
    bank, src, b, _ = tiny(); b['cfg'] = cfg(); b['evidence'] = SimpleNamespace()
    b['report'] = dict(baseline_binding={}, metrics={run.saved.A_NAME: {}, run.saved.B_NAME: {}})
    monkeypatch.setenv('WR_CODE_REVISION', 'f'*40)
    monkeypatch.setattr(run.native, 'original_sources', lambda *_: (bank, src))
    monkeypatch.setattr(run.real.saved, 'load_npz', lambda *_: {})
    monkeypatch.setattr(run.real.contact, 'hand_indices', lambda *_: src['QA_hand_ids'])
    monkeypatch.setattr(run.saved, 'saved_B', lambda *_: b)
    monkeypatch.setattr(run.native, 'quality', lambda *_: {})
    monkeypatch.setattr(run, 'layer_factory', lambda *_: SimpleNamespace())
    def fail(*args, **kw): raise ValueError('Actual decoded dynamic A fails frozen contact bounds')
    monkeypatch.setattr(run, 'continue_native_contact', fail)
    cuda = SimpleNamespace(empty_cache=lambda: None)
    report = run.run_episode(9, tmp_path/'episode_000009', SimpleNamespace(), {}, SimpleNamespace(cuda=cuda), {})
    assert report['status'] == 'fail'
    assert (tmp_path/'episode_000009/A_B_QA.json').is_file()
    assert (tmp_path/'episode_000009/report.json').is_file()
    assert not (tmp_path/'episode_000009/target.npy').exists()


def test_wrapper_GPU_lease_no_reference_network_and_pinned_original_native_paths():
    wrapper = (Path(__file__).parents[1]/'infra/run_native_contact_continuation_real.sh').read_text()
    assert '--gpus all' in wrapper and '.world-reward-h100.lock' in wrapper and 'flock -n 8' in wrapper
    assert '--network none' in wrapper and '--read-only' in wrapper and '903s docker' in wrapper
    assert 'vendor/video_to_data' in wrapper and 'jobs/$B_SOURCE/run_native_joint_real' in wrapper
    assert '--cidfile' in wrapper and 'container_absence_verified' in wrapper
    assert not any(s in wrapper for s in ('eval_private', 'form_hoi_external', 'track_2', 'track_3'))
    assert all((Path(__file__).parents[1]/p).is_file() for p in run.HELPERS)
    assert run.native.HELPERS == run.saved.native.HELPERS


def test_native_decode_source_pinned_body_ABI_and_direct_forward_identical_native_call():
    text = (Path(__file__).parents[1]/'infra/native_contact_continuation_real.py').read_text()
    assert 'global_trans=trans*context.flip' in text
    assert 'pose=controls[:, :136].copy()' in text and 'scales=controls[0, 136:].copy()' in text
    assert "src['human'].astype(float)" in text and 'delta > 1e-5' in text
    assert "ledger.record(location/'lib_mhr/body_pose.py'" in text

"""Tiny manufactured provenance/runtime checks; no challenge bytes or models."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import sequence_pose_probe as runtime

ROOT = Path(__file__).resolve().parents[1]
SHA = 'a'*64


def raw_json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def frontend_fixture(tmp_path, monkeypatch, count=3):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    frontend = tmp_path/'results'/('gemini-sam31-'+runtime.FRONTEND)
    experiment = tmp_path/'experiments'/('full4d-v1-'+runtime.SOURCE)
    inventory = {f'{role}/{index:06d}.png': dict(bytes=7, sha256=SHA)
                 for role in (0, 1) for index in range(count)}
    inventory_raw = raw_json(inventory)
    mask_inventory = dict(files=2*count, bytes=2*count*7,
                          sha256=hashlib.sha256(inventory_raw).hexdigest())
    aggregates, numerical_rows = [], []
    for episode in (9, 1, 14, 7):
        directory = frontend/f'episode_{episode:06d}'/'automatic_masks'
        base = experiment/'outputs'/f'episode_{episode:06d}'
        report = dict(stage='automatic_masks', status='pass', episode_index=episode,
            frames=count, producer_revision=runtime.FRONTEND, input_track='track_1',
            ground_truth_used=False, hand_labeled_test=False, manual_labels=False,
            oracle_modes=[], full_original_grid=True, fixed_native_ids=[0, 1],
            empty_mask_interpolation=False, mask_inventory=mask_inventory, input_sha256=SHA)
        report_pin = write(directory/'report.json', raw_json(report))
        write(base/'automatic_masks/report.json', raw_json(report))
        write(directory/'mask-inventory.json', inventory_raw)
        write(base/'automatic_masks/mask-inventory.json', inventory_raw)
        aggregates.append(dict(episode_index=episode, status='pass', frames=count,
            automatic_masks=str(directory.relative_to(frontend)), report=report_pin))
        numerical_rows.append(dict(episode=episode, original_frames=count,
            status='complete_full4d_visual_diagnostic_not_quality_pass' if episode in (9, 14) else 'fail',
            frontend=dict(producer_revision=runtime.FRONTEND, source_directory=str(directory),
                files={'report.json': report_pin}, aggregate_report=None,
                mask_inventory=mask_inventory, byte_identical_copy=True,
                hand_modified_masks=False, model_calls=0)))
    aggregate = dict(schema='world_reward.gemini_sam31_tracking.v1', producer_revision=runtime.FRONTEND,
        status='complete_diagnostic_not_quality_pass', ground_truth_used=False,
        manual_labels=False, original_frame_grid_preserved=True, episodes=aggregates)
    native_pin = write(frontend/'native-report.json', raw_json(aggregate))
    monkeypatch.setattr(runtime, 'FRONTEND_NATIVE_PIN', native_pin)
    for row in numerical_rows: row['frontend']['aggregate_report'] = native_pin
    numerical = dict(schema='world_reward.gemini_full4d.v1', producer_revision=runtime.SOURCE,
        status='complete_diagnostic_not_quality_pass', frontend_producer=runtime.FRONTEND,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], baseline_modified=False,
        per_frame_alignment=False, per_frame_scale=False, object_geometry_clip_constant=True,
        episodes=numerical_rows,
        inputs=[dict(episode=episode, total=count, video=str(tmp_path
            /'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{episode:06d}.mp4'),
            video_pin=dict(bytes=100, sha256=SHA)) for episode in (9, 1, 14, 7)])
    write(experiment/'report.json', raw_json(numerical))
    return frontend, experiment, aggregate, numerical, inventory


def test_artifact_ledger_stream_and_read_only_bound_identity(tmp_path):
    path = tmp_path/'source'; pin = write(path, b'manufactured')
    ledger = runtime.ArtifactLedger()
    assert ledger.record(path, pin) == pin
    assert ledger.read(path, pin) == b'manufactured'
    assert json.loads(json.dumps(ledger.records))[str(path)] == pin
    ledger.verify()


def test_read_induced_atime_is_not_a_source_mutation(tmp_path, monkeypatch):
    path = tmp_path/'source'; write(path, b'manufactured')
    ledger = runtime.ArtifactLedger(); ledger.record(path)
    s = path.stat(); keys = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns',
                            'st_ctime_ns', 'st_nlink', 'st_uid', 'st_gid')
    atimes = iter((s.st_atime_ns+10**6, s.st_atime_ns+2*10**6))
    def changed_access_stat():
        return SimpleNamespace(**{key: getattr(s, key) for key in keys}, st_atime_ns=next(atimes))
    monkeypatch.setattr(runtime, 'canonical', lambda _: SimpleNamespace(lstat=changed_access_stat))
    assert runtime.ArtifactLedger.stable_stat(path) == ledger.stats[str(path)]
    assert runtime.ArtifactLedger.stable_stat(path) == ledger.stats[str(path)]


@pytest.mark.parametrize('kind', ['changed', 'symlink', 'hardlink', 'directory', 'too_large'])
def test_artifact_ledger_rejects_changed_or_noncanonical_source(tmp_path, kind):
    path = tmp_path/'source'; write(path, b'manufactured')
    ledger = runtime.ArtifactLedger(); ledger.record(path)
    if kind == 'changed': path.write_bytes(b'manufactureD')
    if kind == 'symlink': path.unlink(); path.symlink_to(tmp_path/'missing')
    if kind == 'hardlink': os.link(path, tmp_path/'other')
    if kind == 'directory': path.unlink(); path.mkdir()
    with pytest.raises(ValueError):
        ledger.record(path, maximum=3 if kind == 'too_large' else 1024)


def test_independent_native_report_authenticates_copied_masks_and_transport(tmp_path, monkeypatch):
    frontend, experiment, _, _, expected = frontend_fixture(tmp_path, monkeypatch)
    ledger = runtime.ArtifactLedger(); actual_base, rows = runtime.saved_frontend(ledger)
    assert actual_base == frontend and list(rows) == [9, 1, 14, 7]
    numerical = runtime.saved_numerical_frontend(ledger, experiment, rows)
    assert set(numerical) == {9, 14}
    inventory, report = runtime.authenticated_masks(ledger, frontend, rows[9],
        experiment/'outputs/episode_000009', 9, 3)
    assert inventory == expected and numerical[9]['mask_inventory'] == report['mask_inventory']
    ledger.verify()


@pytest.mark.parametrize('kind', ['native_bytes', 'producer', 'oracle', 'cohort', 'grid'])
def test_native_frontend_rejects_hash_or_contract_change(tmp_path, monkeypatch, kind):
    frontend, _, aggregate, _, _ = frontend_fixture(tmp_path, monkeypatch)
    if kind == 'native_bytes': aggregate['unapproved'] = True
    if kind == 'producer': aggregate['producer_revision'] = 'f'*40
    if kind == 'oracle': aggregate['ground_truth_used'] = True
    if kind == 'cohort': aggregate['episodes'].reverse()
    if kind == 'grid': aggregate['original_frame_grid_preserved'] = False
    pin = write(frontend/'native-report.json', raw_json(aggregate))
    if kind != 'native_bytes': monkeypatch.setattr(runtime, 'FRONTEND_NATIVE_PIN', pin)
    with pytest.raises(ValueError): runtime.saved_frontend(runtime.ArtifactLedger())


@pytest.mark.parametrize('kind', ['copied_report', 'copied_inventory', 'missing_frame', 'oracle', 'inventory_pin'])
def test_masks_cannot_be_self_pinned_or_omit_missing_frames(tmp_path, monkeypatch, kind):
    frontend, experiment, aggregate, _, inventory = frontend_fixture(tmp_path, monkeypatch)
    row = aggregate['episodes'][0]; directory = frontend/'episode_000009/automatic_masks'
    copied = experiment/'outputs/episode_000009/automatic_masks'
    if kind == 'copied_report': write(copied/'report.json', b'{}\n')
    if kind == 'copied_inventory': write(copied/'mask-inventory.json', b'{}\n')
    if kind in ('missing_frame', 'inventory_pin'):
        if kind == 'missing_frame': del inventory['1/000001.png']
        else: inventory['1/000001.png']['sha256'] = 'not-an-sha'
        for destination in (directory, copied): write(destination/'mask-inventory.json', raw_json(inventory))
    if kind == 'oracle':
        report = json.loads((directory/'report.json').read_text()); report['oracle_modes'] = ['oracle']
        row['report'] = write(directory/'report.json', raw_json(report)); write(copied/'report.json', raw_json(report))
    with pytest.raises(ValueError):
        runtime.authenticated_masks(runtime.ArtifactLedger(), frontend, row,
                                    experiment/'outputs/episode_000009', 9, 3)


@pytest.mark.parametrize('kind', ['native_pin', 'report_pin', 'inventory', 'source', 'manual', 'oracle', 'cohort', 'video_path', 'video_pin'])
def test_original_transport_is_not_an_independent_replacement_prediction(tmp_path, monkeypatch, kind):
    _, experiment, aggregate, numerical, _ = frontend_fixture(tmp_path, monkeypatch)
    receipt = numerical['episodes'][0]['frontend']
    if kind == 'native_pin': receipt['aggregate_report'] = dict(bytes=9, sha256=SHA)
    if kind == 'report_pin': receipt['files']['report.json'] = dict(bytes=9, sha256=SHA)
    if kind == 'inventory': receipt['byte_identical_copy'] = False
    if kind == 'source': receipt['source_directory'] = '/foreign'
    if kind == 'manual': receipt['hand_modified_masks'] = True
    if kind == 'oracle': numerical['oracle_modes'] = ['GT']
    if kind == 'cohort': numerical['episodes'].reverse()
    if kind == 'video_path': numerical['inputs'][0]['video'] = '/foreign.mp4'
    if kind == 'video_pin': numerical['inputs'][0]['video_pin']['sha256'] = 'invalid'
    write(experiment/'report.json', raw_json(numerical))
    with pytest.raises(ValueError):
        runtime.saved_numerical_frontend(runtime.ArtifactLedger(), experiment,
                                         {row['episode_index']: row for row in aggregate['episodes']})


@pytest.mark.parametrize('fps', [0., -30., float('nan'), float('inf')])
def test_actual_fps_rejects_invalid_source_time(monkeypatch, fps):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(CAP_PROP_FPS=3))
    with pytest.raises(ValueError): runtime.actual_fps(SimpleNamespace(get=lambda _: fps))


def test_fps_uses_actual_decoder_not_assumed_30(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(CAP_PROP_FPS=3))
    capture = SimpleNamespace(get=lambda _: 23.976)
    assert runtime.actual_fps(capture) == 23.976
    assert runtime.actual_fps(capture, 23.976) == 23.976
    with pytest.raises(ValueError): runtime.actual_fps(capture, 30.)


class LK:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def calcOpticalFlowPyrLK(self, previous, gray, current, initial):
        assert current.dtype == np.float32 and np.isfinite(current).all()
        self.calls.append(current.copy()); return self.responses.pop(0)


def points(): return np.array([[3., 4.], [5., 6.], [7., 8.]], np.float32)


@pytest.mark.parametrize('stage', ['forward', 'backward'])
def test_lk_normal_loss_is_unknown_not_exception_or_fake_track(stage):
    p = points(); valid = (p[:, None], np.ones((3, 1), np.uint8), None)
    lk = LK([(None, None, None)] if stage == 'forward' else [valid, (None, None, None)])
    output, support = runtime.lk_step(lk, np.zeros((12, 16)), np.zeros((12, 16)), p)
    assert not support.any() and output.dtype == np.float32
    assert len(lk.calls) == (1 if stage == 'forward' else 2)
    if stage == 'forward': assert np.isnan(output).all()


def test_lk_backward_only_receives_finite_forward_supported_subset():
    p = points(); q = p.copy(); q[1] = np.nan
    lk = LK([(q[:, None], np.array([[1], [1], [0]], np.uint8), None),
             (p[:1, None], np.ones((1, 1), np.uint8), None)])
    output, support = runtime.lk_step(lk, np.zeros((12, 16)), np.zeros((12, 16)), p.astype(np.float64))
    np.testing.assert_array_equal(support, [True, False, False])
    np.testing.assert_array_equal(lk.calls[1][:, 0], p[:1])
    assert output.dtype == np.float32


def test_lk_forward_backward_status_bounds_and_error_are_all_required():
    p = points(); q = p.copy(); q[2] = [20, 8]
    back = p.copy(); back[1] += 3
    lk = LK([(q[:, None], np.ones((3, 1), np.uint8), None),
             (back[:, None], np.ones((3, 1), np.uint8), None)])
    _, support = runtime.lk_step(lk, np.zeros((12, 16)), np.zeros((12, 16)), p)
    np.testing.assert_array_equal(support, [True, False, False])


@pytest.mark.parametrize('kind', ['shape', 'status', 'dtype', 'back_shape', 'input'])
def test_lk_malformed_api_results_reject_instead_of_silent_indexing(kind):
    p = points(); q = p[:, None]; status = np.ones((3, 1), np.uint8)
    if kind == 'shape': q = p
    if kind == 'status': status[1] = 2
    if kind == 'dtype': q = q.astype(np.int32)
    if kind == 'input': p[0] = np.inf
    back = p[:, None] if kind != 'back_shape' else p
    lk = LK([(q, status, None), (back, np.ones((3, 1), np.uint8), None)])
    with pytest.raises(ValueError): runtime.lk_step(lk, np.zeros((12, 16)), np.zeros((12, 16)), p)


def test_path_gate_detects_erased_curved_motion_even_with_correct_endpoints():
    u = np.linspace(0, 1, 24)
    truth = np.stack((u, .5*np.sin(2*np.pi*u), u*0), axis=1)[:, None]
    old = truth.copy(); old[1:-1, 0, 2] += .01*(-1.)**np.arange(22)
    straight = truth.copy(); straight[:, 0, 1] = 0
    assert np.allclose(straight[[0, -1]], truth[[0, -1]])
    assert not runtime.motion_gate(truth, old, straight)['passed']
    assert runtime.motion_gate(truth, old, truth)['passed']


def test_pose_motion_summary_uses_centroid_and_seconds_not_acceleration_score():
    vertices = np.array([[-1., -1, 0], [1., -1, 0], [0., 2, 0]])
    r = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
    t = np.array([[0., 0, 2], [.01, 0, 2], [.02, 0, 2]])
    summary = runtime.pose_motion_summary(vertices, r, t, 25.)
    assert summary['full_centroid_path_m'] == pytest.approx(.02)
    assert summary['centroid_speed_m_s_median'] == pytest.approx(.25)
    assert summary['angular_speed_rad_s_median'] == 0
    assert 'not_accuracy' in summary['scope']


def test_new_missing_pose_protocol_does_not_send_gt_or_fake_observations(monkeypatch):
    seen = []
    def fit(vertices, points, tracks, visible, rotations, translations, observed, K, indices, fps, cfg):
        assert not observed[9:15].any() and np.isnan(tracks[9:15]).all()
        assert not visible[9:15].any() and np.isfinite(rotations).all() and np.isfinite(translations).all()
        assert np.array_equal(indices, np.arange(24)) and cfg is runtime.CFG
        seen.append(True)
        return SimpleNamespace(rotations=rotations.copy(), translations=translations.copy(),
                               diagnostics={'converged': False})
    monkeypatch.setattr(runtime, 'refine_sequence', fit)
    report = runtime.manufactured_missing_pose_gate(20261012)
    assert seen and report['missing_pose_frames'] == 6
    assert report['protocol'] == 'new_manufactured_missing_pose_full_motion_v2'
    assert report['original_DEV_RESERVED_unchanged'] is True
    assert report['converged'] is False and not report['passed']
    assert report['external_quality_or_probability_calibration'] is False
    json.dumps(report, allow_nan=False)


def test_source_bound_failure_receipt_is_serializable_and_stops_before_challenge(tmp_path, monkeypatch):
    revision = 'f'*40; code = tmp_path/'jobs'/revision/runtime.ENTRY/'code'
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setenv('WR_ROOT', str(tmp_path)); monkeypatch.setenv('WR_CODE', str(code))
    monkeypatch.setenv('WR_CODE_REVISION', revision)
    out = tmp_path/'results'/('sequence-pose-probe-'+revision); out.mkdir(parents=True)
    (out/'.container.cid').write_bytes(b'a'*64)
    for name in runtime.HELPERS: write(code/name, b'manufactured source\n')
    monkeypatch.setattr(runtime, 'source', lambda *args: {'manufactured': True})
    calls = []
    def gate(seed, reserved, config=None):
        calls.append((seed, reserved)); return dict(passed=not reserved, converged=True)
    monkeypatch.setattr(runtime, 'manufactured_gate', gate)
    monkeypatch.setattr(runtime, 'saved_frontend', lambda _: pytest.fail('Challenge accessed after failed RESERVED'))
    with pytest.raises(SystemExit) as error: runtime.run()
    assert error.value.code == 1 and calls == [(20261010, False), (20261011, True)]
    receipt = json.loads((out/'report.json').read_text())
    assert receipt['status'] == 'fail' and receipt['reserved']['passed'] is False
    assert 'RESERVED' in receipt['error'] and receipt['full_4D_export_replaced'] is False
    assert not (out/'report.json').stat().st_mode & 0o222


def test_original_config_and_splits_remain_frozen():
    assert asdict(runtime.CFG) == dict(pixel_sigma=1.5, prior_centroid_sigma_diameter=.05,
        prior_rotation_sigma_rad=.15, acceleration_sigma_diameter_s2=20.,
        angular_acceleration_sigma_rad_s2=100., prior_weight=.1, temporal_weight=.02,
        max_nfev=60, development_reference='manufactured_nonchallenge_sequence_DEV_20261010_v1')
    raw = (ROOT/'infra/sequence_pose_probe.py').read_text()
    assert runtime.profile_config(ROOT,'v1') == (runtime.CFG,(20261010,20261011,20261012))
    assert raw.index('missing_pose_validation') < raw.index('saved_frontend(ledger)', raw.index('def run(profile='))


@pytest.mark.parametrize('kind', ['extra', 'empty', 'symlink', 'hardlink', 'invalid', 'large'])
def test_runtime_accepts_only_exact_owned_cid_control_file(tmp_path, kind):
    out = tmp_path/'output'; out.mkdir(); cid = out/'.container.cid'; cid.write_bytes(b'a'*64)
    if kind == 'extra': (out/'old.npz').write_bytes(b'old')
    if kind == 'empty': cid.unlink()
    if kind == 'symlink': cid.unlink(); cid.symlink_to(tmp_path/'missing')
    if kind == 'hardlink': os.link(cid, tmp_path/'other')
    if kind == 'invalid': cid.write_bytes(b'docker-name')
    if kind == 'large': cid.write_bytes(b'a'*66)
    with pytest.raises(ValueError): runtime.fresh_runtime_output(out)


def test_runtime_owned_cid_may_exist_before_python_starts(tmp_path):
    out = tmp_path/'output'; out.mkdir(); (out/'.container.cid').write_bytes(b'a'*64+b'\n')
    runtime.fresh_runtime_output(out)


def test_wrapper_is_cpu_only_explicit_readonly_source_and_owned_cleanup():
    raw = (ROOT/'infra/run_sequence_pose_probe.sh').read_text()
    assert 'exec 9<' in raw and 'exec 9>' not in raw
    assert '.world-reward-sequence-pose-probe.lock' in raw and '.world-reward-h100.lock' not in raw
    assert 'os.O_EXCL' in raw and 'os.O_NOFOLLOW' in raw and 's.st_nlink!=1' in raw
    assert '--gpus' not in raw and 'CUDA_VISIBLE_DEVICES=-1' in raw
    assert 'src=$ROOT/experiments/full4d-v1-$SOURCE' in raw and 'src=$ROOT/experiments,dst=' not in raw
    assert 'src=$ROOT/results/gemini-sam31-$FRONTEND' in raw
    assert '--network none' in raw and '--read-only' in raw
    assert 'trap cleanup EXIT' in raw and 'world_reward.sequence_pose.owner' in raw
    assert 'timeout 20s docker rm -f "$cid"' in raw and 'docker rm -f "$NAME"' not in raw
    assert 'container_absence_verified' in raw and '--kill-after=10s 1200s' in raw


def test_helper_closure_and_output_are_bound_not_serialized_raw_bytes():
    assert {'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/sequence_pose.py',
            'src/world_reward/point_surface_queries.py', 'src/world_reward/point_pose_cost.py',
            'src/world_reward/mesh_geometry.py'} <= set(runtime.HELPERS)
    raw = (ROOT/'infra/sequence_pose_probe.py').read_text()
    assert 'source(ROOT, code, revision, ENTRY, HELPERS)' in raw
    assert 'output=ledger.record(path)' in raw and 'output=pin(path)' not in raw
    assert "ledger.record(target_path,pins['target.npy'])" in raw
    assert 'provisional_budget_limited_not_quality_qualified' in raw


def test_v2_is_separate_frozen_external_development_profile():
    config, seeds = runtime.profile_config(ROOT,'v2')
    assert seeds == (20261020,20261022,20261023)
    assert config.temporal_weight == .2 and runtime.CFG.temporal_weight == .02
    assert config.development_reference.startswith('manufactured_nonchallenge_v2_DEV')
    with pytest.raises(ValueError): runtime.profile_config(ROOT,'challenge-tuned')

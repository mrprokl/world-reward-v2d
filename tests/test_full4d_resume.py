"""Synthetic continuation gates only; no original clips, Docker or GPU calls."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import full4d_resume as resume


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def pose_case(tmp_path):
    base, code = tmp_path/'episode_000009', tmp_path/'code'
    script = 'infra/object_pose_smoke.py'; script_sha = write(code/script, b'synthetic original producer')
    (code/script).chmod(0o444)
    item = dict(episode=9, total=96, video_pin=dict(sha256='a'*64, bytes=1))
    directory = base/'object_pose_full_surface'
    source = {}
    for field, relative in (('object_report_sha256','object_grounded/report.json'),
                           ('alignment_report_sha256','scale_smoke/report.json'),
                           ('full_depth_report_sha256','depth_full/report.json')):
        source[field] = write(base/relative, b'opaque synthetic upstream receipt')
    report = dict(status='pass', stage=resume.STAGE_NAMES['object_pose'], episode_index=9,
        input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        script_sha256=script_sha, input_sha256='a'*64,
        frames=[{'frame_index': index} for index in range(96)], **source)
    report['geometry_and_poses_sha256'] = write(directory/'geometry_and_poses.npz', b'opaque synthetic numerical payload')
    report['fixed_canonical_mesh_sha256'] = write(directory/'object_fixed_canonical.glb', b'opaque synthetic mesh')
    write(directory/'report.json', json.dumps(report).encode())
    return base, code, script, item, report


def test_existing_full_pose_pass_is_hashed_without_decode_or_rerun(pose_case):
    base, code, script, item, _ = pose_case
    evidence = resume.report_pass(base, 'object_pose', script, code, item)
    assert evidence['files'] == 2 and evidence['inference_replayed'] is False
    assert (base/'object_pose_full_surface/report.json').stat().st_mode & 0o222


@pytest.mark.parametrize('fault', ['failed', 'short_timeline', 'wrong_video', 'changed_source', 'changed_pose', 'wrong_oracle'])
def test_existing_pose_refuses_nonpass_or_changed_original_lineage(pose_case, fault):
    base, code, script, item, report = pose_case
    if fault == 'failed': report['status'] = 'fail'
    elif fault == 'short_timeline': report['frames'].pop()
    elif fault == 'wrong_video': report['input_sha256'] = 'b'*64
    elif fault == 'changed_source':
        (code/script).chmod(0o644)
        write(code/script, b'new numerical source must not replace original')
        (code/script).chmod(0o444)
    elif fault == 'changed_pose': write(base/'object_pose_full_surface/geometry_and_poses.npz', b'changed')
    else: report['oracle_modes'] = ['synthetic_forbidden_oracle']
    write(base/'object_pose_full_surface/report.json', json.dumps(report).encode())
    with pytest.raises(ValueError): resume.report_pass(base, 'object_pose', script, code, item)


def test_missing_stage_is_fresh_but_occupied_incomplete_stage_is_never_deleted(tmp_path):
    base = tmp_path/'episode'; base.mkdir()
    item = dict(episode=9, total=96)
    assert resume.report_pass(base, 'body_full', 'infra/body_smoke.py', tmp_path/'code', item) is None
    output = base/'body_full'; output.mkdir(); original = output/'predictions.npz'; original.write_bytes(b'partial')
    with pytest.raises(ValueError, match='complete PASS'):
        resume.report_pass(base, 'body_full', 'infra/body_smoke.py', tmp_path/'code', item)
    assert original.read_bytes() == b'partial'


@pytest.mark.parametrize('stage,expected_locks', [('object_pose', 0), ('body_full', 1), ('refined', 1)])
def test_cooperative_workers_use_original_command_and_serialize_learned_stages(monkeypatch, tmp_path, stage, expected_locks):
    calls = []; counters = {'locks': 0}
    @contextmanager
    def model_lock():
        counters['locks'] += 1
        yield
    oldcode = tmp_path/'old_code'; experiment = tmp_path/'experiment'; control = tmp_path/'continuation'
    (control/'logs').mkdir(parents=True)
    driver = SimpleNamespace(command=lambda *args: calls.append(args) or ['opaque', 'original', 'command'])
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(resume.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(resume, 'run_process', lambda cmd, log, seconds, stop: 0)
    resume.native(driver, oldcode, experiment, {'body_image': 'exact_original_image'}, {'episode': 9},
        (stage, 'infra/original.py', (), 'body_image', stage, None), control, 600, 'c'*40, model_lock(), threading.Event())
    assert counters['locks'] == expected_locks
    assert calls[0][0] == oldcode and calls[0][-1] == resume.OLD
    assert (control/'logs'/f'9-{stage}.log').is_file()
    assert not (experiment/'logs').exists()


def test_worker_stops_before_launch_and_keeps_original_reports(monkeypatch, tmp_path):
    stop = threading.Event(); stop.set()
    driver = SimpleNamespace(command=lambda *args: ['not', 'executed'])
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(resume, 'run_process', lambda *a, **k: pytest.fail('no launch after stop'))
    with pytest.raises(ValueError, match='interrupted'):
        resume.native(driver, tmp_path, tmp_path, {'body_image': 'exact'}, {'episode': 9},
            ('refined', 'infra/original.py', (), 'body_image', 'refined', None), tmp_path,
            1, 'c'*40, threading.Lock(), stop)


def test_full_original_report_and_frozen_population_are_not_relabelled():
    assert resume.OLD == 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
    assert resume.ORIGINAL_REPORT == dict(bytes=8613,
        sha256='d20a4f074c0bf5ffeb2e489a995e66655f5a65773e445638cab46d6b197b1e40')
    source = Path(resume.__file__).read_text()
    assert "prior.get('status') == 'running'" in source
    assert 'other_workers_only_after_scout_pass=True' in source
    assert 'ThreadPoolExecutor(max_workers=2)' in source
    assert "with (ROOT/'jobs/.world-reward-h100.lock').open('a')" in source
    assert 'unlink(' not in source and 'rmtree(' not in source
    assert "'resample'" not in source

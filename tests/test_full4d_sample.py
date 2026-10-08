import json
import hashlib
import os
from pathlib import Path
import random

import pytest

import full4d_sample as sample


CODE = Path(__file__).resolve().parents[1]
REV = 'a' * 40


def test_sample_frozen_once_before_results():
    cfg = sample.load_config(CODE)
    assert cfg['episodes'] == [9, 1, 14, 7] == random.Random(20261008).sample(range(30), 4)
    assert cfg['resample_failed_clips'] is False
    assert cfg['native_refinement_steps'] == 300


def test_random_sample_may_not_cherry_pick(tmp_path):
    (tmp_path/'configs').mkdir()
    cfg = sample.load_config(CODE); cfg['episodes'] = [8,9,26,7]
    (tmp_path/sample.CONFIG).write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Frozen uniform'):
        sample.load_config(tmp_path)


def test_stage_sequence_complete_and_surface_fixed():
    stages = [s[0] for s in sample.STAGES]
    assert stages == ['body_smoke','depth_smoke','scale_smoke','object_grounded','surface',
        'body_full','depth_full','adapter','object_pose','inputs','prepare','forward','refined','export','video']
    assert sample.STAGES[4][2] == ('--domain','surface')
    assert sample.STAGES[8][2] == ('--full-video','--mesh-source','surface')


def test_native_mounts_exclude_baseline_and_keep_old_outputs_readonly(monkeypatch):
    cfg = sample.load_config(CODE)
    experiment = sample.ROOT/'experiments'/('full4d-v1-'+REV)
    monkeypatch.setenv('WR_OUTPUT_PREFIX',str((experiment/'outputs').relative_to(sample.ROOT)))
    monkeypatch.setattr(sample,'canonical',lambda p:p)
    argv = sample.command(CODE,experiment,cfg,9,'forward','infra/cari_full_forward.py',(),cfg['body_image'],'test',REV)
    mounts = [argv[i+1] for i,x in enumerate(argv) if x == '--mount']
    assert not any('src='+str(sample.ROOT/'outputs')+',' in x for x in mounts)
    assert any('src='+str(experiment/'outputs')+',' in x and x.endswith(',readonly') for x in mounts)
    assert any('cari_shared_forward_v1' in x and not x.endswith(',readonly') for x in mounts)
    assert 'WR_PIN_ROOT='+str(experiment/'pins') in argv
    assert '--network' in argv and argv[argv.index('--network')+1] == 'none'


def test_object_native_setting_and_no_manual_prompts(monkeypatch):
    cfg=sample.load_config(CODE); experiment=sample.ROOT/'experiments'/('full4d-v1-'+REV)
    monkeypatch.setenv('WR_OUTPUT_PREFIX',str((experiment/'outputs').relative_to(sample.ROOT)))
    monkeypatch.setattr(sample,'canonical',lambda p:p)
    argv=sample.command(CODE,experiment,cfg,9,'object_grounded','infra/object_smoke.py',
        ('--aligned-pointmap',),cfg['object_image'],'test',REV)
    assert 'LIDRA_SKIP_INIT=1' in argv
    assert '--aligned-pointmap' in argv
    assert not any('oracle' in value.lower() for value in argv)


@pytest.mark.parametrize('stage,script', [
    ('object_pose', 'infra/object_pose_smoke.py'),
    ('inputs', 'infra/cari_prepare.py'),
])
def test_surface_consumers_mount_complete_authenticated_original_snapshots_readonly(monkeypatch, stage, script):
    """A valid proposal still fails before inference if its source ancestry is hidden."""
    cfg = sample.load_config(CODE)
    experiment = sample.ROOT / 'experiments' / ('full4d-v1-' + REV)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', str((experiment / 'outputs').relative_to(sample.ROOT)))
    monkeypatch.setattr(sample, 'canonical', lambda path: path)
    argv = sample.command(CODE, experiment, cfg, 9, stage, script,
                          ('--mesh-source', 'surface'), cfg['body_image'], 'test', REV)
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == '--mount']
    expected = []
    for config, entry in (
        ('surface_qslim_qualification_pins.json', 'run_surface_qslim_qualify'),
        ('surface_identity_qualification_pins.json', 'run_surface_identity_qualify'),
    ):
        revision = json.loads((CODE / 'configs' / config).read_text())['producer_revision']
        original = sample.ROOT / 'jobs' / revision / entry
        expected.append(original)
        # Parent includes code AND revision/source-sha256 markers. Source-only
        # mounts cannot satisfy surface_geometry_loader._snapshot's full census.
        assert f'type=bind,src={original},dst={original},readonly' in mounts
        assert not any(f'src={original},' in mount and not mount.endswith(',readonly') for mount in mounts)
    assert f'type=bind,src={sample.ROOT / "jobs"},dst={sample.ROOT / "jobs"},readonly' not in mounts
    assert all(path.name in ('run_surface_qslim_qualify', 'run_surface_identity_qualify') for path in expected)


@pytest.mark.parametrize('field,value', [
    ('geometry_policy', 'repair_or_drop_object'), ('grounding_config', 'configs/other.json'),
    ('virtual_floor_only', False), ('pilot_budget_seconds', 999999),
    ('body_image', 'sha256:' + 'f' * 64), ('scope', 'heldout_verified'),
    ('challenge_performance_verified', True), ('extra_unfrozen_field', 1),
])
def test_immutable_sample_source_protocol_fields_enforced(tmp_path, field, value):
    (tmp_path/'configs').mkdir()
    cfg = sample.load_config(CODE)
    cfg[field] = value
    (tmp_path/sample.CONFIG).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Frozen source'):
        sample.load_config(tmp_path)


def pose_fixture(monkeypatch, tmp_path):
    root = tmp_path/'runtime'
    code = root/'jobs'/REV/sample.ENTRY/'code'
    script = code/'infra/object_pose_smoke.py'
    script.parent.mkdir(parents=True)
    script.write_bytes(b'# Tiny frozen original pose source\n')
    script.chmod(0o444)
    code.chmod(0o555)
    for name, raw in (('revision', (REV+'\n').encode()), ('source-sha256', ('b'*64+'\n').encode())):
        (code.parent/name).write_bytes(raw)
        (code.parent/name).chmod(0o444)
    monkeypatch.setattr(sample, 'ROOT', root)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{REV}/outputs')
    monkeypatch.setenv('WR_CODE', str(root/'jobs'/('b'*40)/'current/code'))
    monkeypatch.setenv('WR_CODE_REVISION', 'b'*40)
    out = root/f'experiments/full4d-v1-{REV}/outputs/episode_000017/object_pose_full_surface'
    out.mkdir(parents=True)
    rows = {}
    for name in ('geometry_and_poses.npz', 'object_fixed_canonical.glb'):
        raw = ('opaque existing predicted '+name).encode()
        (out/name).write_bytes(raw)
        rows[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    report = dict(stage='fixed_scale_full_object_pose_initializer', status='pass', episode_index=17,
        input_track='track_1', mesh_source='surface', ground_truth_used=False, hand_labeled_test=False,
        oracle_modes=[], fixed_shape=True, original_frame_coverage_verified=True, execution_verified=True,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        geometry_and_poses_sha256=rows['geometry_and_poses.npz']['sha256'],
        fixed_canonical_mesh_sha256=rows['object_fixed_canonical.glb']['sha256'],
        frames=[dict(frame_index=index) for index in range(96)])
    (out/'report.json').write_text(json.dumps(report))
    return root, code, out, report


def test_pose_seal_original_producer_before_input_no_decode_or_byte_changes(monkeypatch, tmp_path):
    root, code, out, report = pose_fixture(monkeypatch, tmp_path)
    before = {p.name:p.read_bytes() for p in out.iterdir()}
    value = sample.seal_pose(root,17,96,producer_code=code,producer_revision=REV)
    assert value['producer_revision'] == REV and value['total_frames'] == 96
    assert value['numeric_payload_decoded'] is False and value['inference_replayed'] is False
    assert {p.name:p.read_bytes() for p in out.iterdir()} == before
    assert out.stat().st_mode & 0o777 == 0o555
    assert all(p.stat().st_mode & 0o777 == 0o444 for p in out.iterdir())
    # Idempotent metadata sealing does not resume/rewrite any prediction.
    assert sample.seal_pose(root,17,96,producer_code=code,producer_revision=REV) == value


@pytest.mark.parametrize('mutation', ['fail','episode','gt','manual','oracle','script','timeline','meshhash','npzhash','extra','symlink','hardlink','namespace','source_revision'])
def test_bad_pose_sources_stop_before_any_chmod(monkeypatch,tmp_path,mutation):
    root, code, out, report = pose_fixture(monkeypatch,tmp_path)
    if mutation == 'extra': (out/'unowned.txt').write_bytes(b'foreign')
    elif mutation == 'symlink':
        (out/'object_fixed_canonical.glb').unlink()
        (out/'object_fixed_canonical.glb').symlink_to(code/'infra/object_pose_smoke.py')
    elif mutation == 'hardlink': (tmp_path/'alias').hardlink_to(out/'geometry_and_poses.npz')
    elif mutation == 'namespace': monkeypatch.setenv('WR_OUTPUT_PREFIX','outputs')
    elif mutation == 'source_revision': report['producer_revision'] = 'b'*40
    elif mutation == 'fail': report['status'] = 'fail'
    elif mutation == 'episode': report['episode_index'] = 2
    elif mutation == 'gt': report['ground_truth_used'] = True
    elif mutation == 'manual': report['hand_labeled_test'] = True
    elif mutation == 'oracle': report['oracle_modes'] = ['oracle']
    elif mutation == 'script': report['script_sha256'] = 'f'*64
    elif mutation == 'timeline': report['frames'][10]['frame_index'] = 11
    elif mutation == 'meshhash': report['fixed_canonical_mesh_sha256'] = 'f'*64
    elif mutation == 'npzhash': report['geometry_and_poses_sha256'] = 'f'*64
    (out/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        sample.seal_pose(root,17,96,producer_code=code,producer_revision=REV)
    assert (out/'report.json').stat().st_mode & 0o777 == 0o644
    assert out.stat().st_mode & 0o777 != 0o555

import json
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

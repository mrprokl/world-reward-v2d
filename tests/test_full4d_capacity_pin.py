"""Tiny source-bound mixed producer receipt publication; no numeric decoding."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import full4d_capacity_pin as capacity
import full4d_pins as publication

REPO = Path(__file__).resolve().parents[1]
REV = 'a'*40


def write(path,raw,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def case(monkeypatch,tmp_path):
    root = tmp_path/'root'
    code = root/'jobs'/REV/'run_full4d_resume/code'
    original = root/'jobs'/capacity.ORIGINAL/'run_full4d_sample/code'
    oldsource = subprocess.check_output(['git','show',capacity.ORIGINAL+':infra/cari_prepare.py'],cwd=REPO)
    write(original/'infra/cari_prepare.py',oldsource)
    for name in ('cari_prepare.py','surface_pose_report_capacity.py','full4d_capacity_pin.py'):
        write(code/'infra'/name,(REPO/'infra'/name).read_bytes())
    monkeypatch.setattr(capacity,'ROOT',root)
    monkeypatch.setenv('WR_OUTPUT_PREFIX',f'experiments/full4d-v1-{capacity.ORIGINAL}/outputs')
    base = publication.episode_output(root,17)
    spec = capacity.inputs.PublicClipSpec(17,96,'front_stereo_camera_left',1152,1536)
    names = capacity.inputs.source_paths(spec,object_source='surface')
    for name in names:
        write(root/name,b'opaque complete public source',mode=0o644)
    helper = publication.identity(code/'infra/surface_pose_report_capacity.py')
    report = dict(stage='world_reward_native_cari_inputs',status='pass',episode_index=17,
        frames=96,producer_revision=REV,script_sha256=publication.identity(code/'infra/cari_prepare.py')['sha256'],
        input_track='track_1',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
        object_source='surface',original_frame_coverage_verified=True,
        surface_geometry_validation=dict(files={str(code/'infra/surface_pose_report_capacity.py'):helper}))
    report_path = root/capacity.inputs.relative_paths(spec)['input_report']
    write(report_path,json.dumps(report).encode(),mode=0o644)
    pinroot = root/f'experiments/full4d-v1-{capacity.ORIGINAL}/pins'
    pinroot.mkdir()
    monkeypatch.setattr(publication,'pin_path',lambda _,episode,role:pinroot/f'{role}_{episode:06d}.json')
    monkeypatch.setattr(capacity,'source',lambda root,code,revision,entry,helpers:{'revision':revision,'entry':entry})
    monkeypatch.setattr(capacity.inputs,'validate_reports',lambda *args:None)
    return root,code,original,report_path,pinroot,report,names


def test_new_input_producer_does_not_relabel_original_inference(monkeypatch,tmp_path):
    root,code,original,path,pinroot,report,names = case(monkeypatch,tmp_path)
    value = capacity.input_pin(root,code,REV,17,96,original_code=original,original_revision=capacity.ORIGINAL)
    assert value['input_report']['producer_revision'] == REV != capacity.ORIGINAL
    assert value['input_report']['script_sha256'] == report['script_sha256']
    assert len(value['source_files']) == 15
    assert all(name.startswith(f'experiments/full4d-v1-{capacity.ORIGINAL}/outputs/') for name in names)
    assert all((root/name).stat().st_mode & 0o777 == 0o444 for name in names)
    assert json.loads((pinroot/'input_000017.json').read_text()) == value
    with pytest.raises(FileExistsError):
        capacity.input_pin(root,code,REV,17,96)


@pytest.mark.parametrize('fault',['oldnamespace','wrongsource','helper','producer','numeric'])
def test_bad_mixed_source_stops_without_pin_or_chmod(monkeypatch,tmp_path,fault):
    root,code,original,path,pinroot,report,names = case(monkeypatch,tmp_path)
    if fault == 'oldnamespace': monkeypatch.setenv('WR_OUTPUT_PREFIX','outputs')
    elif fault == 'wrongsource': code = root/'jobs'/REV/'run_full4d_sample/code'
    elif fault == 'helper': report['surface_geometry_validation']['files'] = {}
    elif fault == 'producer': report['producer_revision'] = capacity.ORIGINAL
    elif fault == 'numeric':
        source = code/'infra/cari_prepare.py'
        source.chmod(0o644)
        source.write_bytes(source.read_bytes().replace(b'error > 1e-5',b'error > 1e-4',1))
        source.chmod(0o444)
    path.write_text(json.dumps(report))
    with pytest.raises((ValueError,FileNotFoundError)):
        capacity.input_pin(root,code,REV,17,96)
    assert not list(pinroot.iterdir())
    assert path.stat().st_mode & 0o777 == 0o644

"""J1 public masks contracts only; no local detector/model/image inference."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def masks(monkeypatch):
    infra=Path(__file__).parents[1]/'infra'; monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('joint_masks_test',infra/'joint_rgb_masks.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def fixture_inputs(masks,tmp_path):
    images=[]
    for c in range(3):
        for f in range(3):
            p=tmp_path/f'clip_{c:02d}_frame_{f:03d}.png'; p.write_bytes(b'tiny-nondecoded'+bytes([c,f]))
            images.append({'file':p.name,'sha256':masks.masks.identity(p)['sha256'],'width':1024,'height':768})
    value={'schema':masks.SCHEMA,'images':images}; path=tmp_path/'manifest.json'; path.write_text(json.dumps(value))
    return path,value


def test_exact_public_rgb_manifest_order_hashes(masks,tmp_path):
    path,_=fixture_inputs(masks,tmp_path); records,receipt=masks.validate_inputs(tmp_path)
    assert len(records)==9 and receipt==masks.masks.identity(path)
    assert [(r['clip_index'],r['frame_index']) for r in records]==[(c,f) for c in range(3) for f in range(3)]


@pytest.mark.parametrize('fault',['schema','extra','camera','count','order','traversal','sha','bool','grid'])
def test_oracle_unknown_manifest_or_integrity_failure(masks,tmp_path,fault):
    path,value=fixture_inputs(masks,tmp_path)
    if fault=='schema': value['schema']='other'
    elif fault=='extra': value['camera_K']=[]
    elif fault=='camera': value['images'][0]['camera_K']=[]
    elif fault=='count': value['images'].pop()
    elif fault=='order': value['images'].reverse()
    elif fault=='traversal': value['images'][0]['file']='../clip_00_frame_000.png'
    elif fault=='sha': value['images'][0]['sha256']='a'*64
    elif fault=='bool': value['images'][0]['width']=True
    else: value['images'][0]['width']=512
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): masks.validate_inputs(tmp_path)


def test_tampered_or_symlink_rgb_fails(masks,tmp_path):
    fixture_inputs(masks,tmp_path); p=tmp_path/'clip_00_frame_000.png'; p.write_bytes(b'tampered')
    with pytest.raises(ValueError): masks.validate_inputs(tmp_path)
    p.unlink(); p.symlink_to(tmp_path/'clip_00_frame_001.png')
    with pytest.raises(ValueError): masks.validate_inputs(tmp_path)


def test_detector_queries_and_full_coverage_do_not_read_truth(masks):
    source=Path(masks.__file__).read_text(); wrapper=Path(masks.__file__).with_name('run_joint_rgb_masks.sh').read_text()
    assert masks.STAGE=='public_joint_rgb_automatic_masks'
    assert "('human','person.'),('object','bottle.')" in source
    assert "'frames':9,'records':[]" in source and "'private_truth_read':False" in source
    assert "report['input_manifest_sha256']" in source
    assert "masks.select_person(boxes,1024,768)" in source and "multimask_output=False" in source
    assert 'eval_private' not in source and 'eval_private' not in wrapper
    assert 'src=$INPUT,dst=$INPUT,readonly' in wrapper and 'src=$ROOT/outputs' not in wrapper
    assert '123s docker run' in wrapper and '--network none' in wrapper
    assert 'src=$ROOT/validation,dst=' not in wrapper
    subprocess.run(['bash','-n',str(Path(masks.__file__).with_name('run_joint_rgb_masks.sh'))],check=True)


def test_no_manual_detection_oracle_or_root_override(masks):
    with pytest.raises(SystemExit): masks.main(['--manual-mask','x'])

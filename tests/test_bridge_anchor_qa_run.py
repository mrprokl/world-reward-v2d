"""Tiny synthetic receipts/arrays only; no real media, models or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bridge_anchor_qa_runner_test', REPO/'infra/bridge_anchor_qa_run.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    path.chmod(0o400)
    return driver.binding.identity(path)


@pytest.fixture
def observation(tmp_path):
    root = tmp_path.resolve(); revision = 'a'*40
    code = root/'jobs'/('b'*40)/driver.ENTRY/'code'
    files = {name: write(root/driver.OBSERVATIONS/name, ('tiny '+name).encode()) for name in driver.FILENAMES}
    receipt = dict(schema='world_reward.bridge_rgb_anchor_observations.v1', stage='bridge_rgb_anchor_model_observations',
        status='pass', phase='complete', producer_revision=revision, script_sha256='c'*64,
        image_id=driver.binding.IMAGE, sources_and_inputs_rechecked=True, ground_truth_used=False,
        challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], network='none',
        body_model_loads=1, MoGe_model_loads=1, body_attempts=4, body_returns=4,
        native_decode_returns=4, MoGe_attempts=4, MoGe_returns=4,
        geometry_frame='opencv_x_right_y_down_z_forward',
        joint_rotation_frame='native_MHR_local_axes_in_native_global_frame_NOT_camera',
        focal_prior='RGB_hypot_640_480_800_not_source_calibration', model_parameters_modified=False)
    files['report.json'] = write(root/driver.OBSERVATIONS/'report.json', receipt)
    pins = dict(schema='world_reward.bridge_rgb_anchor_observation_pins.v1', producer_revision=revision,
        script_sha256='c'*64, files=files)
    write(code/driver.PINS, pins)
    return root, code, pins, receipt


def rewrite_receipt(fixture, receipt):
    root, code, pins, _ = fixture
    (root/driver.OBSERVATIONS/'report.json').unlink()
    pins['files']['report.json'] = write(root/driver.OBSERVATIONS/'report.json', receipt)
    (code/driver.PINS).unlink(); write(code/driver.PINS, pins)


def test_all_five_bytes_authenticated_before_original_receipt_json(observation, monkeypatch):
    root, code, pins, _ = observation; checked = []
    original_identity, original_json = driver.binding.identity, driver.binding.strict_json
    def identity(path, *args):
        result = original_identity(path, *args); checked.append(Path(path).name); return result
    def parse(raw):
        if b'bridge_rgb_anchor_observations.v1' in raw:
            assert set(pins['files']) <= set(checked)
        return original_json(raw)
    monkeypatch.setattr(driver.binding, 'identity', identity); monkeypatch.setattr(driver.binding, 'strict_json', parse)
    _, receipt, frozen = driver.observations(root, code/driver.PINS)
    assert receipt['body_returns'] == 4 and len(frozen) == 6


@pytest.mark.parametrize('failure', ['tamper', 'writable', 'public444', 'extra', 'symlink', 'hardlink'])
def test_artifact_integrity_fail_before_receipt_decode(observation, failure):
    root, code, _, _ = observation; path = root/driver.OBSERVATIONS/driver.FILENAMES[0]
    if failure == 'tamper': path.chmod(0o600); path.write_bytes(b'changed'); path.chmod(0o400)
    elif failure == 'writable': path.chmod(0o600)
    elif failure == 'public444': path.chmod(0o444)
    elif failure == 'extra': write(path.parent/'extra.json', b'extra')
    elif failure == 'symlink': path.unlink(); path.symlink_to(path.parent/driver.FILENAMES[1])
    elif failure == 'hardlink':
        import os
        os.link(path, root/'alias')
    with pytest.raises(ValueError): driver.observations(root, code/driver.PINS)


@pytest.mark.parametrize('field,value', [('status','fail'), ('body_returns',3), ('native_decode_returns',True),
    ('sources_and_inputs_rechecked',False), ('ground_truth_used',True), ('producer_revision','d'*40),
    ('script_sha256','e'*64), ('network','bridge')])
def test_original_producer_counters_and_no_gt_required(observation, field, value):
    receipt = copy.deepcopy(observation[3]); receipt[field] = value; rewrite_receipt(observation, receipt)
    with pytest.raises(ValueError, match='completed original'): driver.observations(observation[0], observation[1]/driver.PINS)


def prepared_controls(observation, monkeypatch):
    root, code, pins, receipt = observation
    old = root/'jobs'/pins['producer_revision']/'run_bridge_rgb_anchor_infer/code'
    source = dict(helpers={'infra/bridge_rgb_anchor_infer.py':dict(sha256=pins['script_sha256'])})
    proof = dict(source_binding=source, build_report_identity=dict(bytes=1,sha256='f'*64),
        kernel_report_identity=dict(bytes=2,sha256='e'*64))
    records = [dict(clip_id=c, frame_id=0, sha256=str(c)*64) for c in range(4)]
    public = {code/driver.binding.INPUT_PINS: write(code/driver.binding.INPUT_PINS, b'public-pin')}
    masks = {code/driver.MASK_PINS: write(code/driver.MASK_PINS, b'mask-pin')}
    write(old/driver.binding.INPUT_PINS, b'public-pin'); write(old/driver.MASK_PINS, b'mask-pin')
    receipt.update(source_binding=source, build_report_identity=proof['build_report_identity'], kernel_report_identity=proof['kernel_report_identity'],
        public_input_pins={str(old/path.relative_to(code)):pin for path,pin in (public|masks).items()},
        frames=[dict(clip_id=c,frame_id=0,completed=True,rgb_sha256=records[c]['sha256'],
            output=dict(file=driver.FILENAMES[c],**pins['files'][driver.FILENAMES[c]])) for c in range(4)],
        native_focal_solver=[dict(clip_id=c,original_returned=True,nearest64_valid_pixels=2) for c in range(4)],
        body_model=dict(named_metadata=dict(joint_names=[f'joint_{c}' for c in range(127)])))
    rewrite_receipt(observation,receipt); calls=[]
    def authenticate(actual_root, actual_code, entry, live=False):
        calls.append((actual_root,actual_code,entry,live)); return proof
    monkeypatch.setattr(driver.binding,'authenticate',authenticate)
    monkeypatch.setattr(driver.binding,'public_inputs',lambda *_:(records,public))
    monkeypatch.setattr(driver.binding,'load_masks',lambda *_:([{}]*4,masks))
    monkeypatch.setattr(driver.binding,'kernel_helper',lambda:SimpleNamespace(closure=lambda *args:dict(entry=args[2],helpers=args[3])))
    return root,code,old,receipt,calls


def test_genuine_old_entry_and_new_closure_without_whitelist_spoof(observation,monkeypatch):
    root,code,old,_,calls=prepared_controls(observation,monkeypatch)
    proof,_,_,names,_,actual_old=driver.controls(root,code)
    assert calls==[(root,old,'run_bridge_rgb_anchor_infer',False)] and actual_old==old
    assert proof['current_source']['entry']==driver.ENTRY and len(names)==127
    assert 'infra/triangle_ray_gate.py' in proof['current_source']['helpers']
    assert 'infra/bridge_rgb_anchor_render.py' not in proof['current_source']['helpers']


@pytest.mark.parametrize('bad',['source','public','indices','output','solver','names'])
def test_source_and_all_frame_metadata_fail_closed(observation,monkeypatch,bad):
    root,code,_,receipt,_=prepared_controls(observation,monkeypatch)
    if bad=='source':receipt['source_binding']={}
    elif bad=='public':receipt['public_input_pins']={}
    elif bad=='indices':receipt['frames'][3]['clip_id']=2
    elif bad=='output':receipt['frames'][3]['output']['sha256']='0'*64
    elif bad=='solver':receipt['native_focal_solver'][3]['original_returned']=False
    elif bad=='names':receipt['body_model']['named_metadata']['joint_names'][3]='joint_2'
    rewrite_receipt(observation,receipt)
    with pytest.raises(ValueError):driver.controls(root,code)


def test_only_public_prediction_proof_mounts(observation,monkeypatch):
    root,code,old,_,_=prepared_controls(observation,monkeypatch)
    paths=(root/'build-proof',root/'kernel-proof')
    for path in paths:write(path,b'proof')
    for name in ('inputs','automatic_masks'):(root/driver.BASE/name).mkdir()
    monkeypatch.setattr(driver.binding,'control_paths',lambda:paths)
    mounts=driver.readonly_mounts(root,code,old)
    assert str(code.parent)in mounts and str(old.parent)in mounts
    assert not any('/weights/'in p or 'eval_private'in p or 'render'in Path(p).name for p in mounts)
    assert str(root)not in mounts


@pytest.mark.parametrize('failure',[False,True])
def test_private_receipt_first_byte_and_sealed_fail_without_exception_text(tmp_path,failure):
    out=tmp_path.resolve();report=dict(status='fail',frames=[])
    def operation(persist):
        path=out/'report.json';assert stat.S_IMODE(path.stat().st_mode)==0o400
        report['frames'].append(dict(diagnostics=dict(status='fail' if failure else 'pass')));persist()
        if failure:raise ValueError('Bearer SECRET_MUST_NOT_PERSIST')
        report.update(status='pass',phase='complete')
    if failure:
        with pytest.raises(ValueError):driver.persist_run(out,report,operation)
    else:driver.persist_run(out,report,operation)
    raw=(out/'report.json').read_text();actual=json.loads(raw)
    assert 'SECRET_MUST_NOT_PERSIST'not in raw and len(actual['frames'])==1
    assert actual['status']==('fail'if failure else'pass') and stat.S_IMODE((out/'report.json').stat().st_mode)==0o400
    with pytest.raises(FileExistsError):driver.persist_run(out,{},operation)


def test_four_record_diagnostics_persist_before_scientific_fail(tmp_path,monkeypatch):
    root=tmp_path.resolve();code=root/'code';proof={'same':'source'}
    mask=np.ones((480,640),np.uint8)*255
    masks=[dict(person=dict(mask_file='tiny.png',mask_pixels=mask.size)) for _ in range(4)]
    monkeypatch.setattr(driver,'controls',lambda *_:(proof,[],masks,['name']*127,{},code))
    monkeypatch.setattr(driver.binding,'recheck',lambda *_:None)
    class Picture:
        format='PNG';mode='L';size=(640,480)
        def __enter__(self):return self
        def __exit__(self,*_):return False
        def __array__(self,dtype=None,copy=None):return mask.astype(dtype)if dtype else mask
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda *_:Picture())
    class Prediction(dict):
        @property
        def files(self):return list(self)
        def __enter__(self):return self
        def __exit__(self,*_):return False
    def load(path,allow_pickle):
        assert allow_pickle is False;clip=driver.FILENAMES.index(path.name)
        p=Prediction({name:np.array(0.)for name in driver.ARRAYS})
        p.update(clip_id=np.array(clip,np.int64),frame_id=np.array(0,np.int64),person_mask=mask>0,
            vertices_camera_m=np.zeros((18439,3),np.float32),joints_camera_m=np.zeros((127,3),np.float32),
            faces=np.zeros((36874,3),np.int64),camera_K=np.eye(3,dtype=np.float64))
        return p
    monkeypatch.setattr(np,'load',load)
    called=[]
    def evaluate(*args,deadline):
        assert driver.time.monotonic()<deadline<=driver.time.monotonic()+100
        called.append(deadline);return dict(status='fail'if len(called)==2 else'pass',gate_reasons=['frozen'])
    monkeypatch.setitem(sys.modules,'bridge_anchor_geometry_qa',SimpleNamespace(__file__=str(code/'infra/bridge_anchor_geometry_qa.py'),evaluate_geometry=evaluate,BUDGET_SECONDS=100))
    report=dict(status='fail',frames=[]);snapshots=[]
    with pytest.raises(ValueError,match='no rescue'):driver.run(root,code,report,lambda:snapshots.append(copy.deepcopy(report)))
    assert len(called)==4 and max(called)-min(called)<1 and len(report['frames'])==4
    assert report['sources_and_inputs_rechecked']is True and report['all_frames_completed']is True
    assert sum(r['completed']for r in snapshots[-1]['frames'])==4


def test_wrapper_offline_cpu_bounded_cleanup_and_no_broad_model_mount():
    wrapper=(REPO/'infra/run_bridge_anchor_geometry_qa.sh').read_text()
    assert '--gpus'not in wrapper and 'nvidia-smi'not in wrapper and 'flock'not in wrapper
    assert '--user 0:0 --memory 4g --cpus 4 --read-only'in wrapper and '--network none'in wrapper
    assert '183s docker run'in wrapper and '--kill-after=2s'in wrapper and 'docker rm --force "$NAME"'in wrapper
    assert 'host --preflight'in wrapper and 'host --verify'in wrapper and '/usr/bin/python3 -I -B -'in wrapper
    assert 'env -i'in wrapper and 'src=$ROOT,dst=$ROOT'not in wrapper
    assert 'weights/'not in wrapper and 'eval_private'not in wrapper


def test_host_import_has_no_numerical_or_model_dependency():
    import ast
    tree=ast.parse((REPO/'infra/bridge_anchor_qa_run.py').read_text())
    top_imports={alias.name.split('.')[0]for node in tree.body if isinstance(node,ast.Import)for alias in node.names}
    assert not {'numpy','scipy','PIL','torch','bridge_anchor_geometry_qa'}&top_imports

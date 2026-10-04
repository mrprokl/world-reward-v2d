"""Tiny RGB arrays, opaque JPEG bytes, native API spies; no real models/network."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
SPEC=importlib.util.spec_from_file_location('mediapipe_scan_test',REPO/'infra/mediapipe_hand_scan.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)
import mediapipe_cpu_runtime_verify as rt
from world_reward.hand_scan import NativeHandResult,scan_hands


def seal(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def public(root):
    rows=[];seqs=[]
    for index in(4,39,74):
        seq=dict(subject='20200820-subject-03',sequence=f'20200820_{index:06d}',sequence_lex_index=index,camera='836212060125',frames=3)
        seqs.append(seq)
        for frame in range(3):
            name=f'sequence_{index:03d}_frame_{frame:06d}.jpg';pin=seal(root/gate.BASE/'inputs'/name,b'opaque original JPEG'+name.encode())
            rows.append(dict(file=name,**pin,width=640,height=480,sequence=seq['sequence'],sequence_lex_index=index,
                             camera=seq['camera'],frame_position=frame,source_frame_id=frame))
    manifest=dict(schema='world-reward-dexycb-hand-rgb-v1',subject='20200820-subject-03',sequences=seqs,images=rows,
      source_archive={'bytes':1,'sha256':'a'*64},license='CC-BY-NC-4.0',timestamps_available=False,
      training_overlap_verified=False,challenge_overlap_verified=False)
    pin=seal(root/gate.BASE/'inputs/manifest.json',json.dumps(manifest).encode());return manifest,pin


@pytest.mark.parametrize('fault',['','missing','extra','order','id','grid','sha','private_key'])
def test_full_public_byte_grid_and_index_gate_before_decode(tmp_path,fault):
    manifest,pin=public(tmp_path);path=tmp_path/gate.BASE/'inputs/manifest.json'
    if fault=='missing':manifest['images'].pop()
    elif fault=='extra':seal(path.parent/'private.npz',b'never decode')
    elif fault=='order':manifest['images'].reverse()
    elif fault=='id':manifest['images'][0]['source_frame_id']=3
    elif fault=='grid':manifest['images'][0]['width']=320
    elif fault=='sha':manifest['images'][0]['sha256']='0'*64
    elif fault=='private_key':manifest['images'][0]['joint_3d']='forbidden'
    if fault not in('','extra'):path.chmod(0o644);pin=seal(path,json.dumps(manifest).encode())
    if fault:
        with pytest.raises(ValueError):gate.public_inputs(rt,tmp_path,REPO,{'manifest':pin})
    else:
        found,frozen=gate.public_inputs(rt,tmp_path,REPO,{'manifest':pin})
        assert found==manifest and len(frozen)==10


def native_result(count=1):
    points=[types.SimpleNamespace(x=.2,y=.3,z=.4)for _ in range(21)]
    cats=[types.SimpleNamespace(category_name='Left',display_name='',index=0,score=.9)]
    return types.SimpleNamespace(hand_landmarks=[points]*count,hand_world_landmarks=[points]*count,handedness=[cats]*count)


def detector(results):
    calls=[]
    class Detector:
        def detect(self,image):calls.append(image);return results[len(calls)-1]
    mp=types.SimpleNamespace(Image=lambda **kw:kw,ImageFormat=types.SimpleNamespace(SRGB='SRGB'))
    return Detector(),mp,calls


def test_native_callback_retains_all_instances_scores_world_and_support_without_ids(tmp_path):
    raw=native_result();raw.hand_landmarks[0][3].x=float('nan')
    det,mp,calls=detector([raw,native_result(0),native_result(4)])
    callback=gate.native_callback(det,mp,np,NativeHandResult)
    images=((i,np.zeros((480,640,3),np.uint8))for i in range(3))
    result=scan_hands(images,callback,total_frames=3,image_size=(480,640),method='native fixture',source_refs={'sha':'a'*64},budget_seconds=1)
    stats=gate.serialize(result,tmp_path/'evidence.npz',np)
    with np.load(tmp_path/'evidence.npz',allow_pickle=False)as data:
        assert set(data)=={'frame_index','frame_offsets','capacity_saturation','normalized_xyz','native_world_xyz','pixel_xy','image_z',
                          'xy_supported','image_z_supported','world_supported','handedness_scores'}
        assert data['frame_offsets'].tolist()==[0,1,1,5]and data['frame_index'].tolist()==[0,1,2]
        assert data['capacity_saturation'].tolist()==[False,False,True]
        assert np.isnan(data['normalized_xyz'][0,3,0])and not data['xy_supported'][0,3]
        assert data['pixel_xy'][0,0].tolist()==[128.,144.]and data['native_world_xyz'][0,0].tolist()==[.2,.3,.4]
        assert data['handedness_scores'].tolist()==[.9]*5 and all(not data[n].dtype.hasobject for n in data)
    assert stats['frames']==3 and stats['frames_with_proposals']==2 and len(calls)==3
    assert callback.categories==[[{'category_name':'Left','display_name':'','index':0}],[],[{'category_name':'Left','display_name':'','index':0}]*4]


@pytest.mark.parametrize('fault',['slots','capacity','landmarks','classification','label','score','error'])
def test_native_errors_propagate_not_empty_or_retried(fault):
    result=native_result()
    if fault=='slots':result.hand_world_landmarks=[]
    elif fault=='capacity':result=native_result(5)
    elif fault=='landmarks':result.hand_landmarks[0]=result.hand_landmarks[0][:-1]
    elif fault=='classification':result.handedness[0]*=2
    elif fault=='label':result.handedness[0][0].category_name='actor42'
    elif fault=='score':result.handedness[0][0].score=float('nan')
    det,mp,calls=detector([result])
    if fault=='error':det.detect=lambda *_:(_ for _ in()).throw(RuntimeError('native failed'))
    callback=gate.native_callback(det,mp,np,NativeHandResult)
    with pytest.raises((ValueError,RuntimeError)):callback(0,np.zeros((2,2,3),np.uint8))
    assert len(calls)<=1


@pytest.fixture
def host(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS:seal(code/name,(REPO/name).read_bytes()if(REPO/name).exists()else b'future pinned data-free fixture')
    seal(code.parent/'revision',(revision+'\n').encode());seal(code.parent/'source-sha256',b'b'*64+b'\n')
    for p in reversed(list(code.rglob('*'))):
        if p.is_dir():p.chmod(0o555)
    code.chmod(0o555);(root/'results').mkdir();manifest,pin=public(root)
    image=dict(Id='sha256:'+'c'*64);task=root/'task';task_pin=seal(task,b'opaque no graph')
    proof=dict(manifest=manifest,public={},task=task,task_pin=task_pin,image=image,versions={'mediapipe':'0.10.21'},
               acquisition={'manifest':pin},runtime={})
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]));monkeypatch.setattr(gate,'runtime',lambda _:rt)
    monkeypatch.setattr(gate,'authenticate',lambda *_:proof)
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    return root,code,revision,proof


def controls(host,monkeypatch,fault=''):
    root,code,rev,proof=host;calls=[];source=rt.source(root,code,rev,gate.ENTRY,gate.HELPERS)
    monkeypatch.setattr(rt,'control',lambda args,*a:b'')
    monkeypatch.setattr(rt,'cleanup',lambda *a:(_ for _ in()).throw(ValueError('cleanup'))if fault=='cleanup'else None)
    def run(args,**kwargs):
        calls.append(args);out=root/'results'/('mediapipe-hand-scan-'+rev)/'predictions'
        report=dict(stage='external_dexycb_full_t_mediapipe_hand_scan',status='pass',phase='complete',native_graphs=1,native_calls=9,
          outputs=[{}]*3,producer_revision=rev,source_binding=source,gpu_used=False,private_values_read=False,accuracy_verified=False,
          availability_is_visibility=False,handedness_is_detection_confidence=False,actor_identity_inferred=False)
        if fault=='incomplete':report['native_calls']=8
        seal(out/'report.json',json.dumps(report).encode())
        for i in(4,39,74):
            seal(out/f'sequence_{i:03d}.npz',b'opaque numeric archive');seal(out/f'sequence_{i:03d}.npz.json',b'{"native_categories":[]}')
        if fault=='extra':seal(out/'unexpected.npz',b'no')
        if fault=='source':
            p=code/gate.HELPERS[0];p.chmod(0o644);seal(p,b'changed')
        if fault=='timeout':raise subprocess.TimeoutExpired(args,600)
        if fault=='interrupt':gate.signal.getsignal(gate.signal.SIGTERM)(None,None)
        return subprocess.CompletedProcess(args,1 if fault=='native'else 0)
    monkeypatch.setattr(gate.subprocess,'run',run);return calls


def test_host_cpu_only_exact_mounts_and_postseal(host,monkeypatch):
    calls=controls(host,monkeypatch);root,code,rev,_=host;report=gate.run(root,code,rev)
    args=calls[0];assert '--gpus'not in args and '--cap-drop'in args and 'CUDA_VISIBLE_DEVICES=-1'in args
    assert args[args.index('--network')+1]=='none'and args[args.index('--cpus')+1]=='4'and args[args.index('--memory')+1]=='8g'
    mounts=[args[i+1]for i,v in enumerate(args)if v=='--mount']
    assert len(mounts)==5 and sum(not m.endswith(',readonly')for m in mounts)==1
    assert f'type=bind,src={code.parent},dst={code.parent},readonly'in mounts
    assert {p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'}
    assert all((code.parent/n).stat().st_mode&0o222==0 for n in('revision','source-sha256'))
    assert all('eval_private'not in m and 'acquire'not in m for m in mounts)
    assert report['source_rehashed_after']and report['owned_cleanup_verified']and report['accuracy_verified']is False


@pytest.mark.parametrize('fault',['native','incomplete','extra','source','cleanup','timeout','interrupt'])
def test_host_failure_is_sealed_and_partial_never_reused(host,monkeypatch,fault):
    calls=controls(host,monkeypatch,fault);root,code,rev,_=host
    with pytest.raises(ValueError):gate.run(root,code,rev)
    report=json.loads((root/'results'/('mediapipe-hand-scan-'+rev)/'report.json').read_bytes())
    assert report['status']=='fail'and report['accuracy_verified']is False and len(calls)==1
    assert all('/eval_private'not in arg for arg in calls[0]if arg.startswith('type=bind'))


def test_missing_future_pins_and_existing_output_fail_without_native(host,monkeypatch):
    root,code,rev,_=host;calls=controls(host,monkeypatch)
    monkeypatch.setattr(gate,'authenticate',lambda *_:(_ for _ in()).throw(FileNotFoundError('future actual pins required')))
    with pytest.raises(FileNotFoundError):gate.run(root,code,rev)
    assert calls==[]


def test_data_free_bootstrap_and_actual_static_closure(monkeypatch):
    source=(REPO/'infra/mediapipe_hand_scan.py').read_text().split('def run_native')[0]
    assert 'import numpy'not in source and 'detect(mp.Image'in source
    subprocess.run(['rtk','proxy','bash','-n',str(REPO/'infra/run_mediapipe_hand_scan.sh')],check=True)
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')
           if p.is_file()and '__pycache__'not in p.parts};files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    closure=set(azure_job.runtime_bundle_paths(files,'infra/run_mediapipe_hand_scan.sh'))
    assert set(gate.HELPERS)-{gate.RUNTIME_PINS,gate.ACQUIRE_PINS}<=closure
    assert not any(Path(n).name in('dexycb_hand_evaluate.py','hand_synthetic_render.py','hand_gauge_evaluate.py')for n in closure)


def test_native_full_three_scans_reuse_one_graph_once_per_original_frame(host,monkeypatch,tmp_path):
    root,code,revision,evidence=host;venv=tmp_path/'venv';venv.mkdir();(venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    monkeypatch.setattr(rt,'VENV',venv);monkeypatch.setattr(gate.sys,'prefix',str(venv));monkeypatch.setattr(gate.sys,'base_prefix','different')
    monkeypatch.setenv('WR_CODE_REVISION',revision);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1');monkeypatch.setenv('JAX_PLATFORMS','cpu')
    original_iterdir=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')])if p==Path('/sys/class/net')else original_iterdir(p))
    original_stat=Path.stat
    def owned_stat(p,*a,**kw):
        value=original_stat(p,*a,**kw)
        if p==out:return types.SimpleNamespace(st_uid=0,st_mode=value.st_mode)
        return value
    monkeypatch.setattr(Path,'stat',owned_stat)
    from importlib import metadata
    monkeypatch.setattr(metadata,'version',lambda _:'0.10.21')
    monkeypatch.setattr(np,'__file__',str(venv/'lib/numpy/__init__.py'))
    out=root/'results'/('mediapipe-hand-scan-'+revision)/'predictions';out.mkdir(mode=0o700,parents=True)
    proof=dict(source_binding=rt.source(root,code,revision,gate.ENTRY,gate.HELPERS),manifest=evidence['acquisition']['manifest'],
               task=evidence['task_pin'],versions=evidence['versions'],image_id=evidence['image']['Id'],remaining_seconds=10.)
    proof_path=root/'proof.json';proof_pin=seal(proof_path,json.dumps(proof).encode())
    original_identity=rt.identity
    monkeypatch.setattr(rt,'identity',lambda p,*a,**kw:evidence['task_pin']if str(p)=='/opt/mediapipe-task/hand_landmarker.task'else original_identity(p,*a,**kw))
    calls=[];options=[]
    class Options:
        def __init__(self,**kw):self.__dict__.update(kw)
    class Base(Options):Delegate=types.SimpleNamespace(CPU='CPU')
    class Detector:
        def __enter__(self):calls.append('opened');return self
        def __exit__(self,*_):calls.append('closed')
        def detect(self,image):calls.append('detect');return native_result(0 if calls.count('detect')%3==0 else 1)
    class Landmarker:
        @staticmethod
        def create_from_options(option):options.append(option);return Detector()
    class RGB:
        mode='RGB';size=(640,480)
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def __array__(self,dtype=None,copy=None):return np.zeros((480,640,3),dtype=dtype)
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda _:RGB())
    mp=types.ModuleType('mediapipe');mp.__file__=str(venv/'lib/mediapipe/__init__.py')
    mp.Image=lambda **kw:kw;mp.ImageFormat=types.SimpleNamespace(SRGB='SRGB')
    tasks=types.ModuleType('mediapipe.tasks');python=types.ModuleType('mediapipe.tasks.python');python.BaseOptions=Base
    vision=types.ModuleType('mediapipe.tasks.python.vision');vision.HandLandmarker=Landmarker;vision.HandLandmarkerOptions=Options
    vision.RunningMode=types.SimpleNamespace(IMAGE='IMAGE')
    for name,module in[('mediapipe',mp),('mediapipe.tasks',tasks),('mediapipe.tasks.python',python),('mediapipe.tasks.python.vision',vision)]:
        monkeypatch.setitem(sys.modules,name,module)
    report=gate.run_native(rt,root,code,out,proof_path,proof_pin)
    assert len(options)==1 and options[0].num_hands==4 and options[0].running_mode=='IMAGE'
    assert options[0].base_options.delegate=='CPU'and options[0].min_hand_detection_confidence==.5
    assert calls==['opened']+['detect']*9+['closed']and report['native_calls']==9 and len(report['outputs'])==3
    assert report['accuracy_verified']is False and report['availability_is_visibility']is False
    for index in(4,39,74):
        with np.load(out/f'sequence_{index:03d}.npz',allow_pickle=False)as data:
            assert data['frame_index'].tolist()==[0,1,2]and data['frame_offsets'].tolist()==[0,1,2,2]


def test_isolated_host_import_has_no_numpy_or_native_package():
    script='import sys;sys.path[:0]=['+repr(str(REPO/'infra'))+','+repr(str(REPO/'src'))+'];import mediapipe_hand_scan as s;s.runtime(__import__("pathlib").Path('+repr(str(REPO))+'));assert "numpy" not in sys.modules and "mediapipe" not in sys.modules'
    subprocess.run([sys.executable,'-I','-B','-S','-c',script],check=True,capture_output=True)


def test_source_and_pre_hash_deadline_refuse_before_container(host,monkeypatch):
    root,code,rev,_=host;calls=controls(host,monkeypatch);clock=[0.0];original=gate.authenticate
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0])
    def slow(*args):value=original(*args);clock[0]=601.;return value
    monkeypatch.setattr(gate,'authenticate',slow)
    with pytest.raises(ValueError):gate.run(root,code,rev)
    assert calls==[]


@pytest.mark.parametrize('fault',['foreign_snapshot_entry','marker'])
def test_snapshot_markers_available_native_and_unknown_parent_files_refused(host,monkeypatch,fault):
    root,code,rev,_=host;calls=controls(host,monkeypatch)
    if fault=='foreign_snapshot_entry':seal(code.parent/'unknown-proof.json',b'Not an approved source artifact')
    else:
        (code.parent/'revision').chmod(0o644);seal(code.parent/'revision',b'changed marker')
    with pytest.raises(ValueError):gate.run(root,code,rev)
    assert calls==[]

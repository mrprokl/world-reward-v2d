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


def public(root,cohort='v1'):
    profile=gate.profile(cohort);base=profile['base'];subject=profile['subject']
    rows=[];seqs=[]
    for index in(4,39,74):
        seq=dict(subject=subject,sequence=f'20200820_{index:06d}',sequence_lex_index=index,camera='836212060125',frames=3)
        seqs.append(seq)
        for frame in range(3):
            name=f'sequence_{index:03d}_frame_{frame:06d}.jpg';pin=seal(root/base/'inputs'/name,b'opaque original JPEG'+name.encode())
            rows.append(dict(file=name,**pin,width=640,height=480,sequence=seq['sequence'],sequence_lex_index=index,
                             camera=seq['camera'],frame_position=frame,source_frame_id=frame))
    manifest=dict(schema='world-reward-dexycb-hand-rgb-v1',subject=subject,sequences=seqs,images=rows,
      source_archive={'bytes':1,'sha256':'a'*64},license='CC-BY-NC-4.0',timestamps_available=False,
      training_overlap_verified=False,challenge_overlap_verified=False)
    pin=seal(root/base/'inputs/manifest.json',json.dumps(manifest).encode());return manifest,pin


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


def controls(host,monkeypatch,fault='',cohort='v1'):
    root,code,rev,proof=host;calls=[];source=rt.source(root,code,rev,gate.ENTRY,gate.source_helpers(cohort))
    monkeypatch.setattr(rt,'control',lambda args,*a:b'')
    monkeypatch.setattr(rt,'cleanup',lambda *a:(_ for _ in()).throw(ValueError('cleanup'))if fault=='cleanup'else None)
    def run(args,**kwargs):
        calls.append(args);out=gate.control_path(root,rev,cohort)/'predictions'
        report=dict(stage='external_dexycb_full_t_mediapipe_hand_scan',status='pass',phase='complete',native_graphs=1,native_calls=9,
          outputs=[{}]*3,producer_revision=rev,source_binding=source,gpu_used=False,private_values_read=False,accuracy_verified=False,
          availability_is_visibility=False,handedness_is_detection_confidence=False,actor_identity_inferred=False)
        if cohort!='v1':report.update(cohort=cohort,public_base=gate.profile(cohort)['base'],protocol_identity=proof['acquisition']['protocol'])
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


@pytest.mark.parametrize('cohort',['v1','v2','v3'])
def test_native_full_three_scans_reuse_one_graph_once_per_original_frame(host,monkeypatch,tmp_path,cohort):
    if cohort!='v1':
        host=fresh_host(host,monkeypatch,cohort);root,code,_,evidence=host;profile=gate.profile(cohort)
        path=code/profile['protocol'];path.chmod(0o644)
        evidence['acquisition']['protocol']=seal(path,json.dumps(dict(base=profile['base'],subject=profile['subject'],all_original_frames=True)).encode())
        path=code/profile['acquire'];path.chmod(0o644);config=json.loads(path.read_bytes())
        config['protocol']=evidence['acquisition']['protocol'];seal(path,json.dumps(config).encode())
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
    out=gate.control_path(root,revision,cohort)/'predictions';out.mkdir(mode=0o700,parents=True)
    proof=dict(source_binding=rt.source(root,code,revision,gate.ENTRY,gate.source_helpers(cohort)),manifest=evidence['acquisition']['manifest'],
               task=evidence['task_pin'],versions=evidence['versions'],image_id=evidence['image']['Id'],remaining_seconds=10.)
    if cohort!='v1':proof.update(cohort=cohort,public_base=profile['base'],protocol_identity=evidence['acquisition']['protocol'])
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
    report=gate.run_native(rt,root,code,out,proof_path,proof_pin,cohort)
    assert len(options)==1 and options[0].num_hands==4 and options[0].running_mode=='IMAGE'
    assert options[0].base_options.delegate=='CPU'and options[0].min_hand_detection_confidence==.5
    assert calls==['opened']+['detect']*9+['closed']and report['native_calls']==9 and len(report['outputs'])==3
    assert report['accuracy_verified']is False and report['availability_is_visibility']is False
    assert report.get('cohort','v1')==cohort
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


@pytest.mark.parametrize('cohort',['v1','v2','v3'])
def test_public_profile_whitelist_and_wrong_subject_rejected(tmp_path,cohort):
    manifest,pin=public(tmp_path,cohort)
    assert gate.public_inputs(rt,tmp_path,REPO,{'manifest':pin},cohort)[0]==manifest
    path=tmp_path/gate.profile(cohort)['base']/'inputs/manifest.json';path.chmod(0o644)
    manifest['subject']=gate.profile('v2'if cohort=='v1'else'v1')['subject'];pin=seal(path,json.dumps(manifest).encode())
    with pytest.raises(ValueError):gate.public_inputs(rt,tmp_path,REPO,{'manifest':pin},cohort)


def fresh_host(host,monkeypatch,cohort='v2'):
    root,code,rev,evidence=host;profile=gate.profile(cohort);code.chmod(0o755);(code/'configs').chmod(0o755)
    protocol_pin=seal(code/profile['protocol'],b'future actual frozen protocol tinyfixture')
    manifest,pin=public(root,cohort);evidence['manifest']=manifest;evidence['acquisition']={'manifest':pin,'protocol':protocol_pin}
    seal(code/profile['acquire'],json.dumps(dict(schema='world_reward.dexycb_hand_acquire_pins.v1',producer_revision='b'*40,
        report=dict(bytes=1,sha256='c'*64),manifest=pin,helper=dict(bytes=1,sha256='d'*64),protocol=protocol_pin)).encode())
    (code/'configs').chmod(0o555);code.chmod(0o555)
    return host


@pytest.mark.parametrize('cohort',['v2','v3'])
def test_fresh_selected_namespace_source_and_native_cli_no_other_cohort_media_mount(host,monkeypatch,cohort):
    host=fresh_host(host,monkeypatch,cohort);root,code,rev,evidence=host;calls=controls(host,monkeypatch,cohort=cohort)
    report=gate.run(root,code,rev,cohort);command=calls[0]
    assert command[-2:]==['--cohort',cohort]and report['cohort']==cohort and report['public_base']=='validation/dexycb_hand_'+cohort
    assert report['source_binding']['helpers'][gate.profile(cohort)['protocol']]==evidence['acquisition']['protocol']
    assert gate.PROTOCOL not in report['source_binding']['helpers']and gate.ACQUIRE_PINS not in report['source_binding']['helpers']
    mounts=[command[i+1]for i,v in enumerate(command)if v=='--mount']
    assert any('/validation/dexycb_hand_'+cohort+'/inputs'in m for m in mounts)
    assert not any('/validation/dexycb_hand_v1'in m or 'eval_private'in m for m in mounts)
    assert len({gate.control_path(root,rev,c)for c in('v1','v2','v3')})==3
    if cohort=='v3':assert not any('/validation/dexycb_hand_v2'in m for m in mounts)


@pytest.mark.parametrize('args',[[],['--cohort','v2'],['--cohort','v3']])
def test_default_legacy_and_explicit_fresh_main_arguments(host,monkeypatch,args):
    root,code,rev,_=host;observed=[]
    monkeypatch.setattr(gate.sys,'argv',['scan.py',*args]);monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
    monkeypatch.setattr(gate,'run',lambda *a:observed.append(a))
    gate.main();assert observed==[(root,code,rev,args[-1]if args else'v1')]


@pytest.mark.parametrize('args',[['--cohort','v1'],['--cohort','v4'],['--cohort'],['--cohort','v2','--cohort','v2'],
                               ['--cohort','v2','--threshold','0.1'],['--cohort','v3','--cohort','v3']])
def test_cli_rejects_arbitrary_duplicate_or_malformed_cohort_before_runtime(monkeypatch,args):
    monkeypatch.setattr(gate.sys,'argv',['scan.py',*args])
    with pytest.raises(ValueError):gate.main()


@pytest.mark.parametrize('fault',['','protocol','producer','source','subject'])
@pytest.mark.parametrize('cohort',['v2','v3'])
def test_fresh_historical_acquisition_binding_matches_real_pure_source_contract(tmp_path,monkeypatch,fault,cohort):
    import dexycb_hand_acquire as acquirer
    root=tmp_path/'root';revision='b'*40;original=root/'jobs'/revision/acquirer.JOB/'code'
    code=root/'consumer';p=gate.profile(cohort);protocol=acquirer.expected_protocol(p['protocol'])
    for name in('infra/dexycb_hand_acquire.py','infra/run_dexycb_hand_acquire.sh',*('infra/'+n for n in acquirer.HELPER_PINS)):
        seal(original/name,(REPO/name).read_bytes())
    protocol_pin=seal(original/p['protocol'],json.dumps(protocol).encode());seal(code/p['protocol'],json.dumps(protocol).encode())
    seal(original.parent/'revision',(revision+'\n').encode());seal(original.parent/'source-sha256',b'c'*64+b'\n')
    for path in reversed(list(original.rglob('*'))):
        if path.is_dir():path.chmod(0o555)
    original.chmod(0o555)
    # Use the real producer's source_binding with manufactured bounded primary text.
    primary={};expected_primary={}
    for name in protocol['primary_sources']:
        text=b'DexYCB is licensed under by-nc/4.0'if name=='publisher.html'else b"color_{:06d}.jpg np.arange(meta['num_frames'])"
        pin=seal(root/acquirer.dex.EVIDENCE/name,text);primary[name]=pin;expected_primary[name]={**pin,'url':'https://example.invalid/procedural'}
    protocol=copy.deepcopy(protocol);protocol['primary_sources']=expected_primary
    for path in(original/p['protocol'],code/p['protocol']):
        path.chmod(0o644);protocol_pin=seal(path,json.dumps(protocol).encode())
    monkeypatch.setattr(acquirer,'expected_protocol',lambda _:protocol)
    monkeypatch.setattr(acquirer,'__file__',str(original/'infra/dexycb_hand_acquire.py'))
    for module in(acquirer.dex,acquirer.download,acquirer.lease):monkeypatch.setattr(module,'__file__',str(original/'infra'/Path(module.__file__).name))
    binding=acquirer.source_binding(root,original,revision,p['protocol'])
    manifest,manifest_pin=public(root,cohort)
    receipt=dict(stage='external_dexycb_hand_rgb_private_byte_acquisition',status='pass',phase='complete',producer_revision=revision,
        source_before=binding,source_rehashed_after=True,annotation_values_parsed=False,inference_performed=False,gpu_used=False,
        disposable_archive_removed=True,sequences=3,protocol_file=p['protocol'],acquisition_profile=p['base'],public_manifest=manifest_pin,
        frames=9,selected_sequences=manifest['sequences'],retained_files={r['file']:{k:r[k]for k in('bytes','sha256')}for r in manifest['images']})
    receipt_pin=seal(root/p['base']/'report.json',json.dumps(receipt).encode())
    config=dict(schema='world_reward.dexycb_hand_acquire_pins.v1',producer_revision=revision,report=receipt_pin,manifest=manifest_pin,
                helper=rt.identity(original/'infra/dexycb_hand_acquire.py'),protocol=protocol_pin)
    if fault=='protocol':config['protocol']={'bytes':1,'sha256':'a'*64}
    elif fault=='producer':config['producer_revision']='d'*40
    elif fault=='source':
        path=original/'infra/dexycb_hand_acquire.py';path.chmod(0o644);seal(path,b'changed source')
    elif fault=='subject':
        path=code/p['protocol'];path.chmod(0o644);protocol['subject']='wrong';config['protocol']=seal(path,json.dumps(protocol).encode())
    seal(code/p['acquire'],json.dumps(config).encode())
    reached=[];actual=gate.pins
    def stop_after_acq(*args):
        if args[2]==gate.RUNTIME_PINS:reached.append(True);raise RuntimeError('CPU qualification next; no models')
        return actual(*args)
    monkeypatch.setattr(gate,'pins',stop_after_acq)
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):gate.authenticate(rt,root,code,cohort)
        assert not reached
    else:
        with pytest.raises(RuntimeError,match='CPU qualification next'):gate.authenticate(rt,root,code,cohort)
        assert reached==[True]


def test_v3_explicit_new_profile_does_not_create_or_substitute_future_pins(tmp_path):
    p=gate.profile('v3')
    assert p==dict(base='validation/dexycb_hand_v3',subject='20200908-subject-05',
        protocol='configs/dexycb_hand_protocol_v3.json',acquire='configs/dexycb_hand_acquire_v3_pins.json',
        manifest_schema='world-reward-dexycb-hand-rgb-v1')
    helpers=gate.source_helpers('v3')
    assert p['protocol']in helpers and p['acquire']in helpers
    assert not any(gate.profile(c)['protocol']in helpers or gate.profile(c)['acquire']in helpers for c in('v1','v2'))
    with pytest.raises(FileNotFoundError):gate.authenticate(rt,tmp_path,tmp_path/'code','v3')
    assert not(tmp_path/'code').exists()
    for invalid in('v4','../v3','subject05'):
        with pytest.raises(ValueError):gate.profile(invalid)

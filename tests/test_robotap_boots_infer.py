"""Tiny public-only Boots contracts; fake tensors are not actual inference evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('robotap_boots_infer_test',REPO/'infra/robotap_boots_infer.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def identity(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444);return identity(raw)
def encode(value):return json.dumps(value,allow_nan=False).encode()
def producer():return dict(bytes=2,sha256='a'*64,producer_revision='b'*40,script_sha256='c'*64)
def pins():
    return dict(schema='world-reward-robotap-boots-inference-pins-v1',image_id=gate.IMAGE,
        protocol=dict(bytes=gate.source.PROTOCOL_BYTES,sha256=gate.source.PROTOCOL_SHA256),
        public=producer()|dict(files={n:identity(b'{}')for n in ('manifest.json','video_000.npz','video_001.npz','video_002.npz')}),
        runtime=producer()|dict(report_path='results/bootstapir-runtime-verify-'+('b'*40)+'/report.json'))


def video(index=0):
    t,h,w,n=3+index,4,6,2
    arrays=dict(video=np.arange(t*h*w*3,dtype=np.uint8).reshape(t,h,w,3),query_points=np.array([[0,1,2],[t-1,3,5]],np.float64),point_indices=np.array([0,2],np.int64))
    row=dict(file=f'video_{index:03d}.npz',source_pickle=f'eval_private/pickles/robotap/robotap_split{index}.pkl',video_key=f'lexicographic_{index}',frames=t,height=h,width=w,
        point_indices=[0,2],unavailable_original_indices=[1],query_count=n,all_original_frames_retained=True)
    return arrays,row


def public(directory):
    records=[]
    for index in range(3):
        arrays,row=video(index);path=directory/row['file'];path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as stream:np.savez(stream,**arrays)
        path.chmod(0o444);row.update(identity(path.read_bytes()));records.append(row)
    manifest=dict(schema='world-reward-robotap-boots-public-v1',videos=records,selection=[dict(pickle_file=r['source_pickle'],video_key=r['video_key'])for r in records],
        initial_query_is_external_oracle=True,query_format='t,y,x; normalized_xy multiplied by original W,H; no half-pixel offset',
        future_tracks_or_visibility_public=False,frame_crop_or_resize=False,training_overlap_verified=False,challenge_overlap_verified=False,full_hoi_accuracy_verified=False,
        tap_query_source=dict(url='immutable primary metrics',bytes=24538,sha256='d'*64),public_namespace='public_v2',
        serialization_decoder=dict(source_sha256='279aa5b1c1cf1d5b2e2025f76c8594df6312fdc65e9431636448926271eccca2',source_bytes=73425,allowed_global='mediapy._VideoArray',mediapy_package_imported=False))
    save(directory/'manifest.json',encode(manifest));return manifest


@pytest.fixture
def bound(tmp_path,monkeypatch):
    root=tmp_path/'runtime';rev='e'*40;code=root/'jobs'/rev/gate.JOB/'code';code.mkdir(parents=True)
    save(code/'infra/robotap_boots_infer.py',(REPO/'infra/robotap_boots_infer.py').read_bytes());save(code/'infra/robotap_boots_acquire.py',b'procedural source not executed')
    save(code.parent/'revision',(rev+'\n').encode());save(code.parent/'source-sha256',('f'*64+'\n').encode())
    protocol=json.loads((REPO/gate.PROTOCOL).read_bytes())
    for name,row in protocol['source']['files'].items():row.update(save(root/gate.BASE/'assets/tapnet_source'/name,('procedural pinned native '+name).encode()))
    protocol['checkpoint'].update(save(root/gate.BASE/'assets'/protocol['checkpoint']['file'],b'procedural checkpoint never loaded'))
    save(code/gate.PROTOCOL,(REPO/gate.PROTOCOL).read_bytes())
    old=dict(stage=gate.STAGE,status='fail',phase='preflight',producer_revision=gate.PREVIOUS_REVISION,error_type='PermissionError',native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,videos=[],private_pickles_read=False,future_tracks_or_visibility_read=False,evaluation_performed=False,challenge_inputs_used=False)
    old_identity=save(root/gate.BASE/'infer_v1/report.json',encode(old));monkeypatch.setattr(gate,'PREVIOUS_FAILURE',old_identity)
    manifest=public(root/gate.PUBLIC/'inputs');p=pins();p['public']['files']={name:gate.source.identity(root/gate.PUBLIC/'inputs'/name)for name in p['public']['files']}
    before=dict(files={'infra/robotap_boots_public.py':dict(bytes=9,sha256=p['public']['script_sha256'])},markers={})
    report=dict(stage='external_robotap_oracle_initial_query_public_adapter',status='pass',producer_revision=p['public']['producer_revision'],source_before=before,source_after=before,
        public_files=p['public']['files'],actual_full_frame_matrix=[r['frames']for r in manifest['videos']],source_after_reverified=True,originals_after_reverified=True,initial_queries_are_external_oracles=True,
        future_labels_available_to_inference=False,inference_performed=False,evaluation_performed=False,gpu_used=False,challenge_inputs_used=False)
    p['public'].update(save(root/gate.PUBLIC/'report.json',encode(report)))
    cpu=dict(versions={'dm-tree':'0.1.10','einshape':'1.0','absl-py':'2.5.0','attrs':'26.1.0','wrapt':'1.17.3'},python='3.11',torch='2.5.1+cu124',numpy='1.26.3',
        tree_cpu_verified=True,source_native_einshape_cpu_verified=True,native_bilinear_cpu_verified=True,native_tapir_modules_imported=True,models_instantiated=False,cuda_initialized=False)
    runtime=dict(stage='bootstapir_runtime_verify',status='pass',phase='complete',producer_revision=p['runtime']['producer_revision'],child_image_id=gate.IMAGE,native_source_revision=protocol['source']['revision'],
        original_build_status='fail',original_build_failure_preserved=True,source_rehashed_after=True,gpu_execution=False,checkpoint_read=False,rgb_or_labels_read=False,cpu_import=cpu,
        source_helpers={'infra/run_bootstapir_runtime_verify.sh':dict(bytes=2,sha256=p['runtime']['script_sha256'])},native_sources={n:{k:r[k]for k in ('bytes','sha256')}for n,r in protocol['source']['files'].items()})
    p['runtime'].update(save(root/p['runtime']['report_path'],encode(runtime)));save(code/gate.PINS,encode(p))
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/'infra/robotap_boots_infer.py'))
    monkeypatch.setattr(gate.source,'__file__',str(code/'infra/robotap_boots_acquire.py'))
    # Only native text/checkpoint hashes are procedural; no unverified source is executed.
    monkeypatch.setattr(gate.source,'read_protocol',lambda path:protocol)
    return dict(root=root,code=code,revision=rev,pins=p,report=report,runtime=runtime,manifest=manifest)


def test_full_producer_shaped_pins_and_no_private_io(bound,monkeypatch):
    original=Path.open
    def protected(path,*args,**kwargs):
        assert '/eval_private/' not in str(path) and not str(path).endswith('.pkl');return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',protected)
    value=gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])
    assert len(value['public_files'])==4 and len(value['native_sources'])==7
    assert gate.preflight(bound['root'],bound['code'],bound['revision'])[1]==value
    assert value['source_files']['infra/robotap_boots_infer.py']==gate.source.identity(Path(gate.__file__))


@pytest.mark.parametrize('field,value',[('status','fail'),('phase','partial'),('original_build_failure_preserved',False),('checkpoint_read',True),('rgb_or_labels_read',True),('child_image_id','wrong')])
def test_cpu_proof_failclosed_even_when_resealed(bound,field,value):
    bound['runtime'][field]=value;path=bound['root']/bound['pins']['runtime']['report_path'];path.chmod(0o644);bound['pins']['runtime'].update(save(path,encode(bound['runtime'])))
    with pytest.raises(ValueError):gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])


@pytest.mark.parametrize('field,value',[('future_labels_available_to_inference',True),('status','fail'),('initial_queries_are_external_oracles',False),('actual_full_frame_matrix',[3,4,6])])
def test_public_proof_failclosed_even_when_resealed(bound,field,value):
    bound['report'][field]=value;path=bound['root']/gate.PUBLIC/'report.json';path.chmod(0o644);bound['pins']['public'].update(save(path,encode(bound['report'])))
    with pytest.raises(ValueError):gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])


def test_missing_config_and_input_tamper(bound):
    path=bound['code']/gate.PINS;path.parent.chmod(0o755);path.unlink()
    with pytest.raises(FileNotFoundError):gate.preflight(bound['root'],bound['code'],bound['revision'])
    path=bound['root']/gate.PUBLIC/'inputs/video_001.npz';path.chmod(0o644);path.write_bytes(b'changed');path.chmod(0o444)
    with pytest.raises(ValueError):gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])


@pytest.mark.parametrize('change',[lambda p:p.update(image_id='sha256:'+'f'*64),lambda p:p['public']['files'].pop('video_002.npz'),lambda p:p['runtime'].update(report_path='results/other/report.json'),lambda p:p['public'].update(bytes=True),lambda p:p['protocol'].update(sha256='f'*64)])
def test_exact_pins(change):
    p=pins();change(p)
    with pytest.raises(ValueError):gate.validate_pins(p)


def test_duplicate_nonfinite_json_rejected():
    for text in ('{"a":1,"a":2}','{"a":NaN}'):
        with pytest.raises(ValueError):gate.strict_json(text)


@pytest.mark.parametrize('change',[lambda a,r:a.update(video=np.ma.array(a['video'])),lambda a,r:a.update(query_points=a['query_points'].astype(np.float32)),lambda a,r:a.update(future_tracks=np.zeros(1)),lambda a,r:a['query_points'].__setitem__((0,0),.5),lambda a,r:a['query_points'].__setitem__((0,1),5),lambda a,r:r.update(unavailable_original_indices=[0])])
def test_original_video_contract(change):
    a,r=video();change(a,r)
    with pytest.raises(ValueError):gate.validate_video(a,r)


def test_three_public_original_timelines(tmp_path):
    directory=tmp_path/'inputs';m=public(directory)
    assert gate.public_records(directory)==m['videos']
    for row in m['videos']:
        with np.load(directory/row['file'],allow_pickle=False)as loaded:gate.validate_video(dict(loaded),row)
    m['frame_crop_or_resize']=True;(directory/'manifest.json').chmod(0o644);save(directory/'manifest.json',encode(m))
    with pytest.raises(ValueError):gate.public_records(directory)


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    @property
    def shape(self):return self.value.shape
    @property
    def dtype(self):return self.value.dtype
    def __getitem__(self,key):return Tensor(self.value[key])
    def __mul__(self,x):return Tensor(self.value*(x.value if isinstance(x,Tensor)else x))
    __rmul__=__mul__
    def __truediv__(self,x):return Tensor(self.value/(x.value if isinstance(x,Tensor)else x))
    def __sub__(self,x):return Tensor(self.value-(x.value if isinstance(x,Tensor)else x))
    def __rsub__(self,x):return Tensor(x-self.value)
    def __gt__(self,x):return Tensor(self.value>x)
    def all(self):return Tensor(self.value.all())
    def item(self):return self.value.item()
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class Context:
    def __enter__(self):return self
    def __exit__(self,*args):return False


def fake_native():
    calls=[]
    torch=types.SimpleNamespace(float32=np.dtype(np.float32),Tensor=Tensor,isfinite=lambda x:Tensor(np.isfinite(x.value)),
        as_tensor=lambda x,**kw:Tensor(np.asarray(x,dtype=kw['dtype'])),sigmoid=lambda x:Tensor(1/(1+np.exp(-x.value))),no_grad=Context)
    def resize(x,resolution):
        calls.append(('resize',x.shape,resolution));return Tensor(np.broadcast_to(x.value[:,:,0:1,0:1,:],(1,x.shape[1],*resolution,3)).copy())
    def convert(x,inp,out,coordinate_format='xy'):
        calls.append(('convert',tuple(inp),tuple(out),coordinate_format));return Tensor(x.value*np.array(out,np.float32)/np.array(inp,np.float32))
    utils=types.SimpleNamespace(bilinear=resize,convert_grid_coordinates=convert)
    def model(frames,queries,**kwargs):
        calls.append(('model',frames.value.copy(),queries.value.copy(),kwargs));n,t=queries.shape[1],frames.shape[1]
        coords=np.arange(n*t*2,dtype=np.float32).reshape(1,n,t,2)+.37
        return dict(tracks=Tensor(coords),occlusion=Tensor(np.full((1,n,t),-3,np.float32)),expected_dist=Tensor(np.full((1,n,t),-3,np.float32)))
    return torch,utils,model,calls


def test_native_api_math_full_t_and_actual_counters():
    a,r=video();torch,utils,model,calls=fake_native();report=dict(native_calls_attempted=0,native_calls_returned=0)
    prediction=gate.native_prediction(torch,utils,model,a,report);gate.validate_prediction(prediction,r,a['query_points'],a['point_indices'])
    invocation=next(row for row in calls if row[0]=='model')
    assert invocation[3]==dict(is_training=False,query_chunk_size=32)
    assert invocation[1].shape==(1,r['frames'],256,256,3) and invocation[1].dtype==np.float32
    assert np.array_equal(invocation[2],a['query_points'].astype(np.float32)[None]*np.array([r['frames'],256,256],np.float32)/np.array([r['frames'],r['height'],r['width']],np.float32))
    assert report==dict(native_calls_attempted=1,native_calls_returned=1)
    assert prediction['visible'].all() and prediction['static_visible'].all() and prediction['static_tracks'].dtype==np.float64


@pytest.mark.parametrize('field',('tracks','tracks_256','static_tracks','frame_index','point_indices','visible'))
def test_saved_prediction_tamper(field):
    a,r=video();torch,utils,model,_=fake_native();prediction=gate.native_prediction(torch,utils,model,a)
    if field=='visible':prediction[field]=prediction[field].astype(np.uint8)
    elif field=='tracks_256':prediction[field][0,0,0]+=1
    else:prediction[field].flat[0]+=1
    with pytest.raises(ValueError):gate.validate_prediction(prediction,r,a['query_points'],a['point_indices'])


def test_native_returned_counter_preserves_failure():
    a,_=video();torch,utils,model,_=fake_native();report=dict(native_calls_attempted=0,native_calls_returned=0)
    def failed(*args,**kwargs):raise RuntimeError('fake native failure')
    with pytest.raises(RuntimeError):gate.native_prediction(torch,utils,failed,a,report)
    assert report==dict(native_calls_attempted=1,native_calls_returned=0)


def test_source_native_checkpoint_weights_only_and_no_wrapper():
    events=[];weight=Tensor(np.ones((2,3),np.float32));state={'weight':weight}
    class Model:
        training=True
        def state_dict(self):return state
        def load_state_dict(self,value,strict):events.append(('load_state',value is state,strict))
        def float(self):events.append('float');return self
        def eval(self):self.training=False;return self
        def to(self,device):events.append(device);return self
    weight.is_floating_point=lambda:True
    torch=types.SimpleNamespace(Tensor=Tensor,float32=np.dtype(np.float32),isfinite=lambda x:Tensor(np.isfinite(x.value)))
    def load(path,**kwargs):events.append(('checkpoint',kwargs));return state
    torch.load=load
    tapir=types.SimpleNamespace(TAPIR=lambda **kw:(events.append(('constructor',kw))or Model()))
    gate.load_model(torch,tapir,Path('/procedural/never-real-checkpoint'))
    assert ('constructor',dict(pyramid_level=1))in events and ('checkpoint',dict(map_location='cpu',weights_only=True))in events and ('load_state',True,True)in events
    torch.load=lambda *a,**k:{'state_dict':state}
    with pytest.raises(ValueError):gate.load_model(torch,tapir,Path('/procedural'))


def test_wrapper_firewall_lock_and_real_static_closure():
    path=REPO/'infra/run_robotap_boots_infer.sh';text=path.read_text()
    assert subprocess.run(['bash','-n',str(path)],capture_output=True).returncode==0
    assert '[[ $# == 0 ]]'in text and 'flock --nonblock 9' in text and text.index('flock --nonblock 9')<text.index('docker run')
    assert '--network none' in text and '--gpus all' in text and '--memory 32g' in text and '915s docker run' in text
    assert 'eval_private'not in text and 'pickles'not in text and 'src=$ROOT,dst=$ROOT'not in text and 'src=$BASE,dst=$BASE'not in text
    assert '/infer_v2' in text and 'WR_RUNTIME_PROOF_MIRROR=1' in text and '--publish-runtime-proof' in text
    assert 'src=$src,dst=$dst,readonly' in text
    assert 'exec 9>&-'in text and 'lock_identity fd'in text and 'docker rm -f "$cid"'in text
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for parent in ('infra','configs','src')for p in (REPO/parent).rglob('*')if p.is_file()and p.suffix not in ('.pyc',)}
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_robotap_boots_infer.sh'))
    assert {'infra/robotap_boots_infer.py','infra/robotap_boots_acquire.py','configs/robotap_boots_protocol.json'}<=selected
    import ast
    tree=ast.parse((REPO/'infra/robotap_boots_infer.py').read_text())
    assert not any(isinstance(n,ast.Import)and any(a.name=='robotap_boots_public'for a in n.names)for n in ast.walk(tree))


def test_no_torch_or_numpy_import_on_host_preflight():
    script="import sys;sys.path.insert(0,sys.argv[1]);import robotap_boots_infer;assert 'torch'not in sys.modules and 'numpy'not in sys.modules"
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(REPO/'infra')],capture_output=True)
    assert result.returncode==0,result.stderr.decode()


@pytest.mark.parametrize('failed',(False,True))
def test_main_seals_all_three_or_exact_partial_failure(bound,monkeypatch,failed):
    out=bound['root']/gate.OUT;out.mkdir();save(out/'.container.cid',('1'*64+'\n').encode())
    monkeypatch.setenv('WR_ROOT',str(bound['root']));monkeypatch.setenv('WR_CODE',str(bound['code']));monkeypatch.setenv('WR_CODE_REVISION',bound['revision'])
    monkeypatch.setenv('WR_AZURE_VM02_VERIFIED','1');monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGE);monkeypatch.setenv('WR_RUNTIME_PROOF_MIRROR','1')
    monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(sysname='Linux'));monkeypatch.setattr(gate.os,'geteuid',lambda:1000)
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([Path('lo')])if str(path)=='/sys/class/net'else original(path))
    torch,utils,model,calls=fake_native()
    if failed:
        def model(*args,**kwargs):raise RuntimeError('explicit fake model failure')
    torch.cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda _:None,synchronize=lambda:None)
    torch.manual_seed=lambda _:None;torch.use_deterministic_algorithms=lambda *a,**k:None
    torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True,benchmark=True))
    torch.__version__='2.5.1+cu124';monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setattr(np,'__version__','1.26.3')
    monkeypatch.setattr(gate,'native_modules',lambda path:(None,utils));monkeypatch.setattr(gate,'load_model',lambda *args:model)
    if failed:
        with pytest.raises(RuntimeError):gate.main([])
    else:gate.main([])
    report=json.loads((out/'report.json').read_bytes());assert (out/'report.json').stat().st_mode&0o222==0
    assert report['all_input_assets_sources_rechecked']is True and report['private_pickles_read']is False and report['oracle_initial_queries']is True
    assert report['budget_seconds']==900 and report['inference_config']['AMP']is False
    if failed:
        assert report['status']=='fail' and report['phase']=='video_000' and report['native_calls_attempted']==1 and report['native_calls_returned']==report['native_calls_completed']==0 and report['videos']==[]
    else:
        assert report['status']=='pass' and report['phase']=='complete' and report['native_calls_attempted']==report['native_calls_returned']==report['native_calls_completed']==3
        assert [row['frames']for row in report['videos']]==[3,4,5]
        assert {p.name for p in(out/'predictions').iterdir()}=={f'video_{i:03d}.npz'for i in range(3)}


@pytest.mark.parametrize('key,value', [('source_sha256','f'*64),('allowed_global','mediapy.OtherClass'),('mediapy_package_imported',True)])
def test_public_v2_serialization_evidence_required(tmp_path,key,value):
    directory=tmp_path/'inputs';manifest=public(directory);manifest['serialization_decoder'][key]=value
    (directory/'manifest.json').chmod(0o644);save(directory/'manifest.json',encode(manifest))
    with pytest.raises(ValueError):gate.public_records(directory)


def test_public_v1_not_accepted_as_v2(tmp_path):
    directory=tmp_path/'inputs';manifest=public(directory);manifest['public_namespace']='public_v1'
    (directory/'manifest.json').chmod(0o644);save(directory/'manifest.json',encode(manifest))
    with pytest.raises(ValueError):gate.public_records(directory)


@pytest.fixture
def proof_owners(bound,monkeypatch):
    original=Path.lstat;target=gate.runtime_proof_path(bound['root'],bound['revision'])
    def metadata(path):
        result=original(path)
        if path in (target,target.parent):
            fields={name:getattr(result,name)for name in dir(result)if name.startswith('st_')};fields['st_uid']=0;return types.SimpleNamespace(**fields)
        return result
    monkeypatch.setattr(Path,'lstat',metadata);monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    return target


def test_authorized_single_receipt_mirror_original_unchanged(bound,proof_owners):
    original=bound['root']/bound['pins']['runtime']['report_path'];original.chmod(0o400);before=gate.source.identity(original)
    binding=gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])
    target=gate.publish_runtime_proof(bound['root'],bound['code'],bound['revision'],bound['pins'],binding)
    assert target==proof_owners and target.read_bytes()==original.read_bytes()
    assert target.stat().st_mode&0o777==0o444 and original.stat().st_mode&0o777==0o400
    assert gate.source.identity(original)==before and gate.verify_runtime_proof(bound['root'],bound['revision'],bound['pins'])==target
    assert {p.name for p in target.parent.iterdir()}=={'report.json'}
    with pytest.raises(FileExistsError):gate.publish_runtime_proof(bound['root'],bound['code'],bound['revision'],bound['pins'],binding)


def test_mirror_rejects_changed_or_extra_metadata(bound,proof_owners):
    binding=gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins'])
    target=gate.publish_runtime_proof(bound['root'],bound['code'],bound['revision'],bound['pins'],binding)
    save(target.parent/'other.json',b'{}')
    with pytest.raises(ValueError):gate.verify_runtime_proof(bound['root'],bound['revision'],bound['pins'])
    (target.parent/'other.json').unlink();target.chmod(0o644);target.write_bytes(b'changed');target.chmod(0o444)
    with pytest.raises(ValueError):gate.verify_runtime_proof(bound['root'],bound['revision'],bound['pins'])


def test_host_mounts_only_mirror_at_original_destination(bound,proof_owners,monkeypatch,capsys):
    before=gate.bindings(bound['root'],bound['code'],bound['revision'],bound['pins']);target=gate.publish_runtime_proof(bound['root'],bound['code'],bound['revision'],bound['pins'],before)
    monkeypatch.setenv('WR_ROOT',str(bound['root']));monkeypatch.setenv('WR_CODE',str(bound['code']));monkeypatch.setenv('WR_CODE_REVISION',bound['revision'])
    gate.host_mode('--mounts');rows=[line.split('\t')for line in capsys.readouterr().out.splitlines()]
    original=str(bound['root']/bound['pins']['runtime']['report_path'])
    assert [str(target),original]in rows and [original,original]not in rows
    assert [str(bound['root']/gate.BASE/'infer_v1/report.json')]*2 in rows
    assert all('/eval_private/' not in p and not p.endswith('.pkl')for row in rows for p in row)


def test_original_inference_permission_failure_remains_failed(bound):
    path=bound['root']/gate.BASE/'infer_v1/report.json';before=gate.source.identity(path)
    assert gate.previous_failure(bound['root'])==before
    report=json.loads(path.read_bytes());report['native_calls_completed']=1
    path.chmod(0o644);new=save(path,encode(report))
    with pytest.raises(ValueError):gate.previous_failure(bound['root'])
    gate.PREVIOUS_FAILURE=new
    try:
        with pytest.raises(ValueError):gate.previous_failure(bound['root'])
    finally:gate.PREVIOUS_FAILURE=before


def test_original_failed_inference_numerical_functions_unchanged():
    import ast
    path='infra/robotap_boots_infer.py'
    old=ast.parse(subprocess.check_output(['git','show',gate.PREVIOUS_REVISION+':'+path],cwd=REPO))
    current=ast.parse((REPO/path).read_text())
    for name in ('load_model','native_prediction','validate_prediction','validate_video','public_records','native_modules'):
        before=next(f for f in old.body if isinstance(f,ast.FunctionDef)and f.name==name)
        after=next(f for f in current.body if isinstance(f,ast.FunctionDef)and f.name==name)
        assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)

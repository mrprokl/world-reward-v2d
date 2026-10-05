"""Data-free caller/provenance controls, not a native GPU qualification."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'infra'),str(ROOT/'src')]
import joint_point_authored_qualify as q
import mediapipe_cpu_runtime_verify as rt
import joint_point_native_qualify as pair
from cari_converter import PARAMETER_DIMS


def configuration():return json.loads((ROOT/q.PROTOCOL).read_text())


def test_protocol_byte_identity():
    raw=(ROOT/q.PROTOCOL).read_bytes()
    assert q.PROTOCOL_PIN=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    assert configuration()['runtime_image_id']==q.IMAGE


def test_primary_compact_conversion_invocations_and_units():
    calls=[]
    def body(a):
        calls.append(('body',a.copy()));return np.ones((3,260),np.float32)
    def hand(a):calls.append(('hand',a.copy()));return np.ones((3,54),np.float32)*2
    values=q.parameters(np,configuration(),body,hand,PARAMETER_DIMS)
    assert [(n,a.shape,a.dtype) for n,a in calls]==[('body',(3,133),np.dtype('float32')),('hand',(3,27),np.dtype('float32'))]
    assert all(not a.any() for _,a in calls)
    assert np.array_equal(values['mhr_body_pose_cont'],np.ones((3,260),np.float32))
    assert np.array_equal(values['mhr_hand'],np.full((3,108),2,np.float32))
    assert values['mhr_scale'].shape==(3,28) and not values['mhr_scale'].any()
    assert np.array_equal(values['mhr_global_rot6d'],np.tile([1,0,0,1,0,0],(3,1)))
    assert np.array_equal(values['mhr_trans'][:,0],np.asarray([0,.015,.04],np.float32))
    assert np.array_equal(values['mhr_trans'][:,2],np.array([4,4,4],np.float32))
    assert all(v.dtype==np.float32 for v in values.values())


def test_parameter_schema_refused_before_conversion():
    seen=[]
    with pytest.raises(ValueError,match='ABI'):
        q.parameters(np,configuration(),lambda a:seen.append(a),lambda a:seen.append(a),{'mhr_scale':27})
    assert seen==[]


def test_kernel_observer_delegates_exact_result_and_no_mutation():
    events=[]
    def kernel(value):events.append(value);return value+1
    before=kernel.__code__
    result,counts=q.observed_call(lambda:kernel(4),{'real':kernel})
    assert result==5 and counts=={'real':1} and events==[4]
    assert kernel.__code__ is before and sys.getprofile() is None


def test_observer_restores_after_failure():
    def kernel():raise RuntimeError('fixture')
    with pytest.raises(RuntimeError):q.observed_call(kernel,{'real':kernel})
    assert sys.getprofile() is None


def test_existing_profiler_refused_without_replacement():
    callback=lambda *_:None
    sys.setprofile(callback)
    try:
        with pytest.raises(ValueError):q.observed_call(lambda:None,{})
        assert sys.getprofile() is callback
    finally:sys.setprofile(None)


def archive_fixture(tmp_path):
    root=tmp_path/'root';code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    entries={}
    for name in [q.BODY+'/'+n for n in ('model.ckpt','model_config.yaml','assets/mhr_model.pt','LICENSE')]+['results/weights-acquisition.json']:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());path.chmod(0o444)
        entries[name]={'type':'file',**rt.identity(path)}
    manifest={'original_root':str(root),'images_exported':False,'entries':entries};raw=json.dumps(manifest).encode()
    path=root/'results'/('frontend-asset-archive-'+'a'*40)/'archive.tar';path.parent.mkdir()
    with tarfile.open(path,'w') as tar:
        member=tarfile.TarInfo('world-reward-frontend-assets-manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
        other=tarfile.TarInfo('unread_payload');other.size=3;tar.addfile(other,io.BytesIO(b'xyz'))
    path.chmod(0o444)
    pins={'schema':'world_reward.frontend_asset_archive.pins.v1','independent_external_archive_verification_completed':True,
        'producer_revision':'a'*40,'archive_pathrelative':str(path.relative_to(root)),
        'archive_identity':rt.identity(path),'manifest_identity':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
    (code/q.ARCHIVE_PINS).write_text(json.dumps(pins));return root,code,path,pins


def test_only_independent_first_manifest_derive_exact_five_assets(tmp_path):
    root,code,path,pins=archive_fixture(tmp_path)
    selected,manifest=q.archive_model_pins(rt,root,code)
    assert len(selected)==5 and manifest==pins['manifest_identity']
    assert all(rt.identity(Path(name))==pin for name,pin in selected.items())


@pytest.mark.parametrize('kind',['model','manifest','archive_size','unverified','foreign_path'])
def test_archive_provenance_tamper_fails(tmp_path,kind):
    root,code,path,pins=archive_fixture(tmp_path)
    if kind=='model':
        p=root/q.BODY/'model.ckpt';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    elif kind=='manifest':pins['manifest_identity']['sha256']='b'*64
    elif kind=='archive_size':pins['archive_identity']['bytes']+=1
    elif kind=='unverified':pins['independent_external_archive_verification_completed']=False
    else:pins['archive_pathrelative']='results/unknown/archive.tar'
    (code/q.ARCHIVE_PINS).write_text(json.dumps(pins))
    with pytest.raises((ValueError,FileNotFoundError)):q.archive_model_pins(rt,root,code)


def result(returncode=0,stdout=b'',stderr=b''):return SimpleNamespace(returncode=returncode,stdout=stdout,stderr=stderr)


def cleanup_fixture(tmp_path,monkeypatch,responses):
    cid='c'*64;path=tmp_path/'cid';path.write_text(cid+'\n');calls=[]
    def invoke(args,**kwargs):
        calls.append((args,kwargs));return responses.pop(0)
    monkeypatch.setattr(q.subprocess,'run',invoke)
    return path,cid,calls


@pytest.mark.parametrize('prefix',['error: no such object: ','Error: No such object: ','Error: No such container: '])
def test_exact_cid_absence_split_streams(tmp_path,monkeypatch,prefix):
    path,cid,calls=cleanup_fixture(tmp_path,monkeypatch,[result(1,b'\n',(prefix+'c'*64+'\n').encode())])
    q.cleanup(rt,'owned','a'*40,path)
    assert calls[0][0]==['docker','container','inspect',cid]
    assert calls[0][1]['capture_output'] is True and path.stat().st_mode&0o777==0o400


@pytest.mark.parametrize('response',[result(1,b'',b'Cannot connect daemon'),result(0,b'[]'),result(1,b'',b'error: no such object: '+'d'.encode()*64)])
def test_daemon_failure_or_foreign_absence_not_success(tmp_path,monkeypatch,response):
    path,_,calls=cleanup_fixture(tmp_path,monkeypatch,[response])
    with pytest.raises(ValueError):q.cleanup(rt,'owned','a'*40,path)
    assert len(calls)==1


def test_exact_owned_container_removed_then_verified_absent(tmp_path,monkeypatch):
    cid='c'*64;row={'Id':cid,'Name':'/owned','Image':q.IMAGE,'Config':{'Labels':{'world_reward.authored_pair.owner':'a'*40}}}
    path,_,calls=cleanup_fixture(tmp_path,monkeypatch,[result(stdout=json.dumps([row]).encode()),result(),result(1,b'[]',('error: no such object: '+cid).encode())])
    q.cleanup(rt,'owned','a'*40,path)
    assert calls[1][0]==['docker','rm','-f',cid]


@pytest.mark.parametrize('field',['Image','Name','label'])
def test_foreign_container_never_removed(tmp_path,monkeypatch,field):
    row={'Id':'c'*64,'Name':'/owned','Image':q.IMAGE,'Config':{'Labels':{'world_reward.authored_pair.owner':'a'*40}}}
    if field=='label':row['Config']['Labels']['world_reward.authored_pair.owner']='b'*40
    else:row[field]='foreign'
    path,_,calls=cleanup_fixture(tmp_path,monkeypatch,[result(stdout=json.dumps([row]).encode())])
    with pytest.raises(ValueError):q.cleanup(rt,'owned','a'*40,path)
    assert len(calls)==1


def test_cid_alias_and_hardlink_refused(tmp_path,monkeypatch):
    p=tmp_path/'cid';p.write_text('a'*64);os.link(p,tmp_path/'alias')
    monkeypatch.setattr(q.subprocess,'run',lambda *_a,**_k:pytest.fail('No daemon calls for invalid CID'))
    with pytest.raises(ValueError):q.cleanup(rt,'n','a'*40,p)


def test_retain_no_overwrite_and_fsync_mode(tmp_path):
    path=tmp_path/'raw';q.retain(path,lambda f:f.write(b'unchanged'))
    assert rt.identity(path)['bytes']==9 and path.stat().st_mode&0o777==0o444
    with pytest.raises(FileExistsError):q.retain(path,lambda f:f.write(b'overwrite'))
    assert path.read_bytes()==b'unchanged'


def native_report():
    proof={'source_binding':{'fixture':'exact'}}
    r=dict(status='pass',phase='complete',frames=3,constructor_attempts=2,constructor_returns=2,probe_attempts=4,probe_returns=4,run_attempts=2,run_returns=2,
        actual_native_updates_total=602,actual_native_updates_per_arm=301,exact_full_result_history_parity=True,exact_initial_state_loss_gradient_parity=True,
        manufactured_not_inferred=True,tracker_executed=False,positive_weight_executed=False,quality_verified=False,adoption=False,ground_truth_used=False,challenge_inputs_used=False,
        source_inputs_assets_rehashed_after=True,source_binding=proof['source_binding'],protocol_identity=q.PROTOCOL_PIN)
    for arm in ('A_original','B_point_weight_zero'):
        r[arm]={'initial':{'probes':[{'step':s,'native_calls':{'contact':1,'render':1,'penetration':int(s==181),'kaolin_sign':int(s==181),'kaolin_distance':int(s==181)}} for s in (0,181)]}}
    return r,proof


def test_complete_actual_proof_typed():
    r,proof=native_report();q.validate_native(rt,r,proof)


@pytest.mark.parametrize('key,value',[('actual_native_updates_total',601),('constructor_returns',True),('adoption',True),('tracker_executed',True),('source_inputs_assets_rehashed_after',False)])
def test_no_runtime_claim_from_incomplete_or_wrong_scope(key,value):
    r,proof=native_report();r[key]=value
    with pytest.raises(ValueError):q.validate_native(rt,r,proof)


def test_penetration_metrics_alone_not_actual_kernel_execution():
    r,proof=native_report();r['B_point_weight_zero']['initial']['probes'][1]['native_calls']['kaolin_sign']=0
    with pytest.raises(ValueError):q.validate_native(rt,r,proof)


def test_stub_pair_sequencing_is_not_a_native_qualification():
    report={k:0 for k in ('constructor_attempts','constructor_returns','probe_attempts','probe_returns','run_attempts','run_returns')};events=[]
    pair.paired_execution(lambda arm:events.append(('construct',arm)) or arm,lambda _,step:{'step':step},
        lambda _:events.append(('run',301)) or {'same':'bytes'},lambda:None,lambda:None,lambda _: {},lambda v,_:v,report,lambda:None)
    assert report['actual_native_updates_total']==602 and events.count(('run',301))==2
    with pytest.raises((ValueError,KeyError)):q.validate_native(rt,report,{'source_binding':{}})


def test_bare_host_bootstrap_no_scientific_imports():
    code=f"import runpy,sys;runpy.run_path({str(ROOT/'infra/joint_point_authored_qualify.py')!r},run_name='audit');assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-I','-B','-S','-c',code],check=True,capture_output=True)


def test_runtime_sourceclosure_and_shell_syntax():
    import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ('infra','src/world_reward','configs') for p in (ROOT/folder).rglob('*') if p.is_file()}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    paths=azure_job.runtime_bundle_paths(files,'infra/run_joint_point_authored_qualify.sh')
    assert set(q.HELPERS)<=set(paths)
    subprocess.run(['bash','-n',str(ROOT/'infra/run_joint_point_authored_qualify.sh')],check=True)


def test_no_replay_or_positive_fit_and_frozen_native_calls():
    source=(ROOT/'infra/joint_point_authored_qualify.py').read_text();tree=ast.parse(source)
    assert 'episode_000021' not in source and 'selected_episode(' not in source
    assert 'optimizer.MHRParityPostOptimizer(source,vertices,faces,cfg,mhr_layer=layer)' in source
    assert 'canonical_mask_quantile_queries' in source and 'np.zeros((3,n),bool)' in source
    assert 'PointObjectiveConfig(1.,0.,' in source and 'checkpoint_path=None' in source
    assert 'instance.run()' in source and 'total.backward()' in source
    assert 'setattr(' not in source and 'hasattr(' not in source and 'if False' not in source
    assert "'--network','none','--read-only','--user','1000:1000','--cap-drop','ALL'" in source
    assert 'rt.control' not in source and 'rt.write' not in source
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('numpy' not in ast.unparse(n) and 'torch' not in ast.unparse(n) for n in imports)

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
    assert q.REPEAT_PROTOCOL in paths
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


def test_repeat_protocol_exact_identity_and_inherited_contracts():
    raw=(ROOT/q.REPEAT_PROTOCOL).read_bytes()
    assert q.REPEAT_PIN=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    original=configuration();selected=q.protocol(SimpleNamespace(pinned=lambda p,pin,cap:rt.strict(p.read_bytes()),require=rt.require),ROOT,'native_repeat')
    expected=copy.deepcopy(original);expected['control']='native_repeat'
    expected['human']['x_coefficients']=json.loads(raw)['human_translation']
    expected['object']['vertices_float32_m']=json.loads(raw)['object']['vertices_float32_m']
    expected['optimizer']['arms']=['A_native_first','A_native_second']
    assert selected==expected and configuration()==original
    assert q.helpers()==q.HELPERS and q.helpers('native_repeat')==(*q.HELPERS,q.REPEAT_PROTOCOL)


@pytest.mark.parametrize('kind',['base_bytes','repeat_bytes','base_reference','source','image'])
def test_repeat_bound_protocol_tamper_before_any_scientific_calls(tmp_path,kind):
    (tmp_path/'configs').mkdir();base=configuration();repeat=json.loads((ROOT/q.REPEAT_PROTOCOL).read_bytes())
    if kind=='base_reference':repeat['base_protocol']['sha256']='f'*64
    elif kind=='source':repeat['optimizer_source']['bytes']+=1
    elif kind=='image':repeat['runtime_image_id']='sha256:'+'f'*64
    for name,value in ((q.PROTOCOL,base),(q.REPEAT_PROTOCOL,repeat)):
        raw=(ROOT/name).read_bytes() if kind in ('base_bytes','repeat_bytes') else json.dumps(value).encode()
        if kind=='base_bytes' and name==q.PROTOCOL or kind=='repeat_bytes' and name==q.REPEAT_PROTOCOL:raw+=b' '
        (tmp_path/name).write_bytes(raw);(tmp_path/name).chmod(0o444)
    if kind in ('base_bytes','repeat_bytes'):gate=rt
    else:gate=SimpleNamespace(pinned=lambda p,pin,cap:rt.strict(p.read_bytes()),require=rt.require)
    with pytest.raises(ValueError):q.protocol(gate,tmp_path,'native_repeat')


def test_repeat_fresh_dyadic_scene_and_single_float32_translation_cast():
    gate=SimpleNamespace(pinned=lambda p,pin,cap:rt.strict(p.read_bytes()),require=rt.require);c=q.protocol(gate,ROOT,'native_repeat')
    values=q.parameters(np,c,lambda a:np.zeros((3,260),np.float32),lambda a:np.zeros((3,54),np.float32),PARAMETER_DIMS)
    assert np.array_equal(values['mhr_trans'][:,0],(.0125*np.arange(3)+.0075*np.arange(3)**2).astype(np.float32))
    assert np.array_equal(values['mhr_trans'][:,2],np.full(3,4,np.float32))
    v=np.asarray(c['object']['vertices_float32_m'],np.float32);f=np.asarray(c['object']['faces_int64_outward'],np.int64)
    assert np.array_equal(v*64,np.array([[-3,-2,-2],[4,-1,-1],[-1,5,-1],[1,1,5]]))
    assert all(np.dot(np.cross(v[b]-v[a],v[d]-v[a]),v[a]-v.mean(0))>0 for a,b,d in f)
    assert not np.array_equal(v,np.asarray(configuration()['object']['vertices_float32_m'],np.float32))
    assert np.array_equal(f,np.asarray(configuration()['object']['faces_int64_outward'],np.int64))


@pytest.mark.parametrize('argv',[['--control'],['--control','unknown'],['--control','native_repeat','--control','native_repeat'],['--contro','native_repeat'],['native_repeat']])
def test_repeat_strict_cli_invalid_or_duplicate(argv):
    with pytest.raises(SystemExit):q.arguments(argv)


def test_repeat_cli_default_and_native_forwarding():
    assert q.arguments([]).control is None
    args=q.arguments(['--native','16','a'*64,'--control','native_repeat'])
    assert args.control=='native_repeat' and args.native==['16','a'*64]


def test_namespaces_distinct_old_closed_directory_preserved(tmp_path):
    old=q.output_path(tmp_path,'a'*40);old.mkdir(parents=True);(old/'report.json').write_bytes(b'original sealed FAIL');(old/'report.json').chmod(0o444);old.chmod(0o555)
    new=q.output_path(tmp_path,'a'*40,'native_repeat');assert new!=old and not new.exists()
    new.mkdir();new.chmod(0o555)
    with pytest.raises(FileExistsError):new.mkdir()
    assert (old/'report.json').read_bytes()==b'original sealed FAIL'
    with pytest.raises(ValueError):q.output_path(tmp_path,'../escape','native_repeat')
    with pytest.raises(ValueError):q.output_path(tmp_path,'a'*40,'arbitrary')


def repeat_native_report():
    r,proof=native_report();proof['control']='native_repeat';r['protocol_identity']=q.REPEAT_PIN
    r.update(control='native_repeat',constructor_kind='original_both',arm_names=['A_native_first','A_native_second'],point_optimizer_factory_calls=0,point_evidence_bound=False,point_config=None)
    r['A_native_first']=r.pop('A_original');r['A_native_second']=r.pop('B_point_weight_zero');return r,proof


def test_repeat_native_proof_distinguishes_originals_from_old_ab():
    r,proof=repeat_native_report();q.validate_native(rt,r,proof,'native_repeat')
    with pytest.raises(ValueError):q.validate_native(rt,r,proof)
    original,original_proof=native_report()
    with pytest.raises(ValueError):q.validate_native(rt,original,original_proof,'native_repeat')


@pytest.mark.parametrize('key,value',[('constructor_kind','point_extension'),('point_optimizer_factory_calls',True),('point_optimizer_factory_calls',1),('point_evidence_bound',True),('point_config',{}),('control',None),('protocol_identity',q.PROTOCOL_PIN)])
def test_repeat_wrong_scope_never_certifies(key,value):
    r,proof=repeat_native_report();r[key]=value
    with pytest.raises(ValueError):q.validate_native(rt,r,proof,'native_repeat')


def test_repeat_exact_mismatch_stops_before_second_run_no_tolerance():
    report={k:0 for k in ('constructor_attempts','constructor_returns','probe_attempts','probe_returns','run_attempts','run_returns')};runs=[]
    def probe(arm,step):return {'step':step,'gradient_sha256':('a' if arm=='A_native_first' or step==0 else 'b')*64}
    with pytest.raises(ValueError,match='no tolerance'):
        pair.paired_execution(lambda arm:arm,probe,lambda arm:runs.append(arm) or {'same':'full'},lambda:None,lambda:None,lambda arm:{'same':'state'},lambda r,arm:r,report,lambda:None,arm_names=q.profile('native_repeat')[2])
    assert runs==['A_native_first'] and report['probe_returns']==4 and report['run_attempts']==report['run_returns']==1
    assert 'actual_native_updates_total' not in report


def test_precomparison_index_preserves_both_initials_four_probe_hashes_before_stop():
    report={};saved=[]
    for arm in q.profile('native_repeat')[2]:
        report['phase']=arm+'_constructor'
        q.precomparison_record(report,dict(state_sha256='a'*64,optimizer_sha256='b'*64,scheduler_sha256='c'*64),lambda:saved.append(copy.deepcopy(report)))
        for step in (0,181):
            report['phase']=arm+f'_loss_gradient_{step}'
            q.precomparison_record(report,dict(step=step,loss_sha256='d'*64,metrics_sha256='e'*64,gradients='f'*64,native_calls={'contact':1,'render':1,'penetration':int(step==181),'kaolin_sign':int(step==181),'kaolin_distance':int(step==181)}),lambda:saved.append(copy.deepcopy(report)))
    assert len(saved)==6 and len(saved[-1]['retained_precomparison'])==6
    assert len(saved[0]['retained_precomparison'])==1
    for arm in q.profile('native_repeat')[2]:
        assert report['retained_precomparison'][arm+'_constructor']['optimizer_sha256']=='b'*64
        assert report['retained_precomparison'][arm+'_loss_gradient_181']['native_calls']['kaolin_distance']==1


def test_repeat_constructor_branch_uses_originals_and_quantiles_only(tmp_path,monkeypatch):
    from dataclasses import dataclass
    from world_reward import joint_point_objective as op
    from world_reward.point_surface_queries import SurfaceQueries,MaskQueryDiagnostics
    import world_reward.point_surface_queries as queries
    import cari_full_refine as full
    @dataclass
    class Config:
        penetration_collision_proxy_path:str
        hand_surface_spec_path:str
        report_every:int
        checkpoint_path:object
    class Tensor:
        def __init__(self,array):self.array=array
        def __getitem__(self,i):return Tensor(self.array[i])
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.array
    events=[]
    class Original:
        def __init__(self,*args,**kwargs):events.append('original');self.contact_mask=np.ones((3,2))
        def _object_state(self,*args,**kwargs):return Tensor(np.tile(np.eye(3),(3,1,1))),Tensor(np.zeros((3,3))),None,None
        def _contact_loss(self):pass
    placeholder=lambda:None
    optimizer=SimpleNamespace(MHRParityPostOptConfig=Config,MHRParityPostOptimizer=Original,object_inside_human_penetration_loss=placeholder)
    monkeypatch.setitem(sys.modules,'Utils',SimpleNamespace(nvdiff_color_depth_render=placeholder))
    monkeypatch.setitem(sys.modules,'kaolin.ops.mesh',SimpleNamespace(check_sign=placeholder))
    monkeypatch.setitem(sys.modules,'kaolin.metrics.trianglemesh',SimpleNamespace(point_to_mesh_distance=placeholder))
    monkeypatch.setenv('WR_CODE',str(ROOT));monkeypatch.setattr(op,'native_point_optimizer_class',lambda *a,**k:pytest.fail('No point factory in A/A'))
    n=8;points=SurfaceQueries(np.zeros((n,3)),np.zeros((n,3)),np.zeros((n,2),np.int64),np.arange(n,dtype=np.int64),np.ones(n),np.zeros((n,3)))
    diagnostic=MaskQueryDiagnostics(np.zeros((n,2),np.int64),np.ones(n,bool),np.ones(n,bool),np.zeros(n,bool),np.zeros(n,bool),np.ones(n,bool),n,True,True,n)
    def select(*args,**kwargs):events.append('quantile');return SimpleNamespace(queries=points,diagnostics=diagnostic)
    monkeypatch.setattr(queries,'canonical_mask_quantile_queries',select)
    def execution(construct,*args,arm_names):
        assert arm_names==('A_native_first','A_native_second')
        assert all(type(construct(arm))is Original for arm in arm_names)
    monkeypatch.setattr(pair,'paired_execution',execution)
    gate=SimpleNamespace(pinned=lambda p,pin,cap:rt.strict(p.read_bytes()),require=rt.require);c=q.protocol(gate,ROOT,'native_repeat')
    source={'frames':['000000','000001','000002'],'observations':{'object_mask':np.ones((3,480,640),bool)}}
    monkeypatch.setattr(full,'fingerprint',lambda source:'a'*64)
    monkeypatch.setattr(q,'runtime',lambda code:SimpleNamespace(identity=rt.identity,require=rt.require))
    torch=SimpleNamespace(arange=lambda *a,**k:np.arange(a[0]),load=lambda *a,**k:source)
    report={};q.paired_native(np,torch,object(),optimizer,source,np.zeros((4,3),np.float32),np.zeros((4,3),np.int64),np.ones((3,480,640)),c,tmp_path,report,lambda:None,lambda:None)
    assert events==['original','quantile','original'] and report['point_optimizer_factory_calls']==0
    assert report['constructor_kind']=='original_both' and report['point_evidence_bound'] is False and report['point_config'] is None
    assert {p.name for p in tmp_path.iterdir()}=={'quantile_diagnostic.npz'}
    with np.load(tmp_path/'quantile_diagnostic.npz',allow_pickle=False) as arrays:
        assert set(arrays)==set(vars(points))|{k for k,v in vars(diagnostic).items() if isinstance(v,np.ndarray)}
        assert arrays['face_indices'].dtype==np.int64 and arrays['qualified'].dtype==bool


def test_native_cli_host_control_mismatch_before_torch_or_output(tmp_path,monkeypatch):
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(ROOT));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    gate=SimpleNamespace(pinned=lambda *a,**k:{'control':None},require=rt.require)
    monkeypatch.setattr(q,'runtime',lambda code:gate)
    monkeypatch.setattr(q,'native',lambda *a,**k:pytest.fail('No native calls on control mismatch'))
    with pytest.raises(ValueError,match='selection differs'):q.main(['--native','16','a'*64,'--control','native_repeat'])
    assert list(tmp_path.iterdir())==[]

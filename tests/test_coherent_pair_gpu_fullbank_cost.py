"""Tiny authored/mocked controls only; never import Torch or run Docker."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fullbank_test',ROOT/'infra/coherent_pair_gpu_fullbank_cost.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
g=q.module(ROOT,'generic_fullbank_test','infra/coherent_pair_gpu_probe.py')
fixture=q.module(ROOT,'fixture_fullbank_test','infra/coherent_pair_cost_probe.py')
sys.path.insert(0,str(ROOT/'src'))
from world_reward import coherent_route_scorer as core
from world_reward import coherent_pair_packed_torch as scorer
from world_reward.coherent_pair_packed_score import score_pair_marginal_packed


class Runtime:
    @staticmethod
    def require(value,message):
        if not value:raise ValueError(message)
    @staticmethod
    def strict(raw):return json.loads(raw)
    @staticmethod
    def canonical(path):return path
    @staticmethod
    def write(path,raw,mode):
        with path.open('xb') as stream:stream.write(raw)
        path.chmod(mode)
    @staticmethod
    def identity(path,limit,**_):
        raw=path.read_bytes();assert len(raw)<=limit
        return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class Tensor:
    def __init__(self,array):self.array=array
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array
    def data_ptr(self):return self.array.__array_interface__['data'][0]


def mock_device(monkeypatch,*,fault=None):
    calls=[]
    def prepare(packed,device):return SimpleNamespace(packed=packed,device=device)
    def score(prepared,theta,temperature,alpha):
        calls.append(alpha);cpu=score_pair_marginal_packed(prepared.packed,theta,temperature=temperature,alpha=alpha)
        values={n:Tensor(getattr(cpu,n).copy()) for n in (*g.FLOAT_FIELDS,*g.BOOL_FIELDS)}
        if alpha==0.:
            for x,y in (('native_scores_a','native_scores_b'),('scores_a','scores_b'),('geometry_derivatives_a','geometry_derivatives_b')):values[y]=values[x]
        result=SimpleNamespace(**values,identity=cpu.identity,distribution_fingerprint=cpu.distribution_fingerprint,
            parameter_fingerprint=cpu.parameter_fingerprint,temperature=temperature,alpha=alpha,device='cuda:0')
        if fault:fault(result,len(calls))
        return result
    monkeypatch.setattr(scorer,'prepare_marginal_packed_torch',prepare)
    monkeypatch.setattr(scorer,'score_pair_marginal_packed_torch',score)
    generic=SimpleNamespace(FLOAT_FIELDS=g.FLOAT_FIELDS,BOOL_FIELDS=g.BOOL_FIELDS,bits=g.bits,arrays=g.arrays,
        device_fingerprint=lambda _,p:core._fingerprint((p.packed.arrays,p.packed.identity)))
    torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
    return generic,torch,calls


def run_tiny(monkeypatch,*,fault=None):
    generic,t,calls=mock_device(monkeypatch,fault=fault);report={};snapshots=[]
    q.measure(np,t,generic,fixture,lambda:None,report,snapshots,object_count=3)
    return report,snapshots,calls


def receipt():
    digest='a'*64;binding={'producer_revision':'1'*40};tiny={'receipts':q.PRIOR_PINS}
    v=dict(schema=q.SCHEMA,stage='coherent_pair_gpu_fullbank_native',status='pass',phase='complete',producer_revision='1'*40,
        source_binding=binding,manifest=q.manifest(),image_id=q.IMAGE,tiny_qualification=tiny,counts=[4,3600,64],native_tokens=1500,
        candidate_rows=28800,tuple_rows=512,bridge_rows=230400,person_groups=4,object_groups=3600,
        original_frame_index=0,image_size=[720,960],rows_completed=2,source_fixtures_rehashed_after=True,
        bank_rehashed_after=True,snapshots_rehashed_after=True,source_rehashed_after=True,
        models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,fit_executed=False,optimizer_executed=False,
        quality_verified=False,adoption=False,full_fit_cost_qualified=False,decision=q.DECISION,
        packed_sha256=digest,host_tables_after=digest,device_tables_before=digest,device_tables_after=digest,bank_sha256=digest,
        full_original_observations_sha256=digest,identity_sha256=digest,route_references=5,complete_geometry_count=5,
        native_geometry_entries=5,group_geometry_entries=5,native_margin_entries=5,group_margin_entries=5,
        missing_feature_cells=6,native_supported_cells=7,native_route_supported_cells=6,group_supported_cells=4,
        fixture_prepare_seconds=1.,packed_prepare_seconds=2.,upload_seconds=3.,peak_torch_allocated=32,peak_torch_reserved=64,
        max_rss_bytes=4096,
        runtime=dict(torch='2.5.1+cu124',cuda='12.4',capability=[9,0],deterministic_algorithms=True,tf32=False,
            cublas_workspace_config=':4096:8',gpu_name='NVIDIA H100 NVL',gpu_total_bytes=99456909312,python='3.11.10',numpy='1.26.3'))
    v['rows']=[dict(alpha=alpha,gpu_seconds=4.,repeat_gpu_seconds=5.,cpu_packed_seconds=6.,repeat_bits_exact=True,
        all_cpu_packed_arrays_passed=True,native_shape=[4,2,3600],group_shape=[4,3600],geometry_vjp_shape=[4,3600,17],
        alpha_vjp_shape=[4,3600],result_sha256=digest,parameter_fingerprint=digest,identity_sha256=digest,
        distribution_fingerprint=digest,supported_pairs=4,native_supported=7) for alpha in (0.,.3125)]
    v['descriptive_projection']=q.projection(3.,3.,[4.,5.,4.,5.])
    return v,binding,tiny


def test_real_stdlib_host_import_denies_numerical_packages():
    script="""import importlib.abc,importlib.util,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','torch','transformers'):raise RuntimeError('Forbidden eager numerical import')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('entry',sys.argv[1]);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
g,rt,fixture=q.helpers(__import__('pathlib').Path(sys.argv[2]));assert q.manifest()['objects']==3600
assert not any(k.split('.')[0] in ('numpy','torch','transformers') for k in sys.modules)
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',script,str(ROOT/q.HELPERS[0]),str(ROOT)],capture_output=True,timeout=10)
    assert r.returncode==0,r.stderr.decode()


def test_frozen_source_helpers_and_native_leaf_inventory():
    for name,(size,digest) in q.REUSED.items():
        raw=(ROOT/name).read_bytes();assert (len(raw),hashlib.sha256(raw).hexdigest())==(size,digest)
    leaves=q.leaves(ROOT)
    assert len(set(leaves))==len(q.HELPERS)+2
    assert all('coherent-pair-gpu-probe-v2' not in str(p) for p in leaves)
    assert subprocess.run(['bash','-n',str(ROOT/q.HELPERS[1])],capture_output=True).returncode==0


def test_frozen_population_math_no_fit_or_projection_gate():
    m=q.manifest();assert (m['persons'],m['objects'],m['native_pairs'],m['native_tokens'])==(4,3600,64,1500)
    assert m['theta']==g.manifest()['theta'] and m['scales']==g.manifest()['scales']
    assert m['alphas']==g.manifest()['alphas'] and m['temperature']==g.manifest()['temperature']
    assert m['projection_threshold'] is None and m['projection_is_descriptive_not_bound'] is True
    assert q.projection(2.,3.,[.25,1.])==[dict(evaluations=n,projected_seconds=5.+n) for n in (1,100,1000)]


def test_real_tiny_generator_packed_cpu_mocked_device_complete(monkeypatch):
    report,snapshots,calls=run_tiny(monkeypatch)
    assert calls==[0.,0.,.3125,.3125] and len(snapshots)==1
    assert report['counts']==[4,3,64] and report['native_tokens']==1500
    assert (report['candidate_rows'],report['tuple_rows'],report['bridge_rows'])==(24,512,192)
    assert report['original_frame_index']==0 and report['image_size']==[720,960]
    assert report['phase']=='complete' and report['rows_completed']==2
    assert report['source_fixtures_rehashed_after'] and report['bank_rehashed_after'] and report['snapshots_rehashed_after']
    assert report['missing_feature_cells']>0 and report['host_tables_after']==report['packed_sha256']
    assert all(r['native_shape']==[4,2,3] and r['geometry_vjp_shape']==[4,3,17] for r in report['rows'])
    assert all(r['identity_sha256']==report['identity_sha256'] and r['distribution_fingerprint']==report['packed_sha256'] for r in report['rows'])


@pytest.mark.parametrize('fault',['gradient','repeat','identity','distribution','device','alpha','dtype','shared','repeat_shared'])
def test_full_arrays_and_metadata_corruption_closes_control(monkeypatch,fault):
    def corrupt(result,call):
        if fault=='gradient':result.geometry_derivatives_a.array.flat[0]+=1.
        elif fault=='repeat' and call==2:result.scores_a.array.flat[0]+=1.
        elif fault=='identity':result.identity={}
        elif fault=='distribution':result.distribution_fingerprint='b'*64
        elif fault=='device':result.device='cpu'
        elif fault=='alpha':result.alpha=.125
        elif fault=='dtype':result.alpha_derivatives_b.array=result.alpha_derivatives_b.array.astype(np.float32)
        elif fault=='shared' and call==1:result.scores_b=Tensor(result.scores_b.array.copy())
        elif fault=='repeat_shared' and call==2:result.scores_b=Tensor(result.scores_b.array.copy())
    with pytest.raises((ValueError,AssertionError)):run_tiny(monkeypatch,fault=corrupt)


def test_mutated_device_table_snapshot_is_not_pass(monkeypatch):
    generic,t,_=mock_device(monkeypatch);calls=[]
    def fingerprint(*_):
        calls.append(1);return ('a' if len(calls)==1 else 'b')*64
    generic.device_fingerprint=fingerprint
    with pytest.raises(ValueError,match='tables mutated'):q.measure(np,t,generic,fixture,lambda:None,{},[],object_count=2)


def test_interruption_before_any_score_no_false_complete(monkeypatch):
    generic,t,calls=mock_device(monkeypatch);report={};ticks=[]
    def check():
        ticks.append(1)
        if len(ticks)==2:raise TimeoutError('Manufactured interruption')
    with pytest.raises(TimeoutError):q.measure(np,t,generic,fixture,check,report,[],object_count=2)
    assert not calls and report.get('phase')!='complete' and report.get('source_fixtures_rehashed_after') is True
    assert report['bank_rehashed_after'] is True


def test_failed_upload_rehashes_bank_and_host_packed(monkeypatch):
    generic,t,calls=mock_device(monkeypatch);report={};snapshots=[]
    def failed_upload(packed,device):raise RuntimeError('Authored upload failure')
    monkeypatch.setattr(scorer,'prepare_marginal_packed_torch',failed_upload)
    with pytest.raises(RuntimeError,match='upload failure'):
        q.measure(np,t,generic,fixture,lambda:None,report,snapshots,object_count=2)
    assert not calls and len(snapshots)==1 and set(snapshots[0])=={'bank','bank_sha256','cache','packed','packed_sha256'}
    assert report['bank_rehashed_after'] and report['snapshots_rehashed_after'] and report['source_fixtures_rehashed_after']
    assert report['phase']!='complete'


def test_genuine_complete_receipt():
    v,binding,tiny=receipt();q.validate_native(Runtime,v,binding,'1'*40,tiny)


@pytest.mark.parametrize('path,value',[
    (('counts',),[4,3599,64]),(('rows_completed',),1),(('fit_executed',),True),(('source_rehashed_after',),False),
    (('runtime','torch'),'2.1.2'),(('runtime','cuda'),'11.8'),(('runtime','capability'),[8,0]),
    (('runtime','tf32'),True),(('runtime','deterministic_algorithms'),False),(('runtime','cublas_workspace_config'),''),
    (('runtime','gpu_total_bytes'),q.MEMORY),(('runtime','python'),'3.10.1'),
    (('rows',0,'native_shape'),[4,2,32]),(('rows',1,'geometry_vjp_shape'),[4,3600,16]),
    (('rows',1,'all_cpu_packed_arrays_passed'),False),(('rows',0,'identity_sha256'),'b'*64),
    (('rows',1,'distribution_fingerprint'),'b'*64),(('rows',0,'native_supported'),8),
    (('rows',0,'gpu_seconds'),float('nan')),(('peak_torch_reserved',),q.MEMORY+1),
    (('host_tables_after',),'b'*64),(('descriptive_projection',),[]),
    (('native_route_supported_cells',),8),(('missing_feature_cells',),-1),
    (('route_references',),0),(('native_geometry_entries',),0),(('group_supported_cells',),0),(('max_rss_bytes',),0),
])
def test_tampered_full_receipt_rejected(path,value):
    v,binding,tiny=receipt();target=v
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises(ValueError):q.validate_native(Runtime,v,binding,'1'*40,tiny)


def host_mock(monkeypatch,tmp_path,*,exit_code=0,cleanup=True,post=True):
    monkeypatch.setattr(q,'ROOT',tmp_path);(tmp_path/'results').mkdir();code=ROOT
    v,binding,tiny=receipt();prior=dict(source_binding=binding,manifest=q.manifest(),image={'Id':q.IMAGE},tiny_qualification=tiny)
    calls=[];checks=[]
    def proof(*_):
        checks.append(1)
        return prior if len(checks)==1 or post else dict(prior,image={})
    def run(argv,**kwargs):
        calls.append(argv);out=tmp_path/q.RESULT
        (out/'container.cid').write_bytes(b'a'*64)
        Runtime.write(out/'native.json',(json.dumps(v)+'\n').encode(),0o400)
        return SimpleNamespace(returncode=exit_code,stdout=b'',stderr=b'')
    generic=SimpleNamespace(control=lambda *_:b'',cleanup=lambda *_:cleanup,error_family=g.error_family,publish=g.publish)
    monkeypatch.setattr(q,'helpers',lambda *_:(generic,Runtime,None));monkeypatch.setattr(q,'proof',proof)
    monkeypatch.setattr(q.subprocess,'run',run);monkeypatch.setattr(q.os,'chown',lambda *_:None)
    monkeypatch.setattr(q.os,'getuid',lambda:1000)
    monkeypatch.setattr(q.signal,'signal',lambda *_:None);monkeypatch.setattr(q.signal,'setitimer',lambda *_:None)
    # lstat identity remains real; synthetic ownership check only avoids requiring local UID1000.
    actual_stat=Path.stat
    def mocked_stat(path,*args,**kw):
        s=actual_stat(path,*args,**kw)
        if path==tmp_path/q.RESULT:
            a=list(s);a[4]=1000;return os.stat_result(a)
        return s
    monkeypatch.setattr(Path,'stat',mocked_stat)
    return code,calls


def test_mock_real_host_umask077_cid_before_native_narrow_mounts(monkeypatch,tmp_path):
    code,calls=host_mock(monkeypatch,tmp_path);old=os.umask(0o077)
    try:value=q.host(code,'1'*40)
    finally:os.umask(old)
    assert value['status']=='pass' and value['owned_container_removed'] and value['source_rehashed_after']
    out=tmp_path/q.RESULT
    assert (out.stat().st_mode&0o777)==0o500 and (out/'report.json').stat().st_mode&0o777==0o400
    argv=calls[0];assert '--gpus' in argv and '--network' in argv and argv[argv.index('--network')+1]=='none'
    mounts=[argv[i+1] for i,x in enumerate(argv) if x=='--mount']
    assert len(mounts)==len(q.leaves(code))+1
    assert not any('coherent-pair-gpu-probe-v2' in x or '/data/' in x for x in mounts)


@pytest.mark.parametrize('kwargs',[dict(exit_code=1),dict(cleanup=False),dict(post=False)])
def test_mock_host_failures_do_not_become_pass(monkeypatch,tmp_path,kwargs):
    code,_=host_mock(monkeypatch,tmp_path,**kwargs);value=q.host(code,'1'*40)
    assert value['status']=='fail' and json.loads((tmp_path/q.RESULT/'report.json').read_bytes())['status']=='fail'


def test_publication_late_pass_demoted_on_same_fd(tmp_path):
    value={'status':'pass','decision':q.DECISION};path=tmp_path/'report.json'
    q.publish(g,Runtime,path,value,0.,seal=True)
    assert value['status']=='fail' and json.loads(path.read_bytes())['status']=='fail'
    assert value['decision']=='CLOSED_FULLBANK_CONTROL'
    assert path.stat().st_mode&0o777==0o400


def test_failed_postcheck_never_retains_pass_decision(tmp_path):
    value=dict(status='fail',decision=q.DECISION,post_error_type='ValueError')
    q.publish(g,Runtime,tmp_path/'native.json',value,float('inf'))
    assert json.loads((tmp_path/'native.json').read_bytes())['decision']=='CLOSED_FULLBANK_CONTROL'


def test_no_profile_mutation_old_runs_or_numerical_host_import():
    tree=ast.parse((ROOT/q.HELPERS[0]).read_text())
    assert not any(isinstance(n,ast.Global) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        ('select_profile','arithmetic','loss_gradient','fit') for n in ast.walk(tree))
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name=='torch' for a in n.names)
        for n in tree.body)

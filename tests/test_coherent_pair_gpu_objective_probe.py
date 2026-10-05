"""Tiny mocked objective-control lifecycle; no Torch/SciPy/GPU execution."""
import ast
import base64
from contextlib import nullcontext
import csv
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
spec=importlib.util.spec_from_file_location('objective_probe_test',ROOT/'infra/coherent_pair_gpu_objective_probe.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
full,g,real_rt,fixture=q.helpers(ROOT)
sys.path.insert(0,str(ROOT/'src'))
from world_reward import coherent_pair_packed_torch as scorer,coherent_route_scorer as core
from world_reward.coherent_pair_packed_score import score_pair_marginal_packed


class Runtime:
    @staticmethod
    def require(value,message):
        if not value:raise ValueError(message)
    @staticmethod
    def canonical(path):
        path=Path(path);assert not any(p.is_symlink() for p in (path,*path.parents));return path.resolve()
    @staticmethod
    def identity(path,limit,**_):
        data=Path(path).read_bytes();assert 0<len(data)<=limit
        return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    @staticmethod
    def write(path,raw,mode):
        with path.open('xb') as f:f.write(raw)
        path.chmod(mode)
    @staticmethod
    def strict(raw):return json.loads(raw)


class Tensor:
    def __init__(self,array,device='cuda:0'):self.array=np.asarray(array);self.device=device;self.requires_grad=False
    @property
    def shape(self):return self.array.shape
    @property
    def dtype(self):return self.array.dtype
    def cpu(self):return self
    def detach(self):return self
    def numpy(self):return self.array
    def item(self):return self.array.item()
    def __getitem__(self,key):
        if isinstance(key,tuple):key=tuple(k.array if type(k) is Tensor else k for k in key)
        elif type(key) is Tensor:key=key.array
        return Tensor(self.array[key],self.device)
    def __bool__(self):return bool(self.array)
    def __int__(self):return int(self.array)
    def __invert__(self):return Tensor(~self.array)
    def __and__(self,x):return Tensor(self.array & x.array)
    def any(self):return Tensor(self.array.any())
    def all(self):return Tensor(self.array.all())
    def sum(self):return Tensor(self.array.sum())
    def max(self):return Tensor(self.array.max())
    def __add__(self,x):return Tensor(self.array+(x.array if type(x) is Tensor else x))
    __radd__=__add__
    def __sub__(self,x):return Tensor(self.array-(x.array if type(x) is Tensor else x))
    def __mul__(self,x):return Tensor(self.array*(x.array if type(x) is Tensor else x))
    __rmul__=__mul__
    def __truediv__(self,x):return Tensor(self.array/(x.array if type(x) is Tensor else x))
    def __matmul__(self,x):return Tensor(self.array@x.array)


def fake_torch():
    return SimpleNamespace(Tensor=Tensor,float64=np.dtype('float64'),bool=np.dtype('bool'),no_grad=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda:None),tensor=lambda a,dtype,device:Tensor(np.array(a,dtype=dtype,copy=True),device),
        zeros_like=lambda a:Tensor(np.zeros_like(a.array),a.device),zeros=lambda shape,dtype,device:Tensor(np.zeros(shape,dtype=dtype),device),
        full=lambda shape,value,dtype,device:Tensor(np.full(shape,value,dtype=dtype),device),stack=lambda x:Tensor(np.stack([a.array for a in x])),
        isfinite=lambda x:Tensor(np.isfinite(x.array)),isnan=lambda x:Tensor(np.isnan(x.array)),exp=lambda x:Tensor(np.exp(x.array)),log=lambda x:Tensor(np.log(x.array)))


def mock_device(monkeypatch):
    torch=fake_torch();monkeypatch.setitem(sys.modules,'torch',torch);calls=[]
    def score(p,theta,temperature,alpha):
        calls.append(alpha);cpu=score_pair_marginal_packed(p.packed,theta,temperature=temperature,alpha=alpha)
        fields={n:Tensor(a.copy()) if type(a) is np.ndarray else a for n,a in vars(cpu).items()}
        return scorer.TorchMarginalPairScores(**fields,device='cuda:0')
    monkeypatch.setattr(scorer,'prepare_marginal_packed_torch',lambda packed,device:SimpleNamespace(packed=packed))
    monkeypatch.setattr(scorer,'score_pair_marginal_packed_torch',score)
    generic=SimpleNamespace(FLOAT_FIELDS=g.FLOAT_FIELDS,BOOL_FIELDS=g.BOOL_FIELDS,bits=g.bits,
        device_fingerprint=lambda _,p:core._fingerprint((p.packed.arrays,p.packed.identity)))
    return torch,generic,calls


def distribution(tmp_path,*,cache=True,bad=None):
    root=tmp_path/'site';dist=root/'scipy-1.16.3.dist-info';(root/'scipy').mkdir(parents=True);dist.mkdir()
    data={'scipy/optimize.py':b'authored test source',dist.name+'/LICENSE.txt':b'Redistribution and use in source and binary forms'}
    rows=[]
    for name,raw in data.items():
        (root/name).write_bytes(raw);rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))])
    if cache:
        path=root/'scipy/__pycache__/optimize.cpython-311.pyc';path.parent.mkdir();path.write_bytes(b'opaque cache');rows.append([str(path.relative_to(root)),'',''])
    if bad=='blank':
        (root/'scipy/foreign.py').write_bytes(b'x');rows.append(['scipy/foreign.py','',''])
    if bad=='hash':rows[0][1]='sha256='+'a'*43
    if bad=='escape':rows.append(['../escape.py','sha256='+'a'*43,'1'])
    rows.append([dist.name+'/RECORD','',''])
    with (dist/'RECORD').open('w') as stream:csv.writer(stream).writerows(rows)
    files=[Path(r[0]) for r in rows]
    return SimpleNamespace(version='1.16.3',files=files,locate_file=lambda p:root/p)


def test_real_host_import_denies_numpy_torch_scipy():
    script="""import importlib.abc,importlib.util,sys,pathlib
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','torch','scipy'):raise RuntimeError('eager numerical import')
sys.meta_path.insert(0,Deny());s=importlib.util.spec_from_file_location('entry',sys.argv[1]);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
q.helpers(pathlib.Path(sys.argv[2]));assert q.manifest()['fd_calls']==72
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',script,str(ROOT/q.HELPERS[0]),str(ROOT)],capture_output=True,timeout=10)
    assert r.returncode==0,r.stderr.decode()


def test_frozen_helpers_manifest_and_shell():
    for name,(size,digest) in {**full.REUSED,q.HELPERS[2]:q.FULLBANK_PIN,'src/world_reward/coherent_pair_marginal_objective.py':q.OBJECTIVE_PIN}.items():
        raw=(ROOT/name).read_bytes();assert (len(raw),hashlib.sha256(raw).hexdigest())==(size,digest)
    assert q.manifest()['solver_options']==dict(maxiter=20,maxfun=100,maxls=20,maxcor=10,ftol=1e-12,gtol=1e-6)
    assert len(q.leaves(ROOT))==len(q.HELPERS)+2 and len(set(q.leaves(ROOT)))==len(q.leaves(ROOT))
    assert subprocess.run(['bash','-n',str(ROOT/q.HELPERS[1])],capture_output=True).returncode==0


def test_native_record_populated_claims_and_image_caches(monkeypatch,tmp_path):
    d=distribution(tmp_path);monkeypatch.setattr(q.importlib.metadata,'distribution',lambda _:d)
    evidence=q.scipy_evidence(Runtime)
    assert evidence['version']=='1.16.3' and evidence['populated_record_claims_verified']
    assert evidence['unclaimed_cache_files']==1 and evidence['caches_image_anchored_not_record_certified']
    assert q.scipy_evidence(Runtime)==evidence


@pytest.mark.parametrize('bad',['blank','hash','escape','version','missing'])
def test_bad_native_scipy_contract_rejected(monkeypatch,tmp_path,bad):
    d=distribution(tmp_path,bad=bad)
    if bad=='version':d.version='1.15.0'
    if bad=='missing':d.files=[]
    monkeypatch.setattr(q.importlib.metadata,'distribution',lambda _:d)
    with pytest.raises((ValueError,AssertionError,FileNotFoundError)):q.scipy_evidence(Runtime)


def test_short_solver_fixed_options_exact_two_arms_and_geometry_freeze():
    calls=[];report={}
    def evaluate(theta,alpha,arm):
        calls.append((arm,theta.copy(),alpha))
        return (.5*float(theta@theta),theta.copy()) if arm=='A' else (.5*alpha*alpha,np.array([alpha]))
    def minimize(fn,x,**kwargs):
        assert kwargs['method']=='L-BFGS-B' and kwargs['jac'] is True and kwargs['options']==q.manifest()['solver_options']
        fn(x)
        return SimpleNamespace(x=x.copy(),success=True,status=0,nit=0)
    rows=q.solve(np,evaluate,minimize,np.ones(17),lambda:None,report)
    assert len(rows)==2 and all(r['control_passed'] and r['evaluations']==3 for r in rows)
    assert all(not theta.any() for arm,theta,alpha in calls if arm=='B')


@pytest.mark.parametrize('failure',['not_success','not_stationary','loss_increase','cap'])
def test_short_solver_closes_without_restart(failure):
    report={};attempts=[]
    def evaluate(theta,alpha,arm):return (1.,np.ones(17 if arm=='A' else 1)) if failure=='not_stationary' else (0.,np.zeros(17 if arm=='A' else 1))
    def minimize(fn,x,**kwargs):
        attempts.append(1)
        if failure=='cap':
            for _ in range(101):fn(x)
        return SimpleNamespace(x=x.copy() if failure!='loss_increase' else np.ones_like(x),success=failure!='not_success',status=1,nit=1)
    if failure=='loss_increase':
        def evaluate(theta,alpha,arm):return float(theta.sum()),np.zeros(17 if arm=='A' else 1)
    with pytest.raises(RuntimeError):q.solve(np,evaluate,minimize,np.zeros(17),lambda:None,report)
    assert len(attempts)==1


def test_actual_tiny_packed_objective_fd_with_fake_device_no_native_solver(monkeypatch):
    torch,generic,calls=mock_device(monkeypatch);report={};snapshots=[]
    monkeypatch.setattr(q,'scipy_evidence',lambda _:dict(version='1.16.3'))
    monkeypatch.setitem(sys.modules,'scipy.optimize',SimpleNamespace(minimize=lambda *_:None))
    def solve(np,evaluate,minimize,theta,check,report):
        a,da=evaluate(np.zeros(17),0.,'A');b,db=evaluate(np.zeros(17),0.,'B')
        assert np.isfinite(a) and np.isfinite(b) and da.shape==(17,) and db.shape==(1,)
        report['solvers']=[dict(arm='A'),dict(arm='B')]
    monkeypatch.setattr(q,'solve',solve)
    q.measure(np,torch,full,generic,Runtime,fixture,lambda:None,report,snapshots,object_count=2)
    assert len(calls)==78 and report['fd_calls']==72 and report['counts']==[4,2,64]
    assert report['phase']=='complete' and report['positive_groups']==2 and report['snapshots_rehashed_after']
    assert report['source_fixtures_rehashed_after'] and len(report['controls'])==len(report['solvers'])==2


def test_failed_upload_rehashes_available_original_snapshots(monkeypatch):
    torch,generic,calls=mock_device(monkeypatch);report={};snapshots=[]
    def fail(*_,**__):raise RuntimeError('Manufactured upload failure')
    monkeypatch.setattr(scorer,'prepare_marginal_packed_torch',fail)
    with pytest.raises(RuntimeError,match='upload failure'):
        q.measure(np,torch,full,generic,Runtime,fixture,lambda:None,report,snapshots,object_count=2)
    assert not calls and len(snapshots)==1 and {'bank','cache','packed'}<=snapshots[0].keys()
    assert report['source_fixtures_rehashed_after'] and report['snapshots_rehashed_after']
    assert report['phase']!='complete'


def test_publication_failure_decision_same_owned_fd(tmp_path):
    v=dict(status='pass',decision=q.DECISION);q.publish(g,Runtime,tmp_path/'native.json',v,0.)
    assert v['status']=='fail' and v['decision']=='CLOSED_OBJECTIVE_CONTROL'
    assert json.loads((tmp_path/'native.json').read_bytes())==v


def receipt():
    digest='a'*64;binding=dict(producer_revision='1'*40);qualification=dict(receipts=q.PRIOR_PINS)
    value=dict(schema=q.SCHEMA,stage='coherent_pair_gpu_objective_native',status='pass',phase='complete',producer_revision='1'*40,
        source_binding=binding,manifest=q.manifest(),image_id=q.IMAGE,fullbank_qualification=qualification,counts=[4,3600,64],
        candidate_rows=28800,tuple_rows=512,bridge_rows=230400,native_tokens=1500,group_shape=[4,3600],fd_calls=72,
        route_references=1843200,native_supported_cells=28800,group_supported_cells=14400,positive_groups=2,
        controls_completed=2,solvers_completed=2,source_fixtures_rehashed_after=True,snapshots_rehashed_after=True,source_rehashed_after=True,
        real_fit_executed=False,models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,
        quality_verified=False,adoption=False,global_optimum_claimed=False,decision=q.DECISION,
        source_sha256=digest,packed_sha256=digest,identity_sha256=digest,full_original_observations_sha256=digest,
        runtime=dict(numpy='1.26.3',torch='2.5.1+cu124',cuda='12.4',capability=[9,0],tf32=False,deterministic_algorithms=True,
            cublas_workspace_config=':4096:8',gpu_name='NVIDIA H100 NVL',gpu_total_bytes=99456909312,python='3.11.10'),
        peak_torch_allocated=1024,peak_torch_reserved=2048,max_rss_bytes=4096)
    evidence=dict(version='1.16.3',first_native_record_census=True,populated_record_claims_verified=True,
        caches_image_anchored_not_record_certified=True,entries=20,license_files={'LICENSE':dict(bytes=20,sha256=digest)})
    value.update(scipy_before=evidence,scipy_after=dict(evidence))
    value['controls']=[dict(alpha=a,repeat_bits_exact=True,cpu_packed_objective_passed=True,fd17_and_alpha_passed=True,
        gradient_shape=[17],scalar_gradient_shape=[1],group_shape=[4,3600],native_shape=[4,2,3600],identity_sha256=digest,
        distribution_fingerprint=digest,parameter_fingerprint=digest,loss_a=1.,loss_b=1.,seconds=2.) for a in (0.,.3125)]
    value['solvers']=[dict(arm=arm,success=True,control_passed=True,status=0,iterations=2,evaluations=5,
        projected_gradient_inf=0.,initial_loss=1.,final_loss=.5,seconds=1.,parameters=[0.]*(17 if arm=='A' else 1)) for arm in ('A','B')]
    return value,binding,qualification


def test_complete_native_receipt():
    value,binding,qualification=receipt();q.validate_native(Runtime,value,binding,'1'*40,qualification)


@pytest.mark.parametrize('path,replacement',[
    (('fd_calls',),70),(('counts',),[4,32,64]),(('positive_groups',),1),(('route_references',),1),
    (('source_rehashed_after',),False),(('scipy_after','version'),'1.15'),(('runtime','tf32'),True),
    (('runtime','cuda'),'11.8'),(('controls',0,'repeat_bits_exact'),False),
    (('controls',1,'native_shape'),[4,2,32]),(('controls',0,'distribution_fingerprint'),'b'*64),
    (('solvers',0,'iterations'),21),(('solvers',1,'parameters'),[-1.]),(('solvers',0,'success'),False),
    (('solvers',1,'projected_gradient_inf'),float('nan')),(('peak_torch_reserved',),q.MEMORY+1),
    (('max_rss_bytes',),q.MEMORY+1),(('max_rss_bytes',),0),(('runtime','numpy'),'1.26.4'),
])
def test_tampered_native_receipt_rejected(path,replacement):
    value,binding,qualification=receipt();node=value
    for key in path[:-1]:node=node[key]
    node[path[-1]]=replacement
    with pytest.raises(ValueError):q.validate_native(Runtime,value,binding,'1'*40,qualification)


@pytest.mark.parametrize('failure',[None,'native_exit','cleanup','postsource'])
def test_mock_real_host_owned_cid_narrow_mounts_and_fail_demotion(monkeypatch,tmp_path,failure):
    monkeypatch.setattr(q,'ROOT',tmp_path);(tmp_path/'results').mkdir();value,binding,qualification=receipt()
    prior=dict(source_binding=binding,manifest=q.manifest(),image={'Id':q.IMAGE},fullbank_qualification=qualification);proofs=[];commands=[]
    def proof(*_):
        proofs.append(1);return dict(prior,image={}) if failure=='postsource' and len(proofs)>1 else prior
    def run(argv,**_):
        commands.append(argv);out=tmp_path/q.RESULT
        assert out.stat().st_mode&0o777==0o755
        (out/'container.cid').write_bytes(b'a'*64)
        Runtime.write(out/'native.json',(json.dumps(value)+'\n').encode(),0o400)
        return SimpleNamespace(returncode=1 if failure=='native_exit' else 0,stdout=b'',stderr=b'')
    generic=SimpleNamespace(control=lambda *_:b'',cleanup=lambda *_:failure!='cleanup',error_family=g.error_family)
    monkeypatch.setattr(q,'helpers',lambda *_:(full,generic,Runtime,None));monkeypatch.setattr(q,'proof',proof)
    monkeypatch.setattr(q.subprocess,'run',run);monkeypatch.setattr(q.os,'chown',lambda *_:None)
    monkeypatch.setattr(q.signal,'signal',lambda *_:None);monkeypatch.setattr(q.signal,'setitimer',lambda *_:None)
    actual_stat=Path.stat
    def mocked_stat(path,*a,**kw):
        s=actual_stat(path,*a,**kw)
        if path==tmp_path/q.RESULT:
            fields=list(s);fields[4]=1000;return os.stat_result(fields)
        return s
    monkeypatch.setattr(Path,'stat',mocked_stat);old=os.umask(0o077)
    try:report=q.host(ROOT,'1'*40)
    finally:os.umask(old)
    assert report['status']==('pass' if failure is None else 'fail')
    assert (tmp_path/q.RESULT/'report.json').stat().st_mode&0o777==0o400
    assert (tmp_path/q.RESULT).stat().st_mode&0o777==0o500
    mounts=[commands[0][i+1] for i,x in enumerate(commands[0]) if x=='--mount']
    assert len(mounts)==len(q.leaves(ROOT))+1 and not any('/results/coherent-pair-gpu-fullbank-cost-v1' in p for p in mounts)
    if failure:assert report['decision']=='CLOSED_OBJECTIVE_CONTROL'


def test_source_no_global_mutation_old_run_or_autograd():
    tree=ast.parse((ROOT/q.HELPERS[0]).read_text());assert not any(isinstance(n,ast.Global) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        ('select_profile','backward','grad','fit_coherent_pairs') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Import) and any(a.name in ('numpy','torch','scipy') for a in n.names) for n in tree.body)

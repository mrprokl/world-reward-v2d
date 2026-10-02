"""Pure fresh-load/control-order stubs; no Torch/assets/GPU or large geometry."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('test_fresh_replay',infra/'mhr_fresh_replay_gate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def completed(gate):
    return [{'optimized_execution':o,'apply_correctives':c,'status':'complete','completed_calls':3,
             'fresh_model_loaded':True,'replay':{'bitexact':True}} for o,c in gate.CONDITIONS]


def test_unoptimized_only_bit_gate_optimized_failure_descriptive(gate):
    conditions=completed(gate);conditions[0]['replay']['bitexact']=False
    conditions[1]['status']='fail';assert gate.outcome(conditions)
    conditions[2]['replay']['bitexact']=False;assert not gate.outcome(conditions)


@pytest.mark.parametrize('fault',['condition','duplicate','calls','fresh','unknown','missingcomparison'])
def test_incomplete_or_nonexact_never_pass(gate,fault):
    x=completed(gate)
    if fault=='condition':x.pop()
    if fault=='duplicate':x[3]['apply_correctives']=False
    if fault=='calls':x[2]['completed_calls']=2
    if fault=='fresh':x[3]['fresh_model_loaded']=False
    if fault=='unknown':x[2]['replay']['bitexact']=None
    if fault=='missingcomparison':x[2].pop('replay')
    assert not gate.outcome(x)


def test_four_fresh_models_exact_three_call_order_no_warmup(gate,monkeypatch):
    class Tensor:
        def __init__(self,v):self.v=np.asarray(v)
        def __getitem__(self,item):return Tensor(self.v[item])
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.v
        def __len__(self):return len(self.v)
    models=[];contexts=[];active=[None];calls=[]
    class Context:
        def __init__(self,value):self.value=value
        def __enter__(self):active[0]=self.value;contexts.append(self.value)
        def __exit__(self,*args):active[0]=None
    class Model:
        _c=SimpleNamespace(_get_method=lambda name:SimpleNamespace(schema='actual-forward-schema'))
        def float(self):return self
        def eval(self):return self
        def named_parameters(self):return []
        def named_buffers(self):return []
        def get_joint_names(self):return list(range(127))
        def get_parameter_names(self):return list(range(249))
        def get_num_identity_blendshapes(self):return 45
        def get_num_face_expression_blendshapes(self):return 72
    def load(path,*,map_location):
        assert map_location=='cuda';model=Model();models.append(model);return model
    torch=SimpleNamespace(jit=SimpleNamespace(load=load,optimized_execution=Context),manual_seed=lambda seed:None,
        cuda=SimpleNamespace(manual_seed_all=lambda seed:None,empty_cache=lambda:None),
        as_tensor=lambda x,**kw:Tensor(x),zeros_like=lambda x:Tensor(np.zeros_like(x.v)))
    monkeypatch.setattr(gate.diagnosis,'script_sources',lambda model:[{'module':'','code_sha256':'a'*64}])
    def forward(model,tensors,correctives):
        index=models.index(model);calls.append((index,active[0],correctives,len(tensors[0]),tuple(t.v.copy() for t in tensors)))
        values=(np.zeros((2,3),np.float32),np.ones((2,8),np.float32))
        if active[0] and sum(c[0]==index for c in calls)==3:values[0][0,0]=np.nextafter(np.float32(0),np.float32(1))
        return values
    report={'conditions':[],'forward_calls':0};snapshots=[]
    gate.run_conditions(torch,Path('/own/model.pt'),report,lambda:snapshots.append(report['forward_calls']),forward)
    assert len(models)==4 and len(set(map(id,models)))==4 and contexts==[True,True,False,False]
    assert report['forward_calls']==12 and [x[3] for x in calls]==[1,216,216]*4
    assert report['first_replay_failure']['condition_index']==0 and gate.outcome(report['conditions'])
    own=gate.diagnosis.own_inputs()
    for start in range(0,12,3):
        assert all(not np.count_nonzero(v) for v in calls[start][4])
        assert all(np.array_equal(v,w) for v,w in zip(calls[start+1][4],own))
        assert all(v.tobytes()==w.tobytes() for v,w in zip(calls[start+1][4],calls[start+2][4]))
    assert max(snapshots)==12


def test_frozen_exclusive_failure_report_no_torch_load(gate,monkeypatch,tmp_path):
    (tmp_path/'results').mkdir();monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'b'*64);monkeypatch.setattr(gate.platform,'system',lambda:'Linux')
    original=Path.iterdir;monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original(p))
    with pytest.raises(ValueError,match='model mismatch'):gate.main([])
    path=tmp_path/'results/mhr-fresh-replay.json';before=path.read_bytes();r=json.loads(before)
    assert r['status']=='fail' and r['forward_calls']==0 and r['warmup_calls']==0 and r['kernel_cause_verified'] is False
    with pytest.raises(FileExistsError):gate.main([])
    assert path.read_bytes()==before


def test_four_subprocesses_hard_global_budget_and_worker_nonce(gate,monkeypatch,tmp_path):
    (tmp_path/'results').mkdir();model=tmp_path/'weights/mhr/mhr_model.pt';model.parent.mkdir(parents=True);model.touch()
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'b'*64);monkeypatch.setattr(gate.platform,'system',lambda:'Linux')
    original_iter=Path.iterdir;monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original_iter(p))
    original_stat=Path.stat;monkeypatch.setattr(Path,'stat',lambda p,*a,**kw:SimpleNamespace(st_size=696110248,st_mode=original_stat(p).st_mode) if p==model else original_stat(p,*a,**kw))
    original_hash=gate.sha256;monkeypatch.setattr(gate,'sha256',lambda p:gate.diagnosis.MODEL_SHA if p==model else original_hash(p))
    path=tmp_path/'results/mhr-fresh-replay.json';seen=[]
    def child(args,*,check,timeout,env):
        i=int(args[-1]);seen.append(i);assert check is False and 0<timeout<=60 and len(env['WR_FRESH_REPLAY_NONCE'])==64
        r=json.loads(path.read_text());assert r['pending_condition']==i
        c=completed(gate)[i];c['worker_pid']=1000+i;r['conditions'].append(c);r['forward_calls']+=3
        path.write_text(json.dumps(r));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess,'run',child);gate.main([])
    r=json.loads(path.read_text());assert r['status']=='pass' and r['forward_calls']==12 and seen==[0,1,2,3]
    before=path.read_bytes()
    with pytest.raises(RuntimeError,match='immutable'):gate.main(['--condition','0'])
    assert path.read_bytes()==before


def test_child_timeout_preserves_partial_replay_failure(gate,monkeypatch,tmp_path):
    (tmp_path/'results').mkdir();model=tmp_path/'weights/mhr/mhr_model.pt';model.parent.mkdir(parents=True);model.touch()
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'b'*64);monkeypatch.setattr(gate.platform,'system',lambda:'Linux')
    original_iter=Path.iterdir;monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original_iter(p))
    original_stat=Path.stat;monkeypatch.setattr(Path,'stat',lambda p,*a,**kw:SimpleNamespace(st_size=696110248,st_mode=original_stat(p).st_mode) if p==model else original_stat(p,*a,**kw))
    original_hash=gate.sha256;monkeypatch.setattr(gate,'sha256',lambda p:gate.diagnosis.MODEL_SHA if p==model else original_hash(p))
    path=tmp_path/'results/mhr-fresh-replay.json'
    def child(args,*,check,timeout,env):
        r=json.loads(path.read_text());r['forward_calls']=2;r['first_replay_failure']={'descriptive_only':True}
        path.write_text(json.dumps(r));raise subprocess.TimeoutExpired(args,timeout)
    monkeypatch.setattr(gate.subprocess,'run',child)
    with pytest.raises(subprocess.TimeoutExpired):gate.main([])
    r=json.loads(path.read_text());assert r['status']=='fail' and r['forward_calls']==2 and r['first_replay_failure'] is not None


def test_frozen_fixture_helper_source_runtime_and_no_tolerance(gate):
    assert gate.MAX_CALLS==12 and gate.TOTAL_SECONDS==60 and gate.SEED==0
    source=Path(gate.__file__).read_text();wrapper=Path(gate.__file__).with_name('run_mhr_fresh_replay_gate.sh').read_text()
    assert 'diagnosis.own_inputs()' in source and 'diagnosis.bit_comparison(first,replay)' in source
    assert 'diagnosis.script_sources(model)' in source and 'diagnostic_helper_sha256' in source
    assert "torch.jit.load(str(model_path),map_location='cuda').float().eval()" in source and 'torch.jit.optimized_execution(optimized)' in source
    assert source.index("os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'")<source.index('import torch')
    assert 'torch.use_deterministic_algorithms(True,warn_only=False)' in source and 'allow_tf32=False' in source
    assert '63s docker run' in wrapper and '--network none --memory 16g --cpus 4' in wrapper
    assert not any(f'src=$ROOT/{p}' in wrapper for p in ('data','outputs','vendor','validation','weights/cari4d'))
    assert 'weights/mhr,readonly' in wrapper
    with pytest.raises(SystemExit):gate.main(['--tolerance','1e-6'])

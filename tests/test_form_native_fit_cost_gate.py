"""Real-loop cost hooks with tiny clocks; no Torch/GPU/model/data locally."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_hoi_external_predict as predictor


class Indices:
    device='fake_cuda'
    def __len__(self):return 96


def harness(monkeypatch, *, seconds=1., reserved=20, budget=2400.):
    clock=[10.]
    monkeypatch.setattr(predictor.time,'monotonic',lambda:clock[0])
    cuda=SimpleNamespace(synchronize=lambda:None,current_device=lambda:0,
        get_device_properties=lambda _:SimpleNamespace(total_memory=100),
        max_memory_reserved=lambda:reserved,max_memory_allocated=lambda:10)
    torch=SimpleNamespace(cuda=cuda,equal=lambda a,b:True,arange=lambda *a,**kw:Indices())
    class Native:
        def __init__(self):
            self.frame_indices=np.arange(96);self.start_step=0
            self.cfg=SimpleNamespace(num_steps=300,batch_size=0,penetration_start_fraction=.6)
            self.calls=[]
        def loss(self,indices,step,*,include_diagnostics=True):
            self.calls.append((indices,step,include_diagnostics));return ('untouched_native_loss',step)
        def run(self):
            for step in range(4):
                self.loss(Indices(),step,include_diagnostics=step==0)
                clock[0]+=seconds
            return 'native_run_result'
    report={};cls=predictor._fit_cost_gated_class(Native,torch,report,fit_started=0.,budget_seconds=budget)
    return cls(),clock,report,Native


def test_three_completed_updates_pass_and_original_run_loss_unchanged(monkeypatch):
    instance,_,report,native=harness(monkeypatch)
    assert instance.run()=='native_run_result'
    assert type(instance).run is native.run
    assert len(instance.calls)==4 and [x[1] for x in instance.calls]==list(range(4))
    gate=report['fit_cost_gate'];phase=gate['phases'][0]
    assert phase['measured_update_indices']==[0,1,2] and phase['seconds']==[1.,1.,1.]
    assert phase['remaining_native_updates']==298 and phase['projected_inclusive_seconds']==551.
    assert gate['status']=='early_cost_pass_PEN_unmeasured'
    assert gate['native_run_replaced'] is False and gate['extra_loss_calls']==0


def test_three_actual_updates_reject_before_fourth_native_forward(monkeypatch):
    instance,_,report,_=harness(monkeypatch,seconds=10.)
    with pytest.raises(ValueError,match='projected-time'):instance.run()
    assert len(instance.calls)==3
    assert report['fit_cost_gate']['status']=='rejected_projected_time'
    assert report['fit_cost_gate']['rejected_before_update']==3


def test_initialization_and_active_deadline_are_included(monkeypatch):
    instance,clock,report,_=harness(monkeypatch,budget=500.)
    clock[0]=300.
    with pytest.raises(ValueError,match='projected-time'):instance.run()
    assert report['fit_cost_gate']['initialization_seconds']==300.


def test_VRAM_rejects_without_any_new_native_loss(monkeypatch):
    instance,_,report,_=harness(monkeypatch,reserved=81)
    with pytest.raises(ValueError,match='VRAM'):instance.run()
    assert instance.calls==[] and report['fit_cost_gate']['status']=='rejected_VRAM'


def test_first_scheduled_PEN_updates_remeasured_no_extra_pen_call(monkeypatch):
    instance,clock,report,_=harness(monkeypatch)
    for step in range(185):
        instance.loss(Indices(),step,include_diagnostics=False)
        clock[0]+=1. if step<181 else 2.
    phases=report['fit_cost_gate']['phases']
    assert len(instance.calls)==185 and len(phases)==2
    assert phases[1]['measured_update_indices']==[181,182,183]
    assert phases[1]['seconds']==[2.,2.,2.] and phases[1]['late_penetration_cost_measured']
    assert report['fit_cost_gate']['status']=='cost_pass_not_quality'


def test_original_timeline_numsteps_and_consecutive_call_contracts(monkeypatch):
    instance,_,_,_=harness(monkeypatch)
    instance.cfg.batch_size=16
    with pytest.raises(ValueError,match='full96'):instance.loss(Indices(),0)
    instance,_,_,_=harness(monkeypatch);instance.loss(Indices(),0)
    with pytest.raises(ValueError,match='consecutive'):instance.loss(Indices(),2)


def test_fit_only_integration_no_native_run_loop_or_model_duplication():
    source=Path(predictor.__file__).read_text();tree=ast.parse(source)
    fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    calls=[ast.unparse(n.func) for n in ast.walk(fit) if isinstance(n,ast.Call)]
    assert calls.count('native_layer')==1 and calls.count('instance.run')==1
    assert calls.count('_fit_cost_gated_class')==1
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fit_cost_gated_class')
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in ast.walk(helper))
    assert 'torch.cuda.synchronize' in ast.get_source_segment(source,helper)

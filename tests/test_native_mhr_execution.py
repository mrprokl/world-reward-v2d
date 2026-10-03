"""Instrumented tiny runtime test doubles; no Torch/model inference."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def execution():
    spec=importlib.util.spec_from_file_location("native_execution_test",ROOT/"infra/native_mhr_execution.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Runtime:
    def __init__(self):self.enabled=False;self.warn=False;self.optimized=True;self.syncs=0;self.jit=SimpleNamespace(optimized_execution=self.optimized_execution);self.cuda=SimpleNamespace(synchronize=self.synchronize)
    def are_deterministic_algorithms_enabled(self):return self.enabled
    def is_deterministic_algorithms_warn_only_enabled(self):return self.warn
    def use_deterministic_algorithms(self,enabled,*,warn_only):self.enabled=enabled;self.warn=warn_only
    def synchronize(self):self.syncs+=1
    @contextmanager
    def optimized_execution(self,state):
        old=self.optimized;self.optimized=state
        try:yield
        finally:self.optimized=old


def test_all_calls_delegate_original_args_outputs_restore_flags_no_warmup(execution):
    torch=Runtime();seen=[];token=object();output=object()
    def original(*a,**k):
        assert torch.enabled and not torch.warn and not torch.optimized
        seen.append((a,k));return output
    head=SimpleNamespace(mhr_forward=original);report={"phase":"body","active_branch":"original"}
    with execution.strict_native_head(torch,head,report,lambda:None):
        for i in range(21):assert head.mhr_forward(token,value=token)is output
        execution.completed(report)
    assert len(seen)==21 and all(a==(token,)and k=={"value":token}for a,k in seen)
    assert head.mhr_forward is original and not torch.enabled and not torch.warn and torch.optimized
    assert torch.syncs==21 and report["native_head_method_restored"]
    assert not report["native_operations_modified"]and not report["joint_policy_causal_attribution_verified"]


def test_failed_original_scope_restores_without_fallback(execution):
    torch=Runtime();count=[]
    def original():count.append(1);raise RuntimeError("unsupported native")
    head=SimpleNamespace(mhr_forward=original);report={}
    with pytest.raises(RuntimeError,match="unsupported native"):
        with execution.strict_native_head(torch,head,report,lambda:None):head.mhr_forward()
    assert len(count)==1 and head.mhr_forward is original and not torch.enabled and torch.optimized
    assert report["scoped_MHR_attempts"]==1 and report["scoped_MHR_returns"]==0
    assert report["scoped_MHR_calls"][0]["restored"]
    assert report["scoped_MHR_calls"][0]["synchronized"]and torch.syncs==1
    with pytest.raises(ValueError):execution.completed(report)


def test_failed_synchronization_keeps_original_error_and_restores_scope(execution):
    torch=Runtime()
    def sync():raise RuntimeError("sync error")
    torch.cuda.synchronize=sync
    def original():raise ValueError("original error")
    head=SimpleNamespace(mhr_forward=original);report={}
    with pytest.raises(ValueError,match="original error"):
        with execution.strict_native_head(torch,head,report,lambda:None):head.mhr_forward()
    row=report["scoped_MHR_calls"][0]
    assert not row["synchronized"]and row["restored"]and row["synchronization_error_type"]=="RuntimeError"
    assert head.mhr_forward is original and not torch.enabled


def test_global_policy_reentry_rejected(execution):
    torch=Runtime();head=SimpleNamespace();report={}
    def original():return head.mhr_forward()
    head.mhr_forward=original
    with execution.strict_native_head(torch,head,report,lambda:None):
        with pytest.raises(RuntimeError,match="reentrant"):head.mhr_forward()
    assert not torch.enabled and head.mhr_forward is original


@pytest.mark.parametrize("warn",[False,True])
def test_incorrect_estimator_mode_never_delegates(execution,warn):
    torch=Runtime();torch.enabled=True;torch.warn=warn;seen=[];head=SimpleNamespace(mhr_forward=lambda:seen.append(1));report={}
    with execution.strict_native_head(torch,head,report,lambda:None):
        with pytest.raises(RuntimeError,match="ordinary"):head.mhr_forward()
    assert not seen and report["scoped_MHR_attempts"]==0 and torch.enabled and torch.warn==warn


@pytest.mark.parametrize("fault",["count","trace","restored","strict","delegate"])
def test_complete_actual_scope_evidence_required(execution,fault):
    torch=Runtime();head=SimpleNamespace(mhr_forward=lambda:None);report={}
    with execution.strict_native_head(torch,head,report,lambda:None):
        for _ in range(21):head.mhr_forward()
    if fault=="count":report["scoped_MHR_returns"]-=1
    elif fault=="trace":report["scoped_MHR_calls"].pop()
    else:report["scoped_MHR_calls"][0][{"restored":"restored","strict":"strict_enabled","delegate":"delegated_original"}[fault]]=False
    with pytest.raises(ValueError):execution.completed(report)

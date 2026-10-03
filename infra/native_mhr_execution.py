"""Serial execution-policy instrumentation; native operation/arguments unchanged.

Torch deterministic flags are global. No concurrent/reentrant call, fallback,
warmup, numerical repair or guarantee for operators outside this scope.
"""
from contextlib import contextmanager
import sys


@contextmanager
def strict_native_head(torch,head,report,persist):
    original=head.mhr_forward;busy=False
    report.update(scoped_MHR_attempts=0,scoped_MHR_returns=0,scoped_MHR_validated=0,
        native_operations_modified=False,native_arguments_modified=False,scoped_MHR_execution="strictTrue_warnFalse_JITunoptimized",
        joint_policy_causal_attribution_verified=False,scoped_MHR_calls=[])
    def forward(*args,**kwargs):
        nonlocal busy
        if busy:raise RuntimeError("Global native execution policy cannot be concurrent/reentrant")
        enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
        if enabled or warn:raise RuntimeError("Learned estimator must retain explicit ordinary CUDA mode")
        busy=True;report["scoped_MHR_attempts"]+=1
        row=dict(index=report["scoped_MHR_attempts"],phase=report.get("phase"),branch=report.get("active_branch"),
            input_guard_enabled=enabled,input_warn_only=warn,strict_enabled=True,warn_only=False,JIT_optimized=False,
            delegated_original=True,returned=False,validated=False,restored=False,synchronized=False)
        report["scoped_MHR_calls"].append(row);persist()
        try:
            torch.use_deterministic_algorithms(True,warn_only=False)
            with torch.jit.optimized_execution(False):result=original(*args,**kwargs)
            report["scoped_MHR_returns"]+=1;row["returned"]=True
            torch.cuda.synchronize();row["synchronized"]=True
            if not torch.are_deterministic_algorithms_enabled()or torch.is_deterministic_algorithms_warn_only_enabled():raise RuntimeError("Actual strict native scope changed")
            report["scoped_MHR_validated"]+=1;row["validated"]=True;return result
        finally:
            original_error=sys.exc_info()[0] is not None
            try:
                if not row["synchronized"]:
                    try:torch.cuda.synchronize();row["synchronized"]=True
                    except BaseException as error:
                        row["synchronization_error_type"]=type(error).__name__;row["synchronization_error"]=str(error)
                        if not original_error:raise
            finally:
                torch.use_deterministic_algorithms(enabled,warn_only=warn);busy=False;row["restored"]=True;persist()
    head.mhr_forward=forward
    try:yield
    finally:
        head.mhr_forward=original;report["native_head_method_restored"]=True;persist()


def completed(report):
    attempts=report.get("scoped_MHR_attempts")
    if type(attempts)is not int or attempts<21 or any(report.get(k)!=attempts for k in("scoped_MHR_returns","scoped_MHR_validated")):
        raise ValueError("All actual intermediate/external native scopes must complete")
    rows=report.get("scoped_MHR_calls")
    if type(rows)is not list or len(rows)!=attempts:raise ValueError("Complete native scope trace required")
    for index,row in enumerate(rows,1):
        expected=dict(index=index,input_guard_enabled=False,input_warn_only=False,strict_enabled=True,warn_only=False,
            JIT_optimized=False,delegated_original=True,returned=True,validated=True,restored=True,synchronized=True)
        if any(type(row.get(k))is not type(v)or row[k]!=v for k,v in expected.items()):raise ValueError("Native scope evidence differs")

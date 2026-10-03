"""H100c operational route: ordinary learned CUDA, strict unoptimized MHR calls.

Unchanged native source/weights/operations, explicitly instrumented execution.
This joint policy is not a causal attribution, numeric equivalence to H100b,
cross-process determinism or reconstruction accuracy. Exact six-branch SHAM
and all prior native replay gates remain mandatory.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import photometric_native_replay as prior
import native_mhr_execution as execution
from world_reward.data import sha256

core=prior.core
OUT=core.BASE+"/capability_scoped_v1"
STAGE="public_photometric_native_scoped_execution_replay"
BUDGET=180
FAIL_SHA="110eebee952eb1fcd2c6db4b359a1756831ff1ad1d4549f8807f4f62c6a0e11c"


def public_inputs(root,revision):
    record,inputs=prior.historical_inputs(root,revision)
    path=Path(root)/prior.OUT/"report.json";identity=prior.pinned(path,FAIL_SHA,22005);row=json.loads(path.read_text())
    core.require_fields(row,dict(status="fail",phase="photometric_body_inference",producer_revision="e83146a0a85f1530813228e2feb2a9b5a8f8a02b",
        script_sha256="32cd5626e77b272b70f3bb4dafe91b6a706d2dbcc4e4f1bf205cfc28cfde8ec6",deterministic_algorithms=False,warn_only=False,
        body_attempts=4,body_calls_completed=4,parity_heads_completed=4,keypoint_heads_completed=4,fixed_heads_completed=4,selected_replays_completed=0,
        error_type="ValueError",error="SHAM repeated input did not exactly reproduce frozen original native blocks/geometry"))
    if row.get("public_inputs")!=inputs:raise ValueError("Previous empirical replay used different public inputs")
    return record,inputs|dict(failed_empirical_replay=identity)


def helpers():
    return dict(driver=sha256(Path(__file__)),execution=sha256(Path(execution.__file__)),previous_public_contract=sha256(Path(prior.__file__)),
        native_mechanism=core.helper_hashes())


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));revision=os.environ.get("WR_CODE_REVISION","");out=root/OUT
    if(platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or os.geteuid()!=1000
       or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or not re.fullmatch("[0-9a-f]{40}",revision)
       or os.environ.get("WR_IMAGE_ID")!=core.IMAGE or not out.is_dir()or any(out.iterdir())
       or out.resolve()!=out.absolute()or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError("Fresh offline scoped native replay required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=core.IMAGE,
        script_sha256=sha256(Path(__file__)),source_helpers=helpers(),network="none",budget_seconds=BUDGET,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        fitting_performed=False,camera_fit_performed=False,geometry_averaged=False,all_cases_retained=False,source_inputs_assets_rehashed=False,
        official_source_checkpoint_unchanged=True,execution_policy_instrumented=True,hidden_warmup_calls=0,exact_SHAM_gate_relaxed=False,
        cross_process_determinism_verified=False,all_frame_determinism_verified=False,body_attempts=0,body_calls_completed=0,
        parity_head_attempts=0,parity_heads_completed=0,keypoint_head_attempts=0,keypoint_heads_completed=0,
        fixed_head_attempts=0,fixed_heads_completed=0,selected_replay_attempts=0,selected_replays_completed=0,branches=[],selected_replays=[])
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Scoped native replay exceeded180s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist()
            with ExitStack()as stack:
                def load(root,torch):
                    result=core.native.human.load_model(root,torch)
                    stack.enter_context(execution.strict_native_head(torch,result[0].head_pose,report,persist));return result
                core.run_body(root,out,report,persist,revision,deterministic_algorithms=False,public_input_reader=public_inputs,native_model_loader=load)
                execution.completed(report)
            if helpers()!=report["source_helpers"]:raise ValueError("Scoped replay source changed")
            report.update(status="pass",phase="complete",all_cases_retained=True,source_inputs_assets_rehashed=True);report.pop("active_branch",None)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()

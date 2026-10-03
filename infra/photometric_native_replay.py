"""H100b: unchanged native CUDA, empirically exact SHAM; no quality query.

The previous strict-algorithm fixture failed before its first prediction. This
distinct namespace disables the unsupported global PyTorch algorithm guard,
not the byte-exact repeated-input gate. Public RGB/mask and the failed receipt
are reused only through exact immutable historical producer/source identities.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import photometric_capability as core
from world_reward.data import sha256

REVISION="8c68fba8d4f9034c2e19ed2fa9eac394c9f64758"
SOURCE_SHA="d8ada243dd8bd234aaf640f2bd40f56f0e3abfa7d671829780a7898e54822e5f"
RENDER_SHA="21e81b9dd3c6a7c1436382aaa8f4528d0caf56c71b0a81957cf7e7b0e3410b41"
MASK_SHA="d268cc0bd750cf2eb336da390534faadafd8c1271d442cb28e653cf0c8a349c9"
FAIL_SHA="e136b3cd417fd72d54368885e32163250514b692acda9c808f6ea98207a0caaf"
OUT=core.BASE+"/capability_replay_v1"
STAGE="public_photometric_native_empirical_replay"
BUDGET=180


def pinned(path,digest,size=None):
    actual=core.identity(path)
    if actual["sha256"]!=digest or(size is not None and actual["bytes"]!=size):raise ValueError("Exact frozen historical producer required")
    return actual


def historical_source(root):
    return Path(root)/"jobs"/REVISION/"run_photometric_capability/code/infra/photometric_capability.py"


def historical_inputs(root,revision):
    """Read only historical public input contracts and scalar receipts, no NPZ."""
    if not re.fullmatch("[0-9a-f]{40}",revision):raise ValueError("Current source revision required")
    root=Path(root);base=root/core.BASE;source=historical_source(root)
    source_id=pinned(source,SOURCE_SHA)
    paths=[base/"render-report.json",base/"automatic_masks/report.json",base/"capability_v1/report.json"]
    identities=[pinned(p,d,s)for p,d,s in zip(paths,(RENDER_SHA,MASK_SHA,FAIL_SHA),(1873,6602,17361))]
    render,mask,failed=[json.loads(p.read_text())for p in paths]
    core.require_fields(render,dict(status="pass",phase="complete",code_revision=REVISION,frames=1,private_arrays_created=False,truth_arrays_exported=False))
    core.require_fields(mask,dict(status="pass",phase="complete",producer_revision=REVISION,script_sha256=SOURCE_SHA))
    core.require_fields(failed,dict(status="fail",phase="photometric_body_inference",producer_revision=REVISION,
        script_sha256=SOURCE_SHA,error_type="RuntimeError",deterministic_algorithms=True,body_attempts=1,body_calls_completed=0,
        parity_heads_completed=0,keypoint_heads_completed=0,fixed_heads_completed=0,selected_replays_completed=0))
    if not failed.get("error","").startswith("cumsum_cuda_kernel does not have a deterministic implementation"):raise ValueError("Original unsupported native guard failure required")
    spec=importlib.util.spec_from_file_location("world_reward_h100_frozen_public_contract",source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    record,inputs=old.public_mask(root,REVISION)
    if inputs!=failed.get("public_inputs"):raise ValueError("Failed attempt and current public inputs differ")
    return record,inputs|dict(historical_sources=dict(source=source_id,render=identities[0],mask=identities[1],failed_strict_attempt=identities[2]))


def helper_hashes():
    return dict(replay=sha256(Path(__file__)),native_mechanism=core.helper_hashes())


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));revision=os.environ.get("WR_CODE_REVISION","");out=root/OUT
    if(platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or os.geteuid()!=1000
       or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or not re.fullmatch("[0-9a-f]{40}",revision)
       or os.environ.get("WR_IMAGE_ID")!=core.IMAGE or not out.is_dir()or any(out.iterdir())
       or out.resolve()!=out.absolute()or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError("Fresh offline empirical replay route required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=core.IMAGE,
        script_sha256=sha256(Path(__file__)),source_helpers=helper_hashes(),network="none",budget_seconds=BUDGET,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        fitting_performed=False,camera_fit_performed=False,geometry_averaged=False,all_cases_retained=False,
        source_inputs_assets_rehashed=False,global_strict_algorithm_guard=False,official_cumsum_modified=False,
        exact_SHAM_gate_relaxed=False,hidden_warmup_calls=0,cross_process_determinism_verified=False,
        all_frame_determinism_verified=False,body_attempts=0,body_calls_completed=0,parity_head_attempts=0,parity_heads_completed=0,
        keypoint_head_attempts=0,keypoint_heads_completed=0,fixed_head_attempts=0,fixed_heads_completed=0,
        selected_replay_attempts=0,selected_replays_completed=0,branches=[],selected_replays=[])
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Empirical native replay exceeded180s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();core.run_body(root,out,report,persist,revision,deterministic_algorithms=False,public_input_reader=historical_inputs)
            if helper_hashes()!=report["source_helpers"]:raise ValueError("Replay source/import closure changed")
            report.update(status="pass",phase="complete",all_cases_retained=True,source_inputs_assets_rehashed=True);report.pop("active_branch",None)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()

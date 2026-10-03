"""Emit tiny next-stage pins for an independently identified immutable producer.

Run only on the Azure control host. The caller supplies the actual dispatched
producer revision and script SHA; neither trusted identity is learned from the
report itself. Exact readonly payloads are hashed before strict JSON parsing and
again afterward. This inventories a producer's validated execution receipt; it
does not load models/geometry, rerun inference, validate accuracy, or submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

ROOT = Path("/srv/scenesmith/world-reward")
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
CHECKPOINT_SHA = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"
OPTIMIZER_SHA = "84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b"
STAGES = {
    "prepare": {
        "stage": "world_reward_native_cari_shared_initializer_full_video",
        "script": "infra/cari_shared_prepare.py",
        "payloads": ("shared_initializer.pkl", "direct_parameters.npz", "target.npy"),
    },
    "forward": {
        "stage": "world_reward_native_cari_shared_full_video_forward",
        "script": "infra/cari_full_forward.py", "payloads": ("coconet.pth",),
    },
    "refined": {
        "stage": "world_reward_native_cari_shared_full_video_refinement",
        "script": "infra/cari_full_refine.py", "payloads": ("refined.pth",),
    },
}
SPEC_KEYS = {"episode_index", "total_frames", "camera_name", "height", "width"}


def _hex(value, length, label):
    if type(value) is not str or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"Exact lowercase {length}-hex {label} required")
    return value


def _episode(value):
    if type(value) is not int or not 0 <= value < 30:
        raise ValueError("Explicit Track1 integer episode in 0..29 required")
    return value


def output_relative(episode, stage):
    _episode(episode)
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("Exact stage prepare, forward or refined required")
    return f"outputs/episode_{episode:06d}/cari_shared_{stage}_v1"


def _spec(value, episode):
    if type(value) is not dict or set(value) != SPEC_KEYS:
        raise ValueError("Exact complete public clip spec required")
    _episode(value["episode_index"])
    if value["episode_index"] != episode or type(value["total_frames"]) is not int or value["total_frames"] < 96:
        raise ValueError("Pinned full original native clip requires selected episode and integer N>=96")
    if (type(value["camera_name"]) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["camera_name"])
            or any(type(value[name]) is not int or value[name] < 2 for name in ("height", "width"))):
        raise ValueError("Explicit safe original public camera and image dimensions required")
    return value


def _expected(record, expected, label):
    def same(left,right):
        if type(left) is not type(right):return False
        if isinstance(right,list):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
        if isinstance(right,dict):return set(left)==set(right) and all(same(left[k],v) for k,v in right.items())
        return left==right
    if type(record) is not dict or any(not same(record.get(k),v) for k,v in expected.items()):
        raise ValueError(label)


def _receipt(value):
    if type(value) is not dict or set(value) != {"sha256", "bytes"} or type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ValueError("Exact SHA256/positive integer byte count required")
    _hex(value["sha256"],64,"artifact SHA256")


def _no_oracle(record):
    """Reject optional truth/oracle declarations anywhere in the JSON receipt."""
    if isinstance(record,dict):
        for key,value in record.items():
            if key in {"ground_truth_used", "ground_truth_read", "private_truth_read", "hand_labeled_test"} and value is not False:
                raise ValueError("Explicit no truth access or hand-labeled test records required")
            if key=="oracle_modes" and (type(value) is not list or value):
                raise ValueError("Oracle modes must be an empty list")
            if key in {"gt", "ground_truth"} and value not in ({},None):
                raise ValueError("No ground-truth record permitted")
            _no_oracle(value)
    elif isinstance(record,list):
        for value in record:_no_oracle(value)
    elif isinstance(record,float) and not math.isfinite(record):
        raise ValueError("Nonfinite JSON producer value forbidden")


def _regular_readonly(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents)):
        raise ValueError("Canonical absolute file path without any symlink required")
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222 or info.st_size<=0:
        raise ValueError("Nonempty immutable regular producer file required")
    return info


def identity(path):
    before=_regular_readonly(path);digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    after=_regular_readonly(path)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError("Producer file changed while hashing")
    return {"sha256":digest.hexdigest(),"bytes":after.st_size}


def _strict_json(text):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError("Duplicate JSON producer key forbidden")
            result[key]=value
        return result
    def invalid(value):raise ValueError("Nonfinite JSON producer value forbidden: "+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def _chunks(count,size):
    return [min(size,count-start) for start in range(0,count,size)]


def _window_starts(count):
    starts=list(range(0,count-96+1,96))
    if starts[-1]!=count-96:starts.append(count-96)
    return starts


def validate_report(report,episode,stage,producer_revision,producer_script_sha256,files):
    """Stdlib receipt/scalar gate, not an independent geometry/accuracy check."""
    spec=_spec(report.get("clip_spec") if type(report) is dict else None,episode);count=spec["total_frames"]
    _no_oracle(report)
    _expected(report,dict(stage=STAGES[stage]["stage"],status="pass",phase="complete",episode_index=episode,frames=count,
        original_frame_indices=list(range(count)),producer_revision=producer_revision,script_sha256=producer_script_sha256,
        image_id=IMAGE,input_track="track_1",network="none",ground_truth_used=False,private_truth_read=False,
        hand_labeled_test=False,oracle_modes=[],quality_verified=False,adoption_authorized=False,submission_produced=False,
        input_dataset_revision=DATASET_REVISION,source_inputs_assets_rehashed=True,source_helpers_rehashed=True),
        "Complete source-bound full-video public producer receipt required")
    helpers=report.get("source_helpers")
    if type(helpers) is not dict or STAGES[stage]["script"] not in helpers:
        raise ValueError("Actual producer entrypoint helper identity required")
    _receipt(helpers[STAGES[stage]["script"]])
    if helpers[STAGES[stage]["script"]]["sha256"]!=producer_script_sha256:
        raise ValueError("Producer script helper differs from independently supplied source SHA")
    if any(type(index) is not int for index in report["original_frame_indices"]):
        raise ValueError("Original frame indices must remain actual integers")
    _hex(report.get("input_sha256"),64,"input video SHA256")
    if stage!="forward":_hex(report.get("input_report_sha256"),64,"input report SHA256")
    payloads={name:files[name] for name in STAGES[stage]["payloads"]}
    actual=report.get("output_files")
    if type(actual) is not dict or set(actual)!=set(payloads):raise ValueError("Exact producer payload manifest inventory required")
    for row in actual.values():_receipt(row)
    if actual!=payloads:raise ValueError("Complete actual producer payload SHA/size manifest differs")
    if stage=="prepare":
        _expected(report,dict(learned_inference_calls=0,optimizer_calls=0,converter_LM_calls=0,submission_eligible=False,
            identity_fixed_before_first_decode=True,chunk_counts=_chunks(count,16),predictions_frozen_before_replays=True,
            stored_initializer_native_replay_verified=True,shared_identity_verified=True,original_initializer_unchanged=True,
            frozen_outputs_rehashed_after_reference=True,original_frame_coverage_verified=True,
            source_geometry_masks_depth_pose_unchanged=True),"Complete shared native initializer execution receipt required")
        chunks=len(_chunks(count,16))
        _expected(report,{name+suffix:chunks for name in ("native_geometry","native_direct","native_replay","reference")
            for suffix in ("_attempts","_returns","_validated")},"Every actual native/direct/saved/reference chunk required")
        for key,name in (("shared_initializer_sha256","shared_initializer.pkl"),("direct_parameters_sha256","direct_parameters.npz"),("target_sha256","target.npy")):
            if report.get(key)!=files[name]["sha256"]:raise ValueError("Prepared payload receipt link differs")
        errors=report.get("reference_per_frame_mean_mm")
        if (type(errors) is not list or len(errors)!=count or any(type(error) not in (int,float) or not math.isfinite(error)
                or not 0<=error<=2. for error in errors)):
            raise ValueError("Every original prepared reference-fidelity scalar required")
    else:
        _expected(report,dict(source_frames=count,checkpoint_sha256=CHECKPOINT_SHA),"Full original source timeline/checkpoint receipt required")
        name="coconet.pth" if stage=="forward" else "refined.pth"
        _expected(report,dict(bundle_sha256=files[name]["sha256"],bundle_bytes=files[name]["bytes"]),"Saved native bundle SHA/bytes differs")
        _hex(report.get("prepare_report_sha256"),64,"prepare report SHA256")
        if stage=="forward":
            starts=_window_starts(count);windows=len(starts)
            _expected(report,dict(forward_attempts=1,forward_returns=1,forward_validated=1,captured_windows=windows,
                window_starts=starts,hub_attempts=2,hub_returns=2,shared_identity_verified=True,
                raw_prediction_bytes_preserved=True,initializer_bytes_preserved=True,native_global_unchanged=True,hub_restored=True,
                stored_bundle_reread_verified=True,actual_caller_indices_verified=True,native_first_occurrence_assembly_verified=True,
                actual_native_decoder_identity_verified=True,refinement_performed=False,exact_deterministic_forward_claim=False),
                "Complete actual native windows and stored full assembly receipt required")
            _expected(report,{name:windows for name in ("composition_hook_calls","composition_delegate_calls","composition_delegate_returns","composition_verified_calls")},
                "Every actual native window composition receipt required")
        else:
            _expected(report,dict(optimizer_attempts=1,optimizer_returns=1,optimizer_validated=1,requested_steps=300,
                effective_optimizer_updates=301,budget_seconds=7200,history_steps=[0,100,200,300],ground_truth_read=False,
                learned_inference_calls=0,saved_bundle_reloaded_verified=True,frozen_raw_inputs_byte_preserved=True,
                object_mesh_unchanged=True,native_refinement_verified=True,numerical_bit_determinism_claimed=False,
                optimizer_sha256=OPTIMIZER_SHA),"Complete unchanged native full refinement receipt required")
            _expected(report.get("metadata"),dict(native_refinement_verified=True,full_original_frame_coverage_verified=True,
                frozen_parameters_bit_identical=True,requested_steps=300,effective_optimizer_updates=301,batch_size=0),
                "Full original fixed-parameter/update receipt required")
            for key in ("forward_report_sha256","source_bundle_sha256","object_mesh_sha256"):_hex(report.get(key),64,key)
    return spec


def inventory(root,episode,stage,producer_revision,producer_script_sha256):
    """Hash exact immutable outputs and emit pins; mutate no file or directory."""
    _hex(producer_revision,40,"independently supplied producer revision")
    _hex(producer_script_sha256,64,"independently supplied producer script SHA256")
    root=Path(root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir():
        raise ValueError("Canonical existing control-host root required")
    directory=root/output_relative(episode,stage);names={"report.json",*STAGES[stage]["payloads"]}
    if directory.resolve()!=directory or not directory.is_dir() or any(path.is_symlink() for path in (directory,*directory.parents)):
        raise ValueError("Canonical existing immutable producer directory required")
    def snapshot():
        paths=tuple(directory.iterdir())
        if {path.name for path in paths}!=names:raise ValueError("Exact complete producer inventory required; no extra files or caches")
        return {path.name:identity(path) for path in sorted(paths)}
    files=snapshot()  # ALL payloads before any report JSON interpretation.
    report=_strict_json((directory/"report.json").read_text())
    spec=validate_report(report,episode,stage,producer_revision,producer_script_sha256,files)
    if snapshot()!=files:raise ValueError("Producer inventory changed while its receipt was interpreted")
    return dict(schema=f"world-reward-cari-shared-{stage}-pins-v1",clip_spec=spec,
        **{stage:dict(files["report.json"],producer_revision=producer_revision,script_sha256=producer_script_sha256),stage+"_files":files})


def parser():
    class Once(argparse.Action):
        def __call__(self,parser,namespace,value,option_string=None):
            if getattr(namespace,self.dest,None) is not None:parser.error("Each required control must occur exactly once")
            setattr(namespace,self.dest,value)
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument("--episode",type=int,choices=range(30),required=True,action=Once)
    result.add_argument("--stage",choices=tuple(STAGES),required=True,action=Once)
    result.add_argument("--producer-revision",required=True,action=Once)
    result.add_argument("--producer-script-sha256",required=True,action=Once)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if platform.system()!="Linux" or os.environ.get("WR_ROOT")!=str(ROOT):
        raise RuntimeError("Inventory is restricted to the canonical Azure Linux control host")
    pins=inventory(ROOT,args.episode,args.stage,args.producer_revision,args.producer_script_sha256)
    json.dump(pins,sys.stdout,allow_nan=False,separators=(",",":"));sys.stdout.write("\n")


if __name__=="__main__":main()

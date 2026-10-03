"""CPU-only frozen full-native Track1 episode consumer engineering gate.

One readonly actual export is consumed, and only a concise engineering receipt
is written. No rendering, model execution, Torch/Joblib deserialization, fitting,
prediction repacking, Parquet production, quality or eligibility claim occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import time

ROOT=Path("/srv/scenesmith/world-reward")
IMAGE="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
STAGE="world_reward_shared_native_episode_engineering_gate"
BUDGET=300


def output_relative(episode):
    if type(episode) is not int or not 0<=episode<30:raise ValueError("Explicit Track1 integer episode0..29 required")
    return f"outputs/episode_{episode:06d}/cari_shared_episode_v1"


def parser():
    class Once(argparse.Action):
        def __call__(self,parser,namespace,value,option_string=None):
            if getattr(namespace,self.dest,None) is not None:parser.error("Exactly one explicit episode required")
            setattr(namespace,self.dest,value)
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument("--episode",required=True,type=int,choices=range(30),action=Once)
    return result


def identity(path,*,immutable=True):
    path=Path(path)
    if (not path.is_absolute() or path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents))
            or not stat.S_ISREG(path.lstat().st_mode) or immutable and path.stat().st_mode&0o222):
        raise ValueError("Canonical regular immutable source/pin required")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return dict(sha256=digest.hexdigest(),bytes=path.stat().st_size)


def _json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError("Duplicate pin/report JSON keys forbidden")
            result[key]=value
        return result
    def finite(_):raise ValueError("Nonfinite JSON forbidden")
    return json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=finite)


def source_helpers(code):
    names=("infra/cari_shared_episode_gate.py","infra/run_cari_shared_episode_gate.sh","infra/cari_shared_episode_loader.py")
    return {name:identity(code/name) for name in names}


def run(root,out,code,episode,report,persist,*,consumer=None):
    """Testable lifecycle; production resolves only the real pinned consumer."""
    pins_path=code/f"configs/cari_clip_{episode:06d}_shared_export_pins.json"
    pin_id=identity(pins_path);pins=_json(pins_path)
    import cari_clip_inputs as public
    spec=public.PublicClipSpec(**pins["clip_spec"])
    if spec.episode_index!=episode:raise ValueError("Explicit episode differs from actual export pins")
    helpers=source_helpers(code)
    report.update(phase="frozen_episode_consumer",clip_spec=asdict(spec),export_pins=pin_id,source_helpers=helpers)
    persist()
    if consumer is None:
        from cari_shared_episode_loader import load_shared_track1_episode
        consumer=load_shared_track1_episode
    if any(name in sys.modules for name in ("torch","joblib")):raise ValueError("Fresh CPU hash/trajectory-only consumer process required")
    loaded=consumer(root,code,spec,pins)
    loaded.episode.validate()
    manifest=loaded.manifest
    required=dict(stage="world_reward_shared_native_episode_consumer",episode_index=episode,frames=spec.total_frames,
        clip_spec=asdict(spec),input_track="track_1",ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
        original_frame_coverage_verified=True,integrity_and_schema_verified=True,native_direct_export_consumed=True,
        old_LM_conversion_used=False,numerical_geometry_independently_reverified=False,quality_verified=False,
        challenge_performance_verified=False,submission_eligibility_verified=False,submission_eligible=False,final_Parquet_produced=False)
    if type(manifest) is not dict or any(type(manifest.get(k)) is not type(v) or manifest[k]!=v for k,v in required.items()):
        raise ValueError("Complete engineering-only real consumer manifest required")
    if (loaded.episode.total_video_frames!=spec.total_frames
            or manifest.get("export_report_sha256")!=pins["export"]["sha256"]
            or manifest.get("export_files")!=pins["export_files"]
            or any(name in sys.modules for name in ("torch","joblib"))):
        raise ValueError("Actual full pinned export/consumer coverage differs")
    if (identity(pins_path)!=pin_id or _json(pins_path)!=pins or source_helpers(code)!=helpers
            or {path.name for path in out.iterdir()}!={"report.json"}):
        raise ValueError("Frozen consumer pins/source or exclusive output changed")
    report.update(status="pass",phase="complete",frames=spec.total_frames,consumer_manifest=manifest,
        engineering_integrity_verified=True,source_helpers_rehashed=True,export_pins_rehashed=True,
        numerical_geometry_independently_reverified=False,quality_verified=False,submission_eligible=False,final_Parquet_produced=False)


def execute(root,out,code,episode,revision,*,consumer=None):
    """Exclusive live receipt, finally frozen444 on either PASS or failure."""
    if not out.is_dir() or any(out.iterdir()):raise ValueError("Fresh exclusive consumer output required")
    script=code/"infra/cari_shared_episode_gate.py"
    report=dict(stage=STAGE,status="fail",phase="public_integrity",episode_index=episode,producer_revision=revision,
        script_sha256=identity(script)["sha256"],image_id=IMAGE,input_track="track_1",network="none",budget_seconds=BUDGET,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],GPU_used=False,model_calls=0,
        optimizer_calls=0,render_calls=0,Parquet_calls=0,numerical_geometry_independently_reverified=False,
        quality_verified=False,submission_eligible=False,final_Parquet_produced=False)
    path=out/"report.json";started=time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("CPU shared episode consumer exceeded300s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,episode,report,persist,consumer=consumer)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    out=root/output_relative(args.episode)
    historical_root=os.environ.get("WR_HISTORICAL_READONLY_ROOT","0")
    if (platform.system()!="Linux" or root!=ROOT
            or historical_root not in ("0","1") or os.geteuid()!=(0 if historical_root=="1" else 1000)
            or historical_root=="1" and os.getegid()!=0
            or {path.name for path in Path("/sys/class/net").iterdir()}!={"lo"} or os.environ["WR_IMAGE_ID"]!=IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or code!=root/"jobs"/revision/"run_cari_shared_episode_gate/code"
            or Path(__file__).resolve()!=code/"infra/cari_shared_episode_gate.py"
            or any(path.resolve()!=path.absolute() or any(p.is_symlink() for p in (path,*path.parents)) for path in (root,code,out))):
        raise ValueError("Actual immutable offline CPU consumer launcher/image required")
    if historical_root=="1":
        from cari_historical_source import verify_historical_source
        verify_historical_source(root,code/f"configs/cari_clip_{args.episode:06d}_historical_source_pins.json")
    execute(root,out,code,args.episode,revision)


if __name__=="__main__":main()

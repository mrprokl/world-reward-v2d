"""Hash/JSON-only original frontend pins, bound to independently supplied source.

The Azure caller supplies the original public episode length/image dimensions/
camera and actual frontend dispatch revision/script SHA. Those values are never
inferred as trusted predictions from a receipt. Exactly fifteen original public
input files are hashed before JSON interpretation, validated through the existing
report-only contract and rehashed afterward. No arrays, videos, HDF5, meshes,
models, Joblib or GT are decoded; no file is written or chmod'ed. A producer's
recorded numerical checks are provenance, not independent geometry/accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

import cari_clip_inputs as inputs

ROOT=Path("/srv/scenesmith/world-reward")
PRODUCER_SCRIPT="infra/cari_prepare.py"
STAGE="world_reward_native_cari_inputs"


def _hex(value,length,label):
    if type(value) is not str or not re.fullmatch(f"[0-9a-f]{{{length}}}",value):
        raise ValueError(f"Exact independently supplied {length}-hex {label} required")
    return value


def identity(path,immutable=False):
    path=Path(path)
    if (not path.is_absolute() or path.resolve()!=path
            or any(parent.is_symlink() for parent in (path,*path.parents))):
        raise ValueError("Canonical absolute source file without symlink ancestors required")
    before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size<=0 or immutable and before.st_mode&0o222):
        raise ValueError("Nonempty regular source / immutable helper required")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    after=path.lstat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_mode)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode):
        raise ValueError("Source changed while hashing")
    return dict(sha256=digest.hexdigest(),bytes=after.st_size)


def _strict_json(text):
    def pairs(rows):
        value={}
        for key,item in rows:
            if key in value:raise ValueError("Duplicate JSON source-report key forbidden")
            value[key]=item
        return value
    def invalid(value):raise ValueError("Nonfinite JSON source-report constant forbidden")
    value=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)
    def finite(item):
        if isinstance(item,dict):
            for child in item.values():finite(child)
        elif isinstance(item,list):
            for child in item:finite(child)
        elif isinstance(item,float) and not math.isfinite(item):
            raise ValueError("Nonfinite JSON source-report number forbidden")
    finite(value)
    return value


def _same(left,right):
    if type(left) is not type(right):return False
    if isinstance(right,list):return len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    if isinstance(right,dict):return set(left)==set(right) and all(_same(left[k],v) for k,v in right.items())
    return left==right


def _require(record,expected,label):
    if type(record) is not dict or any(not _same(record.get(key),value) for key,value in expected.items()):
        raise ValueError(label)


def validate_current_reports(root,spec,pins,records):
    """Strict public identities; one exact historical input metadata omission."""
    if pins["input_report"]==inputs.LEGACY_INPUT_REPORT:
        raise ValueError("Legacy episode15 producer cannot be resealed as new current frontend inputs")
    for role,record in records.items():
        flags=dict(status="pass",input_track="track_1",episode_index=spec.episode_index,
            ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
        if role in {"inputs","body","depth"}:
            if role!="inputs" or not inputs._input_dataset_omitted(spec,pins):
                flags["input_dataset_revision"]=inputs.DATASET_REVISION
        if "input_dataset_revision" in record and record["input_dataset_revision"]!=inputs.DATASET_REVISION:
            raise ValueError("Explicit public dataset declaration differs")
        _require(record,flags,"Explicit current full public dependency identity and no-oracle flags required")
        inputs.public._no_oracle(record)
    report=records["inputs"]
    _require(report,dict(stage=STAGE,frames=spec.total_frames,original_frame_coverage_verified=True,
        producer_revision=pins["input_report"]["producer_revision"],script_sha256=pins["input_report"]["script_sha256"],
        submission_eligible=False,challenge_performance_verified=False),
        "Current complete original frontend receipt required")
    _require(report.get("depth_validation"),dict(validation_mode="exhaustive",frame_counts={spec.camera_name:spec.total_frames},
        frame_shapes={spec.camera_name:[spec.height,spec.width]}),"Exact original depth camera/frame/grid producer receipt required")
    error=report.get("mesh_pose_frame_roundtrip_max_error_m")
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1e-5:
        raise ValueError("Recorded original rigid mesh/pose frame-change fidelity required")
    video_error=report.get("jpeg_original_RGB_mean_absolute_error")
    if type(video_error) not in (int,float) or not math.isfinite(video_error) or not 0<=video_error<=255:
        raise ValueError("Finite original-RGB export difference diagnostic required")
    return report


def inventory(root,code,spec,producer_revision,producer_script_sha256,*,producer_code=None):
    """Hash original inputs; a caller-authenticated old source may be separate.

    The caller independently verifies the original dispatch/complete Git closure
    before supplying producer_code. Consumer helpers always remain bound to code;
    old sources are only hashed, never imported, executed, repaired or rewritten.
    """
    _hex(producer_revision,40,"original producer revision");_hex(producer_script_sha256,64,"original producer script SHA256")
    if type(spec) is not inputs.PublicClipSpec:raise ValueError("Explicit original public structural clip spec required")
    root,code=Path(root),Path(code)
    if any(not path.is_absolute() or path.resolve()!=path or not path.is_dir() for path in (root,code)):
        raise ValueError("Canonical existing source-bound Azure root/code required")
    producer=code if producer_code is None else Path(producer_code)
    if (not producer.is_absolute() or producer.resolve()!=producer or not producer.is_dir()
            or producer_code is not None and producer.stat().st_mode&0o222):
        raise ValueError("Canonical existing readonly original producer directory required")
    fields=("st_dev","st_ino","st_mode","st_mtime_ns","st_ctime_ns")
    producer_state=tuple(getattr(producer.stat(),name) for name in fields)
    source=identity(producer/PRODUCER_SCRIPT,immutable=True)
    if source["sha256"]!=producer_script_sha256:
        raise ValueError("Original frontend source differs from independently supplied dispatched source SHA")
    # Bind the small report-only helper closure, not any asset/model/initializer.
    helper_names=("infra/cari_clip_pin_inventory.py","infra/run_cari_clip_pin_inventory.sh",
                  "infra/cari_clip_inputs.py","infra/cari96_inputs.py")
    helpers={name:identity(code/name,immutable=True) for name in helper_names}
    names=inputs.source_paths(spec)
    if len(names)!=15:raise ValueError("Exactly fifteen original public inputs required")
    observed={name:identity(root/name) for name in sorted(names)}
    paths,deps=inputs.relative_paths(spec),inputs.dependency_paths(spec)
    pins=dict(schema="world-reward-cari-clip-input-pins-v1",clip_spec=asdict(spec),
        input_report=dict(observed[paths["input_report"]],producer_revision=producer_revision,script_sha256=producer_script_sha256),
        source_files=observed)
    inputs.validate_pins(spec,pins)
    # Preparse all six reports strictly only AFTER every source payload hash.
    records={"inputs":_strict_json((root/paths["input_report"]).read_text())}
    records.update({role:_strict_json((root/name).read_text()) for role,name in deps.items()})
    validate_current_reports(root,spec,pins,records)
    checked=inputs.validate_reports(root,spec,pins)  # Hash/JSON-only; never verify_public_inputs.
    if checked!=records:raise ValueError("Frontend reports changed during report-only validation")
    if ({name:identity(root/name) for name in sorted(names)}!=observed
            or identity(producer/PRODUCER_SCRIPT,immutable=True)!=source
            or tuple(getattr(producer.stat(),name) for name in fields)!=producer_state
            or {name:identity(code/name,immutable=True) for name in helper_names}!=helpers):
        raise ValueError("Original frontend inputs or source/helper closure changed during inventory")
    return pins


def parser():
    class Once(argparse.Action):
        def __call__(self,parser,namespace,value,option_string=None):
            if getattr(namespace,self.dest,None) is not None:parser.error("Each required original structural/source control must occur once")
            setattr(namespace,self.dest,value)
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument("--episode",type=int,choices=range(30),required=True,action=Once)
    for name in ("frames","height","width"):result.add_argument("--"+name,type=int,required=True,action=Once)
    for name in ("camera-name","producer-revision","producer-script-sha256"):
        result.add_argument("--"+name,required=True,action=Once)
    return result


def main(argv=None):
    args=parser().parse_args(argv);code=Path(os.environ["WR_CODE"])
    if platform.system()!="Linux" or os.environ.get("WR_ROOT")!=str(ROOT):
        raise RuntimeError("Original frontend inventory is restricted to the canonical Azure Linux control host")
    spec=inputs.PublicClipSpec(args.episode,args.frames,args.camera_name,args.height,args.width)
    pins=inventory(ROOT,code,spec,args.producer_revision,args.producer_script_sha256)
    json.dump(pins,sys.stdout,allow_nan=False,separators=(",",":"));sys.stdout.write("\n")


if __name__=="__main__":main()

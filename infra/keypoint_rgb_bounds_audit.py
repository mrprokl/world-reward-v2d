"""Read-only D96 native bound diagnostics, without forward, fit or truth IO.

PASS means the diagnostic completed. Violating predictions are retained, not
clipped, relabelled as valid, or adopted. The original failed fit is immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

BASE = "validation/keypoint_rgb_v1"
OUT = "results/keypoint-rgb-bounds-audit-v2"
STAGE = "public_keypoint_rgb_native_bounds_readonly_audit_v2"
BUDGET = 60
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
MODEL_BYTES = 696110248
BASELINE_SHA = "8ebfce153ea5ff708578c155d60adb97e737b672e2c2df56df38c635fbbc80c3"
BASELINE_REV = "f5c78f7abfec3c6234bbbee7f7dad70736836ae7"
BASELINE_SCRIPT = "4950979a0dfbeb6759f0a7149a9d79ca61fdb13c96cf09c5349fca1f6f73459b"
MANIFEST_SHA = "082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d"
FAILED_SHA = "aacb82e7c48b5b28ec26f1eb3bb88194c76fbb3c0ce07cc54d6032adb25412d2"
FAILED_REV = "52f35c0b0df9a1448adf16c1cff796007fb41198"
FAILED_SCRIPT = "d5146d1f89a0fbbd0e4614abd8896a9e13679bd4307535a3cea0b2f46a84583b"
PREVIOUS = "results/keypoint-rgb-bounds-audit-v1/report.json"
PREVIOUS_SHA = "1a566a9c1e8ac31cce6654372b79ee2dad00fb47a92d24134cafa915c886244f"
PREVIOUS_REV = "d232abb7bc306860bdda254d82a768a257ba2fc0"
PREVIOUS_SCRIPT = "a6b0542eea2d7b8d717ef9fe4eb0f5610f37a5d0c81c36d45c958476d204ebd8"


def regular(path, digest=None, size=None, *, require_mode_immutable=True):
    path=Path(path)
    if path.resolve()!=path.absolute() or not path.is_file() or (require_mode_immutable and path.stat().st_mode & 0o222):
        raise ValueError("Canonical read-only regular input required")
    with path.open("rb") as stream: actual=hashlib.file_digest(stream,"sha256").hexdigest()
    row=dict(path=str(path),sha256=actual,bytes=path.stat().st_size)
    if (digest is not None and actual!=digest) or (size is not None and row["bytes"]!=size):
        raise ValueError("Frozen input SHA/bytes changed")
    return row


def readonly_model_mount(text, model_path):
    """Require the exact file's VFS bind mount to be read-only, not its parent."""
    target=str(Path(model_path));matches=[]
    for line in text.splitlines():
        fields=line.split();separator=fields.index("-") if "-" in fields else -1
        if separator<6 or len(fields)<separator+4: raise ValueError("Malformed kernel mountinfo")
        mount=re.sub(r"\\(040|011|012|134)",lambda m:chr(int(m[1],8)),fields[4])
        if re.search(r"\\[0-9]",mount): raise ValueError("Unsupported kernel mount path escape")
        if mount==target:
            options=fields[5].split(",")
            if "ro" not in options or "rw" in options: raise ValueError("Exact model file mount must be read-only")
            matches.append(dict(mount_point=mount,mount_options=options,super_options=fields[separator+3].split(",")))
    if len(matches)!=1: raise ValueError("One exact read-only model file mount required; parent mounts insufficient")
    return matches[0]


def previous_failure(root):
    path=Path(root)/PREVIOUS;receipt=regular(path,PREVIOUS_SHA);row=json.loads(path.read_text())
    fields(row,dict(stage="public_keypoint_rgb_native_bounds_readonly_audit",status="fail",phase="frozen_input_integrity",
        producer_revision=PREVIOUS_REV,script_sha256=PREVIOUS_SCRIPT,image_id=IMAGE,error_type="ValueError",
        error="Canonical read-only regular input required",model_forward_calls=0,optimizer_updates=0,private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,oracle_modes=[],original_inputs_rehashed=False,records=[]))
    return receipt


def fields(data, expected):
    if not isinstance(data,dict) or any(type(data.get(k)) is not type(v) or data[k]!=v for k,v in expected.items()):
        raise ValueError("Exact original producer fields required")


def array(value, shape, dtype):
    if np.ma.isMaskedArray(value): raise ValueError("Hidden array validity forbidden")
    result=np.asarray(value)
    if result.shape!=shape or result.dtype!=np.dtype(dtype) or not np.isfinite(result).all():
        raise ValueError("Exact finite native array contract required")
    return result


def finite_bound(value):
    return float(value) if np.isfinite(value) else ("+infinity" if value>0 else "-infinity")


def audit_controls(controls, shape, euler, limits, names):
    """All249 native columns; exact violations are evidence, not fatal errors."""
    c=array(controls,(204,),"float32"); s=array(shape,(45,),"float32"); e=array(euler,(3,),"float32")
    bounds=np.asarray(limits)
    if (np.ma.isMaskedArray(limits) or bounds.shape!=(249,2) or bounds.dtype.kind!="f" or np.isnan(bounds).any()
            or np.any(bounds[:,0]>bounds[:,1]) or np.isposinf(bounds[:,0]).any() or np.isneginf(bounds[:,1]).any()
            or not isinstance(names,list) or len(names)!=249
            or len(set(names))!=249 or any(not isinstance(n,str) or not n for n in names)):
        raise ValueError("Actual unique249 names and ordered nonNaN bounds required")
    full=np.r_[c,s].astype(np.float64)
    violations=[]
    for i in np.flatnonzero((full<bounds[:,0]) | (full>bounds[:,1])):
        value=float(full[i]);lo=float(bounds[i,0]);hi=float(bounds[i,1]);lower=value<lo
        violations.append(dict(index=int(i),name=names[i],value=value,lower=finite_bound(lo),upper=finite_bound(hi),
            side="lower" if lower else "upper",excess=float(lo-value if lower else value-hi)))
    locked=[int(i) for i in range(136,204) if bounds[i,0]==bounds[i,1]]
    neutral=bool(np.all(bounds[:204,0]<=0) and np.all(bounds[:204,1]>=0))
    return dict(violations=violations,violation_count=len(violations),within_native_bounds=not violations,
        root_Euler_byte_equal=c[3:6].tobytes()==e.tobytes(),root_controls=c[3:6].tolist(),input_Euler=e.tolist(),
        root_translation_zero=bool(np.all(c[:3]==0)),locked_scale_columns=locked,
        locked_scale_values=[dict(index=i,name=names[i],value=float(full[i]),locked_value=finite_bound(bounds[i,0]),
                                  matches=bool(full[i]==bounds[i,0])) for i in locked],
        neutral_zero_204_permitted_by_metadata=neutral)


def validate_failed(report):
    fields(report,dict(stage="public_keypoint_rgb_native_root_refit",status="fail",phase="native_root_optimization",
        producer_revision=FAILED_REV,script_sha256=FAILED_SCRIPT,image_id=IMAGE,network="none",device="cuda",
        error_type="ValueError",error="Complete249 native bounds/rootEuler mapping violated; no clipping",
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        object_proxies_frozen_before_fit=True,all_candidates_frozen=False,native_arrays_verified=False,
        quality_verified=False,accuracy_verified=False,adoption_authorized=False))
    fields(report.get("counters"),dict(proxy_rasters=15,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0))
    rows=report.get("fit_records")
    if (not isinstance(rows,list) or len(rows)!=1 or rows[0].get("file")!="clip_00_frame_000.png"
            or rows[0].get("evaluated_losses")!=[] or rows[0].get("native_forward_calls")!=0 or rows[0].get("adam_updates")!=0
            or report.get("candidate_outputs")!=[] or len(report.get("proxy_outputs",[]))!=15):
        raise ValueError("Exact stopped first-state failure evidence required")


def read_inputs(root):
    base=root/BASE; bp=base/"baseline_v1/report.json"; fp=base/"root_fit_v1/report.json"; mp=base/"inputs/manifest.json"
    receipts=[regular(bp,BASELINE_SHA),regular(fp,FAILED_SHA),regular(mp,MANIFEST_SHA,2199)]
    baseline,failed,manifest=(json.loads(p.read_text()) for p in (bp,fp,mp));validate_failed(failed)
    fields(baseline,dict(stage="public_keypoint_rgb_body_depth_first_rgb_identity_baseline",status="pass",phase="complete",
        producer_revision=BASELINE_REV,script_sha256=BASELINE_SCRIPT,frames=15,all_cases_retained=True,private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        raw_frozen_before_shared=True,paired_frozen_before_reference=True,conversion_fidelity_verified=True))
    if (set(manifest)!={"schema","images"} or manifest["schema"]!="world-reward-keypoint-rgb-v1"
            or not isinstance(manifest["images"],list) or len(manifest["images"])!=15): raise ValueError("Original RGB-only manifest required")
    source=root/"jobs"/FAILED_REV/"run_keypoint_rgb_fit"/"code"/"infra"/"keypoint_rgb_fit.py"
    receipts.append(regular(source,FAILED_SCRIPT)); raws=[];pairs=[]
    for folder,key,target in (("raw","raw_outputs",raws),("paired","paired_outputs",pairs)):
        rows=baseline.get(key)
        if not isinstance(rows,list) or len(rows)!=15: raise ValueError("Every original baseline artifact required")
        expected=set()
        for i,(row,item) in enumerate(zip(rows,manifest["images"])):
            clip,frame=divmod(i,5);name=f"clip_{clip:02d}_frame_{frame:03d}"
            if (set(item)!={"file","sha256","width","height"} or item["file"]!=name+".png"
                    or type(item["width"])is not int or type(item["height"])is not int or (item["width"],item["height"])!=(1024,768)):
                raise ValueError("All15 original RGB metadata required")
            fields(row,dict(file=name+".png",clip_index=clip,frame_index=frame,rgb_sha256=item["sha256"],artifact=folder+"/"+name+".npz"))
            path=bp.parent/row["artifact"];receipts.append(regular(path,row["sha256"],row["bytes"]));expected.add(path.name)
            with np.load(path,allow_pickle=False)as archive:
                keys=("mhr_model_params","shape_params","global_rot") if folder=="raw" else ("raw_model_controls","shared_model_controls","raw_shape_params","shared_shape_params","shared_scale_params")
                data={k:archive[k] for k in keys}
                for k in ("clip_index","frame_index"):
                    if int(array(archive[k],(),"int64"))!={"clip_index":clip,"frame_index":frame}[k]:raise ValueError("NPZ original frame identity differs")
            target.append(data)
        if {p.name for p in (bp.parent/folder).iterdir()}!=expected:raise ValueError("Exact15 baseline file inventory required")
    if {p.name for p in bp.parent.iterdir()}!={"raw","paired","report.json"}:raise ValueError("Exact baseline directory required")
    proxy_names=set()
    for i,row in enumerate(failed["proxy_outputs"]):
        name=f"clip_{i//5:02d}_frame_{i%5:03d}.npz"
        if row.get("artifact")!="proxies/"+name or row.get("rgb_sha256")!=manifest["images"][i]["sha256"]:raise ValueError("Original15 frozen proxies required")
        path=fp.parent/row["artifact"];receipts.append(regular(path,row["sha256"],row["bytes"]));proxy_names.add(name)
    if ({p.name for p in (fp.parent/"proxies").iterdir()}!=proxy_names or any((fp.parent/"candidates").iterdir())
            or {p.name for p in fp.parent.iterdir()}!={"report.json","proxies","candidates"}):raise ValueError("Original failure/proxies must be unmodified")
    for i,(raw,pair)in enumerate(zip(raws,pairs)):
        if (array(pair["raw_model_controls"],(204,),"float32").tobytes()!=array(raw["mhr_model_params"],(204,),"float32").tobytes()
                or array(pair["raw_shape_params"],(45,),"float32").tobytes()!=array(raw["shape_params"],(45,),"float32").tobytes()
                or array(pair["shared_shape_params"],(45,),"float32").tobytes()!=raws[i//5*5]["shape_params"].tobytes()
                or pair["shared_model_controls"][:136].tobytes()!=raw["mhr_model_params"][:136].tobytes()
                or array(pair["shared_model_controls"],(204,),"float32")[136:].tobytes()!=pairs[i//5*5]["shared_model_controls"][136:].tobytes()
                or array(pair["shared_scale_params"],(28,),"float32").tobytes()!=pairs[i//5*5]["shared_scale_params"].tobytes()):
            raise ValueError("Raw/shared lineage or clip-constant firstRGB identity changed")
    return raws,pairs,receipts


def perform(root,report,persist):
    previous=previous_failure(root)
    raws,pairs,receipts=read_inputs(root);model_path=root/"weights/mhr/mhr_model.pt"
    mount=readonly_model_mount(Path("/proc/self/mountinfo").read_text(),model_path)
    model_receipt=regular(model_path,MODEL_SHA,MODEL_BYTES,require_mode_immutable=False)
    receipts.extend([previous,regular(Path(__file__))])
    report.update(phase="CPU_metadata_load",input_files=receipts,model_file=model_receipt,model_mount=mount,
        model_host_write_bits=int(model_path.stat().st_mode & 0o222),model_exact_mount_readonly_verified=True);persist()
    if "torch"in sys.modules:raise ValueError("Standalone CPU metadata runtime required")
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    model=torch.jit.load(str(model_path),map_location="cpu").float().eval()
    limits=model.get_parameter_limits().detach().cpu().numpy();names=list(model.get_parameter_names())
    report.update(torch_version=str(torch.__version__),model_sha256=MODEL_SHA,model_forward_calls=0,
        metadata_methods=["get_parameter_limits","get_parameter_names"],parameter_count=len(names),phase="native_bounds_diagnostics");persist()
    for i,(raw,pair)in enumerate(zip(raws,pairs)):
        row=dict(clip_index=i//5,frame_index=i%5,raw=audit_controls(raw["mhr_model_params"],raw["shape_params"],raw["global_rot"],limits,names),
            shared=audit_controls(pair["shared_model_controls"],pair["shared_shape_params"],raw["global_rot"],limits,names))
        report["records"].append(row);persist()
    for receipt in receipts:regular(receipt["path"],receipt["sha256"],receipt["bytes"])
    if readonly_model_mount(Path("/proc/self/mountinfo").read_text(),model_path)!=mount:
        raise ValueError("Exact read-only model mount changed")
    regular(model_path,MODEL_SHA,MODEL_BYTES,require_mode_immutable=False)
    report.update(status="pass",phase="complete",frames=15,all_cases_retained=True,original_inputs_rehashed=True,
        raw_violating_frames=sum(bool(r["raw"]["violations"])for r in report["records"]),
        shared_violating_frames=sum(bool(r["shared"]["violations"])for r in report["records"]),
        raw_root_mapping_mismatch_frames=sum(not r["raw"]["root_Euler_byte_equal"]for r in report["records"]),
        shared_root_mapping_mismatch_frames=sum(not r["shared"]["root_Euler_byte_equal"]for r in report["records"]))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));out=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or out.resolve()!=out.absolute()
            or os.geteuid()!=1000 or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}
            or os.environ.get("WR_IMAGE_ID")!=IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES")!=""
            or not re.fullmatch("[0-9a-f]{40}",revision) or not out.is_dir() or any(out.iterdir())):
        raise ValueError("Fresh reserved offline CPU metadata audit required")
    report=dict(stage=STAGE,status="fail",phase="frozen_input_integrity",producer_revision=revision,image_id=IMAGE,
        network="none",device="cpu",budget_seconds=BUDGET,failed_fit_report_sha256=FAILED_SHA,records=[],
        previous_failed_audit_sha256=PREVIOUS_SHA,previous_failure_rewritten=False,model_exact_mount_readonly_verified=False,
        model_forward_calls=0,optimizer_updates=0,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,
        hand_labeled_test=False,oracle_modes=[],accuracy_verified=False,adoption_authorized=False,bounds_relaxed=False,
        predictions_modified=False,original_failure_rewritten=False,actual_known_failed_fit_native_head_calls=1,
        failed_fit_completed_valid_head_calls=0,failed_head_count_basis="returned native head reached post-forward bounds guard before completed counter",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_inputs_rehashed=False,all_cases_retained=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole CPU bounds audit exceeded60s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();perform(root,report,persist)
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()

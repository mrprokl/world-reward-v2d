"""H102 public96 snapshot and frame-zero shared-identity native ABI gate.

No new image inference, quality labels, optimizer or conversion fit. The output
is a new constrained initializer, not recovered historical CARI geometry.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import re
import signal
import sys
import time

import numpy as np
import cari96_inputs as inputs
import body_smoke as body
from world_reward.data import sha256

BASE = "validation/cari96_public_v1"
STAGE, BUDGET, CHUNK = "public_cari96_shared_initializer_native_abi", 180, 16
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
LAYER_SHA = "a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23"
REFERENCE_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
CONVERTER_SHA = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"


def shared_parameters(initializer):
    inputs.validate_initializer(initializer)
    selected = {k: initializer[k][:inputs.FRAMES].copy() for k in inputs.PARAMETER_DIMS}
    for k in ("mhr_shape", "mhr_scale"):
        selected[k] = np.repeat(initializer[k][:1], inputs.FRAMES, axis=0)
    return selected


def shared_metadata(metadata):
    result = copy.deepcopy(metadata)
    historical = {k: result.pop(k) for k in ("native_roundtrip_max_error_m", "translation_once_max_error_m", "projection_max_error_px") if k in result}
    result.update(historical_original_initializer_checks=historical,
        identity_policy="native_original_frame_zero_before_caches_no_quality_selection", human_identity_clip_constant=True,
        original_geometry_recovered=False, source_frames=inputs.SOURCE_FRAMES, original_frame_indices=list(range(inputs.FRAMES)),
        projection_reverified_after_identity_change=False, submission_eligible=False)
    return result


def attempted(report, name, persist):
    report[name + "_attempts"] += 1
    persist()


def checked_geometry(value, count, vertices):
    if (type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != (count,vertices,3)
            or not np.isfinite(value).all() or np.any(value[...,2] <= 0)):
        raise ValueError("Full positive finite native camera geometry required")
    return value


def residuals(target, actual, units):
    if units not in ("m", "mm") or actual.shape != target.shape or actual.dtype not in (np.float32,np.float64) or not np.isfinite(actual).all():
        raise ValueError("Complete finite independent replay and explicit units required")
    error=np.linalg.norm(actual.astype(np.float64)*(1000. if units=="m" else 1.)-target.astype(np.float64)*1000.,axis=-1)
    return dict(per_frame_mean_mm=error.mean(1).tolist(),max_point_mm=float(error.max()))


def binding(path,digest):
    inputs.regular(path)
    if sha256(path)!=digest:raise ValueError("Actual source/model binding differs")
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size)


def run(root,out,code,report,persist):
    pins_path=code/"configs/cari96_input_pins.json";pin_id=inputs.identity(pins_path)
    pins=json.loads(pins_path.read_text());snapshot=inputs.prepare_snapshot(root,out/"inputs",pins)
    original=snapshot["initializer"];selected=shared_parameters(original)
    original_bytes={k:original[k].tobytes() for k in inputs.PARAMETER_DIMS}
    selected_bytes={k:selected[k].tobytes() for k in inputs.PARAMETER_DIMS}
    report.update(phase="snapshot_complete",snapshot_manifest_sha256=sha256(snapshot["snapshot_manifest"]),input_pins=pin_id);persist()
    assets,hashes=body._body_assets(root);source=body._source_identity(root)
    prior=json.loads(snapshot["initializer_report"].read_text())
    body_report=json.loads((root/inputs.DEPENDENCIES["body"][0]).read_text())
    if source!=prior["inference_source_identity"] or hashes!=body_report["body_assets"] or (assets/"mhr_buffers.pt").exists():
        raise ValueError("Exact original native decoder source/checkpoint assets required")
    vendor=root/"vendor/video_to_data";body._pinned_checkout(vendor,body.UPSTREAM_REVISION)
    native=vendor/"reconstruction/modules/v2d_cari4d/lib/cari4d"
    tool=root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py";model=root/"weights/mhr/mhr_model.pt"
    frozen=[binding(native/"lib_mhr/mhr_layer.py",LAYER_SHA),binding(tool,CONVERTER_SHA),binding(model,REFERENCE_SHA)]
    helpers={str(p.relative_to(code)):inputs.identity(p) for p in (Path(__file__),code/"infra/cari96_inputs.py",code/"infra/body_smoke.py",code/"infra/run_cari96_prepare.sh")}
    report.update(source_bindings=frozen,source_helpers=helpers);persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise ValueError("Fresh deterministic native runtime required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4":raise ValueError("Pinned native H100 CUDA required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    report["runtime"] = dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        deterministic_algorithms=True,warn_only=False,jit_optimized_execution=False,tf32=False,cudnn_benchmark=False,seed=0,chunk=CHUNK)
    sys.path[:0]=[str(native),"/workspace/v2d_sam3d_body/lib"]
    from lib_mhr.mhr_layer import MHRLayer
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),checkpoint_path=assets/"model.ckpt",
        buffer_path=out/"never_use_unverified_buffer.pt",mhr_model_path=assets/"assets/mhr_model.pt",device="cuda")
    if layer.decoder_identity()!=prior["decoder_identity"]:raise ValueError("Original decoder identity differs")
    target=np.empty((inputs.FRAMES,18439,3),np.float32);joints=np.empty((inputs.FRAMES,127,3),np.float32);kp=np.empty((inputs.FRAMES,70,3),np.float32)
    controls=np.empty((inputs.FRAMES,204),np.float32)
    tensors=lambda s:{k:torch.tensor(v[s].copy(),device="cuda",dtype=torch.float32) for k,v in selected.items()}
    report.update(phase="shared_native_decode",decoder_identity=prior["decoder_identity"],body_assets=hashes,inference_source_identity=source);persist()
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        for start in range(0,inputs.FRAMES,CHUNK):
            s=slice(start,min(start+CHUNK,inputs.FRAMES));params=tensors(s)
            attempted(report,"native_geometry",persist)
            decoded=layer.mhr_forward(params);report["native_geometry_calls"]+=1
            target[s]=checked_geometry(decoded.vertices.cpu().numpy(),s.stop-s.start,18439)
            joints[s]=checked_geometry(decoded.joints.cpu().numpy(),s.stop-s.start,127)
            kp[s]=checked_geometry(decoded.keypoints.cpu().numpy(),s.stop-s.start,70)
            faces=decoded.faces.cpu().numpy()
            if faces.shape!=(36874,3) or hashlib.sha256(faces.astype("<i4").tobytes()).hexdigest()!=FACE_SHA:raise ValueError("Exact native topology required")
            backend=layer.backend;context=backend._vertices_context(params,detach_fixed=False)
            trans,pose,shape,scale=backend._mutable_vertices_inputs(params,context)
            if context.head.enable_hand_model:raise ValueError("Original body-only native head required")
            attempted(report,"native_direct",persist)
            dv,dc=context.head.mhr_forward(global_trans=trans*context.flip,global_rot=context.global_rot,body_pose_params=pose,
                hand_pose_params=context.hand,scale_params=scale,shape_params=shape,expr_params=context.face,return_model_params=True)
            report["native_direct_calls"]+=1;torch.cuda.synchronize()
            direct=(dv*context.flip).cpu().numpy();controls[s]=dc.cpu().numpy()
            if residuals(target[s],direct,"m")["max_point_mm"]>.01:raise ValueError("Native direct geometry exceeds .01mm point fidelity")
            if not torch.equal(dc[:,136:],context.head.scale_mean[None,:]+scale@context.head.scale_comps):raise ValueError("Original PCA expansion differs")
            persist()
    if not np.isfinite(controls).all() or any(r.tobytes()!=controls[0,136:].tobytes() for r in controls[:,136:]):raise ValueError("Byte-constant expanded native68 required")
    candidate=copy.deepcopy(original);candidate.update(selected,frames=original["frames"][:inputs.FRAMES],mhr_joints=joints,mhr_keypoints=kp)
    candidate["metadata"]=shared_metadata(original["metadata"])
    path=out/"shared_initializer.pkl"
    with path.open("xb") as stream:pickle.dump(candidate,stream,protocol=4)
    path.chmod(0o444)
    with (out/"direct_parameters.npz").open("xb") as stream:np.savez_compressed(stream,frame_index=np.arange(inputs.FRAMES,dtype=np.int64),pose=controls[:,:136],scales=controls[0,136:],shape=selected["mhr_shape"][0])
    (out/"direct_parameters.npz").chmod(0o444)
    with (out/"target.npy").open("xb") as stream:np.save(stream,target,allow_pickle=False)
    (out/"target.npy").chmod(0o444)
    with path.open("rb") as stream:reread=pickle.load(stream)
    if (any(reread[k].tobytes()!=candidate[k].tobytes() for k in (*inputs.PARAMETER_DIMS,"mhr_joints","mhr_keypoints"))
            or reread["frames"]!=candidate["frames"] or reread["metadata"]!=candidate["metadata"]):raise ValueError("Stored shared initializer differs")
    saved=np.load(out/"target.npy",mmap_mode="r",allow_pickle=False)
    with np.load(out/"direct_parameters.npz",allow_pickle=False) as stream:stored={k:stream[k] for k in stream.files}
    if (set(stored)!={"frame_index","pose","scales","shape"} or stored["frame_index"].dtype!=np.int64
            or not np.array_equal(stored["frame_index"],np.arange(inputs.FRAMES,dtype=np.int64)) or saved.flags.writeable
            or not np.array_equal(saved,target) or not np.array_equal(stored["pose"],controls[:,:136])
            or not np.array_equal(stored["scales"],controls[0,136:]) or not np.array_equal(stored["shape"],selected["mhr_shape"][0])):raise ValueError("Frozen direct artifacts changed")
    frozen_outputs={str(p.relative_to(out)):inputs.identity(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="report.json"}
    report.update(phase="stored_initializer_native_replay");persist()
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        for start in range(0,inputs.FRAMES,CHUNK):
            s=slice(start,min(start+CHUNK,inputs.FRAMES))
            attempted(report,"native_replay",persist)
            recovered=layer.mhr_forward({k:torch.tensor(reread[k][s].copy(),device="cuda",dtype=torch.float32) for k in inputs.PARAMETER_DIMS})
            report["native_replay_calls"]+=1;torch.cuda.synchronize()
            for actual,expected in ((recovered.vertices,saved[s]),(recovered.joints,reread["mhr_joints"][s]),(recovered.keypoints,reread["mhr_keypoints"][s])):
                if residuals(expected,actual.cpu().numpy(),"m")["max_point_mm"]>.01:raise ValueError("Stored initializer native replay exceeds .01mm point fidelity")
            persist()
    report.update(phase="official_reference_replay",predictions_frozen_before_reference=True);persist()
    spec=importlib.util.spec_from_file_location("cari96_reference",tool);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=module.MHR(str(model),"cuda",chunk=CHUNK,precision="float32")
    if (Path(module.MHR.run.__code__.co_filename).resolve()!=tool or reference.mdtype!=torch.float32 or reference.dtype!=torch.float64
            or reference.chunk!=CHUNK or reference.device.type!="cuda"):raise ValueError("Exact official FP32 forward/FP64 residual reference required")
    report["reference_settings"]=dict(precision="float32",residual_dtype="float64",chunk=CHUNK,device="cuda",mean_point_gate_mm=2.,native_max_point_gate_mm=.01)
    error=[];maximum=0.
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        z=torch.tensor(np.r_[stored["scales"],stored["shape"]][None],device="cuda",dtype=torch.float64)
        for start in range(0,inputs.FRAMES,CHUNK):
            s=slice(start,min(start+CHUNK,inputs.FRAMES));attempted(report,"reference",persist)
            v,_=reference.run(torch.tensor(stored["pose"][s],device="cuda",dtype=torch.float64),z)
            report["reference_calls"]+=1;torch.cuda.synchronize();values=residuals(saved[s],v.cpu().numpy(),"mm")
            error.extend(values["per_frame_mean_mm"]);maximum=max(maximum,values["max_point_mm"])
            if max(values["per_frame_mean_mm"])>2.:raise ValueError("A frame exceeds unchanged2mm reference fidelity")
            persist()
    if (any(original[k].tobytes()!=original_bytes[k] for k in original_bytes) or any(selected[k].tobytes()!=selected_bytes[k] for k in selected_bytes)
            or inputs.identity(pins_path)!=pin_id
            or {p:inputs.identity(root/p) for p in pins["source_files"]}!=pins["source_files"]
            or body._source_identity(root)!=source or body._body_assets(root)[1]!=hashes):raise ValueError("Original sources/parameters changed")
    for row in frozen:binding(Path(row["path"]),row["sha256"])
    if ({p:inputs.identity(code/p) for p in helpers}!=helpers or {p:inputs.identity(out/p) for p in frozen_outputs}!=frozen_outputs):raise ValueError("Frozen helper/output artifacts changed")
    if any(report[k+suffix]!=6 for k in ("native_geometry","native_direct","native_replay","reference") for suffix in ("_calls","_attempts")):raise ValueError("All96 native/direct/replay/reference coverage required")
    report.update(status="pass",phase="complete",frames=inputs.FRAMES,reference_per_frame_mean_mm=error,
        reference_max_point_mm=maximum,output_files=frozen_outputs,source_helpers_rehashed=True,frozen_outputs_rehashed_after_reference=True,
        shared_identity_verified=True,original_initializer_unchanged=True,source_inputs_assets_rehashed=True,stored_initializer_native_replay_verified=True,
        shared_initializer_sha256=sha256(path),direct_parameters_sha256=sha256(out/"direct_parameters.npz"),target_sha256=sha256(out/"target.npy"))


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);rev=os.environ["WR_CODE_REVISION"];out=root/BASE
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or os.environ["WR_IMAGE_ID"]!=IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",rev) or not out.is_dir() or any(out.iterdir()) or out.resolve()!=out.absolute()):raise ValueError("Fresh source-bound offline Azure output required")
    report=dict(stage=STAGE,status="fail",phase="public_input_audit",producer_revision=rev,script_sha256=sha256(Path(__file__)),image_id=IMAGE,
        input_track="track_1",ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",budget_seconds=BUDGET,
        learned_inference_calls=0,optimizer_calls=0,quality_verified=False,adoption_authorized=False,submission_produced=False,
        native_geometry_calls=0,native_direct_calls=0,native_replay_calls=0,reference_calls=0,
        native_geometry_attempts=0,native_direct_attempts=0,native_replay_attempts=0,reference_attempts=0)
    receipt=out/"report.json";started=time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("H102 public snapshot/native ABI exceeded180s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,report,persist)
        except BaseException as e:report.update(status="fail",error_type=type(e).__name__,error=str(e));raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o444)


if __name__=="__main__":main()

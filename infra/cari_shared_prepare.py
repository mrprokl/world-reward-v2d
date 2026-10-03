"""Full public native shared initializer, direct controls and frozen replay.

Identity is fixed before the first native geometry call. Original full HDF5,
camera, object mesh and poses are only referenced and rehashed, never copied.
No learned RGB inference, optimizer, inverse fit, quality labels or GT access.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
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
import body_smoke as body
import cari_clip_inputs as inputs
import cari96_prepare as geometry
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, share_first_frame_identity, shared_initializer_metadata, validate_native_parameters
from world_reward.timeline import finite_chunks

STAGE = "world_reward_native_cari_shared_initializer_full_video"
BUDGET, CHUNK = 600, 16
IMAGE = geometry.IMAGE
CALLS = ("native_geometry", "native_direct", "native_replay", "reference")
OUTPUTS = {"shared_initializer.pkl", "direct_parameters.npz", "target.npy"}
INITIALIZER_KEYS = {*NATIVE_PARAMETER_DIMS, "body_model", "frames", "kids", "metadata", "mhr_joints", "mhr_keypoints"}


def output_relative(episode):
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Explicit Track1 episode in 0..29 required")
    return f"outputs/episode_{episode:06d}/cari_shared_prepare_v1"


def parser():
    class OneEpisode(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if getattr(namespace,self.dest,None) is not None:
                parser.error("Exactly one explicit episode is required")
            setattr(namespace,self.dest,value)
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--episode", type=int, required=True, choices=range(30), action=OneEpisode)
    return result


def array_bytes(values):
    return {key: (value.shape, value.dtype.str, value.tobytes(order="C")) for key, value in values.items()}


def checked_array(value, shape, dtype=np.float32):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype)
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError("Exact unmasked finite native array required")
    return value


def checked_faces(value, *, vertices=18439, count=36874, digest=geometry.FACE_SHA):
    checked_array(value, (count, 3), np.int64)
    if (np.any(value < 0) or np.any(value >= vertices)
            or hashlib.sha256(value.astype("<i4").tobytes()).hexdigest() != digest):
        raise ValueError("Exact unchanged native int64 topology required")
    return value


def call(report, name, function, persist):
    report[name + "_attempts"] += 1
    persist()
    value = function()
    report[name + "_returns"] += 1
    return value


def validated(report, name, persist):
    report[name + "_validated"] += 1
    persist()


def owned_call(callback, params):
    before = array_bytes(params)
    value = callback(params)
    if array_bytes(params) != before:
        raise ValueError("Native callback modified its owned parameter inputs")
    return value


def freeze(out, candidate, controls, target):
    """Store only new original-timeline parameters and freshly decoded geometry."""
    path = out / "shared_initializer.pkl"
    with path.open("xb") as stream:
        pickle.dump(candidate, stream, protocol=4)
    path.chmod(0o444)
    values = dict(frame_index=np.arange(len(target), dtype=np.int64), pose=controls[:, :136].copy(),
                  scales=controls[0, 136:].copy(), shape=candidate["mhr_shape"][0].copy())
    with (out / "direct_parameters.npz").open("xb") as stream:
        np.savez_compressed(stream, **values)
    (out / "direct_parameters.npz").chmod(0o444)
    with (out / "target.npy").open("xb") as stream:
        np.save(stream, target, allow_pickle=False)
    (out / "target.npy").chmod(0o444)
    with path.open("rb") as stream:
        reread = pickle.load(stream)
    if (set(reread) != set(candidate) or reread["frames"] != candidate["frames"] or reread["kids"] != candidate["kids"]
            or reread["body_model"] != candidate["body_model"] or reread["metadata"] != candidate["metadata"]
            or array_bytes({key:reread[key] for key in (*NATIVE_PARAMETER_DIMS,"mhr_joints","mhr_keypoints")})
            != array_bytes({key:candidate[key] for key in (*NATIVE_PARAMETER_DIMS,"mhr_joints","mhr_keypoints")})):
        raise ValueError("Stored shared initializer bytes or metadata changed")
    with np.load(out / "direct_parameters.npz", allow_pickle=False) as stream:
        stored = {key:stream[key] for key in stream.files}
    saved = np.load(out / "target.npy", mmap_mode="r", allow_pickle=False)
    if (set(stored) != set(values) or array_bytes(stored) != array_bytes(values)
            or saved.flags.writeable or saved.dtype != target.dtype or saved.shape != target.shape
            or saved.tobytes(order="C") != target.tobytes(order="C")):
        raise ValueError("Frozen direct controls or target bytes changed")
    frozen = {name:inputs.identity(out/name) for name in sorted(OUTPUTS)}
    return reread, stored, saved, frozen


def prepare_geometry(original, count, out, native_decode, native_direct, reference_decode, report, persist,
                     *, vertices=18439, joint_count=127, keypoint_count=70, face_count=36874, face_sha=geometry.FACE_SHA,
                     preselected=None):
    """Callback-testable actual full-N execution; runtime binds each callback.

    The production callbacks are unchanged native model forwards. All evidence
    remains per actual call and last-batch count. Tiny synthetic tests supply
    smaller topology explicitly; no runtime option exposes those test sizes.
    """
    if type(original) is not dict or set(original) != INITIALIZER_KEYS:
        raise ValueError("Exact canonical initializer only; no stale geometry extras")
    if not out.is_dir() or any(path.name!="report.json" for path in out.iterdir()):
        raise ValueError("Fresh shared output without prior payloads required before any decode")
    chunks = finite_chunks(count, CHUNK)
    original_params = {key:original[key] for key in NATIVE_PARAMETER_DIMS}
    original_bytes = array_bytes(original_params)
    expected = share_first_frame_identity(original_params,count)
    expected_metadata = shared_initializer_metadata(original["metadata"],count)
    if preselected is None:
        selected,metadata=expected,expected_metadata
    else:
        if (type(preselected) is not tuple or len(preselected)!=2
                or array_bytes(preselected[0])!=array_bytes(expected) or preselected[1]!=expected_metadata):
            raise ValueError("Before-model shared identity differs from the unchanged first-frame policy")
        selected,metadata=preselected
    selected_bytes = array_bytes(selected)
    original_meta = copy.deepcopy(original["metadata"])
    original_timeline = copy.deepcopy({key:original[key] for key in ("body_model","frames","kids")})
    original_cached = array_bytes({key:original[key] for key in ("mhr_joints","mhr_keypoints")})
    if original["frames"] != [f"{index:06d}" for index in range(count)] or original["kids"] != [0] or original["body_model"] != "mhr":
        raise ValueError("Exact complete original timeline/body identity required")
    target = np.empty((count,vertices,3),np.float32)
    joints = np.empty((count,joint_count,3),np.float32)
    keypoints = np.empty((count,keypoint_count,3),np.float32)
    controls = np.empty((count,204),np.float32)
    native_direct_max = 0.
    report.update(phase="shared_native_decode",identity_fixed_before_first_decode=True,
                  original_frame_indices=list(range(count)),chunk_counts=[s.stop-s.start for s in chunks])
    persist()
    params = lambda selection: {key:value[selection].copy() for key,value in selected.items()}
    for selection in chunks:
        actual_count = selection.stop - selection.start
        decoded = call(report,"native_geometry",lambda:owned_call(native_decode,params(selection)),persist)
        target[selection] = geometry.checked_geometry(decoded["vertices"],actual_count,vertices)
        joints[selection] = geometry.checked_geometry(decoded["joints"],actual_count,joint_count)
        keypoints[selection] = geometry.checked_geometry(decoded["keypoints"],actual_count,keypoint_count)
        faces = checked_faces(decoded["faces"],vertices=vertices,count=face_count,digest=face_sha)
        validated(report,"native_geometry",persist)
        direct, dc, pca = call(report,"native_direct",lambda:owned_call(native_direct,params(selection)),persist)
        checked_array(dc,(actual_count,204));checked_array(pca,(actual_count,68))
        error = geometry.residuals(target[selection],direct,"m")["max_point_mm"]
        native_direct_max = max(native_direct_max,error)
        if error > .01 or dc[:,136:].tobytes()!=pca.tobytes():
            raise ValueError("Native direct geometry/PCA failed unchanged .01mm point fidelity")
        controls[selection] = dc
        validated(report,"native_direct",persist)
    if any(row.tobytes()!=controls[0,136:].tobytes() for row in controls[:,136:]):
        raise ValueError("Expanded native68 scale controls must remain byte-constant")
    candidate = {key:copy.deepcopy(original[key]) for key in ("body_model","frames","kids")}
    candidate.update(selected,mhr_joints=joints,mhr_keypoints=keypoints,metadata=metadata)
    candidate["metadata"].update(mhr_geometry_forward_verified=True,shared_native_direct_geometry_verified=True,
        native_direct_max_point_error_mm=native_direct_max,joint_keypoint_redecode_required=False,
        shared_joint_keypoint_native_redecoded=True)
    reread,stored,saved,frozen = freeze(out,candidate,controls,target)
    validate_native_parameters({key:reread[key] for key in NATIVE_PARAMETER_DIMS},count,require_shared_identity=True)
    report.update(phase="stored_initializer_native_replay",predictions_frozen_before_replays=True,output_files=frozen)
    persist()
    replay_max = 0.
    for selection in chunks:
        actual_count = selection.stop-selection.start
        owned = {key:reread[key][selection].copy() for key in NATIVE_PARAMETER_DIMS}
        decoded = call(report,"native_replay",lambda:owned_call(native_decode,owned),persist)
        for name,expected,dimension in (("vertices",saved[selection],vertices),("joints",reread["mhr_joints"][selection],joint_count),
                                         ("keypoints",reread["mhr_keypoints"][selection],keypoint_count)):
            actual = geometry.checked_geometry(decoded[name],actual_count,dimension)
            error = geometry.residuals(expected,actual,"m")["max_point_mm"]
            replay_max=max(replay_max,error)
            if error>.01:raise ValueError("Stored native V/J/KP replay exceeds unchanged .01mm fidelity")
        if not np.array_equal(checked_faces(decoded["faces"],vertices=vertices,count=face_count,digest=face_sha),faces):
            raise ValueError("Stored native topology changed")
        validated(report,"native_replay",persist)
    report.update(phase="official_reference_replay",reference_settings=dict(precision="float32",residual_dtype="float64",
        chunk=CHUNK,device="cuda",mean_point_gate_mm=2.,native_max_point_gate_mm=.01));persist()
    errors=[];maximum=0.
    for selection in chunks:
        owned={key:stored[key].copy() for key in stored}
        before=array_bytes(owned)
        recovered=call(report,"reference",lambda:reference_decode(owned,selection),persist)
        if array_bytes(owned)!=before:raise ValueError("Official consumer modified frozen parameter copies")
        values=geometry.residuals(saved[selection],recovered,"mm")
        if max(values["per_frame_mean_mm"])>2.:raise ValueError("An original frame exceeds unchanged2mm official fidelity")
        errors.extend(values["per_frame_mean_mm"]);maximum=max(maximum,values["max_point_mm"])
        validated(report,"reference",persist)
    if (array_bytes(original_params)!=original_bytes or array_bytes(selected)!=selected_bytes
            or original["metadata"]!=original_meta or array_bytes({key:original[key] for key in original_cached})!=original_cached
            or {key:original[key] for key in original_timeline}!=original_timeline
            or {path.name for path in out.iterdir() if path.name!="report.json"}!=OUTPUTS
            or any((out/name).stat().st_mode&0o222 for name in frozen)
            or {name:inputs.identity(out/name) for name in frozen}!=frozen):
        raise ValueError("Original inputs or frozen shared artifacts changed")
    expected=len(chunks)
    if any(report[name+suffix]!=expected for name in CALLS for suffix in ("_attempts","_returns","_validated")):
        raise ValueError("Every actual full-timeline native/direct/saved/reference chunk required")
    report.update(phase="geometry_routes_complete",frames=count,reference_per_frame_mean_mm=errors,reference_max_point_mm=maximum,
        native_direct_max_point_mm=native_direct_max,native_replay_max_point_mm=replay_max,
        stored_initializer_native_replay_verified=True,shared_identity_verified=True,original_initializer_unchanged=True,
        frozen_outputs_rehashed_after_reference=True,shared_initializer_sha256=frozen["shared_initializer.pkl"]["sha256"],
        direct_parameters_sha256=frozen["direct_parameters.npz"]["sha256"],target_sha256=frozen["target.npy"]["sha256"])
    return frozen


def source_helpers(code):
    names=("infra/cari_shared_prepare.py","infra/run_cari_shared_prepare.sh","infra/cari_clip_inputs.py",
           "infra/cari96_prepare.py","infra/run_cari96_prepare.sh","infra/cari96_inputs.py","infra/body_smoke.py",
           "src/world_reward/shared_identity.py","src/world_reward/timeline.py","src/world_reward/data.py")
    if any((code/name).stat().st_mode&0o222 for name in names):
        raise ValueError("Immutable source-bound helper closure required")
    return {name:inputs.identity(code/name) for name in names}


def run(root,out,code,episode,report,persist):
    pins_path=code/f"configs/cari_clip_{episode:06d}_input_pins.json"
    pin_id=inputs.identity(pins_path);pins=json.loads(pins_path.read_text())
    if pins_path.stat().st_mode&0o222:raise ValueError("Immutable externally supplied source input pins required")
    spec=inputs.PublicClipSpec(**pins["clip_spec"])
    if spec.episode_index!=episode:raise ValueError("Explicit episode differs from pinned full clip")
    public=inputs.verify_public_inputs(root,spec,pins)
    # Establish the policy before model construction as well as before all
    # geometry, neutral-height, crop, render or cache execution.
    before_models=(share_first_frame_identity({key:public["initializer"][key] for key in NATIVE_PARAMETER_DIMS},spec.total_frames),
                   shared_initializer_metadata(public["initializer"]["metadata"],spec.total_frames))
    assets,hashes=body._body_assets(root);source=body._source_identity(root)
    prior=public["reports"]["adapter"];body_report=public["reports"]["body"]
    if (source!=prior.get("inference_source_identity") or source!=body_report.get("inference_source_identity")
            or hashes!=body_report.get("body_assets") or (assets/"mhr_buffers.pt").exists()):
        raise ValueError("Exact original native decoder source/checkpoint assets required")
    vendor=root/"vendor/video_to_data";body._pinned_checkout(vendor,body.UPSTREAM_REVISION)
    native=vendor/"reconstruction/modules/v2d_cari4d/lib/cari4d"
    tool=root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py";model=root/"weights/mhr/mhr_model.pt"
    bindings=[geometry.binding(native/"lib_mhr/mhr_layer.py",geometry.LAYER_SHA),
              geometry.binding(tool,geometry.CONVERTER_SHA),geometry.binding(model,geometry.REFERENCE_SHA)]
    helpers=source_helpers(code)
    report.update(phase="public_input_audited",clip_spec=asdict(spec),source_files=public["source_files"],
        original_input_paths={name:str(path) for name,path in public["paths"].items()},input_pins=pin_id,
        input_report_sha256=pins["input_report"]["sha256"],input_sha256=public["reports"]["inputs"]["input_sha256"],
        source_inputs_report_sha256=pins["input_report"]["sha256"],source_inputs_producer_revision=pins["input_report"]["producer_revision"],
        source_inputs_script_sha256=pins["input_report"]["script_sha256"],
        input_dataset_revision=inputs.DATASET_REVISION,source_bindings=bindings,source_helpers=helpers,
        body_assets=hashes,inference_source_identity=source,decoder_identity=prior["decoder_identity"]);persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise ValueError("Fresh deterministic native runtime required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4":
        raise ValueError("Pinned native H100 CUDA required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    report["runtime"]=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,
        deterministic_algorithms=True,warn_only=False,jit_optimized_execution=False,tf32=False,cudnn_benchmark=False,seed=0,chunk=CHUNK)
    sys.path[:0]=[str(native),"/workspace/v2d_sam3d_body/lib"]
    from lib_mhr.mhr_layer import MHRLayer
    if Path(sys.modules[MHRLayer.__module__].__file__).resolve()!=native/"lib_mhr/mhr_layer.py":
        raise ValueError("Actual native decoder import differs")
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),checkpoint_path=assets/"model.ckpt",
        buffer_path=out/"never_use_unverified_buffer.pt",mhr_model_path=assets/"assets/mhr_model.pt",device="cuda")
    if layer.decoder_identity()!=prior["decoder_identity"]:raise ValueError("Original native decoder identity differs")
    tensors=lambda params:{key:torch.tensor(value.copy(),device="cuda",dtype=torch.float32) for key,value in params.items()}
    def native_decode(params):
        decoded=layer.mhr_forward(tensors(params));torch.cuda.synchronize()
        return {"vertices":decoded.vertices.cpu().numpy(),"joints":decoded.joints.cpu().numpy(),
                "keypoints":decoded.keypoints.cpu().numpy(),"faces":decoded.faces.cpu().numpy()}
    def native_direct(params):
        t=tensors(params);context=layer.backend._vertices_context(t,detach_fixed=False)
        trans,pose,shape,scale=layer.backend._mutable_vertices_inputs(t,context)
        if context.head.enable_hand_model:raise ValueError("Original body-only native head required")
        vertices,controls=context.head.mhr_forward(global_trans=trans*context.flip,global_rot=context.global_rot,
            body_pose_params=pose,hand_pose_params=context.hand,scale_params=scale,shape_params=shape,
            expr_params=context.face,return_model_params=True)
        expected=context.head.scale_mean[None,:]+scale@context.head.scale_comps;torch.cuda.synchronize()
        return (vertices*context.flip).cpu().numpy(),controls.cpu().numpy(),expected.cpu().numpy()
    # Importing the official consumer happens before use, not before identity
    # selection inside prepare_geometry. It never chooses/fits predictions.
    module_spec=importlib.util.spec_from_file_location("cari_shared_official_reference",tool)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    reference=module.MHR(str(model),"cuda",chunk=CHUNK,precision="float32")
    if (Path(module.MHR.run.__code__.co_filename).resolve()!=tool or reference.mdtype!=torch.float32
            or reference.dtype!=torch.float64 or reference.chunk!=CHUNK or reference.device.type!="cuda"):
        raise ValueError("Exact official FP32-model/FP64-residual consumer required")
    def reference_decode(stored,selection):
        pose=torch.tensor(stored["pose"][selection],device="cuda",dtype=torch.float64)
        shared=torch.tensor(np.r_[stored["scales"],stored["shape"]][None],device="cuda",dtype=torch.float64)
        vertices,_=reference.run(pose,shared);torch.cuda.synchronize();return vertices.cpu().numpy()
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        frozen=prepare_geometry(public["initializer"],spec.total_frames,out,native_decode,native_direct,reference_decode,report,persist,
                                preselected=before_models)
    if (inputs.identity(pins_path)!=pin_id or {name:inputs.identity(root/name) for name in public["source_files"]}!=public["source_files"]
            or body._source_identity(root)!=source or body._body_assets(root)[1]!=hashes or source_helpers(code)!=helpers
            or {name:inputs.identity(out/name) for name in frozen}!=frozen
            or {path.name for path in out.iterdir() if path.name!="report.json"}!=OUTPUTS):
        raise ValueError("Original public inputs/native assets/helpers/frozen exports changed")
    for row in bindings:
        if inputs.identity(Path(row["path"]))!={name:row[name] for name in ("sha256","bytes")}:
            raise ValueError("Pinned actual native/reference source changed")
    report.update(status="pass",phase="complete",source_inputs_assets_rehashed=True,source_helpers_rehashed=True,
        original_frame_coverage_verified=True,source_geometry_masks_depth_pose_unchanged=True)


def main(argv=None):
    args=parser().parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    out=root/output_relative(args.episode)
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or os.geteuid()!=1000
            or {path.name for path in Path("/sys/class/net").iterdir()}!={"lo"} or os.environ["WR_IMAGE_ID"]!=IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or root.resolve()!=root.absolute() or code.resolve()!=code.absolute()
            or not out.is_dir() or any(out.iterdir()) or out.resolve()!=out.absolute()):
        raise ValueError("Fresh source-bound offline Azure full shared output required")
    report=dict(stage=STAGE,status="fail",phase="public_input_audit",episode_index=args.episode,producer_revision=revision,
        script_sha256=inputs.public.sha256(Path(__file__)),image_id=IMAGE,input_track="track_1",network="none",
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],budget_seconds=BUDGET,
        learned_inference_calls=0,optimizer_calls=0,converter_LM_calls=0,quality_verified=False,adoption_authorized=False,
        submission_produced=False,submission_eligible=False)
    report.update({name+suffix:0 for name in CALLS for suffix in ("_attempts","_returns","_validated")})
    receipt=out/"report.json";started=time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Full shared native preparation exceeded600s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,args.episode,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o444)


if __name__=="__main__":main()

"""Full original-timeline constrained native CoCoNet; no refine or inverse fit.

Unchanged native96 windows and first-occurrence stitching run once. The sole
isolated compose hook reads actual source-bound caller indices, preserves raw
predictions and verifies the saved full assembly. Identity precedes decoding,
neutral-height, crops, rendering and caching; no old CARI predictions are read.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict
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
import body_smoke as body
import cari_clip_inputs as inputs
import cari_shared_prepare as prepare
import cari96_forward as assets
import cari96_native as constrained
from cari_runner import build_cari_runtime_environment, CHECKPOINT_SHA256, CONFIG_RELATIVE_PATH
from world_reward.contracts import require_rigid_transforms
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, share_first_frame_identity, shared_initializer_metadata, validate_native_parameters
from world_reward.timeline import NATIVE_WINDOW, native_window_starts, finite_chunks, first_occurrence_ownership, native_caller_window_indices

STAGE = "world_reward_native_cari_shared_full_video_forward"
BUDGET, IMAGE = 900, assets.IMAGE
PREPARE_FILES = {*prepare.OUTPUTS, "report.json"}
FORWARD_FILES = {"coconet.pth", "report.json"}


def output_relative(episode):
    prepare.output_relative(episode)  # Exact Python integer, Track1 episode 0..29.
    return f"outputs/episode_{episode:06d}/cari_shared_forward_v1"


def identity(path, *, immutable=False):
    return assets.identity(path, immutable=immutable)


def _expected(record, expected, label):
    if type(record) is not dict or any(type(record.get(k)) is not type(v) or record[k] != v for k,v in expected.items()):
        raise ValueError(label)


def validate_artifact_pins(spec, pins, role):
    if type(spec) is not inputs.PublicClipSpec or role not in {"prepare", "forward"}:
        raise ValueError("Explicit public clip and supported producer role required")
    files = PREPARE_FILES if role == "prepare" else FORWARD_FILES
    if (type(pins) is not dict or set(pins) != {"schema", "clip_spec", role, role+"_files"}
            or pins["schema"] != f"world-reward-cari-shared-{role}-pins-v1"
            or type(pins["clip_spec"]) is not dict or pins["clip_spec"] != asdict(spec)
            or inputs.PublicClipSpec(**pins["clip_spec"]) != spec
            or type(pins[role+"_files"]) is not dict or set(pins[role+"_files"]) != files):
        raise ValueError("Complete exact full-clip producer pins required")
    assets._receipt(pins[role],producer=True)
    for row in pins[role+"_files"].values(): assets._receipt(row)
    if pins[role+"_files"]["report.json"] != {k:pins[role][k] for k in ("sha256", "bytes")}:
        raise ValueError("Producer report SHA/bytes link differs")


def _frozen_inventory(directory, expected):
    inputs.public.regular(directory/"report.json")
    paths = tuple(directory.iterdir())
    if {p.name for p in paths} != set(expected) or any(not p.is_file() for p in paths):
        raise ValueError("Exact frozen producer directory inventory required")
    observed = {p.name:identity(p,immutable=True) for p in paths}
    if observed != expected: raise ValueError("All frozen producer bytes must match before interpretation")
    return observed


def _finite_errors(values, count, maximum, label):
    if (type(values) is not list or len(values) != count
            or any(type(x) not in (int,float) or not np.isfinite(x) or not 0 <= x <= maximum for x in values)):
        raise ValueError(label)


def validate_prepare_report(report, spec, pins, input_pins):
    """Hash/JSON-only actual full shared generator receipt contract."""
    validate_artifact_pins(spec,pins,"prepare");inputs.validate_pins(spec,input_pins)
    count=spec.total_frames;chunks=finite_chunks(count,prepare.CHUNK)
    expected=dict(stage=prepare.STAGE,status="pass",phase="complete",episode_index=spec.episode_index,
        frames=count,clip_spec=asdict(spec),image_id=IMAGE,input_track="track_1",network="none",
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],
        learned_inference_calls=0,optimizer_calls=0,converter_LM_calls=0,quality_verified=False,
        adoption_authorized=False,submission_produced=False,submission_eligible=False,
        identity_fixed_before_first_decode=True,original_frame_indices=list(range(count)),
        chunk_counts=[s.stop-s.start for s in chunks],predictions_frozen_before_replays=True,
        stored_initializer_native_replay_verified=True,shared_identity_verified=True,original_initializer_unchanged=True,
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,frozen_outputs_rehashed_after_reference=True,
        original_frame_coverage_verified=True,source_geometry_masks_depth_pose_unchanged=True,
        input_dataset_revision=inputs.DATASET_REVISION,source_files=input_pins["source_files"],
        source_inputs_report_sha256=input_pins["input_report"]["sha256"],input_report_sha256=input_pins["input_report"]["sha256"],
        source_inputs_producer_revision=input_pins["input_report"]["producer_revision"],
        source_inputs_script_sha256=input_pins["input_report"]["script_sha256"],
        reference_settings=dict(precision="float32",residual_dtype="float64",chunk=16,device="cuda",mean_point_gate_mm=2.,native_max_point_gate_mm=.01))
    expected.update({name+suffix:len(chunks) for name in prepare.CALLS for suffix in ("_attempts","_returns","_validated")})
    _expected(report,expected,"Complete original-timeline shared preparation required")
    _expected(report,{k:pins["prepare"][k] for k in ("producer_revision","script_sha256")},"Actual preparation producer differs")
    outputs={p:v for p,v in pins["prepare_files"].items() if p!="report.json"}
    if report.get("output_files") != outputs: raise ValueError("Complete preparation output manifest differs")
    for key,path in (("shared_initializer_sha256","shared_initializer.pkl"),("direct_parameters_sha256","direct_parameters.npz"),("target_sha256","target.npy")):
        if report.get(key)!=outputs[path]["sha256"]:raise ValueError("Preparation artifact chain differs")
    _finite_errors(report.get("reference_per_frame_mean_mm"),count,2.,"Every original reference-fidelity gate required")
    for key in ("native_direct_max_point_mm","native_replay_max_point_mm"):
        value=report.get(key)
        if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=.01:raise ValueError("Native unchanged .01mm fidelity required")
    _expected(report.get("runtime"),dict(torch="2.5.1+cu124",cuda="12.4",deterministic_algorithms=True,
        warn_only=False,jit_optimized_execution=False,tf32=False,cudnn_benchmark=False,seed=0,chunk=16),"Prepared native runtime differs")


def verify_prepare_artifacts(root, code, spec, pins, input_pins, source_code=None):
    """Verify payloads against explicit producer sources; current code owns pins.

    The caller authenticates a historical source_code before passing it. Only
    readonly source bytes are checked here; historical code is never executed.
    Omitting source_code retains the original current-source contract.
    """
    source_code=code if source_code is None else source_code
    validate_artifact_pins(spec,pins,"prepare");inputs.validate_pins(spec,input_pins)
    directory=root/prepare.output_relative(spec.episode_index)
    files=_frozen_inventory(directory,pins["prepare_files"])
    source_files={name:identity(root/name) for name in input_pins["source_files"]}
    if source_files!=input_pins["source_files"]:raise ValueError("Every original public input SHA/bytes required")
    report=json.loads((directory/"report.json").read_text());validate_prepare_report(report,spec,pins,input_pins)
    pin_path=code/f"configs/cari_clip_{spec.episode_index:06d}_input_pins.json"
    if (json.loads(pin_path.read_text())!=input_pins or report.get("input_pins")!=identity(pin_path,immutable=True)
            or report.get("source_helpers")!=prepare.source_helpers(source_code)
            or identity(source_code/"infra/cari_shared_prepare.py",immutable=True)["sha256"]!=pins["prepare"]["script_sha256"]):
        raise ValueError("Current immutable actual preparation generator/pins differ")
    paths={name:str(root/path) for name,path in inputs.relative_paths(spec).items()}
    if report.get("original_input_paths")!=paths:raise ValueError("Original public input path binding differs")
    bindings=report.get("source_bindings")
    native=root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py"
    expected=[prepare.geometry.binding(native,prepare.geometry.LAYER_SHA),
        prepare.geometry.binding(root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py",prepare.geometry.CONVERTER_SHA),
        prepare.geometry.binding(root/"weights/mhr/mhr_model.pt",prepare.geometry.REFERENCE_SHA)]
    if bindings!=expected:raise ValueError("Actual native/reference generator bindings differ")
    # JSON dependency validation also binds video SHA, geometry/gauge and adapter decoder.
    reports=inputs.validate_reports(root,spec,input_pins)
    if (report.get("input_sha256")!=reports["inputs"]["input_sha256"]
            or report.get("body_assets")!=reports["body"].get("body_assets")
            or report.get("inference_source_identity")!=reports["adapter"].get("inference_source_identity")
            or report.get("decoder_identity")!=reports["adapter"].get("decoder_identity")):
        raise ValueError("Original video/body/decoder producer chain differs")
    if _frozen_inventory(directory,files)!=files or {name:identity(root/name) for name in source_files}!=source_files:
        raise ValueError("Producer sources changed during audit")
    bound={str(directory/name):row for name,row in files.items()}
    bound.update({str(root/name):row for name,row in source_files.items()})
    bound.update({str(source_code/name):row for name,row in report["source_helpers"].items()})
    bound[str(pin_path)]=identity(pin_path,immutable=True)
    bound.update({row["path"]:{key:row[key] for key in ("sha256","bytes")} for row in bindings})
    return dict(directory=directory,report=report,files=files,source_files=source_files,bindings=bound)


def validate_initializer(initializer, original, spec, prepared):
    count=spec.total_frames
    if type(initializer) is not dict or set(initializer)!=prepare.INITIALIZER_KEYS:
        raise ValueError("Only newly decoded full canonical initializer allowed")
    _expected(initializer,dict(body_model="mhr",frames=[f"{i:06d}" for i in range(count)],kids=[0]),"Exact original native initializer timeline required")
    params={key:initializer[key] for key in NATIVE_PARAMETER_DIMS}
    validate_native_parameters(params,count,require_shared_identity=True)
    expected=share_first_frame_identity({key:original[key] for key in NATIVE_PARAMETER_DIMS},count)
    if prepare.array_bytes(params)!=prepare.array_bytes(expected):raise ValueError("Original first-frame identity/five other blocks changed")
    for key,dimension in (("mhr_joints",127),("mhr_keypoints",70)):prepare.checked_array(initializer[key],(count,dimension,3))
    metadata=shared_initializer_metadata(original["metadata"],count)
    metadata.update(mhr_geometry_forward_verified=True,shared_native_direct_geometry_verified=True,
        native_direct_max_point_error_mm=prepared["native_direct_max_point_mm"],joint_keypoint_redecode_required=False,
        shared_joint_keypoint_native_redecoded=True)
    if initializer["metadata"]!=metadata:raise ValueError("Fresh shared geometry provenance differs from actual generator")


def prepared_inputs(root, code, spec, pins, input_pins):
    checked=verify_prepare_artifacts(root,code,spec,pins,input_pins)
    public=inputs.verify_public_inputs(root,spec,input_pins)
    import joblib
    initializer=joblib.load(checked["directory"]/"shared_initializer.pkl")
    validate_initializer(initializer,public["initializer"],spec,checked["report"])
    # No target materialization: readonly memmap shape/dtype is a native ABI gate.
    target=np.load(checked["directory"]/"target.npy",mmap_mode="r",allow_pickle=False)
    if target.dtype!=np.float32 or target.shape!=(spec.total_frames,18439,3) or target.flags.writeable:
        raise ValueError("Full original-timeline readonly decoded target required")
    with np.load(checked["directory"]/"direct_parameters.npz",allow_pickle=False) as stream:
        if set(stream.files)!={"frame_index","pose","scales","shape"}:raise ValueError("Exact direct controls inventory required")
        controls={key:stream[key] for key in stream.files}
    prepare.checked_array(controls["frame_index"],(spec.total_frames,),np.int64)
    if not np.array_equal(controls["frame_index"],np.arange(spec.total_frames,dtype=np.int64)):raise ValueError("No direct-control missing/reordered frames")
    prepare.checked_array(controls["pose"],(spec.total_frames,136));prepare.checked_array(controls["scales"],(68,))
    if prepare.checked_array(controls["shape"],(45,)).tobytes()!=initializer["mhr_shape"][0].tobytes():raise ValueError("Direct native shape identity differs")
    if _frozen_inventory(checked["directory"],checked["files"])!=checked["files"]:raise ValueError("Stored producer changed during deserialization")
    checked.update(public=public,initializer=initializer,poses=public["poses"])
    return checked


def fingerprint(value):
    return dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


class WindowCapture:
    """Owned per-window raw/composed/contact copies, exact full assembly proof.

    Frame identity is read at the hook by sys._getframe(1), passed explicitly
    to the pure pinned-code helper, and corroborated by all seven init blocks
    and original object poses. No frame/batch/global is mutated or retained.
    """
    def __init__(self,torch,cloned,initializer,poses,spec,report,decoder_identity):
        self.torch,self.cloned,self.initializer,self.poses,self.spec,self.report=torch,cloned,initializer,poses,spec,report
        if type(decoder_identity) is not dict or not decoder_identity:raise ValueError("Actual prepared native decoder identity required")
        self.decoder_identity=copy.deepcopy(decoder_identity)
        self.records=[]

    def _array(self,value,shape):
        return prepare.checked_array(constrained._numpy(self.torch,value),shape)

    def install(self):
        delegate=self.cloned.__globals__["compose_mhr_output"]
        def capture(prediction,batch):
            # Must be obtained HERE: helper depth would point at this hook instead.
            caller=sys._getframe(1)
            indices=native_caller_window_indices(caller,native_code=self.cloned.__code__,
                total_frames=self.spec.total_frames,window_ordinal=len(self.records),batch=batch)
            layer=caller.f_locals.get("layer")
            if layer is None or layer.decoder_identity()!=self.decoder_identity:raise ValueError("Actual native layer differs from fresh shared decoder")
            del caller  # Do not retain the native frame through the network/delegate.
            self.report["actual_native_decoder_identity_verified"]=True
            for key,dimension in NATIVE_PARAMETER_DIMS.items():
                actual=self._array(batch.get(key+"_init"),(1,NATIVE_WINDOW,dimension))
                if actual[0].tobytes()!=self.initializer[key][indices].tobytes():raise ValueError("Actual native batch initializer/frame binding differs")
            pose=self._array(batch.get("pose_perturbed"),(1,NATIVE_WINDOW,4,4))
            if pose[0].tobytes()!=self.poses["obj_pose_world"][indices].tobytes():raise ValueError("Actual native object input/frame binding differs")
            if not isinstance(prediction,Mapping):raise ValueError("Original native prediction mapping required")
            raw={}
            for key,value in prediction.items():
                if key in {"rot","trans"} or key.startswith("delta_mhr_"):
                    a=constrained._numpy(self.torch,value)
                    if a.dtype!=np.float32 or a.size%NATIVE_WINDOW:raise ValueError("Full native FP32 window output required")
                    raw[key]=prepare.checked_array(a.reshape(NATIVE_WINDOW,-1),(NATIVE_WINDOW,a.size//NATIVE_WINDOW)).copy()
            if not {"rot","trans","delta_mhr_shape","delta_mhr_hand"}.issubset(raw) or "delta_mhr_scale" in raw:
                raise ValueError("Unchanged commercial heads/supervision required")
            contact=self._array(constrained._numpy(self.torch,prediction.get("contact")).reshape(NATIVE_WINDOW,-1),(NATIVE_WINDOW,2)).copy()
            result=delegate(prediction,batch)
            composed={key:self._array(result.get(key),(1,NATIVE_WINDOW,dim))[0].copy() for key,dim in NATIVE_PARAMETER_DIMS.items()}
            # The delegate already snapshots predictions; capture independently
            # checks its pre-compose owned copy and contact, including signed zero.
            for key,saved in raw.items():
                if constrained._numpy(self.torch,prediction[key]).reshape(NATIVE_WINDOW,-1).tobytes()!=saved.tobytes():
                    raise ValueError("Raw native window changed before capture")
            if constrained._numpy(self.torch,prediction["contact"]).reshape(NATIVE_WINDOW,2).tobytes()!=contact.tobytes():
                raise ValueError("Native contact logits changed during compose")
            if self._array(batch["pose_perturbed"],(1,NATIVE_WINDOW,4,4))[0].tobytes()!=self.poses["obj_pose_world"][indices].tobytes():
                raise ValueError("Native object input changed during composition")
            row=dict(indices=indices,raw=raw,composed=composed,contact=contact)
            row["fingerprints"]={"indices":fingerprint(indices),"raw":{k:fingerprint(v) for k,v in raw.items()},
                "composed":{k:fingerprint(v) for k,v in composed.items()},"contact":fingerprint(contact)}
            for a in (*raw.values(),*composed.values(),contact):a.flags.writeable=False
            self.records.append(row)
            self.report["captured_windows"]=len(self.records)
            return result
        self.cloned.__globals__["compose_mhr_output"]=capture
        return self.cloned

    def verify(self,bundle):
        owner,local=first_occurrence_ownership(self.spec.total_frames,[row["indices"] for row in self.records])
        keys=set(self.records[0]["raw"])
        if any(set(row["raw"])!=keys for row in self.records) or set(bundle.get("raw",{}))!=keys:
            raise ValueError("Raw heads must retain exact same full native inventory")
        for ordinal,row in enumerate(self.records):
            # Retained fingerprints detect accidental capture-buffer mutation.
            current={"indices":fingerprint(row["indices"]),"raw":{k:fingerprint(v) for k,v in row["raw"].items()},
                "composed":{k:fingerprint(v) for k,v in row["composed"].items()},"contact":fingerprint(row["contact"])}
            if current!=row["fingerprints"]:raise ValueError("Untouched owned window fingerprint changed")
            target=np.flatnonzero(owner==ordinal);selection=local[target]
            for kind,actual in (("raw",bundle["raw"]),("composed",bundle["pr"])):
                for key,source in row[kind].items():
                    value=prepare.checked_array(actual.get(key),(self.spec.total_frames,source.shape[1]))
                    if value[target].tobytes()!=source[selection].tobytes():raise ValueError("Stored first-occurrence native assembly differs")
            if bundle["pr"]["contact_logits"][target].tobytes()!=row["contact"][selection].tobytes():
                raise ValueError("Stored first-occurrence contact assembly differs")
        return dict(window_starts=list(native_window_starts(self.spec.total_frames)),
            window_owned_counts=[int(np.count_nonzero(owner==i)) for i in range(len(self.records))],
            window_fingerprints=[dict(start=int(row["indices"][0]),**row["fingerprints"]) for row in self.records],
            owner_window=fingerprint(owner),owner_local=fingerprint(local),actual_caller_indices_verified=True,
            native_first_occurrence_assembly_verified=True)


def validate_bundle(bundle,initializer,poses,spec,report,capture):
    count=spec.total_frames;windows=len(native_window_starts(count));names=initializer["frames"]
    _expected(bundle,dict(schema="cari4d.mhr_wild_inference.v1",gt={},frames=names,kid=0,
        frame_meta=[dict(frame=name,src_frame=i,kid=0) for i,name in enumerate(names)]),"Exact native full original no-GT bundle required")
    _expected(bundle.get("metadata"),dict(ground_truth_used=False,window_length=96,window_stride=96,
        overlap_policy="first_occurrence",materialized_input_cache=None,materialized_input_cache_identity=None,depth_backend="moge2",
        object_pose_frame="centered_axis_aligned",object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
        object_pose_storage_frame="output_aligned_mesh_frame",object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist()),"Ordinary full native no-cache/single object-frame metadata differs")
    _expected(bundle.get("config"),dict(clip_len=96,enable_amp=True,pred_mhr_shape=True,pred_mhr_scale=False,
        body_model="mhr",mhr_joint_supervision_mode="body12_freeze_hand_face_all_losses"),"Unchanged native config required")
    _expected(bundle.get("checkpoint"),dict(step=200000),"Original checkpoint step required")
    for prefix in ("pr","pr_initial","in"):
        values=bundle.get(prefix,{})
        validate_native_parameters({key:values.get(key) for key in NATIVE_PARAMETER_DIMS},count,require_shared_identity=True)
        require_rigid_transforms(prepare.checked_array(values.get("pose_abs"),(count,4,4)),count)
    for key in NATIVE_PARAMETER_DIMS:
        if (bundle["in"][key].tobytes()!=initializer[key].tobytes()
                or bundle["pr_initial"][key].tobytes()!=bundle["pr"][key].tobytes()):raise ValueError("Native full input/initial-prediction bytes changed")
    for key in constrained.IDENTITY_DIMS:
        if bundle["pr"][key].tobytes()!=initializer[key].tobytes():raise ValueError("Original shared identity changed")
    if (bundle["in"]["pose_abs"].tobytes()!=poses["obj_pose_world"].tobytes()
            or bundle["pr_initial"]["pose_abs"].tobytes()!=bundle["pr"]["pose_abs"].tobytes()
            or prepare.checked_array(bundle["pr"].get("pose_abs_1st_delta"),(count,4,4)).tobytes()!=bundle["pr"]["pose_abs"].tobytes()):
        raise ValueError("Original object gauge/native prediction copies differ")
    faces=prepare.checked_array(bundle.get("faces"),(36874,3),np.int32)
    if hashlib.sha256(faces.astype("<i4",copy=False).tobytes()).hexdigest()!=assets.FACE_SHA:raise ValueError("Original native topology differs")
    neutral=prepare.checked_array(bundle.get("mhr_neutral_height_init"),(count,))
    spatial=prepare.checked_array(bundle.get("mhr_spatial_scale"),(count,))
    diameter=prepare.checked_array(bundle.get("mesh_diameter"),(count,))
    if (np.any(neutral<=0) or np.any(spatial<=0) or np.any(diameter<=0)
            or not np.allclose(neutral*spatial,2.,atol=1e-6,rtol=1e-6)
            or any(any(row.tobytes()!=value[0].tobytes() for row in value) for value in (neutral,spatial,diameter))):
        raise ValueError("One clip-constant geometry/height/spatial normalization required")
    K=prepare.checked_array(bundle.get("K_rois"),(count,3,3));boxes=prepare.checked_array(bundle.get("bboxes"),(count,4))
    if np.any(K[:,:2,:2].diagonal(axis1=1,axis2=2)<=0) or not np.array_equal(K[:,2],np.tile(np.array([0,0,1],np.float32),(count,1))) or np.any(boxes[:,2:]<=boxes[:,:2]):
        raise ValueError("Native original crop K/bboxes invalid")
    contact=prepare.checked_array(bundle["pr"].get("contact_logits"),(count,2))
    if prepare.checked_array(bundle["pr_initial"].get("contact_logits"),(count,2)).tobytes()!=contact.tobytes():raise ValueError("Native contacts changed")
    observations=bundle.get("observations",{})
    for key in ("human_mask","object_mask"):
        value=observations.get(key)
        if type(value) is not np.ndarray or value.dtype!=np.bool_ or value.shape!=(count,224,224):raise ValueError("Full observed masks required, including empty frames")
    fullK=prepare.checked_array(observations.get("K_full"),(count,3,3))
    if not np.array_equal(fullK,np.broadcast_to(inputs.inferred_camera(spec).astype(np.float32),(count,3,3))):raise ValueError("Original inferred camera changed")
    prepare.checked_array(observations.get("postopt_K_rois"),(count,3,3))
    if observations.get("postopt_crop_contract")!="cari4d.smplh_postopt_full_resolution_crop.v1":raise ValueError("Original native postopt crop required")
    for key in ("postopt_human_mask","postopt_object_mask"):
        value=prepare.checked_array(observations.get(key),(count,256,256))
        if not np.isin(value,[0,1]).all():raise ValueError("Native postopt binary masks required")
    for key,dimension in (("mhr_hand",108),("mhr_face",72)):
        delta="delta_"+key
        if key=="mhr_hand" and delta not in bundle["raw"]:raise ValueError("Actual frozen hand head required")
        if delta in bundle["raw"]:
            value=prepare.checked_array(bundle["raw"][delta],(count,dimension))
            if value.tobytes()!=np.zeros_like(value).tobytes():raise ValueError("Exact frozen native zeros_like delta required")
        expected=initializer[key]+np.zeros_like(initializer[key])
        if bundle["pr"][key].tobytes()!=expected.tobytes() or not np.array_equal(bundle["pr"][key],initializer[key]):
            raise ValueError("Unchanged native FP32 init+zero hand/face operation required")
    _expected(report,{key:windows for key in constrained.COUNTERS},"Exactly one compose/delegate verification per actual window required")
    _expected(report,dict(raw_prediction_bytes_preserved=True,initializer_bytes_preserved=True,
        native_compose_global_unchanged=True,native_function_code_unchanged=True),"Unchanged native globals/raw evidence required")
    return capture.verify(bundle)


def source_helpers(code):
    names=("infra/cari_full_forward.py","infra/run_cari_full_forward.sh","infra/cari96_forward.py",
        "infra/cari96_native.py","infra/cari_runner.py","src/world_reward/contracts.py")
    return prepare.source_helpers(code)|{name:identity(code/name,immutable=True) for name in names}


def validate_forward_report(report,spec):
    count=spec.total_frames;starts=list(native_window_starts(count))
    expected=dict(stage=STAGE,status="pass",phase="complete",episode_index=spec.episode_index,clip_spec=asdict(spec),frames=count,
        source_frames=count,original_frame_indices=list(range(count)),window_starts=starts,input_track="track_1",
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",
        forward_attempts=1,forward_returns=1,forward_validated=1,captured_windows=len(starts),hub_attempts=2,hub_returns=2,
        shared_identity_verified=True,raw_prediction_bytes_preserved=True,initializer_bytes_preserved=True,
        native_global_unchanged=True,hub_restored=True,source_inputs_assets_rehashed=True,source_helpers_rehashed=True,
        stored_bundle_reread_verified=True,actual_caller_indices_verified=True,native_first_occurrence_assembly_verified=True,
        actual_native_decoder_identity_verified=True,
        quality_verified=False,adoption_authorized=False,refinement_performed=False,unchanged_original_CARI_method=False,
        MHR_strict_scope_performed=False,exact_deterministic_forward_claim=False,submission_produced=False,
        input_dataset_revision=inputs.DATASET_REVISION,native_enable_amp=True,deterministic_algorithms=False,warn_only=False,TF32=False,
        image_id=IMAGE,budget_seconds=BUDGET,native_source_sha256=constrained.NATIVE_SOURCE_SHA256,
        upstream_revision=constrained.UPSTREAM_REVISION,native_function_code_unchanged=True,isolated_function_globals=True,
        overridden_globals=["compose_mhr_output"],native_compose_global_unchanged=True,native_global_replacement_performed=False,
        native_global_restoration_required=False,constrained_identity_composition=True,scale_delta_present=False,
        nonidentity_output_postprocessing=False,original_network_modified=False,
        native_initializer_decode_chunk=8,native_initializer_decode_chunk_counts=[s.stop-s.start for s in finite_chunks(count,8)])
    expected.update({key:len(starts) for key in constrained.COUNTERS})
    _expected(report,expected,"Complete actual full constrained native forward required")
    owner,_=first_occurrence_ownership(count,[np.arange(s,s+96,dtype=np.int64) for s in starts])
    if report.get("window_owned_counts")!=[int(np.count_nonzero(owner==i)) for i in range(len(starts))]:raise ValueError("Exact original overlap ownership required")
    rows=report.get("window_fingerprints")
    if type(rows) is not list or len(rows)!=len(starts) or any(type(row) is not dict or type(row.get("start")) is not int or row["start"]!=start for row,start in zip(rows,starts)):
        raise ValueError("Every actual untouched native window fingerprint required")
    def checked_fingerprint(row,shape,dtype):
        if (type(row) is not dict or set(row)!={"shape","dtype","sha256"} or row["shape"]!=shape or row["dtype"]!=dtype
                or type(row["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}",row["sha256"])):
            raise ValueError("Exact native array fingerprint ABI required")
    raw_keys=None
    for row in rows:
        if set(row)!={"start","indices","raw","composed","contact"}:raise ValueError("Exact window receipt inventory required")
        checked_fingerprint(row["indices"],[96],np.dtype(np.int64).str)
        expected_indices=np.arange(row["start"],row["start"]+96,dtype=np.int64)
        if row["indices"]!=fingerprint(expected_indices):raise ValueError("Recorded actual indices disagree with canonical source-bound schedule")
        checked_fingerprint(row["contact"],[96,2],np.dtype(np.float32).str)
        if type(row["composed"]) is not dict or set(row["composed"])!=set(NATIVE_PARAMETER_DIMS):raise ValueError("All seven composed native fingerprints required")
        for key,dimension in NATIVE_PARAMETER_DIMS.items():checked_fingerprint(row["composed"][key],[96,dimension],np.dtype(np.float32).str)
        raw=row["raw"]
        if (type(raw) is not dict or not {"rot","trans","delta_mhr_shape","delta_mhr_hand"}.issubset(raw)
                or "delta_mhr_scale" in raw or any(k not in {"rot","trans"} and not k.startswith("delta_mhr_") for k in raw)):
            raise ValueError("Unchanged raw commercial fingerprint inventory required")
        if raw_keys is not None and set(raw)!=raw_keys:raise ValueError("Identical native raw heads across every window required")
        raw_keys=set(raw)
        for key,value in raw.items():
            shape=value.get("shape") if type(value) is dict else None
            if type(shape) is not list or len(shape)!=2 or shape[0]!=96 or type(shape[1]) is not int or shape[1]<=0:raise ValueError("Native window raw fingerprint dimensions required")
            expected_dim={"trans":3,"delta_mhr_shape":45,"delta_mhr_hand":108,"delta_mhr_face":72}.get(key,shape[1])
            checked_fingerprint(value,[96,expected_dim],np.dtype(np.float32).str)
    _,local=first_occurrence_ownership(count,[np.arange(s,s+96,dtype=np.int64) for s in starts])
    if report.get("owner_window")!=fingerprint(owner) or report.get("owner_local")!=fingerprint(local):raise ValueError("Exact full original-frame ownership fingerprints required")


def verify_forward_artifacts(root,code,spec,pins,source_code=None):
    """Hash/JSON-only downstream gate; no Torch, Joblib or model loading.

    Rehashes original fifteen, preparation four, forward two and immutable
    code/native/reference generator bindings. An explicit authenticated historical
    source_code changes source checks only, never current pin ownership or code
    execution. Inference-only DINO/checkpoint
    assets are recorded in the forward receipt, not reloaded by this gate.
    """
    validate_artifact_pins(spec,pins,"forward")
    directory=root/output_relative(spec.episode_index);files=_frozen_inventory(directory,pins["forward_files"])
    report=json.loads((directory/"report.json").read_text());validate_forward_report(report,spec)
    _expected(report,{k:pins["forward"][k] for k in ("producer_revision","script_sha256")},"Actual forward producer differs")
    input_path=code/f"configs/cari_clip_{spec.episode_index:06d}_input_pins.json"
    prepare_path=code/f"configs/cari_clip_{spec.episode_index:06d}_shared_prepare_pins.json"
    input_pins=json.loads(input_path.read_text());prepare_pins=json.loads(prepare_path.read_text())
    checked=(verify_prepare_artifacts(root,code,spec,prepare_pins,input_pins)if source_code is None else
        verify_prepare_artifacts(root,code,spec,prepare_pins,input_pins,source_code=source_code))
    source_code=code if source_code is None else source_code
    if (report.get("input_pins")!=identity(input_path,immutable=True) or report.get("prepare_pins")!=identity(prepare_path,immutable=True)
            or report.get("prepare_report_sha256")!=prepare_pins["prepare"]["sha256"] or report.get("prepare_files")!=checked["files"]
            or report.get("public_source_files")!=checked["source_files"] or report.get("source_helpers")!=source_helpers(source_code)
            or report.get("script_sha256")!=identity(source_code/"infra/cari_full_forward.py",immutable=True)["sha256"]
            or report.get("original_input_paths")!=checked["report"]["original_input_paths"]
            or report.get("decoder_identity")!=checked["report"]["decoder_identity"]
            or report.get("body_assets")!=checked["report"]["body_assets"]
            or report.get("inference_source_identity")!=checked["report"]["inference_source_identity"]
            or report.get("input_sha256")!=checked["report"]["input_sha256"]
            or report.get("checkpoint_sha256")!=CHECKPOINT_SHA256 or report.get("config_sha256")!=assets.SOURCES[CONFIG_RELATIVE_PATH]
            or report.get("bundle_sha256")!=files["coconet.pth"]["sha256"] or report.get("bundle_bytes")!=files["coconet.pth"]["bytes"]
            or report.get("output_files")!={"coconet.pth":files["coconet.pth"]}):
        raise ValueError("Complete current actual forward/source/preparation artifact chain differs")
    # Bind inference-only identities without loading/rehashing checkpoints that
    # downstream refinement/export do not consume. Actual forward did pre/post.
    native=root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    checkpoint=root/"weights/cari4d/cari4d"/assets.CHECKPOINT_RELATIVE_PATH
    home=root/"weights/cari4d/sam3d_body/torch_home"
    required={str(native/path):digest for path,digest in assets.SOURCES.items()}
    required[str(checkpoint)]=CHECKPOINT_SHA256
    required.update({str(home/"hub/checkpoints"/name):digest for name,digest in assets.DINO_HASHES.items()})
    receipts={str(root/"results/weights-acquisition.json"),str(root/"results/auxiliary-assets.json")}
    source=report.get("source_files")
    if type(source) is not dict or not (set(required)|receipts).issubset(source):raise ValueError("Complete actual original inference source/asset receipt required")
    repository=home/"hub/facebookresearch_dinov2_main"
    for path,row in source.items():
        assets._receipt(row)
        if type(path) is not str or Path(path).absolute()!=Path(path) or ".." in Path(path).parts:
            raise ValueError("Canonical actual source asset path required")
        if path in required:
            if row["sha256"]!=required[path]:raise ValueError("Pinned original inference source/asset identity differs")
        elif path not in receipts and (not Path(path).is_relative_to(repository) or Path(path).suffix!=".py"):
            raise ValueError("Unknown inference source asset in producer receipt")
    if not any(Path(path).is_relative_to(repository) for path in source):raise ValueError("Actual offline DINO source receipt required")
    if _frozen_inventory(directory,files)!=files:raise ValueError("Forward producer changed during audit")
    bound=dict(checked["bindings"])
    bound.update({str(directory/name):row for name,row in files.items()})
    bound.update({str(source_code/name):row for name,row in report["source_helpers"].items()})
    bound[str(prepare_path)]=identity(prepare_path,immutable=True)
    return dict(directory=directory,report=report,files=files,prepare=checked,bindings=bound)


def run(root,out,code,episode,report,persist):
    input_path=code/f"configs/cari_clip_{episode:06d}_input_pins.json"
    prepare_path=code/f"configs/cari_clip_{episode:06d}_shared_prepare_pins.json"
    input_id=identity(input_path,immutable=True);prepare_id=identity(prepare_path,immutable=True)
    input_pins=json.loads(input_path.read_text());pins=json.loads(prepare_path.read_text())
    spec=inputs.PublicClipSpec(**input_pins["clip_spec"])
    if spec.episode_index!=episode:raise ValueError("Explicit episode differs from full original source pins")
    checked=prepared_inputs(root,code,spec,pins,input_pins)
    prepared,initializer,poses=checked["report"],checked["initializer"],checked["poses"]
    native,checkpoint,home,repository,source,body_assets,files=assets.asset_bindings(root,prepared)
    helpers=source_helpers(code)
    report.update(phase="public_inputs_audited",clip_spec=asdict(spec),input_pins=input_id,prepare_pins=prepare_id,
        prepare_report_sha256=pins["prepare"]["sha256"],prepare_files=checked["files"],public_source_files=checked["source_files"],
        original_input_paths=prepared["original_input_paths"],input_sha256=prepared["input_sha256"],input_dataset_revision=inputs.DATASET_REVISION,
        body_assets=body_assets,inference_source_identity=source,decoder_identity=prepared["decoder_identity"],
        checkpoint_sha256=CHECKPOINT_SHA256,config_sha256=assets.SOURCES[CONFIG_RELATIVE_PATH],source_helpers=helpers,
        composition_source_sha256=assets.SOURCES[constrained.COMPOSITION_RELATIVE_PATH],source_files=files,
        expected_windows=len(native_window_starts(spec.total_frames)),expected_hub_loads=2,
        native_initializer_decode_chunk=8,native_initializer_decode_chunk_counts=[s.stop-s.start for s in finite_chunks(spec.total_frames,8)]);persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise ValueError("Fresh explicit native CUDA process required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4":raise ValueError("Pinned native CUDA image required")
    np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    environment=build_cari_runtime_environment(str(native),str(root/"weights/cari4d/sam3d_body"),str(home),"/workspace/v2d_sam3d_body/lib")
    os.environ.update(environment);os.environ["MPLCONFIGDIR"]="/tmp/cari-full-matplotlib";sys.path[:0]=environment["PYTHONPATH"].split(":")
    original=assets.local_dino_hub(torch,repository,report)
    try:
        from tools import run_mhr_wild_inference as module
        if (Path(module.__file__).resolve()!=native/constrained.NATIVE_RELATIVE_PATH
                or module.compose_mhr_output.__code__.co_filename!=str(native/constrained.COMPOSITION_RELATIVE_PATH)
                or module.MHR_INIT_DECODE_BATCH_SIZE!=8):raise ValueError("Actual pinned native entrypoint/composition/decoder batch differs")
        cloned=constrained.clone_native_forward(torch,module.run_mhr_wild_inference,report,
            source_bytes=(native/constrained.NATIVE_RELATIVE_PATH).read_bytes())
        capture=WindowCapture(torch,cloned,initializer,poses,spec,report,prepared["decoder_identity"]);cloned=capture.install()
        report.update(phase="native_full_constrained_forward");report["forward_attempts"]+=1;persist()
        paths=checked["public"]["paths"]
        result=cloned(paths["export_seq"],paths["depth_h5"],checked["directory"]/"shared_initializer.pkl",paths["object_poses"],
            native/CONFIG_RELATIVE_PATH,checkpoint,out/"coconet.pth",stride=96,render_batch_size=32,crop_workers=8,
            crop_buffer_count=2,input_cache=None,use_input_cache=False,device_name="cuda",overwrite=False,
            wandb_run_path=None,offline_supervision_contract=True)
        report["forward_returns"]+=1;torch.cuda.synchronize()
        if Path(result).resolve()!=out/"coconet.pth":raise ValueError("Native full forward output path differs")
        bundle=torch.load(result,map_location="cpu",weights_only=False)
        evidence=validate_bundle(bundle,initializer,poses,spec,report,capture);report["forward_validated"]+=1
        consumed=dict(depth_source=paths["depth_h5"],mhr_init_source=checked["directory"]/"shared_initializer.pkl",
            foundationpose_source=paths["object_poses"],object_mesh=paths["mesh"])
        if any(bundle["metadata"].get(k)!=str(v) for k,v in consumed.items()) or bundle["checkpoint"].get("path")!=str(checkpoint):
            raise ValueError("Actual native full consumed inputs/checkpoint differ")
        if torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():raise ValueError("Ordinary native CUDA policy changed")
        active,digest=assets.config_receipt(bundle["config"])
        report.update(evidence,native_raw={k:fingerprint(v) for k,v in bundle["raw"].items()},native_metadata=bundle["metadata"],
            native_config=active,native_config_fingerprint=digest,native_frozen_hand_face_operation="unchanged_fp32_init_plus_exact_zero")
    finally:
        torch.hub.load=original;report["hub_restored"]=torch.hub.load is original;persist()
    if report["hub_attempts"]!=2 or report["hub_returns"]!=2:raise ValueError("Exactly two ordinary offline DINO backbone loads required")
    if (verify_prepare_artifacts(root,code,spec,pins,input_pins)["files"]!=checked["files"]
            or identity(input_path,immutable=True)!=input_id or identity(prepare_path,immutable=True)!=prepare_id
            or assets.asset_bindings(root,prepared)[-1]!=files or source_helpers(code)!=helpers):
        raise ValueError("Original public/prepared/model/source/helper inputs changed")
    path=out/"coconet.pth";path.chmod(0o444);frozen=identity(path,immutable=True)
    reread=torch.load(path,map_location="cpu",weights_only=False)
    if validate_bundle(reread,initializer,poses,spec,report,capture)!=evidence or identity(path,immutable=True)!=frozen:
        raise ValueError("Frozen full native bundle reread/assembly differs")
    if {p.name for p in out.iterdir()}!=FORWARD_FILES:raise ValueError("No unexpected native output/cache files allowed")
    report.update(status="pass",phase="complete",frames=spec.total_frames,source_frames=spec.total_frames,
        original_frame_indices=list(range(spec.total_frames)),bundle_sha256=frozen["sha256"],bundle_bytes=frozen["bytes"],
        output_files={"coconet.pth":frozen},shared_identity_verified=True,native_global_unchanged=report["native_compose_global_unchanged"],
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,stored_bundle_reread_verified=True,submission_produced=False)
    validate_forward_report(report,spec)


def main(argv=None):
    args=prepare.parser().parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    out=root/output_relative(args.episode)
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or os.environ["WR_IMAGE_ID"]!=IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or not out.is_dir() or any(out.iterdir())
            or any(p.resolve()!=p.absolute() for p in (root,code,out))):raise ValueError("Fresh offline source-bound full forward required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",episode_index=args.episode,producer_revision=revision,
        script_sha256=inputs.identity(Path(__file__))["sha256"],image_id=IMAGE,input_track="track_1",ground_truth_used=False,
        private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",budget_seconds=BUDGET,
        quality_verified=False,adoption_authorized=False,refinement_performed=False,unchanged_original_CARI_method=False,
        MHR_strict_scope_performed=False,exact_deterministic_forward_claim=False,native_enable_amp=True,
        deterministic_algorithms=False,warn_only=False,TF32=False,forward_attempts=0,forward_returns=0,
        forward_validated=0,hub_attempts=0,hub_returns=0,captured_windows=0)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Full constrained native forward exceeded900s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,args.episode,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist()
            for p in out.iterdir():
                if p.is_file():p.chmod(0o444)


if __name__=="__main__":main()

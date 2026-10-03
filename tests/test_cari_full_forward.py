"""Tiny synthetic full native timelines and producer gates, no model/data/GPU."""
import ast
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"src"));monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("cari_full_forward_test",HERE/"infra/cari_full_forward.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def clip(gate,n=501,episode=15):
    return gate.inputs.PublicClipSpec(episode,n,"front_stereo_camera_left",1152,1536)


def original(gate,n):
    params={k:np.zeros((n,d),np.float32) for k,d in gate.NATIVE_PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:]=[1,0,0,1,0,0];params["mhr_trans"][:,2]=2
    params["mhr_trans"][:,0]=np.arange(n,dtype=np.float32)/100
    params["mhr_shape"][:,0]=np.arange(n,dtype=np.float32)/1000
    params["mhr_scale"][:,0]=np.arange(n,dtype=np.float32)/1000
    params.update(frames=[f"{i:06d}" for i in range(n)],kids=[0],body_model="mhr",
        mhr_joints=np.zeros((n,127,3),np.float32),mhr_keypoints=np.zeros((n,70,3),np.float32),
        metadata=dict(ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],private_truth_read=False,
            mhr_geometry_forward_verified=True,projection_max_error_px=.001))
    return params


def selected(gate,source):
    n=len(source["frames"]);value=copy.deepcopy(source)
    value.update(gate.share_first_frame_identity({k:source[k] for k in gate.NATIVE_PARAMETER_DIMS},n))
    value["metadata"]=gate.shared_initializer_metadata(source["metadata"],n)
    value["metadata"].update(mhr_geometry_forward_verified=True,shared_native_direct_geometry_verified=True,
        native_direct_max_point_error_mm=.001,joint_keypoint_redecode_required=False,shared_joint_keypoint_native_redecoded=True)
    return value


def pose(n):
    value=np.tile(np.eye(4,dtype=np.float32),(n,1,1));value[:,0,3]=np.arange(n,dtype=np.float32)/100
    return dict(obj_pose_world=value)


def compose(prediction,batch):
    out={}
    for key in ("mhr_global_rot6d","mhr_trans","mhr_body_pose_cont","mhr_hand","mhr_shape","mhr_scale","mhr_face"):
        init=batch[key+"_init"]
        out[key]=init.copy() if key=="mhr_global_rot6d" else init+prediction.get("delta_"+key,np.zeros_like(init))
    return out


def run_mhr_wild_inference(initializer,poses,starts,compose_function,dimensions,fault=None):
    # Synthetic source-shaped caller, not claimed as actual pinned upstream code.
    names=initializer["frames"];clip_len=96;rows=[]
    layer=SimpleNamespace(decoder_identity=lambda:{"synthetic":"native" if fault!="decoder" else "wrong"})
    for ordinal,start in enumerate(starts):
        indices=np.arange(start,start+clip_len,dtype=np.int64)
        batch={k+"_init":initializer[k][indices][None].copy() for k in dimensions}
        batch["pose_perturbed"]=poses["obj_pose_world"][indices][None].copy()
        prediction=dict(rot=np.zeros((1,96,6),np.float32),trans=np.zeros((1,96,3),np.float32),
            delta_mhr_shape=np.full((1,96,45),.5+ordinal,np.float32),delta_mhr_hand=np.zeros((1,96,108),np.float32),
            delta_mhr_body_pose_cont=np.full((1,96,260),.01*(ordinal+1),np.float32),contact=np.full((1,96,2),ordinal,np.float32))
        if fault=="indices":indices[0]=indices[1]
        elif fault=="names":names=list(reversed(names))
        elif fault=="batchidentity":batch["mhr_trans_init"][0,0,0]+=.001
        elif fault=="poseidentity":batch["pose_perturbed"][0,0,0,3]+=.001
        elif fault=="deltadtype":prediction["delta_mhr_shape"]=prediction["delta_mhr_shape"].astype(np.float64)
        elif fault=="rawnonfinite":prediction["trans"][0,0,0]=np.nan
        elif fault=="scalehead":prediction["delta_mhr_scale"]=np.zeros((1,96,28),np.float32)
        elif fault=="contact":prediction["contact"]=np.zeros((1,96,1),np.float32)
        result=compose_mhr_output(prediction,batch)
        rows.append((indices,prediction,result))
    return rows


def captured(gate,n=501,fault=None,starts=None,delegate=compose):
    source=original(gate,n);initializer=selected(gate,source);poses=pose(n);report={}
    torch=SimpleNamespace(is_tensor=lambda _:False)
    def constrained(prediction,batch):
        return gate.constrained.constrain_composition(torch,delegate,prediction,batch,report)
    globals_copy=dict(run_mhr_wild_inference.__globals__);globals_copy["compose_mhr_output"]=constrained
    clone=FunctionType(run_mhr_wild_inference.__code__,globals_copy,"run_mhr_wild_inference")
    collector=gate.WindowCapture(torch,clone,initializer,poses,clip(gate,n),report,{"synthetic":"native"})
    collector.install()
    rows=clone(initializer,poses,starts or gate.native_window_starts(n),compose,gate.NATIVE_PARAMETER_DIMS,fault)
    owner,local=gate.first_occurrence_ownership(n,[row[0] for row in rows])
    raw={key:np.stack([rows[owner[i]][1][key].reshape(96,-1)[local[i]] for i in range(n)]) for key in rows[0][1] if key!="contact"}
    predicted={key:np.stack([rows[owner[i]][2][key].reshape(96,-1)[local[i]] for i in range(n)]) for key in gate.NATIVE_PARAMETER_DIMS}
    predicted["contact_logits"]=np.stack([rows[owner[i]][1]["contact"].reshape(96,-1)[local[i]] for i in range(n)])
    return collector,dict(raw=raw,pr=predicted),initializer,poses,report


@pytest.mark.parametrize("n",[96,97,191,192,193,501,790])
def test_actual_caller_indices_first_occurrence_and_owned_byte_copies(gate,n):
    capture,bundle,init,poses,report=captured(gate,n)
    before=copy.deepcopy(report);evidence=capture.verify(bundle)
    assert evidence["window_starts"]==list(gate.native_window_starts(n))
    assert sum(evidence["window_owned_counts"])==n and report==before
    for row in capture.records:
        assert not row["indices"].flags.writeable and all(not a.flags.writeable for a in row["raw"].values())
        assert all(not np.shares_memory(a,init[key]) for key,a in row["composed"].items())
    assert bundle["raw"]["delta_mhr_shape"].max()>.4
    assert bundle["pr"]["mhr_shape"].tobytes()==init["mhr_shape"].tobytes()
    assert all(report[key]==len(capture.records) for key in gate.constrained.COUNTERS)
    if n==501:
        assert evidence["window_starts"]==[0,96,192,288,384,405]
        assert evidence["window_owned_counts"]==[96,96,96,96,96,21]
        assert np.all(bundle["pr"]["contact_logits"][405:480]==4) and np.all(bundle["pr"]["contact_logits"][480:]==5)


@pytest.mark.parametrize("fault",["indices","names","batchidentity","poseidentity","deltadtype","rawnonfinite","scalehead","contact","decoder"])
def test_capture_rejects_actual_identity_abi_faults_before_delegate(gate,fault):
    with pytest.raises((ValueError,TypeError)):captured(gate,97,fault)


@pytest.mark.parametrize("kind",["raw","composed","contact","fingerprint","drop","repeat","reorder","buffer"])
def test_no_stitch_repair_or_ordinal_only_acceptance(gate,kind):
    capture,bundle,*_=captured(gate,501)
    if kind=="raw":bundle["raw"]["delta_mhr_shape"][480,0]+=.001
    elif kind=="composed":bundle["pr"]["mhr_body_pose_cont"][480,0]+=.001
    elif kind=="contact":bundle["pr"]["contact_logits"][480,0]+=.001
    elif kind=="fingerprint":capture.records[0]["fingerprints"]["raw"]["rot"]["sha256"]="a"*64
    elif kind=="drop":capture.records.pop()
    elif kind=="repeat":capture.records[-1]=capture.records[-2]
    elif kind=="reorder":capture.records[-1],capture.records[-2]=capture.records[-2],capture.records[-1]
    else:
        a=capture.records[0]["raw"]["rot"];a.flags.writeable=True;a[0,0]=1;a.flags.writeable=False
    with pytest.raises(ValueError):capture.verify(bundle)


def preparation_receipt(gate,spec):
    inputs=dict(schema="world-reward-cari-clip-input-pins-v1",clip_spec=asdict(spec),
        input_report=dict(sha256="b"*64,bytes=4,producer_revision="c"*40,script_sha256="d"*64),
        source_files={p:dict(sha256="b"*64,bytes=4) for p in gate.inputs.source_paths(spec)})
    files={p:dict(sha256="e"*64,bytes=4) for p in gate.PREPARE_FILES}
    pins=dict(schema="world-reward-cari-shared-prepare-pins-v1",clip_spec=asdict(spec),
        prepare=files["report.json"]|dict(producer_revision="a"*40,script_sha256="f"*64),prepare_files=files)
    chunks=gate.finite_chunks(spec.total_frames,16)
    report=dict(stage=gate.prepare.STAGE,status="pass",phase="complete",episode_index=spec.episode_index,
        frames=spec.total_frames,clip_spec=asdict(spec),image_id=gate.IMAGE,input_track="track_1",network="none",
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],learned_inference_calls=0,
        optimizer_calls=0,converter_LM_calls=0,quality_verified=False,adoption_authorized=False,submission_produced=False,
        submission_eligible=False,identity_fixed_before_first_decode=True,original_frame_indices=list(range(spec.total_frames)),
        chunk_counts=[s.stop-s.start for s in chunks],predictions_frozen_before_replays=True,
        stored_initializer_native_replay_verified=True,shared_identity_verified=True,original_initializer_unchanged=True,
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,frozen_outputs_rehashed_after_reference=True,
        original_frame_coverage_verified=True,source_geometry_masks_depth_pose_unchanged=True,input_dataset_revision=gate.inputs.DATASET_REVISION,
        source_files=inputs["source_files"],source_inputs_report_sha256="b"*64,input_report_sha256="b"*64,
        source_inputs_producer_revision="c"*40,source_inputs_script_sha256="d"*64,
        reference_settings=dict(precision="float32",residual_dtype="float64",chunk=16,device="cuda",mean_point_gate_mm=2.,native_max_point_gate_mm=.01),
        producer_revision="a"*40,script_sha256="f"*64,output_files={p:v for p,v in files.items() if p!="report.json"},
        shared_initializer_sha256="e"*64,direct_parameters_sha256="e"*64,target_sha256="e"*64,
        reference_per_frame_mean_mm=[.001]*spec.total_frames,native_direct_max_point_mm=.001,native_replay_max_point_mm=.001,
        runtime=dict(torch="2.5.1+cu124",cuda="12.4",deterministic_algorithms=True,warn_only=False,
            jit_optimized_execution=False,tf32=False,cudnn_benchmark=False,seed=0,chunk=16))
    report.update({name+suffix:len(chunks) for name in gate.prepare.CALLS for suffix in ("_attempts","_returns","_validated")})
    return pins,inputs,report


@pytest.mark.parametrize("n,ep",[(96,0),(97,29),(501,15)])
def test_full_prepare_receipt_exact_actual_last_batch(gate,n,ep):
    spec=clip(gate,n,ep);pins,inputs,report=preparation_receipt(gate,spec)
    gate.validate_prepare_report(report,spec,pins,inputs)
    assert report["chunk_counts"][-1]==((n-1)%16+1)


@pytest.mark.parametrize("fault",["schema","missing","extra","boolbytes","clipbool","hash","producer","reportlink"])
def test_prepare_pins_exact_inventory(gate,fault):
    spec=clip(gate);pins,_,_=preparation_receipt(gate,spec)
    if fault=="schema":pins["schema"]="world-reward-cari96-prepare-pins-v1"
    elif fault=="missing":pins["prepare_files"].pop("target.npy")
    elif fault=="extra":pins["prepare_files"]["GT.npy"]=dict(sha256="e"*64,bytes=4)
    elif fault=="boolbytes":pins["prepare_files"]["target.npy"]["bytes"]=True
    elif fault=="clipbool":pins["clip_spec"]["episode_index"]=True
    elif fault=="hash":pins["prepare_files"]["target.npy"]["sha256"]="x"
    elif fault=="producer":pins["prepare"]["producer_revision"]="a"*39
    else:pins["prepare"]["sha256"]="1"*64
    with pytest.raises(ValueError):gate.validate_artifact_pins(spec,pins,"prepare")


@pytest.mark.parametrize("fault",["GT","status","phase","quality","count","boolcount","frames","indices","lastbatch",
    "dataset","nativefidelity","referencefidelity","referencecount","runtime","source","replay","producer","output"])
def test_prepare_report_failclosed_no_quality_adjustment(gate,fault):
    spec=clip(gate);pins,inputs,report=preparation_receipt(gate,spec)
    if fault=="GT":report["ground_truth_used"]=True
    elif fault=="status":report["status"]="fail"
    elif fault=="phase":report["phase"]="running"
    elif fault=="quality":report["quality_verified"]=True
    elif fault=="count":report["native_geometry_attempts"]-=1
    elif fault=="boolcount":report["optimizer_calls"]=False
    elif fault=="frames":report["frames"]-=1
    elif fault=="indices":report["original_frame_indices"][-1]-=1
    elif fault=="lastbatch":report["chunk_counts"][-1]=16
    elif fault=="dataset":report["input_dataset_revision"]="0"*40
    elif fault=="nativefidelity":report["native_direct_max_point_mm"]=.0100001
    elif fault=="referencefidelity":report["reference_per_frame_mean_mm"][-1]=2.00001
    elif fault=="referencecount":report["reference_per_frame_mean_mm"].pop()
    elif fault=="runtime":report["runtime"]["jit_optimized_execution"]=True
    elif fault=="source":report["source_files"]={}
    elif fault=="replay":report["stored_initializer_native_replay_verified"]=False
    elif fault=="producer":report["producer_revision"]="1"*40
    else:report["output_files"].pop("target.npy")
    with pytest.raises(ValueError):gate.validate_prepare_report(report,spec,pins,inputs)


@pytest.mark.parametrize("fault",["shape","otherblock","oldmetadata","J","dtype","frame","extra"])
def test_shared_initializer_policy_freshgeometry(gate,fault):
    spec=clip(gate);source=original(gate,501);init=selected(gate,source)
    if fault=="shape":init["mhr_shape"][-1,0]=1
    elif fault=="otherblock":init["mhr_trans"][-1,0]+=.001
    elif fault=="oldmetadata":init["metadata"]["projection_reverified_after_identity_change"]=True
    elif fault=="J":init["mhr_joints"]=np.zeros((500,127,3),np.float32)
    elif fault=="dtype":init["mhr_hand"]=init["mhr_hand"].astype(np.float64)
    elif fault=="frame":init["frames"][-1]="000499"
    else:init["mhr_vertices"]=np.zeros((1,3),np.float32)
    with pytest.raises(ValueError):gate.validate_initializer(init,source,spec,dict(native_direct_max_point_mm=.001))


def test_shared_initializer_exact_all501_native_blocks(gate):
    source=original(gate,501);init=selected(gate,source)
    gate.validate_initializer(init,source,clip(gate),dict(native_direct_max_point_mm=.001))
    assert init["metadata"]["historical_original_initializer_checks"]["mhr_geometry_forward_verified"] is True
    assert init["metadata"]["projection_reverified_after_identity_change"] is False


@pytest.mark.parametrize("fault",["extra","missing","symlink","writable","hash"])
def test_four_payload_inventory_all_sha_before_json(gate,tmp_path,fault):
    directory=tmp_path/"producer";directory.mkdir()
    for name in gate.PREPARE_FILES:
        (directory/name).write_bytes(b"fixture");(directory/name).chmod(0o444)
    pins={name:gate.identity(directory/name,immutable=True) for name in gate.PREPARE_FILES}
    path=directory/"target.npy"
    if fault=="extra":(directory/"old_prediction.npy").write_bytes(b"x")
    elif fault=="missing":path.unlink()
    elif fault=="symlink":path.unlink();path.symlink_to(directory/"shared_initializer.pkl")
    elif fault=="writable":path.chmod(0o644)
    else:path.chmod(0o644);path.write_bytes(b"different");path.chmod(0o444)
    with pytest.raises((ValueError,FileNotFoundError)):gate._frozen_inventory(directory,pins)


def test_saved_inventory_identity_has_no_json_deserialization(gate,tmp_path):
    for name in gate.FORWARD_FILES:(tmp_path/name).write_bytes(b"not JSON or pickle");(tmp_path/name).chmod(0o444)
    pins={name:gate.identity(tmp_path/name,immutable=True) for name in gate.FORWARD_FILES}
    assert gate._frozen_inventory(tmp_path,pins)==pins


@pytest.mark.parametrize("ep",[0,15,29])
def test_explicit_episode_single_option(gate,ep):
    assert gate.prepare.parser().parse_args(["--episode",str(ep)]).episode==ep
    assert gate.output_relative(ep)==f"outputs/episode_{ep:06d}/cari_shared_forward_v1"


@pytest.mark.parametrize("args",[[],["--episode","30"],["--episode","-1"],["--ep","15"],["--episode","15","--episode","16"]])
def test_no_implicit_or_duplicate_episode(gate,args):
    with pytest.raises(SystemExit):gate.prepare.parser().parse_args(args)


def test_wrapper_syntax_exact_mounts_no_prefix_private_or_cache():
    wrapper=HERE/"infra/run_cari_full_forward.sh"
    subprocess.run(["bash","-n",str(wrapper)],check=True)
    text=wrapper.read_text()
    assert "source_paths(spec)" in text and "target.npy" in text and "903s" in text and "--network none" in text
    assert "--memory 32g" in text and "--mount \"type=bind,src=$OUT,dst=$OUT\"" in text
    assert "/validation/" not in text and "multiview" not in text and 'src=$ROOT,dst=$ROOT' not in text


def test_native_callable_hook_stack_and_config_source_unchanged_by_architecture():
    tree=ast.parse((HERE/"infra/cari_full_forward.py").read_text())
    text=(HERE/"infra/cari_full_forward.py").read_text()
    assert "sys._getframe(1)" in text and "constrained.clone_native_forward" in text
    assert "use_input_cache=False" in text and "offline_supervision_contract=True" in text
    assert "MHR_INIT_DECODE_BATCH_SIZE!=8" in text and "native_initializer_decode_chunk=8" in text
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=="FRAMES" for t in n.targets) for n in ast.walk(tree))
    assert "torch.use_deterministic_algorithms(False,warn_only=False)" in text
    assert "torch.autocast" not in text  # AMP remains inside the pinned native source.


def complete_bundle(gate,n,monkeypatch):
    capture,bundle,init,poses,report=captured(gate,n)
    pr=bundle["pr"];pr.update(pose_abs=poses["obj_pose_world"].copy(),pose_abs_1st_delta=poses["obj_pose_world"].copy())
    bundle.update(schema="cari4d.mhr_wild_inference.v1",gt={},frames=init["frames"],kid=0,
        frame_meta=[dict(frame=name,src_frame=i,kid=0) for i,name in enumerate(init["frames"])],
        metadata=dict(ground_truth_used=False,window_length=96,window_stride=96,overlap_policy="first_occurrence",
            materialized_input_cache=None,materialized_input_cache_identity=None,depth_backend="moge2",
            object_pose_frame="centered_axis_aligned",object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
            object_pose_storage_frame="output_aligned_mesh_frame",object_pose_storage_to_training_transform=np.eye(4).tolist(),
            object_mesh_to_training_transform=np.eye(4).tolist()),
        config=dict(clip_len=96,enable_amp=True,pred_mhr_shape=True,pred_mhr_scale=False,body_model="mhr",
            mhr_joint_supervision_mode="body12_freeze_hand_face_all_losses"),checkpoint=dict(step=200000),
        pr_initial=copy.deepcopy(pr),**{"in":dict(pose_abs=poses["obj_pose_world"].copy(),**{key:init[key].copy() for key in gate.NATIVE_PARAMETER_DIMS})},
        faces=np.zeros((36874,3),np.int32),mhr_neutral_height_init=np.full(n,1.8,np.float32),
        mhr_spatial_scale=np.full(n,2/1.8,np.float32),mesh_diameter=np.full(n,2,np.float32),
        K_rois=np.tile(np.eye(3,dtype=np.float32),(n,1,1)),bboxes=np.tile(np.array([0,0,224,224],np.float32),(n,1)),
        observations=dict(human_mask=np.zeros((n,224,224),bool),object_mask=np.zeros((n,224,224),bool),
            K_full=np.tile(gate.inputs.inferred_camera(clip(gate,n)).astype(np.float32),(n,1,1)),
            postopt_K_rois=np.tile(np.eye(3,dtype=np.float32),(n,1,1)),
            postopt_crop_contract="cari4d.smplh_postopt_full_resolution_crop.v1",
            postopt_human_mask=np.zeros((n,256,256),np.float32),postopt_object_mask=np.zeros((n,256,256),np.float32)))
    monkeypatch.setattr(gate.assets,"FACE_SHA",hashlib.sha256(bundle["faces"].tobytes()).hexdigest())
    report.update(native_compose_global_unchanged=True,native_function_code_unchanged=True)
    return capture,bundle,init,poses,report


def test_practical_full501_bundle_saved_reread_native_byte_assembly(gate,monkeypatch):
    capture,bundle,init,poses,report=complete_bundle(gate,501,monkeypatch)
    evidence=gate.validate_bundle(bundle,init,poses,clip(gate),report,capture)
    assert evidence["window_owned_counts"]==[96,96,96,96,96,21]
    assert gate.validate_bundle(copy.deepcopy(bundle),init,poses,clip(gate),report,capture)==evidence
    assert not bundle["observations"]["human_mask"].any()  # No per-frame deletion/fabrication.


@pytest.mark.parametrize("fault",["face","negativeT","root","hand","negativezero","contact","gauge","K","neutral","crop","mask","identity"])
def test_full_bundle_native_abi_cannot_be_repaired(gate,monkeypatch,fault):
    capture,bundle,init,poses,report=complete_bundle(gate,97,monkeypatch)
    if fault=="face":bundle["pr"]["mhr_face"][0,0]=1
    elif fault=="negativeT":bundle["pr"]["mhr_trans"][0,2]=-1
    elif fault=="root":bundle["pr"]["mhr_global_rot6d"][0]=0
    elif fault=="hand":bundle["raw"]["delta_mhr_hand"][0,0]=1e-10
    elif fault=="negativezero":bundle["raw"]["delta_mhr_hand"][:]=-0.
    elif fault=="contact":bundle["pr_initial"]["contact_logits"][0,0]+=1
    elif fault=="gauge":bundle["in"]["pose_abs"][0,0,0]=-1
    elif fault=="K":bundle["observations"]["K_full"][0,0,0]+=1
    elif fault=="neutral":bundle["mhr_neutral_height_init"][0]+=1e-5
    elif fault=="crop":bundle["bboxes"][0]=0
    elif fault=="mask":bundle["observations"]["human_mask"]=bundle["observations"]["human_mask"].astype(np.float32)
    else:bundle["pr"]["mhr_scale"][-1,0]+=1
    with pytest.raises(ValueError):gate.validate_bundle(bundle,init,poses,clip(gate,97),report,capture)


def forward_receipt(gate,capture,bundle):
    n=capture.spec.total_frames;report=dict(capture.report);report.update(capture.verify(bundle))
    report.update(stage=gate.STAGE,status="pass",phase="complete",episode_index=15,clip_spec=asdict(capture.spec),frames=n,source_frames=n,
        original_frame_indices=list(range(n)),input_track="track_1",ground_truth_used=False,private_truth_read=False,
        hand_labeled_test=False,oracle_modes=[],network="none",forward_attempts=1,forward_returns=1,forward_validated=1,
        hub_attempts=2,hub_returns=2,shared_identity_verified=True,native_global_unchanged=True,hub_restored=True,
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,stored_bundle_reread_verified=True,quality_verified=False,
        adoption_authorized=False,refinement_performed=False,unchanged_original_CARI_method=False,MHR_strict_scope_performed=False,
        exact_deterministic_forward_claim=False,submission_produced=False,input_dataset_revision=gate.inputs.DATASET_REVISION,
        native_enable_amp=True,deterministic_algorithms=False,warn_only=False,TF32=False,image_id=gate.IMAGE,budget_seconds=gate.BUDGET,
        native_source_sha256=gate.constrained.NATIVE_SOURCE_SHA256,upstream_revision=gate.constrained.UPSTREAM_REVISION,
        native_function_code_unchanged=True,isolated_function_globals=True,overridden_globals=["compose_mhr_output"],
        native_compose_global_unchanged=True,native_global_replacement_performed=False,native_global_restoration_required=False,
        native_initializer_decode_chunk=8,native_initializer_decode_chunk_counts=[s.stop-s.start for s in gate.finite_chunks(n,8)])
    return report


def test_practical_full501_forward_json_finite_consumer_receipt(gate):
    capture,bundle,*_=captured(gate,501);report=forward_receipt(gate,capture,bundle)
    gate.validate_forward_report(report,capture.spec)
    json.dumps(report,allow_nan=False)
    assert report["native_initializer_decode_chunk_counts"][-1]==5


@pytest.mark.parametrize("fault",["count","hub","quality","native","drop","overlap","indices","missingraw","fp32","owner","scale","chunk"])
def test_actual_forward_report_no_partial_native_evidence(gate,fault):
    capture,bundle,*_=captured(gate,97);report=forward_receipt(gate,capture,bundle)
    if fault=="count":report["composition_verified_calls"]-=1
    elif fault=="hub":report["hub_attempts"]=3
    elif fault=="quality":report["quality_verified"]=True
    elif fault=="native":report["native_source_sha256"]="1"*64
    elif fault=="drop":report["window_fingerprints"].pop()
    elif fault=="overlap":report["window_owned_counts"]=[96,96]
    elif fault=="indices":report["window_fingerprints"][-1]["indices"]["sha256"]="1"*64
    elif fault=="missingraw":report["window_fingerprints"][-1]["raw"].pop("delta_mhr_hand")
    elif fault=="fp32":report["window_fingerprints"][-1]["composed"]["mhr_shape"]["dtype"]="<f8"
    elif fault=="owner":report["owner_local"]["sha256"]="1"*64
    elif fault=="scale":report["window_fingerprints"][-1]["raw"]["delta_mhr_scale"]=gate.fingerprint(np.zeros((96,28),np.float32))
    else:report["native_initializer_decode_chunk_counts"][-1]=8
    with pytest.raises(ValueError):gate.validate_forward_report(report,capture.spec)


def full_chain_fixture(gate,tmp_path,monkeypatch,n=97):
    spec=clip(gate,n);pins,input_pins,prepared=preparation_receipt(gate,spec)
    root=tmp_path/"root";code=tmp_path/"code";code.mkdir();root.mkdir()
    directory=root/gate.prepare.output_relative(15);directory.mkdir(parents=True)
    for path in input_pins["source_files"]:
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"tiny public fixture")
    input_pins["source_files"]={p:gate.identity(root/p) for p in input_pins["source_files"]}
    input_pins["input_report"].update(input_pins["source_files"][gate.inputs.relative_paths(spec)["input_report"]])
    for name in gate.prepare.OUTPUTS:(directory/name).write_bytes(b"tiny opaque native payload");(directory/name).chmod(0o444)
    source=code/"infra/cari_shared_prepare.py";source.parent.mkdir();source.write_bytes(b"actual synthetic generator");source.chmod(0o444)
    forward_source=code/"infra/cari_full_forward.py";forward_source.write_bytes(b"actual synthetic forward generator");forward_source.chmod(0o444)
    helpers={"infra/cari_shared_prepare.py":gate.identity(source)}
    monkeypatch.setattr(gate.prepare,"source_helpers",lambda c:helpers)
    forward_helpers=helpers|{"infra/cari_full_forward.py":gate.identity(forward_source)}
    monkeypatch.setattr(gate,"source_helpers",lambda c:forward_helpers)
    prepared.update(script_sha256=helpers["infra/cari_shared_prepare.py"]["sha256"],source_files=input_pins["source_files"],
        input_report_sha256=input_pins["input_report"]["sha256"],source_inputs_report_sha256=input_pins["input_report"]["sha256"],
        source_helpers=helpers,input_sha256="9"*64,body_assets={"model":"synthetic"},inference_source_identity={"body":"synthetic"},decoder_identity={"decoder":"synthetic"},
        original_input_paths={name:str(root/path) for name,path in gate.inputs.relative_paths(spec).items()})
    config=code/"configs/cari_clip_000015_input_pins.json";config.parent.mkdir();config.write_text(json.dumps(input_pins));config.chmod(0o444)
    prepared["input_pins"]=gate.identity(config)
    binding_paths=[root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py",
        root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py",root/"weights/mhr/mhr_model.pt"]
    for path in binding_paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"tiny binding")
    def binding(path,digest):return dict(path=str(path),**gate.identity(path))
    monkeypatch.setattr(gate.prepare.geometry,"binding",binding)
    prepared["source_bindings"]=[binding(path,"ignored synthetic") for path in binding_paths]
    monkeypatch.setattr(gate.inputs,"validate_reports",lambda *_:dict(inputs={"input_sha256":"9"*64},body={"body_assets":prepared["body_assets"]},
        adapter={"inference_source_identity":prepared["inference_source_identity"],"decoder_identity":prepared["decoder_identity"]}))
    files={name:gate.identity(directory/name) for name in gate.prepare.OUTPUTS}
    prepared["output_files"]=files
    for key,path in (("shared_initializer_sha256","shared_initializer.pkl"),("direct_parameters_sha256","direct_parameters.npz"),("target_sha256","target.npy")):prepared[key]=files[path]["sha256"]
    (directory/"report.json").write_text(json.dumps(prepared));(directory/"report.json").chmod(0o444)
    pins["prepare_files"]=files|{"report.json":gate.identity(directory/"report.json")}
    pins["prepare"]=pins["prepare_files"]["report.json"]|{k:prepared[k] for k in ("producer_revision","script_sha256")}
    prepare_pin=code/"configs/cari_clip_000015_shared_prepare_pins.json";prepare_pin.write_text(json.dumps(pins));prepare_pin.chmod(0o444)
    capture,bundle,*_=captured(gate,n);report=forward_receipt(gate,capture,bundle)
    forward_dir=root/gate.output_relative(15);forward_dir.mkdir()
    (forward_dir/"coconet.pth").write_bytes(b"not loaded by hash JSON-only consumer");(forward_dir/"coconet.pth").chmod(0o444)
    report.update(producer_revision="6"*40,script_sha256=gate.identity(forward_source)["sha256"],input_pins=gate.identity(config),
        prepare_pins=gate.identity(prepare_pin),prepare_report_sha256=pins["prepare"]["sha256"],prepare_files=pins["prepare_files"],
        public_source_files=input_pins["source_files"],source_helpers=forward_helpers,original_input_paths=prepared["original_input_paths"],
        input_sha256=prepared["input_sha256"],decoder_identity=prepared["decoder_identity"],body_assets=prepared["body_assets"],
        inference_source_identity=prepared["inference_source_identity"],checkpoint_sha256=gate.CHECKPOINT_SHA256,config_sha256=gate.assets.SOURCES[gate.CONFIG_RELATIVE_PATH],
        bundle_sha256=gate.identity(forward_dir/"coconet.pth")["sha256"],bundle_bytes=gate.identity(forward_dir/"coconet.pth")["bytes"],
        output_files={"coconet.pth":gate.identity(forward_dir/"coconet.pth")})
    native=root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    report["source_files"]={str(native/path):dict(sha256=digest,bytes=1) for path,digest in gate.assets.SOURCES.items()}
    report["source_files"][str(root/"weights/cari4d/cari4d"/gate.assets.CHECKPOINT_RELATIVE_PATH)]=dict(sha256=gate.CHECKPOINT_SHA256,bytes=1)
    home=root/"weights/cari4d/sam3d_body/torch_home"
    report["source_files"].update({str(home/"hub/checkpoints"/name):dict(sha256=digest,bytes=1) for name,digest in gate.assets.DINO_HASHES.items()})
    report["source_files"].update({str(root/"results"/name):dict(sha256="1"*64,bytes=1) for name in ("weights-acquisition.json","auxiliary-assets.json")})
    report["source_files"][str(home/"hub/facebookresearch_dinov2_main/hubconf.py")]=dict(sha256="1"*64,bytes=1)
    def save_forward():
        path=forward_dir/"report.json"
        if path.exists():path.chmod(0o644)
        path.write_text(json.dumps(report));path.chmod(0o444)
        row=gate.identity(path)
        return dict(schema="world-reward-cari-shared-forward-pins-v1",clip_spec=asdict(spec),
            forward=row|{k:report[k] for k in ("producer_revision","script_sha256")},
            forward_files=report["output_files"]|{"report.json":row})
    forward_pins=save_forward()
    return root,code,spec,pins,input_pins,forward_pins,report,save_forward


def test_practical_downstream_full_chain_hash_json_only_no_model_pickle(gate,tmp_path,monkeypatch):
    root,code,spec,pins,inputs,forward_pins,report,_=full_chain_fixture(gate,tmp_path,monkeypatch)
    monkeypatch.setitem(sys.modules,"joblib",SimpleNamespace(load=lambda *_:pytest.fail("Hash/JSON consumer must never unpickle")))
    first=gate.verify_prepare_artifacts(root,code,spec,pins,inputs)
    value=gate.verify_forward_artifacts(root,code,spec,forward_pins)
    assert set(first["files"])==gate.PREPARE_FILES and len(first["source_files"])==15
    assert set(value["files"])==gate.FORWARD_FILES and value["prepare"]["files"]==first["files"]
    assert all(Path(path).is_absolute() for path in value["bindings"])
    assert len(value["bindings"])==28  # Original15 + prepared4 + forward2 + helpers2 + pins2 + native/reference3.


@pytest.mark.parametrize("fault",["body","input","source","dino","unknown","helper","prepared"])
def test_full_chain_strict_actual_generator_asset_source_bindings(gate,tmp_path,monkeypatch,fault):
    root,code,spec,_,_,_,report,save=full_chain_fixture(gate,tmp_path,monkeypatch)
    if fault=="body":report["body_assets"]={}
    elif fault=="input":report["input_sha256"]="0"*64
    elif fault=="source":next(iter(report["source_files"].values()))["sha256"]="0"*64
    elif fault=="dino":report["source_files"]={p:v for p,v in report["source_files"].items() if not p.endswith("hubconf.py")}
    elif fault=="unknown":report["source_files"][str(root/"old_predictions.pkl")]=dict(sha256="0"*64,bytes=1)
    elif fault=="helper":report["source_helpers"]={}
    else:report["prepare_report_sha256"]="0"*64
    pins=save()
    with pytest.raises(ValueError):gate.verify_forward_artifacts(root,code,spec,pins)

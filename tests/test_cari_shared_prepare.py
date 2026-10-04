"""Tiny full-N native callback contracts; no Torch/model/media/Azure access."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "infra"))
    path = root / "infra/cari_shared_prepare.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_shared_prepare", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def original(gate, count=97):
    params = {key:np.zeros((count,dim),np.float32) for key,dim in gate.NATIVE_PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:]=[1,0,0,1,0,0]
    params["mhr_trans"][:,2]=np.linspace(1.,2.,count,dtype=np.float32)
    params["mhr_shape"][:,0]=np.linspace(.1,.5,count,dtype=np.float32)
    params["mhr_scale"][:,0]=np.linspace(.2,.8,count,dtype=np.float32)
    params["mhr_hand"][:,0]=np.arange(count,dtype=np.float32)/100
    return params | dict(body_model="mhr",frames=[f"{i:06d}" for i in range(count)],kids=[0],
        mhr_joints=np.full((count,2,3),22,np.float32),mhr_keypoints=np.full((count,4,3),33,np.float32),
        metadata=dict(ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],mhr_geometry_forward_verified=True,
                      native_roundtrip_max_error_m={"vertices":.1},translation_once_max_error_m=.2,projection_max_error_px=.3,
                      camera_intrinsics=[[1920,0,768],[0,1920,576],[0,0,1]]))


def report(gate):
    return {name+suffix:0 for name in gate.CALLS for suffix in ("_attempts","_returns","_validated")}


FACES=np.array([[0,1,2]],np.int64)
FACE_SHA=hashlib.sha256(FACES.astype("<i4").tobytes()).hexdigest()


def shape_from_params(params):
    n=len(params["mhr_trans"])
    points=np.zeros((n,3,3),np.float32)
    points[:,:,2]=params["mhr_trans"][:,2,None]
    points[:,1,0]=.1
    points[:,2,1]=.1
    return points


def callbacks(events=None,fault=None):
    events=[] if events is None else events
    native_calls=0
    def native(params):
        nonlocal native_calls
        native_calls+=1
        events.append(("native",len(params["mhr_trans"]),{k:v.copy() for k,v in params.items()}))
        pts=shape_from_params(params)
        joints=np.repeat(pts[:,:1],2,axis=1)
        kp=np.repeat(pts[:,:1],4,axis=1)
        result={"vertices":pts,"joints":joints,"keypoints":kp,"faces":FACES.copy()}
        if fault=="nonfinite":result["vertices"][0,0,0]=np.nan
        elif fault=="negativegeometry":result["joints"][0,0,2]=0
        elif fault=="native_dtype":result["vertices"]=result["vertices"].astype(np.float64)
        elif fault=="face_dtype":result["faces"]=result["faces"].astype(np.int32)
        elif fault=="topology":result["faces"][0]=[0,2,1]
        elif fault=="native_mutation":params["mhr_hand"][0,0]+=1
        elif fault=="saved_replay" and native_calls>7:result["vertices"][0,0,0]+=.01
        elif fault=="saved_joints" and native_calls>7:result["joints"][0,0,0]+=.01
        elif fault=="saved_keypoints" and native_calls>7:result["keypoints"][0,0,0]+=.01
        return result
    def direct(params):
        events.append(("direct",len(params["mhr_trans"]),{k:v.copy() for k,v in params.items()}))
        pts=shape_from_params(params)
        dc=np.zeros((len(pts),204),np.float32)
        dc[:,2]=params["mhr_trans"][:,2]
        dc[:,136:]=np.repeat(params["mhr_scale"][:,0,None],68,axis=1)
        pca=dc[:,136:].copy()
        if fault=="direct_error":pts[0,0,0]+=.01
        elif fault=="direct_dtype":dc=dc.astype(np.float64)
        elif fault=="pca":pca[0,0]+=.1
        elif fault=="vary_scale":dc[0,136:]+=.01;pca=dc[:,136:].copy()
        elif fault=="direct_mutation":params["mhr_hand"][0,0]+=1
        return pts,dc,pca
    def reference(stored,selection):
        events.append(("reference",selection.stop-selection.start,stored["frame_index"][selection].copy()))
        pts=np.zeros((selection.stop-selection.start,3,3),np.float32)
        pts[:,:,2]=stored["pose"][selection,2,None]
        pts[:,1,0]=.1;pts[:,2,1]=.1
        result=pts.astype(np.float64)*1000.
        if fault=="reference_error":result+=3.
        elif fault=="reference_nonfinite":result[0,0,0]=np.nan
        elif fault=="reference_mutation":stored["pose"][0,0]=1
        return result
    return native,direct,reference


def run_core(gate,out,count=97,fault=None,original_data=None,events=None,receipt=None,persist=None):
    receipt=report(gate) if receipt is None else receipt
    data=original(gate,count) if original_data is None else original_data
    functions=callbacks(events,fault)
    result=gate.prepare_geometry(data,count,out,*functions,receipt,persist or (lambda:None),
        vertices=3,joint_count=2,keypoint_count=4,face_count=1,face_sha=FACE_SHA)
    return result,receipt,data


@pytest.mark.parametrize("count,expected",[(96,[16]*6),(97,[16]*6+[1]),(501,[16]*31+[5]),(668,[16]*41+[12])])
def test_fullN_original_order_tail_and_four_actual_routes_before_complete(gate,tmp_path,count,expected):
    events=[];data=original(gate,count);before=pickle.dumps(data,protocol=4)
    frozen,receipt,_=run_core(gate,tmp_path,count,original_data=data,events=events)
    assert before==pickle.dumps(data,protocol=4)
    assert set(frozen)==gate.OUTPUTS
    assert receipt["chunk_counts"]==expected
    assert all(receipt[name+suffix]==len(expected) for name in gate.CALLS for suffix in ("_attempts","_returns","_validated"))
    assert receipt["phase"]=="geometry_routes_complete"
    assert receipt["identity_fixed_before_first_decode"] is True
    assert receipt["reference_per_frame_mean_mm"]==[0.]*count
    assert receipt["original_frame_indices"]==list(range(count))
    assert [size for role,size,_ in events if role=="direct"]==expected
    assert [size for role,size,_ in events if role=="native"]==expected*2
    assert [size for role,size,_ in events if role=="reference"]==expected
    firsts=[params for role,_,params in events if role=="native"][:len(expected)]
    combined={key:np.concatenate([batch[key] for batch in firsts]) for key in gate.NATIVE_PARAMETER_DIMS}
    for key in gate.NATIVE_PARAMETER_DIMS:
        if key in ("mhr_shape","mhr_scale"):
            assert all(row.tobytes()==data[key][0].tobytes() for row in combined[key])
        else:assert combined[key].tobytes()==data[key].tobytes()
    with (tmp_path/"shared_initializer.pkl").open("rb") as stream:saved=pickle.load(stream)
    assert saved["mhr_joints"].shape==(count,2,3) and saved["mhr_keypoints"].shape==(count,4,3)
    assert not np.array_equal(saved["mhr_joints"],data["mhr_joints"])
    meta=saved["metadata"]
    assert meta["mhr_geometry_forward_verified"] is True
    assert meta["joint_keypoint_redecode_required"] is False and meta["shared_joint_keypoint_native_redecoded"] is True
    assert meta["source_joint_keypoint_reuse_permitted"] is False
    assert meta["historical_original_initializer_checks"]["mhr_geometry_forward_verified"] is True
    assert "projection_max_error_px" not in meta and meta["projection_reverified_after_identity_change"] is False
    assert meta["quality_verified"] is False and meta["submission_eligible"] is False
    assert all(not (tmp_path/name).stat().st_mode&0o222 for name in frozen)
    assert {name:gate.inputs.identity(tmp_path/name) for name in frozen}==frozen


@pytest.mark.parametrize("fault",["nonfinite","negativegeometry","native_dtype","face_dtype","topology","native_mutation",
                                  "direct_error","direct_dtype","pca","vary_scale","direct_mutation","saved_replay",
                                  "saved_joints","saved_keypoints","reference_error","reference_nonfinite","reference_mutation"])
def test_geometry_direct_saved_and_reference_faults_fail_without_native_claim(gate,tmp_path,fault):
    receipt=report(gate)
    with pytest.raises(ValueError):run_core(gate,tmp_path,fault=fault,receipt=receipt)
    assert receipt.get("phase")!="geometry_routes_complete"
    assert "stored_initializer_native_replay_verified" not in receipt


@pytest.mark.parametrize("extra",["target.npy","old_vertices","mhr_neutral_height_init","cached_geometry"])
def test_original_initializer_extras_cannot_reuse_stale_decoded_geometry(gate,tmp_path,extra):
    data=original(gate);data[extra]=np.zeros((97,3),np.float32)
    events=[]
    with pytest.raises(ValueError):run_core(gate,tmp_path,original_data=data,events=events)
    assert events==[] and list(tmp_path.iterdir())==[]


def test_invalid_identity_fails_before_first_native_decode(gate,tmp_path):
    data=original(gate);data["mhr_global_rot6d"][0]=0
    events=[]
    with pytest.raises(ValueError):run_core(gate,tmp_path,original_data=data,events=events)
    assert events==[]


def test_frozen_outputs_exist_before_native_saved_and_official_consumer_calls(gate,tmp_path):
    receipt=report(gate);observed=[]
    def persist():
        if receipt.get("phase") in ("stored_initializer_native_replay","official_reference_replay"):
            assert all((tmp_path/name).is_file() for name in gate.OUTPUTS)
            assert all(not (tmp_path/name).stat().st_mode&0o222 for name in gate.OUTPUTS)
            observed.append(receipt["phase"])
    run_core(gate,tmp_path,receipt=receipt,persist=persist)
    assert "stored_initializer_native_replay" in observed and "official_reference_replay" in observed


def test_actual_reference_reads_frozen_direct_archive_not_original_parameters(gate,tmp_path,monkeypatch):
    freeze=gate.freeze
    saved_calls=[]
    def capture(*args):
        reread,stored,target,frozen=freeze(*args)
        saved_calls.append(stored)
        return reread,stored,target,frozen
    monkeypatch.setattr(gate,"freeze",capture)
    run_core(gate,tmp_path)
    assert set(saved_calls[0])=={"frame_index","pose","scales","shape"}
    assert saved_calls[0]["frame_index"].dtype==np.int64


def test_mutated_frozen_target_after_replay_is_detected(gate,tmp_path):
    receipt=report(gate);mutated=False
    def persist():
        nonlocal mutated
        if receipt.get("phase")=="official_reference_replay" and not mutated:
            mutated=True
            path=tmp_path/"direct_parameters.npz";path.chmod(0o644);path.write_bytes(b"corrupt frozen archive")
    with pytest.raises(ValueError,match="frozen shared artifacts changed"):
        run_core(gate,tmp_path,receipt=receipt,persist=persist)


def test_attempt_return_and_validation_counters_distinguish_failed_call(gate):
    receipt=report(gate)
    def failed():raise RuntimeError("failed native forward")
    with pytest.raises(RuntimeError):gate.call(receipt,"native_geometry",failed,lambda:None)
    assert receipt["native_geometry_attempts"]==1
    assert receipt["native_geometry_returns"]==receipt["native_geometry_validated"]==0


@pytest.mark.parametrize("bad",[True,-1,30,15.])
def test_output_path_has_explicit_bounded_episode(gate,bad):
    with pytest.raises(ValueError):gate.output_relative(bad)


def test_argument_requires_one_explicit_episode_and_no_runtime_alternative(gate):
    assert gate.parser().parse_args(["--episode","15"]).episode==15
    for args in ([],["--episode","30"],["--episode","15","--GT"],["--episode","15","--chunk","1"]):
        with pytest.raises(SystemExit):gate.parser().parse_args(args)
    with pytest.raises(SystemExit):gate.parser().parse_args(["--episode","15","--episode","15"])


def test_native_face_dtype_winding_indices_hash_are_not_repaired(gate):
    assert gate.checked_faces(FACES,vertices=3,count=1,digest=FACE_SHA) is FACES
    for value in (FACES.astype(np.int32),np.array([[0,2,1]],np.int64),np.array([[0,1,3]],np.int64),np.ma.array(FACES,mask=False)):
        with pytest.raises(ValueError):gate.checked_faces(value,vertices=3,count=1,digest=FACE_SHA)


def test_full_native_gate_and_budget_are_same_fixed_fidelity_not_quality(gate):
    assert gate.BUDGET==600 and gate.CHUNK==16
    assert gate.STAGE=="world_reward_native_cari_shared_initializer_full_video"
    assert gate.OUTPUTS=={"shared_initializer.pkl","direct_parameters.npz","target.npy"}
    source=Path(gate.__file__).read_text()
    assert "precision=\"float32\"" in source and "torch.use_deterministic_algorithms(True,warn_only=False)" in source
    assert "torch.jit.optimized_execution(False)" in source
    assert "converter_LM_calls=0" in source and "learned_inference_calls=0" in source
    assert "source_files=public[\"source_files\"]" in source and "original_input_paths" in source
    assert source.index("before_models=(share_first_frame_identity") < source.index("import torch")


def test_occupied_output_fails_before_any_native_geometry_call(gate,tmp_path):
    (tmp_path/"prior.json").write_text("unexpected frozen output")
    events=[]
    with pytest.raises(ValueError,match="Fresh shared output"):
        run_core(gate,tmp_path,events=events)
    assert events==[]


def test_unexpected_new_output_during_replay_is_rejected(gate,tmp_path):
    receipt=report(gate);created=False
    def persist():
        nonlocal created
        if receipt.get("phase")=="official_reference_replay" and not created:
            created=True;(tmp_path/"unexpected.pt").write_bytes(b"not an allowed payload")
    with pytest.raises(ValueError,match="frozen shared artifacts changed"):
        run_core(gate,tmp_path,receipt=receipt,persist=persist)


def test_helper_manifest_complete_and_byte_bound_in_readonly_code_snapshot(gate,tmp_path):
    names=("infra/cari_shared_prepare.py","infra/run_cari_shared_prepare.sh","infra/cari_clip_inputs.py",
           "infra/cari96_prepare.py","infra/run_cari96_prepare.sh","infra/cari96_inputs.py","infra/body_smoke.py",
           "src/world_reward/shared_identity.py","src/world_reward/timeline.py","src/world_reward/data.py")
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name);path.chmod(0o444)
    rows=gate.source_helpers(tmp_path)
    assert set(rows)==set(names)
    assert rows=={name:gate.inputs.identity(tmp_path/name) for name in names}
    path=tmp_path/names[0];path.chmod(0o644)
    with pytest.raises(ValueError,match="Immutable"):
        gate.source_helpers(tmp_path)


@pytest.mark.parametrize("fault",[None,"parameters","metadata"])
def test_identity_fixed_before_model_construction_must_match_pure_policy(gate,tmp_path,fault):
    data=original(gate)
    shared=gate.share_first_frame_identity({key:data[key] for key in gate.NATIVE_PARAMETER_DIMS},97)
    meta=gate.shared_initializer_metadata(data["metadata"],97)
    if fault=="parameters":shared["mhr_shape"][0,0]+=.1
    elif fault=="metadata":meta["identity_selection_frame_index"]=1
    functions=callbacks();receipt=report(gate)
    def invoke():
        return gate.prepare_geometry(data,97,tmp_path,*functions,receipt,lambda:None,
            vertices=3,joint_count=2,keypoint_count=4,face_count=1,face_sha=FACE_SHA,preselected=(shared,meta))
    if fault:
        with pytest.raises(ValueError,match="Before-model"):
            invoke()
        assert all(value==0 for value in receipt.values())
    else:
        invoke()
        assert receipt["frames"]==97


def test_wrapper_bash_syntax_and_no_argument_paths_fail_before_cloud_cli(gate):
    path=Path(gate.__file__).with_name("run_cari_shared_prepare.sh")
    result=subprocess.run(["rtk","proxy","bash","-n",str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for arguments in ([],["--episode","30"],["--episode","015"],["--episode","15","--overwrite"]):
        result=subprocess.run(["rtk","proxy","bash",str(path),*arguments],capture_output=True,text=True,env={"PATH":os.environ["PATH"]})
        assert result.returncode==2
    source=path.read_text()
    assert "--network none" in source and "--gpus all" in source
    assert "603s docker run" in source and "read -r relative" in source
    assert "source_paths(spec,object_source=source_profile(pins))" in source and "validate_pins(spec,pins)" in source
    assert "src=$ROOT/outputs,dst=$ROOT/outputs" not in source
    assert "src=$ROOT/data" not in source and "track_2" not in source and "track_3" not in source

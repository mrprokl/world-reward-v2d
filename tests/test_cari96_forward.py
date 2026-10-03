"""Producer-shaped native96 contracts and isolated fake runtime; no assets/GPU."""
import ast
import copy
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("cari96_forward_test",ROOT/"infra/cari96_forward.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def initializer(gate):
    v={k:np.zeros((96,d),np.float32) for k,d in gate.inputs.PARAMETER_DIMS.items()}
    v["mhr_trans"][:,2]=2;v["mhr_global_rot6d"][:]=[1,0,0,1,0,0]
    v.update(body_model="mhr",frames=[f"{i:06d}" for i in range(96)],kids=[0],mhr_joints=np.ones((96,127,3),np.float32),
        mhr_keypoints=np.ones((96,70,3),np.float32),metadata=dict(human_identity_clip_constant=True,original_geometry_recovered=False,
        source_frames=501,original_frame_indices=list(range(96)),ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]))
    return v


def fixture_prepare(gate,root,monkeypatch):
    directory=root/gate.PREPARE;directory.mkdir(parents=True)
    for p in gate.PREPARE_FILES:
        path=directory/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"own fixture payload")
    init=initializer(gate);pose=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy()
    objects=dict(frames=init["frames"],obj_pose_world=pose)
    (directory/"shared_initializer.pkl").write_bytes(pickle.dumps(init))
    (directory/"inputs/own_object_poses.pkl").write_bytes(pickle.dumps(objects))
    report=dict(stage="public_cari96_shared_initializer_native_abi",status="pass",phase="complete",frames=96,image_id=gate.IMAGE,
        input_track="track_1",ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",
        quality_verified=False,adoption_authorized=False,submission_produced=False,shared_identity_verified=True,
        original_initializer_unchanged=True,source_inputs_assets_rehashed=True,stored_initializer_native_replay_verified=True,
        learned_inference_calls=0,optimizer_calls=0,native_geometry_calls=6,native_direct_calls=6,native_replay_calls=6,reference_calls=6,
        native_geometry_attempts=6,native_direct_attempts=6,native_replay_attempts=6,reference_attempts=6,source_helpers_rehashed=True,frozen_outputs_rehashed_after_reference=True,
        producer_revision="a"*40,script_sha256=gate.inputs.sha256(ROOT/"infra/cari96_prepare.py"),reference_per_frame_mean_mm=[.001]*96,
        inference_source_identity={"sha256":"b"*64},body_assets={"own fixture":{}})
    report["source_helpers"]={p:gate.inputs.identity(ROOT/p) for p in ("infra/cari96_prepare.py","infra/cari96_inputs.py","infra/body_smoke.py","infra/run_cari96_prepare.sh")}
    report["source_bindings"]=[dict(path="/native/lib_mhr/mhr_layer.py",sha256=gate.SOURCES["lib_mhr/mhr_layer.py"],bytes=20514),dict(path="/kit/mesh_to_mhr_params.py",sha256="c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e",bytes=1),dict(path="/model/mhr_model.pt",sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",bytes=1)]
    real_identity=gate.identity
    monkeypatch.setattr(gate,"identity",lambda p,immutable=False:real_identity(p,immutable=False if Path(p).is_relative_to(ROOT) else immutable))
    snap=dict(stage="world_reward_public_cari96_inputs_snapshot",status="pass",frames=96,source_frames=501,source_episode_index=15,
        original_frame_indices=list(range(96)),ground_truth_used=False,ground_truth_read=False,hand_labeled_test=False,oracle_modes=[],
        no_padding_or_reencoding=True,initializer_modified=False,camera_name=gate.CAMERA,stored_payload_and_attribute_reread_verified=True)
    source_pins=json.loads((ROOT/"configs/cari96_input_pins.json").read_text())
    report["input_pins"]=gate.identity(ROOT/"configs/cari96_input_pins.json")
    snap["source_files"]=source_pins["source_files"];snap["source_inputs_report_sha256"]=source_pins["source_files"][gate.inputs.REPORT]["sha256"]
    snap["output_files"]={p.removeprefix("inputs/"):gate.identity(directory/p) for p in gate.PREPARE_FILES if p.startswith("inputs/")and p!="inputs/manifest.json"}
    (directory/"inputs/manifest.json").write_text(json.dumps(snap))
    for k,p in {"shared_initializer_sha256":"shared_initializer.pkl","direct_parameters_sha256":"direct_parameters.npz",
        "target_sha256":"target.npy","snapshot_manifest_sha256":"inputs/manifest.json"}.items():report[k]=gate.inputs.sha256(directory/p)
    report["output_files"]={p:gate.identity(directory/p) for p in gate.PREPARE_FILES if p!="report.json"}
    (directory/"report.json").write_text(json.dumps(report))
    for p in directory.rglob("*"):
        if p.is_file():p.chmod(0o444)
    files={p:gate.identity(directory/p)for p in gate.PREPARE_FILES}
    pins=dict(schema="world-reward-cari96-prepare-pins-v1",prepare=files["report.json"]|{k:report[k]for k in("producer_revision","script_sha256")},prepare_files=files)
    monkeypatch.setitem(sys.modules,"joblib",SimpleNamespace(load=lambda p:pickle.loads(Path(p).read_bytes())))
    return directory,pins,report,init,objects


def fixture_bundle(gate,init,objects,monkeypatch):
    params={k:init[k].copy()for k in gate.inputs.PARAMETER_DIMS};params["mhr_body_pose_cont"][:,0]=.02
    params["contact_logits"]=np.zeros((96,2),np.float32)
    predicted=dict(pose_abs=objects["obj_pose_world"].copy(),pose_abs_1st_delta=objects["obj_pose_world"].copy(),**params)
    faces=np.zeros((36874,3),np.int32);monkeypatch.setattr(gate,"FACE_SHA",hashlib.sha256(faces.tobytes()).hexdigest())
    bundle=dict(schema="cari4d.mhr_wild_inference.v1",frames=init["frames"],kid=0,gt={},
        frame_meta=[dict(frame=n,src_frame=i,kid=0)for i,n in enumerate(init["frames"])],checkpoint={"step":200000},
        config=dict(clip_len=96,enable_amp=True,pred_mhr_shape=True,pred_mhr_scale=False,body_model="mhr",mhr_joint_supervision_mode="body12_freeze_hand_face_all_losses"),
        metadata=dict(ground_truth_used=False,window_length=96,window_stride=96,overlap_policy="first_occurrence",
            materialized_input_cache=None,materialized_input_cache_identity=None,depth_backend="moge2"),
        pr=predicted,pr_initial=copy.deepcopy(predicted),**{"in":dict(pose_abs=objects["obj_pose_world"].copy(),**{k:init[k].copy()for k in gate.inputs.PARAMETER_DIMS})},
        faces=faces,mhr_neutral_height_init=np.full(96,1.8,np.float32),mhr_spatial_scale=np.full(96,2/1.8,np.float32),mesh_diameter=np.full(96,2.,np.float32),
        K_rois=np.broadcast_to(np.eye(3,dtype=np.float32),(96,3,3)).copy(),bboxes=np.tile(np.array([0,0,224,224],np.float32),(96,1)),
        observations=dict(human_mask=np.zeros((96,224,224),bool),object_mask=np.zeros((96,224,224),bool),K_full=np.tile(np.array([[1920,0,768],[0,1920,576],[0,0,1]],np.float32),(96,1,1)),postopt_K_rois=np.tile(np.eye(3,dtype=np.float32),(96,1,1)),postopt_crop_contract="cari4d.smplh_postopt_full_resolution_crop.v1",postopt_human_mask=np.zeros((96,256,256),np.float32),postopt_object_mask=np.zeros((96,256,256),np.float32)),raw=dict(rot=np.zeros((96,6),np.float32),trans=np.zeros((96,3),np.float32),delta_mhr_shape=np.full((96,45),.5,np.float32),delta_mhr_hand=np.zeros((96,108),np.float32)))
    hook={k:1 for k in gate.constrained.COUNTERS}|dict(raw_prediction_bytes_preserved=True,initializer_bytes_preserved=True,
        native_compose_global_unchanged=True,native_function_code_unchanged=True)
    return bundle,hook


def test_producer_shaped_complete13_readonly_inventory(gate,tmp_path,monkeypatch):
    directory,pins,report,init,objects=fixture_prepare(gate,tmp_path,monkeypatch)
    result=gate.prepared_inputs(tmp_path,pins)
    assert len(result[-1])==13 and result[0]==directory and result[2]["mhr_trans"].tobytes()==init["mhr_trans"].tobytes()
    assert result[1]["native_replay_calls"]==6 and report["reference_per_frame_mean_mm"]==[.001]*96


@pytest.mark.parametrize("fault",["missing","extra","boolbytes","hash","revision","reportlink","schema"])
def test_exact_pins_no_absent_or_ambiguous_acceptance(gate,tmp_path,monkeypatch,fault):
    _,pins,*_=fixture_prepare(gate,tmp_path,monkeypatch)
    if fault=="missing":pins["prepare_files"].pop("target.npy")
    elif fault=="extra":pins["prepare_files"]["private/GT.npy"]={"sha256":"b"*64,"bytes":1}
    elif fault=="boolbytes":pins["prepare"]["bytes"]=True
    elif fault=="hash":pins["prepare_files"]["target.npy"]["sha256"]="no"
    elif fault=="revision":pins["prepare"]["producer_revision"]="b"*39
    elif fault=="reportlink":pins["prepare"]["sha256"]="b"*64
    else:pins["schema"]="unknown"
    with pytest.raises(ValueError):gate.validate_prepare_pins(pins)


@pytest.mark.parametrize("fault",["failed","running","nocount","boolcount","oracle","replay","geometry","fidelity","snapshot","snapshotfiles","source"])
def test_completed_actual_prepare_contract_before_deserialization(gate,tmp_path,monkeypatch,fault):
    directory,pins,report,*_=fixture_prepare(gate,tmp_path,monkeypatch)
    if fault=="snapshotfiles":
        path=directory/"inputs/manifest.json";data=json.loads(path.read_text());data["output_files"].pop("aligned_depth.h5")
    elif fault=="snapshot":
        path=directory/"inputs/manifest.json";data=json.loads(path.read_text());data["original_frame_indices"][-1]=94
    else:
        path=directory/"report.json";data=report
        if fault=="failed":data["status"]="fail"
        elif fault=="running":data["phase"]="running"
        elif fault=="nocount":data.pop("native_geometry_calls")
        elif fault=="boolcount":data["native_geometry_calls"]=True
        elif fault=="oracle":data["ground_truth_used"]=True
        elif fault=="replay":data["stored_initializer_native_replay_verified"]=False
        elif fault=="geometry":data["shared_identity_verified"]=False
        elif fault=="fidelity":data["reference_per_frame_mean_mm"][-1]=2.0001
        else:data["script_sha256"]="e"*64
    path.chmod(0o644);path.write_text(json.dumps(data));path.chmod(0o444)
    relative=str(path.relative_to(directory));pins["prepare_files"][relative]=gate.identity(path)
    if relative=="report.json":pins["prepare"].update(gate.identity(path))
    else:
        report["snapshot_manifest_sha256"]=gate.identity(path)["sha256"]
        report["output_files"]["inputs/manifest.json"]=gate.identity(path)
        rp=directory/"report.json";rp.chmod(0o644);rp.write_text(json.dumps(report));rp.chmod(0o444)
        pins["prepare_files"]["report.json"]=gate.identity(rp);pins["prepare"].update(gate.identity(rp))
    monkeypatch.setitem(sys.modules,"joblib",SimpleNamespace(load=lambda *_:pytest.fail("No deserialize before public gates")))
    with pytest.raises(ValueError):gate.prepared_inputs(tmp_path,pins)


@pytest.mark.parametrize("fault",["extra","symlink","writable","tamper"])
def test_all_source_files_before_deserialize(gate,tmp_path,monkeypatch,fault):
    directory,pins,*_=fixture_prepare(gate,tmp_path,monkeypatch)
    path=directory/"target.npy"
    if fault=="extra":(directory/"untracked.npy").write_bytes(b"x")
    elif fault=="symlink":
        path.unlink();path.symlink_to(directory/"direct_parameters.npz")
    elif fault=="writable":path.chmod(0o644)
    else:path.chmod(0o644);path.write_bytes(b"different");path.chmod(0o444)
    monkeypatch.setitem(sys.modules,"joblib",SimpleNamespace(load=lambda *_:pytest.fail("No deserialize before SHA gates")))
    with pytest.raises(ValueError):gate.prepared_inputs(tmp_path,pins)


def test_native_composition_changes_identity_only_rawshape_remains_original(gate,monkeypatch):
    init=initializer(gate);objects=dict(obj_pose_world=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy())
    b,h=fixture_bundle(gate,init,objects,monkeypatch);before={k:v.tobytes()for k,v in b["raw"].items()}
    summary=gate.validate_bundle(b,init,objects,h)
    assert b["pr"]["mhr_shape"].tobytes()==init["mhr_shape"].tobytes() and b["raw"]["delta_mhr_shape"].max()==.5
    assert before=={k:v.tobytes()for k,v in b["raw"].items()} and summary["delta_mhr_shape"]["shape"]==[96,45]
    assert b["pr"]["mhr_body_pose_cont"][0,0]==np.float32(.02)


def test_frozen_zero_expression_requires_exact_native_addition_not_signed_zero_copy(gate,monkeypatch):
    init=initializer(gate); init["mhr_face"][:]=-0.; init["mhr_hand"][:,0]=-0.
    objects=dict(obj_pose_world=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy())
    b,h=fixture_bundle(gate,init,objects,monkeypatch)
    with pytest.raises(ValueError,match="exact FP32 init\\+zero"):
        gate.validate_bundle(b,init,objects,h)
    for key in ("mhr_hand","mhr_face"):
        b["pr"][key]=init[key]+np.zeros_like(init[key]); b["pr_initial"][key]=b["pr"][key].copy()
    gate.validate_bundle(b,init,objects,h)
    assert not np.any(b["pr"]["mhr_face"]) and np.signbit(init["mhr_face"]).all()
    assert not np.signbit(b["pr"]["mhr_face"]).any()


@pytest.mark.parametrize("fault",["missing","nonzero","negativezero","shape","face_delta"])
def test_frozen_raw_delta_cannot_hide_a_real_change_or_invented_supervision(gate,monkeypatch,fault):
    init=initializer(gate);objects=dict(obj_pose_world=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy())
    b,h=fixture_bundle(gate,init,objects,monkeypatch)
    if fault=="missing":b["raw"].pop("delta_mhr_hand")
    elif fault=="nonzero":b["raw"]["delta_mhr_hand"][0,0]=np.float32(1e-10)
    elif fault=="negativezero":b["raw"]["delta_mhr_hand"][:]=-0.
    elif fault=="shape":b["raw"]["delta_mhr_hand"]=np.zeros((96,107),np.float32)
    else:b["raw"]["delta_mhr_face"]=np.full((96,72),1e-10,np.float32)
    with pytest.raises(ValueError):gate.validate_bundle(b,init,objects,h)


def test_unused_infinite_training_config_is_unchanged_but_never_logged_as_nonfinite_json(gate):
    cfg=dict(clip_len=96,enable_amp=True,pred_mhr_shape=True,pred_mhr_scale=False,body_model="mhr",
             mhr_joint_supervision_mode="body12_freeze_hand_face_all_losses",clip_grad_norm=float("inf"))
    before=dict(cfg); active,digest=gate.config_receipt(cfg)
    json.dumps(active,allow_nan=False)
    assert cfg==before and "clip_grad_norm" not in active and len(digest)==64


@pytest.mark.parametrize("fault",["GT","timeline","shape","scale","hand","face","dtype","reflection","raw","rawscale","rawnan","count","preserved","config","checkpoint","input","posecopy","root","neutral","K","bbox","contact","masks","fullK"])
def test_full_bundle_contract_no_repair(gate,monkeypatch,fault):
    init=initializer(gate);objects=dict(obj_pose_world=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy())
    b,h=fixture_bundle(gate,init,objects,monkeypatch)
    if fault=="GT":b["gt"]={"labels":1}
    elif fault=="timeline":b["frames"]=b["frames"][:-1]
    elif fault in("shape","scale","hand","face"):
        key="mhr_"+fault;b["pr"][key][0,0]=.1;b["pr_initial"][key][0,0]=.1
    elif fault=="dtype":b["pr"]["mhr_trans"]=b["pr"]["mhr_trans"].astype(np.float64)
    elif fault=="reflection":b["pr"]["pose_abs"][0,0,0]=-1
    elif fault=="raw":b["raw"].pop("rot")
    elif fault=="rawscale":b["raw"]["delta_mhr_scale"]=np.zeros((96,28),np.float32)
    elif fault=="rawnan":b["raw"]["delta_mhr_shape"][0,0]=np.nan
    elif fault=="count":h["composition_delegate_returns"]=0
    elif fault=="preserved":h["raw_prediction_bytes_preserved"]=False
    elif fault=="config":b["config"]["enable_amp"]=False
    elif fault=="checkpoint":b["checkpoint"]["step"]=199999
    elif fault=="input":b["in"]["mhr_trans"][0,0]=.1
    elif fault=="posecopy":b["pr"]["pose_abs_1st_delta"][0,0,3]=.1
    elif fault=="root":b["pr"]["mhr_global_rot6d"][0]=0
    elif fault=="neutral":b["mhr_neutral_height_init"][0]=0
    elif fault=="K":b["K_rois"][0,0,0]=0
    elif fault=="bbox":b["bboxes"][0,2]=0
    elif fault=="contact":b["pr"]["contact_logits"][0,0]=np.nan
    elif fault=="masks":b["observations"]["human_mask"]=b["observations"]["human_mask"].astype(np.uint8)
    else:b["observations"]["K_full"][0,0,0]=1280
    with pytest.raises(ValueError):gate.validate_bundle(b,init,objects,h)


def test_hub_guard_same_args_source_only_and_exception_restore(gate):
    called=[]
    def delegate(*args,**kwargs):called.append((args,kwargs));return "actual loader result"
    t=SimpleNamespace(hub=SimpleNamespace(load=delegate));r=dict(hub_attempts=0,hub_returns=0)
    original=gate.local_dino_hub(t,Path("/tmp/pinned-dino"),r)
    try:
        assert t.hub.load("facebookresearch/dinov2","dinov2_vitb14",pretrained=True)=="actual loader result"
        assert called==[(("/tmp/pinned-dino","dinov2_vitb14"),dict(pretrained=True,source="local"))]
        with pytest.raises(ValueError):t.hub.load("other/repo","dinov2_vitb14")
        with pytest.raises(ValueError):t.hub.load("facebookresearch/dinov2","vitg14")
    finally:t.hub.load=original
    assert t.hub.load is delegate and r==dict(hub_attempts=1,hub_returns=1)


def test_missing_config_fail_before_torch_or_output_native(gate,tmp_path):
    code=tmp_path/"code";code.mkdir();out=tmp_path/"out";out.mkdir()
    with pytest.raises(FileNotFoundError):gate.run(tmp_path,out,code,{},lambda:None)
    assert not list(out.iterdir())


def test_wrapper_fresh_narrow_no_history_or_scope_patch(gate):
    p=ROOT/"infra/run_cari96_forward.sh";text=p.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(p)],check=True)
    assert subprocess.run(["rtk","proxy","bash",str(p),"--resume"],capture_output=True).returncode==2
    assert "--network none --memory 32g --cpus 4"in text and "363s"in text and text.count("docker run")==1
    for forbidden in ("outputs/episode_","cari_refined","eval_private","native_mhr_execution"):
        assert forbidden not in text
    assert "configs/cari96_prepare_pins.json"in text and "src=$path,dst=$path,readonly"in text and "src=$OUT,dst=$OUT"in text
    tree=ast.parse(Path(gate.__file__).read_text())
    assert not any(isinstance(n,ast.Import)and any(a.name=="torch"for a in n.names)for n in tree.body)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="run");source=ast.unparse(run)
    assert "clone_native_forward"in source and "use_input_cache=False"in source and "crop_workers=8"in source
    assert "use_deterministic_algorithms(False, warn_only=False)"in source and "strict_native_head"not in source


def test_capture_uses_only_clone_lookup_and_matches_stored_native_raw(gate,monkeypatch):
    init=initializer(gate);objects=dict(obj_pose_world=np.broadcast_to(np.eye(4,dtype=np.float32),(96,4,4)).copy())
    bundle,hook=fixture_bundle(gate,init,objects,monkeypatch)
    prediction={k:v.reshape(1,96,-1).copy()for k,v in bundle["raw"].items()};before={k:v.tobytes()for k,v in prediction.items()}
    delegated=[]
    def original(pred,batch):delegated.append((pred,batch));return {"unaltered":True}
    from types import FunctionType
    def shell():pass
    globals_dict={"compose_mhr_output":original,"unrelated":object()};clone=FunctionType(shell.__code__,globals_dict)
    fake=SimpleNamespace(is_tensor=lambda _:False)
    gate.capture_native_raw(fake,clone,hook)
    marker={"batch":True}
    assert clone.__globals__["compose_mhr_output"](prediction,marker)=={"unaltered":True}
    assert delegated==[(prediction,marker)] and {k:v.tobytes()for k,v in prediction.items()}==before
    assert hook["original_native_raw"]==gate.validate_bundle(bundle,init,objects,hook)
    assert globals_dict["unrelated"]is clone.__globals__["unrelated"]
    assert original is not clone.__globals__["compose_mhr_output"]
    bundle["raw"]["delta_mhr_shape"][0,0]=np.float32(.51)
    assert hook["original_native_raw"]!=gate.validate_bundle(bundle,init,objects,hook)


def test_actual_prepare_config_and_source_helpers_without_data(gate):
    pins=json.loads((ROOT/"configs/cari96_prepare_pins.json").read_text());gate.validate_prepare_pins(pins)
    assert len(pins["prepare_files"])==13
    assert gate.inputs.sha256(ROOT/"infra/cari96_prepare.py")==pins["prepare"]["script_sha256"]
    # Primary acquisition producer fields, not simplified invented runtime keys.
    aux=ast.parse((ROOT/"infra/acquire_auxiliary.py").read_text());weights=ast.parse((ROOT/"infra/acquire_weights.py").read_text())
    assert any(isinstance(n,ast.Constant)and n.value=="source_revisions"for n in ast.walk(aux))
    assert any(isinstance(n,ast.Constant)and n.value=="cari4d/cari4d"for n in ast.walk(weights))
    assert any(isinstance(n,ast.Constant)and n.value=="path"for n in ast.walk(weights))


def test_forward_complete_report_strict_counters_and_no_quality_claim(gate):
    import inspect
    fn=next(n for n in ast.parse(inspect.getsource(gate)).body if isinstance(n,ast.FunctionDef)and n.name=="validate_forward_report")
    assignment=fn.body[0]
    # The expected public-only contract is a literal dict call with module STAGE.
    expected=eval(compile(ast.Expression(assignment.value),"<own literal contract>","eval"),{"STAGE":gate.STAGE,"list":list,"range":range})
    expected.update({k:1 for k in gate.constrained.COUNTERS});gate.validate_forward_report(expected)
    for key in("forward_attempts","forward_returns","forward_validated","stored_bundle_reread_verified","MHR_strict_scope_performed","quality_verified"):
        bad=copy.deepcopy(expected);bad[key]=False if type(bad[key])is int else not bad[key]
        with pytest.raises(ValueError):gate.validate_forward_report(bad)

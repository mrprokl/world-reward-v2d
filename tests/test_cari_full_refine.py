"""Tiny pure full-N contract fixtures; no Torch, model, video or GPU execution."""
import ast
import copy
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"src"));monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("cari_full_refine_fixture",ROOT/"infra/cari_full_refine.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@dataclass
class Config:
    num_steps: int=300
    batch_size: int=0
    frame_start: int=0
    frame_limit: int=0
    freeze_object_rotation: bool=True
    freeze_body_internal_translations: bool=True
    report_every: int=100


def bundle(gate,count=3,mesh=Path("/own/aligned.glb")):
    params={key:np.zeros((count,dim),np.float32) for key,dim in gate.inputs.public.PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:]=[1,0,0,1,0,0];params["mhr_trans"][:,2]=2
    params["pose_abs"]=np.broadcast_to(np.eye(4,dtype=np.float32),(count,4,4)).copy()
    params["contact_logits"]=np.ones((count,2),np.float32)
    metadata=dict(ground_truth_used=False,object_mesh=str(mesh),materialized_input_cache=None,
        object_pose_frame="centered_axis_aligned",object_pose_storage_frame="output_aligned_mesh_frame",
        object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
        object_pose_storage_to_training_transform=np.eye(4).tolist(),object_mesh_to_training_transform=np.eye(4).tolist())
    obs={key:np.ones((count,2,2),bool) for key in ("human_mask","object_mask")}
    obs.update({key:np.ones((count,256,256),np.float32) for key in ("postopt_human_mask","postopt_object_mask")})
    return dict(schema="cari4d.mhr_wild_inference.v1",frames=[f"{i:06d}" for i in range(count)],gt={},
        metadata=metadata,pr=params,pr_initial=copy.deepcopy(params),**{"in":copy.deepcopy(params)},
        observations=obs,raw={"rot":np.zeros((count,6),np.float32),"trans":np.zeros((count,3),np.float32)},
        faces=np.array([[0,1,2]],np.int32),K_rois=np.broadcast_to(np.eye(3),(count,3,3)).copy(),
        bboxes=np.zeros((count,4),np.float32),frame_meta=[dict(frame=f"{i:06d}",src_frame=i,kid=0) for i in range(count)])


def result(gate,source,cfg=None):
    count=len(source["frames"]);out=copy.deepcopy(source)
    out["pr"]["mhr_body_pose_cont"][:,0]+=.02;out["pr"]["pose_abs"][:,0,3]+=.003
    out["pr"]["pose_abs_postopt"]=out["pr"]["pose_abs"].copy()
    out["postopt"]=dict(mode="smplh_parity",config=asdict(cfg or Config()),
        frame_indices=list(range(count)),resolved_batch_size=count,batch_sampling="full_clip_v1",
        optimized_parameters=gate.contract.OPTIMIZED_PARAMETERS,fixed_parameters=gate.contract.FIXED_PARAMETERS,
        history=[dict(iter=float(i),batch_start=0.,batch_size=float(count),total_loss=.1) for i in (0,100,200,300)],
        final_diagnostics=dict(contact=.01))
    return out


@pytest.mark.parametrize("episode",[0,15,29])
def test_explicit_episode_and_no_inferred_default(gate,episode):
    parsed=gate.parser().parse_args(["--episode",str(episode)])
    assert parsed.episode==episode
    assert gate.output_relative(episode)==f"outputs/episode_{episode:06d}/cari_shared_refined_v1"


@pytest.mark.parametrize("args",[[],["--ep","15"],["--episode","-1"],["--episode","30"],
    ["--episode","15","--episode","15"],["--episode","15","--budget","5"]])
def test_no_default_repeated_episode_abbreviation_or_fit_override(gate,args):
    with pytest.raises(SystemExit):gate.parser().parse_args(args)


@pytest.mark.parametrize("episode",[False,True,-1,30,15.,"15",None])
def test_output_requires_actual_int_episode(gate,episode):
    with pytest.raises(ValueError):gate.output_relative(episode)


def test_count_is_explicit_no_framespace_monkeypatch(gate):
    source=bundle(gate,3);out=result(gate,source)
    assert gate.validate_result(source,out,3)["effective_optimizer_updates"]==301
    with pytest.raises(ValueError):gate.validate_result(source,out,2)
    assert not hasattr(gate,"FRAMES")


def test_occlusion_retains_full_trajectory_and_original_contact(gate):
    source=bundle(gate)
    for value in source["observations"].values():value[1]=0
    params,pose=gate.validate_source_bundle(source,Path("/own/aligned.glb"),3)
    assert len(params)==7 and pose.shape==(3,4,4)
    out=result(gate,source);gate.validate_result(source,out,3)
    assert out["pr"]["contact_logits"].tobytes()==source["pr"]["contact_logits"].tobytes()


@pytest.mark.parametrize("fault",["GT","timeline","dtype","shape","scale","face","mesh","cache","oldrefine",
    "maskmissing","maskdtype","masknan","postgrid","contact","contactmask"])
def test_native_source_has_no_repair_or_partial_input(gate,fault):
    source=bundle(gate)
    if fault=="GT":source["gt"]={"testlabels":1}
    elif fault=="timeline":source["frames"][-1]="000000"
    elif fault=="dtype":source["pr"]["mhr_trans"]=source["pr"]["mhr_trans"].astype(np.float64)
    elif fault in ("shape","scale","face"):source["pr"]["mhr_"+fault][1,0]=.01
    elif fault=="mesh":source["metadata"]["object_mesh"]="another"
    elif fault=="cache":source["metadata"]["materialized_input_cache"]="cache"
    elif fault=="oldrefine":source["postopt"]={}
    elif fault=="maskmissing":source["observations"].pop("human_mask")
    elif fault=="maskdtype":source["observations"]["object_mask"]=source["observations"]["object_mask"].astype(np.float32)
    elif fault=="masknan":source["observations"]["postopt_object_mask"][0,0,0]=np.nan
    elif fault=="postgrid":source["observations"]["postopt_human_mask"]=np.ones((3,4,4),np.float32)
    elif fault=="contact":source["pr"]["contact_logits"]=np.ones((3,1),np.float32)
    else:source["pr"]["contact_logits"]=np.ma.array(source["pr"]["contact_logits"],mask=False)
    with pytest.raises(ValueError):gate.validate_source_bundle(source,Path("/own/aligned.glb"),3)


@pytest.mark.parametrize("fault",["root","trans","hand","shape","scale","face","internal","objectR","mesh",
    "raw","K","mask","faces","contact","pr_initial","in","postoptpose","extra_pr","extra_top",
    "missing_pr","partial","updates","history_middle","history_duplicate","history_bool","history_batch","history_start","diagnostics"])
def test_native_result_moves_only_original_two_groups(gate,fault):
    source=bundle(gate);out=result(gate,source)
    if fault in ("root","trans","hand","shape","scale","face"):
        key="mhr_global_rot6d" if fault=="root" else "mhr_"+fault;out["pr"][key][0,0]+=.01
    elif fault=="internal":out["pr"]["mhr_body_pose_cont"][0,254]=.01
    elif fault=="objectR":out["pr"]["pose_abs"][0,:3,:3]=np.diag([-1,-1,1])
    elif fault=="mesh":out["metadata"]["object_mesh"]="another"
    elif fault=="raw":out["raw"]["rot"][0,0]=.01
    elif fault=="K":out["K_rois"][0,0,0]=2
    elif fault=="mask":out["observations"]["human_mask"][0,0,0]=False
    elif fault=="faces":out["faces"][0,0]=1
    elif fault=="contact":out["pr"]["contact_logits"][0,0]+=.1
    elif fault in ("pr_initial","in"):out[fault]["mhr_shape"][0,0]+=.1
    elif fault=="postoptpose":out["pr"]["pose_abs_postopt"][0,0,3]+=.1
    elif fault=="extra_pr":out["pr"]["unexpected"]=np.zeros(3)
    elif fault=="extra_top":out["unexpected"]=np.zeros(3)
    elif fault=="missing_pr":out["pr"].pop("contact_logits")
    elif fault=="partial":out["postopt"]["frame_indices"]=[0,1]
    elif fault=="updates":out["postopt"]["config"]["num_steps"]=299
    elif fault=="history_middle":out["postopt"]["history"][1]["iter"]=99
    elif fault=="history_duplicate":out["postopt"]["history"].insert(1,dict(iter=0.,total_loss=.1))
    elif fault=="history_bool":out["postopt"]["history"][0]["iter"]=False
    elif fault=="history_batch":out["postopt"]["history"][0]["batch_size"]=2.
    elif fault=="history_start":out["postopt"]["history"][0]["batch_start"]=1.
    else:out["postopt"]["final_diagnostics"]["contact"]=np.nan
    with pytest.raises(ValueError):gate.validate_result(source,out,3)


@pytest.mark.parametrize("outcome",["success","exception","mutated","config"])
def test_actual_native_accounting(gate,outcome):
    source=bundle(gate);report=dict(optimizer_attempts=0,optimizer_returns=0,optimizer_validated=0);seen=[]
    def callback(s,v,f,cfg,*,mhr_layer):
        assert mhr_layer=="layer" and report["optimizer_attempts"]==1 and report["optimizer_returns"]==0
        if outcome=="exception":raise RuntimeError("native failure")
        out=result(gate,s,cfg)
        if outcome=="mutated":s["raw"]["rot"][0,0]=.1
        if outcome=="config":out["postopt"]["config"]["report_every"]=50
        return out
    call=lambda:gate.invoke_optimizer(source,np.zeros((3,3)),np.array([[0,1,2]]),Config(),"layer",callback,3,report,
        lambda:seen.append(report.copy()))
    if outcome=="success":call();assert report["optimizer_validated"]==1
    else:
        with pytest.raises(RuntimeError if outcome=="exception" else ValueError):call()
        assert report["optimizer_validated"]==0
    assert report["optimizer_attempts"]==1 and report["optimizer_returns"]==int(outcome!="exception")
    assert seen[0]["optimizer_attempts"]==1 and seen[0]["optimizer_returns"]==0


def completed_report(gate,spec):
    report=dict(stage=gate.STAGE,status="pass",phase="complete",image_id=gate.IMAGE,episode_index=spec.episode_index,
        clip_spec=asdict(spec),frames=spec.total_frames,source_frames=spec.total_frames,
        original_frame_indices=list(range(spec.total_frames)),input_track="track_1",ground_truth_used=False,
        ground_truth_read=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",
        requested_steps=300,effective_optimizer_updates=301,optimizer_attempts=1,optimizer_returns=1,optimizer_validated=1,
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,saved_bundle_reloaded_verified=True,
        frozen_raw_inputs_byte_preserved=True,object_mesh_unchanged=True,native_refinement_verified=True,
        learned_inference_calls=0,quality_verified=False,adoption_authorized=False,submission_produced=False,
        numerical_bit_determinism_claimed=False,optimizer_sha256=gate.contract.OPTIMIZER_SHA256,
        refinement_assets=gate.contract.REFINEMENT_ASSETS,checkpoint_sha256=gate.contract.CHECKPOINT_SHA256,
        budget_seconds=7200,history_steps=[0,100,200,300],bundle_bytes=32,
        metadata=dict(native_refinement_verified=True,full_original_frame_coverage_verified=True,
            frozen_parameters_bit_identical=True,requested_steps=300,effective_optimizer_updates=301,batch_size=0,
            optimized_parameters=gate.contract.OPTIMIZED_PARAMETERS,fixed_parameters=gate.contract.FIXED_PARAMETERS))
    report.update({key:"c"*64 for key in ("forward_report_sha256","prepare_report_sha256","input_report_sha256",
        "source_bundle_sha256","bundle_sha256","object_mesh_sha256","input_sha256")})
    report["output_files"]={"refined.pth":dict(sha256=report["bundle_sha256"],bytes=32)}
    return report


@pytest.mark.parametrize("count",[96,97,501,673])
def test_report_full_timeline_not_fixed_96(gate,count):
    spec=gate.inputs.PublicClipSpec(15,count,"front_stereo_camera_left",1152,1536)
    report=completed_report(gate,spec);gate.validate_refinement_report(report,spec)
    assert report["quality_verified"] is False and report["adoption_authorized"] is False


@pytest.mark.parametrize("fault",["count","coverage","episode","spec","GT","quality","updates","attempts","bool",
    "metadata","sha","checkpoint","budget","history","output"])
def test_report_incomplete_or_altered_protocol_rejected(gate,fault):
    spec=gate.inputs.PublicClipSpec(15,501,"front_stereo_camera_left",1152,1536);report=completed_report(gate,spec)
    if fault=="count":report["frames"]=96
    elif fault=="coverage":report["original_frame_indices"][-1]=0
    elif fault=="episode":report["episode_index"]=0
    elif fault=="spec":report["clip_spec"]["total_frames"]=96
    elif fault=="GT":report["ground_truth_read"]=True
    elif fault=="quality":report["quality_verified"]=True
    elif fault=="updates":report["effective_optimizer_updates"]=300
    elif fault=="attempts":report["optimizer_attempts"]=2
    elif fault=="bool":report["optimizer_returns"]=True
    elif fault=="metadata":report["metadata"]["frozen_parameters_bit_identical"]=False
    elif fault=="sha":report["bundle_sha256"]="unknown"
    elif fault=="checkpoint":report["checkpoint_sha256"]="c"*64
    elif fault=="budget":report["budget_seconds"]=100
    elif fault=="history":report["history_steps"]=[0,300]
    else:report["output_files"]["old_prediction.pth"]=report["output_files"]["refined.pth"]
    with pytest.raises(ValueError):gate.validate_refinement_report(report,spec)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_text(json.dumps(value)) if isinstance(value,dict) else path.write_bytes(value)
    path.chmod(0o444)


def producer(gate,root,spec):
    directory=root/gate.output_relative(spec.episode_index);write(directory/"refined.pth",b"own automatic native refined bundle")
    report=completed_report(gate,spec)|dict(producer_revision="a"*40,script_sha256="b"*64)
    row=gate.identity(directory/"refined.pth");report.update(bundle_sha256=row["sha256"],bundle_bytes=row["bytes"],output_files={"refined.pth":row})
    write(directory/"report.json",report)
    files={name:gate.identity(directory/name) for name in ("report.json","refined.pth")}
    pins=dict(schema="world-reward-cari-shared-refined-pins-v1",clip_spec=asdict(spec),
        refined=files["report.json"]|{key:report[key] for key in ("producer_revision","script_sha256")},refined_files=files)
    return directory,pins,report


def test_frozen_producer_exact_inventory_before_torch(gate,tmp_path):
    spec=gate.inputs.PublicClipSpec(15,501,"front_stereo_camera_left",1152,1536);directory,pins,report=producer(gate,tmp_path,spec)
    selected=gate.pinned_refined_report(tmp_path,spec,pins)
    assert selected["files"]==pins["refined_files"] and len(selected["bindings"])==2
    assert selected["report"]==report and selected["directory"]==directory


@pytest.mark.parametrize("fault",["schema","spec","missing","extra","hash","boolbytes","producer","source",
    "writable","symlink","tamper","traversal","report_outputs"])
def test_producer_pin_and_inventory_fail_closed(gate,tmp_path,fault):
    spec=gate.inputs.PublicClipSpec(15,501,"front_stereo_camera_left",1152,1536);directory,pins,report=producer(gate,tmp_path,spec)
    path=directory/"refined.pth"
    if fault=="schema":pins["schema"]="old-refinement"
    elif fault=="spec":pins["clip_spec"]["episode_index"]=0
    elif fault=="missing":pins["refined_files"].pop("refined.pth")
    elif fault=="extra":write(directory/"cache.pth",b"old prediction")
    elif fault=="hash":pins["refined_files"]["refined.pth"]["sha256"]="c"*64
    elif fault=="boolbytes":pins["refined_files"]["refined.pth"]["bytes"]=True
    elif fault=="producer":pins["refined"]["producer_revision"]="a"*39
    elif fault=="source":pins["refined"]["script_sha256"]="c"*64
    elif fault=="writable":path.chmod(0o644)
    elif fault=="symlink":path.unlink();path.symlink_to(directory/"report.json")
    elif fault=="tamper":write(path,b"altered")
    elif fault=="traversal":pins["refined_files"]["../private"]=pins["refined_files"]["refined.pth"]
    else:
        report["output_files"]["refined.pth"]["bytes"]+=1;write(directory/"report.json",report)
        pins["refined_files"]["report.json"]=gate.identity(directory/"report.json");pins["refined"].update(pins["refined_files"]["report.json"])
    with pytest.raises(ValueError):gate.pinned_refined_report(tmp_path,spec,pins)


def test_complete_fingerprint_byte_exact_and_nonopaque(gate):
    original=dict(values=np.array([0.],np.float32),nested=[None,"source",1]);copy_=copy.deepcopy(original)
    assert gate.fingerprint(original)==gate.fingerprint(copy_)
    copy_["values"][0]=-0.
    assert gate.fingerprint(original)!=gate.fingerprint(copy_)
    for value in (np.ma.array([1],mask=[False]),np.array([object()],dtype=object),Path("/opaque"),{0:1}):
        with pytest.raises(ValueError):gate.fingerprint(value)


def verifier_fixture(gate,root,code,spec,monkeypatch):
    """Exercise our actual consumer composition; fake only prior stage verifier."""
    directory,pins,report=producer(gate,root,spec)
    forwarddir=root/f"outputs/{spec.sequence}/cari_shared_forward_v1"
    preparedir=root/f"outputs/{spec.sequence}/cari_shared_prepare_v1"
    write(forwarddir/"report.json",b"upstream verified report")
    write(forwarddir/"coconet.pth",b"upstream verified source bundle")
    write(preparedir/"report.json",b"upstream verified shared preparation report")
    mesh=root/gate.inputs.relative_paths(spec)["mesh"];write(mesh,b"unchanged original aligned object triangles")
    model=root/gate.BODY_ASSET_RELATIVE/"model.ckpt";write(model,b"actual pinned model")
    native=root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    consumed={native/gate.contract.OPTIMIZER_RELATIVE_PATH,native/"lib_mhr/mhr_layer.py",
        root/"results/cari-refinement-assets.json",root/"results/weights-acquisition.json",
        *(root/"weights/cari4d/refinement"/name for name in gate.contract.REFINEMENT_ASSETS),model}
    for path in consumed:
        if not path.exists():write(path,b"actual pinned consumed refinement source or asset")
    proxy=root/"weights/cari4d/refinement/mhr_collision_proxy_4000v.npz"
    hand=root/"weights/cari4d/refinement/mhr_hand_surface_spec.npz"
    monkeypatch.setattr(gate.contract,"OPTIMIZER_SHA256",gate.identity(native/gate.contract.OPTIMIZER_RELATIVE_PATH)["sha256"])
    monkeypatch.setattr(gate,"LAYER_SHA",gate.identity(native/"lib_mhr/mhr_layer.py")["sha256"])
    monkeypatch.setattr(gate.contract,"REFINEMENT_ASSETS",{name:gate.identity(root/"weights/cari4d/refinement"/name)|dict(revision=gate.contract.UPSTREAM_REVISION)
        for name in (proxy.name,hand.name)})
    monkeypatch.setattr(gate.contract,"require_asset_receipt",lambda record:None)
    write(root/"results/cari-refinement-assets.json",{"actual":"asset receipt"})
    helper=code/"infra/cari_full_refine.py";write(helper,b"actual committed refinement helper")
    helpers={"infra/cari_full_refine.py":gate.identity(helper)}
    monkeypatch.setattr(gate,"source_helpers",lambda c: helpers)
    fp=code/f"configs/cari_clip_{spec.episode_index:06d}_shared_forward_pins.json";write(fp,{"actual":"forward pins"})
    public={str(mesh.relative_to(root)):gate.identity(mesh)}
    paths={name:str(root/path) for name,path in gate.inputs.relative_paths(spec).items()}
    pr=dict(input_report_sha256="d"*64,input_sha256="e"*64,source_files=public,original_input_paths=paths,
        body_assets={"model.ckpt":gate.identity(model)},inference_source_identity={"native":"actual"},decoder_identity={"model":"actual"})
    fr={key:pr[key] for key in ("body_assets","inference_source_identity")}
    bindings={path:gate.identity(path) for path in (forwarddir/"report.json",forwarddir/"coconet.pth",preparedir/"report.json",mesh)}
    checked=dict(directory=forwarddir,report=fr,files={"report.json":bindings[forwarddir/"report.json"],"coconet.pth":bindings[forwarddir/"coconet.pth"]},
        prepare=dict(directory=preparedir,report=pr,files={"report.json":bindings[preparedir/"report.json"]}),bindings=bindings)
    fake=types.ModuleType("cari_full_forward")
    fake.verify_forward_artifacts=lambda r,c,s,p:checked
    monkeypatch.setitem(sys.modules,"cari_full_forward",fake)
    report.update(forward_report_sha256=gate.identity(forwarddir/"report.json")["sha256"],
        prepare_report_sha256=gate.identity(preparedir/"report.json")["sha256"],
        source_bundle_sha256=gate.identity(forwarddir/"coconet.pth")["sha256"],
        input_report_sha256=pr["input_report_sha256"],input_sha256=pr["input_sha256"],input_dataset_revision=gate.inputs.DATASET_REVISION,
        source_public_files=copy.deepcopy(public),original_input_paths=copy.deepcopy(paths),body_assets=copy.deepcopy(pr["body_assets"]),inference_source_identity=copy.deepcopy(pr["inference_source_identity"]),
        decoder_identity=pr["decoder_identity"],source_helpers=helpers,script_sha256=helpers["infra/cari_full_refine.py"]["sha256"],
        refinement_bindings={str(path):gate.identity(path) for path in consumed},object_mesh_sha256=gate.identity(mesh)["sha256"],
        optimizer_sha256=gate.contract.OPTIMIZER_SHA256,refinement_assets=gate.contract.REFINEMENT_ASSETS)
    write(directory/"report.json",report);pins["refined_files"]["report.json"]=gate.identity(directory/"report.json")
    pins["refined"].update(pins["refined_files"]["report.json"],script_sha256=report["script_sha256"])
    ownpin=code/f"configs/cari_clip_{spec.episode_index:06d}_shared_refined_pins.json";write(ownpin,pins)
    return directory,pins,report,checked,ownpin,consumed


def test_consumer_full_bindings_composition_and_relative_inventory(gate,tmp_path,monkeypatch):
    root=tmp_path/"root";code=tmp_path/"code";spec=gate.inputs.PublicClipSpec(15,501,"front_stereo_camera_left",1152,1536)
    directory,pins,report,checked,ownpin,consumed=verifier_fixture(gate,root,code,spec,monkeypatch)
    selected=gate.verify_refined_artifacts(root,code,spec,pins)
    assert selected["files"]==pins["refined_files"] and set(selected["files"])=={"report.json","refined.pth"}
    assert selected["forward"] is checked and selected["prepare"] is checked["prepare"]
    assert consumed <= set(selected["bindings"]) and ownpin in selected["bindings"]
    assert all(type(path) is Path or isinstance(path,Path) for path in selected["bindings"])
    assert selected["bindings"][directory/"refined.pth"]==report["output_files"]["refined.pth"]


@pytest.mark.parametrize("fault",["forward","prepare","source_bundle","video","public","paths","model","source",
    "decoder","helpers","ownpin","asset","assetextra","assetmissing","mesh"])
def test_consumer_exact_lineage_and_consumed_asset_union(gate,tmp_path,monkeypatch,fault):
    root=tmp_path/"root";code=tmp_path/"code";spec=gate.inputs.PublicClipSpec(15,501,"front_stereo_camera_left",1152,1536)
    directory,pins,report,checked,ownpin,consumed=verifier_fixture(gate,root,code,spec,monkeypatch)
    if fault in ("forward","prepare","source_bundle"):
        report[{"forward":"forward_report_sha256","prepare":"prepare_report_sha256","source_bundle":"source_bundle_sha256"}[fault]]="f"*64
    elif fault=="video":report["input_sha256"]="f"*64
    elif fault=="public":report["source_public_files"]={}
    elif fault=="paths":report["original_input_paths"]["mesh"]="/private"
    elif fault=="model":report["body_assets"]={}
    elif fault=="source":report["inference_source_identity"]={}
    elif fault=="decoder":report["decoder_identity"]={}
    elif fault=="helpers":report["source_helpers"]={}
    elif fault=="ownpin":write(ownpin,{"different":"pins"})
    elif fault=="asset":write(next(iter(consumed)),b"consumed native asset changed")
    elif fault=="assetextra":report["refinement_bindings"][str(root/"private")]=dict(sha256="f"*64,bytes=32)
    elif fault=="assetmissing":report["refinement_bindings"].pop(next(iter(report["refinement_bindings"])))
    else:report["object_mesh_sha256"]="f"*64
    if fault not in ("ownpin","asset"):
        write(directory/"report.json",report);pins["refined_files"]["report.json"]=gate.identity(directory/"report.json")
        pins["refined"].update(pins["refined_files"]["report.json"]);write(ownpin,pins)
    with pytest.raises(ValueError):gate.verify_refined_artifacts(root,code,spec,pins)


@pytest.mark.parametrize('tamper',[None,'historical','writable','alias','currentpin'])
def test_refined_explicit_historical_sources_and_current_pin_ownership(gate,tmp_path,monkeypatch,tamper):
    root=tmp_path/'root';historical=tmp_path/'historical'
    spec=gate.inputs.PublicClipSpec(15,501,'front_stereo_camera_left',1152,1536)
    _,pins,_,checked,_,_=verifier_fixture(gate,root,historical,spec,monkeypatch)
    monkeypatch.setattr(gate,'source_helpers',lambda c:{
        'infra/cari_full_refine.py':gate.identity(c/'infra/cari_full_refine.py')})
    consumer=tmp_path/'current';(consumer/'configs').mkdir(parents=True);(consumer/'infra').mkdir()
    for path in(historical/'configs').iterdir():write(consumer/'configs'/path.name,path.read_bytes())
    write(consumer/'infra/cari_full_refine.py',b'new current consumer code, not original producer')
    calls=[]
    def preceding(r,c,s,p,source_code=None):
        calls.append((c,source_code));return checked
    monkeypatch.setattr(sys.modules['cari_full_forward'],'verify_forward_artifacts',preceding)
    with pytest.raises(ValueError):gate.verify_refined_artifacts(root,consumer,spec,pins)
    path=historical/'infra/cari_full_refine.py'
    if tamper=='historical':write(path,b'changed historical producer')
    elif tamper=='writable':path.chmod(0o644)
    elif tamper=='alias':path.unlink();path.symlink_to(consumer/'infra/cari_full_refine.py')
    elif tamper=='currentpin':write(consumer/'configs/cari_clip_000015_shared_refined_pins.json',{})
    if tamper:
        with pytest.raises(ValueError):gate.verify_refined_artifacts(root,consumer,spec,pins,source_code=historical)
        return
    selected=gate.verify_refined_artifacts(root,consumer,spec,pins,source_code=historical)
    assert calls[-1]==(consumer,historical)
    assert historical/'infra/cari_full_refine.py'in selected['bindings']
    assert consumer/'infra/cari_full_refine.py'not in selected['bindings']
    assert consumer/'configs/cari_clip_000015_shared_refined_pins.json'in selected['bindings']
    assert consumer/'configs/cari_clip_000015_shared_forward_pins.json'in selected['bindings']


def test_wrapper_and_runtime_no_private_inputs_or_hyperparameter_rewrite():
    shell=ROOT/"infra/run_cari_full_refine.sh";source=(ROOT/"infra/cari_full_refine.py").read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    text=shell.read_text()
    assert "--network none" in text and "7203s docker run" in text and "--memory 64g --cpus 4" in text
    assert "cari_shared_prepare_v1" in text and "cari_shared_forward_v1" in text
    assert "cari96_public" not in text and "cari96_forward" not in text and "track_2" not in text and "track_3" not in text
    assert "--mount \"type=bind,src=$OUT,dst=$OUT\"" in text
    assert "MHRParityPostOptConfig(penetration_collision_proxy_path=" in source
    assert "report_every=100)" in source and "frame_limit=" not in source and "cfg.num_steps" not in source
    tree=ast.parse(source)
    assert not any(isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FRAMES" for t in node.targets) for node in ast.walk(tree))
    assert "torch.use_deterministic_algorithms" not in source


def test_full_forward_actual_peer_import_is_model_free(gate,monkeypatch):
    peer=ROOT/"infra/cari_full_forward.py"
    assert peer.is_file(), "Required full-forward producer must be present"
    spec=importlib.util.spec_from_file_location("cari_full_forward_abi",peer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.verify_forward_artifacts) and callable(module.validate_forward_report)
    assert callable(module.verify_prepare_artifacts) and "torch" not in sys.modules

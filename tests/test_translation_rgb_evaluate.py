"""Own CPU fixtures cover paired gates/firewalls, never real model/GT results."""

import copy
import ast

import importlib.util

import json

from pathlib import Path

import subprocess

import sys

from types import SimpleNamespace

import numpy as np

import pytest

INFRA=Path(__file__).resolve().parents[1]/"infra"

@pytest.fixture
def quality(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA))
    spec=importlib.util.spec_from_file_location("own_translation_quality",INFRA/"translation_rgb_evaluate.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_centroid_vector_gate_not_mean_per_vertex_or_joint_error(quality,monkeypatch):
    monkeypatch.setattr(quality,"VERTICES",120)
    truth=np.tile([0.,0.,5.],(120,1));left=np.arange(120)<50;right=(np.arange(120)>=50)&(np.arange(120)<100)
    baseline=truth.copy();baseline[left,0]=np.tile([-.1,.1],25)
    proxy=np.tile([.3,.2,5.],(32,1));humans={"raw":baseline,"baseline":baseline,"fitted":truth.copy()}
    result=quality.frame_metrics(humans,truth,proxy,np.array([.3,.2,5.]),left,right)
    assert result["baseline"]["per_hand_relative_vector_cm"]==pytest.approx([0.,0.],abs=1e-12)
    assert result["baseline"]["per_hand_vertex_pve_cm"]==pytest.approx([10.,0.])
    assert result["baseline"]["human_pve_cm"]==pytest.approx(50/120*10)
    assert result["fitted"]["human_pve_cm"]==0

@pytest.mark.parametrize("fault",["hidden","missinghand","overlap","negativez","nan","geometryshape"])
def test_complete_geometry_and_fixed_common_proxy_guards(quality,monkeypatch,fault):
    monkeypatch.setattr(quality,"VERTICES",120)
    truth=np.tile([0.,0.,5.],(120,1));left=np.arange(120)<50;right=(np.arange(120)>=50)&(np.arange(120)<100)
    proxy=np.tile([.3,.2,5.],(32,1));humans={k:truth.copy() for k in ("raw","baseline","fitted")}
    if fault=="hidden":proxy=np.ma.array(proxy,mask=False)
    elif fault=="missinghand":left[:2]=False
    elif fault=="overlap":right=left.copy()
    elif fault=="negativez":proxy[:,2]=-1
    elif fault=="nan":truth[0,0]=np.nan
    else:humans["fitted"]=truth[:-1]
    with pytest.raises(ValueError):quality.frame_metrics(humans,truth,proxy,np.array([.3,.2,5.]),left,right)

def score(clip,frame,human=(10.,9.),hands=(10.,9.)):
    return dict(clip_index=clip,frame_index=frame,
        raw=dict(human_pve_cm=100.,per_hand_relative_vector_cm=[100.,100.]),
        baseline=dict(human_pve_cm=human[0],per_hand_relative_vector_cm=[hands[0],hands[0]]),
        fitted=dict(human_pve_cm=human[1],per_hand_relative_vector_cm=[hands[1],hands[1]]))

def test_equal_five_frame_means_allclips_bothhands_raw_not_gate(quality):
    scores=[score(c,f) for c in range(3) for f in range(5)]
    clips,decision=quality.aggregate(scores)
    assert len(clips)==3 and decision["synthetic_translation_hypothesis_supported"]
    assert decision["adoption_authorized"] is False and clips[0]["raw"]["human_pve_cm"]==100
    scores[0]["fitted"]["human_pve_cm"]=19.
    clips,decision=quality.aggregate(scores)
    assert clips[0]["fitted"]["human_pve_cm"]==11. and not decision["gates"]["no_clip_camera_pve_regression_over_5pct"]
    scores=[score(c,f) for c in range(3) for f in range(5)];scores[14]["fitted"]["per_hand_relative_vector_cm"][1]=17.
    assert not quality.aggregate(scores)[1]["synthetic_translation_hypothesis_supported"]

@pytest.mark.parametrize("fault",["drop","order","idbool","nan","zero"])
def test_score_completeness_and_zero_baseline_no_undefined_success(quality,fault):
    scores=[score(c,f) for c in range(3) for f in range(5)]
    if fault=="drop":scores.pop()
    elif fault=="order":scores.reverse()
    elif fault=="idbool":scores[0]["clip_index"]=False
    elif fault=="nan":scores[0]["fitted"]["human_pve_cm"]=np.nan
    else:
        scores=[score(c,f,human=(0.,0.),hands=(0.,0.)) for c in range(3) for f in range(5)]
        result=quality.aggregate(scores)[1]
        assert result["zero_human_baseline_relative_gain_undefined"] and not result["synthetic_translation_hypothesis_supported"]
        scores[0]["fitted"]["per_hand_relative_vector_cm"][0]=1e-12
        assert not quality.aggregate(scores)[1]["gates"]["no_per_hand_clip_relative_object_regression_over_5pct"]
        return
    with pytest.raises(ValueError):quality.aggregate(scores)

def own_truth(quality,record):
    human=np.tile([0.,0.,5.],(quality.VERTICES,1)).astype(np.float32)
    faces=np.tile(np.array([[0,1,2]],np.int64),(36874,1));of=np.tile(np.array([[0,1,2]],np.int64),(384,1))
    ids=np.full((quality.HEIGHT,quality.WIDTH),-1,np.int64);ids.flat[:64]=0;ids.flat[64:128]=len(faces)
    depth=np.full(ids.shape,np.nan,np.float32);depth[ids>=0]=5.
    return dict(human_vertices_camera_m=human,human_faces=faces,object_vertices_camera_m=np.tile([.2,.1,5.],(194,1)).astype(np.float64),
        object_faces=of,camera_K=np.array([[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]]),scene_depth_m=depth,
        visible_face_indices=ids,clip_index=np.array(record["clip_index"],np.int64),frame_index=np.array(record["frame_index"],np.int64))

@pytest.mark.parametrize("fault",["valid","key","room","z","humanempty","objectempty","camera","face","objectface","frame","masked"])
def test_exact_private_ninefields_foreground_and_fixedcamera(quality,monkeypatch,fault):
    monkeypatch.setattr(quality,"VERTICES",120);monkeypatch.setattr(quality,"WIDTH",16);monkeypatch.setattr(quality,"HEIGHT",16)
    record=dict(clip_index=0,frame_index=0);truth=own_truth(quality,record);faces=truth["human_faces"].copy()
    if fault=="key":truth["keypoints"]=np.zeros((17,2))
    elif fault=="room":truth["visible_face_indices"].flat[0]=len(faces)+384
    elif fault=="z":truth["scene_depth_m"].flat[0]=0
    elif fault=="humanempty":truth["visible_face_indices"][truth["visible_face_indices"]==0]=len(faces)
    elif fault=="objectempty":truth["visible_face_indices"][truth["visible_face_indices"]>=len(faces)]=0
    elif fault=="camera":truth["camera_K"][0,0]=1279
    elif fault=="face":truth["human_faces"][0,0]=119
    elif fault=="objectface":truth["object_faces"][0,0]=194
    elif fault=="frame":truth["frame_index"]=np.array(1,np.int64)
    elif fault=="masked":truth["human_vertices_camera_m"]=np.ma.array(truth["human_vertices_camera_m"],mask=False)
    if fault=="valid":
        quality.validate_truth(truth,record,faces);median,count=quality.visible_object_median(truth)
        yy,xx=np.nonzero(truth["visible_face_indices"]>=len(faces));z=truth["scene_depth_m"][yy,xx]
        expected=np.median(np.c_[(xx+.5-512)/1280*z,(yy+.5-384)/1280*z,z],axis=0)
        assert count==64 and np.allclose(median,expected,rtol=0,atol=1e-7)
    else:
        with pytest.raises(ValueError):quality.validate_truth(truth,record,faces)

def test_public_failure_never_opens_private_directory(quality,tmp_path,monkeypatch):
    events=[]
    def fail(root):events.append("public");raise ValueError("public rejected")
    monkeypatch.setattr(quality,"public_fit",fail)
    def forbidden(*args,**kwargs):raise AssertionError("private IO attempted")
    monkeypatch.setattr(quality,"regular",forbidden)
    report={}
    with pytest.raises(ValueError,match="public rejected"):quality.run(tmp_path,report)
    assert events==["public"] and report.get("private_truth_used_for_evaluation_only",False) is False

def full_quality_fixture(quality,tmp_path,monkeypatch):
    monkeypatch.setattr(quality,"VERTICES",120);monkeypatch.setattr(quality,"WIDTH",16);monkeypatch.setattr(quality,"HEIGHT",16)
    records=[];pairs=[];candidates=[];proxies=[];cases=[]
    sources=tmp_path/"sources";sources.mkdir()
    for name in ("translation_rgb_fit.py","keypoint_rgb_render.py","identity_rgb_render.py","joint_rgb_render.py"):
        target=sources/name;target.write_bytes((INFRA/name).read_bytes());target.chmod(0o444)
    monkeypatch.setattr(quality,"render_source",lambda root,name:sources/name)
    monkeypatch.setattr(quality,"fit_source",lambda root:sources/"translation_rgb_fit.py")
    private=tmp_path/quality.BASE/"eval_private";private.mkdir(parents=True)
    code=tmp_path/"own_render.py";code.write_text("# own fixture renderer\n");monkeypatch.setattr(quality,"RENDER_SOURCE_SHA",quality.sha256(INFRA/"keypoint_rgb_render.py"))
    sem_path=private/"semantic-report.json";sem_path.write_text(json.dumps(dict(source_image_id=quality.IMAGE)));sem_path.chmod(0o400)
    rig_path=private/"rig.npz";controls=np.zeros((15,204),np.float32);shapes=np.zeros((15,45),np.float32)
    with rig_path.open("wb") as stream:np.savez(stream,controls=controls,shape45=shapes,parameter_limits=np.zeros((249,2)))
    rig_path.chmod(0o400)
    renderer=SimpleNamespace(__file__=str(code),STAGE="own_fresh_keypoint_rgb_render",helper_hashes=lambda:{"own":"a"*64},
        named_controls=lambda names,bounds:(controls.copy(),shapes.copy(),[]),
        render=SimpleNamespace(semantics=SimpleNamespace(MODEL_SHA="c"*64),require_semantic_report=lambda report:None))
    monkeypatch.setitem(sys.modules,"hand_synthetic_render",SimpleNamespace(require_semantic_report=lambda report:None))
    monkeypatch.setitem(sys.modules,"translation_rgb_public",SimpleNamespace(helper_identities=lambda *args:{"source":"a"*64}))
    for c in range(3):
        for f in range(5):
            record=dict(file=f"clip_{c:02d}_frame_{f:03d}.png",clip_index=c,frame_index=f,sha256=f"{c*5+f+1:064x}");records.append(record)
            truth=own_truth(quality,record);path=private/(Path(record["file"]).stem+".npz")
            with path.open("wb") as stream:np.savez_compressed(stream,**truth)
            path.chmod(0o400);cases.append(dict(file=record["file"],rgb_sha256=record["sha256"],clip_index=c,frame_index=f,truth_sha256=quality.sha256(path)))
            pair=dict(human_faces=truth["human_faces"],hand_mask_left=np.arange(120)<50,
                hand_mask_right=(np.arange(120)>=50)&(np.arange(120)<100),raw_vertices_camera_m=truth["human_vertices_camera_m"]+[.3,0,0],
                shared_vertices_camera_m=truth["human_vertices_camera_m"]+[.2,0,0],shared_keypoints_camera_m=np.tile([.2,0.,5.],(308,1)),
                camera_K=truth["camera_K"]);pairs.append(pair)
            candidates.append(dict(vertices_camera_m=truth["human_vertices_camera_m"]+[.1,0,0],keypoints_camera_m=np.tile([.1,0.,5.],(308,1))))
            q,_=quality.visible_object_median(truth);proxies.append(dict(object_points_camera_m=np.tile(q,(32,1))))
    render=dict(stage=renderer.STAGE,status="pass",phase="complete",frames=15,code_revision=quality.RENDER_REVISION,
        script_sha256=quality.RENDER_SOURCE_SHA,helper_source_sha256={k:quality.sha256(INFRA/n) for k,n in {"render":"hand_synthetic_render.py","primitives":"identity_rgb_render.py","joint":"joint_rgb_render.py","camera":"camera_render.py"}.items()},image_id=quality.IMAGE,
        actual_MHR_reference_used=True,actual_reference_forward_calls=2,challenge_inputs_used=False,
        synthetic_truth_used_for_rendering_only=True,inference_performed=False,predictions_performed=False,all_truth_private=True,
        identity_clip_constant=True,true_camera_private=True,scene_depth_scope="human/object foreground only; background IDs=-1/depth=NaN",
        accuracy_verified=False,quality_verified=False,photorealism_verified=False,public_manifest_sha256=quality.MANIFEST_SHA,
        public_manifest_bytes=2199,truth_keys=sorted(quality.TRUTH_KEYS),model_sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",semantic_report_sha256=quality.sha256(sem_path),
        rig_sha256=quality.sha256(rig_path),parameter_names=[f"parameter_{i}" for i in range(249)],cases=cases)
    render_path=private/"render-report.json";render_path.write_text(json.dumps(render));render_path.chmod(0o400)
    monkeypatch.setattr(quality,"RENDER_SHA",quality.sha256(render_path))
    public=tmp_path/"public_seal.json";public.write_text("{}") ;public.chmod(0o444);frozen=[quality.regular(public)]
    producer=dict(lineage=dict(own_fixture=True),source_helpers={"source":"a"*64})
    monkeypatch.setattr(quality,"public_fit",lambda root:(records,pairs,candidates,proxies,[dict(keypoints=np.zeros((1,133,2)),scores=np.ones((1,133),np.float32))]*15,producer,frozen.copy()))
    return private,public,records

def test_all15_end_to_end_private_fixture_pairs_and_rehash_no_alignment(quality,tmp_path,monkeypatch):
    _,_,_=full_quality_fixture(quality,tmp_path,monkeypatch);report={};quality.run(tmp_path,report)
    assert report["status"]=="pass" and report["all_frames_scored"] and len(report["frame_metrics"])==15
    assert report["decision"]["synthetic_translation_hypothesis_supported"] and not report["decision"]["adoption_authorized"]
    assert report["clip_metrics"][0]["baseline"]["human_pve_cm"]==pytest.approx(20.)
    assert report["clip_metrics"][0]["fitted"]["human_pve_cm"]==pytest.approx(10.)
    assert report["no_gt_alignment"] and report["wrists_or_joint_truth_used"] is False

@pytest.mark.parametrize("fault",["truthsha","privatemissing","publicposthash"])
def test_private_source_truth_or_public_mutation_never_pass(quality,tmp_path,monkeypatch,fault):
    private,public,records=full_quality_fixture(quality,tmp_path,monkeypatch)
    if fault=="truthsha":
        path=private/(Path(records[0]["file"]).stem+".npz");path.chmod(0o600);path.write_bytes(b"changed");path.chmod(0o400)
    elif fault=="privatemissing":(private/(Path(records[0]["file"]).stem+".npz")).unlink()
    else:
        original=quality.frame_metrics
        def mutate(*args):
            public.chmod(0o644);public.write_text("changed");public.chmod(0o444);return original(*args)
        monkeypatch.setattr(quality,"frame_metrics",mutate)
    report={}
    with pytest.raises(ValueError):quality.run(tmp_path,report)
    assert report.get("status")!="pass"

def test_wrapper_offline_cpu_only_one_private_stage_and_scoped_output(quality):
    path=INFRA/"run_translation_rgb_evaluate.sh";subprocess.run(["bash","-n",str(path)],check=True);text=path.read_text()
    assert "--gpus" not in text and "--network none --cpus 4 --memory 8g" in text
    assert "CUDA_VISIBLE_DEVICES=" in text and 'OUT="$BASE/translation_quality_v1"' in text
    assert 'chown scenesmith:scenesmith "$OUT"' in text and "chown -R" not in text
    assert '"$BASE/eval_private"' in text and 'type=bind,src=$path,dst=$path,readonly' in text
    assert 'src=$OUT,dst=$OUT' in text and "translation_rgb_evaluate.py" in text

def test_heldout_native_rgb_only_counts_no_sample_drop_or_gt_joint(quality):
    K=np.array([[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]])
    pair=dict(camera_K=K,shared_keypoints_camera_m=np.tile([0.,0.,5.],(308,1)))
    candidate=dict(keypoints_camera_m=pair["shared_keypoints_camera_m"].copy())
    observation=dict(keypoints=np.tile([512.,384.],(1,133,1)),scores=np.zeros((1,133),np.float32))
    output=quality.heldout_diagnostics(pair,candidate,observation)
    assert output["positive_score_count"]==0 and output["baseline_mean_pixel_error"] is None
    assert not output["used_for_fit"] and not output["private_keypoint_truth_used"]
    observation["scores"][0,list(quality.policy.HELDOUT_COCO)]=1
    output=quality.heldout_diagnostics(pair,candidate,observation)
    assert output["positive_score_count"]==7 and output["baseline_mean_pixel_error"]==0

def test_source_only_aliases_bind_immutable_producer_jobs_not_native_imports():
    text=(INFRA/"run_translation_rgb_evaluate.sh").read_text()
    assert 'jobs/$REV/run_translation_rgb_fit/code/infra/$NAME' in text
    assert 'jobs/42f457fc46fbeb814befb48b8104403b06922f63/run_keypoint_rgb_prepare/code/infra/$NAME' in text
    assert 'src=$SOURCE,dst=$SOURCE,readonly' in text
    source=(INFRA/"translation_rgb_evaluate.py").read_text()
    assert 'import keypoint_rgb_render' not in source and 'import translation_rgb_fit' not in source



def load_test_module(name, file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def complete_public_fit_fixture(quality,tmp_path,monkeypatch):
    """Real15-file freeze and60-state NumPy fitting; model/asset ABI mocked only."""
    import translation_rgb_public as public
    import translation_rgb_fit as fitter
    fixture=load_test_module("historical_public_fixture","test_keypoint_rgb_fit.py")
    old_frame=fixture.frame
    def frame(gate, patch, clip=0, index=0):
        raw,pair,record=old_frame(gate,patch,clip,index)
        points=np.c_[np.linspace(-.7,.7,10),np.sin(np.arange(10))*.6,np.linspace(1.5,2.5,10)].astype(np.float32)
        for key in ("raw_keypoints_camera_m","shared_keypoints_camera_m"):
            pair[key][list(quality.policy.TRAIN_MHR)]=points
        raw["raw_keypoints_camera_m"]=pair["raw_keypoints_camera_m"].copy()
        return raw,pair,record
    def observed(gate, record):
        raw,pair,_=frame(gate,monkeypatch,record["clip_index"],record["frame_index"])
        data=fixture_dw(gate,record)
        target,_=quality.policy.project_and_jacobian(pair["shared_keypoints_camera_m"][list(quality.policy.TRAIN_MHR)].astype(np.float64),
            raw["pred_cam_t"],np.array([.1,-.08,.07]),pair["camera_K"])
        data["keypoints"][0,list(quality.policy.TRAIN_COCO)]=target
        return data
    fixture_dw=fixture.dw
    monkeypatch.setattr(fixture,"frame",frame);monkeypatch.setattr(fixture,"dw",observed)
    prior=SimpleNamespace(baseline=public.previous.baseline,native=public.native,public=public.previous,sha256=quality.sha256)
    base,_,_,_=fixture.public_fixture(prior,tmp_path,monkeypatch)
    records,raw,pairs,dw,lineage=public.previous.public_predictions(tmp_path)
    failed=base/"root_fit_v1";failed.mkdir();(failed/"candidates").mkdir();(failed/"proxies").mkdir();rows=[]
    for record,original in zip(records,raw):
        points,pixels=public.previous.object_sample(original,1.)
        data=dict(object_points_camera_m=points,pixel_indices=pixels,shared_depth_scale=np.array(1.),
            clip_index=original["clip_index"],frame_index=original["frame_index"])
        rows.append(public.native.save(failed,"proxies",record,data))
    failed_report=dict(stage=public.previous.STAGE,status="fail",phase="native_root_optimization",producer_revision=public.FAILED_REV,
        script_sha256=public.FAILED_SCRIPT,image_id=quality.IMAGE,network="none",device="cuda",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],object_proxies_frozen_before_fit=True,
        all_candidates_frozen=False,quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        error_type="ValueError",error="Complete249 native bounds/rootEuler mapping violated; no clipping",candidate_outputs=[],proxy_outputs=rows,
        counters=dict(proxy_rasters=15,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0))
    failed_path=failed/"report.json";failed_path.write_text(json.dumps(failed_report));failed_path.chmod(0o444)
    monkeypatch.setattr(public,"FAILED_SHA",quality.sha256(failed_path))
    monkeypatch.setenv("WR_CODE_REVISION","a"*40)
    source=quality.fit_source(tmp_path);source.parent.mkdir(parents=True);source.write_bytes(Path(fitter.__file__).read_bytes());source.chmod(0o444)
    renderer=quality.render_source(tmp_path,"keypoint_rgb_render.py");renderer.parent.mkdir(parents=True)
    renderer.write_bytes((INFRA/"keypoint_rgb_render.py").read_bytes());renderer.chmod(0o444)
    out=tmp_path/public.OUT;out.mkdir();(out/"candidates").mkdir();path=out/"report.json";path.touch()
    report=dict(stage=public.STAGE,status="fail",phase="public_integrity",device="cpu",network="none",producer_revision="a"*40,
        image_id=quality.IMAGE,script_sha256=quality.sha256(source),optimizer=public.OPTIMIZER,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        object_proxies_reused=True,proxies_recomputed=False,all_candidates_frozen=False,inputs_unchanged=False,native_controls_unchanged=True,
        gpu_used=False,Torch_imported=False,native_forward_calls=0,shared_identity_constant=True,body_hands_fixed=True,camera_fixed=True,
        methodological_prior_cohort_reuse=True,initial_jacobian_includes_priors=False,
        training_COCO_indices=list(quality.policy.TRAIN_COCO),training_MHR_indices=list(quality.policy.TRAIN_MHR),
        full_HOI_verified=False,accuracy_verified=False,quality_verified=False,adoption_authorized=False,all_cases_retained=False,
        counters=dict(evaluated_states=0,adam_updates=0,initial_jacobian_rows=0,native_forward_calls=0,proxy_rasters=0),
        candidate_outputs=[],fit_records=[])
    fitter.perform(tmp_path,out,report,lambda:None)
    def seal():
        path.chmod(0o644);path.write_text(json.dumps(report));path.chmod(0o444)
    seal();return public,report,seal,path


def test_all15_public_actual_numpy_schedule_before_first_private_io(quality,tmp_path,monkeypatch):
    _,producer,_,_=complete_public_fit_fixture(quality,tmp_path,monkeypatch)
    records,pairs,candidates,proxies,dw,audited,frozen=quality.public_fit(tmp_path)
    assert all(len(value)==15 for value in (records,pairs,candidates,proxies,dw))
    assert audited["counters"]==dict(evaluated_states=900,adam_updates=885,initial_jacobian_rows=300,native_forward_calls=0,proxy_rasters=0)
    assert len(frozen)>100 and not (tmp_path/quality.BASE/"eval_private").exists()
    events=[]
    def private(root, records, frozen):
        events.append("private_after900_states");raise RuntimeError("truth intentionally absent")
    monkeypatch.setattr(quality,"private_truth",private)
    report={}
    with pytest.raises(RuntimeError,match="truth intentionally absent"):quality.run(tmp_path,report)
    assert events==["private_after900_states"] and report["predictions_frozen_before_private"]


@pytest.mark.parametrize("fault",["GT","phase","counts","rank","state","objective","source","revision","candidate","proxy","extra"])
def test_producer_shaped_public_tamper_abstains_before_private(quality,tmp_path,monkeypatch,fault):
    public,report,seal,path=complete_public_fit_fixture(quality,tmp_path,monkeypatch)
    if fault=="GT":report["ground_truth_used"]=True
    elif fault=="phase":report["phase"]="running"
    elif fault=="counts":report["counters"]["evaluated_states"]=899
    elif fault=="rank":report["fit_records"][0]["initial_observation_jacobian"]["singular_values"][2]=0.
    elif fault=="state":report["fit_records"][4]["evaluated_latents"][30][0]+=.01
    elif fault=="objective":report["fit_records"][3]["evaluated_losses"][4]+=.01
    elif fault=="source":
        source=quality.fit_source(tmp_path);source.chmod(0o644);source.write_bytes(b"changed");source.chmod(0o444)
    elif fault=="revision":report["producer_revision"]="b"*40
    elif fault=="candidate":
        artifact=path.parent/report["candidate_outputs"][0]["artifact"];artifact.chmod(0o644);artifact.write_bytes(b"changed");artifact.chmod(0o444)
    elif fault=="proxy":
        artifact=tmp_path/public.previous.OUT/report["lineage"]["reused_object_proxy_outputs"][0]["artifact"]
        artifact.chmod(0o644);artifact.write_bytes(b"changed");artifact.chmod(0o444)
    else:(path.parent/"extra.npz").write_bytes(b"hidden")
    seal()
    def forbidden(*args,**kwargs):raise AssertionError("private truth read before public audit")
    monkeypatch.setattr(quality,"private_truth",forbidden)
    with pytest.raises(ValueError):quality.run(tmp_path,{})


def test_copied_pure_metrics_ast_equal_frozen_d96_not_new_metric(quality):
    names=("require_fields","regular","floating","validate_truth","visible_object_median","frame_metrics","heldout_diagnostics")
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False)for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    original=functions(INFRA/"keypoint_rgb_evaluate.py");current=functions(Path(quality.__file__))
    assert all(current[name]==original[name] for name in names)


def test_quality_refuses_torch_before_public_or_private(quality,tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    monkeypatch.setattr(quality,"public_fit",lambda *_:pytest.fail("public IO after forbidden Torch import"))
    with pytest.raises(ValueError,match="must not import Torch"):quality.run(tmp_path,{})


def test_quality_source_mutation_during_scoring_fails(quality,tmp_path,monkeypatch):
    full_quality_fixture(quality,tmp_path,monkeypatch)
    actual=quality.sha256;calls=0
    def source_hash(path):
        nonlocal calls
        if Path(path)==Path(quality.__file__):
            calls+=1
            if calls>1:return "e"*64
        return actual(path)
    monkeypatch.setattr(quality,"sha256",source_hash)
    with pytest.raises(ValueError,match="Quality source changed"):quality.run(tmp_path,{})


def test_reserved_quality_output_argv_cpu_and_revision_fail_before_files(quality,tmp_path,monkeypatch):
    monkeypatch.setenv("WR_ROOT",str(tmp_path));monkeypatch.setenv("WR_CODE_REVISION","a"*40)
    monkeypatch.setenv("WR_IMAGE_ID",quality.IMAGE);monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    with pytest.raises(ValueError,match="Fresh canonical"):quality.main([])
    assert not (tmp_path/quality.OUT).exists()
    with pytest.raises(SystemExit):quality.main(["--reuse"])


def test_complete_static_code_closure_within_explicit_control_budget(quality):
    import io
    import tarfile
    import azure_job
    root=INFRA.parent
    files={p.relative_to(root).as_posix():p.read_bytes()for directory in ("infra","src","configs")
        for p in (root/directory).rglob("*")if p.is_file() and (p.suffix in (".py",".sh",".cpp",".json",".yaml",".toml") or p.name.startswith("Dockerfile"))}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_translation_rgb_evaluate.sh")
    assert "infra/translation_rgb_public.py" in paths and "src/world_reward/translation_refit.py" in paths
    assert "infra/translation_rgb_fit.py" not in paths and "infra/keypoint_rgb_render.py" not in paths
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w",format=tarfile.PAX_FORMAT)as archive:
        for path in paths:
            info=tarfile.TarInfo(path);info.size=len(files[path]);archive.addfile(info,io.BytesIO(files[path]))
    encoded,_=azure_job.encoded_runtime_archive(stream.getvalue())
    assert len(encoded)<=128000

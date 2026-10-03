"""Synthetic CPU contracts only, not native inference/GT/quality evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path.cwd()
MODULE = Path(__file__).parents[1]/"infra/root5_rgb_public.py"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"src")); monkeypatch.syspath_prepend(str(REPO/"infra"))
    spec = importlib.util.spec_from_file_location("root5_public_test", MODULE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def pins(gate):
    return dict(schema=gate.PIN_SCHEMA, producer_revision=dict(masks=gate.PRODUCER_REVISION,baseline=gate.NATIVE_PRODUCER_REVISION,dwpose=gate.NATIVE_PRODUCER_REVISION), manifest_sha256=gate.MANIFEST_SHA,
        automatic_masks_sha256="a"*64, baseline_sha256="b"*64, dwpose_sha256="c"*64)


@pytest.mark.parametrize("fault", ["missing","extra","revision","schema","manifest","bool","upper"])
def test_no_dynamic_or_weak_receipt_pins(gate, fault):
    value=pins(gate)
    if fault=="missing":value.pop("dwpose_sha256")
    elif fault=="extra":value["camera_truth"]={}
    elif fault=="revision":value["producer_revision"]["masks"]="a"*40
    elif fault=="schema":value["schema"]="other"
    elif fault=="manifest":value["manifest_sha256"]="d"*64
    elif fault=="bool":value["baseline_sha256"]=True
    else:value["baseline_sha256"]="A"*64
    with pytest.raises(ValueError):gate.validate_pins(value)


def test_pins_load_requires_frozen_regular_bytes(gate,tmp_path):
    path=tmp_path/"pins.json";path.write_text(json.dumps(pins(gate)));path.chmod(0o444)
    assert gate.load_pins(path)==pins(gate)
    path.chmod(0o644)
    with pytest.raises(ValueError):gate.load_pins(path)


def frame(gate,monkeypatch,clip=0,index=0):
    n=gate.native;monkeypatch.setattr(n,"WIDTH",16);monkeypatch.setattr(n,"HEIGHT",8);monkeypatch.setattr(n,"VERTICES",100)
    yy,xx=np.indices((8,16));z=np.full((8,16),2.,np.float32);K=gate.baseline.CAMERA_K.copy()
    points=np.stack(((xx+.5-K[0,2])/K[0,0]*z,(yy+.5-K[1,2])/K[1,1]*z,z),-1).astype(np.float32)
    raw={k:np.zeros(s,np.float32)for k,s in n.BLOCKS.items()};raw["pred_cam_t"][:]=[0,0,2]
    raw.update(raw_depth=z,raw_points=points,validity=np.ones((8,16),bool),rendered_depth=z.copy(),silhouette=np.ones((8,16),bool),
        human_mask=np.ones((8,16),bool),object_mask=np.ones((8,16),bool),raw_vertices_camera_m=np.ones((100,3),np.float32),
        raw_joints_camera_m=np.ones((127,3),np.float32),raw_keypoints_camera_m=np.ones((308,3),np.float32),
        raw_joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)),human_faces=np.zeros((36874,3),np.int64),camera_K=K,
        clip_index=np.array(clip,np.int64),frame_index=np.array(index,np.int64))
    pair={}
    for mode in("raw","shared"):
        for suffix,source in(("vertices_camera_m","raw_vertices_camera_m"),("joints_camera_m","raw_joints_camera_m"),
            ("keypoints_camera_m","raw_keypoints_camera_m"),("joint_global_rotations","raw_joint_global_rotations"),
            ("model_controls","mhr_model_params"),("shape_params","shape_params"),("scale_params","scale_params")):
            pair[mode+"_"+suffix]=raw[source].copy()
    pair.update(human_faces=raw["human_faces"].copy(),camera_K=K,pred_cam_t=raw["pred_cam_t"].copy(),expression=raw["expr_params"].copy(),
        hand_mask_left=np.arange(100)<50,hand_mask_right=np.arange(100)>=50,clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy())
    record=dict(file=f"clip_{clip:02d}_frame_{index:03d}.png",clip_index=clip,frame_index=index,sha256="a"*64)
    return raw,pair,record


def candidate(gate,raw,pair):
    result={k:raw[k].copy()for k in gate.native.BLOCKS}
    for k,source in(("vertices_camera_m","shared_vertices_camera_m"),("keypoints_camera_m","shared_keypoints_camera_m"),
        ("joints_camera_m","shared_joints_camera_m"),("joint_global_rotations","shared_joint_global_rotations"),
        ("human_faces","human_faces"),("camera_K","camera_K"),("hand_mask_left","hand_mask_left"),("hand_mask_right","hand_mask_right")):
        result[k]=pair[source].copy()
    result.update(clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy(),latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32))
    return result


def test_fixed_z_candidate_float32_saturation_and_no_mutation(gate,monkeypatch):
    raw,pair,record=frame(gate,monkeypatch);c=candidate(gate,raw,pair)
    gate.validate_candidate(c,record,raw,pair)
    c["latent"][:]=100;c["physical_delta"][:]=gate.BOUNDS
    c["pred_cam_t"][:2]+=c["physical_delta"][:2];c["global_rot"]+=c["physical_delta"][2:]
    c["mhr_model_params"][3:6]=c["global_rot"]
    gate.validate_candidate(c,record,raw,pair)
    assert c["pred_cam_t"][2:].tobytes()==raw["pred_cam_t"][2:].tobytes()
    assert not raw["global_rot"].any()and not pair["shared_model_controls"].any()


@pytest.mark.parametrize("fault",["Z","scale","shape","finger","root","pose","R","negativeZ","delta","masked","dtype","K","handdtype","extra"])
def test_candidate_does_not_repair_other_native_blocks(gate,monkeypatch,fault):
    raw,pair,record=frame(gate,monkeypatch);c=candidate(gate,raw,pair)
    if fault=="Z":c["pred_cam_t"][2]+=np.float32(1e-5)
    elif fault=="scale":c["scale_params"][0]=.1
    elif fault=="shape":c["shape_params"][0]=.1
    elif fault=="finger":c["hand_pose_params"][0]=.1
    elif fault=="root":c["mhr_model_params"][0]=.1
    elif fault=="pose":c["mhr_model_params"][6]=.1
    elif fault=="R":c["joint_global_rotations"][0,0,0]=-1
    elif fault=="negativeZ":c["vertices_camera_m"][0,2]=0
    elif fault=="delta":c["physical_delta"][0]=np.nextafter(gate.BOUNDS[0],np.float32(np.inf))
    elif fault=="masked":c["latent"]=np.ma.array(c["latent"],mask=False)
    elif fault=="dtype":c["latent"]=c["latent"].astype(np.float64)
    elif fault=="K":c["camera_K"][0,0]+=1
    elif fault=="handdtype":c["hand_mask_left"]=c["hand_mask_left"].astype(np.uint8)
    else:c["extra"]=np.array(1)
    with pytest.raises(ValueError):gate.validate_candidate(c,record,raw,pair)


def test_proxy_is_fixed_sorted_original_pixels_and_scale(gate,monkeypatch):
    raw,pair,record=frame(gate,monkeypatch);points,indices=gate.object_sample(raw,.83)
    data=dict(object_points_camera_m=points,pixel_indices=indices,shared_depth_scale=np.array(.83,np.float64),clip_index=raw["clip_index"],frame_index=raw["frame_index"])
    gate.validate_proxy(data,record,raw)
    assert np.all(indices[1:]>indices[:-1])
    data["pixel_indices"]=indices[::-1]
    with pytest.raises(ValueError):gate.validate_proxy(data,record,raw)


def trace(gate):
    xy=np.c_[np.arange(133)*10.,np.arange(133)*20.][None]
    dw=dict(keypoints=xy,scores=np.ones((1,133),np.float32))
    target,indices,valid=gate.policy.training_observations(xy[0],dw["scores"][0]);n=len(target)
    J=np.zeros((2*n,5));J[:5]=np.eye(5)
    row=dict(training_validity=valid.tolist(),training_MHR_indices=indices.tolist(),training_points=n,adam_updates=59,heldout_used_for_fit=False,
        heldout_COCO_indices=list(gate.policy.HELDOUT_COCO),observation_compute_dtype="float32",evaluated_latents=np.zeros((60,5)).tolist(),evaluated_physical_deltas=np.zeros((60,5)).tolist(),
        evaluated_projected_points=np.broadcast_to(target.astype(np.float32),(60,n,2)).tolist(),evaluated_losses=np.zeros(60).tolist(),
        update_gradients=np.zeros((59,5)).tolist(),initial_observation_jacobian_values=J.tolist(),best_evaluated_index=0,selected_total_objective=0.,
        selected_physical_delta=np.zeros(5,np.float32).tolist(),initial_observation_jacobian=gate.jacobian_evidence(J,n))
    c=dict(latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32))
    return row,dw,c


def test_trace_exact_all_sixty_states_adam_first_tie_and_rank(gate):
    row,dw,c=trace(gate);result=gate.validate_schedule(row,dw,c)
    assert result["states"]==60 and result["updates"]==59 and result["best_evaluated_index"]==0


@pytest.mark.parametrize("fault",["state","gradient","loss","heldout","support","rank","candidate","tie"])
def test_trace_rejects_public_tampering(gate,fault):
    row,dw,c=trace(gate)
    if fault=="state":row["evaluated_latents"][31][1]=.1
    elif fault=="gradient":row["update_gradients"][30][1]=.1
    elif fault=="loss":row["evaluated_losses"][29]=.01
    elif fault=="heldout":row["heldout_used_for_fit"]=True
    elif fault=="support":row["training_validity"][0]=False
    elif fault=="rank":row["initial_observation_jacobian_values"]=[[0.]*5]*20
    elif fault=="candidate":c["latent"][0]=.1
    else:row["best_evaluated_index"]=1
    with pytest.raises(ValueError):gate.validate_schedule(row,dw,c)


def test_no_private_torch_native_forward_or_dynamic_import(gate):
    tree=ast.parse(MODULE.read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(a.name.startswith(("torch","root5_rgb_fit","root5_rgb_render"))for a in n.names)for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute)and n.attr in("mhr_forward","InferenceSession","import_module")for n in ast.walk(tree))
    assert "eval_private"not in MODULE.read_text()


def stage_report(gate,stage):
    from dataclasses import asdict
    revision=gate.PRODUCER_REVISION if stage=="masks"else gate.NATIVE_PRODUCER_REVISION
    sources=gate.observe.source_identity()
    if stage=="masks":sources=sources|dict(observer=gate.MASK_SOURCE_SHA)
    return dict(stage=gate.observe.STAGES[stage],status="pass",phase="complete",frames=15,cohort=asdict(gate.COHORT),
        producer_revision=revision,script_sha256=sources["observer"],source_helpers=sources,network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],fitting_performed=False,
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,all_cases_retained=True,
        budget_seconds=gate.observe.BUDGETS[stage],gpu_used=stage!="dwpose",image_id=gate.native.IMAGE_ID)


@pytest.mark.parametrize("stage",["masks","baseline","dwpose"])
def test_stage_producers_have_distinct_pinned_sources(gate,stage):
    row=stage_report(gate,stage)
    gate.require_stage(row,stage,row["source_helpers"],row["producer_revision"])
    row["producer_revision"]=gate.PRODUCER_REVISION if stage!="masks"else gate.NATIVE_PRODUCER_REVISION
    with pytest.raises(ValueError):gate.require_stage(row,stage,row["source_helpers"],stage_report(gate,stage)["producer_revision"])


@pytest.mark.parametrize("fault",["GT","partial","boolcount","fallback","source","error"])
def test_stage_safety_contract_is_not_numerical_truth(gate,fault):
    row=stage_report(gate,"baseline");original=json.loads(json.dumps(row))
    if fault=="GT":row["private_truth_read"]=True
    elif fault=="partial":row["frames"]=14
    elif fault=="boolcount":row["frames"]=True
    elif fault=="fallback":row["oracle_modes"]=["knowncamera"]
    elif fault=="source":row["source_helpers"]["observer"]="a"*64
    else:row["error"]="retainederror"
    with pytest.raises(ValueError):gate.require_stage(row,"baseline",original["source_helpers"],original["producer_revision"])


def test_actual_float32_adam_history_with_continuous_reconstruction(gate):
    row,dw,c=trace(gate)
    gradients=np.random.default_rng(19).normal(0,.03,(59,5)).astype(np.float32)
    latent=np.zeros(5,np.float32);m=np.zeros(5,np.float32);v=np.zeros(5,np.float32);states=[latent.copy()]
    for i,g in enumerate(gradients):
        m=np.float32(.9)*m+np.float32(.1)*g;v=np.float32(.999)*v+np.float32(.001)*g*g
        latent-=np.float32(.01)*(m/np.float32(1-.9**(i+1)))/(np.sqrt(v/np.float32(1-.999**(i+1)))+np.float32(1e-8))
        states.append(latent.copy())
    states=np.asarray(states);d=np.tanh(states)*gate.BOUNDS
    row["evaluated_latents"]=states.tolist();row["evaluated_physical_deltas"]=d.tolist();row["update_gradients"]=gradients.tolist()
    target,_,_=gate.policy.training_observations(dw["keypoints"][0],dw["scores"][0]);target=target.astype(np.float32).astype(np.float64)
    row["evaluated_losses"]=[gate.policy.objective(target,target,np.r_[value[:2],0.,value[2:]])["total"]for value in d]
    # First state remains best; the final unevaluated update is never selected.
    assert gate.validate_schedule(row,dw,c)["best_evaluated_index"]==0


def full_public_fixture(gate,tmp_path,monkeypatch):
    """15 real NPZs; only remote model/package attestations use explicit test doubles."""
    from PIL import Image
    base=tmp_path/gate.BASE
    for folder in("inputs","automatic_masks","baseline_v1","baseline_v2/raw","baseline_v2/paired","dwpose_v1"):
        (base/folder).mkdir(parents=True,exist_ok=True)
    records=[];raw=[];pairs=[];rgb=np.zeros((8,16,3),np.uint8);mask=np.full((8,16),255,np.uint8)
    monkeypatch.setattr(gate.native.human,"WIDTH",16);monkeypatch.setattr(gate.native.human,"HEIGHT",8)
    monkeypatch.setattr(gate.native.joint,"WIDTH",16);monkeypatch.setattr(gate.native.joint,"HEIGHT",8)
    for i in range(15):
        r,p,record=frame(gate,monkeypatch,i//5,i%5)
        path=base/"inputs"/record["file"];Image.fromarray(rgb).save(path);path.chmod(0o444)
        record.update(path=path,sha256=gate.protocol.identity(path)["sha256"])
        for label in("human","object"):
            target=base/"automatic_masks"/(Path(record["file"]).stem+"_"+label+".png")
            Image.fromarray(mask).save(target);target.chmod(0o444)
            record[label+"_mask_path"]=target;record[label+"_mask_pixels"]=128
        records.append(record);raw.append(r);pairs.append(p)
    input_path=base/"inputs/manifest.json";input_path.write_text('synthetic public manifest, reader mocked below');input_path.chmod(0o444)
    sources=gate.observe.source_identity();reports={stage:stage_report(gate,stage)for stage in("masks","baseline","dwpose")}
    assets={name:dict(path=str(tmp_path/"weights"/name),sha256=pin[0],bytes=pin[1])for name,pin in gate.observe.masks.ASSETS.items()}
    critical={name:dict(path="/synthetic/sam2/"+name,sha256="e"*64,bytes=123)for name in("build_sam.py","sam2_image_predictor.py","modeling/sam2_base.py","utils/transforms.py")}
    reports["masks"].update(model_assets=assets,sam2_source=dict(vcs=dict(url="https://github.com/facebookresearch/sam2.git",vcs_info=dict(vcs="git",commit_id="d"*40)),
        python_files=4,python_source_sha256="e"*64,prior_upstream_source_pin_verified=False,full_import_license_closure_verified=False,
        critical_sources=critical,config=dict(path="/synthetic/sam2/config.yaml",sha256="e"*64,bytes=99)))
    for row in assets.values():
        path=Path(row["path"]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"test-only-pinned-asset-double");path.chmod(0o444)
    for relative in("results/image-grounding.json","results/weights-acquisition.json"):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}');path.chmod(0o444)
    failed=base/"baseline_v1/report.json";failed.write_text('synthetic zero-call preserved failure');failed.chmod(0o444)
    b=reports["baseline"];b.update(raw_frozen_before_shared=True,paired_frozen_before_reference=True,sources_assets_rechecked=True,
        conversion_fidelity_verified=True,actual_body_inference=True,actual_MoGe_inference=True,actual_shared_native_forward=True,
        body_calls_completed=15,MoGe_calls_completed=15,raw_parity_head_calls_completed=15,raw_keypoint_head_calls_completed=15,
        shared_head_calls_completed=15,official_reference_calls=1,body_inference_type="body",camera_K=[list(r)for r in gate.COHORT.fixed_K],
        focal_fitted=False,scale_fit=False,inverse_fit_performed=False,identity_source="first_original_RGB_shape45_and_scale28_per_five_frame_clip",
        expression_zero=True,keypoint_count=308,joint_count=127,hand_regions_native_verified=True,model_sha256=gate.native.REFERENCE_MODEL_SHA256,
        converter_sha256=gate.native.CONVERTER_SHA256,seed=0,TF32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",official_reference_per_frame_mean_mm=[.01]*15,
        reference_fidelity=dict(per_frame_mean_mm=[.01]*15,max_point_mm_diagnostic=.03,max_joint_distance_mm_diagnostic=.03),
        previous_failed_baseline=gate.protocol.identity(failed),previous_failure_rewritten=False,
        hand_region_sha256={side:gate.hashlib.sha256(pairs[0]["hand_mask_"+side].tobytes()).hexdigest()for side in("left","right")},calls=[],native_focal_solver_calls=[])
    b["raw_outputs"]=[gate.native.save(base/"baseline_v2","raw",r,v)for r,v in zip(records,raw)]
    b["paired_outputs"]=[gate.native.save(base/"baseline_v2","paired",r,v)for r,v in zip(records,pairs)]
    d=reports["dwpose"];d.update(actual_sessions=1,private_prefix_packages_installed=True,private_prefix_removed=True,final_inputs_source_assets_rehashed=True,
        native_cpu_abi_verified=True,native_source_modified=False,own_feed_cast=False,channel_swap=False,full_image_fallback=False,
        raw_scores_clamped=False,confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,records=[],calls=[],
        graph=dict(inputs=[dict(name="input",type="tensor(float)",shape=["batch",3,384,288])],outputs=[dict(name="simcc_"+a,type="tensor(float)",shape=["batch","MatMulsimcc_"+a+"_dim_1","MatMulsimcc_"+a+"_dim_2"])for a in("x","y")],providers=["CPUExecutionProvider"],custom_metadata={}))
    for record,original in zip(records,raw):
        b["calls"].append(dict(file=record["file"],bbox_xyxy=[0.,0.,16.,8.],decoded_RGB_sha256=gate.hashlib.sha256(rgb.tobytes()).hexdigest(),
            native_forward_errors={k:0. for k in("vertices_m","joints_m","keypoints_m","controls")}))
        b["native_focal_solver_calls"].append({k:record[k]for k in("file","clip_index","frame_index")}|dict(native_nearest64_valid_pixels=4096,focal_prior_supplied=True,original_solver_returned=True))
        data=dict(keypoints=np.c_[np.arange(133)*10.,np.arange(133)*20.][None],scores=np.ones((1,133),np.float32),validity=np.ones((1,133),bool),
            bbox=np.array([[0,0,16,8]],np.float32),clip_index=original["clip_index"],frame_index=original["frame_index"])
        path=base/"dwpose_v1"/(Path(record["file"]).stem+".npz");np.savez(path,**data);path.chmod(0o444)
        simcc=[dict(dtype="float32",shape=s,sha256="f"*64)for s in([1,133,576],[1,133,768])]
        d["records"].append({k:record[k]for k in("file","clip_index","frame_index")}|dict(rgb_sha256=record["sha256"],prediction_file=path.name,
            prediction=gate.protocol.identity(path),raw_simcc=simcc,positive_score_count=133,**{k:gate.observe.smoke.array_identity(data[k])for k in("keypoints","scores","validity","bbox")}))
        d["calls"].append(dict(supplied_container="list",effective_feed_shape=[1,3,384,288],effective_runtime_conversion_observed=False,
            delegated_unmodified=True,run_completed=True,supplied_array=dict(dtype="float64",shape=[3,384,288],sha256="f"*64),raw_simcc=simcc))
    value=pins(gate)
    monkeypatch.setattr(gate,"observer_sources",lambda *args:[])
    monkeypatch.setattr(gate.observe,"mask_source_helpers",lambda root:sources|dict(observer=gate.MASK_SOURCE_SHA))
    monkeypatch.setattr(gate.observe,"read_rgb",lambda *args:rgb.copy())
    monkeypatch.setattr(gate.observe.masks,"validate_assets",lambda *args:assets)
    monkeypatch.setattr(gate,"validate_body_bindings",lambda *args:[])
    monkeypatch.setattr(gate,"dw_bound_files",lambda *args:[])
    monkeypatch.setattr(gate.observe,"previous_baseline_failure",lambda *args:gate.protocol.identity(failed))
    def seal():
        # Report pin constants are replaced only for these own procedural fixture bytes.
        for stage in("masks","baseline","dwpose"):
            path=base/gate.observe.FOLDERS[stage]/"report.json"
            if path.exists():path.chmod(0o644)
            path.write_text(json.dumps(reports[stage]));path.chmod(0o444)
            key=dict(masks="automatic_masks_sha256",baseline="baseline_sha256",dwpose="dwpose_sha256")[stage]
            value[key]=gate.protocol.identity(path)["sha256"]
    seal()
    inputs=dict(manifest=dict(sha256=gate.MANIFEST_SHA,bytes=2196),automatic_masks=gate.protocol.identity(base/"automatic_masks/report.json"))
    b["public_inputs"]=inputs;d["public_inputs"]=inputs;seal()
    monkeypatch.setattr(gate.observe,"public_inputs",lambda *args:(records,inputs))
    return value,reports,records,seal


def test_fifteen_real_npz_artifacts_audited_without_private_read(gate,tmp_path,monkeypatch):
    value,reports,records,_=full_public_fixture(gate,tmp_path,monkeypatch)
    private=tmp_path/gate.BASE/"eval_private";private.mkdir();(private/"forbidden.json").write_text('never read')
    original=Path.open
    def guarded(path,*args,**kwargs):
        if "eval_private"in path.parts:raise AssertionError("Private payload read before frozen predictions")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    r,raw,pairs,dw,lineage=gate.public_predictions(tmp_path,value)
    assert len(r)==len(raw)==len(pairs)==len(dw)==15 and lineage["public_pins"]==value
    assert not any("eval_private"in row["path"]for row in lineage["frozen_files"])
    assert lineage["observer_revisions"]["masks"]!=lineage["observer_revisions"]["baseline"]


@pytest.mark.parametrize("fault",["reference","frameorder","bbox","rawmask","pairidentity","DWdtype","simcc","rawRGB","source","extra"])
def test_full_public_consumer_rejects_tampered_frozen_producer(gate,tmp_path,monkeypatch,fault):
    value,reports,records,seal=full_public_fixture(gate,tmp_path,monkeypatch)
    base=tmp_path/gate.BASE
    if fault=="reference":reports["baseline"]["official_reference_per_frame_mean_mm"][1]=2.01
    elif fault=="frameorder":reports["dwpose"]["records"][1]["frame_index"]=2
    elif fault=="bbox":reports["baseline"]["calls"][0]["bbox_xyxy"][0]=1.
    elif fault=="simcc":reports["dwpose"]["calls"][0]["raw_simcc"]=[]
    elif fault=="rawRGB":reports["baseline"]["calls"][0]["decoded_RGB_sha256"]="a"*64
    elif fault=="source":reports["baseline"]["source_helpers"]["observer"]="a"*64
    elif fault=="extra":(base/"baseline_v2/raw/extra.npz").write_bytes(b"extra")
    else:
        folder="raw"if fault=="rawmask"else"paired"if fault=="pairidentity"else"DW"
        rows=reports["baseline"]["raw_outputs"if folder=="raw"else"paired_outputs"]if folder!="DW"else reports["dwpose"]["records"]
        path=base/("baseline_v2/"+rows[0]["artifact"]if folder!="DW"else"dwpose_v1/"+rows[0]["prediction_file"])
        with np.load(path,allow_pickle=False)as archive:data={k:archive[k]for k in archive.files}
        if fault=="rawmask":data["human_mask"][0,0]=False
        elif fault=="pairidentity":data["shared_shape_params"][0]=.1
        else:data["keypoints"]=data["keypoints"].astype(np.float32)
        path.chmod(0o644);np.savez(path,**data);path.chmod(0o444)
        identity=gate.protocol.identity(path)
        if folder!="DW":rows[0].update(identity)
        else:rows[0]["prediction"]=identity
    seal()
    with pytest.raises(ValueError):gate.public_predictions(tmp_path,value)


def test_actual_canonical_source_inventory_not_imported_or_replaced(gate,tmp_path,monkeypatch):
    historical=tmp_path/"historical-mask.py";historical.write_bytes(b"test-only historical SHA double");historical.chmod(0o444)
    monkeypatch.setattr(gate.observe,"historical_mask_source",lambda root:(historical,gate.protocol.identity(historical)))
    modules=(gate.observe,gate.protocol,gate.observe.masks,gate.baseline,gate.observe.dw_helper,gate.observe.depth_camera,gate.native,
        gate.native.masks,gate.native.masks.masks,gate.native.human,gate.native.human.body,gate.native.depth_model,gate.native.joint,
        gate.native.regions_helper,gate.baseline.masks,gate.observe.smoke,gate.observe.smoke.acquisition,gate.observe.smoke.audit)
    current=[Path(m.__file__)for m in modules]+[Path(gate.native.__file__).with_name("camera_render.py"),Path(gate.observe.smoke.__file__).with_name("run_dwpose_smoke.sh")]
    folder=tmp_path/"jobs"/gate.NATIVE_PRODUCER_REVISION/"run_root5_rgb_observe/code/infra";folder.mkdir(parents=True)
    for path in {p.name:p for p in current}.values():
        target=folder/path.name;target.write_bytes(path.read_bytes());target.chmod(0o444)
    actual=gate.observer_sources(tmp_path,pins(gate)["producer_revision"])
    assert len(actual)>15 and historical in actual
    target=folder/"hand_synthetic_infer.py";target.chmod(0o644);target.write_bytes(b"altered source");target.chmod(0o444)
    with pytest.raises(ValueError):gate.observer_sources(tmp_path,pins(gate)["producer_revision"])


@pytest.mark.parametrize("fault",["trainablecount","bufferdup","buffercount","unexpected","mode","checkpoint","revision"])
def test_real_loader_shaped_checkpoint_evidence_fail_closed(gate,tmp_path,monkeypatch,fault):
    model=dict(body_revision=gate.native.human.body.BODY_REVISION,upstream_revision=gate.native.human.body.UPSTREAM_REVISION,
        dinov3_revision=gate.native.human.body.DINOV3_REVISION,body_assets={"model.ckpt":dict(sha256=gate.native.human.BODY_SHA,bytes=gate.native.human.BODY_BYTES)},
        checkpoint_loading=dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",unexpected_keys=[],parameter_tensors_loaded=1101,
            retained_mhr_asset_buffer_names=["head_pose.mhr.asset.buffer_"+str(i)for i in range(113)]))
    report=dict(body_model=model)
    if fault=="trainablecount":model["checkpoint_loading"]["parameter_tensors_loaded"]=1100
    elif fault=="bufferdup":model["checkpoint_loading"]["retained_mhr_asset_buffer_names"][1]=model["checkpoint_loading"]["retained_mhr_asset_buffer_names"][0]
    elif fault=="buffercount":model["checkpoint_loading"]["retained_mhr_asset_buffer_names"].pop()
    elif fault=="unexpected":model["checkpoint_loading"]["unexpected_keys"]=["randomweight"]
    elif fault=="mode":model["checkpoint_loading"]["mode"]="non-strict"
    elif fault=="checkpoint":model["body_assets"]["model.ckpt"]["sha256"]="a"*64
    else:model["body_revision"]="a"*40
    monkeypatch.setattr(gate.native.human.body,"_source_identity",lambda root:pytest.fail("Invalid checkpoint metadata passed source gate"))
    with pytest.raises(ValueError):gate.validate_body_bindings(tmp_path,report)


def test_public_import_does_not_load_torch(gate):
    assert gate.body is gate.baseline and callable(gate.validate_schedule)
    # Module-level imports must remain ordinary CPU sources, without dynamic/native execution.
    tree=ast.parse(MODULE.read_text())
    assert not any(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)
        and n.value.func.attr in("mhr_forward","load_model","InferenceSession")for n in tree.body)

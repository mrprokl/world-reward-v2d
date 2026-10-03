"""Data-free D96 public/optimizer contracts, not actual native CUDA evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

INFRA = Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA)); monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec = importlib.util.spec_from_file_location("keypoint_fit_test", INFRA/"keypoint_rgb_fit.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def frame(gate, monkeypatch, clip=0, index=0):
    n = gate.native; monkeypatch.setattr(n, "WIDTH", 16); monkeypatch.setattr(n, "HEIGHT", 8); monkeypatch.setattr(n, "VERTICES", 100)
    yy, xx = np.indices((8, 16)); z = np.full((8, 16), 2., np.float32); K=gate.baseline.CAMERA_K.copy()
    points=np.stack(((xx+.5-K[0,2])/K[0,0]*z,(yy+.5-K[1,2])/K[1,1]*z,z),-1).astype(np.float32)
    raw={k:np.zeros(s,np.float32) for k,s in n.BLOCKS.items()}; raw["pred_cam_t"][:]=[0,0,2]
    raw.update(raw_depth=z,raw_points=points,validity=np.ones((8,16),bool),rendered_depth=z.copy(),silhouette=np.ones((8,16),bool),
        human_mask=np.ones((8,16),bool),object_mask=np.ones((8,16),bool),raw_vertices_camera_m=np.ones((100,3),np.float32),
        raw_joints_camera_m=np.ones((127,3),np.float32),raw_keypoints_camera_m=np.ones((308,3),np.float32),
        raw_joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)),human_faces=np.zeros((36874,3),np.int64),camera_K=K,
        clip_index=np.array(clip,np.int64),frame_index=np.array(index,np.int64))
    pair={}
    for mode in ("raw","shared"):
        for suffix,source in (("vertices_camera_m","raw_vertices_camera_m"),("joints_camera_m","raw_joints_camera_m"),
                ("keypoints_camera_m","raw_keypoints_camera_m"),("joint_global_rotations","raw_joint_global_rotations"),
                ("model_controls","mhr_model_params"),("shape_params","shape_params"),("scale_params","scale_params")):
            pair[mode+"_"+suffix]=raw[source].copy()
    pair.update(human_faces=raw["human_faces"].copy(),camera_K=K,pred_cam_t=raw["pred_cam_t"].copy(),expression=raw["expr_params"].copy(),
        hand_mask_left=np.arange(100)<50,hand_mask_right=np.arange(100)>=50,clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy())
    record=dict(file=f"clip_{clip:02d}_frame_{index:03d}.png",clip_index=clip,frame_index=index,sha256="a"*64)
    return raw,pair,record


def candidate(gate,raw,pair):
    result={k:raw[k].copy() for k in gate.native.BLOCKS}
    for k,source in (("vertices_camera_m","shared_vertices_camera_m"),("keypoints_camera_m","shared_keypoints_camera_m"),
            ("joints_camera_m","shared_joints_camera_m"),("joint_global_rotations","shared_joint_global_rotations"),
            ("human_faces","human_faces"),("camera_K","camera_K"),("hand_mask_left","hand_mask_left"),("hand_mask_right","hand_mask_right")):
        result[k]=pair[source].copy()
    result.update(clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy(),latent=np.zeros(6,np.float32),physical_delta=np.zeros(6,np.float32))
    return result


def dw(gate,record):
    xy=np.c_[np.arange(133)*10.,np.arange(133)*20.][None]
    return dict(keypoints=xy,scores=np.ones((1,133),np.float32),validity=np.ones((1,133),bool),bbox=np.array([[0,0,16,8]],np.float32),
        clip_index=np.array(record["clip_index"],np.int64),frame_index=np.array(record["frame_index"],np.int64))


def test_exact_schedule_and_first_tie_only_evaluated(gate):
    evaluated=[];updated=[]
    loss,best=gate.optimization_schedule(lambda i:evaluated.append(i) or (0. if i in (20,40) else 1.),updated.append)
    assert evaluated==list(range(60)) and updated==list(range(59)) and best==20 and len(loss)==60


@pytest.mark.parametrize("invalid",[np.nan,np.inf,-1.])
def test_schedule_fails_before_fabricated_remaining_states(gate,invalid):
    updates=[]
    with pytest.raises(ValueError):gate.optimization_schedule(lambda i:invalid if i==3 else 1.,updates.append)
    assert updates==[0,1,2]


def test_candidate_exact_fixed_geometry_and_zero_latents(gate,monkeypatch):
    raw,pair,record=frame(gate,monkeypatch);data=candidate(gate,raw,pair)
    assert gate.validate_candidate(data,record,raw,pair) is data
    assert gate.baseline.validate_raw(raw,record) is raw
    assert gate.baseline.validate_pair(pair,record,raw) is pair


@pytest.mark.parametrize("fault",["shape","scales","body","hands","expression","roottranslation","rootEuler","expanded","K","region","negativeZ","rotation","delta","latent","mask","frame","extra"])
def test_candidate_tamper_fails_no_extra_degrees_of_freedom(gate,monkeypatch,fault):
    raw,pair,record=frame(gate,monkeypatch);data=candidate(gate,raw,pair)
    if fault in ("shape","scales","body","hands","expression"):
        key={"shape":"shape_params","scales":"scale_params","body":"body_pose_params","hands":"hand_pose_params","expression":"expr_params"}[fault];data[key][0]=.01
    elif fault=="roottranslation":data["mhr_model_params"][0]=1
    elif fault=="rootEuler":data["mhr_model_params"][3]=.01
    elif fault=="expanded":data["mhr_model_params"][136]=.1
    elif fault=="K":data["camera_K"][0,0]+=1
    elif fault=="region":data["hand_mask_left"][0]=False
    elif fault=="negativeZ":data["keypoints_camera_m"][0,2]=0
    elif fault=="rotation":data["joint_global_rotations"][0,0,0]=-1
    elif fault=="delta":data["physical_delta"][0]=.301
    elif fault=="latent":data["latent"][0]=.1
    elif fault=="mask":data["latent"]=np.ma.array(data["latent"],mask=False)
    elif fault=="frame":data["frame_index"]=np.array(1,np.int64)
    else:data["oracle_pose"]=np.array(1.)
    with pytest.raises(ValueError):gate.validate_candidate(data,record,raw,pair)


def test_proxy_seed_sorted_and_common_positive_scale(gate,monkeypatch):
    raw,_,record=frame(gate,monkeypatch)
    points,pixels=gate.object_sample(raw,1.5)
    assert len(points)==128 and np.array_equal(pixels,np.arange(128))
    assert np.array_equal(points,raw["raw_points"].reshape(-1,3)*np.float32(1.5))
    data=dict(object_points_camera_m=points,pixel_indices=pixels,shared_depth_scale=np.array(1.5),clip_index=raw["clip_index"],frame_index=raw["frame_index"])
    assert gate.validate_proxy(data,record,raw) is data
    raw["object_mask"][:]=False
    with pytest.raises(ValueError):gate.object_sample(raw,1.5)


def test_proxy_sampling_8192_stable_independent_of_pose(gate,monkeypatch):
    raw,_,_=frame(gate,monkeypatch)
    raw["object_mask"]=np.ones((100,100),bool);raw["validity"]=np.ones((100,100),bool);raw["raw_points"]=np.ones((100,100,3),np.float32)
    a,i=gate.object_sample(raw,.8);b,j=gate.object_sample(raw,.8)
    assert a.tobytes()==b.tobytes() and i.tobytes()==j.tobytes() and len(i)==8192 and np.all(np.diff(i)>0)


@pytest.mark.parametrize("fault",["few","collapsed","scoresNaN","validity","bbox","masked","shape"])
def test_independent_observation_strict_native133_no_confidence_fallback(gate,monkeypatch,fault):
    _,_,record=frame(gate,monkeypatch);data=dw(gate,record)
    if fault=="few":data["scores"][0,list(gate.policy.TRAIN_COCO[:5])]=0;data["validity"]=data["scores"]>0
    elif fault=="collapsed":data["keypoints"][:,:,0]=0
    elif fault=="scoresNaN":data["scores"][0,0]=np.nan
    elif fault=="validity":data["validity"][0,0]=False
    elif fault=="bbox":data["bbox"][0,2]=17
    elif fault=="masked":data["scores"]=np.ma.array(data["scores"],mask=False)
    else:data["keypoints"]=data["keypoints"][:,:17]
    with pytest.raises(ValueError):gate.public.validate_dw(data,record)


def public_fixture(gate,tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("baseline_fixture_source",Path(__file__).with_name("test_keypoint_rgb_baseline.py"))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    _,_,base=fixture.public_fixture(gate.baseline,tmp_path,monkeypatch)
    records,_=gate.baseline.public_inputs(tmp_path)
    bp=base/"baseline_v1";dp=base/"dwpose_v1";bp.mkdir();dp.mkdir();(bp/"raw").mkdir();(bp/"paired").mkdir()
    b=dict(stage=gate.baseline.STAGE,status="pass",phase="complete",producer_revision=gate.public.BASELINE_REVISION,
        script_sha256=gate.public.BASELINE_SCRIPT,frames=15,all_cases_retained=True,network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],raw_frozen_before_shared=True,
        paired_frozen_before_reference=True,sources_assets_rechecked=True,conversion_fidelity_verified=True,actual_body_inference=True,
        actual_MoGe_inference=True,actual_shared_native_forward=True,body_calls_completed=15,MoGe_calls_completed=15,
        raw_parity_head_calls_completed=15,raw_keypoint_head_calls_completed=15,shared_head_calls_completed=15,official_reference_calls=1,
        accuracy_verified=False,adoption_performed=False,official_reference_per_frame_mean_mm=[.001]*15,raw_outputs=[],paired_outputs=[],
        hand_region_sha256={})
    d=dict(stage="public_keypoint_rgb_native_dwpose133_observations",status="pass",phase="complete",producer_revision=gate.public.DW_REVISION,
        script_sha256=gate.public.DW_SCRIPT,image_id=gate.native.IMAGE_ID,device="cpu",network="none",actual_sessions=1,requested_calls=15,
        all_cases_retained=True,native_cpu_abi_verified=True,final_inputs_source_assets_rehashed=True,private_prefix_removed=True,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],fitting_performed=False,
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,native_source_modified=False,own_feed_cast=False,channel_swap=False,
        full_image_fallback=False,raw_scores_clamped=False,confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,gpu_used=False,
        graph=dict(inputs=[dict(name="input",type="tensor(float)",shape=["batch",3,384,288])],
            outputs=[dict(name="simcc_"+a,type="tensor(float)",shape=["batch","MatMulsimcc_"+a+"_dim_1","MatMulsimcc_"+a+"_dim_2"])for a in ("x","y")],
            providers=["CPUExecutionProvider"],custom_metadata={}),calls=[],records=[])
    for record in records:
        raw,pair,_=frame(gate,monkeypatch,record["clip_index"],record["frame_index"])
        b["raw_outputs"].append(gate.native.save(bp,"raw",record,raw));b["paired_outputs"].append(gate.native.save(bp,"paired",record,pair))
        data=dw(gate,record);name=Path(record["file"]).stem+".npz";path=dp/name
        with path.open("xb")as stream:np.savez(stream,**data)
        path.chmod(0o444);simcc=[dict(dtype="float32",shape=s,sha256="a"*64)for s in ([1,133,576],[1,133,768])]
        call=dict(supplied_container="list",supplied_array=dict(dtype="float64",shape=[3,384,288],sha256="b"*64),
            effective_feed_shape=[1,3,384,288],effective_runtime_conversion_observed=False,delegated_unmodified=True,run_completed=True,raw_simcc=simcc)
        d["calls"].append(call);d["records"].append({k:record[k]for k in ("file","clip_index","frame_index")}|dict(rgb_sha256=record["sha256"],prediction_file=name,
            prediction={k:gate.native.regular(path)[k]for k in ("sha256","bytes")},raw_simcc=simcc,**{k:gate.public.array_identity(data[k])for k in ("keypoints","scores","validity","bbox")}))
    b["hand_region_sha256"]={s:gate.public.array_identity(pair["hand_mask_"+s])["sha256"]for s in ("left","right")}
    def save():
        for path,data,key in ((bp/"report.json",b,"BASELINE_SHA"),(dp/"report.json",d,"DW_SHA")):
            if path.exists():path.chmod(0o644)
            path.write_text(json.dumps(data));path.chmod(0o444);monkeypatch.setattr(gate.public,key,gate.sha256(path))
    save()
    monkeypatch.setattr(gate.baseline,"validate_producer_bindings",lambda root,report:True)
    monkeypatch.setattr(gate.public,"dw_bound_files",lambda root,report:[])
    return base,b,d,save


def test_full_public_real_arrays_audited_before_fit_no_private(gate,tmp_path,monkeypatch):
    base,_,_,_=public_fixture(gate,tmp_path,monkeypatch)
    records,raw,pairs,dwrows,lineage=gate.public_predictions(tmp_path)
    assert len(records)==len(raw)==len(pairs)==len(dwrows)==15 and len(lineage["frozen_files"])==94
    assert not (base/"eval_private").exists()


@pytest.mark.parametrize("fault",["baselineGT","baselinecounts","fidelity","DWcast","DWshape","DWscore","filetamper","extra"])
def test_public_chain_tamper_stops_before_any_fit_private(gate,tmp_path,monkeypatch,fault):
    base,b,d,save=public_fixture(gate,tmp_path,monkeypatch)
    if fault=="baselineGT":b["ground_truth_used"]=True
    elif fault=="baselinecounts":b["body_calls_completed"]=14
    elif fault=="fidelity":b["official_reference_per_frame_mean_mm"][0]=2.1
    elif fault=="DWcast":d["own_feed_cast"]=True
    elif fault=="DWshape":d["calls"][0]["raw_simcc"][0]["shape"][1]=134
    elif fault=="DWscore":d["records"][0]["scores"]["sha256"]="c"*64
    elif fault=="extra":(base/"dwpose_v1/extra.npz").write_bytes(b"noise")
    else:
        path=base/"dwpose_v1/clip_00_frame_000.npz";path.chmod(0o644);path.write_bytes(b"changed");path.chmod(0o444)
    save()
    with pytest.raises(ValueError):gate.public_predictions(tmp_path)


def test_shell_syntax_offline_scoped_outputs_and_no_private(gate):
    path=INFRA/"run_keypoint_rgb_fit.sh";subprocess.run(["bash","-n",str(path)],check=True)
    text=path.read_text()
    assert "--network none"in text and "183s"in text and "--memory 32g"in text
    assert "eval_private"not in text and 'OUT="$BASE/root_fit_v1"'in text
    assert "src=$BASE/baseline_v1,dst=$BASE/baseline_v1,readonly"in text
    assert "src=$BASE/dwpose_v1,dst=$BASE/dwpose_v1,readonly"in text


def test_runtime_does_not_import_or_suppress_grad_backend(gate):
    assert "torch"not in gate.__dict__
    source=Path(gate.__file__).read_text()
    assert "torch.use_deterministic_algorithms(True, warn_only=False)"in source
    assert "use_deterministic_algorithms(False"not in source and "inference_mode()"not in source
    assert "get_parameter_limits()"in source and "initial_jacobian_rows"in source


def test_source_only_historical_fit_hash_same_without_import(gate,tmp_path):
    source=tmp_path/"jobs/revision/run_keypoint_rgb_fit/code/infra/keypoint_rgb_fit.py"
    source.parent.mkdir(parents=True);source.write_bytes(Path(gate.__file__).read_bytes());source.chmod(0o444)
    assert gate.public.helper_identities(source)==gate.public.helper_identities()
    source.chmod(0o644);source.write_bytes(b"different frozen producer")
    assert gate.public.helper_identities(source)["fit"]!=gate.public.helper_identities()["fit"]
    link=tmp_path/"alias.py";link.symlink_to(source)
    with pytest.raises(ValueError):gate.public.helper_identities(link)


@pytest.mark.parametrize("fault",[None,"asset","notice","source","sourcepath","receipt"])
def test_receipt_bound_nine_assets_notices_historical_sources_rehashed(gate,tmp_path,fault):
    root=tmp_path; assets=[]
    for i in range(9):
        name=f"source/asset{i}.txt";p=root/"weights/dwpose_native_v1"/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(i).encode())
        row={"file":name,"url":"https://example.invalid/pinned",**{k:gate.native.regular(p)[k]for k in ("sha256","bytes")}};assets.append(row)
    notice=root/"results/dwpose-wheel-audit-v3/flatbuffers/LICENSE";notice.parent.mkdir(parents=True);notice.write_bytes(b"notice")
    a=dict(files=assets,retained_notices=[dict(file="flatbuffers/LICENSE",**{k:gate.native.regular(notice)[k]for k in ("sha256","bytes")})])
    paths=["results/dwpose-acquisition-v1.json","results/dwpose-wheel-audit-v2/report.json","results/dwpose-wheel-audit-v3/report.json",
           "validation/dwpose_smoke_v2/report.json","validation/dwpose_smoke_v1/report.json"]
    receipts=[]
    for relative in paths:
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("{}");p.chmod(0o444);receipts.append(gate.sha256(p))
    a.update(original_receipt_sha256=receipts[0],previous_failed_receipt_sha256=receipts[1],audit_receipt={"sha256":receipts[2]})
    directory=root/"jobs"/gate.public.DW_REVISION/"run_keypoint_rgb_dwpose/code/infra";directory.mkdir(parents=True)
    source=directory/"onnxpose.py";source.write_bytes(b"# pinned own source fixture\n")
    report=dict(assets=a,capability_evidence=dict(actual_pass_receipt={"sha256":receipts[3]},previous_failed_receipt={"sha256":receipts[4]}),
        sources={source.name:{k:gate.native.regular(source)[k]for k in ("sha256","bytes")}})
    if fault=="asset":(root/"weights/dwpose_native_v1"/assets[0]["file"]).write_bytes(b"changed")
    elif fault=="notice":notice.write_bytes(b"changed")
    elif fault=="source":source.write_bytes(b"changed")
    elif fault=="sourcepath":report["sources"]["../onnxpose.py"]=report["sources"].pop("onnxpose.py")
    elif fault=="receipt":p=root/paths[0];p.chmod(0o644);p.write_text("tamper");p.chmod(0o444)
    if fault:
        with pytest.raises(ValueError):gate.public.dw_bound_files(root,report)
    else:
        files=gate.public.dw_bound_files(root,report)
        assert len(files)==16 and source in files and notice in files

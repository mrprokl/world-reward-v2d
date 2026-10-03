"""Tiny manufactured numerical/firewall fixtures; no GPU/models/private media."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("factorial_quality_test",ROOT/"infra/factorial_rgb_evaluate.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def freeze(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_text(json.dumps(value));path.chmod(0o444)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def pins(gate,tmp_path):
    p=tmp_path/"public.json";freeze(p,{"public":True})
    value=dict(schema=gate.PIN_SCHEMA,public_pins_sha256=gate.sha256(p),**{r:dict(producer_revision="a"*40,report_sha256="b"*64,report_bytes=20,script_sha256="c"*64)for r in gate.ROLES})
    q=tmp_path/"quality.json";freeze(q,value);return p,q,value


@pytest.mark.parametrize("fault",["none","missing","extra","schema","source","revision","size","bool","public","writable"])
def test_completed_pins_no_discovery_or_repair(gate,tmp_path,fault):
    p,q,v=pins(gate,tmp_path)
    if fault=="none":assert gate.load_pins(q,p)[0]==v;return
    if fault=="missing":v.pop("render")
    elif fault=="extra":v["private_labels"]="forbidden"
    elif fault=="schema":v["schema"]="other"
    elif fault=="source":v["body"]["script_sha256"]="B"*64
    elif fault=="revision":v["body"]["producer_revision"]="main"
    elif fault=="size":v["body"]["report_bytes"]=0
    elif fault=="bool":v["body"]["report_bytes"]=True
    elif fault=="public":v["public_pins_sha256"]="0"*64
    freeze(q,v)
    if fault=="writable":q.chmod(0o644)
    with pytest.raises(ValueError):gate.load_pins(q,p)


def cloud(count):
    rng=np.random.default_rng(33);p=rng.normal(0,.10,(count,3));p[:,2]+=3.;return p


def test_rms_decomposes_bias_not_mean_norm(gate):
    target=cloud(8);pred=target+[.03,.04,.12];result=gate.geometry_metrics(pred,target)
    assert result["pve_cm"]==pytest.approx(13.)
    assert result["centered_rms_cm"]==pytest.approx(0,abs=1e-12)
    pred[::2,0]+=.1;pred[1::2,0]-=.1;r=gate.geometry_metrics(pred,target)
    assert r["rms_cm"]**2==pytest.approx(r["centroid_error_cm"]**2+r["centered_rms_cm"]**2)
    assert r["rms_cm"]!=pytest.approx(r["pve_cm"])


def frame(gate):
    kp=cloud(308);v=cloud(30);j=cloud(127);K=np.asarray(gate.COHORT.fixed_K)
    body=dict(vertices_camera_m=v.copy(),keypoints_camera_m=kp.copy(),joints_camera_m=j.copy())
    truth=dict(human_vertices_camera_m=v,human_keypoints_camera_m=kp,human_joints_camera_m=j,camera_K=K)
    xy=np.zeros((1,133,2));xy[0,:17]=gate.project(kp[np.asarray(gate.COCO_MHR)],K)
    dw=dict(keypoints=xy,validity=np.zeros((1,133),bool));return body,dw,truth


def test_detector_vs_body_same_raw_positive_indices(gate):
    b,d,t=frame(gate);b["keypoints_camera_m"][gate.COCO_MHR[0],0]+=.02
    d["validity"][0,1]=True;d["keypoints"][0,1,0]+=1
    r=gate.frame_metrics(b,d,t)
    assert r["body_all17_reprojection_px"]>0 and r["body_paired_reprojection_px"]==0
    assert r["dwpose_paired_reprojection_px"]==1 and r["native_DW_positive_count"]==1
    d["validity"][:]=False;r=gate.frame_metrics(b,d,t)
    assert r["body_paired_reprojection_px"]is None and r["dwpose_paired_reprojection_px"]is None


def rows(gate,gains=None):
    gains=[.2]*8 if gains is None else gains;result=[]
    for i in range(24):
        b,d,t=frame(gate);d["validity"][0,:17]=True;r=gate.frame_metrics(b,d,t)
        r.update(group_index=i//3,frame_index=i%3,body_paired_reprojection_px=10.,dwpose_paired_reprojection_px=10*(1-gains[i//3]),
            body_all17_reprojection_px=10.,body_COCO17_reprojection_px=[10.]*17,dwpose_COCO17_errors_px=[10*(1-gains[i//3])]*17,
            first_human_sim3_geometry=r["body_geometry"])
        result.append(r)
    return result


def test_equal_eight_groups_rule_no_missing_frame_drop(gate):
    data=rows(gate,[.2]*6+[0.]*2);groups,contrasts,decision=gate.aggregate(data)
    assert len(groups)==8 and len(contrasts)==24 and decision["automatic_prompt_pilot_evidence_supported"]
    assert not decision["adoption_authorized"]and not decision["CARI4D_superiority_verified"]
    data[23].update(native_DW_positive_count=0,body_paired_reprojection_px=None,dwpose_paired_reprojection_px=None)
    _,_,d=gate.aggregate(data)
    assert not d["automatic_prompt_pilot_evidence_supported"]and len(d["supporting_groups"])==6
    with pytest.raises(ValueError):gate.aggregate(data[:-1])


def test_factor_contrasts_intersection_not_different_positive_subset(gate):
    data=rows(gate);data[0]["native_DW_positive_COCO17"]=[True]+[False]*16;data[6]["native_DW_positive_COCO17"]=[False,True]+[False]*15
    _,contrasts,_=gate.aggregate(data)
    c=next(c for c in contrasts if c["factor"]=="appearance"and c["level0_group"]==0 and c["frame_index"]==0)
    assert c["common_native_positive_count"]==0 and c["dwpose_common_positive_reprojection_level1_minus_level0_px"]is None


def test_first_human_sim3_shared_future_pose_cannot_erase_later_drift(gate):
    target=cloud(10);R=gate.alignment.np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    pred=(target-[.1,.2,.3])@R/1.2;fit=gate.alignment.fit_human_sim3(pred,target)
    np.testing.assert_allclose(gate.alignment.transform(pred,fit),target,atol=1e-14)
    late=gate.alignment.transform(pred+[.1,0,0],fit)
    assert gate.geometry_metrics(late,target)["pve_cm"]==pytest.approx(12.)


@pytest.mark.parametrize("fault",["none","partial","cast","delegate","output","row","sha"])
def test_native_dw_actual_list_feed_trace(gate,fault):
    array=lambda shape,dtype:dict(shape=shape,dtype=dtype,sha256="a"*64)
    call=dict(supplied_container="list",effective_feed_shape=[1,3,384,288],effective_runtime_conversion_observed=False,
        delegated_unmodified=True,run_completed=True,supplied_array=array([3,384,288],"float64"),
        raw_simcc=[array([1,133,576],"float32"),array([1,133,768],"float32")])
    report=dict(calls=[json.loads(json.dumps(call))for _ in range(24)],records=[dict(raw_simcc=call["raw_simcc"])for _ in range(24)])
    if fault=="none":gate.dwpose_call_trace(report);return
    if fault=="partial":report["calls"].pop()
    elif fault=="cast":report["calls"][0]["supplied_array"]["dtype"]="float32"
    elif fault=="delegate":report["calls"][0]["delegated_unmodified"]=False
    elif fault=="output":report["calls"][0]["raw_simcc"][0]["shape"]=[1,134,576]
    elif fault=="row":report["records"][0]["raw_simcc"]=[]
    else:report["calls"][0]["supplied_array"]["sha256"]="bad"
    with pytest.raises(ValueError):gate.dwpose_call_trace(report)


def truth(gate,monkeypatch):
    hf=np.tile([0,1,2],(36874,1)).astype(np.int32);monkeypatch.setattr(gate,"FACE_SHA",hashlib.sha256(hf.tobytes()).hexdigest())
    ids=np.full((768,1024),-1,np.int64);ids.flat[:64]=0;ids.flat[64:128]=36874
    depth=np.full(ids.shape,np.nan,np.float32);depth[ids>=0]=2.
    data=dict(human_vertices_camera_m=cloud(18439),human_faces=hf,object_vertices_camera_m=cloud(194),
        object_faces=np.tile([0,1,2],(384,1)).astype(np.int64),camera_K=np.asarray(gate.COHORT.fixed_K),scene_depth_m=depth,
        visible_face_indices=ids,group_index=np.array(0,np.int64),frame_index=np.array(0,np.int64),
        human_joints_camera_m=cloud(127),human_keypoints_camera_m=cloud(308))
    return data,hf.astype(np.int64),dict(group_index=0,frame_index=0)


@pytest.mark.parametrize("fault",["none","typedfaces","nativefaces","kp32","K","ids","depth","missing","identity","object"])
def test_exact_full11_manufacturing_truth(gate,monkeypatch,fault):
    t,f,r=truth(gate,monkeypatch)
    if fault=="none":gate.validate_truth(t,r,f);return
    if fault=="typedfaces":t["human_faces"]=t["human_faces"].astype(np.int64)
    elif fault=="nativefaces":f=f.astype(np.int32)
    elif fault=="kp32":t["human_keypoints_camera_m"]=t["human_keypoints_camera_m"].astype(np.float32)
    elif fault=="K":t["camera_K"][0,0]=1279
    elif fault=="ids":t["visible_face_indices"].flat[0]=40000
    elif fault=="depth":t["scene_depth_m"].flat[0]=0
    elif fault=="missing":t.pop("human_joints_camera_m")
    elif fault=="identity":t["frame_index"]=np.array(1,np.int64)
    else:t["object_faces"][0,0]=194
    with pytest.raises(ValueError):gate.validate_truth(t,r,f)


def test_failed_public_gate_cannot_open_private(gate,tmp_path,monkeypatch):
    p,q,_=pins(gate,tmp_path);seen=[]
    monkeypatch.setattr(gate.public,"validate_pins",lambda *_:{})
    monkeypatch.setattr(gate,"public_observations",lambda *_:(_ for _ in()).throw(ValueError("public failed")))
    monkeypatch.setattr(gate,"private_truth",lambda *_:seen.append("forbidden"))
    report={}
    with pytest.raises(ValueError):gate.run(tmp_path,report,p,q)
    assert seen==[]and not report.get("private_truth_read",False)


def test_actual_private_failure_records_label_access(gate,tmp_path,monkeypatch):
    p,q,_=pins(gate,tmp_path)
    monkeypatch.setattr(gate.public,"validate_pins",lambda *_:{})
    monkeypatch.setattr(gate,"public_observations",lambda *_:([],{}, {}, {}, []))
    monkeypatch.setattr(gate,"quality_sources",lambda:[])
    monkeypatch.setattr(gate,"private_truth",lambda *_:(_ for _ in()).throw(ValueError("private file rejected")))
    report={}
    with pytest.raises(ValueError):gate.run(tmp_path,report,p,q)
    assert report["private_truth_read"]and report["predictions_frozen_before_private"]


def test_wrapper_canonical_sources_RO_cpu_no_args(gate):
    path=ROOT/"infra/run_factorial_rgb_evaluate.sh";text=path.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(path)],check=True)
    assert "SOURCE_LIST=\"$(python3"in text and "paths=set()"in text and "< <("not in text
    assert "src=$SOURCE,dst=$SOURCE,readonly"in text and "--gpus"not in text
    for fragment in ("$BASE/body_v1","$BASE/dwpose_v1","$BASE/eval_private","weights/dwpose_native_v1","weights/cari4d/sam3d_body"):
        assert fragment in text
    result=subprocess.run(["rtk","proxy","bash",str(path),"unexpected"],capture_output=True)
    assert result.returncode==2


def test_full24_once_geometry_independence_and_fixed_transform(gate,tmp_path,monkeypatch):
    p,q,_=pins(gate,tmp_path);template,faces,_=truth(gate,monkeypatch);records=[];cases=[];bodies=[];detectors=[];private=tmp_path/"own_private";private.mkdir()
    frozen=[]
    for i in range(24):
        g,f=divmod(i,3);r=dict(file=gate.COHORT.frame_name(i),sha256="a"*64,group_index=g,frame_index=f);records.append(r)
        t={k:v.copy()for k,v in template.items()};t["group_index"]=np.array(g,np.int64);t["frame_index"]=np.array(f,np.int64)
        path=private/(Path(r["file"]).stem+".npz")
        with path.open("xb")as stream:np.savez_compressed(stream,**t)
        path.chmod(0o400)
        cases.append(dict(file=r["file"],rgb_sha256=r["sha256"],group_index=g,frame_index=f,morphology_index=g//4,appearance_index=g//2%2,occlusion_index=g%2,truth_sha256=gate.sha256(path)))
        b=dict(vertices_camera_m=t["human_vertices_camera_m"].copy(),keypoints_camera_m=t["human_keypoints_camera_m"].copy(),joints_camera_m=t["human_joints_camera_m"].copy(),human_faces=faces)
        d=dict(keypoints=np.zeros((1,133,2)),validity=np.ones((1,133),bool));d["keypoints"][0,:17]=gate.project(b["keypoints_camera_m"][np.asarray(gate.COCO_MHR)],t["camera_K"])
        bodies.append(b);detectors.append(d)
    semantic=tmp_path/"results/mhr-finger-semantics-v4.json";sid=freeze(semantic,{"source":True});semantic.chmod(0o644)
    producers=dict(body=dict(body_model=dict(inference_source_identity={},body_assets={}),semantic_report_sha256=sid["sha256"]),dwpose=dict(assets={},capability_evidence={}))
    monkeypatch.setattr(gate.public,"validate_pins",lambda *_:{})
    monkeypatch.setattr(gate,"quality_sources",lambda:[])
    monkeypatch.setattr(gate,"public_observations",lambda *_:(records,{"body":bodies,"dwpose":detectors},producers,{},frozen))
    monkeypatch.setattr(gate,"private_truth",lambda *_:(private,cases,{}))
    monkeypatch.setattr(gate.public,"public_inputs",lambda *_:(records,{}))
    monkeypatch.setattr(gate.public.native.human.body,"_source_identity",lambda _: {})
    monkeypatch.setattr(gate.public.native.human.body,"_body_assets",lambda _:(None,{}))
    monkeypatch.setattr(gate.public.smoke,"validate_assets",lambda _: {})
    monkeypatch.setattr(gate.public.dw_helper,"validate_smoke",lambda _: {})
    calls=[];original=gate.alignment.fit_human_sim3
    monkeypatch.setattr(gate.alignment,"fit_human_sim3",lambda *args:(calls.append(1),original(*args))[1])
    report={};gate.run(tmp_path,report,p,q)
    assert report["status"]=="pass"and len(report["frame_metrics"])==24 and len(calls)==8
    assert report["private_optimizer_or_model_calls"]==0 and report["all_factors_geometry_independent_verified"]
    assert not report["accuracy_verified"]and not report["CARI4D_superiority_verified"]
    # Different appearance must never secretly change geometry.
    path=private/(Path(records[6]["file"]).stem+".npz");path.chmod(0o600)
    with np.load(path,allow_pickle=False)as saved:t={k:saved[k]for k in saved.files}
    t["human_keypoints_camera_m"][0,0]+=.01
    with path.open("wb")as stream:np.savez_compressed(stream,**t)
    path.chmod(0o400);cases[6]["truth_sha256"]=gate.sha256(path)
    with pytest.raises(ValueError,match="Appearance/occlusion"):gate.run(tmp_path,{},p,q)

"""Small exact diagnostic/file contracts; no local Torch/model/forward."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location("bounds_audit_test",INFRA/"keypoint_rgb_bounds_audit.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def metadata():
    return np.tile(np.array([-10.,10.],np.float32),(249,1)),[f"native_{i}"for i in range(249)]


def test_exact_all249_violation_and_root_mapping_diagnostics(gate):
    limits,names=metadata();limits[136]=0;limits[204]=[-1,1]
    controls=np.zeros(204,np.float32);shape=np.zeros(45,np.float32);euler=np.zeros(3,np.float32)
    controls[136]=.125;controls[8]=-12;shape[0]=1.25;controls[3]=.02
    before=controls.copy();row=gate.audit_controls(controls,shape,euler,limits,names)
    assert row["violation_count"]==3 and [v["index"]for v in row["violations"]]==[8,136,204]
    assert [v["excess"]for v in row["violations"]]==[2.,.125,.25]
    assert not row["root_Euler_byte_equal"] and row["locked_scale_columns"]==[136]
    assert np.array_equal(controls,before) and row["root_translation_zero"]


def test_infinite_bounds_json_safe_and_no_alignment(gate):
    limits,names=metadata();limits[0]=[-np.inf,np.inf]
    row=gate.audit_controls(np.zeros(204,np.float32),np.zeros(45,np.float32),np.zeros(3,np.float32),limits,names)
    assert row["within_native_bounds"] and row["neutral_zero_204_permitted_by_metadata"]
    json.dumps(row,allow_nan=False)
    assert gate.finite_bound(np.inf)=="+infinity" and gate.finite_bound(-np.inf)=="-infinity"


@pytest.mark.parametrize("fault",["controls_dtype","shape","NaN","masked","limitsNaN","unordered","impossibleinf","names","duplicate"])
def test_malformed_data_fails_not_violating_prediction(gate,fault):
    limits,names=metadata();c=np.zeros(204,np.float32);s=np.zeros(45,np.float32);e=np.zeros(3,np.float32)
    if fault=="controls_dtype":c=c.astype(np.float64)
    elif fault=="shape":s=s[:-1]
    elif fault=="NaN":c[1]=np.nan
    elif fault=="masked":c=np.ma.array(c,mask=False)
    elif fault=="limitsNaN":limits[1,0]=np.nan
    elif fault=="unordered":limits[1]=[2,1]
    elif fault=="impossibleinf":limits[0]=[np.inf,np.inf]
    elif fault=="names":names=names[:-1]
    else:names[1]=names[0]
    with pytest.raises(ValueError):gate.audit_controls(c,s,e,limits,names)


def failed(gate):
    return dict(stage="public_keypoint_rgb_native_root_refit",status="fail",phase="native_root_optimization",producer_revision=gate.FAILED_REV,
        script_sha256=gate.FAILED_SCRIPT,image_id=gate.IMAGE,network="none",device="cuda",error_type="ValueError",
        error="Complete249 native bounds/rootEuler mapping violated; no clipping",private_truth_read=False,ground_truth_used=False,
        challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],object_proxies_frozen_before_fit=True,all_candidates_frozen=False,
        native_arrays_verified=False,quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        counters=dict(proxy_rasters=15,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0),
        fit_records=[dict(file="clip_00_frame_000.png",evaluated_losses=[],native_forward_calls=0,adam_updates=0)],candidate_outputs=[],proxy_outputs=[{}]*15)


@pytest.mark.parametrize("fault",[None,"status","private","phase","head","updates","proxies","candidate"])
def test_exact_historical_failure_guard(gate,fault):
    row=failed(gate)
    if fault=="status":row["status"]="pass"
    elif fault=="private":row["private_truth_read"]=True
    elif fault=="phase":row["phase"]="fit_complete"
    elif fault=="head":row["counters"]["objective_native_heads"]=1
    elif fault=="updates":row["fit_records"][0]["adam_updates"]=1
    elif fault=="proxies":row["proxy_outputs"]=row["proxy_outputs"][:-1]
    elif fault=="candidate":row["candidate_outputs"]=[{}]
    if fault:
        with pytest.raises(ValueError):gate.validate_failed(row)
    else:gate.validate_failed(row)


def input_fixture(gate,tmp_path,monkeypatch):
    base=tmp_path/gate.BASE;bp=base/"baseline_v1";fp=base/"root_fit_v1"
    for path in (bp/"raw",bp/"paired",fp/"proxies",fp/"candidates",base/"inputs"):path.mkdir(parents=True)
    manifest=dict(schema="world-reward-keypoint-rgb-v1",images=[dict(file=f"clip_{i//5:02d}_frame_{i%5:03d}.png",sha256="a"*64,width=1024,height=768)for i in range(15)])
    mp=base/"inputs/manifest.json";mp.write_text(json.dumps(manifest));assert mp.stat().st_size==2199;mp.chmod(0o444)
    monkeypatch.setattr(gate,"MANIFEST_SHA",gate.regular(mp)["sha256"])
    b=dict(stage="public_keypoint_rgb_body_depth_first_rgb_identity_baseline",status="pass",phase="complete",producer_revision=gate.BASELINE_REV,
        script_sha256=gate.BASELINE_SCRIPT,frames=15,all_cases_retained=True,private_truth_read=False,ground_truth_used=False,
        challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],raw_frozen_before_shared=True,paired_frozen_before_reference=True,
        conversion_fidelity_verified=True,raw_outputs=[],paired_outputs=[])
    f=failed(gate);f["proxy_outputs"]=[]
    for i in range(15):
        clip,frame=divmod(i,5);name=f"clip_{clip:02d}_frame_{frame:03d}";controls=np.zeros(204,np.float32);shape=np.zeros(45,np.float32)
        raw=dict(mhr_model_params=controls,shape_params=shape,global_rot=np.zeros(3,np.float32),clip_index=np.array(clip,np.int64),frame_index=np.array(frame,np.int64))
        pair=dict(raw_model_controls=controls.copy(),shared_model_controls=controls.copy(),raw_shape_params=shape.copy(),shared_shape_params=shape.copy(),
            shared_scale_params=np.zeros(28,np.float32),clip_index=raw["clip_index"],frame_index=raw["frame_index"])
        for folder,key,data in (("raw","raw_outputs",raw),("paired","paired_outputs",pair)):
            path=bp/folder/(name+".npz")
            with path.open("xb")as stream:np.savez_compressed(stream,**data)
            path.chmod(0o444);b[key].append(dict(file=name+".png",clip_index=clip,frame_index=frame,rgb_sha256="a"*64,artifact=folder+"/"+name+".npz",**gate.regular(path)))
        path=fp/"proxies"/(name+".npz");path.write_bytes(b"own frozen proxy");path.chmod(0o444)
        f["proxy_outputs"].append(dict(artifact="proxies/"+name+".npz",rgb_sha256="a"*64,**gate.regular(path)))
    source=tmp_path/"jobs"/gate.FAILED_REV/"run_keypoint_rgb_fit/code/infra/keypoint_rgb_fit.py";source.parent.mkdir(parents=True);source.write_bytes(b"frozen source");source.chmod(0o444)
    monkeypatch.setattr(gate,"FAILED_SCRIPT",gate.regular(source)["sha256"]);f["script_sha256"]=gate.FAILED_SCRIPT
    def save():
        for path,row,key in ((bp/"report.json",b,"BASELINE_SHA"),(fp/"report.json",f,"FAILED_SHA")):
            if path.exists():path.chmod(0o644)
            path.write_text(json.dumps(row));path.chmod(0o444);monkeypatch.setattr(gate,key,gate.regular(path)["sha256"])
    save();return base,b,f,save


def test_15_real_npz_hashes_order_and_shapes_no_rgb_private_model(gate,tmp_path,monkeypatch):
    base,_,_,_=input_fixture(gate,tmp_path,monkeypatch)
    raws,pairs,files=gate.read_inputs(tmp_path)
    assert len(raws)==len(pairs)==15 and len(files)==49 and not(base/"eval_private").exists()


@pytest.mark.parametrize("fault",["rawSHA","extra","proxySHA","candidate","sharedclip","rootpose","private"])
def test_original_chain_changes_fatal_before_metadata(gate,tmp_path,monkeypatch,fault):
    base,b,f,save=input_fixture(gate,tmp_path,monkeypatch)
    if fault=="rawSHA":b["raw_outputs"][0]["sha256"]="b"*64
    elif fault=="extra":(base/"baseline_v1/raw/noise.npz").write_bytes(b"x")
    elif fault=="proxySHA":f["proxy_outputs"][0]["sha256"]="b"*64
    elif fault=="candidate":(base/"root_fit_v1/candidates/hidden.npz").write_bytes(b"x")
    elif fault=="private":f["private_truth_read"]=True
    else:
        path=base/"baseline_v1/paired/clip_00_frame_001.npz"
        with np.load(path,allow_pickle=False)as saved:data={k:saved[k]for k in saved.files}
        data["shared_shape_params"if fault=="sharedclip"else"shared_model_controls"][0]=.1
        path.chmod(0o644)
        with path.open("wb")as stream:np.savez_compressed(stream,**data)
        path.chmod(0o444);b["paired_outputs"][1].update(gate.regular(path))
    save()
    with pytest.raises(ValueError):gate.read_inputs(tmp_path)


def test_readonly_source_cpu_firewall_and_bash(gate):
    source=Path(gate.__file__).read_text();shell=INFRA/"run_keypoint_rgb_bounds_audit.sh"
    subprocess.run(["bash","-n",str(shell)],check=True)
    assert "keypoint_rgb_public"not in source and "keypoint_rgb_fit as"not in source
    assert "model.forward"not in source and ".run("not in source
    assert "--gpus"not in shell.read_text() and "--network none"in shell.read_text() and "--memory 8g"in shell.read_text()
    assert "eval_private"not in shell.read_text() and "63s"in shell.read_text()

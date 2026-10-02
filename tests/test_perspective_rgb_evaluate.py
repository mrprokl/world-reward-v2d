"""Tiny camera metrics and public/private firewall; no model/GPU truth data."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def module(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_perspective_evaluation", infra/"perspective_rgb_evaluate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def cameras(focals):
    return np.array([[[f,0.,512.],[0.,f,384.],[0.,0.,1.]] for f in focals], dtype=np.float64)


def truth():
    return dict(camera_K=cameras([800.,960.,1120.,1440.,1600.,1760.,900.,1000.,1800.]),
        gravity=np.tile([0.,-1.,0.], (9,1)).astype(np.float64), strong=np.array([True]*6+[False]*3))


def predictions():
    T = truth(); return [dict(K=T["camera_K"][i].copy() if i<6 else cameras([1280.])[0],
        accepted=i<6, reason="accepted" if i<6 else "weak", gravity=(0.,-1.,0.) if i<6 else None) for i in range(9)]


def test_all_nine_scored_including_abstention_and_no_adoption(module):
    result = module.camera_metrics(predictions(), truth())
    assert len(result["cases"]) == 9 and result["strong_accepted"] == 6 and result["weak_accepted"] == 0
    assert result["gravity_missing_count"] == 3 and result["strong_gravity_count"] == 6
    assert result["decision"]["camera_only_synthetic_hypothesis_supported"]
    assert result["decision"]["adoption_authorized"] is False and result["decision"]["full_HOI_verified"] is False
    assert all(r["mean_relative_focal_error"] > 0 for r in result["cases"][6:])


def test_fallback_error_is_retained_and_not_fitted_to_truth(module):
    predicted = predictions()
    for row in predicted: row.update(K=cameras([1280.])[0], accepted=False, gravity=None)
    result = module.camera_metrics(predicted, truth())
    assert result["all_nine_median_relative_gain"] == 0. and result["strong_accepted"] == 0
    assert not result["decision"]["camera_only_synthetic_hypothesis_supported"]
    assert result["strong_gravity_median_deg"] is None and result["gravity_missing_count"] == 9


def test_zero_baseline_denominator_fails_gain_not_divide(module):
    T=truth(); T["camera_K"] = cameras([1280.]*9)
    result=module.camera_metrics(predictions(), T)
    assert result["all_nine_median_relative_gain"] is None
    assert result["decision"]["gates"]["all_nine_focal_gain"] is False


@pytest.mark.parametrize("fault", ["keys", "count", "strongorder", "strongdtype", "Kdtype", "zero", "nan", "skew", "center", "gravitynorm",
    "acceptedtype", "missinggravity", "gravitynan", "fallbackfit", "fallbackgravity", "camera"])
def test_strict_geometry_and_no_hidden_weak_drop(module, fault):
    T=truth(); pred=predictions()
    if fault=="keys": T["depth"]=np.ones(1)
    elif fault=="count": pred.pop()
    elif fault=="strongorder": T["strong"][0],T["strong"][8]=False,True
    elif fault=="strongdtype": T["strong"]=T["strong"].astype(np.int8)
    elif fault=="Kdtype": T["camera_K"]=T["camera_K"].astype(np.float32)
    elif fault=="zero": T["camera_K"][0,0,0]=0
    elif fault=="nan": T["camera_K"][0,0,0]=np.nan
    elif fault=="skew": T["camera_K"][0,0,1]=1
    elif fault=="center": T["camera_K"][0,0,2]+=1
    elif fault=="gravitynorm": T["gravity"][0]*=2
    elif fault=="acceptedtype": pred[0]["accepted"]=1
    elif fault=="missinggravity": pred[0]["gravity"]=None
    elif fault=="gravitynan": pred[0]["gravity"]=(0.,np.nan,0.)
    elif fault=="fallbackfit": pred[8]["K"]=cameras([1800.])[0]
    elif fault=="fallbackgravity": pred[8]["gravity"]=(0.,-1.,0.)
    else: pred[0]["K"][0,1]=.1
    with pytest.raises(ValueError): module.camera_metrics(pred,T)


def test_one_weak_acceptance_and_bad_strong_gravity_rejects(module):
    p=predictions(); p[8].update(accepted=True,gravity=(0.,-1.,0.))
    p[0]["gravity"]=(1.,0.,0.)
    result=module.camera_metrics(p,truth())
    assert not result["decision"]["gates"]["weak_abstention"]
    assert not result["decision"]["gates"]["strong_gravity_worst"]


@pytest.fixture
def public(module, monkeypatch, tmp_path):
    inf=module.inference; base=tmp_path/inf.BASE; inputs=base/"inputs"; out=base/"predictions_v1"
    inputs.mkdir(parents=True);out.mkdir();(tmp_path/"results").mkdir()
    images=[]
    for index in range(9):
        name=f"case_{index:02d}.png"; path=inputs/name;Image.new("RGB",(1024,768),(31,45,82)).save(path)
        images.append(dict(file=name,sha256=module.sha256(path),width=1024,height=768))
    (inputs/"manifest.json").write_text(json.dumps(dict(schema=inf.SCHEMA,images=images)))
    records,inputs_receipt=inf.public_inputs(tmp_path)
    fields=dict(up_field=np.stack([np.zeros((320,416),np.float32),-np.ones((320,416),np.float32)]),
        latitude_field=np.zeros((1,320,416),np.float32),up_confidence=np.ones((320,416),np.float32),latitude_confidence=np.ones((320,416),np.float32))
    fit=SimpleNamespace(K=((426/1024*1280,0.,207.5),(0.,320/768*1280,159.5),(0.,0.,1.)),
        focal_px=1280.,accepted=False,reason="fixture_abstain",gravity=None)
    fit.to_dict=lambda:dict(accepted=False,reason=fit.reason,focal_px=1280.,gravity=None,K=fit.K)
    monkeypatch.setattr(inf,"fit_perspective_calibration",lambda *a,**k:fit)
    rows=[]
    for record in records:
        path=out/(Path(record["file"]).stem+".npz");np.savez_compressed(path,**fields);path.chmod(0o444)
        rows.append(dict(case_index=record["case_index"],file=record["file"],rgb_sha256=record["rgb_sha256"],
            fields_file=path.name,fields_sha256=module.sha256(path),fields_bytes=path.stat().st_size,
            fields_summary=inf.validate_fields(fields),calibration=fit.to_dict(),camera_K=inf.original_camera(fit).tolist()))
    acquisition=dict(stage="pinned_geocalib_frontend_assets_acquisition",status="pass",phase="complete",
        source_revision=inf.native.assets.SOURCE_REV,segnext_revision=inf.native.assets.SEGNEXT_REV,source_subset_verified=True,
        camera_solver_acquired=False,challenge_inputs_used=False,ground_truth_used=False,files={"fixture":"immutable"})
    ap=tmp_path/inf.native.assets.REPORT;ap.write_text(json.dumps(acquisition))
    binding=dict(receipt_sha256=module.sha256(ap),files=acquisition["files"])
    dp=tmp_path/inf.native.REPORT;dp.write_text("{}")
    monkeypatch.setattr(inf,"D82_SHA",module.sha256(dp))
    d82=dict(sha256=inf.D82_SHA);monkeypatch.setattr(inf,"validate_d82",lambda root,value:d82 if value==binding else (_ for _ in ()).throw(ValueError()))
    report=dict(stage=inf.STAGE,status="pass",phase="complete",network="none",ground_truth_used=False,private_truth_read=False,
        challenge_inputs_used=False,oracle_modes=[],accuracy_verified=False,adoption_authorized=False,actual_frontend_calls=9,seeds_reset_calls=9,
        frames=9,all_cases_retained=True,inputs_assets_rechecked=True,full_geocalib_package_imported=False,
        perspective_fields_imported=False,lm_optimizer_imported=False,original_camera_convention="edge_center_W/2_H/2",image_id=inf.native.IMAGE_ID,
        inputs=inputs_receipt,script_sha256=module.sha256(Path(inf.__file__)),native_helper_sha256=inf.NATIVE_SHA,
        solver_sha256=module.sha256(Path(inf.calibration.__file__)),preprocess={**inf.PREPROCESS,"output_width":416,"output_height":320,
            "resize":"bilinear_antialias_align_corners_false","crop":"center_left5"},producer_revision="b"*40,code_revision="b"*40,assets=binding,d82=d82,
        checkpoint_load=dict(state_load_strict=True,state_all_finite=True,parameter_keys=748,buffer_keys=141,state_key_mapping="none"),outputs=rows,accepted_cases=0)
    rp=out/"report.json"
    def save():rp.chmod(0o644) if rp.exists() else None;rp.write_text(json.dumps(report));rp.chmod(0o444)
    save()
    return tmp_path,report,save,out,fit


def test_complete_public_audit_replays_all_before_private(module,public):
    root,report,_,_,_=public
    predicted,records,_,frozen,_=module.public_predictions(root)
    assert len(predicted)==len(records)==9 and len(frozen)==12
    assert all(not p["accepted"] and p["gravity"] is None for p in predicted)


@pytest.mark.parametrize("fault",["calls","flags","checkpoint","solver","frameorder","K","fit","summary","archivehash","writable","extra","manifestcamera","manifestbool"])
def test_tamper_fails_before_any_private_read(module,public,monkeypatch,fault):
    root,report,save,out,_=public
    if fault=="calls":report["actual_frontend_calls"]=8
    elif fault=="flags":report["private_truth_read"]=True
    elif fault=="checkpoint":report["checkpoint_load"]["buffer_keys"]=140
    elif fault=="solver":report["solver_sha256"]="0"*64
    elif fault=="frameorder":report["outputs"][0]["case_index"]=1
    elif fault=="K":report["outputs"][0]["camera_K"][0][0]+=1
    elif fault=="fit":report["outputs"][0]["calibration"]["accepted"]=True
    elif fault=="summary":report["outputs"][0]["fields_summary"]["up_unit_max_error"]+=.1
    elif fault=="archivehash":report["outputs"][0]["fields_sha256"]="0"*64
    elif fault=="writable":(out/"case_00.npz").chmod(0o644)
    elif fault=="extra":(out/"camera_truth.json").write_text("{}")
    else:
        p=root/module.inference.BASE/"inputs/manifest.json";data=json.loads(p.read_text())
        if fault=="manifestcamera":data["images"][0]["K"]=[[1,0,0]]
        else:data["images"][0]["width"]=True
        p.write_text(json.dumps(data))
    save()
    original=Path.read_text;seen=[]
    def guarded(path,*a,**k):
        if "eval_private" in path.parts:seen.append(path);raise AssertionError("Private truth read before public validation")
        return original(path,*a,**k)
    monkeypatch.setattr(Path,"read_text",guarded)
    with pytest.raises(ValueError):module.run(root,{})
    assert seen==[]


def test_private_camera_values_are_never_solver_arguments(module,public,monkeypatch):
    root,_,_,_,_=public;calls=[]
    def fit(*args,**kwargs):
        calls.append(kwargs.copy());return public[-1]
    monkeypatch.setattr(module.inference,"fit_perspective_calibration",fit)
    report={}
    with pytest.raises(ValueError):module.run(root,report) # Private files intentionally absent.
    assert calls==[module.inference.PREPROCESS]*9
    assert report["predictions_frozen_before_private"] is True


def test_wrapper_offline_cpu_scoped_write_and_source(module):
    wrapper=Path(module.__file__).with_name("run_perspective_rgb_evaluate.sh")
    subprocess.run(["bash","-n",str(wrapper)],check=True)
    text=wrapper.read_text()
    assert "--gpus" not in text and "63s" in text and "--memory 8g" in text and "--network none" in text
    assert "src=$BASE/predictions_v1,dst=$BASE/predictions_v1,readonly" in text
    assert "src=$BASE/eval_private,dst=$BASE/eval_private,readonly" in text
    assert "src=$OUT,dst=$OUT" in text and "chown 1000:1000" in text
    assert "src=$ROOT/weights" not in text and "src=$ROOT/data" not in text


def test_unknown_arguments_fail_without_private_reads(module):
    with pytest.raises(SystemExit):module.main(["--use-gt-focal"])


def test_complete_private_camera_quality_after_all_public_freeze(module,public,monkeypatch):
    root,prediction_report,_,_,_=public
    base=root/module.inference.BASE;private=base/"eval_private";private.mkdir()
    tp=private/"calibration_truth.npz";np.savez_compressed(tp,**truth())
    semantic=dict(source_image_id=module.RENDER_IMAGE)
    semantic_path=root/"results/mhr-finger-semantics-v4.json";semantic_path.write_text(json.dumps(semantic))
    import hand_synthetic_render as hand
    import joint_rgb_render as joint
    monkeypatch.setattr(hand,"require_semantic_report",lambda report: None)
    manifest_hash=module.sha256(base/"inputs/manifest.json")
    renderer=dict(stage="own_procedural_perspective_rgb_render",status="pass",synthetic_truth_used_for_rendering_only=True,
        inference_performed=False,challenge_inputs_used=False,photorealism_verified=False,accuracy_verified=False,
        actual_MHR_reference_used=True,actual_reference_forward_calls=2,strong_cases=6,weak_cases=3,image_id=module.RENDER_IMAGE,
        public_manifest_sha256=manifest_hash,code_revision="c"*40,model_sha256=hand.semantics.MODEL_SHA,
        script_sha256=module.sha256(Path(module.__file__).with_name("perspective_rgb_render.py")),
        render_helper_sha256=module.sha256(Path(hand.__file__)),joint_helper_sha256=module.sha256(Path(joint.__file__)),
        camera_helper_sha256=module.sha256(Path(module.__file__).with_name("camera_render.py")),
        semantic_report_sha256=module.sha256(semantic_path),truth_sha256=module.sha256(tp),
        cases=[dict(file=r["file"],rgb_sha256=r["rgb_sha256"]) for r in prediction_report["outputs"]])
    (private/"render-report.json").write_text(json.dumps(renderer))
    report={};module.run(root,report)
    assert report["status"]=="pass" and report["phase"]=="complete"
    assert report["decision"]["camera_only_synthetic_hypothesis_supported"] is False # Fixtures abstain all9.
    assert report["private_synthetic_truth_used_for_evaluation"] is True
    assert report["renderer_image_id"]!=report["frontend_image_id"]
    assert report["renderer_and_frontend_images_asserted_identical"] is False


def test_runtime_source_closure_includes_renderer_audit_files(module):
    import azure_job
    root=Path(module.__file__).resolve().parents[1]
    files={str(p.relative_to(root)):p.read_bytes() for base in (root/"infra",root/"src",root/"configs") for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    closure=azure_job.runtime_bundle_paths(files,"infra/run_perspective_rgb_evaluate.sh")
    assert "infra/perspective_rgb_render.py" in closure and "infra/camera_render.py" in closure


def test_literal_inverse_resize_transport_ulp_preserved_not_repaired(module):
    f=206.7987987987988
    result=SimpleNamespace(K=((426/1024*f,0.,207.5),(0.,320/768*f,159.5),(0.,0.,1.)),focal_px=f)
    K=module.inference.original_camera(result)
    assert K[0,0]!=K[1,1]
    pred=predictions();pred[0]["K"]=K.copy();saved=K.tobytes()
    score=module.camera_metrics(pred,truth())
    assert score["cases"][0]["relative_fx_error"]==abs(K[0,0]/800.-1.)
    assert score["cases"][0]["relative_fy_error"]==abs(K[1,1]/800.-1.)
    assert pred[0]["K"].tobytes()==saved
    pred[0]["K"][1,1]+=1e-5
    with pytest.raises(ValueError):module.camera_metrics(pred,truth())

"""Own tiny numeric/public-native preparation contracts; no models or GPU."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("cari96_prepare_test",ROOT/"infra/cari96_prepare.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def original(gate):
    value={k:np.zeros((501,d),np.float32) for k,d in gate.inputs.PARAMETER_DIMS.items()}
    value["mhr_trans"][:,2]=2
    value["mhr_shape"][:,0]=np.arange(501)/100
    value["mhr_scale"][:,1]=np.arange(501)/1000
    value["mhr_body_pose_cont"][:,0]=np.arange(501)/500
    value.update(body_model="mhr",kids=[0],frames=[f"{i:06d}" for i in range(501)],
        mhr_joints=np.ones((501,127,3),np.float32),mhr_keypoints=np.ones((501,70,3),np.float32),
        metadata=dict(ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],mhr_geometry_forward_verified=True))
    return value


def test_shared_identity_before_decode_keeps_frame_motion_original_unchanged(gate):
    value=original(gate);before={k:v.tobytes()for k,v in value.items()if isinstance(v,np.ndarray)}
    selected=gate.shared_parameters(value)
    for k in gate.inputs.PARAMETER_DIMS:
        expected=np.repeat(value[k][:1],96,axis=0)if k in("mhr_shape","mhr_scale")else value[k][:96]
        assert selected[k].shape==(96,gate.inputs.PARAMETER_DIMS[k])and selected[k].tobytes()==expected.tobytes()
        assert not np.shares_memory(selected[k],value[k])
    assert {k:v.tobytes()for k,v in value.items()if isinstance(v,np.ndarray)}==before
    assert np.unique(selected["mhr_body_pose_cont"][:,0]).size==96


def test_historical_geometry_checks_not_new_shared_identity_claim(gate):
    metadata=original(gate)["metadata"]
    metadata.update(native_roundtrip_max_error_m={"vertices":0},translation_once_max_error_m=0,projection_max_error_px=.001)
    before=metadata.copy();result=gate.shared_metadata(metadata)
    assert metadata==before and result["historical_original_initializer_checks"]["projection_max_error_px"]==.001
    assert "projection_max_error_px" not in result and "native_roundtrip_max_error_m" not in result
    assert result["projection_reverified_after_identity_change"] is False
    assert result["human_identity_clip_constant"] is True and result["submission_eligible"] is False


def test_attempt_persisted_before_call_not_claimed_complete(gate):
    report={"reference_attempts":0,"reference_calls":0};seen=[]
    gate.attempted(report,"reference",lambda:seen.append(report.copy()))
    assert seen==[{"reference_attempts":1,"reference_calls":0}]


@pytest.mark.parametrize("fault",["missing","count","f64","nan","face","timeline"])
def test_original_native_contract_fails_before_any_decode(gate,fault):
    value=original(gate)
    if fault=="missing":value.pop("mhr_hand")
    elif fault=="count":value["mhr_scale"]=value["mhr_scale"][:96]
    elif fault=="f64":value["mhr_shape"]=value["mhr_shape"].astype(np.float64)
    elif fault=="nan":value["mhr_trans"][0,0]=np.nan
    elif fault=="face":value["mhr_face"][0,0]=.1
    else:value["frames"][0]="000001"
    with pytest.raises((ValueError,TypeError)):gate.shared_parameters(value)


def test_residuals_explicit_units_and_all_frames(gate):
    target=np.ones((3,7,3),np.float32);actual=target.astype(np.float64)+[.001,0,0]
    values=gate.residuals(target,actual,"m")
    assert values["per_frame_mean_mm"]==pytest.approx([1,1,1])
    assert gate.residuals(target,actual*1000,"mm")==values
    with pytest.raises(ValueError):gate.residuals(target,actual[:1],"m")
    with pytest.raises(ValueError):gate.residuals(target,actual,"cm")


@pytest.mark.parametrize("fault",["negative","nan","count","dtype"])
def test_native_geometry_no_hidden_drop(gate,fault):
    v=np.ones((2,7,3),np.float32)
    if fault=="negative":v[0,0,2]=0
    elif fault=="nan":v[0,0,1]=np.nan
    elif fault=="count":v=v[:1]
    else:v=v.astype(np.float64)
    with pytest.raises(ValueError):gate.checked_geometry(v,2,7)


def test_wrapper_offline_public_only_and_exact_exclusive_budget(gate):
    p=ROOT/"infra/run_cari96_prepare.sh";text=p.read_text();subprocess.run(["rtk","proxy","bash","-n",str(p)],check=True)
    assert subprocess.run(["rtk","proxy","bash",str(p),"--resume"],capture_output=True).returncode==2
    assert text.count("docker run")==1 and "183s"in text and "--gpus all --network none --memory 32g --cpus 4"in text
    assert "eval_private"not in text and "cari_forward"not in text and "cari_refined"not in text
    assert "src=$path,dst=$path,readonly"in text and "src=$OUT,dst=$OUT"in text


def test_no_top_level_torch_or_native_model_and_no_quality_fit(gate):
    tree=ast.parse(Path(gate.__file__).read_text())
    assert not any(isinstance(n,ast.Import)and any(a.name=="torch"for a in n.names)for n in tree.body)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="run")
    text=ast.unparse(run)
    assert "reference.run"in text and "prepare_snapshot"in text and "shared_parameters"in text
    assert "process_one_image"not in text and "joint_fit"not in text and "eval_private"not in text

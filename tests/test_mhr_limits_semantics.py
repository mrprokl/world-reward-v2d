"""Tiny getter/state contracts; fake scripted APIs do not prove native semantics."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location("limits_semantics_test",INFRA/"mhr_limits_semantics.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Tensor:
    def __init__(self,data):self.data=data
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.data


def model():
    limits=np.tile(np.array([-10.,10.],np.float32),(249,1));limits[20]=0
    method=SimpleNamespace(code="def get_parameter_limits(self):\n  return self.parameter_limits\n",graph="gettergraph")
    scripted=SimpleNamespace(_method_names=lambda:["forward","get_parameter_limits","get_parameter_names"],_get_method=lambda name:method)
    return SimpleNamespace(_c=scripted,get_parameter_limits=lambda:Tensor(limits),get_parameter_names=lambda:[f"p{i}"for i in range(249)],
        named_buffers=lambda:iter([("character.parameter_limits",Tensor(limits)),("mesh.vertices",Tensor(np.zeros((3000,3),np.float32)))]),
        state_dict=lambda:{"limits.minmax":Tensor(np.arange(2050,dtype=np.float32)),"other":Tensor(np.ones(3))},
        named_modules=lambda:iter([("limits",SimpleNamespace(_c=scripted)),("mesh",SimpleNamespace())]))


def test_actual_source_required_and_selected_state_only(gate):
    evidence=gate.snapshot(model())
    assert evidence["getter_code"].startswith("def get_parameter_limits") and evidence["zero_zero_columns"]==[20]
    assert set(evidence["selected_named_buffers"])=={"character.parameter_limits"}
    assert "values"in evidence["dense_parameter_limits"] and "values"not in evidence["selected_state_dict"]["limits.minmax"]
    assert evidence["selected_named_modules"][0]["path"]=="limits"
    json.dumps(evidence,allow_nan=False)


@pytest.mark.parametrize("fault",["shape","names","NaN","unordered","masked","missinggetter","emptycode","hugecode"])
def test_metadata_or_source_missing_fails_not_semantics_guess(gate,fault):
    m=model()
    if fault=="shape":m.get_parameter_limits=lambda:Tensor(np.ones((248,2),np.float32))
    elif fault=="names":m.get_parameter_names=lambda:["same"]*249
    elif fault=="NaN":m.get_parameter_limits=lambda:Tensor(np.full((249,2),np.nan,np.float32))
    elif fault=="unordered":m.get_parameter_limits=lambda:Tensor(np.tile(np.array([1.,-1.]),(249,1)))
    elif fault=="masked":m.get_parameter_limits=lambda:Tensor(np.ma.array(np.ones((249,2)),mask=False))
    elif fault=="missinggetter":m._c._method_names=lambda:["forward"]
    elif fault=="emptycode":m._c._get_method=lambda name:SimpleNamespace(code="",graph="graph")
    else:m._c._get_method=lambda name:SimpleNamespace(code="a"*100001,graph="graph")
    with pytest.raises(ValueError):gate.snapshot(m)


def test_nonfinite_values_json_and_bounded_optional_graph(gate):
    row=gate.array_metadata(np.array([-np.inf,np.inf,np.nan],np.float32));assert row["values"]==["-infinity","+infinity","NaN"]
    json.dumps(row,allow_nan=False)
    m=model();m._c._get_method=lambda name:SimpleNamespace(code="actualgetter",graph="g"*100000)
    assert"getter_graph"not in gate.getter_source(m)


@pytest.mark.parametrize("name,selected",[("parameter_limits",True),("upper_minmax",True),("MaxLimit",True),("mesh",False),(None,False)])
def test_exact_metadata_filter(gate,name,selected):assert gate.selected_name(name)is selected


@pytest.mark.parametrize("fault",[None,"rw","parent","double","escaped","bad"])
def test_model_exact_readonly_mount(gate,fault):
    target="/srv/model.pt";line="31 22 0:8 /src /srv/model.pt ro,relatime - ext4 /dev/root rw"
    if fault=="rw":line=line.replace("ro,relatime","rw,relatime")
    elif fault=="parent":line=line.replace(target,"/srv")
    elif fault=="double":line+="\n"+line
    elif fault=="escaped":target="/srv/model file.pt";line=line.replace("/srv/model.pt",r"/srv/model\040file.pt")
    elif fault=="bad":line="bad mount"
    if fault not in(None,"escaped"):
        with pytest.raises(ValueError):gate.readonly_model_mount(line,target)
    else:assert gate.readonly_model_mount(line,target)["mount_point"]==target


def test_previous_exact_pass_and_zero_forward_firewall(gate,tmp_path,monkeypatch):
    path=tmp_path/gate.PREVIOUS;path.parent.mkdir(parents=True)
    row=dict(stage="public_keypoint_rgb_native_bounds_readonly_audit_v2",status="pass",phase="complete",producer_revision=gate.PREVIOUS_REV,
        script_sha256=gate.PREVIOUS_SCRIPT,image_id=gate.IMAGE,network="none",device="cpu",model_sha256=gate.MODEL_SHA,model_forward_calls=0,
        optimizer_updates=0,frames=15,all_cases_retained=True,original_inputs_rehashed=True,raw_root_mapping_mismatch_frames=0,shared_root_mapping_mismatch_frames=0,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],bounds_relaxed=False,
        predictions_modified=False,original_failure_rewritten=False,adoption_authorized=False,
        records=[dict(clip_index=i//5,frame_index=i%5,raw=dict(root_Euler_byte_equal=True),shared=dict(root_Euler_byte_equal=True))for i in range(15)])
    def save():
        if path.exists():path.chmod(0o644)
        path.write_text(json.dumps(row));path.chmod(0o444);monkeypatch.setattr(gate,"PREVIOUS_SHA",gate.regular(path)["sha256"])
    save();assert gate.previous(tmp_path)["path"]==str(path)
    for key in("model_forward_calls","private_truth_read","raw_root_mapping_mismatch_frames"):
        old=row[key];row[key]=True if key=="private_truth_read"else 1;save()
        with pytest.raises(ValueError):gate.previous(tmp_path)
        row[key]=old


def test_shell_tiny_offline_cpu_no_predictions_or_quality_mount(gate):
    shell=INFRA/"run_mhr_limits_semantics.sh";subprocess.run(["bash","-n",str(shell)],check=True)
    source=Path(gate.__file__).read_text();text=shell.read_text()
    assert"--gpus"not in text and"--network none"in text and"--memory 8g"in text and"63s"in text
    assert"eval_private"not in text and"baseline_v1"not in text and"keypoint_rgb_public"not in source
    assert"model.forward("not in source and".run("not in source

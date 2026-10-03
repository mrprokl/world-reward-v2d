"""Own small synthetic native-contract/selection/firewall tests, no learned model."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("photometric_capability_test",ROOT/"infra/photometric_capability.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def proposal(gate,offset=0.):
    data={key:np.zeros(shape,np.float32)for key,shape in gate.native.BLOCKS.items()}
    data["pred_cam_t"][:]=[0.,0.,3.]
    data.update(vertices_camera_m=np.tile([offset,0.,3.],(18439,1)).astype(np.float32),
        keypoints_camera_m=np.tile([0.,0.,3.],(308,1)).astype(np.float32),joints_camera_m=np.tile([0.,0.,3.],(127,1)).astype(np.float32),
        joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)))
    return data


@pytest.mark.parametrize("fault",["none","key","dtype","kp","root","camera","face","rotation","Euler","hand","shape","scale","fixedcontrols"])
def test_full_native_representation_and_anchored_fields(gate,fault):
    anchor=proposal(gate);candidate={k:v.copy()for k,v in anchor.items()}
    if fault=="none":assert gate.validate_proposal(candidate,anchor)is candidate;return
    if fault=="key":candidate["private_truth"]=np.zeros(1)
    elif fault=="dtype":candidate["shape_params"]=candidate["shape_params"].astype(np.float64)
    elif fault=="kp":candidate["keypoints_camera_m"]=candidate["keypoints_camera_m"][:70]
    elif fault=="root":candidate["mhr_model_params"][0]=1
    elif fault=="camera":candidate["pred_cam_t"][2]=0
    elif fault=="face":candidate["expr_params"][0]=1
    elif fault=="rotation":candidate["joint_global_rotations"][0,0,0]=-1
    elif fault=="Euler":candidate["mhr_model_params"][3]=1
    elif fault=="hand":candidate["hand_pose_params"][0]=1
    elif fault=="shape":candidate["shape_params"][0]=1
    elif fault=="scale":candidate["scale_params"][0]=1
    else:candidate["mhr_model_params"][136]=1
    with pytest.raises(ValueError):gate.validate_proposal(candidate,anchor)


def test_body_proposal_is_not_asserted_arm_only(gate):
    anchor=proposal(gate);candidate={k:v.copy()for k,v in anchor.items()}
    candidate["body_pose_params"][124]=.2;candidate["mhr_model_params"][130]=.2
    gate.validate_proposal(candidate,anchor)
    assert np.array_equal(gate.FIXED_CONTROLS,np.r_[0:6,68:122,136:204])
    assert len(gate.FIXED_CONTROLS)==128


def test_fixed_decoder_copies_anchor_except_native_body133(monkeypatch,gate):
    anchor=proposal(gate);raw=proposal(gate)
    raw["global_rot"][:]=1;raw["pred_cam_t"][:]=4;raw["shape_params"][:]=2;raw["hand_pose_params"][:]=2;raw["scale_params"][:]=3
    raw["body_pose_params"][4]=.2;seen=[]
    def shared(torch,head,blocks,first):
        seen.append((blocks,first))
        controls=anchor["mhr_model_params"].copy();controls[10]=.2
        return anchor["vertices_camera_m"],anchor["keypoints_camera_m"],anchor["joints_camera_m"],controls,anchor["joint_global_rotations"]
    monkeypatch.setattr(gate.baseline,"shared_decode",shared)
    out=gate.fixed_decode(None,None,raw,anchor)
    for key in gate.FIXED_BLOCKS:assert out[key].tobytes()==anchor[key].tobytes()
    assert out["body_pose_params"].tobytes()==raw["body_pose_params"].tobytes()and seen[0][1]is anchor
    assert out["mhr_model_params"][10]==pytest.approx(.2)
    assert np.all(raw["global_rot"]==1)and not np.any(anchor["global_rot"])


def test_replay_exact_sham_vs_bounded_native_roundtrip(gate):
    a=proposal(gate);b=proposal(gate)
    assert gate.parity(a,b,exact=True)["all_native_arrays_byte_equal"]
    b["vertices_camera_m"][0,0]=np.float32(1e-6)
    assert not gate.parity(a,b)["all_native_arrays_byte_equal"]
    with pytest.raises(ValueError,match="SHAM"):gate.parity(a,b,exact=True)
    b["vertices_camera_m"][0,0]=1e-3
    with pytest.raises(ValueError):gate.parity(a,b)


def test_complete15_actual_npz_roundtrip_and_frozen_inventory(gate,tmp_path):
    out=tmp_path/"capability";out.mkdir();(out/"report.json").write_text("{}");anchor=proposal(gate);rows=[]
    for prefix in("raw","fixed"):
        for name,_ in gate.ALL_BRANCHES:rows.append(gate.save_proposal(out,prefix+"_"+name,anchor))
    for mode in("baseline","sham","tta"):rows.append(gate.save_proposal(out,"selected_"+mode,anchor))
    values=gate.reread_proposals(out,rows,anchor)
    assert len(values)==15 and all(gate.parity(v,anchor,exact=True)["all_native_arrays_byte_equal"]for v in values)
    with pytest.raises(ValueError):gate.reread_proposals(out,rows[:-1],anchor)
    (out/"extra.txt").write_text("noise")
    with pytest.raises(ValueError):gate.reread_proposals(out,rows,anchor)


def test_bad_public_mask_fails_before_import_or_model(gate,tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(gate,"public_mask",lambda *_:(_ for _ in()).throw(ValueError("public rejected")))
    monkeypatch.setattr(gate.native.human,"load_model",lambda *_:calls.append("forbidden"))
    with pytest.raises(ValueError):gate.run_body(tmp_path,tmp_path,{},lambda:None,"a"*40)
    assert calls==[]


def test_source_recipe_has_no_fit_private_or_average(gate):
    tree=ast.parse(Path(gate.__file__).read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert "geometric_medoid"in calls and "baseline.shared_decode"in calls
    assert not any(name in calls for name in("torch.optim.Adam","scipy.optimize.minimize","manufacture.main","native.run_inference"))
    text=Path(gate.__file__).read_text()
    assert "eval_private"not in text and "factorial_rgb_v1"not in text and "SHAM_exact_replay_verified"in text
    assert "native_body_block_scope"in text and 'hard249_bounds_gate_applied=False'in text


def test_medoid_chosen_native_arrays_not_average(gate):
    variants=[proposal(gate,delta)for delta in(0.,.01,.02)]
    choice=gate.geometric_medoid(np.stack([v["vertices_camera_m"]for v in variants]))
    assert choice.index==1
    assert choice.chosen_vertices.tobytes()==variants[1]["vertices_camera_m"].tobytes()
    assert not choice.chosen_vertices.flags.writeable


def test_wrapper_isolates_all_three_stages_no_private_route(gate):
    path=ROOT/"infra/run_photometric_capability.sh";text=path.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(path)],check=True)
    assert "(( $# == 0 ))"in text and "eval_private"not in text
    assert text.count('docker run')==3 and text.count('src=$BASE,dst=$BASE"')==1
    assert '--stage masks'in text and '--stage body'in text
    assert '--mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly"'in text
    assert '--mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly"'in text
    result=subprocess.run(["rtk","proxy","bash",str(path),"unexpected"],capture_output=True)
    assert result.returncode==2


def test_cli_requires_explicit_stage(gate):
    with pytest.raises(SystemExit)as caught:gate.main([])
    assert caught.value.code==2

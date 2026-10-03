"""Data-free root5 API/gradient guards, not actual native CUDA verification."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA));monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec=importlib.util.spec_from_file_location("root5_test",INFRA/"root5_native_capability.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture():
    r=np.array([.1,-.2,.3],np.float32);t=np.array([.2,-.3,4.],np.float32);c=np.zeros(204,np.float32);c[3:6]=r
    return c,r,t,dict(pred_cam_t=t),dict(shared_model_controls=c.copy()),np.tile(np.array([-1.,1.]),(3,1))


def test_root_only_contract_retains_dense_remainder_without_clamping(gate):
    c,r,t,raw,pair,bounds=fixture();c[15]=.4;pair["shared_model_controls"][15]=.4
    gate.root_contract(c,r,t,raw,pair,bounds)
    assert c[15]==np.float32(.4)


@pytest.mark.parametrize("fault",["root","rootzero","rest","Z","bounds","metadata","masked"])
def test_only_XY_Euler_and_actual_root_bounds(gate,fault):
    c,r,t,raw,pair,bounds=fixture();t=t.copy()
    if fault=="root":c[3]+=.1
    elif fault=="rootzero":c[0]=1
    elif fault=="rest":c[68]=.1
    elif fault=="Z":t[2]+=.1
    elif fault=="bounds":bounds[:]=[-.01,.01]
    elif fault=="metadata":bounds[0,0]=np.nan
    else:c=np.ma.array(c,mask=False)
    with pytest.raises(ValueError):gate.root_contract(c,r,t,raw,pair,bounds)


def test_five_column_unregularized_jacobian_rank(gate):
    j=np.tile(np.eye(5),(4,1));evidence=gate.jacobian_evidence(j,10)
    assert evidence["columns"]==5 and evidence["minimum_maximum_ratio"]==1 and not evidence["prior_rows_included"]
    j[:,-1]=0
    assert gate.jacobian_evidence(j,10)["minimum_maximum_ratio"]==0
    with pytest.raises(ValueError):gate.jacobian_evidence(np.r_[j,np.eye(5)],10)


def test_native_XY_observation_jacobian_matches_analytic_current_projection(gate):
    points=np.column_stack((np.linspace(-.4,.4,10),np.linspace(-.3,.3,10),np.linspace(2.,5.,10))).astype(np.float32)
    K=np.array([[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]],np.float32)
    j=np.zeros((20,5),np.float32)
    j[::2,0]=np.float32(.30)*K[0,0]/points[:,2];j[1::2,1]=np.float32(.30)*K[1,1]/points[:,2]
    assert gate.analytic_xy_evidence(j,points,K)["additional_native_calls"]==0
    j[0,1]=1e-3
    with pytest.raises(ValueError,match="analytic"):gate.analytic_xy_evidence(j,points,K)
    with pytest.raises(ValueError):gate.analytic_xy_evidence(j,np.ma.array(points,mask=False),K)


def test_native_counter_attempt_precedes_call_failure(gate,monkeypatch):
    events=[];report=dict(attempted_native_heads=0,returned_native_heads=0)
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def __getitem__(self,key):return Tensor(self.value[key])
        def __mul__(self,other):return Tensor(self.value*(other.value if isinstance(other,Tensor)else other))
        def __add__(self,other):return Tensor(self.value+(other.value if isinstance(other,Tensor)else other))
    torch=SimpleNamespace(float32=np.float32,tensor=lambda value,**kwargs:Tensor(value),zeros=lambda n,**kwargs:Tensor(np.zeros(n)),
        tanh=lambda value:Tensor(np.tanh(value.value)),stack=lambda values:Tensor(np.stack([v.value for v in values])),zeros_like=lambda value:Tensor(np.zeros_like(value.value)))
    def forward(**kwargs):
        events.append((report["attempted_native_heads"],report["returned_native_heads"]))
        raise RuntimeError("native failure")
    head=SimpleNamespace(mhr_forward=forward,enable_hand_model=False)
    raw={k:np.zeros(s,np.float32)for k,s in gate.native.BLOCKS.items()};raw["pred_cam_t"][2]=4
    pair=dict(shared_shape_params=np.zeros(45,np.float32),shared_scale_params=np.zeros(28,np.float32))
    monkeypatch.setattr(gate,"strict",lambda torch:None)
    with pytest.raises(RuntimeError,match="native failure"):gate.native_forward(torch,head,raw,pair,report,lambda:None)
    assert events==[(1,0)]and report==dict(attempted_native_heads=1,returned_native_heads=0)
    head.enable_hand_model=True
    with pytest.raises(ValueError,match="enable_hand_model=False"):gate.native_forward(torch,head,raw,pair,report,lambda:None)
    assert events==[(1,0)]


def test_exact_shared_geometry_parity_and_SO3(gate,monkeypatch):
    monkeypatch.setattr(gate.native,"VERTICES",100)
    pair=dict(shared_vertices_camera_m=np.ones((100,3),np.float32),shared_keypoints_camera_m=np.ones((308,3),np.float32),
        shared_joints_camera_m=np.ones((127,3),np.float32),shared_model_controls=np.zeros(204,np.float32),
        shared_joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)))
    values=[pair[k].copy()for k in("shared_vertices_camera_m","shared_keypoints_camera_m","shared_joints_camera_m","shared_model_controls","shared_joint_global_rotations")]
    assert all(v==0 for v in gate.parity(values,pair).values())
    values[1][0,0]+=.001
    with pytest.raises(ValueError):gate.parity(values,pair)


def test_no_optimizer_or_private_and_strict_settings_deferred(gate):
    source=Path(gate.__file__).read_text();shell=INFRA/"run_root5_native_capability.sh"
    subprocess.run(["bash","-n",str(shell)],check=True)
    assert"Adam("not in source and".backward("not in source and"inference_mode()"not in source
    assert"torch.autograd.grad"in source and"use_deterministic_algorithms(True"in source
    assert"--gpus all"in shell.read_text()and"--network none"in shell.read_text()and"123s"in shell.read_text()
    assert"eval_private"not in shell.read_text()and"root_fit_v1"not in shell.read_text()and"translation_quality"not in shell.read_text()

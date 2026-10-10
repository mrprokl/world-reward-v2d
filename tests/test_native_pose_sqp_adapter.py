"""Tiny ABI/chain rule mocks only; not native GPU/Jacobian validation."""
from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import native_pose_sqp_adapter as adapter
from world_reward.native_pose_sqp import retract_native, STATE_DIM
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS


def parameters(n=3):
    rng=np.random.default_rng(104)
    p={k:np.zeros((n,d),np.float32) for k,d in NATIVE_PARAMETER_DIMS.items()}
    p['mhr_global_rot6d'][:]=[1,0,0,1,0,0]
    r=Rotation.random(n*23,random_state=rng).as_matrix().reshape(n,23,3,3)
    p['mhr_body_pose_cont'][:,:138]=np.concatenate((r[..., :,0],r[..., :,1]),axis=-1).reshape(n,138)
    theta=rng.uniform(-3,3,(n,58))
    p['mhr_body_pose_cont'][:,138:254]=np.stack((np.sin(theta),np.cos(theta)),axis=-1).reshape(n,116)
    p['mhr_trans'][:,2]=2.
    return p


@pytest.mark.parametrize('axes',[(0,1,2),(66,67,68),(69,70,126)])
def test_native_body254_gradient_right_SO3_SO2_chain_finite_difference(axes):
    p=parameters(96);rng=np.random.default_rng(118);g=rng.normal(size=(96,254)).astype(float)
    actual=adapter.body_gradient_to_tangent(p['mhr_body_pose_cont'],g)
    for axis in axes:
        step=np.zeros((96,STATE_DIM));step[:,axis]=1e-3
        plus,_=retract_native(p,p['mhr_trans'].astype(float),step)
        minus,_=retract_native(p,p['mhr_trans'].astype(float),-step)
        finite=((plus['mhr_body_pose_cont'][:,:254].astype(float)-minus['mhr_body_pose_cont'][:,:254].astype(float))*g).sum(-1)/2e-3
        np.testing.assert_allclose(actual[:,axis],finite,rtol=0,atol=3e-4)


def test_vector_Jacobian_general_prefix_and_zero_gradient_not_compact136():
    p=parameters();g=np.zeros((2,3,254),float)
    assert adapter.body_gradient_to_tangent(p['mhr_body_pose_cont'],g).shape==(2,3,127)
    for bad in [g.astype(np.float32),np.zeros((3,136),float),np.full((3,254),np.nan)]:
        with pytest.raises(ValueError):adapter.body_gradient_to_tangent(p['mhr_body_pose_cont'],bad)


class TensorMock:
    def __init__(self,a):self.a=a
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


def callback():
    p=parameters();e=SimpleNamespace(activations=np.ones((3,2),bool),frame_index=np.arange(3))
    instance=SimpleNamespace(contact_mask=TensorMock(e.activations),frame_indices=e.frame_index,
        params_fixed={k:TensorMock(v) for k,v in p.items()})
    binding=dict(layer=adapter.LAYER_PIN,optimizer=adapter.OPTIMIZER_PIN)
    c=adapter.NativeSQPCallbacks(None,None,instance,None,e,deadline=time.monotonic()+30,source_binding=binding)
    return c,p,e,binding,instance


def test_fixed_raw_native_priors_must_not_be_rebound_to_candidate_translation():
    c,p,_,_,instance=callback();p=deepcopy(p);p['mhr_trans'][:,0]=.01;p['mhr_body_pose_cont'][:,138]=.04
    c.validate_fixed(p)
    assert np.all(instance.params_fixed['mhr_trans'].numpy()[:,0]==0)
    p['mhr_hand'][0,0]=.01
    with pytest.raises(ValueError):c.validate_fixed(p)
    p=parameters();p['mhr_body_pose_cont'][0,259]=.01
    with pytest.raises(ValueError):c.validate_fixed(p)


def test_source_activation_frame_chunk_and_deadline_are_failclosed():
    c,p,e,binding,instance=callback()
    for kwargs in [dict(chunk=8),dict(source_binding=dict(layer={},optimizer=adapter.OPTIMIZER_PIN))]:
        args=dict(deadline=time.monotonic()+30,source_binding=binding);args.update(kwargs)
        with pytest.raises(ValueError):adapter.NativeSQPCallbacks(None,None,instance,None,e,**args)
    instance.contact_mask=TensorMock(~e.activations)
    with pytest.raises(ValueError):adapter.NativeSQPCallbacks(None,None,instance,None,e,deadline=time.monotonic()+30,source_binding=binding)
    c.deadline=time.monotonic()-1
    with pytest.raises(adapter._BudgetExhausted):c.check()
    assert adapter.OBJECTIVE_STAGE==181


def test_completion_receipts_survive_a_later_callback_failure():
    c,*_=callback();sealed=[];c.on_completion=sealed.append
    c.completed(dict(kind='full_native_objective',seconds=1.))
    c.deadline=time.monotonic()-1
    with pytest.raises(adapter._BudgetExhausted):c.check()
    assert sealed==c.calls and len(sealed)==1


def test_nonunit_encoding_gauge_derivative_matches_same_native_raw_prior():
    p=parameters(96);body=p['mhr_body_pose_cont'].reshape(96,260)
    raw=body[:,:138].reshape(96,23,6);first=raw[...,:3].copy();second=raw[...,3:].copy()
    raw[...,:3]=first*1.7;raw[...,3:]=first*.4+second*.8
    body[:,138:254]*=2.3
    rng=np.random.default_rng(1);g=rng.normal(size=(96,254)).astype(float)
    actual=adapter.body_gradient_to_tangent(body,g)
    for axis in [0,1,2,67,69,126]:
        d=np.zeros((96,STATE_DIM));d[:,axis]=1e-3
        plus,_=retract_native(p,p['mhr_trans'].astype(float),d)
        minus,_=retract_native(p,p['mhr_trans'].astype(float),-d)
        finite=((plus['mhr_body_pose_cont'][:,:254].astype(float)-minus['mhr_body_pose_cont'][:,:254].astype(float))*g).sum(-1)/2e-3
        np.testing.assert_allclose(finite,actual[:,axis],atol=6e-4,rtol=0)

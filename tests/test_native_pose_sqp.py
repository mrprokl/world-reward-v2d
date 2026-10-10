"""Independent authored two-chain DEV, not RGB inference/HOI validation."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import native_pose_sqp as sqp
from world_reward.contact_feasible_placement import ContactPlacementConfig, freeze_contact_witnesses
from world_reward.native_contact_continuation import ObservationGate
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS


def authored(seed=60261010):
    n=96;rng=np.random.default_rng(seed)
    p={k:np.zeros((n,d),np.float32) for k,d in NATIVE_PARAMETER_DIMS.items()}
    p['mhr_global_rot6d'][:]=[1,0,0,1,0,0]
    p['mhr_body_pose_cont'][:,:138]=np.tile([1,0,0,0,1,0],23)
    p['mhr_body_pose_cont'][:,138:254]=np.tile([0,1],58)
    p['mhr_trans'][:,2]=2.;p['mhr_trans'][:,1]=.05*np.sin(np.arange(n)/13)
    obj=p['mhr_trans'].astype(float).copy()
    vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)],float)
    faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int64)
    active=np.ones((n,2),bool);active[30:37]=False
    ids=np.tile([0,1],(n,1));ids[~active]=-1
    target=np.broadcast_to([.16,.16],(n,2)).copy()+rng.uniform(-.005,.005,(1,2))
    def angles(params): return np.arctan2(params['mhr_body_pose_cont'][:,[138,140]],params['mhr_body_pose_cont'][:,[139,141]]).astype(float)
    def decode(params):
        theta=angles(params);v=np.broadcast_to(params['mhr_trans'][:,None],(n,18439,3)).copy()
        v[:,:,2]+=.1
        v[:,0,0]+=(-.52+.3*np.sin(theta[:,0])).astype(np.float32)
        v[:,1,0]+=(.52+.5*np.sin(theta[:,1])).astype(np.float32)
        v[:,:2,2]=params['mhr_trans'][:,2,None]
        return dict(human_vertices=v,human_joints=v[:,:127].copy(),human_keypoints=v[:,:70].copy(),
            human_faces=np.array([[0,1,2]],np.int64),frame_index=np.arange(n,dtype=np.int64),
            pose=np.zeros((n,136),np.float32),scales=np.zeros(68,np.float32))
    g=decode(p);w=g['human_vertices'][:,:2].astype(float).copy();w[~active]=np.nan
    evidence=freeze_contact_witnesses(w,np.array([[0],[1]],np.int64),ids,np.tile(np.eye(3),(n,1,1)),
        obj,vertices,faces,active,np.arange(n,dtype=np.int64),
        config=ContactPlacementConfig(.05,1e-7,1e-10,400,8,'independent_authored_chain_DEV'),
        source_reference='authored_two_chain_not_native_MHR_or_real_reference',witness_selection_reference='authored_identity')
    def objective(params,ot,geometry): return float(.5*np.square(angles(params)-target).sum())
    def linearize(params,ot,geometry,e):
        theta=angles(params);gradient=np.zeros((n,sqp.STATE_DIM));gradient[:,69:71]=theta-target
        jac=np.zeros((n,2,3,sqp.ROTATION_DIM));jac[:,0,0,69]=.3*np.cos(theta[:,0]);jac[:,1,0,70]=.5*np.cos(theta[:,1])
        return dict(gradient=gradient,metric_diagonal=np.ones_like(gradient),witness_jacobian_camera=jac,
            frame_index=e.frame_index,activations=e.activations,witness_source_indices=e.witness_source_indices)
    def observe(geometry,ot):
        # Known independently authored trajectory component; imposing a cap on
        # total centroid path would reject genuinely added articulation motion.
        # The real runner's full motion/RGB gates must still be assessed later.
        return dict(known_root_y_increment_error=float(np.abs(np.diff(geometry['human_keypoints'][:,2,1])-
                np.diff(p['mhr_trans'][:,1])).max()),
            constant_reference=1.)
    gates=(ObservationGate('known_root_y_increment_error',1.,1e-7),ObservationGate('constant_reference',1.,0.))
    return p,obj,evidence,decode,linearize,objective,observe,gates,angles


def test_sparse_diagonal_QP_projects_two_incompatible_translational_contact_rows():
    grad=np.zeros((1,sqp.STATE_DIM));grad[0,:2]=[-1,-1]
    rows=np.zeros((2,sqp.STATE_DIM));rows[0,:2]=[1,0];rows[1,:2]=[0,1]
    step,receipt=sqp.project_diagonal_qp(grad,np.ones_like(grad),[rows],[np.zeros(2)],np.ones_like(grad))
    np.testing.assert_allclose(step,0,atol=1e-10)
    assert receipt['contact_rows']==2 and not receipt['dense_full_clip_matrix_built']


@pytest.mark.parametrize('seed',[60261010,60261011])
def test_independent_authored_DEV_bimanual_fixed_pose_translation_infeasible_articulation_succeeds(seed):
    p,obj,e,decode,linearize,objective,observe,gates,angles=authored(seed)
    # Target B alone is wider than mesh width +both fixed gap radii: no common
    # translation can put those two witnesses within both exact proximity tubes.
    d=np.zeros((96,sqp.STATE_DIM));d[:,69:71]=.16
    b,bt=sqp.retract_native(p,obj,d);bv=decode(b)['human_vertices'][:,:2]
    span=bv[:,1,0].astype(float)-bv[:,0,0].astype(float)
    radii=e.baseline_gap_m+e.config.numerical_slack_m
    assert np.all(span[e.activations.all(1)] > 1+radii.sum(1)[e.activations.all(1)])
    result=sqp.optimize_native_pose(p,obj,e,decode_native=decode,linearize_native=linearize,
        evaluate_objective=objective,evaluate_observations=observe,gates=gates)
    assert result['status']=='accepted_native_pose_sqp' and result['accepted_steps']>0
    assert result['objective_after']<result['objective_before']*.4
    assert np.max(angles(result['parameters']))>.05
    assert np.all(result['witness_gaps_m'][e.activations] <= radii[e.activations])
    np.testing.assert_array_equal(result['original_activations'],e.activations)
    np.testing.assert_array_equal(result['geometry']['frame_index'],e.frame_index)
    assert np.linalg.norm(result['geometry']['human_vertices'][-1].mean(0)-result['geometry']['human_vertices'][0].mean(0))>.02
    assert not result['physical_contact_verified'] and not result['heldout_accuracy_verified']


def test_native_tangent_Jacobian_matches_finite_difference_and_frozen_blocks():
    p,obj,e,decode,linearize,*_=authored();g=decode(p);lin=linearize(p,obj,g,e)
    for tangent in (69,70):
        d=np.zeros((96,sqp.STATE_DIM));d[:,tangent]=1e-4
        plus,_=sqp.retract_native(p,obj,d);minus,_=sqp.retract_native(p,obj,-d)
        finite=(decode(plus)['human_vertices'][:,:2].astype(float)-decode(minus)['human_vertices'][:,:2].astype(float))/2e-4
        np.testing.assert_allclose(finite,lin['witness_jacobian_camera'][...,tangent],atol=3e-4,rtol=0)
    d=np.zeros((96,sqp.STATE_DIM));d[3,:3]=[.01,-.02,.03]
    updated,_=sqp.retract_native(p,obj,d)
    for key in NATIVE_PARAMETER_DIMS.keys()-{'mhr_trans','mhr_body_pose_cont'}:
        assert updated[key].tobytes()==p[key].tobytes()
    assert updated['mhr_body_pose_cont'][:,254:].tobytes()==p['mhr_body_pose_cont'][:,254:].tobytes()
    actual=sqp._so3(updated['mhr_body_pose_cont'][3,:6].astype(float))
    np.testing.assert_allclose(actual,Rotation.from_rotvec(d[3,:3]).as_matrix(),atol=1e-7)


@pytest.mark.parametrize('bad',['IDs','frames','activity','jacshape','nonfinite','metric'])
def test_callback_cannot_drop_frame_contact_ID_or_native_tangent_abi(bad):
    p,obj,e,decode,linearize,objective,observe,gates,_=authored()
    def wrong(*args):
        value=linearize(*args)
        if bad=='IDs': value['witness_source_indices']=value['witness_source_indices'].copy();value['witness_source_indices'][0,0]=1
        elif bad=='frames': value['frame_index']=value['frame_index'][1:]
        elif bad=='activity': value['activations']=value['activations'].copy();value['activations'][0]=False
        elif bad=='jacshape': value['witness_jacobian_camera']=value['witness_jacobian_camera'][...,:126]
        elif bad=='nonfinite': value['gradient'][0,0]=np.nan
        elif bad=='metric': value['metric_diagonal'][0,0]=0
        return value
    with pytest.raises(ValueError): sqp.optimize_native_pose(p,obj,e,decode_native=decode,linearize_native=wrong,
        evaluate_objective=objective,evaluate_observations=observe,gates=gates)


def test_failed_independent_QA_returns_complete_dynamic_A_not_best_unchecked_candidate():
    p,obj,e,decode,linearize,objective,observe,gates,_=authored()
    def qa(geometry,ot):
        metrics=observe(geometry,ot);metrics['constant_reference']=1. if np.array_equal(geometry['human_vertices'],decode(p)['human_vertices']) else 2.
        return metrics
    result=sqp.optimize_native_pose(p,obj,e,decode_native=decode,linearize_native=linearize,
        evaluate_objective=objective,evaluate_observations=qa,gates=gates)
    assert result['status']=='dynamic_A_fallback_no_improvement'
    assert all(result['parameters'][k].tobytes()==p[k].tobytes() for k in p)
    np.testing.assert_array_equal(result['object_translation'],obj)
    assert result['objective_after']==result['objective_before']


def test_no_contact_unconstrained_frames_not_deleted_or_static_and_policy_frozen():
    p,obj,e,decode,linearize,objective,observe,gates,_=authored()
    assert sqp.PoseSQPConfig().max_steps==20 and sqp.PoseSQPConfig().budget_seconds==900
    for kwargs in [dict(max_steps=300),dict(budget_seconds=901),dict(qp_tolerance=.1)]:
        with pytest.raises(ValueError): sqp.PoseSQPConfig(**kwargs)
    step=np.zeros((96,sqp.STATE_DIM));step[32,69]=.02
    after,ot=sqp.retract_native(p,obj,step)
    assert after['mhr_body_pose_cont'][31].tobytes()==p['mhr_body_pose_cont'][31].tobytes()
    assert after['mhr_body_pose_cont'][32].tobytes()!=p['mhr_body_pose_cont'][32].tobytes()
    np.testing.assert_array_equal(ot,obj)


def test_incompatible_local_QP_and_exact_zero_gap_cusp_do_not_fake_signed_normal_or_feasibility():
    grad=np.zeros((1,sqp.STATE_DIM));a=np.zeros((2,sqp.STATE_DIM));a[:,0]=[1,-1]
    with pytest.raises(sqp.QPProjectionFailure):
        sqp.project_diagonal_qp(grad,np.ones_like(grad),[a],[np.array([-.1,-.1])],np.ones_like(grad))
    p,obj,e,decode,linearize,*_=authored();g=decode(p)
    # Force one query onto the triangle interior and require BOTH unsigned
    # face-plane derivative rows. It must not masquerade as a signed SDF.
    g['human_vertices'][0,0]=[-.5,0,2.]
    surface=sqp.contact._NearestSurface(e.object_vertices,e.object_faces)
    _,rows,_=sqp._contact_rows(surface,e,g,obj,linearize(p,obj,g,e)['witness_jacobian_camera'])
    assert len(rows[0])==3
    np.testing.assert_allclose(rows[0][0],-rows[0][1],atol=0)


def test_over_budget_linearization_return_is_discarded_before_new_native_decode(monkeypatch):
    p,obj,e,decode,linearize,objective,observe,gates,_=authored();clock=[0.];decodes=[]
    monkeypatch.setattr(sqp.time,'monotonic',lambda:clock[0])
    def too_long(*args):
        result=linearize(*args);clock[0]=901.;return result
    def native(params):decodes.append(params);return decode(params)
    result=sqp.optimize_native_pose(p,obj,e,decode_native=native,linearize_native=too_long,
        evaluate_objective=objective,evaluate_observations=observe,gates=gates)
    assert len(decodes)==1 and result['budget_exhausted']
    assert result['accepted_steps']==0 and result['status']=='dynamic_A_fallback_no_improvement'
    assert result['attempts'][0]['status']=='budget_exhausted_callback_result_discarded'


def test_retraction_preserves_nonunit_native_encoding_gauge_and_every_zero_block():
    p,obj,*_=authored();raw=p['mhr_body_pose_cont'][:,:138].reshape(96,23,6)
    raw[:,:,:3]*=1.7;raw[:,:,3:]*=.8;raw[:,:,3:]+=raw[:,:,:3]*.3
    p['mhr_body_pose_cont'][:,138:254]*=2.4
    before=p['mhr_body_pose_cont'].copy();d=np.zeros((96,sqp.STATE_DIM));d[:,1]=.04;d[:,71]=.02
    changed,_=sqp.retract_native(p,obj,d);after=changed['mhr_body_pose_cont']
    assert before[:,6:138].tobytes()==after[:,6:138].tobytes()
    for block in set(range(58))-{2}:
        assert before[:,138+2*block:140+2*block].tobytes()==after[:,138+2*block:140+2*block].tobytes()
    def gauge(x):
        v=x[:,:138].astype(float).reshape(96,23,6);r=sqp._so3(v)
        return (np.linalg.norm(v[...,:3],axis=-1),np.sum(v[...,3:]*r[..., :,0],axis=-1),
            np.linalg.norm(v[...,3:]-np.sum(v[...,3:]*r[..., :,0],axis=-1)[...,None]*r[..., :,0],axis=-1),
            np.linalg.norm(x[:,138:254].astype(float).reshape(96,58,2),axis=-1))
    for a,b in zip(gauge(before),gauge(after)):np.testing.assert_allclose(a,b,rtol=1e-7,atol=2e-7)
    zero,_=sqp.retract_native(p,obj,np.zeros_like(d))
    assert all(zero[k].tobytes()==p[k].tobytes() for k in p)

"""Tiny integration fixtures only; the complete stress grid executes on Azure."""
from dataclasses import replace
from pathlib import Path
import subprocess
import numpy as np
import pytest

import sequence_evidence_stress as stress
import world_reward.sequence_pose as pose
from world_reward.depth_covariance import CommonModeDepthConfig
from world_reward.contact_patch import ContactPatchConfig
from test_sequence_pose_rgbd import fixture, config, depth_config


def fit(f, **kw):
    v,k,r,t,xy,depth,visible=f
    return pose.refine_sequence(v,v,xy,visible,r,t,np.ones(len(t),bool),k,np.arange(len(t)),30,config(),
        tracks_depth_m=depth,depth_visible=visible,depth_config=depth_config(),**kw)


def test_zero_covariance_integrated_route_is_bit_exact_with_legacy():
    f=fixture(6);f[3][1:,0]+=.002
    a=fit(f);b=fit(f,depth_covariance_config=CommonModeDepthConfig(0.,'owned_unit'))
    np.testing.assert_array_equal(a.rotations,b.rotations)
    np.testing.assert_array_equal(a.translations,b.translations)
    assert a.diagnostics==b.diagnostics


def test_covariance_does_not_densify_sparse_frame_dependencies(monkeypatch):
    f=fixture(6);v,k,r,t,xy,depth,visible=f;actual=pose.least_squares;seen=[]
    def record(fun,start,**kwargs):
        n,q=visible.shape;rows=(n-1)*q;pattern=kwargs['jac_sparsity'][-rows:].toarray()
        for i in range(rows): assert np.flatnonzero(pattern[i]).tolist()==list(range((i//q)*6,(i//q+1)*6))
        seen.append(fun(start).shape)
        return actual(fun,start,**kwargs)
    monkeypatch.setattr(pose,'least_squares',record)
    r=fit(f,depth_covariance_config=CommonModeDepthConfig(.1,'owned_unit'))
    assert seen and r.diagnostics['depth_temporal_covariance_modeled'] is False
    assert r.diagnostics['depth_weighting']=='symmetric_rank_one_withinframe_whitened_soft_l1'


def test_covariance_cannot_be_enabled_without_inferred_depth():
    f=fixture(6);v,k,r,t,xy,_,visible=f
    with pytest.raises(ValueError,match='requires explicit inferred depth'):
        pose.refine_sequence(v,v,xy,visible,r,t,np.ones(len(t),bool),k,np.arange(len(t)),30,config(),
            depth_covariance_config=CommonModeDepthConfig(.1,'owned_unit'))


def test_single_contact_patch_matches_existing_fit_and_no_contact_parity():
    from test_sequence_pose_contact import fixture as contact_fixture, call, contact_config
    v,faces,k,r,t,xy,visible,e=contact_fixture(6)
    singleton=pose.SequenceContactEvidence(e.activations,e.hand_points_camera[:,:,:1],e.hand_visible[:,:,:1],faces,e.source_reference)
    cfg=replace(contact_config(),max_points_per_hand=1)
    f=(v,faces,k,r,t,xy,visible,singleton)
    a=call(f,contact_evidence=singleton,contact_config=cfg)
    b=call(f,contact_evidence=singleton,contact_config=cfg,contact_patch_config=ContactPatchConfig(.01,1,'owned_unit'))
    np.testing.assert_array_equal(a.translations,b.translations)
    np.testing.assert_array_equal(a.rotations,b.rotations)
    assert b.diagnostics['contact_tangential_lock'] is False
    absent=pose.SequenceContactEvidence(np.zeros_like(singleton.activations),singleton.hand_points_camera,
        singleton.hand_visible,faces,'owned_unit')
    a=call(f);b=call(f,contact_evidence=absent,contact_config=cfg,contact_patch_config=ContactPatchConfig(.01,1,'owned_unit'))
    np.testing.assert_array_equal(a.translations,b.translations);assert a.diagnostics==b.diagnostics


def test_full_contact_pool_never_silently_reduced():
    from test_sequence_pose_contact import fixture as contact_fixture, call, contact_config
    f=contact_fixture(6);e=f[-1]
    with pytest.raises(ValueError,match='pool exceeds'):
        call(f,contact_evidence=e,contact_config=replace(contact_config(),max_points_per_hand=1),
            contact_patch_config=ContactPatchConfig(.01,8,'owned_unit'))


def test_stress_inputs_are_analytic_distinct_from_scoring_and_source_pinned():
    root=Path(__file__).resolve().parents[1];cfg=stress.settings(root)
    assert cfg['development_seed']!=cfg['reserved_seed']
    assert cfg['production_adopted'] is False and cfg['challenge_inputs_allowed'] is False
    f=stress.observations(cfg,np.random.default_rng(2),True,True)
    v,_,p,k,r,t,truth,xy,prior_r,prior_t,d=f
    assert not np.array_equal(prior_t,t) and not np.array_equal(prior_r,r)
    assert d>0 and np.isfinite(xy).all() and len(truth)==24
    perfect=stress.scores(v,p,r,t,truth,np.einsum('tij,j->ti',r,v.mean(0))+t,cfg['fps'])
    assert perfect['point_error_m']==0 and perfect['motion_displacement_retention']==1
    assert perfect['motion_path_retention']==1


def test_assay_declares_exact_authored_anchor_and_nonphysical_observations():
    disclosures=stress.protocol_disclosures()
    assert disclosures['challenge_ground_truth_used_for_inference'] is False
    for key in ('authored_known_mesh_and_intrinsics_used',
                'authored_exact_first_frame_gauge_anchor_used',
                'authored_noisy_truth_derived_priors_used',
                'rolling_case_is_two_axis_planar_slide'):
        assert disclosures[key] is True
    assert disclosures['oscillation_amplitude_and_phase_gated'] is False
    cfg=stress.settings(Path(__file__).resolve().parents[1])
    v,_,p,_,r,t,_,_,prior_r,prior_t,_=stress.observations(cfg,np.random.default_rng(2),True,True)
    np.testing.assert_array_equal(prior_r[0],r[0])
    np.testing.assert_array_equal(prior_t[0],t[0])
    assert np.all(np.abs(p)<np.max(np.abs(v),axis=0))


def test_endpoint_gate_cannot_certify_preservation_of_fast_motion():
    cfg=stress.settings(Path(__file__).resolve().parents[1])
    v,_,p,_,r,t,truth,_,_,_,_=stress.observations(cfg,np.random.default_rng(2),True,True)
    removed=t.copy()
    removed[:,2]-=.035*np.sin(2*np.pi*3*np.arange(len(t))/cfg['fps'])
    diagnostic=stress.scores(v,p,r,removed,truth,t,cfg['fps'])
    # A counterexample, not an additional post-hoc acceptance criterion:
    # deleting the real oscillation PASSES the frozen endpoint-only gate.
    assert cfg['gates']['motion_retention_minimum']<diagnostic['motion_displacement_retention']<cfg['gates']['motion_retention_maximum']
    assert diagnostic['motion_path_retention']<.7
    assert diagnostic['temporal_error_m_s']>0


def test_stress_failure_is_not_successful_gate():
    root=Path(__file__).resolve().parents[1];cfg=stress.settings(root)
    row=dict(status='fail',split='reserved',mechanism='depth')
    decisions=stress.gates(cfg,[row])
    assert not next(r for r in decisions if r['mechanism']=='depth' and r['split']=='reserved')['passed']


def test_all_authored_case_plumbing_without_running_full_grid_locally(monkeypatch):
    from types import SimpleNamespace
    cfg=stress.settings(Path(__file__).resolve().parents[1]);calls=[]
    def fake(v,p,xy,visible,r,t,observed,k,grid,fps,fit_cfg,**kwargs):
        assert np.isnan(xy[~visible]).all()
        if 'tracks_depth_m' in kwargs:
            assert np.isnan(kwargs['tracks_depth_m'][~kwargs['depth_visible']]).all()
        if 'contact_evidence' in kwargs:
            e=kwargs['contact_evidence']
            assert np.isnan(e.hand_points_camera[~e.hand_visible]).all()
        calls.append(kwargs)
        return SimpleNamespace(rotations=r,translations=t,diagnostics={'converged':True})
    monkeypatch.setattr(stress,'refine_sequence',fake)
    for mechanism,cases in cfg['cases'].items():
        for i,case in enumerate(cases):
            row=stress.worker((cfg,123+i,mechanism,case,'development'))
            assert row['status']=='complete',row
            assert len(row['metrics'])==2
    assert len(calls)==24


def test_wrapper_no_models_or_challenge_mount_and_parallel_bounded_cpu():
    path=Path(stress.__file__).with_name('run_sequence_evidence_stress.sh');text=path.read_text()
    assert subprocess.run(['rtk','proxy','bash','-n',str(path)],capture_output=True).returncode==0
    assert '--gpus' not in text and 'data/track_1' not in text and 'weights/' not in text
    for x in ('--cpus 4','--network none --read-only','--memory 8g','600s docker run',
              'OPENBLAS_NUM_THREADS=1','container_absence_verified','.world-reward-sequence-evidence-stress.lock'):
        assert x in text

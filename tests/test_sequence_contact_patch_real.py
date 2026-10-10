"""Tiny wiring/contracts only; actual full-T challenge ablation stays on Azure."""
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import sequence_contact_patch_real as real
from test_sequence_pose_contact import fixture, config


def tiny_bank():
    v,faces,k,r,t,xy,visible,evidence=fixture(6)
    # Distinct genuine anatomical IDs, all geometry supported; fixed six frames.
    hands=np.repeat(evidence.hand_points_camera[:,:,:1],12,axis=2)
    hands[:,1]=hands[:,0]  # Inactive anatomy is still predicted finite geometry.
    hands[:,:, :,1]+=np.linspace(-.004,.004,12)[None,None,:]
    return dict(vertices=v,faces=faces,rotations=r,translations=t,points=v,
        K=k,original_K=k,xy=xy,visible=visible,frame_index=np.arange(len(t)),fps=30.,
        hands=hands,ids=np.arange(24).reshape(2,12),evidence=evidence)


def protocol():
    return real.settings(Path(__file__).resolve().parents[1])


def test_settings_frozen_coefficients_and_execution_only_cap():
    cfg=protocol()
    assert cfg['max_nfev']==300 and cfg['workers']==2
    assert cfg['temperature_diameter']==.01 and cfg['contact_sigma_diameter']==.02
    assert not cfg['depth_used'] and not cfg['covariance_used'] and not cfg['production_adopted']
    assert cfg['max_point_triangle_pairs']==1_000_000_000


def test_frozen_anatomical_pool_preserves_original_activation_timeline_and_ids():
    bank=tiny_bank();before=bank['hands'].copy()
    evidence,ids=real.bounded_pool(bank,protocol())
    assert evidence.hand_points_camera.shape==(6,2,8,3)
    np.testing.assert_array_equal(evidence.activations,bank['evidence'].activations)
    np.testing.assert_array_equal(bank['hands'],before)
    assert np.isnan(evidence.hand_points_camera[~evidence.hand_visible]).all()
    for side in range(2):
        rows=np.flatnonzero(evidence.activations[:,side])
        if len(rows):
            assert set(ids[rows[0],side])<=set(bank['ids'][side])
            np.testing.assert_array_equal(ids[rows,side],np.broadcast_to(ids[rows[0],side],(len(rows),8)))


def test_real_metrics_use_same_original_j1_not_first_patch_point(monkeypatch):
    bank=tiny_bank();pool,_=real.bounded_pool(bank,protocol());calls=[]
    actual=real.contact.residuals
    def recorded(v,p,r,t,k,xy,visible,e,fps):
        calls.append(e);return actual(v,p,r,t,k,xy,visible,e,fps)
    monkeypatch.setattr(real.contact,'residuals',recorded)
    result=real.measure(bank,bank['rotations'],bank['translations'],pool)
    assert calls==[bank['evidence']]
    assert result['RGB_motion_increment_error_mean_px']>=0
    assert result['RGB_motion_increment_observations']>0
    assert result['same_J1_surface_gap_p95_m']>=0
    assert result['full_pool_existential_surface_gap_p95_m']>=0


def test_three_variant_wiring_same_budget_different_factor(monkeypatch):
    bank=tiny_bank();pool,_=real.bounded_pool(bank,protocol());calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(rotations=args[4],translations=args[5],diagnostics={'converged':True})
    monkeypatch.setattr(real,'refine_sequence',fake)
    fit_cfg=replace(config(),max_nfev=300)
    for name in ('J1','hard_pool','soft_pool'):
        result=real.fit_worker((name,bank,bank['evidence'] if name=='J1' else pool,protocol(),fit_cfg))
        assert result['status']=='complete'
    assert all(args[10].max_nfev==300 for args,_ in calls)
    assert [kw['contact_config'].max_points_per_hand for _,kw in calls]==[1,8,8]
    assert ['contact_patch_config' in kw for _,kw in calls]==[False,False,True]
    assert all('depth_config' not in kw and 'tracks_depth_m' not in kw for _,kw in calls)


def test_no_silent_execution_cap_retry_or_candidate_drop(monkeypatch):
    bank=tiny_bank();cfg=protocol()
    def reject(*args,**kw):raise ValueError('cap exceeded')
    monkeypatch.setattr(real,'refine_sequence',reject)
    row=real.fit_worker(('soft_pool',bank,bank['evidence'],cfg,config()))
    assert row['status']=='fail' and row['error']=='cap exceeded'


def test_actual_tiny_worker_integration_preserves_geometry_and_original_anchor():
    bank=tiny_bank();pool,_=real.bounded_pool(bank,protocol())
    for name in ('J1','hard_pool','soft_pool'):
        row=real.fit_worker((name,bank,bank['evidence'] if name=='J1' else pool,
            protocol(),replace(config(),max_nfev=300)))
        assert row['status']=='complete',row
        np.testing.assert_array_equal(row['rotations'][0],bank['rotations'][0])
        np.testing.assert_array_equal(row['translations'][0],bank['translations'][0])
        assert row['fit']['static_constraint'] is False
        assert 'depth_weighting' not in row['fit']


def metrics(value=1.):
    return {name:value for name in ('RGB_mean_px','selected_anatomical_triangle_mean_m',
        'same_J1_surface_gap_p95_m','acceleration_proxy_m_s2_p95',
        'angular_acceleration_proxy_rad_s2_p95','RGB_motion_increment_error_mean_px')}


def test_decision_requires_convergence_both_controls_and_contact_motion():
    cfg=protocol();m={'original':metrics(),'J1':metrics(),'soft_pool':metrics(.8)}
    assert real.decision(cfg,m,{'J1':True,'soft_pool':True},'soft_pool')['passed']
    assert not real.decision(cfg,m,{'J1':False,'soft_pool':True},'soft_pool')['passed']
    assert not real.decision(cfg,m,{'J1':True,'soft_pool':False},'soft_pool')['passed']
    for key in metrics():
        worse={'original':metrics(),'J1':metrics(),'soft_pool':{**metrics(.8),key:1.1}}
        assert not real.decision(cfg,worse,{'J1':True,'soft_pool':True},'soft_pool')['passed']
    assert not real.decision(cfg,{'original':metrics(),'J1':metrics()}, {'J1':True},'soft_pool')['passed']


def test_freeze_cannot_pass_by_acceleration_alone():
    cfg=protocol();m={'original':metrics(),'J1':metrics(),'soft_pool':metrics(.1)}
    m['soft_pool']['RGB_motion_increment_error_mean_px']=3.
    row=real.decision(cfg,m,{'J1':True,'soft_pool':True},'soft_pool')
    assert not row['passed']
    assert row['production_adopted'] is False and row['heldout_accuracy_verified'] is False


def test_wrapper_mounts_saved_inputs_only_bounds_concurrency_and_cleans_owned():
    p=Path(real.__file__).with_name('run_sequence_contact_patch_real.sh');s=p.read_text()
    assert subprocess.run(['rtk','proxy','bash','-n',str(p)],capture_output=True).returncode==0
    assert '--gpus' not in s and 'data/track_1' not in s
    assert 'sequence-pose-contact-$CONTACT_SOURCE' in s
    assert '--cpus 4' in s and '--memory 16g' in s and '2400s docker run' in s
    assert 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1' in s
    assert '--network none --read-only' in s and 'container_absence_verified' in s
    assert 'world_reward.sequence_contact_patch_real.owner' in s

"""Owned CPU diagnostic fixtures; no challenge artifacts or model calls."""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from PIL import Image
import pose_gain_probe as probe


def fixture(n=8):
    rng=np.random.default_rng(2);v=rng.uniform(-.1,.1,(33,3));r=np.broadcast_to(np.eye(3),(n,3,3)).copy();t=np.zeros((n,3));t[:,2]=2.
    original=dict(object_vertices=v,object_faces=np.array([[0,1,2]]),object_scale=np.array(1.),camera_K=np.diag([2.,2.,1.]),
        frame_index=np.arange(n),object_rotation=r,object_translation=t)
    bank={k:original[k] for k in ('object_vertices','object_faces','object_scale','camera_K','frame_index')}
    bank.update(rotation=r,translation=t,points=v.copy(),tracks_xy=np.ones((n,33,2)),RGB_visible=np.ones((n,33),bool))
    return original,bank


def test_saved_bank_requires_identical_shape_scale_camera_grid_witnesses():
    a,b=fixture();probe.same_bank(a,b,b)
    for key in ('object_vertices','object_scale','camera_K','frame_index'):
        bad=dict(b);bad[key]=np.array(b[key],copy=True);bad[key].flat[0]+=1
        with pytest.raises(ValueError):probe.same_bank(a,bad,b)
    for key in ('points','tracks_xy','RGB_visible'):
        bad=dict(b);bad[key]=np.array(b[key],copy=True);bad[key].flat[0]+=1 if key!='RGB_visible' else False
        if key=='RGB_visible':bad[key].flat[0]=False
        with pytest.raises(ValueError):probe.same_bank(a,bad,b)


def test_malformed_rigid_pose_and_observation_semantics_fail():
    a,b=fixture();bad=dict(b);bad['rotation']=b['rotation'].copy();bad['rotation'][0,0,0]=2
    with pytest.raises(ValueError):probe.same_bank(a,bad)
    bad=dict(b);bad['tracks_xy']=b['tracks_xy'].copy();bad['tracks_xy'][0,0]=np.nan
    with pytest.raises(ValueError):probe.same_bank(a,bad)


def test_physical_centroid_and_seconds_acceleration():
    a,b=fixture();v=a['object_vertices'];r=b['rotation'];t=b['translation'].copy();t[:,0]=np.arange(len(t))**2*.001
    linear,angular=probe.acceleration(v,r,t,30.)
    np.testing.assert_allclose(linear,1.8);np.testing.assert_array_equal(angular,0)
    shift=np.array([.2,.1,-.05]);shifted_t=t-shift@r.swapaxes(-1,-2)
    aa,ab=probe.acceleration(v+shift,r,shifted_t,30.)
    np.testing.assert_allclose(aa,linear,atol=1e-10);np.testing.assert_array_equal(ab,angular)


def test_projection_jacobian_centroid_pivot_finite_difference_and_origin_invariance():
    a,b=fixture();v=a['object_vertices'];p=b['points'];r=Rotation.from_rotvec([.1,.2,-.05]).as_matrix();t=np.array([.1,.05,2.]);K=np.array([[320.,0,128],[0,310.,128],[0,0,1.]])
    j=probe.projection_jacobian(p,v,r,t,K);base=probe.old.project(p,r[None],t[None],K)[0];diam=np.linalg.norm(np.ptp(v,axis=0));centroid=v.mean(0)@r.T+t
    for k in range(6):
        e=np.eye(6)[k]*1e-7
        rr=Rotation.from_rotvec(e[:3]).as_matrix()@r;tt=centroid+e[3:]*diam-v.mean(0)@rr.T
        diff=(probe.old.project(p,rr[None],tt[None],K)[0]-base).reshape(-1)/1e-7
        np.testing.assert_allclose(diff,j[:,k],atol=1e-5,rtol=1e-5)
    shift=np.array([.3,-.2,.7]);jj=probe.projection_jacobian(p+shift,v+shift,r,t-shift@r.T,K)
    np.testing.assert_allclose(j,jj,atol=1e-11)


def test_projection_jacobian_negative_depth_no_clipping():
    a,b=fixture()
    with pytest.raises(ValueError):probe.projection_jacobian(b['points'],a['object_vertices'],b['rotation'][0],np.array([0.,0.,-2.]),a['camera_K'])


def test_contact_top_spikes_are_ranked_not_removed_and_nominal_boundaries_not_cause():
    values=np.ones(220);values[95]=20.;values[144]=40.;row=probe.concentration(values)
    assert row['top10_frame_indices'][:2]==[145,96]
    assert row['nominal_96_window_boundary']['count']+row['other_frames']['count']==220
    assert row['boundary_scope']=='descriptive_nominal_window_not_verified_cause'
    assert 0<row['top10_sum_fraction']<1


def test_depth_commonmode_keeps_unknown_and_is_not_truth():
    a,b=fixture();p=b['points'];r=b['rotation'];t=b['translation'];z=(p[None]@r.swapaxes(-1,-2)+t[:,None])[...,2]
    offset=np.arange(len(t))*.02;depth=z+offset[:,None];support=np.ones(depth.shape,bool);support[2]=False;depth[2]=np.nan
    row=probe.depth_commonmode(p,r,t,depth,support,30.)
    assert row['frame_median_residual_m'][2] is None
    np.testing.assert_allclose([x for x in row['frame_median_residual_m'] if x is not None],offset[np.arange(len(t))!=2])
    assert row['withinframe_absolute_about_median']['p95']<1e-12
    assert row['commonmode_velocity_m_s']['median']==pytest.approx(.6)
    assert 'not_truth' in row['scope']


def test_invalid_supported_depth_fails_not_interpolated():
    a,b=fixture();depth=np.ones((8,33));support=np.ones(depth.shape,bool);depth[0,0]=np.nan
    with pytest.raises(ValueError):probe.depth_commonmode(b['points'],b['rotation'],b['translation'],depth,support,30.)


def test_mask_overlap_exact_pinned_pixel_mapping_and_not_amodal_claim(tmp_path):
    base=tmp_path/'episode';inventory={};scale=np.array([640/5,480/5]);xy=np.array([[(np.array([2.,3.])+.5)*scale-.5,(np.array([1.,1.])+.5)*scale-.5,[np.nan,np.nan]]]);visible=np.array([[True,True,False]])
    for role in (0,1):
        mask=np.zeros((5,5),np.uint8);mask[3,2]=255
        if role==1:mask[1,1]=255
        path=base/f'automatic_masks/masks/{role}/000000.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(path)
        inventory[f'{role}/000000.png']=probe.old.ArtifactLedger().record(path)
    ledger=probe.old.ArtifactLedger();row,frac=probe.support_overlap(ledger,base,inventory,xy,visible,scale)
    assert row['person_queries']==1 and row['object_any_queries']==2 and row['object_only_queries']==1
    assert frac['person'].tolist()==[.5];ledger.verify()
    path=base/'automatic_masks/masks/0/000000.png';Image.fromarray(np.zeros((5,5),np.uint8)).save(path)
    with pytest.raises(ValueError,match='SHA mismatch'):probe.support_overlap(probe.old.ArtifactLedger(),base,inventory,xy,visible,scale)


def test_correlations_abstain_for_uninformative_or_missing_evidence():
    assert probe.correlation(np.ones(6),np.arange(6)) is None
    assert probe.correlation([0.,np.nan],[1.,2.]) is None
    assert probe.correlation(np.arange(6),np.arange(6))==pytest.approx(1.)


def test_stage_times_report_recorded_parallel_nonadditivity():
    row=probe.stage_runtime([dict(episode=9,stages=[dict(stage='x',elapsed_seconds=5,reused=False),dict(stage='y',elapsed_seconds=8,reused=True),dict(stage='none')])])[0]
    assert row['stages'][0]['stage']=='y' and row['measured_sum_seconds']==13.
    assert 'not_additive_parallel_total' in row['scope']


def test_runtime_is_saved_only_cpu_readonly_sources_and_exact_cid():
    source=Path(probe.__file__).read_text();wrapper=Path(probe.__file__).with_name('run_pose_gain_probe.sh').read_text()
    assert 'refine_sequence(' not in source and 'manufactured_gate(' not in source and 'new_fits=0' in source
    assert '--gpus' not in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper and '--network none --read-only' in wrapper
    assert '--cpus 4' in wrapper and '--memory 16g' in wrapper and '300s docker run' in wrapper
    assert 'data/track_1' not in wrapper and 'weights/' not in wrapper
    assert 'sequence-pose-rgbd-$RGBD_SOURCE' in wrapper and 'sequence-pose-contact-$CONTACT_SOURCE' in wrapper
    assert 'world_reward.pose_gain_probe.owner' in wrapper and 'docker rm -f "$cid"' in wrapper and 'container_absence_verified' in wrapper
    assert 'GPU_requested":false' in wrapper

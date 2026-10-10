"""Tiny manufactured memory contracts, never challenge performance validation."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from world_reward.contact_memory import (ContactMemoryConfig, vertex_normals,
    temporal_path, evolving_contact_patch)
import contact_memory_real as real


def config():
    return ContactMemoryConfig(.02,.02,np.pi/3,2,3,'manufactured:unit_contract_not_quality')


def anatomy(count=8):
    # Connected triangulated hand patch, all native points legitimately predicted.
    local=np.array([[x,y,0.] for y in (-.02,0,.02) for x in (-.02,0,.02)])
    faces=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4],[3,4,7],[3,7,6],[4,5,8],[4,8,7]])
    hands=np.broadcast_to(local,(count,2,9,3)).copy();hands[...,2]=2.01
    v=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])
    f=np.array([[0,1,2],[0,2,3]])
    r=np.broadcast_to(np.eye(3),(count,3,3)).copy();t=np.tile([0.,0,2.],(count,1))
    active=np.zeros((count,2),bool);active[:,0]=True
    ids=np.arange(18).reshape(2,9)
    return hands,ids,active,r,t,v,f,(faces,faces)


def test_configuration_is_frozen_and_scales_explicit():
    c=config()
    with pytest.raises(FrozenInstanceError):c.proposal_count=8
    for change in ({'distance_sigma_diameter':0},{'normal_sigma_rad':np.nan},
                   {'patch_count':9},{'proposal_count':True},{'development_reference':' '}):
        with pytest.raises(ValueError):replace(c,**change)


def test_area_weighted_normal_zero_is_unknown_not_fabricated():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[2,2,2]])
    n,valid=vertex_normals(v,np.array([[0,1,2],[3,3,3]]))
    np.testing.assert_allclose(n[:3],[[0,0,1]]*3)
    assert not valid[3] and not n[3].any()


def test_chain_keeps_stable_material_id_but_allows_real_motion():
    ids=[np.array([0,1])]*5
    points=[np.array([[frame*.1,0,0],[frame*.1+.02,0,0]]) for frame in range(5)]
    scores=[np.array([0.,.01]) if frame%2==0 else np.array([.01,0.]) for frame in range(5)]
    path=temporal_path(ids,points,scores,.01)
    np.testing.assert_array_equal(path,[0]*5)
    assert points[-1][path[-1],0]-points[0][path[0],0]==.4
    assert not path.flags.writeable


def test_chain_switches_when_geometry_support_really_changes():
    ids=[np.array([0,1])]*4
    points=[np.array([[0.,0,0],[.02,0,0]])]*4
    scores=[np.array([0.,100.])]*2+[np.array([100.,0.])]*2
    np.testing.assert_array_equal(temporal_path(ids,points,scores,.01),[0,0,1,1])


def test_ragged_candidates_allow_new_hand_patch_without_interval_union_cap():
    ids=[np.array([0,1]),np.array([1,2]),np.array([2,3])]
    points=[np.array([[0.,0,0],[.01,0,0]])]*3
    scores=[np.array([0.,2.]),np.array([2.,0.]),np.array([2.,0.])]
    prior=[None,np.array([[-.01,0,0],[0.,0,0]]),np.array([[-.01,0,0],[0.,0,0]])]
    path=temporal_path(ids,points,scores,.02,prior)
    assert path.shape==(3,) and all(path[i] in ids[i] for i in range(3))


def test_evolving_patch_preserves_activation_geometry_and_no_static_substitution():
    f=list(anatomy());f[0][:,0,:,0]+=np.arange(8)[:,None]*.002
    before=f[0].copy();result=evolving_contact_patch(*f,config())
    np.testing.assert_array_equal(result['native_activations'],f[2])
    np.testing.assert_array_equal(f[0],before)
    assert result['hand_points_camera'].shape==(8,2,3,3)
    assert result['geometry_supported'][:,0].all() and not result['geometry_supported'][:,1].any()
    assert np.isnan(result['hand_points_camera'][:,1]).all()
    for frame in range(8):
        for j,ident in enumerate(result['selected_ids'][frame,0]):
            local=np.flatnonzero(f[1][0]==ident)[0]
            np.testing.assert_array_equal(result['hand_points_camera'][frame,0,j],f[0][frame,0,local])
    for value in result.values():
        if isinstance(value,np.ndarray):
            with pytest.raises(ValueError):value.setflags(write=True)
    assert not result['material_trajectory_frozen'] and not result['physical_contact_verified']
    saved=result['hand_points_camera'].copy();f[0][:]=100.
    np.testing.assert_array_equal(result['hand_points_camera'],saved)


def test_inactive_intervals_reset_memory_and_no_positive_logits_means_no_patch():
    f=list(anatomy());f[2][3:5]=False
    r=evolving_contact_patch(*f,config())
    assert (r['selected_ids'][3:5]==-1).all() and np.isnan(r['hand_points_camera'][3:5]).all()
    f[2][:]=False;r=evolving_contact_patch(*f,config())
    assert (r['winner_ids']==-1).all() and not r['geometry_supported'].any()


def test_shared_rigid_frame_change_preserves_anatomical_associations():
    from scipy.spatial.transform import Rotation
    f=list(anatomy())
    # Avoid symmetric geometry ties: tiny FP roundoff may legitimately choose a
    # different equally good corner; material IDs are not geometry accuracy.
    f[0][...,0]+=.013;f[0][...,1]+=.009
    original=evolving_contact_patch(*f,config())
    rotation=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();translation=np.array([1.,2.,3.])
    f[0]=f[0]@rotation.T+translation;f[3]=rotation[None]@f[3];f[4]=f[4]@rotation.T+translation
    transformed=evolving_contact_patch(*f,config())
    np.testing.assert_array_equal(original['selected_ids'],transformed['selected_ids'])
    np.testing.assert_allclose(transformed['hand_points_camera'][transformed['geometry_supported']],
        (original['hand_points_camera']@rotation.T+translation)[original['geometry_supported']],atol=1e-14)


@pytest.mark.parametrize('bad',['NaN','rotation','topology','support','duplicate'])
def test_invalid_predicted_geometry_or_anatomy_fail_not_self_disable(bad):
    f=list(anatomy())
    if bad=='NaN':f[0][0,0,0]=np.nan
    if bad=='rotation':f[3][0,0,0]=2
    if bad=='topology':f[7]=(np.array([[0,1,100]]),f[7][1])
    if bad=='support':f[2]=f[2].astype(float)
    if bad=='duplicate':f[1][0,1]=f[1][0,0]
    with pytest.raises(ValueError):evolving_contact_patch(*f,config())


def test_frozen_real_protocol_and_sandbox():
    root=Path(__file__).resolve().parents[1];cfg=real.settings(root)
    assert cfg['max_nfev']==300 and cfg['association']['patch_count']==8
    assert cfg['gates']['compare_against']==['original','J1','soft_pool']
    shell=(root/'infra/run_contact_memory_real.sh').read_text()
    assert '--network none' in shell and '--read-only' in shell and '--gpus' not in shell
    assert 'sequence-contact-patch-real-$POOL_SOURCE' in shell
    assert 'CUDA_VISIBLE_DEVICES=-1' in shell and 'run_contact_memory_real/code' in shell


def test_worker_keeps_original_pose_priors_and_no_new_weights(monkeypatch):
    from test_sequence_contact_patch_real import tiny_bank,protocol
    from test_sequence_pose_contact import config as fit_config
    bank=tiny_bank();pool,ids=real.real.bounded_pool(bank,protocol())
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(rotations=args[4],translations=args[5],diagnostics={'converged':True})
    monkeypatch.setattr(real,'refine_sequence',fake)
    cfg=real.settings(Path(__file__).resolve().parents[1])
    row=real.fit_worker(bank,pool,{'selected_ids':ids},cfg,replace(fit_config(),max_nfev=300))
    assert row['status']=='complete'
    args,kw=calls[0]
    assert args[4] is bank['rotations'] and args[5] is bank['translations']
    assert args[10].max_nfev==300 and kw['contact_config'].contact_sigma_diameter==.02
    assert kw['contact_patch_config'].temperature_diameter==.01
    assert 'depth_config' not in kw


def test_tiny_real_fitter_keeps_complete_pose_gauge_and_no_static_or_depth():
    from test_sequence_contact_patch_real import tiny_bank,protocol
    from test_sequence_pose_contact import config as fit_config
    bank=tiny_bank();pool,ids=real.real.bounded_pool(bank,protocol())
    cfg=real.settings(Path(__file__).resolve().parents[1])
    row=real.fit_worker(bank,pool,{'selected_ids':ids},cfg,replace(fit_config(),max_nfev=300))
    assert row['status']=='complete',row
    np.testing.assert_array_equal(row['rotations'][0],bank['rotations'][0])
    np.testing.assert_array_equal(row['translations'][0],bank['translations'][0])
    assert row['rotations'].shape==bank['rotations'].shape and not row['fit']['static_constraint']
    assert 'depth_weighting' not in row['fit']

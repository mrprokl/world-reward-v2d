"""Tiny owned RGBD/provenance fixtures only, never challenge data or models."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import sequence_pose_rgbd_probe as probe
from test_sequence_pose_probe import raw_json, write


def data():
    return np.arange(1,26,dtype=float).reshape(5,5),np.ones((5,5),bool),np.ones((5,5),bool)


def test_exact_saved_pixel_centre_mapping_and_single_shared_depth_scale():
    depth,valid,mask=data(); xy=np.array([[.75,1.25],[2.,2.]]); scale=np.array([.5,.5])
    # (q+.5)/.5-.5 = (2,3), (4.5,4.5): only first all-four footprint fits.
    z,good=probe.bilinear_depth(depth,valid,mask,xy,np.ones(2,bool),scale,.25)
    assert good.tolist()==[True,False] and z[0]==depth[3,2]*.25 and np.isnan(z[1])


def test_bilinear_noninteger_query_uses_all_four_literal_axial_z_samples():
    depth,valid,mask=data(); z,good=probe.bilinear_depth(depth,valid,mask,np.array([[1.25,2.5]]),np.ones(1,bool),np.ones(2),2.)
    expected=(depth[2,1]*.375+depth[2,2]*.125+depth[3,1]*.375+depth[3,2]*.125)*2
    assert good[0] and z[0]==expected


@pytest.mark.parametrize('kind',['invalid','nan','negative','object_mask'])
@pytest.mark.parametrize('corner',[(0,0),(1,0),(0,1),(1,1)])
def test_one_bad_neighbor_abstains_even_zero_bilinear_weight(kind,corner):
    depth,valid,mask=data();dx,dy=corner;x,y=1+dx,1+dy
    if kind=='invalid':valid[y,x]=False
    elif kind=='nan':depth[y,x]=np.nan
    elif kind=='negative':depth[y,x]=-1
    else:mask[y,x]=False
    z,good=probe.bilinear_depth(depth,valid,mask,np.array([[1.,1.]]),np.ones(1,bool),np.ones(2),1.)
    assert not good[0] and np.isnan(z[0])


def test_hidden_and_offimage_queries_are_nan_not_clamped_or_filled():
    depth,valid,mask=data();xy=np.array([[np.nan,np.nan],[-.1,1],[4.,1]])
    z,good=probe.bilinear_depth(depth,valid,mask,xy,np.array([False,True,True]),np.ones(2),1.)
    assert not good.any() and np.isnan(z).all()


@pytest.mark.parametrize('kind',['mask_dtype','scale','hidden_finite','visible_nan','depth_dtype','scalar'])
def test_malformed_depth_or_support_fails_closed(kind):
    d,v,m=data();xy=np.array([[1.,1.]]);visible=np.ones(1,bool);scale=np.ones(2);scalar=1.
    if kind=='mask_dtype':v=v.astype(int)
    elif kind=='scale':scale[0]=0
    elif kind=='hidden_finite':visible[0]=False
    elif kind=='visible_nan':xy[:]=np.nan
    elif kind=='depth_dtype':d=d.astype(int)
    else:scalar=float('nan')
    with pytest.raises(ValueError):probe.bilinear_depth(d,v,m,xy,visible,scale,scalar)


def depth_fixture(tmp_path,count=3):
    base=tmp_path/'episode';inventory={};rows=[];body=[];h,w=5,5;K=np.eye(3);sha='a'*64
    for i in range(count):
        p=base/f'depth_full/{i:06d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,depth=np.ones((h,w),float)*2,mask=np.ones((h,w),bool),intrinsics=np.diag([1/w,1/h,1]),frame_index=np.array(i))
        pin=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        rows.append(dict(frame_index=i,output_sha256=pin['sha256'],decoded_rgb_sha256='b'*64))
        body.append(dict(frame_index=i,decoded_rgb_sha256='b'*64))
        for role in (0,1):
            p=base/f'automatic_masks/masks/{role}/{i:06d}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.ones((h,w),np.uint8)*(255 if role==1 else 0)).save(p)
            inventory[f'{role}/{i:06d}.png']=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    common=dict(status='pass',episode_index=9,input_track='track_1',input_sha256=sha,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    reports={
        'scale_smoke':dict(common,stage='predicted_human_anchored_moge2_pointmaps',depth_alignment=dict(shared_scale=.25)),
        'depth_full':dict(common,stage='monocular_moge2_full_video',total_video_frames=count,frames=rows),
        'body_full':dict(common,stage='sam3d_body_full_video_initializer',total_video_frames=count,frames=body)}
    for name,row in reports.items():write(base/name/'report.json',raw_json(row))
    scale=np.array([640/w,480/h]);tracks=np.broadcast_to(np.array([[(1.5)*scale[0]-.5,(1.5)*scale[1]-.5]]),(count,1,2)).copy()
    return base,inventory,tracks,np.ones((count,1),bool),scale,K,reports


def test_authenticated_depth_frame_grid_rgb_hash_and_same_gauge(tmp_path):
    base,inventory,xy,visible,scale,K,_=depth_fixture(tmp_path)
    ledger=probe.old.ArtifactLedger();z,good=probe.original_depth(ledger,base,xy,visible,scale,K,inventory,'a'*64)
    assert good.all();np.testing.assert_array_equal(z,np.ones((3,1))*.5);ledger.verify()
    assert len(ledger.records)==12


def test_human_object_overlap_rejects_hand_depth_and_preserves_other_frames(tmp_path):
    base,inventory,xy,visible,scale,K,_=depth_fixture(tmp_path)
    human=np.zeros((5,5),np.uint8);human[2,2]=255  # One of four object-query neighbors.
    path=base/'automatic_masks/masks/0/000001.png';Image.fromarray(human).save(path)
    inventory['0/000001.png']=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ledger=probe.old.ArtifactLedger();z,good=probe.original_depth(ledger,base,xy,visible,scale,K,inventory,'a'*64)
    assert good[:,0].tolist()==[True,False,True] and np.isnan(z[1,0])
    np.testing.assert_array_equal(z[[0,2]],np.ones((2,1))*.5)
    assert str(path) in ledger.records and visible.all()  # RGB tracks never overwritten.
    ledger.verify()


def test_human_mask_cannot_be_unpinned_or_silently_changed(tmp_path):
    base,inventory,xy,visible,scale,K,_=depth_fixture(tmp_path)
    path=base/'automatic_masks/masks/0/000001.png';Image.fromarray(np.ones((5,5),np.uint8)*255).save(path)
    with pytest.raises(ValueError,match='Source SHA mismatch'):
        probe.original_depth(probe.old.ArtifactLedger(),base,xy,visible,scale,K,inventory,'a'*64)


@pytest.mark.parametrize('kind',['rgb','sha','oracle','indices','camera','mask','stored_frame','scale'])
def test_depth_source_mismatch_is_not_relabelled_missing_observation(tmp_path,kind):
    base,inventory,xy,visible,scale,K,reports=depth_fixture(tmp_path)
    if kind=='rgb':reports['body_full']['frames'][1]['decoded_rgb_sha256']='c'*64
    elif kind=='sha':reports['depth_full']['frames'][1]['output_sha256']='c'*64
    elif kind=='oracle':reports['scale_smoke']['ground_truth_used']=True
    elif kind=='indices':reports['depth_full']['frames'][1]['frame_index']=2
    elif kind=='camera':K[0,0]=2
    elif kind=='mask':inventory['1/000001.png']['sha256']='d'*64
    elif kind=='scale':reports['scale_smoke']['depth_alignment']['shared_scale']=0.
    else:
        p=base/'depth_full/000001.npz';np.savez(p,depth=np.ones((5,5),float)*2,mask=np.ones((5,5),bool),intrinsics=np.diag([.2,.2,1]),frame_index=np.array(2))
        reports['depth_full']['frames'][1]['output_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    for name,row in reports.items():write(base/name/'report.json',raw_json(row))
    with pytest.raises(ValueError):probe.original_depth(probe.old.ArtifactLedger(),base,xy,visible,scale,K,inventory,'a'*64)


def test_synthetic_rgbd_truth_never_enters_optimizer_interface(monkeypatch):
    calls=[]
    def fit(vertices,points,xy,visible,r,t,observed,K,indices,fps,cfg,**kwargs):
        assert kwargs['depth_config'] is probe.DEPTH_CFG and kwargs['tracks_depth_m'].shape==visible.shape
        assert kwargs['depth_visible'].dtype==bool and np.array_equal(indices,np.arange(24))
        assert observed.all() and cfg is probe.old.CFG and np.isfinite(xy).all()
        calls.append(True)
        return SimpleNamespace(rotations=r,translations=t,diagnostics=dict(converged=False))
    monkeypatch.setattr(probe,'refine_sequence',fit)
    row=probe.manufactured_gate(20261030,True,probe.old.CFG)
    assert len(calls)==3 and row['seed']==20261030 and not row['passed'] and not row['converged']
    assert [r['case'] for r in row['cases']]==['static','moving','axial_motion']
    json.dumps(row,allow_nan=False)


def test_frozen_depth_sigma_and_profile_do_not_depend_on_challenge_results():
    assert probe.DEPTH_CFG.depth_sigma_diameter==.02
    assert '20261030' in probe.DEPTH_CFG.development_reference
    source=Path(probe.__file__).read_text()
    assert '(20261029,False' in source and '(20261030,True' in source
    assert source.index('manufactured_gate(seed,reserved,config)') < source.index('old.saved_frontend(ledger)')
    assert 'production_adopted=False' in source and 'contact_truth_claimed=False' in source


@pytest.mark.parametrize('kind',[None,'pin','producer','episode','output','oracle','status'])
def test_independent_track_report_pin_cannot_hide_failed_14_or_swap_9(tmp_path,monkeypatch,kind):
    row=dict(status='fail',profile='v2',producer_revision=probe.TRACK_SOURCE,numerical_source=probe.old.SOURCE,
        ground_truth_used=False,manual_labels=False,full_4D_export_replaced=False,model_calls=0,
        episodes=[dict(episode=9,frames=415,points=33,output=probe.TRACK_PIN)])
    if kind=='producer':row['producer_revision']='f'*40
    elif kind=='episode':row['episodes'][0]['episode']=14
    elif kind=='output':row['episodes'][0]['output']=dict(bytes=1,sha256='a'*64)
    elif kind=='oracle':row['ground_truth_used']=True
    elif kind=='status':row['status']='complete'
    path=tmp_path/'report.json';pin=write(path,raw_json(row))
    monkeypatch.setattr(probe,'TRACK_REPORT_PIN',pin if kind!='pin' else dict(bytes=1,sha256='a'*64))
    if kind is None:assert probe.source_track_report(probe.old.ArtifactLedger(),path)==row
    else:
        with pytest.raises(ValueError):probe.source_track_report(probe.old.ArtifactLedger(),path)


def test_production_binds_all_depth_gauge_stage_pins_before_sampling():
    assert set(probe.STAGE_PINS)=={'scale_smoke','body_full','depth_full'}
    assert probe.STAGE_PINS['depth_full']['bytes']==628821
    source=Path(probe.__file__).read_text()
    assert source.index('for stage,pin in STAGE_PINS.items()') < source.index('depth,support = original_depth(')


def test_fixed_fresh_synthetic_rgbd_gates_retain_actual_motion_and_converge():
    config,_=probe.old.profile_config(Path(probe.__file__).resolve().parents[1],'v2')
    for seed,reserved in ((20261029,False),(20261030,True)):
        row=probe.manufactured_gate(seed,reserved,config)
        assert row['passed'] and row['converged']
        for case in row['cases']:
            assert case['new_error_m'] <= .9*case['old_error_m']
            retention=case['full_motion']['full_path_retention']
            if retention is not None:assert .8 <= retention <= 1.2


def test_cpu_wrapper_exact_cid_cleanup_readonly_sources_and_no_gpu():
    s=Path(probe.__file__).with_name('run_sequence_pose_rgbd_probe.sh').read_text()
    assert '--gpus' not in s and 'CUDA_VISIBLE_DEVICES=-1' in s and '--network none --read-only' in s
    assert '.world-reward-sequence-pose-rgbd.lock' in s and 'flock -n 9' in s
    assert 'docker rm -f "$cid"' in s and 'docker rm -f "$NAME"' not in s
    assert 'world_reward.sequence_pose_rgbd.owner' in s and '[[ $# -eq 0 ]]' in s
    assert 'sequence-pose-probe-$TRACK_SOURCE-v2,readonly' in s
    assert 'full4d-v1-$SOURCE,readonly' in s and 'gemini-sam31-$FRONTEND,readonly' in s


def test_output_error_gate_seals_receipt_and_does_not_read_challenge(tmp_path,monkeypatch):
    rev='f'*40;code=tmp_path/'jobs'/rev/probe.ENTRY/'code';out=tmp_path/'results'/('sequence-pose-rgbd-'+rev)
    out.mkdir(parents=True);(out/'.container.cid').write_bytes(b'a'*64)
    monkeypatch.setattr(probe,'ROOT',tmp_path);monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
    for name in probe.HELPERS:write(code/name,b'owned synthetic source')
    monkeypatch.setattr(probe,'source',lambda *_:dict(manufactured=True))
    monkeypatch.setattr(probe.old,'profile_config',lambda *_:(probe.old.CFG,()))
    monkeypatch.setattr(probe,'manufactured_gate',lambda seed,reserved,cfg:dict(passed=not reserved,converged=True,seed=seed))
    monkeypatch.setattr(probe.old,'saved_frontend',lambda *_:pytest.fail('Challenge accessed after failed fresh reserved gate'))
    with pytest.raises(SystemExit):probe.run()
    row=json.loads((out/'report.json').read_text())
    assert row['status']=='fail' and row['reserved']['seed']==20261030 and row['reserved']['passed'] is False
    assert 'reserved gate rejected' in row['error'] and not (out/'report.json').stat().st_mode&0o222

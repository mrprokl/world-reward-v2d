"""Owned tiny automatic contact/provenance tests; no challenge assets."""
from pathlib import Path
from types import SimpleNamespace
import json
import hashlib
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import sequence_pose_contact_probe as p


def bundle(n=3):
    a = np.ones((n,2),np.float32)
    return dict(schema='cari4d.mhr_wild_inference.v1',gt={},frames=[f'{i:06d}' for i in range(n)],
        metadata=dict(ground_truth_used=False),pr=dict(contact_logits=a),pr_initial=dict(contact_logits=a.copy()))


@pytest.mark.parametrize('kind',['gt','schema','frames','metadata','postopt','initial','nan','dtype','masked'])
def test_untouched_native_logits_fail_closed(kind):
    b = bundle()
    if kind=='gt': b['gt']={'contact':True}
    elif kind=='schema': b['schema']='other'
    elif kind=='frames': b['frames'][1]='000002'
    elif kind=='metadata': b['metadata']['ground_truth_used']=True
    elif kind=='postopt': b['postopt']={}
    elif kind=='initial': b['pr_initial']['contact_logits'][0,0]=-1
    elif kind=='nan': b['pr']['contact_logits'][0,0]=np.nan
    elif kind=='dtype': b['pr']['contact_logits']=b['pr']['contact_logits'].astype(float)
    else: b['pr']['contact_logits']=np.ma.array(b['pr']['contact_logits'])
    with pytest.raises(ValueError): p.contact_logits(b,3)


def test_logits_byte_preservation_and_copy():
    b=bundle(); out=p.contact_logits(b,3); assert out.tobytes()==b['pr']['contact_logits'].tobytes()
    out[:]=-1; assert (b['pr']['contact_logits']==1).all()


def geometry():
    # A coarse triangle: point-to-vertex is .707 but continuous surface is .01.
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
    r=np.broadcast_to(np.eye(3),(3,3,3)).copy();t=np.zeros((3,3));t[:,2]=2
    h=np.broadcast_to(np.array([[[.5,.5,2.01],[.5,.5,2.1]],[[3.,3.,2.],[4.,4.,2.]]]),(3,2,2,3)).copy()
    return v,f,r,t,h,np.array([[10,11],[12,13]]),np.ones((3,2),float)


def test_proposal_is_real_anatomical_vertex_gate_is_exact_triangles():
    v,f,r,t,h,ids,log=geometry();e,q=p.automatic_evidence(v,f,r,t,h,ids,log,'owned')
    assert e.activations[:,0].all() and not e.activations[:,1].any()
    assert (q['selected_vertex_indices'][:,0]==10).all()
    assert (q['vertex_proposal_distance_m'][:,0]>.7).all()
    np.testing.assert_allclose(q['exact_initial_triangle_distance_m'][:,0],.01)
    np.testing.assert_array_equal(e.hand_points_camera[:,0,0],h[:,0,0])
    for a in (e.activations,e.hand_points_camera,e.hand_visible,e.object_faces):
        with pytest.raises(ValueError): a.setflags(write=True)
    # Neither caller mutation nor later candidate motion can reselect an anchor.
    h[:]=100;log[:]=-1; assert e.activations[:,0].all()


def test_no_positive_logit_never_activates_even_at_exact_contact():
    v,f,r,t,h,ids,log=geometry();log[:]=0
    e,q=p.automatic_evidence(v,f,r,t,h,ids,log,'owned')
    assert not e.activations.any() and not e.hand_visible.any() and np.isnan(e.hand_points_camera).all()
    assert (q['selected_vertex_indices']==-1).all() and np.isnan(q['exact_initial_triangle_distance_m']).all()


def test_same_frame_so3_gauge_and_mesh_origin_invariance():
    v,f,r,t,h,ids,log=geometry();a,_=p.automatic_evidence(v,f,r,t,h,ids,log,'owned')
    shift=np.array([.4,-.3,.8]);rr=Rotation.from_rotvec([.1,-.2,.3]).as_matrix();tt=np.array([1.,2,3])
    r2=rr[None]@r;t2=t@rr.T+tt;h2=h@rr.T+tt
    b,_=p.automatic_evidence(v,f,r2,t2,h2,ids,log,'owned')
    c,_=p.automatic_evidence(v+shift,f,r,t-shift,h,ids,log,'owned')
    np.testing.assert_array_equal(a.activations,b.activations);np.testing.assert_array_equal(a.activations,c.activations)


@pytest.mark.parametrize('kind',['nan','timeline','logits','rotation','indices'])
def test_bad_observation_is_technical_error_not_fake_missing(kind):
    v,f,r,t,h,ids,log=geometry()
    if kind=='nan':h[0,0,0]=np.nan
    elif kind=='timeline':h=h[:2]
    elif kind=='logits':log=log.astype(int)
    elif kind=='rotation':r[0,0,0]=2
    else:ids=ids.astype(float)
    with pytest.raises(ValueError):p.automatic_evidence(v,f,r,t,h,ids,log,'owned')


def spec_fixture(monkeypatch):
    f=np.array([[0,1,2]],np.int64);ids=np.arange(4636,dtype=np.int32).reshape(2,2318)
    spec=dict(vertex_indices=ids,sample_local_indices=np.broadcast_to(np.arange(256,dtype=np.int32),(2,256)).copy(),
        sample_assignments=np.zeros((2,2318),np.int32),hand_faces_left=np.zeros((4603,3),np.int32),
        hand_faces_right=np.zeros((4603,3),np.int32),mhr_model_sha256=np.array(p.HAND_MODEL))
    sha=p.array_sha(f.astype(np.int32));keys=('vertex_indices','sample_local_indices','sample_assignments','hand_faces_left','hand_faces_right')
    topology=p.array_sha(f.astype(np.int32),*(spec[k] for k in keys))
    spec['faces_sha256']=np.array(sha);spec['topology_sha256']=np.array(topology)
    monkeypatch.setattr(p,'HAND_FACES',sha);monkeypatch.setattr(p,'HAND_TOPOLOGY',topology)
    return spec,f


def test_official_typed_topology_hash_normalizes_native_faces_int32(monkeypatch):
    s,f=spec_fixture(monkeypatch);np.testing.assert_array_equal(p.hand_indices(s,f,4636),s['vertex_indices'])
    assert p.array_sha(f)!=p.array_sha(f.astype(np.int32))


@pytest.mark.parametrize('kind',['faces','indices','dtype','model','samples'])
def test_hand_anatomy_cannot_be_swapped(monkeypatch,kind):
    s,f=spec_fixture(monkeypatch)
    if kind=='faces':f[0,0]=3
    elif kind=='indices':s['vertex_indices'][0,0]=1
    elif kind=='dtype':s['vertex_indices']=s['vertex_indices'].astype(np.int64)
    elif kind=='model':s['mhr_model_sha256']=np.array('f'*64)
    else:s['sample_local_indices'][0,0]=1
    with pytest.raises(ValueError):p.hand_indices(s,f,4636)


def test_external_static_moving_and_absent_contact_qualification_keeps_motion():
    cfg,_=p.old.profile_config(Path('.'),'v2')
    for seed,reserved in ((20261031,False),(20261101,True)):
        row=p.manufactured_gate(seed,reserved,cfg);assert row['passed']
        assert [c['case'] for c in row['cases']]==['static','moving','absent_contacts']
        assert all(c['converged'] for c in row['cases']) and row['cases'][2]['active_hand_frames']==0
        assert .8<=row['cases'][1]['full_motion']['full_path_retention']<=1.2
        json.dumps(row,allow_nan=False)


def test_no_truth_or_depth_in_manufactured_optimizer_interface(monkeypatch):
    calls=[]
    def fit(vertices,points,xy,visible,r,t,observed,K,indices,fps,cfg,**kw):
        assert set(kw)=={'contact_evidence','contact_config'} and kw['contact_config'] is p.CONTACT_CFG
        assert kw['contact_evidence'].hand_points_camera.shape==(24,2,1,3)
        assert observed.all() and np.array_equal(indices,np.arange(24));calls.append(True)
        return SimpleNamespace(rotations=r,translations=t,diagnostics=dict(converged=False))
    monkeypatch.setattr(p,'refine_sequence',fit);cfg,_=p.old.profile_config(Path('.'),'v2')
    assert not p.manufactured_gate(20261101,True,cfg)['passed'] and len(calls)==3


def test_frozen_protocol_and_isolation_before_challenge_reads():
    assert p.CONTACT_CFG.contact_sigma_diameter==.02 and p.CONTACT_CFG.max_points_per_hand==1
    src=Path(p.__file__).read_text();assert src.index('manufactured_gate(seed,reserved,config)')<src.index('old.saved_frontend(ledger)')
    assert 'depth_used=False' in src and 'production_adopted=False' in src and 'native_contact_parity_claimed=False' in src
    assert 'tracks_depth_m=' not in src
    sh=Path('infra/run_sequence_pose_contact_probe.sh').read_text()
    assert '--gpus' not in sh and '--network none' in sh and '--read-only' in sh and 'CUDA_VISIBLE_DEVICES=-1' in sh
    assert 'sequence_pose_contact.owner' in sh and 'docker rm -f "$cid"' in sh and '1200s docker run' in sh
    assert 'mhr_hand_surface_spec.npz,readonly' in sh and 'sequence-pose-contact-$REV' in sh


def forward_fixture(tmp_path,monkeypatch,kind):
    experiment=tmp_path/'experiment';base=experiment/'outputs/episode_000009';forward=base/'cari_shared_forward_v1'
    forward.mkdir(parents=True);(experiment/'pins').mkdir()
    spec=dict(episode_index=9,total_frames=3,camera_name='front_stereo_camera_left',width=8,height=8)
    path=forward/'coconet.pth';path.write_bytes(b'owned tiny fake bundle no pickle')
    pin=lambda q:dict(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())
    bp=pin(path);monkeypatch.setattr(p,'FORWARD_PIN',bp)
    report=dict(stage='world_reward_native_cari_shared_full_video_forward',status='pass',producer_revision=p.old.SOURCE,
        clip_spec=spec,input_track='track_1',input_sha256='a'*64,frames=3,ground_truth_used=False,private_truth_read=False,
        hand_labeled_test=False,oracle_modes=[],original_frame_indices=list(range(3)),bundle_sha256=bp['sha256'],bundle_bytes=bp['bytes'])
    if kind=='gt':report['ground_truth_used']=True
    elif kind=='oracle':report['oracle_modes']=['truth']
    elif kind=='frames':report['original_frame_indices']=[0,2,3]
    rp=forward/'report.json';rp.write_text(json.dumps(report));rpin=pin(rp);monkeypatch.setattr(p,'FORWARD_REPORT_PIN',rpin)
    pins=dict(schema='world-reward-cari-shared-forward-pins-v1',clip_spec=spec,
        forward_files={'coconet.pth':bp,'report.json':rpin},forward=dict(producer_revision=p.old.SOURCE))
    if kind=='producer':pins['forward']['producer_revision']='f'*40
    pp=experiment/'pins/cari_clip_000009_shared_forward_pins.json';pp.write_text(json.dumps(pins))
    monkeypatch.setattr(p,'FORWARD_PINS_PIN',pin(pp))
    if kind=='bundle_pin':path.write_bytes(b'changed')
    elif kind=='report_pin':rp.write_text(json.dumps(dict(report,status='fail')))
    return experiment,base,spec,path


@pytest.mark.parametrize('kind',[None,'gt','oracle','frames','producer','bundle_pin','report_pin'])
def test_authenticated_forward_refuses_bad_lineage_before_pickle_load(tmp_path,monkeypatch,kind):
    experiment,base,spec,path=forward_fixture(tmp_path,monkeypatch,kind);calls=[]
    def load(q,**kw):
        assert q==path and kw==dict(map_location='cpu',weights_only=False);calls.append(q)
        return bundle()
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(load=load));ledger=p.old.ArtifactLedger()
    if kind is None:
        out=p.authenticate_forward(ledger,experiment,base,spec,'a'*64);assert out.shape==(3,2) and len(calls)==1;ledger.verify()
    else:
        with pytest.raises(ValueError):p.authenticate_forward(ledger,experiment,base,spec,'a'*64)
        assert not calls


def diagnostic_fixture():
    return dict(RGB_mean_px=2.,selected_anatomical_triangle_mean_m=.02,
        acceleration_proxy_m_s2_median=1.,acceleration_proxy_m_s2_p95=2.,
        angular_acceleration_proxy_rad_s2_median=3.,angular_acceleration_proxy_rad_s2_p95=4.)


def test_rgb_improvement_cannot_hide_worse_translation_acceleration():
    original=diagnostic_fixture();candidate=dict(original,RGB_mean_px=1.,acceleration_proxy_m_s2_p95=2.01)
    q=p.diagnostic_gates(original,candidate)
    assert q['gates']['RGB_reprojection_improves'] and not q['gates']['acceleration_proxy_m_s2_p95'] and not q['passed']
    assert q['production_adopted'] is False


def test_no_active_contact_cannot_be_claimed_nonworse_contact_success():
    original=diagnostic_fixture();original['selected_anatomical_triangle_mean_m']=None
    candidate=dict(original,RGB_mean_px=1.)
    q=p.diagnostic_gates(original,candidate);assert not q['gates']['active_anatomical_contact_nonworse'] and not q['passed']


def test_all_proxy_gates_still_never_claim_heldout_quality():
    original=diagnostic_fixture();candidate={key:value*.9 for key,value in original.items()}
    q=p.diagnostic_gates(original,candidate)
    assert q['passed'] and q['scope']=='predicted_observation_QA_not_accuracy' and not q['production_adopted']

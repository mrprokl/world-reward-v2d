"""Tiny frozen-receipt and metric-array checks; no real GLB, CPU binary or GPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def qualified(tmp_path,monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra'
    monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('wr_volume_loader_test',infra/'volume_geometry_loader.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=tmp_path/'outputs/episode_000000/object_budget_volume';base.mkdir(parents=True)
    v=np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    scale=2.5
    pv=np.concatenate([v,np.repeat(v[:1],4092,axis=0)])*scale
    pf=np.concatenate([f,np.zeros((4092,3),dtype=int)])
    geometry=base/'geometry.npz'
    np.savez_compressed(geometry,vertices=pv,faces=pf,episode_index=np.array(0),object_scale=np.array(1.),grounded_scale_baked=np.array(scale))
    glb=base/'object_fixed_canonical.glb';glb.write_bytes(b'tiny fictional canonical artifact')
    monkeypatch.setattr(module,'_load_mesh',lambda path:(v.copy(),f.copy()))
    image='sha256:'+'b'*64
    build_path=tmp_path/'results/image-volume-qem.json';build_path.parent.mkdir()
    build_path.write_text(json.dumps({'stage':'world_reward_volume_qem_build','status':'pass','image_id':image}))
    evidence={'build_report_sha256':module.sha256(build_path),'binary_sha256':'a'*64,'build_info':{'source_sha256':'c'*64}}
    control_path=tmp_path/'validation/volume_qem_v1/report.json';control_path.parent.mkdir(parents=True)
    control_path.write_text(json.dumps({'stage':'own_volume_constrained_intersection_qem_geometry','status':'pass',
        'script_sha256':module.CONTROL_SHA,'build':evidence}))
    metrics={'birthface_matched_shells':[{'relative_volume_error':.001}],
             'sampled_bidirectional_chamfer_diagonal_ratio':.001,'net_volume_relative_error':.001}
    report={'stage':'world_reward_cpu_volume_constrained_object_mesh','status':'pass','episode_index':0,
        'producer_revision':module.PRODUCER_REVISION,'script_sha256':module.PRODUCER_SHA,'input_track':'track_1',
        'input_sha256':'1'*64,'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
        'target_faces':4096,'target_vertices':4096,'components_deleted':False,'holes_filled':False,'normals_repaired':False,
        'frame_poses_changed':False,'source_shell_volume_relative_limit':.05,'native_cost_and_placement_unchanged':True,
        'independent_candidate_intersecting_faces':0,'packed_intersecting_faces':0,'metric_scale_baked_once':scale,
        'source_hashes':{'object_report':'2'*64,'alignment_report':'3'*64},'image_id':image,
        'control_evidence':{'volume_gate_sha256':module.sha256(control_path),**evidence},
        'geometry_sha256':module.sha256(geometry),'canonical_glb_sha256':module.sha256(glb),
        'packed_topology':module.mesh_topology(v,f),'candidate_geometry':metrics,'export_geometry':metrics,
        'source_arrays_unchanged':True,'official_pack_fidelity':{'oriented_triangles_exact':True,'official_helper_simplification_invoked':False}}
    path=base/'report.json';path.write_text(json.dumps(report))
    return module,tmp_path,base,report,pv,pf


def call(fixture,scale=2.5):
    module,root,*_=fixture
    return module.load(root,0,'1'*64,'2'*64,'3'*64,scale)


def test_exact_metric_arrays_without_second_scale_or_binary(qualified):
    *_,pv,pf=qualified
    v,f,active,cleanup,glb,receipt=call(qualified)
    assert np.array_equal(v,pv) and np.array_equal(f,pf)
    assert active.tolist()==[0,1,2,3] and cleanup['excluded_faces']==4092
    assert receipt['resimplification_performed'] is False
    assert receipt['independent_embedding_reverified_here'] is False
    assert receipt['metric_oriented_triangle_fidelity']['oriented_triangles_exact'] is True


@pytest.mark.parametrize('field,bad',[
    ('producer_revision','f'*40),('script_sha256','f'*64),('episode_index',False),
    ('ground_truth_used',0),('oracle_modes',['GT']),('packed_intersecting_faces',1),
    ('geometry_sha256','f'*64),('canonical_glb_sha256','f'*64),('source_arrays_unchanged',False)])
def test_frozen_producer_rejects_changed_provenance(qualified,field,bad):
    _,_,base,report,*_=qualified
    report[field]=bad;(base/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):call(qualified)


@pytest.mark.parametrize('kind',['control','build','alignment','volume','packed_topology','double_scale','symlink'])
def test_ancestry_geometry_and_already_baked_scale_fail_closed(qualified,kind):
    module,root,base,report,*_=qualified
    if kind in ('control','build'):
        path=root/('validation/volume_qem_v1/report.json' if kind=='control' else 'results/image-volume-qem.json')
        path.write_text(path.read_text()+' ')
    elif kind=='alignment':report['source_hashes']['alignment_report']='4'*64
    elif kind=='volume':report['candidate_geometry']['birthface_matched_shells'][0]['relative_volume_error']=.051
    elif kind=='packed_topology':report['packed_topology']['components'][0]['volume_sign']=-1
    elif kind=='symlink':
        path=base/'geometry.npz';other=base/'other.npz';path.rename(other);path.symlink_to(other)
    (base/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):call(qualified,scale=3. if kind=='double_scale' else 2.5)


def test_exact_triangle_check_detects_metric_change_even_with_new_npz_hash(qualified):
    module,_,base,report,pv,pf=qualified
    pv=pv.copy();pv[0,0]+=1e-9
    path=base/'geometry.npz'
    np.savez_compressed(path,vertices=pv,faces=pf,episode_index=np.array(0),object_scale=np.array(1.),grounded_scale_baked=np.array(2.5))
    report['geometry_sha256']=module.sha256(path);(base/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='oriented triangle'):call(qualified)


def test_float32_producer_scale_arithmetic_matches_exactly(qualified,monkeypatch):
    module,_,base,report,pv,pf=qualified
    canonical=(pv[:4]/2.5).astype(np.float32)*np.float32(.173)
    monkeypatch.setattr(module,'_load_mesh',lambda path:(canonical.astype(np.float64),pf[:4].copy()))
    packed=np.concatenate([canonical,np.repeat(canonical[:1],4092,axis=0)])*2.5
    path=base/'geometry.npz'
    np.savez_compressed(path,vertices=packed,faces=pf,episode_index=np.array(0),object_scale=np.array(1.),grounded_scale_baked=np.array(2.5))
    report['geometry_sha256']=module.sha256(path);(base/'report.json').write_text(json.dumps(report))
    assert np.array_equal(call(qualified)[0],packed)


def test_legacy_pins_none_is_episode0_only_no_global_const_monkeypatch(qualified):
    module,root,*_=qualified
    with pytest.raises(ValueError,match='episode0 only'):
        module.load(root,2,'1'*64,'2'*64,'3'*64,2.5)


def test_generic_branch_calls_same_numeric_fidelity_and_posthash_gate(qualified,monkeypatch):
    module,root,base,report,pv,pf=qualified
    import volume_mesh_pin_inventory as inventory
    pins={'report':{'producer_revision':'a'*40,'script_sha256':'b'*64},'files':{'tiny/pin':{'sha256':'c'*64,'bytes':1}}}
    report.update(producer_revision='a'*40,script_sha256='b'*64)
    (base/'report.json').write_text(json.dumps(report))
    observed=pins['files'];events=[]
    monkeypatch.setattr(inventory,'verify_pinned_artifacts',lambda *args:events.append('pins')or(report,observed))
    monkeypatch.setattr(inventory,'identity',lambda path:observed['tiny/pin'])
    v,f,*_,receipt=module.load(root,0,'1'*64,'2'*64,'3'*64,2.5,pins=pins)
    assert np.array_equal(v,pv)and np.array_equal(f,pf)and events==['pins']
    assert receipt['generic_artifact_pins_verified']and receipt['cpu_producer_revision']=='a'*40
    monkeypatch.setattr(inventory,'identity',lambda path:{'sha256':'d'*64,'bytes':1})
    with pytest.raises(ValueError,match='changed during numerical'):
        module.load(root,0,'1'*64,'2'*64,'3'*64,2.5,pins=pins)


def test_generic_branch_rejects_npz_extra_payload_even_if_report_pinned(qualified,monkeypatch):
    module,root,base,report,pv,pf=qualified
    import volume_mesh_pin_inventory as inventory
    pins={'report':{'producer_revision':'a'*40,'script_sha256':'b'*64},'files':{}}
    report.update(producer_revision='a'*40,script_sha256='b'*64)
    np.savez_compressed(base/'geometry.npz',vertices=pv,faces=pf,episode_index=np.array(0),object_scale=np.array(1.),
        grounded_scale_baked=np.array(2.5),unexpected_private=np.array([0]))
    report['geometry_sha256']=module.sha256(base/'geometry.npz');(base/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(inventory,'verify_pinned_artifacts',lambda *args:(report,{}))
    with pytest.raises(ValueError,match='Exact original CPU proposal NPZ'):
        module.load(root,0,'1'*64,'2'*64,'3'*64,2.5,pins=pins)

"""Manufactured receipts only: these fixtures assert no native qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
import solid_geometry_loader as loader
from world_reward.mesh_conditioning_v2 import POLICY_SHA256


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): path.chmod(0o644)
    path.write_bytes(data); path.chmod(0o444)
    return loader.identity(path)


def seal(path,row): return write(path,json.dumps(row,sort_keys=True).encode())


def topology(volume=1.):
    return dict(vertices=4,active_vertices=4,faces=4,closed_oriented_vertex_manifold=True,diagonal=2.,
        components=[dict(euler=2,volume_sign=1,signed_volume=volume,vertices=4,faces=4)])


def query(query_sha,volume=1.):
    c=dict(schema='world_reward.certified_solid_query.v1',status='pass',source_sha256=query_sha,cgal_version='6.0.1',
        vertices=4,faces=4,component_count=1,
        represented_coordinates='EPECK exact values of original parsed IEEE754 binary64; no perturbation',
        **{k:True for k in loader.TRUE},**{k:False for k in loader.FALSE},inside=[[False]],
        components=[dict(original_component_id=0,original_vertices=4,original_faces=4,witness_original_vertex=0,
            exact_volume_sign=1,diagnostic_signed_volume=volume)])
    return dict(query_attempted=True,query_returned=True,query_returncode=0,query_input=dict(bytes=1,sha256='e'*64),
        native_certificate=c,forest=dict(component_keys=['source-loaded-first-face-0'],signs=[1],inside=[[False]],parents=[-1],depths=[0]),
        topology=topology(volume),stored_array_sha256=['f'*64,'e'*64])


def measured(query_sha,source=None):
    row=query(query_sha); source=source or row
    row['candidate_to_source_components']=[0]
    row['fidelity']=dict(sampled_bidirectional_chamfer_diagonal_ratio=.001,net_volume_relative_error=0.,scale_or_pose_fitted=False,
        source_topology=deepcopy(source['topology']),candidate_topology=deepcopy(row['topology']),
        birthface_matched_shells=[dict(source_component=0,euler=2,volume_sign=1,source_volume=1.,candidate_volume=1.,relative_volume_error=0.)])
    return row


def compiler(code,sha,scale=.375):
    source=query(sha); source.update(conditioning=dict(roundtrip_numerically_exact=True,source_geometry_repaired=False,
        source_arrays_modified=False,policy_sha256=POLICY_SHA256),float32_orientation=dict(exact_positive_normal_dot=True,area_tolerance_used=False),
        conditioning_header=loader.identity(code/'infra/mesh_conditioned_chart_v2.hpp'),float32_certificate=query(sha))
    source['float32_certificate']['candidate_to_source_components']=[0]
    metric=query(sha); metric['candidate_to_source_components']=[0]
    stages={loader.STAGES[0]:source,**{n:measured(sha,metric if n==loader.STAGES[-1] else source) for n in loader.STAGES[1:]}}
    return dict(stage='oriented_solid_compiler_v2',conditioning_version=2,status='pass',phase='complete',native_attempts=1,native_returned=True,
        native_returncode=0,source_arrays_unchanged=True,output_vertices=4096,output_faces=4096,metric_scale_baked_once=scale,
        **{k:False for k in ('geometry_repaired','components_deleted','orientation_changed','cost_backend_changed','ground_truth_used','adoption',
            'reconstruction_accuracy_verified','frame_poses_changed','metric_scale_accuracy_verified')},
        artifacts_before={},artifacts_after={},stages=stages,metric_source_certificate=metric,
        native_mapping=dict(conditioning=dict(chart_version=2,source_roundtrip_numerically_exact=True,chart_refitted=False,adopted=False,
            header_sha256=source['conditioning_header']['sha256'],policy_sha256=POLICY_SHA256),
            serialization=dict(committed_collapses=1,serialization_safe=True,serialization_vetoes=2),
            native_volume=dict(committed_collapses=1,native_cost_and_placement_unchanged=False,cost_normalization=True,final_shell_volumes_verified=True,volume_relative_limit=.05)),
        official_pack_fidelity=dict(oriented_triangles_exact=True,official_helper_simplification_invoked=False,nonexact_merge_or_face_deletion=False))


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'root'; code=tmp_path/'code'; monkeypatch.setattr(loader,'CODE',code)
    import world_reward.oriented_solid_forest as forest
    import world_reward.mesh_conditioning_v2 as chart
    monkeypatch.setattr(forest,'__file__',str(code/'src/world_reward/oriented_solid_forest.py'))
    monkeypatch.setattr(chart,'__file__',str(code/'src/world_reward/mesh_conditioning_v2.py'))
    for name in loader.SOURCE_HELPERS: write(code/name,(name+' synthetic source').encode())
    source_helpers={n:loader.identity(code/n) for n in loader.SOURCE_HELPERS}
    qrev='b'*40; rev='a'*40; querysha='c'*64; scale=.375; names=loader.paths(8,qrev,rev)
    obj=dict(stage='sam3d_objects_grounded_fixed_frame',status='pass',episode_index=8,input_track='track_1',input_sha256='d'*64,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],frame_index=0,scale_source='already_human_anchored_MoGe2_no_second_scalar',
        pointmap_grounding={},transform={'scale':[scale]*3})
    align=dict(stage='predicted_human_anchored_moge2_pointmaps',status='pass',episode_index=8,input_track='track_1',input_sha256='d'*64,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],coordinate_frame='OpenCV_x_right_y_down_z_forward',
        pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied')
    apin=seal(root/names['alignment'],align); obj['pointmap_grounding']['alignment_report_sha256']=apin['sha256']
    for role,field in (('source_glb','object_sha256'),('transform','transform_sha256'),('intrinsics','intrinsics_sha256')):
        raw=json.dumps(obj['transform']).encode() if role=='transform' else b'opaque original bytes'
        obj[field]=write(root/names[role],raw)['sha256']
    opin=seal(root/names['object'],obj)
    out={Path(names[k]).name:write(root/names[k],b'opaque output only no actual geometry') for k in ('geometry','glb')}
    c=compiler(code,querysha); c['glb_identity']=out['object_fixed_canonical.glb']
    b=dict(schema='world_reward.solid_chart_v2_build_pins.v1',image_id='sha256:'+'e'*64,**{k:dict(bytes=1,sha256='f'*64) for k in ('report','native','binary')})
    g=dict(schema='world_reward.certified_solid_qualification_pins.v1',qualified_controls=15,child_image_id=b['image_id'],native_source=dict(bytes=1,sha256=querysha),
        **{k:dict(bytes=1,sha256='1'*64) for k in ('report','native','binary')})
    bp=seal(code/loader.BUILD,b); seal(code/loader.CGAL,g)
    ledger=dict(producer_revision=qrev,source_files=210,source_files_sha256='2'*64,source_readonly_ledger_sha256='3'*64)
    build=dict(pins_identity=bp,built_artifacts={k:b[k] for k in ('report','native','binary')},cgal={k:g[k] for k in ('report','native','binary')})
    common=dict(status='pass',phase='complete',qualified_procedural_controls=4,source_rehashed_after=True,artifacts_rehashed_after=True,official_rehashed_after=True,
        **{k:False for k in ('gpu_used','gt_used','adoption','reconstruction_accuracy_verified','competition_eligibility_verified')},
        source_binding=ledger,source_binding_after=ledger,qualified_build=build)
    qnative=dict(common,stage='solid_chart_v2_qualification_native_v1',source_arrays_unchanged=True,image_id=b['image_id'])
    qhost=dict(common,stage='solid_chart_v2_qualification_host_v1',owned_container_removed=True,owned_scratch_removed=True,native=qnative)
    q=dict(schema='world_reward.solid_chart_v2_qualification_pins.v1',**ledger,qualified_controls=4,native_qem_calls=4,native_query_calls=40,adoption=False,reconstruction_accuracy_verified=False,
        report=seal(root/names['qualification_host'],qhost),native=seal(root/names['qualification_native'],qnative),build_pins=bp)
    qp=seal(code/loader.QUALIFICATION,q); proof=dict(pins_identity=qp,report=q['report'],native=q['native'],source=ledger,build=build)
    sb=dict(producer_revision=rev,source_files=300,source_files_sha256='4'*64,source_readonly_ledger_sha256='5'*64,
        helpers={loader.PROCESSOR:dict(bytes=12,sha256='6'*64)},markers=dict(revision=dict(bytes=41,sha256=hashlib.sha256((rev+'\n').encode()).hexdigest()),
        **{'source-sha256':dict(bytes=65,sha256='7'*64)}))
    sources=dict(video='d'*64,object_report=opin['sha256'],alignment_report=apin['sha256'],**{n:obj[f] for n,f in
        (('object.glb','object_sha256'),('transform.json','transform_sha256'),('intrinsics.json','intrinsics_sha256'))})
    inputs=dict(video_sha256='d'*64)
    native=dict(stage='world_reward_object_budget_solid_native_v1',status='pass',phase='complete',episode_index=8,input_track='track_1',input_sha256='d'*64,
        source_binding=sb,source_binding_after=sb,input_binding=inputs,qualification=proof,image_id=b['image_id'],source_hashes=sources,compiler=c,outputs=out,
        native_query_calls=8,budget_seconds=1800,qem_seconds=1200,query_seconds=180,maximum_qem_calls=1,maximum_query_calls=8,object_scale=1.,original_grounded_scale=scale,metric_scale_baked_once=scale,
        elapsed_seconds=1.,oracle_modes=[],**{k:False for k in ('gpu_used','ground_truth_used','hand_labeled_test','adoption','reconstruction_accuracy_verified',
            'competition_eligibility_verified','media_decoded','frame_poses_changed')},**{k:True for k in ('input_video_hashed','source_rehashed_after','inputs_qualification_rehashed_after','runtime_rehashed_after')})
    native['source_geometry']=dict(source_identity=loader.identity(root/names['source_glb']),raw_oriented_triangles_sha256='8'*64,
        exact_welding=dict(faces_preserved=4,welded_vertices=4,face_order_and_coordinates_exact=True,faces_removed=0,position_merging='exact equality only',source_modified=False))
    host=dict(stage='world_reward_object_budget_solid_host_v1',status='pass',phase='complete',episode_index=8,producer_revision=rev,source_binding=sb,source_binding_after=sb,
        native=native,input_binding=inputs,qualification=proof,image_identity={'child_id':b['image_id']},outputs=out,elapsed_seconds=2.,
        **{k:False for k in ('gpu_used','ground_truth_used','adoption','reconstruction_accuracy_verified','competition_eligibility_verified','media_decoded')},
        **{k:True for k in ('owned_container_removed','owned_scratch_removed','source_rehashed_after','inputs_qualification_rehashed_after')})
    seal(root/names['native'],native); hp=seal(root/names['report'],host); (root/names['report']).parent.chmod(0o555)
    pins=dict(schema=loader.SCHEMA,episode_index=8,input_sha256='d'*64,metric_scale_baked_once=scale,report=dict(hp,producer_revision=rev,script_sha256='6'*64),
        files={n:loader.identity(root/n) for n in names.values()},source_helpers=source_helpers)
    return root,code,pins,names,host


def refresh(root,pins,names,host):
    if (root/names['report']).parent.exists(): (root/names['report']).parent.chmod(0o755)
    seal(root/names['native'],host['native']); hp=seal(root/names['report'],host)
    pins['report'].update(hp); pins['files'].update({n:loader.identity(root/n) for n in (names['report'],names['native'])})
    (root/names['report']).parent.chmod(0o555)


def test_proposal_namespace_binds_actual_producer_and_preserves_failed_predecessor(tmp_path,monkeypatch):
    root,code,pins,names,host=fixture(tmp_path,monkeypatch)
    old=root/'outputs/episode_000008/object_budget_solid/report.json'
    old_pin=write(old,b'original failed producer receipt');old.parent.chmod(0o555)
    loader.verify_pinned_artifacts(root,pins,8,'d'*64,
        pins['files'][names['object']]['sha256'],pins['files'][names['alignment']]['sha256'],.375)
    assert loader.identity(old)==old_pin and old.parent.stat().st_mode&0o777==0o555
    assert all('/object_budget_solid_'+pins['report']['producer_revision']+'/' in names[k]
        for k in ('report','native','geometry','glb'))
    with pytest.raises(ValueError):loader.paths(8,'b'*40,'../previous')


def verify(data):
    root,_,pins,names,_=data
    return loader.verify_pinned_artifacts(root,pins,8,'d'*64,pins['files'][names['object']]['sha256'],pins['files'][names['alignment']]['sha256'],.375)


def test_full_inert_receipt_and_all_eight_queries_are_bound(tmp_path,monkeypatch):
    data=fixture(tmp_path,monkeypatch); host,native,_,_=verify(data)
    assert host['status']==native['status']=='pass' and len(data[2]['files'])==11
    assert data[4]['source_binding']['source_files']>len(data[4]['source_binding']['helpers'])


@pytest.mark.parametrize('fault',['missing','byte','script','source_helper','gt','qualification','query','empty_stage','count_bool','inside_bool','parent','birth','volume','euler','header','extra_query','scale','serialization'])
def test_forged_or_changed_qualification_stages_fail_before_any_geometry_decode(tmp_path,monkeypatch,fault):
    data=fixture(tmp_path,monkeypatch); root,code,pins,names,host=data; c=host['native']['compiler']; row=c['stages']['native_candidate']
    if fault=='missing': pins['files'].pop(names['glb'])
    elif fault=='byte': write(root/names['geometry'],b'changed')
    elif fault=='script': pins['report']['script_sha256']='0'*64
    elif fault=='source_helper': write(code/loader.SOURCE_HELPERS[0],b'changed source')
    elif fault=='gt': host['native']['ground_truth_used']=True
    elif fault=='qualification': host['qualification']['report']['sha256']='0'*64
    elif fault=='query': row['native_certificate']['source_sha256']='0'*64
    elif fault=='empty_stage': c['stages']['native_candidate']={}
    elif fault=='count_bool': row['native_certificate']['component_count']=True
    elif fault=='inside_bool': row['forest']['inside']=[[0]]
    elif fault=='parent': row['forest']['parents']=[0]
    elif fault=='birth': row['candidate_to_source_components']=[True]
    elif fault=='volume': row['fidelity']['birthface_matched_shells'][0]['candidate_volume']=.9
    elif fault=='euler': row['fidelity']['birthface_matched_shells'][0]['euler']=0
    elif fault=='header': c['native_mapping']['conditioning']['header_sha256']='0'*64
    elif fault=='extra_query': c['extra_query']=query('c'*64)
    elif fault=='scale': host['native']['metric_scale_baked_once']=.2
    elif fault=='serialization': c['native_mapping']['serialization']['serialization_safe']=False
    if fault not in ('missing','byte','script','source_helper'): refresh(root,pins,names,host)
    with pytest.raises((ValueError,KeyError,FileNotFoundError,TypeError)): verify(data)


def test_json_duplicates_overflow_and_identity_symlinks_are_rejected(tmp_path):
    for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
        with pytest.raises(ValueError): loader.strict_json(raw)
    original=tmp_path/'original'; write(original,b'bytes'); (tmp_path/'link').symlink_to(original)
    with pytest.raises(ValueError): loader.identity(tmp_path/'link')


def test_loader_source_never_imports_or_executes_native_producers():
    import ast
    tree=ast.parse(Path(loader.__file__).read_text()); forbidden={'object_budget_solid','oriented_solid_compiler','certified_solid_source','solid_chart_v2_qualify','certified_solid_build','subprocess'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): assert not forbidden.intersection(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom): assert node.module not in forbidden
    assert 'build-info' not in '\n'.join(Path(loader.__file__).read_text().splitlines()[7:])


def test_legacy_volume_ids_are_not_reinterpreted_as_native_first_face_ids():
    source=query('c'*64); row=query('c'*64)
    for record in (source,row):
        native=record['native_certificate']; native.update(component_count=2,vertices=8,faces=8,inside=[[False,True],[False,False]],
            components=[dict(original_component_id=i,original_vertices=4,original_faces=4,witness_original_vertex=4*i,exact_volume_sign=s,
                diagnostic_signed_volume=v) for i,s,v in ((0,-1,-.125),(1,1,1.))])
        record['forest']=dict(component_keys=['source-loaded-first-face-0','source-loaded-first-face-4'],signs=[-1,1],
            inside=[[False,True],[False,False]],parents=[1,-1],depths=[1,0])
        record['topology']=dict(vertices=8,active_vertices=8,faces=8,closed_oriented_vertex_manifold=True,diagonal=2.,
            components=[dict(euler=2,volume_sign=1,signed_volume=1.,vertices=4,faces=4),
                        dict(euler=2,volume_sign=-1,signed_volume=-.125,vertices=4,faces=4)])
    row['candidate_to_source_components']=[0,1]
    row['fidelity']=dict(scale_or_pose_fitted=False,sampled_bidirectional_chamfer_diagonal_ratio=.001,net_volume_relative_error=0.,
        source_topology=source['topology'],candidate_topology=row['topology'],birthface_matched_shells=[
        dict(source_component=0,euler=2,volume_sign=1,source_volume=1.,candidate_volume=1.,relative_volume_error=0.),
        dict(source_component=1,euler=2,volume_sign=-1,source_volume=-.125,candidate_volume=-.125,relative_volume_error=0.)])
    loader._match(row,source,'c'*64,measured=True)
    row['candidate_to_source_components']=[1,0]
    with pytest.raises(ValueError,match='forest'): loader._match(row,source,'c'*64,measured=True)


@pytest.mark.parametrize('fault',['none','second_scale','face_flip','nonzero_padding','source_mutation','extra_npz','wrong_scalar'])
def test_real_pure_payload_math_keeps_scale_once_and_every_meaningful_triangle(tmp_path,monkeypatch,fault):
    data=fixture(tmp_path,monkeypatch); root,code,pins,names,host=data
    import exact_mesh_geometry as geometry
    import object_budget_endpoint as endpoint
    v=np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int64); scale=.375
    pv=np.r_[v,np.repeat(v[:1],4092,axis=0)]*scale; pf=np.r_[f,np.zeros((4092,3),np.int64)]
    compact,compactf=endpoint.exact_weld(v*scale,f)[:2]
    final=host['native']['compiler']['stages'][loader.STAGES[-1]]
    final['topology']=final['fidelity']['candidate_topology']=geometry.exact_mesh_topology(compact,compactf)
    def digest(a): return hashlib.sha256(json.dumps(dict(dtype=a.dtype.str,shape=a.shape),sort_keys=True).encode()+b'\0'+a.tobytes()).hexdigest()
    final['stored_array_sha256']=[digest(compact),digest(compactf)]
    if fault=='second_scale': pv*=scale
    elif fault=='face_flip': pf[0]=pf[0,::-1]
    elif fault=='nonzero_padding': pf[4]=[1,1,1]
    path=root/names['geometry']; path.chmod(0o644)
    arrays=dict(vertices=pv,faces=pf,episode_index=np.array(8,np.int64),object_scale=np.array(1.),grounded_scale_baked=np.array(scale))
    if fault=='extra_npz': arrays['unknown']=np.zeros(1)
    elif fault=='wrong_scalar': arrays['grounded_scale_baked']=np.array(scale*2)
    with path.open('wb') as stream: np.savez_compressed(stream,**arrays)
    path.chmod(0o444); pins['files'][names['geometry']]=loader.identity(path)
    monkeypatch.setattr(loader,'verify_pinned_artifacts',lambda *_:(host,host['native'],pins['files'],names))
    monkeypatch.setattr(geometry,'__file__',str(code/'infra/exact_mesh_geometry.py'))
    monkeypatch.setattr(endpoint,'__file__',str(code/'infra/object_budget_endpoint.py'))
    def raw(_):
        if fault=='source_mutation': write(code/loader.SOURCE_HELPERS[0],b'changed during payload')
        return v.copy(),f.copy()
    monkeypatch.setattr(endpoint,'_load_mesh',raw)
    if fault=='none':
        actual=loader.load(root,8,'d'*64,'e'*64,'f'*64,scale,pins=pins)
        np.testing.assert_array_equal(actual[0],pv); np.testing.assert_array_equal(actual[1],pf)
        assert actual[2].tolist()==[0,1,2,3] and actual[3]['meaningful_faces_removed']==0
        assert actual[5]['metric_scale_already_baked'] and actual[5]['independent_embedding_reverified_here'] is False
    else:
        with pytest.raises(ValueError): loader.load(root,8,'d'*64,'e'*64,'f'*64,scale,pins=pins)

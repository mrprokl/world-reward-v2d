"""Tiny consumer-seam controls; no native inference, renderer or challenge data."""
import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def consumers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'infra'))
    result=[]
    for name in ('object_pose_smoke','cari_prepare'):
        spec=importlib.util.spec_from_file_location('tiny_surface_'+name,ROOT/'infra'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result.append(module)
    return result


def test_explicit_surface_profile_defaults_preserved(consumers):
    pose,prepare=consumers
    assert pose._argument_parser().parse_args([]).mesh_source=='default'
    assert prepare._argument_parser().parse_args([]).mesh_source=='default'
    for module in consumers:
        for episode in range(30):
            args=module._argument_parser().parse_args(['--episode',str(episode),'--mesh-source','surface'])
            assert args.mesh_source=='surface' and args.episode==episode
        for mode in ('oracle','../surface','triangle_soup'):
            with pytest.raises(SystemExit):module._argument_parser().parse_args(['--mesh-source',mode])


def test_surface_removes_only_explicit_object_volume_gate_not_default():
    for name,object_name in (('object_pose_smoke.py','mesh'),('cari_prepare.py','metric_mesh')):
        tree=ast.parse((ROOT/'infra'/name).read_bytes())
        gates=[n for n in ast.walk(tree) if isinstance(n,ast.If) and
               f'{object_name}.is_watertight' in ast.unparse(n.test)]
        assert len(gates)==1
        gate=ast.unparse(gates[0].test)
        assert "args.mesh_source != 'surface'" in gate and '.is_winding_consistent' in gate and '.volume <= 0' in gate
        assert any(isinstance(n,ast.Raise) for n in ast.walk(gates[0]))


def test_surface_geometry_callbacks_are_explicit_no_solver():
    tree=ast.parse((ROOT/'infra/cari_prepare.py').read_bytes())
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('_surface_preflight','_surface_serialized_mesh'):
        source=ast.unparse(funcs[name])
        assert 'surface' in source and 'process=True' not in source
        for forbidden in ('compile_solid','exact_mesh_topology','fix_normals','simplify_quadric_decimation'):
            assert forbidden not in source
    source=ast.unparse(funcs['_surface_serialized_mesh'])
    assert '_solid_scene_matrix' in source and 'fix_rigid' in source
    assert 'raw_glb' in source and 'native_load' in source and 'serialized_mesh' in source


def test_same_represented_scene_projection_operator_is_shared(consumers):
    _,prepare=consumers
    trimesh=pytest.importorskip('trimesh',reason='Native Trimesh stays on Azure; no local installation')
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,.25]])
    f=np.array([[0,1,2]],np.int64)
    scene=trimesh.Scene(trimesh.Trimesh(v,f,process=False))
    raw,effective=prepare._solid_scene_matrix(scene,np.eye(4),trimesh,np)
    assert np.array_equal(raw,np.eye(4)) and np.array_equal(effective,np.eye(4))
    assert not scene.geometry[next(iter(scene.geometry))].is_watertight


def test_surface_prep_validates_same_original_frame_pose_and_saved_f32():
    text=(ROOT/'infra/cari_prepare.py').read_text()
    assert 'camera_roundtrip(' in text and "saved['obj_pose_world']" in text
    assert "metric_mesh=trimesh.Trimesh(*surface_compact,process=False)" in text
    assert "result['object_source']='surface'" in text
    assert 'source_rehashed_after=True' in text


def _saved_fixture():
    names=['000000','000001','000002']
    poses=np.repeat(np.eye(4)[None],len(names),axis=0)
    poses[:,0,3]=[0.,.01,.02]
    matrix=np.eye(4)
    sha='a'*64
    saved={'frames':names.copy(),'obj_pose_world':poses.astype(np.float32),
        'metadata':{'source':'World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose',
            'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
            'source_pose_sha256':sha,'mesh_frame_change':matrix.tolist()}}
    return saved,names,poses,sha,matrix


def test_actual_saved_surface_payload(consumers):
    _,prepare=consumers
    args=_saved_fixture()
    assert prepare._surface_saved_poses(*args,np) is args[0]['obj_pose_world']


@pytest.mark.parametrize('mutation',['drop_frame','drop_pose','float64','alter_pose',
    'wrong_sha','oracle','extra_field','missing_metadata','wrong_frame_change'])
def test_saved_surface_payload_fails_closed(consumers,mutation):
    _,prepare=consumers
    saved,names,poses,sha,matrix=_saved_fixture()
    if mutation=='drop_frame':saved['frames'].pop()
    elif mutation=='drop_pose':saved['obj_pose_world']=saved['obj_pose_world'][:-1]
    elif mutation=='float64':saved['obj_pose_world']=saved['obj_pose_world'].astype(np.float64)
    elif mutation=='alter_pose':saved['obj_pose_world'][1,0,3]+=.001
    elif mutation=='wrong_sha':saved['metadata']['source_pose_sha256']='b'*64
    elif mutation=='oracle':saved['metadata']['oracle_modes']=['GT']
    elif mutation=='extra_field':saved['other']=True
    elif mutation=='missing_metadata':del saved['metadata']
    else:saved['metadata']['mesh_frame_change'][0][3]=.01
    with pytest.raises(ValueError,match='Saved surface'):
        prepare._surface_saved_poses(saved,names,poses,sha,matrix,np)


def test_surface_preflight_checks_input_total_not_self_declared_payload_length():
    tree=ast.parse((ROOT/'infra/cari_prepare.py').read_text())
    preflight=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_surface_preflight')
    source=ast.unparse(preflight)
    assert source.count("inputs['total_frames']")==2
    assert 'SOURCE_HELPERS' in source


@pytest.mark.parametrize('mutation',[None,'hide_equal_first_orphan','wrong_faces','drop_final'])
def test_surface_preflight_uses_independent_canonical_counts(consumers,tmp_path,monkeypatch,mutation):
    """A legitimate orphan equal to v0 must never be mistaken for padding."""
    import json
    import surface_geometry_loader as loader
    _,prepare=consumers
    root=(tmp_path/'root').resolve();code=(tmp_path/'code').resolve()
    monkeypatch.setattr(prepare,'__file__',str(code/'infra/cari_prepare.py'))
    monkeypatch.setattr(loader,'SOURCE_HELPERS',())
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        path.chmod(0o444)
    for n in ('cari_prepare.py','cari_wrapper_common.sh','run_cari_prepare.sh'):
        write(code/'infra'/n,b'explicit mocked consumer source for unit test only')
    pinpath=code/'configs/surface_mesh_000026_pins.json';write(pinpath,{'scope':'unit_test'})
    base=root/'outputs/episode_000026'
    write(base/'object_grounded/transform.json',{'scale':[.5,.5,.5]})
    write(base/'object_grounded/report.json',{})
    write(base/'scale_smoke/report.json',{})
    canonical=base/'object_budget_surface'/'object_fixed_canonical.glb';write(canonical,b'fake canonical bytes, no GLB decoder invoked')
    parent=base/'object_pose_full_surface';parent.mkdir()
    write(parent/'object_fixed_canonical.glb',canonical.read_bytes())
    cv=np.array([[0.,0.,0.],[.5,0.,0.],[0.,.5,0.],[0.,0.,0.]])
    cf=np.array([[0,1,2]],np.int64)
    v=np.concatenate((cv,np.repeat(cv[:1],4092,axis=0)));f=np.concatenate((cf,np.zeros((4095,3),np.int64)))
    active=np.array([0],np.int64)
    count=2 if mutation=='drop_final' else 3
    posepath=parent/'geometry_and_poses.npz'
    with posepath.open('xb')as stream:
        np.savez_compressed(stream,vertices=v,faces=f,frame_index=np.arange(count,dtype=np.int64),
            rotation=np.repeat(np.eye(3)[None],count,axis=0),translation=np.zeros((count,3)),object_scale=np.array(1.))
    posepath.chmod(0o444)
    receipt=dict(committed_pins_sha256=loader.identity(pinpath)['sha256'],producer_report_sha256='b'*64,
        cpu_native_report_sha256='c'*64,source_domain='surface',metric_scale_baked_once=.5,
        geometry_operations_applied=False,canonical_vertices_count=4,canonical_faces_count=1)
    report=dict(mesh_source='surface',execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,
        object_report_sha256=prepare.sha256(base/'object_grounded/report.json'),
        geometry_and_poses_sha256=prepare.sha256(posepath),fixed_canonical_mesh_sha256=prepare.sha256(canonical),
        topology_budget=receipt.copy())
    if mutation=='hide_equal_first_orphan':report['topology_budget']['canonical_vertices_count']=3
    elif mutation=='wrong_faces':report['topology_budget']['canonical_faces_count']=2
    write(parent/'report.json',report);parent.chmod(0o555)
    monkeypatch.setattr(loader,'load',lambda *args,**kwargs:(v,f,active,{},canonical,receipt))
    args=(root,26,{'video_sha256':'a'*64,'total_frames':3},report,posepath,np)
    if mutation is not None:
        with pytest.raises(ValueError):prepare._surface_preflight(*args)
    else:
        result=prepare._surface_preflight(*args)
        assert len(result[5][0])==4 and np.array_equal(result[5][0][-1],cv[0])


def test_surface_serialized_caller_promotes_real_accessor_f32(consumers,monkeypatch,tmp_path):
    """Caller ABI only: explicit fake GLB/scene callbacks, not native evidence."""
    from types import SimpleNamespace
    import mesh_precision_diagnostic as precision
    import object_budget_endpoint as endpoint
    _,prepare=consumers
    v=np.array([[0.,0.,0.],[.25,0.,0.],[0.,.25,0.]],np.float32)
    f=np.array([[0,1,2]],np.int64)
    monkeypatch.setattr(precision,'raw_glb',lambda path:([(v,f)],v.astype(np.float64)[f],[{'node_transform_identity':True}]))
    monkeypatch.setattr(endpoint,'_load_mesh',lambda path:(v.astype(np.float64),f))
    monkeypatch.setattr(prepare,'_solid_scene_matrix',lambda *args:(np.eye(4),np.eye(4)))
    fake=SimpleNamespace(load=lambda *args,**kwargs:object(),
        transformations=SimpleNamespace(fix_rigid=lambda matrix,**kwargs:matrix.copy()))
    proof,native=prepare._surface_serialized_mesh(tmp_path/'fake.glb',(v.astype(np.float64),f),fake,np,lambda path:(v,f))
    assert proof['raw_world_exact'] and proof['native_fp32_copy_exact']
    assert np.array_equal(native[0],v)
    monkeypatch.setattr(precision,'raw_glb',lambda path:([(v.astype(np.float64),f)],v.astype(np.float64)[f],[{'node_transform_identity':True}]))
    with pytest.raises(ValueError,match='POSITION F32'):
        prepare._surface_serialized_mesh(tmp_path/'fake.glb',(v.astype(np.float64),f),fake,np,lambda path:(v,f))

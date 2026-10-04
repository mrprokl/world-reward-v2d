"""Tiny sealed lineage, manufactured geometry and mocked GLB; no assets/jobs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

INFRA = Path(__file__).resolve().parents[1]/'infra'


def store(path, data, mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(data); path.chmod(mode)
    return path


def array_hash(a):
    return hashlib.sha256(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode()
                          + b'\0' + np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA))
    spec = importlib.util.spec_from_file_location('wr_conditioned_loader_test', INFRA/'conditioned_geometry_loader.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    import exact_mesh_geometry as geometry
    import object_budget_endpoint as endpoint
    code, root = tmp_path/'code', tmp_path/'root'; rev, cache_rev = 'a'*40, 'b'*40
    names = loader.paths(9, cache_rev); base = root/'outputs/episode_000009/object_budget_conditioned'
    v = np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f = np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]], np.int64); scale = 2.5
    pv = np.r_[v,np.repeat(v[:1],4092,axis=0)]*scale; pf = np.r_[f,np.zeros((4092,3),np.int64)]
    base.mkdir(parents=True)
    np.savez_compressed(base/'geometry.npz', vertices=pv, faces=pf, episode_index=np.array(9,np.int64),
                        object_scale=np.array(1.), grounded_scale_baked=np.array(scale))
    (base/'geometry.npz').chmod(0o444)
    store(root/names['glb'], b'tiny fictional canonical GLB')
    store(root/names['source_glb'], b'tiny fictional source GLB', 0o644)
    transform = dict(scale=[scale]*3, rotation=[1.,0.,0.,0.], translation=[0.,0.,1.])
    store(root/names['transform'], json.dumps(transform).encode(), 0o644)
    store(root/names['intrinsics'], b'{}', 0o644)
    alignment = dict(stage='predicted_human_anchored_moge2_pointmaps', status='pass', episode_index=9, input_track='track_1',
        input_sha256='1'*64, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        coordinate_frame='OpenCV_x_right_y_down_z_forward', pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied')
    store(root/names['alignment'], json.dumps(alignment).encode(), 0o644)
    obj = dict(stage='sam3d_objects_grounded_fixed_frame', status='pass', episode_index=9, input_track='track_1',
        input_sha256='1'*64, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], frame_index=0,
        pointmap_grounding={'alignment_report_sha256':loader.identity(root/names['alignment'], readonly=False)['sha256']},
        scale_source='already_human_anchored_MoGe2_no_second_scalar', transform=transform,
        **{key:loader.identity(root/names[role], readonly=False)['sha256'] for key,role in
           (('object_sha256','source_glb'),('transform_sha256','transform'),('intrinsics_sha256','intrinsics'))})
    store(root/names['object'], json.dumps(obj).encode(), 0o644)
    cpp = {'bytes': 1, 'sha256': 'c'*64}; binary = {'bytes': 2, 'sha256': 'd'*64}; image = 'sha256:'+'e'*64
    store(root/names['build'], json.dumps(dict(status='pass', image_id=image)).encode(), 0o644)
    marker = lambda raw: dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    cache_binding = dict(producer_revision=cache_rev, source_files=176, source_files_sha256='f'*64,
        markers={'revision':marker((cache_rev+'\n').encode()),'source-sha256':marker(('9'*64+'\n').encode())},
        helpers={'infra/mesh_conditioned_qem.cpp':cpp})
    info = dict(source_sha256=cpp['sha256'], physical_coordinate_cache=True, physical_geometry_rescaled=False,
                native_cost_and_placement_unchanged=False, cost_normalization=True, volume_relative_limit=.05)
    common = dict(gpu_used=False, production_mesh_used=False, challenge_performance_verified=False, adoption=False)
    native = dict(stage='mesh_conditioned_cache_native_v1', status='pass', phase='complete', source_binding=cache_binding,
        build={'binary':binary,'build_info':info}, retained_binary=binary, original_runtime={'frozen':True},
        originals_rehashed_after=True, owned_scratch_removed=True,
        geometry={'status':'pass','exact_implementation_regression_verified':True}, **common)
    store(root/names['cache_native'], json.dumps(native).encode())
    host = dict(stage='mesh_conditioned_cache_host_v1', status='pass', source_binding=cache_binding, retained_binary=binary,
        native_identity=loader.identity(root/names['cache_native']), source_rehashed_after=True, original_build_rehashed_after=True,
        owned_container_removed=True, original_image_id=image, **common)
    store(root/names['cache_host'], json.dumps(host).encode())
    cp = dict(schema='world_reward.mesh_conditioned_cache_qualification.v1', producer_revision=cache_rev,
        host_report=loader.identity(root/names['cache_host']), native_report=loader.identity(root/names['cache_native']),
        retained_binary=binary, source_files=176, source_files_sha256='f'*64, source_archive_sha256='9'*64,
        source_cpp=cpp, exact_implementation_regression_verified=True, native_calls=8)
    store(code/loader.CACHE_PINS, json.dumps(cp).encode())
    monkeypatch.setattr(loader, 'CODE', code)
    for module in (geometry, endpoint): monkeypatch.setattr(module, '__file__', str(code/'infra'/Path(module.__file__).name))
    monkeypatch.setattr(endpoint, '_load_mesh', lambda path:(v.copy(),f.copy()))
    rows = {}
    for name in (*loader.MATH, 'configs/object_budget_conditioned_protocol_v1.json',
                 'configs/mesh_conditioned_cache_protocol_v1.json', 'infra/mesh_conditioned_qem.cpp', loader.PROCESSOR):
        store(code/name,(name+'\n').encode()); rows[name] = loader.identity(code/name)
    rows[loader.CACHE_PINS] = loader.identity(code/loader.CACHE_PINS)
    binding = dict(producer_revision=rev, helpers=rows, source_files=len(rows), markers={
        'revision':marker((rev+'\n').encode()),'source-sha256':marker(('8'*64+'\n').encode())},
        source_files_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    compact = geometry.verify_pack_fidelity((v*scale,f),pv,pf)[0]
    topology = geometry.exact_mesh_topology(*compact)
    measurements = dict(birthface_matched_shells=[{'relative_volume_error':.001}], sampled_bidirectional_chamfer_diagonal_ratio=.001,
        net_volume_relative_error=.001, scale_or_pose_fitted=False, candidate_topology=topology,
        stored_array_sha256=[array_hash(a) for a in compact])
    qualification = dict(source_binding=cache_binding,host_report=cp['host_report'],native_report=cp['native_report'],
        pins_identity=loader.identity(code/loader.CACHE_PINS),retained_binary=binary,build_info=info,original_runtime={'frozen':True})
    report = dict(stage=loader.STAGE,status='pass',phase='complete',episode_index=9,producer_revision=rev,
        script_sha256=rows[loader.PROCESSOR]['sha256'],input_track='track_1',input_sha256='1'*64,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],target_faces=4096,target_vertices=4096,
        components_deleted=False,holes_filled=False,normals_repaired=False,frame_poses_changed=False,
        native_cost_and_placement_unchanged=False,new_numeric_algorithm=True,cost_normalization=True,
        adoption_performed=False,challenge_performance_verified=False,metric_scale_accuracy_verified=False,
        source_arrays_unchanged=True,object_scale=1.,metric_scale_baked_once=scale,owned_scratch_removed=True,
        source_rehashed_after=True,cache_rehashed_after=True,runtime_rehashed_after=True,helpers_rehashed_after=True,
        inputs_rehashed_after=True,native_attempts=1,native_returned=True,native_exit_code=0,
        cache_qualification=qualification,source_binding=binding,image_id=image,
        source_hashes={'video':'1'*64,'object_report':loader.identity(root/names['object'],readonly=False)['sha256'],
          'alignment_report':loader.identity(root/names['alignment'],readonly=False)['sha256'],
          **{name:loader.identity(root/names[role],readonly=False)['sha256'] for name,role in
             (('object.glb','source_glb'),('transform.json','transform'),('intrinsics.json','intrinsics'))}},
        stages={name:measurements.copy() for name in loader.STAGES},
        candidate_serialization={'position_weld_admissible':True,'float32_triangles_exactly_active':True},
        conditioning={'roundtrip_numerically_exact':True,'source_arrays_modified':False,'source_geometry_repaired':False},
        official_helper_identity={'bytes':2031,'sha256':'42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'},
        official_pack_fidelity={'oriented_triangles_exact':True,'official_helper_simplification_invoked':False,'nonexact_merge_or_face_deletion':False},
        geometry_sha256=loader.identity(root/names['geometry'])['sha256'],canonical_glb_sha256=loader.identity(root/names['glb'])['sha256'],
        outputs={Path(names[k]).name:loader.identity(root/names[k]) for k in ('geometry','glb')},
        runtime_identity={'image_receipt':loader.identity(root/names['build'],readonly=False),'actual_fast_build_info':info})
    fixture = dict(loader=loader,root=root,code=code,names=names,report=report,pv=pv,pf=pf,canonical=v,canonical_faces=f)
    refresh(fixture)
    return fixture


def refresh(q):
    m=q['loader'];root=q['root']; names=q['names']; r=q['report']
    store(root/names['report'],json.dumps(r).encode())
    files={name:m.identity(root/name,readonly=False) for name in names.values()}
    q['pins']=dict(schema=m.SCHEMA,episode_index=9,input_sha256='1'*64,metric_scale_baked_once=2.5,
        report=dict(files[names['report']],producer_revision=r['producer_revision'],script_sha256=r['script_sha256']),files=files)


def call(q, **kwargs):
    m=q['loader']; return m.load(q['root'],9,'1'*64,q['report']['source_hashes']['object_report'],
        q['report']['source_hashes']['alignment_report'],2.5,pins=q['pins'],**kwargs)


def test_exact_metric_arrays_keep_padding_scale_and_closed_topology(qualified):
    v,f,active,cleanup,_,receipt=call(qualified)
    assert np.array_equal(v,qualified['pv']) and np.array_equal(f,qualified['pf'])
    assert active.tolist()==[0,1,2,3] and cleanup['excluded_faces']==4092
    assert receipt['resimplification_performed'] is False and receipt['independent_embedding_reverified_here'] is False
    assert receipt['metric_scale_already_baked'] and receipt['original_artifacts_rehashed']


@pytest.mark.parametrize('bad',['no_pins','extra_path','source_hash','processor','missing_stage','cd','volume','unsafe','gt','cost'])
def test_lineage_and_geometry_fail_closed_without_fallback(qualified,bad):
    q=qualified
    if bad=='no_pins':q['pins']=None
    elif bad=='extra_path':q['pins']['files']['../other']=q['pins']['files'][q['names']['glb']]
    elif bad=='source_hash':store(q['root']/q['names']['source_glb'],b'changed',0o644)
    else:
        r=q['report']
        if bad=='processor':r['script_sha256']='0'*64
        elif bad=='missing_stage':r['stages'].pop('float32_glb')
        elif bad=='cd':r['stages']['native_candidate']['sampled_bidirectional_chamfer_diagonal_ratio']=.011
        elif bad=='volume':r['stages']['original_grounding_metric_bake']['birthface_matched_shells'][0]['relative_volume_error']=.051
        elif bad=='unsafe':r['candidate_serialization']['position_weld_admissible']=False
        elif bad=='gt':r['ground_truth_used']=True
        elif bad=='cost':r['native_cost_and_placement_unchanged']=True
        refresh(q)
    with pytest.raises(ValueError):call(q)


@pytest.mark.parametrize('bad',['extra','scale','dtype','repeated','collinear','metric'])
def test_exact_cpu_payload_and_meaningful_geometry_cannot_be_changed(qualified,bad):
    q=qualified;v,f=q['pv'].copy(),q['pf'].copy();scale=2.5;extra={}
    if bad=='extra':extra['unknown']=np.array([1])
    elif bad=='scale':scale=3.
    elif bad=='dtype':v=v.astype(np.float32)
    elif bad=='repeated':f[0]=[1,1,1]
    elif bad=='collinear':v[0]=(v[1]+v[2])/2
    elif bad=='metric':v[0,0]+=1e-8
    path=q['root']/q['names']['geometry'];path.chmod(0o600)
    np.savez_compressed(path,vertices=v,faces=f,episode_index=np.array(9,np.int64),object_scale=np.array(1.),
                        grounded_scale_baked=np.array(scale),**extra);path.chmod(0o444)
    q['report']['geometry_sha256']=q['loader'].identity(path)['sha256'];q['report']['outputs']['geometry.npz']=q['loader'].identity(path);refresh(q)
    with pytest.raises(ValueError):call(q)


def test_normalization_must_not_delete_any_meaningful_exact_face(qualified,monkeypatch):
    import world_reward.mesh_geometry as legacy
    monkeypatch.setattr(legacy,'normalize_degenerate_faces',lambda *a:(np.array([0,1,2],np.int64),{}))
    with pytest.raises(ValueError,match='delete meaningful'):call(qualified)


def test_no_gpu_cpu_solver_or_historical_module_execution():
    text=(INFRA/'conditioned_geometry_loader.py').read_text()
    for word in ('subprocess','compile_binary','object_budget_conditioned import','--build-info','import torch','source_intersections('):
        assert word not in text
    assert 'geometry.exact_mesh_topology' in text and 'geometry.verify_pack_fidelity' in text


@pytest.mark.parametrize('kind',['cache_receipt','math_source','symlink','writable_new','late_change'])
def test_independent_cache_source_and_posthash_controls(qualified,monkeypatch,kind):
    q=qualified;m=q['loader']
    if kind=='cache_receipt':store(q['root']/q['names']['cache_native'],b'changed receipt')
    elif kind=='math_source':store(q['code']/'src/world_reward/mesh_conditioning.py',b'changed pure math')
    elif kind=='symlink':
        path=q['root']/q['names']['glb'];other=path.with_name('other.glb');path.rename(other);path.symlink_to(other)
    elif kind=='writable_new':(q['root']/q['names']['geometry']).chmod(0o644)
    else:
        import object_budget_endpoint as endpoint
        old=endpoint._load_mesh
        def decode(path):
            values=old(path);store(q['root']/q['names']['source_glb'],b'changed after frozen inputs',0o644);return values
        monkeypatch.setattr(endpoint,'_load_mesh',decode)
    with pytest.raises(ValueError):call(q)

"""Tiny contracts/stubs only; no PyMeshLab, real source GLB, models or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def driver(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra'; monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('test_object_budget_endpoint',infra/'object_budget_endpoint.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def tetra():
    v=np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    return v,f


def padded(v,f):
    return np.concatenate([v,np.repeat(v[:1],4096-len(v),axis=0)]),np.concatenate([f,np.zeros((4096-len(f),3),dtype=int)])


def test_exact_seam_weld_no_face_loss_or_source_change(driver):
    v,f=tetra();rawv=v[f].reshape(-1,3);rawf=np.arange(len(rawv)).reshape(-1,3)
    before=rawv.copy(),rawf.copy();vv,ff,report=driver.exact_weld(rawv,rawf)
    assert len(vv)==4 and len(ff)==4 and report['faces_removed']==0
    assert np.array_equal(vv[ff],rawv[rawf])
    assert np.array_equal(before[0],rawv) and np.array_equal(before[1],rawf)
    assert driver.mesh_topology(vv,ff)['components'][0]['volume_sign']==1


def test_exact_weld_nearby_positions_not_rounded(driver):
    v,f=tetra();rawv=v[f].reshape(-1,3);rawf=np.arange(len(rawv)).reshape(-1,3)
    rawv[0,0]+=1e-12
    vv,ff,_=driver.exact_weld(rawv,rawf)
    assert len(vv)==5
    with pytest.raises(ValueError):driver.mesh_topology(vv,ff)
    rawv[1]=rawv[0]
    with pytest.raises(ValueError,match='collapsed'):driver.exact_weld(rawv,rawf)


def test_official_weld_padding_oriented_geometry_exact(driver):
    v,f=tetra();pv,pf=padded(v,f)
    compact,report=driver.verify_pack_fidelity((v,f),pv,pf)
    assert report['active_faces']==4 and report['padding_faces']==4092
    assert report['official_helper_simplification_invoked'] is False
    assert np.array_equal(driver._triangle_rows(*compact),driver._triangle_rows(v,f))
    assert np.array_equal(driver._triangle_rows(v,f),driver._triangle_rows(v,f[:,[1,2,0]]))
    assert not np.array_equal(driver._triangle_rows(v,f),driver._triangle_rows(v,f[:,::-1]))


@pytest.mark.parametrize('kind',['delete','flip','perturb','unused'])
def test_official_hidden_repair_or_nonexact_merge_rejected(driver,kind):
    v,f=tetra();pv,pf=padded(v,f)
    if kind=='delete':pf[0]=0
    if kind=='flip':pf[0]=pf[0,::-1]
    if kind=='perturb':pv[0,0]+=1e-12
    if kind=='unused':pv[-1,0]+=1
    with pytest.raises(ValueError):driver.verify_pack_fidelity((v,f),pv,pf)


def test_parser_requires_explicit_episode_and_boundaries(driver):
    for episode in [0,15,29]:assert driver._argument_parser().parse_args(['--episode',str(episode)]).episode==episode
    for args in [[],['--episode','-1'],['--episode','30'],['--episode','true']]:
        with pytest.raises(SystemExit):driver._argument_parser().parse_args(args)
    assert driver.MAX_SECONDS==900. and len(driver.BUDGET_HELPER_SHA)==64


def test_strict_provenance_bool_integer_no_oracle(driver):
    r={'stage':'s','status':'pass','episode_index':0,'input_track':'track_1','input_sha256':'a'*64,
       'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[]}
    driver._provenance(r,'s',0,'a'*64)
    for field,bad in [('episode_index',False),('ground_truth_used',0),('hand_labeled_test',0),('oracle_modes',['GT']),('input_sha256','b'*64)]:
        row=r|{field:bad}
        with pytest.raises(ValueError):driver._provenance(row,'s',0,'a'*64)


def test_regular_path_symlink_traversal_fail(driver,tmp_path):
    file=tmp_path/'file';file.write_text('x');assert driver.regular(tmp_path,file)==file
    link=tmp_path/'link';link.symlink_to(file)
    with pytest.raises(ValueError):driver.regular(tmp_path,link)
    with pytest.raises(ValueError):driver.regular(tmp_path,tmp_path.parent/'outside')


def endpoint_record(driver,image):
    fixture=[]
    for name in ['radial_star_hollow','torus_thin_wall','offcenter_ellipsoid_hollow','disconnected_asymmetric']:
        run={'self_intersecting_faces':0,'topology':{'vertices':2000,'faces':4096},
             'geometry':{'sampled_bidirectional_chamfer_diagonal_ratio':.001,'net_volume_relative_error':.001,'per_component_relative_volume_errors':[.001]},
             'containment':{'true_containment_verified':True}}
        fixture.append({'fixture':name,'status':'pass','two_runs_arrays_identical':True,'source_arrays_unchanged':True,'runs':[copy.deepcopy(run),copy.deepcopy(run)]})
    return {'stage':driver.ENDPOINT_STAGE,'status':'pass','pymeshlab_version':driver.VERSION,'wheel_sha256':driver.WHEEL_SHA,
            'script_sha256':driver.sha256(Path(driver.__file__).with_name('mesh_endpoint_gate.py')),
            'shared_gate_sha256':driver.sha256(Path(driver.__file__).with_name('mesh_link_gate.py')),
            'parameters':driver.ENDPOINT_PARAMETERS,'old_cohort_rerun':False,'challenge_inputs_used':False,
            'challenge_ground_truth_used':False,'adoption_performed':False,'face_budget':4096,'vertex_budget':4096,
            'max_gate_seconds':120.,'image_id':image,'fixtures':fixture}


def test_endpoint_pass_manifest_source_hash_image_and_full_coverage(driver,monkeypatch,tmp_path):
    (tmp_path/'results').mkdir();image='sha256:'+'a'*64
    build={'stage':'world_reward_topology_cpu_build','status':'pass','image_id':image}
    build_path=tmp_path/'results/image-topology-cpu.json';build_path.write_text(json.dumps(build))
    record=endpoint_record(driver,image);record['build_manifest_sha256']=driver.sha256(build_path)
    path=tmp_path/'results/mesh-endpoint.json';path.write_text(json.dumps(record))
    monkeypatch.setenv('WR_TOPOLOGY_IMAGE_ID',image)
    assert driver.experiment_gate(tmp_path)[0]['status']=='pass'
    for change in ['status','script','coverage','containment']:
        bad=copy.deepcopy(record)
        if change=='status':bad['status']='fail'
        if change=='script':bad['script_sha256']='f'*64
        if change=='coverage':bad['fixtures'].pop()
        if change=='containment':bad['fixtures'][0]['runs'][0]['containment']['true_containment_verified']=False
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):driver.experiment_gate(tmp_path)


def test_wrapper_no_gpu_selected_episode_and_readonly_inputs():
    p=Path(__file__).resolve().parents[1]/'infra/run_object_budget_endpoint.sh';s=p.read_text()
    assert '--gpus' not in s and '--cpus 4 --memory 16g --network none' in s
    assert 'explicit --episode 0..29' in s and 'object_budget_endpoint.py" "$@"' in s
    for directory in ['vendor','data','results']:assert f'src=$ROOT/{directory},dst=$ROOT/{directory},readonly' in s
    assert 'src=$ROOT/outputs,dst=$ROOT/outputs"' in s


def test_object_parent900s_and_nooverwrite_no_gpu(driver,monkeypatch,tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(driver.platform,'system',lambda:'Linux')
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original(p))
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','c'*40)
    monkeypatch.setattr(sys,'argv',[driver.__file__,'--episode','0'])
    monkeypatch.setattr(driver,'prerequisites',lambda root,episode:({'video_sha256':'a'*64},{'video':'a'*64},2.))
    monkeypatch.setattr(driver,'experiment_gate',lambda root:({'image_id':'sha256:'+'a'*64},{'endpoint_gate':'b'*64}))
    path=tmp_path/'outputs/episode_000000/object_budget_endpoint/report.json'
    def run(args,*,check,timeout,env):
        assert timeout==900. and check is False and args[-1]=='--worker'
        assert len(env['WR_OBJECT_BUDGET_NONCE'])==64
        r=json.loads(path.read_text());r['source_topology']={'partial':True};path.write_text(json.dumps(r))
        raise subprocess.TimeoutExpired(args,timeout)
    monkeypatch.setattr(driver.subprocess,'run',run)
    with pytest.raises(subprocess.TimeoutExpired):driver.main()
    r=json.loads(path.read_text())
    assert r['status']=='fail' and r['source_topology']=={'partial':True} and r['adoption_performed'] is False
    assert r['container_memory_bytes']==16*1024**3 and r['metric_scale_accuracy_verified'] is False
    before=path.read_bytes()
    with pytest.raises(RuntimeError,match='Frozen'):driver.main()
    monkeypatch.setattr(sys,'argv',[driver.__file__,'--episode','0','--worker'])
    with pytest.raises(RuntimeError,match='immutable'):driver.main()
    assert path.read_bytes()==before


def test_prerequisites_routes_hashes_and_scale_once(driver,monkeypatch,tmp_path):
    episode=29;base=tmp_path/'outputs/episode_000029'
    for folder in ['object_grounded','scale_smoke','body_smoke','depth_smoke']:(base/folder).mkdir(parents=True,exist_ok=True)
    inputs={'video_sha256':'a'*64,'mask_report_sha256':'b'*64,'prompts_sha256':'c'*64}
    monkeypatch.setattr(driver,'_validate_inputs',lambda root,episode_index: inputs if episode_index==episode else None)
    common={'status':'pass','episode_index':episode,'input_track':'track_1','input_sha256':'a'*64,
            'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[]}
    for folder,stage in [('body_smoke','sam3d_body_three_frame_smoke'),('depth_smoke','monocular_moge2_three_frame')]:
        (base/folder/'report.json').write_text(json.dumps(common|{'stage':stage}))
    alignment=common|{'stage':'predicted_human_anchored_moge2_pointmaps','coordinate_frame':'OpenCV_x_right_y_down_z_forward',
        'pointmap_scale_application':'one_clip_scalar_to_MoGe2_XYZ_already_applied',
        'body_report_sha256':driver.sha256(base/'body_smoke/report.json'),'depth_report_sha256':driver.sha256(base/'depth_smoke/report.json')}
    alignment_path=base/'scale_smoke/report.json';alignment_path.write_text(json.dumps(alignment))
    transform={'scale':[2.,2.,2.],'rotation':[1.,0.,0.,0.],'translation':[0.,0.,3.]}
    (base/'object_grounded/object.glb').write_bytes(b'ownfakegeometry')
    (base/'object_grounded/transform.json').write_text(json.dumps(transform));(base/'object_grounded/intrinsics.json').write_text('{}')
    obj=common|{'stage':'sam3d_objects_grounded_fixed_frame','frame_index':0,'transform':transform,
        'scale_source':'already_human_anchored_MoGe2_no_second_scalar',
        'pointmap_grounding':{'alignment_report_sha256':driver.sha256(alignment_path)}}
    for name,field in [('object.glb','object_sha256'),('transform.json','transform_sha256'),('intrinsics.json','intrinsics_sha256')]:
        obj[field]=driver.sha256(base/'object_grounded'/name)
    obj_path=base/'object_grounded/report.json';obj_path.write_text(json.dumps(obj))
    validated,hashes,scale=driver.prerequisites(tmp_path,episode)
    assert validated==inputs and scale==2. and hashes['video']=='a'*64
    (base/'object_grounded/object.glb').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):driver.prerequisites(tmp_path,episode)

"""Tiny contract/I/O tests; no local images, inference, rendering or reference data."""
import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
import form_hoi_external_predict as p


def protocol():return p.config(REPO)


def sample(tmp_path):
    sequence='FORMDEV_clip'
    cohort={'cohort':[dict(sequence_id=x,split='development') for x in (sequence,'two','three','four')]}
    code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    (code/'configs/form_hoi_insight_v1.json').write_text(json.dumps(cohort))
    (code/'configs/form_hoi_external_dev_v1.json').write_bytes((REPO/'configs/form_hoi_external_dev_v1.json').read_bytes())
    directory=tmp_path/sequence/'inputs';directory.mkdir(parents=True)
    video=directory/'rgb.mp4';video.write_bytes(b'tiny-not-decoded-video');video.chmod(0o444)
    spec=dict(schema='world_reward.external_rgb_input.v1',sequence_id=sequence,dataset='nvidia/form-hoi',
        dataset_revision='c63db107e84c7f74bb4929ef643b67b5c8bcc00e',split='development',video=str(video),
        video_pin=p.artifact(video),total=96,full_source_frames=764,camera='front_stereo_camera_left',height=1152,width=1536,
        fps=30,original_frame_indices=list(range(96)),object_prompt='a large beige container',action='lift the container',
        inference_ready=True,reference_inputs_present=False)
    path=directory/'input.json';pin=p.save_json(path,spec)
    return code,path,pin,spec


def test_exact_generic_FORM_schema_no_fake_Track1_ID(tmp_path):
    code,path,pin,spec=sample(tmp_path)
    assert p.load_public(path,pin,code)==spec
    assert 'episode_index' not in spec and 'input_track' not in spec


@pytest.mark.parametrize('fault',['reference','not_ready','indices','revision','testsplit','native_id','reserved'])
def test_fail_closed_forbidden_or_unqualified_inputs(tmp_path,fault):
    code,path,pin,spec=sample(tmp_path)
    if fault=='reference':spec['reference_inputs_present']=True
    if fault=='not_ready':spec['inference_ready']=False
    if fault=='indices':spec['original_frame_indices'][47]=48
    if fault=='revision':spec['dataset_revision']='0'*40
    if fault=='testsplit':spec['split']='reserved_unopened'
    if fault=='native_id':spec['person_id']='human-label'
    if fault=='reserved':spec['sequence_id']='reserved'
    path.chmod(0o600);path.write_text(json.dumps(spec));path.chmod(0o444);pin=p.artifact(path)
    with pytest.raises(ValueError):p.load_public(path,pin,code)


def test_public_directory_forbids_private_sibling_files(tmp_path):
    code,path,pin,_=sample(tmp_path)
    (path.parent/'masks.npz').write_bytes(b'not opened')
    with pytest.raises(ValueError):p.load_public(path,pin,code)


def test_seal_is_atomic_owned_and_never_overwrites(tmp_path):
    path=tmp_path/'result.json';pin=p.save_json(path,{'value':3})
    assert pin==dict(bytes=13,sha256=hashlib.sha256(b'{"value": 3}\n').hexdigest())
    assert not path.stat().st_mode&0o222 and path.stat().st_nlink==1
    with pytest.raises(ValueError):p.save_json(path,{'value':4})
    assert not (tmp_path/'.result.json.part').exists()


def test_callback_failure_leaves_no_partial_or_publication(tmp_path):
    target=tmp_path/'fail.bin'
    def broken(stream):stream.write(b'half');raise RuntimeError()
    with pytest.raises(RuntimeError):p.seal(target,broken)
    assert list(tmp_path.iterdir())==[]


def test_artifact_rejects_writable_hardlinked_and_symlink(tmp_path):
    f=tmp_path/'f';f.write_bytes(b'source')
    with pytest.raises(ValueError):p.artifact(f)
    f.chmod(0o444);link=tmp_path/'link';link.symlink_to(f)
    with pytest.raises(ValueError):p.artifact(link)
    link.unlink();import os;os.link(f,link)
    with pytest.raises(ValueError):p.artifact(f)


def test_stage_checks_entire_payload_bank(tmp_path):
    base=tmp_path/'run';out=base/'track';out.mkdir(parents=True)
    data=p.save_npz(out/'native.npz',frame_index=np.arange(4,dtype=np.int64))
    report=dict(status='complete',stage='track',ground_truth_used=False,private_truth_read=False,artifacts={'native.npz':data})
    p.save_json(out/'report.json',report)
    assert p.read_stage(base,'track')[1]==report
    (out/'extra.txt').write_text('foreign');(out/'extra.txt').chmod(0o444)
    with pytest.raises(ValueError):p.read_stage(base,'track')


def test_fixed_camera_is_RGB_size_not_sensor_calibration():
    K=p.fixed_K(dict(width=1536,height=1152));assert np.array_equal(K,[[1920,0,768],[0,1920,576],[0,0,1]])


def test_pointmap_shared_scale_applied_once_and_unknown_preserved():
    depth=np.array([[2.,3.],[4.,5.]],np.float32);valid=np.array([[True,False],[True,True]])
    K=np.array([[10.,0,1.],[0,10.,1.],[0,0,1.]])
    xyz=p.pointmap(depth,valid,K,2.)
    assert np.isnan(xyz[0,1]).all() and np.array_equal(xyz[...,2][valid],depth[valid]*2)
    assert np.allclose(xyz[0,0],[-.2,-.2,4.])
    assert np.array_equal(depth,[[2,3],[4,5]])


def test_mounts_are_public_only_and_distinct_from_private_evaluator(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path/'root')
    code=tmp_path/'code';public_dir=tmp_path/'FORMDEV'/'inputs';base=tmp_path/'results'
    for stage in ('localize','body_depth','object','prepare','forward','fit_A','fit_B'):
        rows=p.mount_sources(stage,protocol(),code,public_dir,base)
        assert (public_dir,True) in rows
        assert all(not {'eval_private','track_1','gt'}.intersection(x.parts) for x,_ in rows)
        assert (base,False) in rows and (public_dir.parent,True) not in rows
    with pytest.raises(ValueError):p.mount_sources('prepare',protocol(),code,tmp_path/'eval_private',base)


def test_native_no_Track1_driver_or_truth_fitting_and_same300_protocol():
    cfg=protocol();assert cfg['num_steps']==300 and cfg['batch_size']==0
    tree=ast.parse((REPO/'infra/form_hoi_external_predict.py').read_text())
    calls=[x.func for x in ast.walk(tree) if isinstance(x,ast.Call)]
    names={x.id for x in calls if isinstance(x,ast.Name)}
    assert 'Track1Episode' not in names and '_validate_inputs' not in names
    assert 'validate_public_package' in {x.attr for x in calls if isinstance(x,ast.Attribute)}
    assert all(k in p.STAGES for k in ('fit_A','fit_B'))
    text=(REPO/'infra/form_hoi_external_predict.py').read_text()
    assert "allow_empty_object_evidence=True" in text
    assert "offline_supervision_contract=True" in text
    assert "eval_geometry.npz" in text
    assert "human_vertices=target, human_joints=joints, human_faces=faces_human" in text
    assert "input=r.stdout" not in text  # Only opaque envelope decrypt result goes to child stdin.


def test_same300_no_external_parameter_sweep():
    cfg=protocol();assert cfg['extension']['human_image_weight']==1 and cfg['extension']['object_image_weight']==1
    assert cfg['production_adopted'] is False and cfg['training_overlap_verified'] is False
    assert cfg['reference_policy'].startswith('eval_private_never_mounted')


def body_records(count=3):
    import body_smoke as body
    rows={k:[np.zeros(shape,np.float32) for _ in range(count)] for k,shape in body.PARAMETER_SHAPES.items()}
    for i in range(count):
        rows['pred_cam_t'][i][2]=2
        rows['focal_length'][i][...]=800
        rows['pred_keypoints_2d'][i][:]=[320,240]
    return rows


def test_actual_Body_NPZ_adapter_ABI_translation_once_and_all_blocks():
    import cari_body_adapter as adapter
    rows=body_records(); raw=[x.copy() for x in rows['pred_vertices']]
    arrays=p.body_transport(rows,np.array([[0,1,2]],np.int32),[0,1,2],(480,640))
    required=set(adapter._RAW_DIMS)|{'frame_index','vertices_root_camera_m','vertices_camera_m',
        'pred_joint_coords','pred_keypoints_3d','pred_keypoints_2d','focal_length','faces'}
    assert required<=arrays.keys() and 'pred_vertices' not in arrays
    assert np.array_equal(arrays['vertices_root_camera_m'],np.stack(raw))
    assert np.array_equal(arrays['vertices_camera_m'][...,2],np.full((3,18439),2))
    assert np.array_equal(arrays['frame_index'],[0,1,2])
    assert arrays['faces'].dtype==np.int64
    assert all(np.array_equal(a,b) for a,b in zip(rows['pred_vertices'],raw))


@pytest.mark.parametrize('fault',['missing_block','wrong_type','wrong_count','expr','camera','projection','indices'])
def test_body_transport_refuses_original_native_contract_changes(fault):
    rows=body_records();indices=[0,1,2]
    if fault=='missing_block':rows.pop('mhr_model_params')
    elif fault=='wrong_type':rows['global_rot'][1]=np.zeros(3,np.float64)
    elif fault=='wrong_count':rows['hand_pose_params'].pop()
    elif fault=='expr':rows['expr_params'][0][0]=1
    elif fault=='camera':rows['focal_length'][0][...]=1920
    elif fault=='projection':rows['pred_keypoints_2d'][1][0,0]=0
    else:indices=[0,2,1]
    with pytest.raises(ValueError):p.body_transport(rows,np.array([[0,1,2]],np.int64),indices,(480,640))


@dataclass
class NativeConfig:
    num_steps:int=300
    batch_size:int=0
    frame_start:int=0
    frame_limit:int=0
    freeze_object_rotation:bool=True
    freeze_body_internal_translations:bool=True
    report_every:int=100


def native_source_and_result(enabled):
    import cari_full_refine as full
    n=3
    params={key:np.zeros((n,dim),np.float32) for key,dim in p.NATIVE_PARAMETER_DIMS.items()}
    params['mhr_global_rot6d'][:]=[1,0,0,1,0,0];params['mhr_trans'][:,2]=2
    params['pose_abs']=np.broadcast_to(np.eye(4,dtype=np.float32),(n,4,4)).copy()
    params['contact_logits']=np.ones((n,2),np.float32)
    metadata=dict(ground_truth_used=False,object_mesh='/own/FORMDEV/output_aligned.glb',
        object_pose_frame='centered_axis_aligned',object_pose_storage_frame='output_aligned_mesh_frame',
        object_pose_frame_revision='cari4d.object_pose_frame.centered_axis_aligned.v1',
        object_pose_storage_to_training_transform=np.eye(4).tolist(),object_mesh_to_training_transform=np.eye(4).tolist())
    source=dict(schema='cari4d.mhr_wild_inference.v1',frames=[f'{i:06d}' for i in range(n)],
        gt={},metadata=metadata,pr=params,raw={'trans':np.zeros((n,3),np.float32)},
        observations={'object_mask':np.ones((n,2,2),bool)})
    result=copy.deepcopy(source);result['pr']['mhr_body_pose_cont'][:,0]+=.02
    result['pr']['pose_abs'][:,0,3]+=.003
    if enabled:result['pr']['mhr_trans'][:,0]+=.01
    result['pr']['pose_abs_postopt']=result['pr']['pose_abs'].copy()
    fixed=list(full.contract.FIXED_PARAMETERS);optimized=list(full.contract.OPTIMIZED_PARAMETERS)
    if enabled:fixed.remove('mhr_trans');optimized.append('mhr_trans')
    result['postopt']=dict(mode='native_joint_image_translation_extension_v1' if enabled else 'smplh_parity',
        config=asdict(NativeConfig()),frame_indices=list(range(n)),resolved_batch_size=n,
        batch_sampling='full_clip_v1',fixed_parameters=fixed,optimized_parameters=optimized,
        history=[dict(iter=float(i),batch_start=0.,batch_size=float(n),loss_total=.1) for i in (0,100,200,300)],
        final_diagnostics={'contact':.01})
    return source,result


@pytest.mark.parametrize('enabled',[False,True])
def test_same_frontend_native_A_B_preserve_actual_native_fields(enabled):
    source,result=native_source_and_result(enabled)
    params,poses=p.validate_solver_result(source,result,NativeConfig(),3,enabled=enabled)
    assert params['mhr_trans'][0,0]==(np.float32(.01) if enabled else 0)
    assert poses.shape==(3,4,4)


@pytest.mark.parametrize('fault',['contacts','raw','GT','objectR','internal','identity','posecopy',
    'extra_prediction','partial','history','batch','diagnostics'])
def test_candidate_may_not_silently_change_inputs_contact_or_full_T(fault):
    source,result=native_source_and_result(True)
    if fault=='contacts':result['pr']['contact_logits'][1,0]=0
    elif fault=='raw':result['raw']['trans'][1,0]=.01
    elif fault=='GT':result['gt']={'forbidden':0}
    elif fault=='objectR':
        result['pr']['pose_abs'][0,:3,:3]=np.diag([-1,-1,1])
        result['pr']['pose_abs_postopt']=result['pr']['pose_abs'].copy()
    elif fault=='internal':result['pr']['mhr_body_pose_cont'][0,254]=.1
    elif fault=='identity':result['pr']['mhr_shape'][1,0]=.1
    elif fault=='posecopy':result['pr']['pose_abs_postopt'][0,0,3]=0
    elif fault=='extra_prediction':result['pr']['invented']=np.zeros(3)
    elif fault=='partial':result['postopt']['frame_indices']=[0,1]
    elif fault=='history':result['postopt']['history'][1]['iter']=99.
    elif fault=='batch':result['postopt']['history'][1]['batch_size']=2.
    else:result['postopt']['final_diagnostics']['contact']=np.nan
    with pytest.raises(ValueError):p.validate_solver_result(source,result,NativeConfig(),3,enabled=True)


def test_actual_native_tools_scopes_shadowed_package_and_restores_on_error(tmp_path, monkeypatch):
    import types
    native=tmp_path/'cari'; tools=native/'tools'; tools.mkdir(parents=True)
    (tools/'__init__.py').write_text('# qualified test package\n')
    (tools/'pipeline_timing.py').write_text('class PipelineTimer: pass\n')
    for file in tools.iterdir(): file.chmod(0o444)
    monkeypatch.setattr(p,'NATIVE_TOOLS_PINS',{str(f.relative_to(native)):p.identity(f,16384) for f in tools.iterdir()})
    shadow=types.ModuleType('tools'); shadow.__path__=[str(tmp_path/'sam_body/tools')]
    prior=types.ModuleType('tools.old'); monkeypatch.setitem(sys.modules,'tools',shadow); monkeypatch.setitem(sys.modules,'tools.old',prior)
    previous=sys.path.copy()
    with pytest.raises(RuntimeError):
        with p.native_tools(native):
            assert Path(sys.modules['tools.pipeline_timing'].__file__)==tools/'pipeline_timing.py'
            assert sys.modules['tools'] is not shadow and 'tools.old' not in sys.modules
            raise RuntimeError('test exception')
    assert sys.modules['tools'] is shadow and sys.modules['tools.old'] is prior and sys.path == previous
    assert 'tools.pipeline_timing' not in sys.modules


def test_actual_prepare_and_forward_use_native_tools_without_signature_workaround():
    tree=ast.parse((REPO/'infra/form_hoi_external_predict.py').read_bytes())
    for name in ('prepare','forward'):
        n=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
        assert any(isinstance(x,ast.With) and 'native_tools(native)' in ast.unparse(x.items[0].context_expr) for x in ast.walk(n))
    n=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
    call=next(x for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='prepare_mhr_wild_export')
    assert len(call.args)==5  # Exact13145B7c0d native API; output_root is NOT keyword-only.
    seen=[]
    def native(video,mask_h5,object_mesh,intrinsics_file,output_root,*,redo=False):
        seen.append((video,mask_h5,object_mesh,intrinsics_file,output_root,redo));return output_root
    assert native(*range(5)) == 4 and seen[0][-1] is False

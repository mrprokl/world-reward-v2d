"""Manufactured tiny occlusion/provenance contracts, not real pose accuracy."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

import object_pose_smoke as pose


def _default_projection(tree):
    """Evaluate ONLY explicit opt-in flag=False; retain every legacy statement."""
    class Default(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.value, ast.Name) and node.value.id == 'args'
                    and node.attr == 'allow_unobserved_poses'):
                return ast.copy_location(ast.Constant(False), node)
            return node
        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
                return ast.copy_location(ast.Constant(not node.operand.value), node)
            return node
        def visit_BoolOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.And):
                if any(isinstance(x, ast.Constant) and x.value is False for x in node.values):
                    return ast.copy_location(ast.Constant(False), node)
                node.values = [x for x in node.values if not (isinstance(x, ast.Constant) and x.value is True)]
            elif isinstance(node.op, ast.Or):
                if any(isinstance(x, ast.Constant) and x.value is True for x in node.values):
                    return ast.copy_location(ast.Constant(True), node)
                node.values = [x for x in node.values if not (isinstance(x, ast.Constant) and x.value is False)]
            return node.values[0] if len(node.values) == 1 else node
        def visit_If(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
                return node.body if node.test.value else node.orelse
            return node
        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
                return node.body if node.test.value else node.orelse
            return node
    return Default().visit(tree)


def candidate(index, angle=0., translation=None, slot=0):
    return dict(hypothesis_index=slot, rotation=Rotation.from_euler('z',angle,degrees=True).as_matrix().tolist(),
        translation=[index*.1,0.,2.] if translation is None else np.asarray(translation).tolist(),
        initial_silhouette_iou=.8, fitted_silhouette_iou=.8,
        selected_silhouette_iou=.8, selected_depth_residual_m=.01)


def reports(n=7, observed=(0,3,6)):
    return [dict(frame_index=i, candidates=[candidate(i,angle=i*10.)], selected=candidate(i), pose_observed=True)
            if i in observed else pose._missing_pose_report(i,0,'a'*64,'empty_automatic_mask')
            for i in range(n)]


def test_optin_requires_full_video_before_input_io(monkeypatch):
    assert not pose._argument_parser().parse_args([]).allow_unobserved_poses
    assert pose._argument_parser().parse_args(['--full-video','--allow-unobserved-poses']).allow_unobserved_poses
    monkeypatch.setattr(pose.platform,'system',lambda:'Linux')
    monkeypatch.setattr(pose.Path,'iterdir',lambda _:iter([Path('lo')]))
    monkeypatch.setattr(pose,'_validate_inputs',lambda *_a,**_k:pytest.fail('No invalid mode source IO'))
    monkeypatch.setattr(pose.sys,'argv',['object_pose_smoke.py','--allow-unobserved-poses'])
    with pytest.raises(ValueError,match='require --full-video'): pose.main()


def test_full_t_bilateral_initializer_retains_observed_exact_and_true_motion():
    data=reports(); r,t,flags,record=pose._select_latent_pose_path(data,list(range(7)),2,np.zeros(3))
    assert flags.tolist()==[True,False,False,True,False,False,True]
    for i in (0,3,6):
        np.testing.assert_array_equal(r[i],data[i]['candidates'][0]['rotation'])
        np.testing.assert_array_equal(t[i],data[i]['candidates'][0]['translation'])
    np.testing.assert_allclose(r @ r.swapaxes(-1,-2),np.broadcast_to(np.eye(3),r.shape),atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-14)
    np.testing.assert_allclose(t[:,0],np.arange(7)*.1,atol=1e-14)
    assert np.all(np.linalg.norm(np.diff(t,axis=0),axis=1)>0)
    assert record['latent_frame_indices']==[1,2,4,5]
    assert record['candidate_indices']==[0,-1,-1,0,-1,-1,0]
    assert record['latent_pose_status']=='initializer_not_measured_or_final_truth'
    assert not record['edge_extrapolation'] and not record['zero_velocity_prior'] and not record['mask_interpolation']


def test_centroid_interpolation_is_canonical_origin_invariant_and_fixed_mesh():
    centre=np.array([2.,-1.,.3]); data=reports(5,(0,4)); vertices=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,1.]])
    before=vertices.copy(); r,t,_,_=pose._select_latent_pose_path(data,list(range(5)),2,centre)
    shift=np.array([20.,-8.,3.]); shifted=[]
    for row in data:
        item=dict(row); item['candidates']=[]
        for c in row['candidates']:
            q=dict(c); q['translation']=(np.asarray(c['translation'])-np.asarray(c['rotation'])@shift).tolist()
            item['candidates'].append(q)
        shifted.append(item)
    r2,t2,_,_=pose._select_latent_pose_path(shifted,list(range(5)),2,centre+shift)
    np.testing.assert_array_equal(r2,r)
    np.testing.assert_allclose(t2+np.einsum('tij,j->ti',r2,shift),t,atol=1e-13)
    np.testing.assert_array_equal(vertices,before)


@pytest.mark.parametrize('observed',[(1,3,6),(0,3,5),(),(3,)])
def test_unsupported_edges_or_all_hidden_are_not_static_extrapolated(observed):
    with pytest.raises(ValueError,match='Leading/trailing unknown'):
        pose._select_latent_pose_path(reports(observed=observed),list(range(7)),2,np.zeros(3))


def test_original_frame_times_and_legacy_weights_are_kept(monkeypatch):
    real=pose.select_pose_path; calls=[]
    def capture(*args,**kwargs): calls.append(kwargs);return real(*args,**kwargs)
    monkeypatch.setattr(pose,'select_pose_path',capture)
    pose._select_latent_pose_path(reports(),list(range(7)),2,np.zeros(3))
    np.testing.assert_array_equal(calls[0]['frame_times'],[0,3,6])
    assert calls[0]['translation_weight']==1. and calls[0]['rotation_weight']==.1


@pytest.mark.parametrize('fault',['indices','frame','duplicate','nonfinite','nonrigid'])
def test_latent_helper_rejects_reindexed_or_invalid_hypotheses(fault):
    data=reports();indices=list(range(7))
    if fault=='indices':indices[2]=8
    elif fault=='frame':data[2]['frame_index']=99
    elif fault=='duplicate':data[0]['candidates']*=2
    elif fault=='nonfinite':data[0]['candidates'][0]['translation'][0]=float('nan')
    elif fault=='nonrigid':data[0]['candidates'][0]['rotation'][0][0]=2.
    with pytest.raises(ValueError):pose._select_latent_pose_path(data,indices,2,np.zeros(3))


def test_missing_observation_receipt_contains_no_measured_pose():
    row=pose._missing_pose_report(4,12,'a'*64,'insufficient_inferred_visible_points',sampled=12)
    assert row['frame_index']==4 and row['sampled_observations']==12 and row['visible_point_pixels']==12
    assert row['selected'] is None and row['candidates']==[] and row['pose_observed'] is False
    assert 'rotation' not in row and 'translation' not in row and row['object_mask_sha256']=='a'*64


def test_observed_flags_cannot_silently_relabel_latent_poses_as_measurements():
    data=reports();data[2]['pose_observed']=True
    with pytest.raises(ValueError,match='honest explicit pose-observed'):
        pose._select_latent_pose_path(data,list(range(7)),2,np.zeros(3))


def _loop_fixture(tmp_path, *, kind='empty', allow=True, fault=None):
    """Execute exact production loop AST with tiny original PNG/NPZ evidence."""
    h=w=8;base=tmp_path/'episode';masks=base/'automatic_masks/masks/1';masks.mkdir(parents=True)
    depths=base/'depth_full';depths.mkdir();pointmaps={};body={}
    for i in range(3):
        target=np.ones((h,w),np.uint8)*255
        if i==1 and kind=='empty':target[:]=0
        elif i==1 and kind=='small':target[:]=0;target[:3,:4]=255
        Image.fromarray(target).save(masks/f'{i:06d}.png')
        path=depths/f'{i:06d}.npz'
        np.savez(path,depth=np.ones((h,w),float)*(3. if kind=='numerical' and i==1 else 2.),mask=np.ones((h,w),bool),
                 intrinsics=np.diag([1/w,1/h,1]),frame_index=np.array(i))
        pointmaps[i]=dict(output_sha256=pose.sha256(path),decoded_rgb_sha256='a'*64)
        body[i]=dict(decoded_rgb_sha256='a'*64)
    if fault=='sha':pointmaps[1]['output_sha256']='b'*64
    elif fault=='rgb':body[1]['decoded_rgb_sha256']='b'*64
    elif fault=='grid':Image.fromarray(np.zeros((7,8),np.uint8)).save(masks/'000001.png')
    def raster(vertices,faces,camera,width,height):
        if fault=='cuda':raise RuntimeError('CUDA infrastructure error')
        raw=np.ones((h,w),bool)
        return SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:raw)),None
    def align(sampled,observed,r,t):
        if kind=='numerical' and len(observed)==64 and np.median(observed[:,2])==3:
            raise ValueError('underconstrained numerical hypothesis')
        return SimpleNamespace(rotation=r,translation=t,final_residual=.01,initial_residual=.01,to_dict=lambda:{})
    env=dict(vars(pose));env.update(np=np,Image=Image,base=base,indices=[0,1,2],pointmaps=pointmaps,
        body_frames=body,alignment=dict(depth_alignment=dict(shared_scale=1.)),camera=np.eye(3),
        camera_dict=dict(height=h,width=w),height=h,width=w,args=SimpleNamespace(full_video=True,allow_unobserved_poses=allow),
        rotation=np.eye(3),translation=np.array([0.,0.,2.]),orientation_hypotheses=np.eye(3)[None],
        mesh=SimpleNamespace(vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]]),faces=np.array([[0,1,2]]),centroid=np.zeros(3)),
        sampled=np.zeros((4,3)),previous_rotation=np.eye(3),candidate_reports=[],poses_R=[],poses_t=[],
        raster_camera_mesh=raster,align_observed_points=align,silhouette_iou=lambda *_:.8,started=0.)
    tree=ast.parse(Path(pose.__file__).read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
              and n.target.id=='index' and isinstance(n.iter,ast.Name) and n.iter.id=='indices')
    return env,compile(ast.Module(body=[loop],type_ignores=[]),'<exact production pose loop>','exec')


@pytest.mark.parametrize('kind',['empty','small','numerical'])
def test_actual_loop_preserves_every_missing_frame_without_invented_candidate(tmp_path,kind):
    env,code=_loop_fixture(tmp_path,kind=kind)
    exec(code,env)
    assert [r['frame_index'] for r in env['candidate_reports']]==[0,1,2]
    row=env['candidate_reports'][1]
    assert row['selected'] is None and row['candidates']==[] and not row['pose_observed']
    assert row['object_mask_sha256']==pose.sha256(env['base']/'automatic_masks/masks/1/000001.png')


def test_default_empty_observation_still_fails_with_original_error(tmp_path):
    env,code=_loop_fixture(tmp_path,allow=False)
    with pytest.raises(RuntimeError,match='Insufficient inferred visible object points'):exec(code,env)


@pytest.mark.parametrize('fault',['sha','rgb','grid','cuda'])
def test_missing_pose_mode_cannot_hide_original_provenance_or_infrastructure_errors(tmp_path,fault):
    env,code=_loop_fixture(tmp_path,fault=fault)
    with pytest.raises(RuntimeError):exec(code,env)


def test_optin_has_separate_namespace_and_pose_observed_npz_flags():
    source=Path(pose.__file__).read_text();tree=ast.parse(source)
    guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.allow_unobserved_poses']
    assert any("output.name + '_latent'" in ast.unparse(n) for n in guards)
    assert any('pose_observed=pose_observed' in ast.unparse(n) and 'np.savez_compressed' in ast.unparse(n) for n in guards)
    assert 'latent_poses_measured=False' in source


def test_missing_frames_still_publish_original_progress_counts(capsys):
    pose._missing_pose_progress(49,0.)
    assert '"frames_complete": 50' in capsys.readouterr().out

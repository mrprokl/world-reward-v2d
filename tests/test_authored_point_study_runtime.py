"""Tiny authored component caller controls; never native GPU qualification."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'infra'),str(ROOT/'src')]
import joint_point_authored_qualify as q
import mediapipe_cpu_runtime_verify as rt
from world_reward import authored_point_study as study
from cari_converter import PARAMETER_DIMS


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


def metadata_fixture():
    names=[f'parameter_{i}' for i in range(249)]
    controls=('uparm_ry','elbow_bend','thumb2_rz','index1_rz','index2_rz','middle1_rz')
    for side,start in (('l',6),('r',32)):
        for j,name in enumerate(controls):names[start+j]=side+'_'+name
    for side,start in (('l',68),('r',95)):
        for j in range(27):names[start+j]=side+f'_finger_{j}'
    return names,np.tile([-np.inf,np.inf],(249,1)),SimpleNamespace(hand_pose_mean=Tensor(np.full(54,.125,np.float32)),
        hand_pose_comps=Tensor(np.eye(54,dtype=np.float32)*2),hand_joint_idxs_left=Tensor(np.arange(68,95,dtype=np.int64)),
        hand_joint_idxs_right=Tensor(np.arange(95,122,dtype=np.int64)))


def test_primary_converter_pca_mapping_exact_roles_not_guessed_columns():
    names,bounds,head=metadata_fixture();recipe=study.named204_recipe(names,bounds,0,scale68=np.full(68,.25,np.float32),identity45=np.zeros(45,np.float32))
    calls=[]
    def body(value):calls.append(('body',value.copy()));return np.tile(np.arange(260,dtype=np.float32),(24,1))
    def hand(value):calls.append(('hand',value.copy()));return np.repeat(value,2,axis=1)
    values,record=q.study_parameters(np,recipe,head,body,hand)
    assert [(name,a.shape) for name,a in calls]==[('body',(24,133)),('hand',(24,27)),('hand',(24,27))]
    assert np.array_equal(calls[0][1][:,:130],recipe.parameters[:,6:136]) and not calls[0][1][:,130:].any()
    assert np.array_equal(calls[1][1],recipe.parameters[:,68:95]) and np.array_equal(calls[2][1],recipe.parameters[:,95:122])
    for coeff,desired in zip(np.split(values['mhr_hand'],2,axis=1),(np.repeat(a,2,axis=1) for _,a in calls[1:])):
        assert np.array_equal(.125+coeff*2,desired)
    assert set(values)==set(PARAMETER_DIMS) and all(a.dtype==np.float32 and a.shape==(24,PARAMETER_DIMS[name]) for name,a in values.items())
    assert np.array_equal(values['mhr_trans'],np.tile([0,0,4],(24,1))) and not values['mhr_scale'].any()
    assert record['network_executed'] is False and record['hand_pca_reconstruction_max_abs']==[0.,0.]


@pytest.mark.parametrize('fault',['singular','badmean','overlap','negative','q_dtype'])
def test_mapping_unknown_or_singular_stops_no_fallback(fault):
    names,bounds,head=metadata_fixture();recipe=study.named204_recipe(names,bounds,0,scale68=np.zeros(68,np.float32),identity45=np.zeros(45,np.float32))
    if fault=='singular':head.hand_pose_comps=Tensor(np.zeros((54,54),np.float32))
    elif fault=='badmean':head.hand_pose_mean=Tensor(np.zeros(53,np.float32))
    elif fault=='overlap':head.hand_joint_idxs_right=head.hand_joint_idxs_left
    elif fault=='negative':head.hand_joint_idxs_left=Tensor(np.arange(-1,26,dtype=np.int64))
    else:recipe=SimpleNamespace(**{**vars(recipe),'parameters':recipe.parameters.astype(np.float64)})
    with pytest.raises(ValueError):q.study_parameters(np,recipe,head,lambda a:np.zeros((24,260),np.float32),lambda a:np.zeros((24,54),np.float32))


def dynamic_report():
    proof={'control':'component_preflight','source_binding':{'fixture':'source'}}
    rows=[dict(scene_id=name,split=split,side=side,articulation={'passed':True},material_proximity={'passed':True},sentinel_metrics={'passed':True},translation_observability={'passed':True},
        query_diagnostics={'qualified_candidates':8,'distinct_canonical_witnesses':8,'raycast_complete':True},hand_face_index=0,
        native_contact={'effective_authored_side_all_frames':True},all_vertices_positive_Z=True) for name,split,side in study.SCENES]
    report=dict(stage='joint_point_component_preflight_v1',status='pass',phase='complete',control='component_preflight',frames=144,native_decode_calls=6,native_decoded_frames=144,sentinel_composites=18,primitive_render_calls=12,primitive_render_frames=36,
        optimizer_constructors=0,optimizer_loss_calls=0,optimizer_updates=0,component_preflight_verified=True,native_metadata_rehashed_after=True,penetration_proxy_executed=False,
        manufactured_not_inferred=True,tracker_executed=False,network_executed=False,positive_weight_executed=False,quality_verified=False,adoption=False,
        ground_truth_used=False,challenge_inputs_used=False,source_inputs_assets_rehashed_after=True,source_binding=proof['source_binding'],protocol_identity=q.STUDY_PIN,component_scenes=rows)
    return report,proof


def test_component_scope_separate_from_old_pair_proof():
    report,proof=dynamic_report();q.validate_dynamic(rt,report,proof)
    with pytest.raises(ValueError):q.validate_native(rt,report,proof)


@pytest.mark.parametrize('field,value',[('native_decode_calls',5),('primitive_render_calls',18),('frames',143),('optimizer_constructors',1),
    ('optimizer_updates',301),('tracker_executed',True),('positive_weight_executed',True),('native_metadata_rehashed_after',False)])
def test_wrong_runtime_census_does_not_qualify(field,value):
    report,proof=dynamic_report();report[field]=value
    with pytest.raises(ValueError):q.validate_dynamic(rt,report,proof)


@pytest.mark.parametrize('field',['articulation','material_proximity','sentinel_metrics','translation_observability','native_contact','query','sceneorder','positiveZ'])
def test_one_failed_scene_whole_cohort_stops(field):
    report,proof=dynamic_report();row=report['component_scenes'][3]
    if field=='native_contact':row[field]['effective_authored_side_all_frames']=False
    elif field=='query':row['query_diagnostics']['qualified_candidates']=7
    elif field=='sceneorder':report['component_scenes'].reverse()
    elif field=='positiveZ':row['all_vertices_positive_Z']=False
    else:row[field]['passed']=False
    with pytest.raises(ValueError):q.validate_dynamic(rt,report,proof)


def test_explicit_cli_disjoint_namespace_and_future_missing_pin(monkeypatch,tmp_path):
    assert q.arguments(['--control','component_preflight']).control=='component_preflight'
    with pytest.raises(SystemExit):q.arguments(['--control','component_preflight','--control','zero_delegate'])
    assert len({q.output_path(tmp_path,'a'*40,mode) for mode in (None,'native_repeat','zero_delegate','component_preflight')})==4
    monkeypatch.setattr(q,'STUDY_PIN',None)
    gate=SimpleNamespace(pinned=lambda path,pin,cap:json.loads(path.read_text()),require=rt.require)
    with pytest.raises(ValueError,match='release'):q.protocol(gate,ROOT,'component_preflight')


def test_component_protocol_exact_hash_and_default_unchanged():
    raw=(ROOT/q.STUDY_PROTOCOL).read_bytes()
    assert q.STUDY_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    gate=SimpleNamespace(pinned=lambda p,pin,cap:rt.strict(p.read_bytes()),require=rt.require)
    c=q.protocol(gate,ROOT,'component_preflight')
    assert c['component_study']['cohort']['decode_frames']==144 and c['component_study']['cohort']['sentinel_render_frames']==18
    assert q.protocol(gate,ROOT)==json.loads((ROOT/q.PROTOCOL).read_text())


def test_bare_host_lazy_and_static_no_optimizer_path_or_old_scene():
    raw=(ROOT/'infra/joint_point_authored_qualify.py').read_text();tree=ast.parse(raw)
    component=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='component_native'))
    assert 'MHRParityPostOptimizer(' not in component and '.loss(' not in component and '.run(' not in component
    assert 'manufacture(' not in component and 'paired_native(' not in component and 'torch.no_grad()' in component
    assert '_initial_contact_activation(' in component and 'canonical_mask_quantile_queries' in component
    assert "dtype=np.float32" in component and 'pose[0, :3, :3]' in component and 'object_world[sentinels]' in component
    code=f"import runpy,sys;runpy.run_path({str(ROOT/'infra/joint_point_authored_qualify.py')!r},run_name='audit');assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-I','-B','-S','-c',code],check=True,capture_output=True)


def test_static_runtime_sourceclosure_contains_core_and_protocol():
    import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ('infra','src/world_reward','configs') for p in (ROOT/folder).rglob('*') if p.is_file()}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    paths=azure_job.runtime_bundle_paths(files,'infra/run_joint_point_authored_qualify.sh')
    assert set(q.helpers('component_preflight'))<=set(paths)


def test_component_pipeline_six_decodes_only18_render_frames_and_same_native_pose(tmp_path,monkeypatch):
    """Tiny native-shaped callbacks exercise real caller, not model performance."""
    import cari_full_refine as full
    import world_reward.point_surface_queries as selectors
    from world_reward.point_surface_queries import SurfaceQueries
    names,bounds,head=metadata_fixture()
    for i in range(68,95):names[i]='l_finger_'+str(i)
    for i in range(95,122):names[i]='r_finger_'+str(i)
    joints=['l_wrist','r_wrist']+[f'joint_{i}' for i in range(125)]
    faces=np.array([[0,1,2]],np.int64);parents=np.arange(127,dtype=np.int64)-1;transform=np.zeros((889,249),np.float32)
    lbs=(np.tile(np.arange(8),(3,1)),np.full((3,8),.125,np.float32));head.scale_mean=Tensor(np.zeros(68,np.float32));head.scale_comps=Tensor(np.zeros((28,68),np.float32))
    model=SimpleNamespace(get_parameter_names=lambda:names,get_joint_names=lambda:joints,get_parameter_limits=lambda:Tensor(bounds),
        get_parameter_transform=lambda:Tensor(transform),get_lbsw=lambda:tuple(Tensor(a) for a in lbs),get_num_identity_blendshapes=lambda:45,get_num_face_expression_blendshapes=lambda:72,
        character_torch=SimpleNamespace(parameter_transform=SimpleNamespace(parameter_names=names,parameter_transform=Tensor(transform)),
            skeleton=SimpleNamespace(joint_names=joints,joint_parents=Tensor(parents)),mesh=SimpleNamespace(faces=Tensor(faces))))
    head.mhr=model;events=[];renders=[];decoded=[]
    class DeviceTensor(Tensor):
        @property
        def shape(self):return self.value.shape
        def __len__(self):return len(self.value)
        def __getitem__(self,i):
            if isinstance(i,tuple):i=tuple(a.value if isinstance(a,DeviceTensor) else a for a in i)
            return DeviceTensor(self.value[i])
        def __matmul__(self,other):return DeviceTensor(self.value@other.value)
        def __add__(self,other):return DeviceTensor(self.value+other.value)
        def __gt__(self,value):return DeviceTensor(self.value>value)
        def __eq__(self,other):return DeviceTensor(self.value==(other.value if isinstance(other,DeviceTensor) else other))
        def __and__(self,other):return DeviceTensor(self.value&other.value)
        def __or__(self,other):return DeviceTensor(self.value|other.value)
        def __invert__(self):return DeviceTensor(~self.value)
        def __lt__(self,other):return DeviceTensor(self.value<other.value)
        def __setitem__(self,i,v):self.value[i]=v
        def all(self):return self.value.all()
        def transpose(self,a,b):return DeviceTensor(self.value.swapaxes(a,b))
        def contiguous(self):return self
        def cuda(self):return self
    class Context:
        def __enter__(self):return self
        def __exit__(self,*args):return False
    torch=SimpleNamespace(device=lambda v:v,int32=np.int32,float32=np.float32,no_grad=lambda:Context(),
        from_numpy=lambda a:DeviceTensor(a),tensor=lambda a,**k:DeviceTensor(np.asarray(a,dtype=k.get('dtype'))),zeros=lambda shape,**k:DeviceTensor(np.zeros(shape,np.float32)),
        any=lambda a:np.any(a.value),zeros_like=lambda a:DeviceTensor(np.zeros_like(a.value)),where=lambda cond,a,b:DeviceTensor(np.where(cond.value,a.value,b.value)))
    def decode(params):
        events.append('decode24');assert all(a.value.shape[0]==24 for a in params.values())
        v=np.tile(np.array([[-.01,-.01,4.],[.01,-.01,4.],[0,.01,4.]],np.float32),(24,1,1))
        output=SimpleNamespace(vertices=DeviceTensor(v),joints=DeviceTensor(np.zeros((24,127,3),np.float32)),keypoints=DeviceTensor(np.zeros((24,70,3),np.float32)),
            joint_global_rots=DeviceTensor(np.tile(np.eye(3,dtype=np.float32),(24,127,1,1))))
        decoded.append(output);return output
    layer=SimpleNamespace(backend=SimpleNamespace(_ensure_head=lambda device:head),mesh_faces=lambda **k:Tensor(faces),mhr_forward=decode)
    spec=SimpleNamespace(vertex_indices=np.tile(np.arange(3,dtype=np.int32),(2,1)))
    def activate(weights,hand,r,t,v,f,threshold):
        events.append('contact24');assert hand.value.shape==(24,2,3,3) and r.value.dtype==t.value.dtype==np.float32 and threshold==.005
        return weights,DeviceTensor(np.zeros((24,2),np.float32)),DeviceTensor(np.ones((24,2),bool))
    optimizer=SimpleNamespace(MHRParityPostOptConfig=lambda:SimpleNamespace(contact_activation_distance_m=.005),_initial_contact_activation=activate)
    def render(K,context,mesh,size,world):
        renders.append(world.value.copy());assert size==(480,640) and len(world.value)==3
        depth=DeviceTensor(np.ones((3,480,640),np.float32)*(4 if len(renders)%2 else 3.9))
        return DeviceTensor(np.full((3,480,640,3),.5,np.float32)),depth,None
    monkeypatch.setitem(sys.modules,'Utils',SimpleNamespace(nvdiff_color_depth_render=render))
    monkeypatch.setitem(sys.modules,'nvdiffrast.torch',SimpleNamespace(RasterizeCudaContext=lambda:object()))
    package=SimpleNamespace();monkeypatch.setitem(sys.modules,'nvdiffrast',package);package.torch=sys.modules['nvdiffrast.torch']
    monkeypatch.setitem(sys.modules,'lib_mhr.body_pose',SimpleNamespace(compact_model_params_to_cont_body_np=lambda a:np.zeros((24,260),np.float32),compact_model_params_to_cont_hand_np=lambda a:np.repeat(a,2,axis=1)))
    monkeypatch.setitem(sys.modules,'lib_mhr.hand_surface_contact',SimpleNamespace(MHR_HAND_ORDER=('left_hand','right_hand')))
    monkeypatch.setitem(sys.modules,'lib_mhr.postopt_crop',SimpleNamespace(build_postopt_crop=lambda *a:np.zeros(1),stack_postopt_crops=lambda a,k:{}))
    monkeypatch.setattr(study,'articulation_metrics',lambda *a:{'passed':True});monkeypatch.setattr(study,'select_hand_triangle',lambda *a:0)
    monkeypatch.setattr(study,'sentinel_texture_metrics',lambda *a:{'passed':True});monkeypatch.setattr(study,'translation_observability',lambda *a:{'passed':True})
    def select(v,f,r,t,k,mask,valid,**kwargs):
        events.append('queries');assert r.dtype==t.dtype==np.float32
        points=np.zeros((8,3));points[:,0]=np.linspace(-.01,.01,8)
        qarrays=(points,np.zeros((8,3)),np.zeros((8,2),np.int64),np.zeros(8,np.int64),np.ones(8),np.full((8,3),1/3))
        return SimpleNamespace(queries=SurfaceQueries(*qarrays),diagnostics=SimpleNamespace(scalar_report=lambda:{'qualified_candidates':8,'distinct_canonical_witnesses':8,'raycast_complete':True}))
    stored={}
    def tiny_save(stream,**arrays):
        # No full-size rendered test media on disk: verify shape/call binding,
        # retain only small geometry arrays needed for the exact arithmetic check.
        keys=('object_pose','vertices','object_vertices')
        stored[Path(stream.name).name]={k:arrays[k].copy() for k in keys if k in arrays}
        stream.write(full.fingerprint({k:(a.shape,a.dtype.str) for k,a in arrays.items()}).encode())
    monkeypatch.setattr(selectors,'canonical_mask_quantile_queries',select);monkeypatch.setenv('WR_CODE',str(ROOT))
    monkeypatch.setattr(np,'savez',tiny_save)
    monkeypatch.setattr(q,'runtime',lambda code:SimpleNamespace(identity=rt.identity));report={}
    q.component_native(np,torch,layer,optimizer,spec,json.loads((ROOT/q.PROTOCOL).read_text()),tmp_path,report,lambda:None,lambda:None)
    assert events==['decode24','contact24','queries']*6 and len(renders)==12 and sum(len(a) for a in renders)==36
    assert report['native_decode_calls']==6 and report['native_decoded_frames']==144 and report['sentinel_composites']==18 and report['primitive_render_calls']==12
    assert report['component_preflight_verified'] is True and report['optimizer_constructors']==report['optimizer_updates']==0
    assert len(list(tmp_path.iterdir()))==14  # Sixfull24 components/sixsentinels/twometadata; mainsealsreport15th.
    for i in range(6):
        arrays=stored[f'scene_{i:02d}_components.npz'];assert arrays['object_pose'].dtype==np.float32 and arrays['vertices'].shape[0]==24
        pose=arrays['object_pose'];native=(arrays['object_vertices'][None]@pose[[0,11,23],:3,:3].transpose(0,2,1)+pose[[0,11,23],None,:3,3])
        assert np.array_equal(native,renders[2*i+1])

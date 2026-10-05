"""Procedural CPU contracts only; no model, RGB manufacture or Azure run."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import proposal_stress_render as render

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return render.load_config(REPO)


def test_new_frozen_plan_has_two_frames_constant_identity_shapes_and_objects(config):
    plan=render.recipe(config)
    assert plan==render.recipe(config)
    assert len(plan)==16 and [p['split'] for p in plan]==['DEV']*8+['reserved']*8
    assert {len(p['humans']) for p in plan}=={2,3}
    for scene in plan:
        assert len(scene['objects'])==4
        assert len({p['identity_coefficient'] for p in scene['humans']})==len(scene['humans'])
        assert {r['form'] for r in scene['objects']}=={'cuboid','cylinder','ellipsoid'}
    rows=render.frame_plan(config,plan)
    assert len(rows)==32 and len({r['image_id'] for r in rows})==32
    assert rows==render.frame_plan(config,plan)
    assert sorted((r['scene'],r['frame']) for r in rows)==[(i,j)for i in range(16)for j in range(2)]
    assert [r['image_id'] for r in rows]==sorted(r['image_id'] for r in rows)
    assert config['resamples']==0


@pytest.mark.parametrize('change',[{'scenes':15},{'resamples':1},{'budget_seconds':301},
    {'minimum_visible_human_pixels':1},{'seed':2026100501},{'scope':'ownership'}, {'extra':True}])
def test_runtime_recipe_retuning_rejected(config,tmp_path,change):
    root=tmp_path; (root/'configs').mkdir(); config.update(change)
    (root/render.CONFIG).write_text(json.dumps(config))
    with pytest.raises(ValueError):render.load_config(root)


@pytest.mark.parametrize('form',['cuboid','cylinder','ellipsoid'])
def test_own_primitive_nonzero_closed_manifold_without_download(form):
    v,f=render.primitive(form)
    assert v.dtype==np.float64 and f.dtype==np.int64
    assert v.shape[1:]==(3,) and f.shape[1:]==(3,) and np.isfinite(v).all()
    tri=v[f]; normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    assert np.all(np.linalg.norm(normals,axis=1)>0)
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    assert np.all(counts==2)
    v2,f2=render.primitive(form);np.testing.assert_array_equal(v,v2);np.testing.assert_array_equal(f,f2)


def test_unknown_primitive_rejected():
    with pytest.raises(ValueError):render.primitive('previous_object')


def native_names():
    names=[f'parameter_{i}'for i in range(249)]
    for slot,name in zip((11,31,7,45),('l_uparm_ry','r_uparm_ry','l_elbow_bend','r_elbow_bend')):
        names[slot]=name
    return names


def test_named_controls_native_full_batch_shapes_constant_over_two_frames(config):
    names=native_names();limits=np.tile([-1.,1.],(249,1));plan=render.recipe(config)
    params,shapes,refs=render.controls(names,limits,plan)
    assert params.shape==(80,204) and shapes.shape==(80,45) and len(refs)==80
    assert params.dtype==shapes.dtype==np.float32
    slots=[names.index(n)for n in ('l_uparm_ry','r_uparm_ry','l_elbow_bend','r_elbow_bend')]
    fixed=np.setdiff1d(np.arange(204),slots)
    assert np.count_nonzero(params[:,fixed])==0
    assert np.array_equal(shapes[::2],shapes[1::2])
    assert not np.array_equal(params[::2],params[1::2])
    np.testing.assert_array_equal(limits,np.tile([-1.,1.],(249,1)))


def test_native_unbounded_limits_are_valid_not_a_toy_finite_bounds_gate(config):
    limits=np.tile([-np.inf,np.inf],(249,1))
    params,shapes,refs=render.controls(native_names(),limits,render.recipe(config))
    assert np.isfinite(params).all() and np.isfinite(shapes).all() and len(refs)==80


@pytest.mark.parametrize('fault',['missing','duplicate','bounds','neutral','nan','shape','outside_pose'])
def test_actual_abi_fails_before_forward_without_pose_clipping(config,fault):
    names=native_names();limits=np.tile([-1.,1.],(249,1))
    if fault=='missing':names[11]='unknown'
    elif fault=='duplicate':names[-1]=names[0]
    elif fault=='bounds':limits[0]=[1,-1]
    elif fault=='neutral':limits[0]=[.1,1]
    elif fault=='nan':limits[0,0]=np.nan
    elif fault=='shape':limits=limits.T
    else:limits[names.index('l_elbow_bend')]=[-1,.01]
    with pytest.raises(ValueError):render.controls(names,limits,render.recipe(config))


def test_fixed_reference_origin_not_perframe_alignment_and_every_mesh_retained(config):
    scene=render.recipe(config)[0]; original=copy.deepcopy(scene)
    neutral=np.array([[-.2,-.8,-.1],[.2,-.8,-.1],[.2,.8,.1],[-.2,.8,.1]])
    topology=np.array([[0,1,2],[0,2,3]],np.int64)
    humans={}
    for p in range(2):
        humans[(0,p,0)]=neutral.copy()
        humans[(0,p,1)]=neutral+np.array([.2,0,0])
    v0,f0,c0,l0,i0=render.combine_scene(scene,0,humans,topology)
    v1,f1,c1,l1,i1=render.combine_scene(scene,1,humans,topology)
    assert scene==original
    np.testing.assert_array_equal(f0,f1);np.testing.assert_array_equal(l0,l1)
    assert len(i0)==len(i1)==6 and len(f0)==2*len(topology)+sum(len(render.primitive(o['form'])[1])for o in scene['objects'])
    assert np.all(v0[:,2]>.05) and np.all(v1[:,2]>.05)
    assert i0[0]['fixed_origin_m']==i1[0]['fixed_origin_m']
    # Authored 20cm motion is retained, not removed by recentering frame1.
    delta=(v1[:4]-np.array(i1[0]['translation_m'])).mean(0)
    expected=(neutral+np.array([.2,0,0])-neutral.mean(0))*scene['humans'][0]['scale']
    np.testing.assert_allclose(delta,(expected@render.rotation_y(scene['humans'][0]['yaw']+.04).T).mean(0))
    assert c0.shape==v0.shape and c1.shape==v1.shape


def test_all_objects_and_background_humans_remain_even_zero_visible_pixels():
    face=np.full((480,640),-1,np.int64);face[10,10]=0;face[20,20]=1
    ids=np.array([0,2],np.int32)
    label,masks,counts=render.visible_entities(face,ids,6)
    assert masks.shape==(6,480,640) and masks.dtype==np.bool_
    assert counts.tolist()==[1,0,1,0,0,0]
    assert np.all(label[face==-1]==-1)
    assert counts.sum()==2


@pytest.mark.parametrize('ids,entities',[(np.array([-1]),2),(np.array([3]),2),
    (np.array([0.]),2),(np.empty(0,np.int64),2),(np.array([0]),0)])
def test_invalid_original_entity_inventory_cannot_alias_truth(ids,entities):
    with pytest.raises(ValueError):render.visible_entities(np.full((480,640),-1,np.int64),ids,entities)


@pytest.mark.parametrize('fault',['shape','range','negative','float'])
def test_invalid_raster_no_background_negative_index_alias(fault):
    face=np.full((480,640),-1,np.int64)
    if fault=='shape':face=face[:16,:16]
    elif fault=='range':face[0,0]=3
    elif fault=='negative':face[0,0]=-2
    else:face=face.astype(float)
    with pytest.raises(ValueError):render.visible_entities(face,np.array([0,1]),2)


def public_rows(config):
    return [dict(image_id=r['image_id'],file=r['file'],bytes=123,sha256='a'*64,width=640,height=480)
            for r in render.frame_plan(config,render.recipe(config))]


def test_public_has_exact_rgb_fields_and_no_recipe_mapping(config):
    rows=public_rows(config);value=render.public_manifest(rows)
    assert set(value)=={'schema','images'} and value['schema']=='world_reward.rgb_proposal_inputs.v1'
    assert all(set(r)=={'image_id','file','bytes','sha256','width','height'}for r in rows)
    text=json.dumps(value)
    assert not any(word in text for word in ('DEV','reserved','scene','frame','human','object','camera','recipe','split'))


@pytest.mark.parametrize('fault',['private','missing','duplicate','filename','dimensions','hash'])
def test_public_leak_or_partial_cohort_rejected(config,fault):
    rows=public_rows(config)
    if fault=='private':rows[0]['split']='DEV'
    elif fault=='missing':rows.pop()
    elif fault=='duplicate':rows[0]['image_id']=rows[1]['image_id']
    elif fault=='filename':rows[0]['file']='../eval_private/truth.png'
    elif fault=='dimensions':rows[0]['width']=1024
    else:rows[0]['sha256']='bad'
    with pytest.raises(ValueError):render.public_manifest(rows)


def test_source_and_wrapper_are_scoped_offline_model_once_and_owned_cleanup_only():
    source=Path(render.__file__).read_text();tree=ast.parse(source)
    wrapper=(REPO/'infra/run_proposal_stress_render.sh').read_text()
    assert source.count('torch.jit.load(')==1
    assert 'model(ti,tp,te,True)' in source and 'recipe_supplied_in_public_inputs=False' in source
    assert '--network none' in wrapper and '--read-only' in wrapper and '--gpus all' in wrapper
    assert 'flock --nonblock 9' in wrapper and '--cap-drop ALL' in wrapper
    assert '305s docker run' in wrapper and 'timeout --signal=TERM --kill-after=2s 8s' in wrapper
    assert 'world_reward.proposal_stress.owner=$REV-$BEFORE' in wrapper
    assert '--finalize "$RESULT" --expected-binding "$BEFORE"' in wrapper
    assert all(bad not in wrapper for bad in ('docker system prune','src=$ROOT/data','src=$ROOT/outputs','src=$DEST,dst='))
    assert all(old not in source for old in ('hand_synthetic_render','bridge_rgb_anchor_render','hocap','openimages','triangle_ray_gate'))
    assert all(h in source for h in render.HELPERS)
    assert any(isinstance(n,ast.FunctionDef)and n.name=='render_rgb'for n in ast.walk(tree))


def test_tiny_private_npz_is_immutable_exclusive_and_roundtrips(tmp_path):
    path=tmp_path/'tiny.npz';render.save_npz(path,faces=np.array([[0,1,2]],np.int64))
    assert path.stat().st_mode&0o777==0o400
    with np.load(path,allow_pickle=False)as bank:np.testing.assert_array_equal(bank['faces'],[[0,1,2]])
    with pytest.raises(FileExistsError):render.save_npz(path,faces=np.array([[2,1,0]],np.int64))


@pytest.mark.parametrize('fault',[None,'missing','revision','url','malformed'])
def test_actual_installed_pytorch3d_commit_not_constant_or_version_only(config,fault):
    from types import SimpleNamespace
    direct=dict(url='https://github.com/facebookresearch/pytorch3d.git',
        vcs_info=dict(commit_id=config['pytorch3d_revision'],requested_revision='v0.7.9',vcs='git'))
    if fault=='revision':direct['vcs_info']['commit_id']='b'*40
    if fault=='url':direct['url']='https://example.org/pytorch3d.git'
    raw=None if fault=='missing'else '{' if fault=='malformed'else json.dumps(direct)
    distribution=SimpleNamespace(read_text=lambda filename:raw)
    if fault is None:assert render.installed_pytorch3d(distribution,config['pytorch3d_revision'])==direct
    else:
        with pytest.raises(ValueError):render.installed_pytorch3d(distribution,config['pytorch3d_revision'])

"""Own procedural/manifest tests, no local model/render/decode."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    infra=Path(__file__).parents[1]/'infra'; monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('joint_render_test',infra/'joint_rgb_render.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_public_manifest_only_rgb_order_grid(render):
    manifest=render.public_manifest(['a'*64]*9)
    assert set(manifest)=={'schema','images'} and manifest['schema']=='world-reward-joint-rgb-v1'
    assert [x['file'] for x in manifest['images']]==[f'clip_{c:02d}_frame_{f:03d}.png' for c in range(3) for f in range(3)]
    assert all(set(x)=={'file','sha256','width','height'} and (x['width'],x['height'])==(1024,768) for x in manifest['images'])


@pytest.mark.parametrize('values',[[],['a'*64]*8,['a'*64]*10,['z'*64]*9,[None]*9])
def test_public_manifest_rejects_bad_digest_or_coverage(render,values):
    with pytest.raises(ValueError): render.public_manifest(values)


def test_own_bottle_closed_edge_oriented_positive_volume(render):
    v,f=render.bottle_mesh()
    directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    assert np.all(counts==2)
    assert np.all(np.bincount(inverse,weights=np.where(directed[:,0]<directed[:,1],1,-1))==0)
    assert np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()>0
    assert len(v)==194 and len(f)==384


def control_fixture(render):
    names=[f'unused_{i}' for i in range(249)]
    required=[f'{side}_{suffix}' for side in 'lr' for suffix in ('uparm_ry','elbow_bend','index1_rz','middle1_rz','ring1_rz','pinky1_rz')]
    for i,name in enumerate(required): names[i]=name
    return names,np.tile([-1.,1.],(249,1))


def test_named_control_values_fixed_scales_and_identity_not_guessed_indices(render):
    names,bounds=control_fixture(render); controls,changes=render.named_controls(names,bounds)
    assert controls.shape==(9,204) and np.count_nonzero(controls[:,136:])==0
    assert len(changes)==54 and all(x['name']==names[x['column']] for x in changes)
    assert np.count_nonzero(controls[0])==6 and not np.array_equal(controls[0],controls[1])
    assert controls[0,names.index('l_elbow_bend')]==pytest.approx(.30)


@pytest.mark.parametrize('kind',['missing','duplicate','limit','neutral','shape','nan'])
def test_invalid_named_limits_fail_not_clipped(render,kind):
    names,bounds=control_fixture(render)
    if kind=='missing': names[0]='absent'
    elif kind=='duplicate': names[-1]=names[0]
    elif kind=='limit': bounds[0]=[-1,.05]
    elif kind=='neutral': bounds[50]=[.1,1]
    elif kind=='shape': bounds=bounds[:204]
    else: bounds[0,0]=np.nan
    with pytest.raises(ValueError): render.named_controls(names,bounds)


def test_fixed_camera_one_neutral_extent_and_no_focal_truth_in_public(render):
    neutral=np.array([[-.4,-.9,-.1],[.4,.9,.1],[0,0,0]])
    t=render.fixed_camera(neutral); shifted=render.fixed_camera(neutral+[2,3,4])
    np.testing.assert_allclose(neutral+t,neutral+[2,3,4]+shifted)
    assert render.FOCALS==(1280.,960.,1600.)


def test_render_truth_schema_and_private_mounts(render):
    assert render.TRUTH_KEYS=={'human_vertices_camera_m','human_faces','object_vertices_camera_m','object_faces',
        'camera_K','scene_depth_m','visible_face_indices','clip_index','frame_index'}
    wrapper=Path(render.__file__).with_name('run_joint_rgb_render.sh').read_text()
    assert '123s docker run' in wrapper and '--network none' in wrapper
    assert 'chown scenesmith:scenesmith "$DEST"' in wrapper and 'src=$DEST,dst=$DEST' in wrapper
    assert 'src=$ROOT/validation,dst=' not in wrapper and 'src=$ROOT/outputs' not in wrapper
    assert 'src=$ROOT/weights/mhr' in wrapper and 'src=$ROOT/weights,dst=' not in wrapper
    subprocess.run(['bash','-n',str(Path(render.__file__).with_name('run_joint_rgb_render.sh'))],check=True)


def test_render_accepts_no_override_or_oracle(render):
    with pytest.raises(SystemExit): render.main(['--camera-focal','1280'])

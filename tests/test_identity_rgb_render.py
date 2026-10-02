"""Own geometry/native-control contracts, no local model/GT/render execution."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra";monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location("identity_render_test",infra/"identity_rgb_render.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_fifteen_rgb_only_manifest_order_and_no_truth(render):
    data=render.public_manifest(["a"*64]*15)
    assert set(data)=={"schema","images"} and data["schema"]=="world-reward-identity-rgb-v1"
    assert [r["file"] for r in data["images"]]==[f"clip_{c:02d}_frame_{f:03d}.png" for c in range(3) for f in range(5)]
    assert all(set(r)=={"file","sha256","width","height"} and (r["width"],r["height"])==(1024,768) for r in data["images"])


@pytest.mark.parametrize("hashes",[[],["a"*64]*14,["a"*64]*16,["G"*64]*15,[None]*15])
def test_bad_manifest_hash_coverage(render,hashes):
    with pytest.raises(ValueError):render.public_manifest(hashes)


def metadata():
    names=[f"unused_{i}" for i in range(249)]
    required=[f"{side}_{suffix}" for side in "lr" for suffix in ("uparm_ry","elbow_bend","wrist_ry","index1_rz","middle1_rz","ring1_rz","pinky1_rz")]
    for i,name in enumerate(required):names[i]=name
    return names,np.tile([-1.,1.],(249,1))


def test_clip_fixed_nonzero_native_shape_and_all68scale_controls(render):
    names,limits=metadata();controls,identity,changes=render.named_controls(names,limits)
    assert controls.shape==(15,204) and identity.shape==(15,45) and controls.dtype==identity.dtype==np.float32
    assert len(changes)==15*8
    for clip in range(3):
        segment=slice(clip*5,(clip+1)*5)
        assert np.all(controls[segment,136:]==np.float32(render.SCALES[clip]))
        assert np.all(identity[segment,:2]==np.asarray(render.SHAPES[clip],np.float32))
        assert np.all(identity[segment,2:]==0)
        assert not np.array_equal(controls[clip*5,:136],controls[clip*5+4,:136])
    assert all(row["name"]==names[row["column"]] and row["column"]<136 for row in changes)
    assert len(set(controls[:,136:].tobytes()[i*68*4:(i+1)*68*4] for i in (0,5,10)))==3


@pytest.mark.parametrize("bad",["duplicate","empty","missing","shape","nan","neutral","scale","identity","unused_identity_zero","pose"])
def test_bad_native_metadata_abstains_no_clipping(render,bad):
    names,limits=metadata()
    if bad=="duplicate":names[-1]=names[0]
    if bad=="empty":names[-1]=""
    if bad=="missing":names[0]="unknown"
    if bad=="shape":limits=limits[:204]
    if bad=="nan":limits[0,0]=np.nan
    if bad=="neutral":limits[30]=[.1,1.]
    if bad=="scale":limits[150]=[-.02,.02]
    if bad=="identity":limits[204]=[-.1,.1]
    if bad=="unused_identity_zero":limits[207]=[.1,1.]
    if bad=="pose":limits[0]=[-1.,.1]
    with pytest.raises(ValueError):render.named_controls(names,limits)


def test_every_animated_and_neutral249_value_obeys_limits(render):
    names,bounds=metadata();controls,identity,_=render.named_controls(names,bounds)
    full=np.c_[controls,identity];neutral=full[::5].copy();neutral[:,:136]=0
    assert full.shape==(15,249) and neutral.shape==(3,249)
    assert np.all(full>=bounds[:,0]) and np.all(full<=bounds[:,1])
    assert np.all(neutral>=bounds[:,0]) and np.all(neutral<=bounds[:,1])
    assert np.array_equal(neutral[:,136:],full[::5,136:])


def test_fixed_framing_is_translation_invariant_and_not_per_frame(render):
    v=np.array([[-.4,-.9,-.1],[.4,.9,.1],[0,0,0]])
    values=np.stack([v,v*1.05,v*.9]);center,distance,floor=render.fixed_framing(values)
    shifted=render.fixed_framing(values+[2,3,4])
    assert shifted[0]==pytest.approx(center+[2,3,4]) and shifted[1:]==pytest.approx((distance,floor))
    assert render.FOCALS==(1160.,1480.,1720.)


def test_own_bottle_closed_oriented_positive_volume(render):
    v,f=render.bottle_mesh();edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,inverse,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    assert np.all(count==2) and np.all(np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))==0)
    assert np.einsum("ij,ij->i",v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()>0


def test_fresh_background_finite_textured_original_geometry(render):
    v,f,c=render.background(4.,1.)
    assert v.shape==c.shape==(640,3) and f.shape==(320,3)
    assert np.isfinite(v).all() and np.isfinite(c).all() and c.min()>0 and c.max()<1
    assert np.any(np.ptp(c,axis=0)>.05) and np.min(v[:,2]+4.)>=2.4
    assert np.all(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)>0)


def test_scene_moves_geometry_without_changing_topology_and_truth_order(render):
    human=np.array([[-.1,-.5,0],[.1,.5,.1],[0,.3,-.1]])
    human=np.tile(human,(40,1));faces=np.array([[0,1,2]],np.int64)
    mask=np.ones(len(human),bool);regions={s:{"vertex_mask":mask} for s in "lr"}
    first=render.scene(human,faces,regions,np.zeros(3),4.,1.,0,0)
    last=render.scene(human,faces,regions,np.zeros(3),4.,1.,0,4)
    hv,ov,of,sv,sf,colors,K=first
    assert np.array_equal(sf[:len(faces)],faces) and np.array_equal(sf[len(faces):len(faces)+len(of)],of+len(human))
    assert not np.array_equal(hv,last[0]) and not np.array_equal(ov,last[1])
    assert np.array_equal(sf,last[4]) and np.min(sv[:,2])>.01
    assert K==pytest.approx(np.array([[1160,0,512],[0,1160,384],[0,0,1]]))


def test_foreground_truth_removes_room_ids_but_preserves_occlusion(render,monkeypatch):
    monkeypatch.setattr(render,"WIDTH",32);monkeypatch.setattr(render,"HEIGHT",16)
    face=np.full((16,32),9,np.int64);face.flat[:64]=0;face.flat[64:128]=2
    depth=np.full(face.shape,3.,np.float32)
    result,z=render.foreground_truth(face,depth,np.zeros((2,3),int),np.zeros((3,3),int))
    assert np.count_nonzero(result==0)==64 and np.count_nonzero(result==2)==64
    assert np.all(result.flat[128:]==-1) and np.isnan(z.flat[128:]).all() and np.all(z.flat[:128]==3)
    assert result.dtype==np.int64 and z.dtype==np.float32


@pytest.mark.parametrize("bad",["human","object","negative_depth","nan_depth","shape"])
def test_visibility_and_foreground_depth_guards(render,monkeypatch,bad):
    monkeypatch.setattr(render,"WIDTH",32);monkeypatch.setattr(render,"HEIGHT",16)
    face=np.full((16,32),9,np.int64);face.flat[:64]=0;face.flat[64:128]=2;depth=np.full(face.shape,3.,np.float32)
    if bad=="human":face.flat[:64]=9
    if bad=="object":face.flat[64:128]=9
    if bad=="negative_depth":depth.flat[0]=-1
    if bad=="nan_depth":depth.flat[0]=np.nan
    if bad=="shape":face=face[:8]
    with pytest.raises(ValueError):render.foreground_truth(face,depth,np.zeros((2,3),int),np.zeros((3,3),int))


def test_prepare_serial_two_offline_runs_scoped_truth_firewall(render):
    wrapper=Path(render.__file__).with_name("run_identity_rgb_prepare.sh");source=wrapper.read_text()
    assert source.count("--network none")==2 and "123s docker run" in source and "183s docker run" in source
    assert "identity_rgb_render.py" in source and "identity_rgb_masks.py" in source
    masks=source.split('MASK_IMAGE=')[1]
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in masks and 'src=$OUT,dst=$OUT' in masks
    assert 'src=$BASE,dst=$BASE' not in masks and 'eval_private' not in masks and 'src=$ROOT/weights/mhr' not in masks
    assert '-L "$ROOT/validation"' in source and '-L "$BASE"' in source
    assert render.TRUTH_KEYS=={"human_vertices_camera_m","human_faces","object_vertices_camera_m","object_faces","camera_K",
                              "scene_depth_m","visible_face_indices","clip_index","frame_index"}
    subprocess.run(["bash","-n",str(wrapper)],check=True)


def test_no_cli_override_to_tune_samecohort(render):
    with pytest.raises(SystemExit):render.main(["--focal","1280"])

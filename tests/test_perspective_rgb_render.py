"""Tiny own room/camera/manifest tests, no local rendering or model assets."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra";monkeypatch.syspath_prepend(str(infra));monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec=importlib.util.spec_from_file_location("own_perspective_render",infra/"perspective_rgb_render.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_publicmanifest_only_rgb_grid_and_order(render):
    value=render.public_manifest(["a"*64]*9)
    assert set(value)=={"schema","images"} and value["schema"]=="world-reward-perspective-rgb-v1"
    assert [x["file"] for x in value["images"]]==[f"case_{i:02d}.png" for i in range(9)]
    assert all(set(x)=={"file","sha256","width","height"} and (x["width"],x["height"])==(1024,768) for x in value["images"])


@pytest.mark.parametrize("digests",[[],["a"*64]*8,["a"*64]*10,["x"*64]*9,[None]*9])
def test_publicmanifest_rejects_partial_or_invalid(render,digests):
    with pytest.raises(ValueError):render.public_manifest(digests)


def test_camera_frozen3x2_gravity_and_principal_conventions(render):
    K,R,g,strong=render.camera_truth()
    assert K.shape==(9,3,3) and K.dtype==np.float64 and g.shape==(9,3) and g.dtype==np.float64
    assert strong.dtype==bool and strong.tolist()==[True]*6+[False]*3
    np.testing.assert_array_equal(K[:,0,0],[1024,1024,1408,1408,1792,1792,1024,1408,1792])
    np.testing.assert_array_equal(K[:,0,2],np.full(9,512));np.testing.assert_array_equal(K[:,1,2],np.full(9,384))
    np.testing.assert_allclose(R@R.transpose(0,2,1),np.broadcast_to(np.eye(3),(9,3,3)),atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R),1,atol=1e-15)
    np.testing.assert_allclose(g,np.einsum("bij,j->bi",R,[0.,-1.,0.]),atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(g,axis=1),1,atol=1e-15)


def test_tessellated_room_openfront_furniture_finite_positive_scene(render):
    K,R,_,_=render.camera_truth();v,f,c=render.room_mesh(8.,1.0)
    assert v.shape[1]==3 and f.shape[1]==3 and c.shape==v.shape
    assert f.dtype==np.int64 and np.all(f>=0) and f.max()<len(v)
    assert np.isfinite(v).all() and np.all((c>=0)&(c<=1))
    for rotation in R:
        camera=v@rotation.T+[0,0,8.]
        assert camera[:,2].min()>.01
    front=2.5-8
    assert not np.any(np.all(np.isclose(v[f,2],front),axis=1))  # No front wall blocks perspective cues.
    assert np.any(np.ptp(c,axis=0)>.1)


def control_fixture():
    names=[f"unused_{i}" for i in range(249)]
    required=[f"{side}_{suffix}" for side in "lr" for suffix in ("uparm_ry","elbow_bend","index1_rz","middle1_rz","ring1_rz","pinky1_rz")]
    for i,name in enumerate(required):names[i]=name
    return names,np.tile([-1.,1.],(249,1))


def test_freshnamed_poses_identity_and_no_scale_indexguess(render):
    names,bounds=control_fixture();controls,shape,changes=render.named_controls(names,bounds)
    assert controls.shape==(9,204) and shape.shape==(9,45) and controls.dtype==shape.dtype==np.float32
    assert np.count_nonzero(controls[:,136:])==0 and np.count_nonzero(shape[:,1:])==0
    assert len(changes)==54 and all(names[x["column"]]==x["name"] for x in changes)
    np.testing.assert_allclose(shape[:6,0],np.repeat(render.SHAPES,2))
    assert not np.array_equal(controls[0],controls[1])


@pytest.mark.parametrize("fault",["missing","duplicate","limits","identity","neutral","nan"])
def test_invalid_reference_names_or_limits_fail(render,fault):
    names,bounds=control_fixture()
    if fault=="missing":names[0]="missing"
    elif fault=="duplicate":names[-1]=names[0]
    elif fault=="limits":bounds[0]=[-1,.1]
    elif fault=="identity":bounds[204]=[-.05,.05]
    elif fault=="neutral":bounds[30]=[.1,1]
    else:bounds[0,0]=np.nan
    with pytest.raises(ValueError):render.named_controls(names,bounds)


def test_fixed_neutral_framing_translation_invariance(render):
    neutral=np.array([[-.4,-.9,-.1],[.4,.9,.1],[0,0,0]],float);_,R,_,_=render.camera_truth()
    center,distance,floor=render.fixed_framing(neutral,R)
    center2,distance2,floor2=render.fixed_framing(neutral+[3,4,5],R)
    np.testing.assert_allclose(center2-center,[3,4,5]);assert distance==pytest.approx(distance2) and floor==pytest.approx(floor2)
    for rotation in R:
        camera=(neutral-center)@rotation.T+[0,0,distance]
        pixels=render.render.project_camera_points(camera,np.array([[1792,0,512],[0,1792,384],[0,0,1]]))
        assert np.all(pixels>16) and np.all(pixels<[1008,752])


def test_privateonly_calibration_schema_and_scopedmounts(render):
    assert render.TRUTH_KEYS=={"camera_K","gravity","strong"}
    wrapper=Path(render.__file__).with_name("run_perspective_rgb_render.sh");text=wrapper.read_text();source=Path(render.__file__).read_text()
    subprocess.run(["bash","-n",str(wrapper)],check=True)
    assert "123s" in text and "--network none" in text and "--memory 12g" in text
    assert "src=$DEST,dst=$DEST" in text and 'chown 1000:1000 "$DEST"' in text
    assert "src=$ROOT/validation,dst=" not in text and "src=$ROOT/outputs" not in text
    assert "src=$ROOT/weights/mhr" in text and "source_image_id" in source and "optimized_execution(False)" in source
    assert '"inference_performed":False' in source and '"challenge_inputs_used":False' in source


def test_no_percase_camera_or_prompt_override(render):
    with pytest.raises(SystemExit):render.main(["--focal","1280"])

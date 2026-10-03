"""Own tiny procedural fixtures; no learned inference, assets or private data."""
import ast
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"infra"));monkeypatch.syspath_prepend(str(REPO/"src"))
    spec=importlib.util.spec_from_file_location("factorial_rgb_render_test",REPO/"infra/factorial_rgb_render.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def metadata():
    names=["unused"+str(i)for i in range(249)]
    requested=["l_uparm_ry","l_elbow_bend","l_wrist_ry","r_uparm_ry",*["l_"+f+"1_rz"for f in("index","middle","ring","pinky")]]
    names[:len(requested)]=requested;bounds=np.tile([-1.,1.],(249,1));bounds[136:204]=[-.1,.1]
    return names,bounds


def test_factor_recipe_and_same_three_articulations(renderer):
    names,bounds=metadata();bounds[136+np.array([0,1,2,11,12,15,16])]=0
    controls,shape,changes=renderer.named_controls(names,bounds)
    assert controls.shape==(6,204) and shape.shape==(6,45) and len(changes)==48
    assert np.array_equal(controls[:3,:136],controls[3:,:136])
    assert len({row.tobytes()for row in controls[:3,:136]})==3
    for morphology in range(2):
        assert np.array_equal(shape[morphology*3:(morphology+1)*3,:2],np.tile(np.array(renderer.SHAPES[morphology],np.float32),(3,1)))
    full=np.c_[controls,shape];assert np.all(full>=bounds[:,0]) and np.all(full<=bounds[:,1])
    assert np.all(controls[:,136+np.array([0,1,2,11,12,15,16])]==0)
    assert [renderer.factors(g)for g in range(8)]==[(m,a,o)for m in range(2)for a in range(2)for o in range(2)]


@pytest.mark.parametrize("fault",["masked","names","missing","bounds","locked","shape","articulation"])
def test_recipe_has_no_parameter_clipping_or_repair(renderer,fault):
    names,bounds=metadata()
    if fault=="masked":bounds=np.ma.array(bounds,mask=False)
    elif fault=="names":names[-1]=names[0]
    elif fault=="missing":names[0]="absent"
    elif fault=="bounds":bounds[140]=[-.01,.01]
    elif fault=="locked":bounds[136:204]=0
    elif fault=="shape":bounds[204]=[-.01,.01]
    else:bounds[names.index("l_elbow_bend")]=[-1.,.2]
    with pytest.raises(ValueError):renderer.named_controls(names,bounds)


def test_appearance_occlusion_do_not_modify_human_geometry(renderer):
    x=np.linspace(-.3,.3,200);human=np.c_[x,np.linspace(-.8,.8,200),.04*np.sin(x*8)]
    faces=np.array([[0,1,2],[197,198,199]],np.int32);before=human.copy()
    center,distance,floor=renderer.fixed_framing(np.stack([human,human*1.03]));elbow=np.array([-.12,-.1,0.])
    bottle,bf=renderer.primitives.bottle_mesh()
    for frame in range(3):
        scenes=[renderer.scene(human,faces,elbow,center,distance,floor,g,frame)for g in range(4)]
        for other in scenes[1:]:assert np.array_equal(other[0],scenes[0][0])
        assert np.array_equal(scenes[0][1],scenes[2][1]) and np.array_equal(scenes[1][1],scenes[3][1])
        assert np.allclose(scenes[1][1]-scenes[0][1],[0.,0.,.5])
        assert not np.array_equal(scenes[0][5][:200],scenes[2][5][:200])
        for result in scenes:
            assert np.array_equal(result[2],bf) and np.array_equal(result[-1],renderer.COHORT.fixed_K)
            assert np.isclose(np.linalg.norm(result[1][0]-result[1][100]),renderer.BOTTLE_SCALE*np.linalg.norm(bottle[0]-bottle[100]))
    assert np.array_equal(human,before) and renderer.primitives.BASE=="validation/identity_rgb_v2"


def test_single_framing_translation_invariant(renderer):
    points=np.array([[-.3,-.8,-.1],[.3,.8,.1],[.1,-.5,0.]])
    a=renderer.fixed_framing(np.stack([points,points*1.1]));b=renderer.fixed_framing(np.stack([points,points*1.1])+[2.,3.,4.])
    assert np.allclose(b[0],a[0]+[2.,3.,4.]) and np.isclose(a[1],b[1]) and np.isclose(a[2],b[2])


def test_native_landmark_mapping_before_camera_without_unit_sum_assumption(renderer):
    v=np.zeros((1,18439,3),np.float32);v[0,0]=[100.,200.,300.]
    sk=np.zeros((1,127,8),np.float32);sk[0,0,:3]=[400.,500.,600.]
    mapping=np.zeros((308,18566),np.float32);mapping[0,0]=2.;mapping[1,18439]=1.
    original=mapping.copy();j,k=renderer.reference_landmarks(v,sk,mapping)
    assert np.array_equal(k[0,0],[2.,4.,6.]) and np.array_equal(k[0,1],j[0,0])
    assert np.array_equal(mapping,original)
    center=np.array([.2,.3,.4]);actual=renderer.transform_points(k[0]*[1.,-1.,-1.],center,4.,1)
    incorrectly_mapped=renderer.transform_points(v[0]/100*[1.,-1.,-1.],center,4.,1)
    assert not np.allclose(actual[0],2*incorrectly_mapped[0])
    with pytest.raises(ValueError):renderer.reference_landmarks(v,sk,mapping.astype(np.float64))
    with pytest.raises(ValueError):renderer.reference_landmarks(v,sk,np.ma.array(mapping,mask=False))


def test_bundled_rig_geometry_equivalence_does_not_renormalize(renderer):
    v=np.zeros((6,18439,3),np.float32);s=np.zeros((6,127,8),np.float32);s[...,6]=1;s[...,7]=1
    assert renderer.bundled_parity(v,s,v.copy(),s.copy())==dict(vertices_m=0.,joints_m=0.)
    other=s.copy();other[2,9,0]=.01
    with pytest.raises(ValueError):renderer.bundled_parity(v,s,v,other)
    other=v.copy();other[1,0,0]=.01
    with pytest.raises(ValueError):renderer.bundled_parity(v,s,other,s)


def test_occlusion_is_measured_not_assumed(renderer,monkeypatch):
    monkeypatch.setattr(renderer,"WIDTH",16);monkeypatch.setattr(renderer,"HEIGHT",8)
    front=np.full((8,16),100,np.int64);back=front.copy();back[:4]=0
    assert renderer.occlusion_evidence(front,back,10)["newly_visible_human_pixels"]==64
    with pytest.raises(ValueError):renderer.occlusion_evidence(front,front,10)


def test_wrapper_private_firewall_and_no_native_network_inference(renderer):
    wrapper=REPO/"infra/run_factorial_rgb_prepare.sh";subprocess.run(["bash","-n",str(wrapper)],check=True);text=wrapper.read_text()
    assert text.count("docker run")==1 and "--gpus all --network none --memory 32g --cpus 4"in text
    assert "123s"in text and "weights/mhr"in text and "checkpoints/sam-3d-body-dinov3"in text
    assert "vendor/video_to_data/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body"in text
    assert "results/weights-acquisition.json"in text
    assert "grounding"not in text and "validation/factorial_rgb_v1"in text
    assert subprocess.run(["bash",str(wrapper),"--retry"],capture_output=True).returncode==2
    tree=ast.parse(Path(renderer.__file__).read_text())
    imported={n.module for n in tree.body if isinstance(n,ast.ImportFrom)}|{a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names}
    assert not imported&{"torch","pytorch3d","PIL","body_smoke","hand_synthetic_infer"}
    assert not any(isinstance(n,ast.Attribute)and n.attr in("process_one_image","load_sam_3d_body")for n in ast.walk(tree))
    source=Path(renderer.__file__).read_text()
    assert "bundled_forward_attempts=0"in source and "actual_total_native_forward_calls=3"in source
    assert "bundled.get_joint_names()!=joints"in source and "reference_landmarks(raw,sk,mapping)"in source


def test_render_only_closure_bounded_ordinary_sources(renderer,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"infra"));import azure_job
    files={p.relative_to(REPO).as_posix():p.read_bytes()for d in("infra","src","configs")for p in(REPO/d).rglob("*")
        if p.is_file()and"__pycache__"not in p.parts};files["pyproject.toml"]=(REPO/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_factorial_rgb_prepare.sh")
    assert "infra/factorial_rgb_render.py"in paths and "infra/factorial_rgb_protocol.py"in paths
    assert "infra/root5_rgb_render.py"not in paths and "infra/root5_rgb_observe.py"not in paths
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w",format=tarfile.PAX_FORMAT)as archive:
        for path in paths:
            info=tarfile.TarInfo(path);info.size=len(files[path]);archive.addfile(info,io.BytesIO(files[path]))
    encoded,_=azure_job.encoded_runtime_archive(stream.getvalue());assert len(encoded)<=160000

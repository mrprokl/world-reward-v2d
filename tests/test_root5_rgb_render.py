"""Own procedural geometry/recipes only; real rendering is Azure-only."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
INFRA=REPO/"infra"


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd()/"infra"));monkeypatch.syspath_prepend(str(Path.cwd()/"src"));monkeypatch.syspath_prepend(str(INFRA))
    spec=importlib.util.spec_from_file_location("root5_rgb_render_test",INFRA/"root5_rgb_render.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def metadata():
    names=[f"unused{i}"for i in range(249)]
    required=[f"{s}_{n}"for s in"lr"for n in("uparm_ry","elbow_bend","wrist_ry","index1_rz","middle1_rz","ring1_rz","pinky1_rz")]
    names[:len(required)]=required;bounds=np.tile([-1.,1.],(249,1));return names,bounds


def test_fresh_recipe_differs_oldcohort_no_globals_modified(renderer):
    r=renderer
    assert r.SHAPES==((-.19,-.07),(.11,.24),(.43,-.09))and r.SCALES==(-.035,.020,.055)
    assert r.YAWS==(.23,-.17,.08)and r.SIDES==("l","r","l")and r.BOTTLE_DEPTH_OFFSETS==(.11,.04,-.04,-.095,-.14)
    assert r.primitives.BASE=="validation/identity_rgb_v2"and r.primitives.FOCALS==(1160.,1480.,1720.)
    assert r.BASE=="validation/root5_rgb_v1"and r.COHORT.frames==5


def test_full249_animated_and_neutral_legal_clipconstant(renderer):
    names,bounds=metadata();bounds[136:204]=[-.1,.1];locked=np.array([0,1,2,11,12,15,16]);bounds[136+locked]=0
    controls,shape,changes=renderer.named_controls(names,bounds)
    assert controls.shape==(15,204)and shape.shape==(15,45)and controls.dtype==shape.dtype==np.float32
    full=np.c_[controls,shape];neutral=full[::5].copy();neutral[:,:136]=0
    assert(full>=bounds[:,0]).all()and(full<=bounds[:,1]).all()and(neutral>=bounds[:,0]).all()and(neutral<=bounds[:,1]).all()
    assert len(changes)==120 and len({r.tobytes()for r in controls[::5,136:]})==3
    for c,side in enumerate(renderer.SIDES):
        assert np.array_equal(shape[c*5:(c+1)*5,:2],np.tile(np.array(renderer.SHAPES[c],np.float32),(5,1)))
        assert np.all(controls[c*5:(c+1)*5,136+locked]==0)
        for f in range(5):assert controls[c*5+f,names.index(side+"_elbow_bend")]==np.float32(.27+.055*f)


@pytest.mark.parametrize("fault",["duplicate","missing","masked","bounds","nan","free_scale","all_locked","shape","motion"])
def test_no_native_bounds_or_recipe_adjustment(renderer,fault):
    names,bounds=metadata()
    if fault=="duplicate":names[-1]=names[0]
    elif fault=="missing":names[names.index("l_uparm_ry")]="missing"
    elif fault=="masked":bounds=np.ma.array(bounds,mask=False)
    elif fault=="bounds":bounds=bounds[:204]
    elif fault=="nan":bounds[0,0]=np.nan
    elif fault=="free_scale":bounds[140]=[-.01,.01]
    elif fault=="all_locked":bounds[136:204]=0
    elif fault=="shape":bounds[204]=[-.01,.01]
    else:bounds[names.index("l_elbow_bend")]=[-1.,.2]
    with pytest.raises(ValueError):renderer.named_controls(names,bounds)


def actor():
    x=np.linspace(-.35,.35,200);y=np.linspace(-.8,.8,200);v=np.c_[x,y,.05*np.sin(5*x)]
    faces=np.array([[0,1,2],[197,198,199]],np.int32);left=np.arange(200)<60;right=np.arange(200)>=140
    return v,faces,{"l":{"vertex_mask":left},"r":{"vertex_mask":right}}


def test_fixed_framing_translation_invariant_no_camera_fit(renderer):
    v,_,_=actor();c,d,f=renderer.fixed_framing(np.stack([v,v*1.05,v*.9]));other=renderer.fixed_framing(np.stack([v,v*1.05,v*.9])+[3.,-4.,2.])
    assert np.allclose(other[0],c+[3.,-4.,2.])and np.isclose(other[1],d)and np.isclose(other[2],f)


def test_all15_new_scenes_rigid_object_camera_indices_no_input_mutation(renderer):
    from scipy.spatial.transform import Rotation
    human,faces,regions=actor();before=human.copy();center,distance,floor=renderer.fixed_framing(np.stack([human]*3))
    bottle,bf=renderer.primitives.bottle_mesh();seen=[]
    for c in range(3):
        for f in range(5):
            hv,ov,of,sv,sf,colors,K=renderer.scene(human,faces,regions,center,distance,floor,c,f)
            R=Rotation.from_rotvec([0.,renderer.YAWS[c]+.028*(f-2),0.]).as_matrix()
            expected=(human-center)@R.T+[.016*(f-2),.006*np.sin(.9*f),-.024*(f-2)+distance]
            assert np.allclose(hv,expected)and np.array_equal(K,renderer.COHORT.fixed_K)
            assert len(ov)==194 and of.shape==(384,3)and of.dtype==np.int64 and np.array_equal(of,bf)
            assert np.isclose(np.linalg.norm(ov[0]-ov[100]),1.04*np.linalg.norm(bottle[0]-bottle[100]))
            assert np.isfinite(sv).all()and sv[:,2].min()>.01 and sf.min()>=0 and sf.max()<len(sv)
            assert np.allclose(colors[len(hv)],[.37,.30,.20]);seen.append(hv.tobytes())
    assert len(set(seen))==15 and np.array_equal(human,before)and faces.dtype==np.int32


def test_public_manifest_no_private_fields_native_truth_i32_source(renderer):
    manifest=renderer.public_manifest(["a"*64]*15)
    assert set(manifest)=={"schema","images"}and manifest["schema"]==renderer.COHORT.schema
    assert all(set(r)=={"file","sha256","width","height"}for r in manifest["images"])
    source=Path(renderer.__file__).read_text()
    assert"faces.dtype!=np.int32"in source and"human_faces_sha256=sha256_array(faces)"in source
    assert"human_faces=faces"in source and"faces.astype("not in source
    assert"protocol.public_inputs(public,COHORT)"in source


def test_wrapper_render_only_offline_no_observer_private_alias(renderer):
    path=INFRA/"run_root5_rgb_prepare.sh";subprocess.run(["bash","-n",str(path)],check=True);text=path.read_text()
    assert"--gpus all --network none --memory 32g --cpus 4"in text and"123s"in text
    assert text.count("docker run")==1 and"MASK_IMAGE"not in text and"observer"not in text.lower().split("set -euo",1)[1]
    assert"weights/mhr"in text and"mhr-finger-semantics-v4.json"in text and"grounding_dino"not in text
    assert'"$CODE/infra/root5_rgb_render.py"'in text and"validation/root5_rgb_v1"in text
    assert subprocess.run(["bash",str(path),"--retry"],capture_output=True).returncode==2


def test_no_local_heavy_import_or_main_sideeffect(renderer):
    tree=ast.parse(Path(renderer.__file__).read_text());imports={n.module for n in tree.body if isinstance(n,ast.ImportFrom)}
    imports|={a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names}
    assert not {"torch","pytorch3d","PIL"}&imports
    with pytest.raises(SystemExit):renderer.main(["--oracle"])



def test_public_producer_counts_track_attempt_return_and_accepted(renderer):
    source=Path(renderer.__file__).read_text()
    assert 'reference_forward_attempts=0' in source and 'reference_forward_returns=0' in source
    assert source.count('report["reference_forward_attempts"]+=1')==2
    assert source.count('report["reference_forward_returns"]+=1')==2
    assert source.index('report["actual_reference_forward_calls"]=2')>source.index('render.semantics.check_geometry(base.cpu().numpy()')
    assert 'bottle_manufacturing_scale=BOTTLE_SCALE' in source and renderer.BOTTLE_SCALE==1.04


def test_render_only_static_closure_complete_small(renderer):
    import io
    import tarfile
    import azure_job
    root=REPO
    files={p.relative_to(root).as_posix():p.read_bytes()for directory in("infra","src","configs")
        for p in(root/directory).rglob("*")if p.is_file()and"__pycache__"not in p.parts}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_root5_rgb_prepare.sh")
    assert "infra/root5_rgb_protocol.py"in paths and"infra/root5_rgb_render.py"in paths
    assert "infra/root5_rgb_observe.py"not in paths and"infra/keypoint_rgb_baseline.py"not in paths
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w",format=tarfile.PAX_FORMAT)as archive:
        for path in paths:
            info=tarfile.TarInfo(path);info.size=len(files[path]);archive.addfile(info,io.BytesIO(files[path]))
    encoded,_=azure_job.encoded_runtime_archive(stream.getvalue())
    assert len(encoded)<=azure_job.MAX_CODE_CONTROL_BYTES

"""Tiny private procedural math/contracts only; no model or historical geometry."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("own_grasp_capability_test",ROOT/"infra/own_grasp_capability.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_bottle_exact_only_original_pure_function_no_old_recipe_or_results(gate):
    tree=ast.parse((ROOT/"infra/joint_rgb_render.py").read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="bottle_mesh")
    namespace={"np":np};exec(compile(ast.Module(body=[function],type_ignores=[]),"own_bottle_only","exec"),namespace)
    first=gate.bottle_mesh();second=namespace["bottle_mesh"]()
    assert all(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(first,second))
    v,f=first;assert v.shape==(194,3) and f.shape==(384,3)
    assert np.ptp(v[128:160,0])*1000==pytest.approx(56.43)
    assert np.ptp(v[128:160,2])*1000==pytest.approx(46.413)
    certificate=gate.audit_closed_surface(v,f);gate.require_closed(certificate)


@pytest.mark.parametrize("fault",["nan","dtype","scale","shape","locked","boundsnan","wrongdims"])
def test_manufacturing_249_limits_zeroidentity_scales_failclosed(gate,fault):
    q=np.zeros(204,np.float32);bounds=np.tile([-1.,1.],(249,1)).astype(np.float32)
    if fault=="nan":q[68]=np.nan
    elif fault=="dtype":q=q.astype(np.float64)
    elif fault=="scale":q[136]=.001
    elif fault=="shape":bounds[204]=[.1,1.]
    elif fault=="locked":bounds[68]=[0.,0.];q[68]=.01
    elif fault=="boundsnan":bounds[68,0]=np.nan
    else:bounds=bounds[:204]
    with pytest.raises(ValueError):gate.legal_controls(q,bounds)


def semantic_fixture():
    # Deliberately manufactured metadata fixture, not real model semantics.
    suffix=["thumb0_rx","thumb0_ry","thumb0_rz","thumb1_rx","thumb1_ry","thumb1_rz","thumb2_rz","thumb3_rz",
        "index1_rx","index1_ry","index1_rz","index2_rz","index3_rz","middle1_rx","middle1_ry","middle1_rz","middle2_rz","middle3_rz",
        "ring1_rx","ring1_ry","ring1_rz","ring2_rz","ring3_rz","pinky1_ry","pinky1_rz","pinky2_rz","pinky3_rz"]
    names=[f"other{i}" for i in range(249)];names[68:122]=[f"{side}_{s}" for side in "lr" for s in suffix]
    joints=["root","l_wrist","r_wrist"]+[f"{side}_{finger}{digit}" for side in "lr"
        for finger in ("thumb","index","middle","ring","pinky") for digit in range(4)]
    joints += [f"body{i}" for i in range(127-len(joints))]
    parents=np.zeros(127,np.int64);parents[0]=-1
    for side in "lr":
        for finger in ("thumb","index","middle","ring","pinky"):
            for digit in range(4):parents[joints.index(f"{side}_{finger}{digit}")]=joints.index(f"{side}_{finger}{digit-1}") if digit else joints.index(f"{side}_wrist")
    transform=np.zeros((889,249),np.float32)
    for col in range(68,122):transform[joints.index(names[col].rsplit("_",1)[0])*7+3,col]=1.
    bounds=np.tile([-.7,.9],(249,1)).astype(np.float32)
    indices=np.zeros((18439,1),np.int64);weights=np.ones((18439,1),np.float32)
    indices[:90,0]=joints.index("l_thumb3");indices[90:180,0]=joints.index("l_index3")
    faces=np.resize(np.array([[0,1,2],[90,91,92],[3,4,5],[93,94,95],[6,7,8],[96,97,98]],np.int64),(36874,3))
    return names,joints,parents,transform,bounds,indices,weights,faces


def test_actual_semantic_contract_uses_named_distal_lbs_triangles_not_aabb(gate):
    args=semantic_fixture();result=gate.metadata(*args)
    assert len(result["active"])==13 and all(args[0][i].startswith(("l_thumb","l_index")) for i in result["active"])
    for finger in ("thumb","index"):
        assert result["distal"][finger]["joint"]==args[1].index(f"l_{finger}3")
        assert len(result["distal"][finger]["faces"])>=3


@pytest.mark.parametrize("fault",["duplicate_names","finger_columns","cycle","foreign_rotation","translation","badweights",
    "foreign_distal","no_patch","locked_zero","unbounded"])
def test_metadata_semantics_reject_guesses_missingpatches_and_illegalzeros(gate,fault):
    args=list(semantic_fixture());names,joints,parents,transform,bounds,indices,weights,faces=args
    if fault=="duplicate_names":names[0]=names[1]
    elif fault=="finger_columns":names[68]="guessed_thumb_parameter"
    elif fault=="cycle":parents[0]=1
    elif fault=="foreign_rotation":transform[joints.index("r_thumb0")*7+3,68]=1.
    elif fault=="translation":transform[joints.index("l_thumb0")*7,68]=1.
    elif fault=="badweights":weights[0]=.2
    elif fault=="foreign_distal":parents[joints.index("r_index0")]=joints.index("l_thumb3")
    elif fault=="no_patch":indices[:90]=0
    elif fault=="locked_zero":bounds[200]=[.1,.1]
    else:bounds[68]=[-np.inf,np.inf]
    with pytest.raises(ValueError):gate.metadata(*args)


def test_fivepoint_original_stencils_respect_actual_limits(gate):
    for lower,upper in ((-1.,1.),(0.,1.),(-1.,0.)):
        offsets,mode=gate.five_point_offsets(0.,lower,upper)
        assert np.all(offsets>=lower) and np.all(offsets<=upper)
        values=offsets[:,None]**4+3*offsets[:,None]
        observed=gate.five_point_derivative(np.zeros(1),values,offsets,mode)
        assert observed==pytest.approx([3.],abs=1e-10)
    with pytest.raises(ValueError):gate.five_point_offsets(0.,0.,1e-8)


def test_bounded_update_scales_direction_no_native_clipping(gate):
    current=np.array([0.,.09]);delta=np.array([.5,.2]);lo=np.array([-.1,-.1]);hi=np.array([.1,.1])
    result=gate.feasible_step(current,delta,lo,hi)
    assert np.all(result>=lo) and np.all(result<=hi)
    assert (result[0]-current[0])/delta[0]==pytest.approx((result[1]-current[1])/delta[1],abs=1e-7)
    with pytest.raises(ValueError):gate.feasible_step([.1],[.2],[-.1],[.1])


def contacts(gate):
    # Two actual planar interior triangle witnesses with opposed normals.
    left=np.array([[0.,0.,0.],[0.,.001,0.],[0.,0.,.001]])
    right=left+np.array([.054,0.,0.]);right=right[::-1]
    ov=np.r_[left,right];of=np.array([[0,2,1],[3,5,4]],np.int64)
    op,on=gate.centers_normals(ov,of)
    hv=ov.copy();hv[:3]+=gate.GAP*on[0];hv[3:]+=gate.GAP*on[1]
    hf=of[:,::-1].copy();patches=dict(human_faces=np.array([0,1]),object_faces=np.array([0,1]))
    return hv,hf,ov,of,patches


def test_positive_opposed_interior_triangle_proximity_not_forceclosure(gate):
    args=contacts(gate);report=gate.contact_evidence(*args)
    assert report["passed"] and report["surface_pair_distance_m"]==pytest.approx([.0005,.0005])
    assert report["force_closure_verified"] is report["touching_certified"] is False


def test_selection_uses_fresh_distal_lbs_faces_and_actual_opposed_neck_triangles(gate):
    bv,bf=gate.bottle_mesh();centers,normals=gate.centers_normals(bv,bf)
    pair=None
    for a in range(192,256):
        for b in range(a+1,256):
            direction=(centers[b]-centers[a])/np.linalg.norm(centers[b]-centers[a])
            if normals[a]@normals[b]<=-.95 and normals[a]@(-direction)>=.8 and normals[b]@direction>=.8:
                pair=(a,b);break
        if pair:break
    assert pair is not None
    verts=[];faces=[]
    for first in pair:
        triangle=bv[bf[first]]+.0005*normals[first]
        for offset in (-1e-4,0.,1e-4):
            start=len(verts);verts.extend(triangle+[0.,offset,0.]);faces.append([start,start+2,start+1])
    hv=np.asarray(verts);hf=np.asarray(faces,np.int64)
    semantic=dict(distal={"thumb":dict(joint=1,parent=0,faces=np.array([0,1,2])),
        "index":dict(joint=2,parent=0,faces=np.array([3,4,5]))},wrist=0)
    sk=np.zeros((3,8));sk[0,:3]=hv.mean(0)+[0.,.1,0.];sk[1,:3]=sk[0,:3]+[0.,.01,0.];sk[2,:3]=sk[0,:3]+[0.,.01,0.]
    patches=gate.select_patches(hv,sk,hf,semantic,bv,bf)
    assert patches["human_faces"][0] in (0,1,2) and patches["human_faces"][1] in (3,4,5)
    assert np.all((patches["object_faces"]>=192)&(patches["object_faces"]<256))
    assert np.linalg.det(patches["R"])==pytest.approx(1.) and np.allclose(patches["R"]@patches["R"].T,np.eye(3))


@pytest.mark.parametrize("fault",["touch","inside","far","normals","scalegap"])
def test_surface_contacts_failwrongside_normals_or_gap(gate,fault):
    hv,hf,ov,of,p=contacts(gate)
    if fault=="touch":hv=ov.copy()
    elif fault=="inside":hv[:3]=2*ov[:3]-hv[:3]
    elif fault=="far":hv[:3,0]-=.01
    elif fault=="normals":hf[0]=hf[0,::-1]
    else:hv[:3]=ov[:3]+4*(hv[:3]-ov[:3])
    assert not gate.contact_evidence(hv,hf,ov,of,p)["passed"]


def test_ledger_every_actual_attempt_return_validation_and_max100(gate):
    report={};ledger=gate.NativeLedger(report,lambda:None)
    for _ in range(100):assert ledger.call("test",lambda:123,lambda value:None)==123
    ledger.complete();assert report["native_attempts"]==report["native_returns"]==report["native_validated"]==100
    with pytest.raises(RuntimeError):ledger.call("forbidden101",lambda:pytest.fail("No actual101 forward"),lambda _:None)
    report={};ledger=gate.NativeLedger(report,lambda:None)
    with pytest.raises(ValueError):ledger.call("native_invalid",lambda:123,lambda _:(_ for _ in ()).throw(ValueError("invalid")))
    assert report["native_calls"][0]["returned"] and not report["native_calls"][0]["validated"]
    with pytest.raises(ValueError):ledger.complete()


@pytest.mark.parametrize("parameter_count",[0,1,3])
def test_freeze_uses_only_native_parameter_tensor_api_not_scriptmodule_api(gate,parameter_count):
    class Parameter:
        def __init__(self):
            self.requires_grad=True
            self.value=b"unchanged-native-parameter-values"
            self.calls=[]
        def requires_grad_(self,value):
            self.calls.append(value)
            self.requires_grad=value
    class ScriptModule:
        def __init__(self):self.weights=[Parameter() for _ in range(parameter_count)]
        def parameters(self):return iter(self.weights)
        def requires_grad_(self,_):pytest.fail("TorchScript module-wide requires_grad_ is unsupported")
    model=ScriptModule();gate.freeze_native_weights(model)
    assert all(p.requires_grad is False and p.calls==[False]
        and p.value==b"unchanged-native-parameter-values" for p in model.weights)
    tree=ast.parse((ROOT/"infra/own_grasp_capability.py").read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="freeze_native_weights")
    assert not any(isinstance(n,ast.Name) and n.id=="torch" for n in ast.walk(function))


def test_freeze_rejects_a_native_parameter_that_remains_trainable(gate):
    class Parameter:
        requires_grad=True
        def requires_grad_(self,_):return self
    class Model:
        def parameters(self):return iter([Parameter()])
    with pytest.raises(ValueError,match="weights must remain frozen"):gate.freeze_native_weights(Model())


@pytest.mark.parametrize("fault",["controls","identity","expression","signedzero"])
def test_synthetic_original_forward_mutation_never_validated(gate,fault):
    q=np.zeros(204,np.float32);identity=np.zeros((1,45),np.float32);expr=np.zeros((1,72),np.float32)
    before=gate.native_input_identity(q,identity,expr);report={};ledger=gate.NativeLedger(report,lambda:None)
    def forward():
        if fault=="controls":q[68]=.01
        elif fault=="identity":identity[0,0]=.01
        elif fault=="expression":expr[0,0]=.01
        else:identity[0,0]=-0.
        return None
    def validate(_):
        if gate.native_input_identity(q,identity,expr)!=before:raise ValueError("mutated original inputs")
    with pytest.raises(ValueError):ledger.call("fake_original_mutation",forward,validate)
    assert report["native_calls"][0]["returned"] and not report["native_calls"][0]["validated"]


def test_failure_cleanup_only_disposable_current_payloads_keeps_report(gate,tmp_path):
    (tmp_path/"report.json").write_text('{"status":"fail"}')
    for name in ("controls.npz","geometry.npz"):(tmp_path/name).write_bytes(b"this attempt only")
    gate.clear_failed_payloads(tmp_path)
    assert {p.name for p in tmp_path.iterdir()}=={"report.json"}


@pytest.mark.parametrize("fault",["status","nonpenetration","coverage","broadphase","sampled","touch","force","exact","intersection","embedding","inside"])
def test_cross_certificate_cannot_omit_full_embedding_intersections_or_containment(gate,fault):
    record=dict(status="pass",nonpenetration_certified=True,full_original_faces_retained=True,broadphase_complete=True,
        sampled_collision_test_used=False,touching_certified=False,force_closure_verified=False,exact_arithmetic_proof=False,
        intersection_counts=dict(proper_intersection=0,coplanar_overlap=0,boundary_contact=0,ambiguous=0),
        embedding=[dict(verified=True),dict(verified=True)],containment=[[dict(state="outside")],[dict(state="outside")]],
        tolerance_m=1e-8,proximity_m=.002,max_candidate_pairs=2_000_000)
    gate.require_cross(record)
    key={"status":"status","nonpenetration":"nonpenetration_certified","coverage":"full_original_faces_retained", "broadphase":"broadphase_complete",
        "sampled":"sampled_collision_test_used","touch":"touching_certified","force":"force_closure_verified","exact":"exact_arithmetic_proof"}
    if fault in key:record[key[fault]]="fail" if fault=="status" else not record[key[fault]]
    elif fault=="intersection":record["intersection_counts"]["ambiguous"]=1
    elif fault=="embedding":record["embedding"][1]["verified"]=False
    else:record["containment"][0][0]["state"]="inside"
    with pytest.raises(ValueError):gate.require_cross(record)


@pytest.mark.parametrize("value",[None,[],[[]],[[],[]],[[dict(state="outside")]],[[dict(state="outside")],[]]])
def test_cross_containment_cannot_be_vacuous(gate,value):
    record=dict(status="pass",nonpenetration_certified=True,full_original_faces_retained=True,broadphase_complete=True,
        sampled_collision_test_used=False,touching_certified=False,force_closure_verified=False,exact_arithmetic_proof=False,
        intersection_counts=dict(proper_intersection=0,coplanar_overlap=0,boundary_contact=0,ambiguous=0),
        embedding=[dict(verified=True),dict(verified=True)],containment=value,tolerance_m=1e-8,proximity_m=.002,max_candidate_pairs=2_000_000)
    with pytest.raises(ValueError):gate.require_cross(record)


def test_explicit_code_revision_bundle_namespace_and_exact_mount_modes(gate,tmp_path):
    root=tmp_path.resolve();revision="a"*40;code=root/"jobs"/revision/"run_own_grasp_capability"/"code";code.mkdir(parents=True)
    assert gate.bound_code(root,code,revision)==code
    for bad in (root/"code",root/"jobs"/revision/"oldjob"/"code",root/"jobs"/revision/"own_grasp_capability"/"code"):
        with pytest.raises(ValueError):gate.bound_code(root,bad,revision)
    text=f"36 25 0:32 / {code} ro,relatime - ext4 /dev/sda rw\n"
    assert gate.exact_mount(text,code,True)["mount_options"]==["ro","relatime"]
    with pytest.raises(ValueError):gate.exact_mount(text,code,False)
    with pytest.raises(ValueError):gate.exact_mount(text+text,code,True)


@pytest.mark.parametrize("fault",["masked_bounds","masked_lbs","masked_controls"])
def test_masked_original_metadata_is_not_silently_unpacked(gate,fault):
    args=list(semantic_fixture())
    if fault=="masked_bounds":args[4]=np.ma.array(args[4])
    elif fault=="masked_lbs":args[6]=np.ma.array(args[6])
    else:
        with pytest.raises(ValueError):gate.legal_controls(np.ma.array(np.zeros(204,np.float32)),args[4])
        return
    with pytest.raises(ValueError):gate.metadata(*args)


@pytest.mark.parametrize("field",["names","joints","parents","map","bounds","faces","LBSindices","LBSweights"])
def test_every_original_native_metadata_snapshot_binds_exact_bytes(gate,field):
    names,joints,parents,transform,bounds,indices,weights,faces=semantic_fixture()
    arrays=[names,joints,parents,transform,bounds,faces,indices,weights]
    before=gate.native_metadata_identity(*arrays)
    index=["names","joints","parents","map","bounds","faces","LBSindices","LBSweights"].index(field)
    if index<2:arrays[index][0]="changed_actual_metadata"
    else:arrays[index].flat[0]+=1
    assert gate.native_metadata_identity(*arrays)!=before


@pytest.mark.parametrize("fault",["missing","droppedfaces","droppedvertices","boolcount","embedding","topology"])
def test_full_original_human_bottle_coverage_inventory_is_required(gate,fault):
    certificate=dict(meshes=[dict(vertices=18439,faces=36874,original_face_coverage=36874,topology_closed_oriented=True,embedding_verified=True),
        dict(vertices=194,faces=384,original_face_coverage=384,topology_closed_oriented=True,embedding_verified=True)])
    gate.require_original_coverage(certificate)
    if fault=="missing":certificate.pop("meshes")
    elif fault=="droppedfaces":certificate["meshes"][0]["original_face_coverage"]-=1
    elif fault=="droppedvertices":certificate["meshes"][1]["vertices"]-=1
    elif fault=="boolcount":certificate["meshes"][1]["faces"]=True
    elif fault=="embedding":certificate["meshes"][0]["embedding_verified"]=False
    else:certificate["meshes"][1]["topology_closed_oriented"]=False
    with pytest.raises(ValueError):gate.require_original_coverage(certificate)


def test_private_source_only_wrapper_and_no_historical_runtime_closure(gate):
    source=(ROOT/"infra/own_grasp_capability.py").read_text();tree=ast.parse(source)
    imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports|={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imports<={"__future__","argparse","hashlib","json","math","os","pathlib","platform","re","signal","stat","sys","time","numpy","torch","world_reward.cross_surface"}
    assert gate.MAX_FORWARDS==100 and gate.BUDGET==300 and gate.SEARCH_STATES+7<=100
    assert source.index('audit_closed_surface(cpu(hv)')<source.index('torch.autograd.grad(value,q')<source.index('for step in range(SEARCH_STATES)')
    assert "apply_correctives" in source and "JIT_optimized=False" in source
    shell=ROOT/"infra/run_own_grasp_capability.sh";text=shell.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    result=subprocess.run(["rtk","proxy","bash",str(shell),"--episode","15"],capture_output=True,env={"PATH":os.environ["PATH"]})
    assert result.returncode==2
    assert text.count('--mount "type=bind')==3 and 'src=$MODEL,dst=$MODEL,readonly' in text and 'src=$CODE,dst=$CODE,readonly' in text
    assert "303s docker run" in text and "--network none" in text and "mkdir -m 700" in text
    assert "weights/cari4d" not in text and "/results" not in text and "eval_private" not in text and "/data" not in text


def test_static_archive_only_own_driver_cross_surface_and_package(gate):
    import azure_job
    files={str(path.relative_to(ROOT)):path.read_bytes() for folder in ("infra","src","configs")
        for path in (ROOT/folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts
        and path.suffix in (".py",".sh",".json",".toml")}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_own_grasp_capability.sh")
    assert {name for name in paths if name.endswith((".py",".sh"))}==set(gate.HELPERS)


@pytest.mark.parametrize("fault",[None,"wrong_revision","bad_source_sha","symlink_marker","writable_extra_code","wrong_entrypoint"])
def test_real_azure_bundle_namespace_and_original_marker_preflight_without_gpu(gate,tmp_path,fault):
    root=tmp_path.resolve();revision="a"*40
    # Derive with the unchanged real launcher's algorithm, not a guessed name.
    script="infra/run_own_grasp_capability.sh";bundle=script.removeprefix("infra/").removesuffix(".sh")
    code=root/"jobs"/revision/bundle/"code";out=root/gate.OUTPUT;out.parent.mkdir()
    model=root/"weights/mhr/mhr_model.pt";model.parent.mkdir(parents=True)
    with model.open("wb") as stream:stream.truncate(gate.MODEL_BYTES)  # Sparse metadata fixture, never a downloaded model.
    for name in gate.HELPERS:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"immutable tiny source");path.chmod(0o444)
    (code.parent/"revision").write_text(revision+"\n");(code.parent/"source-sha256").write_text("b"*64+"\n")
    for path in sorted((p for p in code.rglob("*") if p.is_dir()),reverse=True):path.chmod(0o555)
    code.chmod(0o555)
    executing=code/script
    if fault=="wrong_revision":(code.parent/"revision").write_text("c"*40+"\n")
    elif fault=="bad_source_sha":(code.parent/"source-sha256").write_text("not-a-hash\n")
    elif fault=="symlink_marker":
        target=code.parent/"source-sha256";target.unlink();target.symlink_to(code.parent/"revision")
    elif fault=="writable_extra_code":
        code.chmod(0o755);extra=code/"unbound.txt";extra.write_text("writable snapshot noise");code.chmod(0o555)
    elif fault=="wrong_entrypoint":executing=root/"elsewhere.sh"
    text=(ROOT/script).read_text();program=text.split("<<'PYSAFE'\n",1)[1].split("\nPYSAFE",1)[0]
    result=subprocess.run(["rtk","proxy",sys.executable,"-I","-B","-c",program,str(root),str(code),str(out),
        str(model),revision,str(executing)],capture_output=True,text=True)
    assert (result.returncode==0)==(fault is None),result.stderr
    assert not out.exists()  # This is strictly preflight; no docker/GPU/write.

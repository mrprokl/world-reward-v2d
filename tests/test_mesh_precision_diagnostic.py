import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "infra"))
    spec = importlib.util.spec_from_file_location("precision_test", REPO / "infra/mesh_precision_diagnostic.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    def quaternion_matrix(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0], [0,0,0,1.]])
    fake = types.SimpleNamespace(transformations=types.SimpleNamespace(quaternion_matrix=quaternion_matrix),
                                 transform_points=lambda v, mat: v.astype(np.float64) @ mat[:3, :3].T + mat[:3, 3])
    monkeypatch.setitem(sys.modules, "trimesh", fake)
    return m


def geometry():
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])


def glb(path, v=None, f=None, mutate=None):
    if v is None: v, f = geometry()
    v = v.astype("<f4"); f = f.astype("<u4")
    binary = v.tobytes() + f.tobytes()
    doc = {"asset":{"version":"2.0"}, "buffers":[{"byteLength":len(binary)}],
           "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":v.nbytes}, {"buffer":0,"byteOffset":v.nbytes,"byteLength":f.nbytes}],
           "accessors":[{"bufferView":0,"componentType":5126,"count":len(v),"type":"VEC3"},
                        {"bufferView":1,"componentType":5125,"count":f.size,"type":"SCALAR"}],
           "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"mode":4}]}],
           "nodes":[{"mesh":0}], "scenes":[{"nodes":[0]}], "scene":0}
    if mutate: mutate(doc)
    header = json.dumps(doc,separators=(",", ":")).encode(); header += b" " * (-len(header)%4)
    raw = struct.pack("<4sII",b"glTF",2,28+len(header)+len(binary)) + struct.pack("<II",len(header),0x4e4f534a) + header + struct.pack("<II",len(binary),0x004e4942) + binary
    path.write_bytes(raw); return v, f.astype(np.int64)


def test_accessors_loader_exact_weld_diagnostic(driver, monkeypatch, tmp_path):
    p=tmp_path/"source.glb";v,f=glb(p); before=p.read_bytes()
    monkeypatch.setattr(driver.endpoint,"_load_mesh",lambda _: (v.astype(np.float64), f))
    result=driver.measure(p)
    assert result["raw_scene_loader_triangles_identical"] and result["welding_triangles_identical"]
    assert result["original_exact_weld_accepts"]
    assert result["raw_accessor_area"][0]["dtype"]=="float32"
    assert result["loaded_area"]["dtype"]=="float64"
    assert result["loaded_area"]["arithmetic_zero_faces"]==0 and p.read_bytes()==before


def test_f64_to_glb_f32_loss_without_repair(driver, monkeypatch, tmp_path):
    v,f=geometry();v += 1e8
    assert driver.area_metrics(v,f)["arithmetic_zero_faces"]==0
    p=tmp_path/"loss.glb";v32,f=glb(p,v,f)
    monkeypatch.setattr(driver.endpoint,"_load_mesh",lambda _: (v32.astype(np.float64), f))
    result=driver.measure(p)
    assert result["raw_accessor_area"][0]["arithmetic_zero_faces"]==4
    assert result["welded_area"]["float64_recomputed_zero_faces"]==4
    assert not result["original_exact_weld_accepts"] and result["faces_preserved"]==4


def test_tiny_well_shaped_global_false_positive_not_angular(driver):
    v=np.array([[0.,0.,0.],[1e-8,0.,0.],[0.,1e-8,0.],[1.,1.,1.]],dtype=np.float64)
    f=np.array([[0,1,2]])
    r=driver.area_metrics(v,f)
    assert r["arithmetic_zero_faces"]==0 and r["positive_below_global_threshold"]==1
    assert r["positive_angular_roundoff_faces"]==0 and r["minimum_positive_sin_angle"]==1
    assert r["global_area_guard_would_reject"] and r["angular_threshold_is_diagnostic_only"]


def test_nearly_collinear_positive_angular_roundoff(driver):
    v=np.array([[0.,0.,0.],[1.,0.,0.],[2.,1e-15,0.]])
    r=driver.area_metrics(v,np.array([[0,1,2]]))
    assert r["arithmetic_zero_faces"]==0 and r["positive_angular_roundoff_faces"]==1


def test_transformed_raw_accessors_and_cyclic_orientation(driver,monkeypatch,tmp_path):
    p=tmp_path/"transform.glb";v,f=glb(p,mutate=lambda d:d["nodes"][0].update(translation=[.25,.5,.75],scale=[2,3,4]))
    expected=v.astype(np.float64)*[2,3,4]+[.25,.5,.75]
    monkeypatch.setattr(driver.endpoint,"_load_mesh",lambda _:(expected,np.roll(f,1,axis=1)))
    r=driver.measure(p);assert not r["accessors"][0]["node_transform_identity"]
    assert driver.triangle_hash(expected[f])==driver.triangle_hash(expected[np.roll(f,1,axis=1)])


@pytest.mark.parametrize("fault",["flip","perturb","drop"])
def test_loader_triangle_changes_fail(driver,monkeypatch,tmp_path,fault):
    p=tmp_path/"x.glb";v,f=glb(p);v=v.astype(np.float64)
    if fault=="flip":f=f[:,::-1]
    if fault=="perturb":v[0,0]+=1e-9
    if fault=="drop":f=f[:-1]
    monkeypatch.setattr(driver.endpoint,"_load_mesh",lambda _:(v,f))
    with pytest.raises(ValueError,match="triangle identity"):driver.measure(p)


@pytest.mark.parametrize("mutate",[
    lambda d:d.update(extensionsRequired=["KHR_draco_mesh_compression"]),
    lambda d:d["buffers"][0].update(uri="external.bin"),
    lambda d:d["accessors"][0].update(sparse={"count":1}),
    lambda d:d["accessors"][0].update(normalized=True),
    lambda d:d["accessors"][0].update(count=1000000),
    lambda d:d["bufferViews"][0].update(byteStride=2),
    lambda d:d["meshes"][0]["primitives"][0].update(mode=5),
    lambda d:d["nodes"][0].update(children=[0]),
    lambda d:d["scenes"][0].update(nodes=[0,0]),
    lambda d:d["nodes"][0].update(rotation=[0,0,0,0]),
])
def test_unsupported_ambiguous_accessor_or_scene_fails(driver,tmp_path,mutate):
    p=tmp_path/"x.glb";glb(p,mutate=mutate)
    with pytest.raises((ValueError,IndexError,KeyError)):driver.raw_glb(p)


def provenance(driver,root,episode=9):
    rev="a"*40; image="sha256:"+"b"*64
    paths={"glb":root/f"outputs/episode_{episode:06d}/object_grounded/object.glb",
           "object_report":root/f"outputs/episode_{episode:06d}/object_grounded/report.json",
           "failed_report":root/f"outputs/episode_{episode:06d}/object_budget_volume/report.json",
           "original_producer":root/f"jobs/{rev}/run_track1_frontends/code/infra/object_smoke.py",
           "image_evidence":root/"results/image-objects.json"}
    for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
    glb(paths["glb"]);paths["original_producer"].write_text("# immutable original producer\n")
    obj={"stage":"sam3d_objects_grounded_fixed_frame","status":"pass","episode_index":episode,"frame_index":0,
         "ground_truth_used":False,"hand_labeled_test":False,"oracle_modes":[],"object_sha256":driver.identity(paths["glb"])["sha256"],
         "script_sha256":driver.identity(paths["original_producer"])["sha256"],"image_id":image,"input_sha256":"d"*64}
    paths["object_report"].write_text(json.dumps(obj));paths["image_evidence"].write_text(json.dumps({"Id":image}))
    old={"stage":"world_reward_cpu_volume_constrained_object_mesh","status":"fail","episode_index":episode,"error_type":"ValueError",
         "ground_truth_used":False,"hand_labeled_test":False,"oracle_modes":[],"adoption_performed":False,"input_sha256":"d"*64,"error":"Collapsed/numerically zero-area faces are forbidden",
         "source_hashes":{"object.glb":driver.identity(paths["glb"])["sha256"],"object_report":driver.identity(paths["object_report"])["sha256"]}}
    paths["failed_report"].write_text(json.dumps(old))
    pins={"schema":driver.PIN_SCHEMA,"episode_index":episode,"diagnostic_image_id":image,"image_source_id":image,
          "original_producer_revision":rev,"image_evidence_format":"json_image_identity",
          "files":{k:{"path":str(p.relative_to(root)),**driver.identity(p)}for k,p in paths.items()}}
    pin=root/"pins.json";pin.write_text(json.dumps(pins));return paths,pins,pin


def test_original_source_image_failure_all_five_bound(driver,tmp_path):
    paths,pins,pin=provenance(driver,tmp_path)
    _,receipt=driver.bindings(tmp_path,9,pin)
    assert set(receipt["files"])==driver.ROLES and receipt["image_source_id"]==pins["image_source_id"]
    paths["glb"].write_bytes(paths["glb"].read_bytes()+b"x")
    with pytest.raises(ValueError,match="hash differs"):driver.bindings(tmp_path,9,pin)


@pytest.mark.parametrize("fault",["episode","sourcepath","image","failure","private","script"])
def test_provenance_closed_before_mesh_parse(driver,tmp_path,fault):
    paths,pins,pin=provenance(driver,tmp_path)
    if fault=="episode":pins["episode_index"]=8
    if fault=="sourcepath":pins["files"]["original_producer"]["path"]=str(paths["object_report"].relative_to(tmp_path));pins["files"]["original_producer"].update(driver.identity(paths["object_report"]))
    if fault=="private":pins["files"]["image_evidence"]["path"]="outputs/eval_private/report.json"
    if fault in ("image","failure","script"):
        role="image_evidence"if fault=="image"else "failed_report"if fault=="failure"else "object_report"
        data=json.loads(paths[role].read_text())
        data.update({"Id":"sha256:"+"c"*64}if fault=="image"else {"status":"pass"}if fault=="failure"else {"script_sha256":"c"*64})
        paths[role].write_text(json.dumps(data));pins["files"][role].update(driver.identity(paths[role]))
    pin.write_text(json.dumps(pins))
    with pytest.raises(ValueError):driver.bindings(tmp_path,9,pin)


def test_dispatcher_log_identity_supported_without_publishing_log(driver,tmp_path):
    paths,pins,pin=provenance(driver,tmp_path)
    paths["image_evidence"].write_text("Docker frozen image="+pins["image_source_id"]+"\n")
    pins["image_evidence_format"]="dispatcher_log";pins["files"]["image_evidence"].update(driver.identity(paths["image_evidence"]));pin.write_text(json.dumps(pins))
    _,receipt=driver.bindings(tmp_path,9,pin);assert receipt["image_evidence_format"]=="dispatcher_log"
    assert "Docker frozen"not in json.dumps(receipt)


def test_wrapper_only_five_provenance_no_gpu_or_data(driver):
    shell=Path(driver.__file__).with_name("run_mesh_precision_diagnostic.sh").read_text()
    assert "--memory 8g --cpus 4"in shell and "60s docker run"in shell
    assert "--network none --read-only --cap-drop ALL"in shell
    assert "--gpus"not in shell and "src=$ROOT/data"not in shell and "src=$ROOT/outputs"not in shell
    assert "${MOUNTS[@]}"in shell and "readonly"in shell and "chown -R"not in shell
    assert "--episode"in shell and "--pins"not in shell and "image inspect"in shell


def test_cli_duplicate_or_noncanonical_episode_rejected(driver):
    for args in ([],["--episode","09"],["--episode","30"],["--episode","9","--episode","8"]):
        with pytest.raises((SystemExit,KeyError)):driver.main(args)


def test_full_main_seals_scalar_only_all_inputs_read_only(driver,monkeypatch,tmp_path):
    import importlib.metadata
    root=tmp_path/"root";root.mkdir();paths,pins,pin=provenance(driver,root)
    code=tmp_path/"code";infra=code/"infra";infra.mkdir(parents=True)
    config=code/driver.PIN_FILE;config.parent.mkdir();config.write_bytes(pin.read_bytes())
    script=infra/"mesh_precision_diagnostic.py";script.write_text("# readonly fixture\n")
    monkeypatch.setattr(driver,"__file__",str(script));monkeypatch.setattr(driver,"source_helpers",lambda:{"fixture":{"bytes":1,"sha256":"e"*64}})
    monkeypatch.setattr(driver,"selftest",lambda:{"passed":True,"test_only_fake":True})
    monkeypatch.setattr(driver.platform,"system",lambda:"Linux")
    real_iter=Path.iterdir
    monkeypatch.setattr(Path,"iterdir",lambda p: iter([Path("lo")]) if p==Path("/sys/class/net") else real_iter(p))
    monkeypatch.setattr(importlib.metadata,"version",lambda _:"test-only-fake-trimesh")
    monkeypatch.setenv("WR_ROOT",str(root));monkeypatch.setenv("WR_CODE_REVISION","e"*40);monkeypatch.setenv("WR_IMAGE_ID",pins["diagnostic_image_id"])
    out=root/"diagnostic/episode000009-mesh-precision-v1";out.mkdir(parents=True)
    local,world,_=driver.raw_glb(paths["glb"]);v,f=local[0]
    monkeypatch.setattr(driver.endpoint,"_load_mesh",lambda _:(v.astype(np.float64),f))
    before={k:p.read_bytes()for k,p in paths.items()};driver.main(["--episode","9"])
    report=json.loads((out/"report.json").read_text())
    assert report["status"]=="measurement_complete" and report["historical_failure_unchanged"]
    assert report["inputs_sources_rehashed_after"] and not report["historical_preexport_identity_provable"]
    assert not report["arrays_exported"] and not report["production_gate_pass_claimed"]
    assert {p.name for p in out.iterdir()}=={"report.json"} and (out/"report.json").stat().st_mode&0o777==0o444
    assert {k:p.read_bytes()for k,p in paths.items()}==before


def test_post_mutation_fail_receipt_not_measurement_pass(driver,monkeypatch,tmp_path):
    root=tmp_path/"root";root.mkdir();paths,pins,pin=provenance(driver,root)
    code=tmp_path/"code";infra=code/"infra";infra.mkdir(parents=True)
    config=code/driver.PIN_FILE;config.parent.mkdir();config.write_bytes(pin.read_bytes())
    script=infra/"mesh_precision_diagnostic.py";script.write_text("# fixture\n")
    monkeypatch.setattr(driver,"__file__",str(script));monkeypatch.setattr(driver,"source_helpers",lambda:{})
    monkeypatch.setattr(driver,"selftest",lambda:{"passed":True,"test_only_fake":True})
    monkeypatch.setattr(driver.platform,"system",lambda:"Linux");real_iter=Path.iterdir
    monkeypatch.setattr(Path,"iterdir",lambda p:iter([Path("lo")])if p==Path("/sys/class/net")else real_iter(p))
    monkeypatch.setattr(driver.importlib.metadata,"version",lambda _:"fixture")
    monkeypatch.setenv("WR_ROOT",str(root));monkeypatch.setenv("WR_CODE_REVISION","e"*40);monkeypatch.setenv("WR_IMAGE_ID",pins["diagnostic_image_id"])
    out=root/"diagnostic/episode000009-mesh-precision-v1";out.mkdir(parents=True)
    def mutate(_):
        paths["glb"].write_bytes(paths["glb"].read_bytes()+b"x");return {"test_only_mutation":True}
    monkeypatch.setattr(driver,"measure",mutate)
    with pytest.raises(RuntimeError,match="sealed receipt"):driver.main(["--episode","9"])
    report=json.loads((out/"report.json").read_text());assert report["status"]=="fail"
    assert not report["historical_failure_unchanged"] and not report.get("inputs_sources_rehashed_after",False)


def test_source_identity_accepts_historical644_but_rejects_symlink(driver,tmp_path):
    p=tmp_path/"historical";p.write_text("source");p.chmod(0o644)
    assert driver.identity(p)["bytes"]==6
    q=tmp_path/"alias";q.symlink_to(p)
    with pytest.raises(ValueError,match="Symlink"):driver.identity(q)


def test_arithmetic_norm_underflow_separated_from_cross_zero(driver):
    v=np.array([[0,0,0],[1e-15,0,0],[0,1e-15,0]],dtype=np.float32)
    r=driver.area_metrics(v,np.array([[0,1,2]]))
    assert r["arithmetic_zero_faces"]==1 and r["cross_component_zero_faces"]==0 and r["norm_underflow_zero_faces"]==1


def test_real_trimesh_optional_offline_fixture(tmp_path,monkeypatch):
    trimesh=pytest.importorskip("trimesh")
    monkeypatch.syspath_prepend(str(REPO/"infra"));import mesh_precision_diagnostic as actual
    p=tmp_path/"actual.glb";v,f=geometry();p.write_bytes(trimesh.Trimesh(v,f,process=False).export(file_type="glb"))
    assert actual.measure(p)["raw_scene_loader_triangles_identical"]


@pytest.mark.parametrize("raw",['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_strict_duplicate_nonfinite_json(driver,raw):
    with pytest.raises(ValueError):driver.strict_json(raw)


@pytest.mark.parametrize("mutate",[
    lambda d:d["meshes"][0]["primitives"][0]["attributes"].update(POSITION=-1),
    lambda d:d["accessors"][0].update(bufferView=-1),
    lambda d:d["nodes"][0].update(mesh=-1),
    lambda d:d["scenes"][0].update(nodes=[-1]),
    lambda d:d.update(scene=-1),
    lambda d:d["meshes"][0]["primitives"][0].update(indices=True),
])
def test_negative_bool_indices_fail(driver,tmp_path,mutate):
    p=tmp_path/"negative.glb";glb(p,mutate=mutate)
    with pytest.raises(ValueError,match="index required"):driver.raw_glb(p)


def test_wrapper_exact_canonical_dispatch_markers_before_reservation(driver):
    s=Path(driver.__file__).with_name("run_mesh_precision_diagnostic.sh").read_text()
    assert '/srv/scenesmith/world-reward' in s and '/run_mesh_precision_diagnostic/code' in s
    assert "'revision'"in s and "'source-sha256'"in s and 'readonly' in s
    assert s.index("source-sha256") < s.index('mkdir -p "$ROOT/diagnostic"')

"""Tiny source/receipt and real mesh-math fixtures; never a local native job."""
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"infra"))
import object_budget_conditioned as p


def write(path, value, mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode() if isinstance(value, (dict, list)) else value
    path.write_bytes(raw); path.chmod(mode)
    return p.identity(path)


def tetra():
    return (np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64))


def mapping(mesh):
    v, f = mesh
    return dict(I=list(range(len(v))), J=list(range(len(f))), source_vertices=len(v), source_faces=len(f),
                output_vertices=len(v), output_faces=len(f), target_reached=True, mapping_complete=True,
                native_cost_and_placement_unchanged=False, cost_normalization=True, final_shell_volumes_verified=True,
                volume_relative_limit=.05, committed_collapses=0)


def native_document(mesh):
    chart = p.prepare_conditioning(*mesh)
    return dict(conditioning=dict(origin=chart.origin.tolist(), scale=chart.scale, scale_exponent=chart.scale_exponent,
                    source_roundtrip_vertices=len(mesh[0]), chart_scale_positive=True,
                    source_roundtrip_numerically_exact=True, source_roundtrip_byte_exact=True,
                    physical_geometry_rescaled=False, new_numeric_algorithm=True, native_qslim_implementation_reused=True,
                    chart_refitted=False, adopted=False),
                serialization=dict(serialization_safe=True, committed_collapses=0, serialization_vetoes=0),
                native_volume=mapping(mesh))


def cache_fixture(tmp_path):
    root, code = tmp_path/"root", tmp_path/"code"; code.mkdir()
    for name in (*p.REUSED, p.CPP, "infra/mesh_volume_qem.cpp", "infra/mesh_guarded_qem.cpp", "infra/mesh_serialization_qem.cpp"):
        write(code/name, name.encode())
    old_binary = dict(bytes=6, sha256="f"*64)
    previous = dict(producer_revision="b"*40, source_cpp=dict(bytes=8, sha256="c"*64),
                    host_report=dict(bytes=9, sha256="d"*64), native_report=dict(bytes=10, sha256="e"*64),
                    measured_temporary_binary=old_binary)
    write(code/p.PREVIOUS_PINS, previous); write(code/p.CACHE_PROTOCOL, {"frozen": True})
    rev = "a"*40; out = root/"results"/("mesh-conditioned-cache-"+rev); out.mkdir(parents=True)
    binary = write(out/"mesh_conditioned_qem", b"binary", 0o555)
    markers = {n: dict(bytes=len(raw), sha256=p.hashlib.sha256(raw).hexdigest()) for n, raw in
               (("revision", (rev+"\n").encode()), ("source-sha256", ("2"*64+"\n").encode()))}
    bound = dict(producer_revision=rev, source_files=100, source_files_sha256="3"*64, markers=markers,
                 helpers={p.CPP:p.identity(code/p.CPP), p.CACHE_PROTOCOL:p.identity(code/p.CACHE_PROTOCOL)})
    q = dict(pins_identity=p.identity(code/p.PREVIOUS_PINS), source_cpp=previous["source_cpp"],
             host_report=previous["host_report"], native_report=previous["native_report"], measured_binary=old_binary,
             original_runtime={"frozen": True}, qualified_build={"compiler": "native c++"},
             reused_sources={n:p.identity(code/n) for n in p.REUSED}, expected_sources={str(i):["4"*64,"5"*64] for i in range(4)})
    info = dict(physical_coordinate_cache=True, source_sha256=p.identity(code/p.CPP)["sha256"],
                volume_source_sha256=p.identity(code/"infra/mesh_volume_qem.cpp")["sha256"],
                base_source_sha256=p.identity(code/"infra/mesh_guarded_qem.cpp")["sha256"],
                serialization_source_sha256=p.identity(code/"infra/mesh_serialization_qem.cpp")["sha256"],
                target_faces=4096, volume_relative_limit=.05, physical_geometry_rescaled=False, cost_normalization=True,
                new_numeric_algorithm=True, native_cost_and_placement_unchanged=False)
    fixture = lambda i: dict(fixture=str(i), source_array_sha256=q["expected_sources"][str(i)],
        candidate_and_mapping_byte_exact=True, physical_stage_evidence_equal=True,
        comparisons=[dict(implementation=kind, method="conditioned", status="pass", native_attempts=1,
                          native_returned=True, committed_collapses=5) for kind in ("slow","cached")])
    common = dict(status="pass", source_binding=bound, cache_protocol_identity=p.identity(code/p.CACHE_PROTOCOL),
                  retained_binary=binary, conditioned_qualification=q, production_mesh_used=False,
                  challenge_performance_verified=False, adoption=False, gpu_used=False,
                  simplification_validated=False, geometry_quality_validated=False, predicate_parity_only=False,
                  physical_coordinate_cache=True)
    native = common | dict(stage="mesh_conditioned_cache_native_v1", phase="complete", originals_rehashed_after=True,
        owned_scratch_removed=True, original_runtime=q["original_runtime"],
        build=dict(binary=binary, build_info=info, compiler="native c++"),
        slow_build=dict(binary=old_binary, compiler="native c++"),
        parity=dict(cases=128, mismatches=0, policy_sha256=p.POLICY_SHA256), orientation_parity=dict(cases=128, mismatches=0),
        geometry=dict(stage="mesh_conditioned_cache_controls_v1", status="pass", sources_rehashed_after=True,
            owned_scratch_removed=True, previous_successful_sources_only=True, exact_implementation_regression_verified=True,
            failed_controls_replayed=False, adoption=False, maximum_native_calls=8, native_budget_seconds=450,
            paired_fixtures=[fixture(i) for i in range(4)]))
    native_pin = write(out/"native.json", native)
    host = common | dict(stage="mesh_conditioned_cache_host_v1", native_identity=native_pin, original_image_id=p.IMAGE,
                         geometry_qualification_status="pass", source_rehashed_after=True, original_build_rehashed_after=True,
                         owned_container_removed=True)
    host_pin = write(out/"report.json", host)
    pins = dict(schema="world_reward.mesh_conditioned_cache_qualification.v1", producer_revision=rev,
                source_files=100, source_archive_sha256="2"*64, source_files_sha256="3"*64,
                source_cpp=p.identity(code/p.CPP), protocol=p.identity(code/p.CACHE_PROTOCOL),
                host_report=host_pin, native_report=native_pin, retained_binary=binary,
                previous_qualification_pins=p.identity(code/p.PREVIOUS_PINS), binary_mode="0555", receipt_mode="0444",
                directory_mode="0555", exact_implementation_regression_verified=True, qualified_procedural_controls=4,
                native_calls=8, production_mesh_validated=False, challenge_performance_verified=False, adoption=False)
    write(code/p.PINS, pins); out.chmod(0o555)
    return root, code, out, host, native, pins


def repin_receipts(code, out, host, native, pins):
    out.chmod(0o755)
    (out/"native.json").chmod(0o644); pins["native_report"] = write(out/"native.json", native)
    host["native_identity"] = pins["native_report"]
    (out/"report.json").chmod(0o644); pins["host_report"] = write(out/"report.json", host)
    (code/p.PINS).chmod(0o644); write(code/p.PINS, pins); out.chmod(0o555)


def test_missing_pins_fails_without_native_or_io_to_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: pytest.fail("No native call"))
    with pytest.raises(FileNotFoundError): p.cache_qualification(tmp_path, tmp_path)


def test_complete_producer_shaped_cache_fixture(tmp_path):
    root, code, out, *_ = cache_fixture(tmp_path)
    binary, evidence = p.cache_qualification(root, code)
    assert binary == out/"mesh_conditioned_qem" and evidence["retained_binary"] == p.identity(binary)
    assert "old_code" not in evidence and len(evidence["source_binding"]["markers"]) == 2


@pytest.mark.parametrize("field,value", [("native_calls",True),("native_calls",7),("adoption",True),
    ("receipt_mode","0400"),("producer_revision","../a"),("source_files",False),("source_archive_sha256","wrong")])
def test_strict_qualification_pin_types_and_scopes(tmp_path, field, value):
    root, code, out, host, native, pins = cache_fixture(tmp_path)
    pins[field] = value; repin_receipts(code,out,host,native,pins)
    with pytest.raises(ValueError): p.cache_qualification(root,code)


@pytest.mark.parametrize("mutate", [
    lambda h,n: n.update(status="fail"), lambda h,n: n.update(phase="controls"),
    lambda h,n: h.update(owned_container_removed=False),
    lambda h,n: n["source_binding"]["markers"]["revision"].update(sha256="0"*64),
    lambda h,n: n["geometry"]["paired_fixtures"][0]["comparisons"][1].update(native_returned=False),
    lambda h,n: n["geometry"]["paired_fixtures"][0].update(candidate_and_mapping_byte_exact=False),
    lambda h,n: n["build"]["build_info"].update(physical_coordinate_cache=False),
    lambda h,n: n["orientation_parity"].update(mismatches=1),
])
def test_rehashed_forged_pass_still_rejected(tmp_path, mutate):
    root,code,out,host,native,pins=cache_fixture(tmp_path)
    mutate(host,native); repin_receipts(code,out,host,native,pins)
    with pytest.raises(ValueError):p.cache_qualification(root,code)


@pytest.mark.parametrize("which", [p.CPP,p.CACHE_PROTOCOL,p.PREVIOUS_PINS,p.REUSED[0]])
def test_current_implementation_protocol_or_math_cannot_change(tmp_path,which):
    root,code,*_=cache_fixture(tmp_path)
    (code/which).chmod(0o644);(code/which).write_bytes(b"changed")
    with pytest.raises(ValueError):p.cache_qualification(root,code)


@pytest.mark.parametrize("name,mode", [("report.json",0o644),("native.json",0o400),("mesh_conditioned_qem",0o755)])
def test_cache_exact_publication_modes(tmp_path,name,mode):
    root,code,out,*_=cache_fixture(tmp_path);(out/name).chmod(mode)
    with pytest.raises(ValueError):p.cache_qualification(root,code)


def test_strict_json_and_protocol_exact_current(tmp_path):
    for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}'):
        with pytest.raises(ValueError):p.strict(raw)
    code=tmp_path;write(code/p.PROTOCOL,(REPO/p.PROTOCOL).read_bytes())
    assert p.protocol(code)["native_seconds"]==1200
    value=json.loads((code/p.PROTOCOL).read_bytes());value["target_faces"]=4000
    (code/p.PROTOCOL).chmod(0o644);write(code/p.PROTOCOL,value)
    with pytest.raises(ValueError):p.protocol(code)


@pytest.mark.parametrize("change", [lambda d:d["conditioning"].update(scale=4.),
    lambda d:d["conditioning"].update(chart_refitted=True),
    lambda d:d["conditioning"].update(source_roundtrip_vertices=3),
    lambda d:d["serialization"].update(serialization_safe=False),
    lambda d:d["serialization"].update(committed_collapses=True),
    lambda d:d["native_volume"].update(native_cost_and_placement_unchanged=True),
    lambda d:d["native_volume"].update(volume_relative_limit=.06)])
def test_native_chart_and_legal_collapse_contract(change):
    mesh=tetra(); document=native_document(mesh);change(document)
    with pytest.raises(ValueError):p.validate_mapping(mesh,mesh,document,p.prepare_conditioning(*mesh))


def test_underbudget_identity_is_legal_not_a_static_trajectory():
    mesh=tetra(); doc=native_document(mesh)
    result, evidence=p.validate_mapping(mesh,mesh,doc,p.prepare_conditioning(*mesh))
    assert result["committed_collapses"]==0 and evidence["conditioning"]["chart_refitted"] is False
    changed=(mesh[0]+.01,mesh[1])
    with pytest.raises(ValueError):p.validate_mapping(mesh,changed,doc,p.prepare_conditioning(*mesh))


def test_component_policy_accepts_all_positive_requires_true_hollow_and_blocks_other_nesting(monkeypatch):
    calls=[]
    monkeypatch.setattr(p,"topology_and_embedding",lambda mesh,cavity:calls.append(cavity) or {"checked":True})
    v,f=tetra(); source=(np.r_[v,v+[3.,0.,0.]],np.r_[f,f+4])
    assert p.component_policy(source)[0] is False and calls==[False]
    monkeypatch.setattr(p.geometry,"exact_mesh_topology",lambda *a:{"components":[{"volume_sign":-1},{"volume_sign":1}]})
    assert p.component_policy(source)[0] is True and calls[-1] is True
    monkeypatch.setattr(p.geometry,"exact_mesh_topology",lambda *a:{"components":[{"volume_sign":-1},{"volume_sign":1},{"volume_sign":1}]})
    with pytest.raises(ValueError):p.component_policy(source)


def test_native_once_preserves_input_and_full_f64_mapping(tmp_path,monkeypatch):
    mesh=tetra(); calls=[]
    def run(argv,**kw):
        calls.append((argv,kw));p.geometry.write_obj(Path(argv[2]),*mesh)
        write(Path(argv[3]),native_document(mesh),0o644)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,"run",run)
    result={};candidate,m=p.native_once(mesh,tmp_path/"qualified",tmp_path,lambda:1500.,result)
    assert calls[0][1]["timeout"]==1200 and result["native_attempts"]==1 and result["native_returned"]
    np.testing.assert_array_equal(candidate[0],mesh[0]);assert m["I"]==[0,1,2,3]


@pytest.mark.parametrize("kind",["timeout","changed_input","nonzero"])
def test_native_failure_no_fallback_and_input_posthash(tmp_path,monkeypatch,kind):
    def run(argv,**kw):
        if kind=="timeout":raise p.subprocess.TimeoutExpired(argv,kw["timeout"])
        if kind=="changed_input":Path(argv[1]).write_text("changed")
        return SimpleNamespace(returncode=2, stderr=b"native general rejection")
    monkeypatch.setattr(p.subprocess,"run",run);record={}
    with pytest.raises((ValueError,p.subprocess.TimeoutExpired)):
        p.native_once(tetra(),tmp_path/"binary",tmp_path,lambda:42.,record)
    assert record["native_attempts"]==1 and record["native_elapsed_seconds"]>=0


def prepare_produce(tmp_path,monkeypatch,*,scale=2.5):
    root=tmp_path;mesh=tetra();sorted_v,sorted_f,weld=p.endpoint.exact_weld(*mesh);source=(sorted_v,sorted_f)
    calls=[]
    monkeypatch.setattr(p.endpoint,"prerequisites",lambda r,e:({"video_sha256":"a"*64},{"object.glb":"b"*64},scale))
    monkeypatch.setattr(p,"topology_and_embedding",lambda m,c:{"checked":True})
    def real_stage(src, candidate, mapping, cavity, remaining):
        calls.append((src[0].copy(),candidate[0].copy()));remaining()
        return p.geometry.mapped_geometry(src,candidate,mapping)
    monkeypatch.setattr(p,"stage",real_stage)
    monkeypatch.setattr(p,"native_once",lambda s,*a:(s,mapping(s)))
    class Mesh:
        def __init__(self,*a,**kw):assert kw=={"process":False}
        def export(self,path):path.write_bytes(b"opaque tiny GLB callback")
    monkeypatch.setitem(sys.modules,"trimesh",SimpleNamespace(Trimesh=Mesh))
    monkeypatch.setattr(p.endpoint,"_load_mesh",lambda path:mesh if path.name=="object.glb" else source)
    helper=tmp_path/"helper.py"
    helper.write_text('import numpy as np\ndef budget_mesh(path,faces,vertices):\n assert faces==vertices==4096\n v=np.array('+repr(sorted_v.tolist())+')\n f=np.array('+repr(sorted_f.tolist())+',dtype=np.int64)\n return np.r_[v,np.repeat(v[:1],4096-len(v),axis=0)], np.r_[f,np.zeros((4096-len(f),3),dtype=np.int64)]\n')
    work=tmp_path/"work";work.mkdir()
    return root,helper,work,calls


def test_full_six_stages_actual_math_single_original_scale_and_saved_npz(tmp_path,monkeypatch):
    root,helper,work,calls=prepare_produce(tmp_path,monkeypatch)
    before=tetra()[0].copy();report={}
    files=p.produce(root,9,tmp_path/"binary",helper,work,lambda:100.,report)
    assert [f.name for f in files]==["object_fixed_canonical.glb","geometry.npz"]
    assert list(report["stages"])==list(p.STAGES) and report["metric_scale_baked_once"]==2.5
    assert len(calls)==5 and not report["frame_poses_changed"]
    assert all(s["scale_or_pose_fitted"] is False and s["candidate_topology"]["faces"] == 4
               for k,s in report["stages"].items() if k != "physical_source")
    np.testing.assert_array_equal(calls[-1][0],calls[0][0]*2.5)
    with np.load(files[1],allow_pickle=False) as d:
        assert d["vertices"].shape==(4096,3) and d["faces"].shape==(4096,3)
        assert d["object_scale"].item()==1. and d["grounded_scale_baked"].item()==2.5
    np.testing.assert_array_equal(tetra()[0],before)


def test_source_collisions_are_diagnostic_only_but_candidate_must_be_safe(tmp_path,monkeypatch):
    root,helper,work,calls=prepare_produce(tmp_path,monkeypatch)
    original=p.serialization_preflight;counts=[]
    def preflight(*a):
        result=dict(original(*a));counts.append(1)
        if len(counts)==1:result.update(position_weld_admissible=False,nonexact_weld_collision_groups=2)
        return result
    monkeypatch.setattr(p,"serialization_preflight",preflight);report={}
    p.produce(root,0,tmp_path/"binary",helper,work,lambda:100.,report)
    assert report["source_serialization"]["nonexact_weld_collision_groups"]==2
    assert report["candidate_serialization"]["position_weld_admissible"]


def test_unsafe_candidate_stops_before_export_and_outputs(tmp_path,monkeypatch):
    root,helper,work,_=prepare_produce(tmp_path,monkeypatch);original=p.serialization_preflight;calls=[]
    def preflight(*a):
        result=dict(original(*a));calls.append(1)
        if len(calls)>1:result["position_weld_admissible"]=False
        return result
    monkeypatch.setattr(p,"serialization_preflight",preflight)
    with pytest.raises(ValueError):p.produce(root,0,tmp_path/"binary",helper,work,lambda:100.,{})
    assert not (work/"object_fixed_canonical.glb").exists() and not (work/"geometry.npz").exists()


@pytest.mark.parametrize("episode",[True,1.,-1,30,"9"])
def test_episode_fails_before_any_runtime_or_io(episode):
    with pytest.raises(ValueError):p.main(episode)


def main_fixture(tmp_path,monkeypatch):
    root=tmp_path/"root";out=root/"outputs/episode_000009/object_budget_conditioned";out.mkdir(parents=True)
    code=tmp_path/"code";code.mkdir();binary=tmp_path/"qualified";binary.write_bytes(b"bin")
    monkeypatch.setattr(p,"ROOT",root);monkeypatch.setattr(p.platform,"system",lambda:"Linux")
    monkeypatch.setattr(p.os,"geteuid",lambda:1000)
    native_iter=Path.iterdir
    monkeypatch.setattr(Path,"iterdir",lambda path:iter([Path("lo")]) if path==Path("/sys/class/net") else native_iter(path))
    native_lstat=Path.lstat
    def lstat(path):
        meta=native_lstat(path)
        if path==out:
            return SimpleNamespace(**{k:(1000 if k == "st_uid" else getattr(meta,k)) for k in p.IDENTITY_FIELDS})
        return meta
    monkeypatch.setattr(Path,"lstat",lstat)
    for key,value in dict(WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION="c"*40,WR_IMAGE_ID=p.IMAGE).items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(p,"source_binding",lambda *a:{"source":"same"})
    monkeypatch.setattr(p,"protocol",lambda *a:{})
    monkeypatch.setattr(p,"cache_qualification",lambda *a:(binary,{"cache":"same"}))
    monkeypatch.setattr(p,"runtime_identity",lambda *a:{"runtime":"same"})
    monkeypatch.setattr(p.geometry,"validate_legacy_sources",lambda:{"frozen":"same"})
    monkeypatch.setattr(p.endpoint,"prerequisites",lambda *a:({}, {"source":"same"},2.5))
    helper=root/"vendor/v2d_submission_kit/v2dlb/mesh_budget.py";helper.parent.mkdir(parents=True);helper.write_bytes(b"x"*2031)
    monkeypatch.setattr(p.endpoint,"BUDGET_HELPER_SHA",p.identity(helper)["sha256"])
    return out


def test_main_sealed_missing_cache_failure_receipt_before_native(tmp_path,monkeypatch):
    out=main_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,"cache_qualification",lambda *a:(_ for _ in ()).throw(FileNotFoundError("actual cache absent")))
    monkeypatch.setattr(p,"produce",lambda *a:pytest.fail("Must not run geometry"))
    with pytest.raises(FileNotFoundError):p.main(9)
    r=p.strict((out/"report.json").read_bytes())
    assert r["status"]=="fail" and r["native_attempts"]==0 and r["source_rehashed_after"]
    assert stat.S_IMODE((out/"report.json").stat().st_mode)==0o444


def test_post_source_change_removes_only_own_outputs_and_keeps_failure(tmp_path,monkeypatch):
    out=main_fixture(tmp_path,monkeypatch);counter=[]
    def bound(*a):counter.append(1);return {"source":"same" if len(counter)==1 else "changed"}
    monkeypatch.setattr(p,"source_binding",bound)
    def produce(root,e,b,h,work,left,report):
        files=[work/"object_fixed_canonical.glb",work/"geometry.npz"]
        for f in files:f.write_bytes(b"owned")
        report.update(native_attempts=1,native_returned=True)
        return files
    monkeypatch.setattr(p,"produce",produce)
    with pytest.raises(ValueError):p.main(9)
    assert {f.name for f in out.iterdir()}=={"report.json"}
    assert p.strict((out/"report.json").read_bytes())["own_candidate_outputs_removed"] is True


def test_no_model_compiler_gt_or_historical_solver_execution_in_new_source():
    text=(REPO/"infra/object_budget_conditioned.py").read_text()
    assert "def main(episode):" in text and "timeout=min(NATIVE_SECONDS, remaining())" in text
    assert '"--build-info"' in text and 'volume/"mesh_volume_qem"' in text
    assert "compile_binary(" not in text and "geometry_controls(" not in text and "METRIC_SCALE" not in text
    assert "torch" not in text and "eval_private" not in text and "repair(" not in text


def runtime_fixture(tmp_path,monkeypatch):
    root,code,out,*_=cache_fixture(tmp_path)
    binary,q=p.cache_qualification(root,code)
    volume,base,boost=(tmp_path/n for n in ("volume","base","boost"))
    monkeypatch.setattr(p,"VOLUME",volume);monkeypatch.setattr(p,"BASE",base);monkeypatch.setattr(p,"BOOST",boost)
    write(volume/"mesh_volume_qem.cpp",(code/"infra/mesh_volume_qem.cpp").read_bytes())
    write(base/"mesh_guarded_qem.cpp",(code/"infra/mesh_guarded_qem.cpp").read_bytes())
    oldbinary=write(volume/"mesh_volume_qem",b"old binary",0o555)
    basebinary=write(base/"mesh_guarded_qem",b"base binary",0o555)
    write(base/"source/libigl/header.hpp",b"native source header")
    inventory={"libigl/header.hpp":p.identity(base/"source/libigl/header.hpp")["sha256"]}
    digest=p.hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    inherited=dict(status="pass",libigl_revision="igl",eigen_revision="eigen",binary_sha256=basebinary["sha256"],
                   source_inventory_sha256=digest,pinned_primary_sha256=inventory)
    built=dict(status="pass",source_cpp_sha256=p.identity(code/"infra/mesh_volume_qem.cpp")["sha256"],
               binary_sha256=oldbinary["sha256"],source_inventory_sha256=digest)
    write(base/"build.json",inherited);write(volume/"build.json",built)
    boostpin=write(boost/"multiprecision/cpp_int.hpp",b"native exact arithmetic")
    imagepin=write(root/"results/image-volume-qem.json",built|dict(image_id=p.IMAGE))
    config=dict(original_build_receipt=imagepin,original_volume_cpp_sha256=built["source_cpp_sha256"],
                original_binary_sha256=oldbinary["sha256"],libigl_revision="igl",eigen_revision="eigen",
                boost_cpp_int_header_sha256=boostpin["sha256"])
    write(code/"configs/mesh_serialization_compiler_protocol_v1.json",dict(source_authentication=config))
    boost_inventory={"multiprecision/cpp_int.hpp":boostpin["sha256"]}
    q["original_runtime"]=dict(original_build=p.identity(volume/"build.json"),base_build=p.identity(base/"build.json"),
        original_binary=oldbinary,inherited_inventory_sha256=digest,boost_files=1,
        boost_inventory_sha256=p.hashlib.sha256(json.dumps(boost_inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest())
    calls=[]
    def output(argv,**kw):
        calls.append((argv,kw));return json.dumps(q["build_info"]).encode()
    monkeypatch.setattr(p.subprocess,"check_output",output)
    return root,code,q,calls


def test_live_runtime_rehash_and_only_fast_build_info_no_old_execute(tmp_path,monkeypatch):
    root,code,q,calls=runtime_fixture(tmp_path,monkeypatch)
    evidence=p.runtime_identity(root,code,q,lambda:100.)
    assert evidence["original_binary"]==q["original_runtime"]["original_binary"]
    assert len(calls)==1 and calls[0][0][1:]==["--build-info"]
    assert "mesh-conditioned-cache-" in calls[0][0][0] and calls[0][1]["timeout"]==10.


def test_live_fast_build_info_mismatch_before_geometry(tmp_path,monkeypatch):
    root,code,q,_=runtime_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p.subprocess,"check_output",lambda *a,**kw:b'{"physical_coordinate_cache":false}')
    with pytest.raises(ValueError,match="ABI differs"):p.runtime_identity(root,code,q,lambda:100.)


def test_changed_native_header_rejected_without_any_executable(tmp_path,monkeypatch):
    root,code,q,calls=runtime_fixture(tmp_path,monkeypatch)
    header=p.BASE/"source/libigl/header.hpp";header.chmod(0o644);header.write_bytes(b"changed header")
    with pytest.raises(ValueError,match="inventory"):p.runtime_identity(root,code,q,lambda:100.)
    assert calls==[]


def test_positive_nested_shells_rejected_without_union_repair(monkeypatch):
    v,f=tetra();mesh=(np.r_[v,(v*.1)+.1],np.r_[f,f+4])
    monkeypatch.setattr(p,"topology_and_embedding",lambda *a:{"nonintersecting":True})
    with pytest.raises(ValueError,match="independent"):p.component_policy(mesh)


def test_bounded_scratch_cleanup_preserves_unknown_or_hardlinked_files(tmp_path):
    work=tmp_path/"work";work.mkdir();owner=work.lstat()
    file=work/"input.obj";file.write_bytes(b"original")
    unknown=work/"foreign";unknown.write_bytes(b"not owned known filename")
    with pytest.raises(ValueError):p.cleanup_scratch(work,owner)
    assert file.exists() and unknown.exists()
    unknown.unlink();link=tmp_path/"link";link.hardlink_to(file)
    with pytest.raises(ValueError):p.cleanup_scratch(work,owner)
    assert file.exists() and link.exists()
    link.unlink();p.cleanup_scratch(work,owner)
    assert not work.exists()


def test_full_main_tiny_success_seals_source_outputs_and_input_posthash(tmp_path,monkeypatch):
    out=main_fixture(tmp_path,monkeypatch)
    def produce(root,e,b,h,work,left,report):
        files=[work/"object_fixed_canonical.glb",work/"geometry.npz"]
        for f in files:f.write_bytes(b"owned callback bytes")
        report.update(native_attempts=1,native_returned=True)
        return files
    monkeypatch.setattr(p,"produce",produce)
    report=p.main(9)
    assert report["status"]=="pass" and report["inputs_rehashed_after"] and report["cache_rehashed_after"]
    assert report["runtime_rehashed_after"] and report["owned_scratch_removed"]
    assert report["script_sha256"]==p.identity(Path(p.__file__))["sha256"]
    assert all(stat.S_IMODE(x.stat().st_mode)==0o444 for x in out.iterdir())


def test_input_posthash_failure_also_closes_proposal(tmp_path,monkeypatch):
    out=main_fixture(tmp_path,monkeypatch);calls=[]
    def inputs(*a):
        calls.append(1);return ({},{"source":"same" if len(calls)==1 else "changed"},2.5)
    monkeypatch.setattr(p.endpoint,"prerequisites",inputs)
    monkeypatch.setattr(p,"produce",lambda *a:(_ for _ in ()).throw(ValueError("Source numeric failure")))
    with pytest.raises(ValueError,match="Frozen input"):p.main(9)
    report=p.strict((out/"report.json").read_bytes())
    assert report["status"]=="fail" and report["cache_rehashed_after"] and report["owned_scratch_removed"]

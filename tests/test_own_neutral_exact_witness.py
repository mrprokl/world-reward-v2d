"""Tiny source/provenance/one-call diagnostic fixtures, never native execution."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("own_neutral_exact_witness_test", ROOT / "infra/own_neutral_exact_witness.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture_report(g):
    report = dict(stage="world_reward_own_native_surface_grasp_capability", status="fail",
        phase="neutral_full_surface_prerequisite", producer_revision=g.ORIGINAL_REVISION,
        script_sha256=g.ORIGINAL_SCRIPT_SHA, image_id=g.IMAGE, network="none", budget_seconds=300,
        max_native_forwards=100, own_fresh_procedural_geometry_only=True, historical_poses_results_read=False,
        challenge_inputs_used=False, ground_truth_read=False, hand_labeled_test=False, RGB_produced=False,
        public_manifest_produced=False, quality_verified=False, challenge_performance_verified=False,
        force_closure_verified=False, biomechanics_verified=False, touching_certified=False, motion_verified=False,
        adoption_authorized=False, fixed_identity="zero45", fixed_scales="zero68", fixed_expression="zero72",
        object_scale=1., target_positive_gap_m=.0005, geometry_frame="native_cm_to_metres_proper_diag_1_minus1_minus1",
        error_type="ValueError", error="Fresh full original neutral closed/outward/self-embedded surface prerequisite failed",
        model=dict(sha256=g.original.MODEL_SHA, bytes=g.original.MODEL_BYTES),
        runtime=dict(torch="2.5.1+cu124", CUDA="12.4", deterministic_algorithms=True, warn_only=False,
            JIT_optimized=False, TF32=False, seed=0, threads=4),
        native_calls=[dict(index=1, phase="fresh_neutral", batch_size=1, attempted=True, returned=True, validated=True)],
        source_helpers={name:dict(sha256="a"*64, bytes=100) for name in g.original.HELPERS},
        metadata=dict(parameter_names=["own"+str(i) for i in range(249)], joint_names=["own_joint"+str(i) for i in range(127)],
            parents_sha256="a"*64, transform_sha256="a"*64, bounds_sha256="a"*64, faces_sha256="a"*64, lbs_sha256=["a"*64]*2))
    report["source_helpers"]["infra/own_grasp_capability.py"]["sha256"] = g.ORIGINAL_SCRIPT_SHA
    flags = [dict(faces=[2*i,2*i+1], kind="proper_intersection" if i<110 else "boundary_contact") for i in range(114)]
    report["neutral_surface"] = dict(schema="world-reward-own-closed-surface-certificate-v1", status="fail",
        tolerance_m=1e-8, full_original_faces_retained=True, exact_arithmetic_proof=False, max_candidate_pairs=2000000,
        budget_seconds=30., mesh=dict(vertices=18439, faces=36874, components=1, original_face_coverage=36874,
            topology_closed_oriented=True, embedding_verified=False, original_vertices_sha256=g.VERTICES_SHA,
            original_faces_sha256=g.FACES_SHA),
        embedding=dict(verified=False, candidate_pairs=240760, permitted_shared_simplex_pairs=226454,
            nested_or_ambiguous_components=[], forbidden_intersections=flags))
    return report


def tiny_geometry(g, monkeypatch):
    faces = np.zeros((36874,3), np.int32)
    for i in range(114):
        first = np.arange(6*i,6*i+3, dtype=np.int32); last = np.arange(6*i+3,6*i+6, dtype=np.int32)
        if i in (108,109,112,113): last[0] = first[0]
        faces[2*i], faces[2*i+1] = first, last
    vertices = np.zeros((18439,3), np.float32)
    a = np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]], np.float32)
    b = np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]], np.float32)
    for i in range(8): vertices[faces[2*i]], vertices[faces[2*i+1]] = a, b
    monkeypatch.setattr(g, "VERTICES_SHA", g.original.array_id(vertices))
    monkeypatch.setattr(g, "FACES_SHA", g.original.array_id(faces))
    return vertices, faces


def test_exact_independent_original_provenance_and_native_geometry_pins(gate):
    assert gate.ORIGINAL_RECEIPT == dict(sha256="38c56607f2abfbae7f53a1fce55e55970f97b2f4aca740773ddcc161f9ff5391", bytes=18294)
    assert gate.VERTICES_SHA == "4f862130b7bafffd983cc776cf19a82de4d27085a7c58c501d55d88269bc00de"
    assert gate.FACES_SHA == "51d08a7d2e92893ca42c7525bda768afd5c223a33d8302b6674c832e943e8788"
    assert gate.EXACT_HELPER_SHA == hashlib.sha256((ROOT/"src/world_reward/exact_triangle_witness.py").read_bytes()).hexdigest()
    assert len(gate.validate_original_report(fixture_report(gate))) == 114


@pytest.mark.parametrize("field", ["stage","status","phase","producer_revision","script_sha256","image_id","network",
    "budget_seconds","max_native_forwards","own_fresh_procedural_geometry_only","historical_poses_results_read",
    "challenge_inputs_used","ground_truth_read","hand_labeled_test","RGB_produced","public_manifest_produced",
    "quality_verified","challenge_performance_verified","touching_certified","force_closure_verified","motion_verified",
    "adoption_authorized","fixed_identity","fixed_scales","fixed_expression","object_scale","target_positive_gap_m",
    "error","error_type"])
def test_original_report_field_mismatch_fails_without_native(gate, field):
    report = fixture_report(gate)
    report[field] = not report[field] if type(report[field]) is bool else None
    with pytest.raises(ValueError): gate.validate_original_report(report)


@pytest.mark.parametrize("fault", ["calls_missing","calls_two","unvalidated","wrongcallkey","runtime","modelbytes",
    "geometryhash","facehash","boolcount","tolerance","embeddingpass","nested","flags_missing","flags_duplicate",
    "flags_reverse","flags_bool","flags_kind","after_gradient","oldhelpers","driverSHA","metadata_missing"])
def test_strict_actual_receipt_fields_fail_closed(gate, fault):
    report = fixture_report(gate); neutral = report["neutral_surface"]; flags = neutral["embedding"]["forbidden_intersections"]
    if fault=="calls_missing": report.pop("native_calls")
    elif fault=="calls_two": report["native_calls"]*=2
    elif fault=="unvalidated": report["native_calls"][0]["validated"]=False
    elif fault=="wrongcallkey": report["native_forward_calls"]=report.pop("native_calls")
    elif fault=="runtime": report["runtime"]["JIT_optimized"]=True
    elif fault=="modelbytes": report["model"]["bytes"]-=1
    elif fault=="geometryhash": neutral["mesh"]["original_vertices_sha256"]="0"*64
    elif fault=="facehash": neutral["mesh"]["original_faces_sha256"]="0"*64
    elif fault=="boolcount": neutral["mesh"]["components"]=True
    elif fault=="tolerance": neutral["tolerance_m"]*=2
    elif fault=="embeddingpass": neutral["embedding"]["verified"]=True
    elif fault=="nested": neutral["embedding"]["nested_or_ambiguous_components"]=[0]
    elif fault=="flags_missing": flags.pop()
    elif fault=="flags_duplicate": flags[1]=copy.deepcopy(flags[0])
    elif fault=="flags_reverse": flags[0]["faces"]=flags[0]["faces"][::-1]
    elif fault=="flags_bool": flags[0]["faces"][0]=False
    elif fault=="flags_kind": flags[0]["kind"]="ambiguous"
    elif fault=="after_gradient": report["native_gradient_probe"]={}
    elif fault=="oldhelpers": report["source_helpers"].pop("src/world_reward/cross_surface.py")
    elif fault=="driverSHA": report["source_helpers"]["infra/own_grasp_capability.py"]["sha256"]="0"*64
    else: report.pop("metadata")
    with pytest.raises(ValueError): gate.validate_original_report(report)


def test_hash_and_size_gate_precedes_any_original_json(gate, tmp_path, monkeypatch):
    path = tmp_path/"report.json"; path.write_text("not JSON"); path.chmod(0o400)
    monkeypatch.setattr(gate.json, "loads", lambda value: pytest.fail("Unpinned JSON must not be parsed"))
    with pytest.raises(ValueError, match="SHA/bytes"): gate.read_original_report(path)


def test_pinned_synthetic_receipt_reread_and_readonly_required(gate, tmp_path, monkeypatch):
    path = tmp_path/"report.json"; path.write_text(json.dumps(fixture_report(gate))); path.chmod(0o400)
    expected = gate.original.identity(path, True); monkeypatch.setattr(gate,"ORIGINAL_RECEIPT",expected)
    report, identity = gate.read_original_report(path)
    assert identity==expected and report["status"]=="fail"
    path.chmod(0o600)
    with pytest.raises(ValueError): gate.read_original_report(path)


def test_native_vertex_face_identity_and_first8_nonadjacent_selection(gate, monkeypatch):
    vertices, faces = tiny_geometry(gate, monkeypatch); report=fixture_report(gate)
    gate.verify_native_arrays(vertices, faces)
    selected, counts = gate.select_original_pairs(report, faces)
    assert [row["faces"] for row in selected]==[[2*i,2*i+1] for i in range(8)]
    assert all(not row["common_vertex_ids"] for row in selected)
    assert sum(row["count"] for row in counts)==114
    faces64=faces.astype(np.int64)
    with pytest.raises(ValueError): gate.select_original_pairs(report, faces64)
    vertices[0,0]+=.001
    with pytest.raises(ValueError): gate.verify_native_arrays(vertices, faces)


def test_adjacency_classification_is_not_wholesale_skip_or_reclassification(gate, monkeypatch):
    vertices, faces=tiny_geometry(gate,monkeypatch); faces[1,0]=faces[0,0]
    monkeypatch.setattr(gate,"FACES_SHA",gate.original.array_id(faces))
    with pytest.raises(ValueError,match="classification"): gate.select_original_pairs(fixture_report(gate),faces)


def test_one_actual_attempt_return_validation_and_never_retry(gate):
    report={}; events=[]
    value=gate.one_native_forward(report,lambda:events.append("invoke")or"own-value",lambda result:events.append(result),lambda:None)
    assert value=="own-value" and events==["invoke","own-value"]
    assert report["native_calls"][0]==dict(index=1,phase="hash_identical_fresh_neutral",batch_size=1,attempted=True,returned=True,validated=True)
    with pytest.raises(RuntimeError): gate.one_native_forward(report,lambda:pytest.fail("Second native call"),lambda value:None,lambda:None)


@pytest.mark.parametrize("failure",["invoke","validate"])
def test_failed_native_attempt_stays_counted_without_replay(gate,failure):
    report={}
    def fail(): raise ValueError("own tiny failure")
    with pytest.raises(ValueError): gate.one_native_forward(report,fail if failure=="invoke"else lambda:"own",lambda value:fail(),lambda:None)
    row=report["native_calls"][0]
    assert row["attempted"] and not row["validated"] and row["returned"]==(failure=="validate")
    with pytest.raises(RuntimeError): gate.one_native_forward(report,lambda:"own",lambda value:None,lambda:None)


def test_exact_pair_inspection_reportonly_and_source_geometry_preserved(gate,monkeypatch):
    vertices,faces=tiny_geometry(gate,monkeypatch); selected,_=gate.select_original_pairs(fixture_report(gate),faces)
    report=dict(selected_pairs=selected); before=vertices.tobytes(),faces.tobytes()
    gate.inspect_pairs(vertices,faces,selected,report,lambda:None)
    assert report["strict_transverse_witness_count"]==8 and len(report["pairs"])==8
    assert (vertices.tobytes(),faces.tobytes())==before
    assert all(row["exact"]["budgets"]["seconds"]==2 for row in report["pairs"])
    assert all(row["exact"]["full_separation_proven"] is False for row in report["pairs"])
    assert json.loads(json.dumps(report,allow_nan=False))==report
    with pytest.raises(ValueError): gate.inspect_pairs(vertices,faces,selected,report,lambda:None)


def test_exact_pair_negative_is_diagnostic_not_separation(gate,monkeypatch):
    vertices,faces=tiny_geometry(gate,monkeypatch); selected,_=gate.select_original_pairs(fixture_report(gate),faces)
    vertices[faces[1],2]+=10
    monkeypatch.setattr(gate,"VERTICES_SHA",gate.original.array_id(vertices))
    report=dict(selected_pairs=selected);gate.inspect_pairs(vertices,faces,selected,report,lambda:None)
    assert report["strict_transverse_witness_count"]==7
    assert report["pairs"][0]["exact"]["strict_transverse_witness"] is False
    assert report["pairs"][0]["exact"]["full_separation_proven"] is False


def test_exact_budget_failure_does_not_reselect_or_invent_results(gate,monkeypatch):
    vertices,faces=tiny_geometry(gate,monkeypatch);selected,_=gate.select_original_pairs(fixture_report(gate),faces)
    calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise RuntimeError("tiny exact budget")
    monkeypatch.setattr(gate,"exact_triangle_witness",fail);report=dict(selected_pairs=selected)
    with pytest.raises(RuntimeError):gate.inspect_pairs(vertices,faces,selected,report,lambda:None)
    assert len(calls)==1 and report["pairs"]==[]


@pytest.mark.parametrize("bundle",["run_own_grasp_capability","own_neutral_exact_witness","run_own_neutral_exact_witness"])
def test_actual_dispatch_namespace_mandatory(gate,tmp_path,bundle):
    root=tmp_path;revision="a"*40;code=root/"jobs"/revision/bundle/"code";code.mkdir(parents=True)
    if bundle=="run_own_neutral_exact_witness":assert gate.bound_code(root,code,revision)==code
    else:
        with pytest.raises(ValueError):gate.bound_code(root,code,revision)


def test_additive_no_gt_media_optimize_or_model_local_wrapper(gate):
    source=(ROOT/"infra/own_neutral_exact_witness.py").read_text();tree=ast.parse(source)
    ownimports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    assert ownimports<={"__future__","collections","pathlib","world_reward.exact_triangle_witness"}
    assert source.count("result = model(identity, q[None], expression, True)")==1
    assert "torch.no_grad"not in source and "torch.inference_mode"not in source and "np.save"not in source
    assert gate.BUDGET==120 and gate.PAIR_BUDGET==2.0 and gate.MAX_PAIRS==8
    assert gate.OUTPUT=="validation/own_neutral_exact_witness_v1"
    shell=ROOT/"infra/run_own_neutral_exact_witness.sh";text=shell.read_text()
    mounts=[line for line in text.splitlines() if "--mount"in line]
    assert len(mounts)==4 and sum("readonly"in line for line in mounts)==3
    assert 'src=$RECEIPT,dst=$RECEIPT,readonly'in text and "--network none"in text
    assert "hostname"not in text and "--entrypoint python"in text
    assert "flock --nonblock 9" in text and "nvidia-smi --query-compute-apps=pid" in text
    assert text.index("flock --nonblock 9") < text.index('mkdir -m 700 "$OUT"')
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    result=subprocess.run(["rtk","proxy","bash",str(shell),"--episode","15"],capture_output=True,env={"PATH":os.environ["PATH"]})
    assert result.returncode==2


def test_actual_peer_driver_import_does_not_load_torch_or_joblib():
    code="import sys; sys.path[:0]=sys.argv[1:3]; import own_neutral_exact_witness; assert 'torch' not in sys.modules and 'joblib' not in sys.modules"
    result=subprocess.run(["rtk","proxy",sys.executable,"-I","-B","-c",code,str(ROOT/"infra"),str(ROOT/"src")],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize("fails", [False, True])
def test_private_report_only_lifecycle_and_failed_attempt_preservation(gate,tmp_path,monkeypatch,fails):
    root=tmp_path;revision="a"*40;code=root/"jobs"/revision/"run_own_neutral_exact_witness"/"code"
    code.mkdir(parents=True);out=root/gate.OUTPUT;out.mkdir(parents=True,mode=0o700)
    script=code/"infra/own_neutral_exact_witness.py";script.parent.mkdir();script.write_text("own tiny source fixture");script.chmod(0o400)
    monkeypatch.setattr(gate,"ROOT",root);monkeypatch.setattr(gate,"__file__",str(script))
    monkeypatch.setattr(gate.platform,"system",lambda:"Linux");monkeypatch.setattr(gate.os,"geteuid",lambda:1000)
    actual_iter=Path.iterdir
    monkeypatch.setattr(Path,"iterdir",lambda path:iter([Path("lo")])if str(path)=="/sys/class/net"else actual_iter(path))
    for key,value in dict(WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,
            WR_IMAGE_ID=gate.IMAGE,CUBLAS_WORKSPACE_CONFIG=":4096:8").items():monkeypatch.setenv(key,value)
    def synthetic_run(root,code,out,report,persist):
        report["native_calls"]=[dict(attempted=True,returned=False,validated=False)];persist()
        if fails:raise ValueError("own mocked diagnostic failure")
        report.update(status="pass",phase="complete",diagnostic_complete=True)
    monkeypatch.setattr(gate,"run",synthetic_run)
    if fails:
        with pytest.raises(ValueError,match="mocked diagnostic"):gate.main([])
    else:gate.main([])
    assert {path.name for path in out.iterdir()}=={"report.json"}
    receipt=out/"report.json";report=json.loads(receipt.read_text())
    assert receipt.stat().st_mode&0o777==0o400
    assert report["status"]==("fail"if fails else"pass") and report["native_calls"][0]["attempted"]
    assert report["original_v2_reclassified"] is False and report["geometry_embedding_certified"] is False
    assert report["geometry_payload_produced"] is False and report["optimization_performed"] is False

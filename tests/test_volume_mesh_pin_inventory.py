"""Tiny stdlib report inventories, never real NPZ/GLB/model/video decoding."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
REVISION="3242cb23486726c9636ca66241020f6aa276be1b"
SCRIPT="bb6b17696ed928b749e805ed6405fb9144419421fbef70691a2dd8235cc714e9"


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location("volume_mesh_pin_inventory_test",ROOT/"infra/volume_mesh_pin_inventory.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def artifacts(gate,tmp_path):
    p=gate.paths(2);image="sha256:"+"b"*64;video="1"*64;scale=2.5
    def write(role,value):
        path=tmp_path/p[role];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value)if isinstance(value,dict)else value);return gate.identity(path)
    geometry=write("geometry","own tiny bytes not NPZ");glb=write("glb","own tiny bytes not GLB")
    build=write("build",dict(stage="world_reward_volume_qem_build",status="pass",image_id=image))
    evidence=dict(build_report_sha256=build["sha256"],binary_sha256="c"*64,build_info={"source_sha256":"d"*64})
    control=write("control",dict(stage="own_volume_constrained_intersection_qem_geometry",status="pass",script_sha256=gate.CONTROL_SHA,
        challenge_inputs_used=False,adoption_performed=False,target_faces=4096,target_vertices=4096,build=evidence))
    flags=dict(status="pass",episode_index=2,input_track="track_1",input_sha256=video,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    alignment=dict(flags,stage="predicted_human_anchored_moge2_pointmaps",coordinate_frame="OpenCV_x_right_y_down_z_forward",
        pointmap_scale_application="one_clip_scalar_to_MoGe2_XYZ_already_applied",body_report_sha256="e"*64,depth_report_sha256="f"*64)
    alignment_id=write("alignment",alignment)
    obj=dict(flags,stage="sam3d_objects_grounded_fixed_frame",frame_index=0,scale_source="already_human_anchored_MoGe2_no_second_scalar",
        pointmap_grounding={"alignment_report_sha256":alignment_id["sha256"]},transform={"scale":[scale]*3},
        object_sha256="2"*64,transform_sha256="3"*64,intrinsics_sha256="4"*64)
    object_id=write("object",obj)
    metrics=dict(birthface_matched_shells=[{"relative_volume_error":.001}],sampled_bidirectional_chamfer_diagonal_ratio=.001,net_volume_relative_error=.001)
    report=dict(flags,stage=gate.STAGE,producer_revision=REVISION,script_sha256=SCRIPT,image_id=image,
        adoption_performed=False,challenge_performance_verified=False,budget_seconds=900,target_faces=4096,target_vertices=4096,
        source_shell_volume_relative_limit=.05,components_deleted=False,holes_filled=False,normals_repaired=False,
        frame_poses_changed=False,native_cost_and_placement_unchanged=True,source_embedding_exact_universal_proof=False,
        source_intersecting_faces=0,independent_candidate_intersecting_faces=0,packed_intersecting_faces=0,metric_scale_baked_once=scale,
        source_arrays_unchanged=True,official_helper_sha256=gate.OFFICIAL_HELPER_SHA,geometry_sha256=geometry["sha256"],canonical_glb_sha256=glb["sha256"],
        candidate_geometry=copy.deepcopy(metrics),export_geometry=copy.deepcopy(metrics),
        official_pack_fidelity=dict(oriented_triangles_exact=True,official_helper_simplification_invoked=False,nonexact_merge_or_face_deletion=False),
        source_hashes=dict(object_report=object_id["sha256"],alignment_report=alignment_id["sha256"],video=video,
            **{"object.glb":obj["object_sha256"],"transform.json":obj["transform_sha256"],"intrinsics.json":obj["intrinsics_sha256"]},
            body_smoke_report=alignment["body_report_sha256"],depth_smoke_report=alignment["depth_report_sha256"],mask_report="5"*64,prompts="6"*64),
        control_evidence=dict(evidence,volume_gate_sha256=control["sha256"]))
    write("report",report)
    return gate,tmp_path,p,report,video,object_id["sha256"],alignment_id["sha256"],scale


def call(data,**kwargs):
    gate,root,p,report,video,obj,alignment,scale=data
    return gate.inventory(root,2,kwargs.get("revision",REVISION),kwargs.get("script",SCRIPT),video,obj,alignment,scale)


def test_all7_files_no_decoding_actual_producer_ids_0644_and_stdout_schema(artifacts):
    gate,root,p,report,video,obj,alignment,scale=artifacts;pins=call(artifacts)
    assert set(pins["files"])==set(p.values()) and len(pins["files"])==7
    assert pins["report"]["producer_revision"]==REVISION and pins["report"]["script_sha256"]==SCRIPT
    assert gate.verify_pinned_artifacts(root,pins,2,video,obj,alignment,scale)[0]==report
    assert (root/p["geometry"]).read_text()=="own tiny bytes not NPZ"
    assert json.loads(json.dumps(pins))==pins


@pytest.mark.parametrize("field",["producer_revision","script_sha256","input_sha256","ground_truth_used","oracle_modes","episode_index",
    "components_deleted","holes_filled","normals_repaired","frame_poses_changed","source_intersecting_faces",
    "independent_candidate_intersecting_faces","packed_intersecting_faces","source_arrays_unchanged","official_helper_sha256","target_faces"])
def test_current_report_contract_mismatch_cannot_be_resealed(artifacts,field):
    gate,root,p,report,*_=artifacts;report[field]=None;(root/p["report"]).write_text(json.dumps(report))
    with pytest.raises(ValueError):call(artifacts)


@pytest.mark.parametrize("field",["candidate_geometry","export_geometry"])
@pytest.mark.parametrize("bad",[.010001,True,float("nan")])
def test_fixed_cd_threshold_and_real_scalar_types(gate,artifacts,field,bad):
    _,root,p,report,*_=artifacts;report[field]["sampled_bidirectional_chamfer_diagonal_ratio"]=bad
    (root/p["report"]).write_text(json.dumps(report))
    with pytest.raises(ValueError):call(artifacts)


@pytest.mark.parametrize("fault",["control","build","object","alignment","double_scale","shellvolume","helper_simplify","lost_source"])
def test_source_control_and_metric_ancestry_failclosed(artifacts,fault):
    gate,root,p,report,*_=artifacts
    if fault in ("control","build","object","alignment"):
        value=json.loads((root/p[fault]).read_text());value["status"]="fail";(root/p[fault]).write_text(json.dumps(value))
    elif fault=="double_scale":report["metric_scale_baked_once"]*=2
    elif fault=="shellvolume":report["candidate_geometry"]["birthface_matched_shells"][0]["relative_volume_error"]=.051
    elif fault=="helper_simplify":report["official_pack_fidelity"]["official_helper_simplification_invoked"]=True
    else:report["source_hashes"].pop("video")
    (root/p["report"]).write_text(json.dumps(report))
    with pytest.raises(ValueError):call(artifacts)


def test_independent_expected_source_revision_not_inferred_from_receipt(artifacts):
    with pytest.raises(ValueError):call(artifacts,revision="f"*40)
    with pytest.raises(ValueError):call(artifacts,script="f"*64)


def test_all_pinned_bytes_checked_before_any_json(artifacts,monkeypatch):
    gate,root,p,report,video,obj,alignment,scale=artifacts;pins=call(artifacts)
    (root/p["geometry"]).write_text("changed tiny bytes")
    monkeypatch.setattr(gate,"strict_json",lambda text:pytest.fail("JSON before all pinned bytes"))
    with pytest.raises(ValueError,match="before JSON"):gate.verify_pinned_artifacts(root,pins,2,video,obj,alignment,scale)


@pytest.mark.parametrize("field",["report","geometry","glb","control","build","object","alignment"])
def test_exact7_payloads_rehashed_and_bytes_pin_independent(artifacts,field):
    gate,root,p,report,video,obj,alignment,scale=artifacts;pins=call(artifacts)
    with (root/p[field]).open("a")as stream:stream.write(" ")
    with pytest.raises(ValueError):gate.verify_pinned_artifacts(root,pins,2,video,obj,alignment,scale)


def test_symlink_ancestor_and_duplicate_json_keys_rejected(artifacts,tmp_path):
    gate,root,p,*_=artifacts
    alias=root/"alias";alias.symlink_to(root/"outputs",target_is_directory=True)
    with pytest.raises(ValueError):gate.identity(alias/"episode_000002/object_budget_volume/report.json")
    with pytest.raises(ValueError):gate.strict_json('{"status":"pass","status":"fail"}')


@pytest.mark.parametrize("fault",["extra_file","missing_file","extra_top","reportboolbytes","wrongsource","wrongscale","reporthash"])
def test_manifest_is_exact_independently_typed_and_sourcebound(artifacts,fault):
    gate,root,p,report,video,obj,alignment,scale=artifacts;pins=call(artifacts)
    if fault=="extra_file":pins["files"]["outputs/episode_000002/not_allowed"]={"sha256":"0"*64,"bytes":1}
    elif fault=="missing_file":pins["files"].pop(p["glb"])
    elif fault=="extra_top":pins["oracle"]="GT"
    elif fault=="reportboolbytes":pins["report"]["bytes"]=True
    elif fault=="wrongsource":pins["files"][p["object"]]["sha256"]="0"*64
    elif fault=="wrongscale":pins["metric_scale_baked_once"]=int(scale)
    else:pins["report"]["sha256"]="0"*64
    with pytest.raises(ValueError):gate.verify_pinned_artifacts(root,pins,2,video,obj,alignment,scale)


def test_changed_artifact_after_json_is_rejected_by_posthash(artifacts,monkeypatch):
    gate,root,p,report,video,obj,alignment,scale=artifacts;pins=call(artifacts);original=gate.validate_reports
    def mutate(records,pins):
        value=original(records,pins);(root/p["glb"]).write_text("mutation after reports");return value
    monkeypatch.setattr(gate,"validate_reports",mutate)
    with pytest.raises(ValueError,match="changed during report"):
        gate.verify_pinned_artifacts(root,pins,2,video,obj,alignment,scale)


def test_import_is_stdlib_only_no_numpy_numeric_closure(gate):
    path=ROOT/"infra/volume_mesh_pin_inventory.py";tree=ast.parse(path.read_text())
    imports={node.module for node in ast.walk(tree)if isinstance(node,ast.ImportFrom)}
    imports|={alias.name for node in ast.walk(tree)if isinstance(node,ast.Import)for alias in node.names}
    assert imports<={"__future__","argparse","hashlib","json","math","os","pathlib","platform","re","stat","sys"}
    code="import importlib.util,sys;s=importlib.util.spec_from_file_location('v',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert not any(n.split('.')[0]in{'numpy','scipy','torch','joblib','trimesh'}for n in sys.modules)"
    result=subprocess.run(["rtk","proxy",sys.executable,"-I","-B","-c",code,str(path)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    subprocess.run(["rtk","proxy","bash","-n",str(ROOT/"infra/run_volume_mesh_pin_inventory.sh")],check=True)


@pytest.mark.parametrize("fault",[None,"namespace","entrypoint","writable","symlink"])
def test_actual_source_dispatch_and_producer_file_frozen(gate,tmp_path,fault):
    revision="a"*40;code=tmp_path/"jobs"/revision/"run_volume_mesh_pin_inventory"/"code";code.mkdir(parents=True)
    for name in gate.HELPERS+(gate.PRODUCER_SCRIPT,):
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("own source fixture");path.chmod(0o400)
    executing=code/gate.HELPERS[0]
    if fault=="namespace":code=code.parent
    elif fault=="entrypoint":executing=code/"wrong.py"
    elif fault=="writable":(code/gate.PRODUCER_SCRIPT).chmod(0o600)
    elif fault=="symlink":
        path=code/gate.PRODUCER_SCRIPT;path.unlink();path.symlink_to(code/gate.HELPERS[0])
    if fault is None:assert set(gate.bound_source(tmp_path,code,revision,executing))==set(gate.HELPERS+(gate.PRODUCER_SCRIPT,))
    else:
        with pytest.raises(ValueError):gate.bound_source(tmp_path,code,revision,executing)


@pytest.mark.parametrize("args",[[],["--episode","02"],["--episode","30"],["--episode","2","--episode","3"]])
def test_explicit_complete_controls_required(gate,args):
    with pytest.raises(SystemExit):gate.parser().parse_args(args)


def test_cli_full_explicit_sources_and_duplicate_rejection(gate):
    args=["--episode","2","--producer-revision",REVISION,"--producer-script-sha256",SCRIPT,"--input-sha256","1"*64,
        "--object-report-sha256","2"*64,"--alignment-report-sha256","3"*64,"--scale","2.5"]
    actual=gate.parser().parse_args(args);assert actual.episode==2 and actual.scale==2.5
    for suffix in (["--episode","3"],["--scale","3"]):
        with pytest.raises(SystemExit):gate.parser().parse_args(args+suffix)
    args[1]="02"
    with pytest.raises(SystemExit):gate.parser().parse_args(args)

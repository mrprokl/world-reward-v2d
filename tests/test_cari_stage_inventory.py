"""Data-free immutable producer inventory, explicit trust and stdout-only CLI."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location("cari_stage_inventory_test",ROOT/"infra/cari_stage_inventory.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_text(json.dumps(value)) if isinstance(value,dict) else path.write_bytes(value)
    path.chmod(0o444)


def fixture(gate,root,stage,episode=15,count=501):
    root=root.resolve();directory=root/gate.output_relative(episode,stage)
    for name in gate.STAGES[stage]["payloads"]:write(directory/name,("tiny automatic producer "+name).encode())
    payloads={name:gate.identity(directory/name) for name in gate.STAGES[stage]["payloads"]}
    report=dict(stage=gate.STAGES[stage]["stage"],status="pass",phase="complete",episode_index=episode,frames=count,
        clip_spec=dict(episode_index=episode,total_frames=count,camera_name="front_stereo_camera_left",height=1152,width=1536),
        original_frame_indices=list(range(count)),producer_revision="a"*40,script_sha256="b"*64,image_id=gate.IMAGE,
        input_track="track_1",network="none",ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,
        oracle_modes=[],quality_verified=False,adoption_authorized=False,submission_produced=False,
        input_dataset_revision=gate.DATASET_REVISION,source_inputs_assets_rehashed=True,source_helpers_rehashed=True,
        input_sha256="c"*64,input_report_sha256="d"*64,source_helpers={gate.STAGES[stage]["script"]:dict(sha256="b"*64,bytes=2468)},
        output_files=payloads)
    if stage=="export":
        for key in ("input_dataset_revision","input_sha256","input_report_sha256"):report.pop(key)
        chunks=gate._chunks(count,16)
        sources={name:dict(sha256="0"*64,bytes=123) for name in gate._public_source_paths(report["clip_spec"])}
        mesh=f"outputs/episode_{episode:06d}/cari_inputs/export/episode_{episode:06d}/object_mesh/output_aligned.glb"
        sources[mesh]=dict(payloads["object_aligned.glb"])
        report.update(chunk_counts=chunks,budget_seconds=600,ground_truth_read=False,learned_inference_calls=0,
            optimizer_calls=0,converter_LM_calls=0,identity_reselection_performed=False,submission_eligible=False,
            final_Parquet_produced=False,predictions_frozen_before_replays=True,unchanged_refined_predictions_verified=True,
            stored_native_replay_verified=True,native_Track1Episode_schema_verified=True,original_frame_coverage_verified=True,
            frozen_outputs_rehashed_after_reference=True,raw_masks_contacts_object_pose_unchanged=True,
            full_original_native_export_verified=True,reference_per_frame_mean_mm=[.001]*count,
            native_direct_max_point_mm=.001,native_replay_max_point_mm=0.,reference_max_point_mm=.005,
            reference_settings=dict(precision="float32",residual_dtype="float64",chunk=16,device="cuda",
                mean_point_gate_mm=2.,native_max_point_gate_mm=.01),
            refined_report_sha256="d"*64,refined_bundle_sha256="e"*64,aligned_object_mesh_sha256=payloads["object_aligned.glb"]["sha256"],
            input_pins=dict(sha256="f"*64,bytes=3456),refined_pins=dict(sha256="1"*64,bytes=789),source_files=sources,
            object_roundtrip=dict(frames=count,vertices_per_frame=4000,max_point_error_m=1e-16,
                aligned_local_frame_retained=True,object_scale=1.,additional_frame_transform=False))
        report.update({name+suffix:len(chunks) for name in ("native_geometry","native_direct","native_replay","reference")
            for suffix in ("_attempts","_returns","_validated")})
    elif stage=="prepare":
        report.update(learned_inference_calls=0,optimizer_calls=0,converter_LM_calls=0,submission_eligible=False,
            identity_fixed_before_first_decode=True,chunk_counts=gate._chunks(count,16),predictions_frozen_before_replays=True,
            stored_initializer_native_replay_verified=True,shared_identity_verified=True,original_initializer_unchanged=True,
            frozen_outputs_rehashed_after_reference=True,original_frame_coverage_verified=True,source_geometry_masks_depth_pose_unchanged=True,
            reference_per_frame_mean_mm=[.001]*count,
            shared_initializer_sha256=payloads["shared_initializer.pkl"]["sha256"],direct_parameters_sha256=payloads["direct_parameters.npz"]["sha256"],
            target_sha256=payloads["target.npy"]["sha256"])
        report.update({name+suffix:len(gate._chunks(count,16)) for name in ("native_geometry","native_direct","native_replay","reference")
            for suffix in ("_attempts","_returns","_validated")})
    else:
        name="coconet.pth" if stage=="forward" else "refined.pth"
        report.update(source_frames=count,checkpoint_sha256=gate.CHECKPOINT_SHA,bundle_sha256=payloads[name]["sha256"],
            bundle_bytes=payloads[name]["bytes"],prepare_report_sha256="e"*64)
        if stage=="forward":
            windows=len(gate._window_starts(count));report.pop("input_report_sha256")  # Actual forward's fields, not invented.
            report.update(forward_attempts=1,forward_returns=1,forward_validated=1,captured_windows=windows,
                window_starts=gate._window_starts(count),hub_attempts=2,hub_returns=2,shared_identity_verified=True,
                raw_prediction_bytes_preserved=True,initializer_bytes_preserved=True,native_global_unchanged=True,hub_restored=True,
                stored_bundle_reread_verified=True,actual_caller_indices_verified=True,native_first_occurrence_assembly_verified=True,
                actual_native_decoder_identity_verified=True,refinement_performed=False,exact_deterministic_forward_claim=False)
            report.update({key:windows for key in ("composition_hook_calls","composition_delegate_calls","composition_delegate_returns","composition_verified_calls")})
        else:
            report.update(optimizer_attempts=1,optimizer_returns=1,optimizer_validated=1,requested_steps=300,
                effective_optimizer_updates=301,budget_seconds=7200,history_steps=[0,100,200,300],ground_truth_read=False,
                learned_inference_calls=0,saved_bundle_reloaded_verified=True,frozen_raw_inputs_byte_preserved=True,
                object_mesh_unchanged=True,native_refinement_verified=True,numerical_bit_determinism_claimed=False,
                optimizer_sha256=gate.OPTIMIZER_SHA,forward_report_sha256="f"*64,source_bundle_sha256="0"*64,object_mesh_sha256="1"*64,
                metadata=dict(native_refinement_verified=True,full_original_frame_coverage_verified=True,
                    frozen_parameters_bit_identical=True,requested_steps=300,effective_optimizer_updates=301,batch_size=0))
    write(directory/"report.json",report)
    return root,directory,report


@pytest.mark.parametrize("stage",["prepare","forward","refined","export"])
@pytest.mark.parametrize("episode,count",[(0,96),(15,501),(29,673)])
def test_exact_generic_stage_pins_only_after_immutable_inventory(gate,tmp_path,stage,episode,count):
    root,directory,report=fixture(gate,tmp_path,stage,episode,count)
    before={path:(path.stat().st_mode,path.read_bytes()) for path in directory.iterdir()}
    pins=gate.inventory(root,episode,stage,"a"*40,"b"*64)
    assert set(pins)=={"schema","clip_spec",stage,stage+"_files"}
    assert pins["schema"]==f"world-reward-cari-shared-{stage}-pins-v1"
    assert pins["clip_spec"]==report["clip_spec"]
    assert pins[stage]==gate.identity(directory/"report.json")|dict(producer_revision="a"*40,script_sha256="b"*64)
    assert set(pins[stage+"_files"])=={"report.json",*gate.STAGES[stage]["payloads"]}
    assert before=={path:(path.stat().st_mode,path.read_bytes()) for path in directory.iterdir()}


@pytest.mark.parametrize("stage",["prepare","forward","refined","export"])
@pytest.mark.parametrize("fault",["revision","script","readonly","symlink","extra","missing","subdir","payloadhash","payloadbytes","reportstage",
    "phase","status","episode","frames","frameindex_bool","spec","GT","hand","oracle","nestedGT","private","quality","helper","image"])
def test_bad_or_mutable_producer_never_pinned(gate,tmp_path,stage,fault):
    root,directory,report=fixture(gate,tmp_path,stage);payload=directory/gate.STAGES[stage]["payloads"][0]
    if fault=="revision":report["producer_revision"]="c"*40
    elif fault=="script":report["script_sha256"]="c"*64
    elif fault=="readonly":payload.chmod(0o644)
    elif fault=="symlink":payload.unlink();payload.symlink_to(directory/"report.json")
    elif fault=="extra":write(directory/"old_prediction.pth",b"old unused cache")
    elif fault=="missing":payload.unlink()
    elif fault=="subdir":payload.unlink();payload.mkdir()
    elif fault=="payloadhash":report["output_files"][payload.name]["sha256"]="c"*64
    elif fault=="payloadbytes":report["output_files"][payload.name]["bytes"]=True
    elif fault=="reportstage":report["stage"]="old_partial_pipeline"
    elif fault=="phase":report["phase"]="native_forward"
    elif fault=="status":report["status"]="fail"
    elif fault=="episode":report["episode_index"]=0
    elif fault=="frames":report["frames"]=96
    elif fault=="frameindex_bool":report["original_frame_indices"][0]=False
    elif fault=="spec":report["clip_spec"]["height"]=True
    elif fault=="GT":report["ground_truth_used"]=True
    elif fault=="hand":report["hand_labeled_test"]=True
    elif fault=="oracle":report["oracle_modes"]=["gt"]
    elif fault=="nestedGT":report["nested"]={"ground_truth_read":True}
    elif fault=="private":report["private_truth_read"]=True
    elif fault=="quality":report["quality_verified"]=True
    elif fault=="helper":report["source_helpers"][gate.STAGES[stage]["script"]]["sha256"]="c"*64
    else:report["image_id"]="sha256:"+"c"*64
    if fault not in ("readonly","symlink","extra","missing","subdir"):write(directory/"report.json",report)
    with pytest.raises((ValueError,FileNotFoundError)):gate.inventory(root,15,stage,"a"*40,"b"*64)


@pytest.mark.parametrize("fault",["call","reference_tail","target_sha","coverage"])
def test_prepare_final_chunk_and_saved_routes_required(gate,tmp_path,fault):
    root,directory,report=fixture(gate,tmp_path,"prepare")
    if fault=="call":report["reference_validated"]-=1
    elif fault=="reference_tail":report["reference_per_frame_mean_mm"][-1]=2.00001
    elif fault=="target_sha":report["target_sha256"]="f"*64
    else:report["chunk_counts"][-1]=16
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare","a"*40,"b"*64)


@pytest.mark.parametrize("fault",["windows","delegate","boolcount","reread","raw","bundle","checkpoint"])
def test_forward_actual_fullwindow_receipt_not_checkpoint_load(gate,tmp_path,fault):
    root,directory,report=fixture(gate,tmp_path,"forward")
    if fault=="windows":report["window_starts"]=[0]
    elif fault=="delegate":report["composition_delegate_returns"]-=1
    elif fault=="boolcount":report["forward_returns"]=True
    elif fault=="reread":report["stored_bundle_reread_verified"]=False
    elif fault=="raw":report["raw_prediction_bytes_preserved"]=False
    elif fault=="bundle":report["bundle_sha256"]="f"*64
    else:report["checkpoint_sha256"]="f"*64
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"forward","a"*40,"b"*64)


@pytest.mark.parametrize("fault",["updates","history","metadata","raw","object","saved","optimizer","sourcehash"])
def test_refined_is_exactfinal_stage_not_refine_spelling_or_partial(gate,tmp_path,fault):
    root,directory,report=fixture(gate,tmp_path,"refined")
    if fault=="updates":report["effective_optimizer_updates"]=300
    elif fault=="history":report["history_steps"]=[0,300]
    elif fault=="metadata":report["metadata"]["frozen_parameters_bit_identical"]=False
    elif fault=="raw":report["frozen_raw_inputs_byte_preserved"]=False
    elif fault=="object":report["object_mesh_unchanged"]=False
    elif fault=="saved":report["saved_bundle_reloaded_verified"]=False
    elif fault=="optimizer":report["optimizer_sha256"]="f"*64
    else:report["source_bundle_sha256"]="wrong"
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"refined","a"*40,"b"*64)
    with pytest.raises(ValueError):gate.output_relative(15,"refine")


@pytest.mark.parametrize("stage",["prepare","forward","refined"])
def test_existing_stages_still_require_dataset_revision(gate,tmp_path,stage):
    root,directory,report=fixture(gate,tmp_path,stage);report["input_dataset_revision"]="c"*40
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,stage,"a"*40,"b"*64)


@pytest.mark.parametrize("fault",["tailchunk","missingroute","boolroute","lastreference","shortreference","boolreference",
    "native_direct","native_replay","reference_max","precision","native_gate","reference_gate","refined_report","refined_bundle",
    "input_pin","refined_pin","source_missing","source_extra","source_bytes","mesh_sha","aligned_sha","roundtrip_frame",
    "roundtrip_vertices","roundtrip_error","object_scale","extra_transform","raw_inputs","fixed_predictions","full_export",
    "stored_replay","Parquet","identity_fit","optimizer","LM"])
def test_export_every_original_route_scalar_mesh_and_lineage_required(gate,tmp_path,fault):
    root,directory,report=fixture(gate,tmp_path,"export")
    if fault=="tailchunk":report["chunk_counts"][-1]=16
    elif fault=="missingroute":report["reference_validated"]-=1
    elif fault=="boolroute":report["native_direct_returns"]=True
    elif fault=="lastreference":report["reference_per_frame_mean_mm"][-1]=2.00001
    elif fault=="shortreference":report["reference_per_frame_mean_mm"].pop()
    elif fault=="boolreference":report["reference_per_frame_mean_mm"][0]=False
    elif fault=="native_direct":report["native_direct_max_point_mm"]=.010001
    elif fault=="native_replay":report["native_replay_max_point_mm"]=-.001
    elif fault=="reference_max":report["reference_max_point_mm"]=float("inf")
    elif fault=="precision":report["reference_settings"]["precision"]="float16"
    elif fault=="native_gate":report["reference_settings"]["native_max_point_gate_mm"]=.02
    elif fault=="reference_gate":report["reference_settings"]["mean_point_gate_mm"]=3.
    elif fault=="refined_report":report["refined_report_sha256"]="bad"
    elif fault=="refined_bundle":report["refined_bundle_sha256"]="old_prediction.pth"
    elif fault=="input_pin":report["input_pins"]["bytes"]=True
    elif fault=="refined_pin":report["refined_pins"]["sha256"]="bad"
    elif fault=="source_missing":report["source_files"].pop(next(iter(report["source_files"])))
    elif fault=="source_extra":report["source_files"]["outputs/episode_000015/GT_mesh.ply"]=dict(sha256="0"*64,bytes=123)
    elif fault=="source_bytes":report["source_files"][next(iter(report["source_files"]))]["bytes"]=False
    elif fault=="mesh_sha":
        mesh=next(name for name in report["source_files"] if name.endswith("output_aligned.glb"))
        report["source_files"][mesh]["sha256"]="0"*64
    elif fault=="aligned_sha":report["aligned_object_mesh_sha256"]="0"*64
    elif fault=="roundtrip_frame":report["object_roundtrip"]["frames"]=96
    elif fault=="roundtrip_vertices":report["object_roundtrip"]["vertices_per_frame"]=0
    elif fault=="roundtrip_error":report["object_roundtrip"]["max_point_error_m"]=1.00001e-5
    elif fault=="object_scale":report["object_roundtrip"]["object_scale"]=.9
    elif fault=="extra_transform":report["object_roundtrip"]["additional_frame_transform"]=True
    elif fault=="raw_inputs":report["raw_masks_contacts_object_pose_unchanged"]=False
    elif fault=="fixed_predictions":report["unchanged_refined_predictions_verified"]=False
    elif fault=="full_export":report["full_original_native_export_verified"]=False
    elif fault=="stored_replay":report["stored_native_replay_verified"]=False
    elif fault=="Parquet":report["final_Parquet_produced"]=True
    elif fault=="identity_fit":report["identity_reselection_performed"]=True
    elif fault=="optimizer":report["optimizer_calls"]=1
    else:report["converter_LM_calls"]=1
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"export","a"*40,"b"*64)


def test_export_actual_fields_no_invented_dataset_or_video_and_inclusive_original_gates(gate,tmp_path):
    root,directory,report=fixture(gate,tmp_path,"export")
    assert not {"input_dataset_revision","input_sha256","input_report_sha256","checkpoint_sha256","source_frames"}&set(report)
    report["native_direct_max_point_mm"]=.01;report["native_replay_max_point_mm"]=.01
    report["reference_per_frame_mean_mm"][-1]=2.;report["reference_max_point_mm"]=50.
    report["object_roundtrip"]["max_point_error_m"]=1e-5
    write(directory/"report.json",report);pins=gate.inventory(root,15,"export","a"*40,"b"*64)
    assert len(pins["export_files"])==5 and len(report["reference_per_frame_mean_mm"])==501


@pytest.mark.parametrize("stage",["prepare","forward","refined","export"])
def test_all_stage_payloads_hashed_before_json_and_exactly_rehashed(gate,tmp_path,monkeypatch,stage):
    root,directory,report=fixture(gate,tmp_path,stage);calls=[];actual=gate.identity;parse=gate._strict_json
    names={"report.json",*gate.STAGES[stage]["payloads"]}
    def read(path):calls.append(Path(path).name);return actual(path)
    def interpret(text):
        assert set(calls)==names
        return parse(text)
    monkeypatch.setattr(gate,"identity",read);monkeypatch.setattr(gate,"_strict_json",interpret)
    gate.inventory(root,15,stage,"a"*40,"b"*64)
    assert len(calls)==2*len(names) and all(calls.count(name)==2 for name in names)


def test_export_original_source_path_schema_matches_existing_public_contract(gate,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));import cari_clip_inputs as public
    spec=public.PublicClipSpec(29,673,"front_stereo_camera_left",1152,1536)
    from dataclasses import asdict
    assert gate._public_source_paths(asdict(spec))==public.source_paths(spec)


@pytest.mark.parametrize('profile',['solid','surface'])
def test_solid_public_profile_substitutes_only_one_of_fifteen_paths(gate,profile):
    spec=dict(episode_index=29,total_frames=673,camera_name="front_stereo_camera_left",height=1152,width=1536)
    default=gate._public_source_paths(spec);solid=gate._public_source_paths(spec,profile)
    base="outputs/episode_000029/"
    assert gate._public_source_paths(spec,"default")==default
    assert len(default)==len(solid)==15 and len(default&solid)==14
    assert default-solid=={base+"object_pose_full/report.json"}
    assert solid-default=={base+f"object_pose_full_{profile}/report.json"}


@pytest.mark.parametrize("profile",[None,True,0,[],{},"","volume","conditioned","solid/../default","SOLID"])
def test_public_source_profile_is_a_bounded_enum(gate,profile):
    spec=dict(episode_index=15,total_frames=501,camera_name="front_stereo_camera_left",height=1152,width=1536)
    with pytest.raises(ValueError):gate._public_source_paths(spec,profile)


@pytest.mark.parametrize("episode,count",[(0,96),(15,501),(29,673)])
@pytest.mark.parametrize('profile',['solid','surface'])
def test_solid_export_inventory_uses_exact_fifteen_sources_without_new_claim_fields(gate,tmp_path,episode,count,profile):
    root,directory,report=fixture(gate,tmp_path,"export",episode,count)
    original_fields=set(report);base=f"outputs/episode_{episode:06d}/"
    report["source_files"][base+f"object_pose_full_{profile}/report.json"]=report["source_files"].pop(base+"object_pose_full/report.json")
    assert set(report["source_files"])==gate._public_source_paths(report["clip_spec"],profile)
    write(directory/"report.json",report)
    before={p.name:(p.stat().st_mode,p.read_bytes())for p in directory.iterdir()}
    pins=gate.inventory(root,episode,"export","a"*40,"b"*64)
    assert set(report)==original_fields and len(pins["export_files"])==5
    assert pins["export"]==gate.identity(directory/"report.json")|dict(producer_revision="a"*40,script_sha256="b"*64)
    assert before=={p.name:(p.stat().st_mode,p.read_bytes())for p in directory.iterdir()}
    assert report["quality_verified"] is False and report["adoption_authorized"] is False


@pytest.mark.parametrize("fault",["both","missing","volume","conditioned","foreign_episode","prefix","traversal","extra","bad_identity","aligned_mesh"])
def test_solid_export_never_accepts_mixed_or_arbitrary_source_paths(gate,tmp_path,fault):
    root,directory,report=fixture(gate,tmp_path,"export");base="outputs/episode_000015/"
    old=base+"object_pose_full/report.json";new=base+"object_pose_full_solid/report.json"
    sources=report["source_files"];sources[new]=sources.pop(old)
    if fault=="both":sources[old]=dict(sources[new])
    elif fault=="missing":sources.pop(new)
    elif fault in("volume","conditioned"):
        sources[base+f"object_pose_full_{fault}/report.json"]=sources.pop(new)
    elif fault=="foreign_episode":sources[new.replace("000015","000014")]=sources.pop(new)
    elif fault=="prefix":sources[new+".old"]=sources.pop(new)
    elif fault=="traversal":sources[base+"object_pose_full_solid/../object_pose_full/report.json"]=sources.pop(new)
    elif fault=="extra":sources[base+"object_budget_solid/report.json"]=dict(sources[new])
    elif fault=="bad_identity":sources[new]["bytes"]=True
    else:
        mesh=next(name for name in sources if name.endswith("output_aligned.glb"));sources[mesh]["sha256"]="0"*64
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"export","a"*40,"b"*64)


def test_export_independently_supplied_actual_dispatch_identity_not_inferred(gate,tmp_path):
    root,directory,report=fixture(gate,tmp_path,"export")
    revision="bf54ceb75d3b4240f24639184de47a7ac209f02e"
    digest="967b45eb50959aa3854f01d1baae89fb946679f68bbaa8aff8a2d59287b27ee0"
    report.update(producer_revision=revision,script_sha256=digest)
    report["source_helpers"][gate.STAGES["export"]["script"]]["sha256"]=digest
    write(directory/"report.json",report)
    pins=gate.inventory(root,15,"export",revision,digest)
    assert pins["export"]["producer_revision"]==revision and pins["export"]["script_sha256"]==digest
    with pytest.raises(ValueError):gate.inventory(root,15,"export","a"*40,digest)


def test_export_fresh_stdlib_host_does_not_load_geometry_models_or_arrays(gate,tmp_path):
    root,_,_=fixture(gate,tmp_path,"export")
    program='''import importlib.abc,importlib.util,json,sys
from pathlib import Path
class NoHeavy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','joblib','h5py','torch','cv2','trimesh','PIL','world_reward','cari_full_export'}:
   raise AssertionError('Heavy geometry/model import forbidden: '+fullname)
sys.meta_path.insert(0,NoHeavy())
spec=importlib.util.spec_from_file_location('stage_inventory',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pins=module.inventory(Path(sys.argv[2]),15,'export','a'*40,'b'*64)
assert len(pins['export_files'])==5
assert not {'numpy','joblib','h5py','torch','cv2','trimesh','PIL','world_reward','cari_full_export'}&set(sys.modules)
print(json.dumps(pins,separators=(',',':')))
'''
    result=subprocess.run(["rtk","proxy",sys.executable,"-I","-B","-c",program,
        str(ROOT/"infra/cari_stage_inventory.py"),str(root)],check=True,capture_output=True,text=True)
    assert result.stderr=="" and len(result.stdout)<2048 and len(json.loads(result.stdout)["export_files"])==5


@pytest.mark.parametrize("revision,script",[("main","b"*64),("a"*39,"b"*64),("A"*40,"b"*64),("a"*40,""),("a"*40,"B"*64),(None,"b"*64)])
def test_expected_producer_identity_never_learned_from_report(gate,tmp_path,revision,script):
    root,_,_=fixture(gate,tmp_path,"prepare")
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare",revision,script)


def controls(stage="prepare"):
    return ["--episode","15","--stage",stage,"--producer-revision","a"*40,"--producer-script-sha256","b"*64]


@pytest.mark.parametrize("stage",["prepare","forward","refined","export"])
def test_cli_all_controls_explicit_once(gate,stage):
    args=gate.parser().parse_args(controls(stage))
    assert args.episode==15 and args.stage==stage and args.producer_revision=="a"*40


@pytest.mark.parametrize("args",[[],["--episode","15"],controls()+["--episode","15"],controls()+["--stage","prepare"],
    controls()+["--producer-revision","a"*40],controls()+["--producer-script-sha256","b"*64],controls("refine"),
    controls()+["--root","/tmp"],controls()+["--overwrite"],controls()+["--resume"],controls("exports"),
    ["--ep","15",*controls()[2:]]])
def test_cli_no_defaults_abbreviations_resumes_or_extra_arguments(gate,args):
    with pytest.raises(SystemExit):gate.parser().parse_args(args)


@pytest.mark.parametrize("text",['{"status":"pass","status":"fail"}','{"elapsed":NaN}','{"elapsed":Infinity}','{"elapsed":1e400}'])
def test_strict_json_duplicate_nonfinite_fails_before_pins(gate,tmp_path,text):
    root,directory,_=fixture(gate,tmp_path,"prepare");write(directory/"report.json",text.encode())
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare","a"*40,"b"*64)


def test_all_payloads_hashed_before_json_and_rehashed_after(gate,tmp_path,monkeypatch):
    root,directory,report=fixture(gate,tmp_path,"prepare");calls=[];actual=gate.identity
    def read(path):calls.append(Path(path).name);return actual(path)
    monkeypatch.setattr(gate,"identity",read);parse=gate._strict_json
    def interpret(text):
        assert set(calls)=={"report.json",*gate.STAGES["prepare"]["payloads"]}
        return parse(text)
    monkeypatch.setattr(gate,"_strict_json",interpret)
    gate.inventory(root,15,"prepare","a"*40,"b"*64)
    assert len(calls)==8 and all(calls.count(name)==2 for name in set(calls))


def test_file_change_during_interpretation_rejected(gate,tmp_path,monkeypatch):
    root,directory,_=fixture(gate,tmp_path,"prepare");parse=gate._strict_json
    def interpret(text):write(directory/"target.npy",b"mutated while interpreted");return parse(text)
    monkeypatch.setattr(gate,"_strict_json",interpret)
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare","a"*40,"b"*64)


def test_onebyte_manifest_bool_is_not_integer_size(gate,tmp_path):
    root,directory,report=fixture(gate,tmp_path,"prepare")
    write(directory/"target.npy",b"x");row=gate.identity(directory/"target.npy")
    report["output_files"]["target.npy"]=row|dict(bytes=True);report["target_sha256"]=row["sha256"]
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare","a"*40,"b"*64)


def test_producer_report_itself_must_be_readonly(gate,tmp_path):
    root,directory,_=fixture(gate,tmp_path,"prepare");(directory/"report.json").chmod(0o644)
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare","a"*40,"b"*64)


def test_missing_expected_identity_controls_fail_before_any_file_read(gate,tmp_path,monkeypatch):
    monkeypatch.setattr(gate,"identity",lambda _:pytest.fail("No producer read without independent expected source"))
    with pytest.raises(ValueError):gate.inventory(tmp_path.resolve(),15,"prepare","a"*40,None)


def test_refined_bool_zero_history_is_not_original_iteration(gate,tmp_path):
    root,directory,report=fixture(gate,tmp_path,"refined");report["history_steps"][0]=False
    write(directory/"report.json",report)
    with pytest.raises(ValueError):gate.inventory(root,15,"refined","a"*40,"b"*64)


def test_symlinked_ancestor_and_noncanonical_root_rejected(gate,tmp_path):
    root,_,_=fixture(gate,tmp_path/"actual","prepare");link=tmp_path/"link";link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):gate.inventory(link,15,"prepare","a"*40,"b"*64)
    with pytest.raises(ValueError):gate.inventory(Path("relative"),15,"prepare","a"*40,"b"*64)


def test_main_stdout_only_compact_pins_and_azure_host_boundary(gate,tmp_path,monkeypatch,capsys):
    root,directory,_=fixture(gate,tmp_path,"prepare");monkeypatch.setattr(gate,"ROOT",root)
    monkeypatch.setenv("WR_ROOT",str(root));monkeypatch.setattr(gate.platform,"system",lambda:"Linux")
    gate.main(controls());out=capsys.readouterr();pins=json.loads(out.out)
    assert pins["prepare"]["producer_revision"]=="a"*40 and out.err=="" and len(out.out)<2048
    assert set(path.name for path in directory.iterdir())=={"report.json",*gate.STAGES["prepare"]["payloads"]}
    monkeypatch.setattr(gate.platform,"system",lambda:"Darwin")
    with pytest.raises(RuntimeError):gate.main(controls())


def test_stdlib_import_closure_and_readonly_wrapper():
    source=(ROOT/"infra/cari_stage_inventory.py").read_text();tree=ast.parse(source)
    allowed={"__future__","argparse","hashlib","json","math","os","pathlib","platform","re","stat","sys"}
    imported={node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imported|={alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imported<=allowed
    # Host control Python is 3.8: syntax-only parsing does not detect dict
    # union's runtime version requirement, so prohibit BitOr in this module.
    assert not any(isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitOr) for node in ast.walk(tree))
    ast.parse(source,feature_version=(3,8))
    assert ".chmod(" not in source and ".write_text(" not in source and '.open("w' not in source
    shell=ROOT/"infra/run_cari_stage_inventory.sh";text=shell.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    assert "python3 -I -B" in text and "docker" not in text and "--gpus" not in text and "chmod" not in text

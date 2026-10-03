"""Data-free immutable producer inventory, explicit trust and stdout-only CLI."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

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
    if stage=="prepare":
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


@pytest.mark.parametrize("stage",["prepare","forward","refined"])
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


@pytest.mark.parametrize("stage",["prepare","forward","refined"])
@pytest.mark.parametrize("fault",["revision","script","readonly","symlink","extra","missing","subdir","payloadhash","payloadbytes","reportstage",
    "phase","status","episode","frames","frameindex_bool","spec","GT","hand","oracle","nestedGT","private","quality","helper","dataset","image"])
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
    elif fault=="dataset":report["input_dataset_revision"]="c"*40
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


@pytest.mark.parametrize("revision,script",[("main","b"*64),("a"*39,"b"*64),("A"*40,"b"*64),("a"*40,""),("a"*40,"B"*64),(None,"b"*64)])
def test_expected_producer_identity_never_learned_from_report(gate,tmp_path,revision,script):
    root,_,_=fixture(gate,tmp_path,"prepare")
    with pytest.raises(ValueError):gate.inventory(root,15,"prepare",revision,script)


def controls(stage="prepare"):
    return ["--episode","15","--stage",stage,"--producer-revision","a"*40,"--producer-script-sha256","b"*64]


@pytest.mark.parametrize("stage",["prepare","forward","refined"])
def test_cli_all_controls_explicit_once(gate,stage):
    args=gate.parser().parse_args(controls(stage))
    assert args.episode==15 and args.stage==stage and args.producer_revision=="a"*40


@pytest.mark.parametrize("args",[[],["--episode","15"],controls()+["--episode","15"],controls()+["--stage","prepare"],
    controls()+["--producer-revision","a"*40],controls()+["--producer-script-sha256","b"*64],controls("refine"),
    controls()+["--root","/tmp"],controls()+["--overwrite"],controls()+["--resume"],controls()+["--stage","export"],
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

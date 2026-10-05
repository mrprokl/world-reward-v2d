"""Tiny actual-like frontend reports, opaque payloads and no heavy decoding."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"))
    name="cari_clip_pin_inventory_test";spec=importlib.util.spec_from_file_location(name,ROOT/"infra/cari_clip_pin_inventory.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write(path,value,immutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_text(json.dumps(value)) if isinstance(value,dict) else path.write_bytes(value)
    if immutable:path.chmod(0o444)


def actual_fixture(gate,tmp_path,episode=15,count=501):
    root=(tmp_path/"runtime").resolve();code=(tmp_path/"code").resolve();code.mkdir(parents=True)
    spec=gate.inputs.PublicClipSpec(episode,count,"front_stereo_camera_left",1152,1536)
    paths,deps=gate.inputs.relative_paths(spec),gate.inputs.dependency_paths(spec)
    source=code/gate.PRODUCER_SCRIPT;write(source,b"actual independently supplied frontend source",True)
    for name in ("infra/cari_clip_pin_inventory.py","infra/run_cari_clip_pin_inventory.sh","infra/cari_clip_inputs.py","infra/cari96_inputs.py"):
        write(code/name,b"actual immutable report-only helper "+name.encode(),True)
    for name in gate.inputs.source_paths(spec):write(root/name,b"opaque public payload not decoded")
    flags=dict(status="pass",input_track="track_1",input_sha256="b"*64,episode_index=episode,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    records={}
    for role in ("body","depth","alignment","object","adapter"):
        row=dict(flags,stage=gate.inputs.DEPENDENCY_STAGES[role])
        if role in ("body","depth"):
            row.update(input_dataset_revision=gate.inputs.DATASET_REVISION,total_video_frames=count)
        if role in ("body","depth","object"):row["frames"]=[dict(frame_index=i) for i in range(count)]
        if role=="alignment":row["depth_alignment"]={"shared_scale":.83}
        if role=="object":row.update(geometry_and_poses_sha256="a"*64,
            full_depth_report_sha256=gate.identity(root/deps["depth"])["sha256"],
            alignment_report_sha256=gate.identity(root/deps["alignment"])["sha256"])
        if role=="adapter":
            row.pop("input_sha256")  # Actual current adapter does not emit this optional field.
            row.update(frames=count,canonical_initializer_sha256=gate.identity(root/paths["initializer"])["sha256"],
                body_report_sha256=gate.identity(root/deps["body"])["sha256"],decoder_identity={"native":"actual"},inference_source_identity={"source":"actual"})
        write(root/deps[role],row);records[role]=row
    report=dict(flags,stage=gate.STAGE,frames=count,input_dataset_revision=gate.inputs.DATASET_REVISION,
        original_frame_coverage_verified=True,producer_revision="c"*40,script_sha256=gate.identity(source)["sha256"],
        object_pose_initializer="own_ICP_Viterbi_not_FoundationPose",export_seq=str(root/paths["export_seq"]),
        depth_h5=str(root/paths["depth_h5"]),mhr_init=str(root/paths["initializer"]),object_poses=str(root/paths["object_poses"]),
        submission_eligible=False,challenge_performance_verified=False,mesh_pose_frame_roundtrip_max_error_m=9.02e-16,
        jpeg_original_RGB_mean_absolute_error=.05,
        depth_validation=dict(validation_mode="exhaustive",frame_counts={spec.camera_name:count},frame_shapes={spec.camera_name:[spec.height,spec.width]}),
        input_report_sha256={role:gate.identity(root/name)["sha256"] for role,name in deps.items()},
        file_sha256={name:gate.identity(root/paths[key])["sha256"] for name,key in
            (("depth_h5","depth_h5"),("mhr_init","initializer"),("object_poses","object_poses"),("wild_export","wild_export"))})
    write(root/paths["input_report"],report);records["inputs"]=report
    return root,code,spec,records


def repin_report(gate,root,spec,records):
    paths,deps=gate.inputs.relative_paths(spec),gate.inputs.dependency_paths(spec)
    for role,name in deps.items():write(root/name,records[role])
    report=records["inputs"]
    report["input_report_sha256"]={role:gate.identity(root/name)["sha256"] for role,name in deps.items()}
    write(root/paths["input_report"],report)


def test_inventory_solid_profile_is_explicit_not_self_selected(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path,9,97)
    paths=gate.inputs.relative_paths(spec);old=gate.inputs.dependency_paths(spec)
    deps=gate.inputs.dependency_paths(spec,object_source="solid")
    destination=root/deps["object"];destination.parent.mkdir();(root/old["object"]).rename(destination)
    records["object"]["mesh_source"]="solid";write(destination,records["object"])
    report=records["inputs"]
    report.update(object_source="solid",object_pose_source=dict(report=deps["object"],
        geometry_and_poses=str(destination.with_name("geometry_and_poses.npz").relative_to(root)),geometry_and_poses_sha256="a"*64),
        input_report_sha256={role:gate.identity(root/name)["sha256"] for role,name in deps.items()})
    write(root/paths["input_report"],report)
    source=report["script_sha256"]
    monkeypatch.setattr(gate.inputs,"verify_public_inputs",lambda *_:pytest.fail("Inventory remains report-only"))
    pins=gate.inventory(root,code,spec,"c"*40,source,object_source="solid")
    assert set(pins)=={"schema","clip_spec","input_report","source_files","object_source"}
    assert pins["schema"]=="world-reward-cari-clip-input-pins-v2" and pins["object_source"]=="solid"
    assert len(pins["source_files"])==15 and set(pins["source_files"])==gate.inputs.source_paths(spec,object_source="solid")
    assert old["object"] not in pins["source_files"]
    # No inference of solid from a self-authored public preparation receipt.
    with pytest.raises((ValueError,FileNotFoundError)):gate.inventory(root,code,spec,"c"*40,source)


def test_inventory_surface_v3_is_report_only_and_requires_native_binding(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path,29,97)
    paths=gate.inputs.relative_paths(spec);old=gate.inputs.dependency_paths(spec)
    deps=gate.inputs.dependency_paths(spec,object_source='surface');destination=root/deps['object']
    destination.parent.mkdir();(root/old['object']).rename(destination)
    native=dict(committed_pins_sha256='1'*64,producer_report_sha256='2'*64,cpu_native_report_sha256='3'*64,
        source_domain='surface',metric_scale_baked_once=.25,geometry_operations_applied=False)
    records['object'].update(mesh_source='surface',topology_budget=native);write(destination,records['object'])
    proof=dict(native,metric_glb={'synthetic':'metric'},native_aligned_glb={'synthetic':'aligned'},
        source_rehashed_after=True,files={'synthetic_only':{'sha256':'4'*64,'bytes':4}})
    report=records['inputs'];report.update(object_source='surface',surface_geometry_validation=proof,
        object_pose_source=dict(report=deps['object'],geometry_and_poses=str(destination.with_name('geometry_and_poses.npz').relative_to(root)),geometry_and_poses_sha256='a'*64),
        input_report_sha256={role:gate.identity(root/name)['sha256'] for role,name in deps.items()})
    write(root/paths['input_report'],report)
    monkeypatch.setattr(gate.inputs,'verify_public_inputs',lambda *_:pytest.fail('No payloads decoded by inventory'))
    pins=gate.inventory(root,code,spec,'c'*40,report['script_sha256'],object_source='surface')
    assert pins['schema']=='world-reward-cari-clip-input-pins-v3' and pins['object_source']=='surface'
    assert len(pins['source_files'])==15 and old['object'] not in pins['source_files']
    with pytest.raises((ValueError,FileNotFoundError)):gate.inventory(root,code,spec,'c'*40,report['script_sha256'])
    report['surface_geometry_validation']['source_rehashed_after']=False;write(root/paths['input_report'],report)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,'c'*40,report['script_sha256'],object_source='surface')


def test_optional_source_cli_defaults_legacy_and_rejects_duplicates_or_paths(gate):
    assert gate.parser().parse_args(controls()).object_source is None
    assert gate.parser().parse_args(controls()+["--object-source","solid"]).object_source=="solid"
    assert gate.parser().parse_args(controls()+["--object-source","surface"]).object_source=="surface"
    for suffix in (["--object-source","../solid"],["--object-source","solid","--object-source","solid"]):
        with pytest.raises(SystemExit):gate.parser().parse_args(controls()+suffix)


@pytest.mark.parametrize("episode,count",[(0,96),(15,501),(29,673)])
def test_all15_generic_frontend_pins_without_array_decode_or_mutation(gate,tmp_path,episode,count):
    root,code,spec,records=actual_fixture(gate,tmp_path,episode,count)
    names=gate.inputs.source_paths(spec);before={name:(root/name).read_bytes() for name in names}
    source=records["inputs"]["script_sha256"]
    pins=gate.inventory(root,code,spec,"c"*40,source)
    assert pins["schema"]=="world-reward-cari-clip-input-pins-v1" and len(pins["source_files"])==15
    assert pins["clip_spec"]==dict(episode_index=episode,total_frames=count,camera_name=spec.camera_name,height=1152,width=1536)
    assert pins["input_report"]==dict(gate.identity(root/gate.inputs.relative_paths(spec)["input_report"]),producer_revision="c"*40,script_sha256=source)
    assert before=={name:(root/name).read_bytes() for name in names}
    assert "numpy" not in gate.__dict__ and "joblib" not in gate.__dict__


@pytest.mark.parametrize("role",["inputs","body","depth","object","alignment","adapter"])
@pytest.mark.parametrize("fault",["episode","absentepisode","GT","hand","oracle","private","status","stage"])
def test_current_report_identity_fail_closed_not_legacy(gate,tmp_path,role,fault):
    root,code,spec,records=actual_fixture(gate,tmp_path)
    row=records[role];source=records["inputs"]["script_sha256"]
    if fault=="episode":row["episode_index"]=True
    elif fault=="absentepisode":row.pop("episode_index")
    elif fault=="GT":row["ground_truth_used"]=True
    elif fault=="hand":row["hand_labeled_test"]=True
    elif fault=="oracle":row["oracle_modes"]=["gt"]
    elif fault=="private":row["private_truth_read"]=True
    elif fault=="status":row["status"]="fail"
    else:row["stage"]="old-prediction-producer"
    repin_report(gate,root,spec,records)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source)


@pytest.mark.parametrize("role",["inputs","body","depth"])
@pytest.mark.parametrize("fault",["absent","wrong"])
def test_newproducer_dataset_revision_cannot_use_legacy_defaults(gate,tmp_path,role,fault):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    if fault=="absent":records[role].pop("input_dataset_revision")
    else:records[role]["input_dataset_revision"]="f"*40
    repin_report(gate,root,spec,records)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source)


@pytest.mark.parametrize("fault",["expectedrev","expectedsha","receiptsha","sourcecode","sourcewritable","depth","body","object",
    "adapter","path","coverage","roundtrip","jpeg","grid","scale","geometryhash","video","identityhash"])
def test_sources_scales_camera_and_original_artifact_chain_bound(gate,tmp_path,fault):
    root,code,spec,records=actual_fixture(gate,tmp_path);report=records["inputs"];source=report["script_sha256"];revision="c"*40
    if fault=="expectedrev":revision="d"*40
    elif fault=="expectedsha":source="d"*64
    elif fault=="receiptsha":report["script_sha256"]="d"*64
    elif fault=="sourcecode":write(code/gate.PRODUCER_SCRIPT,b"changed producer code",True)
    elif fault=="sourcewritable":(code/gate.PRODUCER_SCRIPT).chmod(0o644)
    elif fault=="depth":records["object"]["full_depth_report_sha256"]="f"*64
    elif fault=="body":records["adapter"]["body_report_sha256"]="f"*64
    elif fault=="object":records["object"]["alignment_report_sha256"]="f"*64
    elif fault=="adapter":records["adapter"]["canonical_initializer_sha256"]="f"*64
    elif fault=="path":report["mhr_init"]="/old/cari_forward/prediction.pth"
    elif fault=="coverage":report["frames"]=96
    elif fault=="roundtrip":report["mesh_pose_frame_roundtrip_max_error_m"]=1e-4
    elif fault=="jpeg":report["jpeg_original_RGB_mean_absolute_error"]=float("nan")
    elif fault=="grid":report["depth_validation"]["frame_shapes"][spec.camera_name]=[2,4]
    elif fault=="scale":records["alignment"]["depth_alignment"]["shared_scale"]=0
    elif fault=="geometryhash":records["object"]["geometry_and_poses_sha256"]="wrong"
    elif fault=="video":records["body"]["input_sha256"]="f"*64
    else:report["file_sha256"]["mhr_init"]="f"*64
    repin_report(gate,root,spec,records)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,revision,source)


@pytest.mark.parametrize("role",["body","depth","object"])
@pytest.mark.parametrize("fault",["missing","duplicate","bool","reorder"])
def test_complete_original_native_frame_records(gate,tmp_path,role,fault):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    rows=records[role]["frames"]
    if fault=="missing":rows.pop()
    elif fault=="duplicate":rows[-1]["frame_index"]=0
    elif fault=="bool":rows[0]["frame_index"]=False
    else:rows.reverse()
    repin_report(gate,root,spec,records)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source)


@pytest.mark.parametrize("fault",["missing","symlink","directory","zero","parent"])
def test_exact15_regular_files_hashed_before_json(gate,tmp_path,monkeypatch,fault):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    path=root/gate.inputs.relative_paths(spec)["depth_h5"]
    if fault=="parent":
        parent=path.parent;new=parent.with_name("original");parent.rename(new);parent.symlink_to(new,target_is_directory=True)
    else:
        path.unlink()
        if fault=="symlink":path.symlink_to(root/gate.inputs.relative_paths(spec)["input_report"])
        elif fault=="directory":path.mkdir()
        elif fault=="zero":path.touch()
    monkeypatch.setattr(gate,"_strict_json",lambda _:pytest.fail("No JSON before all fifteen regular source hashes"))
    with pytest.raises((ValueError,FileNotFoundError)):gate.inventory(root,code,spec,"c"*40,source)


def test_all15_hash_before_first_json_and_twice_afterward(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    names=gate.inputs.source_paths(spec);seen=[];original=gate.identity;parse=gate._strict_json
    def identity(path,*args,**kwargs):
        path=Path(path)
        if path.is_relative_to(root):seen.append(str(path.relative_to(root)))
        return original(path,*args,**kwargs)
    def interpret(text):
        assert set(seen)==names
        return parse(text)
    monkeypatch.setattr(gate,"identity",identity);monkeypatch.setattr(gate,"_strict_json",interpret)
    gate.inventory(root,code,spec,"c"*40,source)
    assert len(seen)==30 and all(seen.count(name)==2 for name in names)


def test_existing_report_gate_invoked_only_not_heavy_public_validator(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"];seen=[]
    original=gate.inputs.validate_reports
    monkeypatch.setattr(gate.inputs,"verify_public_inputs",lambda *_:pytest.fail("No Joblib/NumPy/H5 decode in source inventory"))
    def validate(*args):seen.append(True);return original(*args)
    monkeypatch.setattr(gate.inputs,"validate_reports",validate)
    gate.inventory(root,code,spec,"c"*40,source)
    assert seen==[True]


def test_sources_changed_during_report_audit_rejected(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"];original=gate.inputs.validate_reports
    def validate(*args):
        result=original(*args);write(root/gate.inputs.relative_paths(spec)["mesh"],b"changed original aligned mesh")
        return result
    monkeypatch.setattr(gate.inputs,"validate_reports",validate)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source)


@pytest.mark.parametrize("text",['{"stage":"x","stage":"y"}','{"elapsed":NaN}','{"elapsed":1e400}'])
def test_strict_json_before_tolerant_legacy_report_reads(gate,tmp_path,text):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    write(root/gate.inputs.relative_paths(spec)["input_report"],text.encode())
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source)


def test_legacy_identity_never_reissued_as_current_frontend(gate,tmp_path,monkeypatch):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    current=dict(gate.identity(root/gate.inputs.relative_paths(spec)["input_report"]),producer_revision="c"*40,script_sha256=source)
    monkeypatch.setattr(gate.inputs,"LEGACY_INPUT_REPORT",current)
    with pytest.raises(ValueError,match="Legacy"):gate.inventory(root,code,spec,"c"*40,source)


def controls():
    return ["--episode","15","--frames","501","--height","1152","--width","1536","--camera-name","front_stereo_camera_left",
        "--producer-revision","c"*40,"--producer-script-sha256","d"*64]


@pytest.mark.parametrize("flag",["--episode","--frames","--height","--width","--camera-name","--producer-revision","--producer-script-sha256"])
def test_cli_every_structural_and_source_control_required_once(gate,flag):
    args=controls();index=args.index(flag)
    with pytest.raises(SystemExit):gate.parser().parse_args(args[:index]+args[index+2:])
    with pytest.raises(SystemExit):gate.parser().parse_args(args+[flag,args[index+1]])


@pytest.mark.parametrize("extra",[["--resume"],["--overwrite"],["--root","/tmp"],["--stage","prepare"],["--ep","15"]])
def test_cli_no_extra_defaults_or_inference_controls(gate,extra):
    with pytest.raises(SystemExit):gate.parser().parse_args(controls()+extra)


def test_main_compact_stdout_only_and_newer_inventory_revision_allowed(gate,tmp_path,monkeypatch,capsys):
    root,code,spec,records=actual_fixture(gate,tmp_path);source=records["inputs"]["script_sha256"]
    monkeypatch.setattr(gate,"ROOT",root);monkeypatch.setenv("WR_ROOT",str(root));monkeypatch.setenv("WR_CODE",str(code))
    monkeypatch.setenv("WR_CODE_REVISION","f"*40);monkeypatch.setattr(gate.platform,"system",lambda:"Linux")
    args=controls();args[-1]=source;gate.main(args);out=capsys.readouterr();pins=json.loads(out.out)
    assert out.err=="" and len(out.out)<4096 and pins["input_report"]["producer_revision"]=="c"*40
    monkeypatch.setattr(gate.platform,"system",lambda:"Darwin")
    with pytest.raises(RuntimeError):gate.main(args)


def test_stdlib_report_only_imports_host_compatibility_and_wrapper():
    source=(ROOT/"infra/cari_clip_pin_inventory.py").read_text();tree=ast.parse(source,feature_version=(3,8))
    allowed={"__future__","argparse","dataclasses","hashlib","json","math","os","pathlib","platform","re","stat","sys","cari_clip_inputs"}
    imported={node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imported|={alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imported<=allowed and "verify_public_inputs(" not in source and "numpy" not in imported and "joblib" not in imported
    assert not any(isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitOr) for node in ast.walk(tree))
    assert ".chmod(" not in source and ".write_text(" not in source and '.open("w' not in source
    shell=ROOT/"infra/run_cari_clip_pin_inventory.sh";text=shell.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(shell)],check=True)
    assert "python3 -I -B" in text and "sys.path.insert(0,str(code/'infra'))" in text
    assert "docker" not in text and "--gpus" not in text and "chmod" not in text


def test_actual_immutable_runtime_archive_includes_explicit_entrypoint_and_helper_closure(gate):
    import azure_job
    files={str(path.relative_to(ROOT)):path.read_bytes() for name in ("infra","src","configs")
        for path in (ROOT/name).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    selected=azure_job.runtime_bundle_paths(files,"infra/run_cari_clip_pin_inventory.sh")
    assert {"infra/cari_clip_pin_inventory.py","infra/run_cari_clip_pin_inventory.sh",
        "infra/cari_clip_inputs.py","infra/cari96_inputs.py","infra/cari_prepare.py",
        "infra/mesh_conditioned_chart_v2.hpp","infra/Dockerfile.volume_qem"}<=set(selected)


def separate_producer_fixture(gate,tmp_path):
    root,code,spec,records=actual_fixture(gate,tmp_path,episode=0,count=790)
    producer=(tmp_path/"original_producer").resolve()
    write(producer/gate.PRODUCER_SCRIPT,(code/gate.PRODUCER_SCRIPT).read_bytes(),True)
    (producer/"infra").chmod(0o555);producer.chmod(0o555)
    write(code/gate.PRODUCER_SCRIPT,b"new consumer source is not the historical producer",True)
    return root,code,producer,spec,records


def test_cli_explicit_original_producer_path_is_optional_and_single_occurrence(gate):
    args = ['--episode', '0', '--frames', '790', '--height', '1152', '--width', '1536',
            '--camera-name', 'front_stereo_camera_left', '--producer-revision', 'c' * 40,
            '--producer-script-sha256', 'd' * 64]
    assert gate.parser().parse_args(args).producer_code is None
    assert gate.parser().parse_args(args + ['--producer-code', '/original']).producer_code == '/original'
    with pytest.raises(SystemExit):
        gate.parser().parse_args(args + ['--producer-code', '/original', '--producer-code', '/other'])


def test_separate_original_producer_keeps_consumer_helpers_and_every_input_unchanged(gate,tmp_path):
    root,code,producer,spec,records=separate_producer_fixture(gate,tmp_path)
    source=records["inputs"]["script_sha256"]
    before={name:(root/name).read_bytes() for name in gate.inputs.source_paths(spec)}
    with pytest.raises(ValueError,match="Original frontend source"):
        gate.inventory(root,code,spec,"c"*40,source)
    pins=gate.inventory(root,code,spec,"c"*40,source,producer_code=producer)
    assert pins["input_report"]["script_sha256"]==source and pins["input_report"]["producer_revision"]=="c"*40
    assert before=={name:(root/name).read_bytes() for name in before}
    assert gate.identity(code/gate.PRODUCER_SCRIPT)["sha256"]!=source


@pytest.mark.parametrize("fault",["writable","symlink","missing","wrong_source","writable_source","consumer_helper"])
def test_separate_producer_source_and_consumer_helper_guards(gate,tmp_path,fault):
    root,code,producer,spec,records=separate_producer_fixture(gate,tmp_path)
    source=records["inputs"]["script_sha256"]
    if fault=="writable":producer.chmod(0o755)
    elif fault=="symlink":
        alias=tmp_path/"alias";alias.symlink_to(producer,target_is_directory=True);producer=alias
    elif fault=="missing":producer=tmp_path/"missing"
    elif fault=="wrong_source":write(producer/gate.PRODUCER_SCRIPT,b"tampered original source",True)
    elif fault=="writable_source":(producer/gate.PRODUCER_SCRIPT).chmod(0o644)
    else:(code/"infra/cari_clip_inputs.py").chmod(0o644)
    with pytest.raises((ValueError,FileNotFoundError)):
        gate.inventory(root,code,spec,"c"*40,source,producer_code=producer)


@pytest.mark.parametrize("target",["producer_source","producer_directory","consumer_helper","public_file"])
def test_separate_original_source_and_public_closure_rehashed_after_audit(gate,tmp_path,monkeypatch,target):
    root,code,producer,spec,records=separate_producer_fixture(gate,tmp_path)
    source=records["inputs"]["script_sha256"];original=gate.inputs.validate_reports
    def validate(*args):
        result=original(*args)
        if target=="producer_source":write(producer/gate.PRODUCER_SCRIPT,b"changed original source",True)
        elif target=="producer_directory":producer.chmod(0o755)
        elif target=="consumer_helper":write(code/"infra/cari_clip_inputs.py",b"changed consumer helper",True)
        else:write(root/gate.inputs.relative_paths(spec)["mesh"],b"changed public geometry")
        return result
    monkeypatch.setattr(gate.inputs,"validate_reports",validate)
    with pytest.raises(ValueError):gate.inventory(root,code,spec,"c"*40,source,producer_code=producer)


def test_historical_input_omission_inventory_does_not_rewrite_or_bypass_dependencies(gate,tmp_path,monkeypatch):
    root,code,producer,spec,records=separate_producer_fixture(gate,tmp_path)
    records["inputs"].pop("input_dataset_revision");repin_report(gate,root,spec,records)
    source=records["inputs"]["script_sha256"]
    # Bind procedural receipt bytes; production constant is separately checked.
    receipt=dict(gate.identity(root/gate.inputs.relative_paths(spec)["input_report"]),
        producer_revision="c"*40,script_sha256=source)
    monkeypatch.setattr(gate.inputs,"HISTORICAL_INPUT_DATASET_OMISSION",receipt)
    before={name:(root/name).read_bytes() for name in gate.inputs.source_paths(spec)}
    pins=gate.inventory(root,code,spec,"c"*40,source,producer_code=producer)
    assert pins["input_report"]==receipt and not gate.inputs._legacy(spec,pins)
    assert before=={name:(root/name).read_bytes() for name in before}
    assert "input_dataset_revision" not in json.loads((root/gate.inputs.relative_paths(spec)["input_report"]).read_text())
    for role in ("body","depth"):
        bad=copy.deepcopy(records);bad[role].pop("input_dataset_revision")
        with pytest.raises(ValueError):gate.validate_current_reports(root,spec,pins,bad)
    bad=copy.deepcopy(records);bad["inputs"]["input_dataset_revision"]="f"*40
    with pytest.raises(ValueError):gate.validate_current_reports(root,spec,pins,bad)


def test_fresh_host_process_runs_full_inventory_without_heavy_imports(gate,tmp_path):
    root,code,spec,records=actual_fixture(gate,tmp_path,0,96)
    program='''import importlib.abc,json,sys
from pathlib import Path
class NoHeavy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','joblib','h5py','torch','cv2','trimesh','PIL','world_reward'}:
   raise AssertionError('Heavy/payload import forbidden: '+fullname)
sys.meta_path.insert(0,NoHeavy())
sys.path.insert(0,sys.argv[1])
import cari_clip_pin_inventory as gate
spec=gate.inputs.PublicClipSpec(0,96,'front_stereo_camera_left',1152,1536)
result=gate.inventory(Path(sys.argv[2]),Path(sys.argv[3]),spec,'c'*40,sys.argv[4])
assert len(result['source_files'])==15
assert not {'numpy','joblib','h5py','torch','cv2','trimesh','PIL','world_reward'}&set(sys.modules)
print(json.dumps(result,separators=(',',':')))
'''
    result=subprocess.run(["rtk","proxy",sys.executable,"-I","-B","-c",program,str(ROOT/"infra"),
        str(root),str(code),records["inputs"]["script_sha256"]],check=True,capture_output=True,text=True)
    assert result.stderr=="" and len(result.stdout)<4096 and len(json.loads(result.stdout)["source_files"])==15

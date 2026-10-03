"""H99 tiny public-contract tests, not evidence of real native inference accuracy."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"infra"));monkeypatch.syspath_prepend(str(REPO/"src"))
    spec=importlib.util.spec_from_file_location("factorial_observe_test",REPO/"infra/factorial_rgb_observe.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def pins(gate,stage="masks"):
    value=dict(schema=gate.PIN_SCHEMA,manifest_sha256="a"*64,manifest_bytes=100)
    if stage!="masks":value["automatic_masks"]=dict(producer_revision="b"*40,report_sha256="c"*64,report_bytes=100,script_sha256="d"*64)
    return value


def test_exact_opaque_public_cohort_and_no_old_stage_reader(gate):
    assert gate.cohort_contract(gate.COHORT)==asdict(gate.COHORT)
    assert gate.COHORT.count==24 and gate.COHORT.frame_name(23)=="group_07_frame_002.png"
    source=Path(gate.__file__).read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Import)and any(a.name in("torch","moge","root5_rgb_observe","factorial_rgb_render")for a in n.names)for n in tree.body)
    assert not any(isinstance(n,ast.Attribute)and n.attr in("shared_decode","raster_camera_mesh","run_baseline","fit_shared_depth_scale")for n in ast.walk(tree))
    assert "eval_private"not in source and "root5_rgb_v1"not in source


@pytest.mark.parametrize("fault",["extra","missing","schema","sha","bytes","bool","revision","mask_source","mask_size"])
def test_pins_have_no_dynamic_discovery_or_unknown_fields(gate,fault):
    value=pins(gate,"body")
    if fault=="extra":value["camera_truth"]=[]
    elif fault=="missing":value.pop("automatic_masks")
    elif fault=="schema":value["schema"]="wrong"
    elif fault=="sha":value["manifest_sha256"]="A"*64
    elif fault=="bytes":value["manifest_bytes"]=0
    elif fault=="bool":value["automatic_masks"]["report_bytes"]=True
    elif fault=="revision":value["automatic_masks"]["producer_revision"]="main"
    elif fault=="mask_source":value["automatic_masks"]["script_sha256"]="D"*64
    else:value["automatic_masks"]["report_bytes"]=-1
    with pytest.raises(ValueError):gate.validate_pins(value,"body")
    with pytest.raises(ValueError):gate.validate_pins(pins(gate,"body"),"masks")


def body_data(gate,monkeypatch,index=0):
    monkeypatch.setattr(gate.native,"VERTICES",100)
    data={k:np.zeros(s,np.float32)for k,s in gate.native.BLOCKS.items()}
    data["pred_cam_t"][2]=2.
    data.update(vertices_camera_m=np.tile([0.,0.,2.],(100,1)).astype(np.float32),keypoints_camera_m=np.tile([0.,0.,2.],(308,1)).astype(np.float32),
        joints_camera_m=np.tile([0.,0.,2.],(127,1)).astype(np.float32),joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)),
        human_faces=np.tile([0,1,2],(36874,1)).astype(np.int64),camera_K=gate.native.CAMERA_K.copy(),bbox=np.array([[1.,1.,10.,7.]],np.float32),
        group_index=np.array(index//3,np.int64),frame_index=np.array(index%3,np.int64))
    record=dict(file=gate.COHORT.frame_name(index),sha256=f"{index:064x}",group_index=index//3,frame_index=index%3)
    return data,record


@pytest.mark.parametrize("fault",["none","depth","identity_field","group","root","root_rotation","expression","Z","topology","rotation","K","bbox","dtype","masked"])
def test_compact_body_schema_is_full308_not_depth_or_shared_identity(gate,monkeypatch,fault):
    data,record=body_data(gate,monkeypatch)
    if fault=="none":assert gate.validate_body(data,record)is data;return
    if fault=="depth":data["raw_depth"]=np.ones((8,16),np.float32)
    elif fault=="identity_field":data["shared_shape_params"]=data["shape_params"].copy()
    elif fault=="group":data["group_index"]=np.array(1,np.int64)
    elif fault=="root":data["mhr_model_params"][0]=.1
    elif fault=="root_rotation":data["mhr_model_params"][3]=.1
    elif fault=="expression":data["expr_params"][0]=.1
    elif fault=="Z":data["keypoints_camera_m"][300,2]=0
    elif fault=="topology":data["human_faces"][0,0]=100
    elif fault=="rotation":data["joint_global_rotations"][0,0,0]=-1
    elif fault=="K":data["camera_K"][0,0]=1279
    elif fault=="bbox":data["bbox"][0,2]=2048
    elif fault=="dtype":data["keypoints_camera_m"]=data["keypoints_camera_m"].astype(np.float64)
    else:data["shape_params"]=np.ma.array(data["shape_params"],mask=False)
    with pytest.raises(ValueError):gate.validate_body(data,record)


def dw_data(gate,index=0):
    scores=np.ones((1,133),np.float32);scores[0,0]=-2.;points=np.zeros((1,133,2),np.float64)
    return dict(keypoints=points,scores=scores,validity=scores>0,bbox=np.array([[1.,1.,10.,7.]],np.float32),
        group_index=np.array(index//3,np.int64),frame_index=np.array(index%3,np.int64))


def test_dw_preserves133_original_float64_coordinates_raw_score_validity(gate):
    data=dw_data(gate);record=dict(group_index=0,frame_index=0)
    assert gate.validate_dw(data,record)is data and data["scores"][0,0]==-2.
    data["keypoints"]=data["keypoints"].astype(np.float32)
    with pytest.raises(ValueError):gate.validate_dw(data,record)
    data=dw_data(gate);data["validity"][0,0]=True
    with pytest.raises(ValueError):gate.validate_dw(data,record)


@pytest.mark.parametrize("kind",["body","dwpose"])
def test_all24_real_npz_frozen_roundtrip_no_drop_or_refit(gate,monkeypatch,tmp_path,kind):
    out=tmp_path/kind;out.mkdir();(out/"report.json").write_text("{}");records=[];rows=[]
    for i in range(24):
        original,record=body_data(gate,monkeypatch,i);data=original if kind=="body"else dw_data(gate,i)
        if kind=="body":data["shape_params"][0]=np.float32(.01*i) # Framewise identity is permitted for diagnostics.
        records.append(record);rows.append(gate.save(out,record,data))
    validator=gate.validate_body if kind=="body"else gate.validate_dw
    decoded=gate.frozen_artifacts(out,rows,records,validator)
    assert len(decoded)==24 and decoded[23]["group_index"].item()==7
    assert all(not Path(row["prediction"]["path"]).stat().st_mode&0o222 for row in rows)
    rows[13]["rgb_sha256"]="f"*64
    with pytest.raises(ValueError):gate.frozen_artifacts(out,rows,records,validator)


def small_public_fixture(gate,monkeypatch,tmp_path):
    """Real RGB/mask/receipt bytes; only grid dimensions are reduced for fixture speed."""
    from PIL import Image
    monkeypatch.setattr(gate.protocol.PublicCohort,"__post_init__",lambda _:None)
    cohort=gate.protocol.PublicCohort(width=16,height=8)
    base=tmp_path/cohort.base;(base/"inputs").mkdir(parents=True);(base/"automatic_masks").mkdir()
    records=[];digests=[];rows=[]
    for i in range(24):
        filename=cohort.frame_name(i);rgb=np.full((8,16,3),i,np.uint8);target=base/"inputs"/filename
        Image.fromarray(rgb).save(target);target.chmod(0o444);digest=gate.protocol.identity(target)["sha256"];digests.append(digest)
        row=dict(file=filename,group_index=i//3,frame_index=i%3,rgb_sha256=digest)
        for label,query in gate.QUERIES:
            mask=np.full((8,16),255,np.uint8);path=base/"automatic_masks"/(Path(filename).stem+"_"+label+".png")
            Image.fromarray(mask).save(path);path.chmod(0o444);identity=gate.protocol.identity(path)
            row.update({label+"_query":query,label+"_mask_file":path.name,label+"_mask_sha256":identity["sha256"],label+"_mask_bytes":identity["bytes"],label+"_mask_pixels":128})
        rows.append(row)
    manifest=base/"inputs/manifest.json";manifest.write_text(json.dumps(gate.protocol.public_manifest(digests,cohort)));manifest.chmod(0o444)
    mi=gate.protocol.identity(manifest);source=tmp_path/"jobs"/("b"*40)/"run_factorial_rgb_observe/code/infra/factorial_rgb_observe.py"
    source.parent.mkdir(parents=True);source.write_text("# frozen fixture mask producer\n");source.chmod(0o444);si=gate.protocol.identity(source)
    assets={name:dict(path=str(tmp_path/"weights"/name),sha256=digest,bytes=size)for name,(digest,size)in gate.masks.ASSETS.items()}
    report=dict(stage=gate.STAGES["masks"],status="pass",phase="complete",cohort=asdict(cohort),frames=24,script_sha256=si["sha256"],
        source_helpers=gate.source_identity()|dict(observer=si["sha256"]),producer_revision="b"*40,network="none",private_truth_read=False,ground_truth_used=False,
        challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],human_query="person.",object_query="bottle.",input_manifest_sha256=mi["sha256"],input_manifest_bytes=mi["bytes"],
        all_cases_retained=True,actual_automatic_inference_verified=True,confidence=gate.masks.CONFIDENCE,text_threshold=gate.masks.TEXT_THRESHOLD,nms_iou=gate.masks.NMS_IOU,
        ambiguity_margin=gate.masks.AMBIGUITY_MARGIN,actual_detector_calls=48,actual_sam2_calls=48,actual_sam2_image_encoder_calls=24,records=rows,model_assets=assets)
    path=base/"automatic_masks/report.json";path.write_text(json.dumps(report));path.chmod(0o444);ri=gate.protocol.identity(path)
    value=dict(schema=gate.PIN_SCHEMA,manifest_sha256=mi["sha256"],manifest_bytes=mi["bytes"],
        automatic_masks=dict(producer_revision="b"*40,report_sha256=ri["sha256"],report_bytes=ri["bytes"],script_sha256=si["sha256"]))
    return cohort,base,path,report,value


def test_public24_reader_authenticates_rgb_masks_and_source_no_private(gate,monkeypatch,tmp_path):
    cohort,base,path,report,value=small_public_fixture(gate,monkeypatch,tmp_path)
    records,receipt=gate.public_inputs(tmp_path,value,cohort=cohort)
    assert len(records)==24 and records[23]["group_index"]==7 and receipt["automatic_masks"]["sha256"]==value["automatic_masks"]["report_sha256"]
    # DWPose mounts/reads only human PNGs, though both query metadata remains audited.
    for row in report["records"]:(base/"automatic_masks"/row["object_mask_file"]).unlink()
    assert len(gate.public_inputs(tmp_path,value,("human",),cohort)[0])==24
    with pytest.raises((ValueError,FileNotFoundError)):gate.public_inputs(tmp_path,value,cohort=cohort)
    assert not(base/"eval_private").exists()


@pytest.mark.parametrize("fault",["private","counter","source","row_order","RGB","empty_mask","partial"])
def test_receipt_structure_remains_strict_even_if_new_sha_is_supplied(gate,monkeypatch,tmp_path,fault):
    cohort,base,path,report,value=small_public_fixture(gate,monkeypatch,tmp_path)
    if fault=="private":report["private_truth_read"]=True
    elif fault=="counter":report["actual_detector_calls"]=47
    elif fault=="source":report["source_helpers"]["body_helper"]="a"*64
    elif fault=="row_order":report["records"][0],report["records"][1]=report["records"][1],report["records"][0]
    elif fault=="RGB":report["records"][5]["rgb_sha256"]="a"*64
    elif fault=="empty_mask":report["records"][0]["human_mask_pixels"]=0
    else:report["records"].pop()
    path.chmod(0o644);path.write_text(json.dumps(report));path.chmod(0o444);identity=gate.protocol.identity(path)
    value["automatic_masks"].update(report_sha256=identity["sha256"],report_bytes=identity["bytes"])
    with pytest.raises(ValueError):gate.public_inputs(tmp_path,value,cohort=cohort)


def test_body_uses_original_native_decoders_and_same_public_K_no_renderer(gate):
    tree=ast.parse(Path(gate.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="run_body")
    calls=[ast.unparse(n.func)for n in ast.walk(function)if isinstance(n,ast.Call)]
    assert "native.human.load_model"in calls and "native.human.decode_prediction"in calls and "body.raw_keypoint_decode"in calls
    assert "native.human.body._source_identity"in calls and "native.human.body._body_assets"in calls
    assert not any("MoGe"in c or "render"in c or "shared_decode"in c for c in calls)
    source=ast.unparse(function);assert "inference_type='body'"in source and "cam_int=camera"in source
    assert source.index("report['body_attempts'] += 1")<source.index("estimator.process_one_image")


def test_dw_calls_unmodified_133_pipeline_24_times_one_cpu_session(gate):
    tree=ast.parse(Path(gate.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="run_dwpose")
    source=ast.unparse(function)
    assert "implementation.inference_pose(proxy, box.copy(), rgb)"in source and "providers=['CPUExecutionProvider']"in source
    assert "smoke.validate_options"in source and "smoke.private_prefix"in source and "COHORT.count"in source
    assert "astype(np.float32)"not in source and "channel"not in source


def test_wrapper_canonical_stages_readonly_inputs_no_private_or_aliases(gate):
    shell=REPO/"infra/run_factorial_rgb_observe.sh";subprocess.run(["bash","-n",str(shell)],check=True);source=shell.read_text()
    assert "eval_private"not in source and "--network none"in source and "--memory"in source
    assert "MEMORY=32g"in source and "MEMORY=8g"in source and "SECONDS_LIMIT=303"in source
    assert "factorial_rgb_manifest_pins_v1.json"in source and "factorial_rgb_public_pins_v1.json"in source
    assert "run_factorial_rgb_observe/code/infra/factorial_rgb_observe.py"in source
    assert "for g in 00 01 02 03 04 05 06 07"in source and "for f in 000 001 002"in source
    assert "dst=$path,readonly"in source and "dst=$CODE,readonly"in source and "--env CUDA_VISIBLE_DEVICES="in source


def test_runtime_source_closure_keeps_all_literal_sibling_proofs(gate):
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in("infra","src","configs")for p in(REPO/folder).rglob("*")
        if p.is_file()and"__pycache__"not in p.parts and p.suffix!=".pyc"}
    files["pyproject.toml"]=(REPO/"pyproject.toml").read_bytes()
    selected=azure_job.runtime_bundle_paths(files,"infra/run_factorial_rgb_observe.sh")
    assert "infra/run_dwpose_smoke.sh"in selected and "infra/factorial_rgb_protocol.py"in selected
    assert "infra/factorial_rgb_render.py"not in selected and "infra/root5_rgb_fit.py"not in selected
    assert all(p.startswith(("infra/","src/","configs/"))or p=="pyproject.toml"for p in selected)

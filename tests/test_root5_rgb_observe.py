"""H98 explicit public-stage contracts; no model/images/GT network execution."""
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA));monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec=importlib.util.spec_from_file_location("root5_observer_test",INFRA/"root5_rgb_observe.py")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def inputs(gate,tmp_path):
    cohort=gate.protocol.COHORT;base=tmp_path/cohort.base;directory=base/"inputs";directory.mkdir(parents=True)
    body=struct.pack(">IIBBBBB",cohort.width,cohort.height,8,2,0,0,0)
    png=b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+body+struct.pack(">I",zlib.crc32(b"IHDR"+body)&0xffffffff)+b"test-only-not-an-image"
    hashes=[]
    for i in range(cohort.count):
        p=directory/cohort.frame_name(i);p.write_bytes(png);p.chmod(0o444);hashes.append(gate.protocol.identity(p)["sha256"])
    p=directory/"manifest.json";p.write_text(json.dumps(gate.protocol.public_manifest(hashes,cohort)));p.chmod(0o444)
    records,receipt=gate.protocol.public_inputs(directory,cohort)
    return base,records,receipt


def mask_fixture(gate,tmp_path):
    cohort=gate.protocol.COHORT;base,records,manifest=inputs(gate,tmp_path);out=base/"automatic_masks";out.mkdir()
    report=dict(stage=gate.STAGES["masks"],status="pass",phase="complete",cohort=asdict(cohort),frames=15,
        script_sha256=gate.MASK_SOURCE_SHA,source_helpers=gate.source_identity()|{"observer":gate.MASK_SOURCE_SHA},network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],human_query="person.",object_query="bottle.",
        producer_revision=gate.MASK_REVISION,input_manifest_sha256=manifest["sha256"],input_manifest_bytes=manifest["bytes"],all_cases_retained=True,
        actual_automatic_inference_verified=True,confidence=gate.masks.CONFIDENCE,text_threshold=gate.masks.TEXT_THRESHOLD,
        nms_iou=gate.masks.NMS_IOU,ambiguity_margin=gate.masks.AMBIGUITY_MARGIN,actual_detector_calls=30,actual_sam2_calls=30,
        actual_sam2_image_encoder_calls=15,model_assets={n:dict(path=str(tmp_path/"weights"/n),sha256=d,bytes=s)for n,(d,s)in gate.masks.ASSETS.items()},records=[])
    for record in records:
        row={k:record[k]for k in("file","clip_index","frame_index")}|dict(rgb_sha256=record["sha256"])
        for label,query in gate.QUERIES:
            name=Path(record["file"]).stem+"_"+label+".png";p=out/name;p.write_bytes(b"public-mask-only");p.chmod(0o444);identity=gate.protocol.identity(p)
            row.update({label+"_query":query,label+"_mask_file":name,label+"_mask_sha256":identity["sha256"],label+"_mask_bytes":identity["bytes"],label+"_mask_pixels":10})
        report["records"].append(row)
    p=out/"report.json"
    def save():
        if p.exists():p.chmod(0o644)
        p.write_text(json.dumps(report));p.chmod(0o444)
    save()
    historical=tmp_path/"jobs"/gate.MASK_REVISION/"run_root5_rgb_observe/code/infra/root5_rgb_observe.py"
    historical.parent.mkdir(parents=True);historical.write_text("synthetic historical source fixture");historical.chmod(0o444)
    # Exact-source pin only is replaced for a synthetic byte fixture, never runtime.
    actual=gate.protocol.identity(historical)["sha256"]
    gate.MASK_SOURCE_SHA=actual;report["script_sha256"]=actual;report["source_helpers"]["observer"]=actual;save()
    return base,records,report,save


def test_explicit_cohort_reader_has_no_private_data(gate,tmp_path):
    base,records,_,_=mask_fixture(gate,tmp_path)
    result,receipt=gate.public_inputs(tmp_path,gate.protocol.COHORT)
    assert len(result)==15 and result[0]["human_mask_path"].parent==base/"automatic_masks"
    assert result[-1]["clip_index"]==2 and result[-1]["frame_index"]==4
    assert receipt["manifest"]["bytes"]>0 and not(base/"eval_private").exists()


@pytest.mark.parametrize("fault",["status","GT","calls","count","order","query","RGBAsha","asset","source","extra","emptyarea","maskbytes","manifestsha","revision"])
def test_completed_mask_binding_fail_closed(gate,tmp_path,fault):
    base,records,report,save=mask_fixture(gate,tmp_path)
    if fault=="status":report["status"]="fail"
    elif fault=="GT":report["private_truth_read"]=True
    elif fault=="calls":report["actual_detector_calls"]=29
    elif fault=="count":report["records"].pop()
    elif fault=="order":report["records"][0]["frame_index"]=1
    elif fault=="query":report["records"][0]["human_query"]="woman."
    elif fault=="RGBAsha":report["records"][0]["rgb_sha256"]="b"*64
    elif fault=="asset":report["model_assets"]["sam2/sam2.1_hiera_large.pt"]["bytes"]+=1
    elif fault=="source":report["source_helpers"]["protocol"]="b"*64
    elif fault=="extra":(base/"automatic_masks"/"private-camera.json").write_bytes(b"hidden")
    elif fault=="emptyarea":report["records"][0]["human_mask_pixels"]=0
    elif fault=="maskbytes":report["records"][0]["human_mask_bytes"]+=1
    elif fault=="manifestsha":report["input_manifest_sha256"]="b"*64
    else:report["producer_revision"]="main"
    save()
    with pytest.raises(ValueError):gate.public_inputs(tmp_path,gate.protocol.COHORT)


def test_dwpose_reader_does_not_open_object_pixels(gate,tmp_path):
    base,_,_,_=mask_fixture(gate,tmp_path)
    for p in(base/"automatic_masks").glob("*_object.png"):p.unlink()
    records,_=gate.public_inputs(tmp_path,gate.protocol.COHORT,("human",))
    assert len(records)==15 and all("object_mask_path"not in r for r in records)
    with pytest.raises(ValueError):gate.public_inputs(tmp_path,gate.protocol.COHORT)
    with pytest.raises(ValueError):gate.public_inputs(tmp_path,gate.protocol.COHORT,("object",))


def test_no_old_global_patch_or_renderer_reader_call(gate):
    tree=ast.parse(Path(gate.__file__).read_text());imports=[];calls=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom):imports.append(n.module or "")
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):calls.append(ast.unparse(n.func))
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign)else[n.target]
            assert not any(isinstance(t,ast.Attribute)and ast.unparse(t).startswith(("body.","native.","dw_helper.","protocol."))for t in targets)
    assert not any("render"in name for name in imports)
    assert "body.public_inputs"not in calls and"dw_helper.public_inputs"not in calls and"native.public_inputs"not in calls
    assert "body.validate_raw"in calls and"body.validate_pair"in calls and"body.validate_clip_constants"in calls
    assert "body.shared_decode"in calls and"body.raw_keypoint_decode"in calls
    assert "dw_helper.validate_prediction"in calls and"dw_helper.validate_smoke"in calls


def test_input_cohort_and_original_K_exact_no_old_provenance(gate):
    assert gate.cohort_contract(gate.protocol.COHORT)==asdict(gate.protocol.COHORT)
    with pytest.raises(ValueError):gate.cohort_contract(SimpleNamespace(count=15,width=1024,height=768))
    source=Path(gate.__file__).read_text()
    assert "validation/keypoint_rgb_v1"not in source and "eval_private"not in source
    assert "private_truth_read=False"in source and "inference_type=\"body\""in source
    assert "cam_int=camera"in source and "cohort.prior_focal"in source
    assert "identity_source=\"first_original_RGB_shape45_and_scale28_per_five_frame_clip\""in source


def test_stage_argument_parser_fails_before_data(gate,tmp_path,monkeypatch):
    monkeypatch.setenv("WR_ROOT",str(tmp_path));monkeypatch.setenv("WR_CODE_REVISION","a"*40)
    for args in([], ["wrong"], ["masks","--unknown"], ["baseline","baseline"]):
        with pytest.raises(SystemExit):gate.main(args)
    with pytest.raises(ValueError):gate.main(["dwpose"])
    assert not(tmp_path/gate.protocol.BASE).exists()


def test_native_mock_decoder_forwards_only_existing_helpers(gate,monkeypatch):
    raw=dict(global_rot=np.zeros(3,np.float32));marker=object();seen=[]
    monkeypatch.setattr(gate.body,"shared_decode",lambda *a:(seen.append(a)or marker))
    assert gate.body.shared_decode("torch","head",raw,raw)is marker and len(seen)==1
    # Real production helpers remain unchanged and callable; no replacement in main.
    assert callable(gate.native.human.load_model)and callable(gate.native.human.decode_prediction)


def test_native_inputs_are_identical_RGB_no_channel_swap_or_confidence_tuning(gate):
    source=Path(gate.__file__).read_text()
    assert "implementation.inference_pose(proxy,box.copy(),np.ascontiguousarray(rgb))"in source
    assert "smoke.SessionProxy"in source and "session.disable_fallback()"in source
    assert "confidence_threshold_applied=False"in source and "raw_scores_clamped=False"in source
    assert "multimask_output=False"in source and "masks.select_person"in source
    assert "actual_detector_calls\"]+=1"in source and "actual_sam2_image_encoder_calls\"]+=1"in source


def test_wrapper_three_stages_output_scoped_private_firewall(gate):
    p=INFRA/"run_root5_rgb_observe.sh";subprocess.run(["bash","-n",str(p)],check=True);s=p.read_text()
    assert 'case "$STAGE" in masks|baseline|dwpose)'in s and "eval_private"not in s
    assert 'BASE="$ROOT/validation/root5_rgb_v1"'in s and 'OUT="$BASE/$FOLDER"'in s
    assert "BUDGET=183"in s and"BUDGET=603"in s and"--network none"in s
    assert 'GPUTAGS=(--gpus all)'in s and'CPU_ENV=(--env CUDA_VISIBLE_DEVICES=)'in s
    assert '"$BASE/automatic_masks/report.json"'in s and '"$BASE/automatic_masks/clip_${c}_frame_${f}_human.png"'in s
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly'in s
    assert "chown -R"not in s and "run_root5_rgb_observe.py"not in s
    assert '"$CODE/infra/root5_rgb_observe.py" "$STAGE"'in s
    assert "dwpose_acquire.py"in s and"run_keypoint_rgb_dwpose.sh"in s


def test_actual_mask_pixel_area_rechecked_for_both_modalities(gate,monkeypatch):
    arrays={"h":np.array([[0,255],[0,255]],np.uint8),"o":np.array([[255,0],[0,0]],np.uint8)}
    monkeypatch.setattr(gate.native.joint,"read_mask",lambda path,image:arrays[path])
    record=dict(human_mask_path="h",object_mask_path="o",human_mask_pixels=2,object_mask_pixels=1)
    hm,om=gate.read_public_masks(record,None)
    assert np.array_equal(hm,arrays["h"])and np.array_equal(om,arrays["o"])
    record["object_mask_pixels"]=2
    with pytest.raises(ValueError):gate.read_public_masks(record,None)
    assert len(gate.read_public_masks(record,None,("human",)))==1
    record["human_mask_pixels"]=1
    with pytest.raises(ValueError):gate.read_public_masks(record,None,("human",))


def test_native_pointmap_ABI_validator_is_reused(gate):
    source=Path(gate.__file__).read_text()
    assert "native.joint.pointmap_contract(z,points,valid,normalized,K)"in source
    assert "checks=validate_camera_pointmap"not in source


def test_historical_mask_source_remains_required_and_never_executed(gate,tmp_path):
    base,_,_,_=mask_fixture(gate,tmp_path)
    path,_=gate.historical_mask_source(tmp_path)
    assert gate.mask_source_helpers(tmp_path)["observer"]==gate.MASK_SOURCE_SHA
    path.chmod(0o644)
    with pytest.raises(ValueError):gate.public_inputs(tmp_path,gate.protocol.COHORT)
    source=Path(gate.__file__).read_text()
    assert 'FOLDERS={"masks":"automatic_masks","baseline":"baseline_v2"'in source
    assert "semantic_id=native.regular(semantic_path)"in source
    assert "protocol.identity(semantic_path)"not in source


def test_preserved_premodel_failure_exact_structural_guard(gate,tmp_path):
    c=gate.protocol.COHORT;path=tmp_path/c.base/"baseline_v1/report.json";path.parent.mkdir(parents=True)
    row=dict(stage=gate.STAGES["baseline"],status="fail",phase="public_integrity",producer_revision=gate.MASK_REVISION,
        script_sha256=gate.MASK_SOURCE_SHA,image_id=gate.native.IMAGE_ID,network="none",frames=15,private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],body_calls_completed=0,
        MoGe_calls_completed=0,raw_parity_head_calls_completed=0,raw_keypoint_head_calls_completed=0,shared_head_calls_completed=0,
        official_reference_calls=0,error_type="ValueError",error="Canonical immutable public regular file required")
    def write():
        if path.exists():path.chmod(0o644)
        raw=json.dumps(row).encode();assert len(raw)<3971;path.write_bytes(raw+b" "*(3971-len(raw)));path.chmod(0o444)
        gate.BASELINE_V1_FAIL_SHA=gate.protocol.identity(path)["sha256"]
    write();receipt=gate.previous_baseline_failure(tmp_path,c);assert receipt["bytes"]==3971
    for key,value in(("body_calls_completed",1),("private_truth_read",True),("decision",{})):
        original=row.copy();row[key]=value;write()
        with pytest.raises(ValueError):gate.previous_baseline_failure(tmp_path,c)
        row=original

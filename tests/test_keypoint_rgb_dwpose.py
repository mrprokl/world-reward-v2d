"""Tiny own fixtures exercise contracts; never claim native ORT/model execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

INFRA = Path(__file__).resolve().parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA))
    spec = importlib.util.spec_from_file_location("own_keypoint_dwpose", INFRA/"keypoint_rgb_dwpose.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def public_fixture(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 8)
    public = tmp_path/gate.BASE/"inputs"; masks = tmp_path/gate.BASE/"automatic_masks"
    public.mkdir(parents=True); masks.mkdir()
    images, rows = [], []
    for index in range(15):
        clip, frame = divmod(index, 5); stem = f"clip_{clip:02d}_frame_{frame:03d}"
        rgb = public/(stem+".png"); human = masks/(stem+"_human.png")
        Image.fromarray(np.full((8,8,3), 40+index, np.uint8)).save(rgb)
        value=np.zeros((8,8), np.uint8); value[1:7,2:6]=255; Image.fromarray(value).save(human)
        r, h = gate.smoke.identity(rgb), gate.smoke.identity(human)
        images.append(dict(file=rgb.name, sha256=r["sha256"], width=8, height=8))
        row=dict(file=rgb.name, rgb_sha256=r["sha256"], clip_index=clip, frame_index=frame)
        for label, query in (("human","person."),("object","bottle.")):
            row.update({label+"_query":query,label+"_mask_file":stem+"_"+label+".png",
                        label+"_mask_sha256":h["sha256"],label+"_mask_bytes":h["bytes"],label+"_mask_pixels":24,
                        label+"_box":[2.,1.,6.,7.],label+"_detector_score":.8,label+"_sam2_predicted_score":.9})
        rows.append(row)
    manifest=public/"manifest.json"; manifest.write_text(json.dumps(dict(schema="world-reward-keypoint-rgb-v1",images=images)))
    mid=gate.smoke.identity(manifest); monkeypatch.setattr(gate,"MANIFEST_SHA",mid["sha256"]); monkeypatch.setattr(gate,"MANIFEST_BYTES",mid["bytes"])
    report=dict(stage="public_keypoint_rgb_automatic_masks",status="pass",phase="complete",frames=15,
        producer_revision=gate.MASK_REVISION,script_sha256=gate.MASK_SOURCE_SHA,network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],human_query="person.",object_query="bottle.",
        actual_detector_calls=30,actual_sam2_calls=30,actual_sam2_image_encoder_calls=15,actual_automatic_inference_verified=True,
        all_cases_retained=True,input_manifest_sha256=mid["sha256"],input_manifest_bytes=mid["bytes"],records=rows)
    path=masks/"report.json"; path.write_text(json.dumps(report)); monkeypatch.setattr(gate,"MASK_SHA",gate.smoke.identity(path)["sha256"])
    return public,masks,report


def test_public_producer_shaped_all15_no_private_or_object_files_read(gate,tmp_path,monkeypatch):
    public,masks,_=public_fixture(gate,tmp_path,monkeypatch)
    before={p:p.read_bytes() for d in (public,masks) for p in d.iterdir()}
    records,receipt=gate.public_inputs(tmp_path)
    assert len(records)==15 and receipt["manifest"]["sha256"]==gate.MANIFEST_SHA
    assert [r["clip_index"] for r in records]==[0]*5+[1]*5+[2]*5
    assert all(r["human_mask_pixels"]==24 for r in records)
    assert not list(masks.glob("*_object.png"))
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize("fault",["private","order","boolwidth","rgb","human","query","calls","clipbool","missing","extra","masksha","size","symlink"])
def test_public_firewall_tamper_and_no_partial_selection(gate,tmp_path,monkeypatch,fault):
    public,masks,report=public_fixture(gate,tmp_path,monkeypatch)
    manifest=json.loads((public/"manifest.json").read_text())
    if fault=="private":manifest["camera_K"]=[[1,0,0],[0,1,0],[0,0,1]]
    elif fault=="order":manifest["images"].reverse()
    elif fault=="boolwidth":manifest["images"][0]["width"]=True
    elif fault=="rgb":(public/manifest["images"][0]["file"]).write_bytes(b"tampered")
    elif fault=="human":(masks/report["records"][0]["human_mask_file"]).write_bytes(b"tampered")
    elif fault=="query":report["records"][0]["object_query"]="person."
    elif fault=="calls":report["actual_sam2_calls"]=29
    elif fault=="clipbool":report["records"][0]["clip_index"]=False
    elif fault=="missing":manifest["images"].pop()
    elif fault=="extra":(public/"labels.json").write_text("{}")
    elif fault=="masksha":report["records"][0]["object_mask_sha256"]="bad"
    elif fault=="size":report["records"][0]["human_mask_bytes"]+=1
    elif fault=="symlink":
        path=public/manifest["images"][0]["file"]; path.rename(tmp_path/"original.png"); path.symlink_to(tmp_path/"original.png")
    (public/"manifest.json").write_text(json.dumps(manifest)); mid=gate.smoke.identity(public/"manifest.json")
    monkeypatch.setattr(gate,"MANIFEST_SHA",mid["sha256"]); monkeypatch.setattr(gate,"MANIFEST_BYTES",mid["bytes"])
    report["input_manifest_sha256"]=mid["sha256"];report["input_manifest_bytes"]=mid["bytes"]
    (masks/"report.json").write_text(json.dumps(report));monkeypatch.setattr(gate,"MASK_SHA",gate.smoke.identity(masks/"report.json")["sha256"])
    with pytest.raises(ValueError):gate.public_inputs(tmp_path)


def prediction():
    points=np.ones((1,133,2),np.float64); scores=np.ones((1,133),np.float32); scores[0,0]=-1
    return dict(keypoints=points,scores=scores,validity=scores>0,bbox=np.array([[1,2,10,12]],np.float32),
                clip_index=np.array(0,np.int64),frame_index=np.array(0,np.int64))


def test_native_score_not_mapped_coordinate_sentinel(gate):
    data=prediction();gate.validate_prediction(data,dict(clip_index=0,frame_index=0))
    assert not data["validity"][0,0] and (data["keypoints"][0,0]>0).all()


@pytest.mark.parametrize("fault",["extra","133","scoretype","validity","bbox","bboxoutside","bboxmasked","clip","framebool","nan"])
def test_prediction_exact133_dtype_and_frame_contract(gate,fault):
    data=prediction()
    if fault=="extra":data["pose"]=np.zeros(3)
    elif fault=="133":data["keypoints"]=np.ones((1,134,2))
    elif fault=="scoretype":data["scores"]=data["scores"].astype(np.float64)
    elif fault=="validity":data["validity"][0,0]=True
    elif fault=="bbox":data["bbox"][0,2]=0
    elif fault=="bboxoutside":data["bbox"][0,2]=1025
    elif fault=="bboxmasked":data["bbox"]=np.ma.array(data["bbox"],mask=False)
    elif fault=="clip":data["clip_index"]=np.array(1,np.int64)
    elif fault=="framebool":data["frame_index"]=np.array(False)
    else:data["keypoints"][0,0,0]=np.nan
    with pytest.raises(ValueError):gate.validate_prediction(data,dict(clip_index=0,frame_index=0))


def test_fixed_helpers_and_pins_source_identity(gate):
    rows=gate.source_identity()
    assert rows["dwpose_smoke.py"]["sha256"]==gate.SMOKE_SOURCE_SHA
    assert gate.MANIFEST_BYTES==2199 and gate.BUDGET==180
    assert gate.PREDICTION_KEYS=={"keypoints","scores","validity","bbox","clip_index","frame_index"}


@pytest.mark.parametrize("failure",[False,True])
def test_full15_native_style_orchestration_and_cleanup_no_mock_execution_claim(gate,tmp_path,monkeypatch,failure):
    public_fixture(gate,tmp_path,monkeypatch)
    # Test-only tiny grid; production uses the unchanged helper's1024×768 grid.
    monkeypatch.setattr(gate.smoke,"WIDTH",8);monkeypatch.setattr(gate.smoke,"HEIGHT",8)
    out=tmp_path/gate.OUT;out.mkdir();(out/"report.json").write_text("{}");report=dict(calls=[],records=[],actual_sessions=0)
    monkeypatch.setattr(gate,"source_identity",lambda:{"ownfixture":True})
    monkeypatch.setattr(gate,"validate_smoke",lambda root:{"ownfixture":True})
    monkeypatch.setattr(gate.smoke,"validate_assets",lambda root:{"ownfixture":True})
    monkeypatch.setattr(gate.smoke,"dependency_identity",lambda:{"ownfixture":True})
    root_source=tmp_path/gate.smoke.acquisition.BASE/"source";root_source.mkdir(parents=True)
    (root_source/"onnxpose.py").write_text("""import numpy as np
def inference_pose(session,bbox,image):
    norm=(np.full((384,288,3),128,np.uint8)-np.array([123.675,116.28,103.53]))/np.array([58.395,57.12,57.375])
    x,y=session.run(['simcc_x','simcc_y'],{'input':[norm.transpose(2,0,1)]})
    points=np.stack([x.argmax(-1),y.argmax(-1)],-1).astype(np.float64)/2
    return points,np.minimum(x.max(-1),y.max(-1))
""")
    calls=[];constructed=[]
    class Session:
        def __init__(self,path,sess_options,providers):self.options=sess_options;constructed.append(self)
        def disable_fallback(self):self.fallback=True
        def get_session_options(self):return self.options
        def get_providers(self):return ["CPUExecutionProvider"]
        def get_inputs(self):return [SimpleNamespace(name="input",type="tensor(float)",shape=["batch",3,384,288])]
        def get_outputs(self):return [SimpleNamespace(name="simcc_"+a,type="tensor(float)",shape=["batch","MatMulsimcc_"+a+"_dim_1","MatMulsimcc_"+a+"_dim_2"]) for a in ("x","y")]
        def get_modelmeta(self):return SimpleNamespace(custom_metadata_map={})
        def run(self,names,feed):
            assert self.fallback and type(feed["input"]) is list and feed["input"][0].dtype==np.float64
            calls.append(feed["input"][0].copy())
            if failure:raise RuntimeError("ownfixture failure")
            x=np.zeros((1,133,576),np.float32);y=np.zeros((1,133,768),np.float32);x[...,100]=2;y[...,150]=3;return [x,y]
    ort=SimpleNamespace(SessionOptions=lambda:SimpleNamespace(),ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),InferenceSession=Session)
    monkeypatch.setattr(gate.smoke,"import_runtime",lambda prefix:(ort,{"ownfixture":True}))
    def piprun(command,**kwargs):assert "--no-index" in command and "--no-deps" in command;return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess,"run",piprun)
    if failure:
        with pytest.raises(RuntimeError,match="ownfixture"):gate.perform(tmp_path,out,report,lambda:None,time.perf_counter())
        assert len(calls)==1 and not report["calls"][0]["run_completed"] and not report.get("native_cpu_abi_verified",False)
    else:
        gate.perform(tmp_path,out,report,lambda:None,time.perf_counter())
        assert report["status"]=="pass" and len(calls)==len(report["records"])==15 and len(constructed)==1
        assert report["native_cpu_abi_verified"] and report["final_inputs_source_assets_rehashed"]
        gate.validate_artifacts(out,report["records"])
        record=report["records"][0];(out/record["prediction_file"]).chmod(0o644);(out/record["prediction_file"]).write_bytes(b"tampered")
        with pytest.raises(ValueError):gate.validate_artifacts(out,report["records"])
    assert report["private_prefix_removed"] is True and not Path(report["private_prefix"]).exists()


def test_cli_no_tuning_and_wrapper_cpu_public_firewall(gate):
    for argv in (["--episode","15"],["--dtype","float32"],["--threshold",".5"]):
        with pytest.raises(SystemExit):gate.main(argv)
    path=INFRA/"run_keypoint_rgb_dwpose.sh";subprocess.run(["bash","-n",str(path)],check=True);text=path.read_text()
    assert gate.BASE in text and 'OUT="$BASE/dwpose_v1"' in text and "--network none --memory 8g --cpus 4" in text
    assert "--gpus" not in text and "eval_private" not in text and "_object.png" not in text and "body_full" not in text
    assert "_human.png" in text and "--env CUDA_VISIBLE_DEVICES=" in text
    tree=ast.parse(Path(gate.__file__).read_text());imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any("torch" in ast.unparse(n) or "onnxruntime" in ast.unparse(n) for n in imports)


def completed_smoke_fixture(gate,tmp_path,monkeypatch):
    """Actual D95v2 producer fields, with own numeric arrays (not real outputs)."""
    rows=[]
    for file in ("clip_00_frame_000.png","clip_01_frame_000.png"):
        values=prediction()
        rows.append(dict(file=file,**{k:gate.smoke.array_identity(values[k]) for k in ("keypoints","scores","validity")},
            raw_simcc=[gate.smoke.array_identity(np.ones((1,133,n),np.float32)) for n in (576,768)]))
    report=dict(stage=gate.smoke.STAGE,status="pass",phase="complete",producer_revision=gate.SMOKE_REVISION,
        script_sha256=gate.SMOKE_SOURCE_SHA,image_id=gate.smoke.audit.IMAGE,
        previous_failed_smoke_sha256=gate.smoke.PREVIOUS_SMOKE_SHA,previous_failure_rewritten=False,
        device="cpu",network="none",native_cpu_abi_verified=True,two_session_byte_replay_verified=True,
        private_prefix_packages_installed=True,private_prefix_removed=True,final_inputs_source_assets_rehashed=True,
        native_source_modified=False,own_feed_cast=False,channel_swap=False,full_image_fallback=False,
        raw_scores_clamped=False,confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,
        gpu_used=False,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,
        oracle_modes=[],independent_quality_cohort=False,accuracy_verified=False,license_clearance_verified=False,
        adoption_authorized=False,semantic_detection_verified=False,training_overlap_excluded=False,
        sessions=[dict(predictions=copy.deepcopy(rows),calls=[dict(run_completed=True)]*2) for _ in range(2)])
    directory=tmp_path/gate.smoke.OUT;directory.mkdir(parents=True);path=directory/"report.json"
    previous={"path":"own historical failed receipt","sha256":"a"*64,"bytes":7}
    monkeypatch.setattr(gate.smoke,"validate_previous_smoke",lambda root:previous)
    def seal():
        path.write_text(json.dumps(report));monkeypatch.setattr(gate,"SMOKE_SHA",gate.smoke.identity(path)["sha256"])
    seal()
    return report,path,seal,previous


def test_actual_pass_predicate_receipt_and_replay_fields(gate,tmp_path,monkeypatch):
    _,path,_,previous=completed_smoke_fixture(gate,tmp_path,monkeypatch)
    before=path.read_bytes();evidence=gate.validate_smoke(tmp_path)
    assert evidence["actual_pass_receipt"]["sha256"]==gate.SMOKE_SHA and evidence["previous_failed_receipt"]==previous
    assert path.read_bytes()==before


@pytest.mark.parametrize("fault",["stage","status","phase","revision","cleanup","booltype","feedcast","missingcall","replay","sha"])
def test_actual_smoke_predicate_no_weakened_producer_or_replay(gate,tmp_path,monkeypatch,fault):
    report,path,seal,_=completed_smoke_fixture(gate,tmp_path,monkeypatch)
    if fault=="stage":report["stage"]="other"
    elif fault=="status":report["status"]="fail"
    elif fault=="phase":report["phase"]="native_source_load"
    elif fault=="revision":report["producer_revision"]="0"*40
    elif fault=="cleanup":report["private_prefix_removed"]=False
    elif fault=="booltype":report["native_cpu_abi_verified"]=1
    elif fault=="feedcast":report["own_feed_cast"]=True
    elif fault=="missingcall":report["sessions"][0]["calls"].pop()
    elif fault=="replay":report["sessions"][1]["predictions"][0]["raw_simcc"][0]["sha256"]="0"*64
    else:path.write_bytes(b"changed immutable receipt")
    if fault!="sha":seal() # Predicate negatives remain bound to their own fixture bytes.
    with pytest.raises(ValueError):gate.validate_smoke(tmp_path)

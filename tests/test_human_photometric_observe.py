"""Tiny public/native-contract tests; no actual learned model or GPU execution."""
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("human_photometric_observe_test",ROOT/"infra/human_photometric_observe.py")
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def proposal(gate,index=0):
    data={k:np.zeros(s,np.float32) for k,s in gate.native.BLOCKS.items()}
    data["global_rot"][0]=data["mhr_model_params"][3]=np.float32(index*.001)
    data["pred_cam_t"][:]=[index*.001,0.,3.]
    data["shape_params"][0]=np.float32(index*.01);data["scale_params"][0]=np.float32(index*.001)
    data["mhr_model_params"][136:]=np.float32(index*.001)
    data.update(vertices_camera_m=np.tile([0.,0.,3.],(18439,1)).astype(np.float32),keypoints_camera_m=np.tile([0.,0.,3.],(308,1)).astype(np.float32),
        joints_camera_m=np.tile([0.,0.,3.],(127,1)).astype(np.float32),joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)))
    return data


def decoder(gate,blocks,first):
    p=proposal(gate);controls=blocks["mhr_model_params"].copy();controls[3:6]=blocks["global_rot"];controls[136:]=first["mhr_model_params"][136:]
    return p["vertices_camera_m"],p["keypoints_camera_m"],p["joints_camera_m"],controls,p["joint_global_rotations"]


def test_frame_anchor_only_identity_clipconstant_not_frame0_camera(gate,monkeypatch):
    original,first=proposal(gate,2),proposal(gate,0);before={k:v.copy() for k,v in original.items()};seen=[]
    def shared(torch,head,blocks,identity):
        seen.append(blocks);return decoder(gate,blocks,identity)
    monkeypatch.setattr(gate.baseline,"shared_decode",shared)
    anchor=gate.frame_anchor(None,None,original,first)
    for k in ("global_rot","pred_cam_t","hand_pose_params","expr_params","body_pose_params"):
        assert anchor[k].tobytes()==original[k].tobytes()
    for k in ("shape_params","scale_params"):assert anchor[k].tobytes()==first[k].tobytes()
    assert anchor["pred_cam_t"].tobytes()!=first["pred_cam_t"].tobytes()
    assert anchor["mhr_model_params"][136:].tobytes()==first["mhr_model_params"][136:].tobytes()
    for k in original:np.testing.assert_array_equal(original[k],before[k])
    assert seen[0]["body_pose_params"].tobytes()==original["body_pose_params"].tobytes()


@pytest.mark.parametrize("fault",["root","hand","expanded_scale"])
def test_anchor_returned_native_fixed_control_guard(gate,monkeypatch,fault):
    def wrong(torch,head,blocks,first):
        v,k,j,c,r=decoder(gate,blocks,first)
        c[{"root":3,"hand":68,"expanded_scale":136}[fault]]+=.1
        return v,k,j,c,r
    monkeypatch.setattr(gate.baseline,"shared_decode",wrong)
    with pytest.raises(ValueError):gate.frame_anchor(None,None,proposal(gate,2),proposal(gate))


def rig(gate):
    left=np.zeros(18439,bool);right=left.copy();left[:50]=True;right[50:100]=True
    return dict(human_faces=np.tile([0,1,2],(36874,1)).astype(np.int64),hand_mask_left=left,hand_mask_right=right,camera_K=np.asarray(gate.COHORT.fixed_K))


@pytest.mark.parametrize("fault",["none","faces","overlap","missing","K","masked"])
def test_actual_rig_metadata_finite_topology_hands_camera(gate,fault):
    data=rig(gate)
    if fault=="none":assert gate.validate_rig(data)is data;return
    if fault=="faces":data["human_faces"][0,0]=18439
    elif fault=="overlap":data["hand_mask_right"][0]=True
    elif fault=="missing":data.pop("hand_mask_left")
    elif fault=="K":data["camera_K"][0,0]=1279
    else:data["hand_mask_left"]=np.ma.array(data["hand_mask_left"],mask=False)
    with pytest.raises(ValueError):gate.validate_rig(data)


def frame_fixture(gate,monkeypatch,tmp_path):
    """All24×15 real tiny-compressed native-shaped arrays; only decoder is fake."""
    monkeypatch.setattr(gate.baseline,"shared_decode",lambda torch,head,blocks,first:decoder(gate,blocks,first))
    out=tmp_path/"predictions";out.mkdir();(out/"report.json").write_text("{}");r=rig(gate)
    with (out/"rig_metadata.npz").open("xb") as stream:np.savez_compressed(stream,**r)
    (out/"rig_metadata.npz").chmod(0o444);ri=gate.core.identity(out/"rig_metadata.npz")
    records=[];rows=[];first={}
    for i in range(24):
        group,frame=divmod(i,3);original=proposal(gate,i)
        if frame==0:first[group]=original
        anchor=gate.frame_anchor(None,None,original,first[group]);folder=out/Path(gate.COHORT.frame_name(i)).stem;folder.mkdir()
        record=dict(file=gate.COHORT.frame_name(i),group_index=group,frame_index=frame,sha256=f"{i:064x}",human_mask_sha256="a"*64)
        row={k:record[k] for k in ("file","group_index","frame_index")} | dict(rgb_sha256=record["sha256"],human_mask_sha256=record["human_mask_sha256"],
            clip_anchor_file=gate.COHORT.frame_name(group*3),artifacts=[])
        for prefix,p in (("raw",original),("fixed",anchor)):
            for name,_ in gate.core.ALL_BRANCHES:row["artifacts"].append(gate.core.save_proposal(folder,prefix+"_"+name,p))
        fixed=[anchor]*6;row["selection"],row["SHAM"]=gate.selection_record(fixed)
        for mode in ("baseline","sham","tta"):row["artifacts"].append(gate.core.save_proposal(folder,"selected_"+mode,anchor))
        records.append(record);rows.append(row)
    return out,records,rows,ri


def test_all360_frozen_outputs_real_roundtrip(gate,monkeypatch,tmp_path):
    out,records,rows,ri=frame_fixture(gate,monkeypatch,tmp_path)
    values=gate.frozen_frames(out,rows,records,ri)
    assert len(values)==24 and sum(1 for _ in out.rglob("*.npz"))==361
    assert values[2]["baseline"]["shape_params"].tobytes()==values[0]["raw"]["shape_params"].tobytes()
    assert values[2]["baseline"]["pred_cam_t"].tobytes()!=values[0]["baseline"]["pred_cam_t"].tobytes()
    assert all(v["baseline"]["vertices_camera_m"].tobytes()==v["tta"]["vertices_camera_m"].tobytes() for v in values)


@pytest.mark.parametrize("fault",["partial","order","selection","mask","anchor","extra"])
def test_complete_frozen_frame_lineage_no_drop_or_reselect(gate,monkeypatch,tmp_path,fault):
    out,records,rows,ri=frame_fixture(gate,monkeypatch,tmp_path)
    if fault=="partial":rows.pop()
    elif fault=="order":rows[0],rows[1]=rows[1],rows[0]
    elif fault=="selection":rows[0]["selection"]["selected_gamma_index"]=2
    elif fault=="mask":rows[0]["human_mask_sha256"]="b"*64
    elif fault=="anchor":rows[1]["clip_anchor_file"]=records[1]["file"]
    else:(out/"extra").write_text("noise")
    with pytest.raises(ValueError):gate.frozen_frames(out,rows,records,ri)


def small_masks(gate,monkeypatch,tmp_path):
    from PIL import Image
    monkeypatch.setattr(gate.protocol.PublicCohort,"__post_init__",lambda _:None);cohort=gate.protocol.PublicCohort(width=16,height=8)
    base=tmp_path/cohort.base;(base/"inputs").mkdir(parents=True);(base/"automatic_masks").mkdir();digests=[];rows=[]
    for i in range(24):
        name=cohort.frame_name(i);png=base/"inputs"/name;Image.fromarray(np.full((8,16,3),i,np.uint8)).save(png);png.chmod(0o444)
        digest=gate.protocol.identity(png)["sha256"];digests.append(digest)
        mask=base/"automatic_masks"/(Path(name).stem+"_human.png");Image.fromarray(np.full((8,16),255,np.uint8)).save(mask);mask.chmod(0o444);mi=gate.protocol.identity(mask)
        rows.append(dict(file=name,group_index=i//3,frame_index=i%3,rgb_sha256=digest,human_query="person.",human_mask_file=mask.name,
            human_mask_sha256=mi["sha256"],human_mask_bytes=mi["bytes"],human_mask_pixels=128))
    m=base/"inputs/manifest.json";m.write_text(json.dumps(gate.protocol.public_manifest(digests,cohort)));m.chmod(0o444);mi=gate.protocol.identity(m)
    report=dict(stage=gate.STAGES["masks"],status="pass",phase="complete",cohort=asdict(cohort),frames=24,producer_revision="a"*40,
        script_sha256=gate.sha256(Path(gate.__file__)),source_helpers=gate.helper_hashes(),image_id=gate.core.MASK_IMAGE,network="none",
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],human_query="person.",
        input_manifest_sha256=mi["sha256"],input_manifest_bytes=mi["bytes"],all_cases_retained=True,source_inputs_assets_rehashed=True,actual_automatic_inference_verified=True,
        confidence=gate.masks.CONFIDENCE,text_threshold=gate.masks.TEXT_THRESHOLD,nms_iou=gate.masks.NMS_IOU,ambiguity_margin=gate.masks.AMBIGUITY_MARGIN,
        model_assets={n:dict(path=str(tmp_path/"weights"/n),sha256=d,bytes=s) for n,(d,s) in gate.masks.ASSETS.items()},records=rows)
    report.update({k:24 for k in ("detector_attempts","actual_detector_calls","image_encoder_attempts","actual_sam2_image_encoder_calls","sam2_attempts","actual_sam2_calls")})
    path=base/"automatic_masks/report.json";path.write_text(json.dumps(report));path.chmod(0o444)
    return cohort,base,path,report


def test_real24_rgb_mask_reader_no_other_cohort_or_private(gate,monkeypatch,tmp_path):
    cohort,base,path,report=small_masks(gate,monkeypatch,tmp_path)
    records,inputs=gate.public_masks(tmp_path,"a"*40,cohort)
    assert len(records)==24 and records[23]["group_index"]==7 and inputs["mask_report"]["sha256"]==gate.sha256(path)
    assert not (base/"eval_private").exists()
    with pytest.raises(ValueError):gate.public_masks(tmp_path,"b"*40,cohort)


@pytest.mark.parametrize("fault",["private","counter","source","RGB","empty","partial"])
def test_mask_receipt_exact_whole_coverage(gate,monkeypatch,tmp_path,fault):
    cohort,base,path,report=small_masks(gate,monkeypatch,tmp_path)
    if fault=="private":report["private_truth_read"]=True
    elif fault=="counter":report["actual_detector_calls"]=23
    elif fault=="source":report["source_helpers"]["protocol"]="f"*64
    elif fault=="RGB":report["records"][2]["rgb_sha256"]="f"*64
    elif fault=="empty":report["records"][0]["human_mask_pixels"]=0
    else:report["records"].pop()
    path.chmod(0o644);path.write_text(json.dumps(report));path.chmod(0o444)
    with pytest.raises(ValueError):gate.public_masks(tmp_path,"a"*40,cohort)


@pytest.mark.parametrize("mode",["none","true","warn","tf32"])
def test_runtime_empirical_false_no_warning_explicit(gate,mode):
    torch=SimpleNamespace(are_deterministic_algorithms_enabled=lambda:mode=="true",is_deterministic_algorithms_warn_only_enabled=lambda:mode=="warn",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=mode=="tf32")),cudnn=SimpleNamespace(allow_tf32=False)))
    if mode=="none":gate.runtime_guard(torch)
    else:
        with pytest.raises(ValueError):gate.runtime_guard(torch)


def test_bad_public_inputs_before_torch_or_model(gate,tmp_path,monkeypatch):
    monkeypatch.setattr(gate,"public_masks",lambda *_:(_ for _ in ()).throw(ValueError("public rejected")))
    with pytest.raises(ValueError,match="public rejected"):gate.run_body(tmp_path,tmp_path,{},lambda:None,"a"*40)


def test_source_has_original_native_methods_attempt_counters_and_no_fit(gate):
    tree=ast.parse(Path(gate.__file__).read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert "native.human.load_model" in calls and "native.human.decode_prediction" in calls and "baseline.raw_keypoint_decode" in calls
    assert "core.fixed_decode" in calls and "frame_anchor" in calls and "selection_record" in calls and "frozen_frames" in calls
    assert not any(c in calls for c in ("torch.optim.Adam","core.main","core.run_body","manufacture.main"))
    source=Path(gate.__file__).read_text();assert "eval_private" not in source and "photometric_native_v1" not in source
    assert gate.COUNTS==dict(body=144,parity_head=144,keypoint_head=144,anchor_head=24,fixed_head=144,selected_replay=72)
    assert source.index('report["body_attempts"]+=1')<source.index('estimator.process_one_image')


def test_wrapper_is_one_stage_readonly_public_only_no_reference_or_private(gate):
    path=ROOT/"infra/run_human_photometric_observe.sh";text=path.read_text();subprocess.run(["rtk","proxy","bash","-n",str(path)],check=True)
    assert text.count("docker run")==1 and "--stage masks|body" in text
    assert "LIMIT=603" in text and "LIMIT=123" in text and "--memory 32g" in text
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in text and "$BASE/automatic_masks" in text
    assert "eval_private" not in text and "render-report" not in text and "weights/mhr" not in text
    assert 'src=$OUT,dst=$OUT' in text and 'src=$BASE,dst=$BASE' not in text
    assert subprocess.run(["rtk","proxy","bash",str(path),"bad"],capture_output=True).returncode==2

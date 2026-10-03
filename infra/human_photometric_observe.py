"""H101 public gamma/native-BODY pilot, with frame camera and clip identity fixed.

Only original RGB and automatic person masks are observations. Raw gamma/SHAM
predictions remain auditable; selection uses full native-vertex consistency,
never IoU, a private label, coefficient averaging or an evaluation alignment.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import human_photometric_protocol as protocol
import photometric_capability as core
import native_mhr_execution as execution
from world_reward.data import sha256
from world_reward.prompt_selection import BoxDetection

native, masks, baseline = core.native, core.masks, core.baseline
COHORT = protocol.COHORT
BASE, OUT = COHORT.base, COHORT.base+"/predictions_v1"
STAGES = {"masks": "public_human_photometric_automatic_masks", "body": "public_human_photometric_native_predictions"}
FOLDERS, BUDGETS = {"masks": "automatic_masks", "body": "predictions_v1"}, {"masks": 120, "body": 600}
RIG_KEYS = {"human_faces", "hand_mask_left", "hand_mask_right", "camera_K"}
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
COUNTS = {"body": 144, "parity_head": 144, "keypoint_head": 144, "anchor_head": 24, "fixed_head": 144, "selected_replay": 72}
SCOPED_HEADS = 1392


def helper_hashes():
    return dict(observer=sha256(Path(__file__)), protocol=sha256(Path(protocol.__file__)),
        execution=sha256(Path(execution.__file__)), native_mechanism=core.helper_hashes())


def rgb_inputs(root, cohort=COHORT):
    return protocol.public_inputs(Path(root)/cohort.base/"inputs", cohort)


def read_rgb(record, Image, cohort=COHORT):
    with Image.open(record["path"]) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (cohort.width, cohort.height):
            raise ValueError("Original public RGB decode required")
        return np.array(image, dtype=np.uint8, copy=True)


def completed_masks(report, cohort=COHORT):
    core.require_fields(report, dict(detector_attempts=cohort.count, actual_detector_calls=cohort.count,
        image_encoder_attempts=cohort.count, actual_sam2_image_encoder_calls=cohort.count,
        sam2_attempts=cohort.count, actual_sam2_calls=cohort.count))
    rows = report.get("records")
    if type(rows) is not list or len(rows) != cohort.count: raise ValueError("Every ordered automatic person mask required")
    for i, row in enumerate(rows):
        core.require_fields(row, dict(file=cohort.frame_name(i), group_index=i//cohort.frames, frame_index=i%cohort.frames,
            human_query="person.", human_mask_file=Path(cohort.frame_name(i)).stem+"_human.png"))
        if (type(row.get("human_mask_pixels")) is not int or not 20 <= row["human_mask_pixels"] <= cohort.width*cohort.height
                or type(row.get("human_mask_bytes")) is not int or row["human_mask_bytes"] <= 0
                or not re.fullmatch("[0-9a-f]{64}", str(row.get("human_mask_sha256", "")))):
            raise ValueError("Nonempty immutable original-grid automatic masks required")


def public_masks(root, revision, cohort=COHORT):
    records, manifest = rgb_inputs(root, cohort);out = Path(root)/cohort.base/"automatic_masks"
    path = out/"report.json";receipt = protocol.identity(path);report = json.loads(path.read_text())
    core.require_fields(report, dict(stage=STAGES["masks"], status="pass", phase="complete", cohort=asdict(cohort), frames=cohort.count,
        producer_revision=revision, script_sha256=sha256(Path(__file__)), source_helpers=helper_hashes(), image_id=core.MASK_IMAGE,
        network="none", private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False,
        oracle_modes=[], human_query="person.", input_manifest_sha256=manifest["sha256"], input_manifest_bytes=manifest["bytes"],
        all_cases_retained=True, source_inputs_assets_rehashed=True, actual_automatic_inference_verified=True,
        confidence=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, nms_iou=masks.NMS_IOU, ambiguity_margin=masks.AMBIGUITY_MARGIN))
    if any(k in report for k in ("error", "error_type")): raise ValueError("Passing mask producer cannot retain an error")
    completed_masks(report, cohort)
    expected_assets = {name: dict(path=str(Path(root)/"weights"/name), sha256=digest, bytes=size) for name,(digest,size) in masks.ASSETS.items()}
    core.require_fields(report, dict(model_assets=expected_assets))
    for record, row in zip(records, report["records"]):
        core.require_fields(row, dict(rgb_sha256=record["sha256"]))
        mask_path = out/row["human_mask_file"];mask = protocol.identity(mask_path)
        if mask != dict(sha256=row["human_mask_sha256"], bytes=row["human_mask_bytes"]): raise ValueError("Automatic person mask changed")
        record.update(human_mask_path=mask_path, human_mask_pixels=row["human_mask_pixels"], human_mask_sha256=mask["sha256"])
    if {p.name for p in out.iterdir()} != {"report.json", *[r["human_mask_path"].name for r in records]}:
        raise ValueError("Exact24 automatic person mask inventory required")
    return records, dict(manifest=manifest, mask_report=receipt)


def frame_anchor(torch, head, original, clip_first):
    """Replace only the identity; preserve this frame's original root/camera/hands."""
    core.validate_proposal(original);core.validate_proposal(clip_first)
    blocks = {k: original[k].copy() for k in native.BLOCKS}
    for k in ("shape_params", "scale_params"): blocks[k] = clip_first[k].copy()
    v, kp, j, controls, rotations = baseline.shared_decode(torch, head, blocks, clip_first)
    anchor = blocks | dict(vertices_camera_m=v, keypoints_camera_m=kp, joints_camera_m=j,
        mhr_model_params=controls, joint_global_rotations=rotations)
    core.validate_proposal(anchor)
    for k in ("global_rot", "pred_cam_t", "hand_pose_params", "expr_params", "body_pose_params"):
        if anchor[k].tobytes() != original[k].tobytes(): raise ValueError("Frame-original nonidentity input changed")
    for k in ("shape_params", "scale_params"):
        if anchor[k].tobytes() != clip_first[k].tobytes(): raise ValueError("Clip-first identity changed")
    columns = np.r_[0:6, 68:122]
    if anchor["mhr_model_params"][columns].tobytes() != original["mhr_model_params"][columns].tobytes():
        raise ValueError("Frame-original native root/hand controls changed")
    if anchor["mhr_model_params"][136:].tobytes() != clip_first["mhr_model_params"][136:].tobytes():
        raise ValueError("Clip-first expanded68 native scales changed")
    return anchor


def validate_rig(data):
    if set(data) != RIG_KEYS: raise ValueError("Exact native rig metadata required")
    faces = native.array(data["human_faces"], (36874,3), "int64")
    if np.any(faces < 0) or np.any(faces >= 18439): raise ValueError("Complete native topology required")
    if hashlib.sha256(faces.astype("<i4").tobytes()).hexdigest() != FACE_SHA:
        raise ValueError("Exact source-backed native topology SHA required")
    left, right = (native.array(data["hand_mask_"+side], (18439,), "bool") for side in ("left", "right"))
    if left.sum() < 50 or right.sum() < 50 or np.any(left & right): raise ValueError("Disjoint actual native LBS hand regions required")
    if not np.array_equal(native.array(data["camera_K"], (3,3), "float64"), np.asarray(COHORT.fixed_K)):
        raise ValueError("Fixed RGB camera prior required")
    return data


def rig_metadata():
    return dict(human_faces_sha256=FACE_SHA, human_faces_hash_dtype="little-endian int32",
        human_faces_stored_dtype="int64", topology_source="actual_native_SAM3DBodyEstimator.faces",
        vertices=18439, faces=36874)


def completed_body(report):
    """Completed-call contract; this is not independent runtime attestation."""
    core.require_fields(report, {name+suffix:count for name,count in COUNTS.items() for suffix in ("_attempts","_completed")})
    core.require_fields(report, dict(actual_body_inference=True, all_artifacts_frozen_and_reloaded=True,
        SHAM_exact_replay_verified=True, original_native_replay_verified=True,
        empirical_same_process_reproducibility_only=True, hand_regions_native_verified=True,
        native_topology=rig_metadata(), execution_policy_instrumented=True,
        native_operations_modified=False, native_arguments_modified=False,
        scoped_MHR_execution="strictTrue_warnFalse_JITunoptimized", native_head_method_restored=True,
        scoped_MHR_attempts=SCOPED_HEADS, scoped_MHR_returns=SCOPED_HEADS, scoped_MHR_validated=SCOPED_HEADS,
        deterministic_algorithms=False, warn_only=False, TF32=False))
    execution.completed(report)
    if type(report.get("records")) is not list or len(report["records"]) != COHORT.count:
        raise ValueError("Every24 completed native frame required")


def error_record(errors, keys):
    if (type(errors) is not dict or set(errors) != set(keys) or any(type(v) not in (int,float)
            or not np.isfinite(v) or not 0 <= v <= 1e-5 for v in errors.values())):
        raise ValueError("Complete finite nonnegative native parity errors <=1e-5 required")


def parity_record(value, *, exact=False):
    if type(value) is not dict or set(value) != {"maximum_errors", "all_native_arrays_byte_equal"}:
        raise ValueError("Exact native replay metadata required")
    error_record(value["maximum_errors"], ("vertices_camera_m", "keypoints_camera_m", "joints_camera_m", "controls", "rotations"))
    if type(value["all_native_arrays_byte_equal"]) is not bool or (exact and not value["all_native_arrays_byte_equal"]):
        raise ValueError("Exact repeated-input native replay metadata required")


def frame_metadata(row, record):
    box = row.get("bbox")
    if (type(box) is not list or len(box) != 4 or any(type(v) not in (float,int) or not np.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= COHORT.width or not 0 <= box[1] < box[3] <= COHORT.height
            or box[2]-box[0] <= 1 or box[3]-box[1] <= 1):
        raise ValueError("Finite automatic original-grid actor bbox required")
    core.require_fields(row, dict(camera_K=[list(r) for r in COHORT.fixed_K]))
    branches = row.get("branches")
    if type(branches) is not list or len(branches) != len(core.ALL_BRANCHES):
        raise ValueError("All six ordered native branches required")
    source_hash = None
    for branch,(name,gamma) in zip(branches,core.ALL_BRANCHES):
        keys = {"name", "gamma", "source_RGB_sha256", "transformed_RGB_sha256", "native_forward_errors"}
        if name == "original":
            keys.add("anchor_native_parity")
            if record["frame_index"] == 0: keys.add("clip_first_native_parity")
        if name.startswith("sham"): keys.update(("raw_SHAM_parity", "fixed_SHAM_parity"))
        if type(branch) is not dict or set(branch) != keys: raise ValueError("Exact native branch metadata required")
        core.require_fields(branch, dict(name=name, gamma=gamma))
        for key in ("source_RGB_sha256", "transformed_RGB_sha256"):
            if type(branch[key]) is not str or not re.fullmatch("[0-9a-f]{64}",branch[key]):
                raise ValueError("Native input/transformed RGB byte identities required")
        if source_hash is None: source_hash = branch["source_RGB_sha256"]
        if branch["source_RGB_sha256"] != source_hash or (gamma == 1. and branch["transformed_RGB_sha256"] != source_hash):
            raise ValueError("Every branch must retain the original RGB and exact SHAM bytes")
        error_record(branch["native_forward_errors"], ("vertices_m", "joints_m", "keypoints_m", "controls"))
        for key in keys & {"anchor_native_parity", "clip_first_native_parity", "raw_SHAM_parity", "fixed_SHAM_parity"}:
            parity_record(branch[key], exact="SHAM" in key)
    replays = row.get("selected_replays")
    if type(replays) is not list or len(replays) != 3: raise ValueError("All three ordered selected native replays required")
    for mode,replay in zip(("baseline","sham","tta"),replays):
        if type(replay) is not dict or set(replay) != {"mode", "native_replay", "artifact"}:
            raise ValueError("Exact selected native replay metadata required")
        core.require_fields(replay,dict(mode=mode));parity_record(replay["native_replay"])
        artifact = replay["artifact"]
        if type(artifact) is not dict or artifact.get("file") != "selected_"+mode+".npz":
            raise ValueError("Selected replay mode/artifact link required")
        if sum(artifact == row_artifact for row_artifact in row["artifacts"]) != 1:
            raise ValueError("Selected replay must bind exactly one frozen artifact")


def selection_record(fixed):
    tta = core.geometric_medoid(np.stack([d["vertices_camera_m"] for d in fixed[:3]]))
    sham = core.geometric_medoid(np.stack([d["vertices_camera_m"] for d in fixed[3:]]))
    if sham.index != 0 or np.any(sham.scores) or np.any(sham.pairwise_squared_distances):
        raise ValueError("Repeated identical input must not manufacture a medoid change")
    return dict(gammas=list(core.GAMMAS), selected_gamma_index=tta.index, selected_gamma=core.GAMMAS[tta.index],
        pairwise_squared_camera_vertex_distance_m2=tta.pairwise_squared_distances.tolist(), medoid_scores_m2=tta.scores.tolist(),
        existing_native_proposal_selected=True, geometry_averaged=False), dict(index=sham.index, medoid_scores_m2=sham.scores.tolist(), native_arrays_byte_equal=True)


def frozen_frames(out, rows, records, rig_identity):
    """Reread all360 frozen proposals; selection is recomputed from public arrays."""
    if type(rows) is not list or len(rows) != COHORT.count or len(records) != COHORT.count: raise ValueError("Every24 frozen prediction required")
    rig_path = out/"rig_metadata.npz"
    if core.identity(rig_path) != rig_identity: raise ValueError("Frozen native rig metadata changed")
    with np.load(rig_path, allow_pickle=False) as saved: validate_rig({k:saved[k] for k in saved.files})
    clip_first = {};result = []
    names = [prefix+"_"+name+".npz" for prefix in ("raw","fixed") for name,_ in core.ALL_BRANCHES]+["selected_"+m+".npz" for m in ("baseline","sham","tta")]
    for i,(record,row) in enumerate(zip(records,rows)):
        core.require_fields(record, dict(file=COHORT.frame_name(i), group_index=i//COHORT.frames, frame_index=i%COHORT.frames))
        core.require_fields(row, {k:record[k] for k in ("file","group_index","frame_index")} | dict(rgb_sha256=record["sha256"],
            human_mask_sha256=record["human_mask_sha256"], clip_anchor_file=COHORT.frame_name(i//COHORT.frames*COHORT.frames)))
        folder = out/Path(record["file"]).stem
        if folder.resolve()!=folder.absolute() or not folder.is_dir() or any(p.is_symlink() for p in (folder,*folder.parents)):
            raise ValueError("Canonical complete frame artifact directory required")
        artifact_rows = row.get("artifacts")
        if type(artifact_rows) is not list or len(artifact_rows)!=15:
            raise ValueError("Exact six raw/six fixed/three selected native artifacts required")
        if any(type(r) is not dict or set(r) != {"file", "identity"} or type(r["file"]) is not str for r in artifact_rows):
            raise ValueError("Exact native artifact identity metadata required")
        if {r["file"] for r in artifact_rows} != set(names):raise ValueError("Exact native artifact names required")
        frame_metadata(row, record)
        decoded = {}
        for artifact in artifact_rows:
            path = folder/artifact["file"]
            if core.identity(path)!=artifact.get("identity"): raise ValueError("Frozen native frame arrays changed")
            with np.load(path,allow_pickle=False) as saved: decoded[artifact["file"]]={k:saved[k] for k in saved.files}
        original = core.validate_proposal(decoded["raw_original.npz"])
        if record["frame_index"] == 0: clip_first[record["group_index"]] = original
        first = clip_first[record["group_index"]];anchor = core.validate_proposal(decoded["fixed_original.npz"])
        for k in ("shape_params","scale_params"):
            if anchor[k].tobytes()!=first[k].tobytes(): raise ValueError("Shared clip identity differs")
        for k in ("global_rot","pred_cam_t","hand_pose_params","expr_params","body_pose_params"):
            if anchor[k].tobytes()!=original[k].tobytes(): raise ValueError("Frame-original nonidentity input differs")
        if (anchor["mhr_model_params"][np.r_[0:6,68:122]].tobytes()!=original["mhr_model_params"][np.r_[0:6,68:122]].tobytes()
                or anchor["mhr_model_params"][136:].tobytes()!=first["mhr_model_params"][136:].tobytes()):
            raise ValueError("Frame root/hands or clip expanded scales differ")
        fixed = []
        for name,_ in core.ALL_BRANCHES:
            raw = core.validate_proposal(decoded["raw_"+name+".npz"]);candidate = core.validate_proposal(decoded["fixed_"+name+".npz"],anchor)
            if candidate["body_pose_params"].tobytes()!=raw["body_pose_params"].tobytes(): raise ValueError("Existing native BODY proposal changed")
            if name.startswith("sham"):
                core.parity(raw,original,exact=True);core.parity(candidate,anchor,exact=True)
            fixed.append(candidate)
        selection, sham = selection_record(fixed)
        core.require_fields(row, dict(selection=selection, SHAM=sham))
        selected = [fixed[0],fixed[3],fixed[selection["selected_gamma_index"]]]
        for mode,candidate in zip(("baseline","sham","tta"),selected):
            core.parity(decoded["selected_"+mode+".npz"],candidate,exact=True)
        if {p.name for p in folder.iterdir()} != set(names): raise ValueError("Exact native frame artifact inventory required")
        result.append(dict(raw=original, baseline=fixed[0], sham=fixed[3], tta=selected[2]))
    if {p.name for p in out.iterdir()} != {"report.json","rig_metadata.npz",*[Path(r["file"]).stem for r in records]}:
        raise ValueError("Exact24 native frame directories and metadata required")
    return result


def run_masks(root,out,report,persist):
    records,receipt=rgb_inputs(root);assets=masks.validate_assets(root,report["image_id"])
    report.update(input_manifest_sha256=receipt["sha256"],input_manifest_bytes=receipt["bytes"],model_assets=assets,phase="automatic_detector_load");persist()
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from PIL import Image
    if not torch.cuda.is_available(): raise ValueError("Actual automatic CUDA detector required")
    dist=metadata.distribution("SAM-2");direct=dist.read_text("direct_url.json")
    sam_source=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct) if direct else {})
    report["sam2_source"]=sam_source
    processor=AutoProcessor.from_pretrained(root/"weights/grounding_dino",local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(root/"weights/grounding_dino",local_files_only=True).cuda().eval()
    try:
        for record in records:
            row={k:record[k] for k in ("file","group_index","frame_index")} | dict(rgb_sha256=record["sha256"],human_query="person.")
            report["records"].append(row);report["detector_attempts"]+=1;report["phase"]="automatic_person_detection";persist()
            inputs=processor(images=Image.fromarray(read_rgb(record,Image)),text="person.",return_tensors="pt").to("cuda")
            with torch.inference_mode(): result=processor.post_process_grounded_object_detection(detector(**inputs),inputs.input_ids,
                threshold=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,target_sizes=[(COHORT.height,COHORT.width)])[0]
            report["actual_detector_calls"]+=1
            choices=[BoxDetection(tuple(b),float(s)) for b,s in zip(result["boxes"].cpu().tolist(),result["scores"].cpu().tolist())]
            chosen=masks.select_person(choices,COHORT.width,COHORT.height)
            row.update(box_xyxy=list(chosen.box),detector_score=chosen.score,candidate_count=len(choices));persist();del inputs,result
    finally: del detector,processor;torch.cuda.empty_cache()
    predictor=SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",str(root/"weights/sam2/sam2.1_hiera_large.pt"),device="cuda",mode="eval"))
    try:
        for record,row in zip(records,report["records"]):
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                report["image_encoder_attempts"]+=1;persist();predictor.set_image(read_rgb(record,Image));report["actual_sam2_image_encoder_calls"]+=1
                report["sam2_attempts"]+=1;persist();predicted,scores,_=predictor.predict(box=np.asarray(row["box_xyxy"],np.float32),multimask_output=False);report["actual_sam2_calls"]+=1
            mask=masks.binary_mask(predicted,scores,COHORT.width,COHORT.height);path=out/(Path(record["file"]).stem+"_human.png")
            with path.open("xb") as stream:Image.fromarray(mask).save(stream,format="PNG")
            path.chmod(0o444);mi=protocol.identity(path);row.update(human_mask_file=path.name,human_mask_sha256=mi["sha256"],human_mask_bytes=mi["bytes"],human_mask_pixels=int(np.count_nonzero(mask)));persist()
            predictor.reset_predictor()
    finally: del predictor;torch.cuda.empty_cache()
    completed_masks(report)
    if rgb_inputs(root)!=(records,receipt) or masks.validate_assets(root,report["image_id"])!=assets:
        raise ValueError("Public RGB/mask assets changed")
    if sam_source!=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct) if direct else {}):raise ValueError("Installed SAM2 source changed")
    for row in report["records"]:
        if protocol.identity(out/row["human_mask_file"])!=dict(sha256=row["human_mask_sha256"],bytes=row["human_mask_bytes"]):raise ValueError("Automatic mask bytes changed")
    if {p.name for p in out.iterdir()}!={"report.json",*[r["human_mask_file"] for r in report["records"]]}:raise ValueError("Exact automatic mask output required")
    report["actual_automatic_inference_verified"]=True


def runtime_guard(torch):
    if (torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("Empirical native runtime false/noWarn/TF32off required")


def run_body(root,out,report,persist,revision):
    with ExitStack() as stack:
        def load(root,torch):
            result=native.human.load_model(root,torch)
            stack.enter_context(execution.strict_native_head(torch,result[0].head_pose,report,persist))
            return result
        run_body_native(root,out,report,persist,revision,load)
        execution.completed(report)
        core.require_fields(report,dict(scoped_MHR_attempts=SCOPED_HEADS,scoped_MHR_returns=SCOPED_HEADS,scoped_MHR_validated=SCOPED_HEADS))
    completed_body(report)


def run_body_native(root,out,report,persist,revision,load):
    records,inputs=public_masks(root,revision);report.update(public_inputs=inputs,phase="native_model_load");persist()
    if "torch" in sys.modules:raise ValueError("CUBLAS configuration precedes Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4":raise ValueError("Pinned native CUDA runtime required")
    core.reset_seed(torch);torch.set_num_threads(4);torch.use_deterministic_algorithms(False,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;runtime_guard(torch)
    semantic_path=root/"results/mhr-finger-semantics-v4.json";semantic_id=native.regular(semantic_path)
    semantic=json.loads(semantic_path.read_text());native.regions_helper.require_semantic_report(semantic)
    if semantic.get("source_image_id")!=core.IMAGE:raise ValueError("Native semantic image differs")
    model,estimator,faces,source=load(root,torch);model.eval();model.requires_grad_(False)
    if model.head_pose.enable_hand_model is not False:raise ValueError("Body-only unmodified native head required")
    core.require_fields(source.get("checkpoint_loading"),dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",parameter_tensors_loaded=1101,unexpected_keys=[]))
    retained=source["checkpoint_loading"].get("retained_mhr_asset_buffer_names")
    if type(retained) is not list or len(retained)!=113 or len(set(retained))!=113:raise ValueError("Exact113 immutable MHR buffers required")
    joints=model.head_pose.mhr.get_joint_names();lbs=model.head_pose.mhr.get_lbsw()
    if joints!=semantic["joint_names"] or not isinstance(lbs,tuple) or len(lbs)!=2:raise ValueError("Actual native LBS semantic API differs")
    regions=native.regions_helper.lbs_regions(*(v.detach().cpu().numpy() for v in lbs),joints)
    rig=validate_rig(dict(human_faces=faces,hand_mask_left=regions["l"]["vertex_mask"],hand_mask_right=regions["r"]["vertex_mask"],camera_K=np.asarray(COHORT.fixed_K,np.float64)))
    with (out/"rig_metadata.npz").open("xb") as stream:np.savez_compressed(stream,**rig)
    (out/"rig_metadata.npz").chmod(0o444)
    report.update(rig_metadata=core.identity(out/"rig_metadata.npz"),native_topology=rig_metadata(),body_model=source,semantic_report_sha256=semantic_id["sha256"],joint_names=joints,
        hand_regions_source="actual_native_head_MHR_get_lbsw_and_joint_names",hand_region_sha256={side:hashlib.sha256(rig["hand_mask_"+side].tobytes()).hexdigest() for side in ("left","right")},
        seed=0,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,TF32=False,deterministic_algorithms=False,warn_only=False,CUBLAS_WORKSPACE_CONFIG=":4096:8")
    camera=torch.as_tensor(np.asarray(COHORT.fixed_K,np.float32),device="cuda")[None];clip_first={}
    for record in records:
        report.update(active_file=record["file"],phase="native_photometric_predictions");persist()
        rgb=read_rgb(record,Image);human=native.joint.read_mask(record["human_mask_path"],Image)
        if int(np.count_nonzero(human))!=record["human_mask_pixels"]:raise ValueError("Automatic person area differs")
        box,prompt=native.human.derived_bbox(rgb,human);folder=out/Path(record["file"]).stem;folder.mkdir()
        row={k:record[k] for k in ("file","group_index","frame_index")} | dict(rgb_sha256=record["sha256"],human_mask_sha256=record["human_mask_sha256"],
            bbox=box.tolist(),camera_K=[list(r) for r in COHORT.fixed_K],clip_anchor_file=COHORT.frame_name(record["group_index"]*COHORT.frames),branches=[],selected_replays=[],artifacts=[])
        report["records"].append(row);raw=[];fixed=[];anchor=None
        for name,gamma in core.ALL_BRANCHES:
            runtime_guard(torch);variant=core.gamma_rgb(rgb,gamma)
            branch=dict(name=name,gamma=gamma,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),transformed_RGB_sha256=hashlib.sha256(variant.tobytes()).hexdigest())
            if gamma==1. and rgb.tobytes()!=variant.tobytes():raise ValueError("Original/SHAM RGB must be exact")
            core.reset_seed(torch);report["body_attempts"]+=1;row["active_branch"]=name;report["active_branch"]=name;persist()
            with torch.inference_mode(): predictions=estimator.process_one_image(img=variant,bboxes=box[None],masks=prompt,cam_int=camera,inference_type="body")
            report["body_completed"]+=1;persist()
            if type(predictions) is not list or len(predictions)!=1:raise ValueError("One automatic native actor required")
            report["parity_head_attempts"]+=1;persist();values,errors=native.human.decode_prediction(torch,model,predictions[0],"body");report["parity_head_completed"]+=1
            report["keypoint_head_attempts"]+=1;persist();kp,rotations=baseline.raw_keypoint_decode(torch,model.head_pose,predictions[0],values);report["keypoint_head_completed"]+=1
            if not np.isclose(float(values["focal_length"]),COHORT.prior_focal,atol=1e-4,rtol=1e-6):raise ValueError("Native model ignored the fixed public camera prior")
            original={k:values[k] for k in native.BLOCKS} | dict(vertices_camera_m=values["vertices_camera_m"],keypoints_camera_m=kp,joints_camera_m=values["joints_camera_m"],joint_global_rotations=rotations)
            core.validate_proposal(original);raw.append(original);row["artifacts"].append(core.save_proposal(folder,"raw_"+name,original))
            if anchor is None:
                if record["frame_index"]==0:clip_first[record["group_index"]]=original
                report["anchor_head_attempts"]+=1;persist();anchor=frame_anchor(torch,model.head_pose,original,clip_first[record["group_index"]]);report["anchor_head_completed"]+=1
                if record["frame_index"]==0:branch["clip_first_native_parity"]=core.parity(anchor,original)
            report["fixed_head_attempts"]+=1;persist();candidate=core.fixed_decode(torch,model.head_pose,original,anchor);report["fixed_head_completed"]+=1
            fixed.append(candidate);row["artifacts"].append(core.save_proposal(folder,"fixed_"+name,candidate));branch["native_forward_errors"]=errors
            if name=="original":branch["anchor_native_parity"]=core.parity(candidate,anchor)
            if name.startswith("sham"):
                branch["raw_SHAM_parity"]=core.parity(original,raw[0],exact=True);branch["fixed_SHAM_parity"]=core.parity(candidate,fixed[0],exact=True)
            row["branches"].append(branch);persist();del predictions,values
        row["selection"],row["SHAM"]=selection_record(fixed)
        selected=(fixed[0],fixed[3],fixed[row["selection"]["selected_gamma_index"]])
        for mode,candidate in zip(("baseline","sham","tta"),selected):
            runtime_guard(torch);report["selected_replay_attempts"]+=1;persist();replay=core.fixed_decode(torch,model.head_pose,candidate,anchor);report["selected_replay_completed"]+=1
            row["selected_replays"].append(dict(mode=mode,native_replay=core.parity(replay,candidate),artifact=core.save_proposal(folder,"selected_"+mode,candidate)))
            row["artifacts"].append(row["selected_replays"][-1]["artifact"]);persist()
        row.pop("active_branch",None)
        report.pop("active_branch",None)
    frozen_frames(out,report["records"],records,report["rig_metadata"])
    if public_masks(root,revision)!=(records,inputs) or native.regular(semantic_path)!=semantic_id:raise ValueError("Original public masks/RGB/semantic changed")
    if native.human.body._source_identity(root)!=source["inference_source_identity"] or native.human.body._body_assets(root)[1]!=source["body_assets"]:raise ValueError("Native source/assets changed")
    report.update(actual_body_inference=True,all_artifacts_frozen_and_reloaded=True,SHAM_exact_replay_verified=True,
        original_native_replay_verified=True,empirical_same_process_reproducibility_only=True,hand_regions_native_verified=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument("--stage",choices=tuple(STAGES),required=True);args=parser.parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));revision=os.environ.get("WR_CODE_REVISION","");image=os.environ.get("WR_IMAGE_ID","");out=root/BASE/FOLDERS[args.stage]
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or not re.fullmatch("[0-9a-f]{40}",revision)
            or image!=(core.MASK_IMAGE if args.stage=="masks" else core.IMAGE) or not out.is_dir() or any(out.iterdir())
            or out.resolve()!=out.absolute() or any(p.is_symlink() for p in (out,*out.parents))):raise ValueError("Fresh canonical offline H101 observer required")
    report=dict(stage=STAGES[args.stage],status="fail",phase="public_integrity",cohort=asdict(COHORT),frames=COHORT.count,producer_revision=revision,
        image_id=image,script_sha256=sha256(Path(__file__)),source_helpers=helper_hashes(),network="none",budget_seconds=BUDGETS[args.stage],
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,fitting_performed=False,camera_fit_performed=False,
        geometry_averaged=False,all_cases_retained=False,source_inputs_assets_rehashed=False,human_query="person.",records=[],
        execution_policy_instrumented=args.stage=="body", native_operations_modified=False,native_arguments_modified=False,
        cross_process_determinism_verified=False,all_frame_determinism_verified=False,exact_SHAM_gate_relaxed=False,
        confidence=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,nms_iou=masks.NMS_IOU,ambiguity_margin=masks.AMBIGUITY_MARGIN,
        detector_attempts=0,actual_detector_calls=0,image_encoder_attempts=0,actual_sam2_image_encoder_calls=0,sam2_attempts=0,actual_sam2_calls=0)
    report.update({name+suffix:0 for name in COUNTS for suffix in ("_attempts","_completed")})
    started=time.perf_counter();path=out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("H101 observer exceeded its frozen whole-stage budget")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGETS[args.stage])
        try:
            persist()
            if args.stage=="masks":run_masks(root,out,report,persist)
            else:run_body(root,out,report,persist,revision)
            if helper_hashes()!=report["source_helpers"]:raise ValueError("Complete own native source closure changed")
            report.update(status="pass",phase="complete",all_cases_retained=True,source_inputs_assets_rehashed=True);report.pop("active_file",None)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()

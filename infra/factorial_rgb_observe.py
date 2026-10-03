"""H99 public-only factorial observations: automatic masks, framewise Body, DWPose.

The groups are opaque image identities. No manufacturing factor, private camera,
depth, mesh or label is read; no fitting, shared-identity selection or quality claim.
"""
import argparse
from dataclasses import asdict
import gc
from importlib import metadata, util
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

import numpy as np
import factorial_rgb_protocol as protocol
import hand_synthetic_masks as masks
import keypoint_rgb_baseline as body
import keypoint_rgb_dwpose as dw_helper
from world_reward.data import sha256
from world_reward.prompt_selection import BoxDetection

native, smoke = body.native, dw_helper.smoke
COHORT = protocol.COHORT
BASE = COHORT.base
STAGES = {"masks": "public_factorial_rgb_automatic_masks", "body": "public_factorial_rgb_framewise_body_observations",
          "dwpose": "public_factorial_rgb_native_dwpose133_observations"}
FOLDERS = {"masks": "automatic_masks", "body": "body_v1", "dwpose": "dwpose_v1"}
BUDGETS = {"masks": 180, "body": 300, "dwpose": 180}
QUERIES = (("human", "person."), ("object", "bottle."))
PIN_SCHEMA = "world-reward-factorial-public-pins-v1"
BODY_KEYS = frozenset(native.BLOCKS) | {"vertices_camera_m", "keypoints_camera_m", "joints_camera_m", "joint_global_rotations",
    "human_faces", "camera_K", "bbox", "group_index", "frame_index"}
DW_KEYS = {"keypoints", "scores", "validity", "bbox", "group_index", "frame_index"}


def require_fields(row, expected):
    if not isinstance(row, dict) or any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
        raise ValueError("Exact public factorial producer contract required")


def cohort_contract(cohort):
    if not isinstance(cohort, protocol.PublicCohort) or (cohort.groups, cohort.frames, cohort.count, cohort.width, cohort.height) != (8, 3, 24, 1024, 768):
        raise ValueError("Exact preregistered factorial cohort required")
    if not np.array_equal(np.asarray(cohort.fixed_K), native.CAMERA_K): raise ValueError("Fixed RGB camera prior1280 required")
    return asdict(cohort)


def source_identity():
    return {"observer": sha256(Path(__file__)), "protocol": sha256(Path(protocol.__file__)), "mask_helper": sha256(Path(masks.__file__)),
            "body_helper": sha256(Path(body.__file__)), "dwpose_helper": sha256(Path(dw_helper.__file__)),
            "body_helpers": body.helper_identities(), "dwpose_runtime": smoke.source_identity()}


def validate_pins(value, stage):
    expected = {"schema", "manifest_sha256", "manifest_bytes"} | ({"automatic_masks"} if stage != "masks" else set())
    if not isinstance(value, dict) or set(value) != expected or value.get("schema") != PIN_SCHEMA:
        raise ValueError("Explicit immutable stage-specific public pins required")
    if type(value["manifest_sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", value["manifest_sha256"]) or type(value["manifest_bytes"]) is not int or value["manifest_bytes"] <= 0:
        raise ValueError("Exact original manifest identity required")
    if stage != "masks":
        row = value["automatic_masks"]
        if not isinstance(row, dict) or set(row) != {"producer_revision", "report_sha256", "report_bytes", "script_sha256"}:
            raise ValueError("Exact completed mask/source pins required")
        for k, length in (("producer_revision", 40), ("report_sha256", 64), ("script_sha256", 64)):
            if type(row[k]) is not str or not re.fullmatch("[0-9a-f]{"+str(length)+"}", row[k]): raise ValueError("Canonical mask producer pin required")
        if type(row["report_bytes"]) is not int or row["report_bytes"] <= 0: raise ValueError("Positive mask receipt size required")
    return value


def rgb_inputs(root, pins, cohort=COHORT):
    records, receipt = protocol.public_inputs(Path(root)/cohort.base/"inputs", cohort)
    if receipt != dict(sha256=pins["manifest_sha256"], bytes=pins["manifest_bytes"]): raise ValueError("Pinned original RGB manifest changed")
    return records, receipt


def completed_masks(report, cohort=COHORT):
    require_fields(report, dict(actual_detector_calls=2*cohort.count, actual_sam2_calls=2*cohort.count, actual_sam2_image_encoder_calls=cohort.count))
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != cohort.count: raise ValueError("Every ordered automatic mask required")
    for index, row in enumerate(rows):
        require_fields(row, dict(file=cohort.frame_name(index), group_index=index//cohort.frames, frame_index=index%cohort.frames))
        for label, query in QUERIES:
            require_fields(row, {label+"_query": query, label+"_mask_file": Path(cohort.frame_name(index)).stem+"_"+label+".png"})
            if (not re.fullmatch("[0-9a-f]{64}", str(row.get(label+"_mask_sha256", ""))) or type(row.get(label+"_mask_bytes")) is not int or row[label+"_mask_bytes"] <= 0
                    or type(row.get(label+"_mask_pixels")) is not int or not 0 < row[label+"_mask_pixels"] <= cohort.width*cohort.height):
                raise ValueError("Complete nonempty automatic mask evidence required")


def public_inputs(root, pins, labels=("human", "object"), cohort=COHORT):
    if labels not in (("human",), ("human", "object")): raise ValueError("Explicit automatic mask modalities required")
    validate_pins(pins, "body");records, manifest = rgb_inputs(root, pins, cohort);base = Path(root)/cohort.base
    path = base/"automatic_masks/report.json";pin = pins["automatic_masks"];receipt = protocol.identity(path)
    if receipt != dict(sha256=pin["report_sha256"], bytes=pin["report_bytes"]): raise ValueError("Pinned automatic mask receipt changed")
    historical = Path(root)/"jobs"/pin["producer_revision"]/"run_factorial_rgb_observe"/"code"/"infra"/"factorial_rgb_observe.py"
    historical_id = protocol.identity(historical)
    if historical_id["sha256"] != pin["script_sha256"]: raise ValueError("Exact source-only mask producer required")
    report = json.loads(path.read_text());helpers = source_identity() | {"observer": pin["script_sha256"]}
    require_fields(report, dict(stage=STAGES["masks"], status="pass", phase="complete", cohort=asdict(cohort), frames=cohort.count,
        script_sha256=pin["script_sha256"], source_helpers=helpers, producer_revision=pin["producer_revision"], network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], human_query="person.", object_query="bottle.",
        input_manifest_sha256=manifest["sha256"], input_manifest_bytes=manifest["bytes"], all_cases_retained=True, actual_automatic_inference_verified=True,
        confidence=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, nms_iou=masks.NMS_IOU, ambiguity_margin=masks.AMBIGUITY_MARGIN))
    if any(k in report for k in ("error", "error_type")): raise ValueError("Passing masks cannot retain an error")
    completed_masks(report, cohort)
    assets = report.get("model_assets")
    if not isinstance(assets, dict) or set(assets) != set(masks.ASSETS): raise ValueError("Complete pinned detector/SAM assets required")
    for name, (digest, size) in masks.ASSETS.items(): require_fields(assets[name], dict(path=str(Path(root)/"weights"/name), sha256=digest, bytes=size))
    for record, row in zip(records, report["records"]):
        require_fields(row, dict(rgb_sha256=record["sha256"]))
        for label in labels:
            target = path.parent/row[label+"_mask_file"];identity = protocol.identity(target)
            if identity != dict(sha256=row[label+"_mask_sha256"], bytes=row[label+"_mask_bytes"]): raise ValueError("Original mask PNG changed")
            record[label+"_mask_path"] = target;record[label+"_mask_pixels"] = row[label+"_mask_pixels"]
    if labels == ("human", "object") and {p.name for p in path.parent.iterdir()} != {"report.json", *[r[k+"_mask_path"].name for r in records for k in labels]}:
        raise ValueError("Only exact automatic masks may be exposed")
    return records, dict(manifest=manifest, automatic_masks=receipt, mask_source=historical_id)


def read_rgb(record, Image, cohort=COHORT):
    with Image.open(record["path"]) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (cohort.width, cohort.height): raise ValueError("Original RGB decode required")
        return np.ascontiguousarray(np.asarray(image))


def read_masks(record, Image, labels=("human", "object")):
    result = tuple(native.joint.read_mask(record[k+"_mask_path"], Image) for k in labels)
    if any(int(np.count_nonzero(value)) != record[k+"_mask_pixels"] for k, value in zip(labels, result)):
        raise ValueError("Automatic mask area differs")
    return result


def scalar_indices(data, record):
    for k in ("group_index", "frame_index"):
        if int(native.array(data[k], (), "int64")) != record[k]: raise ValueError("Original factorial group/frame changed")


def validate_body(data, record):
    if set(data) != BODY_KEYS: raise ValueError("Exact compact native Body schema required")
    scalar_indices(data, record)
    for k, shape in native.BLOCKS.items(): native.array(data[k], shape, "float32")
    if (np.any(data["expr_params"]) or np.any(data["mhr_model_params"][:3]) or data["pred_cam_t"][2] <= 0
            or data["mhr_model_params"][3:6].tobytes() != data["global_rot"].tobytes()):
        raise ValueError("Native relative-root/zero-expression/positive-camera required")
    for k, count in (("vertices_camera_m", native.VERTICES), ("keypoints_camera_m", 308), ("joints_camera_m", 127)):
        if np.any(native.array(data[k], (count, 3), "float32")[:, 2] <= 0): raise ValueError("Every native camera point must remain positiveZ")
    body.rotations(data["joint_global_rotations"])
    faces = native.array(data["human_faces"], (36874, 3), "int64")
    if np.any(faces < 0) or np.any(faces >= native.VERTICES): raise ValueError("Actual complete native topology required")
    if not np.array_equal(native.array(data["camera_K"], (3, 3), "float64"), native.CAMERA_K): raise ValueError("Same public camera prior required")
    box = native.array(data["bbox"], (1, 4), "float32")
    if np.any(box[:, :2] < 0) or np.any(box[:, 2:] > [COHORT.width, COHORT.height]) or np.any(box[:, 2:] <= box[:, :2]): raise ValueError("Automatic bounded actor bbox required")
    return data


def validate_dw(data, record):
    if set(data) != DW_KEYS: raise ValueError("Exact low-level native133 schema required")
    scalar_indices(data, record);native.array(data["keypoints"], (1, 133, 2), "float64")
    valid = smoke.validate_prediction(data["keypoints"], data["scores"])
    if not np.array_equal(native.array(data["validity"], (1, 133), "bool"), valid): raise ValueError("Native positive-score validity differs")
    box = native.array(data["bbox"], (1, 4), "float32")
    if np.any(box[:, :2] < 0) or np.any(box[:, 2:] > [COHORT.width, COHORT.height]) or np.any(box[:, 2:] <= box[:, :2]): raise ValueError("Automatic bounded actor bbox required")
    return data


def save(out, record, data):
    path = out/(Path(record["file"]).stem+".npz")
    with path.open("xb") as stream: np.savez_compressed(stream, **data)
    path.chmod(0o444)
    return {k: record[k] for k in ("file", "group_index", "frame_index")} | dict(prediction_file=path.name, rgb_sha256=record["sha256"], prediction=native.regular(path, immutable=True))


def frozen_artifacts(out, rows, records, validator):
    if not isinstance(rows, list) or len(rows) != COHORT.count or len(records) != COHORT.count:
        raise ValueError("Every24 original observation required")
    values = []
    for index, (row, record) in enumerate(zip(rows, records)):
        require_fields(row, {k: record[k] for k in ("file", "group_index", "frame_index")} | dict(prediction_file=Path(record["file"]).stem+".npz", rgb_sha256=record["sha256"]))
        require_fields(record, dict(file=COHORT.frame_name(index), group_index=index//COHORT.frames, frame_index=index%COHORT.frames))
        path = out/row["prediction_file"]
        if native.regular(path, immutable=True) != row["prediction"]: raise ValueError("Frozen observation bytes changed")
        with np.load(path, allow_pickle=False) as saved: data = {k: saved[k] for k in saved.files}
        values.append(validator(data, record))
    if {p.name for p in out.iterdir()} != {"report.json", *[r["prediction_file"] for r in rows]}: raise ValueError("Exact24 predictions/receipt required")
    return values


def run_masks(root, out, report, persist, pins):
    records, receipt = rgb_inputs(root, pins);assets = masks.validate_assets(root, report["image_id"])
    report.update(input_manifest_sha256=receipt["sha256"], input_manifest_bytes=receipt["bytes"], model_assets=assets, phase="detector_load");persist()
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from PIL import Image
    if not torch.cuda.is_available(): raise ValueError("Actual pinned CUDA mask inference required")
    dist = metadata.distribution("SAM-2");direct = dist.read_text("direct_url.json")
    sam_source = masks.sam2_source_identity(Path(sam2.__file__).parent, json.loads(direct) if direct else {})
    report.update(sam2_source=sam_source, torch_version=str(torch.__version__));persist()
    processor = AutoProcessor.from_pretrained(root/"weights/grounding_dino", local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(root/"weights/grounding_dino", local_files_only=True).cuda().eval()
    try:
        for item in records:
            row = {k: item[k] for k in ("file", "group_index", "frame_index")} | dict(rgb_sha256=item["sha256"]);report["records"].append(row)
            rgb = Image.fromarray(read_rgb(item, Image));report["phase"] = "automatic_detection"
            for label, query in QUERIES:
                report["detector_attempts"] += 1;persist()
                entry = processor(images=rgb, text=query, return_tensors="pt").to("cuda")
                with torch.inference_mode(): result = processor.post_process_grounded_object_detection(detector(**entry), entry.input_ids,
                    threshold=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, target_sizes=[(COHORT.height, COHORT.width)])[0]
                report["actual_detector_calls"] += 1
                boxes = [BoxDetection(tuple(b), float(s)) for b, s in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist())]
                chosen = masks.select_person(boxes, COHORT.width, COHORT.height)
                row.update({label+"_query": query, label+"_box": list(chosen.box), label+"_detector_score": chosen.score, label+"_candidate_count": len(boxes)})
                del entry, result;persist()
    finally: del detector, processor;torch.cuda.empty_cache()
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(root/"weights/sam2/sam2.1_hiera_large.pt"), device="cuda", mode="eval"))
    try:
        for item, row in zip(records, report["records"]):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                report["image_encoder_attempts"] += 1;persist();predictor.set_image(read_rgb(item, Image));report["actual_sam2_image_encoder_calls"] += 1
                for label, _ in QUERIES:
                    report["sam2_attempts"] += 1;persist();predicted, scores, _ = predictor.predict(box=np.array(row[label+"_box"], np.float32), multimask_output=False)
                    report["actual_sam2_calls"] += 1;mask = masks.binary_mask(predicted, scores, COHORT.width, COHORT.height)
                    filename = Path(item["file"]).stem+"_"+label+".png";target = out/filename
                    with target.open("xb") as stream: Image.fromarray(mask, mode="L").save(stream, format="PNG")
                    target.chmod(0o444);identity = protocol.identity(target)
                    row.update({label+"_mask_file": filename, label+"_mask_sha256": identity["sha256"], label+"_mask_bytes": identity["bytes"],
                                label+"_mask_pixels": int(np.count_nonzero(mask)), label+"_sam2_predicted_score": float(scores[0])});persist()
                predictor.reset_predictor()
    finally: del predictor;torch.cuda.empty_cache()
    completed_masks(report)
    if (records, receipt) != rgb_inputs(root, pins) or assets != masks.validate_assets(root, report["image_id"]): raise ValueError("RGB/model bytes changed")
    if sam_source != masks.sam2_source_identity(Path(sam2.__file__).parent, json.loads(direct) if direct else {}): raise ValueError("SAM source changed")
    for row in report["records"]:
        for label, _ in QUERIES:
            if protocol.identity(out/row[label+"_mask_file"]) != dict(sha256=row[label+"_mask_sha256"], bytes=row[label+"_mask_bytes"]): raise ValueError("Automatic PNG changed")
    if {p.name for p in out.iterdir()} != {"report.json", *[r[k+"_mask_file"] for r in report["records"] for k, _ in QUERIES]}: raise ValueError("Exact automatic mask inventory required")
    report["actual_automatic_inference_verified"] = True


def run_body(root, out, report, persist, pins):
    records, inputs = public_inputs(root, pins);semantic_path = root/"results/mhr-finger-semantics-v4.json";semantic_id = native.regular(semantic_path)
    semantic = json.loads(semantic_path.read_text());native.regions_helper.require_semantic_report(semantic)
    if semantic.get("source_image_id") != native.IMAGE_ID: raise ValueError("Pinned native semantic image required")
    report.update(public_inputs=inputs, semantic_report_sha256=semantic_id["sha256"], phase="native_model_load");persist()
    if "torch" in sys.modules: raise ValueError("CUBLAS setup precedes Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4": raise ValueError("Pinned native CUDA runtime required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model, estimator, faces, source = native.human.load_model(root, torch);model.eval();model.requires_grad_(False)
    if model.head_pose.enable_hand_model is not False: raise ValueError("Body-only native decoder required")
    camera = torch.tensor(np.asarray(COHORT.fixed_K, np.float32), device="cuda")[None]
    report.update(body_model=source, torch_version=str(torch.__version__), CUDA_version=torch.version.cuda, seed=0, TF32=False, CUBLAS_WORKSPACE_CONFIG=":4096:8");persist()
    for record in records:
        report.update(phase="framewise_body_inference", active_file=record["file"]);persist();rgb = read_rgb(record, Image);human, _ = read_masks(record, Image)
        box, prompt = native.human.derived_bbox(rgb, human);report["body_attempts"] += 1;persist()
        with torch.inference_mode(): predictions = estimator.process_one_image(img=rgb, bboxes=box[None], masks=prompt, cam_int=camera, inference_type="body")
        report["body_calls_completed"] += 1;persist()
        if not isinstance(predictions, list) or len(predictions) != 1: raise ValueError("Exactly one automatically selected native actor required")
        report["parity_head_attempts"] += 1;persist();values, errors = native.human.decode_prediction(torch, model, predictions[0], "body")
        report["parity_heads_completed"] += 1;report["keypoint_head_attempts"] += 1;persist()
        keypoints, rotations = body.raw_keypoint_decode(torch, model.head_pose, predictions[0], values);report["keypoint_heads_completed"] += 1
        if not np.isclose(float(values["focal_length"]), COHORT.prior_focal, atol=1e-4, rtol=1e-6): raise ValueError("Body ignored the fixed public camera prior")
        data = {k: values[k] for k in native.BLOCKS} | dict(vertices_camera_m=values["vertices_camera_m"], keypoints_camera_m=keypoints,
            joints_camera_m=values["joints_camera_m"], joint_global_rotations=rotations, human_faces=faces, camera_K=np.asarray(COHORT.fixed_K, np.float64),
            bbox=box[None].astype(np.float32), group_index=np.array(record["group_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64))
        validate_body(data, record);row = save(out, record, data);row.update(native_forward_errors=errors, bbox_xyxy=box.tolist(),
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());report["records"].append(row);persist();del predictions, values, data
    frozen_artifacts(out, report["records"], records, validate_body)
    if source["inference_source_identity"] != native.human.body._source_identity(root) or source["body_assets"] != native.human.body._body_assets(root)[1]: raise ValueError("Native model/source changed")
    if semantic_id != native.regular(semantic_path) or (records, inputs) != public_inputs(root, pins): raise ValueError("Public inputs/semantic receipt changed")
    if any(report[k] != COHORT.count for k in ("body_attempts", "body_calls_completed", "parity_head_attempts", "parity_heads_completed", "keypoint_head_attempts", "keypoint_heads_completed")):
        raise ValueError("All24 framewise Body/two native replay calls required")
    report.update(actual_body_inference=True, native_arrays_verified=True, body_inference_type="body", camera_K=[list(r) for r in COHORT.fixed_K],
        identity_policy="ordinary framewise prediction; no shared identity/adoption", keypoint_count=308, joint_count=127)


def run_dwpose(root, out, report, persist, pins, started):
    records, inputs = public_inputs(root, pins, ("human",));evidence = dw_helper.validate_smoke(root);assets = smoke.validate_assets(root)
    report.update(public_inputs=inputs, capability_evidence=evidence, assets=assets, dependencies=smoke.dependency_identity(), phase="private_prefix_install");persist()
    if any(name in sys.modules or util.find_spec(name) is not None for name in ("onnxruntime", "flatbuffers")): raise ValueError("Fresh disposable native CPU runtime required")
    from PIL import Image
    with smoke.private_prefix(report, persist) as prefix:
        command = smoke.pip_argv(prefix, root);report["pip_argv"] = command;persist()
        subprocess.run(command, check=True, timeout=max(.001, BUDGETS["dwpose"]-(time.perf_counter()-started)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report["private_prefix_packages_installed"] = True;ort, origins = smoke.import_runtime(prefix);report.update(runtime_imports=origins, phase="native_source_load");persist()
        source = root/smoke.acquisition.BASE/"source/onnxpose.py";spec = util.spec_from_file_location("world_reward_factorial_native_onnxpose", source)
        implementation = util.module_from_spec(spec);spec.loader.exec_module(implementation)
        options = ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(root/smoke.acquisition.BASE/smoke.acquisition.ASSETS[0][0]), sess_options=options, providers=["CPUExecutionProvider"])
        session.disable_fallback();graph = smoke.session_metadata(session);report.update(actual_sessions=1, graph=graph);persist()
        smoke.validate_session(session, graph=graph);smoke.validate_options(session, ort);proxy = smoke.SessionProxy(session, report["calls"], persist)
        for record in records:
            report.update(phase="native_rgb_inference", active_file=record["file"]);persist();rgb = read_rgb(record, Image);human, = read_masks(record, Image, ("human",))
            box = smoke.actor_bbox(human);points, scores = implementation.inference_pose(proxy, box.copy(), rgb)
            data = dict(keypoints=points, scores=scores, validity=smoke.validate_prediction(points, scores), bbox=box,
                group_index=np.array(record["group_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64))
            validate_dw(data, record);row = save(out, record, data);row.update(raw_simcc=report["calls"][-1]["raw_simcc"], positive_score_count=int(data["validity"].sum()),
                **{k: smoke.array_identity(data[k]) for k in ("keypoints", "scores", "validity", "bbox")});report["records"].append(row);persist()
        del proxy, session;gc.collect();frozen_artifacts(out, report["records"], records, validate_dw)
        if report["actual_sessions"] != 1 or len(report["calls"]) != COHORT.count or any(not r["run_completed"] for r in report["calls"]): raise ValueError("One CPU session/all24 native observations required")
        if evidence != dw_helper.validate_smoke(root) or assets != smoke.validate_assets(root) or (records, inputs) != public_inputs(root, pins, ("human",)):
            raise ValueError("Public inputs/native assets/capability changed")
        report["native_cpu_abi_verified"] = True
    if not report["private_prefix_removed"]: raise ValueError("Disposable runtime prefix not removed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False);parser.add_argument("stage", choices=tuple(STAGES));parser.add_argument("--public-pins", type=Path, required=True);args=parser.parse_args(argv)
    cohort_contract(COHORT);root = Path(os.environ.get("WR_ROOT", ""));code = Path(os.environ.get("WR_CODE", ""));revision = os.environ.get("WR_CODE_REVISION", "");image = os.environ.get("WR_IMAGE_ID", "")
    pin_name = "factorial_rgb_manifest_pins_v1.json" if args.stage == "masks" else "factorial_rgb_public_pins_v1.json"
    if not code.is_absolute() or code.resolve()!=code.absolute() or args.public_pins != code/"configs"/pin_name: raise ValueError("Explicit immutable source-bundle public pins required")
    pin_identity = protocol.identity(args.public_pins);pins = validate_pins(json.loads(args.public_pins.read_text()), args.stage)
    out = root/BASE/FOLDERS[args.stage]
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve()!=root.absolute() or os.geteuid()!=1000
        or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or not re.fullmatch("[0-9a-f]{40}", revision) or not out.is_dir() or any(out.iterdir())
        or out.resolve()!=out.absolute() or any(p.is_symlink() for p in (out,*out.parents)) or (args.stage!="masks" and image!=native.IMAGE_ID)
        or (args.stage=="dwpose" and os.environ.get("CUDA_VISIBLE_DEVICES")!="")): raise ValueError("Fresh reserved offline public observer required")
    report = dict(stage=STAGES[args.stage], status="fail", phase="public_integrity", cohort=asdict(COHORT), frames=COHORT.count, producer_revision=revision,
        image_id=image, script_sha256=sha256(Path(__file__)), source_helpers=source_identity(), network="none", budget_seconds=BUDGETS[args.stage], public_pins_file=pin_identity,
        private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], fitting_performed=False,
        quality_verified=False, accuracy_verified=False, adoption_authorized=False, full_HOI_verified=False, all_cases_retained=False, source_inputs_assets_rehashed=False,
        records=[], calls=[], human_query="person.", object_query="bottle.", confidence=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, nms_iou=masks.NMS_IOU,
        ambiguity_margin=masks.AMBIGUITY_MARGIN, detector_attempts=0, actual_detector_calls=0, image_encoder_attempts=0, actual_sam2_image_encoder_calls=0,
        sam2_attempts=0, actual_sam2_calls=0, actual_automatic_inference_verified=False, body_attempts=0, body_calls_completed=0,
        parity_head_attempts=0, parity_heads_completed=0, keypoint_head_attempts=0, keypoint_heads_completed=0,
        actual_sessions=0, private_prefix_packages_installed=False, private_prefix_removed=False, native_cpu_abi_verified=False,
        native_source_modified=False, own_feed_cast=False, channel_swap=False, full_image_fallback=False, raw_scores_clamped=False, confidence_threshold_applied=False,
        wrapper_neck134_used=False, global_image_modified=False, gpu_used=args.stage!="dwpose", license_clearance_verified=False, training_overlap_excluded=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole public factorial observer exceeded stage budget")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGETS[args.stage])
        try:
            persist()
            if args.stage=="masks":run_masks(root,out,report,persist,pins)
            elif args.stage=="body":run_body(root,out,report,persist,pins)
            else:run_dwpose(root,out,report,persist,pins,started)
            if source_identity()!=report["source_helpers"] or protocol.identity(args.public_pins)!=pin_identity: raise ValueError("Public source/config changed")
            report.update(status="pass",phase="complete",all_cases_retained=True,source_inputs_assets_rehashed=True);report.pop("active_file",None)
        except BaseException as error: report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()

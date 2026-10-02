"""Automatic person/bottle masks for a fresh RGB-only identity cohort.

Queries and detector thresholds are identical for every frame. No private
camera, identity, geometry, hand label or fallback mask is accessible. A pass
means all native inference/byte contracts completed, not mask accuracy.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import hand_synthetic_masks as masks
from world_reward.prompt_selection import BoxDetection

BASE = "validation/identity_rgb_v2"
SCHEMA = "world-reward-identity-rgb-v1"
STAGE = "public_identity_rgb_automatic_masks"
CLIPS, FRAMES, WIDTH, HEIGHT, BUDGET = 3, 5, 1024, 768, 180
QUERIES = (("human", "person."), ("object", "bottle."))


def validate_inputs(directory):
    directory = Path(directory); path = directory/"manifest.json"; receipt = masks.identity(path)
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != CLIPS*FRAMES):
        raise ValueError("Require exact15 public RGB-only identity manifest")
    records = []
    for index, item in enumerate(manifest["images"]):
        clip, frame = divmod(index, FRAMES); filename = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (not isinstance(item, dict) or set(item) != {"file", "sha256", "width", "height"} or item["file"] != filename
                or not isinstance(item["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                or type(item["width"]) is not int or type(item["height"]) is not int
                or (item["width"], item["height"]) != (WIDTH, HEIGHT)):
            raise ValueError("Require ordered original1024x768 RGB identities; private fields forbidden")
        image = directory/filename; actual = masks.identity(image)
        if actual["sha256"] != item["sha256"] or actual["bytes"] <= 0:
            raise ValueError("Original public RGB SHA/size differs")
        records.append({**item, "path": image, "clip_index": clip, "frame_index": frame})
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[r["file"] for r in records]}:
        raise ValueError("Only15 original RGBs and manifest may be exposed")
    return records, receipt


def completed(report):
    records = report.get("records")
    if (report.get("actual_detector_calls") != 30 or report.get("actual_sam2_calls") != 30 or report.get("actual_sam2_image_encoder_calls") != 15
            or not isinstance(records, list) or len(records) != 15):
        raise ValueError("All30 detector and30 segmentation calls required")
    for index, row in enumerate(records):
        clip, frame = divmod(index, FRAMES); stem = f"clip_{clip:02d}_frame_{frame:03d}"
        if (type(row.get("clip_index")) is not int or row["clip_index"] != clip
                or type(row.get("frame_index")) is not int or row["frame_index"] != frame
                or row.get("file") != stem+".png"):
            raise ValueError("All original ordered frame records required")
        for label, query in QUERIES:
            if (row.get(label+"_query") != query or row.get(label+"_mask_file") != stem+"_"+label+".png"
                    or not re.fullmatch("[0-9a-f]{64}", str(row.get(label+"_mask_sha256", "")))
                    or type(row.get(label+"_mask_pixels")) is not int or not 0 < row[label+"_mask_pixels"] <= WIDTH*HEIGHT):
                raise ValueError("Complete original-grid nonempty automatic mask evidence required")


def run(root, report, persist):
    base = root/BASE; records, receipt = validate_inputs(base/"inputs")
    assets = masks.validate_assets(root, report["image_id"])
    report.update(input_manifest_sha256=receipt["sha256"], input_manifest_bytes=receipt["bytes"], model_assets=assets,
                  phase="detector_load"); persist()
    import numpy as np
    from PIL import Image
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    if not torch.cuda.is_available(): raise RuntimeError("Require actual offline CUDA masks")
    distribution = metadata.distribution("SAM-2"); direct = distribution.read_text("direct_url.json")
    report["sam2_source"] = masks.sam2_source_identity(Path(sam2.__file__).parent, json.loads(direct) if direct else {})
    report.update(torch_version=str(torch.__version__), transformers_version=metadata.version("transformers"),
                  device_name=torch.cuda.get_device_name(0))
    processor = AutoProcessor.from_pretrained(root/"weights/grounding_dino", local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(root/"weights/grounding_dino", local_files_only=True).cuda().eval()
    try:
        for item in records:
            row = {k: item[k] for k in ("file", "clip_index", "frame_index")}; row["rgb_sha256"] = item["sha256"]
            report["records"].append(row); report["phase"] = "automatic_detection"; persist()
            with Image.open(item["path"]) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                    raise ValueError("Original RGB decode contract failed")
                for label, query in QUERIES:
                    entry = processor(images=image, text=query, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        result = processor.post_process_grounded_object_detection(detector(**entry), entry.input_ids,
                            threshold=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, target_sizes=[(HEIGHT, WIDTH)])[0]
                    report["actual_detector_calls"] += 1
                    boxes = [BoxDetection(tuple(b), float(s)) for b, s in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist())]
                    chosen = masks.select_person(boxes, WIDTH, HEIGHT)
                    row[label+"_query"], row[label+"_candidate_count"] = query, len(boxes)
                    row[label+"_box"], row[label+"_detector_score"] = list(chosen.box), chosen.score
                    del entry, result; persist()
    finally: del detector, processor; torch.cuda.empty_cache()
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(root/"weights/sam2/sam2.1_hiera_large.pt"), device="cuda", mode="eval"))
    try:
        for item, row in zip(records, report["records"]):
            report["phase"] = "automatic_segmentation"; persist()
            with Image.open(item["path"]) as image, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(np.asarray(image)); report["actual_sam2_image_encoder_calls"] += 1
                for label, _ in QUERIES:
                    predicted, scores, _ = predictor.predict(box=np.array(row[label+"_box"], np.float32), multimask_output=False)
                    report["actual_sam2_calls"] += 1
                    mask = masks.binary_mask(predicted, scores, WIDTH, HEIGHT)
                    filename = Path(item["file"]).stem+"_"+label+".png"; target = base/"automatic_masks"/filename
                    with target.open("xb") as handle: Image.fromarray(mask, mode="L").save(handle, format="PNG")
                    target.chmod(0o444)
                    identity = masks.identity(target)
                    row[label+"_mask_file"], row[label+"_mask_sha256"] = filename, identity["sha256"]
                    row[label+"_mask_bytes"], row[label+"_mask_pixels"] = identity["bytes"], int(np.count_nonzero(mask))
                    row[label+"_sam2_predicted_score"] = float(scores[0]); persist()
                predictor.reset_predictor()
    finally: del predictor; torch.cuda.empty_cache()
    if validate_inputs(base/"inputs") != (records, receipt) or masks.validate_assets(root, report["image_id"]) != assets:
        raise ValueError("Frozen RGB/model asset integrity changed during masks")
    completed(report)
    names = {"report.json", *[r[label+"_mask_file"] for r in report["records"] for label, _ in QUERIES]}
    if {p.name for p in (base/"automatic_masks").iterdir()} != names: raise ValueError("Unexpected mask outputs")
    report.update(status="pass", phase="complete", actual_automatic_inference_verified=True, all_cases_retained=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote network-none automatic masks")
    root = Path(os.environ["WR_ROOT"]); out = root/BASE/"automatic_masks"
    if (root != Path("/srv/scenesmith/world-reward") or out.resolve() != out.absolute()
            or not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Require fresh exclusively reserved canonical mask output")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable mask source/image identities")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", frames=15, records=[], code_revision=revision,
        producer_revision=revision, image_id=image, script_sha256=masks.identity(Path(__file__))["sha256"],
        shared_mask_helper_sha256=masks.identity(Path(masks.__file__))["sha256"], budget_seconds=BUDGET, network="none",
        private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        human_query="person.", object_query="bottle.", accuracy_verified=False, identity_inference_performed=False,
        detector_revision=masks.DETECTOR_REVISION, sam2_weights_revision=masks.SAM2_REVISION,
        confidence=masks.CONFIDENCE, text_threshold=masks.TEXT_THRESHOLD, nms_iou=masks.NMS_IOU, ambiguity_margin=masks.AMBIGUITY_MARGIN,
        actual_detector_calls=0, actual_sam2_calls=0, actual_sam2_image_encoder_calls=0)
    start = time.perf_counter()
    with (out/"report.json").open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole automatic identity masks exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, report, persist)
        except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
            (out/"report.json").chmod(0o444)


if __name__ == "__main__": main()

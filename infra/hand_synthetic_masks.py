"""Offline RGB-only automatic person masks for six independent synthetic images.

No renderer evaluation directory, GT mask, pose, camera or identity is readable.
This initializer abstains on missing/ambiguous detections; it measures no accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

from world_reward.prompt_selection import BoxDetection, _select_detection, non_maximum_suppression


SCHEMA = "world-reward-hands-rgb-inputs-v1"
OUTPUT_SCHEMA = "world-reward-hands-automatic-masks-v1"
DETECTOR_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
SAM2_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
CONFIDENCE, TEXT_THRESHOLD, NMS_IOU, AMBIGUITY_MARGIN = .3, .25, .7, .05
ASSETS = {
    "grounding_dino/model.safetensors": ("5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21", 933400872),
    "grounding_dino/config.json": ("eda416dae6f49419ff831b1c190ec430a060b19aae688dbaf2425a075b650608", 1737),
    "grounding_dino/preprocessor_config.json": ("8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60", 457),
    "grounding_dino/special_tokens_map.json": ("b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3", 125),
    "grounding_dino/tokenizer.json": ("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711396),
    "grounding_dino/tokenizer_config.json": ("d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56", 1237),
    "grounding_dino/vocab.txt": ("07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
    "sam2/sam2.1_hiera_large.pt": ("2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318", 898083611),
    "sam2/sam2.1_hiera_l.yaml": ("545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636", 3798),
}


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError(f"Require a regular nonsymlink canonical file: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def validate_inputs(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    receipt = identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA:
        raise ValueError("Require public RGB-only input manifest, without private fields")
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != 6:
        raise ValueError("Require exactly six original RGB images")
    checked = []
    for index, item in enumerate(images):
        if not isinstance(item, dict) or set(item) != {"file", "sha256", "width", "height"}:
            raise ValueError("RGB records must contain only file/SHA/width/height")
        if item["file"] != f"case_{index:03d}.png" or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Require ordered, unique original image identities and SHA256")
        if any(type(item[key]) is not int or item[key] <= 0 for key in ("width", "height")):
            raise ValueError("Image grid requires positive integer width/height")
        path = directory / item["file"]
        actual = identity(path)
        if actual["sha256"] != item["sha256"]:
            raise ValueError("Public RGB SHA256 mismatch")
        checked.append({**item, "frame_index": index, "path": path})
    return checked, receipt


def select_person(detections, width, height):
    retained = non_maximum_suppression(detections, width, height, CONFIDENCE, NMS_IOU)
    chosen, reason = _select_detection(retained, width, height, CONFIDENCE, AMBIGUITY_MARGIN)
    if chosen is None:
        raise ValueError(f"Automatic person initializer abstained: {reason}")
    return chosen


def validate_assets(root, image_id):
    image = json.loads((root / "results/image-grounding.json").read_text())
    if not isinstance(image, dict) or image.get("Id") != image_id:
        raise RuntimeError("Installed grounding image differs from frozen build receipt")
    acquisition = json.loads((root / "results/weights-acquisition.json").read_text())
    for repo, revision, folder in (("IDEA-Research/grounding-dino-base", DETECTOR_REVISION, "grounding_dino"),
                                   ("facebook/sam2.1-hiera-large", SAM2_REVISION, "sam2")):
        matches = [item for item in acquisition["assets"] if item["repo_id"] == repo]
        if len(matches) != 1 or matches[0]["revision"] != revision or Path(matches[0]["path"]) != root / "weights" / folder:
            raise RuntimeError("Require exact pinned local detector/SAM2 acquisition")
    records = {}
    for relative, expected in ASSETS.items():
        record = identity(root / "weights" / relative)
        if (record["sha256"], record["bytes"]) != expected:
            raise RuntimeError(f"Independent pinned asset integrity failed: {relative}")
        records[relative] = record
    return records


def sam2_source_identity(package_directory, direct_url):
    """Actual VCS receipt, not a claim the original dependency had a source pin."""
    if not isinstance(direct_url, dict):
        raise RuntimeError("Require SAM2 direct_url Git metadata")
    vcs = direct_url.get("vcs_info", {}) if isinstance(direct_url, dict) else {}
    if (direct_url.get("url") != "https://github.com/facebookresearch/sam2.git"
            or vcs.get("vcs") != "git" or not isinstance(vcs.get("commit_id"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", vcs["commit_id"])):
        raise RuntimeError("Require actual SAM2 installed Git source URL and 40hex commit")
    directory = Path(package_directory)
    python = {str(path.relative_to(directory)): identity(path) for path in sorted(directory.rglob("*.py"))}
    critical = ("build_sam.py", "sam2_image_predictor.py", "modeling/sam2_base.py", "utils/transforms.py")
    if any(name not in python for name in critical):
        raise RuntimeError("Incomplete installed SAM2 Python source inventory")
    hashes = {name: record["sha256"] for name, record in python.items()}
    return {"vcs": direct_url, "python_files": len(hashes),
            "python_source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "critical_sources": {name: python[name] for name in critical},
            "config": identity(directory / "configs/sam2.1/sam2.1_hiera_l.yaml"),
            "prior_upstream_source_pin_verified": False, "full_import_license_closure_verified": False}


def binary_mask(masks, scores, width, height):
    import numpy as np
    if (np.ma.isMaskedArray(masks) or np.ma.isMaskedArray(scores) or masks.shape != (1, height, width)
            or scores.shape != (1,) or not np.isfinite(masks).all() or not np.isfinite(scores).all()):
        raise RuntimeError("SAM2 returned invalid original-grid mask/score")
    mask = (masks[0] > 0).astype(np.uint8) * 255
    if not mask.any():
        raise RuntimeError("Automatic SAM2 person mask empty; no GT fallback")
    return mask


def run(root, report):
    import numpy as np
    from PIL import Image
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("Require CUDA; no CPU fallback")
    inputs = root / "validation/hands_rgb_v1/inputs"
    output = root / "validation/hands_rgb_v1/automatic_masks"
    images, report["input_manifest"] = validate_inputs(inputs)
    report["model_assets"] = validate_assets(root, report["image_id"])
    distribution = metadata.distribution("SAM-2")
    direct = distribution.read_text("direct_url.json")
    report["sam2_source"] = sam2_source_identity(Path(sam2.__file__).parent, json.loads(direct) if direct else {})
    report.update(torch_version=torch.__version__, transformers_version=metadata.version("transformers"), gpu=torch.cuda.get_device_name())
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(root / "weights/grounding_dino", local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(root / "weights/grounding_dino", local_files_only=True).to(device).eval()
    report["images"] = []
    try:
        for item in images:
            with Image.open(item["path"]) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (item["width"], item["height"]):
                    raise ValueError("Decoded original RGB/grid differs from public manifest")
                model_input = processor(images=image, text="person.", return_tensors="pt").to(device)
                with torch.inference_mode():
                    detected = processor.post_process_grounded_object_detection(detector(**model_input), model_input.input_ids,
                        threshold=CONFIDENCE, text_threshold=TEXT_THRESHOLD, target_sizes=[(image.height, image.width)])[0]
            boxes = tuple(BoxDetection(tuple(box), float(score)) for box, score in zip(detected["boxes"].cpu().tolist(), detected["scores"].cpu().tolist()))
            record = {"frame_index": item["frame_index"], "file": item["file"], "rgb_sha256": item["sha256"],
                      "width": item["width"], "height": item["height"], "candidate_count": len(boxes)}
            report["images"].append(record)
            chosen = select_person(boxes, item["width"], item["height"])
            record.update(person_box=list(chosen.box), detector_score=chosen.score)
            del model_input, detected
            print(json.dumps({"detected_frame": item["frame_index"]}), flush=True)
    finally:
        del detector, processor
        torch.cuda.empty_cache()
    # All detections must pass, including the first two before later images.
    predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(root / "weights/sam2/sam2.1_hiera_large.pt"), device="cuda", mode="eval"))
    try:
        for item, record in zip(images, report["images"]):
            with Image.open(item["path"]) as image, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(np.asarray(image))
                masks, scores, _ = predictor.predict(box=np.asarray(record["person_box"], dtype=np.float32), multimask_output=False)
            mask = binary_mask(masks, scores, item["width"], item["height"])
            path = output / item["file"]
            with path.open("xb") as handle:
                Image.fromarray(mask, mode="L").save(handle, format="PNG")
            record.update(mask_file=item["file"], mask_sha256=identity(path)["sha256"],
                          mask_pixels=int(np.count_nonzero(mask)), sam2_predicted_score=float(scores[0]))
            predictor.reset_predictor()
    finally:
        del predictor
        torch.cuda.empty_cache()


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux network-none container")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("Require immutable code/image digests")
    output = root / "validation/hands_rgb_v1/automatic_masks"
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("Require exclusively created, empty automatic mask output")
    report = {"schema": OUTPUT_SCHEMA, "stage": "hand_synthetic_automatic_person_masks", "status": "abstain", "frames": 6,
              "code_revision": revision, "image_id": image, "script_sha256": identity(Path(__file__))["sha256"],
              "ground_truth_used": False, "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [],
              "person_query": "person.", "confidence": CONFIDENCE, "text_threshold": TEXT_THRESHOLD,
              "nms_iou": NMS_IOU, "ambiguity_margin": AMBIGUITY_MARGIN, "accuracy_evaluated": False}
    started = time.perf_counter()
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Frozen120s mask budget exceeded")))
    signal.alarm(120)
    try:
        run(root, report)
        if time.perf_counter() - started > 120:
            raise TimeoutError("Frozen120s mask budget exceeded")
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report["elapsed_seconds"] = time.perf_counter() - started
        with (output / "report.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

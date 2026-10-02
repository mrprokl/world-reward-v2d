"""Azure-only detector-seeded SAM2 masks, with no manual challenge annotations.

Run inside the isolated World Reward SAM2+GroundingDINO image. The target phrase
comes from official Track 1 metadata, not a human guess about a test frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

from world_reward.data import sha256
from world_reward.prompt_selection import BoxDetection, FrameDetections, select_seed_prompts


DETECTOR_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Video/model processing is restricted to Azure Linux")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed-frames", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.episode < 30 or args.seed_frames < 2:
        raise ValueError("Require a valid episode and at least two seed candidate frames")
    root = args.root.resolve()
    track = root / "data/track_1"
    episodes = {x["episode_index"]: x for x in map(json.loads, (track / "meta/episodes.jsonl").read_text().splitlines())}
    metadata = {x["episode_index"]: x for x in map(json.loads, (track / "meta/episodes_metadata.jsonl").read_text().splitlines())}
    record = metadata[args.episode]
    total = episodes[args.episode]["length"]
    relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{args.episode:06d}.mp4"
    video = root / "data" / relative
    manifest = json.loads((root / "results/input-manifest.json").read_text())
    expected = next(x for x in manifest["files"] if x["path"] == relative)
    if manifest["track"] != "track_1" or sha256(video) != expected["sha256"]:
        raise ValueError("Video does not match the audited Track 1 input manifest")
    output = root / f"outputs/episode_{args.episode:06d}/automatic_masks"
    output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from v2d.sam2.lib.video_to_masks import video_to_masks

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently move video processing to CPU")
    device = torch.device("cuda")
    weights = root / "weights/grounding_dino"
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(weights, local_files_only=True).to(device).eval()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != total:
        raise RuntimeError("Video decoder disagrees with official full-frame count")
    candidates = []
    evidence = []
    indices = np.unique(np.linspace(0, total - 1, min(total, args.seed_frames), dtype=int))
    try:
        for frame_index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
                raise RuntimeError("Video seek failed")
            ok, bgr = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index + 1:
                raise RuntimeError("Video seek/decode returned a wrong original frame")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            groups = []
            for text in ("person.", record["object_prompt"].strip()):
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    result = model(**inputs)
                detected = processor.post_process_grounded_object_detection(
                    result, inputs.input_ids, threshold=args.confidence,
                    text_threshold=0.25, target_sizes=[(image.height, image.width)],
                )[0]
                groups.append(tuple(
                    BoxDetection(tuple(box), float(score))
                    for box, score in zip(detected["boxes"].cpu().tolist(), detected["scores"].cpu().tolist())
                ))
            candidates.append(FrameDetections(int(frame_index), image.width, image.height, groups[0], groups[1]))
            evidence.append({"frame": int(frame_index), "person_candidates": len(groups[0]), "object_candidates": len(groups[1])})
    finally:
        cap.release()
    selected = select_seed_prompts(
        candidates, object_prompt=record["object_prompt"], confidence_threshold=args.confidence,
        ambiguity_margin=args.ambiguity_margin, total_frames=total,
    )
    prompt_file = output / "prompts.json"
    prompt_file.write_text(json.dumps(selected.to_sam2_json(), indent=2) + "\n")
    del model, processor
    torch.cuda.empty_cache()
    masks = output / "masks"
    video_to_masks(str(video), str(prompt_file), str(masks), str(root / "weights/sam2"))
    areas = {}
    for obj_id in (0, 1):
        paths = sorted((masks / str(obj_id)).glob("*.png"))
        expected_stems = [f"{index:06d}" for index in range(total)]
        if [path.stem for path in paths] != expected_stems:
            raise RuntimeError("SAM2 masks do not cover every original video frame")
        counts = [int(np.count_nonzero(np.asarray(Image.open(path)))) for path in paths]
        if max(counts) == 0:
            raise RuntimeError("SAM2 produced no visible mask for a required entity")
        areas[str(obj_id)] = {"empty_frames": sum(x == 0 for x in counts), "median_pixels": float(np.median(counts))}
    report = {
        "stage": "automatic_masks", "status": "pass", "episode_index": args.episode,
        "frames": total, "seed_frame": selected.frame_index, "seed_candidates": evidence,
        "seed_confidence": selected.confidence, "rejected_seed_frames": selected.rejected_frames,
        "confidence": args.confidence, "ambiguity_margin": args.ambiguity_margin,
        "detector_revision": DETECTOR_REVISION, "input_sha256": expected["sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mask_areas": areas, "elapsed_seconds": time.perf_counter() - started,
        "input_track": "track_1", "ground_truth_used": False, "oracle_modes": [],
        "hand_labeled_test": False, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

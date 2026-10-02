"""Azure-only detector-seeded SAM2 masks, with no manual challenge annotations.

Run inside the isolated World Reward SAM2+GroundingDINO image. The target phrase
comes from official Track 1 metadata, not a human guess about a test frame.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

from body_smoke import DATASET_REVISION, TRACK1_EPISODE_COUNT, _manifest_file
from world_reward.actor_selection import select_interacting_actor
from world_reward.prompt_selection import (
    BoxDetection, FrameDetections, non_maximum_suppression, select_seed_prompts,
)


DETECTOR_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def _episode_record(path: Path, episode_index: int, label: str) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(record, dict) or type(record.get("episode_index")) is not int
           or not 0 <= record["episode_index"] < TRACK1_EPISODE_COUNT for record in records):
        raise ValueError(f"{label} requires integer Track 1 episode identities")
    matches = [record for record in records if record["episode_index"] == episode_index]
    if len(matches) != 1:
        raise ValueError(f"{label} requires exactly one record for episode {episode_index}")
    return matches[0]


def _validate_inputs(root: Path, episode_index: int) -> dict:
    """Verify only the chosen Track1 RGB and public metadata, never decode media.

    Metadata is integrity-checked *before* its prompt/frame count is read.
    Duplicate episode records cannot silently overwrite each other. The only
    video path is constructed from the bounded integer index, not metadata.
    """
    if type(episode_index) is not int or not 0 <= episode_index < TRACK1_EPISODE_COUNT:
        raise ValueError("Require an integer Track 1 episode index in 0..29")
    manifest = json.loads((root / "results/input-manifest.json").read_text())
    if not isinstance(manifest, dict) or (manifest.get("track"), manifest.get("repo_id"), manifest.get("revision")) != (
        "track_1", "nvidia/video_to_data_challenge", DATASET_REVISION,
    ):
        raise ValueError("Require the audited, pinned official Track 1 input manifest")
    episodes_path, episodes_sha = _manifest_file(root, manifest, "track_1/meta/episodes.jsonl")
    metadata_path, metadata_sha = _manifest_file(root, manifest, "track_1/meta/episodes_metadata.jsonl")
    episode = _episode_record(episodes_path, episode_index, "Official episodes metadata")
    metadata = _episode_record(metadata_path, episode_index, "Official object prompt metadata")
    total = episode.get("length")
    if type(total) is not int or total < 3:
        raise ValueError("Official episode frame count must be an integer >=3")
    prompt = metadata.get("object_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Official object_prompt must be a nonempty string")
    relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode_index:06d}.mp4"
    video, video_sha = _manifest_file(root, manifest, relative)
    return {
        "episode_index": episode_index, "video": video, "video_sha256": video_sha,
        "object_prompt": prompt.strip(), "total_frames": total,
        "dataset_revision": DATASET_REVISION,
        "metadata_sha256": {"episodes.jsonl": episodes_sha, "episodes_metadata.jsonl": metadata_sha},
    }


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Video/model processing is restricted to Azure Linux")
    if {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require an isolated container launched with docker --network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed-frames", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--actor-seed-observations", type=int, default=3)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.episode < 30 or args.seed_frames < 3 or not 3 <= args.actor_seed_observations <= args.seed_frames:
        raise ValueError("Require a valid episode and at least three actor seed observations")
    root = args.root.resolve()
    inputs = _validate_inputs(root, episode_index=args.episode)
    record = {"object_prompt": inputs["object_prompt"]}
    total, video = inputs["total_frames"], inputs["video"]
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
            observations = []
            for text in ("person.", record["object_prompt"].strip()):
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    result = model(**inputs)
                detected = processor.post_process_grounded_object_detection(
                    result, inputs.input_ids, threshold=args.confidence,
                    text_threshold=0.25, target_sizes=[(image.height, image.width)],
                )[0]
                raw = tuple(
                    BoxDetection(tuple(box), float(score))
                    for box, score in zip(detected["boxes"].cpu().tolist(), detected["scores"].cpu().tolist())
                )
                groups.append(non_maximum_suppression(
                    raw, image.width, image.height, confidence_threshold=args.confidence,
                    iou_threshold=args.nms_iou,
                ))
                observations.append({
                    "query": text,
                    "labels": detected["text_labels"],
                    "boxes": [{"box": x.box, "score": x.score} for x in raw],
                    "retained": [{"box": x.box, "score": x.score} for x in groups[-1]],
                })
            candidates.append(FrameDetections(int(frame_index), image.width, image.height, groups[0], groups[1]))
            evidence.append({"frame": int(frame_index), "person_candidates": len(groups[0]), "object_candidates": len(groups[1])})
            if len(evidence) <= 3:
                evidence[-1]["detector_observations"] = observations
    finally:
        cap.release()
    diagnostics = {"episode": args.episode, "frames": total, "observations": evidence,
                   "confidence": args.confidence, "nms_iou": args.nms_iou,
                   "detector_revision": DETECTOR_REVISION,
                   "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False}
    (output / "seed-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if args.diagnose_only:
        print(json.dumps({"status":"diagnostics_saved", "episode":args.episode}))
        return
    # This is only a SAM2 initializer, not a reconstructed person trajectory.
    # Establish identity on a fixed short prefix; full-resolution SAM2 then tracks
    # every video frame. Never emit the sparse bbox associations as final motion.
    actor = select_interacting_actor(candidates[:args.actor_seed_observations], confidence_threshold=args.confidence,
                                     object_ambiguity_margin=args.ambiguity_margin)
    selected = select_seed_prompts(
        actor.frames, object_prompt=record["object_prompt"], confidence_threshold=args.confidence,
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
        "actor_track_id": actor.track_id, "actor_scores": [asdict(score) for score in actor.scores],
        "actor_seed_observations": args.actor_seed_observations,
        "confidence": args.confidence, "ambiguity_margin": args.ambiguity_margin,
        "nms_iou": args.nms_iou,
        "detector_revision": DETECTOR_REVISION, "input_sha256": inputs["video_sha256"],
        "input_dataset_revision": inputs["dataset_revision"], "input_metadata_sha256": inputs["metadata_sha256"],
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

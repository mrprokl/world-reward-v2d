"""Azure-only streaming mask proxies, no ground-truth accuracy claim."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform

from world_reward.data import sha256
from world_reward.mask_quality import analyze_mask_sequence
from world_reward.prompt_selection import BoxDetection, FrameDetections


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Mask processing stays on Azure")
    import numpy as np
    from PIL import Image
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    directory = root / "outputs/episode_000015/automatic_masks"
    mask_report = json.loads((directory / "report.json").read_text())
    if mask_report["status"] != "pass" or mask_report["ground_truth_used"] is not False or mask_report["hand_labeled_test"] is not False:
        raise RuntimeError("Require audited automatic masks, no GT/manual labels")
    total = mask_report["frames"]
    with Image.open(directory / "masks/0/000000.png") as image:
        width, height = image.size
    frame_indices = list(range(total))
    def stream(entity: int):
        for index in frame_indices:
            path = directory / f"masks/{entity}/{index:06d}.png"
            with Image.open(path) as image:
                mask = np.asarray(image)
            if mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
                raise RuntimeError("Expected original-resolution binary mask PNG")
            yield mask != 0
    detections = []
    for observation in mask_report["seed_candidates"]:
        if "detector_observations" not in observation:
            continue
        groups = tuple(tuple(BoxDetection(tuple(box["box"]), box["score"])
                             for box in query["retained"])
                       for query in observation["detector_observations"])
        detections.append(FrameDetections(observation["frame"], width, height, groups[0], groups[1]))
    report = analyze_mask_sequence(stream(0), stream(1), frame_indices,
                                   detections=detections, total_frames=total)
    report["mask_report_sha256"] = sha256(directory / "report.json")
    report["code_revision"] = os.environ.get("WR_CODE_REVISION")
    report["input_track"] = "track_1"
    report["ground_truth_used"] = False
    report["hand_labeled_test"] = False
    with (directory / "quality-diagnostics.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    scalar_summary = dict(report["summary"])
    for entity in ("person", "object"):
        fractions = [frame[entity]["largest_component_fraction"] for frame in report["frames"]]
        motion = [pair[entity]["centroid_distance_over_image_diagonal"] for pair in report["temporal_pairs"]]
        agreement = [frame["detection_agreement"][entity]["max_bbox_iou"]
                     for frame in report["frames"] if frame["detection_agreement"]]
        scalar_summary[entity] = {
            "median_largest_component_fraction": float(np.median([x for x in fractions if x is not None])),
            "max_normalized_centroid_jump": float(max(x for x in motion if x is not None)),
            "sparse_independent_detector_bbox_iou": agreement,
        }
    print(json.dumps({"stage": "mask_quality_proxies", "summary": scalar_summary,
                      "accuracy_verified": False}))


if __name__ == "__main__":
    main()

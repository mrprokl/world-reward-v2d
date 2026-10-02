"""Estimate three-frame monocular MoGe2 depth offline with an RGB-only FOV prior."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import time

from body_smoke import _validate_inputs
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-video", action="store_true")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    inputs = _validate_inputs(root)
    revision = "b135031bae30b5ac2ae141a0e68717795ce38340"
    path = root / f"weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/{revision}/model.pt"
    if not path.is_file() or not torch.cuda.is_available():
        raise RuntimeError("Pinned local MoGe2/CUDA missing")
    output = root / "outputs/episode_000015" / ("depth_full" if args.full_video else "depth_smoke")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    model = MoGeModel.from_pretrained(str(path)).cuda().eval()
    cap = cv2.VideoCapture(str(inputs["video"]))
    records = []
    indices = range(inputs["total_frames"]) if args.full_video else inputs["indices"]
    try:
        for index in indices:
            if not args.full_video and not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError("Video seek failed")
            ok, bgr = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                raise RuntimeError("Wrong original frame decode")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            focal = float(np.hypot(h, w))
            fov = float(np.degrees(2 * np.arctan(w / (2 * focal))))
            image = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
            with torch.inference_mode():
                predicted = model.infer(image[None], fov_x=fov)
            depth = predicted["depth"][0].cpu().numpy()
            points = predicted["points"][0].cpu().numpy()
            mask = predicted["mask"][0].cpu().numpy().astype(bool)
            intrinsic = predicted["intrinsics"][0].cpu().numpy()
            if depth.shape != (h, w) or points.shape != (h, w, 3) or mask.shape != (h, w):
                raise RuntimeError("MoGe full original-frame output shape mismatch")
            if not mask.any() or not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any() or not np.isfinite(points[mask]).all():
                raise RuntimeError("Invalid inferred valid-region depth/points")
            if not np.isfinite(intrinsic).all() or intrinsic.shape != (3, 3):
                raise RuntimeError("Invalid inferred normalized intrinsics")
            camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            camera_checks = validate_camera_pointmap(depth, points, mask, intrinsic, camera)
            target = output / f"{index:06d}.npz"
            with target.open("xb") as handle:
                # Full-video XYZ is redundant under the verified pinhole gate.
                # Store only camera Z/validity/K; reconstruct rays exactly at +.5
                # downstream. Never stream dense arrays back to the laptop.
                arrays = {"depth": depth, "mask": mask, "intrinsics": intrinsic, "frame_index": np.array(index)}
                if not args.full_video:
                    arrays["points"] = points
                np.savez_compressed(handle, **arrays)
            records.append({"frame_index": index, "valid_pixels": int(mask.sum()), "median_depth_model_units": float(np.median(depth[mask])),
                            "fov_x_prior_degrees": fov, "intrinsics_normalized": intrinsic.tolist(),
                            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "camera_checks": camera_checks,
                            "output_sha256": sha256(target)})
    finally:
        cap.release()
    report = {"stage": "monocular_moge2_full_video" if args.full_video else "monocular_moge2_three_frame",
              "status": "pass", "frames": records, "total_video_frames": inputs["total_frames"],
              "array_storage": "verified_camera_Z_K_mask_no_redundant_XYZ" if args.full_video else "depth_points_K_mask",
              "elapsed_seconds": time.perf_counter() - started, "model_revision": revision,
              "model_sha256": sha256(path), "input_sha256": inputs["video_sha256"],
              "intrinsics_source": "RGB_size_only_default_FOV_prior_not_calibration",
              "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
              "oracle_modes": [], "network": "none", "challenge_performance_verified": False,
              "metric_scale_accuracy_verified": False, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": report["stage"], "status": "pass", "frames": len(records),
                      "elapsed_seconds": report["elapsed_seconds"], "metric_scale_accuracy_verified": False}))


if __name__ == "__main__":
    main()

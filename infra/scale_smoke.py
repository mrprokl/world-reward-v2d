"""Fit one human-anchored MoGe2 gauge and export same-camera pointmaps.

Both inputs are predictions: this is a consistency experiment, not a score or
independently calibrated metric scale. MoGe1 object outputs are not consumed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

from body_smoke import EPISODE, TRACK1_EPISODE_COUNT, _validate_inputs
from camera_render import raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.artifact_paths import episode_output
from world_reward.metric_alignment import fit_shared_depth_scale
from world_reward.pointmap import validate_camera_pointmap
from world_reward.mask_observation import alignment_masks


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    return parser


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    args = _argument_parser().parse_args()
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    inputs = _validate_inputs(root, episode_index=args.episode)
    import numpy as np
    from PIL import Image
    gate_path = root / "results/camera-render.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "pass":
        raise RuntimeError("Analytic CUDA camera/depth gate must pass first")
    base = episode_output(root, args.episode)
    output = base / "scale_smoke"
    if output.exists():
        raise RuntimeError("Frozen alignment output exists; do not overwrite")
    body_path, depth_path = base / "body_smoke/report.json", base / "depth_smoke/report.json"
    body_report, depth_report = [json.loads(path.read_text()) for path in (body_path, depth_path)]
    for report, stage in ((body_report, "sam3d_body_three_frame_smoke"),
                          (depth_report, "monocular_moge2_three_frame")):
        expected = {"stage": stage, "status": "pass", "episode_index": args.episode, "input_track": "track_1",
                    "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
                    "hand_labeled_test": False}
        if any(report.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Body/depth source provenance mismatch")
        if report["ground_truth_used"] is not False or report["hand_labeled_test"] is not False or report.get("oracle_modes", []) != []:
            raise RuntimeError("No ground truth, manual test labeling or oracle inputs permitted")
    if (body_report.get("mhr_geometry_forward_verified") is not True or body_report.get("geometry_units") != "metres"
            or body_report["mask_report_sha256"] != inputs["mask_report_sha256"]
            or body_report["prompts_sha256"] != inputs["prompts_sha256"]):
        raise RuntimeError("Native human units/forward/mask provenance must be verified")
    predictions_path = base / "body_smoke/predictions.npz"
    if sha256(predictions_path) != body_report["predictions_sha256"]:
        raise RuntimeError("Body data changed after verified native forward")
    with np.load(predictions_path, allow_pickle=False) as data:
        human_vertices = data["vertices_camera_m"].copy()
        human_faces = data["faces"].copy()
        indices = data["frame_index"].tolist()
        focals = data["focal_length"].copy()
    if indices != inputs["indices"] or body_report["frame_indices"] != indices:
        raise RuntimeError("Fixed original-frame scale smoke indices changed")
    body_frames = {r["frame_index"]: r for r in body_report["frames"]}
    depth_frames = {r["frame_index"]: r for r in depth_report["frames"]}
    if (len(body_frames) != len(body_report["frames"]) or len(depth_frames) != len(depth_report["frames"])
            or sorted(body_frames) != indices or sorted(depth_frames) != indices):
        raise RuntimeError("Missing/duplicate body or depth frame records")
    started = time.perf_counter()
    depths, pointmaps, human_depths, masks, validities, cameras = [], [], [], [], [], []
    evidence = []
    for position, index in enumerate(indices):
        source = base / f"depth_smoke/{index:06d}.npz"
        if sha256(source) != depth_frames[index]["output_sha256"]:
            raise RuntimeError("MoGe2 data changed after inference")
        if body_frames[index]["decoded_rgb_sha256"] != depth_frames[index]["decoded_rgb_sha256"]:
            raise RuntimeError("Human and depth do not refer to identical decoded RGB")
        with np.load(source, allow_pickle=False) as data:
            if data["frame_index"].ndim != 0 or int(data["frame_index"]) != index:
                raise RuntimeError("Depth array original frame index mismatch")
            depths.append(data["depth"].copy())
            pointmaps.append(data["points"].copy())
            validity = data["mask"].copy()
            inferred_intrinsics = data["intrinsics"].copy()
        human_path = base / f"automatic_masks/masks/0/{index:06d}.png"
        object_path = base / f"automatic_masks/masks/1/{index:06d}.png"
        if sha256(human_path) != body_frames[index]["mask_sha256"]:
            raise RuntimeError("Human mask changed after body inference")
        masks_image = []
        for path in (human_path, object_path):
            with Image.open(path) as image:
                array = np.asarray(image)
            masks_image.append(array)
        # Object visibility does not determine whether the human anchors scale.
        # Preserve literal absence; no invented mask or negative existence label.
        human_mask, object_mask = alignment_masks(*masks_image)
        if object_mask.shape != human_mask.shape:
            raise RuntimeError("Automatic mask resolution mismatch")
        h, w = human_mask.shape
        camera = np.array([[focals[position], 0, w / 2], [0, focals[position], h / 2], [0, 0, 1]])
        consistency = validate_camera_pointmap(depths[-1], pointmaps[-1], validity, inferred_intrinsics, camera)
        if depths[-1].shape != human_mask.shape:
            raise RuntimeError("Masks and depth original resolution mismatch")
        cameras.append(camera)
        validities.append(validity)
        rendered_mask, rendered_depth = raster_camera_mesh(human_vertices[position], human_faces, camera, w, h)
        silhouette, depth = rendered_mask.cpu().numpy(), rendered_depth.cpu().numpy()
        # Exclude object pixels from the human fit, not final object masks.
        visible = silhouette & human_mask & ~object_mask & validity
        human_depths.append(depth)
        masks.append(visible)
        evidence.append({"frame_index": index, "silhouette_iou": silhouette_iou(silhouette, human_mask),
                         "object_observation_present": bool(object_mask.any()),
                         "alignment_pixels": int(visible.sum()), "pointmap_camera_checks": consistency,
                         "estimated_body_K": camera.tolist(),
                         "decoded_rgb_sha256": body_frames[index]["decoded_rgb_sha256"],
                         "depth_npz_sha256": depth_frames[index]["output_sha256"],
                         "human_mask_sha256": sha256(human_path), "object_mask_sha256": sha256(object_path)})
    aligned = fit_shared_depth_scale(depths, human_depths, masks, indices)
    output.mkdir(exist_ok=False)
    pointmap_records = []
    for position, index in enumerate(indices):
        # Scale XYZ together once; never inpaint invalid observations.
        points = pointmaps[position] * aligned.shared_scale
        if not np.isfinite(points[validities[position]]).all():
            raise RuntimeError("Aligned valid points overflowed")
        points[~validities[position]] = np.nan
        points = points.astype(np.float32)
        if not np.isfinite(points[validities[position]]).all() or (points[..., 2][validities[position]] <= 0).any():
            raise RuntimeError("Aligned valid points cannot be represented as positive finite float32 XYZ")
        path = output / f"{index:06d}.npy"
        with path.open("xb") as handle:
            np.save(handle, points, allow_pickle=False)
        intrinsic_path = output / f"{index:06d}_intrinsics.json"
        camera = cameras[position]
        height, width = depths[position].shape
        intrinsic_path.write_text(json.dumps({"fx": float(camera[0, 0]), "fy": float(camera[1, 1]),
                                              "cx": float(camera[0, 2]), "cy": float(camera[1, 2]),
                                              "width": width, "height": height}) + "\n")
        pointmap_records.append({"frame_index": index, "pointmap_path": str(path), "pointmap_sha256": sha256(path),
                                 "intrinsics_path": str(intrinsic_path), "intrinsics_sha256": sha256(intrinsic_path),
                                 "decoded_rgb_sha256": evidence[position]["decoded_rgb_sha256"]})
    report = {"stage": "predicted_human_anchored_moge2_pointmaps", "status": "pass",
              "execution_verified": True, "candidate_accuracy_validated": False,
              "episode_index": args.episode, "frame_indices": indices, "depth_alignment": aligned.to_dict(),
              "human_evidence": evidence, "pointmaps": pointmap_records,
              "coordinate_frame": "OpenCV_x_right_y_down_z_forward",
              "pointmap_scale_application": "one_clip_scalar_to_MoGe2_XYZ_already_applied",
              "scale_source": "predicted_human_MHR_metres_not_independent_metric_calibration",
              "never_applied_to_independent_moge1_outputs": True,
              "input_sha256": inputs["video_sha256"], "body_report_sha256": sha256(body_path),
              "depth_report_sha256": sha256(depth_path), "camera_gate_sha256": sha256(gate_path),
              "mesh_shape_scale_accuracy_verified": False, "challenge_performance_verified": False,
              "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
              "oracle_modes": [], "network": "none", "submission_eligible": False,
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": report["stage"], "status": "pass", "shared_depth_scale": aligned.shared_scale,
                      "depth_residual": aligned.median_absolute_relative_error,
                      "human_silhouette_iou": [x["silhouette_iou"] for x in evidence],
                      "metric_scale_accuracy_verified": False, "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

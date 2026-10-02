"""Align inferred depth to a predicted human and test one rigid object candidate.

This is same-video image/depth evidence, not challenge ground truth or a score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

from camera_render import raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.metric_alignment import fit_shared_depth_scale
from world_reward.rigid_alignment import align_observed_points


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Geometry and depth stay on Azure")
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    gate = json.loads((root / "results/camera-render.json").read_text())
    if gate["status"] != "pass":
        raise RuntimeError("Analytic CUDA camera/depth gate must pass first")
    base = root / "outputs/episode_000015"
    output = base / "scale_smoke"
    output.mkdir(exist_ok=False)
    body_report = json.loads((base / "body_smoke/report.json").read_text())
    if sha256(base / "body_smoke/predictions.npz") != body_report["predictions_sha256"]:
        raise RuntimeError("Body data changed after verified native forward")
    with np.load(base / "body_smoke/predictions.npz", allow_pickle=False) as data:
        human_vertices = data["vertices_camera_m"]
        human_faces = data["faces"]
        indices = data["frame_index"].tolist()
        focals = data["focal_length"]
    if indices != [0, 250, 500]:
        raise RuntimeError("Fixed scale smoke indices changed")
    started = time.perf_counter()
    depths, pointmaps, human_depths, masks, cameras = [], [], [], [], []
    human_evidence = []
    for position, index in enumerate(indices):
        with np.load(base / f"depth_smoke/{index:06d}.npz", allow_pickle=False) as data:
            depths.append(data["depth"].copy())
            pointmaps.append(data["points"].copy())
            validity = data["mask"].copy()
            inferred_intrinsics = data["intrinsics"].copy()
        with Image.open(base / f"automatic_masks/masks/0/{index:06d}.png") as image:
            human_mask = np.asarray(image) > 0
        with Image.open(base / f"automatic_masks/masks/1/{index:06d}.png") as image:
            object_mask = np.asarray(image) > 0
        h, w = human_mask.shape
        camera = np.array([[focals[position], 0, w / 2], [0, focals[position], h / 2], [0, 0, 1]])
        cameras.append(camera)
        rendered_mask, rendered_depth = raster_camera_mesh(human_vertices[position], human_faces, camera, w, h)
        silhouette = rendered_mask.cpu().numpy()
        depth = rendered_depth.cpu().numpy()
        # Exclude automatically observed object pixels to avoid visible object
        # depth masquerading as a human surface. No depth-oracle occlusion tests.
        visible = silhouette & human_mask & ~object_mask & validity
        human_depths.append(depth)
        masks.append(visible)
        human_evidence.append({"frame_index": index, "silhouette_iou": silhouette_iou(silhouette, human_mask),
                               "alignment_pixels": int(visible.sum()),
                               "inferred_intrinsics_normalized": inferred_intrinsics.tolist(),
                               "estimated_body_K": camera.tolist()})
    aligned = fit_shared_depth_scale(depths, human_depths, masks, indices)
    alignment = aligned.to_dict()
    # Generate fixed local geometry with the model's predicted uniform scale,
    # then the single clip depth alignment; never shrink it against PEN.
    object_report = json.loads((base / "object_smoke/report.json").read_text())
    original = base / "object_smoke/object.glb"
    if sha256(original) != object_report["object_sha256"]:
        raise RuntimeError("Object candidate changed after inference")
    sys.path.insert(0, str(root / "vendor/v2d_submission_kit"))
    from v2dlb.mesh_budget import budget_mesh
    v, f = budget_mesh(str(original), faces=4096, vertices=4096)
    nondegenerate = (f[:, 0] != f[:, 1]) & (f[:, 0] != f[:, 2]) & (f[:, 1] != f[:, 2])
    compact = trimesh.Trimesh(v, f[nondegenerate], process=True)
    transform = object_report["transform"]
    scale = np.asarray(transform["scale"], dtype=float)
    if not np.allclose(scale, scale[0], rtol=1e-5, atol=0):
        raise RuntimeError("Smoke expects uniform generative scale; anisotropic candidates need explicit fixed-mesh baking")
    compact.vertices *= scale[0] * aligned.shared_scale
    compact.export(output / "object_metric_candidate.glb")
    quaternion = transform["rotation"]
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
    translation = np.asarray(transform["translation"]) * aligned.shared_scale
    with Image.open(base / "automatic_masks/masks/1/000000.png") as image:
        object_mask = np.asarray(image) > 0
    mesh_mask, _ = raster_camera_mesh(compact.vertices @ rotation.T + translation, compact.faces,
                                     cameras[0], object_mask.shape[1], object_mask.shape[0])
    initial_iou = silhouette_iou(mesh_mask.cpu().numpy(), object_mask)
    # ICP sees only automatically masked inferred points; deterministic bounded
    # sampling, no handpicked pixels or hidden geometry targets.
    finite = object_mask & np.isfinite(pointmaps[0]).all(-1) & (depths[0] > 0)
    observed = pointmaps[0][finite] * aligned.shared_scale
    rng = np.random.default_rng(0)
    if len(observed) > 2048:
        observed = observed[rng.choice(len(observed), 2048, replace=False)]
    sampled, _ = trimesh.sample.sample_surface(compact, 8192, seed=0)
    fit = align_observed_points(sampled, observed, rotation, translation)
    refined_mask, _ = raster_camera_mesh(compact.vertices @ fit.rotation.T + fit.translation,
                                        compact.faces, cameras[0], object_mask.shape[1], object_mask.shape[0])
    refined_iou = silhouette_iou(refined_mask.cpu().numpy(), object_mask)
    report = {"stage": "predicted_human_depth_and_object_scale_smoke", "status": "pass",
              "depth_alignment": alignment, "human_evidence": human_evidence,
              "object_initial_silhouette_iou": initial_iou, "object_icp_silhouette_iou": refined_iou,
              "object_icp": fit.to_dict(), "object_metric_candidate_extent": compact.extents.tolist(),
              "packed_vertices": len(compact.vertices), "packed_faces": len(compact.faces),
              "object_watertight_after_budget": bool(compact.is_watertight),
              "mesh_shape_scale_accuracy_verified": False, "challenge_performance_verified": False,
              "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": report["stage"], "shared_depth_scale": aligned.shared_scale,
                      "depth_residual": aligned.median_absolute_relative_error,
                      "human_silhouette_iou": [x["silhouette_iou"] for x in human_evidence],
                      "object_initial_iou": initial_iou, "object_icp_iou": refined_iou,
                      "object_extent": compact.extents.tolist(), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

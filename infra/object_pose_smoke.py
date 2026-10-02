"""Fixed-scale, three-frame rigid pose consistency from automatic image evidence.

Predicted depth is not GT. This sparse engineering experiment neither emits a
full trajectory nor claims accuracy, calibrated scale or challenge eligibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

from body_smoke import _validate_inputs
from camera_render import raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.rigid_alignment import align_observed_points


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import trimesh
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    inputs = _validate_inputs(root)
    base = root / "outputs/episode_000015"
    output = base / "object_pose_smoke"
    if output.exists():
        raise RuntimeError("Frozen object pose smoke already exists")
    object_dir = base / "object_grounded"
    object_report_path, alignment_path = object_dir / "report.json", base / "scale_smoke/report.json"
    report = json.loads(object_report_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    for record, stage in ((report, "sam3d_objects_grounded_fixed_frame"),
                          (alignment, "predicted_human_anchored_moge2_pointmaps")):
        expected = {"stage": stage, "status": "pass", "input_track": "track_1",
                    "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
                    "hand_labeled_test": False, "oracle_modes": []}
        if (any(record.get(key) != value for key, value in expected.items())
                or record["ground_truth_used"] is not False or record["hand_labeled_test"] is not False):
            raise RuntimeError("Object/alignment video-only provenance mismatch")
    if report["scale_source"] != "already_human_anchored_MoGe2_no_second_scalar":
        raise RuntimeError("Never apply a MoGe2 scale to an independent MoGe1 object pose")
    if report["pointmap_grounding"]["alignment_report_sha256"] != sha256(alignment_path):
        raise RuntimeError("Object grounding alignment changed")
    for filename, field in (("object.glb", "object_sha256"), ("transform.json", "transform_sha256"),
                            ("intrinsics.json", "intrinsics_sha256")):
        if sha256(object_dir / filename) != report[field]:
            raise RuntimeError("Frozen object mesh/transform/camera changed")
    transform = json.loads((object_dir / "transform.json").read_text())
    if transform != report["transform"]:
        raise RuntimeError("Object transform does not match inference report")
    scale = np.asarray(transform["scale"], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("Object scale must be finite and positive")
    if not np.allclose(scale, scale[0], atol=0, rtol=1e-5):
        raise RuntimeError("Anisotropic generated scale needs explicit fixed-mesh baking; no averaging")
    quaternion = transform["rotation"]
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
    translation = np.asarray(transform["translation"], dtype=float)
    camera_dict = json.loads((object_dir / "intrinsics.json").read_text())
    camera = np.array([[camera_dict["fx"], 0, camera_dict["cx"]],
                       [0, camera_dict["fy"], camera_dict["cy"]], [0, 0, 1]])
    height, width = camera_dict["height"], camera_dict["width"]
    sys.path.insert(0, str(root / "vendor/v2d_submission_kit"))
    from v2dlb.mesh_budget import budget_mesh
    import inspect
    import importlib.metadata
    budget_source_path = Path(inspect.getfile(budget_mesh))
    started = time.perf_counter()
    # The GLB is canonical; pose is stored separately. Keep exact official budget
    # arrays for final packing and derive only a non-padding mesh for raster QA.
    vertices, faces = budget_mesh(str(object_dir / "object.glb"), faces=4096, vertices=4096)
    vertices = vertices * scale[0]  # same grounding gauge, applied exactly once
    active = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    mesh = trimesh.Trimesh(vertices, faces[active], process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise RuntimeError("Official-budget geometry lost closed oriented volume; do not use for PEN")
    sampled, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0)
    pointmaps = {record["frame_index"]: record for record in alignment["pointmaps"]}
    evidence = {record["frame_index"]: record for record in alignment["human_evidence"]}
    if (len(pointmaps) != len(alignment["pointmaps"]) or len(evidence) != len(alignment["human_evidence"])
            or sorted(pointmaps) != inputs["indices"] or sorted(evidence) != inputs["indices"]):
        raise RuntimeError("Sparse evidence must retain exact original frame indices")
    candidate_reports, poses_R, poses_t = [], [], []
    # Finite generic orientation hypotheses, not manually supplied object
    # symmetries. All can compete by observed silhouette; no symmetry averaging.
    orientation_hypotheses = Rotation.create_group("O").as_matrix()
    for index in inputs["indices"]:
        point_record = pointmaps[index]
        point_path = Path(point_record["pointmap_path"])
        if point_path != alignment_path.parent / f"{index:06d}.npy" or sha256(point_path) != point_record["pointmap_sha256"]:
            raise RuntimeError("Pointmap path/hash mismatch")
        if sha256(Path(point_record["intrinsics_path"])) != point_record["intrinsics_sha256"]:
            raise RuntimeError("Pointmap camera hash mismatch")
        if json.loads(Path(point_record["intrinsics_path"]).read_text()) != camera_dict:
            raise RuntimeError("One fixed RGB-size camera must agree across object and human pointmaps")
        mask_path = base / f"automatic_masks/masks/1/{index:06d}.png"
        if sha256(mask_path) != evidence[index]["object_mask_sha256"]:
            raise RuntimeError("Automatic object mask changed after alignment")
        with Image.open(mask_path) as image:
            mask = np.asarray(image) > 0
        points = np.load(point_path, allow_pickle=False)
        if mask.shape != (height, width) or points.shape != (height, width, 3):
            raise RuntimeError("Object mask/pointmap original camera grid mismatch")
        visible = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
        observed = points[visible]
        if len(observed) < 40:
            raise RuntimeError("Insufficient inferred visible object points")
        rng = np.random.default_rng(0)
        if len(observed) > 2048:
            observed = observed[rng.choice(len(observed), 2048, replace=False)]
        candidates, rejected = [], []
        for number, hypothesis in enumerate(orientation_hypotheses):
            initial_R = rotation @ hypothesis
            # Keep the generative translation at frame zero; otherwise seed
            # the surface centre from automatic observed depth, never hand pose.
            initial_t = translation if index == 0 else np.median(observed, axis=0) - mesh.centroid @ initial_R.T
            try:
                initial_mask, _ = raster_camera_mesh(mesh.vertices @ initial_R.T + initial_t, mesh.faces, camera, width, height)
                initial_iou = silhouette_iou(initial_mask.cpu().numpy(), mask)
                fit = align_observed_points(sampled, observed, initial_R, initial_t)
                fitted_mask, _ = raster_camera_mesh(mesh.vertices @ fit.rotation.T + fit.translation, mesh.faces, camera, width, height)
                fitted_iou = silhouette_iou(fitted_mask.cpu().numpy(), mask)
                # Partial depth ICP cannot choose unseen orientation by residual
                # alone. Require non-worse 2D image support before accepting a step.
                accepted = fitted_iou >= initial_iou and fit.final_residual <= fit.initial_residual
                chosen_R, chosen_t = (fit.rotation, fit.translation) if accepted else (initial_R, initial_t)
                chosen_iou = fitted_iou if accepted else initial_iou
                chosen_residual = fit.final_residual if accepted else fit.initial_residual
                candidates.append({"hypothesis_index": number, "initial_silhouette_iou": initial_iou,
                                   "fitted_silhouette_iou": fitted_iou, "icp_accepted_by_image_gate": bool(accepted),
                                   "selected_silhouette_iou": chosen_iou, "selected_depth_residual_m": chosen_residual,
                                   "rotation": chosen_R.tolist(), "translation": chosen_t.tolist(), "icp": fit.to_dict()})
            except ValueError as exc:
                # Candidate numerical/near-plane/underconstraint failure is not
                # permission to bypass global input provenance or CUDA errors.
                rejected.append({"hypothesis_index": number, "reason": str(exc)})
        if not candidates:
            raise RuntimeError(f"No finite supported pose hypothesis at original frame {index}: {rejected}")
        best = max(candidates, key=lambda c: (c["selected_silhouette_iou"], -c["selected_depth_residual_m"], -c["hypothesis_index"]))
        poses_R.append(best["rotation"])
        poses_t.append(best["translation"])
        candidate_reports.append({"frame_index": index, "visible_point_pixels": int(visible.sum()),
                                  "sampled_observations": len(observed), "selected": best, "candidates": candidates,
                                  "rejected_candidates": rejected})
    output.mkdir(exist_ok=False)
    with (output / "geometry_and_poses.npz").open("xb") as handle:
        np.savez_compressed(handle, vertices=vertices, faces=faces, frame_index=np.asarray(inputs["indices"]),
                            rotation=np.asarray(poses_R), translation=np.asarray(poses_t), object_scale=np.array(1.))
    result = {"stage": "fixed_scale_sparse_object_pose_consistency", "status": "pass", "execution_verified": True,
              "candidate_accuracy_validated": False, "full_trajectory_verified": False,
              "input_track": "track_1", "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "submission_eligible": False,
              "challenge_performance_verified": False, "metric_scale_accuracy_verified": False,
              "object_report_sha256": sha256(object_report_path), "alignment_report_sha256": sha256(alignment_path),
              "fixed_shape": True, "scale": "generative_grounded_scale_baked_once_into_fixed_vertices; submission_scale=1",
              "mesh_watertight": bool(mesh.is_watertight), "mesh_winding_consistent": bool(mesh.is_winding_consistent),
              "metric_gauge_extent": mesh.extents.tolist(), "official_budget_vertices": len(vertices), "official_budget_faces": len(faces),
              "pose_hypotheses": "24_octahedral_orientations_not_asserted_true_object_symmetries",
              "objective": "maximum_automatic_mask_IoU_then_partial_depth_RMSE; no_GT_or_challenge_metric",
              "frames": candidate_reports, "geometry_and_poses_sha256": sha256(output / "geometry_and_poses.npz"),
              "budget_source_sha256": sha256(budget_source_path),
              "geometry_library_versions": {name: importlib.metadata.version(name) for name in ("trimesh", "fast-simplification")},
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "status": "pass", "extent": mesh.extents.tolist(),
                      "selected_iou": [frame["selected"]["selected_silhouette_iou"] for frame in candidate_reports],
                      "elapsed_seconds": result["elapsed_seconds"], "challenge_performance_verified": False}))


if __name__ == "__main__":
    main()

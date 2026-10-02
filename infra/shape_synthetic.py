"""Controlled procedural shape experiment, not RGB-pipeline/challenge validation.

Known synthetic poses are explicit oracle conditions. Ground-truth geometry
generates masked camera-Z observations and evaluates proposals; it is never a
fitting argument. No challenge inputs, weights, alignment, media exports or
per-condition tuning. All rasterization and experiment fitting stay on Azure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np


CASES = (
    ("asymmetric", (.40, .23, .31), (.06, -.05, .02, -.03, .01)),
    ("thin", (.10, .22, .35), (.08, -.04, -.02, .01, .025)),
    ("symmetric_control", (.30, .30, .30), (0., 0., 0., 0., 0.)),
)
OCCLUSIONS = (0., .25, .50)
FIT_FRAMES, HELDOUT_FRAMES = (0, 2, 4), (1, 3, 5)
K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
WIDTH, HEIGHT, SURFACE_SAMPLES, MAX_OBSERVATIONS = 1536, 1152, 8192, 2048


def _points(value, name):
    result = np.asarray(value)
    if (np.ma.isMaskedArray(value) or result.dtype.kind not in "iuf" or result.ndim != 2
            or result.shape[1:] != (3,) or not len(result) or not np.isfinite(result).all()):
        raise ValueError(f"{name} requires finite nonempty [N,3] points")
    return result.astype(np.float64, copy=False)


def visible_region(mask, fraction):
    """Hide a left rectangular slab; its pixel fraction is measured, not assumed."""
    image = np.asarray(mask)
    if np.ma.isMaskedArray(mask) or image.dtype != np.bool_ or image.ndim != 2 or not image.any():
        raise ValueError("Require a nonempty foreground boolean image")
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (float, int))
            or not np.isfinite(fraction) or not 0 <= fraction < 1):
        raise ValueError("Occlusion fraction must be finite in [0,1)")
    hidden = np.zeros_like(image)
    if fraction:
        ys, xs = np.where(image)
        cumulative = np.r_[0, np.cumsum(np.count_nonzero(image, axis=0))]
        cut = int(np.argmin(np.abs(cumulative - fraction * len(xs))))
        hidden[ys.min():ys.max() + 1, :cut] = True
    visible = image & ~hidden
    if not visible.any():
        raise ValueError("Occluder removed every observed surface pixel")
    return visible, hidden, float(np.count_nonzero(image & hidden) / np.count_nonzero(image))


def camera_observations(depth, visible, camera_matrix, maximum=MAX_OBSERVATIONS):
    """Deterministic row-major samples, pinhole rays at x+.5,y+.5 times camera Z."""
    z, mask, matrix = np.asarray(depth), np.asarray(visible), np.asarray(camera_matrix)
    if any(np.ma.isMaskedArray(value) for value in (depth, visible, camera_matrix)):
        raise ValueError("Masked arrays cannot hide invalid depth, visibility or intrinsics")
    if (mask.dtype != np.bool_ or mask.ndim != 2 or z.shape != mask.shape
            or z.dtype.kind not in "iuf" or not mask.any()):
        raise ValueError("Depth and visible boolean mask must share a nonempty 2D grid")
    if (matrix.shape != (3, 3) or matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or matrix[0, 1] != 0
            or matrix[1, 0] != 0 or not np.array_equal(matrix[2], [0, 0, 1])):
        raise ValueError("Require finite positive zero-skew pinhole intrinsics")
    if type(maximum) is not int or not 3 <= maximum <= MAX_OBSERVATIONS:
        raise ValueError("Maximum observations must be an integer in [3,2048]")
    if not np.isfinite(z[mask]).all() or (z[mask] <= 0).any():
        raise ValueError("Every visible camera depth must be finite and positive")
    indices = np.flatnonzero(mask)
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]
    y, x = np.unravel_index(indices, mask.shape)
    rays = np.column_stack(((x + .5 - matrix[0, 2]) / matrix[0, 0],
                            (y + .5 - matrix[1, 2]) / matrix[1, 1], np.ones(len(x))))
    points = rays * z[y, x, None]
    return _points(points, "camera observations"), indices


def controlled_poses(perturbed=False):
    from scipy.spatial.transform import Rotation
    if type(perturbed) is not bool:
        raise ValueError("Perturbed pose mode must be explicit boolean")
    rotation = Rotation.from_rotvec([[.1, 0., 0.], [.2, .65, -.1], [-.3, 1.3, .2],
                                     [.25, 2., -.15], [-.2, 2.6, .3], [.35, 3.2, -.2]]).as_matrix()
    translation = np.column_stack((np.linspace(-.2, .2, 6), np.linspace(.12, -.12, 6), np.linspace(3., 4., 6)))
    if perturbed:
        rotation = Rotation.from_rotvec([.012, .016, 0.]).as_matrix()[None] @ rotation
        translation = translation + [.003, .004, 0.]
    return rotation, translation


def physical_metrics(predicted_samples, truth_samples, predicted_vertices, truth_vertices, faces):
    """Two-sided mean Euclidean Chamfer in metres, with no registration/normalization."""
    from scipy.spatial import cKDTree
    predicted, truth = _points(predicted_samples, "predicted samples"), _points(truth_samples, "truth samples")
    vertices, target = _points(predicted_vertices, "predicted vertices"), _points(truth_vertices, "truth vertices")
    indices = np.asarray(faces)
    if (vertices.shape != target.shape or indices.dtype.kind not in "iu" or indices.ndim != 2
            or indices.shape[1:] != (3,) or not len(indices) or indices.min() < 0 or indices.max() >= len(vertices)):
        raise ValueError("Require shared valid fixed mesh face indices")
    def volume(value):
        a, b, c = value[indices].transpose(1, 0, 2)
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    pred_volume, true_volume = volume(vertices), volume(target)
    if pred_volume <= 0 or true_volume <= 0 or not np.isfinite([pred_volume, true_volume]).all():
        raise ValueError("Require positive finite signed volume and original winding")
    forward = float(cKDTree(truth).query(predicted, workers=1)[0].mean())
    backward = float(cKDTree(predicted).query(truth, workers=1)[0].mean())
    return {"chamfer_m": (forward + backward) / 2, "pred_to_truth_m": forward,
            "truth_to_pred_m": backward, "signed_volume_m3": pred_volume,
            "relative_volume_error": abs(pred_volume / true_volume - 1)}


def fit_condition(surface, centroid, observations, rotations, translations):
    from world_reward.shape_fit import fit_shared_shape
    if len(observations) != 6 or np.shape(rotations) != (6, 3, 3) or np.shape(translations) != (6, 3):
        raise ValueError("Require six original observations and fixed poses")
    selected = list(FIT_FRAMES)
    return fit_shared_shape(surface, centroid, [observations[i] for i in selected],
                            rotations[selected], translations[selected])


def adopted_vertices(base, centroid, proposal):
    from world_reward.shape_model import apply_fixed_shape
    return (apply_fixed_shape(base, proposal.params5, centroid=centroid)
            if proposal.accepted else _points(base, "base vertices").copy())


def _experiment(torch):
    import trimesh
    from camera_render import raster_camera_mesh_batch, silhouette_iou
    from world_reward.shape_model import apply_fixed_shape
    records = []
    true_R, true_t = controlled_poses()
    for name, radii, truth_parameters in CASES:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.)
        mesh.vertices = np.asarray(mesh.vertices) * radii
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise RuntimeError("Procedural source must be closed, positively oriented")
        base, faces, pivot = np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy(), mesh.vertices.mean(0)
        samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)
        truth = apply_fixed_shape(base, truth_parameters, centroid=pivot)
        truth_samples = apply_fixed_shape(samples, truth_parameters, centroid=pivot)
        masks_gpu, depths_gpu = raster_camera_mesh_batch(truth[None] @ true_R.transpose(0, 2, 1) + true_t[:, None], faces, K)
        masks, depths = masks_gpu.cpu().numpy(), depths_gpu.cpu().numpy()
        del masks_gpu, depths_gpu
        for perturbed in (False, True):
            rotations, translations = controlled_poses(perturbed)
            base_masks, _ = raster_camera_mesh_batch(base[None] @ rotations.transpose(0, 2, 1) + translations[:, None], faces, K)
            base_masks = base_masks.cpu().numpy()
            for fraction in OCCLUSIONS:
                regions = [visible_region(mask, fraction) for mask in masks]
                observations = [camera_observations(depth, visible, K)[0]
                                for depth, (visible, _, _) in zip(depths, regions)]
                proposal = fit_condition(samples, pivot, observations, rotations, translations)
                predicted = adopted_vertices(base, pivot, proposal)
                predicted_samples = adopted_vertices(samples, pivot, proposal)
                pred_masks, _ = raster_camera_mesh_batch(predicted[None] @ rotations.transpose(0, 2, 1) + translations[:, None], faces, K)
                pred_masks = pred_masks.cpu().numpy()
                before, after, iou_before, iou_after = [], [], [], []
                for i, (visible, occluder, _) in enumerate(regions):
                    posed_truth = truth_samples @ true_R[i].T + true_t[i]
                    before.append(physical_metrics(samples @ rotations[i].T + translations[i], posed_truth, base, truth, faces))
                    after.append(physical_metrics(predicted_samples @ rotations[i].T + translations[i], posed_truth, predicted, truth, faces))
                    iou_before.append(silhouette_iou(base_masks[i] & ~occluder, visible))
                    iou_after.append(silhouette_iou(pred_masks[i] & ~occluder, visible))
                cd_before, cd_after = np.mean([v["chamfer_m"] for v in before]), np.mean([v["chamfer_m"] for v in after])
                held_before, held_after = np.mean(np.array(iou_before)[list(HELDOUT_FRAMES)]), np.mean(np.array(iou_after)[list(HELDOUT_FRAMES)])
                topology_ok = np.array_equal(faces, mesh.faces) and np.linalg.det(proposal.matrix) > 0
                gate = bool(cd_after <= cd_before * (1. if name == "symmetric_control" else .95)
                            + 1e-12 and held_after >= held_before * .95 and topology_ok)
                records.append({"case": name, "pose_condition": "perturbed_fixed_pose" if perturbed else "controlled_known_pose",
                                "synthetic_pose_oracle": True, "occlusion_requested": fraction,
                                "occlusion_actual_per_frame": [r[2] for r in regions], "fit_frames": list(FIT_FRAMES),
                                "observation_counts": [len(points) for points in observations],
                                "heldout_frames": list(HELDOUT_FRAMES), "proposal_accepted": proposal.accepted,
                                "params5": proposal.params5.tolist(), "optimizer_status": proposal.report["status"],
                                "optimizer_nfev": proposal.report["optimizer_nfev"], "chamfer_before_m": float(cd_before),
                                "chamfer_after_m": float(cd_after), "heldout_iou_before": float(held_before),
                                "heldout_iou_after": float(held_after), "translation_error_m": .005 if perturbed else 0.,
                                "rotation_error_rad": .02 if perturbed else 0., "poses_changed_by_fit": False,
                                "volume_relative_error": max(v["relative_volume_error"] for v in after),
                                "fixed_topology_positive_orientation": bool(topology_ok), "adoption_gate_pass": gate})
                print(json.dumps({"completed_conditions": len(records), **{key: records[-1][key] for key in (
                    "case", "pose_condition", "occlusion_requested", "optimizer_status", "chamfer_before_m",
                    "chamfer_after_m", "heldout_iou_after", "adoption_gate_pass")}}, allow_nan=False), flush=True)
    return records


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable WR_CODE_REVISION")
    output = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward")) / "results/shape-synthetic.json"
    if output.exists():
        raise FileExistsError("Synthetic experiment reports are frozen")
    import torch
    import pytorch3d
    if not torch.cuda.is_available():
        raise RuntimeError("Require CUDA; no local/CPU experiment fallback")
    started = time.perf_counter()
    records = _experiment(torch)
    torch.cuda.synchronize()
    passed = all(r["adoption_gate_pass"] for r in records)
    report = {"stage": "controlled_procedural_shared_shape", "status": "pass", "conditions": records,
              "shape_hypothesis_accepted": passed, "decision": "advance_to_RGB_validation" if passed else "stop_keep_experimental",
              "gate": {"deformed_case_minimum_CD_gain": .05, "zero_control_CD_nonregression": True, "maximum_heldout_IoU_regression": .05},
              "challenge_inputs_used": False, "challenge_ground_truth_used": False, "synthetic_ground_truth_used": True,
              "RGB_pipeline_validated": False, "challenge_performance_verified": False, "GT_alignment": False,
              "fit_inputs": "base_canonical_samples_visible_camera_Z_points_given_fixed_poses_only",
              "fit_configuration": {"shape_prior": .01, "robust_transition_m": .01, "max_nfev": 100},
              "geometry_units": "metres", "sample_seed": 0, "surface_samples": SURFACE_SAMPLES,
              "maximum_observations_per_frame": MAX_OBSERVATIONS, "image_size_hw": [HEIGHT, WIDTH],
              "K": K.tolist(), "own_truth_params5": {name: list(p) for name, _, p in CASES},
              "code_revision": revision, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "pytorch3d": pytorch3d.__version__, "gpu": torch.cuda.get_device_name(),
              "elapsed_seconds": time.perf_counter() - started}
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("stage", "status", "shape_hypothesis_accepted", "decision", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

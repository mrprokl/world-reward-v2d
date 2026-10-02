"""New procedural falsification of continuous/isotropic fixed-pose SPD fitting.

Both supplied-pose conditions use an explicit synthetic pose oracle. This is a
measurement-model ablation, NOT RGB inference, joint pose/shape selection or an
isolated proof of discretization bias: surface distance AND robust loss differ
from the previously rejected sampled fitter. Old cases/results stay untouched.
Own geometry alone generates camera-Z evidence and held-out evaluation. No
truth shape or held-out observation is passed to the fitter; all heavy work
remains in the isolated Azure runtime. Passing never authorizes adoption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from shape_synthetic import adopted_vertices, camera_observations, physical_metrics


MESH_FAMILIES = (
    ("ellipsoid", "icosphere_subdivisions_2", (.33, .19, .27)),
    ("rectangular_box", "closed_box", (.52, .26, .38)),
)
TRUTH_PARAMETERS = ((0., 0., 0., 0., 0.), (.045, -.025, .012, -.015, .008))
FIT_FRAMES, HELDOUT_FRAMES = (0, 2, 4), (1, 3, 5)
K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
WIDTH, HEIGHT, SURFACE_SAMPLES, MAX_OBSERVATIONS = 1536, 1152, 8192, 2048
SAMPLE_SEED, OCCLUSION_FRACTION = 19, .35
ROTATION_BIAS = (.008, -.006, .004)
TRANSLATION_BIAS = (.004, -.002, .003)
MAX_GATE_SECONDS = 120.


def controlled_poses(biased=False):
    """Six new immutable proper metric poses; neither condition fits their error."""
    from scipy.spatial.transform import Rotation
    if type(biased) is not bool:
        raise ValueError("Biased pose mode must be explicit boolean")
    rotations = Rotation.from_rotvec([
        [.15, -.2, .08], [-.18, .55, .12], [.28, 1.05, -.22],
        [-.12, 1.75, .26], [.32, 2.45, -.18], [-.24, 3., .11],
    ]).as_matrix()
    translations = np.array([
        [-.18, .08, 3.25], [-.1, -.05, 3.65], [.02, .12, 3.4],
        [.15, -.11, 4.15], [.2, .03, 3.8], [-.06, -.08, 4.],
    ])
    if biased:
        rotations = Rotation.from_rotvec(ROTATION_BIAS).as_matrix()[None] @ rotations
        translations = translations + TRANSLATION_BIAS
    return rotations, translations


def visible_region(mask):
    """Hide a bottom horizontal slab nearest the fixed35% foreground fraction."""
    image = np.asarray(mask)
    if np.ma.isMaskedArray(mask) or image.dtype != np.bool_ or image.ndim != 2 or not image.any():
        raise ValueError("Require nonempty foreground boolean mask")
    cumulative = np.r_[0, np.cumsum(np.count_nonzero(image, axis=1))]
    cut = int(np.argmin(np.abs(cumulative - (1 - OCCLUSION_FRACTION) * np.count_nonzero(image))))
    occluder = np.zeros_like(image)
    occluder[cut:] = True
    visible = image & ~occluder
    if not visible.any():
        raise ValueError("Fixed occluder removed all evidence")
    actual = float(np.count_nonzero(image & occluder) / np.count_nonzero(image))
    return visible, occluder, actual


def fit_condition(vertices, faces, pivot, observations, rotations, translations):
    from world_reward.shape_fit_continuous import fit_shared_shape_continuous
    if len(observations) != 6 or np.shape(rotations) != (6, 3, 3) or np.shape(translations) != (6, 3):
        raise ValueError("Require six original observations and fixed poses")
    selected = list(FIT_FRAMES)
    return fit_shared_shape_continuous(vertices, faces, pivot, [observations[i] for i in selected],
                                       rotations[selected], translations[selected])


def condition_gate(control, canonical_before, canonical_after, heldout_before, heldout_after,
                   fixed_topology, positive_volume):
    """Frozen physical canonical/held-out gates, never fitted-residual approval."""
    if any(type(value) is not bool for value in (control, fixed_topology, positive_volume)):
        raise ValueError("Control/topology/volume conditions must be explicit booleans")
    for value in (canonical_before, canonical_after, heldout_before, heldout_after):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError("Metrics require finite nonnegative real values")
    if max(heldout_before, heldout_after) > 1:
        raise ValueError("Silhouette IoU must be in [0,1]")
    if control and canonical_before != 0:
        raise ValueError("Correct canonical control must have an exactly zero baseline")
    if not control and canonical_before <= 0:
        raise ValueError("Deformed canonical baseline must be positive")
    canonical_ok = (canonical_after <= canonical_before + 1e-12 if control
                    else canonical_after <= .95 * canonical_before)
    return bool(canonical_ok and heldout_after >= .95 * heldout_before
                and fixed_topology and positive_volume)


def _budget(started):
    if time.perf_counter() - started > MAX_GATE_SECONDS:
        raise RuntimeError("Continuous shape gate exceeded frozen120s total budget")


def _experiment(torch, report, started):
    import trimesh
    from camera_render import raster_camera_mesh_batch, silhouette_iou
    from world_reward.shape_model import apply_fixed_shape
    true_R, true_t = controlled_poses()
    for name, kind, dimensions in MESH_FAMILIES:
        _budget(started)
        if kind == "icosphere_subdivisions_2":
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.)
            mesh.vertices = np.asarray(mesh.vertices) * dimensions
        elif kind == "closed_box":
            mesh = trimesh.creation.box(extents=dimensions)
        else:
            raise ValueError("Unknown frozen mesh family")
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise RuntimeError("Own mesh must be closed, consistently wound and positively oriented")
        base, faces, pivot = np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy(), mesh.vertices.mean(0)
        samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=SAMPLE_SEED)
        for truth_parameters in TRUTH_PARAMETERS:
            _budget(started)
            control = all(value == 0 for value in truth_parameters)
            # A correct control is the exact input, not a subtract/add-pivot
            # roundtrip that could manufacture a nonzero baseline in float64.
            truth = base.copy() if control else apply_fixed_shape(base, truth_parameters, centroid=pivot)
            truth_samples = samples.copy() if control else apply_fixed_shape(samples, truth_parameters, centroid=pivot)
            truth_masks_gpu, depths_gpu = raster_camera_mesh_batch(
                truth[None] @ true_R.transpose(0, 2, 1) + true_t[:, None], faces, K)
            truth_masks, depths = truth_masks_gpu.cpu().numpy(), depths_gpu.cpu().numpy()
            del truth_masks_gpu, depths_gpu
            regions = [visible_region(mask) for mask in truth_masks]
            observations = [camera_observations(depth, visible, K)[0]
                            for depth, (visible, _, _) in zip(depths, regions, strict=True)]
            for biased in (False, True):
                _budget(started)
                report["active_condition"] = {
                    "mesh_family": name, "shape_condition": "correct_control" if control else "deformed",
                    "pose_condition": "biased_fixed_pose" if biased else "known_fixed_pose",
                }
                rotations, translations = controlled_poses(biased)
                base_masks, _ = raster_camera_mesh_batch(
                    base[None] @ rotations.transpose(0, 2, 1) + translations[:, None], faces, K)
                base_masks = base_masks.cpu().numpy()
                proposal = fit_condition(base, faces, pivot, observations, rotations, translations)
                predicted = adopted_vertices(base, pivot, proposal)
                predicted_samples = adopted_vertices(samples, pivot, proposal)
                predicted_mesh = trimesh.Trimesh(vertices=predicted, faces=faces, process=False)
                fixed_topology = bool(np.array_equal(faces, predicted_mesh.faces)
                                      and predicted_mesh.is_watertight and predicted_mesh.is_winding_consistent
                                      and np.isfinite(np.linalg.det(proposal.matrix)) and np.linalg.det(proposal.matrix) > 0)
                positive_volume = bool(np.isfinite(predicted_mesh.volume) and predicted_mesh.volume > 0)
                canonical_before = physical_metrics(samples, truth_samples, base, truth, faces)
                canonical_after = physical_metrics(predicted_samples, truth_samples, predicted, truth, faces)
                predicted_masks, _ = raster_camera_mesh_batch(
                    predicted[None] @ rotations.transpose(0, 2, 1) + translations[:, None], faces, K)
                predicted_masks = predicted_masks.cpu().numpy()
                camera_before, camera_after, iou_before, iou_after = [], [], [], []
                for i, (visible, occluder, _) in enumerate(regions):
                    posed_truth = truth_samples @ true_R[i].T + true_t[i]
                    camera_before.append(physical_metrics(
                        samples @ rotations[i].T + translations[i], posed_truth, base, truth, faces)["chamfer_m"])
                    camera_after.append(physical_metrics(
                        predicted_samples @ rotations[i].T + translations[i], posed_truth, predicted, truth, faces)["chamfer_m"])
                    iou_before.append(silhouette_iou(base_masks[i] & ~occluder, visible))
                    iou_after.append(silhouette_iou(predicted_masks[i] & ~occluder, visible))
                held_before = float(np.mean(np.asarray(iou_before)[list(HELDOUT_FRAMES)]))
                held_after = float(np.mean(np.asarray(iou_after)[list(HELDOUT_FRAMES)]))
                gate = condition_gate(control, canonical_before["chamfer_m"], canonical_after["chamfer_m"],
                                      held_before, held_after, fixed_topology, positive_volume)
                record = {
                    **report["active_condition"], "synthetic_pose_oracle": True,
                    "synthetic_truth_shape_used_for_observation_and_evaluation_only": True,
                    "occlusion_requested": OCCLUSION_FRACTION,
                    "occlusion_actual_per_frame": [region[2] for region in regions],
                    "fit_frames": list(FIT_FRAMES), "heldout_frames": list(HELDOUT_FRAMES),
                    "observation_counts": [len(points) for points in observations],
                    "mesh_vertices": len(base), "mesh_faces": len(faces),
                    "proposal_accepted_training_only": proposal.accepted,
                    "candidate_applied_for_evaluation": proposal.accepted,
                    "abstention_kept_base": not proposal.accepted,
                    "params5": proposal.params5.tolist(), "optimizer_status": proposal.report["status"],
                    "optimizer_objective_evaluations": proposal.report["objective_evaluations"],
                    "canonical_chamfer_before_m": canonical_before["chamfer_m"],
                    "canonical_chamfer_after_m": canonical_after["chamfer_m"],
                    "camera_chamfer_before_m": float(np.mean(camera_before)),
                    "camera_chamfer_after_m": float(np.mean(camera_after)),
                    "heldout_iou_before": held_before, "heldout_iou_after": held_after,
                    "heldout_iou_before_per_frame": [iou_before[i] for i in HELDOUT_FRAMES],
                    "heldout_iou_after_per_frame": [iou_after[i] for i in HELDOUT_FRAMES],
                    "fixed_topology_positive_orientation": fixed_topology, "positive_volume": positive_volume,
                    "volume_relative_error": canonical_after["relative_volume_error"],
                    "poses_changed_by_fit": False, "pose_uncertainty_modeled": False,
                    "rotation_bias_rad": float(np.linalg.norm(ROTATION_BIAS)) if biased else 0.,
                    "translation_bias_m": float(np.linalg.norm(TRANSLATION_BIAS)) if biased else 0.,
                    "falsification_gate_pass": gate, "adoption_authorized": False,
                }
                report["conditions"].append(record)
                print(json.dumps({"completed_conditions": len(report["conditions"]), **{key: record[key] for key in (
                    "mesh_family", "shape_condition", "pose_condition", "optimizer_status",
                    "canonical_chamfer_before_m", "canonical_chamfer_after_m", "heldout_iou_after", "falsification_gate_pass",
                )}}, allow_nan=False), flush=True)
                _budget(started)
    report.pop("active_condition", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable source revision")
    root = Path(os.environ["WR_ROOT"])
    output = root / "results/shape-continuous-gate.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Continuous shape reports are frozen")
    import world_reward.continuous_surface as primitive
    import world_reward.shape_fit_continuous as fitter
    report = {
        "stage": "own_procedural_continuous_isotropic_fixed_pose_shape", "status": "fail", "conditions": [],
        "shape_hypothesis_accepted": False, "decision": "stop_keep_experimental", "adoption_authorized": False,
        "challenge_inputs_used": False, "challenge_ground_truth_used": False, "models_used": False,
        "synthetic_ground_truth_used": True, "synthetic_pose_oracle": True, "RGB_pipeline_validated": False,
        "challenge_performance_verified": False, "GT_alignment": False, "joint_pose_shape_selection_validated": False,
        "ablation_changes": ["sampled_surface_to_continuous_triangles", "coordinatewise_to_isotropic_robust_loss"],
        "isolated_discretization_causal_claim": False, "prior_failed_cases_retuned": False,
        "fit_inputs": "base_canonical_mesh_visible_camera_Z_points_given_fixed_synthetic_poses_only",
        "fit_configuration": {"shape_prior": .01, "robust_transition_m": .01, "max_evaluations": 100, "point_chunk_size": 64},
        "gate": {"deformed_minimum_canonical_CD_gain": .05, "correct_control_CD_tolerance_m": 1e-12,
                 "maximum_relative_heldout_IoU_regression": .05, "fixed_topology_positive_volume_required": True},
        "geometry_units": "metres", "surface_samples": SURFACE_SAMPLES, "sample_seed": SAMPLE_SEED,
        "maximum_observations_per_frame": MAX_OBSERVATIONS, "image_size_hw": [HEIGHT, WIDTH], "K": K.tolist(),
        "mesh_families": MESH_FAMILIES, "own_truth_params5": TRUTH_PARAMETERS,
        "occlusion": "bottom_horizontal_slab", "occlusion_requested": OCCLUSION_FRACTION,
        "max_gate_seconds": MAX_GATE_SECONDS, "code_revision": revision, "network": "none",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primitive_sha256": hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest(),
        "fitter_sha256": hashlib.sha256(Path(fitter.__file__).read_bytes()).hexdigest(),
    }
    started = time.perf_counter()
    try:
        gate_path = root / "results/continuous-surface-gate.json"
        primitive_gate = json.loads(gate_path.read_text())
        expected = {"stage": "own_continuous_surface_analytic_cuda", "status": "pass",
                    "own_procedural_inputs_only": True, "challenge_inputs_used": False,
                    "primitive_sha256": report["primitive_sha256"]}
        if any(primitive_gate.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Require the passing analytic gate for these exact primitive bytes")
        report["primitive_gate_sha256"] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no local/CPU experiment fallback")
        report.update({"torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
        _experiment(torch, report, started)
        torch.cuda.synchronize()
        _budget(started)
        if len(report["conditions"]) != 8:
            raise RuntimeError("Require all eight frozen conditions, no filtered failures")
        passed = all(record["falsification_gate_pass"] for record in report["conditions"])
        report.update({"status": "pass", "shape_hypothesis_accepted": passed,
                       "decision": "advance_to_nonoracle_validation" if passed else "stop_keep_experimental"})
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in (
            "stage", "status", "shape_hypothesis_accepted", "decision", "elapsed_seconds")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

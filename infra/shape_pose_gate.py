"""Untouched own synthetic M0/M1 nuisance-pose experiment, never RGB validation.

Checkerboard training/held-out PIXELS share three supplied views. They are not
independent temporal blocks or unseen-view pose evaluation. Both exact and
biased initializations derive from explicitly declared synthetic pose oracles.
All geometry/cameras/shape controls are new; previous failed gates stay frozen.
Canonical truth evaluates outcomes only and never selects M0/M1 or fits poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from shape_synthetic import camera_observations, physical_metrics
from world_reward.shape_model import apply_fixed_shape
from world_reward.shape_selection import select_shape_candidate


FAMILIES = (("ellipsoid", (.31, .17, .29)), ("box", (.48, .30, .34)))
THETAS = ((0., 0., 0., 0., 0.), (.035, -.02, .009, -.011, .006))
IMAGE_HW, K = (1152, 1536), np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
MAX_SECONDS, SURFACE_SAMPLES, SURFACE_SEED, PIXEL_POINTS = 120., 8192, 37, 128


def supplied_poses(biased=False):
    from scipy.spatial.transform import Rotation
    if type(biased) is not bool: raise ValueError("Pose bias must be explicit boolean")
    R = Rotation.from_rotvec([[.22, .18, -.09], [-.27, 1.12, .17], [.16, 2.34, -.21]]).as_matrix()
    t = np.array([[-.13, .09, 3.55], [.11, -.07, 3.95], [.03, .14, 4.3]])
    if biased:
        R = Rotation.from_rotvec([.012, -.004, .007]).as_matrix()[None] @ R
        t = t + [.003, -.004, .002]
    return R, t


def split_visible_pixels(mask):
    """Fixed top horizontal quarter occlusion; checkerboard crossfit within view."""
    image = np.asarray(mask)
    if np.ma.isMaskedArray(mask) or image.dtype != np.bool_ or image.ndim != 2 or not image.any():
        raise ValueError("Require nonempty boolean silhouette")
    cumulative = np.r_[0, np.cumsum(np.count_nonzero(image, axis=1))]
    cut = int(np.argmin(np.abs(cumulative - .25 * np.count_nonzero(image))))
    occluder = np.zeros_like(image); occluder[:cut] = True
    visible = image & ~occluder
    y, x = np.indices(image.shape)
    training = visible & ((x + y) % 2 == 0)
    heldout = visible & ~training
    if min(np.count_nonzero(training), np.count_nonzero(heldout)) < 3:
        raise ValueError("Each split must retain at least three observed pixels")
    return training, heldout, occluder


def selected_parameters(models, selection):
    """No truth argument: conservative M0 unless the proxy returns an M1 proposal."""
    return models[1] if selection.decision == "shape_pose_proposal" else models[0]


def outcome_gate(control, before_cd, after_cd, topology, positive_volume):
    """Synthetic physical falsification only; does not choose the fitted model."""
    if any(type(value) is not bool for value in (control, topology, positive_volume)):
        raise ValueError("Explicit control/topology/volume flags required")
    if not np.isfinite([before_cd, after_cd]).all() or min(before_cd, after_cd) < 0:
        raise ValueError("Require finite nonnegative canonical Chamfer")
    if control and before_cd != 0: raise ValueError("Correct control baseline must be exactly zero")
    if not control and before_cd <= 0: raise ValueError("Deformed baseline must be positive")
    return bool(topology and positive_volume and (after_cd <= 1e-12 if control else after_cd <= .95 * before_cd))


def _experiment(torch, report, deadline):
    import trimesh
    from camera_render import raster_camera_mesh_batch, silhouette_iou
    from world_reward.shape_pose_fit import fit_nested_shape_pose
    true_R, true_t = supplied_poses()
    for family, dimensions in FAMILIES:
        mesh = trimesh.creation.icosphere(subdivisions=1) if family == "ellipsoid" else trimesh.creation.box(extents=dimensions)
        if family == "ellipsoid": mesh.vertices *= dimensions
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise RuntimeError("Own source mesh must have positive closed oriented topology")
        base, faces, pivot = mesh.vertices.copy(), mesh.faces.copy(), mesh.vertices.mean(0)
        samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=SURFACE_SEED)
        for theta in THETAS:
            control = all(value == 0 for value in theta)
            truth = base.copy() if control else apply_fixed_shape(base, theta, centroid=pivot)
            truth_samples = samples.copy() if control else apply_fixed_shape(samples, theta, centroid=pivot)
            masks, depths = raster_camera_mesh_batch(truth[None] @ true_R.transpose(0, 2, 1) + true_t[:, None], faces, K)
            masks, depths = masks.cpu().numpy(), depths.cpu().numpy()
            splits = [split_visible_pixels(mask) for mask in masks]
            training = [camera_observations(depth, split[0], K, maximum=PIXEL_POINTS)[0] for depth, split in zip(depths, splits, strict=True)]
            heldout = [camera_observations(depth, split[1], K, maximum=PIXEL_POINTS)[0] for depth, split in zip(depths, splits, strict=True)]
            canonical_before = physical_metrics(samples, truth_samples, base, truth, faces)
            for biased in (False, True):
                report["active_condition"] = {"family": family, "correct_control": control, "biased_supplied_pose": biased}
                R, t = supplied_poses(biased)
                models = fit_nested_shape_pose(base, faces, pivot, training, heldout, R, t,
                                               heldout_pixels_disjoint=True, deadline_monotonic=deadline)
                ious, proposal_vertices = [], []
                for model in models:
                    v = base.copy() if np.array_equal(model.params5, np.zeros(5)) else apply_fixed_shape(base, model.params5, centroid=pivot)
                    proposal_vertices.append(v)
                    predicted_masks, _ = raster_camera_mesh_batch(v[None] @ model.rotations.transpose(0, 2, 1) + model.translations[:, None], faces, K)
                    predicted_masks = predicted_masks.cpu().numpy()
                    yy, xx = np.indices(masks.shape[1:]); heldout_checkerboard = (xx + yy) % 2 == 1
                    ious.append(np.array([silhouette_iou(predicted_masks[i] & heldout_checkerboard & ~splits[i][2], splits[i][1]) for i in range(3)]))
                nuisance, joint = models
                eig = joint.report["data_schur_eigenvalues"]
                # Known local numerical ambiguity is rejected. Global pose mode
                # absence is not certified by this single-basin experiment.
                # A single basin cannot certify the absence of global modes.
                # Unknown is conservatively treated as ambiguous, never False.
                ambiguity = True
                selection = select_shape_candidate(nuisance.heldout_losses_m2, joint.heldout_losses_m2,
                                                   ious[0], ious[1], np.zeros(5) if eig is None else eig,
                                                   saturated_bounds=bool(joint.report["shape_bounds_saturated"] or joint.report["pose_bounds_saturated"] or nuisance.report["pose_bounds_saturated"]),
                                                   pose_modes_ambiguous=ambiguity, heldout_pixels_disjoint=True)
                model = selected_parameters(models, selection)
                chosen_index = 1 if model is joint else 0
                chosen = proposal_vertices[chosen_index]
                chosen_samples = samples.copy() if np.array_equal(model.params5, np.zeros(5)) else apply_fixed_shape(samples, model.params5, centroid=pivot)
                metric = physical_metrics(chosen_samples, truth_samples, chosen, truth, faces)
                raw_samples = samples.copy() if np.array_equal(joint.params5, np.zeros(5)) else apply_fixed_shape(samples, joint.params5, centroid=pivot)
                raw_metric = physical_metrics(raw_samples, truth_samples, proposal_vertices[1], truth, faces)
                checked = trimesh.Trimesh(chosen, faces, process=False)
                topology = bool(np.array_equal(checked.faces, faces) and checked.is_watertight and checked.is_winding_consistent)
                volume = bool(np.isfinite(checked.volume) and checked.volume > 0)
                raw_checked = trimesh.Trimesh(proposal_vertices[1], faces, process=False)
                raw_topology = bool(np.array_equal(raw_checked.faces, faces) and raw_checked.is_watertight
                                    and raw_checked.is_winding_consistent)
                raw_volume = bool(np.isfinite(raw_checked.volume) and raw_checked.volume > 0)
                selected_pass = outcome_gate(control, canonical_before["chamfer_m"], metric["chamfer_m"], topology, volume)
                # Measure raw M1 separately: fallback M0 must not conceal a bad
                # nuisance-adjusted measurement model behind a trivial safety win.
                raw_pass = outcome_gate(control, canonical_before["chamfer_m"], raw_metric["chamfer_m"], raw_topology, raw_volume)
                passed = bool(raw_pass and (selected_pass if control else True))
                record = {**report["active_condition"], "synthetic_pose_oracle": True,
                          "training_points": [len(p) for p in training], "heldout_points": [len(p) for p in heldout],
                          "heldout_pixels_disjoint": True, "validation_unit": "within_supplied_view_pixel_crossfit_proxy",
                          "independent_temporal_blocks_verified": False, "unseen_view_pose_transfer_verified": False,
                          "models": [m.report for m in models], "selection": selection.report,
                          "global_pose_modes_checked": False, "unknown_pose_modes_forced_M0": True,
                          "params5_M1": joint.params5.tolist(),
                          "heldout_mse_M0_m2": nuisance.heldout_losses_m2.tolist(), "heldout_mse_M1_m2": joint.heldout_losses_m2.tolist(),
                          "heldout_pixel_iou_M0": ious[0].tolist(), "heldout_pixel_iou_M1": ious[1].tolist(),
                          "canonical_before_m": canonical_before["chamfer_m"], "canonical_selected_after_m": metric["chamfer_m"],
                          "canonical_raw_M1_after_m": raw_metric["chamfer_m"], "selected_model": selection.decision,
                          "fixed_positive_topology": topology and volume, "selected_model_outcome_gate_pass": selected_pass,
                          "raw_M1_outcome_gate_pass": raw_pass,
                          "proposal_correct_shape_nonregression": raw_pass if control else None,
                          "falsification_gate_pass": passed,
                          "adoption_authorized": False}
                from scipy.spatial.transform import Rotation
                record["pose_translation_error_rms_m_M0_M1"] = [float(np.sqrt(np.mean(np.sum((m.translations - true_t)**2, axis=1)))) for m in models]
                record["pose_rotation_error_rms_rad_M0_M1"] = [float(np.sqrt(np.mean(Rotation.from_matrix(m.rotations @ true_R.transpose(0, 2, 1)).magnitude()**2))) for m in models]
                report["conditions"].append(record)
                print(json.dumps({"completed_conditions": len(report["conditions"]), **{key: record[key] for key in (
                    "family", "correct_control", "biased_supplied_pose", "selected_model", "canonical_before_m",
                    "canonical_selected_after_m", "canonical_raw_M1_after_m", "falsification_gate_pass")}}, allow_nan=False), flush=True)
                if time.perf_counter() > deadline: raise TimeoutError("Nested synthetic gate exceeded120s")
    report.pop("active_condition", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision): raise RuntimeError("Require immutable source revision")
    root = Path(os.environ["WR_ROOT"]); output = root / "results/shape-pose-gate.json"
    if output.exists() or output.is_symlink(): raise FileExistsError("Nested shape/pose report is frozen")
    import world_reward.shape_pose_fit as fitter
    import world_reward.continuous_surface as primitive
    report = {"stage": "own_nested_pose_nuisance_shape_pixel_crossfit", "status": "fail", "conditions": [],
              "shape_hypothesis_accepted": False, "decision": "stop_keep_experimental", "adoption_authorized": False,
              "challenge_inputs_used": False, "models_used": False, "synthetic_ground_truth_used": True,
              "synthetic_pose_oracle": True, "GT_alignment": False, "RGB_pipeline_validated": False,
              "challenge_performance_verified": False, "prior_failed_cases_retuned": False,
              "validation_unit": "within_supplied_view_pixel_crossfit_proxy", "statistical_calibration_verified": False,
              "independent_temporal_blocks_verified": False, "unseen_view_pose_transfer_verified": False,
              "global_pose_modes_checked": False, "max_gate_seconds": MAX_SECONDS, "code_revision": revision,
              "families": FAMILIES, "truth_controls": THETAS, "sample_seed": SURFACE_SEED,
              "surface_samples": SURFACE_SAMPLES, "points_per_split_per_view": PIXEL_POINTS, "K": K.tolist(),
              "image_size_hw": IMAGE_HW, "network": "none", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fitter_sha256": hashlib.sha256(Path(fitter.__file__).read_bytes()).hexdigest(),
              "primitive_sha256": hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        primitive_gate = json.loads((root / "results/continuous-surface-gate.json").read_text())
        if (primitive_gate.get("status") != "pass" or primitive_gate.get("stage") != "own_continuous_surface_analytic_cuda"
                or primitive_gate.get("primitive_sha256") != report["primitive_sha256"]
                or primitive_gate.get("challenge_inputs_used") is not False):
            raise RuntimeError("Require passing exact own continuous primitive gate")
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no local CPU fallback")
        report.update({"torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
        _experiment(torch, report, started + MAX_SECONDS)
        torch.cuda.synchronize()
        if len(report["conditions"]) != 8 or time.perf_counter() - started > MAX_SECONDS:
            raise RuntimeError("All8 conditions must finish inside frozen120s")
        passed = all(c["falsification_gate_pass"] for c in report["conditions"])
        report.update({"status": "pass", "shape_hypothesis_accepted": passed,
                       "decision": "advance_to_nonoracle_validation" if passed else "stop_keep_experimental"})
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle: handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "status", "decision", "shape_hypothesis_accepted", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()

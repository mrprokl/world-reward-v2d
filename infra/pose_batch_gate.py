"""Whole-candidate CUDA scheduling parity/throughput, own procedural inputs only.

No challenge inputs or pose-accuracy metric. Batch size is the only changed
argument: the same 24 hypotheses, original pixels, samples and ICP defaults
must produce exactly the same candidates, rejections and selected source.
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

from camera_render import raster_camera_mesh_batch
from pose_candidates import evaluate_pose_candidates
from shape_synthetic import camera_observations


K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
WIDTH, HEIGHT = 1536, 1152
RADII = (.35, .21, .28)
SURFACE_SAMPLES, OBSERVATIONS, SAMPLE_SEED = 8192, 2048, 0
MINIMUM_SPEEDUP, MAXIMUM_SECONDS = 1.3, 120.
PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"


def controlled_poses():
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_rotvec([[.1, .2, -.1], [-.2, .8, .15], [.3, 1.3, -.2]]).as_matrix()
    translations = np.array([[-.15, .10, 3.2], [0., -.08, 3.5], [.18, .06, 3.8]])
    return rotations, translations


def orientation_seeds():
    """Generic reference plus the unchanged 24 proper octahedral rotations."""
    from scipy.spatial.transform import Rotation
    reference = Rotation.from_rotvec([.17, -.23, .11]).as_matrix()
    return [reference @ hypothesis for hypothesis in Rotation.create_group("O").as_matrix()]


def candidate_signature(candidates, rejected):
    """Full exact JSON includes original candidate order, rejection reasons and best."""
    if not candidates:
        raise RuntimeError(f"No supported procedural pose candidate: {rejected}")
    indices = [row["hypothesis_index"] for row in candidates + rejected]
    if sorted(indices) != list(range(24)):
        raise RuntimeError("Every hypothesis must retain exactly one original candidate/rejection slot")
    best = max(candidates, key=lambda c: (c["selected_silhouette_iou"],
                                         -c["selected_depth_residual_m"], -c["hypothesis_index"]))
    value = {"candidates": candidates, "rejected_candidates": rejected, "selected": best}
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), best["hypothesis_index"]


def trial_orders():
    return ((1, 8), (8, 1), (1, 8))


def _deadline(started):
    if time.perf_counter() - started >= MAXIMUM_SECONDS:
        raise RuntimeError("Predeclared whole gate runtime exceeded 120 seconds")


def _inputs():
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.)
    mesh.vertices = np.asarray(mesh.vertices) * RADII
    if (len(mesh.vertices) != 642 or len(mesh.faces) != 1280 or not mesh.is_watertight
            or not mesh.is_winding_consistent or mesh.volume <= 0):
        raise RuntimeError("Procedural source must retain fixed closed 642v/1280f positive topology")
    vertices, faces = np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy()
    sampled, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=SAMPLE_SEED)
    rotations, translations = controlled_poses()
    posed = np.stack([vertices @ rotation.T + translation
                      for rotation, translation in zip(rotations, translations, strict=True)])
    masks_gpu, depth_gpu = raster_camera_mesh_batch(posed, faces, K, WIDTH, HEIGHT)
    masks, depths = masks_gpu.cpu().numpy(), depth_gpu.cpu().numpy()
    del masks_gpu, depth_gpu
    seeds, inputs, summaries = orientation_seeds(), [], []
    for frame, (depth, mask) in enumerate(zip(depths, masks, strict=True)):
        points, _ = camera_observations(depth, mask, K)
        if len(points) != OBSERVATIONS:
            raise RuntimeError("Procedural visible front surface must support exactly 2048 sampled pixels")
        seed_translations = [np.median(points, axis=0) - mesh.centroid @ rotation.T for rotation in seeds]
        inputs.append((vertices, faces, sampled, points, mask, K, seeds, seed_translations))
        summaries.append({"frame_index": frame, "visible_pixels": int(mask.sum()), "sampled_observations": len(points)})
    return inputs, summaries


def _run(torch, report, started):
    inputs, report["frames"] = _inputs()
    expected = []
    report["parity"] = []
    for frame, values in enumerate(inputs):
        _deadline(started)
        scalar, best = candidate_signature(*evaluate_pose_candidates(*values, render_batch_size=1))
        _deadline(started)
        batch, batch_best = candidate_signature(*evaluate_pose_candidates(*values, render_batch_size=8))
        exact = scalar == batch and best == batch_best
        report["parity"].append({"frame_index": frame, "complete_candidate_json_equal": exact,
                                 "selected_hypothesis_index": best, "batch_selected_hypothesis_index": batch_best,
                                 "scalar_candidates_sha256": hashlib.sha256(scalar.encode()).hexdigest(),
                                 "batch_candidates_sha256": hashlib.sha256(batch.encode()).hexdigest()})
        if not exact:
            raise RuntimeError(f"Whole-candidate scalar/batch parity failed at own frame {frame}")
        expected.append(scalar)
        print(json.dumps({"phase": "parity", **report["parity"][-1]}, allow_nan=False), flush=True)
    torch.cuda.synchronize()
    report["exact_candidate_rejection_selection_parity"] = True
    durations = report["synchronized_whole_candidate_trials_seconds"] = {"scalar": [], "batch": []}
    for trial, order in enumerate(trial_orders()):
        for batch_size in order:
            torch.cuda.synchronize()
            began = time.perf_counter()
            for frame, values in enumerate(inputs):
                _deadline(started)
                signature, _ = candidate_signature(*evaluate_pose_candidates(*values, render_batch_size=batch_size))
                if signature != expected[frame]:
                    raise RuntimeError(f"Reused-input candidate signature changed in trial {trial} at frame {frame}")
            torch.cuda.synchronize()
            duration = time.perf_counter() - began
            mode = "scalar" if batch_size == 1 else "batch"
            durations[mode].append(duration)
            print(json.dumps({"phase": "throughput", "trial": trial, "batch_size": batch_size,
                              "whole_three_frame_seconds": duration}), flush=True)
            _deadline(started)
    speedup = float(np.median(durations["scalar"]) / np.median(durations["batch"]))
    report["median_whole_candidate_speedup"] = speedup
    report["throughput_hypothesis_accepted"] = bool(speedup >= MINIMUM_SPEEDUP)
    if speedup < MINIMUM_SPEEDUP:
        raise RuntimeError(f"Whole-candidate speedup {speedup:.6g} below predeclared 1.3; scheduling not adopted")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable WR_CODE_REVISION")
    output = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward")) / "results/pose-batch-gate.json"
    if output.exists():
        raise FileExistsError("Pose batch gate reports are frozen")
    import torch
    import pytorch3d
    if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require CUDA and audited PyTorch3D 0.7.9")
    report = {"stage": "whole_pose_candidate_cuda_batch_parity_throughput", "status": "fail",
              "code_revision": revision, "torch": torch.__version__, "pytorch3d": pytorch3d.__version__,
              "expected_pytorch3d_revision": PYTORCH3D_REVISION, "gpu": torch.cuda.get_device_name(),
              "own_procedural_inputs": True, "synthetic_known_poses_used_to_render_observations": True,
              "seed_rotations_use_truth_pose": False, "challenge_inputs_used": False, "models_used": False,
              "challenge_ground_truth_used": False, "pose_accuracy_verified": False,
              "challenge_performance_verified": False, "candidate_execution_schedule_only": True,
              "batch_sizes": [1, 8], "hypotheses_per_frame": 24, "mesh_vertices": 642, "mesh_faces": 1280,
              "radii_m": list(RADII), "surface_samples": SURFACE_SAMPLES, "observation_samples": OBSERVATIONS,
              "sample_seed": SAMPLE_SEED, "image_size_hw": [HEIGHT, WIDTH], "K": K.tolist(),
              "predeclared_minimum_useful_speedup": MINIMUM_SPEEDUP, "predeclared_maximum_elapsed_seconds": MAXIMUM_SECONDS,
              "trial_batch_orders": [list(order) for order in trial_orders()],
              "geometry_units": "metres", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pose_candidate_source_sha256": hashlib.sha256(Path(__file__).with_name("pose_candidates.py").read_bytes()).hexdigest()}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        _run(torch, report, started)
        _deadline(started)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

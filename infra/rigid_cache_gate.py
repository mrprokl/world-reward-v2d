"""Own procedural CPU gate for experimental canonical-tree ICP, not adoption.

Compare all 24 fixed orientation hypotheses over three noisy partial views,
then median throughput over three trials. All source geometry, transforms,
noise and outliers are created here. No challenge input, labels, models or GPU
are read. Exact status/iteration/inlier parity is mandatory; numerical gates
are declared before execution and never relaxed on a failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from world_reward.data import sha256
from world_reward.rigid_alignment import align_observed_points
from world_reward.rigid_alignment_cached import align_observed_points_cached


NUMERICAL_ATOL = 1e-6
MINIMUM_SPEEDUP = 1.3
MAX_GATE_SECONDS = 60.
SURFACE_POINTS = 8192
OBSERVED_POINTS = 2048
TRIALS = 3


def procedural_cases(surface_count=SURFACE_POINTS, observation_count=OBSERVED_POINTS):
    """Deterministic asymmetric surface; own partial views/noise/12.5% outliers."""
    rng = np.random.default_rng(109)
    directions = rng.normal(size=(surface_count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    mesh = directions * [.4, .25, .17]
    mesh[:, 1] += .11 * mesh[:, 0] ** 2
    mesh[:, 2] += .015 * np.sin(mesh[:, 0] * 9) * np.cos(mesh[:, 1] * 13)
    hypotheses = Rotation.create_group("O").as_matrix()
    views = []
    for view_index, rotvec in enumerate(([.1, -.2, .05], [-.15, .12, -.2], [.2, .15, .1])):
        truth_R = Rotation.from_rotvec(rotvec).as_matrix()
        truth_t = np.array([.07 * view_index, -.04 * view_index, 3. + .1 * view_index])
        visible = mesh[mesh[:, 2] > np.quantile(mesh[:, 2], .45)]
        inlier_count = observation_count - observation_count // 8
        selected = visible[rng.choice(len(visible), inlier_count, replace=inlier_count > len(visible))]
        observed = selected @ truth_R.T + truth_t + rng.normal(0, .002, (inlier_count, 3))
        outliers = rng.uniform(-.7, .7, (observation_count - inlier_count, 3)) + truth_t
        observed = np.concatenate([observed, outliers])
        seeds = [(truth_R @ hypothesis, truth_t + [.013, -.011, .009]) for hypothesis in hypotheses]
        views.append((observed, seeds))
    return mesh, views


def compare_results(original, cached) -> dict:
    """Strict discrete ABI; bounded geometry/residual differences, no repair."""
    for name in ("status", "inliers", "iterations"):
        if getattr(original, name) != getattr(cached, name):
            raise ValueError(f"Cached ICP changed exact {name} parity")
    errors = {
        "rotation_max_abs_error": float(np.max(np.abs(original.rotation - cached.rotation))),
        "translation_error_m": float(np.linalg.norm(original.translation - cached.translation)),
        "initial_residual_error_m": abs(float(original.initial_residual) - float(cached.initial_residual)),
        "final_residual_error_m": abs(float(original.final_residual) - float(cached.final_residual)),
    }
    if any(not np.isfinite(error) or error > NUMERICAL_ATOL for error in errors.values()):
        raise ValueError("Cached ICP exceeds the predeclared 1e-6 pose/residual parity gate")
    return errors


def compare_queries(mesh, observed, rotation, translation, canonical_tree) -> dict:
    """Compare exact NN/trim slots and distances at the same supplied pose."""
    original_dist, original_indices = cKDTree(mesh @ rotation.T + translation).query(observed, k=1)
    cached_dist, cached_indices = canonical_tree.query((observed - translation) @ rotation, k=1)
    if not np.isfinite(original_dist).all() or not np.isfinite(cached_dist).all():
        raise ValueError("Nearest-neighbour parity is nonfinite")
    if not np.array_equal(original_indices, cached_indices):
        raise ValueError("Cached query changed exact nearest-neighbour index parity")
    keep = int(np.floor(.8 * len(observed)))
    if not np.array_equal(np.argsort(original_dist, kind="stable")[:keep],
                          np.argsort(cached_dist, kind="stable")[:keep]):
        raise ValueError("Cached query changed exact trimmed-observation index parity")
    error = float(np.max(np.abs(original_dist - cached_dist)))
    if error > NUMERICAL_ATOL:
        raise ValueError("Cached query exceeds the predeclared distance parity gate")
    return {"max_nearest_distance_error_m": error, "nearest_indices_exact": True, "trim_indices_exact": True}


def _deadline(started):
    if time.perf_counter() - started > MAX_GATE_SECONDS:
        raise RuntimeError("Procedural CPU gate exceeded the predeclared 60-second budget")


def run_gate(diagnostics: dict | None = None) -> dict:
    """Remote CPU only; gate failures abort timing/adoption rather than relax."""
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("Mutable gate diagnostics must be a dict")
    started = time.perf_counter()
    mesh, views = procedural_cases()
    tree = cKDTree(mesh)
    result = {} if diagnostics is None else diagnostics
    result.update({
        "surface_points": len(mesh), "observation_points_per_view": OBSERVED_POINTS,
        "views": len(views), "orientation_hypotheses_per_view": 24, "comparisons": [],
        "synthetic_oracle_initialization": True, "challenge_ground_truth_used": False,
        "exact_status_inlier_iteration_parity": False, "nearest_and_trim_index_parity": False,
        "numerical_atol": NUMERICAL_ATOL, "timings_seconds": {"original_rebuild": [], "canonical_cached": []},
        "predeclared_minimum_speedup": MINIMUM_SPEEDUP, "throughput_hypothesis_accepted": False,
        "source_tree_builds_per_alignment": 1, "source_reused_across_clip": False,
    })
    comparisons = result["comparisons"]
    for view_index, (observed, seeds) in enumerate(views):
        for hypothesis_index, (rotation, translation) in enumerate(seeds):
            _deadline(started)
            original = align_observed_points(mesh, observed, rotation, translation)
            cached = align_observed_points_cached(mesh, observed, rotation, translation)
            record = {"view_index": view_index, "hypothesis_index": hypothesis_index,
                      "original": original.to_dict(), "cached": cached.to_dict(),
                      "nearest_and_trim_indices_exact": False, "queries": []}
            comparisons.append(record)
            record["errors"] = compare_results(original, cached)
            for R, t in ((rotation, translation), (original.rotation, original.translation), (cached.rotation, cached.translation)):
                record["queries"].append(compare_queries(mesh, observed, R, t, tree))
            record["query_max_distance_error_m"] = max(query["max_nearest_distance_error_m"] for query in record["queries"])
            record["nearest_and_trim_indices_exact"] = True
    result["exact_status_inlier_iteration_parity"] = result["nearest_and_trim_index_parity"] = True
    durations = result["timings_seconds"]
    functions = {"original_rebuild": align_observed_points, "canonical_cached": align_observed_points_cached}
    for trial in range(TRIALS):
        for mode in (("original_rebuild", "canonical_cached") if trial % 2 == 0 else ("canonical_cached", "original_rebuild")):
            trial_start = time.perf_counter()
            result["current_timing_trial"] = {"trial": trial, "mode": mode, "alignments_completed": 0}
            for observed, seeds in views:
                for rotation, translation in seeds:
                    _deadline(started)
                    functions[mode](mesh, observed, rotation, translation)
                    result["current_timing_trial"]["alignments_completed"] += 1
            durations[mode].append(time.perf_counter() - trial_start)
    del result["current_timing_trial"]
    speedup = float(np.median(durations["original_rebuild"]) / np.median(durations["canonical_cached"]))
    result["median_speedup"] = speedup
    if not np.isfinite(speedup) or speedup < MINIMUM_SPEEDUP:
        raise ValueError(f"Canonical-tree median speedup {speedup:.3f} is below the predeclared 1.3 gate")
    result["throughput_hypothesis_accepted"] = True
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CPU container with network none")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    report_path = root / "results/rigid-cache.json"
    if report_path.exists():
        raise RuntimeError("Frozen canonical-tree gate already exists")
    import world_reward.rigid_alignment as original_module
    import world_reward.rigid_alignment_cached as cached_module
    report = {
        "stage": "procedural_canonical_kdtree_icp_parity_throughput", "status": "fail",
        "network": "none", "device": "cpu", "own_procedural_inputs_only": True,
        "challenge_inputs_used": False, "challenge_ground_truth_used": False, "hand_labeled_test": False,
        "synthetic_oracle_initialization": True,
        "adoption_performed": False, "challenge_performance_verified": False,
        "numerical_atol": NUMERICAL_ATOL, "exact_status_inlier_iteration_parity_required": True,
        "nearest_and_trim_index_parity_required": True, "minimum_median_speedup": MINIMUM_SPEEDUP,
        "max_gate_seconds": MAX_GATE_SECONDS, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "script_sha256": sha256(Path(__file__)), "original_solver_sha256": sha256(Path(original_module.__file__)),
        "cached_solver_sha256": sha256(Path(cached_module.__file__)), "producer_revision": os.environ.get("WR_CODE_REVISION"),
        "numeric_limitations": "Rigid-query finite precision may change ties/zero-fit stopping; never promised bit exact",
        "gates": {},
    }
    started = time.perf_counter()
    try:
        run_gate(report["gates"])
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("stage", "status", "adoption_performed", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

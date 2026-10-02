"""One-frame whole-candidate static-KDtree parity/throughput experiment.

Own rendered geometry only; the 24 generic seeds never use its known pose.
Only the solver binding changes in an isolated copy of the original candidate
function's globals. Its bytecode, scalar raster schedule, samples, objective,
thresholds and selection rule are unchanged. Floating pose/residual JSON is
compared within 1e-6, not claimed bit-exact. A pass is not adoption or proof of
parity on challenge videos, and does not verify pose accuracy or eligibility.
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
from types import FunctionType

import numpy as np

import pose_candidates
from pose_batch_gate import (
    HEIGHT, K, OBSERVATIONS, PYTORCH3D_REVISION, RADII, SAMPLE_SEED,
    SURFACE_SAMPLES, WIDTH, _inputs, candidate_signature,
)
from world_reward.rigid_alignment import align_observed_points
from world_reward.rigid_alignment_cached import align_observed_points_cached


FRAME_INDEX, NUMERICAL_ATOL = 0, 1e-6
MINIMUM_SPEEDUP, MAXIMUM_SECONDS = 1.3, 60.
FLOAT_FIELDS = frozenset({"rotation", "translation", "initial_residual",
                          "final_residual", "selected_depth_residual_m"})


def isolated_cached_evaluator():
    """Same bytecode with one private solver binding; never patch live globals."""
    original = pose_candidates.evaluate_pose_candidates
    if original.__globals__.get("align_observed_points") is not align_observed_points:
        raise RuntimeError("Original candidate helper solver binding has changed")
    namespace = original.__globals__.copy()
    namespace["align_observed_points"] = align_observed_points_cached
    copied = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    copied.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    copied.__annotations__ = dict(original.__annotations__)
    return copied


def compare_candidates(original, cached, diagnostics=None):
    """Strict slots/image/decision/report parity; only R/t/residual get tolerance.

    Reject a different key, list order, hypothesis, rejection reason, image IoU,
    acceptance boolean, solver status, inliers, iterations or selected index.
    No threshold is relaxed to compensate for a changed branch or raster mask.
    """
    original_json, original_best = candidate_signature(*original)
    cached_json, cached_best = candidate_signature(*cached)
    result = {} if diagnostics is None else diagnostics
    result.update({"selected_hypothesis_index": original_best,
                   "cached_selected_hypothesis_index": cached_best,
                   "candidate_count": len(original[0]), "rejected_count": len(original[1]),
                   "exact_slots_images_decisions_status_inliers_iterations": False,
                   "complete_candidate_json_equal": original_json == cached_json,
                   "original_candidates_sha256": hashlib.sha256(original_json.encode()).hexdigest(),
                   "cached_candidates_sha256": hashlib.sha256(cached_json.encode()).hexdigest()})
    if original_best != cached_best:
        result.update(original=json.loads(original_json), cached=json.loads(cached_json))
        raise RuntimeError("Cached solver changed the selected hypothesis index")
    maximum = 0.

    def compare(first, second, path=()):
        nonlocal maximum
        if type(first) is not type(second):
            raise RuntimeError(f"Candidate type parity failed at {path}")
        if isinstance(first, dict):
            if first.keys() != second.keys():
                raise RuntimeError(f"Candidate key parity failed at {path}")
            for key in first:
                compare(first[key], second[key], (*path, key))
        elif isinstance(first, list):
            if len(first) != len(second):
                raise RuntimeError(f"Candidate list length parity failed at {path}")
            for index, (left, right) in enumerate(zip(first, second, strict=True)):
                compare(left, right, (*path, index))
        elif isinstance(first, float) and any(key in FLOAT_FIELDS for key in path):
            error = abs(first - second)
            maximum = max(maximum, error)
            if error > NUMERICAL_ATOL:
                raise RuntimeError(f"Candidate numeric parity failed at {path}: {error}")
        elif first != second:
            raise RuntimeError(f"Candidate exact parity failed at {path}")

    try:
        compare(json.loads(original_json), json.loads(cached_json))
    except RuntimeError:
        result.update(original=json.loads(original_json), cached=json.loads(cached_json))
        raise
    finally:
        result["maximum_pose_residual_absolute_error"] = maximum
    result["exact_slots_images_decisions_status_inliers_iterations"] = True
    return result


def trial_orders():
    return (("original", "cached"), ("cached", "original"), ("original", "cached"))


def _deadline(started):
    if time.perf_counter() - started >= MAXIMUM_SECONDS:
        raise RuntimeError("Predeclared whole-candidate gate exceeded 60 seconds")


def run_gate(torch, report, started):
    """Remote CUDA only; keep partial comparisons/timings on any failed gate."""
    inputs, summaries = _inputs()
    if len(inputs) != 3 or len(summaries) != 3 or summaries[FRAME_INDEX]["frame_index"] != FRAME_INDEX:
        raise RuntimeError("Require unchanged procedural three-view fixture, select only fixed frame zero")
    values = inputs[FRAME_INDEX]
    report["rendered_fixture_views"] = summaries
    report["evaluated_frame_index"] = FRAME_INDEX
    functions = {"original": pose_candidates.evaluate_pose_candidates,
                 "cached": isolated_cached_evaluator()}
    report["solver_globals_isolated"] = True
    report["candidate_function_bytecode_identical"] = (
        functions["original"].__code__ is functions["cached"].__code__)
    outputs = {}
    for mode, function in functions.items():
        _deadline(started)
        outputs[mode] = function(*values, render_batch_size=1)
        torch.cuda.synchronize()
    report["parity"] = {}
    compare_candidates(outputs["original"], outputs["cached"], report["parity"])
    print(json.dumps({"phase": "parity", **report["parity"]}), flush=True)
    durations = report["synchronized_whole_candidate_trials_seconds"] = {"original": [], "cached": []}
    report["trial_parity"] = []
    for trial, order in enumerate(trial_orders()):
        for mode in order:
            _deadline(started)
            report["current_timing_trial"] = {"trial": trial, "mode": mode}
            torch.cuda.synchronize()
            began = time.perf_counter()
            result = functions[mode](*values, render_batch_size=1)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            durations[mode].append(seconds)
            parity = {"trial": trial, "mode": mode}
            report["trial_parity"].append(parity)
            compare_candidates(outputs["original"], result, parity)
            print(json.dumps({"phase": "throughput", "trial": trial, "mode": mode,
                              "whole_one_frame_seconds": seconds}), flush=True)
            _deadline(started)
    del report["current_timing_trial"]
    speedup = float(np.median(durations["original"]) / np.median(durations["cached"]))
    report["median_whole_candidate_speedup"] = speedup
    report["throughput_hypothesis_accepted"] = bool(np.isfinite(speedup) and speedup >= MINIMUM_SPEEDUP)
    if not report["throughput_hypothesis_accepted"]:
        raise RuntimeError(f"Whole-candidate speedup {speedup:.6g} below predeclared 1.3; not adopted")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable WR_CODE_REVISION")
    output = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward")) / "results/cached-pose-gate.json"
    if output.exists():
        raise FileExistsError("Cached pose gate reports are frozen")
    import torch
    import pytorch3d
    import world_reward.rigid_alignment as original_module
    import world_reward.rigid_alignment_cached as cached_module
    if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require CUDA and audited PyTorch3D 0.7.9")
    report = {"stage": "whole_pose_candidate_canonical_kdtree_parity_throughput", "status": "fail",
              "code_revision": revision, "torch": torch.__version__, "pytorch3d": pytorch3d.__version__,
              "expected_pytorch3d_revision": PYTORCH3D_REVISION, "gpu": torch.cuda.get_device_name(),
              "own_procedural_inputs": True, "synthetic_known_poses_used_to_render_observations": True,
              "synthetic_oracle_initialization": False, "seed_rotations_use_truth_pose": False,
              "challenge_inputs_used": False, "challenge_ground_truth_used": False, "models_used": False,
              "adoption_performed": False, "pose_accuracy_verified": False,
              "challenge_performance_verified": False, "all_clip_image_parity_verified": False,
              "solver_query_execution_only": True, "render_batch_size": 1,
              "hypotheses_per_frame": 24, "evaluated_frame_count": 1, "mesh_vertices": 642, "mesh_faces": 1280,
              "radii_m": list(RADII), "surface_samples": SURFACE_SAMPLES, "observation_samples": OBSERVATIONS,
              "sample_seed": SAMPLE_SEED, "image_size_hw": [HEIGHT, WIDTH], "K": K.tolist(),
              "numeric_atol": NUMERICAL_ATOL, "numeric_json_bit_exact_required": False,
              "predeclared_minimum_useful_speedup": MINIMUM_SPEEDUP, "predeclared_maximum_elapsed_seconds": MAXIMUM_SECONDS,
              "trial_mode_orders": [list(order) for order in trial_orders()], "geometry_units": "metres",
              "throughput_hypothesis_accepted": False}
    for name, path in (("script", Path(__file__)), ("pose_candidate_source", Path(pose_candidates.__file__)),
                       ("fixture_source", Path(__file__).with_name("pose_batch_gate.py")),
                       ("original_solver", Path(original_module.__file__)), ("cached_solver", Path(cached_module.__file__))):
        report[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        run_gate(torch, report, started)
        _deadline(started)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

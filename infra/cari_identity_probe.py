"""Pure contracts for one sealed, shared-identity inverse-fidelity experiment.

Callbacks must wrap the unmodified pinned official solvers and reference model.
This module cannot certify those runtime bindings or produce a submission.
Reserved targets fit pose nuisance with the train-fitted identity frozen: they
are identity-reserved reconstruction probes, not held-out accuracy evidence.
"""
from __future__ import annotations

import json

import numpy as np

from cari_converter import canonical_array_identity
from cari_converter_diagnostic import _converted, _float, converter_diagnostics


FIT_INDICES = (0, 197, 394, 592, 789)
RESERVED_INDICES = (98, 296, 493, 690)
JOINT = {"iters": 40, "tol": 1e-5, "pmask": None, "w": None, "prior": 0.0}
POSE = {"iters": 60, "tol": 1e-5}
PROTOCOL = {"joint": JOINT, "reserved_pose": POSE, "precision": "float64", "fd_step": 1e-6,
            "shared_identity_dimension": 113, "train_pose_dimension": 136,
            "all_vertex_unweighted_squared_objective": True,
            "final_controls_dtype": "float32", "final_reference_precision": "float32",
            "gate_mm": 2.0, "no_regression_atol_mm": 1e-4}


def _double(value, shape, name):
    array = _float(value, shape, name)
    if array.dtype != np.float64:
        raise ValueError(f"{name} must remain float64 until explicit quantization")
    return array.copy()


def _unchanged(inputs, saved):
    if any(not np.array_equal(a, b) for a, b in zip(inputs, saved)):
        raise ValueError("Callback mutated sealed controls, identity, or targets")


def identity_probe(converted, native_params, target_vertices_m, *, episode_index,
                   frame_index, provenance, joint_fit, pose_polish, reference_f32):
    """Return a diagnostic proposal only; every original input remains unchanged.

    ``joint_fit(tgt_mm_flat, pose_f64, z_f64, **JOINT) -> (pose, z)``;
    ``pose_polish(..., **POSE) -> (pose, per_frame_mean_error_mm)``;
    ``reference_f32(pose_f32, z_f32) -> vertices_mm``. Chunk sizes are runtime
    memory choices, not solver/selection policy knobs here.
    """
    count, original_errors = _converted(converted)
    if type(episode_index) is not int or episode_index != 0 or count != 790:
        raise ValueError("This frozen probe requires episode 0 and all 790 original frames")
    sealed = converter_diagnostics(converted, native_params, episode_index=episode_index,
        frame_index=frame_index, provenance=provenance)
    target = _float(target_vertices_m, (count, 18439, 3), "native target vertices")
    target_identity = canonical_array_identity(target)
    if target_identity["sha256"] != provenance["native_vertices_sha256"]:
        raise ValueError("Full regenerated native target hash differs from frozen target")
    worst = int(np.argmax(original_errors))  # Original residual only; lowest index wins ties.
    reserved = [i for i in dict.fromkeys((*RESERVED_INDICES, worst)) if i not in FIT_INDICES]
    indices = [*FIT_INDICES, *reserved]
    original_pose = np.asarray(converted["pose"])[indices].astype(np.float64)
    initial_z = np.concatenate([converted["scales"], converted["shape"]])[None].astype(np.float64)
    targets_mm = target[indices].astype(np.float64) * 1000.0

    def replay(pose, identity):
        p, z = pose.astype(np.float32), identity.astype(np.float32)
        saved = (p.copy(), z.copy())
        vertices = _float(reference_f32(p, z), targets_mm.shape, "float32 reference replay")
        _unchanged((p, z), saved)
        residual = vertices.astype(np.float64) - targets_mm
        means = np.linalg.norm(residual, axis=-1).mean(1)
        squared = np.square(residual).sum((1, 2))
        if not np.isfinite(means).all() or not np.isfinite(squared).all():
            raise ValueError("Reference residual statistics overflowed")
        return means, squared

    before, before_cost = replay(original_pose, initial_z)
    if not np.allclose(before, original_errors[indices], rtol=1e-5, atol=1e-4):
        raise ValueError("Original float32 replay differs from sealed original errors")
    train_target = targets_mm[:len(FIT_INDICES)].reshape(len(FIT_INDICES), -1).copy()
    train_pose, train_z = original_pose[:len(FIT_INDICES)].copy(), initial_z.copy()
    saved = (train_target.copy(), train_pose.copy(), train_z.copy())
    fitted_pose, fitted_z = joint_fit(train_target, train_pose, train_z, **JOINT)
    _unchanged((train_target, train_pose, train_z), saved)
    fitted_pose = _double(fitted_pose, (len(FIT_INDICES), 136), "joint pose proposal")
    fitted_z = _double(fitted_z, (1, 113), "one shared identity proposal")
    reserved_target = targets_mm[len(FIT_INDICES):].reshape(len(reserved), -1).copy()
    reserved_pose, reserved_z = original_pose[len(FIT_INDICES):].copy(), fitted_z.copy()
    saved = (reserved_target.copy(), reserved_pose.copy(), reserved_z.copy())
    polished_pose, fit_errors = pose_polish(reserved_target, reserved_pose, reserved_z, **POSE)
    _unchanged((reserved_target, reserved_pose, reserved_z), saved)
    polished_pose = _double(polished_pose, (len(reserved), 136), "reserved pose proposal")
    fit_errors = _double(fit_errors, (len(reserved),), "reserved float64 fit errors")
    if np.any(fit_errors < 0):
        raise ValueError("Reserved fit errors cannot be negative")
    proposal_pose = np.concatenate([fitted_pose, polished_pose])
    after, after_cost = replay(proposal_pose, fitted_z)
    regressions = np.flatnonzero(after - before > PROTOCOL["no_regression_atol_mm"])
    failures = np.flatnonzero(after > PROTOCOL["gate_mm"])
    passed = not len(regressions) and not len(failures)
    report = {"schema": "world-reward-constrained-identity-probe-v1",
        "stage": "sealed_official_converter_shared_identity_inverse_probe", "status": "pass" if passed else "fail",
        "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "full_frame_fidelity_verified": False, "source_bindings_runtime_verified": False,
        "actual_official_joint_verified": False, "actual_official_reserved_pose_verified": False,
        "final_reference_float32_runtime_verified": False,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "input_track": "track_1",
        "episode_index": episode_index, "original_frames": count, "provenance": dict(provenance),
        "target_vertices_identity": target_identity, "protocol": PROTOCOL,
        "fit_indices": list(FIT_INDICES), "reserved_indices": reserved, "probe_indices": indices,
        "original_worst_frame": worst, "worst_selection_rule": "lowest_index_original_residual_argmax_only",
        "reserved_identity_fixed": True, "reserved_pose_uses_reserved_targets": True,
        "heldout_accuracy_verified": False, "native_identity": sealed["native_identity"],
        "before_mean_vertex_error_mm": before.tolist(), "after_mean_vertex_error_mm": after.tolist(),
        "before_squared_objective_mm2": before_cost.tolist(), "after_squared_objective_mm2": after_cost.tolist(),
        "reserved_float64_fit_mean_vertex_error_mm": fit_errors.tolist(),
        "gate_failure_indices": [indices[i] for i in failures],
        "regression_indices": [indices[i] for i in regressions], "probe_gate_pass": passed,
        "shared_identity_update_l2": float(np.linalg.norm(fitted_z - initial_z))}
    json.dumps(report, allow_nan=False)
    return {"report": report, "fit_pose_float64": fitted_pose, "reserved_pose_float64": polished_pose,
            "shared_identity_float64": fitted_z, "quantized_probe_pose": proposal_pose.astype(np.float32),
            "quantized_shared_identity": fitted_z.astype(np.float32)}

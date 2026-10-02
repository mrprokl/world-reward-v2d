"""Experimental, sealed converter diagnostics; never a submission or identity repair.

Callbacks are NumPy boundaries for the pinned official ``lm_pose(mhr, tgt,
pose, z, iters=60, tol=1e-5)`` and ``mhr.run(pose, z)[0]`` (vertices in mm).
The future runtime must bind float64/FD=1e-6 and float32 reference instances.
This module neither loads models nor claims that those bindings were verified.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from cari_converter import (CHECKPOINT_SHA256, CONVERTER_SHA256,
                            MAX_MEAN_VERTEX_ERROR_MM, PARAMETER_DIMS,
                            REFERENCE_MODEL_SHA256, UPSTREAM_REVISION)
from world_reward.data import sha256


PINS = {"upstream_revision": UPSTREAM_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
        "converter_sha256": CONVERTER_SHA256, "reference_model_sha256": REFERENCE_MODEL_SHA256}
POLISH = {"iters": 60, "tol": 1e-5}
PROTOCOL = {**POLISH, "precision": "float64", "fd_step": 1e-6,
            "identity_fixed": True, "all_vertex_unweighted_squared_objective": True,
            "final_controls_dtype": "float32", "final_reference_precision": "float32"}


def _float(value, shape, name):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.shape != shape or array.dtype.kind != "f"
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} requires finite unmasked float {shape}")
    return array


def _converted(value):
    if not isinstance(value, Mapping) or not isinstance(value.get("report"), Mapping):
        raise ValueError("Complete original converter mapping/report required")
    pose = np.asarray(value.get("pose"))
    if pose.ndim != 2 or pose.shape[0] < 1:
        raise ValueError("Nonempty original pose sequence required")
    count = len(pose)
    for key, shape in {"pose": (count, 136), "scales": (68,), "shape": (45,)}.items():
        if _float(value.get(key), shape, key).dtype != np.float32:
            raise ValueError("Original official parameters must be float32")
    errors = _float(value.get("per_frame_vertex_error_mm"), (count,), "errors")
    valid = np.asarray(value.get("valid_input"))
    report = value["report"]
    if (np.any(errors < 0) or valid.dtype != np.bool_ or valid.shape != (count,) or not valid.all()
            or type(report.get("frames")) is not int or type(report.get("fitted_frames")) is not int
            or report.get("frames") != count or report.get("fitted_frames") != count
            or report.get("invalid_input_frames") != [] or report.get("precision") != "float32"):
        raise ValueError("Full finite original-frame coverage and original float32 fit required")
    summary = report.get("vertex_error_mm", {})
    for key, expected in (("mean", errors.mean()), ("worst_frame_mean", errors.max())):
        actual = summary.get(key)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-5, atol=1e-5)):
            raise ValueError("Original converter summary disagrees with per-frame errors")
    json.dumps(dict(report), allow_nan=False)
    return count, errors.astype(np.float64)


def error_statistics(errors):
    if np.ma.isMaskedArray(errors):
        raise ValueError("Errors cannot conceal invalid entries behind a mask")
    values = np.asarray(errors)
    if values.ndim != 1 or not len(values):
        raise ValueError("Nonempty error vector required")
    values = _float(values, values.shape, "errors").astype(np.float64)
    if np.any(values < 0):
        raise ValueError("Errors cannot be negative")
    ordered = np.argsort(values, kind="stable")
    worst = int(np.argmax(values))  # Lowest original index wins exact ties.
    probes = list(dict.fromkeys([0, int(ordered[(len(values) - 1) // 2]), worst]))
    failures = np.flatnonzero(values > MAX_MEAN_VERTEX_ERROR_MM).tolist()
    return {"min_mm": float(values.min()), "median_mm": float(np.median(values)),
            "mean_mm": float(values.mean()), "p99_mm": float(np.percentile(values, 99)),
            "max_mm": float(values.max()), "worst_frame": worst,
            "gate_mm": MAX_MEAN_VERTEX_ERROR_MM, "gate_pass": not failures,
            "gate_failure_count": len(failures), "gate_failure_frames": failures,
            "probe_indices": probes, "probe_rule": "first,stable_lower_median_error,lowest_index_worst;deduplicated",
            "tail": [{"frame_index": int(i), "mean_vertex_error_mm": float(values[i])}
                     for i in np.argsort(-values, kind="stable")[:10]]}


def converter_diagnostics(converted, native_params, *, episode_index, frame_index, provenance):
    count, errors = _converted(converted)
    indices = np.asarray(frame_index)
    if (type(episode_index) is not int or episode_index not in (0, 15)
            or indices.dtype.kind not in "iu" or indices.shape != (count,)
            or not np.array_equal(indices, np.arange(count))):
        raise ValueError("Diagnostic cohort 0/15 requires every original integer frame")
    if not isinstance(provenance, Mapping) or any(provenance.get(k) != v for k, v in PINS.items()):
        raise ValueError("Pinned official source/model/checkpoint provenance required")
    if (provenance.get("ground_truth_used") is not False
            or provenance.get("hand_labeled_test") is not False or provenance.get("oracle_modes") != []
            or provenance.get("input_track") != "track_1"):
        raise ValueError("Explicit Track 1 no-GT/no-label/no-oracle provenance required")
    for key in ("native_bundle_sha256", "native_vertices_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Frozen native bundle/vertices SHA256 required")
    if not isinstance(native_params, Mapping):
        raise ValueError("Original native parameter blocks required")
    for key, dimension in PARAMETER_DIMS.items():
        _float(native_params.get(key), (count, dimension), key)
    if np.any(np.asarray(native_params["mhr_face"]) != 0):
        raise ValueError("Official zero-expression conversion cannot conceal nonzero native expressions")
    identities = {}
    for key in ("mhr_shape", "mhr_scale"):
        array = np.asarray(native_params[key], dtype=np.float64)
        span = np.ptp(array, axis=0)
        identities[key] = {"dimension": array.shape[1], "min": array.min(0).tolist(),
                           "max": array.max(0).tolist(), "range": span.tolist(),
                           "max_range": float(span.max()), "constant": bool(np.all(span == 0))}
    result = {"schema": "world-reward-converter-numerical-diagnostic-v1",
              "stage": "sealed_official_converter_numerical_diagnostic", "status": "sealed", "research_only": True,
              "submission_produced": False, "adoption_authorized": False,
              "actual_official_polish_verified": False, "source_bindings_runtime_verified": False,
              "episode_index": episode_index, "frames": count, "frame_index": indices.tolist(),
              "provenance": dict(provenance), "original": error_statistics(errors),
              "native_identity": identities, "identity_shared_by_original_converter": True,
              "polish_protocol": dict(PROTOCOL)}
    json.dumps(result, allow_nan=False)
    return result


def seal_original_converter(output_directory, converted, native_params, **arguments):
    """Persist even a failed 2-mm result, exclusively; never overwrite an episode."""
    report = converter_diagnostics(converted, native_params, **arguments)
    directory = Path(output_directory)
    if (not directory.is_absolute() or directory.resolve() != directory
            or not directory.parent.is_dir() or directory.exists() or directory.is_symlink()):
        raise ValueError("New absolute nonsymlink diagnostic directory required")
    directory.mkdir()
    archive = directory / "original_converter.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **{k: converted[k] for k in
            ("pose", "scales", "shape", "per_frame_vertex_error_mm", "valid_input")},
            report=np.asarray(json.dumps(dict(converted["report"]), sort_keys=True, allow_nan=False)))
    report["original_archive"] = {"file": archive.name, "sha256": sha256(archive), "bytes": archive.stat().st_size}
    with (directory / "original_report.json").open("x") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
    return report


def polish_probes(converted, diagnostics, target_vertices_m, *, pose_polish, reference_f32):
    """Return probe-only proposals and float32 replay diagnostics, never adopt them."""
    count, original_errors = _converted(converted)
    expected = error_statistics(original_errors)
    if (not isinstance(diagnostics, Mapping) or diagnostics.get("frames") != count
            or diagnostics.get("original") != expected or diagnostics.get("research_only") is not True
            or diagnostics.get("adoption_authorized") is not False
            or diagnostics.get("polish_protocol") != PROTOCOL
            or diagnostics.get("provenance", {}).get("converter_sha256") != CONVERTER_SHA256):
        raise ValueError("Unchanged original diagnostics required before polishing")
    target = _float(target_vertices_m, (count, 18439, 3), "native target vertices")
    indices = expected["probe_indices"]
    initial = np.asarray(converted["pose"])[indices].astype(np.float64)
    identity = np.concatenate([converted["scales"], converted["shape"]])[None].astype(np.float64)
    target_mm = target[indices].astype(np.float64) * 1000

    def replay(pose):
        p, z = pose.astype(np.float32), identity.astype(np.float32)
        saved_p, saved_z = p.copy(), z.copy()
        vertices = _float(reference_f32(p, z), target_mm.shape, "float32 replay vertices")
        if not np.array_equal(p, saved_p) or not np.array_equal(z, saved_z):
            raise ValueError("Replay callback mutated supplied controls/identity")
        residual = vertices.astype(np.float64) - target_mm
        return np.linalg.norm(residual, axis=-1).mean(1), np.square(residual).sum((1, 2))

    before, before_cost = replay(initial)
    if not np.allclose(before, original_errors[indices], rtol=1e-5, atol=1e-4):
        raise ValueError("Original float32 replay disagrees with sealed converter errors")
    p, z, tgt = initial.copy(), identity.copy(), target_mm.reshape(len(indices), -1).copy()
    proposal, fit_errors = pose_polish(tgt, p, z, **POLISH)
    if (not np.array_equal(p, initial) or not np.array_equal(z, identity)
            or not np.array_equal(tgt, target_mm.reshape(len(indices), -1))):
        raise ValueError("Polish callback mutated fixed identity/original inputs")
    proposal = _float(proposal, initial.shape, "float64 polished pose")
    if proposal.dtype != np.float64:
        raise ValueError("Polish proposal must remain float64 until explicit quantization")
    fit_errors = _float(fit_errors, (len(indices),), "float64 fit errors")
    if np.any(fit_errors < 0):
        raise ValueError("Fit errors cannot be negative")
    after, after_cost = replay(proposal)
    report = {"research_only": True, "adoption_authorized": False, "full_frame_fidelity_verified": False,
              "actual_official_polish_verified": False, "identity_unchanged": True,
              "probe_indices": indices, "protocol": dict(diagnostics["polish_protocol"]),
              "before_mean_vertex_error_mm": before.tolist(), "after_mean_vertex_error_mm": after.tolist(),
              "float64_fit_mean_vertex_error_mm": fit_errors.tolist(),
              "before_squared_objective_mm2": before_cost.tolist(), "after_squared_objective_mm2": after_cost.tolist(),
              "pose_update_l2": np.linalg.norm(proposal - initial, axis=1).tolist(),
              "probe_gate_pass": bool(np.all(after <= MAX_MEAN_VERTEX_ERROR_MM))}
    return {"report": report, "original_probe_pose": initial.astype(np.float32),
            "polished_probe_pose_float64": proposal.copy(), "quantized_probe_pose": proposal.astype(np.float32),
            "fixed_scales": np.asarray(converted["scales"]).copy(), "fixed_shape": np.asarray(converted["shape"]).copy()}

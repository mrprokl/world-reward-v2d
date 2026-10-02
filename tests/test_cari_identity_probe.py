"""Tiny own linear callback fixtures, not official solver/model execution proof."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_identity_probe", infra / "cari_identity_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Broadcast-only synthetic full target: do not hash/allocate a heavy local mesh sequence.
    monkeypatch.setattr(module, "canonical_array_identity", lambda value: {
        "sha256": "b" * 64, "dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.nbytes})
    original_float = module._float
    def tiny_target_float(value, shape, name):
        if name == "native target vertices":
            assert value.shape == shape and value.strides[:2] == (0, 0)
            original_float(value[:1, :1], (1, 1, 3), name)
            return value
        return original_float(value, shape, name)
    monkeypatch.setattr(module, "_float", tiny_target_float)
    return module


@pytest.fixture
def inputs(probe):
    from cari_converter_diagnostic import PARAMETER_DIMS, PINS
    errors = np.full(790, .5, np.float32)
    errors[1] = 4.0
    pose = np.zeros((790, 136), np.float32)
    pose[:, 0] = errors
    converted = {"pose": pose, "scales": np.zeros(68, np.float32), "shape": np.zeros(45, np.float32),
        "per_frame_vertex_error_mm": errors, "valid_input": np.ones(790, bool),
        "report": {"frames": 790, "fitted_frames": 790, "invalid_input_frames": [], "precision": "float32",
                   "vertex_error_mm": {"mean": float(errors.mean()), "worst_frame_mean": 4.0}}}
    native = {key: np.zeros((790, dim), np.float32) for key, dim in PARAMETER_DIMS.items()}
    target = np.broadcast_to(np.zeros((1, 1, 3), np.float32), (790, 18439, 3))
    arguments = {"episode_index": 0, "frame_index": np.arange(790), "provenance": {
        **PINS, "native_bundle_sha256": "a" * 64, "native_vertices_sha256": "b" * 64,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "input_track": "track_1"}}
    return converted, native, target, arguments


def replay(pose, identity):
    assert pose.dtype == identity.dtype == np.float32 and identity.shape == (1, 113)
    return np.broadcast_to(np.stack([pose[:, 0] + identity[0, 0], np.zeros(len(pose)),
                                   np.zeros(len(pose))], -1)[:, None], (len(pose), 18439, 3))


def joint(tgt, pose, identity, **kwargs):
    assert kwargs == {"iters": 40, "tol": 1e-5, "pmask": None, "w": None, "prior": 0.0}
    assert pose.shape == (5, 136) and identity.shape == (1, 113) and tgt.shape == (5, 55317)
    assert pose.dtype == identity.dtype == tgt.dtype == np.float64
    p, z = pose.copy(), identity.copy()
    z[0, 0] = .25
    p[:, 0] = -.25
    return p, z


def polish(tgt, pose, identity, **kwargs):
    assert kwargs == {"iters": 60, "tol": 1e-5} and identity.shape == (1, 113)
    assert np.array_equal(identity[0, :1], [.25])
    assert tgt.dtype == pose.dtype == identity.dtype == np.float64
    p = pose.copy()
    p[:, 0] = -identity[0, 0]
    return p, np.zeros(len(pose), np.float64)


def run(probe, inputs, **callbacks):
    converted, native, target, args = inputs
    return probe.identity_probe(converted, native, target, **args,
        **{"joint_fit": joint, "pose_polish": polish, "reference_f32": replay, **callbacks})


def test_fixed_protocol_shared_identity_and_exact_f32_gate(probe, inputs):
    before = copy.deepcopy(inputs[:2])
    result = run(probe, inputs)
    report = result["report"]
    assert report["fit_indices"] == [0, 197, 394, 592, 789]
    assert report["reserved_indices"] == [98, 296, 493, 690, 1]
    assert report["probe_indices"] == [0, 197, 394, 592, 789, 98, 296, 493, 690, 1]
    assert report["before_mean_vertex_error_mm"] == [.5] * 9 + [4.]
    assert report["after_mean_vertex_error_mm"] == [0.] * 10 and report["probe_gate_pass"] is True
    assert report["before_squared_objective_mm2"][-1] == 18439 * 16
    assert result["quantized_shared_identity"].shape == (1, 113)
    assert result["quantized_probe_pose"].dtype == np.float32
    for flag in ("full_frame_fidelity_verified", "source_bindings_runtime_verified", "actual_official_joint_verified",
                 "actual_official_reserved_pose_verified", "final_reference_float32_runtime_verified", "adoption_authorized"):
        assert report[flag] is False
    assert report["reserved_pose_uses_reserved_targets"] is True and report["heldout_accuracy_verified"] is False
    json.dumps(report, allow_nan=False)
    for key in inputs[0]:
        if isinstance(inputs[0][key], np.ndarray): np.testing.assert_array_equal(inputs[0][key], before[0][key])
    np.testing.assert_array_equal(inputs[1]["mhr_shape"], before[1]["mhr_shape"])


@pytest.mark.parametrize("worst,expected", [(0, [98, 296, 493, 690]), (98, [98, 296, 493, 690])])
def test_worst_selection_uses_original_error_only_and_deduplicates(probe, inputs, worst, expected):
    converted = inputs[0]
    converted["per_frame_vertex_error_mm"][1] = .5
    converted["pose"][1, 0] = .5
    converted["per_frame_vertex_error_mm"][worst] = 4.
    converted["pose"][worst, 0] = 4.
    converted["report"]["vertex_error_mm"]["mean"] = float(converted["per_frame_vertex_error_mm"].mean())
    report = run(probe, inputs)["report"]
    assert report["original_worst_frame"] == worst and report["reserved_indices"] == expected
    assert len(report["probe_indices"]) == 9


@pytest.mark.parametrize("failure", ["gt", "labels", "oracle", "track", "pin", "hash", "episode", "bool_episode",
    "frames", "indices", "valid", "pose_dtype", "shape_dim", "native_dim", "native_nan", "face", "precision", "negative"])
def test_bad_frozen_inputs_fail_before_callbacks(probe, inputs, failure):
    converted, native, target, args = inputs
    if failure == "gt": args["provenance"]["ground_truth_used"] = True
    elif failure == "labels": args["provenance"]["hand_labeled_test"] = 0
    elif failure == "oracle": args["provenance"]["oracle_modes"] = ["identity"]
    elif failure == "track": args["provenance"]["input_track"] = "track_3"
    elif failure == "pin": args["provenance"]["reference_model_sha256"] = "0" * 64
    elif failure == "hash": args["provenance"]["native_vertices_sha256"] = "c" * 64
    elif failure == "episode": args["episode_index"] = 15
    elif failure == "bool_episode": args["episode_index"] = False
    elif failure == "frames": converted["pose"] = converted["pose"][:-1]
    elif failure == "indices": args["frame_index"][-1] = 788
    elif failure == "valid": converted["valid_input"][789] = False
    elif failure == "pose_dtype": converted["pose"] = converted["pose"].astype(np.float64)
    elif failure == "shape_dim": converted["shape"] = np.zeros(44, np.float32)
    elif failure == "native_dim": native["mhr_scale"] = np.zeros((790, 68), np.float32)
    elif failure == "native_nan": native["mhr_shape"][0, 0] = np.nan
    elif failure == "face": native["mhr_face"][0, 0] = 1
    elif failure == "precision": converted["report"]["precision"] = "float64"
    elif failure == "negative": converted["per_frame_vertex_error_mm"][0] = -1
    def forbidden(*a, **kw): pytest.fail("invalid inputs reached solver")
    with pytest.raises(ValueError): run(probe, inputs, joint_fit=forbidden)


@pytest.mark.parametrize("failure", ["joint_mutation", "joint_z_mutation", "target_mutation", "joint_dtype", "joint_perframe_z",
    "joint_nan", "pose_mutation", "reserved_z_mutation", "pose_dtype", "negative_fit_error", "replay_mutation", "replay_mismatch"])
def test_callback_contracts_fail_closed(probe, inputs, failure):
    def bad_joint(tgt, p, z, **kw):
        result, identity = joint(tgt, p, z, **kw)
        if failure == "joint_mutation": p[0, 0] += 1
        elif failure == "joint_z_mutation": z[0, 0] += 1
        elif failure == "target_mutation": tgt[0, 0] += 1
        elif failure == "joint_dtype": result = result.astype(np.float32)
        elif failure == "joint_perframe_z": identity = np.repeat(identity, 5, axis=0)
        elif failure == "joint_nan": identity[0, 0] = np.nan
        return result, identity
    def bad_polish(tgt, p, z, **kw):
        result, errors = polish(tgt, p, z, **kw)
        if failure == "pose_mutation": p[0, 0] += 1
        elif failure == "reserved_z_mutation": z[0, 0] += 1
        elif failure == "pose_dtype": result = result.astype(np.float32)
        elif failure == "negative_fit_error": errors[0] = -1
        return result, errors
    def bad_replay(p, z):
        vertices = replay(p, z)
        if failure == "replay_mutation": z[0, 0] += 1
        elif failure == "replay_mismatch": vertices = vertices + 1
        return vertices
    with pytest.raises(ValueError): run(probe, inputs, joint_fit=bad_joint, pose_polish=bad_polish, reference_f32=bad_replay)


@pytest.mark.parametrize("residual,gate,regression", [(.5, False, False), (.5002, False, True), (2.1, True, True)])
def test_all_probes_enforce_2mm_and_absolute_noregression(probe, inputs, residual, gate, regression):
    def candidate(tgt, p, z, **kw):
        fitted, identity = joint(tgt, p, z, **kw)
        fitted[0, 0] = residual - identity[0, 0]
        return fitted, identity
    report = run(probe, inputs, joint_fit=candidate)["report"]
    assert (0 in report["gate_failure_indices"]) is gate
    assert (0 in report["regression_indices"]) is regression
    assert report["probe_gate_pass"] is (not gate and not regression)
    assert report["status"] == ("pass" if report["probe_gate_pass"] else "fail")

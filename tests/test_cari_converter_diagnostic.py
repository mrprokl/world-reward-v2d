"""Own linear NumPy fixtures, not native MHR, numerical-performance or GPU proof."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("test_converter_diagnostic_module", infra / "cari_converter_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(diagnostic):
    count = 5
    errors = np.array([.5, 3, 1, 2, 4], dtype=np.float32)
    pose = np.zeros((count, 136), dtype=np.float32)
    pose[:, 0] = errors
    converted = {"pose": pose, "scales": np.zeros(68, np.float32), "shape": np.zeros(45, np.float32),
                 "per_frame_vertex_error_mm": errors, "valid_input": np.ones(count, bool),
                 "report": {"frames": count, "fitted_frames": count, "invalid_input_frames": [],
                            "precision": "float32", "vertex_error_mm": {"mean": float(errors.mean()), "worst_frame_mean": 4.}}}
    native = {k: np.zeros((count, n), np.float32) for k, n in diagnostic.PARAMETER_DIMS.items()}
    native["mhr_shape"][:, 0] = np.arange(count)
    arguments = {"episode_index": 0, "frame_index": np.arange(count), "provenance": {
        **diagnostic.PINS, "native_bundle_sha256": "a" * 64, "native_vertices_sha256": "b" * 64,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "input_track": "track_1"}}
    return converted, native, arguments


def test_statistics_full_gate_not_average_and_stable_ties(diagnostic):
    report = diagnostic.error_statistics(np.array([2., 2., 3., 2.]))
    assert report["probe_indices"] == [0, 1, 2]
    assert report["gate_failure_frames"] == [2]
    assert report["tail"][1]["frame_index"] == 0
    equal = diagnostic.error_statistics(np.zeros(3))
    assert equal["probe_indices"] == [0, 1]
    assert diagnostic.error_statistics(np.zeros(1))["probe_indices"] == [0]
    assert diagnostic.error_statistics(np.array([2.]))["gate_pass"] is True


def test_complete_identity_ranges_and_declared_not_verified_protocol(diagnostic, inputs):
    converted, native, arguments = inputs
    before = copy.deepcopy(inputs)
    report = diagnostic.converter_diagnostics(converted, native, **arguments)
    assert report["frames"] == 5 and report["original"]["gate_failure_count"] == 2
    assert report["original"]["probe_indices"] == [0, 3, 4]
    assert report["native_identity"]["mhr_shape"]["max_range"] == 4
    assert report["native_identity"]["mhr_scale"]["dimension"] == 28
    assert report["native_identity"]["mhr_scale"]["constant"] is True
    assert report["polish_protocol"] == {"iters": 60, "tol": 1e-5, "precision": "float64", "fd_step": 1e-6,
        "identity_fixed": True, "all_vertex_unweighted_squared_objective": True,
        "final_controls_dtype": "float32", "final_reference_precision": "float32"}
    assert report["actual_official_polish_verified"] is False
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(converted["pose"], before[0]["pose"])
    np.testing.assert_array_equal(native["mhr_shape"], before[1]["mhr_shape"])


@pytest.mark.parametrize("episode", [0, 15])
def test_original_failed_result_is_sealed_before_gate_and_never_replaced(diagnostic, inputs, tmp_path, episode):
    converted, native, args = inputs
    args["episode_index"] = episode
    destination = tmp_path / "diagnostic"
    report = diagnostic.seal_original_converter(destination, converted, native, **args)
    assert report["original"]["gate_pass"] is False
    assert report["original_archive"]["sha256"] == diagnostic.sha256(destination / "original_converter.npz")
    with np.load(destination / "original_converter.npz", allow_pickle=False) as archive:
        for key in ("pose", "scales", "shape", "per_frame_vertex_error_mm", "valid_input"):
            np.testing.assert_array_equal(archive[key], converted[key])
        assert json.loads(str(archive["report"])) == converted["report"]
    assert json.loads((destination / "original_report.json").read_text()) == report
    old = (destination / "original_converter.npz").read_bytes()
    with pytest.raises(ValueError): diagnostic.seal_original_converter(destination, converted, native, **args)
    assert (destination / "original_converter.npz").read_bytes() == old


@pytest.mark.parametrize("failure", ["pin", "gt", "label", "oracle", "track", "hash", "episode", "bool_episode",
                                     "frames", "valid", "native_dim", "native_nan", "face", "pose64", "negative", "summary"])
def test_invalid_inputs_fail_before_sealing(diagnostic, inputs, tmp_path, failure):
    converted, native, args = inputs
    if failure == "pin": args["provenance"]["converter_sha256"] = "0" * 64
    elif failure == "gt": args["provenance"]["ground_truth_used"] = None
    elif failure == "label": args["provenance"]["hand_labeled_test"] = 0
    elif failure == "oracle": args["provenance"]["oracle_modes"] = ["pose"]
    elif failure == "track": args["provenance"]["input_track"] = "track_2"
    elif failure == "hash": args["provenance"]["native_vertices_sha256"] = "X" * 64
    elif failure == "episode": args["episode_index"] = 1
    elif failure == "bool_episode": args["episode_index"] = False
    elif failure == "frames": args["frame_index"] = np.array([0, 1, 2, 3, 3])
    elif failure == "valid": converted["valid_input"][2] = False
    elif failure == "native_dim": native["mhr_scale"] = np.zeros((5, 68))
    elif failure == "native_nan": native["mhr_scale"][2, 0] = np.nan
    elif failure == "face": native["mhr_face"][0, 0] = 1
    elif failure == "pose64": converted["pose"] = converted["pose"].astype(np.float64)
    elif failure == "negative": converted["per_frame_vertex_error_mm"][2] = -1
    elif failure == "summary": converted["report"]["vertex_error_mm"]["mean"] = 1
    destination = tmp_path / "bad"
    with pytest.raises(ValueError): diagnostic.seal_original_converter(destination, converted, native, **args)
    assert not destination.exists()


@pytest.mark.parametrize("values", [[], [-1.], [np.nan], [np.inf], [[0.]], [0]])
def test_bad_error_statistics(diagnostic, values):
    with pytest.raises(ValueError): diagnostic.error_statistics(np.asarray(values))


def test_masked_errors_cannot_hide_bad_values(diagnostic):
    with pytest.raises(ValueError): diagnostic.error_statistics(np.ma.array([np.nan], mask=[True]))


def test_symlink_parent_is_not_a_new_output(diagnostic, inputs, tmp_path):
    converted, native, args = inputs
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError): diagnostic.seal_original_converter(tmp_path / "link" / "new", converted, native, **args)


def _linear_replay(pose, identity):
    assert pose.dtype == identity.dtype == np.float32 and identity.shape == (1, 113)
    vertices = np.zeros((len(pose), 18439, 3), np.float32)
    vertices[:, :, 0] = pose[:, :1]
    return vertices


def test_own_linear_double_polish_and_float32_replay_no_identity_fit(diagnostic, inputs):
    converted, native, args = inputs
    report = diagnostic.converter_diagnostics(converted, native, **args)
    before = copy.deepcopy(converted)
    calls = []
    def polish(tgt, pose, z, *, iters, tol):
        assert tgt.shape == (3, 18439 * 3) and pose.shape == (3, 136)
        assert tgt.dtype == pose.dtype == z.dtype == np.float64
        calls.append((iters, tol))
        candidate = pose.copy()
        candidate[:, 0] = tgt.reshape(3, 18439, 3)[:, :, 0].mean(1)
        return candidate, np.zeros(3, np.float64)
    result = diagnostic.polish_probes(converted, report, np.zeros((5, 18439, 3)),
                                      pose_polish=polish, reference_f32=_linear_replay)
    assert calls == [(60, 1e-5)]
    assert result["report"]["before_mean_vertex_error_mm"] == [.5, 2., 4.]
    assert result["report"]["after_mean_vertex_error_mm"] == [0., 0., 0.]
    assert result["report"]["probe_gate_pass"] is True
    assert result["report"]["full_frame_fidelity_verified"] is False
    assert result["report"]["adoption_authorized"] is False
    np.testing.assert_array_equal(result["fixed_scales"], converted["scales"])
    np.testing.assert_array_equal(converted["pose"], before["pose"])
    json.dumps(result["report"], allow_nan=False)


@pytest.mark.parametrize("failure", ["identity_mutation", "pose_mutation", "target_mutation", "nan", "float32", "negative", "replay_mutation", "replay_mismatch", "changed_diagnostics", "changed_protocol", "changed_pin"])
def test_polish_callbacks_fail_closed(diagnostic, inputs, failure):
    converted, native, args = inputs
    report = diagnostic.converter_diagnostics(converted, native, **args)
    if failure == "changed_diagnostics": report["original"]["probe_indices"] = [0]
    elif failure == "changed_protocol": report["polish_protocol"]["iters"] = 600
    elif failure == "changed_pin": report["provenance"]["converter_sha256"] = "0" * 64
    def polish(tgt, pose, z, **kwargs):
        candidate = pose.copy()
        errors = np.zeros(len(pose), np.float64)
        if failure == "identity_mutation": z[0, 0] = 1
        elif failure == "pose_mutation": pose[0, 0] = 1
        elif failure == "target_mutation": tgt[0, 0] = 1
        elif failure == "nan": candidate[0, 0] = np.nan
        elif failure == "float32": candidate = candidate.astype(np.float32)
        elif failure == "negative": errors[0] = -1
        return candidate, errors
    def replay(pose, z):
        vertices = _linear_replay(pose, z)
        if failure == "replay_mutation": z[0, 0] = 1
        elif failure == "replay_mismatch": vertices += 1
        return vertices
    with pytest.raises(ValueError): diagnostic.polish_probes(converted, report, np.zeros((5, 18439, 3)), pose_polish=polish, reference_f32=replay)

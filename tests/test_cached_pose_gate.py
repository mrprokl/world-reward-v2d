"""Tiny stub/report tests; never render, import Torch, or benchmark locally."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_test_cached_pose_gate", infra / "cached_pose_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    rows = [{"hypothesis_index": i, "initial_silhouette_iou": .4, "fitted_silhouette_iou": .5,
             "selected_silhouette_iou": .5, "selected_depth_residual_m": .1,
             "icp_accepted_by_image_gate": True,
             "rotation": np.eye(3).tolist(), "translation": [0., 0., 3.],
             "icp": {"rotation": np.eye(3).tolist(), "translation": [0., 0., 3.],
                     "schema": "unchanged-schema", "status": "improved", "inliers": 1638,
                     "iterations": 17, "initial_residual": .2, "final_residual": .1,
                     "scale_fitted": False}} for i in range(24)]
    rows[11]["selected_silhouette_iou"] = .8
    return rows, []


def test_fixed_new_whole_candidate_protocol_not_old_budget_relaxation(gate):
    assert (gate.FRAME_INDEX, gate.NUMERICAL_ATOL, gate.MINIMUM_SPEEDUP, gate.MAXIMUM_SECONDS) == (0, 1e-6, 1.3, 60.)
    assert (gate.WIDTH, gate.HEIGHT, gate.SURFACE_SAMPLES, gate.OBSERVATIONS, gate.SAMPLE_SEED) == (1536, 1152, 8192, 2048, 0)
    assert gate.RADII == (.35, .21, .28)
    assert gate.trial_orders() == (("original", "cached"), ("cached", "original"), ("original", "cached"))


def test_isolated_binding_same_bytecode_kwdefaults_without_global_mutation(gate):
    original = gate.pose_candidates.evaluate_pose_candidates
    before = original.__globals__.copy()
    copied = gate.isolated_cached_evaluator()
    assert copied.__code__ is original.__code__ and copied.__closure__ is original.__closure__
    assert copied.__defaults__ is original.__defaults__
    assert copied.__kwdefaults__ == original.__kwdefaults__ == {"render_batch_size": 1}
    assert copied.__globals__ is not original.__globals__
    assert copied.__globals__["align_observed_points"] is gate.align_observed_points_cached
    assert original.__globals__["align_observed_points"] is gate.align_observed_points
    assert before.keys() == original.__globals__.keys()
    assert all(original.__globals__[key] is value for key, value in before.items())
    assert {key for key in before if copied.__globals__[key] is not before[key]} == {"align_observed_points"}


def test_already_patched_production_solver_fails_not_hide_change(gate, monkeypatch):
    monkeypatch.setattr(gate.pose_candidates, "align_observed_points", lambda: None)
    with pytest.raises(RuntimeError, match="binding has changed"):
        gate.isolated_cached_evaluator()


def test_full_signature_allows_only_small_pose_residual_float_error(gate):
    first, second = candidates(), candidates()
    second[0][0]["rotation"][0][0] += 5e-7
    second[0][0]["icp"]["translation"][2] -= 3e-7
    second[0][0]["icp"]["initial_residual"] += 1e-16
    report = gate.compare_candidates(first, second)
    assert report["selected_hypothesis_index"] == report["cached_selected_hypothesis_index"] == 11
    assert report["exact_slots_images_decisions_status_inliers_iterations"] is True
    assert report["complete_candidate_json_equal"] is False
    assert report["maximum_pose_residual_absolute_error"] == pytest.approx(5e-7)
    assert report["original_candidates_sha256"] != report["cached_candidates_sha256"]


@pytest.mark.parametrize("field", ["initial_silhouette_iou", "fitted_silhouette_iou", "selected_silhouette_iou",
                                   "icp_accepted_by_image_gate", "status", "iterations", "inliers", "scale_fitted",
                                   "rotation", "translation", "selected_depth_residual_m", "extra", "order", "type"])
def test_branch_image_report_topology_and_large_numeric_change_fail(gate, field):
    first, second = candidates(), candidates()
    row = second[0][0]
    if "iou" in field: row[field] += 1e-16
    elif field == "icp_accepted_by_image_gate": row[field] = False
    elif field == "status": row["icp"][field] = "unchanged"
    elif field in ("iterations", "inliers"): row["icp"][field] += 1
    elif field == "scale_fitted": row["icp"][field] = True
    elif field == "rotation": row[field][0][1] += 2e-6
    elif field == "translation": row[field][2] += 2e-6
    elif field == "selected_depth_residual_m": row[field] += 2e-6
    elif field == "extra": row["icp"][field] = "invented"
    elif field == "order": second[0][0], second[0][1] = second[0][1], second[0][0]
    elif field == "type": row["icp"]["iterations"] = float(row["icp"]["iterations"])
    diagnostics = {}
    with pytest.raises(RuntimeError, match="parity"):
        gate.compare_candidates(first, second, diagnostics)
    assert diagnostics["exact_slots_images_decisions_status_inliers_iterations"] is False
    assert diagnostics["original"]["candidates"] == first[0]
    assert diagnostics["cached"]["candidates"] == second[0]


def test_best_index_change_reject_string_change_and_invalid_slots_fail(gate):
    first, second = candidates(), candidates()
    second[0][12]["selected_silhouette_iou"] = .9
    with pytest.raises(RuntimeError, match="selected hypothesis"):
        gate.compare_candidates(first, second)
    first[0].pop(0); second = copy.deepcopy(first)
    first[1].append({"hypothesis_index": 0, "reason": "near plane"})
    second[1].append({"hypothesis_index": 0, "reason": "underconstrained"})
    with pytest.raises(RuntimeError, match="parity"):
        gate.compare_candidates(first, second)
    with pytest.raises(RuntimeError, match="Every hypothesis"):
        gate.compare_candidates((first[0], []), first)
    second = candidates(); second[0][0]["icp"]["final_residual"] = float("nan")
    with pytest.raises(ValueError):
        gate.compare_candidates(candidates(), second)


def stub_run(gate, monkeypatch, *, mismatch=False, error=None, clock=None):
    calls = []
    fixture = [object(), object(), object()]
    monkeypatch.setattr(gate, "_inputs", lambda: ([ (frame,) for frame in fixture], [{"frame_index": i} for i in range(3)]))
    monkeypatch.setattr(gate, "_deadline", lambda _: None)
    if clock is None: clock = iter([0., 2., 2., 3., 3., 4., 4., 6., 6., 8., 8., 9.])
    monkeypatch.setattr(gate.time, "perf_counter", lambda: next(clock))
    def original(value, *, render_batch_size):
        calls.append(("original", value, render_batch_size)); return candidates()
    def cached(value, *, render_batch_size):
        calls.append(("cached", value, render_batch_size))
        if error: raise error
        result = candidates()
        if mismatch: result[0][0]["icp"]["iterations"] += 1
        return result
    monkeypatch.setattr(gate.pose_candidates, "evaluate_pose_candidates", original)
    monkeypatch.setattr(gate, "isolated_cached_evaluator", lambda: cached)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    return calls, fixture, torch


def test_whole_pipeline_three_alternating_trials_fixed_frame_scalar_only(gate, monkeypatch):
    calls, fixture, torch = stub_run(gate, monkeypatch)
    report = {}; gate.run_gate(torch, report, 0.)
    assert [mode for mode, _, _ in calls] == ["original", "cached", "original", "cached", "cached", "original", "original", "cached"]
    assert all(value is fixture[0] and batch == 1 for _, value, batch in calls)
    assert report["evaluated_frame_index"] == 0 and len(report["trial_parity"]) == 6
    assert report["synchronized_whole_candidate_trials_seconds"] == {"original": [2., 2., 2.], "cached": [1., 1., 1.]}
    assert report["median_whole_candidate_speedup"] == 2. and report["throughput_hypothesis_accepted"] is True


def test_failed_parity_persists_outputs_and_does_not_start_timing(gate, monkeypatch):
    calls, _, torch = stub_run(gate, monkeypatch, mismatch=True)
    report = {}
    with pytest.raises(RuntimeError, match="parity"):
        gate.run_gate(torch, report, 0.)
    assert len(calls) == 2 and "synchronized_whole_candidate_trials_seconds" not in report
    assert report["parity"]["exact_slots_images_decisions_status_inliers_iterations"] is False
    assert "original" in report["parity"] and "cached" in report["parity"]


def test_backend_runtime_failure_propagates_never_becomes_evidence(gate, monkeypatch):
    _, _, torch = stub_run(gate, monkeypatch, error=RuntimeError("GPU failure"))
    with pytest.raises(RuntimeError, match="GPU failure"):
        gate.run_gate(torch, {}, 0.)


def test_speed_below_predeclared_threshold_retains_all_timings(gate, monkeypatch):
    clock = iter(float(i) for i in range(12))
    _, _, torch = stub_run(gate, monkeypatch, clock=clock)
    report = {}
    with pytest.raises(RuntimeError, match="below predeclared 1.3"):
        gate.run_gate(torch, report, 0.)
    assert report["throughput_hypothesis_accepted"] is False
    assert report["synchronized_whole_candidate_trials_seconds"] == {"original": [1.] * 3, "cached": [1.] * 3}


def test_deadline_sixty_seconds_exact(gate, monkeypatch):
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 59.999)
    gate._deadline(0.)
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 60.)
    with pytest.raises(RuntimeError, match="60 seconds"):
        gate._deadline(0.)


def test_frozen_output_stops_before_torch_or_fixture(gate, monkeypatch, tmp_path):
    output = tmp_path / "results/cached-pose-gate.json"
    output.parent.mkdir(); output.write_text("frozen")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("/sys/class/net/lo")]) if str(p) == "/sys/class/net" else original(p))
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(gate, "_inputs", lambda: pytest.fail("No local rendering"))
    with pytest.raises(FileExistsError, match="frozen"):
        gate.main([])
    assert output.read_text() == "frozen"


@pytest.mark.parametrize("argv", [["--episode", "15"], ["--atol", "0.01"], ["--max-seconds", "120"], ["--frame", "2"]])
def test_no_tunable_controls_or_challenge_arguments(gate, argv):
    with pytest.raises(SystemExit): gate.main(argv)


def test_wrapper_network_none_gpu_source_readonly_results_only(gate):
    wrapper = Path(gate.__file__).with_name("run_cached_pose_gate.sh").read_text()
    assert "--network none" in wrapper and "--gpus all" in wrapper and "--cpus 4" in wrapper
    assert "src=$CODE,dst=$CODE,readonly" in wrapper and "src=$ROOT/results,dst=$ROOT/results" in wrapper
    assert "src=$ROOT/data" not in wrapper and "src=$ROOT/weights" not in wrapper
    assert "OMP_NUM_THREADS=1" in wrapper and "world-reward/cari4d-source:0.1" in wrapper
    assert '"$CODE/infra/cached_pose_gate.py"' in wrapper and '(( $# == 0 ))' in wrapper

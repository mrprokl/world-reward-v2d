"""Tiny planning/report tests, never render or run ICP/Torch locally."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_pose_batch_gate", infra / "pose_batch_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_geometry_camera_samples_and_trials_are_fixed(gate):
    assert gate.RADII == (.35, .21, .28)
    assert (gate.WIDTH, gate.HEIGHT, gate.SURFACE_SAMPLES, gate.OBSERVATIONS, gate.SAMPLE_SEED) == (1536, 1152, 8192, 2048, 0)
    np.testing.assert_array_equal(gate.K, [[1920, 0, 768], [0, 1920, 576], [0, 0, 1]])
    assert gate.MINIMUM_SPEEDUP == 1.3 and gate.MAXIMUM_SECONDS == 120
    assert gate.trial_orders() == ((1, 8), (8, 1), (1, 8))


def test_proper_known_poses_and_generic_24_seeds_without_truth_pose(gate):
    rotations, translations = gate.controlled_poses()
    seeds = gate.orientation_seeds()
    assert rotations.shape == (3, 3, 3) and translations.shape == (3, 3) and len(seeds) == 24
    np.testing.assert_array_equal(translations[:, 2], [3.2, 3.5, 3.8])
    for rotation in [*rotations, *seeds]:
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.)
    assert len({rotation.tobytes() for rotation in seeds}) == 24
    assert all(not np.allclose(seed, truth) for seed in seeds for truth in rotations)
    for first, second in zip(seeds, gate.orientation_seeds(), strict=True):
        np.testing.assert_array_equal(first, second)


def candidates():
    return [{"hypothesis_index": i, "selected_silhouette_iou": .5, "selected_depth_residual_m": .1,
             "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 3],
             "icp": {"status": "unchanged", "iterations": 2}} for i in range(24)]


def test_signature_exact_candidate_rejection_best_and_tie_rules(gate):
    rows = candidates()
    rows[10]["selected_silhouette_iou"] = rows[11]["selected_silhouette_iou"] = .8
    rows[11]["selected_depth_residual_m"] = .05
    rows[12].update(rows[11], hypothesis_index=12)
    signature, best = gate.candidate_signature(rows, [])
    result = json.loads(signature)
    assert best == 11 and result["selected"] == rows[11]
    assert result["candidates"] == rows and result["rejected_candidates"] == []
    rows[0]["icp"]["iterations"] = 3
    assert gate.candidate_signature(rows, [])[0] != signature
    rejected = [{"hypothesis_index": 5, "reason": "underconstrained"}]
    signature, best = gate.candidate_signature([row for row in rows if row["hypothesis_index"] != 5], rejected)
    assert json.loads(signature)["rejected_candidates"] == rejected and best == 11


@pytest.mark.parametrize("mode", ["empty", "missing", "duplicate", "nonfinite"])
def test_invalid_candidate_signatures_fail_not_hide_unsupported_slots(gate, mode):
    rows = candidates()
    if mode == "empty": rows = []
    if mode == "missing": rows.pop()
    if mode == "duplicate": rows[-1]["hypothesis_index"] = 0
    if mode == "nonfinite": rows[0]["selected_silhouette_iou"] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        gate.candidate_signature(rows, [])


def test_deadline_predeclared_no_infinite_run(gate, monkeypatch):
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 119.999)
    gate._deadline(0.)
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 120.)
    with pytest.raises(RuntimeError, match="120 seconds"):
        gate._deadline(0.)


def test_frozen_report_stops_before_torch_or_fixture(gate, monkeypatch, tmp_path):
    output = tmp_path / "results/pose-batch-gate.json"
    output.parent.mkdir()
    output.write_text("frozen")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("/sys/class/net/lo")]) if str(p) == "/sys/class/net" else original(p))
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(gate, "_inputs", lambda: pytest.fail("Never construct/render fixture locally"))
    with pytest.raises(FileExistsError, match="frozen"):
        gate.main([])
    assert output.read_text() == "frozen"


@pytest.mark.parametrize("argv", [["--batch-size", "8"], ["--episode", "15"], ["true"]])
def test_gate_has_no_tunable_geometry_or_challenge_arguments(gate, argv):
    with pytest.raises(SystemExit):
        gate.main(argv)


def test_wrapper_original_cuda_network_none_source_readonly_results_only(gate):
    wrapper = Path(gate.__file__).with_name("run_pose_batch_gate.sh").read_text()
    assert "--network none" in wrapper and "--gpus all" in wrapper
    assert "src=$CODE,dst=$CODE,readonly" in wrapper and "src=$ROOT/results,dst=$ROOT/results" in wrapper
    assert "src=$ROOT/data" not in wrapper and "src=$ROOT/weights" not in wrapper
    assert "world-reward/cari4d-source:0.1" in wrapper
    assert '"$CODE/infra/pose_batch_gate.py" "$@"' in wrapper

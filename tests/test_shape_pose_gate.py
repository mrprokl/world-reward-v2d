"""Tiny fixture/crossfit/report tests, no camera rendering or CUDA solver."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_shape_pose_gate", infra / "shape_pose_gate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_new_fixtures_views_and_pixel_policy_frozen(gate):
    assert gate.FAMILIES == (("ellipsoid", (.31, .17, .29)), ("box", (.48, .30, .34)))
    assert gate.THETAS == ((0., 0., 0., 0., 0.), (.035, -.02, .009, -.011, .006))
    assert gate.PIXEL_POINTS == 128 and gate.SURFACE_SAMPLES == 8192 and gate.SURFACE_SEED == 37
    assert gate.MAX_SECONDS == 120. and gate.IMAGE_HW == (1152, 1536)
    R, t = gate.supplied_poses()
    assert R.shape == (3, 3, 3) and t.shape == (3, 3)
    np.testing.assert_array_equal(t, [[-.13, .09, 3.55], [.11, -.07, 3.95], [.03, .14, 4.3]])


def test_biased_supplied_pose_exact_new_so3_and_translation(gate):
    from scipy.spatial.transform import Rotation
    R, t = gate.supplied_poses(); R1, t1 = gate.supplied_poses(True)
    np.testing.assert_allclose(Rotation.from_matrix(R1 @ R.transpose(0, 2, 1)).as_rotvec(),
                               np.broadcast_to([.012, -.004, .007], (3, 3)), atol=1e-15)
    np.testing.assert_allclose(t1 - t, np.broadcast_to([.003, -.004, .002], (3, 3)), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R1), 1., atol=1e-15)
    with pytest.raises(ValueError): gate.supplied_poses(1)


def test_checkerboard_splits_disjoint_original_pixels_without_occluder_or_mutation(gate):
    mask = np.zeros((12, 12), bool); mask[2:10, 2:10] = True
    copy = mask.copy()
    training, heldout, occluder = gate.split_visible_pixels(mask)
    np.testing.assert_array_equal(mask, copy)
    assert not (training & heldout).any()
    np.testing.assert_array_equal(training | heldout, mask & ~occluder)
    assert occluder[:4].all() and not occluder[4:].any()
    yy, xx = np.indices(mask.shape)
    assert ((xx[training] + yy[training]) % 2 == 0).all()
    assert ((xx[heldout] + yy[heldout]) % 2 == 1).all()
    assert not (training & occluder).any() and not (heldout & occluder).any()


@pytest.mark.parametrize("mask", [np.zeros((3, 3), bool), np.ones((3, 3)), np.ones(3, bool),
                                  np.ma.array(np.ones((3, 3), bool), mask=False), np.ones((1, 2), bool)])
def test_invalid_or_too_sparse_pixel_split_rejected(gate, mask):
    with pytest.raises(ValueError): gate.split_visible_pixels(mask)


def test_truth_cannot_enter_selected_model_only_policy_decision(gate):
    models = ("M0", "M1")
    assert gate.selected_parameters(models, SimpleNamespace(decision="pose_only")) == "M0"
    assert gate.selected_parameters(models, SimpleNamespace(decision="shape_pose_proposal")) == "M1"
    source = Path(gate.__file__).read_text()
    assert "ambiguity = True" in source
    assert "unknown_pose_modes_forced_M0" in source
    assert "raw_M1_outcome_gate_pass" in source


def test_correct_control_strict_zero_and_raw_deformed_five_percent_gate(gate):
    assert gate.outcome_gate(True, 0., 0., True, True)
    assert gate.outcome_gate(True, 0., 1e-12, True, True)
    assert not gate.outcome_gate(True, 0., 1.01e-12, True, True)
    assert gate.outcome_gate(False, .01, .0095, True, True)
    assert not gate.outcome_gate(False, .01, .00951, True, True)
    assert not gate.outcome_gate(False, .01, .008, False, True)
    with pytest.raises(ValueError): gate.outcome_gate(True, .0001, 0., True, True)
    with pytest.raises(ValueError): gate.outcome_gate(False, 0., 0., True, True)


@pytest.mark.parametrize("argument", ["--max-evaluations", "--prior", "--episode", "--oracle", "--helpful"])
def test_no_tunable_cli_before_cuda_or_environment(gate, monkeypatch, argument):
    monkeypatch.setattr(gate.platform, "system", lambda: (_ for _ in ()).throw(AssertionError("must reject args first")))
    with pytest.raises(SystemExit): gate.main([argument])


@pytest.fixture
def fake_runtime(gate, monkeypatch, tmp_path):
    import world_reward.continuous_surface as primitive
    (tmp_path / "results").mkdir()
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    primitive_path = tmp_path / "results/continuous-surface-gate.json"
    primitive_path.write_text(json.dumps({"status": "pass", "stage": "own_continuous_surface_analytic_cuda",
                                         "challenge_inputs_used": False,
                                         "primitive_sha256": hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest()}))
    torch = SimpleNamespace(__version__="fake-test-only", cuda=SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda: "fake-no-gpu", synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return tmp_path, primitive_path


def test_runtime_exception_persists_partial_active_condition(gate, fake_runtime, monkeypatch):
    root, _ = fake_runtime
    def fail(torch, report, deadline):
        report["conditions"].append({"falsification_gate_pass": False})
        report["active_condition"] = {"partial": True}
        raise TimeoutError("synthetic hard deadline")
    monkeypatch.setattr(gate, "_experiment", fail)
    with pytest.raises(TimeoutError): gate.main([])
    report = json.loads((root / "results/shape-pose-gate.json").read_text())
    assert report["status"] == "fail" and report["shape_hypothesis_accepted"] is False
    assert report["conditions"] == [{"falsification_gate_pass": False}] and report["active_condition"] == {"partial": True}
    assert report["adoption_authorized"] is False and report["error_type"] == "TimeoutError"


@pytest.mark.parametrize("passed", [False, True])
def test_complete_measurement_never_claims_crossview_calibration_or_adoption(gate, fake_runtime, monkeypatch, passed):
    root, _ = fake_runtime
    monkeypatch.setattr(gate, "_experiment", lambda torch, report, deadline: report["conditions"].extend(
        {"falsification_gate_pass": passed} for _ in range(8)))
    gate.main([])
    report = json.loads((root / "results/shape-pose-gate.json").read_text())
    assert report["status"] == "pass" and report["shape_hypothesis_accepted"] is passed
    assert report["validation_unit"] == "within_supplied_view_pixel_crossfit_proxy"
    for field in ("adoption_authorized", "statistical_calibration_verified", "independent_temporal_blocks_verified",
                  "unseen_view_pose_transfer_verified", "global_pose_modes_checked", "RGB_pipeline_validated", "challenge_performance_verified"):
        assert report[field] is False
    assert report["synthetic_pose_oracle"] is True


def test_missing_conditions_are_not_filtered_into_success(gate, fake_runtime, monkeypatch):
    root, _ = fake_runtime
    monkeypatch.setattr(gate, "_experiment", lambda torch, report, deadline: report["conditions"].append({"falsification_gate_pass": True}))
    with pytest.raises(RuntimeError, match="All8"): gate.main([])
    assert json.loads((root / "results/shape-pose-gate.json").read_text())["status"] == "fail"


def test_failed_primitive_gate_never_runs_fitter(gate, fake_runtime, monkeypatch):
    root, path = fake_runtime
    report = json.loads(path.read_text()); report["status"] = "fail"; path.write_text(json.dumps(report))
    monkeypatch.setattr(gate, "_experiment", lambda *args: (_ for _ in ()).throw(AssertionError("must not fit")))
    with pytest.raises(RuntimeError, match="primitive"): gate.main([])
    assert json.loads((root / "results/shape-pose-gate.json").read_text())["conditions"] == []


def test_existing_report_never_overwritten(gate, fake_runtime):
    root, _ = fake_runtime
    output = root / "results/shape-pose-gate.json"; output.write_text("frozen")
    with pytest.raises(FileExistsError): gate.main([])
    assert output.read_text() == "frozen"


def test_wrapper_only_code_and_scalar_results_no_assets(gate):
    source = Path(gate.__file__).with_name("run_shape_pose_gate.sh").read_text()
    assert "--gpus all --network none" in source
    assert "src=$CODE,dst=$CODE,readonly" in source and "src=$ROOT/results,dst=$ROOT/results" in source
    assert not any(f"src=$ROOT/{name}" in source for name in ("data", "weights", "outputs", "vendor"))

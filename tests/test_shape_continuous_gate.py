"""Frozen fixture/math/routing guards; no rendered or CUDA experiment locally."""

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
    spec = importlib.util.spec_from_file_location("world_reward_test_shape_continuous_gate", infra / "shape_continuous_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_fixture_does_not_retune_old_rejected_cases(gate):
    assert gate.MESH_FAMILIES == (
        ("ellipsoid", "icosphere_subdivisions_2", (.33, .19, .27)),
        ("rectangular_box", "closed_box", (.52, .26, .38)),
    )
    assert gate.TRUTH_PARAMETERS == ((0., 0., 0., 0., 0.), (.045, -.025, .012, -.015, .008))
    assert gate.FIT_FRAMES == (0, 2, 4) and gate.HELDOUT_FRAMES == (1, 3, 5)
    assert set(gate.FIT_FRAMES).isdisjoint(gate.HELDOUT_FRAMES)
    assert sorted(gate.FIT_FRAMES + gate.HELDOUT_FRAMES) == list(range(6))
    assert gate.SURFACE_SAMPLES == 8192 and gate.MAX_OBSERVATIONS == 2048 and gate.SAMPLE_SEED == 19
    assert gate.OCCLUSION_FRACTION == .35 and gate.MAX_GATE_SECONDS == 120.
    assert gate.ROTATION_BIAS == (.008, -.006, .004)
    assert gate.TRANSLATION_BIAS == (.004, -.002, .003)
    np.testing.assert_array_equal(gate.K, [[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])


def test_new_pose_path_proper_deterministic_and_exact_fixed_bias(gate):
    from scipy.spatial.transform import Rotation
    original_R, original_t = gate.controlled_poses()
    biased_R, biased_t = gate.controlled_poses(True)
    np.testing.assert_allclose(np.linalg.det(original_R), 1., atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(biased_R), 1., atol=1e-15)
    np.testing.assert_allclose(biased_t - original_t, np.broadcast_to(gate.TRANSLATION_BIAS, (6, 3)), atol=1e-15)
    np.testing.assert_allclose(Rotation.from_matrix(biased_R @ original_R.transpose(0, 2, 1)).as_rotvec(),
                               np.broadcast_to(gate.ROTATION_BIAS, (6, 3)), atol=1e-15)
    again_R, again_t = gate.controlled_poses(True)
    np.testing.assert_array_equal(again_R, biased_R)
    np.testing.assert_array_equal(again_t, biased_t)
    np.testing.assert_array_equal(original_t[:, 2], [3.25, 3.65, 3.4, 4.15, 3.8, 4.])


@pytest.mark.parametrize("value", [None, 1, 0, "false", []])
def test_pose_bias_requires_explicit_boolean(gate, value):
    with pytest.raises(ValueError): gate.controlled_poses(value)


def test_horizontal_bottom_occlusion_deterministic_no_mask_mutation(gate):
    mask = np.zeros((12, 8), bool)
    mask[1:11, 2:6] = True
    before = mask.copy()
    visible, hidden, fraction = gate.visible_region(mask)
    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(visible, mask & ~hidden)
    assert fraction == pytest.approx(.4)
    assert hidden[7:, :].all() and not hidden[:7].any()
    assert not np.any(visible & ~mask)
    repeated = gate.visible_region(mask)
    np.testing.assert_array_equal(repeated[0], visible)
    np.testing.assert_array_equal(repeated[1], hidden)


def test_irregular_occlusion_reports_actual_not_requested_fraction(gate):
    mask = np.zeros((4, 5), bool)
    mask[0, 1] = mask[1, 2] = mask[3, 4] = True
    visible, hidden, actual = gate.visible_region(mask)
    assert actual == pytest.approx(1 / 3)
    assert np.count_nonzero(visible) == 2
    assert hidden[2:].all() and not hidden[:2].any()


@pytest.mark.parametrize("mask", [np.zeros((3, 3), bool), np.ones((3, 3)), np.ones(3, bool),
                                  np.ma.array(np.ones((3, 3), bool), mask=False)])
def test_invalid_visibility_fails_instead_of_repair(gate, mask):
    with pytest.raises(ValueError): gate.visible_region(mask)


def test_fitter_receives_only_fit_frames_fixed_poses_and_base_mesh_no_truth(gate, monkeypatch):
    import world_reward.shape_fit_continuous as fitting
    received = []
    def fake(*args, **kwargs):
        received.append((args, kwargs))
        return "proposal"
    monkeypatch.setattr(fitting, "fit_shared_shape_continuous", fake)
    vertices, faces, pivot = np.ones((4, 3)), np.array([[0, 1, 2]]), np.zeros(3)
    observations = [np.full((3, 3), i) for i in range(6)]
    rotations, translations = gate.controlled_poses()
    assert gate.fit_condition(vertices, faces, pivot, observations, rotations, translations) == "proposal"
    args, kwargs = received[0]
    assert args[0] is vertices and args[1] is faces and args[2] is pivot and kwargs == {}
    assert all(left is observations[index] for left, index in zip(args[3], (0, 2, 4), strict=True))
    np.testing.assert_array_equal(args[4], rotations[[0, 2, 4]])
    np.testing.assert_array_equal(args[5], translations[[0, 2, 4]])
    assert len(args) == 6


@pytest.mark.parametrize("field", ["observations", "rotations", "translations"])
def test_incomplete_six_frame_condition_never_reaches_fitter(gate, field):
    values = dict(observations=[np.ones((3, 3)) for _ in range(6)],
                  rotations=np.broadcast_to(np.eye(3), (6, 3, 3)), translations=np.zeros((6, 3)))
    values[field] = values[field][:-1]
    with pytest.raises(ValueError, match="six"):
        gate.fit_condition(np.ones((4, 3)), [[0, 1, 2]], np.zeros(3), **values)


def test_correct_control_strict_zero_and_deformed_five_percent_canonical_gain(gate):
    assert gate.condition_gate(True, 0., 0., .8, .8, True, True)
    assert gate.condition_gate(True, 0., 1e-12, .8, .8, True, True)
    assert not gate.condition_gate(True, 0., 1.01e-12, .8, .8, True, True)
    assert gate.condition_gate(False, .01, .0095, .8, .76, True, True)
    assert not gate.condition_gate(False, .01, .00951, .8, .8, True, True)
    assert not gate.condition_gate(False, .01, .009, .8, .759, True, True)
    assert not gate.condition_gate(False, .01, .009, .8, .8, False, True)
    assert not gate.condition_gate(False, .01, .009, .8, .8, True, False)


@pytest.mark.parametrize("field,value", [("control", 1), ("fixed_topology", 1), ("positive_volume", None),
                                        ("canonical_before", np.nan), ("canonical_after", np.inf),
                                        ("heldout_before", -1.), ("heldout_after", 1.1),
                                        ("canonical_after", True), ("heldout_after", "1")])
def test_ambiguous_invalid_gate_inputs_cannot_fake_success(gate, field, value):
    values = dict(control=False, canonical_before=.01, canonical_after=.009,
                  heldout_before=.8, heldout_after=.8, fixed_topology=True, positive_volume=True)
    values[field] = value
    with pytest.raises(ValueError): gate.condition_gate(**values)


def test_control_must_be_exact_zero_and_deformed_baseline_positive(gate):
    with pytest.raises(ValueError, match="exactly zero"):
        gate.condition_gate(True, .001, 0., .8, .8, True, True)
    with pytest.raises(ValueError, match="positive"):
        gate.condition_gate(False, 0., 0., .8, .8, True, True)


def test_correct_control_truth_has_no_identity_deformation_roundtrip(gate):
    source = Path(gate.__file__).read_text()
    assert "truth = base.copy() if control else apply_fixed_shape" in source
    assert "truth_samples = samples.copy() if control else apply_fixed_shape" in source


def test_unaccepted_proposal_keeps_exact_base_geometry(gate):
    base = np.arange(12, dtype=float).reshape(4, 3)
    proposal = SimpleNamespace(accepted=False, params5=np.full(5, .01))
    adopted = gate.adopted_vertices(base, np.zeros(3), proposal)
    np.testing.assert_array_equal(adopted, base)
    assert adopted is not base


def test_deadline_frozen_no_soft_extension(gate, monkeypatch):
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 120.)
    gate._budget(0.)
    monkeypatch.setattr(gate.time, "perf_counter", lambda: 120.001)
    with pytest.raises(RuntimeError, match="120s"): gate._budget(0.)


@pytest.mark.parametrize("argument", ["--max-evaluations", "--shape-prior", "--episode", "--gt", "--backend", "--helpful"])
def test_no_tunable_experiment_cli_before_remote_or_cuda(gate, monkeypatch, argument):
    monkeypatch.setattr(gate.platform, "system", lambda: (_ for _ in ()).throw(AssertionError("must reject args first")))
    with pytest.raises(SystemExit): gate.main([argument])


@pytest.fixture
def fake_main_runtime(gate, monkeypatch, tmp_path):
    import world_reward.continuous_surface as primitive
    (tmp_path / "results").mkdir()
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    passing = {"stage": "own_continuous_surface_analytic_cuda", "status": "pass", "own_procedural_inputs_only": True,
               "challenge_inputs_used": False, "primitive_sha256": hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest()}
    primitive_path = tmp_path / "results/continuous-surface-gate.json"
    primitive_path.write_text(json.dumps(passing))
    torch = SimpleNamespace(__version__="fake-tiny-test", cuda=SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda: "fake-no-gpu", synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return tmp_path, primitive_path, torch


def test_failure_persists_partial_conditions_without_false_acceptance(gate, fake_main_runtime, monkeypatch):
    root, _, _ = fake_main_runtime
    def failed(torch, report, started):
        report["conditions"].append({"falsification_gate_pass": False, "partial": True})
        report["active_condition"] = {"mesh_family": "test-only-partial"}
        raise RuntimeError("synthetic deliberate failure")
    monkeypatch.setattr(gate, "_experiment", failed)
    with pytest.raises(RuntimeError, match="deliberate failure"): gate.main([])
    report = json.loads((root / "results/shape-continuous-gate.json").read_text())
    assert report["status"] == "fail" and report["shape_hypothesis_accepted"] is False
    assert report["conditions"] == [{"falsification_gate_pass": False, "partial": True}]
    assert report["active_condition"]["mesh_family"] == "test-only-partial"
    assert report["error_type"] == "RuntimeError" and report["adoption_authorized"] is False


@pytest.mark.parametrize("passed", [False, True])
def test_complete_measurement_records_all_gates_and_never_claims_adoption(gate, fake_main_runtime, monkeypatch, passed):
    root, _, _ = fake_main_runtime
    def complete(torch, report, started):
        report["conditions"].extend({"falsification_gate_pass": passed} for _ in range(8))
    monkeypatch.setattr(gate, "_experiment", complete)
    gate.main([])
    report = json.loads((root / "results/shape-continuous-gate.json").read_text())
    assert report["status"] == "pass" and report["shape_hypothesis_accepted"] is passed
    assert report["decision"] == ("advance_to_nonoracle_validation" if passed else "stop_keep_experimental")
    assert report["synthetic_pose_oracle"] is True and report["RGB_pipeline_validated"] is False
    assert report["isolated_discretization_causal_claim"] is False and report["adoption_authorized"] is False
    assert report["joint_pose_shape_selection_validated"] is False
    assert report["challenge_inputs_used"] is False and report["challenge_performance_verified"] is False
    assert report["fit_configuration"] == {"shape_prior": .01, "robust_transition_m": .01, "max_evaluations": 100, "point_chunk_size": 64}


def test_filtered_incomplete_conditions_fail_instead_of_selecting_easy_cases(gate, fake_main_runtime, monkeypatch):
    root, _, _ = fake_main_runtime
    monkeypatch.setattr(gate, "_experiment", lambda torch, report, started: report["conditions"].append({"falsification_gate_pass": True}))
    with pytest.raises(RuntimeError, match="eight"): gate.main([])
    assert json.loads((root / "results/shape-continuous-gate.json").read_text())["status"] == "fail"


@pytest.mark.parametrize("field,value", [("status", "fail"), ("challenge_inputs_used", True),
                                        ("stage", "wrong"), ("primitive_sha256", "b" * 64)])
def test_wrong_or_failed_primitive_gate_fails_before_render_and_persists(gate, fake_main_runtime, monkeypatch, field, value):
    root, path, _ = fake_main_runtime
    source = json.loads(path.read_text())
    source[field] = value
    path.write_text(json.dumps(source))
    monkeypatch.setattr(gate, "_experiment", lambda *args: (_ for _ in ()).throw(AssertionError("must not render")))
    with pytest.raises(RuntimeError, match="exact primitive bytes"): gate.main([])
    report = json.loads((root / "results/shape-continuous-gate.json").read_text())
    assert report["status"] == "fail" and report["conditions"] == []


def test_existing_report_is_never_overwritten(gate, fake_main_runtime):
    root, _, _ = fake_main_runtime
    path = root / "results/shape-continuous-gate.json"
    path.write_text("frozen")
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_text() == "frozen"


def test_wrapper_has_only_readonly_code_and_scalar_results_mounts(gate):
    path = Path(gate.__file__).with_name("run_shape_continuous_gate.sh")
    text = path.read_text()
    assert "--gpus all --network none" in text
    assert "src=$CODE,dst=$CODE,readonly" in text
    assert "src=$ROOT/results,dst=$ROOT/results" in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/weights" not in text and "src=$ROOT/outputs" not in text
    assert '"$CODE/infra/shape_continuous_gate.py" "$@"' in text

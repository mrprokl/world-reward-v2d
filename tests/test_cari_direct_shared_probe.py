"""Tiny direct-output/selection/firewall contracts; no official model execution."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_direct_probe", infra / "cari_direct_shared_probe.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def native(probe):
    return {key: np.repeat(np.arange(790, dtype=np.float32)[:, None], dim, axis=1)
            if key != "mhr_face" else np.zeros((790, dim), np.float32)
            for key, dim in probe.convert.PARAMETER_DIMS.items()}


def test_framezero_selection_uniform_indices_not_residual_or_mutation(probe):
    params = native(probe); before = {k: v.copy() for k, v in params.items()}
    selected = probe.shared_inputs(params)
    assert list(probe.INDICES) == np.rint(np.linspace(0, 789, 12)).astype(int).tolist()
    assert sum(probe.convert.PARAMETER_DIMS.values()) == 522
    for key in params:
        assert np.array_equal(params[key], before[key])
        expected = np.repeat(params[key][:1], 12, axis=0) if key in ("mhr_shape", "mhr_scale") else params[key][list(probe.INDICES)]
        assert np.array_equal(selected[key], expected) and not np.shares_memory(selected[key], params[key])


@pytest.mark.parametrize("fault", ["extra", "missing", "dtype", "coverage", "nan", "masked", "expression"])
def test_original_inputs_fail_closed(probe, fault):
    params = native(probe)
    if fault == "extra": params["ground_truth"] = np.zeros(1)
    elif fault == "missing": del params["mhr_shape"]
    elif fault == "dtype": params["mhr_scale"] = params["mhr_scale"].astype(np.float64)
    elif fault == "coverage": params["mhr_hand"] = params["mhr_hand"][:-1]
    elif fault == "nan": params["mhr_shape"][0, 0] = np.nan
    elif fault == "masked": params["mhr_scale"] = np.ma.array(params["mhr_scale"], mask=False)
    else: params["mhr_face"][0, 0] = .1
    with pytest.raises(ValueError): probe.shared_inputs(params)


def outputs(probe, monkeypatch):
    monkeypatch.setattr(probe, "VERTICES", 3)
    return np.ones((12, 3, 3), np.float32), np.zeros((12, 204), np.float32), probe.shared_inputs(native(probe))


def test_predictions_frozen_before_independent_replays_and_exclusive(probe, monkeypatch, tmp_path):
    target, controls, selected = outputs(probe, monkeypatch)
    loaded, saved, files = probe.freeze(tmp_path, target, controls, selected)
    assert not loaded.flags.writeable and np.array_equal(loaded, target)
    assert saved["shape"].shape == (45,) and saved["scale_pca"].shape == (28,)
    assert saved["expression"].shape == (12, 72) and not saved["expression"].any()
    assert len(files) == 2 and all(not p.stat().st_mode & 0o222 for p, _ in files)
    with pytest.raises(FileExistsError): probe.freeze(tmp_path, target, controls, selected)


@pytest.mark.parametrize("fault", ["vertices", "paramdtype", "shape", "nan", "scale", "signedzero", "identity"])
def test_direct_outputs_strict_shared_identity(probe, monkeypatch, fault):
    target, controls, selected = outputs(probe, monkeypatch)
    if fault == "vertices": target = target[:, :2]
    elif fault == "paramdtype": controls = controls.astype(np.float64)
    elif fault == "shape": controls = controls[:, :203]
    elif fault == "nan": target[0, 0, 0] = np.nan
    elif fault == "scale": controls[1, 136] = 1
    elif fault == "signedzero": controls[1, 136] = -0.
    else: selected["mhr_shape"][1, 0] = 1
    with pytest.raises(ValueError): probe.validate_direct(target, controls, selected)


def test_no_alignment_residual_preserves_translation_and_articulation(probe):
    target = np.ones((12, 3, 3), np.float32)
    assert probe.residuals(target, target*1000., units="mm")["max_point_mm"] == 0
    moved = target.astype(np.float64); moved[3, :, 0] += .003
    result = probe.residuals(target, moved, units="m")
    assert result["per_frame_mean_mm"][3] == pytest.approx(3.)
    moved[3, 0, 1] += .004
    assert probe.residuals(target, moved, units="m")["max_point_mm"] == pytest.approx(5.)


@pytest.mark.parametrize("fault", ["units", "shape", "dtype", "nan"])
def test_replay_contract(probe, fault):
    target = np.ones((12, 3, 3), np.float32); recovered = target.copy(); units = "m"
    if fault == "units": units = "cm"
    elif fault == "shape": recovered = recovered[:1]
    elif fault == "dtype": recovered = recovered.astype(np.int32)
    else: recovered[0, 0, 0] = np.nan
    with pytest.raises(ValueError): probe.residuals(target, recovered, units=units)


def test_preserved_failures_only_receipts_no_old_target(probe, monkeypatch, tmp_path):
    first = tmp_path / "first.json"; first.write_text("{}"); monkeypatch.setattr(probe, "failed_source", lambda _: first)
    path = tmp_path / "outputs/episode_000000/cari_target_reseal_refined_v1/report.json"; path.parent.mkdir(parents=True)
    receipt = dict(stage="world_reward_cari_native_target_strict_reseal", status="fail", phase="paired_target_comparison",
        error="Fresh worker source/version/settings/target identity differs", ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[], original_conversion_rerun=False, solver_calls=0)
    path.write_text(json.dumps(receipt)); monkeypatch.setattr(probe, "RESEAL_SHA", probe.sha256(path))
    assert len(probe.preserved_failures(tmp_path)) == 2
    receipt["status"] = "pass"; path.write_text(json.dumps(receipt)); monkeypatch.setattr(probe, "RESEAL_SHA", probe.sha256(path))
    with pytest.raises(ValueError): probe.preserved_failures(tmp_path)


def test_runtime_source_contract_and_offline_wrapper(probe):
    source = Path(probe.__file__).read_text(); tree = ast.parse(source)
    attrs = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"convert", "lm_pose", "lm_joint"} & attrs
    assert "frozen_target(" not in source and "decode_target(" not in source
    assert "global_trans=trans[i:i+1]*context.flip" in source and "return_model_params=True" in source
    assert source.index("freeze(out, target, direct, selected)") < source.index("layer.mhr_forward_vertices(tensor_inputs)")
    assert '"old_target_read": False' in source and '"accuracy_verified": False' in source
    assert '"reference_max_point_is_diagnostic_only": True' in source
    wrapper = Path(probe.__file__).with_name("run_cari_direct_shared_probe.sh")
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "--network none" in text and "--memory 32g" in text and "183s" in text
    assert "--env CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and 'src=$ROOT/data' not in text

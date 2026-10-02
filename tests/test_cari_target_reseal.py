"""Tiny reseal contract fixtures; no native models, CUDA, or full mesh arrays."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def reseal(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_target_reseal", infra / "cari_target_reseal.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_raw_bit_comparison_includes_signed_zero_and_metric_differences(reseal):
    first = np.zeros((2, 3, 3), np.float32); second = first.copy()
    second[0, 0, 0] = -0.; second[1, 0, 2] = .001
    result = reseal.compare_targets(first, second)
    assert result["bitexact"] is False and result["changed_elements"] == 2
    assert result["per_frame_changed_elements"] == [1, 1]
    assert result["per_frame_max_point_L2_mm"][0] == 0
    assert result["max_point_L2_mm"] == pytest.approx(1., abs=1e-6)
    assert result["mean_point_L2_mm"] == pytest.approx(1/6, abs=1e-6)
    assert reseal.compare_targets(first, first.copy())["bitexact"] is True


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "empty"])
def test_comparison_fails_closed(reseal, fault):
    first = np.zeros((2, 3, 3), np.float32); second = first.copy()
    if fault == "shape": second = second[:, :2]
    elif fault == "dtype": second = second.astype(np.float64)
    elif fault == "nan": second[0, 0, 0] = np.nan
    else: first, second = first[:0], second[:0]
    with pytest.raises(ValueError): reseal.compare_targets(first, second)


def test_historical_error_tolerance_is_unchanged_not_2mm_gate(reseal):
    original = np.array([6.28, 1.917, .5], np.float32)
    replayed = original.copy(); replayed[0] += 1e-4
    assert reseal.historical_agreement(original, replayed)["historical_error_agreement"] is True
    replayed[1] += .001
    result = reseal.historical_agreement(original, replayed)
    assert result["historical_error_agreement"] is False and result["drift_failure_frames"] == [1]
    assert result["rtol"] == 1e-5 and result["atol_mm"] == 1e-4


@pytest.mark.parametrize("fault", ["dtype", "shape", "negative", "nan"])
def test_historical_error_contract(reseal, fault):
    original = np.ones(3, np.float32); replayed = original.copy()
    if fault == "dtype": replayed = replayed.astype(np.float64)
    elif fault == "shape": replayed = replayed[:2]
    elif fault == "negative": replayed[0] = -1
    else: replayed[0] = np.nan
    with pytest.raises(ValueError): reseal.historical_agreement(original, replayed)


def workers(reseal):
    common = dict(stage=reseal.STAGE, status="pass", phase="complete", settings=reseal.SETTINGS.copy(),
        native_full_frame_decode_verified=True, script_sha256="a"*64, producer_revision="b"*40,
        image_id="sha256:"+"c"*64, torch="fixture", CUDA="fixture", cudnn=1, GPU="fixture",
        sealed_report_sha256=reseal.SEALED_SHA, failed_probe_report_sha256=reseal.FAILED_SHA,
        native_parameter_identities={}, inference_source_identity={}, body_assets={}, target_vertices_identity={})
    return [dict(copy.deepcopy(common), worker_index=i, worker_pid=os.getpid()+i+100) for i in range(2)]


def test_two_distinct_fresh_complete_workers(reseal):
    reseal.validate_workers(workers(reseal))


@pytest.mark.parametrize("fault", ["pid", "parent_pid", "index", "strict", "version", "hash", "source", "status", "missing"])
def test_worker_contract_does_not_hide_identity_or_setting_changes(reseal, fault):
    values = workers(reseal)
    if fault == "pid": values[1]["worker_pid"] = values[0]["worker_pid"]
    elif fault == "parent_pid": values[1]["worker_pid"] = os.getpid()
    elif fault == "index": values[1]["worker_index"] = 0
    elif fault == "strict": values[1]["settings"]["jit_optimized_execution"] = True
    elif fault == "version": values[1]["torch"] = "different"
    elif fault == "hash": values[1]["target_vertices_identity"] = {"sha256": "d"*64}
    elif fault == "source": values[1]["inference_source_identity"] = {"sha256": "e"*64}
    elif fault == "status": values[1]["status"] = "fail"
    else: del values[0]["CUDA"]
    with pytest.raises(ValueError): reseal.validate_workers(values)


@pytest.mark.parametrize("fault", ["nonce", "worker", "script", "revision", "status"])
def test_fresh_workers_require_parent_authorization(reseal, fault):
    import hashlib
    parent = dict(stage=reseal.STAGE, status="running", pending_worker=0,
        script_sha256=reseal.sha256(Path(reseal.__file__)), producer_revision="a"*40,
        image_id="sha256:"+"b"*64, nonce_sha256=hashlib.sha256(b"secret-fixture").hexdigest())
    nonce = "secret-fixture"
    if fault == "nonce": nonce = "different"
    elif fault == "worker": parent["pending_worker"] = 1
    elif fault == "script": parent["script_sha256"] = "0"*64
    elif fault == "revision": parent["producer_revision"] = "c"*40
    else: parent["status"] = "pass"
    with pytest.raises(RuntimeError): reseal.worker_authorized(parent, 0, nonce=nonce,
        revision="a"*40, image="sha256:"+"b"*64)


def test_readonly_regular_path_rejects_symlink(reseal, tmp_path):
    original = tmp_path / "file"; original.write_text("tiny"); link = tmp_path / "link"; link.symlink_to(original)
    assert reseal.regular(original) == original
    with pytest.raises(ValueError): reseal.regular(link)


def test_unknown_arguments_fail_before_runtime(reseal):
    with pytest.raises(SystemExit): reseal.main(["--waive-hash"])


def test_wrapper_and_source_no_solver_or_historical_overwrite(reseal):
    source = Path(reseal.__file__).read_text(); wrapper = Path(reseal.__file__).with_name("run_cari_target_reseal.sh")
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "--network none" in text and "--memory 32g" in text and "303s" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and "src=$OUT,dst=$OUT" in text
    assert "src=$ROOT/outputs,dst=$ROOT/outputs,readonly" in text
    assert "converter.convert(" not in source and "lm_joint(" not in source and "lm_pose(" not in source
    assert '"old_target_bitexact_recovered": False' in source and '"solver_calls": 0' in source
    assert '"--worker"' in source and "subprocess.run" in source and "optimized_execution(False)" in source
    assert "np.asarray(target, dtype=np.float64) * 1000" in source

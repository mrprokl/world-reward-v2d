"""Numerical comparisons and real hook state-machine tests, never CUDA claims."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def repro(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_silhouette_repro", infra/"soft_silhouette_repro.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def old_report(repro):
    return dict(stage=repro.original.STAGE, status="fail", phase="translation_autograd", error="RuntimeError",
        message="RasterizeMeshesBackwardCuda does not have a deterministic implementation",
        producer_revision=repro.ORIGINAL_REVISION, image_id=repro.original.IMAGE, script_sha256=repro.ORIGINAL_SHA,
        camera_helper_sha256=repro.CAMERA_SHA, budget_seconds=180, challenge_inputs_used=False, ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[], network="none", actual_raster_calls=4, actual_backward_calls=0,
        actual_translation_steps=0, gradient_capability_verified=False, hard_mask_bit_identical=True,
        target_frozen_before_optimization=True, projection_max_error_px=3.23e-6,
        vertices_sha256="a"*64, synthetic_target_alpha_sha256="b"*64,
        camera_K_source=[[960.,0.,512.],[0.,1040.,384.],[0.,0.,1.]],
        camera_K_scaled=[[240.,0.,128.],[0.,260.,96.],[0.,0.,1.]],
        renderer_sources={"MeshRasterizer":{"file":"/source.py","sha256":"c"*64}},
        torch_version="2.8.0", cuda_version="12.8", pytorch3d_version="0.7.9")


def worker_reports(repro):
    old = old_report(repro); workers = []
    for index in (0, 1):
        w = copy.deepcopy(old)
        w.update(stage=repro.WORKER_STAGE, status="pass", phase="complete", worker_index=index, pid=100+index,
            producer_revision="d"*40, worker_script_sha256="e"*64, original_failed_report_sha256=repro.FAILED_SHA,
            settings=repro.SETTINGS.copy(), actual_raster_calls=5, actual_backward_calls=1, actual_translation_steps=1,
            scoped_backward_attempts=1, scoped_backward_completed=1, deterministic_flags_restored=True,
            original_autograd_backward_restored=True, physical_extent_retained=True, gradient_capability_verified=True,
            loss_decrease_verified=True, source_rehashed_after_run=True, strict_bit_determinism_verified=False,
            numerical_reproducibility_verified=False, accuracy_verified=False, adoption_authorized=False,
            width=256, height=192, source_width=1024, source_height=768, sigma=1e-4, gamma=1e-4,
            blur_radius=repro.original.BLUR_RADIUS, faces_per_pixel=8, translation_step_norm_m=.0005,
            translation_gradient=[.03,-.04,.02], loss_before=.02, loss_after=.019)
        w["actual_translation_step_m"] = repro.original.negative_gradient_step(np.asarray(w["translation_gradient"])).tolist()
        workers.append(w)
    return workers, old


def test_fixed_source_and_numerical_contracts_preserve_old_probe(repro):
    assert repro.sha256(Path(repro.original.__file__)) == repro.ORIGINAL_SHA
    assert repro.sha256(Path(repro.original.camera.__file__)) == repro.CAMERA_SHA
    assert repro.RTOL == 1e-5 and repro.ATOL == 1e-7 and repro.BUDGET == 180
    assert repro.ORIGINAL_REVISION == "35ce1a6c39fe8655a0a4ca46d406b7435d49f82e"
    assert repro.FAILED_SHA == "1ba7780830bcc96269c615e1b5dd03e20e1e0dddc0f4dc2a6ade5ed7cb6995a3"
    workers, old = worker_reports(repro)
    result = repro.validate_workers(workers, old, "d"*40, "e"*64)
    assert result["numerical_reproducibility_verified"] and not result["strict_bit_determinism_verified"]
    assert result["max_absolute_gradient_difference"] == 0 and result["absolute_loss_after_difference"] == 0


def test_symmetric_percomponent_tolerance_not_mean_or_averaging(repro):
    a = np.array([1., 2., 3.]); b = a+np.array([1e-6, -1e-6, 1e-6])
    assert repro.symmetric_close(a, b) == repro.symmetric_close(b, a) is True
    b[0] += 2e-5
    assert repro.symmetric_close(a, b) == repro.symmetric_close(b, a) is False
    workers, old = worker_reports(repro)
    workers[1]["translation_gradient"][0] += 1e-8
    workers[1]["actual_translation_step_m"] = repro.original.negative_gradient_step(np.asarray(workers[1]["translation_gradient"])).tolist()
    workers[1]["loss_after"] += 1e-8
    assert repro.validate_workers(workers, old, "d"*40, "e"*64)["numerical_reproducibility_verified"]


@pytest.mark.parametrize("fault", ["nan", "shape", "integer", "masked"])
def test_numerical_comparison_fail_closed(repro, fault):
    a, b = np.ones(3), np.ones(3)
    if fault == "nan": b[0] = np.nan
    elif fault == "shape": b = b[None]
    elif fault == "integer": b = b.astype(np.int64)
    else: b = np.ma.array(b, mask=False)
    with pytest.raises(ValueError): repro.symmetric_close(a, b)


@pytest.mark.parametrize("fault", ["count", "pid", "order", "source", "oldhash", "target", "vertices", "K", "version",
    "lossbeforebit", "gradient", "lossafter", "zerogradient", "step", "nodecrease", "scope", "restore", "hookrestore",
    "calls", "nobackward", "flags", "shape", "precision", "oracle", "adoption", "render"])
def test_workers_strict_independent_input_and_measurement_contract(repro, fault):
    workers, old = worker_reports(repro); w = workers[1]
    if fault == "count": workers.pop()
    elif fault == "pid": w["pid"] = workers[0]["pid"]
    elif fault == "order": workers.reverse()
    elif fault == "source": w["script_sha256"] = "f"*64
    elif fault == "oldhash": w["original_failed_report_sha256"] = "f"*64
    elif fault == "target": w["synthetic_target_alpha_sha256"] = "f"*64
    elif fault == "vertices": w["vertices_sha256"] = "f"*64
    elif fault == "K": w["camera_K_scaled"][0][0] += 1
    elif fault == "version": w["torch_version"] = "unknown"
    elif fault == "lossbeforebit": w["loss_before"] = float(np.nextafter(w["loss_before"], 1.))
    elif fault == "gradient":
        w["translation_gradient"][0] += 1e-3
        w["actual_translation_step_m"] = repro.original.negative_gradient_step(np.asarray(w["translation_gradient"])).tolist()
    elif fault == "lossafter": w["loss_after"] += 1e-3
    elif fault == "zerogradient": w["translation_gradient"][0] = 0.
    elif fault == "step": w["actual_translation_step_m"][0] *= -1
    elif fault == "nodecrease": w["loss_after"] = w["loss_before"]
    elif fault == "scope": w["scoped_backward_attempts"] = 2
    elif fault == "restore": w["deterministic_flags_restored"] = False
    elif fault == "hookrestore": w["original_autograd_backward_restored"] = False
    elif fault == "calls": w["actual_raster_calls"] = 4
    elif fault == "nobackward": w["actual_backward_calls"] = 0
    elif fault == "flags": w["settings"]["deterministic_algorithms"] = False
    elif fault == "shape": w["width"] = 512
    elif fault == "precision": w["settings"]["dtype"] = "float64"
    elif fault == "oracle": w["oracle_modes"] = ["GT"]
    elif fault == "adoption": w["adoption_authorized"] = True
    else: w["renderer_sources"]["MeshRasterizer"]["sha256"] = "f"*64
    with pytest.raises(ValueError): repro.validate_workers(workers, old, "d"*40, "e"*64)


def fake_torch(*, raise_backward=False, raise_sync_after=False):
    state = {"enabled": True, "warn": False, "calls": [], "sync": 0}
    def deterministic(enabled, warn_only=False):
        state["calls"].append(("flags", enabled, warn_only)); state.update(enabled=enabled, warn=warn_only)
    def synchronize():
        state["sync"] += 1; state["calls"].append(("sync", state["enabled"]))
        if raise_sync_after and state["sync"] >= 2: raise RuntimeError("cuda sync failure")
    def backward(*args, **kwargs):
        state["calls"].append(("backward", state["enabled"], state["warn"], args, kwargs))
        if raise_backward: raise RuntimeError("kernel failure")
        return "actual-backward-result"
    torch = SimpleNamespace(autograd=SimpleNamespace(backward=backward), cuda=SimpleNamespace(synchronize=synchronize),
        are_deterministic_algorithms_enabled=lambda: state["enabled"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn"], use_deterministic_algorithms=deterministic)
    return torch, state


def counters():
    return dict(scoped_backward_attempts=0, scoped_backward_completed=0,
                deterministic_flags_restored=False, original_autograd_backward_restored=False)


def test_actual_scoped_operator_hook_tracing_preserves_all_forward_flags(repro):
    torch, state = fake_torch(); old = torch.autograd.backward; report = counters()
    with repro.scoped_backward(torch, report):
        assert state["enabled"] is True and not state["warn"]  # Forward before backward remains strict.
        assert torch.autograd.backward("loss", retain_graph=False) == "actual-backward-result"
        assert state["enabled"] is True and not state["warn"]  # Subsequent forward remains strict.
        with pytest.raises(ValueError): torch.autograd.backward("second")
    assert torch.autograd.backward is old
    assert report == dict(scoped_backward_attempts=1, scoped_backward_completed=1,
        deterministic_flags_restored=True, original_autograd_backward_restored=True)
    assert state["calls"] == [("sync", True), ("flags", False, False),
        ("backward", False, False, ("loss",), {"retain_graph": False}), ("sync", False), ("sync", False), ("flags", True, False)]


@pytest.mark.parametrize("fault", ["kernel", "sync", "body"])
def test_hook_restores_original_flags_function_even_on_failure(repro, fault):
    torch, state = fake_torch(raise_backward=fault == "kernel", raise_sync_after=fault == "sync")
    old = torch.autograd.backward; report = counters()
    with pytest.raises(RuntimeError):
        with repro.scoped_backward(torch, report):
            if fault == "body": raise RuntimeError("outside backward")
            torch.autograd.backward("loss")
    assert torch.autograd.backward is old and state["enabled"] is True and state["warn"] is False
    assert report["original_autograd_backward_restored"] is True
    if fault != "body": assert report["deterministic_flags_restored"] is True and report["scoped_backward_completed"] == 0


@pytest.mark.parametrize("enabled,warn", [(False, False), (True, True)])
def test_backward_scope_refuses_preexisting_non_strict_state(repro, enabled, warn):
    torch, state = fake_torch(); state.update(enabled=enabled, warn=warn); old = torch.autograd.backward; report = counters()
    with repro.scoped_backward(torch, report):
        with pytest.raises(ValueError): torch.autograd.backward("loss")
    assert torch.autograd.backward is old and state["enabled"] == enabled and state["warn"] == warn
    assert report["scoped_backward_attempts"] == 0


@pytest.fixture
def failed(repro, monkeypatch, tmp_path):
    path = tmp_path/repro.original.BASE/"report.json"; path.parent.mkdir(parents=True)
    old = old_report(repro); path.write_text(json.dumps(old))
    monkeypatch.setattr(repro, "FAILED_SHA", repro.sha256(path))
    return tmp_path, path, old


def test_failed_receipt_is_bound_and_original_inputs_stay_unchanged(repro, failed):
    root, path, old = failed; before = path.read_bytes()
    actual, frozen = repro.failed_source(root)
    assert actual == old and len(frozen) == 3 and all(repro.sha256(p) == h for p, h in frozen)
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["bytes", "phase", "status", "error", "kernel", "deterministic", "source", "calls", "target", "projection"])
def test_legacy_failure_check_not_general_missing_network_waiver(repro, failed, monkeypatch, fault):
    root, path, old = failed
    if fault == "bytes": path.write_text('{}')
    else:
        if fault == "phase": old["phase"] = "complete"
        elif fault == "status": old["status"] = "pass"
        elif fault == "error": old["error"] = "OtherError"
        elif fault == "kernel": old["message"] = "Different deterministic kernel"
        elif fault == "deterministic": old["message"] = "RasterizeMeshesBackwardCuda otherfailure"
        elif fault == "source": old["script_sha256"] = "f"*64
        elif fault == "calls": old["actual_backward_calls"] = 1
        elif fault == "target": old["synthetic_target_alpha_sha256"] = "none"
        else: old["projection_max_error_px"] = .01
        path.write_text(json.dumps(old)); monkeypatch.setattr(repro, "FAILED_SHA", repro.sha256(path))
    with pytest.raises(ValueError): repro.failed_source(root)


def test_child_timeout_kills_entire_processgroup_and_has_no_retry(repro, monkeypatch):
    calls = []; process = SimpleNamespace(pid=444, alive=True)
    def wait(timeout):
        calls.append(("wait", timeout))
        if process.alive: raise subprocess.TimeoutExpired("child", timeout)
        return -9
    process.wait = wait; process.poll = lambda: None if process.alive else -9
    def popen(argv, **kwargs): calls.append(("Popen", argv, kwargs)); return process
    def killpg(pid, sig): calls.append(("killpg", pid, sig)); process.alive = False
    monkeypatch.setattr(repro.subprocess, "Popen", popen); monkeypatch.setattr(repro.os, "killpg", killpg)
    monkeypatch.setattr(repro.time, "monotonic", lambda: 10.)
    with pytest.raises(subprocess.TimeoutExpired): repro.launch_worker(0, 20.)
    assert len([x for x in calls if x[0] == "Popen"]) == 1
    assert calls[0][2]["start_new_session"] is True and calls[0][1][-2:] == ["--worker", "0"]
    assert any(x[0] == "killpg" and x[1] == 444 for x in calls)


def test_budget_expiry_stops_before_worker_launch(repro, monkeypatch):
    monkeypatch.setattr(repro.time, "monotonic", lambda: 20.)
    monkeypatch.setattr(repro.subprocess, "Popen", lambda *a, **k: pytest.fail("expired child must not launch"))
    with pytest.raises(TimeoutError): repro.launch_worker(0, 20.)


def test_wrapper_datafree_exclusive_and_source_closure_reuses_original_no_copy(repro):
    path = Path(repro.__file__).with_name("run_soft_silhouette_repro.sh"); subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "--network none" in text and "--memory 16g --cpus 4" in text and "183s" in text and repro.original.IMAGE in text
    mounts = [s for s in text.splitlines() if '--mount "' in s]
    assert len(mounts) == 3 and all("readonly" in s for s in mounts[:-1]) and "src=$OUT,dst=$OUT" in mounts[-1]
    assert "soft-silhouette-probe-v1/report.json" in mounts[1]
    assert all(f"src=$ROOT/{name}" not in text for name in ("weights", "data", "vendor", "validation", "outputs"))
    tree = ast.parse(Path(repro.__file__).read_text()); source = ast.unparse(tree)
    assert "original.run(report, persist)" in source and "subprocess.Popen" in source and "start_new_session=True" in source
    assert source.count("use_deterministic_algorithms(False") == 1
    assert "SoftSilhouetteShader" not in source and "rasterizer(mesh)" not in source  # Immutable original renderer reused.


def test_cli_only_private_worker_switch_no_parameters(repro):
    for argv in (["--step", "1"], ["--rtol", ".1"], ["--worker", "2"], ["--max-seconds", "999"]):
        with pytest.raises(SystemExit) as exc: repro.main(argv)
        assert exc.value.code == 2

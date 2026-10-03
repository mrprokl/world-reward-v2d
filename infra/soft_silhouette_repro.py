"""Two fresh-process numerical replay of the immutable D91 capability.

D91 remains failed: PyTorch3D CUDA raster backward has no deterministic-mode
implementation. Only the original backward call temporarily disables that
flag, synchronizes CUDA and restores both flags in finally. All forwards and
the original fixed0.5mm step/gates stay unchanged. Two numerical replays are
not bit determinism, calibrated accuracy, or adoption of a reconstruction.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

import numpy as np

import soft_silhouette_probe as original
from world_reward.data import sha256

STAGE = "own_synthetic_soft_silhouette_two_process_numerical_replay"
WORKER_STAGE = STAGE+"_worker"
BASE = "results/soft-silhouette-repro-v1"
BUDGET = 180
FAILED_SHA = "1ba7780830bcc96269c615e1b5dd03e20e1e0dddc0f4dc2a6ade5ed7cb6995a3"
ORIGINAL_SHA = "cb7d4bc9fd3d7b86239c467c4e69ddf549f47cbfab848bae77e8944f635702b4"
CAMERA_SHA = "325fa1c42b6e620f93fe59d6e4f63852557a08343b13e2bea60eee8c88f4743c"
ORIGINAL_REVISION = "35ce1a6c39fe8655a0a4ca46d406b7435d49f82e"
RTOL, ATOL = 1e-5, 1e-7
SETTINGS = {"seed": 0, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False,
            "deterministic_algorithms": True, "CPU_threads": 4, "dtype": "float32"}


def require_fields(report, expected):
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Explicit frozen source/protocol fields differ")


def failed_source(root):
    path = original.regular(root/original.BASE/"report.json")
    if sha256(path) != FAILED_SHA: raise ValueError("Immutable original D91 failure SHA differs")
    old = json.loads(path.read_text())
    require_fields(old, {"stage": original.STAGE, "status": "fail", "phase": "translation_autograd",
        "error": "RuntimeError", "producer_revision": ORIGINAL_REVISION, "image_id": original.IMAGE,
        "script_sha256": ORIGINAL_SHA, "camera_helper_sha256": CAMERA_SHA, "budget_seconds": BUDGET,
        "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "network": "none", "actual_raster_calls": 4, "actual_backward_calls": 0, "actual_translation_steps": 0,
        "gradient_capability_verified": False, "hard_mask_bit_identical": True, "target_frozen_before_optimization": True})
    if (type(old.get("projection_max_error_px")) is not float or not np.isfinite(old["projection_max_error_px"])
            or not 0 <= old["projection_max_error_px"] <= original.PROJECTION_ATOL_PX
            or not re.fullmatch("[0-9a-f]{64}", str(old.get("synthetic_target_alpha_sha256", "")))
            or "RasterizeMeshesBackwardCuda" not in old.get("message", "")
            or "deterministic" not in old.get("message", "")):
        raise ValueError("Original failure must retain successful projection/target provenance")
    paths = [(path, FAILED_SHA), (original.regular(Path(original.__file__)), ORIGINAL_SHA),
             (original.regular(Path(original.camera.__file__)), CAMERA_SHA)]
    for source, wanted in paths:
        if sha256(source) != wanted: raise ValueError("Immutable D91/camera source differs")
    return old, paths


@contextmanager
def scoped_backward(torch, report):
    """Tensor.backward calls this hook; ONLY the one native backward is relaxed."""
    previous = torch.autograd.backward

    def backward(*args, **kwargs):
        if report["scoped_backward_attempts"] != 0: raise ValueError("Only one original backward may be invoked")
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        if not enabled or warn_only: raise ValueError("All forwards must retain strict deterministic flags")
        torch.cuda.synchronize()
        report["scoped_backward_attempts"] += 1
        try:
            torch.use_deterministic_algorithms(False, warn_only=False)
            result = previous(*args, **kwargs)
            torch.cuda.synchronize()
            report["scoped_backward_completed"] += 1
            return result
        finally:
            try:
                torch.cuda.synchronize()
            finally:
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                report["deterministic_flags_restored"] = (
                    torch.are_deterministic_algorithms_enabled() == enabled
                    and torch.is_deterministic_algorithms_warn_only_enabled() == warn_only)

    torch.autograd.backward = backward
    try:
        yield
    finally:
        torch.autograd.backward = previous
        report["original_autograd_backward_restored"] = torch.autograd.backward is previous


def symmetric_close(first, second, *, rtol=RTOL, atol=ATOL):
    if np.ma.isMaskedArray(first) or np.ma.isMaskedArray(second): raise ValueError("No hidden numerical samples")
    a, b = np.asarray(first), np.asarray(second)
    if (a.shape != b.shape or a.dtype.kind not in "f" or b.dtype.kind not in "f"
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Same-shape finite floating numerical evidence required")
    difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
    bound = atol+rtol*np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    return bool(np.all(difference <= bound))


def validate_workers(workers, old, revision, script_sha):
    if not isinstance(workers, list) or len(workers) != 2: raise ValueError("Exactly two fresh workers required")
    identity_fields = ("vertices_sha256", "synthetic_target_alpha_sha256", "camera_K_source", "camera_K_scaled", "renderer_sources")
    for index, report in enumerate(workers):
        require_fields(report, {"stage": WORKER_STAGE, "status": "pass", "phase": "complete", "worker_index": index,
            "producer_revision": revision, "image_id": original.IMAGE, "worker_script_sha256": script_sha,
            "script_sha256": ORIGINAL_SHA, "camera_helper_sha256": CAMERA_SHA, "original_failed_report_sha256": FAILED_SHA,
            "network": "none", "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False,
            "oracle_modes": [], "settings": SETTINGS, "actual_raster_calls": 5, "actual_backward_calls": 1,
            "actual_translation_steps": 1, "scoped_backward_attempts": 1, "scoped_backward_completed": 1,
            "deterministic_flags_restored": True, "original_autograd_backward_restored": True,
            "hard_mask_bit_identical": True, "target_frozen_before_optimization": True, "physical_extent_retained": True,
            "gradient_capability_verified": True, "loss_decrease_verified": True, "source_rehashed_after_run": True,
            "strict_bit_determinism_verified": False, "numerical_reproducibility_verified": False,
            "accuracy_verified": False, "adoption_authorized": False,
            "width": original.WIDTH, "height": original.HEIGHT, "source_width": original.SOURCE_WIDTH,
            "source_height": original.SOURCE_HEIGHT, "sigma": original.SIGMA, "gamma": original.GAMMA,
            "blur_radius": original.BLUR_RADIUS, "faces_per_pixel": original.FACES_PER_PIXEL,
            "translation_step_norm_m": original.STEP_M})
        if (type(report.get("pid")) is not int or report["pid"] <= 0
                or type(report.get("projection_max_error_px")) is not float
                or not np.isfinite(report["projection_max_error_px"])
                or not 0 <= report["projection_max_error_px"] <= original.PROJECTION_ATOL_PX):
            raise ValueError("Actual fresh process and original projection bound required")
        for name in identity_fields:
            if name not in report or report[name] != old.get(name): raise ValueError("Frozen D91 source/input/target identity differs")
        gradient = np.asarray(report.get("translation_gradient"), np.float64)
        step = np.asarray(report.get("actual_translation_step_m"), np.float64)
        original.validate_measurement(report.get("loss_before"), report.get("loss_after"), gradient, step)
    a, b = workers
    if a["pid"] == b["pid"]: raise ValueError("Independent fresh worker PIDs required")
    for key in ("torch_version", "cuda_version", "pytorch3d_version"):
        if not isinstance(a.get(key), str) or not a[key] or a[key] != b.get(key) or a[key] != old.get(key):
            raise ValueError("Exact pinned runtime versions required")
    if np.float64(a["loss_before"]).tobytes() != np.float64(b["loss_before"]).tobytes():
        raise ValueError("Forward loss-before must be bit-identical between fresh workers")
    gradients = [np.asarray(w["translation_gradient"], np.float64) for w in workers]
    if not symmetric_close(*gradients) or not symmetric_close(np.asarray(a["loss_after"]), np.asarray(b["loss_after"])):
        raise ValueError("Fresh gradient/loss replay exceeds predeclared numerical tolerance")
    return {"numerical_reproducibility_verified": True, "strict_bit_determinism_verified": False,
        "forward_target_and_loss_before_bit_identical": True,
        "max_absolute_gradient_difference": float(np.abs(gradients[0]-gradients[1]).max()),
        "absolute_loss_after_difference": float(abs(a["loss_after"]-b["loss_after"]))}


def launch_worker(index, deadline):
    remaining = deadline-time.monotonic()
    if type(index) is not int or index not in (0, 1) or remaining <= 0: raise TimeoutError("Total180s exhausted before fresh worker")
    environment = dict(os.environ, WR_SOFT_REPRO_DEADLINE=str(deadline))
    process = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker", str(index)],
        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
        if code != 0: raise RuntimeError("Fresh numerical worker failed; inspect its immutable report")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def initialize_worker(report, index):
    report.update(stage=WORKER_STAGE, worker_index=index, pid=os.getpid(), script_sha256=ORIGINAL_SHA,
        worker_script_sha256=sha256(Path(__file__)), camera_helper_sha256=CAMERA_SHA,
        original_failed_report_sha256=FAILED_SHA, settings=SETTINGS.copy(), actual_raster_calls=0,
        actual_backward_calls=0, actual_translation_steps=0, scoped_backward_attempts=0, scoped_backward_completed=0,
        deterministic_flags_restored=False, original_autograd_backward_restored=False,
        strict_bit_determinism_verified=False, numerical_reproducibility_verified=False,
        width=original.WIDTH, height=original.HEIGHT, source_width=original.SOURCE_WIDTH, source_height=original.SOURCE_HEIGHT,
        sigma=original.SIGMA, gamma=original.GAMMA, blur_radius=original.BLUR_RADIUS,
        faces_per_pixel=original.FACES_PER_PIXEL, translation_step_norm_m=original.STEP_M)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); revision = os.environ["WR_CODE_REVISION"]; image = os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != original.IMAGE):
        raise ValueError("Canonical Azure offline root/revision/pinned image required")
    out = root/BASE
    if args.worker is not None: out = out/f"worker_{args.worker}"
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty new replay output required")
    start = time.monotonic(); deadline = start+BUDGET
    if args.worker is not None:
        inherited = float(os.environ["WR_SOFT_REPRO_DEADLINE"])
        if not np.isfinite(inherited) or not start < inherited <= deadline: raise ValueError("Parent total deadline required")
        deadline = inherited
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "original_probe_script_sha256": ORIGINAL_SHA, "camera_helper_sha256": CAMERA_SHA,
        "original_failed_report_sha256": FAILED_SHA, "budget_seconds_total": BUDGET, "network": "none",
        "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "accuracy_verified": False, "adoption_authorized": False, "submission_produced": False,
        "strict_bit_determinism_verified": False, "numerical_reproducibility_verified": False,
        "nondeterministic_operation_scope": "only_original_torch_autograd_backward_cuda_raster_atomic_accumulation",
        "kernel_causal_identification_verified": False, "original_D91_failure_waived": False,
        "gradient_rtol": RTOL, "gradient_atol": ATOL, "loss_after_rtol": RTOL, "loss_after_atol": ATOL,
        "forward_loss_before_comparison": "bit_identity", "workers_expected": 2, "workers": []}
    with (out/"report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.monotonic()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Two-process numerical replay exceeded total180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(max(1, math.ceil(deadline-time.monotonic())))
        try:
            persist(); old, frozen = failed_source(root)
            if args.worker is not None:
                initialize_worker(report, args.worker); persist()
                torch = original.strict_torch()
                with scoped_backward(torch, report): original.run(report, persist)
                for path, digest in frozen:
                    if sha256(original.regular(path)) != digest: raise ValueError("Original source/failure changed during worker")
                if (report["scoped_backward_completed"] != 1 or not report["deterministic_flags_restored"]
                        or not report["original_autograd_backward_restored"]): raise ValueError("Backward scope restoration failed")
            else:
                script_sha = sha256(Path(__file__)); report.update(phase="fresh_workers"); persist()
                worker_reports = []
                for index in (0, 1):
                    destination = out/f"worker_{index}"; destination.mkdir()
                    launch_worker(index, deadline)
                    path = original.regular(destination/"report.json"); digest = sha256(path)
                    if path.stat().st_mode & 0o222: raise ValueError("Worker reports must be frozen before parent comparison")
                    result = json.loads(path.read_text()); worker_reports.append(result); frozen.append((path, digest))
                    report["workers"].append({"worker_index": index, "file": str(path.relative_to(out)),
                        "sha256": digest, "bytes": path.stat().st_size, "pid": result.get("pid"), "status": result.get("status")}); persist()
                report.update(phase="paired_numerical_comparison"); persist()
                comparison = validate_workers(worker_reports, old, revision, script_sha)
                for path, digest in frozen:
                    if sha256(original.regular(path)) != digest: raise ValueError("Frozen source/worker report changed")
                report.update(**comparison, status="pass", phase="complete", original_failure_preserved=True,
                    actual_backward_calls=2, actual_translation_steps=2, source_rehashed_after_run=True)
        except BaseException as error:
            report.update(status="fail", error=type(error).__name__, message=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
            persist(); (out/"report.json").chmod(0o444)
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds")}))


if __name__ == "__main__": main()

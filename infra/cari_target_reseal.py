"""Reseal predicted native vertices after a cross-process hash failure.

No conversion or solver is rerun. Two fresh strict native decodes must agree
in raw bits; unchanged historical parameters are then replayed once through
the pinned official FP32 reference. The historical target array was not saved
and is never claimed recovered. All large arrays remain on Azure.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import secrets
import signal
import subprocess
import sys
import time

import numpy as np

import cari_converter as convert
from cari_identity_probe_runtime import SEALED_SHA, sealed_inputs
from world_reward.data import sha256

STAGE = "world_reward_cari_native_target_strict_reseal"
FAILED_SHA = "05a558bc7dee69553d6b34ddc83ab9ff0f3361b7c648d99a75d9b942836602c3"
BUDGET = 300
TARGET_SHAPE = (790, 18439, 3)
SETTINGS = {"seed": 0, "dtype": "float32", "native_batch_size": 16, "warmup_calls": 0,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
    "TF32": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
    "jit_optimized_execution": False, "CPU_threads": 4}


def regular(path):
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Canonical regular file required")
    return path


def failed_source(root):
    path = root / "outputs/episode_000000/cari_identity_probe_refined_v1/report.json"
    convert._require_hash(regular(path), FAILED_SHA)
    report = json.loads(path.read_text())
    expected = {"stage": "world_reward_cari_shared_identity_runtime_probe", "status": "fail",
        "phase": "native_target_replay", "error": "Original frozen target vertices changed",
        "full_frame_fidelity_verified": False, "source_bindings_runtime_verified": False,
        "original_conversion_rerun": False, "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "input_track": "track_1", "network": "none"}
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Exact immutable failed target-replay receipt required")
    return path


def strict_torch():
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Fresh Torch import and deterministic CUBLAS environment required")
    random.seed(0); np.random.seed(0)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("Azure CUDA required; no local/CPU fallback")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return torch


def source_chain(root, params, receipt):
    from body_smoke import _body_assets, _pinned_checkout, _source_identity
    vendor = root / "vendor/video_to_data"; _pinned_checkout(vendor, convert.UPSTREAM_REVISION)
    if _source_identity(root) != receipt["inference_source_identity"]:
        raise ValueError("Original native decoder source changed")
    assets, hashes = _body_assets(root)
    if hashes != receipt["body_assets"] or (assets / "mhr_buffers.pt").exists():
        raise ValueError("Original native Body decoder assets changed")
    refpath = root / "outputs/episode_000000/cari_refined/report.json"
    convert._require_hash(regular(refpath), receipt["input_report_sha256"]["refinement"])
    refinement = json.loads(refpath.read_text())
    bundle = root / "outputs/episode_000000/cari_refined/refined.pth"
    convert._require_hash(regular(bundle), refinement["bundle_sha256"])
    if refinement["bundle_sha256"] != receipt["provenance"]["native_bundle_sha256"]:
        raise ValueError("Original refined bundle lineage differs")
    return assets, vendor, [(refpath, receipt["input_report_sha256"]["refinement"]),
                            (bundle, refinement["bundle_sha256"])]


def worker_authorized(parent, index, *, nonce, revision, image):
    if (parent.get("stage") != STAGE or parent.get("status") != "running"
            or type(parent.get("pending_worker")) is not int or parent["pending_worker"] != index
            or parent.get("script_sha256") != sha256(Path(__file__))
            or parent.get("producer_revision") != revision or parent.get("image_id") != image
            or not nonce or parent.get("nonce_sha256") != hashlib.sha256(nonce.encode()).hexdigest()):
        raise RuntimeError("Only the authorized parent may start a fresh decode worker")


def decode_worker(root, out, index, revision, image):
    parent = json.loads((out / "report.json").read_text())
    worker_authorized(parent, index, nonce=os.environ.get("WR_RESEAL_NONCE", ""), revision=revision, image=image)
    report = {"stage": STAGE, "status": "fail", "worker_index": index, "worker_pid": os.getpid(),
        "script_sha256": sha256(Path(__file__)), "producer_revision": revision, "image_id": image,
        "settings": dict(SETTINGS), "phase": "provenance", "native_full_frame_decode_verified": False}
    with (out / f"worker_{index}.json").open("x") as handle:
        def persist():
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        try:
            persist(); _, params, receipt, frozen = sealed_inputs(root); failpath = failed_source(root)
            assets, vendor, more = source_chain(root, params, receipt); frozen += more + [(failpath, FAILED_SHA)]
            torch = strict_torch()
            actual, _ = convert.validate_native_bundle(torch.load(more[1][0], map_location="cpu", weights_only=False), 790)
            if any(not np.array_equal(actual[k], params[k]) for k in params):
                raise ValueError("Original bundle and sealed native parameters differ")
            del actual
            native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
            os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"),
                              MOMENTUM_ENABLED="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
            from lib_mhr.mhr_layer import MHRLayer
            from lib_mhr.geometry_provider import decode_mhr_vertices_numpy
            for symbol in (MHRLayer, decode_mhr_vertices_numpy):
                if not Path(sys.modules[symbol.__module__].__file__).resolve().is_relative_to(native / "lib_mhr"):
                    raise ValueError("Original native decoder import path differs")
            report.update(phase="strict_native_decode", torch=torch.__version__, CUDA=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), GPU=torch.cuda.get_device_name(0),
                sealed_report_sha256=SEALED_SHA, failed_probe_report_sha256=FAILED_SHA,
                native_parameter_identities=receipt["native_parameter_identities"],
                inference_source_identity=receipt["inference_source_identity"], body_assets=receipt["body_assets"]); persist()
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),
                    checkpoint_path=assets / "model.ckpt", buffer_path=out / "never_use_compact_buffer.pt",
                    mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
                if layer.decoder_identity() != receipt["decoder_identity"]:
                    raise ValueError("Original native decoder identity differs")
                target = decode_mhr_vertices_numpy(layer, params, batch_size=16); torch.cuda.synchronize()
            convert._float_array(target, "strict native target", TARGET_SHAPE)
            if target.dtype != np.float32: raise ValueError("Native target must retain original FP32 values")
            path = out / ("firsttarget_f32.npy" if index == 0 else "secondtarget_f32.npy")
            with path.open("xb") as stream: np.save(stream, target, allow_pickle=False)
            path.chmod(0o444)
            report.update(target_vertices_identity=convert.canonical_array_identity(target), target_file=path.name,
                          target_file_sha256=sha256(path), target_file_bytes=path.stat().st_size)
            for p, digest in frozen: convert._require_hash(regular(p), digest)
            source_chain(root, params, receipt)
            report.update(status="pass", phase="complete", native_full_frame_decode_verified=True); persist()
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); persist(); raise


def compare_targets(first, second):
    if (first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3 or not len(first)
            or first.dtype != np.float32 or second.dtype != np.float32):
        raise ValueError("Paired finite FP32 camera-vertex arrays required")
    counts, means, maxima = [], [], []
    for a, b in zip(first, second):
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError("Native target contains nonfinite values")
        counts.append(int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))))
        distances = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1) * 1000
        means.append(float(distances.mean())); maxima.append(float(distances.max()))
    return {"bitexact": not any(counts), "changed_elements": sum(counts), "per_frame_changed_elements": counts,
            "mean_point_L2_mm": float(np.mean(means)), "max_point_L2_mm": max(maxima),
            "per_frame_mean_point_L2_mm": means, "per_frame_max_point_L2_mm": maxima}


def validate_workers(workers):
    if len(workers) != 2 or len({w.get("worker_pid") for w in workers}) != 2:
        raise ValueError("Two distinct fresh Python workers required")
    for index, worker in enumerate(workers):
        expected = {"stage": STAGE, "status": "pass", "phase": "complete", "worker_index": index,
                    "settings": SETTINGS, "native_full_frame_decode_verified": True}
        if (any(type(worker.get(k)) is not type(v) or worker[k] != v for k, v in expected.items())
                or type(worker.get("worker_pid")) is not int or worker["worker_pid"] == os.getpid()):
            raise ValueError("Every fresh worker must complete the exact strict protocol")
    for field in ("script_sha256", "producer_revision", "image_id", "torch", "CUDA", "cudnn", "GPU",
                  "sealed_report_sha256", "failed_probe_report_sha256", "native_parameter_identities",
                  "inference_source_identity", "body_assets", "target_vertices_identity"):
        if field not in workers[0] or workers[0][field] != workers[1].get(field):
            raise ValueError("Fresh worker source/version/settings/target identity differs")


def historical_agreement(original, replayed):
    if (original.shape != replayed.shape or original.ndim != 1 or not len(original)
            or original.dtype != np.float32 or replayed.dtype != np.float32
            or not np.isfinite(original).all() or not np.isfinite(replayed).all()
            or np.any(original < 0) or np.any(replayed < 0)):
        raise ValueError("Finite nonnegative official FP32 error vectors required")
    valid = np.isclose(replayed, original, rtol=1e-5, atol=1e-4)
    delta = replayed.astype(np.float64) - original.astype(np.float64)
    return {"historical_error_agreement": bool(valid.all()), "rtol": 1e-5, "atol_mm": 1e-4,
            "drift_failure_frames": np.flatnonzero(~valid).tolist(), "max_absolute_drift_mm": float(np.max(np.abs(delta))),
            "historical_mean_mm": float(original.astype(np.float64).mean()),
            "replay_mean_mm": float(replayed.astype(np.float64).mean())}


def replay_history(root, original, target, report):
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    model = root / "weights/mhr/mhr_model.pt"
    convert._require_hash(regular(tool), convert.CONVERTER_SHA256)
    convert._require_hash(regular(model), convert.REFERENCE_MODEL_SHA256)
    torch = strict_torch()
    report.update(reference_settings=dict(SETTINGS), reference_torch=torch.__version__, reference_CUDA=torch.version.cuda)
    spec = importlib.util.spec_from_file_location("world_reward_reseal_official_reference", tool)
    converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
    if Path(converter.MHR.run.__code__.co_filename).resolve() != tool.resolve():
        raise ValueError("Official reference function resolved outside pinned source")
    # Pinned convert(): cast verts_m to float64 before x1000 (lines298,306);
    # run() returns float64 mm, norm/mean are float64, per_frame storage is FP32.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        reference = converter.MHR(str(model), "cuda", chunk=256, precision="float32")
        p = torch.tensor(original["pose"], dtype=torch.float64, device="cuda")
        z = torch.tensor(np.concatenate([original["scales"], original["shape"]])[None], dtype=torch.float64, device="cuda")
        tgt = torch.tensor(np.asarray(target, dtype=np.float64) * 1000, dtype=torch.float64, device="cuda").reshape(790, -1)
        errors64 = (reference.verts(p, z) - tgt).reshape(790, -1, 3).norm(dim=-1).mean(1).cpu().numpy().copy()
    errors32 = np.asarray(errors64, dtype=np.float32)
    report.update(actual_official_reference_replay_calls=1, reference_precision="float32",
        reference_model_chunk=256, reference_residual_dtype="float64", reference_error_storage_dtype="float32",
        target_mm_arithmetic="np.asarray(target,dtype=float64)*1000;official_convert_source_lines298_306",
        official_converter_sha256=convert.CONVERTER_SHA256, reference_model_sha256=convert.REFERENCE_MODEL_SHA256)
    convert._require_hash(tool, convert.CONVERTER_SHA256); convert._require_hash(model, convert.REFERENCE_MODEL_SHA256)
    return errors32, errors64


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline Azure CUDA container required")
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image)):
        raise ValueError("Canonical remote root and immutable producing source/image required")
    out = root / "outputs/episode_000000/cari_target_reseal_refined_v1"
    if out.resolve() != out.absolute() or not out.is_dir(): raise FileExistsError("Exclusive reserved output required")
    if args.worker is not None: return decode_worker(root, out, args.worker, revision, image)
    if any(out.iterdir()): raise FileExistsError("Reseal output is frozen or incomplete")
    start = time.perf_counter(); nonce = secrets.token_hex(32)
    report = {"stage": STAGE, "status": "running", "phase": "provenance", "script_sha256": sha256(Path(__file__)),
        "producer_revision": revision, "image_id": image, "budget_seconds": BUDGET, "network": "none",
        "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "old_target_bitexact_recovered": False, "old_target_array_available": False, "original_conversion_rerun": False,
        "solver_calls": 0, "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(), "workers": [],
        "second_target_removed": False, "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA}
    with (out / "report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Complete native target reseal exceeded300s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); original, params, receipt, frozen = sealed_inputs(root)
            frozen.append((failed_source(root), FAILED_SHA)); _, _, more = source_chain(root, params, receipt); frozen += more
            report.update(old_native_vertices_identity=receipt["native_vertices_identity"], phase="fresh_workers"); persist()
            for index in range(2):
                remaining = BUDGET - (time.perf_counter() - start)
                if remaining <= 0: raise TimeoutError("Reseal budget exhausted before worker")
                report["pending_worker"] = index; persist()
                subprocess.run([sys.executable, str(Path(__file__)), "--worker", str(index)], check=True, timeout=remaining,
                    env=os.environ | {"WR_RESEAL_NONCE": nonce, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
                workerpath = regular(out / f"worker_{index}.json")
                report["workers"].append(json.loads(workerpath.read_text()))
                report.setdefault("worker_report_sha256", []).append(sha256(workerpath)); persist()
            firstpath, secondpath = regular(out / "firsttarget_f32.npy"), regular(out / "secondtarget_f32.npy")
            for path, worker in zip((firstpath, secondpath), report["workers"]):
                if path.stat().st_mode & 0o222: raise ValueError("Saved native target must be immutable")
                convert._require_hash(path, worker["target_file_sha256"])
            first, second = np.load(firstpath, mmap_mode="r", allow_pickle=False), np.load(secondpath, mmap_mode="r", allow_pickle=False)
            if first.shape != TARGET_SHAPE: raise ValueError("Full790 target required")
            report.update(phase="paired_target_comparison", comparison=compare_targets(first, second)); persist()
            validate_workers(report["workers"])
            if not report["comparison"]["bitexact"]: raise ValueError("Fresh strict full native target decodes differ in raw bits")
            del second; secondpath.unlink(); report["second_target_removed"] = True; persist()
            expected = report["workers"][0]["target_vertices_identity"]
            if convert.canonical_array_identity(first) != expected: raise ValueError("Saved first target array identity changed")
            report.update(phase="unchanged_historical_reference_replay", target_vertices_identity=expected,
                target_file="firsttarget_f32.npy", target_file_sha256=sha256(firstpath),
                historical_provenance=receipt["provenance"],
                resealed_provenance={**receipt["provenance"], "native_vertices_sha256": expected["sha256"],
                    "historical_native_vertices_sha256": receipt["native_vertices_identity"]["sha256"]}); persist()
            replayed, precise = replay_history(root, original, first, report)
            report.update(historical_agreement(original["per_frame_vertex_error_mm"], replayed))
            errorspath = out / "replay_errors.npz"
            with errorspath.open("xb") as stream:
                np.savez_compressed(stream, frame_index=np.arange(790, dtype=np.int64),
                    historical_errors_f32=original["per_frame_vertex_error_mm"], replay_errors_f32=replayed,
                    replay_errors_float64=precise, drift_mm=replayed.astype(np.float64)-original["per_frame_vertex_error_mm"])
            errorspath.chmod(0o444); report["replay_errors_sha256"] = sha256(errorspath); persist()
            convert._require_hash(firstpath, report["target_file_sha256"])
            if convert.canonical_array_identity(first) != expected: raise ValueError("First target changed during reference replay")
            for path, digest in frozen: convert._require_hash(regular(path), digest)
            source_chain(root, params, receipt)
            for index, digest in enumerate(report["worker_report_sha256"]):
                convert._require_hash(out / f"worker_{index}.json", digest)
            report["file_inventory"] = [{"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in (firstpath, out / "worker_0.json", out / "worker_1.json", errorspath)]
            if not report["historical_error_agreement"]: raise ValueError("Replayed historical errors disagree beyond unchanged tolerance")
            report.update(status="pass", phase="complete", fresh_worker_bitexact=True,
                original_parameters_unchanged=True, source_bindings_runtime_verified=True); persist()
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "historical_error_agreement")}))


if __name__ == "__main__": main()

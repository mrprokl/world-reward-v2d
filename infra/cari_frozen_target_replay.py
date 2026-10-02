"""Replay historical controls against the first saved, newly frozen target.

The strict native reseal remains failed. Worker zero was selected by protocol
order, never residual/accuracy. No native decode, solver, conversion, identity
fit, or submission occurs here; PASS means numerical warm-start agreement only.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

import cari_converter as convert
from cari_target_reseal import (FAILED_SHA, SEALED_SHA, SETTINGS, TARGET_SHAPE, STAGE as RESEAL_STAGE,
    failed_source, historical_agreement, regular, replay_history, sealed_inputs, source_chain)
from world_reward.data import sha256

STAGE = "world_reward_cari_frozen_first_target_historical_replay"
RESEAL_SHA = "314f80e8bbb19e9883c75ba262c13417cd34754c0eba364516f1060073c9e825"
RESEAL_REVISION = "e2e446a35b2a6258ec52074cd9783964046b179c"
FIRST_IDENTITY_SHA = "df8cfc73aeea41e114825205bc822003d07b471ce976ee458ef12ed229f12397"
BUDGET = 180


def require_fields(report, expected, description):
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError(description)


def frozen_target(root, original_receipt):
    directory = root / "outputs/episode_000000/cari_target_reseal_refined_v1"
    path = regular(directory / "report.json"); convert._require_hash(path, RESEAL_SHA)
    receipt = json.loads(path.read_text())
    require_fields(receipt, {"stage": RESEAL_STAGE, "status": "fail", "phase": "paired_target_comparison",
        "error": "Fresh worker source/version/settings/target identity differs", "producer_revision": RESEAL_REVISION,
        "network": "none", "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
        "oracle_modes": [], "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "old_target_bitexact_recovered": False, "old_target_array_available": False,
        "original_conversion_rerun": False, "solver_calls": 0, "second_target_removed": False,
        "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA},
        "Exact failed strict reseal lineage required")
    workers, hashes = receipt.get("workers"), receipt.get("worker_report_sha256")
    if (not isinstance(workers, list) or len(workers) != 2 or not isinstance(hashes, list) or len(hashes) != 2
            or any(type(w.get("worker_pid")) is not int for w in workers)
            or workers[0]["worker_pid"] == workers[1]["worker_pid"]):
        raise ValueError("Both distinct completed fresh decode worker receipts required")
    frozen = [(path, RESEAL_SHA)]
    for index, worker in enumerate(workers):
        workerpath = regular(directory / f"worker_{index}.json"); convert._require_hash(workerpath, hashes[index])
        if json.loads(workerpath.read_text()) != worker: raise ValueError("Embedded and saved worker receipts differ")
        require_fields(worker, {"stage": RESEAL_STAGE, "status": "pass", "phase": "complete", "worker_index": index,
            "settings": SETTINGS, "native_full_frame_decode_verified": True, "producer_revision": RESEAL_REVISION,
            "script_sha256": receipt["script_sha256"], "image_id": receipt["image_id"],
            "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA,
            "native_parameter_identities": original_receipt["native_parameter_identities"],
            "inference_source_identity": original_receipt["inference_source_identity"], "body_assets": original_receipt["body_assets"],
            "target_file": "firsttarget_f32.npy" if index == 0 else "secondtarget_f32.npy"},
            "Exact complete worker source/parameters/assets/settings required")
        identity = worker.get("target_vertices_identity", {})
        require_fields(identity, {"dtype": "<f4", "shape": list(TARGET_SHAPE), "bytes": 790*18439*3*4},
                       "Complete canonical FP32 target identity required")
        frozen.append((workerpath, hashes[index]))
    for field in ("torch", "CUDA", "cudnn", "GPU"):
        if field not in workers[0] or workers[0][field] != workers[1].get(field):
            raise ValueError("Worker native runtime versions/device differ")
    # This exact historical failure permits target SHA disagreement ONLY;
    # it does not relax the strict reseal validator or claim reproducibility.
    if (workers[0]["target_vertices_identity"]["sha256"] != FIRST_IDENTITY_SHA
            or workers[0]["target_vertices_identity"] == workers[1]["target_vertices_identity"]
            or receipt.get("comparison", {}).get("bitexact") is not False):
        raise ValueError("Declared first-worker target and actual bitexact failure required")
    targetpath = regular(directory / "firsttarget_f32.npy")
    if targetpath.stat().st_mode & 0o222: raise ValueError("First target must remain read-only")
    convert._require_hash(targetpath, workers[0]["target_file_sha256"])
    if targetpath.stat().st_size != workers[0]["target_file_bytes"]: raise ValueError("First target file size differs")
    target = np.load(targetpath, mmap_mode="r", allow_pickle=False)
    if target.shape != TARGET_SHAPE or target.dtype != np.float32 or target.flags.writeable:
        raise ValueError("Immutable full790 FP32 first target mmap required")
    if convert.canonical_array_identity(target) != workers[0]["target_vertices_identity"]:
        raise ValueError("Saved first target canonical SHA differs")
    frozen.append((targetpath, workers[0]["target_file_sha256"]))
    return target, receipt, frozen


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline Azure CUDA container required")
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image)):
        raise ValueError("Canonical root and immutable source/image required")
    out = root / "outputs/episode_000000/cari_frozen_target_replay_refined_v1"
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty diagnostic directory required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "script_sha256": sha256(Path(__file__)),
        "producer_revision": revision, "image_id": image, "budget_seconds": BUDGET, "network": "none",
        "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "strict_reproducibility_verified": False, "historical_target_recovered": False,
        "new_inverse_target": True, "selection_rule": "first_worker_protocol_order_not_error",
        "full_frame_fit_verified": False, "actual_native_decode_calls": 0, "solver_calls": 0,
        "original_conversion_rerun": False, "source_bindings_runtime_verified": False,
        "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA, "failed_reseal_report_sha256": RESEAL_SHA}
    with (out / "report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Frozen target historical replay exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); original, params, original_receipt, frozen = sealed_inputs(root)
            frozen.append((failed_source(root), FAILED_SHA))
            _, _, more = source_chain(root, params, original_receipt); frozen += more
            target, receipt, more = frozen_target(root, original_receipt); frozen += more
            if image != receipt["image_id"]:
                raise ValueError("Historical reference replay must use the unchanged target-producing image")
            original_ids = {k: convert.canonical_array_identity(original[k])
                for k in ("pose", "scales", "shape", "per_frame_vertex_error_mm")}
            report.update(phase="unchanged_official_reference_replay", original_parameter_identities=original_ids,
                historical_native_vertices_identity=original_receipt["native_vertices_identity"],
                target_vertices_identity=receipt["workers"][0]["target_vertices_identity"],
                target_file_sha256=receipt["workers"][0]["target_file_sha256"],
                provenance={**original_receipt["provenance"], "native_vertices_sha256": FIRST_IDENTITY_SHA,
                    "historical_native_vertices_sha256": original_receipt["native_vertices_identity"]["sha256"]}); persist()
            replayed, precise = replay_history(root, original, target, report)
            if report.get("actual_official_reference_replay_calls") != 1: raise ValueError("Exactly one full reference replay required")
            if (report.get("reference_torch") != receipt["workers"][0]["torch"]
                    or report.get("reference_CUDA") != receipt["workers"][0]["CUDA"]):
                raise ValueError("Reference replay runtime differs from target workers")
            report.update(historical_agreement(original["per_frame_vertex_error_mm"], replayed))
            errorspath = out / "replay_errors.npz"
            with errorspath.open("xb") as stream:
                np.savez_compressed(stream, frame_index=np.arange(790, dtype=np.int64),
                    historical_errors_f32=original["per_frame_vertex_error_mm"], replay_errors_f32=replayed,
                    replay_errors_float64=precise, drift_mm=replayed.astype(np.float64)-original["per_frame_vertex_error_mm"])
            errorspath.chmod(0o444); report["replay_errors_sha256"] = sha256(errorspath); persist()
            for path, digest in frozen: convert._require_hash(regular(path), digest)
            if convert.canonical_array_identity(target) != report["target_vertices_identity"]:
                raise ValueError("First target changed during replay")
            if any(convert.canonical_array_identity(original[k]) != identity for k, identity in original_ids.items()):
                raise ValueError("Replay mutated original controls or historical error vector")
            source_chain(root, params, original_receipt)
            if not report["historical_error_agreement"]: raise ValueError("Historical errors disagree beyond unchanged tolerance")
            report.update(status="pass", phase="complete", warmstart_numerical_agreement_only=True,
                source_bindings_runtime_verified=True, original_parameters_unchanged=True,
                file_inventory=[{"file": errorspath.name, "sha256": sha256(errorspath), "bytes": errorspath.stat().st_size}]); persist()
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "historical_error_agreement")}))


if __name__ == "__main__": main()

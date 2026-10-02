"""One constrained identity probe of an explicit, saved new inverse target.

The failed old-target replay and strict reseal stay failed. D81 authorizes only
numerical agreement of the unchanged warm start, not strict determinism or
accuracy. No decoder/cold converter is rerun and no full submission is made.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

import cari_converter as convert
from cari_identity_probe import identity_probe
from cari_identity_probe_runtime import BUDGET, native_callbacks
from cari_frozen_target_replay import (FIRST_IDENTITY_SHA, RESEAL_SHA, STAGE as REPLAY_STAGE,
                                      frozen_target, require_fields)
from cari_target_reseal import (FAILED_SHA, SEALED_SHA, SETTINGS, failed_source, historical_agreement,
                               regular, sealed_inputs, source_chain, strict_torch)
from world_reward.data import sha256

STAGE = "world_reward_cari_frozen_target_shared_identity_runtime_probe"
REPLAY_SHA = "2c0269f5eb02f07e8d9791b059527059cb8d592b521a0abd9edf3cfae8432280"
REPLAY_REVISION = "e1e8e92a63c690174f5bc40733939235fe46d04a"
REPLAY_SCRIPT_SHA = "5ae29d8586975dc2af66489bc943fbc4cf0beac849e2496097408bfea0ac6b51"
REPLAY_ERRORS_SHA = "50b9595422ab2c596e317e6aa6996e79189939c742ea428b28c1df04a5851910"
TARGET_FILE_SHA = "fcb86e264525ca7c6b93a6878061c3523a71af1385d5a05aef8bd017bc76f7d1"


def error_arrays(archive, original_errors):
    keys = {"frame_index", "historical_errors_f32", "replay_errors_f32", "replay_errors_float64", "drift_mm"}
    if set(archive) != keys: raise ValueError("Exact complete D81 replay-error inventory required")
    indices, historical, replayed = (np.asarray(archive[k]) for k in
        ("frame_index", "historical_errors_f32", "replay_errors_f32"))
    precise, drift = (np.asarray(archive[k]) for k in ("replay_errors_float64", "drift_mm"))
    if (indices.dtype != np.int64 or not np.array_equal(indices, np.arange(790))
            or any(v.shape != (790,) for v in (historical, replayed, precise, drift))
            or historical.dtype != np.float32 or replayed.dtype != np.float32
            or precise.dtype != np.float64 or drift.dtype != np.float64
            or any(not np.isfinite(v).all() for v in (historical, replayed, precise, drift))
            or np.any(precise < 0) or historical.tobytes() != original_errors.tobytes()
            or not np.array_equal(replayed, precise.astype(np.float32))
            or not np.array_equal(drift, replayed.astype(np.float64)-historical.astype(np.float64))):
        raise ValueError("D81 errors must preserve full original coverage, values, and exact arithmetic")
    agreement = historical_agreement(historical, replayed)
    if not agreement["historical_error_agreement"]: raise ValueError("Actual unchanged D81 historical agreement required")
    return agreement


def replay_receipt(root, original, original_receipt, target_receipt, image):
    directory = root / "outputs/episode_000000/cari_frozen_target_replay_refined_v1"
    path = regular(directory / "report.json"); convert._require_hash(path, REPLAY_SHA)
    report = json.loads(path.read_text())
    target_identity = target_receipt["workers"][0]["target_vertices_identity"]
    provenance = {**original_receipt["provenance"], "native_vertices_sha256": FIRST_IDENTITY_SHA,
                  "historical_native_vertices_sha256": original_receipt["native_vertices_identity"]["sha256"]}
    parameter_ids = {k: convert.canonical_array_identity(original[k]) for k in
                     ("pose", "scales", "shape", "per_frame_vertex_error_mm")}
    require_fields(report, {"stage": REPLAY_STAGE, "status": "pass", "phase": "complete",
        "producer_revision": REPLAY_REVISION, "script_sha256": REPLAY_SCRIPT_SHA, "image_id": image,
        "network": "none", "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
        "oracle_modes": [], "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "strict_reproducibility_verified": False, "historical_target_recovered": False, "new_inverse_target": True,
        "selection_rule": "first_worker_protocol_order_not_error", "full_frame_fit_verified": False,
        "actual_native_decode_calls": 0, "solver_calls": 0, "original_conversion_rerun": False,
        "source_bindings_runtime_verified": True, "warmstart_numerical_agreement_only": True,
        "original_parameters_unchanged": True, "sealed_report_sha256": SEALED_SHA,
        "failed_probe_report_sha256": FAILED_SHA, "failed_reseal_report_sha256": RESEAL_SHA,
        "original_parameter_identities": parameter_ids, "historical_native_vertices_identity": original_receipt["native_vertices_identity"],
        "target_vertices_identity": target_identity, "target_file_sha256": TARGET_FILE_SHA, "provenance": provenance,
        "actual_official_reference_replay_calls": 1, "reference_precision": "float32", "reference_model_chunk": 256,
        "reference_residual_dtype": "float64", "reference_error_storage_dtype": "float32", "reference_settings": SETTINGS,
        "reference_torch": target_receipt["workers"][0]["torch"], "reference_CUDA": target_receipt["workers"][0]["CUDA"],
        "official_converter_sha256": convert.CONVERTER_SHA256, "reference_model_sha256": convert.REFERENCE_MODEL_SHA256,
        "replay_errors_sha256": REPLAY_ERRORS_SHA}, "Actual pinned D81 warm-start agreement receipt required")
    if (image != target_receipt["image_id"] or target_identity["sha256"] != FIRST_IDENTITY_SHA
            or target_receipt["workers"][0]["target_file_sha256"] != TARGET_FILE_SHA):
        raise ValueError("D81 and new inverse target must use the identical immutable image and array")
    errorspath = regular(directory / "replay_errors.npz"); convert._require_hash(errorspath, REPLAY_ERRORS_SHA)
    if errorspath.stat().st_mode & 0o222: raise ValueError("D81 replay error archive must remain immutable")
    with np.load(errorspath, allow_pickle=False) as archive:
        agreement = error_arrays({k: archive[k] for k in archive.files}, original["per_frame_vertex_error_mm"])
    require_fields(report, agreement, "D81 receipt must agree with its full stored replay errors")
    require_fields(report, {"file_inventory": [{"file": errorspath.name, "sha256": REPLAY_ERRORS_SHA,
        "bytes": errorspath.stat().st_size}]}, "Exact D81 replay-error inventory required")
    return report, [(path, REPLAY_SHA), (errorspath, REPLAY_ERRORS_SHA)]


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline Azure CUDA container required")
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image)):
        raise ValueError("Canonical root and immutable producing source/image required")
    out = root / "outputs/episode_000000/cari_identity_probe_frozen_refined_v1"
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty diagnostic output required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "script_sha256": sha256(Path(__file__)),
        "producer_revision": revision, "image_id": image, "budget_seconds": BUDGET, "network": "none",
        "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "research_only": True, "submission_produced": False, "adoption_authorized": False,
        "strict_reproducibility_verified": False, "historical_target_recovered": False, "new_inverse_target": True,
        "selection_rule": "first_worker_protocol_order_not_error", "full_frame_fidelity_verified": False,
        "actual_native_decode_calls": 0, "original_conversion_rerun": False, "source_bindings_runtime_verified": False,
        "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA,
        "failed_reseal_report_sha256": RESEAL_SHA, "warmstart_replay_report_sha256": REPLAY_SHA}
    with (out / "report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Complete frozen-target shared-identity probe exceeded900s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); original, params, original_receipt, frozen = sealed_inputs(root)
            frozen.append((failed_source(root), FAILED_SHA)); _, _, more = source_chain(root, params, original_receipt); frozen += more
            target, target_receipt, more = frozen_target(root, original_receipt); frozen += more
            warmstart, more = replay_receipt(root, original, original_receipt, target_receipt, image); frozen += more
            report.update(phase="official_reference_bindings", target_vertices_identity=warmstart["target_vertices_identity"],
                target_file_sha256=TARGET_FILE_SHA, warmstart_historical_agreement=True,
                original_parameter_identities=warmstart["original_parameter_identities"]); persist()
            tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"; model = root / "weights/mhr/mhr_model.pt"
            convert._require_hash(regular(tool), convert.CONVERTER_SHA256); convert._require_hash(regular(model), convert.REFERENCE_MODEL_SHA256)
            torch = strict_torch()
            spec = importlib.util.spec_from_file_location("world_reward_frozen_identity_official_converter", tool)
            converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
            provenance = {**warmstart["provenance"], "new_inverse_target": True,
                          "warmstart_replay_report_sha256": REPLAY_SHA, "failed_reseal_report_sha256": RESEAL_SHA}
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                joint, pose, replay, calls = native_callbacks(torch, converter, tool, model, report)
                report.update(phase="shared_identity_probe", reference_settings=dict(SETTINGS)); persist()
                result = identity_probe(original, params, target, episode_index=0, frame_index=np.arange(790, dtype=np.int64),
                    provenance=provenance, joint_fit=joint, pose_polish=pose, reference_f32=replay)
            if calls != {"joint": 1, "pose": 1, "replay": 2}: raise ValueError("Exactly one joint, one reserved pose and two F32 replays required")
            for path, digest in frozen + [(tool, convert.CONVERTER_SHA256), (model, convert.REFERENCE_MODEL_SHA256)]:
                convert._require_hash(regular(path), digest)
            if convert.canonical_array_identity(target) != warmstart["target_vertices_identity"]:
                raise ValueError("Frozen inverse target changed during the probe")
            for key, identity in warmstart["original_parameter_identities"].items():
                if convert.canonical_array_identity(original[key]) != identity: raise ValueError("Original warm start mutated during the probe")
            source_chain(root, params, original_receipt)
            proposal = out / "diagnostic_proposal.npz"
            with proposal.open("xb") as stream: np.savez_compressed(stream, **{k: v for k, v in result.items() if k != "report"})
            proposal.chmod(0o444)
            result["report"].update(source_bindings_runtime_verified=True, actual_official_joint_verified=True,
                actual_official_reserved_pose_verified=True, final_reference_float32_runtime_verified=True,
                strict_reproducibility_verified=False, historical_target_recovered=False, new_inverse_target=True)
            # Completion certifies calls/lineage only. The unchanged numerical
            # hypothesis verdict is independently stored in probe_report.
            report.update(status="pass", phase="complete", source_bindings_runtime_verified=True,
                actual_official_joint_and_pose_verified=True, probe_report=result["report"],
                probe_gate_pass=result["report"]["probe_gate_pass"], proposal_sha256=sha256(proposal)); persist()
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "probe_gate_pass", "elapsed_seconds")}))


if __name__ == "__main__": main()

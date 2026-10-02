"""CPU replay measurement of one frozen rejected mesh; never a prediction.

Historical control evidence stays historical. This cannot prove bitwise identity
to the lost candidate: its original arrays/maps were not hashed or retained.
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
import guarded_mesh_gate as guarded
import object_budget_endpoint as endpoint
from mesh_link_gate import mesh_topology, _array_hash, _write
from mesh_endpoint_gate import source_intersections
from world_reward.data import sha256

STAGE = "frozen_guarded_mesh_rejection_replay_diagnostic"
HISTORICAL_REVISION = "8cdad209b845adaa0cda6036f53bf4265962daf2"
HISTORICAL_PRODUCER_SHA = "b08793257334615694ba283d76733d41800cf3083bcd31b93d7f408af9109e55"
HISTORICAL_HELPER_SHA = "8154b29bc724676066709b3a4ffc60c0fa60c5acac904b29a531fc680713d875"
HISTORICAL_CPP_SHA = "964d89560cbbae6fa2dd8b76e480a6ab7dae3073e0fa419604707e4a0d9af081"
NUMERICAL_REJECTION = "Frozen1% geometry/5% net and per-shell volume gates failed"
BUDGET = 120


def prerequisites(root, episode):
    """Bind frozen historical receipt/source, not current production eligibility."""
    base = root / f"outputs/episode_{episode:06d}"
    paths = {"rejected_report": base/"object_budget_guarded/report.json", "object_report": base/"object_grounded/report.json",
             "object.glb": base/"object_grounded/object.glb", "control_report": root/"validation/guarded_qem_v1/report.json"}
    receipts = {name: sha256(endpoint.regular(root, path)) for name, path in paths.items()}
    old = json.loads(paths["rejected_report"].read_text()); control = json.loads(paths["control_report"].read_text())
    obj = json.loads(paths["object_report"].read_text()); build = guarded.validate_build(root)
    expected = {"stage": "world_reward_cpu_guarded_object_mesh", "status": "fail", "episode_index": episode,
        "producer_revision": HISTORICAL_REVISION, "script_sha256": HISTORICAL_PRODUCER_SHA,
        "image_id": os.environ.get("WR_IMAGE_ID"), "input_track": "track_1", "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "error_type": "ValueError", "error": NUMERICAL_REJECTION, "source_intersecting_faces": 0,
        "independent_candidate_intersecting_faces": 0, "target_faces": 4096, "target_vertices": 4096}
    if episode != 0 or any(type(old.get(k)) is not type(v) or old[k] != v for k, v in expected.items()):
        raise ValueError("Require exact original episode0 numerical rejection receipt")
    sources = old.get("source_hashes", {})
    if (sources.get("object.glb") != receipts["object.glb"] or sources.get("object_report") != receipts["object_report"]
            or obj.get("object_sha256") != receipts["object.glb"] or obj.get("status") != "pass"
            or obj.get("stage") != "sam3d_objects_grounded_fixed_frame" or type(obj.get("episode_index")) is not int or obj["episode_index"] != episode
            or not isinstance(old.get("input_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", old["input_sha256"])
            or obj.get("input_sha256") != old.get("input_sha256") or obj.get("ground_truth_used") is not False
            or obj.get("hand_labeled_test") is not False or obj.get("oracle_modes") != []):
        raise ValueError("Historical object source receipt/hash mismatch")
    if (sha256(Path(__file__).with_name("mesh_guarded_qem.cpp")) != HISTORICAL_CPP_SHA
            or old.get("control_evidence") != {"guarded_gate_sha256": receipts["control_report"], **build}
            or control.get("stage") != guarded.STAGE or control.get("status") != "pass"
            or control.get("script_sha256") != HISTORICAL_HELPER_SHA or control.get("build") != build
            or control.get("target_faces") != 4096 or control.get("target_vertices") != 4096
            or control.get("challenge_inputs_used") is not False or control.get("adoption_performed") is not False
            or [r.get("fixture") for r in control.get("fixtures", [])] != ["new_close_asymmetric_shells", "new_disconnected_smooth_asymmetric"]
            or any(r.get("independent_intersecting_faces") != 0 or r.get("source_arrays_unchanged") is not True for r in control["fixtures"])
            or control["fixtures"][0].get("containment", {}).get("true_containment_verified") is not True):
        raise ValueError("Historical controls/native binary/source binding mismatch")
    receipts.update(build=build, historical_helper_sha256=HISTORICAL_HELPER_SHA,
                    historical_producer_sha256=HISTORICAL_PRODUCER_SHA, historical_revision=HISTORICAL_REVISION)
    return paths, old, receipts


def rejection_axes(metrics):
    shells = metrics["birthface_matched_shells"]
    return {"sampled_chamfer_diagonal_over_1_percent": metrics["sampled_bidirectional_chamfer_diagonal_ratio"] > .01,
            "net_volume_over_5_percent": metrics["net_volume_relative_error"] > .05,
            "source_components_over_5_percent": [r["source_component"] for r in shells if r["relative_volume_error"] > .05]}


def measure(root, episode, report, path):
    paths, old, receipts = prerequisites(root, episode)
    report.update(frozen_receipts=receipts, same_frozen_inputs_and_native_configuration=True); _write(path, report)
    source_v, source_f, weld = endpoint.exact_weld(*endpoint._load_mesh(paths["object.glb"])); source = source_v, source_f
    source_hashes = [_array_hash(x) for x in source]; topology = mesh_topology(*source)
    if topology != old.get("source_topology"): raise ValueError("Source topology differs from frozen source receipt")
    report.update(exact_source_welding=weld, source_array_sha256=source_hashes, phase="same_native_replay"); _write(path, report)
    candidate, mapping = guarded.simplify(source, 110)
    cv, cf = candidate; report["candidate_array_sha256"] = [_array_hash(x) for x in candidate]
    artifact = path.parent/"candidate_diagnostic.npz"
    with artifact.open("xb") as handle: np.savez_compressed(handle, vertices=cv, faces=cf)
    map_path = path.parent/"mapping_diagnostic.json"
    with map_path.open("x") as handle: json.dump(mapping, handle, separators=(",", ":")); handle.write("\n")
    report.update(candidate_diagnostic_sha256=sha256(artifact), mapping_diagnostic_sha256=sha256(map_path),
                  candidate_topology=mesh_topology(*candidate), independent_candidate_intersecting_faces=source_intersections(*candidate))
    report["matches_available_historical_candidate_signature"] = (report["candidate_topology"] == old.get("candidate_topology")
        and report["independent_candidate_intersecting_faces"] == old["independent_candidate_intersecting_faces"])
    _write(path, report)
    if not report["matches_available_historical_candidate_signature"]: raise ValueError("Replay differs from available historical candidate signature")
    metrics = {}; report["measurements"] = metrics
    try:
        guarded.mapped_geometry(source, candidate, mapping, metrics); report["unchanged_geometry_gate_status"] = "pass"
    except ValueError as error:
        if str(error) != NUMERICAL_REJECTION or not metrics: raise
        report.update(unchanged_geometry_gate_status="fail", replay_numerical_rejection=str(error))
    finally: _write(path, report)
    report["rejection_axes"] = rejection_axes(metrics)
    if source_hashes != [_array_hash(x) for x in source] or prerequisites(root, episode)[2] != receipts:
        raise ValueError("Frozen source/provenance changed during diagnostic")
    report.update(status="measurement_complete", phase="complete", source_arrays_unchanged=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--episode", type=int, choices=(0,), required=True); args = parser.parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require isolated remote CPU network-none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ["WR_CODE_REVISION"]; image = os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable diagnostic source/image")
    output = root/f"diagnostic/episode{args.episode:06d}-guarded-replay"; path = output/"report.json"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved diagnostic output")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "episode_index": args.episode, "budget_seconds": BUDGET,
        "producer_revision": revision, "image_id": image, "script_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(guarded.__file__)),
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "historical_rejection_unchanged": True, "production_gate_pass_claimed": False, "prediction_exported": False,
        "historical_candidate_array_sha_available": False, "historical_candidate_bit_identity_proven": False,
        "historical_control_evidence_only": True, "same_frozen_inputs_and_native_configuration": False}
    start = time.perf_counter()
    def expired(*args): raise TimeoutError("Whole diagnostic exceeded120s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: _write(path, report); measure(root, args.episode, report, path)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-start; _write(path, report)
    print(json.dumps({k: report[k] for k in ("stage", "status", "rejection_axes", "elapsed_seconds")}))


if __name__ == "__main__": main()

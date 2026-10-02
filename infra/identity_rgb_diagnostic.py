"""Posthoc error decomposition of frozen D87 predictions, without alignment.

The original rejection is replayed and preserved. Centroid subtraction is only
an algebraic decomposition of errors: no predictions are corrected or exported.
This is neither a Track1 score nor a new candidate selection experiment.
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

import identity_rgb_evaluate as evaluate
from world_reward.data import sha256

STAGE = "posthoc_frozen_identity_rgb_centroid_rms_diagnostic"
ORIGINAL_SHA = "6d598bc7e6f4d0063189f652bef6241ec91a0caa4be281a6c2cc74cee8ee3883"
PREDICTION_SHA = "f01674dd2e196b16a7ec0657dbf033414424a324dc6abd5ada62817fc2188df4"
BUDGET = 120
MODES = ("raw", "shared")


def decomposition(prediction, truth):
    """Exact all-correspondence RMS decomposition in float64, meters in/cm out."""
    if any(np.ma.isMaskedArray(v) for v in (prediction, truth)):
        raise ValueError("Hidden vertex validity is forbidden")
    p, q = np.asarray(prediction), np.asarray(truth)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or p.shape != q.shape
            or p.dtype.kind != "f" or q.dtype.kind != "f"
            or not np.isfinite(p).all() or not np.isfinite(q).all()):
        raise ValueError("Complete paired finite floating camera-space vertices required")
    error = p.astype(np.float64)-q.astype(np.float64)
    if not np.isfinite(error).all():
        raise ValueError("Finite float64 vertex errors required")
    centroid = error.mean(axis=0)
    camera_m2 = float(np.square(error).sum(axis=1).mean())
    centered_m2 = float(np.square(error-centroid).sum(axis=1).mean())
    centroid_m2 = float(centroid@centroid)
    if not np.isfinite([camera_m2, centered_m2, centroid_m2]).all():
        raise ValueError("Finite float64 squared errors required")
    residual_m2 = camera_m2-centroid_m2-centered_m2
    if not np.isclose(camera_m2, centroid_m2+centered_m2, atol=1e-14, rtol=1e-12):
        raise ValueError("All-vertex RMS decomposition identity failed")
    return {"vertex_count": len(p), "camera_pve_cm": float(np.linalg.norm(error, axis=1).mean()*100),
        "centroid_error_xyz_cm": (centroid*100).tolist(),
        "centroid_error_norm_cm": float(np.sqrt(centroid_m2)*100),
        "camera_rms_cm": float(np.sqrt(camera_m2)*100),
        "centered_rms_cm": float(np.sqrt(centered_m2)*100),
        "centroid_squared_error_fraction": centroid_m2/camera_m2 if camera_m2 > 0 else None,
        "rms_squared_identity_residual_cm2": residual_m2*10000}


def historical_rejection(root):
    path = root/evaluate.BASE/"quality_v2/report.json"
    evaluate.regular_hash(path, ORIGINAL_SHA)
    original = json.loads(path.read_text())
    evaluate.require_fields(original, {"stage": evaluate.STAGE, "status": "pass", "phase": "complete",
        "prediction_report_sha256": PREDICTION_SHA, "hyperparameters": evaluate.GATES,
        "adoption_authorized": False, "all_frames_scored": True, "all_cases_retained": True,
        "no_gt_alignment": True, "predictions_frozen_before_private": True})
    evaluate.require_fields(original.get("decision"), {
        "synthetic_identity_hypothesis_supported": False, "adoption_authorized": False})
    replay = {}
    # The existing evaluator audits every public artifact before its FIRST
    # private read and writes only to this in-memory dictionary, never old files.
    evaluate.run(root, replay)
    for key in ("decision", "frame_metrics", "clip_metrics", "prediction_report_sha256",
                "private_render_report_sha256", "common_object_scale_diagnostics"):
        # Historical reports are JSON: native dataclass tuples serialize to
        # lists. Compare identical JSON values, with NO numerical tolerance.
        if json.loads(json.dumps(replay.get(key), allow_nan=False)) != original.get(key):
            raise ValueError("Original frozen D87 rejection replay differs: "+key)
    evaluate.regular_hash(path, ORIGINAL_SHA)
    return original, path


def diagnose(root, report):
    original, original_path = historical_rejection(root)
    records, _, _, pairs, _, frozen, prediction_sha = evaluate.public_predictions(root)
    if prediction_sha != PREDICTION_SHA:
        raise ValueError("Original frozen prediction producer differs")
    private = root/evaluate.BASE/"eval_private"
    render_path = private/"render-report.json"
    render_sha = evaluate.regular_hash(render_path, original["private_render_report_sha256"])
    render = json.loads(render_path.read_text())
    cases = render.get("cases")
    if (not isinstance(cases, list) or len(cases) != evaluate.CLIPS*evaluate.FRAMES
            or len(records) != len(cases) or len(pairs) != len(cases)
            or not isinstance(original.get("frame_metrics"), list)
            or len(original["frame_metrics"]) != len(cases)):
        raise ValueError("All fifteen original frames required")
    frozen += [(original_path, ORIGINAL_SHA), (render_path, render_sha)]
    rows = []
    for record, pair, case, prior in zip(records, pairs, cases, original["frame_metrics"]):
        expected = {key: record[key] for key in ("file", "clip_index", "frame_index")}
        expected["rgb_sha256"] = record["sha256"]
        evaluate.require_fields(case, expected)
        evaluate.require_fields(prior, {key: record[key] for key in ("clip_index", "frame_index")})
        path = private/(Path(record["file"]).stem+".npz")
        digest = evaluate.regular_hash(path, case.get("truth_sha256"))
        with np.load(path, allow_pickle=False) as archive:
            truth = {key: archive[key].copy() for key in archive.files}
        evaluate.validate_truth(truth, record, pair["human_faces"])
        values = {mode: decomposition(pair[mode+"_vertices_camera_m"], truth["human_vertices_camera_m"])
                  for mode in MODES}
        for mode in MODES:
            if (values[mode]["vertex_count"] != evaluate.VERTICES or not np.isclose(
                    values[mode]["camera_pve_cm"], prior[mode]["human_pve_cm"], atol=1e-10, rtol=1e-12)):
                raise ValueError("Complete original camera-space PVE must remain unchanged")
        rows.append({"clip_index": record["clip_index"], "frame_index": record["frame_index"], **values})
        frozen.append((path, digest))
    clips = []
    for clip in range(evaluate.CLIPS):
        frames = rows[clip*evaluate.FRAMES:(clip+1)*evaluate.FRAMES]
        if [r["frame_index"] for r in frames] != list(range(evaluate.FRAMES)) or any(r["clip_index"] != clip for r in frames):
            raise ValueError("Original complete clip/frame order required")
        clips.append({"clip_index": clip, **{mode: {key: float(np.mean([r[mode][key] for r in frames]))
            for key in ("camera_pve_cm", "centroid_error_norm_cm", "camera_rms_cm", "centered_rms_cm")}
            for mode in MODES}})
    for path, digest in frozen:
        evaluate.regular_hash(path, digest)
    report.update(original_quality_report_sha256=ORIGINAL_SHA, prediction_report_sha256=PREDICTION_SHA,
        original_rejection_replayed_and_unchanged=True, original_decision=original["decision"],
        original_object_proxy_public_scale_replayed=True, frame_metrics=rows, clip_mean_frame_metrics=clips,
        all_frames_retained=True, vertices_per_frame=evaluate.VERTICES)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); out = root/evaluate.BASE/"diagnostic_centroid_v2"
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not out.is_dir() or out.resolve() != out.absolute() or any(out.iterdir())
            or not re.fullmatch("[0-9a-f]{40}", revision)
            or image != "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"):
        raise ValueError("Fresh exclusive canonical offline CPU diagnostic/source/image required")
    report = {"stage": STAGE, "status": "fail", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "evaluation_helper_sha256": sha256(Path(evaluate.__file__)),
        "budget_seconds": BUDGET, "network": "none", "gpu_used": False, "model_forward_performed": False,
        "posthoc_diagnostic_only": True, "challenge_inputs_used": False, "inference_truth_used": False,
        "private_truth_used_for_evaluation_only": True, "predictions_modified": False,
        "alignment_performed": False, "human_scale_or_camera_fit_performed": False,
        "adoption_authorized": False, "new_quality_selection_performed": False, "Track1_score_verified": False}
    start = time.perf_counter(); path = out/"report.json"
    with path.open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Frozen identity diagnostic exceeded120s")
        old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        try:
            persist(); diagnose(root, report); report["status"] = "pass"
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term)
            persist(); path.chmod(0o444)


if __name__ == "__main__": main()

"""One frame-zero shared Sim3 diagnostic on frozen own synthetic predictions.

Fit uses the private LBS-nonhand vertex complement, NOT official role_alignment.
Truth is evaluation-only; no transformed predictions are exported, selected or
adopted. The original H1 rejection and its numerical gates remain unchanged.
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
import hand_synthetic_evaluate as evaluate
from world_reward.data import sha256


def fit_sim3(source, target):
    """Own positive scale/proper row-rotation least-squares similarity fit."""
    a, b = np.asarray(source), np.asarray(target)
    if (a.ndim != 2 or a.shape[1] != 3 or len(a) < 3 or b.shape != a.shape
            or a.dtype.kind != "f" or b.dtype.kind != "f" or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Require paired finite floating 3D points")
    a, b = a.astype(np.float64), b.astype(np.float64)
    ma, mb = a.mean(0), b.mean(0); x, y = a-ma, b-mb
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Similarity proxy must have noncollinear extent")
    u, singular, vh = np.linalg.svd(x.T@y)
    correction = np.array([1., 1., -1. if np.linalg.det(u@vh) < 0 else 1.])
    rotation = (u*correction)@vh; scale = float((singular*correction).sum()/np.square(x).sum())
    translation = mb-scale*(ma@rotation)
    if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-12, rtol=0)):
        raise ValueError("Similarity requires positive finite scale and proper rotation")
    return scale, rotation, translation


def apply_sim3(points, transform):
    scale, rotation, translation = transform
    return scale*(np.asarray(points)@rotation)+translation


def diagnose(predictions, truth, regions):
    """One BASELINE frame-zero fit, reused for both modes and every case."""
    before = predictions["baseline_vertices_camera_m"]
    tv, tj = truth
    if before.shape != tv.shape or len(before) != evaluate.CASES: raise ValueError("Require all original cases")
    nonhand = ~(regions["left"]["mask"] | regions["right"]["mask"])
    if nonhand.dtype != np.bool_ or nonhand.shape != before.shape[1:-1] or nonhand.sum() < 3:
        raise ValueError("Require actual disjoint LBS hand complement")
    transform = fit_sim3(before[0, nonhand], tv[0, nonhand])
    cases = []
    for i in range(evaluate.CASES):
        record = {"case_index": i, "fit_case": i == 0, "modes": {}}
        for mode in ("baseline", "candidate"):
            pv, pj = predictions[mode+"_vertices_camera_m"][i], predictions[mode+"_joints_camera_m"][i]
            av, aj = apply_sim3(pv, transform), apply_sim3(pj, transform)
            result = {"raw_nonhand_pve_mm": float(np.linalg.norm(pv[nonhand]-tv[i, nonhand], axis=-1).mean()*1000),
                      "shared_Sim3_nonhand_pve_mm": float(np.linalg.norm(av[nonhand]-tv[i, nonhand], axis=-1).mean()*1000),
                      "raw_all_joint_mpjpe_mm": float(np.linalg.norm(pj-tj[i], axis=-1).mean()*1000),
                      "shared_Sim3_all_joint_mpjpe_mm": float(np.linalg.norm(aj-tj[i], axis=-1).mean()*1000), "hands": {}}
            for side, region in regions.items():
                raw, _ = evaluate.errors(pv, pj, tv[i], tj[i], region)
                aligned, _ = evaluate.errors(av, aj, tv[i], tj[i], region)
                wrist = region["wrist"]
                result["hands"][side] = {"raw": raw, "shared_Sim3": aligned,
                    "raw_wrist_error_vector_mm": ((pj[wrist]-tj[i, wrist])*1000).tolist(),
                    "shared_Sim3_wrist_error_vector_mm": ((aj[wrist]-tj[i, wrist])*1000).tolist()}
            record["modes"][mode] = result
        cases.append(record)
    scale, rotation, translation = transform
    return {"fit_frame_index": 0, "fit_prediction": "baseline_only", "fit_vertex_count": int(nonhand.sum()),
            "alignment_role": "private_LBS_hand_complement_proxy_NOT_official_role_alignment",
            "positive_scale": scale, "proper_row_rotation": rotation.tolist(), "translation_m": translation.tolist(),
            "rotation_determinant": float(np.linalg.det(rotation)), "cases": cases,
            "summary_cases": list(range(1, evaluate.CASES)), "summary": {
                mode: {label: float(np.mean([r["modes"][mode]["hands"][side][label]["hand_pve_mm"]
                    for r in cases[1:] for side in ("left", "right")])) for label in ("raw", "shared_Sim3")}
                for mode in ("baseline", "candidate")}}


def run(root, report):
    base = root/"validation/hands_rgb_v1"; private = base/"eval_private"
    original_path = base/"quality/report.json"; original_sha = sha256(original_path)
    evaluate.require_hash(original_path, original_sha); original = json.loads(original_path.read_text())
    if (original.get("stage") != "private_synthetic_hand_quality" or original.get("status") != "pass"
            or original.get("decision", {}).get("synthetic_hypothesis_supported") is not False):
        raise ValueError("Require the original completed H1 rejection, never overwrite or overturn it")
    # This existing strict loader checks BOTH original prediction SHA chains before
    # its first private truth read. It does not write or perform model inference.
    validated = {}; evaluate.run(root, validated)
    chain = ("proposals_sha256", "proposal_report_sha256", "inference_report_sha256", "render_report_sha256", "rig_sha256")
    if any(validated[key] != original.get(key) for key in chain): raise ValueError("Original H1 evidence chain differs")
    report.update(original_quality_report_sha256=original_sha, original_H1_hypothesis_rejected=True,
                  verified_evidence_chain={key: validated[key] for key in chain}, predictions_frozen_before_private_truth_read=True)
    with np.load(base/"official_proposals/proposals.npz", allow_pickle=False) as source:
        predictions = {mode+"_"+field: evaluate.finite(source[mode+"_"+field], shape)
            for mode in ("baseline", "candidate") for field, shape in (
                ("vertices_camera_m", (6,18439,3)), ("joints_camera_m", (6,127,3)))}
    render = json.loads((private/"render-report.json").read_text())
    with np.load(private/"rig.npz", allow_pickle=False) as source: rig = {k: source[k].copy() for k in source.files}
    regions = evaluate.hand_partition(render["joint_names"], rig); tv, tj = [], []
    for i, record in enumerate(render["cases"]):
        path = private/f"case_{i:03d}.npz"; evaluate.require_hash(path, record["truth_sha256"])
        with np.load(path, allow_pickle=False) as source:
            tv.append(evaluate.finite(source["vertices_camera_m"], (18439,3)))
            tj.append(evaluate.finite(source["joints_camera_m"], (127,3)))
    report["diagnostic"] = diagnose(predictions, (np.stack(tv), np.stack(tj)), regions)
    evaluate.require_hash(original_path, original_sha)
    for key, relative in (("proposals_sha256", "official_proposals/proposals.npz"), ("proposal_report_sha256", "official_proposals/report.json"),
                          ("inference_report_sha256", "predictions-v2/report.json"), ("render_report_sha256", "eval_private/render-report.json"),
                          ("rig_sha256", "eval_private/rig.npz")):
        evaluate.require_hash(base/relative, validated[key])


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote isolated CPU evaluation")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source/image")
    output = root/"validation/hands_rgb_v1/quality-gauge"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusive reserved gauge diagnostic output")
    with (output/"report.json").open("x") as handle:
        started = time.perf_counter()
        report = {"stage": "private_hand_shared_frame0_Sim3_diagnostic", "status": "fail", "code_revision": revision,
                  "image_id": image, "script_sha256": sha256(Path(__file__)), "evaluation_helper_sha256": sha256(Path(evaluate.__file__)),
                  "network": "none", "gpu_used": False, "model_forward_performed": False, "challenge_inputs_used": False,
                  "evaluation_truth_used": True, "inference_truth_used": False, "alignment_for_evaluation_only": True,
                  "official_alignment_roles_used": False, "Track1_score_verified": False, "per_frame_alignment": False,
                  "predictions_modified": False, "adoption_performed": False, "new_quality_selection_performed": False,
                  "object_interaction_evaluated": False, "budget_seconds": 60, "interpretation": "descriptive_common_gauge_vs_remaining_body_articulation_error"}
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Gauge diagnostic exceeded60s")
        previous = signal.signal(signal.SIGALRM, expired); signal.alarm(60)
        try:
            persist(); run(root, report); report["status"] = "pass"
        except Exception as error:
            report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()

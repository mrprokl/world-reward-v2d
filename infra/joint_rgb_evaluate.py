"""Private J1 visible-object metrics, after immutable RGB-only predictions freeze.

Raw camera Chamfer is the main score. A single first-frame human-correspondence
Sim3 per clip is a separately disclosed diagnostic, never inference correction.
This pixel-sampled visible-surface score is not the complete V2D mesh score.
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
from scipy.spatial import cKDTree
from world_reward.data import sha256

STAGE = "private_joint_rgb_shared_grounding_quality"
WIDTH, HEIGHT, CLIPS, FRAMES, SAMPLES = 1024, 768, 3, 3, 8192
FOCAL_GEOMETRY_SHA = "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05"


def require_hash(path, digest):
    path = Path(path)
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path.is_symlink() or not path.is_file() or path.resolve() != path.absolute()
            or sha256(path) != digest):
        raise ValueError("Frozen regular artifact SHA mismatch")


def points(value):
    x = np.asarray(value)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("Require nonempty finite floating [N,3] points")
    return x.astype(np.float64)


def faces(value, count):
    f = np.asarray(value)
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= count):
        raise ValueError("Require valid nonempty integer mesh faces")
    return f


def camera(value):
    K = np.asarray(value)
    if (K.shape != (3, 3) or K.dtype.kind != "f" or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0, 0, 1])
            or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError("Require finite positive zero-skew pinhole K")
    return K.astype(np.float64)


def sample_points(value):
    x = points(value)
    if len(x) <= SAMPLES:
        return x
    selected = np.sort(np.random.default_rng(0).choice(len(x), SAMPLES, replace=False))
    return x[selected]


def visible_object_points(depth, face_index, K, human_face_count, object_face_count):
    """True scene z-buffer includes human occlusion; unproject at pixel +.5."""
    z, ids = np.asarray(depth), np.asarray(face_index); K = camera(K)
    if (z.ndim != 2 or z.dtype.kind != "f" or ids.shape != z.shape or ids.dtype.kind not in "iu"
            or type(human_face_count) is not int or human_face_count < 1
            or type(object_face_count) is not int or object_face_count < 1
            or np.any(ids < -1) or np.any(ids >= human_face_count + object_face_count)):
        raise ValueError("Invalid private scene depth/combined face-index contract")
    visible = ids >= human_face_count
    if not visible.any() or not np.isfinite(z[ids >= 0]).all() or np.any(z[ids >= 0] <= 0):
        raise ValueError("No finite positive true visible object surface")
    y, x = np.nonzero(visible); d = z[visible].astype(np.float64)
    result = np.column_stack(((x + .5 - K[0, 2]) * d / K[0, 0],
                              (y + .5 - K[1, 2]) * d / K[1, 1], d))
    return sample_points(result), int(visible.sum())


def fit_human_sim3(prediction, truth):
    """Proper positive Umeyama; one matched first-frame human, not object ICP."""
    x, y = points(prediction), points(truth)
    if x.shape != y.shape:
        raise ValueError("Human correspondence count differs")
    mx, my = x.mean(0), y.mean(0); a, b = x-mx, y-my
    variance = np.mean(np.sum(a*a, axis=1))
    covariance = b.T @ a / len(a)
    if variance <= 0 or np.linalg.matrix_rank(covariance) < 2:
        raise ValueError("Human correspondence Sim3 underconstrained")
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3); sign[-1] = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag(sign) @ vt
    scale = float(np.dot(singular, sign) / variance); t = my-scale*(mx @ R.T)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(t).all():
        raise ValueError("Invalid positive human Sim3")
    return {"scale": scale, "rotation": R.tolist(), "translation": t.tolist()}


def transform(value, sim3):
    return points(value) @ np.asarray(sim3["rotation"]).T * sim3["scale"] + np.asarray(sim3["translation"])


def metrics(predicted, truth, human, true_human):
    p, q, h, ht = points(predicted), points(truth), points(human), points(true_human)
    directed = cKDTree(q).query(p, workers=1)[0].mean(), cKDTree(p).query(q, workers=1)[0].mean()
    delta, true_delta = np.median(p, axis=0)-h.mean(0), np.median(q, axis=0)-ht.mean(0)
    return {"visible_chamfer_half_cm": float(.5*sum(directed)*100),
            "pred_to_true_cm": float(directed[0]*100), "true_to_pred_cm": float(directed[1]*100),
            "signed_object_median_Z_bias_cm": float((np.median(p[:, 2])-np.median(q[:, 2]))*100),
            "object_minus_human_centroid_m": delta.tolist(), "true_object_minus_human_centroid_m": true_delta.tolist(),
            "relative_visible_centroid_vector_error_cm": float(np.linalg.norm(delta-true_delta)*100),
            "relative_visible_centroid_distance_error_cm": float(abs(np.linalg.norm(delta)-np.linalg.norm(true_delta))*100)}


def decision(clip_metrics):
    if len(clip_metrics) != CLIPS:
        raise ValueError("Require all three clips, including wrong-focal controls")
    before = np.array([r["raw_visible_chamfer_half_cm"] for r in clip_metrics])
    after = np.array([r["aligned_visible_chamfer_half_cm"] for r in clip_metrics])
    if not np.isfinite(np.r_[before, after]).all() or np.any(before < 0) or np.any(after < 0):
        raise ValueError("Finite nonnegative complete scores required")
    gains = np.divide(before-after, before, out=np.full(CLIPS, np.nan), where=before > 0)
    ratios = np.divide(after-before, before, out=np.full(CLIPS, np.nan), where=before > 0)
    defined = bool(np.isfinite(gains).all())
    gain = float(np.median(gains)) if defined else None
    return {"median_paired_clip_relative_gain": gain,
            "per_clip_relative_gain": gains.tolist() if defined else None,
            "per_clip_relative_regression": ratios.tolist() if defined else None,
            "zero_baseline_gain_undefined": not defined,
            "synthetic_hypothesis_supported": bool(defined and gain >= .05 and ratios.max() <= .05),
            "adoption_performed": False, "full_v2d_score_verified": False}


def validate_predictions(data, camera_source="fixed"):
    expected = {"raw_points", "aligned_points", "human_vertices_camera_m", "human_faces", "object_masks",
                "moge_validity", "shared_scale", "frame_index", "clip_index", "camera_K"}
    if set(data) != expected:
        raise ValueError("Exact prediction-only array inventory required")
    if (data["raw_points"].shape != (9, HEIGHT, WIDTH, 3) or data["aligned_points"].shape != data["raw_points"].shape
            or data["human_vertices_camera_m"].shape != (9, 18439, 3)):
        raise ValueError("Fixed nine-frame prediction geometry shape mismatch")
    for key in ("raw_points", "aligned_points", "human_vertices_camera_m", "shared_scale"):
        if data[key].dtype.kind != "f": raise ValueError("Floating prediction geometry required")
    if not np.isfinite(data["human_vertices_camera_m"]).all(): raise ValueError("Nonfinite predicted human")
    faces(data["human_faces"], 18439)
    for key in ("object_masks", "moge_validity"):
        if data[key].shape != (9, HEIGHT, WIDTH) or data[key].dtype != np.dtype(bool):
            raise ValueError("Explicit full-grid boolean prediction masks required")
    for key, wanted in (("clip_index", np.repeat(np.arange(3), 3)), ("frame_index", np.tile(np.arange(3), 3))):
        if data[key].dtype.kind not in "iu" or not np.array_equal(data[key], wanted):
            raise ValueError("Original ordered clip/frame identity mismatch")
    scales = data["shared_scale"]
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("One positive finite scale per clip required")
    if camera_source == "fixed":
        K = camera(data["camera_K"])
        if not np.array_equal(K, [[1280., 0, WIDTH/2], [0, 1280., HEIGHT/2], [0, 0, 1.]]):
            raise ValueError("Require unchanged RGB-size prior K, not private calibration")
    elif camera_source == "learned":
        if data["camera_K"].shape != (3, 3, 3):
            raise ValueError("Require one RGB-learned K per complete clip")
        for value in data["camera_K"]:
            K = camera(value)
            if (K[0, 0] != K[1, 1] or K[0, 2] != WIDTH/2 or K[1, 2] != HEIGHT/2
                    or K[0, 1] != 0 or K[1, 0] != 0):
                raise ValueError("Learned camera must be centered square-pixel positive K")
    else:
        raise ValueError("Require an explicit fixed or learned camera producer")
    for i in range(9):
        valid = data["moge_validity"][i]; a, b = data["raw_points"][i][valid], data["aligned_points"][i][valid]
        if (not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a[:, 2] <= 0)
                or np.any(b[:, 2] <= 0) or not np.allclose(b, a*scales[i//3], rtol=2e-6, atol=1e-7)):
            raise ValueError("Aligned points must be raw XYZ scaled exactly once on valid pixels")
        if not (valid & data["object_masks"][i]).any(): raise ValueError("Missing predicted object coverage")


def load_public_predictions(base, *, camera_source="fixed"):
    public = base/"inputs"
    pred = base/("predictions" if camera_source == "fixed" else "predictions_camera_v1")
    receipt_path = pred/"report.json"; receipt_sha = sha256(receipt_path); require_hash(receipt_path, receipt_sha)
    receipt = json.loads(receipt_path.read_text())
    stage = ("public_joint_rgb_shared_grounding_predictions" if camera_source == "fixed"
             else "public_joint_rgb_learned_camera_shared_grounding_predictions")
    expected = {"stage": stage, "status": "pass", "private_truth_read": False,
                "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Require complete frozen RGB-only prediction producer")
    array_path = pred/"arrays.npz"; array_sha = receipt.get("arrays_sha256"); require_hash(array_path, array_sha)
    with np.load(array_path, allow_pickle=False) as archive: data = {k: archive[k].copy() for k in archive.files}
    validate_predictions(data, camera_source)  # ALL predictions validate before any private truth.
    if camera_source == "learned":
        candidates = np.asarray(receipt.get("focal_candidates_pixels"))
        solver_calls = receipt.get("native_focal_solver_calls")
        if (receipt.get("camera_source") != "learned" or receipt.get("focal_fitted") is not True
                or receipt.get("focal_geometry_source_sha256") != FOCAL_GEOMETRY_SHA
                or candidates.shape != (9,) or candidates.dtype.kind != "f"
                or not np.isfinite(candidates).all() or np.any(candidates <= 0)
                or not np.allclose(np.median(candidates.reshape(3, 3), axis=1), data["camera_K"][:, 0, 0], rtol=1e-6, atol=1e-5)
                or any(receipt.get(k) is not True for k in ("actual_body_inference", "actual_MoGe_inference", "actual_predicted_human_render", "native_camera_support_verified"))
                or any(type(receipt.get(k)) is not int or receipt[k] != n for k, n in
                       (("body_calls_completed", 9), ("MoGe_camera_calls_completed", 9),
                        ("MoGe_fixed_camera_calls_completed", 9), ("MoGe_calls_completed", 18)))
                or not isinstance(solver_calls, list) or len(solver_calls) != 18):
            raise ValueError("Learned camera must come from complete RGB-only median focal inference")
        for i, call in enumerate(solver_calls):
            row = i % 9; clip, frame = divmod(row, 3)
            if (not isinstance(call, dict) or call.get("file") != f"clip_{clip:02d}_frame_{frame:03d}.png"
                    or type(call.get("clip_index")) is not int or call["clip_index"] != clip
                    or type(call.get("frame_index")) is not int or call["frame_index"] != frame
                    or call.get("pass") != ("camera_candidate" if i < 9 else "shared_camera")
                    or call.get("focal_prior_supplied") is not (i >= 9)
                    or call.get("original_solver_returned") is not True
                    or type(call.get("native_nearest64_valid_pixels")) is not int
                    or not 2 <= call["native_nearest64_valid_pixels"] <= 4096):
                raise ValueError("Learned camera native solver support/call order incomplete")
    manifest_path = public/"manifest.json"; manifest_sha = sha256(manifest_path); require_hash(manifest_path, manifest_sha)
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != {"schema", "images"} or manifest["schema"] != "world-reward-joint-rgb-v1" or len(manifest["images"]) != 9:
        raise ValueError("Exact RGB-only public manifest required")
    consumed = receipt.get("public_records", [])
    if (receipt.get("public_inputs_sha") != manifest_sha or len(consumed) != 9
            or not isinstance(receipt.get("mask_report_sha"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["mask_report_sha"])):
        raise ValueError("Frozen inference/public manifest and mask provenance mismatch")
    for i, rgb in enumerate(manifest["images"]):
        clip, frame = divmod(i, 3); r = consumed[i]
        expected_file = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (not isinstance(rgb, dict) or set(rgb) != {"file", "sha256", "width", "height"}
                or rgb.get("file") != expected_file or type(rgb.get("width")) is not int or type(rgb.get("height")) is not int
                or (rgb["width"], rgb["height"]) != (WIDTH, HEIGHT) or not isinstance(r, dict)
                or r.get("file") != rgb.get("file") or r.get("rgb_sha256") != rgb.get("sha256")
                or type(r.get("clip_index")) is not int or r["clip_index"] != clip
                or type(r.get("frame_index")) is not int or r["frame_index"] != frame
                or any(not isinstance(r.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", r[k]) for k in ("human_mask_sha256", "object_mask_sha256"))):
            raise ValueError("Frozen prediction did not consume the original ordered RGB/masks")
        require_hash(public/rgb["file"], rgb["sha256"])
    return {"data": data, "receipt": receipt, "manifest": manifest, "receipt_sha": receipt_sha,
            "array_sha": array_sha, "manifest_sha": manifest_sha,
            "files": [(receipt_path, receipt_sha), (array_path, array_sha), (manifest_path, manifest_sha)]}


def run(root, report, camera_source="fixed"):
    base = root/"validation/joint_rgb_v1"; public = base/"inputs"; private = base/"eval_private"
    selected = load_public_predictions(base, camera_source=camera_source)
    data, receipt, manifest = selected["data"], selected["receipt"], selected["manifest"]
    receipt_sha, array_sha, manifest_sha = selected["receipt_sha"], selected["array_sha"], selected["manifest_sha"]
    baseline = None
    if camera_source == "learned":
        # Both prediction sets must freeze and validate before opening private GT.
        baseline = load_public_predictions(base, camera_source="fixed")
        if (baseline["manifest_sha"] != manifest_sha
                or baseline["receipt"].get("mask_report_sha") != receipt.get("mask_report_sha")
                or baseline["receipt"].get("public_records") != receipt.get("public_records")
                or not np.array_equal(baseline["data"]["object_masks"], data["object_masks"])
                or not np.array_equal(baseline["data"]["human_faces"], data["human_faces"])):
            raise ValueError("Paired camera experiment must consume the exact same ordered RGB/masks/topology")
        report.update(fixed_camera_prediction_report_sha256=baseline["receipt_sha"],
                      fixed_camera_arrays_sha256=baseline["array_sha"], paired_camera_predictions_frozen_before_private_truth_read=True)
    render_path = private/"render-report.json"; render_sha = sha256(render_path); require_hash(render_path, render_sha)
    render = json.loads(render_path.read_text())
    if (render.get("stage") != "own_joint_human_object_rgb_render" or render.get("status") != "pass"
            or render.get("synthetic_truth_used_for_rendering_only") is not True
            or render.get("inference_performed") is not False or render.get("challenge_inputs_used") is not False
            or len(render.get("cases", [])) != 9): raise ValueError("Require complete private own renderer")
    report.update(prediction_report_sha256=receipt_sha, arrays_sha256=array_sha, public_manifest_sha256=manifest_sha,
                  mask_report_sha256=receipt["mask_report_sha"], mask_report_independently_read=False,
                  render_report_sha256=render_sha, predictions_frozen_before_private_truth_read=True, frames=[], clips=[])
    if camera_source == "learned":
        report["learned_focal_pixels"] = data["camera_K"][:, 0, 0].tolist()
        report["private_true_focal_pixels_diagnostic"] = [1280., 960., 1600.]
        report["relative_focal_error_diagnostic_not_selection"] = (
            np.abs(data["camera_K"][:, 0, 0] - [1280., 960., 1600.]) / [1280., 960., 1600.]).tolist()
    sim3 = None; truth_hashes = []
    for i in range(9):
        clip, frame = divmod(i, 3); stem = f"clip_{clip:02d}_frame_{frame:03d}"; rgb = manifest["images"][i]; record = render["cases"][i]
        if (set(rgb) != {"file", "sha256", "width", "height"} or rgb["file"] != stem+".png"
                or type(rgb["width"]) is not int or type(rgb["height"]) is not int or (rgb["width"], rgb["height"]) != (WIDTH, HEIGHT)
                or record.get("file") != rgb["file"] or record.get("rgb_sha256") != rgb["sha256"]
                or type(record.get("clip_index")) is not int or record["clip_index"] != clip
                or type(record.get("frame_index")) is not int or record["frame_index"] != frame):
            raise ValueError("Original public/private RGB case identity mismatch")
        require_hash(public/rgb["file"], rgb["sha256"])
        truth_path = private/(stem+".npz"); require_hash(truth_path, record.get("truth_sha256")); truth_hashes.append((truth_path, record["truth_sha256"]))
        with np.load(truth_path, allow_pickle=False) as archive: truth = {k: archive[k].copy() for k in archive.files}
        if set(truth) != {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces", "camera_K",
                          "scene_depth_m", "visible_face_indices", "clip_index", "frame_index"}:
            raise ValueError("Exact private truth inventory required")
        human = points(truth["human_vertices_camera_m"]); faces(truth["human_faces"], len(human))
        obj = points(truth["object_vertices_camera_m"]); of = faces(truth["object_faces"], len(obj))
        if (human.shape != (18439, 3) or not np.array_equal(truth["human_faces"], data["human_faces"])
                or truth["scene_depth_m"].shape != (HEIGHT, WIDTH)
                or np.any(human[:, 2] <= 0) or np.any(obj[:, 2] <= 0)
                or any(truth[k].ndim != 0 or truth[k].dtype.kind not in "iu" or int(truth[k]) != value for k, value in (("clip_index", clip), ("frame_index", frame)))):
            raise ValueError("Private human correspondence/grid/original identity differs")
        f = (1280., 960., 1600.)[clip]
        if not np.array_equal(camera(truth["camera_K"]), [[f, 0, WIDTH/2], [0, f, HEIGHT/2], [0, 0, 1.]]):
            raise ValueError("Private frozen focal controls differ")
        true_points, true_count = visible_object_points(truth["scene_depth_m"], truth["visible_face_indices"], truth["camera_K"], len(data["human_faces"]), len(of))
        predicted_human = data["human_vertices_camera_m"][i]
        if frame == 0:
            sim3 = fit_human_sim3(predicted_human, human)
            report["clips"].append({"clip_index": clip, "human_first_frame_sim3": sim3})
        mask = data["object_masks"][i] & data["moge_validity"][i]
        p = sample_points(data["raw_points"][i][mask]); aligned = sample_points(data["aligned_points"][i][mask])
        scores = {"raw": metrics(p, true_points, predicted_human, human), "aligned": metrics(aligned, true_points, predicted_human, human)}
        diagnostic = {name: metrics(transform(value, sim3), true_points, transform(predicted_human, sim3), human) for name, value in (("raw", p), ("aligned", aligned))}
        permuted = p*data["shared_scale"][(clip+1)%3]
        report["frames"].append({"clip_index": clip, "frame_index": frame, "raw_camera": scores,
            "common_human_sim3_diagnostic": diagnostic, "alpha_permuted_negative_diagnostic": metrics(permuted, true_points, predicted_human, human),
            "true_visible_object_pixels": true_count, "predicted_object_valid_pixels": int(mask.sum()),
            "samples_each_max": SAMPLES, "truth_sha256": record["truth_sha256"]})
        if baseline is not None:
            prior = baseline["data"]; prior_mask = prior["object_masks"][i] & prior["moge_validity"][i]
            if not np.array_equal(truth["human_faces"], prior["human_faces"]):
                raise ValueError("Fixed and learned human correspondence topology differs")
            report["frames"][-1]["fixed_camera_baseline"] = {
                mode: metrics(sample_points(prior[key][i][prior_mask]), true_points,
                              prior["human_vertices_camera_m"][i], human)
                for mode, key in (("raw", "raw_points"), ("aligned", "aligned_points"))}
    for clip in range(3):
        entries = report["frames"][clip*3:clip*3+3]
        report["clips"][clip].update({f"{mode}_visible_chamfer_half_cm": float(np.mean([r["raw_camera"][mode]["visible_chamfer_half_cm"] for r in entries])) for mode in ("raw", "aligned")})
        report["clips"][clip]["common_human_sim3_diagnostic"] = {mode: float(np.mean([r["common_human_sim3_diagnostic"][mode]["visible_chamfer_half_cm"] for r in entries])) for mode in ("raw", "aligned")}
        if baseline is not None:
            report["clips"][clip]["fixed_camera_baseline"] = {
                mode: float(np.mean([r["fixed_camera_baseline"][mode]["visible_chamfer_half_cm"] for r in entries]))
                for mode in ("raw", "aligned")}
    report.update(decision=decision(report["clips"]), original_frame_coverage_verified=True,
                  shared_human_sim3_identical_for_both_methods=True, negative_diagnostic_used_for_selection=False)
    if baseline is not None:
        paired = [{"raw_visible_chamfer_half_cm": r["fixed_camera_baseline"]["aligned"],
                   "aligned_visible_chamfer_half_cm": r["aligned_visible_chamfer_half_cm"]} for r in report["clips"]]
        report["camera_comparison_decision"] = decision(paired)
        report["camera_comparison_scope"] = "learned_shared_K_aligned_vs_fixed_K1280_aligned_raw_camera_CD_all_nine_frames_no_alignment"
    for path, digest in [*selected["files"], *((baseline or {}).get("files", [])), (render_path, render_sha), *truth_hashes]:
        require_hash(path, digest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--camera-source", choices=("fixed", "learned"), default="fixed")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require isolated remote CPU evaluation")
    root = Path(os.environ["WR_ROOT"])
    out = root/"validation/joint_rgb_v1"/("quality" if args.camera_source == "fixed" else "quality_camera_v1"); path = out/"report.json"
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()): raise FileExistsError("Require exclusively reserved fresh quality directory")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image): raise ValueError("Immutable source/image required")
    report = {"stage": STAGE, "status": "fail", "code_revision": revision, "image_id": image, "script_sha256": sha256(Path(__file__)),
        "gpu_used": False, "private_truth_read": True, "challenge_inputs_used": False, "adoption_performed": False,
        "main_score_alignment_performed": False, "diagnostic_first_frame_human_sim3_per_clip": True,
        "object_alignment_performed": False, "full_v2d_score_verified": False, "metric_scope": "visible pixel-sampled surfaces, symmetric half means, centimetres",
        "budget_seconds": 60, "camera_source": args.camera_source,
        "preregistered_gate": {"median_paired_clip_gain": .05, "max_clip_regression": .05, "complete_frames": 9}}
    started = time.perf_counter()
    def expired(*args): raise TimeoutError("Private J1 evaluation exceeded60s")
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired); signal.alarm(60)
    try: run(root, report, args.camera_source); report["status"] = "pass"
    except Exception as exc: report.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        signal.alarm(0); report["elapsed_seconds"] = time.perf_counter()-started
        with path.open("x") as stream: json.dump(report, stream, allow_nan=False, indent=2)
    print(json.dumps({"stage": STAGE, "status": report["status"], "decision": report["decision"]}), flush=True)


if __name__ == "__main__": main()

"""Private real RGB-D object-only evaluation after all nine RGB predictions freeze.

Sensor points use BOP integer pixels and private K. Predicted XYZ retains its
own RGB-only camera, metre scale and MoGe +.5 rays; no GT recalibration, Sim3,
ICP, human score or complete V2D score. Visible GT masks are evaluation only.
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
from world_reward.pointmap import validate_camera_pointmap

REVISION = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
SELECTION = "sorted_RGB_filenames_first_median_index_n_div_2_last_per_scene_before_private_values"
FRAMES = ((1, 0), (1, 4074), (1, 8227), (2, 3), (2, 4013), (2, 7710), (3, 4), (3, 4028), (3, 7969))
WIDTH, HEIGHT, SAMPLES, BUDGET = 640, 480, 8192, 90
GEOMETRY_SHA = "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05"
MODEL_SHA = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
MODEL_SOURCE = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
ARRAY_KEYS = {"fixed_points", "fixed_depth", "fixed_validity", "fixed_K", "learned_points", "learned_depth", "learned_validity", "learned_K", "scene_id", "frame_id"}


def regular_hash(path, digest=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Require canonical regular artifact, no symlink")
    actual = sha256(path)
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or actual != digest):
        raise ValueError("Frozen artifact SHA mismatch")
    return actual


def camera(value):
    K = np.asarray(value)
    if (K.shape != (3, 3) or K.dtype.kind != "f" or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0
            or not np.array_equal(K[2], [0, 0, 1])):
        raise ValueError("Positive finite zero-skew pinhole camera required")
    return K.astype(np.float64)


def validate_arrays(data, scene, frame, learned_focal):
    if set(data) != ARRAY_KEYS:
        raise ValueError("Exact prediction-only array inventory required")
    for key, value in (("scene_id", scene), ("frame_id", frame)):
        if data[key].ndim != 0 or data[key].dtype.kind not in "iu" or int(data[key]) != value:
            raise ValueError("Original scene/frame scalar identity changed")
    for mode, focal in (("fixed", float(np.hypot(WIDTH, HEIGHT))), ("learned", learned_focal)):
        K = camera(data[mode+"_K"])
        expected = np.array([[focal, 0., WIDTH/2], [0., focal, HEIGHT/2], [0., 0., 1.]])
        if not np.allclose(K, expected, atol=1e-5, rtol=1e-6):
            raise ValueError("Prediction camera differs from RGB-only frozen shared focal")
        z, points, valid = (data[mode+suffix] for suffix in ("_depth", "_points", "_validity"))
        if (z.shape != (HEIGHT, WIDTH) or points.shape != (HEIGHT, WIDTH, 3) or valid.shape != z.shape
                or z.dtype != np.float32 or points.dtype != np.float32 or valid.dtype != np.bool_):
            raise ValueError("Original float32 depth/XYZ and explicit bool validity required")
        validate_camera_pointmap(z, points, valid, np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ K, K)


def public_predictions(base):
    """Validate ALL public RGB/predictions before opening any eval_private file."""
    public, pred = base/"inputs", base/"predictions_v1"
    mp, rp = public/"manifest.json", pred/"report.json"
    mh, rh = regular_hash(mp), regular_hash(rp)
    manifest, receipt = json.loads(mp.read_text()), json.loads(rp.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "revision", "license", "selection", "images"}
            or manifest["schema"] != "world-reward-tudl-rgb-v1" or manifest["revision"] != REVISION
            or manifest["license"] != "CC-BY-SA-4.0" or manifest["selection"] != SELECTION
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != 9):
        raise ValueError("Exact pinned nine-image RGB-only manifest required")
    expected = {"stage": "public_tudl_rgb_native_camera_depth_predictions", "status": "pass", "private_truth_read": False,
        "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "scale_fit": False, "network": "none", "actual_MoGe_inference": True, "native_camera_support_verified": True,
        "apply_mask": False, "force_projection": True, "pointmap_geometry_filled": False, "original_frame_coverage_verified": True}
    if not isinstance(receipt, dict) or any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
        raise ValueError("Complete GT-free real RGB-only prediction receipt required")
    if (not re.fullmatch(r"[0-9a-f]{40}", receipt.get("producer_revision", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("image_id", ""))
            or receipt.get("focal_geometry_source_sha256") != GEOMETRY_SHA
            or receipt.get("model_asset", {}).get("sha256") != MODEL_SHA
            or receipt.get("model_source", {}).get("revision") != MODEL_SOURCE
            or receipt.get("public_input_manifest_sha256") != mh
            or receipt.get("public_records") != manifest["images"]
            or any(type(receipt.get(k)) is not int or receipt[k] != v for k, v in
                (("MoGe_calls_completed", 27), ("prior_calls_completed", 18), ("no_prior_calls_completed", 9),
                 ("fixed_prior_calls_completed", 9), ("camera_candidate_calls_completed", 9), ("shared_camera_calls_completed", 9), ("outputs_completed", 9)))):
        raise ValueError("Immutable actual source/image/focal solver required")
    candidates = np.asarray(receipt.get("focal_candidates_pixels")); shared = np.asarray(receipt.get("scene_focal_pixels"))
    if (candidates.shape != (9,) or shared.shape != (3,) or candidates.dtype.kind != "f" or shared.dtype.kind != "f"
            or not np.isfinite(candidates).all() or np.any(candidates <= 0)
            or not np.array_equal(np.median(candidates.reshape(3, 3), axis=1), shared)):
        raise ValueError("Three complete RGB-only scene-median focals required")
    calls, outputs = receipt.get("native_focal_solver_calls"), receipt.get("outputs")
    if not isinstance(calls, list) or len(calls) != 27 or not isinstance(outputs, list) or len(outputs) != 9:
        raise ValueError("All27 actual native calls/all9 outputs required")
    frozen = [(mp, mh), (rp, rh)]; arrays, records = [], []
    for i, ((scene, frame), rgb, output) in enumerate(zip(FRAMES, manifest["images"], outputs)):
        stem = f"scene_{scene:06d}_frame_{frame:06d}"
        if (not isinstance(rgb, dict) or set(rgb) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or type(rgb["scene_id"]) is not int or rgb["scene_id"] != scene
                or type(rgb["frame_id"]) is not int or rgb["frame_id"] != frame
                or rgb["file"] != stem+".png" or type(rgb["width"]) is not int or type(rgb["height"]) is not int
                or (rgb["width"], rgb["height"]) != (WIDTH, HEIGHT) or not isinstance(output, dict)
                or output.get("file") != stem+".npz" or type(output.get("scene_id")) is not int or output["scene_id"] != scene
                or type(output.get("frame_id")) is not int or output["frame_id"] != frame
                or output.get("rgb_sha256") != rgb["sha256"]):
            raise ValueError("Original ordered RGB/output identity differs")
        regular_hash(public/rgb["file"], rgb["sha256"]); regular_hash(pred/output["file"], output.get("sha256"))
        frozen += [(public/rgb["file"], rgb["sha256"]), (pred/output["file"], output["sha256"])]
        with np.load(pred/output["file"], allow_pickle=False) as archive:
            data = {k: archive[k].copy() for k in archive.files}
        validate_arrays(data, scene, frame, float(shared[i//3])); arrays.append(data); records.append(rgb)
    for i, call in enumerate(calls):
        scene, frame = FRAMES[i % 9]; phase = ("fixed_prior", "camera_candidate", "shared_camera")[i//9]
        if (not isinstance(call, dict) or call.get("pass") != phase
                or type(call.get("scene_id")) is not int or call["scene_id"] != scene
                or type(call.get("frame_id")) is not int or call["frame_id"] != frame
                or call.get("file") != f"scene_{scene:06d}_frame_{frame:06d}.png"
                or call.get("focal_prior_supplied") is not (phase != "camera_candidate")
                or call.get("original_solver_returned") is not True
                or type(call.get("native_nearest64_valid_pixels")) is not int
                or not 2 <= call["native_nearest64_valid_pixels"] <= 4096):
            raise ValueError("Native focal support/call order incomplete")
    if ({p.name for p in public.iterdir()} != {"manifest.json", *[r["file"] for r in records]}
            or {p.name for p in pred.iterdir()} != {"report.json", *[r["file"] for r in outputs]}):
        raise ValueError("Public inference scope contains unexpected artifacts")
    return arrays, records, frozen, {"prediction_report_sha256": rh, "public_manifest_sha256": mh}


def sensor_points(depth_m, K, mask):
    """Private sensor camera-Z to points at INTEGER BOP pixels, not MoGe +.5."""
    z, mask, K = np.asarray(depth_m), np.asarray(mask), camera(K)
    if z.ndim != 2 or z.dtype.kind != "f" or mask.shape != z.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Nonempty sensor support and floating depth required")
    y, x = np.nonzero(mask); d = z[mask].astype(np.float64)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("Positive finite sensor camera-Z required")
    return np.column_stack(((x-K[0, 2])*d/K[0, 0], (y-K[1, 2])*d/K[1, 1], d))


def sample_indices(count):
    if count < 1: raise ValueError("Cannot silently drop empty frame support")
    return np.arange(count) if count <= SAMPLES else np.sort(np.random.default_rng(0).choice(count, SAMPLES, replace=False))


def score_frame(data, depth_m, K, visible):
    """Same paired pixel support; no validity-drop improvement can pass the gate."""
    depth_m, visible = np.asarray(depth_m), np.asarray(visible)
    if depth_m.shape != (HEIGHT, WIDTH) or visible.shape != depth_m.shape or visible.dtype != np.bool_:
        raise ValueError("Private original sensor grid/visible masks required")
    support = visible & np.isfinite(depth_m) & (depth_m > 0)
    total = int(support.sum())
    if total < 32: raise ValueError("All nine frames need32 sensor-valid object pixels; no dropping")
    cover = {m: float(np.count_nonzero(support & data[m+"_validity"])/total) for m in ("fixed", "learned")}
    common = support & data["fixed_validity"] & data["learned_validity"]
    indices = sample_indices(int(common.sum())); truth = sensor_points(depth_m, K, common)[indices]
    scores = {}
    for mode in ("fixed", "learned"):
        p = data[mode+"_points"][common][indices].astype(np.float64)
        errors = np.abs(data[mode+"_depth"][common].astype(np.float64)-depth_m[common])
        directed = cKDTree(truth).query(p, workers=1)[0].mean(), cKDTree(p).query(truth, workers=1)[0].mean()
        scores[mode] = {"sensor_visible_camera_chamfer_half_cm": float(.5*sum(directed)*100),
            "pred_to_sensor_cm": float(directed[0]*100), "sensor_to_pred_cm": float(directed[1]*100),
            "sensor_camera_Z_absrel": float(np.mean(errors/depth_m[common])),
            "sensor_camera_Z_mae_cm": float(errors.mean()*100),
            "signed_median_Z_bias_cm": float(np.median(p[:, 2]-truth[:, 2])*100),
            "sensor_valid_object_coverage": cover[mode]}
    return {"fixed": scores["fixed"], "learned": scores["learned"], "sensor_valid_object_pixels": total,
        "paired_common_pixels": int(common.sum()), "samples": len(indices),
        "coverage_gate_pass": bool(min(cover.values()) >= .95 and cover["learned"] >= cover["fixed"]-.01)}


def decision(frames):
    if len(frames) != 9 or [(r["scene_id"], r["frame_id"]) for r in frames] != list(FRAMES):
        raise ValueError("All nine preselected frames must contribute in original order")
    scenes = []
    for i, scene in enumerate((1, 2, 3)):
        rows = frames[3*i:3*i+3]
        scenes.append({"scene_id": scene, **{mode: float(np.mean([r[mode]["sensor_visible_camera_chamfer_half_cm"] for r in rows])) for mode in ("fixed", "learned")}})
    before = np.array([r["fixed"] for r in scenes]); after = np.array([r["learned"] for r in scenes])
    if not np.isfinite(np.r_[before, after]).all() or np.any(before < 0) or np.any(after < 0):
        raise ValueError("Finite nonnegative complete paired scores required")
    gain = (before-after)/before if np.all(before > 0) else None
    coverage = all(r["coverage_gate_pass"] is True for r in frames)
    return {"scenes": scenes, "median_paired_scene_relative_gain": float(np.median(gain)) if gain is not None else None,
        "per_scene_relative_gain": gain.tolist() if gain is not None else None, "coverage_gate_pass": coverage,
        "real_object_camera_hypothesis_supported": bool(gain is not None and coverage and np.median(gain) >= .05 and np.min(gain) >= -.05),
        "adoption_performed": False, "full_v2d_score_verified": False, "human_quality_verified": False}


def read_png(path, *, depth=False):
    from PIL import Image
    with path.open("rb") as f: header = f.read(26)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR" or header[24:26] != (b"\x10\x00" if depth else b"\x08\x00"):
        raise ValueError("Original gray PNG bit depth required, no sensor resampling")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (WIDTH, HEIGHT): raise ValueError("Private original PNG grid differs")
        value = np.asarray(image).copy()
    if depth:
        if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value > 65535): raise ValueError("uint16 sensor PNG required")
        return value.astype(np.float64)
    if value.dtype != np.uint8 or not np.isin(value, [0, 255]).all(): raise ValueError("Binary GT visible mask required")
    return value != 0


def run(root, report):
    base = root/"validation/tudl_rgb_v1"; private = base/"eval_private"
    arrays, records, frozen, hashes = public_predictions(base)
    report.update(**hashes, predictions_frozen_before_private_truth_read=True, frames=[])
    ap = private/"acquisition-report.json"; ah = regular_hash(ap); acquisition = json.loads(ap.read_text())
    if (acquisition.get("status") != "pass" or acquisition.get("dataset_revision") != REVISION
            or acquisition.get("public_manifest_sha256") != hashes["public_manifest_sha256"]
            or acquisition.get("license") != "CC-BY-SA-4.0"
            or acquisition.get("selection_before_private_annotation_values") is not True
            or acquisition.get("private_annotations_exported_as_inference_inputs") is not False):
        raise ValueError("Complete exact pinned external acquisition receipt required")
    retained = acquisition.get("retained_files")
    if not isinstance(retained, list) or not retained: raise ValueError("Private artifact identities required")
    private_hashes = [(ap, ah)]
    retained_names = set()
    for record in retained:
        relative = record.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Private artifact path escapes evaluation root")
        if relative in retained_names: raise ValueError("Duplicate private artifact identity")
        retained_names.add(relative)
        path = private/relative; regular_hash(path, record.get("sha256")); private_hashes.append((path, record["sha256"]))
        if type(record.get("bytes")) is not int or path.stat().st_size != record["bytes"]: raise ValueError("Private artifact size changed")
    selected = acquisition.get("selected_records")
    expected = [{"scene_id": r["scene_id"], "frame_id": r["frame_id"], "file": r["file"], "rgb_sha256": r["sha256"]} for r in records]
    if selected != expected: raise ValueError("Private/public cohort identity differs")
    identified = {str(p.relative_to(private)) for p, _ in private_hashes}
    def identified_file(path):
        if str(path.relative_to(private)) not in identified: raise ValueError("Private consumed file lacks immutable receipt")
        return path
    for (scene, frame), data in zip(FRAMES, arrays):
        directory = private/f"source/test/{scene:06d}"; key = str(frame)
        camera_json = json.loads(identified_file(directory/"scene_camera.json").read_text())
        gt_json = json.loads(identified_file(directory/"scene_gt.json").read_text())
        if key not in camera_json or key not in gt_json or not isinstance(gt_json[key], list) or not gt_json[key]:
            raise ValueError("Private selected camera/object instances missing")
        value = camera_json[key]; K = camera(np.asarray(value.get("cam_K"), dtype=np.float64).reshape(3, 3))
        scale = value.get("depth_scale")
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0: raise ValueError("Per-image positive depth_scale required")
        depth = read_png(identified_file(directory/f"depth/{frame:06d}.png"), depth=True)*float(scale)/1000
        visible = np.zeros((HEIGHT, WIDTH), bool)
        for instance in range(len(gt_json[key])):
            visible |= read_png(identified_file(directory/f"mask_visib/{frame:06d}_{instance:06d}.png"))
        scores = score_frame(data, depth, K, visible)
        scores.update(scene_id=scene, frame_id=frame,
            private_focal_error_diagnostic={mode: float(abs(data[mode+"_K"][0, 0]-K[0, 0])/K[0, 0]) for mode in ("fixed", "learned")},
            object_instances_scored=len(gt_json[key]))
        report["frames"].append(scores)
    report.update(decision=decision(report["frames"]), acquisition_report_sha256=ah, original_frame_coverage_verified=True,
        private_sensor_K_used_for_truth_only=True, private_GT_masks_used_for_evaluation_only=True,
        predicted_XYZ_camera_or_scale_changed=False, alignment_performed=False, oracle_ray_diagnostic_performed=False,
        GT_pixel_convention="integer_BOP", prediction_pixel_convention="MoGe_plus_half; equivalent integer K subtracts.5 principal point; XYZ untouched",
        training_overlap_excluded=False, sensor_noise_free_verified=False, human_quality_verified=False)
    for path, digest in frozen+private_hashes: regular_hash(path, digest)
    report.update(status="pass", phase="complete")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote network-none CPU evaluation")
    root = Path(os.environ["WR_ROOT"]); out = root/"validation/tudl_rgb_v1/quality_v1"
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()): raise FileExistsError("Require exclusively reserved empty quality output")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image): raise ValueError("Frozen producer/image required")
    report = {"stage": "private_tudl_real_object_rgbd_camera_quality", "status": "fail", "phase": "integrity",
        "producer_revision": revision, "image_id": image, "script_sha256": sha256(Path(__file__)), "budget_seconds": BUDGET,
        "challenge_inputs_used": False, "external_private_sensor_truth_used": True, "hand_labeled_test": False,
        "full_v2d_score_verified": False, "adoption_performed": False}
    start = time.perf_counter()
    def expired(*_): raise TimeoutError("Private real RGBD evaluation exceeded90s")
    previous = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: run(root, report)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-start
        with (out/"report.json").open("x") as f: json.dump(report, f, allow_nan=False, indent=2); f.write("\n")
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "decision")}))


if __name__ == "__main__": main()

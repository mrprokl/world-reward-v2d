"""Real TUD-L RGB-only fixed-versus-native-learned camera/depth predictions.

Exactly nine licensed public RGB images enter inference. Three unchanged MoGe
calls per image: fixed size prior, native focal candidate, then clip-median
focal. No private camera, masks, depth, pose, mesh, human model or scale fit.
Valid pointmaps must obey the camera; invalid pixels are preserved, never filled.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import object_synthetic_observations as helper
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

SCHEMA = "world-reward-tudl-rgb-v1"
STAGE = "public_tudl_rgb_native_camera_depth_predictions"
REVISION = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
SELECTION = "sorted_RGB_filenames_first_median_index_n_div_2_last_per_scene_before_private_values"
LICENSE = "CC-BY-SA-4.0"
WIDTH, HEIGHT, BUDGET = 640, 480, 600
FRAME_IDS = ((0, 4074, 8227), (3, 4013, 7710), (4, 4028, 7969))
GEOMETRY_SHA = "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05"
FIXED_K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])


def public_inputs(directory):
    """Require the predeclared RGB-only cohort and no additional exposed files."""
    directory = Path(directory); path = directory / "manifest.json"
    receipt = helper.identity(path); manifest = json.loads(path.read_text())
    expected = {"schema": SCHEMA, "revision": REVISION, "license": LICENSE, "selection": SELECTION}
    if (not isinstance(manifest, dict) or set(manifest) != {*expected, "images"}
            or any(manifest.get(k) != v for k, v in expected.items())
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != 9):
        raise ValueError("Require exact licensed RGB-only TUD-L manifest")
    records = []
    for index, record in enumerate(manifest["images"]):
        scene = index // 3 + 1; frame = FRAME_IDS[scene - 1][index % 3]
        filename = f"scene_{scene:06d}_frame_{frame:06d}.png"
        if (not isinstance(record, dict) or set(record) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(record.get(k)) is not int for k in ("scene_id", "frame_id", "width", "height"))
                or (record["scene_id"], record["frame_id"], record["width"], record["height"]) != (scene, frame, WIDTH, HEIGHT)
                or record["file"] != filename or not isinstance(record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            raise ValueError("Public RGB order/grid/identity differs; private fields forbidden")
        image = directory / filename; identity = helper.identity(image)
        if identity["bytes"] <= 0 or identity["sha256"] != record["sha256"]:
            raise ValueError("Frozen RGB SHA/size differs")
        records.append({**record, "path": image})
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[r["file"] for r in records]}:
        raise ValueError("Only the nine public RGBs and manifest may be exposed")
    return records, receipt


def camera_candidate(normalized):
    """Convert actual normalized intrinsics to pixel focal; no focal clipping."""
    value = np.asarray(normalized)
    if np.ma.isMaskedArray(normalized) or value.shape != (3, 3) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError("Native camera must be finite floating normalized3x3")
    K = np.diag([WIDTH, HEIGHT, 1.]) @ value
    focal = float(K[0, 0]); result = np.array([[focal, 0., WIDTH/2], [0., focal, HEIGHT/2], [0., 0., 1.]])
    if focal <= 0 or not np.allclose(K, result, rtol=1e-6, atol=1e-3):
        raise ValueError("Require native positive centered square-pixel intrinsics")
    return result


def scene_cameras(focals):
    value = np.asarray(focals)
    if (np.ma.isMaskedArray(focals) or value.shape != (9,) or value.dtype.kind != "f"
            or not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError("Require all nine positive native pixel-focal candidates")
    cameras = np.repeat(FIXED_K[None], 3, axis=0)
    cameras[:, 0, 0] = cameras[:, 1, 1] = np.median(value.reshape(3, 3), axis=1)
    return cameras


def checked_infer(torch, module, network, tensor, fov, diagnostics, context):
    """Observe native support, rejecting only its documented <2-pixel fallback.

    Forward the exact original signature/result unchanged and restore the
    process-local hook after every call, including exceptions. No optimizer
    tolerances or sampled masks are changed.
    """
    original = module.recover_focal_shift; before = len(diagnostics)
    def checked(points, mask=None, focal=None, downsample_size=(64, 64)):
        if (downsample_size != (64, 64) or not torch.is_tensor(mask) or mask.dtype != torch.bool
                or tuple(mask.shape) != (1, HEIGHT, WIDTH) or tuple(points.shape) != (1, HEIGHT, WIDTH, 3)):
            raise RuntimeError("Native focal solver boolean mask/grid/sampling ABI differs")
        sampled = torch.nn.functional.interpolate(mask.float().unsqueeze(1), (64, 64), mode="nearest").squeeze(1) > 0
        count = int(sampled.sum().item())
        entry = {**context, "native_nearest64_valid_pixels": count, "focal_prior_supplied": focal is not None,
                 "original_solver_returned": False}; diagnostics.append(entry)
        if count < 2:
            raise ValueError("Native focal solver default-camera fallback rejected: fewer than two sampled pixels")
        result = original(points, mask, focal=focal, downsample_size=downsample_size)
        entry["original_solver_returned"] = True
        return result
    module.recover_focal_shift = checked
    try:
        with torch.inference_mode():
            result = network.infer(tensor[None], fov_x=fov, apply_mask=False, force_projection=True)
        if len(diagnostics) != before + 1:
            raise RuntimeError("Each native inference must call its verified focal solver exactly once")
        return result
    finally:
        module.recover_focal_shift = original


def camera_arrays(depth, points, validity, normalized, K):
    if (depth.shape != (HEIGHT, WIDTH) or points.shape != (HEIGHT, WIDTH, 3) or validity.shape != (HEIGHT, WIDTH)
            or depth.dtype != np.float32 or points.dtype != np.float32 or validity.dtype != np.bool_):
        raise ValueError("Require original float32 depth/XYZ and boolean full-grid validity")
    checks = validate_camera_pointmap(depth, points, validity, normalized, K)
    return {"points": points, "depth": depth, "validity": validity, "K": K.copy()}, checks


def prediction_arrays(fixed, learned, record):
    return {**{"fixed_" + k: v for k, v in fixed.items()}, **{"learned_" + k: v for k, v in learned.items()},
            "scene_id": np.array(record["scene_id"], np.int64), "frame_id": np.array(record["frame_id"], np.int64)}


def validate_prediction_arrays(data, scene_id, frame_id, learned_K):
    """CPU-only frozen NPZ contract, not validation of camera/depth accuracy.

    Expected scene camera comes from the public inference receipt, never a
    private calibration. Invalid pixels remain excluded and are not repaired.
    """
    keys = {p + k for p in ("fixed_", "learned_") for k in ("points", "depth", "validity", "K")}
    if (set(data) != keys | {"scene_id", "frame_id"} or type(scene_id) is not int
            or type(frame_id) is not int or scene_id not in (1, 2, 3) or frame_id not in FRAME_IDS[scene_id - 1]):
        raise ValueError("Require exact predeclared scene/frame and prediction array keys")
    for name, expected in (("scene_id", scene_id), ("frame_id", frame_id)):
        value = np.asarray(data[name])
        if np.ma.isMaskedArray(data[name]) or value.shape != () or value.dtype != np.int64 or int(value) != expected:
            raise ValueError("Require exact integer scalar scene/frame identities")
    value = np.asarray(learned_K)
    if np.ma.isMaskedArray(learned_K) or value.shape != (3, 3) or value.dtype.kind != "f":
        raise ValueError("Require the public finite floating scene camera")
    expected_camera = camera_candidate(np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ value)
    diagnostics = {}
    for mode, expected in (("fixed", FIXED_K), ("learned", expected_camera)):
        K = np.asarray(data[mode + "_K"])
        if (np.ma.isMaskedArray(data[mode + "_K"]) or K.shape != (3, 3) or K.dtype not in (np.float32, np.float64)
                or not np.isfinite(K).all()):
            raise ValueError("Require finite floating pixel-camera matrix")
        _, diagnostics[mode] = camera_arrays(data[mode + "_depth"], data[mode + "_points"], data[mode + "_validity"],
                                            np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ K, expected)
    return diagnostics


def run(root, report, persist):
    directory = root / "validation/tudl_rgb_v1/inputs"
    records, receipt = public_inputs(directory)
    report.update(input_manifest=receipt, public_input_manifest_sha256=receipt["sha256"],
                  public_records=[{k: v for k, v in r.items() if k != "path"} for r in records], phase="model_load"); persist()
    model_path, acquisition, asset = helper.model_asset(root)
    report.update(acquisition_report=acquisition, model_asset=asset, model_revision=helper.MODEL_REVISION); persist()
    import torch
    import moge
    from moge.model import v2 as module
    from moge.utils import geometry_torch
    from PIL import Image
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no altered CPU/local fallback")
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    report["model_source"] = helper.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    if helper.identity(Path(geometry_torch.__file__))["sha256"] != GEOMETRY_SHA or module.recover_focal_shift is not geometry_torch.recover_focal_shift:
        raise ValueError("Native focal recovery source/alias differs from audited source")
    report.update(focal_geometry_source_sha256=GEOMETRY_SHA, torch=torch.__version__); persist()
    network = module.MoGeModel.from_pretrained(str(model_path)).cuda().eval()
    fixed, decoded_hashes, focals = [], [], []
    def infer(record, phase, K=None):
        report.update(phase=phase, active_file=record["file"]); persist()
        with Image.open(record["path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                raise ValueError("Require actual original640x480 RGB PNG")
            rgb = np.asarray(image).copy()
        decoded = hashlib.sha256(rgb.tobytes()).hexdigest()
        index = records.index(record)
        if phase != "fixed_prior" and decoded != decoded_hashes[index]:
            raise ValueError("All three calls must consume identical decoded RGB")
        fov = None if K is None else float(np.degrees(2*np.arctan(WIDTH/(2*K[0, 0]))))
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        result = checked_infer(torch, module, network, tensor, fov, report["native_focal_solver_calls"],
                               {"file": record["file"], "scene_id": record["scene_id"], "frame_id": record["frame_id"], "pass": phase})
        torch.cuda.synchronize()
        depth, points, valid, normalized = [result[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics")]
        if K is None: K = camera_candidate(normalized)
        arrays, checks = camera_arrays(depth, points, valid, normalized, K)
        report["MoGe_calls_completed"] += 1
        report["no_prior_calls_completed" if phase == "camera_candidate" else "prior_calls_completed"] += 1
        report[phase + "_calls_completed"] += 1
        persist(); del result, tensor
        return arrays, checks, decoded
    for record in records:
        arrays, checks, decoded = infer(record, "fixed_prior", FIXED_K)
        fixed.append((arrays, checks)); decoded_hashes.append(decoded)
    for record in records:
        arrays, checks, decoded = infer(record, "camera_candidate")
        focal = float(arrays["K"][0, 0]); focals.append(focal)
        report["camera_candidates"].append({"file": record["file"], "scene_id": record["scene_id"], "frame_id": record["frame_id"],
                                            "focal_pixels": focal, "decoded_RGB_sha256": decoded, "pointmap_checks": checks})
        persist()
    cameras = scene_cameras(np.asarray(focals, np.float64))
    report.update(focal_candidates_pixels=focals, scene_focal_pixels=cameras[:, 0, 0].tolist(), scene_camera_K=cameras.tolist(),
                  focal_aggregation="median_all_three_frames_per_scene_no_clamp", native_camera_support_verified=True); persist()
    for index, record in enumerate(records):
        learned, checks, decoded = infer(record, "shared_camera", cameras[record["scene_id"] - 1])
        arrays = prediction_arrays(fixed[index][0], learned, record)
        target = root / "validation/tudl_rgb_v1/predictions_v1" / (Path(record["file"]).stem + ".npz")
        with target.open("xb") as handle: np.savez_compressed(handle, **arrays)
        report["outputs"].append({"scene_id": record["scene_id"], "frame_id": record["frame_id"], "file": target.name,
            "sha256": sha256(target), "rgb_sha256": record["sha256"], "decoded_RGB_sha256": decoded,
            "fixed_pointmap_checks": fixed[index][1], "learned_pointmap_checks": checks,
            "arrays": {k: {"sha256": hashlib.sha256(v.tobytes()).hexdigest(), "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()}})
        persist()
    if (public_inputs(directory) != (records, receipt) or helper.model_asset(root)[1:] != (acquisition, asset)
            or report["MoGe_calls_completed"] != 27 or report["prior_calls_completed"] != 18
            or report["no_prior_calls_completed"] != 9 or len(report["outputs"]) != 9
            or any(report[p + "_calls_completed"] != 9 for p in ("fixed_prior", "camera_candidate", "shared_camera"))):
        raise ValueError("Complete original RGB/asset/call coverage required")
    expected_calls = [(p, r["file"], r["scene_id"], r["frame_id"]) for p in ("fixed_prior", "camera_candidate", "shared_camera") for r in records]
    calls = report["native_focal_solver_calls"]
    if ([(r["pass"], r["file"], r["scene_id"], r["frame_id"]) for r in calls] != expected_calls
            or any(r["original_solver_returned"] is not True or type(r["native_nearest64_valid_pixels"]) is not int
                   or r["native_nearest64_valid_pixels"] < 2
                   or r["focal_prior_supplied"] is not (r["pass"] != "camera_candidate") for r in calls)):
        raise ValueError("Require all27 unchanged native focal calls in their declared order")
    report.update(status="pass", phase="complete", original_frame_coverage_verified=True, actual_MoGe_inference=True,
                  outputs_completed=9); report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require isolated remote Linux GPU inference")
    root = Path(os.environ["WR_ROOT"]); output = root / "validation/tudl_rgb_v1/predictions_v1"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved empty predictions")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable inference source/image required")
    with (output / "report.json").open("x") as handle:
        started = time.perf_counter()
        report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
            "script_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(helper.__file__)), "budget_seconds": BUDGET,
            "network": "none", "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
            "hand_labeled_test": False, "oracle_modes": [], "scale_fit": False, "adoption_performed": False,
            "metric_scale_accuracy_verified": False, "calibration_accuracy_verified": False, "apply_mask": False,
            "force_projection": True, "pointmap_geometry_filled": False, "fixed_camera_K": FIXED_K.tolist(),
            "outputs": [], "camera_candidates": [], "native_focal_solver_calls": [], "MoGe_calls_completed": 0,
            "prior_calls_completed": 0, "no_prior_calls_completed": 0, "fixed_prior_calls_completed": 0,
            "camera_candidate_calls_completed": 0, "shared_camera_calls_completed": 0}
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False) + "\n"); handle.truncate(); handle.flush()
        def expired(*_): raise TimeoutError("Whole RGB-only TUD-L inference exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, report, persist); persist()
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "MoGe_calls_completed", "outputs_completed", "elapsed_seconds")}))


if __name__ == "__main__": main()

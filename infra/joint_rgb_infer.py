"""Public RGB-only predicted-human grounding of MoGe2, with fixed camera prior.

Nine images/masks are inference inputs. Neither renderer truth nor calibration
is accessible. Shared alpha aligns two predictions per clip, not metric truth;
human meshes stay unchanged and invalid pointmap values are never filled.
"""
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
import hand_synthetic_infer as human
import object_synthetic_observations as depth_model
from camera_render import raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.metric_alignment import fit_shared_depth_scale
from world_reward.pointmap import validate_camera_pointmap

STAGE = "public_joint_rgb_shared_grounding_predictions"
WIDTH, HEIGHT, CLIPS, FRAMES, BUDGET = 1024, 768, 3, 3, 300
CAMERA_K = np.array([[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]])


def public_inputs(root):
    base = root/"validation/joint_rgb_v1"; inputs, masks = base/"inputs", base/"automatic_masks"
    mp, rp = inputs/"manifest.json", masks/"report.json"
    hashes = {"public_inputs_sha": depth_model.identity(mp)["sha256"], "mask_report_sha": depth_model.identity(rp)["sha256"]}
    manifest, report = json.loads(mp.read_text()), json.loads(rp.read_text())
    expected = {"stage": "public_joint_rgb_automatic_masks", "status": "pass", "private_truth_read": False,
                "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False,
                "oracle_modes": [], "frames": CLIPS*FRAMES}
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != "world-reward-joint-rgb-v1"
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != CLIPS*FRAMES
            or not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items())
            or report.get("input_manifest_sha256") != hashes["public_inputs_sha"]
            or not isinstance(report.get("records"), list) or len(report["records"]) != CLIPS*FRAMES):
        raise ValueError("Require complete RGB-only joint manifest and automatic masks")
    records = []
    for index, (image, mask) in enumerate(zip(manifest["images"], report["records"])):
        clip, frame = divmod(index, FRAMES); filename = f"clip_{clip:02d}_frame_{frame:03d}.png"; stem = Path(filename).stem
        if (not isinstance(image, dict) or set(image) != {"file", "sha256", "width", "height"}
                or image["file"] != filename or type(image["width"]) is not int or type(image["height"]) is not int
                or (image["width"], image["height"]) != (WIDTH, HEIGHT) or not isinstance(mask, dict)
                or mask.get("file") != filename or mask.get("rgb_sha256") != image["sha256"]
                or type(mask.get("clip_index")) is not int or mask["clip_index"] != clip
                or type(mask.get("frame_index")) is not int or mask["frame_index"] != frame):
            raise ValueError("Public image/mask order/grid mismatch")
        record = {"clip_index": clip, "frame_index": frame, "file": filename, "rgb_sha256": image["sha256"], "image_path": inputs/filename}
        for kind in ("human", "object"):
            name = f"{stem}_{kind}.png"
            if mask.get(kind+"_mask_file") != name: raise ValueError("Mask filename is not an exact public route")
            record[kind+"_mask_path"] = masks/name; record[kind+"_mask_sha256"] = mask.get(kind+"_mask_sha256")
        for key, digest in (("image_path", record["rgb_sha256"]), ("human_mask_path", record["human_mask_sha256"]), ("object_mask_path", record["object_mask_sha256"])):
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest) or depth_model.identity(record[key])["sha256"] != digest:
                raise ValueError("Public RGB/mask artifact SHA mismatch")
        records.append(record)
    if ({p.name for p in inputs.iterdir()} != {"manifest.json", *[r["file"] for r in records]}
            or {p.name for p in masks.iterdir()} != {"report.json", *[r[k].name for r in records for k in ("human_mask_path", "object_mask_path")]}):
        raise ValueError("Only public RGB/masks may be exposed")
    return records, hashes


def read_mask(path, Image):
    with Image.open(path) as png:
        if png.format != "PNG" or png.mode != "L" or png.size != (WIDTH, HEIGHT): raise ValueError("Mask must be original-grid uint8 L PNG")
        mask = np.asarray(png).copy()
    if not np.isin(mask, [0, 255]).all() or not mask.any(): raise ValueError("Automatic binary mask empty or invalid; no fallback")
    return mask


def pointmap_contract(depth, points, validity, normalized):
    if (depth.shape != (HEIGHT, WIDTH) or points.shape != (HEIGHT, WIDTH, 3) or validity.shape != (HEIGHT, WIDTH)
            or depth.dtype != np.float32 or points.dtype != np.float32 or validity.dtype != np.bool_):
        raise ValueError("Require original float32 MoGe depth/XYZ and boolean validity")
    return validate_camera_pointmap(depth, points, validity, normalized, CAMERA_K)


def align_clips(raw_points, raw_depths, rendered_depths, render_masks, human_masks, object_masks, validity):
    """One alpha per three-frame clip, XYZ scaled together exactly once."""
    expected = (CLIPS*FRAMES, HEIGHT, WIDTH)
    if (raw_points.shape != (*expected, 3) or raw_points.dtype != np.float32 or raw_depths.shape != expected or raw_depths.dtype != np.float32
            or rendered_depths.shape != expected or rendered_depths.dtype != np.float32
            or any(a.shape != expected or a.dtype != np.bool_ for a in (render_masks, human_masks, object_masks, validity))):
        raise ValueError("Require nine original grids; explicit predicted/automatic visibility")
    aligned = np.empty_like(raw_points); scales = []; diagnostics = []
    for clip in range(CLIPS):
        start = clip*FRAMES; positions = slice(start, start+FRAMES)
        visible = render_masks[positions] & human_masks[positions] & ~object_masks[positions] & validity[positions]
        fit = fit_shared_depth_scale(raw_depths[positions], rendered_depths[positions], visible, list(range(FRAMES)))
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            aligned[positions] = raw_points[positions]*fit.shared_scale
        if not np.isfinite(aligned[positions][validity[positions]]).all() or (aligned[positions][..., 2][validity[positions]] <= 0).any():
            raise ValueError("Aligned valid XYZ overflowed or became nonpositive; no repair")
        scales.append(fit.shared_scale); diagnostics.append({"clip_index": clip, **fit.to_dict()})
    return aligned, np.asarray(scales, np.float64), diagnostics


def run_inference(root, report, path):
    records, hashes = public_inputs(root); report.update(**hashes, public_records=[{k: v for k, v in r.items() if not k.endswith("_path")} for r in records])
    asset_path, acquisition, asset = depth_model.model_asset(root)
    report.update(acquisition_report=acquisition, MoGe_model_asset=asset, MoGe_model_revision=depth_model.MODEL_REVISION, phase="model_load"); human._write(path, report)
    import torch
    import moge
    from moge.model.v2 import MoGeModel
    from PIL import Image
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    report["MoGe_source"] = depth_model.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    model, estimator, faces, source = human.load_model(root, torch); report.update(body_model=source, torch=torch.__version__)
    depth_net = MoGeModel.from_pretrained(str(asset_path)).cuda().eval()
    camera = torch.from_numpy(CAMERA_K.astype(np.float32))[None].cuda()
    fov = float(np.degrees(2*np.arctan(WIDTH/(2*CAMERA_K[0, 0]))))
    vertices = []; points_all = []; depths = []; validities = []; human_masks = []; object_masks = []; silhouettes = []; renders = []
    for record in records:
        report.update(phase="inference", active_file=record["file"]); human._write(path, report)
        with Image.open(record["image_path"]) as png:
            if png.format != "PNG" or png.mode != "RGB" or png.size != (WIDTH, HEIGHT): raise ValueError("Require original RGB PNG")
            rgb = np.asarray(png).copy()
        human_mask = read_mask(record["human_mask_path"], Image); object_mask = read_mask(record["object_mask_path"], Image)
        box, modelmask = human.derived_bbox(rgb, human_mask)
        with torch.inference_mode():
            prediction = estimator.process_one_image(img=rgb, bboxes=box[None], masks=modelmask, cam_int=camera, inference_type="body")
        if not isinstance(prediction, list) or len(prediction) != 1: raise RuntimeError("Exactly one Body prediction required")
        body_values, errors = human.decode_prediction(torch, model, prediction[0], "body")
        if not np.isclose(float(body_values["focal_length"]), CAMERA_K[0, 0], rtol=1e-6, atol=1e-4): raise RuntimeError("Body ignored explicit fixed camera prior")
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        with torch.inference_mode(): predicted = depth_net.infer(tensor[None], fov_x=fov, apply_mask=False)
        torch.cuda.synchronize()
        depth, points, valid, normalized = [predicted[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics")]
        checks = pointmap_contract(depth, points, valid, normalized)
        silhouette, rendered = raster_camera_mesh(body_values["vertices_camera_m"], faces, CAMERA_K, WIDTH, HEIGHT)
        silhouette, rendered = silhouette.cpu().numpy(), rendered.cpu().numpy()
        vertices.append(body_values["vertices_camera_m"]); points_all.append(points); depths.append(depth); validities.append(valid)
        human_masks.append(human_mask > 0); object_masks.append(object_mask > 0); silhouettes.append(silhouette); renders.append(rendered)
        report["frames"].append({"clip_index": record["clip_index"], "frame_index": record["frame_index"], "file": record["file"],
            "decoded_RGB_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "bbox_xyxy": box.tolist(), "native_forward_errors": errors,
            "pointmap_checks": checks, "human_silhouette_iou": silhouette_iou(silhouette, human_mask > 0),
            "alignment_pixels": int(np.count_nonzero(silhouette & (human_mask > 0) & ~(object_mask > 0) & valid))})
        report.update(body_calls_completed=len(vertices), MoGe_calls_completed=len(points_all)); human._write(path, report)
        del predicted, tensor, prediction
    raw_points, raw_depths, validity = np.stack(points_all), np.stack(depths), np.stack(validities)
    aligned, scales, diagnostics = align_clips(raw_points, raw_depths, np.stack(renders), np.stack(silhouettes), np.stack(human_masks), np.stack(object_masks), validity)
    arrays = {"raw_points": raw_points, "aligned_points": aligned, "human_vertices_camera_m": np.stack(vertices), "human_faces": faces,
        "object_masks": np.stack(object_masks), "moge_validity": validity, "shared_scale": scales,
        "frame_index": np.tile(np.arange(FRAMES, dtype=np.int64), CLIPS), "clip_index": np.repeat(np.arange(CLIPS, dtype=np.int64), FRAMES), "camera_K": CAMERA_K.copy()}
    artifact = path.parent/"arrays.npz"
    with artifact.open("xb") as stream: np.savez_compressed(stream, **arrays)
    report.update(arrays_sha256=sha256(artifact), shared_scale=scales.tolist(), per_clip_scale_diagnostics=diagnostics,
        array_identities={k: {"sha256": hashlib.sha256(v.tobytes()).hexdigest(), "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()})
    if public_inputs(root) != (records, hashes) or depth_model.model_asset(root)[1:] != (acquisition, asset): raise ValueError("Frozen public inputs/assets changed")
    report.update(status="pass", phase="complete", actual_body_inference=True, actual_MoGe_inference=True, actual_predicted_human_render=True); report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require remote Linux CUDA network-none")
    root = Path(os.environ["WR_ROOT"]); output = root/"validation/joint_rgb_v1/predictions"; path = output/"report.json"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved empty predictions directory")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image): raise ValueError("Require immutable inference source/image")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "body_loader_sha256": sha256(Path(human.__file__)), "body_helper_sha256": sha256(Path(human.body.__file__)),
        "MoGe_helper_sha256": sha256(Path(depth_model.__file__)), "budget_seconds": BUDGET, "private_truth_read": False,
        "render_helper_sha256": sha256(Path(__file__).with_name("camera_render.py")),
        "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "calibration_accuracy_verified": False, "metric_scale_accuracy_verified": False, "camera_prior": CAMERA_K.tolist(), "focal_fitted": False,
        "alpha_source": "rendered_predicted_human_not_ground_truth", "alpha_scope": "one_scalar_per_three_frame_clip_XYZ_once",
        "human_geometry_scaled": False, "pointmap_geometry_filled": False, "apply_mask": False, "body_inference_type": "body",
        "prompt_mode": "automatic_human_mask_and_derived_bbox", "frames": [], "body_calls_completed": 0, "MoGe_calls_completed": 0}
    start = time.perf_counter()
    def expired(*args): raise TimeoutError("Whole J1 inference exceeded300s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: human._write(path, report); run_inference(root, report, path)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-start; human._write(path, report)
    print(json.dumps({k: report[k] for k in ("stage", "status", "shared_scale", "elapsed_seconds")}))


if __name__ == "__main__": main()

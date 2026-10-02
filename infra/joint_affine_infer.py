"""J3 public RGB-only Body/MoGe observations, without affine fitting or GT.

Every original frame is independently inferred once with K1280, then the fresh
predicted human is rendered in that same camera. Invalid MoGe pixels remain
unchanged with their explicit validity; no scale, offset, mesh or camera fit.
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
from camera_render import raster_camera_mesh
from joint_rgb_infer import checked_depth_infer, GEOMETRY_SHA
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap


STAGE = "public_joint_affine_rgb_predictions"
SCHEMA = "world-reward-joint-affine-rgb-v1"
WIDTH, HEIGHT, CLIPS, FRAMES, BUDGET = 1024, 768, 3, 6, 600
VERTICES, FACES = 18439, 36874
CAMERA_K = np.array([[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]], np.float64)
ARRAY_KEYS = {"raw_depth", "raw_points", "validity", "silhouette", "rendered_depth", "human_mask",
              "object_mask", "human_vertices_camera_m", "human_faces", "K", "clip_index", "frame_index"}


def helper_identities():
    """Path-independent hashes shared by the producer and public consumers."""
    helpers = {"body_loader": Path(human.__file__), "body_helper": Path(human.body.__file__),
               "MoGe_helper": Path(depth_model.__file__), "camera_render": Path(__file__).with_name("camera_render.py"),
               "focal_solver_instrumentation": Path(__file__).with_name("joint_rgb_infer.py")}
    return {key: sha256(value) for key, value in helpers.items()}


def public_inputs(root):
    base = Path(root) / "validation/joint_affine_rgb_v1"
    inputs, masks = base / "inputs", base / "automatic_masks"
    for folder in (inputs, masks):
        if folder.resolve() != folder.absolute() or not folder.is_dir():
            raise ValueError("Public folders must be canonical nonsymlink directories")
    mp, rp = inputs / "manifest.json", masks / "report.json"
    identities = {"public_inputs": depth_model.identity(mp), "mask_report": depth_model.identity(rp)}
    manifest, report = json.loads(mp.read_text()), json.loads(rp.read_text())
    expected = {"stage": "public_joint_affine_rgb_automatic_masks", "status": "pass", "frames": CLIPS*FRAMES,
                "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
                "hand_labeled_test": False, "oracle_modes": [], "human_query": "person.", "object_query": "bottle."}
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != CLIPS*FRAMES
            or not isinstance(report, dict)
            or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items())
            or report.get("input_manifest_sha256") != identities["public_inputs"]["sha256"]
            or not isinstance(report.get("records"), list) or len(report["records"]) != CLIPS*FRAMES):
        raise ValueError("Require complete public eighteen-RGB manifest and automatic-mask provenance")
    records = []
    for index, (image, mask) in enumerate(zip(manifest["images"], report["records"])):
        clip, frame = divmod(index, FRAMES)
        filename = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (not isinstance(image, dict) or set(image) != {"file", "sha256", "width", "height"}
                or image.get("file") != filename or type(image.get("width")) is not int
                or type(image.get("height")) is not int or (image["width"], image["height"]) != (WIDTH, HEIGHT)
                or not isinstance(mask, dict) or mask.get("file") != filename
                or mask.get("rgb_sha256") != image.get("sha256")
                or type(mask.get("clip_index")) is not int or mask["clip_index"] != clip
                or type(mask.get("frame_index")) is not int or mask["frame_index"] != frame):
            raise ValueError("Public image/mask exact order and original-grid identity mismatch")
        record = {"clip_index": clip, "frame_index": frame, "file": filename,
                  "rgb_sha256": image["sha256"], "image_path": inputs / filename}
        for kind in ("human", "object"):
            name = f"{Path(filename).stem}_{kind}.png"
            if mask.get(kind + "_mask_file") != name:
                raise ValueError("Automatic mask must use the exact public filename")
            record[kind + "_mask_path"] = masks / name
            record[kind + "_mask_sha256"] = mask.get(kind + "_mask_sha256")
        for key, digest in (("image_path", record["rgb_sha256"]),
                            ("human_mask_path", record["human_mask_sha256"]),
                            ("object_mask_path", record["object_mask_sha256"])):
            if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                    or depth_model.identity(record[key])["sha256"] != digest):
                raise ValueError("Frozen public RGB/mask SHA mismatch")
        records.append(record)
    if ({p.name for p in inputs.iterdir()} != {"manifest.json", *[r["file"] for r in records]}
            or {p.name for p in masks.iterdir()} != {"report.json", *[r[k].name for r in records
                for k in ("human_mask_path", "object_mask_path")]}):
        raise ValueError("Only exact public RGB and automatic masks may be exposed")
    return records, {"public_inputs_sha": identities["public_inputs"]["sha256"],
                     "mask_report_sha": identities["mask_report"]["sha256"], "receipts": identities}


def read_mask(path, Image):
    with Image.open(path) as png:
        if png.format != "PNG" or png.mode != "L" or png.size != (WIDTH, HEIGHT):
            raise ValueError("Automatic masks must be original-grid binary L PNG")
        mask = np.asarray(png).copy()
    if not np.isin(mask, [0, 255]).all() or not mask.any():
        raise ValueError("Automatic mask absent/invalid; no fallback")
    return mask


def validate_arrays(arrays, normalized):
    """Preserve source pixels/validity; validate exact per-frame output ABI."""
    if not isinstance(arrays, dict) or set(arrays) != ARRAY_KEYS:
        raise ValueError("Exact J3 per-frame arrays required, without GT or fitted fields")
    shapes = {"raw_depth": (HEIGHT, WIDTH), "raw_points": (HEIGHT, WIDTH, 3),
              "rendered_depth": (HEIGHT, WIDTH), "human_vertices_camera_m": (VERTICES, 3)}
    for key, shape in shapes.items():
        value = arrays[key]
        if np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.float32:
            raise ValueError("Original float32 array/grid required: " + key)
    for key in ("validity", "silhouette", "human_mask", "object_mask"):
        value = arrays[key]
        if (np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray) or value.dtype != np.bool_
                or value.shape != (HEIGHT, WIDTH) or not value.any()):
            raise ValueError("Nonempty explicit boolean original-grid mask required: " + key)
    checks = validate_camera_pointmap(arrays["raw_depth"], arrays["raw_points"], arrays["validity"], normalized, CAMERA_K)
    silhouette, rendered = arrays["silhouette"], arrays["rendered_depth"]
    if (not np.isfinite(rendered[silhouette]).all() or np.any(rendered[silhouette] <= 0)
            or not np.isnan(rendered[~silhouette]).all()):
        raise ValueError("Rendered predicted-human camera Z must be positive, NaN exactly outside silhouette")
    vertices, faces, K = arrays["human_vertices_camera_m"], arrays["human_faces"], arrays["K"]
    if not np.isfinite(vertices).all() or np.any(vertices[:, 2] <= 0):
        raise ValueError("Fresh predicted-human vertices require finite positive camera Z")
    if (not isinstance(faces, np.ndarray) or faces.dtype != np.int64 or faces.shape != (FACES, 3)
            or faces.min() < 0 or faces.max() >= VERTICES
            or np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 0] == faces[:, 2])
            or np.any(faces[:, 1] == faces[:, 2])):
        raise ValueError("Native int64 full human topology required")
    if (not isinstance(K, np.ndarray) or K.dtype != np.float64 or K.shape != (3, 3)
            or not np.array_equal(K, CAMERA_K)):
        raise ValueError("Immutable original-grid K1280 required")
    for key, upper in (("clip_index", CLIPS), ("frame_index", FRAMES)):
        value = arrays[key]
        if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != np.int64 or not 0 <= int(value) < upper:
            raise ValueError("Original int64 scalar clip/frame index required")
    return checks


def validate_observation(data, clip, frame):
    """Validate an already frozen public NPZ; actual native-K proof is in producer."""
    if type(clip) is not int or type(frame) is not int or not 0 <= clip < CLIPS or not 0 <= frame < FRAMES:
        raise ValueError("Expected original integer clip/frame required")
    arrays = dict(data)
    normalized = np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ CAMERA_K
    checks = validate_arrays(arrays, normalized)
    if int(arrays["clip_index"]) != clip or int(arrays["frame_index"]) != frame:
        raise ValueError("Frozen observation clip/frame order mismatch")
    return checks


def run_inference(root, report, path):
    records, hashes = public_inputs(root)
    report.update(public_inputs_sha=hashes["public_inputs_sha"], mask_report_sha=hashes["mask_report_sha"],
                  public_receipts=hashes["receipts"], public_records=[{k: v for k, v in r.items() if not k.endswith("_path")} for r in records])
    asset_path, acquisition, asset = depth_model.model_asset(root)
    report.update(acquisition_report=acquisition, MoGe_model_asset=asset, MoGe_model_revision=depth_model.MODEL_REVISION,
                  phase="model_load")
    human._write(path, report)
    import torch
    import moge
    from moge.model import v2 as depth_module
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/local fallback")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    distribution = metadata.distribution("moge")
    direct = distribution.read_text("direct_url.json")
    moge_source = depth_model.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    if (depth_model.identity(Path(geometry_torch.__file__))["sha256"] != GEOMETRY_SHA
            or depth_module.recover_focal_shift is not geometry_torch.recover_focal_shift):
        raise RuntimeError("Exact native focal-solver source required")
    model, estimator, faces, body_source = human.load_model(root, torch)
    report.update(body_model=body_source, MoGe_source=moge_source, torch=torch.__version__, focal_geometry_source_sha256=GEOMETRY_SHA)
    net = MoGeModel.from_pretrained(str(asset_path)).cuda().eval()
    camera = torch.from_numpy(CAMERA_K.astype(np.float32))[None].cuda()
    fov = float(np.degrees(2*np.arctan(WIDTH/(2*CAMERA_K[0, 0]))))
    for record in records:
        report.update(phase="inference", active_file=record["file"]); human._write(path, report)
        with Image.open(record["image_path"]) as png:
            if png.format != "PNG" or png.mode != "RGB" or png.size != (WIDTH, HEIGHT):
                raise ValueError("Original public RGB PNG required")
            rgb = np.asarray(png).copy()
        human_mask, object_mask = (read_mask(record[k + "_mask_path"], Image) for k in ("human", "object"))
        box, model_mask = human.derived_bbox(rgb, human_mask)
        with torch.inference_mode():
            predictions = estimator.process_one_image(img=rgb, bboxes=box[None], masks=model_mask,
                                                       cam_int=camera, inference_type="body")
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise RuntimeError("Exactly one automatic-person Body prediction required")
        values, errors = human.decode_prediction(torch, model, predictions[0], "body")
        if not np.isclose(float(values["focal_length"]), CAMERA_K[0, 0], rtol=1e-6, atol=1e-4):
            raise RuntimeError("Body ignored explicit K1280")
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        context = {k: record[k] for k in ("file", "clip_index", "frame_index")}
        predicted = checked_depth_infer(torch, depth_module, net, tensor, fov, report["native_focal_solver_calls"], context)
        depth, points, valid, normalized = [predicted[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics")]
        silhouette, rendered = raster_camera_mesh(values["vertices_camera_m"], faces, CAMERA_K, WIDTH, HEIGHT)
        arrays = {"raw_depth": depth, "raw_points": points, "validity": valid,
                  "silhouette": silhouette.cpu().numpy(), "rendered_depth": rendered.cpu().numpy(),
                  "human_mask": human_mask > 0, "object_mask": object_mask > 0,
                  "human_vertices_camera_m": values["vertices_camera_m"], "human_faces": faces,
                  "K": CAMERA_K.copy(), "clip_index": np.asarray(record["clip_index"], np.int64),
                  "frame_index": np.asarray(record["frame_index"], np.int64)}
        checks = validate_arrays(arrays, normalized)
        target = path.parent / (Path(record["file"]).stem + ".npz")
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["frames"].append({**context, "artifact_file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
                                "rgb_sha256": record["rgb_sha256"], "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                "native_forward_errors": errors, "pointmap_checks": checks, "focal_pixels": float(values["focal_length"]),
                                "bbox_xyxy": box.tolist()})
        report["outputs"].append({"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
                                  "clip_index": record["clip_index"], "frame_index": record["frame_index"],
                                  "rgb_sha256": record["rgb_sha256"]})
        report.update(body_calls_completed=len(report["frames"]), MoGe_calls_completed=len(report["frames"]))
        human._write(path, report)
        del predictions, predicted, tensor, arrays, silhouette, rendered
    if (public_inputs(root) != (records, hashes) or depth_model.model_asset(root)[1:] != (acquisition, asset)
            or human.body._body_assets(root)[1] != body_source["body_assets"]
            or human.body._source_identity(root) != body_source["inference_source_identity"]
            or depth_model.installed_source(Path(moge.__file__).parent, json.loads(distribution.read_text("direct_url.json"))) != moge_source):
        raise ValueError("Frozen public inputs, model assets or inference sources changed")
    report.update(status="pass", phase="complete", frames_completed=CLIPS*FRAMES, actual_body_inference=True,
                  actual_MoGe_inference=True, actual_predicted_human_render=True, inputs_assets_sources_rechecked=True)
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote Linux CUDA network-none required")
    root = Path(os.environ["WR_ROOT"])
    output = root / "validation/joint_affine_rgb_v1/predictions_v1"
    if output.resolve() != output.absolute() or not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("Exclusively reserved empty nonsymlink J3 predictions directory required")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable producer source/image required")
    hashes = helper_identities()
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
              "script_sha256": sha256(Path(__file__)), "helper_source_sha256": hashes, "budget_seconds": BUDGET, "network": "none",
              "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False, "affine_fit_performed": False,
              "camera_K": CAMERA_K.tolist(), "camera_source": "fixed_RGB_size_prior", "body_inference_type": "body",
              "scale_fit_performed": False, "geometry_filled": False, "apply_mask": False, "force_projection": True,
              "metric_accuracy_verified": False, "frames": [], "outputs": [], "body_calls_completed": 0, "MoGe_calls_completed": 0,
              "native_focal_solver_calls": []}
    path = output / "report.json"; started = time.perf_counter()
    def expired(*args): raise TimeoutError("Whole J3 inference exceeded600s")
    old_alarm, old_term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired)
    signal.alarm(BUDGET)
    try:
        human._write(path, report); run_inference(root, report, path)
        if hashes != helper_identities():
            raise ValueError("Immutable inference helper source changed")
    except Exception as error:
        report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term)
        report["elapsed_seconds"] = time.perf_counter()-started; human._write(path, report)
    print(json.dumps({k: report[k] for k in ("stage", "status", "frames_completed", "elapsed_seconds")}))


if __name__ == "__main__": main()

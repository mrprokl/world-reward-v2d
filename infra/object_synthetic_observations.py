"""Public train-RGB automatic foreground/MoGe2 observations, no private truth.

Unmasked MoGe predictions stay untouched: nonfinite/nonpositive full-grid XYZ
fails, never filled with fabricated background geometry. Metric accuracy is not
established by pinhole consistency or by a pretrained metric-depth model.
"""
import argparse
from collections import deque
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
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

WIDTH, HEIGHT, TRAIN_VIEWS = 512, 384, (0, 2, 4)
SCHEMA = "world-reward-objects-rgb-inputs-v1"
MODEL_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MODEL_SHA = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
MODEL_BYTES = 1323815904
XET_SHA = "9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c"
SOURCE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
SOURCE_V2_SHA = "e736ed59fdb8ad89ec0c29ddfeeccd6a4246c387b3f423b922708366b69758fd"
RGB_DISTANCE, MIN_PIXELS = .08, 64


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Require a regular canonical nonsymlink file")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def public_inputs(directory):
    """Manifest contains all identities; held-out image pixels are not mounted."""
    directory = Path(directory); manifest_path = directory / "manifest.json"
    receipt = identity(manifest_path); manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != 12):
        raise ValueError("Require exact twelve-image public RGB-only manifest")
    selected = []; all_images = []
    for index, image in enumerate(manifest["images"]):
        obj, view = divmod(index, 6); filename = f"object_{obj:02d}_view_{view:02d}.png"
        if (not isinstance(image, dict) or set(image) != {"file", "sha256", "width", "height"}
                or image["file"] != filename or type(image["width"]) is not int or type(image["height"]) is not int
                or (image["width"], image["height"]) != (WIDTH, HEIGHT)
                or not isinstance(image["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", image["sha256"])):
            raise ValueError("Public RGB identity/order/grid mismatch; private fields forbidden")
        record = {"object_index": obj, "view_index": view, "file": filename, "rgb_sha256": image["sha256"]}
        all_images.append(record)
        if view in TRAIN_VIEWS:
            path = directory / filename
            if identity(path)["sha256"] != image["sha256"]: raise ValueError("Train RGB SHA mismatch")
            selected.append(record | {"path": path})
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[r["file"] for r in selected]}:
        raise ValueError("Observation container must expose only manifest and six training RGBs")
    return selected, receipt, all_images


def foreground_mask(rgb):
    """Fixed RGB border-median rule with one substantial 4-connected region."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3 or rgb.dtype != np.uint8:
        raise ValueError("Require original uint8 RGB")
    values = rgb.astype(np.float64)/255
    border = np.concatenate((values[0], values[-1], values[1:-1, 0], values[1:-1, -1]))
    background = np.median(border, axis=0)
    candidate = np.linalg.norm(values-background, axis=-1) > RGB_DISTANCE
    labels = np.zeros(candidate.shape, np.int32); components = []; height, width = candidate.shape
    for y, x in zip(*np.nonzero(candidate)):
        if labels[y, x]: continue
        number = len(components)+1; labels[y, x] = number; queue = deque([(int(y), int(x))]); count = 0
        while queue:
            yy, xx = queue.pop(); count += 1
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if 0 <= ny < height and 0 <= nx < width and candidate[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = number; queue.append((ny, nx))
        components.append(count)
    substantial = [i+1 for i, count in enumerate(components) if count >= MIN_PIXELS]
    if len(substantial) != 1: raise ValueError("Automatic RGB foreground absent or ambiguous; no fallback")
    mask = labels == substantial[0]
    return mask, {"background_median_RGB_unit": background.tolist(), "distance_threshold": RGB_DISTANCE,
                  "minimum_component_pixels": MIN_PIXELS, "component_pixels": components,
                  "selected_pixels": int(mask.sum()), "connectivity": 4}


def observations(rgb, foreground, depth, points, valid, normalized):
    if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8 or foreground.shape != (HEIGHT, WIDTH) or foreground.dtype != np.bool_:
        raise ValueError("Require fixed original RGB and automatic boolean foreground")
    valid, points, depth = np.asarray(valid), np.asarray(points), np.asarray(depth)
    if valid.shape != (HEIGHT, WIDTH) or valid.dtype != np.bool_ or points.dtype != np.float32 or depth.dtype != np.float32:
        raise ValueError("Require genuine MoGe boolean validity and original float32 outputs")
    focal = float(np.hypot(WIDTH, HEIGHT))
    K = np.array([[focal, 0., WIDTH/2], [0., focal, HEIGHT/2], [0., 0., 1.]])
    checks = validate_camera_pointmap(depth, points, np.ones_like(valid), normalized, K)
    mask = foreground & valid
    if np.count_nonzero(mask) < MIN_PIXELS: raise ValueError("Valid inferred foreground absent; no depth fill or scale fit")
    return {"rgb": rgb, "mask": mask, "pointmap": np.ascontiguousarray(points.transpose(2, 0, 1)),
            "K": K, "moge_validity": valid}, checks


def model_asset(root):
    receipt_path = root / "results/weights-acquisition.json"; receipt = identity(receipt_path)
    acquisition = json.loads(receipt_path.read_text()); cache = root / "weights/cari4d/hf_home/hub"
    records = [a for a in acquisition["assets"] if a.get("repo_id") == "Ruicheng/moge-2-vitl-normal"]
    if len(records) != 1 or records[0].get("revision") != MODEL_REVISION or records[0].get("cache_dir") != str(cache):
        raise ValueError("Require actual pinned MoGe2 acquisition revision/cache receipt")
    snapshot = cache / f"models--Ruicheng--moge-2-vitl-normal/snapshots/{MODEL_REVISION}/model.pt"
    # HF snapshots intentionally link their frozen blob. Reject any other link.
    blob = cache / f"models--Ruicheng--moge-2-vitl-normal/blobs/{MODEL_SHA}"
    # Audited cache deduplication may link the repo SHA blob to one global Xet
    # blob; the Xet storage identifier is not the file's required content SHA.
    xet_blob = cache / "blobs" / XET_SHA[:2] / XET_SHA
    target = snapshot.resolve()
    allowed = {blob.absolute(), xet_blob.absolute()}
    if target not in allowed: raise ValueError("HF snapshot link is not an audited SHA/Xet blob")
    path = target
    record = identity(path)
    if record != {"sha256": MODEL_SHA, "bytes": MODEL_BYTES}: raise ValueError("Independent pinned MoGe2 model SHA/size mismatch")
    return path, receipt, record


def installed_source(directory, direct_url):
    vcs = direct_url.get("vcs_info", {}) if isinstance(direct_url, dict) else {}
    if (not isinstance(direct_url, dict) or direct_url.get("url") != "https://github.com/microsoft/MoGe.git"
            or vcs.get("vcs") != "git" or vcs.get("commit_id") != SOURCE_REVISION):
        raise ValueError("Require actual installed pinned MoGe Git source metadata")
    directory = Path(directory); records = {str(p.relative_to(directory)): identity(p) for p in sorted(directory.rglob("*.py"))}
    if records.get("model/v2.py", {}).get("sha256") != SOURCE_V2_SHA: raise ValueError("Actual MoGe infer source differs from pinned primary source")
    return {"revision": SOURCE_REVISION, "direct_url": direct_url, "python_files": len(records),
            "python_source_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "v2_source_sha256": SOURCE_V2_SHA, "full_dependency_license_closure_verified": False}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux GPU container with network none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source revision/image ID")
    base = root / "validation/objects_rgb_v1"; output = base / "observations-v2"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved empty observations directory")
    with (output / "report.json").open("x") as handle:
        started = time.perf_counter()
        report = {"stage": "public_object_rgb_automatic_observations", "status": "fail", "phase": "integrity",
                  "script_sha256": sha256(Path(__file__)), "code_revision": revision, "image_id": image, "network": "none",
                  "private_truth_read": False, "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [],
                  "object_cases": 2, "train_views": list(TRAIN_VIEWS), "heldout_RGB_pixels_read": False, "outputs": [],
                  "metric_scale_accuracy_verified": False, "calibration_accuracy_verified": False, "adoption_performed": False,
                  "geometry_units": "MoGe_predicted_metres_not_verified_metric_scale", "geometry_frame": "opencv_x_right_y_down_z_forward",
                  "budget_seconds": 120, "MoGe_forward_calls": 0, "apply_mask": False, "force_projection": True,
                  "pointmap_storage": "original_full_grid_CHW_float32_no_geometry_fill", "foreground_rule": "RGB_border_median_4connected_then_MoGe_validity"}
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False)+"\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Public object observations exceeded120s")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired); signal.alarm(120)
        try:
            persist(); selected, receipt, all_images = public_inputs(base / "inputs")
            report.update(input_manifest=receipt, public_image_identities=all_images)
            path, acquisition, asset = model_asset(root); report.update(acquisition_report=acquisition, model_asset=asset, model_revision=MODEL_REVISION)
            import torch
            import moge
            from moge.model.v2 import MoGeModel
            from PIL import Image
            if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no local/CPU inference fallback")
            distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
            report["model_source"] = installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
            report.update(torch=torch.__version__, phase="inference"); persist()
            model = MoGeModel.from_pretrained(str(path)).cuda().eval()
            fov = float(np.degrees(2*np.arctan(WIDTH/(2*np.hypot(WIDTH, HEIGHT)))))
            for record in selected:
                report["active_image"] = record["file"]; persist()
                with Image.open(record["path"]) as png:
                    if png.format != "PNG" or png.mode != "RGB" or png.size != (WIDTH, HEIGHT): raise ValueError("Decoded train RGB/grid mismatch")
                    rgb = np.asarray(png).copy()
                foreground, segmentation = foreground_mask(rgb)
                tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
                report["MoGe_forward_calls"] += 1; persist()
                with torch.inference_mode(): predicted = model.infer(tensor[None], fov_x=fov, apply_mask=False)
                torch.cuda.synchronize()
                arrays, checks = observations(rgb, foreground, *(predicted[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics")))
                arrays.update(frame_index=np.array(record["view_index"], np.int64), object_index=np.array(record["object_index"], np.int64))
                target = output / (Path(record["file"]).stem+".npz")
                with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
                report["outputs"].append({"file": target.name, "sha256": sha256(target), "object_index": record["object_index"],
                    "view_index": record["view_index"], "rgb_sha256": record["rgb_sha256"],
                    "decoded_RGB_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "mask_sha256": hashlib.sha256(arrays["mask"].tobytes()).hexdigest(),
                    "pointmap_sha256": hashlib.sha256(arrays["pointmap"].tobytes()).hexdigest(), "segmentation": segmentation,
                    "MoGe_excluded_foreground_pixels": int(np.count_nonzero(foreground & ~arrays["moge_validity"])), "camera_checks": checks})
                persist(); del predicted, tensor
            if public_inputs(base / "inputs")[1] != receipt or model_asset(root)[1:] != (acquisition, asset): raise ValueError("Frozen inputs/assets changed during inference")
            report.update(status="pass", phase="complete", outputs_completed=6, model_forward_performed=True); report.pop("active_image", None)
        except Exception as error:
            report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "outputs_completed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()

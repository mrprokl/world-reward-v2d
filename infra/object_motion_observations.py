"""Public motion RGB-only automatic masks and unchanged MoGe2 pointmaps.

All 24 images are public inference inputs, not held-out validation pixels.
Pinhole consistency is not evidence of metric accuracy or reconstruction quality.
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
import object_synthetic_observations as observations_helper
from world_reward.data import sha256

SCHEMA = "world-reward-object-motion-rgb-v1"
STAGE = "public_object_motion_rgb_automatic_observations"
OBJECTS, FRAMES, WIDTH, HEIGHT, BUDGET = 3, 8, 512, 384, 180


def public_inputs(directory):
    """Exact RGB-only manifest/file whitelist; no private or calibration fields."""
    directory = Path(directory); manifest_path = directory / "manifest.json"
    receipt = observations_helper.identity(manifest_path); manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "images"} or manifest["schema"] != SCHEMA
            or not isinstance(manifest["images"], list) or len(manifest["images"]) != OBJECTS*FRAMES):
        raise ValueError("Require exact 24-image public motion RGB manifest")
    selected = []
    for index, record in enumerate(manifest["images"]):
        obj, frame = divmod(index, FRAMES); filename = f"object_{obj:02d}_frame_{frame:03d}.png"
        if (not isinstance(record, dict) or set(record) != {"file", "sha256", "width", "height"}
                or record["file"] != filename or type(record["width"]) is not int or type(record["height"]) is not int
                or (record["width"], record["height"]) != (WIDTH, HEIGHT)
                or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError("Public motion RGB identity/order/grid mismatch; private fields forbidden")
        path = directory / filename; actual = observations_helper.identity(path)
        if actual["sha256"] != record["sha256"] or actual["bytes"] <= 0:
            raise ValueError("Frozen motion RGB SHA/size mismatch")
        selected.append({"object_index": obj, "frame_index": frame, "file": filename,
                         "rgb_sha256": actual["sha256"], "bytes": actual["bytes"], "path": path})
    if {p.name for p in directory.iterdir()} != {"manifest.json", *[r["file"] for r in selected]}:
        raise ValueError("Only manifest and 24 public RGB files may be exposed")
    return selected, receipt


def array_identities(arrays):
    return {name: {"sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                   "dtype": str(value.dtype), "shape": list(value.shape)} for name, value in arrays.items()}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux GPU container with network none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source revision/image ID")
    base = root / "validation/object_motion_v1"; output = base / "observations"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("Require exclusively reserved empty motion observations directory")
    with (output / "report.json").open("x") as handle:
        started = time.perf_counter()
        report = {"schema": "world-reward-object-motion-observations-v1", "stage": STAGE, "status": "fail", "phase": "integrity",
            "script_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(observations_helper.__file__)),
            "code_revision": revision, "image_id": image, "network": "none", "private_truth_read": False,
            "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [], "object_cases": OBJECTS,
            "frames_per_object": FRAMES, "public_RGB_inputs": OBJECTS*FRAMES, "outputs": [], "MoGe_forward_calls": 0,
            "metric_scale_accuracy_verified": False, "calibration_accuracy_verified": False, "adoption_performed": False,
            "geometry_units": "MoGe_predicted_metres_not_verified_metric_scale", "geometry_frame": "opencv_x_right_y_down_z_forward",
            "budget_seconds": BUDGET, "apply_mask": False, "force_projection": True,
            "pointmap_storage": "original_full_grid_CHW_float32_no_geometry_fill", "foreground_rule": "RGB_border_median_4connected_then_MoGe_validity"}
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False)+"\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Public motion observations exceeded180s")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); selected, receipt = public_inputs(base / "inputs")
            report.update(input_manifest=receipt, public_image_identities=[{k: v for k, v in r.items() if k != "path"} for r in selected])
            path, acquisition, asset = observations_helper.model_asset(root)
            report.update(acquisition_report=acquisition, model_asset=asset, model_revision=observations_helper.MODEL_REVISION)
            import torch
            import moge
            from moge.model.v2 import MoGeModel
            from PIL import Image
            if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no local/CPU fallback")
            distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
            report["model_source"] = observations_helper.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
            report.update(torch=torch.__version__, phase="inference"); persist()
            model = MoGeModel.from_pretrained(str(path)).cuda().eval()
            fov = float(np.degrees(2*np.arctan(WIDTH/(2*np.hypot(WIDTH, HEIGHT)))))
            for record in selected:
                report["active_image"] = record["file"]; persist()
                with Image.open(record["path"]) as png:
                    if png.format != "PNG" or png.mode != "RGB" or png.size != (WIDTH, HEIGHT): raise ValueError("Decoded motion RGB/grid mismatch")
                    rgb = np.asarray(png).copy()
                foreground, segmentation = observations_helper.foreground_mask(rgb)
                tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
                report["MoGe_forward_calls"] += 1; persist()
                with torch.inference_mode(): predicted = model.infer(tensor[None], fov_x=fov, apply_mask=False)
                torch.cuda.synchronize()
                arrays, checks = observations_helper.observations(rgb, foreground, *(predicted[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics")))
                arrays.update(frame_index=np.array(record["frame_index"], np.int64), object_index=np.array(record["object_index"], np.int64))
                target = output / (Path(record["file"]).stem+".npz")
                with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
                report["outputs"].append({"file": target.name, "sha256": sha256(target), "object_index": record["object_index"],
                    "frame_index": record["frame_index"], "rgb_sha256": record["rgb_sha256"], "arrays": array_identities(arrays),
                    "segmentation": segmentation, "MoGe_excluded_foreground_pixels": int(np.count_nonzero(foreground & ~arrays["moge_validity"])), "camera_checks": checks})
                persist(); del predicted, tensor
            if public_inputs(base / "inputs") != (selected, receipt) or observations_helper.model_asset(root)[1:] != (acquisition, asset):
                raise ValueError("Frozen public inputs/assets changed during inference")
            report.update(status="pass", phase="complete", outputs_completed=OBJECTS*FRAMES, model_forward_performed=True); report.pop("active_image", None)
        except Exception as error:
            report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "outputs_completed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()

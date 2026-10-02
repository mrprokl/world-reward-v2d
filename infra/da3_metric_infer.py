"""Pinned DA3 metric inference on nine public TUD-L RGBs, never private truth.

Native resize, normalization, BF16 forward/FP32 head and sky correction remain
unchanged. Metric scaling uses the *actual processed* focal mean / 300 once;
bilinear camera-Z resizing and fixed-prior pinhole unprojection are explicit
World Reward postprocessing, not calibration fitting or accuracy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import time

import numpy as np
import da3_metric_acquire as acquisition
from tudl_infer import FIXED_K, FRAME_IDS, HEIGHT, WIDTH, public_inputs
from object_synthetic_observations import identity
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

STAGE = "public_tudl_rgb_da3_metric_depth_predictions"
BUDGET, STATE_COUNT = 600, 406
PROCESSED_HW = (392, 518)
PROCESSED_K = np.array([[647.5, 0., 259.], [0., 800*392/480, 196.], [0., 0., 1.]])
NATIVE_ARGUMENTS = dict(extrinsics=None, intrinsics=None, export_feat_layers=[],
                        infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced")


def pinned_assets(root):
    """Verify complete immutable acquisition inventory before upstream imports.

    Frozen receipts establish file integrity, not training-overlap, full license
    closure, physical scale, or numerical truth of another producer's claims.
    """
    root = Path(root); path = root/acquisition.REPORT
    receipt = identity(path); report = json.loads(path.read_text())
    expected = dict(stage="pinned_da3_metric_model_and_minimal_source_acquisition", status="pass",
                    model_revision=acquisition.MODEL_REV, source_revision=acquisition.SOURCE_REV,
                    license="Apache-2.0", challenge_inputs_used=False, inference_performed=False,
                    challenge_overlap_verified=False, source_modified=False)
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError("Require successful pinned, unmodified DA3 acquisition receipt")
    files = []
    for base, revision, repository, records in ((acquisition.SOURCE, acquisition.SOURCE_REV, acquisition.SOURCE_REPO, acquisition.SOURCE_RECORDS),
                                               (acquisition.WEIGHTS, acquisition.MODEL_REV, acquisition.MODEL_REPO, acquisition.MODEL_RECORDS)):
        for name, size, digest, blob in records:
            url = (f"https://raw.githubusercontent.com/{repository}/{revision}/{name}" if base == acquisition.SOURCE else
                   f"https://huggingface.co/{repository}/{'resolve' if blob is None else 'raw'}/{revision}/{name}")
            record = dict(file=base+"/"+name, url=url, bytes=size, sha256=digest, git_blob_sha1=blob)
            if identity(root/record["file"]) != dict(bytes=size, sha256=digest):
                raise ValueError("Pinned model/source SHA or byte count differs")
            files.append(record)
    if report.get("files") != files:
        raise ValueError("DA3 acquisition inventory/order differs from pinned assets")
    source = root/acquisition.SOURCE; manifest_path = source/"source_manifest.json"
    manifest_identity = identity(manifest_path); manifest = json.loads(manifest_path.read_text())
    entries = [{**r, "file": str(Path(r["file"]).relative_to(acquisition.SOURCE))} for r in files[:len(acquisition.SOURCE_RECORDS)]]
    expected_manifest = dict(schema="world-reward-da3-metric-source-v1", source_revision=acquisition.SOURCE_REV,
                             python_files=27, python_bytes=192452, source_bytes=225327, files=entries,
                             source_modified=False, root_namespace_init_fabricated=False, high_level_api_acquired=False)
    if (manifest != expected_manifest or report.get("source_manifest_sha256") != manifest_identity["sha256"]
            or any(manifest.get(k) is not False for k in ("source_modified", "root_namespace_init_fabricated", "high_level_api_acquired"))):
        raise ValueError("Require exact minimal native source manifest; no namespace/API fabrication")
    for base, names in ((source, {r[0] for r in acquisition.SOURCE_RECORDS} | {"source_manifest.json"}),
                        (root/acquisition.WEIGHTS, {r[0] for r in acquisition.MODEL_RECORDS})):
        if any(p.is_symlink() for p in (base, *base.rglob("*"))) or {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()} != names:
            raise ValueError("No unrecorded source/model files or symlinks may enter inference")
    return {"acquisition_report": receipt, "source_manifest_sha256": manifest_identity["sha256"],
            "source_identity": {"files": files[:len(acquisition.SOURCE_RECORDS)]},
            "model_sha256": acquisition.MODEL_RECORDS[-1][2]}, source


def checkpoint_state(state, expected):
    """Strip exactly one literal model. prefix; retain all 406 FP32 states.

    CPU arrays/tensors only: no filtering, ignored keys, random initialization,
    dtype conversion, shape adaptation or generated-buffer exception.
    """
    if not isinstance(state, dict) or not isinstance(expected, dict) or len(state) != STATE_COUNT or len(expected) != STATE_COUNT:
        raise ValueError("Require all406 native checkpoint/model states")
    stripped = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name.startswith("model.") or len(name) <= 6:
            raise ValueError("Every checkpoint key requires one literal model. prefix")
        key = name[6:]
        if key in stripped or key.startswith("model.") or key not in expected:
            raise ValueError("Checkpoint namespace/collision/inventory differs")
        if (np.ma.isMaskedArray(value) or str(value.dtype) not in ("float32", "torch.float32") or value.dtype != expected[key].dtype
                or tuple(value.shape) != tuple(expected[key].shape) or not np.isfinite(np.asarray(value)).all()):
            raise ValueError("Every retained state requires exact native shape and finite FP32 values")
        stripped[key] = value
    if set(stripped) != set(expected): raise ValueError("Native parameter/buffer inventory incomplete")
    return stripped


def metric_factor(processed_K):
    """Use native float32 resized K, not original focal, crop or fitted scale."""
    K = np.asarray(processed_K)
    if (np.ma.isMaskedArray(processed_K) or K.shape != (3, 3) or K.dtype not in (np.float32, np.float64)
            or not np.isfinite(K).all() or not np.allclose(K, PROCESSED_K, atol=1e-3, rtol=1e-6)):
        raise ValueError("Native processed grid/camera differs from the declared resize")
    return (float(K[0, 0])+float(K[1, 1]))/2/300


def camera_arrays(depth, scene_id, frame_id):
    """No pixel exclusion: every original-grid positive finite Z becomes XYZ."""
    if np.ma.isMaskedArray(depth): raise ValueError("No masked depth or confidence exclusion permitted")
    depth = np.asarray(depth)
    if (depth.shape != (HEIGHT, WIDTH) or depth.dtype != np.float32
            or not np.isfinite(depth).all() or np.any(depth <= 0)):
        raise ValueError("Require finite positive FP32 camera Z on every original pixel")
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    points = np.stack(((xx+.5-FIXED_K[0, 2])/FIXED_K[0, 0]*depth,
                       (yy+.5-FIXED_K[1, 2])/FIXED_K[1, 1]*depth, depth), axis=-1).astype(np.float32)
    arrays = dict(depth=depth, points=points, validity=np.ones(depth.shape, bool), K=FIXED_K.copy(),
                  scene_id=np.array(scene_id, np.int64), frame_id=np.array(frame_id, np.int64))
    return arrays, validate_prediction_arrays(arrays, scene_id, frame_id)


def validate_prediction_arrays(data, scene_id, frame_id):
    """CPU-only public fixed-camera NPZ geometry contract, not accuracy."""
    if (set(data) != {"depth", "points", "validity", "K", "scene_id", "frame_id"}
            or type(scene_id) is not int or type(frame_id) is not int
            or scene_id not in (1, 2, 3) or frame_id not in FRAME_IDS[scene_id-1]):
        raise ValueError("Require exact public scene/frame and DA3 array keys")
    for name, expected in (("scene_id", scene_id), ("frame_id", frame_id)):
        value = np.asarray(data[name])
        if np.ma.isMaskedArray(data[name]) or value.shape != () or value.dtype != np.int64 or int(value) != expected:
            raise ValueError("Require unchanged integer scalar scene/frame")
    for name, shape, dtype in (("depth", (HEIGHT, WIDTH), np.float32), ("points", (HEIGHT, WIDTH, 3), np.float32),
                               ("validity", (HEIGHT, WIDTH), np.bool_), ("K", (3, 3), np.float64)):
        value = data[name]
        if np.ma.isMaskedArray(value) or value.shape != shape or value.dtype != dtype:
            raise ValueError("DA3 original-grid arrays/dtypes differ")
    if not data["validity"].all() or not np.array_equal(data["K"], FIXED_K):
        raise ValueError("No confidence exclusion, fitted camera or alternate focal permitted")
    return validate_camera_pointmap(data["depth"], data["points"], data["validity"],
                                    np.diag([1/WIDTH, 1/HEIGHT, 1.])@data["K"], FIXED_K)


def native_forward(torch, network, images):
    """One original native call with explicit per-call RNG, never warmup."""
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return network(images[None].cuda(), **NATIVE_ARGUMENTS)


def resize_metric_depth(torch, depth, factor):
    """Multiply processed camera-Z once, then native bilinear original grid."""
    metric = torch.from_numpy(depth[None]).cuda()*factor
    return torch.nn.functional.interpolate(metric, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)


def run(root, output, report, persist):
    directory = root/"validation/tudl_rgb_v1/inputs"; records, receipt = public_inputs(directory)
    assets, source = pinned_assets(root)
    report.update(assets, input_manifest=receipt, public_input_manifest_sha256=receipt["sha256"],
                  public_records=[{k: v for k, v in r.items() if k != "path"} for r in records], phase="model_load"); persist()
    import sys
    sys.path.insert(0, str(source/"src"))
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3 import cfg
    from depth_anything_3.utils.io import input_processor, output_processor
    for module in (cfg, input_processor, output_processor):
        if Path(module.__file__).resolve() != source/"src"/Path(*module.__name__.split(".")).with_suffix(".py"):
            raise ValueError("Native module resolved outside the verified minimal source")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no altered CPU fallback")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    config = json.loads((root/acquisition.WEIGHTS/"config.json").read_text())
    network = cfg.create_object(OmegaConf.create(config["config"])).eval()
    state = checkpoint_state(load_file(str(root/acquisition.WEIGHTS/"model.safetensors"), device="cpu"), dict(network.state_dict()))
    network.load_state_dict(state, strict=True); del state
    network = network.cuda(); torch.cuda.reset_peak_memory_stats()
    processor, postprocessor = input_processor.InputProcessor(), output_processor.OutputProcessor()
    report.update(torch=torch.__version__, checkpoint_states_loaded=STATE_COUNT, checkpoint_load_strict=True); persist()
    for record in records:
        report.update(phase="native_inference", active_file=record["file"]); persist()
        with Image.open(record["path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                raise ValueError("Require actual original640x480 RGB PNG")
            rgb = np.asarray(image).copy()
        images, ext, cameras = processor([rgb], extrinsics=None, intrinsics=np.array([FIXED_K]),
            process_res=518, process_res_method="upper_bound_resize", num_workers=1, sequential=True, print_progress=False)
        if tuple(images.shape) != (1, 3, *PROCESSED_HW) or images.dtype != torch.float32 or ext is not None or tuple(cameras.shape) != (1, 3, 3):
            raise ValueError("Native preprocessing tensor/grid/intrinsics ABI differs")
        K = cameras[0].numpy(); factor = metric_factor(K)
        report.update(processed_camera_K=K.tolist(), metric_depth_factor=factor); persist()
        raw = native_forward(torch, network, images)
        prediction = postprocessor(raw); depth = prediction.depth
        if depth.shape != (1, *PROCESSED_HW) or depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("Native full positive FP32 depth ABI required; no sky/confidence repair")
        resized = resize_metric_depth(torch, depth, factor)
        depth_original = resized[0, 0].cpu().numpy(); torch.cuda.synchronize()
        report["DA3_calls_completed"] += 1
        arrays, checks = camera_arrays(depth_original, record["scene_id"], record["frame_id"])
        target = output/(Path(record["file"]).stem+".npz")
        with target.open("xb") as handle: np.savez_compressed(handle, **arrays)
        report["outputs"].append(dict(scene_id=record["scene_id"], frame_id=record["frame_id"], file=target.name,
            sha256=sha256(target), rgb_sha256=record["sha256"], decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            pointmap_checks=checks, processed_camera_K=K.tolist(), metric_depth_factor=factor,
            arrays={k: dict(sha256=hashlib.sha256(v.tobytes()).hexdigest(), dtype=str(v.dtype), shape=list(v.shape)) for k, v in arrays.items()}))
        report["peak_GPU_allocated_bytes"] = int(torch.cuda.max_memory_allocated()); persist()
        del raw, prediction, images, resized, arrays
    final_assets, final_source = pinned_assets(root)
    if public_inputs(directory) != (records, receipt) or final_assets != assets or final_source != source:
        raise ValueError("Frozen public RGB/model/full native source changed during inference")
    if report["DA3_calls_completed"] != 9 or len(report["outputs"]) != 9: raise ValueError("All nine actual native forwards required")
    report.update(status="pass", phase="complete", outputs_completed=9, original_frame_coverage_verified=True,
                  actual_DA3_inference=True); report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require isolated remote Linux CUDA inference")
    root = Path(os.environ["WR_ROOT"]); output = root/"validation/tudl_rgb_v1/da3_predictions_v1"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved empty predictions")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable inference source/image identity")
    with (output/"report.json").open("x") as handle:
        started = time.perf_counter()
        report = dict(stage=STAGE, status="fail", phase="integrity", producer_revision=revision, image_id=image,
            script_sha256=sha256(Path(__file__)), public_input_helper_sha256=sha256(Path(public_inputs.__code__.co_filename)),
            acquisition_helper_sha256=sha256(Path(acquisition.__file__)), budget_seconds=BUDGET, network="none",
            private_truth_read=False, challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False,
            oracle_modes=[], scale_fit=False, adoption_performed=False, training_overlap_verified=False,
            full_license_closure_verified=False, metric_scale_accuracy_verified=False, calibration_accuracy_verified=False,
            model_revision=acquisition.MODEL_REV, source_revision=acquisition.SOURCE_REV, license="Apache-2.0",
            fixed_camera_K=FIXED_K.tolist(), processed_grid=list(PROCESSED_HW), processed_camera_K_nominal=PROCESSED_K.tolist(),
            native_forward_arguments=NATIVE_ARGUMENTS, autocast="bfloat16_native_head_float32", seed=0,
            RNG_seed_reset_before_each_forward=True, asset_source_reverified_after_inference=True,
            native_sky_correction_unchanged=True, native_sky_threshold=.3, native_sky_quantile=.99,
            metric_depth_scaling_applied_once=True, metric_depth_factor_rule="actual_processed_focal_mean_divided_by_300",
            postprocess="metric_camera_Z_then_bilinear_align_corners_false_then_fixed_K_pixel_centres_plus_0.5",
            confidence_validity_exclusion=False, pointmap_geometry_filled=False, outputs=[], DA3_calls_completed=0)
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n"); handle.truncate(); handle.flush()
        def expired(*_): raise TimeoutError("Whole public DA3 inference exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, output, report, persist); persist()
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "DA3_calls_completed", "outputs_completed", "elapsed_seconds")}))


if __name__ == "__main__": main()

"""Native fixed-camera depth on a fresh public RGB cohort; no person or masks.

MoGe retains its original validity and XYZ. DA3 retains native preprocessing,
sky correction and BF16/FP32 forward, then processed focal/300 metric scaling
once and explicit original-grid camera-Z unprojection. No labels, alignment,
calibration, shared identity or confidence-based pixel removal are used.
"""
import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time
import zipfile

import numpy as np
import da3_metric_infer as da3
import depth_camera_support as moge_camera
import object_synthetic_observations as moge_assets
import rgb_cohort_protocol as protocol
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

BASE, WIDTH, HEIGHT = protocol.BASE, protocol.WIDTH, protocol.HEIGHT
CAMERA_K = protocol.FIXED_K
STAGE, BUDGET = "public_depth_rgb_native_depth_predictions", 180
ADDICT_PATH = "vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl"
ADDICT_SHA = "249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc"
IMAGES = {"moge": "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7",
          "da3": "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"}
ARRAY_KEYS = frozenset({"depth", "points", "validity", "camera_K", "clip_index", "frame_index"})
PROCESSED_HW = (392, 518)
PROCESSED_K = np.array([[1280*518/1024, 0., 259.], [0., 1280*392/768, 196.], [0., 0., 1.]])


def helper_identities():
    return {name: sha256(Path(module.__file__)) for name, module in
            {"public_protocol": protocol, "da3_native": da3, "da3_acquisition": da3.acquisition,
             "MoGe_assets": moge_assets, "MoGe_camera": moge_camera}.items()} | {"producer": sha256(Path(__file__))}


def public_inputs(root):
    records, receipt = protocol.public_inputs(Path(root)/BASE/"inputs")
    return records, {"public_manifest_sha256": receipt["sha256"], "public_inputs_sha256": receipt["sha256"], "public_inputs_bytes": receipt["bytes"]}


def validate_arrays(data, record):
    """Exact public grid/XYZ contract; invalid MoGe records are not repaired."""
    if set(data) != ARRAY_KEYS: raise ValueError("Exact depth RGB prediction schema required")
    for key, shape, dtype in (("depth", (HEIGHT, WIDTH), np.float32), ("points", (HEIGHT, WIDTH, 3), np.float32),
            ("validity", (HEIGHT, WIDTH), np.bool_), ("camera_K", (3, 3), np.float64),
            ("clip_index", (), np.int64), ("frame_index", (), np.int64)):
        value = data[key]
        if np.ma.isMaskedArray(value) or np.shape(value) != shape or np.asarray(value).dtype != dtype:
            raise ValueError("Original grid/native dtype differs; no conversion")
    if (not np.array_equal(data["camera_K"], CAMERA_K) or int(data["clip_index"]) != record["clip_index"]
            or int(data["frame_index"]) != record["frame_index"]):
        raise ValueError("Fixed public camera/original frame identity differs")
    return validate_camera_pointmap(data["depth"], data["points"], data["validity"],
        np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ CAMERA_K, CAMERA_K)


def camera_arrays(depth, record):
    if np.ma.isMaskedArray(depth): raise ValueError("No masked depth input permitted")
    depth = np.asarray(depth)
    if depth.shape != (HEIGHT, WIDTH) or depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("DA3 requires every original pixel positive finite FP32; no drop/fill")
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    points = np.stack(((xx+.5-CAMERA_K[0, 2])/CAMERA_K[0, 0]*depth,
                       (yy+.5-CAMERA_K[1, 2])/CAMERA_K[1, 1]*depth, depth), axis=-1).astype(np.float32)
    return dict(depth=depth, points=points, validity=np.ones(depth.shape, bool), camera_K=CAMERA_K.copy(),
        clip_index=np.array(record["clip_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64))


def metric_factor(camera):
    value = np.asarray(camera)
    if np.ma.isMaskedArray(camera) or value.shape != (3, 3) or value.dtype not in (np.float32, np.float64) or not np.isfinite(value).all() or not np.allclose(value, PROCESSED_K, atol=1e-3, rtol=1e-6):
        raise ValueError("Native processed camera/grid differs from fixed resize")
    return (float(value[0, 0])+float(value[1, 1]))/2/300


def moge_source():
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    spec = util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Pinned installed MoGe package missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    source = moge_assets.installed_source(directory, json.loads(direct) if direct else {})
    if sha256(directory/"utils/geometry_torch.py") != moge_camera.GEOMETRY_SHA: raise ValueError("Pinned native focal solver differs")
    return source


def da3_dependency(root):
    """Bind the exact imported pure-Python wheel, without installing packages."""
    path = Path(root)/ADDICT_PATH
    if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_size != 3832 or sha256(path) != ADDICT_SHA:
        raise ValueError("Exact pinned Addict dependency wheel required")
    spec = util.find_spec("addict")
    if spec is None or spec.origin != str(path)+"/addict/__init__.py":
        raise ValueError("Native Addict must resolve directly from the pinned wheel")
    with zipfile.ZipFile(path) as archive:
        expected = {"addict/__init__.py", "addict/addict.py", *["addict-2.4.0.dist-info/"+name for name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt", "RECORD")]}
        if set(archive.namelist()) != expected: raise ValueError("Pinned Addict wheel source inventory differs")
        sources = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(expected)}
    return {"path": str(path), "sha256": ADDICT_SHA, "bytes": 3832, "import_origin": spec.origin, "source_sha256": sources}


def validate_bindings(root, report):
    """CPU-only current byte/source audit, not proof of reported numeric truth."""
    backend = report.get("backend")
    if (backend not in IMAGES or report.get("image_id") != IMAGES[backend]
            or report.get("script_sha256") != sha256(Path(__file__)) or report.get("helper_source_sha256") != helper_identities()
            or report.get("sources_assets_rechecked") is not True):
        raise ValueError("Pinned backend/image/current source binding differs")
    if backend == "da3":
        assets, _ = da3.pinned_assets(root)
        if (report.get("assets") != assets or report.get("native_arguments") != da3.NATIVE_ARGUMENTS
                or report.get("dependency") != da3_dependency(root)):
            raise ValueError("Exact native DA3 model/source/config differs")
    else:
        _, receipt, asset = moge_assets.model_asset(root)
        if report.get("acquisition_report") != receipt or report.get("model_asset") != asset or report.get("native_source") != moge_source():
            raise ValueError("Exact native MoGe model/source differs")
    return True


def load_da3(root, torch, report):
    assets, source = da3.pinned_assets(root); dependency = da3_dependency(root); sys.path.insert(0, str(source/"src"))
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3 import cfg
    from depth_anything_3.utils.io import input_processor, output_processor
    for module in (cfg, input_processor, output_processor):
        if Path(module.__file__).resolve() != source/"src"/Path(*module.__name__.split(".")).with_suffix(".py"):
            raise ValueError("Native DA3 module resolved outside pinned minimal source")
    config = json.loads((root/da3.acquisition.WEIGHTS/"config.json").read_text())
    network = cfg.create_object(OmegaConf.create(config["config"])).eval()
    state = da3.checkpoint_state(load_file(str(root/da3.acquisition.WEIGHTS/"model.safetensors"), device="cpu"), dict(network.state_dict()))
    network.load_state_dict(state, strict=True); del state
    report.update(assets=assets, dependency=dependency, native_arguments=da3.NATIVE_ARGUMENTS, checkpoint_states_loaded=da3.STATE_COUNT,
        checkpoint_load_strict=True, native_sky_correction_unchanged=True, native_sky_threshold=.3, native_sky_quantile=.99,
        confidence_validity_exclusion=False, metric_depth_scaling_applied_once=True, native_processed_grid=list(PROCESSED_HW))
    return network.cuda(), input_processor.InputProcessor(), output_processor.OutputProcessor()


def da3_frame(torch, network, processor, postprocessor, rgb, record):
    images, extrinsics, cameras = processor([rgb], extrinsics=None, intrinsics=np.array([CAMERA_K]), process_res=518,
        process_res_method="upper_bound_resize", num_workers=1, sequential=True, print_progress=False)
    if tuple(images.shape) != (1, 3, *PROCESSED_HW) or images.dtype != torch.float32 or extrinsics is not None or tuple(cameras.shape) != (1, 3, 3):
        raise ValueError("Native DA3 preprocessing tensor/camera ABI differs")
    processed = cameras[0].numpy(); factor = metric_factor(processed)
    prediction = postprocessor(da3.native_forward(torch, network, images)); depth = prediction.depth
    if depth.shape != (1, *PROCESSED_HW) or depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Native DA3 positive camera-Z ABI differs; no repair")
    metric = torch.from_numpy(depth[None]).cuda()*factor
    original = torch.nn.functional.interpolate(metric, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return camera_arrays(original, record), {"processed_camera_K": processed.tolist(), "metric_depth_factor": factor}


def run(root, output, report, persist):
    records, receipt = public_inputs(root); helpers = helper_identities(); backend = report["backend"]
    report.update(**receipt, helper_source_sha256=helpers, phase="model_load"); persist()
    if backend == "da3": assets, _ = da3.pinned_assets(root); report["assets"] = assets
    else:
        asset_path, acquisition, asset = moge_assets.model_asset(root)
        report.update(acquisition_report=acquisition, model_asset=asset, native_source=moge_source(),
                      focal_geometry_source_sha256=moge_camera.GEOMETRY_SHA)
    import torch
    from PIL import Image
    if not torch.cuda.is_available(): raise RuntimeError("Native CUDA required; no local/CPU fallback")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if backend == "da3": network, processor, postprocessor = load_da3(root, torch, report)
    else:
        from moge.model import v2 as native_module
        from moge.model.v2 import MoGeModel
        from moge.utils import geometry_torch
        if native_module.recover_focal_shift is not geometry_torch.recover_focal_shift: raise ValueError("Exact original native focal callback required")
        network = MoGeModel.from_pretrained(str(asset_path)).cuda().eval()
    torch.cuda.reset_peak_memory_stats(); persist()
    for record in records:
        report.update(phase="native_inference", active_file=record["file"]); persist()
        with Image.open(record["path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT): raise ValueError("Exact original public RGB required")
            rgb = np.asarray(image).copy()
        if backend == "da3": arrays, details = da3_frame(torch, network, processor, postprocessor, rgb, record)
        else:
            tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
            prediction = moge_camera.checked_depth_infer(torch, native_module, network, tensor,
                float(np.degrees(2*np.arctan(WIDTH/(2*CAMERA_K[0, 0])))), report["native_focal_solver_calls"],
                {k: record[k] for k in ("file", "clip_index", "frame_index")})
            depth, points, validity, normalized = (prediction[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics"))
            validate_camera_pointmap(depth, points, validity, normalized, CAMERA_K)
            arrays = dict(depth=depth, points=points, validity=validity, camera_K=CAMERA_K.copy(),
                clip_index=np.array(record["clip_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64)); details = {}
            del prediction, tensor
        torch.cuda.synchronize(); report["actual_network_calls"] += 1; checks = validate_arrays(arrays, record)
        target = output/(Path(record["file"]).stem+".npz")
        with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
        target.chmod(0o444)
        report["outputs"].append({k: record[k] for k in ("file", "clip_index", "frame_index")} | dict(artifact=target.name,
            sha256=sha256(target), bytes=target.stat().st_size, rgb_sha256=record["sha256"],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), pointmap_checks=checks, **details))
        report["peak_GPU_allocated_bytes"] = int(torch.cuda.max_memory_allocated()); persist(); del arrays
    if public_inputs(root) != (records, receipt) or helper_identities() != helpers: raise ValueError("Frozen public inputs/source changed")
    report["sources_assets_rechecked"] = True; validate_bindings(root, report)
    if (report["actual_network_calls"] != 15 or len(report["outputs"]) != 15
            or (backend == "moge" and len(report["native_focal_solver_calls"]) != 15)
            or {p.name for p in output.iterdir()} != {"report.json", *[Path(r["file"]).stem+".npz" for r in records]}):
        raise ValueError("All fifteen genuine calls/outputs required; no extras")
    for row in report["outputs"]:
        target = output/row["artifact"]
        if target.stat().st_mode & 0o222 or sha256(target) != row["sha256"] or target.stat().st_size != row["bytes"]: raise ValueError("Frozen output changed")
    report.update(status="pass", phase="complete", frames=15, all_cases_retained=True, actual_native_inference=True)
    report.pop("active_file", None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False); parser.add_argument("--backend", choices=tuple(IMAGES), required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Remote network-none CUDA required")
    root = Path(os.environ["WR_ROOT"]); output = root/BASE/(args.backend+"_predictions_v1")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if root != Path("/srv/scenesmith/world-reward") or output.resolve() != output.absolute() or not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("Require exclusively reserved fresh backend output")
    if not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGES[args.backend]: raise ValueError("Pinned native backend/image/source required")
    report = dict(stage=STAGE, backend=args.backend, status="fail", phase="public_integrity", producer_revision=revision, image_id=image,
        script_sha256=sha256(Path(__file__)), budget_seconds=BUDGET, network="none", private_truth_read=False, challenge_inputs_used=False,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], camera_K=CAMERA_K.tolist(), camera_fitted=False,
        scale_fit=False, identity_fit=False, alignment_performed=False, human_inference_performed=False, masks_used=False,
        pointmap_geometry_filled=False, adoption_performed=False, quality_verified=False, training_overlap_verified=False,
        full_license_closure_verified=False, actual_network_calls=0, outputs=[], native_focal_solver_calls=[])
    started = time.perf_counter(); path = output/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole native depth RGB producer exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, output, report, persist)
        except Exception as error: report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()

"""One frozen border-proxy DA3/MoGe depth hypothesis on twelve TUD-L frames.

The RGB peripheral band is a background PROXY, never semantic exclusion of
people/objects. One median-of-four-medians coefficient per scene anchors DA3
relative camera-Z to unchanged MoGe. No private truth, calibration, per-frame
anchor, offset, geometry/human changes or candidate-adoption decision is used.
"""
import argparse
import gc
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time
import zipfile

import numpy as np
import da3_metric_infer as da3
import object_synthetic_observations as moge_assets
import tudl_infer as moge_native
import tudl_holdout_inputs as inputs
from world_reward.pointmap import validate_camera_pointmap

ROOT = Path("/srv/scenesmith/world-reward")
BASE = "validation/tudl_frame_holdout_v1"
PIN_FILE = "configs/tudl_frame_holdout_input_pins.json"
STAGE = "public_tudl_holdout_border_anchored_depth_predictions"
BUDGET = 900
IMAGE_ID = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
WIDTH, HEIGHT = inputs.WIDTH, inputs.HEIGHT
K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
MIN_BORDER_PAIRS, MIN_BORDER_COVERAGE = 1024, .95
ADDICT_PATH = "vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl"
ADDICT_SHA = "249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc"
SOURCE_FILES = ("infra/tudl_anchor_infer.py", "infra/tudl_holdout_inputs.py", "infra/tudl_infer.py",
    "infra/object_synthetic_observations.py", "infra/da3_metric_infer.py", "infra/da3_metric_acquire.py",
    "src/world_reward/__init__.py", "src/world_reward/data.py", "src/world_reward/pointmap.py")
ARRAY_KEYS = frozenset({"moge_depth", "moge_points", "moge_validity", "da3_depth", "anchored_depth",
    "anchored_points", "anchored_validity", "K", "scene_id", "frame_id", "scene_anchor"})


def frame_identity(scene, frame):
    if type(scene) is not int or type(frame) is not int or scene not in (1, 2, 3) or frame not in inputs.FRAME_IDS[scene - 1]:
        raise ValueError("Exact actual held-out scene/frame integers required")
    return f"scene_{scene:06d}_frame_{frame:06d}.png"


def border_mask():
    yy, xx = np.indices((HEIGHT, WIDTH))
    return (xx < 64) | (xx >= 576) | (yy < 48) | (yy >= 432)


def _depth(value, name, full_positive=False):
    if (np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray)
            or value.shape != (HEIGHT, WIDTH) or value.dtype != np.float32):
        raise ValueError(name + ": original FP32 full-grid camera-Z required")
    if full_positive and (not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError(name + ": every original pixel must be finite positive, no fill/drop")
    return value


def validate_baseline(baseline):
    if type(baseline) is not dict or set(baseline) != {"depth", "points", "validity", "K"}:
        raise ValueError("Exact unchanged native MoGe camera arrays required")
    _depth(baseline["depth"], "MoGe")
    for name, shape, dtype in (("points", (HEIGHT, WIDTH, 3), np.float32),
            ("validity", (HEIGHT, WIDTH), np.bool_), ("K", (3, 3), np.float64)):
        value = baseline[name]
        if np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError("Original native MoGe pointmap/validity/camera dtype and shape required")
    if not np.array_equal(baseline["K"], K): raise ValueError("Same fixed RGB-size cameraK800 required")
    return validate_camera_pointmap(baseline["depth"], baseline["points"], baseline["validity"],
        np.diag([1 / WIDTH, 1 / HEIGHT, 1.]) @ K, K)


def border_ratio(moge_depth, moge_validity, da3_depth):
    _depth(moge_depth, "MoGe"); _depth(da3_depth, "DA3", True)
    if (np.ma.isMaskedArray(moge_validity) or not isinstance(moge_validity, np.ndarray)
            or moge_validity.shape != (HEIGHT, WIDTH) or moge_validity.dtype != np.bool_):
        raise ValueError("Original boolean MoGe validity required, no support repair")
    if not np.isfinite(moge_depth[moge_validity]).all() or np.any(moge_depth[moge_validity] <= 0):
        raise ValueError("Every native valid MoGe camera-Z must be positive finite")
    border = border_mask(); paired = border & moge_validity
    count, total = int(paired.sum()), int(border.sum())
    if count < MIN_BORDER_PAIRS or count / total < MIN_BORDER_COVERAGE:
        raise ValueError("Insufficient fixed border-proxy paired support; no fallback/sweep")
    ratios = moge_depth[paired].astype(np.float64) / da3_depth[paired].astype(np.float64)
    ratio = float(np.median(ratios))
    if not np.isfinite(ratios).all() or np.any(ratios <= 0) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Finite positive unchanged border depth ratios required")
    return {"border_pixels": total, "paired_valid_pixels": count, "paired_border_coverage": count / total,
        "minimum_paired_pixels": MIN_BORDER_PAIRS, "minimum_border_coverage": MIN_BORDER_COVERAGE,
        "ratio_median": ratio}


def scene_anchors(diagnostics):
    if type(diagnostics) is not list or len(diagnostics) != 12:
        raise ValueError("All twelve ordered image ratios required before any scene anchor")
    values = []
    for index, row in enumerate(diagnostics):
        scene, frame = index // 4 + 1, inputs.FRAME_IDS[index // 4][index % 4]
        if (type(row) is not dict or row.get("scene_id") != scene or type(row.get("scene_id")) is not int
                or row.get("frame_id") != frame or type(row.get("frame_id")) is not int
                or type(row.get("ratio_median")) is not float or not math.isfinite(row["ratio_median"])
                or row["ratio_median"] <= 0 or row.get("minimum_paired_pixels") != MIN_BORDER_PAIRS
                or row.get("minimum_border_coverage") != MIN_BORDER_COVERAGE
                or type(row.get("border_pixels")) is not int or row["border_pixels"] != int(border_mask().sum())
                or type(row.get("paired_valid_pixels")) is not int or not MIN_BORDER_PAIRS <= row["paired_valid_pixels"] <= row["border_pixels"]
                or row.get("paired_border_coverage") != row["paired_valid_pixels"] / row["border_pixels"]
                or row["paired_border_coverage"] < MIN_BORDER_COVERAGE):
            raise ValueError("Complete fixed ordered border-proxy ratio/support diagnostics required")
        values.append(row["ratio_median"])
    return np.median(np.asarray(values, np.float64).reshape(3, 4), axis=1).tolist()


def array_identities(arrays):
    return {key: {"sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "dtype": str(value.dtype), "shape": list(value.shape)} for key, value in arrays.items()}


def candidate_arrays(baseline, raw_da3, coefficient, scene, frame):
    frame_identity(scene, frame); validate_baseline(baseline); _depth(raw_da3, "DA3", True)
    if type(coefficient) is not float or not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("One finite positive scene scalar required; no per-frame or offset fit")
    before = array_identities(baseline)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        anchored = (raw_da3.astype(np.float64) * coefficient).astype(np.float32)
    _depth(anchored, "Anchored DA3", True)
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    points = np.stack(((xx + .5 - K[0, 2]) / K[0, 0] * anchored,
        (yy + .5 - K[1, 2]) / K[1, 1] * anchored, anchored), axis=-1).astype(np.float32)
    if not np.isfinite(points).all(): raise ValueError("Anchored full-grid XYZ overflowed; no repair")
    result = {"moge_depth": baseline["depth"].copy(), "moge_points": baseline["points"].copy(),
        "moge_validity": baseline["validity"].copy(), "da3_depth": raw_da3.copy(), "anchored_depth": anchored,
        "anchored_points": points, "anchored_validity": baseline["validity"].copy(), "K": K.copy(),
        "scene_id": np.array(scene, np.int64), "frame_id": np.array(frame, np.int64),
        "scene_anchor": np.array(coefficient, np.float64)}
    validate_prediction_arrays(result, scene, frame, coefficient)
    if array_identities(baseline) != before: raise ValueError("Original baseline was mutated")
    return result


def validate_prediction_arrays(data, scene, frame, coefficient):
    frame_identity(scene, frame)
    if type(data) is not dict or set(data) != ARRAY_KEYS:
        raise ValueError("Exact baseline/raw/anchored public prediction array inventory required")
    for key, expected in (("scene_id", scene), ("frame_id", frame)):
        value = data[key]
        if np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray) or value.shape != () or value.dtype != np.int64 or int(value) != expected:
            raise ValueError("Exact original scene/frame scalar identity required")
    scalar = data["scene_anchor"]
    if (type(coefficient) is not float or not math.isfinite(coefficient) or coefficient <= 0
            or np.ma.isMaskedArray(scalar) or not isinstance(scalar, np.ndarray) or scalar.shape != ()
            or scalar.dtype != np.float64 or float(scalar) != coefficient):
        raise ValueError("Exact fixed per-scene coefficient required")
    baseline = {"depth": data["moge_depth"], "points": data["moge_points"], "validity": data["moge_validity"], "K": data["K"]}
    original = validate_baseline(baseline)
    _depth(data["da3_depth"], "DA3", True); _depth(data["anchored_depth"], "Anchored", True)
    if (np.ma.isMaskedArray(data["anchored_validity"]) or not isinstance(data["anchored_validity"], np.ndarray)
            or data["anchored_validity"].dtype != np.bool_ or not np.array_equal(data["anchored_validity"], data["moge_validity"])):
        raise ValueError("Anchored validity must be EXACT native MoGe validity, no dropping")
    if not np.array_equal(data["anchored_depth"], (data["da3_depth"].astype(np.float64) * coefficient).astype(np.float32)):
        raise ValueError("Anchored camera-Z must replay single scene multiplication exactly")
    points = data["anchored_points"]
    if (np.ma.isMaskedArray(points) or not isinstance(points, np.ndarray) or points.shape != (HEIGHT, WIDTH, 3)
            or points.dtype != np.float32 or not np.isfinite(points).all()):
        raise ValueError("Original full-grid anchored FP32 XYZ required")
    # Verify all positive candidate pixels, including baseline-invalid positions;
    # exported validity still remains exactly baseline validity, never expanded.
    anchored = validate_camera_pointmap(data["anchored_depth"], points, np.ones((HEIGHT, WIDTH), np.bool_),
        np.diag([1 / WIDTH, 1 / HEIGHT, 1.]) @ K, K)
    return {"moge": original, "anchored": anchored}


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (root != ROOT or not root.is_dir() or root.resolve() != root
            or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / "run_tudl_anchor_infer/code"
            or code.resolve() != code or not code.is_dir() or Path(executing) != code / SOURCE_FILES[0]):
        raise ValueError("Actual immutable anchor producer dispatch/source namespace required")
    imported = {"infra/tudl_holdout_inputs.py": inputs, "infra/tudl_infer.py": moge_native,
        "infra/object_synthetic_observations.py": moge_assets, "infra/da3_metric_infer.py": da3,
        "infra/da3_metric_acquire.py": da3.acquisition}
    if any(Path(module.__file__).resolve() != code / name for name, module in imported.items()):
        raise ValueError("Every original imported numerical/input helper must resolve inside frozen code")
    if (Path(validate_camera_pointmap.__code__.co_filename).resolve() != code / "src/world_reward/pointmap.py"
            or Path(moge_assets.sha256.__code__.co_filename).resolve() != code / "src/world_reward/data.py"):
        raise ValueError("Original numerical/hash utility imports must resolve inside frozen code")
    return {name: inputs.identity(code / name) for name in SOURCE_FILES}


def load_public(root, code):
    path = code / PIN_FILE; pin_identity = inputs.identity(path)
    pins = inputs.strict_json(path.read_bytes()); records, manifest = inputs.public_inputs(root / BASE / "inputs", pins)
    if inputs.identity(path) != pin_identity: raise ValueError("Independently committed input pins changed")
    return records, {"input_pin_identity": pin_identity, "input_manifest": manifest, "input_pins": pins,
        "public_records": [{key: value for key, value in row.items() if key != "path"} for row in records]}


def moge_bindings(root):
    model, acquisition, asset = moge_assets.model_asset(root)
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    spec = util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Pinned installed MoGe package missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    source = moge_assets.installed_source(directory, inputs.strict_json(direct) if direct else {})
    files = {str(path.relative_to(directory)): moge_assets.identity(path) for path in sorted(directory.rglob("*.py"))}
    if files.get("utils/geometry_torch.py", {}).get("sha256") != moge_native.GEOMETRY_SHA:
        raise ValueError("Exact pinned original native focal solver required")
    return model, {"acquisition_report": acquisition, "model_asset": asset, "model_source": source,
        "python_source_files": files, "source_directory": str(directory), "focal_geometry_source_sha256": moge_native.GEOMETRY_SHA}


def da3_dependency(root):
    path = root / ADDICT_PATH; receipt = moge_assets.identity(path)
    if receipt != {"bytes": 3832, "sha256": ADDICT_SHA}: raise ValueError("Exact immutable native Addict wheel required")
    spec = util.find_spec("addict")
    if spec is None or spec.origin != str(path) + "/addict/__init__.py":
        raise ValueError("Addict must resolve directly from the pinned wheel without installation")
    with zipfile.ZipFile(path) as archive:
        expected = {"addict/__init__.py", "addict/addict.py", *["addict-2.4.0.dist-info/" + name
            for name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt", "RECORD")]}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Pinned native Addict source inventory differs")
        files = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(expected)}
    return {**receipt, "path": str(path), "import_origin": spec.origin, "source_sha256": files}


def load_da3(root, torch, assets, source, dependency, report):
    sys.path.insert(0, str(source / "src"))
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3 import cfg
    from depth_anything_3.utils.io import input_processor, output_processor
    for module in (cfg, input_processor, output_processor):
        if Path(module.__file__).resolve() != source / "src" / Path(*module.__name__.split(".")).with_suffix(".py"):
            raise ValueError("DA3 module resolved outside exact pinned minimal source")
    config = inputs.strict_json((root / da3.acquisition.WEIGHTS / "config.json").read_bytes())
    network = cfg.create_object(OmegaConf.create(config["config"])).eval()
    state = da3.checkpoint_state(load_file(str(root / da3.acquisition.WEIGHTS / "model.safetensors"), device="cpu"), dict(network.state_dict()))
    network.load_state_dict(state, strict=True); del state
    report.update(DA3_assets=assets, DA3_dependency=dependency, checkpoint_states_loaded=da3.STATE_COUNT, checkpoint_load_strict=True)
    return network.cuda(), input_processor.InputProcessor(), output_processor.OutputProcessor()


def da3_frame(torch, network, processor, postprocessor, rgb):
    images, extrinsics, cameras = processor([rgb], extrinsics=None, intrinsics=np.array([K]), process_res=518,
        process_res_method="upper_bound_resize", num_workers=1, sequential=True, print_progress=False)
    if (tuple(images.shape) != (1, 3, *da3.PROCESSED_HW) or images.dtype != torch.float32 or extrinsics is not None
            or tuple(cameras.shape) != (1, 3, 3)):
        raise ValueError("Exact native DA3 preprocessing/grid/camera ABI required")
    processed = cameras[0].numpy(); factor = da3.metric_factor(processed)
    prediction = postprocessor(da3.native_forward(torch, network, images)); depth = prediction.depth
    if depth.shape != (1, *da3.PROCESSED_HW) or depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Original native full positive camera-Z ABI required; no sky/confidence changes")
    resized = da3.resize_metric_depth(torch, depth, factor)
    original = resized[0, 0].cpu().numpy(); torch.cuda.synchronize(); _depth(original, "DA3", True)
    return original.copy(), {"processed_camera_K": processed.tolist(), "metric_depth_factor": factor}


def read_rgb(record):
    from PIL import Image
    with Image.open(record["path"]) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
            raise ValueError("Exact original holdout RGB PNG640x480 required")
        rgb = np.asarray(image).copy()
    return rgb, hashlib.sha256(rgb.tobytes()).hexdigest()


def seed(torch):
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)


def verify_bindings(root, code, revision, report):
    """Full byte provenance recheck, including failed numerical runs when bound."""
    if bound_source(root, code, revision, Path(__file__)) != report["source_helpers"]:
        raise ValueError("Frozen producer/helper closure changed")
    if "input_pins" in report:
        _, public = load_public(root, code)
        if any(report.get(key) != value for key, value in public.items()):
            raise ValueError("Frozen public RGB/independent pins changed")
    if "MoGe_bindings" in report and moge_bindings(root)[1] != report["MoGe_bindings"]:
        raise ValueError("Original MoGe model/source/receipt changed")
    if "DA3_assets" in report and da3.pinned_assets(root)[0] != report["DA3_assets"]:
        raise ValueError("Original DA3 full model/source/receipt changed")
    if "DA3_dependency" in report and da3_dependency(root) != report["DA3_dependency"]:
        raise ValueError("Original imported native dependency changed")
    report["source_assets_after_reverified"] = True


def run(root, code, revision, output, report, persist):
    helpers = bound_source(root, code, revision, Path(__file__))
    records, public = load_public(root, code)
    model_path, moge_binding = moge_bindings(root)
    assets, source = da3.pinned_assets(root); dependency = da3_dependency(root)
    report.update(source_helpers=helpers, **public, MoGe_bindings=moge_binding,
        DA3_assets=assets, DA3_dependency=dependency, phase="MoGe_model_load"); persist()
    import torch
    from moge.model import v2 as native_module
    from moge.utils import geometry_torch
    if not torch.cuda.is_available(): raise RuntimeError("Actual remote CUDA required; no CPU/local fallback")
    if native_module.recover_focal_shift is not geometry_torch.recover_focal_shift:
        raise ValueError("Exact original native focal callback alias required")
    seed(torch); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    report.update(torch=torch.__version__, CUDA=torch.version.cuda)
    network = native_module.MoGeModel.from_pretrained(str(model_path)).cuda().eval()
    torch.cuda.reset_peak_memory_stats(); baselines = []; decoded = []; originals = []
    fov = float(np.degrees(2 * np.arctan(WIDTH / (2 * K[0, 0]))))
    for record in records:
        report.update(phase="MoGe_fixed_prior", active_file=record["file"]); persist()
        rgb, digest = read_rgb(record); tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
        seed(torch); report["MoGe_attempts"] += 1; persist()
        prediction = moge_native.checked_infer(torch, native_module, network, tensor, fov,
            report["native_focal_solver_calls"], {"file": record["file"], "scene_id": record["scene_id"],
                "frame_id": record["frame_id"], "pass": "fixed_prior"})
        report["MoGe_returns"] += 1
        depth, points, validity, normalized = (prediction[name][0].detach().cpu().numpy() for name in ("depth", "points", "mask", "intrinsics"))
        arrays, checks = moge_native.camera_arrays(depth, points, validity, normalized, K)
        validate_baseline(arrays); torch.cuda.synchronize(); report["MoGe_calls_completed"] += 1
        baselines.append({key: value.copy() for key, value in arrays.items()}); decoded.append(digest)
        originals.append(array_identities(arrays)); persist(); del prediction, tensor, arrays
    del network; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    report.update(phase="DA3_model_load", MoGe_model_released_before_DA3_load=True); persist()
    network, processor, postprocessor = load_da3(root, torch, assets, source, dependency, report)
    raw_depths = []; ratios = []; details = []
    for index, record in enumerate(records):
        report.update(phase="DA3_native", active_file=record["file"]); persist()
        rgb, digest = read_rgb(record)
        if digest != decoded[index]: raise ValueError("Both backends must consume exactly identical decoded RGB")
        report["DA3_attempts"] += 1; persist()
        depth, detail = da3_frame(torch, network, processor, postprocessor, rgb)
        report["DA3_returns"] += 1; report["DA3_calls_completed"] += 1
        diagnostic = border_ratio(baselines[index]["depth"], baselines[index]["validity"], depth)
        diagnostic.update(scene_id=record["scene_id"], frame_id=record["frame_id"])
        raw_depths.append(depth); ratios.append(diagnostic); details.append(detail)
        report["ratio_diagnostics"] = ratios.copy(); persist()
    del network, processor, postprocessor; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    coefficients = scene_anchors(ratios)
    report.update(phase="immutable_output", scene_anchor_coefficients=coefficients,
        scene_ratio_medians=[[row["ratio_median"] for row in ratios[start:start + 4]] for start in (0, 4, 8)])
    for index, record in enumerate(records):
        coefficient = coefficients[record["scene_id"] - 1]
        arrays = candidate_arrays(baselines[index], raw_depths[index], coefficient, record["scene_id"], record["frame_id"])
        if array_identities(baselines[index]) != originals[index]: raise ValueError("Unchanged MoGe baseline byte identity lost")
        checks = validate_prediction_arrays(arrays, record["scene_id"], record["frame_id"], coefficient)
        path = output / (Path(record["file"]).stem + ".npz")
        with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
        path.chmod(0o444)
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        validate_prediction_arrays(saved, record["scene_id"], record["frame_id"], coefficient)
        identities = array_identities(arrays)
        if array_identities(saved) != identities: raise ValueError("Persisted full prediction arrays did not replay byte-identically")
        report["outputs"].append({"file": path.name, "scene_id": record["scene_id"], "frame_id": record["frame_id"],
            **inputs.identity(path), "rgb_sha256": record["sha256"], "decoded_RGB_sha256": decoded[index],
            "arrays": identities, "original_MoGe_arrays": originals[index], "pointmap_checks": checks,
            "scene_anchor": coefficient, "border_proxy": ratios[index], **details[index]})
        persist()
    if (load_public(root, code) != (records, public) or bound_source(root, code, revision, Path(__file__)) != helpers
            or moge_bindings(root) != (model_path, moge_binding) or da3.pinned_assets(root) != (assets, source)
            or da3_dependency(root) != dependency):
        raise ValueError("Original public inputs, source helpers or native assets changed")
    calls = report["native_focal_solver_calls"]
    if (any(report[key] != 12 for key in ("MoGe_attempts", "MoGe_returns", "MoGe_calls_completed", "DA3_attempts", "DA3_returns", "DA3_calls_completed"))
            or len(calls) != 12 or len(report["outputs"]) != 12
            or any(call.get("file") != record["file"] or call.get("scene_id") != record["scene_id"]
                or call.get("frame_id") != record["frame_id"] or call.get("pass") != "fixed_prior"
                or call.get("focal_prior_supplied") is not True or call.get("original_solver_returned") is not True
                or type(call.get("native_nearest64_valid_pixels")) is not int or call["native_nearest64_valid_pixels"] < 2
                for call, record in zip(calls, records))):
        raise ValueError("Exactly twelve validated actual native calls per backend required")
    if {path.name for path in output.iterdir()} != {"report.json", *[row["file"] for row in report["outputs"]]}:
        raise ValueError("Only twelve frozen NPZs and one producer receipt allowed")
    for row in report["outputs"]:
        if inputs.identity(output / row["file"]) != {key: row[key] for key in ("sha256", "bytes")}:
            raise ValueError("Immutable complete prediction artifact changed")
    report.update(status="pass", phase="complete", outputs_completed=12, original_frame_coverage_verified=True,
        sources_assets_rechecked=True, saved_arrays_reread_verified=True, baseline_MoGe_bytes_preserved=True,
        actual_native_inference=True, peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote Linux CUDA inference with network none required")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    helpers = bound_source(root, code, revision, Path(__file__))
    output = root / BASE / "anchor_predictions_v1"
    if (image != IMAGE_ID or output.resolve() != output or not output.is_dir() or any(output.iterdir())):
        raise ValueError("Exact original VM02 image and exclusively reserved canonical prediction output required")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision,
        "image_id": image, "script_sha256": helpers[SOURCE_FILES[0]]["sha256"], "source_helpers": helpers,
        "budget_seconds": BUDGET, "network": "none", "private_truth_read": False, "challenge_inputs_used": False,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "training_overlap_verified": False, "full_license_closure_verified": False, "metric_scale_accuracy_verified": False,
        "calibration_accuracy_verified": False, "independent_scenes_or_objects": False, "same_development_scenes_and_objects": True,
        "human_or_object_geometry_changed": False, "per_frame_anchor_fit": False, "offset_fit": False,
        "semantic_background_exclusion_performed": False, "border_is_background_proxy_only": True,
        "anchor_rule": "median_of_four_per_image_median_MoGe_Z_div_DA3_Z_on_fixed_10percent_border_per_scene",
        "border_definition": "xx<64_or_xx>=576_or_yy<48_or_yy>=432",
        "minimum_border_pairs": MIN_BORDER_PAIRS, "minimum_border_coverage": MIN_BORDER_COVERAGE,
        "fixed_camera_K": K.tolist(), "prediction_pixel_convention": "pixel_centres_plus_0.5",
        "candidate_validity": "exact_unchanged_native_MoGe_validity_no_drop_or_expansion",
        "apply_mask": False, "force_projection": True, "native_camera_support_verified": False,
        "native_DA3_arguments": da3.NATIVE_ARGUMENTS, "checkpoint_states_loaded": 0, "checkpoint_load_strict": False,
        "native_sky_correction_unchanged": True, "native_sky_threshold": .3, "native_sky_quantile": .99,
        "metric_depth_scaling_applied_once": True, "metric_depth_factor_rule": "actual_processed_focal_mean_divided_by_300",
        "postprocess": "metric_camera_Z_then_bilinear_align_corners_false_then_one_scene_anchor_then_fixed_K_plus_0.5",
        "confidence_validity_exclusion": False, "pointmap_geometry_filled": False, "seed": 0,
        "RNG_seed_reset_before_each_forward": True, "native_focal_solver_calls": [], "outputs": [], "ratio_diagnostics": [],
        "MoGe_attempts": 0, "MoGe_returns": 0, "MoGe_calls_completed": 0, "DA3_attempts": 0, "DA3_returns": 0, "DA3_calls_completed": 0}
    started = time.perf_counter(); path = output / "report.json"; post_error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole public TUD-L border anchor inference exceeded900s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, code, revision, output, report, persist)
            report["native_camera_support_verified"] = True
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            try:
                verify_bindings(root, code, revision, report)
            except Exception as caught:
                report.update(status="fail", source_assets_after_reverified=False,
                    source_postcheck_error_type=type(caught).__name__, source_postcheck_error=str(caught))
                post_error = caught
            finally:
                signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
                persist(); path.chmod(0o400)
            if post_error is not None: raise post_error


if __name__ == "__main__": main()

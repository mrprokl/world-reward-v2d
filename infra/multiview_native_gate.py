"""Offline native MV imports/configured preprocessing only; no model constructor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

from world_reward.multiview_representation import validate_camera_pointmap, validate_ssi_roundtrip

PIN = "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"
OBJECTS_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
TARGET = "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap"


def parse_args(argv=None):
    return argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)


def _identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def validate_source(root):
    root = Path(root)
    manifest = root / "results/multiview-source.json"
    data = json.loads(manifest.read_text())
    source = root / "vendor/mv-sam3d" / PIN
    expected = {"stage": "pinned_mv_sam3d_source_acquisition", "status": "pass", "revision": PIN,
                "repository": "https://github.com/devinli123/MV-SAM3D.git", "path": str(source),
                "source_git_clean_verified": True, "source_read_only": True,
                "weights_downloaded": False, "challenge_assets_downloaded": False}
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Require exact pinned source acquisition manifest")
    seen = set()
    for record in data["files"]:
        name = record["path"]
        if not isinstance(name, str):
            raise RuntimeError("Source inventory paths must be strings")
        path = source / name
        if (name in seen or Path(name).is_absolute() or ".." in Path(name).parts
                or path.is_symlink() or path.resolve() != path.absolute() or not path.is_file()
                or record["bytes"] != path.stat().st_size or _identity(path)["sha256"] != record["sha256"]):
            raise RuntimeError("Source file path/hash/size inventory mismatch")
        seen.add(name)
    required = {"LICENSE", "sam3d_objects/pipeline/inference_pipeline.py",
                "sam3d_objects/pipeline/inference_pipeline_pointmap.py", "sam3d_objects/data/dataset/tdfy/preprocessor.py"}
    python_files = {str(p.relative_to(source)) for p in (source / "sam3d_objects").rglob("*.py")}
    if not required <= seen or python_files != {p for p in seen if p.endswith(".py")}:
        raise RuntimeError("Incomplete or augmented native Python source inventory")
    return source, _identity(manifest)


def procedural_fixture():
    height, width = 48, 64
    y, x = np.mgrid[:height, :width]
    k = np.array([[100., 0., width / 2], [0., 100., height / 2], [0., 0., 1.]])
    z = 2.5 + x * .001 + y * .002
    points = np.stack(((x + .5 - k[0, 2]) / 100 * z, (y + .5 - k[1, 2]) / 100 * z, z)).astype(np.float32)
    mask = (y >= 10) & (y < 38) & (x >= 15) & (x < 49)
    rgba = np.stack((x * 3, y * 4, (x + y) * 2, mask * 255), axis=-1).astype(np.uint8)
    return rgba, points, k, mask


def _preprocessor_specs(config, load):
    ss = config.get("ss_preprocessor")
    if ss is None:
        ss = load(config["ss_generator_config_path"])["tdfy"]["val_preprocessor"]
    slat = config.get("slat_preprocessor")
    if slat is None:
        raise RuntimeError("Require explicit checkpoint SLAT preprocessor, no invented default")
    return ss, slat


def _configured_preprocess(pipe, torch, preprocessor, rgba, pointmap):
    if type(preprocessor).__module__ != "sam3d_objects.data.dataset.tdfy.preprocessor":
        raise RuntimeError("Require actual pinned native PreProcessor instance")
    original = preprocessor._normalize_pointmap
    normalization = []
    def trace(points, mask, normalizer, scale=None, shift=None):
        result = original(points, mask, normalizer, scale, shift)
        if points is not None:
            metric, normalized = points.detach().cpu().numpy(), result[0].detach().cpu().numpy()
            valid = np.isfinite(metric).all(0)
            if preprocessor.normalize_pointmap:
                scale_value = result[1].detach().cpu().numpy().reshape(-1)
                scale_value = float(scale_value[0]) if scale_value.size == 1 else scale_value
                normalization.append(validate_ssi_roundtrip(metric, normalized,
                    scale_value, result[2].detach().cpu().numpy().reshape(3), valid))
            else:
                if not np.array_equal(metric, normalized, equal_nan=True):
                    raise RuntimeError("Disabled native normalization altered metric pointmap")
                normalization.append({"normalization_disabled_points_unchanged": True})
        return result
    preprocessor._normalize_pointmap = trace
    try:
        output = pipe.preprocess_image(rgba, preprocessor, pointmap=pointmap)
    finally:
        preprocessor._normalize_pointmap = original
    if pointmap is not None:
        if "pointmap" not in output or not normalization:
            raise RuntimeError("Configured SS preprocessing dropped supplied pointmap")
        # Probe the actual native joint transform with identical coordinate-ID
        # channels; individual RGB/point interpolation may intentionally differ.
        height, width = rgba.shape[:2]
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        probe = torch.stack((x, y, x + y)).float()
        mask = torch.tensor(rgba[..., 3] > 0).float()[None]
        rgb, cropped_mask, points = preprocessor._preprocess_image_mask_pointmap(probe, mask, probe.clone())
        if rgb.shape != points.shape or cropped_mask.shape[-2:] != rgb.shape[-2:]:
            raise RuntimeError("Native joint crop produced mismatched RGB/mask/point grids")
        finite = torch.isfinite(points).all(0)
        if not finite.any() or float(torch.abs(rgb[:, finite] - points[:, finite]).max()) > 1e-5:
            raise RuntimeError("Native joint crop/resize coordinate-ID registration failed")
    return {"normalization_checks": normalization, "normalize_pointmap": bool(preprocessor.normalize_pointmap),
            "output_shapes": {key: list(value.shape) for key, value in output.items()},
            "individual_interpolation_equivalence_verified": False}


def run_native(root, source, diagnostics):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Require CUDA for actual native import, no CPU substitute")
    original_hub = torch.hub.load
    def forbidden_hub(*args, **kwargs):
        raise RuntimeError("Preprocessing-only gate must not request Torch Hub models")
    torch.hub.load = forbidden_hub
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
        config_path = root / "weights/sam3d/hf-download/checkpoints/pipeline.yaml"
        acquisition = json.loads((root / "results/weights-acquisition.json").read_text())
        records = [r for r in acquisition["assets"] if r["repo_id"] == "facebook/sam-3d-objects"]
        if len(records) != 1 or records[0]["revision"] != OBJECTS_REVISION:
            raise RuntimeError("Require existing pinned Objects acquisition")
        diagnostics["pipeline_config"] = _identity(config_path)
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        diagnostics["config_constructor_target_not_instantiated"] = config.get("_target_")
        workspace = config_path.parent
        configs = []
        def load(name):
            path = workspace / name
            if not path.is_file() or not path.resolve().is_relative_to(workspace.resolve()):
                raise RuntimeError("Model config dependency must already exist inside checkpoint workspace")
            configs.append(_identity(path))
            return OmegaConf.load(path)
        ss, slat = _preprocessor_specs(config, load)
        diagnostics["configured_condition_input_keys"] = config.get("ss_condition_input_mapping", ["image"])
        diagnostics["preprocessor_config_sha256"] = hashlib.sha256(json.dumps([ss, slat], sort_keys=True, default=str).encode()).hexdigest()
        # Deliberately bypass the heavy constructor: native methods, actual native
        # configured preprocessors, but no checkpoint/models/conditioners loaded.
        pipe = object.__new__(InferencePipelinePointMap)
        pipe.device, pipe.dtype = torch.device("cuda"), InferencePipelinePointMap._get_dtype(config.get("dtype", "bfloat16"))
        pipe.clip_pointmap_beyond_scale = config.get("clip_pointmap_beyond_scale")
        pipe.depth_model = forbidden_hub
        pipe.ss_preprocessor, pipe.slat_preprocessor = instantiate(ss), instantiate(slat)
        rgba, points, k, mask = procedural_fixture()
        camera = validate_camera_pointmap(points, k, np.ones_like(mask))
        computed = pipe.compute_pointmap(rgba, torch.tensor(points))
        native = computed["pointmap"].detach().cpu().numpy()
        expected = points * np.array([-1., -1., 1.])[:, None, None]
        if native.shape != points.shape or not np.allclose(native, expected, rtol=0, atol=1e-6):
            raise RuntimeError("Native compute_pointmap is not exactly one XY flip on same grid")
        inferred = computed.get("intrinsics")
        if inferred is None or not torch.isfinite(inferred).all():
            raise RuntimeError("Native intrinsic inference failed even on procedural rays")
        diagnostics.update({"config_dependencies": configs, "original_ray_check": camera.report,
                            "native_flip_max_error_m": float(np.abs(native - expected).max()),
                            "inferred_intrinsics_authoritative": False})
        diagnostics["ss_preprocess"] = _configured_preprocess(pipe, torch, pipe.ss_preprocessor, rgba, computed["pointmap"])
        diagnostics["slat_preprocess"] = _configured_preprocess(pipe, torch, pipe.slat_preprocessor, rgba, None)
        imported = {}
        for name, module in sys.modules.copy().items():
            if name == "sam3d_objects" or name.startswith("sam3d_objects."):
                file = getattr(module, "__file__", None)
                if file:
                    path = Path(file).resolve()
                    if not path.is_relative_to(source.resolve()):
                        raise RuntimeError("Mixed installed/vendor sam3d_objects import tree")
                    imported[name] = _identity(path)
        diagnostics["imported_source"] = imported
        torch.cuda.synchronize()
    finally:
        torch.hub.load = original_hub


def main(argv=None):
    parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable source revision")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    output = root / "results/multiview-native-preprocess-gate-v2.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Native gate report is frozen")
    report = {"stage": "native_mv_sam3d_import_config_preprocess", "status": "fail", "code_revision": revision,
              "vendor_revision": PIN, "image_id": os.environ.get("WR_IMAGE_ID"), "script": _identity(Path(__file__)),
              "procedural_inputs_only": True, "challenge_inputs_used": False, "ground_truth_used": False,
              "models_loaded": False, "native_constructor_verified": False, "native_conditioner_verified": False,
              "native_dynamics_verified": False, "shape_generation_implemented": False,
              "submission_eligible": False, "license_status": "SAM_custom_competition_eligibility_unresolved"}
    started = time.perf_counter()
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Native preprocessing gate exceeded 120 seconds")))
    signal.alarm(120)
    try:
        source, identity = validate_source(root)
        report["source_manifest"] = identity
        run_native(root, source, report)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("stage", "status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()

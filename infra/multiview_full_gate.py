"""Offline native single/three-view decoded proposals; execution, not accuracy.

Own rendered RGB/masks/camera pointmaps are oracle synthetic observations. The
generating mesh and poses are never model inputs. No challenge input, shape
selection, evaluation alignment, entropy fusion or submission adoption occurs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time

import numpy as np

import multiview_native_gate as prep
import multiview_ss_gate as ss


# Independent HF metadata at the already acquired Objects revision; no download.
CHECKPOINTS = {
    "ss_generator": (ss.SS_SHA256, ss.SS_BYTES),
    "slat_generator": ("91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb", 4906537684),
    "ss_decoder": ("6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340", 147609242),
    "slat_decoder_gs": ("f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111", 171476155),
    "slat_decoder_gs_4": ("731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186", 170269801),
    "slat_decoder_mesh": ("85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33", 363726862),
}
YAMLS = {
    "pipeline": ("53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c", 3548),
    "ss_generator": ("3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd", 5076),
    "slat_generator": ("53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e", 1986),
    "ss_decoder": ("baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae", 244),
    "slat_decoder_gs": ("53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670", 576),
    "slat_decoder_gs_4": ("3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48", 575),
    "slat_decoder_mesh": ("8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e", 300),
}
WIDTH, HEIGHT, RADII = 256, 192, np.array([.29, .17, .23])
K = np.array([[240., 0., 128.], [0., 240., 96.], [0., 0., 1.]])


def parse_args(argv=None):
    return argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)


def constructor_config(config, workspace):
    """Remove only the unneeded depth constructor; preserve learned settings."""
    if not isinstance(config, dict) or config.get("_target_") != prep.TARGET:
        raise RuntimeError("Require exact native PointMap pipeline target")
    for name in CHECKPOINTS:
        if config.get(name + "_config_path") != name + ".yaml" or config.get(name + "_ckpt_path") != name + ".ckpt":
            raise RuntimeError("Unexpected native checkpoint/config path")
    if any(config.get(key) is not None for key in ("ss_encoder_config_path", "ss_encoder_ckpt_path")):
        raise RuntimeError("Unexpected optional SS encoder")
    result = copy.deepcopy(config)
    result.update(depth_model=None, compile_model=False, workspace_dir=str(workspace),
                  rendering_engine="pytorch3d", layout_post_optimization_method=None)
    return result


def validate_ss(root, source_identity, image):
    preprocess = ss.validate_preprocess(root, source_identity, image)
    path = Path(root) / "results/multiview-ss-one-call-gate.json"
    data = json.loads(path.read_text())
    expected = {"stage": "native_mv_sam3d_partial_ss_one_call", "status": "pass", "vendor_revision": prep.PIN,
                "image_id": image, "procedural_inputs_only": True, "ground_truth_used": False,
                "challenge_inputs_used": False, "native_conditioner_verified": True, "native_dynamics_verified": True,
                "shape_generation_implemented": False, "entropy_fusion_verified": False}
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Require actual passed SS dynamics gate on same source/image")
    workspace = Path(root) / "weights/sam3d/hf-download/checkpoints"
    script, current_script = data.get("script", {}), ss._identity(Path(ss.__file__))
    # Producer and consumer run in distinct immutable snapshot directories.
    if (data.get("preprocess_gate") != preprocess
            or any(script.get(key) != current_script[key] for key in ("sha256", "bytes"))
            or data.get("ss_config") != ss._identity(workspace / "ss_generator.yaml")):
        raise RuntimeError("SS prerequisite script/config/preprocessing changed")
    checkpoint = data.get("ss_checkpoint", {})
    if (checkpoint.get("path") != str(workspace / "ss_generator.ckpt")
            or checkpoint.get("sha256") != ss.SS_SHA256 or checkpoint.get("bytes") != ss.SS_BYTES):
        raise RuntimeError("SS prerequisite checkpoint binding changed")
    return ss._identity(path)


def observation_arrays(depth, mask, rotation, translation):
    """Synthesis only: registered RGB and OpenCV rays, finite scene background."""
    if np.ma.isMaskedArray(depth) or np.ma.isMaskedArray(mask):
        raise ValueError("Synthetic support must be explicit, not a masked array")
    depth, mask = np.asarray(depth), np.asarray(mask)
    if (depth.shape != (HEIGHT, WIDTH) or mask.shape != depth.shape or mask.dtype != np.bool_
            or not mask.any() or not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any()):
        raise ValueError("Invalid synthetic visible camera-Z/mask")
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    z = np.where(mask, depth, 4.)
    xyz = np.stack(((x + .5 - 128) / 240 * z, (y + .5 - 96) / 240 * z, z), axis=-1)
    canonical = (xyz - translation) @ rotation
    rgb = np.broadcast_to(np.array([28, 35, 42], dtype=np.uint8), (HEIGHT, WIDTH, 3)).copy()
    rgb[mask] = np.rint(128 + 100 * np.clip(canonical[mask] / RADII, -1, 1)).astype(np.uint8)
    points = np.moveaxis(xyz, -1, 0).astype(np.float32)
    prep.validate_camera_pointmap(points, K, np.ones_like(mask), frame_index=0)
    return rgb, points


def procedural_views(report):
    from scipy.spatial.transform import Rotation
    import trimesh
    from camera_render import raster_camera_mesh_batch
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices, faces = np.asarray(mesh.vertices) * RADII, np.asarray(mesh.faces)
    rotations = Rotation.from_rotvec([[.1, -.2, .05], [-.15, .75, .1], [.2, 1.4, -.1]]).as_matrix()
    translations = np.array([[-.03, .02, 1.4], [.04, -.02, 1.5], [-.02, .01, 1.45]])
    masks_gpu, depths_gpu = raster_camera_mesh_batch(np.stack([vertices @ r.T + t for r, t in zip(rotations, translations)]), faces, K, WIDTH, HEIGHT)
    masks, depths = masks_gpu.cpu().numpy(), depths_gpu.cpu().numpy()
    del masks_gpu, depths_gpu
    images, maps = [], []
    report["observations"] = []
    for index, (z, mask, r, t) in enumerate(zip(depths, masks, rotations, translations)):
        rgb, points = observation_arrays(z, mask, r, t)
        images.append(rgb); maps.append(points)
        report["observations"].append({"frame_index": index, "visible_pixels": int(mask.sum()),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "pointmap_sha256": hashlib.sha256(points.tobytes()).hexdigest()})
    report["synthetic_fixture"] = {"width": WIDTH, "height": HEIGHT, "K": K.tolist(), "radii_m": RADII.tolist(),
        "oracle_observations_for_integration_only": True, "generating_mesh_or_pose_supplied_to_model": False}
    return images, list(masks), maps


def export_proposal(result, destination, name):
    mesh = result.get("glb")
    if mesh is None or not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise RuntimeError("Native decoder returned no finite nonempty GLB proposal")
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(mesh.vertices)):
        raise RuntimeError("Decoded mesh face indices invalid")
    pose = {}
    for key, size in (("rotation", 4), ("translation", 3), ("scale", 3)):
        value = result[key].detach().cpu().numpy()
        if value.size != size or not np.isfinite(value).all() or (key == "scale" and (value <= 0).any()):
            raise RuntimeError("Native anchor pose invalid")
        pose[key] = value.reshape(-1).tolist()
    if not np.isclose(np.linalg.norm(pose["rotation"]), 1., atol=1e-3):
        raise RuntimeError("Native anchor quaternion not unit WXYZ")
    path = destination / (name + ".glb")
    with path.open("xb") as handle:
        handle.write(mesh.export(file_type="glb"))
    with (destination / (name + "-pose.json")).open("x") as handle:
        json.dump({"schema": "native_mv_proposal_unadopted", "anchor_pose": pose,
                   "pose_applied_to_glb": False, "coordinate_conversion_verified": False}, handle, allow_nan=False)
    return {"artifact": ss._identity(path), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "anchor_pose": pose, "quality_evaluated": False}


def run(root, source, report, started):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from body_smoke import _pinned_checkout
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    workspace = root / "weights/sam3d/hf-download/checkpoints"
    records = {}
    for suffix, inventory in ((".ckpt", CHECKPOINTS), (".yaml", YAMLS)):
        for name, expected in inventory.items():
            path = workspace / (name + suffix)
            if path.is_symlink() or path.resolve() != path.absolute():
                raise RuntimeError("Checkpoint/config path must be nonsymlink canonical")
            record = ss._identity(path)
            if (record["sha256"], record["bytes"]) != expected:
                raise RuntimeError("Independent pinned checkpoint/config integrity failed")
            records[name + suffix] = record
    report["model_assets"] = records
    config = constructor_config(OmegaConf.to_container(OmegaConf.load(workspace / "pipeline.yaml"), resolve=True), workspace)
    report["effective_pipeline_config"] = config
    hub = root / "weights/sam3d/torch_home/hub"
    repository = hub / "facebookresearch_dinov2_main"
    _pinned_checkout(repository, ss.DINO_REVISION)
    auxiliary = json.loads((root / "results/auxiliary-assets.json").read_text())
    dino = [r for r in auxiliary["checkpoints"] if "_reg4_" in r["filename"]]
    if len(dino) != 2 or {r["filename"] for r in dino} != {"dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth"}:
        raise RuntimeError("Require both acquired DINO register checkpoint records")
    for record in dino:
        if ss._identity(hub / "checkpoints" / record["filename"])["sha256"] != record["sha256"]:
            raise RuntimeError("DINO checkpoint integrity failed")
    original = torch.hub.load
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    calls, depth_calls = [], []
    def local_load(repo, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if repo not in ("facebookresearch/dinov2", "facebookresearch/dinov2:main") or model not in ("dinov2_vitl14_reg", "dinov2_vitb14_reg"):
            raise RuntimeError("Unexpected Torch Hub model request")
        calls.append(model); kwargs["source"] = "local"
        return original(str(repository), *args, **kwargs)
    def forbidden_depth(*args, **kwargs):
        depth_calls.append(True)
        raise RuntimeError("Every view requires external pointmap; no depth model")
    torch.hub.load = local_load
    try:
        torch.cuda.reset_peak_memory_stats()
        pipe = instantiate(config)
        if type(pipe).__module__ != "sam3d_objects.pipeline.inference_pipeline_pointmap" or pipe.compile_model or pipe.depth_model is not None:
            raise RuntimeError("Full native constructor or depth/compile override incorrect")
        pipe.depth_model = forbidden_depth
        report.update(native_constructor_verified=True, models_loaded=True, dino_revision=ss.DINO_REVISION, dino_checkpoints=dino)
        images, masks, maps = procedural_views(report)
        destination = root / "results/multiview-full-proposals"
        destination.mkdir()
        generation_start = time.perf_counter()
        signal.alarm(max(1, min(180, int(300 - (generation_start - started)))))
        common = dict(seed=42, stage1_inference_steps=50, stage2_inference_steps=25, decode_formats=["gaussian", "mesh"],
                      use_stage1_distillation=False, use_stage2_distillation=False, stage1_only=False,
                      with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True)
        report["generation_configuration"] = {**common, "ss_weighting": False, "slat_weighting": False}
        report["proposals"] = {}
        for name in ("single", "three_view"):
            ss._seed(torch)
            with torch.no_grad():
                if name == "single":
                    result = pipe.run(images[0], masks[0], pointmap=torch.from_numpy(maps[0]), with_layout_postprocess=False, **common)
                else:
                    result = pipe.run_multi_view(images, masks, view_pointmaps=maps, num_samples=1, mode="multidiffusion",
                                                ss_weighting=False, weighting_config=None, **common)
            report["proposals"][name] = export_proposal(result, destination, name)
            del result
            torch.cuda.synchronize()
            if torch.cuda.max_memory_allocated() > 80 * 1024**3 or depth_calls:
                raise RuntimeError("Frozen 80GiB or external-depth-only gate failed")
            print(json.dumps({"proposal": name, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        report["generation_seconds"] = time.perf_counter() - generation_start
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if report["generation_seconds"] > 180 or report["peak_cuda_allocated_bytes"] > 80 * 1024**3 or depth_calls:
            raise RuntimeError("Frozen generation180s/80GiB or external-depth-only gate failed")
        imported = {name: prep._identity(Path(module.__file__)) for name, module in sys.modules.copy().items()
                    if (name == "sam3d_objects" or name.startswith("sam3d_objects.")) and getattr(module, "__file__", None)}
        if any(not Path(r["path"]).resolve().is_relative_to(source.resolve()) for r in imported.values()):
            raise RuntimeError("Mixed vendor/installed native imports")
        report.update(imported_source=imported, hub_requests=calls, depth_call_count=len(depth_calls), native_decode_verified=True)
    finally:
        torch.hub.load = original
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])


def main(argv=None):
    parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux network-none container")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("Require immutable source/image digests")
    output = root / "results/multiview-full-execution-gate.json"
    if output.exists() or (root / "results/multiview-full-proposals").exists():
        raise FileExistsError("Full native reports/proposals are frozen")
    report = {"stage": "native_mv_sam3d_full_execution_only", "status": "fail", "code_revision": revision,
              "vendor_revision": prep.PIN, "objects_revision": prep.OBJECTS_REVISION, "image_id": image,
              "script": ss._identity(Path(__file__)), "procedural_inputs_only": True, "challenge_inputs_used": False,
              "external_ground_truth_used": False, "synthetic_oracle_observations": True, "gt_pose_input": False,
              "native_constructor_verified": False, "native_decode_verified": False, "models_loaded": False,
              "depth_model_constructed": False, "entropy_fusion_verified": False, "accuracy_evaluated": False,
              "adoption_authorized": False, "submission_eligible": False,
              "license_status": "SAM_custom_competition_eligibility_unresolved"}
    started = time.perf_counter()
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Frozen total300s/generation180s budget exceeded")))
    signal.alarm(300)
    try:
        source, identity = prep.validate_source(root)
        report["ss_prerequisite"] = validate_ss(root, identity, image)
        run(root, source, report, started)
        if time.perf_counter() - started > 300:
            raise TimeoutError("Full gate exceeded total300s")
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()

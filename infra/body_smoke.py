"""Azure-only, offline SAM 3D Body gate on three fixed Track 1 frames.

Run in the pinned CARI image with ``docker --network none``. Only automatic
human masks are accepted. Outputs are raw, per-image model predictions, not a
submission: no identity fitting, temporal optimization, or official converter
is performed. Geometry is in metres; raw MHR parameters retain native units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from world_reward.data import sha256


EPISODE = 15
UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
BODY_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
BODY_PACKAGE = Path("reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body")
PARAMETER_SHAPES = {
    "global_rot": (3,),
    "body_pose_params": (133,),
    "hand_pose_params": (108,),
    "scale_params": (28,),
    "shape_params": (45,),
    "expr_params": (72,),
    "mhr_model_params": (204,),
    "pred_pose_raw": (266,),
    "pred_joint_coords": (127, 3),
    "pred_keypoints_3d": (70, 3),
    "pred_keypoints_2d": (70, 2),
    "pred_global_rots": (127, 3, 3),
    "pred_vertices": (18439, 3),
    "pred_cam_t": (3,),
    "focal_length": (),
}


def _git(path: Path, *arguments: str) -> str:
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1")
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), *arguments],
        text=True, env=environment,
    ).strip()


def _pinned_checkout(path: Path, revision: str) -> None:
    if not path.is_dir() or _git(path, "rev-parse", "HEAD") != revision:
        raise RuntimeError(f"Missing or unpinned source checkout: {path}")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Modified source checkout: {path}")


def _manifest_file(root: Path, manifest: dict, relative: str) -> tuple[Path, str]:
    matches = [record for record in manifest["files"] if record["path"] == relative]
    if len(matches) != 1:
        raise ValueError(f"Input manifest needs exactly one record for {relative}")
    record = matches[0]
    path = root / "data" / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root / "data"):
        raise ValueError("Input file escapes the audited data root")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("Input manifest contains an invalid SHA-256")
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Track 1 input integrity failed: {relative}")
    return path, record["sha256"]


def _source_identity(root: Path) -> dict:
    """Bind the installed inference package to the audited, clean upstream tree."""
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    expected = vendor / BODY_PACKAGE
    installed = Path("/workspace/v2d_sam3d_body/lib/sam_3d_body")
    required_data_sources = {
        "data/__init__.py", "data/transforms/__init__.py",
        "data/transforms/bbox_utils.py", "data/transforms/common.py",
        "data/utils/io.py", "data/utils/prepare_batch.py",
    }
    if {str(path.relative_to(expected)) for path in (expected / "data").rglob("*.py")} != required_data_sources:
        raise RuntimeError("Pinned Body source inventory is incomplete; data here means Python source")
    relative_paths = sorted(path.relative_to(expected) for path in expected.rglob("*.py"))
    if not relative_paths or relative_paths != sorted(
        path.relative_to(installed) for path in installed.rglob("*.py")
    ):
        raise RuntimeError("Installed SAM 3D Body source file set differs from upstream")
    records = {}
    for relative in relative_paths:
        wanted, actual = expected / relative, installed / relative
        if actual.is_symlink() or sha256(wanted) != sha256(actual):
            raise RuntimeError(f"Installed SAM 3D Body source differs: {relative}")
        records[str(relative)] = sha256(actual)
    return {
        "python_files": len(records),
        "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    }


def _validate_inputs(root: Path) -> dict:
    manifest = json.loads((root / "results/input-manifest.json").read_text())
    if (manifest.get("track"), manifest.get("repo_id"), manifest.get("revision")) != (
        "track_1", "nvidia/video_to_data_challenge", DATASET_REVISION,
    ):
        raise ValueError("Require the audited, pinned official Track 1 input manifest")
    episodes_path, _ = _manifest_file(root, manifest, "track_1/meta/episodes.jsonl")
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    matches = [episode for episode in episodes if episode["episode_index"] == EPISODE]
    if len(matches) != 1 or type(matches[0]["length"]) is not int or matches[0]["length"] < 3:
        raise ValueError("Missing/invalid official episode 15 frame count")
    total = matches[0]["length"]
    relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{EPISODE:06d}.mp4"
    video, video_hash = _manifest_file(root, manifest, relative)
    masks_root = root / f"outputs/episode_{EPISODE:06d}/automatic_masks"
    mask_report_path = masks_root / "report.json"
    mask_report = json.loads(mask_report_path.read_text())
    required = {
        "stage": "automatic_masks", "status": "pass", "episode_index": EPISODE,
        "frames": total, "input_track": "track_1", "input_sha256": video_hash,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
    }
    for key, value in required.items():
        if key not in mask_report or mask_report[key] != value:
            raise ValueError(f"Automatic mask provenance failed: {key}")
    for key in ("ground_truth_used", "hand_labeled_test"):
        if mask_report[key] is not False:
            raise ValueError(f"Automatic mask provenance must explicitly forbid {key}")
    prompts = json.loads((masks_root / "prompts.json").read_text())["prompts"]
    if len(prompts) != 2 or {prompt["object_id"] for prompt in prompts} != {0, 1}:
        raise ValueError("Require the automatic person/object SAM2 initializer")
    if any(prompt.get(key) is not None for prompt in prompts for key in (
        "points", "point_labels", "mask_path",
    )):
        raise ValueError("No manual point/mask prompts are permitted")
    human_masks = masks_root / "masks/0"
    if [path.stem for path in sorted(human_masks.glob("*.png"))] != [
        f"{index:06d}" for index in range(total)
    ]:
        raise ValueError("Automatic person masks do not cover every original frame")
    return {
        "video": video, "video_sha256": video_hash, "total_frames": total,
        "indices": [0, total // 2, total - 1], "human_masks": human_masks,
        "mask_report_sha256": sha256(mask_report_path),
        "prompts_sha256": sha256(masks_root / "prompts.json"),
    }


def _body_assets(root: Path) -> tuple[Path, dict]:
    record = json.loads((root / "results/weights-acquisition.json").read_text())
    assets = [asset for asset in record["assets"] if asset["repo_id"] == "facebook/sam-3d-body-dinov3"]
    directory = root / "weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"
    if len(assets) != 1 or assets[0]["revision"] != BODY_REVISION:
        raise RuntimeError("SAM 3D Body acquisition must use the audited HF revision")
    if Path(assets[0]["path"]).resolve() != directory:
        raise RuntimeError("SAM 3D Body acquisition path differs from the runtime contract")
    records = {}
    for relative in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt", "LICENSE"):
        path = directory / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required local SAM 3D Body artifact is missing: {relative}")
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return directory, records


def _install_local_dinov3_loader(torch, repository: Path):
    """Never resolve GitHub/default branches or fetch weights through Torch Hub."""
    _pinned_checkout(repository, DINOV3_REVISION)
    original = torch.hub.load
    calls = []

    def load(repo_or_dir, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov3" or kwargs.get("source") != "github":
            raise RuntimeError(f"Unexpected Torch Hub request in direct Body inference: {repo_or_dir}")
        if kwargs.get("pretrained") is not False:
            raise RuntimeError("Direct Body backbone must use checkpoint weights, not a Hub download")
        calls.append(str(args[0]) if args else "missing_model_name")
        kwargs["source"] = "local"
        return original(str(repository), *args, **kwargs)

    torch.hub.load = load
    return original, calls


def _load_checkpoint_with_asset_buffers(module, state_dict, original_loader, torch, *, explicit_asset_state=None) -> dict:
    """Require all checkpoint state except immutable rig buffers loaded from MHR.

    The official model constructs these buffers from the explicit TorchScript
    asset before loading the network checkpoint. Frozen learned parameters and
    head topology/PCA buffers still must appear in the checkpoint.
    """
    prefixes = ("head_pose.mhr.character_torch.", "head_pose_hand.mhr.character_torch.")
    current = module.state_dict()
    parameters = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers(remove_duplicate=False))
    explicit = {}
    if explicit_asset_state is not None:
        for prefix in ("head_pose.mhr.", "head_pose_hand.mhr."):
            local_keys = {name[len(prefix):] for name in current if name.startswith(prefix)}
            if local_keys != set(explicit_asset_state):
                raise RuntimeError("Constructed MHR submodule state differs from explicit asset inventory")
            for name, value in explicit_asset_state.items():
                key = prefix + name
                if not torch.equal(current[key].cpu(), value.cpu()):
                    raise RuntimeError(f"Constructed MHR differs from independent explicit asset: {key}")
                explicit[key] = current[key].detach().clone()
        for key in ("head_pose.hand_pose_comps_ori", "head_pose_hand.hand_pose_comps_ori"):
            value = current[key]
            if value.shape != (54, 54) or value.requires_grad or not torch.equal(value, torch.eye(54, device=value.device, dtype=value.dtype)):
                raise RuntimeError("Unused original hand PCA copy must be deterministic identity")
            explicit[key] = value.detach().clone()
        key = "backbone.encoder.mask_token"
        token = current[key]
        if token.ndim != 2 or token.shape[0] != 1 or token.shape[1] != module.backbone.encoder.embed_dim or not torch.isfinite(token).all() or torch.count_nonzero(token).item():
            raise RuntimeError("Unused DINO mask token must equal the pinned zero initialization")
        explicit[key] = token.detach().clone()
    unexpected = set(state_dict) - set(current)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)[:8]}")
    missing = set(current) - set(state_dict)
    allowed = {name for name in missing if name.startswith(prefixes) and name in buffers
               and name not in parameters and not buffers[name].requires_grad}
    allowed.update(missing & set(explicit))
    if missing - allowed:
        raise RuntimeError(f"Required model state missing: {sorted(missing - allowed)[:8]}")
    for name in parameters:
        if name in explicit:
            continue
        if name not in state_dict or not torch.isfinite(state_dict[name]).all():
            raise RuntimeError(f"Learned/frozen parameter missing or nonfinite: {name}")
    asset_buffers = {name: value.detach().clone() for name, value in current.items()
                     if name.startswith(prefixes) and name in buffers and name not in parameters}
    asset_buffers.update(explicit)
    for name, asset_value in asset_buffers.items():
        if name in state_dict and not torch.equal(state_dict[name].cpu(), asset_value.cpu()):
            raise RuntimeError(f"Checkpoint contradicts explicit immutable MHR asset: {name}")
    merged = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        merged._metadata = state_dict._metadata
    for name in allowed:
        merged[name] = asset_buffers[name].clone()
    original_loader(module, merged, strict=True)
    after = module.state_dict()
    if any(not torch.equal(after[name].cpu(), value.cpu()) for name, value in asset_buffers.items()):
        raise RuntimeError("Immutable MHR asset buffers changed during checkpoint loading")
    return {"mode": "strict_network_and_head_state_with_explicit_asset_buffer_retention",
            "retained_mhr_asset_buffer_names": sorted(allowed),
            "parameter_tensors_loaded": len(parameters), "unexpected_keys": []}


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Video/model processing is restricted to Azure Linux")
    if {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require an isolated container launched with docker --network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--episode", type=int, choices=[EPISODE], default=EPISODE)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.dont_write_bytecode = True
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "MOMENTUM_ENABLED": "0", "WANDB_MODE": "disabled",
    })
    inputs = _validate_inputs(root)
    source_hashes = _source_identity(root)
    body_directory, asset_hashes = _body_assets(root)
    output_directory = root / f"outputs/episode_{EPISODE:06d}/body_smoke"
    if output_directory.exists():
        raise FileExistsError("Body smoke outputs are frozen; use a clean run directory, not overwrite")
    import cv2
    import numpy as np
    from PIL import Image
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; never silently process challenge images on CPU")
    started = time.perf_counter()
    repository = root / "weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main"
    original_hub_load, hub_calls = _install_local_dinov3_loader(torch, repository)
    try:
        import sam_3d_body
        from sam_3d_body import build_models

        if Path(sam_3d_body.__file__).resolve() != Path(
            "/workspace/v2d_sam3d_body/lib/sam_3d_body/__init__.py"
        ):
            raise RuntimeError("Unexpected SAM 3D Body import location")
        original_state_loader = build_models.load_state_dict
        checkpoint_loading = {}
        independent_asset = torch.jit.load(str(body_directory / "assets/mhr_model.pt"), map_location="cpu").state_dict()

        def strict_state_loader(module, state_dict, strict=False, logger=None):
            checkpoint_loading.update(_load_checkpoint_with_asset_buffers(module, state_dict, original_state_loader, torch,
                                                                          explicit_asset_state=independent_asset))
            original_prepare = module.backbone.encoder.prepare_tokens_with_masks
            def unmasked_rgb_tokens(x, masks=None):
                if masks is not None:
                    raise RuntimeError("The retained zero mask token is allowed only for unmasked DINO RGB tokens")
                return original_prepare(x, masks=None)
            module.backbone.encoder.prepare_tokens_with_masks = unmasked_rgb_tokens

        build_models.load_state_dict = strict_state_loader
        try:
            model, config = sam_3d_body.load_sam_3d_body(
                checkpoint_path=str(body_directory / "model.ckpt"), device="cuda",
                mhr_path=str(body_directory / "assets/mhr_model.pt"),
            )
        finally:
            build_models.load_state_dict = original_state_loader
    finally:
        torch.hub.load = original_hub_load
    if len(hub_calls) != 1 or config.MODEL.BACKBONE.TYPE != hub_calls[0]:
        raise RuntimeError("The audited single DINOv3 backbone load contract changed")
    estimator = sam_3d_body.SAM3DBodyEstimator(model, config)
    faces = np.asarray(estimator.faces)
    if (faces.shape != (36874, 3) or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0 or faces.max() >= 18439
        or np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 0] == faces[:, 2])
        or np.any(faces[:, 1] == faces[:, 2])):
        raise RuntimeError("Native MHR face topology is invalid or differs from the audited release")
    capture = cv2.VideoCapture(str(inputs["video"]))
    if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != inputs["total_frames"]:
        capture.release()
        raise RuntimeError("Video decoder disagrees with official full-frame count")
    arrays = {key: [] for key in PARAMETER_SHAPES}
    frame_records = []
    camera_vertices = []
    try:
        for index in inputs["indices"]:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError("Video seek failed")
            ok, bgr = capture.read()
            if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                raise RuntimeError("Video decode returned a wrong original frame")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask_path = inputs["human_masks"] / f"{index:06d}.png"
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            if mask.ndim != 2 or mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 255]).all():
                raise RuntimeError("Expected a binary original-resolution automatic human mask")
            ys, xs = np.where(mask > 0)
            if len(xs) < 20:
                raise RuntimeError(f"Automatic human mask is absent at fixed frame {index}; no bbox fallback")
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            if min(box[2:] - box[:2]) <= 1:
                raise RuntimeError("Automatic human mask does not define a valid bbox")
            with torch.inference_mode():
                predictions = estimator.process_one_image(
                    # Match CARI's production _prepare_batch, not the generic
                    # estimator wrapper which forwards raw 0/255 PNG values.
                    img=rgb, bboxes=box[None], masks=(mask > 0).astype(np.uint8)[..., None],
                    cam_int=None, inference_type="body",
                )
            if not isinstance(predictions, list) or len(predictions) != 1:
                raise RuntimeError("Expected exactly one body prediction per fixed image")
            prediction = predictions[0]
            for key, shape in PARAMETER_SHAPES.items():
                value = prediction.get(key)
                if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise RuntimeError(f"SAM 3D Body output shape/finite contract failed: {key}")
                arrays[key].append(value.detach().float().cpu().numpy())
            if torch.count_nonzero(prediction["expr_params"]).item() != 0:
                raise RuntimeError("SAM 3D Body facial expressions must be disabled")
            with torch.inference_mode():
                native, _ = model.head_pose.mhr(
                    prediction["shape_params"][None], prediction["mhr_model_params"][None],
                    prediction["expr_params"][None],
                )
                recovered = native[0] * torch.tensor([1., -1., -1.], device=native.device) / 100.
                unit_error = torch.linalg.vector_norm(recovered - prediction["pred_vertices"], dim=-1).max()
            if not torch.isfinite(unit_error) or float(unit_error) > 1e-5:
                raise RuntimeError("Raw MHR forward does not reproduce camera-relative vertices in metres")
            vertices = arrays["pred_vertices"][-1] + arrays["pred_cam_t"][-1][None]
            if arrays["pred_cam_t"][-1][2] <= 0 or not np.isfinite(vertices).all():
                raise RuntimeError("Invalid camera translation/translated native mesh")
            triangles = vertices[faces]
            if np.max(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)) <= 0:
                raise RuntimeError("MHR prediction collapsed to a zero-area mesh")
            h, w = rgb.shape[:2]
            focal = float(arrays["focal_length"][-1])
            if not np.isclose(focal, np.hypot(h, w), rtol=1e-6, atol=1e-4):
                raise RuntimeError("Unexpected camera calibration: require the RGB-size-only default FOV")
            keypoints = arrays["pred_keypoints_3d"][-1] + arrays["pred_cam_t"][-1][None]
            if np.any(keypoints[:, 2] <= 0):
                raise RuntimeError("Predicted human keypoints are behind the camera")
            projected = focal * keypoints[:, :2] / keypoints[:, 2:] + np.array([w / 2., h / 2.])
            projection_error = float(np.linalg.norm(projected - arrays["pred_keypoints_2d"][-1], axis=1).max())
            if not np.isfinite(projection_error) or projection_error > 0.05:
                raise RuntimeError("Native camera projection does not reproduce the model's image keypoints")
            camera_vertices.append(vertices)
            frame_records.append({
                "frame_index": index, "mask_sha256": sha256(mask_path),
                "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "mask_pixels": len(xs), "bbox_xyxy": box.tolist(),
                "native_forward_max_error_m": float(unit_error),
                "projection_max_error_px": projection_error,
            })
    finally:
        capture.release()
    arrays = {key: np.stack(values) for key, values in arrays.items()}
    arrays["vertices_root_camera_m"] = arrays.pop("pred_vertices")
    arrays["vertices_camera_m"] = np.stack(camera_vertices)
    arrays["faces"] = faces.astype(np.int64)
    arrays["frame_index"] = np.asarray(inputs["indices"], dtype=np.int64)
    output_directory.mkdir(parents=True, exist_ok=False)
    predictions_path = output_directory / "predictions.npz"
    with predictions_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    report = {
        "stage": "sam3d_body_three_frame_smoke", "status": "pass", "episode_index": EPISODE,
        "total_video_frames": inputs["total_frames"], "frame_indices": inputs["indices"],
        "input_track": "track_1", "input_sha256": inputs["video_sha256"],
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "network": "none", "inference_type": "body", "camera_intrinsics": "RGB_size_default_FOV",
        "human_mask_id": 0, "prompt_mode": "automatic_mask_and_derived_bbox_no_fallback",
        "model_mask_range": "uint8_0_1_matches_CARI_prepare_batch",
        "vertices": 18439, "faces": 36874, "geometry_units": "metres",
        "geometry_frame": "SAM3D_camera_x_right_y_down_z_forward",
        "translation": "vertices_camera_m = vertices_root_camera_m + pred_cam_t (exactly once)",
        "parameter_format": "raw_sam3d_body_mhr_204_plus_shape_45_not_submission",
        "raw_mhr_geometry_units": "centimetres_before_/100_and_YZ_flip",
        "human_identity_clip_constant": False, "submission_eligible": False,
        "mhr_geometry_forward_verified": True, "challenge_performance_verified": False,
        "body_revision": BODY_REVISION, "body_assets": asset_hashes,
        "asset_hash_assurance": "recorded_local_hashes_after_pinned_acquisition_not_independent_release_hashes",
        "upstream_revision": UPSTREAM_REVISION, "inference_source_identity": source_hashes,
        "dinov3_revision": DINOV3_REVISION, "backbone": hub_calls[0], "checkpoint_loading": checkpoint_loading,
        "mask_report_sha256": inputs["mask_report_sha256"], "prompts_sha256": inputs["prompts_sha256"],
        "frames": frame_records, "predictions_sha256": sha256(predictions_path),
        "script_sha256": sha256(Path(__file__)), "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "elapsed_seconds": time.perf_counter() - started,
        "license": "SAM_3D_Body_Materials_license_not_Apache_wrapper_license",
    }
    with (output_directory / "report.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "stage", "status", "episode_index", "frame_indices", "geometry_units",
        "mhr_geometry_forward_verified", "submission_eligible", "elapsed_seconds",
    )}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        message = str(exc)
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        print(json.dumps({"stage": "sam3d_body_three_frame_smoke", "status": "fail",
                          "error": type(exc).__name__, "message": message}), file=sys.stderr)
        sys.exit(1)

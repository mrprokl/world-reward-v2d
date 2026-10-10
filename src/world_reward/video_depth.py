"""Thin native offline metric-VDA adapter; RGB+FPS only, no fitted alignment."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

CHECKPOINT_BYTES = 116_444_063
CHECKPOINT_SHA256 = "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
API_SHA256 = "208e882d60a41434b3a0d9935a7025b01cce0756c17b3d889c6db0c4c945eb29"
NATIVE_CONFIGURATION = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=True)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_metric_small(source_directory, checkpoint_path):
    """Load audited complete native state offline; never call hub or CLI exports."""
    source, checkpoint = Path(source_directory), Path(checkpoint_path)
    api = source / "video_depth_anything/video_depth.py"
    if (not source.is_dir() or source.resolve() != source or checkpoint.is_symlink()
            or checkpoint.stat().st_size != CHECKPOINT_BYTES or _digest(checkpoint) != CHECKPOINT_SHA256
            or api.is_symlink() or _digest(api) != API_SHA256):
        raise ValueError("Pinned native metric-VDA source/checkpoint required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Native metric-VDA experiment requires Azure CUDA")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("video_depth_anything.video_depth")
        if Path(module.__file__).resolve() != api:
            raise ValueError("Another cached VDA source module is already loaded")
        network = module.VideoDepthAnything(**NATIVE_CONFIGURATION)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(state, strict=True)
        network = network.to("cuda").eval()
    finally:
        sys.path.pop(0)
    return network


def infer_metric_video(network, frames, fps, *, fp32=False):
    """Full native offline timeline; upstream repeats boundary padding, not observations."""
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
            or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1
            or type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0
            or type(fp32) is not bool):
        raise ValueError("Nonempty native RGB uint8[T,H,W,3], positive original FPS required")
    if getattr(network, "metric", None) is not True or getattr(network, "encoder", None) != "vits":
        raise ValueError("Only native Small metric depth is permitted; relative alignment forbidden")
    readonly = frames.view(); readonly.flags.writeable = False
    depth, returned_fps = network.infer_video_depth(readonly, fps, input_size=518, device="cuda", fp32=fp32)
    if (not isinstance(depth, np.ndarray) or depth.shape != frames.shape[:3] or depth.dtype != np.float32
            or not np.isfinite(depth).all() or np.any(depth < 0) or returned_fps != fps):
        raise ValueError("Every original frame must return native finite nonnegative FP32 metric depth")
    if np.any(np.count_nonzero(depth > 0, axis=(1, 2)) == 0):
        raise ValueError("Every frame requires positive native depth support; zeros remain unsupported")
    return depth

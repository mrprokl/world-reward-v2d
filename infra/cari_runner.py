"""Construct a native, forward-only CARI4D command; never execute it.

Pinned source: video_to_data@7c0d3b94, tools/run_mhr_wild_inference.py
522–579 and 681–706. The historical ``--foundationpose-file`` is an input
pose-file argument, not a request to execute FoundationPose. Its reader
requires only ``frames`` and ``obj_pose_world``. Caller-produced ICP poses
must remain identified as such in the reproducibility manifest.

This pure command builder does not read assets, prove their provenance or
licensing, inspect a GPU, hash checkpoints, launch a container, or establish
challenge performance. The runtime must do those checks separately.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import PurePosixPath
from typing import Any


UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
CHECKPOINT_REVISION = "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9"
CHECKPOINT_SHA256 = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"
CHECKPOINT_RELATIVE_PATH = "2026-08-25-09-35-57/step200000.pth"
CONFIG_RELATIVE_PATH = "learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml"
SCRIPT_RELATIVE_PATH = "tools/run_mhr_wild_inference.py"
MINIMUM_WINDOW_FRAMES = 96


def _runtime_path(value: Any, name: str, *, suffix: str | None = None) -> str:
    # Remote Linux paths, not local resolve()/exists(): no asset reads or hidden
    # filesystem dependence when planning an Azure run on the local machine.
    if not isinstance(value, (str, PurePosixPath)):
        raise TypeError(f"{name} must be an absolute Linux path")
    text = str(value)
    if not text or any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{name} contains an empty path or control characters")
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or "\\" in text or text.startswith("//"):
        raise ValueError(f"{name} must be an absolute Linux path without traversal")
    if str(path) == "/":
        raise ValueError(f"{name} cannot be the filesystem root")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"{name} must end with {suffix}")
    return str(path)


def build_cari_runtime_environment(
    native_root: str | PurePosixPath,
    mhr_assets_root: str | PurePosixPath,
    torch_home: str | PurePosixPath,
    sam3d_source_root: str | PurePosixPath,
) -> dict[str, str]:
    """Offline runtime configuration, usable before episode inputs exist.

    This plans paths only. The caller must verify original assets/source and
    enforce network isolation; setting offline flags is not an access barrier.
    """
    paths = {key: _runtime_path(value, key) for key, value in (
        ("native_root", native_root), ("mhr_assets_root", mhr_assets_root),
        ("torch_home", torch_home), ("sam3d_source_root", sam3d_source_root),
    )}
    if any(":" in paths[key] for key in ("native_root", "sam3d_source_root")):
        raise ValueError("Python source paths cannot contain the PYTHONPATH separator colon")
    return {
        "MHR_ASSETS_ROOT": paths["mhr_assets_root"],
        "TORCH_HOME": paths["torch_home"],
        "PYTHONPATH": paths["native_root"] + ":" + paths["sam3d_source_root"],
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "MOMENTUM_ENABLED": "0",
    }


def build_cari_forward_command(
    native_root: str | PurePosixPath,
    export_seq: str | PurePosixPath,
    depth_h5: str | PurePosixPath,
    mhr_init: str | PurePosixPath,
    object_poses: str | PurePosixPath,
    checkpoint: str | PurePosixPath,
    output: str | PurePosixPath,
    *,
    total_frames: int,
    mhr_assets_root: str | PurePosixPath,
    torch_home: str | PurePosixPath,
    sam3d_source_root: str | PurePosixPath,
    python_executable: str = "python",
    input_cache: str | PurePosixPath | None = None,
) -> dict[str, Any]:
    """Return ``argv``, ``cwd``, environment updates and an honest plan manifest.

    ``native_root`` is the pinned ``.../v2d_cari4d/lib/cari4d`` directory.
    ``mhr_assets_root`` contains ``checkpoints/sam-3d-body-dinov3/model.ckpt``
    and its original ``assets/mhr_model.pt``; ``sam3d_source_root`` is the
    directory containing the verified ``sam_3d_body`` source package.

    The caller must use ``subprocess.run(plan['argv'], shell=False, ...)``
    inside its offline Azure GPU runtime; environment entries are *updates*,
    not a complete environment. A verified local DINOv2 Hub loader or complete
    pinned Hub cache remains required. HF offline variables alone cannot
    prohibit Hub network calls: the runtime must enforce network isolation.

    All original frames are processed with the native terminal-window policy,
    window/stride96. There are no supported ``--no-gt`` or ``--no-refine`` flags
    in this script: it intrinsically performs CoCoNet forward only, with no
    FoundationPose generation, ground-truth features, or contact postoptimizer.
    No overwrite, legacy W&B query, oracle, short-window padding or arbitrary
    extra flags are exposed. Input semantic verification is the caller's job.
    """
    if isinstance(total_frames, bool) or not isinstance(total_frames, Integral) or total_frames < MINIMUM_WINDOW_FRAMES:
        raise ValueError(f"Full native CoCoNet inference requires integer total_frames >= {MINIMUM_WINDOW_FRAMES}; no padding or fabricated frames")
    paths = {
        "native_root": _runtime_path(native_root, "native_root"),
        "export_seq": _runtime_path(export_seq, "export_seq"),
        "depth_h5": _runtime_path(depth_h5, "depth_h5", suffix=".h5"),
        "mhr_init": _runtime_path(mhr_init, "mhr_init", suffix=".pkl"),
        "object_poses": _runtime_path(object_poses, "object_poses", suffix=".pkl"),
        "checkpoint": _runtime_path(checkpoint, "checkpoint", suffix=".pth"),
        "output": _runtime_path(output, "output", suffix=".pth"),
        "mhr_assets_root": _runtime_path(mhr_assets_root, "mhr_assets_root"),
        "torch_home": _runtime_path(torch_home, "torch_home"),
        "sam3d_source_root": _runtime_path(sam3d_source_root, "sam3d_source_root"),
    }
    if any(":" in paths[key] for key in ("native_root", "sam3d_source_root")):
        raise ValueError("Python source paths cannot contain the PYTHONPATH separator colon")
    if not paths["checkpoint"].endswith("/" + CHECKPOINT_RELATIVE_PATH):
        raise ValueError("checkpoint must name the audited commercial step200000 release; its bytes must still be verified by the runtime")
    if not isinstance(python_executable, str):
        raise TypeError("python_executable must be a command name or absolute executable path")
    if python_executable not in {"python", "python3"}:
        python_executable = _runtime_path(python_executable, "python_executable")
    native = PurePosixPath(paths["native_root"])
    script = str(native / SCRIPT_RELATIVE_PATH)
    config = str(native / CONFIG_RELATIVE_PATH)
    file_inputs = [paths[key] for key in ("depth_h5", "mhr_init", "object_poses", "checkpoint")] + [script, config]
    if len(set(file_inputs)) != len(file_inputs):
        raise ValueError("Distinct input files are required; body and object pose inputs cannot alias")
    if paths["output"] in file_inputs or paths["output"] in {paths["export_seq"], paths["native_root"]}:
        raise ValueError("output must not replace an input or source path")
    cache = None if input_cache is None else _runtime_path(input_cache, "input_cache", suffix=".h5")
    if cache is not None and cache in set(file_inputs) | {paths["output"], paths["export_seq"], paths["native_root"]}:
        raise ValueError("input_cache must not replace an input or output")
    argv = [
        python_executable, script, paths["export_seq"],
        "--depth-h5", paths["depth_h5"],
        "--mhr-init", paths["mhr_init"],
        # Native ABI name only. No FoundationPose executable/code stage runs.
        "--foundationpose-file", paths["object_poses"],
        "--config", config, "--checkpoint", paths["checkpoint"],
        "--output", paths["output"], "--stride", "96",
        "--render-batch-size", "32", "--crop-workers", "8",
        "--crop-buffer-count", "2", "--device", "cuda",
        "--offline-supervision-contract",
    ]
    argv.extend(["--no-input-cache"] if cache is None else ["--input-cache", cache])
    environment = build_cari_runtime_environment(
        paths["native_root"], paths["mhr_assets_root"], paths["torch_home"], paths["sam3d_source_root"],
    )
    metadata = {
        "stage": "cari4d_native_coconet_full_forward_only",
        "upstream_revision_required": UPSTREAM_REVISION,
        "checkpoint_repo": "nvidia/cari4d_commercial",
        "checkpoint_revision_required": CHECKPOINT_REVISION,
        "checkpoint_sha256_required": CHECKPOINT_SHA256,
        "input_verification": "required_at_runtime_not_performed_by_command_builder",
        "original_frame_count": int(total_frames),
        "window_length": 96, "window_stride": 96,
        "terminal_window": "include_final_full_window_first_occurrence_overlap_policy",
        "object_pose_source": paths["object_poses"],
        "object_pose_method": "caller_provided_own_estimated_trajectory_not_FoundationPose",
        "object_pose_native_cli_argument": "--foundationpose-file_historical_input_ABI_only",
        "foundationpose_stage_invoked": False,
        "contact_postoptimization_invoked": False,
        "ground_truth_or_oracle_modes_requested": False,
        "wandb_policy": "checkpoint_embedded_supervision_contract_offline",
        "input_cache": cache,
        "network_isolation": "required_at_runtime_not_established_by_environment",
        "submission_eligible": False, "challenge_performance_verified": False,
    }
    return {"argv": argv, "cwd": paths["native_root"], "environment": environment, "metadata": metadata}

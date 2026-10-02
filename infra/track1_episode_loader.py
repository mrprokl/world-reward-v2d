"""Lightweight frozen native-conversion loader, not scoring or eligibility.

This verifies selected-episode report/file integrity and numerical schema. A
SHA-bound passing report is not independent proof that its original numerical
claims are true: upstream model-forward/conversion gates supply that evidence.
No video, depth, model, official template, GT or Torch bundle is opened here.
Linux/remote restrictions belong to callers; tiny synthetic fixtures run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from cari_converter import (
    CHECKPOINT_SHA256, CONVERTER_SHA256, REFERENCE_MODEL_SHA256, UPSTREAM_REVISION,
    require_full_forward_report, require_report,
)
from world_reward.contracts import Reconstruction
from world_reward.data import sha256
from world_reward.submission import Track1Episode


# This older converter was hard-wired to outputs/episode_000015. Its passing
# report omitted episode_index; the source digest
# and all selected report/file links still bind the exact original artifact.
LEGACY_CONVERSION_REVISION = "414aac2a988f444eb166494c9d0d6aca5872ad0c"
LEGACY_CONVERSION_SCRIPT_SHA256 = "d2642f9816a6c7d6146b550a4b6500e108a33ff48ce55f54dec9bd7deddd1bc3"


def legacy_episode15_conversion(report, episode_index):
    return (type(episode_index) is int and episode_index == 15
            and "episode_index" not in report
            and report.get("producer_revision") == LEGACY_CONVERSION_REVISION
            and report.get("script_sha256") == LEGACY_CONVERSION_SCRIPT_SHA256)


@dataclass(frozen=True)
class LoadedTrack1Episode:
    episode: Track1Episode
    manifest: dict


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Require a lowercase 64-hex SHA-256")
    return value


def _regular(root: Path, relative: str) -> Path:
    path = root / relative
    if any(part == ".." for part in path.parts) or not path.is_relative_to(root):
        raise ValueError("Artifact escapes the selected root")
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root)):
        raise ValueError("Frozen artifact paths cannot contain symlinks")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Regular frozen artifact missing: {path}")
    return path


def _hash_file(root: Path, relative: str, expected: str) -> Path:
    path = _regular(root, relative)
    if sha256(path) != _digest(expected):
        raise ValueError(f"Frozen artifact SHA mismatch: {path}")
    return path


def _report(root, relative, stage, episode_index, *, expected_hash=None):
    path = _regular(root, relative) if expected_hash is None else _hash_file(root, relative, expected_hash)
    report = json.loads(path.read_text())
    require_report(report, stage)
    selected = report.get("episode_index", episode_index)
    if type(selected) is not int or selected != episode_index:
        raise ValueError("Frozen report belongs to another episode")
    return report, sha256(path)


def _float(value, name, shape):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or array.shape != shape or not np.isfinite(array).all()):
        raise ValueError(f"{name} requires finite float32/64 {shape}")
    return array.copy()


def load_track1_episode(
    root: str | Path, episode_index: int, total_frames: int, input_video_sha256: str,
) -> LoadedTrack1Episode:
    """Load exact full trajectories/shared identity, preserving all mesh padding.

    Paths are fixed to the selected episode. All five upstream report links are
    SHA-bound, including the actual full-forward gate. Missing report fields,
    invalid dtypes/frames and malformed geometry fail (only the exact allowlisted
    hard-wired episode15 converter may omit its episode field); no interpolation, pose
    repair, unit inference or alternative source is used. The params archive
    is verified by SHA, not decompressed redundantly. Returned arrays are copies.
    """
    if type(episode_index) is not int or not 0 <= episode_index < 30:
        raise ValueError("Require an episode integer in 0..29")
    if type(total_frames) is not int or total_frames < 1:
        raise ValueError("Require a positive original frame count")
    _digest(input_video_sha256)
    root = Path(root).absolute()
    if ".." in root.parts or root.is_symlink() or not root.is_dir():
        raise ValueError("Require an existing non-symlink artifact root without traversal")
    root = root.resolve()
    base = f"outputs/episode_{episode_index:06d}"
    final, report_hash = _report(root, f"{base}/cari_conversion/report.json", "world_reward_native_cari_official_conversion", episode_index)
    legacy_episode_format = legacy_episode15_conversion(final, episode_index)
    required = {
        "episode_index": episode_index, "frames": total_frames, "input_sha256": input_video_sha256,
        "original_frame_coverage_verified": True, "human_shared_identity_verified": True,
        "facial_expressions_zero": True, "human_parameter_format": "mhr_model_params_136_68_45",
        "geometry_frame": "camera_x_right_y_down_z_forward", "geometry_units": "metres",
        "upstream_revision": UPSTREAM_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
        "reference_model_sha256": REFERENCE_MODEL_SHA256, "official_converter_sha256": CONVERTER_SHA256,
        "network": "none",
    }
    for key, expected in required.items():
        if key == "episode_index" and legacy_episode_format:
            continue  # No mutation/invented field; exact legacy source only.
        actual = final.get(key)
        if actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected)):
            raise ValueError(f"Final native conversion contract mismatch: {key}")
    dependencies = {
        "forward": ("cari_forward/report.json", "world_reward_native_cari_full_forward"),
        "inputs": ("cari_inputs/report.json", "world_reward_native_cari_inputs"),
        "body": ("body_full/report.json", "sam3d_body_full_video_initializer"),
        "adapter": ("body_full/cari_adapter/report.json", "native_cari_body_adapter_full_video"),
        "object": ("object_pose_full/report.json", "fixed_scale_full_object_pose_initializer"),
    }
    source, links = {}, {}
    for name, (relative, stage) in dependencies.items():
        source[name], links[name] = _report(root, f"{base}/{relative}", stage, episode_index,
                                          expected_hash=final["input_report_sha256"][name])
    require_full_forward_report(source["forward"])
    if (source["forward"].get("inputs_report_sha256") != links["inputs"]
            or source["forward"].get("bundle_sha256") != _digest(final.get("bundle_sha256"))
            or source["adapter"].get("body_report_sha256") != links["body"]):
        raise ValueError("Native forward/adapter report chain is inconsistent")
    for name in ("inputs", "body", "object"):
        if source[name].get("input_sha256") != input_video_sha256:
            raise ValueError("Dependency report addresses another original video")
    if (type(source["inputs"].get("frames")) is not int or source["inputs"]["frames"] != total_frames
            or source["inputs"].get("original_frame_coverage_verified") is not True
            or type(source["adapter"].get("frames")) is not int or source["adapter"]["frames"] != total_frames
            or type(source["body"].get("total_video_frames")) is not int or source["body"]["total_video_frames"] != total_frames
            or source["body"].get("frame_indices") != list(range(total_frames))
            or any(type(frame) is not int for frame in source["body"]["frame_indices"])
            or source["body"].get("mhr_geometry_forward_verified") is not True
            or source["object"].get("original_frame_coverage_verified") is not True or source["object"].get("fixed_shape") is not True):
        raise ValueError("Dependency full-frame or fixed-geometry contract mismatch")
    for name in ("body", "adapter", "object"):
        if source["inputs"].get("input_report_sha256", {}).get(name) != links[name]:
            raise ValueError("Native inputs differ from final dependency report chain")
    identity = final.get("inference_source_identity")
    if (not isinstance(identity, dict) or not identity or any(source[name].get("inference_source_identity") != identity
            for name in ("forward", "body", "adapter")) or not final.get("body_assets")
            or source["body"].get("body_assets") != final["body_assets"]
            or not final.get("decoder_identity") or source["adapter"].get("decoder_identity") != final["decoder_identity"]):
        raise ValueError("Original source/assets/decoder identities differ")
    episode_path = _hash_file(root, f"{base}/cari_conversion/episode.npz", final.get("episode_sha256"))
    _hash_file(root, f"{base}/cari_conversion/params.npz", final.get("params_sha256"))
    shapes = {"pose": (total_frames, 136), "scales": (68,), "shape": (45,), "expression": (72,),
              "object_rotation": (total_frames, 3, 3), "object_translation": (total_frames, 3), "object_scale": ()}
    with np.load(episode_path, allow_pickle=False) as archive:
        if set(archive.files) != set(shapes) | {"frame_index", "object_vertices", "object_faces"}:
            raise ValueError("Final episode archive must have exactly the fixed native-conversion keys")
        arrays = {name: _float(archive[name], name, shape) for name, shape in shapes.items()}
        frames = archive["frame_index"]
        if frames.dtype.kind not in "iu" or frames.shape != (total_frames,) or not np.array_equal(frames, np.arange(total_frames)):
            raise ValueError("Final trajectory must retain every original frame in exact order")
        vertices, faces = archive["object_vertices"], archive["object_faces"]
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 3 <= len(vertices) <= 4096:
            raise ValueError("Object vertex count/shape exceeds the fixed 4096 budget")
        vertices = _float(vertices, "object_vertices", vertices.shape)
        if (faces.dtype not in (np.dtype("int32"), np.dtype("int64")) or faces.ndim != 2
                or faces.shape[1:] != (3,) or not 1 <= len(faces) <= 4096
                or np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError("Object faces require valid int32/64 indices within the 4096 budget")
        faces = faces.copy()
    if float(arrays["object_scale"]) != 1. or np.any(arrays["expression"] != 0):
        raise ValueError("Require one baked object scale and zero facial expression")
    reconstruction = Reconstruction(**{name: arrays[name] for name in shapes if name != "expression"})
    episode = Track1Episode(reconstruction, vertices, faces, arrays["expression"],
                            {"input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
                             "oracle_modes": [], "input_sha256": input_video_sha256}, total_frames)
    episode.validate()
    manifest = {"episode_index": episode_index, "frames": total_frames, "input_video_sha256": input_video_sha256,
                "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
                "conversion_report_sha256": report_hash, "episode_sha256": final["episode_sha256"],
                "params_sha256": final["params_sha256"], "dependency_report_sha256": links,
                "upstream_revision": UPSTREAM_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                "reference_model_sha256": REFERENCE_MODEL_SHA256, "official_converter_sha256": CONVERTER_SHA256,
                "legacy_episode15_conversion_format": legacy_episode_format,
                "episode_identity_basis": "allowlisted_hardwired_episode15_source_and_all_artifact_links" if legacy_episode_format else "explicit_report_field_and_all_artifact_links",
                "integrity_and_schema_verified": True, "numerical_truth_independently_reverified": False,
                "submission_eligibility_verified": False, "challenge_performance_verified": False}
    return LoadedTrack1Episode(episode, manifest)

"""Azure-only conversion of verified CoCoNet output to official Track 1 MHR.

Native parameters are decoded by the original, asset-bound MHRLayer. The
official converter fits one shared identity; no parameter-layout shortcut,
extra axis flip, translation gain, scale fit, or nearest-frame fallback is
allowed. The 2 mm per-frame mean residual is an engineering fidelity gate,
not evidence of accuracy against challenge ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import json
from numbers import Integral
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from world_reward.contracts import require_rigid_transforms, require_video_only_provenance
from world_reward.data import sha256
from world_reward.submission import parameters_from_official_converter


UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
CHECKPOINT_SHA256 = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"
REFERENCE_MODEL_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
CONVERTER_SHA256 = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
MAX_MEAN_VERTEX_ERROR_MM = 2.0
FRAME_CHANGE_ATOL_M = 1e-5
PARAMETER_DIMS = {
    "mhr_global_rot6d": 6, "mhr_trans": 3, "mhr_body_pose_cont": 260,
    "mhr_hand": 108, "mhr_shape": 45, "mhr_scale": 28, "mhr_face": 72,
}


def _float_array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide invalid entries behind a mask")
    array = np.asarray(value)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def require_report(report: Mapping, stage: str) -> None:
    """Require explicit no-oracle, no-hand-label, Track 1 pass provenance."""
    if not isinstance(report, Mapping) or report.get("stage") != stage or report.get("status") != "pass":
        raise ValueError(f"Require a passing {stage} report")
    require_video_only_provenance(dict(report))
    if report.get("hand_labeled_test") is not False:
        raise ValueError("Explicit hand_labeled_test=False required")


def require_full_forward_report(report: Mapping) -> None:
    """A checkpoint-load gate is never accepted as actual full-video inference."""
    require_report(report, "world_reward_native_cari_full_forward")
    metadata = report.get("metadata")
    if (not isinstance(metadata, Mapping)
            or metadata.get("actual_network_forward_verified") is not True
            or metadata.get("full_original_frame_coverage_verified") is not True
            or report.get("episode_inputs_used") is not True
            or report.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("Require actual full-frame CoCoNet forward with the pinned checkpoint")


def validate_native_bundle(bundle: Mapping, total_frames: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Validate native ABI and preserve every original frame, without decoding.

    Root 6D controls use CARI's interleaved first two matrix columns, unlike
    the separate continuous-body layout. A degenerate root basis fails; it is
    never repaired to an arbitrary rotation. Extra cached geometry is ignored.
    Only the prediction mapping is passed to the actual decoder.
    """
    if (isinstance(total_frames, (bool, np.bool_)) or not isinstance(total_frames, Integral)
            or total_frames < 1):
        raise ValueError("total_frames must be a positive integer")
    if not isinstance(bundle, Mapping):
        raise ValueError("Native bundle must be a mapping")
    gt, metadata = bundle.get("gt"), bundle.get("metadata")
    if (not isinstance(gt, dict) or gt
            or not isinstance(metadata, Mapping) or metadata.get("ground_truth_used") is not False):
        raise ValueError("Native bundle must explicitly contain no ground truth")
    if bundle.get("frames") != [f"{index:06d}" for index in range(total_frames)]:
        raise ValueError("Native bundle must cover all original frames in exact order")
    prediction = bundle.get("pr")
    if not isinstance(prediction, Mapping):
        raise ValueError("Native prediction mapping missing")
    params = {}
    for name, dimension in PARAMETER_DIMS.items():
        if name not in prediction:
            raise ValueError(f"Native prediction missing {name}")
        params[name] = _float_array(prediction[name], name, (total_frames, dimension)).copy()
    basis = params["mhr_global_rot6d"].astype(np.float64).reshape(total_frames, 3, 2)
    first = basis[:, :, 0]
    norms = np.linalg.norm(first, axis=1)
    if np.any(norms <= 1e-8) or not np.isfinite(norms).all():
        raise ValueError("Native root 6D first column is degenerate")
    first = first / norms[:, None]
    second = basis[:, :, 1] - np.sum(first * basis[:, :, 1], axis=1)[:, None] * first
    norms = np.linalg.norm(second, axis=1)
    if np.any(norms <= 1e-8) or not np.isfinite(norms).all():
        raise ValueError("Native root 6D columns are degenerate")
    second /= norms[:, None]
    root_pose = np.broadcast_to(np.eye(4), (total_frames, 4, 4)).copy()
    root_pose[:, :3, :3] = np.stack([first, second, np.cross(first, second)], axis=-1)
    require_rigid_transforms(root_pose, total_frames)
    pose = _float_array(prediction.get("pose_abs"), "pose_abs", (total_frames, 4, 4))
    require_rigid_transforms(pose, total_frames)
    return params, pose.copy()


def require_aligned_object_metadata(metadata: Mapping) -> None:
    """Prove that native training poses still address the exported aligned mesh.

    The pinned wild exporter selects an already aligned GLB, so both native
    storage/training transforms must be identity. Refuse an additional gauge
    instead of silently guessing a second inverse when restoring source poses.
    """
    if (not isinstance(metadata, Mapping)
            or metadata.get("object_pose_frame") != "centered_axis_aligned"
            or metadata.get("object_pose_frame_revision") != "cari4d.object_pose_frame.centered_axis_aligned.v1"
            or metadata.get("object_pose_storage_frame") != "output_aligned_mesh_frame"):
        raise ValueError("Native object poses must use the verified aligned-mesh frame")
    for name in ("object_pose_storage_to_training_transform", "object_mesh_to_training_transform"):
        transform = _float_array(metadata.get(name), name, (4, 4))
        require_rigid_transforms(transform[None], 1)
        if not np.allclose(transform, np.eye(4), atol=1e-6, rtol=0):
            raise ValueError("Unexpected additional native object frame change")


def restore_object_source_frame(aligned_poses, frame_change, vertices, faces) -> tuple[np.ndarray, dict]:
    """Undo only native mesh-frame alignment: P_source = P_aligned @ A.

    A maps source metric mesh coordinates into aligned coordinates. Vertices,
    faces (including official padding), and baked scale remain untouched.
    Check equivalence for every frame and vertex; no pose averaging, rescaling,
    SVD projection, decimation, cavity inversion, or topology repair occurs.
    """
    pose = _float_array(aligned_poses, "aligned_poses")
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or not len(pose):
        raise ValueError("aligned_poses must have nonempty shape [T,4,4]")
    require_rigid_transforms(pose, len(pose))
    transform = _float_array(frame_change, "frame_change", (4, 4))
    require_rigid_transforms(transform[None], 1)
    mesh_vertices = _float_array(vertices, "vertices")
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1:] != (3,) or not 3 <= len(mesh_vertices) <= 4096:
        raise ValueError("vertices must have shape [V,3], 3 <= V <= 4096")
    if np.ma.isMaskedArray(faces):
        raise ValueError("faces cannot hide indices behind a mask")
    mesh_faces = np.asarray(faces)
    if (mesh_faces.dtype.kind not in "iu" or mesh_faces.ndim != 2
            or mesh_faces.shape[1:] != (3,) or not 1 <= len(mesh_faces) <= 4096
            or np.any(mesh_faces < 0) or np.any(mesh_faces >= len(mesh_vertices))):
        raise ValueError("faces must contain valid integer indices within the 4096-row budget")
    a, b, c = (mesh_vertices[mesh_faces[:, column]].astype(np.float64) for column in range(3))
    with np.errstate(over="ignore", invalid="ignore"):
        areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.isfinite(areas).all() or not np.any(areas > 0):
        raise ValueError("Mesh must have a finite nonzero surface area")
    pose = pose.astype(np.float64)
    transform = transform.astype(np.float64)
    source_poses = pose @ transform
    require_rigid_transforms(source_poses, len(pose))
    points = mesh_vertices.astype(np.float64)
    aligned_points = points @ transform[:3, :3].T + transform[:3, 3]
    maximum = 0.0
    for aligned, source in zip(pose, source_poses, strict=True):
        before = aligned_points @ aligned[:3, :3].T + aligned[:3, 3]
        after = points @ source[:3, :3].T + source[:3, 3]
        error = np.linalg.norm(before - after, axis=1)
        if not np.isfinite(error).all():
            raise ValueError("Mesh-frame roundtrip overflowed")
        maximum = max(maximum, float(error.max()))
    if maximum > FRAME_CHANGE_ATOL_M:
        raise ValueError("Native mesh-frame roundtrip exceeds the declared numerical gate")
    return source_poses, {
        "frame_change": "P_source = P_aligned @ A; v_aligned = A @ v_source",
        "roundtrip_max_error_m": maximum, "roundtrip_atol_m": FRAME_CHANGE_ATOL_M,
        "frames_verified": len(pose), "vertices_verified_per_frame": len(points),
        "source_vertices_and_faces_preserved": True, "object_scale": 1.0,
        "additional_scale_applied": False,
    }


def vertex_residual_mm(recovered_m, target_m) -> tuple[np.ndarray, float]:
    """Return per-frame mean Euclidean error and maximum point error in mm."""
    target = _float_array(target_m, "target_m")
    recovered = _float_array(recovered_m, "recovered_m", target.shape)
    if target.ndim != 3 or target.shape[-1] != 3 or not target.shape[0] or not target.shape[1]:
        raise ValueError("Vertex sequences must have nonempty shape [T,V,3]")
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.linalg.norm(recovered.astype(np.float64) - target.astype(np.float64), axis=-1) * 1000
    if not np.isfinite(errors).all():
        raise ValueError("Independent vertex residual is nonfinite")
    return errors.mean(axis=1), float(errors.max())


def require_fidelity(errors_mm, *, limit_mm: float = MAX_MEAN_VERTEX_ERROR_MM) -> dict:
    """Gate each frame's mean, not each individual vertex or challenge score."""
    errors = _float_array(errors_mm, "per_frame_mean_mm")
    if errors.ndim != 1 or not len(errors) or np.any(errors < 0):
        raise ValueError("Require nonnegative per-frame mean residuals")
    if (isinstance(limit_mm, (bool, np.bool_)) or not np.isfinite(limit_mm) or limit_mm < 0
            or np.any(errors > limit_mm)):
        raise ValueError("Independent per-frame mean vertex residual exceeds the engineering fidelity gate")
    return {
        "mean_vertex_error_mm": float(errors.mean()), "worst_frame_mean_mm": float(errors.max()),
        "worst_frame": int(np.argmax(errors)), "per_frame_mean_mm": errors.tolist(),
        "max_per_frame_mean_gate_mm": float(limit_mm),
        "gate_scope": "engineering_representation_fidelity_not_heldout_accuracy",
    }


def _read_report(path: Path, stage: str) -> dict:
    report = json.loads(path.read_text())
    require_report(report, stage)
    return report


def _require_hash(path: Path, expected: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or path.is_symlink() or not path.is_file() or sha256(path) != expected):
        raise RuntimeError(f"Frozen artifact missing or altered: {path}")


def main() -> None:
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    import torch
    from body_smoke import _body_assets, _pinned_checkout, _source_identity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/local fallback")
    root = Path(os.environ["WR_ROOT"])
    base = root / "outputs/episode_000015"
    forward_path = base / "cari_forward/report.json"
    forward = json.loads(forward_path.read_text())
    require_full_forward_report(forward)
    bundle_path = base / "cari_forward/coconet.pth"
    _require_hash(bundle_path, forward.get("bundle_sha256"))
    inputs_path = base / "cari_inputs/report.json"
    _require_hash(inputs_path, forward.get("inputs_report_sha256"))
    inputs = _read_report(inputs_path, "world_reward_native_cari_inputs")
    count = inputs["frames"]
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or inputs.get("original_frame_coverage_verified") is not True):
        raise RuntimeError("Require full original-frame CARI input provenance")
    for field in ("depth_h5", "mhr_init", "object_poses"):
        path = Path(inputs[field])
        if not path.resolve().is_relative_to(base):
            raise RuntimeError("Native input artifact escapes the episode directory")
        _require_hash(path, inputs["file_sha256"][field])
    body_path = base / "body_full/report.json"
    adapter_path = base / "body_full/cari_adapter/report.json"
    object_path = base / "object_pose_full/report.json"
    for key, path in (("body", body_path), ("adapter", adapter_path), ("object", object_path)):
        _require_hash(path, inputs["input_report_sha256"][key])
    body = _read_report(body_path, "sam3d_body_full_video_initializer")
    adapter = _read_report(adapter_path, "native_cari_body_adapter_full_video")
    objects = _read_report(object_path, "fixed_scale_full_object_pose_initializer")
    if (body.get("total_video_frames") != count or body.get("frame_indices") != list(range(count))
            or body.get("mhr_geometry_forward_verified") is not True
            or body.get("upstream_revision") != UPSTREAM_REVISION
            or body.get("input_sha256") != inputs["input_sha256"]
            or objects.get("input_sha256") != inputs["input_sha256"]
            or objects.get("original_frame_coverage_verified") is not True
            or objects.get("fixed_shape") is not True
            or adapter.get("frames") != count or adapter.get("body_report_sha256") != sha256(body_path)
            or adapter.get("canonical_initializer_sha256") != inputs["file_sha256"]["mhr_init"]):
        raise RuntimeError("Full human/object/initializer provenance does not match native forward inputs")
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    source_identity = _source_identity(root)
    if any(record.get("inference_source_identity") != source_identity for record in (body, adapter, forward)):
        raise RuntimeError("Original Body and native decoder source identities differ")
    assets, body_hashes = _body_assets(root)
    if body_hashes != body["body_assets"] or (assets / "mhr_buffers.pt").exists():
        raise RuntimeError("Original Body assets changed or an unverified compact buffer override exists")
    checkpoints = list((root / "weights/cari4d").rglob("step200000.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("Require one pinned native CoCoNet checkpoint")
    _require_hash(checkpoints[0], CHECKPOINT_SHA256)
    model = root / "weights/mhr/mhr_model.pt"
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    _require_hash(model, REFERENCE_MODEL_SHA256)
    _require_hash(tool, CONVERTER_SHA256)
    export = base / "cari_inputs/export/episode_000015"
    if Path(inputs["export_seq"]).resolve() != export:
        raise RuntimeError("Unexpected native export sequence path")
    wild_path = export / "wild_export.json"
    _require_hash(wild_path, inputs["file_sha256"]["wild_export"])
    frame_change = np.asarray(json.loads(wild_path.read_text())["source_object_mesh_to_aligned_transform"], dtype=np.float64)
    geometry_path = base / "object_pose_full/geometry_and_poses.npz"
    _require_hash(geometry_path, objects["geometry_and_poses_sha256"])
    with np.load(geometry_path, allow_pickle=False) as archive:
        vertices, faces = archive["vertices"].copy(), archive["faces"].copy()
        frame_indices, scale = archive["frame_index"].copy(), archive["object_scale"].copy()
    if (frame_indices.dtype.kind not in "iu" or not np.array_equal(frame_indices, np.arange(count))
            or scale.shape != () or scale.dtype.kind != "f" or not np.isfinite(scale) or float(scale) != 1.0):
        raise RuntimeError("Object geometry must have all original frames and scale baked exactly once")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    params, aligned_poses = validate_native_bundle(bundle, count)
    require_aligned_object_metadata(bundle["metadata"])
    if Path(bundle["metadata"]["object_mesh"]).resolve() != export / "object_mesh/output_aligned.glb":
        raise RuntimeError("Native object poses refer to an unexpected mesh")
    del bundle
    source_poses, object_report = restore_object_source_frame(aligned_poses, frame_change, vertices, faces)

    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    os.environ.update({"MHR_ASSETS_ROOT": str(root / "weights/cari4d/sam3d_body"),
                       "MOMENTUM_ENABLED": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    # Native first: SAM's own top-level tools package otherwise shadows CARI.
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.geometry_provider import decode_mhr_vertices_numpy
    from lib_mhr.schema import MHR_PARAM_DIMS
    if MHR_PARAM_DIMS != PARAMETER_DIMS:
        raise RuntimeError("Pinned native MHR ABI differs from the converter contract")
    for symbol in (MHRLayer, decode_mhr_vertices_numpy):
        loaded = Path(sys.modules[symbol.__module__].__file__).resolve()
        if not loaded.is_relative_to(native / "lib_mhr"):
            raise RuntimeError("Native MHR import resolved outside the pinned source")
    output = base / "cari_conversion"
    if output.exists():
        raise RuntimeError("Frozen native CARI conversion exists")
    absent_buffer = output / "never_use_an_unverified_compact_buffer.pt"
    layer = MHRLayer.from_mhr_assets(
        mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets / "model.ckpt",
        buffer_path=absent_buffer, mhr_model_path=assets / "assets/mhr_model.pt", device="cuda",
    )
    decoder_identity = layer.decoder_identity()
    if decoder_identity != adapter["decoder_identity"]:
        raise RuntimeError("Decoder identity differs from the full original-Body adapter")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    native_vertices = decode_mhr_vertices_numpy(layer, params, batch_size=16)
    _float_array(native_vertices, "native_camera_vertices_m", (count, 18439, 3))
    del params, layer
    spec = importlib.util.spec_from_file_location("world_reward_official_cari_converter", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the verified official converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    converted = converter.convert(native_vertices, str(model), device="cuda", precision="float32",
                                  model_batch=256, log=lambda *args, **kwargs: None)
    official = parameters_from_official_converter(converted, max_mean_vertex_error_mm=MAX_MEAN_VERTEX_ERROR_MM)
    if len(official.pose) != count or np.any(official.expression != 0):
        raise RuntimeError("Official conversion changed coverage or the zero-expression contract")
    reference = converter.MHR(str(model), "cuda", chunk=16, precision="float32")
    identity = torch.tensor(np.concatenate([official.scales, official.shape])[None], dtype=torch.float64, device="cuda")
    errors, max_point_mm = [], 0.0
    for start in range(0, count, 16):
        stop = min(count, start + 16)
        recovered_mm, _ = reference.run(torch.tensor(official.pose[start:stop], dtype=torch.float64, device="cuda"), identity)
        recovered_m = (recovered_mm / 1000).detach().cpu().numpy()
        means, point_max = vertex_residual_mm(recovered_m, native_vertices[start:stop])
        errors.extend(means.tolist())
        max_point_mm = max(max_point_mm, point_max)
    fidelity = require_fidelity(np.asarray(errors, dtype=np.float64))
    fidelity["max_point_error_mm_diagnostic_not_gate"] = max_point_mm
    with (output / "params.npz").open("xb") as handle:
        np.savez_compressed(handle, pose=official.pose, scales=official.scales, shape=official.shape,
                            expression=official.expression, frame_index=frame_indices,
                            valid_input=converted["valid_input"], per_frame_vertex_error_mm=converted["per_frame_vertex_error_mm"],
                            report=np.asarray(json.dumps(official.report, sort_keys=True)))
    with (output / "episode.npz").open("xb") as handle:
        np.savez_compressed(handle, pose=official.pose, scales=official.scales, shape=official.shape,
                            expression=official.expression, frame_index=frame_indices,
                            object_vertices=vertices, object_faces=faces,
                            object_rotation=source_poses[:, :3, :3], object_translation=source_poses[:, :3, 3],
                            object_scale=np.asarray(1.0))
    with np.load(output / "episode.npz", allow_pickle=False) as packed:
        if not np.array_equal(packed["object_vertices"], vertices) or not np.array_equal(packed["object_faces"], faces):
            raise RuntimeError("Final archive changed fixed source mesh arrays")
        require_rigid_transforms(source_poses, count)
    result = {
        "stage": "world_reward_native_cari_official_conversion", "status": "pass", "frames": count,
        "input_track": "track_1", "input_sha256": inputs["input_sha256"], "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "network": "none",
        "original_frame_coverage_verified": True, "human_shared_identity_verified": True,
        "human_parameter_format": "mhr_model_params_136_68_45", "facial_expressions_zero": True,
        "geometry_units": "metres", "geometry_frame": "camera_x_right_y_down_z_forward",
        "native_decode": "original_MHRLayer_certified_parameters_no_extra_flip_scale_or_root_transform",
        "decoder_identity": decoder_identity, "body_assets": body_hashes,
        "inference_source_identity": source_identity, "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256, "reference_model_sha256": REFERENCE_MODEL_SHA256,
        "official_converter_sha256": CONVERTER_SHA256, "official_converter_report": official.report,
        "independent_reference_forward": fidelity, "object_frame_restoration": object_report,
        "source_object_geometry_sha256": sha256(geometry_path), "wild_export_sha256": sha256(wild_path),
        "input_report_sha256": {"forward": sha256(forward_path), "inputs": sha256(inputs_path),
                                "body": sha256(body_path), "adapter": sha256(adapter_path), "object": sha256(object_path)},
        "bundle_sha256": sha256(bundle_path), "params_sha256": sha256(output / "params.npz"),
        "episode_sha256": sha256(output / "episode.npz"), "script_sha256": sha256(Path(__file__)),
        "producer_revision": os.environ.get("WR_CODE_REVISION"), "elapsed_seconds": time.perf_counter() - started,
        "submission_eligible": False, "challenge_performance_verified": False,
        "gate_basis": "predeclared_2mm_engineering_fidelity_consistent_with_prior_full_Body_adapter_not_GT_validation",
    }
    with (output / "report.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "status": "pass", "frames": count,
                      "worst_frame_mean_mm": fidelity["worst_frame_mean_mm"],
                      "object_roundtrip_max_m": object_report["roundtrip_max_error_m"],
                      "elapsed_seconds": result["elapsed_seconds"], "challenge_performance_verified": False}))


if __name__ == "__main__":
    main()

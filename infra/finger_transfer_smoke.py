"""Offline sparse H1 engineering proposal, never adoption or accuracy testing.

Keep the final official pose136/shared identity, and replace only the 54
finger controls with already decoded full-mode SAM Body controls. Use the
verified checkpoint's absolute hand-column indices, not hand108/raw266 or
cached joint rotations. Only original frames 0/T//2/T-1 are exported. The
official reference model checks finite geometry with its existing camera
convention; this is not independent image evidence for the proposed hands.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from world_reward.contracts import require_rigid_transforms, require_video_only_provenance
from world_reward.data import sha256
from world_reward.hand_transfer import transfer_finger_controls


UPSTREAM_REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
CHECKPOINT_SHA256 = "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"
REFERENCE_MODEL_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
CONVERTER_SHA256 = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
FINAL_STAGE = "world_reward_native_cari_official_conversion"
HANDS_STAGE = "sam3d_body_three_frame_hand_proposals"
STAGE = "world_reward_sparse_finger_transfer_engineering"


def _float_array(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide entries behind a mask")
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite floating-point {shape}")
    return array


def _require_report(report: Mapping, stage: str, episode: int) -> None:
    if not isinstance(report, Mapping) or report.get("stage") != stage or report.get("status") != "pass":
        raise ValueError(f"Require a passing {stage} report")
    require_video_only_provenance(dict(report))
    if report.get("hand_labeled_test") is not False or report.get("network") != "none":
        raise ValueError("Require explicit no-hand-label and offline provenance")
    if "episode_index" not in report:
        from track1_episode_loader import legacy_episode15_conversion
        if stage != FINAL_STAGE or not legacy_episode15_conversion(report, episode):
            raise ValueError("Only the exact hard-wired legacy final converter may omit episode_index")
        return
    selected = report["episode_index"]
    if type(selected) is not int or selected != episode:
        raise ValueError("Producer report belongs to another episode")


def _require_hash(path: Path, expected: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or path.is_symlink() or not path.is_file() or sha256(path) != expected):
        raise RuntimeError(f"Frozen artifact missing or altered: {path}")


def require_forward_network(forward, report_hash, bundle_hash, episode, proof_path):
    """Explicit offline field or exact frozen audit, never an implicit default."""
    if "network" in forward:
        _require_report(forward, "world_reward_native_cari_full_forward", episode)
        return None
    from forward_network_proof import validate_proof
    from cari_converter import require_full_forward_report
    require_full_forward_report(forward)
    if proof_path.is_symlink() or not proof_path.is_file():
        raise ValueError("Missing forward offline field requires the exact legacy network proof")
    digest = sha256(proof_path)
    validate_proof(json.loads(proof_path.read_text()), forward, report_hash, bundle_hash, episode)
    return digest


def validate_producer_reports(final: Mapping, hands: Mapping, episode: int) -> tuple[int, np.ndarray]:
    """Require matching, explicit native/full-mode provenance; no GT is read."""
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Require an episode integer in 0..29")
    _require_report(final, FINAL_STAGE, episode)
    _require_report(hands, HANDS_STAGE, episode)
    count = final.get("frames")
    if type(count) is not int or count < 3:
        raise ValueError("Require at least three original frames")
    indices = np.asarray([0, count // 2, count - 1], dtype=np.int64)
    required_final = {
        "original_frame_coverage_verified": True, "human_shared_identity_verified": True,
        "facial_expressions_zero": True, "human_parameter_format": "mhr_model_params_136_68_45",
        "geometry_units": "metres", "geometry_frame": "camera_x_right_y_down_z_forward",
        "reference_model_sha256": REFERENCE_MODEL_SHA256, "official_converter_sha256": CONVERTER_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256, "upstream_revision": UPSTREAM_REVISION,
    }
    required_hands = {
        "episode_index": episode, "total_video_frames": count, "frame_indices": indices.tolist(),
        "inference_type": "full", "hand_decoder_proposals": True, "mhr_geometry_forward_verified": True,
        "mhr_geometry_forward_basis": "original_body133_hand108_global_rot3_scale28_shape45_expr72_not_raw_logits",
        "joint_global_rotation_source": "fresh_native_parameter_block_decode",
        "raw_pose_logits_role": "audit_only_zeroed_after_full_hand_fusion",
        "prompt_mode": "automatic_mask_and_derived_bbox_no_fallback", "human_mask_id": 0,
        "geometry_units": "metres", "geometry_frame": "SAM3D_camera_x_right_y_down_z_forward",
        "upstream_revision": UPSTREAM_REVISION, "vertices": 18439, "faces": 36874,
    }
    for report, required in ((final, required_final), (hands, required_hands)):
        for key, expected in required.items():
            value = report.get(key)
            if (value != expected or (type(expected) is bool and value is not expected)
                    or (type(expected) is int and type(value) is not int)):
                raise ValueError(f"Producer provenance mismatch: {key}")
    video_hash = final.get("input_sha256")
    if (not isinstance(video_hash, str) or len(video_hash) != 64
            or any(c not in "0123456789abcdef" for c in video_hash) or hands.get("input_sha256") != video_hash):
        raise ValueError("Human producers must address the same SHA-bound original Track 1 video")
    for name in ("inference_source_identity", "body_assets"):
        value = final.get(name)
        if not isinstance(value, Mapping) or not value or hands.get(name) != value:
            raise ValueError(f"Human source/asset provenance mismatch: {name}")
    records = hands.get("frames")
    if (not isinstance(records, list) or len(records) != 3
            or [r.get("frame_index") for r in records if isinstance(r, Mapping)] != indices.tolist()
            or any(type(r.get("frame_index")) is not int for r in records if isinstance(r, Mapping))):
        raise ValueError("Require the three exact original-frame native control-forward records")
    for record in records:
        for key in ("native_forward_max_error_m", "native_joint_forward_max_error_m", "native_controls_forward_max_error"):
            error = record.get(key)
            if (isinstance(error, bool) or not isinstance(error, (int, float))
                    or not np.isfinite(error) or not 0 <= error <= 1e-5):
                raise ValueError(f"Source native parameter-block forward failed: {key}")
    return count, indices


def validate_final_arrays(archive: Mapping, count: int) -> dict[str, np.ndarray]:
    """Freeze shared identity and full original-frame coverage; never fit/repair."""
    arrays = {name: _float_array(archive[name], name, shape) for name, shape in {
        "pose": (count, 136), "scales": (68,), "shape": (45,), "expression": (72,),
        "object_rotation": (count, 3, 3), "object_translation": (count, 3), "object_scale": (),
    }.items()}
    frames = np.asarray(archive["frame_index"])
    if (np.ma.isMaskedArray(archive["frame_index"]) or frames.dtype.kind not in "iu"
            or frames.shape != (count,) or not np.array_equal(frames, np.arange(count))):
        raise ValueError("Final archive must cover all original frames in exact order")
    if np.any(arrays["expression"] != 0) or float(arrays["object_scale"]) != 1:
        raise ValueError("Require zero expressions and exactly once baked object scale")
    transforms = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    transforms[:, :3, :3], transforms[:, :3, 3] = arrays["object_rotation"], arrays["object_translation"]
    require_rigid_transforms(transforms, count)
    vertices = np.asarray(archive["object_vertices"])
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 3 <= len(vertices) <= 4096:
        raise ValueError("Final object vertices exceed the fixed mesh budget")
    _float_array(archive["object_vertices"], "object_vertices", vertices.shape)
    _faces(archive["object_faces"], len(vertices), max_faces=4096)
    arrays["frame_index"] = frames
    return arrays


def _faces(value, vertex_count: int, *, max_faces: int) -> np.ndarray:
    faces = np.asarray(value)
    if (np.ma.isMaskedArray(value) or faces.dtype.kind not in "iu" or faces.ndim != 2
            or faces.shape[1:] != (3,) or not 1 <= len(faces) <= max_faces
            or np.any(faces < 0) or np.any(faces >= vertex_count)):
        raise ValueError("Require valid integer topology indices within the face budget")
    return faces


def validate_source_arrays(archive: Mapping, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consume fresh decoded204 only; invalidated raw266 is audit-only."""
    count = len(indices)
    frames = np.asarray(archive["frame_index"])
    if (np.ma.isMaskedArray(archive["frame_index"]) or frames.dtype.kind not in "iu"
            or frames.shape != indices.shape or not np.array_equal(frames, indices)):
        raise ValueError("Source must contain exactly 0/T//2/T-1, without reordering or interpolation")
    controls = _float_array(archive["mhr_model_params"], "decoded_source_controls", (count, 204))
    expressions = _float_array(archive["expr_params"], "source_expression", (count, 72))
    raw = _float_array(archive["pred_pose_raw"], "audit_only_raw_logits", (count, 266))
    if np.any(expressions != 0) or np.any(raw != 0):
        raise ValueError("Require disabled expression and upstream-invalidated raw logits")
    rotations = _float_array(archive["pred_global_rots"], "fresh_source_rotations", (count, 127, 3, 3))
    flattened = rotations.reshape(-1, 3, 3).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if (not np.allclose(flattened @ flattened.swapaxes(-1, -2), np.eye(3), atol=1e-4, rtol=0)
                or not np.allclose(np.linalg.det(flattened), 1, atol=1e-4, rtol=0)):
            raise ValueError("Fresh source joint rotations must be SO(3), never repaired")
    faces = _faces(archive["faces"], 18439, max_faces=36874)
    if faces.shape != (36874, 3):
        raise ValueError("Require the fixed native MHR topology")
    return controls, faces


def checkpoint_hand_indices(state: Mapping, torch) -> tuple[np.ndarray, np.ndarray]:
    """Read only verified checkpoint buffers; the transfer helper checks the union."""
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state must be a mapping")
    values = []
    for side in ("left", "right"):
        value = state.get(f"head_pose.hand_joint_idxs_{side}")
        if not torch.is_tensor(value):
            raise ValueError("Verified checkpoint hand-column tensor missing")
        array = value.detach().cpu().numpy()
        if array.shape != (27,) or array.dtype.kind not in "iu":
            raise ValueError("Checkpoint hand-column tensor must be integer [27]")
        values.append(array.copy())
    combined = np.concatenate(values)
    if not np.array_equal(np.sort(combined), np.arange(68, 122)):
        raise ValueError("Checkpoint hand columns must partition exactly 68:122")
    return tuple(values)


def reference_arrays(vertices_mm, joints_m, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Official MHR.run already flips axes: only vertices mm->m is needed."""
    vertices = _float_array(vertices_mm, "reference_vertices_mm", (count, 18439, 3)) / 1000
    joints = _float_array(joints_m, "reference_joints_m", (count, 127, 3))
    if not np.isfinite(vertices).all():
        raise ValueError("Reference unit conversion is nonfinite")
    return vertices, joints


def geometry_diagnostics(baseline, candidate, baseline_joints, candidate_joints, faces) -> dict:
    """Finite/noncollapsed gates only; displacement is not an accuracy metric."""
    count = len(baseline)
    baseline = _float_array(baseline, "baseline_vertices_m", (count, 18439, 3))
    candidate = _float_array(candidate, "candidate_vertices_m", baseline.shape)
    baseline_joints = _float_array(baseline_joints, "baseline_joints_m", (count, 127, 3))
    candidate_joints = _float_array(candidate_joints, "candidate_joints_m", baseline_joints.shape)
    topology = _faces(faces, 18439, max_faces=36874)
    for vertices in (baseline, candidate):
        triangles = vertices[:, topology].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            areas = np.linalg.norm(np.cross(triangles[:, :, 1] - triangles[:, :, 0],
                                           triangles[:, :, 2] - triangles[:, :, 0]), axis=-1)
        if not np.isfinite(areas).all() or np.any(np.max(areas, axis=1) <= 0):
            raise ValueError("Reference geometry is nonfinite or collapsed")
    with np.errstate(over="ignore", invalid="ignore"):
        vertex_delta = np.linalg.norm(candidate.astype(np.float64) - baseline.astype(np.float64), axis=-1) * 1000
        joint_delta = np.linalg.norm(candidate_joints.astype(np.float64) - baseline_joints.astype(np.float64), axis=-1) * 1000
    if not np.isfinite(vertex_delta).all() or not np.isfinite(joint_delta).all():
        raise ValueError("Geometry displacement overflowed")
    return {
        "per_frame_mean_vertex_displacement_mm": vertex_delta.mean(axis=1).tolist(),
        "per_frame_max_vertex_displacement_mm": vertex_delta.max(axis=1).tolist(),
        "per_frame_max_joint_displacement_mm": joint_delta.max(axis=1).tolist(),
        "displacement_scope": "diagnostic_not_accuracy_or_adoption_gate",
        "finite_reference_geometry_verified": True, "fixed_faces_preserved": True,
        "joint_semantic_partition_verified": False, "nonhand_joint_invariance_verified": False,
    }


def _argument_parser() -> argparse.ArgumentParser:
    from body_smoke import EPISODE, TRACK1_EPISODE_COUNT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    return parser


def main() -> None:
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    args = _argument_parser().parse_args()
    root = Path(os.environ["WR_ROOT"])
    base = root / f"outputs/episode_{args.episode:06d}"
    output = base / "finger_transfer_smoke"
    if output.exists():
        raise RuntimeError("Frozen finger-transfer proposal already exists")
    final_path, hands_path = base / "cari_conversion/report.json", base / "body_hands_smoke/report.json"
    report_hashes = {"final": sha256(final_path), "hands": sha256(hands_path)}
    final, hands = json.loads(final_path.read_text()), json.loads(hands_path.read_text())
    count, indices = validate_producer_reports(final, hands, args.episode)
    episode_path, params_path = base / "cari_conversion/episode.npz", base / "cari_conversion/params.npz"
    source_path = base / "body_hands_smoke/predictions.npz"
    _require_hash(episode_path, final.get("episode_sha256"))
    _require_hash(params_path, final.get("params_sha256"))
    _require_hash(source_path, hands.get("predictions_sha256"))
    forward_path = base / "cari_forward/report.json"
    _require_hash(forward_path, final["input_report_sha256"]["forward"])
    report_hashes["forward"] = final["input_report_sha256"]["forward"]
    forward = json.loads(forward_path.read_text())
    # A separate audit binds legacy guard/wrapper source; no launch attestation.
    proof_path = base / "cari_forward/network_proof.json"
    network_proof_hash = require_forward_network(forward, report_hashes["forward"], final.get("bundle_sha256"), args.episode, proof_path)
    metadata = forward.get("metadata")
    if (not isinstance(metadata, Mapping) or metadata.get("actual_network_forward_verified") is not True
            or metadata.get("full_original_frame_coverage_verified") is not True
            or forward.get("episode_inputs_used") is not True
            or forward.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or forward.get("bundle_sha256") != final.get("bundle_sha256")):
        raise ValueError("Require SHA-bound actual full-video native network forward, not checkpoint load")
    _require_hash(base / "cari_forward/coconet.pth", final.get("bundle_sha256"))
    with np.load(episode_path, allow_pickle=False) as archive:
        fixed = {name: value.copy() for name, value in validate_final_arrays(archive, count).items()}
    with np.load(params_path, allow_pickle=False) as archive:
        for name in ("pose", "scales", "shape", "expression", "frame_index"):
            if archive[name].dtype != fixed[name].dtype or archive[name].tobytes() != fixed[name].tobytes():
                raise ValueError("Final episode human parameters differ from the frozen official conversion")
    with np.load(source_path, allow_pickle=False) as archive:
        controls, faces = (value.copy() for value in validate_source_arrays(archive, indices))
    # Import torch only after selected-episode provenance, SHA and array gates.
    import torch
    from body_smoke import BODY_REVISION, _body_assets, _source_identity
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/local fallback")
    source_identity = _source_identity(root)
    assets, asset_hashes = _body_assets(root)
    if (hands.get("body_revision") != BODY_REVISION or any(
        record.get("inference_source_identity") != source_identity or record.get("body_assets") != asset_hashes
        for record in (final, hands))):
        raise ValueError("Current pinned source/assets do not match both human producers")
    decoder = final.get("decoder_identity", {})
    if (decoder.get("mhr_model_sha256") != asset_hashes["assets/mhr_model.pt"]["sha256"]
            or decoder.get("mhr_buffer_sha256") != asset_hashes["model.ckpt"]["sha256"]
            or decoder.get("mhr_decoder_revision") != "cari4d.mhr_layer.vertices.v1"
            or (assets / "mhr_buffers.pt").exists()):
        raise ValueError("Native converter did not decode the same original SAM Body asset")
    payload = torch.load(assets / "model.ckpt", map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else None
    left, right = checkpoint_hand_indices(state, torch)
    del payload, state
    proposal = transfer_finger_controls(fixed["pose"], controls, indices, left, right)
    model, tool = root / "weights/mhr/mhr_model.pt", root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    _require_hash(model, REFERENCE_MODEL_SHA256)
    _require_hash(tool, CONVERTER_SHA256)
    spec = importlib.util.spec_from_file_location("world_reward_official_finger_reference", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the SHA-verified official reference tool")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    started = time.perf_counter()
    reference = converter.MHR(str(model), "cuda", chunk=16, precision="float32")
    identity = torch.tensor(np.concatenate([fixed["scales"], fixed["shape"]])[None], dtype=torch.float64, device="cuda")
    geometry = []
    for pose in (fixed["pose"][indices], proposal.pose[indices]):
        vertices_mm, joints_m = reference.run(torch.tensor(pose, dtype=torch.float64, device="cuda"), identity)
        geometry.append(reference_arrays(vertices_mm.detach().cpu().numpy(), joints_m.detach().cpu().numpy(), 3))
    diagnostics = geometry_diagnostics(geometry[0][0], geometry[1][0], geometry[0][1], geometry[1][1], faces)
    # Recheck producer bytes before publication; never modify the full baseline.
    for path, expected in ((episode_path, final["episode_sha256"]), (params_path, final["params_sha256"]),
                           (source_path, hands["predictions_sha256"]),
                           (final_path, report_hashes["final"]), (hands_path, report_hashes["hands"]),
                           (forward_path, report_hashes["forward"])):
        _require_hash(path, expected)
    if network_proof_hash is not None:
        _require_hash(proof_path, network_proof_hash)
    output.mkdir(exist_ok=False)
    with (output / "proposals.npz").open("xb") as handle:
        np.savez_compressed(handle, frame_index=indices, baseline_pose=fixed["pose"][indices],
                            candidate_pose=proposal.pose[indices], scales=fixed["scales"], shape=fixed["shape"],
                            expression=fixed["expression"], hand_indices_left=left, hand_indices_right=right)
    report = {
        "stage": STAGE, "status": "pass", "episode_index": args.episode,
        "forward_network_proof_sha256": network_proof_hash,
        "forward_network_provenance_basis": "explicit_report" if network_proof_hash is None else "audited_legacy_runtime_guard_contract_not_security_attestation",
        "input_track": "track_1", "input_sha256": final["input_sha256"], "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "network": "none",
        "total_video_frames": count, "frame_indices": indices.tolist(), "exported_candidate_frames": 3,
        "full_video_candidate_exported": False, "actual_reference_forward_verified": True,
        "proposal": proposal.report, "geometry": diagnostics,
        "human_identity_and_scales_preserved": True, "facial_expressions_zero": True,
        "geometry_units": "metres", "geometry_frame": "camera_x_right_y_down_z_forward",
        "reference_units": "MHR.run vertices_mm/1000; joints_m unchanged; no second flip/root/scale",
        "body_assets": asset_hashes, "inference_source_identity": source_identity,
        "body_revision": BODY_REVISION, "upstream_revision": UPSTREAM_REVISION,
        "reference_model_sha256": REFERENCE_MODEL_SHA256, "official_converter_sha256": CONVERTER_SHA256,
        "input_report_sha256": report_hashes,
        "final_episode_sha256": final["episode_sha256"], "final_params_sha256": final["params_sha256"],
        "source_predictions_sha256": hands["predictions_sha256"],
        "checkpoint_hand_index_source": "SHA-verified model.ckpt head_pose.hand_joint_idxs_left/right",
        "proposals_sha256": sha256(output / "proposals.npz"), "script_sha256": sha256(Path(__file__)),
        "producer_revision": os.environ.get("WR_CODE_REVISION"), "elapsed_seconds": time.perf_counter() - started,
        "adoption_performed": False, "submission_eligible": False, "hand_accuracy_verified": False,
        "challenge_performance_verified": False, "roi_image_validation_performed": False,
        "self_keypoint_reprojection_used_as_accuracy_gate": False,
        "gate_scope": "sparse_control_ABI_and_finite_reference_geometry_not_image_accuracy",
    }
    with (output / "report.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("stage", "status", "episode_index", "frame_indices",
                                                  "adoption_performed", "hand_accuracy_verified", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

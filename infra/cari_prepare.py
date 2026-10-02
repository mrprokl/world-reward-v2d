"""Prepare a GT-free native CARI export from verified World Reward predictions.

Original RGB/masks, fixed inferred camera, one human-anchored depth gauge and
one fixed object mesh. No tracking, oracle pose, hidden labels or GT files.
All large intermediates and model inputs remain on the remote managed disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from body_smoke import _validate_inputs, _pinned_checkout, UPSTREAM_REVISION
from world_reward.data import sha256
from world_reward.mesh_geometry import normalize_degenerate_faces


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux container with network none")
    import cv2
    import h5py
    import joblib
    import numpy as np
    from PIL import Image
    import trimesh
    root = Path(os.environ["WR_ROOT"])
    inputs = _validate_inputs(root)
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    native_root = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    sys.path.insert(0, str(native_root))
    from prep.prepare_mhr_wild_export import prepare_mhr_wild_export
    from prep.mhr_depth_h5 import MHRDepthH5Writer, validate_depth_h5, read_metric_depth
    from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
    from prep.mhr_export_utils import MHR_CAMERA_NAMES, frame_names, read_rgb, read_mask, camera_calibration, load_edex
    base = root / "outputs/episode_000015"
    output = base / "cari_inputs"
    if output.exists():
        raise RuntimeError("Frozen CARI inputs exist; never overwrite")
    reports = {}
    report_paths = {"body": base / "body_full/report.json", "depth": base / "depth_full/report.json",
                    "object": base / "object_pose_full/report.json", "alignment": base / "scale_smoke/report.json",
                    "adapter": base / "body_full/cari_adapter/report.json"}
    for key, path in report_paths.items():
        record = json.loads(path.read_text())
        expected = {"status": "pass", "input_track": "track_1", "ground_truth_used": False,
                    "hand_labeled_test": False, "oracle_modes": []}
        if (any(record.get(field) != value for field, value in expected.items())
                or record["ground_truth_used"] is not False or record["hand_labeled_test"] is not False):
            raise RuntimeError(f"{key} must have explicitly verified no-oracle provenance")
        if key != "adapter" and record.get("input_sha256") != inputs["video_sha256"]:
            raise RuntimeError(f"{key} belongs to another original video")
        reports[key] = record
    expected_stages = {"body": "sam3d_body_full_video_initializer", "depth": "monocular_moge2_full_video",
                       "object": "fixed_scale_full_object_pose_initializer", "alignment": "predicted_human_anchored_moge2_pointmaps",
                       "adapter": "native_cari_body_adapter_full_video"}
    if any(reports[key]["stage"] != value for key, value in expected_stages.items()):
        raise RuntimeError("Require exact full initializer stages, not sparse/test replacements")
    if reports["adapter"]["body_report_sha256"] != sha256(report_paths["body"]):
        raise RuntimeError("CARI Body adapter no longer matches verified original Body source")
    if reports["object"]["full_depth_report_sha256"] != sha256(report_paths["depth"]):
        raise RuntimeError("Object poses no longer match full inferred depth")
    if reports["object"]["alignment_report_sha256"] != sha256(report_paths["alignment"]):
        raise RuntimeError("Object poses no longer match shared human depth gauge")
    adapter_path = base / "body_full/cari_adapter/canonical_initializer.pkl"
    if sha256(adapter_path) != reports["adapter"]["canonical_initializer_sha256"]:
        raise RuntimeError("Canonical human initializer changed")
    pose_path = base / "object_pose_full/geometry_and_poses.npz"
    if sha256(pose_path) != reports["object"]["geometry_and_poses_sha256"]:
        raise RuntimeError("Frozen object geometry/poses changed")
    count = inputs["total_frames"]
    names = [f"{index:06d}" for index in range(count)]
    body_frames = {record["frame_index"]: record for record in reports["body"]["frames"]}
    depth_frames = {record["frame_index"]: record for record in reports["depth"]["frames"]}
    object_frames = {record["frame_index"]: record for record in reports["object"]["frames"]}
    for key, records in (("body", body_frames), ("depth", depth_frames), ("object", object_frames)):
        if len(records) != len(reports[key]["frames"]) or sorted(records) != list(range(count)):
            raise RuntimeError(f"{key} lacks exact full original-frame coverage")
    with np.load(pose_path, allow_pickle=False) as arrays:
        vertices, faces = arrays["vertices"].copy(), arrays["faces"].copy()
        rotations, translations = arrays["rotation"].copy(), arrays["translation"].copy()
        if not np.array_equal(arrays["frame_index"], np.arange(count)) or float(arrays["object_scale"]) != 1.:
            raise RuntimeError("Require full fixed-metric-gauge mesh with scale already baked once")
    if (rotations.shape != (count, 3, 3) or translations.shape != (count, 3)
            or not np.isfinite(rotations).all() or not np.isfinite(translations).all()
            or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        raise RuntimeError("Full object trajectory must contain proper finite rigid poses")
    active, _ = normalize_degenerate_faces(vertices, faces)
    metric_mesh = trimesh.Trimesh(vertices, faces[active], process=True)
    if not metric_mesh.is_watertight or not metric_mesh.is_winding_consistent or metric_mesh.volume <= 0:
        raise RuntimeError("Packed fixed geometry must remain closed and correctly oriented")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    sequence = "episode_000015"
    video_link = output / (sequence + ".0.color.mp4")
    # Native prep resolves symlinks before validating its .0.color.mp4 ABI.
    # Same managed disk: a hardlink preserves bytes without duplicating video.
    os.link(inputs["video"], video_link)
    if sha256(video_link) != inputs["video_sha256"]:
        raise RuntimeError("Native video alias differs from original Track 1 bytes")
    metric_path = output / "object_metric.glb"
    metric_mesh.export(metric_path)
    focal = float(np.hypot(1152, 1536))
    intrinsics = {"fx": focal, "fy": focal, "cx": 768., "cy": 576., "H": 1152, "W": 1536,
                  "depth_backend": "moge2", "model_id": MOGE2_MODEL_ID, "model_revision": MOGE2_MODEL_REVISION,
                  "source_commit": MOGE2_SOURCE_COMMIT, "camera_policy": "original_RGB_size_prior_K_no_GT_calibration"}
    intrinsic_path = output / "intrinsics.pkl"
    joblib.dump(intrinsics, intrinsic_path)
    masks_path = output / "automatic_masks.h5"
    with h5py.File(masks_path, "x") as handle:
        for index, name in enumerate(names):
            for mask_id, kind in ((0, "person_mask.png"), (1, "obj_rend_mask.png")):
                path = base / f"automatic_masks/masks/{mask_id}/{name}.png"
                expected_hash = body_frames[index]["mask_sha256"] if mask_id == 0 else object_frames[index]["object_mask_sha256"]
                if sha256(path) != expected_hash:
                    raise RuntimeError("Automatic mask changed after body/object prediction")
                with Image.open(path) as image:
                    array = np.asarray(image)
                if array.shape != (1152, 1536) or not np.isin(array, [0, 255]).all() or not (array > 0).any():
                    raise RuntimeError("Automatic mask must retain original binary full-resolution observation")
                handle.create_dataset(f"{sequence}/{name}-k0.{kind}", data=array, compression="lzf")
    export_seq = prepare_mhr_wild_export(video_link, masks_path, metric_path, intrinsic_path, output / "export")
    metadata_path = export_seq / "wild_export.json"
    metadata = json.loads(metadata_path.read_text())
    A = np.asarray(metadata["source_object_mesh_to_aligned_transform"], dtype=np.float64)
    if A.shape != (4, 4) or not np.allclose(A[3], [0, 0, 0, 1]) or not np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(A[:3, :3]), 1, atol=1e-5):
        raise RuntimeError("Native mesh preparation must be a proper rigid frame change, not rescaling")
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    poses[:, :3, :3], poses[:, :3, 3] = rotations, translations
    aligned_poses = poses @ np.linalg.inv(A)
    original_points = vertices[faces[active].reshape(-1)]
    aligned_points = original_points @ A[:3, :3].T + A[:3, 3]
    frame_transform_error = 0.
    for index in range(count):
        before = original_points @ poses[index, :3, :3].T + poses[index, :3, 3]
        after = aligned_points @ aligned_poses[index, :3, :3].T + aligned_poses[index, :3, 3]
        frame_transform_error = max(frame_transform_error, float(np.max(np.linalg.norm(after - before, axis=-1))))
    if frame_transform_error > 1e-5:
        raise RuntimeError("Aligned mesh/pose pair changed camera-space geometry")
    object_poses_path = output / "own_object_poses.pkl"
    joblib.dump({"frames": names, "obj_pose_world": aligned_poses.astype(np.float32),
                 "metadata": {"source": "World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose",
                              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
                              "source_pose_sha256": sha256(pose_path), "mesh_frame_change": A.tolist()}}, object_poses_path)
    aligned_depth_path = output / "aligned_depth.h5"
    scale = reports["alignment"]["depth_alignment"]["shared_scale"]
    identity = {"depth_backend": "moge2", "depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION,
                "depth_source_commit": MOGE2_SOURCE_COMMIT, "alignment_report_sha256": sha256(report_paths["alignment"]),
                "monocular_depth": {"backend": "moge2", "model_id": MOGE2_MODEL_ID, "model_revision": MOGE2_MODEL_REVISION,
                                    "source_commit": MOGE2_SOURCE_COMMIT},
                "ground_truth_used": False, "alignment": "one_predicted_human_anchored_clip_scalar_no_offset"}
    camera_name = MHR_CAMERA_NAMES[0]
    with MHRDepthH5Writer(aligned_depth_path, {camera_name: names}, alignment_method="world_reward_shared_predicted_human_scale",
                          alignment_input_identity=identity, encoding_workers=8) as writer:
        for index, name in enumerate(names):
            path = base / f"depth_full/{name}.npz"
            if sha256(path) != depth_frames[index]["output_sha256"]:
                raise RuntimeError("Full depth artifact changed")
            with np.load(path, allow_pickle=False) as arrays:
                depth, valid = arrays["depth"].copy(), arrays["mask"].copy()
            raw = np.where(valid, depth, 0.)
            aligned = raw * scale
            if not np.isfinite(aligned).all() or (aligned < 0).any() or (raw > 65.535).any() or (aligned > 65.535).any():
                raise RuntimeError("Depth encoding would silently saturate uint16 metres-to-mm representation")
            writer.write_frame(camera_name, index, raw, aligned, scale=scale, shift=0., valid_count=int(valid.sum()))
            if (index + 1) % 50 == 0:
                print(json.dumps({"stage": "cari_prepare_depth", "frames_complete": index + 1}), flush=True)
        writer.mark_complete()
    depth_validation = validate_depth_h5(aligned_depth_path, expected_cameras=[camera_name], expected_alignment_input_identity=identity,
                                         validation_workers=8)
    if frame_names(export_seq) != names:
        raise RuntimeError("Native RGB export changed original frame identities")
    K, extrinsic = camera_calibration(load_edex(export_seq), 0)
    if not np.array_equal(extrinsic, np.eye(4)) or not np.allclose(K, [[focal, 0, 768], [0, focal, 576], [0, 0, 1]], atol=1e-5):
        raise RuntimeError("Native export changed the immutable inferred camera")
    # Original RGB decode hash checked before official JPEG export; quantify,
    # do not pretend JPEG q100 is byte-identical to original model observations.
    cap = cv2.VideoCapture(str(inputs["video"]))
    jpeg_errors = []
    try:
        for index, name in enumerate(names):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("Original RGB verification decode failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != body_frames[index]["decoded_rgb_sha256"] or body_frames[index]["decoded_rgb_sha256"] != depth_frames[index]["decoded_rgb_sha256"]:
                raise RuntimeError("Prepared RGB must derive from the same Body/depth original video")
            exported = read_rgb(export_seq, 0, name)
            jpeg_errors.append(float(np.mean(np.abs(exported.astype(float) - rgb))))
            for mask_id, kind in ((0, "human"), (1, "object")):
                with Image.open(base / f"automatic_masks/masks/{mask_id}/{name}.png") as image:
                    expected = np.asarray(image) > 0
                if not np.array_equal(read_mask(export_seq, kind, 0, name), expected):
                    raise RuntimeError("Native mask export changed automatic observations")
    finally:
        cap.release()
    for name in names:
        quantized = read_metric_depth(aligned_depth_path, "aligned", camera_name, name)
        if quantized.shape != (1152, 1536) or not np.isfinite(quantized).all():
            raise RuntimeError("Native quantized depth decode contract failed")
    result = {"stage": "world_reward_native_cari_inputs", "status": "pass", "frames": count,
              "export_seq": str(export_seq), "depth_h5": str(aligned_depth_path), "mhr_init": str(adapter_path),
              "object_poses": str(object_poses_path), "object_pose_initializer": "own_ICP_Viterbi_not_FoundationPose",
              "depth_validation": depth_validation, "mesh_pose_frame_roundtrip_max_error_m": frame_transform_error,
              "jpeg_original_RGB_mean_absolute_error": float(np.mean(jpeg_errors)), "original_frame_coverage_verified": True,
              "input_track": "track_1", "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "submission_eligible": False,
              "challenge_performance_verified": False, "producer_revision": os.environ.get("WR_CODE_REVISION"),
              "input_report_sha256": {key: sha256(path) for key, path in report_paths.items()},
              "file_sha256": {key: sha256(path) for key, path in {"depth_h5": aligned_depth_path, "mhr_init": adapter_path,
                                                               "object_poses": object_poses_path, "wild_export": metadata_path}.items()},
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    # Transient combined masks are redundant after verified native export.
    masks_path.unlink()
    print(json.dumps({"stage": result["stage"], "status": "pass", "frames": count,
                      "mesh_pose_error_m": frame_transform_error, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

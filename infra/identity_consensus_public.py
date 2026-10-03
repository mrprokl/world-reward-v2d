"""Public-video-only identity medoid: consistency, never accuracy/adoption.

Twelve native neutral meshes choose one actual raw shape45/PCA28 identity by
full-vertex RMS, before scoring eleven disjoint temporal midpoints. Both posed
renders and the observed human use the SAME frozen automatic-object exclusion.
No GT, coefficient averaging, labels, image inference, inverse fit or alignment.
Historical 0644 inputs stay unchanged; exact hashes and read-only mounts, not
invented immutable file modes, establish the inputs of this bounded experiment.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import time

import numpy as np

import body_smoke as body
import camera_render as camera
from world_reward.data import sha256

STAGE = "public_native_neutral_mesh_identity_consensus"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BASE = "outputs/episode_000000/identity_consensus_public_v1"
FRAMES, WIDTH, HEIGHT, VERTICES, FACES, BUDGET = 790, 1536, 1152, 18439, 36874, 180
ANCHORS = (0, 72, 143, 215, 287, 359, 430, 502, 574, 646, 717, 789)
MIDPOINTS = (36, 107, 179, 251, 323, 394, 466, 538, 610, 681, 753)
SELECTED = tuple(sorted(ANCHORS + MIDPOINTS))
BODY_REPORT_SHA = "d11d9caf37c8f1d2a92731799f6c05395f7b77cfdd200b01c98f778bf77e0fa5"
PREDICTIONS_SHA = "7d0b67054ec6bf0dd07b9bb16ee5aa70626c91aef476d2e86987437ca762e6b2"
MASK_REPORT_SHA = "2bb43c1ff04f0e8ad7ce5d95dddeb66f12096ce8ee79819684fd57298b4f01f0"
BODY_SCRIPT_SHA = "e4d659de33bacff9d5c85c1cb2fedce34f4aa34bdbae93b3b9b7e08950613ce3"
# BODY_SCRIPT_SHA belongs to the original receipt, not today's consumer helper.
# Authenticate the helper actually imported without relabeling the producer.
BODY_HELPER_BYTES = 33592
BODY_HELPER_SHA = "3f662f88cd212ad1a04b59b66ccdb5e32a00829b52f5e224daadfbe519e1bc93"
MASK_SCRIPT_SHA = "b6d5b543ace41c1cca6b327182ff23528328a1052ead0f4f1d7423dc7404e234"
VIDEO_SHA = "1eb6293e4552397668a5013adbcee95e0bf230dd8c79a594aa7d7f8ea4991ab1"
CHECKPOINT_SHA = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
BODY_REPORT_BYTES, MASK_REPORT_BYTES, PREDICTIONS_BYTES, CHECKPOINT_BYTES = 504222, 8525, 315456910, 2109129346
BLOCKS = {"global_rot": 3, "body_pose_params": 133, "hand_pose_params": 108,
          "scale_params": 28, "shape_params": 45, "expr_params": 72}
PARITY_BOUND, MIDPOINT_REGRESSION_BOUND = 1e-5, 1e-4


def regular(path):
    path = Path(path)
    if path.resolve() != path.absolute() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Require a regular file without symlink parents")
    return path


def freeze_input(path, frozen, *, digest=None, size=None):
    path = regular(path); actual = sha256(path)
    if (digest is not None and actual != digest) or (size is not None and path.stat().st_size != size):
        raise ValueError("Frozen public source SHA/size differs")
    frozen.append((path, actual))
    return path


def require_fields(report, expected):
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Explicit source provenance/ABI fields differ")


def validate_receipts(raw, masks):
    common = {"status": "pass", "episode_index": 0, "input_track": "track_1", "input_sha256": VIDEO_SHA,
              "input_dataset_revision": body.DATASET_REVISION, "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": []}
    require_fields(raw, {**common, "stage": "sam3d_body_full_video_initializer", "total_video_frames": FRAMES,
        "frame_indices": list(range(FRAMES)), "script_sha256": BODY_SCRIPT_SHA, "predictions_sha256": PREDICTIONS_SHA, "network": "none",
        "mask_report_sha256": MASK_REPORT_SHA, "inference_type": "body", "body_revision": body.BODY_REVISION,
        "upstream_revision": body.UPSTREAM_REVISION, "camera_intrinsics": "RGB_size_default_FOV",
        "human_mask_id": 0, "prompt_mode": "automatic_mask_and_derived_bbox_no_fallback",
        "model_mask_range": "uint8_0_1_matches_CARI_prepare_batch", "vertices": VERTICES, "faces": FACES,
        "geometry_units": "metres", "geometry_frame": "SAM3D_camera_x_right_y_down_z_forward",
        "translation": "vertices_camera_m = vertices_root_camera_m + pred_cam_t (exactly once)",
        "mhr_geometry_forward_verified": True,
        "mhr_geometry_forward_basis": "original_body133_hand108_global_rot3_scale28_shape45_expr72_not_raw_logits"})
    require_fields(masks, {**common, "stage": "automatic_masks", "frames": FRAMES, "script_sha256": MASK_SCRIPT_SHA})
    rows = raw.get("frames")
    if not isinstance(rows, list) or len(rows) != FRAMES:
        raise ValueError("Original Body receipt must cover all790 frames")
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("frame_index")) is not int or row["frame_index"] != index
                or any(not re.fullmatch("[0-9a-f]{64}", str(row.get(k, ""))) for k in ("mask_sha256", "decoded_rgb_sha256"))):
            raise ValueError("Full original ordered RGB/mask provenance required")
    loading = raw.get("checkpoint_loading", {})
    require_fields(loading, {"mode": "strict_network_and_head_state_with_explicit_asset_buffer_retention",
                            "parameter_tensors_loaded": 1101, "unexpected_keys": []})
    retained = loading.get("retained_mhr_asset_buffer_names")
    if not isinstance(retained, list) or len(retained) != 113 or len(set(retained)) != 113:
        raise ValueError("Exact original113 retained immutable asset buffers required")


def load_predictions(path):
    """Validate all original rows; retain only23 fixed rows, not a new subset fit."""
    shapes = {k: (FRAMES, *v) for k, v in body.PARAMETER_SHAPES.items() if k != "pred_vertices"}
    shapes.update(vertices_root_camera_m=(FRAMES, VERTICES, 3), vertices_camera_m=(FRAMES, VERTICES, 3),
                  faces=(FACES, 3), frame_index=(FRAMES,))
    retained = {*BLOCKS, "vertices_root_camera_m", "vertices_camera_m", "mhr_model_params", "pred_cam_t", "focal_length"}
    selected = {}
    with np.load(path, allow_pickle=False) as file:
        if len(file.files) != len(shapes) or set(file.files) != set(shapes):
            raise ValueError("Exact frozen Body NPZ schema required; no extra oracle arrays")
        for key, shape in shapes.items():
            value = file[key]
            dtype = np.int64 if key in ("faces", "frame_index") else np.float32
            if (np.ma.isMaskedArray(value) or value.dtype != dtype or value.shape != shape or not np.isfinite(value).all()):
                raise ValueError("Complete original finite native dtype/shape required: " + key)
            if key == "frame_index" and not np.array_equal(value, np.arange(FRAMES)):
                raise ValueError("Original full frame indices changed")
            if key == "expr_params" and np.any(value != 0): raise ValueError("Native face expressions must be zero")
            if key == "focal_length" and not np.allclose(value, np.hypot(WIDTH, HEIGHT), rtol=1e-6, atol=1e-4):
                raise ValueError("Original RGB-size-only focal changed; no calibration replacement")
            if key == "faces":
                if (value.min() < 0 or value.max() >= VERTICES or np.any(np.diff(np.sort(value, axis=1), axis=1) == 0)):
                    raise ValueError("Original full face topology invalid")
                selected[key] = value.copy()
            elif key in retained: selected[key] = value[list(SELECTED)].copy()
    if (not np.array_equal(selected["vertices_root_camera_m"] + selected["pred_cam_t"][:, None], selected["vertices_camera_m"])
            or (selected["pred_cam_t"][:, 2] <= 0).any()):
        raise ValueError("Original camera translation must occur exactly once")
    return selected


def mask_evidence(raw_mask, candidate_mask, human, object_mask, index):
    values = (raw_mask, candidate_mask, human, object_mask)
    if any(not isinstance(v, np.ndarray) or np.ma.isMaskedArray(v) or v.dtype != np.bool_
           or v.shape != (HEIGHT, WIDTH) for v in values):
        raise ValueError("Four complete original-grid boolean masks required")
    region = ~object_mask; observed = human & region
    count = int(np.count_nonzero(observed))
    if count <= 32: raise ValueError("Every selected human must have >32 pixels outside automatic object")
    raw_iou = camera.silhouette_iou(raw_mask & region, observed)
    candidate_iou = camera.silhouette_iou(candidate_mask & region, observed)
    return {"frame_index": index, "raw_iou": raw_iou, "candidate_iou": candidate_iou,
            "delta_iou": candidate_iou-raw_iou, "observed_human_pixels_outside_object": count,
            "excluded_object_pixels": int(object_mask.sum()), "comparison_region_pixels": int(region.sum())}


def neutral_medoid(meshes):
    """Full native vertex RMS in metres, without normalization/alignment."""
    if (not isinstance(meshes, np.ndarray) or np.ma.isMaskedArray(meshes) or meshes.dtype != np.float32
            or meshes.shape != (len(ANCHORS), VERTICES, 3) or not np.isfinite(meshes).all()
            or (np.ptp(meshes.astype(np.float64), axis=1) <= 0).any()):
        raise ValueError("Twelve complete finite noncollapsed neutral native meshes required")
    distances = np.zeros((len(ANCHORS), len(ANCHORS)), np.float64)
    for i in range(len(ANCHORS)):
        for j in range(i):
            d = meshes[i].astype(np.float64)-meshes[j].astype(np.float64)
            distances[i, j] = distances[j, i] = np.sqrt(np.mean(np.sum(d*d, axis=-1)))
    if not np.isfinite(distances).all(): raise ValueError("Neutral mesh RMS overflow")
    costs = distances.sum(axis=1)
    return int(np.argmin(costs)), distances, costs


def candidate_blocks(raw, anchor_position):
    if type(anchor_position) is not int or not 0 <= anchor_position < len(ANCHORS):
        raise ValueError("Medoid must be one protocol anchor, not an averaged identity")
    result = {key: raw[key].copy() for key in BLOCKS}
    row = SELECTED.index(ANCHORS[anchor_position])
    for key in ("shape_params", "scale_params"):
        result[key] = np.repeat(raw[key][row:row+1], len(SELECTED), axis=0)
    return result


def freeze_candidate(out, raw, position, distances, costs):
    blocks = candidate_blocks(raw, position); path = out/"selected_identity.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, anchor_index=np.asarray(ANCHORS[position], np.int64),
            shape_params=blocks["shape_params"][0], scale_params=blocks["scale_params"][0],
            anchors=np.asarray(ANCHORS, np.int64), midpoints=np.asarray(MIDPOINTS, np.int64),
            neutral_vertex_rms_m=distances, medoid_costs_m=costs)
    path.chmod(0o444)
    with np.load(path, allow_pickle=False) as saved:
        if any(saved[k].tobytes() != blocks[k][0].tobytes() for k in ("shape_params", "scale_params")):
            raise ValueError("Frozen selected identity differs before image scoring")
    return blocks, (path, sha256(path))


def decision(records):
    if not isinstance(records, list) or len(records) != len(SELECTED):
        raise ValueError("All23 ordered frames required; no failed midpoint dropping")
    for index, row in zip(SELECTED, records):
        if type(row.get("frame_index")) is not int or row["frame_index"] != index:
            raise ValueError("Exact original selected frame order required")
        for key in ("raw_iou", "candidate_iou", "delta_iou"):
            if type(row.get(key)) is not float or not np.isfinite(row[key]): raise ValueError("Finite IoU evidence required")
        if (not 0 <= row["raw_iou"] <= 1 or not 0 <= row["candidate_iou"] <= 1
                or row["delta_iou"] != row["candidate_iou"]-row["raw_iou"]
                or type(row.get("observed_human_pixels_outside_object")) is not int
                or row["observed_human_pixels_outside_object"] <= 32):
            raise ValueError("Complete unmodified full-region mask evidence required")
    deltas = [r["delta_iou"] for r in records if r["frame_index"] in MIDPOINTS]
    mean = float(np.mean(deltas)); worst = float(min(deltas))
    return {"midpoint_mean_delta_iou": mean, "midpoint_worst_delta_iou": worst,
            "consensus_gate_pass": mean >= 0 and worst >= -MIDPOINT_REGRESSION_BOUND,
            "adoption_authorized": False, "accuracy_verified": False}


def source_inputs(root):
    frozen = []; base = root/"outputs/episode_000000"; masks_dir = base/"automatic_masks"
    freeze_input(Path(body.__file__), frozen, digest=BODY_HELPER_SHA, size=BODY_HELPER_BYTES)
    raw_path = freeze_input(base/"body_full/report.json", frozen, digest=BODY_REPORT_SHA, size=BODY_REPORT_BYTES)
    masks_path = freeze_input(masks_dir/"report.json", frozen, digest=MASK_REPORT_SHA, size=MASK_REPORT_BYTES)
    raw, masks = json.loads(raw_path.read_text()), json.loads(masks_path.read_text()); validate_receipts(raw, masks)
    manifest_path = freeze_input(root/"results/input-manifest.json", frozen)
    manifest = json.loads(manifest_path.read_text())
    require_fields(manifest, {"track": "track_1", "repo_id": "nvidia/video_to_data_challenge", "revision": body.DATASET_REVISION})
    meta, _ = body._manifest_file(root, manifest, "track_1/meta/episodes.jsonl"); freeze_input(meta, frozen)
    rows = [json.loads(line) for line in meta.read_text().splitlines()]
    episodes = [r for r in rows if type(r.get("episode_index")) is int and r["episode_index"] == 0]
    if len(episodes) != 1 or type(episodes[0].get("length")) is not int or episodes[0]["length"] != FRAMES:
        raise ValueError("Original manifest790 frame coverage differs")
    video = "track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4"
    videos = [r for r in manifest["files"] if r.get("path") == video]
    if len(videos) != 1 or videos[0].get("sha256") != VIDEO_SHA:
        raise ValueError("Original public video receipt identity differs; video is not opened")
    prompts_path = freeze_input(masks_dir/"prompts.json", frozen, digest=raw["prompts_sha256"])
    prompts = json.loads(prompts_path.read_text()).get("prompts")
    if (not isinstance(prompts, list) or len(prompts) != 2 or {p.get("object_id") for p in prompts} != {0, 1}
            or any(p.get(k) is not None for p in prompts for k in ("points", "point_labels", "mask_path"))):
        raise ValueError("Only original automatic person/object prompts allowed")
    expected = [f"{i:06d}.png" for i in range(FRAMES)]
    for label in (0, 1):
        directory = masks_dir/f"masks/{label}"
        if (directory.resolve() != directory.absolute() or sorted(p.name for p in directory.iterdir()) != expected
                or any(not p.is_file() or p.is_symlink() for p in directory.iterdir())):
            raise ValueError("Automatic mask inventory must retain all790 original indices")
    masks_selected, mask_records = [], []
    from PIL import Image
    for index in SELECTED:
        pair, row = [], {"frame_index": index, "decoded_rgb_sha256": raw["frames"][index]["decoded_rgb_sha256"]}
        for label, name in ((0, "human"), (1, "object")):
            path = freeze_input(masks_dir/f"masks/{label}/{index:06d}.png", frozen,
                                digest=raw["frames"][index]["mask_sha256"] if label == 0 else None)
            with Image.open(path) as image:
                value = np.asarray(image)
                if (image.format != "PNG" or image.mode != "L" or value.dtype != np.uint8
                        or value.shape != (HEIGHT, WIDTH) or not np.isin(value, [0, 255]).all()):
                    raise ValueError("Original automatic binary PNG grid differs")
                pair.append(value > 0)
            row[name+"_mask_sha256"] = frozen[-1][1]
        mask_evidence(pair[0], pair[0], *pair, index)  # Fail missing support before selecting identity.
        masks_selected.append(pair); mask_records.append(row)
    predictions = freeze_input(base/"body_full/predictions.npz", frozen, digest=PREDICTIONS_SHA, size=PREDICTIONS_BYTES)
    arrays = load_predictions(predictions)
    freeze_input(root/"results/weights-acquisition.json", frozen)
    assets, identities = body._body_assets(root)
    if identities != raw.get("body_assets") or identities["model.ckpt"] != {"bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA}:
        raise ValueError("Original pinned checkpoint/explicit head asset identity differs")
    for name, identity in identities.items(): freeze_input(assets/name, frozen, digest=identity["sha256"], size=identity["bytes"])
    installed = body._source_identity(root)
    if installed != raw.get("inference_source_identity"): raise ValueError("Pinned installed Body source differs from original")
    return arrays, masks_selected, mask_records, assets, identities, installed, frozen


def strict_torch():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise ValueError("CUBLAS required BEFORE torch import")
    import random
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise RuntimeError("Azure CUDA required; no CPU fallback")
    return torch


def forward_blocks(head, blocks, torch, *, neutral=False):
    tensors = {k: torch.tensor(blocks[k].copy(), device="cuda", dtype=torch.float32) for k in BLOCKS}
    count = len(tensors["shape_params"])
    if neutral:
        for key in ("global_rot", "body_pose_params", "hand_pose_params", "expr_params"):
            tensors[key] = torch.zeros_like(tensors[key])
    return head.mhr_forward(global_trans=torch.zeros((count, 3), device="cuda", dtype=torch.float32),
        **tensors, return_keypoints=False, return_joint_coords=False,
        return_model_params=not neutral, return_joint_rotations=False)


def run(root, out, report, persist):
    raw, masks, mask_records, assets, asset_ids, installed, frozen = source_inputs(root)
    report.update(phase="native_head_load", input_mask_records=mask_records, body_assets=asset_ids,
        inference_source_identity=installed, input_sha256=VIDEO_SHA,
        historical_inputs_modes={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):
            oct(stat.S_IMODE(p.stat().st_mode)) for p, _ in frozen}, historical_body_source_hash_matched=False,
        consumer_body_helper_hash_verified=True, historical_body_script_identity_verified_from_receipt=True,
        historical_receipt_hash_allowlist_verified=True); persist()
    torch = strict_torch()
    vendor = root/"vendor/video_to_data"; native = vendor/"reconstruction/modules/v2d_cari4d/lib/cari4d"
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root/"weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
    from lib_mhr.mhr_layer import MHRLayer
    layer_source = regular(Path(sys.modules[MHRLayer.__module__].__file__))
    if layer_source != native/"lib_mhr/mhr_layer.py": raise ValueError("Pinned native layer source path differs")
    freeze_input(layer_source, frozen)
    absent_buffer = out/"unverified_buffer_must_not_exist.pt"
    if absent_buffer.exists() or absent_buffer.is_symlink(): raise ValueError("No unverified compact head buffer allowed")
    with torch.no_grad(), torch.jit.optimized_execution(False):
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),
            checkpoint_path=assets/"model.ckpt", buffer_path=absent_buffer,
            mhr_model_path=assets/"assets/mhr_model.pt", device="cuda")
        head = layer.backend._ensure_head(torch.device("cuda"))
        head_source = regular(Path(sys.modules[type(head).__module__].__file__))
        if (head_source != Path("/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py")
                or head.enable_hand_model or not np.array_equal(head.faces.cpu().numpy(), raw["faces"])
                or tuple(head.scale_mean.shape) != (68,) or tuple(head.scale_comps.shape) != (28, 68)):
            raise ValueError("Original full-body native topology/PCA head ABI differs")
        freeze_input(head_source, frozen)
        report.update(native_layer_sha256=sha256(layer_source), native_head_sha256=sha256(head_source),
            phase="neutral_mesh_identity_selection", torch_version=str(torch.__version__), cuda_version=torch.version.cuda); persist()
        anchor_rows = [SELECTED.index(i) for i in ANCHORS]
        neutral = forward_blocks(head, {k: raw[k][anchor_rows] for k in BLOCKS}, torch, neutral=True)
        report["actual_head_calls"] += 1
        position, distances, costs = neutral_medoid(neutral.cpu().numpy().copy()); del neutral
        candidate, candidate_file = freeze_candidate(out, raw, position, distances, costs); frozen.append(candidate_file)
        report.update(identity_frozen_before_image_scoring=True, medoid_anchor_index=ANCHORS[position],
            selected_identity_sha256=candidate_file[1], neutral_vertex_rms_m=distances.tolist(), medoid_costs_m=costs.tolist(),
            phase="raw_native_parity"); persist()
        root_v, controls = forward_blocks(head, raw, torch); report["actual_head_calls"] += 1
        raw_root = root_v.cpu().numpy().copy()*np.array([1., -1., -1.], np.float32)
        raw_controls = controls.cpu().numpy().copy(); del root_v, controls
        raw_camera = raw_root + raw["pred_cam_t"][:, None]
        parity = {"root_max_point_error_m": float(np.linalg.norm(raw_root.astype(np.float64)-raw["vertices_root_camera_m"], axis=-1).max()),
            "camera_max_point_error_m": float(np.linalg.norm(raw_camera.astype(np.float64)-raw["vertices_camera_m"], axis=-1).max()),
            "controls_max_absolute_error": float(np.abs(raw_controls.astype(np.float64)-raw["mhr_model_params"]).max())}
        report["raw_native_parity"] = parity; persist()
        if not all(np.isfinite(v) and v <= PARITY_BOUND for v in parity.values()):
            raise ValueError("Original native parameter/vertex replay exceeds frozen1e-5 bound")
        verts, controls = forward_blocks(head, candidate, torch); report["actual_head_calls"] += 1
        candidate_camera = verts.cpu().numpy().copy()*np.array([1., -1., -1.], np.float32) + raw["pred_cam_t"][:, None]
        candidate_controls = controls.cpu().numpy().copy(); del verts, controls
        if (candidate_controls[:, :136].tobytes() != raw_controls[:, :136].tobytes()
                or any(row.tobytes() != candidate_controls[0, 136:].tobytes() for row in candidate_controls[:, 136:])
                or not np.isfinite(candidate_camera).all()):
            raise ValueError("Pose/root/hands or clip-constant native68 scale identity changed")
        for j, index in enumerate(SELECTED):
            K = np.array([[raw["focal_length"][j], 0., WIDTH/2.], [0., raw["focal_length"][j], HEIGHT/2.], [0., 0., 1.]], np.float64)
            report.update(phase="public_silhouette_consistency", current_frame_index=index); persist()
            pair = []
            for vertices in (raw_camera[j], candidate_camera[j]):
                predicted, depth = camera.raster_camera_mesh(vertices, raw["faces"], K, WIDTH, HEIGHT)
                report["actual_raster_calls"] += 1
                pair.append(predicted.cpu().numpy().copy()); del predicted, depth
            report["records"].append(mask_evidence(*pair, *masks[j], index)); persist()
    report.update(phase="post_run_integrity"); persist()
    for path, digest in frozen:
        if sha256(regular(path)) != digest: raise ValueError("Frozen public input/candidate/source changed")
    if body._source_identity(root) != installed: raise ValueError("Installed native source changed during gate")
    if report["actual_head_calls"] != 3 or report["actual_raster_calls"] != 46:
        raise ValueError("Exact3 headcalls/46 identical rastercalls required")
    report.update(**decision(report["records"]), status="pass", phase="complete",
        source_bindings_runtime_verified=True, original_pose_root_hands_camera_unchanged=True,
        historical_inputs_rehashed_after_run=True, original_masks_rehashed_after_run=True,
        selected_identity_rehashed_after_scoring=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); revision = os.environ["WR_CODE_REVISION"]; image = os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGE):
        raise ValueError("Canonical Azure offline root/source revision/pinned image required")
    out = root/BASE
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty new public consensus output required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "helper_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("body_smoke.py", "camera_render.py")}, "input_track": "track_1", "episode_index": 0, "original_frames": FRAMES,
        "anchor_indices": list(ANCHORS), "midpoint_indices": list(MIDPOINTS), "selected_frame_indices": list(SELECTED),
        "body_report_sha256": BODY_REPORT_SHA, "predictions_sha256": PREDICTIONS_SHA, "automatic_mask_report_sha256": MASK_REPORT_SHA,
        "historical_body_script_sha256": BODY_SCRIPT_SHA, "historical_mask_script_sha256": MASK_SCRIPT_SHA,
        "consumer_body_helper_identity": {"bytes": BODY_HELPER_BYTES, "sha256": BODY_HELPER_SHA},
        "consumer_body_helper_hash_verified": False, "historical_body_script_identity_verified_from_receipt": False,
        "historical_receipt_hash_allowlist_verified": False, "historical_body_source_hash_matched": False,
        "historical_body_producer_revision_available": False,
        "historical_producer_source_executed_by_this_gate": False, "input_video_read": False,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "network": "none",
        "accuracy_verified": False, "adoption_authorized": False, "submission_produced": False,
        "full_trajectory_reconstruction_performed": False, "official_converter_called": False,
        "coefficient_averaging_performed": False, "alignment_calls": 0, "solver_calls": 0,
        "identity_selection_rule": "minimum_sum_full_neutral_vertex_RMS_actual_anchor_first_protocol_tie",
        "comparison_region": "complement_of_same_frozen_automatic_object_mask_for_both_modes_and_observation",
        "human_pixels_outside_object_minimum_exclusive": 32, "midpoint_mean_delta_iou_minimum": 0.,
        "midpoint_per_frame_regression_bound": MIDPOINT_REGRESSION_BOUND, "raw_native_parity_bound": PARITY_BOUND,
        "statistical_block_independence_verified": False, "heldout_accuracy_verified": False,
        "input_filesystem_immutability_claimed": False, "historical_object_PNG_hashes_available": False,
        "read_only_input_mounts_required": True, "source_bindings_runtime_verified": False,
        "actual_head_calls": 0, "actual_raster_calls": 0, "records": [], "budget_seconds": BUDGET,
        "settings": {"seed": 0, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False,
            "deterministic_algorithms": True, "jit_optimized_execution": False, "dtype": "float32", "CPU_threads": 4}}
    with (out/"report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Public identity consensus exceeded180s")
        previous = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, out, report, persist)
        except BaseException as error:
            report.update(status="fail", error=type(error).__name__, message=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous); signal.signal(signal.SIGTERM, term)
            persist(); (out/"report.json").chmod(0o444)
    print(json.dumps({k: report[k] for k in ("stage", "status", "consensus_gate_pass", "medoid_anchor_index", "elapsed_seconds")}))


if __name__ == "__main__": main()

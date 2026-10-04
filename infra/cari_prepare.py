"""Prepare a GT-free native CARI export from verified World Reward predictions.

Original RGB/masks, fixed inferred camera, one human-anchored depth gauge and
one fixed object mesh. No tracking, oracle pose, hidden labels or GT files.
All large intermediates and model inputs remain on the remote managed disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import errno
import shutil
import stat
from pathlib import Path
import platform
import sys
import time

from body_smoke import EPISODE, TRACK1_EPISODE_COUNT, _validate_inputs, _pinned_checkout, UPSTREAM_REVISION
from world_reward.data import sha256
from world_reward.mesh_geometry import normalize_degenerate_faces


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    parser.add_argument("--mesh-source", choices=("default", "solid"), default="default")
    return parser


def _solid_compact(vertices, faces, np):
    """Remove only official zero padding, never weld or repair a surface."""
    from exact_mesh_geometry import _exact_faces, exact_mesh_topology
    if (vertices.dtype != np.float64 or faces.dtype != np.int64
            or vertices.shape != (4096, 3) or faces.shape != (4096, 3)
            or not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= 4096)):
        raise ValueError("Solid poses must retain exact F64/I64 official 4096 geometry")
    active = np.flatnonzero(~np.all(faces == 0, axis=1))
    ids, inverse = np.unique(faces[active], return_inverse=True)
    unused = np.setdiff1d(np.arange(len(vertices)), ids)
    if len(unused) and not np.array_equal(vertices[unused], np.repeat(vertices[:1], len(unused), axis=0)):
        raise ValueError("Only original repeated-first-vertex padding is permitted")
    compact = vertices[ids].copy(), inverse.reshape(-1, 3).astype(np.int64)
    _exact_faces(*compact)
    topology = exact_mesh_topology(*compact)
    stored = compact[0].astype(np.float32).astype(np.float64)
    if not np.isfinite(stored).all():
        raise ValueError("Solid metric positions cannot be represented in GLB float32")
    _exact_faces(stored, compact[1])
    _solid_float32_orientation(compact[0], stored, compact[1])
    _solid_topology_equal(topology, exact_mesh_topology(stored, compact[1]))
    if np.max(np.linalg.norm(stored - compact[0], axis=1)) > 1e-5:
        raise ValueError("Solid GLB quantization exceeds inherited 1e-5 metric roundtrip bound")
    return active, compact, topology


def _solid_float32_orientation(original, stored, faces):
    """Exact dyadic local normal dot; no area tolerance or sign repair."""
    from fractions import Fraction
    def normal(triangle):
        a, b, c = [[Fraction(float(x)) for x in row] for row in triangle]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    for face in faces:
        if sum(a*b for a, b in zip(normal(original[face]), normal(stored[face]))) <= 0:
            raise ValueError('Float32 storage collapsed or reversed a meaningful local triangle')


def _solid_topology_equal(before, after):
    signature = lambda t: (t['vertices'], t['active_vertices'], t['faces'],
        sorted((c['euler'], c['volume_sign'], c['vertices'], c['faces']) for c in t['components']))
    if signature(before) != signature(after):
        raise ValueError("Solid serialization changed whole-shell Euler/orientation/counts")


def _solid_preflight(root, episode, inputs, report, pose_path, np):
    """Independent frozen proposal and full trajectory, before reserving output."""
    from solid_geometry_loader import load, identity, strict_json, SOURCE_HELPERS
    code = Path(__file__).resolve().parent.parent
    base = root / f'outputs/episode_{episode:06d}'
    pin_path = code / 'configs' / f'solid_mesh_{episode:06d}_pins.json'
    pin = identity(pin_path)
    pins = strict_json(pin_path.read_text())
    object_path, alignment_path = base / 'object_grounded/report.json', base / 'scale_smoke/report.json'
    transform_path = base / 'object_grounded/transform.json'
    transform = strict_json(transform_path.read_text())
    scale = np.asarray(transform['scale'], np.float64)
    if (scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.allclose(scale, scale[0], atol=0, rtol=1e-5)):
        raise ValueError("Original positive grounding scale is fixed; never average or rebake")
    values = load(root, episode, inputs['video_sha256'], sha256(object_path), sha256(alignment_path),
                  float(scale[0]), pins=pins)
    expected_v, expected_f, expected_active, _, canonical, receipt = values
    pose_canonical = pose_path.parent / 'object_fixed_canonical.glb'
    pose_report_path = pose_path.parent / 'report.json'
    parent = pose_path.parent.lstat()
    if (pose_path.parent.resolve() != pose_path.parent or not stat.S_ISDIR(parent.st_mode)
            or stat.S_IMODE(parent.st_mode) != 0o555
            or {p.name for p in pose_path.parent.iterdir()} != {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'}
            or any(stat.S_IMODE(p.lstat().st_mode) != 0o444 for p in (pose_path, pose_canonical, pose_report_path))
            or strict_json(pose_report_path.read_text()) != report):
        raise ValueError('Complete immutable solid pose output required; unsealed/native-only PASS is insufficient')
    ledger = {p: identity(p, readonly=p in (pin_path, pose_path, pose_canonical, canonical)) for p in
              (pin_path, pose_path, pose_canonical, canonical, object_path, alignment_path,
               transform_path, pose_report_path)}
    if (ledger[pin_path] != pin or report.get('mesh_source') != 'solid'
            or report.get('execution_verified') is not True or report.get('original_frame_coverage_verified') is not True
            or report.get('fixed_shape') is not True or report.get('object_report_sha256') != ledger[object_path]['sha256']
            or report.get('geometry_and_poses_sha256') != ledger[pose_path]['sha256']
            or report.get('fixed_canonical_mesh_sha256') != ledger[canonical]['sha256']
            or ledger[pose_canonical] != ledger[canonical]
            or report.get('topology_budget', {}).get('committed_pins_sha256') != pin['sha256']
            or any(report['topology_budget'].get(k) != receipt[k] for k in
                   ('cpu_report_sha256', 'native_report_sha256', 'cpu_producer_revision', 'cpu_script_sha256'))):
        raise ValueError("Solid pose report does not match the independently pinned canonical proposal")
    with np.load(pose_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'vertices', 'faces', 'frame_index', 'rotation', 'translation', 'object_scale'}:
            raise ValueError("Exact six-array solid full-trajectory payload required")
        v, f, r, t = (arrays[k].copy() for k in ('vertices', 'faces', 'rotation', 'translation'))
        frames, scalar = arrays['frame_index'], arrays['object_scale']
        if (frames.dtype != np.int64 or frames.shape != (inputs['total_frames'],)
                or not np.array_equal(frames, np.arange(inputs['total_frames']))
                or scalar.dtype != np.float64 or scalar.shape != () or scalar.item() != 1.
                or r.dtype != np.float64 or t.dtype != np.float64
                or not np.array_equal(v, expected_v) or not np.array_equal(f, expected_f)):
            raise ValueError("Solid geometry/gauge/original frame IDs differ from frozen CPU inputs")
    active, compact, topology = _solid_compact(v, f, np)
    if not np.array_equal(active, expected_active):
        raise ValueError("Meaningful solid triangles differ from independently verified padding")
    helpers = {*SOURCE_HELPERS, 'infra/cari_prepare.py', 'infra/solid_geometry_loader.py',
               'infra/mesh_precision_diagnostic.py', 'src/world_reward/mesh_geometry.py',
               'infra/cari_wrapper_common.sh', 'infra/run_cari_prepare.sh'}
    ledger.update({code / name: identity(code / name) for name in helpers})
    _solid_recheck(ledger)
    return v, f, active, r, t, compact, topology, ledger


def _solid_recheck(ledger):
    from solid_geometry_loader import identity
    code = Path(__file__).resolve().parent.parent
    if any(identity(path, readonly=path.is_relative_to(code) or path.suffix in ('.npz', '.glb')
                    or path.parent.name == 'object_pose_full_solid') != value for path, value in ledger.items()):
        raise ValueError("Frozen solid geometry, poses, pins or pure source changed")
    pose_parents = {p.parent for p in ledger if p.parent.name == 'object_pose_full_solid'}
    if any(stat.S_IMODE(p.lstat().st_mode) != 0o555 or {x.name for x in p.iterdir()} !=
           {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'} for p in pose_parents):
        raise ValueError('Immutable solid pose namespace changed after validation')


def _solid_serialized_mesh(path, source, topology, trimesh, np, transform=None):
    """Check actual GLB positions and scene transforms; no guessed baking step."""
    from mesh_precision_diagnostic import raw_glb, triangle_hash
    from object_budget_endpoint import _load_mesh
    from exact_mesh_geometry import exact_mesh_topology
    local, world, records = raw_glb(path)
    stored = source[0].astype(np.float32).astype(np.float64)
    local_triangles = np.concatenate([v[f] for v, f in local])
    if triangle_hash(local_triangles) != triangle_hash(stored[source[1]]):
        raise ValueError("GLB POSITION/indices altered meaningful oriented triangles")
    if transform is None:
        if not all(row['node_transform_identity'] for row in records):
            raise ValueError("Metric GLB must not introduce a hidden scene transform")
        expected = stored
    else:
        # Native _write_object_template applies its actual float32 matrix to
        # the scene graph, then exports. POSITION need not be baked/recast.
        expected = trimesh.transform_points(stored, transform)
    expected_hash = triangle_hash(expected[source[1]])
    loaded_v, loaded_f = _load_mesh(path)
    if triangle_hash(world) != expected_hash or triangle_hash(loaded_v[loaded_f]) != expected_hash:
        raise ValueError("Native aligned GLB changed represented rigid-transform geometry")
    _solid_topology_equal(topology, exact_mesh_topology(loaded_v, loaded_f))
    return {'oriented_triangles_sha256': expected_hash, 'position_accessors_float32': True,
            'faces': len(loaded_f), 'components': len(topology['components']),
            'all_meaningful_triangles_preserved': True, 'geometry_repaired': False,
            'independent_embedding_reverified': False}


def _solid_camera_roundtrip(source, rotations, translations, aligned_poses, transform, trimesh, np):
    """Actual GLB positions plus the exact F32 poses written to native input."""
    original = source[0][source[1].reshape(-1)]
    stored = source[0].astype(np.float32).astype(np.float64)
    aligned = trimesh.transform_points(stored, transform)[source[1].reshape(-1)]
    saved_poses = aligned_poses.astype(np.float32).astype(np.float64)
    error = 0.
    for index in range(len(rotations)):
        before = original @ rotations[index].T + translations[index]
        after = aligned @ saved_poses[index, :3, :3].T + saved_poses[index, :3, 3]
        error = max(error, float(np.max(np.linalg.norm(after - before, axis=1))))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError('Represented GLB/F32 object poses exceed inherited 1e-5 camera roundtrip bound')
    return error


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux container with network none")
    args = _argument_parser().parse_args()
    root = Path(os.environ["WR_ROOT"])
    inputs = _validate_inputs(root, episode_index=args.episode)
    import cv2
    import h5py
    import joblib
    import numpy as np
    from PIL import Image
    import trimesh
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    native_root = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    sys.path.insert(0, str(native_root))
    from prep.prepare_mhr_wild_export import prepare_mhr_wild_export
    from prep.mhr_depth_h5 import MHRDepthH5Writer, validate_depth_h5, read_metric_depth
    from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
    from prep.mhr_export_utils import MHR_CAMERA_NAMES, frame_names, read_rgb, read_mask, camera_calibration, load_edex
    base = root / f"outputs/episode_{args.episode:06d}"
    output = base / "cari_inputs"
    if output.exists():
        raise RuntimeError("Frozen CARI inputs exist; never overwrite")
    reports = {}
    report_paths = {"body": base / "body_full/report.json", "depth": base / "depth_full/report.json",
                    "object": base / "object_pose_full/report.json", "alignment": base / "scale_smoke/report.json",
                    "adapter": base / "body_full/cari_adapter/report.json"}
    if args.mesh_source == 'solid':
        report_paths['object'] = base / 'object_pose_full_solid/report.json'
    for key, path in report_paths.items():
        record = json.loads(path.read_text())
        if type(record.get("episode_index", args.episode)) is not int or record.get("episode_index", args.episode) != args.episode:
            raise RuntimeError(f"{key} report belongs to another episode")
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
    if args.mesh_source == 'solid':
        pose_path = base / 'object_pose_full_solid/geometry_and_poses.npz'
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
    if args.mesh_source == 'solid':
        vertices, faces, active, rotations, translations, solid_compact, solid_topology, solid_ledger = _solid_preflight(
            root, args.episode, inputs, reports['object'], pose_path, np)
        from solid_geometry_loader import identity as solid_identity
        native_pins = {'prep/prepare_mhr_wild_export.py': {'bytes': 13145, 'sha256': 'b465516cc96a8c5472aec995cff12e32a9d033c7c5157a6a601b96e332e45f4f'},
            'prep/mhr_export_utils.py': {'bytes': 25383, 'sha256': 'a9f499dad2f73eb7b8c526f33c94a785cc9468423760afcf6e5ced46d2f49e3b'}}
        for name, pin in native_pins.items():
            path = native_root / name
            if solid_identity(path, readonly=False) != pin:
                raise ValueError('Native unprocessed scene alignment/export source differs from pinned upstream')
            solid_ledger[path] = pin
    else:
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
    if args.mesh_source == 'solid':
        for path in (*report_paths.values(), adapter_path):
            solid_ledger[path] = solid_identity(path, readonly=False)
        if any(sha256(report_paths[key]) != solid_ledger[report_paths[key]]['sha256'] or
               reports[key] != json.loads(report_paths[key].read_text()) for key in reports):
            raise ValueError('Full initializer reports changed before preparing solid inputs')
        metric_mesh = trimesh.Trimesh(*solid_compact, process=False)
    else:
        active, _ = normalize_degenerate_faces(vertices, faces)
        metric_mesh = trimesh.Trimesh(vertices, faces[active], process=True)
    if not metric_mesh.is_watertight or not metric_mesh.is_winding_consistent or metric_mesh.volume <= 0:
        raise RuntimeError("Packed fixed geometry must remain closed and correctly oriented")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    sequence = f"episode_{args.episode:06d}"
    video_link = output / (sequence + ".0.color.mp4")
    # Native prep resolves symlinks before validating its .0.color.mp4 ABI.
    # Same managed disk: a hardlink preserves bytes without duplicating video.
    try:
        os.link(inputs["video"], video_link)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # Docker bind mounts are separate mountpoints even on one disk.
        # Copy only within Azure, never across the laptop/tethered connection.
        shutil.copyfile(inputs["video"], video_link)
    if sha256(video_link) != inputs["video_sha256"]:
        raise RuntimeError("Native video alias differs from original Track 1 bytes")
    metric_path = output / "object_metric.glb"
    metric_mesh.export(metric_path)
    if args.mesh_source == 'solid':
        metric_proof = _solid_serialized_mesh(metric_path, solid_compact, solid_topology, trimesh, np)
        metric_identity = solid_identity(metric_path, readonly=False)
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
    if args.mesh_source == 'solid':
        if not np.array_equal(A.astype(np.float32).astype(np.float64), A) or not np.array_equal(A[3], [0., 0., 0., 1.]):
            raise ValueError('Native aligned transform must preserve its actual float32 affine metadata')
        aligned_path = output / 'export' / sequence / 'object_mesh/output_aligned.glb'
        if metadata.get('object_mesh_file') != str(aligned_path) or export_seq != aligned_path.parent.parent:
            raise ValueError('Native solid export must retain the canonical prepared aligned-GLB route')
        aligned_proof = _solid_serialized_mesh(aligned_path, solid_compact, solid_topology, trimesh, np, A)
        aligned_proof['represented_mesh_pose_frame_roundtrip_max_error_m'] = _solid_camera_roundtrip(
            solid_compact, rotations, translations, aligned_poses, A, trimesh, np)
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
    result = {"stage": "world_reward_native_cari_inputs", "status": "pass", "episode_index": args.episode, "frames": count,
              "export_seq": str(export_seq), "depth_h5": str(aligned_depth_path), "mhr_init": str(adapter_path),
              "object_poses": str(object_poses_path), "object_pose_initializer": "own_ICP_Viterbi_not_FoundationPose",
              "depth_validation": depth_validation, "mesh_pose_frame_roundtrip_max_error_m": frame_transform_error,
              "jpeg_original_RGB_mean_absolute_error": float(np.mean(jpeg_errors)), "original_frame_coverage_verified": True,
              "input_track": "track_1", "input_sha256": inputs["video_sha256"],
              "input_dataset_revision": inputs["dataset_revision"], "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "submission_eligible": False,
              "challenge_performance_verified": False, "producer_revision": os.environ.get("WR_CODE_REVISION"),
              "input_report_sha256": {key: sha256(path) for key, path in report_paths.items()},
              "file_sha256": {key: sha256(path) for key, path in {"depth_h5": aligned_depth_path, "mhr_init": adapter_path,
                                                               "object_poses": object_poses_path, "wild_export": metadata_path}.items()},
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    if args.mesh_source == 'solid':
        _solid_recheck(solid_ledger)
        _pinned_checkout(vendor, UPSTREAM_REVISION)
        if (solid_identity(metric_path, readonly=False) != metric_identity
                or metadata['source_object_mesh'] != {'path': str(metric_path.resolve()),
                    'size': metric_path.stat().st_size, 'mtime_ns': metric_path.stat().st_mtime_ns}):
            raise ValueError('Native solid metric GLB source changed during preparation')
        _solid_serialized_mesh(metric_path, solid_compact, solid_topology, trimesh, np)
        _solid_serialized_mesh(aligned_path, solid_compact, solid_topology, trimesh, np, A)
        result['object_source'] = 'solid'
        result['object_pose_source'] = {'report': str(report_paths['object'].relative_to(root)),
            'geometry_and_poses': str(pose_path.relative_to(root)), 'geometry_and_poses_sha256': sha256(pose_path)}
        result['solid_geometry_validation'] = {'metric_glb': metric_proof, 'native_aligned_glb': aligned_proof,
            'source_rehashed_after': True, 'files': {str(p): value for p, value in solid_ledger.items()},
            'geometry_repaired': False, 'metric_scale_applied_again': False}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    # Transient combined masks are redundant after verified native export.
    masks_path.unlink()
    print(json.dumps({"stage": result["stage"], "status": "pass", "frames": count,
                      "mesh_pose_error_m": frame_transform_error, "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

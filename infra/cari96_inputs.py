"""Hash-bound ep15 frames 0..95 export; no initializer edit, inference or GT.

``prepare_snapshot(root, out, receiptpins)`` requires an absent output under
``root/validation`` and explicit source identities. Joblib/HDF5 imports are
lazy; the trusted pinned initializer is returned unchanged for the caller's
separate shared-identity decode. Existing sources are never chmod'ed or copied
from caches, forward predictions, refined outputs, videos or model assets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import stat
from pathlib import Path

SOURCE_FRAMES, FRAMES = 501, 96
IMAGE_SIZE = (1152, 1536)
BASE = "outputs/episode_000015"
EXPORT = BASE + "/cari_inputs/export/episode_000015"
REPORT = BASE + "/cari_inputs/report.json"
INITIALIZER = BASE + "/body_full/cari_adapter/canonical_initializer.pkl"
DEPTH = BASE + "/cari_inputs/aligned_depth.h5"
POSES = BASE + "/cari_inputs/own_object_poses.pkl"
DEPENDENCIES = {
    "body": (BASE + "/body_full/report.json", "sam3d_body_full_video_initializer"),
    "depth": (BASE + "/depth_full/report.json", "monocular_moge2_full_video"),
    "object": (BASE + "/object_pose_full/report.json", "fixed_scale_full_object_pose_initializer"),
    "alignment": (BASE + "/scale_smoke/report.json", "predicted_human_anchored_moge2_pointmaps"),
    "adapter": (BASE + "/body_full/cari_adapter/report.json", "native_cari_body_adapter_full_video"),
}
PARAMETER_DIMS = {"mhr_global_rot6d": 6, "mhr_trans": 3, "mhr_body_pose_cont": 260,
                  "mhr_hand": 108, "mhr_shape": 45, "mhr_scale": 28, "mhr_face": 72}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path).absolute()
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Canonical regular source required; no symlinks")
    return path


def identity(path):
    path = regular(path)
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def source_paths(camera):
    if not isinstance(camera, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera):
        raise ValueError("Explicit native camera name required")
    return {REPORT, INITIALIZER, DEPTH, POSES, *(p for p, _ in DEPENDENCIES.values()),
            *(EXPORT + f"/{kind}/{camera}.h5" for kind in ("images", "human_masks", "object_masks")),
            *(EXPORT + "/" + name for name in ("edex", "wild_export.json", "object_mesh/output_aligned.glb"))}


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "camera_name", "input_report", "source_files"}
            or pins["schema"] != "world-reward-cari96-input-pins-v1"
            or type(pins["source_files"]) is not dict
            or set(pins["source_files"]) != source_paths(pins["camera_name"])):
        raise ValueError("Exact explicit ep15 input pin inventory required")
    for value in pins["source_files"].values():
        if (type(value) is not dict or set(value) != {"sha256", "bytes"}
                or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                or type(value["bytes"]) is not int or value["bytes"] <= 0):
            raise ValueError("Strict SHA/size pin required")
    expected = pins["input_report"]
    if (type(expected) is not dict or set(expected) != {"producer_revision", "script_sha256", "sha256", "bytes"}
            or {k: expected[k] for k in ("sha256", "bytes")} != pins["source_files"][REPORT]
            or not isinstance(expected["producer_revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", expected["producer_revision"])
            or not isinstance(expected["script_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected["script_sha256"])):
        raise ValueError("Exact historical producer/source required")


def _no_oracle(record):
    if (type(record) is not dict or record.get("ground_truth_used") is not False
            or record.get("hand_labeled_test") is not False or record.get("oracle_modes") != []):
        raise ValueError("Explicit no-GT/no-hand-label/no-oracle provenance required")
    if any(record.get(k, False) is not False for k in ("ground_truth_read", "private_truth_read")):
        raise ValueError("Private/ground-truth access forbidden")


def validate_reports(root, pins):
    root = Path(root)
    read = lambda path: json.loads(regular(root / path).read_text())
    report = read(REPORT)
    _no_oracle(report)
    if (report.get("stage") != "world_reward_native_cari_inputs" or report.get("status") != "pass"
            or report.get("input_track") != "track_1" or type(report.get("frames")) is not int
            or report["frames"] != SOURCE_FRAMES or report.get("original_frame_coverage_verified") is not True
            or any(report.get(k) != pins["input_report"][k] for k in ("producer_revision", "script_sha256"))
            or not isinstance(report.get("input_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", report["input_sha256"])):
        raise ValueError("Pinned passing full501 native preparation required")
    if "episode_index" in report and (type(report["episode_index"]) is not int or report["episode_index"] != 15):
        raise ValueError("Preparation episode mismatch")
    if report.get("object_pose_initializer") != "own_ICP_Viterbi_not_FoundationPose":
        raise ValueError("Prepared inferred object trajectory required")
    paths = {"export_seq": EXPORT, "depth_h5": DEPTH, "mhr_init": INITIALIZER, "object_poses": POSES}
    if any(report.get(k) != str(root / v) for k, v in paths.items()):
        raise ValueError("Preparation canonical paths differ")
    hashes = {"depth_h5": DEPTH, "mhr_init": INITIALIZER, "object_poses": POSES,
              "wild_export": EXPORT + "/wild_export.json"}
    if any(report.get("file_sha256", {}).get(k) != pins["source_files"][v]["sha256"] for k, v in hashes.items()):
        raise ValueError("Preparation artifact chain differs")
    dependencies = {}
    for key, (path, stage) in DEPENDENCIES.items():
        item = read(path); _no_oracle(item)
        if (item.get("status") != "pass" or item.get("stage") != stage or item.get("input_track") != "track_1"
                or report.get("input_report_sha256", {}).get(key) != pins["source_files"][path]["sha256"]):
            raise ValueError("Full initializer dependency chain differs")
        if "episode_index" in item and (type(item["episode_index"]) is not int or item["episode_index"] != 15):
            raise ValueError("Dependency episode mismatch")
        if key in {"body", "depth", "object"}:
            indices = [row.get("frame_index") for row in item.get("frames", []) if type(row) is dict]
            if (indices != list(range(SOURCE_FRAMES)) or any(type(i) is not int for i in indices)
                    or item.get("input_sha256") != report["input_sha256"]):
                raise ValueError("Exact ordered501 original frame identities required")
        elif key == "adapter":
            if (type(item.get("frames")) is not int or item["frames"] != SOURCE_FRAMES
                    or item.get("canonical_initializer_sha256") != pins["source_files"][INITIALIZER]["sha256"]
                    or item.get("body_report_sha256") != pins["source_files"][DEPENDENCIES["body"][0]]["sha256"]):
                raise ValueError("Initializer/Body binding differs")
        dependencies[key] = item
    validation = report.get("depth_validation", {})
    if (validation.get("validation_mode") != "exhaustive"
            or validation.get("frame_counts") != {pins["camera_name"]: SOURCE_FRAMES}
            or validation.get("frame_shapes") != {pins["camera_name"]: list(IMAGE_SIZE)}):
        raise ValueError("Original exhaustive full501 depth validation required")
    return report, dependencies


def _float_array(value, shape):
    import numpy as np
    if np.ma.isMaskedArray(value): raise ValueError("Masked arrays forbidden")
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError("Exact finite native float32 shape required")
    return a


def validate_initializer(initializer):
    import numpy as np
    if (type(initializer) is not dict or initializer.get("body_model") != "mhr"
            or initializer.get("frames") != [f"{i:06d}" for i in range(SOURCE_FRAMES)]
            or initializer.get("kids") != [0]):
        raise ValueError("Full501 canonical initializer required")
    _no_oracle(initializer.get("metadata"))
    if initializer["metadata"].get("mhr_geometry_forward_verified") is not True:
        raise ValueError("Original native initializer parity required")
    for key, dim in PARAMETER_DIMS.items(): _float_array(initializer.get(key), (SOURCE_FRAMES, dim))
    if np.count_nonzero(initializer["mhr_face"]) or np.any(initializer["mhr_trans"][:, 2] <= 0):
        raise ValueError("Zero expression/positive camera translation required")
    for key in ("mhr_joints", "mhr_keypoints"):
        if np.ma.isMaskedArray(initializer.get(key)): raise ValueError("Masked geometry forbidden")
        a = np.asarray(initializer.get(key))
        if a.ndim != 3 or a.shape[0] != SOURCE_FRAMES or a.shape[1] == 0 or a.shape[2] != 3:
            raise ValueError("Full initializer geometry required")
        _float_array(a, a.shape)


def validate_poses(poses, wild):
    import numpy as np
    if (type(poses) is not dict or set(poses) != {"frames", "obj_pose_world", "metadata"}
            or poses["frames"] != [f"{i:06d}" for i in range(SOURCE_FRAMES)]):
        raise ValueError("Exact full501 automatic object pose schema required")
    _no_oracle(poses["metadata"])
    if poses["metadata"].get("source") != "World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose":
        raise ValueError("Predicted ICP/Viterbi poses required")
    a = _float_array(poses["obj_pose_world"], (SOURCE_FRAMES, 4, 4))
    R = a[:, :3, :3]
    if (not np.array_equal(a[:, 3], np.broadcast_to(np.array([0, 0, 0, 1], np.float32), (SOURCE_FRAMES, 4)))
            or not np.allclose(R @ R.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(R), 1, atol=1e-5, rtol=0)):
        raise ValueError("Proper rigid poses required")
    A = np.asarray(wild.get("source_object_mesh_to_aligned_transform"), dtype=np.float64)
    if (A.shape != (4, 4) or not np.isfinite(A).all() or not np.array_equal(A[3], [0, 0, 0, 1])
            or not np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(A[:3, :3]), 1, atol=1e-5, rtol=0)
            or not np.array_equal(A, poses["metadata"].get("mesh_frame_change"))):
        raise ValueError("Same proper rigid mesh/pose frame change required; no scale fit")


def validate_wild(wild, root):
    import numpy as np
    if (type(wild) is not dict or wild.get("schema") != "cari4d.mhr_wild_export.v2"
            or type(wild.get("frame_count")) is not int or wild["frame_count"] != SOURCE_FRAMES
            or wild.get("sequence") != "episode_000015" or type(wild.get("camera_id")) is not int
            or wild["camera_id"] != 0 or (wild.get("height"), wild.get("width")) != IMAGE_SIZE
            or wild.get("object_mesh_file") != str(Path(root) / EXPORT / "object_mesh/output_aligned.glb")
            or wild.get("depth_backend") != "moge2"
            or wild.get("object_pose_frame") != "centered_axis_aligned"
            or wild.get("object_pose_frame_revision") != "cari4d.object_pose_frame.centered_axis_aligned.v1"
            or wild.get("object_pose_storage_frame") != "output_aligned_mesh_frame"):
        raise ValueError("Original501 one-camera native predicted export required")
    for key in ("object_pose_storage_to_training_transform", "object_mesh_to_training_transform"):
        if not np.array_equal(wild.get(key), np.eye(4)): raise ValueError("Aligned object frame must remain identity")
    h, w = IMAGE_SIZE
    K = np.array([[np.hypot(h, w), 0, w / 2], [0, np.hypot(h, w), h / 2], [0, 0, 1]])
    if not np.array_equal(wild.get("intrinsics"), K): raise ValueError("Frozen inferred RGB-size camera differs")
    edex = json.loads((Path(root) / EXPORT / "edex").read_text())
    camera = edex[0]["cameras"][0]
    if (len(edex) != 1 or len(edex[0]["cameras"]) != 1
            or camera.get("intrinsics") != {"focal": [K[0, 0], K[1, 1]], "principal": [K[0, 2], K[1, 2]]}
            or not np.array_equal(camera.get("transform"), np.eye(4)[:3])):
        raise ValueError("Same inferred K/world-camera identity required")


def _attrs_equal(left, right):
    import numpy as np
    return set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)


def _h5_nodes(handle):
    import h5py
    nodes = {}
    def visit(group, prefix=""):
        for name in group:
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink): raise ValueError("HDF5 external/soft links forbidden")
            node = group[name]; path = prefix + name
            address = h5py.h5o.get_info(node.id).addr
            if any(address == old for old, _ in nodes.values()): raise ValueError("HDF5 alias/cycle forbidden")
            nodes[path] = (address, node)
            if isinstance(node, h5py.Group): visit(node, path + "/")
            elif node.is_virtual or node.external: raise ValueError("External/virtual HDF5 storage forbidden")
    visit(handle)
    return {k: node for k, (_, node) in nodes.items()}


def subset_h5(source, target, *, kind, camera, image_size, shared_scale=None):
    """Copy native stored payloads, not decoded/re-encoded images or depths."""
    import h5py
    import numpy as np
    with h5py.File(source, "r") as src:
        nodes = _h5_nodes(src)
        if not isinstance(src.attrs.get("complete"), (bool, np.bool_)) or not bool(src.attrs["complete"]):
            raise ValueError("Complete source HDF5 required")
        if kind == "depth":
            groups = {"frame_names", "raw", "aligned", "alignment", f"alignment/{camera}"}
            datasets = {f"{g}/{camera}" for g in ("frame_names", "raw", "aligned")}
            datasets |= {f"alignment/{camera}/{k}" for k in ("scale", "shift", "valid_count")}
            if (src.attrs.get("format") != "cari4d_mhr_metric_depth_png_v1"
                    or src.attrs.get("depth_alignment_method") != "world_reward_shared_predicted_human_scale"):
                raise ValueError("Pinned native metric depth format/alignment required")
            depth_identity = json.loads(src.attrs["depth_alignment_input_identity_json"])
            if (depth_identity.get("ground_truth_used") is not False or depth_identity.get("depth_backend") != "moge2"
                    or depth_identity.get("alignment") != "one_predicted_human_anchored_clip_scalar_no_offset"):
                raise ValueError("Native predicted common depth identity required")
            if depth_identity.get("depth_model_revision") != "b135031bae30b5ac2ae141a0e68717795ce38340":
                raise ValueError("Original pinned MoGe depth model required")
            names = nodes[f"frame_names/{camera}"][:]
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
            if names != [f"{i:06d}" for i in range(SOURCE_FRAMES)]: raise ValueError("Depth frame names differ")
            scale, shift, counts = (nodes[f"alignment/{camera}/{k}"][:] for k in ("scale", "shift", "valid_count"))
            if (scale.dtype != np.float32 or shift.dtype != np.float32 or counts.dtype != np.int32
                    or not np.array_equal(scale, np.full(SOURCE_FRAMES, shared_scale, np.float32))
                    or np.any(shift != 0) or np.any(counts < 0) or np.any(counts > image_size[0] * image_size[1])):
                raise ValueError("Stored shared scale/zero shift/valid counts differ")
        elif kind in {"images", "human_masks", "object_masks"}:
            groups, datasets = set(), {"frames"}
            if src.attrs.get("sequence") != "episode_000015" or src.attrs.get("camera_id") != 0:
                raise ValueError("Source export episode/camera differs")
        else: raise ValueError("Unsupported native subset kind")
        if set(nodes) != groups | datasets: raise ValueError("Unexpected native HDF5 inventory")
        for name in groups:
            if not isinstance(nodes[name], h5py.Group): raise ValueError("Native group schema differs")
        for name in datasets:
            ds = nodes[name]
            expected_shape = (SOURCE_FRAMES, *image_size) if kind.endswith("masks") else (SOURCE_FRAMES,)
            if not isinstance(ds, h5py.Dataset) or ds.shape != expected_shape: raise ValueError("Full501 dataset shape differs")
            if kind.endswith("masks"):
                if ds.dtype != np.uint8: raise ValueError("Native binary mask dtype differs")
            elif kind == "images" or name.startswith(("raw/", "aligned/")):
                if h5py.check_dtype(vlen=ds.dtype) != np.dtype("uint8"): raise ValueError("Native uint8 payload schema differs")
            elif name.startswith("frame_names/") and h5py.check_string_dtype(ds.dtype) is None:
                raise ValueError("Native frame-name string dtype differs")
        with h5py.File(target, "x") as dst:
            for key, value in src.attrs.items(): dst.attrs[key] = value
            for name in sorted(groups, key=lambda x: (x.count("/"), x)):
                group = dst.create_group(name)
                for key, value in nodes[name].attrs.items(): group.attrs[key] = value
            for name in sorted(datasets):
                ds = nodes[name]
                options = {}
                if ds.chunks: options["chunks"] = (min(ds.chunks[0], FRAMES), *ds.chunks[1:])
                if ds.compression: options.update(compression=ds.compression, compression_opts=ds.compression_opts)
                if ds.shuffle: options["shuffle"] = True
                if ds.fletcher32: options["fletcher32"] = True
                copied = dst.create_dataset(name, shape=(FRAMES, *ds.shape[1:]), dtype=ds.dtype, **options)
                for key, value in ds.attrs.items(): copied.attrs[key] = value
                for i in range(FRAMES):
                    row = ds[i]
                    if kind.endswith("masks") and not np.isin(row, [0, 255]).all(): raise ValueError("Binary automatic mask required")
                    if kind == "images" and (len(row) < 4 or row.tobytes()[:2] != b"\xff\xd8" or row.tobytes()[-2:] != b"\xff\xd9"):
                        raise ValueError("Native encoded JPEG payload required")
                    if name.startswith(("raw/", "aligned/")):
                        b = row.tobytes()
                        if (len(b) < 33 or b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR"
                                or b[24:26] != b"\x10\x00"
                                or (int.from_bytes(b[20:24], "big"), int.from_bytes(b[16:20], "big")) != image_size):
                            raise ValueError("Native uint16 PNG full-grid depth payload required")
                    copied[i] = row
            dst.flush()
        with h5py.File(target, "r") as dst:
            if not _attrs_equal(src.attrs, dst.attrs): raise ValueError("Root attributes changed")
            for name in groups | datasets:
                if not _attrs_equal(nodes[name].attrs, dst[name].attrs): raise ValueError("Dataset/group attributes changed")
            for name in datasets:
                if dst[name].dtype != nodes[name].dtype: raise ValueError("Stored dtype changed")
                for i in range(FRAMES):
                    if not np.array_equal(nodes[name][i], dst[name][i]): raise ValueError("Reread payload/metadata changed")


def prepare_snapshot(root, out, receiptpins):
    """Return new native paths + unchanged full initializer identity; not runner-ready."""
    import numpy as np
    import joblib
    root, out = Path(root).absolute(), Path(out).absolute()
    validate_pins(receiptpins)
    pins = copy.deepcopy(receiptpins)
    if (root.resolve() != root or out.resolve() != out or out == root / "validation"
            or not out.is_relative_to(root / "validation")):
        raise ValueError("Canonical validation output under the managed root required")
    if out.exists() or out.is_symlink(): raise FileExistsError("Never overwrite complete or incomplete outputs")
    frozen = {p: identity(root / p) for p in source_paths(pins["camera_name"])}
    if frozen != pins["source_files"]: raise ValueError("Pinned original source SHA/size differs")
    prep, deps = validate_reports(root, pins)
    wild = json.loads((root / EXPORT / "wild_export.json").read_text())
    validate_wild(wild, root)
    # Deserialization occurs only after exact externally supplied report/file hashes.
    initializer = joblib.load(root / INITIALIZER); validate_initializer(initializer)
    poses = joblib.load(root / POSES); validate_poses(poses, wild)
    if poses["metadata"].get("source_pose_sha256") != deps["object"].get("geometry_and_poses_sha256"):
        raise ValueError("Automatic source pose report chain differs")
    scale = deps["alignment"].get("depth_alignment", {}).get("shared_scale")
    if type(scale) not in {int, float} or not np.isfinite(scale) or scale <= 0: raise ValueError("Pinned positive predicted shared scale required")
    out.mkdir(exist_ok=False)
    export = out / "export/episode_000015"
    export.mkdir(parents=True)
    for kind in ("images", "human_masks", "object_masks"):
        (export / kind).mkdir()
        subset_h5(root / EXPORT / kind / (pins["camera_name"] + ".h5"), export / kind / (pins["camera_name"] + ".h5"),
                  kind=kind, camera=pins["camera_name"], image_size=IMAGE_SIZE)
    subset_h5(root / DEPTH, out / "aligned_depth.h5", kind="depth", camera=pins["camera_name"],
              image_size=IMAGE_SIZE, shared_scale=scale)
    for name in ("edex", "object_mesh/output_aligned.glb"):
        target = export / name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / EXPORT / name, target)
        if identity(target) != frozen[EXPORT + "/" + name]: raise ValueError("Fixed camera/mesh bytes changed")
    provenance = {"source_episode_index": 15, "source_frames": SOURCE_FRAMES, "original_frame_indices": list(range(FRAMES)),
                  "source_inputs_report_sha256": frozen[REPORT]["sha256"], "no_padding_or_reencoding": True}
    selected_poses = copy.deepcopy(poses)
    selected_poses["frames"] = poses["frames"][:FRAMES]
    selected_poses["obj_pose_world"] = poses["obj_pose_world"][:FRAMES].copy()
    selected_poses["metadata"]["world_reward_subset"] = provenance
    pose_output = out / "own_object_poses.pkl"
    joblib.dump(selected_poses, pose_output, compress=0, protocol=4)
    reread = joblib.load(pose_output)
    if (reread["frames"] != selected_poses["frames"] or reread["metadata"] != selected_poses["metadata"]
            or reread["obj_pose_world"].dtype != poses["obj_pose_world"].dtype
            or reread["obj_pose_world"].tobytes() != poses["obj_pose_world"][:FRAMES].tobytes()):
        raise ValueError("Stored selected poses/frame change differ")
    selected_wild = copy.deepcopy(wild)
    selected_wild.update(frame_count=FRAMES, object_mesh_file=str(export / "object_mesh/output_aligned.glb"),
                         world_reward_subset=provenance)
    (export / "wild_export.json").write_text(json.dumps(selected_wild, indent=2, allow_nan=False) + "\n")
    if {p: identity(root / p) for p in frozen} != frozen: raise ValueError("Original source changed during export")
    files = {str(p.relative_to(out)): identity(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"stage": "world_reward_public_cari96_inputs_snapshot", "status": "pass", "frames": FRAMES,
        **provenance, "input_track": "track_1", "input_sha256": prep["input_sha256"], "ground_truth_used": False,
        "ground_truth_read": False, "hand_labeled_test": False, "oracle_modes": [], "initializer_modified": False,
        "independent_validation": False, "submission_eligible": False, "camera_name": pins["camera_name"],
        "source_files": frozen, "output_files": files, "initializer_source": str(root / INITIALIZER),
        "historical_preparation_missing_fields": [k for k in ("episode_index", "input_dataset_revision") if k not in prep],
        "stored_payload_and_attribute_reread_verified": True, "depth_payload_decode_reverified": False,
        "script_sha256": sha256(Path(__file__))}
    manifest_path = out / "manifest.json"
    with manifest_path.open("x") as stream: json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
    for path in out.rglob("*"):
        if path.is_file(): path.chmod(0o444)
    return {"export_seq": export, "depth_h5": out / "aligned_depth.h5", "object_poses": pose_output,
            "initializer_source": root / INITIALIZER, "initializer_report": root / DEPENDENCIES["adapter"][0],
            "initializer": initializer, "snapshot_manifest": manifest_path, "manifest": manifest}

"""Hash-bound full original-timeline public CARI input audit, not inference.

Only the exact fifteen prepared public-input files are read. All supplied file
hashes are verified before JSON interpretation or trusted Joblib deserialization.
No videos, forward/refined predictions, caches, GT, model assets or other tracks
are opened. HDF5/GLB payloads are hash-checked, not independently decoded: their
original producer's exhaustive validation remains explicitly provenance-bound.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import cari96_inputs as public

identity = public.identity
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
LEGACY_INPUT_REPORT = {
    "sha256": "d502064e3cc4817a10f55e8e3f5e0f531b2f41f87bc8a81158f4eda078d44589",
    "bytes": 3264,
    "producer_revision": "5d4f5db0d115bea084ce696531f8e4f430c7d5e6",
    "script_sha256": "579498b140805824398adc83c62d732246f95e3a200138a7a9af5acff52c4f0f",
}
# Original producer verified against its immutable Git source and all fifteen
# public artifacts. Only this input receipt omitted the dataset declaration;
# dependency identities and every numerical/data validation remain unchanged.
HISTORICAL_INPUT_DATASET_OMISSION = {
    "sha256": "9a9a93481432df5846a9ae94a4bdd32552f02167a30cf7e41d4f5ea629ad2799",
    "bytes": 3283,
    "producer_revision": "baba81be965dff878d7c16c9f132f85dcf712bd7",
    "script_sha256": "5719132ec864cad59b907f47835691f9e53657bd91a372dff84ec61997259b3c",
}
DEPENDENCY_STAGES = {
    "body": "sam3d_body_full_video_initializer",
    "depth": "monocular_moge2_full_video",
    "object": "fixed_scale_full_object_pose_initializer",
    "alignment": "predicted_human_anchored_moge2_pointmaps",
    "adapter": "native_cari_body_adapter_full_video",
}


@dataclass(frozen=True)
class PublicClipSpec:
    episode_index: int
    total_frames: int
    camera_name: str
    height: int
    width: int

    def __post_init__(self):
        if type(self.episode_index) is not int or not 0 <= self.episode_index < 30:
            raise ValueError("Explicit Track1 episode integer in 0..29 required")
        if type(self.total_frames) is not int or self.total_frames < 96:
            raise ValueError("Full original native timeline requires integer N>=96")
        if type(self.camera_name) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.camera_name):
            raise ValueError("Explicit safe native camera name required")
        if any(type(value) is not int or value < 2 for value in (self.height, self.width)):
            raise ValueError("Explicit original integer image dimensions >=2 required")

    @property
    def sequence(self):
        return f"episode_{self.episode_index:06d}"


def _spec(spec):
    if type(spec) is not PublicClipSpec:
        raise ValueError("Explicit frozen PublicClipSpec required")
    return spec


def relative_paths(spec):
    spec = _spec(spec)
    base = "outputs/" + spec.sequence
    export = base + "/cari_inputs/export/" + spec.sequence
    return {
        "export_seq": export, "depth_h5": base + "/cari_inputs/aligned_depth.h5",
        "initializer": base + "/body_full/cari_adapter/canonical_initializer.pkl",
        "object_poses": base + "/cari_inputs/own_object_poses.pkl",
        "mesh": export + "/object_mesh/output_aligned.glb", "wild_export": export + "/wild_export.json",
        "input_report": base + "/cari_inputs/report.json",
    }


def _object_source(value):
    if type(value) is not str or value not in {"default", "solid", "surface"}:
        raise ValueError("Only the fixed default, solid or surface object source is permitted")
    return value


def source_profile(pins):
    """Select only from the exact independent pin schema, before input reads."""
    if type(pins) is not dict:
        raise ValueError("Exact independent public input pins required")
    keys = {"schema", "clip_spec", "input_report", "source_files"}
    if set(pins) == keys and pins.get("schema") == "world-reward-cari-clip-input-pins-v1":
        return "default"
    if (set(pins) == keys | {"object_source"}
            and pins.get("schema") == "world-reward-cari-clip-input-pins-v2"
            and type(pins["object_source"]) is str and pins["object_source"] == "solid"):
        return "solid"
    if (set(pins) == keys | {"object_source"}
            and pins.get("schema") == "world-reward-cari-clip-input-pins-v3"
            and type(pins["object_source"]) is str and pins["object_source"] == "surface"):
        return "surface"
    raise ValueError("Exact v1 legacy, v2 solid-only or v3 surface-only public input pin schema required")


def dependency_paths(spec, *, object_source="default"):
    base = "outputs/" + _spec(spec).sequence
    object_source = _object_source(object_source)
    object_directory = "object_pose_full" if object_source == "default" else "object_pose_full_" + object_source
    return {"body": base + "/body_full/report.json", "depth": base + "/depth_full/report.json",
            "object": base + "/" + object_directory + "/report.json", "alignment": base + "/scale_smoke/report.json",
            "adapter": base + "/body_full/cari_adapter/report.json"}


def source_paths(spec, *, object_source="default"):
    paths = relative_paths(spec)
    return {paths[name] for name in paths if name != "export_seq"} | set(dependency_paths(spec, object_source=object_source).values()) | {
        paths["export_seq"] + "/edex",
        *(paths["export_seq"] + f"/{kind}/{spec.camera_name}.h5" for kind in ("images", "human_masks", "object_masks")),
    }


def _receipt(value, *, producer=False):
    keys = {"sha256", "bytes"} | ({"producer_revision", "script_sha256"} if producer else set())
    if (type(value) is not dict or set(value) != keys or type(value["bytes"]) is not int or value["bytes"] <= 0
            or type(value["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Exact explicit SHA256/positive byte count required")
    if producer and (type(value["producer_revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", value["producer_revision"])
            or type(value["script_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["script_sha256"])):
        raise ValueError("Exact actual input producer/source pin required")


def validate_pins(spec, pins):
    _spec(spec)
    object_source = source_profile(pins)
    if type(pins["clip_spec"]) is not dict or set(pins["clip_spec"]) != set(asdict(spec)):
        raise ValueError("Complete explicit generic public clip pins required")
    parsed = PublicClipSpec(**pins["clip_spec"])
    if parsed != spec:
        raise ValueError("Pinned clip spec differs from the explicitly requested clip")
    if type(pins["source_files"]) is not dict or set(pins["source_files"]) != source_paths(spec, object_source=object_source):
        raise ValueError("Exactly fifteen public input files required; no other predictions or assets")
    for row in pins["source_files"].values():
        _receipt(row)
    _receipt(pins["input_report"], producer=True)
    if {key: pins["input_report"][key] for key in ("sha256", "bytes")} != pins["source_files"][relative_paths(spec)["input_report"]]:
        raise ValueError("Pinned input report SHA/bytes differs from the source inventory")


def _legacy(spec, pins):
    return (spec == PublicClipSpec(15, 501, "front_stereo_camera_left", 1152, 1536)
            and pins["input_report"] == LEGACY_INPUT_REPORT)


def _input_dataset_omitted(spec, pins):
    """Exact historical input metadata compatibility, never a legacy mode."""
    return (spec == PublicClipSpec(0, 790, "front_stereo_camera_left", 1152, 1536)
            and pins["input_report"] == HISTORICAL_INPUT_DATASET_OMISSION)


def _record_identity(record, spec, *, legacy, dataset_required=False):
    public._no_oracle(record)
    if ("episode_index" not in record and not legacy) or ("episode_index" in record
            and (type(record["episode_index"]) is not int or record["episode_index"] != spec.episode_index)):
        raise ValueError("Report must explicitly identify the selected original episode")
    if (dataset_required and "input_dataset_revision" not in record and not legacy
            or "input_dataset_revision" in record and record["input_dataset_revision"] != DATASET_REVISION):
        raise ValueError("Report input dataset revision differs or is missing")


def validate_reports(root, spec, pins):
    validate_pins(spec, pins)
    object_source = source_profile(pins)
    paths, deps = relative_paths(spec), dependency_paths(spec, object_source=object_source)
    read = lambda name: json.loads(public.regular(root / name).read_text())
    report = read(paths["input_report"])
    legacy = _legacy(spec, pins)
    _record_identity(report, spec, legacy=legacy, dataset_required=not _input_dataset_omitted(spec, pins))
    if (report.get("stage") != "world_reward_native_cari_inputs" or report.get("status") != "pass"
            or report.get("input_track") != "track_1" or type(report.get("frames")) is not int
            or report["frames"] != spec.total_frames or report.get("original_frame_coverage_verified") is not True
            or any(report.get(name) != pins["input_report"][name] for name in ("producer_revision", "script_sha256"))
            or type(report.get("input_sha256")) is not str or not re.fullmatch(r"[0-9a-f]{64}", report["input_sha256"])
            or report.get("object_pose_initializer") != "own_ICP_Viterbi_not_FoundationPose"):
        raise ValueError("Pinned passing full original native input preparation required")
    for name, key in (("export_seq", "export_seq"), ("depth_h5", "depth_h5"), ("mhr_init", "initializer"), ("object_poses", "object_poses")):
        if report.get(name) != str(root / paths[key]):
            raise ValueError("Preparation canonical public input paths differ")
    for name, key in (("depth_h5", "depth_h5"), ("mhr_init", "initializer"), ("object_poses", "object_poses"), ("wild_export", "wild_export")):
        if report.get("file_sha256", {}).get(name) != pins["source_files"][paths[key]]["sha256"]:
            raise ValueError("Preparation public file artifact chain differs")
    records = {"inputs": report}
    for role, relative in deps.items():
        item = read(relative)
        _record_identity(item, spec, legacy=legacy, dataset_required=role in {"body", "depth"})
        if (item.get("status") != "pass" or item.get("stage") != DEPENDENCY_STAGES[role]
                or item.get("input_track") != "track_1"
                or report.get("input_report_sha256", {}).get(role) != pins["source_files"][relative]["sha256"]):
            raise ValueError("Pinned full initializer dependency report differs")
        if role != "adapter" and item.get("input_sha256") != report["input_sha256"]:
            raise ValueError("Dependency original video SHA differs")
        if role == "adapter" and "input_sha256" in item and item["input_sha256"] != report["input_sha256"]:
            raise ValueError("Adapter original video SHA differs")
        if role in {"body", "depth", "object"}:
            rows = item.get("frames")
            if (type(rows) is not list or any(type(row) is not dict for row in rows)
                    or [row.get("frame_index") for row in rows] != list(range(spec.total_frames))
                    or any(type(row.get("frame_index")) is not int for row in rows)):
                raise ValueError("Exact ordered full original frame coverage required")
            if "total_video_frames" in item and (type(item["total_video_frames"]) is not int or item["total_video_frames"] != spec.total_frames):
                raise ValueError("Dependency full original frame count differs")
        elif role == "adapter":
            if (type(item.get("frames")) is not int or item["frames"] != spec.total_frames
                    or item.get("canonical_initializer_sha256") != pins["source_files"][paths["initializer"]]["sha256"]
                    or item.get("body_report_sha256") != pins["source_files"][deps["body"]]["sha256"]):
                raise ValueError("Canonical initializer/Body report binding differs")
        records[role] = item
    if (records["object"].get("full_depth_report_sha256") != pins["source_files"][deps["depth"]]["sha256"]
            or records["object"].get("alignment_report_sha256") != pins["source_files"][deps["alignment"]]["sha256"]):
        raise ValueError("Object poses must bind full inferred depth and the shared human gauge")
    source_pose = records["object"].get("geometry_and_poses_sha256")
    if type(source_pose) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_pose):
        raise ValueError("Original automatic object pose artifact SHA required")
    if object_source == "solid":
        expected_source = dict(report=deps["object"],
            geometry_and_poses="outputs/" + spec.sequence + "/object_pose_full_solid/geometry_and_poses.npz",
            geometry_and_poses_sha256=source_pose)
        if (type(report.get("object_source")) is not str or report["object_source"] != "solid"
                or report.get("object_pose_source") != expected_source
                or type(report.get("object_pose_source")) is not dict
                or set(report["object_pose_source"]) != set(expected_source)
                or records["object"].get("mesh_source") != "solid"):
            raise ValueError("Pinned solid source/report/pose SHA must agree with the preparation")
    elif object_source == "surface":
        expected_source=dict(report=deps['object'],
            geometry_and_poses='outputs/'+spec.sequence+'/object_pose_full_surface/geometry_and_poses.npz',
            geometry_and_poses_sha256=source_pose)
        if (report.get('object_source')!='surface' or type(report.get('object_pose_source')) is not dict
                or report['object_pose_source']!=expected_source or records['object'].get('mesh_source')!='surface'):
            raise ValueError('Pinned surface source/report/pose SHA must agree with preparation')
        proof=report.get('surface_geometry_validation');native=records['object'].get('topology_budget')
        keys={'committed_pins_sha256','producer_report_sha256','cpu_native_report_sha256','source_domain',
              'metric_scale_baked_once','geometry_operations_applied'}
        extras={'metric_glb','native_aligned_glb','files','source_rehashed_after'}
        if type(proof) is not dict or set(proof)!=keys|extras or type(native) is not dict:
            raise ValueError('Exact actual native surface geometry proof required')
        if (proof['source_rehashed_after'] is not True or type(proof['metric_glb']) is not dict
                or not proof['metric_glb'] or type(proof['native_aligned_glb']) is not dict or not proof['native_aligned_glb']
                or type(proof['files']) is not dict or not 0<len(proof['files'])<=64):
            raise ValueError('Actual metric/aligned surface representation and frozen input ledger required')
        for name,row in proof['files'].items():
            if type(name) is not str or not 0<len(name)<=1024:
                raise ValueError('Bounded explicit surface provenance name required')
            _receipt(row)
        for key in ('committed_pins_sha256','producer_report_sha256','cpu_native_report_sha256'):
            if type(proof[key]) is not str or not re.fullmatch('[0-9a-f]{64}',proof[key]) or proof[key]!=native.get(key):
                raise ValueError('Surface producer/pin/native byte binding differs')
        import math
        scale=proof['metric_scale_baked_once']
        if (proof['source_domain']!='surface' or proof['geometry_operations_applied'] is not False
                or type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0
                or any(type(native.get(k)) is not type(proof[k]) or native[k]!=proof[k] for k in keys)):
            raise ValueError('Native surface geometry must retain the original metric gauge without operations')
    elif report.get("object_source", "default") != "default" or "object_pose_source" in report:
        raise ValueError("Legacy pins cannot select a different object source")
    validation = report.get("depth_validation", {})
    if (validation.get("validation_mode") != "exhaustive"
            or validation.get("frame_counts") != {spec.camera_name: spec.total_frames}
            or validation.get("frame_shapes") != {spec.camera_name: [spec.height, spec.width]}):
        raise ValueError("Original producer exhaustive full-grid depth validation required")
    scale = records["alignment"].get("depth_alignment", {}).get("shared_scale")
    import math
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Original positive shared predicted human depth gauge required")
    return records


def _float_array(value, shape):
    import numpy as np
    if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Exact unmasked finite original native float32 array required")
    return value


def validate_initializer(initializer, spec):
    import numpy as np
    from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters
    if (type(initializer) is not dict or initializer.get("body_model") != "mhr"
            or initializer.get("frames") != [f"{index:06d}" for index in range(spec.total_frames)]
            or initializer.get("kids") != [0] or any(type(kid) is not int for kid in initializer["kids"])):
        raise ValueError("Complete full native canonical original initializer required")
    public._no_oracle(initializer.get("metadata"))
    if initializer["metadata"].get("mhr_geometry_forward_verified") is not True:
        raise ValueError("Original initializer native decoder parity required")
    validate_native_parameters({key: initializer.get(key) for key in NATIVE_PARAMETER_DIMS}, spec.total_frames)
    _float_array(initializer.get("mhr_joints"), (spec.total_frames, 127, 3))
    _float_array(initializer.get("mhr_keypoints"), (spec.total_frames, 70, 3))
    if "camera_intrinsics" in initializer["metadata"]:
        expected = inferred_camera(spec).astype(np.float32)
        actual = np.asarray(initializer["metadata"]["camera_intrinsics"], np.float64)
        if not np.array_equal(actual, expected):
            raise ValueError("Original initializer camera must retain the inferred RGB-size K")


def inferred_camera(spec):
    import numpy as np
    _spec(spec)
    focal = np.hypot(spec.height, spec.width)
    return np.array([[focal, 0, spec.width / 2], [0, focal, spec.height / 2], [0, 0, 1]], np.float64)


def _rigid(value):
    import numpy as np
    a = np.asarray(value, np.float64)
    if (a.ndim not in (2, 3) or a.shape[-2:] != (4, 4) or not np.isfinite(a).all()
            or not np.array_equal(a[..., 3, :], np.broadcast_to([0, 0, 0, 1], a[..., 3, :].shape))
            or not np.allclose(a[..., :3, :3] @ a[..., :3, :3].swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(a[..., :3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError("Original proper rigid mesh/pose frame required; no scale or reflection")
    return a


def validate_poses(poses, wild, spec):
    import numpy as np
    if (type(poses) is not dict or set(poses) != {"frames", "obj_pose_world", "metadata"}
            or poses["frames"] != [f"{index:06d}" for index in range(spec.total_frames)]):
        raise ValueError("Exact full automatic native object pose schema required")
    public._no_oracle(poses["metadata"])
    if poses["metadata"].get("source") != "World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose":
        raise ValueError("Own inferred ICP/Viterbi trajectory required")
    _rigid(_float_array(poses["obj_pose_world"], (spec.total_frames, 4, 4)))
    transform = _rigid(wild.get("source_object_mesh_to_aligned_transform"))
    if transform.shape != (4, 4) or not np.array_equal(transform, poses["metadata"].get("mesh_frame_change")):
        raise ValueError("Same original mesh/pose rigid frame change required")


def validate_wild(wild, edex, root, spec):
    import numpy as np
    paths = relative_paths(spec)
    if (type(wild) is not dict or wild.get("schema") != "cari4d.mhr_wild_export.v2"
            or type(wild.get("frame_count")) is not int or wild["frame_count"] != spec.total_frames
            or wild.get("sequence") != spec.sequence or type(wild.get("camera_id")) is not int or wild["camera_id"] != 0
            or type(wild.get("height")) is not int or type(wild.get("width")) is not int
            or (wild["height"], wild["width"]) != (spec.height, spec.width)
            or wild.get("object_mesh_file") != str(root / paths["mesh"])
            or wild.get("depth_backend") != "moge2" or wild.get("object_pose_frame") != "centered_axis_aligned"
            or wild.get("object_pose_frame_revision") != "cari4d.object_pose_frame.centered_axis_aligned.v1"
            or wild.get("object_pose_storage_frame") != "output_aligned_mesh_frame"):
        raise ValueError("Original full single-camera predicted wild export required")
    for key in ("object_pose_storage_to_training_transform", "object_mesh_to_training_transform"):
        if not np.array_equal(wild.get(key), np.eye(4)):
            raise ValueError("Original aligned mesh storage/training frame must remain identity")
    if not np.array_equal(wild.get("intrinsics"), inferred_camera(spec)):
        raise ValueError("Original inferred RGB-size camera changed")
    if (type(edex) is not list or len(edex) != 1 or type(edex[0]) is not dict
            or type(edex[0].get("cameras")) is not list or len(edex[0]["cameras"]) != 1
            or type(edex[0]["cameras"][0]) is not dict):
        raise ValueError("Exactly one original native camera in edex required")
    camera = edex[0]["cameras"][0]
    K = inferred_camera(spec)
    if (camera.get("intrinsics") != {"focal": [K[0, 0], K[1, 1]], "principal": [K[0, 2], K[1, 2]]}
            or not np.array_equal(camera.get("transform"), np.eye(4)[:3])):
        raise ValueError("Same inferred K and camera-as-world identity required")


def verify_public_inputs(root, spec, pins):
    """Return full original inputs only after complete strict source hashing.

    No copies/outputs/chmod, geometry decoding, source repair or prefix export.
    The original files and explicit pins are rehashed after interpretation.
    A producer's recorded numerical claims are not independently reverified.
    """
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Canonical existing public runtime root required")
    validate_pins(spec, pins)
    pinned = copy.deepcopy(pins)
    observed = {name: identity(root / name) for name in sorted(source_paths(spec, object_source=source_profile(pinned)))}
    if observed != pinned["source_files"]:
        raise ValueError("Complete public source SHA/bytes differs before deserialization")
    records = validate_reports(root, spec, pinned)
    relative = relative_paths(spec)
    paths = {name: root / path for name, path in relative.items()}
    wild = json.loads(paths["wild_export"].read_text())
    edex = json.loads((paths["export_seq"] / "edex").read_text())
    validate_wild(wild, edex, root, spec)
    import joblib
    initializer = joblib.load(paths["initializer"])
    validate_initializer(initializer, spec)
    poses = joblib.load(paths["object_poses"])
    validate_poses(poses, wild, spec)
    if poses["metadata"].get("source_pose_sha256") != records["object"].get("geometry_and_poses_sha256"):
        raise ValueError("Original object pose artifact/producer chain differs")
    if (pins != pinned or {name: identity(root / name) for name in observed} != observed):
        raise ValueError("Original public source or explicit pins changed during validation")
    return {"spec": spec, "initializer": initializer, "poses": poses, "wild": wild,
            "reports": records, "paths": paths, "source_files": observed}

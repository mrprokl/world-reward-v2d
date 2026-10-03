"""Generic full clip audit with tiny actual Joblib files and no media decode."""
import copy
from dataclasses import FrozenInstanceError, asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    name = "world_reward_test_cari_clip_inputs"
    spec = importlib.util.spec_from_file_location(name, infra / "cari_clip_inputs.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def flags(gate, spec):
    return dict(status="pass", input_track="track_1", input_sha256="b" * 64,
                episode_index=spec.episode_index, input_dataset_revision=gate.DATASET_REVISION,
                ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])


def initializer(gate, spec):
    from world_reward.shared_identity import NATIVE_PARAMETER_DIMS
    n = spec.total_frames
    value = {name: np.zeros((n, dimension), np.float32) for name, dimension in NATIVE_PARAMETER_DIMS.items()}
    value["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    value["mhr_trans"][:, 2] = 2
    value["mhr_shape"][:, 0] = np.arange(n, dtype=np.float32) / 100
    return dict(value, body_model="mhr", kids=[0], frames=[f"{i:06d}" for i in range(n)],
                mhr_joints=np.ones((n, 127, 3), np.float32), mhr_keypoints=np.ones((n, 70, 3), np.float32),
                metadata=dict(ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
                              mhr_geometry_forward_verified=True, camera_intrinsics=gate.inferred_camera(spec).astype(np.float32).tolist()))


def poses(spec):
    values = np.broadcast_to(np.eye(4, dtype=np.float32), (spec.total_frames, 4, 4)).copy()
    values[:, 0, 3] = np.arange(spec.total_frames, dtype=np.float32) / 100
    return dict(frames=[f"{i:06d}" for i in range(spec.total_frames)], obj_pose_world=values,
                metadata=dict(ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
                              source="World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose",
                              source_pose_sha256="a" * 64, mesh_frame_change=np.eye(4).tolist()))


def wild(gate, root, spec):
    paths = gate.relative_paths(spec)
    return dict(schema="cari4d.mhr_wild_export.v2", sequence=spec.sequence, camera_id=0,
                frame_count=spec.total_frames, height=spec.height, width=spec.width,
                object_mesh_file=str(root / paths["mesh"]), intrinsics=gate.inferred_camera(spec).tolist(),
                depth_backend="moge2", object_pose_frame="centered_axis_aligned",
                object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
                object_pose_storage_frame="output_aligned_mesh_frame",
                object_pose_storage_to_training_transform=np.eye(4).tolist(),
                object_mesh_to_training_transform=np.eye(4).tolist(),
                source_object_mesh_to_aligned_transform=np.eye(4).tolist())


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(gate, root, spec):
    paths, deps = gate.relative_paths(spec), gate.dependency_paths(spec)
    for relative in gate.source_paths(spec):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque public payload: not decoded")
    joblib.dump(initializer(gate, spec), root / paths["initializer"])
    joblib.dump(poses(spec), root / paths["object_poses"])
    write_json(root / paths["wild_export"], wild(gate, root, spec))
    k = gate.inferred_camera(spec)
    write_json(root / paths["export_seq"] / "edex", [{"cameras": [dict(
        intrinsics={"focal": [k[0, 0], k[1, 1]], "principal": [k[0, 2], k[1, 2]]},
        transform=np.eye(4)[:3].tolist())]}])
    for role in ("body", "depth", "alignment", "object", "adapter"):
        row = flags(gate, spec) | {"stage": gate.DEPENDENCY_STAGES[role]}
        if role in {"body", "depth", "object"}:
            row["frames"] = [{"frame_index": i} for i in range(spec.total_frames)]
            row["total_video_frames"] = spec.total_frames
        if role == "alignment": row["depth_alignment"] = {"shared_scale": .83}
        if role == "object": row.update(geometry_and_poses_sha256="a" * 64,
            full_depth_report_sha256=gate.identity(root / deps["depth"])["sha256"],
            alignment_report_sha256=gate.identity(root / deps["alignment"])["sha256"])
        if role == "adapter": row.update(frames=spec.total_frames,
            canonical_initializer_sha256=gate.identity(root / paths["initializer"])["sha256"],
            body_report_sha256=gate.identity(root / deps["body"])["sha256"])
        write_json(root / deps[role], row)
    write_json(root / paths["input_report"], flags(gate, spec) | dict(
        stage="world_reward_native_cari_inputs", frames=spec.total_frames,
        original_frame_coverage_verified=True, producer_revision="c" * 40, script_sha256="d" * 64,
        object_pose_initializer="own_ICP_Viterbi_not_FoundationPose",
        export_seq=str(root / paths["export_seq"]), depth_h5=str(root / paths["depth_h5"]),
        mhr_init=str(root / paths["initializer"]), object_poses=str(root / paths["object_poses"]),
        depth_validation={"validation_mode": "exhaustive", "frame_counts": {spec.camera_name: spec.total_frames},
                          "frame_shapes": {spec.camera_name: [spec.height, spec.width]}},
        input_report_sha256={role: gate.identity(root / relative)["sha256"] for role, relative in deps.items()},
        file_sha256={name: gate.identity(root / paths[key])["sha256"] for name, key in
                     (("depth_h5", "depth_h5"), ("mhr_init", "initializer"), ("object_poses", "object_poses"), ("wild_export", "wild_export"))}))
    return repin(gate, root, spec)


def repin(gate, root, spec):
    paths, deps = gate.relative_paths(spec), gate.dependency_paths(spec)
    adapter = json.loads((root / deps["adapter"]).read_text())
    adapter.update(canonical_initializer_sha256=gate.identity(root / paths["initializer"])["sha256"],
                   body_report_sha256=gate.identity(root / deps["body"])["sha256"])
    write_json(root / deps["adapter"], adapter)
    obj = json.loads((root / deps["object"]).read_text())
    # Keep deliberately corrupted links if a test requested that corruption.
    if obj.get("full_depth_report_sha256") != "f" * 64:
        obj["full_depth_report_sha256"] = gate.identity(root / deps["depth"])["sha256"]
    if obj.get("alignment_report_sha256") != "f" * 64:
        obj["alignment_report_sha256"] = gate.identity(root / deps["alignment"])["sha256"]
    write_json(root / deps["object"], obj)
    row = json.loads((root / paths["input_report"]).read_text())
    row["input_report_sha256"] = {role: gate.identity(root / relative)["sha256"] for role, relative in deps.items()}
    row["file_sha256"] = {name: gate.identity(root / paths[key])["sha256"] for name, key in
                          (("depth_h5", "depth_h5"), ("mhr_init", "initializer"), ("object_poses", "object_poses"), ("wild_export", "wild_export"))}
    write_json(root / paths["input_report"], row)
    return dict(schema="world-reward-cari-clip-input-pins-v1", clip_spec=asdict(spec),
                input_report=gate.identity(root / paths["input_report"]) | {name: row[name] for name in ("producer_revision", "script_sha256")},
                source_files={name: gate.identity(root / name) for name in gate.source_paths(spec)})


@pytest.mark.parametrize("episode", [0, 15, 29])
@pytest.mark.parametrize("count", [96, 97, 501])
def test_generic_original_clip_all_frames_reuses_only_public_verified_sources(gate, tmp_path, episode, count):
    spec = gate.PublicClipSpec(episode, count, "front_stereo_camera_left", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    before = copy.deepcopy(pins)
    result = gate.verify_public_inputs(tmp_path, spec, pins)
    assert pins == before
    assert set(result) == {"spec", "initializer", "poses", "wild", "reports", "paths", "source_files"}
    assert result["spec"] is spec and result["source_files"] == pins["source_files"]
    assert set(result["reports"]) == {"inputs", "body", "depth", "object", "alignment", "adapter"}
    assert result["initializer"]["mhr_keypoints"].shape == (count, 70, 3)
    assert result["initializer"]["mhr_joints"].shape == (count, 127, 3)
    assert result["initializer"]["frames"] == result["poses"]["frames"] == [f"{i:06d}" for i in range(count)]
    assert len(gate.source_paths(spec)) == 15
    assert not any(token in path for path in result["source_files"] for token in ("cari_forward", "cari_refined", "target.npy", "track_2", "track_3"))


@pytest.mark.parametrize("field,bad", [
    ("episode_index", -1), ("episode_index", 30), ("episode_index", True), ("episode_index", np.int64(15)),
    ("total_frames", 95), ("total_frames", True), ("total_frames", 96.),
    ("camera_name", "../camera"), ("camera_name", ""), ("camera_name", "a" * 65),
    ("height", 1), ("height", True), ("width", 4.),
])
def test_spec_explicit_frozen_and_cannot_invent_episode_frames_camera(gate, field, bad):
    values = dict(episode_index=15, total_frames=501, camera_name="front", height=2, width=4)
    values[field] = bad
    with pytest.raises(ValueError): gate.PublicClipSpec(**values)


def test_spec_is_frozen_and_real_ep15_inventory_matches_prior_pins(gate):
    spec = gate.PublicClipSpec(15, 501, "front_stereo_camera_left", 1152, 1536)
    pins = json.loads((Path(__file__).parents[1] / "configs/cari96_input_pins.json").read_text())
    assert gate.source_paths(spec) == set(pins["source_files"])
    assert gate.LEGACY_INPUT_REPORT == pins["input_report"]
    assert gate.DATASET_REVISION == "5f68335f3acc802033d1e80728c1633197521de8"
    with pytest.raises(FrozenInstanceError): spec.total_frames = 96


@pytest.mark.parametrize("fault", ["schema", "extra", "missing", "clip", "boolclip", "gtfile", "hash", "bytes", "reporthash", "producer", "script"])
def test_explicit_pins_fail_before_deserializing_or_reading_unknown_inputs(gate, tmp_path, monkeypatch, fault):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    if fault == "schema": pins["schema"] = "wrong"
    elif fault == "extra": pins["other"] = 1
    elif fault == "missing": pins["source_files"].pop(next(iter(pins["source_files"])))
    elif fault == "clip": pins["clip_spec"]["episode_index"] = 1
    elif fault == "boolclip": pins["clip_spec"]["episode_index"] = False
    elif fault == "gtfile": pins["source_files"]["private/GT.pkl"] = {"sha256": "a" * 64, "bytes": 1}
    elif fault == "hash": pins["source_files"][gate.relative_paths(spec)["initializer"]]["sha256"] = "f" * 64
    elif fault == "bytes": pins["source_files"][gate.relative_paths(spec)["initializer"]]["bytes"] = True
    elif fault == "reporthash": pins["input_report"]["sha256"] = "f" * 64
    elif fault == "producer": pins["input_report"]["producer_revision"] = "invalid"
    elif fault == "script": pins["input_report"]["script_sha256"] = "invalid"
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("No unverified deserialization"))
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, pins)


@pytest.mark.parametrize("role", ["inputs", "body", "depth", "object", "alignment", "adapter"])
@pytest.mark.parametrize("fault", ["GT", "hand", "oracle", "privateread", "episode", "episodeabsent", "dataset", "stage", "status"])
def test_all_public_report_roles_reject_misbound_or_ambiguous_provenance(gate, tmp_path, role, fault):
    spec = gate.PublicClipSpec(29, 97, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    relative = gate.relative_paths(spec)["input_report"] if role == "inputs" else gate.dependency_paths(spec)[role]
    path = tmp_path / relative
    row = json.loads(path.read_text())
    if fault == "GT": row["ground_truth_used"] = True
    elif fault == "hand": row["hand_labeled_test"] = True
    elif fault == "oracle": row["oracle_modes"] = ["gt"]
    elif fault == "privateread": row["private_truth_read"] = True
    elif fault == "episode": row["episode_index"] = True
    elif fault == "episodeabsent": row.pop("episode_index")
    elif fault == "dataset": row["input_dataset_revision"] = "f" * 40
    elif fault == "stage": row["stage"] = "old_forward"
    elif fault == "status": row["status"] = "fail"
    write_json(path, row)
    pins = repin(gate, tmp_path, spec)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, pins)


@pytest.mark.parametrize("role", ["inputs", "body", "depth"])
def test_new_full_inputs_require_explicit_dataset_revision(gate, tmp_path, role):
    spec = gate.PublicClipSpec(15, 96, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    path = tmp_path / (gate.relative_paths(spec)["input_report"] if role == "inputs" else gate.dependency_paths(spec)[role])
    row = json.loads(path.read_text()); row.pop("input_dataset_revision"); write_json(path, row)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


@pytest.mark.parametrize("role", ["body", "depth", "object"])
@pytest.mark.parametrize("fault", ["missingframe", "duplicate", "boolindex", "reorder", "video", "count", "nondict"])
def test_complete_original_body_depth_object_timelines_are_not_reindexed(gate, tmp_path, role, fault):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    path = tmp_path / gate.dependency_paths(spec)[role]
    row = json.loads(path.read_text())
    if fault == "missingframe": row["frames"].pop()
    elif fault == "duplicate": row["frames"][1]["frame_index"] = 0
    elif fault == "boolindex": row["frames"][0]["frame_index"] = False
    elif fault == "reorder": row["frames"].reverse()
    elif fault == "video": row["input_sha256"] = "f" * 64
    elif fault == "count": row["total_video_frames"] = 95
    elif fault == "nondict": row["frames"][0] = None
    write_json(path, row)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


@pytest.mark.parametrize("fault", ["sourcepath", "framecount", "depthvalidation", "fixedmesh", "objectdepth", "objectalignment", "sourceposehash", "gauge"])
def test_native_paths_geometry_and_depth_gauge_are_frozen(gate, tmp_path, fault):
    spec = gate.PublicClipSpec(15, 97, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    paths, deps = gate.relative_paths(spec), gate.dependency_paths(spec)
    role = "object" if fault in {"objectdepth", "objectalignment", "sourceposehash"} else "alignment" if fault == "gauge" else "inputs"
    path = tmp_path / (paths["input_report"] if role == "inputs" else deps[role])
    row = json.loads(path.read_text())
    if fault == "sourcepath": row["mhr_init"] = str(tmp_path / "cari_forward/coconet.pth")
    elif fault == "framecount": row["frames"] = 96
    elif fault == "depthvalidation": row["depth_validation"]["validation_mode"] = "sampled"
    elif fault == "fixedmesh": row["object_pose_initializer"] = "GT"
    elif fault == "objectdepth": row["full_depth_report_sha256"] = "f" * 64
    elif fault == "objectalignment": row["alignment_report_sha256"] = "f" * 64
    elif fault == "sourceposehash": row.pop("geometry_and_poses_sha256")
    elif fault == "gauge": row["depth_alignment"]["shared_scale"] = 0
    write_json(path, row)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


@pytest.mark.parametrize("fault", ["root", "face", "z", "dtype", "masked", "joints", "keypoints", "geometrydtype", "frames", "kids", "camera", "parity", "oracle"])
def test_original_initializer_native7_and_geometry_are_exact(gate, tmp_path, fault):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    path = tmp_path / gate.relative_paths(spec)["initializer"]
    row = joblib.load(path)
    if fault == "root": row["mhr_global_rot6d"][2] = 0
    elif fault == "face": row["mhr_face"][2, 0] = .1
    elif fault == "z": row["mhr_trans"][2, 2] = 0
    elif fault == "dtype": row["mhr_shape"] = row["mhr_shape"].astype(np.float64)
    elif fault == "masked": row["mhr_hand"] = np.ma.array(row["mhr_hand"], mask=False)
    elif fault == "joints": row["mhr_joints"] = row["mhr_joints"][:, :-1]
    elif fault == "keypoints": row["mhr_keypoints"] = row["mhr_keypoints"][:, :-1]
    elif fault == "geometrydtype": row["mhr_joints"] = row["mhr_joints"].astype(np.float64)
    elif fault == "frames": row["frames"].reverse()
    elif fault == "kids": row["kids"] = [False]
    elif fault == "camera": row["metadata"]["camera_intrinsics"][0][0] += 1
    elif fault == "parity": row["metadata"]["mhr_geometry_forward_verified"] = False
    elif fault == "oracle": row["metadata"]["oracle_modes"] = ["gt"]
    joblib.dump(row, path)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


@pytest.mark.parametrize("fault", ["scale", "reflection", "perspective", "dtype", "frames", "oracle", "source", "sourcehash", "framechange"])
def test_original_object_trajectory_and_local_mesh_frame_cannot_be_repaired(gate, tmp_path, fault):
    spec = gate.PublicClipSpec(29, 96, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    path = tmp_path / gate.relative_paths(spec)["object_poses"]
    row = joblib.load(path)
    if fault == "scale": row["obj_pose_world"][0, 0, 0] = 2
    elif fault == "reflection": row["obj_pose_world"][0, 0, 0] = -1
    elif fault == "perspective": row["obj_pose_world"][0, 3, 0] = .1
    elif fault == "dtype": row["obj_pose_world"] = row["obj_pose_world"].astype(np.float64)
    elif fault == "frames": row["frames"].reverse()
    elif fault == "oracle": row["metadata"]["oracle_modes"] = ["gt"]
    elif fault == "source": row["metadata"]["source"] = "GT"
    elif fault == "sourcehash": row["metadata"]["source_pose_sha256"] = "f" * 64
    elif fault == "framechange": row["metadata"]["mesh_frame_change"][0][3] = .1
    joblib.dump(row, path)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


@pytest.mark.parametrize("fault", ["camera", "K", "mesh", "scale", "frame", "framecount", "height", "multiview", "extrinsic"])
def test_wild_export_single_camera_and_fixed_mesh_metadata_are_unchanged(gate, tmp_path, fault):
    spec = gate.PublicClipSpec(15, 96, "front", 2, 4)
    fixture(gate, tmp_path, spec)
    paths = gate.relative_paths(spec)
    path = tmp_path / paths["wild_export"]
    row = json.loads(path.read_text())
    if fault == "camera": row["camera_id"] = True
    elif fault == "K": row["intrinsics"][0][0] += 1
    elif fault == "mesh": row["object_mesh_file"] = str(tmp_path / "private/mesh.glb")
    elif fault == "scale": row["object_mesh_to_training_transform"][0][0] = 2
    elif fault == "frame": row["object_pose_storage_frame"] = "GT"
    elif fault == "framecount": row["frame_count"] = 97
    elif fault == "height": row["height"] = True
    else:
        edex_path = tmp_path / paths["export_seq"] / "edex"
        edex = json.loads(edex_path.read_text())
        if fault == "multiview": edex[0]["cameras"].append(edex[0]["cameras"][0])
        else: edex[0]["cameras"][0]["transform"][0][3] = 1
        write_json(edex_path, edex)
    write_json(path, row)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, repin(gate, tmp_path, spec))


def test_all_fifteen_hashes_verified_before_any_json_or_pickle_interpretation(gate, tmp_path, monkeypatch):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    seen = []
    real_identity = gate.identity
    def recording(path):
        seen.append(str(Path(path).relative_to(tmp_path)))
        return real_identity(path)
    real_loads, real_load = json.loads, joblib.load
    def json_after_hash(*args, **kwargs):
        assert set(seen) == gate.source_paths(spec)
        return real_loads(*args, **kwargs)
    def joblib_after_hash(*args, **kwargs):
        assert set(seen) == gate.source_paths(spec)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(gate, "identity", recording)
    monkeypatch.setattr(json, "loads", json_after_hash)
    monkeypatch.setattr(joblib, "load", joblib_after_hash)
    gate.verify_public_inputs(tmp_path, spec, pins)
    assert len(seen) == 30


def test_no_other_video_gt_cache_or_geometry_prediction_paths_are_opened(gate, tmp_path, monkeypatch):
    spec = gate.PublicClipSpec(15, 96, "front", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    allowed = {str(tmp_path / relative) for relative in gate.source_paths(spec)}
    actual = Path.open
    seen = []
    def only_public(path, *args, **kwargs):
        assert str(path) in allowed
        seen.append(str(path))
        return actual(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", only_public)
    gate.verify_public_inputs(tmp_path, spec, pins)
    assert set(seen) == allowed


def test_original_file_mutation_during_interpretation_is_rejected(gate, tmp_path, monkeypatch):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    original_load = joblib.load
    def mutate(path):
        value = original_load(path)
        (tmp_path / gate.relative_paths(spec)["mesh"]).write_bytes(b"changed")
        return value
    monkeypatch.setattr(joblib, "load", mutate)
    with pytest.raises(ValueError, match="changed during"):
        gate.verify_public_inputs(tmp_path, spec, pins)


def test_symlinks_rejected_before_any_deserialization(gate, tmp_path, monkeypatch):
    spec = gate.PublicClipSpec(0, 96, "front", 2, 4)
    pins = fixture(gate, tmp_path, spec)
    path = tmp_path / gate.relative_paths(spec)["initializer"]
    target = path.with_suffix(".bak"); path.rename(target); path.symlink_to(target)
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("No symlink deserialization"))
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, pins)


def test_specific_known_legacy_ep15_input_identity_only_not_arbitrary_missing_fields(gate):
    spec = gate.PublicClipSpec(15, 501, "front_stereo_camera_left", 1152, 1536)
    pins = {"input_report": copy.deepcopy(gate.LEGACY_INPUT_REPORT)}
    assert gate._legacy(spec, pins)
    for field, value in (("producer_revision", "f" * 40), ("script_sha256", "f" * 64), ("sha256", "f" * 64), ("bytes", 1)):
        changed = copy.deepcopy(pins); changed["input_report"][field] = value
        assert not gate._legacy(spec, changed)
    assert not gate._legacy(gate.PublicClipSpec(0, 501, "front_stereo_camera_left", 1152, 1536), pins)
    assert not gate._legacy(gate.PublicClipSpec(15, 96, "front_stereo_camera_left", 1152, 1536), pins)
    row = dict(ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    gate._record_identity(row, spec, legacy=True, dataset_required=True)
    assert "episode_index" not in row and "input_dataset_revision" not in row
    with pytest.raises(ValueError): gate._record_identity(row, spec, legacy=False, dataset_required=True)
    with pytest.raises(ValueError): gate._record_identity(dict(row, episode_index=0), spec, legacy=True)

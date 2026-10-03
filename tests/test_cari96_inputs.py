"""Tiny public snapshot contracts; actual HDF5 cases skip without h5py."""
import copy
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    path = Path(__file__).parents[1] / "infra/cari96_inputs.py"
    spec = importlib.util.spec_from_file_location("cari96_inputs_tests", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SOURCE_FRAMES", 5)
    monkeypatch.setattr(module, "FRAMES", 3)
    monkeypatch.setattr(module, "IMAGE_SIZE", (2, 4))
    return module


def initializer(gate):
    n = gate.SOURCE_FRAMES
    arrays = {key: np.zeros((n, dim), np.float32) for key, dim in gate.PARAMETER_DIMS.items()}
    arrays["mhr_trans"][:, 2] = 2
    return arrays | {"body_model": "mhr", "frames": [f"{i:06d}" for i in range(n)], "kids": [0],
        "mhr_joints": np.ones((n, 2, 3), np.float32), "mhr_keypoints": np.ones((n, 3, 3), np.float32),
        "metadata": {"ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
                     "mhr_geometry_forward_verified": True}}


def wild_metadata(gate, root):
    h, w = gate.IMAGE_SIZE
    return {"schema": "cari4d.mhr_wild_export.v2", "sequence": "episode_000015", "camera_id": 0,
        "frame_count": gate.SOURCE_FRAMES, "height": h, "width": w,
        "object_mesh_file": str(root / gate.EXPORT / "object_mesh/output_aligned.glb"),
        "intrinsics": [[np.hypot(h, w), 0, w / 2], [0, np.hypot(h, w), h / 2], [0, 0, 1]],
        "depth_backend": "moge2", "object_pose_frame": "centered_axis_aligned",
        "object_pose_frame_revision": "cari4d.object_pose_frame.centered_axis_aligned.v1",
        "object_pose_storage_frame": "output_aligned_mesh_frame",
        "object_pose_storage_to_training_transform": np.eye(4).tolist(),
        "object_mesh_to_training_transform": np.eye(4).tolist(),
        "source_object_mesh_to_aligned_transform": np.eye(4).tolist()}


def poses(gate):
    a = np.broadcast_to(np.eye(4, dtype=np.float32), (gate.SOURCE_FRAMES, 4, 4)).copy()
    a[:, 0, 3] = np.arange(gate.SOURCE_FRAMES) / 100
    return {"frames": [f"{i:06d}" for i in range(gate.SOURCE_FRAMES)], "obj_pose_world": a,
        "metadata": {"ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
            "source": "World_Reward_fixed_scale_depth_ICP_Viterbi_not_FoundationPose",
            "source_pose_sha256": "a" * 64, "mesh_frame_change": np.eye(4).tolist()}}


def source_fixture(gate, root):
    camera = "front_stereo_camera_left"
    for relative in gate.source_paths(camera):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"placeholder")
    w = wild_metadata(gate, root)
    (root / gate.EXPORT / "wild_export.json").write_text(json.dumps(w))
    h, width = gate.IMAGE_SIZE
    edex = [{"cameras": [{"intrinsics": {"focal": [np.hypot(h, width)] * 2,
        "principal": [width / 2, h / 2]}, "transform": np.eye(4)[:3].tolist()}]}]
    (root / gate.EXPORT / "edex").write_text(json.dumps(edex))
    for relative, value in ((gate.INITIALIZER, initializer(gate)), (gate.POSES, poses(gate))):
        (root / relative).write_bytes(pickle.dumps(value, protocol=4))
    flags = {"status": "pass", "input_track": "track_1", "input_sha256": "b" * 64,
             "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    for key, (relative, stage) in gate.DEPENDENCIES.items():
        item = flags | {"stage": stage}
        if key in {"body", "depth", "object"}: item["frames"] = [{"frame_index": i} for i in range(gate.SOURCE_FRAMES)]
        if key == "object": item["geometry_and_poses_sha256"] = "a" * 64
        if key == "alignment": item["depth_alignment"] = {"shared_scale": .83}
        if key == "adapter": item.update(frames=gate.SOURCE_FRAMES,
            canonical_initializer_sha256=gate.sha256(root / gate.INITIALIZER),
            body_report_sha256=gate.sha256(root / gate.DEPENDENCIES["body"][0]))
        (root / relative).write_text(json.dumps(item))
    prep = flags | {"stage": "world_reward_native_cari_inputs", "frames": gate.SOURCE_FRAMES,
        "original_frame_coverage_verified": True, "producer_revision": "c" * 40, "script_sha256": "d" * 64,
        "object_pose_initializer": "own_ICP_Viterbi_not_FoundationPose",
        "depth_validation": {"validation_mode": "exhaustive", "frame_counts": {camera: gate.SOURCE_FRAMES},
                             "frame_shapes": {camera: list(gate.IMAGE_SIZE)}},
        "export_seq": str(root / gate.EXPORT), "depth_h5": str(root / gate.DEPTH),
        "mhr_init": str(root / gate.INITIALIZER), "object_poses": str(root / gate.POSES),
        "input_report_sha256": {key: gate.sha256(root / p) for key, (p, _) in gate.DEPENDENCIES.items()},
        "file_sha256": {k: gate.sha256(root / p) for k, p in {"depth_h5": gate.DEPTH, "mhr_init": gate.INITIALIZER,
            "object_poses": gate.POSES, "wild_export": gate.EXPORT + "/wild_export.json"}.items()}}
    (root / gate.REPORT).write_text(json.dumps(prep))
    return {"schema": "world-reward-cari96-input-pins-v1", "camera_name": camera,
        "input_report": gate.identity(root / gate.REPORT) | {"producer_revision": "c" * 40, "script_sha256": "d" * 64},
        "source_files": {p: gate.identity(root / p) for p in gate.source_paths(camera)}}


def test_real_committed_pin_config_inventory():
    path = Path(__file__).parents[1] / "infra/cari96_inputs.py"
    spec = importlib.util.spec_from_file_location("cari96_config_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.validate_pins(json.loads((path.parents[1] / "configs/cari96_input_pins.json").read_text()))
    assert module.SOURCE_FRAMES == 501 and module.FRAMES == 96 and len(module.source_paths("front_stereo_camera_left")) == 15


@pytest.mark.parametrize("fault", ["extra", "camera", "hash", "boolbytes", "missing", "reportsha", "producer"])
def test_pin_fail_closed(gate, tmp_path, fault):
    pins = source_fixture(gate, tmp_path)
    if fault == "extra": pins["source_files"]["private/GT.npz"] = {"sha256": "a" * 64, "bytes": 1}
    elif fault == "camera": pins["camera_name"] = "../camera"
    elif fault == "hash": pins["source_files"][gate.REPORT]["sha256"] = "NO"
    elif fault == "boolbytes": pins["source_files"][gate.REPORT]["bytes"] = True
    elif fault == "missing": pins["source_files"].pop(gate.INITIALIZER)
    elif fault == "reportsha": pins["input_report"]["sha256"] = "a" * 64
    else: pins["input_report"]["producer_revision"] = "bad"
    with pytest.raises(ValueError): gate.validate_pins(pins)


def test_full_source_report_and_initializer_remain_unmodified(gate, tmp_path):
    pins = source_fixture(gate, tmp_path); before = copy.deepcopy(pins)
    gate.validate_pins(pins); prep, _ = gate.validate_reports(tmp_path, pins)
    assert "episode_index" not in prep  # Exact pinned legacy receipt, no invented episode field.
    data = initializer(gate); original = pickle.dumps(data)
    gate.validate_initializer(data); gate.validate_poses(poses(gate), wild_metadata(gate, tmp_path))
    gate.validate_wild(wild_metadata(gate, tmp_path), tmp_path)
    assert pins == before and pickle.dumps(data) == original


@pytest.mark.parametrize("fault", ["GT", "private", "count", "duplicate", "video", "path", "episode", "adapterhash"])
def test_reports_strict_exact_full_identity(gate, tmp_path, fault):
    pins = source_fixture(gate, tmp_path)
    key = gate.DEPENDENCIES["body"][0] if fault in {"duplicate", "video"} else gate.REPORT
    path = tmp_path / key; value = json.loads(path.read_text())
    if fault == "GT": value["ground_truth_used"] = True
    elif fault == "private": value["private_truth_read"] = True
    elif fault == "count": value["frames"] = 4
    elif fault == "duplicate": value["frames"][1]["frame_index"] = 0
    elif fault == "video": value["input_sha256"] = "x" * 64
    elif fault == "path": value["mhr_init"] = str(tmp_path / "cari_forward/coconet.pth")
    elif fault == "episode": value["episode_index"] = True
    else: value["input_report_sha256"]["adapter"] = "f" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): gate.validate_reports(tmp_path, pins)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "masked", "frames", "expr", "Z"])
def test_initializer_shape_dtype_no_invention(gate, fault):
    data = initializer(gate)
    if fault == "shape": data["mhr_scale"] = data["mhr_scale"][:, :-1]
    elif fault == "dtype": data["mhr_shape"] = data["mhr_shape"].astype(np.float64)
    elif fault == "nan": data["mhr_shape"][0, 0] = np.nan
    elif fault == "masked": data["mhr_hand"] = np.ma.array(data["mhr_hand"])
    elif fault == "frames": data["frames"][1] = "000000"
    elif fault == "expr": data["mhr_face"][0, 0] = .1
    else: data["mhr_trans"][1, 2] = 0
    with pytest.raises(ValueError): gate.validate_initializer(data)


@pytest.mark.parametrize("fault", ["scale", "reflection", "framechange", "oracle", "frames"])
def test_object_no_scale_no_basis_change(gate, tmp_path, fault):
    data = poses(gate); wild = wild_metadata(gate, tmp_path)
    if fault == "scale": data["obj_pose_world"][0, 0, 0] = 2
    elif fault == "reflection": data["obj_pose_world"][0, 0, 0] = -1
    elif fault == "framechange": wild["source_object_mesh_to_aligned_transform"][0][3] = .1
    elif fault == "oracle": data["metadata"]["oracle_modes"] = ["gt"]
    else: data["frames"].reverse()
    with pytest.raises(ValueError): gate.validate_poses(data, wild)


def test_symlink_and_occupied_output_refused_before_deserialization(gate, tmp_path, monkeypatch):
    pins = source_fixture(gate, tmp_path)
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda *_: pytest.fail("must not deserialize")))
    out = tmp_path / "validation/existing"; out.mkdir(parents=True)
    with pytest.raises(FileExistsError): gate.prepare_snapshot(tmp_path, out, pins)
    original = tmp_path / gate.INITIALIZER
    alias = original.with_suffix(".bak"); original.rename(alias); original.symlink_to(alias)
    with pytest.raises(ValueError): gate.prepare_snapshot(tmp_path, tmp_path / "validation/new", pins)
    assert not (tmp_path / "validation/new").exists()


def make_h5(gate, root, pins):
    h5py = pytest.importorskip("h5py")
    n, h, w = gate.SOURCE_FRAMES, *gate.IMAGE_SIZE
    camera = pins["camera_name"]
    def png(i):
        b = bytearray(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
            + w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x10\x00\x00\x00\x00" + b"fixture")
        return np.frombuffer(bytes(b) + bytes([i]), np.uint8)
    for kind in ("images", "human_masks", "object_masks"):
        path = root / gate.EXPORT / kind / (camera + ".h5")
        with h5py.File(path, "w") as f:
            f.attrs.update(complete=True, sequence="episode_000015", camera_id=0)
            if kind == "images":
                ds = f.create_dataset("frames", (n,), dtype=h5py.vlen_dtype(np.uint8)); ds.attrs["encoding"] = "jpeg"
                for i in range(n): ds[i] = np.array([255, 216, i, 255, 217], np.uint8)
            else:
                ds = f.create_dataset("frames", (n, h, w), dtype=np.uint8, chunks=(1, h, w), compression="lzf", shuffle=True)
                for i in range(n): ds[i] = np.full((h, w), 255 if i % 2 else 0, np.uint8)
    with h5py.File(root / gate.DEPTH, "w") as f:
        f.attrs.update(complete=True, format="cari4d_mhr_metric_depth_png_v1",
            depth_alignment_method="world_reward_shared_predicted_human_scale",
            depth_alignment_input_identity_json=json.dumps({"ground_truth_used": False, "depth_backend": "moge2",
                "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
                "alignment": "one_predicted_human_anchored_clip_scalar_no_offset"}))
        f.create_dataset("frame_names/" + camera, data=np.array([f"{i:06d}" for i in range(n)], object), dtype=h5py.string_dtype())
        for kind in ("raw", "aligned"):
            ds = f.create_dataset(kind + "/" + camera, (n,), dtype=h5py.vlen_dtype(np.uint8))
            for i in range(n): ds[i] = png(i)
        for key, dtype, value in (("scale", np.float32, .83), ("shift", np.float32, 0), ("valid_count", np.int32, h * w)):
            f.create_dataset("alignment/" + camera + "/" + key, data=np.full(n, value, dtype))
    # Rebind fixture prep after the exact stored HDF5 bytes exist.
    prep_path = root / gate.REPORT; prep = json.loads(prep_path.read_text())
    prep["file_sha256"]["depth_h5"] = gate.sha256(root / gate.DEPTH)
    prep_path.write_text(json.dumps(prep))
    pins["source_files"] = {p: gate.identity(root / p) for p in gate.source_paths(camera)}
    pins["input_report"].update(gate.identity(prep_path))
    return h5py


def test_real_h5_subset_reread_no_reencoding_no_initializer_copy(gate, tmp_path, monkeypatch):
    pins = source_fixture(gate, tmp_path); h5py = make_h5(gate, tmp_path, pins)
    def dump(value, path, **_): Path(path).write_bytes(pickle.dumps(value, protocol=4))
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda path: pickle.loads(Path(path).read_bytes()), dump=dump))
    before = copy.deepcopy(pins["source_files"])
    out = tmp_path / "validation/cari96_public_v1"
    out.parent.mkdir()
    result = gate.prepare_snapshot(tmp_path, out, pins)
    assert result["manifest"]["original_frame_indices"] == [0, 1, 2]
    assert result["initializer_source"] == tmp_path / gate.INITIALIZER
    assert result["initializer"]["frames"] == [f"{i:06d}" for i in range(5)]
    assert not any("initializer" in str(p.relative_to(out)) for p in out.rglob("*"))
    assert {p: gate.identity(tmp_path / p) for p in before} == before
    for source, target, dataset in ((tmp_path / gate.DEPTH, result["depth_h5"], "raw/" + pins["camera_name"]),
            (tmp_path / gate.EXPORT / "images" / (pins["camera_name"] + ".h5"),
             result["export_seq"] / "images" / (pins["camera_name"] + ".h5"), "frames")):
        with h5py.File(source) as src, h5py.File(target) as dst:
            assert dst[dataset].shape == (3,) and src[dataset].shape == (5,)
            assert all(src[dataset][i].tobytes() == dst[dataset][i].tobytes() for i in range(3))
    selected = pickle.loads(result["object_poses"].read_bytes())
    assert selected["obj_pose_world"].tobytes() == poses(gate)["obj_pose_world"][:3].tobytes()
    assert all(p.stat().st_mode & 0o222 == 0 for p in out.rglob("*") if p.is_file())


@pytest.mark.parametrize("fault", ["external", "extra", "incomplete", "fullshape", "binary", "dtype", "names", "shift", "scale"])
def test_h5_fail_closed(gate, tmp_path, fault):
    pins = source_fixture(gate, tmp_path); h5py = make_h5(gate, tmp_path, pins)
    depth_fault = fault in {"names", "shift", "scale"}
    source = tmp_path / gate.DEPTH if depth_fault else tmp_path / gate.EXPORT / "human_masks" / (pins["camera_name"] + ".h5")
    with h5py.File(source, "r+") as f:
        if fault == "external": f["other"] = h5py.ExternalLink("private_GT.h5", "/values")
        elif fault == "extra": f.create_dataset("extra", data=0)
        elif fault == "incomplete": f.attrs["complete"] = False
        elif fault in {"fullshape", "dtype"}:
            del f["frames"]; f.create_dataset("frames", (4, 2, 4) if fault == "fullshape" else (5, 2, 4), dtype=np.float32)
        elif fault == "binary": f["frames"][0] = np.ones((2, 4), np.uint8)
        elif fault == "names": f["frame_names/" + pins["camera_name"]][1] = "000000"
        else: f["alignment/" + pins["camera_name"] + "/" + fault][1] = 1
    with pytest.raises((ValueError, KeyError)):
        gate.subset_h5(source, tmp_path / "subset.h5", kind="depth" if depth_fault else "human_masks",
            camera=pins["camera_name"], image_size=gate.IMAGE_SIZE, shared_scale=.83)

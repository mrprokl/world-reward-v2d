"""Tiny direct-export contracts, no models, Torch, media or network."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra")); monkeypatch.syspath_prepend(str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("cari96_export_test", ROOT / "infra/cari96_export.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bundle(gate):
    n = gate.FRAMES
    params = {k: np.zeros((n, d), np.float32) for k, d in gate.inputs.PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    params["mhr_trans"][:, 2] = 3
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (n, 4, 4)).copy(); pose[:, 2, 3] = 3
    params["pose_abs"] = pose
    params["contact_logits"] = np.zeros((n, 2), np.float32)
    metadata = dict(ground_truth_used=False, object_mesh="mesh.glb", materialized_input_cache=None,
        object_pose_frame="centered_axis_aligned", object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
        object_pose_storage_frame="output_aligned_mesh_frame", object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist())
    source = dict(schema="cari4d.mhr_wild_inference.v1", gt={}, frames=[f"{i:06d}" for i in range(n)], pr=params,
        raw={"delta_mhr_shape": np.zeros((n, 45), np.float32)}, metadata=metadata,
        observations={k: np.ones((n, 256, 256) if k.startswith("postopt_") else (n, 2, 2),
            np.float32 if k.startswith("postopt_") else bool) for k in
            ("human_mask", "object_mask", "postopt_human_mask", "postopt_object_mask")},
        faces=np.array([[0, 1, 2]], np.int32), K_rois=np.eye(3, dtype=np.float32)[None],
        bboxes=np.zeros((n, 4), np.float32), frame_meta=[{"frame": x} for x in range(n)])
    source["in"] = {"pose": pose.copy()}
    source["pr_initial"] = copy.deepcopy(params)
    result = copy.deepcopy(source); result["pr"]["mhr_body_pose_cont"][:, 0] = .01
    result["pr"]["pose_abs"][:, 0, 3] = .02
    result["pr"]["pose_abs_postopt"] = result["pr"]["pose_abs"].copy()
    result["postopt"] = dict(mode="smplh_parity", frame_indices=list(range(n)), resolved_batch_size=n,
        batch_sampling="full_clip_v1", optimized_parameters=gate.lineage.contract.OPTIMIZED_PARAMETERS,
        fixed_parameters=gate.lineage.contract.FIXED_PARAMETERS,
        config=dict(num_steps=300, batch_size=0, frame_start=0, frame_limit=0, freeze_object_rotation=True,
                    freeze_body_internal_translations=True), history=[{"iter": 0, "loss": 1.}, {"iter": 300, "loss": .1}],
        final_diagnostics={"loss": .1})
    return source, result


def test_complete_refined_preserves_raw_and_allowed_motion(gate):
    source, result = bundle(gate); before = gate.lineage.fingerprint(result)
    params, poses, metadata = gate.validate_refined(source, result, Path("mesh.glb"))
    assert metadata["effective_optimizer_updates"] == 301
    assert params["mhr_body_pose_cont"][0, 0] == np.float32(.01)
    assert poses[0, 0, 3] == np.float32(.02)
    assert gate.lineage.fingerprint(result) == before
    assert not np.shares_memory(params["mhr_shape"], result["pr"]["mhr_shape"])


@pytest.mark.parametrize("fault", ["identity", "raw", "mask", "camera", "faces", "contact", "pr_contact", "dtype", "expression", "partial"])
def test_no_changed_prediction_or_refinement_recipe(gate, fault):
    source, result = bundle(gate)
    if fault == "identity": result["pr"]["mhr_shape"][1, 0] = .1
    elif fault == "raw": result["raw"]["delta_mhr_shape"][0, 0] = .1
    elif fault == "mask": result["observations"]["human_mask"][0, 0, 0] = False
    elif fault == "camera": result["K_rois"][0, 0, 0] = 2
    elif fault == "faces": result["faces"][0, 0] = 1
    elif fault == "contact":
        source["raw"]["contact_logits"] = np.zeros((96, 2), np.float32)
        result["raw"]["contact_logits"] = np.ones((96, 2), np.float32)
    elif fault == "pr_contact": result["pr"]["contact_logits"][0, 0] = .1
    elif fault == "dtype": result["pr"]["mhr_body_pose_cont"] = result["pr"]["mhr_body_pose_cont"].astype(np.float64)
    elif fault == "expression": result["pr"]["mhr_face"][0, 0] = .1
    else: result["postopt"]["history"][-1]["iter"] = 299
    with pytest.raises(ValueError): gate.validate_refined(source, result, Path("mesh.glb"))


def mesh():
    return np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32), np.array([[0, 1, 2]], np.int64)


def test_aligned_object_camera_formula_no_source_A_no_rescale(gate):
    vertices, faces = mesh(); _, result = bundle(gate)
    poses = result["pr"]["pose_abs"]
    before = vertices.tobytes(), poses.tobytes()
    report = gate.object_roundtrip(vertices, faces, poses)
    assert report["frames"] == 96 and report["additional_frame_transform"] is False
    assert report["object_scale"] == 1 and report["max_point_error_m"] == 0
    assert (vertices.tobytes(), poses.tobytes()) == before
    poses[0, :3, 3] = [0, 0, -1]
    with pytest.raises(ValueError): gate.object_roundtrip(vertices, faces, poses)


def data(gate):
    source, refined = bundle(gate); vertices, faces = mesh()
    params, poses, _ = gate.validate_refined(source, refined, Path("mesh.glb"))
    controls = np.zeros((96, 204), np.float32); controls[:, 2] = 30; controls[:, 136:] = .03
    K = np.array([[1920, 0, 768], [0, 1920, 576], [0, 0, 1]], np.float64)
    return gate.trajectory(controls, params, poses, vertices, faces, K), params


def test_track1_schema_exact96_shared_native_blocks_and_scale1(gate):
    values, params = data(gate)
    assert set(values) == gate.TRAJECTORY_KEYS
    assert values["pose"].shape == (96, 136) and values["scales"].shape == (68,)
    assert values["shape"].tobytes() == params["mhr_shape"][0].tobytes()
    assert values["object_scale"].shape == () and values["object_scale"] == 1
    assert not values["expression"].any() and np.array_equal(values["frame_index"], np.arange(96))
    assert values["object_faces"].dtype == np.int64


def test_no_scale_reselection_or_camera_fit(gate):
    source, refined = bundle(gate); vertices, faces = mesh()
    params, poses, _ = gate.validate_refined(source, refined, Path("mesh.glb"))
    controls = np.zeros((96, 204), np.float32); controls[5, 136] = .1
    K = np.array([[1920, 0, 768], [0, 1920, 576], [0, 0, 1]], np.float64)
    with pytest.raises(ValueError): gate.trajectory(controls, params, poses, vertices, faces, K)
    controls[5, 136] = 0; K[0, 0] = 1900
    with pytest.raises(ValueError): gate.trajectory(controls, params, poses, vertices, faces, K)


def test_complete_pinned_refined_producer_attestation(gate, tmp_path):
    source, refined = bundle(gate); _, _, metadata = gate.validate_refined(source, refined, Path("mesh.glb"))
    path = tmp_path / "mesh.glb"; path.write_bytes(b"only-own-fixture-no-mesh-read")
    report = dict(stage=gate.lineage.STAGE, status="pass", phase="complete", image_id=gate.IMAGE, frames=96,
        input_track="track_1", ground_truth_used=False, private_truth_read=False, hand_labeled_test=False,
        oracle_modes=[], network="none", optimizer_attempts=1, optimizer_returns=1, optimizer_validated=1,
        requested_steps=300, effective_optimizer_updates=301, source_frames=501, original_frame_indices=list(range(96)),
        source_inputs_assets_rehashed=True, saved_bundle_reloaded_verified=True, frozen_raw_inputs_byte_preserved=True,
        object_mesh_unchanged=True, native_refinement_verified=True, ground_truth_read=False, learned_inference_calls=0,
        quality_verified=False, adoption_authorized=False, submission_produced=False, numerical_bit_determinism_claimed=False,
        optimizer_sha256=gate.lineage.contract.OPTIMIZER_SHA256, refinement_assets=gate.lineage.contract.REFINEMENT_ASSETS,
        metadata=metadata, forward_report_sha256="a"*64, prepare_report_sha256="b"*64,
        source_bundle_sha256="c"*64, bundle_sha256="d"*64, object_mesh_sha256=gate.inputs.sha256(path), bundle_bytes=123,
        output_files={"refined.pth": {"sha256": "d"*64, "bytes": 123}})
    pins = {"refined_files": {"refined.pth": {"sha256": "d"*64, "bytes": 123}}}
    gate.require_refined_report(report, metadata, pins, path)
    for key, bad in (("effective_optimizer_updates",300),("optimizer_returns",0),("private_truth_read",True),
                     ("frames",95),("bundle_bytes",124),("ground_truth_read",True)):
        changed = report | {key: bad}
        with pytest.raises(ValueError): gate.require_refined_report(changed, metadata, pins, path)


def test_source_helper_manifest_must_bind_current_unchanged_producer(gate, tmp_path):
    code = tmp_path / "code"; path = code / "infra/cari96_refine.py"; path.parent.mkdir(parents=True); path.write_bytes(b"own")
    path.chmod(0o444); row = gate.lineage.identity(path)
    producer = dict(script_sha256=row["sha256"], source_helpers={"infra/cari96_refine.py":row})
    gate.source_helpers(code,producer)
    with pytest.raises(ValueError): gate.source_helpers(code,producer | {"script_sha256":"0"*64})
    path.chmod(0o644)
    with pytest.raises(ValueError): gate.source_helpers(code,producer)


def test_stage_pins_cover_complete_inventory_and_refuse_symlink(gate,tmp_path):
    directory=tmp_path/"new_refined";directory.mkdir()
    producer=dict(producer_revision="a"*40,script_sha256="b"*64)
    (directory/"report.json").write_text(__import__("json").dumps(producer))
    (directory/"refined.pth").write_bytes(b"not-deserialized-own-tiny")
    for p in directory.iterdir():p.chmod(0o444)
    files={p.name:gate.lineage.identity(p)for p in directory.iterdir()}
    pins=dict(schema="world-reward-cari96-refined-pins-v1",refined=files["report.json"]|producer,refined_files=files)
    actual,observed=gate.lineage.pinned_report(tmp_path,"new_refined",pins,"refined")
    assert actual==producer and len(observed)==2
    (directory/"extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError):gate.lineage.pinned_report(tmp_path,"new_refined",pins,"refined")
    (directory/"extra.bin").unlink();(directory/"alias").symlink_to(directory/"refined.pth")
    with pytest.raises(ValueError):gate.lineage.pinned_report(tmp_path,"new_refined",pins,"refined")


def test_freeze_rereads_exact7_nativeblocks_and_memmapped_geometry(gate, tmp_path):
    values, params = data(gate)
    target = np.ones((96, 2, 3), np.float32); joints = np.ones((96, 3, 3), np.float32)
    kp = np.ones((96, 4, 3), np.float32); faces = np.array([[0, 1, 0]], np.int64)
    frozen = gate.freeze(tmp_path, values, params, target, joints, kp, faces)
    assert set(frozen) == {"trajectory.npz", "native_parameters.npz", "target.npy"}
    saved = np.load(tmp_path / "target.npy", mmap_mode="r", allow_pickle=False)
    assert isinstance(saved, np.memmap) and gate.same_arrays({"v": target}, {"v": saved})
    with np.load(tmp_path / "native_parameters.npz", allow_pickle=False) as arrays:
        assert set(params).issubset(arrays.files)
        assert arrays["human_faces"].dtype == np.int64
        assert gate.same_arrays(params, {k: arrays[k] for k in params})
    assert all(not (tmp_path / name).stat().st_mode & 0o222 for name in frozen)
    with pytest.raises(FileExistsError): gate.freeze(tmp_path, values, params, target, joints, kp, faces)


def test_masked_dtype_or_roundtrip_changes_rejected(gate):
    value = np.array([1.], np.float32)
    assert not gate.same_arrays({"x": value}, {"x": np.ma.array(value)})
    assert not gate.same_arrays({"x": value}, {"x": value.astype(np.float64)})
    with pytest.raises(ValueError): gate.array(np.ma.array(value), (1,), np.float32)


def test_attempt_return_validated_accounting_failure_is_not_success(gate):
    report = dict(native_geometry_attempts=0, native_geometry_returns=0, native_geometry_validated=0)
    with pytest.raises(RuntimeError): gate.call(report, "native_geometry", lambda: (_ for _ in ()).throw(RuntimeError()), lambda: None)
    assert report == dict(native_geometry_attempts=1, native_geometry_returns=0, native_geometry_validated=0)
    assert gate.call(report, "native_geometry", lambda: 7, lambda: None) == 7
    assert report["native_geometry_attempts"] == 2 and report["native_geometry_returns"] == 1
    gate.validated(report, "native_geometry", lambda: None)
    assert report["native_geometry_validated"] == 1


def test_wrapper_strict_offline_no_old_output_gt_or_model_fit(gate):
    path = ROOT / "infra/run_cari96_export.sh"; text = path.read_text()
    subprocess.run(["rtk", "proxy", "bash", "-n", str(path)], check=True)
    assert subprocess.run(["rtk", "proxy", "bash", str(path), "--fit"], capture_output=True).returncode == 2
    assert text.count("docker run") == 1 and "183s" in text
    assert "--gpus all --network none --memory 32g --cpus 4" in text
    assert "eval_private" not in text and "outputs/episode" not in text
    assert "src=$path,dst=$path,readonly" in text and "src=$OUT,dst=$OUT" in text
    assert "configs/cari96_${name}_pins.json" in text and "refined; do" in text


def test_runtime_reads_saved7blocks_and_all_V_J_KP_before_official(gate):
    tree = ast.parse(Path(gate.__file__).read_text()); text = ast.unparse(tree)
    assert not any(isinstance(n, ast.Import) and any(x.name == "torch" for x in n.names) for n in tree.body)
    assert '"native_replay"' not in text or "layer.mhr_forward" in text
    assert "stored_native.items()" in text and "native_artifact['mhr_joints']" in text and "native_artifact['mhr_keypoints']" in text
    assert "precision='float32'" in text and "reference.dtype != torch.float64" in text
    assert "global_trans=trans * context.flip" in text and "return_model_params=True" in text
    assert "convert(" not in text and "lm_pose(" not in text and "process_one_image(" not in text
    assert "same_arrays" in text and "lineage.fingerprint" in text and "source_helpers(code, refined_report)" in text

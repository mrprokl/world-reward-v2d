"""Synthetic sparse H1 contracts only: no network, assets, torch, GPU or GT."""

import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from world_reward.hand_transfer import transfer_finger_controls


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("world_reward_test_finger_smoke", ROOT / "infra/finger_transfer_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reports(smoke):
    common = {
        "status": "pass", "episode_index": 15, "input_track": "track_1", "input_sha256": "a" * 64,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "network": "none",
        "upstream_revision": smoke.UPSTREAM_REVISION, "geometry_units": "metres",
        "inference_source_identity": {"python_files": 100, "sha256": "b" * 64},
        "body_assets": {"model.ckpt": {"bytes": 5, "sha256": "c" * 64}},
    }
    final = dict(copy.deepcopy(common), stage=smoke.FINAL_STAGE, frames=5,
                 original_frame_coverage_verified=True, human_shared_identity_verified=True,
                 facial_expressions_zero=True, human_parameter_format="mhr_model_params_136_68_45",
                 geometry_frame="camera_x_right_y_down_z_forward", reference_model_sha256=smoke.REFERENCE_MODEL_SHA256,
                 official_converter_sha256=smoke.CONVERTER_SHA256, checkpoint_sha256=smoke.CHECKPOINT_SHA256)
    hands = dict(copy.deepcopy(common), stage=smoke.HANDS_STAGE, total_video_frames=5, frame_indices=[0, 2, 4],
                 inference_type="full", hand_decoder_proposals=True, mhr_geometry_forward_verified=True,
                 mhr_geometry_forward_basis="original_body133_hand108_global_rot3_scale28_shape45_expr72_not_raw_logits",
                 joint_global_rotation_source="fresh_native_parameter_block_decode",
                 raw_pose_logits_role="audit_only_zeroed_after_full_hand_fusion",
                 prompt_mode="automatic_mask_and_derived_bbox_no_fallback", human_mask_id=0,
                 geometry_frame="SAM3D_camera_x_right_y_down_z_forward", vertices=18439, faces=36874,
                 frames=[{"frame_index": index, "native_forward_max_error_m": 1e-6,
                          "native_joint_forward_max_error_m": 1e-6, "native_controls_forward_max_error": 1e-6}
                         for index in [0, 2, 4]])
    return final, hands


def test_matching_producer_provenance_and_legacy_final_episode(smoke, reports):
    final, hands = reports
    before = copy.deepcopy(reports)
    count, indices = smoke.validate_producer_reports(final, hands, 15)
    assert count == 5
    np.testing.assert_array_equal(indices, [0, 2, 4])
    assert reports == before
    del final["episode_index"]
    smoke.validate_producer_reports(final, hands, 15)


@pytest.mark.parametrize("record,key,value", [
    (0, "stage", "native_cari_checkpoint_load_gate"), (0, "frames", True), (0, "frames", 2),
    (0, "human_shared_identity_verified", False), (0, "facial_expressions_zero", 1),
    (0, "reference_model_sha256", "0" * 64), (0, "official_converter_sha256", "0" * 64),
    (0, "checkpoint_sha256", "0" * 64), (0, "input_sha256", "0" * 64),
    (1, "episode_index", 0), (1, "inference_type", "body"), (1, "human_mask_id", True),
    (1, "total_video_frames", 6), (1, "frame_indices", [4, 2, 0]),
    (1, "hand_decoder_proposals", False), (1, "mhr_geometry_forward_verified", False),
    (1, "joint_global_rotation_source", "cached_original_body_prediction"),
    (1, "raw_pose_logits_role", "decoder_input"), (1, "prompt_mode", "manual_points"),
    (1, "body_assets", {}), (1, "inference_source_identity", {"sha256": "0" * 64}),
    (1, "geometry_frame", "world"), (1, "frames", []),
])
def test_wrong_native_or_hand_contract_fails_closed(smoke, reports, record, key, value):
    reports[record][key] = value
    with pytest.raises(ValueError):
        smoke.validate_producer_reports(*reports, 15)


@pytest.mark.parametrize("record", [0, 1])
@pytest.mark.parametrize("key,value", [("ground_truth_used", True), ("ground_truth_used", 0),
                                       ("hand_labeled_test", None), ("oracle_modes", ["gt_camera"]),
                                       ("input_track", "track_2"), ("network", "default")])
def test_explicit_no_oracle_labels_or_network_required(smoke, reports, record, key, value):
    reports[record][key] = value
    with pytest.raises(ValueError):
        smoke.validate_producer_reports(*reports, 15)


@pytest.mark.parametrize("bad", [None, True, -1e-6, 1.00001e-5, float("nan")])
def test_source_controls_must_have_verified_block_forward(smoke, reports, bad):
    reports[1]["frames"][1]["native_controls_forward_max_error"] = bad
    with pytest.raises(ValueError, match="parameter-block"):
        smoke.validate_producer_reports(*reports, 15)


@pytest.fixture
def final_arrays():
    return {
        "pose": np.arange(5 * 136, dtype=np.float32).reshape(5, 136) / 1000,
        "scales": np.zeros(68, dtype=np.float32), "shape": np.zeros(45, dtype=np.float32),
        "expression": np.zeros(72, dtype=np.float32), "frame_index": np.arange(5),
        "object_rotation": np.tile(np.eye(3), (5, 1, 1)), "object_translation": np.zeros((5, 3)),
        "object_scale": np.asarray(1.),
        "object_vertices": np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        "object_faces": np.array([[0, 1, 2], [0, 0, 0]]),
    }


def test_shared_identity_full_frames_and_padding_retained(smoke, final_arrays):
    before = {name: value.copy() for name, value in final_arrays.items()}
    result = smoke.validate_final_arrays(final_arrays, 5)
    assert result["pose"].dtype == np.float32
    assert result["shape"].shape == (45,) and result["scales"].shape == (68,)
    for name in final_arrays:
        assert final_arrays[name].tobytes() == before[name].tobytes()


@pytest.mark.parametrize("name,kind", [("pose", "nan"), ("pose", "integer"), ("pose", "masked"),
                                     ("scales", "perframe"), ("shape", "perframe"),
                                     ("expression", "nonzero"), ("object_scale", "nonzero"),
                                     ("frame_index", "reorder"), ("object_rotation", "reflection"),
                                     ("object_faces", "overflow"), ("object_vertices", "masked")])
def test_final_arrays_not_silently_repaired(smoke, final_arrays, name, kind):
    value = final_arrays[name]
    if kind == "nan": value.flat[0] = np.nan
    elif kind == "integer": final_arrays[name] = value.astype(np.int64)
    elif kind == "masked": final_arrays[name] = np.ma.array(value, mask=False)
    elif kind == "perframe": final_arrays[name] = np.tile(value, (5, 1))
    elif kind == "nonzero": value.flat[0] = 2
    elif kind == "reorder": final_arrays[name] = value[::-1]
    elif kind == "reflection": value[0, 0, 0] = -1
    elif kind == "overflow": value[0, 0] = 3
    with pytest.raises(ValueError): smoke.validate_final_arrays(final_arrays, 5)


@pytest.fixture
def source_arrays():
    return {"mhr_model_params": np.arange(3 * 204, dtype=np.float32).reshape(3, 204) / 1000,
            "frame_index": np.array([0, 2, 4]), "expr_params": np.zeros((3, 72), dtype=np.float32),
            "pred_pose_raw": np.zeros((3, 266), dtype=np.float32),
            "pred_global_rots": np.tile(np.eye(3), (3, 127, 1, 1)),
            "faces": np.tile(np.array([[0, 1, 2]]), (36874, 1))}


def test_only_decoded204_used_and_sparse_proposal_keeps_baseline(smoke, final_arrays, source_arrays):
    controls, faces = smoke.validate_source_arrays(source_arrays, np.array([0, 2, 4]))
    before = final_arrays["pose"].copy()
    proposal = transfer_finger_controls(before, controls, np.array([0, 2, 4]), np.arange(95, 122), np.arange(68, 95))
    assert faces.shape == (36874, 3)
    np.testing.assert_array_equal(proposal.pose[[0, 2, 4], 68:122], controls[:, 68:122])
    assert proposal.pose[[1, 3]].tobytes() == before[[1, 3]].tobytes()
    fixed = np.r_[np.arange(68), np.arange(122, 136)]
    assert proposal.pose[:, fixed].tobytes() == before[:, fixed].tobytes()
    assert final_arrays["pose"].tobytes() == before.tobytes()
    assert proposal.report["adoption_performed"] is False


@pytest.mark.parametrize("failure", ["hand108", "nan", "raw_nonzero", "expr_nonzero", "stale_rotation",
                                    "reordered", "duplicate", "float_frames", "wrong_topology", "invalid_faces"])
def test_source_native_contract_failures(smoke, source_arrays, failure):
    if failure == "hand108": source_arrays["mhr_model_params"] = np.zeros((3, 108))
    elif failure == "nan": source_arrays["mhr_model_params"][0, 68] = np.nan
    elif failure == "raw_nonzero": source_arrays["pred_pose_raw"][0, 0] = 1
    elif failure == "expr_nonzero": source_arrays["expr_params"][0, 0] = 1
    elif failure == "stale_rotation": source_arrays["pred_global_rots"][0, 0] = 0
    elif failure == "reordered": source_arrays["frame_index"] = np.array([4, 2, 0])
    elif failure == "duplicate": source_arrays["frame_index"] = np.array([0, 0, 4])
    elif failure == "float_frames": source_arrays["frame_index"] = np.array([0., 2., 4.])
    elif failure == "wrong_topology": source_arrays["faces"] = source_arrays["faces"][:-1]
    elif failure == "invalid_faces": source_arrays["faces"][0, 0] = 18439
    with pytest.raises(ValueError): smoke.validate_source_arrays(source_arrays, np.array([0, 2, 4]))


class FakeTensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class FakeTorch:
    @staticmethod
    def is_tensor(value): return isinstance(value, FakeTensor)


def test_actual_checkpoint_absolute_indices_not_vector_position_or_guessed_order(smoke):
    state = {"head_pose.hand_joint_idxs_left": FakeTensor(np.arange(95, 122)[::-1]),
             "head_pose.hand_joint_idxs_right": FakeTensor(np.arange(68, 95)[::-1])}
    left, right = smoke.checkpoint_hand_indices(state, FakeTorch)
    np.testing.assert_array_equal(left, np.arange(95, 122)[::-1])
    assert np.array_equal(np.sort(np.r_[left, right]), np.arange(68, 122))
    left[:] = 0
    assert state["head_pose.hand_joint_idxs_left"].value.min() == 95
    for bad in (None, np.arange(27), FakeTensor(np.arange(68, 95, dtype=float)), FakeTensor(np.arange(27))):
        state["head_pose.hand_joint_idxs_left"] = bad
        with pytest.raises(ValueError): smoke.checkpoint_hand_indices(state, FakeTorch)


def test_reference_mm_to_metres_only_and_no_joint_partition_claim(smoke):
    vertices_mm = np.zeros((3, 18439, 3), dtype=np.float64)
    vertices_mm[:, :3] = [[10., -20., 2000.], [1010., -20., 2000.], [10., 980., 2000.]]
    joints_m = np.ones((3, 127, 3)) * [1., -2., 3.]
    vertices, joints = smoke.reference_arrays(vertices_mm, joints_m, 3)
    np.testing.assert_array_equal(vertices[:, 0], np.tile([.01, -.02, 2.], (3, 1)))
    assert joints.tobytes() == joints_m.tobytes()
    candidate = vertices.copy()
    candidate[:, 1, 0] += .002
    diagnostics = smoke.geometry_diagnostics(vertices, candidate, joints, joints + .001, np.array([[0, 1, 2]]))
    np.testing.assert_allclose(diagnostics["per_frame_max_vertex_displacement_mm"], 2.)
    assert diagnostics["finite_reference_geometry_verified"] is True
    assert diagnostics["nonhand_joint_invariance_verified"] is False
    assert "not_accuracy" in diagnostics["displacement_scope"]
    json.dumps(diagnostics)
    with pytest.raises(ValueError, match="collapsed"):
        smoke.geometry_diagnostics(vertices, np.zeros_like(vertices), joints, joints, np.array([[0, 1, 2]]))
    vertices_mm[0, 0, 0] = np.inf
    with pytest.raises(ValueError): smoke.reference_arrays(vertices_mm, joints_m, 3)


def test_hash_gate_rejects_changed_missing_and_symlink_artifacts(smoke, tmp_path):
    artifact = tmp_path / "tiny.npz"
    artifact.write_bytes(b"synthetic")
    digest = smoke.sha256(artifact)
    smoke._require_hash(artifact, digest)
    link = tmp_path / "link.npz"
    link.symlink_to(artifact)
    for path, expected in ((link, digest), (tmp_path / "missing", digest), (artifact, None), (artifact, "0" * 64)):
        with pytest.raises(RuntimeError): smoke._require_hash(path, expected)
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError): smoke._require_hash(artifact, digest)


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_selected_episode_gate_precedes_heavy_import(smoke, monkeypatch, episode):
    assert smoke._argument_parser().parse_args([]).episode == 15
    assert smoke._argument_parser().parse_args(["--episode", str(episode)]).episode == episode
    monkeypatch.setenv("WR_ROOT", "/srv/synthetic-root")
    monkeypatch.setattr(smoke.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["finger_transfer_smoke.py", "--episode", str(episode)])
    received = []
    def stop_at_report(path):
        received.append(path)
        raise RuntimeError("synthetic selected-report gate")
    monkeypatch.setattr(smoke, "sha256", stop_at_report)
    original_import = builtins.__import__
    def no_heavy(package, *args, **kwargs):
        if package.split(".")[0] in {"torch", "cv2", "trimesh", "h5py"}:
            raise AssertionError("Heavy import preceded selected-report gate")
        return original_import(package, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_heavy)
    with pytest.raises(RuntimeError, match="selected-report"):
        smoke.main()
    assert received == [Path(f"/srv/synthetic-root/outputs/episode_{episode:06d}/cari_conversion/report.json")]


@pytest.mark.parametrize("episode", ["-1", "30", "true", "15.0", "../track_3"])
def test_invalid_episode_fails_cli_before_model(smoke, episode):
    with pytest.raises(SystemExit): smoke._argument_parser().parse_args(["--episode", episode])


@pytest.fixture
def fake_wrapper(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "docker.log"
    for name, content in {"docker": 'printf "%s\\0" "$@" > "$FAKE_LOG"', "id": 'echo 123',
                          "systemctl": 'echo "Unexpected implicit dependency wait" >&2; exit 9'}.items():
        path = binary / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + content + "\n")
        path.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"], WR_ROOT=str(tmp_path / "remote"),
                       WR_CODE=str(ROOT), WR_CODE_REVISION="a" * 40, FAKE_LOG=str(log))
    def report(episode, stage):
        path = Path(environment["WR_ROOT"]) / f"outputs/episode_{episode:06d}/{stage}/report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    def run(*arguments):
        return subprocess.run(["bash", str(ROOT / "infra/run_finger_transfer_smoke.sh"), *arguments],
                              env=environment, capture_output=True, text=True, timeout=3)
    return run, report, log


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_wrapper_requires_both_selected_producers_and_no_data_mount(fake_wrapper, episode):
    run, report, log = fake_wrapper
    report(episode, "cari_conversion")
    args = [] if episode == 15 else ["--episode", str(episode)]
    assert run(*args).returncode != 0 and not log.exists()
    report(episode, "body_hands_smoke")
    result = run(*args)
    assert result.returncode == 0, result.stderr
    arguments = log.read_bytes().decode().strip("\0").split("\0")
    assert arguments[-3:] == [str(ROOT / "infra/finger_transfer_smoke.py"), "--episode", str(episode)]
    assert arguments[arguments.index("--network") + 1] == "none"
    assert arguments[arguments.index("--gpus") + 1] == "all"
    for directory in ("vendor", "weights", "results"):
        assert any(f"/{directory},dst=" in argument and argument.endswith(",readonly") for argument in arguments)
    assert not any("/data," in argument or "/cache," in argument for argument in arguments)


def test_wrapper_never_substitutes_episode15_or_accepts_extra_modes(fake_wrapper):
    run, report, log = fake_wrapper
    for stage in ("cari_conversion", "body_hands_smoke"): report(15, stage)
    assert run("--episode", "0").returncode != 0 and not log.exists()
    for arguments in (["--episode", "30"], ["--episode", "0", "--episode", "15"], ["--kernel-only"], ["--full-video"]):
        assert run(*arguments).returncode != 0 and not log.exists()

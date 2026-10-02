"""Tiny synthetic contracts only; no assets, networks, torch, or GPU inference."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def converter():
    path = Path(__file__).resolve().parents[1] / "infra/cari_converter.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(converter):
    count = 3
    params = {name: np.zeros((count, dimension), dtype=np.float32)
              for name, dimension in converter.PARAMETER_DIMS.items()}
    rotation = Rotation.from_euler("xyz", [[0, 0, 0], [.1, -.2, .8], [.2, .4, -.5]]).as_matrix()
    params["mhr_global_rot6d"] = rotation[:, :, :2].reshape(count, 6).astype(np.float32)
    params["mhr_shape"][:, 0] = [1, 2, 3]  # Native identity is not averaged here.
    params["pose_abs"] = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    params["pose_abs"][:, :3, :3] = rotation
    params["pose_abs"][:, :3, 3] = [[0, 0, 3], [.2, .1, 3.1], [.5, -.2, 3.3]]
    return {"gt": {}, "metadata": {"ground_truth_used": False}, "frames": ["000000", "000001", "000002"], "pr": params}


def test_native_exact_dimensions_full_timeline_and_no_mutation(converter, bundle):
    before = copy.deepcopy(bundle)
    params, pose = converter.validate_native_bundle(bundle, 3)
    assert set(params) == set(converter.PARAMETER_DIMS)
    assert params["mhr_body_pose_cont"].shape == (3, 260)
    assert params["mhr_scale"].shape == (3, 28)  # Not official scales68.
    assert np.array_equal(params["mhr_shape"][:, 0], [1, 2, 3])
    for name in bundle["pr"]:
        assert np.array_equal(bundle["pr"][name], before["pr"][name])
    params["mhr_shape"][:] = 0
    pose[:] = 0
    assert np.array_equal(bundle["pr"]["mhr_shape"][:, 0], [1, 2, 3])
    assert bundle["pr"]["pose_abs"][0, 3, 3] == 1


@pytest.mark.parametrize("count", [0, -1, True, 3., "3", None, np.bool_(True)])
def test_native_invalid_frame_counts(converter, bundle, count):
    with pytest.raises(ValueError, match="positive integer"):
        converter.validate_native_bundle(bundle, count)


@pytest.mark.parametrize("frames", [["000000", "000002"], ["000000", "000001", "000001"],
                                    ["000002", "000001", "000000"], [0, 1, 2]])
def test_no_sliced_reordered_or_repeated_native_frames(converter, bundle, frames):
    bundle["frames"] = frames
    with pytest.raises(ValueError, match="all original frames"):
        converter.validate_native_bundle(bundle, 3)


@pytest.mark.parametrize("gt, flag", [(None, False), ({"secret": 1}, False), ({}, None), ({}, 0), ({}, True)])
def test_explicit_no_gt_is_mandatory(converter, bundle, gt, flag):
    bundle["gt"] = gt
    bundle["metadata"]["ground_truth_used"] = flag
    with pytest.raises(ValueError, match="no ground truth"):
        converter.validate_native_bundle(bundle, 3)


@pytest.mark.parametrize("name", ["mhr_trans", "mhr_body_pose_cont", "mhr_face", "mhr_shape"])
@pytest.mark.parametrize("failure", ["missing", "shape", "nan", "integer", "masked"])
def test_native_parameter_blocks_fail_closed(converter, bundle, name, failure):
    value = bundle["pr"][name]
    if failure == "missing": del bundle["pr"][name]
    elif failure == "shape": bundle["pr"][name] = value[:-1]
    elif failure == "nan": value[0, 0] = np.nan
    elif failure == "integer": bundle["pr"][name] = value.astype(np.int64)
    elif failure == "masked": bundle["pr"][name] = np.ma.array(value, mask=False)
    with pytest.raises(ValueError):
        converter.validate_native_bundle(bundle, 3)


@pytest.mark.parametrize("bad", [[0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0]])
def test_degenerate_or_wrong_concatenated_root_layout_fails(converter, bundle, bad):
    bundle["pr"]["mhr_global_rot6d"][0] = bad
    with pytest.raises(ValueError, match="degenerate"):
        converter.validate_native_bundle(bundle, 3)


@pytest.mark.parametrize("failure", ["reflection", "scale", "shear", "perspective"])
def test_native_object_pose_not_repaired(converter, bundle, failure):
    pose = bundle["pr"]["pose_abs"][0]
    if failure == "reflection": pose[0, 0] = -1
    elif failure == "scale": pose[0, 0] = 1.01
    elif failure == "shear": pose[0, 1] = .01
    elif failure == "perspective": pose[3, 0] = .01
    with pytest.raises(ValueError):
        converter.validate_native_bundle(bundle, 3)


@pytest.fixture
def object_arrays(bundle):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.4, -.3, .7]).as_matrix()
    transform[:3, 3] = [.17, -.08, .13]
    vertices = np.array([[0, 0, 0], [.2, 0, 0], [0, .1, 0], [0, 0, .05], [0, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 0, 0]], dtype=np.int64)
    original = bundle["pr"]["pose_abs"].copy()
    return original, original @ np.linalg.inv(transform), transform, vertices, faces


def test_object_frame_inverse_relation_all_vertices_and_no_second_scale(converter, object_arrays):
    original, aligned, transform, vertices, faces = object_arrays
    before = [array.copy() for array in (aligned, transform, vertices, faces)]
    restored, report = converter.restore_object_source_frame(aligned, transform, vertices, faces)
    np.testing.assert_allclose(restored, original, atol=1e-14, rtol=0)
    assert report["roundtrip_max_error_m"] < 1e-14
    assert report["object_scale"] == 1
    assert report["additional_scale_applied"] is False
    assert report["frames_verified"] == 3
    assert report["vertices_verified_per_frame"] == len(vertices)
    assert not np.allclose(aligned @ np.linalg.inv(transform), original)  # Wrong inverse is detectable.
    for value, old in zip((aligned, transform, vertices, faces), before):
        assert np.array_equal(value, old)
    json.dumps(report)


@pytest.mark.parametrize("failure", ["scaled_A", "reflection_A", "A_shape", "nan_vertex", "integer_vertices",
                                    "float_faces", "negative_index", "index_overflow", "zero_mesh", "face_budget"])
def test_object_contract_failures(converter, object_arrays, failure):
    _, aligned, transform, vertices, faces = object_arrays
    if failure == "scaled_A": transform[:3, :3] *= 1.1
    elif failure == "reflection_A": transform[:3, 0] *= -1
    elif failure == "A_shape": transform = transform[:3]
    elif failure == "nan_vertex": vertices[0, 0] = np.nan
    elif failure == "integer_vertices": vertices = vertices.astype(np.int64)
    elif failure == "float_faces": faces = faces.astype(np.float32)
    elif failure == "negative_index": faces[0, 0] = -1
    elif failure == "index_overflow": faces[0, 0] = len(vertices)
    elif failure == "zero_mesh": vertices[:] = 0
    elif failure == "face_budget": faces = np.zeros((4097, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        converter.restore_object_source_frame(aligned, transform, vertices, faces)


def test_mean_per_frame_gate_not_pointwise_or_global_average(converter):
    target = np.zeros((2, 5, 3), dtype=np.float32)
    recovered = target.copy()
    recovered[0, 0, 0] = .005  # One 5 mm point, but 1 mm frame mean.
    recovered[1, :, 0] = .002
    errors, maximum = converter.vertex_residual_mm(recovered, target)
    np.testing.assert_allclose(errors, [1, 2], atol=1e-6)
    assert maximum == pytest.approx(5)
    # Float64 exact boundary, separate from float32 2 mm representation noise.
    report = converter.require_fidelity(np.array([1., 2.]))
    assert report["worst_frame_mean_mm"] == 2
    assert "not_heldout_accuracy" in report["gate_scope"]
    with pytest.raises(ValueError, match="fidelity"):
        converter.require_fidelity(np.array([0., 2.00001]))
    with pytest.raises(ValueError, match="fidelity"):
        converter.require_fidelity(np.array([0., 3.]))  # Global average1.5 does not pass.
    with pytest.raises(ValueError, match="fidelity"):
        converter.require_fidelity(np.array([2000.]))  # Metres/mm mistake.


@pytest.mark.parametrize("errors", [np.array([]), np.array([-1.]), np.array([np.nan]), np.ones((1, 1)), np.array([True])])
def test_invalid_fidelity_arrays(converter, errors):
    with pytest.raises(ValueError):
        converter.require_fidelity(errors)


def test_actual_full_forward_report_not_checkpoint_only(converter):
    report = {"stage": "world_reward_native_cari_full_forward", "status": "pass", "input_track": "track_1",
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
              "checkpoint_sha256": converter.CHECKPOINT_SHA256, "episode_inputs_used": True,
              "metadata": {"actual_network_forward_verified": True, "full_original_frame_coverage_verified": True}}
    converter.require_full_forward_report(report)
    for key, bad in (("stage", "native_cari_checkpoint_load_gate"), ("checkpoint_sha256", "0" * 64),
                     ("episode_inputs_used", False), ("ground_truth_used", 0), ("hand_labeled_test", None)):
        broken = copy.deepcopy(report)
        broken[key] = bad
        with pytest.raises(ValueError): converter.require_full_forward_report(broken)
    for key in report["metadata"]:
        broken = copy.deepcopy(report)
        broken["metadata"][key] = False
        with pytest.raises(ValueError): converter.require_full_forward_report(broken)


def test_no_hidden_training_mesh_gauge_before_source_frame_restoration(converter):
    metadata = {"object_pose_frame": "centered_axis_aligned",
                "object_pose_frame_revision": "cari4d.object_pose_frame.centered_axis_aligned.v1",
                "object_pose_storage_frame": "output_aligned_mesh_frame",
                "object_pose_storage_to_training_transform": np.eye(4).tolist(),
                "object_mesh_to_training_transform": np.eye(4).tolist()}
    converter.require_aligned_object_metadata(metadata)
    for name in ("object_pose_storage_to_training_transform", "object_mesh_to_training_transform"):
        broken = copy.deepcopy(metadata)
        broken[name][0][3] = .01
        with pytest.raises(ValueError, match="additional"):
            converter.require_aligned_object_metadata(broken)
    broken = copy.deepcopy(metadata)
    broken["object_pose_storage_frame"] = "source_mesh_frame"
    with pytest.raises(ValueError, match="aligned-mesh"):
        converter.require_aligned_object_metadata(broken)


def test_wrapper_enforces_offline_readonly_and_full_forward_wait():
    root = Path(__file__).resolve().parents[1]
    text = (root / "infra/run_cari_converter.sh").read_text()
    assert 'source "$CODE/infra/cari_wrapper_common.sh"' in text
    assert "wr_cari_dependency converter" in text
    helper = (root / "infra/cari_wrapper_common.sh").read_text()
    assert "converter) WR_WAIT_FOR=world-reward-cari-forward.service" in helper
    assert "--property=ActiveState --value" in helper
    assert "--network none" in text and "--gpus all" in text
    for directory in ("vendor", "weights", "results"):
        assert f'src=$ROOT/{directory},dst=$ROOT/{directory},readonly' in text
    assert 'src=$CODE,dst=$CODE,readonly' in text
    assert "converter) stage=cari_forward" in helper
    assert 'episode_$WR_EPISODE_PADDED/$stage/report.json' in helper
    assert "--kernel-only" not in text and "--full-video" not in text

"""Tiny synthetic regression gates; no Torch, model, video, network or GT."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[1] / "infra/cari_body_adapter.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_body_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_root(euler, camera_to_world):
    np.testing.assert_array_equal(camera_to_world, np.eye(4))
    return Rotation.from_euler("ZYX", euler).as_matrix()[..., :, :2].reshape(-1, 6).astype(np.float32)


def encode_body(body):
    # A spy encoder: meaningful sentinels catch raw133 vs pred_pose_raw204/260
    # confusion without reimplementing the pinned, independently audited code.
    result = np.zeros((len(body), 260), dtype=np.float32)
    result[:, :133] = body
    result[:, 133:] = 0.125
    return result


@pytest.fixture
def example(adapter, monkeypatch):
    count, vertices = 5, 7
    rng = np.random.default_rng(42)
    raw = {key: rng.normal(0, .05, (count, dim)).astype(np.float32) for key, dim in adapter._RAW_DIMS.items()}
    raw["expr_params"][:] = 0
    raw["pred_cam_t"] = np.tile(np.array([.02, -.03, 3.], dtype=np.float32), (count, 1))
    raw["pred_cam_t"][:, 0] += np.arange(count) * .015
    raw["frame_index"] = np.arange(count, dtype=np.int64)
    raw["vertices_root_camera_m"] = rng.normal(0, .1, (count, vertices, 3)).astype(np.float32)
    raw["vertices_camera_m"] = raw["vertices_root_camera_m"] + raw["pred_cam_t"][:, None]
    raw["pred_joint_coords"] = rng.normal(0, .1, (count, 4, 3)).astype(np.float32)
    raw["pred_keypoints_3d"] = rng.normal(0, .1, (count, 70, 3)).astype(np.float32)
    raw["faces"] = np.array([[0, 1, 2], [2, 3, 4], [3, 5, 6]], dtype=np.int64)
    raw["focal_length"] = np.full(count, 100., dtype=np.float32)
    K = np.array([[100., 0, 40], [0, 100., 30], [0, 0, 1]], dtype=np.float32)
    xyz = raw["pred_keypoints_3d"] + raw["pred_cam_t"][:, None]
    raw["pred_keypoints_2d"] = (xyz[..., :2] * 100. / xyz[..., 2:] + [40, 30]).astype(np.float32)
    calls = []

    class Layer:
        def mhr_forward(self, parameters):
            return decode(parameters)

    layer = Layer()

    def decode(parameters):
        start = sum(len(previous["mhr_trans"]) for previous in calls)
        stop = start + len(parameters["mhr_trans"])
        for name, original in (("mhr_hand", "hand_pose_params"), ("mhr_shape", "shape_params"), ("mhr_scale", "scale_params"), ("mhr_face", "expr_params"), ("mhr_trans", "pred_cam_t")):
            np.testing.assert_array_equal(parameters[name], raw[original][start:stop])
        np.testing.assert_array_equal(parameters["mhr_global_rot6d"], encode_root(raw["global_rot"][start:stop], np.eye(4)))
        np.testing.assert_array_equal(parameters["mhr_body_pose_cont"], encode_body(raw["body_pose_params"][start:stop]))
        calls.append({key: value.copy() for key, value in parameters.items()})
        return SimpleNamespace(
            vertices=raw["vertices_camera_m"][start:stop].copy(),
            joints=(raw["pred_joint_coords"] + raw["pred_cam_t"][:, None])[start:stop].copy(),
            keypoints=(raw["pred_keypoints_3d"] + raw["pred_cam_t"][:, None])[start:stop].copy(),
            faces=raw["faces"].copy(),
        )

    monkeypatch.setattr(adapter, "_native_helpers", lambda: (encode_body, encode_root, lambda: {"mhr_init_root_revision": "fixture", "mhr_init_translation_revision": "fixture"}))
    monkeypatch.setattr(adapter, "_native_decode", lambda native_layer, parameters: native_layer.mhr_forward(parameters))
    return SimpleNamespace(raw=raw, K=K, layer=layer, calls=calls)


def run(adapter, example, **kwargs):
    arguments = dict(total_frames=len(example.raw["frame_index"]), mhr_layer=example.layer, decode_batch_size=2)
    arguments.update(kwargs)
    return adapter.candidate_mhr_parameters(example.raw, example.K, (60, 80), **arguments)


def test_exact_blocks_frames_geometry_and_nonmutation(adapter, example):
    original = {key: value.copy() for key, value in example.raw.items()}
    original_K = example.K.copy()
    for value in example.raw.values():
        value.flags.writeable = False
    example.K.flags.writeable = False
    result = run(adapter, example)
    assert [len(call["mhr_trans"]) for call in example.calls] == [2, 2, 1]
    assert result["frames"] == [f"{i:06d}" for i in range(5)]
    assert result["kids"] == [0] and result["body_model"] == "mhr"
    for key, dimension in adapter.PARAMETER_DIMS.items():
        assert result[key].shape == (5, dimension)
    np.testing.assert_array_equal(result["mhr_keypoints"], example.raw["pred_keypoints_3d"] + example.raw["pred_cam_t"][:, None])
    for key, value in original.items():
        np.testing.assert_array_equal(example.raw[key], value)
    np.testing.assert_array_equal(example.K, original_K)
    assert not any(np.shares_memory(value, source) for value in result.values() if isinstance(value, np.ndarray) for source in example.raw.values())
    metadata = result["metadata"]
    assert metadata["mhr_geometry_forward_verified"] is True
    assert metadata["native_roundtrip_max_error_m"] == dict(vertices=0, joints=0, keypoints=0)
    assert metadata["human_identity_clip_constant"] is False
    assert metadata["submission_eligible"] is False
    assert metadata["ground_truth_used"] is False
    json.dumps(metadata)
    assert "mhr_vertices" not in result


def test_clip_constant_identity_reported_only_when_exact(adapter, example):
    for key in ("shape_params", "scale_params"):
        example.raw[key][:] = example.raw[key][0]
    assert run(adapter, example)["metadata"]["human_identity_clip_constant"] is True


@pytest.mark.parametrize("missing", ["body_pose_params", "hand_pose_params", "scale_params", "shape_params", "expr_params", "vertices_camera_m", "pred_joint_coords"])
def test_missing_raw_blocks_never_inferred_from_204_or_pose_raw(adapter, example, missing):
    del example.raw[missing]
    example.raw["mhr_model_params"] = np.zeros((5, 204), dtype=np.float32)
    example.raw["pred_pose_raw"] = np.zeros((5, 266), dtype=np.float32)
    with pytest.raises(ValueError, match="missing.*no inverse-PCA"):
        run(adapter, example)
    assert example.calls == []


@pytest.mark.parametrize("indices", [[1, 2, 3, 4, 5], [0, 1, 1, 3, 4], [0, 1, 2, 4, 5], [4, 3, 2, 1, 0], [0., 1., 2., 3., 4.], [False]*5])
def test_original_full_timeline_required(adapter, example, indices):
    example.raw["frame_index"] = np.asarray(indices)
    with pytest.raises(ValueError, match="exactly cover"):
        run(adapter, example)


@pytest.mark.parametrize("count", [0, -1, True, 5., "5"])
def test_total_frames_strict_positive_integer(adapter, example, count):
    with pytest.raises(ValueError, match="total_frames"):
        run(adapter, example, total_frames=count)


def test_declared_total_not_silently_replaced(adapter, example):
    with pytest.raises(ValueError, match="exactly cover"):
        run(adapter, example, total_frames=6)


@pytest.mark.parametrize("key", list({"global_rot": 3, "pred_cam_t": 3, "body_pose_params": 133, "hand_pose_params": 108, "shape_params": 45, "scale_params": 28, "expr_params": 72}))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_parameter_nonfinite_rejected(adapter, example, key, bad):
    example.raw[key][0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        run(adapter, example)


@pytest.mark.parametrize("key", ["global_rot", "body_pose_params", "hand_pose_params", "shape_params", "scale_params"])
def test_original_block_shapes_not_repacked(adapter, example, key):
    example.raw[key] = example.raw[key][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        run(adapter, example)


@pytest.mark.parametrize("bad", [1e-5, 1e-100])
def test_nonzero_face_even_float32_underflow_rejected_not_zeroed(adapter, example, bad):
    example.raw["expr_params"] = example.raw["expr_params"].astype(np.float64)
    example.raw["expr_params"][0, 0] = bad
    with pytest.raises(ValueError, match="exactly zero"):
        run(adapter, example)


@pytest.mark.parametrize("edit", [lambda K: K.__setitem__((0, 0), 110), lambda K: K.__setitem__((0, 2), 41), lambda K: K.__setitem__((0, 1), .1), lambda K: K.__setitem__((2, 2), 2), lambda K: K.__setitem__((0, 0), np.nan)])
def test_immutable_K_substitution_or_invalid_rejected(adapter, example, edit):
    edit(example.K)
    with pytest.raises(ValueError, match="K|intrinsics"):
        run(adapter, example)


def test_original_focal_must_match_K(adapter, example):
    example.raw["focal_length"][2] = 105
    with pytest.raises(ValueError, match="Immutable K"):
        run(adapter, example)


def test_camera_translation_double_application_fails(adapter, example):
    example.raw["vertices_camera_m"] += example.raw["pred_cam_t"][:, None]
    with pytest.raises(ValueError, match="translation exactly once"):
        run(adapter, example)


def test_wrong_image_keypoint_projection_fails(adapter, example):
    example.raw["pred_keypoints_2d"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="projection failed"):
        run(adapter, example)


@pytest.mark.parametrize("z", [0, -1])
def test_camera_translation_depth_must_be_positive(adapter, example, z):
    example.raw["pred_cam_t"][0, 2] = z
    with pytest.raises(ValueError, match="positive depth"):
        run(adapter, example)


def test_keypoints_behind_camera_rejected(adapter, example):
    example.raw["pred_keypoints_3d"][0, 0, 2] = -4
    with pytest.raises(ValueError, match="in front"):
        run(adapter, example)


@pytest.mark.parametrize("geometry", ["vertices", "joints", "keypoints"])
@pytest.mark.parametrize("mode", ["wrong_units", "double_translation", "nonfinite", "shape", "missing"])
def test_native_decoder_mandatory_geometry_agreement(adapter, example, monkeypatch, geometry, mode):
    def bad_decode(layer, parameters):
        result = layer.mhr_forward(parameters)
        value = getattr(result, geometry)
        if mode == "wrong_units": value *= 100
        elif mode == "double_translation": value += parameters["mhr_trans"][:, None]
        elif mode == "nonfinite": value.flat[0] = np.nan
        elif mode == "shape": value = value[:, :-1]
        else: value = None
        setattr(result, geometry, value)
        return result
    monkeypatch.setattr(adapter, "_native_decode", bad_decode)
    with pytest.raises(ValueError, match="round-trip|shape/finite"):
        run(adapter, example)


def test_native_wrong_topology_fails(adapter, example, monkeypatch):
    def bad_decode(layer, parameters):
        result = layer.mhr_forward(parameters)
        result.faces = result.faces[:, ::-1].copy()
        return result
    monkeypatch.setattr(adapter, "_native_decode", bad_decode)
    with pytest.raises(ValueError, match="topology"):
        run(adapter, example)


@pytest.mark.parametrize("faces", [np.array([[0, 1, 9]]), np.array([[-1, 0, 1]]), np.empty((0, 3), dtype=int), np.array([[0., 1., 2.]]), np.array([[0, 1]])])
def test_invalid_original_topology_fails(adapter, example, faces):
    example.raw["faces"] = faces
    with pytest.raises(ValueError, match="faces"):
        run(adapter, example)


@pytest.mark.parametrize("size", [(0, 80), (60, True), (60., 80), (60, 80, 3), None])
def test_original_image_size_strict(adapter, example, size):
    with pytest.raises(ValueError, match="image"):
        adapter.candidate_mhr_parameters(example.raw, example.K, size, total_frames=5, mhr_layer=example.layer)


@pytest.mark.parametrize("size", [0, -1, True, 2.])
def test_decode_batch_size_strict(adapter, example, size):
    with pytest.raises(ValueError, match="decode_batch_size"):
        run(adapter, example, decode_batch_size=size)


def test_no_native_layer_no_candidate(adapter, example):
    with pytest.raises(ValueError, match="native MHRLayer"):
        run(adapter, example, mhr_layer=None)


def test_parameter_encoder_wrong_shape_fails_before_decode(adapter, example, monkeypatch):
    monkeypatch.setattr(adapter, "_native_helpers", lambda: (lambda values: np.zeros((len(values), 204), dtype=np.float32), encode_root, lambda: {}))
    with pytest.raises(ValueError, match="mhr_body_pose_cont.*260"):
        run(adapter, example)
    assert example.calls == []


def test_helper_imports_lazy_and_exact_native_contract(adapter, monkeypatch):
    body = SimpleNamespace(compact_model_params_to_cont_body_np=object())
    camera = SimpleNamespace(sam3d_root_camera_to_world_rot6d=object(), mhr_init_parameter_frame_metadata=object())
    monkeypatch.setitem(sys.modules, "lib_mhr", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lib_mhr.body_pose", body)
    monkeypatch.setitem(sys.modules, "lib_mhr.camera_conventions", camera)
    actual = adapter._native_helpers()
    assert actual == (body.compact_model_params_to_cont_body_np, camera.sam3d_root_camera_to_world_rot6d, camera.mhr_init_parameter_frame_metadata)


def test_native_decode_uses_torch_inference_mode_without_real_torch(adapter, monkeypatch):
    events = []
    class Context:
        def __enter__(self): events.append("enter")
        def __exit__(self, *args): events.append("exit")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: Context()))
    layer = SimpleNamespace(mhr_forward=lambda parameters: (events.append("decode"), parameters)[1])
    parameters = {"test": np.array([1.])}
    assert adapter._native_decode(layer, parameters) is parameters
    assert events == ["enter", "decode", "exit"]


def test_root6D_interleaved_identity_and_euler_not_axis_angle():
    np.testing.assert_array_equal(encode_root(np.zeros((1, 3)), np.eye(4)), [[1, 0, 0, 1, 0, 0]])
    euler = np.array([[.2, -.3, .4]])
    expected = Rotation.from_euler("ZYX", euler).as_matrix()[..., :, :2].reshape(1, 6)
    np.testing.assert_allclose(encode_root(euler, np.eye(4)), expected, atol=1e-7)
    assert not np.allclose(expected, Rotation.from_rotvec(euler).as_matrix()[..., :, :2].reshape(1, 6))


def test_tensor_output_canonical_cpu_conversion(adapter):
    events = []
    class Tensor:
        def detach(self): events.append("detach"); return self
        def cpu(self): events.append("cpu"); return self
        def numpy(self): events.append("numpy"); return np.array([1.])
    np.testing.assert_array_equal(adapter._to_numpy(Tensor()), [1.])
    assert events == ["detach", "cpu", "numpy"]

"""Tiny synthetic numerical evidence only; no detector, model, media or labels."""
from dataclasses import FrozenInstanceError, replace
import ast
from pathlib import Path

import numpy as np
import pytest

from world_reward.hand_observations import (
    LANDMARK_NAMES, HandInstances, HandObservations, LandmarkEvidence,
    normalized_hand_instances,
)


def frame(count=2, **changes):
    xy = np.arange(count * 42, dtype=np.float32).reshape(count, 21, 2) / 10
    return HandInstances(**(dict(original_xy=LandmarkEvidence(
        xy, np.ones((count, 21), bool), "original_image", "pixel")) | changes))


def observations(**changes):
    return HandObservations(**(dict(frame_index=np.arange(3, dtype=np.int64),
        image_size=(480, 640), frames=(frame(2), frame(0), frame(3)), method="automatic native method",
        source_refs={"revision": "a" * 40, "checkpoint": {"sha256": "b" * 64},
                     "operations": ["native_original_image_coordinates"]}) | changes))


def assert_owned(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()
    assert not np.shares_memory(a, b) and not b.flags.writeable


def test_ragged_complete_timeline_preserves_empty_and_all_instances():
    result = observations()
    assert result.frame_index.tolist() == [0, 1, 2]
    assert tuple(x.count for x in result.frames) == (2, 0, 3)
    assert result.timestamps_seconds is None
    assert result.landmark_names == LANDMARK_NAMES and len(LANDMARK_NAMES) == 21
    assert result.frames[1].original_xy.values.shape == (0, 21, 2)


def test_readonly_copies_metadata_no_external_aliases():
    f = frame(); ids = np.arange(3, dtype=np.int64)
    refs = {"native": {"operations": ["original"]}}
    result = observations(frames=(f, f, f), frame_index=ids, source_refs=refs)
    assert_owned(f.original_xy.values, result.frames[0].original_xy.values)
    assert_owned(f.original_xy.supported, result.frames[0].original_xy.supported)
    assert not np.shares_memory(result.frames[0].original_xy.values, result.frames[1].original_xy.values)
    ids[:] = 9; refs["native"]["operations"].append("changed")
    assert result.frame_index.tolist() == [0, 1, 2]
    assert result.source_refs["native"]["operations"] == ("original",)
    with pytest.raises(ValueError): result.frames[0].original_xy.values[0, 0, 0] = 1
    with pytest.raises(ValueError): result.frames[0].original_xy.supported[0, 0] = False
    with pytest.raises(TypeError): result.source_refs["new"] = "changed"
    with pytest.raises(FrozenInstanceError): result.method = "changed"
    replacement = replace(result, method="another automatic source")
    assert_owned(result.frames[0].original_xy.values, replacement.frames[0].original_xy.values)


def test_native_instance_permutation_is_not_sorted_or_identity_associated():
    native = np.arange(3 * 63, dtype=np.float64).reshape(3, 21, 3) / 100
    order = [2, 0, 1]
    scores = np.array([.9, .51, .8])
    labels = ("Right", "Left", "Right")
    first = normalized_hand_instances(native, (20, 40), native_handedness_labels=labels,
                                      native_handedness_scores=scores)
    permuted = normalized_hand_instances(native[order], (20, 40),
        native_handedness_labels=tuple(labels[i] for i in order), native_handedness_scores=scores[order])
    assert np.array_equal(permuted.original_xy.values, first.original_xy.values[order])
    assert permuted.native_handedness_labels == ("Right", "Right", "Left")
    assert not hasattr(first, "instance_id") and not hasattr(first, "accepted")


def test_normalized_xy_exact_native_multiply_once_no_half_pixel_or_mirror():
    native = np.zeros((2, 21, 3), np.float32)
    native[:, :, 0] = .3; native[:, :, 1] = .125; native[:, :, 2] = -.23
    native[0, 0, 0] = -0.0
    world = np.arange(126, dtype=np.float32).reshape(2, 21, 3) / 1000
    result = normalized_hand_instances(native, (768, 1024),
        native_handedness_labels=("Left", "Right"), native_handedness_scores=np.array([.999, .501]),
        native_world_xyz=world)
    expected = native[..., :2] * np.asarray([1024, 768], dtype=np.float32)
    assert result.original_xy.values.tobytes() == expected.tobytes()
    assert_owned(native[..., 2:3], result.image_z.values)
    assert_owned(world, result.hand_centred_world_xyz.values)
    assert result.image_z.coordinate_frame == "wrist_relative_image_z"
    assert result.hand_centred_world_xyz.coordinate_frame == "hand_centred_world"
    assert result.hand_centred_world_xyz.unit == "metre"
    assert not hasattr(result, "camera_translation")
    native[:] = 99; world[:] = 99
    assert result.original_xy.values.tobytes() == expected.tobytes()


def test_nonfinite_unsupported_preserved_outside_flag_never_drops():
    native = np.zeros((1, 21, 3), np.float64)
    native[0, 0] = (np.nan, .2, 0)
    native[0, 1] = (.2, np.inf, 0)
    native[0, 2] = (-.1, .3, 0)
    native[0, 3] = (1, .3, np.nan)
    native[0, 4] = (.2, .4, -np.inf)
    f = normalized_hand_instances(native, (100, 200))
    result = observations(frames=(f, frame(0), frame(0)), image_size=(100, 200))
    assert f.count == 1 and f.original_xy.supported[0, :5].tolist() == [False, False, True, True, True]
    assert f.image_z.supported[0, :5].tolist() == [True, True, True, False, False]
    assert result.outside_image[0][0, :5].tolist() == [False, False, True, True, False]
    assert not result.outside_image[0].flags.writeable
    assert np.isnan(f.original_xy.values[0, 0, 0]) and np.isinf(f.original_xy.values[0, 1, 1])
    assert np.isneginf(f.image_z.values[0, 4, 0])


def test_finite_but_producer_unsupported_remains_unsupported():
    f = frame(1)
    support = np.ones((1, 21), bool); support[0, 0] = False
    xy = LandmarkEvidence(f.original_xy.values, support, "original_image", "pixel")
    result = HandInstances(xy, native_handedness_labels=("Left",),
                           native_handedness_scores=np.array([1.0]))
    assert not result.original_xy.supported[0, 0]
    assert not hasattr(result, "visibility") and not hasattr(result, "detection_confidence")
    assert not hasattr(result, "per_joint_confidence")
    assert result.native_handedness_scores[0] == 1


def test_optional_local_coordinates_have_independent_support_and_no_root_recentring():
    values = np.full((1, 21, 3), .7, np.float32)
    support = np.ones((1, 21), bool); values[0, 3, 0] = np.nan; support[0, 3] = False
    local = LandmarkEvidence(values, support, "wrist_relative", "native_model_unit")
    result = frame(1, wrist_relative_xyz=local)
    assert_owned(values, result.wrist_relative_xyz.values)
    assert result.wrist_relative_xyz.values[0, 0, 0] == np.float32(.7)  # no forced wrist zero
    assert result.original_xy.supported[0, 3] and not result.wrist_relative_xyz.supported[0, 3]


def test_real_monotonic_timestamps_preserved_without_fps_inference():
    times = np.array([12., 12.017, 12.053], np.float64)
    result = observations(timestamps_seconds=times)
    assert_owned(times, result.timestamps_seconds)
    times[:] = 0
    assert result.timestamps_seconds.tolist() == [12., 12.017, 12.053]


@pytest.mark.parametrize("change", [
    {"frame_index": np.arange(3, dtype=np.int32)}, {"frame_index": np.array([1, 2, 3])},
    {"frame_index": np.array([0, 2, 3])}, {"frame_index": np.array([], np.int64)},
    {"frame_index": np.ma.array(np.arange(3), mask=False)},
    {"frames": (frame(),)}, {"frames": [frame(), frame(), frame()]},
    {"image_size": (True, 20)}, {"image_size": (0, 20)}, {"image_size": [20, 20]},
    {"method": ""}, {"source_refs": {}}, {"source_refs": {"model": np.zeros(3)}},
    {"source_refs": {"value": np.nan}}, {"source_refs": {"huge": "x" * 9000}},
    {"timestamps_seconds": np.array([0., 0., 1.])},
    {"timestamps_seconds": np.array([0., 2., 1.])}, {"timestamps_seconds": np.array([0., np.nan, 1.])},
    {"timestamps_seconds": np.array([0, 1, 2])}, {"timestamps_seconds": np.array([0., 1.])},
    {"landmark_names": tuple(reversed(LANDMARK_NAMES))},
])
def test_malformed_scene_contracts_fail_without_conversion(change):
    with pytest.raises(ValueError): observations(**change)


@pytest.mark.parametrize("values,support", [
    (np.zeros((1, 20, 2)), np.ones((1, 20), bool)),
    (np.zeros((1, 21, 4)), np.ones((1, 21), bool)),
    (np.zeros((1, 21, 2), int), np.ones((1, 21), bool)),
    (np.ma.array(np.zeros((1, 21, 2)), mask=False), np.ones((1, 21), bool)),
    (np.zeros((1, 21, 2)), np.ones((1, 21))),
    (np.zeros((1, 21, 2)), np.ma.array(np.ones((1, 21), bool), mask=False)),
    (np.full((1, 21, 2), np.nan), np.ones((1, 21), bool)),
    (np.full((1, 21, 2), np.inf), np.ones((1, 21), bool)),
])
def test_numeric_landmark_guards(values, support):
    with pytest.raises(ValueError): LandmarkEvidence(values, support, "original_image", "pixel")


@pytest.mark.parametrize("change", [
    {"original_xy": LandmarkEvidence(np.zeros((1, 21, 2)), np.ones((1, 21), bool), "crop", "pixel")},
    {"original_xy": LandmarkEvidence(np.zeros((1, 21, 2)), np.ones((1, 21), bool), "original_image", "normalized")},
    {"image_z": LandmarkEvidence(np.zeros((1, 21, 3)), np.ones((1, 21), bool), "wrist_relative_image_z", "native")},
    {"hand_centred_world_xyz": LandmarkEvidence(np.zeros((1, 21, 3)), np.ones((1, 21), bool), "global_camera", "metre")},
    {"native_handedness_labels": ["Left"]}, {"native_handedness_labels": ("",)},
    {"native_handedness_labels": ("Left", "Right")},
    {"native_handedness_scores": np.array([.5])},
    {"native_handedness_labels": ("Left",), "native_handedness_scores": np.array([np.nan])},
    {"native_handedness_labels": ("Left",), "native_handedness_scores": np.array([1.01])},
    {"native_handedness_labels": ("Left",), "native_handedness_scores": np.array([-0.1])},
])
def test_wrong_coordinate_or_classification_metadata(change):
    with pytest.raises(ValueError): frame(1, **change)


def test_empty_native_adapter_and_masked_native_rejected():
    result = normalized_hand_instances(np.empty((0, 21, 3), np.float32), (20, 40),
                                      native_handedness_labels=(), native_handedness_scores=np.empty(0, np.float32))
    assert result.count == 0 and result.original_xy.values.shape == (0, 21, 2)
    with pytest.raises(ValueError): normalized_hand_instances(np.ma.array(np.zeros((1, 21, 3))), (20, 40))
    with pytest.raises(ValueError): normalized_hand_instances(np.zeros((1, 21, 3)), (20, 40),
                                                            native_world_xyz=np.zeros((2, 21, 3)))


def test_semantics_and_module_has_no_io_models_or_runtime_environment():
    assert LANDMARK_NAMES[0] == "wrist" and LANDMARK_NAMES[4] == "thumb_tip"
    assert LANDMARK_NAMES[8] == "index_finger_tip" and LANDMARK_NAMES[20] == "pinky_tip"
    path = Path(__file__).parents[1] / "src/world_reward/hand_observations.py"
    tree = ast.parse(path.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert set(imports) <= {"__future__", "collections.abc", "dataclasses", "json", "types", "numpy"}
    assert not any(isinstance(node, ast.Name) and node.id == "open" for node in ast.walk(tree))

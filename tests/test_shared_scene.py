"""Tiny procedural arrays only: no datasets, files, models, camera labels or GT."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from world_reward.contracts import Reconstruction
from world_reward.shared_scene import ArtifactRef, ObservationRef, SceneCamera, SharedScene
from world_reward.submission import Track1Episode, MHR_PARAMETER_FORMAT


def episode():
    count = 4
    pose = np.arange(count * 272, dtype=np.float32).reshape(count, 272)[:, ::2]
    pose[0, 0] = -0.0
    r = Reconstruction(pose, np.arange(68, dtype=np.float64) / 100,
        np.arange(45, dtype=np.float32) / 10, np.tile(np.eye(3), (count, 1, 1)),
        np.column_stack((np.arange(count, dtype=np.float32) / 9, np.zeros((count, 2), np.float32))),
        np.float32(.25))
    return Track1Episode(r, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32), np.zeros(72, np.float32),
        dict(input_track="track_1", ground_truth_used=False, oracle_modes=[], hand_labeled_test=False,
             reference={"path": "results/example.json", "tags": ["automatic"]}), count)


def camera(**changes):
    args = dict(intrinsics=np.array([[800., 0, 320], [0, 800.0000381469727, 240], [0, 0, 1]]),
        image_size=(480, 640), gauge_convention="one RGB-human gauge; object scale not rebaked",
        pixel_convention="half_pixel_centres")
    return SceneCamera(**(args | changes))


def scene(original=None, **changes):
    return SharedScene.from_track1_episode(original or episode(),
        **(dict(clip_id="episode_000008", frame_index=np.arange(4, dtype=np.int64), camera=camera()) | changes))


def assert_bytes(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes(order="C") == right.tobytes(order="C")
    assert not np.shares_memory(left, right)


def test_exact_roundtrip_without_scale_rebake_or_intrinsics_rounding():
    original = episode(); shared = scene(original); recovered = shared.to_track1_episode()
    shared.validate(); recovered.validate()
    for key in ("pose", "shape", "scales", "object_rotation", "object_translation"):
        a, b, c = (getattr(item.reconstruction, key) for item in (original, shared, recovered))
        assert_bytes(a, b); assert_bytes(b, c)
        assert not b.flags.writeable and not c.flags.writeable
    for key in ("object_vertices", "object_faces", "human_expression"):
        assert_bytes(getattr(original, key), getattr(shared, key))
        assert_bytes(getattr(shared, key), getattr(recovered, key))
    assert recovered.reconstruction.object_scale.dtype == original.reconstruction.object_scale.dtype
    assert recovered.reconstruction.object_scale.tobytes() == original.reconstruction.object_scale.tobytes()
    assert shared.camera.intrinsics[1, 1] == 800.0000381469727
    assert recovered.provenance == original.provenance
    assert shared.mhr_parameter_format == MHR_PARAMETER_FORMAT


def test_owns_readonly_arrays_and_frozen_nested_provenance():
    original = episode(); K = camera(); frames = np.arange(4, dtype=np.int64)
    shared = scene(original, camera=K, frame_index=frames)
    original.reconstruction.pose[:] = 999; original.object_vertices[:] = 999; frames[:] = 7
    original.provenance["reference"]["tags"].append("changed")
    assert shared.reconstruction.pose[0, 1] == 2 and shared.object_vertices[0, 0] == 0
    assert shared.frame_index.tolist() == [0, 1, 2, 3]
    assert shared.provenance["reference"]["tags"] == ("automatic",)
    assert not np.shares_memory(shared.camera.intrinsics, K.intrinsics)
    with pytest.raises(ValueError): shared.reconstruction.pose[0, 0] = 1
    with pytest.raises(TypeError): shared.provenance["ground_truth_used"] = True
    with pytest.raises(FrozenInstanceError): shared.clip_id = "changed"
    recovered = shared.to_track1_episode(); recovered.provenance["reference"]["tags"].append("new")
    assert shared.provenance["reference"]["tags"] == ("automatic",)


def test_scene_metadata_copy_preserves_internal_frozen_provenance():
    shared = scene(); updated = replace(shared, clip_id="episode_copy")
    assert updated.provenance == shared.provenance
    assert_bytes(updated.reconstruction.pose, shared.reconstruction.pose)
    updated.validate()


def test_reconstruction_adapter_and_missing_support_remain_explicit():
    original = episode(); supported = np.array([True, False, False, True])
    ref = ArtifactRef("outputs/automatic/points.npz", 12, "a" * 64)
    observation = ObservationRef("object tracks", np.arange(4), supported, ref)
    absent = ObservationRef("hands unavailable", np.arange(4), np.zeros(4, bool))
    shared = SharedScene.from_reconstruction(original.reconstruction, object_vertices=original.object_vertices,
        object_faces=original.object_faces, human_expression=original.human_expression,
        provenance=original.provenance, total_video_frames=4, clip_id="episode_000008",
        frame_index=np.arange(4), camera=camera(), observations=(observation, absent))
    supported[:] = False
    assert shared.observations[0].supported.tolist() == [True, False, False, True]
    assert not shared.observations[1].supported.any() and shared.observations[1].artifact is None
    assert not np.shares_memory(shared.observations[0].supported, observation.supported)
    assert shared.total_video_frames == 4


@pytest.mark.parametrize("change", [
    {"frame_index": np.array([0, 1, 3, 4])}, {"frame_index": np.arange(4, dtype=np.int32)},
    {"frame_index": np.arange(3)}, {"clip_id": ""},
    {"observations": (ObservationRef("missing", np.arange(3), np.zeros(3, bool)),)},
    {"observations": [ObservationRef("missing", np.arange(4), np.zeros(4, bool))]},
    {"observations": tuple(ObservationRef("duplicate", np.arange(4), np.zeros(4, bool)) for _ in range(2))},
])
def test_original_timeline_and_metadata_reject_implicit_conversion(change):
    with pytest.raises(ValueError): scene(**change)


@pytest.mark.parametrize("change", [
    {"intrinsics": np.eye(3)[None]}, {"intrinsics": np.diag([-800., 800., 1.])},
    {"intrinsics": np.diag([800., 800., 2.])}, {"intrinsics": np.full((3, 3), np.nan)},
    {"intrinsics": np.eye(3, dtype=int)}, {"image_size": [480, 640]},
    {"image_size": (True, 640)}, {"gauge_convention": ""}, {"pixel_convention": "unknown"},
    {"camera_policy": "source_calibration"}, {"coordinate_frame": "object"}, {"linear_unit": "mm"},
])
def test_camera_requires_explicit_supplied_shared_frame(change):
    with pytest.raises(ValueError): camera(**change)


@pytest.mark.parametrize("change", [
    {"pose": np.zeros((3, 136))}, {"scales": np.zeros((4, 68))}, {"shape": np.zeros((4, 45))},
    {"pose": np.full((4, 136), np.nan)}, {"object_scale": np.ones(4)}, {"object_scale": 0},
    {"object_rotation": np.tile(np.diag([-1., 1, 1]), (4, 1, 1))},
    {"object_rotation": np.tile(np.eye(3) * 2, (4, 1, 1))},
    {"object_translation": np.zeros((3, 3))}, {"object_translation": np.full((4, 3), np.inf)},
    {"pose": np.ma.array(np.zeros((4, 136)), mask=False)},
])
def test_existing_full_reconstruction_gates_without_repair(change):
    original = episode()
    with pytest.raises(ValueError): scene(replace(original, reconstruction=replace(original.reconstruction, **change)))


@pytest.mark.parametrize("change", [
    {"mhr_parameter_format": "cari_continuous_pose260_scalePCA28"},
    {"human_expression": np.ones(72)}, {"object_vertices": np.zeros((3, 3))},
    {"object_faces": np.array([[0, 1, 99]])},
    {"provenance": dict(input_track="track_2", ground_truth_used=False, oracle_modes=[], hand_labeled_test=False)},
    {"provenance": dict(input_track="track_1", ground_truth_used=True, oracle_modes=[], hand_labeled_test=False)},
    {"provenance": dict(input_track="track_1", ground_truth_used=False, oracle_modes=[], hand_labeled_test=True)},
])
def test_native_layout_mesh_and_video_only_provenance(change):
    with pytest.raises(ValueError): scene(replace(episode(), **change))


@pytest.mark.parametrize("args", [("../x", 1, "a"*64), ("/x", 1, "a"*64), ("x//y", 1, "a"*64),
    ("x", 0, "a"*64), ("x", 1, None), ("x", 1, "bad")])
def test_refs_are_only_bounded_canonical_metadata(args):
    with pytest.raises(ValueError): ArtifactRef(*args)


def test_unproven_supported_observation_and_non_tiny_provenance_rejected():
    with pytest.raises(ValueError): ObservationRef("tracks", np.arange(4), np.ones(4, bool))
    with pytest.raises(ValueError): ObservationRef("tracks", np.arange(4), np.ones(4))
    for extra in ({"data": np.zeros(3)}, {"data": "x" * 17000}, {"data": np.nan}):
        original = episode()
        with pytest.raises(ValueError): scene(replace(original, provenance=original.provenance | extra))

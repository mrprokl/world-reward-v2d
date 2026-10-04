"""Data-free track logic only, not tracking or relational identity accuracy."""
import numpy as np
import pytest

from world_reward.relational_motion import relational_motion_features


def scene(frames=3):
    background = np.array([[0., 0.], [100., 0.], [0., 100.], [100., 100.], [30., 60.]])
    hand = np.array([[20., 20.], [25., 20.], [20., 25.]])
    obj = np.array([[40., 30.], [45., 30.], [40., 35.]])
    repeated = lambda p: np.repeat(p[:, None], frames, axis=1)
    return dict(frame_index=np.arange(frames, dtype=np.int64), timestamps=np.arange(frames, dtype=float),
                hand_tracks=(repeated(hand),), hand_observed=(np.ones((3, frames), dtype=bool),),
                object_tracks=(repeated(obj),), object_observed=(np.ones((3, frames), dtype=bool),),
                background_tracks=repeated(background), background_observed=np.ones((5, frames), dtype=bool),
                image_width=300, image_height=400)


def test_affine_camera_only_motion_is_removed_from_all_pairs_without_physical_camera_claim():
    data = scene()
    matrices = [np.array([[1.1, .2], [-.1, .9]]), np.array([[.9, -.1], [.15, 1.05]])]
    shifts = [np.array([10., -4.]), np.array([-5., 8.])]
    for t, (matrix, shift) in enumerate(zip(matrices, shifts)):
        for track in (*data["hand_tracks"], *data["object_tracks"], data["background_tracks"]):
            track[:, t + 1] = track[:, t] @ matrix.T + shift
    result = relational_motion_features(**data)
    assert result.camera_supported.tolist() == [True, True]
    assert result.pair_supported.all()
    np.testing.assert_allclose(result.features, 0., atol=1e-16)
    np.testing.assert_allclose(result.background_rmse, 0., atol=1e-16)
    for t, (matrix, shift) in enumerate(zip(matrices, shifts)):
        np.testing.assert_allclose(result.camera_affine[t, :, :2], matrix, atol=1e-14)
        np.testing.assert_allclose(result.camera_affine[t, :, 2], shift, atol=1e-13)
    assert not hasattr(result, "physical_camera") and not hasattr(result, "contact")


def test_observed_co_movement_and_differential_motion_have_declared_diagonal_time_units():
    data = scene()
    data["timestamps"] = np.array([7., 9., 13.])
    hand_velocity, object_velocity = np.array([3., 4.]), np.array([3., -4.])
    for t, time in enumerate(data["timestamps"] - data["timestamps"][0]):
        data["hand_tracks"][0][:, t] += hand_velocity * time
        data["object_tracks"][0][:, t] += object_velocity * time
    result = relational_motion_features(**data)
    np.testing.assert_allclose(result.time_intervals, [2., 4.])
    np.testing.assert_allclose(result.hand_velocity[:, 0], [hand_velocity / 500] * 2, atol=1e-16)
    np.testing.assert_allclose(result.object_velocity[:, 0], [object_velocity / 500] * 2, atol=1e-16)
    np.testing.assert_allclose(result.features[:, 0, 0], [[8/500, 25/500**2, 25/500**2]] * 2, atol=1e-16)
    assert result.hand_correspondence_count.tolist() == [[3], [3]]


def test_common_image_translation_and_point_permutation_do_not_change_features():
    original = scene()
    original["hand_tracks"][0][:, 1:] += [4., -3.]
    original["object_tracks"][0][:, 1:] += [-3., 1.]
    shifted = scene()
    offset = np.array([10000., -4000.])
    for name in ("hand_tracks", "object_tracks"):
        shifted[name] = tuple(value[::-1].copy() + offset for value in original[name])
        support = name.replace("tracks", "observed")
        shifted[support] = tuple(value[::-1].copy() for value in original[support])
    shifted["background_tracks"] = original["background_tracks"][[4, 2, 1, 0, 3]].copy() + offset
    shifted["background_observed"] = original["background_observed"][[4, 2, 1, 0, 3]].copy()
    a, b = relational_motion_features(**original), relational_motion_features(**shifted)
    for name in ("features", "hand_velocity", "object_velocity", "hand_dispersion", "object_dispersion"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-14)
    np.testing.assert_array_equal(a.pair_supported, b.pair_supported)


def test_missing_points_and_instances_are_retained_never_interpolated_or_zero_filled():
    data = scene(frames=4)
    data["hand_observed"][0][:, 1] = False
    data["hand_tracks"][0][:, 1] = np.nan
    missing_object = np.full((2, 4, 2), np.nan)
    data["object_tracks"] += (missing_object,)
    data["object_observed"] += (np.zeros((2, 4), dtype=bool),)
    result = relational_motion_features(**data)
    assert result.features.shape == (3, 1, 2, 3)
    assert result.hand_supported[:, 0].tolist() == [False, False, True]
    assert result.hand_correspondence_count[:, 0].tolist() == [0, 0, 3]
    assert result.object_correspondence_count[:, 1].tolist() == [0, 0, 0]
    assert not result.pair_supported[:, :, 1].any()
    assert np.isnan(result.features[:, :, 1]).all()
    assert np.isnan(result.features[:2]).all()
    assert np.isnan(result.hand_velocity[:2]).all()
    assert result.pair_supported[2, 0, 0]


@pytest.mark.parametrize("background", [np.empty((0, 3, 2)), np.zeros((2, 3, 2)),
                                          np.zeros((3, 3, 2)),
                                          np.repeat(np.array([[0., 0.], [1., 2.], [2., 4.]])[:, None], 3, axis=1)])
def test_insufficient_or_collinear_background_does_not_supply_zero_confident_motion(background):
    data = scene()
    data["background_tracks"] = background
    data["background_observed"] = np.ones(background.shape[:2], dtype=bool)
    result = relational_motion_features(**data)
    assert not result.camera_supported.any() and not result.pair_supported.any()
    assert np.isnan(result.features).all() and np.isnan(result.camera_affine).all()
    assert result.hand_correspondence_count.tolist() == [[3], [3]]
    assert not result.hand_supported.any()


def test_exact_three_noncollinear_background_points_are_algebraically_sufficient():
    data = scene()
    data["background_tracks"] = data["background_tracks"][:3]
    data["background_observed"] = data["background_observed"][:3]
    result = relational_motion_features(**data)
    assert result.camera_supported.all()
    assert result.background_design_rank.tolist() == [2, 2]
    assert result.background_correspondence_count.tolist() == [3, 3]
    assert np.isfinite(result.background_condition).all()


def test_same_global_camera_fit_is_shared_by_every_instance_and_has_raw_residuals():
    data = scene()
    extra_hand = data["hand_tracks"][0].copy()
    extra_hand[:, 1:] += [12., 4.]
    data["hand_tracks"] += (extra_hand,)
    data["hand_observed"] += (data["hand_observed"][0].copy(),)
    data["object_tracks"] += (data["object_tracks"][0].copy(),)
    data["object_observed"] += (data["object_observed"][0].copy(),)
    data["background_tracks"][-1, 1] += [200., -100.]  # Deliberately bad background; OLS is not robust.
    result = relational_motion_features(**data)
    assert result.pair_supported.shape == (2, 2, 2) and result.pair_supported.all()
    assert result.background_rmse[0] > 0 and result.camera_supported[0]
    np.testing.assert_array_equal(result.features[:, 0, 0], result.features[:, 0, 1])
    np.testing.assert_array_equal(result.features[:, 1, 0], result.features[:, 1, 1])
    assert not np.allclose(result.hand_velocity[0, 0], 0)


def test_componentwise_median_retains_all_counts_and_exposes_rotational_cancellation():
    data = scene(frames=2)
    hand = np.array([[20., 0.], [0., 20.], [-20., 0.], [0., -20.], [0., 0.]])
    rotation = np.array([[0., -1.], [1., 0.]])
    data["hand_tracks"] = (np.stack((hand, hand @ rotation.T), axis=1),)
    data["hand_observed"] = (np.ones((5, 2), dtype=bool),)
    result = relational_motion_features(**data)
    np.testing.assert_allclose(result.hand_velocity, 0, atol=1e-16)
    assert (result.hand_dispersion > .01).all()
    assert result.hand_correspondence_count.tolist() == [[5]]
    # A zero coherent-motion energy is explicitly not evidence that the hand is stationary.


def test_disjoint_observation_times_cannot_become_correspondences():
    data = scene(frames=2)
    data["hand_observed"][0][:] = [[True, False], [False, True], [False, True]]
    result = relational_motion_features(**data)
    assert result.hand_correspondence_count.tolist() == [[0]]
    assert not result.pair_supported.any()
    assert np.isnan(result.features).all()


def test_single_frame_produces_empty_adjacent_arrays_without_dropping_timeline():
    result = relational_motion_features(**scene(frames=1))
    assert result.frame_index.tolist() == [0]
    assert result.time_intervals.shape == (0,)
    assert result.features.shape == (0, 1, 1, 3)
    assert result.camera_affine.shape == (0, 2, 3)


def test_arrays_are_owned_readonly_inputs_unchanged_and_not_aliased():
    data = scene()
    before = data["hand_tracks"][0].copy()
    result = relational_motion_features(**data)
    np.testing.assert_array_equal(before, data["hand_tracks"][0])
    for value in vars(result).values():
        assert value.flags.owndata and not value.flags.writeable
        assert not np.shares_memory(value, data["background_tracks"])
        assert not np.shares_memory(value, data["hand_tracks"][0])
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = 9
    data["frame_index"][:] = 99
    data["timestamps"][:] = 99
    data["hand_tracks"][0][:] = 99
    assert result.frame_index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(result.features, 0., atol=1e-16)


@pytest.mark.parametrize("change", [
    {"frame_index": np.array([0, 2, 3], dtype=np.int64)},
    {"frame_index": np.arange(3, dtype=np.int32)},
    {"frame_index": np.ma.array(np.arange(3, dtype=np.int64))},
    {"timestamps": np.array([0., 2., 1.])}, {"timestamps": np.array([0., 0., 1.])},
    {"timestamps": np.array([0., np.inf, 2.])}, {"timestamps": np.zeros(2)},
    {"timestamps": np.ma.array(np.arange(3.))},
    {"image_width": 0}, {"image_width": True}, {"image_height": 4.5},
    {"hand_tracks": ()}, {"hand_tracks": [np.zeros((3, 3, 2))]},
    {"hand_tracks": (np.zeros((0, 3, 2)),)}, {"hand_tracks": (np.zeros((3, 3, 2), dtype=int),)},
    {"hand_observed": (np.ones((3, 3)),)}, {"hand_observed": (np.zeros((2, 3), dtype=bool),)},
    {"hand_tracks": (np.full((3, 3, 2), np.nan),)},
    {"hand_tracks": (np.ma.array(np.zeros((3, 3, 2))),)},
    {"object_tracks": (np.zeros((3, 2, 2)),)}, {"object_observed": ()},
    {"background_observed": np.ones((5, 3))},
    {"background_tracks": np.full((5, 3, 2), np.inf)},
])
def test_bad_inputs_fail_instead_of_casting_support_or_fabricating_tracks(change):
    data = scene(); data.update(change)
    with pytest.raises(ValueError):
        relational_motion_features(**data)


def test_float_overflow_fails_closed_instead_of_infinite_supported_features():
    data = scene(frames=2)
    data["timestamps"] = np.array([0., 1e-308])
    data["hand_tracks"][0][:, 1] += [100., 100.]
    with pytest.raises(ValueError, match="overflowed"):
        relational_motion_features(**data)
    data = scene(frames=2)
    data["timestamps"] = np.array([-1e308, 1e308])
    with pytest.raises(ValueError, match="overflowed"):
        relational_motion_features(**data)

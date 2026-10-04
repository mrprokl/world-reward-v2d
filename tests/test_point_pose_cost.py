"""Tiny authored numeric contracts; no challenge/media/model inputs."""
import numpy as np
import pytest

from world_reward.point_pose_cost import point_pose_cost, select_query_indices


def fixture(frames=3, candidates=2):
    points = np.column_stack((np.linspace(-.35, .35, 8), np.linspace(-.2, .25, 8)**2, np.zeros(8)))
    r = np.broadcast_to(np.eye(3), (frames, candidates, 3, 3)).copy()
    t = np.zeros((frames, candidates, 3)); t[..., 2] = 2
    camera = np.array([[400., 0., 300.], [0., 500., 210.], [0., 0., 1.]])
    xyz = points + t[:, 0, None]
    tracks = (xyz[..., :2] / xyz[..., 2, None] * [400, 500] + [300, 210]) * [256/640, 256/480]
    return dict(canonical_points=points, tracks_256=tracks, native_visible=np.ones((frames, 8), bool),
                rotations=r, translations=t, K=camera, frame_index=np.arange(frames, dtype=np.int64),
                image_width=640, image_height=480, valid_candidates=np.ones((frames, candidates), bool))


def grid_fixture():
    mask = np.ones((24, 32), bool)
    points = np.zeros((24, 32, 3), np.float32); points[..., 2] = 2
    return dict(automatic_mask=mask, depth_valid=mask.copy(), surface_visible=mask.copy(), predicted_pointmap=points)


def test_known_pose_and_original_dimension_conversion():
    data = fixture(); result = point_pose_cost(**data)
    np.testing.assert_allclose(result.costs, 0, atol=1e-15)
    assert result.costs.shape == (3, 2) and result.costs.dtype == np.float64
    np.testing.assert_array_equal(result.visible_count, [8, 8, 8])
    assert result.valid_candidates.all() and not result.no_visible_evidence.any()
    assert all(not a.flags.writeable for a in vars(result).values())
    wrong = dict(data, image_height=640)
    assert point_pose_cost(**wrong).costs.min() > .02


def test_frozen_mean_cap_and_visibility_only():
    data = fixture(); data['tracks_256'][..., 0] += 4
    np.testing.assert_allclose(point_pose_cost(**data).costs, .05)
    data['tracks_256'][..., 0] += 100
    np.testing.assert_allclose(point_pose_cost(**data).costs, .1)
    data['native_visible'][:, 1:] = False
    data['tracks_256'][:, 0, 0] -= 104
    np.testing.assert_allclose(point_pose_cost(**data).costs, 0, atol=1e-15)


def test_distinguishable_true_200_degree_rotation_not_symmetry_averaged():
    data = fixture(); angle = np.deg2rad(200)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    data['rotations'][:, 1] = rotation
    result = point_pose_cost(**data)
    np.testing.assert_allclose(result.costs[:, 0], 0, atol=1e-15)
    assert (result.costs[:, 1] > .05).all()
    xyz = data['canonical_points'] @ rotation.T + data['translations'][:, 1, None]
    data['tracks_256'] = (xyz[..., :2] / xyz[..., 2, None] * [400, 500] + [300, 210]) * [256/640, 256/480]
    np.testing.assert_allclose(point_pose_cost(**data).costs[:, 1], 0, atol=1e-15)


def test_occlusion_zero_is_flagged_and_does_not_fabricate_tracks():
    data = fixture(); data['native_visible'][1] = False
    data['translations'][1, :, 0] = 1
    original = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    result = point_pose_cost(**data)
    np.testing.assert_array_equal(result.costs[1], [0, 0])
    np.testing.assert_array_equal(result.no_visible_evidence, [False, True, False])
    for key, value in original.items(): np.testing.assert_array_equal(data[key], value)


def test_nonfront_witness_invalidates_candidate_explicitly_even_if_invisible():
    data = fixture(); data['translations'][1, 1, 2] = -1; data['native_visible'][1] = False
    result = point_pose_cost(**data)
    assert not result.valid_candidates[1, 1] and result.nonfront_witness_count[1, 1] == 8
    assert result.costs[1, 1] == 0 and result.no_visible_evidence[1]
    data['translations'][1, 0, 2] = 0
    with pytest.raises(ValueError, match='no admissible'): point_pose_cost(**data)


def test_invalid_candidate_remains_masked_not_deleted():
    data = fixture(); data['valid_candidates'][0, 1] = False
    result = point_pose_cost(**data)
    assert result.costs.shape == (3, 2) and not result.valid_candidates[0, 1]
    data['valid_candidates'][1] = False
    with pytest.raises(ValueError, match='no admissible'): point_pose_cost(**data)


@pytest.mark.parametrize('key,value', [
    ('frame_index', np.array([0, 2, 3], np.int64)), ('frame_index', np.arange(3, dtype=np.int32)),
    ('frame_index', np.array([[0, 1, 2]], np.int64)), ('image_width', True), ('image_height', 0),
    ('tracks_256', np.zeros((8, 3, 2))), ('tracks_256', np.zeros((3, 8, 2), np.int64)),
    ('tracks_256', np.full((3, 8, 2), np.nan)), ('tracks_256', np.full((3, 8, 2), 1j)),
    ('canonical_points', np.zeros((3, 8, 3))), ('canonical_points', np.zeros((8, 3))),
    ('canonical_points', np.zeros((7, 3))), ('translations', np.zeros((3, 2, 4))),
    ('native_visible', np.ones((3, 8), np.uint8)), ('valid_candidates', np.ones((3, 2), np.int64)),
    ('native_visible', np.ones((2, 8), bool)), ('K', np.tile(np.eye(3), (3, 1, 1))),
    ('K', np.eye(3, dtype=np.int64)), ('K', np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 1]])),
    ('K', np.array([[1., .01, 0], [0, 1, 0], [0, 0, 1]])),
])
def test_dtype_shape_camera_and_original_indices_fail_closed(key, value):
    data = fixture(); data[key] = value
    with pytest.raises(ValueError): point_pose_cost(**data)


@pytest.mark.parametrize('rotation', [np.diag([1., 1., -1.]), np.diag([2., 1., 1.]), np.full((3, 3), np.nan)])
def test_non_so3_rejected_even_masked(rotation):
    data = fixture(); data['rotations'][0, 1] = rotation; data['valid_candidates'][0, 1] = False
    with pytest.raises(ValueError): point_pose_cost(**data)


def test_numeric_overflow_and_masked_array_fail_closed():
    data = fixture(); data['translations'][..., 0] = 1e308
    with pytest.raises(ValueError): point_pose_cost(**data)
    data = fixture(); data['tracks_256'] = np.ma.array(data['tracks_256'], mask=False)
    with pytest.raises(ValueError): point_pose_cost(**data)


def test_query_grid_centres_raster_order_first32_and_no_mutation():
    data = grid_fixture(); original = {k: v.copy() for k, v in data.items()}
    indices = select_query_indices(**data)
    expected = np.array([(y, x) for y in (1, 3) for x in range(1, 32, 2)], np.int64)
    np.testing.assert_array_equal(indices, expected)
    assert not indices.flags.writeable
    for key, value in original.items(): np.testing.assert_array_equal(data[key], value)


def test_query_intersection_does_not_select_non_grid_or_refill_past_fixed_grid():
    data = grid_fixture(); data['automatic_mask'][1, 1] = False
    data['depth_valid'][1, 3] = False; data['surface_visible'][1, 5] = False
    indices = select_query_indices(**data)
    np.testing.assert_array_equal(indices[0], [1, 7])
    assert len(indices) == 32 and indices[-1].tolist() == [5, 5]
    data['automatic_mask'][:] = False; data['automatic_mask'][::2, ::2] = True
    with pytest.raises(ValueError, match='Fewer than8'): select_query_indices(**data)


def test_exact_eight_queries_and_no_support_based_silent_invalid_filter():
    data = grid_fixture(); data['automatic_mask'][:] = False; data['automatic_mask'][1, 1:16:2] = True
    assert len(select_query_indices(**data)) == 8
    data['predicted_pointmap'][1, 1, 2] = 0
    with pytest.raises(ValueError, match='support'): select_query_indices(**data)
    data['depth_valid'][1, 1] = False
    data['predicted_pointmap'][1, 1] = np.nan
    with pytest.raises(ValueError, match='Fewer than8'): select_query_indices(**data)


@pytest.mark.parametrize('key,value', [
    ('automatic_mask', np.ones((24, 32), np.uint8)), ('automatic_mask', np.ones((24, 32, 1), bool)),
    ('depth_valid', np.ones((24, 31), bool)), ('surface_visible', np.ones((24, 32), float)),
    ('predicted_pointmap', np.ones((24, 32, 3), np.int64)), ('predicted_pointmap', np.ones((24, 32, 4))),
    ('automatic_mask', np.ones((11, 32), bool)), ('predicted_pointmap', np.ma.array(np.ones((24, 32, 3)))),
])
def test_query_schema_fail_closed(key, value):
    data = grid_fixture(); data[key] = value
    with pytest.raises(ValueError): select_query_indices(**data)


def test_no_visibility_does_not_hide_impossible_positive_z_projection():
    data = fixture(); data['native_visible'][:] = False
    data['translations'][..., 2] = np.nextafter(0., 1.)
    with pytest.raises(ValueError, match='finite numeric range'): point_pose_cost(**data)


def test_mean_is_over_every_visible_query_not_median_or_best_subset():
    data = fixture(); data['tracks_256'][:, 0, 0] += 8
    np.testing.assert_allclose(point_pose_cost(**data).costs, .1/8)


def test_off_grid_declared_support_invalidity_is_not_silently_hidden():
    data = grid_fixture(); data['predicted_pointmap'][0, 0] = np.nan
    with pytest.raises(ValueError, match='support'): select_query_indices(**data)


def test_float32_inputs_are_supported_without_input_mutation():
    data = fixture()
    for name in ('canonical_points', 'tracks_256', 'rotations', 'translations', 'K'):
        data[name] = data[name].astype(np.float32)
    before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    assert np.max(point_pose_cost(**data).costs) < 1e-6
    for key, value in before.items(): np.testing.assert_array_equal(data[key], value)

"""Tiny procedural pose/quality tests; no images, models or challenge assets."""

import json

import numpy as np
import pytest

from world_reward.view_selection import select_object_views


def fixture():
    return dict(
        frame_indices=np.array([500, 0, 250, 100]),
        rotations=np.repeat(np.eye(3)[None], 4, axis=0),
        translations=np.array([[0., 0., -3.], [3., 0., 0.], [0., 0., 3.], [-3., 0., 0.]]),
        mask_visibility_fractions=np.array([1., .8, .9, .7]),
        depth_valid_fractions=np.ones(4), sharpness=np.ones(4),
        valid=np.ones(4, dtype=bool),
    )


def test_anchor_not_frame_zero_and_selection_order_not_chronological():
    result = select_object_views(**fixture())
    assert result.frame_indices == (500, 250, 0)
    assert result.positions == (0, 2, 1)
    assert result.minimum_angles_rad == pytest.approx((0, np.pi, np.pi / 2))
    assert result.coverage_valid
    np.testing.assert_allclose(result.directions, [[0, 0, 1], [0, 0, -1], [-1, 0, 0]])


def test_direction_uses_object_coordinates_not_translation_only():
    data = fixture()
    quarter_turn = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    data["rotations"][1] = quarter_turn
    data["translations"][:] = [0., 0., 3.]
    result = select_object_views(**data, num_views=2)
    assert result.frame_indices == (500, 0)
    assert result.minimum_angles_rad[1] == pytest.approx(np.pi / 2)
    assert result.directions[1] == pytest.approx((1., 0., 0.))


def test_quality_is_lexicographic_not_weighted_or_normalized():
    data = fixture()
    data["mask_visibility_fractions"][:] = [.9, 1., 1., 1.]
    data["depth_valid_fractions"][:] = [1., .8, .9, .9]
    data["sharpness"][:] = [1e100, 1e90, 2., 3.]
    assert select_object_views(**data, num_views=1).frame_indices == (100,)
    data["sharpness"][2] = 3.
    assert select_object_views(**data, num_views=1).frame_indices == (100,)


def test_zero_sharpness_is_not_a_hidden_rejection_threshold():
    data = fixture()
    data["sharpness"][:] = 0
    assert select_object_views(**data).coverage_valid


def test_invalid_flags_and_zero_support_skip_without_inventing_observations():
    data = fixture()
    data["valid"][0] = False
    data["mask_visibility_fractions"][1] = 0
    data["depth_valid_fractions"][3] = 0
    result = select_object_views(**data, num_views=1)
    assert result.frame_indices == (250,)
    assert result.report["eligible_count"] == 1
    with pytest.raises(ValueError, match="only 1"):
        select_object_views(**data)


def test_row_permutation_changes_positions_not_selected_original_ids():
    data = fixture()
    expected = select_object_views(**data)
    order = np.array([3, 2, 0, 1])
    permuted = {name: value[order] for name, value in data.items()}
    result = select_object_views(**permuted)
    assert result.frame_indices == expected.frame_indices
    assert result.positions == (2, 1, 3)
    assert result.directions == expected.directions


def test_canonical_rotation_preserves_relative_angles_and_selection():
    data = fixture()
    expected = select_object_views(**data)
    axis = np.array([1., 2., 3.]) / np.sqrt(14)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    q = np.eye(3) + np.sin(.37) * skew + (1 - np.cos(.37)) * (skew @ skew)
    data["rotations"] = data["rotations"] @ q
    result = select_object_views(**data)
    assert result.frame_indices == expected.frame_indices
    assert result.minimum_angles_rad == pytest.approx(expected.minimum_angles_rad)
    assert np.asarray(result.directions) == pytest.approx(np.asarray(expected.directions) @ q)


@pytest.mark.parametrize("angle", [0., 1e-7, 1e-6])
def test_degenerate_coverage_abstains_instead_of_claiming_multiview(angle):
    data = fixture()
    data["translations"][:] = [0., 0., -3.]
    data["translations"][1] = [-3 * np.sin(angle), 0., -3 * np.cos(angle)]
    result = select_object_views(**data, num_views=2)
    assert not result.coverage_valid
    assert result.report["status"] == "abstain"
    assert result.report["reason"] == "insufficient_distinct_view_directions"
    assert not result.report["adoption_authorized"]


def test_duplicate_direction_causes_abstention_for_requested_three_views():
    data = fixture()
    data["translations"][:] = [0., 0., -3.]
    data["translations"][1] = [-3., 0., 0.]
    assert not select_object_views(**data).coverage_valid
    assert select_object_views(**data, num_views=1).coverage_valid


def test_large_translation_and_scale_do_not_change_direction_or_overflow():
    data = fixture()
    expected = select_object_views(**data)
    data["translations"] *= 1e307
    result = select_object_views(**data)
    assert result.directions == expected.directions
    assert result.frame_indices == expected.frame_indices


def test_report_is_json_safe_and_explicitly_only_a_hypothesis():
    result = select_object_views(**fixture())
    assert json.loads(json.dumps(result.report, allow_nan=False))["ground_truth_used"] is False
    assert result.report["pose_predictions_unverified"]
    assert result.report["quality_is_proxy"]
    assert not result.report["adoption_authorized"]


def test_inputs_are_not_modified():
    data = fixture()
    before = {name: value.copy() for name, value in data.items()}
    select_object_views(**data)
    for name in data:
        np.testing.assert_array_equal(data[name], before[name])


@pytest.mark.parametrize("num_views", [False, True, np.bool_(True), 0, -1, 1.5, "3", None, 5])
def test_invalid_requested_count(num_views):
    with pytest.raises(ValueError):
        select_object_views(**fixture(), num_views=num_views)


def test_numpy_integer_requested_count_is_supported():
    assert len(select_object_views(**fixture(), num_views=np.int64(2)).frame_indices) == 2


@pytest.mark.parametrize("indices", [[], [0., 1., 2., 3.], [False, True, False, True], [0, 1, 1, 3],
                                    [-1, 1, 2, 3], [[0, 1, 2, 3]], ["0", "1", "2", "3"], [False, 1, 2, 3]])
def test_invalid_original_frame_indices(indices):
    data = fixture()
    data["frame_indices"] = indices
    with pytest.raises(ValueError, match="frame_indices"):
        select_object_views(**data)


@pytest.mark.parametrize("field", ["rotations", "translations", "mask_visibility_fractions", "depth_valid_fractions", "sharpness"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_even_when_explicit_invalid(field, bad):
    data = fixture()
    data["valid"][0] = False
    data[field].flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        select_object_views(**data, num_views=1)


@pytest.mark.parametrize("field", ["frame_indices", "rotations", "translations", "mask_visibility_fractions", "depth_valid_fractions", "sharpness", "valid"])
def test_masked_array_rejected_without_losing_mask(field):
    data = fixture()
    data[field] = np.ma.array(data[field], mask=False)
    with pytest.raises(ValueError, match="masked"):
        select_object_views(**data)


@pytest.mark.parametrize("field", ["rotations", "translations", "mask_visibility_fractions", "depth_valid_fractions", "sharpness", "valid"])
def test_shapes_must_cover_all_original_frames(field):
    data = fixture()
    data[field] = data[field][:-1]
    with pytest.raises(ValueError, match="shape|vector"):
        select_object_views(**data)


@pytest.mark.parametrize("field", ["rotations", "translations", "mask_visibility_fractions", "depth_valid_fractions", "sharpness"])
def test_bool_numeric_arrays_not_accepted(field):
    data = fixture()
    data[field] = data[field].astype(bool)
    with pytest.raises(ValueError, match="numeric"):
        select_object_views(**data)


@pytest.mark.parametrize("valid", [[1, 1, 1, 1], ["true"] * 4, [True] * 3])
def test_explicit_boolean_validity_required(valid):
    data = fixture()
    data["valid"] = valid
    with pytest.raises(ValueError, match="boolean"):
        select_object_views(**data)


@pytest.mark.parametrize("field,bad", [("mask_visibility_fractions", -.01), ("mask_visibility_fractions", 1.01),
                                       ("depth_valid_fractions", -.01), ("depth_valid_fractions", 1.01), ("sharpness", -.01)])
def test_out_of_range_quality_rejected(field, bad):
    data = fixture()
    data[field][0] = bad
    with pytest.raises(ValueError, match="fractions|nonnegative"):
        select_object_views(**data)


@pytest.mark.parametrize("bad_rotation", [np.diag([-1., 1., 1.]), np.diag([1., 1., 1.01]), np.zeros((3, 3))])
def test_improper_or_nonorthogonal_pose_rejected_without_repair(bad_rotation):
    data = fixture()
    data["rotations"][0] = bad_rotation
    data["valid"][0] = False
    with pytest.raises(ValueError, match="SO"):
        select_object_views(**data)


def test_zero_translation_rejected_even_if_ineligible():
    data = fixture()
    data["translations"][0] = 0
    data["valid"][0] = False
    with pytest.raises(ValueError, match="nonzero"):
        select_object_views(**data)

"""Tiny synthetic control arrays, not predictions or challenge validation."""

import json

import numpy as np
import pytest

from world_reward.hand_transfer import transfer_finger_controls


@pytest.fixture
def arrays():
    final = (np.arange(5 * 136).reshape(5, 136) / 200).astype(np.float32)
    source = (10 + np.arange(3 * 204).reshape(3, 204) / 100).astype(np.float64)
    frames = np.array([0, 2, 4], dtype=np.int64)
    left = np.arange(95, 122, dtype=np.int64)
    right = np.arange(68, 95, dtype=np.int64)
    return final, source, frames, left, right


def test_exact_finger_only_transfer_and_sparse_original_frame_coverage(arrays):
    final, source, frames, left, right = arrays
    final[1, 0] = -0.0
    result = transfer_finger_controls(*arrays)
    assert result.pose.shape == final.shape and result.pose.dtype == final.dtype
    for row, original in enumerate(frames):
        np.testing.assert_array_equal(result.pose[original, 68:122], source[row, 68:122].astype(np.float32))
    np.testing.assert_array_equal(result.pose[[1, 3]], final[[1, 3]])
    fixed = np.r_[np.arange(68), np.arange(122, 136)]
    assert result.pose[:, fixed].tobytes() == final[:, fixed].tobytes()
    assert np.signbit(result.pose[1, 0])
    assert result.report["final_frames"] == 5 and result.report["source_frames"] == 3
    assert result.report["source_missing_frames_interpolated"] is False
    assert result.report["source_scale_columns_copied"] is False
    assert result.report["adoption_performed"] is False
    json.dumps(result.report, allow_nan=False)


def test_full_source_frame_coverage(arrays):
    final, source, _, left, right = arrays
    all_source = np.tile(source[:1], (5, 1))
    result = transfer_finger_controls(final, all_source, np.arange(5), left, right)
    np.testing.assert_array_equal(result.pose[:, 68:122], all_source[:, 68:122].astype(np.float32))
    assert result.report["selected_left_hand_frames"] == 5


def test_hand_index_order_not_reinterpreted_as_vector_layout(arrays):
    final, source, frames, left, right = arrays
    normal = transfer_finger_controls(*arrays)
    shuffled = transfer_finger_controls(final, source, frames, left[::-1], np.roll(right, 8))
    np.testing.assert_array_equal(shuffled.pose, normal.pose)
    assert shuffled.report["hand_indices_left"] == left[::-1].tolist()


def test_optional_boolean_selection_keeps_unselected_hand_not_zero(arrays):
    final, source, frames, left, right = arrays
    mask = np.array([[True, False], [False, True], [False, False]])
    result = transfer_finger_controls(*arrays, handoff_mask=mask)
    np.testing.assert_array_equal(result.pose[0, left], source[0, left].astype(np.float32))
    np.testing.assert_array_equal(result.pose[0, right], final[0, right])
    np.testing.assert_array_equal(result.pose[2, left], final[2, left])
    np.testing.assert_array_equal(result.pose[2, right], source[1, right].astype(np.float32))
    np.testing.assert_array_equal(result.pose[4], final[4])
    assert result.report["selected_original_frames_by_hand"] == [[0], [2]]


def test_false_mask_does_not_zero_controls_or_alias_final_input(arrays):
    result = transfer_finger_controls(*arrays, handoff_mask=np.zeros((3, 2), dtype=bool))
    np.testing.assert_array_equal(result.pose, arrays[0])
    assert result.pose is not arrays[0] and not np.shares_memory(result.pose, arrays[0])
    assert result.report["selected_left_hand_frames"] == result.report["selected_right_hand_frames"] == 0


def test_no_input_arrays_or_boolean_mask_are_mutated(arrays):
    mask = np.array([[True, False], [False, True], [True, True]])
    before = [array.copy() for array in (*arrays, mask)]
    result = transfer_finger_controls(*arrays, handoff_mask=mask)
    result.pose[:] = 0
    for original, copy in zip((*arrays, mask), before):
        assert original.tobytes() == copy.tobytes()


@pytest.mark.parametrize("frames", [[0, 4, 2], [0, 2, 2], [-1, 2, 4], [0, 2, 5], [0, 2],
                                    [0., 2., 4.], [True, False, True], np.zeros((3, 1), int),
                                    np.ma.array([0, 2, 4], mask=False)])
def test_invalid_frame_mappings_fail_no_interpolation_or_reordering(arrays, frames):
    final, source, _, left, right = arrays
    with pytest.raises(ValueError): transfer_finger_controls(final, source, frames, left, right)


@pytest.mark.parametrize("side", [3, 4])
@pytest.mark.parametrize("failure", ["duplicate", "wrong_column", "float", "bool", "wrong_size", "masked"])
def test_invalid_hand_index_sets_fail(arrays, side, failure):
    args = list(arrays)
    indices = args[side].copy()
    if failure == "duplicate": indices[1] = indices[0]
    elif failure == "wrong_column": indices[0] = 67
    elif failure == "float": indices = indices.astype(float)
    elif failure == "bool": indices = indices.astype(bool)
    elif failure == "wrong_size": indices = indices[:-1]
    elif failure == "masked": indices = np.ma.array(indices, mask=False)
    args[side] = indices
    with pytest.raises(ValueError): transfer_finger_controls(*args)


def test_duplicate_side_partition_or_wrist_column_is_rejected(arrays):
    args = list(arrays)
    args[4] = args[3].copy()
    with pytest.raises(ValueError, match="partition"): transfer_finger_controls(*args)
    args = list(arrays)
    args[3][0] = 47  # Body/wrist proposal is out of scope.
    with pytest.raises(ValueError, match="partition"): transfer_finger_controls(*args)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("failure", ["empty", "wrong_columns", "integer", "nan", "inf", "masked"])
def test_finite_float_control_arrays_strict(arrays, which, failure):
    args = list(arrays)
    value = args[which].copy()
    if failure == "empty": value = value[:0]
    elif failure == "wrong_columns": value = value[:, :-1]
    elif failure == "integer": value = value.astype(int)
    elif failure == "nan": value[0, 0] = np.nan
    elif failure == "inf": value[0, 0] = np.inf
    elif failure == "masked": value = np.ma.array(value, mask=False)
    args[which] = value
    with pytest.raises(ValueError): transfer_finger_controls(*args)


@pytest.mark.parametrize("mask", [np.zeros((3, 2), int), np.zeros((3, 2), float), np.zeros((3,), bool),
                                  np.zeros((5, 2), bool), np.ma.array(np.ones((3, 2), bool), mask=False)])
def test_optional_mask_must_be_explicit_source_row_left_right_booleans(arrays, mask):
    with pytest.raises(ValueError): transfer_finger_controls(*arrays, handoff_mask=mask)


def test_selected_controls_cannot_overflow_final_dtype(arrays):
    final, source, frames, left, right = arrays
    source[0, left[0]] = 1e308
    with pytest.raises(ValueError, match="represented"): transfer_finger_controls(*arrays)
    mask = np.ones((3, 2), bool)
    mask[0, 0] = False
    result = transfer_finger_controls(*arrays, handoff_mask=mask)
    np.testing.assert_array_equal(result.pose[0, left], final[0, left])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_final_float_dtype_is_preserved_without_fitting_identity(arrays, dtype):
    final, source, frames, left, right = arrays
    result = transfer_finger_controls(final.astype(dtype), source, frames, left, right)
    assert result.pose.dtype == dtype
    np.testing.assert_array_equal(result.pose[frames, 68:122], source[:, 68:122].astype(dtype))
    assert result.report["identity_fitted"] is False

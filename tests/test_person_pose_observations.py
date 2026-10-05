"""Procedural fake-native ABI controls; NOT pose/ownership accuracy validation."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from world_reward.person_pose_observations import (
    BODY_WRIST_INDICES, HAND_ROOT_INDICES, NUM_KEYPOINTS, PersonPoseObservations,
    infer_person_pose_frame,
)


def inputs(count=3):
    rgb = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    boxes = np.array([[i + .25, 1, i + 4.5, 14] for i in range(count)], dtype=np.float32).reshape(count, 4)
    ids = tuple(f"person:original:{i}" for i in range(count))
    scores = np.linspace(.01, .99, count, dtype=np.float32)
    return rgb, 37, ids, boxes, scores


class Native:
    def __init__(self):
        self.calls = []

    def __call__(self, session, boxes, rgb):
        assert session is SESSION
        self.calls.append((boxes.copy(), rgb.copy()))
        assert not boxes.flags.writeable and not rgb.flags.writeable
        n = len(boxes)
        self.points = np.zeros((n, 133, 2), dtype=np.float64)
        self.points[:] = boxes[:, None, :2]
        self.scores = np.ones((n, 133), dtype=np.float32)
        return self.points, self.scores


SESSION = object()


def run(native=None, values=None):
    return infer_person_pose_frame(native or Native(), SESSION, *(values or inputs()))


@pytest.mark.parametrize("count", [2, 3, 9])
def test_one_whole_bank_call_preserves_every_person_order_grid_and_raw_dtype(count):
    values = inputs(count)
    native = Native()
    result = run(native, values)
    assert len(native.calls) == 1
    assert result.original_frame_index == 37 and result.image_size == (16, 24)
    assert result.person_ids == values[2] and len(result.person_ids) == count
    np.testing.assert_array_equal(native.calls[0][0], values[3])
    np.testing.assert_array_equal(native.calls[0][1], values[0])
    np.testing.assert_array_equal(result.boxes_original_xyxy, values[3])
    np.testing.assert_array_equal(result.detector_scores, values[4])
    np.testing.assert_array_equal(result.keypoints_original_xy, native.points)
    assert result.boxes_original_xyxy.dtype == np.float32
    assert result.keypoints_original_xy.dtype == np.float64 and result.raw_scores.dtype == np.float32
    assert result.anatomical_ownership_verified is False


def test_permuted_person_ids_and_boxes_remain_aligned_no_sort_or_rematch():
    rgb, frame, ids, boxes, scores = inputs()
    permutation = [2, 0, 1]
    values = (rgb, frame, tuple(ids[i] for i in permutation), boxes[permutation], scores[permutation])
    result = run(values=values)
    assert result.person_ids == values[2]
    np.testing.assert_array_equal(result.keypoints_original_xy[:, 9], boxes[permutation, :2])
    np.testing.assert_array_equal(result.detector_scores, scores[permutation])


def test_identical_boxes_distinct_ids_are_not_deduplicated():
    values = list(inputs(2)); values[3][1] = values[3][0]
    result = run(values=values)
    assert len(result.person_ids) == 2
    np.testing.assert_array_equal(result.keypoints_original_xy[0], result.keypoints_original_xy[1])


def test_no_detections_skips_native_full_image_fallback():
    def forbidden(*_):
        raise AssertionError("Empty boxes must never reach native")
    result = run(forbidden, inputs(0))
    assert result.person_ids == () and result.boxes_original_xyxy.shape == (0, 4)
    assert result.detector_scores.shape == (0,)
    assert result.keypoints_original_xy.shape == (0, 133, 2) and result.keypoints_original_xy.dtype == np.float64
    assert result.raw_scores.shape == (0, 133) and result.raw_scores.dtype == np.float32
    assert result.native_valid.shape == result.in_original_image.shape == (0, 133)


def test_misses_and_offgrid_keep_raw_scores_coordinates_and_separate_flags():
    native = Native()
    def original(session, boxes, rgb):
        points, scores = native(session, boxes, rgb)
        scores[0, :6] = [-3, -0., 0., np.nextafter(np.float32(0), np.float32(1)), .2, 4]
        points[0, :6] = [[7, 7], [-1, -1], [2, 3], [-1, 5], [24, 16], [23.999, 15.999]]
        scores[1] = 0  # missed person stays present, not removed
        return points, scores
    result = run(original)
    assert result.person_ids == inputs()[2]
    np.testing.assert_array_equal(result.raw_scores, native.scores)
    np.testing.assert_array_equal(result.keypoints_original_xy, native.points)
    assert np.signbit(result.raw_scores[0, 1])
    assert result.native_valid[0, :6].tolist() == [False, False, False, True, True, True]
    assert result.in_original_image[0, :6].tolist() == [True, False, True, False, False, True]
    assert not result.native_valid[1].any()


def test_distinct_original_wrist_and_handroot_source_indices():
    assert NUM_KEYPOINTS == 133 and BODY_WRIST_INDICES == (9, 10) and HAND_ROOT_INDICES == (91, 112)
    def native(*args):
        n = len(args[1])
        p = np.broadcast_to(np.arange(133, dtype=np.float64)[None, :, None], (n, 133, 2)).copy()
        return p, np.ones((n, 133), np.float32)
    result = run(native)
    assert result.keypoints_original_xy[0, BODY_WRIST_INDICES, 0].tolist() == [9, 10]
    assert result.keypoints_original_xy[0, HAND_ROOT_INDICES, 0].tolist() == [91, 112]


def test_owned_readonly_outputs_and_nonmutated_inputs():
    values = inputs(); before = [a.copy() for a in (values[0], values[3], values[4])]
    native = Native(); result = run(native, values)
    native.points[:] = 999; native.scores[:] = 999
    assert not (result.keypoints_original_xy == 999).any() and not (result.raw_scores == 999).any()
    for a, b in zip((values[0], values[3], values[4]), before):
        np.testing.assert_array_equal(a, b)
    for name in ("boxes_original_xyxy", "detector_scores", "keypoints_original_xy", "raw_scores",
                 "native_valid", "in_original_image"):
        assert not getattr(result, name).flags.writeable
    with pytest.raises(FrozenInstanceError):
        result.person_ids = ()


@pytest.mark.parametrize("field", ["boxes", "rgb", "original_rgb", "original_boxes", "detector_scores"])
def test_native_mutation_even_after_unsetting_readonly_fails_closed(field):
    values = inputs()
    def native(session, boxes, rgb):
        points, scores = Native()(session, boxes, rgb)
        target = {"boxes": boxes, "rgb": rgb, "original_rgb": values[0],
                  "original_boxes": values[3], "detector_scores": values[4]}[field]
        target.flags.writeable = True
        target.flat[0] += 1
        return points, scores
    with pytest.raises(ValueError, match="mutated"):
        run(native, values)


@pytest.mark.parametrize("box", [[-1, 0, 4, 4], [0, -1, 4, 4], [0, 0, 25, 4], [0, 0, 4, 17],
                                [2, 2, 2, 4], [3, 2, 1, 4], [0, 4, 4, 4], [0, 0, np.nan, 4]])
def test_invalid_boxes_rejected_before_native_without_clipping(box):
    values = list(inputs(2)); values[3][1] = box
    native = Native()
    with pytest.raises(ValueError):
        run(native, values)
    assert native.calls == []


@pytest.mark.parametrize("ids", [("same", "same", "other"), ("only",), ("", "b", "c"),
                                (1, "b", "c"), ["a", "b", "c"], (" a", "b", "c")])
def test_invalid_or_duplicate_original_ids_rejected_before_native(ids):
    values = list(inputs()); values[2] = ids
    native = Native()
    with pytest.raises(ValueError):
        run(native, values)
    assert native.calls == []


@pytest.mark.parametrize("change", ["rgb_float", "rgb_gray", "frame_bool", "frame_negative", "boxes_masked",
                                   "scores_nan", "scores_shape", "points_nan", "points_int", "points_shape",
                                   "pose_scores_inf", "pose_scores_shape", "list_output"])
def test_invalid_wholebank_inputs_or_native_outputs_fail_no_partial_salvage(change):
    values = list(inputs()); original = Native()
    if change == "rgb_float": values[0] = values[0].astype(np.float32)
    if change == "rgb_gray": values[0] = values[0][..., 0]
    if change == "frame_bool": values[1] = True
    if change == "frame_negative": values[1] = -1
    if change == "boxes_masked": values[3] = np.ma.array(values[3])
    if change == "scores_nan": values[4][0] = np.nan
    if change == "scores_shape": values[4] = values[4][:1]
    def native(*args):
        points, scores = original(*args)
        if change == "points_nan": points[-1, -1, 0] = np.nan
        if change == "points_int": points = points.astype(np.int64)
        if change == "points_shape": points = points[:1]
        if change == "pose_scores_inf": scores[-1, -1] = np.inf
        if change == "pose_scores_shape": scores = scores[:1]
        return [points, scores] if change == "list_output" else (points, scores)
    with pytest.raises(ValueError):
        run(native, values)


def test_native_exceptions_propagate_without_retry_or_empty_rescue():
    calls = []
    def failed(*args):
        calls.append(args)
        raise RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        run(failed)
    assert len(calls) == 1


def test_direct_record_construction_cannot_bypass_shape_or_ids_contract():
    result = run()
    with pytest.raises(ValueError):
        replace(result, person_ids=("duplicate",) * 3)
    with pytest.raises(ValueError):
        replace(result, keypoints_original_xy=result.keypoints_original_xy[:1])
    assert isinstance(result, PersonPoseObservations)

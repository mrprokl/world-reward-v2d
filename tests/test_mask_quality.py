import json
import math

import numpy as np
import pytest

from world_reward.mask_quality import analyze_mask_sequence
from world_reward.prompt_selection import BoxDetection, FrameDetections


def blank(count=1, height=8, width=10):
    return np.zeros((count, height, width), dtype=bool)


def analyze(person, obj, indices=None, **kwargs):
    return analyze_mask_sequence(person, obj, list(range(len(person))) if indices is None else indices, **kwargs)


def test_rectangle_area_half_open_bbox_pixel_center_and_component():
    person, obj = blank(), blank()
    person[0, 1:5, 2:8] = True
    result = analyze(person, obj, [17], total_frames=18)
    geometry = result["frames"][0]["person"]
    assert geometry["area_pixels"] == 24
    assert geometry["area_fraction"] == 24 / 80
    assert geometry["bbox_xyxy"] == [2, 1, 8, 5]
    assert geometry["centroid_xy"] == [5, 3]
    assert geometry["component_count"] == 1
    assert geometry["largest_component_fraction"] == 1
    assert geometry["principal_components"][0]["bbox_xyxy"] == [2, 1, 8, 5]
    assert result["original_frame_indices"] == [17]
    assert result["temporal_pairs"] == []
    assert "not accuracy" in result["interpretation"]
    json.dumps(result, allow_nan=False)


def test_empty_masks_report_none_without_rejecting_occlusion():
    result = analyze(blank(3), blank(3))
    geometry = result["frames"][0]["object"]
    assert geometry["area_pixels"] == 0
    assert geometry["bbox_xyxy"] is geometry["centroid_xy"] is None
    assert geometry["component_count"] == 0
    assert geometry["largest_component_fraction"] is None
    assert geometry["principal_components"] == []
    assert result["frames"][0]["overlap"]["mask_iou"] is None
    assert result["summary"]["object_empty_frames"] == 3
    assert result["temporal_pairs"][0]["object"]["centroid_distance_pixels"] is None
    assert result["temporal_pairs"][0]["object"]["absolute_area_change_over_max_area"] == 0
    json.dumps(result, allow_nan=False)


def test_eight_connected_diagonal_is_one_component():
    person, obj = blank(), blank()
    person[0, 1, 1] = person[0, 2, 2] = True
    result = analyze(person, obj)
    assert result["component_connectivity"] == 8
    assert result["frames"][0]["person"]["component_count"] == 1


def test_fragmentation_top_three_components_are_bounded_and_deterministic():
    person, obj = blank(), blank()
    person[0, 0:2, 0:2] = True
    person[0, 0, 5] = person[0, 3, 5] = person[0, 6, 5] = True
    geometry = analyze(person, obj)["frames"][0]["person"]
    assert geometry["area_pixels"] == 7
    assert geometry["component_count"] == 4
    assert geometry["largest_component_area_pixels"] == 4
    assert geometry["largest_component_fraction"] == 4 / 7
    assert len(geometry["principal_components"]) == 3
    assert [c["bbox_xyxy"] for c in geometry["principal_components"]] == [
        [0, 0, 2, 2], [5, 0, 6, 1], [5, 3, 6, 4],
    ]


def test_person_object_overlap_ratios_are_not_contact_or_accuracy_claims():
    person, obj = blank(), blank()
    person[0, 0:4, 0:4] = True
    obj[0, 2:4, 2:6] = True
    overlap = analyze(person, obj)["frames"][0]["overlap"]
    assert overlap == {
        "intersection_pixels": 4, "fraction_of_person": 0.25,
        "fraction_of_object": 0.5, "mask_iou": 4 / 20,
    }


def test_temporal_large_motion_sparse_gap_and_area_change_report_without_reject():
    person, obj = blank(2), blank(2)
    obj[0, 0:2, 0:2] = True
    obj[1, 4:8, 6:10] = True
    result = analyze(person, obj, [5, 15])
    pair = result["temporal_pairs"][0]
    assert pair["from_frame_index"] == 5 and pair["to_frame_index"] == 15
    assert pair["frame_gap"] == 10 and pair["contiguous"] is False
    change = pair["object"]
    assert change["area_delta_pixels"] == 12
    assert change["absolute_area_change_over_max_area"] == 0.75
    assert change["centroid_distance_pixels"] == pytest.approx(math.hypot(7, 5))
    assert change["centroid_distance_over_image_diagonal_per_frame"] == pytest.approx(
        math.hypot(7, 5) / math.hypot(10, 8) / 10
    )
    assert change["empty_transition"] is False


def test_empty_transition_records_area_but_no_invented_centroid():
    person, obj = blank(3), blank(3)
    obj[1, 0:2, 0:2] = True
    pairs = analyze(person, obj)["temporal_pairs"]
    assert pairs[0]["contiguous"]
    assert pairs[0]["object"]["empty_transition"]
    assert pairs[1]["object"]["empty_transition"]
    assert pairs[0]["object"]["absolute_area_change_over_max_area"] == 1
    assert pairs[1]["object"]["area_delta_pixels"] == -4
    assert all(pair["object"]["centroid_distance_pixels"] is None for pair in pairs)


def test_independent_box_agreement_report_all_hypotheses_not_truth():
    person, obj = blank(2), blank(2)
    person[:, 1:5, 2:8] = True
    detections = [FrameDetections(9, 10, 8, (
        BoxDetection((2, 1, 8, 5), 0.8), BoxDetection((0, 0, 2, 2), 0.9),
    ), ())]
    result = analyze(person, obj, [2, 9], detections=detections)
    assert result["frames"][0]["detection_agreement"] is None
    agreement = result["frames"][1]["detection_agreement"]
    assert agreement["person"]["max_bbox_iou"] == 1
    assert len(agreement["person"]["detections"]) == 2
    assert agreement["person"]["detections"][0]["score"] == 0.9
    assert agreement["person"]["detections"][0]["bbox_iou"] == 0
    assert agreement["object"] == {"detections": [], "max_bbox_iou": None}
    assert result["summary"]["frames_with_independent_detections"] == 1


def test_empty_mask_and_available_independent_box_agreement_is_none_not_fake_zero():
    detection = FrameDetections(0, 10, 8, (), (BoxDetection((0, 0, 2, 2), 0.9),))
    agreement = analyze(blank(), blank(), detections=[detection])["frames"][0]["detection_agreement"]
    assert agreement["object"]["detections"][0]["bbox_iou"] is None
    assert agreement["object"]["max_bbox_iou"] is None


def test_streaming_iterables_equal_array_results_and_inputs_unchanged():
    person, obj = blank(2), blank(2)
    person[0, 1:3, 2:4] = True
    before = person.copy()
    expected = analyze(person, obj, [1, 5])
    actual = analyze_mask_sequence((mask for mask in person), (mask for mask in obj), [1, 5])
    assert expected == actual
    assert np.array_equal(person, before)


@pytest.mark.parametrize("bad", [
    np.zeros((1, 8, 10), dtype=np.uint8), np.zeros((1, 8, 10), dtype=float),
    np.full((1, 8, 10), np.nan), np.full((1, 8, 10), np.inf),
    np.full((1, 8, 10), "False"), np.zeros((8, 10), dtype=bool),
    np.zeros((1, 0, 10), dtype=bool), np.ma.array(blank(), mask=False),
])
def test_invalid_binary_mask_dtype_nonfinite_or_shapes_fail_without_coercion(bad):
    with pytest.raises(ValueError):
        analyze_mask_sequence(bad, blank(), [0])


@pytest.mark.parametrize("person,obj,indices", [
    (blank(2), blank(1), [0, 1]), (blank(1), blank(2), [0]),
    (blank(2), blank(2), [0]), (blank(1), blank(1), [0, 1]),
    (blank(1), blank(1, width=9), [0]),
    ([np.zeros((8, 10), bool), np.zeros((9, 10), bool)], blank(2), [0, 1]),
])
def test_mask_counts_and_fixed_shapes_match_original_ids(person, obj, indices):
    with pytest.raises(ValueError):
        analyze_mask_sequence(person, obj, indices)


@pytest.mark.parametrize("indices", [[], [1, 1], [1, 0], [-1], [True], [0.0], [float("nan")], [float("inf")]])
def test_invalid_original_indices_fail(indices):
    with pytest.raises(ValueError):
        analyze_mask_sequence(blank(len(indices)), blank(len(indices)), indices)


@pytest.mark.parametrize("total_frames", [0, True, 1.5, float("inf")])
def test_invalid_total_frames_fail(total_frames):
    with pytest.raises(ValueError):
        analyze(blank(), blank(), total_frames=total_frames)


def test_original_index_outside_declared_clip_fails():
    with pytest.raises(ValueError, match="outside total_frames"):
        analyze(blank(), blank(), [4], total_frames=4)


@pytest.mark.parametrize("detections", [
    [FrameDetections(9, 10, 8)], [FrameDetections(0, 10, 8), FrameDetections(0, 10, 8)],
    [FrameDetections(0, 11, 8)], [FrameDetections(0, 10, True)], [{}], {},
])
def test_invalid_independent_detection_frame_ids_or_dimensions_fail(detections):
    with pytest.raises(ValueError):
        analyze(blank(), blank(), detections=detections)


@pytest.mark.parametrize("detection", [
    BoxDetection((-1, 0, 2, 2), 0.9), BoxDetection((0, 0, 11, 2), 0.9),
    BoxDetection((0, 0, 0, 2), 0.9), BoxDetection((0, 0, float("nan"), 2), 0.9),
    BoxDetection((0, 0, 2), 0.9), BoxDetection(None, 0.9),
    BoxDetection((0, 0, 2, 2), float("inf")), BoxDetection((0, 0, 2, 2), True),
    {"box": (0, 0, 2, 2), "score": 0.9},
])
def test_invalid_independent_boxes_are_not_silently_used_as_proxy(detection):
    with pytest.raises(ValueError):
        analyze(blank(), blank(), detections=[FrameDetections(0, 10, 8, [detection], ())])


def test_numpy_original_ids_and_detector_scalars_are_json_native():
    result = analyze(blank(), blank(), np.array([2], dtype=np.int64), detections=[
        FrameDetections(np.int64(2), np.int64(10), np.int64(8),
                        (BoxDetection(tuple(np.float32(x) for x in (0, 0, 2, 2)), np.float32(0.9)),), ())
    ])
    json.dumps(result, allow_nan=False)
    assert type(result["original_frame_indices"][0]) is int

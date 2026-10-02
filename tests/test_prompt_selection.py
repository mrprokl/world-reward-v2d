import json

import numpy as np
import pytest

from world_reward.prompt_selection import (
    BoxDetection, FrameDetections, non_maximum_suppression, select_seed_prompts,
)


PERSON = (10.0, 5.0, 75.0, 95.0)
OBJECT = (55.0, 35.0, 95.0, 65.0)
OTHER = (2.0, 2.0, 8.0, 8.0)


def detection(box=OBJECT, score=0.8):
    return BoxDetection(box, score)


def frame(index=0, *, persons=None, objects=None, width=100, height=100):
    return FrameDetections(
        index, width, height,
        [detection(PERSON, 0.9)] if persons is None else persons,
        [detection()] if objects is None else objects,
    )


def select(frames, **kwargs):
    return select_seed_prompts(frames, object_prompt="public metadata target", **kwargs)


def test_official_sam2_schema_identity_and_original_frame_index():
    result = select([frame(17)], total_frames=18)
    expected = {
        "prompts": [
            {"frame_index": 17, "object_id": 0, "points": None,
             "point_labels": None,
             "box": {"x0": 10.0, "y0": 5.0, "x1": 75.0, "y1": 95.0},
             "mask_path": None},
            {"frame_index": 17, "object_id": 1, "points": None,
             "point_labels": None,
             "box": {"x0": 55.0, "y0": 35.0, "x1": 95.0, "y1": 65.0},
             "mask_path": None},
        ]
    }
    assert result.frame_index == 17
    assert result.to_sam2_json() == expected
    assert json.loads(json.dumps(result.to_sam2_json(), allow_nan=False)) == expected
    assert result.confidence == 0.8


def test_exact_public_prompt_retained_without_label_rewriting():
    prompt = "  public Target / exact wording.  "
    result = select_seed_prompts([frame()], object_prompt=prompt)
    assert result.object_prompt == prompt


def test_pair_selection_does_not_mix_detections_across_frames():
    frames = [
        frame(0, persons=[detection(PERSON, 0.99)], objects=[detection(score=0.4)]),
        frame(20, persons=[detection(PERSON, 0.8)], objects=[detection(score=0.8)]),
        frame(40, persons=[detection(PERSON, 0.4)], objects=[detection(score=0.99)]),
    ]
    result = select(frames)
    assert result.frame_index == 20
    assert result.person.score == result.object.score == 0.8


def test_pair_tie_secondary_confidence_sum_then_earliest_index():
    low = frame(0, persons=[detection(PERSON, 0.8)], objects=[detection(score=0.8)])
    late = frame(9, persons=[detection(PERSON, 0.9)], objects=[detection(score=0.8)])
    early = frame(3, persons=[detection(PERSON, 0.9)], objects=[detection(score=0.8)])
    assert select([late, low, early]).frame_index == 3
    assert select([early, late, low]).frame_index == 3


def test_detection_order_independent_highest_confidence_not_first():
    candidates = [detection(OTHER, 0.4), detection(OBJECT, 0.9)]
    assert select([frame(objects=candidates)]).object.box == OBJECT
    assert select([frame(objects=list(reversed(candidates)))]).object.box == OBJECT


def test_duplicate_identical_boxes_are_one_hypothesis():
    result = select([frame(objects=[detection(score=0.8), detection(score=0.9)])])
    assert result.object.score == 0.9


@pytest.mark.parametrize("kind", ["persons", "objects"])
@pytest.mark.parametrize("second_score", [0.9, 0.86, 0.85])
def test_exact_or_near_ties_reject_frame_but_valid_other_frame_survives(kind, second_score):
    candidates = [detection(OBJECT, 0.9), detection(OTHER, second_score)]
    result = select([frame(2, **{kind: candidates}), frame(5)])
    assert result.frame_index == 5
    assert result.rejected_frames[0][0] == 2
    assert "ambiguous" in result.rejected_frames[0][1]


def test_zero_margin_still_rejects_exact_distinct_tie():
    with pytest.raises(ValueError, match="ambiguous"):
        select([frame(objects=[detection(score=0.9), detection(OTHER, 0.9)])],
               ambiguity_margin=0)


@pytest.mark.parametrize("bad_box", [
    (-1, 0, 50, 50), (0, -1, 50, 50), (0, 0, 101, 50), (0, 0, 50, 101),
    (20, 0, 10, 50), (10, 0, 10, 50), (0, 10, 50, 10),
    (0, 0, float("nan"), 50), (0, 0, float("inf"), 50),
    (0, 0, True, 50), (0, 0, "50", 50), (0, 0, 50), None,
])
def test_invalid_boxes_rejected_not_clamped_or_invented(bad_box):
    with pytest.raises(ValueError, match="No unambiguous valid"):
        select([frame(objects=[detection(bad_box, 0.99)])])


@pytest.mark.parametrize("bad_score", [
    -1, 1.01, float("nan"), float("inf"), True, "0.9", 0.29,
])
def test_invalid_or_low_confidence_scores_rejected(bad_score):
    with pytest.raises(ValueError, match="confidence threshold"):
        select([frame(objects=[detection(score=bad_score)])])


def test_invalid_high_score_does_not_shadow_valid_detection():
    result = select([frame(objects=[detection((-1, 0, 10, 10), 0.99), detection()])])
    assert result.object.box == OBJECT


def test_pixel_frame_boundary_and_exact_threshold_are_valid():
    result = select([frame(objects=[detection((0, 0, 100, 100), 0.3)])])
    assert result.object.box == (0, 0, 100, 100)
    assert result.object.score == 0.3


@pytest.mark.parametrize("frames", [[], [frame(persons=[])], [frame(objects=[])]])
def test_missing_inputs_fail_closed(frames):
    with pytest.raises(ValueError, match="No unambiguous valid"):
        select(frames)


def test_person_and_object_must_both_exist_on_same_frame():
    with pytest.raises(ValueError, match="No unambiguous valid"):
        select([frame(0, persons=[]), frame(1, objects=[])])


@pytest.mark.parametrize("kwargs", [
    {"confidence_threshold": -0.01}, {"confidence_threshold": 1.01},
    {"confidence_threshold": float("nan")}, {"confidence_threshold": True},
    {"ambiguity_margin": -0.01}, {"ambiguity_margin": float("inf")},
    {"ambiguity_margin": 1.01}, {"total_frames": 0}, {"total_frames": True},
])
def test_bad_configuration_fails_before_selection(kwargs):
    with pytest.raises(ValueError):
        select([frame()], **kwargs)


@pytest.mark.parametrize("prompt", [None, "", "  ", 4])
def test_no_invented_default_target_prompt(prompt):
    with pytest.raises(ValueError, match="object_prompt"):
        select_seed_prompts([frame()], object_prompt=prompt)


@pytest.mark.parametrize("bad_frame", [
    frame(-1), frame(0.5), frame(True), frame(width=0), frame(height=-1),
    frame(width=100.0), frame(height=True),
])
def test_invalid_frame_indices_and_image_dimensions_fail(bad_frame):
    with pytest.raises(ValueError):
        select([bad_frame])


def test_duplicate_or_out_of_clip_frame_indices_fail():
    with pytest.raises(ValueError, match="duplicate frame_index"):
        select([frame(3), frame(3)])
    with pytest.raises(ValueError, match="outside total_frames"):
        select([frame(3)], total_frames=3)


def test_numpy_detector_scalars_convert_to_json_native_numbers():
    boxes = tuple(np.float32(x) for x in OBJECT)
    result = select([frame(np.int64(2), width=np.int64(100),
                           objects=[detection(boxes, np.float32(0.8))])])
    json.dumps(result.to_sam2_json(), allow_nan=False)
    assert type(result.frame_index) is int
    assert all(type(x) is float for x in result.object.box)


def test_wrong_detection_or_frame_schema_is_explicit_error():
    with pytest.raises(ValueError, match="BoxDetection"):
        select([frame(objects=[{"box": OBJECT, "score": 0.9}])])
    with pytest.raises(ValueError, match="FrameDetections"):
        select([{"frame_index": 0}])


def test_missing_sequence_container_has_explicit_schema_error():
    with pytest.raises(ValueError, match="frames must be a sequence"):
        select(None)
    missing = FrameDetections(0, 100, 100, None, [])
    with pytest.raises(ValueError, match="detections must be a sequence"):
        select([missing])


def nms(detections, **kwargs):
    return non_maximum_suppression(detections, 100, 100, 0.3, **kwargs)


def test_nms_suppresses_high_iou_duplicate_before_unchanged_ambiguity_gate():
    near_duplicate = (11.0, 6.0, 74.0, 94.0)
    candidates = [detection(PERSON, 0.9), detection(near_duplicate, 0.89)]
    with pytest.raises(ValueError, match="ambiguous"):
        select([frame(persons=candidates)])
    kept = nms(candidates)
    assert kept == (detection(PERSON, 0.9),)
    assert select([frame(persons=kept)]).person.box == PERSON


def test_nms_keeps_genuine_separated_near_tied_instances_ambiguous():
    candidates = [detection(OBJECT, 0.9), detection(OTHER, 0.89)]
    kept = nms(candidates)
    assert len(kept) == 2
    with pytest.raises(ValueError, match="ambiguous"):
        select([frame(objects=kept)])


def test_nms_does_not_merge_giant_enclosing_box_with_small_instance():
    enclosing = (0.0, 0.0, 100.0, 100.0)
    candidates = [detection(enclosing, 0.9), detection(OTHER, 0.89)]
    assert nms(candidates) == tuple(candidates)
    with pytest.raises(ValueError, match="ambiguous"):
        select([frame(objects=nms(candidates))])


def test_nms_equal_scores_use_box_tie_break_not_input_order():
    first = detection((10, 10, 80, 80), 0.9)
    second = detection((11, 11, 81, 81), 0.9)
    assert nms([first, second]) == nms([second, first]) == (first,)


def test_nms_is_greedy_against_retained_boxes_not_suppressed_boxes():
    # Adjacent overlaps exceed 0.3; first and last overlap less. The suppressed
    # middle box must not transitively suppress the third distinct hypothesis.
    a = detection((0, 0, 40, 40), 0.9)
    b = detection((20, 0, 60, 40), 0.8)
    c = detection((40, 0, 80, 40), 0.7)
    assert nms([c, b, a], iou_threshold=0.3) == (a, c)


def test_nms_threshold_zero_keeps_disjoint_and_touching_boxes():
    a = detection((0, 0, 10, 10), 0.9)
    overlap = detection((9, 0, 20, 10), 0.8)
    touching = detection((10, 0, 20, 10), 0.7)
    separate = detection((50, 50, 60, 60), 0.6)
    assert nms([separate, overlap, touching, a], iou_threshold=0) == (a, touching, separate)


def test_nms_threshold_one_still_exact_deduplicates_but_keeps_distinct_boxes():
    a = detection(PERSON, 0.9)
    b = detection((11, 6, 74, 94), 0.89)
    assert nms([detection(PERSON, 0.8), b, a], iou_threshold=1) == (a, b)


def test_nms_equal_iou_threshold_not_suppressed():
    a = detection((0, 0, 10, 10), 0.9)
    b = detection((0, 0, 5, 10), 0.8)
    assert nms([b, a], iou_threshold=0.5) == (a, b)


@pytest.mark.parametrize("bad_detection", [
    detection((-1, 0, 10, 10), 0.99),
    detection((0, 0, 101, 10), 0.99),
    detection((0, 0, 10, 101), 0.99),
    detection((0, 0, 0, 10), 0.99),
    detection((0, 0, float("nan"), 10), 0.99),
    detection((0, 0, float("inf"), 10), 0.99),
    detection((0, 0, 10), 0.99),
    detection(None, 0.99),
    detection(score=1.1), detection(score=-1),
    detection(score=float("nan")), detection(score=float("inf")),
    detection(score=True), detection(score=0.29),
])
def test_nms_invalids_are_dropped_without_shadowing_good_box(bad_detection):
    good = detection(OTHER, 0.8)
    assert nms([bad_detection, good]) == (good,)


def test_nms_empty_or_all_invalid_returns_no_invented_boxes():
    assert nms([]) == ()
    assert nms([detection(score=0.2)]) == ()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.7"])
def test_nms_invalid_iou_threshold_is_explicit_error(threshold):
    with pytest.raises(ValueError, match="iou_threshold"):
        nms([detection()], iou_threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_nms_invalid_confidence_threshold_is_explicit_error(threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        non_maximum_suppression([], 100, 100, threshold)


@pytest.mark.parametrize("dimensions", [(0, 100), (100, -1), (100.0, 100), (True, 100)])
def test_nms_invalid_image_dimensions_are_explicit_error(dimensions):
    with pytest.raises(ValueError):
        non_maximum_suppression([], *dimensions, 0.3)


def test_nms_bad_schema_is_explicit_error():
    with pytest.raises(ValueError, match="sequence"):
        nms(None)
    with pytest.raises(ValueError, match="BoxDetection"):
        nms([{"box": OTHER, "score": 0.9}])


def test_nms_returns_native_json_numbers_for_numpy_detector_scalars():
    kept = nms([detection(tuple(np.float32(x) for x in PERSON), np.float32(0.9))])
    assert type(kept[0].score) is float
    assert all(type(x) is float for x in kept[0].box)


def test_nms_finite_large_coordinates_do_not_overflow_iou_area():
    a = detection((0.0, 0.0, 1e200, 1e200), 0.9)
    b = detection((0.0, 0.0, 9e199, 9e199), 0.8)
    assert non_maximum_suppression([a, b], 10**201, 10**201, 0.3) == (a,)

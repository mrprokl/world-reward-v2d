"""Stub model callbacks only; no real DINO/SAM2 execution or media assets."""
import numpy as np
import pytest

from world_reward.automatic_candidate_bank import build_automatic_candidate_bank


class Predictor:
    def __init__(self, output=None):
        self.output, self.images, self.calls = output, [], []

    def set_image(self, image):
        self.images.append(image)

    def predict(self, *, box, multimask_output):
        self.calls.append((box.copy(), multimask_output))
        if self.output is not None:
            return self.output
        h, w = self.images[0].shape[:2]; count = len(box)
        masks = np.ones((count, 1, h, w), dtype=np.float32)
        scores = np.arange(count, dtype=np.float32) - .2  # Not a calibrated [0,1] probability.
        low = np.zeros((count, 1, 256, 256), dtype=np.float32)
        return (masks[0], scores.reshape(count, 1)[0], low[0]) if count == 1 else (masks, scores[:, None], low)


def detections(boxes=(), scores=None):
    raw_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    raw_scores = np.full(len(raw_boxes), .8) if scores is None else np.asarray(scores, dtype=np.float32)
    calls = []
    def detector(rgb, query):
        calls.append((rgb, query))
        return raw_boxes, raw_scores
    return detector, calls, raw_boxes, raw_scores


def build(hand_boxes=(), object_boxes=(), *, predictor=None, image=None, hand_scores=None, object_scores=None):
    hand, hc, hb, hs = detections(hand_boxes, hand_scores)
    obj, oc, ob, os = detections(object_boxes, object_scores)
    predictor = Predictor() if predictor is None else predictor
    image = np.zeros((16, 24, 3), dtype=np.uint8) if image is None else image
    result = build_automatic_candidate_bank(image, hand, obj, predictor)
    return result, predictor, (hc, oc), (hb, hs, ob, os)


def test_one_encoder_one_batch_preserves_all_stably_ordered_candidates_and_raw_scores():
    bank, model, calls, raw = build(hand_boxes=((12, 0, 20, 8), (0, 0, 8, 8)),
                                  object_boxes=((8, 8, 16, 16),))
    assert bank.image_size == (16, 24)
    assert len(model.images) == len(model.calls) == 1
    assert len(calls[0]) == len(calls[1]) == 1
    assert calls[0][0][1] == "hand." and calls[1][0][1] == "object."
    assert model.images[0] is calls[0][0][0] is calls[1][0][0]
    assert model.calls[0][0].dtype == np.float32 and model.calls[0][1] is False
    assert [c.kind for c in bank.candidates] == ["hand", "hand", "object"]
    assert [c.stable_id for c in bank.candidates] == ["hand:000000", "hand:000001", "object:000002"]
    np.testing.assert_array_equal(model.calls[0][0], [[0, 0, 8, 8], [12, 0, 20, 8], [8, 8, 16, 16]])
    assert [c.sam2_score for c in bank.candidates] == pytest.approx([-.2, .8, 1.8])
    for record, boxes, scores in zip(bank.detector_records, raw[::2], raw[1::2]):
        np.testing.assert_array_equal(record.boxes, boxes)
        np.testing.assert_array_equal(record.scores, scores)
        assert record.boxes.dtype == boxes.dtype and record.scores.dtype == scores.dtype


@pytest.mark.parametrize("count", [1, 2])
def test_actual_native_singleton_and_batch_output_shapes(count):
    boxes = ((0, 0, 8, 8), (12, 0, 20, 8))[:count]
    bank, _, _, _ = build(hand_boxes=boxes)
    assert len(bank.candidates) == count
    assert all(c.mask.dtype == np.bool_ and c.mask.shape == (16, 24) for c in bank.candidates)
    assert all(c.query_points.shape == (16, 3) for c in bank.candidates)


def test_native_boolean_mask_allowed_and_empty_mask_retained_without_query_placeholder():
    output = (np.zeros((1, 16, 24), dtype=bool), np.array([4.2]), None)
    bank, _, _, _ = build(object_boxes=((0, 0, 8, 8),), predictor=Predictor(output))
    candidate = bank.candidates[0]
    assert candidate.kind == "object" and candidate.sam2_score == 4.2
    assert not candidate.mask.any() and candidate.query_points.shape == (0, 3)
    assert candidate.query_points.dtype == np.float64
    assert bank.background_query_points.shape == (64, 3)


def test_empty_detector_bank_keeps_raw_records_and_skips_sam2():
    bank, predictor, _, _ = build()
    assert bank.candidates == () and len(bank.detector_records) == 2
    assert predictor.images == predictor.calls == []
    assert bank.background_query_points.shape == (64, 3)
    for record in bank.detector_records:
        assert record.boxes.shape == (0, 4) and record.scores.shape == (0,)


def test_classwise_nms_preserves_cross_class_overlap_and_all_raw_rejected_records():
    bank, _, _, _ = build(hand_boxes=((0, 0, 8, 8), (1, 0, 8, 8), (-1, 0, 8, 8), (12, 0, 20, 8)),
                         hand_scores=[.9, .8, .99, .29], object_boxes=((0, 0, 8, 8),))
    assert [(c.kind, c.box.tolist()) for c in bank.candidates] == [("hand", [0, 0, 8, 8]), ("object", [0, 0, 8, 8])]
    assert len(bank.detector_records[0].boxes) == 4
    np.testing.assert_array_equal(bank.detector_records[0].boxes[2], [-1, 0, 8, 8])
    assert bank.detector_records[0].scores[3] < .3


def test_detector_permutation_and_ties_do_not_select_one_instance_or_change_ids():
    boxes = ((12, 0, 20, 8), (0, 0, 8, 8))
    a, _, _, _ = build(hand_boxes=boxes)
    b, _, _, _ = build(hand_boxes=tuple(reversed(boxes)))
    assert [(c.stable_id, c.box.tolist()) for c in a.candidates] == [(c.stable_id, c.box.tolist()) for c in b.candidates]
    assert len(a.candidates) == 2  # Equal score is not an actor acceptance decision.


def test_fixed_bbox_grid_uses_mask_membership_original_continuous_pixel_centres_no_refill():
    mask = np.zeros((1, 16, 24), dtype=np.float32)
    mask[0, 1, 1] = 1; mask[0, 7, 7] = 1; mask[0, 2, 2] = 1  # Last pixel is not on the 4x4 grid.
    bank, _, _, _ = build(hand_boxes=((0, 0, 8, 8),), predictor=Predictor((mask, np.array([.6]), None)))
    np.testing.assert_array_equal(bank.candidates[0].query_points, [[0, 1.5, 1.5], [0, 7.5, 7.5]])
    for _, y, x in bank.candidates[0].query_points:
        assert bank.candidates[0].mask[int(np.floor(y)), int(np.floor(x))]
    assert len(bank.candidates[0].query_points) == 2


def test_fractional_boxes_tiny_grid_dedup_and_full_image_background_coordinates():
    bank, _, _, _ = build(hand_boxes=((.1, .2, .9, .8),), image=np.zeros((1, 1, 3), dtype=np.uint8))
    np.testing.assert_array_equal(bank.candidates[0].query_points, [[0, .5, .5]])
    assert bank.background_query_points.shape == (0, 3)
    empty, _, _, _ = build(image=np.zeros((1, 1, 3), dtype=np.uint8))
    np.testing.assert_array_equal(empty.background_query_points, [[0, .5, .5]])


def test_background_grid_excludes_union_of_all_masks_not_only_a_selected_object():
    masks = np.zeros((2, 1, 16, 24), dtype=np.float32)
    masks[0, 0, :8] = 1; masks[1, 0, :, :12] = 1
    bank, _, _, _ = build(hand_boxes=((0, 0, 8, 8),), object_boxes=((12, 0, 20, 8),),
                         predictor=Predictor((masks, np.ones((2, 1)), None)))
    queries = bank.background_query_points
    assert len(queries) == 16
    assert np.all(queries[:, 1] >= 8) and np.all(queries[:, 2] >= 12)


def test_results_are_owned_readonly_without_aliasing_model_or_rgb_arrays():
    image = np.full((16, 24, 3), 7, dtype=np.uint8)
    masks, scores = np.ones((1, 16, 24), dtype=np.float32), np.array([.7])
    bank, predictor, _, raw = build(hand_boxes=((0, 0, 8, 8),), image=image,
                                  predictor=Predictor((masks, scores, None)))
    arrays = [bank.background_query_points]
    for record in bank.detector_records:
        arrays.extend((record.boxes, record.scores))
    for candidate in bank.candidates:
        arrays.extend((candidate.box, candidate.mask, candidate.query_points))
    for value in arrays:
        assert value.flags.owndata and not value.flags.writeable
        assert not np.shares_memory(value, image) and not np.shares_memory(value, masks)
        if value.size:
            with pytest.raises(ValueError, match="read-only"):
                value.flat[0] = 99
    np.testing.assert_array_equal(image, 7)
    assert not predictor.images[0].flags.writeable
    assert not np.shares_memory(predictor.images[0], image)
    masks[:] = 0; scores[:] = 99; raw[0][:] = 0
    assert bank.candidates[0].mask.all() and bank.candidates[0].sam2_score == .7
    np.testing.assert_array_equal(bank.candidates[0].box, [0, 0, 8, 8])


@pytest.mark.parametrize("rgb", [np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8),
                                   np.zeros((4, 4, 4), dtype=np.uint8), np.zeros((4, 4, 3)),
                                   np.ma.array(np.zeros((4, 4, 3), dtype=np.uint8))])
def test_invalid_rgb_fails_before_any_model_call(rgb):
    with pytest.raises(ValueError, match="RGB"):
        build(image=rgb)


@pytest.mark.parametrize("queries", [("person.", "object."), ("hand.", "bottle."), ("hand", "object."), (True, "object.")])
def test_queries_are_fixed_no_per_clip_labels(queries):
    hand, _, _, _ = detections(); obj, _, _, _ = detections()
    with pytest.raises(ValueError, match="query contracts"):
        build_automatic_candidate_bank(np.zeros((4, 4, 3), np.uint8), hand, obj, Predictor(),
                                       hand_query=queries[0], object_query=queries[1])


@pytest.mark.parametrize("output", [[], (np.zeros((4,)), np.ones(1)), (np.zeros((1, 4)), np.ones(2)),
                                       (np.full((1, 4), np.nan), np.ones(1)),
                                       (np.zeros((1, 4)), np.array([True])),
                                       (np.ma.array(np.zeros((1, 4))), np.ones(1))])
def test_bad_detector_contracts_fail(output):
    with pytest.raises(ValueError):
        build_automatic_candidate_bank(np.zeros((4, 4, 3), np.uint8), lambda *_: output,
                                       lambda *_: (np.empty((0, 4)), np.empty(0)), Predictor())


@pytest.mark.parametrize("output", [(np.zeros((1, 1, 16, 24)), np.ones((1, 1)), None),
                                       (np.zeros((1, 16, 24)), np.ones((1, 1)), None),
                                       (np.full((1, 16, 24), .5), np.ones(1), None),
                                       (np.zeros((1, 16, 24)), np.array([np.inf]), None),
                                       (np.zeros((1, 16, 24)), np.ones(1), np.zeros((1, 128, 128))),
                                       [np.zeros((1, 16, 24)), np.ones(1), None]])
def test_bad_sam2_shapes_logits_or_nonfinite_outputs_fail_without_reinterpretation(output):
    with pytest.raises(ValueError):
        build(hand_boxes=((0, 0, 8, 8),), predictor=Predictor(output))


def test_model_errors_propagate_and_no_fake_segmentation_fallback_occurs():
    def broken(*_): raise RuntimeError("actual detector failed")
    with pytest.raises(RuntimeError, match="actual detector failed"):
        build_automatic_candidate_bank(np.zeros((4, 4, 3), np.uint8), broken, broken, Predictor())
    with pytest.raises(ValueError, match="SAM2 predictor"):
        build(hand_boxes=((0, 0, 8, 8),), predictor=object())

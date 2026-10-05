"""Tiny analytic controls only; no native model, real RGB or labels."""
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import itertools

import numpy as np
import pytest

from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.hoi_mask_bridge import bridge_hoi_masks
from world_reward.point_mask_association import MaskAnchor


def observations(boxes, *, frame=7, grid=(3, 4), query_ids=None, classes=None):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(boxes)
    classes = np.asarray(np.arange(n) % 3 if classes is None else classes, np.int64)
    scores = np.full(n, .75, np.float32)
    pairs = []
    for left_role, right_role in ((0, 1), (1, 2)):
        left, right = np.flatnonzero(classes == left_role), np.flatnonzero(classes == right_role)
        pair = np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64)
        pairs.extend((pair, np.zeros((len(pair), 2), np.float32)))
    return HOIDetrObservations(frame, grid, np.zeros((1500, 3), np.float32),
        np.zeros((1500, 4), np.float32), np.zeros((1500, 256), np.float32),
        np.column_stack((boxes, scores)), np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64), np.asarray(np.arange(n) if query_ids is None else query_ids, np.int64),
        classes, boxes, scores, scores, *pairs)


def anchor(masks, *, frame=7, ids=None, grid=(3, 4)):
    masks = np.asarray(masks, bool).reshape(-1, *grid)
    return MaskAnchor(frame, tuple(f"mask{i}" for i in range(len(masks))) if ids is None else ids, masks)


def fraction_reference(boxes, masks):
    """Independent scalar rational cell integration, no NumPy overlap formula."""
    answer = np.zeros((len(boxes), len(masks)), np.float64)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = map(lambda x: Fraction(float(x)), box)
        for j, mask in enumerate(masks):
            total = Fraction(0)
            for y, x in itertools.product(range(mask.shape[0]), range(mask.shape[1])):
                if mask[y, x]:
                    width = max(Fraction(0), min(x + 1, x1) - max(x, x0))
                    height = max(Fraction(0), min(y + 1, y1) - max(y, y0))
                    total += width * height
            answer[i, j] = float(total)
    return answer


def test_fractional_cells_not_centres_and_all_routes_including_null():
    masks = np.zeros((3, 3, 4), bool)
    masks[0, 0, 0] = masks[1, 0, 1] = True
    masks[1, 0, 0] = True
    o = observations([[.25, .25, 1.25, .75], [1., 0., 1., 2.]])
    result = bridge_hoi_masks(o, anchor(masks))
    np.testing.assert_array_equal(result.intersection_area, [[.375, .5, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result.detector_box_area, [.5, 0])
    np.testing.assert_array_equal(result.mask_area, [1, 2, 0])
    np.testing.assert_array_equal(result.box_iou[0], [1/3, .25, 0])
    np.testing.assert_array_equal(result.mask_coverage[0], [.375, .25, 0])
    np.testing.assert_array_equal(result.detection_indices, [0, 0, 0, 1])
    np.testing.assert_array_equal(result.mask_indices, [0, 1, -1, -1])


def test_negative_outside_boxes_keep_full_area_and_boundary_only_is_null():
    masks = np.ones((1, 3, 4), bool)
    result = bridge_hoi_masks(observations([[-2, -1, 2, 1], [-4, -3, 0, 0], [4, 0, 5, 1]]), anchor(masks))
    np.testing.assert_array_equal(result.intersection_area[:, 0], [2, 0, 0])
    np.testing.assert_array_equal(result.detector_box_area, [8, 12, 1])
    assert result.box_iou[0, 0] == 2 / 18
    np.testing.assert_array_equal(result.mask_indices, [0, -1, -1, -1])


@pytest.mark.parametrize("detections,masks", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_empty_banks_and_masks_preserved(detections, masks):
    o = observations(np.zeros((detections, 4), np.float32))
    a = anchor(np.zeros((masks, 3, 4), bool))
    result = bridge_hoi_masks(o, a)
    assert result.intersection_area.shape == (detections, masks)
    assert len(result.mask_ids) == masks
    np.testing.assert_array_equal(result.detection_indices, np.arange(detections))
    np.testing.assert_array_equal(result.mask_indices, np.full(detections, -1))
    assert not result.supported.any() and not result.box_iou.any()


def test_duplicate_query_across_roles_and_proposal_slots_not_collapsed():
    o = observations([[0, 0, 1, 1]] * 3, query_ids=[12, 12, 12], classes=[0, 1, 2])
    r = bridge_hoi_masks(o, anchor(np.ones((1, 3, 4), bool)))
    np.testing.assert_array_equal(r.query_ids, [12, 12, 12])
    np.testing.assert_array_equal(r.class_ids, [0, 1, 2])
    np.testing.assert_array_equal(r.proposal_slots, [0, 1, 2])
    np.testing.assert_array_equal(r.retained_nms_positions, o.retained_nms_positions)
    np.testing.assert_array_equal(r.detection_indices, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(r.mask_indices, [0, -1] * 3)


def test_independent_rational_reference_dyadic_and_random_masks():
    rng = np.random.default_rng(8401)
    raw = rng.integers(-8, 25, (30, 4)).astype(np.float32) / 8
    boxes = np.column_stack((np.minimum(raw[:, 0], raw[:, 2]), np.minimum(raw[:, 1], raw[:, 3]),
                             np.maximum(raw[:, 0], raw[:, 2]), np.maximum(raw[:, 1], raw[:, 3])))
    a = anchor(rng.integers(0, 2, (4, 3, 4)).astype(bool))
    r = bridge_hoi_masks(observations(boxes), a)
    np.testing.assert_array_equal(r.intersection_area, fraction_reference(boxes, a.masks))
    np.testing.assert_array_equal(r.supported, r.intersection_area > 0)


def test_extreme_f32_boxes_subnormal_signed_zero_remain_finite():
    maximum = np.finfo(np.float32).max
    tiny = np.nextafter(np.float32(0), np.float32(1))
    o = observations([[-maximum, -maximum, maximum, maximum], [0, -0., tiny, tiny],
                      [maximum / 2, 0, maximum, 1]])
    r = bridge_hoi_masks(o, anchor(np.ones((1, 3, 4), bool)))
    assert np.isfinite(r.detector_box_area).all() and np.isfinite(r.box_iou).all()
    np.testing.assert_array_equal(r.intersection_area[:, 0], [12, float(tiny)**2, 0])
    assert r.supported[:, 0].tolist() == [True, True, False]
    assert np.signbit(r.boxes_original_xyxy[1, 1])


def test_permutations_preserve_keyed_correspondences_not_invent_identity():
    masks = np.zeros((2, 3, 4), bool); masks[0, 0, 0] = True; masks[1, 1, 2] = True
    boxes = [[0, 0, 1, 1], [2, 1, 3, 2]]
    a = anchor(masks, ids=("z", "a"))
    b = anchor(masks[::-1], ids=("a", "z"))
    r = bridge_hoi_masks(observations(boxes, query_ids=[8, 4], classes=[0, 1]), a)
    s = bridge_hoi_masks(observations(boxes[::-1], query_ids=[4, 8], classes=[1, 0]), b)
    assert r.mask_ids == s.mask_ids == ("a", "z")
    np.testing.assert_array_equal(r.intersection_area, s.intersection_area[::-1])
    np.testing.assert_array_equal(r.query_ids, s.query_ids[::-1])


@pytest.mark.parametrize("bad", ["inverted_x", "inverted_y", "nan", "inf", "frame", "grid", "dtype", "type"])
def test_invalid_contracts_fail(bad):
    o, a = observations([[0, 0, 1, 1]]), anchor(np.ones((1, 3, 4), bool))
    if bad.startswith("inverted"):
        box = [[1, 0, 0, 1]] if bad == "inverted_x" else [[0, 1, 1, 0]]
        o = replace(o, boxes_original_xyxy=np.asarray(box, np.float32))
    elif bad in ("nan", "inf", "dtype"):
        # HOI constructor already rejects these; bridge also checks supplied boxes.
        box = np.array([[0, 0, 1, 1]], np.float64 if bad == "dtype" else np.float32)
        if bad != "dtype": box[0, 0] = np.nan if bad == "nan" else np.inf
        object.__setattr__(o, "boxes_original_xyxy", box)
    elif bad == "frame": a = anchor(np.ones((1, 3, 4), bool), frame=8)
    elif bad == "grid": a = anchor(np.ones((1, 2, 4), bool), grid=(2, 4))
    else: o = object()
    with pytest.raises(ValueError): bridge_hoi_masks(o, a)


def test_outputs_immutable_owned_and_source_bytes_unchanged():
    o, a = observations([[.25, .25, 1.25, 1.25]]), anchor(np.ones((2, 3, 4), bool))
    before = {f.name: getattr(o, f.name).tobytes() for f in fields(o) if isinstance(getattr(o, f.name), np.ndarray)}
    mask_bytes = a.masks.tobytes()
    r = bridge_hoi_masks(o, a)
    for f in fields(r):
        v = getattr(r, f.name)
        if isinstance(v, np.ndarray):
            assert not v.flags.writeable
            with pytest.raises(ValueError): v.flags.writeable = True
            assert not np.shares_memory(v, o.boxes_original_xyxy)
            assert not np.shares_memory(v, a.masks)
    with pytest.raises(FrozenInstanceError): r.image_size = (1, 1)
    assert a.masks.tobytes() == mask_bytes
    assert before == {f.name: getattr(o, f.name).tobytes() for f in fields(o) if isinstance(getattr(o, f.name), np.ndarray)}
    o.boxes_original_xyxy.flags.writeable = True
    o.boxes_original_xyxy[:] = 0
    assert r.detector_box_area[0] == 1 and r.boxes_original_xyxy[0, 0] == .25

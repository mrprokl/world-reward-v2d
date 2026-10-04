"""Manufactured private-array diagnostics; no annotation/media/model acquisition."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from world_reward.hand_evaluation import (
    VERIFIED_JOINT_INDICES, evaluate_hand_clip, pool_hand_evaluations,
)
from world_reward.hand_observations import HandInstances, HandObservations, LandmarkEvidence


def hand(offset=0.):
    return np.column_stack((np.linspace(3., 7., 21), np.linspace(4., 8., 21))) + offset


def observations(bank, supports=None):
    frames = []
    for t, points in enumerate(bank):
        xy = np.asarray(points, np.float64).reshape(-1, 21, 2)
        support = np.isfinite(xy).all(axis=-1) if supports is None else supports[t]
        frames.append(HandInstances(LandmarkEvidence(xy, support, "original_image", "pixel")))
    return HandObservations(np.arange(len(frames), dtype=np.int64), (20, 30), tuple(frames),
                            "synthetic automatic", {"source": "synthetic-only"})


def labels(frames, positive=None):
    segments = [np.zeros((20, 30), np.uint8) for _ in range(frames)]
    for t in range(frames) if positive is None else positive:
        segments[t][3:10, 2:10] = 255
    return segments, np.tile(hand()[None], (frames, 1, 1))


def evaluate(bank, *, positive=None, gt=None, supports=None, clip="clip"):
    obs = observations(bank, supports); seg, joints = labels(len(bank), positive)
    return evaluate_hand_clip(clip, obs, obs.frame_index, seg, joints if gt is None else gt)


def test_unique_positive_error_and_original_slots_no_alignment():
    result = evaluate([[hand(1.), hand(15.)]])
    assert result.diagnostic_status == "pass" and result.matched_prediction.tolist() == [0]
    assert result.counts.scored_joints == 17 and result.counts.valid_gt_joints == 17
    assert result.joint_weighted_epe_pixels == pytest.approx(np.sqrt(2))
    assert result.frame_epe_pixels == (pytest.approx(np.sqrt(2)),)
    assert result.counts.other_predictions == 1 and result.counts.returned_predictions == 2


def test_no_predictions_positive_fails_with_nullable_error_not_zero():
    result = evaluate([[]])
    assert result.diagnostic_status == "fail" and result.missed.tolist() == [True]
    assert result.frame_epe_pixels == (None,) and result.joint_weighted_epe_pixels is None
    assert result.counts.scored_joints == 0 and result.counts.valid_gt_joints == 17


def test_two_overlaps_ambiguous_never_choose_lowest_gt_error():
    result = evaluate([[hand(), hand(1.)]])
    assert result.ambiguous.tolist() == [True] and result.matched_prediction.tolist() == [-1]
    assert result.diagnostic_status == "fail" and result.frame_epe_pixels == (None,)
    assert result.counts.other_predictions == 2


def test_original_early_absent_frames_preserved_not_required_frame0():
    gt = np.tile(hand()[None], (3, 1, 1)); gt[0] = -1
    result = evaluate([[], [hand()], []], positive=[1], gt=gt)
    assert result.frame_index.tolist() == [0, 1, 2] and result.diagnostic_status == "pass"
    assert result.unlabelled.tolist() == [True, False, False]
    assert result.annotation_without_mask.tolist() == [False, False, True]
    assert result.frame_epe_pixels == (None, 0., None)


def test_empty_seg_and_sentinel_is_unlabelled_not_certified_negative():
    result = evaluate([[hand()]], positive=[], gt=np.full((1, 21, 2), -1.))
    assert result.diagnostic_status == "inconclusive" and result.counts.unlabelled_frames == 1
    assert result.counts.other_predictions == 1 and result.counts.annotated_positive_frames == 0
    assert not hasattr(result.counts, "false_positives") and not hasattr(result, "absence_recall")


def test_partial_invalid_gt_joints_independent_scored_denominator():
    gt = hand()[None]; gt[0, 0] = -1; gt[0, 5] = np.nan; gt[0, 6, 0] = 30
    result = evaluate([[hand(1.)]], gt=gt)
    assert result.counts.valid_gt_joints == result.counts.scored_joints == 14
    assert result.valid_gt_joints[0, :3].tolist() == [False, False, False]
    assert result.joint_error_pixels[0][:3] == (None, None, None)
    assert result.joint_weighted_epe_pixels == pytest.approx(np.sqrt(2))


def test_unique_box_without_valid_annotations_does_not_invent_error():
    result = evaluate([[hand()]], gt=np.full((1, 21, 2), -1.))
    assert result.diagnostic_status == "pass" and result.counts.uniquely_associated_frames == 1
    assert result.counts.positive_frames_without_valid_joints == 1 and result.frame_epe_pixels == (None,)


def test_thumb_joint_errors_excluded_but_raw_box_uses_all21():
    native = hand(); native[1:5] += 10.
    gt = hand()[None]; gt[0, 1:5] = np.nan
    result = evaluate([[native]], gt=gt)
    assert VERIFIED_JOINT_INDICES == (0, *range(5, 21))
    assert result.counts.scored_joints == 17 and result.joint_weighted_epe_pixels == 0.


@pytest.mark.parametrize("kind", ["nonfinite_thumb", "unsupported_finite_thumb"])
def test_all21_finite_and_supported_required_for_native_box(kind):
    native = hand(); support = np.ones((1, 21), bool)
    if kind == "nonfinite_thumb": native[1, 0] = np.nan
    support[0, 1] = False
    result = evaluate([[native]], supports=[support])
    assert result.diagnostic_status == "fail" and result.counts.unavailable_prediction_boxes == 1
    assert result.counts.returned_predictions == 1


def test_outside_native_xy_retained_in_error_and_box_no_clipping():
    native = hand(); native[5, 0] = 100.
    result = evaluate([[native]])
    assert result.counts.outside_prediction_joints == 1
    assert result.joint_error_pixels[0][1] == pytest.approx(100.-hand()[5, 0])


def test_boundary_touch_and_degenerate_box_not_positive_area():
    for native in (np.full((21, 2), 5.), hand(7.)):
        result = evaluate([[native]])
        assert result.missed.tolist() == [True] and result.diagnostic_status == "fail"


def test_single_hand_pixel_has_pixel_cell_area_not_zero_gt_box():
    obs = observations([[hand()]]); seg, gt = labels(1); seg[0][:] = 0; seg[0][5, 5] = 255
    result = evaluate_hand_clip("clip", obs, obs.frame_index, iter(seg), gt)
    assert result.unique_association.tolist() == [True]


def test_three_clip_pool_joint_weighted_vs_frame_mean_and_counts():
    a = evaluate([[hand()]], clip="a")
    gt = hand()[None]; gt[0, list(VERIFIED_JOINT_INDICES[1:])] = -1
    b = evaluate([[hand(1.)]], gt=gt, clip="b")
    c = evaluate([[hand(3.)]], clip="c")
    pooled = pool_hand_evaluations((a, b, c))
    assert pooled.counts.scored_joints == 35 and pooled.counts.scored_frames == 3
    assert pooled.joint_weighted_epe_pixels == pytest.approx(52.*np.sqrt(2)/35)
    assert pooled.frame_mean_epe_pixels == pytest.approx(4.*np.sqrt(2)/3)
    assert pooled.joint_weighted_epe_pixels != pytest.approx(pooled.frame_mean_epe_pixels)
    assert pooled.diagnostic_status == "pass" and pooled.clips == (a, b, c)


def test_pool_does_not_hide_a_failed_or_unlabelled_clip():
    good = evaluate([[hand()]], clip="good")
    failed = evaluate([[]], clip="failed")
    empty = evaluate([[]], positive=[], gt=np.full((1, 21, 2), -1.), clip="unlabelled")
    assert pool_hand_evaluations((good, failed)).diagnostic_status == "fail"
    assert pool_hand_evaluations((good, empty)).diagnostic_status == "inconclusive"
    assert pool_hand_evaluations((empty,)).joint_weighted_epe_pixels is None
    with pytest.raises(ValueError): pool_hand_evaluations((good, good))


def test_results_owned_readonly_and_no_input_edits():
    obs = observations([[hand()]]); seg, gt = labels(1); indices = np.array([0], np.int64)
    original = gt.copy(); result = evaluate_hand_clip("clip", obs, indices, seg, gt)
    assert np.array_equal(gt, original) and not np.shares_memory(indices, result.frame_index)
    for name in ("frame_index", "annotated_positive", "valid_gt_joints", "matched_prediction", "outside_joint_count"):
        with pytest.raises(ValueError): getattr(result, name).flat[0] = 9
    with pytest.raises(FrozenInstanceError): result.clip = "changed"
    gt[:] = -1; seg[0][:] = 0; indices[:] = 8
    assert result.counts.scored_joints == 17 and result.frame_index.tolist() == [0]


def test_native_permutation_preserves_association_error_not_slot_identity():
    a = evaluate([[hand(1.), hand(15.)]])
    b = evaluate([[hand(15.), hand(1.)]])
    assert a.matched_prediction.tolist() == [0] and b.matched_prediction.tolist() == [1]
    assert a.joint_error_pixels == b.joint_error_pixels and a.counts == b.counts
    c = evaluate([[hand(1.), hand()]])
    d = evaluate([[hand(), hand(1.)]])
    assert c.ambiguous.tolist() == d.ambiguous.tolist() == [True]
    assert c.frame_epe_pixels == d.frame_epe_pixels == (None,)


@pytest.mark.parametrize("fault", ["indices", "index_dtype", "seg_short", "seg_long", "seg_dtype", "seg_shape",
                                  "gt_projection", "gt_dtype", "masked_gt", "clip"])
def test_invalid_full_t_grid_and_canonical_contract(fault):
    obs = observations([[hand()]]); seg, gt = labels(1); indices = np.array([0], np.int64); clip = "clip"
    if fault == "indices": indices[:] = 1
    elif fault == "index_dtype": indices = indices.astype(np.int32)
    elif fault == "seg_short": seg = []
    elif fault == "seg_long": seg += seg
    elif fault == "seg_dtype": seg = [seg[0].astype(np.int64)]
    elif fault == "seg_shape": seg = [seg[0][:-1]]
    elif fault == "gt_projection": gt = gt[:, None]
    elif fault == "gt_dtype": gt = gt.astype(np.int64)
    elif fault == "masked_gt": gt = np.ma.array(gt, mask=False)
    else: clip = ""
    with pytest.raises(ValueError): evaluate_hand_clip(clip, obs, indices, seg, gt)

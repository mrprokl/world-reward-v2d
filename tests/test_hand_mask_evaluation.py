"""Manufactured paired masks only; no image/model/private-data acquisition."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from world_reward.hand_mask_evaluation import evaluate_hand_masks
from world_reward.hand_mask_proposals import propose_hand_masks
from world_reward.hand_observations import HandInstances, LandmarkEvidence


def readonly(value):
    value = np.array(value, copy=True);value.flags.writeable = False;return value


def proposal(index, a, b=None, *, support=None, reuse=None):
    a = np.asarray(a, bool).reshape(-1, 4, 5);b = a if b is None else np.asarray(b, bool).reshape(a.shape)
    n = len(a);xy = np.tile(np.linspace(.5, 2.5, 21)[None, :, None], (n, 1, 2))
    hands = HandInstances(LandmarkEvidence(xy, np.ones((n, 21), bool), "original_image", "pixel"))
    class Predictor:
        def set_image(self, _):pass
        def predict(self, **kw):
            masks = np.zeros((len(kw['box']), 1, 4, 5), bool);scores = np.ones((len(masks), 1))
            return (masks[0], scores[0], None) if len(masks) == 1 else (masks, scores, None)
    p = propose_hand_masks(np.zeros((4, 5, 3), np.uint8), hands, Predictor(), frame_index=index)
    valid = np.ones(n, bool) if support is None else np.asarray(support, bool)
    reused = np.zeros(n, bool) if reuse is None else np.asarray(reuse, bool)
    return replace(p, masks_a=readonly(a), masks_b=readonly(b), mask_supported_a=readonly(valid),
                   mask_supported_b=readonly(valid), b_reuses_a=readonly(reused))


def gt():
    seg = np.zeros((4, 5), np.uint8);seg[1:3, 1:3] = 255;return seg


def mask(*pixels):
    a = np.zeros((4, 5), bool)
    for y, x in pixels:a[y, x] = True
    return a


def run(proposals, segs):
    return evaluate_hand_masks(np.arange(len(proposals), dtype=np.int64), iter(proposals), iter(segs))


def test_all_slots_union_overlap_once_no_gt_best_selection():
    a = [mask((1, 1), (1, 2)), mask((1, 2), (2, 1), (2, 2))]
    r = run([proposal(0, a)], [gt()])
    assert r.a.predicted_pixels.tolist() == [4] and r.a.intersection_pixels.tolist() == [4]
    assert r.a.dice == r.b.dice == (1.,) and r.a.iou == (1.,)
    assert r.proposal_count.tolist() == [2]


def test_missing_and_empty_masks_count_real_zero_on_all_positive_frames():
    p = [proposal(0, []), proposal(1, [mask()]), proposal(2, [gt() == 255])]
    r = run(p, [gt(), gt(), gt()])
    assert r.a.dice == (0., 0., 1.) and r.a.mean_positive_dice == pytest.approx(1/3)
    assert r.positive_frames == 3 and r.positive_no_proposal_frames == 1 and r.no_proposal_frames == 1
    assert r.a.positive_empty_union_frames == 2 and r.a.supported_empty_slot_count.tolist() == [0, 1, 0]


def test_unsupported_and_no_slot_missing_are_explicit_not_excluded():
    r = run([proposal(0, [mask()], support=[False], reuse=[True]), proposal(1, [])], [gt(), gt()])
    assert r.a.unsupported_slot_count.tolist() == [1, 0] and r.a.dice == (0., 0.)
    assert r.b_intervention_slot_count.tolist() == [0, 0] and r.b_reuse_slot_count.tolist() == [1, 0]


def test_unlabelled_no_hand_pixels_never_true_negative_dice_or_fp():
    r = run([proposal(0, [mask((0, 0))])], [np.zeros((4, 5), np.uint8)])
    assert r.positive_frames == 0 and r.unlabelled.tolist() == [True]
    assert r.a.dice == r.a.iou == (None,) and r.a.mean_positive_dice is None
    assert r.mean_positive_paired_dice_delta is None and r.a.total_predicted_pixels == 1
    assert not hasattr(r, "true_negatives") and not hasattr(r.a, "false_positives")


def test_object_and_background_contamination_raw_counts_denominator_separate():
    seg = gt();seg[0, 0] = 1;seg[0, 1] = 21
    selected = mask((1, 1), (0, 0), (0, 1), (3, 4))
    r = run([proposal(0, [selected])], [seg])
    assert r.a.object_label_pixels.tolist() == [2] and r.a.background_label_pixels.tolist() == [1]
    assert r.a.predicted_pixels.tolist() == [4] and r.a.total_predicted_pixels == 4
    assert r.a.positive_object_label_pixels == 2 and r.a.positive_background_label_pixels == 1
    assert r.a.dice == (.25,) and r.a.iou == (pytest.approx(1/7),)


def test_paired_b_intervention_reuses_same_frame_denominator_including_miss():
    p = [proposal(0, [mask()], [gt() == 255]), proposal(1, [], reuse=[])]
    r = run(p, [gt(), gt()])
    assert r.a.dice == (0., 0.) and r.b.dice == (1., 0.)
    assert r.mean_positive_paired_dice_delta == .5 and r.b_intervention_frames == 1
    assert r.positive_b_intervention_frames == 1 and r.b_intervention_slot_count.tolist() == [1, 0]


def test_slot_permutation_does_not_change_union_or_native_score_filter():
    a = [mask((1, 1)), mask((2, 2))];b = [mask((1, 2)), mask((2, 1))]
    p = proposal(0, a, b);q = proposal(0, a[::-1], b[::-1])
    p = replace(p, raw_scores_a=readonly([-100., 100.]), raw_scores_b=readonly([100., -100.]))
    x, y = run([p], [gt()]), run([q], [gt()])
    assert x.a.dice == y.a.dice and x.b.dice == y.b.dice
    assert x.a.predicted_pixels.tolist() == y.a.predicted_pixels.tolist()


def test_result_owned_readonly_inputs_never_edited():
    p = proposal(0, [mask((1, 1))]);seg = gt();indices = np.array([0], np.int64);before = p.masks_a.copy()
    r = evaluate_hand_masks(indices, [p], [seg])
    assert np.array_equal(p.masks_a, before)
    for a in (r.frame_index, r.annotated_positive, r.a.predicted_pixels, r.b_intervention_slot_count):
        assert not a.flags.writeable and a.flags.owndata
        with pytest.raises(ValueError):a.flat[0] = 9
    assert not np.shares_memory(indices, r.frame_index)
    indices[:] = 9;seg[:] = 0
    assert r.frame_index.tolist() == [0] and r.gt_hand_pixels.tolist() == [4]
    with pytest.raises(FrozenInstanceError):r.positive_frames = 0


def test_explicit_none_extra_record_is_not_mistaken_for_iterator_end():
    with pytest.raises(ValueError):run([proposal(0, [])], [gt(), None])


@pytest.mark.parametrize("fault", ["seg_dtype", "seg_shape", "seg_label", "masked_seg", "index_dtype", "indices",
                                  "short_gt", "extra_gt", "short_proposals", "extra_proposals", "proposal_index",
                                  "mutable", "mask_dtype", "support", "reuse_changed", "empty_timeline"])
def test_contract_failures_cannot_filter_or_reindex_to_improve_metrics(fault):
    p = proposal(0, [mask((1, 1))]);segs = [gt()];proposals = [p];indices = np.array([0], np.int64)
    if fault == "seg_dtype":segs = [segs[0].astype(np.int64)]
    elif fault == "seg_shape":segs = [segs[0][:-1]]
    elif fault == "seg_label":segs[0][0, 0] = 22
    elif fault == "masked_seg":segs = [np.ma.array(segs[0], mask=False)]
    elif fault == "index_dtype":indices = indices.astype(np.int32)
    elif fault == "indices":indices[:] = 1
    elif fault == "short_gt":segs = []
    elif fault == "extra_gt":segs += segs
    elif fault == "short_proposals":proposals = []
    elif fault == "extra_proposals":proposals += proposals
    elif fault == "proposal_index":proposals = [replace(p, frame_index=1)]
    elif fault == "mutable":proposals = [replace(p, masks_a=p.masks_a.copy())]
    elif fault == "mask_dtype":proposals = [replace(p, masks_a=readonly(p.masks_a.astype(float)))]
    elif fault == "support":proposals = [replace(p, mask_supported_a=readonly([False]))]
    elif fault == "reuse_changed":proposals = [replace(p, b_reuses_a=readonly([True]), masks_b=readonly(np.zeros_like(p.masks_b)))]
    else:indices = np.array([], np.int64)
    with pytest.raises(ValueError):evaluate_hand_masks(indices, proposals, segs)

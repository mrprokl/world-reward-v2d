"""Manufactured paired masks only; no image/model/private-data acquisition."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from world_reward.hand_mask_evaluation import evaluate_hand_masks,evaluate_temporal_hand_masks,validate_temporal_hand_frames
from world_reward.hand_temporal_masks import TemporalMaskFrame
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


def temporal_frame(t,branch,masks,*,usable=None,evidence=None):
    masks=np.asarray(masks,bool).reshape(-1,4,5);n=len(masks)
    usable=np.ones(n,bool)if usable is None else np.asarray(usable,bool)
    boxes=np.tile([1.,1.,3.,3.],(n,1));scores=np.where(usable,.7,np.nan)if branch=='A'else None
    return TemporalMaskFrame(branch,t,t,tuple(range(n)),readonly(boxes),readonly(boxes.astype(np.float32)),
        readonly(usable),readonly(masks),readonly(usable),None if scores is None else readonly(scores),
        evidence or('image_prompt'if branch=='A'else'anchor_prompt'if t==1 else'video_inferred'))


def temporal_pairs():
    return [(temporal_frame(t,'A',[]if t!=1 else[mask((1,1)),mask()],usable=[]if t!=1 else[True,False]),
        temporal_frame(t,'B',[gt()==255,mask()],usable=[True,False]))for t in range(3)]


def test_temporal_different_N_retains_original_anchor_bank_and_all_positive_missing_frames():
    pairs=temporal_pairs();r=evaluate_temporal_hand_masks(np.arange(3,dtype=np.int64),pairs,[gt()]*3)
    assert r.proposal_count_a.tolist()==[0,2,0]and r.proposal_count_b.tolist()==[2,2,2]
    assert r.anchor_position==1 and r.seeded_proposal_ids==(0,)
    assert r.a.dice==(0.,.4,0.)and r.b.dice==(1.,1.,1.)and r.a.positive_empty_union_frames==2
    assert r.a.mean_positive_dice==pytest.approx(.4/3)and r.mean_positive_paired_dice_delta==pytest.approx(1-.4/3)
    assert r.b.unsupported_slot_count.tolist()==[1,1,1]and not hasattr(r,'b_intervention_frames')
    assert validate_temporal_hand_frames(np.arange(3,dtype=np.int64),pairs)==dict(anchor_position=1,seeded_proposal_ids=(0,))
    for array in(r.frame_index,r.proposal_count_a,r.proposal_count_b,r.b.predicted_pixels):assert array.flags.owndata and not array.flags.writeable


def test_temporal_no_anchor_positive_zero_unlabelled_null_not_negatives_or_fake_ID():
    pairs=[(temporal_frame(t,'A',[]),temporal_frame(t,'B',[],evidence='no_anchor_abstention'))for t in range(2)]
    r=evaluate_temporal_hand_masks(np.arange(2,dtype=np.int64),pairs,[gt(),np.zeros((4,5),np.uint8)])
    assert r.anchor_position is None and r.seeded_proposal_ids==()and r.a.dice==r.b.dice==(0.,None)
    assert r.a.positive_empty_union_frames==r.b.positive_empty_union_frames==1 and r.unlabelled.tolist()==[False,True]
    assert not hasattr(r,'accepted_identity')


def test_temporal_uses_identical_branch_union_math_to_v2_without_GT_slot_selection():
    a=[mask((1,1)),mask((1,2), (0,0))];b=[mask((2,1)),mask((2,2), (0,1))];seg=gt();seg[0,1]=21
    old=run([proposal(0,a,b)],[seg]);pair=(temporal_frame(0,'A',a),temporal_frame(0,'B',b,evidence='anchor_prompt'))
    new=evaluate_temporal_hand_masks(np.array([0],np.int64),[pair],[seg])
    for branch in('a','b'):
        for key,value in vars(getattr(old,branch)).items():
            actual=getattr(getattr(new,branch),key)
            assert np.array_equal(actual,value)if isinstance(value,np.ndarray)else actual==value
    assert old.paired_dice_delta==new.paired_dice_delta


@pytest.mark.parametrize('fault',['order','original_id','branch','mutable','slots','dtype','support','raw_nan','native_box',
    'scores','video_scores','anchor_bank','fixed_bank','evidence','short','extra','extra_seg','bad_seg'])
def test_temporal_provenance_malformed_records_fail_not_filter_or_reseed(fault):
    pairs=temporal_pairs();indices=np.arange(3,dtype=np.int64);segments=[gt()]*3;a,b=pairs[0]
    if fault=='order':a=replace(a,position=1)
    elif fault=='original_id':a=replace(a,original_frame_id=7)
    elif fault=='branch':a=replace(a,branch='B')
    elif fault=='mutable':b=replace(b,masks=b.masks.copy())
    elif fault=='slots':b=replace(b,proposal_ids=(1,0))
    elif fault=='dtype':b=replace(b,masks=readonly(b.masks.astype(np.uint8)))
    elif fault=='support':b=replace(b,supported=readonly([False,False]))
    elif fault=='raw_nan':boxes=b.raw_boxes.copy();boxes[0,0]=np.nan;b=replace(b,raw_boxes=readonly(boxes))
    elif fault=='native_box':b=replace(b,native_boxes=readonly(b.native_boxes+1))
    elif fault=='scores':a=replace(a,raw_scores=None)
    elif fault=='video_scores':b=replace(b,raw_scores=readonly([.7,np.nan]))
    elif fault=='anchor_bank':pairs[1]=(temporal_frame(1,'A',[mask()]),pairs[1][1])
    elif fault=='fixed_bank':pairs[2]=(pairs[2][0],temporal_frame(2,'B',[mask()]))
    elif fault=='evidence':b=replace(b,evidence='anchor_prompt')
    elif fault=='short':pairs=pairs[:-1]
    elif fault=='extra':pairs=pairs+[pairs[-1]]
    elif fault=='extra_seg':segments=segments+[gt()]
    elif fault=='bad_seg':segments=[gt().astype(np.int64)]*3
    pairs[0]=(a,b)
    with pytest.raises(ValueError):evaluate_temporal_hand_masks(indices,pairs,segments)


def test_source_only_temporal_prepass_rejects_crossframe_anchor_change_without_GT_argument():
    pairs=temporal_pairs();pairs[-1]=(pairs[-1][0],replace(pairs[-1][1],raw_boxes=readonly([[0,0,1,1]]*2),native_boxes=readonly(np.array([[0,0,1,1]]*2,np.float32))))
    with pytest.raises(ValueError):validate_temporal_hand_frames(np.arange(3,dtype=np.int64),pairs)


def test_true_stream_temporal_shapes_reordered_by_original_position_not_physical_matching():
    from test_hand_temporal_masks import run as stream,H,W
    _,records,_,_,_,indices=stream(frame_ids=np.arange(4,dtype=np.int64))
    pairs=[tuple(next(r for r in records if r.position==t and r.branch==arm)for arm in('A','B'))for t in indices]
    segments=[np.full((H,W),255,np.uint8)for _ in indices]
    r=evaluate_temporal_hand_masks(indices,pairs,segments)
    assert r.proposal_count_a.tolist()==[0,3,0,2]and r.proposal_count_b.tolist()==[3]*4
    assert r.seeded_proposal_ids==(0,2)and r.anchor_position==1
    assert r.a.dice==(0.,)*4 and all(value>0 for value in r.b.dice)
    assert r.b.unsupported_slot_count.tolist()==[1]*4 and r.a.positive_empty_union_frames==4

"""Small masks and exact graph fixtures, not external/challenge quality labels."""
import itertools
import json

import numpy as np
import pytest

from world_reward.proposal_recall import (THRESHOLDS, _matching,
    evaluate_visible_proposals, validate_raster_truth, aggregate_visible_recall)


def evaluate(truth, predicted, kinds=None, **kwargs):
    g,h,w = truth.shape; n = len(predicted)
    packed = np.packbits(predicted.reshape(n,h*w),axis=1,bitorder='little')
    return evaluate_visible_proposals(packed,np.array([h,w],np.int64),truth,
        tuple('human'for _ in range(g)) if kinds is None else kinds,
        native_areas=predicted.sum((1,2),dtype=np.int64), **kwargs)


def fixtures():
    t=np.zeros((3,4,5),bool);t[0,:2,:2]=True;t[1,2:,3:]=True
    p=np.stack([t[0],t[0],t[1],np.zeros((4,5),bool)])
    return t,p


def test_full_duplicates_empty_native_and_zero_visible_truth_retained():
    t,p=fixtures();t0,p0=t.copy(),p.copy()
    r=evaluate(t,p,('human','object','object'),entity_ids=('human_a','prop_a','occluded_or_outside'),generate_seconds=.3)
    assert r.entity_ids==('human_a','prop_a','occluded_or_outside')
    assert r.metrics['native_masks']==4 and r.metrics['zero_visible_instances']==1
    np.testing.assert_array_equal(r.visible_areas,[4,4,0]);np.testing.assert_array_equal(r.eligible,[True,True,False])
    np.testing.assert_allclose(r.best_iou[:2],[1,1]);assert np.isnan(r.best_iou[2])
    assert np.isnan(r.iou[2]).all()
    np.testing.assert_array_equal(r.native_areas,[4,4,4,0])
    for key in ('0.25','0.5','0.75'):
        assert r.metrics['recalls'][key]['all']['recall']==1
        assert r.metrics['recalls'][key]['all']['one_to_one_recall']==1
    assert r.metrics['ownership_verified'] is r.metrics['accuracy_verified'] is r.metrics['adoption'] is False
    np.testing.assert_array_equal(t,t0);np.testing.assert_array_equal(p,p0)
    for a in (r.visible_areas,r.eligible,r.iou,r.best_iou,r.matching_slots,r.native_areas):
        assert not a.flags.writeable
        with pytest.raises(ValueError):a.flat[0]=0
    with pytest.raises(TypeError):r.metrics['native_masks']=0


def test_union_proposal_recall_not_one_to_one_recovery():
    t=np.zeros((2,2,4),bool);t[0,:,:2]=True;t[1,:,2:]=True
    r=evaluate(t,np.ones((1,2,4),bool),('human','object'))
    np.testing.assert_array_equal(r.best_iou,[.5,.5])
    for key in ('0.25','0.5'):
        m=r.metrics['recalls'][key]
        assert m['all']['recall']==1 and m['all']['one_to_one_recall']==.5
        assert m['human']['one_to_one_recall']==m['object']['one_to_one_recall']==1
        assert np.count_nonzero(r.matching_slots[int(key=='0.5')]>=0)==1
    assert r.metrics['recalls']['0.75']['all']['recall']==0


def test_empty_native_bank_is_misses_not_perfect_score():
    t,_=fixtures();r=evaluate(t,np.empty((0,4,5),bool))
    assert r.iou.shape==(3,0) and r.metrics['native_masks']==0
    np.testing.assert_array_equal(r.best_iou[:2],[0,0]);assert np.isnan(r.best_iou[2])
    assert np.all(r.matching_slots==-1)
    assert all(r.metrics['recalls'][str(x)]['all']['recall']==0 for x in THRESHOLDS)


def test_empty_truth_has_undefined_recall_and_retains_native_bank():
    r=evaluate(np.empty((0,4,5),bool),np.ones((2,4,5),bool))
    assert r.iou.shape==(0,2) and r.metrics['native_masks']==2
    assert r.metrics['recalls']['0.5']['all']['recall'] is None
    s=aggregate_visible_recall([r]);assert s['recalls']['0.5']['all']['macro_image_recall'] is None
    json.dumps(s,allow_nan=False)


def test_all_invisible_instances_never_manufacture_occlusion_claim():
    r=evaluate(np.zeros((2,4,5),bool),np.ones((1,4,5),bool),('human','object'))
    assert np.isnan(r.best_iou).all() and r.metrics['zero_visible_instances']==2
    assert r.metrics['zero_visible_cause']=='unknown_from_visible_masks'
    assert aggregate_visible_recall([r])['occlusion_fraction_available'] is False


def brute_cardinality(edges):
    g,n=edges.shape
    for size in range(min(g,n),-1,-1):
        for rows in itertools.combinations(range(g),size):
            for cols in itertools.permutations(range(n),size):
                if all(edges[r,c]for r,c in zip(rows,cols)):return size


def test_matching_requires_augmenting_reassignment_not_greedy_iou():
    edges=np.array([[True,True],[True,False]])
    match=_matching(edges)
    np.testing.assert_array_equal(match,[1,0])


@pytest.mark.parametrize('g,n',[(0,0),(0,3),(3,0),(2,3),(3,3)])
def test_exact_max_cardinality_matches_exhaustive_independent_reference(g,n):
    for bits in range(1<<(g*n)):
        edges=np.array([(bits>>i)&1 for i in range(g*n)],bool).reshape(g,n)
        r=_matching(edges);selected=r[r>=0]
        assert len(selected)==brute_cardinality(edges) and len(set(selected))==len(selected)
        assert all(edges[i,c]for i,c in enumerate(r)if c>=0)


def test_packed_iou_matches_unpacked_independent_original_boolean_math():
    rng=np.random.default_rng(173); labels=rng.integers(-1,4,(7,9))
    t=np.stack([labels==i for i in range(4)]);p=rng.random((13,7,9))>.55
    r=evaluate(t,p)
    for i in range(4):
        for j in range(13):
            expected=(t[i]&p[j]).sum()/(t[i]|p[j]).sum()
            assert r.iou[i,j]==expected


def test_macro_per_image_not_silently_instance_weighted_and_cost_retained():
    t,p=fixtures();good=evaluate(t,p,('human','object','object'),generate_seconds=.2)
    t2=t[:1];bad=evaluate(t2,np.empty((0,4,5),bool),('human',),generate_seconds=.7)
    s=aggregate_visible_recall([good,bad])
    assert s['recalls']['0.5']['all']['macro_image_recall']==.5
    assert s['recalls']['0.5']['all']['micro_instance_recall']==pytest.approx(2/3)
    assert s['recalls']['0.5']['object']['supported_images']==1
    assert s['recalls']['0.5']['object']['macro_image_recall']==1
    assert s['native_masks']==4 and s['native_masks_per_image']==[4,0]
    assert s['native_generate_seconds']==pytest.approx(.9) and s['cost_images']==2
    json.dumps(s,allow_nan=False)


@pytest.mark.parametrize('fault',['pad','area','dtype','header','grid','truth_dtype','overlap','kinds','ids','cost'])
def test_mismatched_or_polluted_arrays_fail_closed(fault):
    t,p=fixtures();packed=np.packbits(p.reshape(4,20),axis=1,bitorder='little')
    header=np.array([4,5],np.int64);areas=p.sum((1,2),dtype=np.int64);kinds=('human','object','object')
    ids=('a','b','c');cost=.3
    if fault=='pad':packed[0,-1]|=128
    elif fault=='area':areas[0]+=1
    elif fault=='dtype':packed=packed.astype(np.int8)
    elif fault=='header':header=header.astype(np.int32)
    elif fault=='grid':header=np.array([5,4],np.int64)
    elif fault=='truth_dtype':t=t.astype(np.uint8)
    elif fault=='overlap':t[1,0,0]=True
    elif fault=='kinds':kinds=('person','object','object')
    elif fault=='ids':ids=('a','a','b')
    else:cost=float('nan')
    with pytest.raises(ValueError):evaluate_visible_proposals(packed,header,t,kinds,native_areas=areas,entity_ids=ids,generate_seconds=cost)


def test_padding_validation_even_all_empty_mask_payload():
    with pytest.raises(ValueError):evaluate_visible_proposals(np.array([[128]],np.uint8),np.array([1,1],np.int64),
        np.zeros((1,1,1),bool),('object',),native_areas=np.array([0],np.int64))


def test_optional_raster_consistency_uses_all_original_face_labels():
    faces=np.array([[-1,0,1],[-1,2,1]],np.int64);ids=np.array([0,1,0],np.int32)
    label=np.array([[-1,0,1],[-1,0,1]],np.int32);t=np.stack([label==i for i in range(3)])
    assert validate_raster_truth(t,label,faces,ids)
    # Zero-area third entity is retained, not removed.
    assert t.shape==(3,2,3) and not t[2].any()
    broken=t.copy();broken[0,0,0]=True
    with pytest.raises(ValueError):validate_raster_truth(broken,label,faces,ids)


@pytest.mark.parametrize('fault',['labels','face_id','background','dtype','entity_id','shape'])
def test_optional_geometry_mismatch_fails_without_truth_patch(fault):
    faces=np.array([[-1,0],[1,2]],np.int64);ids=np.array([0,1,0],np.int32)
    label=np.array([[-1,0],[1,0]],np.int32);t=np.stack([label==i for i in range(2)])
    if fault=='labels':label[0,1]=1
    elif fault=='face_id':faces[0,1]=3
    elif fault=='background':faces[0,0]=-2
    elif fault=='dtype':faces=faces.astype(np.int32)
    elif fault=='entity_id':ids[0]=2
    else:t=t[:,:,:1]
    with pytest.raises(ValueError):validate_raster_truth(t,label,faces,ids)


def test_invalid_aggregate_empty_or_foreign_record_rejected():
    with pytest.raises(ValueError):aggregate_visible_recall([])
    with pytest.raises(ValueError):aggregate_visible_recall([{}])

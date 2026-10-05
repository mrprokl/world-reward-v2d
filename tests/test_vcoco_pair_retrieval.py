"""Authored static role/instance and numeric bank fixtures; no actual references."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from world_reward import vcoco_pair_retrieval as m
from world_reward.vcoco_role_reference import parse_vcoco_role_reference


def fixture():
    images=[dict(id=501,width=100,height=80)]
    instances=[dict(id=11,image_id=501,category_id=1,iscrowd=0,bbox=[0,0,10,20],area=200),
        dict(id=12,image_id=501,category_id=1,iscrowd=0,bbox=[60,0,10,20],area=200),
        dict(id=31,image_id=501,category_id=17,iscrowd=0,bbox=[10,30,10,10],area=100),
        dict(id=32,image_id=501,category_id=44,iscrowd=0,bbox=[60,30,10,10],area=100),
        dict(id=40,image_id=501,category_id=7,iscrowd=0,bbox=[80,50,5,5],area=25)]
    actions=[dict(action_name='role_A',role_name=['agent','obj','instr'],ann_id=[11,12],label=[1,1],
        image_id=[501,501],role_object_id=[11,12,31,32,0,0]),
        dict(action_name='role_B',role_name=['agent','obj'],ann_id=[11],label=[1],image_id=[501],role_object_id=[11,31]),
        dict(action_name='unlabelled_negative',role_name=['agent','obj'],ann_id=[12],label=[0],image_id=[501],role_object_id=[999,999])]
    p=np.array([[0,0,10,20],[60,0,70,20]],np.float64)
    o=np.array([[10,30,20,40],[60,30,70,40],[80,50,85,55]],np.float64)
    scores=np.array([[5.,-3.,-8.],[-4.,2.,-7.]],np.float64)
    return actions,instances,images,p,o,scores


def evaluate(f=None,*,scores=None,support=None,pids=None,oids=None):
    a,i,images,p,o,s=fixture()if f is None else f
    return m.evaluate_vcoco_pair_retrieval(p,o,s if scores is None else scores,
        np.ones((len(p),len(o)),bool)if support is None else support,parse_vcoco_role_reference(a,i,images),i,501,(80,100),
        tuple(f'p{j}'for j in range(len(p)))if pids is None else pids,tuple(f'o{j}'for j in range(len(o)))if oids is None else oids)


def test_localized_positive_pair_weight_and_unlabelled_alternatives_not_negative():
    f=fixture();before=deepcopy(f[:3]);result=evaluate(f)
    assert f[:3]==before and result.localized_positive_pair_count==2 and result.retrieved_positive_pair_count==2
    assert result.image_positive_retrieval==result.positive_pair_proposal_recall==result.supported_positive_pair_recall==1.
    assert result.person_reference_annotation_ids.tolist()==[11,12] and result.object_reference_annotation_ids.tolist()==[31,32,40]
    assert result.role_diagnostics['positive_role_slots']==5 and result.role_diagnostics['missing_positive_role_slots']==2
    assert result.role_diagnostics['unscorable_positive_role_slots']==2 and not result.scope['unknown_pairs_verified_negative']
    unknown=np.zeros((2,3),np.float64);unknown[0,2]=100
    out=evaluate(f,scores=unknown);assert out.image_positive_retrieval==0. and out.positive_pair_proposal_recall==1.


def test_exact_ties_fractional_over_unique_pairs_no_epsilon():
    f=fixture();scores=np.zeros((2,3),np.float64)
    out=evaluate(f,scores=scores);assert out.top_unique_pair_count==6 and out.top_known_positive_count==2
    assert out.image_positive_retrieval==2/6
    scores[0,0]=np.nextafter(0.,1.)
    out=evaluate(f,scores=scores);assert out.top_unique_pair_count==1 and out.image_positive_retrieval==1.


def test_exact_automatic_aliases_neutral_not_gt_alias_collapse():
    a,i,images,p,o,s=fixture();p=np.concatenate((p,p[:1]));o=np.concatenate((o,o[:1]))
    scores=s[np.ix_([0,1,0],[0,1,2,0])]
    out=evaluate((a,i,images,p,o,scores));assert out.raw_proposal_counts==(3,4)and out.unique_proposal_counts==(2,3)
    assert out.person_group_indices.tolist()==[0,1,0]and out.object_group_indices.tolist()==[0,1,2,0]
    assert out.image_positive_retrieval==1. and out.localized_positive_pair_count==2
    tied=evaluate((a,i,images,p,o,np.zeros((3,4))));assert tied.image_positive_retrieval==2/6 and tied.top_unique_pair_count==6


@pytest.mark.parametrize('fault',['score','support'])
def test_exact_alias_disagreement_rejected_no_mean_max_repair(fault):
    a,i,images,p,o,s=fixture();p=np.concatenate((p,p[:1]));scores=s[[0,1,0]].copy();support=np.ones(scores.shape,bool)
    if fault=='score':scores[2,1]=50.
    else:support[2,1]=False;scores[2,1]=np.nan
    with pytest.raises(ValueError,match='aliases'):evaluate((a,i,images,p,o,scores),support=support)


def test_permutation_equivariance_and_source_ids_not_changed():
    a,i,images,p,o,s=fixture();orderp=[1,0];ordero=[2,0,1]
    out=evaluate((a,i,images,p[orderp],o[ordero],s[np.ix_(orderp,ordero)]),pids=('second','first'),oids=('ctx','first','second'))
    assert out.image_positive_retrieval==1. and out.person_ids==('second','first') and out.object_ids==('ctx','first','second')
    assert out.person_reference_annotation_ids.tolist()==[12,11]and out.object_reference_annotation_ids.tolist()==[40,31,32]


@pytest.mark.parametrize('bank',['person','object','both','unsupported'])
def test_empty_or_all_unsupported_is_positive_miss_not_off(bank):
    a,i,images,p,o,s=fixture()
    if bank in('person','both'):p=np.empty((0,4),np.float64)
    if bank in('object','both'):o=np.empty((0,4),np.float64)
    support=np.zeros((len(p),len(o)),bool);scores=np.full(support.shape,np.nan,np.float64)
    out=evaluate((a,i,images,p,o,scores),support=support)
    assert out.image_positive_retrieval==0. and out.top_score is None and out.top_unique_pair_count==0
    assert out.supported_positive_pair_recall==0. and out.localized_positive_pair_count==2
    assert out.positive_pair_proposal_recall==(1. if bank=='unsupported'else 0.)


def test_geometry_proposal_ceiling_and_score_support_ceiling_separate():
    f=fixture();support=np.ones((2,3),bool);support[0,0]=False;scores=f[-1].copy();scores[0,0]=np.nan
    out=evaluate(f,scores=scores,support=support)
    assert out.positive_pair_proposal_recall==1. and out.supported_positive_pair_recall==.5 and out.image_positive_retrieval==1.


@pytest.mark.parametrize('kind',['sameperson','differentclass','crowd','unlabelled_context'])
def test_match_all_original_context_before_category_and_crowd_censor(kind):
    a,i,images,p,o,s=fixture()
    row=deepcopy(i[2]);row['id']=80
    if kind=='sameperson':row=deepcopy(i[0]);row['id']=80
    elif kind=='differentclass':row['category_id']=1
    elif kind=='crowd':row['iscrowd']=1
    i.append(row);out=evaluate((a,i,images,p,o,s))
    assert out.image_positive_retrieval==0. and out.positive_pair_proposal_recall==.5
    assert (out.person_reference_annotation_ids[0]if kind=='sameperson'else out.object_reference_annotation_ids[0])==-1


def test_unique_crowd_or_wrong_class_match_unknown_never_positive():
    a,i,images,p,o,s=fixture();i.append(dict(id=81,image_id=501,category_id=1,iscrowd=1,bbox=[20,0,10,20],area=200))
    p[0]=[20,0,30,20];out=evaluate((a,i,images,p,o,s))
    assert out.person_reference_annotation_ids[0]==-1 and out.positive_person_proposal_recall==.5
    p[0]=o[0];out=evaluate((a,i,images,p,o,s));assert out.person_reference_annotation_ids[0]==-1


def test_continuous_pixel_iou_exact_half_no_plusone():
    a,i,images,p,o,s=fixture();o[0]=[10,30,15,40]
    out=evaluate((a,i,images,p,o,s));assert out.object_reference_annotation_ids[0]==31
    o[0,2]=np.nextafter(15.,0.)
    out=evaluate((a,i,images,p,o,s));assert out.object_reference_annotation_ids[0]==-1


def test_offgrid_zeroarea_not_dropped_or_clipped():
    a,i,images,p,o,s=fixture();i[2]['bbox']=[-10,30,10,10];o[0]=[-10,30,0,40]
    out=evaluate((a,i,images,p,o,s));assert out.image_positive_retrieval==1. and out.object_reference_annotation_ids[0]==31
    o=np.concatenate((o,np.array([[0,0,0,0]],np.float64)));scores=np.pad(s,((0,0),(0,1)),constant_values=100)
    out=evaluate((a,i,images,p,o,scores));assert out.raw_proposal_counts==(2,4) and out.image_positive_retrieval==0.


def test_no_localized_pair_explicit_unscorable_with_missing_role_diagnostics():
    a,i,images,p,o,s=fixture()
    for x in a:
        if x['label'][0]:x['role_object_id']=[*x['ann_id'],*([0]*(len(x['role_object_id'])-len(x['ann_id'])))]
    out=evaluate((a,i,images,p,o,s));assert out.status=='no_localized_positive'
    assert out.positive_pair_proposal_recall is None and out.supported_positive_pair_recall is None
    assert out.image_positive_retrieval==0. and out.role_diagnostics['missing_positive_role_slots']==5


@pytest.mark.parametrize('fault',['score_inf','score_nan_supported','score_finite_unsupported','bool_score','mask_int','inverted','box_inf',
    'box_bool','box_shape','person_ids','object_ids','other_image','missing_context_endpoint','ref_grid','forged_pair'])
def test_invalid_or_unbound_contract_rejected_without_repair(fault):
    a,i,images,p,o,s=fixture();support=np.ones(s.shape,bool);pids=('p0','p1');oids=('o0','o1','o2')
    if fault=='score_inf':s[0,0]=np.inf
    elif fault=='score_nan_supported':s[0,0]=np.nan
    elif fault=='score_finite_unsupported':support[0,0]=False
    elif fault=='bool_score':s=s.astype(bool)
    elif fault=='mask_int':support=support.astype(np.int64)
    elif fault=='inverted':p[0,2]=-1
    elif fault=='box_inf':p[0,0]=np.inf
    elif fault=='box_bool':p=p.astype(bool)
    elif fault=='box_shape':p=p[:,:3]
    elif fault=='person_ids':pids=('repeat','repeat')
    elif fault=='object_ids':oids=('o0',)
    ref=parse_vcoco_role_reference(a,i,images);grid=(80,100)
    if fault=='other_image':i[0]['image_id']=502
    elif fault=='missing_context_endpoint':i.pop(2)
    elif fault=='ref_grid':grid=(10,10)
    elif fault=='forged_pair':ref=replace(ref,localized_positive_pairs=())
    with pytest.raises(ValueError):m.evaluate_vcoco_pair_retrieval(p,o,s,support,ref,i,501,grid,pids,oids)


def test_result_arrays_scope_immutable_and_input_snapshots_unchanged():
    f=fixture();before=[x.tobytes()for x in f[3:]];out=evaluate(f)
    assert before==[x.tobytes()for x in f[3:]]
    for a in(out.person_group_indices,out.object_group_indices,out.person_reference_annotation_ids,out.object_reference_annotation_ids):
        with pytest.raises(ValueError):a.flags.writeable=True
    with pytest.raises(TypeError):out.scope['contact_verified']=True
    with pytest.raises(TypeError):out.role_diagnostics['missing_positive_role_slots']=0


def test_all3600_object_candidates_no_topk_or_hidden_selection():
    a,i,images,p,o,s=fixture();o=np.vstack((o,np.tile([90,60,91,61],(3597,1)))).astype(np.float32)
    scores=np.zeros((2,3600),np.float32);scores[0,3:]=5.
    out=evaluate((a,i,images,p.astype(np.float32),o,scores))
    assert out.raw_proposal_counts==(2,3600)and out.unique_proposal_counts==(2,4)
    scores[0,3599]=6.
    with pytest.raises(ValueError,match='aliases'):evaluate((a,i,images,p.astype(np.float32),o,scores))


def test_full3600_unique_objects_without_selection_can_pick_unknown_winner():
    a,i,images,p,o,s=fixture();x=np.arange(3597,dtype=np.float64)+1000
    o=np.vstack((o,np.stack((x,x,x+1,x+1),axis=1)));scores=np.zeros((2,3600),np.float64);scores[0,-1]=5.
    out=evaluate((a,i,images,p,o,scores));assert out.raw_proposal_counts==out.unique_proposal_counts==(2,3600)
    assert out.image_positive_retrieval==0. and out.positive_pair_proposal_recall==1.


def test_fingerprint_detects_source_mutation_during_matching(monkeypatch):
    f=fixture();original=m._match;done=[False]
    def match(*args):
        result=original(*args)
        if not done[0]:f[3][0,0]=1.;done[0]=True
        return result
    monkeypatch.setattr(m,'_match',match)
    with pytest.raises(ValueError,match='source bank changed'):evaluate(f)

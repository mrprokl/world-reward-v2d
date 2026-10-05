"""New manufactured references only; no actual V-COCO/COCO files or models."""
from copy import deepcopy
import math

import pytest

from world_reward.vcoco_role_reference import parse_vcoco_role_reference


def supplied():
    images=[dict(id=501,width=80,height=60),dict(id=502,width=80,height=60)]
    instances=[dict(id=11,image_id=501,category_id=1,iscrowd=0,bbox=[1,2,22,45],area=990),
        dict(id=12,image_id=501,category_id=1,iscrowd=0,bbox=[34,3,23,43],area=989),
        dict(id=31,image_id=501,category_id=17,iscrowd=0,bbox=[18,16,9,11],area=99),
        dict(id=32,image_id=501,category_id=44,iscrowd=0,bbox=[45,17,10,8],area=80),
        dict(id=41,image_id=502,category_id=18,iscrowd=0,bbox=[5,5,7,8],area=56)]
    actions=[dict(action_name='new_action',role_name=['agent','obj','instr'],ann_id=[11,12],
        label=[1,1],image_id=[501,501],role_object_id=[11,12,31,32,0,31],extra={'source_note':'procedural'}),
        dict(action_name='another_action',role_name=['agent','obj'],ann_id=[[11]],label=[[1]],
        image_id=[[501]],role_object_id=[11,31]),
        dict(action_name='negative_action',role_name=['agent','obj'],ann_id=[12],label=[0],
        image_id=[501],role_object_id=[999,999])]
    return actions,instances,images


def test_role_major_original_ids_and_unique_pair_weight_preserving_metadata():
    raw=supplied();before=deepcopy(raw);v=parse_vcoco_role_reference(*raw)
    assert raw==before and len(v.rows)==4
    assert [r.role_object_ids for r in v.rows]==[(11,31,0),(12,32,31),(11,31),(999,999)]
    assert v.rows[0].pair_unscorable_reasons[2]==('missing_role_unknown_not_off',)
    assert [(p.agent_annotation_id,p.object_annotation_id) for p in v.localized_positive_pairs]==[(11,31),(12,32),(12,31)]
    assert len(v.localized_positive_pairs[0].row_role_references)==2
    assert v.action_metadata[0]['extra']['source_note']=='procedural'
    assert v.rows[-1].positive_role_endpoints==(None,None) and not any(v.rows[-1].nonperson_pair_eligible)
    assert not v.scope['source_authenticated'] and not v.scope['official_AP_computed']


def test_no_role_action_and_empty_reference_are_genuine_no_pair():
    actions,instances,images=supplied();actions=[dict(action_name='agent_only',role_name=['agent'],
        ann_id=[11],label=[1],image_id=[501],role_object_id=[11])]
    v=parse_vcoco_role_reference(actions,instances,images)
    assert len(v.rows)==1 and not v.localized_positive_pairs
    empty=parse_vcoco_role_reference([],[],[]);assert empty.rows==empty.localized_positive_pairs==()


def test_same_pair_multiple_roles_retains_each_role_without_multiplying_pair_weight():
    actions,instances,images=supplied();actions=actions[:1]
    actions[0].update(ann_id=[11],label=[1],image_id=[501],role_object_id=[11,31,31])
    v=parse_vcoco_role_reference(actions,instances,images)
    assert len(v.localized_positive_pairs)==1 and len(v.localized_positive_pairs[0].row_role_references)==2
    assert v.rows[0].nonperson_pair_eligible==(False,True,True)


@pytest.mark.parametrize('variant',['person','self','crowd_target','crowd_agent','invalid_object','invalid_agent'])
def test_general_roles_retained_but_unscorable_not_negative(variant):
    actions,instances,images=supplied();actions=actions[:1];actions[0].update(ann_id=[11],label=[1],image_id=[501],role_object_id=[11,31,0])
    if variant=='person':instances[2]['category_id']=1
    elif variant=='self':actions[0]['role_object_id'][1]=11
    elif variant=='crowd_target':instances[2]['iscrowd']=1
    elif variant=='crowd_agent':instances[0]['iscrowd']=1
    elif variant=='invalid_object':instances[2]['bbox'][2]=-1
    elif variant=='invalid_agent':instances[0]['area']=0
    v=parse_vcoco_role_reference(actions,instances,images)
    assert len(v.rows)==1 and not v.localized_positive_pairs and v.rows[0].role_object_ids[2]==0
    assert v.rows[0].pair_unscorable_reasons[1] and not v.scope['nonpositive_pairs_verified_negative']


def test_raw_offgrid_box_preserved_without_clip_or_invalid_area_repair():
    actions,instances,images=supplied();instances[2]['bbox']=[-2.,16.,9.,11.]
    v=parse_vcoco_role_reference(actions,instances,images);target=v.rows[0].positive_role_endpoints[1]
    assert target.bbox_xywh==(-2.,16.,9.,11.) and target.raw_box_positive and not target.raw_box_inside_image
    assert v.rows[0].nonperson_pair_eligible[1] and not v.scope['raw_geometry_clipped']


@pytest.mark.parametrize('fault',['dangling','cross_image','duplicate_annotation','duplicate_image','positive_agent_conflict',
    'duplicate_positive','agent_nonperson','wrong_label','float_ID','bool_ID','negative_role','twoD_role','mixed_vector',
    'shape','nonfinite_box','nonfinite_extra','duplicate_roles','wrong_agent_role','duplicate_action'])
def test_structural_ambiguity_rejected_without_repair(fault):
    actions,instances,images=supplied()
    if fault=='dangling':actions[0]['role_object_id'][2]=777
    elif fault=='cross_image':actions[0]['role_object_id'][2]=41
    elif fault=='duplicate_annotation':instances.append(deepcopy(instances[0]))
    elif fault=='duplicate_image':images.append(deepcopy(images[0]))
    elif fault=='positive_agent_conflict':actions[0]['role_object_id'][0]=12
    elif fault=='duplicate_positive':actions[0].update(ann_id=[11,11],role_object_id=[11,11,31,31,0,0])
    elif fault=='agent_nonperson':instances[0]['category_id']=18
    elif fault=='wrong_label':actions[0]['label'][0]=-1
    elif fault=='float_ID':actions[0]['ann_id'][0]=11.
    elif fault=='bool_ID':actions[0]['ann_id'][0]=True
    elif fault=='negative_role':actions[0]['role_object_id'][2]=-1
    elif fault=='twoD_role':actions[0]['role_object_id']=[[11,12],[31,32],[0,31]]
    elif fault=='mixed_vector':actions[0]['ann_id']=[[11],12]
    elif fault=='shape':actions[0]['image_id'].pop()
    elif fault=='nonfinite_box':instances[2]['bbox'][0]=math.nan
    elif fault=='nonfinite_extra':actions[0]['extra']['bad']=math.inf
    elif fault=='duplicate_roles':actions[0]['role_name']=['agent','obj','obj']
    elif fault=='wrong_agent_role':actions[0]['role_name'][0]='person'
    elif fault=='duplicate_action':actions[1]['action_name']=actions[0]['action_name']
    with pytest.raises((ValueError,OverflowError)):parse_vcoco_role_reference(actions,instances,images)


def test_nested_metadata_and_rows_immutable_after_caller_mutation():
    raw=supplied();v=parse_vcoco_role_reference(*raw);digest=v.source_fingerprint
    raw[0][0]['role_object_id'][2]=32;raw[1][2]['bbox'][0]=50;raw[0][0]['extra']['source_note']='changed'
    assert v.rows[0].role_object_ids[1]==31 and v.rows[0].positive_role_endpoints[1].bbox_xywh[0]==18
    assert v.action_metadata[0]['extra']['source_note']=='procedural' and v.source_fingerprint==digest
    with pytest.raises(TypeError):v.scope['source_authenticated']=True
    with pytest.raises(TypeError):v.action_metadata[0]['extra']['source_note']='edited'


def test_positive_and_nonpositive_may_share_agent_without_negative_pair_claim():
    actions,instances,images=supplied();actions[0]['label']=[1,0]
    v=parse_vcoco_role_reference(actions,instances,images)
    assert v.rows[1].label==0 and not any(v.rows[1].nonperson_pair_eligible)
    assert v.rows[1].role_object_ids==(12,32,31) and not v.scope['nonpositive_role_endpoints_consulted']


def test_explicit_projection_preserves_original_action_and_row_slots_without_changing_role_major_join():
    actions,instances,images=supplied()
    for action,slot,rows,count in zip(actions,(4,7,8),([2,5],[0],[3]),(6,1,4)):
        action['projection_source']=dict(action_slot=slot,row_slots=rows,row_count=count)
    v=parse_vcoco_role_reference(actions,instances,images)
    assert [(r.action_slot,r.row_slot) for r in v.rows]==[(4,2),(4,5),(7,0),(8,3)]
    assert v.rows[0].role_object_ids==(11,31,0) and v.rows[1].role_object_ids==(12,32,31)
    assert v.localized_positive_pairs[0].row_role_references==((4,2,1,'new_action','obj'),(7,0,1,'another_action','obj'))
    assert v.action_metadata[0]['projection_source']['row_slots']==(2,5)


@pytest.mark.parametrize('fault',['bool_slot','duplicate_slot','negative_count','wrong_length','unordered',
    'duplicate_row','outside','bool_row','unknown_field'])
def test_ambiguous_explicit_projection_provenance_rejected(fault):
    actions,instances,images=supplied()
    p=actions[0]['projection_source']=dict(action_slot=4,row_slots=[2,5],row_count=6)
    if fault=='bool_slot':p['action_slot']=True
    elif fault=='duplicate_slot':actions[1]['projection_source']=dict(action_slot=4,row_slots=[0],row_count=1)
    elif fault=='negative_count':p['row_count']=-1
    elif fault=='wrong_length':p['row_slots']=[2]
    elif fault=='unordered':p['row_slots']=[5,2]
    elif fault=='duplicate_row':p['row_slots']=[2,2]
    elif fault=='outside':p['row_slots']=[2,6]
    elif fault=='bool_row':p['row_slots']=[True,5]
    elif fault=='unknown_field':p['extra']=0
    with pytest.raises(ValueError):parse_vcoco_role_reference(actions,instances,images)

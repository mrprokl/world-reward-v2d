"""Manufactured saved-bank ABI only; no actual RGB, DWPose, model or references."""
from copy import deepcopy
from dataclasses import fields
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import vcoco_interaction_observations as p
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.owlv2_object_observations import Owlv2ObjectObservations
from world_reward.owlv2_candidate_bridge import bridge_owlv2_candidates
from world_reward.interaction_candidate_evidence import build_interaction_candidate_evidence


def identities(arrays):
    return {n:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            for n,a in arrays.items()}


def fixture(n=2, k=True):
    image_id='a'*32;size=(8,12);ids=np.array([f'image:{image_id}/person/retained:{i:06d}'for i in range(n)],dtype=str)
    boxes=np.array([[1+i,1,4+i,7]for i in range(n)],np.float64).reshape(n,4)
    scores=np.full(n,.8,np.float64)
    pose=PersonPoseObservations(0,size,tuple(ids.tolist()),boxes,scores,
        np.full((n,133,2),3.,np.float64),np.ones((n,133),np.float32))
    classes=np.array([0,0,1,1,2]if k else [],np.int64);m=len(classes)
    hp=np.array([[0,2],[0,3],[1,2],[1,3]],np.int64)if k else np.empty((0,2),np.int64)
    op=np.array([[2,4],[3,4]],np.int64)if k else np.empty((0,2),np.int64)
    hb=np.tile(np.array([[2,2,5,5]],np.float32),(m,1))
    hoi=HOIDetrObservations(0,size,np.zeros((1500,3),np.float32),np.zeros((1500,4),np.float32),
        np.zeros((1500,256),np.float32),np.column_stack((hb,np.full(m,.6,np.float32))),
        np.arange(m,dtype=np.int64),np.arange(m,dtype=np.int64),
        np.array([1,2,1,3,4]if k else [],np.int64),classes,hb,np.full(m,.7,np.float32),np.full(m,.6,np.float32),
        hp,np.tile(np.array([[-.5,.25]],np.float32),(len(hp),1)),op,np.zeros((len(op),2),np.float32))
    padded=np.tile(np.array([[.4,.4,.2,.2]],np.float32),(3600,1))
    corners=np.concatenate((padded[:,:2]-padded[:,2:]/2,padded[:,:2]+padded[:,2:]/2),axis=1)*np.float32(max(size))
    owl=Owlv2ObjectObservations(0,size,(60,60),np.arange(3600,dtype=np.int64),padded,
        np.arange(3600,dtype=np.float32)-1800,corners)
    e=dict(person_raw_boxes=boxes.astype(np.float32),person_raw_scores=scores.astype(np.float32),
        person_raw_labels=np.array(['person']*n,dtype=str),person_retained_boxes=boxes,person_retained_scores=scores,
        person_retained_raw_slots=np.arange(n,dtype=np.int64),person_retained_ids=ids,
        person_model_pred_boxes=np.zeros((1,900,4),np.float32),person_model_logits=np.full((1,900,256),-np.inf,np.float32),
        person_model_input_ids=np.array([[101,102]],np.int64),person_model_attention_mask=np.ones((1,2),np.int64),
        owl_patch_ids=owl.patch_ids,owl_boxes_padded_normalized_cxcywh=owl.boxes_padded_normalized_cxcywh,
        owl_objectness_logits=owl.objectness_logits,owl_boxes_original_xyxy=owl.boxes_original_xyxy,
        image_size=np.array(size,np.int64),original_frame_index=np.array(0,np.int64))
    e['person_model_logits'][:,:,:2]=0
    po={f.name:getattr(pose,f.name)for f in fields(pose)if type(getattr(pose,f.name))is np.ndarray}
    po.update(person_ids=ids.copy(),image_size=np.array(size,np.int64),original_frame_index=np.array(0,np.int64),
        original_slot=np.array(2,np.int64),acquired_ordinal=np.array(1,np.int64))
    ho={f.name:getattr(hoi,f.name)for f in fields(hoi)if type(getattr(hoi,f.name))is np.ndarray}
    ho.update(image_size=np.array(size,np.int64),original_frame_index=np.array(0,np.int64),
        original_slot=np.array(2,np.int64),acquired_ordinal=np.array(1,np.int64))
    pin=dict(bytes=1234,sha256='b'*64)
    common=dict(image_id=image_id,original_slot=2,acquired_ordinal=1,original_frame_index=0,image_size=list(size),file='image_000002.npz')
    er=dict(common,arrays=identities(e),person_ids=ids.tolist(),person_retained_rows=n,identity=pin,
        person_native_queries=900,person_postprocessor_rows=n,owl_patches=3600)
    pr=dict(common,arrays=identities(po),person_ids=ids.tolist(),persons=n)
    hr=dict(common,arrays=identities(ho),source_person_ids=ids.tolist(),endpoint_bank_identity=dict(pin),
        native_detections=m,hand_object_pairs=len(hp),object_target_pairs=len(op),owl_patches=3600)
    return [er,e,pr,po,hr,ho],(pose,hoi,owl)


def test_all_raw_arrays_native_ids_and_complete_factorized_graph():
    args,(pose,hoi,owl)=fixture();before=[identities(args[i])for i in (1,3,5)]
    value=p.reconstruct_interaction(*args)
    expected=build_interaction_candidate_evidence(pose,bridge_owlv2_candidates(owl),hoi)
    assert value.image_id=='a'*32 and value.original_slot==2 and value.acquired_ordinal==1
    assert value.evidence.source_person_ids==pose.person_ids
    assert value.evidence.features.shape==(2*2*3600,10)
    assert value.evidence.hoi_evidence.features.shape==(2*2*4,15)
    assert value.evidence.route_features.shape==(4*3600,2)
    assert np.array_equal(value.hoi.query_ids,np.array([1,2,1,3,4]))  # cross-class query alias retained
    assert value.evidence.source_observation_references==expected.source_observation_references
    for actual,wanted in ((value.evidence.features,expected.features),(value.evidence.route_features,expected.route_features),
                          (value.evidence.hoi_evidence.features,expected.hoi_evidence.features)):
        assert actual.dtype==wanted.dtype and actual.shape==wanted.shape and actual.tobytes()==wanted.tobytes()
    for index,arrays in enumerate((value.endpoint_arrays,value.pose_arrays,value.hoi_arrays)):
        assert identities(arrays)==before[index]
        for a in arrays.values():
            with pytest.raises(ValueError):a.setflags(write=True)
    args[1]['person_model_logits'][0,0,0]=9
    assert value.endpoint_arrays['person_model_logits'][0,0,0]==0
    assert value.evidence.scope['scoring_performed'] is False


@pytest.mark.parametrize('n,k',[(0,True),(0,False),(2,False)])
def test_genuine_empty_populations_not_missing_pose_fallback(n,k):
    args,_=fixture(n,k);value=p.reconstruct_interaction(*args)
    count=4 if k else 0
    assert value.evidence.features.shape==(n*2*3600,10)
    assert value.evidence.hoi_evidence.features.shape==(n*2*count,15)
    assert value.evidence.route_features.shape==(count*3600,2)
    assert value.evidence.scope['hoi_provided']is True and value.evidence.scope['absence_predictions_generated']is False
    assert len(value.owl.patch_ids)==3600


def test_missed_and_offgrid_pose_and_zeroarea_owl_are_retained():
    args,_=fixture();po=args[3]
    po['keypoints_original_xy']=po['keypoints_original_xy'].copy();po['raw_scores']=po['raw_scores'].copy()
    po['keypoints_original_xy'][0,9]=[-4,100];po['raw_scores'][0,9]=0
    po['keypoints_original_xy'][1,91]=[-2,40]
    po['native_valid']=po['raw_scores']>0
    xy=po['keypoints_original_xy'];po['in_original_image']=(xy[...,0]>=0)&(xy[...,0]<12)&(xy[...,1]>=0)&(xy[...,1]<8)
    args[2]['arrays']=identities(po)
    e=args[1];e['owl_boxes_padded_normalized_cxcywh']=e['owl_boxes_padded_normalized_cxcywh'].copy()
    e['owl_boxes_padded_normalized_cxcywh'][0]=[.05,.05,0,0]
    b=e['owl_boxes_padded_normalized_cxcywh'];e['owl_boxes_original_xyxy']=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),axis=1)*np.float32(12)
    args[0]['arrays']=identities(e)
    value=p.reconstruct_interaction(*args)
    assert not value.person.native_valid[0,9] and not value.person.in_original_image[1,91]
    assert np.isnan(value.evidence.features[0,0])and not value.evidence.feature_supported[0,0]
    assert np.array_equal(value.owl.boxes_original_xyxy[0],e['owl_boxes_original_xyxy'][0])
    assert len(value.evidence.objects.object_ids)==3600


@pytest.mark.parametrize('fault',[
    'missing_pose','extra','badpin','arraybytes','opaque','slot','ordinal','frame','grid','idorder','idalias',
    'boxchange','scorechange','flags','dtype','scalar','endpointref','malformedpin','truncatedowl','truncatedhoi','pairdrop'])
def test_malformed_or_mismatched_saved_transport_fails_closed(fault):
    args,_=fixture();args=[deepcopy(a)for a in args]
    if fault=='missing_pose':args[3]=None
    elif fault=='extra':args[1]['unknown']=np.array(0)
    elif fault=='badpin':args[0]['arrays']['owl_objectness_logits']['sha256']='0'*64
    elif fault=='arraybytes':args[1]['person_model_logits'][0,0,0]=1
    elif fault=='opaque':args[2]['image_id']='c'*32
    elif fault=='slot':args[2]['original_slot']=3
    elif fault=='ordinal':args[4]['acquired_ordinal']=0
    elif fault=='frame':args[0]['original_frame_index']=1
    elif fault=='grid':args[2]['image_size']=[8,11]
    elif fault=='idorder':args[2]['person_ids'].reverse()
    elif fault=='idalias':args[4]['source_person_ids'][0]=args[4]['source_person_ids'][1]
    elif fault in ('boxchange','scorechange','flags','dtype','scalar'):
        name={'boxchange':'boxes_original_xyxy','scorechange':'detector_scores','flags':'native_valid',
              'dtype':'keypoints_original_xy','scalar':'original_slot'}[fault]
        if fault=='dtype':args[3][name]=args[3][name].astype(np.float32)
        elif fault=='scalar':args[3][name]=np.array(2.,np.float64)
        else:args[3][name].flat[0]=False if fault=='flags'else 2
        args[2]['arrays']=identities(args[3])
    elif fault=='endpointref':args[4]['endpoint_bank_identity']['sha256']='c'*64
    elif fault=='malformedpin':args[0]['arrays']['owl_patch_ids']['shape']=[True]
    elif fault=='truncatedowl':
        args[1]['owl_patch_ids']=args[1]['owl_patch_ids'][:-1];args[0]['arrays']=identities(args[1])
    elif fault in ('truncatedhoi','pairdrop'):
        name='query_tokens'if fault=='truncatedhoi'else'hand_object_pairs'
        args[5][name]=args[5][name][:-1];args[4]['arrays']=identities(args[5])
    with pytest.raises(ValueError):p.reconstruct_interaction(*args)


@pytest.mark.parametrize('fault',['raw_shape','raw_dtype','raw_count','query_count','patch_count',
    'slotbounds','tokens','mask','masked_tail','active_inf','hoi_count'])
def test_selfconsistent_forged_array_metadata_does_not_bypass_native_abi(fault):
    args,_=fixture();e=args[1]
    if fault=='raw_shape':e['person_raw_boxes']=e['person_raw_boxes'][:1]
    elif fault=='raw_dtype':e['person_raw_scores']=e['person_raw_scores'].astype(np.float64)
    elif fault=='raw_count':args[0]['person_postprocessor_rows']=1
    elif fault=='query_count':args[0]['person_native_queries']=899
    elif fault=='patch_count':args[4]['owl_patches']=3599
    elif fault=='slotbounds':e['person_retained_raw_slots']=np.array([0,2],np.int64)
    elif fault=='tokens':e['person_model_input_ids']=np.array([[-1,2]],np.int64)
    elif fault=='mask':e['person_model_attention_mask']=np.array([[1,2]],np.int64)
    elif fault=='masked_tail':e['person_model_logits'][0,0,2]=0
    elif fault=='active_inf':e['person_model_logits'][0,0,0]=-np.inf
    elif fault=='hoi_count':args[4]['hand_object_pairs']=3
    args[0]['arrays']=identities(e)
    with pytest.raises(ValueError):p.reconstruct_interaction(*args)


def test_person_coordinate_aliases_and_missing_data_not_reordered_or_fused():
    args,_=fixture();e,po=args[1],args[3]
    e['person_retained_boxes']=e['person_retained_boxes'].copy();e['person_retained_boxes'][1]=e['person_retained_boxes'][0]
    po['boxes_original_xyxy']=e['person_retained_boxes'].copy()
    po['raw_scores']=np.zeros((2,133),np.float32);po['native_valid']=np.zeros((2,133),bool)
    args[0]['arrays']=identities(e);args[2]['arrays']=identities(po)
    value=p.reconstruct_interaction(*args)
    assert len(value.person.person_ids)==2 and value.person.person_ids[0]!=value.person.person_ids[1]
    assert np.array_equal(value.person.boxes_original_xyxy[0],value.person.boxes_original_xyxy[1])
    assert value.evidence.features.shape==(14400,10)
    assert np.isnan(value.evidence.features[:,:5]).all()
    assert value.hoi.query_ids.tolist()==[1,2,1,3,4]


def test_construction_time_original_array_mutation_fails(monkeypatch):
    import world_reward.interaction_candidate_evidence as builder
    args,_=fixture();original=builder.build_interaction_candidate_evidence
    def changed(*a):
        result=original(*a);args[1]['person_model_logits'][0,0,0]=1;return result
    monkeypatch.setattr(builder,'build_interaction_candidate_evidence',changed)
    with pytest.raises(ValueError,match='Source arrays mutated'):p.reconstruct_interaction(*args)


def test_host_import_is_stdlib_only_and_no_orchestrator_or_io():
    root=Path(__file__).resolve().parents[1]
    code=f'''import sys,importlib.abc
sys.path.insert(0,{str(root/'infra')!r})
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','torch','PIL','onnxruntime','transformers'):raise AssertionError('ML eager import')
sys.meta_path.insert(0,Deny())
import vcoco_interaction_observations
print('stdlib')
'''
    r=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout=='stdlib\n'
    src=(root/'infra/vcoco_interaction_observations.py').read_text()
    assert 'np.load'not in src and '.open('not in src and 'import vcoco_'not in src

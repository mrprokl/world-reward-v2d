"""Continuous pixel-IoU positive role-pair retrieval, NOT official V-COCO AP.

Only evaluation sees references. Automatic aliases are exact coordinate groups,
not physical identities. Unknown alternatives/missing roles are never negatives,
OFF, contact, anatomical ownership or task-target truth. Caller authenticates
source bytes, image grid, complete banks and split before this pure API.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .vcoco_role_reference import (CocoInstanceReference,LocalizedPositivePair,VcocoRoleReference,
                                   VcocoRoleRow,parse_vcoco_role_reference)

IOU_THRESHOLD = .5


def _require(ok,message):
    if not ok:raise ValueError(message)


def _sealed(value):
    value=np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def _snapshot(value,name):
    _require(type(value)is np.ndarray and value.dtype in (np.float32,np.float64),name+': plain FP32/FP64 required')
    return np.array(value,dtype=np.float64,copy=True,order='C')


def _boxes(value,name):
    box=_snapshot(value,name)
    _require(box.ndim==2 and box.shape[1:]==(4,)and np.isfinite(box).all(),name+': finite pixel Nx4 xyxy required')
    with np.errstate(over='ignore',invalid='ignore'):
        extent=box[:,2:]-box[:,:2];area=extent[:,0]*extent[:,1]
    _require(np.all(extent>=0)and np.isfinite(extent).all()and np.isfinite(area).all(),name+': finite noninverted geometry required')
    return box  # Zero-area/off-grid proposals remain, without clipping.


def _ids(value,count):
    _require(type(value)is tuple and len(value)==count and all(type(s)is str and s and s.strip()==s for s in value)
        and len(set(value))==count,'Every unique original proposal ID required')
    return value


def _aliases(boxes):
    seen={};first=[];inverse=[]
    for i,box in enumerate(boxes):
        key=tuple(box)
        if key not in seen:seen[key]=len(first);first.append(i)
        inverse.append(seen[key])
    return np.asarray(first,np.int64),np.asarray(inverse,np.int64)


def _match(boxes,reference,ids,valid):
    result=np.full(len(boxes),-1,np.int64)
    with np.errstate(over='ignore',invalid='ignore'):
        width=reference[:,2:]-reference[:,:2];area=np.prod(np.maximum(width,0.),axis=1)
    _require(np.isfinite(area).all(),'Finite represented reference area required')
    for i,box in enumerate(boxes):
        intersection=np.prod(np.maximum(0.,np.minimum(box[2:],reference[:,2:])-np.maximum(box[:2],reference[:,:2])),axis=1)
        with np.errstate(over='ignore',invalid='ignore'):union=np.prod(box[2:]-box[:2])+area-intersection
        _require(np.isfinite(union).all(),'Finite represented IoU union required')
        iou=np.divide(intersection,union,out=np.zeros(len(reference)),where=union>0)
        matches=np.flatnonzero(iou>=IOU_THRESHOLD)
        # All classes/crowds/context are considered BEFORE endpoint censoring.
        if len(matches)==1 and valid[matches[0]]:result[i]=ids[matches[0]]
    return result


def _source_fingerprint(instances,reference):
    _require(type(instances)is list and type(reference)is VcocoRoleReference,'Plain original source records required')
    raw=json.dumps(instances,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest(),repr(reference)


def _reference(reference,instances,image_id,image_size):
    _require(type(reference)is VcocoRoleReference and type(instances)is list,'Original parsed reference plus ALL same-image instances required')
    _require(type(image_id)is int and image_id>0 and type(image_size)is tuple and len(image_size)==2
        and all(type(x)is int and x>0 for x in image_size),'Authenticated original image ID/(height,width) required')
    _require(all(type(r)is dict and r.get('image_id')==image_id for r in instances),'Only complete same-image instance context required')
    parse_vcoco_role_reference([],instances,[dict(id=image_id,height=image_size[0],width=image_size[1])])
    _require(type(reference.rows)is tuple and type(reference.localized_positive_pairs)is tuple
        and all(type(row)is VcocoRoleRow for row in reference.rows)
        and all(type(pair)is LocalizedPositivePair for pair in reference.localized_positive_pairs),'Original parser rows/pairs required')
    index={r['id']:r for r in instances};_require(all(i<=np.iinfo(np.int64).max for i in index),'Representable native annotation IDs required')
    ids=np.asarray(list(index),np.int64)
    boxes=np.asarray([r['bbox']for r in instances],np.float64).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]  # Original XYWH -> continuous XYXY, no +1/clipping.
    _require(np.isfinite(boxes).all(),'Representable original reference endpoints required')
    def endpoint(e):
        _require(type(e)is CocoInstanceReference,'Original parsed endpoint required')
        r=index.get(e.annotation_id)
        _require(r is not None and (e.image_id,e.category_id,e.iscrowd,e.bbox_xywh,e.area)==
            (image_id,r['category_id'],r['iscrowd'],tuple(r['bbox']),r['area']),'Role/reference instance binding differs')
        x,y,w,h=r['bbox'];inside=w>0 and h>0 and x>=0 and y>=0 and x+w<=image_size[1]and y+h<=image_size[0]
        _require(e.raw_box_positive==(w>0 and h>0)and e.positive_area==(r['area']>0)
            and e.raw_box_inside_image==inside,'Reference flags/original grid differ')
    actual={};diagnostic=dict(positive_role_slots=0,missing_positive_role_slots=0,unscorable_positive_role_slots=0,
        person_role_slots=0,self_role_slots=0,crowd_role_slots=0)
    for row in reference.rows:
        if row.image_id!=image_id:continue
        _require(row.label in (0,1)and len(row.role_names)==len(row.role_object_ids)==len(row.positive_role_endpoints)
            ==len(row.nonperson_pair_eligible)==len(row.pair_unscorable_reasons)and row.role_names[0]=='agent',
            'Exact original role slots required')
        endpoint(row.agent)
        for role,(rid,e,ok,reasons)in enumerate(zip(row.role_object_ids,row.positive_role_endpoints,row.nonperson_pair_eligible,row.pair_unscorable_reasons)):
            if e is not None:endpoint(e)
            if row.label!=1 or role==0:continue
            diagnostic['positive_role_slots']+=1
            diagnostic['missing_positive_role_slots']+=rid==0
            diagnostic['unscorable_positive_role_slots']+=not ok
            diagnostic['person_role_slots']+='person_role_not_nonperson_target'in reasons
            diagnostic['self_role_slots']+='self_role'in reasons
            diagnostic['crowd_role_slots']+=bool({'crowd_agent','crowd_target'}&set(reasons))
            if ok:actual.setdefault((row.agent_annotation_id,rid),[]).append((row.action_slot,row.row_slot,role,row.action_name,row.role_names[role]))
    positives={}
    for pair in reference.localized_positive_pairs:
        if pair.image_id==image_id:
            key=(pair.agent_annotation_id,pair.object_annotation_id)
            _require(key not in positives,'Unique localized native pair weight required');positives[key]=pair.row_role_references
    _require(positives=={k:tuple(v)for k,v in actual.items()},'Localized pair/action references differ')
    valid=np.asarray([r['iscrowd']==0 and r['area']>0 and r['bbox'][2]>0 and r['bbox'][3]>0 for r in instances],bool)
    isperson=np.asarray([r['category_id']==1 for r in instances],bool)
    return boxes,ids,valid&isperson,valid&~isperson,set(positives),diagnostic


@dataclass(frozen=True,eq=False)
class VcocoPairRetrieval:
    status: str
    image_positive_retrieval: float
    top_score: float | None
    top_unique_pair_count: int
    top_known_positive_count: int
    localized_positive_pair_count: int
    retrieved_positive_pair_count: int
    positive_pair_proposal_recall: float | None
    supported_positive_pair_count: int
    supported_positive_pair_recall: float | None
    positive_person_proposal_recall: float | None
    positive_object_proposal_recall: float | None
    raw_proposal_counts: tuple
    unique_proposal_counts: tuple
    supported_unique_pair_count: int
    person_ids: tuple
    object_ids: tuple
    person_group_indices: np.ndarray
    object_group_indices: np.ndarray
    person_reference_annotation_ids: np.ndarray
    object_reference_annotation_ids: np.ndarray
    role_diagnostics: object
    scope: object


def evaluate_vcoco_pair_retrieval(person_boxes,object_boxes,scores,support,reference,instances,
                                 image_id,image_size,person_ids,object_ids):
    """Full frozen NxM scores/support; no inference filtering or target repair.

    Supported scores are finite. Unsupported scores are raw NaN, not fabricated
    OFF scores; no supported candidates yields0 missed positive retrieval.
    Exact-coordinate aliases must agree in score AND support across their full
    rows/columns, then receive one tie weight per unique P/O coordinate pair.
    The denominator is unique localized positive pairs, not missing/unlocalized
    roles; these remain explicit diagnostics. No localized pair is unscorable.
    """
    source=_source_fingerprint(instances,reference)
    _require(all(type(a)is np.ndarray for a in(person_boxes,object_boxes,scores,support)),'Plain source arrays required')
    original_arrays=tuple((a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest())for a in(person_boxes,object_boxes,scores,support))
    p,o=_boxes(person_boxes,'Persons'),_boxes(object_boxes,'Objects');bank=_snapshot(scores,'Scores')
    _require(bank.shape==(len(p),len(o))and type(support)is np.ndarray and support.dtype==np.bool_
        and support.shape==bank.shape,'Complete NxM score/support bank required')
    mask=np.array(support,copy=True)
    _require(np.isfinite(bank[mask]).all()and np.isnan(bank[~mask]).all(),'Finite supported / NaN unsupported scores required')
    person_ids,object_ids=_ids(person_ids,len(p)),_ids(object_ids,len(o))
    pi,pa=_aliases(p);oi,oa=_aliases(o);unique=bank[np.ix_(pi,oi)];usable=mask[np.ix_(pi,oi)]
    _require(np.array_equal(mask,usable[np.ix_(pa,oa)])and np.array_equal(bank,unique[np.ix_(pa,oa)],equal_nan=True),
        'Exact numerical box aliases require equal frozen score/support')
    boxes,ids,pvalid,ovalid,positive,diagnostics=_reference(reference,instances,image_id,image_size)
    pm,om=_match(p,boxes,ids,pvalid),_match(o,boxes,ids,ovalid)
    persons,objects=set(pm[pm>=0]),set(om[om>=0]);rp,ro={a for a,b in positive},{b for a,b in positive}
    retrieved=sum(a in persons and b in objects for a,b in positive)
    supported_pairs={(int(pm[pi[a]]),int(om[oi[b]]))for a,b in zip(*np.nonzero(usable))}
    supported=len(positive&supported_pairs);top=None;tied=hit=0
    if np.any(usable):
        top=float(np.max(unique[usable]));a,b=np.nonzero(usable&(unique==top));tied=len(a)
        hit=sum((int(pm[pi[x]]),int(om[oi[y]]))in positive for x,y in zip(a,b))
    _require(source==_source_fingerprint(instances,reference),'Reference source changed during evaluation')
    _require(original_arrays==tuple((a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).digest())for a in(person_boxes,object_boxes,scores,support)),
        'Automatic source bank changed during evaluation')
    count=len(positive)
    scope=MappingProxyType(dict(official_AP=False,continuous_pixel_iou=.5,contact_verified=False,ownership_verified=False,
        anatomical_side_verified=False,task_target_verified=False,unknown_pairs_verified_negative=False,
        alias_identity='exact_numerical_coordinates_not_physical',source_authenticated=False))
    return VcocoPairRetrieval('scorable_localized_positive'if count else'no_localized_positive',hit/tied if tied else 0.,top,tied,hit,
        count,retrieved,retrieved/count if count else None,supported,supported/count if count else None,
        len(rp&persons)/len(rp)if rp else None,len(ro&objects)/len(ro)if ro else None,
        (len(p),len(o)),(len(pi),len(oi)),int(usable.sum()),person_ids,object_ids,_sealed(pa),_sealed(oa),_sealed(pm),_sealed(om),
        MappingProxyType(diagnostics),scope)

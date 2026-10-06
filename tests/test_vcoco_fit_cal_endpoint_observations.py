"""Tiny authored all48 adapters; no real images, models or private references."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import vcoco_fit_cal_endpoint_observations as p


def fixture(tmp_path):
    inputs=tmp_path/'inputs';out=tmp_path/'out';inputs.mkdir();out.mkdir()
    rows=[]
    for i in range(48):
        path=inputs/f'image_{i:06d}.jpg';path.write_bytes(f'authored{i}'.encode());path.chmod(0o400)
        rows.append(dict(image_id=f'{i+1:032x}',file=path.name,**p.rt.identity(path),width=4,height=3))
    value=dict(schema=p.public.SCHEMA,images=rows)
    manifest=inputs/'manifest.json';manifest.write_bytes(p.original.encode(value));manifest.chmod(0o400)
    inputs.chmod(0o500)
    return p.EndpointContext(inputs,out),value,p.rt.identity(manifest)


def arrays(row, persons=1):
    boxes=np.asarray([[0.,0.,2.,2.]]*persons,np.float32).reshape(-1,4)
    scores=np.full(persons,.7,np.float32);labels=['person']*persons
    person=p.original.gdi.retained_bank(boxes,scores,labels,row['image_id'],'person',4,3)
    result={'person_'+n:v for n,v in person.items()}
    logits=np.full((1,900,256),-np.inf,np.float32);logits[:,:,:3]=.125
    result.update(person_model_pred_boxes=np.zeros((1,900,4),np.float32),person_model_logits=logits,
        person_model_input_ids=np.asarray([[101,102,103]],np.int64),person_model_attention_mask=np.ones((1,3),np.int64),
        owl_patch_ids=np.arange(3600,dtype=np.int64),image_size=np.asarray([3,4],np.int64),
        original_frame_index=np.asarray(0,np.int64),owl_objectness_logits=np.zeros(3600,np.float32))
    padded=np.tile(np.asarray([.25,.5,.125,.25],np.float32),(3600,1))
    result['owl_boxes_padded_normalized_cxcywh']=padded
    result['owl_boxes_original_xyxy']=np.concatenate((padded[:,:2]-padded[:,2:]/np.float32(2),
        padded[:,:2]+padded[:,2:]/np.float32(2)),axis=1)*np.float32(4)
    return result


def fake_model(monkeypatch, events, fail=None):
    def bank(rgb,row,index,detect,model,operations):
        events.append(index)
        if fail==index:raise RuntimeError('Authored inference failure')
        a=arrays(row,persons=0 if index%2 else 1)
        return a,dict(image_id=row['image_id'],bank_index=index,original_frame_index=0,image_size=[3,4],
            input_file=row['file'],input_identity={k:row[k]for k in('bytes','sha256')},person_native_queries=900,
            owl_patches=3600,person_postprocessor_rows=len(a['person_raw_labels']),
            person_retained_rows=len(a['person_retained_ids']),person_ids=a['person_retained_ids'].tolist())
    monkeypatch.setattr(p.original,'bank_arrays',bank)
    monkeypatch.setattr(p.public,'decode_rgb',lambda *a,**k:np.zeros((3,4,3),np.uint8))
    op=SimpleNamespace(tensor_ops=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)))
    return (None,None,op,object())


def test_all48_public_manifest_strict_no_private_values(tmp_path):
    context,value,pin=fixture(tmp_path)
    assert p.public_inputs(context,pin)==value
    bad=deepcopy(value);bad['images'][0]['split']='FIT'
    path=context.inputs/'manifest.json';path.chmod(0o600);path.write_bytes(p.original.encode(bad));path.chmod(0o400)
    with pytest.raises(ValueError):p.public_inputs(context,p.rt.identity(path))


@pytest.mark.parametrize('fault',['count','slot','duplicate','grid','bytepin'])
def test_complete_population_validation_before_model(tmp_path,fault):
    context,value,_=fixture(tmp_path);row=value['images'][0]
    if fault=='count':value['images'].pop()
    elif fault=='slot':row['file']='image_000048.jpg'
    elif fault=='duplicate':value['images'][1]['image_id']=row['image_id']
    elif fault=='grid':row['width']=True
    else:row['sha256']='f'*64
    path=context.inputs/'manifest.json';path.chmod(0o600);path.write_bytes(p.original.encode(value));path.chmod(0o400)
    with pytest.raises(ValueError):p.public_inputs(context,p.rt.identity(path))


def test_original_full900_padding_and3600_inverse_empty_person_no_repair():
    row=dict(image_id='1'*32,width=4,height=3);a=arrays(row,0);before=p.array_identities(a)
    p.validate_arrays(a,row)
    assert p.array_identities(a)==before and a['person_retained_boxes'].shape==(0,4)
    assert np.isneginf(a['person_model_logits']).sum()==900*253


@pytest.mark.parametrize('fault',['missing','raw899','padding','positive_inf','owl3599','inverse','object','frame','person_id','person_slot'])
def test_invalid_native_arrays_reject(fault):
    row=dict(image_id='1'*32,width=4,height=3);a=arrays(row)
    if fault=='missing':del a['person_model_attention_mask']
    elif fault=='raw899':a['person_model_pred_boxes']=a['person_model_pred_boxes'][:,:899]
    elif fault=='padding':a['person_model_logits'][0,0,3]=0
    elif fault=='positive_inf':a['person_model_logits'][0,0,0]=np.inf
    elif fault=='owl3599':a['owl_patch_ids']=a['owl_patch_ids'][:-1]
    elif fault=='inverse':a['owl_boxes_original_xyxy'][0,0]+=1
    elif fault=='object':a['person_retained_ids']=np.asarray(['x'],object)
    elif fault=='frame':a['original_frame_index']=np.asarray(1,np.int64)
    elif fault=='person_id':a['person_retained_ids']=np.asarray(['invented'])
    else:a['person_retained_raw_slots'][0]=2
    with pytest.raises(ValueError):p.validate_arrays(a,row)


def test_complete_observe_all48_original_slots_and_lossless17_arrays(tmp_path,monkeypatch):
    context,value,pin=fixture(tmp_path);events=[];models=fake_model(monkeypatch,events)
    before=p.original.encode(value);records=p.observe(context,p.public_inputs(context,pin),models,time.monotonic()+20)
    assert events==list(range(48)) and len(records)==48 and p.original.encode(value)==before
    assert len(list(context.output.glob('image_*.npz')))==48
    p.validate_records(context,value,records,time.monotonic()+20)
    assert all(set(r['arrays'])==p.KEYS and r['owl_patches']==3600 for r in records)
    assert records[1]['person_retained_rows']==0
    records[0]['person_ids']=['invented']
    with pytest.raises(ValueError):p.validate_records(context,value,records,time.monotonic()+20)


def test_failure_preserves_incremental_rows_and_no_retry(tmp_path,monkeypatch):
    context,value,_=fixture(tmp_path);events=[];models=fake_model(monkeypatch,events,fail=2);records=[]
    with pytest.raises(RuntimeError):p.observe(context,value,models,time.monotonic()+20,records=records)
    assert events==[0,1,2]and len(records)==2
    with pytest.raises(ValueError):p.validate_records(context,value,records,time.monotonic()+20)


def test_all_input_bytes_checked_before_first_model_and_after_failure(tmp_path,monkeypatch):
    context,value,_=fixture(tmp_path);events=[];models=fake_model(monkeypatch,events)
    path=context.inputs/value['images'][-1]['file'];path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400)
    with pytest.raises(ValueError):p.observe(context,value,models,time.monotonic()+20)
    assert not events


def test_saved_array_mutation_rejected(tmp_path,monkeypatch):
    context,value,_=fixture(tmp_path);events=[];models=fake_model(monkeypatch,events);save=p.original.save_bank
    def mutate(out,index,a):
        record=save(out,index,a);a['owl_objectness_logits'][0]=1;return record
    monkeypatch.setattr(p.original,'save_bank',mutate)
    with pytest.raises(ValueError):p.observe(context,value,models,time.monotonic()+20)
    assert events==[0]


def test_public_metadata_mutation_on_forward_failure_rejected_in_finally(tmp_path,monkeypatch):
    context,value,_=fixture(tmp_path);events=[];models=fake_model(monkeypatch,events)
    def fail(*args):value['images'][0]['image_id']='f'*32;raise RuntimeError('authored')
    monkeypatch.setattr(p.original,'bank_arrays',fail)
    with pytest.raises(ValueError,match='metadata mutated'):p.observe(context,value,models,time.monotonic()+20)


def test_context_origin_checks_and_sourcepin_tamper(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();source=dict(helpers=deepcopy(p.REUSE))
    for name,pin in source['helpers'].items():
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());path.chmod(0o400)
    monkeypatch.setattr(p.rt,'identity',lambda path,*a,**k:deepcopy(p.REUSE[str(path.relative_to(code))]))
    for module,name in((p,'infra/vcoco_fit_cal_endpoint_observations.py'),(p.original,'infra/rgb_endpoint_bank.py'),
        (p.public,'src/world_reward/rgb_bank_inputs.py'),(p.rt,'infra/mediapipe_cpu_runtime_verify.py'),
        (p.original.owl,'infra/owlv2_native_qualify.py'),(p.original.gdi,'infra/openimages_joint_pair_gdi.py')):
        monkeypatch.setattr(module,'__file__',str(code/name))
    assert p.source_identity(code,source)==p.REUSE
    source['helpers']['infra/rgb_endpoint_bank.py']['sha256']='0'*64
    with pytest.raises(ValueError):p.source_identity(code,source)


def test_stdlib_import_and_no_hidden_model_lifecycle():
    script="""import sys,importlib.abc
sys.path[:0]=['infra','src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in ('numpy','torch','transformers','PIL','cv2'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
import vcoco_fit_cal_endpoint_observations
print('stdlib')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',script],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    tree=ast.parse(Path(p.__file__).read_text())
    calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
           and isinstance(n.func.value,ast.Name)and n.func.value.id=='original']
    assert 'bank_arrays'in calls and 'save_bank'in calls and not set(calls)&{'native','dispatch','load_models','run'}
    assert p.original.DATA!=Path('/srv/world-reward-data/vcoco_fit_cal_v1/inputs')
    assert p.PROFILE['person_query']=='person.'and p.PROFILE['images']==48


def test_deadline_and_context_safety(tmp_path):
    with pytest.raises(ValueError):p.EndpointContext(tmp_path,tmp_path/'nested')
    with pytest.raises(ValueError):p.check(float('nan'))
    with pytest.raises(TimeoutError):p.check(time.monotonic()-1)

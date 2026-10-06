"""Exact public callback AST plus authored all-person/empty/partial controls."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import vcoco_public_pose_core as p


def banks(count=48):
    images, rows, arrays = [], [], {}
    for i in range(count):
        n = i % 3; ids = tuple(f'person:{i}:{j}' for j in range(n))
        images.append(dict(slot=i)); rows.append(dict(slot=i, person_ids=list(ids), person_retained_rows=n, image_size=[8,12]))
        arrays[i] = dict(person_retained_ids=np.asarray(ids, dtype=str), person_retained_boxes=np.ones((n,4), np.float64), person_retained_scores=np.ones(n, np.float64))
    return images, rows, arrays


def callbacks(arrays, events, fail=None):
    from types import SimpleNamespace
    def load(bank, validate): events.append(('load', bank['slot'], validate)); return arrays[bank['slot']]
    def decode(image): events.append(('decode', image['slot'])); return np.zeros((8,12,3), np.uint8)
    def infer(rgb, ids, boxes, scores):
        events.append(('infer', len(ids)))
        if fail is not None and sum(e[0]=='infer' for e in events)==fail: raise RuntimeError('authored failure')
        return SimpleNamespace(person_ids=ids, original_frame_index=0, image_size=(8,12))
    def save(result, image, bank, ordinal): events.append(('save', ordinal)); return dict(slot=ordinal, count=len(result.person_ids))
    return load, decode, infer, save


def test_exact_original_callback_ast_all11_fields_and_public_closure():
    root=Path(__file__).resolve().parents[1]
    functions=lambda name:{n.name:n for n in ast.parse((root/'infra'/name).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert ast.dump(functions('vcoco_public_pose_core.py')['observe'],include_attributes=False)==ast.dump(functions('vcoco_person_pose_observations.py')['observe'],include_attributes=False)
    assert len(p.FIELDS)==11 and p.FIELDS[3:7]==('keypoints_original_xy','raw_scores','native_valid','in_original_image')
    tree=ast.parse((root/'infra/vcoco_public_pose_core.py').read_text())
    imports={a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names}
    assert imports=={'hashlib','mediapipe_cpu_runtime_verify'}


def test_full48_all_banks_before_first_callback_and_raw_population_retained():
    images,rows,arrays=banks();events=[];records=[]
    result,total=p.observe(images,rows,*callbacks(arrays,events),lambda:None,records)
    assert result is records and len(result)==48 and total==48
    assert events[:48]==[('load',i,True)for i in range(48)]
    assert [r['slot']for r in result]==list(range(48))
    assert [r['count']for r in result]==[i%3 for i in range(48)]


def test_original16_behavior_not_a_new_fixed48_driver():
    images,rows,arrays=banks(16);events=[]
    result,total=p.observe(images,rows,*callbacks(arrays,events),lambda:None)
    assert len(result)==16 and total==sum(i%3 for i in range(16))


def test_partial_failure_retained_not_replayed():
    images,rows,arrays=banks();events=[];records=[]
    with pytest.raises(RuntimeError):p.observe(images,rows,*callbacks(arrays,events,fail=3),lambda:None,records)
    assert len(records)==2 and sum(e[0]=='infer'for e in events)==3


def test_original_bank_mutation_rejected():
    images,rows,arrays=banks(2);events=[];load,decode,infer,save=callbacks(arrays,events)
    def mutate(*args):
        result=infer(*args)
        if len(args[1]): args[3][0]=.3
        return result
    with pytest.raises(ValueError,match='mutated'):p.observe(images,rows,load,decode,mutate,save,lambda:None)


def test_lazy_import_no_private_model_data_modules():
    root=Path(__file__).resolve().parents[1]
    script="""import importlib.abc,runpy,sys
sys.path.insert(0,sys.argv[1]+'/infra')
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in{'numpy','torch','transformers','PIL','vcoco_person_pose_observations','vcoco_observation_replica','vcoco_replica_completion','dwpose_smoke'}:raise ImportError('private/eager import')
sys.meta_path.insert(0,Deny());runpy.run_path(sys.argv[1]+'/infra/vcoco_public_pose_core.py')
"""
    assert subprocess.run([sys.executable,'-I','-B','-c',script,str(root)],capture_output=True).returncode==0

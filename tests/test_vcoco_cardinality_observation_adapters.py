"""Real authored11/19-array saves; supplied callbacks only, no native models."""
import ast
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import vcoco_cardinality_observation_adapters as p
import test_vcoco_public_observation_context as context_fixture
import test_vcoco_interaction_observations as hoi_fixture
from world_reward.person_pose_observations import infer_person_pose_frame
from world_reward.hoi_detr_observations import HOIDetrObservations
import vcoco_person_pose_observations as original_pose


def setup(tmp_path,monkeypatch):
    ref,args,projection=context_fixture.native_fixture(tmp_path,monkeypatch)
    root=Path(__file__).resolve().parents[1];helpers={}
    for name in p.HELPERS:
        path=args['code']/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((root/name).read_bytes());path.chmod(0o400);helpers[name]=p.rt.identity(path)
    args['current_source']['helpers']=helpers
    monkeypatch.setattr(p,'__file__',str(args['code']/'infra/vcoco_cardinality_observation_adapters.py'))
    import world_reward.person_pose_observations as pose_module
    import world_reward.hoi_detr_observations as hoi_module
    for module,name in ((pose_module,'person_pose_observations.py'),(hoi_module,'hoi_detr_observations.py')):
        monkeypatch.setattr(module,'__file__',str(args['code']/'src/world_reward'/name))
    reference=p.public.public_bank_reference(ref.context,projection_path=ref._projection_path,
        projection_pin=p.rt.strict(ref._projection_identity),code=args['code'],native_source={'helpers':helpers},
        deadline=time.monotonic()+30,native_mounts=False)
    monkeypatch.setattr(p.endpoint.public,'decode_rgb',lambda *a,**k:np.zeros((3,4,3),np.uint8))
    return reference


def pose_infer(events):
    def native(session,boxes,rgb):
        events.append(len(boxes));return np.full((len(boxes),133,2),2.,np.float64),np.ones((len(boxes),133),np.float32)
    return lambda rgb,ids,boxes,scores:infer_person_pose_frame(native,None,rgb,0,ids,boxes,scores)


def hoi(empty=False):
    value=hoi_fixture.fixture(0,k=not empty)[1][1]
    arrays={f.name:getattr(value,f.name)for f in fields(value)if type(getattr(value,f.name))is np.ndarray}
    return HOIDetrObservations(0,(3,4),**arrays)


def test_full48_pose_reuses_original_loop_preserves_empty_person_and_11arrays(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch);events=[]
    rows=p.observe_pose(ref,original_pose.observe,pose_infer(events),time.monotonic()+30)
    assert len(rows)==48 and len(events)==24 and rows[47]['persons']==0
    assert all(set(r['arrays'])==set(p.POSE_FIELDS)for r in rows)
    p.validate_saved_records(ref,'pose',rows,time.monotonic()+30)
    with np.load(ref.context.output/'image_000047.npz',allow_pickle=False)as z:
        assert z['keypoints_original_xy'].shape==(0,133,2)and int(z['original_slot'])==47


def test_full48_hoi_preserves1500_cartesian_pairs_and_slots47(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch);calls=[]
    def infer(rgb):calls.append(1);return hoi(empty=len(calls)%2==0)
    rows=p.observe_hoi(ref,infer,time.monotonic()+30)
    assert len(calls)==48 and len(rows)==48 and rows[0]['hand_object_pairs']==4 and rows[1]['hand_object_pairs']==0
    assert all(set(r['arrays'])==set(p.HOI_FIELDS)for r in rows)
    p.validate_saved_records(ref,'hoi',rows,time.monotonic()+30)
    with np.load(ref.context.output/'image_000047.npz',allow_pickle=False)as z:
        assert z['query_tokens'].shape==(1500,256)and int(z['original_slot'])==47


@pytest.mark.parametrize('kind',['pose','hoi'])
def test_input_last_bank_and_jpeg_checked_before_first_callback(tmp_path,monkeypatch,kind):
    ref=setup(tmp_path,monkeypatch);path=ref.context.endpoint_banks/'image_000047.npz';path.chmod(0o600);path.write_bytes(b'bad');path.chmod(0o400)
    calls=[]
    with pytest.raises(ValueError):
        if kind=='pose':p.observe_pose(ref,original_pose.observe,pose_infer(calls),time.monotonic()+30)
        else:p.observe_hoi(ref,lambda _:calls.append(1),time.monotonic()+30)
    assert not calls


@pytest.mark.parametrize('kind',['pose','hoi'])
def test_partial_failure_visible_no_retry_or_complete_pass(tmp_path,monkeypatch,kind):
    ref=setup(tmp_path,monkeypatch);records=[];calls=[];native=pose_infer([])
    def fail(*args):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('Authored no retry')
        return native(*args)if kind=='pose'else hoi()
    with pytest.raises(RuntimeError):
        if kind=='pose':p.observe_pose(ref,original_pose.observe,fail,time.monotonic()+30,records=records)
        else:p.observe_hoi(ref,fail,time.monotonic()+30,records=records)
    assert len(calls)==3 and len(records)==2
    p.validate_saved_records(ref,kind,records,time.monotonic()+30,complete=False)
    with pytest.raises(ValueError):p.validate_saved_records(ref,kind,records,time.monotonic()+30)


@pytest.mark.parametrize('fault',['person','score','flags','slot','scalar','bytes','frame','grid','source_ids','owl'])
def test_saved_pose_raw_ids_flags_grid_and_file_tampering_rejected(tmp_path,monkeypatch,fault):
    ref=setup(tmp_path,monkeypatch);image,bank=ref.images[0],ref.banks[0]
    a=p.public.load_bank(ref,bank,time.monotonic()+30)
    value=pose_infer([])(np.zeros((3,4,3),np.uint8),tuple(bank['person_ids']),a['person_retained_boxes'],a['person_retained_scores'])
    if fault=='person':object.__setattr__(value,'person_ids',('foreign',))
    elif fault=='score':object.__setattr__(value,'detector_scores',value.detector_scores+.1)
    elif fault=='flags':object.__setattr__(value,'native_valid',np.zeros_like(value.native_valid))
    if fault in('person','score','flags'):
        with pytest.raises(ValueError):p.save_observation(ref,'pose',value,image,bank,0,time.monotonic()+30)
        return
    row=p.save_observation(ref,'pose',value,image,bank,0,time.monotonic()+30)
    if fault=='slot':row['original_slot']=1
    elif fault=='scalar':row['persons']=2
    elif fault=='frame':row['original_frame_index']=1
    elif fault=='grid':row['image_size']=[4,3]
    elif fault=='source_ids':row['source_person_ids']=['foreign']
    elif fault=='owl':row['owl_patches']=3599
    else:
        path=ref.context.output/row['file'];path.chmod(0o600);path.write_bytes(b'bad');path.chmod(0o400)
    with pytest.raises(ValueError):p.validate_saved_records(ref,'pose',[row],time.monotonic()+30,complete=False)


def test_exclusive_save_and_borrowed_inputs_unchanged(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch);image,bank=ref.images[0],ref.banks[0];value=hoi();before=p.endpoint.array_identities({n:getattr(value,n)for n in p.HOI_FIELDS[:15]})
    row=p.save_observation(ref,'hoi',value,image,bank,0,time.monotonic()+30)
    with pytest.raises(FileExistsError):p.save_observation(ref,'hoi',value,image,bank,0,time.monotonic()+30)
    assert before==p.endpoint.array_identities({n:getattr(value,n)for n in p.HOI_FIELDS[:15]})
    assert (ref.context.output/row['file']).stat().st_mode&0o777==0o400


def test_actual_adapter_source_pin_checked_before_any_model_callback(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch);path=ref._code/'src/world_reward/hoi_detr_observations.py'
    path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400);calls=[]
    with pytest.raises(ValueError):p.observe_hoi(ref,lambda _:calls.append(1),time.monotonic()+30)
    assert not calls


def test_import_boundary_no_private_role_model_or_numeric_modules():
    root=Path(__file__).resolve().parents[1]
    script="""import importlib.abc,runpy,sys
sys.path[:0]=[sys.argv[1]+'/infra',sys.argv[1]+'/src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','transformers','PIL','vcoco_person_pose_observations','vcoco_hoi_observations','vcoco_fit_cal_acquisition_inputs','vcoco_fit_cal_context','vcoco_role_reference'}:raise ImportError('Forbidden import')
sys.meta_path.insert(0,Deny())
runpy.run_path(sys.argv[1]+'/infra/vcoco_cardinality_observation_adapters.py',run_name='authored')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(root)],capture_output=True)
    assert result.returncode==0,result.stderr
    for name in ('vcoco_public_observation_context.py','vcoco_cardinality_observation_adapters.py'):
        tree=ast.parse((root/'infra'/name).read_text())
        assert not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)for t in n.targets)for n in ast.walk(tree))


def test_save_rejects_foreign_public_metadata_before_creating_output(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch);image,bank=ref.images[0],ref.banks[0]
    image['image_id']='f'*32
    with pytest.raises(ValueError,match='public save row'):
        p.save_observation(ref,'hoi',hoi(),image,bank,0,time.monotonic()+30)
    assert not list(ref.context.output.iterdir())


def test_current_numerical_class_origin_checked_before_forward(tmp_path,monkeypatch):
    ref=setup(tmp_path,monkeypatch)
    import world_reward.hoi_detr_observations as hoi_module
    monkeypatch.setattr(hoi_module,'__file__',str(tmp_path/'foreign.py'));calls=[]
    with pytest.raises(ValueError,match='class origins'):
        p.observe_hoi(ref,lambda _:calls.append(1),time.monotonic()+30)
    assert not calls

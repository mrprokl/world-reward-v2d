"""Manufactured-only DEV ceiling and saved CPU boundary controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np
import pytest
import vcoco_pilot_pair_capacity as p
from world_reward.owlv2_object_observations import Owlv2ObjectObservations


def seal(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o400)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def authored():
    instances=[dict(id=i,image_id=501,category_id=cat,iscrowd=0,bbox=box,area=box[2]*box[3])for i,cat,box in
        [(11,1,[0,0,10,20]),(12,1,[60,0,10,20]),(31,17,[10,30,10,10]),(32,44,[60,30,10,10])]]
    actions=[dict(action_name='role',role_name=['agent','obj'],ann_id=[11,12],image_id=[501,501],label=[1,1],role_object_id=[11,12,31,32])]
    image=dict(id=501,width=100,height=80)
    ref=p.c.parse_vcoco_role_reference(actions,instances,[image])
    value=dict(schema='world_reward.vcoco_pilot_reference.v1',image=image,instances=instances,actions=actions,
        localized_positive_pairs=[dict(image_id=x.image_id,agent_annotation_id=x.agent_annotation_id,object_annotation_id=x.object_annotation_id,
            row_role_references=x.row_role_references)for x in ref.localized_positive_pairs])
    boxes=np.array([[0,0,10,20],[60,0,70,20]],np.float64)
    raw=np.zeros((3600,4),np.float32);raw[:2]=[[.15,.35,.1,.1],[.65,.35,.1,.1]]
    o=np.concatenate((raw[:,:2]-raw[:,2:]/np.float32(2),raw[:,:2]+raw[:,2:]/np.float32(2)),axis=1)*np.float32(100)
    objects=Owlv2ObjectObservations(0,(80,100),(60,60),np.arange(3600,dtype=np.int64),raw,np.zeros(3600,np.float32),o)
    return value,boxes,objects


def binding_fixture(acquired=range(16)):
    acquired=set(acquired);rows=[];native=[];mapping={}
    for slot in range(16):
        rows.append(dict(slot=slot,split='DEV'if slot<8 else'RESERVED',image_id=501,image=dict(id=501,width=100,height=80),
            reference_file=f'reference_{slot:06d}.json',reference_identity=dict(bytes=1,sha256='a'*64)))
        if slot in acquired:
            mapping[slot]=dict(public_image_id=f'{slot+1:032x}')
            native.append(dict(original_slot=slot,image_id=f'{slot+1:032x}',person_ids=['p0','p1'],file=f'image_{slot:06d}.npz'))
    return dict(values=dict(cohort=dict(records=rows),native=dict(images=native,proof_identity=dict(bytes=1,sha256='a'*64))),mapping=mapping,pins={n:dict(bytes=1,sha256='a'*64)for n in p.PIN_NAMES},revisions={n:v['revision']for n,v in p.PRODUCERS.items()})


def test_full3600_ceiling_no_topchoice_report_or_reserved_open():
    value,persons,objects=authored();binding=binding_fixture();called=[]
    def load(record):called.append(record['slot']);return deepcopy(value)
    out=p.evaluate(binding,{i:(persons,objects)for i in range(16)},load)
    assert called==list(range(8))and out['metrics']['slots']==8 and out['metrics']['pair_macro_fixed_slots']==1.
    assert out['decision']=='DEV_PAIR_PROPOSAL_CAPACITY_PASS_PENDING_OBSERVATIONS'
    assert all(row['proposal_objects']==3600 and row['localized_positive_pairs']==2 for row in out['per_image'])
    assert 'top_score'not in p.encode(out).decode()and 'image_positive_retrieval'not in p.encode(out).decode()


def test_missing_slots_keep8_denominator_and_never_load_missing_or_reserved():
    value,persons,objects=authored();binding=binding_fixture([0,1,8]);calls=[]
    out=p.evaluate(binding,{0:(persons,objects),1:(persons,objects),8:(persons,objects)},lambda r:(calls.append(r['slot'])or deepcopy(value)))
    assert calls==[0,1]and out['metrics']['pair_macro_fixed_slots']==.25 and len(out['per_image'])==8
    assert out['decision'].startswith('CLOSED_DEV_PAIR_PROPOSAL')and sum(r['missing']for r in out['per_image'])==6


def test_ambiguity_crowd_context_not_silently_positive_or_removed():
    value,persons,objects=authored();value['instances'].append(dict(id=99,image_id=501,category_id=44,iscrowd=1,bbox=[0,0,10,20],area=200))
    result=p.evaluate(binding_fixture(),{i:(persons,objects)for i in range(16)},lambda r:deepcopy(value))
    assert result['metrics']['pair_macro_fixed_slots']==.5 and result['decision'].startswith('CLOSED_')


@pytest.mark.parametrize('fault',['pairs','grid','schema','reserved','noninverted'])
def test_invalid_original_reference_or_box_is_not_repaired(fault):
    value,persons,objects=authored();binding=binding_fixture()
    if fault=='pairs':value['localized_positive_pairs']=[]
    elif fault=='grid':value['image']['width']=99
    elif fault=='schema':value['extra']='secret'
    elif fault=='reserved':binding['values']['cohort']['records'][0]['split']='RESERVED'
    else:persons[0,2]=-1
    with pytest.raises(ValueError):p.evaluate(binding,{i:(persons,objects)for i in range(16)},lambda r:deepcopy(value))


@pytest.mark.parametrize('fraction,passed',[(.7,True),(np.nextafter(.7,0.),False),(0.,False),(1.,True)])
def test_declared_gate_exact_no_posthoc_margin(fraction,passed):
    metrics=dict(slots=8,iou=.5,pair_gate=.7,missing_slot_recall=0.,pair_macro_fixed_slots=float(fraction),
        person_macro_fixed_slots=1.,object_macro_fixed_slots=1.,pair_gate_passed=passed)
    assert ('PASS_PENDING'in p.decision(metrics))is passed
    metrics['pair_gate_passed']=not passed
    with pytest.raises(ValueError):p.decision(metrics)


def test_original_validate_npz_complete17_fields_no_old_data_or_output(tmp_path):
    import world_reward.owlv2_object_observations as owl
    from unittest.mock import patch
    rgb=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    def detect(image):
        boxes=np.array([[0,0,3,4]],np.float32);tokens=np.array([[101,102]],np.int64);mask=np.ones((1,2),np.int64)
        logits=np.full((1,900,256),-np.inf,np.float32);logits[:,:,:2]=0
        return boxes,np.array([.8],np.float32),['person'],np.full((1,900,4),.5,np.float32),logits,tokens,mask,256
    _,_,objects=authored()
    ob=Owlv2ObjectObservations(0,(7,9),(60,60),np.arange(3600,dtype=np.int64),np.zeros((3600,4),np.float32),np.zeros(3600,np.float32),np.zeros((3600,4),np.float32))
    image=dict(image_id='a'*32,file='image_000000.jpg',bytes=1,sha256='b'*64,width=9,height=7)
    with patch.object(owl,'infer_owlv2_object_frame',lambda *a:ob):arrays,row=p.bank.original.bank_arrays(rgb,image,0,detect,None,None)
    row.update(p.bank.original.save_bank(tmp_path,0,arrays));persons,result=p.saved.validate_npz(tmp_path/row['file'],row)
    assert persons.shape==(1,4)and len(result.patch_ids)==3600
    row['arrays']['person_model_logits']['sha256']='c'*64
    with pytest.raises(ValueError):p.saved.validate_npz(tmp_path/row['file'],row)


def test_same_fd_late_publication_demotes_fail(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir(mode=0o700);s=out.stat();owner=[s.st_dev,s.st_ino,s.st_uid,s.st_gid]
    clock=[0.];monkeypatch.setattr(p.time,'monotonic',lambda:clock[0]);original=p.os.fsync
    def sync(fd):original(fd);clock[0]=11.
    monkeypatch.setattr(p.os,'fsync',sync);report=dict(status='pass',decision='PASS')
    p.publish(out/'host.json',report,10.,owner,seal=True)
    saved=json.loads((out/'host.json').read_bytes())
    assert saved['status']=='fail'and saved['publication_error_type']=='TimeoutError'and out.stat().st_mode&0o777==0o500


def test_publication_rejects_foreign_leaf_before_chmod(tmp_path):
    out=tmp_path/'out';out.mkdir(mode=0o700);s=out.stat();owner=[s.st_dev,s.st_ino,s.st_uid,s.st_gid]
    seal(out/'foreign',b'secret')
    with pytest.raises(ValueError):p.publish(out/'host.json',dict(status='pass'),time.monotonic()+1,owner,seal=True)
    assert not(out/'host.json').exists()


@pytest.mark.parametrize('ending',['','\n'])
def test_exact_owned_CID_name_cleanup_and_foreign_identity(ending,tmp_path,monkeypatch):
    path=tmp_path/'cid';path.write_text('a'*64+ending);removed=[False];calls=[]
    def command(args,deadline):
        calls.append(args)
        if args[1]=='inspect':return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'b'*40
        if args[1]=='rm':removed[0]=True
        if 'id='+'a'*64 in args:return ''if removed[0]else'a'*64
        return ''
    monkeypatch.setattr(p.bank,'command',command);p.cleanup(path,'owned','b'*40,time.monotonic()+1)
    assert path.stat().st_mode&0o777==0o400 and ['docker','rm','-f','a'*64]in calls
    monkeypatch.setattr(p.bank,'command',lambda a,d:'foreign'if a[1]=='inspect'else'a'*64)
    with pytest.raises(ValueError):p.cleanup(path,'owned','b'*40,time.monotonic()+1)


def test_cleanup_rm_success_renamed_CID_survival_is_failure(tmp_path,monkeypatch):
    path=tmp_path/'cid';path.write_text('a'*64);calls=[]
    def command(args,d):
        calls.append(args)
        if args[1]=='inspect':return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'b'*40
        if 'id='+'a'*64 in args:return'a'*64
        return ''
    monkeypatch.setattr(p.bank,'command',command)
    with pytest.raises(ValueError,match='survives'):p.cleanup(path,'owned','b'*40,time.monotonic()+1)
    assert [a for a in calls if a[1]=='rm']==[['docker','rm','-f','a'*64]]


def cli():
    return [item for n in p.PIN_NAMES for item in('--'+n+'-bytes','1','--'+n+'-sha256','a'*64)]+[
        '--bank-revision',p.PRODUCERS['bank']['revision'],'--prepare-revision',p.PRODUCERS['prepare']['revision']]


@pytest.mark.parametrize('fault',[None,'duplicate','partial','old','native_refs'])
def test_independent_arguments_no_reserved_phase_or_unknown_producer(fault):
    args=cli()
    if fault=='duplicate':args+=['--host-bytes','1']
    elif fault=='partial':args=args[:-2]
    elif fault=='old':args[-1]='b'*40
    elif fault=='native_refs':args=['--native']+args
    if fault:
        with pytest.raises(ValueError):p.arguments(args)
    else:
        out=p.arguments(args);assert out.host_bytes==1 and not out.native
    native=p.arguments(['--native','--proof-bytes','1','--proof-sha256','a'*64]);assert native.native


def test_source_import_closure_isolated_no_Torch_models_or_old_global_mutation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for name in p.NATIVE_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((root/name).read_bytes()if(root/name).exists()else b'{}')
    denied="import sys;sys.meta_path.insert(0,type('D',(),{'find_spec':lambda s,n,*a: (_ for _ in ()).throw(AssertionError(n)) if n.split('.')[0] in {'torch','transformers','onnxruntime'} else None})());"
    run=subprocess.run([sys.executable,'-I','-B','-c',denied+'import runpy;runpy.run_path('+repr(str(tmp_path/p.NATIVE_FILES[0]))+')'],capture_output=True,timeout=10)
    assert run.returncode==0,run.stderr.decode()
    source=(root/p.NATIVE_FILES[0]).read_text()
    assert 'saved.validate_npz('in source and 'bank.load_models('not in source and 'prep.reproduce('not in source
    assert 'bank.OUTPUT ='not in source and 'prep.DATA ='not in source
    wrapper=root/'infra/run_vcoco_pilot_pair_capacity.sh';assert subprocess.run(['bash','-n',str(wrapper)],capture_output=True).returncode==0
    assert '195s'in wrapper.read_text()and 'env -i'in wrapper.read_text()


def cpu_fixture(tmp_path,monkeypatch,fail_bank=None):
    code=tmp_path/'code';code.mkdir();out=tmp_path/'out';out.mkdir(mode=0o700);meta=tmp_path/'data/metadata';meta.mkdir(parents=True)
    value,persons,objects=authored();binding=binding_fixture();binding['values']['public']=dict(images=[{}]*16);binding['original_qualification']=dict(owl=dict(native_runtime=dict(wheel_RECORD_identities=dict(transformers={}))))
    helper={};frozen={}
    for name in p.NATIVE_FILES:helper[name]=seal(code/name,b'# originalsource')
    for i in range(8):binding['values']['cohort']['records'][i]['reference_identity']=seal(meta/f'reference_{i:06d}.json',p.encode(value))
    seal(meta/'report.json',b'{}');seal(meta/'cohort.json',b'{}')
    for row in binding['values']['native']['images']:
        row['identity']=seal(tmp_path/'banks'/row['file'],b'bank');frozen[str(tmp_path/'banks'/row['file'])]=row['identity']
    s=out.stat();proof=dict(binding=binding,source=dict(producer_revision='a'*40,helpers=helper),native_files=helper,
        frozen=frozen,image_id=p.IMAGE,output_owner=[s.st_dev,s.st_ino,s.st_uid,s.st_gid])
    pin=seal(out/'proof.json',p.encode(proof));monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p,'DATA',meta.parent)
    monkeypatch.setenv('WR_IMAGE_ID',p.IMAGE);monkeypatch.setattr(p,'configuration',lambda *a:p.FIXED)
    monkeypatch.setattr(p.bank,'configuration',lambda *a:({},{}));monkeypatch.setattr(p.bank.owl,'installed',lambda *a:dict(wheel_RECORD_identities=dict(transformers={})));monkeypatch.setattr(p.bank.original,'installed_grounding',lambda *a:{});monkeypatch.setattr(p.bank,'validate_report',lambda *a:None)
    monkeypatch.setattr(p.bank,'OUTPUT',tmp_path/'banks');old=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,pin,maximum=1<<20:{}if Path(path)in(code/p.bank.CONFIG,code/p.bank.owl.CONFIG) else old(path,pin,maximum))
    net=tmp_path/'net';net.mkdir();(net/'lo').mkdir();olditer=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:olditer(net)if path==Path('/sys/class/net')else olditer(path))
    calls=[]
    def validate(path,row):
        calls.append(row['original_slot'])
        if row['original_slot']==fail_bank:raise ValueError('PRIVATE_SECRET')
        return persons,objects
    monkeypatch.setattr(p.saved,'validate_npz',validate)
    return code,out,pin,calls


@pytest.mark.parametrize('fail_bank',[None,15])
def test_all16_full_banks_before_any_DEV_reference_even_RESERVED_bank_failure(tmp_path,monkeypatch,fail_bank):
    code,out,pin,calls=cpu_fixture(tmp_path,monkeypatch,fail_bank)
    reads=[];original=p.rt.pinned
    def pinned(path,pin,maximum=1<<20):
        if Path(path).name.startswith('reference_'):reads.append(len(calls))
        return original(path,pin,maximum)
    monkeypatch.setattr(p.rt,'pinned',pinned)
    result=p.cpu(code,'a'*40,pin,time.monotonic()+10)
    if fail_bank is None:
        assert result['status']=='pass'and calls==list(range(16))and reads==[16]*8
        assert result['reference_files_decoded']==8 and result['all16_banks_validated_before_DEV']
    else:
        assert result['status']=='fail'and calls==list(range(16))and not reads and result['reference_files_decoded']==0
    assert 'PRIVATE_SECRET'not in p.encode(result).decode()


def test_native_posthash_failure_cannot_pass_and_still_publishes(tmp_path,monkeypatch):
    code,out,pin,calls=cpu_fixture(tmp_path,monkeypatch)
    original=p.evaluate
    def evaluate(*a):
        result=original(*a);(code/p.NATIVE_FILES[-1]).chmod(0o600);(code/p.NATIVE_FILES[-1]).write_text('mutation');return result
    monkeypatch.setattr(p,'evaluate',evaluate);result=p.cpu(code,'a'*40,pin,time.monotonic()+10)
    assert result['status']=='fail'and result['post_error_type']=='ValueError'and(out/'native.json').exists()


def test_original_producer_reuse_rejects_changed_revision_before_source(tmp_path,monkeypatch):
    revisions=dict(bank='b'*40);calls=[];monkeypatch.setattr(p.rt,'source',lambda *a:calls.append(a))
    with pytest.raises(ValueError):p.original_source(tmp_path,'bank',revisions,p.bank)
    assert not calls


def test_current_config_helper_pins_full_original_closure_and_fixed_policy():
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/p.CONFIG).read_bytes())
    assert set(cfg)==set(p.FIXED)|{'helper_pins','producers'}and cfg['producers']==p.PRODUCERS
    assert all(type(cfg[k])is type(v)and cfg[k]==v for k,v in p.FIXED.items())
    assert set(cfg['helper_pins'])==set(p.HELPERS)-{p.NATIVE_FILES[0],p.CONFIG,'infra/run_vcoco_pilot_pair_capacity.sh'}
    for name,pin in cfg['helper_pins'].items():
        raw=(root/name).read_bytes();assert pin==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def host_fixture(tmp_path,monkeypatch,*,quality=True):
    root=tmp_path/'root';code=tmp_path/'code';code.mkdir();out=root/'results/capacity';out.parent.mkdir(parents=True)
    data=tmp_path/'data';(data/'metadata').mkdir(parents=True);bankout=tmp_path/'banks';bankout.mkdir()
    value,persons,objects=authored();binding=binding_fixture();frozen={};source=dict(producer_revision='a'*40,helpers={},markers={})
    for name in p.NATIVE_FILES:source['helpers'][name]=seal(code/name,b'# immutable source')
    for row in binding['values']['cohort']['records']:
        row['reference_identity']=seal(data/'metadata'/row['reference_file'],p.encode(value))
    for row in binding['values']['native']['images']:
        row['identity']=seal(bankout/row['file'],b'bank');frozen[str(bankout/row['file'])]=row['identity']
    binding['frozen']=frozen;binding['values']['public']=dict(images=[{}]*16);binding['original_qualification']=dict(owl=dict(native_runtime=dict(wheel_RECORD_identities=dict(transformers={}))))
    monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p,'DATA',data)
    monkeypatch.setattr(p.rt,'source',lambda *a:source);monkeypatch.setattr(p,'configuration',lambda *a:p.FIXED)
    monkeypatch.setattr(p,'authenticate',lambda *a:binding);monkeypatch.setattr(p,'image_state',lambda *a:dict(Id=p.IMAGE))
    monkeypatch.setattr(p.bank,'command',lambda *a:'');monkeypatch.setattr(p,'cleanup',lambda *a:None)
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd);(out/'.container.cid').write_bytes(b'd'*64)
        proof_pin=p.rt.identity(out/'proof.json',2 << 20);metrics=dict(slots=8,iou=.5,pair_gate=.7,missing_slot_recall=0.,
            pair_macro_fixed_slots=1. if quality else 0.,person_macro_fixed_slots=1.,object_macro_fixed_slots=1.,pair_gate_passed=quality)
        native=dict(status='pass',schema=p.FIXED['schema'],producer_revision='a'*40,proof_identity=proof_pin,runtime_identity=dict(wheel_RECORD_identities=dict(transformers={})),
            inputs=binding['pins'],stage='saved_cpu_vcoco_pair_proposals',phase='complete',models_loaded=0,
            GPU_used=False,RGB_decoded=False,FIT_performed=False,selector_evaluated=False,RESERVED_reference_values_read=False,
            all_acquired_banks_validated_before_DEV=True,source_inputs_predictions_references_rehashed_after=True,
            validated_bank_count=16,reference_files_decoded=8,metrics=metrics,decision=p.decision(metrics))
        seal(out/'native.json',p.encode(native));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run)
    return code,out,binding,source,calls


@pytest.mark.parametrize('quality',[True,False])
def test_host_narrow8DEV_mounts_noGPU_noRGB_and_technical_capacity_distinct(tmp_path,monkeypatch,quality):
    code,out,binding,source,calls=host_fixture(tmp_path,monkeypatch,quality=quality)
    result=p.dispatch(code,'a'*40,binding['pins'],binding['revisions'])
    assert result['status']=='pass'and result['metrics']['pair_gate_passed']is quality
    assert result['owned_cleanup_verified']and result['outputs_sealed']and out.stat().st_mode&0o777==0o500
    assert all(f.stat().st_mode&0o777==0o400 for f in out.iterdir())
    cmd=calls[0];mounts=[cmd[i+1]for i,s in enumerate(cmd)if s=='--mount']
    for i in range(8):assert any(f'reference_{i:06d}.json'in m for m in mounts)
    for i in range(8,16):assert not any(f'reference_{i:06d}.json'in m for m in mounts)
    assert not any('.jpg'in m or f'src={code},'in m for m in mounts)
    assert '--gpus'not in cmd and 'none'==cmd[cmd.index('--network')+1]and '6g'==cmd[cmd.index('--memory')+1]
    assert all(f'image_{i:06d}.npz'in '|'.join(mounts)for i in range(16))


@pytest.mark.parametrize('fault',['native','cleanup','source','late','image'])
def test_host_failures_posthash_and_sameFD_cannot_be_PASS(tmp_path,monkeypatch,fault):
    code,out,binding,source,calls=host_fixture(tmp_path,monkeypatch);actual=p.subprocess.run
    if fault=='cleanup':monkeypatch.setattr(p,'cleanup',lambda *a:(_ for _ in()).throw(ValueError('PRIVATE_SECRET')))
    elif fault=='image':monkeypatch.setattr(p,'image_state',lambda *a:(_ for _ in()).throw(ValueError('foreign')))
    def run(*a,**k):
        result=actual(*a,**k)
        if fault=='native':
            f=out/'native.json';v=json.loads(f.read_bytes());v['status']='fail';f.chmod(0o600);seal(f,p.encode(v))
        elif fault=='source':(code/p.NATIVE_FILES[-1]).chmod(0o600);(code/p.NATIVE_FILES[-1]).write_bytes(b'changed')
        return result
    monkeypatch.setattr(p.subprocess,'run',run)
    if fault=='late':
        clock=[0.];monkeypatch.setattr(p.time,'monotonic',lambda:clock[0]);original=p.os.fsync
        def sync(fd):
            original(fd)
            if(out/'host.json').exists():clock[0]=181.
        monkeypatch.setattr(p.os,'fsync',sync)
    result=p.dispatch(code,'a'*40,binding['pins'],binding['revisions'])
    assert result['status']=='fail'and json.loads((out/'host.json').read_bytes())['status']=='fail'
    assert 'PRIVATE_SECRET'not in p.encode(result).decode()


def test_host_posthash_calls_execute_after_native_failure(tmp_path,monkeypatch):
    code,out,binding,source,calls=host_fixture(tmp_path,monkeypatch);auth=p.authenticate;seen=[];actual=p.subprocess.run
    monkeypatch.setattr(p,'authenticate',lambda *a:(seen.append('auth')or auth(*a)))
    def run(*a,**k):actual(*a,**k);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(p.subprocess,'run',run)
    result=p.dispatch(code,'a'*40,binding['pins'],binding['revisions'])
    assert result['status']=='fail'and seen==['auth','auth']and result['source_inputs_predictions_references_rehashed_after']


@pytest.mark.parametrize('zero',['person','object'])
def test_empty_person_or_zero_area_complete3600_bank_scores_missing_not_OFF(zero):
    value,persons,objects=authored();binding=binding_fixture()
    if zero=='person':
        persons=np.empty((0,4),np.float64)
        for row in binding['values']['native']['images']:row['person_ids']=[]
    else:
        objects=Owlv2ObjectObservations(0,(80,100),(60,60),np.arange(3600,dtype=np.int64),np.zeros((3600,4),np.float32),
            np.zeros(3600,np.float32),np.zeros((3600,4),np.float32))
    result=p.evaluate(binding,{i:(persons,objects)for i in range(16)},lambda r:deepcopy(value))
    assert result['metrics']['pair_macro_fixed_slots']==0. and result['decision'].startswith('CLOSED_DEV_PAIR_PROPOSAL')
    assert all(r['proposal_objects']==3600 and r['supported_pair_proposal_recall']==0. for r in result['per_image'])


def test_image_metadata_same_explicitdaemon_no_env_secret_fields(monkeypatch):
    value=dict(Id=p.IMAGE,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=['sha256:'+'a'*64]*50));calls=[]
    monkeypatch.setattr(p.bank,'command',lambda args,d:(calls.append(args)or p.encode(value)))
    assert p.image_state(time.monotonic()+1)==value and calls[0][:3]==['docker','image','inspect']
    assert '.Config.Env'not in calls[0][-1]and '.RootFS'in calls[0][-1]
    value['Id']='sha256:'+'b'*64
    with pytest.raises(ValueError):p.image_state(time.monotonic()+1)

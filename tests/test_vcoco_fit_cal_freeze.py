"""Authored identity/source/publication controls; no real metadata or models."""
from copy import deepcopy
import ast
import hashlib
import os
from pathlib import Path
import stat
import time

import pytest
import vcoco_fit_cal_freeze as p


def save(path,value):
    raw=value if type(value)is bytes else p.c.encode(value)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o400)
    return p.c.pin(raw)


def root_stat(monkeypatch,parent):
    original=Path.lstat
    def projected(path,*args,**kwargs):
        s=original(path,*args,**kwargs)
        if path==parent or parent in path.parents:
            class S:pass
            v=S()
            for k in ('st_dev','st_ino','st_mode','st_size','st_uid','st_gid','st_nlink','st_mtime_ns','st_ctime_ns'):setattr(v,k,getattr(s,k))
            v.st_uid=v.st_gid=0;return v
        return s
    monkeypatch.setattr(Path,'lstat',projected)


def identity_fixture(tmp_path,monkeypatch):
    code=tmp_path/'current/code';code.mkdir(parents=True);source=code/'infra/vcoco_fit_cal_freeze.py';save(source,b'caller')
    monkeypatch.setattr(p,'__file__',str(source));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION','d'*40)
    monkeypatch.setattr(p.os,'geteuid',lambda:0);monkeypatch.setattr(p.sys,'platform','linux')
    class Host:nodename='world-reward-ncc-h100-02'
    monkeypatch.setattr(p.os,'uname',lambda:Host())
    data=tmp_path/'new';monkeypatch.setattr(p,'DATA',data);root_stat(monkeypatch,data)
    class Usage:ru_maxrss=1024
    monkeypatch.setattr(p.resource,'getrusage',lambda *_:Usage())
    original=tmp_path/'old';original.mkdir();monkeypatch.setattr(p.f,'OUTPUT',original)
    proof=dict(current=dict(binding=dict(producer_revision=p.selection.CENSUS_REVISION,entries=313,
        closure_sha256=p.selection.CENSUS_CLOSURE)),original_prepare={'original':1},original_census={'inputs':2},
        prepare_proof={'original':3},pilot={'references':'HASH_ONLY'})
    rows=[dict(split=n,image_id=start+i,photo_id=str(start+i+100000)) for n,start,count in (('train',1000,195),('val',10000,191)) for i in range(count)]
    ip=save(original/'inventory.json',rows);monkeypatch.setattr(p.selection,'INVENTORY_PIN',ip)
    report=dict(schema='world_reward.vcoco_fit_cal_census.v1',producer_revision=p.selection.CENSUS_REVISION,
        input_proof=proof,status='pass',stage='complete',decision='CAPACITY_METADATA_ONLY_NO_SELECTION',
        capacity_gate_passed=True,outputs_sealed=True,source_and_inputs_rehashed_after=True,historical_photos=448,
        minimum_distinct_photos=dict(train=32,val=16),artifact_identities={'inventory.json':ip},
        eligibility_inventory_identity=ip,eligibility_inventory_rows=386,fresh_reference_geometry_consulted=True,
        splits={n:dict(distinct_eligible_photos=v,eligible_images=v) for n,v in p.selection.COUNTS.items()})
    for k in ('historical_reference_values_read','pilot_reference_values_read','TEST_role_values_read','RGB_read',
        'network_used','GPU_used','models_loaded','FIT_performed','selection_performed','author_disjointness_verified',
        'challenge_overlap_verified','training_overlap_verified','adopted'):report[k]=False
    rp=save(original/'report.json',report);monkeypatch.setattr(p.selection,'REPORT_PIN',rp)
    excluded=dict(photos={str(i) for i in range(1,449)},authors=set(),md5=set())
    evidence=dict(current_source={'actual':'current'},original_input_proof=proof,live_states={'old':1},
        census_report_identity=rp,census_inventory_identity=ip)
    calls=[]
    def authenticate(*args):calls.append(args);return deepcopy(excluded),deepcopy(evidence)
    monkeypatch.setattr(p,'authenticate',authenticate)
    return data,calls,evidence


def test_source_constants_and_no_old_run_global_mutation():
    root=Path(p.__file__).resolve().parents[1]
    assert all(p.rt.identity(root/n,empty=True,readonly=False)==pin for n,pin in p.FROZEN.items())
    assert p.OLD['files']==308 and p.OLD['entries']==313 and p.OLD['revision']==p.selection.CENSUS_REVISION
    assert p.RECIPE['selected_slots']==48 and p.RECIPE['all_48_acquired_required'] is True
    assert p.RECIPE['budget_seconds']==1200 and p.RECIPE['outer_seconds']==1215
    tree=ast.parse(Path(p.__file__).read_text())
    forbidden={'run','authenticate','census_population','census','reproduce','catalog','project_vcoco_actions','parse_vcoco_role_reference'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name):
            if node.func.value.id=='f':assert node.func.attr not in forbidden
            if node.func.value.id=='c':assert node.func.attr not in forbidden-{'authenticate'}
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign)else[node.target]
            assert not any(isinstance(t,ast.Attribute)for t in targets)
    assert p.f.ENTRY=='run_vcoco_fit_cal_census' and p.prep.ENTRY=='run_vcoco_role_prepare'


def test_complete_authored_projection48_only_private_identity(tmp_path,monkeypatch):
    data,calls,before=identity_fixture(tmp_path,monkeypatch);start=time.monotonic();report=p.run(start,start+30)
    assert report['status']=='pass'and report['stage']=='complete'and report['selection_performed']is True
    assert report['source_and_inputs_rehashed_after']is report['outputs_sealed']is True and len(calls)==2
    out=data/'metadata';assert {q.name for q in out.iterdir()}=={'report.json','cohort.json'}
    assert stat.S_IMODE(data.lstat().st_mode)==0o700 and stat.S_IMODE(out.lstat().st_mode)==0o500
    assert all(stat.S_IMODE(q.lstat().st_mode)==0o400 for q in out.iterdir())
    cohort=p.rt.strict((out/'cohort.json').read_bytes());assert len(cohort['records'])==48
    assert [r['slot']for r in cohort['records']]==list(range(48))
    assert [r['split']for r in cohort['records']]==['FIT']*32+['CAL']*16
    assert cohort['source_authenticated']is False and report['current_helpers_live_verified']is True
    assert all(set(r)=={'slot','split','official_split','image_id','photo_id','rank_sha256'}for r in cohort['records'])
    assert report['input_proof']==before and report['cohort_identity']==p.rt.identity(out/'cohort.json')
    assert all(report[k]is False for k in ('network_used','GPU_used','models_loaded','FIT_performed','CAL_evaluated',
        'reference_geometry_consulted','pilot_reference_values_read','TEST_role_values_read','adopted'))


@pytest.mark.parametrize('fault',['projection','posthash','parent','expiry','preflight','existing'])
def test_failures_preserve_originals_no_partial_rescue(tmp_path,monkeypatch,fault):
    data,calls,_=identity_fixture(tmp_path,monkeypatch);old=p.f.OUTPUT
    original={q.name:(q.read_bytes(),p.c.state(q))for q in old.iterdir()}
    if fault=='projection':monkeypatch.setattr(p.selection,'freeze_fit_cal_cohort',lambda *a,**k:(_ for _ in ()).throw(ValueError('closed')))
    elif fault=='posthash':
        auth=p.authenticate
        def changed(*a):
            result=auth(*a)
            if len(calls)>1:result[1]['live_states']['old']=2
            return result
        monkeypatch.setattr(p,'authenticate',changed)
    elif fault=='parent':
        auth=p.authenticate
        def changed(*a):
            result=auth(*a)
            if len(calls)>1:data.chmod(0o755)
            return result
        monkeypatch.setattr(p,'authenticate',changed)
    elif fault=='expiry':
        def limited(deadline):
            if len(calls)>1:raise ValueError('expired')
        monkeypatch.setattr(p.c,'check',limited)
    elif fault=='preflight':monkeypatch.setattr(p,'authenticate',lambda *a:(_ for _ in ()).throw(ValueError('preflight')))
    elif fault=='existing':data.mkdir(mode=0o700)
    start=time.monotonic()
    if fault in ('preflight','existing','parent'):
        with pytest.raises(ValueError):p.run(start,start+30)
        if fault in ('preflight','existing'):assert not(data/'metadata').exists()
        else:assert not(data/'metadata/report.json').exists()
    else:
        r=p.run(start,start+30);assert r['status']=='fail'and p.rt.strict((data/'metadata/report.json').read_bytes())['status']=='fail'
        assert r['FIT_performed']is r['CAL_evaluated']is False
    assert {q.name:(q.read_bytes(),p.c.state(q))for q in old.iterdir()}==original


@pytest.mark.parametrize('fault',['foreign_leaf','parent','expiry','inode'])
def test_same_fd_publication_and_late_failure(tmp_path,monkeypatch,fault):
    data=tmp_path/'new';data.mkdir(mode=0o700);out=data/'metadata';out.mkdir(mode=0o700)
    monkeypatch.setattr(p,'DATA',data);root_stat(monkeypatch,data)
    class Usage:ru_maxrss=1024
    monkeypatch.setattr(p.resource,'getrusage',lambda *_:Usage());parent=p.prep.namespace_identity(data);owned=out.lstat();expected={}
    p.prep.raw_write(out/'cohort.json',b'{"identities_only":true}\n',expected)
    report=dict(status='pass',decision='ready',outputs_sealed=False)
    if fault=='foreign_leaf':save(out/'foreign',b'unknown')
    elif fault=='parent':data.chmod(0o755)
    elif fault=='expiry':monkeypatch.setattr(p.c,'check',lambda *_:(_ for _ in ()).throw(ValueError('expired')))
    elif fault=='inode':
        sync=p.prep.acq.sync_directory
        def swapped(path):
            if path==out and (out/'report.json').exists():
                out.chmod(0o700)
                (out/'report.json').rename(out/'moved.json');save(out/'report.json',b'foreign')
            else:sync(path)
        monkeypatch.setattr(p.prep.acq,'sync_directory',swapped)
    if fault in ('foreign_leaf','parent','inode'):
        with pytest.raises(ValueError):p.publish(out,report,time.monotonic()+30,time.monotonic(),owned,expected,parent)
    else:
        p.publish(out,report,time.monotonic()+30,time.monotonic(),owned,expected,parent)
        assert report['status']=='fail'and p.rt.strict((out/'report.json').read_bytes())['status']=='fail'
    if fault=='inode':assert (out/'report.json').read_bytes()==b'foreign'


def test_alarm_set_before_inclusive_preflight_and_restored(monkeypatch):
    events=[]
    monkeypatch.setattr(p.signal,'signal',lambda s,h:events.append(('signal',s))or 'old')
    monkeypatch.setattr(p.signal,'setitimer',lambda *a:events.append(('timer',a[1])))
    monkeypatch.setattr(p.resource,'setrlimit',lambda *a:None)
    monkeypatch.setattr(p.os,'sched_setaffinity',lambda *a:None,raising=False)
    monkeypatch.setattr(p,'run',lambda *a:events.append(('run',))or {'status':'pass'})
    assert p.main()=={'status':'pass'} and events.index(('timer',1200))<events.index(('run',))
    assert ('timer',0)in events


def auth_fixture(tmp_path,monkeypatch):
    code=tmp_path/'current/code';code.mkdir(parents=True)
    save(code.parent/'revision',('d'*40+'\n').encode());save(code.parent/'source-sha256',('f'*64+'\n').encode())
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(p,'ROOT',root)
    # Only imported-origin locations are projected; original validators remain
    # independently covered in caller/census/prepare tests, never run old jobs.
    origins=((p.f,'infra/vcoco_fit_cal_census.py'),(p.selection,'infra/vcoco_fit_cal_selection.py'),
        (p.prep,'infra/vcoco_role_prepare.py'),(p.c,'infra/vcoco_role_census.py'),
        (p.rt,'infra/mediapipe_cpu_runtime_verify.py'),(p.c.js,'infra/metadata_json_stream.py'),
        (p.c.projection,'infra/vcoco_role_stream.py'),(p.c.identities,'infra/openimages_joint_pair_acquire.py'),
        (p.c.coco,'infra/coco_proposal_prepare.py'),(p.f.population,'infra/vcoco_population_census.py'))
    for module,name in origins:monkeypatch.setattr(module,'__file__',str(code/name))
    parser_module=__import__(p.c.parse_vcoco_role_reference.__module__,fromlist=['x'])
    monkeypatch.setattr(parser_module,'__file__',str(code/'src/world_reward/vcoco_role_reference.py'))
    current=dict(binding=dict(helpers=deepcopy(p.FROZEN)),stat_identity=dict(bytes=3,sha256='a'*64))
    old=dict(binding={'old_current':True},stat_identity={'old_current_stat':True})
    prepare=dict(binding={'old_prepare':True},stat_identity={'prepare_state':True})
    cdata=tmp_path/'census';cdata.mkdir(mode=0o700);monkeypatch.setattr(p.c,'DATA',cdata)
    fdata=tmp_path/'fitcal';fdata.mkdir(mode=0o700);monkeypatch.setattr(p.f,'OUTPUT',fdata)
    root_stat(monkeypatch,tmp_path)
    ccode=root/'jobs'/('c'*40)/p.c.ENTRY/'code';ccode.mkdir(parents=True)
    config_pin=save(ccode/p.c.CONFIG,{'original_cfg':True})
    original_census=dict(current={'qualified_source':True},historical={'3':True},inputs={'41':True})
    prior=dict(status='pass',stage='complete',capacity_gate_passed=True,source_binding=original_census,
        configuration_identity=config_pin,outputs_sealed=True,source_and_inputs_rehashed_after=True,historical_slots=432,eligibility_inventory_rows=561)
    priorpin=save(cdata/'report.json',prior);spec=dict(producer_revision='c'*40,
        configuration=dict(path=str(ccode/p.c.CONFIG),pin=config_pin),report=dict(path=str(cdata/'report.json'),pin=priorpin))
    calls=[]
    def original(_code,rev,entry,helpers,spec):
        calls.append((rev,entry));path=root/'jobs'/rev/entry/'code';path.mkdir(parents=True,exist_ok=True)
        save(path.parent/'revision',(rev+'\n').encode());save(path.parent/'source-sha256',('f'*64+'\n').encode())
        return path,old if entry==p.f.ENTRY else prepare if entry==p.prep.ENTRY else {'census':True}
    monkeypatch.setattr(p.f,'original',original);monkeypatch.setattr(p.c,'source',lambda *_:deepcopy(current))
    monkeypatch.setattr(p.prep,'configuration',lambda *_:dict(census=spec))
    histories=['one','two'];inputs={}
    for n in (*histories,'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired'):
        path=tmp_path/(n+'.json');save(path,{'metadata_identity_only':n});inputs[n]=dict(path=str(path))
    cfg=dict(sources={},inputs=inputs,history_names=histories)
    monkeypatch.setattr(p.c,'configuration',lambda *_:cfg)
    monkeypatch.setattr(p.c,'authenticate',lambda *_:deepcopy(original_census))
    excluded=dict(photos={str(i)for i in range(1,449)},authors=set(),md5=set())
    monkeypatch.setattr(p.c,'exclusions',lambda *a:deepcopy(excluded))
    pilot={'references':'hash_only'};captured=[]
    def inspect_pilot(ex,proof,checkpoint):captured.append(deepcopy(proof));return ex,deepcopy(pilot)
    monkeypatch.setattr(p.f,'pilot',inspect_pilot)
    prepare_proof=dict(source_binding=prepare['binding'],census_binding=original_census,
        source_stat_identity=prepare['stat_identity'],census_report_identity=priorpin,
        census_report_state=p.c.state(cdata/'report.json'),census_directory_state=p.c.state(cdata))
    # The census directory mode becomes final before its original proof is frozen.
    cdata.chmod(0o500);prepare_proof['census_directory_state']=p.c.state(cdata)
    expected=dict(current=old,original_prepare=prepare,original_census=original_census,prepare_proof=prepare_proof,pilot=pilot)
    rp=save(fdata/'report.json',{'input_proof':expected});ip=save(fdata/'inventory.json',[{'identity_only':True}]);fdata.chmod(0o500)
    monkeypatch.setattr(p.selection,'REPORT_PIN',rp);monkeypatch.setattr(p.selection,'INVENTORY_PIN',ip)
    return code,expected,current,calls,captured,fdata


def test_authentication_reconstructs_original_proof_current_helper_origins(tmp_path,monkeypatch):
    code,expected,current,calls,captured,_=auth_fixture(tmp_path,monkeypatch)
    excluded,proof=p.authenticate(code,'d'*40,lambda:None)
    assert len(excluded['photos'])==448 and proof['original_input_proof']==expected
    assert proof['current_source']==current and captured==[expected['prepare_proof']]
    assert calls==[(p.OLD['revision'],p.f.ENTRY),(p.f.PREPARE['revision'],p.prep.ENTRY),('c'*40,p.c.ENTRY)]
    assert p.f.ENTRY=='run_vcoco_fit_cal_census'and p.prep.ENTRY=='run_vcoco_role_prepare'


@pytest.mark.parametrize('fault',['current_pin','origin','oldproof','inventory_pin','namespace','prior_status'])
def test_authentication_actual_shape_mismatches_closed(tmp_path,monkeypatch,fault):
    code,expected,current,_,_,fdata=auth_fixture(tmp_path,monkeypatch)
    if fault=='current_pin':current['binding']['helpers']['infra/vcoco_fit_cal_census.py']={'bytes':1,'sha256':'z'*64}
    elif fault=='origin':monkeypatch.setattr(p.selection,'__file__',str(tmp_path/'foreign.py'))
    elif fault=='oldproof':
        tampered=deepcopy(expected);tampered['current']['old_marker']='tampered';(fdata/'report.json').chmod(0o600);save(fdata/'report.json',{'input_proof':tampered})
        monkeypatch.setattr(p.selection,'REPORT_PIN',p.rt.identity(fdata/'report.json'))
    elif fault=='inventory_pin':monkeypatch.setattr(p.selection,'INVENTORY_PIN',dict(bytes=1,sha256='0'*64))
    elif fault=='namespace':fdata.chmod(0o700)
    elif fault=='prior_status':
        path=p.c.DATA/'report.json';value=p.rt.strict(path.read_bytes());value['status']='fail';path.chmod(0o600);save(path,value)
    with pytest.raises(ValueError):p.authenticate(code,'d'*40,lambda:None)

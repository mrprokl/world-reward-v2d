"""Manufactured lineage, publisher metadata and JPEG headers, no real corpus."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import pytest
import vcoco_fit_cal_acquisition_inputs as p
import test_vcoco_fit_cal_acquire as generated
import test_vcoco_fit_cal_freeze as helper


def fixture(tmp_path,monkeypatch):
    data=tmp_path/'private';data.mkdir(mode=0o700);public=data/'inputs';private=data/'acquisition'
    public.mkdir(mode=0o700);private.mkdir(mode=0o700);monkeypatch.setattr(p,'DATA',data)
    helper.root_stat(monkeypatch,data)
    code=tmp_path/'current/code';code.mkdir(parents=True);old=tmp_path/'old/code';old.mkdir(parents=True)
    for d,prefix in((code,b'current'),(old,b'old')):
        helper.save(d.parent/'revision',prefix+b' revision');helper.save(d.parent/'source-sha256',prefix+b' archive')
    monkeypatch.setattr(p,'__file__',str(code/'infra/vcoco_fit_cal_acquisition_inputs.py'))
    monkeypatch.setattr(p.acquisition,'__file__',str(code/'infra/vcoco_fit_cal_acquire.py'))
    selected=generated.rows();cohort={'records':[{k:r[k]for k in('slot','split','official_split','image_id','photo_id','rank_sha256')}for r in selected]}
    current=dict(current_source=dict(binding=dict(helpers=deepcopy(p.HELPER_PINS))),freeze_source={'old_freeze':True},
        original_freeze_input_proof={'all_originals':'qualified'},live_states={str(q):p.c.state(q)for q in
            (code.parent,code.parent/'revision',code.parent/'source-sha256')},private_parent=p.prep.namespace_identity(data),
        freeze_report_identity=p.context.REPORT_PIN,cohort_identity=p.context.COHORT_PIN)
    current['live_states']['original_frozen_marker']=['immutable']
    old_source=dict(binding={'old_acquisition':True},stat_identity={'old_stat':True})
    original=deepcopy(current);original['current_source']=old_source
    for q in(code.parent,code.parent/'revision',code.parent/'source-sha256'):del original['live_states'][str(q)]
    for q in(old.parent,old.parent/'revision',old.parent/'source-sha256'):original['live_states'][str(q)]=p.c.state(q)
    calls=[]
    def context(*args):calls.append(args);return {'metadata':'only'},dict(md5=set()),deepcopy(cohort),deepcopy(current)
    monkeypatch.setattr(p.context,'authenticate',context)
    def origin(*args):assert args[1]==p.ORIGINAL['revision']and args[2]==p.acquisition.ENTRY;return old,deepcopy(old_source)
    monkeypatch.setattr(p.context.f,'original',origin)
    ledger={};urls={r['original_url']:r['slot']for r in selected}
    result=p.acquisition.acquire(selected,dict(md5=set()),public,time.monotonic()+30,ledger,
        request=lambda url,*_:generated.jpeg(urls[url]))
    report=dict(**p.acquisition.RECIPE,**result,producer_revision=p.ORIGINAL['revision'],input_proof=p.context.normalized(original),
        status='pass',stage='complete',freeze_report_identity=p.context.REPORT_PIN,cohort_identity=p.context.COHORT_PIN,
        source_and_inputs_rehashed_after=True,outputs_sealed=True,public_outputs_sealed=True,network_used=True,
        elapsed_seconds=2.0,peak_rss_bytes=1024,publisher_metadata=selected,artifact_identities={},public_artifact_identities=ledger)
    for k in('annotation_values_consulted','role_values_consulted','pilot_reference_values_read','RGB_decoded','GPU_used',
        'models_loaded','FIT_performed','CAL_evaluated','selection_performed','author_disjointness_verified',
        'training_overlap_verified','challenge_overlap_verified','adopted'):report[k]=False
    monkeypatch.setattr(p,'PUBLIC_PIN',p.rt.identity(public/'manifest.json'));rp=helper.save(private/'report.json',report)
    monkeypatch.setattr(p,'REPORT_PIN',rp);public.chmod(0o500);private.chmod(0o500)
    return code,old,data,current,cohort,report,calls


def update_report(data,report,monkeypatch):
    path=data/'acquisition/report.json';path.chmod(0o600);monkeypatch.setattr(p,'REPORT_PIN',helper.save(path,report))


def test_fixed_source_pins_and_no_old_execution_or_profile_mutation():
    assert p.ORIGINAL['files']==316 and p.ORIGINAL['entries']==321
    assert p.REPORT_PIN['bytes']==133045 and p.PUBLIC_PIN['bytes']==9754
    root=Path(p.__file__).resolve().parents[1]
    assert all(p.rt.identity(root/n,readonly=False)==pin for n,pin in p.HELPER_PINS.items())
    tree=ast.parse(Path(p.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in('run','reproduce','census','acquire','freeze_fit_cal_cohort','project_vcoco_actions','parse_vcoco_role_reference')
        if isinstance(n,ast.Assign):assert not any(isinstance(t,ast.Attribute)for t in n.targets)


def test_original_proof_projection_and_only_public_return(tmp_path,monkeypatch):
    code,old,data,current,cohort,report,calls=fixture(tmp_path,monkeypatch)
    original=deepcopy(current);records,proof=p.authenticate_actual_acquisition(code,'d'*40,'new_entry',(),lambda:None)
    assert current==original and calls[0][:4]==(code,'d'*40,'new_entry',())
    assert len(records)==48 and all(set(r)==p.prep.PUBLIC_KEYS|{'path'}for r in records)
    assert proof['source_authenticated']is True and proof['current_source']==current['current_source']
    assert proof['original_acquisition_source']!=proof['current_source']
    raw=p.c.encode((records,proof)).decode()
    for key in('"official_split"','"photo_id"','"publisher_metadata"','"rank_sha256"','"original_md5"','image_id": 1000'):assert key not in raw
    assert not any(k in proof for k in('input_proof','cohort','publisher_metadata','private_records'))
    assert proof['acquisition_report_identity']==p.REPORT_PIN and proof['public_manifest_identity']==p.PUBLIC_PIN


def test_repeat_check_preserves_original_source_metadata_and_bytes(tmp_path,monkeypatch):
    code,old,data,current,_,_,_=fixture(tmp_path,monkeypatch)
    before={str(q):(q.read_bytes(),p.c.state(q))for q in tmp_path.rglob('*')if q.is_file()}
    a=p.authenticate_actual_acquisition(code,'d'*40,'new_entry',(),lambda:None)
    b=p.authenticate_actual_acquisition(code,'d'*40,'new_entry',(),lambda:None)
    assert a==b and {str(q):(q.read_bytes(),p.c.state(q))for q in tmp_path.rglob('*')if q.is_file()}==before


@pytest.mark.parametrize('fault',['new_source_in_oldproof','marker','count','status','sourcepost','flag','missing',
    'mapping','publisher','grant','md5','header','manifest','metadata_mode','rootmode','helper','origin','jpeg_writeable','symlink'])
def test_actual_lineage_and_all48_byte_contracts_fail_closed(tmp_path,monkeypatch,fault):
    code,old,data,current,cohort,report,_=fixture(tmp_path,monkeypatch)
    if fault=='new_source_in_oldproof':report['input_proof']['current_source']=deepcopy(current['current_source'])
    elif fault=='marker':report['input_proof']['live_states'][str(old.parent/'revision')][0]+=1
    elif fault=='count':report['counts']['acquired']=47
    elif fault=='status':report['status']='fail'
    elif fault=='sourcepost':report['source_and_inputs_rehashed_after']=False
    elif fault=='flag':report['role_values_consulted']=True
    elif fault=='missing':report['records'][0]['status']='unavailable'
    elif fault=='mapping':report['public_mappings'][0]['slot']=1
    elif fault=='publisher':report['publisher_metadata'][0]['image_id']=999
    elif fault=='grant':report['publisher_metadata'][0]['license_id']=1
    elif fault=='md5':report['records'][0]['original_md5']='Z'*24
    elif fault=='header':report['records'][0]['jpeg_header']['width']=4
    elif fault=='manifest':
        path=data/'inputs/manifest.json';path.chmod(0o600);value=p.rt.strict(path.read_bytes());value['images'][0]['width']=4
        public=helper.save(path,value);monkeypatch.setattr(p,'PUBLIC_PIN',public)
        report['public_inputs_identity']=public;report['public_artifact_identities']['manifest.json']=public
    elif fault=='metadata_mode':(data/'acquisition').chmod(0o700)
    elif fault=='rootmode':data.chmod(0o755)
    elif fault=='helper':current['current_source']['binding']['helpers']['infra/vcoco_fit_cal_context.py']['bytes']=1
    elif fault=='origin':monkeypatch.setattr(p.acquisition,'__file__',str(tmp_path/'foreign.py'))
    elif fault=='jpeg_writeable':(data/'inputs/image_000000.jpg').chmod(0o600)
    else:
        public=data/'inputs';public.chmod(0o700);(public/'image_000000.jpg').rename(public/'original.jpg')
        (public/'image_000000.jpg').symlink_to(public/'original.jpg');public.chmod(0o500)
    update_report(data,report,monkeypatch)
    with pytest.raises(ValueError):p.authenticate_actual_acquisition(code,'d'*40,'new_entry',(),lambda:None)

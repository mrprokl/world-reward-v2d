"""Explicit-context source/ref-hash lineage mechanics, no real inputs."""
import ast
import hashlib
from copy import deepcopy
from pathlib import Path
import pytest
import vcoco_fit_cal_context as p
import test_vcoco_fit_cal_freeze as helper


def fixture(tmp_path,monkeypatch):
    code,expected,current,calls,_,_=helper.auth_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'ROOT',helper.p.ROOT)
    monkeypatch.setattr(p.freeze,'__file__',str(code/'infra/vcoco_fit_cal_freeze.py'))
    monkeypatch.setattr(p,'__file__',str(code/'infra/vcoco_fit_cal_context.py'))
    current['binding']['helpers']['infra/vcoco_fit_cal_freeze.py']=deepcopy(p.FREEZE_HELPER)
    original=p.f.original;freeze_source=dict(binding={'oldfreeze':True},stat_identity={'freeze_stat':True})
    def sources(code,rev,entry,helpers,spec):
        if entry==p.freeze.ENTRY:
            path=p.ROOT/'jobs'/rev/entry/'code';path.mkdir(parents=True)
            helper.save(path.parent/'revision',(rev+'\n').encode());helper.save(path.parent/'source-sha256',('e'*64+'\n').encode())
            return path,freeze_source
        return original(code,rev,entry,helpers,spec)
    monkeypatch.setattr(p.f,'original',sources)
    data=tmp_path/'private';data.mkdir(mode=0o700);meta=data/'metadata';meta.mkdir(mode=0o700)
    monkeypatch.setattr(p.freeze,'DATA',data)
    # Snapshot reconstructed old auth output before writing its fixed receipt.
    def frozen_states():
        paths=[p.ROOT/'jobs'/rev/entry/'code'for rev,entry in
            ((p.FREEZE['revision'],p.freeze.ENTRY),(p.OLD['revision'],p.f.ENTRY),
             (p.f.PREPARE['revision'],p.prep.ENTRY),('c'*40,p.c.ENTRY))]
        states={str(q):p.c.state(q)for d in paths for q in(d.parent,d.parent/'revision',d.parent/'source-sha256')}
        states.update({str(p.f.OUTPUT):p.c.state(p.f.OUTPUT),str(p.f.OUTPUT/'report.json'):p.c.state(p.f.OUTPUT/'report.json'),
            str(p.f.OUTPUT/'inventory.json'):p.c.state(p.f.OUTPUT/'inventory.json')})
        return states
    # Create original source markers once without executing their algorithms.
    for rev,entry in((p.FREEZE['revision'],p.freeze.ENTRY),(p.OLD['revision'],p.f.ENTRY),
                      (p.f.PREPARE['revision'],p.prep.ENTRY),('c'*40,p.c.ENTRY)):
        sources(code,rev,entry,(),{})
    # Original source fake must be read-only/repeatable, not rewrite markers after snapshot.
    def stable(code,rev,entry,helpers,spec):
        path=p.ROOT/'jobs'/rev/entry/'code'
        return path,freeze_source if entry==p.freeze.ENTRY else expected['current']if entry==p.f.ENTRY else expected['original_prepare']if entry==p.prep.ENTRY else {'census':True}
    monkeypatch.setattr(p.f,'original',stable)
    freeze_proof=dict(current_source=freeze_source,original_input_proof=expected,live_states=frozen_states(),
                     census_report_identity=p.selection.REPORT_PIN,census_inventory_identity=p.selection.INVENTORY_PIN)
    rows=[dict(slot=i,split='FIT'if i<32 else'CAL',official_split='train'if i<32 else'val',image_id=1000+i,
               photo_id=str(100000+i),rank_sha256=hashlib.sha256((p.selection.NAMESPACE+f'{1000+i:012d}').encode()).hexdigest())for i in range(48)]
    cohort=dict(schema='world_reward.vcoco_fit_cal_identity_cohort.v1',namespace=p.selection.NAMESPACE,
        census_producer_revision=p.OLD['revision'],census_report_identity=p.selection.REPORT_PIN,census_inventory_identity=p.selection.INVENTORY_PIN,
        selected_slots=48,split_counts=dict(FIT=32,CAL=16),all_48_acquired_required=True,retry_count=0,replacement_count=0,
        reference_values_exposed=False,predictor_input=False,source_authenticated=False,input_proof_identity=p.c.pin(p.c.encode(p.normalized(expected))),
        excluded_photo_ids_identity=p.c.pin(p.c.encode([str(i)for i in sorted(range(1,449),key=lambda i:str(i))])),records=rows)
    cp=helper.save(meta/'cohort.json',cohort);monkeypatch.setattr(p,'COHORT_PIN',cp)
    report=dict(input_proof=p.normalized(freeze_proof),status='pass',producer_revision=p.FREEZE['revision'],stage='complete',
        source_and_inputs_rehashed_after=True,outputs_sealed=True,decision='FROZEN48_IDENTITIES_PENDING_SEPARATE_ACQUISITION',cohort_identity=cp)
    rp=helper.save(meta/'report.json',report);monkeypatch.setattr(p,'REPORT_PIN',rp);meta.chmod(0o500)
    return code,report,cohort,data


def test_context_no_old_execution_or_profile_mutation():
    tree=ast.parse(Path(p.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name):
            if n.func.value.id in('freeze','f','c'):assert n.func.attr not in('run','census','reproduce','census_population','freeze_fit_cal_cohort')
            if n.func.value.id in('freeze','f'):assert n.func.attr!='authenticate'
    assert p.FREEZE['revision']=='8cf55c23755975325fe00efb8fa3229392b38475'
    assert p.REPORT_PIN['bytes']==63579 and p.COHORT_PIN['bytes']==10151


def test_context_reconstructs_old_current_source_not_newcaller(tmp_path,monkeypatch):
    code,_,cohort,_=fixture(tmp_path,monkeypatch)
    cfg,excluded,actual,proof=p.authenticate(code,'d'*40,'new_entry',(),lambda:None)
    assert actual==cohort and len(excluded['photos'])==448
    assert proof['original_freeze_input_proof']['current_source']==proof['freeze_source']
    assert proof['current_source']!=proof['freeze_source']
    assert proof['original_freeze_input_proof']['live_states']==p.normalized(proof['original_freeze_input_proof'])['live_states']
    assert str(p.freeze.DATA/'metadata')not in proof['original_freeze_input_proof']['live_states']
    assert str(p.freeze.DATA/'metadata')in proof['live_states']
    assert p.freeze.ENTRY=='run_vcoco_fit_cal_freeze'and p.f.ENTRY=='run_vcoco_fit_cal_census'


@pytest.mark.parametrize('fault',['proof','status','source','slot','rank','photo','metadata_mode','parent_mode','origin','helper'])
def test_current_and_original_authentication_fail_closed(tmp_path,monkeypatch,fault):
    code,report,cohort,data=fixture(tmp_path,monkeypatch)
    for path in(data/'metadata').iterdir():path.chmod(0o600)
    if fault in('proof','status','source'):
        if fault=='proof':report['input_proof']['current_source']={'new_caller':True}
        elif fault=='status':report['status']='fail'
        else:report['producer_revision']='e'*40
        monkeypatch.setattr(p,'REPORT_PIN',helper.save(data/'metadata/report.json',report))
    elif fault in('slot','rank','photo'):
        cohort['records'][0][{'slot':'slot','rank':'rank_sha256','photo':'photo_id'}[fault]]={'slot':True,'rank':'f'*64,'photo':'1'}[fault]
        pin=helper.save(data/'metadata/cohort.json',cohort);monkeypatch.setattr(p,'COHORT_PIN',pin)
        report['cohort_identity']=pin;monkeypatch.setattr(p,'REPORT_PIN',helper.save(data/'metadata/report.json',report))
    elif fault=='metadata_mode':(data/'metadata').chmod(0o700)
    elif fault=='parent_mode':data.chmod(0o755)
    elif fault=='origin':monkeypatch.setattr(p.freeze,'__file__',str(tmp_path/'foreign.py'))
    else:
        old=p.c.source
        def corrupt(*args):
            value=old(*args);value['binding']['helpers']['infra/vcoco_fit_cal_freeze.py']['sha256']='e'*64;return value
        monkeypatch.setattr(p.c,'source',corrupt)
    with pytest.raises(ValueError):p.authenticate(code,'d'*40,'new_entry',(),lambda:None)


def test_repeat_context_auth_does_not_rewrite_or_mutate_old_proof(tmp_path,monkeypatch):
    code,_,_,data=fixture(tmp_path,monkeypatch)
    before={str(q):(q.read_bytes(),p.c.state(q))for q in tmp_path.rglob('*')if q.is_file()}
    a=p.authenticate(code,'d'*40,'new_entry',(),lambda:None)
    (data/'inputs').mkdir(mode=0o700)
    b=p.authenticate(code,'d'*40,'new_entry',(),lambda:None)
    assert a[:3]==b[:3]and p.normalized(a[3])==p.normalized(b[3])
    assert {str(q):(q.read_bytes(),p.c.state(q))for q in tmp_path.rglob('*')if q.is_file()}==before

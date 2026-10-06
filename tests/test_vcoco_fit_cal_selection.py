"""Procedural metadata only; no real image IDs, labels, files, RGB or Azure."""
from copy import deepcopy
import ast
import hashlib
from pathlib import Path

import pytest
import vcoco_fit_cal_selection as s


def fixture(monkeypatch):
    excluded = {str(i) for i in range(1, 449)}
    rows = [dict(split=name, image_id=start+i, photo_id=str(100000+start+i))
            for name, start, count in (('train', 1000, 195), ('val', 10000, 191)) for i in range(count)]
    proof = dict(current=dict(binding=dict(producer_revision=s.CENSUS_REVISION, entries=313,
                 closure_sha256=s.CENSUS_CLOSURE, helpers={'tiny': {'bytes': 1, 'sha256': 'a'*64}})),
                 original_prepare=dict(markers={i:dict(slot=i) for i in range(16)}),
                 original_census=dict(inputs={'opaque':dict(bytes=7, sha256='b'*64)}),
                 prepare_proof=dict(source='old'), pilot=dict(pins='hash_only'))
    report = dict(schema='world_reward.vcoco_fit_cal_census.v1', producer_revision=s.CENSUS_REVISION,
                  input_proof=s._strict(s._encode(proof)), status='pass', stage='complete',
                  decision='CAPACITY_METADATA_ONLY_NO_SELECTION', capacity_gate_passed=True,
                  outputs_sealed=True, source_and_inputs_rehashed_after=True, historical_photos=448,
                  minimum_distinct_photos=dict(train=32, val=16), eligibility_inventory_rows=386,
                  fresh_reference_geometry_consulted=True,
                  splits={n:dict(distinct_eligible_photos=c, eligible_images=c) for n,c in s.COUNTS.items()})
    for name in ('historical_reference_values_read', 'pilot_reference_values_read', 'TEST_role_values_read',
                 'RGB_read', 'network_used', 'GPU_used', 'models_loaded', 'FIT_performed', 'selection_performed',
                 'author_disjointness_verified', 'challenge_overlap_verified', 'training_overlap_verified', 'adopted'):
        report[name] = False
    return excluded, rows, proof, report


def repin(monkeypatch, rows, report):
    inventory = s._encode(rows); ip = s._pin(inventory)
    report['eligibility_inventory_identity'] = ip; report['artifact_identities'] = {'inventory.json':ip}
    raw = s._encode(report); monkeypatch.setattr(s, 'REPORT_PIN', s._pin(raw)); monkeypatch.setattr(s, 'INVENTORY_PIN', ip)
    return raw, inventory


def call(monkeypatch, data):
    excluded, rows, proof, report = data
    raw, inventory = repin(monkeypatch, rows, report)
    return s.freeze_fit_cal_cohort(raw, inventory, excluded_photo_ids=excluded, expected_input_proof=proof)


def test_exact_constants_and_no_model_or_semantic_imports():
    assert s.NAMESPACE == 'world_reward.vcoco_fit_cal_v1/'
    assert s.COUNTS == dict(train=195, val=191) and s.SLOTS == (('train','FIT',32),('val','CAL',16))
    assert s.REPORT_PIN == dict(bytes=57524, sha256='703e0e341d3ceb8c44d40f5a90823299719ade1559c36945bc6f8d92120ffee0')
    assert s.INVENTORY_PIN == dict(bytes=24991, sha256='ea7503e6e2e09cc6b4536ce12bfed9ad73698488ebc50b22db0fcfe8276fbd70')
    tree=ast.parse(Path(s.__file__).read_text()); imports={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert imports == {'hashlib','json','re'} and not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('open','exec','eval') for n in ast.walk(tree))


def test_first32_first16_independent_rank_fixed_slots_and_no_mutation(monkeypatch):
    data=fixture(monkeypatch); before=deepcopy(data); cohort=call(monkeypatch,data)
    expected=[]
    for official,study,count in (('train','FIT',32),('val','CAL',16)):
        pop=[r for r in data[1] if r['split']==official]
        ranked=sorted(pop,key=lambda r:(hashlib.sha256(('world_reward.vcoco_fit_cal_v1/'+str(r['image_id']).zfill(12)).encode()).hexdigest(),r['image_id']))
        expected.extend((study,official,r['image_id'],r['photo_id']) for r in ranked[:count])
    assert [(r['split'],r['official_split'],r['image_id'],r['photo_id']) for r in cohort['records']] == expected
    assert [r['slot'] for r in cohort['records']] == list(range(48))
    assert all(set(r)=={'slot','split','official_split','image_id','photo_id','rank_sha256'} for r in cohort['records'])
    assert cohort['all_48_acquired_required'] is True and cohort['selected_slots']==48
    assert cohort['split_counts']==dict(FIT=32,CAL=16) and cohort['retry_count']==cohort['replacement_count']==0
    assert cohort['source_authenticated'] is cohort['predictor_input'] is cohort['reference_values_exposed'] is False
    # Only repin deliberately updates the fixture report's byte ledger.
    assert data[:3]==before[:3]
    assert not {'url','boxes','actions','role','reference_file','available'} & set(cohort)


def test_input_order_is_not_selection_and_json_integer_key_roundtrip(monkeypatch):
    data=fixture(monkeypatch); first=call(monkeypatch,data)
    proof_parsed=s._strict(s._encode(data[2])); raw,inventory=repin(monkeypatch,data[1],data[3])
    assert s.freeze_fit_cal_cohort(raw,inventory,excluded_photo_ids=frozenset(data[0]),expected_input_proof=proof_parsed)==first
    data[1].reverse(); second=call(monkeypatch,data)
    assert second['records']==first['records'] and second['census_inventory_identity']!=first['census_inventory_identity']
    first['records'][0]['slot']=999; first['split_counts']['FIT']=0
    assert call(monkeypatch,data)['records'][0]['slot']==0 and s.SLOTS[0][2]==32


@pytest.mark.parametrize('fault', ['pin_report','pin_inventory','notbytes','proof_key','proof_value','proof_source',
    'proof_count','proof_closure','report_status','report_decision','report_flag','report_gate','report_rows',
    'report_split','wrong_minimum','excluded_count','excluded_type','excluded_photo','row_bool','row_float',
    'row_zero','row_large','row_schema','row_unknownsplit','photo_type','photo_empty','photo_ws','photo_duplicate',
    'id_duplicate','crosssplit_id','historical_photo','subset','wrongpopulation','nonfinite','dupejson'])
def test_conservative_metadata_failures_no_subset_or_rescue(monkeypatch,fault):
    data=fixture(monkeypatch);excluded,rows,proof,report=data
    if fault=='proof_key': proof['foreign']={}
    elif fault=='proof_value': proof['original_prepare']['markers'][0]['slot']=2
    elif fault=='proof_source': report['input_proof']['current']['binding']['producer_revision']='a'*40;proof['current']['binding']['producer_revision']='a'*40
    elif fault=='proof_count': report['input_proof']['current']['binding']['entries']=312;proof['current']['binding']['entries']=312
    elif fault=='proof_closure': report['input_proof']['current']['binding']['closure_sha256']='e'*64;proof['current']['binding']['closure_sha256']='e'*64
    elif fault=='report_status': report['status']='fail'
    elif fault=='report_decision': report['decision']='READY_SMALLER'
    elif fault=='report_flag': report['TEST_role_values_read']=True
    elif fault=='report_gate': report['capacity_gate_passed']=1
    elif fault=='report_rows': report['eligibility_inventory_rows']=385
    elif fault=='report_split': report['splits']['test']=report['splits'].pop('val')
    elif fault=='wrong_minimum': report['minimum_distinct_photos']['val']=12
    elif fault=='excluded_count':excluded.remove('1')
    elif fault=='excluded_type':data=(list(excluded),rows,proof,report)
    elif fault=='excluded_photo':excluded.remove('1');excluded.add(' 1')
    elif fault=='row_bool':rows[0]['image_id']=True
    elif fault=='row_float':rows[0]['image_id']=1000.
    elif fault=='row_zero':rows[0]['image_id']=0
    elif fault=='row_large':rows[0]['image_id']=10**12
    elif fault=='row_schema':rows[0]['annotation']={'GT':'notallowed'}
    elif fault=='row_unknownsplit':rows[0]['split']='test'
    elif fault=='photo_type':rows[0]['photo_id']=100000
    elif fault=='photo_empty':rows[0]['photo_id']=''
    elif fault=='photo_ws':rows[0]['photo_id']='100000 '
    elif fault=='photo_duplicate':rows[1]['photo_id']=rows[0]['photo_id']
    elif fault=='id_duplicate':rows[1]['image_id']=rows[0]['image_id']
    elif fault=='crosssplit_id':rows[195]['image_id']=rows[0]['image_id']
    elif fault=='historical_photo':rows[0]['photo_id']='1'
    elif fault=='subset':rows.pop()
    elif fault=='wrongpopulation':rows[0]['split']='val'
    raw,inventory=repin(monkeypatch,rows,report)
    if fault=='pin_report': raw+=b' '
    elif fault=='pin_inventory':inventory+=b' '
    elif fault=='notbytes':raw=bytearray(raw)
    elif fault=='nonfinite':raw=raw[:-2]+b',"bad":NaN}\n';monkeypatch.setattr(s,'REPORT_PIN',s._pin(raw))
    elif fault=='dupejson':raw=raw[:-2]+b',"status":"pass"}\n';monkeypatch.setattr(s,'REPORT_PIN',s._pin(raw))
    with pytest.raises((ValueError,KeyError,TypeError)):
        s.freeze_fit_cal_cohort(raw,inventory,excluded_photo_ids=data[0],expected_input_proof=proof)


def test_pin_check_precedes_json_decode_and_no_hidden_io(monkeypatch):
    monkeypatch.setattr(s,'_strict',lambda *_:pytest.fail('Unpinned payload decoded'))
    with pytest.raises(ValueError,match='pins'):s.freeze_fit_cal_cohort(b'poison',b'poison',excluded_photo_ids=set(),expected_input_proof={})

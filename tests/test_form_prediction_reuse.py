"""Tiny byte/source replay contracts; no RGB decoding, model, credentials or GPU."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
import form_prediction_reuse as reuse
import form_hoi_external_predict as p

OLD='a'*40;NEW='b'*40


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    oldcode=tmp_path/'jobs'/OLD/p.ENTRY/'code';code=tmp_path/'jobs'/NEW/p.ENTRY/'code'
    source=(REPO/'infra/form_hoi_external_predict.py').read_bytes()
    for root in (oldcode,code):
        for name in (*reuse.SOURCE_FILES,'infra/form_hoi_external_predict.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(source if name=='infra/form_hoi_external_predict.py'else (REPO/name).read_bytes());path.chmod(0o444)
    binding={'producer_revision':OLD,'closure_sha256':'c'*64,'helpers':{}}
    monkeypatch.setattr(p,'source',lambda *args:binding)
    packages={};rows=[];records=[];cfg=p.config(code)
    for i in range(4):
        sid='FORM_DEV_'+str(i);ip={'bytes':123,'sha256':str(i)*64}
        row=dict(sequence_id=sid,input=tmp_path/sid/'inputs/input.json',pin=ip);rows.append(row)
        package=dict(video_pin={'bytes':456,'sha256':'f'*64},full_source_frames=764)
        packages[str(row['input'])]=package
        directory=tmp_path/'results'/('form-hoi-external-predict-'+OLD)/sid/'localize';directory.mkdir(parents=True)
        calls=[dict(frame_index=k,rgb_sha256='d'*64,person_bbox=[1,2,3,4],object_bbox=[5,6,7,8]) for k in cfg['seed_frames']]
        boxes=p.save_json(directory/'boxes.json',dict(rows=calls,conditioning='public_RGB_object_prompt_action_only'))
        report=dict(status='complete',stage='localize',producer_revision=OLD,source_binding=binding,
            sequence_id=sid,input_pin=ip,video_pin=package['video_pin'],dataset='nvidia/form-hoi',
            full_source_frames=764,original_frame_indices=list(range(96)),ground_truth_used=False,
            private_truth_read=False,reference_inputs_mounted=False,hand_labeled_test=False,oracle_modes=[],
            calls=calls,localizer=cfg['localizer'],artifacts={'boxes.json':boxes})
        rp=p.save_json(directory/'report.json',report)
        records.append(dict(sequence_id=sid,status='complete',input=dict(path=str(row['input']),pin=ip),
            stages={'localize':dict(path=str(directory/'report.json'),pin=rp)}))
        if i==0:
            failed=directory.parent/'track';failed.mkdir();(failed/'failed.txt').write_text('notcopied')
    cp=tmp_path/'results'/f'form-hoi-external-cohort-{OLD}-localize.json'
    p.save_json(cp,dict(status='complete',stage='localize',producer_revision=OLD,source_binding=binding,
        ground_truth_used=False,private_truth_read=False,sequence_order=[r['sequence_id']for r in rows],sequences=records))
    monkeypatch.setattr(p,'load_public',lambda path,pin,code:packages[str(path)])
    return code,oldcode,rows,binding


def rewrite(path,mutate):
    value=json.loads(path.read_bytes());mutate(value);path.chmod(0o600)
    path.write_text(json.dumps(value,sort_keys=True)+'\n');path.chmod(0o444)


def test_literal_original_source_inventory_without_executing_code():
    raw=(REPO/'infra/form_hoi_external_predict.py').read_bytes()
    assert reuse.helper_inventory(raw)==p.HELPERS
    assert reuse.function_identity(raw,'localize')==reuse.function_identity(raw,'localize')
    with pytest.raises(ValueError):reuse.helper_inventory(b'HELPERS = ("../bad",)\nCONFIG="config.json"')


def test_only_successful_localize_is_byte_exact_copied_old_producer_retained(tmp_path,monkeypatch):
    code,oldcode,rows,binding=fixture(tmp_path,monkeypatch)
    proof=reuse.reuse(p,code,NEW,OLD,rows)
    assert proof['localization_API_calls']==0 and proof['failed_tracking_artifacts_copied'] is False
    for row in rows:
        old=tmp_path/'results'/('form-hoi-external-predict-'+OLD)/row['sequence_id']/'localize'
        new=tmp_path/'results'/('form-hoi-external-predict-'+NEW)/row['sequence_id']
        assert {x.name for x in new.iterdir()}=={'localize','localization_reuse.json'}
        assert {x.name for x in (new/'localize').iterdir()}=={'boxes.json','report.json'}
        for name in ('boxes.json','report.json'):
            assert (old/name).read_bytes()==(new/'localize'/name).read_bytes()
            assert p.artifact(old/name)==p.artifact(new/'localize'/name)
        report=json.loads((new/'localize/report.json').read_bytes())
        assert report['producer_revision']==OLD and report['source_binding']==binding
        lineage=reuse.localization_lineage(p,new,NEW)
        assert lineage['producer_revision']==OLD and 'reuse_proof'in lineage
    assert (tmp_path/'results'/('form-hoi-external-predict-'+OLD)/rows[0]['sequence_id']/'track/failed.txt').is_file()


@pytest.mark.parametrize('fault',['config','localizer_source','partial','input','GT','payload','condition','frames','destination','algorithm'])
def test_reuse_prevalidates_all_four_and_never_rerolls_or_changes_conditioning(tmp_path,monkeypatch,fault):
    code,oldcode,rows,binding=fixture(tmp_path,monkeypatch)
    report=tmp_path/'results'/('form-hoi-external-predict-'+OLD)/rows[-1]['sequence_id']/'localize/report.json'
    cohort=tmp_path/'results'/f'form-hoi-external-cohort-{OLD}-localize.json'
    if fault in ('config','localizer_source','algorithm'):
        name={'config':reuse.SOURCE_FILES[0],'localizer_source':reuse.SOURCE_FILES[-1],
            'algorithm':'infra/form_hoi_external_predict.py'}[fault]
        path=code/name;path.chmod(0o600)
        content=path.read_bytes()
        path.write_bytes(content.replace(b'tasks = []',b'tasks = [None]',1) if fault=='algorithm'else content+b'\n')
        path.chmod(0o444)
    elif fault=='partial':rewrite(cohort,lambda r:r['sequences'][-1].update(status='fail'))
    elif fault=='input':rows[-1]['pin']['sha256']='e'*64
    elif fault=='GT':rewrite(report,lambda r:r.update(ground_truth_used=True))
    elif fault=='payload':
        path=report.parent/'boxes.json';path.chmod(0o600);path.write_bytes(b'{}');path.chmod(0o444)
    elif fault=='condition':rewrite(report,lambda r:r['localizer'].update(model='anothermodel'))
    elif fault=='frames':rewrite(report,lambda r:r.update(original_frame_indices=[0]))
    else:
        out=tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[-1]['sequence_id'];out.mkdir(parents=True)
        (out/'untouched').write_bytes(b'baseline')
    with pytest.raises((ValueError,FileNotFoundError)):reuse.reuse(p,code,NEW,OLD,rows)
    root=tmp_path/'results'/('form-hoi-external-predict-'+NEW)
    assert not root.exists() or {x.name for x in root.iterdir()}=={rows[-1]['sequence_id']}


def test_foreign_old_report_requires_valid_explicit_reuse_proof(tmp_path,monkeypatch):
    code,_,rows,_=fixture(tmp_path,monkeypatch);reuse.reuse(p,code,NEW,OLD,rows)
    base=tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[0]['sequence_id']
    rewrite(base/'localization_reuse.json',lambda r:r.update(new_prediction_producer='c'*40))
    with pytest.raises(ValueError):reuse.localization_lineage(p,base,NEW)


def test_track_PYTHONPATH_matches_actual_qualified_native_runtime():
    text=(REPO/'infra/form_hoi_external_predict.py').read_text()
    assert "('/opt/sam3:' if stage=='track' else '')"in text
    assert '--reuse-localizations-from'in text and 'localization_lineage'in text

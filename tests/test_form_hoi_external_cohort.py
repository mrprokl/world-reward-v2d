"""Tiny dispatcher/ownership contracts; no data decoding, API, models or GPU."""
import json
from pathlib import Path
import sys
import threading
import time

import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
import form_hoi_external_cohort as cohort
import form_hoi_external_predict as predictor


def prepared(tmp_path,monkeypatch):
    revision='a'*40
    sequences=['FORM_DEV_1','FORM_DEV_2','FORM_DEV_3','FORM_DEV_4']
    code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    (code/'configs/form_hoi_insight_v1.json').write_text(json.dumps(dict(cohort=[
        dict(sequence_id=s,split='development') for s in sequences]+[
        dict(sequence_id='RESERVED_NOT_READ',split='reserved_unopened')])))
    (code/'configs/form_hoi_external_dev_v1.json').write_bytes(
        (REPO/'configs/form_hoi_external_dev_v1.json').read_bytes())
    monkeypatch.setattr(cohort,'DATA',tmp_path/'public_only')
    root=cohort.DATA/revision
    for sid in sequences:
        d=root/sid/'inputs';d.mkdir(parents=True)
        video=d/'rgb.mp4';video.write_bytes(b'tiny_not_video');video.chmod(0o444)
        spec=dict(schema='world_reward.external_rgb_input.v1',sequence_id=sid,dataset='nvidia/form-hoi',
            dataset_revision='c63db107e84c7f74bb4929ef643b67b5c8bcc00e',split='development',
            video=str(video),video_pin=predictor.artifact(video),total=96,full_source_frames=764,
            camera='front_stereo_camera_left',height=1152,width=1536,fps=30,original_frame_indices=list(range(96)),
            object_prompt='a container',action='lift container',inference_ready=True,reference_inputs_present=False)
        predictor.save_json(d/'input.json',spec)
    predictor.save_json(root/'public-transfer.json',dict(status='pass',dev_revision=revision,sequences=sequences,
        manifest_identity=dict(bytes=123,sha256='b'*64),private_references_transferred=False,heavy_data_local=False))
    return code,revision,root,sequences


def test_discovery_freezes_only_exact_four_DEV_public_packages(tmp_path,monkeypatch):
    code,rev,root,seqs=prepared(tmp_path,monkeypatch)
    rows,receipt=cohort.discover(predictor,code,rev)
    assert [r['sequence_id'] for r in rows]==seqs
    assert receipt['private_references_transferred'] is False
    assert all(r['pin']==predictor.artifact(r['input']) for r in rows)
    assert all(r['input'].parent.name=='inputs' for r in rows)
    assert not (root/'RESERVED_NOT_READ').exists()


@pytest.mark.parametrize('fault',['GT','notpass','reserved','manifest','extra','missing'])
def test_no_unqualified_transfer_partial_cohort_or_reference_admission(tmp_path,monkeypatch,fault):
    code,rev,root,seqs=prepared(tmp_path,monkeypatch)
    path=root/'public-transfer.json';data=json.loads(path.read_text())
    if fault=='GT':data['private_references_transferred']=True
    elif fault=='notpass':data['status']='fail'
    elif fault=='reserved':data['sequences'][-1]='RESERVED_NOT_READ'
    elif fault=='manifest':data['manifest_identity']['sha256']='not-hash'
    elif fault=='extra':(root/'eval_private').mkdir()
    else:(root/seqs[-1]/'inputs/input.json').unlink()
    path.chmod(0o600);path.write_text(json.dumps(data));path.chmod(0o444)
    with pytest.raises((ValueError,FileNotFoundError)):cohort.discover(predictor,code,rev)


def test_each_sequence_has_isolated_auth_pair_and_unambiguous_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(predictor,'ROOT',tmp_path)
    pairs=[predictor.credential_paths('a'*40,'FORM_DEV_'+str(i)) for i in range(4)]
    assert len({str(p) for pair in pairs for p in pair})==8
    assert all(p.parent==tmp_path/'.secrets' for pair in pairs for p in pair)
    assert pairs[0][0].name=='form-hoi-external-'+'a'*40+'-FORM_DEV_0-key.pem'
    for sid in ('../unsafe','/absolute','with space','',True):
        with pytest.raises(ValueError):predictor.credential_paths('a'*40,sid)


def test_CPU_localization_is_actually_parallel_and_all_completed_before_return():
    barrier=threading.Barrier(4);done=[];lock=threading.Lock()
    def invoke(row):
        barrier.wait(timeout=5)
        with lock:done.append(row)
    cohort.dispatch(list(range(4)),'localize',invoke)
    assert sorted(done)==list(range(4))


def test_CPU_failure_waits_for_started_workers_and_never_swallows_error():
    barrier=threading.Barrier(4);done=[];lock=threading.Lock()
    def invoke(row):
        barrier.wait(timeout=5)
        if row==1:raise RuntimeError('technical')
        time.sleep(.005)
        with lock:done.append(row)
    with pytest.raises(RuntimeError,match='technical'):cohort.dispatch(list(range(4)),'localize',invoke)
    assert sorted(done)==[0,2,3]


def test_GPU_chains_are_serial_ordered_and_stop_first_technical_failure():
    seen=[];active=0
    def invoke(row):
        nonlocal active
        active+=1;assert active==1;seen.append(row);active-=1
    cohort.dispatch(list(range(4)),'all',invoke)
    assert seen==list(range(4))
    def broken(row):
        seen.append(row)
        if row==1:raise RuntimeError('model contract')
    seen.clear()
    with pytest.raises(RuntimeError):cohort.dispatch(list(range(4)),'all',broken)
    assert seen==[0,1]


def test_wrappers_use_original_predictor_ENTRY_without_parent_GPU_lease():
    import subprocess
    for name in ('run_form_hoi_external_cohort.sh','run_form_hoi_external_predict.sh'):
        subprocess.run(['bash','-n',str(REPO/'infra'/name)],check=True)
    text=(REPO/'infra/run_form_hoi_external_cohort.sh').read_text()
    assert 'run_form_hoi_external_predict/code' in text and '--cohort-stage' in text
    assert '62000s' in text and 'flock' not in text and 'docker run' not in text
    assert 'infra/form_hoi_external_cohort.py' in predictor.HELPERS


def test_resume_preflight_failure_is_sealed_with_all_four_not_started(tmp_path,monkeypatch,capsys):
    code,dev,_,seqs=prepared(tmp_path,monkeypatch); revision='c'*40
    monkeypatch.setattr(predictor,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(predictor,'source',lambda *_a:dict(producer_revision=revision))
    import form_stage_reuse
    seen=[]
    def broken(*_a,**kwargs):
        seen.append(kwargs['stop']);raise FileNotFoundError('missing terminal receipt')
    monkeypatch.setattr(form_stage_reuse,'reuse',broken)
    monkeypatch.setattr(predictor,'driver',lambda *_a:pytest.fail('No inference before admission'))
    stop=dict(path=tmp_path/'results/stop.json',pin=dict(bytes=123,sha256='f'*64))
    capacity=dict(path=tmp_path/'results/capacity.json',bytes=1,sha256='a'*64)
    prefix=dict(path=tmp_path/'results/prefix.json',bytes=2,sha256='b'*64)
    prepare=dict(path=tmp_path/'results/prepare-prefix.json',bytes=3,sha256='c'*64)
    import raster_prefix_activation
    monkeypatch.setattr(raster_prefix_activation,'activation',lambda *_a,**_k:None)
    monkeypatch.setattr(predictor,'config',lambda *_a:dict(body_image='native-pinned-image'))
    with pytest.raises(FileNotFoundError):
        cohort.run(predictor,code,revision,stage='all',dev_revision=dev,reuse_stages_from='d'*40,
            raster_gate=capacity,raster_prefix=prefix,prepare_prefix=prepare,terminal_stop=stop)
    value=json.loads((tmp_path/'results'/f'form-hoi-external-cohort-{revision}-all.json').read_text())
    assert seen==[stop] and value['status']=='fail' and value['error_type']=='FileNotFoundError'
    assert value['failure_phase']=='preflight' and value['predictions_started'] is False
    assert value['sequence_order']==seqs and all(r['status']=='not_started' for r in value['sequences'])
    assert value['ground_truth_used'] is False and value['private_truth_read'] is False
    assert not (tmp_path/'results'/('form-hoi-external-predict-'+revision)).exists()
    assert value['raster_runtime_controls']['capacity']==dict(capacity,path=str(capacity['path']))
    assert value['raster_runtime_controls']['prefix']==dict(prefix,path=str(prefix['path']))
    assert value['technical_resume_controls']==dict(prepare_prefix=dict(prepare,path=str(prepare['path'])),
        terminal_stop=dict(stop,path=str(stop['path'])))
    assert isinstance(capacity['path'],Path) and isinstance(stop['path'],Path)
    stderr=json.loads(capsys.readouterr().err)
    assert stderr==dict(status='fail',producer_revision=revision,error_type='FileNotFoundError',failure_phase='preflight')


def test_success_receipt_uses_JSON_paths_without_changing_execution_controls(tmp_path,monkeypatch):
    code,dev,_,seqs=prepared(tmp_path,monkeypatch); revision='d'*40
    monkeypatch.setattr(predictor,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(predictor,'source',lambda *_a:dict(producer_revision=revision))
    capacity=dict(path=tmp_path/'capacity.json',bytes=1,sha256='a'*64)
    prefix=dict(path=tmp_path/'prefix.json',bytes=2,sha256='b'*64)
    stop=dict(path=tmp_path/'stop.json',pin=dict(bytes=3,sha256='c'*64))
    prepare=dict(path=tmp_path/'prepare.json',bytes=4,sha256='e'*64)
    import raster_prefix_activation,form_stage_reuse
    monkeypatch.setattr(raster_prefix_activation,'activation',lambda path,*_a,**_k:None)
    monkeypatch.setattr(predictor,'config',lambda *_a:dict(body_image='native-pinned-image'))
    def reuse(*_a,**kwargs):
        assert kwargs['stop'] is stop and kwargs['prepare_prefix'] is prepare
        for sid in seqs:
            d=tmp_path/'results'/('form-hoi-external-predict-'+revision)/sid/'localize';d.mkdir(parents=True)
            predictor.save_json(d/'report.json',dict(status='complete',stage='localize',ground_truth_used=False,private_truth_read=False,artifacts={}))
        return dict(original_localization_proof={},localization_API_calls=0)
    monkeypatch.setattr(form_stage_reuse,'reuse',reuse)
    def driver(args):
        # Metadata-only stage reporting; no models or real stage payloads.
        for stage in predictor.STAGES[1:]:
            d=args.out/stage;d.mkdir()
            predictor.save_json(d/'report.json',dict(status='complete',stage=stage,ground_truth_used=False,private_truth_read=False,artifacts={}))
    monkeypatch.setattr(predictor,'driver',driver)
    value=cohort.run(predictor,code,revision,stage='all',dev_revision=dev,reuse_stages_from='f'*40,
        raster_gate=capacity,raster_prefix=prefix,prepare_prefix=prepare,terminal_stop=stop)
    sealed=json.loads((tmp_path/'results'/f'form-hoi-external-cohort-{revision}-all.json').read_bytes())
    assert sealed['status']=='complete' and len(sealed['sequences'])==4
    assert sealed['technical_resume_controls']['terminal_stop']['path']==str(stop['path'])
    assert sealed['raster_runtime_controls']['capacity']['path']==str(capacity['path'])
    assert isinstance(capacity['path'],Path) and all(r['status']=='complete' for r in value['sequences'])

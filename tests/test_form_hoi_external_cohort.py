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

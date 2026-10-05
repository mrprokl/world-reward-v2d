"""Tiny failure-lifecycle controls only; no native geometry/runtime qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('surface_failure_test',ROOT/'infra/object_budget_solid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def failure():
    return dict(status='fail',stage='world_reward_object_budget_surface_native_v1',
        producer_revision='a'*40,domain='surface',episode_index=26,phase='mapping_replay',
        elapsed_seconds=600.25,failure_type='TimeoutError',failure_reason='Original compute deadline',
        source_binding=dict(producer_revision='a'*40,closure_sha256='b'*64),
        compiler=dict(native_counts=dict(preflight_attempts=1,preflight_returns=1,qem_attempts=1,qem_returns=1),
            mapping=['oversize payload']*10000),outputs=dict(geometry='not evidence'),qualification=dict(secret='omitted'),
        phase_progress=dict(qem_completed=True,mapping_read_seconds=2.5),last_completed_operation='mapping_decode',
        phase_timings=[dict(phase='native_qem',elapsed_seconds=1.,qem_attempts=1),dict(phase='mapping_replay',elapsed_seconds=2.)])


def test_late_failure_preserves_original_reason_and_only_scalar_evidence(gate,tmp_path):
    report=failure();path=tmp_path/'native.json'
    gate.surface_failure_report(path,report,gate.time.monotonic()+9,computation_finished=True)
    value=json.loads(path.read_bytes())
    assert value['status']=='fail' and value['failure_type']=='TimeoutError'
    assert value['failure_reason']==report['failure_reason'] and value['computation_deadline_exceeded']
    assert value['native_counts']==report['compiler']['native_counts'] and value['phase_progress']==report['phase_progress']
    assert value['phase_timings']==report['phase_timings']
    assert value['source_binding_sha256']==hashlib.sha256(json.dumps(report['source_binding'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert not {'source_binding','compiler','outputs','qualification','mapping'}&set(value)
    assert value['adoption'] is value['geometry_published'] is value['qualification_verified'] is False
    assert stat.S_IMODE(path.stat().st_mode)==0o444 and path.stat().st_nlink==1 and path.stat().st_size<16384
    assert report==failure()


@pytest.mark.parametrize('fault',['pass','unfinished','expired','excessive_grace','nonfinite_time','bool_episode'])
def test_late_writer_rejects_nonfailure_or_unbounded_contract(gate,tmp_path,fault):
    report=failure();finished=True;deadline=gate.time.monotonic()+9
    if fault=='pass':report['status']='pass'
    elif fault=='unfinished':finished=False
    elif fault=='expired':deadline=gate.time.monotonic()-1
    elif fault=='excessive_grace':deadline=gate.time.monotonic()+11
    elif fault=='nonfinite_time':report['elapsed_seconds']=float('nan')
    else:report['episode_index']=True
    with pytest.raises(ValueError):gate.surface_failure_report(tmp_path/'native.json',report,deadline,computation_finished=finished)
    assert not (tmp_path/'native.json').exists()


@pytest.mark.parametrize('kind',['file','symlink'])
def test_failure_writer_exclusive_never_overwrites_existing_receipt(gate,tmp_path,kind):
    target=tmp_path/'original';target.write_bytes(b'unchanged historical receipt')
    path=tmp_path/'native.json'
    if kind=='file':path.write_bytes(b'owned or foreign existing receipt')
    else:path.symlink_to(target)
    before=path.read_bytes()
    with pytest.raises(OSError):gate.surface_failure_report(path,failure(),gate.time.monotonic()+9,computation_finished=True)
    assert path.read_bytes()==before and target.read_bytes()==b'unchanged historical receipt'


@pytest.mark.parametrize('timings',[[{'bad':[1,2]}],[{'phase':'x'*2049}],['not a scalar row'],[{}]*33])
def test_oversized_or_nonscalar_phase_records_cannot_escape_projection(gate,tmp_path,timings):
    report=failure();report['phase_timings']=timings
    with pytest.raises(ValueError):gate.surface_failure_report(tmp_path/'native.json',report,gate.time.monotonic()+9,computation_finished=True)
    assert not(tmp_path/'native.json').exists()


def test_authored_control_minus_one_is_only_allowed_for_exact_control_stage(gate,tmp_path):
    report=failure();report['episode_index']=-1
    with pytest.raises(ValueError):gate.surface_failure_report(tmp_path/'wrong.json',report,gate.time.monotonic()+9,computation_finished=True)
    report['stage']='world_reward_surface_consumer_control_native_v1'
    gate.surface_failure_report(tmp_path/'native.json',report,gate.time.monotonic()+9,computation_finished=True)
    assert json.loads((tmp_path/'native.json').read_bytes())['episode_index']==-1


def test_bounded_phase_history_keeps_latest_records_with_explicit_omission(gate,tmp_path):
    report=failure();report['phase_timings']=[dict(phase='phase_'+str(i),elapsed_seconds=float(i),message='x'*400)for i in range(32)]
    path=tmp_path/'native.json';gate.surface_failure_report(path,report,gate.time.monotonic()+9,computation_finished=True)
    value=json.loads(path.read_bytes())
    assert value['phase_timings']==report['phase_timings'][-8:] and value['phase_timings_omitted']==24
    assert path.stat().st_size<=16384


@pytest.mark.parametrize('fault',['short_write','fsync_error','write_overrun'])
def test_failed_write_is_never_a_pass_and_descriptor_is_closed(gate,monkeypatch,tmp_path,fault):
    path=tmp_path/'native.json';closed=[];original_close=gate.os.close
    monkeypatch.setattr(gate.os,'close',lambda fd:(closed.append(fd),original_close(fd))[1])
    if fault=='short_write':monkeypatch.setattr(gate.os,'write',lambda *_:0)
    elif fault=='fsync_error':monkeypatch.setattr(gate.os,'fsync',lambda *_:(_ for _ in()).throw(OSError('fsync failed')))
    else:
        clock=iter((0.,0.,11.));monkeypatch.setattr(gate.time,'monotonic',lambda:next(clock))
    with pytest.raises((ValueError,OSError)):gate.surface_failure_report(path,failure(),10. if fault=='write_overrun'else gate.time.monotonic()+9,computation_finished=True)
    assert len(closed)==1
    if path.stat().st_size:assert json.loads(path.read_bytes())['status']=='fail'


def test_normal_sealing_deadline_revokes_pass_without_masking_existing_failure(gate,tmp_path):
    calls=[]
    def left():
        calls.append(1)
        if len(calls)>1:raise TimeoutError('Receipt completion crossed compute deadline')
        return 1
    report=dict(status='pass');path=tmp_path/'normal.json'
    with pytest.raises(TimeoutError):gate.surface_report(path,report,None,left)
    assert json.loads(path.read_bytes())['status']=='fail'
    calls.clear();report=failure();path=tmp_path/'failure.json'
    with pytest.raises(TimeoutError):gate.surface_report(path,report,None,left)
    value=json.loads(path.read_bytes());assert value['failure_type']=='TimeoutError' and value['failure_reason']=='Original compute deadline'


@pytest.mark.parametrize('mode',['timeout','ordinary_failure','late_success'])
def test_native_finally_persists_fail_after_computation_no_geometry_published(gate,monkeypatch,tmp_path,mode):
    clock=[0.];calls=[];source=dict(producer_revision='a'*40,closure_sha256='b'*64)
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0])
    # Environment guards are not an Azure runtime claim; only the actual finally is exercised.
    original_require=gate.require
    monkeypatch.setattr(gate,'require',lambda ok,msg:None if msg=='Restricted offline CPU required'else original_require(ok,msg))
    monkeypatch.setattr(gate.signal,'signal',lambda *args:None)
    monkeypatch.setattr(gate.signal,'alarm',lambda n:calls.append(('alarm',n)))
    monkeypatch.setattr(gate,'surface_source',lambda *a,**k:source)
    proof=dict(build=dict(source_cpp=dict(sha256='c'*64)))
    monkeypatch.setattr(gate,'surface_qualification',lambda *_,**kw:(tmp_path/'never_executed',(),proof))
    monkeypatch.setenv('WR_CPU_IMAGE_ID',gate.SURFACE_IMAGE)
    monkeypatch.setattr(gate,'input_binding',lambda *_:(None,(),dict(video_sha256='d'*64)))
    def remaining(started,budget):
        if clock[0]-started>=budget:raise TimeoutError('Original compute deadline')
        return budget-(clock[0]-started)
    q=SimpleNamespace(modules=lambda:(None,None,None,SimpleNamespace(remaining=remaining)),binary_runtime=lambda *args:{})
    def produce(*args):
        report=args[5];report.update(phase='mapping_replay',compiler=dict(native_counts=dict(qem_attempts=1,qem_returns=1)))
        clock[0]=600.25 if mode!='ordinary_failure'else 2.
        if mode!='late_success':raise TimeoutError('Original compute deadline')if mode=='timeout'else ValueError('Original scalar failure')
    monkeypatch.setattr(gate,'surface_produce',produce)
    rt=SimpleNamespace(identity=lambda *args:dict(bytes=1,sha256='e'*64))
    assert gate.surface_native(26,tmp_path,'a'*40,tmp_path,rt,q,None,None)==1
    value=json.loads((tmp_path/'native.json').read_bytes());assert value['status']=='fail'
    assert value['failure_type']==('ValueError'if mode=='ordinary_failure'else'TimeoutError')
    assert value['phase']=='mapping_replay'if mode!='late_success'else value['phase']=='complete'
    if mode!='ordinary_failure':assert value['failure_only_receipt'] and 'outputs'not in value
    assert not any((tmp_path/n).exists()for n in gate.SURFACE_OUTPUTS)
    assert ('alarm',0)==calls[-1] and not any(n>600 for kind,n in calls)

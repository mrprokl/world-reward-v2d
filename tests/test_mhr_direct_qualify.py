"""Manufactured metadata/callback tests: no real MHR, Torch, Docker or GPU."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'infra'))
spec=importlib.util.spec_from_file_location('mhr_direct_qualify_test',ROOT/'infra/mhr_direct_qualify.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def test_host_authentication_precedes_any_load(monkeypatch,tmp_path):
    code=ROOT;calls=[];model=tmp_path/'model';old=tmp_path/'old'
    rt=SimpleNamespace(source=lambda *a:(calls.append('source')or dict(real=False)))
    rt.require=lambda x,m:None if x else (_ for _ in()).throw(ValueError(m))
    bridge=SimpleNamespace(authenticate_release=lambda *a:(calls.append('release')or dict(
        source={'historical':True},release_report_identity={'bytes':1,'sha256':'a'*64},
        frozen={model:{'bytes':1,'sha256':'b'*64}},model_path=model,original_code=old)),
        recheck_release=lambda *a:calls.append('recheck'))
    monkeypatch.setattr(q,'runtime',lambda:rt);monkeypatch.setattr(q,'bridge_module',lambda:bridge)
    proof=q.host_proof(tmp_path,code,'a'*40)
    assert calls==['source','release','recheck','source']and proof['model_path']==str(model)


def test_mounts_are_only_current_historical_source_notices_and_model(tmp_path):
    code=tmp_path/'jobs/current/run_mhr_direct_qualify/code';old=tmp_path/'jobs/c416/acquire_weights/code'
    model=tmp_path/'weights/mhr/mhr_model.pt';notice=tmp_path/'results/mhr-official-release-license-v2/report.json'
    paths=q.mounts(dict(original_code=str(old),model_path=str(model),frozen={str(notice):{}}),code)
    assert paths==(code.parent,old.parent,model,notice.parent)
    assert not any('sam3d' in str(p)or 'data/' in str(p)for p in paths)


@pytest.fixture
def callback_case():
    calls=[];dtype=object();floating=SimpleNamespace(is_floating_point=lambda:True,dtype=dtype)
    model=SimpleNamespace(eval=lambda: model,named_parameters=lambda:[('native',floating)],named_buffers=lambda:[])
    @contextmanager
    def optimized(value):
        calls.append(('optimized',value));yield
    torch=SimpleNamespace(float32=dtype,jit=SimpleNamespace(optimized_execution=optimized,
        load=lambda path,**kw:(calls.append(('load',path,kw))or model)))
    def control(actual,t,*,check):
        assert actual is model and t is torch
        calls.append('bridge');check();return {'explicit_manufactured':True}
    return calls,torch,SimpleNamespace(run_control=control),floating


def test_one_load_one_nine_batch_bridge_call_no_cast_or_repeat(callback_case):
    calls,torch,bridge,_=callback_case;events=[]
    result=q.execute_control({'model_path':'exact-model'},torch,bridge,lambda:calls.append('check'),event=events.append)
    assert result=={'explicit_manufactured':True}
    assert calls==['check',('optimized',False),('load','exact-model',{'map_location':'cuda'}),'check','bridge','check']
    assert events==['model_load_attempts','model_load_returns','control_attempts','control_returns']


def test_wrong_dtype_abstains_before_decode(callback_case):
    calls,torch,bridge,floating=callback_case;floating.dtype=object()
    with pytest.raises(ValueError,match='already be float32'):
        q.execute_control({'model_path':'exact-model'},torch,bridge,lambda:None)
    assert 'bridge'not in calls


def test_callback_failure_propagates_without_retry(callback_case):
    calls,torch,bridge,_=callback_case
    bridge.run_control=lambda *a,**k:(_ for _ in()).throw(RuntimeError('manufactured failure'))
    with pytest.raises(RuntimeError,match='manufactured'):
        q.execute_control({'model_path':'exact-model'},torch,bridge,lambda:None)
    assert sum(isinstance(c,tuple)and c[0]=='load'for c in calls)==1


def test_receipt_exclusive_sealed_and_late_pass_demoted(tmp_path):
    parent=tmp_path/'output';parent.mkdir();path=parent/'native.json';record={'status':'pass'}
    def overrun():raise TimeoutError('manufactured deadline')
    with pytest.raises(TimeoutError):q.write_receipt(path,record,overrun,seal_parent=True)
    assert json.loads(path.read_bytes())==dict(status='fail',phase='receipt',failure_type='TimeoutError')
    assert path.stat().st_mode&0o777==0o444 and parent.stat().st_mode&0o777==0o555
    parent.chmod(0o755)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):q.write_receipt(path,{'status':'pass'},lambda:None)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_oversize_report_cannot_be_a_pass(tmp_path):
    with pytest.raises(ValueError,match='Bounded scalar'):
        q.write_receipt(tmp_path/'native.json',{'status':'pass','payload':'x'*100},lambda:None,maximum=32)
    assert (tmp_path/'native.json').read_bytes()==b''


@pytest.mark.parametrize('stdout,stderr',[(b'',b'Error: No such object: '),(b'\n',b'error: no such object: '),
    (b'[]\n',b'\nerror: no such object: ')])
def test_docker_exact_absence_separate_streams(stdout,stderr):
    cid='a'*64
    def run(*args,**kwargs):
        assert kwargs==dict(capture_output=True,timeout=5)
        return SimpleNamespace(returncode=1,stdout=stdout,stderr=stderr+cid.encode()+b'\n')
    assert q.inspect_container(cid,'owned','b'*40,run=run)is False


@pytest.mark.parametrize('rc,stdout,stderr',[(1,b'',b'Cannot connect to daemon'),(124,b'',b''),
    (1,b'Error: No such object: '+b'a'*64,b''),(0,b'{}\n',b''),
    (1,b'[]\n',b' daemon context\nerror: no such object: '+b'a'*64)])
def test_docker_errors_or_foreign_never_mean_absence(rc,stdout,stderr):
    with pytest.raises(ValueError):
        q.inspect_container('a'*64,'owned','b'*40,run=lambda *a,**k:SimpleNamespace(returncode=rc,stdout=stdout,stderr=stderr))


def test_live_container_requires_exact_name_image_label():
    cid='a'*64;revision='b'*40
    run=lambda *a,**k:SimpleNamespace(returncode=0,stdout=f'{cid} /owned {q.IMAGE} {revision}\n'.encode(),stderr=b'')
    assert q.inspect_container(cid,'owned',revision,run=run)is True
    with pytest.raises(ValueError):q.inspect_container(cid,'foreign',revision,run=run)


@pytest.mark.parametrize('present_after,rm_status',[(False,0),(True,0),(False,1)])
def test_cleanup_requires_removal_and_independent_absence(monkeypatch,tmp_path,present_after,rm_status):
    cid='a'*64;(tmp_path/'.container.cid').write_text(cid);queries=[];removals=[]
    def inspect(*args):queries.append(args);return len(queries)==1 or present_after
    monkeypatch.setattr(q,'inspect_container',inspect)
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**k:(removals.append(a[0])or SimpleNamespace(returncode=rm_status)))
    if not present_after and rm_status==0:
        q.cleanup(tmp_path,'owned','b'*40);assert len(queries)==2
    else:
        with pytest.raises(ValueError):q.cleanup(tmp_path,'owned','b'*40)
    assert removals==[['docker','rm','-f',cid]]


def test_invalid_cid_never_reaches_docker(monkeypatch,tmp_path):
    (tmp_path/'.container.cid').write_text('a'*63)
    monkeypatch.setattr(q,'inspect_container',lambda *a:pytest.fail('must not inspect invalid CID'))
    with pytest.raises(ValueError):q.cleanup(tmp_path,'owned','b'*40)


@pytest.mark.parametrize('status,cleanup_verified',[(17,True),(0,False)])
def test_host_seal_never_hides_original_fail_or_uncertified_cleanup(monkeypatch,tmp_path,status,cleanup_verified):
    revision='a'*40;out,control=q.namespaces(tmp_path,revision)
    out.mkdir(parents=True);control.mkdir()
    proof={'source':{'manufactured':True},'release_report_identity':{'bytes':1,'sha256':'b'*64}}
    (control/'proof.json').write_text(json.dumps(proof));(control/'.container.cid').write_text('c'*64)
    (out/'native.json').write_text(json.dumps({'status':'pass','stage':'not-a-native-proof'}))
    (out/'native.json').chmod(0o444)
    monkeypatch.setattr(q,'host_proof',lambda *a:proof)
    monkeypatch.setattr(q,'cleanup',lambda *a:None)
    with pytest.raises(ValueError):q.seal(tmp_path,ROOT,revision,status,cleanup_verified)
    host=json.loads((control/'report.json').read_bytes())
    assert host['status']=='fail'and host['original_exit_status']==status
    assert host['owned_container_absence_verified']is cleanup_verified
    assert control.stat().st_mode&0o777==0o555


def test_wrapper_and_native_source_contract():
    path=ROOT/'infra/run_mhr_direct_qualify.sh';text=path.read_text();source=Path(q.__file__).read_text()
    subprocess.run(['bash','-n',str(path)],check=True)
    for marker in('PYHOST','PYLOCK'):ast.parse(text.split("<<'"+marker+"'\n",1)[1].split('\n'+marker,1)[0])
    assert 'exec 9<"$LOCK"'in text and 'flock --timeout 43200 9'in text
    assert text.index('flock --timeout')<text.index('nvidia-smi')<text.index('PATHS="$(host before)"')
    assert '--name "$NAME" --cidfile "$CONTROL/.container.cid"'in text
    assert '--gpus all --network none --read-only --cap-drop ALL'in text
    assert '--env CUBLAS_WORKSPACE_CONFIG=:4096:8'in text
    assert '"$IMAGE" "$CODE/infra/mhr_direct_qualify.py"'in text
    assert '$ROOT/data'not in text and '$ROOT/vendor'not in text and 'sam3d'not in text
    assert 'use_deterministic_algorithms(True,warn_only=False)'in source
    assert 'jit_optimized_execution=False'in source and 'BUDGET = 120'in source
    assert 'bridge.authenticate_release(root,code,rt)'in source and 'torch.jit.load'in source


def test_finish_checks_lock_before_sealing_and_retains_nonzero(tmp_path):
    text=(ROOT/'infra/run_mhr_direct_qualify.sh').read_text()
    function='finish() {'+text.split('finish() {',1)[1].split('\ntrap finish EXIT',1)[0]
    for original,lockfail,expected in ((0,True,1),(17,True,17),(0,False,0)):
        log=tmp_path/f'{original}-{lockfail}'
        script=f'''LOCK_BEFORE=original
host() {{ [ "$1" != seal ] || echo "$2" > "{log}";return 0; }}
lock_identity() {{ echo {'changed'if lockfail else'original'}; }}
{function}
exec 9</dev/null
trap finish EXIT
exit {original}
'''
        result=subprocess.run(['bash','-c',script],capture_output=True,text=True)
        assert result.returncode==expected and log.read_text().strip()==str(expected)


def test_write_postseal_failure_demotes_pass_under_restrictive_umask(tmp_path):
    directory=tmp_path/'own';directory.mkdir();record={'status':'pass'}
    previous=os.umask(0o777)
    try:
        def fail_after_seal():
            assert directory.stat().st_mode&0o777==0o555
            assert (directory/'report.json').stat().st_mode&0o777==0o444
            raise ValueError('manufactured post-seal failure')
        with pytest.raises(ValueError,match='post-seal'):
            q.write_receipt(directory/'report.json',record,fail_after_seal,seal_parent=True)
    finally:os.umask(previous)
    assert json.loads((directory/'report.json').read_text())['status']=='fail'

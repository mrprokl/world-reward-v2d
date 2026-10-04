"""Manufactured failure-only metadata: no source dataset, annotations or jobs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('ycbv_transition_test',REPO/'infra/ycbv_acquire_transition.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_bytes(data);path.chmod(0o400)
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def fixture(tmp_path,monkeypatch):
    root=tmp_path;rev='a'*40;old=root/'jobs'/rev/'run_ycbv_point_acquire/code'
    for name in gate.FILES:
        data=json.dumps({'limits':{'seconds':900},'output':{'base':gate.BASE}}).encode() if name.endswith('.json')else ('tiny'+name).encode()
        write(old/name,data)
    write(old.parent/'revision',(rev+'\n').encode());write(old.parent/'source-sha256',('b'*64+'\n').encode())
    original=gate.source(old,rev)
    base=root/gate.BASE;base.mkdir(parents=True,mode=0o700)
    report=dict(stage='external_ycbv_contiguous_rgb_only_acquisition',status='fail',phase='download',error_type='TimeoutError',
        producer_revision=rev,script_sha256=original['files'][gate.FILES[0]]['sha256'],source_helpers=original,budget_seconds=900,
        device='cpu',gpu_used=False,inference_performed=False,challenge_inputs_used=False,models_downloaded=False,
        train_downloaded=False,sparse_test_downloaded=False,source_rehashed_after=True,disposable_archives_removed=True,active_archive='ycbv_test_all.zip')
    identity=write(base/'report.json',json.dumps(report).encode());cid='c'*64;cidpin=write(base/'.container.cid',(cid+'\n').encode())
    pins=dict(schema='world-reward-ycbv-failed-acquisition-pins-v1',failure={**identity,'producer_revision':rev,
        'script_sha256':report['script_sha256']},unit='world-reward-ycbv-acquire-first.service',pid=12345,
        log=write(root/'results/ycbv-acquire-first.log',b'old private-free failure log'),cid=cidpin)
    current='d'*40;code=root/'jobs'/current/gate.JOB/'code';write(code/gate.FAILED_PINS,json.dumps(pins).encode())
    for n in ('infra/ycbv_acquire_transition.py','infra/run_ycbv_acquire_transition.sh','infra/atomic_metadata.py'):write(code/n,('current'+n).encode())
    write(code.parent/'revision',(current+'\n').encode());write(code.parent/'source-sha256',('e'*64+'\n').encode())
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/'infra/ycbv_acquire_transition.py'))
    # Native dirs must really be UID1000; local root-run CI instead checks own current uid.
    original_lstat=Path.lstat
    def lstat(path):
        s=original_lstat(path)
        if path in (base,root/gate.ARCHIVE):return SimpleNamespace(**{n:getattr(s,n)for n in dir(s)if n.startswith('st_')},st_uid_override=1000)
        return s
    # Do not mock bytes, source, report or inode checks; only host UID projection.
    real_failure=gate.failure
    def failure(root,pins,folder):
        if os.getuid()==1000:return real_failure(root,pins,folder)
        def projected(path):
            s=original_lstat(path)
            if path==folder:
                fields={n:getattr(s,n)for n in dir(s)if n.startswith('st_')};fields['st_uid']=1000;return SimpleNamespace(**fields)
            return s
        with monkeypatch.context()as m:
            m.setattr(Path,'lstat',projected)
            return real_failure(root,pins,folder)
    monkeypatch.setattr(gate,'failure',failure)
    state=dict(LoadState='loaded',ActiveState='failed',SubState='failed',Result='exit-code',MainPID='0',ExecMainPID=str(pins['pid']),ControlGroup='')
    def query(args):
        return '\n'.join(k+'='+v for k,v in state.items()) if args[0]=='systemctl'else ''
    return root,current,code,base,pins,report,state,query


def test_atomic_archive_preserves_entire_old_failure_bytes_and_inode(tmp_path,monkeypatch):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch)
    before={p.name:p.read_bytes()for p in base.iterdir()};inode=base.stat().st_ino
    def rename(src,dst):
        assert not dst.exists();src.rename(dst)
    result=gate.archive(root,code,rev,query=query,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=rename)
    dst=root/gate.ARCHIVE
    assert result['status']=='pass'and result['acquisition_attempts_authorized']==2
    assert dst.stat().st_ino==inode and {p.name:p.read_bytes()for p in dst.iterdir()}==before and not base.exists()
    receipt=root/'results'/('ycbv-acquire-continuation-'+rev)/'report.json'
    assert receipt.stat().st_mode&0o777==0o400
    with pytest.raises(ValueError):gate.archive(root,code,rev,query=query,rename=rename)


def test_original_systemd_log_0644_is_pinned_without_chmod(tmp_path,monkeypatch):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch)
    log=root/'results/ycbv-acquire-first.log';log.chmod(0o644)
    before=log.lstat();raw=log.read_bytes()
    gate.archive(root,code,rev,query=query,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=lambda a,b:a.rename(b))
    assert log.read_bytes()==raw and gate.atomic._state(log)==(
        before.st_dev,before.st_ino,before.st_mode,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_nlink)


def test_measured_original_failure_config_is_typed():
    pins=json.loads((REPO/gate.FAILED_PINS).read_bytes());gate.validate_pins(pins)
    assert pins['failure']['producer_revision']=='40ee2cb4710ec15f0b2bda32bbf0e1f727804b18'
    assert pins['failure']['bytes']==2038 and pins['log']['bytes']==65 and pins['cid']['bytes']==64 and pins['pid']==213665


@pytest.mark.parametrize('fault',['not_timeout','private_phase','selected','cleanup','source','report','extra','destination','pid','active','container'])
def test_transition_fails_without_mutating_original(tmp_path,monkeypatch,fault):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch)
    if fault=='not_timeout':report['error_type']='ValueError'
    elif fault=='private_phase':report['phase']='retain'
    elif fault=='selected':report['selected_frames']=288
    elif fault=='cleanup':report['disposable_archives_removed']=False
    elif fault=='source':report['source_helpers']['files'][gate.FILES[0]]['sha256']='f'*64
    elif fault=='report':pins['failure']['sha256']='f'*64
    elif fault=='extra':write(base/'unknown',b'foreign')
    elif fault=='destination':(root/gate.ARCHIVE).mkdir()
    elif fault=='pid':(tmp_path/'proc'/str(pins['pid'])).mkdir(parents=True)
    elif fault=='active':state.update(ActiveState='active',SubState='running')
    elif fault=='container':
        original=query;query=lambda args:'c'*64 if args[0]=='docker'else original(args)
    if fault in ('not_timeout','private_phase','selected','cleanup','source'):
        pins['failure'].update(write(base/'report.json',json.dumps(report).encode()))
    write(code/gate.FAILED_PINS,json.dumps(pins).encode())
    before={p.name:p.read_bytes()for p in base.iterdir()};called=[]
    with pytest.raises(ValueError):gate.archive(root,code,rev,query=query,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=lambda *a:called.append(a))
    assert called==[]and {p.name:p.read_bytes()for p in base.iterdir()}==before


def test_rename_failure_is_sealed_fail_not_authorization(tmp_path,monkeypatch):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch)
    def fail(*a):raise OSError('rename blocked')
    with pytest.raises(OSError):gate.archive(root,code,rev,query=query,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=fail)
    receipt=json.loads((root/'results'/('ycbv-acquire-continuation-'+rev)/'report.json').read_bytes())
    assert receipt['status']=='fail'and receipt['atomic_noreplace']is False and receipt['acquisition_attempts_authorized']==0 and base.exists()


def test_linux_noreplace_never_overwrites_occupied_directory(tmp_path):
    import platform
    if platform.system()!='Linux':pytest.skip('Native Linux renameat2 only')
    src=tmp_path/'source';dst=tmp_path/'occupied';src.mkdir();dst.mkdir()
    with pytest.raises(FileExistsError):gate.atomic.rename_noreplace(src,dst)
    assert src.exists()and dst.exists()


def test_wrapper_host_cpu_only_and_exact_sources():
    import subprocess
    path=REPO/'infra/run_ycbv_acquire_transition.sh';s=path.read_text()
    assert subprocess.run(['bash','-n',str(path)]).returncode==0
    assert '--gpus'not in s and 'docker run'not in s and 'run_ycbv_acquire_transition/code'in s
    assert '[[ $# == 0'in s and '120s'in s and 'atomic_metadata.py'in s


def test_actual_transition_authorization_and_second_fail_closed(tmp_path,monkeypatch):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch)
    gate.archive(root,code,rev,query=query,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=lambda a,b:a.rename(b))
    retry=root/'jobs'/('e'*40)/'run_ycbv_point_acquire/code';retry.mkdir(parents=True)
    write(retry/gate.FAILED_PINS,json.dumps(pins).encode())
    transition_path=root/'results'/('ycbv-acquire-continuation-'+rev)/'report.json'
    transition_id=gate.read(transition_path)[1]
    config=dict(schema='world-reward-ycbv-continuation-pins-v1',failed_pins=gate.read(code/gate.FAILED_PINS)[1],
        transition={**transition_id,'producer_revision':rev,'script_sha256':gate.read(code/'infra/ycbv_acquire_transition.py')[1]['sha256']})
    write(retry/gate.CONTINUATION_PINS,json.dumps(config).encode())
    protocol={'limits':{'seconds':3600,'cleanup_seconds':180},'output':{'base':gate.BASE}}
    write(retry/gate.FILES[-1],json.dumps(protocol).encode())
    assert len(gate.continuation(root,retry))==64
    protocol['output']['base']='validation/alternate';write(retry/gate.FILES[-1],json.dumps(protocol).encode())
    with pytest.raises(ValueError):gate.continuation(root,retry)
    protocol['output']['base']=gate.BASE;write(retry/gate.FILES[-1],json.dumps(protocol).encode())
    archived=root/gate.ARCHIVE/'report.json';raw=archived.read_bytes();write(archived,raw+b' ')
    with pytest.raises(ValueError):gate.continuation(root,retry)


def test_container_reappearing_at_atomic_boundary_refuses_rename(tmp_path,monkeypatch):
    root,rev,code,base,pins,report,state,query=fixture(tmp_path,monkeypatch);calls=[]
    def racing(args):
        if args[0]=='docker':
            calls.append('docker');return '' if len(calls)==1 else 'c'*64
        return query(args)
    renamed=[]
    with pytest.raises(ValueError):gate.archive(root,code,rev,query=racing,proc=tmp_path/'proc',cgroups=tmp_path/'cgroups',rename=lambda *a:renamed.append(a))
    assert not renamed and base.exists()


def test_small_atomic_primitive_equivalence_to_existing_audited_rename():
    import ast
    old=ast.parse((REPO/'infra/failed_masks_transition.py').read_text())
    new=ast.parse((REPO/'infra/atomic_metadata.py').read_text())
    for tree in(old,new):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='rename_noreplace')
        assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='call'
            and isinstance(n.args[-1],ast.Constant)and n.args[-1].value==1 for n in ast.walk(node))
        assert not any(isinstance(n,ast.Attribute)and n.attr in('rename','replace','unlink')for n in ast.walk(node))
    assert 'no fallback'in ast.unparse(next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='rename_noreplace'))


def test_atomic_metadata_alias_duplicate_nonfinite_and_marker_guard(tmp_path):
    path=tmp_path/'metadata';write(path,b'original')
    assert gate.atomic.identity(path)['bytes']==8
    link=tmp_path/'alias';os.link(path,link)
    with pytest.raises(ValueError):gate.atomic.identity(path)
    for raw in(b'{"x":1,"x":2}',b'{"x":NaN}'):
        with pytest.raises(ValueError):gate.atomic._json(raw)
    with pytest.raises(ValueError):gate.atomic._pin({'bytes':True,'sha256':'a'*64})

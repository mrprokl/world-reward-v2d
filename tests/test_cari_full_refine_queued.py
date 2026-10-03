"""Tiny Linux/tool-spy scheduling fixtures; no Azure, model or GPU work."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_cari_full_refine_queued.sh'


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';revision='a'*40;episode='3';padded='000003'
    code=root/'jobs'/revision/'run_cari_full_refine_queued/code';(code/'infra').mkdir(parents=True)
    (code/'configs').mkdir();(root/'outputs'/('episode_'+padded)).mkdir(parents=True)
    (code.parent/'revision').write_text(revision+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    lock=root/'jobs/.world-reward-h100.lock';lock.write_text('original lock bytes')
    wrapper=code/'infra/run_cari_full_refine_queued.sh';wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)))
    (code/'infra/cari_full_refine.py').write_text('"""Procedural source only; not executed."""\n')
    child=code/'infra/run_cari_full_refine.sh'
    # The child is invoked by bash, so it spies on FD9 and exact own arguments
    # using a tiny Python process; no native optimizer or container is started.
    child.write_text('''#!/usr/bin/env bash
"$FAKE_PYTHON" - "$@" <<'PYCHILD'
import json,os,pathlib,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['child',*sys.argv[1:],os.fstat(9).st_ino])+'\\n')
code=pathlib.Path(os.environ['WR_CODE']);root=pathlib.Path(os.environ['WR_ROOT'])
fault=os.environ.get('FAKE_MUTATION','')
if fault=='source':
 p=code/'infra/cari_full_refine.py';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
elif fault=='marker':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='lock':
 p=root/'jobs/.world-reward-h100.lock';p.rename(p.with_name('retained-original'));p.write_text('foreign lock')
sys.exit(int(os.environ.get('FAKE_STATUS','0')))
PYCHILD
''')
    pins=[]
    for suffix in('input','shared_prepare','shared_forward'):
        path=code/'configs'/('cari_clip_'+padded+'_'+suffix+'_pins.json');path.write_text('{}\n');pins.append(path)
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'calls.jsonl'
    scripts={
        'uname':'#!/bin/sh\nprintf "%s\\n" Linux\n',
        'python3':f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        'systemctl':f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['systemctl',*sys.argv[1:]])+'\\n')
print(os.environ.get('FAKE_LOAD','loaded'));sys.exit(int(os.environ.get('FAKE_SYSTEM_STATUS','0')))
''',
        'flock':f'''#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['flock',*sys.argv[1:],os.fstat(9).st_ino])+'\\n')
if os.environ.get('FAKE_WAIT_TARGET')=='1':
 (pathlib.Path(os.environ['WR_ROOT'])/'outputs/episode_000003/cari_shared_refined_v1').mkdir()
sys.exit(int(os.environ.get('FAKE_FLOCK_STATUS','0')))
''',
        'nvidia-smi':f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['nvidia-smi',*sys.argv[1:]])+'\\n')
print(os.environ.get('FAKE_APPS',''));sys.exit(int(os.environ.get('FAKE_GPU_STATUS','0')))
'''}
    for name,source in scripts.items():path=bindir/name;path.write_text(source);path.chmod(0o755)
    env=dict(WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,FAKE_LOG=str(log),
        FAKE_PYTHON=sys.executable,PATH=str(bindir)+':'+os.environ['PATH'],HOME=str(tmp_path))
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=10)
    default=['--episode',episode,'--wait-for','world-reward-track1-episode4-volume-frontends-v1']
    def calls():return [json.loads(line)for line in log.read_text().splitlines()]if log.exists()else[]
    return dict(root=root,code=code,wrapper=wrapper,lock=lock,pins=pins,env=env,run=run,args=default,calls=calls)


@pytest.mark.parametrize('status',[0,7,124,137])
def test_queue_waits_on_lock_not_unit_cpu_completion_and_unchanged_child(runtime,status):
    runtime['env']['FAKE_STATUS']=str(status);before=runtime['lock'].read_bytes()
    result=runtime['run'](*runtime['args']);assert result.returncode==status,result.stderr
    calls=runtime['calls']();assert [row[0]for row in calls]==['systemctl','flock','nvidia-smi','child']
    assert calls[0]==['systemctl','show','world-reward-track1-episode4-volume-frontends-v1.service','--property=LoadState','--value']
    assert calls[1][:4]==['flock','--timeout','43200','9']
    assert calls[2]==['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']
    assert calls[3][1:]==['--episode','3',runtime['lock'].stat().st_ino]
    assert runtime['lock'].read_bytes()==before and not (runtime['root']/'outputs/episode_000003/cari_shared_refined_v1').exists()
    phases=[json.loads(line)['phase']for line in result.stdout.splitlines()]
    assert phases[:3]==['preflight','waiting_for_gpu_lock','running_native_child']
    assert phases[-1]==('child_complete'if status==0 else'fail')


@pytest.mark.parametrize('args',[[],['--episode','3'],['--wait-for','world-reward-test'],['--help'],
    ['--episode','03','--wait-for','world-reward-test'],['--episode','30','--wait-for','world-reward-test'],
    ['--episode','3','--episode','3','--wait-for','world-reward-test'],
    ['--episode','3','--wait-for','world-reward-test','--wait-for','world-reward-test'],
    ['--episode','3','--wait-for','world-reward-test;echo'],['--episode','3','--wait-for','../../unit'],
    ['--episode','3','--wait-for','world-reward-test','--budget','1']])
def test_exact_arguments_no_override_or_shell_injection(runtime,args):
    assert runtime['run'](*args).returncode==2 and not runtime['calls']()


@pytest.mark.parametrize('load',['not-found','masked','error',''])
def test_existing_independently_named_unit_required_before_lock(runtime,load):
    runtime['env']['FAKE_LOAD']=load
    assert runtime['run'](*runtime['args']).returncode!=0
    assert [row[0]for row in runtime['calls']()]==['systemctl']


@pytest.mark.parametrize('fault',['source','writable','symlink','marker','archive','missingpin','root','namespace','entrypoint'])
def test_immutable_source_root_and_own_pins_before_any_wait(runtime,fault,tmp_path):
    code=runtime['code'];path=code/'infra/cari_full_refine.py'
    if fault=='writable':path.chmod(0o644)
    elif fault=='source':code.chmod(0o755)
    elif fault=='marker':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archive':(code.parent/'source-sha256').write_text('bad')
    elif fault=='missingpin':runtime['pins'][-1].parent.chmod(0o755);runtime['pins'][-1].unlink()
    elif fault=='symlink':path.parent.chmod(0o755);path.unlink();path.symlink_to('missing')
    elif fault=='root':runtime['env']['WR_ROOT']=str(tmp_path)
    elif fault=='namespace':runtime['env']['WR_CODE']=str(tmp_path)
    else:
        outside=tmp_path/'copy.sh';outside.write_text(runtime['wrapper'].read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside),*runtime['args']],env=runtime['env'],capture_output=True,text=True,timeout=10)
        assert result.returncode!=0 and not runtime['calls']();return
    assert runtime['run'](*runtime['args']).returncode!=0 and not runtime['calls']()


@pytest.mark.parametrize('fault',['missing','symlink','hardlink','directory','gpu','gpuquery','waittimeout','targetduringwait'])
def test_gpu_and_lock_fail_closed_without_child(runtime,fault,tmp_path):
    lock=runtime['lock']
    if fault=='missing':lock.unlink()
    elif fault=='symlink':lock.unlink();lock.symlink_to('missing')
    elif fault=='hardlink':os.link(lock,tmp_path/'alias')
    elif fault=='directory':lock.unlink();lock.mkdir()
    elif fault=='gpu':runtime['env']['FAKE_APPS']='12345'
    elif fault=='gpuquery':runtime['env']['FAKE_GPU_STATUS']='9'
    elif fault=='waittimeout':runtime['env']['FAKE_FLOCK_STATUS']='1'
    else:runtime['env']['FAKE_WAIT_TARGET']='1'
    assert runtime['run'](*runtime['args']).returncode!=0
    assert not any(row[0]=='child'for row in runtime['calls']())


@pytest.mark.parametrize('mutation',['source','marker','lock'])
def test_source_and_lock_rechecked_after_child(runtime,mutation):
    runtime['env']['FAKE_MUTATION']=mutation
    assert runtime['run'](*runtime['args']).returncode!=0
    assert any(row[0]=='child'for row in runtime['calls']())
    if mutation=='lock':assert runtime['lock'].read_text()=='foreign lock'


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_existing_refinement_never_overwritten_or_deleted(runtime,kind):
    out=runtime['root']/'outputs/episode_000003/cari_shared_refined_v1'
    if kind=='directory':out.mkdir();(out/'retained').write_text('retained')
    elif kind=='file':out.write_text('retained')
    else:out.symlink_to('missing')
    assert runtime['run'](*runtime['args']).returncode!=0 and not runtime['calls']()
    assert out.exists()or out.is_symlink()


def test_static_source_closure_and_no_old_job_control(monkeypatch):
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text();assert 'flock --timeout 43200 9'in source
    assert not any(word in source for word in('systemctl stop','systemctl kill','--property=ActiveState','sleep 30','docker run','pkill','--lock-held','--gpus'))
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and'__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_cari_full_refine_queued.sh')
    assert {'infra/run_cari_full_refine_queued.sh','infra/run_cari_full_refine.sh','infra/cari_full_refine.py'}<=set(selected)

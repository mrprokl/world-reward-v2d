"""Tiny scheduling/tool-spy fixtures: no Azure, GPU, models or network."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_track1_frontends_queued.sh'
TARGETS=('automatic_masks','body_smoke','depth_smoke','scale_smoke','object_grounded',
         'body_full','depth_full','body_full/cari_adapter','object_pose_full','cari_inputs')


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';revision='a'*40
    code=root/'jobs'/revision/'run_track1_frontends_queued/code';(code/'infra').mkdir(parents=True)
    (root/'outputs').mkdir();base=root/'outputs/episode_000005'
    (code.parent/'revision').write_text(revision+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    lock=root/'jobs/.world-reward-h100.lock';lock.write_text('original lock bytes')
    wrapper=code/'infra/run_track1_frontends_queued.sh'
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)))
    for name in('run_episode_initializers.sh','run_automatic_masks.sh','run_object_pose_smoke.sh','run_cari_prepare.sh'):
        (code/'infra'/name).write_text('# Procedural immutable source; never executed.\n')
    child=code/'infra/run_track1_frontends.sh'
    child.write_text('''#!/usr/bin/env bash
"$FAKE_PYTHON" - "$@" <<'PYCHILD'
import json,os,pathlib,sys
try:os.fstat(9)
except OSError:closed=True
else:closed=False
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['child',*sys.argv[1:],closed])+'\\n')
code=pathlib.Path(os.environ['WR_CODE']);root=pathlib.Path(os.environ['WR_ROOT'])
fault=os.environ.get('FAKE_MUTATION','')
if fault=='source':
 p=code/'infra/run_episode_initializers.sh';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
elif fault=='marker':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='lock':
 p=root/'jobs/.world-reward-h100.lock';p.rename(p.with_name('retained-original'));p.write_text('foreign lock')
sys.exit(int(os.environ.get('FAKE_STATUS','0')))
PYCHILD
''')
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'calls.jsonl'
    scripts={
        'uname':'#!/bin/sh\nprintf "%s\\n" Linux\n',
        'python3':f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        'systemctl':f'''#!{sys.executable}
import json,os,pathlib,sys
log=pathlib.Path(os.environ['FAKE_LOG'])
prior=[json.loads(x)for x in log.read_text().splitlines()]if log.exists()else[]
with log.open('a')as h:h.write(json.dumps(['systemctl',*sys.argv[1:]])+'\\n')
if '--value'in sys.argv:print(os.environ.get('FAKE_LOAD','loaded'))
else:
 n=sum(row[0]=='systemctl'and'--value'not in row for row in prior)
 active=os.environ.get('FAKE_ACTIVE','inactive')
 if active=='transition':active='active'if n==0 else'inactive'
 output='LoadState='+os.environ.get('FAKE_LOAD','loaded')+'\\nActiveState='+active+'\\nResult='+os.environ.get('FAKE_RESULT','success')+'\\nExecMainStatus='+os.environ.get('FAKE_MAIN','0')
 print(os.environ.get('FAKE_DETAIL',output))
sys.exit(int(os.environ.get('FAKE_SYSTEM_STATUS','0')))
''',
        'flock':f'''#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['flock',*sys.argv[1:],os.fstat(9).st_ino])+'\\n')
root=pathlib.Path(os.environ['WR_ROOT'])
if '--timeout'in sys.argv:
 if os.environ.get('FAKE_WAIT_TARGET')=='1':(root/'outputs/episode_000005/automatic_masks').mkdir(parents=True)
 if os.environ.get('FAKE_WAIT_LOCK')=='1':
  p=root/'jobs/.world-reward-h100.lock';p.rename(p.with_name('retained-original'));p.write_text('foreign lock')
sys.exit(int(os.environ.get('FAKE_FLOCK_STATUS','0')))
''',
        'nvidia-smi':f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['nvidia-smi',*sys.argv[1:]])+'\\n')
print(os.environ.get('FAKE_APPS',''));sys.exit(int(os.environ.get('FAKE_GPU_STATUS','0')))
''',
        'sleep':f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['sleep',*sys.argv[1:]])+'\\n')
'''}
    for name,source in scripts.items():path=bindir/name;path.write_text(source);path.chmod(0o755)
    env=dict(WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,FAKE_LOG=str(log),
             FAKE_PYTHON=sys.executable,PATH=str(bindir)+':'+os.environ['PATH'],HOME=str(tmp_path))
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=10)
    def calls():return[json.loads(line)for line in log.read_text().splitlines()]if log.exists()else[]
    return dict(root=root,base=base,code=code,wrapper=wrapper,lock=lock,env=env,run=run,calls=calls,
                args=['--episode','5','--wait-for','world-reward-cari-episode3-refine-queued-v1'])


@pytest.mark.parametrize('status',[0,7,124,137])
def test_successful_predecessor_lock_then_unchanged_global_policy(runtime,status):
    runtime['env']['FAKE_STATUS']=str(status);before=runtime['lock'].read_bytes()
    result=runtime['run'](*runtime['args']);assert result.returncode==status,result.stderr
    calls=runtime['calls']();assert[row[0]for row in calls]==['systemctl','systemctl','flock','systemctl','nvidia-smi','flock','child']
    assert calls[0][1:]==['show','world-reward-cari-episode3-refine-queued-v1.service','--property=LoadState','--value']
    assert calls[2][1]=='--timeout'and 43195<=int(calls[2][2])<=43200 and calls[2][3]=='9'
    assert calls[5][1:3]==['--unlock','9']
    assert calls[6][1:]==['--episode','5','--actor-policy','fixed_all16',True]
    assert runtime['lock'].read_bytes()==before and not runtime['base'].exists()
    phases=[json.loads(line)['phase']for line in result.stdout.splitlines()]
    assert phases[:4]==['preflight','waiting_for_successful_predecessor','waiting_for_gpu_lock','running_original_frontends']
    assert phases[-1]==('child_complete'if status==0 else'fail')


def test_active_predecessor_waits_until_success_before_gpu_or_lock(runtime):
    runtime['env']['FAKE_ACTIVE']='transition'
    result=runtime['run'](*runtime['args']);assert result.returncode==0,result.stderr
    assert[row[0]for row in runtime['calls']()][:5]==['systemctl','systemctl','sleep','systemctl','flock']


@pytest.mark.parametrize('args',[[],['--episode','5'],['--wait-for','world-reward-test'],['--help'],
    ['--episode','05','--wait-for','world-reward-test'],['--episode','30','--wait-for','world-reward-test'],
    ['--episode','5','--episode','5','--wait-for','world-reward-test'],
    ['--episode','5','--wait-for','world-reward-test','--wait-for','world-reward-test'],
    ['--episode','5','--wait-for','world-reward-test;echo'],['--episode','5','--wait-for','../../unit'],
    ['--episode','5','--wait-for','world-reward-test','--actor-policy','default_three']])
def test_exact_arguments_no_policy_override(runtime,args):
    assert runtime['run'](*args).returncode==2 and not runtime['calls']()


@pytest.mark.parametrize('settings',[dict(FAKE_LOAD='not-found'),dict(FAKE_LOAD='masked'),dict(FAKE_LOAD=''),
    dict(FAKE_ACTIVE='failed'),dict(FAKE_ACTIVE='inactive',FAKE_RESULT='exit-code',FAKE_MAIN='1'),
    dict(FAKE_MAIN='1'),dict(FAKE_RESULT='timeout'),dict(FAKE_SYSTEM_STATUS='9'),
    dict(FAKE_DETAIL='LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=0\nResult=success'),
    dict(FAKE_DETAIL='LoadState=loaded\nActiveState=inactive'),dict(FAKE_MAIN='garbage')])
def test_missing_failed_or_malformed_predecessor_never_runs_gpu(runtime,settings):
    runtime['env'].update(settings)
    assert runtime['run'](*runtime['args']).returncode!=0
    assert all(row[0]=='systemctl'for row in runtime['calls']())


@pytest.mark.parametrize('fault',['writable','directory','symlink','marker','archive','missingchild','root','namespace','entrypoint'])
def test_source_and_namespace_preflight_fail_before_wait(runtime,fault,tmp_path):
    code=runtime['code'];path=code/'infra/run_episode_initializers.sh'
    if fault=='writable':path.chmod(0o644)
    elif fault=='directory':code.chmod(0o755)
    elif fault=='marker':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archive':(code.parent/'source-sha256').write_text('bad')
    elif fault in('symlink','missingchild'):
        path.parent.chmod(0o755);path.unlink()
        if fault=='symlink':path.symlink_to('missing')
    elif fault=='root':runtime['env']['WR_ROOT']=str(tmp_path)
    elif fault=='namespace':runtime['env']['WR_CODE']=str(tmp_path)
    else:
        outside=tmp_path/'copy.sh';outside.write_text(runtime['wrapper'].read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside),*runtime['args']],env=runtime['env'],capture_output=True,text=True,timeout=10)
        assert result.returncode!=0 and not runtime['calls']();return
    assert runtime['run'](*runtime['args']).returncode!=0 and not runtime['calls']()


@pytest.mark.parametrize('fault',['missing','symlink','hardlink','directory','gpu','gpuquery','locktimeout','targetduringwait','lockduringwait'])
def test_lock_gpu_and_wait_fail_closed_without_child(runtime,fault,tmp_path):
    lock=runtime['lock']
    if fault=='missing':lock.unlink()
    elif fault=='symlink':lock.unlink();lock.symlink_to('missing')
    elif fault=='hardlink':os.link(lock,tmp_path/'alias')
    elif fault=='directory':lock.unlink();lock.mkdir()
    elif fault=='gpu':runtime['env']['FAKE_APPS']='12345'
    elif fault=='gpuquery':runtime['env']['FAKE_GPU_STATUS']='9'
    elif fault=='locktimeout':runtime['env']['FAKE_FLOCK_STATUS']='1'
    elif fault=='targetduringwait':runtime['env']['FAKE_WAIT_TARGET']='1'
    else:runtime['env']['FAKE_WAIT_LOCK']='1'
    assert runtime['run'](*runtime['args']).returncode!=0
    assert not any(row[0]=='child'for row in runtime['calls']())


@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_any_existing_target_is_retained_before_wait(runtime,target,kind):
    out=runtime['base']/target;out.parent.mkdir(parents=True,exist_ok=True)
    if kind=='directory':out.mkdir();(out/'retained').write_text('retained')
    elif kind=='file':out.write_text('retained')
    else:out.symlink_to('missing')
    assert runtime['run'](*runtime['args']).returncode!=0 and not runtime['calls']()
    assert out.exists()or out.is_symlink()


@pytest.mark.parametrize('mutation',['source','marker','lock'])
def test_source_and_original_lock_rechecked_after_child(runtime,mutation):
    runtime['env']['FAKE_MUTATION']=mutation
    assert runtime['run'](*runtime['args']).returncode!=0
    assert any(row[0]=='child'for row in runtime['calls']())


def test_total_wait_budget_is_shared_before_gpu(runtime):
    wrapper=runtime['wrapper'];wrapper.chmod(0o644)
    wrapper.write_text(wrapper.read_text().replace('43200','0'));wrapper.chmod(0o444)
    runtime['env']['FAKE_ACTIVE']='active'
    assert runtime['run'](*runtime['args']).returncode!=0
    assert all(row[0]=='systemctl'for row in runtime['calls']())


def test_complete_static_closure_no_native_or_network_work(monkeypatch):
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text()
    assert 'exec 9<"$LOCK"'in source and 'flock --timeout "$REMAINING" 9'in source
    assert '--actor-policy fixed_all16'in source
    assert not any(x in source for x in('systemctl stop','systemctl kill','docker run','--gpus','--lock-held','curl ','ssh ','pkill'))
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and'__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_track1_frontends_queued.sh'))
    child=set(azure_job.runtime_bundle_paths(files,'infra/run_track1_frontends.sh'))
    assert child<=selected
    assert {'infra/run_track1_frontends_queued.sh','infra/object_pose_smoke.py','src/world_reward/rigid_alignment.py'}<=selected


def terminal_args(runtime):
    return [*runtime['args'][:2],'--after-terminal',runtime['args'][3]]


@pytest.mark.parametrize('active,result,status',[
    ('failed','exit-code','1'),('failed','exit-code','124'),('failed','signal','15'),
    ('failed','core-dump','11'),('failed','timeout','0'),('failed','oom-kill','9'),
    ('inactive','success','0'),
])
def test_terminal_policy_starts_distinct_successor_and_records_failure_not_reclassified(runtime,active,result,status):
    runtime['env'].update(FAKE_ACTIVE=active,FAKE_RESULT=result,FAKE_MAIN=status)
    before=runtime['lock'].read_bytes();completed=runtime['run'](*terminal_args(runtime))
    assert completed.returncode==0,completed.stderr
    calls=runtime['calls']();assert calls[-1][1:]==['--episode','5','--actor-policy','fixed_all16',True]
    assert [row[0]for row in calls]==['systemctl','systemctl','flock','systemctl','nvidia-smi','flock','child']
    phases=[json.loads(line)for line in completed.stdout.splitlines()]
    summary=next(row for row in phases if row['phase']=='predecessor_terminal')
    assert summary['predecessor_state']==active and summary['predecessor_result']==result
    assert summary['exec_main_status']==int(status) and summary['predecessor_unit']==runtime['args'][3]+'.service'
    assert phases[1]['phase']=='waiting_for_terminal_predecessor'
    assert runtime['lock'].read_bytes()==before and not runtime['base'].exists()


@pytest.mark.parametrize('tail',[
    ['--after-terminal'],['--after-terminal','world-reward-test;echo'],
    ['--after-terminal','../../unit'],['--after-terminal','world-reward-a','--after-terminal','world-reward-b'],
    ['--wait-for','world-reward-a','--after-terminal','world-reward-b'],
    ['--after-terminal','world-reward-a','--wait-for','world-reward-b'],
])
def test_terminal_arguments_mutually_exclusive_once_and_canonical(runtime,tail):
    assert runtime['run']('--episode','5',*tail).returncode==2 and not runtime['calls']()


@pytest.mark.parametrize('settings',[
    dict(FAKE_LOAD='not-found'),dict(FAKE_LOAD='masked'),dict(FAKE_ACTIVE='unknown'),
    dict(FAKE_ACTIVE='failed',FAKE_RESULT='success'),
    dict(FAKE_ACTIVE='failed',FAKE_RESULT='exit-code',FAKE_MAIN='0'),
    dict(FAKE_ACTIVE='failed',FAKE_RESULT='foreign',FAKE_MAIN='1'),
    dict(FAKE_ACTIVE='inactive',FAKE_RESULT='exit-code',FAKE_MAIN='1'),
    dict(FAKE_MAIN='-1'),dict(FAKE_MAIN='256'),dict(FAKE_MAIN='00'),dict(FAKE_MAIN='garbage'),
    dict(FAKE_DETAIL='LoadState=loaded\nActiveState=failed\nResult=exit-code\nExecMainStatus=1\nResult=exit-code'),
])
def test_terminal_unknown_inconsistent_or_missing_never_runs_child(runtime,settings):
    runtime['env'].update(settings)
    assert runtime['run'](*terminal_args(runtime)).returncode!=0
    assert all(row[0]=='systemctl'for row in runtime['calls']())


def test_terminal_active_waits_and_post_lock_checks_original_source_targets_gpu(runtime):
    runtime['env']['FAKE_ACTIVE']='transition'
    result=runtime['run'](*terminal_args(runtime));assert result.returncode==0,result.stderr
    assert [row[0]for row in runtime['calls']()][:5]==['systemctl','systemctl','sleep','systemctl','flock']


@pytest.mark.parametrize('fault',['gpu','target','lock','source'])
def test_terminal_mode_does_not_bypass_existing_guards(runtime,fault):
    runtime['env'].update(FAKE_ACTIVE='failed',FAKE_RESULT='exit-code',FAKE_MAIN='1')
    if fault=='gpu':runtime['env']['FAKE_APPS']='foreign'
    elif fault=='target':runtime['env']['FAKE_WAIT_TARGET']='1'
    elif fault=='lock':runtime['env']['FAKE_WAIT_LOCK']='1'
    else:(runtime['code']/'infra/run_episode_initializers.sh').chmod(0o644)
    assert runtime['run'](*terminal_args(runtime)).returncode!=0
    assert not any(row[0]=='child'for row in runtime['calls']())


def test_terminal_never_accepts_reloading_as_ready(runtime):
    wrapper=runtime['wrapper'];wrapper.chmod(0o644)
    wrapper.write_text(wrapper.read_text().replace('43200','0'));wrapper.chmod(0o444)
    runtime['env']['FAKE_ACTIVE']='reloading'
    result=runtime['run'](*terminal_args(runtime));assert result.returncode!=0
    assert all(row[0]=='systemctl'for row in runtime['calls']())


@pytest.mark.parametrize('status',[7,124,137])
def test_terminal_predecessor_does_not_hide_or_retry_successor_failure(runtime,status):
    runtime['env'].update(FAKE_ACTIVE='failed',FAKE_RESULT='exit-code',FAKE_MAIN='1',FAKE_STATUS=str(status))
    result=runtime['run'](*terminal_args(runtime));assert result.returncode==status,result.stderr
    assert sum(row[0]=='child'for row in runtime['calls']())==1
    phases=[json.loads(line)['phase']for line in result.stdout.splitlines()]
    assert phases[-1]=='fail'and'child_complete'not in phases


def lock_args(runtime):
    return [*runtime['args'][:2],'--after-gpu-lock']


@pytest.mark.parametrize('status',[0,7,124,137])
def test_explicit_gpu_lock_only_is_independent_of_collected_units_and_preserves_child(runtime,status):
    runtime['env'].update(FAKE_LOAD='not-found',FAKE_SYSTEM_STATUS='9',FAKE_STATUS=str(status))
    before=runtime['lock'].read_bytes();result=runtime['run'](*lock_args(runtime))
    assert result.returncode==status,result.stderr
    calls=runtime['calls']();assert [row[0]for row in calls]==['flock','nvidia-smi','flock','child']
    assert calls[0][1]=='--timeout'and 43195<=int(calls[0][2])<=43200 and calls[0][3]=='9'
    assert calls[-1][1:]==['--episode','5','--actor-policy','fixed_all16',True]
    assert runtime['lock'].read_bytes()==before and not runtime['base'].exists()
    phases=[json.loads(line)for line in result.stdout.splitlines()]
    assert [row['phase']for row in phases[:3]]==['preflight','waiting_for_gpu_lock','running_original_frontends']
    assert not any('predecessor_unit'in row or row['phase']=='predecessor_terminal'for row in phases)
    assert phases[-1]['phase']==('child_complete'if status==0 else'fail')


@pytest.mark.parametrize('tail',[
    ['--after-gpu-lock',''],['--after-gpu-lock','world-reward-any'],
    ['--after-gpu-lock','--after-gpu-lock'],
    ['--after-gpu-lock','--wait-for','world-reward-test'],
    ['--after-gpu-lock','--after-terminal','world-reward-test'],
    ['--wait-for','world-reward-test','--after-gpu-lock'],
    ['--after-terminal','world-reward-test','--after-gpu-lock'],
    ['--after-gpu-lock=true'],['--after-gpu-lock','--actor-policy','default_three'],
    ['--after-gpu-lock','--oracle','true'],
])
def test_lock_only_flag_has_no_value_and_cannot_mix_policies_or_override_child(runtime,tail):
    result=runtime['run']('--episode','5',*tail)
    assert result.returncode==2 and not runtime['calls']()


@pytest.mark.parametrize('fault',['missing','alias','hardlink','locktimeout','gpu','query','wait_target','wait_lock','source','occupied'])
def test_gpu_lock_only_retains_all_existing_source_target_and_lock_guards(runtime,fault,tmp_path):
    if fault=='missing':runtime['lock'].unlink()
    elif fault=='alias':
        original=runtime['lock'].with_name('retained');runtime['lock'].rename(original);runtime['lock'].symlink_to(original)
    elif fault=='hardlink':os.link(runtime['lock'],tmp_path/'alias')
    elif fault=='locktimeout':runtime['env']['FAKE_FLOCK_STATUS']='1'
    elif fault=='gpu':runtime['env']['FAKE_APPS']='12345'
    elif fault=='query':runtime['env']['FAKE_GPU_STATUS']='9'
    elif fault=='wait_target':runtime['env']['FAKE_WAIT_TARGET']='1'
    elif fault=='wait_lock':runtime['env']['FAKE_WAIT_LOCK']='1'
    elif fault=='source':(runtime['code']/'infra/run_episode_initializers.sh').chmod(0o644)
    else:
        occupied=runtime['base']/'object_pose_full';occupied.mkdir(parents=True);(occupied/'retained').write_text('preserve')
    assert runtime['run'](*lock_args(runtime)).returncode!=0
    assert not any(row[0]in('systemctl','child')for row in runtime['calls']())
    if fault=='occupied':assert (occupied/'retained').read_text()=='preserve'


def test_gpu_lock_only_uses_same_bounded_wait_and_never_marks_unknown_unit_pass(runtime):
    wrapper=runtime['wrapper'];wrapper.chmod(0o644)
    wrapper.write_text(wrapper.read_text().replace('43200','0'));wrapper.chmod(0o444)
    assert runtime['run'](*lock_args(runtime)).returncode!=0 and not runtime['calls']()

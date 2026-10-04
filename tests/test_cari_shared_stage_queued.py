"""Tiny synchronous scheduling controls; no Azure, model, media or GPU work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'infra/run_cari_shared_stage_queued.sh'
CHILDREN = {
    'prepare': ('run_cari_shared_prepare.sh', 'cari_shared_prepare.py'),
    'forward': ('run_cari_full_forward.sh', 'cari_full_forward.py'),
    'refined': ('run_cari_full_refine.sh', 'cari_full_refine.py'),
    'export': ('run_cari_full_export.sh', 'cari_full_export.py'),
}


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / 'runtime'; revision = 'a' * 40
    code = root / 'jobs' / revision / 'run_cari_shared_stage_queued/code'
    (code / 'infra').mkdir(parents=True); (code / 'configs').mkdir()
    (root / 'outputs/episode_000005').mkdir(parents=True)
    (code.parent / 'revision').write_text(revision + '\n')
    (code.parent / 'source-sha256').write_text('b' * 64 + '\n')
    lock = root / 'jobs/.world-reward-h100.lock'; lock.write_text('unchanged lock bytes')
    wrapper = code / 'infra/run_cari_shared_stage_queued.sh'
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward', str(root)))
    for stage, (child, driver) in CHILDREN.items():
        (code / 'infra' / driver).write_text('"""Procedural source, never imported."""\n')
        (code / 'infra' / child).write_text(r'''#!/usr/bin/env bash
"$FAKE_PYTHON" - "$0" "$@" <<'PYCHILD'
import json,os,pathlib,sys
root=pathlib.Path(os.environ['WR_ROOT']);code=pathlib.Path(os.environ['WR_CODE'])
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['child',pathlib.Path(sys.argv[1]).name,*sys.argv[2:],os.fstat(9).st_ino])+'\n')
if os.environ.get('FAKE_CHECK_REAL_LOCK')=='1':
 import fcntl
 with (root/'jobs/.world-reward-h100.lock').open('r')as stream:
  try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Parent did not retain the original lock')
fault=os.environ.get('FAKE_MUTATION','')
if fault=='source':
 p=code/'infra/cari_shared_prepare.py';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
elif fault=='marker':(code.parent/'revision').write_text('c'*40+'\n')
elif fault=='pin':
 p=code/'configs/cari_clip_000005_input_pins.json';p.chmod(0o644);p.write_text('{}');p.chmod(0o444)
elif fault=='lock':
 p=root/'jobs/.world-reward-h100.lock';p.rename(p.with_name('retained-original'));p.write_text('foreign lock')
sys.exit(int(os.environ.get('FAKE_STATUS','0')))
PYCHILD
''')
    spec = dict(episode_index=5, total_frames=668, camera_name='front_stereo_camera_left', height=1152, width=1536)
    receipt = dict(sha256='c'*64, bytes=20, producer_revision='d'*40, script_sha256='e'*64)
    pins = {}
    for role in ('input', 'prepare', 'forward', 'refined'):
        suffix = 'input' if role == 'input' else 'shared_' + role
        path = code / f'configs/cari_clip_000005_{suffix}_pins.json'
        record = dict(schema='world-reward-cari-clip-input-pins-v1' if role == 'input' else f'world-reward-cari-shared-{role}-pins-v1', clip_spec=spec)
        record.update(dict(input_report=receipt, source_files={}) if role == 'input' else {role: receipt, role+'_files': {}})
        path.write_text(json.dumps(record)); pins[role] = path
    # Genuine dispatches can contain an empty package initializer.
    (code / 'empty_init.py').write_bytes(b'')
    for path in (code, *code.rglob('*')):
        path.chmod(0o555 if path.is_dir() else 0o444)
    bindir = tmp_path / 'bin'; bindir.mkdir(); log = tmp_path / 'calls.jsonl'
    spies = {
        'uname': '#!/bin/sh\nprintf "%s\\n" Linux\n',
        'python3': f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        'systemctl': f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['systemctl',*sys.argv[1:]])+'\\n')
print(os.environ.get('FAKE_LOAD','loaded'));sys.exit(int(os.environ.get('FAKE_SYSTEM_STATUS','0')))
''',
        'flock': f'''#!{sys.executable}
import json,os,pathlib,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['flock',*sys.argv[1:],os.fstat(9).st_ino])+'\\n')
root=pathlib.Path(os.environ['WR_ROOT'])
if os.environ.get('FAKE_WAIT_TARGET')=='1':(root/'outputs/episode_000005'/('cari_shared_'+os.environ['FAKE_STAGE']+'_v1')).mkdir()
if os.environ.get('FAKE_WAIT_SOURCE')=='1':
 p=pathlib.Path(os.environ['WR_CODE'])/'infra/cari_shared_prepare.py';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
if os.environ.get('FAKE_REAL_FLOCK'):os.execv(os.environ['FAKE_REAL_FLOCK'],[os.environ['FAKE_REAL_FLOCK'],*sys.argv[1:]])
sys.exit(int(os.environ.get('FAKE_FLOCK_STATUS','0')))
''',
        'nvidia-smi': f'''#!{sys.executable}
import json,os,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['nvidia-smi',*sys.argv[1:]])+'\\n')
print(os.environ.get('FAKE_APPS',''));sys.exit(int(os.environ.get('FAKE_GPU_STATUS','0')))
''',
    }
    for name, source in spies.items():
        path = bindir / name; path.write_text(source); path.chmod(0o755)
    # Explicit nonsecret environment, never copy credentials into fixture repr.
    env = dict(WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION=revision,
               FAKE_LOG=str(log), FAKE_PYTHON=sys.executable,
               PATH=str(bindir)+':'+os.environ['PATH'], HOME=str(tmp_path))
    args = ['--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-track1-episode6-frontends-v1']
    def run(*arguments):
        if '--stage' in arguments:env['FAKE_STAGE'] = arguments[arguments.index('--stage')+1] if arguments.index('--stage')+1<len(arguments) else ''
        return subprocess.run(['rtk', 'proxy', 'bash', str(wrapper), *arguments], env=env, capture_output=True, text=True, timeout=10)
    def calls():return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return dict(root=root, code=code, wrapper=wrapper, lock=lock, pins=pins, env=env, args=args, run=run, calls=calls)


@pytest.mark.parametrize('stage', CHILDREN)
@pytest.mark.parametrize('status', [0, 7, 124])
def test_synchronous_unchanged_child_under_existing_lock(runtime, stage, status):
    args = runtime['args'][:]; args[1] = stage; runtime['env']['FAKE_STATUS'] = str(status)
    before = runtime['lock'].read_bytes(); result = runtime['run'](*args)
    assert result.returncode == status, result.stderr
    calls = runtime['calls'](); assert [row[0] for row in calls] == ['systemctl', 'flock', 'nvidia-smi', 'child']
    assert calls[0] == ['systemctl', 'show', 'world-reward-track1-episode6-frontends-v1.service', '--property=LoadState', '--value']
    assert calls[1] == ['flock', '--timeout', '43200', '9', runtime['lock'].stat().st_ino]
    assert calls[2] == ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    assert calls[3] == ['child', CHILDREN[stage][0], '--episode', '5', runtime['lock'].stat().st_ino]
    phases = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row['phase'] for row in phases] == ['preflight', 'waiting_for_gpu_lock', 'running_native_child', 'child_complete' if status == 0 else 'fail']
    assert all(row['native_stage'] == stage for row in phases)
    assert runtime['lock'].read_bytes() == before
    assert not (runtime['root'] / f'outputs/episode_000005/cari_shared_{stage}_v1').exists()


@pytest.mark.parametrize('args', [[], ['--help'], ['--stage', 'prepare'],
    ['--stage', 'other', '--episode', '5', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '05', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '30', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '5', '--episode', '5', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-test', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-test;echo'],
    ['--stage', 'prepare', '--episode', '5', '--wait-for', '../../unit'],
    ['--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-test', '--budget', '1']])
def test_arguments_fail_before_any_tool(runtime, args):
    assert runtime['run'](*args).returncode == 2 and not runtime['calls']()


@pytest.mark.parametrize('stage', CHILDREN)
def test_only_required_preceding_role_pins_needed(runtime, stage):
    index = list(CHILDREN).index(stage)
    (runtime['code'] / 'configs').chmod(0o755)
    for role in list(runtime['pins'])[index+1:]:runtime['pins'][role].unlink()
    (runtime['code'] / 'configs').chmod(0o555)
    args = runtime['args'][:]; args[1] = stage
    result = runtime['run'](*args); assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('fault', ['missingpin', 'pinschema', 'pinspec', 'pinproducer', 'duplicatepin', 'writable', 'symlink', 'hardlink', 'marker', 'archive', 'namespace', 'root', 'entry'])
def test_source_and_pins_fail_before_wait(runtime, fault, tmp_path):
    path = runtime['pins']['input']; code = runtime['code']
    if fault == 'missingpin':path.parent.chmod(0o755); path.unlink()
    elif fault.startswith('pin'):
        record = json.loads(path.read_text()); path.chmod(0o644)
        if fault == 'pinschema':record['schema'] = 'other'
        elif fault == 'pinspec':record['clip_spec']['episode_index'] = 3
        else:record['input_report']['producer_revision'] = 'invalid'
        path.write_text(json.dumps(record)); path.chmod(0o444)
    elif fault == 'duplicatepin':
        path.chmod(0o644); path.write_text('{"schema":"a","schema":"b"}'); path.chmod(0o444)
    elif fault == 'writable':(code / 'infra/cari_shared_prepare.py').chmod(0o644)
    elif fault == 'symlink':
        path.parent.chmod(0o755); path.unlink(); path.symlink_to('missing')
    elif fault == 'hardlink':os.link(path, tmp_path / 'alias')
    elif fault == 'marker':(code.parent / 'revision').write_text('c'*40+'\n')
    elif fault == 'archive':(code.parent / 'source-sha256').write_text('invalid')
    elif fault == 'namespace':runtime['env']['WR_CODE'] = str(tmp_path)
    elif fault == 'root':runtime['env']['WR_ROOT'] = str(tmp_path)
    else:
        outside = tmp_path / 'copy.sh'; outside.write_text(runtime['wrapper'].read_text())
        result = subprocess.run(['rtk', 'proxy', 'bash', str(outside), *runtime['args']], env=runtime['env'], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and not runtime['calls'](); return
    assert runtime['run'](*runtime['args']).returncode != 0 and not runtime['calls']()


@pytest.mark.parametrize('load', ['not-found', 'masked', 'error', ''])
def test_loaded_explicit_predecessor_required(runtime, load):
    runtime['env']['FAKE_LOAD'] = load
    assert runtime['run'](*runtime['args']).returncode != 0
    assert [row[0] for row in runtime['calls']()] == ['systemctl']


@pytest.mark.parametrize('stage', CHILDREN)
def test_explicit_idle_mode_requires_original_lock_without_invented_unit_success(runtime, stage):
    args = ['--stage', stage, '--episode', '5', '--when-idle']
    result = runtime['run'](*args)
    assert result.returncode == 0, result.stderr
    calls = runtime['calls']()
    assert [row[0] for row in calls] == ['flock', 'nvidia-smi', 'child']
    assert calls[0] == ['flock', '--nonblock', '9', runtime['lock'].stat().st_ino]
    assert calls[-1][1:4] == [CHILDREN[stage][0], '--episode', '5']


@pytest.mark.parametrize('args', [
    ['--stage', 'prepare', '--episode', '5', '--when-idle', '--when-idle'],
    ['--stage', 'prepare', '--episode', '5', '--when-idle', '--wait-for', 'world-reward-test'],
    ['--stage', 'prepare', '--episode', '5', '--wait-for', 'world-reward-test', '--when-idle'],
    ['--stage', 'prepare', '--episode', '5', '--when-idle', 'true'],
])
def test_idle_mode_is_explicit_and_exclusive_before_any_runtime_query(runtime, args):
    assert runtime['run'](*args).returncode == 2 and not runtime['calls']()


@pytest.mark.parametrize('fault', ['busy_lock', 'gpu', 'gpuquery', 'target'])
def test_idle_mode_aborts_before_child_on_contention_or_foreign_target(runtime, fault):
    if fault == 'busy_lock':runtime['env']['FAKE_FLOCK_STATUS'] = '1'
    elif fault == 'gpu':runtime['env']['FAKE_APPS'] = '12345'
    elif fault == 'gpuquery':runtime['env']['FAKE_GPU_STATUS'] = '9'
    else:(runtime['root'] / 'outputs/episode_000005/cari_shared_prepare_v1').mkdir()
    result = runtime['run']('--stage', 'prepare', '--episode', '5', '--when-idle')
    assert result.returncode != 0 and not any(row[0] == 'child' for row in runtime['calls']())


@pytest.mark.parametrize('fault', ['missing', 'symlink', 'hardlink', 'directory', 'gpu', 'gpuquery', 'timeout', 'targetduringwait', 'sourceduringwait'])
def test_lock_and_gpu_fail_closed(runtime, fault, tmp_path):
    lock = runtime['lock']
    if fault == 'missing':lock.unlink()
    elif fault == 'symlink':lock.unlink(); lock.symlink_to('missing')
    elif fault == 'hardlink':os.link(lock, tmp_path / 'alias')
    elif fault == 'directory':lock.unlink(); lock.mkdir()
    elif fault == 'gpu':runtime['env']['FAKE_APPS'] = '12345'
    elif fault == 'gpuquery':runtime['env']['FAKE_GPU_STATUS'] = '9'
    elif fault == 'timeout':runtime['env']['FAKE_FLOCK_STATUS'] = '1'
    elif fault == 'targetduringwait':runtime['env']['FAKE_WAIT_TARGET'] = '1'
    else:runtime['env']['FAKE_WAIT_SOURCE'] = '1'
    assert runtime['run'](*runtime['args']).returncode != 0
    assert not any(row[0] == 'child' for row in runtime['calls']())


@pytest.mark.parametrize('mutation', ['source', 'marker', 'pin', 'lock'])
def test_original_source_and_lock_rechecked_after_child(runtime, mutation):
    runtime['env']['FAKE_MUTATION'] = mutation
    assert runtime['run'](*runtime['args']).returncode != 0
    assert any(row[0] == 'child' for row in runtime['calls']())


@pytest.mark.parametrize('kind', ['directory', 'file', 'broken_symlink'])
def test_target_never_overwritten_or_deleted(runtime, kind):
    out = runtime['root'] / 'outputs/episode_000005/cari_shared_prepare_v1'
    if kind == 'directory':out.mkdir(); (out / 'retained').write_text('retained')
    elif kind == 'file':out.write_text('retained')
    else:out.symlink_to('missing')
    assert runtime['run'](*runtime['args']).returncode != 0 and not runtime['calls']()
    assert out.exists() or out.is_symlink()


@pytest.mark.skipif(sys.platform != 'linux' or not shutil.which('flock'), reason='Original Linux flock only')
def test_real_linux_parent_retains_lock(runtime):
    runtime['env']['FAKE_REAL_FLOCK'] = shutil.which('flock')
    runtime['env']['FAKE_CHECK_REAL_LOCK'] = '1'
    result = runtime['run'](*runtime['args']); assert result.returncode == 0, result.stderr


def test_bash_syntax_and_complete_literal_runtime_closure(monkeypatch):
    subprocess.run(['rtk', 'proxy', 'bash', '-n', str(WRAPPER)], check=True)
    source = WRAPPER.read_text(); assert len(source.splitlines()) <= 180
    assert 'flock --timeout 43200 9' in source and 'bash "$CODE/$CHILD" --episode "$EPISODE"' in source
    assert not any(word in source for word in ('systemctl stop', 'systemctl kill', 'docker run', 'pkill', '--gpus', '--actor-policy', 'flock --unlock', 'eval ', 'mkdir ', '--property=ActiveState'))
    monkeypatch.syspath_prepend(str(ROOT / 'infra')); import azure_job
    files = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ('infra', 'src', 'configs') for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml'] = (ROOT / 'pyproject.toml').read_bytes()
    selected = azure_job.runtime_bundle_paths(files, 'infra/run_cari_shared_stage_queued.sh')
    assert {f'infra/{name}' for names in CHILDREN.values() for name in names} | {'infra/run_cari_shared_stage_queued.sh'} <= set(selected)

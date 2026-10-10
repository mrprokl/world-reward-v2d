from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import terminal_success as wait

UNIT = 'world-reward-form-native-sealed-recovery-1010.service'


def state(**changes):
    return dict(LoadState='loaded', ActiveState='inactive', SubState='dead',
        Result='success', ExecMainStatus='0', MainPID='0', **{}) | changes


@pytest.mark.parametrize('changes', [dict(LoadState='not-found'), dict(ActiveState='failed'),
    dict(Result='exit-code', ExecMainStatus='1'), dict(MainPID='100'), dict(SubState='running'),
    dict(ExecMainStatus='256'), dict(ExecMainStatus='00'), dict(MainPID='-1')])
def test_missing_failed_or_live_not_terminal_success(changes):
    with pytest.raises(ValueError):
        wait.unit_ready(state(**changes))


@pytest.mark.parametrize('active', ['active', 'activating', 'deactivating', 'reloading'])
def test_running_waits_for_entire_unit(active):
    assert not wait.unit_ready(state(ActiveState=active, SubState='running', MainPID='123'))


def test_loaded_success_no_live_main_process():
    assert wait.unit_ready(state())
    with pytest.raises(ValueError):
        wait.unit_ready(state() | {'ExecMainPID': '123'})


@pytest.mark.parametrize('unit', ['foreign.service', '--help', 'world-reward-a;echo',
    'world-reward-a.service.service', 'world-reward-a/b', 'world-reward-', 'world-reward-a\n'])
def test_exact_owned_unit_not_shell_fragment(unit):
    with pytest.raises(ValueError):
        wait.owned_unit(unit)


def test_snapshot_exact_props_command_binding_and_no_command_leak(monkeypatch):
    expected = ('WR_CODE=/srv/code', '--dev-revision '+'a'*40)
    command = '{ path=/usr/bin/env ; argv[]=/usr/bin/env WR_CODE=/srv/code /bin/bash /srv/run --dev-revision '+'a'*40+' ; }'
    raw = '\n'.join(f'{key}={value}' for key, value in (state() | {'ExecStart': command}).items()).encode()
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs)); return SimpleNamespace(returncode=0, stdout=raw, stderr=b'not displayed')
    monkeypatch.setattr(wait.subprocess, 'run', run)
    assert wait.snapshot(UNIT, expected) == state()
    assert calls[0][0][:3] == ['/usr/bin/systemctl', 'show', UNIT]
    assert calls[0][1]['timeout'] == 15 and calls[0][1]['capture_output']
    with pytest.raises(ValueError, match='exact original'):
        wait.snapshot(UNIT, ('WR_CODE=/srv/code-suffix',))


@pytest.mark.parametrize('raw', [b'', b'LoadState=loaded\nLoadState=loaded', b'other=value', b'x'*65537])
def test_missing_duplicate_giant_or_foreign_properties_rejected(monkeypatch, raw):
    monkeypatch.setattr(wait.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=raw))
    with pytest.raises(ValueError):
        wait.snapshot(UNIT)


def test_wait_bounded_no_restart_or_gpu_lease(monkeypatch):
    snapshots = iter([state(ActiveState='active', SubState='running', MainPID='123'), state()])
    monkeypatch.setattr(wait, 'snapshot', lambda *a: next(snapshots))
    monkeypatch.setattr(wait.time, 'monotonic', lambda: 0.)
    slept = []; monkeypatch.setattr(wait.time, 'sleep', slept.append)
    assert wait.wait_success_unit(UNIT, timeout=30)['MainPID'] == '0'
    assert slept == [30]
    monkeypatch.setattr(wait, 'snapshot', lambda *a: state(ActiveState='active', MainPID='123'))
    times = iter([0., 31.]); monkeypatch.setattr(wait.time, 'monotonic', lambda: next(times))
    with pytest.raises(TimeoutError):
        wait.wait_success_unit(UNIT, timeout=30)
    assert slept == [30]


def test_failed_immediate_never_sleeps(monkeypatch):
    monkeypatch.setattr(wait, 'snapshot', lambda *a: state(ActiveState='failed', Result='exit-code', ExecMainStatus='1'))
    monkeypatch.setattr(wait.time, 'sleep', lambda _: pytest.fail('Failed predecessor must not wait'))
    with pytest.raises(ValueError):
        wait.wait_success_unit(UNIT)


def test_cli_redacts_systemctl_stderr_and_source_is_cpu_only():
    script = Path(wait.__file__)
    result = subprocess.run([sys.executable, '-B', str(script), 'not-owned'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'not-owned' not in result.stdout
    text = script.read_text()
    assert 'restart' not in text[text.index('def snapshot'):]
    assert 'docker' not in text and 'nvidia' not in text and 'import torch' not in text

"""Tiny authored publication controls; no Azure, network, data or deletion."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import pytest

import sealed_callback_publication as p


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def snapshot(path):
    s = path.lstat()
    return s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink


def identity(path, maximum):
    path = Path(path); before = snapshot(path); s = path.lstat()
    require(not path.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_nlink == 1
            and not s.st_mode & 0o222 and 0 < s.st_size <= maximum, 'Original regular readonly file')
    raw = path.read_bytes(); require(snapshot(path) == before, 'Original file changed')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def error(exc):
    return type(exc).__name__ if type(exc) in (ValueError, RuntimeError, TimeoutError,
        OSError, KeyError, ImportError, ModuleNotFoundError) else 'other'


def fixture(tmp_path):
    out = tmp_path/'receipt'; out.mkdir(mode=0o700)
    leaf = out/'metadata.json'; leaf.write_bytes(b'{"authored":true}\n'); leaf.chmod(0o400)
    s = out.lstat(); report = dict(status='pass', outputs_sealed=False,
        blob_cleanup_verified=False, source_inputs_rehashed_after=True, archive_removed=True, files=36)
    options = dict(encode=encode, identity=identity, snapshot=snapshot, require=require,
                   check=lambda d: require(time.monotonic() < d, 'Inclusive deadline'), sync=sync, error=error)
    return out, report, (s.st_dev, s.st_ino, s.st_uid), options


def call(out, report, owner, options, callback=None):
    return p.publish(out, report, time.monotonic()+10, time.monotonic(), owner,
                     {'metadata.json'}, callback, **options)


def saved(out):
    return json.loads((out/'report.json').read_bytes())


def test_success_seals_before_callback_and_preserves_original_fd_and_leaves(tmp_path):
    out, r, owner, options = fixture(tmp_path); original = (out/'metadata.json').read_bytes(); events = []
    def callback():
        first = saved(out); events.append(snapshot(out/'report.json')[:2])
        assert first['status'] == 'pass' and first['outputs_sealed'] is True
        assert first['blob_cleanup_verified'] is False
        assert stat.S_IMODE(out.stat().st_mode) == 0o500
    assert call(out, r, owner, options, callback) is r
    assert saved(out) == r and r['blob_cleanup_verified'] is True
    assert 'publication_failed' not in r and events == [snapshot(out/'report.json')[:2]]
    assert (out/'metadata.json').read_bytes() == original
    assert all(stat.S_IMODE(q.stat().st_mode) == 0o400 for q in out.iterdir())


@pytest.mark.parametrize('failure,stage,category',[
    ('verify', 'sealed_verify', 'ValueError'), ('callback', 'after_seal', 'RuntimeError'),
    ('deadline', 'sealed_deadline', 'TimeoutError'), ('sync', 'sealing', 'OSError'),
    ('cleanup_verify', 'cleanup_verify', 'ValueError'), ('cleanup_deadline', 'cleanup_deadline', 'TimeoutError')])
def test_failure_classes_and_stages_persist_without_secrets(tmp_path, failure, stage, category):
    out, r, owner, options = fixture(tmp_path); events = []; calls = []
    def callback():
        events.append(True)
        if failure == 'callback':raise RuntimeError('SECRET_TOKEN_URL_MUST_NOT_BE_SAVED')
    original_identity = options['identity']
    def inspect(path, maximum):
        if Path(path).name == 'report.json':
            calls.append(True)
            if failure == 'verify' or failure == 'cleanup_verify' and len(calls) == 2:
                raise ValueError('SECRET_TOKEN_URL_MUST_NOT_BE_SAVED')
        return original_identity(path, maximum)
    options['identity'] = inspect
    checks = []
    def check(deadline):
        checks.append(True)
        if failure == 'deadline' or failure == 'cleanup_deadline' and len(checks) == 2:
            raise TimeoutError('SECRET_TOKEN_URL_MUST_NOT_BE_SAVED')
    options['check'] = check
    if failure == 'sync':options['sync'] = lambda *_: (_ for _ in ()).throw(OSError('SECRET_TOKEN_URL_MUST_NOT_BE_SAVED'))
    call(out, r, owner, options, callback)
    assert saved(out) == r and r['status'] == 'fail' and r['publication_failed'] is True
    assert r['publication_failure_stage'] == stage and r['publication_error_type'] == category
    assert b'SECRET' not in (out/'report.json').read_bytes()
    assert events == ([True] if failure.startswith('cleanup_') or failure == 'callback' else [])
    assert r['blob_cleanup_verified'] is (failure.startswith('cleanup_'))


@pytest.mark.parametrize('fault', ['extra', 'outmode', 'owner', 'existing_report'])
def test_foreign_namespace_preflight_never_calls_callback_or_mutates_foreign_file(tmp_path, fault):
    out, r, owner, options = fixture(tmp_path); events = []
    if fault == 'extra':(out/'foreign').write_bytes(b'foreign')
    elif fault == 'outmode':out.chmod(0o755)
    elif fault == 'owner':owner = (owner[0], owner[1]+1, owner[2])
    else:(out/'report.json').write_bytes(b'foreign')
    before = {q.name:(q.read_bytes(), snapshot(q)) for q in out.iterdir()}; directory = snapshot(out)
    with pytest.raises(ValueError):call(out, r, owner, options, lambda: events.append(True))
    assert not events and before == {q.name:(q.read_bytes(), snapshot(q)) for q in out.iterdir()}
    assert directory == snapshot(out)


def test_writable_leaf_fails_before_seal_or_callback(tmp_path):
    out, r, owner, options = fixture(tmp_path); (out/'metadata.json').chmod(0o600); events = []
    call(out, r, owner, options, lambda: events.append(True))
    assert saved(out)['status'] == 'fail' and r['publication_failure_stage'] == 'namespace'
    assert not events and stat.S_IMODE(out.stat().st_mode) == 0o700
    assert stat.S_IMODE((out/'metadata.json').stat().st_mode) == 0o600


def test_foreign_report_inode_is_not_overwritten_on_late_callback_race(tmp_path):
    out, r, owner, options = fixture(tmp_path)
    def callback():
        out.chmod(0o700); (out/'report.json').rename(out/'original.json')
        (out/'report.json').write_bytes(b'FOREIGN_KEEP')
    with pytest.raises(ValueError, match='Foreign report inode'):call(out, r, owner, options, callback)
    assert (out/'report.json').read_bytes() == b'FOREIGN_KEEP'
    original = json.loads((out/'original.json').read_bytes())
    assert original['blob_cleanup_verified'] is False


def test_already_failed_report_never_executes_callback(tmp_path):
    out, r, owner, options = fixture(tmp_path); r['status'] = 'fail'; events = []
    call(out, r, owner, options, lambda: events.append(True))
    assert not events and saved(out)['status'] == 'fail' and r['blob_cleanup_verified'] is False


def test_injected_error_classifier_cannot_serialize_exception_text(tmp_path):
    out, r, owner, options = fixture(tmp_path); options['error'] = lambda _: 'SECRET_TOKEN_URL'
    call(out, r, owner, options, lambda: (_ for _ in ()).throw(RuntimeError('SECRET_TOKEN_URL')))
    assert saved(out)['publication_error_type'] == 'other'
    assert b'SECRET' not in (out/'report.json').read_bytes()

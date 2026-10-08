"""Small manufactured files exercise limits; no heavy report is stored."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import subprocess

import pytest

import surface_pose_report_capacity as capacity

REPO = Path(__file__).resolve().parents[1]
REV = 'a' * 40


def case(monkeypatch, tmp_path):
    monkeypatch.setattr(capacity, 'ROOT', tmp_path)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{REV}/outputs')
    path = tmp_path/f'experiments/full4d-v1-{REV}/outputs/episode_000001/object_pose_full_surface/report.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'opaque full original pose receipt')
    path.chmod(0o444)
    return path


def test_only_named_readonly_pose_receipt_gets_capacity(monkeypatch, tmp_path):
    path = case(monkeypatch, tmp_path)
    assert capacity._pose_report_role(path)
    expected = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert capacity.identity(path) == expected == capacity.identity(path, readonly=False)
    assert not capacity._pose_report_role(path.with_name('geometry_and_poses.npz'))
    assert not capacity._pose_report_role(path.parent.parent/'object_pose_full_solid/report.json')
    assert not capacity._pose_report_role(tmp_path/'outputs/episode_000001/object_pose_full_surface/report.json')


def test_capacity_bound_with_stat_spy_not_large_fixture(monkeypatch, tmp_path):
    path = case(monkeypatch, tmp_path)
    original = Path.lstat
    declared = [32 * 1024 * 1024 + 1]
    def lstat(current):
        info = original(current)
        if current != path:
            return info
        keys = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns')
        return SimpleNamespace(**{key:getattr(info,key) for key in keys}, st_size=declared[0])
    monkeypatch.setattr(Path, 'lstat', lstat)
    assert capacity.identity(path)['bytes'] == declared[0]
    declared[0] = capacity.MAX_POSE_REPORT_BYTES + 1
    with pytest.raises(ValueError, match='Bounded'):
        capacity.identity(path)


@pytest.mark.parametrize('fault', ['writable', 'hardlink', 'symlink'])
def test_capacity_never_weakens_regular_readonly_singlelink_contract(monkeypatch, tmp_path, fault):
    path = case(monkeypatch, tmp_path)
    if fault == 'writable': path.chmod(0o644)
    elif fault == 'hardlink': (tmp_path/'alias').hardlink_to(path)
    else: path.unlink(); path.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):
        capacity.identity(path, readonly=False)


def test_other_file_roles_delegate_original_limits_unchanged(monkeypatch, tmp_path):
    path = case(monkeypatch, tmp_path).with_name('geometry_and_poses.npz')
    calls = []
    monkeypatch.setattr(capacity, '_loader', lambda: SimpleNamespace(
        identity=lambda *args, **kwargs: calls.append((args,kwargs)) or {'old':True}))
    assert capacity.identity(path, readonly=False) == {'old':True}
    assert calls == [((path,), {'readonly':False})]


def test_ledger_recheck_uses_capacity_role_but_keeps_original_empty_source(monkeypatch,tmp_path):
    path = case(monkeypatch,tmp_path)
    row = capacity.identity(path)
    empty = tmp_path/'__init__.py'
    empty.write_bytes(b'')
    calls = []
    monkeypatch.setattr(capacity, '_loader', lambda: SimpleNamespace(
        _source_identity=lambda value: calls.append(value) or {'empty':True}))
    capacity.recheck({path:row,empty:{'empty':True}})
    assert calls == [empty]
    with pytest.raises(ValueError):
        capacity.recheck({path:row | {'sha256':'f'*64}})


def test_actual_preparation_patch_is_exactly_source_only():
    original = subprocess.check_output(['git','show','de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba:infra/cari_prepare.py'],cwd=REPO)
    candidate = (REPO/'infra/cari_prepare.py').read_bytes()
    result = capacity.verify_capacity_source(original,candidate)
    assert result['numeric_source_unchanged'] and result['exact_source_substitutions'] == 3
    assert result['maximum_pose_report_bytes'] == 64 << 20
    with pytest.raises(ValueError):
        capacity.verify_capacity_source(original,candidate.replace(b'error > 1e-5',b'error > 1e-4',1))

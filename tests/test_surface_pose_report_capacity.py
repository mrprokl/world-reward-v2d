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
    # This proves the historical lossless capacity/scheduling producer, not
    # the new, opt-in scientific latent initializer contract.
    candidate = subprocess.check_output(['git','show','a51482f63911fa05188ceb3e74d5f5ddc416c8a1:infra/cari_prepare.py'],cwd=REPO)
    result = capacity.verify_capacity_source(original,candidate)
    assert result['numeric_source_unchanged'] and result['exact_source_substitutions'] == 10
    assert result['maximum_pose_report_bytes'] == 64 << 20
    with pytest.raises(ValueError):
        capacity.verify_capacity_source(original,candidate.replace(b'error > 1e-5',b'error > 1e-4',1))


@pytest.mark.parametrize('total', [1, 7, 8, 9, 16, 17, 50])
def test_actual_production_depth_loop_batches_and_tail_preserve_every_record(tmp_path, total):
    """Execute actual old/new loop AST on tiny manufactured arrays, not inference."""
    import ast
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class Record:
        index: int
        raw_depth_m: object
        aligned_depth_m: object
        scale: float
        shift: float
        valid_count: int

    class Writer:
        def __init__(self, path, cameras, **kwargs):
            self.path, self.cameras, self.kwargs = path, cameras, kwargs
            self.records, self.batches, self.complete = [], [], False
        def __enter__(self):
            made.append(self)
            return self
        def __exit__(self, *args):
            assert self.complete
        def write_frame(self, camera, index, raw, aligned, **kwargs):
            self.write_frames(camera, [Record(index, raw, aligned, **kwargs)])
        def write_frames(self, camera, records):
            assert camera == 'inferred_camera' and not self.complete
            self.batches.append(len(records))
            # Mirrors native synchronous write_frames completion: it consumes
            # records before returning, but never changes arrays or metadata.
            self.records.extend(Record(r.index, r.raw_depth_m.copy(), r.aligned_depth_m.copy(),
                r.scale, r.shift, r.valid_count) for r in records)
        def mark_complete(self):
            self.complete = True

    names = [f'{i:06d}' for i in range(total)]
    base = tmp_path/'episode_000022'
    (base/'depth_full').mkdir(parents=True)
    depth_frames, sources = [], []
    for index, name in enumerate(names):
        depth = (np.arange(12, dtype=np.float32).reshape(3, 4) + index) / np.float32(10)
        valid = np.ones((3, 4), dtype=bool)
        valid[0, 1] = False
        path = base/f'depth_full/{name}.npz'
        np.savez(path, depth=depth, mask=valid)
        sources.append((path, path.read_bytes(), depth.copy(), valid.copy()))
        depth_frames.append(dict(output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    original = subprocess.check_output(['git','show','de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba:infra/cari_prepare.py'], cwd=REPO)
    candidate = (REPO/'infra/cari_prepare.py').read_bytes()
    receipts = []
    for source in (original, candidate):
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        block = next(node for node in main.body if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and getattr(node.items[0].context_expr.func, 'id', None) == 'MHRDepthH5Writer')
        made = []
        progress = []
        scope = dict(json=__import__('json'), print=lambda payload, **kwargs: progress.append(__import__('json').loads(payload)), MHRDepthH5Writer=Writer, DepthFrameRecord=Record,
            aligned_depth_path=tmp_path/'unused.h5', camera_name='inferred_camera', names=names,
            identity={'ground_truth_used':False}, base=base, depth_frames=depth_frames, np=np,
            sha256=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), scale=.83)
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<actual-depth-loop>', 'exec'), scope)
        assert len(made) == 1 and made[0].kwargs == dict(
            alignment_method='world_reward_shared_predicted_human_scale',
            alignment_input_identity={'ground_truth_used':False}, encoding_workers=8)
        writer = made[0]
        assert writer.cameras == {'inferred_camera':names} and writer.complete
        if total == 50:
            if source == original:
                assert progress == [{'stage':'cari_prepare_depth','frames_complete':50}]
            else:
                assert progress == [{'stage':'cari_prepare_depth','frames_submitted':50,'frames_complete':48}]
        else:
            assert progress == []
        receipts.append(writer)
    old, new = receipts
    assert old.batches == [1] * total
    assert new.batches == [8] * (total // 8) + ([total % 8] if total % 8 else [])
    assert len(old.records) == len(new.records) == total
    for index, (before, after) in enumerate(zip(old.records, new.records)):
        assert before.index == after.index == index
        assert before.scale == after.scale == .83 and before.shift == after.shift == 0.
        assert before.valid_count == after.valid_count == 11
        for role in ('raw_depth_m', 'aligned_depth_m'):
            a, b = getattr(before, role), getattr(after, role)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert all(path.read_bytes() == payload for path, payload, _, _ in sources)


def test_depth_batch_source_and_receipt_are_pinned_without_validation_shortcuts():
    source = (REPO/'infra/cari_prepare.py').read_text()
    digest = '1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5'
    assert f'sha256(native_root / "prep/mhr_depth_h5.py") != "{digest}"' in source
    assert source.index('Exact qualified native depth-writer source required') < source.index('from prep.mhr_depth_h5 import')
    assert '"native_source_sha256": "'+digest+'"' in source
    original = subprocess.check_output(['git','show','de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba:infra/cari_prepare.py'], cwd=REPO)
    candidate = subprocess.check_output(['git','show','a51482f63911fa05188ceb3e74d5f5ddc416c8a1:infra/cari_prepare.py'],cwd=REPO)
    proof = capacity.verify_capacity_source(original, candidate)
    assert proof['depth_batch_size'] == 8 and proof['depth_native_source_sha256'] == digest
    for old, new in ((b'len(pending) == 8', b'len(pending) == 16'),
                     (b'validation_workers=8', b'validation_workers=1'),
                     (b'aligned = raw * scale', b'aligned = raw * 1.0'),
                     (b'int(valid.sum())', b'int((raw > 0).sum())')):
        assert old in candidate
        with pytest.raises(ValueError):
            capacity.verify_capacity_source(original, candidate.replace(old, new, 1))

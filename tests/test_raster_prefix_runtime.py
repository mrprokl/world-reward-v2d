"""No services/CUDA; exact patch, native-prefix proof and build isolation."""
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'infra'))
spec = importlib.util.spec_from_file_location('raster_prefix_runtime_test', ROOT/'infra/raster_prefix_runtime.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_prefix_proof_for_all_chunk_orders_no_holes_or_changed_valid_order():
    rng = np.random.default_rng(42)
    for _ in range(30):
        chunks = [rng.integers(0, 1000, rng.integers(0, 10)).tolist() for _ in range(20)]
        order = rng.permutation(len(chunks)); hits = sum(len(c) for c in chunks)
        bins = np.full(hits+20, -1); start = 0
        for i in order:
            c = chunks[i]; bins[start:start+len(c)] = c; start += len(c)
        old = [int(x) for x in bins if x >= 0]
        new = []
        for x in bins:
            if x < 0: break
            new.append(int(x))
        assert old == new and np.all(bins[:hits] >= 0) and np.all(bins[hits:] == -1)


def test_exact_source_hash_is_required_before_unique_patch(monkeypatch):
    import hashlib
    raw = b'prefix\n' + module.OLD + b'\nsuffix\n'
    with pytest.raises(ValueError): module.patch_source(raw)
    monkeypatch.setattr(module, 'PATCH_SHA256', hashlib.sha256(raw).hexdigest())
    changed = module.patch_source(raw)
    assert changed.replace(module.NEW, module.OLD) == raw
    assert changed.count(module.NEW) == 1 and module.OLD not in changed


def test_duplicate_context_or_any_publisher_source_change_rejected(monkeypatch):
    import hashlib
    raw = module.OLD + b'\n' + module.OLD
    monkeypatch.setattr(module, 'PATCH_SHA256', hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError): module.patch_source(raw)


def test_child_compiles_only_extension_offline_no_global_or_pip_mutation():
    recipe = module.recipe()
    assert recipe.startswith('FROM '+module.BASE+'\n')
    assert 'setup.py build_ext --build-lib '+module.SITE in recipe
    assert 'TORCH_CUDA_ARCH_LIST=9.0' in recipe and 'FORCE_CUDA=1' in recipe
    assert 'CUDA_VISIBLE_DEVICES=-1' in recipe and 'MAX_JOBS=4' in recipe
    assert 'pip' not in recipe and 'apt' not in recipe and 'git ' not in recipe
    assert 'site-packages' not in recipe and 'rm -rf /tmp/' in recipe


def test_manifest_rejects_wrong_revision_or_unverified_closure():
    with pytest.raises(ValueError): module.source_manifest(dict(sha='a'*40, truncated=False, tree=[]))
    with pytest.raises(ValueError): module.source_manifest(dict(sha=module.REVISION, truncated=False, tree=[]))


def archive_for(name, raw, *, symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        item = tarfile.TarInfo('pytorch3d-'+module.REVISION+'/'+name)
        if symlink: item.type = tarfile.SYMTYPE; item.linkname = '/etc/passwd'; tar.addfile(item)
        else: item.size = len(raw); tar.addfile(item, io.BytesIO(raw))
    return stream.getvalue()


def test_exact_selected_gitblob_only_no_tar_extract(tmp_path):
    import hashlib
    raw = b'public source only'; name = 'setup.py'
    rows = [dict(path=name, bytes=len(raw), git_blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())]
    module.extract_selected(archive_for(name, raw), rows, tmp_path)
    assert (tmp_path/name).read_bytes() == raw and not (tmp_path/name).stat().st_mode & 0o222


@pytest.mark.parametrize('bad', ['symlink', 'hash', 'path'])
def test_changed_or_unsafe_selected_archive_member_rejected(tmp_path, bad):
    rows = [dict(path='setup.py', bytes=3, git_blob_sha='0'*40)]
    raw = archive_for('../escape' if bad == 'path' else 'setup.py', b'abc', symlink=bad == 'symlink')
    with pytest.raises(ValueError): module.extract_selected(raw, rows, tmp_path)


def test_wrapper_has_shared_owned_gpu_lease_durable_budget_and_no_service_calls():
    s = (ROOT/'infra/run_raster_prefix_runtime.sh').read_text()
    assert 'exec 9<' in s and 'flock -n 9' in s and '2100s' in s
    assert 'run_raster_prefix_runtime/code' in s
    assert 'az ' not in s and 'ssh ' not in s and 'pip ' not in s

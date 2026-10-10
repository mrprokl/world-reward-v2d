"""No services/CUDA; exact patch, native-prefix proof and build isolation."""
import importlib.util
from pathlib import Path
import sys

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
    assert 'setup_raster.py build_ext --build-lib '+module.SITE in recipe
    assert 'TORCH_CUDA_ARCH_LIST=9.0' in recipe and 'FORCE_CUDA=1' in recipe
    assert 'CUDA_VISIBLE_DEVICES=-1' in recipe and 'MAX_JOBS=4' in recipe
    assert 'pip' not in recipe and 'apt' not in recipe and 'git ' not in recipe
    assert 'site-packages' not in recipe and 'rm -rf /tmp/' in recipe


def test_manifest_rejects_wrong_revision_or_unverified_closure():
    with pytest.raises(ValueError): module.source_manifest(dict(sha='a'*40, truncated=False, tree=[]))
    with pytest.raises(ValueError): module.source_manifest(dict(sha=module.REVISION, truncated=False, tree=[]))


def blob_row(name, raw):
    import hashlib
    return dict(path=name, bytes=len(raw),
        git_blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def test_exact_concurrent_selected_gitblob_only_verified_before_write(monkeypatch, tmp_path):
    import threading
    calls = []; lock = threading.Lock()
    sources = {'setup.py': b'public build', 'LICENSE': b'public BSD', 'pytorch3d/csrc/x.cu': b'public code'}
    def fake(url, limit, *, deadline):
        name = url.split(module.REVISION+'/')[1]
        with lock: calls.append((name, limit, deadline))
        return sources[name]
    monkeypatch.setattr(module, 'acquire', fake)
    rows = [blob_row(name, raw) for name, raw in sources.items()]
    report = module.acquire_selected(rows, tmp_path)
    assert report['files'] == 3 and report['concurrent_requests'] == 8
    assert report['publisher_archive_downloaded'] is False
    assert {c[0] for c in calls} == set(sources)
    for name, raw in sources.items():
        assert (tmp_path/name).read_bytes() == raw
        assert not (tmp_path/name).stat().st_mode & 0o222


@pytest.mark.parametrize('bad', ['hash', 'path', 'size', 'duplicate'])
def test_changed_or_unsafe_raw_blob_rejected_before_write(monkeypatch, tmp_path, bad):
    row = blob_row('setup.py', b'abc'); rows = [row]
    if bad == 'hash': row['git_blob_sha'] = '0'*40
    if bad == 'path': row['path'] = '../escape'
    if bad == 'size': row['bytes'] = 2
    if bad == 'duplicate': rows.append(dict(row))
    monkeypatch.setattr(module, 'acquire', lambda *a, **kw: b'abc')
    with pytest.raises(ValueError): module.acquire_selected(rows, tmp_path)
    assert not (tmp_path/'setup.py').exists()


@pytest.mark.parametrize('url', [
    'https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/'+module.REVISION,
    'https://raw.githubusercontent.com/facebookresearch/pytorch3d/'+module.REVISION+'/tests/a.py',
    'https://raw.githubusercontent.com/facebookresearch/pytorch3d/'+module.REVISION+'/pytorch3d/csrc/../data',
    'https://example.org/setup.py',
])
def test_nonprimary_media_archive_or_path_traversal_rejected_without_network(url):
    with pytest.raises(ValueError): module.acquire(url, 100)


def test_only_transient_network_response_retried(monkeypatch):
    import urllib.error
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, n): assert n == 4; return b'abc'
    class Opener:
        def open(self, request, timeout):
            calls.append(timeout)
            if len(calls) < 3: raise urllib.error.HTTPError(request.full_url, 503, 'busy', {}, None)
            return Response()
    monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *_: Opener())
    monkeypatch.setattr(module.time, 'sleep', lambda *_: None)
    assert module.acquire('https://raw.githubusercontent.com/facebookresearch/pytorch3d/'+module.REVISION+'/setup.py', 3) == b'abc'
    assert len(calls) == 3


@pytest.mark.parametrize('mode', ['404', 'oversize'])
def test_hard_http_or_wrong_bound_not_retried(monkeypatch, mode):
    import urllib.error
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, n): return b'abcd'
    class Opener:
        def open(self, request, timeout):
            calls.append(timeout)
            if mode == '404': raise urllib.error.HTTPError(request.full_url, 404, 'absent', {}, None)
            return Response()
    monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *_: Opener())
    with pytest.raises((ValueError, urllib.error.HTTPError)):
        module.acquire('https://raw.githubusercontent.com/facebookresearch/pytorch3d/'+module.REVISION+'/setup.py', 3)
    assert len(calls) == 1


def test_wrapper_has_shared_owned_gpu_lease_durable_budget_and_no_service_calls():
    s = (ROOT/'infra/run_raster_prefix_runtime.sh').read_text()
    assert 'exec 9<' in s and 'flock -n 9' in s and '2100s' in s
    assert 'run_raster_prefix_runtime/code' in s
    assert 'az ' not in s and 'ssh ' not in s and 'pip ' not in s


def test_mesh_only_authored_binding_excludes_all_class_registrations():
    raw = module.raster_binding().decode()
    assert raw.count('m.def(') == 5
    assert 'py::class_' not in raw and 'Pulsar' not in raw
    for name in ('RasterizeMeshes', 'RasterizeMeshesBackward', 'RasterizeMeshesCoarse', 'RasterizeMeshesFine', 'RasterizeMeshesNaive'):
        assert '&'+name in raw
    assert '#include "rasterize_meshes/rasterize_meshes.h"' in raw


def test_four_native_translation_units_only_preserve_published_compile_flags():
    import ast
    source = module.raster_setup().decode(); tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign))
    paths = ast.literal_eval(assignment.value)
    assert paths == ['ext_raster.cpp', 'pytorch3d/csrc/rasterize_meshes/rasterize_meshes_cpu.cpp',
                     'pytorch3d/csrc/rasterize_meshes/rasterize_meshes.cu', 'pytorch3d/csrc/rasterize_coarse/rasterize_coarse.cu']
    assert 'ext.cpp' not in paths and 'Pulsar' not in source
    for flag in ('WITH_CUDA', 'THRUST_IGNORE_CUB_VERSION_CHECK', '-std=c++17',
                 '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'):
        assert flag in source
    assert 'fast_math' not in source and 'setup.py' not in module.recipe()

"""Tiny manufactured archives and mocked containers only; no CUDA/build/network."""
import ast
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import types
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('wr_sm90_test', REPO / 'infra/masa_sm90_build.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny_tar(path, entries):
    with tarfile.open(path, 'w:gz') as t:
        for name, value, kind in entries:
            item = tarfile.TarInfo(name)
            item.mode = 0o755 if '/bin/' in name else 0o644
            if kind == 'link':
                item.type = tarfile.SYMTYPE
                item.linkname = value
                t.addfile(item)
            elif kind == 'hardlink':
                item.type = tarfile.LNKTYPE
                item.linkname = value
                t.addfile(item)
            else:
                item.size = len(value)
                t.addfile(item, io.BytesIO(value))
    return path.read_bytes()


def tiny_wheel(path, *, ninja=False, bad=None):
    name, version = ('ninja', '1.11.1.1') if ninja else ('mmcv', '2.1.0')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name + '.dist-info/METADATA', 'Name: ' + name + '\nVersion: ' + version + '\n')
        z.writestr(name + '.dist-info/LICENSE', b'Tiny license\n')
        z.writestr('ninja/data/bin/ninja' if ninja else 'mmcv/_ext.cpython-311-x86_64-linux-gnu.so', b'Tiny ELF')
        if bad:
            z.writestr(bad, b'bad')
    return path.read_bytes()


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def test_actual_config_source_scripts_parse_and_fixed_limits(gate):
    raw = (REPO / gate.CONFIG).read_bytes()
    c = json.loads(raw)
    assert gate.CONFIG_PIN == pin(raw)
    assert c['prior_runtime']['image_id'] == gate.BASE
    assert c['source']['root_tree'] == '4db7bd85dc95a1a8c3f7ff6e9d3382eb582be462'
    assert c['source']['recursive_tree_metadata']['url'].split('/trees/')[1].startswith(c['source']['root_tree'])
    assert c['source']['blob_count_including_symlink'] == 837
    assert c['source']['symlink_policy']['file'] == 'docs/zh_cn/mmcv-logo.png'
    assert c['compiler']['environment'] == dict(CUDA_HOME=gate.TOOLKIT, CC='/usr/bin/gcc', CXX='/usr/bin/g++',
          MAX_JOBS='32', FORCE_CUDA='1', MMCV_WITH_OPS='1', TORCH_CUDA_ARCH_LIST='9.0', LD_LIBRARY_PATH='')
    assert c['limits']['root_budget_seconds_including_acquisition_build_and_publication'] == 2400
    assert c['limits']['cpu_compile_budget_seconds_within_root_budget'] == 1800
    assert c['claims']['submission_qualified'] is False and c['license']['license_eligibility_verified'] is False
    ast.parse(gate.COMPILE)
    ast.parse(gate.DIST_PROBE)
    assert "'--no-build-isolation'" in gate.COMPILE and "'-arch=sm_90'" in gate.COMPILE
    assert '--no-index --no-deps' in gate.recipe(Path('mmcv-2.1.0-cp311-cp311-linux_x86_64.whl')).decode()
    assert 'TORCH_DONT_CHECK_COMPILER_ABI' not in gate.COMPILE


def test_original_helper_pins_are_actual(gate):
    for name, identity in gate.HELPER_PINS.items():
        assert pin((REPO / name).read_bytes()) == identity


def test_git_root_reconstruction_not_flat_list(gate):
    rows = [dict(path='z.txt', mode='100644', sha=gate.git_hash('blob', b'z')),
            dict(path='a/file.py', mode='100755', sha=gate.git_hash('blob', b'a')),
            dict(path='a.c', mode='100644', sha=gate.git_hash('blob', b'b'))]
    child = gate.git_hash('tree', b'100755 file.py\0' + bytes.fromhex(rows[1]['sha']))
    raw = (b'100644 a.c\0' + bytes.fromhex(rows[2]['sha']) + b'40000 a\0' + bytes.fromhex(child)
           + b'100644 z.txt\0' + bytes.fromhex(rows[0]['sha']))
    assert gate.git_root(rows) == gate.git_hash('tree', raw)
    assert gate.git_root(list(reversed(rows))) == gate.git_root(rows)
    changed = copy.deepcopy(rows)
    changed[1]['mode'] = '100644'
    assert gate.git_root(changed) != gate.git_root(rows)


@pytest.mark.parametrize('path', ['/escape', '../escape', 'a/../escape'])
def test_git_noncanonical_path_rejected(gate, path):
    with pytest.raises(ValueError):
        gate.git_root([dict(path=path, mode='100644', sha='a' * 40)])


@pytest.mark.parametrize('name,kind,value', [('../escape', 'file', b'x'), ('/escape', 'file', b'x'),
    ('root/link', 'link', '../../escape'), ('root/link', 'hardlink', 'root/file'),
    ('root/link/file', 'file', b'x')])
def test_archive_escape_links_and_devices_rejected(gate, tmp_path, name, kind, value):
    entries = [(name, value, kind)]
    if name == 'root/link/file':
        entries = [('root/link', 'file', 'link'), *entries]
    archive = tmp_path / 'a.tgz'
    tiny_tar(archive, entries)
    with pytest.raises(ValueError):
        gate.safe_tar(archive, tmp_path / 'out', prefix='root', maximum=1000)


def test_archive_safe_git_symlink_and_source_posthash(gate, tmp_path):
    archive = tmp_path / 'a.tgz'
    entries = [('root/LICENSE', b'Apache', 'file'), ('root/docs/logo', b'logo', 'file'),
               ('root/docs/local/logo', '../logo', 'link')]
    tiny_tar(archive, entries)
    root = tmp_path / 'source'
    gate.safe_tar(archive, root, prefix='root', maximum=1000)
    rows = [dict(path=n.removeprefix('root/'), size=len(v), mode='120000' if k == 'link' else '100644',
                 sha=gate.git_hash('blob', v.encode() if k == 'link' else v)) for n, v, k in entries]
    c = dict(source=dict(symlink_policy=dict(file='docs/local/logo', target='../logo'),
                         required_file_pins=[dict(file='LICENSE', **pin(b'Apache'))]))
    proof = gate.source_inventory(root, rows, c)
    assert proof['root_tree'] == gate.git_root(rows)
    (root / 'LICENSE').chmod(0o644)
    (root / 'LICENSE').write_bytes(b'Bad')
    (root / 'LICENSE').chmod(0o444)
    with pytest.raises(ValueError):
        gate.source_inventory(root, rows, c)


def test_toolkit_merge_preserves_layout_and_rejects_collision(gate, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    (a / 'include').mkdir(parents=True)
    (b / 'include').mkdir(parents=True)
    (a / 'include/cuda.h').write_bytes(b'one')
    (b / 'include/cuda.h').write_bytes(b'one')
    gate.toolkit_merge([a, b], tmp_path / 'combined')
    assert (tmp_path / 'combined/include/cuda.h').read_bytes() == b'one'
    (b / 'include/cuda.h').write_bytes(b'two')
    with pytest.raises(ValueError):
        gate.toolkit_merge([a, b], tmp_path / 'rejected')


@pytest.mark.parametrize('bad', ['../escape', '/escape', 'bad\\escape'])
def test_ninja_and_compiled_wheel_unsafe_inventory(gate, tmp_path, bad):
    p = tmp_path / 'ninja.whl'
    tiny_wheel(p, ninja=True, bad=bad)
    with pytest.raises(ValueError):
        gate.ninja_binary(p, tmp_path / 'ninja')
    p = tmp_path / 'mmcv.whl'
    tiny_wheel(p, bad=bad)
    with pytest.raises(ValueError):
        gate.compiled_wheel(p)


def test_ninja_binary_not_installed_into_runtime(gate, tmp_path):
    p = tmp_path / 'ninja.whl'
    tiny_wheel(p, ninja=True)
    found = gate.ninja_binary(p, tmp_path / 'ninja')
    assert found and (tmp_path / 'ninja/ninja').read_bytes() == b'Tiny ELF'
    assert stat.S_IMODE((tmp_path / 'ninja/ninja').stat().st_mode) == 0o555


@pytest.mark.parametrize('url', ['http://developer.download.nvidia.com/a', 'https://user:secret@pypi.org/a',
    'https://pypi.org/a?token=secret', 'https://evil.example/a', 'https://pypi.org/a#secret'])
def test_unsigned_credentials_and_unlisted_URLs_rejected(gate, url):
    with pytest.raises(ValueError):
        gate.endpoint(url)


class Response(io.BytesIO):
    def __init__(self, raw, url):
        super().__init__(raw)
        self.status, self.url = 200, url
        self.headers = email.message.Message()
        self.headers['Content-Length'] = str(len(raw))
    def geturl(self):
        return self.url


class Opener:
    def __init__(self, payload):
        self.payload = payload
    def open(self, req, timeout):
        assert 0 < timeout <= 30
        return Response(self.payload[req.full_url], req.full_url)


def test_exact_fetch_hash_and_partial_cleanup_key(gate, tmp_path):
    row = dict(url='https://files.pythonhosted.org/a.whl', **pin(b'tiny'))
    partials = []
    assert gate.fetch(None, row, tmp_path / 'a.whl', Opener({row['url']: b'tiny'}), gate.time.monotonic() + 10, partials) == pin(b'tiny')
    assert partials and not partials[0][0].exists()
    assert (tmp_path / 'a.whl').stat().st_nlink == 1
    with pytest.raises(ValueError):
        gate.fetch(None, row, tmp_path / 'bad.whl', Opener({row['url']: b'bad'}), gate.time.monotonic() + 10, [])


def test_cpu_compile_dispatch_no_GPU_pinned_environment(gate, tmp_path, monkeypatch):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    calls = []
    monkeypatch.setattr(gate, 'absent_container', lambda *args: None)
    def command(args, deadline, log=None):
        calls.append((args, deadline))
        work = tmp_path / 'work/wheels'
        work.mkdir(parents=True)
        p = work / 'mmcv-2.1.0-cp311-cp311-linux_x86_64.whl'
        tiny_wheel(p)
        data = dict(compiler='11.8.89', gcc='11.4.0', gxx='11.4.0', ninja='1.11.1', arch='sm_90',
                    tiny_object_compiled=True, ATen_CUDAContext_and_Python_headers_compiled=True,
                    full_original_source_posthash=True, gpu_used=False, wheel_file=p.name)
        log.write_text(json.dumps(data))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(gate, 'command', command)
    rt = types.SimpleNamespace(strict=json.loads)
    proof, wheel = gate.compile_full(rt, tmp_path, 'a' * 40, c, gate.time.monotonic() + 2400)
    args, deadline = calls[0]
    assert '--gpus' not in args and args[args.index('--runtime') + 1] == 'runc'
    assert 'NVIDIA_VISIBLE_DEVICES=void' in args and 'CUDA_VISIBLE_DEVICES=' in args
    assert 'CUDA_HOME=' + gate.TOOLKIT in args and 'LD_LIBRARY_PATH=' in args
    assert 'FORCE_CUDA=1' in args and 'TORCH_CUDA_ARCH_LIST=9.0' in args
    assert args[args.index('--cpus') + 1] == '32' and args[args.index('--memory') + 1] == '128g'
    assert deadline <= gate.time.monotonic() + 1800 and wheel.is_file() and proof['gpu_used'] is False


def test_same_inode_late_seal_demotes_PASS(gate, tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(gate.time, 'monotonic', lambda: clock[0])
    original = gate.os.fsync
    def fsync(fd):
        original(fd)
        clock[0] = 3401.
    monkeypatch.setattr(gate.os, 'fsync', fsync)
    report = dict(status='pass', gpu_used=False)
    gate.seal_report(None, tmp_path, report, 1000.)
    p = tmp_path / 'report.json'
    actual = json.loads(p.read_bytes())
    assert actual['status'] == 'fail' and actual['failure_gate'] == 'root_deadline'
    assert actual['elapsed_seconds'] == 2401 and p.stat().st_nlink == 1
    assert stat.S_IMODE(p.stat().st_mode) == 0o444


@pytest.mark.parametrize('error', ['Cannot connect to daemon', 'permission denied', 'error: no such image: other'])
def test_daemon_failure_not_fresh_image_absence(gate, monkeypatch, error):
    monkeypatch.setattr(gate, 'command', lambda *args, **kw: subprocess.CompletedProcess(args, 1, b'', error.encode()))
    with pytest.raises(ValueError):
        gate.image(types.SimpleNamespace(strict=json.loads), 'target', 123, absent=True)


def test_foreign_container_never_removed(gate, tmp_path, monkeypatch):
    p = tmp_path / 'x.cid'
    p.write_text('c' * 64)
    calls = []
    def command(args, *a, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(dict(Id='c' * 64, Name='/foreign',
                        Config=dict(Labels={gate.LABEL: 'another'}))).encode(), b'')
    monkeypatch.setattr(gate, 'command', command)
    rt = types.SimpleNamespace(identity=lambda p, *a, **kw: pin(p.read_bytes()), strict=json.loads)
    with pytest.raises(ValueError):
        gate.remove_container(rt, p, 'ours', 'a' * 40, 123)
    assert not any(a[:2] == ['docker', 'rm'] for a in calls)


def test_shell_source_closure(gate):
    subprocess.run(['bash', '-n', str(REPO / 'infra/run_masa_sm90_build.sh')], check=True)
    spec = importlib.util.spec_from_file_location('wr_sm90_closure', REPO / 'infra/azure_job.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for folder in ('infra', 'src', 'configs')
             for p in (REPO / folder).rglob('*') if p.is_file()}
    assert set(gate.HELPERS) <= set(m.runtime_bundle_paths(files, 'infra/run_masa_sm90_build.sh'))


def host_fixture(gate, tmp_path, monkeypatch, fault=None):
    root = tmp_path / 'root'
    (root / 'results').mkdir(parents=True)
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    code = root / 'code'
    code.mkdir()
    revision = 'a' * 40
    args = types.SimpleNamespace(revision=revision, runtime_revision=c['prior_runtime']['producer_revision'],
        runtime_report_bytes=c['prior_runtime']['report']['bytes'], runtime_report_sha256=c['prior_runtime']['report']['sha256'],
        image_id=gate.BASE)
    marker = root / 'prior.json'
    marker.write_bytes(b'prior')
    marker.chmod(0o444)
    binding = dict(helpers={}, closure='tiny')
    class RT:
        strict = staticmethod(json.loads)
        canonical = staticmethod(lambda p: Path(p))
        source = staticmethod(lambda *args: binding)
        write = staticmethod(lambda p, raw: p.write_bytes(raw))
        @staticmethod
        def identity(p, *args, **kw):
            return pin(p.read_bytes())
    rt = RT()
    parent = dict(image_id=gate.BASE, layers=['sha256:' + '1' * 64])
    child = dict(image_id='sha256:' + '2' * 64, layers=parent['layers'] + ['sha256:' + '3' * 64])
    monkeypatch.setattr(gate, 'ROOT', root)
    monkeypatch.setattr(gate, 'load_helpers', lambda _: (rt, None))
    monkeypatch.setattr(gate, 'current_source', lambda *args: binding)
    monkeypatch.setattr(gate, 'policy', lambda *args: c)
    versions = {'torch': '2.1.2+cu118', 'mmcv': '2.1.0'}
    monkeypatch.setattr(gate, 'authenticate', lambda *args: dict(versions=versions, frozen={marker: pin(b'prior')},
                      source=binding, old_code=code, image=parent, report_pin=c['prior_runtime']['report']))
    monkeypatch.setattr(gate.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=30 << 30))
    state = dict(built=False, removed=False)
    def image(rt, name, *args, absent=False, owner=None):
        if name == gate.BASE:
            return parent
        if absent:
            assert not state['built']
            return None
        if not state['built']:
            raise ValueError('absent')
        return child
    monkeypatch.setattr(gate, 'image', image)
    monkeypatch.setattr(gate, 'remove_container', lambda *args: None)
    monkeypatch.setattr(gate, 'cpu', lambda *args: dict(versions=versions, non_mmcv_distribution_file_digests={'torch': '1' * 64},
                          mmcv_extension_sha256=hashlib.sha256(b'Tiny ELF').hexdigest()))
    original_fetch = gate.fetch
    def fetch(rt, row, path, opener, deadline, partials, **kw):
        if path.name == 'redistrib_11.8.0.json':
            content = {}
            for r in c['assets']:
                if r['component'].startswith('cuda_'):
                    content[r['component']] = {'linux-x86_64': dict(sha256=r['sha256'], size=str(r['bytes']),
                                  relative_path=r['url'].split('/redist/')[1])}
            raw = json.dumps(content).encode()
        elif path.name.endswith('-pypi.json'):
            r = next(r for r in c['assets'] if r['component'] == 'ninja')
            raw = json.dumps(dict(urls=[dict(filename=r['file'], digests={'sha256': r['sha256']}, size=r['bytes'], url=r['url'])])).encode()
        else:
            raw = b'Tiny publisher bytes'
        path.write_bytes(raw)
        path.chmod(0o444)
        return pin(raw)
    monkeypatch.setattr(gate, 'fetch', fetch)
    monkeypatch.setattr(gate, 'tree_rows', lambda *args: [])
    def safe_tar(path, destination, **kw):
        destination.mkdir()
        if destination.name == 'source':
            for n in ('LICENSE', 'LICENSES.md'):
                (destination / n).write_bytes(b'Original notice')
        else:
            (destination / 'LICENSE').write_bytes(b'CUDA notice')
    monkeypatch.setattr(gate, 'safe_tar', safe_tar)
    def deb_headers(path, destination, c):
        (destination / 'include').mkdir(parents=True)
        (destination / 'include/library.h').write_bytes(b'header')
        (destination / 'notices').mkdir()
        (destination / 'notices/copyright').write_bytes(b'NVIDIA notice')
        return dict(selected_header_and_notice_bytes=20, headers=1, notices=1)
    monkeypatch.setattr(gate, 'deb_headers', deb_headers)
    monkeypatch.setattr(gate, 'source_inventory', lambda *args: dict(root_tree=c['source']['root_tree'], blobs=837))
    def toolkit_merge(roots, destination):
        destination.mkdir()
        for n in c['toolkit_inventory']['before_compile_required'] + ['lib/libcudart.so']:
            p = destination / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'tiny')
    monkeypatch.setattr(gate, 'toolkit_merge', toolkit_merge)
    def ninja_binary(path, destination):
        destination.mkdir()
        (destination / 'ninja').write_bytes(b'tiny')
        return dict(license=pin(b'notice'))
    monkeypatch.setattr(gate, 'ninja_binary', ninja_binary)
    def compile_full(rt, out, *args):
        work = out / 'work'
        work.mkdir()
        if fault == 'compile':
            raise RuntimeError('sensitive token=DO_NOT_PUBLISH')
        wheel = out / 'mmcv-2.1.0-cp311-cp311-linux_x86_64.whl'
        tiny_wheel(wheel)
        wheel.chmod(0o444)
        return dict(full_original_source_posthash=True, gpu_used=False), wheel
    monkeypatch.setattr(gate, 'compile_full', compile_full)
    def command(args, *a, **kw):
        if args[:2] == ['docker', 'build']:
            state['built'] = True
            if fault == 'build':
                return subprocess.CompletedProcess(args, 1)
        elif args[:3] == ['docker', 'image', 'rm']:
            assert args[3] != gate.BASE
            state.update(built=False, removed=True)
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(gate, 'command', command)
    return root, code, args, state


def test_mock_host_PASS_separate_lineage_cleaned_source_no_model(gate, tmp_path, monkeypatch):
    root, code, args, state = host_fixture(gate, tmp_path, monkeypatch)
    report = gate.run(code, args, opener=object())
    assert report['schema'] == 'world_reward.masa_sm90_build_receipt.v1' and report['status'] == 'pass'
    assert report['gpu_used'] is report['model_constructed'] is report['operator_qualified'] is False
    assert report['prior_image_unchanged'] and report['owned_disposable_cleanup'] and report['source_rehashed_after']
    assert report['public_assets'] and report['preserved_notices'] and state['built'] and not state['removed']
    out = root / 'results' / ('masa-sm90-build-' + args.revision)
    assert not any((out / n).exists() for n in ('source', 'toolkit', 'work', 'components', 'ninja'))
    assert stat.S_IMODE(out.stat().st_mode) == 0o555 and stat.S_IMODE((out / 'report.json').stat().st_mode) == 0o444


@pytest.mark.parametrize('fault', ['compile', 'build'])
def test_mock_host_fail_cleanup_no_raw_sensitive_message(gate, tmp_path, monkeypatch, fault):
    root, code, args, state = host_fixture(gate, tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        gate.run(code, args, opener=object())
    out = root / 'results' / ('masa-sm90-build-' + args.revision)
    raw = (out / 'report.json').read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'fail' and b'DO_NOT_PUBLISH' not in raw and b'token=' not in raw
    assert report['owned_disposable_cleanup'] and not (out / 'work').exists()
    if fault == 'build':
        assert state['removed']


def test_mock_posthash_tampering_demotes_and_removes_child(gate, tmp_path, monkeypatch):
    root, code, args, state = host_fixture(gate, tmp_path, monkeypatch)
    original = gate.cpu
    def cpu(*a):
        result = original(*a)
        if a[4] == 'child':
            p = root / 'prior.json'
            p.chmod(0o644)
            p.write_bytes(b'changed')
            p.chmod(0o444)
        return result
    monkeypatch.setattr(gate, 'cpu', cpu)
    with pytest.raises(ValueError):
        gate.run(code, args, opener=object())
    report = json.loads((root / 'results' / ('masa-sm90-build-' + args.revision) / 'report.json').read_bytes())
    assert report['status'] == 'fail' and report['failure_gate'] == 'postcheck_or_owned_cleanup' and state['removed']


def test_compile_diagnostic_fixed_steps_and_log_identity_no_raw(gate, tmp_path):
    (tmp_path / 'work').mkdir()
    (tmp_path / 'work/compiler-output.log').write_bytes(b'compiler failure private detail')
    (tmp_path / 'compile.log').write_text('null\n{"native_compile_step":"toolchain_probe"}\n'
        'private sensitive error\n{"native_compile_step":"full_original_build"}\n'
        '{"native_compile_step":"SECRET_TOKEN"}\n')
    result = gate.compile_diagnostic(types.SimpleNamespace(strict=json.loads), tmp_path)
    assert result == dict(native_compile_step='full_original_build',
                          private_compiler_output_identity=pin(b'compiler failure private detail'))
    assert 'private' not in json.dumps(result).replace('private_compiler_output_identity', '')


def make_deb(path, *, extra=False, unsafe=None, root_entry=False, trailing_slash=True):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as t:
        if root_entry:
            root = tarfile.TarInfo('./')
            root.type, root.mode = tarfile.DIRTYPE, 0o755
            t.addfile(root)
        entries = [('usr/local/cuda-11.8/targets/x86_64-linux/include/library.h', b'header'),
                   ('usr/local/cuda-11.8/targets/x86_64-linux/lib/libunused.a', b'large static skip'),
                   ('usr/share/doc/library/copyright', b'NVIDIA notice')]
        if unsafe:
            entries.append((unsafe, b'bad'))
        for n, raw in entries:
            m = tarfile.TarInfo(n)
            m.mode, m.size = 0o644, len(raw)
            t.addfile(m, io.BytesIO(raw))
    control = io.BytesIO()
    with tarfile.open(fileobj=control, mode='w:gz') as t:
        pass
    members = [('debian-binary', b'2.0\n'), ('control.tar.gz', control.getvalue()), ('data.tar.gz', data.getvalue())]
    if extra:
        members.append(('malicious', b'bad'))
    raw = bytearray(b'!<arch>\n')
    for name, body in members:
        header = ((name + ('/' if trailing_slash else '')).ljust(16) + '0'.ljust(12) + '0'.ljust(6) + '0'.ljust(6)
                  + '100644'.ljust(8) + str(len(body)).ljust(10) + '`\n').encode()
        assert len(header) == 60
        raw += header + body + (b'\n' if len(body) % 2 else b'')
    path.write_bytes(raw)


def test_deb_full_members_only_header_notice_selected_no_libraries(gate, tmp_path):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    p = tmp_path / 'library.deb'
    make_deb(p)
    report = gate.deb_headers(p, tmp_path / 'out', c)
    assert report['headers'] == report['notices'] == 1
    assert report['data_members'] == 3 and report['shared_or_static_libraries_installed'] is False
    assert (tmp_path / 'out/include/library.h').read_bytes() == b'header'
    assert not list((tmp_path / 'out').rglob('*.a')) and not list((tmp_path / 'out').rglob('*.so'))
    assert [r['name'] for r in report['ar_members']] == ['debian-binary', 'control.tar.gz', 'data.tar.gz']


def test_debian_tar_canonical_root_directory_is_not_a_header_or_traversal(gate, tmp_path):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    p = tmp_path / 'library.deb'
    make_deb(p, root_entry=True)
    report = gate.deb_headers(p, tmp_path / 'out', c)
    assert report['headers'] == report['notices'] == 1 and report['data_members'] == 3


def test_actual_nvidia_deb_short_ar_names_without_slash(gate, tmp_path):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    p = tmp_path / 'library.deb'
    make_deb(p, trailing_slash=False, root_entry=True)
    report = gate.deb_headers(p, tmp_path / 'out', c)
    assert [r['name'] for r in report['ar_members']] == ['debian-binary', 'control.tar.gz', 'data.tar.gz']
    assert report['headers'] == 1


@pytest.mark.parametrize('fault', ['extra_ar', 'unsafe_tar', 'oversize_selected', 'missing_hash_record'])
def test_deb_and_frozen_metadata_fail_fast(gate, tmp_path, fault):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    p = tmp_path / 'library.deb'
    make_deb(p, extra=fault == 'extra_ar', unsafe='../escape' if fault == 'unsafe_tar' else None)
    if fault == 'oversize_selected':
        c['limits']['maximum_DEB_selected_header_and_notice_bytes_total'] = 1
    if fault == 'missing_hash_record':
        c['header_assets'][0]['sha256'] = '1' * 64
        with pytest.raises(ValueError):
            gate.header_metadata(c)
    else:
        with pytest.raises(ValueError):
            gate.deb_headers(p, tmp_path / 'out', c)


def test_exact_header_metadata_and_transitive_ATen_probe(gate):
    c = json.loads((REPO / gate.CONFIG).read_bytes())
    gate.header_metadata(c)
    assert sum(r['bytes'] for r in c['header_assets']) == 421032616
    assert '<ATen/cuda/CUDAContext.h>' in gate.COMPILE and '<Python.h>' in gate.COMPILE
    assert 'include_paths(cuda=True)' in gate.COMPILE and 'torch._C._GLIBCXX_USE_CXX11_ABI' in gate.COMPILE
    assert "'dpkg'" not in gate.COMPILE and "'apt'" not in gate.COMPILE

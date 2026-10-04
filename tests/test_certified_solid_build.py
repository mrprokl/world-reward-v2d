"""Tiny metadata/archive and mocked commands; no local build/download/model."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('wr_solid_build', ROOT / 'infra/certified_solid_build.py')
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_protocol_exact_measured_pins_and_numerical_rights_scope():
    actual = build.strict_json((ROOT / build.CONFIG).read_bytes())
    assert actual == build.EXPECTED
    assert actual['parent_image_id'] == build.PARENT
    assert actual['controls'] == 15 and actual['budgets_seconds']['inclusive'] == 2400
    assert actual['compiler_flags'] == ['-O2', '-std=c++17', '-ffp-contract=off', '-frounding-math', '-fno-fast-math']
    assert actual['installed_libraries']['libgmp10'].startswith('2:')
    assert actual['rights']['binary_is_apache_only'] is False
    assert not actual['rights']['competition_eligibility_verified']
    assert all(value is False for value in actual['scope'].values())
    assert sum(p['bytes'] for p in actual['dependencies']) == 617710


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_integrity_json_rejects_duplicates_nonfinite(raw):
    with pytest.raises(ValueError):
        build.strict_json(raw)


def test_identity_reads_excludes_access_time_and_preserves_original(tmp_path):
    p = tmp_path / 'source'; p.write_bytes(b'abc'); p.chmod(0o444)
    expected = {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert build.identity(p, readonly=True) == expected
    assert build.identity(p, readonly=True) == expected
    assert p.read_bytes() == b'abc' and stat.S_IMODE(p.stat().st_mode) == 0o444
    p.chmod(0o644)
    with pytest.raises(ValueError):
        build.identity(p, readonly=True)
    link = tmp_path / 'link'; link.symlink_to(p)
    with pytest.raises(ValueError):
        build.identity(link)


class Response(io.BytesIO):
    status = 200
    def __init__(self, data, **headers):
        super().__init__(data); self.headers = headers


class Opener:
    def __init__(self, response):
        self.response = response; self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, dict(request.headers), timeout))
        return self.response


def pin(data):
    return {'url': 'https://archive.ubuntu.com/ubuntu/pool/fixture.deb', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def test_bounded_direct_https_exactsha_no_proxy_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(build.time, 'monotonic', lambda: 1)
    opener = Opener(Response(b'abc', **{'Content-Length': '3'}))
    result = build.download(pin(b'abc'), tmp_path / 'deb', 10, opener)
    assert result == {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert opener.calls == [('https://archive.ubuntu.com/ubuntu/pool/fixture.deb', {'Accept-encoding': 'identity'}, 9)]


@pytest.mark.parametrize('response,changed', [(b'abcd', {}), (b'ab', {}), (b'xyz', {}),
                                          (b'abc', {'Content-Encoding': 'gzip'}), (b'abc', {'Content-Length': '4'})])
def test_download_fails_size_hash_encoding_before_publication(tmp_path, monkeypatch, response, changed):
    monkeypatch.setattr(build.time, 'monotonic', lambda: 1)
    with pytest.raises(ValueError):
        build.download(pin(b'abc'), tmp_path / 'deb', 10, Opener(Response(response, **changed)))


@pytest.mark.parametrize('url', ['http://archive.ubuntu.com/x', 'https://secret@example.com/x',
                                'https://evil.test/x', 'https://github.com:444/x', 'file:///etc/passwd'])
def test_redirect_route_rejects_nonpublic_or_auth(url):
    with pytest.raises(ValueError):
        build.safe_url(url)


def make_archive(path, extras=()):
    with tarfile.open(path, 'w:xz') as tar:
        for name, data in [('CGAL-6.0.1/include/CGAL/version.h', b'version'),
                           ('CGAL-6.0.1/include/CGAL/Exact_predicates_exact_constructions_kernel.h', b'kernel'),
                           ('CGAL-6.0.1/README', b'not an extracted asset'), *extras]:
            info = tarfile.TarInfo(name)
            if isinstance(data, bytes):
                info.size = len(data); tar.addfile(info, io.BytesIO(data))
            else:
                info.type = data[1]; info.linkname = 'target'; tar.addfile(info)


def test_archive_inventory_before_only_headers_extracted(tmp_path):
    archive = tmp_path / 'library.xz'; make_archive(archive)
    target = tmp_path / 'headers'
    proof = build.extract_headers(archive, target, build.EXPECTED['capacities'])
    assert proof['members'] == 3 and proof['header_files'] == 2
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == {
        'CGAL/version.h', 'CGAL/Exact_predicates_exact_constructions_kernel.h'}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o444 for p in target.rglob('*') if p.is_file())
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o755 for p in target.rglob('*') if p.is_dir())


def test_headers_traversable_under_dispatch_private_umask(tmp_path):
    archive = tmp_path / 'library.xz'; make_archive(archive)
    target = tmp_path / 'headers'
    previous = os.umask(0o077)
    try:
        build.extract_headers(archive, target, build.EXPECTED['capacities'])
    finally:
        os.umask(previous)
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o755
               for p in (target, *(p for p in target.rglob('*') if p.is_dir())))


@pytest.mark.parametrize('extra', [('CGAL-6.0.1/../bad', b'a'), ('/absolute', b'a'),
                                 ('CGAL-6.0.1/linked', ('special', tarfile.SYMTYPE)), ('CGAL-6.0.1/hard', ('special', tarfile.LNKTYPE)),
                                 ('CGAL-6.0.1/device', ('special', tarfile.CHRTYPE)), ('CGAL-6.0.1/include/CGAL/version.h', b'duplicate'),
                                 ('CGAL-6.0.1/include/CGAL/version.h/child', b'ancestor')])
def test_archive_unsafe_whole_inventory_fails_before_any_extract(tmp_path, extra):
    archive = tmp_path / 'library.xz'; make_archive(archive, [extra])
    target = tmp_path / 'headers'
    with pytest.raises(ValueError):
        build.extract_headers(archive, target, build.EXPECTED['capacities'])
    assert not target.exists()


def test_archive_capacity_and_source_version_fail_closed(tmp_path):
    archive = tmp_path / 'library.xz'; make_archive(archive)
    with pytest.raises(ValueError):
        build.extract_headers(archive, tmp_path / 'headers', {'members': 2, 'member_bytes': 1024, 'expanded_bytes': 1024})
    with pytest.raises(ValueError):
        build.extract_headers(archive, tmp_path / 'other', {'members': 10, 'member_bytes': 1024, 'expanded_bytes': 1})


def control_report():
    return dict(stage='certified_solid_procedural_controls_v1', status='pass', phase='complete',
                native_source_sha256='a' * 64, source_binary_rehashed_after=True,
                maximum_calls=15, call_seconds=60, inclusive_seconds=900, elapsed_seconds=1.0,
                records=[{'name': n, 'status': 'pass'} for n in build.CONTROL_NAMES],
                **{k: False for k in ('challenge_inputs_used', 'gt_used', 'qem_executed', 'adoption',
                                     'production_mesh_validated', 'reconstruction_accuracy_verified')})


def test_all_fifteen_procedural_controls_required_before_binary_retention():
    r = control_report(); build.validate_controls(r, 'a' * 64)
    for mutation in [lambda x: x['records'].pop(), lambda x: x['records'].reverse(),
                     lambda x: x['records'][0].update(status='fail'), lambda x: x.update(native_source_sha256='b' * 64),
                     lambda x: x.update(source_binary_rehashed_after=False), lambda x: x.update(gt_used=True),
                     lambda x: x.update(elapsed_seconds=901)]:
        r = control_report(); mutation(r)
        with pytest.raises(ValueError):
            build.validate_controls(r, 'a' * 64)


def environment():
    return {'python': 'actual.version', 'numpy': '1.26.3', 'compiler': '11.4.0', 'cmake': 'cmake version 3.30.5',
            'packages': {**build.EXPECTED['installed_libraries'], 'libc6': 'existing'}}


def test_child_adds_exact_three_packages_parent_versions_untouched():
    parent = environment(); child = environment()
    child['packages'].update({p['package']: p['version'] for p in build.DEBS})
    result = build.compare_environment(parent, child)
    assert result['python'] == 'actual.version' and result['parent_existing_packages_unchanged']
    for mutation in [lambda x: x.update(python='invented'), lambda x: x['packages'].update(libc6='changed'),
                     lambda x: x['packages'].update(unexpected='new'), lambda x: x['packages'].update({'libgmp-dev': 'wrong'})]:
        candidate = json.loads(json.dumps(child)); mutation(candidate)
        with pytest.raises(ValueError):
            build.compare_environment(parent, candidate)


def test_output_seal_exclusive_modes_no_overwrite(tmp_path):
    p = tmp_path / 'report'; build.write_json(p, {'status': 'fail'})
    assert stat.S_IMODE(p.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        build.write_json(p, {'status': 'pass'})
    assert build.strict_json(p.read_bytes()) == {'status': 'fail'}


def test_source_binding_exact_canonical_full_immutable_closure(tmp_path, monkeypatch):
    revision = 'a' * 40
    root = tmp_path / 'root'; code = root / 'jobs' / revision / build.ENTRY / 'code'
    for name, raw in [('infra/certified_solid_build.py', b'own code'), ('infra/certified_solid_query.cpp', b'native'),
                      ('src/world_reward/__init__.py', b''), (build.CONFIG, json.dumps(build.EXPECTED).encode())]:
        p = code / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw); p.chmod(0o444)
    for name, raw in [('revision', revision + '\n'), ('source-sha256', 'b' * 64 + '\n')]:
        p = code.parent / name; p.write_text(raw); p.chmod(0o444)
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    monkeypatch.setattr(build, 'ROOT', root); monkeypatch.setattr(build, '__file__', str(code / 'infra/certified_solid_build.py'))
    report = build.source_binding(code, revision)
    assert report['files']['src/world_reward/__init__.py']['bytes'] == 0
    assert report['markers']['revision']['bytes'] == 41
    with pytest.raises(ValueError): build.source_binding(code, 'f' * 40)
    code.chmod(0o755)
    with pytest.raises(ValueError): build.source_binding(code, revision)


def test_shell_exact_entrypoint_markers_and_complete_real_closure():
    import importlib.util
    spec = importlib.util.spec_from_file_location('wr_azure_bundle', ROOT / 'infra/azure_job.py')
    azure = importlib.util.module_from_spec(spec); spec.loader.exec_module(azure)
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for folder in ('infra', 'src/world_reward', 'configs')
             for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.cpp', '.sh', '.json')}
    files['pyproject.toml'] = (ROOT / 'pyproject.toml').read_bytes()
    paths = azure.runtime_bundle_paths(files, 'infra/run_certified_solid_build.sh')
    for expected in ('infra/certified_solid_build.py', 'infra/certified_solid_query.cpp', 'infra/certified_solid_controls.py',
                     'src/world_reward/oriented_solid_forest.py', build.CONFIG):
        assert expected in paths
    shell = (ROOT / 'infra/run_certified_solid_build.sh').read_text()
    assert 'docker.sock' in shell and '(( $# != 0 ))' in shell
    assert '[[ $# == 1 && "$1" == --reuse-qualified-runtime ]]' in shell
    source = (ROOT / 'infra/certified_solid_build.py').read_text()
    assert 'code.parent / "revision"' in source and 'code.parent / "source-sha256"' in source
    assert 'FROM ' in source and 'docker image tag' not in source
    assert "'DOCKER_BUILDKIT=0'" in source
    assert "'--network', 'none'" in source and "'--user', '1000:1000'" in source
    assert '--gpus' not in source and 'validation/' not in shell


def native_fixture(tmp_path, monkeypatch):
    work = tmp_path / 'work'; work.mkdir(mode=0o700)
    monkeypatch.setattr(build.sys, 'platform', 'linux')
    monkeypatch.setattr(build.os, 'getuid', lambda: 1000)
    real_stat = Path.stat
    def local_stat(p, *args, **kwargs):
        value = real_stat(p, *args, **kwargs)
        if p == work:
            return SimpleNamespace(st_uid=1000, st_mode=value.st_mode)
        return value
    monkeypatch.setattr(Path, 'stat', local_stat)
    monkeypatch.setenv('WR_CPU_IMAGE_ID', 'sha256:' + 'a' * 64)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1'); monkeypatch.setenv('WR_NATIVE_NETWORK', 'none')
    evidence = {'native_source': {'bytes': 1, 'sha256': 'a' * 64}}
    monkeypatch.setattr(build, 'source_binding', lambda *_: evidence)
    return work, evidence


def test_native_flow_compile_once_qualified_only_after_fifteen(monkeypatch, tmp_path):
    import sys
    work, evidence = native_fixture(tmp_path, monkeypatch)
    calls = []
    def compile(argv, seconds, log):
        calls.append(argv); assert 0 < seconds <= 600
        (work / 'compiled-query').write_bytes(b'fakequalifiedbinary')
        log.write_bytes(b'')
    monkeypatch.setattr(build, 'run', compile)
    monkeypatch.setitem(sys.modules, 'certified_solid_controls', SimpleNamespace(run_controls=lambda *_: control_report(), ControlError=ValueError))
    assert build.native(tmp_path, 'b' * 40, work, tmp_path / 'headers') == 0
    r = build.strict_json((work / 'native.json').read_bytes())
    assert len(calls) == 1 and '-frounding-math' in calls[0]
    assert '-DWR_SOLID_QUERY_SOURCE_SHA256="' + 'a' * 64 + '"' in calls[0]
    assert r['qualified_controls'] == 15 and r['source_binding_after'] == evidence
    assert r['source_rehashed_after'] and r['status'] == 'pass'
    assert stat.S_IMODE((work / 'native.json').lstat().st_mode) == 0o444


def test_native_compile_failure_preserves_bounded_reason_and_source(monkeypatch, tmp_path):
    work, _ = native_fixture(tmp_path, monkeypatch)
    def fail(_argv, _seconds, log):
        log.write_bytes(b'x' * 1500 + b'compiler error')
        raise ValueError('command failed')
    monkeypatch.setattr(build, 'run', fail)
    assert build.native(tmp_path, 'b' * 40, work, tmp_path / 'headers') == 1
    r = build.strict_json((work / 'native.json').read_bytes())
    assert r['phase'] == 'compile' and r['status'] == 'fail'
    assert len(r['compiler_failure_tail'].encode()) == 1000
    assert r['compiler_failure_tail'].endswith('compiler error') and r['source_rehashed_after']
    assert 'qualified_controls' not in r


def test_cleanup_requires_exact_owned_cid_image_label(monkeypatch, tmp_path):
    cid = tmp_path / 'cid'; cid.write_text('c' * 64)
    record = {'Id': 'c' * 64, 'Name': '/own', 'Image': 'sha256:' + 'a' * 64,
              'Config': {'Labels': {'world_reward.certified_solid.owner': 'b' * 40}}}
    calls = []
    def invoke(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps([record]).encode(), stderr=b'')
    monkeypatch.setattr(build.subprocess, 'run', invoke)
    build.cleanup_container('own', 'sha256:' + 'a' * 64, 'b' * 40, cid)
    assert [c[1] for c in calls] == ['container', 'kill', 'rm']
    calls.clear(); record['Image'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError): build.cleanup_container('own', 'sha256:' + 'a' * 64, 'b' * 40, cid)
    assert len(calls) == 1  # No foreign kill/remove.

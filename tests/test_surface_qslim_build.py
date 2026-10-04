"""Tiny source/header ledgers and command spies; no compiler/Docker/cloud calls."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'infra'))
import surface_qslim_build as gate
import certified_solid_build as build


def write(path, raw=b'owned fixture', mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(mode)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source(tmp_path, monkeypatch):
    root = tmp_path/'root'; revision = 'a'*40; code = root/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS: write(code/name)
    write(code/'src/world_reward/__init__.py', b'')
    write(code.parent/'revision', (revision+'\n').encode()); write(code.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', root); monkeypatch.setattr(gate, '__file__', str(code/gate.HELPERS[0]))
    return root, code, revision


def test_complete_current_source_and_markers_no_historic_entry_spoof(tmp_path, monkeypatch):
    _, code, revision = source(tmp_path, monkeypatch)
    proof = gate.binding(code, revision, build)
    assert set(proof['helpers']) == set(gate.HELPERS) and proof['source_files'] == 6
    assert proof == gate.binding(code, revision, build)
    with pytest.raises(ValueError): gate.binding(code, 'c'*40, build)
    with pytest.raises(ValueError): gate.binding(code.parent/'run_volume_qem_build/code', revision, build)


@pytest.mark.parametrize('mutation', ['file_mode', 'dir_mode', 'marker', 'symlink', 'hardlink', 'extra_snapshot'])
def test_source_mutation_aliases_or_namespace_fail_without_repair(tmp_path, monkeypatch, mutation):
    _, code, revision = source(tmp_path, monkeypatch); p = code/gate.CPP
    if mutation == 'file_mode': p.chmod(0o644)
    elif mutation == 'dir_mode': p.parent.chmod(0o755)
    elif mutation == 'marker': write(code.parent/'revision', ('c'*40+'\n').encode())
    elif mutation == 'symlink':
        p.parent.chmod(0o755); p.unlink(); p.symlink_to(code/gate.HELPERS[0]); p.parent.chmod(0o555)
    elif mutation == 'hardlink': os.link(p, tmp_path/'foreign')
    else: write(code.parent/'unknown')
    with pytest.raises(ValueError): gate.binding(code, revision, build)


def compile_fixture(tmp_path, monkeypatch):
    code, work = tmp_path/'code', tmp_path/'work'; work.mkdir()
    pin = write(code/gate.CPP, b'new standalone source')
    expected = dict(stage='surface_qslim_build_info_v1', source_sha256=pin['sha256'], libigl_revision=gate.LIBIGL,
        eigen_revision=gate.EIGEN, target_vertices=4096, target_faces=4096, boundary_policy='fixed_original_vertices',
        intersection_blocking='upstream_floating_point', prepared_only=True, adoption=False)
    calls = []; monkeypatch.setattr(gate.shutil, 'which', lambda _: '/existing/c++')
    def run(argv, seconds, log=None):
        calls.append((argv, seconds, log))
        if argv[-1] == '--version': return b'qualified compiler\n'
        if argv[-1] == '--build-info': return json.dumps(expected).encode()
        write(Path(argv[-1]), b'fake native executable never run', 0o755); write(log, b'fake compile log')
    monkeypatch.setattr(build, 'run', run)
    return code, work, {'compiler': 'qualified compiler\n'}, expected, calls


def test_only_compile_and_buildinfo_no_qem_install_cgal_or_mesh_call(tmp_path, monkeypatch):
    code, work, old, expected, calls = compile_fixture(tmp_path, monkeypatch)
    report = gate.compile_binary(code, work, old, build, lambda: 599.)
    assert [c[0][-1] for c in calls] == ['--version', str(work/'surface_qslim'), '--build-info']
    command = calls[1][0]
    assert command[1:1+len(gate.FLAGS)] == list(gate.FLAGS)
    assert f'-DWR_SURFACE_QSLIM_SOURCE_SHA256="{expected["source_sha256"]}"' in command
    assert all(0 < c[1] <= 600 for c in calls)
    assert all('cgal' not in item.lower() and 'install' not in item for item in command)
    assert report['native_backend_qualified'] is report['adopted'] is False and report['qem_calls'] == 0
    assert report['build_info'] == expected and stat.S_IMODE((work/'surface_qslim').stat().st_mode) == 0o555


@pytest.mark.parametrize('mutation', ['compiler', 'missing', 'bool_type', 'budget', 'source'])
def test_compile_fail_closed_no_backend_fallback_or_retry(tmp_path, monkeypatch, mutation):
    code, work, old, expected, calls = compile_fixture(tmp_path, monkeypatch)
    if mutation == 'compiler': old['compiler'] = 'different'
    elif mutation == 'missing': monkeypatch.setattr(gate.shutil, 'which', lambda _: None)
    elif mutation == 'bool_type': expected['adoption'] = 0
    elif mutation == 'budget': monkeypatch.setattr(build, 'run', lambda *_: (_ for _ in ()).throw(TimeoutError()))
    else:
        original = build.run
        def mutate(argv, seconds, log=None):
            result = original(argv, seconds, log)
            if argv[-1] == '--build-info': write(code/gate.CPP, b'changed')
            return result
        monkeypatch.setattr(build, 'run', mutate)
    with pytest.raises((ValueError, TimeoutError)): gate.compile_binary(code, work, old, build, lambda: 599.)
    assert sum(c[2] is not None for c in calls) <= 1


def test_original_receipt_independently_hashed_before_json_no_fake_runtime(tmp_path, monkeypatch):
    root, code = tmp_path/'root', tmp_path/'code'; monkeypatch.setattr(gate, 'ROOT', root)
    original = dict(stage='world_reward_volume_qem_build', status='pass', image_id=gate.IMAGE,
                    libigl_revision=gate.LIBIGL, eigen_revision=gate.EIGEN, compiler='same', source_inventory_sha256='c'*64)
    pin = write(root/gate.BUILD_RECEIPT, json.dumps(original).encode(), 0o644)
    config = dict(original_image_id=gate.IMAGE, libigl_revision=gate.LIBIGL, eigen_revision=gate.EIGEN, original_build_receipt=pin)
    write(code/gate.PROTOCOL, json.dumps({'source_authentication': config}).encode())
    assert gate.prerequisite(code, build)[2]['receipt'] == pin
    write(root/gate.BUILD_RECEIPT, b'not JSON unchanged pin now fails', 0o644)
    with pytest.raises(ValueError, match='receipt'): gate.prerequisite(code, build)


def test_full_existing_headers_and_boost_inventory_not_arbitrary_new_sources(tmp_path, monkeypatch):
    base = tmp_path/'base'; boost = tmp_path/'boost'; monkeypatch.setattr(gate, 'BASE', base)
    actual_path = gate.Path
    monkeypatch.setattr(gate, 'Path', lambda value: boost if value == '/usr/include/boost' else actual_path(value))
    header = write(base/'source/libigl/include/igl/qslim.h', b'original source')
    write(base/'source/eigen/Eigen/Core', b'original Eigen')
    inventory = {p.relative_to(base/'source').as_posix(): build.identity(p)['sha256'] for p in sorted((base/'source').rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    authority = dict(status='pass', libigl_revision=gate.LIBIGL, eigen_revision=gate.EIGEN,
                     source_inventory_sha256=digest, pinned_primary_sha256={'libigl/include/igl/qslim.h': header['sha256']})
    authority_pin = write(base/'build.json', json.dumps(authority).encode())
    cpp_int = write(boost/'multiprecision/cpp_int.hpp', b'original Boost')
    old = dict(inherited_build_report_sha256=authority_pin['sha256'], source_inventory_sha256=digest)
    report = gate.header_environment({'boost_cpp_int_header_sha256': cpp_int['sha256']}, old, build, lambda: 500.)
    assert report['headers'] == 2 and report['boost_headers'] == 1
    write(base/'source/eigen/Eigen/Core', b'changed')
    with pytest.raises(ValueError): gate.header_environment({'boost_cpp_int_header_sha256': cpp_int['sha256']}, old, build, lambda: 500.)


def test_cleanup_requires_exact_cid_name_image_owner_and_daemon_errors_not_absence(tmp_path, monkeypatch):
    cid = tmp_path/'cid'; write(cid, b'c'*64)
    name, revision = 'wr-surface-qslim-build-'+('a'*40), 'a'*40
    row = dict(Name='/'+name, Image=gate.IMAGE, Id='c'*64, Config=dict(Labels={'world_reward.surface_qslim.owner': revision}))
    responses = [SimpleNamespace(returncode=0, stdout=json.dumps([row]).encode(), stderr=b''),
                 SimpleNamespace(returncode=0, stdout=b'', stderr=b''),
                 SimpleNamespace(returncode=1, stdout=b'[]\n', stderr=b'error: no such object: '+b'c'*64+b'\n')]
    calls = []
    def run(argv, **kwargs): calls.append(argv); return responses.pop(0)
    monkeypatch.setattr(gate.subprocess, 'run', run)
    gate.cleanup(name, revision, cid, lambda: 40.)
    assert calls[1] == ['docker', 'rm', '-f', 'c'*64]
    assert calls[0] == calls[2] == ['docker', 'container', 'inspect', 'c'*64]
    responses[:] = [SimpleNamespace(returncode=1, stdout=b'', stderr=b'Cannot connect to daemon')]
    with pytest.raises(ValueError, match='exact CID absence'): gate.cleanup(name, revision, cid, lambda: 40.)
    responses[:] = [SimpleNamespace(returncode=0, stdout=json.dumps([row | {'Image': 'foreign'}]).encode(), stderr=b'')]
    with pytest.raises(ValueError, match='foreign'): gate.cleanup(name, revision, cid, lambda: 40.)


@pytest.mark.parametrize('failure', ['wrong_id', 'substring', 'unexpected_stdout', 'hardlink', 'mutation'])
def test_cleanup_only_exact_cid_absence_and_unchanged_single_link_control(tmp_path, monkeypatch, failure):
    cid = tmp_path/'cid'; write(cid, b'c'*64)
    error = b'error: no such object: '+b'c'*64
    result = SimpleNamespace(returncode=1, stdout=b'', stderr=error)
    if failure == 'wrong_id': result.stderr = b'error: no such object: '+b'd'*64
    elif failure == 'substring': result.stderr = b'daemon outage; '+error
    elif failure == 'unexpected_stdout': result.stdout = b'foreign JSON'
    elif failure == 'hardlink': os.link(cid, tmp_path/'alias')
    def run(*_, **__):
        if failure == 'mutation': write(cid, b'd'*64)
        return result
    monkeypatch.setattr(gate.subprocess, 'run', run)
    with pytest.raises(ValueError): gate.cleanup('own', 'a'*40, cid, lambda: 40.)


def test_native_deadline_includes_posthash_and_receipt_fsync(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); work = tmp_path/'work'; work.mkdir(mode=0o700)
    monkeypatch.setattr(gate.sys, 'platform', 'linux')
    # Avoid pretending the real native UID is present on this local machine.
    real_stat = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda self, *a, **k: SimpleNamespace(st_uid=1000, st_mode=0o40700) if self == work else real_stat(self, *a, **k))
    monkeypatch.setattr(gate.os, 'getuid', lambda: 1000)
    monkeypatch.setenv('WR_CPU_IMAGE_ID', gate.IMAGE); monkeypatch.setenv('WR_NATIVE_NETWORK', 'none')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    listdir = gate.os.listdir
    monkeypatch.setattr(gate.os, 'listdir', lambda path: ['lo'] if str(path) == '/sys/class/net' else listdir(path))
    monkeypatch.setattr(gate, 'prerequisite', lambda *_: ({}, {}, {}))
    monkeypatch.setattr(gate, 'header_environment', lambda *_: {})
    monkeypatch.setattr(gate, 'compile_binary', lambda *_: {})
    now = [0.]; monkeypatch.setattr(gate.time, 'monotonic', lambda: now[0])
    alarms = []; monkeypatch.setattr(gate.signal, 'alarm', lambda value: alarms.append(value))
    original = gate.seal
    def slow_seal(path, raw, mode=0o444):
        assert alarms[-1] == gate.NATIVE_SECONDS
        original(path, raw, mode); now[0] = gate.NATIVE_SECONDS+1
    monkeypatch.setattr(gate, 'seal', slow_seal)
    with pytest.raises(ValueError, match='deadline'): gate.native(code, revision, work, build)
    assert alarms == [gate.NATIVE_SECONDS, 0]


def test_declared_network_none_does_not_replace_actual_native_isolation(tmp_path, monkeypatch):
    work = tmp_path/'work'; work.mkdir(mode=0o700)
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 1000)
    real_stat = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda self, *a, **k: SimpleNamespace(st_uid=1000, st_mode=0o40700) if self == work else real_stat(self, *a, **k))
    monkeypatch.setenv('WR_CPU_IMAGE_ID', gate.IMAGE); monkeypatch.setenv('WR_NATIVE_NETWORK', 'none')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1'); listdir = gate.os.listdir
    monkeypatch.setattr(gate.os, 'listdir', lambda path: ['lo', 'eth0'] if str(path) == '/sys/class/net' else listdir(path))
    with pytest.raises(ValueError, match='Actual native network'): gate.native(tmp_path, 'a'*40, work, build)


def test_owned_scratch_only_and_no_unknown_alias_cleanup(tmp_path):
    work = tmp_path/'work'; work.mkdir(); owner = (work.stat().st_dev, work.stat().st_ino)
    write(work/'surface_qslim'); write(work/'unknown')
    with pytest.raises(ValueError): gate.remove_work(work, owner)
    assert (work/'unknown').exists()
    (work/'unknown').unlink(); gate.remove_work(work, owner); assert not work.exists()


def test_host_argv_offline_narrow_source_receipt_work_and_fail_cleanup_first(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); (root/'results').mkdir()
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None); monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate, 'prerequisite', lambda *_: ({}, {}, {'pin': 'same'}))
    monkeypatch.setattr(gate, 'image_identity', lambda *_: {'image_id': gate.IMAGE})
    calls = []; cleaned = []
    def run(argv, seconds, log=None):
        calls.append(argv)
        if argv[1] == 'ps': return b''
        write(log, b'bounded fake CLI failure'); raise TimeoutError('fake command')
    monkeypatch.setattr(build, 'run', run)
    monkeypatch.setattr(gate, 'cleanup', lambda *args: cleaned.append(args[:3]))
    assert gate.host(code, revision, build) == 1
    command = calls[-1]
    for option in ('--network', '--read-only', '--cap-drop', '--security-opt', '--user', '--cpus', '--memory'):
        assert option in command
    assert '--gpus' not in command and command[command.index('--entrypoint')+1] == '/usr/bin/env'
    mounts = [command[i+1] for i, x in enumerate(command) if x == '--mount']
    assert len(mounts) == 3 and str(code.parent) in mounts[0] and gate.BUILD_RECEIPT in mounts[1]
    assert all('validation' not in item and 'vendor' not in item for item in mounts)
    assert cleaned and not (root/'results'/('surface-qslim-build-'+revision)/'disposable').exists()
    report = json.loads((root/'results'/('surface-qslim-build-'+revision)/'report.json').read_bytes())
    assert report['status'] == 'fail' and report['owned_container_removed'] is True


def test_wrapper_strict_entry_literal_closure_and_syntax_no_network_or_install():
    script = REPO/'infra/run_surface_qslim_build.sh'; text = script.read_text()
    assert subprocess.run(['bash', '-n', str(script)], capture_output=True).returncode == 0
    assert '[[ $# == 0 ]]' in text and 'run_surface_qslim_build/code' in text
    for name in ('/infra/surface_qslim_build.py', '/infra/surface_qslim.cpp'): assert name in text
    assert 'DOCKER_HOST="unix://$ROOT/docker.sock"' in text
    assert '960s' in text and 'python3 -I -B' in text and ' pip ' not in text and 'curl ' not in text


def test_actual_static_dispatch_closure_contains_cpp_and_host_helper():
    import importlib.util
    spec = importlib.util.spec_from_file_location('surface_build_bundle_test', REPO/'infra/azure_job.py')
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for folder in ('infra', 'src', 'configs')
             for p in (REPO/folder).rglob('*') if p.is_file() and not p.is_symlink()
             and p.suffix in ('.py', '.sh', '.cpp', '.hpp', '.h', '.json')}
    files['pyproject.toml'] = (REPO/'pyproject.toml').read_bytes()
    closure = launcher.runtime_bundle_paths(files, 'infra/run_surface_qslim_build.sh')
    assert set(gate.HELPERS) <= set(closure)
    assert all(not p.startswith(('tests/', 'validation/', 'weights/')) for p in closure)


def test_native_pass_host_scope_boolean_and_source_proof_cannot_be_relabelled(tmp_path, monkeypatch):
    code, work, old, _, _ = compile_fixture(tmp_path, monkeypatch)
    measured = gate.compile_binary(code, work, old, build, lambda: 599.)
    before = {'helpers': {gate.CPP: measured['source']}}
    report = dict(stage='surface_qslim_build_native_v1', status='pass', phase='complete', image_id=gate.IMAGE,
                  gpu_used=False, mesh_calls=0, qem_calls=0, native_backend_qualified=False, adopted=False,
                  source_rehashed_after=True, headers_rehashed_after=True, source_binding=before,
                  elapsed_seconds=1., build=measured)
    gate.validate_native(report, before)
    for change in ({'gpu_used': 0}, {'mesh_calls': True}, {'qem_calls': False}, {'elapsed_seconds': True},
                   {'elapsed_seconds': 601.}, {'phase': 'compile'}, {'native_backend_qualified': True},
                   {'source_binding': {}}):
        with pytest.raises(ValueError): gate.validate_native(report | change, before)


def test_native_failure_receipt_or_parse_error_does_not_skip_owned_container_cleanup(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); (root/'results').mkdir()
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None); monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate, 'prerequisite', lambda *_: ({}, {}, {})); monkeypatch.setattr(gate, 'image_identity', lambda *_: {})
    cleaned = []
    def run(argv, *_):
        if argv[1] == 'ps': return b''
        out = root/'results'/('surface-qslim-build-'+revision)
        write(out/'disposable/native.json', b'{bad JSON')
        return b''
    monkeypatch.setattr(build, 'run', run); monkeypatch.setattr(gate, 'cleanup', lambda *_: cleaned.append(True))
    assert gate.host(code, revision, build) == 1 and cleaned == [True]
    assert not (root/'results'/('surface-qslim-build-'+revision)/'disposable').exists()


def successful_host(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); (root/'results').mkdir()
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None); monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate, 'prerequisite', lambda *_: ({}, {}, {'pin': 'unchanged'}))
    monkeypatch.setattr(gate, 'image_identity', lambda *_: {'image_id': gate.IMAGE})
    out = root/'results'/('surface-qslim-build-'+revision); cleaned = []
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        before = gate.binding(code, revision, build)
        binary = write(out/'disposable/surface_qslim', b'fake build never executed', 0o555)
        measured = dict(source=before['helpers'][gate.CPP], binary=binary, flags=list(gate.FLAGS),
            build_info=gate.expected_info(before['helpers'][gate.CPP]['sha256']), qem_calls=0,
            native_backend_qualified=False, adopted=False)
        report = dict(stage='surface_qslim_build_native_v1', status='pass', phase='complete', image_id=gate.IMAGE,
            gpu_used=False, mesh_calls=0, qem_calls=0, native_backend_qualified=False, adopted=False,
            source_rehashed_after=True, headers_rehashed_after=True, source_binding=before, elapsed_seconds=1., build=measured)
        write(out/'disposable/native.json', json.dumps(report).encode()); write(log, b'', 0o644)
        write(out/'.container.cid', b'c'*64, 0o644); return b''
    monkeypatch.setattr(build, 'run', run); monkeypatch.setattr(gate, 'cleanup', lambda *_: cleaned.append(True))
    return code, revision, out, cleaned


def test_host_retains_only_source_bound_build_artifacts_sealed_and_rejects_reuse(tmp_path, monkeypatch):
    code, revision, out, cleaned = successful_host(tmp_path, monkeypatch)
    assert gate.host(code, revision, build) == 0 and cleaned == [True]
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'pass' and report['owned_scratch_removed'] is True
    assert report['retained_binary'] == build.identity(out/'surface_qslim', readonly=True)
    assert report['source_rehashed_after'] is report['runtime_rehashed_after'] is True
    assert set(p.name for p in out.iterdir()) == {'.container.cid', 'native.log', 'native.json', 'surface_qslim', 'report.json'}
    assert all(not p.stat().st_mode & 0o222 for p in (out, *out.iterdir()))
    frozen = (out/'report.json').read_bytes()
    with pytest.raises(ValueError, match='Fresh'): gate.host(code, revision, build)
    assert (out/'report.json').read_bytes() == frozen


@pytest.mark.parametrize('foreign', [False, True])
def test_publish_fsync_failure_rolls_back_only_owned_partial_not_replacement(tmp_path, monkeypatch, foreign):
    code, revision, out, _ = successful_host(tmp_path, monkeypatch); original = gate.os.fsync
    def fail(fd):
        target = out/'surface_qslim'
        if target.exists() and os.fstat(fd).st_ino == target.stat().st_ino:
            if foreign:
                target.unlink(); write(target, b'foreign replacement retained')
            raise OSError('manufactured publish failure')
        return original(fd)
    monkeypatch.setattr(gate.os, 'fsync', fail)
    assert gate.host(code, revision, build) == 1
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'fail' and 'retained_binary' not in report
    assert not (out/'disposable').exists()
    if foreign: assert (out/'surface_qslim').read_bytes() == b'foreign replacement retained'
    else: assert not (out/'surface_qslim').exists()


def test_host_final_receipt_sealing_remains_inside_inclusive_deadline(tmp_path, monkeypatch):
    code, revision, out, _ = successful_host(tmp_path, monkeypatch)
    clock = [0.]; monkeypatch.setattr(gate.time, 'monotonic', lambda: clock[0])
    alarms = []; monkeypatch.setattr(gate.signal, 'alarm', lambda value: alarms.append(value))
    original = gate.seal
    def slow_seal(path, raw, mode=0o444):
        if path.name == 'report.json':
            assert alarms[-1] > 0
            original(path, raw, mode); clock[0] = gate.HOST_SECONDS+1
        else: original(path, raw, mode)
    monkeypatch.setattr(gate, 'seal', slow_seal)
    with pytest.raises(ValueError, match='deadline'): gate.host(code, revision, build)
    assert alarms[-1] == 0

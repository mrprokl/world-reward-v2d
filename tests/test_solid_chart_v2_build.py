"""Tiny source ledgers and mocked commands; no compiler, model or Azure calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'infra'), str(REPO/'src')]
import solid_chart_v2_build as gate
import certified_solid_build as build
import mesh_conditioned_chart_v2_source as generator
import mesh_serialization_compile as inherited


def write(path, raw=b'owned tiny metadata', mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(mode)
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def source_fixture(tmp_path, monkeypatch):
    root = tmp_path/'root'; revision = 'a'*40
    code = root/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS: write(code/name, b'complete immutable code')
    write(code/'src/world_reward/__init__.py', b'')
    write(code.parent/'revision', (revision+'\n').encode())
    write(code.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', root)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/solid_chart_v2_build.py'))
    return root, code, revision


def test_exact_new_namespace_whole_source_and_markers(tmp_path, monkeypatch):
    _, code, revision = source_fixture(tmp_path, monkeypatch)
    result = gate.binding(code, revision, build)
    assert set(result['helpers']) == set(gate.HELPERS) and result['source_files'] == len(gate.HELPERS)+1
    assert result['producer_revision'] == revision and result['markers']['revision']['bytes'] == 41
    assert gate.binding(code, revision, build) == result
    with pytest.raises(ValueError): gate.binding(code, 'c'*40, build)
    with pytest.raises(ValueError): gate.binding(code.parent/'run_object_budget_volume'/'code', revision, build)


@pytest.mark.parametrize('change', ('writable', 'marker', 'symlink', 'hardlink'))
def test_source_or_marker_mutation_is_never_repaired(tmp_path, monkeypatch, change):
    _, code, revision = source_fixture(tmp_path, monkeypatch)
    p = code/gate.HELPERS[-1]
    if change == 'writable': p.chmod(0o644)
    elif change == 'marker': write(code.parent/'revision', ('c'*40+'\n').encode())
    elif change == 'symlink':
        p.parent.chmod(0o755); p.unlink(); p.symlink_to(code/gate.HELPERS[0])
    else:
        os.link(p, tmp_path/'foreign')
    with pytest.raises(ValueError): gate.binding(code, revision, build)


def compile_fixture(tmp_path, monkeypatch):
    code, work, volume = (tmp_path/n for n in ('code', 'work', 'volume'))
    work.mkdir()
    for name in ('infra/mesh_conditioned_qem.cpp', 'infra/mesh_conditioned_chart_v2.hpp',
                 'infra/mesh_serialization_qem.cpp', 'configs/mesh_conditioned_qem_protocol_v1.json',
                 'configs/mesh_serialization_compiler_protocol_v1.json'):
        write(code/name, (REPO/name).read_bytes())
    write(volume/'mesh_volume_qem.cpp', (REPO/'infra/mesh_volume_qem.cpp').read_bytes())
    monkeypatch.setattr(inherited, 'VOLUME', volume)
    monkeypatch.setattr(gate.shutil, 'which', lambda _: '/owned/c++')
    _, derivation = generator.derive_cached_backend(code/'infra/mesh_conditioned_qem.cpp', code/'infra/mesh_conditioned_chart_v2.hpp')
    source = inherited.conditioned_protocol(code)['source_authentication']
    old_info = dict(source_sha256=generator.BACKBONE_SHA256, physical_coordinate_cache=True, adopted=False,
        native_cost_and_placement_unchanged=False, new_numeric_algorithm=True, native_qslim_implementation_reused=True,
        volume_relative_limit=.05, target_faces=4096)
    info = old_info | dict(chart_version=2, origin_search_performed=False,
        backbone_source_sha256=generator.BACKBONE_SHA256, chart_header_sha256=derivation['header']['sha256'],
        chart_policy_sha256=derivation['compiler_macros']['WR_CONDITIONED_CHART_V2_POLICY_SHA256'],
        source_sha256=derivation['generated_source']['sha256'])
    calls = []
    def run(argv, seconds, log=None):
        calls.append((argv, seconds, log))
        if argv[-1] == '--version': return b'qualified compiler\n'
        if argv[-1] == '--build-info': return json.dumps(info).encode()
        write(Path(argv[-1]), b'fake native executable never executed', 0o755)
        write(log, b'bounded fake compiler output')
    monkeypatch.setattr(build, 'run', run)
    return code, work, dict(compiler='qualified compiler\n', build_info=old_info), info, calls, source


def test_compile_once_exact_prefixes_macros_no_mesh_call(tmp_path, monkeypatch):
    code, work, cache, info, calls, source = compile_fixture(tmp_path, monkeypatch)
    binary, result = gate.compile_binary(code, work, cache, build, lambda: 599.)
    assert [c[0][-1] for c in calls] == ['--version', str(binary), '--build-info']
    command = calls[1][0]
    for flag in ('-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off', '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY', '-DWR_CONDITIONED_CACHE=1'):
        assert flag in command
    assert all(0 < c[1] <= 600 for c in calls)
    assert f'-DWR_CONDITIONED_V2_GENERATED_SHA256="{info["source_sha256"]}"' in command
    assert result['build_info'] == info and result['native_backend_qualified'] is result['adopted'] is False
    assert stat.S_IMODE(binary.stat().st_mode) == 0o555
    assert (work/'wr_volume_core.hpp').read_bytes() == (inherited.VOLUME/'mesh_volume_qem.cpp').read_bytes().split(b'\nint main(int argc, char** argv) {')[0]
    assert hashlib.sha256((work/'wr_serialization_core.hpp').read_bytes()).hexdigest() == source['serialization_core_prefix_sha256']
    assert (work/'mesh_conditioned_chart_v2.hpp').read_bytes() == (code/'infra/mesh_conditioned_chart_v2.hpp').read_bytes()


@pytest.mark.parametrize('mutation', ('compiler', 'abi', 'prefix', 'timeout'))
def test_compile_fail_fast_without_other_backend_or_retries(tmp_path, monkeypatch, mutation):
    code, work, cache, info, calls, _ = compile_fixture(tmp_path, monkeypatch)
    if mutation == 'compiler': cache['compiler'] = 'different compiler\n'
    elif mutation == 'abi': info['physical_coordinate_cache'] = 1
    elif mutation == 'prefix': write(code/'infra/mesh_serialization_qem.cpp', b'wrong original\n')
    else:
        def fail(*_args, **_kwargs): raise TimeoutError('bounded command')
        monkeypatch.setattr(build, 'run', fail)
    with pytest.raises((ValueError, TimeoutError)):
        gate.compile_binary(code, work, cache, build, lambda: 599.)
    assert sum(c[2] is not None for c in calls) <= 1


def test_missing_existing_compiler_stops_no_install(tmp_path, monkeypatch):
    code, work, cache, _, calls, _ = compile_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gate.shutil, 'which', lambda _: None)
    with pytest.raises(ValueError, match='no install'):
        gate.compile_binary(code, work, cache, build, lambda: 599.)
    assert calls == []


def test_actual_cache_leaf_host_preflight_pins_and_modes_without_numpy(tmp_path, monkeypatch):
    root = tmp_path/'root'; code = root/'code'; revision = 'd'*40
    out = root/'results'/('mesh-conditioned-cache-'+revision)
    binary = write(out/'mesh_conditioned_qem', b'fake never run', 0o555)
    native = dict(status='pass', phase='complete', stage='mesh_conditioned_cache_native_v1', build=dict(binary=binary, build_info={}, compiler='same'))
    n = write(out/'native.json', json.dumps(native).encode())
    host = dict(status='pass', stage='mesh_conditioned_cache_host_v1', original_image_id=build.PARENT, native_identity=n, retained_binary=binary)
    h = write(out/'report.json', json.dumps(host).encode()); out.chmod(0o555)
    pins = dict(schema='world_reward.mesh_conditioned_cache_qualification.v1', producer_revision=revision,
                exact_implementation_regression_verified=True, host_report=h, native_report=n, retained_binary=binary)
    write(code/gate.CACHE_PINS, json.dumps(pins).encode()); monkeypatch.setattr(gate, 'ROOT', root)
    paths, proof = gate.cache_leaves(code, build)
    assert proof['artifacts'] == dict(host_report=h, native_report=n, retained_binary=binary)
    paths['retained_binary'].chmod(0o755)
    with pytest.raises(ValueError, match='mode'): gate.cache_leaves(code, build)


def test_current_image_is_cgal_child_historical_cache_parent_not_spoofed(monkeypatch):
    parent, child = build.PARENT, 'sha256:'+'a'*64
    layers = {parent: ['sha256:'+'b'*64], child: ['sha256:'+'b'*64, 'sha256:'+'c'*64]}
    def run(argv, _seconds):
        return argv[3].encode() if argv[-1] == '{{.Id}}' else json.dumps(layers[argv[3]]).encode()
    monkeypatch.setattr(build, 'run', run)
    pins = dict(parent_image_id=parent, child_image_id=child)
    result = gate.image_identity(pins, build, lambda: 100.)
    assert result['parent_id'] == parent and result['child_id'] == child and result['child_layers'] == 2
    layers[child][0] = 'sha256:'+'d'*64
    with pytest.raises(ValueError, match='inherit'): gate.image_identity(pins, build, lambda: 100.)


def host_fixture(tmp_path, monkeypatch):
    root, code, revision = source_fixture(tmp_path, monkeypatch)
    (root/'results').mkdir()
    proof = {'independently_qualified': True}
    pins = {'parent_image_id': build.PARENT, 'child_image_id': 'sha256:'+'c'*64}
    image = {'child_id': pins['child_image_id']}
    monkeypatch.setattr(gate.sys, 'platform', 'linux')
    monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None)
    monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate, 'qualified', lambda *_: (pins, (root/'tiny-qualified-leaf',), proof))
    monkeypatch.setattr(gate, 'image_identity', lambda *_: image)
    cleanup = []
    monkeypatch.setattr(build, 'cleanup_container', lambda *args: cleanup.append(args))
    calls = []
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        calls.append(argv)
        out = root/'results'/('solid-chart-v2-build-'+revision)
        binary = out/'disposable/mesh_conditioned_chart_v2'
        write(binary, b'fake never run', 0o555)
        evidence = gate.binding(code, revision, build)
        result = dict(stage='solid_chart_v2_build_native_v1', status='pass', phase='complete', source_binding=evidence,
            source_binding_after=evidence, qualified_inputs=proof, source_rehashed_after=True,
            qualified_inputs_rehashed_after=True, native_backend_qualified=False, adopted=False,
            gpu_used=False, gt_used=False, mesh_calls=0, elapsed_seconds=1., image_id=pins['child_image_id'],
            build=dict(binary=build.identity(binary, readonly=True), native_backend_qualified=False, adopted=False))
        write(out/'disposable/native.json', json.dumps(result).encode())
        write(log, b'owned stdout')
    monkeypatch.setattr(build, 'run', run)
    return root, code, revision, pins, calls, cleanup


def test_host_pass_seals_only_build_preserves_source_no_gpu(tmp_path, monkeypatch):
    root, code, revision, pins, calls, cleanup = host_fixture(tmp_path, monkeypatch)
    source = gate.binding(code, revision, build)
    assert gate.host(code, revision, build, None) == 0
    out = root/'results'/('solid-chart-v2-build-'+revision)
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'pass' and report['mesh_calls'] == 0
    assert report['native_backend_qualified'] is report['adopted'] is report['gpu_used'] is False
    assert report['source_binding_after'] == source and report['owned_scratch_removed'] and report['owned_container_removed']
    assert stat.S_IMODE(out.stat().st_mode) == 0o555
    assert stat.S_IMODE((out/'report.json').stat().st_mode) == 0o444
    assert stat.S_IMODE((out/'mesh_conditioned_chart_v2').stat().st_mode) == 0o555
    argv = calls[0]
    assert '--gpus' not in argv and argv[argv.index('--user')+1] == '1000:1000'
    assert argv[argv.index('--network')+1] == 'none' and '--read-only' in argv
    assert str(code.parent/'revision') in ' '.join(argv) and str(code.parent/'source-sha256') in ' '.join(argv)
    assert argv[argv.index('--entrypoint')+1] == '/usr/bin/env' and pins['child_image_id'] in argv
    assert cleanup[0][1:3] == (pins['child_image_id'], revision)
    with pytest.raises(ValueError, match='Fresh'): gate.host(code, revision, build, None)


@pytest.mark.parametrize('failure', ('cleanup', 'missing', 'source', 'nonzero'))
def test_host_failure_never_publishes_usable_binary(tmp_path, monkeypatch, failure):
    root, code, revision, _, _, _ = host_fixture(tmp_path, monkeypatch)
    run = build.run
    if failure == 'cleanup': monkeypatch.setattr(build, 'cleanup_container', lambda *_: (_ for _ in ()).throw(ValueError('foreign cleanup refused')))
    elif failure == 'missing': monkeypatch.setattr(build, 'run', lambda argv, *_: b'' if argv[1] == 'ps' else None)
    else:
        def alter(*args, **kwargs):
            if args[0][1] == 'ps': return b''
            run(*args, **kwargs)
            if failure == 'source': write(code.parent/'revision', ('b'*40+'\n').encode())
            else: raise ValueError('child exit nonzero')
        monkeypatch.setattr(build, 'run', alter)
    assert gate.host(code, revision, build, None) == 1
    out = root/'results'/('solid-chart-v2-build-'+revision)
    assert json.loads((out/'report.json').read_bytes())['status'] == 'fail'
    assert not (out/'mesh_conditioned_chart_v2').exists()


def test_foreign_container_fails_before_creating_output_or_cleanup(tmp_path, monkeypatch):
    root, code, revision, _, calls, cleanup = host_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(build, 'run', lambda *_: b'foreign-cid\n')
    with pytest.raises(ValueError, match='preexisting'):
        gate.host(code, revision, build, None)
    assert not (root/'results'/('solid-chart-v2-build-'+revision)).exists() and calls == cleanup == []


@pytest.mark.parametrize('blocked', (False, True))
def test_native_qualified_headers_before_one_compile_and_source_post(tmp_path, monkeypatch, blocked):
    import object_budget_conditioned as cache
    root, code, revision, pins, _, _ = host_fixture(tmp_path, monkeypatch)
    work = root/'results'/('solid-chart-v2-build-'+revision)/'disposable'; work.mkdir(parents=True, mode=0o700)
    stat_method = Path.stat
    def native_stat(path, *args, **kwargs):
        metadata = stat_method(path, *args, **kwargs)
        return SimpleNamespace(st_uid=1000, st_mode=metadata.st_mode) if path == work else metadata
    monkeypatch.setattr(Path, 'stat', native_stat)
    monkeypatch.setattr(gate.os, 'getuid', lambda: 1000)
    monkeypatch.setenv('WR_NATIVE_NETWORK', 'none'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('WR_CPU_IMAGE_ID', pins['child_image_id'])
    monkeypatch.setattr(gate, 'qualified', lambda *_: (pins, (), {'cache': {}}))
    monkeypatch.setattr(cache, 'cache_qualification', lambda *_: (root/'qualified-noexecute', {'original_runtime': {'frozen': True}}))
    monkeypatch.setattr(cache, 'runtime_identity', lambda *_: {'frozen': True})
    monkeypatch.setattr(inherited, 'conditioned_protocol', lambda *_: {})
    def original(*_args):
        if blocked: raise FileNotFoundError('Required immutable headers absent')
        return {'frozen': True}
    monkeypatch.setattr(inherited, 'original', original)
    calls = []
    def compile(*_args):
        calls.append('one compile only')
        return root/'new-unqualified-binary', {'native_backend_qualified': False, 'adopted': False}
    monkeypatch.setattr(gate, 'compile_binary', compile)
    assert gate.native(code, revision, work, build, None) == int(blocked)
    result = json.loads((work/'native.json').read_bytes())
    assert result['status'] == ('fail' if blocked else 'pass')
    assert len(calls) == int(not blocked) and result['source_rehashed_after'] is True
    assert result['mesh_calls'] == 0 and result['native_backend_qualified'] is result['adopted'] is False


def test_hostbootstrap_is_stdlib_only_and_shell_requires_exact_fresh_entry():
    raw = (REPO/'infra/solid_chart_v2_build.py').read_bytes()
    bootstrap = 'import runpy,sys;runpy.run_path(sys.argv[1],run_name="not_main");assert "numpy" not in sys.modules'
    subprocess.run([sys.executable, '-I', '-B', '-S', '-c', bootstrap, str(REPO/'infra/solid_chart_v2_build.py')], check=True)
    subprocess.run(['bash', '-n', str(REPO/'infra/run_solid_chart_v2_build.sh')], check=True)
    assert b'import numpy' not in raw and b'compile_binary' in raw
    shell = (REPO/'infra/run_solid_chart_v2_build.sh').read_text()
    assert 'run_solid_chart_v2_build/code' in shell and '[[ $# == 0 ]]' in shell
    assert 'run_object_budget_volume/code' not in shell and 'run_volume_qem_build/code' not in shell


def test_static_runtime_closure_includes_new_header_generator_without_fake_namespace():
    import azure_job
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for folder in ('infra', 'src', 'configs')
             for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    closure = azure_job.runtime_bundle_paths(files, 'infra/run_solid_chart_v2_build.sh')
    assert set(gate.HELPERS) <= set(closure)
    assert 'infra/oriented_solid_controls_v2.py' not in closure  # Build never observes geometry fixtures.

"""Manufactured source/transport controls only; no compiler, Docker or models."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

INFRA = Path(__file__).resolve().parents[1] / 'infra'
spec = importlib.util.spec_from_file_location('conditioned_compile_test', INFRA / 'mesh_serialization_compile.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
REVISION = 'a' * 40


def sealed(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o444)
    return path


def test_new_cli_is_exclusive_and_old_native_dispatch_retained(monkeypatch):
    calls = []
    monkeypatch.setenv('WR_CODE', '/immutable/code')
    monkeypatch.setenv('WR_CODE_REVISION', REVISION)
    monkeypatch.setattr(gate, 'host', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(gate, 'native', lambda *a, **k: calls.append((a, k)))
    gate.main(['--conditioned'])
    assert calls.pop()[1] == {'conditioned': True}
    gate.main(['--native-conditioned', '/immutable/code', REVISION, '/fresh/out'])
    assert calls.pop()[1] == {'conditioned': True}
    gate.main(['--native', '/immutable/code', REVISION, '/fresh/out'])
    assert calls.pop()[1] == {'geometry_phase': False}
    gate.main(['--geometry'])
    assert calls.pop()[1] == {'geometry_phase': True}
    for args in [['--conditioned', '--geometry'], ['--conditioned', '--serialization'],
                 ['--native-conditioned'], ['--conditioned', '--anything']]:
        with pytest.raises(ValueError):
            gate.main(args)


def test_shell_new_branch_only_one_flag_and_socket_before_exec():
    text = (INFRA / 'run_volume_qem_build.sh').read_text()
    branch = text.split('if [[ "${1:-}" == --conditioned ]]; then')[1].split('\nfi')[0]
    assert '(( $# == 1 )) || exit 2' in branch
    assert branch.index('export DOCKER_HOST="unix://${WR_ROOT:?}/docker.sock"') < branch.index('exec python3')
    assert 'infra/mesh_serialization_compile.py" --conditioned' in branch
    assert 'if [[ "${1:-}" == --serialization ]]; then' in text


def protocol_fixture(code):
    for name in [gate.PROTOCOL, gate.CONDITIONED_PROTOCOL, gate.CPP]:
        sealed(code/name, (INFRA.parent/name).read_bytes())
    return gate.conditioned_protocol(code)


def test_conditioned_protocol_preserves_original_auth_and_prefix(tmp_path):
    config = protocol_fixture(tmp_path/'code')
    raw = (INFRA / 'mesh_serialization_qem.cpp').read_bytes()
    marker = b'\nint main(int argc,char** argv) {'
    assert raw.count(marker) == 1
    prefix = raw[:raw.index(marker)]
    assert len(prefix) == 14676
    assert hashlib.sha256(prefix).hexdigest() == config['source_authentication']['serialization_core_prefix_sha256']
    assert config['native_calls_maximum'] == 8 and config['native_seconds_per_call'] == 600
    assert config['algorithm']['native_cost_and_placement_unchanged'] is False
    assert config['chart']['physical_geometry_rescaled'] is False


def compile_fixture(tmp_path, monkeypatch):
    code, scratch, volume = (tmp_path/n for n in ('code', 'scratch', 'volume'))
    scratch.mkdir()
    sealed(volume/'mesh_volume_qem.cpp', (INFRA/'mesh_volume_qem.cpp').read_bytes())
    sealed(code/gate.CONDITIONED_CPP, b'// tiny unexecuted own source\n')
    monkeypatch.setattr(gate, 'VOLUME', volume)
    monkeypatch.setattr(gate.shutil, 'which', lambda _: '/never-executed/c++')
    config = protocol_fixture(code)
    pins = config['source_authentication']
    info = dict(source_sha256=gate.identity(code/gate.CONDITIONED_CPP)['sha256'],
        serialization_source_sha256=pins['original_serialization_cpp_sha256'],
        serialization_core_prefix_sha256=pins['serialization_core_prefix_sha256'],
        volume_core_prefix_sha256=pins['derived_core_prefix']['sha256'],
        volume_source_sha256=pins['original_volume_cpp_sha256'], base_source_sha256=pins['original_base_cpp_sha256'],
        native_cost_and_placement_unchanged=False, new_numeric_algorithm=True,
        native_qslim_implementation_reused=True, cost_normalization=True, physical_geometry_rescaled=False,
        volume_relative_limit=.05, adopted=False, block_intersections=True, target_faces=4096,
        libigl_revision=pins['libigl_revision'], eigen_revision=pins['eigen_revision'])
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        path = Path(command[-1]); path.write_bytes(b'not a native binary'); path.chmod(0o755)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(gate.subprocess, 'run', child)
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda args, **k:
        'fake compiler version' if args[-1] == '--version' else json.dumps(info).encode())
    return code, scratch, config, info, calls


def test_conditioned_compile_byte_exact_prefixes_flags_and_truthful_abi(tmp_path, monkeypatch):
    code, scratch, config, _, calls = compile_fixture(tmp_path, monkeypatch)
    binary, report = gate.compile_binary(code, scratch, config, lambda: 1000, conditioned=True)
    assert binary.name == 'mesh_conditioned_qem' and len(calls) == 1
    command, kwargs = calls[0]
    assert kwargs['timeout'] == 600 and str(code/gate.CONDITIONED_CPP) in command
    for flag in ['-fno-fast-math', '-ffp-contract=off', '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY']:
        assert flag in command
    assert any(x.startswith('-DWR_CONDITIONED_SOURCE_SHA256=') for x in command)
    assert any(x.startswith('-DWR_SERIALIZATION_CORE_PREFIX_SHA256=') for x in command)
    raw = (code/gate.CPP).read_bytes()
    assert (scratch/'wr_serialization_core.hpp').read_bytes() == raw[:raw.index(b'\nint main(int argc,char** argv) {')]
    assert report['derived_serialization_core']['bytes'] == 14676
    assert report['build_info']['native_cost_and_placement_unchanged'] is False
    assert report['build_info']['physical_geometry_rescaled'] is False


@pytest.mark.parametrize('field,value', [('native_cost_and_placement_unchanged', True),
    ('physical_geometry_rescaled', True), ('new_numeric_algorithm', False), ('cost_normalization', False),
    ('serialization_core_prefix_sha256', '0'*64), ('native_qslim_implementation_reused', False)])
def test_conditioned_abi_changes_fail_closed(tmp_path, monkeypatch, field, value):
    code, scratch, config, info, _ = compile_fixture(tmp_path, monkeypatch)
    info[field] = value
    with pytest.raises(ValueError, match='ABI differs'):
        gate.compile_binary(code, scratch, config, lambda: 1000, conditioned=True)


def test_serialization_prefix_tamper_fails_before_compile(tmp_path, monkeypatch):
    code, scratch, config, _, calls = compile_fixture(tmp_path, monkeypatch)
    config['source_authentication']['serialization_core_prefix_sha256'] = '0'*64
    with pytest.raises(ValueError, match='prefix differs'):
        gate.compile_binary(code, scratch, config, lambda: 1000, conditioned=True)
    assert not calls


def test_current_source_adds_conditioned_closure_but_historical_stays_original(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    code = tmp_path/'jobs'/REVISION/gate.ENTRY/'code'
    code.mkdir(parents=True)
    for name in gate.HELPERS + gate.CONDITIONED_HELPERS:
        sealed(code/name, b'not executed\n')
    sealed(code/'src/world_reward/__init__.py', b'')
    for name, raw in [('revision', (REVISION+'\n').encode()), ('source-sha256', ('b'*64+'\n').encode())]:
        sealed(code.parent/name, raw)
    for p in [*sorted(code.rglob('*'), reverse=True), code]:
        if p.is_dir(): p.chmod(0o555)
    monkeypatch.setattr(gate, '__file__', str(code/gate.HELPERS[1]))
    new = gate.source(code, REVISION, conditioned=True)
    old = gate.source(code, REVISION, historical=True)
    assert set(new['helpers']) == set(gate.HELPERS + gate.CONDITIONED_HELPERS)
    assert set(old['helpers']) == set(gate.HELPERS)
    assert new['source_files_sha256'] == old['source_files_sha256']


def native_fixture(tmp_path, monkeypatch, *, fail=False):
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate.sys, 'platform', 'linux')
    monkeypatch.setattr(gate.os, 'geteuid', lambda: 0)
    temporary_directory = gate.tempfile.TemporaryDirectory
    monkeypatch.setattr(gate.tempfile, 'TemporaryDirectory', lambda **k:
        temporary_directory(prefix=k['prefix'], dir=tmp_path))
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda p: iter([Path('lo')]) if str(p) == '/sys/class/net' else original_iterdir(p))
    code = tmp_path/'code'; out = tmp_path/'results'/('mesh-conditioned-qem-'+REVISION); out.mkdir(parents=True)
    for name in [gate.CPP, gate.PROTOCOL, gate.CONDITIONED_PROTOCOL]: sealed(code/name, b'{}\n')
    prior = {'frozen': True}
    phase1 = dict(original_runtime=prior, source_cpp=gate.identity(code/gate.CPP), protocol=gate.identity(code/gate.PROTOCOL))
    sealed(code/gate.PHASE1_PINS, json.dumps(phase1).encode())
    calls = []
    monkeypatch.setattr(gate, 'source', lambda *a, **k: calls.append(('source', k)) or {'source': True})
    monkeypatch.setattr(gate, 'conditioned_protocol', lambda _: {'source_authentication': {'position_weld_policy_sha256': 'policy'}})
    monkeypatch.setattr(gate, 'original', lambda *a: prior)
    def compile(*args, **kwargs):
        calls.append(('compile', kwargs)); binary = args[1]/'new'; binary.write_bytes(b'not executed')
        return binary, {'binary': gate.identity(binary, readonly=False)}
    monkeypatch.setattr(gate, 'compile_binary', compile)
    monkeypatch.setattr(gate, 'parity', lambda *a: {'cases': 128, 'mismatches': 0, 'policy_sha256': 'policy'})
    monkeypatch.setattr(gate, 'orientation_parity', lambda *a: {'cases': 128, 'mismatches': 0})
    monkeypatch.setattr(gate, 'controls', lambda *a: {'key_rounding_cases': 7})
    class Error(RuntimeError):
        def __init__(self, report): self.report = report
    def geometry(*a, **k):
        calls.append(('new_geometry', k))
        report = {'stage': 'mesh_conditioned_geometry_controls_v1', 'status': 'fail' if fail else 'pass'}
        if fail: raise Error(report)
        return report
    monkeypatch.setitem(sys.modules, 'mesh_conditioned_geometry', SimpleNamespace(GeometryControlError=Error, geometry_controls=geometry))
    monkeypatch.setenv('WR_PHASE1_DEADLINE', str(time.monotonic()+1000))
    return code, out, calls


def test_native_conditioned_runs_new_geometry_after_delegated_scalar_controls(tmp_path, monkeypatch):
    code, out, calls = native_fixture(tmp_path, monkeypatch)
    gate.native(code, REVISION, out, conditioned=True)
    report = json.loads((out/'native.json').read_bytes())
    assert report['status'] == 'pass' and report['stage'] == 'mesh_conditioned_qem_native_v1'
    assert report['predicate_parity_only'] is False and report['physical_geometry_rescaled'] is False
    assert report['parity']['cases'] == report['orientation_parity']['cases'] == 128
    assert report['controls']['key_rounding_cases'] == 7 and report['geometry']['status'] == 'pass'
    assert [k for k, _ in calls] == ['source', 'compile', 'new_geometry', 'source']
    assert calls[1][1] == {'conditioned': True} and calls[0][1] == calls[-1][1] == {'conditioned': True}
    assert report['owned_scratch_removed'] and report['originals_rehashed_after'] and not report['adoption']


def test_new_geometry_failure_retains_partial_and_never_becomes_pass(tmp_path, monkeypatch):
    code, out, _ = native_fixture(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='gate failed'):
        gate.native(code, REVISION, out, conditioned=True)
    report = json.loads((out/'native.json').read_bytes())
    assert report['status'] == report['geometry']['status'] == 'fail'
    assert report['phase'] == 'conditioned_geometry_controls' and report['originals_rehashed_after']


def test_modes_cannot_share_historical_geometry_or_results(tmp_path):
    for fn in [gate.host, gate.native]:
        args = (tmp_path, REVISION) if fn is gate.host else (tmp_path, REVISION, tmp_path)
        with pytest.raises(ValueError, match='exclusive'):
            fn(*args, conditioned=True, geometry_phase=True)


def host_fixture(tmp_path, monkeypatch, *, malformed=False):
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate.sys, 'platform', 'linux')
    monkeypatch.setattr(gate.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'uname', lambda: SimpleNamespace(nodename='scenesmith-ncc-h100-01'))
    code = tmp_path/'code'; (tmp_path/'results').mkdir()
    for name in [gate.PROTOCOL, gate.CONDITIONED_PROTOCOL]: sealed(code/name, b'{}\n')
    build = sealed(tmp_path/'results/image-volume-qem.json', b'frozen old build\n')
    config = dict(source_authentication={'original_image_id': 'sha256:'+'b'*64,
        'original_build_receipt': gate.identity(build), 'position_weld_policy_sha256': 'policy'})
    binding = {'whole_source': True}; source_calls = []; commands = []
    monkeypatch.setattr(gate, 'source', lambda *a, **k: source_calls.append(k) or binding)
    monkeypatch.setattr(gate, 'conditioned_protocol', lambda _: config)
    monkeypatch.setattr(gate, 'phase1_qualification', lambda _: {'actual_phase1': True})
    monkeypatch.setattr(gate, 'technical_replay', lambda _: pytest.fail('New algorithm is not a replay'))
    monkeypatch.setattr(gate, 'geometry_protocol', lambda _: pytest.fail('Closed geometry cannot be rerun'))
    monkeypatch.setattr(gate.signal, 'signal', lambda *a: None)
    cid = 'c'*64
    def subprocess_stub(command, **kwargs):
        commands.append(command)
        assert command[0] == 'docker'
        if command[1] == 'run':
            Path(command[command.index('--cidfile')+1]).write_text(cid)
            out = tmp_path/'results'/('mesh-conditioned-qem-'+REVISION)
            geometry = dict(stage='mesh_conditioned_geometry_controls_v1', status='pass',
                sources_rehashed_after=True, owned_scratch_removed=True, adoption=False,
                maximum_native_calls=8, native_budget_seconds=600,
                paired_fixtures=[{'comparisons': [{'method': 'original', 'status': 'fail'},
                    {'method': 'conditioned', 'status': 'pass', 'committed_collapses': 1}]} for _ in range(4)])
            if malformed: geometry['paired_fixtures'][-1]['comparisons'][1]['status'] = 'fail'
            native = dict(stage='mesh_conditioned_qem_native_v1', status='pass', phase='complete',
                source_binding=binding, originals_rehashed_after=True, predicate_parity_only=False,
                new_numeric_algorithm=True, physical_geometry_rescaled=False,
                conditioned_protocol_identity=gate.identity(code/gate.CONDITIONED_PROTOCOL), geometry=geometry,
                parity={'cases': 128, 'mismatches': 0, 'policy_sha256': 'policy'},
                orientation_parity={'cases': 128, 'mismatches': 0},
                controls={'identity_native_calls': 1, 'committed_collapses': 0,
                          'expected_rejections': 4, 'key_rounding_cases': 7})
            native.update({k: False for k in ['simplification_validated', 'geometry_quality_validated',
                'production_mesh_used', 'challenge_performance_verified', 'gpu_used', 'adoption']})
            gate.write(out/'native.json', (json.dumps(native)+'\n').encode())
            return SimpleNamespace(returncode=0)
        output = (config['source_authentication']['original_image_id']+'\n').encode() if command[1] == 'image' else b''
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(gate.subprocess, 'run', subprocess_stub)
    return code, commands, source_calls


@pytest.mark.parametrize('malformed', [False, True])
def test_host_authenticates_four_real_new_arms_and_owns_narrow_mounts(tmp_path, monkeypatch, malformed):
    code, commands, source_calls = host_fixture(tmp_path, monkeypatch, malformed=malformed)
    if malformed:
        with pytest.raises(ValueError, match='failed'):
            gate.host(code, REVISION, conditioned=True)
    else:
        gate.host(code, REVISION, conditioned=True)
    out = tmp_path/'results'/('mesh-conditioned-qem-'+REVISION)
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == ('fail' if malformed else 'pass')
    assert report['stage'] == 'mesh_conditioned_qem_host_v1' and not report['predicate_parity_only']
    assert report['total_budget_seconds'] == 5400 and report['receipt_publication_grace_seconds'] == 10
    assert report['owned_container_removed'] and report['source_rehashed_after']
    assert source_calls == [{'conditioned': True}, {'conditioned': True}]
    command = next(c for c in commands if c[1] == 'run')
    assert '--native-conditioned' in command and '--native-geometry' not in command and '--gpus' not in command
    mounts = [command[i+1] for i, v in enumerate(command) if v == '--mount']
    assert len(mounts) == 5 and len([m for m in mounts if not m.endswith(',readonly')]) == 1
    assert mounts[-1] == f'type=bind,src={tmp_path}/vendor/v2d_submission_kit/v2dlb/mesh_budget.py,dst={tmp_path}/vendor/v2d_submission_kit/v2dlb/mesh_budget.py,readonly'
    assert 'src='+str(tmp_path)+',dst='+str(tmp_path) not in ','.join(mounts)
    assert '--cap-drop' in command and '--network' in command and '--read-only' in command
    assert not (out/'.container.cid').exists() and not (out/'.native.log').exists()

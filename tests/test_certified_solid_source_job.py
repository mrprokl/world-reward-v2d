"""Tiny source ledgers/receipts and mocked containers, never native or Azure."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    return (load('wr_solid_job_test', ROOT / 'infra/certified_solid_source_job.py'),
            load('wr_solid_build_helper_test', ROOT / 'infra/certified_solid_build.py'))


def write(path, raw=b'owned metadata', mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw); path.chmod(mode)
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def readonly(code):
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)


def control_report(build, sha):
    return dict(stage='certified_solid_procedural_controls_v1', status='pass', phase='complete',
                native_source_sha256=sha, source_binary_rehashed_after=True, maximum_calls=15,
                call_seconds=60, inclusive_seconds=900, elapsed_seconds=1.,
                records=[{'name': n, 'status': 'pass'} for n in build.CONTROL_NAMES],
                **{k: False for k in ('challenge_inputs_used', 'gt_used', 'qem_executed', 'adoption',
                                     'production_mesh_validated', 'reconstruction_accuracy_verified')})


def qualify_fixture(job, build, monkeypatch, tmp_path, mutate=None):
    root = tmp_path / 'root'; code = root / 'code'; code.mkdir(parents=True)
    revision = 'a' * 40; out = root / 'results' / ('certified-solid-build-' + revision)
    source = {'bytes': 1, 'sha256': 'b' * 64}
    binding = {'native_source': source}
    scope = {k: False for k in ('gpu_used', 'gt_used', 'adoption', 'production_mesh_validated',
                               'reconstruction_accuracy_verified', 'competition_eligibility_verified')}
    child = 'sha256:' + 'c' * 64
    binary = write(out / 'certified_solid_query', b'owned fake executable, never run', 0o555)
    native = dict(status='pass', phase='complete', compiled_binary=binary, image_id=child,
                  source_binding=binding, source_binding_after=binding,
                  controls=control_report(build, source['sha256']), **scope)
    host = dict(status='pass', phase='complete', native=native, retained_binary=binary, child_image_id=child,
                parent_image_id=build.PARENT, producer_revision=revision, source_binding=binding,
                source_binding_after=binding, **scope,
                **{k: True for k in ('source_rehashed_after', 'parent_rehashed_after', 'build_inputs_rehashed_after',
                                     'disposable_build_inputs_removed', 'child_image_parent_verified')})
    if mutate: mutate(host, native)
    identities = {'native': write(out / 'native.json', json.dumps(native).encode()),
                  'report': write(out / 'report.json', json.dumps(host).encode()), 'binary': binary}
    pins = dict(schema='world_reward.certified_solid_qualification_pins.v1', producer_revision=revision,
                parent_image_id=build.PARENT, child_image_id=child, native_source=source,
                qualified_controls=15, **identities)
    write(code / job.QUALIFICATION, json.dumps(pins).encode())
    monkeypatch.setattr(job, 'ROOT', root)
    return root, code, pins, out


def test_actual_frozen_pins_primary_values_and_no_placeholder():
    pins = json.loads((ROOT / 'configs/certified_solid_qualification_pins.json').read_bytes())
    assert pins['producer_revision'] == '253fc9d2f645cb3a2edcc9e9d6a3ea6ba6876ae7'
    assert pins['qualified_controls'] == 15
    assert pins['report'] == {'bytes': 102696, 'sha256': 'c047ba89d5982fdede115fb38b147bfddd0b11435478f3fd08185733d27154eb'}
    assert pins['binary']['bytes'] == 896712
    # Historical pins do not qualify the new balanced accumulator. Real
    # consumers reject a mismatching current source before a geometry call.
    assert pins['native_source'] == {'bytes': 13914, 'sha256':
        '72098be329146be0c48f32bf1473731c195120fb53b65511955d69e222c59ee4'}
    assert 'eligibility' in pins['qualification_scope']


def test_qualification_validates_actual_bytes_full_receipts_scope_no_native(modules, monkeypatch, tmp_path):
    job, build = modules
    _, code, pins, _ = qualify_fixture(job, build, monkeypatch, tmp_path)
    monkeypatch.setattr(build, 'run', lambda *_a, **_k: pytest.fail('Qualification must not execute binary/model'))
    result, paths, measured = job.qualification(code, build)
    assert result == pins and measured == {k: pins[k] for k in ('report', 'native', 'binary')}
    assert set(paths) == {'report', 'native', 'binary'}


@pytest.mark.parametrize('mutate', [lambda h, n: h.update(status='fail'),
                                   lambda h, n: n.update(phase='compile'),
                                   lambda h, n: h.update(parent_rehashed_after=False),
                                   lambda h, n: n.update(gt_used=True),
                                   lambda h, n: n.update(gpu_used=True),
                                   lambda h, n: h.update(adoption=True),
                                   lambda h, n: n['controls']['records'].pop(),
                                   lambda h, n: n['controls'].update(source_binary_rehashed_after=False),
                                   lambda h, n: h.update(child_image_id='sha256:' + 'f' * 64)])
def test_qualification_rejects_incomplete_or_unsafe_even_byte_bound_fixture(modules, monkeypatch, tmp_path, mutate):
    job, build = modules
    _, code, _, _ = qualify_fixture(job, build, monkeypatch, tmp_path, mutate)
    with pytest.raises(ValueError): job.qualification(code, build)


def test_qualification_hash_change_fails_before_certificate(modules, monkeypatch, tmp_path):
    job, build = modules
    _, code, _, out = qualify_fixture(job, build, monkeypatch, tmp_path)
    (out / 'certified_solid_query').chmod(0o755); (out / 'certified_solid_query').write_bytes(b'changed')
    with pytest.raises(ValueError): job.qualification(code, build)


def test_current_source_binding_includes_empty_files_markers_and_exact_entrypoint(modules, monkeypatch, tmp_path):
    job, build = modules
    root = tmp_path / 'root'; revision = 'd' * 40; code = root / 'jobs' / revision / job.ENTRY / 'code'
    write(code / 'infra/certified_solid_source_job.py', b'own calling source')
    write(code / 'src/world_reward/__init__.py', b'')
    write(code.parent / 'revision', (revision + '\n').encode()); write(code.parent / 'source-sha256', b'a' * 64 + b'\n')
    readonly(code)
    monkeypatch.setattr(job, 'ROOT', root); monkeypatch.setattr(job, '__file__', str(code / 'infra/certified_solid_source_job.py'))
    result = job.binding(code, revision, build)
    assert result['files']['src/world_reward/__init__.py']['bytes'] == 0
    assert result['markers']['revision']['bytes'] == 41
    with pytest.raises(ValueError): job.binding(code, 'e' * 40, build)
    code.chmod(0o755)
    with pytest.raises(ValueError): job.binding(code, revision, build)


def test_original_full_source_ledger_modes_paths_markers_immutable(modules, monkeypatch, tmp_path):
    job, build = modules
    root = tmp_path / 'root'; code = root / 'current'; revision = 'e' * 40
    original = root / 'jobs' / revision / 'run_track1_frontends' / 'code'
    write(original / 'infra/object_smoke.py', b'original source')
    write(original / 'infra/run_track1_frontends.sh', b'original entrypoint', 0o555)
    write(original.parent / 'revision', (revision + '\n').encode())
    write(original.parent / 'source-sha256', b'f' * 64 + b'\n'); readonly(original)
    digest = hashlib.sha256()
    for p in sorted(original.rglob('*')):
        if p.is_file():
            digest.update(p.relative_to(original).as_posix().encode() + b'\0' + str(p.stat().st_mode & 0o777).encode()
                          + b'\0' + bytes.fromhex(build.identity(p)['sha256']))
    lineage = dict(schema='world_reward.certified_solid_source_lineage.v1', episode_index=9, producer_revision=revision,
                   entrypoint='run_track1_frontends', files=2, archive_sha256='f' * 64, ledger_sha256=digest.hexdigest())
    write(code / job.LINEAGE, json.dumps(lineage).encode())
    monkeypatch.setattr(job, 'ROOT', root)
    assert job.original_binding(code, build, 9) == lineage
    with pytest.raises(ValueError): job.original_binding(code, build, 8)
    original.chmod(0o755); write(original / 'unlisted', b'extra'); original.chmod(0o555)
    with pytest.raises(ValueError): job.original_binding(code, build, 9)


def host_fixture(job, build, monkeypatch, tmp_path, native_status='pass', mutate_native=None):
    root = tmp_path / 'root'; revision = 'd' * 40; code = root / 'jobs' / revision / job.ENTRY / 'code'
    code.mkdir(parents=True); (root / 'results').mkdir()
    original = {'episode_index': 9, 'original_producer_revision': 'e' * 40, 'files': {}}
    for role in ('glb', 'object_report', 'original_producer', 'image_evidence', 'failed_report'):
        p = root / 'original' / role; pin = write(p, ('source-' + role).encode())
        original['files'][role] = {'path': p.relative_to(root).as_posix(), **pin}
    write(code / job.ORIGINAL_PINS, json.dumps(original).encode())
    paths = {k: root / 'qualified' / k for k in ('report', 'native', 'binary')}
    for p in paths.values(): write(p)
    pins = {'child_image_id': 'sha256:' + 'a' * 64}
    measured = {k: build.identity(p) for k, p in paths.items()}
    bound = {'files': {'ownsource': {'bytes': 4, 'sha256': 'f' * 64}}, 'markers': {}, 'producer_revision': revision}
    lineage = {'producer_revision': 'e' * 40}
    monkeypatch.setattr(job, 'ROOT', root); monkeypatch.setattr(job.sys, 'platform', 'linux')
    monkeypatch.setattr(job.os, 'getuid', lambda: 0); monkeypatch.setattr(job.os, 'chown', lambda *_: None)
    monkeypatch.setenv('DOCKER_HOST', 'unix://' + str(root / 'docker.sock'))
    monkeypatch.setattr(job, 'binding', lambda *_: bound)
    monkeypatch.setattr(job, 'qualification', lambda *_: (pins, paths, measured))
    monkeypatch.setattr(job, 'original_binding', lambda *_: lineage)
    cleanup_calls = []
    monkeypatch.setattr(build, 'cleanup_container', lambda *args: cleanup_calls.append(args))
    calls = []
    def run(argv, seconds, log=None):
        calls.append((argv, seconds))
        if argv[1:3] == ['image', 'inspect']: return (pins['child_image_id'] + '\n').encode()
        assert argv[:2] == ['docker', 'run']
        out = root / 'results' / f'certified-solid-source-000009-{revision}'
        certificate = dict(status=native_status, phase='complete' if native_status == 'pass' else 'native_exact_geometry',
                           native_attempts=1, native_returned=True, represented_embedding_certified=native_status == 'pass',
                           source_binary_helpers_rehashed_after=True,
                           **{k: False for k in ('ground_truth_used', 'adoption', 'qem_executed', 'geometry_repaired',
                                                 'orientation_changed', 'packing_validated', 'reconstruction_accuracy_verified')})
        native = {'stage': 'certified_solid_source_native_job_v1', 'status': native_status,
                  'phase': 'complete' if native_status == 'pass' else 'native_exact_geometry',
                  'source_binding': bound, 'source_binding_after': bound, 'source_rehashed_after': True,
                  'certificate': certificate, 'episode_index': 9,
                  'gpu_used': False, 'gt_used': False, 'adoption': False, 'reconstruction_accuracy_verified': False}
        if mutate_native:
            mutate_native(native)
        write(out / 'disposable/native.json', json.dumps(native).encode())
        write(log, b'known technical stderr')
        if native_status == 'fail': raise ValueError('mock owned native failure')
    monkeypatch.setattr(build, 'run', run)
    return root, code, revision, paths, original, calls, cleanup_calls


def test_host_exact_cpu_mounts_no_models_oldfullsource_or_gpu_and_cleanup(modules, monkeypatch, tmp_path):
    job, build = modules
    root, code, revision, paths, original, calls, clean = host_fixture(job, build, monkeypatch, tmp_path)
    assert job.host(code, revision, 9, build) == 0
    command = next(argv for argv, _ in calls if argv[:2] == ['docker', 'run'])
    assert '--gpus' not in command and 'CUDA_VISIBLE_DEVICES=-1' in command
    for flag, value in [('--network', 'none'), ('--user', '1000:1000'), ('--cap-drop', 'ALL'), ('--memory', '16g'), ('--cpus', '4')]:
        assert command[command.index(flag) + 1] == value
    assert '--read-only' in command
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == '--mount']
    for path in (code, code.parent / 'revision', code.parent / 'source-sha256', *paths.values(),
                 *(root / row['path'] for row in original['files'].values())):
        assert f'type=bind,src={path},dst={path},readonly' in mounts
    assert len(mounts) == 12  # CODE + 2markers + 3qualified + 5original + exact sole RW work.
    assert not any('/vendor' in m or '/weights' in m or 'eval_private' in m or 'depth' in m for m in mounts)
    assert not any('jobs/' + 'e' * 40 in m for m in mounts)  # Original fullsource host-only authentication.
    out = root / 'results' / f'certified-solid-source-000009-{revision}'
    r = build.strict_json((out / 'report.json').read_bytes())
    assert r['status'] == 'pass' and r['source_original_runtime_rehashed_after']
    assert r['historical_gate_reclassified'] is False and r['disposable_removed']
    assert not (out / 'disposable').exists() and len(clean) == 1
    assert stat.S_IMODE((out / 'report.json').stat().st_mode) == 0o444
    assert stat.S_IMODE(out.stat().st_mode) == 0o555


def test_host_preserves_failed_certificate_and_technical_tail_without_adoption(modules, monkeypatch, tmp_path):
    job, build = modules
    root, code, revision, _, _, _, _ = host_fixture(job, build, monkeypatch, tmp_path, native_status='fail')
    assert job.host(code, revision, 9, build) == 1
    out = root / 'results' / f'certified-solid-source-000009-{revision}'
    r = build.strict_json((out / 'report.json').read_bytes())
    assert r['status'] == 'fail' and r['native']['certificate']['status'] == 'fail'
    assert r['technical_failure_tail'] == 'known technical stderr' and len(r['technical_failure_tail']) <= 1000
    assert not r['adoption'] and r['source_original_runtime_rehashed_after'] and r['disposable_removed']


def test_host_existing_output_fail_before_launch_no_overwrite(modules, monkeypatch, tmp_path):
    job, build = modules
    root, code, revision, _, _, calls, _ = host_fixture(job, build, monkeypatch, tmp_path)
    out = root / 'results' / f'certified-solid-source-000009-{revision}'; out.mkdir(parents=True)
    write(out / 'old', b'original preserved')
    with pytest.raises(ValueError): job.host(code, revision, 9, build)
    assert not any(argv[:2] == ['docker', 'run'] for argv, _ in calls)
    assert (out / 'old').read_bytes() == b'original preserved'


def test_shell_strict_arguments_real_source_closure_no_gpu():
    azure = load('wr_job_azure_closure', ROOT / 'infra/azure_job.py')
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for folder in ('infra', 'src/world_reward', 'configs')
             for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.cpp', '.sh', '.json')}
    files['pyproject.toml'] = (ROOT / 'pyproject.toml').read_bytes()
    paths = azure.runtime_bundle_paths(files, 'infra/run_certified_solid_source.sh')
    required = ('infra/certified_solid_source_job.py', 'infra/certified_solid_source.py', 'infra/certified_solid_query.cpp',
                'infra/certified_solid_build.py', 'infra/mesh_precision_diagnostic.py', 'src/world_reward/oriented_solid_forest.py',
                'configs/certified_solid_qualification_pins.json', 'configs/certified_solid_source_lineage_pins.json')
    assert all(p in paths for p in required)
    shell = (ROOT / 'infra/run_certified_solid_source.sh').read_text()
    assert '[[ $# == 2 && "$1" == --episode' in shell and '--kill-after=10s 450s' in shell
    assert '--gpus' not in shell and 'docker.sock' in shell


@pytest.mark.parametrize('mutation', [lambda n: n['certificate'].update(status='fail'),
                                      lambda n: n['certificate'].update(phase='native_exact_geometry'),
                                      lambda n: n['certificate'].update(native_attempts=0),
                                      lambda n: n['certificate'].update(native_returned=False),
                                      lambda n: n['certificate'].update(represented_embedding_certified=False),
                                      lambda n: n['certificate'].update(source_binary_helpers_rehashed_after=False),
                                      lambda n: n['certificate'].update(ground_truth_used=True),
                                      lambda n: n['certificate'].update(geometry_repaired=True),
                                      lambda n: n.update(gpu_used=True),
                                      lambda n: n.update(gt_used=True),
                                      lambda n: n.update(adoption=True),
                                      lambda n: n.update(episode_index=8),
                                      lambda n: n.update(episode_index=True),
                                      lambda n: n.update(stage='other_native_stage')])
def test_host_rejects_forged_native_pass_scope_or_incomplete_certificate(modules, monkeypatch, tmp_path, mutation):
    job, build = modules
    root, code, revision, _, _, _, clean = host_fixture(job, build, monkeypatch, tmp_path, mutate_native=mutation)
    assert job.host(code, revision, 9, build) == 1
    out = root / 'results' / f'certified-solid-source-000009-{revision}'
    report = build.strict_json((out / 'report.json').read_bytes())
    assert report['status'] == 'fail' and report['posthash_failure_type'] == 'ValueError'
    assert report['adoption'] is False and report['historical_gate_reclassified'] is False
    assert report['disposable_removed'] and len(clean) == 1

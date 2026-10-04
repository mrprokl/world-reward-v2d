"""Tiny real receipt authentication; Docker/native computation is mocked only."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

from test_solid_chart_v2_qualify import build, built, controls, evidence, gate, write
import certified_solid_source_job as certificate


def native_controls(source):
    return dict(stage='certified_solid_procedural_controls_v1', status='pass', phase='complete',
        native_source_sha256=source, source_binary_rehashed_after=True, maximum_calls=15,
        call_seconds=60, inclusive_seconds=900, elapsed_seconds=1.,
        records=[dict(name=name, status='pass') for name in build.CONTROL_NAMES],
        **dict.fromkeys(('challenge_inputs_used', 'gt_used', 'qem_executed', 'adoption',
            'production_mesh_validated', 'reconstruction_accuracy_verified'), False))


def query_receipt(root, revision, source, image):
    out = root/'results'/('certified-solid-build-'+revision)
    binary = write(out/'certified_solid_query', ('never executed '+revision).encode(), 0o555)
    binding = dict(producer_revision=revision, native_source=source)
    scope = dict.fromkeys(('gpu_used', 'gt_used', 'adoption', 'production_mesh_validated',
        'reconstruction_accuracy_verified', 'competition_eligibility_verified'), False)
    native = dict(status='pass', phase='complete', image_id=image, compiled_binary=binary,
        source_binding=binding, source_binding_after=binding,
        controls=native_controls(source['sha256']), **scope)
    host = dict(status='pass', phase='complete', native=native, retained_binary=binary,
        child_image_id=image, parent_image_id=build.PARENT, producer_revision=revision,
        source_binding=binding, source_binding_after=binding, **scope,
        **dict.fromkeys(('source_rehashed_after', 'parent_rehashed_after', 'build_inputs_rehashed_after',
            'disposable_build_inputs_removed', 'child_image_parent_verified'), True))
    native_pin = write(out/'native.json', json.dumps(native).encode())
    report_pin = write(out/'report.json', json.dumps(host).encode()); out.chmod(0o555)
    return dict(schema='world_reward.certified_solid_qualification_pins.v1', qualified_controls=15,
        producer_revision=revision, child_image_id=image, parent_image_id=build.PARENT,
        report=report_pin, native=native_pin, binary=binary, native_source=source)


def rewrite_query(case, mutate):
    """Repin manufactured evidence to test semantic gates, not mere hash errors."""
    out = case.root/'results'/('certified-solid-build-'+case.active['producer_revision'])
    host = json.loads((out/'report.json').read_bytes()); mutate(host)
    case.active['native'] = write(out/'native.json', json.dumps(host['native']).encode())
    case.active['report'] = write(out/'report.json', json.dumps(host).encode())
    write(case.code/gate.BALANCED_PINS, json.dumps(case.active).encode())


@pytest.fixture
def composition(tmp_path, monkeypatch):
    from types import SimpleNamespace
    code, producer, pins, _ = evidence(tmp_path, monkeypatch)
    root = gate.ROOT; image = pins['image_id']
    write(producer/'infra/certified_solid_query.cpp', b'original immutable exact query')
    write(code/'infra/certified_solid_query.cpp', b'new balanced exact query')
    old = query_receipt(root, 'd'*40, build.identity(producer/'infra/certified_solid_query.cpp'), image)
    active = query_receipt(root, 'e'*40, build.identity(code/'infra/certified_solid_query.cpp'), image)
    for p, value in ((producer/certificate.QUALIFICATION, old), (code/certificate.QUALIFICATION, old)):
        write(p, json.dumps(value).encode())
    (code/'configs').chmod(0o755)
    write(code/gate.BALANCED_PINS, json.dumps(active).encode()); (code/'configs').chmod(0o555)
    monkeypatch.setattr(certificate, 'ROOT', root)
    _, old_paths, old_measured = certificate.qualification(producer, build)
    _, active_paths, active_measured = certificate.qualification(code, build, query_requalification=True)
    rows, bound = gate.snapshot(producer, pins['producer_revision'], build)
    actual = {k: bound[k] for k in ('producer_revision', 'markers', 'source_files', 'source_files_sha256')}
    actual['helpers'] = {n: rows[n] for n in built.HELPERS}
    out = root/'results'/('solid-chart-v2-build-'+pins['producer_revision'])
    host = json.loads((out/'report.json').read_bytes()); native = host['native']
    native.update(source_binding=actual, source_binding_after=actual,
        qualified_inputs=dict(cgal=old_measured, cache={'unchanged': True}, historical_image_receipt={'unchanged': True}))
    host.update(source_binding=actual, source_binding_after=actual)
    pins['native'] = write(out/'native.json', json.dumps(native).encode())
    pins['report'] = write(out/'report.json', json.dumps(host).encode())
    for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256'): pins[k] = bound[k]
    write(code/gate.BUILD_PINS, json.dumps(pins).encode())
    monkeypatch.setattr(gate, '__file__', str(code/'infra/solid_chart_v2_qualify.py'))
    return SimpleNamespace(root=root, code=code, producer=producer, pins=pins, old=old, active=active,
        old_paths=old_paths, active_paths=active_paths, old_measured=old_measured,
        active_measured=active_measured, native=native, revision='a'*40)


def test_original_build_proof_preserved_active_query_separately_authenticated(composition):
    c = composition
    with pytest.raises(ValueError, match='query/image'): gate.built_qualification(c.code, build, certificate, built)
    pins, qem, query, mounts, proof = gate.built_qualification(c.code, build, certificate, built, query_requalification=True)
    assert pins == c.pins and query == c.active_paths['binary'] and qem.name == 'mesh_conditioned_chart_v2'
    assert proof['cgal'] == c.old_measured and proof['build'] == c.native['build']
    extra = proof['query_requalification']
    assert extra['original_query']['artifacts'] == c.old_measured
    assert extra['active_query']['artifacts'] == c.active_measured
    assert extra['original_qem_build_unchanged'] is True and extra['original_qem_recompiled'] is False
    assert extra['procedural_controls_required'] == 15
    assert c.producer in mounts and c.producer.parent/'revision' in mounts
    assert c.old_paths['binary'].parent in mounts and c.active_paths['binary'].parent in mounts
    before = gate.binding(c.code, c.revision, build, query_requalification=True)
    assert before['helpers'][gate.BALANCED_PINS] == build.identity(c.code/gate.BALANCED_PINS)
    assert 'infra/certified_solid_query.cpp' in before['helpers']
    assert gate.BALANCED_PINS not in gate.binding(c.code, c.revision, build)['helpers']


@pytest.mark.parametrize('fault', ['missing_pin', 'fewer_controls', 'old_controls', 'failed_controls',
    'unsealed_query', 'wrong_current_source', 'changed_historical_source', 'wrong_image', 'changed_original_binary'])
def test_active_query_cannot_skip_independent_qualification_or_relabel_qem(composition, fault):
    c = composition
    if fault == 'missing_pin':
        (c.code/'configs').chmod(0o755); (c.code/gate.BALANCED_PINS).unlink()
    elif fault == 'fewer_controls': rewrite_query(c, lambda h: h['native']['controls']['records'].pop())
    elif fault == 'old_controls': rewrite_query(c, lambda h: h['native']['controls'].update(native_source_sha256=c.old['native_source']['sha256']))
    elif fault == 'failed_controls': rewrite_query(c, lambda h: h['native']['controls']['records'][0].update(status='fail'))
    elif fault == 'unsealed_query': c.active_paths['binary'].chmod(0o755)
    elif fault == 'wrong_current_source': write(c.code/'infra/certified_solid_query.cpp', b'unqualified third query')
    elif fault == 'changed_historical_source': write(c.producer/'infra/certified_solid_source_job.py', b'changed historical source')
    elif fault == 'wrong_image':
        c.active['child_image_id'] = 'sha256:'+'f'*64
        rewrite_query(c, lambda h: (h.update(child_image_id=c.active['child_image_id']), h['native'].update(image_id=c.active['child_image_id'])))
    else: write(c.old_paths['binary'], b'changed original query', 0o555)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.built_qualification(c.code, build, certificate, built, query_requalification=True)
    assert not (c.root/'results'/('solid-chart-v2-query-requalify-'+c.revision)).exists()


@pytest.mark.parametrize('invalid', ['arbitrary/path', 1, None])
def test_query_profile_is_boolean_not_an_arbitrary_path(composition, invalid):
    with pytest.raises(ValueError, match='profile'):
        gate.built_qualification(composition.code, build, certificate, built, query_requalification=invalid)
    with pytest.raises(ValueError, match='profile'):
        certificate.qualification(composition.code, build, query_requalification=invalid)


@pytest.mark.parametrize('fault', ['none', 'exit', 'active_posthash', 'original_posthash', 'native_proof', 'image', 'runtime_proof'])
def test_host_reuses_original_qem_and_runtime_but_mounts_active_query(composition, monkeypatch, fault):
    c = composition; calls = []; qualified_codes = []; cleanup = []
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None)
    monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(c.root/'docker.sock'))
    before_proof = gate.built_qualification(c.code, build, certificate, built, query_requalification=True)[4]
    def qualified(code, *args):
        qualified_codes.append(code); assert code == c.producer  # Never current pins as original build.
        return c.old, (c.old_paths['binary'].parent,), copy.deepcopy(c.native['qualified_inputs'])
    monkeypatch.setattr(built, 'qualified', qualified)
    image_calls = []
    def image(*args):
        image_calls.append(args)
        return dict(child_id=('sha256:'+'f'*64 if fault == 'image' and len(image_calls) > 1 else c.pins['image_id']))
    monkeypatch.setattr(built, 'image_identity', image)
    monkeypatch.setattr(build, 'cleanup_container', lambda *args: cleanup.append(args))
    monkeypatch.setattr(gate, 'OFFICIAL_PIN', write(c.root/gate.OFFICIAL, mode=0o644))
    out = c.root/'results'/('solid-chart-v2-query-requalify-'+c.revision)
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        calls.append(argv); assert 0 < seconds <= 1810 and argv[:2] == ['docker', 'run']
        manifest, full = controls(); before = gate.binding(c.code, c.revision, build, query_requalification=True)
        native = dict(stage='solid_chart_v2_qualification_native_v1', status='pass', phase='complete',
            source_binding=before, source_binding_after=before, source_rehashed_after=True,
            artifacts_rehashed_after=True, source_arrays_unchanged=True, elapsed_seconds=1.,
            official_helper=gate.official_identity(build), official_rehashed_after=True,
            qualified_build=before_proof, query_requalification=copy.deepcopy(before_proof['query_requalification']),
            image_id=c.pins['image_id'], gpu_used=False, gt_used=False, adoption=False,
            reconstruction_accuracy_verified=False, control_manifest=manifest, controls=full,
            control_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
        if fault == 'native_proof': native['query_requalification']['original_qem_recompiled'] = True
        write(out/'disposable/native.json', json.dumps(native).encode()); write(log)
        if fault == 'active_posthash': write(c.active_paths['binary'], b'changed active query', 0o555)
        if fault == 'original_posthash': write(c.producer/'infra/solid_chart_v2_build.py', b'changed original closure')
        if fault == 'runtime_proof': c.native['qualified_inputs']['cache'] = {'changed': True}
        if fault == 'exit': raise ValueError('Native failure with receipt')
    monkeypatch.setattr(build, 'run', run)
    assert gate.host(c.code, c.revision, build, certificate, built, query_requalification=True) == int(fault != 'none')
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == ('pass' if fault == 'none' else 'fail')
    assert len(calls) == 1 and cleanup and qualified_codes and not (out/'disposable').exists()
    assert not (c.root/'results'/('solid-chart-v2-qualify-'+c.revision)).exists()
    argv = calls[0]; assert '--query-requalification' in argv and '--native' in argv and '--gpus' not in argv
    assert argv[argv.index('--network')+1] == 'none' and argv[argv.index('--cpus')+1] == '4'
    assert argv[argv.index('--memory')+1] == '16g' and '--read-only' in argv
    mounts = [argv[i+1] for i, x in enumerate(argv) if x == '--mount']
    for p in (c.producer, c.producer.parent/'revision', c.producer.parent/'source-sha256',
              c.old_paths['binary'].parent, c.active_paths['binary'].parent):
        assert f'type=bind,src={p},dst={p},readonly' in mounts
    assert not any('compile' in x or x in ('build', 'rmi', 'tag') for x in argv)
    assert report['qualified_build']['cgal'] == c.old_measured
    if fault != 'none': assert 'qualified_procedural_controls' not in report


def test_native_four_fresh_controls_use_active_query_without_qem_compile(composition, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    c = composition; monkeypatch.setattr(np, '__version__', '1.26.3')
    original_qualification = gate.built_qualification
    def qualified(*args, **kwargs):
        result = list(original_qualification(*args, **kwargs))
        result[4]['build'].update(derivation={'chart': 2}, build_info={'actual': 'unchanged original QEM'})
        return tuple(result)
    monkeypatch.setattr(gate, 'built_qualification', qualified)
    work = c.root/'results'/('solid-chart-v2-query-requalify-'+c.revision)/'disposable'
    work.mkdir(parents=True, mode=0o700)
    real_stat = Path.stat
    def native_stat(path, *args, **kwargs):
        value = real_stat(path, *args, **kwargs)
        return SimpleNamespace(st_uid=1000, st_mode=value.st_mode) if path == work else value
    monkeypatch.setattr(Path, 'stat', native_stat); monkeypatch.setattr(gate.sys, 'platform', 'linux')
    monkeypatch.setattr(gate.os, 'getuid', lambda: 1000)
    monkeypatch.setenv('WR_NATIVE_NETWORK', 'none'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('WR_CPU_IMAGE_ID', c.pins['image_id'])
    monkeypatch.setattr(gate, 'OFFICIAL_PIN', write(c.root/gate.OFFICIAL, mode=0o644))
    binary_calls = []
    def run(argv, seconds):
        binary_calls.append(argv); assert argv == [str(c.root/'results'/('solid-chart-v2-build-'+c.pins['producer_revision'])/'mesh_conditioned_chart_v2'), '--build-info']
        return json.dumps({'actual': 'unchanged original QEM'}).encode()
    monkeypatch.setattr(build, 'run', run)
    mesh = (np.zeros((4, 3), np.float64), np.ones((4, 3), np.int64))
    hashes = lambda m: [hashlib.sha256(a.tobytes()).hexdigest() for a in m]
    source = tuple((f'fresh_{n}', mesh, dict(source_array_sha256=tuple(hashes(mesh)),
        expected_parents=np.array([-1]), expected_signs=np.array([1]), specification_sha256='b'*64)) for n in range(4))
    controls_calls = []
    def geometry(binary, query, query_sha, scratch, official, **kwargs):
        controls_calls.append((binary, query, query_sha, kwargs))
        assert query == c.active_paths['binary'] and query_sha == c.active['native_source']['sha256']
        assert kwargs == {'controls': source, 'conditioning_version': 2}
        _, full = controls()
        for row in full['controls']: row['source_array_sha256'] = tuple(hashes(mesh))
        return full
    modules = dict(mesh_conditioned_chart_v2_source=SimpleNamespace(derive_cached_backend=lambda *_: (b'no compilation', {'chart': 2})),
        oriented_solid_controls_v2=SimpleNamespace(fixtures=lambda: source, FIXTURE_NAMES=tuple(n for n, _, _ in source)),
        oriented_solid_compiler=SimpleNamespace(geometry_controls=geometry, SolidCompilerError=RuntimeError, array_hashes=hashes),
        object_budget_conditioned=SimpleNamespace(cache_qualification=lambda *_: (c.producer, {}), runtime_identity=lambda *_: {}))
    for name, module in modules.items(): monkeypatch.setitem(sys.modules, name, module)
    assert gate.native(c.code, c.revision, work, build, certificate, built, query_requalification=True) == 0
    report = json.loads((work/'native.json').read_bytes())
    assert len(binary_calls) == len(controls_calls) == 1 and report['qualified_procedural_controls'] == 4
    assert report['qualified_build']['cgal'] == c.old_measured and report['query_requalification']['active_query']['artifacts'] == c.active_measured
    assert report['artifacts_rehashed_after'] and report['source_arrays_unchanged']


def test_shell_optin_forwarding_and_bootstrap_are_stdlib_only():
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(['bash', '-n', str(repo/'infra/run_solid_chart_v2_qualify.sh')], check=True)
    command = 'import runpy,sys;runpy.run_path(sys.argv[1],run_name="not_main");assert "numpy" not in sys.modules'
    subprocess.run([sys.executable, '-I', '-B', '-S', '-c', command, str(repo/'infra/solid_chart_v2_qualify.py')], check=True)
    shell = (repo/'infra/run_solid_chart_v2_qualify.sh').read_text()
    assert '"$@"' in shell and '--query-requalification' in shell and '1960s' in shell
    import azure_job
    files = {p.relative_to(repo).as_posix(): p.read_bytes() for d in ('infra', 'src', 'configs')
        for p in (repo/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    closure = azure_job.runtime_bundle_paths(files, 'infra/run_solid_chart_v2_qualify.sh')
    assert {gate.BALANCED_PINS, certificate.QUALIFICATION, 'infra/certified_solid_query.cpp'} <= set(closure)

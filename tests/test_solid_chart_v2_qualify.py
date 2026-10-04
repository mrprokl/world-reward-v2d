import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'infra'), str(REPO/'src')]
import solid_chart_v2_qualify as gate
import solid_chart_v2_build as built
import certified_solid_build as build


def write(path, raw=b'owned tiny metadata', mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(mode)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source(tmp_path, monkeypatch, *, revision='a'*40, entry=gate.ENTRY):
    root = tmp_path/'root'; code = root/'jobs'/revision/entry/'code'
    for name in set(gate.HELPERS) | set(built.HELPERS) | {'infra/certified_solid_query.cpp'}:
        write(code/name)
    write(code/'src/world_reward/__init__.py', b'')
    write(code.parent/'revision', (revision+'\n').encode())
    write(code.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', root)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/solid_chart_v2_qualify.py'))
    return root, code, revision


def test_snapshot_exact_readonly_ledger_and_empty_init(tmp_path, monkeypatch):
    _, code, revision = source(tmp_path, monkeypatch)
    rows, result = gate.snapshot(code, revision, build); digest = hashlib.sha256()
    for name, pin in sorted(rows.items()):
        digest.update(name.encode()+b'\0'+str((code/name).stat().st_mode & 0o555).encode()+b'\0'+bytes.fromhex(pin['sha256']))
    assert result['source_readonly_ledger_sha256'] == digest.hexdigest()
    assert result['source_files_sha256'] == hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert rows['src/world_reward/__init__.py']['bytes'] == 0
    assert gate.binding(code, revision, build)['helpers'] == {n: rows[n] for n in gate.HELPERS}
    with pytest.raises(ValueError): gate.binding(code, 'c'*40, build)


@pytest.mark.parametrize('change', ['writable', 'marker', 'alias', 'hardlink', 'extra_snapshot'])
def test_changed_snapshot_is_not_repaired(tmp_path, monkeypatch, change):
    _, code, revision = source(tmp_path, monkeypatch); p = code/gate.HELPERS[0]
    if change == 'writable': p.chmod(0o644)
    elif change == 'marker': write(code.parent/'revision', ('c'*40+'\n').encode())
    elif change == 'alias': p.parent.chmod(0o755); p.unlink(); p.symlink_to(code/gate.HELPERS[1])
    elif change == 'hardlink': os.link(p, tmp_path/'foreign')
    else: write(code.parent/'foreign')
    with pytest.raises(ValueError): gate.binding(code, revision, build)


def evidence(tmp_path, monkeypatch):
    root, code, _ = source(tmp_path, monkeypatch)
    _, producer, revision = source(tmp_path, monkeypatch, revision='c'*40, entry='run_solid_chart_v2_build')
    rows, bound = gate.snapshot(producer, revision, build)
    binding = {k: bound[k] for k in ('producer_revision', 'markers', 'source_files', 'source_files_sha256')} | dict(helpers={n: rows[n] for n in built.HELPERS})
    out = root/'results'/('solid-chart-v2-build-'+revision)
    binary = write(out/'mesh_conditioned_chart_v2', b'fake never executed', 0o555)
    cgal_paths = {k: root/'qualified-cgal'/n for k, n in (('report', 'report.json'), ('native', 'native.json'), ('binary', 'certified_solid_query'))}
    for p in cgal_paths.values(): write(p)
    measured = {'independently_pinned': True}; image = 'sha256:'+'d'*64
    cgal = dict(child_image_id=image, native_source=build.identity(code/'infra/certified_solid_query.cpp'))
    native = dict(stage='solid_chart_v2_build_native_v1', status='pass', phase='complete', source_binding=binding,
        source_binding_after=binding, build=dict(binary=binary), original_runtime={}, qualified_inputs={'cgal': measured},
        image_id=image, elapsed_seconds=2., mesh_calls=0, source_rehashed_after=True, qualified_inputs_rehashed_after=True,
        gpu_used=False, gt_used=False, native_backend_qualified=False, adopted=False)
    n = write(out/'native.json', json.dumps(native).encode())
    host = dict(stage='solid_chart_v2_build_host_v1', status='pass', phase='complete', native=native, source_binding=binding,
        source_binding_after=binding, retained_binary=binary, image_identity={'child_id': image}, elapsed_seconds=3., mesh_calls=0,
        source_rehashed_after=True, qualified_inputs_rehashed_after=True, owned_container_removed=True, owned_scratch_removed=True,
        gpu_used=False, gt_used=False, native_backend_qualified=False, adopted=False)
    h = write(out/'report.json', json.dumps(host).encode()); out.chmod(0o555)
    pins = dict(schema='world_reward.solid_chart_v2_build_pins.v1', producer_revision=revision, report=h, native=n, binary=binary,
        image_id=image, source_archive_sha256='b'*64, **{k: bound[k] for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256')})
    write(code/gate.BUILD_PINS, json.dumps(pins).encode())
    cert = SimpleNamespace(qualification=lambda *_: (cgal, cgal_paths, measured))
    return code, producer, pins, cert


def test_completed_historical_build_is_authenticated_not_current_source(tmp_path, monkeypatch):
    code, producer, pins, cert = evidence(tmp_path, monkeypatch)
    write(code/'infra/solid_chart_v2_build.py', b'changed current consumer, historical producer untouched')
    result = gate.built_qualification(code, build, cert, built)
    assert result[0] == pins and result[4]['original_source']['source_files'] == pins['source_files']
    assert producer in result[3] and producer.parent/'revision' in result[3]
    assert result[1].parent in result[3] and result[1] not in result[3]  # Preserve authenticated directory0555 in Docker.


@pytest.mark.parametrize('change', ['missing', 'ledger', 'archive', 'producer', 'binary', 'scope', 'chart'])
def test_independent_pins_or_producer_change_stops_before_native(tmp_path, monkeypatch, change):
    code, producer, pins, cert = evidence(tmp_path, monkeypatch)
    if change == 'missing':
        (code/gate.BUILD_PINS).parent.chmod(0o755); (code/gate.BUILD_PINS).unlink()
    elif change in ('ledger', 'archive', 'producer'):
        key = {'ledger': 'source_files_sha256', 'archive': 'source_archive_sha256', 'producer': 'producer_revision'}[change]
        pins[key] = 'e'*(40 if change == 'producer' else 64); write(code/gate.BUILD_PINS, json.dumps(pins).encode())
    elif change == 'binary': write(gate.ROOT/'results'/('solid-chart-v2-build-'+pins['producer_revision'])/'mesh_conditioned_chart_v2', b'tampered', 0o555)
    elif change == 'scope':
        p = gate.ROOT/'results'/('solid-chart-v2-build-'+pins['producer_revision'])/'report.json'
        h = json.loads(p.read_bytes()); h['gpu_used'] = True; pins['report'] = write(p, json.dumps(h).encode()); write(code/gate.BUILD_PINS, json.dumps(pins).encode())
    else: write(code/'infra/mesh_conditioned_chart_v2.hpp', b'changed new chart')
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.built_qualification(code, build, cert, built)


def controls():
    stages = ('physical_source', 'native_candidate', 'float32_glb', 'default8_exact_weld', 'unmodified_official_pack', 'original_grounding_metric_bake')
    manifest = [dict(control=f'fresh_{n}', source_array_sha256=['a'*64, 'b'*64]) for n in range(4)]
    rows = [s | dict(status='pass', phase='complete', native_attempts=1, native_returned=True, native_returncode=0,
        native_mapping={'serialization': {'committed_collapses': 1}}, source_arrays_unchanged=True,
        output_vertices=4096, output_faces=4096, metric_scale_baked_once=.375, stages=dict.fromkeys(stages, {})) for s in manifest]
    report = dict(status='pass', phase='complete', conditioning_version=2, stage='oriented_solid_compiler_controls_v2',
        maximum_native_calls=4, budget_seconds=1800, owned_scratch_removed=True, adoption=False, reconstruction_accuracy_verified=False, controls=rows)
    return manifest, report


@pytest.mark.parametrize('change', ['none', 'zero_collapse', 'short', 'dropped', 'order', 'metric', 'claim', 'bool_count'])
def test_whole_four_control_gate_no_partial_or_adoption(change):
    manifest, report = controls()
    if change == 'zero_collapse': report['controls'][0]['native_mapping']['serialization']['committed_collapses'] = 0
    elif change == 'short': report['controls'].pop()
    elif change == 'dropped': manifest.pop()
    elif change == 'order': report['controls'].reverse()
    elif change == 'metric': report['controls'][0]['metric_scale_baked_once'] = .4
    elif change == 'claim': report['adoption'] = True
    elif change == 'bool_count': report['controls'][0]['native_attempts'] = True
    if change == 'none': gate.validate_controls(report, manifest)
    else:
        with pytest.raises(ValueError): gate.validate_controls(report, manifest)


def test_native_tuple_hashes_and_sorted_json_stages_preserve_six_stage_gate():
    manifest, report = controls()
    for row in report['controls']: row['source_array_sha256'] = tuple(row['source_array_sha256'])
    gate.validate_controls(report, manifest)
    restored = build.strict_json(json.dumps(report, sort_keys=True).encode())
    gate.validate_controls(restored, manifest)
    del restored['controls'][0]['stages']['physical_source']
    with pytest.raises(ValueError): gate.validate_controls(restored, manifest)


def host_fixture(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); (root/'results').mkdir()
    image = 'sha256:'+'d'*64; pins = {'image_id': image}; proof = {'exact_source': True}
    original = root/'jobs'/('c'*40)/'run_solid_chart_v2_build/code'
    paths = (root/'results/frozen-build', original, original.parent/'revision', original.parent/'source-sha256')
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None); monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate, 'built_qualification', lambda *_: (pins, root/'retained-qem', root/'qualified-query', paths, proof))
    monkeypatch.setattr(built, 'qualified', lambda *_: ({}, (root/'cgal-dir', root/'cache-dir'), {}))
    monkeypatch.setattr(built, 'image_identity', lambda *_: {'child_id': image})
    monkeypatch.setattr(gate, 'OFFICIAL_PIN', write(root/gate.OFFICIAL))
    calls = []; cleanup = []; monkeypatch.setattr(build, 'cleanup_container', lambda *a: cleanup.append(a))
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        calls.append((argv, seconds)); out = root/'results'/('solid-chart-v2-qualify-'+revision)
        manifest, full = controls(); before = gate.binding(code, revision, build)
        native = dict(stage='solid_chart_v2_qualification_native_v1', status='pass', phase='complete', source_binding=before,
            source_binding_after=before, source_rehashed_after=True, artifacts_rehashed_after=True, source_arrays_unchanged=True,
            qualified_build=proof, image_id=image, gpu_used=False, gt_used=False, adoption=False,
            reconstruction_accuracy_verified=False, elapsed_seconds=1., control_manifest=manifest, controls=full,
            control_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
        write(out/'disposable/native.json', json.dumps(native).encode()); write(log, b'owned native log')
    monkeypatch.setattr(build, 'run', run)
    return root, code, revision, calls, cleanup, original


def test_host_cpu_exact_mounts_single_invocation_fullseal(tmp_path, monkeypatch):
    root, code, revision, calls, cleanup, original = host_fixture(tmp_path, monkeypatch)
    assert gate.host(code, revision, build, None, built) == 0
    out = root/'results'/('solid-chart-v2-qualify-'+revision); report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'pass' and report['qualified_procedural_controls'] == 4
    assert report['gpu_used'] is report['adoption'] is report['reconstruction_accuracy_verified'] is False
    assert report['owned_container_removed'] and report['owned_scratch_removed'] and not (out/'disposable').exists()
    assert stat.S_IMODE((out/'native.json').stat().st_mode) == 0o444 and stat.S_IMODE(out.stat().st_mode) == 0o555
    argv, seconds = calls[0]; assert len(calls) == 1 and seconds <= 1810
    assert '--gpus' not in argv and argv[argv.index('--user')+1] == '1000:1000' and argv[argv.index('--network')+1] == 'none'
    assert argv[argv.index('--entrypoint')+1] == '/usr/bin/env' and 'WR_ROOT='+str(root) in argv
    mounts = [argv[i+1] for i, x in enumerate(argv) if x == '--mount']
    assert all(f'type=bind,src={p},dst={p},readonly' in mounts for p in (original, original.parent/'revision', code.parent/'source-sha256', root/'cgal-dir', root/gate.OFFICIAL))
    assert cleanup and not any(f'src={root},' in m for m in mounts)
    with pytest.raises(ValueError, match='Fresh'): gate.host(code, revision, build, None, built)


@pytest.mark.parametrize('change', ['cleanup', 'exit', 'source', 'missing', 'partial'])
def test_host_failures_preserve_native_fail_without_qualification(tmp_path, monkeypatch, change):
    root, code, revision, _, _, _ = host_fixture(tmp_path, monkeypatch); run = build.run
    if change == 'cleanup': monkeypatch.setattr(build, 'cleanup_container', lambda *_: (_ for _ in ()).throw(ValueError('owned cleanup refused')))
    else:
        def altered(*args):
            if args[0][1] == 'ps': return b''
            if change == 'missing': return None
            run(*args)
            if change == 'exit': raise ValueError('native nonzero after receipt')
            if change == 'source': write(code.parent/'revision', ('e'*40+'\n').encode())
            if change == 'partial':
                p = root/'results'/('solid-chart-v2-qualify-'+revision)/'disposable/native.json'
                r = json.loads(p.read_bytes()); r.update(status='fail', failure_type='SolidCompilerError'); write(p, json.dumps(r).encode())
        monkeypatch.setattr(build, 'run', altered)
    assert gate.host(code, revision, build, None, built) == 1
    out = root/'results'/('solid-chart-v2-qualify-'+revision); result = json.loads((out/'report.json').read_bytes())
    assert result['status'] == 'fail' and 'qualified_procedural_controls' not in result
    if change != 'missing': assert (out/'native.json').is_file()  # Evidence retained even cleanup/posthashFAIL.


def test_missing_future_build_pin_stops_before_output(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch)
    (code/gate.BUILD_PINS).parent.chmod(0o755); (code/gate.BUILD_PINS).unlink()
    (code/gate.BUILD_PINS).parent.chmod(0o555)
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    with pytest.raises((ValueError, FileNotFoundError)): gate.host(code, revision, build, None, built)
    assert not (root/'results').exists()


@pytest.mark.parametrize('failure', ['none', 'prerequisites', 'abi', 'geometry'])
def test_native_one_frozen_cohort_and_partial_failure_seal(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(np, '__version__', '1.26.3')  # Manufactured qualified-image metadata; no local installation.
    root, code, revision = source(tmp_path, monkeypatch)
    work = root/'results'/('solid-chart-v2-qualify-'+revision)/'disposable'; work.mkdir(parents=True, mode=0o700)
    original_stat = Path.stat
    def native_stat(p, *args, **kwargs):
        s = original_stat(p, *args, **kwargs)
        return SimpleNamespace(st_uid=1000, st_mode=s.st_mode) if p == work else s
    monkeypatch.setattr(Path, 'stat', native_stat); monkeypatch.setattr(gate.sys, 'platform', 'linux')
    monkeypatch.setattr(gate.os, 'getuid', lambda: 1000); monkeypatch.setenv('WR_NATIVE_NETWORK', 'none')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1'); monkeypatch.setenv('WR_CPU_IMAGE_ID', 'sha256:'+'d'*64)
    proof = {'build': {'derivation': {'new': True}, 'build_info': {'chart_version': 2}}, 'original_runtime': {'exact': True}}
    monkeypatch.setattr(gate, 'OFFICIAL_PIN', write(root/gate.OFFICIAL))
    def qualified(*_):
        if failure == 'prerequisites': raise FileNotFoundError('Pinned build unavailable')
        return {'image_id': 'sha256:'+'d'*64}, root/'binary', root/'query', (), proof
    monkeypatch.setattr(gate, 'built_qualification', qualified)
    monkeypatch.setattr(build, 'run', lambda *_: json.dumps({'chart_version': 1 if failure == 'abi' else 2}).encode())
    mesh = (np.zeros((4, 3), np.float64), np.ones((4, 3), np.int64))
    hashes = lambda m: [hashlib.sha256(a.tobytes()).hexdigest() for a in m]
    sources = tuple((f'fresh_{n}', mesh, dict(source_array_sha256=tuple(hashes(mesh)), expected_parents=np.array([-1]),
        expected_signs=np.array([1]), specification_sha256='b'*64)) for n in range(4))
    calls = []
    class Partial(RuntimeError):
        def __init__(self, report): self.report = report
    def geometry(*args, **kwargs):
        calls.append((args, kwargs)); manifest, full = controls()
        for row in full['controls']: row['source_array_sha256'] = tuple(hashes(mesh))
        if failure == 'geometry': full.update(status='fail', failure_type='SourcePreflight'); raise Partial(full)
        return full
    modules = dict(mesh_conditioned_chart_v2_source=SimpleNamespace(derive_cached_backend=lambda *_: (b'exact', {'new': True})),
        oriented_solid_controls_v2=SimpleNamespace(fixtures=lambda: sources, FIXTURE_NAMES=tuple(n for n, _, _ in sources)),
        oriented_solid_compiler=SimpleNamespace(geometry_controls=geometry, SolidCompilerError=Partial, array_hashes=hashes),
        object_budget_conditioned=SimpleNamespace(cache_qualification=lambda *_: (root/'old-no-execute', {}), runtime_identity=lambda *_: {'exact': True}))
    for name, module in modules.items(): monkeypatch.setitem(sys.modules, name, module)
    cert = SimpleNamespace(qualification=lambda *_: ({'native_source': {'sha256': 'c'*64}}, {}, {}))
    assert gate.native(code, revision, work, build, cert, built) == int(failure != 'none')
    report = json.loads((work/'native.json').read_bytes())
    assert report['source_rehashed_after'] and report['status'] == ('pass' if failure == 'none' else 'fail')
    assert len(calls) == int(failure in ('none', 'geometry'))
    if calls:
        assert calls[0][1] == {'controls': sources, 'conditioning_version': 2}
        assert report['control_manifest'] == gate.cohort_manifest(sources) and report['artifacts_rehashed_after']
    if failure == 'geometry': assert report['controls']['failure_type'] == 'SourcePreflight' and 'qualified_procedural_controls' not in report


def test_isolated_bootstrap_and_static_real_closure():
    command = 'import runpy,sys;runpy.run_path(sys.argv[1],run_name="not_main");assert "numpy" not in sys.modules'
    subprocess.run([sys.executable, '-I', '-B', '-S', '-c', command, str(REPO/'infra/solid_chart_v2_qualify.py')], check=True)
    subprocess.run(['bash', '-n', str(REPO/'infra/run_solid_chart_v2_qualify.sh')], check=True)
    import azure_job
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for d in ('infra', 'src', 'configs') for p in (REPO/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files[gate.BUILD_PINS] = b'{}'  # Future config only, never a forged completion pin.
    closure = azure_job.runtime_bundle_paths(files, 'infra/run_solid_chart_v2_qualify.sh')
    assert set(gate.HELPERS) <= set(closure) and 'infra/object_budget_conditioned.py' in closure
    assert 'infra/run_object_budget_volume.sh' not in closure

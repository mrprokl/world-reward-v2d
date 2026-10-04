"""Real host dispatch/old receipt authentication with tiny manufactured bytes.

Only Docker, download, compilation and privilege operations are mocked. No
qualified image, native binary, network or challenge input is used locally.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_module(name, relative, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module); spec.loader.exec_module(module)
    return module


def controls(build, source):
    return dict(stage='certified_solid_procedural_controls_v1', status='pass', phase='complete',
        native_source_sha256=source, source_binary_rehashed_after=True, maximum_calls=15,
        call_seconds=60, inclusive_seconds=900, elapsed_seconds=1.,
        records=[dict(name=name, status='pass') for name in build.CONTROL_NAMES],
        **dict.fromkeys(('challenge_inputs_used', 'gt_used', 'qem_executed', 'adoption',
                        'production_mesh_validated', 'reconstruction_accuracy_verified'), False))


@pytest.fixture
def reuse(tmp_path, monkeypatch):
    build = load_module('wr_reuse_build', 'infra/certified_solid_build.py', monkeypatch)
    certificate = load_module('certified_solid_source_job', 'infra/certified_solid_source_job.py', monkeypatch)
    root = tmp_path / 'root'; (root / 'results').mkdir(parents=True)
    revision, old_revision = 'a' * 40, 'b' * 40
    code = root / 'jobs' / revision / build.ENTRY / 'code'; (code / 'configs').mkdir(parents=True)
    old = root / 'results' / ('certified-solid-build-' + old_revision); old.mkdir()
    image = 'sha256:' + '9' * 64
    old_source = dict(bytes=4, sha256='c' * 64)
    old_binding = dict(native_source=old_source, producer_revision=old_revision)
    binary = old / 'certified_solid_query'; binary.write_bytes(b'old verified binary'); binary.chmod(0o555)
    binary_pin = build.identity(binary, readonly=True)
    false_scope = dict.fromkeys(('gpu_used', 'gt_used', 'adoption', 'production_mesh_validated',
                                'reconstruction_accuracy_verified', 'competition_eligibility_verified'), False)
    old_native = dict(status='pass', phase='complete', image_id=image, compiled_binary=binary_pin,
        source_binding=old_binding, source_binding_after=old_binding,
        controls=controls(build, old_source['sha256']), **false_scope)
    old_host = dict(status='pass', phase='complete', native=old_native, retained_binary=binary_pin,
        child_image_id=image, parent_image_id=build.PARENT, producer_revision=old_revision,
        source_binding=old_binding, source_binding_after=old_binding, **false_scope,
        **dict.fromkeys(('source_rehashed_after', 'parent_rehashed_after', 'build_inputs_rehashed_after',
                        'disposable_build_inputs_removed', 'child_image_parent_verified'), True))
    build.write_json(old / 'native.json', old_native); build.write_json(old / 'report.json', old_host)
    old.chmod(0o555)
    pins = dict(schema='world_reward.certified_solid_qualification_pins.v1', qualified_controls=15,
        producer_revision=old_revision, child_image_id=image, parent_image_id=build.PARENT,
        native_source=old_source, binary=binary_pin,
        report=build.identity(old / 'report.json', readonly=True),
        native=build.identity(old / 'native.json', readonly=True))
    build.write_json(code / certificate.QUALIFICATION, pins)
    binding = dict(native_source=dict(bytes=7, sha256='d' * 64), producer_revision=revision)
    case = SimpleNamespace(build=build, certificate=certificate, root=root, code=code, revision=revision,
        old=old, image=image, binding=binding, pins=pins, fault=None, downloads=[], commands=[],
        cleanup=[], source_calls=0, prior_calls=0, parent_probe_calls=0)
    case.old_bytes = {p: p.read_bytes() for p in old.iterdir()}
    case.old_stats = {p: (p.stat().st_ino, p.stat().st_mode) for p in old.iterdir()}
    monkeypatch.setattr(build, 'ROOT', root); monkeypatch.setattr(certificate, 'ROOT', root)
    monkeypatch.setattr(build.sys, 'platform', 'linux')
    monkeypatch.setattr(build.os, 'getuid', lambda: 0); monkeypatch.setattr(build.os, 'chown', lambda *_: None)
    monkeypatch.setenv('DOCKER_HOST', 'unix://' + str(root / 'docker.sock'))
    monkeypatch.setattr(build.signal, 'signal', lambda *_: None); monkeypatch.setattr(build.signal, 'alarm', lambda *_: None)
    def source_binding(*_):
        case.source_calls += 1
        if case.fault == 'source_before': raise ValueError('source before')
        result = copy.deepcopy(binding)
        if case.fault == 'source_after' and case.source_calls > 1: result['producer_revision'] = 'e' * 40
        return result
    monkeypatch.setattr(build, 'source_binding', source_binding)
    original_qualification = certificate.qualification
    def qualification(*args):
        case.prior_calls += 1
        return original_qualification(*args)
    monkeypatch.setattr(certificate, 'qualification', qualification)
    parent = dict(python='3.11.synthetic', numpy='1.26.3', compiler='synthetic', cmake='synthetic',
        packages={**build.EXPECTED['installed_libraries'], 'libc6': 'unchanged'})
    child = copy.deepcopy(parent); child['packages'].update({p['package']: p['version'] for p in build.DEBS})
    def probe(selected, _seconds):
        if selected == build.PARENT:
            case.parent_probe_calls += 1
            if case.fault == 'parent_after' and case.parent_probe_calls > 1: return dict(parent, compiler='changed')
            return copy.deepcopy(parent)
        assert selected == image
        return copy.deepcopy(child)
    monkeypatch.setattr(build, 'probe', probe)
    def download(pin, path, _deadline):
        case.downloads.append(pin['url'])
        assert pin in (build.CGAL, build.SUM), 'Runtime reuse must not download dependencies'
        data = ((build.CGAL['sha256'] + '  CGAL-6.0.1-library.tar.xz\n').encode()
                if pin == build.SUM else b'manufactured CGAL archive')
        path.write_bytes(data)
        return build.identity(path)
    monkeypatch.setattr(build, 'download', download)
    def extract(_archive, target, capacities):
        assert capacities == build.EXPECTED['capacities']
        target.mkdir(); (target / 'exact.hpp').write_bytes(b'fresh synthetic exact headers')
        return dict(header_files=1, header_inventory_sha256='f' * 64)
    monkeypatch.setattr(build, 'extract_headers', extract)
    monkeypatch.setattr(build, 'cleanup_container', lambda *args: case.cleanup.append(args))
    def run(argv, seconds, log=None):
        case.commands.append(argv); assert 0 < seconds <= 1505
        if argv[:3] == ['docker', 'image', 'inspect']:
            actual = argv[3]
            assert actual in (build.PARENT, image)
            if actual == image and case.fault == 'image_wrong': actual = 'sha256:' + '8' * 64
            return (actual + '\n').encode()
        assert argv[:2] == ['docker', 'run'] and '--native' in argv
        assert '--reuse-qualified-runtime' not in argv and '--network' in argv and argv[argv.index('--network') + 1] == 'none'
        for flag, value in (('--user', '1000:1000'), ('--cap-drop', 'ALL'), ('--cpus', '4'), ('--memory', '16g')):
            assert argv[argv.index(flag) + 1] == value
        assert '--read-only' in argv and '--gpus' not in argv
        work = Path(argv[argv.index('--work') + 1]); new_binary = work / 'compiled-query'
        new_binary.write_bytes(b'new query requires new controls')
        native = dict(status='pass', phase='complete', source_binding=copy.deepcopy(binding),
            source_binding_after=copy.deepcopy(binding), source_rehashed_after=True, image_id=image,
            controls=controls(build, binding['native_source']['sha256']), compiled_binary=build.identity(new_binary))
        if case.fault == 'old_controls': native['controls'] = controls(build, old_source['sha256'])
        if case.fault == 'missing_control': native['controls']['records'].pop()
        if case.fault == 'native_failure': native.update(status='fail', phase='procedural_controls')
        build.write_json(work / 'native.json', native)
        if case.fault == 'headers_after': (work.parent / 'context/include/exact.hpp').write_bytes(b'mutated headers')
        if case.fault == 'prior_after':
            original_binary = old / 'certified_solid_query'; original_binary.chmod(0o755)
            original_binary.write_bytes(b'original binary externally mutated'); original_binary.chmod(0o555)
        if case.fault in ('scratch_replaced', 'scratch_alias'):
            scratch = work.parent; retained = scratch.with_name('retained-original-disposable')
            scratch.rename(retained)
            foreign = scratch if case.fault == 'scratch_replaced' else scratch.with_name('foreign-disposable')
            foreign.mkdir(); (foreign / 'untouched').write_bytes(b'foreign must survive')
            if case.fault == 'scratch_alias': scratch.symlink_to(foreign, target_is_directory=True)
            case.foreign = foreign
        if case.fault == 'native_failure': raise ValueError('new native failed')
        return b''
    monkeypatch.setattr(build, 'run', run)
    return case


def invoke(case):
    result = case.build.host(case.code, case.revision, reuse_qualified_runtime=True)
    out = case.root / 'results' / ('certified-solid-build-' + case.revision)
    return result, out, case.build.strict_json((out / 'report.json').read_bytes())


def assert_old_unchanged(case):
    assert {p: p.read_bytes() for p in case.old.iterdir()} == case.old_bytes
    assert {p: (p.stat().st_ino, p.stat().st_mode) for p in case.old.iterdir()} == case.old_stats
    assert not any(command[:3] in (['docker', 'image', 'rm'], ['docker', 'image', 'tag']) for command in case.commands)
    assert not any('build' in command for command in case.commands)


def test_reused_runtime_success_still_qualifies_only_fresh_source_and_all_fifteen(reuse):
    result, out, report = invoke(reuse)
    assert result == 0 and report['status'] == 'pass'
    assert reuse.prior_calls == 2 and reuse.downloads == [reuse.build.CGAL['url'], reuse.build.SUM['url']]
    assert report['reused_qualified_runtime']['current_query_qualified_by_reuse'] is False
    assert report['reused_qualified_runtime']['image_rebuilt'] is False
    assert report['reused_runtime_rehashed_after'] and report['build_inputs_rehashed_after']
    assert report['native']['controls']['native_source_sha256'] == 'd' * 64 != reuse.pins['native_source']['sha256']
    assert (out / 'certified_solid_query').read_bytes() == b'new query requires new controls'
    assert not (out / 'disposable').exists()
    assert_old_unchanged(reuse)


def test_prior_pin_failure_stops_before_download_or_new_native_execution(reuse):
    path = reuse.code / reuse.certificate.QUALIFICATION
    pins = copy.deepcopy(reuse.pins); pins['binary']['sha256'] = '0' * 64
    path.chmod(0o644); path.write_text(json.dumps(pins)); path.chmod(0o444)
    result, out, report = invoke(reuse)
    assert result == 1 and report['status'] == 'fail' and not reuse.downloads
    assert not any(command[:2] == ['docker', 'run'] for command in reuse.commands)
    assert not (out / 'certified_solid_query').exists()
    assert report['disposable_build_inputs_removed'] is True and not (out / 'disposable').exists()
    assert_old_unchanged(reuse)


@pytest.mark.parametrize('fault', ['native_failure', 'old_controls', 'missing_control', 'source_after', 'parent_after', 'image_wrong'])
def test_reuse_failure_never_removes_or_retags_original_image_and_cannot_publish_query(reuse, fault):
    reuse.fault = fault
    result, out, report = invoke(reuse)
    assert result == 1 and report['status'] == 'fail'
    assert not (out / 'certified_solid_query').exists() and not (out / 'native.json').exists()
    assert report['disposable_build_inputs_removed'] is True and not (out / 'disposable').exists()
    if fault == 'image_wrong': assert not reuse.downloads
    assert_old_unchanged(reuse)


def test_current_source_failure_precedes_namespace_and_all_runtime_activity(reuse):
    reuse.fault = 'source_before'
    with pytest.raises(ValueError, match='source before'): invoke(reuse)
    assert not reuse.commands and not reuse.downloads and not reuse.prior_calls
    assert not (reuse.root / 'results' / ('certified-solid-build-' + reuse.revision)).exists()
    assert_old_unchanged(reuse)


def test_original_binary_mutation_after_native_prevents_retention_without_image_deletion(reuse):
    reuse.fault = 'prior_after'
    result, out, report = invoke(reuse)
    assert result == 1 and report['status'] == 'fail' and not (out / 'certified_solid_query').exists()
    assert not report.get('reused_runtime_rehashed_after', False)
    assert report['disposable_build_inputs_removed'] is True and not (out / 'disposable').exists()
    assert (reuse.old / 'certified_solid_query').read_bytes() == b'original binary externally mutated'
    assert not any(command[:3] == ['docker', 'image', 'rm'] for command in reuse.commands)


def test_acquired_header_posthash_failure_prevents_retention(reuse):
    reuse.fault = 'headers_after'
    result, out, report = invoke(reuse)
    assert result == 1 and report['status'] == 'fail' and not (out / 'certified_solid_query').exists()
    assert not report.get('build_inputs_rehashed_after', False)
    assert report['disposable_build_inputs_removed'] is True and not (out / 'disposable').exists()
    assert_old_unchanged(reuse)


@pytest.mark.parametrize('fault', ['scratch_replaced', 'scratch_alias'])
def test_replaced_or_aliased_namespace_cannot_trigger_foreign_cleanup_or_pass(reuse, fault):
    reuse.fault = fault
    result, out, report = invoke(reuse)
    assert result == 1 and report['status'] == 'fail' and report['cleanup_failure_type'] == 'ValueError'
    assert not report.get('disposable_build_inputs_removed', False)
    assert (reuse.foreign / 'untouched').read_bytes() == b'foreign must survive'
    assert not (out / 'certified_solid_query').exists()
    assert_old_unchanged(reuse)

"""New fixed query composition with tiny immutable evidence, never test data."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_solid_chart_query_requalification import composition, certificate
from test_solid_chart_v2_qualify import build, built, controls, gate as qual, write
from test_object_budget_solid import gate, inputs, native_result


@pytest.fixture
def production(composition, monkeypatch):
    c = composition; source_code = c.code; source_revision = c.revision
    source_rows, source = qual.snapshot(source_code, source_revision, build)
    bound = source | dict(helpers={n: source_rows[n] for n in qual.HELPERS+(qual.BALANCED_PINS, 'infra/certified_solid_query.cpp')})
    buildpins, binary, query, buildpaths, buildproof = qual.built_qualification(source_code, build, certificate, built, query_requalification=True)
    manifest, full = controls()
    common = dict(status='pass', phase='complete', source_binding=bound, source_binding_after=bound,
        source_rehashed_after=True, artifacts_rehashed_after=True, official_rehashed_after=True,
        gpu_used=False, gt_used=False, adoption=False, reconstruction_accuracy_verified=False,
        competition_eligibility_verified=False, qualified_build=buildproof, qualified_procedural_controls=4,
        query_requalification=buildproof['query_requalification'])
    native = common | dict(stage='solid_chart_v2_qualification_native_v1', elapsed_seconds=1., source_arrays_unchanged=True,
        image_id=c.pins['image_id'], control_manifest=manifest, controls=full,
        control_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    host = common | dict(stage='solid_chart_v2_qualification_host_v1', elapsed_seconds=2., native=native,
        owned_container_removed=True, owned_scratch_removed=True)
    out = c.root/'results'/('solid-chart-v2-query-requalify-'+source_revision)
    native_pin = write(out/'native.json', json.dumps(native).encode())
    report_pin = write(out/'report.json', json.dumps(host).encode()); out.chmod(0o555)
    pins = dict(schema='world_reward.solid_chart_v2_qualification_pins.v1', producer_revision=source_revision,
        report=report_pin, native=native_pin, build_pins=build.identity(source_code/qual.BUILD_PINS),
        source_archive_sha256='b'*64, **{k: source[k] for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256')},
        qualified_controls=4, native_qem_calls=4, native_query_calls=40, adoption=False, reconstruction_accuracy_verified=False)
    revision = 'f'*40; code = c.root/'jobs'/revision/gate.ENTRY/'code'
    shutil.copytree(source_code, code)
    for p in (code, *code.rglob('*')):
        if p.is_dir(): p.chmod(0o755)
    for name in gate.HELPERS:
        if not (code/name).exists(): write(code/name)
    write(code/gate.BALANCED_PINS, json.dumps(pins).encode())
    write(code.parent/'revision', (revision+'\n').encode()); write(code.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in (code, *code.rglob('*')):
        if p.is_dir(): p.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', c.root)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/object_budget_solid.py'))
    return SimpleNamespace(c=c, root=c.root, code=code, revision=revision, producer=source_code,
        pins=pins, out=out, buildproof=buildproof, binary=binary, query=query)


def rewrite_qualification(p, mutate):
    host = json.loads((p.out/'report.json').read_bytes()); mutate(host)
    p.pins['native'] = write(p.out/'native.json', json.dumps(host['native']).encode())
    p.pins['report'] = write(p.out/'report.json', json.dumps(host).encode())
    write(p.code/gate.BALANCED_PINS, json.dumps(p.pins).encode())


def test_frozen_four_control_composition_selects_only_active_query(production):
    p = production
    pins, binary, query, mounts, proof = gate.qualification(p.code, qual, build, certificate, built, query_requalification=True)
    assert pins == p.c.pins and binary == p.binary and query == p.c.active_paths['binary']
    assert proof['build']['cgal'] == p.c.old_measured
    assert proof['query_requalification'] == p.buildproof['query_requalification']
    assert proof['query_requalification']['active_query']['artifacts'] == p.c.active_measured
    assert p.producer in mounts and p.producer.parent/'revision' in mounts
    assert p.c.active_paths['binary'].parent in mounts and p.c.old_paths['binary'].parent in mounts
    binding = gate.binding(p.code, p.revision, qual, build, query_requalification=True)
    assert gate.BALANCED_PINS in binding['helpers'] and gate.PINS not in binding['helpers']
    assert gate.PINS in gate.binding(p.code, p.revision, qual, build)['helpers']


@pytest.mark.parametrize('fault', ['missing', 'historical_source', 'active_source', 'counts', 'native_missing_composition',
    'host_changed_composition', 'partial_four_controls', 'unqualified_current_query', 'old_query_binary', 'image'])
def test_qualification_stops_before_inputs_and_outputs_on_unproved_composition(production, fault):
    p = production
    if fault == 'missing': (p.code/'configs').chmod(0o755); (p.code/gate.BALANCED_PINS).unlink()
    elif fault == 'historical_source': write(p.producer/'infra/oriented_solid_compiler.py', b'changed producer')
    elif fault == 'active_source': write(p.code/'infra/oriented_solid_compiler.py', b'changed active mathematical source')
    elif fault == 'counts': p.pins['native_query_calls'] = 39; write(p.code/gate.BALANCED_PINS, json.dumps(p.pins).encode())
    elif fault == 'native_missing_composition': rewrite_qualification(p, lambda h: h['native'].pop('query_requalification'))
    elif fault == 'host_changed_composition': rewrite_qualification(p, lambda h: h.update(query_requalification={'recompiled': True}))
    elif fault == 'partial_four_controls': rewrite_qualification(p, lambda h: h['native']['controls']['controls'].pop())
    elif fault == 'unqualified_current_query': write(p.code/'infra/certified_solid_query.cpp', b'not qualified source')
    elif fault == 'old_query_binary': write(p.c.old_paths['binary'], b'changed original', 0o555)
    else: rewrite_qualification(p, lambda h: h['native'].update(image_id='sha256:'+'0'*64))
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.qualification(p.code, qual, build, certificate, built, query_requalification=True)
    assert not (p.root/'outputs').exists()


@pytest.mark.parametrize('invalid', ['arbitrary/path', 1, None])
def test_profile_not_a_free_path_or_truthy_value(production, invalid):
    with pytest.raises(ValueError, match='profile'):
        gate.qualification(production.code, qual, build, certificate, built, query_requalification=invalid)


def test_missing_balanced_pin_cannot_fallback_to_valid_historical_profile(tmp_path, monkeypatch):
    from test_object_budget_solid import evidence
    _, code, _, _, _ = evidence(tmp_path, monkeypatch)
    assert gate.qualification(code, qual, build, None, None)[4]['source']
    with pytest.raises(FileNotFoundError):
        gate.qualification(code, qual, build, None, None, query_requalification=True)


def test_host_missing_actual_balanced_pin_stops_before_any_input_read_or_namespace(production,monkeypatch):
    p=production; (p.code/'configs').chmod(0o755); (p.code/gate.BALANCED_PINS).unlink(); (p.code/'configs').chmod(0o555)
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'getuid',lambda:0)
    monkeypatch.setenv('DOCKER_HOST','unix://'+str(p.root/'docker.sock'))
    def forbidden(*args,**kwargs):raise AssertionError('Input/runtime dispatch before independent actual pins')
    monkeypatch.setattr(gate,'input_binding',forbidden);monkeypatch.setattr(build,'run',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):
        gate.host(8,p.code,p.revision,qual,build,certificate,built,None,query_requalification=True)
    assert not (p.root/'outputs').exists()


@pytest.mark.parametrize('fault', ['none', 'exit', 'active_posthash', 'original_posthash', 'input_posthash', 'proof', 'runtime', 'image'])
def test_host_narrow_new_namespace_same_one_native_call_and_no_recompile(production, monkeypatch, fault):
    p = production; body, _ = inputs(p.root)
    result = gate.qualification(p.code, qual, build, certificate, built, query_requalification=True); proof = result[4]
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_: None)
    monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(p.root/'docker.sock'))
    monkeypatch.setattr(qual, 'OFFICIAL_PIN', write(p.root/qual.OFFICIAL, mode=0o644))
    qualified_codes = []; runtime = copy.deepcopy(p.c.native['qualified_inputs'])
    def qualified(code, *_):
        qualified_codes.append(code); assert code == p.c.producer
        return p.c.old, (p.c.old_paths['binary'].parent, p.root/'cache-namespace'), copy.deepcopy(runtime)
    monkeypatch.setattr(built, 'qualified', qualified)
    image_calls = []
    def image(*_):
        image_calls.append(1); return dict(child_id='sha256:'+'0'*64 if fault == 'image' and len(image_calls)>1 else p.c.pins['image_id'])
    monkeypatch.setattr(built, 'image_identity', image)
    cleanup = []; monkeypatch.setattr(build, 'cleanup_container', lambda *args: cleanup.append(args))
    out = p.root/'outputs/episode_000008'/('object_budget_solid_balanced_'+p.revision)
    old = p.root/'outputs/episode_000008'/('object_budget_solid_'+p.revision)
    write(old/'report.json', b'historical failed receipt untouched'); old.chmod(0o555)
    old_pin = build.identity(old/'report.json'); old_inode = (old/'report.json').stat().st_ino
    calls = []
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        assert argv[:2] == ['docker', 'run'] and seconds <= 1810
        calls.append(argv)
        outputs = {n: write(out/'disposable'/n, ('opaque '+n).encode()) for n in gate.OUTPUTS}
        native = native_result(8, gate.binding(p.code,p.revision,qual,build,query_requalification=True), proof,
            gate.input_binding(8,body,build)[2], outputs)
        native['query_requalification'] = copy.deepcopy(proof['query_requalification'])
        if fault == 'proof': native['query_requalification']['original_qem_recompiled'] = True
        write(out/'disposable/native.json', json.dumps(native).encode()); write(log)
        if fault == 'active_posthash': write(p.c.active_paths['binary'], b'changed active query', 0o555)
        if fault == 'original_posthash': write(p.c.producer/'infra/mesh_conditioned_chart_v2.hpp', b'changed original header')
        if fault == 'input_posthash': write(p.root/'outputs/episode_000008/object_grounded/object.glb', b'changed source prediction')
        if fault == 'runtime': runtime['cache'] = {'changed': True}
        if fault == 'exit': raise ValueError('native exit nonzero')
    monkeypatch.setattr(build, 'run', run)
    assert gate.host(8,p.code,p.revision,qual,build,certificate,built,body,query_requalification=True) == int(fault != 'none')
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == ('pass' if fault == 'none' else 'fail')
    assert calls and len(calls) == 1 and cleanup and qualified_codes
    assert not (out/'disposable').exists()
    assert build.identity(old/'report.json') == old_pin and (old/'report.json').stat().st_ino == old_inode
    argv = calls[0]; assert '--native' in argv and '--query-requalification' in argv and '--gpus' not in argv
    for option,value in (('--network','none'),('--cpus','4'),('--memory','16g'),('--user','1000:1000')):
        assert argv[argv.index(option)+1] == value
    mounts = [argv[i+1] for i,v in enumerate(argv) if v == '--mount']
    for path in (p.producer,p.producer.parent/'revision',p.c.producer,p.c.old_paths['binary'].parent,p.c.active_paths['binary'].parent):
        assert f'type=bind,src={path},dst={path},readonly' in mounts
    assert sum(not m.endswith(',readonly') for m in mounts) == 1
    if fault == 'none': assert report['outputs'] == report['native']['outputs'] and report['maximum_query_calls'] == 8
    else: assert 'outputs' not in report and not any((out/n).exists() for n in gate.OUTPUTS)


def test_native_supplies_active_query_sha_to_unchanged_produce(production, monkeypatch):
    import object_budget_conditioned as cache
    p = production; body, _ = inputs(p.root)
    work = p.root/'outputs/episode_000008'/('object_budget_solid_balanced_'+p.revision)/'disposable'
    work.mkdir(mode=0o700,parents=True)
    real_stat = Path.stat
    def native_stat(path,*args,**kwargs):
        value = real_stat(path,*args,**kwargs)
        return SimpleNamespace(st_uid=1000,st_mode=value.st_mode) if path == work else value
    monkeypatch.setattr(Path,'stat',native_stat); monkeypatch.setattr(gate.os,'getuid',lambda:1000)
    monkeypatch.setattr(gate.sys,'platform','linux'); monkeypatch.setenv('WR_NATIVE_NETWORK','none')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1'); monkeypatch.setenv('WR_CPU_IMAGE_ID',p.c.pins['image_id'])
    monkeypatch.setattr(qual,'OFFICIAL_PIN',write(p.root/qual.OFFICIAL,mode=0o644))
    monkeypatch.setattr(cache,'cache_qualification',lambda *_:({},{})); monkeypatch.setattr(cache,'runtime_identity',lambda *_:{})
    calls=[]
    def produce(episode,binary,query,query_sha,scratch,official,left,report,_):
        calls.append((binary,query,query_sha))
        assert binary == p.binary and query == p.query and query_sha == p.c.active['native_source']['sha256']
        paths=tuple(scratch/n for n in gate.OUTPUTS)
        for path in paths:write(path,b'opaque candidate')
        report['compiler']=native_result(8,{},{},{'video_sha256':'a'*64},{n:build.identity(work/n) for n in gate.OUTPUTS})['compiler']
        return paths
    monkeypatch.setattr(gate,'produce',produce)
    assert gate.native(8,p.code,p.revision,work,qual,build,certificate,built,body,query_requalification=True)==0
    report=json.loads((work/'native.json').read_bytes())
    assert len(calls)==1 and report['native_query_calls']==8 and report['maximum_qem_calls']==1
    assert report['query_requalification']['active_query']['artifacts']==p.c.active_measured
    assert report['inputs_qualification_rehashed_after'] and report['source_rehashed_after']


def test_cli_profile_is_fixed_and_uses_disjoint_native_namespace(tmp_path,monkeypatch):
    revision='a'*40;code=tmp_path/'jobs'/revision/gate.ENTRY/'code';captured=[]
    monkeypatch.setattr(gate,'ROOT',tmp_path); monkeypatch.setattr(gate.sys,'argv',['driver','--episode','8','--native','--query-requalification'])
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',revision)
    monkeypatch.setattr(gate,'helpers',lambda *_:())
    monkeypatch.setattr(gate,'native',lambda *args,**kwargs:captured.append((args,kwargs)) or 0)
    assert gate.main()==0
    assert captured[0][0][3]==tmp_path/'outputs/episode_000008'/('object_budget_solid_balanced_'+revision)/'disposable'
    assert captured[0][1]=={'query_requalification':True}
    for extra in (['--query-requalification','--query-requalification'],['--query-requalification','--native'],['--query-requalification=1']):
        monkeypatch.setattr(gate.sys,'argv',['driver','--episode','8',*extra])
        with pytest.raises(ValueError):gate.main()


def test_produce_and_numerical_validation_ast_are_unchanged():
    repo=Path(__file__).resolve().parents[1]
    original=subprocess.check_output(['git','show','HEAD:infra/object_budget_solid.py'],cwd=repo)
    before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((repo/'infra/object_budget_solid.py').read_bytes()).body if isinstance(n,ast.FunctionDef)}
    assert all(before[name]==after[name] for name in ('produce','validate_native','successful_queries','query_record','validate_compiler_proof','input_binding','publish'))


def test_shell_profile_syntax_static_closure_and_no_free_path():
    repo=Path(__file__).resolve().parents[1];wrapper=repo/'infra/run_object_budget_solid.sh'
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    for args in (['--episode','8','--query-requalification','evil'],['--episode','8','--pins','elsewhere'],['--query-requalification','--episode','8']):
        assert subprocess.run(['bash',str(wrapper),*args],capture_output=True,env={'PATH':os.defpath}).returncode==2
    import azure_job
    files={p.relative_to(repo).as_posix():p.read_bytes() for d in ('infra','src','configs') for p in (repo/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files[gate.BALANCED_PINS]=b'{}'  # Closure fixture only, not an actual qualification.
    closure=azure_job.runtime_bundle_paths(files,wrapper.relative_to(repo).as_posix())
    assert {gate.BALANCED_PINS,qual.BALANCED_PINS,'infra/certified_solid_query.cpp'}<=set(closure)

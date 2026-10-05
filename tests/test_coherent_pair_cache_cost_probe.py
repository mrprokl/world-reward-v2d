"""Tiny actual numerical parity and mocked host lifecycle; no cloud or realdata."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO =Path(__file__).resolve().parents[1]
spec =importlib.util.spec_from_file_location('paired_probe_test', REPO/'infra/coherent_pair_cache_cost_probe.py')
m =importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rt, fixture =m.helpers(REPO)


def progress(): return dict(fixtures=[], rows=[])


def test_real_tiny_six_paired_rows_exact_all_arrays_loss_counts_and_projection():
    p =progress(); m.measure(np, fixture, lambda: None, p, object_count=8)
    assert len(p['rows']) ==6 and len(p['fixtures']) ==2
    assert [(r['case'], r['coefficient']) for r in p['rows']] ==[(i, n)for i in range(2)for n in ('zero_geometry', 'dyadic_geometry', 'dyadic_relational')]
    assert all(r['loss_gradient_bits_exact'] and r['full_scores_arrays_metadata_bits_exact'] for r in p['rows'])
    assert [len(r['gradient'])for r in p['rows']] ==[17, 17, 1, 17, 17, 1]
    expected =max(r['cached_seconds']for r in p['rows'])*32*1026+32*max(r['cache_preparation_seconds']for r in p['fixtures'])
    assert p['projected_recipe_seconds'] ==expected and not p['full_fit_executed']
    assert p['source_fixtures_rehashed_after'] and p['cache_factors_rehashed_after']


def test_complete_comparison_preserves_nan_signedzero_dtype_and_metadata():
    left =dict(raw=np.array([0., -0., np.nan]), names=('a', 'b'))
    m.identical(np, left, dict(raw=left['raw'].copy(), names=('a', 'b')))
    for right in (dict(raw=np.array([0., 0., np.nan]), names=('a', 'b')),
                  dict(raw=left['raw'].astype(np.float32), names=('a', 'b')),
                  dict(raw=left['raw'], names=('b', 'a'))):
        with pytest.raises(ValueError): m.identical(np, left, right)


def test_interruption_after_original_loss_stops_without_cached_retry_or_partial_pass(monkeypatch):
    from world_reward import coherent_pair_learning as old, coherent_pair_cache as cached
    calls =[]; original =old.loss_gradient; count =[]
    def run(*args, **kwargs): calls.append('original'); return original(*args, **kwargs)
    monkeypatch.setattr(old, 'loss_gradient', run)
    monkeypatch.setattr(cached, 'loss_gradient', lambda *a, **k: calls.append('cached'))
    def check():
        count.append(True)
        if len(count) ==5: raise TimeoutError('fixedstop')
    p =progress()
    with pytest.raises(TimeoutError): m.measure(np, fixture, check, p, object_count=8)
    assert calls ==['original'] and len(p['fixtures']) ==1 and not p['rows'] and 'decision' not in p


def test_bit_mismatch_closes_first_row_and_never_proclaims_cached_success(monkeypatch):
    from world_reward import coherent_pair_cache as cached
    original =cached.loss_gradient
    def bad(*args, **kwargs):
        loss, grad, counts =original(*args, **kwargs)
        return np.nextafter(loss, np.inf), grad, counts
    monkeypatch.setattr(cached, 'loss_gradient', bad)
    p =progress()
    with pytest.raises(ValueError, match='loss bits'): m.measure(np, fixture, lambda: None, p, object_count=8)
    assert not p['rows'] and 'decision' not in p


def test_host_import_denies_numpy_torch_transformers_without_host_dependencies():
    script =f'''import importlib.abc,importlib.util,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','torch','transformers'):raise ImportError('no host numerical packages')
sys.meta_path.insert(0,Block())
s=importlib.util.spec_from_file_location('paired_host',{str(REPO/'infra/coherent_pair_cache_cost_probe.py')!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rt,fixture=m.helpers(__import__('pathlib').Path({str(REPO)!r}))
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', script], check=True, capture_output=True)


def test_exact_config_all_old_fixture_and_new_cache_hash_pins(tmp_path):
    source =dict(helpers={n:rt.identity(REPO/n, readonly=False) for n in m.HELPERS})
    p =tmp_path/m.CONFIG; p.parent.mkdir(); p.write_bytes((REPO/m.CONFIG).read_bytes()); p.chmod(0o444)
    c =m.configuration(rt, tmp_path, source)
    assert c['object_slots'] ==3600 and c['prior_cost_source_reference']['experiment_retried'] is False
    source['helpers']['src/world_reward/coherent_pair_cache.py']['sha256'] ='0'*64
    with pytest.raises(ValueError, match='Frozen'): m.configuration(rt, tmp_path, source)


def receipt(binding, revision, prior):
    v =dict(schema='world_reward.coherent_pair_cache_cost_probe.v1', stage='coherent_pair_cache_cost_native', status='pass', phase='complete',
        source_binding=binding, producer_revision=revision, image_id=m.IMAGE, rows_completed=6, source_rehashed_after=True,
        source_fixtures_rehashed_after=True, cache_factors_rehashed_after=True, full_fit_executed=False, prior_cost_source_reference=prior,
        gpu_used=False, models_loaded=False, rgb_read=False, references_read=False, challenge_inputs_used=False, quality_verified=False, adoption=False)
    v['fixtures'] =[dict(case=i, persons=n, objects=3600, native_pairs=k, native_tokens=1500, candidate_rows=n*2*3600, tuple_rows=n*2*k,
        bridge_rows=k*3600, person_groups=n, object_groups=3600, cache_preparation_seconds=.1,
        bank_sha256=str(i)*64, factor_sha256='b'*64, full_original_observations_sha256=str(i+2)*64)for i, (n, k)in enumerate(((2, 4), (4, 64)))]
    v['rows'] =[dict(case=i, coefficient=name, original_seconds=.5, cached_seconds=.1, loss=1., gradient=[0.]*length,
        loss_gradient_bits_exact=True, full_scores_arrays_metadata_bits_exact=True, max_rss_bytes=100000000, full_scores_sha256='e'*64,
        counts=dict(fit_records=1, used=1, missing_positive=0, no_alternative=0))for i in range(2)for name, length in (('zero_geometry', 17), ('dyadic_geometry', 17), ('dyadic_relational', 1))]
    v.update(projected_recipe_seconds=.1*32*1026+32*.1, decision='CACHE_RECIPE_COST_UNQUALIFIED_NO_REAL_FIT')
    return v


@pytest.mark.parametrize('fault', ['none', 'partial', 'tokens', 'objects', 'parity', 'gradient', 'projection', 'prior'])
def test_native_validator_rejects_partial_or_false_parity(fault):
    prior =dict(receipts_opened=False, experiment_retried=False); binding =dict(source='frozen'); v =receipt(binding, 'b'*40, prior)
    if fault =='partial': v['rows'].pop()
    elif fault =='tokens': v['fixtures'][0]['native_tokens'] =1499
    elif fault =='objects': v['fixtures'][1]['objects'] =3599
    elif fault =='parity': v['rows'][1]['full_scores_arrays_metadata_bits_exact'] =False
    elif fault =='gradient': v['rows'][2]['gradient'] =[0.]*17
    elif fault =='projection': v['projected_recipe_seconds'] =0.
    elif fault =='prior': v['prior_cost_source_reference'] ={}
    if fault =='none': m.validate_native(rt, v, binding, 'b'*40, prior)
    else:
        with pytest.raises(ValueError): m.validate_native(rt, v, binding, 'b'*40, prior)


@pytest.mark.parametrize('fault', ['none', 'timeout', 'postsource', 'foreign', 'umask077', 'late_seal'])
def test_mock_real_host_leaf_mount_shared_deadline_cleanup_and_late_demotions(tmp_path, monkeypatch, fault):
    root =tmp_path/'root'; (root/'results').mkdir(parents=True); revision ='d'*40; code =root/'jobs'/revision/m.ENTRY/'code'; code.mkdir(parents=True)
    prior =json.loads((REPO/m.CONFIG).read_bytes())['prior_cost_source_reference']; binding =dict(helpers={}, closure_sha256='f'*64); before =dict(source_binding=binding, image=dict(Id=m.IMAGE))
    monkeypatch.setattr(m, 'ROOT', root); monkeypatch.setattr(m, 'helpers', lambda _: (rt, fixture)); monkeypatch.setattr(m, 'configuration', lambda *a:dict(schema='world_reward.coherent_pair_cache_cost_probe.v1', prior_cost_source_reference=prior))
    out =root/'results'/('coherent-pair-cache-cost-probe-'+revision)
    def proof(*args):
        if fault =='postsource' and (out/'native.json').exists(): raise ValueError('postsource')
        return before
    monkeypatch.setattr(m, 'proof', proof); monkeypatch.setattr(m, 'leaves', lambda _: [code/'source.py'])
    monkeypatch.setattr(m.os, 'chown', lambda *a:None); original_stat =Path.stat
    class Owner:
        def __init__(self, value): self.value =value; self.st_uid =1000
        def __getattr__(self, name): return getattr(self.value, name)
    monkeypatch.setattr(Path, 'stat', lambda p,*a,**k:Owner(original_stat(p,*a,**k))if p ==out else original_stat(p,*a,**k))
    monkeypatch.setattr(m.signal, 'signal', lambda *a:None); monkeypatch.setattr(m.signal, 'setitimer', lambda *a:None)
    calls =[]; cid ='c'*64
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] ==['docker', 'run']:
            assert out.stat().st_mode&0o777 ==0o755
            (out/'container.cid').write_text(cid)
            if fault =='timeout': raise subprocess.TimeoutExpired(argv, 720)
            p =out/'native.json'; p.write_text(json.dumps(receipt(binding, revision, prior))); p.chmod(0o444)
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
        if fault =='foreign': return SimpleNamespace(returncode=0, stdout=(cid+'|'+m.IMAGE+'|/foreign|'+revision).encode(), stderr=b'')
        return SimpleNamespace(returncode=1, stdout=b'\n', stderr=('error: no such object: '+cid+'\n').encode())
    monkeypatch.setattr(m.subprocess, 'run', run)
    real_publish =m.publish
    def publish(rt_, path, value, end, **kwargs): return real_publish(rt_, path, value, 0. if fault =='late_seal' else end, **kwargs)
    monkeypatch.setattr(m, 'publish', publish)
    umask =os.umask(0o077) if fault =='umask077' else None
    try: h =m.host(code, revision)
    finally:
        if umask is not None: os.umask(umask)
    assert h['status'] ==('pass' if fault in ('none', 'umask077') else 'fail')
    assert h['owned_container_removed'] is (fault !='foreign')
    argv =calls[0]; mounts =[argv[i+1]for i,x in enumerate(argv)if x =='--mount']
    assert len(mounts) ==2 and mounts[0].endswith(',readonly') and '--gpus' not in argv
    assert 'CUDA_VISIBLE_DEVICES=-1' in argv and '6g' in argv and '4' in argv and '--network' in argv
    assert not any(x[:3] ==['docker', 'rm', '-f']for x in calls)
    assert out.stat().st_mode&0o777 ==0o555 and (out/'report.json').stat().st_mode&0o777 ==0o444


def test_cost_decision_uses_worst_cached_pass_and_max_preparation_not_fastest():
    prior =dict(receipts_opened=False, experiment_retried=False); binding =dict(source='frozen'); v =receipt(binding, 'b'*40, prior)
    for r in v['rows']: r['cached_seconds'] =.001
    v['fixtures'][0]['cache_preparation_seconds'] =1.; v['fixtures'][1]['cache_preparation_seconds'] =2.
    v['projected_recipe_seconds'] =.001*32*1026+32*2.
    v['decision'] ='PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION'; m.validate_native(rt, v, binding, 'b'*40, prior)
    v['rows'][-1]['cached_seconds'] =.03
    v['projected_recipe_seconds'] =.03*32*1026+32*2.; v['decision'] ='CACHE_RECIPE_COST_UNQUALIFIED_NO_REAL_FIT'
    m.validate_native(rt, v, binding, 'b'*40, prior)
    v['projected_recipe_seconds'] =.001*32*1026+32*2.; v['decision'] ='PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION'
    with pytest.raises(ValueError, match='worst-observed'): m.validate_native(rt, v, binding, 'b'*40, prior)


@pytest.mark.parametrize('fault', ['identity', 'seal'])
def test_publish_exception_demotes_original_owned_fd_not_foreign_or_prior_pass(tmp_path, monkeypatch, fault):
    path =tmp_path/'report.json'; value =dict(status='pass'); real_chmod =Path.chmod
    if fault =='identity':
        runtime =SimpleNamespace(require=rt.require, identity=lambda *a: (_ for _ in ()).throw(ValueError('postidentity')))
    else:
        runtime =rt
        monkeypatch.setattr(Path, 'chmod', lambda p,*a,**k: (_ for _ in ()).throw(OSError('seal')) if p ==tmp_path else real_chmod(p,*a,**k))
    with pytest.raises((ValueError, OSError)): m.publish(runtime, path, value, m.time.monotonic()+30, seal_directory=fault =='seal')
    saved =json.loads(path.read_bytes()); assert saved['status'] =='fail' and saved['publish_error_type'] in ('ValueError', 'OSError')
    assert path.stat().st_mode&0o777 ==0o444
    with pytest.raises(FileExistsError): m.publish(rt, path, dict(status='pass'), m.time.monotonic()+30)


def test_real_narrow_native_import_source_and_proof_posthash(tmp_path, monkeypatch):
    root =tmp_path/'root'; revision ='b'*40; code =root/'jobs'/revision/m.ENTRY/'code'; code.mkdir(parents=True)
    for name in m.HELPERS:
        p =code/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((REPO/name).read_bytes()); p.chmod(0o444)
    for name, value in (('revision', revision), ('source-sha256', 'e'*64)):
        p =code.parent/name; p.write_text(value+'\n'); p.chmod(0o444)
    out =root/'results'/('coherent-pair-cache-cost-probe-'+revision); out.mkdir(parents=True); out.chmod(0o755)
    binding =dict(producer_revision=revision, helpers={n:rt.identity(code/n)for n in m.HELPERS}, markers={n:rt.identity(code.parent/n)for n in ('revision', 'source-sha256')})
    (out/'proof.json').write_text(json.dumps(dict(source_binding=binding))); (out/'proof.json').chmod(0o444); (out/'container.cid').write_text('c'*64)
    pin =rt.identity(out/'proof.json'); monkeypatch.setattr(m, 'ROOT', root); monkeypatch.setattr(m.sys, 'platform', 'linux'); monkeypatch.setattr(m.os, 'geteuid', lambda:1000)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1'); monkeypatch.setenv('WR_IMAGE_ID', m.IMAGE)
    original_stat, original_iter =Path.stat, Path.iterdir
    class Owner:
        def __init__(self, x): self.x =x; self.st_uid =1000
        def __getattr__(self, k): return getattr(self.x, k)
    monkeypatch.setattr(Path, 'stat', lambda p,*a,**k:Owner(original_stat(p,*a,**k))if p ==out else original_stat(p,*a,**k))
    monkeypatch.setattr(Path, 'iterdir', lambda p:iter([Path('lo')])if p ==Path('/sys/class/net') else original_iter(p))
    monkeypatch.setattr(m.signal, 'signal', lambda *a:None); monkeypatch.setattr(m.signal, 'setitimer', lambda *a:None)
    def measured(np_, old_, check, value): check(); value.update(phase='complete', full_fit_executed=False)
    monkeypatch.setattr(m, 'measure', measured)
    modules ={k:v for k,v in sys.modules.items()if k =='world_reward' or k.startswith('world_reward.')}; path =sys.path[:]
    for k in modules: del sys.modules[k]
    try:
        result =m.native(code, revision, out, pin, m.time.monotonic()+30)
        assert result['status'] =='pass' and result['source_rehashed_after']
        assert Path(sys.modules['world_reward.coherent_pair_cache'].__file__) ==code/'src/world_reward/coherent_pair_cache.py'
    finally:
        for k in list(sys.modules):
            if k =='world_reward' or k.startswith('world_reward.'): del sys.modules[k]
        sys.modules.update(modules); sys.path[:] =path


def test_wrapper_syntax_no_arguments_cpu_namespace_and_inclusive_outer_budget():
    p =REPO/'infra/run_coherent_pair_cache_cost_probe.sh'; subprocess.run(['bash', '-n', str(p)], check=True)
    text =p.read_text(); assert '740s' in text and '$# == 0' in text and 'DOCKER_HOST=' in text and 'python3 -I -B' in text
    assert '--gpus' not in text

"""NEW data-free paired parity/cost control; not a retry of the closed cost study."""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_coherent_pair_cache_cost_probe'
CONFIG = 'configs/coherent_pair_cache_cost_probe_v1.json'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET = 720
HELPERS = ('infra/coherent_pair_cache_cost_probe.py', 'infra/run_coherent_pair_cache_cost_probe.sh', CONFIG,
    'infra/coherent_pair_cost_probe.py', 'src/world_reward/coherent_pair_cache.py',
    'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/__init__.py',
    'src/world_reward/coherent_pair_learning.py', 'src/world_reward/coherent_route_scorer.py',
    'src/world_reward/interaction_candidate_evidence.py', 'src/world_reward/interaction_tuple_evidence.py',
    'src/world_reward/person_pose_observations.py', 'src/world_reward/hoi_detr_observations.py')


def module(code, name, filename):
    spec = importlib.util.spec_from_file_location(name, code/filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def helpers(code):
    return module(code, 'paired_cost_runtime', HELPERS[5]), module(code, 'original_cost_fixture', HELPERS[3])


def configuration(rt, code, source):
    value = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16384)
    fixed = dict(schema='world_reward.coherent_pair_cache_cost_probe.v1', budget_seconds=720,
        cpu_count=4, memory_bytes=6*1024**3, object_slots=3600, native_tokens=1500,
        fixtures=[[2, 4], [4, 64]], coefficients=['zero_geometry', 'dyadic_geometry', 'dyadic_relational'],
        alpha=.25, fit_images=32, fit_evaluations=1026, object_block_size=128, image_id=IMAGE,
        output_prefix='results/coherent-pair-cache-cost-probe-',
        prior_cost_source_reference=dict(producer_revision='1bf7109d0d24d3618ee74f458d72e5107011ecf5',
            host=dict(bytes=2432, sha256='c0a6f2b291e93fe2edaa5979db4b2da4df6cfe15d3fe48c2e5de1dde1c43c02d'),
            native=dict(bytes=6628, sha256='eb2ed10363bd933b018db3db127144f72668bfdc7e3602cfdc88c3ff4f1c9790'),
            decision='COST_RECIPE_UNQUALIFIED_NO_REAL_FIT', receipts_opened=False, experiment_retried=False))
    rt.require(set(value) == set(fixed)|{'helper_pins'} and all(type(value.get(k)) is type(v) and value[k] == v for k, v in fixed.items()), 'Exact NEW paired control scope required')
    rt.require(set(value['helper_pins']) == set(HELPERS[3:]) and all(source['helpers'].get(k) == pin for k, pin in value['helper_pins'].items()), 'Frozen old fixtures/numerical/new cache source differs')
    return value


def proof(rt, code, revision):
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); configuration(rt, code, source)
    rt.require({p.name for p in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Only original current source namespace allowed')
    r = subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True, timeout=10, check=False)
    rt.require(r.returncode == 0 and len(r.stdout) <=16384 and len(r.stderr) <=8192, 'Actual immutable CPU image unavailable')
    rows = rt.strict(r.stdout); rt.require(len(rows) ==1, 'One CPU image required'); row = rows[0]
    rt.require(row['Id'] ==IMAGE and row['Os'] =='linux' and row['Architecture'] =='amd64', 'Exact B47 CPU image required')
    return dict(source_binding=source, image={k: row[k] for k in ('Id', 'Os', 'Architecture', 'RootFS')})


def leaves(code):
    return [*(code/name for name in HELPERS), code.parent/'revision', code.parent/'source-sha256']


def publish(rt, path, value, deadline, *, seal_directory=False):
    """Original owned FD permits FAIL demotion even after chmod/validation errors."""
    with path.open('xb') as stream:
        def write():
            raw =(json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
            stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
            return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        try:
            os.fchmod(stream.fileno(), 0o444); pin =write()
            if seal_directory: path.parent.chmod(0o555)
            if time.monotonic() >=deadline: value.update(status='fail', error_type='TimeoutError'); pin =write()
            rt.require(rt.identity(path, 1<<20) ==pin, 'Exact sealed paired receipt required')
            if time.monotonic() >=deadline and value['status'] =='pass': value.update(status='fail', error_type='TimeoutError'); write()
        except BaseException as e:
            value.update(status='fail', publish_error_type=type(e).__name__)
            try: write()
            except BaseException: pass  # A failed filesystem write never yields process success.
            raise


def identical(np, left, right):
    """ALL native/group arrays, metadata, dtypes and NaN/signedzero bytes exact."""
    if type(left) is not type(right): raise ValueError('Original/cache result type differs')
    if type(left) is np.ndarray:
        if left.dtype !=right.dtype or left.shape !=right.shape or left.tobytes() !=right.tobytes(): raise ValueError('Original/cache array bytes differ')
    elif is_dataclass(left):
        for f in fields(left): identical(np, getattr(left, f.name), getattr(right, f.name))
    elif isinstance(left, Mapping):
        if left.keys() !=right.keys(): raise ValueError('Original/cache metadata keys differ')
        for key in left: identical(np, left[key], right[key])
    elif type(left) in (tuple, list):
        if len(left) !=len(right): raise ValueError('Original/cache metadata count differs')
        for x, y in zip(left, right): identical(np, x, y)
    elif left !=right: raise ValueError('Original/cache metadata differs')


def measure(np, old_fixture, check, progress, *, object_count=3600):
    """Small count override is test-only; native always uses fixed original3600."""
    from world_reward import coherent_pair_learning as old, coherent_pair_cache as cached
    from world_reward.coherent_route_scorer import _fingerprint
    sources = []; iterator = old_fixture.fixtures(np, object_count=object_count, source_records=sources)
    for case in range(2):
        progress.update(phase='fixture_prepare', case=case); check(); bank = next(iterator)
        digest = _fingerprint(bank); n, o, k = old._bank(bank)
        scale = old.PairScale(np.ones(12), np.ones(12, bool)); positive = np.zeros((len(bank.person_boxes), len(bank.object_boxes)), bool); positive[0, 0] =True
        check(); start = time.monotonic(); cache = cached.prepare_pair_cache(bank, scale); prep = time.monotonic()-start; check()
        factor_digest = _fingerprint((cache.factors, cache.scales, cache.person_members, cache.object_members))
        progress['fixtures'].append(dict(case=case, persons=n, objects=o, native_pairs=k, native_tokens=sources[-1]['native_tokens'],
            candidate_rows=n*2*o, tuple_rows=n*2*k, bridge_rows=k*o, person_groups=len(bank.person_boxes), object_groups=len(bank.object_boxes),
            cache_preparation_seconds=prep, bank_sha256=digest, factor_sha256=factor_digest,
            full_original_observations_sha256=sources[-1]['full_original_observations_sha256']))
        theta = np.array([(-1 if j%2 ==0 else 1)*(j+1)/32. for j in range(17)])
        for name, t, alpha in (('zero_geometry', np.zeros(17), None), ('dyadic_geometry', theta, None), ('dyadic_relational', theta, .25)):
            progress.update(phase=name); check(); start = time.monotonic()
            original = old.loss_gradient(t, (bank,), (positive,), scale, alpha=alpha); original_seconds = time.monotonic()-start; check()
            progress['phase'] =name+'_cached'
            start = time.monotonic(); result = cached.loss_gradient(t, (cache,), (positive,), alpha=alpha, object_block_size=128); cached_seconds = time.monotonic()-start; check()
            if np.float64(original[0]).tobytes() !=np.float64(result[0]).tobytes(): raise ValueError('Original/cache loss bits differ')
            identical(np, original[1:], result[1:])
            progress['phase'] ='full_score_parity'; a = old.score_pair_groups(bank, t, scale, alpha=alpha); check()
            b = cached.score_pair_groups(cache, t, alpha=alpha, object_block_size=128); check(); identical(np, a, b)
            progress['rows'].append(dict(case=case, coefficient=name, original_seconds=original_seconds, cached_seconds=cached_seconds,
                loss=result[0], gradient=list(map(float, result[1])), counts=result[2],
                loss_gradient_bits_exact=True, full_scores_arrays_metadata_bits_exact=True, full_scores_sha256=_fingerprint(a),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform =='darwin' else 1024)))
        if digest !=_fingerprint(bank) or factor_digest !=_fingerprint((cache.factors, cache.scales, cache.person_members, cache.object_members)): raise ValueError('Original/cache immutable factors changed')
    if next(iterator, None) is not None or not all(x['source_rehashed_after'] for x in sources): raise ValueError('All original procedural observations required')
    projection = max(x['cached_seconds'] for x in progress['rows'])*32*1026+32*max(x['cache_preparation_seconds'] for x in progress['fixtures'])
    progress.update(phase='complete', rows_completed=6, source_fixtures_rehashed_after=True, cache_factors_rehashed_after=True,
        projected_recipe_seconds=projection, decision='PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION' if projection <=720 else 'CACHE_RECIPE_COST_UNQUALIFIED_NO_REAL_FIT', full_fit_executed=False)


def native(code, revision, out, proof_pin, deadline):
    rt, fixture = helpers(code)
    rt.require(rt.canonical(out) ==ROOT/'results'/('coherent-pair-cache-cost-probe-'+revision)
        and out.stat().st_uid ==1000 and out.stat().st_mode&0o777 ==0o755, 'Exact fresh owned output required')
    rt.identity(out/'container.cid', 65, readonly=False); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', (out/'container.cid').read_bytes()), 'Exact original CID required')
    p = rt.pinned(out/'proof.json', proof_pin, 1<<20); cfg = configuration(rt, code, p['source_binding'])
    before = {str(path): rt.identity(path, 2<<20) for path in leaves(code)}
    rt.require(sys.platform =='linux' and os.geteuid() ==1000 and os.environ.get('CUDA_VISIBLE_DEVICES') =='-1'
        and os.environ.get('WR_IMAGE_ID') ==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()} =={'lo'}
        and p['source_binding']['producer_revision'] ==revision and {p.name for p in out.iterdir()} =={'proof.json', 'container.cid'}, 'Restricted CPU-only native scope required')
    rt.require(all(before[str(code/k)] ==v for k,v in p['source_binding']['helpers'].items())
        and all(before[str(code.parent/k)] ==v for k,v in p['source_binding']['markers'].items())
        and (code.parent/'revision').read_bytes() ==(revision+'\n').encode(), 'Original mounted source/helper/markers differ')
    def check():
        if time.monotonic() >=deadline: raise TimeoutError('Inclusive paired720s budget exhausted')
    def cancelled(*_): raise TimeoutError('Paired control cancellation/deadline')
    signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGALRM, cancelled); signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    report = dict(schema=cfg['schema'], stage='coherent_pair_cache_cost_native', status='fail', phase='imports', producer_revision=revision,
        source_binding=p['source_binding'], image_id=IMAGE, fixtures=[], rows=[], source_rehashed_after=False,
        prior_cost_source_reference=cfg['prior_cost_source_reference'], gpu_used=False, models_loaded=False, rgb_read=False,
        references_read=False, challenge_inputs_used=False, quality_verified=False, adoption=False)
    try:
        sys.path[:0] =[str(code/'src')]; import numpy as np
        from world_reward import coherent_pair_cache, coherent_pair_learning
        for m, name in ((coherent_pair_cache, HELPERS[4]), (coherent_pair_learning, HELPERS[7])): rt.require(Path(m.__file__).resolve() ==code/name, 'Actual numerical module origin differs')
        measure(np, fixture, check, report); check(); report['status'] ='pass'
    except BaseException as e: report.update(status='fail', error_type=type(e).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            rt.require(before =={str(path): rt.identity(path, 2<<20) for path in leaves(code)} and rt.identity(out/'proof.json', 1<<20) ==proof_pin, 'Native source/proof posthash differs')
            report['source_rehashed_after'] =True
        except Exception as e: report.update(status='fail', post_error_type=type(e).__name__)
        publish(rt, out/'native.json', report, deadline)
    return report


def validate_native(rt, value, binding, revision, prior):
    fixed = dict(schema='world_reward.coherent_pair_cache_cost_probe.v1', stage='coherent_pair_cache_cost_native', status='pass', phase='complete',
        source_binding=binding, producer_revision=revision, image_id=IMAGE, rows_completed=6, source_rehashed_after=True,
        source_fixtures_rehashed_after=True, cache_factors_rehashed_after=True, full_fit_executed=False,
        prior_cost_source_reference=prior, gpu_used=False, models_loaded=False, rgb_read=False, references_read=False,
        challenge_inputs_used=False, quality_verified=False, adoption=False)
    rt.require(all(type(value.get(k)) is type(v) and value[k] ==v for k,v in fixed.items()), 'Exact complete NEW paired control required')
    fixtures, rows = value['fixtures'], value['rows']
    rt.require([(r['case'], r['persons'], r['objects'], r['native_pairs']) for r in fixtures] ==[(0, 2, 3600, 4), (1, 4, 3600, 64)], 'All original full-slot fixtures required')
    for r in fixtures:
        rt.require(r['native_tokens'] ==1500 and r['candidate_rows'] ==r['persons']*2*3600 and r['tuple_rows'] ==r['persons']*2*r['native_pairs']
            and r['bridge_rows'] ==r['native_pairs']*3600 and r['person_groups'] ==r['persons'] and r['object_groups'] ==3600
            and all(re.fullmatch('[0-9a-f]{64}', r[k]) for k in ('bank_sha256', 'factor_sha256', 'full_original_observations_sha256'))
            and math.isfinite(r['cache_preparation_seconds']) and r['cache_preparation_seconds'] >0, 'No source slot or cache preparation omitted')
    rt.require(len({r['bank_sha256'] for r in fixtures}) ==2 and len({r['full_original_observations_sha256'] for r in fixtures}) ==2,
               'Both distinct original fixture observations required')
    rt.require([(r['case'], r['coefficient']) for r in rows] ==[(i, name) for i in range(2) for name in ('zero_geometry', 'dyadic_geometry', 'dyadic_relational')], 'Six original-then-cached rows required')
    for row, length in zip(rows, (17, 17, 1, 17, 17, 1)):
        rt.require(row['loss_gradient_bits_exact'] is True and row['full_scores_arrays_metadata_bits_exact'] is True
            and len(row['gradient']) ==length and all(math.isfinite(x) for x in [row['loss'], *row['gradient']])
            and all(math.isfinite(row[k]) and row[k] >0 for k in ('original_seconds', 'cached_seconds'))
            and type(row['max_rss_bytes']) is int and 0<row['max_rss_bytes']<=6*1024**3
            and re.fullmatch('[0-9a-f]{64}', row['full_scores_sha256'])
            and row['counts'] ==dict(fit_records=1, used=1, missing_positive=0, no_alternative=0), 'Every finite full parity measurement required')
    projected = max(r['cached_seconds'] for r in rows)*32*1026+32*max(r['cache_preparation_seconds'] for r in fixtures)
    rt.require(value['projected_recipe_seconds'] ==projected and value['decision'] ==('PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION' if projected <=720 else 'CACHE_RECIPE_COST_UNQUALIFIED_NO_REAL_FIT'), 'Exact worst-observed descriptive projection required')


def host(code, revision):
    started = time.monotonic(); deadline = started+BUDGET
    def cancelled(*_): raise TimeoutError('Inclusive host deadline/cancellation')
    signal.signal(signal.SIGALRM, cancelled); signal.signal(signal.SIGTERM, cancelled); signal.setitimer(signal.ITIMER_REAL, BUDGET)
    rt, fixture = helpers(code); before = proof(rt, code, revision); cfg = configuration(rt, code, before['source_binding'])
    out = rt.canonical(ROOT/'results'/('coherent-pair-cache-cost-probe-'+revision)); rt.require(not out.exists(), 'No reuse/overwrite of paired namespace')
    out.mkdir(mode=0o755); os.chown(out, 1000, 1000); out.chmod(0o755); owner = (out.stat().st_dev, out.stat().st_ino)
    rt.write(out/'proof.json', (json.dumps(before, sort_keys=True)+'\n').encode(), 0o444); proof_pin = rt.identity(out/'proof.json', 1<<20)
    cidfile = out/'container.cid'; name ='world-reward-pair-cache-cost-'+revision[:12]; removed =False
    report = dict(schema=cfg['schema'], stage='coherent_pair_cache_cost_host', status='fail', producer_revision=revision, source_binding=before['source_binding'])
    def cleanup():
        if not cidfile.exists(): return False
        rt.identity(cidfile, 65, readonly=False); cid =cidfile.read_text().strip(); rt.require(re.fullmatch('[0-9a-f]{64}', cid), 'Owned CID required')
        def inspect(): return subprocess.run(['docker', 'inspect', cid, '--format', '{{.Id}}|{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.cache_cost.owner"}}'], capture_output=True, timeout=5)
        r =inspect()
        if not fixture.absent(r, cid):
            rt.require(r.returncode ==0 and r.stdout.decode().strip() ==cid+'|'+IMAGE+'|/'+name+'|'+revision, 'Never delete a foreign container')
            subprocess.run(['docker', 'rm', '-f', cid], capture_output=True, timeout=10, check=True); rt.require(fixture.absent(inspect(), cid), 'Owned container survives')
        cidfile.chmod(0o444); return True
    try:
        argv =['docker', 'run', '--rm', '--cidfile', str(cidfile), '--name', name, '--label', 'world_reward.cache_cost.owner='+revision,
            '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--memory', '6g', '--cpus', '4',
            '--pids-limit', '256', '--user', '1000:1000', '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=128m', '--entrypoint', '/usr/bin/env']
        for path in leaves(code): argv +=['--mount', f'type=bind,src={path},dst={path},readonly']
        argv +=['--mount', f'type=bind,src={out},dst={out}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=-1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'PYTHONDONTWRITEBYTECODE=1',
            'WR_IMAGE_ID='+IMAGE, 'python', '-I', '-B', str(code/HELPERS[0]), 'native', str(code), revision, str(out), str(proof_pin['bytes']), proof_pin['sha256'], str(deadline)]
        r =subprocess.run(argv, capture_output=True, timeout=max(.001, deadline-time.monotonic()), check=False)
        report['native_exit_code'] =r.returncode; rt.require(len(r.stdout) <=16384 and len(r.stderr) <=16384, 'Bounded native diagnostics required')
        removed =cleanup(); value =rt.strict((out/'native.json').read_bytes())
        rt.require(r.returncode ==0, 'Native paired control failed'); validate_native(rt, value, before['source_binding'], revision, cfg['prior_cost_source_reference'])
        report.update(status='pass', native_report_identity=rt.identity(out/'native.json', 1<<20), decision=value['decision'])
    except BaseException as e: report.update(status='fail', error_type=type(e).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try: removed =cleanup() or removed
        except Exception as e: report.update(status='fail', cleanup_error_type=type(e).__name__)
        try: report['source_rehashed_after'] =proof(rt, code, revision) ==before
        except Exception as e: report.update(status='fail', post_error_type=type(e).__name__)
        rt.require((out.stat().st_dev, out.stat().st_ino) ==owner and out.stat().st_uid ==1000, 'Owned paired output replaced')
        rt.require({p.name for p in out.iterdir()} <={'proof.json', 'container.cid', 'native.json'}, 'Unknown paired output node')
        report.update(owned_container_removed=removed, elapsed_seconds=time.monotonic()-started)
        if not removed or not report.get('source_rehashed_after') or time.monotonic() >=deadline: report['status'] ='fail'
        publish(rt, out/'report.json', report, deadline, seal_directory=True)
    return report


def main():
    parser =argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('mode', choices=('host', 'native')); parser.add_argument('code'); parser.add_argument('revision'); parser.add_argument('native_args', nargs='*'); a =parser.parse_args()
    code =Path(a.code)
    if not re.fullmatch('[0-9a-f]{40}', a.revision) or code !=ROOT/'jobs'/a.revision/ENTRY/'code' or Path(__file__).resolve() !=code/HELPERS[0]: raise ValueError('Exact new published source required')
    if a.mode =='host':
        if a.native_args or sys.platform !='linux' or os.geteuid() !=0 or os.uname().nodename !='scenesmith-ncc-h100-01' or os.environ.get('DOCKER_HOST') !='unix://'+str(ROOT)+'/docker.sock': raise ValueError('Actual VM01 root CPU host required')
        result =host(code, a.revision)
    else:
        if len(a.native_args) !=4: raise ValueError('Exact proof/deadline arguments required')
        out, size, digest, end =a.native_args; deadline =float(end)
        if not math.isfinite(deadline) or deadline-time.monotonic() >720: raise ValueError('Shared finite deadline required')
        result =native(code, a.revision, Path(out), dict(bytes=int(size), sha256=digest), deadline)
    print(json.dumps(dict(status=result['status'], decision=result.get('decision'), error_type=result.get('error_type')), sort_keys=True))
    if result['status'] !='pass': raise SystemExit(1)


if __name__ =='__main__': main()

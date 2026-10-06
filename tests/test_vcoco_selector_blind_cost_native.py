"""Tiny manufactured/mocked coordination only; never Torch, photos or FIT."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import vcoco_selector_blind_cost_native as worker
import vcoco_full_interaction_join as join
import coherent_pair_gpu_probe as generic
import coherent_pair_gpu_objective_probe as qualified
from world_reward import coherent_pair_learning as learner, coherent_route_scorer as core
from world_reward import coherent_pair_packed_torch as scorer
from world_reward.coherent_pair_cache import prepare_pair_cache
from world_reward.coherent_pair_packed import prepare_marginal_packed
from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
from world_reward.coherent_pair_packed_torch import TorchMarginalPairScores
from test_vcoco_full_interaction_join import metadata_proof
from test_vcoco_interaction_observations import fixture
from test_vcoco_full_interaction_core import at_slot

spec = importlib.util.spec_from_file_location('blind_cost_partition_controls',
    Path(__file__).with_name('test_coherent_pair_partition.py'))
controls = importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)


def selection(proof):
    ids = [r['endpoint']['image_id'] for r in proof['records'][:32]]
    order = sorted(ids, key=lambda x: (hashlib.sha256(('world_reward.vcoco_selector_cv_v1/'+x).encode()).hexdigest(), x))
    fold = {x: i % 4 for i, x in enumerate(order)}
    return tuple(dict(slot=i, image_id=x, fold=fold[x]) for i, x in enumerate(ids))


def meta_inputs():
    proof = metadata_proof(); rows = []
    for r in proof['records']:
        expected = join.expected_evidence_metadata(r)
        row = dict(image_id=r['endpoint']['image_id'], original_slot=r['endpoint']['original_slot'],
            acquired_ordinal=r['endpoint']['acquired_ordinal'], persons=0, objects=3600, native_pairs=0,
            person_side_object_rows=0, person_side_pair_rows=0, pair_object_rows=0, supported_counts=[0, 0, 0],
            nan_counts=[0, 0, 0], evidence_fingerprint='a'*64, raw_bank_fingerprint='b'*64,
            source_observation_references=[], arrays={n: dict(shape=s, dtype=d, sha256='c'*64) for n, (s, d) in expected.items()})
        rows.append(row)
    return proof, rows, selection(proof)


def fake_device(monkeypatch, fault=None):
    controls.fake_backend(monkeypatch); tensor = controls.controls.Tensor; calls = []
    monkeypatch.setattr(tensor, 'data_ptr', lambda self: self.array.__array_interface__['data'][0], raising=False)
    def prepare(packed, device): return SimpleNamespace(packed=packed, device=device)
    def score(device, theta, temperature, alpha):
        calls.append((theta.copy(), alpha)); cpu = score_pair_marginal_packed(device.packed, theta, temperature=temperature, alpha=alpha)
        values = {n: tensor(getattr(cpu, n).copy()) for n in (*worker.FLOAT_FIELDS, *worker.BOOL_FIELDS)}
        if alpha == 0.:
            for x, y in (('native_scores_a', 'native_scores_b'), ('scores_a', 'scores_b'), ('geometry_derivatives_a', 'geometry_derivatives_b')):
                values[y] = values[x]
        result = TorchMarginalPairScores(**values, identity=cpu.identity, distribution_fingerprint=cpu.distribution_fingerprint,
            parameter_fingerprint=cpu.parameter_fingerprint, temperature=temperature, alpha=alpha, device='cuda:0')
        if fault: fault(result, len(calls))
        return result
    monkeypatch.setattr(scorer, 'prepare_marginal_packed_torch', prepare)
    monkeypatch.setattr(scorer, 'score_pair_marginal_packed_torch', score)
    g = SimpleNamespace(arrays=generic.arrays, bits=generic.bits, device_fingerprint=lambda t, d:
                        core._fingerprint((d.packed.arrays, d.packed.identity)))
    t = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))
    monkeypatch.setattr(worker, '_memory', lambda t, report: report.update(max_rss_bytes=1024,
        peak_torch_allocated=1024, peak_torch_reserved=2048, max_observed_cgroup_bytes=4096))
    return g, t, calls


def tiny_packed(n=2, o=2, k=2):
    variant = 'missing' if n and k else ('no_hoi' if n else ('empty_persons' if k else 'empty'))
    args = next(bank for bank in generic.fixtures(np) if learner._bank(bank)[0] == n
                and (learner._bank(bank)[2] > 0) == bool(k) and len(bank.object_boxes) == (3 if variant == 'missing' else (2 if variant != 'empty' else 0)))
    scales = learner.PairScale(np.ones(12), np.ones(12, bool))
    return prepare_marginal_packed(prepare_pair_cache(args, scales))


def test_profile_has_no_pruning_or_manufactured_positive_argument():
    assert worker.BUDGET == 3600 and worker.MEMORY == 64 << 30
    proof, rows, chosen = meta_inputs(); worker._inputs(join, proof, rows, chosen)
    assert len(proof['files']) == 144 and len(chosen) == 32
    assert sorted(x['fold'] for x in chosen) == [0]*8+[1]*8+[2]*8+[3]*8


@pytest.mark.parametrize('fault', ['count', 'duplicate_slot', 'duplicate_id', 'fold', 'foreign_id', 'extra', 'boolslot', 'reorder', 'missing_summary', 'truncatedK', 'missing46'])
def test_input_complete_population_fixed_fold_and_all46_metadata(fault):
    proof, rows, chosen = meta_inputs(); chosen = list(chosen)
    if fault == 'count': chosen.pop()
    elif fault == 'duplicate_slot': chosen[-1]['slot'] = chosen[0]['slot']
    elif fault == 'duplicate_id': chosen[-1]['image_id'] = chosen[0]['image_id']
    elif fault == 'fold': chosen[0]['fold'] = (chosen[0]['fold']+1) % 4
    elif fault == 'foreign_id': chosen[0]['image_id'] = 'f'*32
    elif fault == 'extra': chosen[0]['role'] = 'private'
    elif fault == 'boolslot': chosen[0]['slot'] = True
    elif fault == 'reorder': chosen[0], chosen[1] = chosen[1], chosen[0]
    elif fault == 'missing_summary': rows.pop()
    elif fault == 'truncatedK': rows[0]['native_pairs'] = 1
    else: rows[0]['arrays'].pop('object_ids')
    with pytest.raises(ValueError): worker._inputs(join, proof, rows, tuple(chosen))


@pytest.mark.parametrize('n,k', [(2, 2), (2, 0), (0, 2), (0, 0)])
def test_real_tiny_complete_score_partition_parity_empty_nohoi(monkeypatch, n, k):
    g, t, calls = fake_device(monkeypatch); packed = tiny_packed(n=n, k=k)
    device = scorer.prepare_marginal_packed_torch(packed, device='cuda:0'); before = core._fingerprint((packed.arrays, packed.identity))
    report = {}; rows = worker._controls(np, t, g, packed, device, lambda: None, report)
    assert len(rows) == 4 and len(calls) == 8 and [r['alpha'] for r in rows] == [0., 1., 0., 1.]
    assert all(r['native_shape'] == list(packed.arrays['native_supported'].shape) for r in rows)
    assert all(len(r['partitions']) == (2 if r['alpha'] == 0. else 1) for r in rows)
    assert all(p['repeat_bits_exact'] and p['cpu_full_parity'] for r in rows for p in r['partitions'])
    assert before == core._fingerprint((packed.arrays, packed.identity))
    assert not calls[0][0].any() and np.linalg.norm(calls[4][0]) == pytest.approx(1.)
    if not n: assert all(p['status'] == 'no_supported' and p['log_partition'] is None for r in rows for p in r['partitions'])


@pytest.mark.parametrize('fault', ['repeat', 'vjp', 'identity', 'distribution', 'parameter', 'shared_first', 'shared_repeat', 'dtype', 'device'])
def test_full_native_group_vjp_metadata_repeat_or_shared_failure(monkeypatch, fault):
    def corrupt(s, count):
        if fault == 'repeat' and count == 2: s.scores_a.array.flat[0] += 1.
        elif fault == 'vjp': s.geometry_derivatives_a.array.flat[0] += 1.
        elif fault == 'identity': object.__setattr__(s, 'identity', {})
        elif fault == 'distribution': object.__setattr__(s, 'distribution_fingerprint', 'f'*64)
        elif fault == 'parameter': object.__setattr__(s, 'parameter_fingerprint', 'f'*64)
        elif fault in ('shared_first', 'shared_repeat') and count == (1 if fault == 'shared_first' else 2):
            object.__setattr__(s, 'scores_b', controls.controls.Tensor(s.scores_b.array.copy()))
        elif fault == 'dtype': s.alpha_derivatives_b.array = s.alpha_derivatives_b.array.astype(np.float32)
        elif fault == 'device': object.__setattr__(s, 'device', 'cpu')
    g, t, _ = fake_device(monkeypatch, corrupt); packed = tiny_packed(); device = scorer.prepare_marginal_packed_torch(packed, device='cuda:0')
    with pytest.raises((ValueError, AssertionError)):
        worker._controls(np, t, g, packed, device, lambda: None, {})


def test_all47_reconstruct_matches_saved46_summary_and_metadata(monkeypatch):
    args, _ = fixture(0, False); args = at_slot(args, 47)
    value = join.numerical.reconstruct_interaction(*args, population=48); summary = join.summary(value)
    summary['arrays'] = {n: join.numerical._identity(a) for n, a in join.evidence_arrays(value).items()}
    record = dict(endpoint=args[0], pose=args[2], hoi=args[4], paths=['/unopened/e', '/unopened/p', '/unopened/h'])
    arrays = iter((args[1], args[3], args[5])); calls = []
    monkeypatch.setattr(join, 'load_bank', lambda path, row, fields: calls.append(len(fields)) or next(arrays))
    actual = worker._reconstruct(np, join, record, summary, lambda: None)
    assert calls == [17, 11, 19] and actual.original_slot == 47
    summary['arrays']['object_ids']['sha256'] = 'f'*64
    arrays = iter((args[1], args[3], args[5]))
    with pytest.raises(ValueError): worker._reconstruct(np, join, record, summary, lambda: None)


def measure_setup(monkeypatch, *, fail=None):
    proof, rows, chosen = meta_inputs(); g, t, calls = fake_device(monkeypatch)
    args, _ = fixture(0, False); value = join.numerical.reconstruct_interaction(*at_slot(args, 0), population=48)
    bank = learner.pair_route_bank(value.person, value.evidence); visited = []
    def rebuild(np, join, record, summary, check):
        visited.append(record['endpoint']['original_slot']); check(); return value
    monkeypatch.setattr(worker, '_reconstruct', rebuild)
    monkeypatch.setattr(join, 'load_bank', lambda path, row, fields: {})
    monkeypatch.setattr(join.rt, 'canonical', lambda x: Path(x)); monkeypatch.setattr(join.rt, 'identity', lambda path, limit: proof['files'][str(path)])
    monkeypatch.setattr(join, 'snapshot', lambda path: ('fixed', str(path)))
    g.inventory = dict(version='1.16.3', identity='a'*64)
    monkeypatch.setattr(qualified, 'scipy_evidence', lambda rt: g.inventory.copy())
    def runtime(np, t, g, rt, check, report):
        report['runtime'] = {}; check(); return qualified.scipy_evidence(rt)
    monkeypatch.setattr(worker, '_runtime', runtime)
    if fail:
        monkeypatch.setattr(worker, '_controls', fail)
    return proof, rows, chosen, g, t, calls, visited, bank


def test_full48_prevalidation_resident32_then_streamed32_coordination(monkeypatch):
    proof, rows, chosen, g, t, calls, visited, bank = measure_setup(monkeypatch)
    checked = []; report = {}; before = core._fingerprint((proof, rows, chosen, bank))
    worker.measure(np, t, g, join.rt, proof, rows, chosen, lambda: checked.append(1), report)
    assert report['status'] == 'pass' and report['phase'] == 'complete' and report['images_completed'] == 32
    assert visited == list(range(48))+list(range(32)) and len(calls) == 32*8
    assert report['all144_arrays_prevalidated'] and report['all48_summaries_verified'] and report['resident_banks_rehashed_after']
    assert report['snapshots_rehashed_after'] and report['raw_inputs_rehashed_after'] and report['scipy_inventory_rehashed_after']
    assert all(r['objects'] == 3600 and r['persons'] == r['native_pairs'] == 0 and len(r['controls']) == 4 for r in report['images'])
    assert not any(report[n] for n in ('references_read', 'positive_masks_constructed', 'FIT_performed', 'optimizer_executed', 'RGB_decoded', 'quality_verified'))
    assert before == core._fingerprint((proof, rows, chosen, bank)) and len(checked) > 144


@pytest.mark.parametrize('stage', ['controls', 'raw_drift', 'metadata_drift', 'inventory_drift', 'timeout'])
def test_failure_stays_failure_partial_counters_and_postcheck(monkeypatch, stage):
    proof, rows, chosen, g, t, calls, visited, bank = measure_setup(monkeypatch)
    report = {}; count = []
    def fail(*args):
        if stage == 'raw_drift': monkeypatch.setattr(join.rt, 'identity', lambda path, limit: dict(bytes=2, sha256='f'*64))
        elif stage == 'metadata_drift': rows[0]['persons'] = 99
        elif stage == 'inventory_drift': g.inventory = {}
        raise TimeoutError('Manufactured deadline') if stage == 'timeout' else RuntimeError('SECRET-NOT-IN-REPORT')
    monkeypatch.setattr(worker, '_controls', fail)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        worker.measure(np, t, g, join.rt, proof, rows, chosen, lambda: count.append(1), report)
    assert report['status'] == 'fail' and report['images_completed'] == 0 and report['failure_stage'] == 'image_device_upload'
    assert len(report['images']) == 1 and 'SECRET' not in worker._json(report).decode()
    if stage in ('raw_drift', 'metadata_drift', 'inventory_drift'): assert report['post_error_type'] == 'ValueError'
    else: assert report['raw_inputs_rehashed_after'] and report['snapshots_rehashed_after']


@pytest.mark.parametrize('fault', ['rss', 'cgroup_current', 'cgroup_unbounded', 'device_reserved', 'cgroup_peak'])
def test_actual_memory_gate_no_reduced_graph_rescue(monkeypatch, fault):
    t = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 1024,
        max_memory_reserved=lambda: worker.MEMORY+1 if fault == 'device_reserved' else 2048))
    monkeypatch.setattr(worker.resource, 'getrusage', lambda _: SimpleNamespace(ru_maxrss=(worker.MEMORY//1024+1 if fault == 'rss' else 1)))
    monkeypatch.setattr(Path, 'read_text', lambda self: ('max' if fault == 'cgroup_unbounded' else str(worker.MEMORY))
        if self.name == 'memory.max' else str(worker.MEMORY+1 if fault == 'cgroup_current' or self.name == 'memory.peak' and fault == 'cgroup_peak' else 4096))
    monkeypatch.setattr(Path, 'exists', lambda self: True)
    with pytest.raises(ValueError): worker._memory(t, {'runtime': {}})


def test_import_does_not_load_numpy_torch_scipy_or_private_context():
    code = '''import importlib.abc,importlib.util,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','torch','scipy','vcoco_full_interaction_inputs','vcoco_role_reference'):
   raise AssertionError('Forbidden eager/private import')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('worker',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.BUDGET==3600 and m.MEMORY==64<<30
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(Path(worker.__file__).resolve())], capture_output=True)
    assert result.returncode == 0, result.stderr


def receipt():
    _, _, selected = meta_inputs(); digest = 'a'*64
    partitions = lambda alpha: [dict(arm=arm, supported_groups=4, status='supported', log_partition=1., gradient_norm=1.,
        gradient_sha256=digest, seconds=.1, repeat_seconds=.1, cpu_seconds=.1, repeat_bits_exact=True, cpu_full_parity=True)
        for arm in (('A', 'B') if alpha == 0. else ('B',))]
    images = []
    for item in selected:
        images.append(dict(item, persons=2, objects=3600, native_pairs=100, candidate_rows=14400, tuple_rows=400, bridge_rows=360000,
            person_groups=2, object_groups=3600, route_references=1440000, complete_geometries=100, snapshots_rehashed_after=True,
            bank_sha256=digest, packed_sha256=digest, device_sha256=digest, reconstruct_seconds=.1, cache_packed_seconds=.1, upload_seconds=.1,
            controls=[dict(theta_index=i, alpha=alpha, native_shape=[2, 2, 3600], group_shape=[2, 3600],
                full_result_sha256=digest, identity_sha256=digest, distribution_fingerprint=digest, parameter_fingerprint=digest,
                score_seconds=.1, repeat_score_seconds=.1, copy_seconds=.1, repeat_copy_seconds=.1, cpu_score_seconds=.1,
                partitions=partitions(alpha)) for i, alpha in ((0, 0.), (0, 1.), (1, 0.), (1, 1.))]))
    value = dict(status='pass', phase='complete', images_completed=32, images=images, elapsed_seconds=10., models_loaded=0,
        runtime=dict(python='3.11.10', numpy='1.26.3', torch='2.5.1+cu124', cuda='12.4', capability=[9, 0], gpu_name='NVIDIA H100 NVL',
            gpu_total_bytes=99456909312, deterministic_algorithms=True, tf32=False, cublas_workspace_config=':4096:8'),
        max_rss_bytes=1024, max_observed_cgroup_bytes=1024, peak_torch_allocated=1024, peak_torch_reserved=2048,
        cgroup_memory_limit=worker.MEMORY, cgroup_peak_available=True, cgroup_peak_bytes=4096,
        scales=[1.]*12, variable_values=[True]*12, scale_fingerprint=digest,
        resident_prepare_seconds=1., scale_seconds=1., scipy_inventory=dict(version='1.16.3', first_native_record_census=True,
            populated_record_claims_verified=True, caches_image_anchored_not_record_certified=True,
            empty_file_policy='explicit_original_record_sha256_empty_and_exact_size0_only', entries=2381,
            source_fingerprint=digest, record=dict(bytes=1, sha256=digest), license_files={'LICENSE': {}}),
        segment_controls=[dict(width=w, empty=e, operation=op, bytes_sha256=digest) for w in (1, 9, 6, 2, 17)
            for e in (False, True) for op in ('sum', 'max')])
    value.update({n: True for n in ('all144_arrays_prevalidated', 'all48_summaries_verified', 'resident_banks_rehashed_after',
        'raw_inputs_rehashed_after', 'snapshots_rehashed_after', 'scipy_inventory_rehashed_after')})
    value.update({n: False for n in ('references_read', 'positive_masks_constructed', 'FIT_performed', 'optimizer_executed',
        'RGB_decoded', 'selection_performed', 'ownership_verified', 'quality_verified', 'adoption')})
    return value, selected


def test_saved_scalar_validator_all32_nativeK_not_capped64():
    value, selected = receipt(); worker.validate_receipt(join.rt, value, selected)
    assert value['images'][0]['native_pairs'] == 100


@pytest.mark.parametrize('path,value', [(('images_completed',), 31), (('elapsed_seconds',), 3600), (('all48_summaries_verified',), False),
    (('positive_masks_constructed',), True), (('runtime', 'torch'), '2.1.2'), (('runtime', 'numpy'), '1.26.4'),
    (('runtime', 'tf32'), True), (('max_rss_bytes',), worker.MEMORY+1), (('peak_torch_reserved',), worker.MEMORY+1),
    (('segment_controls', 0, 'width'), 3), (('scipy_inventory', 'version'), '1.16.2'),
    (('images', 0, 'native_pairs'), 64), (('images', 0, 'objects'), 128), (('images', 0, 'slot'), 47),
    (('images', 0, 'controls', 0, 'native_shape'), [2, 2, 32]),
    (('images', 0, 'controls', 0, 'partitions', 0, 'cpu_full_parity'), False),
    (('images', 0, 'controls', 0, 'partitions', 0, 'log_partition'), None),
    (('images', 0, 'controls', 0, 'partitions', 0, 'gradient_norm'), float('nan'))])
def test_saved_scalar_validator_rejects_partial_or_corrupt_claims(path, value):
    actual, selected = receipt(); target = actual
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError): worker.validate_receipt(join.rt, actual, selected)


def test_saved_undefined_partition_legal_only_zero_support():
    actual, selected = receipt()
    for row in actual['images'][0]['controls']:
        for part in row['partitions']:
            part.update(supported_groups=0, status='no_supported', log_partition=None, gradient_norm=0.)
    worker.validate_receipt(join.rt, actual, selected)


def test_last_raw_bank_failure_prevents_every_reconstruction_or_gpu_score(monkeypatch):
    proof, rows, chosen, g, t, calls, visited, _ = measure_setup(monkeypatch); seen = []; report = {}
    def load(path, row, fields):
        seen.append(str(path))
        if len(seen) == 144: raise ValueError('Manufactured last raw member')
        return {}
    monkeypatch.setattr(join, 'load_bank', load)
    with pytest.raises(ValueError): worker.measure(np, t, g, join.rt, proof, rows, chosen, lambda: None, report)
    assert len(seen) == 144 and not visited and not calls and report['images_completed'] == 0
    assert report['failure_stage'] == 'all144_array_prevalidation' and report['raw_inputs_rehashed_after']


def test_upload_finished_then_deadline_checks_device_snapshot_on_failure(monkeypatch):
    proof, rows, chosen, g, t, calls, visited, _ = measure_setup(monkeypatch); report = {}; uploaded = []
    original = scorer.prepare_marginal_packed_torch
    def prepare(packed, device):
        value = original(packed, device); uploaded.append(value); return value
    monkeypatch.setattr(scorer, 'prepare_marginal_packed_torch', prepare)
    count = []
    def check():
        if uploaded:
            count.append(1)
            if len(count) == 1: raise TimeoutError('Manufactured post-upload deadline')
    fingerprints = []; original_fingerprint = g.device_fingerprint
    g.device_fingerprint = lambda t, value: fingerprints.append(value) or original_fingerprint(t, value)
    with pytest.raises(TimeoutError): worker.measure(np, t, g, join.rt, proof, rows, chosen, check, report)
    assert report['failure_stage'] == 'image_device_upload' and len(uploaded) == 1 and len(fingerprints) == 2
    assert report['snapshots_rehashed_after'] and report['raw_inputs_rehashed_after'] and not calls


def test_score_buffer_mutation_after_partition_is_not_repeat_pass(monkeypatch):
    import world_reward.coherent_pair_partition as primitive
    g, t, _ = fake_device(monkeypatch); packed = tiny_packed(); device = scorer.prepare_marginal_packed_torch(packed, device='cuda:0')
    original = primitive.marginal_partition
    def bad(score, theta, **kwargs):
        answer = original(score, theta, **kwargs)
        if kwargs.get('backend') == 'torch': score.scores_a.array.flat[0] += 1.
        return answer
    monkeypatch.setattr(primitive, 'marginal_partition', bad)
    with pytest.raises((ValueError, AssertionError)):
        worker._controls(np, t, g, packed, device, lambda: None, {})


@pytest.mark.parametrize('fault', ['numpy', 'torch', 'cuda', 'capability', 'cublas', 'gpu_capacity', 'segment_count'])
def test_runtime_failfast_is_before_photo_scoring(monkeypatch, fault):
    monkeypatch.setattr(worker.sys, 'platform', 'linux'); monkeypatch.setattr(worker.sys, 'version_info', (3, 11, 10))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8' if fault != 'cublas' else '')
    properties = SimpleNamespace(name='NVIDIA H100 NVL', major=9, minor=0,
                                 total_memory=worker.MEMORY-1 if fault == 'gpu_capacity' else 99456909312)
    calls = []; t = SimpleNamespace(__version__='2.5.1+cu124' if fault != 'torch' else '2.1.2',
        version=SimpleNamespace(cuda='12.4' if fault != 'cuda' else '11.8'),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)), cudnn=SimpleNamespace(allow_tf32=True)),
        use_deterministic_algorithms=lambda x: calls.append('deterministic'), set_num_threads=lambda x: calls.append('threads'),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_capability=lambda _: (9, 0) if fault != 'capability' else (8, 0), get_device_name=lambda _: 'H100',
            get_device_properties=lambda _: properties, set_per_process_memory_fraction=lambda *x: calls.append('fraction'),
            reset_peak_memory_stats=lambda: calls.append('reset')))
    native_numpy = SimpleNamespace(__version__='1.26.3' if fault != 'numpy' else '1.26.4')
    monkeypatch.setattr(qualified, 'scipy_evidence', lambda rt: {'version': '1.16.3'})
    g = SimpleNamespace(segment_controls=lambda *x: [None]*(19 if fault == 'segment_count' else 20))
    with pytest.raises(ValueError): worker._runtime(native_numpy, t, g, join.rt, lambda: None, {})
    assert 'fraction' not in calls if fault in ('numpy', 'torch', 'cuda', 'capability', 'cublas', 'gpu_capacity') else True

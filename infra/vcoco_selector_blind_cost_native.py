"""Complete saved-bank partition cost, never positive labels, FIT or selection.

The host authenticates source/image/private FIT membership and passes only the
public48 raw-bank proof, saved46 row metadata and fixed32 opaque slot/ID/folds.
This worker has no CLI, model, reference loader or alternate memory profile.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time

BUDGET, MEMORY = 3600, 64 << 30
SUMMARY_FIELDS = ('image_id', 'original_slot', 'acquired_ordinal', 'persons', 'objects',
    'native_pairs', 'person_side_object_rows', 'person_side_pair_rows', 'pair_object_rows',
    'supported_counts', 'nan_counts', 'evidence_fingerprint', 'source_observation_references', 'raw_bank_fingerprint')
FLOAT_FIELDS = ('native_scores_a', 'native_scores_b', 'scores_a', 'scores_b',
    'geometry_derivatives_a', 'geometry_derivatives_b', 'alpha_derivatives_b')
BOOL_FIELDS = ('native_supported', 'native_route_supported', 'supported')


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _memory(t, report):
    current = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    peak_path = Path('/sys/fs/cgroup/memory.peak')
    peak = int(peak_path.read_text().strip()) if peak_path.exists() else None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    allocated, reserved = (t.cuda.max_memory_allocated(), t.cuda.max_memory_reserved()) if 'runtime' in report else (0, 0)
    if (limit == 'max' or not 0 < int(limit) <= MEMORY or not 0 <= current <= MEMORY
            or not 0 < rss <= MEMORY or not 0 <= allocated <= reserved <= MEMORY
            or peak is not None and not 0 <= peak <= MEMORY):
        raise ValueError('Fixed64GiB CPU/cgroup/device memory gate')
    report.update(max_rss_bytes=rss, max_observed_cgroup_bytes=max(current, report.get('max_observed_cgroup_bytes', 0)),
                  peak_torch_allocated=allocated, peak_torch_reserved=reserved, cgroup_memory_limit=int(limit),
                  cgroup_peak_available=peak is not None, cgroup_peak_bytes=peak)


def _runtime(np, t, g, rt, check, report):
    from coherent_pair_gpu_objective_probe import scipy_evidence
    check()
    if (sys.platform != 'linux' or sys.version_info[:2] != (3, 11) or np.__version__ != '1.26.3'
            or t.__version__ != '2.5.1+cu124' or t.version.cuda != '12.4' or not t.cuda.is_available()
            or t.cuda.device_count() != 1 or t.cuda.get_device_capability(0) != (9, 0)
            or 'H100' not in t.cuda.get_device_name(0) or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('Exact original7eb FP64 H100 runtime required')
    t.backends.cuda.matmul.allow_tf32 = False; t.backends.cudnn.allow_tf32 = False
    t.use_deterministic_algorithms(True); t.set_num_threads(4)
    props = t.cuda.get_device_properties(0)
    if props.total_memory < MEMORY: raise ValueError('Actual device below frozen64GiB profile')
    t.cuda.set_per_process_memory_fraction(MEMORY/props.total_memory, 0); t.cuda.reset_peak_memory_stats()
    report['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__, torch=t.__version__, cuda=t.version.cuda,
        gpu_name=props.name, capability=[props.major, props.minor], gpu_total_bytes=props.total_memory,
        deterministic_algorithms=True, tf32=False, cublas_workspace_config=':4096:8')
    report['scipy_inventory'] = scipy_evidence(rt); check()
    report['segment_controls'] = g.segment_controls(np, t, __import__('world_reward.coherent_pair_packed_torch', fromlist=['_segment']), check)
    if len(report['segment_controls']) != 20: raise ValueError('Complete native segment controls required')
    return report['scipy_inventory']


def _inputs(join, proof, rows, selection):
    join.validate_population(proof)
    if (type(rows) is not list or len(rows) != 48 or type(selection) is not tuple or len(selection) != 32
            or any(type(x) is not dict or set(x) != {'slot', 'image_id', 'fold'} for x in selection)
            or any(type(x['slot']) is not int or not 0 <= x['slot'] < 48 or type(x['image_id']) is not str
                   or type(x['fold']) is not int or not 0 <= x['fold'] < 4 for x in selection)
            or [x['slot'] for x in selection] != list(range(32)) or len({x['image_id'] for x in selection}) != 32
            or any(re.fullmatch('[0-9a-f]{32}', x['image_id']) is None for x in selection)):
        raise ValueError('Complete fixed32 opaque slot/ID/fold selection required')
    ids = sorted((x['image_id'] for x in selection), key=lambda x: (hashlib.sha256(
        ('world_reward.vcoco_selector_cv_v1/'+x).encode()).hexdigest(), x))
    folds = {identifier: i % 4 for i, identifier in enumerate(ids)}
    for x in selection:
        if x['image_id'] != proof['records'][x['slot']]['endpoint']['image_id'] or x['fold'] != folds[x['image_id']]:
            raise ValueError('Original opaque identity/frozen fold differs')
    for i, (row, record) in enumerate(zip(rows, proof['records'])):
        expected = join.expected_evidence_metadata(record)
        if (type(row) is not dict or not set(SUMMARY_FIELDS) <= set(row) or row['original_slot'] != i
                or row['acquired_ordinal'] != i or row['image_id'] != record['endpoint']['image_id']
                or type(row.get('arrays')) is not dict or set(row['arrays']) != set(expected)
                or any(row['arrays'][n]['shape'] != s or row['arrays'][n]['dtype'] != d for n, (s, d) in expected.items())):
            raise ValueError('Full original46 saved summary metadata required')
        if row.get('persons') != record['pose']['persons'] or row.get('native_pairs') != record['hoi']['hand_object_pairs']:
            raise ValueError('Original untruncated P/K summary differs')


def _reconstruct(np, join, record, summary, check):
    args = []
    for key, path, fields in zip(('endpoint', 'pose', 'hoi'), record['paths'],
            (join.numerical.ENDPOINT_FIELDS, join.numerical.POSE_FIELDS, join.numerical.HOI_FIELDS)):
        args.extend((record[key], join.load_bank(Path(path), record[key], fields))); check()
    value = join.numerical.reconstruct_interaction(*args, population=48)
    actual = join.summary(value)
    if (_json(actual) != _json({n: summary[n] for n in SUMMARY_FIELDS})
            or any(join.numerical._identity(a) != summary['arrays'][n] for n, a in join.evidence_arrays(value).items())):
        raise ValueError('Original47 reconstruction/saved46 numerical fingerprints differ')
    return value


def _timed(t, check, report, function):
    check(); _memory(t, report); t.cuda.synchronize(); start = time.monotonic()
    value = function(); t.cuda.synchronize(); check(); _memory(t, report)
    return value, time.monotonic()-start


def _partition_bits(g, a, b):
    if (a.status != b.status or a.supported_groups != b.supported_groups or a.arm != b.arm
            or a.temperature != b.temperature or a.score_reference != b.score_reference):
        raise ValueError('Complete partition metadata differs')
    import numpy as np
    g.bits(np.asarray(a.log_partition if a.log_partition is not None else np.nan, np.float64),
           np.asarray(b.log_partition if b.log_partition is not None else np.nan, np.float64))
    g.bits(a.gradient, b.gradient)


def _controls(np, t, g, packed, device, check, report):
    from world_reward import coherent_pair_packed_torch as scorer, coherent_route_scorer as core
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    from world_reward.coherent_pair_partition import marginal_partition
    result = []; coefficients = (np.zeros(17, np.float64), np.arange(1, 18, dtype=np.float64)/math.sqrt(sum(i*i for i in range(1, 18))))
    for theta_index, theta in enumerate(coefficients):
        for alpha in (0., 1.):
            report.update(phase='score_partition', current_theta_index=theta_index, current_alpha=alpha)
            a, seconds = _timed(t, check, report, lambda: scorer.score_pair_marginal_packed_torch(device, theta, temperature=1., alpha=alpha))
            b, repeat_seconds = _timed(t, check, report, lambda: scorer.score_pair_marginal_packed_torch(device, theta, temperature=1., alpha=alpha))
            actual, copy_seconds = _timed(t, check, report, lambda: g.arrays(a))
            repeat, repeat_copy_seconds = _timed(t, check, report, lambda: g.arrays(b))
            report['phase'] = 'cpu_full_oracle'
            cpu, cpu_seconds = _timed(t, check, report, lambda: score_pair_marginal_packed(packed, theta, temperature=1., alpha=alpha))
            for name in (*FLOAT_FIELDS, *BOOL_FIELDS):
                g.bits(actual[name], repeat[name]); reference = getattr(cpu, name)
                if actual[name].shape != reference.shape or actual[name].dtype != reference.dtype: raise ValueError('Full oracle shape/dtype differs')
                if name in BOOL_FIELDS: g.bits(actual[name], reference)
                else: np.testing.assert_allclose(actual[name], reference, rtol=1e-12, atol=1e-12, equal_nan=True)
            for value in (a, b):
                if (core._fingerprint(value.identity) != core._fingerprint(cpu.identity) or value.device != 'cuda:0'
                        or value.distribution_fingerprint != cpu.distribution_fingerprint
                        or value.parameter_fingerprint != cpu.parameter_fingerprint or value.temperature != 1. or value.alpha != alpha):
                    raise ValueError('Original complete identity/distribution/parameter differs')
                if alpha == 0.:
                    for x, y in (('native_scores_a', 'native_scores_b'), ('scores_a', 'scores_b'), ('geometry_derivatives_a', 'geometry_derivatives_b')):
                        g.bits(actual[x], actual[y])
                        if getattr(value, x).data_ptr() != getattr(value, y).data_ptr(): raise ValueError('Alpha0 shared buffers differ')
            partitions = []
            for arm in ('A', 'B') if alpha == 0. else ('B',):
                first, part_seconds = _timed(t, check, report, lambda: marginal_partition(a, theta, alpha=alpha, arm=arm, backend='torch'))
                second, part_repeat_seconds = _timed(t, check, report, lambda: marginal_partition(b, theta, alpha=alpha, arm=arm, backend='torch'))
                expected, part_cpu_seconds = _timed(t, check, report, lambda: marginal_partition(cpu, theta, alpha=alpha, arm=arm))
                _partition_bits(g, first, second)
                if first.status != expected.status or first.supported_groups != expected.supported_groups: raise ValueError('Partition support differs')
                if first.log_partition is not None:
                    np.testing.assert_allclose(first.log_partition, expected.log_partition, rtol=1e-12, atol=1e-12)
                elif expected.log_partition is not None: raise ValueError('Undefined support differs')
                np.testing.assert_allclose(first.gradient, expected.gradient, rtol=1e-12, atol=1e-12)
                partitions.append(dict(arm=arm, status=first.status, supported_groups=first.supported_groups,
                    log_partition=first.log_partition, gradient_norm=math.hypot(*map(float, first.gradient)),
                    gradient_sha256=hashlib.sha256(first.gradient.tobytes()).hexdigest(), seconds=part_seconds,
                    repeat_seconds=part_repeat_seconds, cpu_seconds=part_cpu_seconds, repeat_bits_exact=True, cpu_full_parity=True))
            for value, saved in ((a, actual), (b, repeat)):
                after = g.arrays(value)
                for name in (*FLOAT_FIELDS, *BOOL_FIELDS): g.bits(after[name], saved[name])
                if (core._fingerprint(value.identity) != core._fingerprint(cpu.identity)
                        or value.parameter_fingerprint != cpu.parameter_fingerprint
                        or value.distribution_fingerprint != cpu.distribution_fingerprint):
                    raise ValueError('Score buffers/metadata mutated during partition')
            result.append(dict(theta_index=theta_index, alpha=alpha, score_seconds=seconds, repeat_score_seconds=repeat_seconds,
                copy_seconds=copy_seconds, repeat_copy_seconds=repeat_copy_seconds, cpu_score_seconds=cpu_seconds,
                full_result_sha256=core._fingerprint(actual), identity_sha256=core._fingerprint(cpu.identity),
                distribution_fingerprint=cpu.distribution_fingerprint, parameter_fingerprint=cpu.parameter_fingerprint,
                native_shape=list(actual['native_scores_a'].shape), group_shape=list(actual['scores_a'].shape), partitions=partitions))
            del a, b, actual, repeat, cpu
    return result


def measure(np, t, g, rt, public_proof, summary_rows, selection, check, report):
    """One fixed blind32 run; caller seals failure/receipt and owns source checks.

    ``check`` is a no-argument inclusive source/deadline callback. No file is
    written. A partial report contains only counts, times, fingerprints and
    fixed safe failure stages. It never contains a positive mask or fitted loss.
    """
    import vcoco_full_interaction_join as join
    from coherent_pair_gpu_objective_probe import scipy_evidence
    from world_reward import coherent_pair_learning as learner, coherent_route_scorer as core
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_torch import prepare_marginal_packed_torch
    start = time.monotonic(); files = {}; bank = cache = packed = device = value = None; resident = []; before = inventory = None
    report.update(status='fail', phase='public_inputs', images=[], images_completed=0, all144_arrays_prevalidated=False,
        all48_summaries_verified=False, raw_inputs_rehashed_after=False, snapshots_rehashed_after=False,
        references_read=False, positive_masks_constructed=False, FIT_performed=False, optimizer_executed=False,
        models_loaded=False, RGB_decoded=False, selection_performed=False, ownership_verified=False, quality_verified=False, adoption=False)
    def tick():
        check()
        if time.monotonic()-start >= BUDGET: raise TimeoutError('Inclusive3600s blind full32 cost')
        _memory(t, report)
    try:
        _inputs(join, public_proof, summary_rows, selection); before = core._fingerprint((public_proof, summary_rows, selection))
        for path, pin in public_proof['files'].items():
            p = rt.canonical(path); state = join.snapshot(p)
            if rt.identity(p, join.MAXIMUM) != pin: raise ValueError('Full144 original raw file differs')
            files[path] = (pin, state); tick()
        report['phase'] = 'runtime'; inventory = _runtime(np, t, g, rt, tick, report)
        report['phase'] = 'all144_array_prevalidation'
        for record in public_proof['records']:
            for key, path, fields in zip(('endpoint', 'pose', 'hoi'), record['paths'],
                    (join.numerical.ENDPOINT_FIELDS, join.numerical.POSE_FIELDS, join.numerical.HOI_FIELDS)):
                join.load_bank(Path(path), record[key], fields); tick()
        report['all144_arrays_prevalidated'] = True; report['phase'] = 'all48_reconstruction_resident32'
        chosen = {x['slot'] for x in selection}; ordered = {}; prep = time.monotonic()
        for slot, record in enumerate(public_proof['records']):
            value = _reconstruct(np, join, record, summary_rows[slot], tick)
            if slot in chosen:
                bank = learner.pair_route_bank(value.person, value.evidence)
                if learner._bank(bank)[1] != 3600: raise ValueError('All3600 original objects required')
                ordered[slot] = bank; resident.append((bank, core._fingerprint(bank)))
            value = bank = None; tick()
        report['all48_summaries_verified'] = True; report['resident_prepare_seconds'] = time.monotonic()-prep
        banks = tuple(ordered[x['slot']] for x in selection); report['phase'] = 'fit_only_scales_no_references'; prep = time.monotonic()
        scales = learner.fit_scales(banks); tick(); report['scale_seconds'] = time.monotonic()-prep
        for b, fingerprint in resident:
            if core._fingerprint(b) != fingerprint: raise ValueError('Resident full32 bank changed during scales')
        report.update(scales=list(map(float, scales.scale)), variable_values=list(map(bool, scales.variable)),
                      scale_fingerprint=core._fingerprint(scales), resident_banks_rehashed_after=True)
        del banks, ordered, b; resident.clear(); gc.collect(); tick()
        for item in selection:
            slot = item['slot']; image = dict(slot=slot, image_id=item['image_id'], fold=item['fold'])
            report['images'].append(image); report.update(phase='image_reconstruction', current_slot=slot); prep = time.monotonic()
            value = _reconstruct(np, join, public_proof['records'][slot], summary_rows[slot], tick)
            bank_pin = packed_pin = device_pin = None
            bank = learner.pair_route_bank(value.person, value.evidence); bank_pin = core._fingerprint(bank)
            image['reconstruct_seconds'] = time.monotonic()-prep
            report['phase'] = 'image_cache_packed'; prep = time.monotonic()
            cache = prepare_pair_cache(bank, scales); tick(); packed = prepare_marginal_packed(cache)
            packed_pin = core._fingerprint((packed.arrays, packed.identity)); image['cache_packed_seconds'] = time.monotonic()-prep; tick()
            n, o, k = learner._bank(bank)
            expected = int((cache.factors['good']*np.maximum(cache.factors['usable'].sum(axis=2), 1)[..., None]).sum())
            if o != 3600 or len(packed.arrays['route_refs']) != expected: raise ValueError('All original supported route incidences required')
            image.update(persons=n, objects=o, native_pairs=k, candidate_rows=n*2*o, tuple_rows=n*2*k, bridge_rows=k*o,
                person_groups=len(bank.person_boxes), object_groups=len(bank.object_boxes), route_references=expected,
                complete_geometries=len(packed.arrays['geometry_components']), bank_sha256=bank_pin, packed_sha256=packed_pin)
            report['phase'] = 'image_device_upload'
            tick(); t.cuda.synchronize(); prep = time.monotonic()
            device = prepare_marginal_packed_torch(packed, device='cuda:0')
            device_pin = g.device_fingerprint(t, device); t.cuda.synchronize(); tick()
            image['upload_seconds'] = time.monotonic()-prep
            image['controls'] = _controls(np, t, g, packed, device, tick, report)
            if (core._fingerprint(bank) != bank_pin or core._fingerprint((packed.arrays, packed.identity)) != packed_pin
                    or core._fingerprint((cache.factors, cache.scales, cache.person_members, cache.object_members)) != cache.factor_fingerprint
                    or g.device_fingerprint(t, device) != device_pin): raise ValueError('Complete raw/factor/CSR/device snapshot changed')
            image.update(snapshots_rehashed_after=True, device_sha256=device_pin)
            bank = cache = packed = device = value = None; gc.collect(); t.cuda.empty_cache(); tick(); report['images_completed'] += 1
        report['phase'] = 'complete'; tick(); report['status'] = 'pass'
    except BaseException as exc:
        report.update(status='fail', failure_stage=report['phase'], error_type=join.error(exc)); raise
    finally:
        try:
            for b, fingerprint in resident:
                if core._fingerprint(b) != fingerprint: raise ValueError('Resident bank drift on failure')
            if bank is not None and 'bank_pin' in locals() and bank_pin is not None and core._fingerprint(bank) != bank_pin: raise ValueError('Image bank drift on failure')
            if cache is not None and core._fingerprint((cache.factors, cache.scales, cache.person_members, cache.object_members)) != cache.factor_fingerprint:
                raise ValueError('Image factor drift on failure')
            if packed is not None and 'packed_pin' in locals() and packed_pin is not None and core._fingerprint((packed.arrays, packed.identity)) != packed_pin:
                raise ValueError('Image CSR drift on failure')
            if device is not None and 'device_pin' in locals() and device_pin is not None and g.device_fingerprint(t, device) != device_pin:
                raise ValueError('Image device drift on failure')
            if before is not None and before != core._fingerprint((public_proof, summary_rows, selection)): raise ValueError('Public metadata drift')
            for path, (pin, state) in files.items():
                if rt.identity(path, join.MAXIMUM) != pin or join.snapshot(Path(path)) != state: raise ValueError('Original raw file drift')
            report['raw_inputs_rehashed_after'] = len(files) == 144; report['snapshots_rehashed_after'] = before is not None
            if inventory is not None:
                if scipy_evidence(rt) != inventory: raise ValueError('Native SciPy inventory drift')
                report['scipy_inventory_rehashed_after'] = True
            report['elapsed_seconds'] = time.monotonic()-start; tick()
        except BaseException as exc:
            report.update(status='fail', post_error_type=join.error(exc)); raise
        finally:
            bank = cache = packed = device = value = None; resident.clear(); gc.collect()


def validate_receipt(rt, value, selection):
    """Stdlib saved-scalar ABI checks, not independent numerical recomputation."""
    finite = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    hashed = lambda x: type(x) is str and re.fullmatch('[0-9a-f]{64}', x) is not None
    r = value['runtime']; images = value['images']
    rt.require(value['status'] == 'pass' and value['phase'] == 'complete' and len(selection) == len(images) == 32
        and value['images_completed'] == 32 and all(value[n] is True for n in ('all144_arrays_prevalidated',
            'all48_summaries_verified', 'resident_banks_rehashed_after', 'raw_inputs_rehashed_after',
            'snapshots_rehashed_after', 'scipy_inventory_rehashed_after'))
        and all(value[n] is False for n in ('references_read', 'positive_masks_constructed', 'FIT_performed',
            'optimizer_executed', 'RGB_decoded', 'selection_performed', 'ownership_verified', 'quality_verified', 'adoption'))
        and value['models_loaded'] == 0 and finite(value['elapsed_seconds']) and value['elapsed_seconds'] < BUDGET,
        'Complete original blind32 control required')
    rt.require(r['python'].startswith('3.11.') and r['numpy'] == '1.26.3' and r['torch'] == '2.5.1+cu124'
        and r['cuda'] == '12.4' and r['capability'] == [9, 0] and 'H100' in r['gpu_name']
        and type(r['gpu_total_bytes']) is int and r['gpu_total_bytes'] >= MEMORY and r['deterministic_algorithms'] is True
        and r['tf32'] is False and r['cublas_workspace_config'] == ':4096:8'
        and all(finite(value[n]) and value[n] <= MEMORY for n in ('max_rss_bytes', 'max_observed_cgroup_bytes',
            'peak_torch_allocated', 'peak_torch_reserved', 'cgroup_memory_limit'))
        and 0 < value['max_rss_bytes'] and 0 < value['cgroup_memory_limit']
        and value['peak_torch_allocated'] <= value['peak_torch_reserved']
        and type(value['cgroup_peak_available']) is bool
        and ((finite(value['cgroup_peak_bytes']) and value['cgroup_peak_bytes'] <= MEMORY)
            if value['cgroup_peak_available'] else value['cgroup_peak_bytes'] is None), 'Actual fixed runtime/memory required')
    segments = value['segment_controls']
    rt.require(type(segments) is list and len(segments) == 20 and [(x['width'], x['empty'], x['operation']) for x in segments]
        == [(w, e, op) for w in (1, 9, 6, 2, 17) for e in (False, True) for op in ('sum', 'max')]
        and all(hashed(x['bytes_sha256']) for x in segments), 'Actual ordered20 segment controls required')
    s = value['scipy_inventory']
    rt.require(s['version'] == '1.16.3' and s['first_native_record_census'] is s['populated_record_claims_verified'] is True
        and s['caches_image_anchored_not_record_certified'] is True
        and s['empty_file_policy'] == 'explicit_original_record_sha256_empty_and_exact_size0_only'
        and type(s['entries']) is int and s['entries'] > 0 and hashed(s['source_fingerprint'])
        and s['record']['bytes'] > 0 and hashed(s['record']['sha256']) and bool(s['license_files']), 'Image-bound RECORD inventory required')
    rt.require(len(value['scales']) == len(value['variable_values']) == 12
        and all(finite(x) and x > 0 for x in value['scales']) and all(type(x) is bool for x in value['variable_values'])
        and hashed(value['scale_fingerprint']) and all(finite(value[n]) for n in ('resident_prepare_seconds', 'scale_seconds')),
        'Full32 measured FIT-only scales required')
    for image, selected in zip(images, selection):
        p, o, k = image['persons'], image['objects'], image['native_pairs']
        rt.require(all(image[n] == selected[n] for n in ('slot', 'image_id', 'fold'))
            and all(type(x) is int and x >= 0 for x in (p, o, k, image['route_references'], image['complete_geometries']))
            and o == 3600 and image['candidate_rows'] == p*2*o and image['tuple_rows'] == p*2*k and image['bridge_rows'] == k*o
            and 0 <= image['person_groups'] <= p and 0 < image['object_groups'] <= o
            and image['route_references'] <= p*2*o*max(k, 1) and image['snapshots_rehashed_after'] is True
            and all(hashed(image[n]) for n in ('bank_sha256', 'packed_sha256', 'device_sha256'))
            and all(finite(image[n]) for n in ('reconstruct_seconds', 'cache_packed_seconds', 'upload_seconds')),
            'Complete untruncated native person/object/pair population required')
        controls = image['controls']
        rt.require(len(controls) == 4 and [(x['theta_index'], x['alpha']) for x in controls] == [(0, 0.), (0, 1.), (1, 0.), (1, 1.)],
                   'All four frozen coefficient controls required')
        for row in controls:
            rt.require(row['native_shape'] == [p, 2, 3600] and row['group_shape'] == [image['person_groups'], image['object_groups']]
                and all(hashed(row[n]) for n in ('full_result_sha256', 'identity_sha256', 'distribution_fingerprint', 'parameter_fingerprint'))
                and all(finite(row[n]) for n in ('score_seconds', 'repeat_score_seconds', 'copy_seconds', 'repeat_copy_seconds', 'cpu_score_seconds'))
                and [x['arm'] for x in row['partitions']] == (['A', 'B'] if row['alpha'] == 0. else ['B']), 'Full rows/ordered A/B partition required')
            for part in row['partitions']:
                count = part['supported_groups']
                rt.require(type(count) is int and 0 <= count <= image['person_groups']*image['object_groups']
                    and part['status'] == ('supported' if count else 'no_supported')
                    and ((type(part['log_partition']) is float and math.isfinite(part['log_partition'])) if count else part['log_partition'] is None)
                    and hashed(part['gradient_sha256']) and finite(part['gradient_norm'])
                    and all(finite(part[n]) for n in ('seconds', 'repeat_seconds', 'cpu_seconds'))
                    and part['repeat_bits_exact'] is part['cpu_full_parity'] is True, 'Defined/undefined label-free partition ABI required')

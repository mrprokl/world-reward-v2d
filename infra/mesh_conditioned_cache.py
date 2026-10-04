"""Exact implementation regression on the four previously qualified sources.

No failed control is replayed and no physical geometry or threshold changes.
The caller owns binary/source authentication, offline runtime and total budget.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import stat

import numpy as np

import mesh_conditioned_geometry as controls

GeometryControlError = controls.GeometryControlError
NATIVE_SECONDS = 450


def cache_controls(slow_binary, fast_binary, scratch, remaining, *, official_helper, expected_sources):
    """Eight calls, byte-identical candidates/maps and equal physical metrics.

    All arms must pass. Failure retains partial tiny evidence; only owned
    procedural scratch is removed. No model/video, source changes or fallback.
    """
    slow_binary, fast_binary, scratch, official_helper = map(
        Path, (slow_binary, fast_binary, scratch, official_helper))
    require, identity = controls.require, controls.identity
    require(scratch.is_absolute() and scratch.resolve() == scratch and scratch.is_dir()
            and not any(p.is_symlink() for p in (scratch, *scratch.parents)), 'Canonical owned scratch required')
    require(slow_binary != fast_binary, 'Two independently authenticated implementations required')
    require(type(expected_sources) is dict and set(expected_sources) == set(controls.FIXTURE_NAMES),
            'Exact actual previously qualified source hashes required')
    work = scratch/'cache-regression-controls'
    require(not work.exists() and not work.is_symlink(), 'Fresh regression scratch required')
    artifacts = (slow_binary, fast_binary, official_helper)
    before = {str(p): identity(p) for p in artifacts}
    require(before[str(official_helper)] == dict(bytes=2031, sha256=controls.endpoint.BUDGET_HELPER_SHA),
            'Exact official single helper required')
    legacy = controls.geometry.validate_legacy_sources()
    report = dict(stage='mesh_conditioned_cache_controls_v1', status='fail', adoption=False,
                  production_mesh_validated=False, challenge_inputs_used=False, challenge_performance_verified=False,
                  previous_successful_sources_only=True, failed_controls_replayed=False, source_artifacts=before,
                  legacy_sources=legacy, native_budget_seconds=NATIVE_SECONDS, maximum_native_calls=8,
                  paired_fixtures=[], universal_equivalence_claimed=False)
    work.mkdir(mode=0o700); owner = work.stat()
    sources_before = []; failure = None
    try:
        remaining()
        spec = importlib.util.spec_from_file_location('wr_cache_official_single_helper', official_helper)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        sources = controls.fixtures()
        require(tuple(name for name, _, _ in sources) == controls.FIXTURE_NAMES, 'Unchanged qualified source cohort required')
        for name, source, cavity in sources:
            remaining(); hashes = controls.array_hashes(source); sources_before.append((source, hashes))
            require(hashes == expected_sources[name], 'Procedural source differs from actual qualified source bytes')
            chart = controls.prepare_conditioning(*source)
            require(chart.diagnostics['roundtrip_numerically_exact'], 'Source roundtrip must remain exact')
            require(len(source[1]) > 4096 and controls.serialization_preflight(*source)['position_weld_admissible'],
                    'Source must exercise real admissible reduction')
            physical = controls.topology_and_embedding(controls.math_mesh(source), cavity)
            stored = source[0].astype(np.float32), source[1]
            physical32 = controls.topology_and_embedding(controls.math_mesh(stored), cavity)
            require(controls.serialization_preflight(*stored)['position_weld_admissible'], 'Source F32/weld safety required')
            report['paired_fixtures'].append(dict(fixture=name, source_array_sha256=hashes,
                source_float32_array_sha256=controls.array_hashes(stored), source_topology=physical,
                source_float32_topology=physical32, conditioning=dict(chart.diagnostics), comparisons=[]))
        # The same remaining callback keeps total-budget accounting while
        # capping each native subprocess at450s, without changing old helpers.
        def bounded_remaining():
            return min(NATIVE_SECONDS, remaining())
        for (name, source, cavity), fixture in zip(sources, report['paired_fixtures']):
            for implementation, binary in (('slow', slow_binary), ('cached', fast_binary)):
                remaining(); row = dict(implementation=implementation, method='conditioned', status='fail',
                                        phase='pre_native', native_attempts=0, native_returned=False)
                fixture['comparisons'].append(row)
                directory = work/(name+'-'+implementation); directory.mkdir(mode=0o700)
                try:
                    controls.branch(source, cavity, binary, 'conditioned', directory, helper, bounded_remaining, row)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc)[-500:])
                    raise
                require(controls.array_hashes(source) == fixture['source_array_sha256'], 'Shared source changed')
            a, b = fixture['comparisons']
            for filename in ('input.obj', 'candidate.obj', 'mapping.json'):
                require(a['native_artifacts'][filename] == b['native_artifacts'][filename],
                        'Cached implementation changed exact input/candidate/mapping bytes')
            for field in ('conditioning', 'serialization', 'native_volume', 'committed_collapses', 'serialization_vetoes',
                          'candidate', 'exported', 'packed', 'metric_baked', 'glb_identity', 'official_pack_fidelity',
                          'metric_scale_baked_once'):
                require(a[field] == b[field], 'Cached implementation changed native or physical-stage evidence')
            fixture.update(candidate_and_mapping_byte_exact=True, physical_stage_evidence_equal=True)
        require(len(report['paired_fixtures']) == 4 and all(
            len(f['comparisons']) == 2 and all(r['status'] == 'pass' and r['committed_collapses'] > 0
                for r in f['comparisons']) for f in report['paired_fixtures']), 'All eight real-collapse arms required')
        report.update(status='pass', exact_implementation_regression_verified=True,
                      reroll_performed=False, thresholds_or_geometry_changed=False)
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__, error=str(exc)[-500:])
    finally:
        try:
            require(all(controls.array_hashes(mesh) == hashes for mesh, hashes in sources_before), 'Source arrays changed')
            require({str(p): identity(p) for p in artifacts} == before
                    and controls.geometry.validate_legacy_sources() == legacy, 'Original binary/helper/source changed')
            remaining(); report['sources_rehashed_after'] = True
        except Exception as exc:
            failure = failure or exc; report['post_error_type'] = type(exc).__name__
        try:
            now = work.lstat()
            require(stat.S_ISDIR(now.st_mode) and (now.st_dev, now.st_ino) == (owner.st_dev, owner.st_ino),
                    'Owned scratch replaced')
            shutil.rmtree(work); report['owned_scratch_removed'] = True; remaining()
        except Exception as exc:
            failure = failure or exc; report['cleanup_error_type'] = type(exc).__name__
    if failure:
        report['status'] = 'fail'
        raise GeometryControlError('Frozen cached implementation regression failed', report) from failure
    return report

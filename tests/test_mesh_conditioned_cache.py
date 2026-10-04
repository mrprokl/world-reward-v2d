"""Tiny mocked cache regression orchestration; no native compilation or data."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'infra'))
import mesh_conditioned_cache as regression


def setup(tmp_path, monkeypatch):
    controls = regression.controls
    slow, fast, helper = (tmp_path/name for name in ('slow', 'fast', 'helper'))
    for p in (slow, fast, helper):
        p.write_bytes(b'owned tiny file')
    monkeypatch.setattr(controls, 'identity', lambda p: dict(bytes=2031, sha256=controls.endpoint.BUDGET_HELPER_SHA))
    monkeypatch.setattr(controls.geometry, 'validate_legacy_sources', lambda: {'unmodified': True})
    monkeypatch.setattr(regression.importlib.util, 'spec_from_file_location',
        lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(regression.importlib.util, 'module_from_spec', lambda s: SimpleNamespace())
    v, f = controls.radial_mesh(8, 6)
    f = np.tile(f, (52, 1))  # Repeated tiny faces only behind mock topology spies.
    sources = tuple((name, (v.copy(), f.copy()), index >= 2)
                    for index, name in enumerate(controls.FIXTURE_NAMES))
    monkeypatch.setattr(controls, 'fixtures', lambda: sources)
    monkeypatch.setattr(controls, 'topology_and_embedding', lambda *a: {'mock_only': True})
    monkeypatch.setattr(controls, 'serialization_preflight', lambda *a: {'position_weld_admissible': True})
    return slow, fast, helper, sources


def good_row():
    stage = dict(sampled_bidirectional_chamfer_diagonal_ratio=.005, net_volume_relative_error=.001,
                 birthface_matched_shells=[{'relative_volume_error': .001}], stored_array_sha256=['tiny'])
    return dict(status='pass', phase='complete', native_attempts=1, native_returned=True,
        committed_collapses=5, serialization_vetoes=0, native_artifacts={
            name: {'bytes': 10, 'sha256': 'a'*64} for name in ('input.obj', 'candidate.obj', 'mapping.json')},
        conditioning={'physical_geometry_rescaled': False}, serialization={'committed_collapses': 5},
        native_volume={'committed_collapses': 5}, candidate=copy.deepcopy(stage), exported=copy.deepcopy(stage),
        packed=copy.deepcopy(stage), metric_baked=copy.deepcopy(stage), glb_identity={'bytes': 15, 'sha256': 'b'*64},
        official_pack_fidelity={'oriented_triangles_exact': True}, metric_scale_baked_once=.375)


def invoke(slow, fast, scratch, remaining, helper, sources):
    return regression.cache_controls(slow, fast, scratch, remaining, official_helper=helper,
        expected_sources={name: regression.controls.array_hashes(mesh) for name, mesh, _ in sources})


def test_same_qualified_four_sources_all_eight_real_collapses_and_exact_bytes(tmp_path, monkeypatch):
    slow, fast, helper, sources = setup(tmp_path, monkeypatch); seen = []
    def branch(source, cavity, binary, method, work, h, remaining, record):
        assert method == 'conditioned' and remaining() == 450
        seen.append((binary.name, regression.controls.array_hashes(source)))
        record.update(good_row())
    monkeypatch.setattr(regression.controls, 'branch', branch)
    report = invoke(slow, fast, tmp_path, lambda: 5400, helper, sources)
    assert report['status'] == 'pass' and len(seen) == 8
    assert all(seen[i][1] == seen[i+1][1] for i in range(0, 8, 2))
    assert report['maximum_native_calls'] == 8 and report['native_budget_seconds'] == 450
    assert report['sources_rehashed_after'] and report['owned_scratch_removed']
    assert all(f['candidate_and_mapping_byte_exact'] and f['physical_stage_evidence_equal']
               for f in report['paired_fixtures'])
    assert not report['adoption'] and not report['failed_controls_replayed']
    assert not (tmp_path/'cache-regression-controls').exists()


@pytest.mark.parametrize('fault', ['candidate', 'mapping', 'counter', 'stage', 'timeout', 'technical', 'roundtrip', 'source'])
def test_any_divergence_or_failure_closes_with_partial_receipt_no_reroll(tmp_path, monkeypatch, fault):
    slow, fast, helper, sources = setup(tmp_path, monkeypatch); calls = []
    foreign = tmp_path/'foreign'; foreign.write_bytes(b'preserve')
    if fault == 'roundtrip':
        monkeypatch.setattr(regression.controls, 'prepare_conditioning',
                            lambda *a: SimpleNamespace(diagnostics={'roundtrip_numerically_exact': False}))
    def branch(source, cavity, binary, method, work, h, remaining, row):
        calls.append(binary.name)
        row.update(good_row())
        if binary == fast:
            if fault in ('candidate', 'mapping'):
                row['native_artifacts']['candidate.obj' if fault == 'candidate' else 'mapping.json']['sha256'] = 'c'*64
            if fault == 'counter': row['serialization_vetoes'] = 1
            if fault == 'stage': row['packed']['net_volume_relative_error'] = .002
            if fault == 'timeout': raise regression.controls.subprocess.TimeoutExpired('native', 450)
            if fault == 'technical': raise PermissionError('unavailable')
        if fault == 'source': source[0][0, 0] += 1
    monkeypatch.setattr(regression.controls, 'branch', branch)
    with pytest.raises(regression.GeometryControlError) as exc:
        invoke(slow, fast, tmp_path, lambda: 5400, helper, sources)
    report = exc.value.report
    assert report['status'] == 'fail' and len(calls) <= 2 and report['owned_scratch_removed']
    assert foreign.read_bytes() == b'preserve' and not report['failed_controls_replayed']
    if fault == 'roundtrip': assert not calls
    if fault != 'source': assert report['sources_rehashed_after']


def test_existing_namespace_or_identical_binary_rejected_before_models(tmp_path):
    (tmp_path/'cache-regression-controls').mkdir()
    with pytest.raises(ValueError, match='Fresh'):
        regression.cache_controls(tmp_path/'slow', tmp_path/'fast', tmp_path, lambda: 1,
                                  official_helper=tmp_path/'helper', expected_sources=dict.fromkeys(regression.controls.FIXTURE_NAMES))
    with pytest.raises(ValueError, match='Two'):
        regression.cache_controls(tmp_path/'same', tmp_path/'same', tmp_path, lambda: 1,
                                  official_helper=tmp_path/'helper', expected_sources=dict.fromkeys(regression.controls.FIXTURE_NAMES))


def test_subprocess_remaining_cap_does_not_replace_inclusive_deadline(tmp_path, monkeypatch):
    slow, fast, helper, sources = setup(tmp_path, monkeypatch); calls = []
    def branch(source, cavity, binary, method, work, h, remaining, row):
        calls.append(remaining()); row.update(good_row())
    monkeypatch.setattr(regression.controls, 'branch', branch)
    invoke(slow, fast, tmp_path, lambda: 12.25, helper, sources)
    assert calls == [12.25]*8


def test_changed_success_fixture_bytes_fail_before_any_native(tmp_path, monkeypatch):
    slow, fast, helper, sources = setup(tmp_path, monkeypatch)
    pins = {name: regression.controls.array_hashes(mesh) for name, mesh, _ in sources}
    pins[regression.controls.FIXTURE_NAMES[0]] = ['0'*64, '1'*64]
    monkeypatch.setattr(regression.controls, 'branch', lambda *a: pytest.fail('No native allowed'))
    with pytest.raises(regression.GeometryControlError) as exc:
        regression.cache_controls(slow, fast, tmp_path, lambda: 5400, official_helper=helper, expected_sources=pins)
    assert exc.value.report['status'] == 'fail' and exc.value.report['paired_fixtures'] == []

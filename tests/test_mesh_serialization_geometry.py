"""Tiny procedural/source contracts; no Trimesh install or native QEM locally."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

INFRA = Path(__file__).resolve().parents[1]/'infra'
sys.path.insert(0, str(INFRA))
spec = importlib.util.spec_from_file_location('geometry_controls_test', INFRA/'mesh_serialization_geometry.py')
controls = importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)
from world_reward.mesh_serialization import serialization_preflight


def test_capsule_formula_full_closed_topology_and_safe_source_without_native_backend():
    v, f = controls.capsule()
    assert v.shape == (6050, 3) and f.shape == (12096, 3)
    assert v.dtype == np.float64 and f.dtype == np.int64
    before = v.tobytes(), f.tobytes()
    report = serialization_preflight(v, f)
    assert report['position_weld_admissible'] and report['ignored_orphan_vertices'] == 0
    topology = controls.geometry.exact_mesh_topology(v, f)
    assert topology['components'][0]['euler'] == 2 and topology['components'][0]['volume_sign'] == 1
    assert len(topology['components']) == 1
    assert v[-2, 2] == pytest.approx(-controls.SCALE) and v[-1, 2] == pytest.approx(controls.SCALE)
    assert (v.tobytes(), f.tobytes()) == before


def test_capsule_periodic_seam_no_duplicate_poles_or_rings_and_fixed_corrugation():
    v, _ = controls.capsule()
    canonical = v/controls.SCALE
    assert len(np.unique(v, axis=0)) == len(v)
    assert np.count_nonzero(np.all(canonical[:, :2] == 0, axis=1)) == 2
    for k in (0, 1, 7, 16, 25, 32):
        z = -.65+1.3*k/32
        radius = .35*(1+.035*np.sin(24*np.pi*k/32)*(1-(z/.65)**2)**2)
        ring = canonical[(15+k)*96:(16+k)*96]
        np.testing.assert_allclose(np.linalg.norm(ring[:, :2], axis=1), radius, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(ring[:, 2], np.full(96, z))


def test_ellipsoid_resolution_axes_inward_winding_and_both_same_scale(monkeypatch):
    calls = []
    def sphere(*, subdivisions):
        calls.append(subdivisions)
        vertices = np.ones((2562 if subdivisions == 4 else 642, 3))
        faces = np.tile([0, 1, 2], (5120 if subdivisions == 4 else 1280, 1))
        return SimpleNamespace(vertices=vertices, faces=faces)
    monkeypatch.setitem(sys.modules, 'trimesh', SimpleNamespace(creation=SimpleNamespace(icosphere=sphere)))
    shapes = controls.fixtures()
    assert calls == [4, 3] and [s[0] for s in shapes] == list(controls.FIXTURE_NAMES)
    v, f = shapes[0][1]
    np.testing.assert_array_equal(v[:2562], np.tile(np.array([1., .75, .55])*controls.SCALE, (2562, 1)))
    np.testing.assert_array_equal(v[2562:], np.tile(np.array([1., .75, .55])*.96*controls.SCALE, (642, 1)))
    assert f[5120].tolist() == [2564, 2563, 2562]
    assert shapes[0][2] is True and shapes[1][2] is False


def test_exact_f32_promotion_only_changes_private_math_view_not_stored_bytes():
    v = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    before = v.tobytes(), f.tobytes()
    promoted, faces = controls.math_mesh((v, f))
    assert promoted.dtype == np.float64 and faces is f
    assert np.array_equal(promoted, v) and not np.shares_memory(promoted, v)
    assert (v.tobytes(), f.tobytes()) == before


def test_source_api_no_old_run_globals_or_old_simplifier_and_only_four_native_attempts():
    text = (INFRA/'mesh_serialization_geometry.py').read_text()
    ast.parse(text)
    assert 'def geometry_controls(new_binary, scratch, remaining, *, official_helper: Path)' in text
    assert "(('original', ORIGINAL_BINARY), ('serialization', new_binary))" in text
    assert 'timeout=min(NATIVE_SECONDS, remaining())' in text and controls.NATIVE_SECONDS == 900
    assert 'maximum_native_calls=4' in text and controls.METRIC_SCALE == .375
    assert "mapping = mapping['native_volume']" in text
    assert "serialization['committed_collapses'] > 0" in text
    assert "status='pass' if vetoes else 'inconclusive'" in text
    assert 'precision_volume_gate.run(' not in text and 'volume.simplify(' not in text
    assert 'if __name__' not in text and 'global ' not in text
    assert "helper.budget_mesh(str(glb), faces=4096, vertices=4096)" in text
    assert 'geometry.verify_pack_fidelity(exported, pv, pf)' in text
    assert 'export_birth_mapping(candidate, exported, mapping)' in text
    assert 'pmap = packed_mapping(exported, compact, emap)' in text
    assert 'source_float32_topology=stored_topology' in text
    assert 'topology_and_embedding(math_mesh(stored_source), cavity)' in text


def setup_orchestration(tmp_path, monkeypatch):
    old, new, helper = (tmp_path/n for n in ('old', 'new', 'helper'))
    for p in (old, new, helper):
        p.write_bytes(b'x')
    monkeypatch.setattr(controls, 'ORIGINAL_BINARY', old)
    monkeypatch.setattr(controls, 'identity', lambda p: dict(bytes=2031, sha256=controls.endpoint.BUDGET_HELPER_SHA))
    monkeypatch.setattr(controls.geometry, 'validate_legacy_sources', lambda: {'legacy': 'proof'})
    monkeypatch.setattr(controls.importlib.util, 'spec_from_file_location', lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(controls.importlib.util, 'module_from_spec', lambda s: SimpleNamespace())
    v, f = controls.capsule()
    monkeypatch.setattr(controls, 'fixtures', lambda: ((controls.FIXTURE_NAMES[0], (v, f), True),
                                                    (controls.FIXTURE_NAMES[1], (v.copy(), f.copy()), False)))
    monkeypatch.setattr(controls, 'topology_and_embedding', lambda mesh, cavity: {'checked': True})
    return new, helper


def test_original_failure_is_comparison_and_new_arms_same_frozen_source(tmp_path, monkeypatch):
    new, helper = setup_orchestration(tmp_path, monkeypatch)
    seen = []
    def branch(source, cavity, binary, method, work, module, remaining, record):
        seen.append((method, controls.array_hashes(source)))
        record['native_attempts'] = 1
        if method == 'original':
            raise ValueError('Comparator gate failed')
        record.update(status='pass', serialization_vetoes=1, committed_collapses=3)
    monkeypatch.setattr(controls, 'branch', branch)
    report = controls.geometry_controls(new, tmp_path, lambda: 20, official_helper=helper)
    assert report['status'] == 'pass' and report['serialization_vetoes'] == 2
    assert [a[0] for a in seen] == ['original', 'serialization', 'original', 'serialization']
    assert seen[0][1] == seen[1][1] and seen[2][1] == seen[3][1]
    assert report['sources_rehashed_after'] and report['owned_scratch_removed'] and not report['adoption']
    assert not (tmp_path/'geometry-controls').exists()


def test_zero_veto_scope_inconclusive_not_adopted_or_retried(tmp_path, monkeypatch):
    new, helper = setup_orchestration(tmp_path, monkeypatch)
    calls = []
    def branch(*args):
        calls.append(args[3]); args[-1].update(status='pass', serialization_vetoes=0)
    monkeypatch.setattr(controls, 'branch', branch)
    report = controls.geometry_controls(new, tmp_path, lambda: 20, official_helper=helper)
    assert report['status'] == 'inconclusive' and not report['mechanism_exercised']
    assert len(calls) == 4 and not report['reroll_performed']


def test_new_failure_stops_before_second_pair_and_rehashes_then_removes_only_own_scratch(tmp_path, monkeypatch):
    new, helper = setup_orchestration(tmp_path, monkeypatch)
    sentinel = tmp_path/'foreign'; sentinel.write_bytes(b'untouched')
    calls = []
    def branch(*args):
        calls.append(args[3])
        if args[3] == 'serialization':
            raise ValueError('New metric gate failed')
        args[-1].update(status='pass')
    monkeypatch.setattr(controls, 'branch', branch)
    with pytest.raises(controls.GeometryControlError) as exc:
        controls.geometry_controls(new, tmp_path, lambda: 20, official_helper=helper)
    assert calls == ['original', 'serialization']
    assert exc.value.report['status'] == 'fail' and exc.value.report['sources_rehashed_after']
    assert sentinel.read_bytes() == b'untouched' and not (tmp_path/'geometry-controls').exists()


def test_reused_scratch_namespace_fails_without_any_native_work(tmp_path, monkeypatch):
    (tmp_path/'geometry-controls').mkdir()
    with pytest.raises(ValueError, match='Fresh'):
        controls.geometry_controls(tmp_path/'new', tmp_path, lambda: 20, official_helper=tmp_path/'helper')


def test_comparator_technical_unavailable_stops_not_reclassified_geometry_failure(tmp_path, monkeypatch):
    new, helper = setup_orchestration(tmp_path, monkeypatch)
    seen = []
    def unavailable(*args):
        seen.append(args[3])
        raise PermissionError('Cannot execute comparator')
    monkeypatch.setattr(controls, 'branch', unavailable)
    with pytest.raises(controls.GeometryControlError) as exc:
        controls.geometry_controls(new, tmp_path, lambda: 20, official_helper=helper)
    assert seen == ['original']
    assert exc.value.report['paired_fixtures'][0]['comparisons'][0]['failure_scope'] == 'technical_unavailable'
    assert exc.value.report['status'] == 'fail'


@pytest.mark.parametrize('timeout', [False, True])
def test_failed_native_retains_input_hash_count_elapsed_and_bounded_stderr(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(controls.geometry, 'write_obj', lambda path, *mesh: path.write_bytes(b'own input'))
    def child(*args, **kwargs):
        if timeout:
            raise controls.subprocess.TimeoutExpired(args[0], 900)
        return SimpleNamespace(returncode=3, stderr=b'e'*700)
    monkeypatch.setattr(controls.subprocess, 'run', child)
    record = {}
    with pytest.raises((ValueError, controls.subprocess.TimeoutExpired)):
        controls.branch((np.zeros((4,3)), np.array([[0,1,2]])), False, tmp_path/'binary',
                        'original', tmp_path, None, lambda: 900, record)
    assert record['native_attempts'] == 1 and record['native_input']['bytes'] == 9
    assert record['native_elapsed_seconds'] >= 0
    if timeout:
        assert record['native_returned'] is False
    else:
        assert record['native_returned'] is True and record['native_exit_code'] == 3
        assert len(record['native_stderr_tail']) == 500

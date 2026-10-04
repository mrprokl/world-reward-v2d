"""Manufactured boxes, mocked native callback; no production mesh or native run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'infra'))
    spec = importlib.util.spec_from_file_location('wr_source_certificate', ROOT / 'infra/certified_solid_source.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tetra(offset=(0, 0, 0)):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) + offset
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return vertices, faces


def fixture(source, monkeypatch, tmp_path, *, pair=False):
    mesh = tmp_path / 'input.glb'; mesh.write_bytes(b'own mocked GLB, no media')
    binary = tmp_path / 'binary'; binary.write_bytes(b'own mock binary, not executable')
    v, f = tetra()
    if pair:
        other, of = tetra((4, 0, 0)); v = np.vstack([v, other]); f = np.vstack([of + 4, f])
    monkeypatch.setattr(source.precision, 'raw_glb', lambda _: ([(v.copy(), f.copy())], v[f].copy(),
                        [{'position_dtype': 'float32', 'vertices': len(v), 'faces': len(f)}]))
    monkeypatch.setattr(source.endpoint, '_load_mesh', lambda _: (v.copy(), f.copy()))
    sha = hashlib.sha256((ROOT / 'infra/certified_solid_query.cpp').read_bytes()).hexdigest()
    calls = []
    def native(argv, *, input, capture_output, check, timeout):
        calls.append((argv, input, timeout))
        assert capture_output and check is False and 0 < timeout <= 180
        return SimpleNamespace(returncode=0, stdout=json.dumps(native_report(source, input, sha)).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', native)
    return mesh, binary, sha, calls, v, f


def parsed(raw):
    tokens = raw.decode().split(); countv, countf, countc = map(int, tokens[1:4])
    v = np.array(tokens[4:4 + 3 * countv], np.float64).reshape(countv, 3)
    f = np.array(tokens[4 + 3 * countv:], np.int64).reshape(countf, 4)
    return v, f[:, :3], f[:, 3], countc


def native_report(source, raw, sha):
    v, f, labels, count = parsed(raw)
    components = []
    for i in range(count):
        ids = np.unique(f[labels == i])
        components.append(dict(original_component_id=i, original_vertices=len(ids),
                               original_faces=int(np.count_nonzero(labels == i)), witness_original_vertex=int(ids[0]),
                               exact_volume_sign=1, diagnostic_signed_volume=1 / 6))
    return dict(schema='world_reward.certified_solid_query.v1', status='pass', source_sha256=sha, cgal_version='6.0.1',
                vertices=len(v), faces=len(f), component_count=count, components=components,
                inside=np.zeros((count, count), dtype=bool).tolist(),
                represented_coordinates='EPECK exact values of original parsed IEEE754 binary64; no perturbation',
                **{k: True for k in source.TRUE_FLAGS}, **{k: False for k in source.FALSE_FLAGS})


def test_readonly_fullsource_certificate_single_native_query(source, monkeypatch, tmp_path):
    mesh, binary, sha, calls, v, f = fixture(source, monkeypatch, tmp_path)
    before = mesh.read_bytes(), binary.read_bytes(), v.tobytes(), f.tobytes()
    result = source.certify_source(mesh, binary, sha)
    assert result['status'] == 'pass' and result['phase'] == 'complete'
    assert result['native_attempts'] == 1 and result['native_returned'] and len(calls) == 1
    assert result['component_keys'] == ('source-loaded-first-face-0',)
    assert result['forest']['parents'] == [-1] and result['forest']['signs'] == [1]
    assert result['source_binary_helpers_rehashed_after'] and result['artifacts_before'] == result['artifacts_after']
    assert result['source_orphans_removed'] is False and result['exact_welding']['faces_removed'] == 0
    assert before == (mesh.read_bytes(), binary.read_bytes(), v.tobytes(), f.tobytes())
    assert all(result[k] is False for k in ('adoption', 'geometry_repaired', 'qem_executed', 'packing_validated',
                                          'reconstruction_accuracy_verified', 'physical_scale_accuracy_verified', 'ground_truth_used'))


def test_components_first_face_order_not_vertex_or_volume_sort(source, monkeypatch, tmp_path):
    mesh, binary, sha, calls, v, f = fixture(source, monkeypatch, tmp_path, pair=True)
    result = source.certify_source(mesh, binary, sha)
    assert result['component_keys'] == ('source-loaded-first-face-0', 'source-loaded-first-face-4')
    encoded_v, encoded_f, labels, count = parsed(calls[0][1])
    assert count == 2 and labels.tolist() == [0] * 4 + [1] * 4
    # The first component is lexically later in coordinates; original face order wins.
    assert encoded_v[encoded_f[0]].min(axis=0)[0] == 4
    np.testing.assert_array_equal(encoded_v[encoded_f], v[f])
    assert result['forest']['parents'] == [-1, -1]


def test_exact_seam_weld_preserves_every_oriented_face(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, v, f = fixture(source, monkeypatch, tmp_path)
    seam = v[f].reshape(-1, 3); sf = np.arange(len(seam), dtype=np.int64).reshape(-1, 3)
    monkeypatch.setattr(source.endpoint, '_load_mesh', lambda _: (seam.copy(), sf.copy()))
    monkeypatch.setattr(source.precision, 'raw_glb', lambda _: ([(seam.copy(), sf.copy())], seam[sf].copy(), []))
    report = source.certify_source(mesh, binary, sha)
    assert report['exact_welding']['raw_vertices'] == 12 and report['vertices'] == 4
    assert report['faces'] == 4 and report['source_loaded_face_order_preserved']


def test_raw_triangle_mismatch_rejected_before_native(source, monkeypatch, tmp_path):
    mesh, binary, sha, calls, v, f = fixture(source, monkeypatch, tmp_path)
    monkeypatch.setattr(source.precision, 'raw_glb', lambda _: ([(v.copy(), f.copy())], v[f[:, ::-1]], []))
    with pytest.raises(source.SourceCertificationError) as error:
        source.certify_source(mesh, binary, sha)
    assert not calls and error.value.report['native_attempts'] == 0
    assert error.value.report['source_binary_helpers_rehashed_after']


@pytest.mark.parametrize('which', ['loaded', 'accessor'])
def test_orphans_never_removed_or_sent_native(source, monkeypatch, tmp_path, which):
    mesh, binary, sha, calls, v, f = fixture(source, monkeypatch, tmp_path)
    orphan = np.vstack([v, [5., 6., 7.]])
    if which == 'loaded': monkeypatch.setattr(source.endpoint, '_load_mesh', lambda _: (orphan.copy(), f.copy()))
    else: monkeypatch.setattr(source.precision, 'raw_glb', lambda _: ([(orphan.copy(), f.copy())], v[f], []))
    with pytest.raises(source.SourceCertificationError): source.certify_source(mesh, binary, sha)
    assert not calls


@pytest.mark.parametrize('mutation', ['sign_bool', 'sign_zero', 'inside_int', 'inside_partial', 'wrong_source',
                                    'face_bool', 'witness', 'falseflag', 'extra', 'nan'])
def test_full_native_abi_and_unknown_signs_are_strict(source, monkeypatch, tmp_path, mutation):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    def native(_argv, **kwargs):
        r = native_report(source, kwargs['input'], sha)
        if mutation == 'sign_bool': r['components'][0]['exact_volume_sign'] = True
        if mutation == 'sign_zero': r['components'][0]['exact_volume_sign'] = 0
        if mutation == 'inside_int': r['inside'][0][0] = 0
        if mutation == 'inside_partial': r['inside'] = []
        if mutation == 'wrong_source': r['source_sha256'] = 'b' * 64
        if mutation == 'face_bool': r['faces'] = True
        if mutation == 'witness': r['components'][0]['witness_original_vertex'] += 1
        if mutation == 'falseflag': r['geometry_repaired'] = True
        if mutation == 'extra': r['secret'] = 1
        if mutation == 'nan': r['components'][0]['diagnostic_signed_volume'] = float('nan')
        return SimpleNamespace(returncode=0, stdout=json.dumps(r).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', native)
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    assert error.value.report['status'] == 'fail' and error.value.report['native_attempts'] == 1
    assert error.value.report['source_binary_helpers_rehashed_after']


def test_native_signs_drive_forest_not_diagnostic_float_or_expected_sign(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    def native(_argv, **kwargs):
        r = native_report(source, kwargs['input'], sha)
        r['components'][0].update(exact_volume_sign=-1, diagnostic_signed_volume=100.0)
        return SimpleNamespace(returncode=0, stdout=json.dumps(r).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', native)
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    r = error.value.report
    assert r['failure_scope'] == 'material_forest_rejection' and r['native_certificate']['components'][0]['exact_volume_sign'] == -1


def test_native_geometric_rejection_preserved_no_partial_pass(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    monkeypatch.setattr(source.subprocess, 'run', lambda *_a, **_k: SimpleNamespace(
        returncode=1, stdout=b'', stderr=b'certified_solid_query FAIL: Exact component self-intersection\n'))
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    r = error.value.report
    assert r['failure_scope'] == 'native_geometric_rejection' and r['native_returned']
    assert 'self-intersection' in r['native_rejection'] and len(r['native_rejection']) <= 300
    assert 'native_certificate' not in r


def test_native_timeout_is_not_geometric_failure(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    def timeout(*_a, **_k): raise subprocess.TimeoutExpired('mockonly', 180)
    monkeypatch.setattr(source.subprocess, 'run', timeout)
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    assert error.value.report['failure_scope'] == 'deadline' and not error.value.report['native_returned']
    assert error.value.report['source_binary_helpers_rehashed_after']


def test_posthash_detects_mutation_and_invalidates_prior_pass(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    def mutate(_argv, **kwargs):
        r = native_report(source, kwargs['input'], sha); mesh.write_bytes(b'changed')
        return SimpleNamespace(returncode=0, stdout=json.dumps(r).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', mutate)
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    assert error.value.report['failure_scope'] == 'posthash_integrity' and error.value.report['status'] == 'fail'


def test_hardlink_and_source_sha_preflight_no_calls(source, monkeypatch, tmp_path):
    mesh, binary, sha, calls, _, _ = fixture(source, monkeypatch, tmp_path)
    with pytest.raises(source.SourceCertificationError): source.certify_source(mesh, binary, 'f' * 64)
    assert not calls
    import os
    os.link(mesh, tmp_path / 'alias')
    with pytest.raises(source.SourceCertificationError): source.certify_source(mesh, binary, sha)
    assert not calls


def test_ascii_17digits_preserves_units_and_signedzero(source):
    v, f = tetra(); v[0, 0] = -0.0; v[1, 0] = np.nextafter(1., 2.)
    raw = source.input_ascii(v, f, np.zeros(len(f), dtype=np.int64), 1)
    parsed_v, parsed_f, _, _ = parsed(raw)
    assert parsed_v.tobytes() == v.tobytes()
    np.testing.assert_array_equal(parsed_f, f)
    assert raw.startswith(b'WR_SOLID_QUERY_V1 4 4 1\n')


def test_inclusive_deadline_after_native_invalidates_candidate_certificate(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(source.time, 'monotonic', lambda: clock[0])
    def native(_argv, **kwargs):
        r = native_report(source, kwargs['input'], sha); clock[0] = 301.0
        return SimpleNamespace(returncode=0, stdout=json.dumps(r).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', native)
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    assert error.value.report['failure_scope'] == 'deadline' and error.value.report['elapsed_seconds'] == 301
    assert error.value.report['source_binary_helpers_rehashed_after']


def test_nullable_volume_diagnostic_does_not_weaken_exact_sign_contract(source, monkeypatch, tmp_path):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    def native(_argv, **kwargs):
        r = native_report(source, kwargs['input'], sha); r['components'][0]['diagnostic_signed_volume'] = None
        return SimpleNamespace(returncode=0, stdout=json.dumps(r).encode(), stderr=b'')
    monkeypatch.setattr(source.subprocess, 'run', native)
    assert source.certify_source(mesh, binary, sha)['status'] == 'pass'


def test_scalar_json_overflow_duplicates_and_nonfinite_fail_closed(source):
    for raw in (b'{"x":1e400}', b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError): source.strict_json(raw)


@pytest.mark.parametrize('reason', ['std::bad_alloc', 'unknown exception', 'Original component construction differs',
                                  'Wrong input header'])
def test_native_execution_failures_never_imply_bad_geometry(source, monkeypatch, tmp_path, reason):
    mesh, binary, sha, _, _, _ = fixture(source, monkeypatch, tmp_path)
    monkeypatch.setattr(source.subprocess, 'run', lambda *_a, **_k: SimpleNamespace(
        returncode=1, stdout=b'', stderr=('certified_solid_query FAIL: ' + reason + '\n').encode()))
    with pytest.raises(source.SourceCertificationError) as error: source.certify_source(mesh, binary, sha)
    r = error.value.report
    assert r['failure_scope'] == 'native_execution_contract' and r['native_rejection'] == reason
    assert r['native_returned'] and r['source_binary_helpers_rehashed_after']
    assert 'represented_embedding_certified' not in r


def test_raw_multiset_parity_does_not_claim_original_glb_face_order(source, monkeypatch, tmp_path):
    mesh, binary, sha, calls, v, f = fixture(source, monkeypatch, tmp_path, pair=True)
    # Raw primitive ordering may differ from Trimesh's unprocessed concatenation.
    monkeypatch.setattr(source.precision, 'raw_glb', lambda _: ([(v.copy(), f.copy())], v[f][::-1].copy(), []))
    r = source.certify_source(mesh, binary, sha)
    assert r['status'] == 'pass' and r['order_scope'] == source.ORDER_SCOPE
    assert r['component_keys'] == ('source-loaded-first-face-0', 'source-loaded-first-face-4')
    assert r['source_loaded_face_order_preserved'] and 'original_face_order_preserved' not in r
    encoded_v, encoded_f, _, _ = parsed(calls[0][1])
    np.testing.assert_array_equal(encoded_v[encoded_f], v[f])

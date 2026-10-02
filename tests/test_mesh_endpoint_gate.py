"""Tiny CPU contracts/stubs; no PyMeshLab, challenge inputs or heavy fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / 'infra'
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location('wr_test_endpoint', infra / 'mesh_endpoint_gate.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tetra(scale=1., offset=(0., 0., 0.)):
    v = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]) * scale + offset
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return v, f


def hollow(inner_scale=.3, offset=(0., 0., 0.)):
    v, f = tetra(); inner, _ = tetra(inner_scale, offset)
    return np.concatenate([v, inner]), np.concatenate([f, f[:, ::-1] + 4])


def test_new_algorithm_same_budgets_no_old_mutation(gate):
    assert gate.ENDPOINT_PARAMETERS == gate.PARAMETERS | {'optimalplacement': False}
    assert gate.PARAMETERS['optimalplacement'] is True
    assert gate.ENDPOINT_PARAMETERS['autoclean'] is False and gate.ENDPOINT_PARAMETERS['preservetopology'] is True
    assert (gate.FACE_BUDGET, gate.VERTEX_BUDGET, gate.MAX_SECONDS) == (4096, 4096, 120.)
    assert (gate.MAX_CHAMFER_DIAGONAL_RATIO, gate.MAX_VOLUME_RELATIVE_ERROR, gate.WINDING_ATOL) == (.01, .05, 1e-6)


def test_winding_inside_outside_boundary_exact_triangle_distance(gate):
    v, f = tetra()
    p = np.array([[0., 0., 0.], [4., 0., 0.], v[0]])
    w, d = gate.solid_winding_and_distances(p, v[f])
    assert w[:2] == pytest.approx([1., 0.], abs=1e-12)
    assert d[0] == pytest.approx(1 / np.sqrt(3))
    assert d[1] > 0 and d[2] == 0.
    reversed_w, _ = gate.solid_winding_and_distances(p[:1], v[f[:, ::-1]])
    assert reversed_w[0] == pytest.approx(-1.)
    shift = np.array([10., -4., 3.])
    wt, dt = gate.solid_winding_and_distances(p[:2] + shift, v[f] + shift)
    assert wt == pytest.approx(w[:2], abs=1e-12) and dt == pytest.approx(d[:2], abs=1e-12)


@pytest.mark.parametrize('kind', ['masked', 'nan', 'empty', 'collapsed'])
def test_winding_invalid_arrays_fail(gate, kind):
    v, f = tetra(); p, triangles = v[:1].copy(), v[f].copy()
    if kind == 'masked': p = np.ma.array(p, mask=False)
    if kind == 'nan': p[0, 0] = np.nan
    if kind == 'empty': p = p[:0]
    if kind == 'collapsed': triangles[0, 1] = triangles[0, 0]
    with pytest.raises(ValueError): gate.solid_winding_and_distances(p, triangles)


def test_outer_only_true_containment_distance_is_not_global_separation(gate):
    v, f = hollow(); originals = v.copy(), f.copy()
    report = gate.true_hollow_containment(v, f, self_intersecting_faces=0)
    assert report['true_containment_verified'] is True and report['winding_max_error'] < 1e-12
    assert report['min_inner_vertex_to_outer_surface_distance'] > report['boundary_margin']
    assert report['min_surface_shell_separation'] is None
    assert report['continuous_surface_separation_computed'] is False
    assert np.array_equal(v, originals[0]) and np.array_equal(f, originals[1])


@pytest.mark.parametrize('kind', ['outside', 'crossed', 'boundary'])
def test_escape_crossing_or_boundary_cannot_pass_containment(gate, kind):
    if kind == 'outside': source = hollow(offset=(4., 0., 0.)); intersections = 0
    elif kind == 'crossed': source = hollow(offset=(.8, 0., 0.)); intersections = 2
    else: source = hollow(inner_scale=1.); intersections = 0
    with pytest.raises(ValueError): gate.true_hollow_containment(*source, self_intersecting_faces=intersections)


def test_pinched_vertex_link_rejected_before_containment(gate):
    v, f = tetra(); second, g = tetra(offset=(0., 0., 2.)); second[0] = v[0]
    vv = np.concatenate([v, second[1:]]); ff = np.concatenate([f, np.where(g == 0, 0, g + 3)])
    with pytest.raises(ValueError, match='Disconnected vertex link'):
        gate.mesh_topology(vv, ff)


def test_containment_ambiguity_not_silently_repaired(gate, monkeypatch):
    monkeypatch.setattr(gate, 'solid_winding_and_distances', lambda p, t: (np.ones(len(p)) + 2e-6, np.ones(len(p))))
    report = {}
    with pytest.raises(ValueError, match='ambiguous'):
        gate.true_hollow_containment(*hollow(), self_intersecting_faces=0, diagnostics=report)
    assert report['true_containment_verified'] is False and report['winding_max_error'] > 1e-6


def test_endpoint_wrapper_parameters_and_selection_only(gate, monkeypatch):
    v, f = tetra(); calls = []
    mesh = SimpleNamespace(compact=lambda: calls.append('compact'), vertex_matrix=lambda: v,
                           face_matrix=lambda: f, selected_face_number=lambda: 0)
    class MeshSet:
        def add_mesh(self, value, name): calls.append(('add', value, name))
        def meshing_decimation_quadric_edge_collapse(self, **kw): calls.append(kw)
        def current_mesh(self): return mesh
        def compute_selection_by_self_intersections_per_face(self): calls.append('intersection')
    monkeypatch.setitem(sys.modules, 'pymeshlab', SimpleNamespace(MeshSet=MeshSet, Mesh=lambda **kw: kw))
    monkeypatch.setattr(gate.importlib.metadata, 'version', lambda _: gate.VERSION)
    before = copy.deepcopy(gate.PARAMETERS)
    result, count = gate.endpoint_simplify(v, f)
    assert calls[1] == {'targetfacenum': 4096, **gate.PARAMETERS, 'optimalplacement': False}
    assert calls[-2:] == ['compact', 'intersection'] and count == 0
    assert gate.PARAMETERS == before and not np.shares_memory(result[0], v)
    calls.clear(); assert gate.source_intersections(v, f) == 0
    assert not any(isinstance(x, dict) and 'targetfacenum' in x for x in calls)


def stub_gate(gate, monkeypatch, *, intersections=0, changed=False, budget=False):
    fixtures = [('new_star', hollow(), True), ('new_torus', hollow(), True),
                ('new_ellipsoid', hollow(), True), ('new_disconnected', tetra(), False)]
    monkeypatch.setattr(gate, 'new_fixtures', lambda: fixtures)
    monkeypatch.setattr(gate, 'source_intersections', lambda *_: 0)
    calls = []
    def simplify(v, f):
        calls.append(1); vv, ff = v.copy(), f.copy()
        if changed and len(calls) == 2: vv[0, 0] += 1e-6
        if budget: vv = np.concatenate([vv, np.zeros((4096, 3))])
        return (vv, ff), intersections
    monkeypatch.setattr(gate, 'endpoint_simplify', simplify)
    return {'status': 'running', 'fixtures': []}, calls


def test_new_cohort_repeats_and_persists_no_adoption(gate, monkeypatch, tmp_path):
    report, calls = stub_gate(gate, monkeypatch)
    path = tmp_path / 'r.json'; gate.run_gate(report, path)
    assert report['status'] == 'pass' and len(calls) == 8
    assert all(row['two_runs_arrays_identical'] and row['source_arrays_unchanged'] for row in report['fixtures'])
    assert all(row['source_convexity_diagnostic'][0]['convexity_used_as_gate'] is False for row in report['fixtures'])
    assert json.loads(path.read_text())['status'] == 'pass'


@pytest.mark.parametrize('kind,match', [('intersections', 'self-intersections'), ('changed', 'deterministic'), ('budget', '4096')])
def test_gate_failure_never_runs_remaining_cohort(gate, monkeypatch, tmp_path, kind, match):
    report, calls = stub_gate(gate, monkeypatch, **{kind: 2 if kind == 'intersections' else True})
    path = tmp_path / 'r.json'
    with pytest.raises(ValueError, match=match): gate.run_gate(report, path)
    assert len(calls) <= 2 and report['fixtures'][0]['status'] == 'fail'
    assert json.loads(path.read_text())['fixtures'][0]['runs']


def test_parent_hard_timeout_and_frozen_worker_guards(gate, monkeypatch, tmp_path):
    (tmp_path / 'results').mkdir(); image = 'sha256:'+'a'*64
    build = {'stage': 'world_reward_topology_cpu_build', 'status': 'pass', 'pymeshlab_version': gate.VERSION,
             'wheel_sha256': gate.WHEEL_SHA, 'image_id': image, 'base_image_id': 'sha256:'+'b'*64}
    (tmp_path / 'results/image-topology-cpu.json').write_text(json.dumps(build))
    monkeypatch.setenv('WR_ROOT', str(tmp_path)); monkeypatch.setenv('WR_TOPOLOGY_IMAGE_ID', image)
    monkeypatch.setenv('WR_CODE_REVISION', 'c'*40)
    monkeypatch.setattr(gate.platform, 'system', lambda: 'Linux')
    original = Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda p: iter([Path('lo')]) if str(p) == '/sys/class/net' else original(p))
    monkeypatch.setattr(sys, 'argv', [gate.__file__])
    path = tmp_path / 'results/mesh-endpoint.json'
    def run(args, *, timeout, check, env):
        assert timeout == 120. and len(env['WR_ENDPOINT_WORKER_NONCE']) == 64
        r = json.loads(path.read_text()); r['fixtures'] = [{'fixture': 'partial'}]; path.write_text(json.dumps(r))
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(gate.subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired): gate.main()
    r = json.loads(path.read_text())
    assert r['status'] == 'fail' and r['old_cohort_rerun'] is False and r['adoption_performed'] is False
    assert r['fixtures'] == [{'fixture': 'partial'}]
    with pytest.raises(RuntimeError, match='Frozen'): gate.main()
    before = path.read_bytes(); monkeypatch.setattr(sys, 'argv', [gate.__file__, '--worker'])
    with pytest.raises(RuntimeError, match='never overwrite'): gate.main()
    assert path.read_bytes() == before


def test_resource_wrapper_no_gpu_or_challenge_mount():
    p = Path(__file__).resolve().parents[1] / 'infra/run_mesh_endpoint_gate.sh'
    s = p.read_text()
    assert '--cpus 4 --memory 4g --network none' in s and '--gpus' not in s
    assert 'mesh_endpoint_gate.py' in s and 'readonly' in s and 'vendor' not in s


def test_true_containment_passes_nonconvex_torus_without_convexity_gate(gate):
    u, w = np.meshgrid(np.arange(12) * 2*np.pi/12, np.arange(8) * 2*np.pi/8, indexing='ij')
    def vertices(tube):
        return np.stack([(1+tube*np.cos(w))*np.cos(u), (1+tube*np.cos(w))*np.sin(u), tube*np.sin(w)], axis=-1).reshape(-1, 3)
    f = []
    for i in range(12):
        for j in range(8):
            a, b = i*8+j, ((i+1)%12)*8+j
            c, d = ((i+1)%12)*8+(j+1)%8, i*8+(j+1)%8
            f.extend(((a,b,c),(a,c,d)))
    f = np.array(f)
    v, faces = gate._combine(vertices(.3), f, vertices(.12), f)
    topology = gate.mesh_topology(v, faces)
    assert [(x['euler'], x['volume_sign']) for x in topology['components']] == [(0,-1),(0,1)]
    assert gate.true_hollow_containment(v, faces, self_intersecting_faces=0)['true_containment_verified'] is True
    diagnostic = gate.convexity_diagnostic(v, faces)
    assert diagnostic[0]['sampled_support_plane_max_violation'] > .01
    assert diagnostic[0]['convexity_used_as_gate'] is False


@pytest.mark.parametrize('bad', ['script', 'shared', 'nonce'])
def test_endpoint_shared_gate_sha_and_nonce_fail_closed(gate, monkeypatch, tmp_path, bad):
    (tmp_path/'results').mkdir()
    path = tmp_path/'results/mesh-endpoint.json'
    r = {'status': 'running', 'stage': gate.STAGE, 'script_sha256': gate.sha256(Path(gate.__file__)),
         'shared_gate_sha256': gate.sha256(Path(gate.__file__).with_name('mesh_link_gate.py')),
         'producer_revision': 'c'*40, 'worker_nonce_sha256': gate.hashlib.sha256(b'nonce').hexdigest()}
    r[{'script': 'script_sha256', 'shared': 'shared_gate_sha256', 'nonce': 'worker_nonce_sha256'}[bad]] = 'wrong'
    path.write_text(json.dumps(r)); before=path.read_bytes()
    monkeypatch.setenv('WR_ROOT', str(tmp_path)); monkeypatch.setenv('WR_CODE_REVISION', 'c'*40)
    monkeypatch.setenv('WR_ENDPOINT_WORKER_NONCE', 'nonce')
    monkeypatch.setattr(gate.platform, 'system', lambda:'Linux')
    original=Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda p: iter([Path('lo')]) if str(p)=='/sys/class/net' else original(p))
    monkeypatch.setattr(sys,'argv',[gate.__file__,'--worker'])
    with pytest.raises(RuntimeError, match='current'): gate.main()
    assert path.read_bytes()==before

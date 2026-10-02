"""Tiny procedural/stub checks; no PyMeshLab installation, GPU or real meshes."""
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
def gate():
    path = Path(__file__).resolve().parents[1] / 'infra/mesh_link_gate.py'
    spec = importlib.util.spec_from_file_location('test_mesh_link_gate_module', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tetra(radius=1., offset=(0., 0., 0.)):
    v = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]) * radius
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return v + offset, f[:, ::-1].copy()


def small_torus():
    u, w = np.meshgrid(np.arange(8) * 2 * np.pi / 8, np.arange(6) * 2 * np.pi / 6, indexing='ij')
    v = np.stack([(1. + .3 * np.cos(w)) * np.cos(u), (1. + .3 * np.cos(w)) * np.sin(u), .3 * np.sin(w)], axis=-1).reshape(-1, 3)
    f = []
    for i in range(8):
        for j in range(6):
            a, b = i * 6 + j, ((i + 1) % 8) * 6 + j
            c, d = ((i + 1) % 8) * 6 + (j + 1) % 6, i * 6 + (j + 1) % 6
            f.extend(((a, b, c), (a, c, d)))
    return v, np.array(f)


def hollow():
    v, f = tetra()
    return np.concatenate([v, v * .5]), np.concatenate([f, f[:, ::-1] + 4])


def test_predeclared_fixed_protocol(gate):
    assert gate.FACE_BUDGET == gate.VERTEX_BUDGET == 4096
    assert (gate.MAX_SECONDS, gate.MAX_CHAMFER_DIAGONAL_RATIO, gate.MAX_VOLUME_RELATIVE_ERROR) == (120., .01, .05)
    assert gate.SURFACE_SAMPLES == 8192
    assert gate.PARAMETERS['preservetopology'] is True and gate.PARAMETERS['preservenormal'] is True
    assert gate.PARAMETERS['autoclean'] is False and gate.PARAMETERS['targetperc'] == 0.
    assert gate.VERSION == '2025.7.post1'


def test_closed_sphere_torus_cavity_signatures_no_mutation(gate):
    for source, expected in [(tetra(), [(2, 1)]), (small_torus(), [(0, 1)]), (hollow(), [(2, -1), (2, 1)])]:
        before = [x.copy() for x in source]
        topology = gate.mesh_topology(*source)
        assert [(x['euler'], x['volume_sign']) for x in topology['components']] == expected
        assert topology['closed_oriented_vertex_manifold'] is True
        assert all(np.array_equal(a, b) for a, b in zip(source, before))
    gate.hollow_containment(*hollow())


@pytest.mark.parametrize('bad', ['nan', 'masked', 'floatfaces', 'invalid', 'open', 'reverseone', 'zeroarea', 'negative'])
def test_invalid_meshes_fail_without_repair(gate, bad):
    v, f = tetra()
    if bad == 'nan': v[0, 0] = np.nan
    if bad == 'masked': v = np.ma.array(v, mask=False)
    if bad == 'floatfaces': f = f.astype(float)
    if bad == 'invalid': f[0, 0] = 20
    if bad == 'open': f = f[:-1]
    if bad == 'reverseone': f[0] = f[0, ::-1]
    if bad == 'zeroarea': v[1] = v[0]
    if bad == 'negative': f = f[:, ::-1]
    with pytest.raises(ValueError): gate.mesh_topology(v, f)


def test_closed_edge_manifold_but_bowtie_vertex_fails(gate):
    v, f = tetra()
    other, g = tetra(offset=(0., 0., 2.))
    other[0] = v[0]
    vv = np.concatenate([v, other[1:]])
    gg = np.where(g == 0, 0, g + 3)
    with pytest.raises(ValueError, match='Disconnected vertex link'):
        gate.mesh_topology(vv, np.concatenate([f, gg]))


def test_synthetic_geometry_limits_volume_and_topology_not_tuned(gate, monkeypatch):
    source = tetra(); top = gate.mesh_topology(*source)
    assert gate.geometry_gates(source, source, top, top)['sampled_bidirectional_chamfer_diagonal_ratio'] == 0.
    enlarged = (source[0] * 1.1, source[1])
    monkeypatch.setattr(gate, '_surface', lambda *_: np.zeros((8, 3)))
    with pytest.raises(ValueError, match='5%'):
        gate.geometry_gates(source, enlarged, top, gate.mesh_topology(*enlarged))
    bad = copy.deepcopy(top); bad['components'][0]['euler'] = 0
    with pytest.raises(ValueError, match='Euler/sign'):
        gate.geometry_gates(source, source, top, bad)
    counter = iter([np.zeros((8, 3)), np.ones((8, 3))])
    monkeypatch.setattr(gate, '_surface', lambda *_: next(counter))
    with pytest.raises(ValueError, match='1%'):
        gate.geometry_gates(source, source, top, top)


def test_hollow_containment_is_fixture_only_and_cannot_pass_crossed_shell(gate):
    v, f = hollow(); v[4:] += 2
    with pytest.raises(ValueError, match='containment'): gate.hollow_containment(v, f)


def test_qem_exact_flags_compaction_selfintersection_no_cleanup(gate, monkeypatch):
    calls = []; v, f = tetra()
    mesh = SimpleNamespace(compact=lambda: calls.append('compact'), vertex_matrix=lambda: v,
                           face_matrix=lambda: f, selected_face_number=lambda: 0)
    class MeshSet:
        def add_mesh(self, value, name): calls.append(('add', value, name))
        def meshing_decimation_quadric_edge_collapse(self, **parameters): calls.append(parameters)
        def current_mesh(self): return mesh
        def compute_selection_by_self_intersections_per_face(self): calls.append('selfintersection')
    fake = SimpleNamespace(MeshSet=MeshSet, Mesh=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, 'pymeshlab', fake)
    monkeypatch.setattr(gate.importlib.metadata, 'version', lambda name: gate.VERSION)
    result, intersections = gate.simplify(v, f)
    assert intersections == 0
    assert calls[1] == {'targetfacenum': 4096, **gate.PARAMETERS}
    assert calls[-2:] == ['compact', 'selfintersection']
    assert calls[0][1]['face_matrix'].dtype == np.int32
    assert not np.shares_memory(result[0], v) and not np.shares_memory(result[1], f)
    monkeypatch.setattr(gate.importlib.metadata, 'version', lambda name: 'wrong')
    with pytest.raises(RuntimeError, match='version'): gate.simplify(v, f)


def tiny_run(gate, monkeypatch, *, intersections=0, nondeterministic=False, overbudget=False):
    sources = [('sphere', tetra()), ('torus', small_torus()), ('thin_hollow', hollow()), ('disconnected_positive', tetra())]
    monkeypatch.setattr(gate, 'procedural_fixtures', lambda: sources)
    calls = []
    def simplify(v, f, face_budget=4096):
        calls.append(face_budget); result = (v.copy(), f.copy())
        if nondeterministic and len(calls) == 2: result[0][0, 0] += 1e-5
        if overbudget: result = (np.concatenate([v, np.zeros((4096, 3))]), f.copy())
        return result, intersections
    monkeypatch.setattr(gate, 'simplify', simplify)
    report = {'status': 'running', 'fixtures': []}
    return report, calls


def test_worker_two_runs_and_expected_impossible_torus_budget(gate, monkeypatch, tmp_path):
    report, calls = tiny_run(gate, monkeypatch)
    path = tmp_path / 'report.json'; gate.run_gate(report, path)
    assert calls == [4096] * 8 + [6]
    assert report['status'] == 'pass'
    assert all(row['two_runs_arrays_identical'] and row['source_arrays_unchanged'] for row in report['fixtures'])
    assert report['torus6_expected_budget_failure']['retained_faces'] > 6
    assert json.loads(path.read_text())['status'] == 'pass'


@pytest.mark.parametrize('mode,match', [('intersections', 'self-intersections'), ('nondeterministic', 'deterministic'), ('overbudget', '4096')])
def test_worker_failure_preserves_partial_evidence(gate, monkeypatch, tmp_path, mode, match):
    report, calls = tiny_run(gate, monkeypatch, **{mode: 1 if mode == 'intersections' else True})
    path = tmp_path / 'report.json'
    with pytest.raises(ValueError, match=match): gate.run_gate(report, path)
    assert report['fixtures'][0]['status'] == 'fail'
    assert json.loads(path.read_text())['fixtures'][0]['runs']
    assert len(calls) <= 2


def test_worker120seconds_not_relaxed(gate, monkeypatch, tmp_path):
    report, _ = tiny_run(gate, monkeypatch)
    times = iter([0., 120.])
    monkeypatch.setattr(gate.time, 'perf_counter', lambda: next(times))
    with pytest.raises(TimeoutError, match='120'): gate.run_gate(report, tmp_path / 'r.json')


def test_build_wrapper_immutable_id_cpu_nooverwrite(tmp_path):
    infra = Path(__file__).resolve().parents[1] / 'infra'
    (tmp_path / 'results').mkdir(); (tmp_path / 'bin').mkdir()
    log = tmp_path / 'calls'
    fake = tmp_path / 'bin/docker'
    fake.write_text('#!/bin/bash\necho "$*" >> "$LOG"\nif [[ "$*" == *"image inspect world-reward/cari4d-source"* || "$*" == *"image inspect world-reward/topology-base"* ]]; then echo "sha256:' + 'a'*64 + '"; elif [[ "$*" == *"image inspect world-reward/topology-cpu"* ]]; then echo "sha256:' + 'b'*64 + '"; fi\n')
    fake.chmod(0o755)
    env = os.environ | {'PATH': str(tmp_path / 'bin') + ':' + os.environ['PATH'], 'LOG': str(log),
                        'WR_ROOT': str(tmp_path), 'WR_CODE': str(infra.parent), 'WR_CODE_REVISION': 'c'*40}
    subprocess.run(['bash', str(infra / 'run_topology_cpu_build.sh')], env=env, check=True)
    r = json.loads((tmp_path / 'results/image-topology-cpu.json').read_text())
    assert r['base_image_id'] == 'sha256:'+'a'*64 and r['image_id'] == 'sha256:'+'b'*64
    assert '--build-arg BASE_IMAGE=world-reward/topology-base:'+'a'*64 in log.read_text()
    assert '--pull=false' in log.read_text()
    assert 'image tag sha256:'+'a'*64+' world-reward/topology-base:'+'a'*64 in log.read_text()
    assert log.read_text().count('image inspect world-reward/topology-base:') == 2
    assert r['base_image_reference'] == 'world-reward/topology-base:'+'a'*64
    assert '--gpus' not in log.read_text()
    previous = log.read_text()
    assert subprocess.run(['bash', str(infra / 'run_topology_cpu_build.sh')], env=env, capture_output=True).returncode == 2
    assert log.read_text() == previous


def test_dockerfile_pinned_wheel_no_deps_no_apt_no_gpu():
    p = Path(__file__).resolve().parents[1] / 'infra/Dockerfile.topology_cpu'
    s = p.read_text()
    assert 'ARG BASE_IMAGE\nFROM ${BASE_IMAGE}' in s
    assert 'cp311-cp311-manylinux_2_35_x86_64.whl' in s
    assert 'c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30' in s
    assert '--no-deps --no-index' in s and 'apt-get' not in s
    assert 'selected_face_number() == 0' in s


def test_gate_wrapper_cpu_readonly_source_and_exact_image():
    p = Path(__file__).resolve().parents[1] / 'infra/run_mesh_link_gate.sh'
    s = p.read_text()
    assert '--gpus' not in s and '--network none' in s and '--cpus 4' in s
    assert 'src=$CODE,dst=$CODE,readonly' in s and '"$IMAGE_ID" python' in s
    assert 'image-topology-cpu.json' in s and 'sha256:' in s
    assert 'vendor' not in s and 'track_1' not in s


def offline_main(gate, monkeypatch, tmp_path):
    root = tmp_path
    (root / 'results').mkdir()
    image = 'sha256:' + 'a'*64
    build = {'stage': 'world_reward_topology_cpu_build', 'status': 'pass',
             'pymeshlab_version': gate.VERSION, 'wheel_sha256': gate.WHEEL_SHA,
             'base_image_id': 'sha256:'+'b'*64, 'image_id': image}
    (root / 'results/image-topology-cpu.json').write_text(json.dumps(build))
    monkeypatch.setenv('WR_ROOT', str(root)); monkeypatch.setenv('WR_TOPOLOGY_IMAGE_ID', image)
    monkeypatch.setenv('WR_CODE_REVISION', 'c'*40)
    monkeypatch.setattr(gate.platform, 'system', lambda: 'Linux')
    original = Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda self: iter([Path('lo')]) if str(self) == '/sys/class/net' else original(self))
    monkeypatch.setattr(sys, 'argv', [str(Path(gate.__file__))])
    return root / 'results/mesh-link.json'


def test_parent_timeout_kills_worker_and_preserves_frozen_partial(gate, monkeypatch, tmp_path):
    path = offline_main(gate, monkeypatch, tmp_path)
    def run(args, *, timeout, check, env):
        assert args[-1] == '--worker' and timeout == 120. and check is False
        assert len(env['WR_MESH_LINK_WORKER_NONCE']) == 64
        report = json.loads(path.read_text()); report['fixtures'] = [{'fixture': 'torus', 'completed': False}]
        path.write_text(json.dumps(report))
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(gate.subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired): gate.main()
    r = json.loads(path.read_text())
    assert r['status'] == 'fail' and r['error_type'] == 'TimeoutError'
    assert r['fixtures'] == [{'fixture': 'torus', 'completed': False}]
    assert r['adoption_performed'] is False and r['max_gate_seconds'] == 120.
    assert r['container_memory_bytes'] == 4 * 1024**3
    with pytest.raises(RuntimeError, match='Frozen'): gate.main()


def test_main_worker_failure_and_wrong_image_never_pass(gate, monkeypatch, tmp_path):
    path = offline_main(gate, monkeypatch, tmp_path)
    def run(*args, **kwargs):
        r = json.loads(path.read_text()); r.update(status='fail', error='non-manifold')
        path.write_text(json.dumps(r)); return SimpleNamespace(returncode=1)
    monkeypatch.setattr(gate.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='non-manifold'): gate.main()
    path.unlink()
    monkeypatch.setenv('WR_TOPOLOGY_IMAGE_ID', 'sha256:'+'d'*64)
    with pytest.raises(ValueError, match='image'): gate.main()
    assert not path.exists()


def test_failed_synthetic_geometry_exposes_measurement_without_threshold_relaxation(gate, monkeypatch):
    source = tetra(); top = gate.mesh_topology(*source)
    counter = iter([np.zeros((8, 3)), np.ones((8, 3))])
    monkeypatch.setattr(gate, '_surface', lambda *_: next(counter))
    diagnostics = {}
    with pytest.raises(ValueError, match='1%'):
        gate.geometry_gates(source, source, top, top, diagnostics)
    assert diagnostics['sampled_bidirectional_chamfer_diagonal_ratio'] > .01
    assert diagnostics['scale_fitted'] is False


def test_translation_does_not_change_per_shell_signed_volume(gate):
    v, f = hollow()
    first = gate.mesh_topology(v, f)
    second = gate.mesh_topology(v + [100., -50., 30.], f)
    assert [(x['euler'], x['volume_sign']) for x in first['components']] == [(x['euler'], x['volume_sign']) for x in second['components']]
    assert [x['signed_volume'] for x in first['components']] == pytest.approx([x['signed_volume'] for x in second['components']])


def test_worker_refuses_frozen_pass_or_missing_parent_nonce(gate, monkeypatch, tmp_path):
    path = offline_main(gate, monkeypatch, tmp_path)
    frozen = {'status': 'pass', 'stage': 'procedural_pymeshlab_link_condition_topology'}
    path.write_text(json.dumps(frozen)); original = path.read_bytes()
    monkeypatch.setattr(sys, 'argv', [gate.__file__, '--worker'])
    with pytest.raises(RuntimeError, match='frozen'): gate.main()
    assert path.read_bytes() == original
    frozen['status'] = 'running'; path.write_text(json.dumps(frozen)); original = path.read_bytes()
    with pytest.raises(RuntimeError, match='parent-created'): gate.main()
    assert path.read_bytes() == original


def test_thin_cavity_net_volume_cannot_hide_behind_small_component_error(gate, monkeypatch):
    v, f = tetra(); source = np.concatenate([v, v * .96]), np.concatenate([f, f[:, ::-1] + 4])
    top = gate.mesh_topology(*source); after = copy.deepcopy(top)
    after['components'][0]['signed_volume'] *= .99
    monkeypatch.setattr(gate, '_surface', lambda *_: np.zeros((8, 3)))
    diagnostics = {}
    with pytest.raises(ValueError, match='5%'): gate.geometry_gates(source, source, top, after, diagnostics)
    assert max(diagnostics['per_component_relative_volume_errors']) < .05
    assert diagnostics['net_volume_relative_error'] > .05


@pytest.mark.parametrize('failure', ['binding_before', 'binding_after', 'build'])
def test_build_fails_without_manifest_on_local_base_mismatch_or_build_error(tmp_path, failure):
    infra = Path(__file__).resolve().parents[1] / 'infra'
    (tmp_path / 'results').mkdir(); (tmp_path / 'bin').mkdir()
    fake = tmp_path / 'bin/docker'
    script = '''#!/bin/bash
echo "$*" >> "$LOG"
if [[ "$*" == *"image inspect world-reward/cari4d-source"* ]]; then
 echo "sha256:A"
elif [[ "$*" == *"image inspect world-reward/topology-base"* ]]; then
 count=0; [[ ! -f "$COUNT" ]] || count=$(cat "$COUNT"); count=$((count+1)); echo "$count" > "$COUNT"
 if [[ "$FAILURE" == binding_before || ( "$FAILURE" == binding_after && "$count" == 2 ) ]]; then echo "sha256:B"; else echo "sha256:A"; fi
elif [[ "$*" == build* && "$FAILURE" == build ]]; then exit 7
elif [[ "$*" == *"image inspect world-reward/topology-cpu"* ]]; then echo "sha256:B"
fi
'''.replace('sha256:A', 'sha256:'+'a'*64).replace('sha256:B', 'sha256:'+'b'*64)
    fake.write_text(script); fake.chmod(0o755)
    env = os.environ | {'PATH': str(tmp_path / 'bin') + ':' + os.environ['PATH'], 'LOG': str(tmp_path / 'calls'),
                        'COUNT': str(tmp_path / 'count'), 'FAILURE': failure, 'WR_ROOT': str(tmp_path),
                        'WR_CODE': str(infra.parent), 'WR_CODE_REVISION': 'c'*40}
    result = subprocess.run(['bash', str(infra / 'run_topology_cpu_build.sh')], env=env, capture_output=True)
    assert result.returncode == (7 if failure == 'build' else 2)
    assert not (tmp_path / 'results/image-topology-cpu.json').exists()
    calls = (tmp_path / 'calls').read_text()
    assert 'pull ' not in calls and '--gpus' not in calls
    if failure == 'binding_before': assert 'build --pull=false' not in calls

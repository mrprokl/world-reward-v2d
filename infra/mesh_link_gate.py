"""CPU experiment for a pinned topology-preserving QEM, never production adoption.

Only procedural meshes are read. The link condition preserves combinatorial
manifold topology; normal preservation is a soft QEM preference, not a proof
of an embedding or reconstruction accuracy. No hole filling, shell deletion,
reorientation, merging, scale fitting, or repair is permitted. Geometry gates
below are engineering thresholds declared on our own synthetic fixtures.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from world_reward.data import sha256

VERSION = "2025.7.post1"
WHEEL_SHA = "c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30"
FACE_BUDGET = VERTEX_BUDGET = 4096
MAX_SECONDS = 120.
MAX_CHAMFER_DIAGONAL_RATIO = .01
MAX_VOLUME_RELATIVE_ERROR = .05
SURFACE_SAMPLES = 8192
PARAMETERS = dict(targetperc=0., qualitythr=.3, preserveboundary=True,
                  boundaryweight=1., preservenormal=True, preservetopology=True,
                  optimalplacement=True, planarquadric=False, planarweight=.001,
                  qualityweight=False, autoclean=False, selected=False)


def mesh_topology(vertices, faces) -> dict:
    """Validate an oriented closed 2-manifold, including each vertex link cycle.

    Disconnected inward components are legitimate cavity boundaries. Sign and
    Euler signatures are compared separately; positive total volume alone is
    insufficient. No arrays are mutated or faces deleted to obtain a pass.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    if (np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces)
            or v.ndim != 2 or v.shape[1] != 3 or len(v) < 4
            or f.ndim != 2 or f.shape[1] != 3 or len(f) < 4
            or v.dtype.kind != 'f' or f.dtype.kind not in 'iu'
            or not np.isfinite(v).all() or np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError("Mesh arrays must be finite floating Vx3 and valid integer Fx3")
    if np.any(np.sort(f, axis=1)[:, 1:] == np.sort(f, axis=1)[:, :-1]):
        raise ValueError("Repeated face indices are not a manifold surface")
    extent = float(np.linalg.norm(np.ptp(v, axis=0)))
    double_area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    if extent <= 0 or np.any(double_area <= np.finfo(v.dtype).eps * extent ** 2 * 32):
        raise ValueError("Collapsed/numerically zero-area faces are forbidden")
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientations = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(inverse, weights=orientations) != 0):
        raise ValueError("Edges must each have exactly two opposite incidences")
    active = np.unique(f)
    adjacency = [[] for _ in range(len(v))]
    links = [[] for _ in range(len(v))]
    for a, b in edges:
        adjacency[a].append(int(b)); adjacency[b].append(int(a))
    for a, b, c in f:
        links[a].append((int(b), int(c)))
        links[b].append((int(c), int(a)))
        links[c].append((int(a), int(b)))
    for vertex in active:
        graph = {}
        for a, b in links[vertex]:
            graph.setdefault(a, []).append(b); graph.setdefault(b, []).append(a)
        if any(len(neighbors) != 2 for neighbors in graph.values()):
            raise ValueError("Every vertex link must be a cycle")
        seen, pending = set(), [next(iter(graph))]
        while pending:
            point = pending.pop()
            if point not in seen:
                seen.add(point); pending.extend(graph[point])
        if len(seen) != len(graph):
            raise ValueError("Disconnected vertex link is non-manifold")
    labels = np.full(len(v), -1, dtype=np.int64)
    component = 0
    for vertex in active:
        if labels[vertex] != -1:
            continue
        labels[vertex] = component
        pending = [int(vertex)]
        while pending:
            point = pending.pop()
            for neighbor in adjacency[point]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component; pending.append(neighbor)
        component += 1
    signatures = []
    for index in range(component):
        vf = np.flatnonzero(labels == index)
        ff = f[labels[f[:, 0]] == index]
        ef = edges[labels[edges[:, 0]] == index]
        # Translation-invariant volume evaluation improves conditioning.
        x = v[ff] - np.mean(v[vf], axis=0)
        volume = float(np.einsum('ij,ij->i', x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6.)
        if not np.isfinite(volume) or abs(volume) <= np.finfo(v.dtype).eps * extent ** 3 * 32:
            raise ValueError("Each closed component must have nonzero signed volume")
        signatures.append({"euler": int(len(vf) - len(ef) + len(ff)), "volume_sign": 1 if volume > 0 else -1,
                           "signed_volume": volume, "vertices": int(len(vf)), "faces": int(len(ff))})
    signatures.sort(key=lambda row: (row['euler'], row['volume_sign'], abs(row['signed_volume'])))
    if sum(row['signed_volume'] for row in signatures) <= 0:
        raise ValueError("Complete oriented solid must have positive net volume")
    return {"vertices": len(v), "active_vertices": len(active), "faces": len(f),
            "components": signatures, "closed_oriented_vertex_manifold": True,
            "diagonal": extent}


def simplify(vertices, faces, face_budget=FACE_BUDGET):
    """One global pinned QEM call; hard link condition, no cleanup or fallback."""
    import pymeshlab
    if importlib.metadata.version('pymeshlab') != VERSION:
        raise RuntimeError("PyMeshLab version does not match the pinned wheel")
    mesh_topology(vertices, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(vertices, dtype=np.float64),
                              face_matrix=np.asarray(faces, dtype=np.int32)), 'procedural_fixed_metric')
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=face_budget, **PARAMETERS)
    mesh = ms.current_mesh()
    mesh.compact()  # Compact deleted indices only; not duplicate merging/repair.
    result = mesh.vertex_matrix().copy(), mesh.face_matrix().copy()
    ms.compute_selection_by_self_intersections_per_face()
    intersections = int(mesh.selected_face_number())
    return result, intersections


def _surface(vertices, faces, count=SURFACE_SAMPLES):
    """Area-weighted fixed seed samples, independent of any challenge geometry."""
    v, f = np.asarray(vertices), np.asarray(faces)
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    rng = np.random.default_rng(109)
    triangles = v[f[rng.choice(len(f), count, p=area / area.sum())]]
    uv = rng.random((count, 2)); uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])


def geometry_gates(source, result, source_topology, result_topology, diagnostics=None) -> dict:
    """Synthetic-only bidirectional sampled NN and per-shell signed volume gates."""
    signatures = lambda t: [(x['euler'], x['volume_sign']) for x in t['components']]
    if signatures(source_topology) != signatures(result_topology):
        raise ValueError("Simplification changed component Euler/sign signatures")
    before, after = _surface(*source), _surface(*result)
    distance = .5 * (cKDTree(before).query(after)[0].mean() + cKDTree(after).query(before)[0].mean())
    ratio = float(distance / source_topology['diagonal'])
    # Pair sorted same-sign/Euler components; fixture volumes are deliberately distinct.
    volume_errors = [abs(b['signed_volume'] - a['signed_volume']) / abs(a['signed_volume'])
                     for a, b in zip(source_topology['components'], result_topology['components'])]
    source_volume = sum(x['signed_volume'] for x in source_topology['components'])
    result_volume = sum(x['signed_volume'] for x in result_topology['components'])
    net_volume_error = abs(result_volume - source_volume) / source_volume
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(sampled_bidirectional_chamfer_diagonal_ratio=ratio,
                       per_component_relative_volume_errors=volume_errors,
                       net_volume_relative_error=net_volume_error, scale_fitted=False, translation_fitted=False)
    if not np.isfinite(ratio) or ratio > MAX_CHAMFER_DIAGONAL_RATIO:
        raise ValueError("Synthetic Chamfer exceeds predeclared 1% source diagonal")
    if any(not np.isfinite(x) or x > MAX_VOLUME_RELATIVE_ERROR for x in volume_errors + [net_volume_error]):
        raise ValueError("Synthetic component volume error exceeds predeclared 5%")
    return diagnostics


def procedural_fixtures():
    """Sphere, genus-one torus, thin cavity, disconnected positive solids."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.)
    sphere_v, sphere_f = sphere.vertices.copy(), sphere.faces.copy()
    u, w = np.meshgrid(np.arange(96) * 2 * np.pi / 96, np.arange(64) * 2 * np.pi / 64, indexing='ij')
    torus_v = np.stack([(1. + .3 * np.cos(w)) * np.cos(u),
                        (1. + .3 * np.cos(w)) * np.sin(u), .3 * np.sin(w)], axis=-1).reshape(-1, 3)
    torus_f = []
    for i in range(96):
        for j in range(64):
            a, b = i * 64 + j, ((i + 1) % 96) * 64 + j
            c, d = ((i + 1) % 96) * 64 + (j + 1) % 64, i * 64 + (j + 1) % 64
            torus_f.extend(((a, b, c), (a, c, d)))
    hollow = (np.concatenate([sphere_v, sphere_v * .96]),
              np.concatenate([sphere_f, sphere_f[:, ::-1] + len(sphere_v)]))
    disconnected = (np.concatenate([sphere_v + [-1.5, 0, 0], sphere_v * .7 + [1.5, 0, 0]]),
                    np.concatenate([sphere_f, sphere_f + len(sphere_v)]))
    return [('sphere', (sphere_v, sphere_f)), ('torus', (torus_v, np.asarray(torus_f))),
            ('thin_hollow', hollow), ('disconnected_positive', disconnected)]


def hollow_containment(vertices, faces) -> None:
    """Fixture-only convex support-plane test: inward shell remains inside outer.

    This does not claim arbitrary challenge shell containment or local normals.
    A closed outward triangulated boundary with every point on its inward side
    remains a convex embedding; every inner vertex must be strictly inside it.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    # Recover components without depending on ordering after compaction.
    adjacency = [[] for _ in range(len(v))]
    for a, b, c in f:
        adjacency[a].extend((b, c)); adjacency[b].extend((a, c)); adjacency[c].extend((a, b))
    seen, components = set(), []
    for start in np.unique(f):
        if int(start) in seen: continue
        ids, pending = [], [int(start)]
        while pending:
            a = int(pending.pop())
            if a in seen: continue
            seen.add(a); ids.append(a); pending.extend(adjacency[a])
        components.append(np.asarray(ids))
    if len(components) != 2:
        raise ValueError("Hollow control requires exactly two shells")
    outer, inner = sorted(components, key=lambda ids: np.max(np.linalg.norm(v[ids], axis=1)), reverse=True)
    outer_f = f[np.isin(f[:, 0], outer)]
    origins = v[outer_f[:, 0]]
    normals = np.cross(v[outer_f[:, 1]] - origins, v[outer_f[:, 2]] - origins)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    for offset in range(0, len(outer_f), 128):
        n, p = normals[offset:offset + 128], origins[offset:offset + 128]
        outer_distances = v[outer] @ n.T - np.einsum('ij,ij->i', p, n)
        inner_distances = v[inner] @ n.T - np.einsum('ij,ij->i', p, n)
        if np.max(outer_distances) > 1e-10 or np.max(inner_distances) >= -1e-10:
            raise ValueError("Thin hollow fixture is nonconvex or lost strict shell containment")


def run_gate(report: dict, report_path: Path) -> None:
    """Worker persists each result; its parent enforces a hard 120-second limit."""
    started = time.perf_counter()
    fixtures = procedural_fixtures()
    for name, source in fixtures:
        record = {'fixture': name, 'status': 'fail'}
        report['fixtures'].append(record)
        source_hashes = [_array_hash(x) for x in source]
        source_topology = mesh_topology(*source)
        record['source_topology'] = source_topology
        _write(report_path, report)
        results = []
        for repeat in range(2):
            record['current_repeat'] = repeat
            _write(report_path, report)
            result, intersections = simplify(*source)
            top = mesh_topology(*result)
            record.setdefault('runs', []).append({'topology': top, 'self_intersecting_faces': intersections})
            _write(report_path, report)
            if intersections != 0:
                raise ValueError("Simplification introduced self-intersections")
            if top['vertices'] > VERTEX_BUDGET or top['faces'] > FACE_BUDGET:
                raise ValueError("Hard link condition cannot reach the fixed 4096 budget")
            record['runs'][-1]['geometry'] = {}
            try:
                geometry_gates(source, result, source_topology, top, record['runs'][-1]['geometry'])
            finally:
                _write(report_path, report)
            if name == 'thin_hollow': hollow_containment(*result)
            results.append(result)
        if not all(np.array_equal(a, b) for a, b in zip(*results)):
            raise ValueError("Repeated QEM arrays are not exactly deterministic")
        if source_hashes != [_array_hash(x) for x in source]:
            raise ValueError("Input source geometry was modified")
        record.update(status='pass', two_runs_arrays_identical=True, source_arrays_unchanged=True,
                      containment_verified=(name == 'thin_hollow'))
        _write(report_path, report)
        if time.perf_counter() - started >= MAX_SECONDS:
            raise TimeoutError("Topology worker exceeded the predeclared 120 seconds")
    torus = fixtures[1][1]
    result, intersections = simplify(*torus, face_budget=6)
    top = mesh_topology(*result)
    if (intersections or top['faces'] <= 6
            or [(x['euler'], x['volume_sign']) for x in top['components']] != [(0, 1)]):
        raise ValueError("Impossible torus6 target was met by changing topology/embedding")
    report['torus6_expected_budget_failure'] = {'pass': True, 'retained_faces': top['faces'],
                                               'topology': top, 'self_intersecting_faces': intersections}
    report.update(status='pass', elapsed_seconds=time.perf_counter() - started)
    _write(report_path, report)


def _array_hash(array):
    import hashlib
    a = np.asarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def _write(path: Path, report: dict):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError("Require isolated Linux CPU container with network none")
    root = Path(os.environ.get('WR_ROOT', '/srv/scenesmith/world-reward'))
    path = root / 'results/mesh-link.json'
    if sys.argv[1:] == ['--worker']:
        import hashlib
        report = json.loads(path.read_text())
        nonce = os.environ.get('WR_MESH_LINK_WORKER_NONCE', '')
        if (report.get('status') != 'running'
                or report.get('stage') != 'procedural_pymeshlab_link_condition_topology'
                or report.get('script_sha256') != sha256(Path(__file__))
                or report.get('producer_revision') != os.environ.get('WR_CODE_REVISION')
                or not nonce or report.get('worker_nonce_sha256') != hashlib.sha256(nonce.encode()).hexdigest()):
            raise RuntimeError('Worker requires current parent-created running report; frozen results cannot be rewritten')
        try:
            run_gate(report, path)
        except Exception as exc:
            report.update(status='fail', error_type=type(exc).__name__, error=str(exc))
            _write(path, report)
            raise
        return
    if sys.argv[1:]: raise ValueError("Procedural controls are fixed, no configurable arguments")
    if path.exists() or path.is_symlink(): raise RuntimeError("Frozen mesh-link report exists")
    build_path = root / 'results/image-topology-cpu.json'
    build = json.loads(build_path.read_text())
    if (build.get('stage') != 'world_reward_topology_cpu_build' or build.get('status') != 'pass'
            or build.get('pymeshlab_version') != VERSION or build.get('wheel_sha256') != WHEEL_SHA
            or build.get('image_id') != os.environ.get('WR_TOPOLOGY_IMAGE_ID')):
        raise ValueError("Immutable topology image does not match frozen build manifest")
    import hashlib
    import secrets
    nonce = secrets.token_hex(32)
    report = {'worker_nonce_sha256': hashlib.sha256(nonce.encode()).hexdigest(),
              'stage': 'procedural_pymeshlab_link_condition_topology', 'status': 'running',
              'own_procedural_inputs_only': True, 'challenge_inputs_used': False,
              'challenge_ground_truth_used': False, 'hand_labeled_test': False,
              'adoption_performed': False, 'challenge_performance_verified': False,
              'pymeshlab_version': VERSION, 'wheel_sha256': WHEEL_SHA,
              'pymeshlab_source': '1dc199f9b6c43e58b6db346ba4600866b950b8ae',
              'meshlab_source': 'd876376e3cc4f92d257e248023d82cbac5b03c7d',
              'vcglib_source': 'c94ef4e12e9ea3ae986d9af91005be8328d13719',
              'base_image_id': build['base_image_id'], 'image_id': build['image_id'],
              'build_manifest_sha256': sha256(build_path), 'script_sha256': sha256(Path(__file__)),
              'producer_revision': os.environ.get('WR_CODE_REVISION'), 'device': 'cpu', 'network': 'none',
              'container_cpus': 4, 'container_memory_bytes': 4 * 1024 ** 3,
              'numpy_version': np.__version__, 'face_budget': FACE_BUDGET, 'vertex_budget': VERTEX_BUDGET,
              'parameters': PARAMETERS, 'max_gate_seconds': MAX_SECONDS, 'surface_samples': SURFACE_SAMPLES,
              'max_sampled_chamfer_diagonal_ratio': MAX_CHAMFER_DIAGONAL_RATIO,
              'max_component_relative_volume_error': MAX_VOLUME_RELATIVE_ERROR,
              'no_holes_filled': True, 'no_components_deleted': True, 'no_winding_repair': True,
              'normal_preservation_is_soft': True, 'accuracy_or_universal_embedding_claim': False,
              'fixtures': []}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle: handle.write(json.dumps(report, indent=2) + '\n')
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__)), '--worker'], timeout=MAX_SECONDS, check=False,
                                 env=os.environ | {'WR_MESH_LINK_WORKER_NONCE': nonce})
    except subprocess.TimeoutExpired:
        report = json.loads(path.read_text())
        report.update(status='fail', error_type='TimeoutError', error='Hard 120-second CPU worker limit exceeded',
                      elapsed_seconds=time.perf_counter() - started)
        _write(path, report)
        raise
    report = json.loads(path.read_text())
    if process.returncode or report['status'] != 'pass':
        raise RuntimeError(f"Topology gate failed: {report.get('error', 'worker did not pass')}")
    report['worker_elapsed_seconds'] = report['elapsed_seconds']
    report['elapsed_seconds'] = time.perf_counter() - started
    report['elapsed_includes_child_startup'] = True
    _write(path, report)
    print(json.dumps({'stage': report['stage'], 'status': report['status'], 'adoption_performed': False,
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()

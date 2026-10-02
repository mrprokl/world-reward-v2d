"""Independent CPU endpoint-QEM experiment on deliberately nonconvex solids.

The earlier optimal-placement/convex-control failure remains frozen and is not
rerun. This cohort tests a new algorithm (surviving vertices are source edge
endpoints), with the same fixed topology, geometry and resource budgets. Solid
angle winding on each outward shell separately tests physical containment,
not convexity. No challenge inputs, shell repair, rescaling or adoption occur.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time

import numpy as np

from mesh_link_gate import (FACE_BUDGET, VERTEX_BUDGET, MAX_SECONDS, VERSION, WHEEL_SHA,
                            MAX_CHAMFER_DIAGONAL_RATIO, MAX_VOLUME_RELATIVE_ERROR, SURFACE_SAMPLES,
                            PARAMETERS, mesh_topology, geometry_gates, _array_hash, _write)
from world_reward.data import sha256

ENDPOINT_PARAMETERS = PARAMETERS | {'optimalplacement': False}
WINDING_ATOL = 1e-6
POINT_CHUNK = 32
STAGE = 'procedural_pymeshlab_endpoint_qem_nonconvex_containment'


def endpoint_simplify(vertices, faces):
    """New local fixed parameter mapping, never mutate the old driver's globals."""
    import pymeshlab
    if importlib.metadata.version('pymeshlab') != VERSION:
        raise RuntimeError('Require the exact pinned PyMeshLab version')
    mesh_topology(vertices, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.array(vertices, dtype=np.float64, copy=True),
                              face_matrix=np.array(faces, dtype=np.int32, copy=True)), 'own_endpoint_control')
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=FACE_BUDGET, **ENDPOINT_PARAMETERS)
    mesh = ms.current_mesh(); mesh.compact()
    result = mesh.vertex_matrix().copy(), mesh.face_matrix().copy()
    ms.compute_selection_by_self_intersections_per_face()
    return result, int(mesh.selected_face_number())


def source_intersections(vertices, faces):
    """Selection-only embedding preflight, without simplification or cleanup."""
    import pymeshlab
    if importlib.metadata.version('pymeshlab') != VERSION:
        raise RuntimeError('Require the exact pinned PyMeshLab version')
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(vertices, dtype=np.float64),
                              face_matrix=np.asarray(faces, dtype=np.int32)), 'own_source_embedding')
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def solid_winding_and_distances(points, triangles):
    """Float64 solid angle and exact point/triangle distances in bounded chunks.

    O(Npoints*Ntriangles), no acceleration or approximate winding. Distances
    certify sampled vertices are not on the boundary. They are NOT the minimum
    separation between two continuous triangle surfaces (edge-edge minima may
    be smaller). Global pair intersection checks are an independent required
    precondition for turning connected-boundary winding into containment.
    """
    p, t = np.asarray(points), np.asarray(triangles)
    if (np.ma.isMaskedArray(points) or np.ma.isMaskedArray(triangles)
            or p.dtype.kind != 'f' or t.dtype.kind != 'f' or p.ndim != 2 or p.shape[1:] != (3,)
            or t.ndim != 3 or t.shape[1:] != (3, 3) or not len(p) or not len(t)
            or not np.isfinite(p).all() or not np.isfinite(t).all()):
        raise ValueError('Require finite nonempty floating points and triangles')
    p, t = p.astype(np.float64), t.astype(np.float64)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac); norm2 = np.einsum('ij,ij->i', normal, normal)
    if np.any(norm2 <= 0) or not np.isfinite(norm2).all():
        raise ValueError('No collapsed triangle or boundary repair allowed')
    windings, distances = [], []
    for start in range(0, len(p), POINT_CHUNK):
        q = p[start:start + POINT_CHUNK, None, :]
        x, y, z = a - q, b - q, c - q
        lx, ly, lz = np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), np.linalg.norm(z, axis=-1)
        numerator = np.einsum('pfi,pfi->pf', x, np.cross(y, z))
        denominator = (lx * ly * lz + np.einsum('pfi,pfi->pf', x, y) * lz
                       + np.einsum('pfi,pfi->pf', y, z) * lx + np.einsum('pfi,pfi->pf', z, x) * ly)
        winding = np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi)
        # Interior planar projection plus each closed segment gives exact
        # unsigned point-to-triangle distance, without mesh-wide NN sampling.
        delta = q - a
        height = np.einsum('pfi,fi->pf', delta, normal)
        v = np.einsum('pfi,fi->pf', np.cross(delta, ac), normal) / norm2
        w = np.einsum('pfi,fi->pf', np.cross(ab, delta), normal) / norm2
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        best = np.where(inside, height * height / norm2, np.inf)
        for first, last in ((a, b), (b, c), (c, a)):
            edge = last - first; length = np.einsum('ij,ij->i', edge, edge)
            if np.any(length <= 0): raise ValueError('Zero triangle edge')
            ratio = np.einsum('pfi,fi->pf', q - first, edge) / length
            # Closed-segment closest point, not geometry or parameter clipping.
            ratio = np.minimum(1., np.maximum(0., ratio))
            d = q - first - ratio[..., None] * edge
            best = np.minimum(best, np.einsum('pfi,pfi->pf', d, d))
        distance = np.sqrt(np.min(best, axis=1))
        if not np.isfinite(winding).all() or not np.isfinite(distance).all():
            raise ValueError('Containment arithmetic is nonfinite')
        windings.append(winding); distances.append(distance)
    return np.concatenate(windings), np.concatenate(distances)


def _components(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    adjacency = [[] for _ in range(len(v))]
    for a, b, c in f:
        adjacency[a].extend((int(b), int(c))); adjacency[b].extend((int(a), int(c))); adjacency[c].extend((int(a), int(b)))
    seen, result = set(), []
    for first in np.unique(f):
        if int(first) in seen: continue
        ids, pending = [], [int(first)]
        while pending:
            vertex = pending.pop()
            if vertex in seen: continue
            seen.add(vertex); ids.append(vertex); pending.extend(adjacency[vertex])
        ids = np.asarray(ids)
        triangles = v[f[np.isin(f[:, 0], ids)]]
        centered = triangles - np.mean(v[ids], axis=0)
        volume = float(np.einsum('ij,ij->i', centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6)
        result.append((ids, triangles, volume))
    return result


def true_hollow_containment(vertices, faces, *, self_intersecting_faces, diagnostics=None):
    """Two closed, nonintersecting shells; all inward vertices wind inside outer.

    Query the positive component ALONE: odd parity of combined nested shells
    would wrongly mark the actual cavity interior outside. Winding error and
    sampled boundary distance never replace global pair self-intersections.
    """
    if type(self_intersecting_faces) is not int or self_intersecting_faces != 0:
        raise ValueError('Containment requires zero pair self-intersections')
    mesh_topology(vertices, faces)
    components = _components(vertices, faces)
    positive = [c for c in components if c[2] > 0]
    negative = [c for c in components if c[2] < 0]
    if len(components) != 2 or len(positive) != 1 or len(negative) != 1:
        raise ValueError('Hollow control requires one outward and one inward shell')
    ids, _, _ = negative[0]
    winding, distance = solid_winding_and_distances(np.asarray(vertices)[ids], positive[0][1])
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    margin = 64 * np.finfo(np.float64).eps * diagonal
    errors = np.abs(winding - 1.)
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(winding_max_error=float(errors.max()), winding_atol=WINDING_ATOL,
                       inner_vertex_count=len(ids), boundary_margin=margin,
                       min_inner_vertex_to_outer_surface_distance=float(distance.min()),
                       min_surface_shell_separation=None,
                       continuous_surface_separation_computed=False,
                       vertex_distance_is_upper_bound_on_surface_separation=True,
                       global_pair_self_intersections=0,
                       true_containment_verified=False)
    if np.max(errors) > WINDING_ATOL or np.min(distance) <= margin:
        raise ValueError('Inner shell escapes or touches the outer boundary, or winding is numerically ambiguous')
    diagnostics['true_containment_verified'] = True
    return diagnostics


def convexity_diagnostic(vertices, faces):
    """Per-shell sampled support-plane violations; diagnostic, never convex gate.

    Positive support-plane violations distinguish old convexity from containment.
    The fixed set of 128 source vertices is only a diagnostic lower bound on
    full support violations, not a convexity proof.
    """
    output = []
    for ids, triangles, volume in _components(vertices, faces):
        if volume < 0: continue
        points = np.asarray(vertices)[ids]
        points = points[np.linspace(0, len(points) - 1, min(128, len(points)), dtype=int)]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        maximum = -np.inf
        for start in range(0, len(normals), 256):
            n, p = normals[start:start + 256], triangles[start:start + 256, 0]
            maximum = max(maximum, float((points @ n.T - np.einsum('ij,ij->i', p, n)).max()))
        output.append({'sampled_support_plane_max_violation': maximum, 'convexity_used_as_gate': False})
    return output


def _combine(outer_v, outer_f, inner_v, inner_f):
    return np.concatenate([outer_v, inner_v]), np.concatenate([outer_f, inner_f[:, ::-1] + len(outer_v)])


def new_fixtures():
    """New cohort; no old sphere/torus/thin-sphere/disconnected-sphere reruns."""
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=4)
    unit, faces = sphere.vertices.copy(), sphere.faces.copy()
    azimuth = np.arctan2(unit[:, 1], unit[:, 0]); sin_polar = np.sqrt(1 - unit[:, 2] ** 2)
    radius = 1 + .12 * np.sin(3 * azimuth) * sin_polar ** 2 + .08 * np.cos(5 * azimuth) * sin_polar ** 3
    star = unit * radius[:, None]
    star_hollow = _combine(star, faces, star * .97, faces)
    u, w = np.meshgrid(np.arange(80) * 2 * np.pi / 80, np.arange(48) * 2 * np.pi / 48, indexing='ij')
    def torus(tube):
        return np.stack([(1 + tube * np.cos(w)) * np.cos(u), (1 + tube * np.cos(w)) * np.sin(u), tube * np.sin(w)], axis=-1).reshape(-1, 3)
    torus_faces = []
    for i in range(80):
        for j in range(48):
            a, b = i * 48 + j, ((i + 1) % 80) * 48 + j
            c, d = ((i + 1) % 80) * 48 + (j + 1) % 48, i * 48 + (j + 1) % 48
            torus_faces.extend(((a, b, c), (a, c, d)))
    torus_faces = np.asarray(torus_faces)
    torus_hollow = _combine(torus(.24), torus_faces, torus(.22), torus_faces)
    ellipsoid = unit * [1.2, .8, .6]
    ellipsoid_hollow = _combine(ellipsoid, faces, ellipsoid * .93 + [.02, -.01, .005], faces)
    disconnected = (np.concatenate([star * [.7, .5, .6] + [-1.3, 0, 0], ellipsoid * .55 + [1.3, 0, 0]]),
                    np.concatenate([faces, faces + len(unit)]))
    return [('radial_star_hollow', star_hollow, True), ('torus_thin_wall', torus_hollow, True),
            ('offcenter_ellipsoid_hollow', ellipsoid_hollow, True), ('disconnected_asymmetric', disconnected, False)]


def run_gate(report, path):
    started = time.perf_counter()
    for name, source, hollow in new_fixtures():
        record = {'fixture': name, 'status': 'fail', 'runs': []}; report['fixtures'].append(record)
        hashes = [_array_hash(x) for x in source]
        topology = mesh_topology(*source)
        record['source_topology'] = topology
        _write(path, report)
        intersections = source_intersections(*source)
        record['source_self_intersections'] = intersections
        if intersections: raise ValueError('New procedural source embedding intersects')
        if hollow:
            record['source_containment'] = {}
            true_hollow_containment(*source, self_intersecting_faces=intersections, diagnostics=record['source_containment'])
        record['source_convexity_diagnostic'] = convexity_diagnostic(*source)
        _write(path, report)
        results = []
        for repeat in range(2):
            record['current_repeat'] = repeat; _write(path, report)
            result, intersections = endpoint_simplify(*source)
            item = {'self_intersecting_faces': intersections}; record['runs'].append(item)
            _write(path, report)
            post = mesh_topology(*result); item['topology'] = post
            if intersections: raise ValueError('Endpoint QEM introduced global self-intersections')
            if post['vertices'] > VERTEX_BUDGET or post['faces'] > FACE_BUDGET:
                raise ValueError('Endpoint QEM cannot attain fixed 4096 V/F budget')
            item['geometry'] = {}; geometry_gates(source, result, topology, post, item['geometry'])
            if hollow:
                item['containment'] = {}
                true_hollow_containment(*result, self_intersecting_faces=intersections, diagnostics=item['containment'])
            item['convexity_diagnostic'] = convexity_diagnostic(*result)
            results.append(result); _write(path, report)
        if not all(np.array_equal(a, b) for a, b in zip(*results)):
            raise ValueError('Endpoint QEM two runs are not exactly deterministic')
        if hashes != [_array_hash(x) for x in source]: raise ValueError('Source geometry modified')
        record.update(status='pass', two_runs_arrays_identical=True, source_arrays_unchanged=True)
        _write(path, report)
        if time.perf_counter() - started >= MAX_SECONDS: raise TimeoutError('Endpoint CPU gate reached 120 seconds')
    report.update(status='pass', elapsed_seconds=time.perf_counter() - started)
    _write(path, report)


def main():
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Require Linux CPU container network none')
    root = Path(os.environ.get('WR_ROOT', '/srv/scenesmith/world-reward'))
    path = root / 'results/mesh-endpoint.json'
    if sys.argv[1:] == ['--worker']:
        report = json.loads(path.read_text()); nonce = os.environ.get('WR_ENDPOINT_WORKER_NONCE', '')
        if (report.get('status') != 'running' or report.get('stage') != STAGE
                or report.get('script_sha256') != sha256(Path(__file__))
                or report.get('shared_gate_sha256') != sha256(Path(__file__).with_name('mesh_link_gate.py'))
                or report.get('producer_revision') != os.environ.get('WR_CODE_REVISION')
                or not nonce or report.get('worker_nonce_sha256') != hashlib.sha256(nonce.encode()).hexdigest()):
            raise RuntimeError('Require parent-created current running report; never overwrite frozen pass')
        try: run_gate(report, path)
        except Exception as exc:
            report.update(status='fail', error_type=type(exc).__name__, error=str(exc)); _write(path, report); raise
        return
    if sys.argv[1:]: raise ValueError('Fixed controls accept no arguments')
    if path.exists() or path.is_symlink(): raise RuntimeError('Frozen endpoint report exists')
    build_path = root / 'results/image-topology-cpu.json'; build = json.loads(build_path.read_text())
    if (build.get('stage') != 'world_reward_topology_cpu_build' or build.get('status') != 'pass'
            or build.get('pymeshlab_version') != VERSION or build.get('wheel_sha256') != WHEEL_SHA
            or build.get('image_id') != os.environ.get('WR_TOPOLOGY_IMAGE_ID')):
        raise ValueError('Endpoint image/build manifest mismatch')
    nonce = secrets.token_hex(32)
    report = {'stage': STAGE, 'status': 'running', 'worker_nonce_sha256': hashlib.sha256(nonce.encode()).hexdigest(),
              'own_procedural_inputs_only': True, 'old_cohort_rerun': False, 'old_gate_changed': False,
              'challenge_inputs_used': False, 'challenge_ground_truth_used': False, 'hand_labeled_test': False,
              'adoption_performed': False, 'challenge_performance_verified': False,
              'pymeshlab_version': VERSION, 'wheel_sha256': WHEEL_SHA,
              'pymeshlab_source': '1dc199f9b6c43e58b6db346ba4600866b950b8ae',
              'meshlab_source': 'd876376e3cc4f92d257e248023d82cbac5b03c7d',
              'vcglib_source': 'c94ef4e12e9ea3ae986d9af91005be8328d13719',
              'embedding_check': 'pinned VCGLib numerical global self-intersections, not universal exact predicates',
              'base_image_id': build['base_image_id'], 'image_id': build['image_id'],
              'script_sha256': sha256(Path(__file__)), 'shared_gate_sha256': sha256(Path(__file__).with_name('mesh_link_gate.py')),
              'build_manifest_sha256': sha256(build_path), 'producer_revision': os.environ.get('WR_CODE_REVISION'),
              'device': 'cpu', 'network': 'none', 'container_cpus': 4, 'container_memory_bytes': 4 * 1024 ** 3,
              'parameters': ENDPOINT_PARAMETERS, 'max_gate_seconds': MAX_SECONDS,
              'face_budget': FACE_BUDGET, 'vertex_budget': VERTEX_BUDGET,
              'max_sampled_chamfer_diagonal_ratio': MAX_CHAMFER_DIAGONAL_RATIO,
              'max_component_and_net_volume_relative_error': MAX_VOLUME_RELATIVE_ERROR, 'surface_samples': SURFACE_SAMPLES,
              'winding_atol': WINDING_ATOL, 'point_chunk': POINT_CHUNK,
              'convexity_used_as_gate': False, 'true_containment_method': 'outer_only_float64_solid_angle_all_inner_vertices',
              'continuous_minimum_shell_separation_computed': False,
              'complexity': 'O(inner vertices times outer faces); bounded chunks, whole worker hard timeout', 'fixtures': []}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as h: h.write(json.dumps(report, indent=2) + '\n')
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__)), '--worker'], timeout=MAX_SECONDS, check=False,
                                 env=os.environ | {'WR_ENDPOINT_WORKER_NONCE': nonce})
    except subprocess.TimeoutExpired:
        report = json.loads(path.read_text()); report.update(status='fail', error_type='TimeoutError',
            error='Hard endpoint worker 120-second limit exceeded', elapsed_seconds=time.perf_counter() - started)
        _write(path, report); raise
    report = json.loads(path.read_text())
    if process.returncode or report['status'] != 'pass': raise RuntimeError(f"Endpoint gate failed: {report.get('error', 'worker incomplete')}")
    report.update(worker_elapsed_seconds=report['elapsed_seconds'], elapsed_seconds=time.perf_counter() - started,
                  elapsed_includes_child_startup=True); _write(path, report)
    print(json.dumps({'stage': STAGE, 'status': 'pass', 'adoption_performed': False, 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()

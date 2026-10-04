"""Fresh paired procedural geometry controls; no production inputs or adoption.

Caller authenticates the new binary/runtime and owns the inclusive deadline.
This module neither builds nor launches a container and never runs old gates.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import time

import numpy as np

import exact_mesh_geometry as geometry
import object_budget_endpoint as endpoint
from object_budget_guarded import export_birth_mapping
from precision_volume_gate import topology_and_embedding, measure, packed_mapping
from volume_mesh_gate import BINARY as ORIGINAL_BINARY
from world_reward.mesh_serialization import serialization_preflight

SCALE, METRIC_SCALE, NATIVE_SECONDS = 2.**-16, .375, 900
FIXTURE_NAMES = ("fresh_thin_ellipsoid_cavity", "fresh_corrugated_capsule")


class GeometryControlError(RuntimeError):
    """Partial scalar-only evidence survives a closed gate failure."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capsule():
    """Fixed closed 6050V/12096F surface; no duplicated angular seam/poles."""
    count, radius, half = 96, .35, .65
    rings = []
    for k in range(1, 16):
        u = -np.pi/2 + k*np.pi/32
        rings.append((radius*np.cos(u), -half+radius*np.sin(u)))
    for k in range(33):
        z = -half+2*half*k/32
        r = radius*(1+.035*np.sin(24*np.pi*k/32)*(1-(z/half)**2)**2)
        rings.append((r, z))
    for k in range(1, 16):
        u = k*np.pi/32
        rings.append((radius*np.cos(u), half+radius*np.sin(u)))
    theta = 2*np.pi*np.arange(count)/count
    vertices = [[r*np.cos(t), r*np.sin(t), z] for r, z in rings for t in theta]
    lower, upper = len(vertices), len(vertices)+1
    vertices.extend(((0., 0., -half-radius), (0., 0., half+radius)))
    faces = []
    for ring in range(len(rings)-1):
        for j in range(count):
            a, b = ring*count+j, ring*count+(j+1)%count
            c, d = a+count, b+count
            faces.extend(((a, b, c), (b, d, c)))
    for j in range(count):
        faces.append((lower, (j+1)%count, j))
        faces.append((upper, (len(rings)-1)*count+j, (len(rings)-1)*count+(j+1)%count))
    return np.asarray(vertices, np.float64)*SCALE, np.asarray(faces, np.int64)


def fixtures():
    import trimesh  # Only the Azure caller constructs the two icospheres.
    axes = np.array([1., .75, .55])
    outer = trimesh.creation.icosphere(subdivisions=4)
    inner = trimesh.creation.icosphere(subdivisions=3)
    hollow = (np.r_[outer.vertices*axes, inner.vertices*axes*.96]*SCALE,
              np.r_[outer.faces, inner.faces[:, ::-1]+len(outer.vertices)].astype(np.int64))
    require(len(hollow[0]) == 3204 and len(hollow[1]) == 6400, "Pinned icosphere resolution differs")
    return ((FIXTURE_NAMES[0], hollow, True), (FIXTURE_NAMES[1], capsule(), False))


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)), "Canonical artifact required")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 32 << 20,
            "Bounded original artifact required")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            "Artifact changed during hashing")
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def array_hashes(mesh):
    return [hashlib.sha256(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode()
                           + b'\0' + np.ascontiguousarray(a).tobytes()).hexdigest() for a in mesh]


def math_mesh(mesh):
    v, f = mesh
    promoted = v.astype(np.float64, copy=True)
    require(np.array_equal(promoted, v), "Float64 predicate view changed represented coordinates")
    return promoted, f


def stage(source, mesh, mapping, cavity, remaining):
    remaining()
    topology_and_embedding(math_mesh(mesh), cavity)
    result = measure(math_mesh(source), math_mesh(mesh), mapping)
    remaining()
    return result | dict(stored_array_sha256=array_hashes(mesh),
                         oriented_triangles_sha256=hashlib.sha256(endpoint._triangle_rows(*mesh).tobytes()).hexdigest())


def branch(source, cavity, binary, method, work, helper, remaining, record):
    a, b, m = (work/n for n in ('input.obj', 'candidate.obj', 'mapping.json'))
    geometry.write_obj(a, *source)
    before = identity(a)
    record.update(phase='native', native_attempts=1, native_returned=False, native_input=before)
    started = time.monotonic()
    try:
        child = subprocess.run([str(binary), str(a), str(b), str(m)], capture_output=True,
                               timeout=min(NATIVE_SECONDS, remaining()))
    finally:
        record['native_elapsed_seconds'] = time.monotonic()-started
        require(identity(a) == before, "Native input OBJ changed")
    record.update(native_returned=True, native_exit_code=child.returncode,
                  native_elapsed_seconds=time.monotonic()-started)
    if child.returncode:
        record['native_stderr_tail'] = child.stderr[-500:].decode(errors='replace')
    require(child.returncode == 0, "Native compiler failed fixed geometry control")
    candidate = geometry.read_obj(b)
    mapping = json.loads(m.read_text())
    if method == 'serialization':
        serialization = mapping['serialization']
        require(serialization['serialization_safe'] is True and serialization['committed_collapses'] > 0,
                "New compiler did not perform admissible real collapses")
        require(type(serialization['serialization_vetoes']) is int and serialization['serialization_vetoes'] >= 0,
                "Actual serialization veto count required")
        record['serialization_vetoes'] = serialization['serialization_vetoes']
        mapping = mapping['native_volume']
        require(serialization['committed_collapses'] == mapping['committed_collapses'], "Collapse accounting differs")
    require(mapping['native_cost_and_placement_unchanged'] is True and mapping['cost_normalization'] is False
            and mapping['final_shell_volumes_verified'] is True and mapping['volume_relative_limit'] == .05
            and mapping['committed_collapses'] > 0, "Native original volume/cost/real-collapse contract differs")
    record.update(native_artifacts={p.name: identity(p) for p in (a, b, m)},
                  committed_collapses=mapping['committed_collapses'], phase='candidate')
    record['candidate'] = stage(source, candidate, mapping, cavity, remaining)
    require(serialization_preflight(*candidate)['position_weld_admissible'], "Candidate serialization is unsafe")
    import trimesh
    glb = work/'candidate.glb'
    record['phase'] = 'float32_export'
    trimesh.Trimesh(*candidate, process=False).export(glb)
    loaded = endpoint._load_mesh(glb)
    require(np.array_equal(loaded[0], loaded[0].astype(np.float32).astype(np.float64)),
            "GLB loader did not preserve exact binary32 positions")
    exported = endpoint.exact_weld(*loaded)[:2]
    emap = export_birth_mapping(candidate, exported, mapping)
    record['exported'] = stage(source, exported, emap, cavity, remaining)
    record['glb_identity'] = identity(glb)
    record['phase'] = 'official_pack'
    pv, pf = helper.budget_mesh(str(glb), faces=4096, vertices=4096)
    compact, fidelity = geometry.verify_pack_fidelity(exported, pv, pf)
    pmap = packed_mapping(exported, compact, emap)
    record['packed'] = stage(source, compact, pmap, cavity, remaining)
    record['official_pack_fidelity'] = fidelity
    record['phase'] = 'metric_bake'
    metric = pv*METRIC_SCALE
    expected = (compact[0].astype(pv.dtype)*METRIC_SCALE, compact[1])
    metric_compact, _ = geometry.verify_pack_fidelity(expected, metric, pf)
    record['metric_baked'] = stage((source[0]*METRIC_SCALE, source[1]), metric_compact, pmap, cavity, remaining)
    record.update(status='pass', phase='complete', metric_scale_baked_once=METRIC_SCALE)


def geometry_controls(new_binary, scratch, remaining, *, official_helper: Path):
    """At most four native calls; old failures are comparisons, new failures stop.

    `remaining()` raises on the caller's inclusive deadline. Only the explicitly
    supplied, independently authenticated single official helper is imported.
    The caller persists returned evidence or GeometryControlError.report; this
    module removes only its newly created scratch subtree and returns no arrays.
    """
    new_binary, scratch, official_helper = map(Path, (new_binary, scratch, official_helper))
    require(scratch.is_absolute() and scratch.resolve() == scratch and scratch.is_dir()
            and not any(p.is_symlink() for p in (scratch, *scratch.parents)), "Canonical caller scratch required")
    work = scratch/'geometry-controls'
    require(not work.exists() and not work.is_symlink(), "Fresh geometry scratch required")
    remaining()
    before = {str(p): identity(p) for p in (new_binary, ORIGINAL_BINARY, official_helper)}
    require(new_binary != ORIGINAL_BINARY, "Distinct new compiler required")
    require(before[str(official_helper)]['sha256'] == endpoint.BUDGET_HELPER_SHA
            and before[str(official_helper)]['bytes'] == 2031, "Exact single official helper required")
    legacy = geometry.validate_legacy_sources()
    report = dict(stage='mesh_serialization_geometry_controls_v1', status='fail', adoption=False,
                  challenge_inputs_used=False, reconstruction_accuracy_verified=False,
                  procedural_geometry_qualification_only=True, native_budget_seconds=NATIVE_SECONDS,
                  maximum_native_calls=4, paired_fixtures=[], source_artifacts=before,
                  legacy_sources=legacy, metric_scale=METRIC_SCALE)
    work.mkdir(mode=0o700)
    failure = None
    source_records = []
    try:
        spec = importlib.util.spec_from_file_location('wr_serialization_official_single_helper', official_helper)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        sources = fixtures()
        # Both complete sources qualify before any native method observes one.
        for name, source, cavity in sources:
            remaining()
            source_records.append((source, array_hashes(source)))
            require(len(source[1]) > 4096 and serialization_preflight(*source)['position_weld_admissible'],
                    "Frozen source does not exercise reduction with safe serialization")
            topology = topology_and_embedding(math_mesh(source), cavity)
            stored_source = (source[0].astype(np.float32), source[1])
            stored_topology = topology_and_embedding(math_mesh(stored_source), cavity)
            require(array_hashes(source) == source_records[-1][1], "Source validation modified geometry")
            report['paired_fixtures'].append(dict(fixture=name, source_array_sha256=array_hashes(source),
                source_topology=topology, source_float32_topology=stored_topology,
                source_float32_array_sha256=array_hashes(stored_source), comparisons=[]))
        for (name, source, cavity), record in zip(sources, report['paired_fixtures']):
            for method, binary in (('original', ORIGINAL_BINARY), ('serialization', new_binary)):
                remaining()
                selected = dict(method=method, status='fail', phase='pre_native', native_attempts=0,
                                native_returned=False)
                record['comparisons'].append(selected)
                directory = work/(name+'-'+method); directory.mkdir(mode=0o700)
                try:
                    branch(source, cavity, binary, method, directory, helper, remaining, selected)
                except Exception as exc:
                    selected.update(error_type=type(exc).__name__, error=str(exc)[-500:])
                    selected['failure_scope'] = ('native_timeout' if isinstance(exc, subprocess.TimeoutExpired)
                        else 'measured_geometry_or_native_rejection'
                        if selected.get('native_attempts') == 1 and isinstance(exc, ValueError)
                        else 'technical_unavailable')
                    if method == 'serialization':
                        raise
                    if selected['failure_scope'] == 'technical_unavailable':
                        raise
                require(array_hashes(source) == record['source_array_sha256'], "Shared source changed between arms")
        new_rows = [row for fixture in report['paired_fixtures'] for row in fixture['comparisons']
                    if row['method'] == 'serialization']
        vetoes = sum(row['serialization_vetoes'] for row in new_rows)
        report.update(status='pass' if vetoes else 'inconclusive', serialization_vetoes=vetoes,
                      mechanism_exercised=bool(vetoes), reroll_performed=False)
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__, error=str(exc)[-500:])
    finally:
        try:
            require(all(array_hashes(mesh) == hashes for mesh, hashes in source_records), "Source arrays changed")
            require({str(p): identity(p) for p in (new_binary, ORIGINAL_BINARY, official_helper)} == before
                    and geometry.validate_legacy_sources() == legacy, "Original source/binary/helper changed")
            remaining(); report['sources_rehashed_after'] = True
        except Exception as exc:
            failure = failure or exc; report.update(post_error_type=type(exc).__name__)
        try:
            shutil.rmtree(work); report['owned_scratch_removed'] = True
            remaining()
        except Exception as exc:
            failure = failure or exc; report.update(cleanup_error_type=type(exc).__name__)
    if failure:
        report['status'] = 'fail'
        raise GeometryControlError("Frozen geometry control failed", report) from failure
    return report

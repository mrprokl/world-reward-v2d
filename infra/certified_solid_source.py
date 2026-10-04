"""Read-only certification of represented source geometry, never repair or QEM."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import time

import numpy as np
import object_budget_endpoint as endpoint
import mesh_precision_diagnostic as precision
import guarded_mesh_gate as guarded
import world_reward.oriented_solid_forest as material

BUDGET_SECONDS, NATIVE_SECONDS = 300, 180
MAX_VERTICES, MAX_FACES, MAX_COMPONENTS = 1000000, 2000000, 256
MAX_STDOUT, MAX_STDERR = 1 << 20, 512
TRUE_FLAGS = ('closed_oriented_vertex_manifold_verified', 'all_original_faces_retained',
              'all_original_vertices_referenced', 'exact_nondegenerate_triangles_verified',
              'component_self_intersections_absent', 'inter_component_surface_contacts_absent')
GEOMETRY_REJECTIONS = frozenset((
    'Repeated face indices', 'Exact degenerate triangle', 'Duplicate indexed triangle',
    'A vertex belongs to multiple component labels', 'Closed opposite edge incidences required',
    'Unreferenced original vertex forbidden', 'Vertex link is not a cycle', 'Disconnected vertex link',
    'Empty or incomplete component label', 'Component edge crosses labels',
    'One label contains disconnected surfaces', 'Surface_mesh rejected an original face',
    'Exact zero signed component volume', 'Exact component self-intersection',
    'Components intersect or touch', 'Containment witness is on boundary',
))
ORDER_SCOPE = 'unprocessed loader source face order; raw GLB triangle multiset parity only'
FALSE_FLAGS = ('geometry_repaired', 'orientation_changed', 'qem_executed', 'forest_adjudicated',
               'reconstruction_accuracy_verified')
FIELDS = set(('schema', 'status', 'source_sha256', 'cgal_version', 'vertices', 'faces', 'component_count',
              'components', 'inside', 'represented_coordinates') + TRUE_FLAGS + FALSE_FLAGS)


class SourceCertificationError(RuntimeError):
    def __init__(self, report):
        super().__init__('Source solid certification failed')
        self.report = report


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(path, maximum=256 << 20):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not any(x.is_symlink() for x in (p, *p.parents)),
            'Canonical source/binary/helper required')
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum,
            'Bounded single-link regular artifact required')
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = p.lstat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, f) == getattr(after, f) for f in fields), 'Artifact changed while hashed')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def components(vertices, faces):
    """Connected labels compacted by first unprocessed LOADED source face, never volume/sign."""
    v, f = np.asarray(vertices), np.asarray(faces)
    require(v.dtype == np.float64 and v.ndim == 2 and v.shape[1] == 3 and
            4 <= len(v) <= MAX_VERTICES and np.isfinite(v).all() and
            f.dtype == np.int64 and f.ndim == 2 and f.shape[1] == 3 and 4 <= len(f) <= MAX_FACES and
            int(f.min()) >= 0 and int(f.max()) < len(v), 'Finite binary64 geometry/index bounds required')
    require(np.array_equal(np.unique(f), np.arange(len(v))), 'Unreferenced source vertices forbidden')
    raw = guarded.component_labels(v, f)
    require(raw.shape == (len(f),) and raw.dtype.kind in 'iu', 'Connected component labels differ')
    first = {}
    labels = np.empty(len(f), dtype=np.int64)
    keys = []
    for position, label in enumerate(raw):
        native = int(label)
        if native not in first:
            first[native] = len(first); keys.append('source-loaded-first-face-' + str(position))
        labels[position] = first[native]
    require(1 <= len(keys) <= MAX_COMPONENTS, 'Component capacity exceeded')
    # Shared-vertex boundaries belong to one connected label. Native exact link
    # checks must reject pinches; do not silently split or invent a manifold.
    rows = []
    for label in range(len(keys)):
        selected = f[labels == label]; ids = np.unique(selected)
        require(len(ids) >= 4 and len(selected) >= 4, 'Incomplete connected boundary')
        rows.append(dict(original_component_id=label, original_vertices=len(ids),
                         original_faces=len(selected), witness_original_vertex=int(ids[0])))
    return labels, tuple(keys), rows


def input_ascii(vertices, faces, labels, count):
    lines = [f'WR_SOLID_QUERY_V1 {len(vertices)} {len(faces)} {count}']
    lines.extend(' '.join(format(float(x), '.17g') for x in row) for row in vertices)
    lines.extend(' '.join(str(int(x)) for x in (*row, label)) for row, label in zip(faces, labels))
    return ('\n'.join(lines) + '\n').encode('ascii')


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate native JSON key')
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite native JSON')))
    json.dumps(value, allow_nan=False)
    return value


def native_certificate(raw, source_sha256, vertices, faces, expected_rows):
    require(len(raw) <= MAX_STDOUT, 'Native output exceeds bound')
    report = strict_json(raw)
    require(type(report) is dict and set(report) == FIELDS and
            report['schema'] == 'world_reward.certified_solid_query.v1' and report['status'] == 'pass' and
            report['source_sha256'] == source_sha256 and report['cgal_version'] == '6.0.1', 'Native schema/source/CGAL differs')
    for key, value in (('vertices', len(vertices)), ('faces', len(faces)), ('component_count', len(expected_rows))):
        require(type(report[key]) is int and report[key] == value, 'Native complete counts differ')
    require(all(report[k] is True for k in TRUE_FLAGS) and all(report[k] is False for k in FALSE_FLAGS) and
            report['represented_coordinates'] == 'EPECK exact values of original parsed IEEE754 binary64; no perturbation',
            'Native scope/represented coordinates differ')
    rows = report['components']
    require(type(rows) is list and len(rows) == len(expected_rows), 'Native complete component lineage required')
    signs = []
    for row, expected in zip(rows, expected_rows):
        require(type(row) is dict and set(row) == set(expected) | {'exact_volume_sign', 'diagnostic_signed_volume'} and
                all(type(row[k]) is int and row[k] == v for k, v in expected.items()) and
                type(row['exact_volume_sign']) is int and row['exact_volume_sign'] in (-1, 1),
                'Native original component/witness/sign differs')
        diagnostic = row['diagnostic_signed_volume']
        require(diagnostic is None or type(diagnostic) in (int, float) and math.isfinite(diagnostic),
                'Invalid optional native volume diagnostic')
        signs.append(row['exact_volume_sign'])
    inside = report['inside']; count = len(rows)
    require(type(inside) is list and len(inside) == count and all(type(row) is list and len(row) == count and
            all(type(x) is bool for x in row) for row in inside), 'Complete typed containment matrix required')
    return report, np.array(signs, dtype=np.int64), np.array(inside, dtype=bool)


def certify_source(mesh_path, binary, native_source_sha256):
    """One bounded native query, scalar certificate or exception.report on FAIL.

    Caller authenticates the actual qualified binary/image and producer receipts
    before this core. The result certifies represented geometry, not RGB accuracy,
    physical scale, official packing, historical acceptance or adoption.
    """
    started = time.monotonic()
    report = dict(stage='certified_solid_source_v1', status='fail', phase='source_identity', native_attempts=0,
                  native_returned=False, budget_seconds=BUDGET_SECONDS, native_seconds=NATIVE_SECONDS,
                  geometry_repaired=False, orientation_changed=False, qem_executed=False,
                  adoption=False, reconstruction_accuracy_verified=False, physical_scale_accuracy_verified=False,
                  packing_validated=False, ground_truth_used=False)
    paths = (Path(mesh_path), Path(binary), Path(__file__).resolve(), Path(endpoint.__file__).resolve(),
             Path(precision.__file__).resolve(), Path(guarded.__file__).resolve(), Path(material.__file__).resolve(),
             Path(__file__).with_name('certified_solid_query.cpp').resolve())
    before = None
    def remaining():
        value = BUDGET_SECONDS - (time.monotonic() - started)
        require(value > 0, 'Inclusive source-certificate deadline')
        return value
    try:
        require(type(native_source_sha256) is str and re.fullmatch('[0-9a-f]{64}', native_source_sha256), 'Native source SHA required')
        before = {str(p): identity(p) for p in paths}; report['artifacts_before'] = before
        require(before[str(paths[-1])]['sha256'] == native_source_sha256, 'Current native source differs from qualified source')
        remaining(); report['phase'] = 'represented_geometry'
        local, raw_triangles, accessors = precision.raw_glb(paths[0])
        vertices, faces = endpoint._load_mesh(paths[0])
        require(not np.ma.isMaskedArray(vertices) and not np.ma.isMaskedArray(faces), 'Masked source unsupported')
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        require(vertices.dtype == np.float64 and faces.dtype == np.int64 and
                vertices.ndim == faces.ndim == 2 and vertices.shape[1:] == faces.shape[1:] == (3,) and
                4 <= len(vertices) <= MAX_VERTICES and 4 <= len(faces) <= MAX_FACES and
                np.isfinite(vertices).all() and int(faces.min()) >= 0 and int(faces.max()) < len(vertices), 'Loaded source bounds/representation differ')
        require(np.array_equal(np.unique(faces), np.arange(len(vertices))), 'Raw loaded orphans forbidden')
        for v, f in local:
            require(np.array_equal(np.unique(f), np.arange(len(v))), 'Original accessor orphans forbidden')
        raw_hash = precision.triangle_hash(raw_triangles)
        require(precision.triangle_hash(vertices[faces]) == raw_hash, 'Raw scene/loader oriented triangle mismatch')
        welded_v, welded_f, weld = endpoint.exact_weld(vertices, faces)
        require(len(welded_f) == len(faces) and np.array_equal(welded_v[welded_f], vertices[faces]) and
                precision.triangle_hash(welded_v[welded_f]) == raw_hash, 'Weld changed original oriented faces')
        labels, keys, rows = components(welded_v, welded_f)
        report.update(raw_scene_oriented_triangles_sha256=raw_hash, loaded_oriented_triangles_sha256=raw_hash,
                      welded_oriented_triangles_sha256=raw_hash, exact_welding=weld, accessors=accessors,
                      vertices=len(welded_v), faces=len(welded_f), component_count=len(keys), component_keys=keys,
                      source_loaded_face_order_preserved=True, order_scope=ORDER_SCOPE, source_orphans_removed=False)
        data = input_ascii(welded_v, welded_f, labels, len(keys))
        report['input_ascii'] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        report['phase'] = 'native_exact_geometry'; report['native_attempts'] = 1
        child = subprocess.run([str(paths[1])], input=data, capture_output=True, check=False,
                               timeout=min(NATIVE_SECONDS, remaining()))
        report.update(native_returned=True, native_returncode=child.returncode,
                      native_stdout=dict(bytes=len(child.stdout), sha256=hashlib.sha256(child.stdout).hexdigest()),
                      native_stderr=dict(bytes=len(child.stderr), sha256=hashlib.sha256(child.stderr).hexdigest()))
        require(len(child.stdout) <= MAX_STDOUT and len(child.stderr) <= MAX_STDERR, 'Native output capacity exceeded')
        if child.returncode != 0:
            report['failure_scope'] = 'native_execution_contract'
            prefix = b'certified_solid_query FAIL: '
            require(child.returncode == 1 and not child.stdout and child.stderr.startswith(prefix) and
                    child.stderr.endswith(b'\n'), 'Native execution failed outside rejection ABI')
            reason = child.stderr[len(prefix):-1].decode('ascii', errors='strict')
            require(len(reason) <= 300 and all(32 <= ord(c) <= 126 for c in reason), 'Native failure reason malformed')
            report['native_rejection'] = reason
            if reason in GEOMETRY_REJECTIONS:
                report['failure_scope'] = 'native_geometric_rejection'
                raise ValueError('Exact represented geometry rejected')
            raise ValueError('Native failure does not certify geometric inadmissibility')
        require(not child.stderr, 'Successful native emitted unexpected stderr')
        certificate, signs, inside = native_certificate(child.stdout, native_source_sha256, welded_v, welded_f, rows)
        report['native_certificate'] = certificate
        report['phase'] = 'oriented_material_forest'
        report['failure_scope'] = 'material_forest_rejection'
        forest = material.adjudicate_oriented_solid_forest(keys, signs, inside)
        report['forest'] = dict(component_keys=forest.component_keys, parents=forest.parents.tolist(),
                                signs=forest.signs.tolist(), depths=forest.depths.tolist(), inside=forest.inside.tolist())
        remaining(); report.update(status='pass', phase='complete', represented_embedding_certified=True)
        report.pop('failure_scope', None)
    except Exception as error:
        report['failure_type'] = type(error).__name__
        report.setdefault('failure_scope', 'deadline' if isinstance(error, subprocess.TimeoutExpired) else 'source_or_runtime_contract')
    finally:
        if before is not None:
            try:
                report['artifacts_after'] = {str(p): identity(p) for p in paths}
                require(report['artifacts_after'] == before, 'Source/binary/helper changed during certification')
                report['source_binary_helpers_rehashed_after'] = True
            except Exception as error:
                report.update(status='fail', failure_type=type(error).__name__, failure_scope='posthash_integrity')
        report['elapsed_seconds'] = time.monotonic() - started
        if report['elapsed_seconds'] > BUDGET_SECONDS:
            report.update(status='fail', failure_type='InclusiveDeadline', failure_scope='deadline')
    if report['status'] != 'pass':
        raise SourceCertificationError(report)
    return report

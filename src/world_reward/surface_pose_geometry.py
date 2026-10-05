"""Array-only surface/pose consumer proofs, not a solid or reconstruction certificate.

All original canonical vertices (including orphans) and faces are retained.
The caller authenticates the GLB decoder, native loader and rigid-projection
implementation. This module compares their supplied results; it never executes
models, geometry I/O, a solver, a weld, a projection or a repair.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import numpy as np

from world_reward.surface_identity import SurfaceIdentity

ROUNDTRIP_LIMIT = 1e-5  # Existing represented-metric consumer limit, not accuracy.


def _owned(a):
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def surface_signature(vertices, faces):
    """Component-local oriented surface and boundary geometry, no solid signs."""
    from official_pack_geometry import canonical_oriented_triangles
    surface = SurfaceIdentity(vertices, faces)
    records = []
    for k in range(len(surface.component_keys)):
        triangles = canonical_oriented_triangles(vertices, faces[surface.face_components == k])
        loops = []
        for loop, component in zip(surface.boundary_loops, surface.boundary_components):
            if component == k:
                points = np.asarray(vertices[list(loop)], dtype=np.float64)
                rotations = [np.roll(points, -i, axis=0).tobytes() for i in range(len(points))]
                loops.append(hashlib.sha256(min(rotations)).hexdigest())
        records.append((hashlib.sha256(triangles.tobytes()).hexdigest(), tuple(sorted(loops))))
    return tuple(sorted(records))


def _array(a, dtype, shape, name):
    if np.ma.isMaskedArray(a):
        raise ValueError(f"{name}: explicit unmasked array required")
    a = np.asarray(a)
    if a.dtype != np.dtype(dtype) or a.ndim != len(shape) or any(
            n is not None and n != actual for n, actual in zip(shape, a.shape)):
        raise ValueError(f"{name}: exact dtype/shape required")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError(f"{name}: finite represented coordinates required")
    return a


def compact_surface(vertices, faces, *, canonical_vertex_count=None,
                    canonical_face_count=None, reference_vertices=None,
                    reference_faces=None):
    """Remove only certified trailing official padding; never compact references.

    Padded4096 arrays require independently bound canonical counts or exact
    reference arrays. Counts are declarations, not provenance authentication.
    All unused canonical rows remain in the returned immutable mesh.
    """
    v = _array(vertices, "float64", (None, 3), "vertices")
    f = _array(faces, "int64", (None, 3), "faces")
    if (reference_vertices is None) != (reference_faces is None):
        raise ValueError("Both canonical reference arrays required")
    if reference_vertices is not None:
        rv = _array(reference_vertices, "float64", (None, 3), "reference vertices")
        rf = _array(reference_faces, "int64", (None, 3), "reference faces")
        if canonical_vertex_count not in (None, len(rv)) or canonical_face_count not in (None, len(rf)):
            raise ValueError("Canonical counts disagree with original reference")
        canonical_vertex_count, canonical_face_count = len(rv), len(rf)
    if canonical_vertex_count is None or canonical_face_count is None:
        if len(v) == 4096 or len(f) == 4096:
            raise ValueError("Padded mesh requires independently bound original canonical counts")
        canonical_vertex_count, canonical_face_count = len(v), len(f)
    nv, nf = canonical_vertex_count, canonical_face_count
    if (type(nv) is not int or type(nf) is not int or not 3 <= nv <= len(v) <= 4096
            or not 1 <= nf <= len(f) <= 4096):
        raise ValueError("Canonical counts must preserve both full original budgets")
    if not np.all(v[nv:] == v[0]) or not np.all(f[nf:] == 0):
        raise ValueError("Only trailing repeat-first-vertex/all-zero-face official padding allowed")
    cv, cf = v[:nv], f[:nf]
    if reference_vertices is not None and (not _same(cv, rv) or not _same(cf, rf)):
        raise ValueError("Original canonical rows/faces changed before surface preparation")
    surface = SurfaceIdentity(cv, cf)
    topology = surface.report()
    topology.update(canonical_vertex_count=nv, canonical_face_count=nf,
                    official_padding_faces=len(f) - nf, official_padding_vertices=len(v) - nv,
                    canonical_unused_vertices_preserved=int(surface.diagnostics["unused_vertices_preserved"]))
    return surface.vertices, surface.faces, _owned(np.arange(nf, dtype=np.int64)), topology


def _matrix(a, name):
    a = _array(a, "float64", (4, 4), name)
    if (not np.array_equal(a[3], [0., 0., 0., 1.])
            or not np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3), atol=ROUNDTRIP_LIMIT, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=ROUNDTRIP_LIMIT, rtol=0)):
        raise ValueError(f"{name}: finite represented proper rigid affine transform required")
    return a


def transform_points(vertices, matrix):
    """Pinned Trimesh5.1.0 arithmetic, including its native identity fast path.

    The 1e-8 branch is the original loader implementation, not an additional
    geometry acceptance tolerance. Matrix provenance remains caller-verified.
    """
    v = _array(vertices, "float64", (None, 3), "transform vertices")
    m = _matrix(matrix, "transform")
    if np.abs(m - np.eye(4)).max() < 1e-8:
        return np.ascontiguousarray(v.copy())
    stack = np.column_stack((v, np.ones(len(v))))
    result = np.dot(m, stack.T).T[:, :3]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite represented transform result")
    return result


def _orientation(original, represented, faces):
    # Exact dyadic dot of oriented normals: no extent-dependent threshold.
    for old, new in zip(original[faces], represented[faces]):
        normals = []
        for tri in (old, new):
            p = [[Fraction.from_float(float(x)) for x in row] for row in tri]
            a = [p[1][k] - p[0][k] for k in range(3)]
            b = [p[2][k] - p[0][k] for k in range(3)]
            normals.append([a[i] * b[j] - a[j] * b[i] for i, j in ((1, 2), (2, 0), (0, 1))])
        if sum(x * y for x, y in zip(*normals)) <= 0:
            raise ValueError("Native FP32 consumer collapsed or reversed an original oriented triangle")


def serialized_mesh(source_vertices, source_faces, *, local_vertices, local_faces,
                    raw_world_triangles, loaded_vertices, loaded_faces,
                    native_vertices, native_faces, raw_matrix, effective_matrix,
                    projected_matrix):
    """Prove raw A and source-verified native-effective B as distinct stages.

    Single original geometry/instance only. ``projected_matrix`` must be the
    caller's independently source-authenticated native fix_rigid(A) result;
    comparing it here is not authentication or evidence of no projection.
    """
    sv = _array(source_vertices, "float64", (None, 3), "source vertices")
    sf = _array(source_faces, "int64", (None, 3), "source faces")
    original = SurfaceIdentity(sv, sf)
    lv = _array(local_vertices, "float64", sv.shape, "local GLB vertices")
    lf = _array(local_faces, "int64", sf.shape, "local GLB faces")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        stored = sv.astype(np.float32).astype(np.float64)
    if not np.isfinite(stored).all() or not _same(lv, stored) or not _same(lf, sf):
        raise ValueError("GLB changed original local vertex rows/faces rather than native FP32 storage")
    _orientation(sv, stored, sf)
    a, b = _matrix(raw_matrix, "raw A"), _matrix(effective_matrix, "effective B")
    projected = _matrix(projected_matrix, "source-verified projected B")
    if not np.array_equal(b, projected):
        raise ValueError("Native effective graph differs from supplied source-verified projection")
    raw = _array(raw_world_triangles, "float64", (len(sf), 3, 3), "raw world triangles")
    expected_raw, effective = transform_points(stored, a), transform_points(stored, b)
    if not _same(raw, expected_raw[sf]):
        raise ValueError("Raw GLB world triangles differ from A(stored local geometry)")
    loaded = _array(loaded_vertices, "float64", sv.shape, "native scene-loaded vertices")
    loaded_f = _array(loaded_faces, "int64", sf.shape, "native scene-loaded faces")
    if not _same(loaded, effective) or not _same(loaded_f, sf):
        raise ValueError("Native loader differs from B(stored), or dropped/reindexed original rows")
    native = _array(native_vertices, "float32", sv.shape, "native optimizer vertices")
    native_f = _array(native_faces, "int64", sf.shape, "native optimizer faces")
    if not _same(native, loaded.astype(np.float32)) or not _same(native_f, sf):
        raise ValueError("Native optimizer changed its actual FP32 loader copy")
    represented = native.astype(np.float64)
    _orientation(effective, represented, sf)
    SurfaceIdentity(represented, sf)
    error = float(np.max(np.linalg.norm(represented - effective, axis=1)))
    if not np.isfinite(error) or error > ROUNDTRIP_LIMIT:
        raise ValueError("Native represented metric surface exceeds inherited roundtrip limit")
    proof = dict(source_domain="surface", raw_node_matrix=a.tolist(), effective_native_node_matrix=b.tolist(),
        native_rigid_projection_used=not np.array_equal(a, b), local_geometry_repaired=False,
        original_local_rows_and_faces_preserved=True, raw_world_exact=True, effective_loader_exact=True,
        component_boundary_signature_preserved=surface_signature(loaded, loaded_f) == surface_signature(effective, sf),
        native_fp32_copy_exact=True, native_fp32_orientation_preserved=True,
        native_metric_quantization_max_error=error, metric_roundtrip_limit=ROUNDTRIP_LIMIT,
        canonical_vertex_count=len(sv), canonical_face_count=len(sf),
        canonical_unused_vertices_preserved=int(original.diagnostics["unused_vertices_preserved"]),
        single_geometry_single_instance_policy=True, embedding_verified=False,
        reconstruction_accuracy_verified=False, adoption_authorized=False)
    return _owned(native), _owned(native_f), proof


def camera_roundtrip(source_vertices, source_faces, rotations, translations,
                     saved_poses, native_vertices, native_faces):
    """Full original timeline and ALL canonical vertices, including orphans."""
    v = _array(source_vertices, "float64", (None, 3), "source vertices")
    f = _array(source_faces, "int64", (None, 3), "source faces")
    SurfaceIdentity(v, f)
    r = _array(rotations, "float64", (None, 3, 3), "rotations")
    t = _array(translations, "float64", (len(r), 3), "translations")
    poses = _array(saved_poses, "float32", (len(r), 4, 4), "saved native poses")
    nv = _array(native_vertices, "float32", v.shape, "native vertices")
    nf = _array(native_faces, "int64", f.shape, "native faces")
    if not len(r) or not np.array_equal(nf, f):
        raise ValueError("Full timeline and original face lineage required")
    error = 0.
    for i in range(len(r)):
        m = np.eye(4); m[:3, :3], m[:3, 3] = r[i], t[i]
        _matrix(m, "original frame pose")
        p = poses[i].astype(np.float64)
        _matrix(p, "saved represented frame pose")
        before = v @ r[i].T + t[i]
        after = nv.astype(np.float64) @ p[:3, :3].T + p[:3, 3]
        current = float(np.max(np.linalg.norm(after.astype(np.float64) - before, axis=1)))
        if not np.isfinite(current):
            raise ValueError("Nonfinite camera-space represented roundtrip")
        error = max(error, current)
    if error > ROUNDTRIP_LIMIT:
        raise ValueError("Full camera-space surface/pose pair exceeds inherited roundtrip limit")
    return error

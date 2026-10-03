"""Exact GLB-surface and native-FP32 proofs at the official packing boundary.

Pure arrays only: no files, official imports, models, alignment or tolerances.
Packing must preserve the decoded GLB surface exactly; the separately frozen
native geometry must equal that surface's declared FP32 quantization. These
are different claims and neither can replace the other.
"""
from __future__ import annotations


def _mesh_arrays(vertices, faces, *, vertex_dtype=None, name="mesh"):
    import numpy as np

    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError(f"{name}: explicit unmasked arrays required")
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or vertex_dtype is not None and vertices.dtype != np.dtype(vertex_dtype)
            or vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or not np.isfinite(vertices).all()
            or faces.dtype != np.dtype("int64") or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces)
            or np.min(faces) < 0 or np.max(faces) >= len(vertices)):
        raise ValueError(f"{name}: finite exact floating vertices and int64 triangle indices required")
    return vertices, faces


def _nonzero_triangles(points):
    """Classify exact dyadic triangles, with a fast conservative FP64 filter.

    A comfortably nonzero determinant certifies non-collinearity. Cancellation,
    subnormal products/differences and overflow instead use exact rational
    arithmetic on the original floating-point coordinates. An underflowed cross
    product therefore never silently deletes a real surface.
    """
    import numpy as np
    from fractions import Fraction

    points = np.asarray(points, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        a = points[:, 1] - points[:, 0]
        b = points[:, 2] - points[:, 0]
        left = a[:, [1, 2, 0]] * b[:, [2, 0, 1]]
        right = a[:, [2, 0, 1]] * b[:, [1, 2, 0]]
        determinant = left - right
        bound = 64 * np.finfo(np.float64).eps * np.maximum(np.abs(left), np.abs(right))
    tiny = np.finfo(np.float64).tiny
    normal_or_zero = lambda value: (value == 0) | (np.abs(value) >= tiny)
    safe = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            & normal_or_zero(a).all(axis=1) & normal_or_zero(b).all(axis=1))
    products_safe = (np.isfinite(left) & np.isfinite(right) & np.isfinite(determinant)
                     & normal_or_zero(left) & normal_or_zero(right)
                     & normal_or_zero(determinant)
                     & ~((left == 0) & (a[:, [1, 2, 0]] != 0) & (b[:, [2, 0, 1]] != 0))
                     & ~((right == 0) & (a[:, [2, 0, 1]] != 0) & (b[:, [1, 2, 0]] != 0)))
    nonzero = safe & np.any(products_safe & (np.abs(determinant) > bound), axis=1)
    repeated = (np.all(points[:, 0] == points[:, 1], axis=1)
                | np.all(points[:, 1] == points[:, 2], axis=1)
                | np.all(points[:, 2] == points[:, 0], axis=1))
    for index in np.flatnonzero(~nonzero & ~repeated):
        exact = [[Fraction.from_float(float(value)) for value in corner] for corner in points[index]]
        u = [exact[1][axis] - exact[0][axis] for axis in range(3)]
        v = [exact[2][axis] - exact[0][axis] for axis in range(3)]
        nonzero[index] = any(u[i] * v[j] != u[j] * v[i] for i, j in ((1, 2), (2, 0), (0, 1)))
    return nonzero


def canonical_oriented_triangles(vertices, faces):
    """Exact cyclic-oriented nonzero triangle multiset, without geometry repair.

    Permuting rows/vertices or rotating a face's three corners is equivalent.
    Reversing winding, deleting/duplicating a positive-area triangle or moving
    any of its coordinates is not. Only exactly zero-area faces are ignored.
    """
    import numpy as np

    vertices, faces = _mesh_arrays(vertices, faces)
    points = vertices.astype(np.float64)[faces]
    points = points[_nonzero_triangles(points)]
    if not len(points):
        raise ValueError("At least one original nonzero triangle required")
    choices = np.stack((points, np.roll(points, 1, axis=1), np.roll(points, 2, axis=1)), axis=1).reshape(-1, 3, 9)
    order = np.lexsort(tuple(choices[:, :, column] for column in range(8, -1, -1)), axis=1)
    rows = choices[np.arange(len(points)), order[:, 0]]
    return rows[np.lexsort(tuple(rows[:, column] for column in range(8, -1, -1)))]


def nonzero_triangle_mask(vertices, faces):
    """Exact original face activity, for strict official padding validation."""
    vertices, faces = _mesh_arrays(vertices, faces)
    return _nonzero_triangles(vertices.astype("float64")[faces])


def _quantized_mesh(vertices, faces, *, name):
    import numpy as np

    if np.any(np.abs(vertices) > np.finfo(np.float32).max):
        raise ValueError(f"{name}: GLB coordinates overflow native FP32")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        quantized = vertices.astype(np.float32)
    if not np.isfinite(quantized).all():
        raise ValueError(f"{name}: nonfinite native FP32 quantization")
    original_active = _nonzero_triangles(vertices[faces])
    quantized_active = _nonzero_triangles(quantized.astype(np.float64)[faces])
    if not np.array_equal(original_active, quantized_active):
        raise ValueError(f"{name}: FP32 quantization collapses or creates a nonzero triangle")
    return quantized


def verify_exact_dual_surfaces(native_vertices, native_faces, raw_vertices, raw_faces, packed_vertices, packed_faces):
    """Prove raw GLB preservation AND exact independently declared native cast.

    Native vertices are strictly FP32; decoded raw and packed vertices strictly
    FP64; all faces int64. Original nonzero triangles must survive quantization.
    No epsilon matching, mesh simplification, scale/origin fit or fallback.
    """
    import numpy as np

    native_vertices, native_faces = _mesh_arrays(native_vertices, native_faces, vertex_dtype="float32", name="native")
    raw_vertices, raw_faces = _mesh_arrays(raw_vertices, raw_faces, vertex_dtype="float64", name="raw GLB")
    packed_vertices, packed_faces = _mesh_arrays(packed_vertices, packed_faces, vertex_dtype="float64", name="packed")
    raw = canonical_oriented_triangles(raw_vertices, raw_faces)
    packed = canonical_oriented_triangles(packed_vertices, packed_faces)
    if not np.array_equal(raw, packed):
        raise ValueError("Packed oriented triangle geometry differs from exact raw GLB surface")
    quantized_raw = _quantized_mesh(raw_vertices, raw_faces, name="raw GLB")
    quantized_packed = _quantized_mesh(packed_vertices, packed_faces, name="packed")
    native = canonical_oriented_triangles(native_vertices, native_faces)
    if (not np.array_equal(native, canonical_oriented_triangles(quantized_raw, raw_faces))
            or not np.array_equal(native, canonical_oriented_triangles(quantized_packed, packed_faces))):
        raise ValueError("Native oriented triangles differ from exact raw/packed FP32 quantization")
    if len(native) != len(raw) or len(packed) != len(raw):
        raise ValueError("Exact original nonzero triangle counts must remain unchanged")
    return dict(raw_oriented_triangles_exact=True, native_fp32_quantization_exact=True,
                original_nonzero_triangles_preserved=True, oriented_triangles_exact=True,
                source_active_faces=len(raw), active_faces=len(packed), native_active_faces=len(native),
                raw_exact_degenerate_faces=len(raw_faces) - len(raw),
                native_exact_degenerate_faces=len(native_faces) - len(native),
                packed_exact_degenerate_faces=len(packed_faces) - len(packed),
                fp32_quantization_changed_face_activity=False,
                additional_scale_or_alignment=False, geometry_repaired=False)

"""Identify only collapsed mesh triangles; never modify geometry or topology.

Return original face indices for a non-padding render view. The caller keeps
the complete official padded arrays unchanged and separately validates closure,
winding and signed volume. No decimation, component deletion, hole filling,
normal inversion, scale fitting or physical/contact metric is implemented here.
"""

from __future__ import annotations

import numpy as np


_ROUNDOFF_FACTOR = 32


def normalize_degenerate_faces(vertices, faces) -> tuple[np.ndarray, dict]:
    """Return (active_original_face_indices, JSON diagnostics), without mutation.

    Positions/order, scale, meaningful surfaces and nested cavity orientation
    are preserved. Integer-index padding and exactly collinear triangles are
    excluded from the render view. Numerical collapse uses float64 arithmetic,
    not float32 mesh-format precision: after normalization by maximum extent,
    ||e1 x e2|| <= 32*eps64*||e1||*||e2|| is the roundoff-level criterion.

    This relative error bound applies per triangle, so a tiny well-shaped
    component is retained even when its area is below eps64*mesh_extent**2.
    Thin triangles above the machine-level angular threshold are also retained.
    The factor 32 conservatively covers coordinate subtraction, normalization,
    products/differences in the cross product and norm evaluation. It is fixed,
    never adjusted from challenge geometry, score or volume. The threshold is
    not a licence to drop meaningful small components or repair non-manifolds.
    """
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Explicit vertex and face arrays required, not masked arrays")
    try:
        positions, indices = np.asarray(vertices), np.asarray(faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid mesh arrays") from exc
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or positions.dtype.kind not in "iuf" or not np.isfinite(positions).all()):
        raise ValueError("vertices must be finite real nonempty (V, 3)")
    if (indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"):
        raise ValueError("faces must have shape (F, 3) and integer indices")
    if (indices < 0).any() or (indices >= len(positions)).any():
        raise ValueError("faces contain invalid vertex indices")
    positions = positions.astype(np.float64, copy=False)
    indices = indices.astype(np.int64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        extents = positions.max(axis=0) - positions.min(axis=0)
    if not np.isfinite(extents).all():
        raise ValueError("Vertex extent exceeds finite float64 range")
    extent = float(extents.max())
    epsilon = float(np.finfo(np.float64).eps)
    relative_tolerance = _ROUNDOFF_FACTOR * epsilon
    repeated = ((indices[:, 0] == indices[:, 1]) | (indices[:, 0] == indices[:, 2])
                | (indices[:, 1] == indices[:, 2]))
    if extent == 0:
        cross_norm = np.zeros(len(indices), dtype=np.float64)
        edge_product = np.zeros(len(indices), dtype=np.float64)
    else:
        # Normalize edge differences rather than absolute positions to avoid
        # avoidable cancellation for translated scenes. Scale never changes
        # the caller's stored geometry or the returned original indices.
        triangles = positions[indices]
        edge1 = (triangles[:, 1] - triangles[:, 0]) / extent
        edge2 = (triangles[:, 2] - triangles[:, 0]) / extent
        cross = np.cross(edge1, edge2)
        cross_norm = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
        norm1 = np.hypot(np.hypot(edge1[:, 0], edge1[:, 1]), edge1[:, 2])
        norm2 = np.hypot(np.hypot(edge2[:, 0], edge2[:, 1]), edge2[:, 2])
        edge_product = norm1 * norm2
        if not np.isfinite(cross_norm).all() or not np.isfinite(edge_product).all():
            raise ValueError("Triangle arithmetic exceeds finite float64 range")
    exact_zero = ~repeated & (cross_norm == 0)
    numerical = ~repeated & ~exact_zero & (cross_norm <= relative_tolerance * edge_product)
    removed = repeated | exact_zero | numerical
    active = np.flatnonzero(~removed).astype(np.int64, copy=False)
    removed_ids = np.flatnonzero(removed)
    numerical_ids = np.flatnonzero(numerical)
    ratios = np.divide(cross_norm, edge_product, out=np.zeros_like(cross_norm), where=edge_product > 0)
    diagnostics = {
        "schema": "world-reward-collapsed-triangle-view-v1",
        "input_vertices": len(positions), "input_faces": len(indices), "active_faces": len(active),
        "excluded_faces": int(np.count_nonzero(removed)),
        "repeated_index_faces": int(np.count_nonzero(repeated)),
        "exact_zero_area_faces": int(np.count_nonzero(exact_zero)),
        "numerically_collapsed_faces": int(np.count_nonzero(numerical)),
        "extent_max_input_units": extent, "arithmetic_dtype": "float64", "epsilon": epsilon,
        "relative_cross_product_tolerance": relative_tolerance,
        "criterion": "norm(cross(e1,e2)) <= 32*eps64*norm(e1)*norm(e2); edges divided by maximum mesh extent",
        "excluded_original_face_indices": removed_ids.tolist(),
        "numerically_collapsed_original_face_indices": numerical_ids.tolist(),
        "maximum_excluded_normalized_double_area": float(cross_norm[removed].max()) if removed.any() else 0.,
        "minimum_active_relative_cross_product": float(ratios[~removed].min()) if len(active) else None,
        "input_arrays_modified": False, "components_selected_or_removed": False,
        "closure_winding_volume_verified": False,
    }
    return active, diagnostics

"""Full-triangle numerical solid queries AFTER a separate embedding certificate.

This helper neither proves closure/embedding nor repairs, samples or removes any
geometry. A winding near an integer alone is NOT an embedding certificate.
"""
import math
import time

import numpy as np

from world_reward.cross_surface import _winding

MAX_TRIANGLES_PER_CHUNK = 32768
WINDING_INTEGER_TOLERANCE = 1e-6  # Same integration guard as original _winding.


def _check_deadline(deadline):
    if deadline is not None:
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("Finite absolute monotonic deadline required")
        if time.monotonic() >= deadline:
            raise TimeoutError("Full original surface query deadline exceeded")


def _inputs(point, vertices, faces, tolerance):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Finite positive metric tolerance required")
    if any(np.ma.isMaskedArray(value) for value in (point, vertices, faces)):
        raise ValueError("Masked point or original geometry forbidden")
    p, v, f = map(np.asarray, (point, vertices, faces))
    floats = (np.dtype("float32"), np.dtype("float64"))
    if p.dtype not in floats or p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("One finite float32/64 point required")
    if v.dtype not in floats or v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("Full finite float32/64 original vertex matrix required")
    if (f.dtype.kind not in "iu" or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError("Nonempty valid original integer triangle indices required")
    # Exact same coordinate-conditioning guard as validate_closed_mesh. Including
    # the query point also prevents overflow in its relative distance arithmetic.
    magnitude = max(1., float(np.abs(v).max()), float(np.abs(p).max()))
    if 128 * np.finfo(np.float64).eps * magnitude > tolerance / 8:
        raise ValueError("Coordinates insufficiently conditioned for declared tolerance")
    return p.astype(np.float64, copy=False), v.astype(np.float64, copy=False), f.astype(np.int64, copy=False)


def _triangle_distances(point, triangles, tolerance):
    """Vectorized continuous projection or all clipped segment regions.

    Each triangle is examined, including face interiors and all three edges.
    Uses the original _point_triangle half-plane predicates, not a barycentric
    epsilon or vertex-only proxy. Nondegenerate conditioning matches the closed
    surface validator, but this routine does not validate manifold incidence.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edges = np.stack((b - a, c - b, a - c), axis=1)
    lengths = np.linalg.norm(edges, axis=-1)
    normal = np.cross(edges[:, 0], -edges[:, 2])
    normal_length = np.linalg.norm(normal, axis=1)
    altitude = normal_length / lengths.max(axis=1)
    if (not np.isfinite(lengths).all() or not np.isfinite(normal_length).all()
            or not np.isfinite(altitude).all() or np.any(lengths <= 8 * tolerance)
            or np.any(altitude <= 8 * tolerance)):
        raise ValueError("Degenerate/ill-conditioned original triangle; no face removal")
    denominator = np.einsum("ij,ij->i", normal, normal)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Original normal arithmetic insufficiently conditioned")
    projected = point - normal * (np.einsum("ij,ij->i", point - a, normal) / denominator)[:, None]
    starts = np.stack((a, b, c), axis=1)
    tests = np.einsum("nkj,nj->nk", np.cross(edges, projected[:, None] - starts), normal)
    projection_inside = (tests >= 0).all(axis=1)
    projection_squared = np.einsum("ij,ij->i", point - projected, point - projected)
    edge_squared = np.einsum("nkj,nkj->nk", edges, edges)
    parameter = np.clip(np.einsum("nkj,nkj->nk", point - starts, edges) / edge_squared, 0., 1.)
    closest_edges = starts + parameter[..., None] * edges
    delta = point - closest_edges
    segment_squared = np.einsum("nkj,nkj->nk", delta, delta)
    squared = segment_squared.min(axis=1)
    squared = np.minimum(squared, np.where(projection_inside, projection_squared, np.inf))
    if not np.isfinite(squared).all() or np.any(squared < 0):
        raise ValueError("Nonfinite original continuous point-triangle distance")
    return np.sqrt(squared)


def point_surface_query(point, vertices, faces, *, expected_state,
        tolerance_m=1e-8, deadline=None):
    """Query a separately certified original solid; numerical, never exact proof.

    Invalid inputs raise ValueError; an expired deadline raises TimeoutError.
    Geometrically ambiguous/hit/wrong-state queries return status='fail'. Every
    original triangle contributes both to distance and the original winding.
    """
    if expected_state not in ("inside", "outside") or type(expected_state) is not str:
        raise ValueError("Expected state must be inside or outside")
    _check_deadline(deadline)
    p, v, f = _inputs(point, vertices, faces, tolerance_m)
    triangles = v[f]
    minimum, examined = math.inf, 0
    for first in range(0, len(triangles), MAX_TRIANGLES_PER_CHUNK):
        _check_deadline(deadline)
        chunk = triangles[first:first + MAX_TRIANGLES_PER_CHUNK]
        minimum = min(minimum, float(_triangle_distances(p, chunk, tolerance_m).min()))
        examined += len(chunk)
    _check_deadline(deadline)
    winding = _winding(p, triangles, tolerance_m)  # Original implementation, full original faces.
    _check_deadline(deadline)
    expected = 1. if expected_state == "inside" else 0.
    state_matches = winding is not None and abs(winding - expected) <= WINDING_INTEGER_TOLERANCE
    passed = bool(math.isfinite(minimum) and minimum > tolerance_m and state_matches)
    return dict(status="pass" if passed else "fail", expected_state=expected_state,
        winding=winding, continuous_surface_distance_m=minimum,
        full_original_faces_retained=True, triangles_examined=examined,
        tolerance_m=float(tolerance_m), exact_arithmetic_proof=False,
        embedding_verified=False, requires_prior_embedding_certificate=True)

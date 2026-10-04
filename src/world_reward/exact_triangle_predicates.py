"""Certify non-collinearity of stored float64 triangles, never repair a mesh.

Outward-rounded binary64 intervals cheaply certify a nonzero projected
orientation. Ambiguous faces use exact dyadic rational arithmetic on the
original coordinates. No length/area tolerance, global extent, coordinate
normalization, component selection or geometry mutation is involved.

This is a precision-only predicate, not a topology, embedding, numerical
conditioning, physical fidelity or reconstruction-quality certificate.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


_SCHEMA = "world-reward-exact-triangle-nondegeneracy-v1"
_PLANES = ((0, 1), (1, 2), (2, 0))


class ExactTriangleDegeneracyError(ValueError):
    """The complete geometry is rejected; no filtered face view is returned."""


def _require_binary64_subnormals() -> None:
    """Fail closed if nonstandard flush-to-zero arithmetic invalidates filters."""
    info = np.finfo(np.float64)
    with np.errstate(under="ignore"):
        half_normal = np.multiply(np.float64(info.tiny), np.float64(.5))
        smallest = np.nextafter(np.float64(0.), np.float64(1.))
        preserved = np.multiply(smallest, np.float64(1.))
    if (info.nmant != 52 or info.maxexp != 1024 or info.minexp != -1022
            or not half_normal > 0 or not smallest > 0 or preserved != smallest):
        raise RuntimeError("Require IEEE-754 binary64 with gradual underflow")


def _outward(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enclose an IEEE-754 operation's result, including under/overflow.

    One representable neighbor on either side of the rounded value encloses
    the exact real result. NaN propagates to an ambiguous interval and thus to
    the exact fallback; it is never treated as a successful certificate.
    """
    return np.nextafter(value, -np.inf), np.nextafter(value, np.inf)


def _difference(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _outward(np.subtract(a, b))


def _product(a: tuple[np.ndarray, np.ndarray],
             b: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    products = (np.multiply(a[0], b[0]), np.multiply(a[0], b[1]),
                np.multiply(a[1], b[0]), np.multiply(a[1], b[1]))
    lower = np.nextafter(np.minimum.reduce(products), -np.inf)
    upper = np.nextafter(np.maximum.reduce(products), np.inf)
    return lower, upper


def _orientation_interval(edge1: tuple[np.ndarray, np.ndarray],
                          edge2: tuple[np.ndarray, np.ndarray],
                          plane: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    i, j = plane
    first = _product((edge1[0][:, i], edge1[1][:, i]),
                     (edge2[0][:, j], edge2[1][:, j]))
    second = _product((edge1[0][:, j], edge1[1][:, j]),
                      (edge2[0][:, i], edge2[1][:, i]))
    lower = np.nextafter(np.subtract(first[0], second[1]), -np.inf)
    upper = np.nextafter(np.subtract(first[1], second[0]), np.inf)
    return lower, upper


def _exact_noncollinear(triangle: np.ndarray) -> bool:
    # Every finite binary64 value is an exact dyadic rational. Fraction does
    # not infer or recover pre-export geometry: these are the stored values.
    points = [[Fraction.from_float(float(x)) for x in point]
              for point in triangle]
    edge1 = [points[1][i] - points[0][i] for i in range(3)]
    edge2 = [points[2][i] - points[0][i] for i in range(3)]
    return any(edge1[i] * edge2[j] != edge1[j] * edge2[i]
               for i, j in _PLANES)


def validate_exact_triangle_non_degeneracy(vertices, faces) -> dict:
    """Return scalar diagnostics only if every original triangle is nonzero.

    Input positions must already be finite float64, with integer triangle
    indices. The arrays are never cast, reordered, welded, normalized, written
    or reduced. Repeated indices or three exactly collinear stored positions
    reject the *whole* geometry. All ambiguous numerical predicates are
    resolved exactly, rather than accepted/rejected using an area threshold.

    Positive-area but arbitrarily skinny triangles are mathematically valid
    here. A downstream numerical-readiness gate may independently reject
    them; this function cannot certify readiness of QEM/rendering/export.
    """
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Explicit unmasked geometry arrays required")
    try:
        v, f = np.asarray(vertices), np.asarray(faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid geometry arrays") from exc
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3
            or v.dtype.kind != "f" or v.dtype.itemsize != 8
            or not np.isfinite(v).all()):
        raise ValueError("Positions must already be finite nonempty float64 Vx3")
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or int(f.min()) < 0
            or int(f.max()) >= len(v)):
        raise ValueError("Require nonempty valid integer Fx3 triangle indices")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2])
              | (f[:, 1] == f[:, 2])):
        raise ExactTriangleDegeneracyError(
            "Repeated face indices: complete geometry rejected, no face deletion")

    _require_binary64_subnormals()
    certified = np.zeros(len(f), dtype=bool)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        origins = v[f[:, 0]]
        edge1 = _difference(v[f[:, 1]], origins)
        edge2 = _difference(v[f[:, 2]], origins)
        for plane in _PLANES:
            lower, upper = _orientation_interval(edge1, edge2, plane)
            # Any invalid/NaN interval is ambiguous, never a sign certificate.
            certified |= (lower <= upper) & ((lower > 0) | (upper < 0))
            if certified.all():
                break

    fallback_faces = np.flatnonzero(~certified)
    for face_index in fallback_faces:
        if not _exact_noncollinear(v[f[face_index]]):
            raise ExactTriangleDegeneracyError(
                "Exactly collinear stored positions: complete geometry rejected, no repair")

    return {
        "schema": _SCHEMA,
        "status": "pass",
        "input_vertices": len(v),
        "input_faces": len(f),
        "arithmetic_input_dtype": "float64",
        "interval_certified_faces": int(np.count_nonzero(certified)),
        "exact_dyadic_fallback_faces": len(fallback_faces),
        "exact_noncollinearity_proven": True,
        "area_or_length_tolerance_used": False,
        "input_arrays_modified": False,
        "faces_deleted_or_reordered": False,
        "components_selected_or_removed": False,
        "original_coordinate_cast_or_normalization": False,
        "pre_export_geometry_recovered": False,
        "topology_verified": False,
        "embedding_verified": False,
        "volume_or_fidelity_verified": False,
        "backend_numerical_readiness_verified": False,
        "challenge_performance_verified": False,
    }

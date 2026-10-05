"""Exact sign of original oriented-normal dot products, never a geometry repair.

An outward-rounded binary64 interval filters unambiguous signs. All other rows
use exact dyadic Fraction arithmetic on the supplied original coordinates.
Float32 promotion is exact; no normalization, epsilon, scale fit or face
selection is performed. A false decision means the exact dot is <=0, including
degenerate or orthogonal normals. This alone certifies neither an embedding nor
physical fidelity or reconstruction accuracy.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np

from world_reward.exact_triangle_predicates import (
    _require_binary64_subnormals, _difference, _product, _orientation_interval,
)

SCHEMA = "world_reward.exact_normal_dot.v1"


def _triangles(value, name):
    if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value):
        raise ValueError(f"{name}: original unmasked NumPy triangles required")
    if (value.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or value.ndim != 3 or value.shape[1:] != (3, 3)
            or not np.isfinite(value).all()):
        raise ValueError(f"{name}: finite original F32/F64 [N,3,3] required")
    return value


def _normal_interval(triangles):
    original = triangles.astype(np.float64, copy=False)
    edge1 = _difference(original[:, 1], original[:, 0])
    edge2 = _difference(original[:, 2], original[:, 0])
    projected = [_orientation_interval(edge1, edge2, p) for p in ((1, 2), (2, 0), (0, 1))]
    return np.column_stack([p[0] for p in projected]), np.column_stack([p[1] for p in projected])


def _exact_positive(before, after):
    normals = []
    for triangle in (before, after):
        points = [[Fraction.from_float(float(x)) for x in p] for p in triangle]
        a = [points[1][k] - points[0][k] for k in range(3)]
        b = [points[2][k] - points[0][k] for k in range(3)]
        normals.append([a[i] * b[j] - a[j] * b[i] for i, j in ((1, 2), (2, 0), (0, 1))])
    return sum(x * y for x, y in zip(*normals)) > 0


def positive_normal_dot(before, after):
    """Return immutable exact-positive bool[N] decisions and scalar diagnostics.

    Both inputs must contain the same number of original triangles. Dtypes may
    differ (e.g. native F64 versus independently stored F32). Arrays and their
    order remain untouched. Empty batches return an empty result, not an
    acceptance decision. Fraction fallback has no approximate success path.
    """
    before, after = _triangles(before, "before"), _triangles(after, "after")
    if before.shape != after.shape:
        raise ValueError("Original paired triangle batch shapes must agree")
    _require_binary64_subnormals()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        left, right = _normal_interval(before), _normal_interval(after)
        terms = _product(left, right)
        # Two explicit interval sums; no unspecified reduction/FMA order.
        lower = np.nextafter(terms[0][:, 0] + terms[0][:, 1], -np.inf)
        upper = np.nextafter(terms[1][:, 0] + terms[1][:, 1], np.inf)
        lower = np.nextafter(lower + terms[0][:, 2], -np.inf)
        upper = np.nextafter(upper + terms[1][:, 2], np.inf)
    finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    positive = finite & (lower > 0)
    nonpositive = finite & (upper <= 0)
    fallback = ~(positive | nonpositive)
    decisions = positive.copy()
    for index in np.flatnonzero(fallback):
        decisions[index] = _exact_positive(before[index], after[index])
    decisions = np.frombuffer(decisions.tobytes(), dtype=np.bool_)
    return decisions, dict(schema=SCHEMA, triangles=len(before),
        interval_positive_rows=int(np.count_nonzero(positive)),
        interval_nonpositive_rows=int(np.count_nonzero(nonpositive)),
        exact_dyadic_fallback_rows=int(np.count_nonzero(fallback)),
        positive_rows=int(np.count_nonzero(decisions)),
        nonpositive_rows=int(len(before) - np.count_nonzero(decisions)),
        original_arrays_unchanged=True, area_or_length_tolerance_used=False,
        coordinate_normalization_performed=False, reconstruction_accuracy_verified=False)

"""Bounded exact transverse-intersection witnesses for two IEEE triangles.

This independent diagnostic uses only stdlib rational arithmetic. It certifies
one strict interior/interior transverse witness, not mesh embedding, separation,
adjacency, touching, physical contact or force closure. A negative result is
NOT a full separation proof. No tolerance, coordinate rounding or mesh repair
is used. Consumers must independently bind the original vertices/face indices.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from numbers import Real
import struct
import time


SCHEMA = "world-reward-exact-ieee-triangle-witness-v1"
MAX_FRACTION_BITS = 4096
MAX_OPERATIONS = 10000
MAX_SECONDS = 5.0


class WitnessBudgetError(RuntimeError):
    """No witness is returned after any declared arithmetic/time budget fails."""


class _Arithmetic:
    def __init__(self, seconds, bits, operations):
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 < seconds <= MAX_SECONDS
                or type(bits) is not int or not 1 <= bits <= MAX_FRACTION_BITS
                or type(operations) is not int or not 1 <= operations <= MAX_OPERATIONS):
            raise ValueError("Positive bounded seconds, fraction bits and operations required")
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.max_bits, self.max_operations = bits, operations
        self.operations, self.peak_bits = 0, 0

    def deadline_check(self):
        if time.monotonic() > self.deadline:
            raise WitnessBudgetError("Exact witness deadline exceeded; no witness")

    def checked(self, value):
        self.deadline_check()
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
        self.peak_bits = max(self.peak_bits, bits)
        if bits > self.max_bits:
            raise WitnessBudgetError("Exact fraction bit budget exceeded; no witness")
        return value

    def op(self, operation, first, last):
        self.checked(first); self.checked(last)
        self.operations += 1
        if self.operations > self.max_operations:
            raise WitnessBudgetError("Exact arithmetic operation budget exceeded; no witness")
        # Each operand is bounded before arithmetic; a single temporary result
        # is at most ~2*max_bits+1 bits, then checked before any subsequent use.
        if operation == "+": result = first + last
        elif operation == "-": result = first - last
        elif operation == "*": result = first * last
        elif operation == "/": result = first / last
        else: raise AssertionError("Internal exact arithmetic operation")
        return self.checked(result)

    def add(self, a, b): return self.op("+", a, b)
    def sub(self, a, b): return self.op("-", a, b)
    def mul(self, a, b): return self.op("*", a, b)
    def div(self, a, b): return self.op("/", a, b)

    def compare(self, a, b):
        numerator = self.sub(a, b).numerator
        return (numerator > 0) - (numerator < 0)


def _triangle(value, arithmetic):
    if isinstance(value, (str, bytes, dict)):
        raise ValueError("Exactly three original three-coordinate points required")
    try:
        if len(value) != 3: raise ValueError("Exactly three triangle vertices required")
        rows = [value[i] for i in range(3)]
        if any(isinstance(row, (str, bytes, dict)) or len(row) != 3 for row in rows):
            raise ValueError("Exactly three coordinates per original vertex required")
    except (TypeError, IndexError) as error:
        raise ValueError("Exactly three original three-coordinate points required") from error
    result, encoded = [], bytearray()
    for row in rows:
        point = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError("Finite real IEEE coordinates, never bool/masked values, required")
            try: coordinate = float(item)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError("Finite IEEE binary64 coordinate required") from error
            if not math.isfinite(coordinate) or item != coordinate:
                raise ValueError("Finite losslessly representable IEEE binary64 coordinates required")
            encoded.extend(struct.pack("!d", coordinate))
            point.append(arithmetic.checked(Fraction.from_float(coordinate)))
        result.append(tuple(point))
    return tuple(result), hashlib.sha256(b"ieee754-binary64-be:(3,3):" + encoded).hexdigest()


def _sub(a, b, g): return tuple(g.sub(x, y) for x, y in zip(a, b))


def _dot(a, b, g):
    return g.add(g.add(g.mul(a[0], b[0]), g.mul(a[1], b[1])), g.mul(a[2], b[2]))


def _cross(a, b, g):
    return tuple(g.sub(g.mul(a[i], b[j]), g.mul(a[j], b[i]))
        for i, j in ((1, 2), (2, 0), (0, 1)))


def _normal(triangle, g):
    result = _cross(_sub(triangle[1], triangle[0], g), _sub(triangle[2], triangle[0], g), g)
    if not any(component.numerator for component in result):
        raise ValueError("Exactly nondegenerate original triangles required")
    return result


def _clip(triangle, signed, g):
    points = [triangle[i] for i in range(3) if signed[i].numerator == 0]
    for i in range(3):
        j = (i + 1) % 3
        if (signed[i].numerator < 0 < signed[j].numerator
                or signed[j].numerator < 0 < signed[i].numerator):
            ratio = g.div(signed[i], g.sub(signed[i], signed[j]))
            delta = _sub(triangle[j], triangle[i], g)
            points.append(tuple(g.add(triangle[i][k], g.mul(ratio, delta[k])) for k in range(3)))
    points = tuple(dict.fromkeys(points))
    if len(points) != 2:
        raise ArithmeticError("Strict exact plane cut did not produce two endpoints")
    return points


def _barycentric(point, triangle, normal, g):
    drop = next(i for i, item in enumerate(normal) if item.numerator)
    i, j = (index for index in range(3) if index != drop)
    u, v, w = (_sub(row, triangle[0], g) for row in (triangle[1], triangle[2], point))
    determinant = g.sub(g.mul(u[i], v[j]), g.mul(u[j], v[i]))
    beta = g.div(g.sub(g.mul(w[i], v[j]), g.mul(w[j], v[i])), determinant)
    gamma = g.div(g.sub(g.mul(u[i], w[j]), g.mul(u[j], w[i])), determinant)
    alpha = g.sub(g.sub(Fraction(1), beta), gamma)
    weights = (alpha, beta, gamma)
    if not all(item.numerator > 0 for item in weights):
        raise ArithmeticError("Exact overlap midpoint is not strictly inside both triangles")
    if g.add(g.add(alpha, beta), gamma) != 1:
        raise ArithmeticError("Exact barycentric normalization failed")
    for coordinate in range(3):
        value = g.add(g.add(g.mul(alpha, triangle[0][coordinate]), g.mul(beta, triangle[1][coordinate])),
            g.mul(gamma, triangle[2][coordinate]))
        if g.compare(value, point[coordinate]):
            raise ArithmeticError("Exact witness coordinate reconstruction failed")
    return weights


def _rational(value, g):
    g.checked(value)
    # Decimal strings preserve exact integers in JSON consumers with 53-bit
    # numeric types. Denominators are positive and reduced by Fraction.
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def exact_triangle_witness(triangle_a, triangle_b, *, budget_seconds=2.0,
        max_fraction_bits=MAX_FRACTION_BITS, max_operations=MAX_OPERATIONS):
    """Return a rational strict transverse witness, or only 'no strict witness'.

    Accepts two indexable 3x3 arrays/lists of losslessly binary64-representable
    real coordinates, without importing NumPy. Finite float32 values promoted
    to binary64 remain exact. Input indices/adjacency are deliberately absent.
    Budget/input/invariant errors raise, never yielding a positive result.
    """
    g = _Arithmetic(budget_seconds, max_fraction_bits, max_operations)
    a, hash_a = _triangle(triangle_a, g)
    b, hash_b = _triangle(triangle_b, g)
    na, nb = _normal(a, g), _normal(b, g)
    da = tuple(_dot(_sub(point, b[0], g), nb, g) for point in a)
    db = tuple(_dot(_sub(point, a[0], g), na, g) for point in b)
    direction = _cross(na, nb, g)
    signs = lambda values: [(item.numerator > 0) - (item.numerator < 0) for item in values]
    base = dict(schema=SCHEMA, strict_transverse_witness=False, status="no_strict_witness",
        reason=None, coordinate_sha256=[hash_a, hash_b], coordinate_encoding="IEEE754-binary64-big-endian",
        plane_side_signs=[signs(da), signs(db)], exact_input_geometry=True,
        adjacency_evaluated=False, full_separation_proven=False, replacement_certificate=False,
        touching_certified=False, force_closure_verified=False, witness=None,
        budgets=dict(seconds=budget_seconds, fraction_bits=max_fraction_bits, operations=max_operations))

    def finish(reason, witness=None):
        base.update(reason=reason, witness=witness)
        if witness is not None:
            base.update(strict_transverse_witness=True, status="strict_transverse_witness")
        g.deadline_check()
        base.update(arithmetic_operations=g.operations, peak_fraction_bits=g.peak_bits,
            elapsed_seconds=time.monotonic() - g.started)
        return base

    if not any(item.numerator for item in direction):
        return finish("coplanar_planes" if not any(item.numerator for item in da) else "parallel_distinct_planes")
    if any(not (min(values) < 0 < max(values)) for values in base["plane_side_signs"]):
        return finish("no_strict_two_sided_plane_cuts")
    pa, pb = _clip(a, da, g), _clip(b, db, g)
    sa, sb = (tuple(_dot(point, direction, g) for point in points) for points in (pa, pb))
    if g.compare(sa[0], sa[1]) > 0: pa, sa = pa[::-1], sa[::-1]
    if g.compare(sb[0], sb[1]) > 0: pb, sb = pb[::-1], sb[::-1]
    lo = sa[0] if g.compare(sa[0], sb[0]) >= 0 else sb[0]
    hi = sa[1] if g.compare(sa[1], sb[1]) <= 0 else sb[1]
    if g.compare(lo, hi) >= 0:
        return finish("plane_cut_intervals_do_not_overlap_strictly")
    middle = g.div(g.add(lo, hi), Fraction(2))
    ratio = g.div(g.sub(middle, sa[0]), g.sub(sa[1], sa[0]))
    delta = _sub(pa[1], pa[0], g)
    point = tuple(g.add(pa[0][i], g.mul(ratio, delta[i])) for i in range(3))
    weights_a, weights_b = _barycentric(point, a, na, g), _barycentric(point, b, nb, g)
    residual_a = _dot(_sub(point, a[0], g), na, g)
    residual_b = _dot(_sub(point, b[0], g), nb, g)
    if residual_a.numerator or residual_b.numerator:
        raise ArithmeticError("Exact witness is not on both original planes")
    witness = dict(point=[_rational(item, g) for item in point],
        triangle_a_barycentric=[_rational(item, g) for item in weights_a],
        triangle_b_barycentric=[_rational(item, g) for item in weights_b],
        plane_residuals=[_rational(residual_a, g), _rational(residual_b, g)],
        normal_cross_squared=_rational(_dot(direction, direction, g), g),
        interval_overlap_projection=_rational(g.sub(hi, lo), g))
    return finish("strict_transverse_interior_intersection", witness)

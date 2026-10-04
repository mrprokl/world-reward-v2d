"""Prepared exact-source chart: one Sterbenz/zero choice, never origin search.

Sterbenz (Floating-Point Computation, 1974, theorem 4.3.1): subtraction
of same-sign binary floats is exact when their magnitudes differ by <=2.
The inverse and power-of-two range are checked separately, with no fallback.
Neither mesh quality nor arbitrary-translation numerical equivariance follows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from world_reward.mesh_conditioning import _hash, _points, _readonly

SCHEMA = "world-reward-mesh-conditioning-v2"
POLICY = MappingProxyType({
    "schema": SCHEMA, "bbox": "referenced vertices only",
    "origin": "per-axis bbox midpoint iff whole-source sufficient Sterbenz condition; else zero",
    "scale": "smallest finite positive power of two >= maximum bbox extent",
    "encode": "ldexp(x-origin,-exponent); bypass subtraction for zero origin",
    "decode": "origin+ldexp(x,exponent); bypass addition for zero origin",
    "verification": "all source rows exact inverse; failure never changes origin",
    "repair": False, "adopted": False,
})
POLICY_SHA256 = hashlib.sha256(json.dumps(
    dict(POLICY), sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def sterbenz_subtraction_certified(values, origin):
    """Exact-rational endpoint test; no rounded half/double boundary checks."""
    a = np.asarray(values)
    if (np.ma.isMaskedArray(values) or a.ndim != 1 or not len(a)
            or a.dtype != np.float64 or not np.isfinite(a).all()
            or type(origin) is not float or not math.isfinite(origin)):
        raise ValueError("Finite nonempty float64 axis and float origin required")
    if origin == 0:
        return True
    if not np.all(a > 0 if origin > 0 else a < 0):
        return False
    magnitude = np.abs(a)
    yn, yd = abs(origin).as_integer_ratio()
    lo, hi = float(magnitude.min()), float(magnitude.max())
    ln, ld = lo.as_integer_ratio()
    hn, hd = hi.as_integer_ratio()
    return 2*ln*yd >= yn*ld and hn*yd <= 2*yn*hd


def _transform(points, origin, exponent, inverse):
    p = _points(points)
    out = np.empty_like(p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for axis in range(3):
            x, o = p[:, axis], float(origin[axis])
            delta = np.ldexp(x, exponent) if inverse else (x if o == 0 else x-o)
            y = (delta if o == 0 else o+delta) if inverse else np.ldexp(delta, -exponent)
            lost = (x != 0) & (delta == 0) if inverse else (delta != 0) & (y == 0)
            if not np.isfinite(delta).all() or not np.isfinite(y).all() or lost.any():
                raise ValueError("Chart overflow or nonzero-to-zero underflow")
            out[:, axis] = y
    return _readonly(out)


@dataclass(frozen=True, eq=False)
class MeshConditioningChartV2:
    origin: np.ndarray
    scale: float
    scale_exponent: int
    referenced_indices: np.ndarray
    origin_modes: tuple[str, str, str]
    diagnostics: MappingProxyType

    def __post_init__(self):
        o, i = np.asarray(self.origin), np.asarray(self.referenced_indices)
        if (o.shape != (3,) or o.dtype != np.float64 or not np.isfinite(o).all()
                or type(self.scale_exponent) is not int or not -1074 <= self.scale_exponent <= 1023
                or type(self.scale) is not float or self.scale != math.ldexp(1., self.scale_exponent)
                or i.ndim != 1 or i.dtype != np.int64 or not len(i) or np.any(i < 0)
                or np.any(i[1:] <= i[:-1]) or type(self.origin_modes) is not tuple
                or len(self.origin_modes) != 3
                or any(m not in ("sterbenz_midpoint", "zero") for m in self.origin_modes)
                or any((m == "zero") != (o[j] == 0) for j, m in enumerate(self.origin_modes))):
            raise ValueError("Explicit finite immutable v2 chart required")
        evidence = dict(self.diagnostics)
        if any(type(v) not in (str, int, float, bool) for v in evidence.values()):
            raise ValueError("Scalar/hash evidence only")
        json.dumps(evidence, allow_nan=False)
        object.__setattr__(self, "origin", _readonly(o))
        object.__setattr__(self, "referenced_indices", _readonly(i))
        object.__setattr__(self, "diagnostics", MappingProxyType(evidence))

    def encode(self, vertices):
        return _transform(vertices, self.origin, self.scale_exponent, False)

    def decode(self, vertices):
        return _transform(vertices, self.origin, self.scale_exponent, True)


def prepare_conditioning_v2(vertices, faces) -> MeshConditioningChartV2:
    """Bind all rows, including orphans; choose each origin exactly once.

    Faces need valid indices, not geometric certification. No face/vertex changes,
    origin trials, rounding tolerance, clipping, physical rescale or topology QA.
    Power-of-two unit equivariance is limited to exactly representable operations.
    """
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Unmasked source arrays required")
    source = np.asarray(vertices)
    v, f = _points(source), np.asarray(faces)
    if (len(v) < 4 or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or int(f.min()) < 0 or int(f.max()) >= len(v)):
        raise ValueError("V>=4 and nonempty valid integer Fx3 required")
    indices = np.unique(f).astype(np.int64)
    referenced = v[indices]
    lo, hi = referenced.min(axis=0), referenced.max(axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        extent = hi-lo
        midpoint = lo+extent*.5
    if not np.isfinite(extent).all() or not np.isfinite(midpoint).all():
        raise ValueError("Bounding-box range outside finite float64")
    maximum = float(extent.max())
    if maximum <= 0:
        raise ValueError("Positive referenced extent required")
    mantissa, exponent = math.frexp(maximum)
    exponent -= int(mantissa == .5)
    if not -1074 <= exponent <= 1023:
        raise ValueError("Covering power-of-two scale outside finite float64")
    scale = math.ldexp(1., exponent)
    origin = np.zeros(3, np.float64)
    modes = []
    for j, candidate in enumerate(midpoint):
        selected = candidate != 0 and sterbenz_subtraction_certified(v[:, j], float(candidate))
        origin[j] = candidate if selected else 0.
        modes.append("sterbenz_midpoint" if selected else "zero")
    encoded = _transform(v, origin, exponent, False)
    decoded = _transform(encoded, origin, exponent, True)
    if not np.array_equal(v, decoded):
        raise ValueError("Selected source chart inverse not exact; no alternate origin")
    evidence = {
        "schema": SCHEMA, "phase": "prepared", "policy_sha256": POLICY_SHA256,
        "input_vertices_sha256": _hash(source), "input_faces_sha256": _hash(f),
        "origin_sha256": _hash(origin), "referenced_indices_sha256": _hash(indices),
        "encoded_source_sha256": _hash(encoded), "decoded_source_sha256": _hash(decoded),
        "vertices": len(v), "faces": len(f), "referenced_vertices": len(indices),
        "all_source_roundtrip_vertices": len(v), "scale": scale, "scale_exponent": exponent,
        "maximum_bbox_extent": maximum, "roundtrip_numerically_exact": True,
        "roundtrip_byte_exact": decoded.tobytes() == v.tobytes(),
        "origin_mode_x": modes[0], "origin_mode_y": modes[1], "origin_mode_z": modes[2],
        "origin_search_performed": False, "source_arrays_modified": False,
        "source_geometry_repaired": False, "physical_geometry_rescaled": False,
        "physical_serialization_verified": False, "metric_fidelity_verified": False,
        "topology_verified": False, "native_backend_qualified": False, "adopted": False,
    }
    evidence["chart_sha256"] = hashlib.sha256(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return MeshConditioningChartV2(origin, scale, exponent, indices, tuple(modes), MappingProxyType(evidence))

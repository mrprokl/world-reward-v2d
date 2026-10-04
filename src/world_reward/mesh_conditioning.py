"""One source-derived coordinate chart, not physical scaling or a mesh repair.

Only referenced vertices determine the bbox. Faces and orphans remain unchanged.
F32 storage, default-eight-digit welding, topology and metric fidelity must be
checked separately in decoded physical coordinates, never in this chart.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

SCHEMA = "world-reward-mesh-conditioning-v1"
_POLICY = {"schema": SCHEMA, "origin": "bbox_min+(bbox_max-bbox_min)/2",
           "scale": "smallest finite positive power of two >= maximum bbox extent",
           "bbox": "referenced vertices only", "encode": "(x-origin)/scale",
           "decode": "origin+scale*x", "repair": False}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    _POLICY, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _hash(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _readonly(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _points(value):
    if np.ma.isMaskedArray(value):
        raise ValueError("Explicit unmasked float64 coordinates required")
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1:] != (3,) or not len(array)
            or array.dtype.kind != "f" or array.dtype.itemsize != 8
            or not np.isfinite(array).all()):
        raise ValueError("Require finite nonempty float64 Nx3 coordinates")
    return array.astype(np.float64, copy=False)


def _transform(points, origin, scale, *, inverse):
    points = _points(points)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = points * scale if inverse else points - origin
        result = origin + delta if inverse else delta / scale
    lost = (points != 0) & (delta == 0) if inverse else (delta != 0) & (result == 0)
    if not np.isfinite(delta).all() or not np.isfinite(result).all() or lost.any():
        raise ValueError("Coordinate transform overflow or nonzero-to-zero underflow")
    return _readonly(result)


@dataclass(frozen=True, eq=False)
class MeshConditioningChart:
    origin: np.ndarray
    scale: float
    scale_exponent: int
    referenced_indices: np.ndarray
    diagnostics: MappingProxyType

    def __post_init__(self):
        origin, indices = np.asarray(self.origin), np.asarray(self.referenced_indices)
        if (origin.shape != (3,) or origin.dtype != np.float64
                or not np.isfinite(origin).all()
                or type(self.scale) is not float or not math.isfinite(self.scale)
                or self.scale <= 0 or type(self.scale_exponent) is not int
                or not -1074 <= self.scale_exponent <= 1023
                or self.scale != math.ldexp(1., self.scale_exponent)
                or indices.ndim != 1 or indices.dtype != np.int64 or not len(indices)
                or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("Explicit finite chart and increasing referenced indices required")
        evidence = dict(self.diagnostics)
        if any(type(v) not in (str, int, float, bool) for v in evidence.values()):
            raise ValueError("Only scalar/hash diagnostics, not geometry or nested metadata")
        json.dumps(evidence, allow_nan=False)
        object.__setattr__(self, "origin", _readonly(origin))
        object.__setattr__(self, "referenced_indices", _readonly(indices))
        object.__setattr__(self, "diagnostics", MappingProxyType(evidence))

    def encode(self, physical_vertices):
        """Apply this same chart to all supplied points; no refit or orphan filter."""
        return _transform(physical_vertices, self.origin, self.scale, inverse=False)

    def decode(self, chart_vertices):
        """Return physical F64 coordinates; finite roundoff is not silently repaired."""
        return _transform(chart_vertices, self.origin, self.scale, inverse=True)


def prepare_conditioning(vertices, faces) -> MeshConditioningChart:
    """Freeze one referenced-source bbox chart, without validating mesh quality.

    Require finite F64 Vx3 (V>=4), nonempty integer Fx3 and in-range indices.
    Orphans cannot change the chart, but their bytes are bound in the source hash.
    Zero extent, unrepresentable scale, overflow and lost underflow fail closed.
    Finite rounding, including signed-zero changes, is measured rather than
    corrected. The source roundtrip covers referenced vertices, not orphans.
    """
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Explicit unmasked mesh arrays required")
    source = np.asarray(vertices)
    v, f = _points(source), np.asarray(faces)
    if (len(v) < 4 or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or int(f.min()) < 0 or int(f.max()) >= len(v)):
        raise ValueError("Require V>=4 and nonempty valid integer Fx3 faces")
    indices = np.unique(f).astype(np.int64)
    referenced = v[indices]
    lo, hi = referenced.min(axis=0), referenced.max(axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        extent = hi - lo
        half = extent * .5
        origin = lo + half
    if (not np.isfinite(extent).all() or not np.isfinite(origin).all()
            or np.any((extent != 0) & (half == 0))):
        raise ValueError("Bounding-box overflow or midpoint underflow")
    maximum = float(extent.max())
    if maximum <= 0:
        raise ValueError("A positive referenced bounding-box extent is required")
    mantissa, exponent = math.frexp(maximum)
    exponent -= int(mantissa == .5)
    if not -1074 <= exponent <= 1023:
        raise ValueError("Covering power-of-two scale is outside finite float64 range")
    scale = math.ldexp(1., exponent)
    encoded = _transform(referenced, origin, scale, inverse=False)
    decoded = _transform(encoded, origin, scale, inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(decoded - referenced)
    if not np.isfinite(error).all():
        raise ValueError("Roundtrip error is outside finite float64 range")
    evidence = {
        "schema": SCHEMA, "phase": "prepared", "policy_sha256": POLICY_SHA256,
        "input_vertices_sha256": _hash(source), "input_faces_sha256": _hash(f),
        "origin_sha256": _hash(origin), "scale_sha256": _hash(np.array([scale])),
        "referenced_indices_sha256": _hash(indices), "encoded_source_sha256": _hash(encoded),
        "decoded_source_sha256": _hash(decoded), "vertices": len(v), "faces": len(f),
        "referenced_vertices": len(indices), "ignored_orphan_vertices": len(v)-len(indices),
        "scale_exponent": exponent, "scale": scale, "maximum_bbox_extent": maximum,
        "encoded_maximum_absolute_coordinate": float(np.abs(encoded).max()),
        "roundtrip_numerically_exact": bool(np.array_equal(decoded, referenced)),
        "roundtrip_byte_exact": decoded.tobytes() == referenced.tobytes(),
        "roundtrip_changed_coordinates": int(np.count_nonzero(decoded != referenced)),
        "roundtrip_maximum_absolute_error": float(error.max()),
        "source_arrays_modified": False, "source_geometry_repaired": False,
        "physical_serialization_verified": False, "metric_fidelity_verified": False,
        "topology_verified": False, "adopted": False,
    }
    evidence["chart_sha256"] = hashlib.sha256(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return MeshConditioningChart(origin, scale, exponent, indices, MappingProxyType(evidence))

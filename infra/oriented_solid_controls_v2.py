"""Four frozen new curved-solid sources; no certificate, chart solver or QEM.

This cohort is separate from all previous controls. Both families are sampled
smooth radial harmonic surfaces, not rescaled historical meshes. Decimal
trigonometry makes their represented source arrays reproducible across libm
implementations; conversion to F64 is ordinary representation, never snapping.
Expected forests and chart domains are hypotheses checked before native calls.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType

import numpy as np

SCHEMA = "world_reward.oriented_solid_controls.v2"
FAMILIES = ("fresh_v2_lobed_twocavities", "fresh_v2_lobed_island_obliqueroot")
FIXTURE_NAMES = tuple(n + suffix for n in FAMILIES for suffix in ("_zero", "_sterbenz"))
OUTER_SAMPLING, OTHER_SAMPLING = (56, 30), (28, 14)
SCALES, TRANSLATIONS = (1., 2.**-11), ((0., 0., 0.), (8., -12., 16.))
METRIC_SCALE = .375
PI = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
DECIMAL_PRECISION, TAYLOR_TERMS = 90, 64
# key, sampling, axes, center, (quadratic harmonic, cubic harmonic), orientation
OUTER = ("outer", OUTER_SAMPLING, (1.125, .9375, .8125), (0., 0., 0.), (1/32, 1/64), 1)
FAMILY_SPECS = (
    (OUTER,
     ("cavity-a", OTHER_SAMPLING, (.1875, .15625, .125), (-.3125, .09375, -.0625), (1/64, -1/128), -1),
     ("cavity-b", OTHER_SAMPLING, (.15625, .140625, .109375), (.359375, -.109375, .078125), (-1/128, 1/128), -1)),
    (OUTER,
     ("void", OTHER_SAMPLING, (.46875, .359375, .28125), (.046875, -.03125, .015625), (-1/64, 1/128), -1),
     ("island", OTHER_SAMPLING, (.109375, .09375, .078125), (.078125, -.015625, .03125), (1/128, -1/256), 1),
     ("independent-root", OTHER_SAMPLING, (.234375, .1875, .15625), (1.8125, .4375, -.21875), (-1/32, -1/128), 1)),
)
PARENTS = ((-1, 0, 0), (-1, 0, 1, -1))
SPEC = dict(schema=SCHEMA, families=FAMILIES, components=FAMILY_SPECS, parents=PARENTS,
            scales=SCALES, translations_in_scale_units=TRANSLATIONS, metric_scale=METRIC_SCALE,
            radius="1+a*(nx*nx-ny*ny)+b*nx*ny*nz", pi=PI,
            decimal_precision=DECIMAL_PRECISION, taylor_terms=TAYLOR_TERMS,
            trig="quarter/hemisphere analytic sign construction; Decimal Taylor then F64",
            intended_chart_domains=("zero,zero,zero", "sterbenz,sterbenz,sterbenz"),
            target_faces=4096, target_vertices=4096, chamfer_diagonal_limit=.01,
            each_shell_and_net_volume_limit=.05, resample=False, geometry_certified=False)
SPEC_SHA256 = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()
# First generation of the already fixed specification, before any chart/QEM.
SOURCE_ARRAY_SHA256 = MappingProxyType({
    "fresh_v2_lobed_twocavities_zero": (
        "4c2de942a20ada403a35da32a1f78b8c8019a8ee895f226a802b26cda3da296d",
        "eea1376dffe888a11e93b92bfeaeb20bc2a570ed159e3a8b3df65c6efa43eb55"),
    "fresh_v2_lobed_twocavities_sterbenz": (
        "2881f3fb8f0c38096f649114e1a51ae6292acd3c68e4fb15aac3260b6d2deb9b",
        "eea1376dffe888a11e93b92bfeaeb20bc2a570ed159e3a8b3df65c6efa43eb55"),
    "fresh_v2_lobed_island_obliqueroot_zero": (
        "f41db76a413c4b759853174634dec296b7b54c99ec73a911eee47da6c11b2166",
        "8d5a34767942a77a25fc8e62ce89bdc496fa8f92cd11b53e5ef8999ef1fa2b3e"),
    "fresh_v2_lobed_island_obliqueroot_sterbenz": (
        "9175f6c5f0e524e0728660ce95d64c3be3b6e05b8c6893a7cbaaa6b414a90f7b",
        "8d5a34767942a77a25fc8e62ce89bdc496fa8f92cd11b53e5ef8999ef1fa2b3e"),
})


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_fingerprints(mesh):
    return tuple(hashlib.sha256(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode()
                                + b"\0" + np.ascontiguousarray(a).tobytes()).hexdigest() for a in mesh)


@lru_cache(maxsize=128)
def _sincos(numerator, denominator):
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        x = Decimal(PI) * Decimal(numerator) / Decimal(denominator)
        square = x * x
        s, c, st, ct = x, Decimal(1), x, Decimal(1)
        for k in range(1, TAYLOR_TERMS):
            st = -st * square / Decimal((2*k) * (2*k+1))
            ct = -ct * square / Decimal((2*k-1) * (2*k))
            s += st; c += ct
        return float(s), float(c)


def radial_surface(meridians, latitudes, axes, center, harmonics, sign):
    """Direct smooth radial sampling, unique angular seam and two unique poles."""
    if (type(meridians) is not int or meridians < 8 or meridians % 4 or
            type(latitudes) is not int or latitudes < 4 or latitudes % 2 or
            type(sign) is not int or sign not in (-1, 1)):
        raise ValueError("Explicit quarter/even sampling and orientation required")
    axes, center, harmonics = map(lambda x: np.asarray(x, dtype=np.float64), (axes, center, harmonics))
    if (axes.shape != (3,) or center.shape != (3,) or harmonics.shape != (2,) or
            not all(np.isfinite(x).all() for x in (axes, center, harmonics)) or
            np.any(axes <= 0) or np.sum(np.abs(harmonics)) >= 1):
        raise ValueError("Finite positive radial surface parameters required")
    quarter = np.array([(c, s) for s, c in (_sincos(2*j, meridians) for j in range(meridians//4))])
    angles = np.vstack((quarter, np.c_[-quarter[:, 1], quarter[:, 0]],
                        -quarter, np.c_[quarter[:, 1], -quarter[:, 0]]))
    north = [np.c_[angles * s, np.full(meridians, c)]
             for s, c in (_sincos(k, latitudes) for k in range(1, latitudes//2))]
    shift = (np.arange(meridians) + meridians//2) % meridians
    directions = np.vstack((*[-r[shift] for r in north], np.c_[angles, np.zeros(meridians)],
                            *reversed(north), [[0., 0., -1.], [0., 0., 1.]]))
    nx, ny, nz = directions.T
    radius = 1. + harmonics[0] * (nx*nx - ny*ny) + harmonics[1] * ((nx*ny)*nz)
    vertices = (directions * radius[:, None]) * axes + center
    bottom, top = len(vertices)-2, len(vertices)-1
    faces = []
    for ring in range(latitudes-2):
        for j in range(meridians):
            a, b = ring*meridians+j, ring*meridians+(j+1) % meridians
            c, d = a+meridians, b+meridians
            faces.extend(((a, b, c), (b, d, c)))
    for j in range(meridians):
        faces.extend(((bottom, (j+1) % meridians, j),
                      (top, (latitudes-2)*meridians+j, (latitudes-2)*meridians+(j+1) % meridians)))
    faces = np.asarray(faces, dtype=np.int64)
    return _owned(vertices), _owned(faces if sign == 1 else faces[:, ::-1])


def midpoint_sterbenz_domains(vertices):
    """Necessary whole-axis Sterbenz conditions, not an executed chart or certificate."""
    vertices = np.asarray(vertices)
    if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Finite original F64 Vx3 required")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    origin = lo + (hi-lo)*.5
    domains = tuple(bool(o != 0 and np.all(np.signbit(vertices[:, axis]) == np.signbit(o))
                        and np.all(np.abs(vertices[:, axis]) >= abs(o)*.5)
                        and np.all(np.abs(vertices[:, axis]) <= abs(o)*2)) for axis, o in enumerate(origin))
    return domains


def fixtures():
    """Four (name, (F64 V,I64 F), immutable metadata) sources, never resampled."""
    records = []
    for family, specs, parents in zip(FAMILIES, FAMILY_SPECS, PARENTS):
        vertices, faces, labels, ranges, keys = [], [], [], [], []
        offset = face_offset = 0
        for component, (key, sampling, axes, center, harmonics, sign) in enumerate(specs):
            v, f = radial_surface(*sampling, axes, center, harmonics, sign)
            vertices.append(v); faces.append(f+offset); labels.extend([component]*len(f)); keys.append(f"{family}:{key}")
            ranges.append((offset, offset+len(v), face_offset, face_offset+len(f)))
            offset += len(v); face_offset += len(f)
        base = np.vstack(vertices), np.vstack(faces)
        for index, (scale, translation) in enumerate(zip(SCALES, TRANSLATIONS)):
            name = family + ("_zero" if index == 0 else "_sterbenz")
            mesh = (_owned(base[0]*scale + np.asarray(translation)*scale), _owned(base[1]))
            hashes = array_fingerprints(mesh)
            if hashes != SOURCE_ARRAY_SHA256[name]:
                raise ValueError("Frozen whole-source arrays changed; never resample")
            metadata = MappingProxyType(dict(schema=SCHEMA, specification_sha256=SPEC_SHA256,
                component_keys=tuple(keys), face_components=_owned(np.array(labels, dtype=np.int64)),
                component_ranges=tuple(ranges), expected_parents=_owned(np.array(parents, dtype=np.int64)),
                expected_signs=_owned(np.array([s[-1] for s in specs], dtype=np.int64)),
                metric_scale=METRIC_SCALE, similarity_scale=scale,
                similarity_translation=tuple(x*scale for x in translation), source_array_sha256=hashes,
                intended_chart_axes=("zero",)*3 if index == 0 else ("sterbenz",)*3,
                midpoint_sterbenz_conditions=midpoint_sterbenz_domains(mesh[0]),
                geometry_certified=False, procedural_source_only=True, chart_qualified=False,
                native_calls=0, adopted=False, historical_failure_replayed=False))
            records.append((name, mesh, metadata))
    return tuple(records)

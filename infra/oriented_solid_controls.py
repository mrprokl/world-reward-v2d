"""Fresh curved forest sources; generation is not an embedding certificate.

No model, mesh library, native call or historical fixture is used. Two families
and two fixed dyadic similarities exercise over-budget geometry while keeping
all original component keys/vertices/faces. Parent relations are procedural
expectations, independently checked by the caller's certified predicates before
QEM. Analytic containment bounds do not claim certified stored F64/F32 geometry.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np

FAMILY_NAMES = ("newforest_multicavity", "newforest_island_independentroot")
FIXTURE_NAMES = tuple(name + suffix for name in FAMILY_NAMES for suffix in ("_identity", "_small_translated"))
OUTER_SAMPLING, OTHER_SAMPLING = (64, 32), (32, 16)
OUTER_AXES = (1., .875, .75)
MULTICAVITY_AXES = ((.21875, .1875, .15625), (.1875, .15625, .125))
MULTICAVITY_CENTERS = ((-.375, 0., 0.), (.375, 0., 0.))
VOID_AXES, ISLAND_AXES, ROOT_AXES = (.5, .4375, .375), (.1875, .15625, .125), (.25, .21875, .1875)
INDEPENDENT_CENTER = (2., 0., 0.)
SCALES, TRANSLATIONS = (1., 2.**-16), ((0., 0., 0.), (2.**-12, -2.**-13, 2.**-14))
METRIC_SCALE = .375


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_fingerprints(mesh):
    """Typed array identity, not a certificate of geometric validity."""
    result = []
    for array in mesh:
        header = json.dumps(dict(dtype=array.dtype.str, shape=array.shape), sort_keys=True).encode()
        result.append(hashlib.sha256(header + b"\0" + np.ascontiguousarray(array).tobytes()).hexdigest())
    return tuple(result)


def ellipsoid(meridians=64, latitudes=32, axes=OUTER_AXES, center=(0., 0., 0.), sign=1):
    """Closed outward UV ellipsoid: unique seam, two unique poles, no repair.

    Cardinal directions/equator are constructed analytically; other vertices
    are direct sin/cos samples. Inward fixture boundaries reverse face order
    during generation, never reorient a loaded prediction to obtain a pass.
    """
    if (type(meridians) is not int or meridians < 8 or meridians % 4
            or type(latitudes) is not int or latitudes < 4 or latitudes % 2
            or sign not in (-1, 1) or type(sign) is not int):
        raise ValueError("Fixed quarter/even sampling and explicit orientation required")
    axes, center = np.asarray(axes, dtype=np.float64), np.asarray(center, dtype=np.float64)
    if axes.shape != (3,) or center.shape != (3,) or not np.isfinite(axes).all() or not np.isfinite(center).all() or np.any(axes <= 0):
        raise ValueError("Finite positive axes and finite center required")
    theta = 2 * np.pi * np.arange(meridians // 4) / meridians
    quarter = np.c_[np.cos(theta), np.sin(theta)]
    directions = np.vstack((quarter, np.c_[-quarter[:, 1], quarter[:, 0]],
                            -quarter, np.c_[quarter[:, 1], -quarter[:, 0]]))
    rings = []
    for latitude in range(1, latitudes):
        if latitude == latitudes // 2:
            radius, z = 1., 0.
        else:
            phi = np.pi * latitude / latitudes
            radius, z = np.sin(phi), -np.cos(phi)
        rings.append(np.c_[directions * radius, np.full(meridians, z)])
    vertices = np.vstack((*rings, [[0., 0., -1.], [0., 0., 1.]])) * axes + center
    bottom, top = len(vertices) - 2, len(vertices) - 1
    faces = []
    for ring in range(latitudes - 2):
        for j in range(meridians):
            a, b = ring * meridians + j, ring * meridians + (j + 1) % meridians
            c, d = a + meridians, b + meridians
            faces.extend(((a, b, c), (b, d, c)))
    for j in range(meridians):
        faces.append((bottom, (j + 1) % meridians, j))
        faces.append((top, (latitudes - 2) * meridians + j,
                      (latitudes - 2) * meridians + (j + 1) % meridians))
    faces = np.asarray(faces, dtype=np.int64)
    if sign == -1:
        faces = faces[:, ::-1]
    return _owned(vertices), _owned(faces)


def _family(specs):
    vertices, faces, labels, ranges, keys, offset, face_offset = [], [], [], [], [], 0, 0
    for component, (key, sampling, axes, center, sign) in enumerate(specs):
        v, f = ellipsoid(*sampling, axes, center, sign)
        vertices.append(v); faces.append(f + offset)
        labels.extend([component] * len(f)); keys.append(key)
        ranges.append((offset, offset + len(v), face_offset, face_offset + len(f)))
        offset += len(v); face_offset += len(f)
    return (np.vstack(vertices), np.vstack(faces)), tuple(keys), np.array(labels, dtype=np.int64), tuple(ranges)


def fixtures():
    """Return (name, (V,F), metadata), original component order, four fresh meshes.

    Metadata arrays are immutable, and expected signs/parents never infer
    geometry from a native result. All certification/volume/Chamfer/packing
    gates and the caller's bounded native runtime remain separate.
    """
    outer = ("outer", OUTER_SAMPLING, OUTER_AXES, (0., 0., 0.), 1)
    first = (outer,) + tuple((f"cavity-{i}", OTHER_SAMPLING, axes, center, -1)
                             for i, (axes, center) in enumerate(zip(MULTICAVITY_AXES, MULTICAVITY_CENTERS)))
    second = (outer, ("void", OTHER_SAMPLING, VOID_AXES, (0., 0., 0.), -1),
              ("island", OTHER_SAMPLING, ISLAND_AXES, (0., 0., 0.), 1),
              ("independent-root", OTHER_SAMPLING, ROOT_AXES, INDEPENDENT_CENTER, 1))
    result = []
    for family, specs, parents in zip(FAMILY_NAMES, (first, second), ((-1, 0, 0), (-1, 0, 1, -1))):
        source, keys, labels, ranges = _family(specs)
        for similarity, (scale, translation) in enumerate(zip(SCALES, TRANSLATIONS)):
            mesh = (_owned(source[0] * scale + np.array(translation)), _owned(source[1]))
            metadata = MappingProxyType(dict(
                component_keys=tuple(f"{family}:{key}" for key in keys),
                face_components=_owned(labels), component_ranges=ranges,
                expected_parents=_owned(np.array(parents, dtype=np.int64)),
                expected_signs=_owned(np.array([spec[-1] for spec in specs], dtype=np.int64)),
                metric_scale=METRIC_SCALE, similarity_scale=scale, similarity_translation=translation,
                source_array_sha256=array_fingerprints(mesh), geometry_certified=False,
                procedural_source_only=True))
            result.append((family + ("_identity" if similarity == 0 else "_small_translated"), mesh, metadata))
    return tuple(result)

"""Birth-provenance correspondence of whole oriented solid forests, not meshes.

The source/candidate forests must come from independently authenticated exact
embedding/containment predicates. J maps every candidate face to an ORIGINAL
source face, I maps every candidate vertex to an ORIGINAL source vertex. These
are the native QEM birth arrays, not its internally numbered volume components.
Full source-face labels tie births to the caller's original source keys. No
orientation, coordinate, face, key or containment relation is changed here.
This primitive verifies provenance/forest structure only: not legal edge
collapses, Euler topology, geometry/volume fidelity or compiler qualification.
Official all-zero padding is handled and proven separately before this call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_reward.oriented_solid_forest import OrientedSolidForest


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _indices(value, shape, bound, name):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype != np.int64 or array.shape != shape
            or np.any(array < 0) or np.any(array >= bound)):
        raise ValueError(f"{name}: full int64 {shape} original index array required")
    return _owned(array)


@dataclass(frozen=True, eq=False)
class SolidForestFidelity:
    """Readonly correspondence, preserving every original source key/order.

    candidate_to_source[c] gives source index for candidate component c;
    source_to_candidate is its inverse. Birth arrays are retained unchanged.
    Keys certify no artifact authenticity. Geometry/collapse legality stays
    with the separate source-bound native and fidelity gates.
    """
    source_component_keys: tuple[str, ...]
    candidate_component_keys: tuple[str, ...]
    candidate_to_source: np.ndarray
    source_to_candidate: np.ndarray
    source_face_counts: np.ndarray
    candidate_face_counts: np.ndarray
    face_birth: np.ndarray
    vertex_birth: np.ndarray


def compare_solid_forest_fidelity(
        source_forest, candidate_forest, source_faces, candidate_faces,
        source_face_components, candidate_face_components, face_birth, vertex_birth,
        *, source_vertex_count):
    """Require bijective shell births and identical complete forest after remapping.

    Faces and component labels are in their original full arrays, never sorted,
    compacted or paired by volume. Every represented forest component must have
    source/candidate faces. Candidate vertex count is len(I); all are referenced,
    with same-source-shell birth for every incident face. All original source
    vertices must be referenced, matching the native certificate. J/I need not be injective: original
    faces may collapse and separate descendants may share a valid birth.
    No source component may disappear, merge, split or change depth/parent/sign.
    Caller must authenticate J/I and prove the actual native transactions.
    """
    if type(source_forest) is not OrientedSolidForest or type(candidate_forest) is not OrientedSolidForest:
        raise ValueError("Explicit adjudicated source and candidate forests required")
    # Recheck potentially direct-constructed result objects, without trusting labels.
    source = OrientedSolidForest(source_forest.component_keys, source_forest.signs, source_forest.inside)
    candidate = OrientedSolidForest(candidate_forest.component_keys, candidate_forest.signs, candidate_forest.inside)
    count = len(source.component_keys)
    if len(candidate.component_keys) != count:
        raise ValueError("Every original source shell must be retained bijectively")
    if type(source_vertex_count) is not int or source_vertex_count < 1:
        raise ValueError("Full original source vertex count required")
    sf, cf, births = np.asarray(source_faces), np.asarray(candidate_faces), np.asarray(vertex_birth)
    if (sf.ndim != 2 or sf.shape[1:] != (3,) or not len(sf)
            or cf.ndim != 2 or cf.shape[1:] != (3,) or not len(cf)
            or births.ndim != 1 or not len(births)):
        raise ValueError("Full nonempty triangular faces and vertex births required")
    sf = _indices(source_faces, sf.shape, source_vertex_count, "Source faces")
    cf = _indices(candidate_faces, cf.shape, len(births), "Candidate faces")
    sl = _indices(source_face_components, (len(sf),), count, "Source face components")
    cl = _indices(candidate_face_components, (len(cf),), count, "Candidate face components")
    J = _indices(face_birth, (len(cf),), len(sf), "Face birth J")
    I = _indices(vertex_birth, (len(births),), source_vertex_count, "Vertex birth I")
    if (np.any(np.diff(np.sort(sf, axis=1), axis=1) == 0)
            or np.any(np.diff(np.sort(cf, axis=1), axis=1) == 0)):
        raise ValueError("Repeated triangular indices/padding are not meaningful faces")
    if len(np.unique(cf)) != len(I):
        raise ValueError("Every candidate vertex birth must belong to an actual face")
    source_counts, candidate_counts = np.bincount(sl, minlength=count), np.bincount(cl, minlength=count)
    if np.any(source_counts == 0) or np.any(candidate_counts == 0):
        raise ValueError("No represented source or candidate shell may lack faces")
    # Full-source vertex ownership, with no assumption about native volume IDs.
    owner = np.full(source_vertex_count, -1, dtype=np.int64)
    for s in range(count):
        vertices = np.unique(sf[sl == s])
        if np.any(owner[vertices] != -1):
            raise ValueError("Source components share an indexed vertex")
        owner[vertices] = s
    if np.any(owner == -1):
        raise ValueError("Unreferenced original source vertex is outside the certified solid")
    c_to_s = np.full(count, -1, dtype=np.int64)
    for c in range(count):
        selected = cl == c
        shell_births = np.unique(sl[J[selected]])
        if len(shell_births) != 1:
            raise ValueError("Candidate component merged multiple original source shells")
        s = int(shell_births[0]); c_to_s[c] = s
        vertices = np.unique(cf[selected])
        if not np.all(owner[I[vertices]] == s):
            raise ValueError("Candidate vertex birth crosses or escapes its source shell")
    if len(np.unique(c_to_s)) != count:
        raise ValueError("Birth correspondence split or removed an original source shell")
    s_to_c = np.empty(count, dtype=np.int64); s_to_c[c_to_s] = np.arange(count, dtype=np.int64)
    if (not np.array_equal(candidate.signs, source.signs[c_to_s])
            or not np.array_equal(candidate.depths, source.depths[c_to_s])
            or not np.array_equal(candidate.inside, source.inside[np.ix_(c_to_s, c_to_s)])):
        raise ValueError("Original shell orientation/depth/complete containment changed")
    parents = np.full(count, -1, dtype=np.int64)
    nonroots = candidate.parents >= 0
    parents[nonroots] = c_to_s[candidate.parents[nonroots]]
    if not np.array_equal(parents, source.parents[c_to_s]):
        raise ValueError("Original shell parent relation changed")
    return SolidForestFidelity(source.component_keys, candidate.component_keys,
        _owned(c_to_s), _owned(s_to_c), _owned(source_counts), _owned(candidate_counts), J, I)

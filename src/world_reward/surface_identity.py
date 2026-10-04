"""Whole under-budget surface identity, not a solid, loader or LOD certificate.

Native arrays are copied into immutable byte backing, including orphan vertices,
signed zero and original face order. Only exact non-collinearity and combinatorial
oriented vertex-manifoldness (with optional boundaries) are adjudicated. There
is no weld, scaling, repair, geometry I/O, QEM, embedding or accuracy claim.
Float32 export readiness is separate diagnostic evidence, never a silent cast
or an acceptance claim about the real loader/official packer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from types import MappingProxyType

import numpy as np

from world_reward.exact_triangle_predicates import (
    ExactTriangleDegeneracyError, validate_exact_triangle_non_degeneracy,
)
from world_reward.raw_shape_proposal import (
    RawShapeProposal, REFERENCE_ROLES, _identity, _raw_array, _ref, _ref_record,
)

SCHEMA = "world_reward.surface_identity.v1"
BUDGET = 4096


def _readonly(array):
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _topology(faces, vertex_count):
    edges, links, seen = {}, {}, set()
    parent = list(range(len(faces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for index, row in enumerate(faces):
        a, b, c = map(int, row)
        key = tuple(sorted((a, b, c)))
        if key in seen:
            raise ValueError("Duplicate indexed triangle unsupported; no face removal")
        seen.add(key)
        for v, x, y in ((a, b, c), (b, c, a), (c, a, b)):
            graph = links.setdefault(v, {})
            graph.setdefault(x, set()).add(y)
            graph.setdefault(y, set()).add(x)
            edges.setdefault(tuple(sorted((v, x))), []).append((index, v, x))

    boundary, boundary_degree = [], {}
    for records in edges.values():
        if len(records) > 2:
            raise ValueError("Non-manifold edge unsupported; every face retained")
        if len(records) == 2:
            left, right = records
            if left[1:] != right[1:][::-1]:
                raise ValueError("Shared-edge orientations must oppose; no sign repair")
            x, y = root(left[0]), root(right[0])
            parent[max(x, y)] = min(x, y)
        else:
            boundary.append(records[0])
            for v in records[0][1:]:
                boundary_degree[v] = boundary_degree.get(v, 0) + 1

    for v, graph in links.items():
        visited, pending = set(), [next(iter(graph))]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(graph[node] - visited)
        degrees = [len(neighbors) for neighbors in graph.values()]
        boundary_count = boundary_degree.get(v, 0)
        if (len(visited) != len(graph) or any(d not in (1, 2) for d in degrees)
                or (boundary_count == 0 and any(d != 2 for d in degrees))
                or (boundary_count != 0 and (boundary_count != 2 or degrees.count(1) != 2))):
            raise ValueError("Vertex link must be one cycle or one boundary path")

    roots = sorted({root(i) for i in range(len(faces))})
    labels = {first: k for k, first in enumerate(roots)}
    components = np.array([labels[root(i)] for i in range(len(faces))], np.int64)
    outgoing = {}
    for face, a, b in boundary:
        if a in outgoing:
            raise ValueError("Boundary must have one oriented outgoing edge per vertex")
        outgoing[a] = (b, int(components[face]))
    loops, loop_components = [], []
    while outgoing:
        start = min(outgoing)
        vertex, loop = start, []
        component = outgoing[start][1]
        while True:
            if vertex not in outgoing or outgoing[vertex][1] != component:
                raise ValueError("Boundary is not a component-local oriented cycle")
            loop.append(vertex)
            vertex, _ = outgoing.pop(vertex)
            if vertex == start:
                break
        loops.append(tuple(loop))
        loop_components.append(component)
    order = sorted(range(len(loops)), key=lambda i: (loop_components[i], loops[i][0]))
    loops = tuple(loops[i] for i in order)
    loop_components = tuple(loop_components[i] for i in order)
    stats = []
    for component, first in enumerate(roots):
        rows = faces[components == component]
        vertices = np.unique(rows)
        edge_count = sum(int(components[records[0][0]]) == component for records in edges.values())
        stats.append(MappingProxyType(dict(first_face=first, vertices=len(vertices), faces=len(rows),
            edges=edge_count, euler_characteristic=len(vertices) - edge_count + len(rows),
            boundary_loops=loop_components.count(component))))
    return (tuple(f"source-array-first-face-{first}" for first in roots), _readonly(components),
            loops, loop_components, tuple(stats), vertex_count - len(links), len(boundary))


@dataclass(frozen=True, eq=False)
class SurfaceIdentity:
    """Identity births and topology of supplied native arrays; refs are declarations.

    Disconnected single triangles and closed inward/outward components are
    allowed. Orphans remain byte-identical and count against the vertex budget.
    Coincident positions are not welded or checked for geometric contact.
    A valid F64 source can have a negative F32-storage diagnostic: callers must
    qualify actual export/packing separately rather than silently repair it.
    """
    vertices: np.ndarray
    faces: np.ndarray
    declared_references: Mapping = field(default_factory=dict)
    declared_proposal_sha256: str | None = None
    vertex_birth_indices: np.ndarray = field(init=False)
    face_birth_indices: np.ndarray = field(init=False)
    face_components: np.ndarray = field(init=False)
    component_keys: tuple[str, ...] = field(init=False)
    boundary_loops: tuple[tuple[int, ...], ...] = field(init=False)
    boundary_components: tuple[int, ...] = field(init=False)
    diagnostics: Mapping = field(init=False)

    def __post_init__(self):
        v = _raw_array(self.vertices, "vertices", (np.dtype(np.float32), np.dtype(np.float64)))
        f = _raw_array(self.faces, "faces", (np.dtype(np.int64),))
        if len(v) < 3 or len(v) > BUDGET or len(f) > BUDGET or np.any(f < 0) or np.any(f >= len(v)):
            raise ValueError("Native source must fit both4096 budgets with in-range faces")
        # F32 -> F64 is exact promotion for predicates only; stored arrays stay native.
        exact = validate_exact_triangle_non_degeneracy(v.astype(np.float64, copy=False), f)
        keys, components, loops, loop_components, stats, orphans, boundary_edges = _topology(f, len(v))
        references = self.declared_references
        if not isinstance(references, Mapping) or (references and set(references) != set(REFERENCE_ROLES)):
            raise ValueError("Declared references must be empty or exactly source/model/config")
        refs = {}
        for role, values in references.items():
            if type(values) is not tuple or not values:
                raise ValueError("Each declared role requires a nonempty ArtifactRef tuple")
            refs[role] = tuple(_ref(value) for value in values)
        if len(json.dumps({role: [_ref_record(ref) for ref in values]
                           for role, values in refs.items()}).encode()) > 16384:
            raise ValueError("Declared references must remain tiny <=16KB metadata")
        sha = self.declared_proposal_sha256
        if sha is not None and (type(sha) is not str or len(sha) != 64
                                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("Declared proposal SHA256 required; this is not authentication")
        referenced = np.unique(f)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            stored = v[referenced].astype(np.float32)
        finite = bool(np.isfinite(stored).all())
        active = False
        if finite:
            # Local compact indices are only for diagnostics; no returned mesh is compacted.
            local = np.searchsorted(referenced, f)
            try:
                validate_exact_triangle_non_degeneracy(stored.astype(np.float64), local)
                active = True
            except ExactTriangleDegeneracyError:
                pass
        diagnostics = MappingProxyType(dict(
            vertices=len(v), faces=len(f), components=stats, unused_vertices_preserved=orphans,
            boundary_edges=boundary_edges, closed=boundary_edges == 0,
            vertex_links_manifold=True, shared_edges_consistently_oriented=True,
            every_native_triangle_exactly_active=exact["exact_noncollinearity_proven"],
            native_exact_fallback_faces=exact["exact_dyadic_fallback_faces"],
            referenced_float32_positions_finite=finite, float32_storage_triangles_exactly_active=active,
            float32_changes_referenced_positions=bool(not finite or not np.array_equal(
                v[referenced].astype(np.float64), stored.astype(np.float64)))))
        for name, value in (("vertices", v), ("faces", f), ("declared_references", MappingProxyType(refs)),
                ("vertex_birth_indices", _readonly(np.arange(len(v), dtype=np.int64))),
                ("face_birth_indices", _readonly(np.arange(len(f), dtype=np.int64))),
                ("face_components", components), ("component_keys", keys), ("boundary_loops", loops),
                ("boundary_components", loop_components), ("diagnostics", diagnostics)):
            object.__setattr__(self, name, value)

    @classmethod
    def from_proposal(cls, proposal):
        """Adjudicate a RawShapeProposal; never authenticate or reinterpret its gauge."""
        if type(proposal) is not RawShapeProposal:
            raise ValueError("An explicit frozen RawShapeProposal is required")
        return cls(proposal.vertices, proposal.faces, proposal.references, proposal.report()["proposal_sha256"])

    def report(self):
        """Fresh JSON metadata; actual loader/packer and embedding qualification remain absent."""
        diagnostics = dict(self.diagnostics)
        diagnostics["components"] = [dict(row) for row in diagnostics["components"]]
        return dict(schema=SCHEMA, status="surface_identity_supported", vertices=_identity(self.vertices),
            faces=_identity(self.faces), vertex_birth_indices=_identity(self.vertex_birth_indices),
            face_birth_indices=_identity(self.face_birth_indices), face_components=_identity(self.face_components),
            component_keys=list(self.component_keys), boundary_loops=[list(loop) for loop in self.boundary_loops],
            boundary_components=list(self.boundary_components), diagnostics=diagnostics,
            declared_references={role: [_ref_record(ref) for ref in refs]
                                 for role, refs in self.declared_references.items()},
            declared_proposal_sha256=self.declared_proposal_sha256,
            component_order_scope="supplied native array face order, not raw GLB primitive order",
            geometry_operations_applied=False, area_or_length_tolerance_used=False,
            source_references_authenticated=False, metric_gauge_verified=False,
            embedding_verified=False, solid_geometry_certified=False, loader_packer_qualified=False,
            reconstruction_accuracy_verified=False, adoption_authorized=False)


def prepare_surface_identity(vertices, faces, *, declared_references=None):
    """Freeze and adjudicate one whole already-budgeted surface; no fallback or I/O."""
    return SurfaceIdentity(vertices, faces, {} if declared_references is None else declared_references)

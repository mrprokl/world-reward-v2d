"""Freeze one whole surface budget proposal; callbacks own native execution.

No I/O, weld, sign repair, scale fitting, solid certification or fallback. The
over-budget branch uses the caller's authenticated QSlim and original verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from world_reward.raw_shape_proposal import _identity, _raw_array
from world_reward.surface_identity import SurfaceIdentity, _topology
from world_reward.exact_triangle_predicates import validate_exact_triangle_non_degeneracy

SCHEMA = "world_reward.surface_budget.v1"
BUDGET = 4096


class SurfaceBudgetError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = copy.deepcopy(report)


@dataclass(frozen=True, eq=False)
class SurfaceBudgetProposal:
    source_vertices: np.ndarray
    source_faces: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    mapping: dict
    proof: dict


def surface_record(vertices, faces):
    """Unbounded source topology: same exact predicates, no 4096 restriction."""
    validate_exact_triangle_non_degeneracy(vertices.astype(np.float64), faces)
    keys, labels, loops, components, stats, orphans, edges = _topology(faces, len(vertices))
    return dict(vertices=_identity(vertices), faces=_identity(faces),
        component_keys=list(keys), face_components=labels.tolist(),
        boundary_loops=[list(loop) for loop in loops], boundary_components=list(components),
        components=[dict(row) for row in stats], unused_vertices_preserved=orphans,
        boundary_edges=edges, closed=edges == 0,
        vertex_links_manifold=True, shared_edges_consistently_oriented=True)


def prepare_surface_budget(vertices, faces, *, simplify=None, verify_mapping=None):
    """Identity under both budgets; otherwise exactly one supplied native call.

    Source and candidate arrays have immutable byte backing. Rejected sources
    retain their complete byte identities in the exception report. An identity
    source retains orphans; downstream serialization must not silently drop them.
    """
    v = _raw_array(vertices, "vertices", (np.dtype(np.float32), np.dtype(np.float64)))
    f = _raw_array(faces, "faces", (np.dtype(np.int64),))
    report = dict(schema=SCHEMA, status="fail", phase="source_domain", domain="surface",
        source_arrays_frozen_before_gates=True, source_vertices=_identity(v), source_faces=_identity(f),
        maximum_qem_calls=1, qem_calls=0, geometry_repaired=False,
        components_deleted=False, orientation_repaired=False, volume_or_closure_required=False,
        embedding_certified=False, full_surface_fidelity_certified=False,
        adoption=False, reconstruction_accuracy_verified=False)
    original = (v.tobytes(), f.tobytes())
    try:
        if len(v) < 3 or np.min(f) < 0 or np.max(f) >= len(v):
            raise ValueError("Full source indices invalid; no compaction")
        report["source_surface"] = surface_record(v, f)
        identity = len(v) <= BUDGET and len(f) <= BUDGET
        report.update(method="identity" if identity else "qslim", phase="candidate")
        if identity:
            SurfaceIdentity(v, f)
            u, g = v, f
            mapping = dict(schema="surface-identity-mapping-v1", source_vertices=len(v),
                source_faces=len(f), output_vertices=len(v), output_faces=len(f),
                I=list(range(len(v))), J=list(range(len(f))), committed_collapses=0,
                target_vertices=BUDGET, target_faces=BUDGET,
                geometry_operations_applied=False)
        else:
            if report["source_surface"]["unused_vertices_preserved"]:
                raise ValueError("Over-budget native QSlim requires every original vertex referenced")
            if not callable(simplify) or not callable(verify_mapping):
                raise ValueError("One authenticated native callback and original verifier required")
            report["qem_calls"] = 1
            u, g, mapping = simplify(v, f)
            u = _raw_array(u, "candidate vertices", (np.dtype(np.float64),))
            g = _raw_array(g, "candidate faces", (np.dtype(np.int64),))
            if type(mapping) is not dict:
                raise ValueError("Original native mapping dictionary required")
            verify_mapping(v, f, u, g, mapping)
        if (v.tobytes(), f.tobytes()) != original:
            raise ValueError("Whole source changed during native callback")
        # Same under-budget domain, including every remaining component/boundary.
        candidate = SurfaceIdentity(u, g)
        if len(candidate.component_keys) != len(report["source_surface"]["component_keys"]):
            raise ValueError("Whole source components cannot disappear")
        report.update(status="pass", phase="complete", source_arrays_unchanged=True,
            candidate_vertices=_identity(u), candidate_faces=_identity(g),
            candidate_surface=surface_record(u, g),
            identity_arrays_byte_exact=identity,
            oriented_birth_quotient_replay_exact=not identity)
        return SurfaceBudgetProposal(v, f, u, g, copy.deepcopy(mapping), copy.deepcopy(report))
    except Exception as error:
        report.update(error_type=type(error).__name__, source_arrays_unchanged=(v.tobytes(), f.tobytes()) == original)
        raise SurfaceBudgetError(str(error), report) from error

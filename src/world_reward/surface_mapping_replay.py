"""Local-incidence replay of an already supplied native contraction ledger.

No simplification, geometry fitting, threshold, certification or I/O. The caller
keeps the original complete I/J/quotient/position/topology checks. Its unchanged
orientation predicate is called in the original three precision combinations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _require(value, message):
    if not value:
        raise ValueError(message)


def _owned(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class SurfaceReplayState:
    parent: np.ndarray
    quotient: np.ndarray
    positions: np.ndarray
    live: np.ndarray
    removed_faces: frozenset[int]


def replay_surface_mapping(vertices, faces, ledger, *, boundary_vertices, orientation, orientation_batch=None):
    """Replay active one-rings; return exactly the old whole-array terminal state.

    ``parent`` keeps the original uncompressed direct-parent assignments;
    ``quotient`` supplies each original vertex's final root. Removed
    face rows receive their final quotient once at the end, matching the old
    per-event full-matrix replacements without scanning all faces per event.
    Source arrays are neither mutated nor returned as writable aliases. Source
    admissibility, mapping counters, I/J and final topology remain caller gates.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    _require(not np.ma.isMaskedArray(vertices) and not np.ma.isMaskedArray(faces)
        and v.dtype in (np.dtype('float32'), np.dtype('float64')) and v.ndim == 2 and v.shape[1:] == (3,)
        and len(v) >= 3 and np.isfinite(v).all() and f.dtype == np.dtype('int64')
        and f.ndim == 2 and f.shape[1:] == (3,) and len(f) > 0 and f.min() >= 0 and f.max() < len(v),
        'Original finite arrays and full face indices required')
    _require(type(ledger) is list and callable(orientation), 'Original ledger and orientation callback required')
    _require(orientation_batch is None or callable(orientation_batch), 'Explicit optional batch orientation callback required')
    boundary = frozenset(boundary_vertices)
    parent = np.arange(len(v), dtype=np.int64)
    live, positions = f.copy(), v.astype(np.float64).copy()
    incident = [set() for _ in range(len(v))]
    for fid, row in enumerate(live):
        for vertex in row:
            incident[int(vertex)].add(fid)
    removed = set()
    for row in ledger:
        _require(type(row) is dict and set(row) == {'survivor', 'removed_vertex', 'placement', 'removed_faces'},
            'Exact committed-event schema required')
        s, d = row['survivor'], row['removed_vertex']
        p = np.asarray(row['placement'])
        _require(type(s) is int and type(d) is int and 0 <= s < d < len(v) and parent[s] == s and parent[d] == d
            and s not in boundary and d not in boundary and p.shape == (3,) and p.dtype.kind in 'fi'
            and np.isfinite(p).all(), 'Committed contraction invalid')
        ids = row['removed_faces']
        _require(type(ids) is list and len(ids) == 2 and all(type(n) is int and -(1 << 63) <= n < (1 << 63) for n in ids),
            'Exact complete integer lineage required')
        expected = incident[s] & incident[d]
        _require(set(ids) == expected and len(expected) == 2, 'Committed removed faces differ from edge incidence')
        neighbors = [set(live[sorted(incident[vertex])].ravel().tolist()) - {vertex} for vertex in (s, d)]
        opposite = {int(n) for fid in expected for n in live[fid] if n not in (s, d)}
        _require(neighbors[0] & neighbors[1] == opposite and len(opposite) == 2,
            'Committed edge violates real interior link condition')
        affected = incident[s] | incident[d]
        affected_ids = sorted(affected - expected)
        if orientation_batch is not None and affected_ids:
            old = positions[live[affected_ids]].copy()
            new = old.copy(); new[np.isin(live[affected_ids], [s, d])] = p
            orientation_batch(old, new)
            orientation_batch(old.astype(np.float32), new.astype(np.float32))
            orientation_batch(new, new.astype(np.float32))
        elif orientation_batch is None:
            for fid in affected_ids:
                old = positions[live[fid]].copy()
                new = old.copy(); new[np.isin(live[fid], [s, d])] = p
                orientation(old, new)
                orientation(old.astype(np.float32), new.astype(np.float32))
                orientation(new, new.astype(np.float32))
        # Update exactly the active local rows, including the two removed faces.
        for fid in sorted(affected):
            for vertex in live[fid]:
                incident[int(vertex)].discard(fid)
            if fid not in expected:
                live[fid, live[fid] == d] = s
                for vertex in live[fid]:
                    incident[int(vertex)].add(fid)
        removed.update(expected); parent[d] = s; positions[s] = positions[d] = p
    # Compress a separate quotient only; keep direct-parent output byte-exact.
    quotient = parent.copy()
    for vertex in range(len(quotient)):
        root = vertex
        while quotient[root] != root:
            root = int(quotient[root])
        current = vertex
        while quotient[current] != current:
            following = int(quotient[current]); quotient[current] = root; current = following
    whole_live = quotient[f]
    # Removed rows were historically remapped too; reconstruct their stale
    # coordinates once, while verifying all actively maintained rows exactly.
    active = np.ones(len(f), dtype=bool)
    active[list(removed)] = False
    _require(np.array_equal(live[active], whole_live[active]), 'Active incidence replay drift')
    return SurfaceReplayState(_owned(parent), _owned(quotient), _owned(positions), _owned(whole_live), frozenset(removed))

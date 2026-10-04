"""Raw global association costs, not accepted identities or predicted geometry.

Each graph is one supplied persistent joint actor/object hypothesis. Missing
states belong to that same hypothesis; they never reset identity or supply an
observation. The caller supplies all automatic costs, admissible edges and
provenance. This module neither calibrates them nor verifies their provenance.
Geometry, camera, shape and scale stay outside the graph and globally fixed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from world_reward.pose_selection import NoFeasiblePathError


def _text(value, name):
    if (type(value) is not str or not value.strip() or len(value) > 256
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{name}: bounded nonempty text required")
    return value


def _array(value, name, shape, *, boolean=False):
    original = np.asarray(value)
    if (np.ma.isMaskedArray(value) or original.shape != shape
            or (original.dtype != np.bool_ if boolean else original.dtype.kind not in "fiu")):
        raise ValueError(f"{name}: declared shape and {'boolean' if boolean else 'real'} array required")
    owned = np.array(original, dtype=np.bool_ if boolean else np.float64, copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{name}: all slots must be finite, including forbidden edges")
    owned.flags.writeable = False
    return owned


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class IdentityNode:
    """Stable state key and supplied automatic observation reference, or missing.

    Different nuisance states may reference the same evidence key. Such states
    are not different actor/object identities. Keys declare references, not
    authenticated artifacts or calibrated visibility.
    """
    state_key: str
    evidence_key: str | None

    def __post_init__(self):
        _text(self.state_key, "state_key")
        if self.evidence_key is not None:
            _text(self.evidence_key, "evidence_key")


@dataclass(frozen=True, eq=False)
class IdentityGraph:
    """One persistent Z=(actor_hypothesis, object_hypothesis), on full arange(T).

    Layer sizes may differ. unary_costs[t] has shape [K_t]; transition_costs[t]
    and explicit boolean allowed_edges[t] have shape [K_t, K_(t+1)]. All cost
    slots are finite, including forbidden edges; negative energies are allowed.
    Missing nodes are explicitly supplied, never inserted by the solver.

    Construction owns readonly float64/bool copies and sorts each layer by its
    unique state keys, reordering cost arrays consistently. These deterministic
    keys are not identity truth. The source_ref is tiny metadata, not a verified
    licence, leakage, calibration or provenance certificate.
    """
    identity_key: tuple[str, str]
    frame_index: np.ndarray
    nodes: tuple[tuple[IdentityNode, ...], ...]
    unary_costs: tuple[np.ndarray, ...]
    transition_costs: tuple[np.ndarray, ...]
    allowed_edges: tuple[np.ndarray, ...]
    source_ref: str

    def __post_init__(self):
        if type(self.identity_key) is not tuple or len(self.identity_key) != 2:
            raise ValueError("identity_key: persistent joint actor/object key required")
        for key in self.identity_key:
            _text(key, "identity_key")
        _text(self.source_ref, "source_ref")
        indices = np.asarray(self.frame_index)
        if (np.ma.isMaskedArray(self.frame_index) or indices.dtype != np.int64
                or indices.ndim != 1 or not len(indices)
                or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
            raise ValueError("Complete original int64 arange(T) required; no dropping/reindexing")
        count = len(indices)
        if (type(self.nodes) is not tuple or len(self.nodes) != count
                or any(type(layer) is not tuple or not layer
                       or any(type(node) is not IdentityNode for node in layer)
                       for layer in self.nodes)):
            raise ValueError("nodes: one nonempty tuple of IdentityNode per original frame required")
        orders = []
        for layer in self.nodes:
            if len({node.state_key for node in layer}) != len(layer):
                raise ValueError("Duplicate state_key in one layer")
            orders.append(np.array(sorted(range(len(layer)), key=lambda i: layer[i].state_key)))
        for name, values, length in (("unary_costs", self.unary_costs, count),
                                     ("transition_costs", self.transition_costs, count - 1),
                                     ("allowed_edges", self.allowed_edges, count - 1)):
            if type(values) is not tuple or len(values) != length:
                raise ValueError(f"{name}: tuple length {length} required")
        unaries, transitions, allowed = [], [], []
        for t, layer in enumerate(self.nodes):
            unary = _array(self.unary_costs[t], "unary_costs", (len(layer),))
            unaries.append(_readonly(unary[orders[t]]))
            if t < count - 1:
                shape = (len(layer), len(self.nodes[t + 1]))
                order = np.ix_(orders[t], orders[t + 1])
                transitions.append(_readonly(_array(self.transition_costs[t], "transition_costs", shape)[order]))
                allowed.append(_readonly(_array(self.allowed_edges[t], "allowed_edges", shape, boolean=True)[order]))
        object.__setattr__(self, "frame_index", _readonly(indices))
        object.__setattr__(self, "nodes", tuple(tuple(layer[i] for i in order)
                                               for layer, order in zip(self.nodes, orders)))
        object.__setattr__(self, "unary_costs", tuple(unaries))
        object.__setattr__(self, "transition_costs", tuple(transitions))
        object.__setattr__(self, "allowed_edges", tuple(allowed))


@dataclass(frozen=True, eq=False)
class IdentityPath:
    """One raw optimum with full-T observed support and local cost diagnostics.

    min_marginals[t][j] is the best full-path cost constrained to state j in
    graph.nodes[t]; infinity means no such path. local_regrets subtract the
    graph optimum. alternative_evidence_costs[t] minimizes over evidence keys
    different from the selected key, so a same-evidence nuisance duplicate is
    not misreported as observation ambiguity. Infinity means no alternative.
    Floating-point regrets are raw, not clipped or thresholded confidences.
    """
    identity_key: tuple[str, str]
    frame_index: np.ndarray
    state_keys: tuple[str, ...]
    evidence_keys: tuple[str | None, ...]
    supported: np.ndarray
    total_cost: float
    min_marginals: tuple[np.ndarray, ...]
    local_regrets: tuple[np.ndarray, ...]
    alternative_evidence_costs: np.ndarray
    status: str


@dataclass(frozen=True)
class IdentityRanking:
    """All supplied identities, ordered by (cost, joint key), without acceptance.

    The first two entries are different joint hypotheses, not merely two paths
    of the same identity. A tie's deterministic key order proves nothing. Gap
    scale depends on supplied costs and T; it is not a posterior probability.
    """
    ranked_paths: tuple[IdentityPath, ...]
    identity_cost_gap: float | None


def _solve_graph(graph):
    count = len(graph.nodes)
    forward = [graph.unary_costs[0].copy()]
    predecessors = []
    for t in range(1, count):
        edge = np.where(graph.allowed_edges[t - 1], graph.transition_costs[t - 1], np.inf)
        costs = forward[-1][:, None] + edge + graph.unary_costs[t][None, :]
        previous = np.argmin(costs, axis=0)
        current = costs[previous, np.arange(len(graph.nodes[t]))]
        if not np.isfinite(current).any():
            raise NoFeasiblePathError(f"Identity {graph.identity_key!r}: no feasible full path at frame {t}")
        predecessors.append(previous)
        forward.append(current)
    selected = [int(np.argmin(forward[-1]))]
    for previous in reversed(predecessors):
        selected.append(int(previous[selected[-1]]))
    selected.reverse()
    total = float(forward[-1][selected[-1]])
    backward = [None] * count
    backward[-1] = np.zeros(len(graph.nodes[-1]))
    for t in range(count - 2, -1, -1):
        edge = np.where(graph.allowed_edges[t], graph.transition_costs[t], np.inf)
        backward[t] = np.min(edge + graph.unary_costs[t + 1][None, :] + backward[t + 1][None, :], axis=1)
    marginals = tuple(_readonly(first + last) for first, last in zip(forward, backward))
    regrets = tuple(_readonly(value - total) for value in marginals)
    evidence = tuple(layer[j].evidence_key for layer, j in zip(graph.nodes, selected))
    supported = _readonly(np.array([key is not None for key in evidence], dtype=bool))
    alternatives = []
    for layer, values, key in zip(graph.nodes, marginals, evidence):
        costs = [values[j] for j, node in enumerate(layer) if node.evidence_key != key]
        alternatives.append(min(costs, default=np.inf))
    return IdentityPath(graph.identity_key, _readonly(graph.frame_index),
                        tuple(layer[j].state_key for layer, j in zip(graph.nodes, selected)),
                        evidence, supported, total, marginals, regrets,
                        _readonly(np.array(alternatives)),
                        "OBSERVED_PATH" if supported.any() else "NO_EVIDENCE")


def rank_identity_paths(graphs: Sequence[IdentityGraph]) -> IdentityRanking:
    """Solve every supplied graph and return raw costs, never an accepted actor.

    No low-coverage or all-missing competitor is deleted before ranking. An
    infeasible graph aborts the whole bank rather than silently removing a
    potentially important competitor. Malformed/infeasible inputs are engineering
    errors, distinct from scientific abstention on unsupported/ambiguous results.
    There are no new weights, thresholds, speed limits or camera assumptions.

    Ties follow canonical state keys and joint identity keys, not input order.
    Runtime is O(sum_Z sum_t K_t*K_(t+1)); stored input edges and forward/backward
    diagnostics dominate memory. Bounds are for a small supplied hypothesis bank.
    Full-T masks remain false at missing states even if another module estimates
    a pose there. Integration requires a separately frozen external calibration
    and uncertainty protocol; this solver cannot prove an identity is observable.
    """
    if (not isinstance(graphs, Sequence) or not graphs
            or any(type(graph) is not IdentityGraph for graph in graphs)):
        raise ValueError("A nonempty sequence of IdentityGraph is required")
    if len({graph.identity_key for graph in graphs}) != len(graphs):
        raise ValueError("Duplicate joint identity_key in bank")
    if any(not np.array_equal(graph.frame_index, graphs[0].frame_index) for graph in graphs):
        raise ValueError("Every graph must use the same full original timeline")
    try:
        with np.errstate(over="raise", invalid="raise"):
            paths = tuple(sorted((_solve_graph(graph) for graph in sorted(graphs, key=lambda g: g.identity_key)),
                                 key=lambda path: (path.total_cost, path.identity_key)))
            gap = None if len(paths) < 2 else float(np.subtract(paths[1].total_cost, paths[0].total_cost))
    except FloatingPointError as error:
        raise ValueError("Identity path costs overflowed; check supplied finite cost magnitudes") from error
    return IdentityRanking(paths, gap)

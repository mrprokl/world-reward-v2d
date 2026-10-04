"""Tiny data-free graphs; exhaustive reference does not reuse the DP recurrence."""
from dataclasses import replace
import itertools

import numpy as np
import pytest

from world_reward.pose_selection import NoFeasiblePathError
from world_reward.temporal_identity import (
    IdentityGraph, IdentityNode, rank_identity_paths,
)


def graph(key=("actor", "object"), sizes=(2, 2, 2), *, nodes=None, unary=None,
          transitions=None, allowed=None):
    return IdentityGraph(
        key, np.arange(len(sizes), dtype=np.int64),
        tuple(tuple(IdentityNode(f"state-{j}", None if j == 0 else f"evidence-{t}-{j}")
                    for j in range(size)) for t, size in enumerate(sizes)) if nodes is None else nodes,
        tuple(np.zeros(size) for size in sizes) if unary is None else unary,
        tuple(np.zeros((a, b)) for a, b in zip(sizes, sizes[1:])) if transitions is None else transitions,
        tuple(np.ones((a, b), dtype=bool) for a, b in zip(sizes, sizes[1:])) if allowed is None else allowed,
        "data-free-synthetic-evidence-v1",
    )


def exhaustive(g):
    paths = []
    for states in itertools.product(*(range(len(layer)) for layer in g.nodes)):
        if all(g.allowed_edges[t][a, b] for t, (a, b) in enumerate(zip(states, states[1:]))):
            cost = sum(float(g.unary_costs[t][j]) for t, j in enumerate(states))
            cost += sum(float(g.transition_costs[t][a, b]) for t, (a, b) in enumerate(zip(states, states[1:])))
            paths.append((cost, states))
    minimum = min(cost for cost, _ in paths)
    marginals = tuple(np.array([min((cost for cost, states in paths if states[t] == j), default=np.inf)
                               for j in range(len(layer))]) for t, layer in enumerate(g.nodes))
    return minimum, marginals, paths


@pytest.mark.parametrize("seed", [7610, 7611, 7612])
def test_costs_and_all_local_min_marginals_match_exhaustive_paths(seed):
    rng = np.random.default_rng(seed)
    for sizes in [(3,), (2, 3), (2, 3, 2, 2)]:
        permitted = tuple(rng.random((a, b)) > .4 for a, b in zip(sizes, sizes[1:]))
        for mask in permitted:
            mask[0, 0] = True  # Keep at least one feasible path, without forcing observed support.
        g = graph(sizes=sizes,
                  unary=tuple(rng.integers(-5, 6, size=size) for size in sizes),
                  transitions=tuple(rng.integers(-3, 4, size=(a, b)) for a, b in zip(sizes, sizes[1:])),
                  allowed=permitted)
        expected, marginals, paths = exhaustive(g)
        result = rank_identity_paths([g]).ranked_paths[0]
        selected = tuple([node.state_key for node in layer].index(key) for layer, key in zip(g.nodes, result.state_keys))
        assert (result.total_cost, selected) in paths
        assert result.total_cost == expected
        for actual, reference, regret in zip(result.min_marginals, marginals, result.local_regrets):
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(regret, reference - expected)
        for t, key in enumerate(result.evidence_keys):
            alternative = min((marginals[t][j] for j, node in enumerate(g.nodes[t])
                               if node.evidence_key != key), default=np.inf)
            assert result.alternative_evidence_costs[t] == alternative


def test_future_evidence_selects_global_path_and_keeps_explicit_missing_support():
    nodes = ((IdentityNode("a", "obs0"), IdentityNode("b", "wrong0")),
             (IdentityNode("a", None), IdentityNode("b", None)),
             (IdentityNode("a", "obs2"), IdentityNode("b", "wrong2")))
    g = graph(nodes=nodes, unary=(np.array([1., 0.]), np.zeros(2), np.array([0., 5.])),
              transitions=(np.array([[0., 8.], [8., 0.]]),) * 2)
    result = rank_identity_paths([g]).ranked_paths[0]
    assert result.state_keys == ("a",) * 3
    assert result.evidence_keys == ("obs0", None, "obs2")
    assert result.supported.tolist() == [True, False, True]
    assert result.status == "OBSERVED_PATH"
    assert result.identity_key == ("actor", "object")


def test_same_identity_nuisance_paths_do_not_become_second_identity():
    nodes = ((IdentityNode("a", "same-observation"), IdentityNode("b", "same-observation"),
              IdentityNode("c", "different-observation")),)
    first = graph(key=("a", "o"), sizes=(3,), nodes=nodes, unary=(np.array([0., .01, 3.]),))
    second = graph(key=("b", "o"), sizes=(1,), unary=(np.array([2.]),))
    ranking = rank_identity_paths([second, first])
    assert ranking.identity_cost_gap == 2
    assert [path.identity_key for path in ranking.ranked_paths] == [("a", "o"), ("b", "o")]
    assert ranking.ranked_paths[0].alternative_evidence_costs.tolist() == [3.]
    np.testing.assert_array_equal(ranking.ranked_paths[0].local_regrets[0], [0, .01, 3])


def test_all_missing_and_low_coverage_competitors_remain_ranked_not_bystander_filtered():
    absent = graph(key=("absent", "o"), sizes=(1, 1, 1, 1),
                   unary=tuple(np.array([-1.]) for _ in range(4)))
    limited_nodes = tuple((IdentityNode("s", "real" if t == 0 else None),) for t in range(4))
    limited = graph(key=("limited", "o"), sizes=(1,) * 4, nodes=limited_nodes)
    visible_nodes = tuple((IdentityNode("s", f"bystander-{t}"),) for t in range(4))
    bystander = graph(key=("bystander", "o"), sizes=(1,) * 4, nodes=visible_nodes,
                      unary=tuple(np.array([2.]) for _ in range(4)))
    result = rank_identity_paths([bystander, limited, absent])
    assert [path.identity_key[0] for path in result.ranked_paths] == ["absent", "limited", "bystander"]
    assert result.ranked_paths[0].status == "NO_EVIDENCE"
    assert not result.ranked_paths[0].supported.any()
    assert result.ranked_paths[1].supported.tolist() == [True, False, False, False]


def test_no_pose_mesh_contact_or_accepted_identity_is_manufactured():
    ranking = rank_identity_paths([graph(sizes=(1,))])
    path = ranking.ranked_paths[0]
    assert ranking.identity_cost_gap is None
    assert path.evidence_keys == (None,)
    for name in ("accepted_identity", "probability", "pose", "mesh", "contact", "visibility"):
        assert not hasattr(path, name) and not hasattr(ranking, name)


def test_identity_ties_are_key_deterministic_but_have_zero_gap():
    a, b = graph(key=("a", "o")), graph(key=("b", "o"))
    forward, reverse = rank_identity_paths([a, b]), rank_identity_paths([b, a])
    assert forward.identity_cost_gap == reverse.identity_cost_gap == 0
    assert forward.ranked_paths[0].identity_key == reverse.ranked_paths[0].identity_key == ("a", "o")
    assert forward.ranked_paths[0].state_keys == ("state-0",) * 3
    assert all(np.array_equal(m, np.zeros(2)) for m in forward.ranked_paths[0].local_regrets)


def test_canonical_state_sorting_reorders_costs_and_edges_without_changing_answer():
    g = graph(unary=(np.array([4., 0.]), np.array([0., 3.]), np.array([5., -1.])),
              transitions=(np.array([[1., 8.], [7., 2.]]), np.array([[0., 8.], [3., -1.]])),
              allowed=(np.array([[True, False], [True, True]]), np.ones((2, 2), dtype=bool)))
    orders = (np.array([1, 0]), np.array([0, 1]), np.array([1, 0]))
    permuted = replace(g, nodes=tuple(tuple(layer[j] for j in order) for layer, order in zip(g.nodes, orders)),
                       unary_costs=tuple(cost[order] for cost, order in zip(g.unary_costs, orders)),
                       transition_costs=tuple(cost[np.ix_(orders[t], orders[t+1])]
                                              for t, cost in enumerate(g.transition_costs)),
                       allowed_edges=tuple(mask[np.ix_(orders[t], orders[t+1])]
                                           for t, mask in enumerate(g.allowed_edges)))
    a, b = rank_identity_paths([g]).ranked_paths[0], rank_identity_paths([permuted]).ranked_paths[0]
    assert a.state_keys == b.state_keys and a.total_cost == b.total_cost
    for x, y in zip(a.min_marginals, b.min_marginals):
        np.testing.assert_array_equal(x, y)


def test_forbidden_low_cost_edges_not_used_and_unreachable_states_are_infinite_diagnostics():
    g = graph(sizes=(2, 2), transitions=(np.array([[7., -1000.], [-1000., -1000.]]),),
              allowed=(np.array([[True, False], [False, False]]),))
    path = rank_identity_paths([g]).ranked_paths[0]
    assert path.total_cost == 7
    assert path.state_keys == ("state-0", "state-0")
    for marginal in path.min_marginals:
        np.testing.assert_array_equal(marginal, [7, np.inf])


def test_one_infeasible_competitor_aborts_entire_bank_without_removal():
    impossible = graph(key=("bad", "o"), allowed=(np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)))
    with pytest.raises(NoFeasiblePathError, match="no feasible full path at frame 1"):
        rank_identity_paths([graph(), impossible])


def test_inputs_and_results_own_readonly_arrays_without_aliases():
    indices = np.arange(2, dtype=np.int64)
    unary = (np.array([0., 1.]), np.array([0., 1.]))
    transitions, allowed = (np.zeros((2, 2)),), (np.ones((2, 2), dtype=bool),)
    g = replace(graph(sizes=(2, 2)), frame_index=indices, unary_costs=unary,
                transition_costs=transitions, allowed_edges=allowed)
    path = rank_identity_paths([g]).ranked_paths[0]
    indices[:] = 99; unary[0][:] = 99; transitions[0][:] = 99; allowed[0][:] = False
    assert path.total_cost == 0
    arrays = [g.frame_index, *g.unary_costs, *g.transition_costs, *g.allowed_edges,
              path.frame_index, path.supported, *path.min_marginals, *path.local_regrets,
              path.alternative_evidence_costs]
    for array in arrays:
        assert array.flags.owndata and not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 7
    assert not np.shares_memory(path.frame_index, g.frame_index)
    assert not any(np.shares_memory(a, b) for a in path.min_marginals for b in g.unary_costs)


@pytest.mark.parametrize("change", [
    {"identity_key": "actor"}, {"identity_key": ("", "object")}, {"source_ref": "\n"},
    {"frame_index": np.array([0, 2, 3], dtype=np.int64)},
    {"frame_index": np.arange(3, dtype=np.int32)}, {"frame_index": np.empty(0, dtype=np.int64)},
    {"frame_index": np.ma.array(np.arange(3, dtype=np.int64))},
    {"nodes": ()}, {"nodes": ((), (), ())},
    {"nodes": ((IdentityNode("same", None), IdentityNode("same", "obs")),) * 3},
    {"unary_costs": [np.zeros(2)] * 3}, {"unary_costs": (np.zeros(3),) * 3},
    {"unary_costs": (np.full(2, np.nan),) * 3},
    {"unary_costs": (np.array([True, False]),) * 3},
    {"unary_costs": (np.zeros(2, dtype=complex),) * 3},
    {"unary_costs": (np.ma.array(np.zeros(2)),) * 3},
    {"transition_costs": ()}, {"transition_costs": (np.zeros((3, 2)),) * 2},
    {"transition_costs": (np.full((2, 2), np.inf),) * 2},
    {"allowed_edges": (np.ones((2, 2)),) * 2}, {"allowed_edges": ()},
])
def test_invalid_graphs_fail_explicitly(change):
    with pytest.raises(ValueError):
        replace(graph(), **change)


@pytest.mark.parametrize("args", [("", None), ("state", ""), ("state\n", None), ("s", 42)])
def test_bad_node_references_fail(args):
    with pytest.raises(ValueError):
        IdentityNode(*args)


def test_bank_validation_and_cost_overflow_fail_without_retuning():
    for bank in ([], None, [object()], [graph(), graph()], [graph(), graph(key=("b", "o"), sizes=(2,))]):
        with pytest.raises(ValueError):
            rank_identity_paths(bank)
    large = graph(sizes=(1, 1), unary=(np.array([1e308]),) * 2)
    with pytest.raises(ValueError, match="overflowed"):
        rank_identity_paths([large])
    a = graph(key=("a", "o"), sizes=(1,), unary=(np.array([-1e308]),))
    b = graph(key=("b", "o"), sizes=(1,), unary=(np.array([1e308]),))
    with pytest.raises(ValueError, match="overflowed"):
        rank_identity_paths([a, b])


def test_common_unary_offset_changes_cost_not_path_or_local_regrets():
    g = graph(unary=(np.array([1., 0.]), np.array([0., 2.]), np.array([0., 3.])))
    shifted = replace(g, unary_costs=tuple(cost + 5 for cost in g.unary_costs))
    a, b = rank_identity_paths([g]).ranked_paths[0], rank_identity_paths([shifted]).ranked_paths[0]
    assert a.state_keys == b.state_keys and b.total_cost == a.total_cost + 15
    for x, y in zip(a.local_regrets, b.local_regrets):
        np.testing.assert_array_equal(x, y)

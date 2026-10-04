"""Tiny manufactured relations only; no native or mesh qualification claims."""
from dataclasses import FrozenInstanceError
from itertools import permutations, product

import numpy as np
import pytest

from world_reward.oriented_solid_forest import (
    OrientedSolidForest, adjudicate_oriented_solid_forest,
)


def ancestor_matrix(parents):
    """Independent parent traversal reference; cycles are not forests."""
    count = len(parents)
    result = np.zeros((count, count), dtype=bool)
    for i in range(count):
        seen, current = {i}, parents[i]
        while current != -1:
            if current in seen:
                return None
            seen.add(current)
            result[i, current] = True
            current = parents[current]
    return result


def relation(parents):
    parents = np.array(parents, dtype=np.int64)
    inside = ancestor_matrix(parents)
    assert inside is not None
    signs = np.where(inside.sum(axis=1) % 2 == 0, 1, -1).astype(np.int64)
    return signs, inside


@pytest.mark.parametrize("parents", [
    (-1, 0, 0),       # One outward root, two inward cavities.
    (-1, 0, -1, 2),   # Two disconnected hollow solids.
    (-1, 0, 1),       # Positive island inside an inward void boundary.
    (-1,),            # A single outward solid.
])
def test_fresh_material_forests_preserve_every_key_and_component(parents):
    signs, inside = relation(parents)
    keys = tuple(f"original-birth-{i}" for i in range(len(parents)))
    result = adjudicate_oriented_solid_forest(keys, signs, inside)
    assert result.component_keys == keys
    np.testing.assert_array_equal(result.parents, parents)
    np.testing.assert_array_equal(result.depths, inside.sum(axis=1))
    np.testing.assert_array_equal(result.signs, signs)
    np.testing.assert_array_equal(result.inside, inside)
    # Every cell has winding 0 or 1, including a void containing an island.
    np.testing.assert_array_equal(
        inside.astype(np.int64) @ signs + signs, (result.depths + 1) % 2)


@pytest.mark.parametrize("parents", [(-1, 0, 0), (-1, 0, -1, 2), (-1, 0, 1)])
def test_input_permutations_relabel_only_indices_not_lineage_or_decision(parents):
    signs, inside = relation(parents)
    count = len(parents)
    keys = tuple(f"birth-{i}" for i in range(count))
    for order in permutations(range(count)):
        ids = np.array(order)
        result = adjudicate_oriented_solid_forest(
            tuple(keys[i] for i in order), signs[ids], inside[np.ix_(ids, ids)])
        mapping = dict(zip(result.component_keys, (
            None if p == -1 else result.component_keys[p] for p in result.parents)))
        assert mapping == {keys[i]: None if p == -1 else keys[p]
                           for i, p in enumerate(parents)}
        assert result.component_keys == tuple(keys[i] for i in order)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_all_tiny_graphs_and_signs_against_independent_parent_enumeration(count):
    reference = {}
    for parents in product(range(-1, count), repeat=count):
        matrix = ancestor_matrix(parents)
        if matrix is not None:
            reference[matrix.tobytes()] = parents
    edges = tuple((i, j) for i in range(count) for j in range(count) if i != j)
    keys = tuple(f"component-{i}" for i in range(count))
    for values in product((False, True), repeat=len(edges)):
        inside = np.zeros((count, count), dtype=bool)
        for edge, value in zip(edges, values):
            inside[edge] = value
        parents = reference.get(inside.tobytes())
        for values in product((-1, 1), repeat=count):
            signs = np.array(values, dtype=np.int64)
            valid = parents is not None and np.array_equal(
                signs, np.where(inside.sum(axis=1) % 2 == 0, 1, -1))
            if valid:
                result = adjudicate_oriented_solid_forest(keys, signs, inside)
                np.testing.assert_array_equal(result.parents, parents)
            else:
                with pytest.raises(ValueError):
                    adjudicate_oriented_solid_forest(keys, signs, inside)


@pytest.mark.parametrize("inside,message", [
    ([[True, False], [False, False]], "itself"),
    ([[False, True], [True, False]], "Mutual"),
    ([[False, True, False], [False, False, True], [False, False, False]], "transitive"),
    ([[False, True, True], [False, False, False], [False, False, False]], "Incomparable"),
    ([[False, True, False], [False, False, True], [True, False, False]], "transitive"),
])
def test_inconsistent_graph_fails_instead_of_repairing(inside, message):
    matrix = np.array(inside, dtype=bool)
    with pytest.raises(ValueError, match=message):
        adjudicate_oriented_solid_forest(
            tuple(f"birth-{i}" for i in range(len(matrix))),
            np.ones(len(matrix), dtype=np.int64), matrix)


@pytest.mark.parametrize("parents,signs", [
    ((-1, 0), [1, 1]),          # Nested outward-outward creates winding 2.
    ((-1,), [-1]),             # Uncontained inward boundary.
    ((-1, 0, 1), [1, -1, -1]),
])
def test_invalid_material_orientation_never_flips_drops_or_reorders(parents, signs):
    _, inside = relation(parents)
    original = np.array(signs, dtype=np.int64)
    before = original.tobytes(), inside.tobytes()
    with pytest.raises(ValueError, match="orientation"):
        adjudicate_oriented_solid_forest(
            tuple(f"birth-{i}" for i in range(len(parents))), original, inside)
    assert (original.tobytes(), inside.tobytes()) == before


def test_deep_forest_is_not_special_cased_to_two_shells():
    parents = tuple(range(-1, 63))
    signs, inside = relation(parents)
    result = adjudicate_oriented_solid_forest(
        tuple(f"birth-{i}" for i in range(64)), signs, inside)
    np.testing.assert_array_equal(result.parents, parents)
    np.testing.assert_array_equal(result.depths, np.arange(64))


@pytest.mark.parametrize("keys,signs,inside", [
    ((), np.zeros(0, np.int64), np.zeros((0, 0), bool)),
    (["a"], np.ones(1, np.int64), np.zeros((1, 1), bool)),
    (("a", "a"), np.ones(2, np.int64), np.zeros((2, 2), bool)),
    (("\n",), np.ones(1, np.int64), np.zeros((1, 1), bool)),
    ((1,), np.ones(1, np.int64), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, np.int32), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, ">i8"), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, float), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, bool), np.zeros((1, 1), bool)),
    (("a",), np.zeros(1, np.int64), np.zeros((1, 1), bool)),
    (("a",), np.array([2], np.int64), np.zeros((1, 1), bool)),
    (("a",), np.ones((1, 1), np.int64), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, np.int64), np.zeros((1, 1), np.int64)),
    (("a",), np.ones(1, np.int64), np.zeros((1, 2), bool)),
    (("a",), np.ma.array([1], dtype=np.int64), np.zeros((1, 1), bool)),
    (("a",), np.ones(1, np.int64), np.ma.array([[False]])),
])
def test_shapes_keys_foreign_dtypes_and_masked_values_are_not_coerced(keys, signs, inside):
    with pytest.raises(ValueError):
        adjudicate_oriented_solid_forest(keys, signs, inside)


def test_result_has_owned_immutable_arrays_and_no_geometry_certificate():
    signs, inside = relation((-1, 0, 1))
    result = adjudicate_oriented_solid_forest(("root", "void", "island"), signs, inside)
    for array in (result.signs, result.inside, result.parents, result.depths):
        assert not array.flags.writeable
        assert not np.shares_memory(array, signs) and not np.shares_memory(array, inside)
        with pytest.raises(ValueError):
            array.flags.writeable = True
    signs[:] = 0
    inside[:] = False
    np.testing.assert_array_equal(result.parents, [-1, 0, 1])
    np.testing.assert_array_equal(result.signs, [1, -1, 1])
    with pytest.raises(FrozenInstanceError):
        result.component_keys = ("replacement",)
    assert "certified" not in result.__dataclass_fields__
    assert "contact" not in result.__dataclass_fields__
    with pytest.raises(ValueError):
        OrientedSolidForest(("invalid",), np.array([-1], np.int64), np.zeros((1, 1), bool))

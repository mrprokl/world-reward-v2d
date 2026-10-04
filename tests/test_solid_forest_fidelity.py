"""Tiny full-array birth controls; no native or embedding certification."""
from dataclasses import FrozenInstanceError
from itertools import permutations

import numpy as np
import pytest

from world_reward.oriented_solid_forest import adjudicate_oriented_solid_forest
from world_reward.solid_forest_fidelity import compare_solid_forest_fidelity


def forest(parents, prefix):
    count = len(parents)
    inside = np.zeros((count, count), bool)
    for i, p in enumerate(parents):
        while p >= 0:
            inside[i, p] = True; p = parents[p]
    signs = np.where(inside.sum(axis=1) % 2 == 0, 1, -1).astype(np.int64)
    return adjudicate_oriented_solid_forest(tuple(f"{prefix}-{i}" for i in range(count)), signs, inside)


def inputs(parents=(-1, 0, 0), order=None):
    count = len(parents)
    if order is None:
        order = tuple(range(count))
    source = forest(parents, "source-born")
    ids = np.array(order, np.int64)
    candidate = adjudicate_oriented_solid_forest(tuple(f"native-label-{i}" for i in range(count)),
                                               source.signs[ids], source.inside[np.ix_(ids, ids)])
    tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    source_faces = np.vstack([tetra + 4 * i for i in range(count)])
    candidate_faces = source_faces.copy()
    face_birth = np.concatenate([np.arange(4 * s, 4 * s + 4, dtype=np.int64) for s in order])
    vertex_birth = np.concatenate([np.arange(4 * s, 4 * s + 4, dtype=np.int64) for s in order])
    return dict(source_forest=source, candidate_forest=candidate, source_faces=source_faces,
                candidate_faces=candidate_faces, source_face_components=np.repeat(np.arange(count, dtype=np.int64), 4),
                candidate_face_components=np.repeat(np.arange(count, dtype=np.int64), 4),
                face_birth=face_birth, vertex_birth=vertex_birth, source_vertex_count=count * 4)


@pytest.mark.parametrize("parents", [(-1, 0, 0), (-1, 0, -1, 2), (-1, 0, 1), (-1, -1, -1)])
def test_all_original_solid_forests_match_through_authoritative_birth_permutations(parents):
    for order in permutations(range(len(parents))):
        args = inputs(parents, order)
        result = compare_solid_forest_fidelity(**args)
        assert result.source_component_keys == args["source_forest"].component_keys
        assert result.candidate_component_keys == args["candidate_forest"].component_keys
        np.testing.assert_array_equal(result.candidate_to_source, order)
        np.testing.assert_array_equal(result.source_to_candidate, np.argsort(order))
        np.testing.assert_array_equal(result.face_birth, args["face_birth"])
        np.testing.assert_array_equal(result.vertex_birth, args["vertex_birth"])


def test_birth_faces_can_collapse_without_requiring_face_or_vertex_injectivity():
    args = inputs((-1,))
    # A source face need not survive; distinct candidate faces may share a birth.
    args["face_birth"] = np.array([0, 0, 2, 3], np.int64)
    result = compare_solid_forest_fidelity(**args)
    np.testing.assert_array_equal(result.face_birth, [0, 0, 2, 3])
    # Whether these transactions are geometrically legal is a separate gate.


def test_face_order_changes_are_not_component_order_assumptions():
    args = inputs((-1, 0, 1), (2, 0, 1))
    rows = np.array([7, 2, 11, 0, 9, 3, 8, 4, 1, 10, 6, 5])
    for name in ("candidate_faces", "candidate_face_components", "face_birth"):
        args[name] = args[name][rows]
    result = compare_solid_forest_fidelity(**args)
    np.testing.assert_array_equal(result.candidate_to_source, [2, 0, 1])


@pytest.mark.parametrize("kind", ["face_merge", "vertex_cross", "split", "orphan_birth", "unused_candidate"])
def test_bad_native_births_merge_split_or_cross_shell_fail(kind):
    args = inputs((-1, -1, -1))
    if kind == "face_merge":
        args["face_birth"][0] = 4
    elif kind == "vertex_cross":
        args["vertex_birth"][0] = 4
    elif kind == "split":
        args["face_birth"][4:8] = [0, 1, 2, 3]
        args["vertex_birth"][4:8] = [0, 1, 2, 3]
    elif kind == "orphan_birth":
        args["source_vertex_count"] += 1
        args["vertex_birth"][0] = 12
    elif kind == "unused_candidate":
        args["vertex_birth"] = np.r_[args["vertex_birth"], np.int64(0)]
    with pytest.raises(ValueError):
        compare_solid_forest_fidelity(**args)


def test_component_drop_is_not_excused_by_positive_net_volume_or_keys():
    args = inputs((-1, 0, 1))
    args["candidate_forest"] = forest((-1, 0), "remaining")
    with pytest.raises(ValueError, match="retained bijectively"):
        compare_solid_forest_fidelity(**args)


def test_source_orphan_is_rejected_even_when_no_birth_uses_it():
    args = inputs()
    args["source_vertex_count"] += 1
    with pytest.raises(ValueError, match="Unreferenced original"):
        compare_solid_forest_fidelity(**args)


def test_cavities_with_same_sign_depth_may_not_swap_original_parent_ownership():
    args = inputs((-1, 0, -1, 2))
    args["candidate_forest"] = forest((-1, 2, -1, 0), "candidate")
    with pytest.raises(ValueError, match="containment"):
        compare_solid_forest_fidelity(**args)


def test_source_components_sharing_indexed_vertices_fail_not_repaired():
    args = inputs((-1, -1))
    args["source_faces"][4, 0] = 0
    with pytest.raises(ValueError, match="share"):
        compare_solid_forest_fidelity(**args)


@pytest.mark.parametrize("field,value", [
    ("face_birth", np.arange(12, dtype=np.int32)),
    ("face_birth", np.arange(12, dtype=float)),
    ("face_birth", np.array([-1] + list(range(1, 12)), np.int64)),
    ("face_birth", np.array([12] + list(range(1, 12)), np.int64)),
    ("vertex_birth", np.array([12] + list(range(1, 12)), np.int64)),
    ("source_face_components", np.zeros(12, bool)),
    ("candidate_face_components", np.zeros(12, np.int64)),
    ("source_vertex_count", np.int64(12)),
    ("candidate_faces", np.zeros((12, 3), np.int64)),
])
def test_dtype_bounds_missing_components_and_padding_are_not_silently_coerced(field, value):
    args = inputs(); args[field] = value
    with pytest.raises(ValueError):
        compare_solid_forest_fidelity(**args)


def test_owned_readonly_mapping_result_retains_births_and_never_edits_inputs():
    args = inputs((-1, 0, 1), (2, 0, 1))
    before = {k: v.tobytes() for k, v in args.items() if isinstance(v, np.ndarray)}
    result = compare_solid_forest_fidelity(**args)
    for value in result.__dict__.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            assert not any(np.shares_memory(value, v) for v in args.values() if isinstance(v, np.ndarray))
            with pytest.raises(ValueError):
                value.flags.writeable = True
    assert before == {k: v.tobytes() for k, v in args.items() if isinstance(v, np.ndarray)}
    args["face_birth"][:] = 0
    np.testing.assert_array_equal(result.candidate_to_source, [2, 0, 1])
    with pytest.raises(FrozenInstanceError):
        result.source_component_keys = ("replacement",)

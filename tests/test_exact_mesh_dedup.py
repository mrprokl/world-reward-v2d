"""Tiny procedural arrays only; no local data/renderer/GPU/model inference."""
import numpy as np
import pytest

import world_reward.exact_mesh_dedup as module


MESH = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.]], np.float64)


def test_unchanged_initial_fitted_restore_every_hypothesis_in_original_order():
    # Caller interleaves each initial/fitted pair; no selection/acceptance changes.
    a = MESH.copy(); b = MESH + [0, .2, 0]; c = MESH + [.1, 0, 0]
    plan = module.deduplicate_meshes([a, a.copy(), b, c, a.copy(), a.copy()])
    assert plan.first_original_indices == (0, 2, 3)
    assert plan.original_to_unique.tolist() == [0, 0, 1, 2, 0, 0]
    assert plan.restore([.91, .7, .8]) == [.91, .91, .7, .8, .91, .91]
    assert plan.counts == dict(original_meshes=6, unique_meshes=3, reused_meshes=3)
    assert plan.unique[0] is a and not plan.original_to_unique.flags.writeable


def test_no_approximate_cast_or_signed_zero_merging():
    tiny = MESH.copy(); tiny[0, 0] = np.nextafter(0., 1.)
    negative_zero = MESH.copy(); negative_zero[0, 0] = -0.
    plan = module.deduplicate_meshes([MESH, MESH.astype(np.float32), tiny, negative_zero])
    assert len(plan.unique) == 4
    assert plan.original_to_unique.tolist() == [0, 1, 2, 3]


def test_digest_collision_never_merges_different_full_geometry(monkeypatch):
    monkeypatch.setattr(module, '_fingerprint', lambda a: b'forced collision')
    plan = module.deduplicate_meshes([MESH, MESH + [.1, 0, 0], MESH.copy()])
    assert plan.original_to_unique.tolist() == [0, 1, 0]


def test_noncontiguous_layout_with_identical_full_order_preserves_original_values():
    a = np.asfortranarray(MESH)
    plan = module.deduplicate_meshes([a, MESH])
    assert len(plan.unique) == 1 and plan.unique[0] is a
    np.testing.assert_array_equal(a, MESH)


def test_new_frame_plan_has_no_cache_or_cross_frame_score_reuse():
    a = module.deduplicate_meshes([MESH, MESH.copy()])
    b = module.deduplicate_meshes([MESH, MESH.copy()])
    assert a is not b and a.restore([.2]) == [.2, .2] and b.restore([.9]) == [.9, .9]


def test_batches_bound_four_and_restore_all_slots_no_mesh_dropped():
    meshes = [MESH + [.01*i, 0, 0] for i in range(11)]
    plan = module.deduplicate_meshes(meshes + [meshes[0]])
    batches = list(plan.batches())
    assert [len(x) for x in batches] == [4, 4, 3]
    assert [id(x) for batch in batches for x in batch] == [id(x) for x in meshes]
    assert plan.restore(list(range(11))) == list(range(11)) + [0]


@pytest.mark.parametrize('batch_size', [0, 5, True, 1.5])
def test_invalid_batch_sizes_fail(batch_size):
    with pytest.raises(ValueError): list(module.deduplicate_meshes([MESH]).batches(batch_size))


def test_result_count_mismatch_fails_instead_of_dropping_hypotheses():
    with pytest.raises(ValueError): module.deduplicate_meshes([MESH]).restore([])


@pytest.mark.parametrize('meshes', [[], [MESH.astype(int)], [MESH[:-1]],
    [np.ones((3, 4))], [MESH, np.ones((4, 3))], [np.full((3, 3), np.nan)]])
def test_invalid_full_meshes_not_silently_discarded(meshes):
    with pytest.raises(ValueError): module.deduplicate_meshes(meshes)

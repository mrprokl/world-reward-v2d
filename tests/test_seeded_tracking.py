import numpy as np
import pytest

from world_reward.seeded_tracking import corner_prompt, native_masks, seed_rows, temporal_summary, singleton_outputs


def test_original_non_square_corner_codec():
    assert corner_prompt([20, 15, 180, 90], 200, 100) == ([[0.1, 0.15], [0.9, 0.9]], [2, 3])
    assert corner_prompt([0, 0, 200, 100], 200, 100) == ([[0, 0], [1, 1]], [2, 3])


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 201, 1],
                                  [True, 0, 1, 1], [0, 0, float('nan'), 1], [0, 0, 1]])
def test_never_repair_invalid_coordinates(box):
    with pytest.raises(ValueError):
        corner_prompt(box, 200, 100)


def saved():
    return [dict(ep=9, index=i, width=4, height=3, rgb_sha256=str(i),
                 request_sha256='f'*64, status='pair_returned',
                 person_bbox=[0, 0, 2, 3], object_bbox=[2, 1, 4, 3]) for i in [0, 14, 29]]


def test_exact_all_prefix_boxes_fixed_ids():
    result = seed_rows(saved(), 9, [0, 14, 29], 4, 3, {i: str(i) for i in [0, 14, 29]})
    assert [r['object_id'] for r in result] == [0, 1, 0, 1, 0, 1]
    assert [r['frame_index'] for r in result] == [0, 0, 14, 14, 29, 29]


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'sha', 'abstain'])
def test_seed_missing_or_changed_is_not_picked_around(kind):
    rows = saved()
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows.append(rows[0])
    if kind == 'sha': rows[0]['rgb_sha256'] = 'changed'
    if kind == 'abstain': rows[0]['status'] = 'abstained'
    with pytest.raises(ValueError):
        seed_rows(rows, 9, [0, 14, 29], 4, 3, {i: str(i) for i in [0, 14, 29]})


def test_native_sign_and_absence_not_filled():
    masks, presence = native_masks([0, 1], np.ones((2, 1, 3, 4)), np.array([5, -1]), 3, 4)
    assert masks[0].all() and not masks[1].any()
    assert presence.tolist() == [True, False]


def test_native_singletons_share_identity_without_swapping_outputs():
    rows = [(14, [i], None, np.ones((1, 1, 3, 4)) * i, np.ones((1, 1))) for i in range(2)]
    ids, masks, scores = singleton_outputs(rows, 14)
    assert ids == [0, 1] and masks[1].mean() == 1 and len(scores) == 2


@pytest.mark.parametrize('kind', ['collision', 'reordered', 'frame', 'extra'])
def test_singleton_frame_identity_collision_and_row_shape_fail_closed(kind):
    rows = [(14, [i], None, np.ones((1, 1, 3, 4)), np.ones((1, 1))) for i in range(2)]
    if kind == 'collision': rows[1] = (14, [0], *rows[1][2:])
    if kind == 'reordered': rows.reverse()
    if kind == 'frame': rows[1] = (15, *rows[1][1:])
    if kind == 'extra': rows[1] = (*rows[1][:3], np.ones((2, 1, 3, 4)), rows[1][4])
    with pytest.raises(ValueError): singleton_outputs(rows, 14)


@pytest.mark.parametrize('ids', [[1, 0], [0, 1, 2], [0, 0], [1]])
def test_fixed_ids_must_not_swap_or_truncate(ids):
    with pytest.raises(ValueError): native_masks(ids, np.ones((2, 1, 3, 4)), np.ones(2), 3, 4)


def test_diagnostics_retain_empty_runs_and_full_adjacency():
    result = temporal_summary([4, 0, 0, 5], [0., None, 0.])
    assert result['longest_empty_run'] == 2
    assert result['empty_frames'] == 2
    assert result['quality_verified'] is False and result['interpolation'] is False


def test_native_parity_compares_identical_kernel_shapes_and_preserves_padding(monkeypatch):
    """Manufactured shape-sensitive PE: padded tokens aren't bare box arrays."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    import gemini_sam31_track as adapter

    class Torch:
        int32 = np.int32
        @staticmethod
        def tensor(value, **kwargs): return np.asarray(value)
        @staticmethod
        def equal(a, b): return np.array_equal(a, b)
        @staticmethod
        def autocast(**kwargs):
            assert kwargs == dict(device_type='cuda', enabled=False)
            return nullcontext()

    class Encoder:
        input_image_size = [10, 10]
        not_a_point_embed = SimpleNamespace(weight=np.array([[11., 12.]]))
        def _embed_points(self, coords, labels, pad):
            corners = coords + np.array([[[2., 2.], [3., 3.]]])
            if pad:
                # Different batch shape may produce a last-bit kernel change.
                corners = corners + 1e-9
                return np.concatenate((corners, self.not_a_point_embed.weight[None]), axis=1)
            return corners
        def _embed_boxes(self, coords):
            return coords.reshape(1, 2, 2) + np.array([[[2., 2.], [3., 3.]]])

    monkeypatch.setitem(sys.modules, 'torch', Torch)
    predictor = SimpleNamespace(model=SimpleNamespace(tracker=SimpleNamespace(
        model=SimpleNamespace(interactive_sam_prompt_encoder=Encoder()))))
    proof = adapter.prompt_parity(predictor)
    assert proof['native_padding_token_retained'] is True
    assert proof['equal_shape_unpadded_kernel_comparison'] is True

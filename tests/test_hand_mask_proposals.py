"""Procedural original-grid hand evidence and SAM2 native-shape spies only."""
import numpy as np
import pytest

from world_reward.hand_mask_proposals import POINT_INDICES, propose_hand_masks
from world_reward.hand_observations import HandInstances, LandmarkEvidence


def hands(count=1):
    xy = np.empty((count, 21, 2), np.float64)
    for i in range(count):
        xy[i, :, 0] = np.linspace(1+i, 5+i, 21)
        xy[i, :, 1] = np.linspace(2, 7, 21)
    return HandInstances(LandmarkEvidence(xy, np.isfinite(xy).all(axis=-1), "original_image", "pixel"))


def alter(bank, mutate, support=None):
    xy = bank.original_xy.values.copy(); mutate(xy)
    valid = np.isfinite(xy).all(axis=-1) if support is None else support
    return HandInstances(LandmarkEvidence(xy, valid, "original_image", "pixel"))


class Predictor:
    def __init__(self, outputs=None): self.images, self.calls, self.outputs = [], [], outputs

    def set_image(self, image): self.images.append(image)

    def predict(self, **kw):
        self.calls.append({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in kw.items()})
        if self.outputs is not None: return self.outputs[len(self.calls)-1]
        n = len(kw['box']); h, w = self.images[0].shape[:2]
        masks = np.zeros((n, 1, h, w), bool)
        if len(self.calls) == 2: masks[:, :, 2:5, 1:4] = True
        scores = np.arange(n, dtype=np.float32)-.4 if len(self.calls) == 1 else np.arange(n, dtype=np.float32)+1.2
        low = np.zeros((n, 1, 256, 256), np.float32)
        return (masks[0], scores, low[0]) if n == 1 else (masks, scores[:, None], low)


def build(bank=None, model=None, image=None, index=19):
    bank = hands() if bank is None else bank; model = Predictor() if model is None else model
    image = np.zeros((12, 16, 3), np.uint8) if image is None else image
    return propose_hand_masks(image, bank, model, frame_index=index), model


@pytest.mark.parametrize('count', [1, 2, 5])
def test_one_encoder_two_native_batches_all_slots_raw_scores_and_original_indices(count):
    bank = hands(count); result, model = build(bank)
    assert result.frame_index == 19 and result.local_ids == tuple(f'hand:{i:06d}' for i in range(count))
    assert len(model.images) == 1 and len(model.calls) == 2 and result.image_size == (12, 16)
    a, b = model.calls
    assert set(a) == {'box', 'multimask_output'}
    assert set(b) == {'box', 'point_coords', 'point_labels', 'multimask_output'}
    assert a['multimask_output'] is b['multimask_output'] is False
    assert a['box'].dtype == b['box'].dtype == b['point_coords'].dtype == np.float32
    np.testing.assert_array_equal(a['box'], b['box'])
    np.testing.assert_array_equal(b['point_coords'], bank.original_xy.values[:, POINT_INDICES].astype(np.float32))
    assert b['point_labels'].dtype == np.int32 and b['point_labels'].shape == (count, 17) and (b['point_labels'] == 1).all()
    assert result.mask_supported_a.all() and result.mask_supported_b.all() and not result.b_reuses_a.any()
    assert not result.masks_a.any() and result.masks_b.any()  # Empty observed mask A retained.
    np.testing.assert_allclose(result.raw_scores_a, np.arange(count)-.4)
    np.testing.assert_allclose(result.raw_scores_b, np.arange(count)+1.2)  # No [0,1] gate.


def test_invalid_point_subbatch_preserves_native_order_and_reuses_a_exactly():
    bank = alter(hands(3), lambda xy: xy.__setitem__((1, 5, 0), 17.))
    result, model = build(bank)
    assert len(model.calls[0]['box']) == 3 and len(model.calls[1]['box']) == 2
    np.testing.assert_array_equal(model.calls[1]['box'], model.calls[0]['box'][[0, 2]])
    assert result.b_reuses_a.tolist() == [False, True, False]
    assert result.points_outside[1, 1] and not result.points_usable[1, 1]
    assert result.original_xy[1, 5, 0] == 17 and result.raw_boxes[1, 2] == 17 and result.clipped_boxes[1, 2] == 16
    np.testing.assert_array_equal(result.masks_a[1], result.masks_b[1])
    assert result.raw_scores_a[1] == result.raw_scores_b[1]


def test_outside_thumb_is_not_in_17_point_branch_but_still_defines_box():
    bank = alter(hands(), lambda xy: xy.__setitem__((0, 2, 0), -3.))
    result, model = build(bank)
    assert not result.b_reuses_a[0] and result.raw_boxes[0, 0] == -3 and result.clipped_boxes[0, 0] == 0
    assert result.point_indices == (0, *range(5, 21)) and model.calls[0]['box'][0, 0] == 0
    assert result.original_xy[0, 2, 0] == -3


def test_all_b_unusable_reuses_a_without_second_call_and_keeps_actual_fp32_prompts():
    bank = alter(hands(2), lambda xy: xy.__setitem__((slice(None), 5, 0), 16.-1e-10))
    result, model = build(bank)
    assert len(model.calls) == 1 and result.b_reuses_a.all()
    assert not result.points_outside.any() and not result.points_usable[:, 1].any()
    assert (result.points17[:, 1, 0] < 16).all() and (result.native_points17[:, 1, 0] == 16).all()
    np.testing.assert_array_equal(result.native_boxes, model.calls[0]['box'])
    np.testing.assert_array_equal(result.masks_b, result.masks_a)
    np.testing.assert_array_equal(result.raw_scores_b, result.raw_scores_a)


def test_native_two_box_a_and_singleton_b_keep_correct_squeeze_shape():
    bank = alter(hands(2), lambda xy: xy.__setitem__((1, 5, 0), 17.))
    result, model = build(bank)
    assert len(model.calls[0]['box']) == 2 and len(model.calls[1]['box']) == 1
    assert result.b_reuses_a.tolist() == [False, True]
    assert result.masks_b[0].any() and not result.masks_b[1].any()


@pytest.mark.parametrize('fault', ['nan', 'unavailable', 'degenerate', 'fully_outside'])
def test_unusable_box_keeps_slot_and_missing_support_without_model_calls(fault):
    bank = hands()
    if fault == 'nan': bank = alter(bank, lambda xy: xy.__setitem__((0, 2, 0), np.nan))
    elif fault == 'unavailable':
        support = np.ones((1, 21), bool); support[0, 2] = False
        bank = alter(bank, lambda _: None, support)
    elif fault == 'degenerate': bank = alter(bank, lambda xy: xy.__setitem__((0, slice(None), 0), 2.))
    else: bank = alter(bank, lambda xy: xy.__setitem__((0, slice(None), 0), 18.))
    result, model = build(bank)
    assert model.images == model.calls == [] and result.local_ids == ('hand:000000',)
    assert not result.box_usable[0] and not result.mask_supported_a[0] and not result.mask_supported_b[0]
    assert result.b_reuses_a[0] and not result.masks_a.any() and not result.masks_b.any()
    assert np.isnan(result.raw_scores_a[0]) and np.isnan(result.raw_scores_b[0])
    if fault == 'nan': assert np.isnan(result.raw_boxes).all() and np.isnan(result.original_xy[0, 2, 0])


def test_empty_native_hands_keep_empty_slots_and_skip_sam2_without_placeholder():
    result, model = build(hands(0))
    assert result.local_ids == result.diagnostics == () and result.masks_a.shape == (0, 12, 16)
    assert result.points17.shape == (0, 17, 2) and result.raw_boxes.shape == (0, 4)
    assert model.images == model.calls == [] and result.raw_scores_a.shape == (0,)


def test_permutation_is_equivariant_no_nms_or_actor_identity_sort():
    bank = hands(3); a, _ = build(bank)
    permutation = [2, 0, 1]
    shuffled = HandInstances(LandmarkEvidence(bank.original_xy.values[permutation], bank.original_xy.supported[permutation], 'original_image', 'pixel'))
    b, _ = build(shuffled)
    np.testing.assert_array_equal(b.raw_boxes, a.raw_boxes[permutation])
    np.testing.assert_array_equal(b.points17, a.points17[permutation])
    assert a.local_ids == b.local_ids  # IDs are slots, never cross-frame physical identity.


def test_result_and_model_rgb_do_not_alias_mutable_inputs_or_native_outputs():
    image = np.zeros((12, 16, 3), np.uint8); masks = np.ones((1, 12, 16), bool); scores = np.array([.7])
    model = Predictor([(masks, scores, None), (masks, scores, None)])
    result, _ = build(model=model, image=image)
    arrays = [v for v in result.__dict__.values() if isinstance(v, np.ndarray)]
    for a in arrays:
        assert a.flags.owndata and not a.flags.writeable
        assert not np.shares_memory(a, masks) and not np.shares_memory(a, scores) and not np.shares_memory(a, image)
        if a.size:
            with pytest.raises(ValueError): a.flat[0] = 0
    assert not model.images[0].flags.writeable and not np.shares_memory(model.images[0], image)
    masks[:] = False; scores[:] = 99; image[:] = 255
    assert result.masks_a.all() and result.raw_scores_a[0] == .7


@pytest.mark.parametrize('bad', [np.zeros((2, 2)), np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 3)),
                               np.ma.array(np.zeros((2, 2, 3), np.uint8))])
def test_invalid_grid_rejected_before_callbacks(bad):
    model = Predictor()
    with pytest.raises(ValueError): build(model=model, image=bad)
    assert model.images == model.calls == []


@pytest.mark.parametrize('index', [-1, True, 1.5, '0'])
def test_original_index_is_explicit_nonnegative_integer(index):
    with pytest.raises(ValueError): build(index=index)


@pytest.mark.parametrize('output', [[], (np.zeros((1, 1, 12, 16)), np.ones(1), None),
                                 (np.zeros((1, 12, 16)), np.ones((1, 1)), None),
                                 (np.full((1, 12, 16), .5), np.ones(1), None),
                                 (np.zeros((1, 12, 16)), np.array([np.nan]), None),
                                 (np.zeros((1, 12, 16)), np.ones(1), np.zeros((1, 128, 128)))])
def test_native_malformed_masks_scores_logits_fail_without_rethresholding(output):
    model = Predictor([output])
    with pytest.raises(ValueError): build(model=model)
    assert len(model.calls) == 1  # No implicit empty, rescue or retry.


def test_predictor_failure_propagates_and_no_mask_fallback_is_fabricated():
    class Broken(Predictor):
        def predict(self, **_): raise RuntimeError('native SAM2 error')
    with pytest.raises(RuntimeError): build(model=Broken())
    with pytest.raises(ValueError): build(model=object())

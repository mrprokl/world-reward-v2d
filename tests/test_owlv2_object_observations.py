"""Manufactured tiny ABI controls, no checkpoint, accuracy, media or GPU."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward import owlv2_object_observations as p


def bank(image_size=(100, 200)):
    boxes = np.tile(np.array([.5, .5, .8, .8], dtype=np.float32), (p.PATCH_COUNT, 1))
    logits = np.arange(p.PATCH_COUNT, dtype=np.float32)-1800
    corners = np.concatenate((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), axis=1)
    return p.Owlv2ObjectObservations(7, image_size, p.GRID, np.arange(p.PATCH_COUNT, dtype=np.int64),
                                   boxes, logits, corners*np.float32(max(image_size)))


def test_full_ordered_raw_bank_immutable_no_objectness_threshold_or_nms():
    b = bank()
    assert len(b.object_ids) == 3600 and b.object_ids[0] == 'owlv2_patch_00_00'
    assert b.object_ids[-1] == 'owlv2_patch_59_59' and len(set(b.object_ids)) == 3600
    assert b.objectness_logits[0] == -1800 and b.objectness_logits[-1] == 1799
    assert np.max(b.boxes_original_xyxy[:, 3]) > b.image_size[0]
    for name in ('patch_ids', 'boxes_padded_normalized_cxcywh', 'objectness_logits', 'boxes_original_xyxy'):
        assert not getattr(b, name).flags.writeable
        with pytest.raises(ValueError): getattr(b, name).flat[0] = 0


def test_snapshot_independent_of_original_mutation_and_preserve_zero_boxes():
    b = bank(); boxes = np.zeros((3600, 4), np.float32); logits = np.full(3600, -200, np.float32)
    zero = p.Owlv2ObjectObservations(0, (1, 1), p.GRID, b.patch_ids, boxes, logits, boxes)
    boxes[:] = 1; logits[:] = 100
    assert np.all(zero.boxes_original_xyxy == 0) and np.all(zero.objectness_logits == -200)


@pytest.mark.parametrize('size', [(100, 200), (200, 100), (100, 100)])
def test_original_square_pad_inverse_not_separate_scale_or_clipping(size):
    b = bank(size)
    expected = np.array([.1, .1, .9, .9], np.float32)*np.float32(max(size))
    assert np.allclose(b.boxes_original_xyxy[0], expected, atol=0.00001)


@pytest.mark.parametrize('failure', ['nanlow', 'infbox', 'dtype', 'empty', 'truncated', 'permuted', 'grid', 'clipped', 'separate_wh', 'framebool', 'range'])
def test_invalid_native_fullbank_rejected_not_filtered_or_repaired(failure):
    b = bank(); values = dict(original_frame_index=7, image_size=b.image_size, patch_grid=b.patch_grid,
        patch_ids=b.patch_ids.copy(), boxes_padded_normalized_cxcywh=b.boxes_padded_normalized_cxcywh.copy(),
        objectness_logits=b.objectness_logits.copy(), boxes_original_xyxy=b.boxes_original_xyxy.copy())
    if failure == 'nanlow': values['objectness_logits'][0] = np.nan
    elif failure == 'infbox': values['boxes_original_xyxy'][0, 0] = np.inf
    elif failure == 'dtype': values['objectness_logits'] = values['objectness_logits'].astype(np.float64)
    elif failure == 'empty': values['objectness_logits'] = np.empty(0, np.float32)
    elif failure == 'truncated': values['patch_ids'] = values['patch_ids'][:-1]
    elif failure == 'permuted': values['patch_ids'] = values['patch_ids'][::-1]
    elif failure == 'grid': values['patch_grid'] = (30, 120)
    elif failure == 'clipped': values['boxes_original_xyxy'][:, 3] = 100
    elif failure == 'separate_wh': values['boxes_original_xyxy'][:, [1, 3]] /= 2
    elif failure == 'framebool': values['original_frame_index'] = True
    else: values['boxes_padded_normalized_cxcywh'][0, 0] = -1
    with pytest.raises(ValueError): p.Owlv2ObjectObservations(**values)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value); self.shape = self.value.shape; self.dtype = self.value.dtype
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def to(self, device): assert device == 'manufactured_cpu'; return self
    def reshape(self, *shape): return Tensor(self.value.reshape(shape))


class Processor:
    do_rescale = do_pad = do_resize = do_normalize = True
    rescale_factor = 1/255
    size = {'height': 960, 'width': 960}
    resample = 2
    image_mean = p.MEAN
    image_std = p.STD
    def __init__(self, calls): self.calls = calls
    def __call__(self, **kwargs):
        self.calls.append(('processor', set(kwargs)))
        assert kwargs['images'].dtype == np.uint8 and not kwargs['images'].flags.writeable
        assert kwargs['return_tensors'] == 'pt' and kwargs['input_data_format'] == 'channels_last'
        return {'pixel_values': Tensor(np.zeros((1, 3, 960, 960), np.float32))}


class Model:
    training = False
    config = SimpleNamespace(vision_config=SimpleNamespace(image_size=960, patch_size=16, hidden_size=768))
    num_patches_height = num_patches_width = 60
    def __init__(self, calls): self.calls = calls; self.bad = None
    def modules(self): return [self]
    def parameters(self): return [Tensor(np.zeros(1, np.float32))]
    def image_embedder(self, **kwargs):
        self.calls.append(('embed', set(kwargs)))
        assert kwargs['interpolate_pos_encoding'] is False
        shape = (1, 60, 60, 768) if self.bad != 'feature_shape' else (1, 1, 1, 768)
        value = np.zeros(shape, np.float32)
        if self.bad == 'feature_nan': value.flat[-1] = np.nan
        return Tensor(value), None
    def objectness_predictor(self, features):
        self.calls.append(('objectness', features.shape))
        value = np.arange(3600, dtype=np.float32)[None, :]-1800
        if self.bad == 'logit_nan': value[0, 0] = np.nan
        return Tensor(value)
    def box_predictor(self, features, feature_map, **kwargs):
        self.calls.append(('box', features.shape, feature_map.shape))
        assert kwargs == {'interpolate_pos_encoding': False}
        return Tensor(bank().boxes_padded_normalized_cxcywh[None, :])
    def class_predictor(self, *_): pytest.fail('No class/text query path permitted')
    def forward(self, *_): pytest.fail('No text-conditioned model forward permitted')


def setup():
    calls = []; model = Model(calls); processor = Processor(calls)
    torch = SimpleNamespace(float32=np.dtype(np.float32), no_grad=nullcontext,
        is_autocast_enabled=lambda *a: False,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False)))
    def corners(boxes):
        calls.append(('corners', boxes.shape)); b = boxes.value
        return Tensor(np.concatenate((b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2), axis=-1))
    def scale(boxes, sizes):
        calls.append(('scale', sizes)); assert sizes == [(100, 200)]
        return Tensor(boxes.value*np.float32(max(sizes[0])))
    operations = p.NativeOwlv2Operations(processor, torch, 'manufactured_cpu', corners, scale)
    return model, operations, calls


def test_exact_imageonly_native_calls_and_original_inverse_fullbank():
    model, operations, calls = setup(); rgb = np.zeros((100, 200, 3), np.uint8)
    result = p.infer_owlv2_object_frame(model, rgb, 9, operations)
    assert result.original_frame_index == 9 and result.image_size == (100, 200)
    assert [r[0] for r in calls] == ['processor', 'embed', 'objectness', 'box', 'corners', 'scale']
    assert calls[2][1] == (1, 3600, 768) and calls[3][2] == (1, 60, 60, 768)
    assert np.array_equal(result.objectness_logits, np.arange(3600, dtype=np.float32)-1800)
    assert not rgb.any()


@pytest.mark.parametrize('bad', ['feature_shape', 'feature_nan', 'logit_nan'])
def test_native_nonfinite_or_wrong_grid_fails_no_fallback(bad):
    model, operations, calls = setup(); model.bad = bad
    with pytest.raises(ValueError): p.infer_owlv2_object_frame(model, np.zeros((100, 200, 3), np.uint8), 0, operations)
    if bad.startswith('feature'): assert 'objectness' not in [r[0] for r in calls]
    assert 'scale' not in [r[0] for r in calls]


@pytest.mark.parametrize('bad', ['processor_pad', 'processor_size', 'processor_mean', 'training', 'model_grid', 'tf32', 'autocast'])
def test_caller_runtime_or_processor_policy_must_be_original(bad):
    model, operations, calls = setup()
    if bad == 'processor_pad': operations.image_processor.do_pad = False
    elif bad == 'processor_size': operations.image_processor.size = {'height': 768, 'width': 768}
    elif bad == 'processor_mean': operations.image_processor.image_mean = (0, 0, 0)
    elif bad == 'training': model.training = True
    elif bad == 'model_grid': model.num_patches_height = 30
    elif bad == 'tf32': operations.tensor_ops.backends.cuda.matmul.allow_tf32 = True
    else: operations.tensor_ops.is_autocast_enabled = lambda *a: True
    with pytest.raises(ValueError): p.infer_owlv2_object_frame(model, np.zeros((100, 200, 3), np.uint8), 0, operations)
    assert calls == []


def test_invalid_rgb_not_rescaled_gray_or_empty_and_not_bool_frame():
    model, operations, calls = setup()
    for rgb in (np.zeros((100, 200, 3), np.float32), np.zeros((100, 200), np.uint8), np.empty((0, 200, 3), np.uint8)):
        with pytest.raises(ValueError): p.infer_owlv2_object_frame(model, rgb, 0, operations)
    with pytest.raises(ValueError): p.infer_owlv2_object_frame(model, np.zeros((100, 200, 3), np.uint8), True, operations)
    assert calls == []

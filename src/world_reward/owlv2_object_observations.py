"""Complete image-only native OWLv2 patch evidence, NOT object/owner truth.

Transformers 4.53.3 modeling_owlv2.py 78926B SHA256
98e94770ce96e7b6b09be14596fad0600aac7a60e47b5fda846464547ae4d05e:
image_embedder -> objectness_predictor / box_predictor, no text/class head.
image_processing_owlv2.py 28040B SHA256
b20c2be9d906b104e65be6e8851d9a597ea228b474830775dbe7ddacf1a2253f:
rescale -> bottom/right square pad -> original SciPy resize -> CLIP normalize.
The original _scale_boxes maps normalized corners by max(original H,W), not
separate W/H factors. Out-of-image/zero-area proposals are retained, not fixed.

Caller authenticates original source, checkpoint, processor and operations;
this module neither loads them nor proves licensing/pretraining independence.
No NMS, ranking, topK, threshold, object categories, hand proximity or fallback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

SCHEMA = 'world_reward.owlv2_object_observations.v1'
IMAGE_SIZE, PATCH_SIZE, HIDDEN_SIZE = 960, 16, 768
GRID = (60, 60)
PATCH_COUNT = 3600
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _array(value, shape, name):
    _require(type(value) is np.ndarray and value.dtype == np.float32
             and value.shape == shape and np.isfinite(value).all(), name+': finite native FP32 full bank required')
    result = np.array(value, copy=True, order='C')
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class Owlv2ObjectObservations:
    original_frame_index: int
    image_size: tuple[int, int]  # original (height,width), before square padding
    patch_grid: tuple[int, int]
    patch_ids: np.ndarray
    boxes_padded_normalized_cxcywh: np.ndarray
    objectness_logits: np.ndarray  # native uncalibrated output, not sigmoid/class scores
    boxes_original_xyxy: np.ndarray

    def __post_init__(self):
        _require(type(self.original_frame_index) is int and self.original_frame_index >= 0,
                 'Nonnegative original frame index required')
        _require(type(self.image_size) is tuple and len(self.image_size) == 2
                 and all(type(x) is int and x > 0 for x in self.image_size), 'Original (height,width) required')
        _require(type(self.patch_grid) is tuple and self.patch_grid == GRID, 'Complete original 60x60 patch grid required')
        ids = self.patch_ids
        _require(type(ids) is np.ndarray and ids.dtype == np.int64 and ids.shape == (PATCH_COUNT,)
                 and np.array_equal(ids, np.arange(PATCH_COUNT, dtype=np.int64)), 'Complete ordered native row-major patch IDs required')
        ids = np.array(ids, copy=True); ids.flags.writeable = False
        object.__setattr__(self, 'patch_ids', ids)
        for name, shape in (('boxes_padded_normalized_cxcywh', (PATCH_COUNT, 4)),
                            ('objectness_logits', (PATCH_COUNT,)), ('boxes_original_xyxy', (PATCH_COUNT, 4))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        b = self.boxes_padded_normalized_cxcywh
        _require(np.all((b >= 0) & (b <= 1)), 'Native box sigmoid range required; no coordinate repairs')
        corners = np.concatenate((b[:, :2]-b[:, 2:]/np.float32(2), b[:, :2]+b[:, 2:]/np.float32(2)), axis=1)
        expected = corners*np.float32(max(self.image_size))
        _require(np.array_equal(self.boxes_original_xyxy, expected), 'Original square-pad inverse required, without clipping/alignment')

    @property
    def object_ids(self):
        return tuple(f'owlv2_patch_{i//GRID[1]:02d}_{i%GRID[1]:02d}' for i in range(PATCH_COUNT))


@dataclass(frozen=True, eq=False)
class NativeOwlv2Operations:
    """Real already-loaded operations; provenance qualification is caller-owned.

    image_processor is the original Owlv2ImageProcessor, not the combined text
    processor or a custom resize. center_to_corners and scale_boxes are original
    Transformers functions; no reimplementation is used to produce predictions.
    Caller disables AMP/TF32 and moves original model to FP32/eval beforehand.
    """
    image_processor: Any
    tensor_ops: Any
    device: Any
    center_to_corners: Callable
    scale_boxes: Callable

    def __post_init__(self):
        _require(callable(self.image_processor) and callable(self.center_to_corners)
                 and callable(self.scale_boxes) and callable(getattr(self.tensor_ops, 'no_grad', None)),
                 'Explicit real native operations required')


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def infer_owlv2_object_frame(model, rgb, original_frame_index, operations: NativeOwlv2Operations):
    """One genuine native forward of all patches. Exceptions propagate unchanged.

    No source/checkpoint download, text query, class head or postprocessing path.
    Empty/truncated/nonfinite evidence fails rather than manufacturing a bank.
    """
    _require(type(operations) is NativeOwlv2Operations, 'Explicit native operations required')
    _require(type(original_frame_index) is int and original_frame_index >= 0, 'Original frame index required')
    _require(type(rgb) is np.ndarray and rgb.dtype == np.uint8 and rgb.ndim == 3
             and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, 'Original uint8 RGB required')
    image_size = tuple(int(x) for x in rgb.shape[:2]); digest = hashlib.sha256(rgb.tobytes()).digest()
    image = np.array(rgb, copy=True, order='C'); image.flags.writeable = False
    o = operations; t = o.tensor_ops; processor = o.image_processor
    _require(model.training is False and all(m.training is False for m in model.modules()), 'Original complete model must be eval')
    parameters = list(model.parameters())
    _require(parameters and all(p.dtype == t.float32 for p in parameters), 'Original model FP32 required, no adapter conversion')
    v = model.config.vision_config
    _require((v.image_size, v.patch_size, v.hidden_size) == (IMAGE_SIZE, PATCH_SIZE, HIDDEN_SIZE)
             and (model.num_patches_height, model.num_patches_width) == GRID, 'Original base-patch16 ensemble dimensions required')
    _require(not t.is_autocast_enabled() and not t.is_autocast_enabled('cpu')
             and t.backends.cuda.matmul.allow_tf32 is False and t.backends.cudnn.allow_tf32 is False,
             'Caller must disable AMP and TF32 before native inference')
    _require(processor.do_rescale is True and processor.rescale_factor == 1/255
             and processor.do_pad is True and processor.do_resize is True and processor.do_normalize is True
             and processor.size == {'height': IMAGE_SIZE, 'width': IMAGE_SIZE} and processor.resample == 2
             and tuple(processor.image_mean) == MEAN and tuple(processor.image_std) == STD,
             'Original pinned image processor defaults required')
    data = processor(images=image, return_tensors='pt', input_data_format='channels_last')
    _require(isinstance(data, Mapping) and set(data) == {'pixel_values'}, 'Image-only native input required')
    pixels = data['pixel_values']
    _require(tuple(pixels.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE) and pixels.dtype == t.float32
             and np.isfinite(_numpy(pixels)).all(), 'Finite original 960-square FP32 pixels required')
    pixels = pixels.to(o.device)  # device only; no dtype conversion or interpolation
    with t.no_grad():
        feature_map, _ = model.image_embedder(pixel_values=pixels, interpolate_pos_encoding=False)
        _require(tuple(feature_map.shape) == (1, *GRID, HIDDEN_SIZE) and feature_map.dtype == t.float32
                 and np.isfinite(_numpy(feature_map)).all(), 'Finite complete native image feature map required')
        features = feature_map.reshape(1, PATCH_COUNT, HIDDEN_SIZE)
        logits = model.objectness_predictor(features)
        boxes = model.box_predictor(features, feature_map, interpolate_pos_encoding=False)
        raw_logits = _array(_numpy(logits), (1, PATCH_COUNT), 'native objectness logits')[0]
        raw_boxes = _array(_numpy(boxes), (1, PATCH_COUNT, 4), 'native padded boxes')[0]
        corners = o.center_to_corners(boxes)
        original = o.scale_boxes(corners, [image_size])
        original_boxes = _array(_numpy(original), (1, PATCH_COUNT, 4), 'native original boxes')[0]
    _require(hashlib.sha256(rgb.tobytes()).digest() == digest, 'Original RGB mutated during inference')
    return Owlv2ObjectObservations(original_frame_index, image_size, GRID,
        np.arange(PATCH_COUNT, dtype=np.int64), raw_boxes, raw_logits, original_boxes)

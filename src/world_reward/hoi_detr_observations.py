"""One-image HOI-DETR evidence, not physical contact or persistent/body identity.

Native source: AhmadDarKhalil/HOI-DETR 1b367292f3833afd64a204bd4d9d84519541d035:
demo/helpers.py (20441B, fa91535e09a2a55cee270ef239403a5a41a512e0d0d049135edb407fe4779420),
projects/models/co_dino_head_w_interaction.py (52313B,
6c56c03557c9bfd1807edfdec725cb2dfb6c806127c8316aec3d553b3a0c8c71), and
projects/models/roi_heads/interaction_head.py (2006B,
8c703b493c2cd7d9a78f8de721cb47184fe85cadd27feb7374c41a44ba6e5474).

The caller authenticates/loads the original eval model and MMCV operations.
prepare_rgb must implement its qualified in-memory original test preprocessing;
this module neither guesses RGB/BGR conversion nor imports mmdet.apis/datasets.
Native flatten-top1000, CPU soft-NMS(.5,min_score=.3), and RAW-score >=.3
retention match the helper. Decayed scores are retained, not substituted for its
raw-score rule. Every surviving ordered H->F/F->S pair uses the ORIGINAL
InteractionHead.forward(..., interaction_targets=None), never demo softmax,
link thresholds, nearest-object selection, target prompts or temporal IDs.
Source/weight/runtime license and overlap qualification remain prerequisites.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
import hashlib
from typing import Any

import numpy as np

SCHEMA = "world_reward.hoi_detr_observations.v1"
CLASS_NAMES = ("hand", "direct_object", "tool_target")
NUM_QUERIES, NUM_CLASSES, TOKEN_DIM, TOP_K = 1500, 3, 256, 1000
SCORE_THRESHOLD, NMS_IOU = .3, .5


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _array(value, name, shape, dtype):
    _require(not np.ma.isMaskedArray(value), name + ": masked evidence forbidden")
    a = np.asarray(value)
    _require(a.shape == shape and a.dtype == np.dtype(dtype), name + ": native shape/dtype required")
    if a.dtype.kind == "f":
        _require(np.isfinite(a).all(), name + ": nonfinite native output")
    result = np.array(a, copy=True, order="C", subok=False)
    result.flags.writeable = False
    return result


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


@dataclass(frozen=True, eq=False)
class NativeHOIOperations:
    """Supplied real operations, never an adapter accepting invented predictions.

    prepare_rgb returns the original MMCV pre-collate img/img_metas structure.
    collate/scatter/bbox_cxcywh_to_xyxy/batched_nms are authenticated originals;
    tensor_ops is the already-loaded Torch namespace (no imports here). Caller
    must disable CUDA/CPU autocast and CUDA-matmul/cuDNN TF32 BEFORE this call;
    the adapter verifies but never changes that runtime policy.
    """
    prepare_rgb: Callable
    collate: Callable
    scatter: Callable
    bbox_cxcywh_to_xyxy: Callable
    batched_nms: Callable
    tensor_ops: Any
    device: Any

    def __post_init__(self):
        for name in ("prepare_rgb", "collate", "scatter", "bbox_cxcywh_to_xyxy", "batched_nms"):
            _require(callable(getattr(self, name)), name + ": explicit native callable required")
        _require(callable(getattr(self.tensor_ops, "no_grad", None))
                 and callable(getattr(self.tensor_ops, "cat", None))
                 and callable(getattr(self.tensor_ops, "as_tensor", None)), "Loaded native tensor operations required")


@dataclass(frozen=True, eq=False)
class HOIDetrObservations:
    original_frame_index: int
    image_size: tuple[int, int]
    query_logits: np.ndarray       # [1500,3], before sigmoid
    query_boxes_cxcywh: np.ndarray # [1500,4], native normalized coordinates
    query_tokens: np.ndarray       # [1500,256], all last-layer tokens
    native_nms_detections: np.ndarray  # [M,5], resized-image xyxy + CPU decayed score
    native_nms_keep: np.ndarray     # [M], positions in native flattened top1000
    retained_nms_positions: np.ndarray # [N], native RAW-score >=.3 rule
    query_ids: np.ndarray           # [N], duplicates across classes are retained
    class_ids: np.ndarray           # [N], native 0/1/2 role classes, not object category
    boxes_original_xyxy: np.ndarray # [N,4], no clipping/half-pixel/mirror/centering
    raw_scores: np.ndarray          # [N], sigmoid scores before soft-NMS decay
    decayed_scores: np.ndarray      # [N], exact corresponding native CPU scores
    hand_object_pairs: np.ndarray  # [P,2], indices into the N retained detections
    hand_object_logits: np.ndarray # [P,2], ORIGINAL uncalibrated logits
    object_target_pairs: np.ndarray
    object_target_logits: np.ndarray

    def __post_init__(self):
        _require(type(self.original_frame_index) is int and self.original_frame_index >= 0,
                 "Actual nonnegative original frame index required")
        _require(type(self.image_size) is tuple and len(self.image_size) == 2
                 and all(type(n) is int and n > 0 for n in self.image_size), "Original (height,width) required")
        n, m = len(self.query_ids), len(self.native_nms_keep)
        specs = {"query_logits": ((NUM_QUERIES, 3), np.float32),
                 "query_boxes_cxcywh": ((NUM_QUERIES, 4), np.float32),
                 "query_tokens": ((NUM_QUERIES, TOKEN_DIM), np.float32),
                 "native_nms_detections": ((m, 5), np.float32), "native_nms_keep": ((m,), np.int64),
                 "retained_nms_positions": ((n,), np.int64), "query_ids": ((n,), np.int64),
                 "class_ids": ((n,), np.int64), "boxes_original_xyxy": ((n, 4), np.float32),
                 "raw_scores": ((n,), np.float32), "decayed_scores": ((n,), np.float32)}
        for name, (shape, dtype) in specs.items():
            object.__setattr__(self, name, _array(getattr(self, name), name, shape, dtype))
        _require(m <= TOP_K and np.all((self.native_nms_keep >= 0) & (self.native_nms_keep < TOP_K))
                 and len(np.unique(self.native_nms_keep)) == m, "Native keep must index original top1000 uniquely")
        pos = self.retained_nms_positions
        _require(np.all((pos >= 0) & (pos < m)) and np.all(np.diff(pos) > 0)
                 and np.all((self.query_ids >= 0) & (self.query_ids < NUM_QUERIES))
                 and np.all((self.class_ids >= 0) & (self.class_ids < NUM_CLASSES)), "Original proposal slots/classes required")
        _require(np.all((self.raw_scores >= SCORE_THRESHOLD) & (self.raw_scores <= 1))
                 and np.array_equal(self.decayed_scores, self.native_nms_detections[pos, 4]), "Raw/native-decayed score distinction required")
        for pair_name, logit_name, roles in (("hand_object_pairs", "hand_object_logits", (0, 1)),
                                             ("object_target_pairs", "object_target_logits", (1, 2))):
            left, right = (np.flatnonzero(self.class_ids == role) for role in roles)
            expected = np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64)
            pairs = _array(getattr(self, pair_name), pair_name, expected.shape, np.int64)
            _require(np.array_equal(pairs, expected), "Every ordered native pair required; no link selection")
            object.__setattr__(self, pair_name, pairs)
            object.__setattr__(self, logit_name, _array(getattr(self, logit_name), logit_name, (len(pairs), 2), np.float32))


def _pairs(classes, roles):
    left, right = (np.flatnonzero(classes == role) for role in roles)
    return np.column_stack((np.repeat(left, len(right)), np.tile(right, len(left)))).astype(np.int64)


def infer_hoi_detr_frame(model, rgb, original_frame_index, operations: NativeHOIOperations):
    """Execute one actual native image forward; exceptions propagate, no fallback.

    Genuine empty native detections have empty pair arrays. No dummy pair/head
    output is manufactured. This function does not load a model or claim source
    authentication; the caller binds the provided model/operations to its receipt.
    """
    _require(type(operations) is NativeHOIOperations, "Explicit native operations required")
    _require(type(original_frame_index) is int and original_frame_index >= 0, "Original frame index required")
    _require(type(rgb) is np.ndarray and rgb.dtype == np.uint8 and rgb.ndim == 3
             and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, "Unconverted original RGB uint8 required")
    image_size = tuple(int(n) for n in rgb.shape[:2])
    original_digest = hashlib.sha256(rgb.tobytes()).digest()
    image = np.array(rgb, copy=True, order="C"); image.flags.writeable = False
    head = model.query_head
    _require(model.training is False and head.training is False and head.interaction_head.training is False,
             "Original model/query/interaction heads must already be eval")
    _require(head.num_query == NUM_QUERIES and head.num_classes == NUM_CLASSES and head.embed_dims == TOKEN_DIM
             and head.test_cfg.get("max_per_img") == TOP_K, "Frozen native query/class/token/top1000 policy required")
    o = operations
    _require(not o.tensor_ops.is_autocast_enabled() and not o.tensor_ops.is_autocast_enabled("cpu")
             and o.tensor_ops.backends.cuda.matmul.allow_tf32 is False
             and o.tensor_ops.backends.cudnn.allow_tf32 is False, "Caller must disable AMP and TF32 before native inference")
    data = o.prepare_rgb(image)
    _require(isinstance(data, Mapping) and set(data) == {"img", "img_metas"}, "Image-only native preprocessing required")
    data = o.scatter(o.collate([data], samples_per_gpu=1), [o.device])[0]
    img = data["img"]; img = img[0] if isinstance(img, list) else img
    metas = data["img_metas"]
    if isinstance(metas, list) and metas and isinstance(metas[0], list):
        metas = metas[0]
    _require(isinstance(metas, list) and len(metas) == 1 and isinstance(metas[0], Mapping)
             and tuple(metas[0]["ori_shape"]) == rgb.shape and len(img.shape) == 4
             and tuple(img.shape[:2]) == (1, 3) and min(img.shape[-2:]) > 0, "One original image/meta frame required")
    processed_shape = metas[0]["img_shape"]
    _require(len(processed_shape) == 3 and processed_shape[2] == 3
             and 0 < processed_shape[0] <= img.shape[-2] and 0 < processed_shape[1] <= img.shape[-1],
             "Native processed image shape must fit its padded tensor")
    metas[0]["batch_input_shape"] = tuple(img.shape[-2:])
    with o.tensor_ops.no_grad():
        features = model.extract_feat(img)
        outs, hs = head(features, metas, return_hs=True)
        _require(tuple(outs[0].shape) == (6, 1, NUM_QUERIES, NUM_CLASSES)
                 and tuple(outs[1].shape) == (6, 1, NUM_QUERIES, 4)
                 and tuple(hs.shape) == (6, 1, NUM_QUERIES, TOKEN_DIM), "Original six-layer decoder outputs required")
        logits, coordinates, tokens = outs[0][-1][0], outs[1][-1][0], hs[-1][0]
        # Fail BEFORE topk/NMS, including nonfinite low-score/unselected queries.
        native_logits = _array(_numpy(logits), "all native class logits", (NUM_QUERIES, 3), np.float32)
        native_coordinates = _array(_numpy(coordinates), "all native boxes", (NUM_QUERIES, 4), np.float32)
        native_tokens = _array(_numpy(tokens), "all native tokens", (NUM_QUERIES, TOKEN_DIM), np.float32)
        scores, flat_ids = logits.sigmoid().view(-1).topk(TOP_K)
        labels, query_ids = flat_ids % NUM_CLASSES, flat_ids // NUM_CLASSES
        h, w = metas[0]["img_shape"][:2]
        boxes = o.bbox_cxcywh_to_xyxy(coordinates[query_ids]) * coordinates.new_tensor([w, h, w, h])
        dets, keep = o.batched_nms(boxes, scores, labels,
                                  dict(type="soft_nms", iou_threshold=NMS_IOU, min_score=SCORE_THRESHOLD))
        keep_np = _numpy(keep)
        _require(keep_np.ndim == 1 and keep_np.dtype == np.int64, "Native keep indices required")
        keep_np = _array(keep_np, "native_keep", keep_np.shape, np.int64)
        _require(np.all((keep_np >= 0) & (keep_np < TOP_K)) and len(np.unique(keep_np)) == len(keep_np), "Native keep range/duplicates invalid")
        det_np = _array(_numpy(dets), "CPU soft-NMS", (len(keep_np), 5), np.float32)
        raw = _array(_numpy(scores[keep]), "raw scores", (len(keep_np),), np.float32)
        pos = np.flatnonzero(raw >= SCORE_THRESHOLD).astype(np.int64)
        qids = _numpy(query_ids[keep])[pos]; classes = _numpy(labels[keep])[pos]
        sf = np.asarray(metas[0].get("scale_factor", 1.0), dtype=np.float32).ravel()
        if sf.size == 1:
            sf = np.tile(sf, 4)
        elif sf.size == 2:
            sf = np.array([sf[0], sf[1], sf[0], sf[1]], dtype=np.float32)
        _require(sf.shape == (4,) and np.isfinite(sf).all() and np.all(sf > 0), "Original native positive scale factor required")
        original_boxes = _numpy(boxes[keep])[pos] / sf
        _array(original_boxes, "original image boxes", (len(pos), 4), np.float32)
        pair_arrays = [_pairs(classes, roles) for roles in ((0, 1), (1, 2))]
        pair_logits = []
        for pairs in pair_arrays:
            if not len(pairs):
                pair_logits.append(np.empty((0, 2), dtype=np.float32)); continue
            ids = o.tensor_ops.as_tensor(qids[pairs], dtype=o.tensor_ops.long, device=tokens.device)
            pair_tokens = o.tensor_ops.cat((tokens[ids[:, 0]], tokens[ids[:, 1]]), dim=-1)
            result, loss = head.interaction_head.forward(pair_tokens, interaction_targets=None)
            _require(loss is None, "Native inference must not compute any supervised interaction loss")
            pair_logits.append(_array(_numpy(result), "raw ordered interaction logits", (len(pairs), 2), np.float32))
        _require(np.array_equal(_numpy(logits), native_logits) and np.array_equal(_numpy(coordinates), native_coordinates)
                 and np.array_equal(_numpy(tokens), native_tokens), "Native postprocessing changed decoder evidence")
    _require(hashlib.sha256(image.tobytes()).digest() == original_digest
             and hashlib.sha256(rgb.tobytes()).digest() == original_digest, "Native preprocessing mutated original RGB evidence")
    return HOIDetrObservations(original_frame_index, image_size, native_logits, native_coordinates, native_tokens,
                               det_np, keep_np, pos, qids, classes, original_boxes, raw[pos], det_np[pos, 4],
                               pair_arrays[0], pair_logits[0], pair_arrays[1], pair_logits[1])

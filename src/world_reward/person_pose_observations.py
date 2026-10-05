"""All automatic person crops through the supplied ORIGINAL DWPose callable.

IDEA-Research/DWPose 3dca5db79d9f9ffdd378753ddf6ec66535aace88:
ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py (11608B,
16fb69ab54f5e1ce8a5ad186e92f357da0162fc2ca2eeec4ccf1db72949291a2),
inference_pose lines 353-360. It preserves crop order and performs N sequential
native batch-one calls, NOT one batch-N ORT call. Its empty-box full-image
fallback is deliberately never invoked here.
mmpose/configs/_base_/datasets/coco_wholebody.py (30735B,
8c73971296517eadef6b7df6de92eb8943119cb467b3f4d488a6e062470cf179)
defines body wrists 9/10 and distinct hand roots 91/112. No neck insertion,
134-point remapping, RGB conversion, resizing, score threshold or pose fitting.

The caller authenticates native source/session/weights, automatic detections,
licenses and training overlap. A crop-attached prediction is NOT verified
anatomical ownership, particularly with overlapping people. This adapter does
not select a target, associate people over time, or claim inference accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

SCHEMA = "world_reward.person_pose_observations.v1"
NUM_KEYPOINTS = 133
BODY_WRIST_INDICES = (9, 10)       # left, right, in native COCO-WholeBody order
HAND_ROOT_INDICES = (91, 112)      # left, right; NOT body wrist aliases
SIDE_NAMES = ("left", "right")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name, *, floating=False):
    _require(type(value) is np.ndarray and value.shape == shape
             and value.dtype.kind in ("f" if floating else "fiu")
             and np.isfinite(value).all(), name + ": finite plain native array required")
    result = np.array(value, copy=True, order="C", subok=False)
    result.flags.writeable = False
    return result


def _ids(value):
    _require(type(value) is tuple and all(type(item) is str and item and item.strip() == item for item in value)
             and len(set(value)) == len(value), "Unique original person IDs in supplied order required")
    return value


def _boxes(value, count, height, width):
    boxes = _array(value, (count, 4), "Original xyxy person boxes")
    _require(np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 1] >= 0)
             and np.all(boxes[:, 2] <= width) and np.all(boxes[:, 3] <= height)
             and np.all(boxes[:, 2] > boxes[:, 0]) and np.all(boxes[:, 3] > boxes[:, 1]),
             "All original-grid boxes must be positive-area and inside image; no clipping")
    return boxes


def _fingerprint(array):
    return array.shape, array.dtype.str, hashlib.sha256(array.tobytes(order="C")).digest()


@dataclass(frozen=True, eq=False)
class PersonPoseObservations:
    """Every original detection row, including missed/off-grid joint predictions.

    native_valid is EXACTLY raw_scores > 0, the upstream SimCC decoder rule.
    in_original_image is a separate geometric diagnostic, never a replacement
    for native validity. Invalid native coordinates may have already undergone
    inverse affine mapping: do not infer validity from a -1 sentinel or position.
    IDs are detector proposal identities, not established person/hand identities.
    """
    original_frame_index: int
    image_size: tuple[int, int]  # original height, width
    person_ids: tuple[str, ...]
    boxes_original_xyxy: np.ndarray
    detector_scores: np.ndarray
    keypoints_original_xy: np.ndarray  # [N,133,2], no pixel offset or coordinate edits
    raw_scores: np.ndarray            # [N,133], unchanged native dtype/values
    native_valid: np.ndarray = field(init=False)
    in_original_image: np.ndarray = field(init=False)
    anatomical_ownership_verified: bool = field(default=False, init=False)

    def __post_init__(self):
        _require(type(self.original_frame_index) is int and self.original_frame_index >= 0,
                 "Nonnegative original frame index required")
        _require(type(self.image_size) is tuple and len(self.image_size) == 2
                 and all(type(x) is int and x > 0 for x in self.image_size), "Original image size required")
        ids = _ids(self.person_ids)
        height, width = self.image_size
        count = len(ids)
        arrays = {"boxes_original_xyxy": _boxes(self.boxes_original_xyxy, count, height, width),
                  "detector_scores": _array(self.detector_scores, (count,), "Original detector scores"),
                  "keypoints_original_xy": _array(self.keypoints_original_xy, (count, NUM_KEYPOINTS, 2),
                                                   "Native keypoints", floating=True),
                  "raw_scores": _array(self.raw_scores, (count, NUM_KEYPOINTS), "Native scores", floating=True)}
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        points = arrays["keypoints_original_xy"]
        valid = arrays["raw_scores"] > 0
        inside = ((points[..., 0] >= 0) & (points[..., 0] < width)
                  & (points[..., 1] >= 0) & (points[..., 1] < height))
        valid.flags.writeable = inside.flags.writeable = False
        object.__setattr__(self, "native_valid", valid)
        object.__setattr__(self, "in_original_image", inside)


def infer_person_pose_frame(native_inference_pose, session, rgb, original_frame_index,
                            person_ids, boxes_original_xyxy, detector_scores):
    """Call native.inference_pose(session, ALL original boxes, RGB) exactly once.

    Inputs must be the complete caller's automatic person census in its original
    order; completeness/provenance cannot be proved by this transport adapter.
    Duplicate boxes with distinct IDs stay separate. No sorting, NMS, top-one,
    cap, score filter, missing-joint fill or target selection occurs here.
    N=0 returns genuine empty shapes (native decoded float64 points/float32
    scores) without invoking upstream's artificial full-image person fallback.
    Native exceptions propagate; inputs/outputs are copied without changing
    numerical values. Mutating supplied/copy RGB or boxes fails closed.
    """
    _require(callable(native_inference_pose), "Supplied original native inference_pose callable required")
    _require(type(original_frame_index) is int and original_frame_index >= 0, "Original frame index required")
    _require(type(rgb) is np.ndarray and rgb.dtype == np.uint8 and rgb.ndim == 3
             and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, "Original plain uint8 RGB required")
    ids = _ids(person_ids)
    height, width = map(int, rgb.shape[:2])
    boxes = _boxes(boxes_original_xyxy, len(ids), height, width)
    scores = _array(detector_scores, (len(ids),), "Original detector scores")
    image = np.array(rgb, copy=True, order="C"); image.flags.writeable = False
    observed = (rgb, boxes_original_xyxy, detector_scores, image, boxes)
    before = tuple(_fingerprint(a) for a in observed)
    if ids:
        outputs = native_inference_pose(session, boxes, image)
        _require(type(outputs) is tuple and len(outputs) == 2, "Original native (keypoints, scores) tuple required")
        points, pose_scores = outputs
    else:
        points = np.empty((0, NUM_KEYPOINTS, 2), dtype=np.float64)
        pose_scores = np.empty((0, NUM_KEYPOINTS), dtype=np.float32)
    _require(before == tuple(_fingerprint(a) for a in observed), "Native inference mutated original/copy evidence")
    return PersonPoseObservations(original_frame_index, (height, width), ids, boxes, scores, points, pose_scores)

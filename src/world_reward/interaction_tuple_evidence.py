"""Complete person/side/native H->O Cartesian evidence, NEVER a selector.

DWPose body wrists (9,10) and distinct hand roots (91,112) retain their native
scores/coordinates. Every HOI-DETR ordered pair and competing query stays a
separate row, in person-major, left/right, native-pair order. No threshold,
calibration, probability, fitting, target prompt or ownership claim is added.

Feature support means numerical availability ONLY: missed native joints yield
NaN distance features, while finite off-grid coordinates remain diagnostics
with separate coordinate flags. Box overlap is not person/hand ownership or
physical contact. Source references fingerprint observation bytes, NOT source
authentication, licensing, bank completeness or independent quality evidence;
these remain the caller's responsibility. No person/object is selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from world_reward.person_pose_observations import (
    BODY_WRIST_INDICES, HAND_ROOT_INDICES, PersonPoseObservations,
    SCHEMA as PERSON_SCHEMA,
)
from world_reward.hoi_detr_observations import HOIDetrObservations, SCHEMA as HOI_SCHEMA

SCHEMA = "world_reward.interaction_tuple_evidence.v1"
SIDE_NAMES = ("left", "right")
POSE_POINT_NAMES = ("body_wrist", "hand_root")
FEATURE_NAMES = (
    "body_wrist_to_hand_box_over_image_diagonal",
    "body_wrist_to_hand_center_over_image_diagonal",
    "hand_root_to_hand_box_over_image_diagonal",
    "hand_root_to_hand_center_over_image_diagonal",
    "body_wrist_hand_root_distance_over_image_diagonal",
    "hand_person_intersection_over_hand_box_area",
    "hand_person_box_iou",
    "hoi_raw_logit_1_minus_0",
    "body_wrist_raw_score", "hand_root_raw_score", "person_detector_raw_score",
    "hoi_hand_raw_score", "hoi_direct_object_raw_score",
    "hoi_hand_decayed_score", "hoi_direct_object_decayed_score",
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sealed(array):
    a = np.asarray(array)
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _reference(observation, schema):
    digest = hashlib.sha256(schema.encode())
    for f in fields(observation):
        value = getattr(observation, f.name)
        digest.update(f.name.encode() + b"\0")
        if type(value) is np.ndarray:
            digest.update(json.dumps([value.dtype.str, value.shape]).encode())
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(json.dumps(value, allow_nan=False).encode())
    return schema, digest.hexdigest()


@dataclass(frozen=True, eq=False)
class InteractionTupleEvidence:
    original_frame_index: int
    image_size: tuple[int, int]
    source_person_ids: tuple[str, ...]
    source_observation_references: tuple[tuple[str, str], ...]
    arrays: Mapping[str, np.ndarray]
    features: np.ndarray
    feature_supported: np.ndarray
    feature_names: tuple[str, ...] = field(default=FEATURE_NAMES, init=False)
    anatomical_ownership_verified: bool = field(default=False, init=False)
    selection_performed: bool = field(default=False, init=False)


def build_interaction_tuple_evidence(person: PersonPoseObservations, hoi: HOIDetrObservations):
    """Build ALL Nperson*2*P rows; arrays' slots refer to unchanged input banks.

    pose_original_xy/raw_scores/native_valid/in_original_image use point order
    (body_wrist,hand_root). hoi boxes/scores/query slots use (hand,direct_object).
    source_person_ids[person_slots] supplies each proposal ID; native_pair_slots
    index hoi.hand_object_pairs. Native NMS positions and flattened keep slots
    are retained separately, even for duplicate boxes/query IDs. Empty banks
    produce zero rows, not null-object predictions or a full-image fallback.
    """
    _require(type(person) is PersonPoseObservations and type(hoi) is HOIDetrObservations,
             "Original person-pose and HOI observation types required")
    _require(person.original_frame_index == hoi.original_frame_index
             and person.image_size == hoi.image_size, "Same original frame/image grid required")
    refs = (_reference(person, PERSON_SCHEMA), _reference(hoi, HOI_SCHEMA))
    height, width = person.image_size
    diagonal = math.hypot(height, width)
    _require(math.isfinite(diagonal) and diagonal > 0, "Finite original image diagonal required")
    n, p = len(person.person_ids), len(hoi.hand_object_pairs)
    person_slots = np.repeat(np.arange(n, dtype=np.int64), 2*p)
    sides = np.tile(np.repeat(np.arange(2, dtype=np.int64), p), n)
    pair_slots = np.tile(np.arange(p, dtype=np.int64), 2*n)
    detections = hoi.hand_object_pairs[pair_slots]
    indices = np.column_stack((np.asarray(BODY_WRIST_INDICES)[sides], np.asarray(HAND_ROOT_INDICES)[sides]))
    points = person.keypoints_original_xy[person_slots[:, None], indices]
    scores = person.raw_scores[person_slots[:, None], indices]
    valid = person.native_valid[person_slots[:, None], indices]
    boxes = hoi.boxes_original_xyxy[detections]
    person_boxes = person.boxes_original_xyxy[person_slots]
    positive = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    inside = ((boxes[..., 0] >= 0) & (boxes[..., 1] >= 0)
              & (boxes[..., 2] <= width) & (boxes[..., 3] <= height) & positive)
    raw_hoi, decayed = hoi.raw_scores[detections], hoi.decayed_scores[detections]
    logits = hoi.hand_object_logits[pair_slots]
    arrays = dict(person_slots=person_slots, side_indices=sides, native_pair_slots=pair_slots,
                  detection_slots=detections, query_ids=hoi.query_ids[detections],
                  retained_nms_positions=hoi.retained_nms_positions[detections],
                  native_flat_keep=hoi.native_nms_keep[hoi.retained_nms_positions[detections]],
                  pose_keypoint_indices=indices, pose_original_xy=points, pose_raw_scores=scores,
                  pose_native_valid=valid,
                  pose_in_original_image=person.in_original_image[person_slots[:, None], indices],
                  person_boxes_original_xyxy=person_boxes,
                  person_detector_scores=person.detector_scores[person_slots],
                  hoi_boxes_original_xyxy=boxes, hoi_raw_scores=raw_hoi,
                  hoi_decayed_scores=decayed, hoi_logits=logits,
                  hoi_box_positive_area=positive, hoi_box_in_original_image=inside)
    feature = np.full((len(person_slots), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    supported = np.zeros(feature.shape, dtype=bool)

    def put(column, values, available=True):
        good = np.isfinite(values) & available
        feature[good, column] = values[good]
        supported[:, column] = good

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hand = boxes[:, 0].astype(np.float64)
        center = hand[:, :2]*.5 + hand[:, 2:]*.5
        for j in range(2):
            xy = points[:, j].astype(np.float64)
            delta = np.maximum(np.maximum(hand[:, :2]-xy, xy-hand[:, 2:]), 0)
            available = valid[:, j] & positive[:, 0]
            put(2*j, np.hypot(delta[:, 0], delta[:, 1])/diagonal, available)
            delta = xy-center
            put(2*j+1, np.hypot(delta[:, 0], delta[:, 1])/diagonal, available)
        delta = points[:, 0].astype(np.float64)-points[:, 1].astype(np.float64)
        put(4, np.hypot(delta[:, 0], delta[:, 1])/diagonal, valid.all(axis=1))
        body = person_boxes.astype(np.float64)
        extent = np.maximum(np.minimum(hand[:, 2:], body[:, 2:])-np.maximum(hand[:, :2], body[:, :2]), 0)
        intersection = extent[:, 0]*extent[:, 1]
        area_h = (hand[:, 2]-hand[:, 0])*(hand[:, 3]-hand[:, 1])
        area_b = (body[:, 2]-body[:, 0])*(body[:, 3]-body[:, 1])
        put(5, intersection/area_h, positive[:, 0])
        put(6, intersection/(area_h+area_b-intersection), positive[:, 0])
        put(7, logits[:, 1].astype(np.float64)-logits[:, 0].astype(np.float64))
    for column, values in enumerate((scores[:, 0], scores[:, 1], arrays["person_detector_scores"],
                                     raw_hoi[:, 0], raw_hoi[:, 1], decayed[:, 0], decayed[:, 1]), 8):
        put(column, values)
    _require(refs == (_reference(person, PERSON_SCHEMA), _reference(hoi, HOI_SCHEMA)),
             "Original observation banks changed during tuple construction")
    return InteractionTupleEvidence(person.original_frame_index, person.image_size, person.person_ids, refs,
                                    MappingProxyType({k: _sealed(v) for k, v in arrays.items()}),
                                    _sealed(feature), _sealed(supported))

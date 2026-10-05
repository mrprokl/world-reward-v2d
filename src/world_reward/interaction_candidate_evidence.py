"""All person/side/generic-object hypotheses; numerical evidence, NOT selection.

Generic objects are supplied automatic retained proposals, not HOI role1 or SAM
masks. Duplicate boxes and original order stay intact. Optional HOI evidence
has a separate full native-pair x generic-object bridge, never an IoU winner or
filter. IDs and SHA references fingerprint proposals, not physical identities,
source authentication, bank completeness, licensing or contact correctness.
OFF/no-contact and UNKNOWN/abstention are distinct future hypotheses in metadata;
neither is generated as a prediction, including when any bank is empty.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType

import numpy as np

from world_reward.person_pose_observations import (
    BODY_WRIST_INDICES, HAND_ROOT_INDICES, PersonPoseObservations,
    SCHEMA as PERSON_SCHEMA,
)
from world_reward.hoi_detr_observations import HOIDetrObservations, SCHEMA as HOI_SCHEMA
from world_reward.interaction_tuple_evidence import (
    InteractionTupleEvidence, _reference, _require, _sealed,
    build_interaction_tuple_evidence,
)

SCHEMA = "world_reward.interaction_candidate_evidence.v1"
OBJECT_SCHEMA = "world_reward.generic_object_observations.v1"
SIDE_NAMES = ("left", "right")
FEATURE_NAMES = (
    "body_wrist_to_object_box_over_image_diagonal",
    "body_wrist_to_object_center_over_image_diagonal",
    "hand_root_to_object_box_over_image_diagonal",
    "hand_root_to_object_center_over_image_diagonal",
    "body_wrist_hand_root_distance_over_image_diagonal",
    "person_object_box_iou", "body_wrist_raw_score", "hand_root_raw_score",
    "person_detector_raw_score", "generic_object_raw_score",
)
ROUTE_FEATURE_NAMES = ("hoi_direct_generic_object_box_iou",
                       "hoi_direct_generic_object_center_distance_over_image_diagonal")


def _text(value):
    return (type(value) is str and 0 < len(value) <= 256 and value.strip() == value
            and all(ord(c) >= 32 and ord(c) != 127 for c in value))


@dataclass(frozen=True, eq=False)
class GenericObjectObservations:
    original_frame_index: int
    image_size: tuple[int, int]  # original (height, width)
    object_ids: tuple[str, ...]
    boxes_original_xyxy: np.ndarray
    raw_scores: np.ndarray
    provenance_references: tuple[tuple[str, str], ...] = ()
    box_positive_area: np.ndarray = field(init=False)
    box_in_original_image: np.ndarray = field(init=False)

    def __post_init__(self):
        _require(type(self.original_frame_index) is int and self.original_frame_index >= 0,
                 "Nonnegative original frame index required")
        _require(type(self.image_size) is tuple and len(self.image_size) == 2
                 and all(type(n) is int and n > 0 for n in self.image_size), "Original image size required")
        _require(type(self.object_ids) is tuple and all(_text(x) for x in self.object_ids)
                 and len(set(self.object_ids)) == len(self.object_ids), "Unique original proposal IDs required")
        for name, shape in (("boxes_original_xyxy", (len(self.object_ids), 4)),
                            ("raw_scores", (len(self.object_ids),))):
            value = getattr(self, name)
            _require(type(value) is np.ndarray and value.shape == shape and value.dtype.kind in "fiu"
                     and np.isfinite(value).all(), name + ": finite plain raw array required")
            object.__setattr__(self, name, _sealed(value))
        refs = self.provenance_references
        _require(type(refs) is tuple and all(type(x) is tuple and len(x) == 2 and _text(x[0])
                 and type(x[1]) is str and re.fullmatch("[0-9a-f]{64}", x[1]) for x in refs),
                 "Bounded fingerprint references required; not source authentication")
        height, width = self.image_size
        b = self.boxes_original_xyxy
        positive = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        inside = ((b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= width)
                  & (b[:, 3] <= height) & positive)
        object.__setattr__(self, "box_positive_area", _sealed(positive))
        object.__setattr__(self, "box_in_original_image", _sealed(inside))


@dataclass(frozen=True, eq=False)
class InteractionCandidateEvidence:
    original_frame_index: int
    image_size: tuple[int, int]
    source_person_ids: tuple[str, ...]
    objects: GenericObjectObservations
    source_observation_references: tuple[tuple[str, str], ...]
    arrays: Mapping[str, np.ndarray]
    features: np.ndarray
    feature_supported: np.ndarray
    hoi_evidence: InteractionTupleEvidence | None
    hoi_routes: Mapping[str, np.ndarray]
    route_features: np.ndarray
    route_supported: np.ndarray
    scope: Mapping[str, str | int | bool]
    feature_names: tuple[str, ...] = field(default=FEATURE_NAMES, init=False)
    route_feature_names: tuple[str, ...] = field(default=ROUTE_FEATURE_NAMES, init=False)
    selection_performed: bool = field(default=False, init=False)
    anatomical_ownership_verified: bool = field(default=False, init=False)


def _iou(first, second):
    extent = np.maximum(np.minimum(first[:, 2:], second[:, 2:])
                        - np.maximum(first[:, :2], second[:, :2]), 0)
    overlap = extent[:, 0]*extent[:, 1]
    a, b = first[:, 2:]-first[:, :2], second[:, 2:]-second[:, :2]
    denominator = a[:, 0]*a[:, 1]+b[:, 0]*b[:, 1]-overlap
    ratio = overlap/denominator
    available = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                 & np.isfinite(overlap) & np.isfinite(denominator) & (denominator > 0))
    return np.where(available, ratio, np.nan)


def _put(values, support, column, raw, available=True):
    good = np.isfinite(raw) & available
    values[good, column], support[:, column] = raw[good], good


def build_interaction_candidate_evidence(person, objects, hoi=None):
    """Base N*2*O remains with zero HOI pairs; no SORT/NMS/filter/selection.

    Optional route rows are P*O in native-pair-major/object order. Join their
    native_pair_slots with hoi_evidence.arrays['native_pair_slots'] to obtain
    every person/side/native-hand/direct-object/generic-object possibility.
    Finite off-grid geometry remains diagnostic; native missed wrist/root
    distances are NaN/unsupported. Feature support is numerical availability,
    never visibility, ownership or contact. Raw banks/dtypes stay unchanged.
    """
    _require(type(person) is PersonPoseObservations and type(objects) is GenericObjectObservations,
             "Original person and generic object observations required")
    _require(person.original_frame_index == objects.original_frame_index
             and person.image_size == objects.image_size, "Same original frame/image grid required")
    _require(hoi is None or type(hoi) is HOIDetrObservations, "Optional original HOI observation required")
    if hoi is not None:
        _require(hoi.original_frame_index == person.original_frame_index and hoi.image_size == person.image_size,
                 "HOI must use the same original frame/image grid")
    observations = ((person, PERSON_SCHEMA), (objects, OBJECT_SCHEMA)) + (() if hoi is None else ((hoi, HOI_SCHEMA),))
    refs = tuple(_reference(x, schema) for x, schema in observations)
    n, o = len(person.person_ids), len(objects.object_ids)
    diagonal = math.hypot(*person.image_size)
    _require(math.isfinite(diagonal) and diagonal > 0, "Finite original image diagonal required")
    ps = np.repeat(np.arange(n, dtype=np.int64), 2*o)
    ss = np.tile(np.repeat(np.arange(2, dtype=np.int64), o), n)
    os = np.tile(np.arange(o, dtype=np.int64), 2*n)
    ki = np.column_stack((np.asarray(BODY_WRIST_INDICES)[ss], np.asarray(HAND_ROOT_INDICES)[ss]))
    xy, raw, valid = (x[ps[:, None], ki] for x in (person.keypoints_original_xy,
                                                 person.raw_scores, person.native_valid))
    boxes = objects.boxes_original_xyxy[os]
    a = dict(person_slots=ps, side_indices=ss, generic_object_slots=os, pose_keypoint_indices=ki,
             pose_original_xy=xy, pose_raw_scores=raw, pose_native_valid=valid,
             pose_in_original_image=person.in_original_image[ps[:, None], ki],
             person_boxes_original_xyxy=person.boxes_original_xyxy[ps],
             person_detector_scores=person.detector_scores[ps],
             object_boxes_original_xyxy=boxes, object_raw_scores=objects.raw_scores[os],
             object_box_positive_area=objects.box_positive_area[os],
             object_box_in_original_image=objects.box_in_original_image[os])
    features = np.full((len(ps), len(FEATURE_NAMES)), np.nan)
    support = np.zeros(features.shape, bool)
    routes = {}; route_features = np.empty((0, 2)); route_support = np.empty((0, 2), bool)
    h = None if hoi is None else build_interaction_tuple_evidence(person, hoi)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        b = boxes.astype(np.float64); center = b[:, :2]*.5+b[:, 2:]*.5
        for j in range(2):
            point = xy[:, j].astype(np.float64)
            delta = np.maximum(np.maximum(b[:, :2]-point, point-b[:, 2:]), 0)
            available = valid[:, j] & a['object_box_positive_area']
            _put(features, support, 2*j, np.hypot(delta[:, 0], delta[:, 1])/diagonal, available)
            delta = point-center
            _put(features, support, 2*j+1, np.hypot(delta[:, 0], delta[:, 1])/diagonal, available)
        delta = xy[:, 0].astype(np.float64)-xy[:, 1].astype(np.float64)
        _put(features, support, 4, np.hypot(delta[:, 0], delta[:, 1])/diagonal, valid.all(axis=1))
        _put(features, support, 5, _iou(a['person_boxes_original_xyxy'].astype(np.float64), b),
             a['object_box_positive_area'])
        for c, value in enumerate((raw[:, 0], raw[:, 1], a['person_detector_scores'], a['object_raw_scores']), 6):
            _put(features, support, c, value)
        if hoi is not None:
            pairs = np.repeat(np.arange(len(hoi.hand_object_pairs), dtype=np.int64), o)
            slots = np.tile(np.arange(o, dtype=np.int64), len(hoi.hand_object_pairs))
            ds = hoi.hand_object_pairs[pairs]
            routes = dict(native_pair_slots=pairs, generic_object_slots=slots, detection_slots=ds,
                          query_ids=hoi.query_ids[ds], raw_logits=hoi.hand_object_logits[pairs])
            hb = hoi.boxes_original_xyxy[ds[:, 1]].astype(np.float64)
            ob = objects.boxes_original_xyxy[slots].astype(np.float64)
            available = (hb[:, 2] > hb[:, 0]) & (hb[:, 3] > hb[:, 1]) & objects.box_positive_area[slots]
            route_features = np.full((len(pairs), 2), np.nan); route_support = np.zeros(route_features.shape, bool)
            _put(route_features, route_support, 0, _iou(hb, ob), available)
            delta = hb[:, :2]*.5+hb[:, 2:]*.5-ob[:, :2]*.5-ob[:, 2:]*.5
            _put(route_features, route_support, 1, np.hypot(delta[:, 0], delta[:, 1])/diagonal, available)
    _require(refs == tuple(_reference(x, schema) for x, schema in observations), "Original banks changed")
    scope = dict(OFF="not_predicted; no-contact requires evidence", UNKNOWN="not_predicted; unresolved evidence",
                 absence_predictions_generated=False, person_bank_empty=n == 0, object_bank_empty=o == 0,
                 hoi_provided=hoi is not None, hoi_native_pairs=0 if hoi is None else len(hoi.hand_object_pairs),
                 source_authenticated=False, bank_completeness_verified=False, scoring_performed=False)
    return InteractionCandidateEvidence(person.original_frame_index, person.image_size, person.person_ids,
        objects, refs, MappingProxyType({k: _sealed(v) for k, v in a.items()}), _sealed(features), _sealed(support),
        h, MappingProxyType({k: _sealed(v) for k, v in routes.items()}), _sealed(route_features),
        _sealed(route_support), MappingProxyType(scope))

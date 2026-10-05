"""Caller-weighted numerical routes, not contact probabilities or an owner selector.

All person/side/object slots survive. A and B use the same supported complete
routes; only B adds that route's original logit margin before max over routes.
The default raw-linear model never imputes unsupported active features. The
explicit masked model uses zero contributions plus numerical-availability
indicators, requiring positive source-box anchors independently of weights.
With no usable HOI route the base
score remains, with route support explicitly false: absence is never OFF.
No learning, calibration, physical-ID fusion, thresholds or model I/O occurs.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .interaction_candidate_evidence import (
    FEATURE_NAMES as BASE_NAMES, ROUTE_FEATURE_NAMES as BRIDGE_NAMES,
    OBJECT_SCHEMA, GenericObjectObservations, InteractionCandidateEvidence, _text,
)
from .interaction_tuple_evidence import (
    FEATURE_NAMES as HOI_NAMES, InteractionTupleEvidence, _reference, _require, _sealed,
)
from .person_pose_observations import BODY_WRIST_INDICES, HAND_ROOT_INDICES, SCHEMA as PERSON_SCHEMA
from .hoi_detr_observations import SCHEMA as HOI_SCHEMA

LOGIT_COLUMN = 7
TUPLE_COLUMNS = tuple(i for i in range(len(HOI_NAMES)) if i != LOGIT_COLUMN)
TUPLE_NAMES = tuple(HOI_NAMES[i] for i in TUPLE_COLUMNS)


def _fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        if type(x) is np.ndarray:
            digest.update(json.dumps([x.dtype.str, x.shape]).encode()); digest.update(x.tobytes(order="C"))
        elif is_dataclass(x):
            for field in fields(x):
                digest.update(field.name.encode() + b"\0"); visit(getattr(x, field.name))
        elif isinstance(x, Mapping):
            for key in sorted(x):
                digest.update(key.encode() + b"\0"); visit(x[key])
        elif type(x) in (tuple, list):
            for item in x: visit(item)
        else:
            digest.update(json.dumps(x, allow_nan=False).encode() + b"\0")
    visit(value)
    return digest.hexdigest()


def _array(value, shape, *, kind="numeric"):
    _require(type(value) is np.ndarray and value.shape == shape, "Original plain array shape required")
    _require(value.dtype == np.bool_ if kind == "bool" else value.dtype == np.int64 if kind == "slots"
             else value.dtype.kind in "fiu" and np.isfinite(value).all(), "Original finite array/type required")
    return value


def _features(values, available, shape):
    _require(type(values) is np.ndarray and values.shape == shape and values.dtype.kind == "f",
             "Plain floating feature bank required")
    _array(available, shape, kind="bool")
    _require(np.isfinite(values[available]).all() and np.isnan(values[~available]).all(),
             "Finite supported features and NaN unsupported features required")


@dataclass(frozen=True, eq=False)
class CoherentRouteLinear:
    """Finite raw-unit weights supplied by the caller, not learned by this module."""
    base_weights: np.ndarray
    tuple_weights: np.ndarray  # HOI_NAMES with logit column7 excluded
    bridge_weights: np.ndarray
    logit_weight: float
    bias: float = 0.

    def __post_init__(self):
        for name, count in (("base_weights", len(BASE_NAMES)), ("tuple_weights", len(TUPLE_NAMES)),
                            ("bridge_weights", len(BRIDGE_NAMES))):
            value = _array(getattr(self, name), (count,))
            object.__setattr__(self, name, _sealed(value.astype(np.float64)))
        for name in ("logit_weight", "bias"):
            value = getattr(self, name)
            _require(type(value) in (int, float) and np.isfinite(value), "Finite scalar weight required")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, eq=False)
class CoherentRouteMaskedLinear(CoherentRouteLinear):
    """Opt-in masked values and availability, not visibility or ownership.

    Caller-supplied scale division may be folded into the value weights. This
    module neither fits nor centers features. Raw features/NaNs remain intact;
    missing values contribute zero, not a claimed zero geometric distance.
    """
    base_availability_weights: np.ndarray = dataclass_field(kw_only=True)
    tuple_availability_weights: np.ndarray = dataclass_field(kw_only=True)
    bridge_availability_weights: np.ndarray = dataclass_field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        for name, count in (("base_availability_weights", len(BASE_NAMES)),
                            ("tuple_availability_weights", len(TUPLE_NAMES)),
                            ("bridge_availability_weights", len(BRIDGE_NAMES))):
            value = _array(getattr(self, name), (count,))
            object.__setattr__(self, name, _sealed(value.astype(np.float64)))


@dataclass(frozen=True, eq=False)
class CoherentRouteScores:
    original_frame_index: int
    image_size: tuple[int, int]
    source_person_ids: tuple[str, ...]
    source_object_ids: tuple[str, ...]
    source_observation_references: tuple
    source_evidence_fingerprint: str
    parameter_fingerprint: str
    base_scores: np.ndarray                 # [person,2,object], unsupported=NaN
    scores_a: np.ndarray
    scores_b: np.ndarray
    supported: np.ndarray                  # raw active-feature or opt-in structural box support
    route_supported: np.ndarray            # at least one common usable native route
    native_pair_slots: np.ndarray          # no deduplication or physical-ID claim
    native_detection_slots: np.ndarray
    native_query_ids: np.ndarray
    native_retained_nms_positions: np.ndarray
    native_flat_keep: np.ndarray
    native_identity_available: Mapping[str, np.ndarray]  # [K,2] per original identity field


def _validate(e):
    _require(type(e) is InteractionCandidateEvidence and type(e.objects) is GenericObjectObservations,
             "Original InteractionCandidateEvidence required")
    _require(type(e.original_frame_index) is int and e.original_frame_index >= 0
             and e.original_frame_index == e.objects.original_frame_index and e.image_size == e.objects.image_size,
             "Same original frame/image grid required")
    _require(type(e.image_size) is tuple and len(e.image_size) == 2
             and all(type(x) is int and x > 0 for x in e.image_size), "Original image size required")
    ids = e.source_person_ids
    _require(type(ids) is tuple and all(_text(x) for x in ids) and len(set(ids)) == len(ids), "Original person IDs required")
    n, o = len(ids), len(e.objects.object_ids); size = n * 2 * o
    _require(e.feature_names == BASE_NAMES and e.route_feature_names == BRIDGE_NAMES, "Original feature order required")
    _features(e.features, e.feature_supported, (size, len(BASE_NAMES)))
    a = e.arrays
    shapes = dict(person_slots=(size,), side_indices=(size,), generic_object_slots=(size,),
        pose_keypoint_indices=(size, 2), pose_original_xy=(size, 2, 2), pose_raw_scores=(size, 2),
        pose_native_valid=(size, 2), pose_in_original_image=(size, 2), person_boxes_original_xyxy=(size, 4),
        person_detector_scores=(size,), object_boxes_original_xyxy=(size, 4), object_raw_scores=(size,),
        object_box_positive_area=(size,), object_box_in_original_image=(size,))
    _require(isinstance(a, Mapping) and set(a) == set(shapes), "Complete original candidate arrays required")
    for name, shape in shapes.items():
        kind = "bool" if name in ('pose_native_valid', 'pose_in_original_image', 'object_box_positive_area',
                                  'object_box_in_original_image') else "slots" if name.endswith('slots') or name in (
                                  'side_indices', 'pose_keypoint_indices') else "numeric"
        _array(a[name], shape, kind=kind)
    expected = dict(person_slots=np.repeat(np.arange(n, dtype=np.int64), 2*o),
                    side_indices=np.tile(np.repeat(np.arange(2, dtype=np.int64), o), n),
                    generic_object_slots=np.tile(np.arange(o, dtype=np.int64), 2*n))
    for key, value in expected.items():
        _array(a[key], (size,), kind="slots"); _require(np.array_equal(a[key], value), "Complete original candidate slots required")
    oi, pi, si = (expected[k] for k in ("generic_object_slots", "person_slots", "side_indices"))
    keys = np.column_stack((np.asarray(BODY_WRIST_INDICES)[si], np.asarray(HAND_ROOT_INDICES)[si]))
    _require(np.array_equal(_array(a['pose_keypoint_indices'], (size, 2), kind="slots"), keys), "Original anatomical slots required")
    _require(np.array_equal(_array(a['object_boxes_original_xyxy'], (size, 4)), e.objects.boxes_original_xyxy[oi])
             and np.array_equal(_array(a['object_raw_scores'], (size,)), e.objects.raw_scores[oi]), "Original object slots required")
    refs = e.source_observation_references
    _require(type(refs) is tuple and len(refs) in (2, 3) and all(type(r) is tuple and len(r) == 2
             and type(r[0]) is str and type(r[1]) is str and len(r[1]) == 64
             and all(c in '0123456789abcdef' for c in r[1]) for r in refs)
             and refs[0][0] == PERSON_SCHEMA and refs[1] == _reference(e.objects, OBJECT_SCHEMA), "Original observation references required")
    h = e.hoi_evidence
    _require(type(e.scope.get('hoi_native_pairs')) is int and e.scope['hoi_native_pairs'] >= 0, "Native route count required")
    k = e.scope['hoi_native_pairs']
    if h is None:
        _require(k == 0 and len(refs) == 2 and not e.hoi_routes, "Absent HOI must keep an empty route bank")
    else:
        _require(type(h) is InteractionTupleEvidence and h.original_frame_index == e.original_frame_index
                 and h.image_size == e.image_size and h.source_person_ids == ids and h.feature_names == HOI_NAMES
                 and len(refs) == 3 and refs[2][0] == HOI_SCHEMA
                 and h.source_observation_references == (refs[0], refs[2]), "Same original HOI frame/grid/references required")
        _features(h.features, h.feature_supported, (n*2*k, len(HOI_NAMES)))
        _require(isinstance(h.arrays, Mapping) and {'person_slots', 'side_indices', 'native_pair_slots',
                 'detection_slots', 'query_ids', 'retained_nms_positions', 'native_flat_keep', 'hoi_logits'} <= set(h.arrays),
                 "Original native tuple identities required")
        for key, value in dict(person_slots=np.repeat(np.arange(n, dtype=np.int64), 2*k),
                              side_indices=np.tile(np.repeat(np.arange(2, dtype=np.int64), k), n),
                              native_pair_slots=np.tile(np.arange(k, dtype=np.int64), n*2)).items():
            _require(np.array_equal(_array(h.arrays[key], (n*2*k,), kind="slots"), value), "Complete original tuple slots required")
        r = e.hoi_routes
        _require(set(r) == {'native_pair_slots', 'generic_object_slots', 'detection_slots', 'query_ids', 'raw_logits'}, "Original bridge routes required")
        for key, value in dict(native_pair_slots=np.repeat(np.arange(k, dtype=np.int64), o),
                              generic_object_slots=np.tile(np.arange(o, dtype=np.int64), k)).items():
            _require(np.array_equal(_array(r[key], (k*o,), kind="slots"), value), "Complete original bridge slots required")
        for key in ('detection_slots', 'query_ids'):
            _array(r[key], (k*o, 2), kind="slots"); _require(np.all(r[key] >= 0), "Nonnegative original native slots required")
            values = _array(h.arrays[key], (n*2*k, 2), kind="slots")
            _require(np.all(values >= 0), "Nonnegative original native tuple slots required")
            if n and o:
                _require(np.array_equal(values.reshape(n*2, k, 2), np.broadcast_to(values[:k], (n*2, k, 2)))
                         and np.array_equal(r[key].reshape(k, o, 2), np.broadcast_to(values[:k, None], (k, o, 2))), "Original route identities must agree")
        for key in ('retained_nms_positions', 'native_flat_keep'):
            values = _array(h.arrays[key], (n*2*k, 2), kind="slots")
            _require(np.all(values >= 0), "Nonnegative original NMS/keep slots required")
            if n:
                _require(np.array_equal(values.reshape(n*2, k, 2), np.broadcast_to(values[:k], (n*2, k, 2))),
                         "Original NMS/keep route identities must agree")
        _array(r['raw_logits'], (k*o, 2)); logits = _array(h.arrays['hoi_logits'], (n*2*k, 2))
        if n and o:
            _require(np.array_equal(logits.reshape(n*2, k, 2), np.broadcast_to(logits[:k], (n*2, k, 2)))
                     and np.array_equal(r['raw_logits'].reshape(k, o, 2), np.broadcast_to(logits[:k, None], (k, o, 2))), "Original route logits must agree")
        with np.errstate(over="ignore", invalid="ignore"):
            margin = logits[:, 1].astype(np.float64)-logits[:, 0].astype(np.float64)
        _require(np.array_equal(h.feature_supported[:, LOGIT_COLUMN], np.isfinite(margin))
                 and np.array_equal(h.features[h.feature_supported[:, LOGIT_COLUMN], LOGIT_COLUMN], margin[np.isfinite(margin)]), "Original native logit margin required")
    _features(e.route_features, e.route_supported, (k*o, len(BRIDGE_NAMES)))
    return n, o, k


def _linear(values, supported, weights):
    active = np.flatnonzero(weights != 0)
    good = supported[..., active].all(axis=-1)
    score = np.zeros(values.shape[:-1], np.float64)
    # Fixed scalar-feature accumulation is independent of object-block width.
    for column in active:
        score += np.where(supported[..., column], values[..., column], 0.) * weights[column]
    _require(np.isfinite(score).all(), "Finite linear score required; no overflow repair")
    return score, good


def _masked_linear(values, supported, weights, availability_weights):
    score = np.zeros(values.shape[:-1], np.float64)
    for column in range(values.shape[-1]):
        if weights[column] != 0:
            score += np.where(supported[..., column], values[..., column], 0.) * weights[column]
        if availability_weights[column] != 0:
            score += supported[..., column] * availability_weights[column]
    _require(np.isfinite(score).all(), "Finite masked linear score required; no overflow repair")
    return score


def _positive_boxes(boxes):
    return (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])


def _masked_base_support(e, n, o):
    person = e.arrays['person_boxes_original_xyxy']
    objects = e.arrays['object_boxes_original_xyxy']
    positive = _positive_boxes(objects)
    _require(np.array_equal(positive, e.arrays['object_box_positive_area']), "Original object positive-area flag differs")
    if o:
        rows = person.reshape(n, 2, o, 4)
        _require(np.array_equal(rows, np.broadcast_to(rows[:, :1, :1], rows.shape)), "Original person boxes must agree across slots")
    return (_positive_boxes(person) & positive).reshape(n, 2, o)


def _masked_route_support(e, n, o, k):
    h = e.hoi_evidence; boxes = _array(h.arrays['hoi_boxes_original_xyxy'], (n*2*k, 2, 4))
    flags = _array(h.arrays['hoi_box_positive_area'], (n*2*k, 2), kind="bool")
    positive = _positive_boxes(boxes)
    _require(np.array_equal(positive, flags), "Original HOI positive-area flags differ")
    rows = boxes.reshape(n*2, k, 2, 4)
    _require(np.array_equal(rows, np.broadcast_to(rows[:1], rows.shape)), "Original HOI source boxes must agree across persons/sides")
    persons = _array(h.arrays['person_boxes_original_xyxy'], (n*2*k, 4))
    if o:
        _require(np.array_equal(persons.reshape(n, 2, k, 4),
            np.broadcast_to(e.arrays['person_boxes_original_xyxy'].reshape(n, 2, o, 4)[:, :, :1], (n, 2, k, 4))),
            "Original tuple/person source boxes differ")
    return positive.all(axis=-1).reshape(n, 2, k)


def score_coherent_routes(evidence, model, *, object_block_size=128):
    """Return both arms, retaining all P×2×O; workspace is P×2×K×block.

    Score = base + max_k(tuple_k + bridge_k,o [+ B logit_k]). Max applies
    only after assembling each coherent route, never to individual features.
    Raw-linear active-weight availability is required; the explicit masked type
    instead uses fixed original person/object and native hand/direct-box anchors.
    Missing anatomy permits a box proxy, not an ownership claim. Both arms require the
    native margin's numerical support. A missing route leaves base, not OFF.
    Evidence fingerprints establish byte stability, not source authentication.
    """
    _require(type(evidence) is InteractionCandidateEvidence and type(model) in (CoherentRouteLinear, CoherentRouteMaskedLinear)
             and type(object_block_size) is int and object_block_size > 0,
             "Caller-supplied linear model and positive object block size required")
    before, parameters = _fingerprint(evidence), _fingerprint(model)
    n, o, k = _validate(evidence)
    masked = type(model) is CoherentRouteMaskedLinear
    route_anchors = _masked_route_support(evidence, n, o, k) if masked and n and k else None
    with np.errstate(over="raise", invalid="raise"):
        values = evidence.features.reshape(n, 2, o, len(BASE_NAMES))
        supported = evidence.feature_supported.reshape(n, 2, o, len(BASE_NAMES))
        if masked:
            base = _masked_linear(values, supported, model.base_weights, model.base_availability_weights)
            good = _masked_base_support(evidence, n, o)
        else:
            base, good = _linear(values, supported, model.base_weights)
        base += model.bias
        _require(np.isfinite(base).all(), "Finite base score required")
        a, b, route_ok = base.copy(), base.copy(), np.zeros(base.shape, bool)
        if k and n and o:
            h = evidence.hoi_evidence
            x, ok = h.features.reshape(n, 2, k, len(HOI_NAMES)), h.feature_supported.reshape(n, 2, k, len(HOI_NAMES))
            if masked:
                local = _masked_linear(x[..., TUPLE_COLUMNS], ok[..., TUPLE_COLUMNS],
                                       model.tuple_weights, model.tuple_availability_weights)
                local_ok = route_anchors.copy()
            else:
                local, local_ok = _linear(x[..., TUPLE_COLUMNS], ok[..., TUPLE_COLUMNS], model.tuple_weights)
            local_ok &= ok[..., LOGIT_COLUMN]
            margin = np.where(ok[..., LOGIT_COLUMN], x[..., LOGIT_COLUMN], 0.) * model.logit_weight
            _require(np.isfinite(margin).all(), "Finite route logit contribution required")
            rv = evidence.route_features.reshape(k, o, len(BRIDGE_NAMES))
            rs = evidence.route_supported.reshape(k, o, len(BRIDGE_NAMES))
            if masked:
                bridge = _masked_linear(rv, rs, model.bridge_weights, model.bridge_availability_weights)
                bridge_ok = np.ones((k, o), bool)  # Native direct/target anchors checked independently above and in base.
            else:
                bridge, bridge_ok = _linear(rv, rs, model.bridge_weights)
            for start in range(0, o, object_block_size):
                end = min(o, start+object_block_size)
                usable = local_ok[..., None] & bridge_ok[None, None, :, start:end] & good[:, :, None, start:end]
                complete = local[..., None] + bridge[None, None, :, start:end]
                complete_b = complete + margin[..., None]
                _require(np.isfinite(complete).all() and np.isfinite(complete_b).all(), "Finite complete route scores required")
                exists = usable.any(axis=2); route_ok[:, :, start:end] = exists
                for destination, values in ((a, complete), (b, complete_b)):
                    reduced = np.max(np.where(usable, values, -np.inf), axis=2)
                    destination[:, :, start:end] += np.where(exists, reduced, 0.)
        for values in (base, a, b):
            _require(np.isfinite(values).all(), "Finite final score required")
            values[~good] = np.nan
    _require(before == _fingerprint(evidence) and parameters == _fingerprint(model), "Original evidence or parameters changed")
    r, h = evidence.hoi_routes, evidence.hoi_evidence
    identities, availability = [], {}
    for key in ('detection_slots', 'query_ids', 'retained_nms_positions', 'native_flat_keep'):
        value = np.full((k, 2), -1, np.int64)
        available = np.zeros((k, 2), bool)
        if h is not None and n:
            value, available = h.arrays[key][:k], np.ones((k, 2), bool)
        elif k and o and key in ('detection_slots', 'query_ids'):
            value, available = r[key].reshape(k, o, 2)[:, 0], np.ones((k, 2), bool)
        identities.append(_sealed(value)); availability[key] = _sealed(available)
    return CoherentRouteScores(evidence.original_frame_index, evidence.image_size, evidence.source_person_ids,
        evidence.objects.object_ids, evidence.source_observation_references, before, parameters,
        *(_sealed(v) for v in (base, a, b, good, route_ok)), _sealed(np.arange(k, dtype=np.int64)),
        *identities, MappingProxyType(availability))

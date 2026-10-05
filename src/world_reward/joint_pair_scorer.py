"""Frozen linear retrieval recipe on automatic boxes/anatomy/HOI evidence.

Feature construction and scoring take NO reference labels. Exact-coordinate
aliases and sides pool symmetrically, without interpreting crops as owners.
Private positive masks enter FIT only. The listwise denominator suppresses
unannotated alternatives implicitly; those are NOT negative/contact/OFF labels.
No prediction winner, threshold, reference repair, training I/O or model runtime.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from .interaction_candidate_evidence import (
    InteractionCandidateEvidence, FEATURE_NAMES as BASE_NAMES,
    ROUTE_FEATURE_NAMES, OBJECT_SCHEMA,
)
from .interaction_tuple_evidence import _reference, FEATURE_NAMES as TUPLE_NAMES
from .person_pose_observations import PersonPoseObservations, SCHEMA as PERSON_SCHEMA

BASE_POOLS = tuple((i, pool) for i in range(10) for pool in (("min", "max", "mean") if i < 5 else ("max",) if i == 5 else ("mean",)))
HOI_NAMES = (
    "hoi_body_wrist_hand_box_distance_min", "hoi_hand_root_hand_box_distance_min",
    "hoi_direct_generic_iou_max", "hoi_direct_generic_center_distance_min",
    "hoi_margin_max", "hoi_margin_geometry_weighted_mean",
    "hoi_hand_raw_score_geometry_weighted_mean", "hoi_direct_raw_score_geometry_weighted_mean",
)
FEATURE_NAMES = tuple(f"{BASE_NAMES[i]}_{pool}" for i, pool in BASE_POOLS) + (
    "object_center_outside_person_over_person_diagonal", "person_object_center_distance_over_person_diagonal",
) + HOI_NAMES
FIT_SLOTS, MIN_USABLE_FIT, L2 = 24, 12, 1.
MAX_ITERATIONS, MAX_BACKTRACKS, GRADIENT_TOLERANCE, FIT_SECONDS = 1000, 50, 1e-7, 120.


class InsufficientPairFitError(ValueError):
    """Too few FIT records have an observed positive and a competing hypothesis."""


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sealed(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _array(value, shape, name, *, boolean=False):
    _require(type(value) is np.ndarray and value.shape == shape
             and (value.dtype == np.bool_ if boolean else value.dtype.kind in "fiu" and np.isfinite(value).all()),
             name + ": plain finite original-shape array required")
    return np.array(value, dtype=np.bool_ if boolean else np.float64, copy=True, order="C")


def _pool(raw, available, operation):
    # Distinct VALUES give identical exact copies no extra mass; not physical IDs.
    values = np.unique(raw[available])
    if not len(values):
        return 0., False
    answer = {"min": np.min, "max": np.max, "mean": np.mean}[operation](values)
    _require(np.isfinite(answer), "Pooled automatic evidence must remain finite")
    return float(answer), True


def _coordinate_groups(boxes, height, width):
    boxes = np.array(boxes, dtype=np.float64, copy=True, order="C")
    _require(np.isfinite(boxes).all() and boxes.ndim == 2 and boxes.shape[1:] == (4,)
             and np.all(boxes[:, 2:] >= boxes[:, :2]), "Finite noninverted automatic boxes required")
    # Exact original PIXEL tuples define aliases before normalized transport.
    unique, inverse = np.unique(boxes, axis=0, return_inverse=True)
    normal = unique / np.array([width, height, width, height], np.float64)
    _require(np.isfinite(normal).all() and len(np.unique(normal, axis=0)) == len(unique),
             "Finite normalized coordinates must preserve exact automatic alias groups")
    return unique, normal, inverse.astype(np.int64)


@dataclass(frozen=True, eq=False)
class JointPairFeatures:
    original_frame_index: int
    image_size: tuple[int, int]
    person_boxes_normalized: np.ndarray
    object_boxes_normalized: np.ndarray
    raw_person_to_unique: np.ndarray
    raw_object_to_unique: np.ndarray
    values: np.ndarray                   # [unique P,unique O,30], unsupported=0
    available: np.ndarray               # numerical availability, NOT visibility
    baseline_a: np.ndarray              # lexicographic negative distances [P,O,2]
    source_observation_references: tuple
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _hoi_pool(evidence, person_slots, object_slots):
    """All sides x native pairs x supplied aliases, not an IoU-best route."""
    result, support = np.zeros(8), np.zeros(8, bool)
    h = evidence.hoi_evidence
    if h is None:
        return result, support
    n, m = len(evidence.source_person_ids), len(evidence.objects.object_ids)
    k = int(evidence.scope["hoi_native_pairs"])
    _require(h.features.shape == (n * 2 * k, 15) and evidence.route_features.shape == (k * m, 2),
             "Complete original HOI/bridge shape required")
    if not k:
        return result, support
    x, ok = h.features.reshape(n, 2, k, 15), h.feature_supported.reshape(n, 2, k, 15)
    route, route_ok = evidence.route_features.reshape(k, m, 2), evidence.route_supported.reshape(k, m, 2)
    # Deduplicate identical supported route EVIDENCE, not query/physical identities.
    rows = []
    for p in person_slots:
        for side in range(2):
            for o in object_slots:
                v = np.column_stack((x[p, side][:, [0, 2, 7, 11, 12]], route[:, o]))
                a = np.column_stack((ok[p, side][:, [0, 2, 7, 11, 12]], route_ok[:, o]))
                _require(np.isfinite(v[a]).all(), "Supported original HOI route values must be finite")
                rows.append(np.column_stack((np.where(a, v, 0.), a.astype(np.float64))))
    rows = np.unique(np.concatenate(rows), axis=0)
    v, a = rows[:, :7], rows[:, 7:].astype(bool)
    for dest, column, op in ((0, 0, "min"), (1, 1, "min"), (2, 5, "max"), (3, 6, "min"), (4, 2, "max")):
        result[dest], support[dest] = _pool(v[:, column], a[:, column], op)
    # Geometric compatibility only: no contact probability or hand-owner claim.
    weight_ok = a[:, 0] & a[:, 1] & a[:, 5]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weights = np.where(weight_ok, v[:, 5] / (1. + v[:, 0] + v[:, 1]), 0.)
    _require(np.isfinite(weights).all() and np.all(weights >= 0), "Finite nonnegative fixed route weights required")
    for dest, column in ((5, 2), (6, 3), (7, 4)):
        usable = a[:, column] & (weights > 0)
        if usable.any():
            w = weights[usable] / weights[usable].max()
            result[dest] = float((w / w.sum()) @ v[usable, column])
            support[dest] = True
    _require(np.isfinite(result).all(), "Finite pooled HOI descriptors required")
    return result, support


def build_joint_pair_features(person, evidence):
    """Pool the complete base evidence into ALL exact-coordinate P×O groups.

    person is needed for empty object banks; its byte fingerprint must match the
    evidence producer. No labels, reference geometry, score threshold or target
    prompt are accepted. Native missing joints become unavailable descriptors,
    not invented zeros: transport zeros have explicit availability indicators.
    """
    _require(type(person) is PersonPoseObservations and type(evidence) is InteractionCandidateEvidence,
             "Original person and candidate evidence required")
    fingerprint = _reference(person, PERSON_SCHEMA)
    _require(fingerprint in evidence.source_observation_references and person.person_ids == evidence.source_person_ids
             and person.image_size == evidence.image_size and person.original_frame_index == evidence.original_frame_index,
             "Person bank must match original candidate evidence")
    n, m = len(person.person_ids), len(evidence.objects.object_ids)
    _require(_reference(evidence.objects, OBJECT_SCHEMA) in evidence.source_observation_references,
             "Generic object bank must match candidate evidence")
    _require(evidence.feature_names == BASE_NAMES and evidence.features.shape == (n * 2 * m, 10)
             and evidence.feature_supported.shape == evidence.features.shape, "Complete base 10-feature bank required")
    a = evidence.arrays
    _require(np.array_equal(a["person_slots"], np.repeat(np.arange(n), 2*m))
             and np.array_equal(a["side_indices"], np.tile(np.repeat(np.arange(2), m), n))
             and np.array_equal(a["generic_object_slots"], np.tile(np.arange(m), 2*n)), "Original full base slot order required")
    if evidence.hoi_evidence is not None:
        h, routes = evidence.hoi_evidence, evidence.hoi_routes
        k = evidence.scope["hoi_native_pairs"]
        _require(type(k) is int and k >= 0 and h.feature_names == TUPLE_NAMES
                 and evidence.route_feature_names == ROUTE_FEATURE_NAMES
                 and np.array_equal(h.arrays["person_slots"], np.repeat(np.arange(n), 2*k))
                 and np.array_equal(h.arrays["side_indices"], np.tile(np.repeat(np.arange(2), k), n))
                 and np.array_equal(h.arrays["native_pair_slots"], np.tile(np.arange(k), 2*n))
                 and np.array_equal(routes["native_pair_slots"], np.repeat(np.arange(k), m))
                 and np.array_equal(routes["generic_object_slots"], np.tile(np.arange(m), k))
                 and fingerprint in h.source_observation_references
                 and h.feature_supported.dtype == np.bool_ and evidence.route_supported.dtype == np.bool_,
                 "Complete original person/side/native-pair/object HOI slot order required")
    height, width = person.image_size
    pb, pn, pg = _coordinate_groups(person.boxes_original_xyxy, height, width)
    ob, on, og = _coordinate_groups(evidence.objects.boxes_original_xyxy, height, width)
    x, valid = evidence.features.reshape(n, 2, m, 10), evidence.feature_supported.reshape(n, 2, m, 10)
    _require(valid.dtype == np.bool_ and np.isfinite(x[valid]).all(), "Finite supported base evidence required")
    values = np.zeros((len(pb), len(ob), len(FEATURE_NAMES)))
    available = np.zeros(values.shape, bool)
    baseline = np.zeros((len(pb), len(ob), 2))
    for p, box in enumerate(pb):
        ps = np.flatnonzero(pg == p)
        diagonal = math.hypot(*(box[2:] - box[:2]))
        _require(math.isfinite(diagonal) and diagonal > 0, "Positive finite automatic person diagonal required")
        for o, object_box in enumerate(ob):
            os = np.flatnonzero(og == o)
            raw, ok = x[np.ix_(ps, np.arange(2), os)], valid[np.ix_(ps, np.arange(2), os)]
            for column, (base_column, pool) in enumerate(BASE_POOLS):
                values[p, o, column], available[p, o, column] = _pool(raw[..., base_column], ok[..., base_column], pool)
            center = object_box[:2] * .5 + object_box[2:] * .5
            delta = np.maximum(np.maximum(box[:2] - center, center - box[2:]), 0.)
            distances = (math.hypot(*delta) / diagonal,
                         math.hypot(*(center - (box[:2] * .5 + box[2:] * .5))) / diagonal)
            _require(np.isfinite(distances).all(), "Finite fixed baseline geometry required")
            baseline[p, o] = -np.asarray(distances)
            j = len(BASE_POOLS)
            values[p, o, j:j+2], available[p, o, j:j+2] = distances, True
            values[p, o, j+2:], available[p, o, j+2:] = _hoi_pool(evidence, ps, os)
    _require(_reference(person, PERSON_SCHEMA) == fingerprint, "Person source changed during pooling")
    return JointPairFeatures(person.original_frame_index, person.image_size,
        *map(_sealed, (pn, on, pg, og, values, available, baseline)), evidence.source_observation_references)


def _bank(bank):
    _require(type(bank) is JointPairFeatures and bank.feature_names == FEATURE_NAMES, "Frozen joint feature schema required")
    shape = (len(bank.person_boxes_normalized), len(bank.object_boxes_normalized), len(FEATURE_NAMES))
    for coordinates, count in ((bank.person_boxes_normalized, shape[0]), (bank.object_boxes_normalized, shape[1])):
        boxes = _array(coordinates, (count, 4), "Unique automatic normalized boxes")
        _require(np.all(boxes[:, 2:] >= boxes[:, :2]) and len(np.unique(boxes, axis=0)) == count,
                 "Exact automatic coordinate aliases must already be pooled without GT")
    x = _array(bank.values, shape, "Joint values")
    ok = _array(bank.available, shape, "Joint availability", boolean=True)
    _require(np.all(x[~ok] == 0.), "Unavailable descriptors require explicit zero transport")
    return x.reshape(-1, len(FEATURE_NAMES)), ok.reshape(-1, len(FEATURE_NAMES))


def score_baseline_pairs(bank):
    """Exact lexicographic tuples -> ordinal scores, same exact ties, no weights."""
    _bank(bank)
    shape = (len(bank.person_boxes_normalized), len(bank.object_boxes_normalized))
    tuples = _array(bank.baseline_a, (*shape, 2), "Fixed baseline tuples").reshape(-1, 2)
    _, inverse = np.unique(tuples, axis=0, return_inverse=True)
    return _sealed(inverse.astype(np.float64).reshape(shape))


@dataclass(frozen=True, eq=False)
class JointPairLinear:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray             # 30 values + 30 availability indicators
    objective: float
    gradient_inf_norm: float
    iterations: int
    fit_record_count: int
    usable_fit_count: int
    missing_positive_fit_count: int
    no_alternative_fit_count: int
    fit_image_ids: tuple[str, ...]
    fit_author_groups: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    l2: float = L2


def _design(x, ok, mean, scale):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = np.where(ok, (x - mean) / scale, 0.)
        result = np.column_stack((normalized, ok.astype(np.float64)))
    _require(np.isfinite(result).all(), "Finite standardized values and availability required")
    return result


def _softmax(scores):
    shifted = scores - scores.max()
    exponential = np.exp(shifted)
    total = exponential.sum()
    return float(scores.max() + np.log(total)), exponential / total


def _loss_gradient(theta, records):
    loss, gradient = .5 * L2 * float(theta @ theta), L2 * theta.copy()
    for design, positive in records:
        scores = design @ theta
        _require(np.isfinite(scores).all(), "Finite listwise scores required")
        all_lse, all_weights = _softmax(scores)
        positive_lse, positive_weights = _softmax(scores[positive])
        loss += (all_lse - positive_lse) / len(records)
        gradient += (design.T @ all_weights - design[positive].T @ positive_weights) / len(records)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), "Finite listwise objective and gradient required")
    return loss, gradient


def fit_joint_pair_linear(banks, observed_positive_masks, *, image_ids, author_groups):
    """ONLY the 24 frozen FIT slots; unknown competitors are not binary labels.

    Image-balanced listwise observed-positive-set loss + L2=1. Missing positive
    or no competing hypothesis is excluded from loss, counted, never repaired;
    >=12 usable distinct authors required. Standardize AVAILABLE FIT values
    only, equal image mass then equal automatic pair mass, conditional on each
    feature's availability. Missing/all-constant feature scale=1. Append raw
    availability indicators. No intercept (listwise translation invariant).

    One deterministic zero start, full-gradient Armijo descent, max1000 steps,
    50 halvings, sufficient decrease1e-4, gradient infinity norm<=1e-7, <=120s.
    Multiple-positive loss need not be convex; stationary does not mean global
    optimum. No random restarts, optimizer switching or hyperparameter grid.
    """
    start = time.monotonic()
    _require(type(banks) is tuple and type(observed_positive_masks) is tuple and len(banks) == len(observed_positive_masks) == FIT_SLOTS,
             "Exactly the 24 frozen FIT records required")
    for values in (image_ids, author_groups):
        _require(type(values) is tuple and len(values) == FIT_SLOTS and len(set(values)) == FIT_SLOTS
                 and all(type(v) is str and v.strip() == v and 0 < len(v) <= 512 and not any(ord(c) < 32 for c in v) for v in values),
                 "24 unique bounded FIT image and source-author keys required")
    records, used, missing, no_alternative = [], [], 0, 0
    for i in sorted(range(FIT_SLOTS), key=lambda j: image_ids[j]):
        x, ok = _bank(banks[i])
        mask = _array(observed_positive_masks[i], banks[i].values.shape[:2], "Observed positive mask", boolean=True).reshape(-1)
        if not mask.any():
            missing += 1
        elif mask.all():
            no_alternative += 1
        else:
            records.append((x, ok, mask)); used.append(i)
    if len(records) < MIN_USABLE_FIT:
        raise InsufficientPairFitError("At least 12 FIT authors need a localized positive and competing hypothesis")
    weights = [np.full(len(x), 1. / (len(records) * len(x))) for x, _, _ in records]
    all_x, all_ok, w = np.concatenate([r[0] for r in records]), np.concatenate([r[1] for r in records]), np.concatenate(weights)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mass = w @ all_ok
        mean = np.divide(w @ np.where(all_ok, all_x, 0.), mass, out=np.zeros(len(FEATURE_NAMES)), where=mass > 0)
        variance = np.divide(w @ np.where(all_ok, (all_x - mean) ** 2, 0.), mass,
                             out=np.zeros(len(FEATURE_NAMES)), where=mass > 0)
        # Repeated non-dyadic constants can have tiny floating variance from a
        # weighted mean; detect exact constants independently, not with a floor.
        constant = np.array([np.ptp(all_x[all_ok[:, j], j]) == 0 if all_ok[:, j].any() else True
                             for j in range(len(FEATURE_NAMES))])
        scale = np.where(constant, 1., np.sqrt(variance))
    _require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0), "Finite FIT-only standardization required")
    training = [(_design(x, ok, mean, scale), mask) for x, ok, mask in records]
    theta = np.zeros(2 * len(FEATURE_NAMES))
    for iteration in range(MAX_ITERATIONS + 1):
        if time.monotonic() - start > FIT_SECONDS:
            raise RuntimeError("Frozen listwise FIT deadline exceeded")
        loss, gradient = _loss_gradient(theta, training)
        norm = float(np.max(np.abs(gradient)))
        if norm <= GRADIENT_TOLERANCE:
            break
        if iteration == MAX_ITERATIONS:
            raise RuntimeError("Frozen listwise FIT did not converge")
        descent = float(gradient @ gradient)
        _require(np.isfinite(descent) and descent > 0, "Finite descent direction required")
        for backtrack in range(MAX_BACKTRACKS):
            alpha = .5 ** backtrack
            trial = theta - alpha * gradient
            trial_loss, _ = _loss_gradient(trial, training)
            if trial_loss <= loss - 1e-4 * alpha * descent:
                theta = trial
                break
        else:
            raise RuntimeError("Frozen listwise line search failed")
    if time.monotonic() - start > FIT_SECONDS:
        raise RuntimeError("Frozen listwise FIT deadline exceeded")
    return JointPairLinear(*map(_sealed, (mean, scale, theta)), loss, norm, iteration, FIT_SLOTS, len(used), missing,
        no_alternative, tuple(image_ids[i] for i in used), tuple(author_groups[i] for i in used))


def score_joint_pairs(model, bank):
    """Every grouped automatic pair receives one raw score, NOT a probability."""
    _require(type(model) is JointPairLinear and model.feature_names == FEATURE_NAMES and model.l2 == L2,
             "Completed frozen joint linear model required")
    mean = _array(model.mean, (len(FEATURE_NAMES),), "FIT mean")
    scale = _array(model.scale, mean.shape, "FIT scale")
    coefficients = _array(model.coefficients, (2 * len(FEATURE_NAMES),), "FIT coefficients")
    _require(np.all(scale > 0), "Positive frozen FIT scale required")
    x, ok = _bank(bank)
    scores = _design(x, ok, mean, scale) @ coefficients
    _require(np.isfinite(scores).all(), "Finite frozen pair scores required")
    return _sealed(scores.reshape(bank.values.shape[:2]))

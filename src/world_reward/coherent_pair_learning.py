"""Mechanical FIT-only masked pair learner, not ownership or calibrated probability.

Native score/support arithmetic remains in coherent_route_scorer. Only complete
winning route vectors are assembled, one exact-box group at a time; no G*K*17
tensor or object pruning. Zero-start ties can still make this CPU-intensive.
Exact coordinate aliases are numerical groups, NOT physical identities. Unknown
alternatives compete in positive-set loss; they are not labelled negatives/OFF.
No split authentication, dataset policy, deadline or performance claim is supplied.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _reference, _require, _sealed
from .person_pose_observations import PersonPoseObservations, SCHEMA as PERSON_SCHEMA

FEATURE_COUNT, STEPS, L2 = 17, 512, 1.
BASE_COLUMNS, LOCAL_COLUMNS = tuple(range(6)), tuple(range(4))
BASE_INDICATORS, LOCAL_INDICATORS = (0, 2, 4), (0, 2)


@dataclass(frozen=True, eq=False)
class PairScale:
    scale: np.ndarray                   # 12 value scales; no centering
    variable: np.ndarray                # constant/all-missing FIT values disabled

    def __post_init__(self):
        _require(type(self.scale) is np.ndarray and self.scale.shape == (12,)
                 and self.scale.dtype.kind == 'f' and np.isfinite(self.scale).all()
                 and np.all(self.scale > 0), 'Twelve positive finite value scales required')
        _require(type(self.variable) is np.ndarray and self.variable.shape == (12,)
                 and self.variable.dtype == np.bool_, 'Twelve explicit variable flags required')
        object.__setattr__(self, 'scale', _sealed(self.scale.astype(np.float64)))
        object.__setattr__(self, 'variable', _sealed(self.variable))


@dataclass(frozen=True, eq=False)
class PairRouteBank:
    evidence: object
    person_boxes: np.ndarray
    object_boxes: np.ndarray
    person_to_group: np.ndarray
    object_to_group: np.ndarray

    def __post_init__(self):
        for name in ('person_boxes', 'object_boxes', 'person_to_group', 'object_to_group'):
            value = getattr(self, name)
            _require(type(value) is np.ndarray, 'Original alias array required')
            object.__setattr__(self, name, _sealed(value))
        _bank(self)


def pair_route_bank(person, evidence):
    """Keep all native slots, exposing only exact original-box alias maps."""
    _require(type(person) is PersonPoseObservations
             and _reference(person, PERSON_SCHEMA) in evidence.source_observation_references
             and person.person_ids == evidence.source_person_ids, 'Original person reference required')
    core.score_coherent_routes(evidence, masked_model(np.zeros(17), PairScale(np.ones(12), np.ones(12, bool))))
    pb, pg = np.unique(person.boxes_original_xyxy, axis=0, return_inverse=True)
    ob, og = np.unique(evidence.objects.boxes_original_xyxy, axis=0, return_inverse=True)
    return PairRouteBank(evidence, _sealed(pb), _sealed(ob), _sealed(pg.astype(np.int64)), _sealed(og.astype(np.int64)))


def _bank(bank):
    _require(type(bank) is PairRouteBank, 'Prepared original pair bank required')
    e = bank.evidence; n, o = len(e.source_person_ids), len(e.objects.object_ids)
    for mapping, boxes, count in ((bank.person_to_group, bank.person_boxes, n),
                                   (bank.object_to_group, bank.object_boxes, o)):
        _require(type(mapping) is np.ndarray and mapping.dtype == np.int64 and mapping.shape == (count,)
                 and type(boxes) is np.ndarray and boxes.ndim == 2 and boxes.shape[1:] == (4,)
                 and np.isfinite(boxes).all() and len(np.unique(boxes, axis=0)) == len(boxes)
                 and np.array_equal(np.unique(mapping), np.arange(len(boxes))), 'Complete exact alias groups required')
    _require(np.array_equal(bank.object_boxes[bank.object_to_group], e.objects.boxes_original_xyxy),
             'Original object alias coordinates differ')
    if o:
        _require(np.array_equal(bank.person_boxes[bank.person_to_group],
                 e.arrays['person_boxes_original_xyxy'].reshape(n, 2, o, 4)[:, 0, 0]), 'Original person aliases differ')
    return n, o, int(e.scope['hoi_native_pairs'])


def _blocks(bank):
    n, o, k = _bank(bank); e = bank.evidence
    base = (e.features[:, :6], e.feature_supported[:, :6])
    if e.hoi_evidence is None:
        local = (np.empty((0, 4)), np.empty((0, 4), bool))
    else:
        h = e.hoi_evidence; local = (h.features[:, :4], h.feature_supported[:, :4])
    return base, local, (e.route_features, e.route_supported)


def fit_scales(fit_banks):
    """Available FIT-only variance, equal image then distinct factor-tuple mass.

    Exact tuple duplicates have no extra mass. Mean is used only to calculate
    variance, NOT subtracted from prediction features. Constants use scale1 and
    a disabled value coefficient; indicators remain explicit.
    """
    _require(type(fit_banks) is tuple and len(fit_banks) > 0, 'Nonempty FIT bank tuple required')
    for bank in fit_banks:
        core.score_coherent_routes(bank.evidence, masked_model(np.zeros(17), PairScale(np.ones(12), np.ones(12, bool))))
    scales, variable = [], []
    for block in range(3):
        rows, weights = [], []
        for bank in fit_banks:
            x, ok = _blocks(bank)[block]
            u = np.unique(np.column_stack((np.where(ok, x, 0.), ok)), axis=0)
            if len(u):
                rows.append(u); weights.append(np.full(len(u), 1. / (len(fit_banks)*len(u))))
        width = (6, 4, 2)[block]
        if not rows:
            scales.extend([1.]*width); variable.extend([False]*width); continue
        u, w = np.concatenate(rows), np.concatenate(weights)
        for j in range(width):
            available = u[:, width+j].astype(bool); values = u[available, j]; weight = w[available]
            varied = len(values) > 0 and np.ptp(values) != 0
            scale = 1.
            if varied:
                weight = weight / weight.sum(); mean = weight @ values
                scale = math.sqrt(float(weight @ ((values-mean)**2)))
            scales.append(scale); variable.append(bool(varied))
    return PairScale(np.array(scales), np.array(variable, bool))


def masked_model(theta, scales, *, alpha=0.):
    """Fold value scales into the original masked scorer; no arithmetic fork."""
    _require(type(theta) is np.ndarray and theta.shape == (17,) and theta.dtype.kind == 'f'
             and np.isfinite(theta).all() and type(scales) is PairScale, 'Finite17 coefficients and scale required')
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha >= 0, 'Nonnegative finite relational coefficient required')
    w = np.where(scales.variable, theta[:12] / scales.scale, 0.)
    base, local, bridge = np.zeros(10), np.zeros(14), w[10:12].copy()
    ba, la = np.zeros(10), np.zeros(14)
    base[:6], local[:4] = w[:6], w[6:10]
    ba[list(BASE_INDICATORS)], la[list(LOCAL_INDICATORS)] = theta[12:15], theta[15:17]
    return core.CoherentRouteMaskedLinear(base, local, bridge, float(alpha),
        base_availability_weights=ba, tuple_availability_weights=la, bridge_availability_weights=np.zeros(2))


@dataclass(frozen=True, eq=False)
class PairGroupScores:
    native: object                       # ALL raw person/side/object scores/IDs
    scores: np.ndarray                   # grouped maxima; unsupported=NaN
    supported: np.ndarray
    derivatives: np.ndarray              # 17 for A; scalar alpha derivative for B


def _values(x, ok, scales, start):
    answer = np.where(ok, x, 0.) / scales.scale[start:start+x.shape[-1]]
    return np.where(scales.variable[start:start+x.shape[-1]], answer, 0.)


def score_pair_groups(bank, theta, scales, *, alpha=None):
    """A alpha=None is logit-blind, including tie averaging; B fits alpha only.

    Average DISTINCT winning complete vectors (17 geometry for A,18 including
    margin for B). Sorted byte keys make duplication/permutation irrelevant.
    This selects a deterministic generalized gradient, not differentiability.
    """
    n, o, k = _bank(bank); e = bank.evidence; relational = alpha is not None
    model = masked_model(theta, scales, alpha=0. if alpha is None else alpha)
    native = core.score_coherent_routes(e, model); raw = native.scores_b if relational else native.scores_a
    bv, bs = e.features.reshape(n, 2, o, 10), e.feature_supported.reshape(n, 2, o, 10)
    base = _values(bv[..., :6], bs[..., :6], scales, 0)
    if k and n and o:
        h = e.hoi_evidence; hv, hs = h.features.reshape(n, 2, k, 15), h.feature_supported.reshape(n, 2, k, 15)
        local = core._masked_linear(hv[..., core.TUPLE_COLUMNS], hs[..., core.TUPLE_COLUMNS], model.tuple_weights, model.tuple_availability_weights)
        bridge = core._masked_linear(e.route_features.reshape(k, o, 2), e.route_supported.reshape(k, o, 2), model.bridge_weights, model.bridge_availability_weights)
        usable = core._masked_route_support(e, n, o, k) & hs[..., 7]
        lv = _values(hv[..., :4], hs[..., :4], scales, 6)
        rv = _values(e.route_features.reshape(k, o, 2), e.route_supported.reshape(k, o, 2), scales, 10)
    shape = (len(bank.person_boxes), len(bank.object_boxes)); scores = np.full(shape, np.nan)
    derivatives = np.zeros((*shape, 1 if relational else 17)); supported = np.zeros(shape, bool)
    for p in range(shape[0]):
        for target in range(shape[1]):
            slots = [(i, s, j) for i in np.flatnonzero(bank.person_to_group == p) for s in range(2)
                     for j in np.flatnonzero(bank.object_to_group == target) if native.supported[i, s, j]]
            if not slots: continue
            best = max(raw[index] for index in slots); vectors = {}
            for i, s, j in slots:
                if raw[i, s, j] != best: continue
                v = np.zeros(18); v[:6], v[12:15] = base[i, s, j], bs[i, s, j, list(BASE_INDICATORS)]
                if native.route_supported[i, s, j]:
                    route = local[i, s] + bridge[:, j]
                    if relational: route = route + np.where(hs[i, s, :, 7], hv[i, s, :, 7], 0.) * float(alpha)
                    winners = np.flatnonzero(usable[i, s] & (route == route[usable[i, s]].max()))
                    for route_index in winners:
                        complete = v.copy(); complete[6:10], complete[10:12] = lv[i, s, route_index], rv[route_index, j]
                        complete[15:17] = hs[i, s, route_index, list(LOCAL_INDICATORS)]
                        complete[17] = hv[i, s, route_index, 7] if relational else 0.
                        vector = complete if relational else complete[:17]; vector[vector == 0.] = 0.
                        vectors[vector.tobytes()] = vector
                else:
                    vector = v if relational else v[:17]; vector[vector == 0.] = 0.
                    vectors[vector.tobytes()] = vector
            _require(bool(vectors), 'Winning complete route required')
            average = np.mean(np.stack([vectors[key] for key in sorted(vectors)]), axis=0)
            scores[p, target], supported[p, target] = best, True
            derivatives[p, target] = average[17:] if relational else average
    _require(np.isfinite(scores[supported]).all() and np.isfinite(derivatives).all(), 'Finite grouped score/derivative required')
    return PairGroupScores(native, _sealed(scores), _sealed(supported), _sealed(derivatives))


def _lse(x):
    offset = x.max(); weights = np.exp(x-offset); total = weights.sum()
    return float(offset+np.log(total)), weights/total


def loss_gradient(theta, fit_banks, positive_masks, scales, *, alpha=None):
    """Equal-image positive-set loss; unmatched positives remain coverage misses.

    No positive or no supported unknown alternative => no loss contribution,
    counted explicitly. Unsupported groups remain in score output/denominators.
    """
    _require(type(fit_banks) is tuple and type(positive_masks) is tuple
             and len(fit_banks) == len(positive_masks) > 0, 'Complete FIT records/masks required')
    params = theta if alpha is None else np.array([alpha], np.float64)
    loss, gradient = .5*L2*float(params @ params), L2*params.copy()
    contributions = []; missing = no_alternative = 0
    for bank, positive in zip(fit_banks, positive_masks):
        result = score_pair_groups(bank, theta, scales, alpha=alpha)
        _require(type(positive) is np.ndarray and positive.dtype == np.bool_ and positive.shape == result.scores.shape,
                 'Explicit FIT positive group mask required')
        mask = positive[result.supported]; score = result.scores[result.supported]; derivative = result.derivatives[result.supported]
        if not mask.any(): missing += 1; continue
        if mask.all(): no_alternative += 1; continue
        a, aw = _lse(score); b, bw = _lse(score[mask])
        contributions.append((a-b, aw @ derivative-bw @ derivative[mask]))
    _require(bool(contributions), 'No informative FIT loss records; no repair')
    loss += math.fsum(x[0] for x in contributions)/len(contributions)
    gradient += np.mean(np.stack([x[1] for x in contributions]), axis=0)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'Finite loss/gradient required')
    return loss, _sealed(gradient), dict(fit_records=len(fit_banks), used=len(contributions), missing_positive=missing, no_alternative=no_alternative)


@dataclass(frozen=True, eq=False)
class CoherentPairFit:
    scales: PairScale
    theta: np.ndarray
    alpha: float
    loss_a_initial: float
    loss_a_best: float
    loss_b_initial: float
    loss_b_best: float
    record_counts: dict
    iterations_per_arm: int = STEPS
    converged: bool = False


def fit_coherent_pairs(fit_banks, positive_masks, *, scales=None):
    """One zero start,512 fixed subgradient steps eta=1/(t+1),L2=1 per arm.

    Save the best FIT-loss iterate, then fit only alpha>=0 with geometry frozen.
    No stationary/global-optimum claim, CAL/RESERVED input, retries or switching.
    Caller authenticates FIT-only scope and enforces scientific/time gates.
    """
    _require(type(fit_banks) is tuple and type(positive_masks) is tuple,
             'Explicit FIT-only bank/mask tuples required')
    scales = fit_scales(fit_banks) if scales is None else scales
    masks = tuple(_sealed(x) for x in positive_masks); theta = np.zeros(17)
    initial_a, gradient, counts = loss_gradient(theta, fit_banks, masks, scales)
    best_a, best_theta = initial_a, theta.copy()
    for t in range(STEPS):
        theta = theta-gradient/(t+1)
        value, gradient, _ = loss_gradient(theta, fit_banks, masks, scales)
        if value < best_a: best_a, best_theta = value, theta.copy()
    alpha = best_alpha = 0.
    initial_b, gradient, _ = loss_gradient(best_theta, fit_banks, masks, scales, alpha=alpha); best_b = initial_b
    for t in range(STEPS):
        alpha = max(0., alpha-float(gradient[0])/(t+1))
        value, gradient, _ = loss_gradient(best_theta, fit_banks, masks, scales, alpha=alpha)
        if value < best_b: best_b, best_alpha = value, alpha
    return CoherentPairFit(scales, _sealed(best_theta), best_alpha, initial_a, best_a, initial_b, best_b, MappingProxyType(counts))

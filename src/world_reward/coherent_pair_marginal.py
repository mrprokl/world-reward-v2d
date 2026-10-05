"""Enumerative FP64 latent reference, not a learner or a qualified runtime.

Prior: equal supported anatomical sides, then distinct complete RAW geometry /
consumed-support states, then distinct native margins conditional on geometry.
Exact coordinate aliases have no extra mass; grouping is numerical, not physical
identity. Complete states are prepared without coefficients or reference labels.
All original slots/IDs remain in metadata and native score arrays. Missing anchors
are unsupported NaN, never OFF; absent usable HOI gives one base-only state.
Temperature and coefficients are caller supplied: nothing is fitted or selected.
"""
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .coherent_pair_cache import PairCache
from . import coherent_pair_learning as original
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


@dataclass(frozen=True, eq=False)
class GeometryState:
    geometry_key: str                 # RAW masked12 + support12 + indicators5 + route bit
    values: np.ndarray                # normalized12 + numerical indicators5
    margins: np.ndarray               # distinct native margins, not a maximum
    native_route_refs: tuple          # (person, side, object, native_pair); base pair=-1
    has_route: bool


def _state(cache, i, side, j, route):
    f = cache.factors
    raw = np.zeros(12); ok = np.zeros(12, bool); x = np.zeros(17)
    raw[:6], ok[:6], x[:6] = f['base'][i, side, j, :6], f['base_ok'][i, side, j, :6], f['base_values'][i, side, j]
    x[12:15] = f['base_ok'][i, side, j, list(original.BASE_INDICATORS)]
    margin = 0.
    if route >= 0:
        raw[6:10], ok[6:10] = f['local'][i, side, route, :4], f['local_ok'][i, side, route, :4]
        raw[10:12], ok[10:12] = f['bridge'][route, j], f['bridge_ok'][route, j]
        x[6:10], x[10:12] = f['local_values'][i, side, route], f['bridge_values'][route, j]
        x[15:17] = f['local_ok'][i, side, route, list(original.LOCAL_INDICATORS)]
        margin = float(f['local'][i, side, route, 7])
    raw = np.where(ok, raw, 0.).astype(np.float64)
    raw[raw == 0.] = 0.; x[x == 0.] = 0.; margin = 0. if margin == 0. else margin
    key = raw.tobytes() + ok.tobytes() + x[12:].astype(bool).tobytes() + bytes([route >= 0])
    _require(np.isfinite(x).all() and np.isfinite(raw).all() and np.isfinite(margin), 'Finite complete state required')
    return key, x, margin, (i, side, j, route)


def _collect(cache, slots):
    states = {}
    for i, side, j in slots:
        if not cache.factors['good'][i, side, j]:
            continue
        routes = np.flatnonzero(cache.factors['usable'][i, side])
        for route in routes if len(routes) else (-1,):
            key, x, margin, ref = _state(cache, int(i), side, int(j), int(route))
            if key not in states:
                states[key] = (x, {}, set())
            old, margins, refs = states[key]
            _require(old.tobytes() == x.tobytes(), 'Same raw geometry must have same fixed-scale values')
            margins[np.float64(margin).tobytes()] = margin; refs.add(ref)
    return tuple(GeometryState(key.hex(), _sealed(states[key][0]),
        _sealed(np.array([states[key][1][k] for k in sorted(states[key][1])], np.float64)),
        tuple(sorted(states[key][2])), bool(key[-1])) for key in sorted(states))


@dataclass(frozen=True, eq=False)
class MarginalPairReference:
    cache: PairCache
    native_states: tuple = field(init=False, repr=False)
    group_sides: tuple = field(init=False, repr=False)
    distribution_fingerprint: str = field(init=False)

    def __post_init__(self):
        _require(type(self.cache) is PairCache, 'Authenticated original PairCache required')
        c = self.cache; n, o, _ = c.counts
        native = tuple(_collect(c, ((i, side, j),)) for i in range(n) for side in range(2) for j in range(o))
        groups = tuple(tuple(_collect(c, ((int(i), side, int(j)) for i in persons for j in objects))
            for side in range(2)) for persons in c.person_members for objects in c.object_members)
        object.__setattr__(self, 'native_states', native); object.__setattr__(self, 'group_sides', groups)
        object.__setattr__(self, 'distribution_fingerprint', core._fingerprint((native, groups)))


@dataclass(frozen=True, eq=False)
class MarginalPairScores:
    identity: object                   # original IDs/grid/references/aliases, NOT hard-max scores
    distribution_fingerprint: str
    parameter_fingerprint: str
    temperature: float
    alpha: float
    native_scores_a: np.ndarray        # ALL [person,2,object] original slots
    native_scores_b: np.ndarray
    native_supported: np.ndarray
    native_route_supported: np.ndarray
    scores_a: np.ndarray               # exact-coordinate P/O groups
    scores_b: np.ndarray
    supported: np.ndarray
    geometry_derivatives_a: np.ndarray # [groupP,groupO,17]
    geometry_derivatives_b: np.ndarray
    alpha_derivatives_b: np.ndarray    # [groupP,groupO], NOT probability/contact


def _mix(values, temperature):
    """Stable uniform-prior logmeanexp and posterior, never unnormalized LSE."""
    _require(type(values) is np.ndarray and values.dtype == np.float64 and values.ndim == 1
             and len(values) > 0 and np.isfinite(values).all(), 'Finite FP64 mixture required')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        offset = values.max(); weights = np.exp((values-offset)/temperature); total = weights.sum()
        score = float(offset + temperature*(np.log(total)-np.log(float(len(values)))))
    _require(np.isfinite(score) and np.isfinite(weights).all(), 'Finite normalized mixture required')
    return score, weights/total


def _side(states, theta, temperature, alpha):
    x = np.stack([state.values for state in states])
    with np.errstate(over='raise', invalid='raise'):
        geometry = x @ theta
    a, wa = _mix(geometry, temperature); da = wa @ x
    if alpha == 0.:
        # Shared A path: unequal numbers/magnitudes of margins cannot reweight A.
        means = np.array([state.margins.mean() for state in states], np.float64)
        return a, a, da, da.copy(), float(wa @ means)
    adjusted, margin_derivatives = [], []
    for state, g in zip(states, geometry):
        with np.errstate(over='raise', invalid='raise'):
            contribution = g + state.margins*alpha
        score, posterior = _mix(contribution, temperature)
        adjusted.append(score); margin_derivatives.append(posterior @ state.margins)
    b, wb = _mix(np.array(adjusted, np.float64), temperature)
    return a, b, da, wb @ x, float(wb @ np.array(margin_derivatives))


def _pair(sides, theta, temperature, alpha):
    active = [_side(states, theta, temperature, alpha) for states in sides if states]
    if not active:
        return np.nan, np.nan, np.zeros(17), np.zeros(17), 0., False
    a, wa = _mix(np.array([x[0] for x in active], np.float64), temperature)
    da = wa @ np.stack([x[2] for x in active])
    if alpha == 0.:
        return a, a, da, da.copy(), float(wa @ np.array([x[4] for x in active])), True
    b, wb = _mix(np.array([x[1] for x in active], np.float64), temperature)
    return a, b, da, wb @ np.stack([x[3] for x in active]), float(wb @ np.array([x[4] for x in active])), True


def score_pair_marginal(reference, theta, *, temperature, alpha=0.):
    """No loss, labels, optimizer or hard-max reinterpretation; analytic derivatives."""
    _require(type(reference) is MarginalPairReference, 'Prepared enumerative reference required')
    _require(type(theta) is np.ndarray and theta.shape == (17,) and theta.dtype == np.float64
             and np.isfinite(theta).all(), 'Finite plain FP64 geometry coefficients required')
    _require(type(temperature) in (int, float) and np.isfinite(temperature) and temperature > 0
             and type(alpha) in (int, float) and np.isfinite(alpha) and alpha >= 0, 'Positive temperature/nonnegative alpha required')
    theta = _sealed(theta); temperature, alpha = float(temperature), float(alpha)
    c = reference.cache; n, o, _ = c.counts
    native_a = np.full((n, 2, o), np.nan); native_b = native_a.copy()
    routes = np.zeros((n, 2, o), bool)
    for index, states in enumerate(reference.native_states):
        i, side, j = np.unravel_index(index, (n, 2, o))
        a, b, _, _, _, _ = _pair((states,), theta, temperature, alpha)
        native_a[i, side, j], native_b[i, side, j] = a, b
        routes[i, side, j] = any(state.has_route for state in states)
    shape = (len(c.person_members), len(c.object_members))
    a = np.full(shape, np.nan); b = a.copy(); good = np.zeros(shape, bool)
    da = np.zeros((*shape, 17)); db = da.copy(); dm = np.zeros(shape)
    for index, sides in enumerate(reference.group_sides):
        p, j = np.unravel_index(index, shape)
        a[p, j], b[p, j], da[p, j], db[p, j], dm[p, j], good[p, j] = _pair(sides, theta, temperature, alpha)
    _require(np.isfinite(a[good]).all() and np.isfinite(b[good]).all()
             and all(np.isfinite(v).all() for v in (da, db, dm)), 'Finite score/derivative required')
    t = c.native_template
    names = ('original_frame_index', 'image_size', 'source_person_ids', 'source_object_ids',
             'source_observation_references', 'source_evidence_fingerprint', 'native_pair_slots',
             'native_detection_slots', 'native_query_ids', 'native_retained_nms_positions',
             'native_flat_keep', 'native_identity_available')
    identity = {name: getattr(t, name) for name in names}
    identity.update(person_to_group=c.factors['person_to_group'], object_to_group=c.factors['object_to_group'],
                    person_members=c.person_members, object_members=c.object_members,
                    source_bank_fingerprint=c.source_bank_fingerprint, factor_fingerprint=c.factor_fingerprint)
    return MarginalPairScores(MappingProxyType(identity), reference.distribution_fingerprint,
        core._fingerprint((theta, temperature, alpha)), temperature, alpha,
        *(_sealed(v) for v in (native_a, native_b, c.factors['good'], routes, a, b, good, da, db, dm)))

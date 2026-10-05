"""Opt-in immutable factors for the original hard-max mechanical pair learner.

No FIT, data selection or runtime qualification. Preparation authenticates the
original numerical contracts once; byte-backed copies cannot become writable.
Scores keep scalar-column accumulation and tuple+bridge[+margin] order. Only
winning complete vectors are assembled per alias group, never G*K*17 globally.
"""
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType

import numpy as np

from . import coherent_pair_learning as original
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


@dataclass(frozen=True, eq=False)
class PairCache:
    counts: tuple
    scales: original.PairScale
    factors: object
    person_members: tuple
    object_members: tuple
    native_template: core.CoherentRouteScores
    source_bank_fingerprint: str
    source_bank: original.PairRouteBank = field(repr=False)
    factor_fingerprint: str = field(init=False)

    def __post_init__(self):
        _require(type(self.counts) is tuple and len(self.counts) == 3
                 and all(type(x) is int and x >= 0 for x in self.counts)
                 and type(self.scales) is original.PairScale
                 and type(self.native_template) is core.CoherentRouteScores
                 and type(self.source_bank) is original.PairRouteBank,
                 'Original immutable cache dimensions/scales/template required')
        # Authenticate every derived factor once, not a self-declared hash. A
        # dataclass replacement must not keep provenance while changing support,
        # derivatives or IDs. No source validation occurs in repeated scoring.
        before = core._fingerprint(self.source_bank)
        expected_template = core.score_coherent_routes(self.source_bank.evidence,
            original.masked_model(np.zeros(17), self.scales))
        counts, expected, pm, om = _cache_parts(self.source_bank, self.scales)
        _require(self.counts == counts and self.source_bank_fingerprint == before
                 and core._fingerprint(self.native_template) == core._fingerprint(expected_template),
                 'Original source dimensions/native provenance/template required')
        # Equal bytes do not make caller-owned arrays immutable. Retain the
        # freshly source-derived, byte-backed template rather than their object.
        object.__setattr__(self, 'native_template', expected_template)
        n, o, k = self.counts
        shapes = dict(base=(n, 2, o, 10), base_ok=(n, 2, o, 10), local=(n, 2, k, 15),
            local_ok=(n, 2, k, 15), bridge=(k, o, 2), bridge_ok=(k, o, 2), usable=(n, 2, k),
            good=(n, 2, o), base_values=(n, 2, o, 6), local_values=(n, 2, k, 4),
            bridge_values=(k, o, 2), person_to_group=(n,), object_to_group=(o,))
        _require(set(self.factors) == set(shapes)|{'person_boxes', 'object_boxes'}, 'Complete original factor bank required')
        for name, shape in shapes.items():
            x = self.factors[name]
            _require(type(x) is np.ndarray and x.shape == shape, 'Exact original factor shape required')
            if name.endswith('_ok') or name in ('good', 'usable'):
                _require(x.dtype == np.bool_, 'Original support flags required')
        for name in ('base', 'local', 'bridge'):
            core._features(self.factors[name], self.factors[name+'_ok'], shapes[name])
        _require(all(np.isfinite(self.factors[name]).all() for name in ('base_values', 'local_values', 'bridge_values')),
                 'Finite original scale-normalized derivatives required')
        for members, mapping, count in ((self.person_members, self.factors['person_to_group'], n),
                                         (self.object_members, self.factors['object_to_group'], o)):
            _require(type(members) is tuple and np.array_equal(np.sort(np.concatenate(members) if members else np.empty(0, np.int64)), np.arange(count)),
                     'All original alias slots required')
            _require(all(np.array_equal(x, np.flatnonzero(mapping == i)) for i, x in enumerate(members)), 'Exact original alias members required')
        _require(core._fingerprint(self.factors) == core._fingerprint(expected)
                 and core._fingerprint((self.person_members, self.object_members)) == core._fingerprint((pm, om)),
                 'All raw/normalized factors, structural supports and aliases must match original source')
        object.__setattr__(self, 'factors', MappingProxyType({name: _sealed(x) for name, x in self.factors.items()}))
        for name in ('person_members', 'object_members'):
            object.__setattr__(self, name, tuple(_sealed(x) for x in getattr(self, name)))
        object.__setattr__(self, 'factor_fingerprint', core._fingerprint((self.factors, self.scales, self.person_members, self.object_members)))
        _require(before == core._fingerprint(self.source_bank), 'Original source changed during cache authentication')


def _cache_parts(bank, scales):
    """Derived exclusively from validated original observations; no label I/O."""
    n, o, k = original._bank(bank); e = bank.evidence
    base = e.features.reshape(n, 2, o, 10)
    base_ok = e.feature_supported.reshape(n, 2, o, 10)
    if e.hoi_evidence is None:
        local, local_ok = np.empty((n, 2, 0, 15)), np.empty((n, 2, 0, 15), bool)
    else:
        local = e.hoi_evidence.features.reshape(n, 2, k, 15)
        local_ok = e.hoi_evidence.feature_supported.reshape(n, 2, k, 15)
    bridge, bridge_ok = e.route_features.reshape(k, o, 2), e.route_supported.reshape(k, o, 2)
    usable = core._masked_route_support(e, n, o, k) & local_ok[..., 7] if n and k else np.zeros((n, 2, k), bool)
    factors = dict(base=base, base_ok=base_ok, local=local, local_ok=local_ok,
        bridge=bridge, bridge_ok=bridge_ok, usable=usable,
        good=core._masked_base_support(e, n, o),
        base_values=original._values(base[..., :6], base_ok[..., :6], scales, 0),
        local_values=original._values(local[..., :4], local_ok[..., :4], scales, 6),
        bridge_values=original._values(bridge, bridge_ok, scales, 10),
        person_to_group=bank.person_to_group, object_to_group=bank.object_to_group,
        person_boxes=bank.person_boxes, object_boxes=bank.object_boxes)
    pm = tuple(np.flatnonzero(bank.person_to_group == i) for i in range(len(bank.person_boxes)))
    om = tuple(np.flatnonzero(bank.object_to_group == i) for i in range(len(bank.object_boxes)))
    return (n, o, k), factors, pm, om


def prepare_pair_cache(bank, scales):
    """Authenticate once, copying ALL factors/aliases; native slots remain intact."""
    _require(type(bank) is original.PairRouteBank and type(scales) is original.PairScale,
             'Original prepared bank and scales required')
    before = core._fingerprint(bank)
    template = core.score_coherent_routes(bank.evidence, original.masked_model(np.zeros(17), scales))
    counts, factors, pm, om = _cache_parts(bank, scales)
    _require(before == core._fingerprint(bank), 'Original bank changed during cache preparation')
    result = PairCache(counts, original.PairScale(scales.scale, scales.variable), factors, pm, om, template, before, bank)
    _require(before == core._fingerprint(bank), 'Original bank changed during immutable copy')
    return result


def _native(cache, model, block):
    """Original scalar-column/order arithmetic, with immutable structural support."""
    n, o, k = cache.counts; f = cache.factors
    with np.errstate(over='raise', invalid='raise'):
        base = core._masked_linear(f['base'], f['base_ok'], model.base_weights, model.base_availability_weights)
        base += model.bias
        a, b = base.copy(), base.copy(); exists = np.zeros((n, 2, o), bool)
        local = bridge = margin = None
        if n and o and k:
            local = core._masked_linear(f['local'][..., core.TUPLE_COLUMNS], f['local_ok'][..., core.TUPLE_COLUMNS],
                                       model.tuple_weights, model.tuple_availability_weights)
            bridge = core._masked_linear(f['bridge'], f['bridge_ok'], model.bridge_weights, model.bridge_availability_weights)
            margin = np.where(f['local_ok'][..., 7], f['local'][..., 7], 0.) * model.logit_weight
            _require(np.isfinite(margin).all(), 'Finite original margin required')
            for start in range(0, o, block):
                end = min(o, start+block)
                usable = f['usable'][..., None] & f['good'][:, :, None, start:end]
                complete = local[..., None] + bridge[None, None, :, start:end]
                complete_b = complete + margin[..., None]
                _require(np.isfinite(complete).all() and np.isfinite(complete_b).all(), 'Finite complete route required')
                available = usable.any(axis=2); exists[:, :, start:end] = available
                for destination, values in ((a, complete), (b, complete_b)):
                    reduced = np.max(np.where(usable, values, -np.inf), axis=2)
                    destination[:, :, start:end] += np.where(available, reduced, 0.)
        for values in (base, a, b):
            _require(np.isfinite(values).all(), 'Finite original score required')
            values[~f['good']] = np.nan
    native = replace(cache.native_template, parameter_fingerprint=core._fingerprint(model),
        base_scores=_sealed(base), scores_a=_sealed(a), scores_b=_sealed(b),
        supported=f['good'], route_supported=_sealed(exists))
    return native, local, bridge, margin


def _distinct_mean(vectors):
    """Exactly original sorted raw-byte keys, including canonical zero handling."""
    rows = np.ascontiguousarray(np.concatenate(vectors))
    rows[rows == 0.] = 0.
    keys = rows.view(np.dtype((np.void, rows.shape[1]*rows.dtype.itemsize))).reshape(-1)
    _, indices = np.unique(keys, return_index=True)
    return np.mean(rows[indices], axis=0)


def score_pair_groups(cache, theta, *, alpha=None, object_block_size=128):
    """Same native/group scores and distinct-winner gradient; B returns scalar."""
    _require(type(cache) is PairCache and type(object_block_size) is int and object_block_size > 0,
             'Prepared immutable cache and positive block width required')
    relational = alpha is not None
    model = original.masked_model(theta, cache.scales, alpha=0. if alpha is None else alpha)
    native, local, bridge, margin = _native(cache, model, object_block_size)
    f = cache.factors; raw = native.scores_b if relational else native.scores_a
    shape = (len(cache.person_members), len(cache.object_members))
    scores = np.full(shape, np.nan); supported = np.zeros(shape, bool)
    derivatives = np.zeros((*shape, 1 if relational else 17))
    for p, persons in enumerate(cache.person_members):
        for target, objects in enumerate(cache.object_members):
            slots = [(i, s, j) for i in persons for s in range(2) for j in objects if native.supported[i, s, j]]
            if not slots: continue
            best = max(raw[index] for index in slots); vectors = []
            for i, s, j in slots:
                if raw[i, s, j] != best: continue
                if native.route_supported[i, s, j]:
                    route = local[i, s] + bridge[:, j]
                    if relational: route = route + margin[i, s]
                    winners = np.flatnonzero(f['usable'][i, s] & (route == route[f['usable'][i, s]].max()))
                    v = np.zeros((len(winners), 18 if relational else 17))
                    v[:, 6:10], v[:, 10:12] = f['local_values'][i, s, winners], f['bridge_values'][winners, j]
                    v[:, 15:17] = f['local_ok'][i, s, winners][:, list(original.LOCAL_INDICATORS)]
                    if relational: v[:, 17] = f['local'][i, s, winners, 7]
                else:
                    v = np.zeros((1, 18 if relational else 17))
                v[:, :6], v[:, 12:15] = f['base_values'][i, s, j], f['base_ok'][i, s, j, list(original.BASE_INDICATORS)]
                vectors.append(v)
            _require(bool(vectors), 'Winning original complete vectors required')
            average = _distinct_mean(vectors)
            scores[p, target], supported[p, target] = best, True
            derivatives[p, target] = average[17:] if relational else average
    _require(np.isfinite(scores[supported]).all() and np.isfinite(derivatives).all(), 'Finite group score/gradient required')
    return original.PairGroupScores(native, _sealed(scores), _sealed(supported), _sealed(derivatives))


def loss_gradient(theta, fit_caches, positive_masks, *, alpha=None, object_block_size=128):
    """Original equal-image positive-set loss+L2, without any optimizer changes."""
    _require(type(fit_caches) is tuple and type(positive_masks) is tuple
             and len(fit_caches) == len(positive_masks) > 0, 'Complete FIT caches/masks required')
    params = theta if alpha is None else np.array([alpha], np.float64)
    loss, gradient = .5*original.L2*float(params @ params), original.L2*params.copy()
    contributions = []; missing = no_alternative = 0
    for cache, positive in zip(fit_caches, positive_masks):
        result = score_pair_groups(cache, theta, alpha=alpha, object_block_size=object_block_size)
        _require(type(positive) is np.ndarray and positive.dtype == np.bool_ and positive.shape == result.scores.shape,
                 'Explicit FIT positive group mask required')
        mask = positive[result.supported]; score = result.scores[result.supported]; derivative = result.derivatives[result.supported]
        if not mask.any(): missing += 1; continue
        if mask.all(): no_alternative += 1; continue
        a, aw = original._lse(score); b, bw = original._lse(score[mask])
        contributions.append((a-b, aw @ derivative-bw @ derivative[mask]))
    _require(bool(contributions), 'No informative FIT loss records; no repair')
    loss += math.fsum(x[0] for x in contributions)/len(contributions)
    gradient += np.mean(np.stack([x[1] for x in contributions]), axis=0)
    _require(np.isfinite(loss) and np.isfinite(gradient).all(), 'Finite loss/gradient required')
    return loss, _sealed(gradient), dict(fit_records=len(fit_caches), used=len(contributions), missing_positive=missing, no_alternative=no_alternative)

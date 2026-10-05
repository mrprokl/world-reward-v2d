"""Prospective FP64 device scores/analytic VJP; no native qualification or FIT.

Host MarginalPairPacked identity is authenticated once. Owned device snapshots
are mutable Torch buffers, NOT immutable evidence; a qualified caller must own
them and check their bytes before/after its probe. No autograd, atomic scatter,
object loop, or global routes×17 features. CPU/GPU 1e-12 agreement is proposed,
not demonstrated. Alpha-zero shares A/B and retains the right alpha derivative.
"""
from dataclasses import dataclass
from types import MappingProxyType
import re

import numpy as np

from .coherent_pair_packed import MarginalPairPacked
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


@dataclass(frozen=True, eq=False, init=False)
class TorchMarginalPairPacked:
    identity: object
    distribution_fingerprint: str
    device: str
    _torch: object
    _arrays: object
    _layouts: object
    _pairs: object

    def __init__(self):
        raise TypeError('Use prepare_marginal_packed_torch; caller tables forbidden')


@dataclass(frozen=True, eq=False)
class TorchMarginalPairScores:
    """CPU field names with explicit device tensors, not the NumPy sealed ABI."""
    identity: object
    distribution_fingerprint: str
    parameter_fingerprint: str
    temperature: float
    alpha: float
    native_scores_a: object
    native_scores_b: object
    native_supported: object
    native_route_supported: object
    scores_a: object
    scores_b: object
    supported: object
    geometry_derivatives_a: object
    geometry_derivatives_b: object
    alpha_derivatives_b: object
    device: str


def _csr(offsets, entries, targets):
    _require(type(offsets) is np.ndarray and offsets.dtype == np.int64
        and offsets.shape == (targets+1,) and offsets[0] == 0
        and offsets[-1] == entries and (np.diff(offsets) >= 0).all(), 'Invalid fixed CSR offsets')
    lengths = np.diff(offsets)
    return lengths, np.repeat(np.arange(targets, dtype=np.int64), lengths)


def _indices(values, count):
    _require(type(values) is np.ndarray and values.dtype == np.int64 and values.ndim == 1
        and ((values >= 0) & (values < count)).all(), 'Invalid fixed component IDs')


def _inverse(targets, components, target_count):
    """Stable target/component inverse CSR: no repeated-index accumulation."""
    order = np.lexsort((np.arange(len(targets)), components, targets)).astype(np.int64)
    keys = np.column_stack((targets[order], components[order]))
    first = np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)] if len(keys) else np.empty(0, bool)
    starts = np.flatnonzero(first); distinct = keys[starts]
    component_offsets = np.r_[starts, len(order)].astype(np.int64)
    target_offsets = np.r_[0, np.cumsum(np.bincount(distinct[:, 0], minlength=target_count))].astype(np.int64)
    _csr(component_offsets, len(order), len(distinct)); _csr(target_offsets, len(distinct), target_count)
    return dict(order=order, component_offsets=component_offsets,
        component_ids=distinct[:, 1], target_offsets=target_offsets)


def prepare_marginal_packed_torch(packed, *, device):
    """Lazy Torch import; authenticate/validate once before uploading snapshots."""
    _require(type(packed) is MarginalPairPacked, 'Original authenticated packed distribution required')
    _require(type(device) is str and re.fullmatch(r'cpu|cuda:[0-9]+', device), 'Explicit cpu or cuda:N device required')
    before = core._fingerprint((packed.arrays, packed.identity))
    _require(before == packed.packed_fingerprint, 'Packed host identity differs')
    c = packed.cache
    _require(core._fingerprint((c.factors, c.scales, c.person_members, c.object_members))
        == c.factor_fingerprint, 'Original host factors differ')
    a = packed.arrays; components = a['geometry_components']; layouts = {}
    _require(components.dtype == np.int64 and components.ndim == 2 and components.shape[1] == 4,
        'Complete geometry component ABI required')
    names = ('base', 'local', 'bridge')
    for column, name in enumerate(names):
        values = a[name+'_values']
        _require(values.dtype == np.float64 and values.ndim == 2
            and values.shape[1] == (9, 6, 2)[column] and np.isfinite(values).all(), 'Finite fixed factors required')
        _indices(components[:, column], len(values))
    _require(a['margin_values'].dtype == np.float64 and a['margin_values'].ndim == 1
        and np.isfinite(a['margin_values']).all(), 'Finite fixed margins required')
    for prefix in ('native', 'group'):
        count = int(a['native_supported'].size) if prefix == 'native' else int(a['group_supported'].size)*2
        offsets, gids = a[prefix+'_offsets'], a[prefix+'_geometry_ids']
        moff, mids = a[prefix+'_margin_offsets'], a[prefix+'_margin_ids']
        lengths, tids = _csr(offsets, len(gids), count); mlengths, msids = _csr(moff, len(mids), len(gids))
        _require((mlengths > 0).all(), 'Every geometry needs conditional margin support')
        _indices(gids, len(components)); _indices(mids, len(a['margin_values']))
        inverse = {name:_inverse(tids, components[gids, column], count) for column, name in enumerate(names)}
        layouts[prefix] = dict(offsets=offsets, gids=gids, moff=moff, mids=mids,
            lengths=lengths, tids=tids, mlengths=mlengths, msids=msids, inverse=inverse)
    shape = a['group_supported'].shape; count = int(a['group_supported'].size)
    prior = a['group_side_prior'].reshape(count, 2); active = prior > 0
    expected = np.zeros_like(prior)
    np.divide(active, active.sum(axis=1)[:, None], out=expected, where=active.sum(axis=1)[:, None] > 0)
    _require(np.array_equal(prior, expected) and np.array_equal(active.any(axis=1).reshape(shape), a['group_supported'])
        and np.array_equal(active.reshape(-1), layouts['group']['lengths'] > 0)
        and np.array_equal(a['native_supported'].reshape(-1), layouts['native']['lengths'] > 0), 'Original support/prior ABI differs')
    entries = np.flatnonzero(active.reshape(-1)).astype(np.int64)
    poff = np.r_[0, np.cumsum(active.sum(axis=1))].astype(np.int64)
    plen, pids = _csr(poff, len(entries), count)
    import torch  # Only runtime import: no local/native execution is implied.
    def upload(value):
        if isinstance(value, dict): return MappingProxyType({k:upload(v) for k, v in value.items()})
        return torch.tensor(np.array(value, copy=True), device=device)
    selected = {name:a[name] for name in ('base_values', 'local_values', 'bridge_values',
        'geometry_components', 'margin_values', 'native_supported', 'native_route_supported', 'group_supported')}
    result = object.__new__(TorchMarginalPairPacked)
    fields = dict(identity=packed.identity, distribution_fingerprint=before, device=str(torch.device(device)),
        _torch=torch, _arrays=upload(selected), _layouts=upload(layouts),
        _pairs=upload(dict(entries=entries, offsets=poff, lengths=plen, ids=pids)))
    _require(before == core._fingerprint((packed.arrays, packed.identity)), 'Host changed during device preparation')
    for name, value in fields.items(): object.__setattr__(result, name, value)
    return result


def _finite(t, *values):
    _require(bool(t.stack([t.isfinite(x).all() for x in values]).all()), 'Device arithmetic overflow/nonfinite')


def _segment(t, x, op, offsets):
    """Width ≥1 keeps the v2.5.1 CUDA owned-output loop, not 1D CUB."""
    flat = x.ndim == 1; data = x.reshape(-1, 1) if flat else x
    if data.shape[0] == 0:
        out = t.full((offsets.numel()-1, data.shape[1]), 0. if op == 'sum' else -float('inf'),
            dtype=data.dtype, device=data.device)
    else:
        out = t.segment_reduce(data, op, offsets=offsets, axis=0)
    return out.reshape(-1) if flat else out


def _mix(t, x, offsets, ids, lengths, tau):
    if not x.numel():
        return t.full((lengths.numel(),), float('nan'), dtype=x.dtype, device=x.device), x
    _finite(t, x)
    maximum = _segment(t, x, 'max', offsets)
    shifted = (x-maximum[ids])/tau; _finite(t, shifted)
    weights = t.exp(shifted); total = _segment(t, weights, 'sum', offsets)
    active = lengths > 0
    safe_m = t.where(active, maximum, t.zeros_like(maximum))
    safe_z = t.where(active, total, t.ones_like(total))
    value = safe_m+tau*(t.log(safe_z)-t.log(lengths.clamp_min(1).to(x.dtype)))
    posterior = weights/total[ids]; _finite(t, value, posterior)
    return t.where(active, value, t.full_like(value, float('nan'))), posterior


def _geometry_vjp(t, arrays, layout, posterior):
    factors = []
    for name in ('base', 'local', 'bridge'):
        inv = layout['inverse'][name]
        weight = _segment(t, posterior[inv['order']], 'sum', inv['component_offsets'])
        values = weight[:, None]*arrays[name+'_values'][inv['component_ids']]
        _finite(t, values)
        factors.append(_segment(t, values, 'sum', inv['target_offsets']))
    base, local, bridge = factors
    result = t.cat((base[:, :6], local[:, :4], bridge, base[:, 6:], local[:, 4:]), dim=1)
    _finite(t, result)
    return result


def _targets(t, arrays, layout, geometry, tau, alpha, derivatives):
    g = geometry[layout['gids']]
    a, wa = _mix(t, g, layout['offsets'], layout['tids'], layout['lengths'], tau)
    margins = arrays['margin_values'][layout['mids']]
    if alpha == 0.:
        b, wb = a, wa
        conditional = _segment(t, margins, 'sum', layout['moff'])/layout['mlengths'].to(g.dtype) if derivatives else None
    else:
        complete = g[layout['msids']]+alpha*margins; _finite(t, complete)
        adjusted, wm = _mix(t, complete, layout['moff'], layout['msids'], layout['mlengths'], tau)
        b, wb = _mix(t, adjusted, layout['offsets'], layout['tids'], layout['lengths'], tau)
        conditional = _segment(t, wm*margins, 'sum', layout['moff']) if derivatives else None
    if not derivatives: return a, b
    _finite(t, conditional)
    da = _geometry_vjp(t, arrays, layout, wa)
    db = da if alpha == 0. else _geometry_vjp(t, arrays, layout, wb)
    dm = _segment(t, wb*conditional, 'sum', layout['offsets']); _finite(t, dm)
    return a, b, da, db, dm


def _pairs(t, arrays, layout, sides, tau, alpha):
    sa, sb, sda, sdb, sdm = sides; idx = layout['entries']
    a, wa = _mix(t, sa[idx], layout['offsets'], layout['ids'], layout['lengths'], tau)
    da = _segment(t, wa[:, None]*sda[idx], 'sum', layout['offsets'])
    if alpha == 0.: b, wb, db = a, wa, da
    else:
        b, wb = _mix(t, sb[idx], layout['offsets'], layout['ids'], layout['lengths'], tau)
        db = _segment(t, wb[:, None]*sdb[idx], 'sum', layout['offsets'])
    dm = _segment(t, wb*sdm[idx], 'sum', layout['offsets']); _finite(t, da, db, dm)
    shape = tuple(arrays['group_supported'].shape)
    return a.reshape(shape), b.reshape(shape), da.reshape((*shape, 17)), db.reshape((*shape, 17)), dm.reshape(shape)


def score_pair_marginal_packed_torch(prepared, theta, *, temperature, alpha=0.):
    """Device tensors remain on caller-owned runtime; no loss/optimizer/FIT."""
    _require(type(prepared) is TorchMarginalPairPacked, 'Factory-prepared device snapshot required')
    _require(type(theta) is np.ndarray and theta.dtype == np.float64 and theta.shape == (17,)
        and np.isfinite(theta).all(), 'Finite plain FP64 coefficients required')
    _require(type(temperature) in (int, float) and np.isfinite(temperature) and temperature > 0
        and type(alpha) in (int, float) and np.isfinite(alpha) and alpha >= 0, 'Positive tau/nonnegative alpha required')
    theta = _sealed(theta); tau, alpha = float(temperature), float(alpha)
    t = prepared._torch; arrays = prepared._arrays
    with t.no_grad():
        coef = t.tensor(np.array(theta, copy=True), dtype=t.float64, device=prepared.device)
        base = arrays['base_values'] @ coef[[0, 1, 2, 3, 4, 5, 12, 13, 14]]
        local = arrays['local_values'] @ coef[[6, 7, 8, 9, 15, 16]]
        bridge = arrays['bridge_values'] @ coef[10:12]
        comp = arrays['geometry_components']
        geometry = base[comp[:, 0]]+local[comp[:, 1]]+bridge[comp[:, 2]]
        _finite(t, base, local, bridge, geometry)
        native = _targets(t, arrays, prepared._layouts['native'], geometry, tau, alpha, False)
        sides = _targets(t, arrays, prepared._layouts['group'], geometry, tau, alpha, True)
        a, b, da, db, dm = _pairs(t, arrays, prepared._pairs, sides, tau, alpha)
    shape = tuple(arrays['native_supported'].shape)
    return TorchMarginalPairScores(prepared.identity, prepared.distribution_fingerprint,
        core._fingerprint((theta, tau, alpha)), tau, alpha, native[0].reshape(shape), native[1].reshape(shape),
        arrays['native_supported'], arrays['native_route_supported'], a, b, arrays['group_supported'],
        da, db, dm, prepared.device)

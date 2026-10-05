"""Immutable coefficient-free packed latent distribution, not a GPU scorer.

Canonical component IDs use RAW masked values/support/indicators, never learned
scales, margins or a hash alone. Composite geometry keeps the same complete
route's base/local/bridge IDs. Native and group-side CSR incidences deduplicate
geometry then conditional margin bytes; all original route references survive.
Coordinate aliases are numerical groups, not physical identities. No FIT,
temperature, optimizer, pruning, Torch or performance claim is supplied.
"""
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .coherent_pair_cache import PairCache
from . import coherent_pair_learning as original
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


def _component(raw, supported, values, indicator_columns):
    width = raw.shape[-1]; raw = raw.reshape(-1, width); supported = supported.reshape(-1, width)
    values = values.reshape(-1, width).astype(np.float64)
    masked = np.where(supported, raw, 0.).astype(np.float64); masked[masked == 0.] = 0.
    indicators = supported[:, list(indicator_columns)]
    key = np.concatenate((masked.view(np.uint8).reshape(len(masked), width*8),
                          supported.astype(np.uint8), indicators.astype(np.uint8)), axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    normalized = np.concatenate((values, indicators.astype(np.float64)), axis=1)
    normalized[normalized == 0.] = 0.
    _require(np.array_equal(normalized, normalized[first][inverse]),
             'Same raw component must have identical fixed-scale values')
    return masked[first], supported[first], normalized[first], inverse.astype(np.int64)


def _margin_table(values):
    values = values.astype(np.float64).copy(); values[values == 0.] = 0.
    keys = values.view(np.uint8).reshape(-1, 8)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return values[first], inverse.astype(np.int64)


def _csr(target, geometry, margin, targets):
    """Target → unique complete geometry → unique margin; retain inverse route map."""
    rows = np.column_stack((target, geometry, margin)).astype(np.int64)
    triples, inverse = np.unique(rows, axis=0, return_inverse=True)
    pairs = np.unique(triples[:, :2], axis=0)
    offsets = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=targets))].astype(np.int64)
    starts = np.r_[True, np.any(triples[1:, :2] != triples[:-1, :2], axis=1)] if len(triples) else np.empty(0, bool)
    margin_offsets = np.r_[np.flatnonzero(starts), len(triples)].astype(np.int64)
    return dict(offsets=offsets, geometry_ids=pairs[:, 1], margin_offsets=margin_offsets,
                margin_ids=triples[:, 2], route_to_margin_entry=inverse.astype(np.int64))


def _prepare(cache):
    n, o, k = cache.counts; f = cache.factors
    braw, bok, bval, bid = _component(f['base'][..., :6], f['base_ok'][..., :6], f['base_values'], original.BASE_INDICATORS)
    lraw, lok, lval, lid = _component(f['local'][..., :4], f['local_ok'][..., :4], f['local_values'], original.LOCAL_INDICATORS)
    rraw, rok, rval, rid = _component(f['bridge'], f['bridge_ok'], f['bridge_values'], ())
    # One explicit zero component supplies an absent route, not an inferred HOI identity.
    lraw = np.vstack((lraw, np.zeros((1, 4)))); lok = np.vstack((lok, np.zeros((1, 4), bool)))
    lval = np.vstack((lval, np.zeros((1, 6)))); absent_local = len(lraw)-1
    rraw = np.vstack((rraw, np.zeros((1, 2)))); rok = np.vstack((rok, np.zeros((1, 2), bool)))
    rval = np.vstack((rval, np.zeros((1, 2)))); absent_bridge = len(rraw)-1
    cell = np.flatnonzero(f['good'].reshape(-1)); count = f['usable'].reshape(n*2, k).sum(axis=1)
    cell_count = count[cell//o] if o else np.empty(0, np.int64)
    route_refs = np.empty((0, 4), np.int64)
    if n and o and k:
        row, route = np.nonzero(f['usable'].reshape(n*2, k))
        mask = f['good'].reshape(n*2, o)[row]
        selected, objects = np.nonzero(mask)
        row, route = row[selected], route[selected]
        route_refs = np.column_stack((row//2, row%2, objects, route)).astype(np.int64)
    base_cells = cell[cell_count == 0]
    base_refs = np.column_stack((base_cells//(2*o), (base_cells//o)%2, base_cells%o,
                                np.full(len(base_cells), -1, np.int64))) if o else np.empty((0, 4), np.int64)
    refs = np.vstack((route_refs, base_refs))
    if len(refs):
        refs = refs[np.lexsort((refs[:, 3], refs[:, 2], refs[:, 1], refs[:, 0]))]
    i, side, obj, route = refs.T; flat = (i*2+side)*o+obj; has_route = route >= 0
    local_ids = np.full(len(refs), absent_local, np.int64); bridge_ids = np.full(len(refs), absent_bridge, np.int64)
    local_ids[has_route] = lid[(i[has_route]*2+side[has_route])*k+route[has_route]]
    bridge_ids[has_route] = rid[route[has_route]*o+obj[has_route]]
    composites, route_geometry = np.unique(np.column_stack((bid[flat], local_ids, bridge_ids, has_route.astype(np.int64))),
                                           axis=0, return_inverse=True)
    margins = np.zeros(len(refs))
    margins[has_route] = f['local'][i[has_route], side[has_route], route[has_route], 7]
    margin_values, route_margin = _margin_table(margins)
    nperson, nobject = len(cache.person_members), len(cache.object_members)
    group_pair = f['person_to_group'][i]*nobject+f['object_to_group'][obj]
    group_side = group_pair*2+side
    native = _csr(flat, route_geometry, route_margin, n*2*o)
    grouped = _csr(group_side, route_geometry, route_margin, nperson*nobject*2)
    sides = np.diff(grouped['offsets']).reshape(nperson*nobject, 2)>0
    active_sides = sides.sum(axis=1)
    side_prior = np.zeros(sides.shape)
    np.divide(sides, active_sides[:, None], out=side_prior, where=active_sides[:, None]>0)
    result = dict(base_raw=braw, base_supported=bok, base_values=bval, local_raw=lraw, local_supported=lok,
        local_values=lval, bridge_raw=rraw, bridge_supported=rok, bridge_values=rval,
        base_component_ids=bid.reshape(n, 2, o), local_component_ids=lid.reshape(n, 2, k),
        bridge_component_ids=rid.reshape(k, o), geometry_components=composites.astype(np.int64),
        margin_values=margin_values, route_refs=refs, route_geometry_ids=route_geometry.astype(np.int64),
        route_margin_ids=route_margin, route_native_cells=flat, route_group_sides=group_side,
        native_supported=f['good'], native_route_supported=(f['good'] & np.repeat((count>0)[:, None], o, axis=1).reshape(n, 2, o)),
        group_supported=sides.any(axis=1).reshape(nperson, nobject), group_side_prior=side_prior.reshape(nperson, nobject, 2))
    for name, csr in (('native', native), ('group', grouped)):
        result.update({name+'_'+key:value for key,value in csr.items()})
    return result


@dataclass(frozen=True, eq=False)
class MarginalPairPacked:
    """No caller tables: source-derived once, immutable component/CSR arrays only.

    geometry_components is [baseID,localID,bridgeID,routebit]. Each CSR has
    offsets[target+1], geometry_ids, margin_offsets[geometry_incidence+1],
    margin_ids, and route_to_margin_entry mapping ALL route_refs to its segment.
    Native targets flatten P×2×O; group targets flatten groupP×groupO×2.
    All raw/native input slot-to-component maps remain even when unsupported.
    Component canonical IDs need not match the oracle's full-byte sorting order:
    equality/prior is exact; future FP64 reduction-order tolerance is undeclared.
    """
    cache: PairCache = field(repr=False)
    arrays: object = field(init=False, repr=False)
    identity: object = field(init=False, repr=False)
    packed_fingerprint: str = field(init=False)

    def __post_init__(self):
        _require(type(self.cache) is PairCache, 'Authenticated original PairCache required')
        c = self.cache; before = core._fingerprint((c.factors, c.scales, c.person_members, c.object_members))
        _require(before == c.factor_fingerprint, 'Original immutable factor fingerprint differs')
        arrays = _prepare(c)
        _require(before == core._fingerprint((c.factors, c.scales, c.person_members, c.object_members)), 'Source changed during packed preparation')
        object.__setattr__(self, 'arrays', MappingProxyType({name:_sealed(value) for name,value in arrays.items()}))
        t = c.native_template
        names = ('original_frame_index', 'image_size', 'source_person_ids', 'source_object_ids',
            'source_observation_references', 'source_evidence_fingerprint', 'native_pair_slots',
            'native_detection_slots', 'native_query_ids', 'native_retained_nms_positions',
            'native_flat_keep', 'native_identity_available')
        identity = {name:getattr(t,name) for name in names}
        identity.update(person_to_group=c.factors['person_to_group'], object_to_group=c.factors['object_to_group'],
            person_members=c.person_members, object_members=c.object_members,
            source_bank_fingerprint=c.source_bank_fingerprint, factor_fingerprint=c.factor_fingerprint)
        object.__setattr__(self, 'identity', MappingProxyType(identity))
        object.__setattr__(self, 'packed_fingerprint', core._fingerprint((self.arrays, self.identity)))


def prepare_marginal_packed(cache):
    """Prepare only. No scores, VJP, labels, GPU/runtime or resource qualification."""
    return MarginalPairPacked(cache)

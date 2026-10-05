"""FP64 packed-CSR latent scores and analytic VJP, not a fitted/GPU runtime.

Uniform prior: distinct margins conditional on complete RAW geometry, then
distinct geometry, then equal supported anatomical sides. ALL native slots/IDs
remain; absent routes are base-only, unsupported anchors NaN, never OFF.
Component dot products and posterior-weighted component VJPs avoid a global
route×17 tensor. No GeometryState objects, optimizer, pruning or labels occur.

Cross-arithmetic reference tolerance is rtol=atol=1e-12 on tiny authored cases;
alpha=0 shares the same A score/geometry-VJP path exactly within this runtime.
Its alpha derivative still averages distinct conditional native margins.
"""
import numpy as np

from .coherent_pair_packed import MarginalPairPacked
from .coherent_pair_marginal import MarginalPairScores
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


def _mix(values, temperature):
    """Stable FP64 uniform logmeanexp and posterior over a nonempty segment."""
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), 'Finite nonempty packed segment required')
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        offset = values.max(); weights = np.exp((values-offset)/temperature); total = weights.sum()
        score = float(offset+temperature*(np.log(total)-np.log(float(len(values)))))
        weights /= total
    _require(np.isfinite(score) and np.isfinite(weights).all(), 'Finite uniform packed mixture required')
    return score,weights


def _component_scores(arrays, theta):
    with np.errstate(over='raise',invalid='raise'):
        base = arrays['base_values'] @ theta[np.r_[0:6,12:15]]
        local = arrays['local_values'] @ theta[np.r_[6:10,15:17]]
        bridge = arrays['bridge_values'] @ theta[10:12]
        b,l,r,_ = arrays['geometry_components'].T
        geometry = base[b]+local[l]+bridge[r]
    _require(all(np.isfinite(x).all() for x in (base,local,bridge,geometry)), 'Finite complete component score required')
    return geometry


def _geometry_vjp(arrays, geometry_ids, posterior):
    """One target's posterior over components; never ALL routes×17."""
    b,l,r,_ = arrays['geometry_components'][geometry_ids].T
    result = np.zeros(17)
    with np.errstate(over='raise',invalid='raise'):
        result[np.r_[0:6,12:15]] = posterior @ arrays['base_values'][b]
        result[np.r_[6:10,15:17]] = posterior @ arrays['local_values'][l]
        result[10:12] = posterior @ arrays['bridge_values'][r]
    _require(np.isfinite(result).all(), 'Finite packed geometry VJP required')
    return result


def _targets(arrays, prefix, geometry, temperature, alpha, derivatives):
    """Direct CSR: margin → complete geometry; no enumerative state objects."""
    offsets = arrays[prefix+'_offsets']; gids = arrays[prefix+'_geometry_ids']
    moff = arrays[prefix+'_margin_offsets']; mids = arrays[prefix+'_margin_ids']; margins = arrays['margin_values']
    count = len(offsets)-1; a = np.full(count,np.nan); b = a.copy()
    da = np.zeros((count,17)) if derivatives else None
    db = np.zeros((count,17)) if derivatives else None
    dm = np.zeros(count) if derivatives else None
    for target in range(count):
        start,end = offsets[target:target+2]
        if start == end: continue
        ids = gids[start:end]; score = geometry[ids]
        a[target],wa = _mix(score,temperature)
        if derivatives: da[target] = _geometry_vjp(arrays,ids,wa)
        if alpha == 0.:
            b[target] = a[target]
            if derivatives:
                means = np.array([margins[mids[moff[i]:moff[i+1]]].mean() for i in range(start,end)])
                db[target] = da[target]; dm[target] = wa @ means
            continue
        adjusted = np.empty(end-start); conditional = np.zeros(end-start)
        for local,i in enumerate(range(start,end)):
            values = margins[mids[moff[i]:moff[i+1]]]
            with np.errstate(over='raise',invalid='raise'): complete = score[local]+alpha*values
            adjusted[local],wm = _mix(complete,temperature)
            if derivatives: conditional[local] = wm @ values
        b[target],wb = _mix(adjusted,temperature)
        if derivatives:
            db[target] = _geometry_vjp(arrays,ids,wb); dm[target] = wb @ conditional
    _require(np.isfinite(a[np.diff(offsets)>0]).all() and np.isfinite(b[np.diff(offsets)>0]).all(), 'Finite supported packed target scores required')
    return a,b,da,db,dm


def _pairs(arrays, sides, temperature, alpha):
    """Equal supported-side prior; unsupported sides never get a zero state."""
    side_a,side_b,side_da,side_db,side_dm = sides
    shape = arrays['group_supported'].shape; count = int(np.prod(shape))
    a = np.full(count,np.nan); b = a.copy(); da = np.zeros((count,17)); db = da.copy(); dm = np.zeros(count)
    prior = arrays['group_side_prior'].reshape(count,2)
    for target in range(count):
        active = np.flatnonzero(prior[target]>0)
        if not len(active): continue
        indices = target*2+active
        a[target],wa = _mix(side_a[indices],temperature); da[target] = wa @ side_da[indices]
        if alpha == 0.:
            b[target] = a[target]; db[target] = da[target]; dm[target] = wa @ side_dm[indices]
        else:
            b[target],wb = _mix(side_b[indices],temperature)
            db[target] = wb @ side_db[indices]; dm[target] = wb @ side_dm[indices]
    supported = arrays['group_supported'].reshape(count)
    _require(all(np.isfinite(x).all() for x in (da,db,dm)) and np.isfinite(a[supported]).all()
        and np.isfinite(b[supported]).all(), 'Finite supported packed group score/VJP required')
    return a.reshape(shape),b.reshape(shape),da.reshape((*shape,17)),db.reshape((*shape,17)),dm.reshape(shape)


def score_pair_marginal_packed(packed, theta, *, temperature, alpha=0.):
    """Pure scoring of authenticated frozen tables, no repeated bank validation."""
    _require(type(packed) is MarginalPairPacked, 'Authenticated immutable packed distribution required')
    _require(type(theta) is np.ndarray and theta.dtype == np.float64 and theta.shape == (17,)
        and np.isfinite(theta).all(), 'Finite plain FP64 geometry coefficients required')
    _require(type(temperature) in (int,float) and np.isfinite(temperature) and temperature > 0
        and type(alpha) in (int,float) and np.isfinite(alpha) and alpha >= 0, 'Positive temperature/nonnegative alpha required')
    theta = _sealed(theta); temperature,alpha = float(temperature),float(alpha)
    arrays = packed.arrays; geometry = _component_scores(arrays,theta)
    native = _targets(arrays,'native',geometry,temperature,alpha,False)
    sides = _targets(arrays,'group',geometry,temperature,alpha,True)
    a,b,da,db,dm = _pairs(arrays,sides,temperature,alpha)
    shape = arrays['native_supported'].shape
    return MarginalPairScores(packed.identity,packed.packed_fingerprint,
        core._fingerprint((theta,temperature,alpha)),temperature,alpha,
        *(_sealed(x) for x in (native[0].reshape(shape),native[1].reshape(shape),arrays['native_supported'],
            arrays['native_route_supported'],a,b,arrays['group_supported'],da,db,dm)))

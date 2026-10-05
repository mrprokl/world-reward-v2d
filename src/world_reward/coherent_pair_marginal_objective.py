"""Fixed-population positive-set loss and analytic VJP; no optimizer or FIT.

Unlisted alternatives compete in the surrogate, NOT certified negatives/OFF.
Missing positives have no defined data-loss term: their zero numerator convention
is explicit, their records remain in the fixed denominator and status is partial.
Caller authenticates reference scope and owns scorer/device snapshot integrity.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import re
from types import MappingProxyType

import numpy as np

from . import coherent_route_scorer as core
from .coherent_pair_marginal import MarginalPairScores
from .interaction_tuple_evidence import _require, _sealed


@dataclass(frozen=True, eq=False)
class MarginalObjective:
    loss: object                       # CPU float or device FP64 scalar
    gradient: object                   # Geometry17 A, scalar1 B; analytic, no autograd
    record_losses: object              # NaN where the data loss is undefined
    record_statuses: tuple
    counts: object
    status: str                        # complete / partial / no_informative
    arm: str
    regularization: float
    score_references: tuple            # Original distribution + parameter fingerprints
    fixed_population_average: bool = True
    unlisted_alternatives_verified_negative: bool = False


def _inputs(scores, masks, theta, alpha, regularization, arm, score_type=MarginalPairScores):
    _require(type(scores) is tuple and type(masks) is tuple and len(scores)==len(masks)>0,
        'Complete nonempty fixed record tuples required')
    _require(type(theta) is np.ndarray and theta.dtype==np.float64 and theta.shape==(17,)
        and np.isfinite(theta).all(), 'Plain finite FP64 geometry17 required')
    _require(all(type(s) is score_type for s in scores), 'Genuine marginal score type required')
    _require(type(alpha) in (int,float) and math.isfinite(alpha) and alpha>=0
        and type(regularization) in (int,float) and math.isfinite(regularization) and regularization>=0
        and type(arm) is str and arm in ('A','B'), 'Explicit arm/nonnegative alpha/regularization required')
    for score,mask in zip(scores,masks):
        _require(type(mask) is np.ndarray and mask.dtype==np.bool_ and mask.shape==tuple(score.supported.shape),
            'Unchanged full-group positive boolean mask required')
        _require(type(score.temperature) is float and math.isfinite(score.temperature) and score.temperature>0
            and score.temperature==scores[0].temperature and type(score.alpha) is float and score.alpha==float(alpha),
            'One uniform temperature and exact supplied scoring parameters required')
        _require(score.parameter_fingerprint==core._fingerprint((theta,score.temperature,float(alpha)))
            and type(score.distribution_fingerprint) is str and re.fullmatch('[0-9a-f]{64}',score.distribution_fingerprint),
            'Scorer parameter/distribution fingerprint required')
        _identity(score)
    return _sealed(theta),float(alpha),float(regularization)


def _identity(score):
    identity=score.identity;_require(isinstance(identity,Mapping), 'Original group identity mapping required')
    sizes=[]
    for prefix in ('person','object'):
        ids=identity['source_'+prefix+'_ids'];members=identity[prefix+'_members'];mapping=identity[prefix+'_to_group']
        _require(type(ids) is tuple and all(type(v) is str for v in ids) and len(set(ids))==len(ids)
            and type(members) is tuple and type(mapping) is np.ndarray and mapping.dtype==np.int64
            and mapping.shape==(len(ids),), 'Complete native IDs/group maps required')
        covered=[]
        for group,slots in enumerate(members):
            _require(type(slots) is np.ndarray and slots.dtype==np.int64 and slots.ndim==1 and len(slots)>0
                and (slots>=0).all() and (slots<len(ids)).all() and (np.diff(slots)>0).all()
                and (mapping[slots]==group).all(), 'Exact complete alias membership required')
            covered.extend(slots.tolist())
        _require(sorted(covered)==list(range(len(ids))), 'Every native slot appears exactly once')
        sizes.append(len(members))
    _require(tuple(score.supported.shape)==tuple(sizes), 'Full exact-group score grid required')


def _counts(rows):
    counts=dict(records=len(rows),used=0,no_positive=0,missing_positive=0,no_alternative=0,
        partially_missing_positive=0,positive_groups=0,supported_positive_groups=0,
        unsupported_positive_groups=0,supported_groups=0)
    statuses=[]
    for positive,matched,supported in rows:
        counts['positive_groups']+=positive;counts['supported_positive_groups']+=matched
        counts['unsupported_positive_groups']+=positive-matched;counts['supported_groups']+=supported
        if positive==0:status='no_positive'
        elif matched==0:status='missing_positive'
        elif matched==supported:status='no_alternative'
        else:status='used';counts['used']+=1
        if status!='used':counts[status]+=1
        if 0<matched<positive:counts['partially_missing_positive']+=1
        statuses.append(status if not 0<matched<positive else status+'_partial_positive')
    status='no_informative' if not counts['used'] else ('partial' if counts['no_positive'] or
        counts['missing_positive'] or counts['unsupported_positive_groups'] else 'complete')
    return tuple(statuses),MappingProxyType(counts),status


def _numpy_score(score):
    _require(type(score) is MarginalPairScores, 'Genuine CPU marginal score ABI required')
    shape=score.supported.shape;n,o=len(score.identity['source_person_ids']),len(score.identity['source_object_ids'])
    _require(len(shape)==2, 'Full group grid required')
    for name,s in (('native_supported',(n,2,o)),('native_route_supported',(n,2,o)),('supported',shape)):
        a=getattr(score,name);_require(type(a) is np.ndarray and a.dtype==np.bool_ and a.shape==s, 'Original boolean supports required')
    _require(not (score.native_route_supported & ~score.native_supported).any(), 'Route support cannot invent base support')
    for name,s,support in (('native_scores_a',(n,2,o),score.native_supported),('native_scores_b',(n,2,o),score.native_supported),
        ('scores_a',shape,score.supported),('scores_b',shape,score.supported)):
        a=getattr(score,name);_require(type(a) is np.ndarray and a.dtype==np.float64 and a.shape==s
            and np.isfinite(a[support]).all() and np.isnan(a[~support]).all(), 'Finite supported scores/raw unsupported NaN required')
    for name,s in (('geometry_derivatives_a',(*shape,17)),('geometry_derivatives_b',(*shape,17)),('alpha_derivatives_b',shape)):
        a=getattr(score,name);_require(type(a) is np.ndarray and a.dtype==np.float64 and a.shape==s
            and np.isfinite(a).all() and not np.any(a[~score.supported]), 'Finite analytic VJP and zero unsupported derivative required')


def _lse_numpy(values):
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        offset=values.max();weight=np.exp(values-offset);total=weight.sum()
        return float(offset),float(np.log(total)),weight/total


def marginal_objective(scores, positive_masks, theta, *, alpha=0., regularization=0., arm='A'):
    """Fixed-N mean; undefined record loss is NaN, never averaged only over used.

    A learns theta only. B freezes theta and learns alpha only. Regularization is
    applied once to the active parameter vector, even with missing records.
    No-informative output must not be interpreted as a qualified fitting loss.
    """
    borrowed_theta=theta;theta,alpha,lam=_inputs(scores,positive_masks,theta,alpha,regularization,arm)
    before=core._fingerprint((scores,positive_masks,borrowed_theta))
    params=theta if arm=='A' else np.array([alpha],np.float64)
    losses=np.full(len(scores),np.nan);gradient=np.zeros(len(params));contributions=[];rows=[]
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        for i,(score,positive) in enumerate(zip(scores,positive_masks)):
            _numpy_score(score);support=score.supported;matched=positive[support]
            rows.append((int(positive.sum()),int(matched.sum()),int(support.sum())))
            if not matched.any():continue
            if matched.all():losses[i]=0.;continue
            values=getattr(score,'scores_'+arm.lower())[support]
            derivative=(getattr(score,'geometry_derivatives_a')[support] if arm=='A' else score.alpha_derivatives_b[support,None])
            derivative=derivative-derivative[0]
            a,az,all_w=_lse_numpy(values);b,bz,pos_w=_lse_numpy(values[matched])
            losses[i]=(a-b)+(az-bz);contributions.append(losses[i])
            gradient+=all_w@derivative-pos_w@derivative[matched]
        loss=math.fsum(contributions)/len(scores)+(.5*lam*float(params@params) if lam else 0.)
        gradient=gradient/len(scores)+lam*params
    _require(math.isfinite(loss) and np.isfinite(gradient).all(), 'Finite complete objective/VJP required')
    _require(before==core._fingerprint((scores,positive_masks,borrowed_theta)), 'Borrowed scorer/mask/coefficients changed')
    statuses,counts,status=_counts(rows)
    return MarginalObjective(loss,_sealed(gradient),_sealed(losses),statuses,counts,status,arm,lam,
        tuple((s.distribution_fingerprint,s.parameter_fingerprint) for s in scores))


def marginal_objective_torch(scores, positive_masks, theta, *, alpha=0., regularization=0., arm='A'):
    """Lazy device equivalent; no numerical import/execution occurs until called.

    Tensors stay on their explicit scorer device. No in-place writes/autograd;
    caller must independently authenticate full owned snapshot bytes pre/post.
    """
    from .coherent_pair_packed_torch import TorchMarginalPairScores
    borrowed_theta=theta;borrowed_before=core._fingerprint((positive_masks,theta))
    theta,alpha,lam=_inputs(scores,positive_masks,theta,alpha,regularization,arm,TorchMarginalPairScores)
    _require(all(type(s) is TorchMarginalPairScores for s in scores), 'Genuine Torch marginal score ABI required')
    import torch as t
    device=scores[0].device;_require(all(s.device==device for s in scores), 'One explicit scorer device required')
    params=theta if arm=='A' else np.array([alpha],np.float64);rows=[];losses=[]
    with t.no_grad():
        p=t.tensor(np.array(params,copy=True),dtype=t.float64,device=device);gradient=t.zeros_like(p)
        total=t.zeros((),dtype=t.float64,device=device)
        for score,positive in zip(scores,positive_masks):
            support=score.supported;shape=tuple(support.shape)
            n,o=len(score.identity['source_person_ids']),len(score.identity['source_object_ids'])
            fields=(('supported',shape,t.bool),('native_supported',(n,2,o),t.bool),('native_route_supported',(n,2,o),t.bool),
                ('scores_a',shape,t.float64),('scores_b',shape,t.float64),('native_scores_a',(n,2,o),t.float64),
                ('native_scores_b',(n,2,o),t.float64),('geometry_derivatives_a',(*shape,17),t.float64),
                ('geometry_derivatives_b',(*shape,17),t.float64),('alpha_derivatives_b',shape,t.float64))
            _require(len(shape)==2, 'Complete device group grid required')
            for name,s,dtype in fields:
                a=getattr(score,name);_require(type(a) is t.Tensor and a.dtype==dtype and tuple(a.shape)==s
                    and str(a.device)==device and not a.requires_grad, 'Original detached FP64/device/support ABI required')
            _require(not bool((score.native_route_supported & ~score.native_supported).any()), 'Device route support differs')
            for name,good in (('scores_a',support),('scores_b',support),('native_scores_a',score.native_supported),('native_scores_b',score.native_supported)):
                a=getattr(score,name);_require(bool(t.isfinite(a[good]).all() & t.isnan(a[~good]).all()), 'Device supported finite/raw NaN required')
            for name in ('geometry_derivatives_a','geometry_derivatives_b','alpha_derivatives_b'):
                a=getattr(score,name);_require(bool(t.isfinite(a).all()) and not bool(a[~support].any()), 'Device analytic VJP invalid')
            pos=t.tensor(np.array(positive,copy=True),dtype=t.bool,device=device)[support]
            rows.append((int(positive.sum()),int(pos.sum()),int(support.sum())))
            if not bool(pos.any()):losses.append(t.full((),float('nan'),dtype=t.float64,device=device));continue
            if bool(pos.all()):losses.append(t.zeros((),dtype=t.float64,device=device));continue
            x=getattr(score,'scores_'+arm.lower())[support]
            d=score.geometry_derivatives_a[support] if arm=='A' else score.alpha_derivatives_b[support,None]
            d=d-d[0]
            _require(bool(t.isfinite(d).all()), 'Centered device derivative overflow')
            def lse(values):
                offset=values.max();weight=t.exp(values-offset);z=weight.sum()
                _require(bool(t.isfinite(weight).all()) and bool(t.isfinite(z)), 'Device objective overflow')
                return offset,t.log(z),weight/z
            a,az,aw=lse(x);b,bz,bw=lse(x[pos]);term=(a-b)+(az-bz)
            losses.append(term);total=total+term;gradient=gradient+aw@d-bw@d[pos]
        loss=total/len(scores)+(.5*lam*(p@p) if lam else 0.);gradient=gradient/len(scores)+lam*p
        _require(bool(t.isfinite(loss)) and bool(t.isfinite(gradient).all()), 'Finite device objective/VJP required')
    statuses,counts,status=_counts(rows)
    _require(borrowed_before==core._fingerprint((positive_masks,borrowed_theta)), 'Borrowed device masks/coefficients changed')
    return MarginalObjective(loss,gradient,t.stack(losses),statuses,counts,status,arm,lam,
        tuple((s.distribution_fingerprint,s.parameter_fingerprint) for s in scores))

"""Feasible-start sparse native pose projection; no model/data I/O.

Variables per original frame: 23 right-SO3 tangents +58 SO2 angles, human T,
object T =133. A caller supplies authenticated full native decode, objective and
native witness Jacobians in THAT tangent ABI. No 136-control/mesh interpolation.
The positive diagonal proximal QP decomposes by frame; temporal objective terms
may contribute to its full-clip gradient/diagonal, not an invented dense Hessian.
Exact all-face witness checks and independent whole-clip gates authorize steps.
This local numerical method does not establish whole-hand/PEN/physical truth.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import contact_feasible_placement as contact
from .native_contact_continuation import (BaselineReferenceFailure, ObservationGate,
    _decode_geometry, _object_in_front, _observation_acceptance, _so3)
from .shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters

ROTATION_DIM = 127
STATE_DIM = 133


@dataclass(frozen=True)
class PoseSQPConfig:
    """Authored-DEV numerical/work policy, never video-fitted quality weights."""
    max_steps: int = 20
    budget_seconds: float = 900.
    max_restorations: int = 4
    rotation_trust_rad: float = .05
    translation_trust_m: float = .01
    qp_max_iterations: int = 200
    qp_tolerance: float = 1e-10
    development_reference: str = 'authored_two_chain_feasible_start_DEV_not_HOI_calibration'

    def __post_init__(self):
        ints = [('max_steps',20),('max_restorations',8),('qp_max_iterations',1000)]
        if (any(type(getattr(self,k)) is not int or not 1 <= getattr(self,k) <= cap for k,cap in ints)
            or any(type(getattr(self,k)) not in (float,int) or not np.isfinite(getattr(self,k)) or getattr(self,k) <= 0
                for k in ('budget_seconds','rotation_trust_rad','translation_trust_m','qp_tolerance'))
            or self.budget_seconds > 900 or self.rotation_trust_rad > .5 or self.translation_trust_m > .1
            or self.qp_tolerance > 1e-8 or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Explicit bounded authored-DEV numeric policy required')


class QPProjectionFailure(RuntimeError):
    """Local numerical/cap failure, not nonconvex/physical infeasibility."""


class _BudgetExhausted(RuntimeError):
    """Discard a returned over-budget callback; keep last fully accepted state."""


def _array(value, shape, name):
    if (type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape
        or not np.isfinite(value).all()):
        raise ValueError('Finite exact FP64 callback array required: '+name)
    return value.copy()


def retract_native(parameters, object_translation, step):
    """Gauge-preserving right-SO3/SO2 retraction; fixed blocks byte-identical.

    Raw6D carries column norms/parallel component and rawSO2 its radius. Preserve
    those gauges so tiny tangent updates do not jump the native raw-control prior.
    Decode rotations move on SO3/SO2; zero-tangent blocks retain original bytes.
    """
    n = len(parameters['mhr_trans']); validate_native_parameters(parameters,n,require_shared_identity=True)
    delta = _array(step,(n,STATE_DIM),'native tangent step')
    obj = contact._float_array(object_translation,(n,3),'object translation')
    out = {k:v.copy() for k,v in parameters.items()}; body = out['mhr_body_pose_cont']
    original = parameters['mhr_body_pose_cont']; raw=original[:,:138].astype(float).reshape(n,23,6)
    rotations = _so3(raw); first_norm=np.linalg.norm(raw[...,:3],axis=-1)
    parallel=np.sum(rotations[..., :,0]*raw[...,3:],axis=-1)
    orthogonal=np.linalg.norm(raw[...,3:]-parallel[...,None]*rotations[..., :,0],axis=-1)
    rotations = rotations@Rotation.from_rotvec(delta[:,:69].reshape(-1,3)).as_matrix().reshape(n,23,3,3)
    updated=np.concatenate((first_norm[...,None]*rotations[..., :,0],
        parallel[...,None]*rotations[..., :,0]+orthogonal[...,None]*rotations[..., :,1]),axis=-1).astype(np.float32)
    zero3=np.all(delta[:,:69].reshape(n,23,3)==0,axis=-1)
    updated[zero3]=original[:,:138].reshape(n,23,6)[zero3]
    body[:,:138]=updated.reshape(n,138)
    sincos = original[:,138:254].astype(float).reshape(n,58,2)
    norm=np.linalg.norm(sincos,axis=-1)
    if (norm<=1e-12).any(): raise ValueError('Degenerate native SO2 control')
    angles = np.arctan2(sincos[...,0],sincos[...,1])+delta[:,69:127]
    updated=(norm[...,None]*np.stack((np.sin(angles),np.cos(angles)),axis=-1)).astype(np.float32)
    zero2=delta[:,69:127]==0
    updated[zero2]=original[:,138:254].reshape(n,58,2)[zero2]
    body[:,138:254]=updated.reshape(n,116)
    out['mhr_trans'] = (parameters['mhr_trans'].astype(float)+delta[:,127:130]).astype(np.float32)
    validate_native_parameters(out,n,require_shared_identity=True)
    if (body[:,254:].tobytes() != parameters['mhr_body_pose_cont'][:,254:].tobytes()
        or any(out[k].tobytes() != parameters[k].tobytes() for k in parameters.keys()-{'mhr_trans','mhr_body_pose_cont'})):
        raise ValueError('Native retraction changed frozen identity/hands/root rotation/internal translations')
    return out,obj+delta[:,130:133]


def project_diagonal_qp(gradient, metric, rows, bounds, limits, *, config=PoseSQPConfig()):
    """Sparse per-frame QP; solve <=4 nonnegative dual multipliers, not NxN.

    min .5*d'D*d+gradient'd, rows[t]@d[t]<=bounds[t], |d|<=limits.
    The box-constrained minimizer is analytic for each dual evaluation. Primal
    and dual-gap checks are mandatory, irrespective of solver success flags.
    """
    n = len(gradient); grad = _array(gradient,(n,STATE_DIM),'objective gradient')
    diagonal = _array(metric,(n,STATE_DIM),'positive diagonal metric')
    radius = _array(limits,(n,STATE_DIM),'trust box')
    if (diagonal <= 0).any() or (radius <= 0).any() or len(rows) != n or len(bounds) != n:
        raise ValueError('Positive full-frame metric/trust bounds and complete sparse rows required')
    output = np.empty_like(grad); residuals=[]; gaps=[]; total_iterations=0
    for f in range(n):
        a = rows[f]; b = bounds[f]
        if (type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != STATE_DIM
            or len(a)>4 or not np.isfinite(a).all()): raise ValueError('At most four exact frame-local contact rows required')
        b = _array(b,(len(a),),'linearized contact bounds')
        lam = np.zeros(len(a))
        def dual(multiplier):
            linear = grad[f]+a.T@multiplier
            d = np.clip(-linear/diagonal[f],-radius[f],radius[f])
            value = .5*np.dot(diagonal[f]*d,d)+np.dot(linear,d)-np.dot(multiplier,b)
            return -float(value), b-a@d
        if len(a):
            result = minimize(dual,lam,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(a),
                options=dict(maxiter=config.qp_max_iterations,gtol=config.qp_tolerance*.01,ftol=0.,maxls=40))
            lam=result.x;total_iterations+=int(result.nit)
            # Cost-roundoff may stop L-BFGS before primal feasibility. Polish
            # the <=4x4 active dual system directly while checking free box
            # coordinates; no enormous full-state solve or tolerance increase.
            for _ in range(16):
                linear=grad[f]+a.T@lam;raw=-linear/diagonal[f];d=np.clip(raw,-radius[f],radius[f])
                slope=b-a@d; active=(lam>0)|(slope<0)
                kkt=np.where(lam>0,np.abs(slope),np.maximum(-slope,0))
                if np.max(kkt)<=config.qp_tolerance*.01:break
                free=np.abs(raw)<radius[f]
                h=(a[:,free]/diagonal[f,free])@a[:,free].T
                correction=np.linalg.lstsq(h[np.ix_(active,active)],-slope[active],rcond=None)[0]
                next_lam=lam.copy();next_lam[active]=np.maximum(0.,lam[active]+correction)
                if np.array_equal(next_lam,lam):break
                lam=next_lam
        d = np.clip(-(grad[f]+a.T@lam)/diagonal[f],-radius[f],radius[f])
        violation = max(0.,float(np.max(a@d-b))) if len(a) else 0.
        gap = abs(float(np.dot(lam,b-a@d))) if len(a) else 0.
        if (not np.isfinite(d).all() or not np.isfinite(lam).all() or (lam<0).any()
            or violation > config.qp_tolerance or gap > config.qp_tolerance*(1+abs(dual(lam)[0]))):
            raise QPProjectionFailure(f'Frame {f}: local QP unverified; violation={violation:.3g}, dual_gap={gap:.3g}')
        output[f]=d; residuals.append(violation); gaps.append(gap)
    return output,dict(full_frames=n,contact_rows=sum(len(a) for a in rows),dual_iterations=total_iterations,
        maximum_primal_violation=max(residuals,default=0.),maximum_dual_gap=max(gaps,default=0.),
        diagonal_metric=True,dense_full_clip_matrix_built=False)


def _linearization(callback,parameters,obj,geometry,evidence):
    n = len(obj)
    value = callback({k:v.copy() for k,v in parameters.items()},obj.copy(),geometry,evidence)
    keys = {'gradient','metric_diagonal','witness_jacobian_camera','frame_index','activations','witness_source_indices'}
    if type(value) is not dict or set(value) != keys: raise ValueError('Explicit native tangent Jacobian/objective contract required')
    for k,expected in [('frame_index',evidence.frame_index),('activations',evidence.activations),
                       ('witness_source_indices',evidence.witness_source_indices)]:
        if (type(value[k]) is not np.ndarray or value[k].dtype != expected.dtype or
            value[k].shape != expected.shape or value[k].tobytes() != expected.tobytes()):
            raise ValueError('Native Jacobian cannot change/drop original witness IDs/activity/timeline')
    return (_array(value['gradient'],(n,STATE_DIM),'gradient'),
        _array(value['metric_diagonal'],(n,STATE_DIM),'metric'),
        _array(value['witness_jacobian_camera'],(n,2,3,ROTATION_DIM),'native anatomical witness Jacobian'))


def _contact_rows(surface,evidence,geometry,obj,jac=None):
    n = len(obj); rows=[[] for _ in range(n)]; bounds=[[] for _ in range(n)]
    gaps = np.full(evidence.activations.shape,np.nan); radii=evidence.baseline_gap_m+evidence.config.numerical_slack_m
    for f,s in zip(*np.nonzero(evidence.activations)):
        point = (geometry['human_vertices'][f,evidence.witness_source_indices[f,s]].astype(float)-obj[f])@evidence.object_rotation[f]
        gap,anchor,face = surface.closest(point); gaps[f,s]=gap
        if jac is None: continue
        r = evidence.object_rotation[f]; j = np.c_[jac[f,s],np.eye(3),-np.eye(3)]
        if gap > 1e-12:
            normals = [(point-anchor)/gap]
        else:
            # Unsigned-distance cusp: both face-plane directions, NOT signed
            # penetration normals. Edges/curvature still need exact restoration.
            triangle = evidence.object_vertices[evidence.object_faces[face]]
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]); norm=np.linalg.norm(normal)
            if norm<=1e-15: raise QPProjectionFailure('Degenerate zero-gap contact normal; exact point stays preserved')
            normals=[normal/norm,-normal/norm]
        for normal in normals:
            # Aim INSIDE the unchanged exact bound by one decoder-coordinate
            # rounding reserve. This is NOT contact tolerance relaxation.
            # FP32 camera vertices may round against FP64 object translations;
            # use half-ULP norm conservatively, capped by existing numeric slack.
            ulp=np.abs(np.spacing(geometry['human_vertices'][f,evidence.witness_source_indices[f,s]]))
            reserve=min(evidence.config.numerical_slack_m,float(.5*np.linalg.norm(ulp.astype(float)))*np.linalg.norm(r,2))
            target=max(evidence.baseline_gap_m[f,s],radii[f,s]-reserve)
            rows[f].append(normal@r.T@j); bounds[f].append(target-gap)
    return gaps,[np.asarray(a,np.float64).reshape(-1,STATE_DIM) for a in rows], [np.asarray(b,np.float64) for b in bounds]


def _objective(callback,p,obj,g):
    value = callback({k:v.copy() for k,v in p.items()},obj.copy(),g)
    if type(value) not in (int,float) or not np.isfinite(value) or value<0:
        raise ValueError('Actual finite nonnegative original objective required')
    return float(value)


def optimize_native_pose(parameters,object_translation,evidence,*,decode_native,linearize_native,
        evaluate_objective,evaluate_observations,gates,config=PoseSQPConfig()):
    """Feasible A -> projected descent/restoration, full native decode callbacks.

    linearize_native(p,obj,g,evidence) returns FP64 full133 gradient/positive
    proximal diagonal and [T,2,3,127] camera-space native witness Jacobians w.r.t.
    RIGHT-SO3/SO2 tangents. IDs/activity/frame arrays must equal frozen evidence.
    The objective/gradient must contain training evidence only; independent QA
    is an acceptance gate, NOT a training target or claimed held-out estimate.
    """
    if (type(config) is not PoseSQPConfig or not isinstance(evidence,contact.ContactFeasibilityEvidence)
        or any(not callable(v) for v in (decode_native,linearize_native,evaluate_objective,evaluate_observations))
        or type(gates) not in (tuple,list) or not gates or any(not isinstance(g,ObservationGate) for g in gates)
        or len({g.name for g in gates}) != len(gates)):
        raise ValueError('Explicit native decode/Jacobian/objective and independent frozen QA required')
    started=time.monotonic(); n=len(evidence.frame_index); validate_native_parameters(parameters,n,require_shared_identity=True)
    if not np.array_equal(evidence.frame_index,np.arange(n)): raise ValueError('Every original frame required')
    p={k:v.copy() for k,v in parameters.items()}; obj=contact._float_array(object_translation,(n,3),'original object T')
    surface=contact._NearestSurface(evidence.object_vertices,evidence.object_faces); decode_calls=0
    def check_budget():
        if time.monotonic()-started>=config.budget_seconds: raise _BudgetExhausted('Declared pose SQP callback budget exhausted')
    def callback(function,*args):
        check_budget();value=function(*args);check_budget();return value
    def decode(candidate):
        nonlocal decode_calls
        check_budget()
        decode_calls+=1
        return _decode_geometry(callback(decode_native,{k:v.copy() for k,v in candidate.items()}),n,
            geometry['human_faces'] if decode_calls>1 else None,geometry['scales'] if decode_calls>1 else None)
    geometry=decode(p); gaps,_,_=_contact_rows(surface,evidence,geometry,obj)
    radii=evidence.baseline_gap_m+evidence.config.numerical_slack_m
    if not np.all(gaps[evidence.activations]<=radii[evidence.activations]) or not _object_in_front(evidence,obj):
        raise BaselineReferenceFailure(evidence,gaps,_object_in_front(evidence,obj))
    initial_metrics=callback(evaluate_observations,geometry,obj.copy())
    if not _observation_acceptance(initial_metrics,initial_metrics,gates)[0]: raise ValueError('Baseline QA unavailable')
    value=callback(_objective,evaluate_objective,p,obj,geometry); initial_value=value
    metrics=initial_metrics; attempts=[]; accepted=0; trust=1.; exhausted=False
    limits=np.tile(np.r_[np.full(ROTATION_DIM,config.rotation_trust_rad),np.full(6,config.translation_trust_m)],(n,1))
    for iteration in range(config.max_steps):
        if time.monotonic()-started>=config.budget_seconds: exhausted=True;break
        row=dict(iteration=iteration,status='rejected',trust_fraction=trust); attempts.append(row)
        try:
            grad,metric,jac=callback(_linearization,linearize_native,p,obj,geometry,evidence)
            _,a,b=_contact_rows(surface,evidence,geometry,obj,jac)
            check_budget();step,qp=project_diagonal_qp(grad,metric,a,b,limits*trust,config=config);check_budget();row['QP']=qp
            predicted=-float((grad*step+.5*metric*step*step).sum());row['predicted_decrease']=predicted
            if predicted<=1e-12*(1+value): row['status']='no_verified_local_descent';break
            proposed,ot=retract_native(p,obj,step); current=decode(proposed)
            current_gaps,_,_=_contact_rows(surface,evidence,current,ot); restores=0
            while (not np.all(current_gaps[evidence.activations]<=radii[evidence.activations]) and restores<config.max_restorations):
                if time.monotonic()-started>=config.budget_seconds: exhausted=True;break
                _,d,j=callback(_linearization,linearize_native,proposed,ot,current,evidence)
                _,a,b=_contact_rows(surface,evidence,current,ot,j)
                check_budget();correction,_=project_diagonal_qp(np.zeros_like(grad),d,a,b,limits*trust,config=config);check_budget()
                proposed,ot=retract_native(proposed,ot,correction);current=decode(proposed)
                current_gaps,_,_=_contact_rows(surface,evidence,current,ot);restores+=1
            row.update(restoration_steps=restores,maximum_contact_violation_m=max(0.,float(
                (current_gaps-radii)[evidence.activations].max())) if evidence.activations.any() else 0.)
            if exhausted: row['status']='budget_exhausted';break
            if row['maximum_contact_violation_m']>0 or not _object_in_front(evidence,ot):
                row['status']='exact_contact_or_depth_rejected';trust*=.5;continue
            candidate_value=callback(_objective,evaluate_objective,proposed,ot,current)
            row['actual_objective']=candidate_value
            if candidate_value>=value-1e-12*(1+value): row['status']='objective_rejected';trust*=.5;continue
            candidate_metrics=callback(evaluate_observations,current,ot.copy())
            passed,checks=_observation_acceptance(initial_metrics,candidate_metrics,gates);row['QA_gates']=checks
            if not passed: row['status']='QA_rejected';trust*=.5;continue
            ratio=(value-candidate_value)/predicted;row.update(status='accepted',actual_predicted_ratio=ratio)
            p,obj,geometry,gaps,value,metrics=proposed,ot,current,current_gaps,candidate_value,candidate_metrics;accepted+=1
            trust=min(1.,trust*2) if ratio>.75 else trust*.5 if ratio<.25 else trust
        except QPProjectionFailure as exc:
            row.update(status='local_QP_unverified',reason=str(exc)[:240],physical_infeasibility_claimed=False);trust*=.5
        except _BudgetExhausted:
            row['status']='budget_exhausted_callback_result_discarded';exhausted=True;break
    return dict(schema='world_reward.native_pose_sqp.v1',
        status='accepted_native_pose_sqp' if accepted else 'dynamic_A_fallback_no_improvement',
        parameters=p,object_translation=obj,geometry=geometry,witness_gaps_m=gaps,
        original_activations=evidence.activations,original_witness_ids=evidence.witness_source_indices,
        objective_before=initial_value,objective_after=value,observation_metrics=metrics,
        baseline_observation_metrics=initial_metrics,accepted_steps=accepted,attempts=attempts,
        native_decode_calls=decode_calls,elapsed_seconds=time.monotonic()-started,budget_exhausted=exhausted,
        config=asdict(config),full_original_frames=n,diagonal_proximal_QP_not_full_Gauss_Newton=True,
        native_callback_provenance_independently_verified=False,native_direct_replay_independently_verified=False,
        ground_truth_used=False,private_truth_read=False,whole_hand_minimum_claimed=False,
        physical_contact_verified=False,penetration_evaluated=False,production_adopted=False,
        heldout_accuracy_verified=False,baseline_fallback_dynamic=True)

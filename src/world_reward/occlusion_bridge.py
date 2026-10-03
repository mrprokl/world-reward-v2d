"""Observed-anchor, hand-conditioned SE(3) initialization for missing poses.

This is an initializer, not a measured trajectory or contact classifier. Inputs
are already measured/predicted camera-frame poses; RGB, geometry, cameras, scale,
truth and evaluation alignment are not accepted. Leading/trailing unknown runs
abstain. Nothing here converts an initialization into a submission prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ANCHORS_PER_SIDE=3
MAX_TRANSLATION_DISPERSION_M=.03
MAX_ROTATION_DISPERSION_RAD=.15


@dataclass(frozen=True)
class BridgeInitialization:
    rotations: np.ndarray
    translations: np.ndarray
    observed: np.ndarray
    inferred: np.ndarray
    valid: np.ndarray
    gaps: tuple[dict,...]

    @property
    def complete(self):
        return bool(self.valid.all())

    def require_complete(self):
        if not self.complete:
            raise ValueError('Unobserved leading/trailing interval: bilateral evidence required; no static extrapolation')
        return self


def _poses(rotations,translations,valid,name,count):
    if np.ma.isMaskedArray(rotations)or np.ma.isMaskedArray(translations):
        raise ValueError(name+': explicit camera-frame poses required')
    r,t=np.asarray(rotations),np.asarray(translations)
    if r.shape!=(count,3,3)or t.shape!=(count,3)or r.dtype.kind!='f'or t.dtype.kind!='f':
        raise ValueError(name+': floating [T,3,3]/[T,3] arrays required')
    if not np.isfinite(r[valid]).all()or not np.isfinite(t[valid]).all():
        raise ValueError(name+': finite observed poses required')
    if not np.isnan(r[~valid]).all()or not np.isnan(t[~valid]).all():
        raise ValueError(name+': missing poses must be explicit NaN, never filled observations')
    if(not np.allclose(r[valid]@r[valid].transpose(0,2,1),np.eye(3),atol=1e-6,rtol=0)
       or not np.allclose(np.linalg.det(r[valid]),1,atol=1e-6,rtol=0)):
        raise ValueError(name+': proper SO(3), never reflection or scale, required')
    return r.copy(),t.copy()


def _interpolate(rotations,times):
    return Slerp([0.,1.],Rotation.from_matrix(rotations))(times).as_matrix()


def _relative_summary(object_r,object_t,hand_r,hand_t,indices):
    h=hand_r[indices];o=object_r[indices]
    relative_r=h.transpose(0,2,1)@o
    relative_t=np.einsum('tij,tj->ti',h.transpose(0,2,1),object_t[indices]-hand_t[indices])
    u,_,vt=np.linalg.svd(relative_r.mean(0))
    correction=np.eye(3);correction[2,2]=1 if np.linalg.det(u@vt)>=0 else-1
    mean_r=u@correction@vt;mean_t=relative_t.mean(0)
    translation_dispersion=float(np.sqrt(np.mean(np.sum((relative_t-mean_t)**2,axis=1))))
    angles=Rotation.from_matrix(mean_r.T@relative_r).magnitude()
    rotation_dispersion=float(np.sqrt(np.mean(angles**2)))
    return mean_r,mean_t,translation_dispersion,rotation_dispersion


def initialize_occluded_gaps(frame_indices,object_rotations,object_translations,observed,
                              hand_rotations,hand_translations,hand_valid):
    """Initialize every bilateral gap; leave observed pose bytes untouched.

    Three consecutive observed anchors on EACH side estimate hand→object SE(3).
    RMS relative-translation dispersion <=3cm and RMS relative-rotation dispersion
    <=.15rad permit a hand-conditioned proposal, not an assertion of contact.
    Hand poses must be valid through the entire gap and all anchors. Slip/regrasp
    instability or missing hand evidence selects explicit endpoint interpolation.
    Smooth endpoint residual correction guarantees the same two observed anchors.
    There is no tunable threshold, causal last-pose/static extrapolation, GT input,
    candidate deletion, geometry update, camera change or scale fitting.
    """
    frames=np.asarray(frame_indices);flags=np.asarray(observed);hands=np.asarray(hand_valid)
    if(frames.ndim!=1 or not len(frames)or frames.dtype.kind not in'iu'
       or not np.array_equal(frames,np.arange(len(frames)))):
        raise ValueError('Complete original contiguous frame indices starting zero required')
    count=len(frames)
    if flags.shape!=(count,)or flags.dtype!=np.bool_ or hands.shape!=(count,)or hands.dtype!=np.bool_:
        raise ValueError('Explicit original boolean observation/hand validity required')
    object_r,object_t=_poses(object_rotations,object_translations,flags,'object',count)
    hand_r,hand_t=_poses(hand_rotations,hand_translations,hands,'hand',count)
    output_r,output_t=object_r.copy(),object_t.copy();inferred=np.zeros(count,bool)
    diagnostics=[];index=0
    while index<count:
        if flags[index]:index+=1;continue
        first=index
        while index<count and not flags[index]:index+=1
        last=index-1;left=first-1;right=index
        row=dict(first_frame_index=first,last_frame_index=last,frames=last-first+1,
                 observed=False,inferred=True,contact_asserted=False,
                 translation_dispersion_m=None,rotation_dispersion_rad=None)
        if left<0 or right>=count:
            row.update(method='abstain',reason='no_bilateral_observed_endpoints',inferred=False)
            diagnostics.append(row);continue
        times=np.arange(first,right);alpha=(times-left)/(right-left)
        reason='insufficient_three_consecutive_anchors_each_side'
        anchors=np.r_[np.arange(first-ANCHORS_PER_SIDE,first),np.arange(right,right+ANCHORS_PER_SIDE)]
        supported=(anchors.min()>=0 and anchors.max()<count and bool(flags[anchors].all()))
        conditioned=False
        if supported:
            if not hands[anchors].all()or not hands[times].all():reason='missing_hand_evidence'
            else:
                mean_r,mean_t,td,rd=_relative_summary(object_r,object_t,hand_r,hand_t,anchors)
                row.update(translation_dispersion_m=td,rotation_dispersion_rad=rd)
                conditioned=td<=MAX_TRANSLATION_DISPERSION_M and rd<=MAX_ROTATION_DISPERSION_RAD
                reason='stable_relative_pose_not_proven_contact'if conditioned else'unstable_relative_pose_slip_regrasp_or_noise'
        if conditioned:
            all_indices=np.r_[left,times,right]
            predicted_r=hand_r[all_indices]@mean_r
            predicted_t=np.einsum('tij,j->ti',hand_r[all_indices],mean_t)+hand_t[all_indices]
            correction_r=np.stack((object_r[left]@predicted_r[0].T,object_r[right]@predicted_r[-1].T))
            output_r[times]=_interpolate(correction_r,alpha)@predicted_r[1:-1]
            delta_left=object_t[left]-predicted_t[0];delta_right=object_t[right]-predicted_t[-1]
            output_t[times]=predicted_t[1:-1]+(1-alpha[:,None])*delta_left+alpha[:,None]*delta_right
            row['method']='hand_conditioned_endpoint_constrained'
        else:
            output_r[times]=_interpolate(object_r[[left,right]],alpha)
            output_t[times]=(1-alpha[:,None])*object_t[left]+alpha[:,None]*object_t[right]
            row['method']='free_endpoint_interpolation'
        row['reason']=reason;inferred[times]=True;diagnostics.append(row)
    return BridgeInitialization(output_r,output_t,flags.copy(),inferred,flags|inferred,tuple(diagnostics))

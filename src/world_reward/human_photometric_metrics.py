"""Frozen human-only paired diagnostic gates, never inference or selection.

Modes are baseline, exact repeated-input SHAM, gamma-medoid. All eight groups
and their three poses count equally. A gain under a shared baseline-derived
alignment is not metric grounding, interaction quality or a challenge score.
"""
import numpy as np

MODES=("baseline","sham","tta")
RULE=dict(groups=8,frames_per_group=3,median_relative_gain=.05,maximum_group_regression=.05,
    maximum_hand_regression=.05,maximum_frame_iou_drop=.01,
    alignment="one proper positive full18439 baseline-firsthuman Sim3 per group, shared all methods and poses",
    silhouette="raw camera mesh against unchanged automatic human mask, no alignment",
    weighting="equal3 poses within each group; paired8-group median gain; no confidence or visibility subsets")


def numeric(value,shape,name):
    if(type(value)is not np.ndarray or value.shape!=shape or value.dtype not in(np.float32,np.float64)
       or not np.isfinite(value).all()):raise ValueError("Complete finite array required: "+name)
    return value.astype(np.float64)


def geometry_diagnostic(prediction,truth):
    """Full correspondence camera error and centroid/centered decomposition."""
    p=numeric(prediction,(18439,3),"prediction");q=numeric(truth,p.shape,"truth")
    delta=p-q;bias=delta.mean(0);centered=delta-bias
    rms2=float(np.mean(np.sum(delta*delta,axis=1)));centered2=float(np.mean(np.sum(centered*centered,axis=1)))
    if not np.isclose(rms2,np.dot(bias,bias)+centered2,atol=1e-12,rtol=1e-12):raise ValueError("Camera RMS decomposition failed")
    return dict(pve_cm=float(np.linalg.norm(delta,axis=1).mean()*100),rms_cm=float(np.sqrt(rms2)*100),
        centroid_error_xyz_cm=(bias*100).tolist(),centered_pve_cm=float(np.linalg.norm(centered,axis=1).mean()*100))


def relative_gains(before,after):
    """Undefined at zero baseline: retain cases and reject the relative gate."""
    if np.any(before<0)or np.any(after<0):raise ValueError("Nonnegative errors required")
    defined=bool(np.all(before>0))
    gains=(before-after)/before if defined else None
    if defined and not np.isfinite(gains).all():raise ValueError("Finite relative gain arithmetic required")
    return defined,gains


def quality_decision(group_pve,frame_iou,group_hand_pve):
    """All8group means, all24frame silhouettes, both semantic hands retained."""
    pve=numeric(group_pve,(8,3),"group/mode aligned human PVE")
    iou=numeric(frame_iou,(24,3),"frame/mode raw camera IoU")
    hands=numeric(group_hand_pve,(8,3,2),"group/mode/hand aligned PVE")
    if np.any(pve<0)or np.any(hands<0)or np.any(iou<0)or np.any(iou>1):raise ValueError("Nonnegative errors and unit IoU required")
    if not all(np.array_equal(value[:,0],value[:,1])for value in(pve,iou,hands)):
        raise ValueError("SHAM metrics must exactly equal frozen baseline, not a numeric tolerance")
    defined,gains=relative_gains(pve[:,0],pve[:,2]);hand_limit=hands[:,0]*1.05
    if not np.isfinite(hand_limit).all():raise ValueError("Finite hand safeguards required")
    drop=iou[:,2]-iou[:,0]
    gates=dict(median_paired_group_aligned_pve_gain_5pct=defined and float(np.median(gains))>=.05,
        no_group_aligned_pve_regression_over_5pct=defined and bool(np.all(pve[:,2]<=pve[:,0]*1.05)),
        no_group_per_hand_aligned_pve_regression_over_5pct=bool(np.all(hands[:,2]<=hand_limit)),
        no_original_frame_automatic_human_iou_drop_over_1pp=bool(np.all(iou[:,2]>=iou[:,0]-.01)))
    return dict(rule=RULE,modes=list(MODES),gates=gates,synthetic_human_photometric_hypothesis_supported=all(gates.values()),
        per_group_aligned_human_relative_gain=gains.tolist()if defined else None,
        median_paired_group_aligned_human_relative_gain=float(np.median(gains))if defined else None,
        zero_baseline_relative_gain_undefined=not defined,per_frame_iou_change=drop.tolist(),
        maximum_per_hand_absolute_regression_cm=float(np.max(hands[:,2]-hands[:,0])),
        SHAM_metrics_exact=True,adoption_authorized=False,full_HOI_verified=False,metric_gauge_verified=False,
        real_domain_verified=False,challenge_performance_verified=False)


def second_difference_diagnostic(joints):
    """Three real poses per group: descriptive error-free motion, not a gate."""
    value=numeric(joints,(8,3,127,3),"group/frame native joints")
    acceleration=value[:,2]-2*value[:,1]+value[:,0]
    return np.linalg.norm(acceleration,axis=-1).mean(1)*100

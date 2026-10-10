"""Bounded saved-only whole-anatomy distance intervals, no exact full search.

Every original anatomical point enters vectorized conservative bounds. An
interval is explicitly NOT an exact gap, and overlapping intervals are
inconclusive. No contact threshold, fitting, observation deletion or new model.
"""
import os
from pathlib import Path
import time

import numpy as np

import contact_real_geometry_audit as old_audit
from mediapipe_cpu_runtime_verify import source
from world_reward.sequence_pose import _ContactTriangleSurface

real=old_audit.real
ROOT=real.ROOT
ENTRY='run_contact_real_interval_audit'
HELPERS=tuple(dict.fromkeys(('infra/contact_real_interval_audit.py',
    'infra/run_contact_real_interval_audit.sh',*old_audit.HELPERS)))
POINT_CHUNK=1024


def whole_hand_interval(surface,points,*,tighten=True):
    """All points -> centre-ball and mesh-AABB lower / referenced-vertex upper."""
    points=np.asarray(points)
    if (points.dtype.kind!='f' or points.ndim!=2 or points.shape[1:]!=(3,)
            or not len(points) or not np.isfinite(points).all()
            or surface._vertex_tree is None):
        raise ValueError('Complete finite anatomy and bounded original mesh required')
    starts=surface.starts.reshape(-1,3);lo=starts.min(axis=0);hi=starts.max(axis=0)
    lower=np.inf;upper=np.inf;winner=-1;maximum_guard=0.;processed=0
    for begin in range(0,len(points),POINT_CHUNK):
        p=points[begin:begin+POINT_CHUNK]
        d=surface._vertex_tree.query(p,k=1,eps=0)[0]
        rounding=old_audit.guard(surface,p,d)
        if not np.isfinite(rounding).all():
            raise ValueError('Finite mesh-bounds FP64 guard required')
        delta=np.maximum(np.maximum(lo-p,0),p-hi)
        bbox=np.linalg.norm(delta,axis=1)
        bbox=np.maximum(0.,np.nextafter(bbox-rounding,-np.inf))
        centre=old_audit.lower_bounds(surface,p)
        lower=min(lower,float(np.maximum(bbox,centre).min()))
        safe_upper=np.nextafter(d+rounding,np.inf)
        j=int(np.argmin(safe_upper))
        if safe_upper[j]<upper:
            upper=float(safe_upper[j]);winner=begin+j
        maximum_guard=max(maximum_guard,float(rounding.max()));processed+=len(p)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower>upper:
        raise ValueError('Whole-hand interval certification failed')
    vertex_upper=upper
    if tighten:
        exact_witness=float(surface.distances(points[winner:winner+1],batch_size=32)[0])
        if not np.isfinite(exact_witness):raise ValueError('Finite original-surface witness distance required')
        # One actual anatomical point's exact continuous distance bounds the
        # whole hand's minimum from above; never assert that point is globally best.
        upper=min(upper,float(np.nextafter(exact_witness+maximum_guard,np.inf)))
    if lower>upper:raise ValueError('Tightened whole-hand interval certification failed')
    return dict(lower_m=lower,upper_m=upper,width_m=upper-lower,
        nearest_vertex_witness_anatomical_index=winner,points_total=len(points),
        points_bound_evaluated=processed,full_anatomy_and_surface_retained=True,
        numerical_guard_m=maximum_guard,lower_bound='max(AABB,centre_ball) over every anatomical point',
        upper_bound=('one_exact_continuous_surface_witness' if tighten else 'nearest_referenced_mesh_vertex'),
        vertex_upper_m=vertex_upper,exact_witness_points=1 if tighten else 0,
        exact_surface_minimum_claimed=False)


def strict_interval_comparison(candidate,original):
    epsilon=max(candidate['numerical_guard_m'],original['numerical_guard_m'])
    if candidate['lower_m']>original['upper_m']+epsilon:
        outcome='certified_larger_gap'
    elif candidate['upper_m']<original['lower_m']-epsilon:
        outcome='certified_smaller_gap'
    else:outcome='inconclusive_overlapping_intervals'
    return dict(outcome=outcome,comparison_epsilon_m=epsilon,
        ground_truth_accuracy_claimed=False,physical_contact_claimed=False)


def variant_intervals(bank,r,t):
    surface=_ContactTriangleSurface(bank['vertices'],bank['faces']);rows=[]
    for frame,side in zip(*np.nonzero(bank['evidence'].activations)):
        local=(bank['hands'][frame,side]-t[frame])@r[frame]
        rows.append(dict(frame_index=int(frame),hand=int(side),**whole_hand_interval(surface,local)))
    return dict(rows=rows,lower_m=old_audit.summary([row['lower_m'] for row in rows]),
        upper_m=old_audit.summary([row['upper_m'] for row in rows]),
        width_m=old_audit.summary([row['width_m'] for row in rows]),
        points_total=sum(row['points_total'] for row in rows),
        points_bound_evaluated=sum(row['points_bound_evaluated'] for row in rows),
        full_active_hand_rows=len(rows),exact_surface_minimum_claimed=False)


def coverage_cost_receipt(bank,r,t,pool,cfg,fit_cfg,name,intervals):
    """Cheap selected-pool exact geometry, never exhaustive full-hand search."""
    surface=_ContactTriangleSurface(bank['vertices'],bank['faces']);coverage=[]
    for row in intervals['rows']:
        frame,side=row['frame_index'],row['hand']
        local=(pool.hand_points_camera[frame,side,pool.hand_visible[frame,side]]-t[frame])@r[frame]
        pool_gap=float(surface.distances(local,batch_size=32).min())
        epsilon=row['numerical_guard_m']
        coverage.append(dict(frame_index=frame,hand=side,pool_surface_gap_m=pool_gap,
            whole_hand_upper_m=row['upper_m'],comparison_epsilon_m=epsilon,
            certified_better_proximity_outside_selected_pool=bool(row['upper_m']<pool_gap-epsilon),
            uncertified_pool_coverage=bool(row['upper_m']>=pool_gap-epsilon)))
    factor=old_audit.factor_costs(bank,r,t,pool,cfg,fit_cfg,name if name!='original' else 'J1')
    return dict(pool_coverage_rows=coverage,
        certified_better_proximity_outside_pool=sum(row['certified_better_proximity_outside_selected_pool'] for row in coverage),
        uncertified_pool_coverage=sum(row['uncertified_pool_coverage'] for row in coverage),
        factor_costs=factor,scope='prediction_geometry_proximity_not_physical_contact_truth')


def run():
    revision=os.environ['WR_CODE_REVISION'];code=Path(os.environ['WR_CODE']);started=time.monotonic()
    if code!=ROOT/'jobs'/revision/ENTRY/'code':raise ValueError('Exact immutable Azure audit source required')
    out=ROOT/'results'/('contact-real-interval-audit-'+revision);real.old.fresh_runtime_output(out)
    ledger=real.old.ArtifactLedger();report=dict(status='fail',producer_revision=revision,
        new_fits=0,model_calls=0,production_adopted=False,ground_truth_used=False,
        heldout_accuracy_verified=False,exact_surface_minimum_claimed=False,variants=[])
    try:
        binding=source(ROOT,code,revision,ENTRY,HELPERS)
        for name in HELPERS:ledger.record(code/name)
        bank=real.load_bank(ledger);variants,cfg,fit_cfg=old_audit.load_candidates(code,ledger,bank)
        pool,_=real.bounded_pool(bank,cfg);deadline=started+120;original=None
        for name,(r,t) in variants.items():
            if time.monotonic()>deadline:raise TimeoutError('Interval diagnostic budget exhausted')
            intervals=variant_intervals(bank,r,t)
            if original is None:original=intervals
            compared=[dict(frame_index=row['frame_index'],hand=row['hand'],
                **strict_interval_comparison(row,reference))
                for row,reference in zip(intervals['rows'],original['rows'])]
            receipt=dict(variant=name,**intervals,compared_to_original=compared,
                producer_revision=revision,ground_truth_used=False,production_adopted=False,
                factor_costs_completed=False)
            receipt_pin=real.seal_json(out/(name+'_intervals.json'),receipt)
            report['variants'].append(dict(variant=name,interval_receipt=receipt_pin,
                outcomes={label:sum(row['outcome']==label for row in compared) for label in
                    ('certified_larger_gap','certified_smaller_gap','inconclusive_overlapping_intervals')}))
            if time.monotonic()>deadline:raise TimeoutError('Interval receipt sealed; factor diagnostic budget exhausted')
            # Geometry intervals survive a later expensive selected-pool/cost failure.
            costs=coverage_cost_receipt(bank,r,t,pool,cfg,fit_cfg,name,intervals)
            report['variants'][-1]['factor_receipt']=real.seal_json(out/(name+'_factors.json'),
                dict(variant=name,producer_revision=revision,**costs))
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS)!=binding:raise ValueError('Own closure changed')
        report.update(status='complete_saved_proximity_interval_diagnostic',sources=ledger.records,
            source_binding=binding,no_new_contact_thresholds=True)
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc)[:300])
    report['elapsed_seconds']=time.monotonic()-started
    real.seal_json(out/'report.json',report)
    import json
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','variants')},allow_nan=False),flush=True)
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':
    if len(os.sys.argv)!=1:raise SystemExit('No arbitrary inputs supported')
    run()

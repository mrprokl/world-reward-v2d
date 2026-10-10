"""Saved-only anatomy coverage / factor-cost diagnostic, never a new fitter.

All 2318 points of each originally active hand participate in a certified
minimum search. Conservative lower bounds skip only points that cannot win;
original faces and anatomical trajectories remain untouched. Normals are not
guessed: positive-gap direction is a distance gradient, not a surface normal.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time

import numpy as np

import sequence_contact_patch_real as real
from world_reward.sequence_pose import _ContactTriangleSurface
from mediapipe_cpu_runtime_verify import source, strict

ROOT=real.ROOT
ENTRY='run_contact_real_geometry_audit'
PIN_CONFIG='configs/contact_real_geometry_audit_pins.json'
HELPERS=tuple(dict.fromkeys(('infra/contact_real_geometry_audit.py',
    'infra/run_contact_real_geometry_audit.sh',PIN_CONFIG,*real.HELPERS)))


def guard(surface,points,values):
    magnitude=np.maximum(np.max(np.abs(points),axis=1),surface._coordinate_magnitude)
    magnitude=np.maximum(magnitude,np.maximum(np.abs(values),surface._maximum_radius))
    return 128*np.finfo(float).eps*np.maximum(magnitude,np.finfo(float).tiny)


def lower_bounds(surface,points):
    """d(point,surface) >= nearest triangle centre distance - maximum radius."""
    if surface._centre_tree is None or not np.isfinite(surface._maximum_radius):
        return np.zeros(len(points))
    centre_distance=surface._centre_tree.query(points,k=1,eps=0)[0]
    bounds=centre_distance-surface._maximum_radius-guard(surface,points,centre_distance)
    if not np.isfinite(bounds).all():return np.zeros(len(points))
    return np.maximum(0.,np.nextafter(bounds,-np.inf))


def whole_hand_minimum(surface,points,*,deadline=None):
    """Exact continuous minimum over ALL supplied points, with safe pruning."""
    p=np.asarray(points)
    if p.dtype.kind!='f' or p.ndim!=2 or p.shape[1:]!=(3,) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite complete anatomical point set required')
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Whole-hand geometry budget exhausted')
    bounds=lower_bounds(surface,p)
    if surface._vertex_tree is not None:
        upper=surface._vertex_tree.query(p,k=1,eps=0)[0]
        first=int(np.argmin(upper))
    else:first=int(np.argmin(bounds))
    best=float(surface.distances(p[first:first+1],batch_size=32)[0]);winner=first;evaluated=1
    # Stable index tie handling is diagnostic only, not a contact assignment.
    order=np.argsort(bounds,kind='stable');pending=order[order!=first];position=0
    while position<len(pending):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Whole-hand geometry budget exhausted')
        next_index=pending[position]
        tolerance=guard(surface,p[next_index:next_index+1],np.array([best]))[0]
        if bounds[next_index]>np.nextafter(best+tolerance,np.inf):break
        ids=pending[position:position+32]
        tolerance=guard(surface,p[ids],np.full(len(ids),best))
        ids=ids[bounds[ids]<=np.nextafter(best+tolerance,np.inf)]
        if len(ids):
            distances=surface.distances(p[ids],batch_size=32);evaluated+=len(ids)
            j=int(np.argmin(distances));value=float(distances[j])
            if value<best or (value==best and int(ids[j])<winner):best=value;winner=int(ids[j])
        position+=32
    return dict(distance_m=best,point_index=winner,points_total=len(p),
        points_exactly_evaluated=evaluated,points_proven_nonwinning=len(p)-evaluated,
        bound='nearest_triangle_centre_minus_global_radius_with_FP64_guard',
        full_anatomy_and_surface_retained=True)


def ray_gradient_alignment(surface,point,ray):
    """Distance finite-difference gradient/ray, NOT a guessed triangle normal."""
    p=np.asarray(point,float);ray=np.asarray(ray,float);ray=ray/np.linalg.norm(ray)
    step=np.cbrt(np.finfo(float).eps)*max(surface._coordinate_magnitude,float(np.abs(p).max()),.001)
    delta=np.eye(3)*step
    values=surface.distances(np.concatenate((p[None]+delta,p[None]-delta)),batch_size=32)
    gradient=(values[:3]-values[3:])/(2*step)
    norm=float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm<1e-8:return dict(distance_gradient_norm=norm,absolute_ray_alignment=None)
    return dict(distance_gradient_norm=norm,absolute_ray_alignment=float(abs(gradient@ray)/norm),
        scope='local_unsigned_distance_gradient_not_oriented_surface_normal')


def summary(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None)


def robust_stats(values):
    a=np.asarray(values,float)
    return dict(count=len(a),cost=float(np.sum(np.sqrt(1+a*a)-1)),normalized_absolute=summary(abs(a)),
        influence_weight=summary(1/np.sqrt(1+a*a)))


def factor_costs(bank,r,t,pool,cfg,fit_cfg,variant):
    """Rebuild unchanged dimensionless factor energies; no optimize/reweight."""
    v,p=bank['vertices'],bank['points'];n=len(t);fps=bank['fps'];centre=v.mean(0)
    diameter=2*np.sqrt(np.mean(np.sum((v-centre)**2,axis=1)))
    visible=bank['visible'];frames,queries=np.nonzero(visible&(np.arange(n)[:,None]>0))
    xyz=np.einsum('nij,nj->ni',r[frames],p[queries])+t[frames]
    predicted=xyz[:,:2]/xyz[:,2,None]*bank['K'].diagonal()[:2]+bank['K'][:2,2]
    weights=np.sqrt((n-1)/visible[1:].sum(0)[queries])
    rgb=(predicted-bank['xy'][frames,queries])/fit_cfg.pixel_sigma*weights[:,None]
    centroid=np.einsum('tij,j->ti',r,centre)+t
    original=np.einsum('tij,j->ti',bank['rotations'],centre)+bank['translations']
    prior_c=np.sqrt(fit_cfg.prior_weight)*(centroid[1:]-original[1:])/(diameter*fit_cfg.prior_centroid_sigma_diameter)
    from scipy.spatial.transform import Rotation
    prior_r=np.sqrt(fit_cfg.prior_weight)*Rotation.from_matrix(r[1:]@bank['rotations'][1:].swapaxes(-1,-2)).as_rotvec()/fit_cfg.prior_rotation_sigma_rad
    accel=np.sqrt(fit_cfg.temporal_weight)*np.diff(centroid,n=2,axis=0)*fps**2/(diameter*fit_cfg.acceleration_sigma_diameter_s2)
    spin=Rotation.from_matrix(r[1:]@r[:-1].swapaxes(-1,-2)).as_rotvec()*fps
    angular=np.sqrt(fit_cfg.temporal_weight)*np.diff(spin,axis=0)*fps/fit_cfg.angular_acceleration_sigma_rad_s2
    surface=_ContactTriangleSurface(v,bank['faces']);evidence=bank['evidence'] if variant=='J1' else pool
    distances=[];nearest=[]
    for frame,side in zip(*np.nonzero(evidence.activations&(np.arange(n)[:,None]>0))):
        indices=(np.array([0]) if variant=='J1' else np.arange(evidence.hand_points_camera.shape[2]))
        indices=indices[evidence.hand_visible[frame,side,indices]]
        local=(evidence.hand_points_camera[frame,side,indices]-t[frame])@r[frame]
        d=surface.distances(local,batch_size=32);nearest.append(float(d.min()))
        if variant=='soft_pool':
            from world_reward.contact_patch import smooth_patch_distances
            distances.append(float(smooth_patch_distances(d,np.zeros(len(d),np.int64),1,float(diameter*cfg['temperature_diameter']))[0]))
        else:distances.append(float(d.min()))
    contact=np.asarray(distances)/(diameter*cfg['contact_sigma_diameter'])
    blocks={key:robust_stats(a.ravel()) for key,a in dict(RGB=rgb,prior_centroid=prior_c,
        prior_rotation=prior_r,centroid_acceleration=accel,angular_acceleration=angular,contact=contact).items()}
    return dict(diameter_m=float(diameter),contact_scale_m=float(diameter*cfg['contact_sigma_diameter']),
        blocks=blocks,total_cost=sum(row['cost'] for row in blocks.values()),
        contact_gap_m=summary(nearest),scope='identical_factors_prediction_diagnostic_not_sensor_error')


def variant_audit(name,bank,r,t,pool,cfg,fit_cfg,deadline):
    surface=_ContactTriangleSurface(bank['vertices'],bank['faces']);rows=[]
    for frame,side in zip(*np.nonzero(bank['evidence'].activations)):
        local=(bank['hands'][frame,side]-t[frame])@r[frame]
        gap=whole_hand_minimum(surface,local,deadline=deadline)
        witness=bank['hands'][frame,side,gap['point_index']]
        ray=witness/np.linalg.norm(witness);ray_local=ray@r[frame]
        alignment=ray_gradient_alignment(surface,local[gap['point_index']],ray_local)
        pool_local=(pool.hand_points_camera[frame,side,pool.hand_visible[frame,side]]-t[frame])@r[frame]
        rows.append(dict(frame_index=int(frame),hand=int(side),**gap,**alignment,
            pool_gap_m=float(surface.distances(pool_local,batch_size=32).min())))
    return dict(variant=name,frames=len(t),active_hand_rows=len(rows),
        whole_hand_gap_m=summary([row['distance_m'] for row in rows]),
        pool_gap_m=summary([row['pool_gap_m'] for row in rows]),
        whole_hand_minus_pool_gap_m=summary([row['distance_m']-row['pool_gap_m'] for row in rows]),
        ray_alignment=summary([row['absolute_ray_alignment'] for row in rows if row['absolute_ray_alignment'] is not None]),
        exact_points=sum(row['points_exactly_evaluated'] for row in rows),
        possible_points=sum(row['points_total'] for row in rows),rows=rows,
        factor_costs=factor_costs(bank,r,t,pool,cfg,fit_cfg,name if name!='original' else 'J1'))


def load_candidates(code,ledger,bank):
    pins=strict((code/PIN_CONFIG).read_bytes())
    if (type(pins) is not dict or set(pins)!={'schema','producer_revision','report','predictions'}
            or pins.get('schema')!='world_reward.contact_real_geometry_audit_pins.v1'
            or type(pins['producer_revision']) is not str
            or re.fullmatch('[0-9a-f]{40}',pins['producer_revision']) is None
            or type(pins['predictions']) is not dict
            or set(pins['predictions'])!={'J1','hard_pool','soft_pool'}):
        raise ValueError('Independently committed completed experiment pins required')
    revision=pins['producer_revision'];folder=ROOT/'results'/('sequence-contact-patch-real-'+revision)
    report=strict(ledger.read(folder/'report.json',pins['report']))
    if (report['producer_revision']!=revision or report['status']!='complete_saved_full_T_contact_ablation'
            or report['ground_truth_used'] is not False or report['production_adopted'] is not False
            or report['full_4D_export_replaced'] is not False):
        raise ValueError('Actual original completed v1 experiment identity required')
    variants={'original':(bank['rotations'],bank['translations'])}
    rows={row['name']:row for row in report['fits']}
    if set(rows)!={'J1','hard_pool','soft_pool'}:raise ValueError('All originally compared variants required')
    frozen=real.settings(code)
    if report['protocol']!=frozen or report['fit_config']['max_nfev']!=frozen['max_nfev']:
        raise ValueError('Completed experiment frozen quality/config differs')
    for name,row in rows.items():
        if row['status']!='complete' or row['output']!=pins['predictions'][name]:
            raise ValueError('Actual complete independently pinned prediction required')
        a=real.saved.load_npz(ledger,folder/('episode_000009_'+name+'.npz'),row['output'])
        real.validate_output(bank,a)
        if not np.array_equal(a['contact_activations'],bank['evidence'].activations):
            raise ValueError('Frozen original automatic contact activations changed')
        variants[name]=(a['rotation'],a['translation'])
    from world_reward.sequence_pose import SequencePoseConfig
    return variants,report['protocol'],SequencePoseConfig(**report['fit_config'])


def run():
    revision=os.environ['WR_CODE_REVISION'];code=Path(os.environ['WR_CODE']);started=time.monotonic()
    if code!=ROOT/'jobs'/revision/ENTRY/'code':raise ValueError('Exact immutable Azure audit source required')
    out=ROOT/'results'/('contact-real-geometry-audit-'+revision);real.old.fresh_runtime_output(out)
    ledger=real.old.ArtifactLedger();report=dict(status='fail',producer_revision=revision,
        new_fits=0,model_calls=0,production_adopted=False,ground_truth_used=False,
        heldout_accuracy_verified=False,normal_angles_claimed=False)
    try:
        binding=source(ROOT,code,revision,ENTRY,HELPERS)
        for name in HELPERS:ledger.record(code/name)
        bank=real.load_bank(ledger);variants,cfg,fit_cfg=load_candidates(code,ledger,bank)
        pool,_=real.bounded_pool(bank,cfg);deadline=started+600
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(variant_audit,name,bank,r,t,pool,cfg,fit_cfg,deadline)
                for name,(r,t) in variants.items()]
            report['variants']=[future.result() for future in futures]
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS)!=binding:raise ValueError('Own closure changed')
        report.update(status='complete_saved_geometry_factor_diagnostic',sources=ledger.records,source_binding=binding)
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc)[:300])
    report['elapsed_seconds']=time.monotonic()-started
    real.seal_json(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds')},allow_nan=False),flush=True)
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':
    if len(os.sys.argv)!=1:raise SystemExit('No arbitrary inputs supported')
    run()

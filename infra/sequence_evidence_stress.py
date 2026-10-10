"""Azure-only independent structural stress, no challenge/model data mounted.

One frozen comparison per mechanism, DEV and separately seeded RESERVED. Owned
analytic observations are not a substitute for real external generalization.
Both methods receive the same noisy authored priors, known authored mesh/K,
and exact authored first-frame gauge anchor. This deliberately controlled
numeric assay is not truth-free RGB inference or physically rendered data.
No per-case coefficient changes, content rerolls or stopping on a favorable case.
"""
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from world_reward.sequence_pose import (SequencePoseConfig, RGBDepthConfig,
    SequenceContactConfig, SequenceContactEvidence, _ContactTriangleSurface,
    refine_sequence)
from world_reward.depth_covariance import CommonModeDepthConfig
from world_reward.contact_patch import (ContactPatchConfig, frozen_interval_candidates)
from mediapipe_cpu_runtime_verify import identity, source, strict, write

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_sequence_evidence_stress'
CONFIG = 'configs/sequence_evidence_stress_v1.json'
HELPERS = ('infra/sequence_evidence_stress.py', 'infra/run_sequence_evidence_stress.sh',
    CONFIG, 'src/world_reward/sequence_pose.py', 'src/world_reward/depth_covariance.py',
    'src/world_reward/contact_patch.py', 'infra/mediapipe_cpu_runtime_verify.py')


def protocol_disclosures():
    """Honest scope of this immutable assay; never imply real-data validation.

    The v1 numerical configuration/thresholds remain unchanged. These labels
    clarify limitations found by independent review, not new acceptance gates.
    """
    return dict(challenge_ground_truth_used_for_inference=False,
        authored_known_mesh_and_intrinsics_used=True,
        authored_exact_first_frame_gauge_anchor_used=True,
        authored_noisy_truth_derived_priors_used=True,
        observation_points_are_visible_surface_samples=False,
        rolling_case_is_two_axis_planar_slide=True,
        regrasp_case_has_explicit_inactive_gap=True,
        reduced_contact_pool_coverage_tested=False,
        rotational_motion_retention_gated=False,
        oscillation_amplitude_and_phase_gated=False,
        independent_contact_gap_is_existential=True,
        physically_rendered_observation_validation=False)


def settings(code):
    cfg = strict((code/CONFIG).read_bytes())
    if (cfg['schema'] != 'world_reward.sequence_evidence_stress.v1'
            or cfg['challenge_inputs_allowed'] is not False or cfg['production_adopted'] is not False
            or cfg['frames'] != 24 or cfg['fps'] != 30 or cfg['parallel_workers'] != 2
            or cfg['cases'] != dict(depth=['common_static','common_axial','common_fast','common_occlusion','iid_control','local_outlier'],
                                   contact=['static_contact','sliding_contact','rolling_contact','regrasp','absent_contact','candidate_outlier'])):
        raise ValueError('Frozen independent no-challenge stress protocol required')
    return cfg


def cube():
    v = np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)])
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int64)
    return v,faces


def observations(cfg, rng, moving, fast=False):
    v,faces = cube(); n=cfg['frames']; u=np.arange(n)/cfg['fps']
    diameter=2*np.sqrt(np.mean(np.sum((v-v.mean(0))**2,axis=1)))
    p=rng.uniform(-.085,.085,(24,3)); r=np.broadcast_to(np.eye(3),(n,3,3)).copy()
    t=np.column_stack((u*0,u*0,2.+u*0))
    if moving:
        t[:,0]+=.18*u; t[:,2]+=.30*u
        r=Rotation.from_rotvec(np.column_stack((u*0,.4*u,u*0))).as_matrix()
    if fast: t[:,2]+=.035*np.sin(2*np.pi*3*u)
    k=np.array([[320.,0,160.],[0,320.,120.],[0,0,1.]])
    xyz=p[None]@r.swapaxes(-1,-2)+t[:,None]
    xy=xyz[...,:2]/xyz[...,2,None]*k.diagonal()[:2]+k[:2,2]
    xy+=rng.normal(0,.35,xy.shape)
    prior_r=Rotation.from_rotvec(rng.normal(0,.015,(n,3))).as_matrix()@r
    prior_t=t+rng.normal(0,.01,t.shape)
    prior_r[0]=r[0]; prior_t[0]=t[0]
    return v,faces,p,k,r,t,xyz,xy,prior_r,prior_t,diameter


def scores(v,p,r,t,truth,truth_centroid,fps):
    xyz=p[None]@r.swapaxes(-1,-2)+t[:,None]
    centroid=np.einsum('tij,j->ti',r,v.mean(0))+t
    speed=np.linalg.norm(np.diff(centroid,axis=0),axis=1)
    true_speed=np.linalg.norm(np.diff(truth_centroid,axis=0),axis=1)
    denom=float(true_speed.sum())
    displacement=float(np.linalg.norm(truth_centroid[-1]-truth_centroid[0]))
    return dict(point_error_m=float(np.linalg.norm(xyz-truth,axis=-1).mean()),
        axial_error_m=float(np.abs(xyz[...,2]-truth[...,2]).mean()),
        temporal_error_m_s=float(np.linalg.norm(np.diff(xyz-truth,axis=0),axis=-1).mean()*fps),
        motion_path_retention=None if denom<1e-10 else float(speed.sum()/denom),
        motion_displacement_retention=None if displacement<1e-10 else float(np.linalg.norm(centroid[-1]-centroid[0])/displacement),
        acceleration_proxy_p95_m_s2=float(np.quantile(np.linalg.norm(np.diff(centroid,n=2,axis=0),axis=1)*fps**2,.95)))


def depth_case(cfg,seed,case):
    rng=np.random.default_rng(seed); f=observations(cfg,rng,case!='common_static',case=='common_fast')
    v,faces,p,k,r,t,truth,xy,prior_r,prior_t,d=f; n=len(t)
    visible=np.ones(xy.shape[:2],bool)
    iid=cfg['depth_independent_sigma_diameter']*d
    common=cfg['depth_common_sigma_diameter']*d
    noise=rng.normal(0,iid,visible.shape); bias=np.zeros(n)
    if case!='iid_control':
        rho=cfg['depth_ar1']; bias[0]=rng.normal(0,common)
        for i in range(1,n): bias[i]=rho*bias[i-1]+rng.normal(0,common*np.sqrt(1-rho*rho))
    depth=truth[...,2]+noise+bias[:,None]
    if case=='local_outlier': depth[5:16,0]+=.25*d
    if case=='common_occlusion':
        visible[8:12]=False; visible[15:,::3]=False
        xy[~visible]=np.nan; depth[~visible]=np.nan
    fit_cfg=SequencePoseConfig(**cfg['fit']); ref='owned_authored_stress_covariance_no_challenge_calibration'
    depth_cfg=RGBDepthConfig(cfg['depth_independent_sigma_diameter'],ref)
    result={}; times={}; convergence={}
    for name,extra in (('legacy',{}),('covariance',dict(depth_covariance_config=CommonModeDepthConfig(cfg['depth_common_sigma_diameter'],ref)))):
        start=time.monotonic()
        fit=refine_sequence(v,p,xy,visible,prior_r,prior_t,np.ones(n,bool),k,np.arange(n),cfg['fps'],fit_cfg,
            tracks_depth_m=depth,depth_visible=visible,depth_config=depth_cfg,**extra)
        times[name]=time.monotonic()-start; convergence[name]=fit.diagnostics['converged']
        result[name]=scores(v,p,fit.rotations,fit.translations,truth,
            np.einsum('tij,j->ti',r,v.mean(0))+t,cfg['fps'])
    return dict(mechanism='depth',case=case,seed=seed,metrics=result,seconds=times,converged=convergence,
        observations=int(visible.sum()),injected_common_sigma_m=common if case!='iid_control' else 0.,
        injected_iid_sigma_m=iid,injected_common_rho=cfg['depth_ar1'] if case!='iid_control' else None)


def contact_case(cfg,seed,case):
    rng=np.random.default_rng(seed); f=observations(cfg,rng,case not in ('static_contact',))
    v,faces,p,k,r,t,truth,xy,prior_r,prior_t,d=f; n=len(t)
    # Eight fixed anatomical candidates. Membership can move across fingers;
    # points are never fixed in object space after generating observations.
    local=np.array([[-.1,y,z] for y in (-.035,.035) for z in (-.06,-.02,.02,.06)])
    hands_local=np.broadcast_to(local,(n,2,8,3)).copy()
    if case in ('sliding_contact','rolling_contact'):
        hands_local[:,0,:,1]+=.035*np.sin(np.linspace(-1,1,n))[:,None]
    if case=='rolling_contact':
        hands_local[:,0,:,2]+=.02*np.sin(np.linspace(0,2,n))[:,None]
    if case=='regrasp': hands_local[n//2:,0,:,0]=.1
    if case=='candidate_outlier': hands_local[:,0,7,0]-=.4*d
    if case=='absent_contact': hands_local[:,:, :,0]-=.4*d
    hand_truth=hands_local@r[:,None].swapaxes(-1,-2)+t[:,None,None]
    hands=hand_truth+rng.normal(0,.004*d,hand_truth.shape)
    supported=np.zeros(hands.shape[:-1],bool); supported[:,0]=True; hands[~supported]=np.nan
    active=np.zeros((n,2),bool); active[:,0]=case!='absent_contact'
    if case=='regrasp': active[n//2-1:n//2+1,0]=False
    local_prior=np.einsum('thji,tik->thjk',hands-prior_t[:,None,None],prior_r)
    distances=np.full(supported.shape,np.nan); surface=_ContactTriangleSurface(v,faces)
    distances[supported]=surface.distances(local_prior[supported])
    selected=np.argmin(np.where(supported,distances,np.inf),axis=2)
    single=np.take_along_axis(hands,selected[:,:,None,None],axis=2)
    single_support=np.take_along_axis(supported,selected[:,:,None],axis=2)
    ids=np.arange(16).reshape(2,8)
    chosen,patch_points,patch_support=frozen_interval_candidates(hands,ids,distances,active,supported,cfg['max_contact_candidates'])
    fit_cfg=SequencePoseConfig(**cfg['fit']); ref='owned_authored_patch_stress_no_challenge_selection'
    contact_cfg=SequenceContactConfig(cfg['contact_sigma_diameter'],8,100_000_000,ref)
    result={};times={};convergence={}
    for name,points,flags,extra in (
        ('legacy',single,single_support,{}),
        ('soft_patch',patch_points,patch_support,dict(contact_patch_config=ContactPatchConfig(cfg['contact_temperature_diameter'],cfg['max_contact_candidates'],ref)))):
        evidence=SequenceContactEvidence(active,points,flags,faces,ref)
        start=time.monotonic()
        fit=refine_sequence(v,p,xy,np.ones(xy.shape[:2],bool),prior_r,prior_t,np.ones(n,bool),k,np.arange(n),cfg['fps'],fit_cfg,
            contact_evidence=evidence,contact_config=contact_cfg,**extra)
        times[name]=time.monotonic()-start;convergence[name]=fit.diagnostics['converged']
        result[name]=scores(v,p,fit.rotations,fit.translations,truth,
            np.einsum('tij,j->ti',r,v.mean(0))+t,cfg['fps'])
        # Score against authored noiseless hand geometry, not the fitted pool.
        true_local=np.einsum('thji,tik->thjk',hand_truth-fit.translations[:,None,None],fit.rotations)
        gaps=surface.distances(true_local[:,0].reshape(-1,3)).reshape(n,8).min(1)
        result[name]['independent_contact_gap_m']=float(gaps[active[:,0]].mean()) if active.any() else None
    return dict(mechanism='contact',case=case,seed=seed,metrics=result,seconds=times,converged=convergence,
        active_hand_frames=int(active.sum()),diameter_m=d,patch_source_indices_identical_within_active_interval=True,
        signed_penetration_verified=False)


def worker(job):
    cfg,seed,mechanism,case,split=job
    try:
        row=(depth_case if mechanism=='depth' else contact_case)(cfg,seed,case)
        return dict(status='complete',split=split,**row)
    except Exception as exc:
        return dict(status='fail',split=split,mechanism=mechanism,case=case,seed=seed,
            error_type=type(exc).__name__,error=str(exc)[:300])


def gates(cfg,rows):
    out=[]; limits=cfg['gates']
    for mechanism,new in (('depth','covariance'),('contact','soft_patch')):
        for split in ('development','reserved'):
            selected=[r for r in rows if r['mechanism']==mechanism and r['split']==split]
            complete=(len(selected)==len(cfg['cases'][mechanism])
                and {r.get('case') for r in selected}==set(cfg['cases'][mechanism])
                and all(r['status']=='complete' for r in selected))
            checks={'all_cases_complete':complete}
            if complete:
                target=[r for r in selected if r['case'] not in ('iid_control','absent_contact')]
                baseline=sum(r['metrics']['legacy']['point_error_m'] for r in target)
                candidate=sum(r['metrics'][new]['point_error_m'] for r in target)
                checks['aggregate_target_error_improves']=candidate<=limits['target_error_ratio_maximum']*baseline
                checks['all_cases_error_nonworse']=all(r['metrics'][new]['point_error_m']<=limits['control_error_ratio_maximum']*r['metrics']['legacy']['point_error_m'] for r in selected)
                # Endpoint displacement cannot establish oscillation/rotation
                # retention. Relabel only; keep the frozen v1 numeric gate.
                checks['all_endpoint_displacements_retained']=all(limits['motion_retention_minimum']<=r['metrics'][new]['motion_displacement_retention']<=limits['motion_retention_maximum'] for r in selected if r['metrics'][new]['motion_displacement_retention'] is not None)
                checks['all_candidates_converged']=all(r['converged'][new] for r in selected)
                if mechanism=='contact':
                    checks['independent_contact_not_sacrificed']=all(r['metrics'][new]['independent_contact_gap_m']<=r['metrics']['legacy']['independent_contact_gap_m']+limits['contact_gap_regression_tolerance_diameter']*r['diameter_m'] for r in target)
            out.append(dict(mechanism=mechanism,split=split,passed=all(checks.values()),checks=checks))
    return out


def run():
    start=time.monotonic();revision=os.environ['WR_CODE_REVISION'];code=Path(os.environ['WR_CODE'])
    if ROOT!=Path(os.environ['WR_ROOT']) or code!=ROOT/'jobs'/revision/ENTRY/'code':raise ValueError('Exact Azure source required')
    out=ROOT/'results'/('sequence-evidence-stress-'+revision)
    if not out.is_dir() or {p.name for p in out.iterdir()}!={'.container.cid'}:raise ValueError('Fresh owned diagnostic output required')
    report=dict(status='fail',producer_revision=revision,challenge_inputs_used=False,models_loaded=False,
        GPU_requested=False,protocol_disclosures=protocol_disclosures(),production_adopted=False,
        real_data_accuracy_verified=False,scope='authored_numeric_stress_not_real_validation')
    try:
        binding=source(ROOT,code,revision,ENTRY,HELPERS);cfg=settings(code)
        jobs=[]
        for split,seed in (('development',cfg['development_seed']),('reserved',cfg['reserved_seed'])):
            for mechanism,cases in cfg['cases'].items():
                for i,case in enumerate(cases):jobs.append((cfg,seed+i+(100 if mechanism=='contact' else 0),mechanism,case,split))
        with multiprocessing.get_context('spawn').Pool(cfg['parallel_workers']) as pool: rows=pool.map(worker,jobs)
        decisions=gates(cfg,rows)
        if source(ROOT,code,revision,ENTRY,HELPERS)!=binding:raise ValueError('Immutable source changed')
        report.update(status='complete_authored_stress_not_quality_pass',config=cfg,source_binding=binding,
            config_pin=identity(code/CONFIG),cases=rows,gates=decisions,new_fits=sum(2 for r in rows if r['status']=='complete'),
            evidence_bank_shared_within_comparison=True,parameter_sweeps=0,
            production_recommendation='not_adopted_requires_real_external_evaluation')
    except Exception as exc:report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds']=time.monotonic()-start
    write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),mode=0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','gates')}),flush=True)
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':run()

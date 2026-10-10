"""One Azure full-T pose fit using evolving automatic anatomical memory.

Sealed soft-pool poses generate associations ONLY. The fit retains original pose
priors, the same 300-evaluation cap, geometry and image evidence. An additional
stage is not a claim of identical complete pipeline compute.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import contact_real_geometry_audit as audit
from mediapipe_cpu_runtime_verify import source, strict
from world_reward.contact_memory import ContactMemoryConfig, evolving_contact_patch
from world_reward.contact_patch import ContactPatchConfig
from world_reward.sequence_pose import SequenceContactConfig, SequenceContactEvidence, refine_sequence

real=audit.real
ROOT=real.ROOT
ENTRY='run_contact_memory_real'
CONFIG='configs/contact_memory_real_v1.json'
HELPERS=tuple(dict.fromkeys(('infra/contact_memory_real.py','infra/run_contact_memory_real.sh',
    CONFIG,'src/world_reward/contact_memory.py',*audit.HELPERS)))


def settings(code):
    cfg=strict((Path(code)/CONFIG).read_bytes())
    expected=dict(schema='world_reward.contact_memory_real.v1',episode=9,frames=415,fps=30.,
        variants=['contact_memory'],workers=1,max_nfev=300,max_candidates=8,
        contact_sigma_diameter=.02,temperature_diameter=.01,max_point_triangle_pairs=1_000_000_000,
        depth_used=False,covariance_used=False,production_adopted=False)
    if any(type(cfg.get(key)) is not type(value) or cfg[key]!=value for key,value in expected.items()):
        raise ValueError('Exact frozen full-T contact-memory structural comparison required')
    ContactMemoryConfig(**cfg['association'])
    previous=real.settings(code)
    for key in ('max_nfev','max_candidates','contact_sigma_diameter','temperature_diameter','max_point_triangle_pairs'):
        if cfg[key]!=previous[key]:raise ValueError('Contact coefficients/budget must match saved pool control')
    return cfg


def owned_output(out,revision):
    path=Path(out)
    if (path!=ROOT/'results'/('contact-memory-real-'+revision) or path.resolve()!=path or not path.is_dir()
            or any(p.is_symlink() for p in (path,*path.parents)) or re.fullmatch('[0-9a-f]{40}',revision) is None
            or os.environ.get('WR_CODE_REVISION')!=revision):raise ValueError('Own contact-memory result namespace required')
    cid=path/'.container.cid';real.old.identity(cid,65,readonly=False)
    if re.fullmatch(b'[0-9a-f]{64}\n?',cid.read_bytes()) is None:raise ValueError('Own exact container receipt required')
    return path


def memory_evidence(bank,proposal_r,proposal_t,spec,cfg):
    """Native faces must index local anatomy; never infer an unverified mapping."""
    faces=tuple(spec[key] for key in ('hand_faces_left','hand_faces_right'))
    for f in faces:
        if (f.dtype.kind not in 'iu' or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f)
                or np.any(f<0) or np.any(f>=bank['hands'].shape[2])):
            raise ValueError('Pinned native hand faces must index exact local anatomy')
    memory=evolving_contact_patch(bank['hands'],bank['ids'],bank['evidence'].activations,
        proposal_r,proposal_t,bank['vertices'],bank['faces'],faces,ContactMemoryConfig(**cfg['association']))
    evidence=SequenceContactEvidence(bank['evidence'].activations,memory['hand_points_camera'],
        memory['geometry_supported'],bank['faces'],'native_activations_temporal_MAP_patch_from_sealed_soft_pool_proposals')
    if not np.array_equal(evidence.activations,memory['native_activations']):raise ValueError('Native activation changed')
    return evidence,memory


def memory_metrics(bank,r,t,evidence):
    """Independent unchanged geometry; report evolving pool separately from QA."""
    from world_reward.sequence_pose import _ContactTriangleSurface
    surface=_ContactTriangleSurface(bank['vertices'],bank['faces'])
    rows=[]
    for frame,side in zip(*np.nonzero(evidence.activations)):
        local=(evidence.hand_points_camera[frame,side,evidence.hand_visible[frame,side]]-t[frame])@r[frame]
        rows.append(float(surface.distances(local,batch_size=32).min()))
    return dict(evolving_memory_pool_gap_mean_m=float(np.mean(rows)) if rows else None,
        evolving_memory_pool_gap_p95_m=float(np.quantile(rows,.95)) if rows else None)


def fit_worker(bank,evidence,memory,cfg,fit_cfg,sealing=None):
    started=time.monotonic();out=None;name='contact_memory'
    try:
        if sealing is not None:
            out=owned_output(sealing['out'],sealing['revision'])
            if any((out/('episode_000009_'+name+suffix)).exists() for suffix in ('.npz','_fit.json')):
                raise FileExistsError('No replay over sealed contact-memory predictions')
        contact_cfg=SequenceContactConfig(cfg['contact_sigma_diameter'],cfg['max_candidates'],
            cfg['max_point_triangle_pairs'],cfg['development_reference'])
        fit=refine_sequence(bank['vertices'],bank['points'],bank['xy'],bank['visible'],
            bank['rotations'],bank['translations'],np.ones(len(bank['frame_index']),bool),
            bank['K'],bank['frame_index'],bank['fps'],fit_cfg,contact_evidence=evidence,
            contact_config=contact_cfg,contact_distance_batch_size=32,
            contact_patch_config=ContactPatchConfig(cfg['temperature_diameter'],cfg['max_candidates'],cfg['development_reference']))
        row=dict(name=name,status='complete',fit=fit.diagnostics,seconds=time.monotonic()-started)
        if sealing is None:return dict(row,rotations=fit.rotations,translations=fit.translations)
        arrays=real.output_arrays(bank,fit.rotations,fit.translations,evidence.activations,memory['selected_ids'])
        arrays.update(hand_points_camera=evidence.hand_points_camera,hand_geometry_supported=evidence.hand_visible,
            memory_winner_ids=memory['winner_ids'],proposal_vertex_gap_m=memory['proposal_vertex_gap_m'],
            normal_parallelness=memory['normal_parallelness'],normal_supported=memory['normal_supported'])
        file='episode_000009_contact_memory.npz'
        row.update(file=file,output=real.seal_file(out/file,lambda stream:np.savez_compressed(stream,**arrays)))
        receipt_file='episode_000009_contact_memory_fit.json'
        row['receipt']=real.seal_json(out/receipt_file,dict(row,producer_revision=sealing['revision'],
            ground_truth_used=False,production_adopted=False,quality_evaluation_complete=False,
            full_4D_export_replaced=False))
        row['receipt_file']=receipt_file
        return row
    except Exception as exc:
        row=dict(name=name,status='fail',error_type=type(exc).__name__,error=str(exc)[:300],seconds=time.monotonic()-started)
        if out is not None:
            try:row['failure_receipt']=real.seal_json(out/'episode_000009_contact_memory_failure.json',dict(row,producer_revision=sealing['revision']))
            except Exception as error:row['failure_receipt_error_type']=type(error).__name__
        return row


def run():
    start=time.monotonic();revision=os.environ['WR_CODE_REVISION'];code=Path(os.environ['WR_CODE'])
    if code!=ROOT/'jobs'/revision/ENTRY/'code' or ROOT!=Path(os.environ['WR_ROOT']):raise ValueError('Immutable Azure source required')
    out=ROOT/'results'/('contact-memory-real-'+revision);real.old.fresh_runtime_output(out);ledger=real.old.ArtifactLedger()
    report=dict(status='fail',producer_revision=revision,model_calls=0,new_fits=0,ground_truth_used=False,
        manual_labels=False,production_adopted=False,heldout_accuracy_verified=False,full_4D_export_replaced=False)
    try:
        binding=source(ROOT,code,revision,ENTRY,HELPERS)
        for name in HELPERS:ledger.record(code/name)
        cfg=settings(code);bank=real.load_bank(ledger);variants,old_cfg,fit_cfg=audit.load_candidates(code,ledger,bank)
        pool,_=real.bounded_pool(bank,old_cfg)
        spec=real.saved.load_npz(ledger,ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',real.contact.HAND_PIN)
        evidence,memory=memory_evidence(bank,*variants['soft_pool'],spec,cfg)
        association_file='episode_000009_contact_memory_associations.npz'
        association_arrays={key:value for key,value in memory.items() if isinstance(value,np.ndarray)}
        association_pin=real.seal_file(out/association_file,lambda stream:np.savez_compressed(stream,**association_arrays))
        report.update(protocol=cfg,fit_config=asdict(fit_cfg),association_file=association_file,
            association_pin=association_pin,association_source='pinned_saved_soft_pool_only_for_proposals',
            original_fit_priors_retained=True,candidate_states=memory['candidate_states'],
            normal_orientation_verified=False,normal_association='unsigned_parallelness_nearest_referenced_vertex_proposal',
            normals_used_as_hard_rejection=False,geometry_support='predicted_geometry_not_optical_visibility',
            actual_friction_or_sticking_constraint=False,original_activations_unchanged=True)
        adjacent=evidence.activations[1:] & evidence.activations[:-1]
        report['association_diagnostics']=dict(
            adjacent_active_hand_pairs=int(adjacent.sum()),
            winner_changes=int(np.count_nonzero(adjacent & (memory['winner_ids'][1:]!=memory['winner_ids'][:-1]))),
            normal_supported_active_entries=int(np.count_nonzero(memory['normal_supported'] & evidence.activations)),
            note='switch_count_is_not_physical_contact_accuracy_or_stability')
        print(json.dumps(dict(stage='memory_associations_sealed',candidate_states=memory['candidate_states'],
            native_active_frames=int(evidence.activations.sum())),allow_nan=False),flush=True)
        row=fit_worker(bank,evidence,memory,cfg,fit_cfg,dict(out=str(out),revision=revision))
        report['fit']=row
        if row['status']!='complete':raise ValueError('Memory worker failed: '+row.get('error','unknown'))
        report['new_fits']=1
        fitted=real.saved.load_npz(ledger,out/row['file'],row['output']);real.validate_output(bank,fitted)
        control_pins=strict((code/audit.PIN_CONFIG).read_bytes())
        control_folder=ROOT/'results'/('sequence-contact-patch-real-'+control_pins['producer_revision'])
        control_report=strict(ledger.read(control_folder/'report.json',control_pins['report']))
        control_fits={fit['name']:fit for fit in control_report['fits']}
        metrics={};converged={name:bool(control_fits[name]['fit']['converged']) for name in ('J1','soft_pool')}
        converged['contact_memory']=row['fit']['converged']
        for name,(r,t) in {**variants,'contact_memory':(fitted['rotation'],fitted['translation'])}.items():
            metrics[name]=real.measure(bank,r,t,pool)
            metrics[name].update(memory_metrics(bank,r,t,evidence))
        report['metrics']=metrics;report['decision']=real.decision(cfg,metrics,converged,'contact_memory')
        # Keep the same previously used pool as an independent conservative gate,
        # not a candidate's changed first finger or an existential contact oracle.
        for control in cfg['gates']['compare_against']:
            for metric in ('full_pool_existential_surface_gap_mean_m','full_pool_existential_surface_gap_p95_m'):
                report['decision']['gates'][control+'_'+metric+'_nonworse']=bool(
                    metrics['contact_memory'][metric]<=metrics[control][metric]*cfg['gates']['contact_gap_ratio_maximum']+1e-12)
        report['decision']['passed']=all(report['decision']['gates'].values())
        report['decision']['decision']='retain_for_visual_QA' if report['decision']['passed'] else 'reject_for_current_production'
        report['qa']=real.seal_json(out/'episode_000009_contact_memory_qa.json',dict(producer_revision=revision,
            metrics=metrics['contact_memory'],decision=report['decision'],production_adopted=False))
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS)!=binding:raise ValueError('Immutable source changed')
        report.update(status='complete_saved_contact_memory_diagnostic',sources=ledger.records,source_binding=binding,
            prediction_sealed_before_QA=True,geometry_K_scale_full_T_unchanged=True)
    except Exception as exc:report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds']=time.monotonic()-start
    real.seal_json(out/'report.json',report)
    print(json.dumps({key:report.get(key) for key in ('status','error_type','error','elapsed_seconds','decision')},allow_nan=False),flush=True)
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':
    if len(os.sys.argv)!=1:raise SystemExit('No arbitrary episode/source/config arguments supported')
    run()

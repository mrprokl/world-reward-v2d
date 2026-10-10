"""Azure saved full-T anatomical pool ablation, not a new baseline or truth eval.

One original automatic activation bank; three identical-budget fits separate
candidate pooling from a soft residual. No depth, model call, label, per-frame
alignment, camera/shape change or static-trajectory substitution is allowed.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

import pose_gain_probe as provenance
from mediapipe_cpu_runtime_verify import source, strict
from world_reward.contact_patch import ContactPatchConfig, frozen_interval_candidates
from world_reward.sequence_pose import (SequenceContactConfig, SequenceContactEvidence,
    _ContactTriangleSurface, refine_sequence)

contact = provenance.contact
saved = contact.saved
old = saved.old
ROOT = old.ROOT
ENTRY = 'run_sequence_contact_patch_real_v2'
CONFIG = 'configs/sequence_contact_patch_real_v1.json'
HELPERS = tuple(dict.fromkeys(('infra/sequence_contact_patch_real.py',
    'infra/run_sequence_contact_patch_real_v2.sh', CONFIG,
    'src/world_reward/contact_patch.py', 'src/world_reward/depth_covariance.py',
    *provenance.HELPERS)))


def settings(code):
    cfg = strict((code/CONFIG).read_bytes())
    expected = dict(schema='world_reward.sequence_contact_patch_real.v1', episode=9,
        frames=415, fps=30.0, variants=['J1', 'hard_pool', 'soft_pool'], workers=2,
        max_nfev=300, max_candidates=8, contact_sigma_diameter=.02,
        temperature_diameter=.01, max_point_triangle_pairs=1_000_000_000,
        depth_used=False, covariance_used=False, production_adopted=False)
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k,v in expected.items()):
        raise ValueError('Exact frozen real saved-contact protocol required')
    return cfg


def load_bank(ledger):
    """Existing independently authenticated producers, no forward pickle/decoder."""
    front, rows = old.saved_frontend(ledger)
    experiment = ROOT/'experiments'/('full4d-v1-'+old.SOURCE)
    transport = old.saved_numerical_frontend(ledger,experiment,rows)[9]
    base = experiment/'outputs/episode_000009'
    directory = base/'cari_shared_export_v1'
    pins = strict(ledger.read(experiment/'pins/cari_clip_000009_shared_export_pins.json'))['export_files']
    export = strict(ledger.read(directory/'report.json',pins['report.json']))
    if (export['status'] != 'pass' or export['producer_revision'] != old.SOURCE
            or export['ground_truth_used'] is not False or export['oracle_modes'] != []):
        raise ValueError('Original no-oracle full4D export lineage required')
    original = saved.load_npz(ledger,directory/'trajectory.npz',pins['trajectory.npz'])
    track = ROOT/'results'/('sequence-pose-probe-'+saved.TRACK_SOURCE+'-v2')
    saved.source_track_report(ledger,track/'report.json')
    rgb = saved.load_npz(ledger,track/'episode_000009.npz',saved.TRACK_PIN)
    previous = ROOT/'results'/('sequence-pose-contact-'+provenance.CONTACT_SOURCE)
    receipt = provenance.saved_report(ledger,previous/'report.json',
        provenance.CONTACT_REPORT_PIN,provenance.CONTACT_SOURCE,provenance.CONTACT_PIN,
        'complete_saved_RGB_contact_diagnostic_not_quality_pass')
    j1 = saved.load_npz(ledger,previous/'episode_000009.npz',provenance.CONTACT_PIN)
    provenance.same_bank(original,rgb)
    provenance.same_bank(original,j1,rgb)
    grid = np.arange(415)
    if (len(original['frame_index']) != 415 or export['frames'] != 415
            or export['original_frame_indices'] != grid.tolist() or receipt['fps'] != 30.
            or not np.array_equal(original['frame_index'],grid)):
        raise ValueError('Exact original 415-frame 30FPS chronology required')
    inventory, mask_report = old.authenticated_masks(ledger,front,rows[9],base,9,415)
    if transport['mask_inventory'] != mask_report['mask_inventory']:
        raise ValueError('Original native mask ancestry differs')
    if pins['native_parameters.npz'] != contact.NATIVE_PIN:
        raise ValueError('Native anatomy independent pin link differs')
    native = saved.load_npz(ledger,directory/'native_parameters.npz',contact.NATIVE_PIN)
    if not np.array_equal(native['frame_index'],grid):
        raise ValueError('Original native human chronology differs')
    path = directory/'target.npy'
    ledger.record(path,pins['target.npy'])
    human = np.load(path,mmap_mode='r',allow_pickle=False)
    if human.shape != (415,18439,3) or human.dtype.kind != 'f' or not np.isfinite(human).all():
        raise ValueError('Finite original full-T predicted MHR human required')
    hand_spec = saved.load_npz(ledger,ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',contact.HAND_PIN)
    ids = contact.hand_indices(hand_spec,native['human_faces'],human.shape[1])
    hands = np.asarray(human[:,ids])
    active = j1['contact_activations']
    if active.dtype != bool or active.shape != (415,2) or int(active.sum()) != 271:
        raise ValueError('Exact original 271 automatic contact activations required')
    # Reproduce the original prior-only qualification; never activate a new hand.
    reproduced, qualification = contact.automatic_evidence(original['object_vertices'],
        original['object_faces'],original['object_rotation'],original['object_translation'],
        hands,ids,j1['contact_logits'],'same_original_052_prior_automatic_contact')
    if (not np.array_equal(reproduced.activations,active)
            or not np.array_equal(reproduced.hand_points_camera,j1['hand_points_camera'],equal_nan=True)
            or not np.array_equal(reproduced.hand_visible,j1['hand_geometry_supported'])
            or not np.array_equal(qualification['selected_vertex_indices'],j1['selected_vertex_indices'])):
        raise ValueError('Saved contact activation/geometry cannot be reproduced from original predictions')
    spec = export['clip_spec']
    k = original['camera_K'].copy()
    k[0] *= 640/spec['width']; k[1] *= 480/spec['height']; k[:2,2] -= .5
    return dict(vertices=original['object_vertices'], faces=original['object_faces'],
        rotations=original['object_rotation'],translations=original['object_translation'],
        K=k, original_K=original['camera_K'],object_scale=original['object_scale'],
        points=rgb['points'],xy=rgb['tracks_xy'],visible=rgb['RGB_visible'],
        frame_index=grid,fps=30.,hands=hands,ids=ids,evidence=reproduced,
        contact_logits=j1['contact_logits'], prior_j1_rotations=j1['rotation'],
        prior_j1_translations=j1['translation'])


def bounded_pool(bank,cfg):
    """Cheap same-prior nearest referenced-vertex proposals; exact chosen QA.

    Ranking is explicitly a proposal, not an exact closest-surface/contact label.
    All original hand trajectories retain finite geometry and full original T.
    """
    h = bank['hands']; r = bank['rotations']; t = bank['translations']
    tree = cKDTree(bank['vertices'][np.unique(bank['faces'])])
    distances = np.empty(h.shape[:-1],np.float64)
    for frame in range(len(h)):
        local = (h[frame]-t[frame])@r[frame]
        distances[frame] = tree.query(local.reshape(-1,3))[0].reshape(h.shape[1:3])
    support = np.ones(distances.shape,bool)
    ids,points,valid = frozen_interval_candidates(h,bank['ids'],distances,
        bank['evidence'].activations,support,cfg['max_candidates'])
    evidence = SequenceContactEvidence(bank['evidence'].activations,points,valid,
        bank['faces'],'frozen_prior_referenced_vertex_ranked_anatomical_interval_pool')
    return evidence,ids


def measure(bank,r,t,pool):
    """All variants scored against the SAME original J1, not first pool member."""
    v,p,k,xy,visible = (bank[x] for x in ('vertices','points','K','xy','visible'))
    result = contact.residuals(v,p,r,t,k,xy,visible,bank['evidence'],bank['fps'])
    surface = _ContactTriangleSurface(v,bank['faces'])
    distances = []
    for frame,side in zip(*np.nonzero(pool.activations)):
        local = (pool.hand_points_camera[frame,side,pool.hand_visible[frame,side]]-t[frame])@r[frame]
        distances.append(float(surface.distances(local,batch_size=32).min()))
    result['full_pool_existential_surface_gap_mean_m'] = float(np.mean(distances)) if distances else None
    result['full_pool_existential_surface_gap_p95_m'] = float(np.quantile(distances,.95)) if distances else None
    # Image motion fidelity prevents treating a frozen trajectory as anti-jitter.
    projected = old.project(p,r,t,k)
    pairs = visible[1:] & visible[:-1]
    difference = np.diff(projected,axis=0)-np.diff(xy,axis=0)
    error = np.linalg.norm(difference[pairs],axis=1)
    result['RGB_motion_increment_error_mean_px'] = float(error.mean()) if len(error) else None
    result['RGB_motion_increment_observations'] = len(error)
    active = bank['evidence'].activations
    f,s = np.nonzero(active)
    local = np.einsum('nj,njk->nk',bank['evidence'].hand_points_camera[f,s,0]-t[f],r[f])
    gap = surface.distances(local,batch_size=32) if len(f) else np.empty(0)
    result['same_J1_surface_gap_p95_m'] = float(np.quantile(gap,.95)) if len(gap) else None
    return result


def decision(cfg,metrics,converged,name):
    gates = {'candidate_converged':bool(converged.get(name,False)),
             'fair_J1_converged':bool(converged.get('J1',False))}
    limits = cfg['gates']
    comparisons = {
        'RGB_mean_px':'reprojection_ratio_maximum',
        'selected_anatomical_triangle_mean_m':'contact_gap_ratio_maximum',
        'same_J1_surface_gap_p95_m':'contact_gap_ratio_maximum',
        'acceleration_proxy_m_s2_p95':'acceleration_tail_ratio_maximum',
        'angular_acceleration_proxy_rad_s2_p95':'acceleration_tail_ratio_maximum',
        'RGB_motion_increment_error_mean_px':'RGB_motion_increment_ratio_maximum'}
    for control in limits['compare_against']:
        for metric,limit in comparisons.items():
            a = metrics.get(name,{}).get(metric); b = metrics.get(control,{}).get(metric)
            gates[control+'_'+metric+'_nonworse'] = bool(a is not None and b is not None
                and np.isfinite(a) and np.isfinite(b) and a <= b*limits[limit]+1e-12)
    return dict(variant=name,passed=all(gates.values()),gates=gates,
        decision='retain_for_visual_QA' if all(gates.values()) else 'reject_for_current_production',
        scope=limits['scope'],heldout_accuracy_verified=False,production_adopted=False)


def owned_output(out,revision):
    """Require the exact new run's CID-owned namespace, never an old output."""
    out = Path(out)
    if (out != ROOT/'results'/('sequence-contact-patch-real-v2-'+revision)
            or out.resolve() != out or not out.is_dir()
            or any(p.is_symlink() for p in (out,*out.parents))
            or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision)
            or os.environ.get('WR_CODE_REVISION') != revision):
        raise ValueError('Exact current owned v2 output required')
    cid = out/'.container.cid'
    old.identity(cid,65,readonly=False)
    import re
    if re.fullmatch(b'[0-9a-f]{64}\n?',cid.read_bytes()) is None:
        raise ValueError('Current run requires its exact owned container CID')
    return out


def output_arrays(bank,rotations,translations,activations,selected_ids):
    return dict(rotation=rotations,translation=translations,frame_index=bank['frame_index'],
        object_vertices=bank['vertices'],object_faces=bank['faces'],object_scale=bank['object_scale'],
        camera_K=bank['original_K'],points=bank['points'],tracks_xy=bank['xy'],RGB_visible=bank['visible'],
        contact_activations=activations,selected_anatomical_pool_ids=selected_ids)


def validate_output(bank,fitted):
    """Full frozen bank, including queries/support (same geometry alone is insufficient)."""
    original=dict(object_translation=bank['translations'],object_vertices=bank['vertices'],
        object_faces=bank['faces'],object_scale=bank['object_scale'],
        camera_K=bank['original_K'],frame_index=bank['frame_index'])
    reference=dict(points=bank['points'],tracks_xy=bank['xy'],RGB_visible=bank['visible'])
    provenance.same_bank(original,fitted,reference)


def seal_file(path,write):
    """Atomic no-replace publish; only this call's temporary file is removed."""
    if path.exists() or path.is_symlink():
        raise FileExistsError('Refuse any existing sealed output')
    fd,temporary = tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            write(stream);stream.flush();os.fchmod(stream.fileno(),0o444);os.fsync(stream.fileno())
        # link is atomic and refuses existing paths; unlink restores the
        # required single-link artifact before any receipt advertises it.
        os.link(temporary,path,follow_symlinks=False)
        os.unlink(temporary);temporary=None
        directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:
        if temporary is not None:os.unlink(temporary)
    return old.identity(path,2<<30,readonly=True)


def seal_json(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    return seal_file(path,lambda stream:stream.write(raw))


def fit_worker(job):
    name,bank,evidence,cfg,fit_cfg = job[:5]
    sealing = job[5] if len(job)==6 else None  # Tiny unit fits need no remote output.
    started = time.monotonic();out=None
    try:
        if name not in cfg['variants']:
            raise ValueError('Only a frozen variant may own output files')
        if sealing is not None:
            out=owned_output(sealing['out'],sealing['revision'])
            if any((out/('episode_000009_'+name+suffix)).exists()
                   or (out/('episode_000009_'+name+suffix)).is_symlink()
                   for suffix in ('.npz','_fit.json')):
                raise FileExistsError('Refuse a previously published worker result before fitting')
        contact_cfg = SequenceContactConfig(cfg['contact_sigma_diameter'],
            1 if name=='J1' else cfg['max_candidates'],cfg['max_point_triangle_pairs'],cfg['development_reference'])
        extra = {} if name!='soft_pool' else dict(contact_patch_config=ContactPatchConfig(
            cfg['temperature_diameter'],cfg['max_candidates'],cfg['development_reference']))
        fit = refine_sequence(bank['vertices'],bank['points'],bank['xy'],bank['visible'],
            bank['rotations'],bank['translations'],np.ones(len(bank['frame_index']),bool),
            bank['K'],bank['frame_index'],bank['fps'],fit_cfg,
            contact_evidence=evidence,contact_config=contact_cfg,contact_distance_batch_size=32,**extra)
        row=dict(name=name,status='complete',fit=fit.diagnostics,seconds=time.monotonic()-started,
            execution_contact_batch_size=32)
        if sealing is None:
            return dict(row,rotations=fit.rotations,translations=fit.translations)
        path=out/('episode_000009_'+name+'.npz')
        row['file']=path.name
        arrays=output_arrays(bank,fit.rotations,fit.translations,evidence.activations,sealing['selected_ids'])
        row['output']=seal_file(path,lambda stream:np.savez_compressed(stream,**arrays))
        receipt=dict(row,producer_revision=sealing['revision'],
            full_4D_export_replaced=False,ground_truth_used=False,production_adopted=False,
            quality_evaluation_complete=False)
        receipt_path=out/('episode_000009_'+name+'_fit.json')
        row['receipt']=seal_json(receipt_path,receipt)
        row['receipt_file']=receipt_path.name
        return row
    except Exception as exc:
        row=dict(name=name,status='fail',error_type=type(exc).__name__,
            error=str(exc)[:300],seconds=time.monotonic()-started)
        if out is not None:
            try:
                receipt_path=out/('episode_000009_'+name+'_fit.json')
                row['receipt']=seal_json(receipt_path,dict(row,producer_revision=sealing['revision'],
                    ground_truth_used=False,production_adopted=False,quality_evaluation_complete=False))
                row['receipt_file']=receipt_path.name
            except Exception as error:
                row['receipt_error_type']=type(error).__name__
        return row


def run():
    started = time.monotonic()
    revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if ROOT != Path(os.environ['WR_ROOT']) or code != ROOT/'jobs'/revision/ENTRY/'code':
        raise ValueError('Exact Azure immutable source required')
    out = ROOT/'results'/('sequence-contact-patch-real-v2-'+revision)
    old.fresh_runtime_output(out); ledger = old.ArtifactLedger()
    report = dict(status='fail',producer_revision=revision,ground_truth_used=False,
        manual_labels=False,model_calls=0,depth_used=False,covariance_used=False,
        production_adopted=False,full_4D_export_replaced=False,heldout_accuracy_verified=False)
    try:
        binding = source(ROOT,code,revision,ENTRY,HELPERS)
        report['source_binding'] = binding
        for name in HELPERS: ledger.record(code/name)
        cfg = settings(code); base_cfg,_ = old.profile_config(code,'v2')
        fit_cfg = replace(base_cfg,max_nfev=cfg['max_nfev'])
        report.update(protocol=cfg,fit_config=asdict(fit_cfg),execution_revision='v2',
            quality_protocol_unchanged=True,execution_contact_batch_size=32,
            sealed_worker_completion_before_QA=True)
        report['metrics']={};report['fits']=[]
        bank = load_bank(ledger)
        pool,selected_ids = bounded_pool(bank,cfg)
        i,s = np.nonzero(pool.activations & (bank['frame_index'][:,None]>0))
        pairs = int(pool.hand_visible[i,s].sum())*len(bank['faces'])
        if pairs > cfg['max_point_triangle_pairs']:
            raise ValueError('Predeclared full-surface execution cap exceeded; no geometry/activation dropping')
        report.update(frames=len(bank['frame_index']),fps=bank['fps'],points=len(bank['points']),
            automatic_active_hand_frames=int(pool.activations.sum()),object_faces=len(bank['faces']),
            complete_surface_point_triangle_pairs=pairs,
            candidate_rank='nearest_referenced_vertex_proposal_not_exact_surface_label',
            support_semantics='finite_predicted_anatomy_not_optical_visibility',
            original_activations_unchanged=True,geometry_K_scale_full_T_unchanged=True,
            baseline_source=old.SOURCE,prior_J1_source=provenance.CONTACT_SOURCE)
        # Only bounded fitting inputs reach workers; full human/mesh source I/O is shared once.
        fit_bank = {k:v for k,v in bank.items() if k not in
            ('hands','ids','prior_j1_rotations','prior_j1_translations','contact_logits')}
        sealing=dict(out=str(out),revision=revision,selected_ids=selected_ids)
        jobs = [(name,fit_bank,bank['evidence'] if name=='J1' else pool,cfg,fit_cfg,sealing)
            for name in cfg['variants']]
        metrics = report['metrics'];converged = {}
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
            futures=[executor.submit(fit_worker,job) for job in jobs]
            metrics['original']=measure(bank,bank['rotations'],bank['translations'],pool)
            for future in as_completed(futures):
                row=future.result()
                print(json.dumps(dict(stage='fit_completed',name=row['name'],status=row['status'],
                    seconds=row['seconds'],converged=row.get('fit',{}).get('converged'),
                    prediction_sealed=row['status']=='complete'),allow_nan=False),flush=True)
                if row['status']=='complete':
                    receipt=strict(ledger.read(out/row['receipt_file'],row['receipt']))
                    if (receipt['producer_revision']!=revision or receipt['name']!=row['name']
                            or receipt['output']!=row['output'] or receipt['file']!=row['file']
                            or receipt['fit']!=row['fit']):
                        raise ValueError('Worker-sealed receipt/output identity differs')
                    fitted=saved.load_npz(ledger,out/row['file'],row['output'])
                    validate_output(bank,fitted)
                    metrics[row['name']]=measure(bank,fitted['rotation'],fitted['translation'],pool)
                    converged[row['name']]=row['fit']['converged']
                    qa=dict(name=row['name'],fit_receipt=row['receipt'],metrics=metrics[row['name']],
                        producer_revision=revision,heldout_accuracy_verified=False,production_adopted=False)
                    row['qa']=seal_json(out/('episode_000009_'+row['name']+'_qa.json'),qa)
                report['fits'].append(row)
        report['fits'].sort(key=lambda row:cfg['variants'].index(row['name']))
        report['metrics'] = metrics
        report['decisions'] = [decision(cfg,metrics,converged,name) for name in ('hard_pool','soft_pool')]
        report['quality_decision'] = ('retain_candidate_for_visual_QA' if any(
            row['passed'] for row in report['decisions']) else 'no_candidate_qualified_for_production')
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS) != binding:
            raise ValueError('Own immutable closure changed')
        report.update(status='complete_saved_full_T_contact_ablation',sources=ledger.records)
    except Exception as exc:
        report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds'] = time.monotonic()-started
    seal_json(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','decisions')},allow_nan=False))
    if report['status']=='fail': raise SystemExit(1)


if __name__=='__main__':
    if len(os.sys.argv)!=1: raise SystemExit('No arbitrary source/config inputs accepted')
    run()

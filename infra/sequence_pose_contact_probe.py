"""CPU saved RGB/contact ablation; depth OFF, no production adoption.

All native hand vertices propose one anatomical witness at the frozen prior;
exact triangles and original positive logits qualify frozen activation. This
conservative J1 protocol is not exhaustive native-contact parity or truth.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from world_reward.sequence_pose import (SequenceContactConfig, SequenceContactEvidence,
    _ContactTriangleSurface, _rigid, refine_sequence)
import sequence_pose_rgbd_probe as saved
from mediapipe_cpu_runtime_verify import source, strict

old = saved.old
ROOT = old.ROOT
ENTRY = 'run_sequence_pose_contact_probe'
CONTACT_CFG = SequenceContactConfig(.02,1,100000000,
    'manufactured_contact_DEV_20261031_RESERVED_20261101_v1_not_real_calibration')
ACTIVATION_DISTANCE_M = .05  # Published CARI activation threshold; not tuned here.
NATIVE_PIN = dict(bytes=1539524,sha256='3a5b70134c945500dead55e00048aa838752c42f1cfcfa5f99b11a81b8d0d488')
HAND_PIN = dict(bytes=47140,sha256='65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8')
FORWARD_PIN = dict(bytes=317633596,sha256='9dfbeea8708e29dd613c02cd8a02563fb17cabaa35affd8ea806b6180672d90e')
FORWARD_REPORT_PIN = dict(bytes=67760,sha256='20c87bca71eb6857849ecde2e94a49f4a61c93a2799abc3a18b151fe023dee0f')
FORWARD_PINS_PIN = dict(bytes=655,sha256='a7979dce63ef28279531aac612d3986656148e153eddaf3049d27cba5c1dda9c')
HAND_TOPOLOGY = '075bf66319bd8fa348e5acf28f838b056952e550419331cf3ed10b68060ea3ba'
HAND_FACES = '7f0898679b24c006e077df9531171a9206ae986d6d5885094cc646669b3784f5'
HAND_MODEL = '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
HELPERS = ('infra/sequence_pose_contact_probe.py','infra/run_sequence_pose_contact_probe.sh',*saved.HELPERS)


def array_sha(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value); digest.update(str(a.dtype).encode())
        digest.update(np.asarray(a.shape,np.int64).tobytes()); digest.update(a.tobytes())
    return digest.hexdigest()


def hand_indices(spec, faces, human_count):
    """Recompute official typed topology digest against exported MHR faces."""
    f = np.asarray(faces)
    if f.dtype.kind not in 'iu' or f.ndim != 2 or f.shape[1] != 3 or (f < 0).any() or (f >= human_count).any():
        raise ValueError('Exact exported human topology required')
    f = f.astype(np.int32); keys = ('vertex_indices','sample_local_indices','sample_assignments','hand_faces_left','hand_faces_right')
    for key,shape in zip(keys,((2,2318),(2,256),(2,2318),(4603,3),(4603,3))):
        if spec[key].dtype != np.int32 or spec[key].shape != shape: raise ValueError('Original typed hand anatomy required')
    ids = spec['vertex_indices']
    if ((ids < 0).any() or (ids >= human_count).any() or len(np.unique(ids)) != ids.size
            or array_sha(f) != HAND_FACES or str(spec['faces_sha256'].item()) != HAND_FACES
            or array_sha(f,*(spec[k] for k in keys)) != HAND_TOPOLOGY
            or str(spec['topology_sha256'].item()) != HAND_TOPOLOGY
            or str(spec['mhr_model_sha256'].item()) != HAND_MODEL):
        raise ValueError('Native anatomy/topology/model identity differs')
    return ids


def contact_logits(bundle, count):
    if (type(bundle) is not dict or bundle.get('schema') != 'cari4d.mhr_wild_inference.v1'
            or bundle.get('gt') != {} or 'postopt' in bundle
            or bundle.get('frames') != [f'{i:06d}' for i in range(count)]
            or bundle.get('metadata',{}).get('ground_truth_used') is not False):
        raise ValueError('Original full-T no-GT native contact bundle required')
    arrays = []
    for prefix in ('pr','pr_initial'):
        value = bundle.get(prefix,{}).get('contact_logits')
        if np.ma.isMaskedArray(value): raise ValueError('Masked contact logits forbidden')
        if hasattr(value,'detach'): value = value.detach().cpu().numpy()
        a = np.asarray(value)
        if a.dtype != np.float32 or a.shape != (count,2) or not np.isfinite(a).all():
            raise ValueError('Untouched finite FP32 native contact logits required')
        arrays.append(a)
    if arrays[0].tobytes() != arrays[1].tobytes(): raise ValueError('Predicted/initial contact logits changed')
    return arrays[0].copy()


def automatic_evidence(vertices, faces, rotations, translations, hands, indices, logits, reference):
    """One genuine anatomical point selected at prior only; no candidate relabels.

    Nearest object-vertex distance only proposes. Exact full-mesh triangle
    distance qualifies; this can miss contacts, not manufacture activation.
    Unsupported means no constraint, not physical absence/optical occlusion.
    """
    n = len(translations); h = np.asarray(hands); ids = np.asarray(indices); log = np.asarray(logits)
    if (h.dtype.kind != 'f' or h.ndim != 4 or h.shape[:2] != (n,2) or h.shape[-1] != 3
            or not h.shape[2] or not np.isfinite(h).all() or ids.dtype.kind not in 'iu'
            or ids.shape != h.shape[1:3] or (ids < 0).any() or log.dtype.kind != 'f'
            or log.shape != (n,2) or not np.isfinite(log).all()
            or np.shape(rotations) != (n,3,3) or np.shape(translations) != (n,3)):
        raise ValueError('Finite same-gauge full-T predicted anatomy/logits required')
    _rigid(rotations,translations)
    tree = cKDTree(vertices); surface = _ContactTriangleSurface(vertices,faces)
    selected = np.full((n,2),-1,np.int64); points = np.full((n,2,1,3),np.nan)
    valid = np.zeros((n,2,1),bool); active = np.zeros((n,2),bool)
    upper = np.full((n,2),np.nan); exact = np.full((n,2),np.nan)
    for i,side in zip(*np.nonzero(log > 0)):
        local = (h[i,side]-translations[i])@rotations[i]; distances,_ = tree.query(local)
        j = int(np.argmin(distances)); selected[i,side] = ids[side,j]
        points[i,side,0] = h[i,side,j]; valid[i,side,0] = True
        upper[i,side] = distances[j]; exact[i,side] = surface.distances(local[j:j+1])[0]
        active[i,side] = exact[i,side] < ACTIVATION_DISTANCE_M
    return SequenceContactEvidence(active,points,valid,faces,reference), dict(selected_vertex_indices=selected,
        vertex_proposal_distance_m=upper,exact_initial_triangle_distance_m=exact,positive_native_logits=log > 0)


def manufactured_gate(seed, reserved, config):
    """Independent noisy sensors; synthetic truth only enters scoring."""
    rng = np.random.default_rng(seed); rows = []
    v = np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)])
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int64)
    for case in ('static','moving','absent_contacts'):
        n = 24; u = np.arange(n)/30; p = rng.uniform(-.085,.085,(24,3)); k = np.array([[256.,0,128],[0,256.,128],[0,0,1.]])
        r = np.broadcast_to(np.eye(3),(n,3,3)).copy(); t = np.column_stack((u*0,u*0,2.+u*0))
        if case != 'static':
            t[:,0] = .15*u; t[:,2] += .12*u; r = Rotation.from_rotvec(np.column_stack((u*0,.3*u,u*0))).as_matrix()
        prior_t = t+rng.normal(0,.01,t.shape); prior_r = Rotation.from_rotvec(rng.normal(0,.015,(n,3))).as_matrix()@r
        prior_t[0] = t[0]; prior_r[0] = r[0]
        xy = old.project(p,r,t,k)+rng.normal(0,.35 if not reserved else .5,(n,len(p),2)); visible = np.ones(xy.shape[:2],bool)
        local = np.array([[[-.1,-.1,-.1],[-.1,.1,.1]],[[.1,-.1,.1],[.1,.1,-.1]]])
        hands = local[None]@r[:,None].swapaxes(-1,-2)+t[:,None,None]
        hands += rng.normal(0,.001 if not reserved else .002,hands.shape)
        logits = np.full((n,2),-1. if case == 'absent_contacts' else 1.,np.float32)
        evidence,_ = automatic_evidence(v,faces,prior_r,prior_t,hands,np.arange(4).reshape(2,2),logits,'owned_synthetic_sensor')
        fit = refine_sequence(v,p,xy,visible,prior_r,prior_t,np.ones(n,bool),k,np.arange(n),30,config,
            contact_evidence=evidence,contact_config=CONTACT_CFG)
        truth = p[None]@r.swapaxes(-1,-2)+t[:,None]; before = p[None]@prior_r.swapaxes(-1,-2)+prior_t[:,None]
        after = p[None]@fit.rotations.swapaxes(-1,-2)+fit.translations[:,None]
        a = float(np.linalg.norm(before-truth,axis=-1).mean()); b = float(np.linalg.norm(after-truth,axis=-1).mean())
        motion = old.motion_gate(truth,before,after)
        rows.append(dict(case=case,old_error_m=a,new_error_m=b,full_motion=motion,active_hand_frames=int(evidence.activations.sum()),
            passed=b <= .9*a and motion['passed'],converged=fit.diagnostics['converged']))
    return dict(seed=seed,reserved=reserved,cases=rows,passed=all(row['passed'] for row in rows),scope='manufactured_not_real_contact_accuracy')


def authenticate_forward(ledger, experiment, base, spec, input_sha):
    pins = strict(ledger.read(experiment/'pins/cari_clip_000009_shared_forward_pins.json',FORWARD_PINS_PIN))
    report = strict(ledger.read(base/'cari_shared_forward_v1/report.json',FORWARD_REPORT_PIN))
    expected = dict(stage='world_reward_native_cari_shared_full_video_forward',status='pass',producer_revision=old.SOURCE,
        clip_spec=spec,input_track='track_1',input_sha256=input_sha,frames=spec['total_frames'],ground_truth_used=False,
        private_truth_read=False,hand_labeled_test=False,oracle_modes=[],original_frame_indices=list(range(spec['total_frames'])))
    if (any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected.items())
            or pins.get('schema') != 'world-reward-cari-shared-forward-pins-v1' or pins.get('clip_spec') != spec
            or pins.get('forward_files') != {'coconet.pth':FORWARD_PIN,'report.json':FORWARD_REPORT_PIN}
            or pins.get('forward',{}).get('producer_revision') != old.SOURCE
            or report.get('bundle_sha256') != FORWARD_PIN['sha256'] or report.get('bundle_bytes') != FORWARD_PIN['bytes']):
        raise ValueError('Independent original no-GT forward/report/pins lineage differs')
    path = base/'cari_shared_forward_v1/coconet.pth'; ledger.record(path,FORWARD_PIN)
    import torch  # Explicitly approved hash-authenticated own producer, never arbitrary pickle.
    bundle = torch.load(path,map_location='cpu',weights_only=False); ledger.record(path,FORWARD_PIN)
    return contact_logits(bundle,spec['total_frames'])


def residuals(vertices,points,r,t,K,xy,visible,evidence,fps):
    pixel = np.linalg.norm(old.project(points,r,t,K)[visible]-xy[visible],axis=1)
    i,s = np.nonzero(evidence.activations)
    local = np.einsum('nj,njk->nk',evidence.hand_points_camera[i,s,0]-t[i],r[i])
    distance = _ContactTriangleSurface(vertices,evidence.object_faces).distances(local) if len(i) else np.empty(0)
    centroid = np.einsum('tij,j->ti',r,vertices.mean(0))+t
    acceleration = np.linalg.norm(np.diff(centroid,n=2,axis=0),axis=1)*fps**2
    spin = Rotation.from_matrix(r[1:]@r[:-1].swapaxes(-1,-2)).as_rotvec()*fps
    angular_acceleration = np.linalg.norm(np.diff(spin,axis=0),axis=1)*fps
    return dict(RGB_mean_px=float(pixel.mean()),RGB_observations=len(pixel),active_hand_frames=len(i),
        acceleration_proxy_m_s2_median=float(np.median(acceleration)),acceleration_proxy_m_s2_p95=float(np.quantile(acceleration,.95)),
        angular_acceleration_proxy_rad_s2_median=float(np.median(angular_acceleration)),
        angular_acceleration_proxy_rad_s2_p95=float(np.quantile(angular_acceleration,.95)),
        selected_anatomical_triangle_mean_m=float(distance.mean()) if len(i) else None,
        motion=old.pose_motion_summary(vertices,r,t,fps))


def diagnostic_gates(original, candidate):
    """Predeclared saved-observation QA; never held-out accuracy or a tuner."""
    names = ('acceleration_proxy_m_s2_median','acceleration_proxy_m_s2_p95',
             'angular_acceleration_proxy_rad_s2_median','angular_acceleration_proxy_rad_s2_p95')
    gates = {key:candidate[key] <= original[key]*(1+1e-8) for key in names}
    gates['RGB_reprojection_improves'] = candidate['RGB_mean_px'] < original['RGB_mean_px']
    a,b = original['selected_anatomical_triangle_mean_m'],candidate['selected_anatomical_triangle_mean_m']
    gates['active_anatomical_contact_nonworse'] = a is not None and b is not None and b <= a*(1+1e-8)
    return dict(gates=gates,passed=all(gates.values()),scope='predicted_observation_QA_not_accuracy',production_adopted=False)


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if ROOT != Path(os.environ['WR_ROOT']) or code != ROOT/'jobs'/revision/ENTRY/'code': raise ValueError('Exact Azure source required')
    out = ROOT/'results'/('sequence-pose-contact-'+revision); old.fresh_runtime_output(out); ledger = old.ArtifactLedger()
    report = dict(status='fail',producer_revision=revision,numerical_source=old.SOURCE,ground_truth_used=False,manual_labels=False,
        model_calls=0,full_4D_export_replaced=False,depth_used=False,contact_config=asdict(CONTACT_CFG),activation_distance_m=ACTIVATION_DISTANCE_M)
    try:
        binding = source(ROOT,code,revision,ENTRY,HELPERS); report['source_binding'] = binding
        for name in HELPERS: ledger.record(code/name)
        config,_ = old.profile_config(code,'v2'); report['config'] = asdict(config)
        for seed,reserved,name in ((20261031,False,'development'),(20261101,True,'reserved')):
            gate = manufactured_gate(seed,reserved,config); report[name] = gate
            if not gate['passed']: raise ValueError('Fresh external contact '+name+' gate rejected; no challenge access')
        front,rows = old.saved_frontend(ledger); experiment = ROOT/'experiments'/('full4d-v1-'+old.SOURCE)
        transport = old.saved_numerical_frontend(ledger,experiment,rows)[9]; base = experiment/'outputs/episode_000009'
        pins = strict(ledger.read(experiment/'pins/cari_clip_000009_shared_export_pins.json'))['export_files']; directory = base/'cari_shared_export_v1'
        export = strict(ledger.read(directory/'report.json',pins['report.json']))
        if export['status'] != 'pass' or export['producer_revision'] != old.SOURCE or export['ground_truth_used'] is not False or export['oracle_modes'] != []:
            raise ValueError('Original video-only export provenance differs')
        a = saved.load_npz(ledger,directory/'trajectory.npz',pins['trajectory.npz']); spec = export['clip_spec']
        track_path = ROOT/'results'/('sequence-pose-probe-'+saved.TRACK_SOURCE+'-v2')/'episode_000009.npz'
        saved.source_track_report(ledger,track_path.with_name('report.json')); b = saved.load_npz(ledger,track_path,saved.TRACK_PIN)
        v,faces,r,t,K = (a[k] for k in ('object_vertices','object_faces','object_rotation','object_translation','camera_K')); n = len(t); grid = np.arange(n)
        if (n != 415 or b['points'].shape != (33,3) or b['tracks_xy'].shape != (n,33,2) or b['RGB_visible'].shape != (n,33)
                or not np.array_equal(a['frame_index'],grid) or not np.array_equal(b['frame_index'],grid) or float(a['object_scale']) != 1.
                or any(not np.array_equal(a[k],b[k]) for k in ('object_vertices','object_faces','object_scale','camera_K'))
                or export['original_frame_indices'] != grid.tolist() or export['frames'] != n):
            raise ValueError('Exact original 33 tracks/shape/gauge/full timeline required')
        _,mask_report = old.authenticated_masks(ledger,front,rows[9],base,9,n)
        if transport['mask_inventory'] != mask_report['mask_inventory']: raise ValueError('Original automatic mask lineage differs')
        logits = authenticate_forward(ledger,experiment,base,spec,mask_report['input_sha256'])
        if pins['native_parameters.npz'] != NATIVE_PIN: raise ValueError('Export/native anatomy pin link differs')
        native = saved.load_npz(ledger,directory/'native_parameters.npz',NATIVE_PIN)
        if not np.array_equal(native['frame_index'],grid): raise ValueError('Native anatomical timeline differs')
        human_path = directory/'target.npy'; ledger.record(human_path,pins['target.npy']); human = np.load(human_path,mmap_mode='r',allow_pickle=False)
        if human.shape != (n,18439,3) or human.dtype.kind != 'f' or not np.isfinite(human).all(): raise ValueError('Full same-gauge predicted human vertices required')
        hand_spec = saved.load_npz(ledger,ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',HAND_PIN)
        ids = hand_indices(hand_spec,native['human_faces'],human.shape[1])
        evidence,qualification = automatic_evidence(v,faces,r,t,np.asarray(human[:,ids]),ids,logits,'original052_forward_logits_and_exported_MHR_anatomy_geometry')
        import cv2
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4'; ledger.record(video,transport['video_pin'])
        capture = cv2.VideoCapture(str(video))
        try:
            fps = old.actual_fps(capture)
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n: raise ValueError('Original video full frame grid differs')
        finally: capture.release()
        scale = np.array([640/spec['width'],480/spec['height']]); kk = K.copy(); kk[0] *= scale[0]; kk[1] *= scale[1]; kk[:2,2] -= .5
        xy,visible,points = b['tracks_xy'],b['RGB_visible'],b['points']
        fit = refine_sequence(v,points,xy,visible,r,t,np.ones(n,bool),kk,grid,fps,config,contact_evidence=evidence,contact_config=CONTACT_CFG)
        output = out/'episode_000009.npz'; np.savez_compressed(output,rotation=fit.rotations,translation=fit.translations,frame_index=grid,
            object_vertices=v,object_faces=faces,object_scale=a['object_scale'],camera_K=K,points=points,tracks_xy=xy,RGB_visible=visible,
            contact_logits=logits,contact_activations=evidence.activations,hand_points_camera=evidence.hand_points_camera,
            hand_geometry_supported=evidence.hand_visible,**qualification); output.chmod(0o444)
        measure = lambda rr,tt: residuals(v,points,rr,tt,kk,xy,visible,evidence,fps)
        report.update(status='complete_saved_RGB_contact_diagnostic_not_quality_pass',frames=n,points=len(points),fps=fps,fit=fit.diagnostics,
            original=measure(r,t),rejected_RGB_only=measure(b['rotation'],b['translation']),RGB_contact=measure(fit.rotations,fit.translations),
            output=ledger.record(output),production_adopted=False,heldout_accuracy_verified=False,contact_truth_claimed=False,native_contact_parity_claimed=False,
            hand_support_semantics='predicted_finite_geometry_not_optical_visibility',qualification='all2318_nearest_mesh_vertex_proposal_exact_selected_point_triangles_frozen_prior052')
        report['diagnostic_gate'] = diagnostic_gates(report['original'],report['RGB_contact'])
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS) != binding: raise ValueError('Own immutable source closure changed')
        report['sources'] = ledger.records
    except Exception as exc: report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds'] = time.monotonic()-started
    path = out/'report.json'; path.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n'); path.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','original','rejected_RGB_only','RGB_contact')},allow_nan=False))
    if report['status']=='fail': raise SystemExit(1)


if __name__ == '__main__':
    if len(os.sys.argv) != 1: raise SystemExit('No arbitrary inputs accepted')
    run()

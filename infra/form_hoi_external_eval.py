"""Paired FORM DEV reference evaluation, strictly after ALL predictions seal.

The five public Track1 operators are unchanged. Human roles and object seeds
are explicitly external: these numbers are not Kaggle scores. No models, GT
selection, reference input to predictions, or per-frame similarity alignment.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time

import numpy as np
from mediapipe_cpu_runtime_verify import canonical, identity, pinned, require, source, strict
import form_hoi_external_dev as public

ROOT = Path('/srv/scenesmith/world-reward')
DEV_DATA = Path('/srv/world-reward-data/form_hoi_external_dev_v1')
ENTRY = 'run_form_hoi_external_eval'
CONFIG = 'configs/form_hoi_external_eval_v1.json'
HELPERS = ('infra/form_hoi_external_eval.py', 'infra/run_form_hoi_external_eval.sh', CONFIG,
    'infra/mediapipe_cpu_runtime_verify.py', 'infra/form_hoi_external_dev.py',
    'infra/form_hoi_external_acquire.py', 'src/world_reward/form_hoi_protocol.py',
    'configs/form_hoi_insight_v1.json', 'configs/form_hoi_external_dev_v1.json')
OFFICIAL = {
    'v2dlb/mhr_metrics.py': dict(bytes=25647, sha256='73077b65b5c3e0204307feef784317aab8c733d7462b6b8e6f4f397f7b91d2e0'),
    'v2dlb/mhr_submission.py': dict(bytes=24344, sha256='06fbd58d07bae1c583a92598f879771a4805ccea3e8346605975bcf5007b1fa3'),
    'v2dlb/mesh_common.py': dict(bytes=2175, sha256='aaeac13286c839a7d7e271888613875c934e46de094b2ac928ff2ff01c5a1f06'),
}
GEOMETRY_KEYS = {'human_vertices', 'human_joints', 'human_faces', 'object_vertices',
    'object_faces', 'object_rotation', 'object_translation', 'object_scale', 'frame_index'}
METRICS = ('cd_h_cm', 'cd_o_cm', 'acc_h_cm', 'acc_o_cm', 'interpenetration_cm')
JOINTS = (1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)


def seal_json(path, value):
    """Complete receipts survive subsequent failures; immutable exclusive write."""
    path = Path(path); temporary = path.with_name('.'+path.name+'.part')
    require(not path.exists() and not temporary.exists(), 'Never overwrite a prediction/result')
    try:
        with temporary.open('xb') as f:
            f.write((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())
            f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
        os.link(temporary, path); temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return identity(path, 1 << 20)


def array_sha(*arrays):
    digest = hashlib.sha256()
    for x in arrays:
        a = np.ascontiguousarray(x); digest.update(str(a.dtype).encode())
        digest.update(np.asarray(a.shape, np.int64).tobytes()); digest.update(a.tobytes())
    return digest.hexdigest()


def configuration(code):
    cfg = strict((code/CONFIG).read_bytes())
    require(cfg['schema']=='world_reward.form_hoi_external_eval.v1' and
        cfg['dataset_revision']=='c63db107e84c7f74bb4929ef643b67b5c8bcc00e' and
        cfg['original_frame_indices']==list(range(96)) and cfg['body_joint_indices']==list(JOINTS) and
        cfg['sampling']['body_alignment_count']==cfg['sampling']['object_count']==64 and
        cfg['sampling']['object_dense_count']==8192 and cfg['metrics']==list(METRICS) and
        cfg['camera']=='front_stereo_camera_left' and cfg['reserved_evaluated']==0 and
        cfg['kaggle_score_equivalence_claimed'] is False and cfg['production_adopted'] is False and
        cfg['training_overlap_verified'] is False and cfg['models_loaded'] is False and
        cfg['budget_seconds']==1500 and cfg['cpu_workers']==4,
        'Frozen first96 external operators/sampling protocol required')
    cohort = pinned(code/cfg['cohort_protocol'], cfg['cohort_protocol_identity'], 16 << 10)
    require(cohort['dataset_revision']==cfg['dataset_revision'], 'Frozen external source differs')
    return cfg, cohort


def artifact_record(record, *, maximum=1 << 30):
    require(type(record) is dict and set(record)=={'path','pin'}, 'Explicit independent path/pin required')
    path = canonical(record['path'])
    require(identity(path, maximum)==record['pin'], 'Sealed independent artifact identity differs')
    return path


def freeze_gate(cfg, cohort, code):
    """Hash every A/B geometry and report before opening ANY reference values."""
    gate=cfg['input_gate']; rev=gate['dev_producer_revision']
    require(type(rev) is str and re.fullmatch('[0-9a-f]{40}',rev), 'Actual qualified DEV producer pin pending')
    dev=DEV_DATA/rev; report=pinned(dev/'report.json',gate['dev_report_identity'],128 << 10)
    require(report['status']=='pass' and report['inference_ready'] is True and
        report['producer_revision']==rev and report['source_before']==report['source_after'] and
        all(report[k] is False for k in ('reference_arrays_decoded','reference_masks_decoded','source_calibration_decoded')) and
        report['reserved_acquired']==0, 'Original RGB-only DEV qualification must precede evaluation')
    expected={r['sequence_id'] for r in cohort['cohort'] if r['split']=='development'}
    require(len(expected)==4 and len(report['sequences'])==4 and
        {r['sequence_id'] for r in report['sequences']}==expected, 'All four fixed DEV; no reroll/reserved')
    manifest_path=canonical(gate['prediction_manifest_path'])
    manifest=pinned(manifest_path,gate['prediction_manifest_identity'],128 << 10)
    require(manifest['schema']=='world_reward.form_hoi_external_prediction_set.v1' and
        manifest['dataset_revision']==cfg['dataset_revision'] and manifest['split']=='development' and
        manifest['ground_truth_used'] is False and manifest['private_truth_read'] is False and
        manifest['original_frame_indices']==list(range(96)) and len(manifest['sequences'])==4 and
        {r['sequence_id'] for r in manifest['sequences']}==expected and
        re.fullmatch('[0-9a-f]{40}',manifest['producer_revision']), 'ALL four paired predictions must be frozen before reference access')
    acquired={r['sequence_id']:r for r in report['sequences']}; rows=[]
    current_public_cfg=strict((code/'configs/form_hoi_external_dev_v1.json').read_bytes())
    for row in manifest['sequences']:
        sid=row['sequence_id']; receipt=pinned(dev/sid/'receipt.json',acquired[sid]['receipt'],128 << 10)
        require(receipt['inference_ready'] is True and receipt['sequence_id']==sid and
            receipt['producer_revision']==rev and receipt['reference_arrays_decoded'] is False and
            receipt['reference_masks_decoded'] is False and receipt['source_calibration_decoded'] is False and
            receipt['alias_guard']['qualified'] is True and
            receipt['alias_guard']['exact_content_duplicate_check_passed'] is True,
            'Independent source/alias qualification required for every external sequence')
        require(set(row)=={'sequence_id','input','variants'} and set(row['variants'])=={'A','B'}, 'Exactly paired complete A/B output required')
        inp=artifact_record(row['input'],maximum=16 << 10)
        require(inp==dev/sid/'inputs/input.json' and row['input']['pin']==receipt['input'], 'Prediction public input lineage differs')
        package=strict(inp.read_bytes()); public.validate_public_package(package,current_public_cfg,require_ready=True)
        require(package['sequence_id']==sid and package['video']==str(inp.parent/'rgb.mp4') and
            package['video_pin']==next({k:x[k] for k in ('bytes','sha256')} for x in receipt['retained'] if x['role']=='rgb'),
            'Original native RGB identity differs')
        variants={}
        for variant, data in row['variants'].items():
            require(set(data)=={'geometry','report'}, 'Prediction geometry/report pins only')
            geometry=artifact_record(data['geometry']); result_path=artifact_record(data['report'],maximum=1 << 20)
            require(all(not p.is_relative_to(dev) and p.is_relative_to(ROOT) and p.is_relative_to(manifest_path.parent) and
                'eval_private' not in p.parts for p in (geometry,result_path)), 'Predictions must be separate public-only producer outputs')
            result=strict(result_path.read_bytes())
            require(result['status']=='complete' and result['ground_truth_used'] is False and
                result['private_truth_read'] is False and result['sequence_id']==sid and
                result['producer_revision']==manifest['producer_revision'] and
                result.get('geometry',result.get('artifacts',{}).get('eval_geometry.npz'))==data['geometry']['pin'] and
                result['original_frame_indices']==list(range(96)) and result.get('full_predictions_sealed_before_evaluation') is True and
                result.get('reference_inputs_mounted') is False and result.get('oracle_modes')==[], 'Literal complete sealed full96 prediction report required')
            variants[variant]=geometry
        rows.append(dict(sequence_id=sid,package=package,receipt=receipt,variants=variants,prediction_records=row['variants'],private=dev/sid/'eval_private'))
    return rows, dict(all_four_paired_predictions_sealed=True, dev_report=gate['dev_report_identity'],
        prediction_manifest=gate['prediction_manifest_identity'], prediction_producer_revision=manifest['producer_revision'],
        reference_values_read=False, exact_content_guard_only=True, near_alias_absence_verified=False)


def mesh_arrays(vertices, faces):
    v=np.asarray(vertices); f=np.asarray(faces)
    require(v.dtype.kind=='f' and v.ndim==2 and v.shape[1:]==(3,) and len(v)>=3 and np.isfinite(v).all() and
        f.dtype.kind in 'iu' and f.ndim==2 and f.shape[1:]==(3,) and len(f)>0 and
        f.min()>=0 and f.max()<len(v), 'Finite original triangle mesh; no repair/deletion allowed')
    require(np.any(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)>0), 'Mesh must have nonzero original surface')
    return v.astype(np.float64),f.astype(np.int64)


def rotation_arrays(r, t, count):
    r=np.asarray(r,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    require(r.shape==(count,3,3) and t.shape==(count,3) and np.isfinite(r).all() and np.isfinite(t).all() and
        np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-5,rtol=0) and
        np.allclose(np.linalg.det(r),1,atol=1e-5,rtol=0), 'Proper literal poses; no SO3 repair, gaps or freeze')
    return r,t


def geometry_arrays(values, *, count=96):
    require(set(values)==GEOMETRY_KEYS, 'Exact sealed evaluation geometry schema required')
    h=np.asarray(values['human_vertices']);j=np.asarray(values['human_joints']);grid=np.asarray(values['frame_index'])
    require(h.dtype.kind=='f' and h.shape==(count,18439,3) and j.dtype.kind=='f' and j.shape==(count,127,3) and
        np.isfinite(h).all() and np.isfinite(j).all() and grid.dtype==np.int64 and
        np.array_equal(grid,np.arange(count,dtype=np.int64)), 'Original full96 native MHR vertices/jointcoords127 required')
    _,hf=mesh_arrays(h[0],values['human_faces']);v,f=mesh_arrays(values['object_vertices'],values['object_faces'])
    r,t=rotation_arrays(values['object_rotation'],values['object_translation'],count)
    s=np.asarray(values['object_scale']); require(s.shape==() and s.dtype.kind=='f' and np.isfinite(s) and s>0, 'One fixed positive object scale; no inferred shrink')
    return dict(human_vertices=h.astype(np.float64),human_joints=j.astype(np.float64),human_faces=hf,
        object_vertices=v,object_faces=f,object_rotation=r,object_translation=t,object_scale=float(s),frame_index=grid)


def read_geometry(path):
    with np.load(path,allow_pickle=False) as z: return geometry_arrays({k:z[k] for k in z.files})


def anatomical_hands(spec, faces, cfg):
    """Pinned CARI neutral-anatomy samples, not hidden scorer vertex roles."""
    keys=('vertex_indices','sample_local_indices','sample_assignments','hand_faces_left','hand_faces_right')
    for key,shape in zip(keys,((2,2318),(2,256),(2,2318),(4603,3),(4603,3))):
        require(spec[key].dtype==np.int32 and spec[key].shape==shape,'Exact original hand anatomy ABI required')
    p=cfg['hand_spec'];ids=spec['vertex_indices'];samples=spec['sample_local_indices']
    require(ids.min()>=0 and ids.max()<18439 and len(np.unique(ids))==ids.size and
        samples.min()>=0 and samples.max()<2318 and all(len(np.unique(s))==256 for s in samples) and
        array_sha(faces.astype(np.int32))==p['faces_sha256']==str(spec['faces_sha256'].item()) and
        array_sha(faces.astype(np.int32),*(spec[k] for k in keys))==p['topology_sha256']==str(spec['topology_sha256'].item()) and
        str(spec['mhr_model_sha256'].item())==p['mhr_model_sha256'], 'Actual hand topology/model digest required')
    return ids,ids[np.arange(2)[:,None],samples].reshape(-1).astype(np.int64)


def farthest_vertices(points, indices, count):
    """Baseline-only geometric FPS; stable native-index tie break, no GT."""
    p=np.asarray(points,np.float64);idx=np.asarray(indices,np.int64)
    require(p.ndim==2 and p.shape[1:]==(3,) and np.isfinite(p).all() and idx.ndim==1 and
        len(idx)>=count and np.all(np.diff(idx)>0) and idx.min()>=0 and idx.max()<len(p), 'Sorted complete candidate anatomy required')
    q=p[idx];first=int(np.argmax(np.sum((q-q.mean(0))**2,axis=1)))
    selected=[first];distance=np.sum((q-q[first])**2,axis=1);distance[first]=-np.inf
    for _ in range(1,count):
        slot=int(np.argmax(distance));selected.append(slot)
        distance=np.minimum(distance,np.sum((q-q[slot])**2,axis=1));distance[selected]=-np.inf
    return idx[np.asarray(selected)]


def sampling_roles(baseline, spec, cfg):
    hands, sampled=anatomical_hands(spec,baseline['human_faces'],cfg)
    body=np.setdiff1d(np.arange(18439,dtype=np.int64),hands.reshape(-1),assume_unique=True)
    ids=farthest_vertices(baseline['human_vertices'][0],body,cfg['sampling']['body_alignment_count'])
    return dict(body=ids,alignment=ids.copy(),hands=sampled)


def front_camera(edex, package):
    """Native EDEX is [header,body], transforms CAMERA->WORLD; no index0 guess."""
    require(type(edex) is list and len(edex)==2 and all(type(x) is dict for x in edex), 'Native EDEX [header,body] required')
    header,body=edex;cameras=header['cameras'];sequence=body['sequence']
    require(header['frame_start']==0 and header['frame_end']==package['full_source_frames'] and
        type(cameras) is list and len(cameras)==len(sequence)>0 and all(type(p) is str for p in sequence),
        'Reference original untrimmed frame range and camera/path map required')
    matches=[i for i,p in enumerate(sequence) if package['camera'] in PurePosixPath(p).parts]
    require(len(matches)==1, 'Native front camera path must identify exactly one camera, never guess')
    camera=cameras[matches[0]];e=np.asarray(camera['transform'],np.float64)
    require(e.shape==(3,4),'Native camera->world transform[3,4] required')
    rotation_arrays(e[:,:3][None],e[:,3][None],1)
    intr=camera['intrinsics'];size=intr.get('resolution',intr.get('size'))
    focal=np.asarray(intr['focal'],np.float64);principal=np.asarray(intr['principal'],np.float64)
    require(list(size)==[package['width'],package['height']] and focal.shape==principal.shape==(2,) and
        np.isfinite(focal).all() and (focal>0).all() and np.isfinite(principal).all(), 'Reference camera grid differs')
    transform=np.eye(4);transform[:3]=e
    K=np.array([[focal[0],0,principal[0]],[0,focal[1],principal[1]],[0,0,1]],np.float64)
    return np.linalg.inv(transform),K,matches[0]


def reference_geometry(mesh, params, poses, object_vertices, object_faces, edex, package):
    require(type(mesh) is dict and set(mesh)=={'pred_vertices','faces'} and type(params) is dict and
        'pred_joint_coords' in params, 'Source-authored native multiview MHR fields required')
    def cpu(x):
        if hasattr(x,'detach'):return x.detach().cpu().numpy()
        return np.asarray(x)
    h=cpu(mesh['pred_vertices']);j=cpu(params['pred_joint_coords']);poses=np.asarray(poses)
    n=package['full_source_frames'];require(h.shape==(n,18439,3) and j.shape==(n,127,3) and poses.shape==(n,4,4) and
        np.isfinite(poses).all() and np.allclose(poses[:,3],[0,0,0,1],atol=1e-7,rtol=0), 'Reference full native timeline; no trimming or root-relative guessing')
    rotation_arrays(poses[:,:3,:3],poses[:,:3,3],n);world_to_camera,K,index=front_camera(edex,package)
    Q,b=world_to_camera[:3,:3],world_to_camera[:3,3];count=package['total']
    # Saved native outputs already in metres with Y/Z camera-convention flip.
    # Undo neither that flip nor an imagined pred_cam_t: exported verts/joints are world coordinates.
    values=dict(human_vertices=h[:count]@Q.T+b,human_joints=j[:count]@Q.T+b,human_faces=cpu(mesh['faces']),
        object_vertices=object_vertices,object_faces=object_faces,
        object_rotation=Q[None]@poses[:count,:3,:3],object_translation=poses[:count,:3,3]@Q.T+b,
        object_scale=np.array(1.,np.float64),frame_index=np.arange(count,dtype=np.int64))
    return geometry_arrays(values,count=count),dict(camera_index=index,source_camera_eval_only=True,
        camera_to_world_inverted_once=True,YZ_flip_applied_again=False,world_units='metres',K=K.tolist(),
        reference_full_frames=n,scored_original_frames=count,failure_segments_used=False,interaction_trim_used=False)


def reference_paths(row):
    found={}
    for r in row['receipt']['retained']:
        if r['role']!='reference':continue
        path=canonical(row['private'].parent/r['file'])
        require(path.is_relative_to(row['private']) and identity(path)=={k:r[k] for k in ('bytes','sha256')}, 'Original quarantined reference identity differs')
        name=path.name
        if name in ('mhr_mesh_mv.pt','mhr_params_mv.pt','poses.npy','edex','output_aligned.glb'):
            require(name not in found,'Ambiguous native reference member');found[name]=path
    require(set(found)=={'mhr_mesh_mv.pt','mhr_params_mv.pt','poses.npy','edex','output_aligned.glb'}, 'Complete native external references required')
    return found


def load_reference(row, proof):
    require(proof.get('all_four_paired_predictions_sealed') is True and
        proof.get('all_four_sampling_frozen_before_reference') is True,'Reference values forbidden until entire prediction/sampler freeze')
    paths=reference_paths(row)
    import torch
    import trimesh
    # No unsafe pickle fallback. All external native references are weights_only tensors/dicts.
    mesh=torch.load(paths['mhr_mesh_mv.pt'],map_location='cpu',weights_only=True)
    params=torch.load(paths['mhr_params_mv.pt'],map_location='cpu',weights_only=True)
    poses=np.load(paths['poses.npy'],allow_pickle=False)
    scene=trimesh.load(paths['output_aligned.glb'],force='scene',process=False)
    vs=[];fs=[];offset=0
    for node in sorted(scene.graph.nodes_geometry):
        matrix,name=scene.graph[node];geometry=scene.geometry[name]
        v=np.asarray(geometry.vertices,np.float64);f=np.asarray(geometry.faces,np.int64)
        matrix=np.asarray(matrix,np.float64)
        require(matrix.shape==(4,4) and np.isfinite(matrix).all() and np.allclose(matrix[3],[0,0,0,1]), 'Literal GLB scene transform required')
        v=v@matrix[:3,:3].T+matrix[:3,3];v,f=mesh_arrays(v,f)
        vs.append(v);fs.append(f+offset);offset+=len(v)
    require(vs,'Native reference GLB surface absent')
    target,meta=reference_geometry(mesh,params,poses,np.concatenate(vs),np.concatenate(fs),strict(paths['edex'].read_bytes()),row['package'])
    meta.update(mesh_vertices=offset,mesh_faces=len(target['object_faces']),surface_repaired=False,GT_private_decode_after_ALL_predictions=True)
    return target,meta


def verify_frozen_predictions(rows):
    for row in rows:
        for record in row['prediction_records'].values():
            artifact_record(record['geometry']); artifact_record(record['report'],maximum=1 << 20)


def official_modules(root):
    kit=canonical(root/'vendor/v2d_submission_kit')
    for name,pin in OFFICIAL.items():require(identity(kit/name,1 << 20)==pin,'Original public metric source pin differs')
    # The participant kit has no package initializer. Do not execute a foreign
    # unpinned __init__.py while importing the three verified public modules.
    import types
    if 'v2dlb' not in sys.modules:
        namespace=types.ModuleType('v2dlb');namespace.__path__=[str(kit/'v2dlb')]
        sys.modules['v2dlb']=namespace
    else:
        require(list(sys.modules['v2dlb'].__path__)==[str(kit/'v2dlb')], 'Foreign metric package forbidden')
    modules=[importlib.import_module('v2dlb.'+name) for name in ('mhr_metrics','mhr_submission','mesh_common')]
    for module,name in zip(modules,('mhr_metrics','mhr_submission','mesh_common')):
        require(Path(module.__file__).resolve()==kit/('v2dlb/'+name+'.py'),'Actual unchanged public metric import required')
    require(tuple(modules[0].MHR_TABLE3_BODY_JOINT_INDICES)==JOINTS, 'Published native22 joint order differs')
    try:
        import numba
        numba.set_num_threads(4)
    except ImportError:pass
    return modules


def object_seed(sequence_id):
    return int.from_bytes(hashlib.sha256(sequence_id.encode()).digest()[:8],'little')


def metric_arrays(geometry, roles, sampler, sequence_id, *, count=64):
    local=sampler(geometry['object_vertices'],geometry['object_faces'],count,object_seed(sequence_id))
    surface=geometry['object_scale']*np.einsum('tij,pj->tpi',geometry['object_rotation'],local)+geometry['object_translation'][:,None]
    arrays=dict(mhr_vertices=geometry['human_vertices'][:,roles['alignment']],mhr_joints=geometry['human_joints'],
        human_surface_points=geometry['human_vertices'][:,roles['body']],object_surface_points=surface,
        object_translation=geometry['object_translation'])
    scene=dict(hands=geometry['human_vertices'][:,roles['hands']],mesh_vertices=geometry['object_vertices'],
        mesh_faces=geometry['object_faces'],rotation=geometry['object_rotation'],translation=geometry['object_translation'],scale=geometry['object_scale'])
    return arrays,scene


def five_metrics(pred, target, roles, modules, sequence_id):
    _,submission,common=modules
    a,scene=metric_arrays(pred,roles,common.sample_mesh_surface,sequence_id)
    b,_=metric_arrays(target,roles,common.sample_mesh_surface,sequence_id)
    measured=submission.episode_metrics(a,b,pred['frame_index'],JOINTS)
    measured['interpenetration_cm']=submission.episode_penetration(a,scene,b)
    require(set(measured)==set(METRICS)|{'cd_c_cm'} and all(np.isfinite(x) and x>=0 for x in measured.values()), 'All unchanged public metrics must be finite')
    return measured


def summary(a):
    a=np.asarray(a,np.float64);require(a.size>0 and np.isfinite(a).all(),'Finite diagnostic series required')
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def full_geometry_diagnostics(pred,target,roles,modules,sequence_id):
    """All native vertices / dense object sample; no fresh alignment or fitting."""
    metrics,_,common=modules
    transform=metrics.fit_similarity_transform(pred['human_vertices'][0,roles['alignment']],target['human_vertices'][0,roles['alignment']])
    hp=metrics.apply_similarity_transform(pred['human_vertices'],transform)
    op,_=metric_arrays(pred,roles,common.sample_mesh_surface,sequence_id,count=8192)
    ot,_=metric_arrays(target,roles,common.sample_mesh_surface,sequence_id,count=8192)
    op=metrics.apply_similarity_transform(op['object_surface_points'],transform);ot=ot['object_surface_points']
    from scipy.spatial import cKDTree
    def symmetric(a,b):
        return cKDTree(a).query(b,k=1,eps=0,workers=1)[0].mean()+cKDTree(b).query(a,k=1,eps=0,workers=1)[0].mean()
    cd_h=[100*symmetric(a,b) for a,b in zip(hp,target['human_vertices'])]
    cd_o=[100*symmetric(a,b) for a,b in zip(op,ot)]
    pve=100*np.linalg.norm(hp-target['human_vertices'],axis=-1).mean(axis=1)
    transformed_translation=metrics.apply_similarity_transform(pred['object_translation'],transform)
    return dict(all18439_vertex_chamfer_cm=summary(cd_h),dense8192_object_point_chamfer_cm=summary(cd_o),
        corresponding_fullbody_vertex_error_cm=summary(pve),object_translation_error_cm=summary(100*np.linalg.norm(transformed_translation-target['object_translation'],axis=1)),
        alignment_scale=float(transform[0]),continuous_surface_chamfer_claimed=False,per_frame_alignment=False)


def motion_coverage(geometry,roles):
    """All96 coverage only; never select scored/contact frames from these values."""
    hands=geometry['human_vertices'][:,roles['hands']]
    local=np.einsum('tpi,tij->tpj',hands-geometry['object_translation'][:,None],geometry['object_rotation'])/geometry['object_scale']
    lo=geometry['object_vertices'].min(0);hi=geometry['object_vertices'].max(0)
    bbox=np.any(((local>=lo)&(local<=hi)).all(axis=2),axis=1)
    return dict(object_translation_step_m=summary(np.linalg.norm(np.diff(geometry['object_translation'],axis=0),axis=1)),
        human_root_joint_step_m=summary(np.linalg.norm(np.diff(geometry['human_joints'][:,0],axis=0),axis=1)),
        hand_bbox_overlap_frames=int(bbox.sum()),all96_frames_scored=True,
        bbox_overlap_is_contact_truth=False,PEN_may_be_uninformative_when_no_interaction=True)


def require_fresh_output(out):
    out=canonical(out)
    require(out.is_dir() and not out.is_symlink(), 'Owned evaluator directory required')
    entries=list(out.iterdir())
    require(not entries or len(entries)==1 and entries[0].name=='.container.cid', 'Fresh output except actual docker CID marker required')
    if entries:
        marker=entries[0];pin=identity(marker,100,readonly=False)
        require(re.fullmatch(b'[0-9a-f]{64}\n?',marker.read_bytes()), 'Actual bounded docker CID marker required')
    return out


def run():
    code=canonical(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION'];started=time.monotonic()
    require(code==ROOT/'jobs'/revision/ENTRY/'code','Exact immutable external evaluator namespace required')
    out=ROOT/'results'/('form-hoi-external-eval-'+revision)
    require_fresh_output(out)
    cfg,cohort=configuration(code);binding=source(ROOT,code,revision,ENTRY,HELPERS)
    result=dict(schema='world_reward.form_hoi_external_eval_report.v1',status='fail',producer_revision=revision,
        source_before=binding,config_identity=identity(code/CONFIG,32 << 10),scope=cfg['scope'],
        reference_kind=cfg['reference_kind'],kaggle_scores=False,kaggle_score_equivalence_claimed=False,
        training_overlap_verified=False,reserved_evaluated=0,production_adopted=False,models_loaded=False,sequences=[])
    try:
        rows,proof=freeze_gate(cfg,cohort,code);modules=official_modules(ROOT)
        hand_path=ROOT/cfg['hand_spec']['path'];require(identity(hand_path)=={k:cfg['hand_spec'][k] for k in ('bytes','sha256')},'Pinned public hand spec required')
        with np.load(hand_path,allow_pickle=False) as z:spec={k:z[k] for k in z.files}
        # Finish and seal the entire baseline-only role selection before opening any GT values.
        prepared=[];all_roles={}
        for row in rows:
            a=read_geometry(row['variants']['A']);b=read_geometry(row['variants']['B'])
            require(np.array_equal(a['human_faces'],b['human_faces']) and np.array_equal(a['object_vertices'],b['object_vertices']) and
                np.array_equal(a['object_faces'],b['object_faces']) and a['object_scale']==b['object_scale'], 'Paired full shared geometry must not change shape, scale or topology')
            roles=sampling_roles(a,spec,cfg)
            all_roles[row['sequence_id']]={k:v.tolist() for k,v in roles.items()}
            prepared.append((row,a,b,roles))
        result['sampler_receipt']=seal_json(out/'sampling.json',dict(schema='world_reward.form_external_sampling.v1',
            rule=cfg['sampling'],indices=all_roles,reference_values_read=False,freeze_gate=proof))
        verify_frozen_predictions(rows)
        proof['all_four_sampling_frozen_before_reference']=True;result['freeze_gate']=proof
        for row,a,b,roles in prepared:
            sid=row['sequence_id'];target,reference=load_reference(row,proof)
            require(np.array_equal(a['human_faces'],target['human_faces']), 'Native reference topology correspondence differs; no nearest remapping')
            measured={};entry=dict(sequence_id=sid,reference_schema=reference,variants={})
            for name,pred in (('A',a),('B',b)):
                start=time.monotonic();scores=five_metrics(pred,target,roles,modules,sid)
                receipt=dict(sequence_id=sid,variant=name,metrics=scores,elapsed_seconds=time.monotonic()-start,
                    scope=cfg['scope'],motion_coverage=motion_coverage(pred,roles),reference_motion_coverage=motion_coverage(target,roles))
                pin=seal_json(out/(sid+'-'+name+'-metrics.json'),receipt)
                entry['variants'][name]=dict(receipt=pin,metrics=scores);measured[name]=scores
                print(json.dumps(dict(stage='external_reference_metrics',sequence_id=sid,variant=name,complete_metrics=5,elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
                diagnostic=full_geometry_diagnostics(pred,target,roles,modules,sid)
                entry['variants'][name]['full_geometry_diagnostic']=seal_json(out/(sid+'-'+name+'-geometry.json'),diagnostic)
            entry['B_minus_A']={k:measured['B'][k]-measured['A'][k] for k in measured['A']}
            result['sequences'].append(entry)
            seal_json(out/(sid+'-paired.json'),entry)
        result['pooled']={name:{key:float(np.mean([r['variants'][name]['metrics'][key] for r in result['sequences']])) for key in (*METRICS,'cd_c_cm')} for name in ('A','B')}
        result['pooled']['B_minus_A']={key:result['pooled']['B'][key]-result['pooled']['A'][key] for key in (*METRICS,'cd_c_cm')}
        verify_frozen_predictions(rows)
        result['prediction_geometry_rehashed_after']=True
        result['source_after']=source(ROOT,code,revision,ENTRY,HELPERS);require(result['source_after']==binding,'Immutable evaluator source changed')
        result['status']='pass';result['decision']='EXTERNAL_DEV_PAIRED_MEASUREMENT_ONLY_not_verified_CARI4D_or_Kaggle_victory'
    except Exception as error:
        result['error_type']=type(error).__name__
        if isinstance(error,ValueError):result['error_context']=str(error)[:400]
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['completed_metric_receipts']=sorted(p.name for p in out.glob('*-metrics.json'))
        seal_json(out/'report.json',result)
    return result


def main():
    require(sys.platform=='linux' and os.environ.get('WR_ROOT')==str(ROOT) and len(sys.argv)==1,'Azure offline CPU evaluation only; no data/frame override')
    code=canonical(os.environ['WR_CODE']);cfg,_=configuration(code)
    def timeout(*_):raise TimeoutError('Inclusive external evaluation budget')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(cfg['budget_seconds'])
    value=run();signal.alarm(0)
    print(json.dumps({k:value[k] for k in ('status','elapsed_seconds','completed_metric_receipts')}),flush=True)
    return 0 if value['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())

"""Azure saved-only RGBD ablation: one fixed mesh, gauge and material tracks.

Fresh manufactured DEV/RESERVED checks precede any challenge artifact access.
MoGe camera-axis depth is inferred evidence, never truth or metric calibration.
Outputs stay isolated; neither the original nor rejected RGB-only fit is replaced.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from world_reward.sequence_pose import RGBDepthConfig, refine_sequence
import sequence_pose_probe as old
from mediapipe_cpu_runtime_verify import source, strict

ROOT = old.ROOT
ENTRY = 'run_sequence_pose_rgbd_probe'
TRACK_SOURCE = '09f516d9085f0d64d725b46b0ad6aa52fe7c0d84'
TRACK_PIN = dict(bytes=153407, sha256='e5cdd5195b9f72befbf50169e2b8cf81fa8720460e585e953f2855d1f3d8bcbe')
TRACK_REPORT_PIN = dict(bytes=12172, sha256='6565b56013e43789d39f9f1188766b72eb32961f126c222337fc98630f049e3c')
STAGE_PINS = {
    'scale_smoke':dict(bytes=9808,sha256='779aed6cf26f86dfdfe7e33da5380abfaa689640a51aa79ea7e249b31dfcfedc'),
    'depth_full':dict(bytes=628821,sha256='2b0a9bcafac2c8d38c1e1df039b315ac8392121838e3f4a98e542584f36a28a2'),
    'body_full':dict(bytes=269937,sha256='85dc26bc68d4b072502e60f8c205dff606432c49acdc8052a9d501af2b82400b')}
DEPTH_CFG = RGBDepthConfig(.02, 'manufactured_RGBD_DEV_20261029_RESERVED_20261030_v1_not_real_calibration')
HELPERS = ('infra/sequence_pose_rgbd_probe.py', 'infra/run_sequence_pose_rgbd_probe.sh', *old.HELPERS)


def bilinear_depth(depth, valid, object_mask, xy, rgb_visible, scale, shared_scale):
    """Same original-grid material query; all four depth/mask neighbours required."""
    z, flags, mask, points, visible = map(np.asarray, (depth, valid, object_mask, xy, rgb_visible))
    scale = np.asarray(scale)
    if (z.ndim != 2 or z.dtype.kind != 'f' or flags.dtype != bool or flags.shape != z.shape
            or mask.dtype != bool or mask.shape != z.shape or points.ndim != 2 or points.shape[1] != 2
            or points.dtype.kind != 'f' or visible.dtype != bool or visible.shape != (len(points),)
            or not np.isfinite(points[visible]).all() or not np.isnan(points[~visible]).all()
            or scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any()
            or type(shared_scale) not in (int, float) or not np.isfinite(shared_scale) or shared_scale <= 0):
        raise ValueError('Explicit original depth/grid/RGB support and one finite shared scalar required')
    out = np.full(len(points), np.nan); supported = np.zeros(len(points), bool); h, w = z.shape
    original = (points[visible]+.5)/scale-.5; ids = np.flatnonzero(visible)
    inside = (original[:, 0] >= 0) & (original[:, 0] < w-1) & (original[:, 1] >= 0) & (original[:, 1] < h-1)
    ids, original = ids[inside], original[inside]
    x, y = np.floor(original).astype(int).T; fx, fy = (original-np.floor(original)).T
    values = np.stack([z[y+dy, x+dx] for dx, dy in ((0,0),(1,0),(0,1),(1,1))], 1)
    good = np.isfinite(values).all(1) & (values > 0).all(1)
    for dx, dy in ((0,0),(1,0),(0,1),(1,1)): good &= flags[y+dy, x+dx] & mask[y+dy, x+dx]
    weights = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), 1)
    metric = (values[good]*weights[good]).sum(1)*shared_scale
    if not np.isfinite(metric).all() or (metric <= 0).any(): raise ValueError('Shared-scale axial depth overflow')
    out[ids[good]] = metric; supported[ids[good]] = True
    return out, supported


def manufactured_gate(seed, reserved, config):
    """Independent owned RGB/axial-motion fixtures; truth enters scoring only."""
    rng = np.random.default_rng(seed); rows = []
    for case in ('static', 'moving', 'axial_motion'):
        n = 24; u = np.arange(n)/30; v = rng.uniform(-.12,.12,(24,3)); k = np.array([[256.,0,128],[0,256.,128],[0,0,1.]])
        t = np.column_stack((u*0,u*0,2.+u*0)); r = np.broadcast_to(np.eye(3),(n,3,3)).copy()
        if case == 'moving': t[:,0] = .15*u; r = Rotation.from_rotvec(np.column_stack((u*0,.3*u,u*0))).as_matrix()
        if case == 'axial_motion': t[:,2] += .25*u
        prior_t = t+rng.normal(0,.01,t.shape); prior_r = Rotation.from_rotvec(rng.normal(0,.015,(n,3))).as_matrix()@r
        prior_t[0] = t[0]; prior_r[0] = r[0]
        xy = old.project(v,r,t,k)+rng.normal(0,.35 if not reserved else .5,(n,len(v),2))
        xyz = v[None]@r.swapaxes(-1,-2)+t[:,None]; depth = xyz[...,2]+rng.normal(0,.003,xyz.shape[:2])
        visible = np.ones(depth.shape,bool)
        fit = refine_sequence(v,v,xy,visible,prior_r,prior_t,np.ones(n,bool),k,np.arange(n),30,config,
            tracks_depth_m=depth,depth_visible=visible,depth_config=DEPTH_CFG)
        truth = xyz; before = v[None]@prior_r.swapaxes(-1,-2)+prior_t[:,None]
        after = v[None]@fit.rotations.swapaxes(-1,-2)+fit.translations[:,None]
        before_error = float(np.linalg.norm(before-truth,axis=-1).mean()); after_error = float(np.linalg.norm(after-truth,axis=-1).mean())
        motion = old.motion_gate(truth,before,after)
        rows.append(dict(case=case,old_error_m=before_error,new_error_m=after_error,full_motion=motion,
            passed=after_error <= .9*before_error and motion['passed'],converged=fit.diagnostics['converged']))
    return dict(seed=seed,reserved=reserved,cases=rows,passed=all(r['passed'] for r in rows),
        converged=all(r['converged'] for r in rows),scope='manufactured_numeric_not_real_accuracy_or_depth_calibration')


def load_npz(ledger, path, pin):
    ledger.record(path,pin)
    with np.load(path,allow_pickle=False) as arrays: return {k:arrays[k] for k in arrays.files}


def stage_report(ledger, path, stage, input_sha, count=None):
    row = strict(ledger.read(path))
    expected = dict(stage=stage,status='pass',episode_index=9,input_track='track_1',input_sha256=input_sha,
                    ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    if count is not None: expected['total_video_frames'] = count
    if any(type(row.get(k)) is not type(v) or row[k] != v for k,v in expected.items()):
        raise ValueError('Exact original video-only depth/gauge stage required')
    return row


def original_depth(ledger, base, tracks, visible, scale, K, inventory, input_sha):
    """Auth SHA/RGB/shared-camera chain before any finite-depth observation."""
    from PIL import Image
    if (type(tracks) is not np.ndarray or tracks.ndim != 3 or tracks.shape[2] != 2 or tracks.dtype.kind != 'f'
            or type(visible) is not np.ndarray or visible.dtype != bool or visible.shape != tracks.shape[:2]):
        raise ValueError('Original full-T fixed material tracks required')
    n = len(tracks); alignment = stage_report(ledger,base/'scale_smoke/report.json','predicted_human_anchored_moge2_pointmaps',input_sha)
    depth_report = stage_report(ledger,base/'depth_full/report.json','monocular_moge2_full_video',input_sha,n)
    body = stage_report(ledger,base/'body_full/report.json','sam3d_body_full_video_initializer',input_sha,n)
    body_rows = body['frames']; records = depth_report['frames']
    if ([r['frame_index'] for r in records] != list(range(n)) or [r['frame_index'] for r in body_rows] != list(range(n))):
        raise ValueError('Depth/body must retain complete original indices')
    scalar = alignment['depth_alignment']['shared_scale']; measured = np.full(visible.shape,np.nan); support = np.zeros(visible.shape,bool)
    for i, record in enumerate(records):
        if (record['decoded_rgb_sha256'] != body_rows[i]['decoded_rgb_sha256']
                or any(row['decoded_rgb_sha256'] != record['decoded_rgb_sha256']
                       for row in alignment.get('human_evidence',[]) if row['frame_index']==i)):
            raise ValueError('Depth/body/alignment original RGB SHA differs')
        path = base/f'depth_full/{i:06d}.npz'; ledger.record(path)
        if ledger.records[str(path)]['sha256'] != record['output_sha256']: raise ValueError('Original depth NPZ SHA differs')
        with np.load(path,allow_pickle=False) as a:
            if set(a.files) != {'depth','mask','intrinsics','frame_index'} or a['frame_index'].ndim != 0 or int(a['frame_index']) != i:
                raise ValueError('Original axial-depth arrays/index required')
            depth, valid, intrinsic = a['depth'],a['mask'],a['intrinsics']
        if depth.ndim != 2 or depth.dtype.kind != 'f' or valid.dtype != bool or valid.shape != depth.shape:
            raise ValueError('Original inferred depth/validity grid differs')
        h,w = depth.shape; masks=[]
        for role in (0,1):
            path = base/f'automatic_masks/masks/{role}/{i:06d}.png'; ledger.record(path,inventory[f'{role}/{i:06d}.png'])
            with Image.open(path) as image:
                if image.mode != 'L' or image.size != (w,h): raise ValueError('Original binary mask/depth grid differs')
                pixels = np.asarray(image)
                if not np.isin(pixels,[0,255]).all(): raise ValueError('Literal native binary mask required')
                masks.append(pixels > 0)
        mask = masks[1] & ~masks[0]  # Overlapping hands/body depth is not object-material evidence.
        if (intrinsic.shape != (3,3) or not np.isfinite(intrinsic).all()
                or not np.allclose(np.diag([w,h,1])@intrinsic,K,atol=1e-3,rtol=1e-6)
                or not np.array_equal(scale,np.array([640/w,480/h]))):
            raise ValueError('One unchanged original K and saved RGB query scaling required')
        measured[i],support[i] = bilinear_depth(depth,valid,mask,tracks[i],visible[i],scale,scalar)
    return measured,support


def source_track_report(ledger, path):
    """Episode 9 succeeded in a failed cohort; never relabel 14's texture failure."""
    row = strict(ledger.read(path,TRACK_REPORT_PIN)); episodes = row.get('episodes',[])
    if (row.get('producer_revision') != TRACK_SOURCE or row.get('profile') != 'v2'
            or row.get('numerical_source') != old.SOURCE or row.get('status') != 'fail'
            or row.get('ground_truth_used') is not False or row.get('manual_labels') is not False
            or row.get('full_4D_export_replaced') is not False or row.get('model_calls') != 0
            or len(episodes) != 1 or episodes[0].get('episode') != 9
            or episodes[0].get('frames') != 415 or episodes[0].get('points') != 33
            or episodes[0].get('output') != TRACK_PIN):
        raise ValueError('Exact rejected RGB-only source report/9 witnesses required')
    return row


def residuals(points,r,t,K,xy,visible,depth,depth_visible):
    xyz = points[None]@r.swapaxes(-1,-2)+t[:,None]; projected = old.project(points,r,t,K)
    return dict(RGB_mean_px=float(np.linalg.norm(projected[visible]-xy[visible],axis=1).mean()),
        axial_depth_mean_m=float(np.abs(xyz[...,2][depth_visible]-depth[depth_visible]).mean()),
        RGB_observations=int(visible.sum()),depth_observations=int(depth_visible.sum()))


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if ROOT != Path(os.environ['WR_ROOT']) or code != ROOT/'jobs'/revision/ENTRY/'code': raise ValueError('Exact Azure source required')
    out = ROOT/'results'/('sequence-pose-rgbd-'+revision); old.fresh_runtime_output(out)
    ledger = old.ArtifactLedger(); report = dict(status='fail',producer_revision=revision,numerical_source=old.SOURCE,
        ground_truth_used=False,manual_labels=False,model_calls=0,full_4D_export_replaced=False,depth_config=asdict(DEPTH_CFG))
    try:
        binding = source(ROOT,code,revision,ENTRY,HELPERS); report['source_binding'] = binding
        for name in HELPERS: ledger.record(code/name)
        config,_ = old.profile_config(code,'v2'); report['config'] = asdict(config)
        for seed,reserved,name in ((20261029,False,'development'),(20261030,True,'reserved')):
            gate = manufactured_gate(seed,reserved,config); report[name] = gate
            if not gate['passed']: raise ValueError('Fresh external RGBD '+name+' gate rejected; no challenge access')
        front,rows = old.saved_frontend(ledger); experiment = ROOT/'experiments'/('full4d-v1-'+old.SOURCE)
        transport = old.saved_numerical_frontend(ledger,experiment,rows)[9]; base = experiment/'outputs/episode_000009'
        pins = strict(ledger.read(experiment/'pins/cari_clip_000009_shared_export_pins.json'))['export_files']
        directory = base/'cari_shared_export_v1'; export = strict(ledger.read(directory/'report.json',pins['report.json']))
        if (export['status'] != 'pass' or export['producer_revision'] != old.SOURCE or export['ground_truth_used'] is not False or export['oracle_modes'] != []):
            raise ValueError('Original saved export provenance differs')
        a = load_npz(ledger,directory/'trajectory.npz',pins['trajectory.npz'])
        track_path = ROOT/'results'/('sequence-pose-probe-'+TRACK_SOURCE+'-v2')/'episode_000009.npz'
        source_track_report(ledger,track_path.with_name('report.json'))
        b = load_npz(ledger,track_path,TRACK_PIN); v,faces,r,t,K = (a[k] for k in ('object_vertices','object_faces','object_rotation','object_translation','camera_K'))
        n = len(t); indices = np.arange(n)
        if (n != 415 or b['points'].shape != (33,3) or not np.array_equal(a['frame_index'],indices)
                or not np.array_equal(b['frame_index'],indices) or float(a['object_scale']) != 1.
                or any(not np.array_equal(a[k],b[k]) for k in ('object_vertices','object_faces','object_scale','camera_K'))):
            raise ValueError('Frozen 33 original material tracks/geometry/gauge/full timeline required')
        if b['tracks_xy'].shape != (n,33,2) or b['RGB_visible'].shape != (n,33):
            raise ValueError('Original full-T RGB witness bank differs')
        inventory, mask_report = old.authenticated_masks(ledger,front,rows[9],base,9,n)
        if transport['mask_inventory'] != mask_report['mask_inventory']: raise ValueError('Original mask source differs')
        spec = export['clip_spec']; w,h = spec['width'],spec['height']; scale = np.array([640/w,480/h]); kk = K.copy()
        if (spec != dict(episode_index=9,total_frames=n,camera_name='front_stereo_camera_left',height=h,width=w)
                or export['frames'] != n or export['original_frame_indices'] != indices.tolist()):
            raise ValueError('Original fixed video grid/export coverage differs')
        kk[0] *= scale[0]; kk[1] *= scale[1]; kk[:2,2] -= .5
        xy,visible,points = b['tracks_xy'],b['RGB_visible'],b['points']
        for stage,pin in STAGE_PINS.items(): ledger.record(base/stage/'report.json',pin)
        depth,support = original_depth(ledger,base,xy,visible,scale,K,inventory,mask_report['input_sha256'])
        import cv2
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4'
        ledger.record(video,transport['video_pin'])
        capture = cv2.VideoCapture(str(video))
        try:
            fps = old.actual_fps(capture)
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n: raise ValueError('Original video frame count differs')
        finally: capture.release()
        fit = refine_sequence(v,points,xy,visible,r,t,np.ones(n,bool),kk,indices,fps,config,
            tracks_depth_m=depth,depth_visible=support,depth_config=DEPTH_CFG)
        output = out/'episode_000009.npz'
        np.savez_compressed(output,rotation=fit.rotations,translation=fit.translations,frame_index=indices,
            object_vertices=v,object_faces=faces,object_scale=a['object_scale'],camera_K=K,points=points,
            tracks_xy=xy,RGB_visible=visible,tracks_depth_m=depth,depth_visible=support)
        output.chmod(0o444); human_path = directory/'target.npy'; ledger.record(human_path,pins['target.npy'])
        human = np.load(human_path,mmap_mode='r',allow_pickle=False); contact=[]
        for i in np.linspace(0,n-1,9).astype(int):
            tree = cKDTree(np.asarray(human[i])); contact.append([int(i),*[float(tree.query(v@rr[i].T+tt[i])[0].min())
                for rr,tt in ((r,t),(b['rotation'],b['translation']),(fit.rotations,fit.translations))]])
        report.update(status='complete_saved_RGBD_diagnostic_not_quality_pass',frames=n,points=len(points),fps=fps,
            original=residuals(points,r,t,kk,xy,visible,depth,support),
            rejected_RGB_only=residuals(points,b['rotation'],b['translation'],kk,xy,visible,depth,support),
            RGBD=residuals(points,fit.rotations,fit.translations,kk,xy,visible,depth,support),fit=fit.diagnostics,
            nearest_human_distance_proxy_original_RGB_RGBD_m=contact,output=ledger.record(output),
            production_adopted=False,heldout_accuracy_verified=False,contact_truth_claimed=False,
            depth_support='all_four_valid_neighbors_in_object_and_outside_same_pinned_human_mask')
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS) != binding: raise ValueError('Own source closure changed')
        report['sources'] = ledger.records
    except Exception as exc: report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds'] = time.monotonic()-started
    path = out/'report.json'; path.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n'); path.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','original','rejected_RGB_only','RGBD')},allow_nan=False))
    if report['status']=='fail': raise SystemExit(1)


if __name__ == '__main__':
    if len(os.sys.argv) != 1: raise SystemExit('No arbitrary source/config inputs accepted')
    run()

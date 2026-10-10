"""Azure CPU saved-only RGB pose ablation; never replaces a baseline/export.

Frozen external manufactured DEV and RESERVED diagnostics precede challenge QA.
Only original legal Track1 RGB and SHA-bound saved predictions are consumed.
No native model calls, source calibration, hidden labels, mesh or scale edits.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from world_reward.sequence_pose import SequencePoseConfig, refine_sequence

ROOT = Path('/srv/scenesmith/world-reward')
SOURCE = '052ba1554e9a573d566713a99a61d89a5f27681c'
CFG = SequencePoseConfig(1.5, .05, .15, 20., 100., .1, .02, 60,
    'manufactured_nonchallenge_sequence_DEV_20261010_v1')


def project(p, r, t, k):
    xyz = p[None] @ r.swapaxes(-1, -2)+t[:, None]
    return xyz[..., :2]/xyz[..., 2, None]*k.diagonal()[:2]+k[:2, 2]


def manufactured_gate(seed, reserved):
    """Independent moving/static/occlusion/turn cases; truth never enters fit."""
    rng = np.random.default_rng(seed); rows = []
    for case in ('static', 'linear', 'acceleration', 'rotation', 'turn', 'occlusion'):
        n = 24; a = np.arange(n)/30
        v = rng.uniform(-.15, .15, (24, 3)); k = np.array([[256., 0, 128], [0, 256., 128], [0, 0, 1.]])
        t = np.zeros((n, 3)); t[:, 2] = 2.
        angle = np.zeros(n)
        if case in ('linear', 'occlusion'): t[:, 0] = .15*a
        if case == 'acceleration': t[:, 0] = .3*a*a
        if case == 'turn': t[:, 0] = .16*np.sin(5*a); t[:, 1] = .06*np.cos(5*a)
        if case == 'rotation': angle = 1.2*a
        if case == 'occlusion': t[:, 1] = .04*np.sin(7*a); angle = .5*a
        r = Rotation.from_rotvec(np.column_stack((a*0, angle, a*0))).as_matrix()
        noisy_t = t+rng.normal(0, .008, t.shape); noisy_t[0] = t[0]
        noisy_r = Rotation.from_rotvec(rng.normal(0, .02, (n, 3))).as_matrix() @ r; noisy_r[0] = r[0]
        xy = project(v, r, t, k)+rng.normal(0, .3 if not reserved else .5, (n, len(v), 2))
        support = np.ones(xy.shape[:2], bool)
        if case == 'occlusion': support[9:15] = False; xy[~support] = np.nan
        out = refine_sequence(v, v, xy, support, noisy_r, noisy_t, np.ones(n, bool), k,
                              np.arange(n), 30, CFG)
        truth = v[None] @ r.swapaxes(-1, -2)+t[:, None]
        old = v[None] @ noisy_r.swapaxes(-1, -2)+noisy_t[:, None]
        new = v[None] @ out.rotations.swapaxes(-1, -2)+out.translations[:, None]
        old_error = float(np.linalg.norm(old-truth, axis=2).mean())
        new_error = float(np.linalg.norm(new-truth, axis=2).mean())
        excursion = float(np.linalg.norm(truth[-1]-truth[0], axis=1).mean())
        recovered = float(np.linalg.norm(new[-1]-new[0], axis=1).mean())
        # Anti-collapse and actual error gates; no judging by low acceleration alone.
        passed = new_error <= old_error*.9 and (excursion < 1e-6 or .8 <= recovered/excursion <= 1.2)
        rows.append(dict(case=case, old_error_m=old_error, new_error_m=new_error,
                         motion_retention=None if excursion < 1e-6 else recovered/excursion,
                         pass_gate=bool(passed), converged=out.diagnostics['converged']))
    return dict(seed=seed, reserved=reserved, cases=rows, passed=all(r['pass_gate'] for r in rows))


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if ROOT != Path(os.environ['WR_ROOT']) or code != ROOT/'jobs'/revision/'run_sequence_pose_probe/code':
        raise ValueError('Exact Azure code/runtime source required')
    out = ROOT/'results'/('sequence-pose-probe-'+revision)
    if not out.is_dir() or any(out.iterdir()): raise ValueError('Fresh isolated output directory required')
    bound = {}
    def pin(p, expected=None):
        p = Path(p)
        before = p.lstat()
        if p.resolve() != p or p.is_symlink() or before.st_nlink != 1 or before.st_size > 2 << 30:
            raise ValueError('Bounded canonical original artifact required')
        raw = p.read_bytes(); record = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if expected is not None and record != expected: raise ValueError('Source SHA mismatch')
        if before != p.lstat(): raise ValueError('Source changed')
        bound[str(p)] = record; return raw
    report = dict(status='fail', producer_revision=revision, numerical_source=SOURCE, ground_truth_used=False,
                  manual_labels=False, full_4D_export_replaced=False, model_calls=0, config=asdict(CFG))
    try:
        for name in ('infra/sequence_pose_probe.py', 'infra/run_sequence_pose_probe.sh', 'src/world_reward/sequence_pose.py'):
            pin(code/name)
        dev = manufactured_gate(20261010, False); report['development'] = dev
        if not dev['passed']: raise ValueError('External DEV motion/error gate rejected; do not touch challenge')
        reserved = manufactured_gate(20261011, True); report['reserved'] = reserved
        if not reserved['passed']: raise ValueError('External RESERVED gate rejected; no challenge adaptation')
        import cv2
        from PIL import Image
        report['episodes'] = []
        for ep in (9, 14):
            exp = ROOT/'experiments'/('full4d-v1-'+SOURCE); base = exp/'outputs'/f'episode_{ep:06d}'
            pins = json.loads(pin(exp/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'))['export_files']
            directory = base/'cari_shared_export_v1'
            source = json.loads(pin(directory/'report.json', pins['report.json']))
            if (source['ground_truth_used'] is not False or source['oracle_modes'] != []
                    or source['status'] != 'pass' or source['producer_revision'] != SOURCE):
                raise ValueError('Original video-only source required')
            pin(directory/'trajectory.npz', pins['trajectory.npz'])
            with np.load(directory/'trajectory.npz', allow_pickle=False) as z: a = {k:z[k] for k in z.files}
            r, t, k = a['object_rotation'], a['object_translation'], a['camera_K']; n = len(t)
            if not np.array_equal(a['frame_index'], np.arange(n)) or float(a['object_scale']) != 1:
                raise ValueError('Fixed original full timeline/geometry required')
            inventory = json.loads(pin(base/'automatic_masks/mask-inventory.json'))
            inventory = inventory.get('files', inventory)
            def original_mask(index):
                p = base/f'automatic_masks/masks/1/{index:06d}.png'
                pin(p,inventory[f'1/{index:06d}.png'])
                with Image.open(p) as image: return np.asarray(image)>0
            mask = original_mask(0)
            mask_report = json.loads(pin(base/'automatic_masks/report.json'))
            video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{ep:06d}.mp4'
            # Actual whole-video SHA, streamed remotely; never local video transit.
            digest = hashlib.sha256()
            with video.open('rb') as f:
                for block in iter(lambda:f.read(1<<20), b''): digest.update(block)
            if digest.hexdigest() != mask_report['input_sha256']: raise ValueError('Original RGB differs')
            capture = cv2.VideoCapture(str(video)); ok, frame = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n: raise ValueError('Original frame count')
            h, w = mask.shape
            # Fixed 640x480 diagnostic, K scaled once. Native centres -.5 become OpenCV integer centres.
            size = (640, 480); scale = np.array([size[0]/w, size[1]/h])
            kk = k.copy(); kk[0] *= scale[0]; kk[1] *= scale[1]; kk[:2, 2] -= .5
            gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
            m = cv2.resize(mask.astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
            m = cv2.erode(m, np.ones((3, 3), np.uint8))
            queries = cv2.goodFeaturesToTrack(gray, maxCorners=64, qualityLevel=.01, minDistance=5, mask=m)
            if queries is None or len(queries) < 8: raise ValueError('Insufficient RGB texture; no invented observations')
            queries = queries.reshape(-1, 2).astype(np.float32); count = len(queries)
            # Bind features to first-hit actual exported mesh triangles, not raw depth samples.
            from world_reward.point_surface_queries import _ray_triangle_hits
            v, faces = a['object_vertices'], a['object_faces'].astype(np.int64)
            original = (queries+.5)/scale-.5
            grid = np.column_stack((original[:, 1], original[:, 0]))
            from world_reward.mesh_geometry import normalize_degenerate_faces
            active_faces, face_diagnostics = normalize_degenerate_faces(v,faces)
            render_faces=faces[active_faces]
            _, depth, face_ids, bary = _ray_triangle_hits(v, render_faces, r[0], t[0], k, grid, np.ones(count, bool))
            keep = face_ids >= 0
            points = (v[render_faces[face_ids[keep]]]*bary[keep, :, None]).sum(1); queries = queries[keep]
            if len(points) < 8: raise ValueError('Insufficient canonical RGB attachments')
            tracks = np.full((n, len(points), 2), np.nan); visible = np.zeros(tracks.shape[:2], bool)
            tracks[0] = queries; visible[0] = True; active = np.arange(len(points)); previous = gray; current = queries.copy()
            for index in range(1, n):
                ok, frame = capture.read()
                if not ok: raise ValueError('Full original decoder coverage required')
                gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
                if len(active):
                    q, sf, _ = cv2.calcOpticalFlowPyrLK(previous, gray, current.reshape(-1,1,2), None)
                    back, sb, _ = cv2.calcOpticalFlowPyrLK(gray, previous, q, None)
                    q, back = q.reshape(-1,2), back.reshape(-1,2)
                    good = (sf.ravel()>0)&(sb.ravel()>0)&np.isfinite(q).all(1)&(np.linalg.norm(back-current,axis=1)<=1.)
                    good &= (q[:,0]>=0)&(q[:,0]<640)&(q[:,1]>=0)&(q[:,1]<480)
                    current_mask = original_mask(index)
                    if current_mask.any():
                        small = cv2.resize(current_mask.astype('uint8'),size,interpolation=cv2.INTER_NEAREST)
                        safe = np.clip(np.rint(q).astype(int),[0,0],[639,479]); good &= small[safe[:,1],safe[:,0]]>0
                    else: good[:] = False  # Missing mask isn't a true negative; no unsupported RGB tracks fabricated.
                    active, current = active[good], q[good]
                    tracks[index, active] = current; visible[index, active] = True
                previous = gray
            capture.release()
            # Predeclared visibility filter uses RGB support only, before any fitting/QA.
            retain = visible[1:].sum(0)>0; points=points[retain]; tracks=tracks[:,retain]; visible=visible[:,retain]
            if len(points)<8: raise ValueError('Insufficient persistent material tracks')
            fitted = refine_sequence(v, points, tracks, visible, r, t, np.ones(n,bool), kk,
                                     np.arange(n),30,CFG)
            path = out/f'episode_{ep:06d}.npz'
            np.savez_compressed(path, rotation=fitted.rotations, translation=fitted.translations,
                                frame_index=fitted.frame_index, object_vertices=v, object_faces=faces,
                                object_scale=a['object_scale'], camera_K=k, points=points,
                                tracks_xy=tracks, RGB_visible=visible)
            path.chmod(0o444)
            before_xy = project(points,r,t,kk); after_xy = project(points,fitted.rotations,fitted.translations,kk)
            before_error = float(np.linalg.norm(before_xy[visible]-tracks[visible],axis=1).mean())
            after_error = float(np.linalg.norm(after_xy[visible]-tracks[visible],axis=1).mean())
            target_path=directory/'target.npy'; pin(target_path,pins['target.npy'])
            target=np.load(target_path,mmap_mode='r',allow_pickle=False)
            # Geometry-only human/object nearest-distance proxy on nine uniform frames; not true contact/penetration.
            contact=[]
            for index in np.linspace(0,n-1,9).astype(int):
                tree=cKDTree(np.asarray(target[index]))
                old=v@r[index].T+t[index]; new=v@fitted.rotations[index].T+fitted.translations[index]
                contact.append([int(index),float(tree.query(old)[0].min()),float(tree.query(new)[0].min())])
            report['episodes'].append(dict(episode=ep,frames=n,points=len(points),RGB_supported_frames=int(visible.any(1).sum()),
                before_reprojection_px=before_error,after_reprojection_px=after_error,
                nearest_human_surface_distance_proxy_m=contact,fit=fitted.diagnostics,output=pin(path),
                query_only_face_diagnostics=face_diagnostics,
                production_adopted=False,heldout_4D_accuracy_verified=False))
        for p, record in tuple(bound.items()): pin(Path(p),record)
        report.update(status='complete_saved_RGB_diagnostic_not_quality_pass',sources=bound)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds']=time.monotonic()-started
    receipt=out/'report.json'; receipt.write_text(json.dumps(report,allow_nan=False,sort_keys=True)+'\n'); receipt.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','episodes','development','reserved')},allow_nan=False))
    if report['status']=='fail': raise SystemExit(1)


if __name__ == '__main__': run()

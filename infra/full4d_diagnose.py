"""Azure CPU, saved-only causal audit; no fitting, model or prediction writes.

Full timelines are measured before/after CoCoNet and refinement. Uniform source
views expose masks and generated geometry. The assumed viewer floor is NOT a
physical measurement. None of these proxies establish held-out 3D accuracy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from full4d_pins import identity, canonical, _strict, _no_oracle
from mediapipe_cpu_runtime_verify import source, require
from reconstruction_preview import raster_depth
from world_reward.mesh_geometry import normalize_degenerate_faces
from world_reward.task_grounding import fixed_frame_indices
from world_reward.timeline import native_window_starts, first_occurrence_ownership
from world_reward.trajectory_diagnostics import object_motion, mask_geometry, summarize

ROOT = Path('/srv/scenesmith/world-reward')
BASELINE = 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
ENTRY = 'run_full4d_diagnose'
HELPERS = ('infra/full4d_diagnose.py', 'infra/run_full4d_diagnose.sh',
           'src/world_reward/trajectory_diagnostics.py')
EPISODES = (9, 1, 14)
WIDTH, HEIGHT, MAX_JPEG = 320, 240, 180_000


class Sources:
    def __init__(self): self.files = {}

    def bind(self, path, expected=None, maximum=2 << 30):
        path = canonical(path)
        pin = identity(path, maximum=maximum)
        require(expected is None or pin == expected, 'Saved prediction identity differs')
        require(path not in self.files or self.files[path] == pin, 'Source changed')
        self.files[path] = pin
        return path

    def json(self, path, expected=None, maximum=4 << 20):
        return _strict(self.bind(path, expected, maximum).read_bytes())

    def verify(self):
        require(all(identity(p) == pin for p, pin in self.files.items()), 'Source changed during diagnostic')


def stats(series):
    return {key: value['statistics'] for key, value in series.items()}


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value


def save(path, value):
    with path.open('x') as stream:
        json.dump(jsonable(value), stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    path.chmod(0o444)


def signature_pose(vertices, poses, total, floor):
    return object_motion(vertices, poses[:, :3, :3], poses[:, :3, 3],
                         np.arange(total), 30, plane=[0, -1, 0, floor])


def analyse(root, experiment, out, episode):
    import cv2
    import joblib
    import torch
    from PIL import Image, ImageDraw, ImageFont
    started = time.monotonic()
    base = experiment/'outputs'/f'episode_{episode:06d}'
    bound = Sources()
    mask_report = bound.json(base/'automatic_masks/report.json')
    pose_report = bound.json(base/'object_pose_full_surface/report.json', maximum=64 << 20)
    object_report = bound.json(base/'object_grounded/report.json')
    ground = bound.json(base/'automatic_masks/grounding.json')
    alignment = bound.json(base/'scale_smoke/report.json')
    reports = {}
    for key in ('forward', 'refined', 'export'):
        pins = bound.json(experiment/'pins'/f'cari_clip_{episode:06d}_shared_{key}_pins.json')
        directory = base/f'cari_shared_{key}_v1'
        reports[key] = bound.json(directory/'report.json', pins[key+'_files']['report.json'])
        require(reports[key]['producer_revision'] == BASELINE, 'Wrong frozen producer')
        for name, pin in pins[key+'_files'].items(): bound.bind(directory/name, pin)
    for record in (mask_report, pose_report, object_report, alignment, *reports.values()):
        _no_oracle(record)
        require(record.get('status') == 'pass' and record.get('input_track') == 'track_1'
                and record.get('episode_index') == episode, 'Exact video-only saved PASS sources required')
    count = mask_report['frames']
    video = root/'data'/f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
    require(all(r['input_sha256'] == mask_report['input_sha256'] for r in
                (pose_report, object_report, alignment)), 'Wrong source video')
    bound.bind(video, ground['video_pin'])
    require(ground['video_pin']['sha256'] == mask_report['input_sha256'], 'Wrong grounding video')
    arrays_path = bound.bind(base/'object_pose_full_surface/geometry_and_poses.npz',
                            dict(bytes=(base/'object_pose_full_surface/geometry_and_poses.npz').stat().st_size,
                                 sha256=pose_report['geometry_and_poses_sha256']))
    with np.load(arrays_path, allow_pickle=False) as data:
        initializer = {key: data[key] for key in data.files}
    export_dir = base/'cari_shared_export_v1'
    for name, pin in reports['export']['output_files'].items(): bound.bind(export_dir/name, pin)
    with np.load(export_dir/'trajectory.npz', allow_pickle=False) as data:
        final = {key: data[key] for key in data.files}
    require(np.array_equal(final['frame_index'], np.arange(count))
            and np.array_equal(initializer['frame_index'], np.arange(count)), 'Full original timeline required')
    viewer = bound.json(experiment/'videos'/f'episode_{episode:06d}'/'report.json')
    floor = viewer['virtual_floor']['camera_y_m']
    vertices = final['object_vertices']
    loaded = {}
    for key, filename in (('forward', 'coconet.pth'), ('refined', 'refined.pth')):
        path = base/f'cari_shared_{key}_v1'/filename
        pin = reports[key]['output_files'][filename]
        bound.bind(path, pin)
        # Deserialization only after original immutable producer hashes match.
        loaded[key] = torch.load(path, map_location='cpu', weights_only=False)
        require(loaded[key]['gt'] == {} and loaded[key]['metadata']['ground_truth_used'] is False,
                'Native GT/oracle mode forbidden')
    native = loaded['forward']
    inputs_report = bound.json(base/'cari_inputs/report.json')
    inputs_path = bound.bind(Path(inputs_report['object_poses']), inputs_report['file_sha256']['object_poses'])
    inputs_poses = joblib.load(inputs_path)['obj_pose_world']
    poses = dict(initializer_aligned=inputs_poses, forward=native['pr']['pose_abs'],
                 refined=loaded['refined']['pr']['pose_abs'])
    export_poses = np.repeat(np.eye(4)[None], count, axis=0)
    export_poses[:, :3, :3], export_poses[:, :3, 3] = final['object_rotation'], final['object_translation']
    poses['export'] = export_poses
    motions = {key: signature_pose(vertices, value, count, floor) for key, value in poses.items()}
    original_motion = object_motion(initializer['vertices'], initializer['rotation'], initializer['translation'],
                                   initializer['frame_index'], 30, plane=[0, -1, 0, floor])
    require(np.allclose(original_motion['series']['surface_rms_step_m'],
                        motions['initializer_aligned']['series']['surface_rms_step_m'], atol=1e-5, rtol=1e-4),
            'Original/aligned frames must describe same object motion')
    require(np.array_equal(poses['refined'][:, :3], poses['export'][:, :3]), 'Export changed refined object poses')
    active, _ = normalize_degenerate_faces(vertices, final['object_faces'])
    faces = final['object_faces'][active]
    indices = list(fixed_frame_indices(count, 9))
    mask_inventory = bound.json(base/'automatic_masks/mask-inventory.json', maximum=4 << 20)
    if 'files' in mask_inventory: mask_inventory = mask_inventory['files']
    K = final['camera_K'].copy(); K[0] *= WIDTH/ground['width']; K[1] *= HEIGHT/ground['height']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    sheet = Image.new('RGB', (WIDTH*3, 48+9*(HEIGHT+22)), (24, 27, 33))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f'World Reward | ep{episode:02d} | SAVED baseline | no fitting / no GT', font=font, fill='white')
    for col, text in enumerate(('Original RGB', 'Automatic SAM2 object mask', 'Same saved rigid mesh / final pose')):
        draw.text((col*WIDTH+8, 26), text, font=font, fill='white')
    capture = cv2.VideoCapture(str(video)); require(capture.isOpened(), 'Source decoder failed')
    require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count, 'Wrong source count')
    object_masks, rgb_flow, samples, depth_samples = [], [], [], []
    previous_gray = previous_mask = None
    scale = alignment['depth_alignment']['shared_scale']
    depth_report = bound.json(base/'depth_full/report.json')
    for index in range(count):
        ok, bgr = capture.read(); require(ok, 'Original frame decode failed')
        mask_path = base/'automatic_masks/masks/1'/f'{index:06d}.png'
        bound.bind(mask_path, mask_inventory[f'1/{index:06d}.png'], maximum=16 << 20)
        with Image.open(mask_path) as image: mask = np.asarray(image) > 0
        geometry = mask_geometry(mask); object_masks.append(geometry)
        rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small_mask = cv2.resize(mask.astype('uint8'), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            # Eroded common visibility avoids mask edges; no static label inferred.
            region = cv2.erode((small_mask & previous_mask).astype('uint8'), np.ones((3, 3), 'uint8')) > 0
            magnitudes = np.linalg.norm(flow[region], axis=1)
            rgb_flow.append(dict(frame_index=index, common_pixels=int(region.sum()),
                                 median_flow_px=None if not len(magnitudes) else float(np.median(magnitudes))))
        previous_gray, previous_mask = gray, small_mask
        if index in indices:
            position = indices.index(index); y = 48+position*(HEIGHT+22)
            masked = rgb.copy(); masked[small_mask] = np.rint(.45*masked[small_mask]+.55*np.array([245,148,56])).astype('uint8')
            obj = vertices @ final['object_rotation'][index].T + final['object_translation'][index]
            obj_depth = raster_depth(obj, faces, K, WIDTH, HEIGHT)
            overlaid = rgb.copy(); covered = np.isfinite(obj_depth)
            overlaid[covered] = np.rint(.45*overlaid[covered]+.55*np.array([245,148,56])).astype('uint8')
            for col, image in enumerate((rgb, masked, overlaid)): sheet.paste(Image.fromarray(image), (col*WIDTH, y))
            draw.text((8,y+HEIGHT+3),f'frame{index:04d} / {count-1} | {index/30:.2f}s | all real components retained',font=font,fill='white')
            candidate_id = pose_report['temporal_selection']['candidate_indices'][index]
            candidate = next(c for c in pose_report['frames'][index]['candidates'] if c['hypothesis_index'] == candidate_id)
            samples.append(dict(frame_index=index, automatic_mask=geometry,
                initializer_visible_iou=candidate['selected_silhouette_iou'],
                initializer_depth_residual_m=candidate['selected_depth_residual_m'],
                final_full_silhouette_iou=None if not (covered|small_mask).any() else
                    float((covered&small_mask).sum()/(covered|small_mask).sum()),
                full_render_vs_visible_mask_is_occlusion_confounded=True))
            path = base/'depth_full'/f'{index:06d}.npz'
            row = depth_report['frames'][index]
            bound.bind(path, dict(bytes=path.stat().st_size,sha256=row['output_sha256']))
            with np.load(path, allow_pickle=False) as data:
                valid = mask & data['mask'] & np.isfinite(data['depth']) & (data['depth']>0)
                depths = data['depth'][valid] * scale
            depth_samples.append(dict(frame_index=index,valid_pixels=len(depths),
                inferred_depth_m=None if not len(depths) else summarize(depths)))
    capture.release()
    for quality in (72, 60, 48, 36):
        from io import BytesIO
        buffer = BytesIO(); sheet.save(buffer, format='JPEG', quality=quality, optimize=True)
        if len(buffer.getvalue()) <= MAX_JPEG: break
    raw = buffer.getvalue(); require(len(raw) <= MAX_JPEG, 'Tiny QA image budget exceeded')
    jpeg = out/f'episode_{episode:06d}.jpg'
    with jpeg.open('xb') as stream: stream.write(raw)
    jpeg.chmod(0o444)
    starts = native_window_starts(count)
    owner, _ = first_occurrence_ownership(count, [np.arange(s,s+96,dtype='int64') for s in starts])
    seams = np.flatnonzero(np.diff(owner)) + 1
    series = {key: value['series'] for key, value in motions.items()}
    # Whole-clip plus equal fixed 2-second blocks, never human-labelled intervals.
    blocks = [dict(start=start,stop=min(start+60,count),
        motion={key:{name:summarize(values[start:min(start+59,len(values))]) for name,values in item.items()
                     if not name.startswith('plane_') and len(values[start:min(start+59,len(values))])}
                for key,item in series.items()},
        rgb_flow_median_px=summarize([r['median_flow_px'] for r in rgb_flow
                if start < r['frame_index'] < min(start+60,count) and r['median_flow_px'] is not None])
                if any(start < r['frame_index'] < min(start+60,count) and r['median_flow_px'] is not None for r in rgb_flow) else None)
        for start in range(0,count,60)]
    result = dict(episode_index=episode,frames=count,task=mask_report['action'],object_prompt=mask_report['object_prompt'],
        grounding_seed=ground['seed'],shape_generation_frame=object_report['frame_index'],
        generated_extent_m=pose_report['metric_gauge_extent'],motion=stats(motions),
        numeric_export_equals_refined=True,original_aligned_motion_equivalent=True,
        fixed_virtual_plane_not_physical_ground=floor,uniform_views=samples,uniform_depth_views=depth_samples,
        fixed_two_second_blocks_not_static_labels=blocks,
        window_boundary_frames=seams.tolist(),
        boundary_surface_step_m={key:[float(value['surface_rms_step_m'][s-1]) for s in seams]
                                 for key,value in series.items()},
        all_mask_area_pixels=summarize([r['area_pixels'] for r in object_masks]),
        missing_rgb_flow_pairs=sum(r['median_flow_px'] is None for r in rgb_flow),
        optical_flow_is_lowres_uncalibrated_rgb_proxy=True,
        sources={str(p):pin for p,pin in bound.files.items()},source_rehashed_after=True,
        poster=identity(jpeg),model_calls=0,optimizer_calls=0,predictions_modified=False,ground_truth_used=False,
        hand_labeled_test=False,quality_verified=False,elapsed_seconds=time.monotonic()-started)
    bound.verify(); save(out/f'episode_{episode:06d}.json',result)
    print(json.dumps({k:result[k] for k in ('episode_index','frames','shape_generation_frame','generated_extent_m',
        'window_boundary_frames','numeric_export_equals_refined','elapsed_seconds')}),flush=True)
    return result


def native(code, experiment, out):
    require(sys.platform == 'linux' and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
            'Azure offline CPU container required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    results = [analyse(ROOT,experiment,out,ep) for ep in EPISODES]
    optimizer = ROOT/'vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py'
    require(identity(optimizer,readonly=False)['sha256'] ==
            '84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b', 'Wrong native optimizer source')
    # Read the actual producer's applied configuration, not a freshly imported
    # default; importing the optimizer is unnecessary and could load assets.
    applied_configs={str(ep):_strict((experiment/'outputs'/f'episode_{ep:06d}'/
        'cari_shared_refined_v1/report.json').read_bytes())['config'] for ep in EPISODES}
    save(out/'report.json',dict(status='complete_diagnostic_not_quality_pass',baseline_revision=BASELINE,
        diagnostic_revision=os.environ['WR_CODE_REVISION'],episodes=list(EPISODES),gpu_used=False,
        model_calls=0,optimizer_calls=0,predictions_modified=False,ground_truth_used=False,hand_labeled_test=False,
        applied_configs=applied_configs,optimizer_source=identity(optimizer,readonly=False),
        episode_results={str(ep):identity(out/f'episode_{ep:06d}.json') for ep in EPISODES}))


def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01',
            'Azure host only')
    code=canonical(Path(os.environ['WR_CODE']));rev=os.environ['WR_CODE_REVISION']
    binding=source(ROOT,code,rev,ENTRY,HELPERS)
    experiment=ROOT/'experiments'/f'full4d-v1-{BASELINE}'
    out=ROOT/'results'/f'full4d-diagnostic-{rev}'
    out.mkdir(mode=0o755);os.chown(out,1000,1000)
    command=['docker','run','--rm','--name','wr-full4d-diagnostic-'+rev[:12],
        '--network','none','--read-only','--user','1000:1000','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--memory','12g','--cpus','8',
        '--tmpfs','/tmp:rw,nosuid,size=256m']
    mounts=[(code.parent,True),(experiment,True),(ROOT/'vendor/video_to_data',True),(out,False)]
    for ep in EPISODES:
        mounts.append((ROOT/'data'/f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True))
    for path,readonly in mounts:command+=['--mount',f'type=bind,src={path},dst={path}'+(',readonly' if readonly else '')]
    command+=['--entrypoint','/usr/bin/env',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin',
        'HOME=/tmp','CUDA_VISIBLE_DEVICES=-1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4',
        'OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','WR_CODE_REVISION='+rev,
        'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'/opt/conda/bin/python','-B',
        str(code/'infra/full4d_diagnose.py'),'--native',str(code),str(experiment),str(out)]
    with (out/'run.log').open('xb') as log:
        result=subprocess.run(command,stdout=log,stderr=log,timeout=600,check=False)
    require(source(ROOT,code,rev,ENTRY,HELPERS)==binding,'Diagnostic code changed')
    require(result.returncode==0,'Saved-only diagnostic failed; never modify baseline or retry scientifically')
    print('FULL4D_DIAGNOSTIC_COMPLETE',str(out))


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--native':native(*map(Path,sys.argv[2:]))
    else:run()

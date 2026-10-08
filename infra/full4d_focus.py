"""Azure-only, saved evidence zoom: automatic Qwen box versus SAM2 versus mesh.

All three frozen clips, nine uniform views. Crop selection affects only the QA
viewport, never masks, geometry, poses, scale, K or predictions. No new model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from full4d_diagnose import ROOT, BASELINE, IMAGE, EPISODES, Sources, save
from mediapipe_cpu_runtime_verify import source, canonical, require, identity

DIAGNOSTIC = '5adcb031d5d5a09c4b06d962a03c5f1fe6b906cb'
ENTRY = 'run_full4d_focus'
WIDTH, HEIGHT = 320, 240


def viewport(boxes, width, height):
    """Union of automatic boxes, 25% context, integer 4:3 viewport; QA only."""
    import numpy as np
    boxes = [np.asarray(b, dtype=float) for b in boxes if b is not None]
    if not boxes:
        return (0, 0, width, height)
    require(all(b.shape == (4,) and np.isfinite(b).all() and
                0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height for b in boxes),
            'Automatic ordinary image boxes required')
    a = np.stack(boxes)
    lo, hi = a[:, :2].min(0), a[:, 2:].max(0)
    centre = (lo+hi)/2
    w, h = (hi-lo)*1.5
    w, h = max(w, h*WIDTH/HEIGHT), max(h, w*HEIGHT/WIDTH)
    w, h = min(width, int(np.ceil(w))), min(height, int(np.ceil(h)))
    left = max(0, min(width-w, int(np.floor(centre[0]-w/2))))
    top = max(0, min(height-h, int(np.floor(centre[1]-h/2))))
    return left, top, left+w, top+h


def native(code, out):
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    from reconstruction_preview import raster_depth
    from world_reward.mesh_geometry import normalize_degenerate_faces
    from world_reward.task_grounding import fixed_frame_indices
    require(sys.platform == 'linux' and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
            'Offline Azure CPU only')
    old = ROOT/'results'/('full4d-diagnostic-'+DIAGNOSTIC)
    summary = json.loads((old/'report.json').read_bytes())
    require(summary['status'] == 'complete_diagnostic_not_quality_pass' and
            summary['diagnostic_revision'] == DIAGNOSTIC and summary['baseline_revision'] == BASELINE,
            'Original saved-only diagnostic required')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    for ep in EPISODES:
        s = Sources()
        d = s.json(old/f'episode_{ep:06d}.json', summary['episode_results'][str(ep)])
        def bind(path):
            path = Path(path)
            return s.bind(path, d['sources'][str(path)])
        base = ROOT/'experiments'/('full4d-v1-'+BASELINE)/'outputs'/f'episode_{ep:06d}'
        ground = json.loads(bind(base/'automatic_masks/grounding.json').read_bytes())
        observations = {r['frame_index']: r for r in ground['records']}
        with np.load(bind(base/'cari_shared_export_v1/trajectory.npz'), allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        active, _ = normalize_degenerate_faces(arrays['object_vertices'], arrays['object_faces'])
        faces = arrays['object_faces'][active]
        indices = list(fixed_frame_indices(d['frames'], 9))
        rows = []
        sheet = Image.new('RGB', (WIDTH*3, 48+9*(HEIGHT+22)), (24, 27, 33))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), f'World Reward | ep{ep:02d} | automatic viewport only | unchanged baseline', font=font, fill='white')
        for col, label in enumerate(('Original / automatic Qwen box', 'Saved SAM2 mask', 'Saved mesh / final pose')):
            draw.text((col*WIDTH+8, 27), label, font=font, fill='white')
        video = ROOT/'data'/f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4'
        cap = cv2.VideoCapture(str(bind(video)))
        require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == d['frames'], 'Original source required')
        for index in range(d['frames']):
            ok, bgr = cap.read(); require(ok, 'Original decoder failed')
            if index not in indices:
                continue
            with Image.open(bind(base/'automatic_masks/masks/1'/f'{index:06d}.png')) as a:
                mask = np.asarray(a) > 0
            qbox = observations[index]['object_bbox']
            ys, xs = np.nonzero(mask)
            mbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None
            left, top, right, bottom = viewport((qbox, mbox), ground['width'], ground['height'])
            rgb = cv2.cvtColor(bgr[top:bottom,left:right], cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            small = cv2.resize(mask[top:bottom,left:right].astype('uint8'), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
            K = arrays['camera_K'].copy()
            K[0,2] -= left; K[1,2] -= top
            K[0] *= WIDTH/(right-left); K[1] *= HEIGHT/(bottom-top)
            vertices = arrays['object_vertices']@arrays['object_rotation'][index].T+arrays['object_translation'][index]
            covered = np.isfinite(raster_depth(vertices, faces, K, WIDTH, HEIGHT))
            images = []
            for region in (None, small, covered):
                im = rgb.copy()
                if region is not None:
                    im[region] = np.rint(.45*im[region]+.55*np.array([245,148,56])).astype('uint8')
                images.append(Image.fromarray(im))
            if qbox is not None:
                box = [(qbox[0]-left)*WIDTH/(right-left), (qbox[1]-top)*HEIGHT/(bottom-top),
                       (qbox[2]-left)*WIDTH/(right-left), (qbox[3]-top)*HEIGHT/(bottom-top)]
                ImageDraw.Draw(images[0]).rectangle(box, outline=(80,220,245), width=2)
            y = 48+indices.index(index)*(HEIGHT+22)
            for col, im in enumerate(images):
                sheet.paste(im, (col*WIDTH, y))
            draw.text((8,y+HEIGHT+3), f'frame {index:04d} | {index/30:.2f}s | cyan: saved automatic Qwen box', font=font, fill='white')
            rows.append(dict(frame_index=index, qwen_object_box=qbox, sam2_bbox=mbox,
                             display_only_crop=[left,top,right,bottom]))
        cap.release()
        for quality in (72,60,48,36,24):
            buffer = BytesIO(); sheet.save(buffer, format='JPEG', quality=quality, optimize=True)
            raw = buffer.getvalue()
            if len(raw) <= 180000:
                break
        require(len(raw) <= 180000, 'Tiny diagnostic image only')
        path = out/f'episode_{ep:06d}.jpg'
        with path.open('xb') as f:
            f.write(raw)
        path.chmod(0o444); s.verify()
        save(out/f'episode_{ep:06d}.json', dict(baseline_revision=BASELINE, diagnostic_revision=DIAGNOSTIC,
            frame_indices=indices, rows=rows, poster=identity(path), sources={str(p):v for p,v in s.files.items()},
            predictions_modified=False, quality_verified=False, model_calls=0, gpu_used=False))


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure host only')
    code = canonical(Path(os.environ['WR_CODE'])); rev = os.environ['WR_CODE_REVISION']
    pin = source(ROOT, code, rev, ENTRY, ('infra/full4d_focus.py', 'infra/run_full4d_focus.sh'))
    out = ROOT/'results'/('full4d-focus-'+rev); out.mkdir(); os.chown(out,1000,1000)
    command = ['docker','run','--rm','--network','none','--read-only','--user','1000:1000',
               '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','8g','--cpus','4',
               '--tmpfs','/tmp:rw,nosuid,size=256m']
    for path, ro in ((code.parent,True),(ROOT/'experiments'/('full4d-v1-'+BASELINE),True),
                     (ROOT/'results'/('full4d-diagnostic-'+DIAGNOSTIC),True),(out,False)):
        command += ['--mount',f'type=bind,src={path},dst={path}'+(',readonly' if ro else '')]
    for ep in EPISODES:
        path = ROOT/'data'/f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4'
        command += ['--mount',f'type=bind,src={path},dst={path},readonly']
    command += ['--entrypoint','/usr/bin/env',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin',
        'HOME=/tmp','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'/opt/conda/bin/python','-B',
        str(code/'infra/full4d_focus.py'),'--native',str(code),str(out)]
    with (out/'run.log').open('xb') as f:
        result = subprocess.run(command,stdout=f,stderr=f,timeout=240)
    require(result.returncode == 0 and source(ROOT, code, rev, ENTRY, ('infra/full4d_focus.py','infra/run_full4d_focus.sh')) == pin,
            'Saved-only automatic focus failed')
    from full4d_publish import PrivatePreviews
    client = PrivatePreviews(); client.require_private(); files = []
    for ep in EPISODES:
        path = out/f'episode_{ep:06d}.jpg'; identity(path,180000)
        row = client.upload(f'full4d-{rev}/episode_{ep:06d}.jpg',path.read_bytes(),'image/jpeg',rev)
        row['episode_index'] = ep; files.append(row)
        save(out/f'published_{ep:06d}.json',row)
        client.head(row['name'],row,row['etag'])
    save(out/'report.json',dict(status='complete_saved_only_qa', producer_revision=rev, baseline_revision=BASELINE,
        gpu_used=False, model_calls=0, predictions_modified=False, quality_verified=False, files=files))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--native':
        native(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        run()

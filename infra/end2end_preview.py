"""Full-timeline RGB / frozen baseline / native candidate diagnostic on Azure.

The candidate may fail QA: that failure is displayed, never hidden. This is not
an official metric evaluation, a new reconstruction, or manual test annotation.
"""
from io import BytesIO
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from mediapipe_cpu_runtime_verify import canonical, identity, source, strict, require

ROOT = Path('/srv/scenesmith/world-reward')
BASELINE = '052ba1554e9a573d566713a99a61d89a5f27681c'
ENTRY = 'run_end2end_preview'
COHORT = [9, 1, 14, 7]
HELPERS = ('infra/end2end_preview.py', 'infra/run_end2end_preview.sh',
    'infra/full4d_video.py', 'infra/full4d_publish.py',
    'infra/mediapipe_cpu_runtime_verify.py')
FILES = {'target': 'target.npy', 'trajectory': 'trajectory.npz',
         'native_parameters': 'native_parameters.npz'}


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}', value), 'Exact revision required')
    return value


def labels(episode, accepted):
    return ['Original RGB', 'Baseline complete (052)',
            'Native joint — QA ' + ('PASS, GT pending' if accepted else 'FAIL / not adopted')]


def probe_contract(probe, total):
    streams = probe.get('streams', [])
    require(len(streams) == 1, 'Exactly one encoded video stream required')
    s = streams[0]
    require(s.get('codec_name') == 'h264' and s.get('width') == 960 and s.get('height') == 280
        and s.get('r_frame_rate') == '30/1' and s.get('avg_frame_rate') == '30/1'
        and s.get('nb_read_frames') == str(total), 'Full original timeline / fixed viewports required')


def render(candidate_revision):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import full4d_video as viewer
    from sequence_pose_probe import ArtifactLedger

    started = time.monotonic(); rev = revision(os.environ['WR_CODE_REVISION'])
    code = canonical(os.environ['WR_CODE']); candidate_revision = revision(candidate_revision)
    require(code == ROOT/'jobs'/rev/ENTRY/'code', 'Immutable render closure required')
    binding = source(ROOT, code, rev, ENTRY, HELPERS)
    output = ROOT/'results'/('end2end-preview-'+rev)
    require(output.is_dir() and set(p.name for p in output.iterdir()) == {'.container.cid'},
            'Fresh owned rendering directory required')
    ledger = ArtifactLedger(); reports = []; states = []
    base = ROOT/'experiments'/('full4d-v1-'+BASELINE)
    for ep in COHORT:
        candidate = ROOT/'results'/('native-joint-real-'+candidate_revision)/f'episode_{ep:06d}'
        report_path = candidate/'report.json'
        if not report_path.exists():
            states.append(dict(episode=ep, status='not_reconstructed_in_this_ablation',
                reason='upstream full-pose unsupported' if ep in (1,7) else 'candidate missing'))
            continue
        r = strict(ledger.read(report_path, maximum=4<<20))
        require(r['producer_revision'] == candidate_revision and r['episode'] == ep
                and r['ground_truth_used'] is False and r['private_truth_read'] is False
                and r['baseline_modified'] is False, 'Automatic no-GT candidate provenance required')
        if r.get('status') != 'complete_diagnostic_not_quality_pass':
            states.append(dict(episode=ep, status='failed', phase=r.get('phase'),
                reason=r.get('error', 'no complete candidate geometry')))
            continue
        require(r.get('native_effective_updates') == 301 and r.get('fitted_outputs_sealed_before_QA') is True
            and r.get('source_inputs_rehashed') is True, 'Complete sealed native candidate required')
        directory = base/'outputs'/f'episode_{ep:06d}'/'cari_shared_export_v1'
        pins = strict(ledger.read(base/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'))
        spec = pins['clip_spec']; total = spec['total_frames']
        require(spec == dict(episode_index=ep, total_frames=total, camera_name='front_stereo_camera_left',
            width=1536, height=1152), 'Original source grid required')
        for name in FILES.values(): ledger.record(directory/name, pins['export_files'][name])
        for key, name in FILES.items(): ledger.record(candidate/name, r['candidate_outputs'][key])
        require(r['baseline_binding']['outputs'] == pins['export_files'], 'Same frozen baseline required')
        ah = np.load(directory/'target.npy', mmap_mode='r', allow_pickle=False)
        bh = np.load(candidate/'target.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory/'trajectory.npz', allow_pickle=False) as a:
            a = {k:a[k] for k in ('object_vertices','object_faces','object_rotation','object_translation',
                                  'object_scale','camera_K','frame_index')}
        with np.load(candidate/'trajectory.npz', allow_pickle=False) as b:
            b = {k:b[k] for k in a}
        with np.load(directory/'native_parameters.npz', allow_pickle=False) as native:
            faces = native['human_faces']; indices = native['frame_index']
        with np.load(candidate/'native_parameters.npz', allow_pickle=False) as native:
            require(np.array_equal(native['human_faces'], faces)
                and np.array_equal(native['frame_index'], indices), 'No topology / timeline substitutions')
        require(ah.shape == bh.shape == (total,18439,3), 'Every original human vertex/frame required')
        for key in ('object_vertices','object_faces','object_scale','camera_K','frame_index'):
            require(np.array_equal(a[key], b[key]), 'Clip-constant geometry, K, scale and original indices required')
        viewer.checked_geometry(ah, faces, a, indices, total)
        viewer.checked_geometry(bh, faces, b, indices, total)
        camera = viewer.display_intrinsics(a['camera_K'], 1536, 1152)
        floor = viewer.fixed_floor_height(ah)
        ra = viewer.SavedSceneRenderer(camera, faces, a['object_faces'], floor)
        rb = viewer.SavedSceneRenderer(camera, faces, b['object_faces'], floor)
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{ep:06d}.mp4'
        require(str(video) in r['input_ledger'], 'Source RGB must be bound by native candidate')
        ledger.record(video, r['input_ledger'][str(video)])
        capture = cv2.VideoCapture(str(video))
        require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
            and abs(capture.get(cv2.CAP_PROP_FPS)-30) < .003, 'Original RGB timeline required')
        accepted = r['decision']['passed']; template = Image.new('RGB', (960,280), (24,27,33))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
        draw = ImageDraw.Draw(template)
        for column, title in enumerate(labels(ep, accepted)):
            draw.text((column*320+4,3), title, font=font, fill='white')
        draw.text((324,263), 'Same inferred scale / fixed virtual floor / grid 0.5m', font=font, fill='white')
        destination = output/f'episode_{ep:06d}.mp4'
        executable, probe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        require(executable and probe, 'Existing ffmpeg/ffprobe required')
        ceiling = min(900000, int((2_000_000-150000)*8*30/total*.85))
        command = [executable,'-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo',
            '-pix_fmt','rgb24','-s','960x280','-r','30','-i','pipe:0','-an','-c:v','libx264',
            '-preset','fast','-crf','30','-maxrate',str(ceiling),'-bufsize',str(ceiling),
            '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
        poster = None
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in range(total):
                ok, rgb = capture.read()
                require(ok and rgb.shape == (1152,1536,3)
                    and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame+1, 'No lost / repeated RGB frames')
                row = template.copy(); row.paste(Image.fromarray(cv2.cvtColor(
                    cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)), (0,20))
                row.paste(Image.fromarray(ra.frame(ah[frame], viewer.object_at_frame(a,frame))), (320,20))
                row.paste(Image.fromarray(rb.frame(bh[frame], viewer.object_at_frame(b,frame))), (640,20))
                ImageDraw.Draw(row).text((4,263), f'ep{ep:02d} f{frame}/{total-1} {frame/30:.2f}s', font=font, fill='white')
                proc.stdin.write(np.asarray(row).tobytes())
                if frame == 0:
                    stream = BytesIO(); row.save(stream, format='JPEG', quality=65, optimize=True); poster=stream.getvalue()
            require(not capture.read()[0], 'No extra original RGB frames')
            proc.stdin.close(); require(proc.wait(timeout=60)==0, 'Encoding failed')
        finally:
            capture.release()
            if proc.poll() is None: proc.kill(); proc.wait()
            if proc.stderr: proc.stderr.close()
        require(0 < destination.stat().st_size <= 2_000_000 and poster and len(poster)<=100000,
                'Lightweight full-clip media bounds required')
        destination.chmod(0o444); jpeg=output/f'episode_{ep:06d}.jpg'
        with jpeg.open('xb') as stream: stream.write(poster)
        jpeg.chmod(0o444)
        encoded = json.loads(subprocess.check_output([probe,'-v','error','-count_frames',
            '-select_streams','v:0','-show_streams','-of','json',str(destination)]))
        probe_contract(encoded,total)
        report = dict(episode=ep, frames=total, original_frame_indices=list(range(total)),
            QA_passed=accepted, quality_verified=False, candidate_report=ledger.records[str(report_path)],
            video=identity(destination,2_000_000), poster=identity(jpeg,100000),
            unchanged_baseline=True, one_fixed_camera=True, one_fixed_floor=True, per_frame_alignment=False)
        reports.append(report); states.append(dict(episode=ep,status='complete',QA_passed=accepted))
        print(json.dumps(dict(stage='full_video_rendered',episode=ep,frames=total,QA_passed=accepted)),flush=True)
    require(reports, 'No completed candidate available for honest visual comparison')
    ledger.verify(); require(source(ROOT,code,rev,ENTRY,HELPERS)==binding, 'Immutable source changed')
    result = dict(schema='world_reward.end2end_preview.v1',status='complete',producer_revision=rev,
        candidate_revision=candidate_revision,baseline_revision=BASELINE,cohort=COHORT,states=states,
        source_binding=binding,sources=ledger.records,reports=reports,quality_verified=False,
        source_rehashed_after=True,heavy_media_local=False,elapsed_seconds=time.monotonic()-started)
    path=output/'render.json'
    with path.open('xb') as stream: stream.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode())
    path.chmod(0o444)


def publish(render_revision):
    from full4d_publish import PrivatePreviews
    rev=revision(os.environ['WR_CODE_REVISION']); render_revision=revision(render_revision)
    code=canonical(os.environ['WR_CODE']); binding=source(ROOT,code,rev,ENTRY,HELPERS)
    output=ROOT/'results'/('end2end-preview-'+render_revision)
    pin=identity(output/'render.json',4<<20); rendered=strict((output/'render.json').read_bytes())
    require(rendered['status']=='complete' and rendered['producer_revision']==render_revision
        and rendered['quality_verified'] is False and rendered['cohort']==COHORT,
        'Actual complete non-quality render required')
    client=PrivatePreviews();client.require_private();files=[]
    for row in rendered['reports']:
        for extension,key,mime,cap in [('mp4','video','video/mp4',2_000_000),('jpg','poster','image/jpeg',100000)]:
            p=output/f"episode_{row['episode']:06d}.{extension}"
            require(identity(p,cap)==row[key],'Original bounded rendered bytes required')
            files.append(client.upload(f"full4d-{rev}/episode_{row['episode']:06d}.{extension}",
                p.read_bytes(),mime,rev)|dict(episode_index=row['episode']))
    receipt=dict(schema='world_reward.end2end_preview_publication.v1',status='pass',producer_revision=rev,
        render_revision=render_revision,candidate_revision=rendered['candidate_revision'],
        baseline_revision=BASELINE,render_report_pin=pin,cohort=COHORT,states=rendered['states'],files=files,
        source_binding=binding,endpoint='https://stworldrewardresearch26.blob.core.windows.net/qa-previews',
        private_container_verified=True,public_access_changed=False,account_keys_used=False,
        quality_verified=False,heavy_data_uploaded=False)
    p=ROOT/'results'/('end2end-preview-publication-'+rev+'.json')
    with p.open('xb') as stream: stream.write((json.dumps(receipt,sort_keys=True)+'\n').encode())
    p.chmod(0o444);print(json.dumps(dict(status='published',receipt=str(p),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('render','publish'));p.add_argument('revision')
    args=p.parse_args();(render if args.mode=='render' else publish)(args.revision)

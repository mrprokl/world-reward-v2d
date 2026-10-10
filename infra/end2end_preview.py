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
KINDS = {'joint': 'native-joint-real-', 'continuation': 'native-contact-continuation-real-'}
CONTINUATION_ENTRY = 'run_native_contact_continuation_real'
CONTINUATION_STATUSES = {'accepted_native_continuation', 'dynamic_A_fallback_no_improvement'}
FROZEN_NATIVE_KEYS = ('mhr_global_rot6d', 'mhr_shape', 'mhr_scale', 'mhr_hand', 'mhr_face')
DISPLAY_TRAJECTORY_KEYS = ('object_vertices', 'object_faces', 'object_rotation',
    'object_translation', 'object_scale', 'camera_K', 'frame_index')


def revision(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}', value), 'Exact revision required')
    return value


def candidate_kind(value):
    require(type(value) is str and value in KINDS, 'Explicit joint or continuation candidate required')
    return value


def display_trajectory(trajectory):
    """Project native export controls onto the strict saved-geometry viewer ABI.

    Native pose/shape/scale controls remain available for the separate producer
    checks. The viewer accepts exactly seven geometric fields; return the same
    array objects without repairing, casting, retiming or rescaling anything.
    """
    require(set(DISPLAY_TRAJECTORY_KEYS) <= set(trajectory), 'Complete saved geometry fields required')
    return {key: trajectory[key] for key in DISPLAY_TRAJECTORY_KEYS}


def labels(episode, accepted, kind='joint', status=None):
    candidate_kind(kind)
    title = 'Native joint — QA ' + ('PASS, GT pending' if accepted else 'FAIL / not adopted')
    if kind == 'continuation':
        require(status in CONTINUATION_STATUSES, 'Original native continuation outcome required')
        title = ('Baseline fallback / no gain' if status == 'dynamic_A_fallback_no_improvement' else
                 'C contact-continuation — QA ' + ('PASS, GT pending' if accepted else 'FAIL / not adopted'))
    return ['Original RGB', 'Baseline complete (052)',
            title]


def continuation_cohort(ledger, candidate_root, candidate_revision):
    """Authenticate the complete native producer and all four retained statuses."""
    path = candidate_root/'report.json'; identity(path, 4<<20)
    r = strict(ledger.read(path, maximum=4<<20))
    require(r.get('schema') == 'world_reward.native_contact_continuation_real.v1'
        and r.get('status') == 'complete_native_continuation_diagnostic'
        and r.get('producer_revision') == candidate_revision and r.get('source_inputs_rehashed') is True
        and r.get('unexpected_failures') == 0 and r.get('native_layers_loaded') == 1
        and r.get('ground_truth_used') is False and r.get('private_truth_read') is False
        and r.get('baseline_modified') is False and r.get('production_adopted') is False
        and r.get('cohort') == dict(random_seed=20261008, population=30, episodes=COHORT),
        'Complete sealed no-GT native continuation cohort required')
    rows = r.get('episodes', [])
    require(type(rows) is list and all(type(v) is dict for v in rows)
        and [v.get('episode') for v in rows] == COHORT,
        'All original continuation statuses required')
    for row in rows:
        require(row.get('status') in (CONTINUATION_STATUSES if row['episode'] in (9,14) else
            {'unsupported_original_frontend_unchanged'}), 'No failed/sampled continuation records')
        if row['episode'] in (1,7):
            require(row.get('baseline_retained') is True and row.get('rerolled') is False
                and row.get('fabricated_predictions') is False and type(row.get('reason')) is str,
                'Unsupported records retain original baseline without reroll')
    oldcode = ROOT/'jobs'/candidate_revision/CONTINUATION_ENTRY/'code'
    binding = r['source_binding']; helpers = binding.get('helpers', {})
    require({'infra/native_contact_continuation_real.py', 'infra/run_native_contact_continuation_real.sh',
        'src/world_reward/native_contact_continuation.py'} <= set(helpers)
        and source(ROOT, oldcode, candidate_revision, CONTINUATION_ENTRY, tuple(helpers)) == binding,
        'Original immutable native continuation source binding required')
    host = candidate_root/'host-exit.json'; identity(host, 4096)
    exit_receipt = strict(ledger.read(host, maximum=4096))
    require(exit_receipt == dict(producer_revision=candidate_revision, container_absence_verified=True,
        process_exit_code=0, GPU_requested=True), 'Actual completed owned native GPU cleanup required')
    require(type(r.get('input_ledger')) is dict and r['input_ledger'], 'Full producer source ledger required')
    return r


def complete_candidate(r, kind, total):
    """Method-specific completion gates; historical joint ABI remains unchanged."""
    if kind == 'joint':
        require(r.get('native_effective_updates') == 301 and r.get('fitted_outputs_sealed_before_QA') is True
            and r.get('source_inputs_rehashed') is True, 'Complete sealed native candidate required')
        return r['decision']['passed']
    require(r.get('status') in CONTINUATION_STATUSES
        and r.get('fitted_outputs_sealed_before_final_QA') is True
        and r.get('full_original_frames') == total and r.get('source_fps') == 30.
        and r.get('native_direct136_generated') is True and r.get('stale_B_pose_used') is False
        and r.get('original_activations_unchanged') is True and r.get('whole_hand_minimum_claimed') is False
        and r.get('penetration_evaluated') is False and r.get('hand_labeled_test') is False
        and r.get('oracle_modes') == [], 'Complete full-native same-witness continuation required')
    c = r.get('continuation', {})
    require(c.get('status') == r['status'] and c.get('full_original_frames') == total
        and c.get('clip_global_alpha') is True and c.get('ground_truth_used') is False
        and c.get('private_truth_read') is False and c.get('metric_improvement_claimed') is False
        and ((r['status'] == 'dynamic_A_fallback_no_improvement' and c.get('alpha') == 0.) or
             (r['status'] == 'accepted_native_continuation' and type(c.get('alpha')) in (int,float)
              and 0 < c['alpha'] <= 1)), 'Original truthful continuation/fallback outcome required')
    require(type(r.get('decision', {}).get('C_vs_A', {}).get('passed')) is bool,
        'Actual final C versus original A QA decision required')
    return r['decision']['C_vs_A']['passed']


def probe_contract(probe, total):
    streams = probe.get('streams', [])
    require(len(streams) == 1, 'Exactly one encoded video stream required')
    s = streams[0]
    require(s.get('codec_name') == 'h264' and s.get('width') == 960 and s.get('height') == 280
        and s.get('r_frame_rate') == '30/1' and s.get('avg_frame_rate') == '30/1'
        and s.get('nb_read_frames') == str(total), 'Full original timeline / fixed viewports required')


def render(candidate_revision, kind='joint'):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import full4d_video as viewer
    from sequence_pose_probe import ArtifactLedger

    started = time.monotonic(); rev = revision(os.environ['WR_CODE_REVISION'])
    code = canonical(os.environ['WR_CODE']); candidate_revision = revision(candidate_revision); candidate_kind(kind)
    require(code == ROOT/'jobs'/rev/ENTRY/'code', 'Immutable render closure required')
    binding = source(ROOT, code, rev, ENTRY, HELPERS)
    output = ROOT/'results'/('end2end-preview-'+rev)
    require(output.is_dir() and set(p.name for p in output.iterdir()) == {'.container.cid'},
            'Fresh owned rendering directory required')
    ledger = ArtifactLedger(); reports = []; states = []
    base = ROOT/'experiments'/('full4d-v1-'+BASELINE)
    candidate_root = ROOT/'results'/(KINDS[kind]+candidate_revision)
    cohort = continuation_cohort(ledger, candidate_root, candidate_revision) if kind == 'continuation' else None
    cohort_rows = {r['episode']: r for r in cohort['episodes']} if cohort else {}
    for ep in COHORT:
        candidate = candidate_root/f'episode_{ep:06d}'
        report_path = candidate/'report.json'
        if kind == 'continuation' and ep in (1,7):
            states.append(dict(episode=ep, status='not_reconstructed_in_this_ablation',
                reason=cohort_rows[ep]['reason']))
            continue
        if not report_path.exists():
            states.append(dict(episode=ep, status='not_reconstructed_in_this_ablation',
                reason='upstream full-pose unsupported' if ep in (1,7) else 'candidate missing'))
            continue
        if kind == 'continuation': identity(report_path, 4<<20)
        r = strict(ledger.read(report_path, cohort_rows[ep]['report'] if cohort else None, maximum=4<<20))
        require(r['producer_revision'] == candidate_revision and r['episode'] == ep
                and r['ground_truth_used'] is False and r['private_truth_read'] is False
                and r['baseline_modified'] is False, 'Automatic no-GT candidate provenance required')
        if r.get('status') not in (CONTINUATION_STATUSES if cohort else {'complete_diagnostic_not_quality_pass'}):
            states.append(dict(episode=ep, status='failed', phase=r.get('phase'),
                reason=r.get('error', 'no complete candidate geometry')))
            continue
        directory = base/'outputs'/f'episode_{ep:06d}'/'cari_shared_export_v1'
        pins = strict(ledger.read(base/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'))
        spec = pins['clip_spec']; total = spec['total_frames']
        accepted = complete_candidate(r, kind, total)
        if cohort:
            require(total == {9:415,14:442}[ep] and r['status'] == cohort_rows[ep]['status'],
                'Exact full original continuation timeline/outcome required')
        require(spec == dict(episode_index=ep, total_frames=total, camera_name='front_stereo_camera_left',
            width=1536, height=1152), 'Original source grid required')
        for name in FILES.values(): ledger.record(directory/name, pins['export_files'][name])
        for key, name in FILES.items():
            if cohort: identity(candidate/name, 2<<30)
            ledger.record(candidate/name, r['candidate_outputs'][key])
        require(r['baseline_binding']['outputs'] == pins['export_files'], 'Same frozen baseline required')
        if cohort:
            require(r['baseline_binding']['directory'] == str(directory), 'Same original baseline route required')
            witness = candidate/'QA_witnesses.npz'; identity(witness, 2<<20)
            ledger.record(witness, r['candidate_outputs']['frozen_witnesses'])
            with np.load(witness, allow_pickle=False) as qa:
                require(set(qa.files) == {'activations','hand_vertex_ids','frame_index','emitted_same_witness_gaps_m'}
                    and qa['activations'].dtype == np.bool_ and qa['activations'].shape == (total,2)
                    and qa['hand_vertex_ids'].dtype == np.int64 and qa['hand_vertex_ids'].shape == (total,2)
                    and np.array_equal(qa['frame_index'], np.arange(total))
                    and qa['emitted_same_witness_gaps_m'].shape == (total,2)
                    and np.isfinite(qa['emitted_same_witness_gaps_m'][qa['activations']]).all(),
                    'Sealed original all-frame automatic witness geometry required')
        ah = np.load(directory/'target.npy', mmap_mode='r', allow_pickle=False)
        bh = np.load(candidate/'target.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory/'trajectory.npz', allow_pickle=False) as a:
            keys = ('object_vertices','object_faces','object_rotation','object_translation',
                    'object_scale','camera_K','frame_index') + (('pose','scales','shape') if cohort else ())
            a = {k:a[k] for k in keys}
        with np.load(candidate/'trajectory.npz', allow_pickle=False) as b:
            b = {k:b[k] for k in a}
        with np.load(directory/'native_parameters.npz', allow_pickle=False) as native:
            faces = native['human_faces']; indices = native['frame_index']
            fixed_keys = FROZEN_NATIVE_KEYS
            frozen = {k:native[k] for k in fixed_keys} if cohort else {}
            internal = native['mhr_body_pose_cont'][:,254:].copy() if cohort else None
        with np.load(candidate/'native_parameters.npz', allow_pickle=False) as native:
            require(np.array_equal(native['human_faces'], faces)
                and np.array_equal(native['frame_index'], indices), 'No topology / timeline substitutions')
            if cohort:
                require(all(native[k].dtype == frozen[k].dtype and native[k].shape == frozen[k].shape
                    and native[k].tobytes() == frozen[k].tobytes() for k in fixed_keys)
                    and native['mhr_body_pose_cont'][:,254:].tobytes() == internal.tobytes()
                    and native['mhr_trans'].dtype == np.float32 and native['mhr_trans'].shape == (total,3)
                    and native['mhr_joints'].dtype == np.float32 and native['mhr_joints'].shape == (total,127,3)
                    and native['mhr_keypoints'].dtype == np.float32 and native['mhr_keypoints'].shape == (total,70,3),
                    'Frozen native identity, hands, root rotation and complete native C geometry required')
        require(ah.shape == bh.shape == (total,18439,3), 'Every original human vertex/frame required')
        if cohort:
            require(ah.dtype == bh.dtype == np.float32 and b['pose'].dtype == np.float32
                and b['pose'].shape == (total,136) and np.isfinite(b['pose']).all()
                and b['scales'].tobytes() == a['scales'].tobytes()
                and b['shape'].tobytes() == a['shape'].tobytes()
                and np.array_equal(a['object_rotation'], b['object_rotation']),
                'Direct full-native C controls and unchanged clip shape/scales/object rotations required')
        for key in ('object_vertices','object_faces','object_scale','camera_K','frame_index'):
            require(np.array_equal(a[key], b[key]), 'Clip-constant geometry, K, scale and original indices required')
        viewer.checked_geometry(ah, faces, display_trajectory(a), indices, total)
        viewer.checked_geometry(bh, faces, display_trajectory(b), indices, total)
        camera = viewer.display_intrinsics(a['camera_K'], 1536, 1152)
        floor = viewer.fixed_floor_height(ah)
        ra = viewer.SavedSceneRenderer(camera, faces, a['object_faces'], floor)
        rb = viewer.SavedSceneRenderer(camera, faces, b['object_faces'], floor)
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{ep:06d}.mp4'
        input_ledger = cohort['input_ledger'] if cohort else r['input_ledger']
        require(str(video) in input_ledger, 'Source RGB must be bound by native candidate')
        ledger.record(video, input_ledger[str(video)])
        capture = cv2.VideoCapture(str(video))
        require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
            and abs(capture.get(cv2.CAP_PROP_FPS)-30) < .003, 'Original RGB timeline required')
        template = Image.new('RGB', (960,280), (24,27,33))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
        draw = ImageDraw.Draw(template)
        for column, title in enumerate(labels(ep, accepted, kind, r['status'])):
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
        if cohort:
            report['candidate_status'] = r['status']; states[-1]['candidate_status'] = r['status']
        print(json.dumps(dict(stage='full_video_rendered',episode=ep,frames=total,QA_passed=accepted)),flush=True)
    require(reports, 'No completed candidate available for honest visual comparison')
    ledger.verify(); require(source(ROOT,code,rev,ENTRY,HELPERS)==binding, 'Immutable source changed')
    result = dict(schema='world_reward.end2end_preview.v1',status='complete',producer_revision=rev,
        candidate_revision=candidate_revision,baseline_revision=BASELINE,cohort=COHORT,states=states,
        source_binding=binding,sources=ledger.records,reports=reports,quality_verified=False,
        source_rehashed_after=True,heavy_media_local=False,elapsed_seconds=time.monotonic()-started)
    if cohort: result.update(schema='world_reward.end2end_preview.v2', candidate_kind=kind)
    path=output/'render.json'
    with path.open('xb') as stream: stream.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode())
    path.chmod(0o444)


def publish(render_revision, kind='joint'):
    from full4d_publish import PrivatePreviews
    rev=revision(os.environ['WR_CODE_REVISION']); render_revision=revision(render_revision)
    candidate_kind(kind); code=canonical(os.environ['WR_CODE']); binding=source(ROOT,code,rev,ENTRY,HELPERS)
    output=ROOT/'results'/('end2end-preview-'+render_revision)
    pin=identity(output/'render.json',4<<20); rendered=strict((output/'render.json').read_bytes())
    require(rendered['status']=='complete' and rendered['producer_revision']==render_revision
        and rendered['quality_verified'] is False and rendered['cohort']==COHORT,
        'Actual complete non-quality render required')
    require((kind == 'joint' and rendered.get('schema') == 'world_reward.end2end_preview.v1'
            and 'candidate_kind' not in rendered) or
        (kind == 'continuation' and rendered.get('schema') == 'world_reward.end2end_preview.v2'
            and rendered.get('candidate_kind') == kind), 'Explicit method matches original render receipt')
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
    if kind == 'continuation':
        receipt.update(schema='world_reward.end2end_preview_publication.v2', candidate_kind=kind)
    p=ROOT/'results'/('end2end-preview-publication-'+rev+'.json')
    with p.open('xb') as stream: stream.write((json.dumps(receipt,sort_keys=True)+'\n').encode())
    p.chmod(0o444);print(json.dumps(dict(status='published',receipt=str(p),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('render','publish'));p.add_argument('revision')
    p.add_argument('--candidate-kind', choices=tuple(KINDS), default='joint')
    args=p.parse_args();(render if args.mode=='render' else publish)(args.revision, args.candidate_kind)

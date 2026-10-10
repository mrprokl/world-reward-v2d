"""Azure CPU one bounded Gemini late-view request for each frozen failed clip.

Two original frames only are decoded. Saved forward masks/input/source hashes
are verified; no segmentation, GPU, whole-video transfer or human labels. The
existing approved Vertex3.5Flash route receives RAM-only one-shot OAuth. Accepted
native-box anchors remain proposals for a subsequent RGB-verified reverse pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, BASE, inputs
from sam31_runtime import control
from sam31_recovery import settings as recovery_settings, saved_forward, read_mask, mask_box
from sam31_late_anchor import (
    AnchorPolicy, MODEL, PROJECT, LOCATION, ENDPOINT, build_request, corroborate, verify_one_request,
)
from world_reward.vertex_retry import RetryExhausted, PermanentFailure

ENTRY = 'run_sam31_late_anchor'
CONFIG = 'configs/sam31_late_anchor_v1.json'
KEY = ROOT / '.secrets/sam31-late-anchor-v1-key.pem'
ENVELOPE = ROOT / '.secrets/sam31-late-anchor-v1.enc'
HELPERS = ('infra/sam31_late_anchor_run.py', 'infra/run_sam31_late_anchor.sh', CONFIG,
           'infra/sam31_late_anchor.py', 'infra/sam31_recovery.py', 'configs/sam31_recovery_v1.json',
           'src/world_reward/gemini_localization.py', 'src/world_reward/vertex_retry.py')


def save(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode(), mode=0o444)


def settings(code):
    c = strict((code / CONFIG).read_bytes())
    require(c['schema'] == 'world_reward.sam31_late_anchor_run.v1'
            and c['episodes'] == [1, 7] and c['original_cohort'] == [9, 1, 14, 7]
            and c['model'] == MODEL and c['project'] == PROJECT and c['location'] == LOCATION
            and c['cpu_image'] == BASE and c['reference_frame'] == 0
            and c['anchor_policy'] == 'one_last_saved_forward_visible_frame'
            and c['frames_decoded_per_clip'] == 2 and c['concurrency'] == 2
            and c['budget_seconds'] == 300 and c['attempt_timeout_seconds'] == 60
            and c['ground_truth_used'] is False and c['manual_labels'] is False
            and c['quality_verified'] is False and c['baseline_modified'] is False,
            'Frozen two-clip, two-image, existing Vertex CPU diagnostic required')
    rc, _ = recovery_settings(code)
    require(c['saved_forward_producer'] == rc['saved_forward_producer']
            and c['saved_forward_native_report'] == rc['saved_forward_native_report']
            and rc['episodes'] == c['original_cohort'], 'Same independently pinned original forward cohort required')
    AnchorPolicy(**c['anchor_gates'])
    return c, rc


def final_visible(areas):
    require(type(areas) is list and len(areas) > 30 and all(type(v) is int and v >= 0 for v in areas),
            'Literal full-T original native object pixel areas required')
    visible = [i for i, value in enumerate(areas) if value > 0]
    require(visible and visible[-1] > 29, 'No independent late native visibility available')
    return visible[-1]


def decode_frame(video, index, expected_rgb_sha256):
    """Original-index OpenCV seek with measured frame-position and RGB proof."""
    import cv2
    import numpy as np
    from PIL import Image
    require(type(index) is int and index >= 0 and re.fullmatch('[0-9a-f]{64}', expected_rgb_sha256),
            'Original requested frame/RGB checksum required')
    cap = cv2.VideoCapture(str(video)); require(cap.isOpened(), 'Original Track1 video must open')
    try:
        require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), 'Original-index decoder seek unsupported')
        position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        require(math.isfinite(position) and abs(position - index) < .01, 'Decoder did not seek requested original index')
        ok, bgr = cap.read(); require(ok, 'Original requested frame decode failed')
        position_after = cap.get(cv2.CAP_PROP_POS_FRAMES)
        require(math.isfinite(position_after) and abs(position_after - (index + 1)) < .01,
                'Decoder did not return requested original frame')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
                and hashlib.sha256(rgb.tobytes()).hexdigest() == expected_rgb_sha256,
                'Sought original RGB bytes differ from saved full-grid decoder')
        im = Image.fromarray(rgb); encoded = BytesIO(); im.save(encoded, format='PNG')
        return encoded.getvalue(), im.width, im.height
    finally:
        cap.release()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Vertex redirects prohibited')


def transport(token):
    """The only API operation; one bounded request, no SDK/internal retries."""
    require(type(token) is str and 0 < len(token) < 16384 and not any(ch.isspace() for ch in token),
            'Existing RAM-only OAuth token required')
    def send(payload, timeout):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(ENDPOINT, data=payload, headers={
            'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(200001)
            require(response.status == 200 and len(raw) <= 200000, 'Bounded successful Vertex response required')
            return raw
    return send


def private_secret(path):
    canonical(path); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size > 0
            and before.st_size <= 16384 and not before.st_mode & 0o077,
            'Exact private single-link one-shot credential envelope required')
    return tuple(getattr(before, key) for key in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink'))


def remove_owned_secret(path, expected):
    """Delete only the originally inspected private file, never a replacement."""
    if not path.exists(): return
    require(path in (KEY, ENVELOPE) and private_secret(path) == expected, 'One-shot secret changed; cannot delete replacement')
    path.unlink()


def native(code, out):
    require(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void', 'Azure CPU-only API image required')
    c, rc = settings(code); started = time.monotonic(); token = sys.stdin.buffer.read(16385).decode().strip()
    send = transport(token); selected = inputs(c['original_cohort']); old, bound, old_binding = saved_forward(code, rc, selected)
    # Entire original source/input/mask cohort is audited before the first billable call.
    tasks = []
    for item in selected:
        ep = item['episode']
        if ep not in c['episodes']: continue
        original = old[ep]; tracking = original['tracking']; late = final_visible(tracking['areas']['1'])
        require(tracking['areas']['0'][late] > 0, 'Same native actor must be visible on the fixed late view')
        seed_hash, late_hash = (tracking['records'][i]['decoded_rgb_sha256'] for i in (0, late))
        seed_png, width, height = decode_frame(item['video'], 0, seed_hash)
        late_png, lw, lh = decode_frame(item['video'], late, late_hash)
        require((width, height) == (lw, lh), 'Original clip grid must be constant')
        initial = [row for row in original['grounding']['records'] if row['frame_index'] == 0]
        require([r['object_id'] for r in initial] == [0, 1]
                and all(r['rgb_sha256'] == seed_hash for r in initial), 'All original automatic seed boxes/RGB required')
        boxes = {}
        for role, name in ((0, 'person'), (1, 'object')):
            key = f'{role}/{late:06d}.png'
            mask = read_mask(original['directory'] / 'masks' / key, original['inventory'][key], height, width)
            require(int(mask.sum()) == tracking['areas'][str(role)][late], 'Literal native late mask pixel support differs')
            boxes[name] = mask_box(mask)
        body, binding = build_request(seed_png=seed_png, late_png=late_png, late_frame_index=late,
            width=width, height=height, seed_rgb_sha256=seed_hash, late_rgb_sha256=late_hash,
            seed_person_bbox=initial[0]['box'], seed_object_bbox=initial[1]['box'],
            object_description=item['object_prompt'], action=item['action'])
        # Images/request exist only in this Azure process's RAM; metadata stays reproducible.
        tasks.append(dict(episode_index=ep, frame_index=late, payload=body, binding=binding,
            native_object_bbox=boxes['object'], native_person_bbox=boxes['person'],
            tracking_sha256=original['pins']['tracking.json']['sha256'],
            frontend_report=original['pins']['report.json'], video_pin=item['video_pin']))
    require([t['episode_index'] for t in tasks] == c['episodes'], 'Both fixed failed clips must be prepared before calls')
    require(time.monotonic() - started < 60, 'Bounded two-frame preparation deadline exceeded')
    policy = AnchorPolicy(**c['anchor_gates'])
    def ask(task):
        ep = task['episode_index']; begin = time.monotonic()
        row = dict(episode_index=ep, frame_index=task['frame_index'], binding=task['binding'],
            frontend_report=task['frontend_report'], video_pin=task['video_pin'], status='technical_failure', accepted=False,
            native_object_bbox=task['native_object_bbox'], native_person_bbox=task['native_person_bbox'])
        def saved_send(payload, timeout):
            raw = send(payload, timeout)
            # Retain the first successful server envelope once, no OAuth header/request images.
            write(out / f'vertex-response_{ep:06d}.json', raw, mode=0o444)
            return raw
        try:
            parsed, ledger = verify_one_request(task['payload'], saved_send, binding=task['binding'],
                deadline=started + c['budget_seconds'] - 5, attempt_timeout=c['attempt_timeout_seconds'])
            anchor = corroborate(parsed, binding=task['binding'], native_object_bbox=task['native_object_bbox'],
                native_person_bbox=task['native_person_bbox'], saved_forward_tracking_sha256=task['tracking_sha256'], policy=policy)
            row.update(anchor, parsed=parsed, request_ledger=ledger)
        except RetryExhausted as error:
            row.update(status='transport_retry_exhausted', retry_reason=error.reason, attempts=list(error.attempts))
        except PermanentFailure as error:
            row.update(status='transport_permanent_error', retry_reason=error.reason, attempts=list(error.attempts))
        except ValueError as error:
            row.update(status='invalid_model_response', failure_type=type(error).__name__)
        response = out / f'vertex-response_{ep:06d}.json'
        if response.exists(): row['vertex_response'] = identity(response, 200000)
        row.update(elapsed_seconds=time.monotonic() - begin, semantic_requests=1, model_content_retries=0,
            images_saved=False, masks_generated=False, ground_truth_used=False, manual_labels=False, quality_verified=False)
        save(out / f'late-anchor_{ep:06d}.json', row)
        return row
    try:
        with ThreadPoolExecutor(max_workers=c['concurrency']) as pool: rows = list(pool.map(ask, tasks))
        require(time.monotonic() - started < c['budget_seconds'], 'Inclusive late-anchor API stage deadline exceeded')
        require(inputs(c['original_cohort']) == selected
                and all(identity(path, max(pin['bytes'], 1)) == pin for path, pin in bound.items()),
                'Original input/full-mask evidence changed')
        from sam31_recovery import FORWARD_ENTRY
        require(source(ROOT, ROOT / 'jobs' / rc['saved_forward_producer'] / FORWARD_ENTRY / 'code',
            rc['saved_forward_producer'], FORWARD_ENTRY, tuple(old_binding['helpers'])) == old_binding,
            'Original immutable forward producer changed')
        save(out / 'late-anchor.json', dict(schema='world_reward.sam31_late_anchor_actual.v1',
            status='complete_diagnostic_not_quality_pass', producer_revision=os.environ['WR_CODE_REVISION'],
            episodes=rows, saved_forward_producer=rc['saved_forward_producer'],
            saved_forward_native_report=rc['saved_forward_native_report'], model=MODEL, project=PROJECT, location=LOCATION,
            config=c, semantic_requests=2, accepted_anchors=sum(bool(row['accepted']) for row in rows),
            elapsed_seconds=time.monotonic() - started, gpu_used=False, masks_generated=False,
            full_video_decode=False, original_frames_decoded=4, images_saved=False,
            ground_truth_used=False, manual_labels=False, quality_verified=False, baseline_modified=False,
            source_rehashed_after=True, inputs_rehashed_after=True))
    finally:
        token = None; send = None
        for task in tasks: task['payload'] = b''


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure host only')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']; c, rc = settings(code)
    binding = source(ROOT, code, revision, ENTRY, HELPERS)
    out = ROOT / 'results' / ('sam31-late-anchor-' + revision); out.mkdir(mode=0o755)
    name = 'wr-sam31-late-anchor-' + revision[:12]; started = time.monotonic(); owned = {}
    report = dict(status='fail', producer_revision=revision, source_binding=binding)
    try:
        # Inspect credentials first so any later preflight failure removes only
        # these exact-owned one-shot files rather than leaving sensitive noise.
        for path in (KEY, ENVELOPE): owned[path] = private_secret(path)
        # Source/input preflight uses only the host stdlib; model/data work is CPU-container-only.
        saved_forward(code, rc, inputs(c['original_cohort']))
        image = strict(control(['docker', 'image', 'inspect', c['cpu_image']]))[0]
        require(image['Id'] == BASE and image['Os'] == 'linux' and image['Architecture'] == 'amd64',
                'Same previously qualified Gemini CPU runtime image required')
        require(not control(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$']).strip(),
                'Fresh exact-owner CPU container name required')
        decrypted = subprocess.run(['openssl', 'pkeyutl', '-decrypt', '-inkey', str(KEY), '-in', str(ENVELOPE),
            '-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256'], capture_output=True, timeout=15)
        require(decrypted.returncode == 0 and 0 < len(decrypted.stdout) < 16384, 'Existing one-shot OAuth decryption failed')
        cmd = ['docker', 'run', '--rm', '-i', '--name', name, '--label', 'world_reward.sam31_late_anchor.owner=' + revision,
            '--network', 'host', '--read-only', '--user', '0:0', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--memory', '8g', '--cpus', '4', '--tmpfs', '/tmp:rw,nosuid,size=1g']
        mounts = [(code.parent, True), (out, False), (ROOT / 'results/input-manifest.json', True),
            (ROOT / 'data/track_1/meta', True),
            (ROOT / 'results' / ('gemini-sam31-' + rc['saved_forward_producer']), True),
            (ROOT / 'jobs' / rc['saved_forward_producer'] / 'run_gemini_sam31_track', True)]
        # saved_forward() verifies the frozen four original RGB identities, though only1/7 are decoded.
        mounts += [(ROOT / f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4', True)
                   for ep in c['original_cohort']]
        for path, readonly in mounts:
            canonical(path); cmd += ['--mount', f'type=bind,src={path},dst={path}' + (',readonly' if readonly else '')]
        cmd += ['--entrypoint', '/usr/bin/env', BASE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=', 'NVIDIA_VISIBLE_DEVICES=void',
            'PYTHONPATH=' + str(code / 'src') + ':' + str(code / 'infra'), 'PYTHONDONTWRITEBYTECODE=1',
            'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2', 'WR_CODE_REVISION=' + revision,
            '/opt/conda/bin/python', '-B', str(code / HELPERS[0]), '--native', str(code), str(out)]
        with (out / 'native.log').open('xb') as log:
            os.fchmod(log.fileno(), 0o400)
            done = subprocess.run(cmd, input=decrypted.stdout, stdout=log, stderr=log, timeout=c['budget_seconds'] + 30)
        decrypted = None
        require(done.returncode == 0, 'Late-anchor CPU native failed; inspect Azure-only log')
        actual = strict((out / 'late-anchor.json').read_bytes())
        report.update(status=actual['status'], native_report=identity(out / 'late-anchor.json', 200000),
            accepted_anchors=actual['accepted_anchors'], episodes=[
                dict(episode_index=row['episode_index'], frame_index=row['frame_index'], status=row['status'],
                     accepted=row['accepted']) for row in actual['episodes']])
    except Exception as error:
        report['error_type'] = type(error).__name__
    finally:
        inspected = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=15)
        if inspected.returncode == 0:
            actual = strict(inspected.stdout)[0]
            require(actual['Image'] == BASE and actual['Name'] == '/' + name
                    and actual['Config']['Labels'].get('world_reward.sam31_late_anchor.owner') == revision,
                    'Can only clean this exact-owned CPU container')
            subprocess.run(['docker', 'rm', '-f', actual['Id']], capture_output=True, timeout=20, check=True)
        for path, expected in owned.items(): remove_owned_secret(path, expected)
        require(source(ROOT, code, revision, ENTRY, HELPERS) == binding, 'Immutable CPU driver source changed')
        report.update(elapsed_seconds=time.monotonic() - started, credential_files_removed=not KEY.exists() and not ENVELOPE.exists(),
            gpu_used=False, baseline_modified=False, ground_truth_used=False, quality_verified=False)
        save(out / 'report.json', report)
        if report['status'] == 'complete_diagnostic_not_quality_pass': (out / 'native.log').unlink()
    require(report['status'] == 'complete_diagnostic_not_quality_pass', 'Late-anchor API diagnostic infrastructure failed closed')
    print(json.dumps(dict(status=report['status'], accepted_anchors=report['accepted_anchors'],
                         elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--native': native(Path(sys.argv[2]), Path(sys.argv[3]))
    else: run()

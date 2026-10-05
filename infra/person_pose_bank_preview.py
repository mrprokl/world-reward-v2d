"""Tiny Azure CPU QA of every saved person bank, never a target selector.

Colors label original detector slots within a frame, NOT temporal identities.
All native points stay unchanged. Numerically valid, in-grid points alone are
drawn; this display rule is not a visibility, anatomical or accuracy claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import mediapipe_cpu_runtime_verify as rt

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_person_pose_bank_preview'
CONFIG = 'configs/person_pose_bank_preview_v1.json'
HELPERS = ('infra/person_pose_bank_preview.py', 'infra/run_person_pose_bank_preview.sh',
           CONFIG, 'infra/mediapipe_cpu_runtime_verify.py')
EXPECTED_BANKS = ((9, 0, 2), (9, 27, 3), (9, 55, 3), (26, 0, 2), (26, 26, 2), (26, 53, 2))


def authenticate(code, revision):
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    c = rt.strict((code/CONFIG).read_bytes())
    rt.require(c['schema'] == 'world_reward.person_pose_bank_preview.v1'
               and c['all_recorded_banks'] is True and c['manual_selection'] is False
               and c['model_execution'] is False and c['budget_seconds'] == 90,
               'Frozen display-only contract required')
    producer = c['producer_revision']
    base = ROOT/('results/person-pose-bank-probe-'+producer)
    report_path = base/'report.json'
    report = rt.pinned(report_path, c['report'], 32 << 10)
    facts = dict(status='pass', stage='automatic_all_person_dwpose_diagnostic',
                 producer_revision=producer, gpu_used=False, ground_truth_used=False,
                 hand_labeled_test=False, oracle_modes=[], anatomical_ownership_verified=False,
                 all_recorded_person_banks_retained=True, source_inputs_assets_rehashed_after=True)
    rt.require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in facts.items()),
               'Actual complete recorded-bank producer required')
    old = ROOT/'jobs'/producer/'run_person_pose_bank_probe'/'code'
    source = report['source_binding']
    rt.require(rt.source(ROOT, old, producer, 'run_person_pose_bank_probe', tuple(source['helpers'])) == source,
               'Original complete producer source changed')
    scope = rt.strict((old/'configs/person_pose_bank_probe_v1.json').read_bytes())
    rt.require([(b['episode'], b['frame_index'], b['persons']) for b in report['banks']]
               == list(EXPECTED_BANKS), 'All original six banks in original order required')
    files = {str(report_path): c['report']}
    for row in scope['episodes']:
        path = ROOT/row['video']; pin = row['files'][row['video']]
        rt.require(rt.identity(path, 32 << 30, readonly=False) == pin
                   and report['input_pins'][row['video']] == pin, 'Original single RGB changed')
        files[str(path)] = pin
    for bank in report['banks']:
        name = f'episode_{bank["episode"]:06d}_frame_{bank["frame_index"]:06d}.npz'
        rt.require(bank['prediction_file'] == name and len(bank['person_ids']) == bank['persons'],
                   'Exact saved person slots required')
        path = base/name
        rt.require(rt.identity(path, 1 << 20) == bank['prediction'], 'Frozen prediction changed')
        files[str(path)] = bank['prediction']
    return c, report, scope, dict(source=own, original_source=source, files=files)


def display_points(points, native_valid, in_image):
    """Geometric display eligibility only, retaining native arrays untouched."""
    import numpy as np
    rt.require(points.ndim == 3 and points.shape[1:] == (133, 2)
               and native_valid.shape == in_image.shape == points.shape[:2]
               and native_valid.dtype == in_image.dtype == np.bool_ and np.isfinite(points).all(),
               'Original native point/diagnostic arrays required')
    return native_valid & in_image


def render(c, report, scope, out):
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    rt.require(os.geteuid() == 1000 and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'},
               'Offline nonroot Azure CPU preview required')
    height, width = c['viewport_hw']
    sheet = Image.new('RGB', (3*width, 116+2*(height+30)), (23, 26, 31))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for y, text in ((8, 'World Reward | toutes les personnes detectees, aucun acteur choisi'),
                    (29, 'Couleurs = slots de detection dans UNE image, pas identites temporelles'),
                    (50, 'Points = 133 keypoints natifs | cercles: poignets du corps | carres: racines des mains'),
                    (71, 'Pas de labels manuels, de contact verifie ou de score de precision')):
        draw.text((10, y), text, font=font, fill=(235, 235, 235))
    colors = ((30, 210, 230), (255, 170, 50), (188, 128, 255))
    base = ROOT/('results/person-pose-bank-probe-'+c['producer_revision'])
    for row_no, row in enumerate(scope['episodes']):
        cap = cv2.VideoCapture(str(ROOT/row['video']))
        try:
            rt.require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row['total_frames'],
                       'Original complete video required')
            banks = [b for b in report['banks'] if b['episode'] == row['episode']]
            for col, bank in enumerate(banks):
                index = bank['frame_index']
                rt.require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), 'Original frame seek required')
                ok, bgr = cap.read()
                rt.require(ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) == index+1,
                           'Exact original frame decode required')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rt.require(hashlib.sha256(rgb.tobytes()).hexdigest() == bank['decoded_RGB_sha256'],
                           'Decoded pixels differ from native inference')
                with np.load(base/bank['prediction_file'], allow_pickle=False) as a:
                    points = a['keypoints_original_xy']; valid = a['native_valid']; inside = a['in_original_image']
                    eligible = display_points(points, valid, inside)
                    rt.require(np.array_equal(valid, a['raw_scores'] > 0) and len(points) == bank['persons'],
                               'Native validity/complete bank differs')
                    boxes = a['boxes_original_xyxy']
                image = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
                d = ImageDraw.Draw(image); scale = np.array([width/rgb.shape[1], height/rgb.shape[0]])
                for slot in range(bank['persons']):
                    color = colors[slot % len(colors)]
                    box = boxes[slot]*np.tile(scale, 2)
                    d.rectangle(tuple(box), outline=color, width=2)
                    d.text((float(box[0]), max(0., float(box[1])-14)), f'slot {slot}', fill=color, font=font)
                    for point in points[slot][eligible[slot]]*scale:
                        x, y = map(float, point); d.ellipse((x-1, y-1, x+1, y+1), fill=color)
                    for k in (9, 10, 91, 112):
                        if eligible[slot, k]:
                            x, y = map(float, points[slot, k]*scale)
                            shape = (x-3, y-3, x+3, y+3)
                            (d.ellipse if k in (9, 10) else d.rectangle)(shape, outline=color, width=1)
                x, y = col*width, 116+row_no*(height+30)
                draw.text((x+6, y-20), f'EP{row["episode"]:02d} | frame {index} | {bank["persons"]} personnes',
                          font=font, fill=(235, 235, 235))
                sheet.paste(image, (x, y))
        finally:
            cap.release()
    stream = BytesIO(); sheet.save(stream, format='JPEG', quality=c['jpeg_quality'], optimize=True)
    rt.require(len(stream.getvalue()) <= c['maximum_preview_bytes'], 'Tiny JPEG cap exceeded')
    rt.write(out/'all-person-banks.jpg', stream.getvalue(), 0o444)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--child', action='store_true'); args = p.parse_args()
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    c, report, scope, before = authenticate(code, revision)
    out = ROOT/('results/person-pose-bank-preview-'+revision)
    if args.child:
        render(c, report, scope, out); return
    rt.require(os.geteuid() == 0 and not out.exists(), 'Exclusive Azure-only preview required')
    out.mkdir(); os.chown(out, 1000, 1000)
    started = time.monotonic()
    env = dict(PATH='/usr/bin:/bin', DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
    name = 'wr-person-bank-preview-'+revision
    def control(command):
        remaining = c['budget_seconds']-(time.monotonic()-started)
        rt.require(remaining > 0, 'Inclusive CPU preview budget exceeded')
        return subprocess.run(command, env=env, capture_output=True, check=True, timeout=remaining)
    rt.require(control(['docker', 'image', 'inspect', c['image_id'], '--format', '{{.Id}}']).stdout.decode().strip()
               == c['image_id'] and not control(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).stdout.strip(),
               'Original image and absent owned container required')
    command = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(out/'.container.cid'),
               '--label', 'world_reward.person_pose_bank_preview.owner='+revision,
               '--network', 'none', '--read-only', '--user', '1000:1000',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '2', '--memory', '2g',
               '--tmpfs', '/tmp:rw,nosuid,size=64m', '--entrypoint', '/usr/bin/env']
    old = ROOT/'jobs'/c['producer_revision']/'run_person_pose_bank_probe'
    for path in (code.parent, old, *map(Path, before['files'])):
        command += ['--mount', f'type=bind,src={path},dst={path},readonly']
    command += ['--mount', f'type=bind,src={out},dst={out}', c['image_id'], '-i',
                'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
                'PYTHONPATH='+str(code/'infra'), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision,
                'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=2', '/opt/conda/bin/python', '-B',
                str(code/'infra/person_pose_bank_preview.py'), '--child']
    try:
        control(command)
    finally:
        # A failed render still gets bounded owned-container cleanup, without
        # borrowing another job's container or extending the scientific budget.
        def cleanup(command):
            return subprocess.run(command, env=env, capture_output=True, check=True, timeout=10)
        ids = cleanup(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$']).stdout.decode().split()
        if ids:
            cid = (out/'.container.cid').read_text().strip()
            rt.require(ids == [cid], 'Exact owned preview CID required')
            actual = cleanup(['docker', 'inspect', cid, '--format',
                              '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.person_pose_bank_preview.owner"}}'])
            rt.require(actual.stdout.decode().strip() == c['image_id']+'|/'+name+'|'+revision,
                       'Cannot remove foreign preview container')
            cleanup(['docker', 'rm', '-f', cid])
        rt.require(not cleanup(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).stdout.strip(),
                   'Owned preview container survives')
        if (out/'.container.cid').exists(): (out/'.container.cid').chmod(0o444)
    rt.require(authenticate(code, revision)[3] == before, 'Original pixels/predictions/source changed')
    receipt = dict(schema=c['schema'], status='pass', source_binding=before, all_recorded_banks=True,
                   banks=list(EXPECTED_BANKS), model_execution=False, gpu_used=False, manual_selection=False,
                   ownership_verified=False, quality_evaluation=False, inputs_unchanged=True,
                   image=rt.identity(out/'all-person-banks.jpg', c['maximum_preview_bytes']),
                   elapsed_seconds=time.monotonic()-started)
    rt.write(out/'report.json', (json.dumps(receipt, sort_keys=True)+'\n').encode(), 0o444); out.chmod(0o555)
    print(json.dumps(dict(status='pass', image=receipt['image'])))


if __name__ == '__main__':
    main()

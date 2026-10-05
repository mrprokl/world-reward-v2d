"""Tiny Azure CPU QA of all frozen native HOI boxes/links, no interpretation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_hoi_observation_preview'
CONFIG = 'configs/hoi_observation_preview_v1.json'
CONFIG_PIN = dict(bytes=1565, sha256='32972302df47ebaed4bfae17786222c80269ca4314ffcbc48f5ce76f8b36668a')
HELPERS = ('infra/hoi_observation_preview.py', 'infra/run_hoi_observation_preview.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py')


def helper(code):
    p = code/HELPERS[-1]; b = p.read_bytes()
    assert len(b) == 23559 and hashlib.sha256(b).hexdigest() == '936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'
    s = importlib.util.spec_from_file_location('wr_hoi_preview_rt', p)
    rt = importlib.util.module_from_spec(s); s.loader.exec_module(rt); return rt


def render(c, out):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    assert os.geteuid() == 1000 and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}
    with Image.open(c['input']) as image:
        assert image.size == tuple(reversed(c['image_size']))
        image = image.convert('RGB'); image.thumbnail((c['maximum_width'], c['maximum_height']))
    with np.load(c['observations'], allow_pickle=False) as a:
        names = ('query_ids', 'class_ids', 'raw_scores', 'boxes_original_xyxy', 'hand_object_pairs', 'hand_object_logits', 'object_target_pairs', 'object_target_logits')
        v = {k: a[k] for k in names}
    w, h = image.size; sx, sy = w/c['image_size'][1], h/c['image_size'][0]
    canvas = Image.new('RGB', (w, h+132), (22, 24, 29)); canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=15)
    draw.text((8, 6), 'HOI-DETR : toutes les propositions natives', font=font, fill='white')
    draw.text((8, 25), 'Controle externe - pas une validation de contact', font=font, fill=(220, 220, 220))
    boxes = v['boxes_original_xyxy'] * np.array([sx, sy, sx, sy]) + np.array([0, 48, 0, 48])
    colors = ((70, 225, 238), (255, 173, 65), (167, 118, 255))
    roles = ('main', 'objet direct', 'cible outil')
    for i, (box, qid, role, score) in enumerate(zip(boxes, v['query_ids'], v['class_ids'], v['raw_scores'])):
        color = colors[int(role)]; draw.rectangle(tuple(box), outline=color, width=2)
        draw.text((float(box[0]), max(48, float(box[1])-17)), f'{roles[int(role)]} q{int(qid)} / {float(score):.3f}', font=font, fill=color)
    y = h+52
    for pairs, logits, label in ((v['hand_object_pairs'], v['hand_object_logits'], 'H->O'), (v['object_target_pairs'], v['object_target_logits'], 'O->C')):
        for (i, j), raw in zip(pairs, logits):
            # Both native classes/logits retained. No threshold, softmax or top1.
            draw.text((8, y), f'{label} q{int(v["query_ids"][i])}->q{int(v["query_ids"][j])} logits [{float(raw[0]):.3f}, {float(raw[1]):.3f}]', font=font, fill='white')
            y += 18
    assert y <= canvas.height-20
    draw.text((8, canvas.height-18), c['attribution'], font=font, fill=(210, 210, 210))
    path = out/'native-hoi.jpg'
    with path.open('xb') as f: canvas.save(f, format='JPEG', quality=c['quality'], optimize=True)
    assert path.stat().st_size <= c['maximum_preview_bytes']
    path.chmod(0o444)


def run(code, revision, child=False):
    rt = helper(code); c = rt.pinned(code/CONFIG, CONFIG_PIN, 16 << 10)
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS); out = ROOT/c['output']
    files = {c[k]: c['host_pin' if k == 'host_report' else k+'_pin'] for k in ('input', 'observations', 'native', 'host_report')}
    for path, pin in files.items(): assert rt.identity(Path(path), 16 << 20) == pin
    n = rt.strict(Path(c['native']).read_bytes()); h = rt.strict(Path(c['host_report']).read_bytes())
    assert n['status'] == h['status'] == 'pass' and h['producer_revision'] == c['source_producer_revision']
    assert n['observations'] == c['observations_pin'] and n['actual_native_pair_head_executed'] is True
    if child:
        render(c, out); return
    assert os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02' and not out.exists()
    out.mkdir(); os.chown(out, 1000, 1000); started = time.monotonic()
    env = dict(PATH='/usr/bin:/bin', DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
    name = 'world-reward-hoi-preview-'+revision[:12]
    assert not subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], env=env, capture_output=True, text=True, check=True).stdout.strip()
    args = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '2', '--memory', '2g', '--tmpfs', '/tmp:rw,nosuid,size=64m']
    for a in (code.parent, *map(Path, files)): args += ['--mount', f'type=bind,src={a},dst={a},readonly']
    args += ['--mount', f'type=bind,src={out},dst={out}', '--entrypoint', '/usr/bin/env', c['image_id'], '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE=1', '/opt/conda/bin/python', '-I', '-B', str(code/HELPERS[0]), '--code', str(code), '--revision', revision, '--child']
    try:
        subprocess.run(args, env=env, check=True, timeout=c['budget_seconds'])
    finally:
        if subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], env=env, capture_output=True, text=True, check=True).stdout.strip():
            subprocess.run(['docker', 'rm', '-f', name], env=env, check=True, timeout=10)
    for path, pin in files.items(): assert rt.identity(Path(path), 16 << 20) == pin
    assert rt.source(ROOT, code, revision, ENTRY, HELPERS) == own
    value = dict(schema=c['schema'], status='pass', source_binding=own, inputs=files, preview=rt.identity(out/'native-hoi.jpg', c['maximum_preview_bytes']), all_native_proposals_and_pairs=True, reference_geometry_used=False, manual_annotations=False, fitting=False, elapsed_seconds=time.monotonic()-started)
    rt.write(out/'report.json', (json.dumps(value, sort_keys=True)+'\n').encode(), 0o444); out.chmod(0o555)
    print(json.dumps(value['preview']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--code', type=Path, required=True); p.add_argument('--revision', required=True); p.add_argument('--child', action='store_true')
    a = p.parse_args(); run(a.code, a.revision, a.child)

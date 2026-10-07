"""Azure-only isolated initializer screen, before costly full-4D replay.

Same native body/depth/scale/object functions as the historical baseline.
Only masks and output storage change. This is not a trajectory submission.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from qwen4d_masks import prepare_masks, PRODUCER
from world_reward.artifact_paths import episode_output

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_qwen4d_initializers'
CONFIG = 'configs/qwen4d_initializers_v1.json'
STAGES = (
    ('body_smoke', 'infra/body_smoke.py', ('--inference-type', 'body'), 'body_image'),
    ('depth_smoke', 'infra/depth_smoke.py', (), 'body_image'),
    ('scale_smoke', 'infra/scale_smoke.py', (), 'body_image'),
    ('object_grounded', 'infra/object_smoke.py', ('--aligned-pointmap',), 'object_image'),
)
HELPERS = ('infra/qwen4d_initializers.py', 'infra/run_qwen4d_initializers.sh',
    'infra/qwen4d_masks.py', 'infra/qwen4d_preview.py', 'src/world_reward/artifact_paths.py',
    CONFIG, 'configs/task_grounding_saved_pins.json', 'configs/task_grounding_pilot_v1.json',
    'infra/body_smoke.py', 'infra/depth_smoke.py', 'infra/scale_smoke.py', 'infra/object_smoke.py')


def worker_command(code, outputs, preview, cfg, episode, script, args, image, name, owner):
    """No writable baseline outputs, data, weights, source, or global cache."""
    preview_mode = script == 'infra/qwen4d_preview.py'
    command = ['docker', 'run', '--rm', '--name', name, '--label', 'world_reward.qwen4d.owner='+owner,
        '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--memory', '160g', '--cpus', '16',
        '--shm-size', '2g', '--tmpfs', '/tmp:rw,nosuid,size=4g']
    if not preview_mode: command += ['--gpus', 'all']
    mounts = [(code, True), (ROOT/'vendor', True), (ROOT/'weights', True),
              (ROOT/'results', True), (ROOT/'data/track_1/meta', True), (outputs, preview_mode)]
    for ep in cfg['episodes']:
        mounts.append((ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4', True))
    if preview_mode:
        mounts += [(ROOT/f'outputs/episode_{episode:06d}/cari_shared_export_v1', True), (preview, False)]
    for path, readonly in mounts:
        canonical(path)
        command += ['--mount', f'type=bind,src={path},dst={path}'+(',readonly' if readonly else '')]
    env = ['PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp', 'WR_ROOT='+str(ROOT),
        'WR_CODE='+str(code), 'WR_CODE_REVISION='+os.environ['WR_CODE_REVISION'],
        'WR_OUTPUT_PREFIX='+os.environ['WR_OUTPUT_PREFIX'],
        'PYTHONPATH='+str(code/'src')+':'+str(code/'infra')+
        ':/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose',
        'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        'MOMENTUM_ENABLED=0', 'WANDB_MODE=disabled', 'OMP_NUM_THREADS=8', 'MPLBACKEND=Agg',
        'TORCH_HOME='+str(ROOT/('weights/sam3d/torch_home' if image == cfg['object_image'] else 'weights/cari4d/sam3d_body/torch_home')),
        'HF_HOME='+str(ROOT/('weights/sam3d/hf_home' if image == cfg['object_image'] else 'weights/cari4d/hf_home'))]
    return command + ['--entrypoint', '/usr/bin/env', image, '-i', *env,
        '/opt/conda/bin/python', '-B', str(code/script), '--episode', str(episode), *args]


def native(code, outputs, preview, logs, cfg, episode, stage, script, args, image, seconds, owner):
    name = f'world-reward-qwen4d-{owner[-12:]}-{episode}-{stage.replace("_", "-")}'
    require(not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], text=True).strip(),
            'Fresh worker only; never resume or replace')
    started = time.monotonic()
    try:
        with (logs/f'{episode}-{stage}.log').open('xb') as log:
            proc = subprocess.run(worker_command(code, outputs, preview, cfg, episode, script, args, image, name, owner),
                                  stdout=log, stderr=log, timeout=seconds, check=False)
        require(proc.returncode == 0, f'{stage} failed; no automatic scientific retry')
    finally:
        found = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=15, check=False)
        if found.returncode == 0:
            value = strict(found.stdout)[0]
            require(value['Image'] == image and value['Config']['Labels'].get('world_reward.qwen4d.owner') == owner,
                    'Container ownership changed; do not remove')
            subprocess.run(['docker', 'rm', '-f', value['Id']], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20, check=True)
    return time.monotonic()-started


def run():
    require(os.uname().sysname == 'Linux' and os.geteuid() == 0 and
            os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure VM01 host driver only')
    root = canonical(ROOT); code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    binding = source(root, code, revision, ENTRY, HELPERS)
    cfg = strict((code/CONFIG).read_bytes())
    require(cfg['schema'] == 'world_reward.qwen4d_initializers.v1' and cfg['episodes'] == [8, 9, 26]
            and cfg['scope'] == 'three_frame_geometry_QA_before_full_video_4D', 'Frozen geometry screen required')
    experiment = root/'experiments'/('qwen4d-v1-'+revision)
    require(not experiment.exists() and not experiment.is_symlink(), 'Fresh experiment; preserve baseline')
    experiment.parent.mkdir(exist_ok=True)
    experiment.mkdir(mode=0o755); outputs = experiment/'outputs'; outputs.mkdir(mode=0o755)
    os.chown(outputs, 1000, 1000)
    previews = experiment/'previews'; previews.mkdir(mode=0o755)
    logs = experiment/'logs'; logs.mkdir(mode=0o700)
    os.environ['WR_OUTPUT_PREFIX'] = str(outputs.relative_to(root))
    report = dict(schema=cfg['schema'], status='fail', phase='preflight', producer_revision=revision,
        source_binding=binding, scope=cfg['scope'], baseline_modified=False, model_math_changed=False,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], full_4D_run=False,
        quality_verified=False, training_overlap_verified=False, challenge_overlap_verified=False, episodes=[])
    started = time.monotonic()
    def interrupted(*_): raise TimeoutError('Bounded pilot interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        with (root/'jobs/.world-reward-h100.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip(), 'No duplicate GPU jobs')
            for image in (cfg['body_image'], cfg['object_image']):
                value = strict(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
                require(value['Id'] == image, 'Pinned existing native image required')
            for episode in cfg['episodes']:
                row = dict(episode=episode, status='fail', phase='mask_adapter', stages=[]); report['episodes'].append(row)
                base = episode_output(root, episode); base.mkdir(mode=0o755); os.chown(base, 1000, 1000)
                try:
                    report['phase'] = f'episode_{episode}'
                    masks = base/'automatic_masks'; receipt = prepare_masks(root, code, episode, masks)
                    for path in (masks, *masks.rglob('*')): os.chown(path, 1000, 1000)
                    row['masks'] = identity(masks/'report.json'); row['full_mask_frames'] = receipt['frames']
                    for stage, script, args, image_key in STAGES:
                        row['phase'] = stage
                        require(time.monotonic()-started < cfg['pilot_budget_seconds'], 'Inclusive pilot budget exceeded')
                        elapsed = native(code, outputs, None, logs, cfg, episode, stage, script, args, cfg[image_key],
                            min(cfg['stage_budget_seconds'], cfg['pilot_budget_seconds']-(time.monotonic()-started)), revision)
                        path = base/stage/'report.json'; saved = strict(path.read_bytes())
                        require(saved['status'] == 'pass' and saved['episode_index'] == episode
                                and saved['ground_truth_used'] is False and saved['hand_labeled_test'] is False
                                and saved['oracle_modes'] == [], 'Native producer provenance differs')
                        row['stages'].append(dict(stage=stage, elapsed_seconds=elapsed, report=identity(path, readonly=False)))
                    row['phase'] = 'preview'; preview = previews/f'episode_{episode:06d}'
                    preview.mkdir(mode=0o755); os.chown(preview, 1000, 1000)
                    native(code, outputs, preview, logs, cfg, episode, 'preview', 'infra/qwen4d_preview.py',
                        ('--output', str(preview)), cfg['body_image'], cfg['preview_budget_seconds'], revision)
                    row.update(status='complete_initializer_QA_not_quality_pass', phase='complete',
                               preview=identity(preview/'report.json', readonly=False))
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                    # Independent clips continue under the same protocol. Failed clip is never repaired/adopted.
            report.update(phase='complete', status='complete_diagnostic_not_quality_pass')
    except Exception as exc:
        report['error_type'] = type(exc).__name__
    finally:
        try:
            require(source(root, code, revision, ENTRY, HELPERS) == binding, 'Immutable code changed')
            report['source_rehashed_after'] = True
        except Exception:
            report['source_rehashed_after'] = False; report['status'] = 'fail'
        report['elapsed_seconds'] = time.monotonic()-started
        write(experiment/'report.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
    print(json.dumps({k: report[k] for k in ('status', 'phase', 'producer_revision', 'elapsed_seconds', 'episodes')}))
    require(report['status'] == 'complete_diagnostic_not_quality_pass', 'Pilot failed closed')


if __name__ == '__main__': run()

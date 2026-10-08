"""Four frozen random Track1 clips; one complete isolated native pipeline.

Only reusable existing inference/fitting functions are called. No trial-specific
prompts, model edits, GT, resampling, cosmetic scene alignment or baseline writes.
This is a visual diagnostic, never an independent performance evaluation.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from world_reward.artifact_paths import episode_output, output_prefix
from full4d_pins import surface_pin, input_pin, shared_pin, identity as frozen_identity, _seal
from task_grounding_pilot import config as grounding_config, inputs, infer, track
from qwen4d_masks import _inventory

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_full4d_sample'
CONFIG = 'configs/full4d_sample_v1.json'
# Literal source paths also make the immutable deployment closure exhaustive.
STAGES = (
    ('body_smoke', 'infra/body_smoke.py', ('--inference-type', 'body'), 'body_image', 'sparse', None),
    ('depth_smoke', 'infra/depth_smoke.py', (), 'body_image', 'sparse', None),
    ('scale_smoke', 'infra/scale_smoke.py', (), 'body_image', 'sparse', None),
    ('object_grounded', 'infra/object_smoke.py', ('--aligned-pointmap',), 'object_image', 'sparse', None),
    ('surface', 'infra/object_budget_solid.py', ('--domain', 'surface'), None, 'surface', None),
    ('body_full', 'infra/body_smoke.py', ('--full-video', '--inference-type', 'body'), 'body_image', 'body_full', None),
    ('depth_full', 'infra/depth_smoke.py', ('--full-video',), 'body_image', 'depth_full', None),
    ('adapter', 'infra/cari_body_adapter_smoke.py', (), 'body_image', 'adapter', None),
    ('object_pose', 'infra/object_pose_smoke.py', ('--full-video', '--mesh-source', 'surface'), 'body_image', 'object_pose', 'object_pose_full_surface'),
    ('inputs', 'infra/cari_prepare.py', ('--mesh-source', 'surface'), 'body_image', 'inputs', None),
    ('prepare', 'infra/cari_shared_prepare.py', (), 'body_image', 'prepare', 'cari_shared_prepare_v1'),
    ('forward', 'infra/cari_full_forward.py', (), 'body_image', 'forward', 'cari_shared_forward_v1'),
    ('refined', 'infra/cari_full_refine.py', (), 'body_image', 'refined', 'cari_shared_refined_v1'),
    ('export', 'infra/cari_full_export.py', (), 'body_image', 'export', 'cari_shared_export_v1'),
    ('video', 'infra/full4d_video.py', (), 'body_image', 'video', None),
)
HELPERS = ('infra/full4d_sample.py', 'infra/run_full4d_sample.sh', CONFIG,
           'infra/full4d_pins.py', 'infra/task_grounding_pilot.py', 'infra/qwen4d_masks.py',
           'src/world_reward/artifact_paths.py', *[s[1] for s in STAGES])


def load_config(code):
    cfg = strict((code/CONFIG).read_bytes())
    sampling = cfg['sampling']
    require(cfg['schema'] == 'world_reward.full4d_sample.v1'
        and sampling == dict(population=30, seed=20261008, count=4, replacement=False)
        and cfg['episodes'] == random.Random(sampling['seed']).sample(range(sampling['population']), sampling['count'])
        and cfg['resample_failed_clips'] is False and cfg['manual_labels'] is False
        and cfg['ground_truth_used'] is False and cfg['native_refinement_steps'] == 300,
        'Frozen uniform sample/protocol required; never replace failed clips')
    expected = dict(
        scope='uniform_random_four_clip_full_pipeline_visual_diagnostic_not_heldout_evaluation',
        geometry_policy='fixed_whole_surface_identity_or_qualified_QSlim_no_volume_repair',
        grounding_config='configs/task_grounding_pilot_v1.json',
        model_folder='results/task-grounding-pilot-a9419e0332e3c6258a8022e9286eaafb04dce223/model',
        body_image='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7',
        object_image='sha256:eb389b26358c49778a14303b5875c66d887824011388ce9f8666ed7cc1841ce5',
        sam_image='sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1',
        pilot_budget_seconds=28800, virtual_floor_only=True, challenge_performance_verified=False)
    expected_budgets = dict(ground=900, track=1800, sparse=900, surface=740, body_full=1800,
        depth_full=1800, adapter=600, object_pose=7200, inputs=1800, prepare=650,
        forward=950, refined=7250, export=650, video=900)
    require(all(type(cfg.get(key)) is type(value) and cfg[key] == value for key, value in expected.items())
        and type(cfg['budgets']) is dict and cfg['budgets'] == expected_budgets
        and all(type(value) is int for value in cfg['budgets'].values())
        and type(cfg['native_refinement_steps']) is int
        and all(type(episode) is int for episode in cfg['episodes'])
        and set(cfg) == set(expected) | {'schema', 'sampling', 'episodes', 'resample_failed_clips',
            'manual_labels', 'ground_truth_used', 'native_refinement_steps', 'budgets'},
        'Frozen source images, geometry, floor and inclusive budget protocol required')
    return cfg


def seal_pose(root, episode, total, *, producer_code=None, producer_revision=None):
    """Seal one complete existing pose producer; no decoding, refit or replay.

    The optional source arguments identify an original immutable producer when
    a separate continuation applies this metadata-only publication fix. Its
    namespace, report source SHA, complete frame indices and artifact hashes
    must agree before any permission change. Historical baselines are refused.
    """
    root = canonical(root)
    require(root == ROOT and type(episode) is int and 0 <= episode < 30
            and type(total) is int and total >= 96, 'Exact Azure Track1 full timeline required')
    revision = os.environ['WR_CODE_REVISION'] if producer_revision is None else producer_revision
    code = canonical(Path(os.environ['WR_CODE']) if producer_code is None else producer_code)
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision)
            and code == root/'jobs'/revision/ENTRY/'code'
            and output_prefix() == 'experiments/full4d-v1-'+revision+'/outputs',
            'Pose sealing requires the exact original experiment producer namespace')
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes())
            and not code.stat().st_mode & 0o222, 'Readonly producer source and dispatch markers required')
    directory = episode_output(root, episode)/'object_pose_full_surface'
    require({path.name for path in directory.iterdir()} ==
            {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'},
            'Exclusive exact three-file complete object-pose inventory required')
    rows = {path.name: frozen_identity(path, readonly=False) for path in directory.iterdir()}
    report_path = directory/'report.json'
    require(rows['report.json']['bytes'] <= 64 << 20, 'Bounded full pose report required')
    report = strict(report_path.read_bytes())
    require(frozen_identity(report_path, readonly=False) == rows['report.json'], 'Pose report changed while read')
    require(type(report) is dict and report.get('status') == 'pass'
            and report.get('stage') == 'fixed_scale_full_object_pose_initializer'
            and type(report.get('episode_index')) is int and report['episode_index'] == episode
            and report.get('input_track') == 'track_1' and report.get('mesh_source') == 'surface'
            and report.get('ground_truth_used') is False and report.get('hand_labeled_test') is False
            and report.get('oracle_modes') == [] and report.get('fixed_shape') is True
            and report.get('original_frame_coverage_verified') is True
            and report.get('execution_verified') is True,
            'Explicit passing fixed-shape full-video surface pose receipt required')
    script = frozen_identity(code/'infra/object_pose_smoke.py', maximum=2 << 20)
    require(report.get('script_sha256') == script['sha256']
            and report.get('producer_revision', revision) == revision
            and report.get('geometry_and_poses_sha256') == rows['geometry_and_poses.npz']['sha256']
            and report.get('fixed_canonical_mesh_sha256') == rows['object_fixed_canonical.glb']['sha256'],
            'Original source and both numeric artifact byte identities must agree')
    frames = report.get('frames')
    require(type(frames) is list and len(frames) == total
            and all(type(row) is dict and type(row.get('frame_index')) is int for row in frames)
            and [row['frame_index'] for row in frames] == list(range(total)),
            'Full ordered original frame coverage required; no gaps or fabricated frames')
    for name, expected in rows.items():
        _seal(directory/name, episode_output(root, episode), expected)
    directory.chmod(0o555)
    require({name: frozen_identity(directory/name) for name in rows} == rows,
            'Sealing changed original pose bytes')
    return dict(stage='full4d_pose_publication_seal', status='pass', episode_index=episode,
        total_frames=total, producer_revision=revision, producer_script=script, files=rows,
        numeric_payload_decoded=False, inference_replayed=False, baseline_modified=False)


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    with path.open('wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def reserve(path, *, uid=1000):
    canonical(path); path.mkdir(mode=0o755); os.chown(path, uid, uid)


def command(code, experiment, cfg, episode, stage, script, args, image, name, revision):
    grounding = stage in {'ground', 'track'}
    uid = 0 if grounding else 1000  # Retained Qwen weights are root-readable only.
    argv = ['docker', 'run', '--rm', '--name', name, '--label', 'world_reward.full4d.owner='+revision,
        '--network', 'none', '--read-only', '--user', f'{uid}:{uid}', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--memory', '160g', '--cpus', '16',
        '--shm-size', '2g', '--tmpfs', '/tmp:rw,nosuid,size=4g', '--gpus', 'all']
    mounts = [(code.parent, True), (ROOT/'vendor', True), (ROOT/'weights', True),
              (ROOT/'results', True), (ROOT/'data/track_1/meta', True), (experiment/'pins', True)]
    if not grounding:
        # The pure surface consumer authenticates these original full source
        # closures, including their two dispatcher markers. No old predictions,
        # compiler executable, GPU job or whole jobs directory is mounted.
        for config_name, entry in (
                ('configs/surface_qslim_qualification_pins.json','run_surface_qslim_qualify'),
                ('configs/surface_identity_qualification_pins.json','run_surface_identity_qualify')):
            producer = strict((code/config_name).read_bytes())['producer_revision']
            mounts.append((ROOT/'jobs'/producer/entry, True))
    for ep in cfg['episodes']:
        mounts.append((ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4', True))
    if grounding:
        mounts.append((experiment/'grounding', False))
    elif stage == 'video':
        mounts.extend(((experiment/'outputs', True), (experiment/'videos'/f'episode_{episode:06d}', False)))
    elif stage in {'prepare', 'forward', 'refined', 'export', 'object_pose'}:
        mounts.append((experiment/'outputs', True))
        out = dict(prepare='cari_shared_prepare_v1', forward='cari_shared_forward_v1',
            refined='cari_shared_refined_v1', export='cari_shared_export_v1', object_pose='object_pose_full_surface')[stage]
        mounts.append((episode_output(ROOT, episode)/out, False))
    else:
        # New episode only; original baseline outputs are not mounted at all.
        mounts.append((episode_output(ROOT, episode), False))
    for path, readonly in mounts:
        canonical(path)
        argv += ['--mount', f'type=bind,src={path},dst={path}'+(',readonly' if readonly else '')]
    pythonpath = str(code/'src')+':'+str(code/'infra')
    if stage in {'prepare','forward','refined','export'}:
        pythonpath += ':/workspace/v2d_sam3d_body/lib'
    elif not grounding:
        pythonpath += ':/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose'
    env = ['PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp',
        'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision,
        'WR_OUTPUT_PREFIX='+os.environ['WR_OUTPUT_PREFIX'], 'WR_PIN_ROOT='+str(experiment/'pins'),
        'WR_IMAGE_ID='+image, 'PYTHONPATH='+pythonpath, 'PYTHONDONTWRITEBYTECODE=1',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'MOMENTUM_ENABLED=0', 'WANDB_MODE=disabled',
        'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'MPLBACKEND=Agg',
        'XDG_CACHE_HOME=/tmp/world-reward-cache',
        'TORCH_HOME='+str(ROOT/('weights/sam3d/torch_home' if image == cfg['object_image'] else 'weights/cari4d/sam3d_body/torch_home')),
        'HF_HOME='+str(ROOT/('weights/sam3d/hf_home' if image == cfg['object_image'] else 'weights/cari4d/hf_home'))]
    if image == cfg['object_image']: env += ['LIDRA_SKIP_INIT=1', 'CUDA_HOME=/usr/local/cuda']
    if stage == 'object_pose': env.append('WR_POSE_OUTPUT_RESERVED=1')
    if grounding:
        invocation = [str(code/'infra/full4d_sample.py'), '--native', stage]
    else:
        invocation = [str(code/script), '--episode', str(episode), *args]
        if stage == 'video': invocation += ['--output', str(experiment/'videos'/f'episode_{episode:06d}')]
    return argv+['--entrypoint', '/usr/bin/env', image, '-i', *env, '/opt/conda/bin/python', '-B', *invocation]


def native(code, experiment, cfg, episode, stage, script, args, image, seconds, revision):
    name = f'wr-full4d-{revision[-12:]}-{episode}-{stage}'
    require(not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], text=True).strip(),
        'Fresh owned worker only')
    started = time.monotonic()
    try:
        with (experiment/'logs'/f'{episode}-{stage}.log').open('xb') as log:
            result = subprocess.run(command(code, experiment, cfg, episode, stage, script, args, image, name, revision),
                stdout=log, stderr=log, timeout=seconds, check=False)
        require(result.returncode == 0, f'{stage} failed; no automatic scientific retry')
    finally:
        found = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=15, check=False)
        if found.returncode == 0:
            actual = strict(found.stdout)[0]
            require(actual['Image'] == image and actual['Config']['Labels'].get('world_reward.full4d.owner') == revision,
                'Foreign container must never be removed')
            subprocess.run(['docker', 'rm', '-f', actual['Id']], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=20, check=True)
    return time.monotonic()-started


def mask_adapter(experiment, item, code, revision):
    episode = item['episode']; src = experiment/'grounding'/f'episode_{episode:06d}'
    ground = strict((src/'grounding.json').read_bytes())
    tracked = strict((src/'tracking.json').read_bytes())
    require(ground['status'] == 'seed_available' and ground['seed'] is not None
        and ground['video_pin'] == item['video_pin'] and tracked['status'] == 'full_T_complete'
        and tracked['frames'] == item['total'], 'Automatic grounding and full-T tracking required')
    records, aggregate, raw = _inventory(src/'masks', item['total'])
    dest = episode_output(ROOT, episode)/'automatic_masks'
    require(not dest.exists(), 'Fresh automatic masks required')
    src.rename(dest)  # No mask duplication and no historical mask adoption.
    write(dest/'mask-inventory.json', raw)
    cfg = grounding_config(code)
    receipt = dict(stage='automatic_masks', status='pass', episode_index=episode, frames=item['total'],
        input_track='track_1', input_sha256=item['video_pin']['sha256'],
        input_dataset_revision='5f68335f3acc802033d1e80728c1633197521de8',
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], quality_verified=False,
        producer_revision=revision, script_sha256=identity(code/'infra/full4d_sample.py')['sha256'],
        grounding_model=cfg['model'], grounding_model_revision=cfg['model_revision'],
        sam2_checkpoint={k:cfg['sam2_checkpoint'][k] for k in ('bytes','sha256')},
        mask_inventory=aggregate, models_rerun=True, hand_modified_masks=False,
        action=item['action'], object_prompt=item['object_prompt'],
        empty_frames={k:sum(n == 0 for n in values) for k, values in tracked['areas'].items()})
    write(dest/'report.json', (json.dumps(receipt, sort_keys=True)+'\n').encode())
    for path in (dest, *dest.rglob('*')):
        os.chown(path,1000,1000); path.chmod(0o555 if path.is_dir() else 0o444)
    require(_inventory(dest/'masks',item['total'])[:2] == (records,aggregate), 'Mask bytes changed in adapter')
    return receipt


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
        'Azure VM01 host only; no local model/data processing')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    binding = source(ROOT, code, revision, ENTRY, HELPERS); cfg = load_config(code)
    experiment = ROOT/'experiments'/('full4d-v1-'+revision)
    require(not experiment.exists(), 'Fresh immutable experiment; preserve every baseline')
    experiment.parent.mkdir(exist_ok=True); reserve(experiment, uid=0)
    for name in ('outputs','pins','grounding','videos','logs'): reserve(experiment/name, uid=0)
    os.environ['WR_OUTPUT_PREFIX'] = str((experiment/'outputs').relative_to(ROOT))
    os.environ['WR_PIN_ROOT'] = str(experiment/'pins')
    started = time.monotonic()
    report = dict(schema=cfg['schema'], status='running', phase='preflight', producer_revision=revision,
        source_binding=binding, sampling=cfg['sampling'], episodes=[], quality_verified=False,
        challenge_performance_verified=False, ground_truth_used=False, hand_labeled_test=False,
        oracle_modes=[], baseline_modified=False, per_frame_alignment=False,
        training_overlap_verified=False, challenge_overlap_verified=False)
    receipt = experiment/'report.json'
    def persist():
        report['elapsed_seconds'] = time.monotonic()-started; save(receipt, report)
    def remaining(budget):
        left = cfg['pilot_budget_seconds']-(time.monotonic()-started)
        require(left > 0, 'Inclusive pilot deadline reached'); return min(budget,left)
    termination = [False]
    def interrupted(*_):
        termination[0] = True
        raise TimeoutError('Bounded sample interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker','ps','-q'],text=True).strip(), 'No duplicate jobs')
            for image in (cfg['body_image'],cfg['object_image'],cfg['sam_image']):
                require(strict(subprocess.check_output(['docker','image','inspect',image]))[0]['Id'] == image,
                    'Exact pre-existing offline image required')
            selected = inputs(cfg['episodes']); report['inputs'] = selected
            report['episodes'] = [dict(episode=item['episode'], original_frames=item['total'],
                status='pending', phase='grounding', stages=[]) for item in selected]
            persist()
            for mode, image in (('ground',cfg['body_image']),('track',cfg['sam_image'])):
                report['phase'] = mode; persist()
                native(code,experiment,cfg,-1,mode,None,(),image,remaining(cfg['budgets'][mode]),revision)
            for item,row in zip(selected,report['episodes']):
                episode = item['episode']; base = episode_output(ROOT,episode); reserve(base)
                row.update(status='running',phase='masks')
                report['phase']=f'episode_{episode}'; persist()
                try:
                    row['masks'] = mask_adapter(experiment,item,code,revision)['mask_inventory']
                    for stage, script, args, image_key, budget_key, reserved in STAGES:
                        row['phase'] = stage; persist(); seconds=remaining(cfg['budgets'][budget_key])
                        if reserved: reserve(base/reserved)
                        if stage == 'video': reserve(experiment/'videos'/f'episode_{episode:06d}')
                        if stage == 'surface':
                            with (experiment/'logs'/f'{episode}-surface.log').open('xb') as log:
                                call = subprocess.run(['/usr/bin/python3','-B',str(code/script),'--episode',str(episode),*args],
                                    stdout=log,stderr=log,timeout=seconds,check=False)
                            require(call.returncode == 0,'Whole-surface budget proposal failed; do not repair')
                            surface_pin(ROOT,code,revision,episode)
                        else:
                            elapsed = native(code,experiment,cfg,episode,stage,script,args,cfg[image_key],seconds,revision)
                            row['stages'].append(dict(stage=stage,elapsed_seconds=elapsed))
                            if stage == 'object_pose':
                                row['pose_seal'] = seal_pose(ROOT,episode,item['total'],
                                    producer_code=code,producer_revision=revision)
                            if stage == 'inputs': input_pin(ROOT,code,revision,episode,item['total'])
                            if stage in {'prepare','forward','refined','export'}:
                                shared_pin(stage,ROOT,code,revision,episode,item['total'])
                    video = experiment/'videos'/f'episode_{episode:06d}'/'report.json'
                    row.update(status='complete_full4d_visual_diagnostic_not_quality_pass',phase='complete',
                        video_report=identity(video))
                except Exception as error:
                    row.update(status='fail',error_type=type(error).__name__,error=str(error)[:400])
                    if termination[0]: raise
                    # Preserve the random denominator, never replace/relabel a failed clip.
                persist()
            report.update(status='complete_diagnostic_not_quality_pass',phase='complete')
            complete = [row['episode'] for row in report['episodes']
                        if row['status'] == 'complete_full4d_visual_diagnostic_not_quality_pass']
            if len(complete) >= 3:
                # Private remote preview publication, not submission or scoring.
                # Import here also binds this helper into the deployment closure.
                from full4d_publish import publish
                try:
                    report['private_previews'] = publish(ROOT,revision,complete)
                except Exception as error:
                    report['preview_publish_error'] = type(error).__name__
    except Exception as error:
        report.update(status='fail',error_type=type(error).__name__,error=str(error)[:400])
        for row in report['episodes']:
            if row['status'] == 'pending': row.update(status='not_run_upstream_failure',phase=report['phase'])
    finally:
        try:
            require(source(ROOT,code,revision,ENTRY,HELPERS) == binding,'Source changed'); report['source_rehashed_after']=True
        except Exception: report['source_rehashed_after']=False; report['status']='fail'
        persist()
    print(json.dumps({k:report[k] for k in ('status','phase','producer_revision','elapsed_seconds','episodes')}),flush=True)
    require(report['status'] == 'complete_diagnostic_not_quality_pass','Bounded sample failed')


def native_grounding(mode):
    require(sys.platform == 'linux' and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},'Offline native worker only')
    code=Path(os.environ['WR_CODE']); cfg=load_config(code); selected=inputs(cfg['episodes'])
    experiment=ROOT/'experiments'/('full4d-v1-'+os.environ['WR_CODE_REVISION'])
    settings=grounding_config(code)
    if mode == 'ground': infer(code,experiment/'grounding',ROOT/cfg['model_folder'],selected_inputs=selected,settings=settings)
    elif mode == 'track': track(code,experiment/'grounding',selected_inputs=selected,settings=settings)
    else: raise ValueError('Unknown native grounding mode')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--native': native_grounding(sys.argv[2])
    else:
        require(len(sys.argv) == 1,'No arbitrary replay arguments'); run()

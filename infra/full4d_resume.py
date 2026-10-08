"""Resume the unchanged frozen random pilot after a publication-only failure.

The original failed report is never edited. Existing PASS payloads are hashed,
not refitted. Only a separate continuation receipt and fresh missing stages are
written. Two episode workers share one lease; learned stages share one mutex.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
from world_reward.artifact_paths import episode_output
from full4d_pins import identity as payload_identity
from full4d_sample import seal_pose
from qwen4d_masks import _inventory

ROOT = Path('/srv/scenesmith/world-reward')
OLD = 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
ORIGINAL_REPORT = dict(bytes=8613, sha256='d20a4f074c0bf5ffeb2e489a995e66655f5a65773e445638cab46d6b197b1e40')
EXISTING_PINS = {
    'outputs/episode_000009/object_pose_full_surface/report.json': dict(bytes=21099626, sha256='296862f8fd5697807147bcee59910a25a6c57db824eb6edf7348fc6867b3207b'),
    'outputs/episode_000009/body_full/report.json': dict(bytes=270336, sha256='5fc8a2fc110ae839210e030e1f3396927ab5168acacac54f4969188dc9e8de00'),
    'outputs/episode_000009/depth_full/report.json': dict(bytes=628821, sha256='2b0a9bcafac2c8d38c1e1df039b315ac8392121838e3f4a98e542584f36a28a2'),
    'pins/surface_mesh_000001_pins.json': dict(bytes=5891, sha256='b808152da1638e43566b7a2542a6460bf4ce826cbd4b1ef8de45fa54b259f48d'),
    'pins/surface_mesh_000009_pins.json': dict(bytes=5892, sha256='4c77534383d4c5fb7fac16046bec4b26a50a7a0a286aef3cf8edba3f08b576c6'),
}
ENTRY = 'run_full4d_resume'
HELPERS = ('infra/full4d_resume.py', 'infra/run_full4d_resume.sh',
           'infra/full4d_sample.py', 'infra/full4d_pins.py', 'infra/qwen4d_masks.py',
           'src/world_reward/artifact_paths.py')
STAGE_NAMES = {
    'body_smoke': 'sam3d_body_three_frame_smoke',
    'depth_smoke': 'monocular_moge2_three_frame',
    'scale_smoke': 'predicted_human_anchored_moge2_pointmaps',
    'object_grounded': 'sam3d_objects_grounded_fixed_frame',
    'body_full': 'sam3d_body_full_video_initializer',
    'depth_full': 'monocular_moge2_full_video',
    'adapter': 'native_cari_body_adapter_full_video',
    'object_pose': 'fixed_scale_full_object_pose_initializer',
}


def load_original(code):
    """Authenticate the complete original closure before importing its driver."""
    original = ROOT/'jobs'/OLD/'run_full4d_sample'/'code'
    binding = source(ROOT, original, OLD, 'run_full4d_sample', ())
    tree = ast.parse((original/'infra/full4d_sample.py').read_bytes())
    helpers = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'HELPERS' for t in n.targets))
    # The original HELPERS includes an explicit STAGES-derived list. Its actual
    # complete closure is authenticated above; execute only after that gate.
    require(isinstance(helpers, ast.Tuple), 'Original driver helper ledger required')
    for path in original.rglob('*.py'):
        relative = path.relative_to(original)
        if relative.as_posix() == 'infra/full4d_sample.py': continue
        require((code/relative).is_file() and identity(path, 2_000_000, empty=True) ==
                identity(code/relative, 2_000_000, empty=True),
                'Continuation must carry identical original numerical helpers')
    spec = importlib.util.spec_from_file_location('world_reward_frozen_full4d_driver', original/'infra/full4d_sample.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    require(source(ROOT, original, OLD, driver.ENTRY, driver.HELPERS)['closure_sha256'] == binding['closure_sha256'],
            'Original closure changed during import')
    return driver, original, source(ROOT, original, OLD, driver.ENTRY, driver.HELPERS)


def report_pass(base, stage, script, code, item):
    """Hash existing full or sparse initializer payloads; no media/array decode."""
    directory = base/('body_full/cari_adapter' if stage == 'adapter' else
                      'object_pose_full_surface' if stage == 'object_pose' else stage)
    if not directory.exists(): return None
    path = directory/'report.json'
    require(path.is_file(), 'Occupied stage lacks a complete PASS receipt; do not overwrite')
    pin = payload_identity(path, readonly=False, maximum=64 << 20)
    report = strict(path.read_bytes())
    require(payload_identity(path, readonly=False, maximum=64 << 20) == pin, 'Existing receipt changed')
    require(report.get('status') == 'pass' and report.get('stage') == STAGE_NAMES[stage]
            and report.get('episode_index') == item['episode'] and report.get('input_track') == 'track_1'
            and report.get('ground_truth_used') is False and report.get('hand_labeled_test') is False
            and report.get('oracle_modes') == [] and report.get('script_sha256') == identity(code/script)['sha256'],
            'Only actual passing unchanged original producer may be reused')
    if stage != 'adapter':
        require(report.get('input_sha256') == item['video_pin']['sha256'], 'Wrong original video')
    expected = list(range(item['total'])) if stage.endswith('_full') or stage == 'object_pose' else [0, item['total']//2, item['total']-1]
    if stage in {'body_smoke', 'body_full', 'depth_smoke', 'depth_full', 'object_pose'}:
        require([row['frame_index'] for row in report['frames']] == expected, 'Original timeline differs')
    checks = {}
    if stage in {'body_smoke', 'body_full'}: checks['predictions.npz'] = report['predictions_sha256']
    if stage in {'body_smoke', 'body_full'}:
        require(report['mask_report_sha256'] == payload_identity(base/'automatic_masks/report.json')['sha256']
                and report['prompts_sha256'] == payload_identity(base/'automatic_masks/prompts.json')['sha256'],
                'Original body/mask source changed')
    if stage in {'depth_smoke', 'depth_full'}:
        checks.update({f"{row['frame_index']:06d}.npz": row['output_sha256'] for row in report['frames']})
    if stage == 'scale_smoke':
        require(report['frame_indices'] == expected and report['body_report_sha256'] ==
                payload_identity(base/'body_smoke/report.json', readonly=False)['sha256'] and
                report['depth_report_sha256'] == payload_identity(base/'depth_smoke/report.json', readonly=False)['sha256'],
                'Alignment initializer chain differs')
        for row in report['pointmaps']:
            for role in ('pointmap', 'intrinsics'):
                target = Path(row[role+'_path'])
                require(target.parent == directory, 'Alignment payload namespace differs')
                checks[target.name] = row[role+'_sha256']
    if stage == 'object_grounded':
        require(report['pointmap_grounding']['alignment_report_sha256'] ==
                payload_identity(base/'scale_smoke/report.json', readonly=False)['sha256'], 'Object alignment differs')
        checks.update({'object.glb': report['object_sha256'],
            'transform.json': report['transform_sha256'], 'intrinsics.json': report['intrinsics_sha256']})
    if stage == 'adapter':
        require(report['frames'] == item['total'] and report['body_report_sha256'] ==
                payload_identity(base/'body_full/report.json', readonly=False)['sha256'], 'Adapter body chain differs')
        checks['canonical_initializer.pkl'] = report['canonical_initializer_sha256']
    if stage == 'object_pose':
        for field, relative in (('object_report_sha256','object_grounded/report.json'),
                               ('alignment_report_sha256','scale_smoke/report.json'),
                               ('full_depth_report_sha256','depth_full/report.json')):
            require(report[field] == payload_identity(base/relative, readonly=False)['sha256'], 'Object pose source chain differs')
        checks.update({'geometry_and_poses.npz': report['geometry_and_poses_sha256'],
                       'object_fixed_canonical.glb': report['fixed_canonical_mesh_sha256']})
    for name, sha in checks.items():
        require(payload_identity(directory/name, readonly=False)['sha256'] == sha, 'Original numerical payload changed')
    return dict(report=pin, files=len(checks), inference_replayed=False)


def run_process(command, log, seconds, stop=None, *, env=None):
    """Bounded wait; host termination must not silently leave a GPU worker alive."""
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
    try:
        while True:
            try: return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                require(stop is None or not stop.is_set(), 'Continuation interrupted')
                require(time.monotonic()-started <= seconds, 'Stage inclusive deadline exceeded')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)


def native(driver, original, experiment, cfg, item, stage_row, continuation, seconds, owner, model_lock, stop):
    stage, script, args, image_key, _, _ = stage_row
    image = cfg[image_key]; name = f'wr-resume-{owner[-12:]}-{item["episode"]}-{stage}'
    require(not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], text=True).strip(),
            'Fresh owned continuation container required')
    command = driver.command(original, experiment, cfg, item['episode'], stage, script, args, image, name, OLD)
    lock = nullcontext() if stage == 'object_pose' else model_lock
    wait_started = time.monotonic()
    with lock:
        require(not stop.is_set(), 'Continuation interrupted before worker')
        seconds -= time.monotonic()-wait_started
        require(seconds > 0, 'Stage deadline exhausted while waiting for learned-stage mutex')
        try:
            with (continuation/'logs'/f'{item["episode"]}-{stage}.log').open('xb') as log:
                result = run_process(command, log, seconds, stop)
            require(result == 0, stage+' failed; preserve actual failure without scientific retry')
        finally:
            found = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=15, check=False)
            if found.returncode == 0:
                actual = strict(found.stdout)[0]
                require(actual['Image'] == image and actual['Config']['Labels'].get('world_reward.full4d.owner') == OLD,
                        'Foreign container may not be removed')
                subprocess.run(['docker', 'rm', '-f', actual['Id']], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20, check=True)


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
            'Azure VM01 continuation only')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    own = source(ROOT, code, revision, ENTRY, HELPERS)
    driver, original, binding = load_original(code)
    experiment = ROOT/'experiments'/('full4d-v1-'+OLD); cfg = driver.load_config(original)
    failed = experiment/'report.json'; failed_pin = payload_identity(failed, readonly=False, maximum=4 << 20)
    prior = strict(failed.read_bytes())
    require(failed_pin == ORIGINAL_REPORT and prior.get('status') == 'running' and prior.get('source_binding') == binding
            and prior.get('producer_revision') == OLD and prior.get('ground_truth_used') is False
            and prior.get('hand_labeled_test') is False and prior.get('oracle_modes') == [],
            'Exact original interrupted source-bound pilot required; do not relabel its report')
    for relative, pin in EXISTING_PINS.items():
        require(payload_identity(experiment/relative, readonly=False, maximum=64 << 20) == pin,
                'Previously audited existing producer changed')
    os.environ.update(WR_OUTPUT_PREFIX=f'experiments/full4d-v1-{OLD}/outputs', WR_PIN_ROOT=str(experiment/'pins'))
    continuation = experiment/'continuations'/revision
    continuation.parent.mkdir(exist_ok=True); driver.reserve(continuation, uid=0); driver.reserve(continuation/'logs', uid=0)
    started = time.monotonic(); write_lock = threading.RLock(); model_lock = threading.Lock(); stop = threading.Event()
    report = dict(stage='full4d_infrastructure_continuation', status='running', producer_revision=revision,
        source_binding=own, original_source=binding, original_report=failed_pin, original_revision=OLD,
        original_report_status=prior['status'], original_interrupted_report_modified=False,
        resumed_scientific_parameters_unchanged=True,
        worker_count=2, learned_stage_concurrency=1, object_pose_concurrency=2, episodes=[],
        scout_episode=9, other_workers_only_after_scout_pass=True,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], quality_verified=False)
    def persist():
        with write_lock:
            report['elapsed_seconds'] = time.monotonic()-started
            driver.save(continuation/'report.json', report)
    def seconds(stage_row):
        require(not stop.is_set(), 'Continuation interrupted')
        left = cfg['pilot_budget_seconds']-(time.monotonic()-started)
        require(left > 0, 'Continuation inclusive frozen budget exceeded')
        return min(left, cfg['budgets'][stage_row[4]])
    def interrupted(*_): stop.set()
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip(), 'No duplicate GPU controller')
            selected = driver.inputs(cfg['episodes'])
            report['episodes'] = [dict(episode=item['episode'], status='pending', stages=[]) for item in selected]
            def episode_worker(pair):
                item, row = pair; base = episode_output(ROOT, item['episode'])
                try:
                    if not base.exists(): driver.reserve(base)
                    if not (base/'automatic_masks').exists(): driver.mask_adapter(experiment, item, original, OLD)
                    mask = strict((base/'automatic_masks/report.json').read_bytes())
                    require(mask['producer_revision'] == OLD and mask['input_sha256'] == item['video_pin']['sha256']
                            and mask['frames'] == item['total'] and mask['status'] == 'pass'
                            and mask['ground_truth_used'] is False and mask['hand_labeled_test'] is False
                            and mask['oracle_modes'] == [] and mask['mask_inventory'] ==
                            _inventory(base/'automatic_masks/masks',item['total'])[1],
                            'Actual original automatic-mask provenance required')
                    for stage_row in driver.STAGES:
                        stage, script, args, _, _, reserved = stage_row
                        seconds(stage_row); row.update(status='running', phase=stage); persist()
                        reuse = report_pass(base, stage, script, original, item) if stage in STAGE_NAMES else None
                        if stage == 'surface' and (base/('object_budget_surface_'+OLD)).exists():
                            pinpath = experiment/'pins'/f'surface_mesh_{item["episode"]:06d}_pins.json'
                            pin = strict(pinpath.read_bytes()); reuse = dict(pin=payload_identity(pinpath))
                            require(pin['report']['producer_revision'] == OLD and pin['episode_index'] == item['episode']
                                    and pin['input_sha256'] == item['video_pin']['sha256'], 'Original surface producer differs')
                            for relative, expected in pin['files'].items():
                                require(payload_identity(ROOT/relative, readonly=False, maximum=256 << 20) == expected,
                                        'Existing surface ancestry changed')
                        if reuse is None:
                            if reserved: driver.reserve(base/reserved)
                            if stage == 'video': driver.reserve(experiment/'videos'/f'episode_{item["episode"]:06d}')
                            if stage == 'surface':
                                env = dict(os.environ, WR_CODE=str(original), WR_CODE_REVISION=OLD,
                                           PYTHONPATH=str(original/'src')+':'+str(original/'infra'))
                                with (continuation/'logs'/f'{item["episode"]}-surface.log').open('xb') as log:
                                    result = run_process(['/usr/bin/python3', '-B', str(original/script), '--episode',
                                        str(item['episode']), *args], log, seconds(stage_row), stop, env=env)
                                require(result == 0, 'Surface failed; no quality-tuned retry')
                                driver.surface_pin(ROOT, original, OLD, item['episode'])
                            else:
                                native(driver, original, experiment, cfg, item, stage_row, continuation,
                                       seconds(stage_row), revision, model_lock, stop)
                                if stage == 'inputs': driver.input_pin(ROOT, original, OLD, item['episode'], item['total'])
                                if stage in {'prepare', 'forward', 'refined', 'export'}:
                                    driver.shared_pin(stage, ROOT, original, OLD, item['episode'], item['total'])
                        if stage == 'object_pose':
                            row['pose_seal'] = seal_pose(ROOT, item['episode'], item['total'],
                                                       producer_code=original, producer_revision=OLD)
                        row['stages'].append(dict(stage=stage, reused=reuse is not None, evidence=reuse)); persist()
                    row.update(status='complete_full4d_visual_diagnostic_not_quality_pass', phase='complete')
                except Exception as error:
                    row.update(status='fail', error_type=type(error).__name__, error=str(error)[:400])
                persist()
            # A shared-stage wiring scout consumes the already complete first
            # clip before any new expensive object-pose work is started.
            episode_worker((selected[0], report['episodes'][0]))
            require(report['episodes'][0]['status'] == 'complete_full4d_visual_diagnostic_not_quality_pass',
                    'First frozen clip failed shared-stage scout; do not burn further GPU time')
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(episode_worker, zip(selected[1:], report['episodes'][1:])))
            complete = [row['episode'] for row in report['episodes'] if row['status'].startswith('complete_')]
            if len(complete) >= 3:
                from full4d_publish import publish
                try: report['private_previews'] = publish(ROOT, OLD, complete)
                except Exception as error: report['preview_publish_error'] = type(error).__name__
            report['status'] = 'complete_diagnostic_not_quality_pass'
    except Exception as error:
        report.update(status='fail', error_type=type(error).__name__, error=str(error)[:400])
        for row in report['episodes']:
            if row['status'] == 'pending': row['status'] = 'not_run_upstream_scout_failure'
    finally:
        require(payload_identity(failed, readonly=False, maximum=4 << 20) == failed_pin, 'Original failed report changed')
        require(source(ROOT, code, revision, ENTRY, HELPERS) == own and
                source(ROOT, original, OLD, driver.ENTRY, driver.HELPERS) == binding, 'Original/current source changed')
        report['source_rehashed_after'] = True; persist()
    print(json.dumps(dict(status=report['status'], continuation_revision=revision, original_revision=OLD,
                          episodes=report['episodes'])), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 1, 'No arbitrary resume arguments')
    run()

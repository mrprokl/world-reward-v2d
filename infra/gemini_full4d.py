"""New automatic Gemini/SAM3.1 masks, unchanged full-timeline 4D downstream.

This is a causal frontend regression diagnostic, not a jitter fix or benchmark.
RGB-only MoGe2 may be reused byte-for-byte; human inference, shape, rigid poses,
native interaction refinement and rendering are recomputed from the new masks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
from full4d_sample import ROOT, STAGES, command, reserve, save
from full4d_pins import surface_pin, input_pin, shared_pin, _seal
from task_grounding_pilot import inputs
from qwen4d_masks import _inventory
from world_reward.artifact_paths import episode_output

ENTRY = 'run_gemini_full4d'
CONFIG = 'configs/gemini_full4d_v1.json'
FRONTEND_ENTRY = 'run_gemini_sam31_track'
FRONTEND_FILES = {'report.json', 'prompts.json', 'mask-inventory.json',
                  'grounding.json', 'tracking.json'}
FRONTEND_DIAGNOSTICS = {'prefix-qualification.json', 'qa.jpg'}
HELPERS = ('infra/gemini_full4d.py', 'infra/run_gemini_full4d.sh', CONFIG,
           'infra/full4d_sample.py', 'infra/full4d_pins.py', 'infra/qwen4d_masks.py',
           'infra/object_budget_solid.py', 'infra/surface_geometry_loader.py',
           'src/world_reward/artifact_paths.py', *[stage[1] for stage in STAGES])
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
OLD = 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
DEPTH_MODEL = 'b135031bae30b5ac2ae141a0e68717795ce38340'
DEPTH_FIELDS = {'depth_smoke': ('monocular_moge2_three_frame', False),
                'depth_full': ('monocular_moge2_full_video', True)}


def config(code):
    cfg = strict((code / CONFIG).read_bytes())
    sampling = dict(population=30, seed=20261008, count=4, replacement=False)
    require(cfg.get('schema') == 'world_reward.gemini_full4d.v1'
            and cfg.get('sampling') == sampling
            and cfg.get('episodes') == random.Random(sampling['seed']).sample(range(30), 4)
            and cfg.get('resample_failed_clips') is False
            and cfg.get('ground_truth_used') is False and cfg.get('manual_labels') is False
            and cfg.get('challenge_performance_verified') is False
            and cfg.get('jitter_fix_claimed') is False
            and cfg.get('native_refinement_steps') == 300
            and cfg.get('virtual_floor_only') is True
            and cfg.get('geometry_policy') == 'fixed_whole_surface_identity_or_qualified_QSlim_no_volume_repair'
            and cfg.get('old_baseline_producer') == OLD
            and cfg.get('object_pose_method') == 'unchanged_rigid_icp_viterbi_causal_frontend_ablation',
            'Frozen random cohort and unchanged downstream diagnostic protocol required')
    require(type(cfg.get('frontend_producer_revision')) is str
            and re.fullmatch('[0-9a-f]{40}', cfg['frontend_producer_revision'])
            and cfg.get('frontend_config') == 'configs/gemini_sam31_v1.json',
            'Exact committed automatic frontend producer required')
    # Preserve every original native stage budget; no silent numerical retuning.
    old_cfg = strict((code/'configs/full4d_sample_v1.json').read_bytes())
    for name in ('body_image', 'object_image', 'budgets', 'pilot_budget_seconds'):
        require(cfg[name] == old_cfg[name], 'Unchanged native runtime/budgets required')
    return cfg


def _json(path, maximum=4 << 20):
    before = identity(path, maximum, readonly=False)
    value = strict(path.read_bytes())
    require(identity(path, maximum, readonly=False) == before, 'Metadata changed during read')
    return value, before


def _copy(src, dst, pin, maximum=64 << 20):
    """Remote-only copy, unique inode, exact payload; never edit a source receipt."""
    require(identity(src, maximum, readonly=False) == pin and not dst.exists(),
            'Exact immutable source and fresh destination required')
    with src.open('rb') as reader, dst.open('xb') as writer:
        os.fchmod(writer.fileno(), 0o444)
        shutil.copyfileobj(reader, writer, 1 << 20)
        writer.flush(); os.fsync(writer.fileno())
    os.chown(dst, 1000, 1000)
    require(identity(dst, maximum) == pin and identity(src, maximum, readonly=False) == pin,
            'Copy differs or original payload changed')


def depth_weight_identity(root, expected_sha):
    """Use the existing acquisition-bound MoGe2 HF→Xet cache resolver.

    HF file content SHA and Xet storage ID differ. The audited helper validates
    the acquisition receipt, independent content SHA/size and exact two-link
    graph. Only the original acquisition writer's canonical metadata receipt may
    be writable: it is hash/stat-checked, not claimed permission-immutable. The
    actual checkpoint and generic control paths retain their readonly contract.
    """
    from bridge_rgb_anchor_infer import moge_asset, host_moge_chain, MOGE_SHA, MOGE_REV
    require(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha),
            'Exact previously recorded MoGe2 model SHA required')
    require(expected_sha == MOGE_SHA and DEPTH_MODEL == MOGE_REV,
            'Depth receipt differs from the independently audited MoGe2 content/revision')
    acquisition_path = canonical(root/'results/weights-acquisition.json')
    acquisition_stat = acquisition_path.lstat()

    def acquisition_identity(path, maximum):
        require(canonical(path) == acquisition_path and maximum == 2_000_000,
                'Only the original canonical acquisition metadata may be writable')
        return _json(acquisition_path, maximum)[1]

    chain = host_moge_chain(root)
    path, acquisition_pin, pin = moge_asset(root, acquisition_identity=acquisition_identity)
    require(host_moge_chain(root) == chain and pin['sha256'] == expected_sha,
            'Audited MoGe2 snapshot graph/content changed during verification')
    after = acquisition_path.lstat()
    require(all(getattr(acquisition_stat, key) == getattr(after, key) for key in
                ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            'Original acquisition metadata changed during model verification')
    return dict(canonical_blob=str(path), artifact=pin, snapshot_alias=True,
                link_graph=chain, acquisition_receipt=acquisition_pin,
                acquisition_metadata_writable=bool(acquisition_stat.st_mode & 0o222),
                acquisition_permission_immutability_claimed=False,
                model_readonly_verified=True,
                resolver='bridge_rgb_anchor_infer.moge_asset_and_host_moge_chain')


def frontend_adapter(cfg, item, dest):
    revision = cfg['frontend_producer_revision']
    source_code = ROOT/'jobs'/revision/FRONTEND_ENTRY/'code'
    source_binding = source(ROOT, source_code, revision, FRONTEND_ENTRY, ())
    directory = ROOT/'results'/('gemini-sam31-'+revision)/f'episode_{item["episode"]:06d}'/'automatic_masks'
    aggregate_root = directory.parent.parent
    host, host_pin = _json(aggregate_root/'report.json')
    aggregate, aggregate_pin = _json(aggregate_root/'native-report.json')
    host_binding = host.get('source_binding', {})
    require(type(host_binding.get('helpers')) is dict, 'Bounded actual frontend helper ledger required')
    source_binding = source(ROOT, source_code, revision, FRONTEND_ENTRY, tuple(host_binding['helpers']))
    require(host.get('status') == 'complete_diagnostic_not_quality_pass'
            and host.get('producer_revision') == revision
            and host.get('source_binding') == source_binding
            and host.get('native_report') == aggregate_pin
            and host.get('ground_truth_used') is False,
            'Actual host/frontend code and independent native receipt pin required')
    episode_rows = [row for row in aggregate.get('episodes', []) if row.get('episode_index') == item['episode']]
    require(aggregate.get('schema') == 'world_reward.gemini_sam31_tracking.v1'
            and aggregate.get('status') == 'complete_diagnostic_not_quality_pass'
            and aggregate.get('producer_revision') == revision
            and aggregate.get('ground_truth_used') is False
            and aggregate.get('manual_labels') is False
            and len(episode_rows) == 1 and episode_rows[0].get('status') == 'pass'
            and episode_rows[0].get('frames') == item['total'],
            'Exact completed independently retained frontend aggregate required')
    canonical(directory)
    require({p.name for p in directory.iterdir()} == FRONTEND_FILES | FRONTEND_DIAGNOSTICS | {'masks'},
            'Exclusive automatic frontend inventory required')
    report, report_pin = _json(directory/'report.json')
    require(episode_rows[0].get('report') == report_pin,
            'Automatic frontend receipt differs from producer aggregate pin')
    expected = dict(stage='automatic_masks', status='pass', episode_index=item['episode'],
        frames=item['total'], input_track='track_1', input_sha256=item['video_pin']['sha256'],
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()),
            'Exact passing automatic full-T frontend transport receipt required')
    require(report.get('producer_revision') == revision
            and report.get('script_sha256') == identity(source_code/'infra/gemini_sam31_track.py', 2 << 20)['sha256'],
            'Actual new frontend producer revision required; no historical relabeling')
    prompts, prompts_pin = _json(directory/'prompts.json')
    prompt_rows = prompts.get('prompts')
    require(type(prompt_rows) is list and len(prompt_rows) == 2
            and {r.get('object_id') for r in prompt_rows} == {0, 1}
            and all(r.get(k) is None for r in prompt_rows for k in ('points', 'point_labels', 'mask_path')),
            'Only two automatic initial pair boxes; no human point/mask prompts')
    rows, aggregate, _ = _inventory(directory/'masks', item['total'])
    require(report.get('mask_inventory') == aggregate, 'Complete frontend PNG byte inventory differs')
    pins = {name: identity(directory/name, 4 << 20, readonly=False) for name in FRONTEND_FILES}
    require(pins['report.json'] == report_pin and pins['prompts.json'] == prompts_pin,
            'Frontend metadata changed')
    reserve(dest); reserve(dest/'masks')
    for object_id in ('0', '1'):
        reserve(dest/'masks'/object_id)
        for index in range(item['total']):
            relative = f'{object_id}/{index:06d}.png'
            _copy(directory/'masks'/relative, dest/'masks'/relative, rows[relative], 16 << 20)
        (dest/'masks'/object_id).chmod(0o555)
    for name, pin in pins.items():
        _copy(directory/name, dest/name, pin, 4 << 20)
    (dest/'masks').chmod(0o555); dest.chmod(0o555)
    require(_inventory(dest/'masks', item['total'])[:2] == (rows, aggregate)
            and _inventory(directory/'masks', item['total'])[:2] == (rows, aggregate)
            and source(ROOT, source_code, revision, FRONTEND_ENTRY, tuple(host_binding['helpers'])) == source_binding
            and _json(aggregate_root/'report.json')[1] == host_pin
            and _json(aggregate_root/'native-report.json')[1] == aggregate_pin,
            'Source/output frontend masks or immutable code changed')
    return dict(producer_revision=revision, source_directory=str(directory),
                source_binding=source_binding, files=pins, mask_inventory=aggregate,
                aggregate_report=aggregate_pin,
                host_report=host_pin,
                byte_identical_copy=True, model_calls=0, hand_modified_masks=False,
                quality_verified=False)


def depth_reuse(root, code, cfg, item, base, stage):
    """Authenticate an RGB-only prior's complete original producer before copy.

    The old receipt stays byte-identical, including its actual model/script SHA.
    A separate experiment ledger records reuse; no new inference is fabricated.
    """
    require(stage in DEPTH_FIELDS and cfg['reuse_depth_only_if_exact_provenance'] is True,
            'Explicit RGB-only depth cache contract required')
    original = root/'jobs'/OLD/'run_full4d_sample'/'code'
    original_binding = source(root, original, OLD, 'run_full4d_sample', ('infra/depth_smoke.py',))
    require(identity(original/'infra/depth_smoke.py', 2 << 20) == identity(code/'infra/depth_smoke.py', 2 << 20),
            'Numerical depth producer source changed; do not adopt cache')
    directory = root/'experiments'/('full4d-v1-'+OLD)/'outputs'/f'episode_{item["episode"]:06d}'/stage
    if not directory.exists():
        return None
    report, report_pin = _json(directory/'report.json')
    stage_name, full = DEPTH_FIELDS[stage]
    indices = list(range(item['total'])) if full else [0, item['total']//2, item['total']-1]
    expected = dict(stage=stage_name, status='pass', episode_index=item['episode'],
        input_track='track_1', input_sha256=item['video_pin']['sha256'],
        input_dataset_revision=DATASET, total_video_frames=item['total'], model_revision=DEPTH_MODEL,
        intrinsics_source='RGB_size_only_default_FOV_prior_not_calibration',
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], network='none',
        script_sha256=identity(original/'infra/depth_smoke.py', 2 << 20)['sha256'])
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items())
            and type(report.get('frames')) is list
            and [r.get('frame_index') for r in report['frames']] == indices,
            'Complete exact original RGB-only MoGe2 receipt required')
    model_identity = depth_weight_identity(root, report['model_sha256'])
    expected_names = {'report.json'} | {f'{index:06d}.npz' for index in indices}
    require({p.name for p in directory.iterdir()} == expected_names,
            'Exact full/sparse original depth inventory required')
    payloads = {'report.json': report_pin}
    for row in report['frames']:
        name = f'{row["frame_index"]:06d}.npz'
        pin = identity(directory/name, 64 << 20, readonly=False)
        require(pin['sha256'] == row['output_sha256']
                and re.fullmatch('[0-9a-f]{64}', row['decoded_rgb_sha256']),
                'Original numeric depth/RGB identity required')
        payloads[name] = pin
    dest = base/stage; reserve(dest)
    for name, pin in payloads.items():
        _copy(directory/name, dest/name, pin)
    dest.chmod(0o555)
    require(source(root, original, OLD, 'run_full4d_sample', ('infra/depth_smoke.py',)) == original_binding
            and depth_weight_identity(root, report['model_sha256']) == model_identity,
            'Original complete source closure changed during depth reuse')
    return dict(producer_revision=OLD, source_directory=str(directory), source_binding=original_binding,
                files=payloads, report=report_pin, inference_replayed=False,
                original_report_byte_preserved=True, rgb_only=True,
                model_identity=model_identity,
                source_camera_calibration_used=False, ground_truth_used=False)


def seal_pose(code, revision, item, base):
    directory = base/'object_pose_full_surface'
    names = {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'}
    require({p.name for p in directory.iterdir()} == names, 'Exact new pose output required')
    report, report_pin = _json(directory/'report.json', 64 << 20)
    require(report.get('status') == 'pass' and report.get('stage') == 'fixed_scale_full_object_pose_initializer'
            and report.get('episode_index') == item['episode'] and report.get('input_track') == 'track_1'
            and report.get('input_sha256') == item['video_pin']['sha256']
            and report.get('mesh_source') == 'surface' and report.get('fixed_shape') is True
            and report.get('ground_truth_used') is False and report.get('hand_labeled_test') is False
            and report.get('oracle_modes') == []
            and report.get('original_frame_coverage_verified') is True
            and report.get('execution_verified') is True
            and report.get('script_sha256') == identity(code/'infra/object_pose_smoke.py', 2 << 20)['sha256']
            and [row['frame_index'] for row in report['frames']] == list(range(item['total'])),
            'Exact automatic new full-T rigid pose producer required')
    payloads = {name: identity(directory/name, 2 << 30, readonly=False) for name in names}
    require(payloads['report.json'] == report_pin
            and payloads['geometry_and_poses.npz']['sha256'] == report['geometry_and_poses_sha256']
            and payloads['object_fixed_canonical.glb']['sha256'] == report['fixed_canonical_mesh_sha256'],
            'Pose numeric output byte identities disagree')
    for name, pin in payloads.items(): _seal(directory/name, base, pin)
    directory.chmod(0o555)
    return dict(producer_revision=revision, files=payloads, original_frames=item['total'])


def invoke(code, experiment, cfg, item, stage_row, seconds, revision, gpu_lock, stopped, *, deadline=None):
    stage, script, args, image_key, _, _ = stage_row
    require(not stopped.is_set(), 'Whole pilot interrupted')
    # CPU mesh compilation and input serialization may overlap a learned GPU job.
    cpu_host = stage == 'surface'
    cpu_container = stage == 'inputs'
    lock = nullcontext() if cpu_host or cpu_container else gpu_lock
    with lock:
        require(not stopped.is_set(), 'Whole pilot interrupted before native stage')
        if deadline is not None:
            seconds = min(seconds, deadline-time.monotonic())
            require(seconds > 0, 'Inclusive pilot deadline reached while waiting for GPU')
        name = f'wr-gemini4d-{revision[-12:]}-{item["episode"]}-{stage}'
        image = cfg[image_key] if image_key else None
        if cpu_host:
            argv = ['/usr/bin/python3', '-B', str(code/script), '--episode', str(item['episode']), *args]
        else:
            argv = command(code, experiment, cfg, item['episode'], stage, script, args, image, name, revision)
            if cpu_container:
                pos = argv.index('--gpus'); del argv[pos:pos+2]
        started = time.monotonic()
        try:
            with (experiment/'logs'/f'{item["episode"]}-{stage}.log').open('xb') as log:
                proc = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)
                try:
                    while True:
                        try: result = proc.wait(timeout=1); break
                        except subprocess.TimeoutExpired:
                            require(not stopped.is_set() and time.monotonic()-started < seconds,
                                    'Inclusive native stage deadline/interruption')
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
            require(result == 0, stage+' failed; no quality-selected scientific retry')
        finally:
            if not cpu_host:
                found = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=15, check=False)
                if found.returncode == 0:
                    actual = strict(found.stdout)[0]
                    require(actual['Image'] == image
                            and actual['Config']['Labels'].get('world_reward.full4d.owner') == revision,
                            'Foreign container must never be removed')
                    subprocess.run(['docker', 'rm', '-f', actual['Id']], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=20, check=True)
        return time.monotonic()-started


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
            'Azure VM01 host only; never local heavy processing')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    binding = source(ROOT, code, revision, ENTRY, HELPERS); cfg = config(code)
    experiment = ROOT/'experiments'/('full4d-v1-'+revision)
    require(not experiment.exists(), 'Fresh new revision namespace; do not overwrite any baseline')
    reserve(experiment, uid=0)
    for name in ('outputs', 'pins', 'grounding', 'videos', 'logs'): reserve(experiment/name, uid=0)
    os.environ.update(WR_OUTPUT_PREFIX=str((experiment/'outputs').relative_to(ROOT)),
                      WR_PIN_ROOT=str(experiment/'pins'))
    started = time.monotonic(); mutex = threading.RLock(); gpu_mutex = threading.Lock(); stopped = threading.Event()
    report = dict(schema=cfg['schema'], status='running', phase='preflight', producer_revision=revision,
        source_binding=binding, sampling=cfg['sampling'], frontend_producer=cfg['frontend_producer_revision'],
        episodes=[], baseline_modified=False, challenge_performance_verified=False, quality_verified=False,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], jitter_fix_claimed=False,
        per_frame_alignment=False, per_frame_scale=False, object_geometry_clip_constant=True,
        body_inference_reused=False, tracking_replayed=False, downstream_method_unchanged=True)
    def persist():
        with mutex:
            report['elapsed_seconds'] = time.monotonic()-started; save(experiment/'report.json', report)
    def remaining(limit):
        left = cfg['pilot_budget_seconds']-(time.monotonic()-started)
        require(left > 0 and not stopped.is_set(), 'Inclusive full4D pilot deadline/interruption')
        return min(left, limit)
    def update(row, **values):
        with mutex: row.update(values); persist()
    def append(row, value):
        with mutex: row['stages'].append(value); persist()
    def interrupted(*_):
        stopped.set(); raise TimeoutError('Bounded full4D pilot interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip(), 'No duplicate GPU jobs')
            for image in (cfg['body_image'], cfg['object_image']):
                require(strict(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]['Id'] == image,
                        'Actual pre-existing pinned offline image required')
            selected = inputs(cfg['episodes']); report['inputs'] = selected
            report['episodes'] = [dict(episode=item['episode'], original_frames=item['total'],
                status='pending', phase='frontend_import', stages=[]) for item in selected]
            persist()
            def worker(pair):
                item, row = pair; base = episode_output(ROOT, item['episode']); reserve(base)
                update(row, status='running', phase='frontend_import')
                try:
                    frontend = frontend_adapter(cfg, item, base/'automatic_masks')
                    update(row, frontend=frontend)
                    for stage_row in STAGES:
                        stage, script, args, _, budget_key, reserved = stage_row
                        update(row, phase=stage); cache = None
                        if stage in DEPTH_FIELDS and cfg['reuse_depth_only_if_exact_provenance']:
                            cache = depth_reuse(ROOT, code, cfg, item, base, stage)
                        if cache is not None:
                            append(row, dict(stage=stage, reused=True, evidence=cache,
                                             inference_seconds=0)); continue
                        if reserved: reserve(base/reserved)
                        if stage == 'video': reserve(experiment/'videos'/f'episode_{item["episode"]:06d}')
                        elapsed = invoke(code, experiment, cfg, item, stage_row,
                            remaining(cfg['budgets'][budget_key]), revision, gpu_mutex, stopped,
                            deadline=started+cfg['pilot_budget_seconds'])
                        if stage == 'surface': surface_pin(ROOT, code, revision, item['episode'])
                        if stage == 'object_pose': update(row, pose_seal=seal_pose(code, revision, item, base))
                        if stage == 'inputs': input_pin(ROOT, code, revision, item['episode'], item['total'])
                        if stage in {'prepare', 'forward', 'refined', 'export'}:
                            shared_pin(stage, ROOT, code, revision, item['episode'], item['total'])
                        append(row, dict(stage=stage, reused=False, elapsed_seconds=elapsed))
                    update(row, status='complete_full4d_visual_diagnostic_not_quality_pass', phase='complete',
                        video_report=identity(experiment/'videos'/f'episode_{item["episode"]:06d}'/'report.json', 4 << 20))
                except Exception as error:
                    update(row, status='fail', error_type=type(error).__name__, error=str(error)[:400])
                finally: persist()
            # Frozen first clip is the wiring scout; scientific failure is not replaced.
            worker((selected[0], report['episodes'][0]))
            require(report['episodes'][0]['status'].startswith('complete_'),
                    'First frozen full4D scout failed; do not burn remaining GPU hours')
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker, zip(selected[1:], report['episodes'][1:])))
            complete = [row['episode'] for row in report['episodes'] if row['status'].startswith('complete_')]
            if len(complete) >= 3:
                from full4d_publish import publish
                try: report['private_previews'] = publish(ROOT, revision, complete)
                except Exception as error: report['preview_publish_error'] = type(error).__name__
            report.update(status='complete_diagnostic_not_quality_pass', phase='complete')
    except Exception as error:
        report.update(status='fail', error_type=type(error).__name__, error=str(error)[:400])
        for row in report['episodes']:
            if row['status'] == 'pending': row.update(status='not_run_upstream_scout_failure', phase=report['phase'])
    finally:
        stopped.set()
        require(source(ROOT, code, revision, ENTRY, HELPERS) == binding, 'Immutable source changed')
        report['source_rehashed_after'] = True; persist()
    print(json.dumps({k: report[k] for k in ('status', 'phase', 'producer_revision', 'elapsed_seconds', 'episodes')}), flush=True)
    require(report['status'] == 'complete_diagnostic_not_quality_pass', 'Bounded full4D pilot failed')


if __name__ == '__main__':
    require(len(sys.argv) == 1, 'No arbitrary episode/quality override'); run()

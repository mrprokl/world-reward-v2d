"""Explicit full-T occlusion-aware native rerun from authenticated saved inputs.

No baseline directory is written or relabeled. Only the original failed cohort
members are repaired; the four-record denominator remains explicit. Automatic
empty masks are observations, latent SO(3)/centroid poses are initialization,
and only completed native forward/refinement/export may become a 4D result.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
from world_reward.artifact_paths import episode_output
from full4d_sample import STAGES, command, reserve, save
from full4d_pins import _seal, input_pin, shared_pin, surface_pin
from gemini_full4d import invoke
from task_grounding_pilot import inputs
from qwen4d_masks import _inventory

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_full4d_coverage'
CONFIG = 'configs/full4d_coverage_v1.json'
BASELINE = '052ba1554e9a573d566713a99a61d89a5f27681c'
BASELINE_REPORT = dict(bytes=137229, sha256='93a2b8c13fa82de1978140c275e37bfd462fe20fbc8c1dbee6e2e19985679b43')
COHORT = [9, 1, 14, 7]
HELPERS = ('infra/full4d_coverage.py', 'infra/run_full4d_coverage.sh', CONFIG,
    'configs/full4d_sample_v1.json', 'infra/mediapipe_cpu_runtime_verify.py', 'infra/cari96_inputs.py',
    'infra/full4d_sample.py', 'infra/gemini_full4d.py', 'infra/full4d_pins.py',
    'infra/cari_clip_pin_inventory.py', 'infra/run_cari_clip_pin_inventory.sh',
    'infra/cari_clip_inputs.py', 'infra/qwen4d_masks.py', 'infra/task_grounding_pilot.py',
    'infra/surface_geometry_loader.py', 'infra/surface_pose_report_capacity.py', 'src/world_reward/mask_observation.py',
    'src/world_reward/sequence_pose.py', 'src/world_reward/artifact_paths.py', *[s[1] for s in STAGES])
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--baseline-report-bytes', type=int, required=True)
    p.add_argument('--baseline-report-sha256', required=True)
    return p


def protocol(code):
    cfg = strict((code/CONFIG).read_bytes())
    require(cfg['schema'] == 'world_reward.full4d_coverage.v1' and cfg['baseline_revision'] == BASELINE
        and cfg['baseline_entry'] == 'run_gemini_full4d' and cfg['cohort'] == COHORT
        and COHORT == random.Random(cfg['random_seed']).sample(range(cfg['population']), 4)
        and cfg['random_seed'] == 20261008 and cfg['population'] == 30 and cfg['episodes'] == [1, 7]
        and cfg['selection'] == 'original_upstream_failure_repair_not_new_validation_sample'
        and cfg['reuse_stages'] == ['automatic_masks', 'body_smoke', 'depth_smoke']
        and cfg['allow_unobserved_poses'] is True and cfg['native_refinement_steps'] == 300
        and cfg['latent_policy'] == 'full_T_initializer_only_then_native_300_requested_301_effective_updates'
        and cfg['edge_policy'] == 'automatic_original_RGB_anchors_required_no_pose_extrapolation'
        and all(cfg[k] is False for k in ('resample_failed_clips', 'ground_truth_used', 'manual_labels',
            'baseline_modified', 'per_frame_alignment', 'per_frame_scale', 'production_adopted',
            'challenge_performance_verified')), 'Exact new occlusion-aware protocol; no per-record overrides')
    frozen = strict((code/'configs/full4d_sample_v1.json').read_bytes())
    require(all(cfg[k] == frozen[k] for k in ('body_image', 'object_image', 'pilot_budget_seconds'))
        and cfg['budgets'] == {k: v for k, v in frozen['budgets'].items() if k not in {'ground', 'track'}},
        'Native model/settings/stage budgets remain predeclared and unchanged')
    return cfg


def read(path, *, expected=None, maximum=4<<20):
    pin = identity(path, maximum, readonly=False)
    require(expected is None or pin == expected, 'Independently bound saved receipt differs')
    value = strict(path.read_bytes())
    require(identity(path, maximum, readonly=False) == pin, 'Saved metadata changed during read')
    return value, pin


def old_source(root, cfg, report_pin):
    code = root/'jobs'/BASELINE/cfg['baseline_entry']/'code'
    binding = source(root, code, BASELINE, cfg['baseline_entry'], ())
    experiment = root/'experiments'/('full4d-v1-'+BASELINE)
    report, pin = read(experiment/'report.json', expected=report_pin, maximum=4<<20)
    require(report.get('producer_revision') == BASELINE
        and report.get('source_binding', {}).get('closure_sha256') == binding['closure_sha256']
        and report.get('ground_truth_used') is False and report.get('hand_labeled_test') is False
        and report.get('oracle_modes') == [] and report.get('baseline_modified') is False
        and [row['episode'] for row in report['episodes']] == COHORT,
        'Actual original full4D producer and all four original records required')
    rows = {row['episode']: row for row in report['episodes']}
    require(all(rows[ep]['status'] == 'fail' and rows[ep]['phase'] == 'scale_smoke' for ep in cfg['episodes']),
        'Only original upstream failures are repaired; no success-selected reroll')
    require(all(rows[ep]['status'].startswith('complete_') for ep in COHORT if ep not in cfg['episodes']),
        'Original completed records must remain in the denominator unchanged')
    return code, experiment, binding, rows, pin


def copy_file(src, dst, pin, *, maximum=64<<20):
    require(not dst.exists() and identity(src, maximum, readonly=False) == pin, 'Fresh copy and original bytes required')
    with src.open('rb') as reader, dst.open('xb') as writer:
        os.fchmod(writer.fileno(), 0o444); shutil.copyfileobj(reader, writer, 1<<20)
        writer.flush(); os.fsync(writer.fileno())
    os.chown(dst, 1000, 1000)
    require(identity(dst, maximum) == pin and identity(src, maximum, readonly=False) == pin,
        'Copy or original changed; never repair source receipts')


def reuse(root, old_code, old_experiment, old_row, item, dest, stage):
    """Only masks and three-frame native Body/depth, with literal old receipts."""
    old = old_experiment/'outputs'/f'episode_{item["episode"]:06d}'; src = old/stage
    report, report_pin = read(src/'report.json')
    expected_stage = dict(automatic_masks='automatic_masks', body_smoke='sam3d_body_three_frame_smoke',
        depth_smoke='monocular_moge2_three_frame')[stage]
    required = dict(status='pass', stage=expected_stage, episode_index=item['episode'], input_track='track_1',
        input_sha256=item['video_pin']['sha256'], ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in required.items()),
        'Exact original automatic RGB-only stage required')
    files = {'report.json': report_pin}; indices = [0, item['total']//2, item['total']-1]
    if stage == 'automatic_masks':
        pinned = old_row['frontend']['files']
        require(pinned['report.json'] == report_pin and report['frames'] == item['total'],
            'Automatic frontend must be bound by original independent root receipt')
        for name, pin in pinned.items():
            require(name in {'report.json', 'prompts.json', 'mask-inventory.json', 'grounding.json', 'tracking.json'},
                'Only original automatic frontend metadata permitted')
            require(identity(src/name, 4<<20, readonly=False) == pin, 'Original frontend metadata changed')
            files[name] = pin
        masks, aggregate, _ = _inventory(src/'masks', item['total'])
        require(aggregate == report['mask_inventory'] == old_row['frontend']['mask_inventory'],
            'All original SAM pixels must retain their bound byte inventory')
        tracking, _ = read(src/'tracking.json', expected=files['tracking.json'])
        areas = tracking['areas']
        require(set(areas) == {'0', '1'} and all(len(areas[k]) == item['total'] for k in areas)
            and all(type(v) is int and v >= 0 for a in areas.values() for v in a)
            and len(tracking['records']) == item['total']
            and [r['frame_index'] for r in tracking['records']] == list(range(item['total'])),
            'Original full-T automatic observation ledger required')
        require(areas['1'][0] > 0 and areas['1'][-1] > 0,
            'Leading/trailing occlusion requires automatic RGB recovery anchors before native pose inference')
        require(all(v >= 20 for v in areas['0']), 'Absent actor needs current-RGB Body crop recovery, never pose copying')
        files.update({'masks/'+name: pin for name, pin in masks.items()})
    else:
        script = 'infra/'+('body_smoke.py' if stage == 'body_smoke' else 'depth_smoke.py')
        require(report['script_sha256'] == identity(old_code/script, 2<<20)['sha256']
            and report['total_video_frames'] == item['total']
            and [r['frame_index'] for r in report['frames']] == indices, 'Original source and sparse RGB indices required')
        if stage == 'body_smoke':
            require(report.get('mhr_geometry_forward_verified') is True and report.get('geometry_units') == 'metres'
                and report['mask_report_sha256'] == identity(old/'automatic_masks/report.json', 4<<20, readonly=False)['sha256']
                and report['prompts_sha256'] == identity(old/'automatic_masks/prompts.json', 4<<20, readonly=False)['sha256'],
                'Original native MHR geometry and identical automatic masks required')
            files['predictions.npz'] = identity(src/'predictions.npz', 512<<20, readonly=False)
            require(files['predictions.npz']['sha256'] == report['predictions_sha256'], 'Original native arrays changed')
            assets = root/'weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3'
            require(all(identity(assets/name, 16<<30, readonly=False) == pin for name, pin in report['body_assets'].items()),
                'Existing pretrained Body assets differ from original producer')
        else:
            for row in report['frames']:
                name = f'{row["frame_index"]:06d}.npz'; files[name] = identity(src/name, 64<<20, readonly=False)
                require(files[name]['sha256'] == row['output_sha256'], 'Original inferred depth array changed')
        for row in report['frames']:
            require(re.fullmatch('[0-9a-f]{64}', row['decoded_rgb_sha256']), 'Original RGB identity required')
        tracking, _ = read(old/'automatic_masks/tracking.json')
        require(all(row['decoded_rgb_sha256'] == tracking['records'][row['frame_index']]['decoded_rgb_sha256']
                    for row in report['frames']), 'Sparse Body/depth must match original automatic tracking RGB')
        require({p.name for p in src.iterdir()} == set(files), 'Exclusive passing stage inventory required')
    target = dest/stage; reserve(target)
    for name, pin in files.items():
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        copy_file(src/name, target/name, pin, maximum=512<<20 if name == 'predictions.npz' else 64<<20)
    for path in sorted(target.rglob('*'), reverse=True):
        if path.is_dir(): os.chown(path, 1000, 1000); path.chmod(0o555)
    target.chmod(0o555)
    return dict(original_directory=str(src), original_report=report_pin, files=files,
        original_receipt_preserved=True, byte_identical_copy=True, model_calls=0, ground_truth_used=False)


def stages():
    for row in STAGES:
        stage, script, args, image, budget, reserved = row
        if stage in {'body_smoke', 'depth_smoke'}: continue
        if stage == 'object_pose':
            args = (*args, '--allow-unobserved-poses'); reserved = 'object_pose_full_surface_latent'
        if stage == 'inputs': args = (*args, '--allow-unobserved-poses')
        yield stage, script, args, image, budget, reserved


def coverage_command(code, experiment, cfg, episode, stage, script, args, image, name, revision):
    # Reuse the audited native command, replacing only its old object output
    # mount with the explicit latent namespace; do not relabel any directory.
    argv = command(code, experiment, cfg, episode, stage, script, args, image, name, revision)
    if stage == 'object_pose':
        old = str(episode_output(ROOT, episode)/'object_pose_full_surface')
        new = old+'_latent'; mount = 'type=bind,src='+old+',dst='+old
        require(argv.count(mount) == 1, 'Exact original writable pose mount required')
        argv[argv.index(mount)] = 'type=bind,src='+new+',dst='+new
    return argv


def seal_latent(code, revision, item, base):
    directory = base/'object_pose_full_surface_latent'; names = {'report.json', 'geometry_and_poses.npz', 'object_fixed_canonical.glb'}
    require({p.name for p in directory.iterdir()} == names, 'Exclusive complete latent initializer inventory required')
    report, pin = read(directory/'report.json', maximum=64<<20)
    require(report.get('status') == 'pass' and report.get('stage') == 'fixed_scale_full_object_pose_initializer'
        and report.get('episode_index') == item['episode'] and report.get('input_sha256') == item['video_pin']['sha256']
        and report.get('allow_unobserved_poses') is True and report.get('latent_pose_initializer') is True
        and report.get('latent_poses_measured') is False and report.get('ground_truth_used') is False
        and report.get('hand_labeled_test') is False and report.get('oracle_modes') == []
        and report.get('script_sha256') == identity(code/'infra/object_pose_smoke.py', 2<<20)['sha256']
        and [r['frame_index'] for r in report['frames']] == list(range(item['total']))
        and report['pose_observed'][0] is True and report['pose_observed'][-1] is True,
        'Complete explicit full-T initializer and automatic edge anchors required')
    pins = {name: identity(directory/name, 512<<20, readonly=False) for name in names}
    require(pins['report.json'] == pin
        and pins['geometry_and_poses.npz']['sha256'] == report['geometry_and_poses_sha256']
        and pins['object_fixed_canonical.glb']['sha256'] == report['fixed_canonical_mesh_sha256'], 'All initializer payload hashes required')
    for name, value in pins.items(): _seal(directory/name, base, value)
    directory.chmod(0o555)
    return dict(files=pins, original_frames=item['total'], requires_native_refinement=True, final_prediction=False)


def run(args):
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
        'Azure VM01 control only; no local data/model work')
    require(0 < args.baseline_report_bytes <= 4<<20 and re.fullmatch('[0-9a-f]{64}', args.baseline_report_sha256),
        'Independent original baseline receipt pin required')
    code = canonical(Path(os.environ['WR_CODE'])); rev = os.environ['WR_CODE_REVISION']
    binding = source(ROOT, code, rev, ENTRY, HELPERS); cfg = protocol(code)
    old_pin = dict(bytes=args.baseline_report_bytes, sha256=args.baseline_report_sha256)
    require(old_pin == BASELINE_REPORT, 'Original independently qualified 052 receipt only; no new first-seen binding')
    old_code, old_experiment, old_binding, old_rows, _ = old_source(ROOT, cfg, old_pin)
    experiment = ROOT/'experiments'/('full4d-v1-'+rev)
    require(not experiment.exists(), 'Fresh experiment only; failed namespaces remain failed, never overwritten')
    reserve(experiment, uid=0)
    for name in ('outputs', 'pins', 'grounding', 'videos', 'logs'): reserve(experiment/name, uid=0)
    os.environ.update(WR_OUTPUT_PREFIX=str((experiment/'outputs').relative_to(ROOT)), WR_PIN_ROOT=str(experiment/'pins'))
    started = time.monotonic(); stopped = threading.Event(); mutex = threading.RLock(); gpu = threading.Lock()
    report = dict(schema=cfg['schema'], status='running', producer_revision=rev, source_binding=binding,
        baseline_revision=BASELINE, baseline_report=old_pin, baseline_source=old_binding, cohort=COHORT,
        selection=cfg['selection'], heldout_evaluation=False, episodes=[dict(episode=ep, status='pending') if ep in cfg['episodes']
            else dict(episode=ep, status='original_complete_not_rerun') for ep in COHORT],
        baseline_modified=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        per_frame_alignment=False, per_frame_scale=False, production_adopted=False, challenge_performance_verified=False)
    def persist():
        with mutex: report['elapsed_seconds'] = time.monotonic()-started; save(experiment/'report.json', report)
    def stop(*_): stopped.set(); raise TimeoutError('Bounded coverage experiment interrupted')
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip(), 'One owned GPU lane; no duplicate jobs')
            for image in (cfg['body_image'], cfg['object_image']):
                require(strict(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]['Id'] == image,
                    'Exact existing offline pretrained image required')
            selected = inputs(cfg['episodes']); report['inputs'] = selected; persist()
            def worker(item):
                row = next(r for r in report['episodes'] if r['episode'] == item['episode'])
                base = episode_output(ROOT, item['episode']); reserve(base); row.update(status='running', stages=[])
                try:
                    for stage in cfg['reuse_stages']:
                        row['phase'] = 'reuse_'+stage; persist()
                        evidence = reuse(ROOT, old_code, old_experiment, old_rows[item['episode']], item, base, stage)
                        row['stages'].append(dict(stage=stage, reused=True, evidence=evidence)); persist()
                    for stage_row in stages():
                        stage, script, argv, _, budget, reserved = stage_row
                        row['phase'] = stage; persist()
                        if reserved: reserve(base/reserved)
                        if stage == 'video': reserve(experiment/'videos'/f'episode_{item["episode"]:06d}')
                        remaining = min(cfg['budgets'][budget], started+cfg['pilot_budget_seconds']-time.monotonic())
                        require(remaining > 0 and not stopped.is_set(), 'Inclusive coverage deadline reached')
                        # invoke's immutable branch globals are copied, not
                        # mutated; native scheduling/cleanup remain the same.
                        call = types.FunctionType(invoke.__code__, dict(invoke.__globals__, command=coverage_command),
                            invoke.__name__, invoke.__defaults__, invoke.__closure__)
                        elapsed = call(code, experiment, cfg, item, stage_row, remaining, rev, gpu, stopped,
                            deadline=started+cfg['pilot_budget_seconds'])
                        if stage == 'surface': surface_pin(ROOT, code, rev, item['episode'])
                        if stage == 'object_pose': row['latent_initializer'] = seal_latent(code, rev, item, base)
                        if stage == 'inputs': input_pin(ROOT, code, rev, item['episode'], item['total'], allow_unobserved_poses=True)
                        if stage in {'prepare', 'forward', 'refined', 'export'}:
                            shared_pin(stage, ROOT, code, rev, item['episode'], item['total'])
                        row['stages'].append(dict(stage=stage, reused=False, elapsed_seconds=elapsed)); persist()
                    row.update(status='complete_native_full4d_not_quality_validated', phase='complete',
                        native_refinement_completed=True, latent_initialization_is_not_final_prediction=True,
                        object_pose_observations=read(base/'cari_inputs/report.json')[0]['object_pose_observations'],
                        export_pins=identity(experiment/'pins'/f'cari_clip_{item["episode"]:06d}_shared_export_pins.json', 4<<20),
                        video_report=identity(experiment/'videos'/f'episode_{item["episode"]:06d}'/'report.json', 4<<20))
                except Exception as error:
                    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)): stopped.set()
                    row.update(status='fail_no_fabricated_prediction', error_type=type(error).__name__, error=str(error)[:300])
                finally: persist()
            with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(worker, selected))
            report.update(status='complete_diagnostic_not_quality_pass', full_cohort_denominator=4,
                repaired_count=sum(r['status'].startswith('complete_native_') for r in report['episodes']))
    except Exception as error: report.update(status='fail', error_type=type(error).__name__, error=str(error)[:300])
    finally:
        stopped.set()
        require(source(ROOT, code, rev, ENTRY, HELPERS) == binding
            and source(ROOT, old_code, BASELINE, cfg['baseline_entry'], ()) == old_binding
            and identity(old_experiment/'report.json', 4<<20, readonly=False) == old_pin,
            'Current/original source and baseline receipt must remain unchanged')
        report['source_rehashed_after'] = True; persist()
    print(json.dumps(dict(status=report['status'], producer_revision=rev,
        repaired_count=report.get('repaired_count', 0), cohort=COHORT)), flush=True)
    require(report['status'] == 'complete_diagnostic_not_quality_pass' and report['repaired_count'] == len(cfg['episodes']),
        'Coverage is incomplete; retain real failure, never substitute interpolation predictions')


if __name__ == '__main__': run(parser().parse_args())

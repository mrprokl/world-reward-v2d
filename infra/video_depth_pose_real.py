"""Full-T VDA -> one predicted-human gauge -> fixed-geometry pose/contact QA.

Allowed Track 1 RGB and automatic predicted geometry only. External sensor
approval does not establish this clip's accuracy; all reported QA remains proxy.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

import sequence_contact_patch_real as real
import video_depth_assets as assets
import video_depth_dependencies as dependencies
from camera_render import raster_camera_mesh
from mediapipe_cpu_runtime_verify import source, strict
from world_reward.contact_patch import ContactPatchConfig
from world_reward.metric_alignment import fit_shared_depth_scale
from world_reward.sequence_pose import SequenceContactConfig, refine_sequence
from world_reward.video_depth import load_metric_small, infer_metric_video

ROOT = real.ROOT
ENTRY = 'run_video_depth_pose_real'
CONFIG = 'configs/video_depth_pose_real_v1.json'
VARIANTS = ('VDA_RGBD', 'VDA_RGBD_soft_pool')
ANCHORS = (0, 207, 414)
HELPERS = tuple(dict.fromkeys(('infra/video_depth_pose_real.py', 'infra/run_video_depth_pose_real.sh', CONFIG,
    'src/world_reward/video_depth.py', 'src/world_reward/metric_alignment.py',
    'infra/camera_render.py', 'infra/video_depth_assets.py',
    'infra/video_depth_dependencies.py', *real.HELPERS)))


def settings(code):
    cfg = strict((code / CONFIG).read_bytes())
    expected = dict(schema='world_reward.video_depth_pose_real.v1', episode=9,
        frames=415, fps=30., variants=list(VARIANTS), anchor_frames=list(ANCHORS),
        max_nfev=300, max_candidates=8, contact_sigma_diameter=.02,
        temperature_diameter=.01, max_point_triangle_pairs=1_000_000_000,
        depth_sigma_diameter=real.saved.DEPTH_CFG.depth_sigma_diameter,
        development_reference='sequence_evidence_stress_v1_owned_DEV_frozen_no_challenge_tuning',
        production_adopted=False)
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in expected.items()):
        raise ValueError('Exact frozen VDA full-T pose protocol required')
    approval = cfg.get('external_validation_approval')
    if approval != dict(passed=True, report_pin=dict(bytes=4504,
            sha256='e7c16df2d488115539f4f3dd2455dcbf9c21fb9e23dca1a2dedb0fa122f2e3af'),
            producer_revision='9bd3c8c3b92168ff58c8c87399248f9f3e8549ab',
            scope='TUM96frames3scenes_depth_only_not4D',
            source_path='/srv/scenesmith/world-reward/validation/video_depth_real_output_v2/evaluation/report.json'):
        raise ValueError('Actual independent sealed sensor-approval identity required')
    gates = cfg.get('gates', {})
    expected_gates = dict(compare_against=['original'], reprojection_ratio_maximum=1.00000001,
        contact_gap_ratio_maximum=1.00000001, acceleration_tail_ratio_maximum=1.00000001,
        RGB_motion_increment_ratio_maximum=1.00000001, convergence_required=True)
    if any(gates.get(k) != v for k, v in expected_gates.items()):
        raise ValueError('Frozen conservative original-trajectory QA gates required')
    return cfg


def asset_bindings(ledger, code):
    """Every native source/dependency byte is authenticated before imports."""
    cfg = assets.configuration(code / assets.CONFIG)
    receipt = strict(ledger.read(ROOT / assets.REPORT))
    expected = dict(status='pass', source_revision=cfg['source_revision'],
        model_revision=cfg['model_revision'], license=cfg['model_license'],
        source_modified=False, inference_performed=False, challenge_inputs_used=False)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Original VDA Small metric asset receipt required')
    native_source = ROOT / assets.SOURCE_DIR
    for row in cfg['source_files']:
        ledger.record(native_source / row['file'], {k: row[k] for k in ('bytes', 'sha256')})
    checkpoint = ROOT / assets.WEIGHTS_DIR / cfg['model_file']
    ledger.record(checkpoint, dict(bytes=cfg['model_bytes'], sha256=cfg['model_sha256']))
    dep = strict(ledger.read(ROOT / dependencies.RECEIPT))
    if dep.get('status') != 'pass' or dep.get('wheel_sha256') != dependencies.WHEEL_SHA256:
        raise ValueError('Pinned unchanged EasyDict dependency required')
    for row in dep['files']:
        ledger.record(ROOT / dependencies.DIRECTORY / row['file'],
            {k: row[k] for k in ('bytes', 'sha256')})
    return native_source, checkpoint


def original_sources(ledger, bank):
    old = real.old
    front, rows = old.saved_frontend(ledger)
    experiment = ROOT / 'experiments' / ('full4d-v1-' + old.SOURCE)
    transport = old.saved_numerical_frontend(ledger, experiment, rows)[9]
    base = experiment / 'outputs/episode_000009'
    pins = strict(ledger.read(experiment / 'pins/cari_clip_000009_shared_export_pins.json'))['export_files']
    directory = base / 'cari_shared_export_v1'
    export = strict(ledger.read(directory / 'report.json', pins['report.json']))
    native = real.saved.load_npz(ledger, directory / 'native_parameters.npz', real.contact.NATIVE_PIN)
    ledger.record(directory / 'target.npy', pins['target.npy'])
    human = np.load(directory / 'target.npy', mmap_mode='r', allow_pickle=False)
    inventory, mask_report = old.authenticated_masks(ledger, front, rows[9], base, 9, 415)
    if transport['mask_inventory'] != mask_report['mask_inventory'] or human.shape != (415, 18439, 3):
        raise ValueError('Original shared predicted human and SAM full-T ancestry required')
    video = ROOT / 'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4'
    ledger.record(video, transport['video_pin'])
    if transport['video_pin']['sha256'] != mask_report['input_sha256']:
        raise ValueError('RGB must match the automatic masks and original shared geometry')
    spec = export['clip_spec']
    return dict(base=base, human=human, human_faces=native['human_faces'],
        inventory=inventory, video=video, video_pin=transport['video_pin'],
        native_grid=(spec['height'], spec['width']))


def decode_video(cv2, path, native_grid, frame_count=415, fps=30.):
    """Same deterministic RGB grid for every original frame, no optical interpolation."""
    capture = cv2.VideoCapture(str(path)); frames = []; digests = []
    try:
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count or real.old.actual_fps(capture) != fps:
            raise ValueError('Exact original video frame count/FPS required')
        for index in range(frame_count):
            ok, bgr = capture.read()
            if (not ok or bgr.shape != (*native_grid, 3) or bgr.dtype != np.uint8
                    or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1):
                raise ValueError('Full original decoder chronology/grid required')
            small = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(rgb); digests.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        if capture.read()[0]:
            raise ValueError('Unexpected extra source frames')
        real.old.actual_fps(capture, fps)
    finally:
        capture.release()
    return np.stack(frames), digests


def original_masks(ledger, sources, frame, cv2):
    from PIL import Image
    masks = []
    for role in (0, 1):
        name = f'{role}/{frame:06d}.png'; path = sources['base'] / 'automatic_masks/masks' / name
        ledger.record(path, sources['inventory'][name])
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
            if image.mode != 'L' or pixels.shape != sources['native_grid'] or not np.isin(pixels, [0, 255]).all():
                raise ValueError('Literal original SAM binary grid required')
        masks.append(cv2.resize(pixels, (640, 480), interpolation=cv2.INTER_NEAREST) > 0)
    return masks


def renderer_camera(track_camera):
    """Integer-centred track coordinates -> identical camera's cell-centred raster."""
    matrix = np.asarray(track_camera).copy(); matrix[:2, 2] += .5
    return matrix


def shared_human_scale(depths, bank, sources, masks, raster=raster_camera_mesh):
    rendered = []; visible = []
    for frame in ANCHORS:
        mask, z = raster(np.asarray(sources['human'][frame]), sources['human_faces'],
            renderer_camera(bank['K']), 640, 480)
        mask, z = mask.cpu().numpy(), z.cpu().numpy()
        human_mask, object_mask = masks(frame)
        rendered.append(z)
        visible.append(mask & human_mask & ~object_mask & np.isfinite(depths[frame]) & (depths[frame] > 0))
    aligned = fit_shared_depth_scale([depths[i] for i in ANCHORS], rendered, visible, ANCHORS)
    return aligned


def sampled_depth(depths, bank, masks, scale):
    measured = np.full(bank['visible'].shape, np.nan); supported = np.zeros(bank['visible'].shape, bool)
    for frame in range(len(bank['frame_index'])):
        human_mask, object_mask = masks(frame)
        valid = np.isfinite(depths[frame]) & (depths[frame] > 0)
        measured[frame], supported[frame] = real.saved.bilinear_depth(depths[frame], valid,
            object_mask & ~human_mask, bank['xy'][frame], bank['visible'][frame], np.ones(2), scale)
    return measured, supported


def quality_decision(cfg, metrics, converged, name):
    """Predeclared nonworsening evidence gates, never a ground-truth victory."""
    gates = {'both_candidate_fits_converged': all(converged.get(variant, False) for variant in VARIANTS)}
    comparisons = {'RGB_mean_px': 'reprojection_ratio_maximum',
        'selected_anatomical_triangle_mean_m': 'contact_gap_ratio_maximum',
        'same_J1_surface_gap_p95_m': 'contact_gap_ratio_maximum',
        'acceleration_proxy_m_s2_p95': 'acceleration_tail_ratio_maximum',
        'angular_acceleration_proxy_rad_s2_p95': 'acceleration_tail_ratio_maximum',
        'RGB_motion_increment_error_mean_px': 'RGB_motion_increment_ratio_maximum'}
    for metric, key in comparisons.items():
        candidate = metrics.get(name, {}).get(metric); original = metrics.get('original', {}).get(metric)
        gates['original_' + metric + '_nonworse'] = bool(candidate is not None and original is not None
            and np.isfinite(candidate) and np.isfinite(original)
            and candidate <= original * cfg['gates'][key] + 1e-12)
    return dict(variant=name, passed=all(gates.values()), gates=gates,
        decision='retain_for_visual_QA' if all(gates.values()) else 'not_qualified_for_production',
        scope='same_frozen_automatic_observation_QA_not_ground_truth_accuracy', production_adopted=False)


def fit_worker(job):
    """Identical fixed RGBD inputs/budget; only the contact factor differs."""
    name, bank, depth, support, pool, ids, cfg, fit_cfg, out, revision = job
    started = time.monotonic(); row = dict(name=name, status='fail')
    try:
        extra = {}
        if name == VARIANTS[1]:
            extra = dict(contact_evidence=pool, contact_config=SequenceContactConfig(
                cfg['contact_sigma_diameter'], cfg['max_candidates'], cfg['max_point_triangle_pairs'],
                cfg['development_reference']), contact_patch_config=ContactPatchConfig(
                cfg['temperature_diameter'], cfg['max_candidates'], cfg['development_reference']),
                contact_distance_batch_size=32)
        elif name != VARIANTS[0]:
            raise ValueError('Only frozen VDA pose variants permitted')
        fit = refine_sequence(bank['vertices'], bank['points'], bank['xy'], bank['visible'],
            bank['rotations'], bank['translations'], np.ones(len(bank['frame_index']), bool),
            bank['K'], bank['frame_index'], bank['fps'], fit_cfg, tracks_depth_m=depth,
            depth_visible=support, depth_config=real.saved.DEPTH_CFG, **extra)
        arrays = real.output_arrays(bank, fit.rotations, fit.translations, pool.activations, ids)
        arrays.update(tracks_depth_m=depth, depth_visible=support)
        path = Path(out) / ('episode_000009_' + name + '.npz')
        result_pin = real.seal_file(path, lambda stream: np.savez_compressed(stream, **arrays))
        row.update(status='complete', fit=fit.diagnostics, file=path.name, output=result_pin,
            seconds=time.monotonic() - started, producer_revision=revision, production_adopted=False)
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc)[:300], seconds=time.monotonic()-started)
    row['receipt_file'] = 'episode_000009_' + name + '_fit.json'
    row['receipt'] = real.seal_json(Path(out) / row['receipt_file'], dict(row))
    return row


def run():
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if Path(os.environ['WR_ROOT']) != ROOT or code != ROOT / 'jobs' / revision / ENTRY / 'code':
        raise ValueError('Immutable Azure full-T VDA pose namespace required')
    out = ROOT / 'results' / ('video-depth-pose-real-' + revision)
    real.old.fresh_runtime_output(out); ledger = real.old.ArtifactLedger()
    report = dict(status='fail', producer_revision=revision, ground_truth_used=False, manual_labels=False,
        production_adopted=False, full_4D_export_replaced=False, heldout_4D_accuracy_verified=False)
    try:
        binding = source(ROOT, code, revision, ENTRY, HELPERS); cfg = settings(code)
        report.update(source_binding=binding, protocol=cfg)
        native_source, checkpoint = asset_bindings(ledger, code)
        bank = real.load_bank(ledger); sources = original_sources(ledger, bank)
        import cv2
        rgb, digests = decode_video(cv2, sources['video'], sources['native_grid'])
        network = load_metric_small(native_source, checkpoint)
        inference_started = time.monotonic(); depths = infer_metric_video(network, rgb, bank['fps'])
        report['VDA_inference_seconds'] = time.monotonic()-inference_started
        del network, rgb
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        masks = lambda frame: original_masks(ledger, sources, frame, cv2)
        aligned = shared_human_scale(depths, bank, sources, masks)
        depth, support = sampled_depth(depths, bank, masks, aligned.shared_scale)
        del depths; gc.collect()
        report.update(depth_alignment=aligned.to_dict(), depth_observations=int(support.sum()),
            frames=415, points=len(bank['points']), decoded_RGB_sha256=digests,
            gauge_source='same_full4D_predicted_human_not_initial_body_or_GT',
            native_RGB_resize='cv2_INTER_AREA_to640x480_all_original_frames',
            masks_resize='nearest_binary_observation_coordinates_not_relabeling',
            depth_config=asdict(real.saved.DEPTH_CFG))
        report['depth_pin'] = real.seal_file(out / 'depth_observations.npz', lambda stream:
            np.savez_compressed(stream, tracks_depth_m=depth, depth_visible=support,
                frame_index=bank['frame_index'], shared_scale=np.array(aligned.shared_scale)))
        report['scale_pin'] = real.seal_json(out / 'depth_scale.json', aligned.to_dict())
        pool_cfg = real.settings(code); pool, ids = real.bounded_pool(bank, pool_cfg)
        fit_cfg, _ = real.old.profile_config(code, 'v2'); fit_cfg = replace(fit_cfg, max_nfev=300)
        report['fit_config'] = asdict(fit_cfg); report['metrics'] = {}; report['fits'] = []; converged = {}
        fit_bank = {k: v for k, v in bank.items() if k not in
            ('hands', 'ids', 'prior_j1_rotations', 'prior_j1_translations', 'contact_logits')}
        jobs = [(name, fit_bank, depth, support, pool, ids, cfg, fit_cfg, str(out), revision) for name in VARIANTS]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(fit_worker, job) for job in jobs]
            report['metrics']['original'] = real.measure(bank, bank['rotations'], bank['translations'], pool)
            for future in as_completed(futures):
                row = future.result(); report['fits'].append(row)
                print(__import__('json').dumps(dict(stage='VDA_pose_fit_complete', name=row['name'],
                    status=row['status'], seconds=row['seconds'], sealed=row['status']=='complete')), flush=True)
                if row['status'] == 'complete':
                    receipt = strict(ledger.read(out / row['receipt_file'], row['receipt']))
                    if receipt['output'] != row['output'] or receipt['fit'] != row['fit']:
                        raise ValueError('Sealed worker receipt differs')
                    fitted = real.saved.load_npz(ledger, out / row['file'], row['output']); real.validate_output(bank, fitted)
                    metrics = real.measure(bank, fitted['rotation'], fitted['translation'], pool)
                    report['metrics'][row['name']] = metrics
                    converged[row['name']] = row['fit']['converged']
                    real.seal_json(out / ('episode_000009_' + row['name'] + '_qa.json'), metrics)
        report['fits'].sort(key=lambda row: VARIANTS.index(row['name']))
        if any(row['status'] != 'complete' for row in report['fits']) or len(report['fits']) != 2:
            raise ValueError('Both original full-T variants must complete; no partial success claim')
        ledger.verify()
        if source(ROOT, code, revision, ENTRY, HELPERS) != binding:
            raise ValueError('Immutable source changed')
        report['decisions'] = [quality_decision(cfg, report['metrics'], converged, name) for name in VARIANTS]
        report.update(status='complete_full_T_VDA_pose_ablation_not_truth_accuracy', sources=ledger.records,
            quality_scope='same_RGB_anatomy_full_motion_proxy_not_ground_truth_or_leaderboard')
    except Exception as exc:
        report.update(status='fail', error_type=type(exc).__name__, error=str(exc)[:400])
    report['elapsed_seconds'] = time.monotonic()-started
    real.seal_json(out / 'report.json', report)
    print(__import__('json').dumps({k: report.get(k) for k in ('status', 'error', 'elapsed_seconds', 'metrics')}))
    if report['status'] == 'fail':
        raise SystemExit(1)


if __name__ == '__main__':
    if len(os.sys.argv) != 1:
        raise SystemExit('No arbitrary test prompts/config overrides allowed')
    run()

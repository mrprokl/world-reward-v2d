"""Azure-only real full-T native joint fit; baseline is frozen and never changed.

A saved native control and B same-native image/translation extension. All fitted
outputs are sealed before automatic QA. Published-coefficient pilot and tiny
factor DEV are explicitly NOT held-out HOI validation or adoption clearance.
"""
from dataclasses import asdict
import argparse
import gc
import hashlib
from importlib import util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import sequence_contact_patch_real as real
import cari_full_refine as full
import cari_full_export as export
import dwpose_smoke as dw
from mediapipe_cpu_runtime_verify import source, strict
from world_reward.native_joint_refinement import (JointImageEvidence, NativeJointConfig,
    native_joint_optimizer_class, reprojection_statistics)
from world_reward.root_refit import COCO_TO_MHR, TRAIN_COCO, HELDOUT_COCO

ROOT = real.ROOT
ENTRY = 'run_native_joint_real'
CONFIG = 'configs/native_joint_real_v1.json'
IMAGE = full.IMAGE
BUDGET = 1800
HELPERS = tuple(dict.fromkeys(('infra/native_joint_real.py', 'infra/run_native_joint_real.sh', CONFIG,
    'src/world_reward/native_joint_refinement.py', 'src/world_reward/root_refit.py',
    'src/world_reward/joint_point_objective.py', 'infra/cari_full_refine.py',
    'infra/cari_full_export.py', 'infra/cari_refine.py', 'infra/cari_converter.py',
    'infra/cari_clip_inputs.py', 'infra/body_smoke.py', 'infra/cari_shared_prepare.py',
    'infra/cari96_prepare.py', 'infra/dwpose_smoke.py', 'infra/dwpose_acquire.py',
    'infra/dwpose_wheel_audit.py', *real.HELPERS)))


def settings(code):
    c = strict((code / CONFIG).read_bytes())
    if (c.get('schema') != 'world_reward.native_joint_real.v1'
            or c.get('cohort') != dict(random_seed=20261008, population=30,
                episodes=random.Random(20261008).sample(range(30), 4))
            or c.get('supported_saved_material_bank_episodes') != [9]
            or c.get('fallback_native_silhouette_contact_no_material_tracks_episodes') != [14]
            or c.get('budget_seconds') != BUDGET or c.get('num_steps') != 300 or c.get('batch_size') != 0
            or c.get('optimizer_sha256') != full.contract.OPTIMIZER_SHA256
            or any(c.get(k) is not v for k, v in dict(production_adopted=False,
                private_truth_read=False, depth_used=False, freeze_object_rotation=True,
                freeze_human_root_rotation=True, freeze_hand_pose=True).items())):
        raise ValueError('Frozen random cohort/native unchanged-geometry pilot protocol required')
    e = NativeJointConfig(**c['extension'])
    if asdict(e) != asdict(NativeJointConfig(5., 1., 1., 10., .1, 1e-4,
            'published_native10_0.1_lr1e-4_and_root_RGB5px_proposal_frozen_tiny_factor_DEV_not_HOI_calibration')):
        raise ValueError('Published coefficient proposal changed; no challenge tuning')
    return c, e


def controlled_factor_DEV():
    """Known moving geometric factor sanity before challenge input access."""
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
    points = np.c_[np.linspace(-.1, .1, 33), np.zeros(33), np.zeros(33)]
    xyz = np.broadcast_to(points, (12, 33, 3)).copy(); xyz[..., 2] += 2
    xyz[..., 0] += np.arange(12)[:, None] * .015
    xy = xyz[..., :2] / xyz[..., 2:] * [800, 800] + [320, 240]
    support = np.ones((12, 33), bool); support[5] = False; xy[5] = np.nan
    displaced = xyz.copy(); displaced[..., 2] += .2
    a = reprojection_statistics(displaced, K, xy, support, 5., omit_initializer=True)
    b = reprojection_statistics(xyz, K, xy, support, 5., omit_initializer=True)
    if not a['loss'] > b['loss'] == 0 or b['observations'] != 330:
        raise ValueError('Known moving factor DEV failed; do not access challenge')
    return dict(passed=True, before=a, after=b, native_optimizer_tested=False,
        geometry_known_authored=True, inference_derived_observations=False,
        scope='factor_sanity_not_calibration_or_external_HOI_validation')


def original_sources(ledger, episode):
    if episode == 1:
        raise ValueError('Frozen random episode1 has unsupported full native pose after occlusion; retain record, no reroll')
    if episode not in (9, 14): raise ValueError('Declared original random episode required')
    bank = real.load_bank(ledger) if episode == 9 else None
    front, rows = real.old.saved_frontend(ledger)
    experiment = ROOT / 'experiments' / ('full4d-v1-' + real.old.SOURCE)
    base = experiment / 'outputs' / f'episode_{episode:06d}'
    transport = real.old.saved_numerical_frontend(ledger, experiment, rows)[episode]
    all_pins = strict(ledger.read(experiment / 'pins' / f'cari_clip_{episode:06d}_shared_export_pins.json'))
    pins = all_pins['export_files']; directory = base / 'cari_shared_export_v1'
    report = strict(ledger.read(directory / 'report.json', pins['report.json']))
    src_count = report['frames']
    inventory, mask_report = real.old.authenticated_masks(ledger, front, rows[episode], base, episode, src_count)
    if transport['mask_inventory'] != mask_report['mask_inventory']:
        raise ValueError('Original actor mask lineage differs')
    forward_pins = strict(ledger.read(experiment / 'pins' / f'cari_clip_{episode:06d}_shared_forward_pins.json',
        real.contact.FORWARD_PINS_PIN if episode == 9 else None))
    forward_pin = forward_pins['forward_files']['coconet.pth']
    forward_report = strict(ledger.read(base / 'cari_shared_forward_v1/report.json',
        forward_pins['forward_files']['report.json']))
    if episode == 9 and (forward_pin != real.contact.FORWARD_PIN
            or forward_pins['forward_files']['report.json'] != real.contact.FORWARD_REPORT_PIN):
        raise ValueError('Independent episode9 raw forward pins differ')
    refined_pins = strict(ledger.read(experiment / 'pins' / f'cari_clip_{episode:06d}_shared_refined_pins.json'))
    refined_pin = refined_pins['refined_files']['report.json']
    if refined_pin['sha256'] != report['refined_report_sha256']:
        raise ValueError('Baseline export must independently bind original native refinement receipt')
    original_refined = strict(ledger.read(base / 'cari_shared_refined_v1/report.json', refined_pin))
    if (original_refined['forward_report_sha256'] != forward_pins['forward_files']['report.json']['sha256']
            or original_refined['source_bundle_sha256'] != forward_pin['sha256']
            or original_refined['bundle_sha256'] != report['refined_bundle_sha256']
            or original_refined['optimizer_sha256'] != full.contract.OPTIMIZER_SHA256):
        raise ValueError('Baseline refinement/raw forward/export ancestry differs')
    expected = dict(status='pass', producer_revision=real.old.SOURCE, frames=src_count,
        input_track='track_1', ground_truth_used=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], input_sha256=mask_report['input_sha256'])
    if any(forward_report.get(k) != v for k, v in expected.items()):
        raise ValueError('Full-T raw native forward no-oracle ancestry required')
    raw = base / 'cari_shared_forward_v1/coconet.pth'; ledger.record(raw, forward_pin)
    native = real.saved.load_npz(ledger, directory / 'native_parameters.npz', pins['native_parameters.npz'])
    ledger.record(directory / 'target.npy', pins['target.npy'])
    human = np.load(directory / 'target.npy', mmap_mode='r', allow_pickle=False)
    video = ROOT / 'data/track_1/videos/chunk-000/observation.images.exo_camera' / f'episode_{episode:06d}.mp4'
    ledger.record(video, transport['video_pin'])
    if report['status'] != 'pass' or report['ground_truth_used'] is not False or report['oracle_modes'] != []:
        raise ValueError('Original complete baseline export required')
    spec = full.inputs.PublicClipSpec(**report['clip_spec'])
    if (spec.episode_index != episode or spec.total_frames != src_count
            or human.shape != (src_count, 18439, 3) or human.dtype != np.float32
            or report['original_frame_indices'] != list(range(src_count))):
        raise ValueError('Full original fixed-shape human/timeline required')
    relative = Path(full.inputs.relative_paths(spec)['mesh']).relative_to(f'outputs/episode_{episode:06d}')
    mesh = base / relative
    ledger.record(mesh, report['source_files'][str(mesh.relative_to(ROOT))])
    if episode == 14:
        a = real.saved.load_npz(ledger, directory / 'trajectory.npz', pins['trajectory.npz'])
        k = a['camera_K'].copy(); k[0] *= 640 / spec.width; k[1] *= 480 / spec.height; k[:2, 2] -= .5
        bank = dict(vertices=a['object_vertices'], faces=a['object_faces'], rotations=a['object_rotation'],
            translations=a['object_translation'], K=k, original_K=a['camera_K'], points=np.empty((0, 3)),
            xy=np.empty((src_count, 0, 2)), visible=np.empty((src_count, 0), bool),
            frame_index=a['frame_index'], fps=30., ids=None, evidence=None, object_scale=a['object_scale'])
    return bank, dict(base=base, spec=spec, directory=directory, export_pins=pins, human=human,
        native=native, raw=raw, mesh=mesh, forward=forward_report, inventory=inventory, video=video)


def native_assets(ledger, src):
    vendor = ROOT / 'vendor/video_to_data'
    full.body._pinned_checkout(vendor, full.body.UPSTREAM_REVISION)
    native = vendor / 'reconstruction/modules/v2d_cari4d/lib/cari4d'
    assets, hashes = full.body._body_assets(ROOT)
    package = full.body._source_identity(ROOT)
    if hashes != src['forward']['body_assets'] or package != src['forward']['inference_source_identity']:
        raise ValueError('Exact native model/source lineage differs')
    ledger.record(ROOT / 'results/weights-acquisition.json')
    for name, pin in hashes.items(): ledger.record(assets / name, pin)
    ledger.record(native / full.contract.OPTIMIZER_RELATIVE_PATH,
        dict(bytes=92824, sha256=full.contract.OPTIMIZER_SHA256))
    pin = ledger.record(native / 'lib_mhr/mhr_layer.py')
    if pin['sha256'] != full.LAYER_SHA: raise ValueError('Exact differentiable MHR layer required')
    receipt = strict(ledger.read(ROOT / 'results/cari-refinement-assets.json'))
    full.contract.require_asset_receipt(receipt)
    for name, pin in full.contract.REFINEMENT_ASSETS.items():
        ledger.record(ROOT / 'weights/cari4d/refinement' / name, {k: pin[k] for k in ('bytes', 'sha256')})
    return native, assets


def dwpose_observations(ledger, src, out, report, persist):
    """One original RGB pass, native CPU predictor, literal automatic SAM boxes."""
    import cv2
    from PIL import Image
    assets = dw.validate_assets(ROOT)
    report['DWPose_assets'] = assets
    ledger.record(ROOT / dw.audit.OUT / 'report.json', assets['audit_receipt'])
    ledger.record(ROOT / dw.acquisition.REPORT)
    ledger.record(ROOT / dw.audit.PREVIOUS_REPORT)
    for row in assets['files']:
        path = Path(row['file'])
        if not path.is_absolute(): path = ROOT / dw.acquisition.BASE / path
        ledger.record(path, {k: row[k] for k in ('bytes', 'sha256')})
    for row in assets['retained_notices']: ledger.record(ROOT / dw.audit.OUT / row['file'],
        {k: row[k] for k in ('bytes', 'sha256')})
    n, h, w = src['spec'].total_frames, src['spec'].height, src['spec'].width
    xy = np.full((n, 133, 2), np.nan); scores = np.zeros((n, 133), np.float32)
    boxes = np.full((n, 4), np.nan, np.float32); present = np.zeros(n, bool); rgb_hashes = []
    started = time.monotonic()
    with dw.private_prefix(report, persist) as prefix:
        subprocess.run(dw.pip_argv(prefix, ROOT), check=True, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ort, origins = dw.import_runtime(prefix); report['DWPose_runtime'] = origins
        path = ROOT / dw.acquisition.BASE / 'source/onnxpose.py'
        spec = util.spec_from_file_location('world_reward_native_joint_DWPose', path)
        predictor = util.module_from_spec(spec); spec.loader.exec_module(predictor)
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(ROOT / dw.acquisition.BASE / dw.acquisition.ASSETS[0][0]),
            sess_options=options, providers=['CPUExecutionProvider']); session.disable_fallback()
        dw.validate_session(session); dw.validate_options(session, ort)
        capture = cv2.VideoCapture(str(src['video']))
        try:
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n:
                raise ValueError('Full original video count required')
            fps = real.old.actual_fps(capture, 30.)
            for frame in range(n):
                ok, bgr = capture.read()
                if not ok or bgr.shape != (h, w, 3) or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1:
                    raise ValueError('Original full RGB chronology/grid required')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); rgb_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                name = f'0/{frame:06d}.png'; mask_path = src['base'] / 'automatic_masks/masks' / name
                ledger.record(mask_path, src['inventory'][name])
                with Image.open(mask_path) as im:
                    mask = np.asarray(im).copy()
                    if im.mode != 'L' or mask.shape != (h, w) or not np.isin(mask, [0, 255]).all():
                        raise ValueError('Original literal automatic binary actor mask required')
                yy, xx = np.nonzero(mask)
                if len(xx):
                    bbox = np.array([[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], np.float32)
                    p, s = predictor.inference_pose(session, bbox, rgb)
                    dw.validate_prediction(p, s)
                    xy[frame], scores[frame], boxes[frame], present[frame] = p[0], s[0], bbox[0], True
            if capture.read()[0]: raise ValueError('Extra original frames found')
            real.old.actual_fps(capture, fps)
        finally: capture.release()
        del session, predictor, ort; gc.collect()
    report.update(DWPose_calls=int(present.sum()), DWPose_seconds=time.monotonic() - started,
        decoded_original_RGB_sha256=rgb_hashes, native_feed_cast=False, original_RGB_resized=False,
        absent_actor_frames=np.flatnonzero(~present).tolist())
    report['DWPose_observation_pin'] = real.seal_file(out / 'automatic_DWPose.npz', lambda stream:
        np.savez_compressed(stream, original_xy=xy, raw_scores=scores, boxes_original_xyxy=boxes,
            actor_present=present, frame_index=np.arange(n, dtype=np.int64), fps=np.array(fps)))
    # Native original-image projections use original K. Convert only the pixel
    # basis to the SAME integer-centred 640x480 grid already used by all33tracks.
    xy = xy * np.array([640 / w, 480 / h]) - .5
    return xy, scores


def validate_candidate(source_bundle, result, cfg, count):
    old, old_pose = full.validate_native_bundle(source_bundle, count)
    new, pose = full.validate_native_bundle(result, count)
    for name in old.keys() - {'mhr_trans', 'mhr_body_pose_cont'}:
        if old[name].tobytes() != new[name].tobytes(): raise ValueError('Frozen native parameter changed: ' + name)
    if old['mhr_body_pose_cont'][:, 254:].tobytes() != new['mhr_body_pose_cont'][:, 254:].tobytes():
        raise ValueError('Fixed internal body translations changed')
    if old_pose[:, :3, :3].tobytes() != pose[:, :3, :3].tobytes(): raise ValueError('Frozen object rotation changed')
    for key in source_bundle.keys() - {'pr'}:
        if full.fingerprint(source_bundle[key]) != full.fingerprint(result[key]): raise ValueError('Raw source field changed')
    if (result['postopt']['config'] != asdict(cfg) or result['postopt']['frame_indices'] != list(range(count))
            or [r['iter'] for r in result['postopt']['history']] != [0., 100., 200., 300.]
            or result['postopt']['mode'] != 'native_joint_image_translation_extension_v1'):
        raise ValueError('Complete native301-update full-T candidate required')
    return new, pose


def decode_and_seal(torch, layer, params, poses, bank, src, out):
    """Native full meshes + direct204 geometry parity, immutable render contract."""
    n = len(poses); target = np.empty((n, 18439, 3), np.float32)
    joints = np.empty((n, 127, 3), np.float32); keypoints = np.empty((n, 70, 3), np.float32)
    controls = np.empty((n, 204), np.float32); human_faces = None; direct_error = 0.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for start in range(0, n, 16):
            sl = slice(start, min(start + 16, n))
            t = {k: torch.tensor(v[sl].copy(), device='cuda', dtype=torch.float32) for k, v in params.items()}
            decoded = layer.mhr_forward(t)
            arrays = [a.cpu().numpy().copy() for a in (decoded.vertices, decoded.joints, decoded.keypoints)]
            if any(not np.isfinite(a).all() for a in arrays) or any((a[..., 2] <= 0).any() for a in arrays):
                raise ValueError('Every full native candidate geometry frame must be finite positive-Z')
            target[sl], joints[sl], keypoints[sl] = arrays
            faces = decoded.faces.cpu().numpy().copy()
            if human_faces is not None and not np.array_equal(human_faces, faces): raise ValueError('Human topology changed')
            human_faces = faces
            context = layer.backend._vertices_context(t, detach_fixed=False)
            trans, body, shape, scale = layer.backend._mutable_vertices_inputs(t, context)
            direct, dc = context.head.mhr_forward(global_trans=trans * context.flip,
                global_rot=context.global_rot, body_pose_params=body, hand_pose_params=context.hand,
                scale_params=scale, shape_params=shape, expr_params=context.face, return_model_params=True)
            direct = (direct * context.flip).cpu().numpy()
            err = float(np.linalg.norm(direct.astype(float) - arrays[0].astype(float), axis=-1).max())
            if err > 1e-5: raise ValueError('Direct204 export differs from actual native geometry')
            direct_error = max(direct_error, err); controls[sl] = dc.cpu().numpy()
    if not np.array_equal(human_faces, src['native']['human_faces']): raise ValueError('Original human faces changed')
    export.object_roundtrip(bank['vertices'], bank['faces'], poses, n)
    trajectory = export.trajectory(controls, params, poses, bank['vertices'], bank['faces'], bank['original_K'], src['spec'])
    native = dict(params, mhr_joints=joints, mhr_keypoints=keypoints, human_faces=human_faces,
        frame_index=bank['frame_index'])
    pins = dict(target=real.seal_file(out / 'target.npy', lambda stream: np.save(stream, target, allow_pickle=False)),
        trajectory=real.seal_file(out / 'trajectory.npz', lambda stream: np.savez_compressed(stream, **trajectory)),
        native_parameters=real.seal_file(out / 'native_parameters.npz', lambda stream: np.savez_compressed(stream, **native)))
    return target, keypoints, pins, direct_error


def frozen_anatomical_witness_ids(bank, src, activation):
    """Prior-only nearest-vertex proposals; same material IDs evaluated exactly in A/B."""
    from scipy.spatial import cKDTree
    tree = cKDTree(bank['vertices'][np.unique(bank['faces'])])
    ids = np.full(activation.shape, -1, np.int64)
    for frame, side in zip(*np.nonzero(activation)):
        hand_ids = src['QA_hand_ids'][side]
        local = (src['human'][frame, hand_ids] - bank['translations'][frame]) @ bank['rotations'][frame]
        distances = tree.query(local)[0]
        ids[frame, side] = hand_ids[int(np.argmin(distances))]
    return ids


def quality(bank, src, human, keypoints, rotation, translation, evidence, activation):
    """Same material/anatomical IDs and fixed automatic activations for A and B."""
    from world_reward.sequence_pose import _ContactTriangleSurface
    projected = real.old.project(bank['points'], rotation, translation, bank['K'])
    m = bank['visible'].copy(); m[0] = False
    errors = np.linalg.norm(projected[m] - bank['xy'][m], axis=-1)
    adjacent = m[1:] & m[:-1]
    increments = np.linalg.norm((np.diff(projected, axis=0) - np.diff(bank['xy'], axis=0))[adjacent], axis=-1)
    human_xyz = keypoints[:, [COCO_TO_MHR[i] for i in HELDOUT_COCO]]
    human_stats = reprojection_statistics(human_xyz, bank['K'], evidence.human_xy[:, HELDOUT_COCO],
        evidence.human_scores[:, HELDOUT_COCO] > 0, 5.)
    surface = _ContactTriangleSurface(bank['vertices'], bank['faces']); gaps = []
    # Freeze original anatomical witness vertex IDs, not world points. Human
    # moves in B: compare the SAME material hand IDs at their new actual pose.
    f, s = np.nonzero(activation)
    for frame, side in zip(f, s):
        vertex_id = src['QA_witness_ids'][frame, side]
        local = (human[frame, vertex_id] - translation[frame]) @ rotation[frame]
        gaps.append(float(surface.distances(local[None], batch_size=32)[0]))
    object_a, angular_a = real.provenance.acceleration(bank['vertices'], rotation, translation, bank['fps'])
    human_a = np.linalg.norm(np.diff(human.mean(1), n=2, axis=0), axis=-1) * bank['fps'] ** 2
    return dict(RGB_mean_px=float(errors.mean()) if len(errors) else None,
        RGB_observations=int(m.sum()), RGB_motion_increment_error_mean_px=float(increments.mean()) if len(increments) else None,
        reserved_human_RGB_mean_px=human_stats['mean_px'], reserved_human_observations=human_stats['observations'],
        same_anatomical_contact_mean_m=float(np.mean(gaps)) if gaps else None,
        same_anatomical_contact_p95_m=float(np.quantile(gaps, .95)) if gaps else None,
        object_acceleration_p95_m_s2=float(np.quantile(object_a, .95)),
        object_angular_acceleration_p95_rad_s2=float(np.quantile(angular_a, .95)),
        human_centroid_acceleration_p95_m_s2=float(np.quantile(human_a, .95)),
        original_native_active_hand_frames=len(f), heldout_accuracy_verified=False)


def quality_decision(cfg, baseline, candidate):
    comparisons = dict(RGB_mean_px='RGB_reprojection_ratio_maximum',
        RGB_motion_increment_error_mean_px='RGB_motion_increment_ratio_maximum',
        reserved_human_RGB_mean_px='reserved_human_RGB_ratio_maximum',
        same_anatomical_contact_mean_m='contact_gap_ratio_maximum',
        same_anatomical_contact_p95_m='contact_gap_ratio_maximum',
        object_acceleration_p95_m_s2='object_acceleration_p95_ratio_maximum',
        object_angular_acceleration_p95_rad_s2='object_acceleration_p95_ratio_maximum',
        human_centroid_acceleration_p95_m_s2='human_acceleration_p95_ratio_maximum')
    gates = {}
    for key, bound in comparisons.items():
        a, b = baseline.get(key), candidate.get(key)
        if key.startswith('RGB') and baseline.get('RGB_observations') == candidate.get('RGB_observations') == 0:
            continue  # Declared textureless14: no invented material RGB factor or claimed RGB gain.
        gates[key + '_nonworse'] = bool(a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
            and b <= a * cfg['gates'][bound] + 1e-12)
    return dict(passed=all(gates.values()), gates=gates, production_adopted=False,
        heldout_4D_accuracy_verified=False, scope=cfg['gates']['scope'])


def reserved_output(out, revision, episode):
    """Consume only this wrapper's exclusive per-episode CID-reserved directory."""
    import re, stat
    from mediapipe_cpu_runtime_verify import canonical
    out = canonical(out)
    if (out != ROOT / 'results' / ('native-joint-real-' + revision) / f'episode_{episode:06d}'
            or not out.is_dir() or out.stat().st_uid != 0
            or stat.S_IMODE(out.stat().st_mode) != 0o755
            or {p.name for p in out.iterdir()} != {'.container.cid'}):
        raise ValueError('Exact wrapper-owned exclusive episode output required')
    cid = out / '.container.cid'
    if cid.is_symlink() or not cid.is_file() or not re.fullmatch(b'[0-9a-f]{64}\n?', cid.read_bytes()):
        raise ValueError('Own original Docker CID must precede native run')
    return out


def run(episode):
    revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if Path(os.environ['WR_ROOT']) != ROOT or code != ROOT / 'jobs' / revision / ENTRY / 'code':
        raise ValueError('Exact Azure immutable native-joint source namespace required')
    out = ROOT / 'results' / ('native-joint-real-' + revision) / f'episode_{episode:06d}'
    reserved_output(out, revision, episode)
    report = dict(status='fail', episode=episode, producer_revision=revision, ground_truth_used=False,
        production_adopted=False, private_truth_read=False, baseline_modified=False,
        depth_used=False, full_4D_quality_verified=False, coefficient_calibration_verified=False)
    started = time.monotonic()
    def persist():
        print('NATIVE_JOINT ' + json.dumps(dict(episode=episode, phase=report.get('phase', 'start'),
            seconds=round(time.monotonic() - started, 3)), sort_keys=True), flush=True)
    try:
        report['source_binding'] = source(ROOT, code, revision, ENTRY, HELPERS)
        cfg, extension = settings(code); report['protocol'] = cfg
        report['controlled_factor_DEV'] = controlled_factor_DEV()
        ledger = real.old.ArtifactLedger()
        bank, src = original_sources(ledger, episode)
        native, assets = native_assets(ledger, src)
        report['phase'] = 'automatic_DWPose_full_T'; persist()
        human_xy, scores = dwpose_observations(ledger, src, out, report, persist)
        if (scores[:, TRAIN_COCO] > 0).sum() < 6:
            raise ValueError('No credible independent body RGB evidence; native baseline remains available')
        evidence = JointImageEvidence(bank['points'], bank['xy'], bank['visible'], human_xy, scores,
            bank['K'], bank['frame_index'], allow_empty_object_evidence=(episode == 14))
        import torch
        if str(torch.__version__) != '2.5.1+cu124' or not torch.cuda.is_available() or torch.version.cuda != '12.4':
            raise ValueError('Exact native Torch/CUDA runtime required')
        if torch.are_deterministic_algorithms_enabled(): raise ValueError('Do not alter native backward policy')
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        sys.path[:0] = [str(native), '/workspace/v2d_sam3d_body/lib']
        os.environ.update(MHR_ASSETS_ROOT=str(ROOT / 'weights/cari4d/sam3d_body'), MOMENTUM_ENABLED='0')
        from learning.training import mhr_opt_refineout as optimizer
        from lib_mhr.mhr_layer import MHRLayer
        if (Path(optimizer.__file__).resolve() != native / full.contract.OPTIMIZER_RELATIVE_PATH
                or Path(sys.modules[MHRLayer.__module__].__file__).resolve() != native / 'lib_mhr/mhr_layer.py'):
            raise ValueError('Actual unchanged native optimizer/decoder imports required')
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),
            checkpoint_path=assets / 'model.ckpt', buffer_path=out / 'never_use_unverified_buffer.pt',
            mhr_model_path=assets / 'assets/mhr_model.pt', device='cuda')
        if layer.decoder_identity() != src['forward']['decoder_identity']:
            raise ValueError('Original prepared native decoder identity differs')
        source_bundle = torch.load(src['raw'], map_location='cpu', weights_only=False)
        full.validate_source_bundle(source_bundle, src['mesh'], len(bank['frame_index']))
        vertices, faces = optimizer._load_object_vertices(src['mesh'])
        if not np.array_equal(vertices, bank['vertices']) or not np.array_equal(faces, bank['faces']):
            raise ValueError('Native loaded mesh differs from immutable original33material-bank geometry')
        native_cfg = optimizer.MHRParityPostOptConfig(
            penetration_collision_proxy_path=str(ROOT / 'weights/cari4d/refinement/mhr_collision_proxy_4000v.npz'),
            hand_surface_spec_path=str(ROOT / 'weights/cari4d/refinement/mhr_hand_surface_spec.npz'), report_every=100)
        before = full.fingerprint(source_bundle)
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        cls = native_joint_optimizer_class(optimizer, evidence, extension)
        report['phase'] = 'native_joint_constructor'; persist()
        report['object_material_RGB_factor'] = 'all33_original_tracks' if episode == 9 else 'unobserved_textureless_zero_points_native_silhouette_fallback'
        report['baseline_binding'] = dict(directory=str(src['directory']), outputs=src['export_pins'])
        instance = cls(source_bundle, vertices, faces, native_cfg, mhr_layer=layer)
        indices = torch.arange(len(bank['frame_index']), device='cuda', dtype=torch.int64)
        report['phase'] = 'native_coupled_gradient_preflight'; persist()
        loss, metrics = instance.loss(indices, 181, include_diagnostics=False)
        if not torch.isfinite(loss): raise ValueError('Nonfinite native full-T initial objective')
        gradients = torch.autograd.grad(loss, (instance.object_translation, instance.human_translation,
            instance.body_pose), allow_unused=False)
        norms = [float(g.norm().detach().cpu()) for g in gradients]
        if not all(np.isfinite(norms)) or not all(x > 0 for x in norms):
            raise ValueError('Actual native pair/articulation gradients absent or invalid')
        report['preflight_gradient_norms'] = dict(zip(('object_translation', 'human_translation', 'MHR_body_rotation'), norms))
        del loss, metrics, gradients; gc.collect(); torch.cuda.empty_cache()
        report['phase'] = 'native_joint_301_updates'; persist(); fit_started = time.monotonic()
        result = instance.run(); report['fit_seconds'] = time.monotonic() - fit_started
        params, poses = validate_candidate(source_bundle, result, native_cfg, len(bank['frame_index']))
        if full.fingerprint(source_bundle) != before: raise ValueError('Original native bundle was modified')
        report['actual_native_fit_completed'] = True
        report['finite_full_T_native_history'] = all(np.isfinite(value) for row in result['postopt']['history'] for value in row.values())
        if not report['finite_full_T_native_history']: raise ValueError('Nonfinite native optimization history')
        report['candidate_bundle_pin'] = real.seal_file(out / 'refined.pth', lambda stream: torch.save(result, stream))
        activation = instance.contact_mask.detach().cpu().numpy().astype(bool)
        src['QA_hand_ids'] = instance.hand_surface_vertex_indices.detach().cpu().numpy().copy()
        report['native_config'] = asdict(native_cfg); report['native_history'] = result['postopt']['history']
        del instance, result; gc.collect(); torch.cuda.empty_cache()
        report['phase'] = 'native_direct_export_and_seal'; persist()
        human, keypoints, pins, error = decode_and_seal(torch, layer, params, poses, bank, src, out)
        report.update(candidate_outputs=pins, native_direct_max_error_m=error,
            fitted_outputs_sealed_before_QA=True, points=len(bank['points']), frames=len(bank['frame_index']),
            native_effective_updates=301, baseline_rerun=False, GPU_bit_parity_claimed=False)
        real.seal_json(out / 'fit.json', report)
        del layer; gc.collect(); torch.cuda.empty_cache()
        report['phase'] = 'same_automatic_observation_QA'; persist()
        src['QA_witness_ids'] = frozen_anatomical_witness_ids(bank, src, activation)
        report['QA_witness_definition'] = 'frozen_baseline_nearest_object_vertex_proposed_hand_vertex_exact_triangle_evaluated_sameIDs'
        report['native_active_witness_pin'] = real.seal_file(out / 'QA_witnesses.npz', lambda stream:
            np.savez_compressed(stream, activations=activation, hand_vertex_ids=src['QA_witness_ids'], frame_index=bank['frame_index']))
        a = quality(bank, src, src['human'], src['native']['mhr_keypoints'], bank['rotations'], bank['translations'], evidence, activation)
        b = quality(bank, src, human, keypoints, poses[:, :3, :3], poses[:, :3, 3], evidence, activation)
        report.update(metrics={'A_saved_native_baseline': a, 'B_native_joint_RGB_translation': b},
            decision=quality_decision(cfg, a, b))
        ledger.verify(); report.update(status='complete_diagnostic_not_quality_pass', phase='complete',
            source_inputs_rehashed=True, input_ledger=ledger.records, elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:400], elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        real.seal_json(out / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--episode', required=True, type=int, choices=(9, 1, 14))
    def expired(*_): raise TimeoutError('Frozen1800second native pilot budget exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(BUDGET)
    try: run(parser.parse_args().episode)
    finally: signal.alarm(0)

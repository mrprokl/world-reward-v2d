"""Public-only FORM96 native HOI pipeline, with sealed same-input A/B outputs.

This adapter deliberately does not call any Track1-bound stage main, declare a
fake episode, import a reference mesh, or assemble a submission. Every model and
geometry kernel is native or an existing generic production primitive. FORM
references are handled by a different, subsequently launched evaluator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
from importlib import util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np
from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
import form_hoi_external_dev as public
from world_reward.shared_identity import (NATIVE_PARAMETER_DIMS, share_first_frame_identity,
    shared_initializer_metadata, validate_native_parameters)

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_form_hoi_external_predict'
CONFIG = 'configs/form_hoi_external_predict_v1.json'
STAGES = ('localize', 'track', 'body_depth', 'object', 'prepare', 'forward', 'fit_A', 'fit_B')
HELPERS = ('infra/form_hoi_external_predict.py', 'infra/run_form_hoi_external_predict.sh', CONFIG,
    'infra/form_hoi_external_cohort.py', 'infra/run_form_hoi_external_cohort.sh',
    'infra/form_prediction_reuse.py',
    'infra/form_hoi_external_dev.py', 'infra/form_hoi_external_acquire.py',
    'infra/mediapipe_cpu_runtime_verify.py', 'configs/form_hoi_external_dev_v1.json',
    'configs/form_hoi_insight_v1.json', 'configs/sam31_runtime_v1.json',
    'infra/body_smoke.py', 'infra/depth_smoke.py', 'infra/bridge_rgb_anchor_infer.py',
    'infra/bridge_frontend_bindings.py', 'infra/object_smoke.py', 'infra/object_pose_smoke.py',
    'infra/cari_body_adapter.py', 'infra/cari96_native.py', 'infra/cari96_forward.py',
    'infra/cari_full_refine.py', 'infra/cari_refine.py', 'infra/cari_converter.py',
    'infra/cari_runner.py', 'infra/camera_render.py', 'infra/sam31_runtime.py',
    'infra/dwpose_smoke.py', 'infra/dwpose_acquire.py', 'infra/dwpose_wheel_audit.py',
    'infra/sequence_pose_probe.py', 'src/world_reward/gemini_localization.py',
    'src/world_reward/vertex_retry.py', 'src/world_reward/seeded_tracking.py',
    'src/world_reward/shared_identity.py', 'src/world_reward/metric_alignment.py',
    'src/world_reward/pointmap.py', 'src/world_reward/rigid_alignment.py',
    'src/world_reward/pose_selection.py', 'src/world_reward/native_joint_refinement.py',
    'src/world_reward/joint_point_objective.py', 'src/world_reward/root_refit.py',
    'src/world_reward/point_surface_queries.py')


def config(code):
    c = strict((code / CONFIG).read_bytes())
    require(c['schema'] == 'world_reward.form_hoi_external_predict.v1' and
        c['num_steps'] == 300 and c['batch_size'] == 0 and c['seed_frames'] == [0, 14, 29] and
        c['body_inference_type'] == 'body' and c['object_pose_orientation_hypotheses'] == 24 and
        all(c[k] is False for k in ('ground_truth_used', 'private_truth_read', 'hand_labeled_test',
            'training_overlap_verified', 'production_adopted', 'full_4D_accuracy_verified')) and
        c['oracle_modes'] == [], 'Frozen external public-only native protocol required')
    from world_reward.native_joint_refinement import NativeJointConfig
    extension = NativeJointConfig(**c['extension'])
    require(asdict(extension) == asdict(NativeJointConfig(5., 1., 1., 10., .1, 1e-4,
        'published_native10_0.1_lr1e-4_and_root_RGB5px_proposal_frozen_tiny_factor_DEV_not_HOI_calibration')),
        'No coefficient sweep or challenge-derived coefficient selection')
    return c


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def artifact(path):
    p = canonical(path); s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and not s.st_mode & 0o222,
        'Sealed regular single-link artifact required')
    return dict(bytes=s.st_size, sha256=sha(p))


def seal(path, callback):
    path = Path(path); tmp = path.with_name('.' + path.name + '.part')
    require(not path.exists() and not tmp.exists(), 'Exclusive output; never overwrite a baseline/result')
    try:
        with tmp.open('xb') as f:
            callback(f); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
        os.link(tmp, path); tmp.unlink()
    finally:
        if tmp.exists(): tmp.unlink()
    return artifact(path)


def save_json(path, value):
    return seal(path, lambda f: f.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()))


def save_npz(path, **values):
    return seal(path, lambda f: np.savez_compressed(f, **values))


def load_public(path, expected, code):
    path = canonical(path)
    require(path.name == 'input.json' and path.parent.name == 'inputs' and
        artifact(path) == expected, 'Independent qualified public input pin required')
    p = strict(path.read_bytes()); cfg = strict((code / 'configs/form_hoi_external_dev_v1.json').read_bytes())
    public.validate_public_package(p, cfg, require_ready=True)
    cohort = strict((code / 'configs/form_hoi_insight_v1.json').read_bytes())
    dev = {r['sequence_id'] for r in cohort['cohort'] if r['split'] == 'development'}
    require(p['sequence_id'] in dev and len(dev) == 4 and
        re.fullmatch(r'[A-Za-z0-9_-]{1,128}', p['sequence_id']) and
        Path(p['video']) == path.parent / 'rgb.mp4' and path.parent.parent.name == p['sequence_id'] and
        {f.name for f in path.parent.iterdir()} == {'input.json', 'rgb.mp4'} and
        artifact(Path(p['video'])) == p['video_pin'], 'Only the frozen DEV RGB/text directory is accessible')
    return p


def fixed_K(p):
    focal = float(np.hypot(p['height'], p['width']))
    return np.array([[focal, 0, p['width'] / 2], [0, focal, p['height'] / 2], [0, 0, 1]], np.float64)


def frames(p):
    """Stream only the declared original prefix; never renumber/select/resize it."""
    import cv2
    cap = cv2.VideoCapture(p['video'])
    require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == p['full_source_frames'] and
        np.isclose(cap.get(cv2.CAP_PROP_FPS), p['fps']), 'Original full video count/fps differs')
    try:
        for i in p['original_frame_indices']:
            ok, bgr = cap.read()
            require(ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) == i + 1 and
                bgr.shape == (p['height'], p['width'], 3), 'Original96 RGB chronology/grid differs')
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally: cap.release()


def read_stage(base, stage):
    d = canonical(base / stage); report = strict((d / 'report.json').read_bytes())
    require(report.get('status') == 'complete' and report.get('stage') == stage and
        report.get('ground_truth_used') is False and report.get('private_truth_read') is False,
        'Only sealed complete public-only stage outputs may be consumed')
    actual = {str(f.relative_to(d)): artifact(f) for f in sorted(d.rglob('*'))
        if f.is_file() and f.name != 'report.json'}
    require(actual == report['artifacts'], 'Sealed stage payload changed')
    artifact(d / 'report.json')
    return d, report


def mask(base, role, index, p):
    from PIL import Image
    with Image.open(base / 'track/masks' / str(role) / f'{index:06d}.png') as im:
        a = np.asarray(im).copy()
        require(im.mode == 'L' and a.shape == (p['height'], p['width']) and np.isin(a, [0, 255]).all(),
            'Literal automatic original-resolution mask required')
    return a > 0


def localize(p, c, base, out, report):
    from io import BytesIO
    from PIL import Image
    from world_reward.gemini_localization import build_request, parse_response
    from world_reward.vertex_retry import call_with_retry
    token = sys.stdin.buffer.read(16385).decode().strip()
    require(0 < len(token) < 16384 and not any(ch.isspace() for ch in token), 'RAM-only one-shot OAuth required')
    tasks = []
    for i, rgb in frames(p):
        if i not in c['seed_frames']: continue
        f = BytesIO(); Image.fromarray(rgb).save(f, format='PNG')
        lc = c['localizer']
        request = build_request(f.getvalue(), i, p['object_prompt'], p['action'],
            thinking_level=lc['thinking_level'], media_resolution=lc['media_resolution'],
            max_output_tokens=lc['max_output_tokens'])
        tasks.append((i, hashlib.sha256(rgb.tobytes()).hexdigest(),
            json.dumps(request, separators=(',', ':')).encode()))
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw): raise ValueError('Credential redirects prohibited')
    lc = c['localizer']; url = f"https://aiplatform.googleapis.com/v1/projects/{lc['project']}/locations/{lc['location']}/publishers/google/models/{lc['model']}:generateContent"
    deadline = time.monotonic() + c['budgets']['localize'] - 5
    def ask(task):
        i, digest, body = task
        def transport(payload, timeout):
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            req = urllib.request.Request(url, data=payload, headers={'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'})
            with op.open(req, timeout=timeout) as response:
                raw = response.read(200001); require(len(raw) <= 200000, 'Bounded Vertex response required')
            return raw
        result = call_with_retry(body, transport, deadline=deadline, attempt_timeout=60, max_attempts=3)
        value = strict(result.response); candidates = value.get('candidates', [])
        require(len(candidates) == 1 and candidates[0].get('finishReason') == 'STOP',
            'Complete single Vertex response required; no scientific response shopping')
        text = ''.join(x.get('text', '') for x in candidates[0].get('content', {}).get('parts', [])
            if not x.get('thought', False))
        row = parse_response(text, target_frame=i, height=p['height'], width=p['width'])
        return dict(row, rgb_sha256=digest, request_sha256=hashlib.sha256(body).hexdigest(),
            attempts=list(result.attempts), model_version=value.get('modelVersion'))
    try:
        with ThreadPoolExecutor(max_workers=3) as pool: rows = list(pool.map(ask, tasks))
    finally: token = None
    require(any(r['person_bbox'] and r['object_bbox'] for r in rows), 'No automatically localized pair; do not substitute a person/object')
    report['calls'] = rows; report['localizer'] = lc
    save_json(out / 'boxes.json', dict(rows=rows, conditioning='public_RGB_object_prompt_action_only'))


def track(p, c, base, out, report):
    import torch
    from PIL import Image
    from sam31_runtime import compatible_init, verify_runtime_source, verify_checkpoint_coverage
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from world_reward.seeded_tracking import corner_prompt, singleton_outputs, native_masks
    _, loc = read_stage(base, 'localize'); rows = loc['calls']
    prep = strict(Path(c['sam_prepare_receipt']).read_bytes())
    require(artifact(Path(c['sam_prepare_receipt'])) == c['sam_prepare_pin'], 'Qualified SAM runtime receipt differs')
    verify_runtime_source(prep)
    weight = Path(prep['weight_file']); require(artifact(weight) == prep['weight'], 'Qualified SAM checkpoint differs')
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(weight), max_num_objects=2,
        multiplex_count=16, use_fa3=False, use_rope_real=True, compile=False, warm_up=False, async_loading_frames=False)
    report['checkpoint_coverage'] = verify_checkpoint_coverage(predictor.model, weight)
    images, hashes = [], []
    for _, rgb in frames(p): images.append(Image.fromarray(rgb)); hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    state, omitted = compatible_init(predictor.model, images)
    require(omitted == ['offload_state_to_cpu'], 'Qualified native PIL initialization changed')
    tracker = predictor.model.tracker
    states = [tracker.init_state(video_height=p['height'], video_width=p['width'], num_frames=96,
        cached_features=state['feature_cache']) for _ in range(2)]
    for row in rows:
        i = row['frame_index']; require(hashes[i] == row['rgb_sha256'], 'Localizer/tracker RGB mismatch')
        for obj, role in enumerate(('person', 'object')):
            box = row[role + '_bbox']
            if box is None: continue
            points, labels = corner_prompt(box, p['width'], p['height'])
            predictor.model._prepare_backbone_feats(state, i, reverse=False)
            tracker.add_new_points(inference_state=states[obj], frame_idx=i, obj_id=obj,
                points=torch.tensor(points, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.int32),
                clear_old_points=True, rel_coordinates=True, use_prev_mem_frame=False)
    for s in states: tracker.propagate_in_video_preflight(s, run_mem_encoder=True)
    areas = [[], []]
    for obj in range(2): (out / 'masks' / str(obj)).mkdir(parents=True)
    for i in range(96):
        predictor.model._prepare_backbone_feats(state, i, reverse=False); outputs = []
        for s in states:
            values = list(tracker.propagate_in_video(s, start_frame_idx=i, max_frame_num_to_track=0,
                reverse=False, tqdm_disable=True, run_mem_encoder=True))
            require(len(values) == 1, 'Native fixed-ID single-frame transport required'); outputs.append(values[0])
        ids, logits, scores = singleton_outputs(outputs, i)
        m, presence = native_masks(ids, torch.cat(logits).detach().float().cpu().numpy(),
            torch.cat(scores).detach().float().cpu().numpy(), p['height'], p['width'])
        for obj in range(2):
            seal(out / 'masks' / str(obj) / f'{i:06d}.png', lambda f, a=m[obj]:
                Image.fromarray(a.astype('uint8') * 255).save(f, format='PNG'))
            areas[obj].append(int(m[obj].sum()))
    require(all(any(a) for a in areas), 'Both physical identities must have native observed support')
    save_json(out / 'observations.json', dict(rgb_sha256=hashes, areas=areas, original_frame_indices=p['original_frame_indices']))
    report.update(areas=areas, source_revision=prep['source_revision'], model_revision=prep['model_revision'],
        checkpoint=prep['weight'], instance_state_policy='two_native_singleton_PVS_shared_backbone', masks_interpolated=False)


def load_body(root, report):
    import torch
    import body_smoke as body
    report['inference_source_identity'] = body._source_identity(root)
    assets, report['body_assets'] = body._body_assets(root)
    original, calls = body._install_local_dinov3_loader(torch,
        root / 'weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main')
    try:
        import sam_3d_body
        from sam_3d_body import build_models
        loader = build_models.load_state_dict
        independent = torch.jit.load(str(assets / 'assets/mhr_model.pt'), map_location='cpu').state_dict()
        def strict_loader(module, state_dict, strict=False, logger=None):
            report['checkpoint_loading'] = body._load_checkpoint_with_asset_buffers(module, state_dict,
                loader, torch, explicit_asset_state=independent)
            prior = module.backbone.encoder.prepare_tokens_with_masks
            def unmasked(x, masks=None):
                require(masks is None, 'Audited DINO unmasked RGB route required'); return prior(x, masks=None)
            module.backbone.encoder.prepare_tokens_with_masks = unmasked
        build_models.load_state_dict = strict_loader
        try:
            model, cfg = sam_3d_body.load_sam_3d_body(checkpoint_path=str(assets / 'model.ckpt'),
                device='cuda', mhr_path=str(assets / 'assets/mhr_model.pt'))
        finally: build_models.load_state_dict = loader
    finally: torch.hub.load = original
    require(len(calls) == 1 and cfg.MODEL.BACKBONE.TYPE == calls[0], 'Exact offline DINOv3 backbone required')
    return model, sam_3d_body.SAM3DBodyEstimator(model, cfg)


def body_transport(records, faces, indices, image_size):
    """Exact existing Body-NPZ ABI; no invented inverse controls or geometry.

    Only the native worker supplies records. This pure transport is separately
    testable without a model: pred_vertices is intentionally renamed exactly
    as body_smoke/candidate_mhr_parameters require; translation is added once.
    """
    from body_smoke import PARAMETER_SHAPES
    count = len(indices); require(count > 0 and np.array_equal(indices, np.arange(count)),
        'Original contiguous Body transport chronology required')
    require(set(records) == set(PARAMETER_SHAPES), 'Every original native Body parameter block required')
    arrays = {}
    for key, shape in PARAMETER_SHAPES.items():
        values = records[key]
        require(len(values) == count, 'Body transport row count differs: ' + key)
        a = np.stack(values)
        require(a.dtype == np.float32 and a.shape == (count, *shape) and np.isfinite(a).all(),
            'Exact original finite FP32 Body transport required: ' + key)
        arrays[key] = a
    f = np.asarray(faces)
    require(f.dtype.kind in 'iu' and f.ndim == 2 and f.shape[1] == 3 and len(f) and
        f.min() >= 0 and f.max() < 18439, 'Native Body triangle topology required')
    height, width = image_size; focal = float(np.hypot(height, width))
    require(np.allclose(arrays['focal_length'], focal, rtol=1e-6, atol=1e-4) and
        np.count_nonzero(arrays['expr_params']) == 0 and (arrays['pred_cam_t'][:, 2] > 0).all(),
        'Original RGB-size Body camera/zero-expression contract required')
    root = arrays.pop('pred_vertices'); trans = arrays['pred_cam_t'][:, None]
    keypoints = arrays['pred_keypoints_3d'] + trans
    require((keypoints[..., 2] > 0).all(), 'Original Body keypoints must remain in front of the camera')
    projected = focal * keypoints[..., :2] / keypoints[..., 2:] + [width / 2., height / 2.]
    require(np.max(np.linalg.norm(projected - arrays['pred_keypoints_2d'], axis=-1)) <= .05,
        'Original Body image projection differs; no calibration repair')
    arrays.update(vertices_root_camera_m=root, vertices_camera_m=root + trans,
        faces=f.astype(np.int64, copy=True), frame_index=np.asarray(indices, dtype=np.int64))
    return arrays


def body_depth(p, c, base, out, report):
    import torch
    import body_smoke as body
    from world_reward.pointmap import validate_camera_pointmap
    from world_reward.metric_alignment import fit_shared_depth_scale
    from camera_render import raster_camera_mesh
    from moge.model.v2 import MoGeModel
    read_stage(base, 'track')
    expected = strict((base / 'track/observations.json').read_bytes())['rgb_sha256']
    model, estimator = load_body(ROOT, report)
    arrays = {k: [] for k in body.PARAMETER_SHAPES}; hashes = []
    last_box = None; absent = []
    for i, rgb in frames(p):
        digest = hashlib.sha256(rgb.tobytes()).hexdigest(); require(digest == expected[i], 'Body/tracking RGB mismatch'); hashes.append(digest)
        m = mask(base, 0, i, p); ys, xs = np.nonzero(m)
        if len(xs): last_box = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1], np.float32)
        else: absent.append(i)
        require(last_box is not None, 'Leading absent actor requires a separate automatic RGB recovery anchor')
        require(np.min(last_box[2:] - last_box[:2]) > 1, 'Automatic actor crop is degenerate')
        # A propagated automatic box is only a crop proposal; RGB is still run
        # through the native model. It is not an invented silhouette or pose.
        with torch.inference_mode():
            results = estimator.process_one_image(img=rgb, bboxes=last_box[None],
                masks=m.astype(np.uint8)[..., None], cam_int=None, inference_type='body')
        require(len(results) == 1, 'One native body proposal required')
        pred = results[0]
        for k, shape in body.PARAMETER_SHAPES.items():
            a = pred.get(k); require(torch.is_tensor(a) and tuple(a.shape) == shape and torch.isfinite(a).all(),
                'Native body parameter finite/shape contract failed: ' + k)
            arrays[k].append(a.detach().float().cpu().numpy())
        with torch.inference_mode():
            decoded, kp, joints, controls, rotations = body._native_forward_from_blocks(model.head_pose, pred)
            flip = torch.tensor([1., -1., -1.], device=decoded.device, dtype=decoded.dtype)
            errors = [float(torch.linalg.vector_norm(actual[0] * flip - pred[key], dim=-1).max())
                for actual, key in ((decoded, 'pred_vertices'), (joints, 'pred_joint_coords'), (kp[:, :70], 'pred_keypoints_3d'))]
            errors.append(float((controls[0] - pred['mhr_model_params']).abs().max()))
        require(max(errors) < 1e-5 and np.count_nonzero(arrays['expr_params'][-1]) == 0,
            'Original native vertices/joints/keypoints/controls or zero expressions changed')
    arrays = body_transport(arrays, np.asarray(estimator.faces, np.int64),
        p['original_frame_indices'], (p['height'], p['width']))
    require(arrays['faces'].shape == (36874, 3), 'Exact native MHR body topology required')
    save_npz(out / 'body.npz', **arrays)
    del model, estimator; gc.collect(); torch.cuda.empty_cache()
    from bridge_rgb_anchor_infer import moge_asset, host_moge_chain
    graph = host_moge_chain(ROOT)
    def acquisition_identity(path, maximum):
        require(path == ROOT / 'results/weights-acquisition.json', 'Only original acquisition metadata allowed')
        return dict(bytes=path.stat().st_size, sha256=sha(path))
    checkpoint, acquisition_pin, model_pin = moge_asset(ROOT, acquisition_identity=acquisition_identity)
    require(host_moge_chain(ROOT) == graph, 'Actual MoGe link graph changed')
    report['MoGe_verified_asset'] = dict(artifact=model_pin, acquisition=acquisition_pin, link_graph=graph)
    depth_model = MoGeModel.from_pretrained(str(checkpoint)).cuda().eval()
    K = fixed_K(p); focal = K[0, 0]; fov = float(np.degrees(2 * np.arctan(p['width']/(2*focal))))
    (out / 'depth').mkdir(); anchor = [0, 47, 95]; depths = []; renders = []; supports = []
    for i, rgb in frames(p):
        require(hashlib.sha256(rgb.tobytes()).hexdigest() == hashes[i], 'Body/depth original RGB differs')
        image = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        with torch.inference_mode(): values = depth_model.infer(image[None], fov_x=fov)
        depth = values['depth'][0].float().cpu().numpy(); valid = values['mask'][0].cpu().numpy().astype(bool)
        points = values['points'][0].float().cpu().numpy(); intrinsic = values['intrinsics'][0].float().cpu().numpy()
        validate_camera_pointmap(depth, points, valid, intrinsic, K)
        save_npz(out / 'depth' / f'{i:06d}.npz', depth=depth, mask=valid, intrinsics=intrinsic, frame_index=np.array(i))
        if i in anchor:
            silhouette, z = raster_camera_mesh(arrays['vertices_camera_m'][i], arrays['faces'], K, p['width'], p['height'])
            depths.append(depth); renders.append(z.cpu().numpy())
            supports.append(silhouette.cpu().numpy() & mask(base, 0, i, p) & ~mask(base, 1, i, p) & valid)
    aligned = fit_shared_depth_scale(depths, renders, supports, anchor)
    del depth_model; gc.collect(); torch.cuda.empty_cache()
    save_json(out / 'gauge.json', dict(K=K.tolist(), alignment=aligned.to_dict(), rgb_sha256=hashes,
        camera_source='RGB_size_prior_only', scale_source='one_predicted_human_anchored_clip_scalar', absent_actor_mask_frames=absent))
    report.update(scale=aligned.shared_scale, gauge_accuracy_verified=False, body_absent_mask_crop_proposal_frames=absent,
        MoGe_checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=sha(checkpoint)))


def pointmap(depth, valid, K, scale):
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    points = np.stack(((xx+.5-K[0, 2])*depth/K[0, 0],
        (yy+.5-K[1, 2])*depth/K[1, 1], depth), axis=-1) * scale
    points[~valid] = np.nan
    return points.astype(np.float32)


def object_mesh(p, c, base, out, report):
    """One native SAM3D object generation, using the shared camera/gauge once."""
    import torch
    import trimesh
    from PIL import Image
    import body_smoke as body
    read_stage(base, 'track'); read_stage(base, 'body_depth')
    gauge = strict((base / 'body_depth/gauge.json').read_bytes()); K = np.asarray(gauge['K'])
    scale = gauge['alignment']['shared_scale']; selected = None
    for i, rgb in frames(p):
        m = mask(base, 1, i, p)
        if m.any(): selected = (i, rgb, m); break
    require(selected is not None, 'No observed target object; cannot reconstruct another object')
    i, rgb, m = selected
    with np.load(base / 'body_depth/depth' / f'{i:06d}.npz', allow_pickle=False) as z:
        points = pointmap(z['depth'], z['mask'], K, scale)
    require((m & np.isfinite(points).all(-1)).any(), 'Object anchor has no valid predicted pointmap')
    save_npz(out / 'anchor.npz', frame_index=np.array(i), K=K)
    seal(out / 'frame.png', lambda f: Image.fromarray(rgb).save(f, format='PNG'))
    seal(out / 'mask.png', lambda f: Image.fromarray(m.astype('uint8')*255).save(f, format='PNG'))
    seal(out / 'pointmap.npy', lambda f: np.save(f, points, allow_pickle=False))
    save_json(out / 'pointmap_intrinsics.json', dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        width=p['width'], height=p['height']))
    weights = ROOT / 'weights/sam3d'; repository = weights / 'torch_home/hub/facebookresearch_dinov2_main'
    body._pinned_checkout(repository, '7764ea0f912e53c92e82eb78a2a1631e92725fc8')
    acquired = strict((ROOT / 'results/weights-acquisition.json').read_bytes())
    rows = [r for r in acquired['assets'] if r['repo_id'] == 'facebook/sam-3d-objects']
    require(len(rows) == 1 and rows[0]['revision'] == '2e73555018d2741ccd486e56c24fac41155a1dc6',
        'Exact acquired native Objects checkpoint required')
    aux = strict((ROOT / 'results/auxiliary-assets.json').read_bytes())
    for name in ('dinov2_vitl14_reg4_pretrain.pth', 'dinov2_vitb14_reg4_pretrain.pth'):
        r = [v for v in aux['checkpoints'] if v['filename'] == name]
        require(len(r) == 1 and sha(weights / 'torch_home/hub/checkpoints' / name) == r[0]['sha256'],
            'Exact independently acquired DINO register weights required')
    original = torch.hub.load
    def local_hub(repo, *a, **kw):
        require(repo in ('facebookresearch/dinov2', 'facebookresearch/dinov2:main'), 'Unexpected Objects hub source')
        require(kw.get('model', a[0] if a else None) in ('dinov2_vitl14_reg', 'dinov2_vitb14_reg'),
            'Unexpected Objects backbone'); kw['source'] = 'local'; return original(str(repository), *a, **kw)
    torch.hub.load = local_hub
    try:
        from v2d.sam3d.lib.image_to_mesh import image_to_mesh
        image_to_mesh(str(out / 'frame.png'), str(out / 'mask.png'), str(out / 'object.glb'),
            str(out / 'transform.json'), str(out / 'intrinsics.json'), str(weights), seed=0,
            with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False,
            use_vertex_color=True, pointmap_path=str(out / 'pointmap.npy'),
            pointmap_intrinsics_path=str(out / 'pointmap_intrinsics.json'))
    finally: torch.hub.load = original
    for name in ('object.glb', 'transform.json', 'intrinsics.json'): (out / name).chmod(0o444)
    mesh = trimesh.load(out / 'object.glb', force='mesh', process=False)
    require(isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) >= 3 and len(mesh.faces) >= 1 and
        np.isfinite(mesh.vertices).all() and mesh.area > 0, 'Native generated whole object must be a nonempty mesh')
    transform = strict((out / 'transform.json').read_bytes()); s = np.asarray(transform['scale'])
    require(s.shape == (3,) and np.isfinite(s).all() and (s > 0).all(), 'Clip-constant positive generated scale required')
    report.update(anchor_frame_index=i, model_revision=rows[0]['revision'], metric_scale_verified=False,
        pointmap_grounding='shared_human_scale_already_applied_no_second_scale',
        source_license='SAM_3D_Objects_Materials_license_not_wrapper_license',
        vertices=len(mesh.vertices), faces=len(mesh.faces), watertight=bool(mesh.is_watertight))


def pose_initializer(vertices, faces, scale, R0, t0, p, base, K):
    """Existing generic ICP + observed-frame Viterbi; missing masks stay missing."""
    import trimesh
    from scipy.spatial.transform import Rotation
    from world_reward.rigid_alignment import align_observed_points
    from camera_render import raster_camera_mesh_batch, silhouette_iou
    from object_pose_smoke import _finite_pose_candidate, _select_latent_pose_path
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    sampled, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0)
    hypotheses = Rotation.create_group('O').as_matrix(); records = []; previous = R0.copy()
    for i in range(96):
        m = mask(base, 1, i, p)
        with np.load(base / 'body_depth/depth' / f'{i:06d}.npz', allow_pickle=False) as z:
            points = pointmap(z['depth'], z['mask'], K, scale)
        visible = m & np.isfinite(points).all(-1) & (points[..., 2] > 0); observed = points[visible]
        if len(observed) < 40:
            records.append(dict(frame_index=i, candidates=[], pose_observed=False)); continue
        if len(observed) > 2048: observed = observed[np.random.default_rng(0).choice(len(observed), 2048, replace=False)]
        rotations = [R0 @ h for h in hypotheses] + [previous]
        initial = []; fitted = []; fits = []; slots = []
        for slot, r in enumerate(rotations):
            t = t0.copy() if i == 0 else np.median(observed, axis=0) - mesh.centroid @ r.T
            try:
                fit = align_observed_points(sampled, observed, r, t)
                # Positivity is an arithmetic admission, not a depth-quality filter.
                a = vertices @ r.T + t; b = vertices @ fit.rotation.T + fit.translation
                if np.any(a[:, 2] <= 1e-4) or np.any(b[:, 2] <= 1e-4): continue
                initial.append(a); fitted.append(b); fits.append((r, t, fit)); slots.append(slot)
            except ValueError: continue
        require(initial, 'No finite RGB/depth pose hypothesis at observed frame')
        # Batch is mathematically identical to existing raster kernels; bound
        # to four meshes at once, not 50*full-resolution raster memory.
        ious = []
        for start in range(0, len(initial), 2):
            batch = np.stack(initial[start:start+2] + fitted[start:start+2])
            rendered, _ = raster_camera_mesh_batch(batch, faces, K, p['width'], p['height'])
            a = [silhouette_iou(x.cpu().numpy(), m) for x in rendered]; mid = len(batch)//2
            ious.extend(zip(a[:mid], a[mid:]))
        candidates = []
        for slot, (r, t, fit), (a, b) in zip(slots, fits, ious):
            accepted = b >= a and fit.final_residual <= fit.initial_residual
            rr, tt = (fit.rotation, fit.translation) if accepted else (r, t)
            row = dict(hypothesis_index=slot, rotation=rr.tolist(), translation=tt.tolist(),
                initial_silhouette_iou=a, fitted_silhouette_iou=b,
                selected_silhouette_iou=b if accepted else a,
                selected_depth_residual_m=fit.final_residual if accepted else fit.initial_residual)
            if _finite_pose_candidate(row): candidates.append(row)
        require(candidates, 'Observed pose has no admissible hypothesis')
        best = max(candidates, key=lambda q: (q['selected_silhouette_iou'], -q['selected_depth_residual_m'], -q['hypothesis_index']))
        previous = np.asarray(best['rotation']); records.append(dict(frame_index=i, candidates=candidates, pose_observed=True))
    r, t, observed, details = _select_latent_pose_path(records, list(range(96)), 25, np.asarray(mesh.centroid))
    return r.astype(np.float32), t.astype(np.float32), observed, details


def native_layer(report):
    import torch
    import body_smoke as body
    native = ROOT / 'vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d'
    body._pinned_checkout(ROOT / 'vendor/video_to_data', body.UPSTREAM_REVISION)
    sys.path[:0] = [str(native), '/workspace/v2d_sam3d_body/lib']
    assets, report['body_assets'] = body._body_assets(ROOT)
    report['inference_source_identity'] = body._source_identity(ROOT)
    from lib_mhr.mhr_layer import MHRLayer
    from cari96_forward import SOURCES
    require(sha(native / 'lib_mhr/mhr_layer.py') == SOURCES['lib_mhr/mhr_layer.py'], 'Pinned native decoder source differs')
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),
        checkpoint_path=assets / 'model.ckpt', buffer_path=Path('/tmp/never_use_unverified_buffer.pt'),
        mhr_model_path=assets / 'assets/mhr_model.pt', device='cuda')
    report['decoder_identity'] = layer.decoder_identity()
    return layer, native


def lossless_prefix(p, path):
    """Native ABI requires one96frame MP4; verify pixel identity after lossless encode."""
    import cv2
    tmp = path.with_name('.' + path.stem + '.part.mp4')
    writer = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f"{p['width']}x{p['height']}", '-r', str(p['fps']), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'ultrafast', '-threads', '2', str(tmp)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    hashes = []
    try:
        for _, rgb in frames(p):
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest()); writer.stdin.write(rgb.tobytes())
        writer.stdin.close(); require(writer.wait(timeout=120) == 0, 'Lossless native prefix export failed')
    except BaseException:
        writer.kill(); writer.wait(); tmp.unlink(missing_ok=True); raise
    cap = cv2.VideoCapture(str(tmp))
    try:
        require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 96, 'Native prefix frame count differs')
        for digest in hashes:
            ok, bgr = cap.read(); require(ok and hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == digest,
                'Native prefix lossy RGB mismatch')
        require(not cap.read()[0], 'Native prefix includes extra frames')
    finally: cap.release()
    require(not path.exists(), 'No native RGB overwrite'); tmp.rename(path); path.chmod(0o444)


def prepare(p, c, base, out, report):
    import torch
    import h5py
    import joblib
    import pickle
    import trimesh
    from scipy.spatial.transform import Rotation
    from cari_body_adapter import candidate_mhr_parameters
    read_stage(base, 'body_depth'); read_stage(base, 'object'); read_stage(base, 'track')
    gauge = strict((base / 'body_depth/gauge.json').read_bytes()); K = np.asarray(gauge['K']); scale = gauge['alignment']['shared_scale']
    layer, native = native_layer(report)
    with np.load(base / 'body_depth/body.npz', allow_pickle=False) as z: body = {k: z[k].copy() for k in z.files}
    original = candidate_mhr_parameters(body, K, (p['height'], p['width']), total_frames=96, mhr_layer=layer, decode_batch_size=16)
    params = share_first_frame_identity({k: original[k] for k in NATIVE_PARAMETER_DIMS}, 96)
    metadata = shared_initializer_metadata(original['metadata'], 96)
    joints, keypoints = [], []
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for start in range(0, 96, 16):
            decoded = layer.mhr_forward({k: torch.tensor(v[start:start+16].copy(), device='cuda') for k, v in params.items()})
            joints.append(decoded.joints.cpu().numpy()); keypoints.append(decoded.keypoints.cpu().numpy())
    initializer = dict(body_model='mhr', **params, frames=[f'{i:06d}' for i in range(96)], kids=[0],
        mhr_joints=np.concatenate(joints), mhr_keypoints=np.concatenate(keypoints),
        metadata=dict(metadata, mhr_geometry_forward_verified=True, original_geometry_recovered=False,
            shared_identity_geometry_redecoded=True, joint_keypoint_redecode_required=False,
            public_dataset=p['dataset'], public_sequence_id=p['sequence_id']))
    seal(out / 'shared_initializer.pkl', lambda f: pickle.dump(initializer, f, protocol=4))
    del layer; gc.collect(); torch.cuda.empty_cache()
    obj = trimesh.load(base / 'object/object.glb', force='mesh', process=False)
    transform = strict((base / 'object/transform.json').read_bytes())
    s = np.asarray(transform['scale'], float); vertices = np.asarray(obj.vertices, float) * s[None]
    faces_array = np.asarray(obj.faces, np.int64)
    # Bake the native generator's entire fixed scale vector once. No anisotropic
    # averaging/shrinking, no deletion or official-padding budget for external eval.
    require(np.isfinite(vertices).all() and len(faces_array) and np.all(faces_array >= 0) and
        faces_array.max() < len(vertices), 'Whole fixed generated geometry must remain finite')
    q = np.asarray(transform['rotation']); require(q.shape == (4,) and np.isclose(np.linalg.norm(q), 1, atol=1e-5), 'Native unit quaternion required')
    R0 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); t0 = np.asarray(transform['translation'], float)
    r, t, observed, path = pose_initializer(vertices, faces_array, scale, R0, t0, p, base, K)
    save_npz(out / 'object_prior.npz', vertices=vertices.astype(np.float32), faces=faces_array,
        rotation=r, translation=t, observed=observed, frame_index=np.arange(96, dtype=np.int64))
    metric = out / 'object_metric.glb'; trimesh.Trimesh(vertices, faces_array, process=False).export(metric); metric.chmod(0o444)
    from prep.prepare_mhr_wild_export import prepare_mhr_wild_export
    from prep.mhr_depth_h5 import DepthFrameRecord, MHRDepthH5Writer, validate_depth_h5
    from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
    from prep.mhr_export_utils import MHR_CAMERA_NAMES
    sequence = p['sequence_id']; alias = out / (sequence + '.0.color.mp4'); lossless_prefix(p, alias)
    calibration = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], H=p['height'], W=p['width'],
        depth_backend='moge2', model_id=MOGE2_MODEL_ID, model_revision=MOGE2_MODEL_REVISION,
        source_commit=MOGE2_SOURCE_COMMIT, camera_policy='RGB_size_prior_K_no_reference_calibration')
    joblib.dump(calibration, out / 'intrinsics.pkl'); (out / 'intrinsics.pkl').chmod(0o444)
    with h5py.File(out / 'automatic_masks.h5', 'x') as h:
        for i in range(96):
            for role, suffix in ((0, 'person_mask.png'), (1, 'obj_rend_mask.png')):
                h.create_dataset(f'{sequence}/{i:06d}-k0.{suffix}', data=mask(base, role, i, p).astype('uint8')*255, compression='lzf')
    (out / 'automatic_masks.h5').chmod(0o444)
    export = prepare_mhr_wild_export(alias, out / 'automatic_masks.h5', metric, out / 'intrinsics.pkl', out / 'export')
    wild = strict((export / 'wild_export.json').read_bytes()); A = np.asarray(wild['source_object_mesh_to_aligned_transform'], float)
    require(A.shape == (4, 4) and np.allclose(A[3], [0, 0, 0, 1]) and
        np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(A[:3, :3]), 1, atol=1e-5),
        'Native template preparation must be only a proper rigid frame change')
    from cari_refine import verify_aligned_geometry
    aligned = trimesh.load(export / 'object_mesh/output_aligned.glb', force='mesh', process=False)
    report['aligned_object_geometry'] = verify_aligned_geometry(vertices, faces_array,
        np.asarray(aligned.vertices), np.asarray(aligned.faces), A)
    poses = np.broadcast_to(np.eye(4), (96, 4, 4)).copy(); poses[:, :3, :3] = r; poses[:, :3, 3] = t
    aligned_poses = poses @ np.linalg.inv(A)
    joblib.dump(dict(frames=initializer['frames'], obj_pose_world=aligned_poses.astype(np.float32),
        metadata=dict(source='automatic_RGB_predicted_depth_ICP_Viterbi', ground_truth_used=False,
            private_truth_read=False, hand_labeled_test=False, oracle_modes=[])), out / 'own_object_poses.pkl')
    (out / 'own_object_poses.pkl').chmod(0o444)
    names = initializer['frames']; camera = MHR_CAMERA_NAMES[0]
    depth_identity = dict(depth_backend='moge2', depth_model_id=MOGE2_MODEL_ID, depth_model_revision=MOGE2_MODEL_REVISION,
        depth_source_commit=MOGE2_SOURCE_COMMIT, ground_truth_used=False,
        alignment='one_predicted_human_anchored_clip_scalar_no_offset')
    with MHRDepthH5Writer(out / 'aligned_depth.h5', {camera: names},
            alignment_method='world_reward_shared_predicted_human_scale', alignment_input_identity=depth_identity, encoding_workers=4) as writer:
        for start in range(0, 96, 8):
            rows = []
            for i in range(start, start+8):
                with np.load(base / 'body_depth/depth' / f'{i:06d}.npz', allow_pickle=False) as z:
                    raw = np.where(z['mask'], z['depth'], 0.); aligned = raw * scale
                    require(np.isfinite(aligned).all() and (raw >= 0).all() and (aligned >= 0).all() and
                        (raw <= 65.535).all() and (aligned <= 65.535).all(), 'Depth encoding saturation is forbidden')
                    rows.append(DepthFrameRecord(i, raw, aligned, scale, 0., int(z['mask'].sum())))
            writer.write_frames(camera, rows)
        writer.mark_complete()
    validate_depth_h5(out / 'aligned_depth.h5', expected_cameras=[camera], expected_alignment_input_identity=depth_identity, validation_workers=4)
    for f in out.rglob('*'):
        if f.is_file(): f.chmod(0o444)
    report.update(native_export=str(export),
        object_pose_path=path, public_sequence_id=sequence, source_object_transform=A.tolist(),
        object_scale_baked_once=True, all_original_object_faces_preserved=True)


def RGB_evidence(p, c, base, out, bundle, vertices, faces_array, report):
    """Native CPU DWPose and FB KLT, attached once to the predicted whole mesh."""
    import cv2
    import dwpose_smoke as dw
    from sequence_pose_probe import lk_step
    from world_reward.point_surface_queries import _ray_triangle_hits
    from cari_converter import validate_native_bundle
    params, poses = validate_native_bundle(bundle, 96)
    size = (c['KLT']['width'], c['KLT']['height']); scaling = np.array(size) / [p['width'], p['height']]
    K = fixed_K(p); kk = K.copy(); kk[0] *= scaling[0]; kk[1] *= scaling[1]; kk[:2, 2] -= .5
    points = np.empty((0, 3), np.float64); tracks = np.empty((96, 0, 2), np.float64); support = np.empty((96, 0), bool)
    seq = frames(p); _, rgb = next(seq); gray = cv2.cvtColor(cv2.resize(rgb, size), cv2.COLOR_RGB2GRAY)
    m = cv2.resize(mask(base, 1, 0, p).astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    queries = cv2.goodFeaturesToTrack(gray, maxCorners=c['KLT']['max_corners'], qualityLevel=c['KLT']['quality_level'],
        minDistance=c['KLT']['min_distance'], mask=m)
    reason = 'no_initial_object_texture'
    if queries is not None:
        queries = queries.reshape(-1, 2).astype(np.float32)
        original = (queries + .5) / scaling - .5; grid = np.c_[original[:, 1], original[:, 0]]
        try:
            _, _, ids, bary = _ray_triangle_hits(vertices.astype(np.float64), faces_array,
                poses[0, :3, :3].astype(np.float64), poses[0, :3, 3].astype(np.float64), K,
                grid, np.ones(len(grid), bool))
            hit = ids >= 0
            points = (vertices[faces_array[ids[hit]]] * bary[hit, :, None]).sum(1); queries = queries[hit]
            reason = 'all_finite_first_hit_attachments_no_future_quality_selection'
        except ValueError:
            # Uncertain ray geometry never becomes a nearest-depth fake attachment.
            queries = np.empty((0, 2), np.float32); reason = 'ambiguous_ray_geometry_native_silhouette_fallback'
        tracks = np.full((96, len(points), 2), np.nan); support = np.zeros((96, len(points)), bool)
        if len(points): tracks[0] = queries; support[0] = True
        active = np.arange(len(points)); current = queries.copy(); previous = gray
        for i, rgb in seq:
            gray = cv2.cvtColor(cv2.resize(rgb, size), cv2.COLOR_RGB2GRAY)
            if len(active):
                q, good = lk_step(cv2, previous, gray, current)
                small = cv2.resize(mask(base, 1, i, p).astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
                ids = np.flatnonzero(good); rounded = np.floor(q[ids]+.5).astype(int)
                inside = (rounded[:, 0] >= 0) & (rounded[:, 0] < size[0]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < size[1])
                good[ids] = False; ids, rounded = ids[inside], rounded[inside]
                good[ids] = small[rounded[:, 1], rounded[:, 0]] > 0
                active, current = active[good], q[good]; tracks[i, active] = current; support[i, active] = True
            previous = gray
    seq.close()
    # Unobserved-but-valid attachments remain in the bank; the generic objective
    # gives missing observations zero weight, never deletes the object or a track.
    raw_xy = np.full((96, 133, 2), np.nan); scores = np.zeros((96, 133), np.float64)
    report['DWPose_assets'] = dw.validate_assets(ROOT)
    with dw.private_prefix(report, lambda: None) as prefix:
        subprocess.run(dw.pip_argv(prefix, ROOT), check=True, timeout=120, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ort, report['DWPose_runtime'] = dw.import_runtime(prefix)
        modspec = util.spec_from_file_location('world_reward_FORM_native_DWPose', ROOT / dw.acquisition.BASE / 'source/onnxpose.py')
        predictor = util.module_from_spec(modspec); modspec.loader.exec_module(predictor)
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(ROOT / dw.acquisition.BASE / dw.acquisition.ASSETS[0][0]),
            sess_options=options, providers=['CPUExecutionProvider']); session.disable_fallback()
        dw.validate_session(session); dw.validate_options(session, ort)
        for i, rgb in frames(p):
            ys, xs = np.nonzero(mask(base, 0, i, p))
            if not len(xs): continue
            boxes = np.array([[xs.min(), ys.min(), xs.max()+1, ys.max()+1]], np.float32)
            xy, confidence = predictor.inference_pose(session, boxes, rgb)
            xy, confidence = np.asarray(xy), np.asarray(confidence)
            require(xy.shape == (1, 133, 2) and confidence.shape == (1, 133) and
                np.isfinite(confidence).all() and np.isfinite(xy[confidence > 0]).all(), 'Literal native DWPose133 output required')
            raw_xy[i], scores[i] = xy[0], confidence[0]
    xy = raw_xy * scaling - .5
    save_npz(out / 'RGB_evidence.npz', points=points, object_xy=tracks, object_supported=support,
        human_xy=xy, human_scores=scores, raw_human_xy=raw_xy, K=kk, frame_index=np.arange(96, dtype=np.int64))
    report.update(material_track_count=len(points), material_evidence=reason, textureless_fallback='native_silhouette',
        material_attachment_frame=0, material_attachment_geometry='actual_native_mesh_and_raw_CoCoNet_pose',
        material_reacquisition=False, human_detector_observations=int((scores > 0).sum()))


def forward(p, c, base, out, report):
    import torch
    import joblib
    import cari96_forward as assets
    import cari96_native as constrained
    import cari_full_refine as full
    prepared, previous = read_stage(base, 'prepare')
    native, checkpoint, home, repository, source_id, body_assets, files = assets.asset_bindings(ROOT, previous)
    sys.path[:0] = [str(native), '/workspace/v2d_sam3d_body/lib']
    os.environ.update(MHR_ASSETS_ROOT=str(ROOT / 'weights/cari4d/sam3d_body'), MOMENTUM_ENABLED='0', TORCH_HOME=str(home))
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    report.update(hub_attempts=0, hub_returns=0)
    original = assets.local_dino_hub(torch, repository, report)
    try:
        from tools import run_mhr_wild_inference as module
        cloned = constrained.clone_native_forward(torch, module.run_mhr_wild_inference, report,
            source_bytes=(native / constrained.NATIVE_RELATIVE_PATH).read_bytes())
        export = Path(previous['native_export'])
        result = cloned(export, prepared / 'aligned_depth.h5', prepared / 'shared_initializer.pkl',
            prepared / 'own_object_poses.pkl', native / assets.CONFIG_RELATIVE_PATH, checkpoint, out / 'coconet.pth',
            stride=96, render_batch_size=32, crop_workers=4, crop_buffer_count=2, input_cache=None,
            use_input_cache=False, device_name='cuda', overwrite=False, wandb_run_path=None, offline_supervision_contract=True)
        require(Path(result) == out / 'coconet.pth', 'Native return path differs')
    finally: torch.hub.load = original
    bundle = torch.load(result, map_location='cpu', weights_only=False)
    mesh = export / 'object_mesh/output_aligned.glb'; full.validate_source_bundle(bundle, mesh, 96)
    require(bundle['gt'] == {} and bundle['metadata']['ground_truth_used'] is False,
        'Native offline supervision contract must explicitly disable GT')
    from learning.training import mhr_opt_refineout as optimizer
    vertices, faces_array = optimizer._load_object_vertices(mesh)
    require(faces_array is not None, 'Complete actual object triangle mesh required')
    (out / 'coconet.pth').chmod(0o444)
    report.update(body_assets=body_assets, inference_source_identity=source_id, checkpoint_sha256=sha(checkpoint),
        decoder_identity=previous['decoder_identity'], actual_network_forward=True, native_mesh=str(mesh),
        native_object_vertices=len(vertices), native_object_faces=len(faces_array), actual_native_source_bindings=files)
    # The shared bank is created once, before either final solver runs.
    RGB_evidence(p, c, base, out, bundle, vertices, faces_array, report)


def validate_solver_result(source_bundle, result, cfg, count, *, enabled):
    """Shared same300 output gate; only declared degrees of freedom may move."""
    import cari_full_refine as full
    from cari_converter import validate_native_bundle
    require(type(enabled) is bool, 'Explicit native control/candidate mode required')
    if not enabled:
        full.validate_result(source_bundle, result, count)
    old, oldposes = validate_native_bundle(source_bundle, count)
    params, poses = validate_native_bundle(result, count)
    require(set(result) == set(source_bundle) | {'postopt'}, 'Only native postopt metadata may be added')
    for key in set(source_bundle) - {'pr'}:
        require(full.fingerprint(source_bundle[key]) == full.fingerprint(result[key]),
            'Native solver changed a frozen raw field: ' + key)
    mutable = {'mhr_body_pose_cont', 'pose_abs'} | ({'mhr_trans'} if enabled else set())
    require(set(result['pr']) == set(source_bundle['pr']) | {'pose_abs_postopt'},
        'Only native final object-pose copy may be added')
    for key in set(source_bundle['pr']) - mutable:
        require(full.fingerprint(source_bundle['pr'][key]) == full.fingerprint(result['pr'][key]),
            'Native solver changed a frozen prediction including contacts: ' + key)
    require(full.fingerprint(result['pr']['pose_abs_postopt']) == full.fingerprint(result['pr']['pose_abs']) and
        old['mhr_body_pose_cont'][:, 254:].tobytes() == params['mhr_body_pose_cont'][:, 254:].tobytes() and
        oldposes[:, :3, :3].tobytes() == poses[:, :3, :3].tobytes(),
        'Native fixed object rotation/internal translations or final pose copies changed')
    post = result['postopt']; history = post['history']
    fixed = list(full.contract.FIXED_PARAMETERS); optimized = list(full.contract.OPTIMIZED_PARAMETERS)
    if enabled: fixed.remove('mhr_trans'); optimized.append('mhr_trans')
    require(post.get('config') == asdict(cfg) and post.get('frame_indices') == list(range(count)) and
        post.get('resolved_batch_size') == count and post.get('batch_sampling') == 'full_clip_v1' and
        post.get('fixed_parameters') == fixed and post.get('optimized_parameters') == optimized and
        post.get('mode') == ('native_joint_image_translation_extension_v1' if enabled else 'smplh_parity'),
        'Complete same300 native full-T solver protocol required')
    require(type(history) is list and len(history) == 4 and
        [x.get('iter') for x in history] == [0., 100., 200., 300.] and
        all(type(x.get('iter')) is float and x.get('batch_start') == 0. and
            x.get('batch_size') == float(count) for x in history) and
        all(type(v) in (int, float) and np.isfinite(v) for x in history for v in x.values()),
        'All finite native301 updates and full-T report rows required')
    diagnostics = post.get('final_diagnostics')
    require(type(diagnostics) is dict and bool(diagnostics) and
        all(type(v) in (int, float) and np.isfinite(v) for v in diagnostics.values()),
        'Complete finite actual final diagnostics required')
    return params, poses


def direct_geometry(p, out, layer, params, poses, vertices, faces_array):
    """Seal full native trajectories without Track1Episode or kit/oracle fitting."""
    import torch
    from cari_converter import require_rigid_transforms
    validate_native_parameters(params, 96, require_shared_identity=True)
    require_rigid_transforms(poses, 96)
    require(vertices.dtype == np.float32 and vertices.ndim == 2 and vertices.shape[1:] == (3,) and
        np.isfinite(vertices).all() and len(vertices) >= 3 and faces_array.dtype == np.int64 and
        faces_array.ndim == 2 and faces_array.shape[1:] == (3,) and len(faces_array) and
        faces_array.min() >= 0 and faces_array.max() < len(vertices), 'Actual native whole object geometry required')
    for r, t in zip(poses[:, :3, :3], poses[:, :3, 3]):
        require(np.all((vertices @ r.T + t)[:, 2] > 0), 'Every object frame must remain in front of the camera')
    target = np.empty((96, 18439, 3), np.float32); joints = np.empty((96, 127, 3), np.float32)
    keypoints = np.empty((96, 70, 3), np.float32); controls = np.empty((96, 204), np.float32); faces_human = None
    maximum = 0.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for start in range(0, 96, 16):
            sl = slice(start, start+16); t = {k: torch.tensor(v[sl].copy(), device='cuda', dtype=torch.float32) for k,v in params.items()}
            decoded = layer.mhr_forward(t); vv = decoded.vertices.cpu().numpy()
            jj = decoded.joints.cpu().numpy(); kk = decoded.keypoints.cpu().numpy()
            require(vv.shape == (16, 18439, 3) and jj.shape == (16, 127, 3) and kk.shape == (16, 70, 3) and
                all(np.isfinite(a).all() and (a[..., 2] > 0).all() for a in (vv, jj, kk)),
                'Full positive finite native human vertices/joints/keypoints required')
            target[sl] = vv; joints[sl] = jj; keypoints[sl] = kk
            ff = decoded.faces.cpu().numpy()
            require(ff.shape == (36874, 3) and ff.dtype.kind in 'iu' and ff.min() >= 0 and ff.max() < 18439,
                'Exact native human triangle topology required')
            if faces_human is not None: require(np.array_equal(ff, faces_human), 'Human topology changed')
            faces_human = ff.copy()
            context = layer.backend._vertices_context(t, detach_fixed=False)
            trans, body, shape, scale = layer.backend._mutable_vertices_inputs(t, context)
            direct, dc = context.head.mhr_forward(global_trans=trans*context.flip, global_rot=context.global_rot,
                body_pose_params=body, hand_pose_params=context.hand, scale_params=scale, shape_params=shape,
                expr_params=context.face, return_model_params=True)
            err = float(np.linalg.norm((direct*context.flip).cpu().numpy().astype(float)-vv.astype(float), axis=-1).max())
            require(err < 1e-5 and dc.shape == (16, 204) and torch.isfinite(dc).all(),
                'Direct204 finite native geometry parity failed'); maximum=max(maximum,err); controls[sl]=dc.cpu().numpy()
    require(all(x.tobytes() == controls[0,136:].tobytes() for x in controls[:,136:]), 'Clip-constant expanded scales required')
    seal(out / 'target.npy', lambda f: np.save(f, target, allow_pickle=False))
    save_npz(out / 'native_parameters.npz', **params, mhr_joints=joints, mhr_keypoints=keypoints,
        human_faces=faces_human, frame_index=np.arange(96, dtype=np.int64))
    save_npz(out / 'trajectory.npz', pose=controls[:,:136], scales=controls[0,136:], shape=params['mhr_shape'][0],
        expression=params['mhr_face'][0], object_vertices=vertices, object_faces=faces_array,
        object_rotation=poses[:,:3,:3], object_translation=poses[:,:3,3], object_scale=np.asarray(1.,np.float32),
        camera_K=fixed_K(p), frame_index=np.arange(96,dtype=np.int64), fps=np.asarray(p['fps']))
    save_npz(out / 'eval_geometry.npz', human_vertices=target, human_joints=joints, human_faces=faces_human,
        object_vertices=vertices, object_faces=faces_array, object_rotation=poses[:, :3, :3],
        object_translation=poses[:, :3, 3], object_scale=np.asarray(1., np.float32),
        frame_index=np.arange(96, dtype=np.int64))
    return maximum


def fit(p, c, base, out, report, enabled):
    import torch
    import cari_full_refine as full
    from world_reward.native_joint_refinement import JointImageEvidence, NativeJointConfig, native_joint_optimizer_class
    from cari_converter import validate_native_bundle
    directory, fr = read_stage(base, 'forward')
    layer, native = native_layer(report)
    from learning.training import mhr_opt_refineout as optimizer
    require(sha(native / full.contract.OPTIMIZER_RELATIVE_PATH) == full.contract.OPTIMIZER_SHA256, 'Exact native optimizer required')
    full.contract.require_asset_receipt(strict((ROOT / 'results/cari-refinement-assets.json').read_bytes()))
    for name, pin in full.contract.REFINEMENT_ASSETS.items():
        path=ROOT/'weights/cari4d/refinement'/name
        require(dict(bytes=path.stat().st_size,sha256=sha(path)) == {k:pin[k] for k in ('bytes','sha256')}, 'Native collision/contact asset changed')
    source_bundle=torch.load(directory/'coconet.pth',map_location='cpu',weights_only=False)
    mesh=Path(fr['native_mesh']); full.validate_source_bundle(source_bundle,mesh,96)
    vertices,faces_array=optimizer._load_object_vertices(mesh)
    with np.load(directory/'RGB_evidence.npz',allow_pickle=False) as z:
        evidence=JointImageEvidence(z['points'],z['object_xy'],z['object_supported'],z['human_xy'],z['human_scores'],
            z['K'],z['frame_index'],allow_empty_object_evidence=True)
    cfg=optimizer.MHRParityPostOptConfig(num_steps=300,batch_size=0,
        penetration_collision_proxy_path=str(ROOT/'weights/cari4d/refinement/mhr_collision_proxy_4000v.npz'),
        hand_surface_spec_path=str(ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'),report_every=100)
    extension=NativeJointConfig(**c['extension'])
    torch.use_deterministic_algorithms(False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
    before=full.fingerprint(source_bundle); cls=native_joint_optimizer_class(optimizer,evidence,extension,enabled=enabled)
    instance=cls(source_bundle,vertices,faces_array,cfg,mhr_layer=layer)
    started=time.monotonic(); result=instance.run(); report['solver_seconds']=time.monotonic()-started
    require(full.fingerprint(source_bundle)==before, 'Original raw bundle was changed')
    params,poses=validate_solver_result(source_bundle,result,cfg,96,enabled=enabled)
    history=result['postopt']['history']
    seal(out/'refined.pth',lambda f:torch.save(result,f,pickle_protocol=4))
    del instance,result;gc.collect();torch.cuda.empty_cache()
    error=direct_geometry(p,out,layer,params,poses,vertices,faces_array)
    report.update(native_history=history,native_config=asdict(cfg),extension=asdict(extension) if enabled else None,
        requested_steps=300,effective_updates=301,raw_bundle_pin=artifact(directory/'coconet.pth'),
        direct_native_max_error_m=error,full_predictions_sealed_before_evaluation=True,
        geometry_pin=artifact(out/'eval_geometry.npz'),
        mode='native_joint_image_translation_extension_v1' if enabled else 'same_frontend_native_parity_control',
        material_track_count=len(evidence.points), heldout_accuracy_verified=False)


def worker(args):
    code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION'];c=config(code)
    require(sys.platform=='linux' and Path(os.environ['WR_ROOT'])==ROOT,'Azure Linux runtime only')
    binding=source(ROOT,code,revision,ENTRY,HELPERS)
    p=load_public(args.input,dict(bytes=args.input_bytes,sha256=args.input_sha256),code)
    base=canonical(args.out); require(base.name==p['sequence_id'],'Actual FORM sequence namespace required')
    stage=args.stage;out=base/stage;require(not out.exists(),'Frozen stage exists; do not overwrite/rerun selected results')
    out.mkdir(mode=0o755); report=dict(schema='world_reward.form_external_prediction_stage.v1',status='fail',stage=stage,
        producer_revision=revision,source_binding=binding,sequence_id=p['sequence_id'],dataset=p['dataset'],
        input_pin=dict(bytes=args.input_bytes,sha256=args.input_sha256),video_pin=p['video_pin'],
        original_frame_indices=p['original_frame_indices'],full_source_frames=p['full_source_frames'],
        ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],
        reference_inputs_mounted=False,training_overlap_verified=False,production_adopted=False,
        scope='first96_original_RGB_only_external_DEV_not_test_or_verified_CARI_victory')
    started=time.monotonic()
    def expired(*_):raise TimeoutError('Frozen inclusive stage budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(c['budgets'][stage])
    try:
        if stage!='localize':
            from form_prediction_reuse import localization_lineage
            report['localization_source']=localization_lineage(sys.modules[__name__],base,revision)
            require({p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline GPU inference required')
            import torch
            require(torch.cuda.is_available(),'GPU required; never use laptop/CPU model fallback')
        if stage=='localize':localize(p,c,base,out,report)
        elif stage=='track':track(p,c,base,out,report)
        elif stage=='body_depth':body_depth(p,c,base,out,report)
        elif stage=='object':object_mesh(p,c,base,out,report)
        elif stage=='prepare':prepare(p,c,base,out,report)
        elif stage=='forward':forward(p,c,base,out,report)
        else:fit(p,c,base,out,report,enabled=stage=='fit_B')
        require(load_public(args.input,report['input_pin'],code)==p and source(ROOT,code,revision,ENTRY,HELPERS)==binding,
            'Original public inputs/code changed during execution')
        for f in out.rglob('*'):
            if f.is_file():f.chmod(0o444)
        report.update(status='complete',artifacts={str(f.relative_to(out)):artifact(f) for f in sorted(out.rglob('*')) if f.is_file()},
            elapsed_seconds=time.monotonic()-started, source_rehashed_after=True)
    except Exception as e:
        # Error messages can contain request/network context; only a typed failure
        # is public. Raw technical diagnostics stay on Azure, never token values.
        report.update(error_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
        if isinstance(e, ValueError): report['error_context'] = str(e)[:400]
        if stage != 'localize':
            # Offline workers have no credentials or reference mounts. Keep a
            # bounded real traceback on Azure so technical failures are fixable;
            # never print it or treat a failed stage as a successful prediction.
            seal(out/'technical_traceback.txt', lambda f: f.write(traceback.format_exc().encode()[-16384:]))
        raise
    finally:
        signal.alarm(0);save_json(out/'report.json',report)
    print(json.dumps(dict(sequence_id=p['sequence_id'],stage=stage,status='complete',seconds=round(report['elapsed_seconds'],2))),flush=True)


def mount_sources(stage,c,code,public_dir,base):
    """File/directory allowlist: never mount a dataset parent or eval_private."""
    rows=[(code.parent,True),(public_dir,True),(base,False)]
    body_sources=(ROOT/'vendor/video_to_data', ROOT/'weights/cari4d/sam3d_body',
        ROOT/'results/weights-acquisition.json',ROOT/'results/auxiliary-assets.json')
    if stage in ('body_depth','prepare','forward','fit_A','fit_B'):rows += [(p,True) for p in body_sources]
    if stage=='body_depth':rows.append((ROOT/'weights/cari4d/hf_home/hub',True))
    if stage=='object':rows += [(ROOT/'weights/sam3d',True),(ROOT/'results/weights-acquisition.json',True),
        (ROOT/'results/auxiliary-assets.json',True)]
    if stage in ('forward','fit_A','fit_B'):
        rows += [(ROOT/'weights/cari4d/refinement',True),(ROOT/'results/cari-refinement-assets.json',True),
            (ROOT/'weights/cari4d/cari4d',True)]
    if stage=='forward':
        rows += [(ROOT/'weights/dwpose_native_v1',True),(ROOT/'results/dwpose-acquisition-v1.json',True),
            (ROOT/'results/dwpose-wheel-audit-v2',True),(ROOT/'results/dwpose-wheel-audit-v3',True)]
    if stage=='track':
        receipt=Path(c['sam_prepare_receipt']);prep=strict(receipt.read_bytes())
        require(artifact(receipt)==c['sam_prepare_pin'],'Qualified SAM receipt differs')
        rows += [(receipt,True),(Path(prep['weight_file']),True)]
    for path, _ in rows:
        require(not {'eval_private','gt','track_1'}.intersection(path.parts), 'No reference/challenge inputs in external inference mount list')
    return list(dict.fromkeys(rows))


def credential_paths(revision, sequence_id):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}',revision) and
        type(sequence_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,128}',sequence_id),
        'Exact producer/sequence credential ownership required')
    prefix=ROOT/'.secrets'/('form-hoi-external-'+revision+'-'+sequence_id)
    return prefix.with_name(prefix.name+'-key.pem'),prefix.with_name(prefix.name+'-envelope.enc')


def driver(args):
    import fcntl
    code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION'];c=config(code)
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01', 'Azure VM01 root driver only')
    binding=source(ROOT,code,revision,ENTRY,HELPERS)
    p=load_public(args.input,dict(bytes=args.input_bytes,sha256=args.input_sha256),code)
    base=canonical(args.out)
    require(base==ROOT/'results'/('form-hoi-external-predict-'+revision)/p['sequence_id'], 'Own exact output namespace required')
    if not base.exists():base.mkdir(mode=0o755,parents=True)
    chosen=STAGES if args.stage=='all' else (args.stage,)
    with (ROOT/'jobs/.world-reward-h100.lock').open('r') as lease:
        if any(s!='localize' for s in chosen):fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in chosen:
            if (base/stage/'report.json').is_file():
                read_stage(base,stage);continue  # Resume consumes the same sealed result, never reruns.
            require(not (base/stage).exists(),'Incomplete stage is preserved; require diagnosis/new producer, not scientific retry')
            image=c['object_image'] if stage=='object' else c['body_image']
            if stage=='track':image=strict(Path(c['sam_prepare_receipt']).read_bytes())['image_id']
            actual=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
            require(actual['Id']==image,'Exact previously qualified native image required')
            name=f'wr-form-{revision[:12]}-{p["sequence_id"][-24:]}-{stage}'
            require(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip(),'No duplicate owned job')
            cmd=['docker','run','--rm','--name',name,'--label','world_reward.form_predict.owner='+revision,
                '--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
                '--memory','64g','--cpus','8','--shm-size','2g','--tmpfs','/tmp:rw,nosuid,exec,size=4g',
                '--network','host' if stage=='localize' else 'none']
            if stage=='localize':cmd.append('-i')
            else:cmd+=['--gpus','all']
            for src,readonly in mount_sources(stage,c,code,args.input.parent,base):
                canonical(src);cmd+=['--mount',f'type=bind,src={src},dst={src}'+(',readonly' if readonly else '')]
            path='/usr/local/bin:/opt/conda/bin:/usr/bin:/bin' if stage=='track' else '/opt/conda/bin:/usr/local/bin:/usr/bin:/bin'
            python='/usr/local/bin/python' if stage=='track' else '/opt/conda/bin/python'
            pythonpath=('/opt/sam3:' if stage=='track' else '')+str(code/'src')+':'+str(code/'infra')+':/workspace/v2d_sam3d_body/lib'
            env=['PATH='+path,'HOME=/tmp','WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,
                'WR_IMAGE_ID='+image,'PYTHONPATH='+pythonpath,
                'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','MOMENTUM_ENABLED=0',
                'WANDB_MODE=disabled','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=1','MPLBACKEND=Agg',
                'XDG_CACHE_HOME=/tmp/cache','TORCH_HOME='+str(ROOT/('weights/sam3d/torch_home' if stage=='object' else 'weights/cari4d/sam3d_body/torch_home')),
                'HF_HOME='+str(ROOT/('weights/sam3d/hf_home' if stage=='object' else 'weights/cari4d/hf_home'))]
            if stage=='object':env+=['LIDRA_SKIP_INIT=1','CUDA_HOME=/usr/local/cuda']
            if stage=='localize':env.append('CUDA_VISIBLE_DEVICES=')
            env += ['CUBLAS_WORKSPACE_CONFIG=:4096:8']
            cmd+=['--entrypoint','/usr/bin/env',image,'-i',*env,python,'-B',str(code/'infra/form_hoi_external_predict.py'),
                '--native','--stage',stage,'--input',str(args.input),'--input-sha256',args.input_sha256,
                '--input-bytes',str(args.input_bytes),'--out',str(base)]
            token=None;private=[]
            try:
                if stage=='localize':
                    key,envelope=credential_paths(revision,p['sequence_id'])
                    private=[key,envelope]
                    for f in private:
                        canonical(f);s=f.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and not s.st_mode&0o077,
                            'Private one-shot auth file permissions required')
                    opened=subprocess.run(['openssl','pkeyutl','-decrypt','-inkey',str(key),'-in',str(envelope),
                        '-pkeyopt','rsa_padding_mode:oaep','-pkeyopt','rsa_oaep_md:sha256'],capture_output=True,timeout=15)
                    require(opened.returncode==0 and 0<len(opened.stdout)<16384,'One-shot RAM OAuth decrypt failed')
                    token=opened.stdout;opened=None
                with (base/(stage+'.log')).open('xb') as log:
                    os.fchmod(log.fileno(),0o400)
                    done=subprocess.run(cmd,input=token,stdout=log,stderr=log,timeout=c['budgets'][stage]+15)
                require(done.returncode==0,'Native stage failed; preserved Azure-only diagnostic, no silent omission')
                read_stage(base,stage)
            finally:
                token=None
                for f in private:
                    if f.exists():canonical(f);require(f.stat().st_nlink==1,'Owned auth cleanup only');f.unlink()
                found=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
                if found.returncode==0:
                    actual=json.loads(found.stdout)[0]
                    require(actual['Image']==image and actual['Config']['Labels'].get('world_reward.form_predict.owner')==revision,'Owned container cleanup only')
                    subprocess.run(['docker','rm','-f',actual['Id']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20,check=True)
            print(json.dumps(dict(stage=stage,sequence_id=p['sequence_id'],status='complete')),flush=True)
    require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Immutable source closure changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--native',action='store_true');parser.add_argument('--stage',choices=(*STAGES,'all'),default='all')
    parser.add_argument('--input',type=Path);parser.add_argument('--input-sha256')
    parser.add_argument('--input-bytes',type=int);parser.add_argument('--out',type=Path)
    parser.add_argument('--cohort-stage',choices=('localize','all'));parser.add_argument('--dev-revision')
    parser.add_argument('--reuse-localizations-from')
    args=parser.parse_args()
    if args.cohort_stage is not None:
        require(not args.native and args.stage=='all' and args.dev_revision is not None and
            all(getattr(args,k) is None for k in ('input','input_sha256','input_bytes','out')),
            'Cohort mode accepts only --cohort-stage and exact --dev-revision')
        code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION']
        require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01',
            'Azure VM01 root cohort dispatcher only')
        from form_hoi_external_cohort import run
        run(sys.modules[__name__],code,revision,stage=args.cohort_stage,dev_revision=args.dev_revision,
            reuse_localizations_from=args.reuse_localizations_from);return
    require(args.dev_revision is None and args.reuse_localizations_from is None and args.input is not None and args.out is not None and
        type(args.input_sha256) is str and re.fullmatch('[0-9a-f]{64}',args.input_sha256) and
        type(args.input_bytes) is int and args.input_bytes>0,'Explicit public artifact input/output SHA/bytes required')
    if args.native:
        require(args.stage!='all','One isolated native stage per worker');worker(args)
    else:driver(args)


if __name__=='__main__':
    try:main()
    except Exception as e:
        # Never print request text, credential data or raw network exception.
        print(json.dumps(dict(status='fail',error_type=type(e).__name__)),flush=True);sys.exit(1)

"""Azure-only one-pass, fixed-ID native SAM3.1 recovery from saved forward masks.

The old person/object masks are immutable inputs, not rerun or edited. Only an
originally empty object observation may use a literal reverse native mask, and
only when independent seed-linked RGB features and their affine silhouette
support it. True/unresolved occlusion remains an empty observation on the full
original timeline; a downstream latent 3D estimator, not a fake mask, handles it.

The global operating gates are provisional manufactured-contract gates. This
diagnostic does not claim external accuracy calibration or challenge victory.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, inputs
from sam31_runtime import control, decode_original, verify_runtime_source, verify_checkpoint_coverage, compatible_init
from gemini_sam31_track import prepared, prompt_parity, preview
from qwen4d_masks import _inventory
from world_reward.seeded_tracking import corner_prompt, temporal_summary

ENTRY = 'run_sam31_recovery'
CONFIG = 'configs/sam31_recovery_v1.json'
FORWARD_ENTRY = 'run_gemini_sam31_track'
FORWARD_FILES = {'report.json', 'prompts.json', 'mask-inventory.json', 'grounding.json',
                 'tracking.json', 'prefix-qualification.json', 'qa.jpg'}
HELPERS = ('infra/sam31_recovery.py', 'infra/run_sam31_recovery.sh', CONFIG,
           'infra/gemini_sam31_track.py', 'configs/gemini_sam31_v1.json',
           'infra/sam31_runtime.py', 'configs/sam31_runtime_v1.json', 'infra/Dockerfile.sam31',
           'configs/hybrid_pair_v1.json', 'infra/task_grounding_pilot.py', 'infra/qwen4d_masks.py',
           'src/world_reward/seeded_tracking.py', 'src/world_reward/occlusion_recovery.py')


def save(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode(), mode=0o444)


def settings(code):
    c = strict((code / CONFIG).read_bytes())
    runtime = strict((code / c['sam_runtime_config']).read_bytes())
    require(c['schema'] == 'world_reward.sam31_recovery.v1'
            and c['episodes'] == random.Random(20261008).sample(range(30), 4)
            and c['saved_forward_producer'] == 'b658881079871508c6b3ec14d001dc1299956996'
            and c['frames'] == [0, 14, 29] and c['ground_truth_used'] is False
            and c['manual_labels'] is False and c['baseline_modified'] is False
            and c['quality_verified'] is False and c['additional_reverse_passes_per_clip'] == 1
            and c['sam_capacity'] == 2 and c['sam_multiplex_count'] == 16
            and c['anchor_policy'] == 'last_saved_forward_visible_seed_rgb_qualified'
            and c['fusion_policy'] == 'preserve_forward_recover_empty_only_with_rgb_and_geometry'
            and c['native_state_policy'] == 'one_fresh_object_singleton_shared_backbone'
            and runtime['source_revision'] == '2345a4ad109ac29c569da749c91d84f10dc08c40'
            and runtime['model_revision'] == 'daa63191845a41281374e725f4c9e51c7a824460'
            and runtime['use_fa3'] is False and runtime['compile'] is False,
            'Frozen saved-forward one-reverse-pass contract required')
    # Host dispatch is stdlib-only; numerical policy construction belongs in
    # the pinned native image, not the Azure system Python without NumPy.
    gates = c['recovery_gates']
    require(set(gates) == {'agreement_iou', 'geometry_iou', 'identity_confidence',
                          'identity_support', 'calibration_source'}
            and all(type(gates[k]) in (int, float) and 0 < gates[k] <= 1
                    for k in ('agreement_iou', 'geometry_iou', 'identity_confidence', 'identity_support'))
            and type(gates['calibration_source']) is str and gates['calibration_source'].startswith('external:'),
            'Explicit external global recovery gates required')
    r = c['rgb_witness']
    require(r['anchor_candidates'] == 1 and r['qualification'] == 'manufactured_contracts_only_not_accuracy'
            and r['seed_patch_radius_pixels'] == 16
            and 6 <= r['minimum_affine_inliers'] <= r['seed_features'] <= 256
            and 0 < r['orb_ratio'] < 1 and 0 < r['fb_error_pixels'] <= 3
            and 0 < r['ransac_error_pixels'] <= 5 and 128 <= r['orb_features'] <= 2048,
            'Bounded global independent RGB witness required')
    return c, runtime


def _copy(src, dst, pin):
    require(identity(src, 16 << 20) == pin and not dst.exists(), 'Exact readonly saved mask required')
    with src.open('rb') as reader, dst.open('xb') as writer:
        os.fchmod(writer.fileno(), 0o444); shutil.copyfileobj(reader, writer, 1 << 20)
        writer.flush(); os.fsync(writer.fileno())
    require(identity(dst, 16 << 20) == pin and identity(src, 16 << 20) == pin, 'Mask continuity copy differs')


def read_mask(path, pin, height, width):
    import numpy as np
    from PIL import Image
    require(identity(path, 16 << 20) == pin, 'Saved native mask changed')
    with Image.open(path) as im:
        require(im.mode == 'L' and im.size == (width, height), 'Original binary mask grid required')
        a = np.asarray(im).copy()
    require(set(np.unique(a)).issubset({0, 255}), 'Exact native binary mask encoding required')
    return a > 0


def mask_box(mask):
    """Actual mask bounds on original exclusive XYXY grid; never expand/clamp."""
    import numpy as np
    require(type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.ndim == 2 and mask.any(),
            'Nonempty literal native mask required for an automatic anchor')
    yy, xx = np.nonzero(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def native_singleton(output, index, height, width, rgb_hash):
    import numpy as np
    from world_reward.occlusion_recovery import native_candidate
    require(type(output) in (list, tuple) and len(output) == 5 and output[0] == index and output[1] == [1],
            'Native reverse must preserve original frame and fixed singleton ID1')
    logits = output[3].detach().float().cpu().numpy()
    presence = output[4].detach().float().cpu().numpy()
    require(logits.shape == (1, 1, height, width) and presence.shape in ((1,), (1, 1))
            and np.isfinite(logits).all() and np.isfinite(presence).all(), 'Actual finite original-grid native logits required')
    row = native_candidate(frame_index=index, object_id=1, rgb_sha256=rgb_hash, branch='reverse',
        mask_logits=logits[0, 0], presence_logit=float(presence.reshape(-1)[0]))
    return row, dict(min=float(logits.min()), max=float(logits.max()),
                     positive_pixels=int((logits > 0).sum()), raw_presence_logit=float(presence.reshape(-1)[0]))


def saved_forward(code, cfg, selected):
    revision = cfg['saved_forward_producer']; base = ROOT / 'results' / ('gemini-sam31-' + revision)
    code_old = ROOT / 'jobs' / revision / FORWARD_ENTRY / 'code'
    require(identity(base / 'native-report.json', 200000) == cfg['saved_forward_native_report'],
            'Exact independently pinned saved forward native report required')
    host = strict((base / 'report.json').read_bytes()); aggregate = strict((base / 'native-report.json').read_bytes())
    require(type(host.get('source_binding', {}).get('helpers')) is dict,
            'Actual complete saved source helper ledger required')
    binding = source(ROOT, code_old, revision, FORWARD_ENTRY, tuple(host['source_binding']['helpers']))
    require(host['source_binding'] == binding and host['native_report'] == cfg['saved_forward_native_report']
            and aggregate['producer_revision'] == revision
            and aggregate['status'] == 'complete_diagnostic_not_quality_pass'
            and aggregate['ground_truth_used'] is False and aggregate['manual_labels'] is False
            and aggregate['original_frame_grid_preserved'] is True
            and [r['episode_index'] for r in aggregate['episodes']] == cfg['episodes'],
            'Complete original source-bound automatic forward masks required')
    bound = {base / 'report.json': identity(base / 'report.json'),
             base / 'native-report.json': cfg['saved_forward_native_report']}
    rows = {}
    for item, ar in zip(selected, aggregate['episodes']):
        ep = item['episode']; directory = base / f'episode_{ep:06d}' / 'automatic_masks'
        require(ar['status'] == 'pass' and ar['frames'] == item['total']
                and ar['automatic_masks'] == str(directory.relative_to(base))
                and {p.name for p in directory.iterdir()} == FORWARD_FILES | {'masks'},
                'Exact complete saved clip namespace required')
        pins = {name: identity(directory / name, 4 << 20) for name in FORWARD_FILES}
        require(pins['report.json'] == ar['report'], 'Saved clip report differs from independently pinned aggregate')
        report = strict((directory / 'report.json').read_bytes())
        tracking = strict((directory / 'tracking.json').read_bytes())
        grounding = strict((directory / 'grounding.json').read_bytes())
        records, inv, raw = _inventory(directory / 'masks', item['total'])
        require(inv == report['mask_inventory'] and raw == (directory / 'mask-inventory.json').read_bytes()
                and report['producer_revision'] == revision and report['input_sha256'] == item['video_pin']['sha256']
                and report['ground_truth_used'] is False and report['oracle_modes'] == []
                and report['fixed_native_ids'] == [0, 1] and report['full_original_grid'] is True
                and tracking['status'] == 'full_T_complete'
                and report['script_sha256'] == binding['helpers']['infra/gemini_sam31_track.py']['sha256']
                and report['source_revision'] == '2345a4ad109ac29c569da749c91d84f10dc08c40'
                and report['model_revision'] == 'daa63191845a41281374e725f4c9e51c7a824460'
                and tracking['original_frame_indices'] == list(range(item['total']))
                and tracking['interpolation'] is False and grounding['video_pin'] == item['video_pin']
                and grounding['ground_truth_used'] is False and grounding['hand_labeled_test'] is False
                and grounding['manual_points'] is False,
                'Original full-T, GT-disabled mask/input inventory required')
        require(len(tracking['records']) == item['total'] and set(tracking['areas']) == {'0', '1'},
                'Both full original streams required')
        for i, record in enumerate(tracking['records']):
            require(record['frame_index'] == i and type(record['native_presence']) is list
                    and len(record['native_presence']) == 2
                    and all(type(v) is bool for v in record['native_presence'])
                    and all(len(tracking['areas'][role]) == item['total'] for role in ('0', '1')),
                    'Literal ordered actual native sign records required')
        for name, pin in pins.items(): bound[directory / name] = pin
        for name, pin in records.items():
            require(identity(directory / 'masks' / name, 16 << 20) == pin, 'Readonly old native PNG required')
            bound[directory / 'masks' / name] = pin
        rows[ep] = dict(directory=directory, pins=pins, inventory=records, tracking=tracking,
                        grounding=grounding, report=report)
    return rows, bound, binding


class RgbWitness:
    """Bounded original-RGB KLT plus mutual descriptor seed re-identification.

    Only measured descriptor/KLT correspondences vote. An affine fit is a 2D
    identity/visibility witness, not a 3D motion model, and may abstain under
    large viewpoint change. Failed matches never become stationary features.
    """
    def __init__(self, frames, seed_mask, hashes, cfg, bound_source, relevant, budget):
        import cv2
        import numpy as np
        self.cv2 = cv2; self.np = np; self.frames = frames; self.seed_mask = seed_mask
        self.hashes = hashes; self.cfg = cfg; self.bound_source = bound_source; self.rows = []
        self.orb = cv2.ORB_create(nfeatures=cfg['orb_features'])
        gray = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_RGB2GRAY)
        # ORB's 31px descriptor samples may otherwise include an adjacent person
        # or static background even when the feature centre is inside the mask.
        # Require the entire descriptor patch to belong to the initial object;
        # thin/textureless targets abstain instead of borrowing scene identity.
        radius = cfg['seed_patch_radius_pixels']
        owned = cv2.erode(seed_mask.astype('uint8'), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        kp, descriptors = self.orb.detectAndCompute(gray, owned * 255)
        # Deterministic response order, bounded unique seed-object features.
        keep = sorted(range(len(kp)), key=lambda i: (-kp[i].response, kp[i].pt[1], kp[i].pt[0]))[:cfg['seed_features']]
        self.seed = np.asarray([kp[i].pt for i in keep], dtype=np.float64).reshape(-1, 2)
        self.descriptors = None if descriptors is None else descriptors[keep]
        n = len(self.seed); current = self.seed.copy(); active = np.ones(n, dtype=np.bool_)
        previous = gray; self.affines = []; self.match_counts = []
        for index, frame in enumerate(frames):
            budget(); gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
            if index and n and active.any():
                ids = np.flatnonzero(active); p = current[ids].astype('float32').reshape(-1, 1, 2)
                q, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, p, None, winSize=(21, 21), maxLevel=3)
                if q is None or status is None:
                    active[:] = False
                else:
                    back, bs, _ = cv2.calcOpticalFlowPyrLK(gray, previous, q, None, winSize=(21, 21), maxLevel=3)
                    good = np.zeros(len(ids), dtype=np.bool_) if back is None or bs is None else (
                        status.reshape(-1).astype(bool) & bs.reshape(-1).astype(bool)
                        & np.isfinite(q).all(axis=(1, 2)) & np.isfinite(back).all(axis=(1, 2))
                        & (np.linalg.norm(back[:, 0] - p[:, 0], axis=1) <= cfg['fb_error_pixels']))
                    active[:] = False; active[ids[good]] = True; current[ids[good]] = q[good, 0]
            if index in relevant and n:
                relink = self._reidentify(gray)
                # Direct seed appearance rejects KLT drift at all rescue/anchor frames.
                active[:] = False
                for seed_id, xy in relink:
                    active[seed_id] = True; current[seed_id] = xy
            h, w = gray.shape
            active &= (current[:, 0] >= 0) & (current[:, 0] < w) & (current[:, 1] >= 0) & (current[:, 1] < h)
            affine = None; inliers = np.zeros(n, dtype=np.bool_)
            if index == 0 and n:
                affine = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64); inliers = active.copy()
            elif int(active.sum()) >= cfg['minimum_affine_inliers']:
                ids = np.flatnonzero(active)
                fit, flags = cv2.estimateAffinePartial2D(self.seed[ids], current[ids], method=cv2.RANSAC,
                    ransacReprojThreshold=cfg['ransac_error_pixels'], maxIters=2000, confidence=.99, refineIters=10)
                if fit is not None and flags is not None and np.isfinite(fit).all():
                    accept = flags.reshape(-1).astype(bool)
                    if int(accept.sum()) >= cfg['minimum_affine_inliers'] and np.linalg.det(fit[:, :2]) > 0:
                        affine = fit.astype(np.float64); inliers[ids[accept]] = True
            self.rows.append((current.copy(), inliers.copy()))
            self.affines.append(affine); self.match_counts.append(int(inliers.sum())); previous = gray

    def _reidentify(self, gray):
        if self.descriptors is None: return []
        cv2 = self.cv2; kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(desc) < 2 or len(self.descriptors) < 2: return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ab = matcher.knnMatch(self.descriptors, desc, k=2)
        ba = matcher.knnMatch(desc, self.descriptors, k=2)
        def unambiguous(rows):
            return {pair[0].queryIdx: pair[0].trainIdx for pair in rows
                    if len(pair) == 2 and pair[0].distance < self.cfg['orb_ratio'] * pair[1].distance}
        f, b = unambiguous(ab), unambiguous(ba)
        return [(i, kp[j].pt) for i, j in sorted(f.items()) if b.get(j) == i]

    def evidence(self, index):
        from world_reward.occlusion_recovery import RgbIdentityEvidence
        affine = self.affines[index]
        geometry = None if affine is None else self.cv2.warpAffine(self.seed_mask.astype('uint8'), affine,
            (self.seed_mask.shape[1], self.seed_mask.shape[0]), flags=self.cv2.INTER_NEAREST,
            borderMode=self.cv2.BORDER_CONSTANT, borderValue=0).astype(self.np.bool_)
        xy, valid = self.rows[index]
        return RgbIdentityEvidence(frame_index=index, object_id=1, rgb_sha256=self.hashes[index],
            xy=xy.astype(self.np.float64), matched=valid.astype(self.np.bool_), visible=valid.astype(self.np.bool_),
            geometry_mask=geometry, source='automatic_rgb:' + self.bound_source, occlusion_confirmed=False)

    def persist(self, target):
        np = self.np
        with target.open('xb') as f:
            os.fchmod(f.fileno(), 0o444)
            np.savez_compressed(f, seed_xy=self.seed, xy=np.asarray([r[0] for r in self.rows]),
                matched=np.asarray([r[1] for r in self.rows]),
                affine=np.asarray([np.full((2, 3), np.nan) if a is None else a for a in self.affines]),
                frame_index=np.arange(len(self.rows), dtype=np.int64))
        return identity(target, 64 << 20)


def select_preserving_forward(forward, reverse, evidence, policy):
    """Only rescue actual old empty frames; never erase/replace old visibility."""
    from world_reward.occlusion_recovery import fuse_frame
    fused = fuse_frame(forward, reverse, evidence, policy)
    if forward.mask.any():
        return forward.mask, 'saved_forward_native', fused
    if fused.source == 'reverse_rgb_recovery':
        return fused.observed_mask, 'reverse_rgb_recovery', fused
    return forward.mask, 'unresolved_native_absence', fused


def reverse_anchor(areas, load, evidence, rgb_hashes, tracking_sha, policy):
    from world_reward.occlusion_recovery import fuse_frame, saved_native_candidate
    visible = [i for i, value in enumerate(areas) if value > 0]
    if not visible: return None
    index = visible[-1]  # One predeclared proposal; no picking around failed identity.
    forward = saved_native_candidate(frame_index=index, object_id=1, rgb_sha256=rgb_hashes[index],
        branch='forward', mask=load(index), native_presence=True, tracking_report_sha256=tracking_sha)
    qualified = fuse_frame(forward, None, evidence(index), policy)
    if qualified.state != 'observed': return None
    return dict(frame_index=index, object_id=1, rgb_sha256=rgb_hashes[index], box=mask_box(forward.mask),
                provenance='automatic_saved_native_bbox_and_independent_seed_rgb')


def native(code, out):
    import numpy as np
    from world_reward.occlusion_recovery import RecoveryPolicy, saved_native_candidate
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    cfg, runtime = settings(code); started = time.monotonic()
    require(sys.version.split()[0] == runtime['python'] and torch.__version__ == runtime['torch']
            and torch.cuda.is_available() and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
            'Exact offline Azure native GPU runtime required')
    prep = strict((out / 'runtime.json').read_bytes()); verify_runtime_source(prep)
    weight = Path(prep['weight_file']); require(identity(weight, runtime['weight_bytes']) == prep['weight'], 'Original checkpoint required')
    selected = inputs(cfg['episodes']); old, bound, old_binding = saved_forward(code, cfg, selected)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(weight), max_num_objects=2,
        multiplex_count=16, use_fa3=False, use_rope_real=True, compile=False, warm_up=False, async_loading_frames=False)
    coverage = verify_checkpoint_coverage(predictor.model, weight); parity = prompt_parity(predictor)
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed']); torch.cuda.reset_peak_memory_stats()
    import cv2
    cv2.setRNGSeed(cfg['seed']); cv2.setNumThreads(4)
    results = []; policy = RecoveryPolicy(**cfg['recovery_gates'])
    try:
        for item in selected:
            clip_start = time.monotonic(); ep = item['episode']; original = old[ep]
            def budget():
                require(time.monotonic() - started < cfg['sam_budget_seconds']
                        and time.monotonic() - clip_start < cfg['sam_episode_budget_seconds'], 'Inclusive native recovery deadline exceeded')
            frames, hashes = decode_original(item); h, w = frames[0].height, frames[0].width
            tracking = original['tracking']; total = item['total']; directory = original['directory']
            require([r['decoded_rgb_sha256'] for r in tracking['records']] == hashes,
                    'Original decoded RGB timeline differs from native saved writer')
            for row in original['grounding']['records']:
                require(row['rgb_sha256'] == hashes[row['frame_index']], 'Original automatic prefix RGB identity changed')
            def load(index, role=1):
                key = f'{role}/{index:06d}.png'
                m = read_mask(directory / 'masks' / key, original['inventory'][key], h, w)
                require(int(m.sum()) == tracking['areas'][str(role)][index], 'Native actual pixels disagree with source areas')
                return m
            dest = out / f'episode_{ep:06d}' / 'automatic_masks'; dest.mkdir(parents=True)
            for role in ('0', '1'): (dest / 'masks' / role).mkdir(parents=True)
            reverse_dir = dest / 'reverse_masks' / '1'; reverse_dir.mkdir(parents=True)
            missing = [i for i, area in enumerate(tracking['areas']['1']) if area == 0]
            relevant = set(missing)
            visible = [i for i, area in enumerate(tracking['areas']['1']) if area > 0]
            if visible: relevant.add(visible[-1])
            seed = load(0); require(seed.any(), 'Original automatic object seed must be visible')
            witness = RgbWitness(frames, seed, hashes, cfg['rgb_witness'], original['pins']['tracking.json']['sha256'],
                                 relevant, budget) if missing else None
            evidence = (lambda i: None) if witness is None else witness.evidence
            anchor = None if not missing else reverse_anchor(tracking['areas']['1'], load, evidence, hashes,
                original['pins']['tracking.json']['sha256'], policy)
            if anchor is not None and anchor['frame_index'] in cfg['frames']:
                anchor = None  # No independent late visibility; preserve full-T instead of aborting.
            state = tracker_state = None
            if anchor is not None:
                state, omitted = compatible_init(predictor.model, frames)
                video = state['input_batch'].img_batch.tensors
                require(omitted == ['offload_state_to_cpu'] and video.dtype == torch.float16
                        and tuple(video.shape) == (total, 3, 1008, 1008), 'Qualified original PIL full-T native input required')
                tracker = predictor.model.tracker
                tracker_state = tracker.init_state(video_height=h, video_width=w, num_frames=total,
                                                   cached_features=state['feature_cache'])
                require(tracker_state['obj_ids'] == [] and tracker_state['cached_features'] is state['feature_cache'],
                        'Fresh object-only singleton must share only image features')
                prefix = [r for r in original['grounding']['records'] if r['object_id'] == 1]
                require([r['frame_index'] for r in prefix] == cfg['frames'], 'All saved object prefix anchors required')
                boxes = prefix + [anchor]
                for row in boxes:
                    index = row['frame_index']; points, labels = corner_prompt(row['box'], w, h)
                    predictor.model._prepare_backbone_feats(state, index, reverse=True)
                    result = tracker.add_new_points(inference_state=tracker_state, frame_idx=index, obj_id=1,
                        points=torch.tensor(points, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.int32),
                        clear_old_points=True, rel_coordinates=True, use_prev_mem_frame=False)
                    require(result[0] == index, 'Native automatic reverse seed original frame changed'); budget()
                tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
            wanted = sorted(set(cfg['frames'] + [total // 2, total - 1])); sampled = {}
            areas = [[], []]; adjacent = [[], []]; previous = None; timeline = []; recovery = []; reverse_inventory = {}
            for index in reversed(range(total)):
                budget(); person = load(index, 0); object_mask = load(index)
                forward = saved_native_candidate(frame_index=index, object_id=1, rgb_sha256=hashes[index], branch='forward',
                    mask=object_mask, native_presence=tracking['records'][index]['native_presence'][1],
                    tracking_report_sha256=original['pins']['tracking.json']['sha256'])
                reverse = None; logit_record = None
                if anchor is not None and index <= anchor['frame_index']:
                    predictor.model._prepare_backbone_feats(state, index, reverse=True)
                    stream = list(tracker.propagate_in_video(tracker_state, start_frame_idx=index,
                        max_frame_num_to_track=0, reverse=True, tqdm_disable=True, run_mem_encoder=True))
                    require(len(stream) == 1, 'Exactly one original-frame native reverse output required')
                    reverse, logit_record = native_singleton(stream[0], index, h, w, hashes[index])
                chosen, choice, fused = select_preserving_forward(forward, reverse, evidence(index), policy)
                masks = np.stack((person, chosen))
                for role in range(2):
                    key = f'{role}/{index:06d}.png'; target = dest / 'masks' / key
                    if role == 0 or choice != 'reverse_rgb_recovery':
                        _copy(directory / 'masks' / key, target, original['inventory'][key])
                    else:
                        with target.open('xb') as f:
                            os.fchmod(f.fileno(), 0o444); Image.fromarray(chosen.astype('uint8') * 255).save(f, format='PNG')
                    areas[role].append(int(masks[role].sum()))
                reverse_mask = np.zeros((h, w), dtype=np.bool_) if reverse is None else reverse.mask
                target = reverse_dir / f'{index:06d}.png'
                with target.open('xb') as f:
                    os.fchmod(f.fileno(), 0o444); Image.fromarray(reverse_mask.astype('uint8') * 255).save(f, format='PNG')
                reverse_inventory[target.name] = identity(target, 16 << 20)
                timeline.append(dict(frame_index=index, decoded_rgb_sha256=hashes[index],
                    person_visible=bool(person.any()), object_visible=bool(chosen.any()),
                    native_presence=[tracking['records'][index]['native_presence'][0],
                                     bool(reverse.native_presence) if choice == 'reverse_rgb_recovery' else forward.native_presence],
                    mask_source=choice))
                recovery.append(dict(frame_index=index, choice=choice, proposed_state=fused.state,
                    proposed_source=fused.source, native_reverse_evaluated=reverse is not None,
                    reverse_logits=logit_record, diagnostics=fused.diagnostics))
                if index in wanted: sampled[index] = masks.copy()
            for role in range(2): areas[role].reverse()
            timeline.reverse(); recovery.reverse()
            for index in range(total):
                m = np.stack([read_mask(dest / 'masks' / f'{role}/{index:06d}.png',
                    identity(dest / 'masks' / f'{role}/{index:06d}.png', 16 << 20), h, w) for role in range(2)])
                if previous is not None:
                    for role in range(2):
                        union = int((previous[role] | m[role]).sum())
                        adjacent[role].append(float((previous[role] & m[role]).sum() / union) if union else None)
                previous = m
            for name in ('prompts.json', 'grounding.json', 'prefix-qualification.json'):
                _copy(directory / name, dest / name, original['pins'][name])
            save(dest / 'tracking.json', dict(status='full_T_complete', episode=ep, frames=total,
                areas={str(k): row for k, row in enumerate(areas)}, records=timeline,
                original_frame_indices=list(range(total)), interpolation=False, quality_verified=False))
            _, inventory, raw = _inventory(dest / 'masks', total); write(dest / 'mask-inventory.json', raw, mode=0o444)
            save(dest / 'reverse-mask-inventory.json', reverse_inventory)
            witness_pin = None if witness is None else witness.persist(dest / 'rgb-witness.npz')
            save(dest / 'recovery.json', dict(schema='world_reward.sam31_recovery_clip.v1',
                original_frame_indices=list(range(total)), anchor=anchor, records=recovery,
                recovered_frames=[r['frame_index'] for r in recovery if r['choice'] == 'reverse_rgb_recovery'],
                unresolved_frames=[r['frame_index'] for r in recovery if r['choice'] == 'unresolved_native_absence'],
                raw_forward_presence_available=False, saved_forward_native_presence_sign_retained=True,
                reverse_raw_presence_recorded=True, reverse_mask_logits_retained='native_sign_mask_plus_range_summary',
                reverse_evaluated_frames=sum(r['native_reverse_evaluated'] for r in recovery),
                person_mask_source='byte_exact_saved_forward', rgb_witness=witness_pin,
                rgb_seed_features=0 if witness is None else len(witness.seed),
                rgb_matched_features=None if witness is None else witness.match_counts,
                geometry='independent_seed_to_rgb_affine_not_3d_pose', policy=cfg['recovery_gates'],
                additional_reverse_passes=int(anchor is not None), model_calls=0 if anchor is None else 1,
                ground_truth_used=False, manual_labels=False, mask_interpolation=False,
                true_occlusion_inferred=False, quality_verified=False))
            qa = preview(frames, sampled, [], dest / 'qa.jpg', ep, cfg['max_jpeg_bytes'])
            forward_sampled = {i: np.stack((load(i, 0), load(i))) for i in wanted}
            reverse_sampled = {i: np.stack((load(i, 0), read_mask(reverse_dir / f'{i:06d}.png',
                reverse_inventory[f'{i:06d}.png'], h, w))) for i in wanted}
            forward_qa = preview(frames, forward_sampled, [], dest / 'qa-forward.jpg', ep, cfg['max_jpeg_bytes'])
            reverse_qa = preview(frames, reverse_sampled, [], dest / 'qa-reverse.jpg', ep, cfg['max_jpeg_bytes'])
            report = dict(original['report'], producer_revision=os.environ['WR_CODE_REVISION'],
                script_sha256=identity(code / HELPERS[0])['sha256'], mask_inventory=inventory,
                diagnostics={str(k): temporal_summary(areas[k], adjacent[k]) for k in range(2)}, qa=qa,
                fixed_native_ids=[0, 1], saved_forward_producer=cfg['saved_forward_producer'],
                saved_forward_report=original['pins']['report.json'], recovery=identity(dest / 'recovery.json', 16 << 20),
                person_masks_modified=False, original_visible_object_masks_modified=False,
                fresh_reverse_pass=anchor is not None, empty_mask_interpolation=False,
                forward_qa=forward_qa, reverse_qa=reverse_qa,
                elapsed_seconds=time.monotonic() - clip_start)
            save(dest / 'report.json', report)
            results.append(dict(episode_index=ep, status='pass', frames=total, automatic_masks=str(dest.relative_to(out)),
                report=identity(dest / 'report.json'), diagnostics=report['diagnostics'], qa=qa,
                originally_empty_frames=len(missing), recovered_frames=sum(r['choice'] == 'reverse_rgb_recovery' for r in recovery),
                reverse_anchor_status='not_needed' if not missing else 'unavailable' if anchor is None else 'available',
                elapsed_seconds=report['elapsed_seconds']))
            del frames, hashes, witness, sampled, forward_sampled, reverse_sampled, previous, person, object_mask, masks, chosen, forward, reverse
            if state is not None: del state, tracker_state, video
            if anchor is not None: del stream, result
            torch.cuda.empty_cache()
            print(json.dumps(dict(stage='sam31_recovery_complete', episode=ep,
                original_empty_frames=len(missing), recovered_frames=results[-1]['recovered_frames'],
                elapsed_seconds=time.monotonic() - started)), flush=True)
        require(inputs(cfg['episodes']) == selected and all(identity(path, max(pin['bytes'], 1)) == pin for path, pin in bound.items())
                and identity(weight, runtime['weight_bytes']) == prep['weight'], 'Original RGB/mask/model evidence changed')
        require(source(ROOT, ROOT / 'jobs' / cfg['saved_forward_producer'] / FORWARD_ENTRY / 'code',
                       cfg['saved_forward_producer'], FORWARD_ENTRY, tuple(old_binding['helpers'])) == old_binding,
                'Original forward code changed')
        verify_runtime_source(prep)
        save(out / 'native-report.json', dict(schema='world_reward.sam31_recovery_tracking.v1',
            status='complete_diagnostic_not_quality_pass', producer_revision=os.environ['WR_CODE_REVISION'],
            episodes=results, elapsed_seconds=time.monotonic() - started, image_id=prep['image_id'],
            model=runtime['model_repo'], model_revision=runtime['model_revision'], source_revision=runtime['source_revision'],
            checkpoint_coverage=coverage, prompt_codec_parity=parity, peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            saved_forward_producer=cfg['saved_forward_producer'], saved_forward_native_report=cfg['saved_forward_native_report'],
            maximum_additional_reverse_passes_per_clip=1, baseline_modified=False, ground_truth_used=False,
            manual_labels=False, quality_verified=False, original_frame_grid_preserved=True,
            source_rehashed_after=True, inputs_rehashed_after=True, gemini_calls=0))
    finally:
        predictor.shutdown()
        for owner in (predictor, getattr(predictor.model, 'tracker', None)):
            if getattr(owner, 'bf16_context', None) is not None:
                owner.bf16_context.__exit__(None, None, None); owner.bf16_context = None


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure host only')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    cfg, runtime = settings(code); binding = source(ROOT, code, revision, ENTRY, HELPERS)
    out = ROOT / 'results' / ('sam31-recovery-' + revision); out.mkdir(mode=0o755)
    prep = prepared(code, cfg, runtime); save(out / 'runtime.json', prep)
    saved_forward(code, cfg, inputs(cfg['episodes']))
    image = strict(control(['docker', 'image', 'inspect', prep['image_id']]))[0]
    require(image['Id'] == cfg['runtime_image'] and image['Config']['Labels']['world_reward.sam31.revision'] == prep['producer_revision'],
            'Exact previously qualified runtime image required')
    name = 'wr-sam31-recovery-' + revision[:12]; started = time.monotonic()
    with (ROOT / 'jobs/.world-reward-h100.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not control(['docker', 'ps', '-q']).strip(), 'No duplicate GPU work permitted')
        cmd = ['docker', 'run', '--rm', '--name', name, '--label', 'world_reward.sam31.owner=' + revision,
            '--gpus', 'all', '--network', 'none', '--read-only', '--user', '0:0', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--memory', '64g', '--cpus', '16', '--shm-size', '2g',
            '--tmpfs', '/tmp:rw,nosuid,size=2g']
        mounts = [(code.parent, True), (out, False), (Path(prep['weight_file']).parent, True),
                  (ROOT / 'results/input-manifest.json', True), (ROOT / 'data/track_1/meta', True),
                  (ROOT / 'results' / ('gemini-sam31-' + cfg['saved_forward_producer']), True),
                  (ROOT / 'jobs' / cfg['saved_forward_producer'] / FORWARD_ENTRY, True)]
        mounts += [(ROOT / f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4', True) for ep in cfg['episodes']]
        for path, readonly in mounts:
            canonical(path); cmd += ['--mount', f'type=bind,src={path},dst={path}' + (',readonly' if readonly else '')]
        cmd += ['--entrypoint', '/usr/bin/env', prep['image_id'], '-i', 'PATH=/usr/local/bin:/usr/bin:/bin',
            'HOME=/tmp', 'PYTHONPATH=/opt/sam3:' + str(code / 'src') + ':' + str(code / 'infra'),
            'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=disabled',
            'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'WR_CODE_REVISION=' + revision,
            '/usr/local/bin/python', '-B', str(code / HELPERS[0]), '--native', str(code), str(out)]
        try:
            with (out / 'native.log').open('xb') as log:
                os.fchmod(log.fileno(), 0o400)
                result = subprocess.run(cmd, stdout=log, stderr=log, timeout=cfg['sam_budget_seconds'] + 60)
            require(result.returncode == 0, 'Native recovery runner failed; inspect Azure-only logs')
        finally:
            if control(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$']).strip():
                owner = control(['docker', 'inspect', name, '--format', '{{index .Config.Labels "world_reward.sam31.owner"}}']).decode().strip()
                require(owner == revision, 'Cannot clean foreign GPU work'); control(['docker', 'rm', '-f', name])
    require(source(ROOT, code, revision, ENTRY, HELPERS) == binding, 'Immutable recovery source changed')
    receipt = strict((out / 'native-report.json').read_bytes())
    save(out / 'report.json', dict(status=receipt['status'], producer_revision=revision,
        source_binding=binding, native_report=identity(out / 'native-report.json'),
        elapsed_seconds=time.monotonic() - started, baseline_modified=False, ground_truth_used=False))
    (out / 'native.log').unlink()  # Success logs are disposable; concise receipts suffice.
    print(json.dumps(dict(status=receipt['status'], episodes=len(receipt['episodes']),
                         elapsed_seconds=receipt['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--native':
        import torch
        with torch.inference_mode(): native(Path(sys.argv[2]), Path(sys.argv[3]))
    else: run()

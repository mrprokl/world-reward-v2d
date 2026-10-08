"""Azure-only saved Gemini boxes -> two fixed-ID native SAM3.1 mask streams.

No semantic/exemplar detection, API replay, manual prompts or model/source patch.
Original full timelines include explicit empty masks for native absence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, inputs
from sam31_runtime import (control, decode_original, verify_runtime_source,
                           verify_checkpoint_coverage, compatible_init)
from qwen4d_masks import _inventory
from world_reward.seeded_tracking import seed_rows, native_masks, temporal_summary

ENTRY = 'run_gemini_sam31_track'
CONFIG = 'configs/gemini_sam31_v1.json'
HELPERS = ('infra/gemini_sam31_track.py', 'infra/run_gemini_sam31_track.sh', CONFIG,
           'src/world_reward/seeded_tracking.py', 'infra/sam31_runtime.py',
           'infra/Dockerfile.sam31', 'configs/sam31_runtime_v1.json',
           'configs/hybrid_pair_v1.json')


def save(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode(), mode=0o444)


def settings(code):
    c = strict((code / CONFIG).read_bytes())
    runtime = strict((code / c['sam_runtime_config']).read_bytes())
    require(c['schema'] == 'world_reward.gemini_sam31.v1'
            and c['episodes'] == random.Random(20261008).sample(range(30), 4)
            and c['frames'] == [0, 14, 29] and c['ground_truth_used'] is False
            and c['manual_labels'] is False and c['quality_verified'] is False
            and runtime['source_revision'] == '2345a4ad109ac29c569da749c91d84f10dc08c40'
            and runtime['model_revision'] == 'daa63191845a41281374e725f4c9e51c7a824460'
            and runtime['use_fa3'] is False and runtime['compile'] is False,
            'Frozen original random cohort / saved boxes / qualified SAM runtime required')
    require(c.get('seed_policy', 'all_valid_prefix_boxes_same_ids') == 'all_valid_prefix_boxes_same_ids'
            and c.get('sam_capacity', 2) == 2 and c.get('sam_multiplex_count', 16) == 16
            and c.get('qualification_prefix_frames', 30) == 30
            and c.get('presence_logit_threshold', 0.0) == 0.0,
            'Fixed identities / native sign / fail-fast prefix contract required')
    return c, runtime


def prepared(code, cfg, runtime):
    old_revision = runtime['reuse_runtime']['producer_revision']
    old = ROOT / 'results' / ('hybrid-pair-' + old_revision) / 'runtime'
    require(identity(old / 'prepare-report.json', 200000) == runtime['reuse_runtime']['prepare_pin'],
            'Exact previously qualified runtime receipt required')
    prep = strict((old / 'prepare-report.json').read_bytes())
    require(prep['status'] == 'pass' and prep['producer_revision'] == old_revision
            and prep['image_id'] == cfg['runtime_image']
            and prep['source_revision'] == runtime['source_revision']
            and prep['model_revision'] == runtime['model_revision'],
            'Qualified reusable actual image/source/model required')
    weight = canonical(Path(prep['weight_file']))
    require(weight == old / runtime['weight_file']
            and identity(weight, runtime['weight_bytes']) == prep['weight']
            == dict(bytes=runtime['weight_bytes'], sha256=runtime['weight_sha256']),
            'Exact native checkpoint required')
    return prep


def prompt_parity(predictor):
    """Prove source-supported corner tokens; padding token is retained natively."""
    import torch
    encoder = predictor.model.tracker.model.interactive_sam_prompt_encoder
    size = encoder.input_image_size
    coords = torch.tensor([[[0.1 * size[1], 0.2 * size[0]],
                            [0.7 * size[1], 0.8 * size[0]]]], device='cuda')
    labels = torch.tensor([[2, 3]], dtype=torch.int32, device='cuda')
    # Compare identical FP32 batches. In BF16, box embeddings use in-place
    # addition whereas point embeddings promote the addition to FP32; that
    # transport proof must not accidentally compare two rounding contracts.
    # Actual native inference still keeps its unmodified BF16/padded route.
    with torch.autocast(device_type='cuda', enabled=False):
        encoded = encoder._embed_points(coords, labels, pad=False)
        boxed = encoder._embed_boxes(coords.reshape(1, 4))
    padded = encoder._embed_points(coords, labels, pad=True)
    require(encoded.shape[1] == 2 and torch.equal(encoded, boxed)
            and padded.shape[1] == 3
            and torch.equal(padded[:, 2], encoder.not_a_point_embed.weight),
            'Native box-corner learned embeddings + native padding parity failed')
    return dict(labels=[2, 3], corners_equal_native_box_embeddings=True,
                native_padding_token_retained=True, upstream_source_modified=False,
                bare_image_box_token_arrays_identical=False,
                equal_shape_unpadded_kernel_comparison=True,
                proof_autocast_disabled=True)


def preview(frames, sampled, seed, target, episode, maximum):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    wanted = sorted(sampled)
    canvas = Image.new('RGB', (384 * len(wanted), 288), (24, 27, 33))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for column, index in enumerate(wanted):
        rgb = np.asarray(frames[index]).copy()
        for role, mask in zip(('person', 'object'), sampled[index]):
            contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (40, 220, 240) if role == 'person' else (255, 155, 45), 2)
        image = Image.fromarray(rgb); image.thumbnail((384, 244))
        canvas.paste(image, (column * 384, 32))
        ImageDraw.Draw(canvas).text((column * 384 + 6, 6), f'Ep {episode} | original frame {index}',
                                   font=font, fill='white')
    for quality in range(85, 24, -5):
        from io import BytesIO
        stream = BytesIO(); canvas.save(stream, format='JPEG', quality=quality)
        raw = stream.getvalue()
        if len(raw) <= maximum:
            write(target, raw, mode=0o444); return identity(target, maximum)
    raise ValueError('Bounded native mask preview encoding failed')


def native(code, out):
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    cfg, runtime = settings(code); started = time.monotonic()
    require(sys.version.split()[0] == runtime['python'] and torch.__version__ == runtime['torch']
            and torch.cuda.is_available() and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
            'Exact qualified Azure offline GPU runtime required')
    prep = strict((out / 'runtime.json').read_bytes()); weight = Path(prep['weight_file'])
    require(identity(weight, runtime['weight_bytes']) == prep['weight'], 'Checkpoint before inference differs')
    verify_runtime_source(prep)
    gemini = ROOT / 'results' / ('gemini-initial-' + cfg['gemini_producer']) / 'native-report.json'
    require(identity(gemini, 200000) == cfg['gemini_native_report'], 'Exact saved automatic box report required')
    saved = strict(gemini.read_bytes())
    require(saved['status'] == 'complete_diagnostic_not_quality_pass' and saved['ground_truth_used'] is False,
            'Saved diagnostic provenance required')
    selected = inputs(cfg['episodes'])
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(weight), max_num_objects=2,
        multiplex_count=16, use_fa3=False, use_rope_real=True, compile=False,
        warm_up=False, async_loading_frames=False)
    coverage = verify_checkpoint_coverage(predictor.model, weight)
    parity = prompt_parity(predictor); results = []; torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20261008); torch.cuda.manual_seed_all(20261008)
    pool = ThreadPoolExecutor(max_workers=1); pending = pool.submit(decode_original, selected[0])
    def budget(clip_start):
        require(time.monotonic() - started < cfg['sam_budget_seconds']
                and time.monotonic() - clip_start < cfg['sam_episode_budget_seconds'],
                'Predeclared inclusive native tracking deadline exceeded')
    try:
        for position, item in enumerate(selected):
            clip_start = time.monotonic(); frames, hashes = pending.result()
            if position + 1 < len(selected): pending = pool.submit(decode_original, selected[position + 1])
            width, height = frames[0].size
            seeds = seed_rows(saved['rows'], item['episode'], cfg['frames'], width, height,
                              {i: hashes[i] for i in cfg['frames']})
            state, omitted = compatible_init(predictor.model, frames)
            video = state['input_batch'].img_batch.tensors
            require(omitted == ['offload_state_to_cpu'] and video.dtype == torch.float16
                    and tuple(video.shape) == (item['total'], 3, 1008, 1008)
                    and float(video.min()) >= -1.01 and float(video.max()) <= 1.01,
                    'Qualified lossless PIL normalized full-grid transport required')
            tracker = predictor.model.tracker
            tracker_state = tracker.init_state(video_height=height, video_width=width,
                                               num_frames=item['total'], cached_features=state['feature_cache'])
            dest = out / f"episode_{item['episode']:06d}" / 'automatic_masks'
            for role in ('0', '1'): (dest / 'masks' / role).mkdir(parents=True, mode=0o755)
            # All six boxes are saved automatic predictions, not manual clicks.
            for seed in seeds:
                predictor.model._prepare_backbone_feats(state, seed['frame_index'], reverse=False)
                result = tracker.add_new_points(inference_state=tracker_state, frame_idx=seed['frame_index'],
                    obj_id=seed['object_id'], points=torch.tensor(seed['points'], dtype=torch.float32),
                    labels=torch.tensor(seed['point_labels'], dtype=torch.int32), clear_old_points=True,
                    rel_coordinates=True, use_prev_mem_frame=False)
                require(result[0] == seed['frame_index'], 'Native seed frame changed')
                budget(clip_start)
            tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
            wanted = sorted(set(cfg['frames'] + [item['total'] // 2, item['total'] - 1]))
            sampled = {}; areas = [[], []]; adjacent = [[], []]; previous = None; timeline = []
            for index in range(item['total']):
                predictor.model._prepare_backbone_feats(state, index, reverse=False)
                stream = tracker.propagate_in_video(tracker_state, start_frame_idx=index,
                    max_frame_num_to_track=0, reverse=False, tqdm_disable=True, run_mem_encoder=True)
                outputs = list(stream)
                require(len(outputs) == 1 and outputs[0][0] == index, 'Exact full original frame output required')
                _, ids, _, video_logits, object_logits = outputs[0]
                masks, presence = native_masks(ids, video_logits.detach().float().cpu().numpy(),
                    object_logits.detach().float().cpu().numpy(), height, width)
                # Keep both original frames even during native absence/occlusion.
                for role in range(2):
                    target = dest / 'masks' / str(role) / f'{index:06d}.png'
                    with target.open('xb') as f:
                        os.fchmod(f.fileno(), 0o444); Image.fromarray(masks[role].astype('uint8') * 255).save(f, format='PNG')
                    area = int(masks[role].sum()); areas[role].append(area)
                    if previous is not None:
                        union = int((previous[role] | masks[role]).sum())
                        adjacent[role].append(float((previous[role] & masks[role]).sum() / union) if union else None)
                timeline.append(dict(frame_index=index, decoded_rgb_sha256=hashes[index],
                    person_visible=bool(masks[0].any()), object_visible=bool(masks[1].any()),
                    native_presence=presence.tolist()))
                if index in wanted: sampled[index] = masks.copy()
                previous = masks
                if index == 29:
                    require(all(any(a) for a in areas), 'Native first30 prefix produced an empty required entity; stop before full tracking')
                    save(dest / 'prefix-qualification.json', dict(status='pass_transport_not_quality',
                        original_frames=30, fixed_ids=[0, 1], native_mask_generation_verified=True,
                        quality_verified=False, prompt_parity=parity))
                    print(json.dumps(dict(stage='sam31_prefix', episode=item['episode'], frames=30,
                                          elapsed_seconds=time.monotonic() - clip_start)), flush=True)
                budget(clip_start)
            require(tracker_state['obj_ids'] == [0, 1] and all(any(a) for a in areas),
                    'Both fixed native identities must retain actual inferred visibility')
            initial = [s for s in seeds if s['frame_index'] == 0]
            save(dest / 'prompts.json', dict(prompts=[dict(frame_index=0, object_id=s['object_id'],
                points=None, point_labels=None, mask_path=None,
                box=dict(zip(('x0', 'y0', 'x1', 'y1'), s['box']))) for s in initial]))
            save(dest / 'grounding.json', dict(**item, width=width, height=height,
                status='seed_available', seed=dict(frame_index=0, person_bbox=initial[0]['box'], object_bbox=initial[1]['box']),
                records=seeds, saved_producer=cfg['gemini_producer'], ground_truth_used=False,
                hand_labeled_test=False, manual_points=False))
            save(dest / 'tracking.json', dict(status='full_T_complete', episode=item['episode'],
                frames=item['total'], areas={str(k): v for k, v in enumerate(areas)}, records=timeline,
                original_frame_indices=list(range(item['total'])), interpolation=False, quality_verified=False))
            _, inventory, raw = _inventory(dest / 'masks', item['total']); write(dest / 'mask-inventory.json', raw, mode=0o444)
            qa = preview(frames, sampled, seeds, dest / 'qa.jpg', item['episode'], cfg['max_jpeg_bytes'])
            report = dict(stage='automatic_masks', status='pass', episode_index=item['episode'],
                frames=item['total'], input_track='track_1', input_sha256=item['video_pin']['sha256'],
                input_dataset_revision='5f68335f3acc802033d1e80728c1633197521de8',
                ground_truth_used=False, hand_labeled_test=False, manual_labels=False, oracle_modes=[],
                quality_verified=False, producer_revision=os.environ['WR_CODE_REVISION'],
                script_sha256=identity(code / 'infra/gemini_sam31_track.py')['sha256'],
                grounding_model='gemini-3.5-flash', grounding_saved_producer=cfg['gemini_producer'],
                grounding_report=cfg['gemini_native_report'], mask_model=runtime['model_repo'],
                model_revision=runtime['model_revision'], source_revision=runtime['source_revision'],
                checkpoint=prep['weight'], fixed_native_ids=[0, 1], seed_frames=cfg['frames'],
                seed_codec='native_PVS_box_corner_tokens_2_3', mask_inventory=inventory,
                diagnostics={str(k): temporal_summary(areas[k], adjacent[k]) for k in range(2)},
                initial_detection_repeated=False, full_original_grid=True, empty_mask_interpolation=False,
                object_prompt=item['object_prompt'], action=item['action'], qa=qa,
                elapsed_seconds=time.monotonic() - clip_start)
            save(dest / 'report.json', report)
            results.append(dict(episode_index=item['episode'], status='pass', frames=item['total'],
                automatic_masks=str(dest.relative_to(out)), report=identity(dest / 'report.json'),
                diagnostics=report['diagnostics'], qa=qa, elapsed_seconds=report['elapsed_seconds']))
            del frames, hashes, state, video, tracker_state, sampled, masks, previous, outputs
            torch.cuda.empty_cache()
            print(json.dumps(dict(stage='sam31_complete', episode=item['episode'], frames=item['total'],
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
        require(inputs(cfg['episodes']) == selected and identity(gemini, 200000) == cfg['gemini_native_report']
                and identity(weight, runtime['weight_bytes']) == prep['weight'], 'Original input/model evidence changed')
        verify_runtime_source(prep)
        save(out / 'native-report.json', dict(schema='world_reward.gemini_sam31_tracking.v1',
            status='complete_diagnostic_not_quality_pass', producer_revision=os.environ['WR_CODE_REVISION'],
            episodes=results, elapsed_seconds=time.monotonic() - started,
            model=runtime['model_repo'], model_revision=runtime['model_revision'],
            source_revision=runtime['source_revision'], image_id=prep['image_id'],
            checkpoint_coverage=coverage, prompt_codec_parity=parity,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(), gemini_calls=0,
            ground_truth_used=False, manual_labels=False, baseline_modified=False, quality_verified=False,
            original_frame_grid_preserved=True, source_rehashed_after=True, inputs_rehashed_after=True))
    finally:
        pending.cancel(); pool.shutdown(wait=True, cancel_futures=True); predictor.shutdown()
        for owner in (predictor, getattr(predictor.model, 'tracker', None)):
            if getattr(owner, 'bf16_context', None) is not None:
                owner.bf16_context.__exit__(None, None, None); owner.bf16_context = None


def run():
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure host only')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    cfg, runtime = settings(code); binding = source(ROOT, code, revision, ENTRY, HELPERS)
    out = ROOT / 'results' / ('gemini-sam31-' + revision); out.mkdir(mode=0o755)
    prep = prepared(code, cfg, runtime); save(out / 'runtime.json', prep)
    image = strict(control(['docker', 'image', 'inspect', prep['image_id']]))[0]
    require(image['Id'] == cfg['runtime_image'] and image['Config']['Labels']['world_reward.sam31.revision'] == prep['producer_revision'],
            'Actual reused pinned Docker image differs')
    name = 'wr-gemini-sam31-' + revision[:12]; started = time.monotonic()
    with (ROOT / 'jobs/.world-reward-h100.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not control(['docker', 'ps', '-q']).strip(), 'No duplicate GPU job permitted')
        cmd = ['docker', 'run', '--rm', '--name', name, '--label', 'world_reward.sam31.owner=' + revision,
            '--gpus', 'all', '--network', 'none', '--read-only', '--user', '0:0', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--memory', '64g', '--cpus', '16', '--shm-size', '2g',
            '--tmpfs', '/tmp:rw,nosuid,size=2g']
        mounts = [(code.parent, True), (out, False), (Path(prep['weight_file']).parent, True),
                  (ROOT / 'results/input-manifest.json', True), (ROOT / 'data/track_1/meta', True),
                  (ROOT / 'results' / ('gemini-initial-' + cfg['gemini_producer']) / 'native-report.json', True)]
        mounts += [(ROOT / f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4', True) for ep in cfg['episodes']]
        for path, readonly in mounts:
            canonical(path); cmd += ['--mount', f'type=bind,src={path},dst={path}' + (',readonly' if readonly else '')]
        cmd += ['--entrypoint', '/usr/bin/env', prep['image_id'], '-i', 'PATH=/usr/local/bin:/usr/bin:/bin',
            'HOME=/tmp', 'PYTHONPATH=/opt/sam3:' + str(code / 'src') + ':' + str(code / 'infra'),
            'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=disabled',
            'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'WR_CODE_REVISION=' + revision,
            '/usr/local/bin/python', '-B', str(code / 'infra/gemini_sam31_track.py'), '--native', str(code), str(out)]
        try:
            with (out / 'native.log').open('xb') as log:
                os.fchmod(log.fileno(), 0o400)
                result = subprocess.run(cmd, stdout=log, stderr=log, timeout=cfg['sam_budget_seconds'] + 60)
            require(result.returncode == 0, 'Native seeded SAM3.1 failed; inspect Azure-only stage logs')
        finally:
            if control(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$']).strip():
                owner = control(['docker', 'inspect', name, '--format', '{{index .Config.Labels "world_reward.sam31.owner"}}']).decode().strip()
                require(owner == revision, 'Cannot clean foreign GPU work'); control(['docker', 'rm', '-f', name])
    require(source(ROOT, code, revision, ENTRY, HELPERS) == binding, 'Immutable source changed')
    receipt = strict((out / 'native-report.json').read_bytes())
    save(out / 'report.json', dict(status=receipt['status'], producer_revision=revision,
        source_binding=binding, native_report=identity(out / 'native-report.json'),
        elapsed_seconds=time.monotonic() - started, baseline_modified=False, ground_truth_used=False))
    print(json.dumps(dict(status=receipt['status'], episodes=len(receipt['episodes']),
                          elapsed_seconds=receipt['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--native':
        import torch
        with torch.inference_mode():
            native(Path(sys.argv[2]), Path(sys.argv[3]))
    else: run()

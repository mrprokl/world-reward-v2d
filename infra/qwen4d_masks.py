"""Copy the frozen task-grounded SAM2 masks into an isolated 4D experiment.

This is a byte/provenance adapter, not detection, fitting or quality validation.
It never changes the original baseline or reruns a model. Original PNGs were
not independently hash-pinned by the first pilot: their measured inventory is
therefore disclosed as a new continuity binding, not historical proof.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from mediapipe_cpu_runtime_verify import canonical, identity, pinned, require, strict, write
from world_reward.task_grounding import (
    build_task_grounding_prompt, choose_seed, fixed_frame_indices, parse_response,
)

ROOT = Path('/srv/scenesmith/world-reward')
PINS = 'configs/task_grounding_saved_pins.json'
CONFIG = 'configs/task_grounding_pilot_v1.json'
PRODUCER = 'a9419e0332e3c6258a8022e9286eaafb04dce223'
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'


def _json(path, maximum=1 << 20):
    pin = identity(path, maximum, readonly=False)
    value = strict(path.read_bytes())
    require(identity(path, maximum, readonly=False) == pin, 'JSON changed while reading')
    return value, pin


def _normalized(value):
    return strict(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _original_inputs(root, episode):
    manifest, manifest_pin = _json(root/'results/input-manifest.json')
    require(type(manifest) is dict and (manifest.get('track'), manifest.get('repo_id'),
        manifest.get('revision')) == ('track_1', 'nvidia/video_to_data_challenge', DATASET),
        'Only pinned official Track1 RGB and metadata are permitted')

    def original(relative):
        rows = [r for r in manifest['files'] if r.get('path') == relative]
        require(len(rows) == 1, 'Unique original input identity required')
        pin = {key: rows[0][key] for key in ('bytes', 'sha256')}
        require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 1_000_000_000
            and type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}', pin['sha256']),
            'Bounded original input pin required')
        path = root/'data'/relative
        require(identity(path, 1_000_000_000, readonly=False) == pin,
                'Original input differs from its acquisition manifest')
        return path, pin

    metadata, metadata_pins = {}, {}
    for name in ('episodes.jsonl', 'episodes_metadata.jsonl', 'tasks.jsonl'):
        path, metadata_pins[name] = original('track_1/meta/'+name)
        metadata[name] = [strict(line) for line in path.read_bytes().splitlines() if line.strip()]

    def one(name, key, value):
        rows = [row for row in metadata[name] if type(row.get(key)) is int and row[key] == value]
        require(len(rows) == 1, 'Unique official episode/task record required')
        return rows[0]

    ep = one('episodes.jsonl', 'episode_index', episode)
    meta = one('episodes_metadata.jsonl', 'episode_index', episode)
    tasks = ep.get('tasks')
    if type(tasks) is list and len(tasks) == 1 and type(tasks[0]) is str:
        rows = [r for r in metadata['tasks.jsonl'] if r.get('task') == tasks[0]]
        require(len(rows) == 1, 'Original action text must match its task catalog')
        action = rows[0]['task']
    else:
        action = one('tasks.jsonl', 'task_index', ep['task_index'])['task']
    require(type(ep.get('length')) is int and ep['length'] >= 9
        and type(action) is str and action.strip()
        and type(meta.get('object_prompt')) is str and meta['object_prompt'].strip(),
        'Original task-conditioned full-T inputs required')
    relative = f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
    video, video_pin = original(relative)
    return dict(episode=episode, total=ep['length'], video=str(video), video_pin=video_pin,
        object_prompt=meta['object_prompt'], action=action, metadata_pins=metadata_pins), manifest_pin


def _inventory(folder, total):
    canonical(folder)
    require(folder.is_dir() and {p.name for p in folder.iterdir()} == {'0', '1'},
            'Exactly the original person/object streams are required')
    records = {}
    names = [f'{i:06d}.png' for i in range(total)]
    for object_id in ('0', '1'):
        stream = canonical(folder/object_id)
        require(stream.is_dir() and sorted(p.name for p in stream.iterdir()) == names,
                'Both PNG streams must cover every original frame with no extras')
        for name in names:
            records[object_id+'/'+name] = identity(stream/name, 16 << 20, readonly=False)
    raw = (json.dumps(records, sort_keys=True, separators=(',', ':'))+'\n').encode()
    return records, dict(files=len(records), bytes=sum(p['bytes'] for p in records.values()),
        sha256=hashlib.sha256(raw).hexdigest()), raw


def _copy(source, destination, pin, maximum):
    require(identity(source, maximum, readonly=False) == pin, 'Original changed before copying')
    with source.open('rb') as src, destination.open('xb') as dst:
        os.fchmod(dst.fileno(), 0o400)
        shutil.copyfileobj(src, dst, 1 << 20)
        dst.flush(); os.fsync(dst.fileno())
    require(identity(destination, maximum) == pin
        and identity(source, maximum, readonly=False) == pin, 'Copy differs or original changed')


def prepare_masks(root, code, episode, destination):
    """Create one new automatic_masks folder; never overwrite or adopt an old one.

    The caller owns the experiment and creates the parent directories. Only
    ``root/experiments/qwen4d-v1-<40-hex-revision>/outputs/episode_N/automatic_masks``
    is accepted. Failed/partial adapters are deliberately retained, not retried.
    """
    root, code, destination = map(canonical, (root, code, destination))
    require(type(episode) is int and episode in (8, 9, 26), 'Frozen three-clip pilot only')
    relative = destination.relative_to(root)
    require(len(relative.parts) == 5 and relative.parts[0] == 'experiments'
        and re.fullmatch('qwen4d-v1-[0-9a-f]{40}', relative.parts[1])
        and relative.parts[2:] == ('outputs', f'episode_{episode:06d}', 'automatic_masks')
        and destination.parent.is_dir() and not destination.exists(),
        'Require a fresh isolated experiment mask destination, never baseline outputs')
    cfg, config_pin = _json(code/CONFIG)
    saved, saved_pin = _json(code/PINS)
    require(cfg['schema'] == 'world_reward.task_grounding_pilot.v1'
        and cfg['episodes'] == [8, 9, 26] and cfg['views'] == 9
        and cfg['model'] == MODEL and cfg['model_revision'] == MODEL_REVISION
        and cfg['training_overlap_verified'] is False and cfg['challenge_overlap_verified'] is False
        and saved['producer_revision'] == PRODUCER, 'Frozen original producer and model required')
    episode_pins = [r for r in saved['episodes'] if type(r.get('episode')) is int and r['episode'] == episode]
    require(len(episode_pins) == 1, 'Unique saved episode pins required')
    episode_pins = episode_pins[0]
    original = root/'results'/('task-grounding-pilot-'+PRODUCER)
    source = original/f'episode_{episode:06d}'
    receipts, bound = {}, {}
    for name in ('report.json', 'inference.json', 'tracking.json'):
        path = original/name
        receipts[name] = pinned(path, saved['files'][name], 4 << 20)
        bound[path] = saved['files'][name]
    pilot, inference, tracking = (receipts[n] for n in ('report.json', 'inference.json', 'tracking.json'))
    require(pilot.get('status') == 'fail' and pilot.get('phase') == 'preview'
        and pilot.get('producer_revision') == PRODUCER and pilot.get('source_rehashed_after') is True
        and pilot.get('quality_verified') is False and pilot.get('ground_truth_used') is False
        and pilot.get('hand_labeled_test') is False and pilot.get('oracle_modes') == [],
        'Preserve the original preview-only failure; no inference/track failure adoption')
    require(inference.get('status') == 'complete' and inference.get('model') == MODEL
        and inference.get('model_revision') == MODEL_REVISION
        and inference.get('quality_verified') is False and tracking.get('quality_verified') is False
        and tracking.get('checkpoint') == {k: cfg['sam2_checkpoint'][k] for k in ('bytes', 'sha256')},
        'Original qualified Qwen/SAM2 model lineage required')
    ground = pinned(source/'grounding.json', episode_pins['grounding'], 1 << 20)
    tracked = pinned(source/'tracking.json', episode_pins['tracking'], 1 << 20)
    bound[source/'grounding.json'] = episode_pins['grounding']
    bound[source/'tracking.json'] = episode_pins['tracking']
    inputs, input_manifest_pin = _original_inputs(root, episode)
    require(all(ground.get(k) == v for k, v in inputs.items())
        and ground.get('status') == 'seed_available' and ground.get('calls') == 1
        and type(episode_pins['frames']) is int and episode_pins['frames'] == inputs['total'],
        'Original grounding must match the exact official task, video and full-T episode')
    for name, document in (('inference', inference), ('tracking', tracking)):
        rows = [r for r in document.get('episodes', []) if type(r.get('episode')) is int and r['episode'] == episode]
        require(len(rows) == 1 and (rows[0] == ground if name == 'inference' else
            rows[0].get('status') == 'full_T_complete' and rows[0].get('frames') == inputs['total']),
            'Global and per-episode receipts disagree')
    indices = fixed_frame_indices(inputs['total'], cfg['views'])
    require(ground.get('indices') == list(indices), 'Original ordered full-frame views required')
    response_pin = identity(source/'response.txt', 65536)
    response = (source/'response.txt').read_text()
    parsed = parse_response(response, indices, ground['width'], ground['height'])
    seed = choose_seed(parsed)
    require(seed is not None and ground.get('records') == _normalized([asdict(r) for r in parsed])
        and ground.get('seed') == _normalized(asdict(seed)), 'No repaired response or new seed selection')
    prompt = build_task_grounding_prompt(inputs['object_prompt'], inputs['action'], indices)
    require(ground.get('prompt_sha256') == hashlib.sha256(prompt.encode()).hexdigest(),
            'One original automatic action/object-conditioned prompt required')
    prompt_pin = identity(source/'prompts.json', 65536)
    prompts = strict((source/'prompts.json').read_bytes())
    expected_prompts = {'prompts': [dict(frame_index=seed.frame_index, object_id=obj,
        points=None, point_labels=None, mask_path=None,
        box=dict(zip(('x0', 'y0', 'x1', 'y1'), getattr(seed, key))))
        for obj, key in ((0, 'person_bbox'), (1, 'object_bbox'))]}
    require(prompts == expected_prompts
        and all(type(row['frame_index']) is int and type(row['object_id']) is int
            and all(type(value) in (int, float) for value in row['box'].values())
            for row in prompts['prompts']),
        'Require original automatic IDs 0/1 and boxes; no manual prompts')
    areas = tracked.get('areas')
    require(tracked.get('status') == 'full_T_complete' and type(tracked.get('frames')) is int
        and tracked['frames'] == inputs['total'] and tracked.get('episode') == episode
        and type(areas) is dict and set(areas) == {'0', '1'}
        and all(type(a) is list and len(a) == inputs['total']
            and all(type(n) is int and n >= 0 for n in a) and max(a) > 0 for a in areas.values()),
        'Require both complete, nonempty-at-least-once original tracks; retain occluded frames')
    mask_records, aggregate, manifest_raw = _inventory(source/'masks', inputs['total'])

    # Only this exclusive new namespace is ever written. No links or baseline adoption.
    destination.mkdir(mode=0o700)
    (destination/'masks').mkdir(mode=0o700)
    for obj in ('0', '1'): (destination/'masks'/obj).mkdir(mode=0o700)
    for name, pin in mask_records.items():
        _copy(source/'masks'/name, destination/'masks'/name, pin, 16 << 20)
    _copy(source/'prompts.json', destination/'prompts.json', prompt_pin, 65536)
    provenance = destination/'provenance'; provenance.mkdir(mode=0o700)
    for name, pin, maximum in (
        ('grounding.json', episode_pins['grounding'], 1 << 20),
        ('tracking.json', episode_pins['tracking'], 1 << 20),
        ('response.txt', response_pin, 65536),
    ):
        _copy(source/name, provenance/name, pin, maximum)
    write(destination/'mask-inventory.json', manifest_raw)
    require(_inventory(source/'masks', inputs['total'])[:2] == (mask_records, aggregate)
        and _inventory(destination/'masks', inputs['total'])[:2] == (mask_records, aggregate),
        'Full-T original/copy mask inventories changed')
    require(all(identity(path, 4 << 20) == pin for path, pin in bound.items())
        and identity(source/'prompts.json', 65536) == prompt_pin
        and identity(source/'response.txt', 65536) == response_pin
        and identity(code/PINS, 1 << 20, readonly=False) == saved_pin
        and identity(code/CONFIG, 1 << 20, readonly=False) == config_pin
        and _original_inputs(root, episode) == (inputs, input_manifest_pin),
        'Frozen producer/config/input provenance changed during mask adaptation')
    report = dict(stage='automatic_masks', status='pass', episode_index=episode,
        frames=inputs['total'], seed_frame=seed.frame_index, input_track='track_1',
        input_sha256=inputs['video_pin']['sha256'], input_dataset_revision=DATASET,
        input_metadata_sha256={k: v['sha256'] for k, v in inputs['metadata_pins'].items()},
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        quality_verified=False, adapter_only=True, models_rerun=False,
        original_failures_reclassified=False, original_preview_touched=False,
        grounding_model=MODEL, grounding_model_revision=MODEL_REVISION,
        object_prompt=inputs['object_prompt'], action=inputs['action'],
        prompt_sha256=ground['prompt_sha256'], prompts_identity=prompt_pin,
        training_overlap_verified=False, challenge_overlap_verified=False,
        original_pilot=dict(path=str(original), producer_revision=PRODUCER,
            status=pilot['status'], phase=pilot['phase'], receipts=saved['files'],
            grounding=episode_pins['grounding'], tracking=episode_pins['tracking'],
            response_identity=response_pin, grounding_status=ground['status'],
            tracking_status=tracked['status']),
        sam2_checkpoint=tracking['checkpoint'], mask_inventory=aggregate,
        mask_inventory_identity=identity(destination/'mask-inventory.json'),
        source_mask_bytes_independently_pinned=False,
        mask_integrity_scope='measured_before_copy_and_rehashed_after_not_historical_independent_pin',
        empty_frames={key: sum(n == 0 for n in values) for key, values in areas.items()},
        adapter_sha256=identity(Path(__file__).resolve(), 1 << 20, readonly=False)['sha256'])
    write(destination/'report.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
    return report


def main():
    require(sys.platform == 'linux', 'Actual mask data stays on Azure Linux')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    require(args.root == ROOT, 'Azure World Reward root required')
    report = prepare_masks(args.root, args.code, args.episode, args.destination)
    print(json.dumps({k: report[k] for k in ('stage', 'status', 'episode_index', 'frames')}))


if __name__ == '__main__':
    main()

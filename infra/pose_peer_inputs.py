"""Exact public full-video pose inputs; Azure-only manifests and byte archives.

No model, image, depth or geometry decoding. Source and report pins are supplied
independently before inventory. Transport archives contain only the allowlist,
not source checkouts, predictions, privileged labels, weights or arbitrary paths.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

ROOT = Path('/srv/scenesmith/world-reward')
DEST = Path('/srv/world-reward-data')
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
KIT = 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
KIT_PIN = {'bytes': 2031, 'sha256': '42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'}
MANIFEST_NAME = 'pose-input-manifest.json'
MAX_MANIFEST = 2_000_000  # Sealed Azure metadata only, never copied locally.
MAX_PINS = 4096
MAX_TOTAL = 19_000_000_000
MAX_ARCHIVE = MAX_TOTAL + 20_000_000
MAX_FILES = 10_000
PRODUCERS = ('run_track1_frontends', 'run_track1_frontends_queued', 'run_track1_volume_frontends', 'run_track1_initializers_only')
TRACKER_HELPERS = ('infra/object_pose_smoke.py', 'infra/body_smoke.py', 'infra/camera_render.py',
    'src/world_reward/__init__.py', 'src/world_reward/data.py', 'src/world_reward/rigid_alignment.py',
    'src/world_reward/mesh_geometry.py', 'src/world_reward/mesh_budget.py', 'src/world_reward/pose_selection.py')
VOLUME_HELPERS = ('infra/volume_geometry_loader.py', 'infra/volume_mesh_pin_inventory.py',
    'infra/object_budget_endpoint.py', 'infra/mesh_link_gate.py', 'infra/mesh_endpoint_gate.py')
IMAGE = 'sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'


def require(value, message):
    if not value: raise ValueError(message)


def episode(value):
    require(type(value) is int and 0 <= value < 30, 'Original Track1 episode integer0..29 required')
    return value


class Once(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        require(getattr(namespace, self.dest, None) is None, 'Each explicit input argument must occur exactly once')
        setattr(namespace, self.dest, value)


def parser(help=True):
    result = argparse.ArgumentParser(allow_abbrev=False, add_help=help)
    def original(value):
        require(bool(re.fullmatch('0|[1-9]|[12][0-9]', value)), 'Canonical original episode0..29 required')
        return int(value)
    result.add_argument('--episode', type=original, choices=range(30), required=True, action=Once)
    return result


def safe_name(name):
    require(type(name) is str and name and not PurePosixPath(name).is_absolute() and
        str(PurePosixPath(name)) == name and '\\' not in name and '\x00' not in name and
        all(part not in ('', '.', '..') for part in name.split('/')), 'Canonical relative allowlist path required')
    return name


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def identity(path, readonly=False, maximum=MAX_TOTAL):
    path = canonical(path); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= maximum and
        (not readonly or not before.st_mode & 0o222), 'Bounded regular original bytes required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): digest.update(chunk)
    after = path.lstat()
    require(all(getattr(before, key) == getattr(after, key) for key in
        ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')), 'Input changed while hashing')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}


def pin(row, empty=False, maximum=MAX_TOTAL):
    require(type(row) is dict and set(row) == {'bytes', 'sha256'} and type(row['bytes']) is int and
        (0 if empty else 1) <= row['bytes'] <= maximum and type(row['sha256']) is str and
        re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Independent SHA/byte pin required')


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate manifest/receipt key forbidden'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON forbidden')))


def write(path, raw):
    with Path(path).open('xb') as stream: os.fchmod(stream.fileno(), 0o400); stream.write(raw)


def digest_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def clip_spec(spec, index):
    require(type(spec) is dict and set(spec) == {'episode_index', 'total_frames', 'width', 'height', 'video_sha256'} and
        type(spec['episode_index']) is int and spec['episode_index'] == episode(index) and
        type(spec['total_frames']) is int and 3 <= spec['total_frames'] <= 3000 and
        all(type(spec[k]) is int and spec[k] > 1 for k in ('width', 'height')) and
        type(spec['video_sha256']) is str and re.fullmatch('[0-9a-f]{64}', spec['video_sha256']), 'Complete original public clip spec required')


def tracker_pins(rows, mesh_source):
    require(mesh_source in ('default', 'volume') and type(rows) is dict and
        set(rows) == set(TRACKER_HELPERS) | (set(VOLUME_HELPERS) if mesh_source == 'volume' else set()), 'Complete immutable tracker helper set required')
    for row in rows.values(): pin(row, True)


def video_path(index):
    return f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode(index):06d}.mp4'


def reports(index):
    base = f'outputs/episode_{episode(index):06d}'
    return ('results/input-manifest.json', *(f'{base}/{name}/report.json' for name in
        ('automatic_masks', 'body_full', 'depth_full', 'object_grounded', 'scale_smoke')))


def paths(index, count, mesh_source):
    episode(index); require(type(count) is int and 3 <= count <= 3000 and mesh_source in ('default', 'volume'), 'Original full frame count/mesh mode required')
    base = f'outputs/episode_{index:06d}'
    result = {video_path(index), 'data/track_1/meta/episodes.jsonl', KIT, *reports(index),
        f'{base}/automatic_masks/prompts.json', *(f'{base}/object_grounded/{name}' for name in ('object.glb', 'transform.json', 'intrinsics.json')),
        *(f'{base}/automatic_masks/masks/{kind}/{frame:06d}.png' for kind in (0, 1) for frame in range(count)),
        *(f'{base}/depth_full/{frame:06d}.npz' for frame in range(count))}
    if mesh_source == 'volume':
        result.update((f'{base}/object_budget_volume/{name}' for name in ('report.json', 'geometry.npz', 'object_fixed_canonical.glb')))
        result.update(('validation/volume_qem_v1/report.json', 'results/image-volume-qem.json'))
    require(len(result) <= MAX_FILES, 'Full allowlist file bound exceeded')
    return result


def code_binding(code, revision, helpers):
    canonical(code); require(re.fullmatch('[0-9a-f]{40}', revision), 'Actual source revision required')
    markers = {}
    for name in ('revision', 'source-sha256'):
        path = code.parent / name; identity(path); raw = path.read_bytes()
        require(raw == (revision + '\n').encode() if name == 'revision' else bool(re.fullmatch(b'[0-9a-f]{64}\n', raw)), 'Original dispatch marker required')
        markers[name] = hashlib.sha256(raw).hexdigest()
    fingerprint = hashlib.sha256()
    for path in (code, *sorted(code.rglob('*'))):
        canonical(path); mode = path.lstat().st_mode
        require(not mode & 0o222 and (stat.S_ISDIR(mode) or stat.S_ISREG(mode)), 'Complete original code must remain readonly')
        if stat.S_ISREG(mode): fingerprint.update(str(path.relative_to(code)).encode() + b'\0' + bytes.fromhex(identity(path, True, 2_000_000)['sha256']))
    for name, row in helpers.items():
        safe_name(name); pin(row, True); require(identity(code / name, True, 2_000_000) == row, 'Independently pinned helper source differs')
    return {'closure_sha256': fingerprint.hexdigest(), 'markers': markers}


def validate_source_pins(value, index, revision, entry, script_sha):
    require(type(value) is dict and set(value) == {'schema', 'episode_index', 'producer_revision', 'producer_entrypoint',
        'producer_helpers', 'tracker_helpers', 'reports', 'mesh_source', 'volume_pins'}, 'Exact independent source pins required')
    require(value['schema'] == 'world_reward.pose_peer.source_pins.v1' and value['episode_index'] == episode(index) and
        value['producer_revision'] == revision and value['producer_entrypoint'] == entry and entry in PRODUCERS and
        re.fullmatch('[0-9a-f]{40}', revision) and re.fullmatch('[0-9a-f]{64}', script_sha), 'Independent producer identity required')
    expected = set(TRACKER_HELPERS) | (set(VOLUME_HELPERS) if value['mesh_source'] == 'volume' else set())
    required_producer = {f'infra/{entry}.sh', *(f'infra/{name}.py' for name in
        ('automatic_masks', 'body_smoke', 'depth_smoke', 'object_smoke', 'scale_smoke'))}
    if entry == 'run_track1_frontends_queued':
        # The queue executes these genuine children in its OWN immutable code
        # namespace; their independent pins must not pretend a child dispatch.
        required_producer.update(('infra/run_track1_frontends.sh',
            'infra/run_track1_initializers_only.sh'))
    require(value['mesh_source'] in ('default', 'volume') and set(value['tracker_helpers']) == expected and
        set(value['reports']) == set(reports(index)) and required_producer <= set(value['producer_helpers']) and
        value['producer_helpers'].get(f'infra/{entry}.sh', {}).get('sha256') == script_sha,
        'Exact reader/helpers/reports/mesh branch pins required')
    for rows in (value['producer_helpers'], value['tracker_helpers'], value['reports']):
        for name, row in rows.items(): safe_name(name); pin(row, True)
    if entry == 'run_track1_frontends_queued':
        for name in required_producer: pin(value['producer_helpers'][name])
    require(value['volume_pins'] is None if value['mesh_source'] == 'default' else type(value['volume_pins']) is dict, 'Explicit volume pins required for volume branch')
    return value


def provenance(root, index, source_pins):
    """Hash every independently pinned JSON before interpretation; no arrays."""
    for name, row in source_pins['reports'].items(): require(identity(root / name) == row, 'Original report pin differs before parsing')
    records = {name: strict_json((root / name).read_bytes()) for name in source_pins['reports']}
    manifest = records['results/input-manifest.json']
    require((manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) ==
        ('track_1', 'nvidia/video_to_data_challenge', DATASET), 'Original Track1 input manifest required')
    def public_file(relative):
        matches = [row for row in manifest['files'] if row['path'] == relative]
        require(len(matches) == 1, 'Exactly one original public input entry required'); row = matches[0]
        require(identity(root / 'data' / relative) == {key: row[key] for key in ('bytes', 'sha256')}, 'Original public input SHA differs'); return row
    public_file('track_1/meta/episodes.jsonl'); video = public_file(video_path(index).removeprefix('data/'))
    rows = [strict_json(line) for line in (root / 'data/track_1/meta/episodes.jsonl').read_text().splitlines()]
    selected = [row for row in rows if type(row.get('episode_index')) is int and row['episode_index'] == index]
    require(len(selected) == 1 and type(selected[0]['length']) is int and 3 <= selected[0]['length'] <= 3000, 'Original complete episode count required')
    count = selected[0]['length']; base = f'outputs/episode_{index:06d}'
    stages = {'automatic_masks': 'automatic_masks', 'body_full': 'sam3d_body_full_video_initializer',
        'depth_full': 'monocular_moge2_full_video', 'object_grounded': 'sam3d_objects_grounded_fixed_frame',
        'scale_smoke': 'predicted_human_anchored_moge2_pointmaps'}
    for name, stage in stages.items():
        record = records[f'{base}/{name}/report.json']; expected = dict(stage=stage, status='pass', episode_index=index,
            input_track='track_1', input_sha256=video['sha256'], ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
        require(all(type(record.get(key)) is type(value) and record[key] == value for key, value in expected.items()), 'Original video-only passed initializer required')
        helper = {'automatic_masks': 'automatic_masks.py', 'body_full': 'body_smoke.py', 'depth_full': 'depth_smoke.py',
            'object_grounded': 'object_smoke.py', 'scale_smoke': 'scale_smoke.py'}[name]
        require(record.get('script_sha256') == source_pins['producer_helpers'].get('infra/' + helper, {}).get('sha256'), 'Actual report helper producer differs')
    masks, body, depth, obj, scale = [records[f'{base}/{name}/report.json'] for name in stages]
    require(masks.get('frames') == count and body.get('total_video_frames') == count and depth.get('total_video_frames') == count,
        'All original frame counts required')
    body_frames, depth_frames = body['frames'], depth['frames']
    require([r['frame_index'] for r in body_frames] == list(range(count)) and [r['frame_index'] for r in depth_frames] == list(range(count)), 'No original frame removed/reindexed')
    for kind in (0, 1):
        directory = canonical(root / base / f'automatic_masks/masks/{kind}')
        require(sorted(p.name for p in directory.iterdir()) == [f'{i:06d}.png' for i in range(count)], 'Automatic mask full original inventory required')
    require(sorted(p.name for p in (root / base / 'depth_full').iterdir()) == [*[f'{i:06d}.npz' for i in range(count)], 'report.json'], 'Exact full depth folder required')
    for i, (b, d) in enumerate(zip(body_frames, depth_frames)):
        require(b['decoded_rgb_sha256'] == d['decoded_rgb_sha256'] and identity(root / base / f'depth_full/{i:06d}.npz')['sha256'] == d['output_sha256'], 'Full depth/body RGB lineage differs')
        require(identity(root / base / f'automatic_masks/masks/0/{i:06d}.png')['sha256'] == b['mask_sha256'], 'Every original inferred person mask/body frame lineage required')
    prompts_path = root / base / 'automatic_masks/prompts.json'; prompts = strict_json(prompts_path.read_bytes())['prompts']
    require(len(prompts) == 2 and {r['object_id'] for r in prompts} == {0, 1} and all(r.get(k) is None for r in prompts for k in ('points', 'point_labels', 'mask_path')), 'Only original automatic box prompts required')
    require(body.get('mask_report_sha256') == source_pins['reports'][f'{base}/automatic_masks/report.json']['sha256'] and
        body.get('prompts_sha256') == identity(prompts_path)['sha256'], 'Original body/mask lineage differs')
    require(obj.get('scale_source') == 'already_human_anchored_MoGe2_no_second_scalar' and
        obj['pointmap_grounding']['alignment_report_sha256'] == source_pins['reports'][f'{base}/scale_smoke/report.json']['sha256'], 'Same original shared metric gauge required')
    for file, key in (('object.glb', 'object_sha256'), ('transform.json', 'transform_sha256'), ('intrinsics.json', 'intrinsics_sha256')):
        require(identity(root / base / 'object_grounded' / file)['sha256'] == obj[key], 'Object geometry/pose/camera pin changed')
    require(strict_json((root / base / 'object_grounded/transform.json').read_bytes()) == obj['transform'], 'Object transform receipt differs')
    camera = strict_json((root / base / 'object_grounded/intrinsics.json').read_bytes())
    require(type(camera['width']) is int and type(camera['height']) is int and camera['width'] > 1 and camera['height'] > 1 and
        all(type(camera[k]) in (int, float) and math.isfinite(camera[k]) for k in ('fx', 'fy', 'cx', 'cy')) and camera['fx'] > 0 and camera['fy'] > 0,
        'Original inferred fixed image camera required')
    require(type(scale['depth_alignment']['shared_scale']) in (int, float) and math.isfinite(scale['depth_alignment']['shared_scale']) and scale['depth_alignment']['shared_scale'] > 0, 'Single original positive scale required')
    require(identity(root / KIT) == KIT_PIN, 'Exact genuine official mesh-budget helper required')
    return {'episode_index': index, 'total_frames': count, 'width': camera['width'], 'height': camera['height'], 'video_sha256': video['sha256']}


def validate_manifest(value, index):
    require(type(value) is dict and set(value) == {'schema', 'episode_index', 'clip_spec', 'mesh_source', 'producer',
        'producer_source_binding', 'tracker_helpers', 'source_pins', 'source_pins_identity', 'volume_pins', 'files'}, 'Exact frozen pose input manifest required')
    require(value['schema'] == 'world_reward.pose_peer.inputs.v1' and value['episode_index'] == episode(index), 'Selected pose manifest required')
    spec = value['clip_spec']; clip_spec(spec, index)
    require(set(value['files']) == paths(index, spec['total_frames'], value['mesh_source']), 'Archive only exact pose input whitelist')
    require(sum(r['bytes'] for r in value['files'].values()) <= MAX_TOTAL, 'Bounded full input bytes required')
    for name, row in value['files'].items(): safe_name(name); pin(row)
    tracker_pins(value['tracker_helpers'], value['mesh_source']); pin(value['source_pins_identity'])
    producer = value['producer']; require(set(producer) == {'revision', 'entrypoint', 'script_sha256'}, 'Actual original producer identity required')
    validate_source_pins(value['source_pins'], index, producer['revision'], producer['entrypoint'], producer['script_sha256'])
    require(value['tracker_helpers'] == value['source_pins']['tracker_helpers'] and value['volume_pins'] == value['source_pins']['volume_pins'] and
        all(value['files'].get(name) == row for name, row in value['source_pins']['reports'].items()), 'Same independently bound source/report pins required')
    if value['mesh_source'] == 'volume':
        require(all(value['files'].get(name) == row for name,row in value['volume_pins']['files'].items()), 'Original seven volume proposal/proof pins must remain exact')
    require(len(digest_json(value)) <= MAX_MANIFEST, 'Sealed Azure manifest exceeds2MB bound')
    return value


def archive(root, manifest, path):
    validate_manifest(manifest, manifest['episode_index']); canonical(path); require(not path.exists(), 'Fresh archive, no overwrite/resume')
    for name, row in manifest['files'].items(): require(identity(root / name) == row, 'Input changed before archive')
    raw = digest_json(manifest)
    with path.open('xb') as output:
        os.fchmod(output.fileno(), 0o400)
        with tarfile.open(fileobj=output, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            member = tarfile.TarInfo(MANIFEST_NAME); member.size = len(raw); member.mode = 0o400; tar.addfile(member, io.BytesIO(raw))
            for name, row in sorted(manifest['files'].items()):
                member = tarfile.TarInfo(name); member.size = row['bytes']; member.mode = 0o400
                with (root / name).open('rb') as stream: tar.addfile(member, stream)
    for name, row in manifest['files'].items(): require(identity(root / name) == row, 'Input changed after archive')
    return identity(path, maximum=MAX_ARCHIVE)


def frozen_pins(value, index):
    require(type(value) is dict and set(value) == {'schema', 'episode_index', 'clip_spec', 'mesh_source',
        'manifest', 'archive', 'inventory_report', 'tracker_helpers', 'source_pins'}, 'Exact tiny independently frozen peer pins required')
    require(value['schema'] == 'world_reward.pose_peer.pins.v1' and value['episode_index'] == episode(index), 'Selected peer transport pins required')
    clip_spec(value['clip_spec'], index); tracker_pins(value['tracker_helpers'], value['mesh_source'])
    pin(value['archive'], maximum=MAX_ARCHIVE); pin(value['manifest'], maximum=MAX_MANIFEST); pin(value['source_pins'])
    require(len(digest_json(value)) <= MAX_PINS, 'Only tiny pins, not full frame/file manifests, may leave Azure')
    producer = value['inventory_report']
    require(type(producer) is dict and set(producer) == {'bytes', 'sha256', 'producer_revision', 'script_sha256'}, 'Independent actual inventory receipt identity required')
    pin({k: producer[k] for k in ('bytes', 'sha256')}); require(re.fullmatch('[0-9a-f]{40}', producer['producer_revision']) and
        re.fullmatch('[0-9a-f]{64}', producer['script_sha256']), 'Actual inventory source revision/SHA required')
    return value


def load_manifest(raw, pins, index):
    """Check independently frozen bytes/SHA BEFORE interpreting any metadata."""
    frozen_pins(pins, index)
    require(len(raw) == pins['manifest']['bytes'] and hashlib.sha256(raw).hexdigest() == pins['manifest']['sha256'], 'Independent inner manifest SHA differs before parsing')
    value = validate_manifest(strict_json(raw), index)
    require(value['clip_spec'] == pins['clip_spec'] and value['mesh_source'] == pins['mesh_source'] and
        value['tracker_helpers'] == pins['tracker_helpers'] and value['source_pins_identity'] == pins['source_pins'], 'Tiny externally frozen manifest/source identity differs')
    return value


def manifest_from_archive(path, pins, index):
    """Only the pinned first bounded member is interpreted before extraction."""
    frozen_pins(pins, index)
    with tarfile.open(path, 'r|') as tar:
        first = next(iter(tar), None)
        require(first is not None and first.name == MANIFEST_NAME and first.isfile() and
            not first.islnk() and first.size == pins['manifest']['bytes'] and first.size <= MAX_MANIFEST, 'Exact pinned manifest must be first TAR member')
        return load_manifest(tar.extractfile(first).read(), pins, index)


def extract(path, manifest, destination):
    """Parse authenticated TAR only; fresh staging never merges original ROOT."""
    validate_manifest(manifest, manifest['episode_index']); canonical(destination); require(not destination.exists(), 'Fresh extraction namespace required')
    destination.mkdir(mode=0o700); owned = destination.stat(); seen = set()
    with tarfile.open(path, 'r|') as tar:
        iterator = iter(tar); first = next(iterator, None)
        require(first is not None and first.name == MANIFEST_NAME and first.isfile() and first.size <= MAX_MANIFEST, 'Exact manifest first member required')
        raw = tar.extractfile(first).read(); require(raw == digest_json(manifest), 'Archive inner manifest differs from independent pins'); write(destination / MANIFEST_NAME, raw)
        for member in iterator:
            safe_name(member.name); require(member.name not in seen and member.name in manifest['files'] and member.isfile() and not member.issym() and not member.islnk(), 'Only unique pinned regular archive payloads allowed')
            row = manifest['files'][member.name]; require(member.size == row['bytes'], 'Archive member size differs'); target = destination / member.name
            parent = destination
            for part in target.relative_to(destination).parts[:-1]:
                parent = parent / part
                if not parent.exists(): parent.mkdir(mode=0o700)
                canonical(parent); info = parent.lstat()
                require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o700, 'Only new owned private payload directories required')
            canonical(target)
            digest = hashlib.sha256(); received = 0
            with target.open('xb') as output:
                os.fchmod(output.fileno(), 0o400); stream = tar.extractfile(member)
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): output.write(chunk); digest.update(chunk); received += len(chunk)
            require(received == row['bytes'] and digest.hexdigest() == row['sha256'], 'Archive payload SHA differs'); seen.add(member.name)
    require(seen == set(manifest['files']), 'Every original required input must be retained')
    after = destination.stat(); require((owned.st_dev, owned.st_ino, owned.st_uid) == (after.st_dev, after.st_ino, os.geteuid()) and
        after.st_mode & 0o777 == 0o700, 'Original private extraction destination changed')
    for name, row in manifest['files'].items():
        require(identity(destination / name, True) == row, 'Retained input changed after extraction')
        # Newly extracted PUBLIC inputs only: original source modes untouched.
        # The owning root0700 ancestor remains private; UID1000 reads RO leaf binds.
        (destination / name).chmod(0o444)
    return {'files': len(seen), 'bytes': sum(r['bytes'] for r in manifest['files'].values()), 'extraction_verified': True}

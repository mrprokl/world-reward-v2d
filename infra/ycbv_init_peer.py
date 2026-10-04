"""Pinned Azure-private initialization transport; no predictor or GT decoder.

Two immutable phases use VM01 -> VM02 SSH connections: pull initial public
bytes, then send raw Objects returns. Source copies are declared replicas, not
new executions or reconstructed producer markers. All failures retain evidence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend_peer_receive as receiver

ROOT = Path('/srv/scenesmith/world-reward')
BASE = 'validation/ycbv_point_pose_v1'
JOB = 'run_ycbv_init_peer'
HELPERS = ('infra/ycbv_init_peer.py', 'infra/run_ycbv_init_peer.sh', 'infra/frontend_peer_receive.py')
OBJECT_HELPERS = ('infra/ycbv_point_objects.py', 'infra/run_ycbv_point_objects.sh',
    'src/world_reward/__init__.py', 'src/world_reward/pointmap.py', 'src/world_reward/mesh_geometry.py',
    'configs/ycbv_point_objects_pins.json')
COMMANDS = {'init': 'world-reward-ycbv-init-public-v1', 'return': 'world-reward-ycbv-objects-return-v1'}
STAGES = {'acquisition': 'external_ycbv_contiguous_rgb_only_acquisition',
    'masks': 'public_ycbv_point_native_object_masks', 'depth': 'public_ycbv_three_frame_zero_native_MoGe2_preflight',
    'objects': 'external_ycbv_three_anchor_native_Objects_initializer'}
REPORT_PATHS = {'acquisition': BASE + '/report.json', 'masks': BASE + '/automatic_masks_v1/report.json',
    'depth': BASE + '/depth_init_v1/report.json', 'objects': BASE + '/objects_init_v1/report.json'}
MANIFEST = 'ycbv_init_peer_manifest.json'
MAX_JSON = 2_000_000
MAX_ARCHIVE = 2 * 1024**3
TTL = 2400
SCENES = (48, 49, 50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON key forbidden'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON forbidden')))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def safe(name):
    require(type(name) is str and name and str(PurePosixPath(name)) == name and not name.startswith('/')
        and '\\' not in name and '\0' not in name and all(p not in ('', '.', '..') for p in name.split('/')),
        'Canonical allowlisted relative path required')
    return name


def canonical(path):
    return receiver.canonical(path)


def state(path):
    s = canonical(path).lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_nlink, s.st_uid, s.st_mtime_ns, s.st_ctime_ns)


def pin(value, empty=False):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'} and type(value['bytes']) is int
        and (0 if empty else 1) <= value['bytes'] <= MAX_ARCHIVE and type(value['sha256']) is str
        and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Exact independent byte/SHA pin required')
    if value['bytes'] == 0:
        require(value['sha256'] == hashlib.sha256(b'').hexdigest(), 'Empty source digest differs')
    return value


def identity(path, empty=False, readonly=True):
    before = state(path)
    require(stat.S_ISREG(before[2]) and before[4] == 1 and (0 if empty else 1) <= before[3] <= MAX_ARCHIVE
        and (not readonly or not before[2] & 0o222), 'Unaliased immutable regular file required')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''): digest.update(block)
    require(state(path) == before, 'Original inode/bytes changed during hash')
    return {'bytes': before[3], 'sha256': digest.hexdigest()}


def write(path, raw):
    canonical(path); require(Path(path).parent.is_dir(), 'Existing private parent required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return identity(path, empty=not raw)


def fresh(path):
    canonical(path); require(not path.exists(), 'Fresh owned namespace required; no overwrite')
    path.mkdir(mode=0o700)
    return state(path)[:2]


def report_pin(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256', 'producer_revision', 'script_sha256'},
        'Closed producer receipt identity required')
    pin({k: value[k] for k in ('bytes', 'sha256')})
    require(value['bytes'] <= MAX_JSON and type(value['producer_revision']) is str
        and re.fullmatch('[0-9a-f]{40}', value['producer_revision']) and type(value['script_sha256']) is str
        and re.fullmatch('[0-9a-f]{64}', value['script_sha256']), 'Original exact producer revision/script required')


def payload_names(direction):
    if direction == 'init':
        return {BASE + '/inputs/manifest.json', *(REPORT_PATHS[k] for k in ('acquisition', 'masks', 'depth')),
            *(f'{BASE}/inputs/scene_{s:06d}_frame_000000.png' for s in SCENES),
            *(f'{BASE}/automatic_masks_v1/scene_{s:06d}/masks/1/000000.png' for s in SCENES),
            *(f'{BASE}/depth_init_v1/scene_{s:06d}_frame_000000.npz' for s in SCENES)}
    return {REPORT_PATHS['objects'], *(f'{BASE}/objects_init_v1/scene_{s:06d}/{name}'
        for s in SCENES for name in ('object.glb', 'transform.json', 'intrinsics.json', 'canonical.npz'))}


def validate_pins(value, direction, network=False):
    require(type(value) is dict and set(value) == {'schema', 'direction', 'files', 'producer_reports',
        'source_replica', 'archive', 'inventory_report'} and value['schema'] == 'world-reward-ycbv-init-peer-pins-v1'
        and direction in COMMANDS and value['direction'] == direction, 'Closed immutable transport pins required')
    require(type(value['files']) is dict and set(value['files']) == payload_names(direction), 'Exactly 13 original payload leaves required')
    for name, row in value['files'].items(): safe(name); pin(row)
    roles = {'acquisition', 'masks', 'depth'} if direction == 'init' else {'objects'}
    require(type(value['producer_reports']) is dict and set(value['producer_reports']) == roles, 'Exact prerequisite receipt roles required')
    for role, row in value['producer_reports'].items():
        report_pin(row); require({k: row[k] for k in ('bytes', 'sha256')} == value['files'][REPORT_PATHS[role]], 'Receipt and payload pins disagree')
    replica = value['source_replica']
    if direction == 'init':
        require(replica is None, 'No source/model/annotation payload in initial public transport')
    else:
        require(type(replica) is dict and set(replica) == {'origin_host', 'replica_host', 'producer_revision', 'files', 'markers'}
            and replica['origin_host'] == 'scenesmith-ncc-h100-01' and replica['replica_host'] == 'world-reward-ncc-h100-02'
            and replica['producer_revision'] == value['producer_reports']['objects']['producer_revision']
            and type(replica['files']) is dict and set(replica['files']) == set(OBJECT_HELPERS)
            and type(replica['markers']) is dict and set(replica['markers']) == {'revision', 'source-sha256'}, 'Explicit exact original Objects source replica required')
        for name, row in replica['files'].items(): pin(row, empty=name.endswith('/__init__.py')); require(row['bytes'] <= MAX_JSON, 'Code-only source bound exceeded')
        for row in replica['markers'].values(): pin(row)
        require(replica['markers']['revision']['bytes'] == 41 and replica['markers']['source-sha256']['bytes'] == 65,
            'Original marker byte lengths required')
        require(replica['files']['infra/ycbv_point_objects.py']['sha256'] == value['producer_reports']['objects']['script_sha256'],
            'Original Objects script and receipt SHA must agree')
    require(sum(row['bytes'] for row in all_files(value).values()) + 1_000_000 <= MAX_ARCHIVE,
        'Entire allowlisted archive must fit frozen byte budget before reads')
    if value['archive'] is not None: pin(value['archive'])
    if value['inventory_report'] is not None: report_pin(value['inventory_report'])
    require(not network or value['archive'] is not None and value['inventory_report'] is not None,
        'Independent completed archive and inventory pins required before networking')
    return value


def all_files(pins):
    result = dict(pins['files']); replica = pins['source_replica']
    if replica:
        prefix = f'jobs/{replica["producer_revision"]}/run_ycbv_point_objects'
        result.update({prefix + '/code/' + name: row for name, row in replica['files'].items()})
        result.update({prefix + '/' + name: row for name, row in replica['markers'].items()})
    return result


def manifest(pins):
    raw = encoded({'schema': 'world-reward-ycbv-init-peer-archive-v1', 'direction': pins['direction'],
        'files': pins['files'], 'producer_reports': pins['producer_reports'], 'source_replica': pins['source_replica'],
        'producer_source_replica_not_execution': pins['source_replica'] is not None,
        'acquisition_receipt_host_only': pins['direction'] == 'init'})
    require(len(raw) <= 65_536, 'Bounded first archive manifest required')
    return raw


def bound_json(path, expected):
    require(expected['bytes'] <= MAX_JSON and identity(path) == expected, 'Independent JSON byte pin required before parsing')
    raw = Path(path).read_bytes()
    require({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == expected, 'JSON changed after hash')
    return strict(raw)


def metadata_only(value):
    """Receipts may contain identities/paths, never credentials or GT values."""
    if type(value) is dict:
        forbidden = {'password', 'secret', 'token', 'access_token', 'api_token', 'api_key', 'private_key',
            'scene_gt', 'scene_camera', 'cam_r_m2c', 'cam_t_m2c', 'cam_k', 'gt_poses', 'ground_truth_values'}
        require(not any(str(k).lower() in forbidden for k in value), 'Credential or private annotation values in transport metadata')
        for child in value.values(): metadata_only(child)
    elif type(value) is list:
        for child in value: metadata_only(child)
    elif type(value) is str:
        require(not re.search(r'(?:https?://[^\s/]+@|[?&](?:token|sig|signature|api_key)=|-----BEGIN [^\n]*PRIVATE KEY-----)', value, re.I),
            'Secret-bearing URL/key in transport metadata')


def verify_files(root, pins):
    wanted = all_files(pins)
    for name, row in wanted.items():
        require(identity(root / name, empty=name.endswith('/__init__.py')) == row, 'Exact original allowlisted source/payload differs')
    receipts = {}
    for role, row in pins['producer_reports'].items():
        value = bound_json(root / REPORT_PATHS[role], {k: row[k] for k in ('bytes', 'sha256')})
        require(value.get('stage') == STAGES[role] and value.get('status') == 'pass' and value.get('phase') == 'complete'
            and value.get('producer_revision') == row['producer_revision'] and value.get('script_sha256') == row['script_sha256']
            and value.get('challenge_inputs_used') is False, 'Original completed no-challenge producer required')
        metadata_only(value)
        post = {'acquisition': 'source_rehashed_after', 'masks': 'all_inputs_sources_assets_outputs_rehashed',
            'depth': 'sources_after_reverified', 'objects': 'source_rehashed_after'}[role]
        require(value.get(post) is True, 'Actual original producer posthash required'); receipts[role] = value
    if pins['direction'] == 'init':
        require(receipts['acquisition'].get('private_annotations_exported_as_inference_inputs') is False
            and receipts['masks'].get('ground_truth_used') is False and receipts['depth'].get('private_truth_read') is False,
            'No private prediction inputs allowed')
        public = bound_json(root / BASE / 'inputs/manifest.json', pins['files'][BASE + '/inputs/manifest.json'])
        require(type(public) is dict and set(public) == {'schema', 'revision', 'license', 'selection', 'attribution', 'images'}
            and public['schema'] == 'world-reward-ycbv-point-rgb-v1' and public['license'] == 'MIT'
            and type(public['images']) is list and len(public['images']) == 288, 'Original RGB-only public manifest required')
        metadata_only(public)
    else:
        replica = pins['source_replica']; original = root / 'jobs' / replica['producer_revision'] / 'run_ycbv_point_objects'
        require(receipts['objects'].get('source_helpers') == replica['files'], 'Source replica must exactly match actual original receipt helper inventory')
        require((original / 'revision').read_bytes() == (replica['producer_revision'] + '\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n', (original / 'source-sha256').read_bytes()), 'Copy original producer markers only; never reconstruct them')
        require(receipts['objects'].get('ground_truth_used') is False and receipts['objects'].get('private_annotations_read') is False,
            'Raw Objects returns cannot derive from private annotations')
    return wanted


def archive(root, pins, out, revision, script_sha, final_check=None):
    validate_pins(pins, pins['direction']); verify_files(root, pins); owned = fresh(out)
    report = {'stage': 'ycbv_init_peer_archive', 'status': 'fail', 'phase': 'archive', 'direction': pins['direction'],
        'producer_revision': revision, 'script_sha256': script_sha, 'private_values_decoded': False,
        'original_files_modified': False, 'source_replica_declared': pins['source_replica'] is not None}
    try:
        raw = manifest(pins); write(out / MANIFEST, raw)
        fd = os.open(out / 'archive.tar', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, 'wb') as stream, tarfile.open(fileobj=stream, mode='w', format=tarfile.GNU_FORMAT) as tar:
            for name in [MANIFEST, *sorted(all_files(pins))]:
                path = out / MANIFEST if name == MANIFEST else root / name
                row = identity(path, empty=name.endswith('/__init__.py')); before = state(path)
                member = tarfile.TarInfo(name); member.size = row['bytes']; member.mode = 0o400
                with path.open('rb') as source: tar.addfile(member, source)
                require(state(path) == before, 'Original file changed during archive')
            stream.flush()
        actual = identity(out / 'archive.tar')
        require(actual['bytes'] <= MAX_ARCHIVE and (pins['archive'] is None or actual == pins['archive']), 'Archive bound/previous pin differs')
        verify_files(root, pins); require(state(out)[:2] == owned, 'Archive directory replaced')
        report.update(status='pass', phase='complete', archive=actual, manifest=identity(out / MANIFEST),
            original_files_rehashed_after=True)
    except Exception:
        report['error'] = 'Public archive gate failed; owned evidence retained'
    seal(out / 'report.json', report, final_check); return report


def original_archive(root, pins):
    validate_pins(pins, pins['direction'], True); row = pins['inventory_report']
    out = root / 'results' / f'ycbv-init-peer-{pins["direction"]}-{row["producer_revision"]}-archive'
    report = bound_json(out / 'report.json', {k: row[k] for k in ('bytes', 'sha256')})
    require(report.get('stage') == 'ycbv_init_peer_archive' and report.get('status') == 'pass' and report.get('phase') == 'complete'
        and report.get('direction') == pins['direction'] and report.get('producer_revision') == row['producer_revision']
        and report.get('script_sha256') == row['script_sha256'] and report.get('archive') == pins['archive']
        and report.get('original_files_rehashed_after') is True and report.get('transport_source_rehashed_after') is True,
        'Original independent completed archive receipt required')
    require(identity(out / 'archive.tar') == pins['archive'] and (out / MANIFEST).read_bytes() == manifest(pins), 'Original sealed archive/manifest differs')
    return out / 'archive.tar'


def inspect_archive(path, pins):
    validate_pins(pins, pins['direction'], True)
    require(identity(path) == pins['archive'], 'Full byte pins required before any archive parse')
    wanted = all_files(pins); names = [MANIFEST, *sorted(wanted)]
    with tarfile.open(path, 'r:') as tar:
        count = 0
        for index, member in enumerate(tar):
            count += 1
            require(index < len(names) and member.name == names[index] and member.type == tarfile.REGTYPE
                and not member.issparse() and not member.pax_headers and member.mode == 0o400
                and member.uid == member.gid == member.mtime == 0, 'Exact ordered regular archive allowlist required')
            expected = len(manifest(pins)) if index == 0 else wanted[member.name]['bytes']
            require(member.size == expected, 'Original exact member size required')
            if index == 0: require(tar.extractfile(member).read(65_537) == manifest(pins), 'Bounded first frozen manifest differs')
            else:
                digest = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''): digest.update(block)
                require(digest.hexdigest() == wanted[member.name]['sha256'], 'Member source/byte hash differs before publication')
        require(count == len(names), 'Missing original archive leaf')
    return wanted


def publish(path, root, pins):
    wanted = inspect_archive(path, pins)
    destination = root / (BASE if pins['direction'] == 'init' else BASE + '/objects_init_v1')
    canonical(destination); require(not destination.exists(), 'New canonical public phase required; no merging originals')
    replica = pins['source_replica']; source = None
    if replica:
        source = root / 'jobs' / replica['producer_revision'] / 'run_ycbv_point_objects'
        canonical(source); require(not source.exists(), 'New explicitly declared producer replica required')
    require(destination.parent.is_dir() and (source is None or source.parent.is_dir()), 'Canonical phase/job parent must be provisioned separately')
    fresh(destination)
    if source: fresh(source)
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            if member.name == MANIFEST: continue
            output = canonical(root / safe(member.name)); prefix = source if member.name.startswith('jobs/') else destination
            require(output.is_relative_to(prefix), 'Output escaped exact owned publication namespace')
            current = prefix
            for part in output.relative_to(prefix).parts[:-1]:
                current = current / part; canonical(current)
                if not current.exists(): current.mkdir(mode=0o700)
                require(current.is_dir() and current.lstat().st_uid == os.geteuid(), 'Owned nonsymlink extraction directory required')
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as stream, tar.extractfile(member) as incoming:
                remaining = member.size
                while remaining:
                    block = incoming.read(min(4 * 1024**2, remaining)); require(block, 'Truncated exact member')
                    stream.write(block); digest.update(block); remaining -= len(block)
                stream.flush(); os.fsync(stream.fileno())
            require(digest.hexdigest() == wanted[member.name]['sha256'], 'Extracted original member hash differs')
    verify_files(root, pins); require(identity(path) == pins['archive'], 'Original archive changed during publication')
    if source:
        for directory in sorted((p for p in source.rglob('*') if p.is_dir()), reverse=True): directory.chmod(0o555)
        source.chmod(0o555)
    return {'extraction_performed': True, 'original_bytes_preserved': True, 'producer_source_replica_not_execution': bool(replica),
        'acquisition_receipt_host_only': pins['direction'] == 'init', 'leaf_count': len(wanted)}


def source_binding(code, revision, direction):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision)
        and code == ROOT / 'jobs' / revision / JOB / 'code' and Path(__file__).resolve() == code / HELPERS[0], 'Actual immutable transport job required')
    files = {name: identity(code / name) for name in HELPERS}
    pins_path = code / f'configs/ycbv_init_peer_{direction}_pins.json'; files[str(pins_path.relative_to(code))] = identity(pins_path)
    for name in ('revision', 'source-sha256'):
        before = state(code.parent / name); require(stat.S_ISREG(before[2]) and before[4] == 1, 'Original regular dispatch marker required')
        raw = (code.parent / name).read_bytes(); require(state(code.parent / name) == before, 'Original dispatch marker changed')
        require(raw == (revision + '\n').encode() if name == 'revision' else re.fullmatch(b'[0-9a-f]{64}\n', raw), 'Original dispatch marker bytes required')
        files['../' + name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return files, validate_pins(bound_json(pins_path, files[str(pins_path.relative_to(code))]), direction)


def public_key(raw):
    require(type(raw) is str and len(raw) < 512 and re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [^\x00-\x1f\x7f]+)?', raw), 'One canonical public Ed25519 key required')
    text = raw.split(' ')[1]; decoded = base64.b64decode(text, validate=True)
    require(len(decoded) == 51 and decoded[:19] == struct.pack('>I', 11) + b'ssh-ed25519' + struct.pack('>I', 32)
        and base64.b64encode(decoded).decode() == text, 'Canonical Ed25519 public blob required')
    return 'ssh-ed25519 ' + text


def key_state(path):
    before = state(path); parent = state(path.parent)
    require(stat.S_ISREG(before[2]) and before[4] == 1 and before[5] == 0 and before[2] & 0o777 == 0o600
        and 1 <= before[3] <= 10_000 and stat.S_ISDIR(parent[2]) and parent[5] == 0 and parent[2] & 0o777 == 0o700,
        'Original private client key metadata required; key bytes never read')
    return before, parent[:2]


def peer(connection, original, direction):
    fields = connection.split() if type(connection) is str else []
    require(len(fields) == 4 and fields[0] == '10.0.0.4' and fields[2:] == ['10.0.0.9', '2222']
        and re.fullmatch('[0-9]{1,5}', fields[1]) and 1 <= int(fields[1]) <= 65535
        and original == COMMANDS[direction], 'Only exact private peer and frozen command allowed')


def seal(path, report, final_check=None):
    try:
        if final_check: final_check()
        report['transport_source_rehashed_after'] = final_check is not None
    except Exception:
        report.update(status='fail', transport_source_rehashed_after=False, error='Final immutable transport source gate failed')
    write(path, encoded(report))


def receive(stream, root, pins, out, final_check=None):
    validate_pins(pins, pins['direction'], True); fresh(out)
    result = receiver.receive(stream, out, pins['archive']['bytes'], pins['archive']['sha256'])
    require(result['status'] == 'pass', 'Byte receiver failed; owned evidence retained')
    report = {'stage': 'ycbv_init_peer_receive', 'status': 'fail', 'direction': pins['direction'], 'archive_retained': True}
    try:
        report.update(publish(out / 'archive.tar', root, pins)); report['status'] = 'pass'
    except Exception: report['error'] = 'Exact publication failed; owned evidence retained'
    seal(out / 'publication.json', report, final_check); return report


def client(root, pins, revision, key, host_public_file, final_check=None):
    validate_pins(pins, pins['direction'], True); private_before = key_state(key)
    host_pin = identity(host_public_file); hostkey = public_key(host_public_file.read_text().rstrip('\n'))
    out = root / 'results' / f'ycbv-init-peer-{pins["direction"]}-{revision}-client'; fresh(out)
    known = out / 'known_hosts'; write(known, ('[10.0.0.9]:2222 ' + hostkey + '\n').encode())
    args = ['/usr/bin/ssh', '-F', '/dev/null', '-T', '-p', '2222', '-i', str(key), '-o', 'IdentitiesOnly=yes',
        '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(known),
        '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3', 'root@10.0.0.9', COMMANDS[pins['direction']]]
    report = {'stage': 'ycbv_init_peer_client', 'status': 'fail', 'direction': pins['direction'], 'private_key_bytes_read_or_recorded': False}
    process = None
    try:
        if pins['direction'] == 'init':
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'})
            report['receiver'] = receive(process.stdout, root, pins, out / 'received', final_check)
            require(process.wait(timeout=15) == 0 and report['receiver']['status'] == 'pass', 'Pull/publication did not complete')
        else:
            path = original_archive(root, pins); original = state(path)
            with path.open('rb') as incoming:
                process = subprocess.Popen(args, stdin=incoming, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'})
                raw = process.stdout.read(4001)
                require(1 <= len(raw) <= 4000 and process.wait(timeout=15) == 0, 'Bounded return receipt required')
            write(out / 'receiver-summary.json', raw); remote = strict(raw)
            require(remote.get('stage') == 'ycbv_init_peer_receive' and remote.get('status') == 'pass'
                and remote.get('direction') == 'return' and remote.get('original_bytes_preserved') is True,
                'Actual complete return publication receipt required')
            require(state(path) == original and original_archive(root, pins) == path, 'Original return archive changed')
            report['receiver'] = remote
        require(key_state(key) == private_before and identity(host_public_file) == host_pin, 'Original key metadata or public host pin changed')
        report['status'] = 'pass'
    except Exception: report['error'] = 'Private transport failed; owned evidence retained'
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        try: require(key_state(key) == private_before and identity(host_public_file) == host_pin, 'Original client/public-key metadata changed')
        except Exception: report.update(status='fail', error='Original client/public-key metadata changed')
        seal(out / 'report.json', report, final_check)
    return report


def control_command(args, seconds=10, missing=False):
    result = subprocess.run(['/usr/bin/timeout', '--signal=TERM', '--kill-after=2s', str(seconds) + 's', *args],
        capture_output=True, text=True, timeout=seconds + 4, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'})
    require(result.returncode == 0 or missing and result.returncode in (1, 4) and result.stdout.strip() == 'not-found', 'Bounded owned server command failed')
    require(len(result.stdout) < 16384, 'Bounded server metadata required'); return result.stdout


def configuration(control, forced=None):
    return '\n'.join(('ListenAddress 10.0.0.9', 'Port 2222', f'HostKey {control}/host_ed25519',
        f'AuthorizedKeysFile {control}/authorized_keys', *(['ForceCommand ' + forced] if forced else []), 'PermitRootLogin forced-commands-only', 'AllowUsers root',
        'StrictModes yes', 'AuthenticationMethods publickey', 'PubkeyAuthentication yes', 'PasswordAuthentication no',
        'KbdInteractiveAuthentication no', 'UsePAM no', 'PermitUserEnvironment no', 'PermitUserRC no', 'PermitTTY no',
        'AllowTcpForwarding no', 'AllowAgentForwarding no', 'X11Forwarding no', 'PermitTunnel no', 'GatewayPorts no',
        'HostbasedAuthentication no', 'GSSAPIAuthentication no', 'AuthorizedKeysCommand none', 'MaxSessions 1',
        'MaxAuthTries 2', 'LogLevel ERROR', f'PidFile {control}/sshd.pid', ''))


def server_paths(revision, direction):
    return Path(f'/run/world-reward-ycbv-init-peer-{direction}-{revision}'), f'world-reward-ycbv-init-peer-{direction}-{revision[:12]}.service'


def stop_owned(unit, control):
    values = dict(line.split('=', 1) for line in control_command(['/usr/bin/systemctl', 'show', unit,
        '--property=ExecStart', '--property=FragmentPath']).splitlines() if '=' in line)
    require(values.get('FragmentPath') == '/run/systemd/transient/' + unit and 'path=/usr/sbin/sshd' in values.get('ExecStart', '')
        and str(control / 'sshd_config') in values.get('ExecStart', ''), 'Only original owned transient server may be stopped')
    control_command(['/usr/bin/systemctl', 'stop', unit])


def server(root, code, revision, pins, client_public_file, final_check=None):
    validate_pins(pins, pins['direction'], True); before = source_binding(code, revision, pins['direction'])[0]
    if pins['direction'] == 'init': original_archive(root, pins)
    client_pin = identity(client_public_file); key = public_key(client_public_file.read_text().rstrip('\n'))
    control, unit = server_paths(revision, pins['direction']); canonical(control)
    for folder in (control.parent, Path('/run/sshd')):
        current = state(folder); require(stat.S_ISDIR(current[2]) and current[5] == 0 and not current[2] & 0o022, 'Original root-controlled server parent required')
    require(not control.exists() and control_command(['/usr/bin/systemctl', 'show', unit, '--property=LoadState', '--value'], missing=True).strip() == 'not-found'
        and not control_command(['/usr/bin/ss', '-H', '-ltnp', 'sport = :2222']).strip(), 'Fresh owned control/unit and unoccupied private port required')
    fresh(control); report = {'stage': 'ycbv_init_peer_server', 'status': 'fail', 'direction': pins['direction'], 'unit': unit,
        'producer_revision': revision, 'runtime_max_seconds': TTL, 'private_key_bytes_read_or_recorded': False}
    attempted = False
    try:
        control_command(['/usr/bin/ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(control / 'host_ed25519')])
        secret = key_state(control / 'host_ed25519'); (control / 'host_ed25519.pub').chmod(0o400)
        hostkey = public_key((control / 'host_ed25519.pub').read_text().rstrip('\n'))
        mode = 'serve' if pins['direction'] == 'init' else 'receive'
        forced = (f'/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT={root} WR_CODE={code} WR_CODE_REVISION={revision} '
            'SSH_CONNECTION="$SSH_CONNECTION" SSH_ORIGINAL_COMMAND="$SSH_ORIGINAL_COMMAND" '
            f'/bin/bash {code}/infra/run_ycbv_init_peer.sh --mode {mode} --direction {pins["direction"]}')
        escaped = forced.replace('\\', '\\\\').replace('"', '\\"')
        write(control / 'authorized_keys', (f'from="10.0.0.4",restrict,command="{escaped}" {key}\n').encode())
        write(control / 'sshd_config', configuration(control, forced).encode())
        control_command(['/usr/sbin/sshd', '-t', '-f', str(control / 'sshd_config')])
        require(identity(client_public_file) == client_pin and key_state(control / 'host_ed25519') == secret, 'Owned key metadata changed before start')
        require(source_binding(code, revision, pins['direction'])[0] == before
            and not control_command(['/usr/bin/ss', '-H', '-ltnp', 'sport = :2222']).strip(), 'Source or private port changed before start')
        attempted = True
        control_command(['/usr/bin/systemd-run', '--quiet', '--unit=' + unit, '--property=Type=exec', '--property=Restart=no',
            '--property=RuntimeMaxSec=' + str(TTL), '/usr/sbin/sshd', '-D', '-e', '-f', str(control / 'sshd_config')])
        values = dict(line.split('=', 1) for line in control_command(['/usr/bin/systemctl', 'show', unit,
            '--property=ActiveState', '--property=MainPID']).splitlines() if '=' in line)
        require(values.get('ActiveState') == 'active' and re.fullmatch('[1-9][0-9]*', values.get('MainPID', '')), 'Owned listener must be active')
        for attempt in range(10):
            rows = control_command(['/usr/bin/ss', '-H', '-ltnp', 'sport = :2222']).splitlines()
            if rows: break
            time.sleep(.2)
        require(len(rows) == 1 and len(rows[0].split()) >= 6 and rows[0].split()[3] == '10.0.0.9:2222'
            and re.findall(r'pid=(\d+)', rows[0]) == [values['MainPID']], 'Exact private listener and owned PID required')
        require(source_binding(code, revision, pins['direction'])[0] == before and identity(client_public_file) == client_pin
            and key_state(control / 'host_ed25519') == secret, 'Actual immutable source/client key changed after start')
        report.update(status='pass', host_public_key=hostkey, client_public_pin=client_pin,
            public_control_pins={n: identity(control / n) for n in ('host_ed25519.pub', 'authorized_keys', 'sshd_config')},
            host_private_key_metadata=list(secret[0]), control_inode=list(state(control)[:2]))
    except Exception:
        if attempted: stop_owned(unit, control)
        report['error'] = 'Owned server setup failed; evidence retained'
    seal(control / 'server-report.json', report, final_check)
    if attempted and report['status'] != 'pass': stop_owned(unit, control)
    return report


def cleanup_server(revision, direction, final_check=None):
    control, unit = server_paths(revision, direction); canonical(control)
    receipt_pin = identity(control / 'server-report.json'); report = bound_json(control / 'server-report.json', receipt_pin)
    require(report.get('stage') == 'ycbv_init_peer_server' and report.get('status') == 'pass' and report.get('unit') == unit
        and report.get('producer_revision') == revision and report.get('direction') == direction
        and list(state(control)[:2]) == report.get('control_inode'), 'Original sealed owned server receipt required')
    require(all(identity(control / n) == row for n, row in report['public_control_pins'].items())
        and list(key_state(control / 'host_ed25519')[0]) == report['host_private_key_metadata'], 'Owned control/key metadata changed')
    stop_owned(unit, control)
    require(not control_command(['/usr/bin/ss', '-H', '-ltnp', 'sport = :2222']).strip(), 'Owned listener must be absent before cleanup')
    require(list(key_state(control / 'host_ed25519')[0]) == report['host_private_key_metadata']
        and identity(control / 'server-report.json') == receipt_pin, 'Original host secret/receipt changed before cleanup')
    (control / 'host_ed25519').unlink()  # Only original owned generated host secret; receipts remain.
    result = {'stage': 'ycbv_init_peer_cleanup', 'status': 'pass', 'owned_host_key_removed': True, 'client_keys_or_cloud_rules_modified': False}
    seal(control / 'cleanup-report.json', result, final_check); return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', required=True, choices=('inventory', 'archive', 'server', 'pull', 'send', 'serve', 'receive', 'stop'))
    parser.add_argument('--direction', required=True, choices=tuple(COMMANDS))
    parser.add_argument('--client-key', type=Path); parser.add_argument('--host-public-key-file', type=Path)
    parser.add_argument('--client-public-key-file', type=Path)
    args = parser.parse_args(argv); revision = os.environ.get('WR_CODE_REVISION'); code = Path(os.environ.get('WR_CODE', '/invalid'))
    require(platform.system() == 'Linux' and os.geteuid() == 0 and os.environ.get('WR_ROOT') == str(ROOT), 'Azure Linux root-only transport required')
    before, pins = source_binding(code, revision, args.direction)
    def final_check(): require(source_binding(code, revision, args.direction)[0] == before, 'Actual immutable transport source/config changed')
    vm02 = args.mode in ('server', 'serve', 'receive', 'stop') or args.mode in ('inventory', 'archive') and args.direction == 'init'
    require(os.uname().nodename == ('world-reward-ncc-h100-02' if vm02 else 'scenesmith-ncc-h100-01'), 'Exact private Azure host role required')
    if args.mode not in ('inventory', 'archive'): validate_pins(pins, args.direction, True)
    def expired(*_): raise TimeoutError('Bounded private transport stopped')
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired); signal.alarm(1900)
    report = None
    try:
        if args.mode in ('inventory', 'archive'):
            verify_files(ROOT, pins)
            report = {'stage': 'ycbv_init_peer_inventory', 'status': 'pass', 'files': len(all_files(pins)), 'private_values_decoded': False}
            if args.mode == 'archive': report = archive(ROOT, pins, ROOT / 'results' / f'ycbv-init-peer-{args.direction}-{revision}-archive', revision, before[HELPERS[0]]['sha256'], final_check)
            else: final_check(); report['transport_source_rehashed_after'] = True
        elif args.mode == 'server':
            require(args.client_public_key_file is not None, 'New independently authenticated client public key required')
            report = server(ROOT, code, revision, pins, args.client_public_key_file, final_check)
        elif args.mode in ('pull', 'send'):
            require(args.mode == ('pull' if args.direction == 'init' else 'send') and args.client_key is not None
                and args.host_public_key_file is not None, 'Exact directional client and authenticated host key required')
            report = client(ROOT, pins, revision, args.client_key, args.host_public_key_file, final_check)
        elif args.mode in ('serve', 'receive'):
            peer(os.environ.get('SSH_CONNECTION'), os.environ.get('SSH_ORIGINAL_COMMAND'), args.direction)
            require(args.mode == ('serve' if args.direction == 'init' else 'receive'), 'Exact forced direction required')
            if args.mode == 'serve':
                path = original_archive(ROOT, pins); original = state(path)
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''): sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush(); require(state(path) == original and original_archive(ROOT, pins) == path, 'Original emitted archive changed')
            else: report = receive(sys.stdin.buffer, ROOT, pins, ROOT / 'results' / f'ycbv-init-peer-return-{revision}-receive', final_check)
        else: report = cleanup_server(revision, args.direction, final_check)
    finally:
        signal.alarm(0); require(source_binding(code, revision, args.direction)[0] == before, 'Actual immutable transport source/config changed')
    if report is not None: print(json.dumps({k: v for k, v in report.items() if k in ('stage', 'status', 'direction', 'host_public_key',
        'archive', 'manifest', 'original_bytes_preserved', 'producer_source_replica_not_execution', 'leaf_count', 'files')}), flush=True)
    return 0 if report is None or report['status'] == 'pass' else 1


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception:
        print('Private YCBV initialization transport failed closed', file=sys.stderr); sys.exit(1)

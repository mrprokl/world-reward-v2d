"""Azure-only FORM-HOI DEV acquisition; opaque references stay quarantined.

The first profile qualifies archive/layout metadata, not model input readiness.
RESERVED is never acquired. No array/checkpoint, mask or calibration is decoded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/form_hoi_external_v1')
ENTRY = 'run_form_hoi_external_acquire'
PROTOCOL = 'configs/form_hoi_insight_v1.json'
PROTOCOL_PIN = dict(bytes=10048, sha256='2de3774a16bb9cdf2ad79555d7f3371dc3bab981230a429944ed7b41481a65b6')
HELPERS = ('infra/form_hoi_external_acquire.py', 'infra/run_form_hoi_external_acquire.sh',
           'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/form_hoi_protocol.py', PROTOCOL,
           'results/audits/form_hoi_insight_v1_metadata.json')
PROFILES = {'inventory_first': (1, 600), 'acquire_dev': (4, 1200)}
BLOCK = 1 << 20
MAX_META = 256 << 10
REFERENCE_NAMES = frozenset({'edex', 'mhr_params_mv.pt', 'mhr_mesh_mv.pt', 'soma_params.npz',
                            'poses.npy', 'output_aligned.glb', 'failure_segments.json'})
METADATA_NAMES = frozenset({'hoi_metadata.yaml', 'hoi_metadata.yml', 'hoi_metadata.json',
                           'metadata.yaml', 'metadata.yml', 'metadata.json', 'prompt.txt'})


def require(value, message):
    if not value:
        raise ValueError(message)


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Inclusive acquisition deadline exhausted')


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not any(q.is_symlink() for q in (p, *p.parents)),
            'Canonical nonsymlink path required')
    return p


def identity(path, maximum):
    p = canonical(path); before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum,
            'Bounded single-link regular artifact required')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    after = p.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')),
            'Artifact changed while hashing')
    return dict(bytes=before.st_size, sha256=h.hexdigest())


def runtime(code):
    sys.path.insert(0, str(Path(code)/'infra'))
    import mediapipe_cpu_runtime_verify as rt
    require(Path(rt.__file__).resolve() == Path(code)/'infra/mediapipe_cpu_runtime_verify.py',
            'Actual source verifier origin required')
    return rt


def source_binding(rt, code, revision):
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    require(Path(__file__).resolve() == code/'infra/form_hoi_external_acquire.py', 'Actual caller source required')
    p = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 16 << 10)
    require(p['dataset'] == 'nvidia/form-hoi' and p['dataset_revision'] == 'c63db107e84c7f74bb4929ef643b67b5c8bcc00e'
            and p['inference_ready'] is False, 'Exact metadata-only frozen cohort required')
    return p, own


def cohort(protocol, profile):
    require(profile in PROFILES, 'Unknown acquisition profile')
    rows = [r for r in protocol['cohort'] if r['split'] == 'development']
    require(len(rows) == 4 and len(protocol['cohort']) == 8 and
            sum(r['split'] == 'reserved_unopened' for r in protocol['cohort']) == 4,
            'Exactly four frozen DEV and four unopened RESERVED required')
    for r in rows:
        require(r['archive_path'] == f"data/{r['sequence_id']}.tar" and
                re.fullmatch(r'[0-9a-f]{64}', r['archive_sha256']) and
                type(r['archive_size']) is int and 0 < r['archive_size'] <= 2 << 30 and
                type(r['member_count']) is int and 0 < r['member_count'] <= 128 and
                r['camera_policy'] == 'front_stereo_camera_left_or_fail_no_substitution' and
                r['original_frame_indices'] == [0, 14, 29], 'Frozen DEV archive/source contract differs')
    return rows[:PROFILES[profile][0]]


def public_url(url):
    p = urllib.parse.urlsplit(url); host = p.hostname or ''
    require(p.scheme == 'https' and p.port in (None, 443) and not p.username and not p.password and
            not p.fragment and (host == 'huggingface.co' or host == 'hf.co' or host.endswith('.hf.co') or
                                host.endswith('.huggingface.co')), 'Only HF publisher HTTPS destinations allowed')
    return url


class PublisherRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        public_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(row, revision, destination, opener, deadline):
    """Up to three bounded attempts, exact Range resume and publisher SHA proof."""
    target = canonical(destination); part = target.with_suffix('.tar.part')
    require(not target.exists() and not part.exists(), 'Fresh download only')
    url = public_url(f"https://huggingface.co/datasets/nvidia/form-hoi/resolve/{revision}/{row['archive_path']}")
    size = 0; digest = hashlib.sha256(); attempts = 0
    try:
        with part.open('xb') as stream:
            for attempt in range(3):
                check(deadline); attempts += 1
                headers = {'User-Agent': 'WorldReward-FORM-DEV-source-acquisition'}
                if size: headers['Range'] = f'bytes={size}-'
                try:
                    with opener.open(urllib.request.Request(url, headers=headers),
                                     timeout=min(30, max(1, deadline-time.monotonic()))) as response:
                        public_url(response.geturl())
                        require(response.status == (206 if size else 200), 'Exact publisher response/range required')
                        if size:
                            require(response.headers.get('Content-Range') ==
                                    f"bytes {size}-{row['archive_size']-1}/{row['archive_size']}",
                                    'Exact resume Content-Range required')
                        length = response.headers.get('Content-Length')
                        require(length is None or length == str(row['archive_size']-size), 'Publisher byte count differs')
                        while True:
                            check(deadline); data = response.read(min(BLOCK, row['archive_size']-size+1))
                            if not data: break
                            size += len(data); require(size <= row['archive_size'], 'Archive download cap exceeded')
                            stream.write(data); digest.update(data)
                        if size != row['archive_size']:
                            raise urllib.error.URLError('Incomplete publisher transport')
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    check(deadline)
                    if attempt == 2: raise
                    time.sleep(min(attempt+1, max(0, deadline-time.monotonic())))
            require(size == row['archive_size'] and digest.hexdigest() == row['archive_sha256'],
                    'Publisher archive SHA/size differs')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o400)
        os.link(part, target); part.unlink()
        return dict(bytes=size, sha256=digest.hexdigest(), attempts=attempts)
    finally:
        if part.exists(): part.unlink()


def safe_member(m):
    n = m.name; p = PurePosixPath(n)
    require(0 < len(n) <= 512 and '\\' not in n and not p.is_absolute() and
            p.as_posix() == n.rstrip('/') and all(q not in ('.', '..') for q in p.parts) and
            str(p) != '.' and not any(ord(c) < 32 or ord(c) == 127 for c in n) and
            (m.isfile() or m.isdir()) and (not m.isdir() or m.size == 0) and
            type(m.size) is int and 0 <= m.size <= 2 << 30,
            'Unsafe tar member path/type/size')
    return p


def member_role(path):
    """Card/guide names only. Unknown layouts remain in the header inventory."""
    p = PurePosixPath(path); parts = p.parts
    if p.name == 'front_stereo_camera_left.mp4' and 'videos' in parts:
        return 'rgb'
    if p.name in METADATA_NAMES:
        return 'native_metadata'
    if p.name in REFERENCE_NAMES:
        return 'reference'
    if p.name == 'front_stereo_camera_left.h5' and any(k in parts for k in ('human_masks', 'object_masks')):
        return 'reference'
    return 'not_retained'


def inventory(archive, row, deadline, *, qualify_rgb=True):
    rows = []; names = {}; expanded = 0
    with tarfile.open(archive, mode='r:') as saved:
        for m in saved:
            check(deadline); p = safe_member(m); key = p.as_posix()
            require(key not in names, 'Duplicate tar member')
            names[key] = m.isdir(); expanded += m.size
            require(len(names) <= 128 and expanded <= row['archive_size'], 'Archive inventory/expanded cap exceeded')
            rows.append(dict(name=m.name, bytes=m.size, directory=m.isdir(),
                             role='directory' if m.isdir() else member_role(key)))
    require(len(rows) == row['member_count'], 'Publisher member census differs')
    for n in names:
        require(all(str(p) not in names or names[str(p)] for p in PurePosixPath(n).parents if str(p) != '.'),
                'Tar file/ancestor collision')
    rgb = [r for r in rows if r['role'] == 'rgb']
    if qualify_rgb:
        require(len(rgb) == 1 and rgb[0]['bytes'] > 0, 'Exactly one original front-left RGB member required')
    return rows


def metadata_schema(raw, name):
    """Keys/types only; YAML is not constructed or executed, no reference values."""
    require(0 < len(raw) <= MAX_META, 'Bounded native metadata required')
    text = raw.decode('utf-8')
    if name.endswith('.json'):
        def pairs(rows):
            out = {}
            for key, value in rows:
                require(key not in out, 'Duplicate native JSON key'); out[key] = value
            return out
        value = json.loads(text, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))
        def walk(x, depth=0):
            require(depth <= 12, 'Metadata nesting cap exceeded')
            if isinstance(x, dict): return {k: walk(v, depth+1) for k,v in x.items()}
            if isinstance(x, list): return dict(type='list', length=len(x))
            return type(x).__name__
        return dict(format='json', schema=walk(value), safe_semantic_fields=semantic_fields(value),
                    reference_values_published=False)
    if name.endswith('.txt'):
        return dict(format='text', nonempty=bool(text.strip()), values_published=False)
    keys = []
    for line in text.splitlines():
        match = re.match(r'^( *)([A-Za-z_][A-Za-z_0-9-]*):(?:\s|$)', line)
        if match: keys.append(dict(indent=len(match[1]), key=match[2]))
    # Read plain metadata scalars only; no YAML tags/anchors/constructors or arrays.
    nesting = []; fields = []
    for line in text.splitlines():
        match = re.match(r'^( *)([A-Za-z_][A-Za-z_0-9-]*):(?:[ \t]*(.*))?$', line)
        if not match: continue
        indent, key, scalar = len(match[1]), match[2], match[3] or ''
        while nesting and nesting[-1][0] >= indent: nesting.pop()
        path = tuple(k for _, k in nesting)+(key,)
        if not scalar or scalar.startswith('#'):
            nesting.append((indent, key)); continue
        if semantic_path(path) and not scalar.startswith(('!', '&', '*', '[', '{', '|', '>')):
            if scalar.startswith('"'):
                try: parsed = json.loads(scalar)
                except (ValueError, TypeError): continue
            elif scalar.startswith("'") and scalar.endswith("'"):
                parsed = scalar[1:-1].replace("''", "'")
            else: parsed = scalar.split(' #', 1)[0].strip()
            if isinstance(parsed, str) and 0 < len(parsed) <= 1024:
                fields.append(dict(path='.'.join(path), value=parsed))
    return dict(format='yaml_lexical_keys_and_safe_scalars', keys=keys, yaml_constructed=False,
                safe_semantic_fields=fields, reference_values_published=False)


def semantic_path(path):
    """Small source-text/ID/frame metadata allowlist; never labels or geometry."""
    lowered = tuple(k.lower() for k in path)
    if any(k in lowered for k in ('calibration', 'intrinsics', 'extrinsics', 'pose', 'poses', 'trajectory',
                                  'mask', 'masks', 'depth', 'ground_truth', 'bbox', 'symmetry', 'robot',
                                  'articulation', 'articulations', 'ground_plane')):
        return False
    return lowered in {('object','prompt'), ('object','name'), ('object','id'), ('object','object_id'),
                       ('person','id'), ('person','person_id'), ('object_prompt',), ('object_id',),
                       ('person_id',), ('action',), ('action','description'), ('action','title'),
                       ('action','name'), ('description',), ('title',), ('sequence_id',), ('fps',),
                       ('width',), ('height',), ('num_frames',), ('camera_names',)} or (
        len(lowered) <= 4 and 'camera' in lowered and lowered[-1] in ('name','fps','width','height'))


def semantic_fields(value):
    fields = []
    def walk(x, path=()):
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, path+(k,))
        elif semantic_path(path) and type(x) in (str, int, float, bool) and len(str(x)) <= 1024:
            fields.append(dict(path='.'.join(path), value=x))
        elif semantic_path(path) and isinstance(x, list) and len(x) <= 8 and all(
                isinstance(v, str) and len(v) <= 128 for v in x):
            fields.append(dict(path='.'.join(path), value=x))
    walk(value)
    return fields


def seal(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    temporary = path.with_name(path.name+'.part')
    require(not path.exists() and not temporary.exists(), 'Receipt collision')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o400)
    os.link(temporary, path); temporary.unlink()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def extract(archive, rows, destination, deadline, *, inventory_only=False):
    public = destination/'inputs'; private = destination/'eval_private'
    public.mkdir(mode=0o755); private.mkdir(mode=0o700)
    kept = []; metadata = []
    with tarfile.open(archive, mode='r:') as saved:
        members = {m.name: m for m in saved}
        for r in rows:
            if r['role'] not in ('rgb', 'native_metadata', 'reference'): continue
            if inventory_only and r['role'] != 'native_metadata': continue
            check(deadline); source = members[r['name']]
            require(source.size == r['bytes'] and source.isfile(), 'Selected member changed')
            relative = 'inputs/rgb.mp4' if r['role'] == 'rgb' else 'eval_private/'+r['name']
            target = destination/relative; target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = hashlib.sha256(); size = 0; small = bytearray()
            require(r['role'] != 'native_metadata' or 0 < r['bytes'] <= MAX_META, 'Native metadata size cap exceeded')
            with saved.extractfile(source) as src, target.open('xb') as dst:
                while True:
                    check(deadline); data = src.read(BLOCK)
                    if not data: break
                    size += len(data); require(size <= r['bytes'], 'Selected member cap exceeded')
                    dst.write(data); digest.update(data)
                    if r['role'] == 'native_metadata': small.extend(data)
                require(size == r['bytes'] and size > 0, 'Empty or truncated selected payload')
                dst.flush(); os.fsync(dst.fileno()); os.fchmod(dst.fileno(), 0o444 if r['role'] == 'rgb' else 0o400)
            pin = dict(bytes=size, sha256=digest.hexdigest())
            require(identity(target, max(1, size)) == pin, 'Retained selected member copy differs')
            kept.append(dict(source=r['name'], file=relative, role=r['role'], **pin))
            if r['role'] == 'native_metadata':
                metadata.append(dict(source=r['name'], **metadata_schema(bytes(small), source.name)))
    return kept, metadata


def acquire(rt, code, revision, profile, *, opener=None, reservation=None):
    started = time.monotonic(); protocol, before = source_binding(rt, code, revision)
    rows = cohort(protocol, profile); budget = PROFILES[profile][1]; deadline = started+budget
    target = canonical(DATA/f'{profile}-{revision}')
    require(target.parent.is_dir() and shutil.disk_usage(target.parent).free >= 6 << 30,
            'Existing Azure data namespace and at least6GiB free required')
    if reservation is None:
        require(not target.exists(), 'Fresh acquisition namespace required'); target.mkdir(mode=0o700)
    else:
        s = target.lstat()
        require(reservation == dict(device=s.st_dev, inode=s.st_ino, source_sha256=before['closure_sha256']) and
                stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o700 and
                not tuple(target.iterdir()), 'Exact fresh owned namespace reservation required')
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), PublisherRedirect())
    report = dict(schema='world_reward.form_hoi_external_acquisition.v1', status='fail', stage=profile,
                  producer_revision=revision, source_before=before, protocol_identity=PROTOCOL_PIN,
                  sequences=[], reserved_acquired=0, models_loaded=False, gpu_used=False,
                  reference_arrays_decoded=False, reference_masks_decoded=False, calibration_values_decoded=False,
                  raw_metadata_public=False, native_public_text_qualified=False, inference_ready=False,
                  rgb_content_duplicate_guard_passed=False, official_object_person_split_verified=False,
                  training_overlap_verified=False, external_reference_kind='multiview_reconstructed_pseudo_GT',
                  budget_seconds=budget)
    active = None; archive = None
    try:
        for row in rows:
            check(deadline); active = target/row['sequence_id']; active.mkdir(mode=0o700)
            archive = active/'source.tar'
            downloaded = fetch(row, protocol['dataset_revision'], archive, opener, deadline)
            first = profile == 'inventory_first'
            members = inventory(archive, row, deadline, qualify_rgb=not first)
            kept, schemas = extract(archive, members, active, deadline, inventory_only=first)
            receipt = dict(schema='world_reward.form_hoi_external_sequence_acquisition.v1',
                           producer_revision=revision, sequence_id=row['sequence_id'], split='development',
                           archive=downloaded, member_inventory=members, retained=kept, native_metadata_schema=schemas,
                           inference_ready=False, original_frame_grid_verified=False, reserved_used=False,
                           annotations_loaded=False, source_revision=protocol['dataset_revision'])
            check(deadline); receipt_pin = seal(active/'receipt.json', receipt)
            if not first: archive.unlink()
            archive = None
            report['sequences'].append(dict(sequence_id=row['sequence_id'], receipt=receipt_pin,
                                           retained_bytes=sum(k['bytes'] for k in kept), archive_removed=not first,
                                           source_archive_retained_for_qualification=first,
                                           source_archive=downloaded if first else None,
                                           member_inventory=members if first else None,
                                           rgb_sha256=next((k['sha256'] for k in kept if k['role'] == 'rgb'), None),
                                           native_metadata_schema=schemas,
                                           retained_roles={k: sum(v['role'] == k for v in kept)
                                                           for k in ('rgb','native_metadata','reference')}))
            print(json.dumps(dict(stage=profile, acquired_DEV=len(report['sequences']),
                                  total_DEV=len(rows), reserved_acquired=0, inference_ready=False)), flush=True)
            active = None
        report['source_after'] = rt.source(ROOT, code, revision, ENTRY, HELPERS)
        require(report['source_after'] == before, 'Immutable source closure changed')
        check(deadline); report['status'] = 'pass'
    except Exception as error:
        report['error_type'] = type(error).__name__
        if isinstance(error, ValueError): report['error_context'] = str(error)
    finally:
        if active is not None:
            # This directory was exclusively created here; sealed earlier clips survive.
            require(active.parent == target and not active.is_symlink() and active.stat().st_uid == os.getuid(),
                    'Refuse foreign active-sequence cleanup')
            shutil.rmtree(active)
        report['elapsed_seconds'] = time.monotonic()-started
        report['owned_failed_partials_removed'] = not any(target.glob('*/source.tar.part'))
        report['verified_first_archive_retained_for_qualification'] = profile == 'inventory_first' and bool(report['sequences'])
        report['owned_raw_archives_removed'] = not any(target.glob('*/source.tar'))
        seal(target/'report.json', report)
    return report


def main():
    require(sys.platform == 'linux' and os.environ.get('WR_ROOT') == str(ROOT), 'Azure Linux runtime only')
    require(len(sys.argv) == 2 and sys.argv[1] in PROFILES, 'One frozen profile required')
    profile = sys.argv[1]; code = canonical(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    require(re.fullmatch(r'[0-9a-f]{40}', revision), 'Pinned source revision required')
    rt = runtime(code)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition deadline')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition terminated')))
    signal.alarm(PROFILES[profile][1])
    result = acquire(rt, code, revision, profile,
                     reservation=rt.strict(os.environ['WR_FORM_EXTERNAL_NAMESPACE_LEASE']))
    signal.alarm(0)
    print(json.dumps({k: result[k] for k in ('stage','status','elapsed_seconds','inference_ready')}), flush=True)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(stage='form_hoi_external_acquisition', status='fail', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)

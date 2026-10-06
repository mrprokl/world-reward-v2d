"""One Azure-only byte replica of all48 public RGB/endpoint banks; no inference."""
import argparse
import base64
import hashlib
import io
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tarfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import articulated_runtime_transfer as transport
import atomic_metadata as atomic
import sealed_callback_publication as publication
import vcoco_fit_cal_endpoint_run as endpoint
import vcoco_fit_cal_acquisition_inputs as acquisition

rt, ROOT, encode = endpoint.rt, endpoint.ROOT, endpoint.original.encode
ENTRY = 'run_vcoco_full_public_replica'
SCHEMA = 'world_reward.vcoco_full_public_replica.v1'
DEST = Path('/srv/world-reward-data/vcoco_full_public_replica_v1')
REV = '6837a74a36a29c7c0cc79dfecfb98fc03d569a3f'
DECLARATION = dict(producer_revision=REV, files=340, entries=345,
    closure_sha256='dff99fe3f87febfed73162c3fc535d8c83ed11d44d1271a4c23564d35e6c1c4c',
    source_XZ_sha256='a9c86bf2b00e48450b6a84e866987e8068e3bde726909f1446953534579cd47c')
PINS = {'report.json': dict(bytes=280261, sha256='0ff65853cc5ecec8060b7206d3cfe3960923c8b2ebef8b11eda3eae8ed75da62'),
    'native.json': dict(bytes=177044, sha256='83cb64b09e026332481fb5227b618acd036edff1d220257c7c3a97470280fe56'),
    'proof.json': dict(bytes=12350, sha256='71f296fc005ffb12191bef5d98dfd7864bce30b7925e7be08d573b60b75df1c3')}
PUBLIC = dict(bytes=9754, sha256='e98657edc8e081ed5b101edf22cdbb490b77bf36384e8952eff49f82d0c7b93c')
BUDGET, MAXIMUM, MAX_CONTROL = 300, 128 << 20, 256 << 10
NAMES = frozenset(('public-reference.json', 'inputs/manifest.json',
    *(f'inputs/image_{i:06d}.jpg' for i in range(48)), *(f'banks/image_{i:06d}.npz' for i in range(48))))
BANK_FIELDS = frozenset(('image_id', 'original_frame_index', 'bank_index', 'image_size', 'input_file', 'input_identity',
    'person_native_queries', 'person_postprocessor_rows', 'person_retained_rows', 'person_ids', 'owl_patches',
    'original_slot', 'acquired_ordinal', 'file', 'identity', 'arrays'))
HELPERS = tuple(dict.fromkeys(('infra/vcoco_full_public_replica.py', 'infra/run_vcoco_full_public_replica.sh',
    'infra/articulated_runtime_transfer.py', 'infra/runtime_image_archive.py', 'infra/atomic_metadata.py',
    'infra/sealed_callback_publication.py', *endpoint.HELPERS, *acquisition.acquisition.HELPERS)))
STAGES = ('source', 'peer', 'blob', 'sender', 'projection', 'pack', 'commit', 'head',
    'receipt', 'download', 'archive', 'install', 'post', 'publication')
ERRORS = frozenset(('ValueError', 'RuntimeError', 'TimeoutError', 'OSError', 'KeyError', 'ImportError', 'ModuleNotFoundError'))


def pin(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
def error(exc): return type(exc).__name__ if type(exc).__name__ in ERRORS else 'other'
def check(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Inclusive replica deadline')
def snapshot(path):
    s = rt.canonical(path).lstat()
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink', 'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns'))
def fixed_pin(value, maximum=MAXIMUM):
    rt.require(type(value) is dict and set(value) == {'bytes', 'sha256'} and type(value['bytes']) is int
        and 0 < value['bytes'] <= maximum and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Exact bounded byte/SHA pin required')
    return value
def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)
def private_directory(path, mode, names=None):
    s = rt.canonical(path).lstat()
    rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == mode and s.st_uid == s.st_gid == 0
        and (names is None or {p.name for p in path.iterdir()} == set(names)), 'Exact root-private namespace required')
def tree_state(path): return {str(p): snapshot(p) for p in (path, *sorted(path.rglob('*')))}


def source(code, revision):
    binding = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    modules = ((sys.modules[__name__], HELPERS[0]), (endpoint, 'infra/vcoco_fit_cal_endpoint_run.py'),
        (acquisition, 'infra/vcoco_fit_cal_acquisition_inputs.py'), (transport, 'infra/articulated_runtime_transfer.py'),
        (atomic, 'infra/atomic_metadata.py'), (publication, 'infra/sealed_callback_publication.py'))
    rt.require(all(Path(m.__file__).resolve() == code/n for m, n in modules), 'Actual explicit helper origins required')
    for p in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        s = rt.canonical(p).lstat()
        rt.require(s.st_uid == s.st_gid == 0 and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1), 'Immutable root-owned whole source required')
    return dict(binding=binding, states=tree_state(code), markers={n: snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def validate_projection(value, files):
    """Metadata only. Native callers must reopen/validate all17 numeric arrays."""
    rt.require(type(value) is dict and set(value) == {'schema', 'inputs', 'manifest_identity', 'banks'}
        and value['schema'] == 'world_reward.public_endpoint_bank_reference.v1' and value['manifest_identity'] == PUBLIC,
        'Only four public projection keys required')
    inputs = value['inputs']; rt.require(type(inputs) is dict and set(inputs) == {'schema', 'images'}
        and inputs['schema'] == endpoint.seam.public.SCHEMA and type(inputs['images']) is list and len(inputs['images']) == 48
        and type(value['banks']) is list and len(value['banks']) == 48 and pin(encode(inputs)) == PUBLIC, 'All48 original public inputs required')
    ids = set()
    for i, (r, b) in enumerate(zip(inputs['images'], value['banks'])):
        rt.require(type(r) is dict and set(r) == endpoint.seam.public.KEYS and r['file'] == f'image_{i:06d}.jpg'
            and type(r['image_id']) is str and re.fullmatch('[0-9a-f]{32}', r['image_id']) and r['image_id'] not in ids
            and all(type(r[k]) is int and r[k] > 0 for k in ('width', 'height')) and r['width']*r['height'] <= 16 << 20
            and fixed_pin({k: r[k] for k in ('bytes', 'sha256')}, 16 << 20) == files['inputs/'+r['file']], 'Original unique six-field RGB required')
        ids.add(r['image_id']); rt.require(type(b) is dict and set(b) == BANK_FIELDS
            and all(type(b[k]) is int and b[k] == i for k in ('bank_index', 'original_slot', 'acquired_ordinal'))
            and type(b['original_frame_index']) is int and b['original_frame_index'] == 0
            and b['image_id'] == r['image_id'] and b['input_file'] == r['file'] and b['input_identity'] == files['inputs/'+r['file']]
            and b['image_size'] == [r['height'], r['width']] and b['file'] == f'image_{i:06d}.npz'
            and b['identity'] == files['banks/'+b['file']] and type(b['person_native_queries']) is int and b['person_native_queries'] == 900
            and type(b['owl_patches']) is int and b['owl_patches'] == 3600
            and all(type(b[k]) is int for k in ('person_postprocessor_rows', 'person_retained_rows'))
            and 0 <= b['person_retained_rows'] <= b['person_postprocessor_rows'] <= 900
            and type(b['person_ids']) is list and b['person_ids'] == [f'image:{r["image_id"]}/person/retained:{j:06d}' for j in range(b['person_retained_rows'])]
            and len(set(b['person_ids'])) == len(b['person_ids']) == b['person_retained_rows']
            and type(b['arrays']) is dict and set(b['arrays']) == endpoint.seam.KEYS, 'Full original bank slots/counts/IDs required')
        for a in b['arrays'].values():
            rt.require(type(a) is dict and set(a) == {'shape', 'dtype', 'sha256'} and type(a['shape']) is list
                and all(type(n) is int and n >= 0 for n in a['shape']) and type(a['dtype']) is str
                and re.fullmatch('[0-9a-f]{64}', str(a['sha256'])), 'Original array metadata required')
    rt.require(files['inputs/manifest.json'] == PUBLIC and files['public-reference.json'] == pin(encode(value)), 'Public projection byte binding differs')
    return value


def sender_inputs(code, binding, deadline):
    check(deadline); old = ROOT/'jobs'/REV/endpoint.ENTRY/'code'
    host, native, proof = (rt.pinned(endpoint.OUTPUT/n, p, 2 << 20) for n, p in PINS.items())
    original = rt.source(ROOT, old, REV, endpoint.ENTRY, tuple(host['source_binding']['helpers']))
    rt.require(original == host['source_binding'] == proof['source'] and original['entries'] == DECLARATION['entries']
        and original['closure_sha256'] == DECLARATION['closure_sha256'] and sum(p.is_file() for p in old.rglob('*')) == DECLARATION['files']
        and (old.parent/'source-sha256').read_bytes() == (DECLARATION['source_XZ_sha256']+'\n').encode()
        and all(binding['helpers'][n] == original['helpers'][n] for n in (*endpoint.HELPERS, *acquisition.acquisition.HELPERS)), 'Complete original340-file source/unchanged helpers required')
    for p in (old, *old.rglob('*'), old.parent/'revision', old.parent/'source-sha256'):
        s = rt.canonical(p).lstat(); mode = 0o555 if p.is_dir() or p == old/'infra/run_rgb_endpoint_bank.sh' else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == mode, 'Exact original Git modes required')
    inputs = endpoint.seam.public_inputs(endpoint.CONTEXT, PUBLIC)
    flags = ('reference_metadata_read', 'split_metadata_read', 'FIT_performed', 'ground_truth_used', 'challenge_inputs_used', 'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption')
    rt.require(host['schema'] == endpoint.SCHEMA and host['stage'] == 'full48_endpoint_observations_host'
        and host['status'] == 'pass' and host['producer_revision'] == REV and host['native_exit_status'] == 0
        and host['public_inputs_identity'] == PUBLIC and host['native_report_identity'] == PINS['native.json']
        and host['acquired_images'] == 48 and host['native_images'] == native['images']
        and host['outputs_sealed'] is host['owned_cleanup_verified'] is host['source_inputs_assets_runtime_rehashed_after'] is True
        and all(host[k] is False for k in flags) and proof['inputs_identity'] == PUBLIC and proof['images'] == 48
        and proof['profile'] == endpoint.seam.PROFILE and proof['image_id'] == host['image_id'] == endpoint.IMAGE
        and proof['native_files'] == {n: original['helpers'][n] for n in endpoint.NATIVE_FILES}, 'Sealed original label-blind host/proof required')
    endpoint.validate_native(native, proof, REV, PINS['proof.json'], inputs)
    cid = endpoint.OUTPUT/'.container.cid'; rt.identity(cid, 65); saved = cid.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', saved) and not endpoint.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+saved.decode().strip()], deadline)
        and not endpoint.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/world-reward-vcoco-fit-cal-endpoint-'+REV[:12]+'$'], deadline), 'Original CID/name must remain absent')
    paths = {'inputs/manifest.json': endpoint.DATA/'manifest.json',
        **{'inputs/'+r['file']: endpoint.DATA/r['file'] for r in inputs['images']},
        **{'banks/'+r['file']: endpoint.OUTPUT/r['file'] for r in native['images']}}
    projection = dict(schema='world_reward.public_endpoint_bank_reference.v1', inputs=inputs, manifest_identity=PUBLIC,
        banks=[{k: r[k] for k in BANK_FIELDS} for r in native['images']])
    files = {n: rt.identity(p, 16 << 20) for n, p in paths.items()}; files['public-reference.json'] = pin(encode(projection))
    validate_projection(projection, files)
    private_directory(endpoint.DATA, 0o500, {'manifest.json', *(r['file'] for r in inputs['images'])})
    private_directory(endpoint.OUTPUT, 0o500, {*PINS, '.container.cid', *(r['file'] for r in native['images'])})
    states = {str(p): snapshot(p) for folder in (endpoint.DATA, endpoint.OUTPUT) for p in (folder, *folder.iterdir())}
    rt.require(all(s[5] == s[6] == 0 and stat.S_IMODE(s[2]) == 0o400 for n, s in states.items() if Path(n).is_file()), 'Original root-owned400 leaves required')
    original_files = {str(p): rt.identity(p, 16 << 20) for folder in (endpoint.DATA, endpoint.OUTPUT) for p in folder.iterdir()}
    return projection, paths, files, dict(source=original, states=states, source_states=tree_state(old),
        markers={n: snapshot(old.parent/n) for n in ('revision', 'source-sha256')}, files=original_files)


def validate_manifest(value, revision):
    rt.require(type(value) is dict and set(value) == {'schema', 'export_revision', 'original_source_declaration', 'files'}
        and value['schema'] == SCHEMA and type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision)
        and value['export_revision'] == revision and value['original_source_declaration'] == DECLARATION
        and type(value['files']) is dict and set(value['files']) == NAMES, 'Exact98 public leaves required')
    for p in value['files'].values(): fixed_pin(p, 16 << 20)
    rt.require(value['files']['inputs/manifest.json'] == PUBLIC
        and value['files']['public-reference.json']['bytes'] <= MAX_CONTROL
        and sum(p['bytes'] for p in value['files'].values()) < MAXIMUM-(1 << 20), 'Complete128MiB payload bound required')
    return value


class BoundedWriter:
    def __init__(self, writer, deadline): self.writer, self.deadline, self.size = writer, deadline, 0
    def write(self, raw):
        check(self.deadline); rt.require(type(raw) is bytes and self.size+len(raw) <= MAXIMUM, 'Complete128MiB USTAR bound')
        self.size += len(raw); return self.writer.write(raw)


def pack(writer, manifest, paths, deadline):
    raw = encode(manifest); rt.require(len(raw) <= MAX_CONTROL and set(paths) == NAMES, 'Bounded pinned first control required')
    with tarfile.open(fileobj=BoundedWriter(writer, deadline), mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        for name in ('manifest.json', *sorted(paths)):
            check(deadline); row = tarfile.TarInfo(name); row.size = len(raw) if name == 'manifest.json' else manifest['files'][name]['bytes']; row.mode = 0o400
            if name == 'manifest.json': archive.addfile(row, io.BytesIO(raw))
            else:
                rt.require(rt.identity(paths[name], 16 << 20) == manifest['files'][name], 'Original source bytes changed before pack')
                with paths[name].open('rb') as stream: archive.addfile(row, stream)


def verify_archive(path, archive_pin, manifest_pin, revision, deadline):
    rt.require(rt.identity(path, MAXIMUM) == fixed_pin(archive_pin), 'Independent archive bytes differ'); fixed_pin(manifest_pin, MAX_CONTROL)
    table = []; manifest = None
    with path.open('rb') as stream:
        while True:
            check(deadline); header = stream.read(512); rt.require(len(header) == 512, 'Truncated USTAR header')
            if not any(header):
                tail_size = 0
                for tail in iter(lambda: stream.read(1 << 20), b''):
                    check(deadline); rt.require(not any(tail), 'Only zero USTAR tail required'); tail_size += len(tail)
                rt.require(tail_size >= 512 and tail_size % 512 == 0, 'Complete zero USTAR tail required'); break
            info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
            rt.require(info.tobuf(format=tarfile.USTAR_FORMAT) == header and info.type == tarfile.REGTYPE
                and not info.linkname and info.mode == 0o400 and info.uid == info.gid == info.mtime == 0
                and not info.uname and not info.gname and len(table) <= len(NAMES), 'Only canonical regular USTAR members required')
            expected_name = 'manifest.json' if not table else sorted(NAMES)[len(table)-1]
            expected = manifest_pin if not table else manifest['files'][expected_name]
            rt.require(info.name == expected_name and info.size == expected['bytes'], 'Exact ordered public member required')
            offset = stream.tell(); digest = hashlib.sha256(); remaining = info.size; first = bytearray()
            while remaining:
                check(deadline); data = stream.read(min(1 << 20, remaining)); rt.require(data, 'Truncated USTAR payload')
                digest.update(data); remaining -= len(data)
                if not table: first.extend(data)
            rt.require(dict(bytes=info.size, sha256=digest.hexdigest()) == expected, 'Member SHA differs')
            if not table: manifest = validate_manifest(rt.strict(first), revision)
            padding = stream.read((-info.size) % 512); rt.require(len(padding) == (-info.size) % 512 and not any(padding), 'Exact zero member padding required')
            table.append((info.name, offset, info.size))
    rt.require(len(table) == 99 and rt.identity(path, MAXIMUM) == archive_pin, 'Complete immutable99-member USTAR required')
    return manifest, table


def install(path, stage, manifest, table, deadline):
    private_directory(DEST.parent, 0o700); rt.canonical(stage); rt.canonical(DEST)
    rt.require(stage.parent == DEST.parent and not DEST.exists() and not DEST.is_symlink(), 'Fresh no-overwrite replica required')
    stage.mkdir(mode=0o700); owner = snapshot(stage); owned = {}
    try:
        for name in ('inputs', 'banks'): (stage/name).mkdir(mode=0o700); owned[name] = snapshot(stage/name)
        with path.open('rb') as stream:
            for name, offset, size in table[1:]:
                check(deadline); rt.require(name in NAMES and name not in owned and size == manifest['files'][name]['bytes'], 'Verified exact member table required')
                stream.seek(offset); target = stage/name
                with target.open('xb') as output:
                    os.fchmod(output.fileno(), 0o400); owned[name] = snapshot(target); remaining = size
                    while remaining:
                        check(deadline); data = stream.read(min(1 << 20, remaining)); rt.require(data, 'Truncated installed payload'); output.write(data); remaining -= len(data)
                    output.flush(); os.fsync(output.fileno())
                rt.require(rt.identity(target, 16 << 20) == manifest['files'][name], 'Installed original bytes differ')
        rt.require(set(owned) == {'inputs', 'banks', *NAMES} and snapshot(stage)[:3] == owner[:3]
            and {str(p.relative_to(stage)) for p in stage.rglob('*')} == set(owned)
            and all(snapshot(stage/n)[:2] == s[:2] and (stage/n).lstat().st_uid == os.getuid()
                and (stage/n).lstat().st_gid == os.getgid() and stat.S_IMODE((stage/n).lstat().st_mode) == (0o700 if n in ('inputs', 'banks') else 0o400)
                for n, s in owned.items()), 'Only complete originally owned leaves required')
        validate_projection(rt.pinned(stage/'public-reference.json', manifest['files']['public-reference.json'], MAX_CONTROL), manifest['files'])
        for name in ('inputs', 'banks'): (stage/name).chmod(0o500); sync(stage/name)
        sync(stage); check(deadline); atomic.rename_noreplace(stage, DEST); sync(DEST.parent)
    except BaseException:
        if stage.exists():
            rt.require(snapshot(stage)[:3] == owner[:3] and {str(p.relative_to(stage)) for p in stage.rglob('*')} == set(owned), 'Foreign stage never cleaned')
            for name in ('inputs', 'banks'):
                folder = stage/name
                if folder.exists(): rt.require(snapshot(folder)[:2] == owned[name][:2], 'Foreign stage directory'); folder.chmod(0o700)
            for name in sorted(set(owned)-{'inputs', 'banks'}):
                p = rt.canonical(stage/name); rt.require(snapshot(p)[:2] == owned[name][:2] and p.lstat().st_nlink == 1, 'Only owned partial leaf cleanup'); p.unlink()
            for name in ('inputs', 'banks'):
                if (stage/name).exists(): (stage/name).rmdir()
            stage.rmdir()
        raise


def installed(manifest):
    private_directory(DEST, 0o700, {'inputs', 'banks', 'public-reference.json'})
    for name in ('inputs', 'banks'): private_directory(DEST/name, 0o500, {Path(n).name for n in NAMES if n.startswith(name+'/')})
    files = {}
    for name in NAMES:
        p = DEST/name; s = rt.canonical(p).lstat()
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Root-owned readonly replica leaves required')
        files[str(p)] = rt.identity(p, 16 << 20)
    rt.require({str(DEST/n): p for n, p in manifest['files'].items()} == files, 'All98 replica byte pins required')
    projection = validate_projection(rt.pinned(DEST/'public-reference.json', manifest['files']['public-reference.json'], MAX_CONTROL), manifest['files'])
    rt.require(rt.pinned(DEST/'inputs/manifest.json', PUBLIC, 1 << 20) == projection['inputs'], 'Public manifest/projection differs')
    return dict(files=files, states=tree_state(DEST), projection=projection)


def export_receipt(raw, expected, revision):
    rt.require(pin(raw) == fixed_pin(expected, MAX_CONTROL), 'Independent export receipt required'); value = rt.strict(raw)
    rt.require(value['schema'] == SCHEMA and value['phase'] == 'export' and value['status'] == 'pass'
        and value['producer_revision'] == value['source_binding']['producer_revision'] == revision
        and value['original_source_declaration'] == DECLARATION and value['files'] == 98 and value['budget_seconds'] == BUDGET
        and value['source_inputs_rehashed_after'] is value['outputs_sealed'] is True
        and re.fullmatch(r'"[0-9A-Za-z-]{1,128}"', value['blob_etag'])
        and all(value[k] is False for k in ('models_loaded', 'GPU_used', 'reference_metadata_read', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified', 'adoption'))
        and not any(k in value for k in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed', 'failure_stage')), 'Complete exclusive byte export required')
    fixed_pin(value['archive_identity']); fixed_pin(value['manifest_identity'], MAX_CONTROL); return value


def authenticate_receiver(code, replica_revision, receipt_pin):
    """HOST ONLY. No sender source/runtime is claimed live on the receiving VM."""
    old = ROOT/'jobs'/replica_revision/ENTRY/'code'; binding = rt.source(ROOT, old, replica_revision, ENTRY, HELPERS)
    rt.require(all(rt.identity(code/n, 2 << 20, empty=True) == p for n, p in binding['helpers'].items()), 'Exact receiver helper source required')
    out = ROOT/f'results/vcoco-full-public-replica-import-{replica_revision}'; private_directory(out, 0o500, {'report.json', 'manifest.json', 'export-receipt.json'})
    before = tree_state(out); report = rt.pinned(out/'report.json', receipt_pin, MAX_CONTROL)
    rt.require(report['schema'] == SCHEMA and report['phase'] == 'import' and report['status'] == 'pass'
        and report['producer_revision'] == replica_revision and report['source_binding'] == binding and report['files'] == 98
        and report['source_inputs_rehashed_after'] is report['outputs_sealed'] is report['blob_cleanup_verified'] is True
        and report['single_etag_DELETE_202'] is True and report['delete_attempts'] == 1 and report['archive_removed'] is True
        and report['replica_directory'] == str(DEST) and report['original_source_declaration'] == DECLARATION
        and report['budget_seconds'] == BUDGET
        and report['original_source_on_receiver_live_verified'] is report['original_runtime_on_receiver_live_verified'] is False
        and all(report[k] is False for k in ('models_loaded', 'GPU_used', 'reference_metadata_read', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified', 'adoption'))
        and not any(k in report for k in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed', 'failure_stage')), 'Actual sealed98-file import required')
    for p in out.iterdir():
        s = p.lstat(); rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Exact readonly technical leaves required')
    rt.require(rt.identity(out/'export-receipt.json', MAX_CONTROL) == fixed_pin(report['export_receipt_identity'], MAX_CONTROL), 'Bounded original export receipt before read')
    raw = (out/'export-receipt.json').read_bytes(); export = export_receipt(raw, report['export_receipt_identity'], report['export_revision'])
    manifest = validate_manifest(rt.pinned(out/'manifest.json', report['manifest_identity'], MAX_CONTROL), report['export_revision'])
    rt.require(report['archive_identity'] == export['archive_identity'] and report['manifest_identity'] == export['manifest_identity'], 'Independent exported artifact pins differ')
    value = installed(manifest); technical = {str(out/'report.json'): receipt_pin, str(out/'manifest.json'): report['manifest_identity'], str(out/'export-receipt.json'): report['export_receipt_identity']}
    rt.require({n: rt.identity(n, MAX_CONTROL) for n in technical} == technical and tree_state(out) == before
        and installed(manifest) == value and rt.source(ROOT, old, replica_revision, ENTRY, HELPERS) == binding, 'Complete receiver bytes/source changed')
    return dict(images=value['projection']['inputs']['images'], banks=value['projection']['banks'],
        projection_path=str(DEST/'public-reference.json'), projection_identity=manifest['files']['public-reference.json'],
        files={**value['files'], **technical}, states={**value['states'], **before}, import_source=binding, import_identity=receipt_pin,
        original_source_declaration=DECLARATION, sender_source_live_verified=False, sender_runtime_live_verified=False)


class OwnedDownload:
    def __init__(self, path, owner): self.path, self.owner = path, owner
    def __fspath__(self): return str(self.path)
    def open(self, mode):
        rt.require(mode == 'xb' and not self.owner, 'Exclusive fresh download required'); stream = self.path.open(mode)
        s = os.fstat(stream.fileno()); self.owner.append((s.st_dev, s.st_ino, s.st_uid)); return stream


def run(args, code, revision):
    started = time.monotonic(); deadline = started+BUDGET
    def expired(*_): raise TimeoutError('Inclusive full public replica deadline')
    handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}; signal.setitimer(signal.ITIMER_REAL, BUDGET)
    out = ROOT/f'results/vcoco-full-public-replica-{args.phase}-{revision}'; rt.canonical(out)
    rt.require(not out.exists() and not out.is_symlink(), 'Fresh technical namespace required'); out.mkdir(mode=0o700)
    s = out.lstat(); owner = (s.st_dev, s.st_ino, s.st_uid); before = original = manifest = replica_before = None
    archive = out/'archive.tar'; archive_owner = []; allowed = set(); stage = 'source'
    report = dict(schema=SCHEMA, phase=args.phase, status='fail', producer_revision=revision, original_source_declaration=DECLARATION,
        files=98, budget_seconds=BUDGET, models_loaded=False, GPU_used=False, reference_metadata_read=False, RGB_NPZ_decoded=False,
        quality_verified=False, ownership_verified=False, adoption=False, source_inputs_rehashed_after=False, outputs_sealed=False,
        original_acquisition_live_reauthenticated=False, original_source_on_receiver_live_verified=False,
        original_runtime_on_receiver_live_verified=False, delete_attempts=0, single_etag_DELETE_202=False, blob_cleanup_verified=False)
    try:
        before = source(code, revision); report['source_binding'] = before['binding']; stage = 'peer'; transport.verify_azure_peer(args.phase)
        stage = 'blob'; export_revision = revision if args.phase == 'export' else args.export_revision
        blob = transport.Blob('https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+export_revision+'.tar', export_revision, managed_identity=True)
        if args.phase == 'export':
            stage = 'sender'; projection, paths, files, original = sender_inputs(code, before['binding'], deadline)
            stage = 'projection'; rt.write(out/'public-reference.json', encode(projection)); allowed.add('public-reference.json'); paths['public-reference.json'] = out/'public-reference.json'
            manifest = validate_manifest(dict(schema=SCHEMA, export_revision=revision, original_source_declaration=DECLARATION, files=files), revision)
            stage = 'pack'; writer = transport.BlockWriter(blob); pack(writer, manifest, paths, deadline)
            stage = 'commit'; archive_pin = fixed_pin(writer.finish()); check(deadline)
            stage = 'head'
            with blob.request('HEAD') as response:
                etag = response.headers.get('ETag', ''); rt.require(response.status == 200 and response.headers.get('Content-Length') == str(archive_pin['bytes'])
                    and re.fullmatch(r'"[0-9A-Za-z-]{1,128}"', etag), 'Exclusive committed archive HEAD required')
            report.update(archive_identity=archive_pin, manifest_identity=pin(encode(manifest)), blob_etag=etag)
        else:
            stage = 'receipt'; raw = base64.b64decode(args.export_receipt_base64, validate=True)
            rt.require(base64.b64encode(raw).decode() == args.export_receipt_base64, 'Canonical export receipt encoding required')
            export = export_receipt(raw, args.export_receipt_pin, export_revision)
            rt.require(export['archive_identity'] == args.archive_pin and export['manifest_identity'] == args.manifest_pin, 'Independent archive/control pins differ')
            rt.write(out/'export-receipt.json', raw); allowed.add('export-receipt.json')
            private_directory(DEST.parent, 0o700); rt.require(not DEST.exists() and not DEST.is_symlink(), 'Never overwrite replica')
            stage = 'download'; transport.download(blob, OwnedDownload(archive, archive_owner), args.archive_pin); archive.chmod(0o400)
            stage = 'archive'; manifest, table = verify_archive(archive, args.archive_pin, args.manifest_pin, export_revision, deadline)
            rt.write(out/'manifest.json', encode(manifest)); allowed.add('manifest.json')
            stage = 'install'; install(archive, DEST.parent/(DEST.name+'.stage-'+revision), manifest, table, deadline); replica_before = installed(manifest)
            report.update(export_revision=export_revision, archive_identity=args.archive_pin, manifest_identity=args.manifest_pin,
                export_receipt_identity=args.export_receipt_pin, blob_etag=export['blob_etag'], replica_directory=str(DEST))
        stage = 'post'; check(deadline); report['status'] = 'pass'
    except BaseException as exc: report.update(error_type=error(exc), failure_stage=stage)
    finally:
        try:
            rt.require(before is not None and source(code, revision) == before, 'Whole current source changed or unavailable')
            if args.phase == 'export':
                again, _, files, proof = sender_inputs(code, before['binding'], deadline)
                rt.require(original is not None and proof == original and files == manifest['files'] and pin(encode(again)) == files['public-reference.json']
                    and rt.identity(out/'public-reference.json', MAX_CONTROL) == files['public-reference.json'], 'Original source/all public inputs changed')
            else: rt.require(replica_before is not None and installed(manifest) == replica_before, 'All installed byte pins changed')
            report['source_inputs_rehashed_after'] = True; check(deadline)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc), post_failure_stage='post')
        if archive.exists():
            try:
                s = archive.lstat(); rt.require(archive_owner and (s.st_dev, s.st_ino, s.st_uid) == archive_owner[0], 'Foreign archive never removed')
                rt.identity(archive, MAXIMUM, readonly=False, empty=True); archive.unlink()
            except BaseException as exc: report.update(status='fail', cleanup_error_type=error(exc)); allowed.add('archive.tar')
        report['archive_removed'] = not archive.exists()
        def finish():
            check(deadline); rt.require(installed(manifest) == replica_before and source(code, revision) == before, 'Sealed full replica/source beforeDELETE')
            report['delete_attempts'] += 1
            with blob.request('DELETE', headers={'If-Match': report['blob_etag']}) as response: rt.require(response.status == 202, 'One exact ETag DELETE202 required')
            report['single_etag_DELETE_202'] = True; check(deadline)
            rt.require(installed(manifest) == replica_before and source(code, revision) == before, 'Complete replica/source afterDELETE')
        try:
            publication.publish(out, report, deadline, started, owner, allowed, finish if args.phase == 'import' else None,
                encode=encode, identity=rt.identity, snapshot=snapshot, require=rt.require, check=check, sync=sync, error=error,
                maximum=MAXIMUM, report_maximum=MAX_CONTROL)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
    return report


def arguments(argv):
    flags = ('--phase', '--export-revision', '--export-receipt-base64', '--archive-bytes', '--archive-sha256',
        '--manifest-bytes', '--manifest-sha256', '--export-receipt-bytes', '--export-receipt-sha256')
    rt.require(all(sum(token.split('=', 1)[0] == flag for token in argv) <= 1 for flag in flags), 'No duplicated independent controls')
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--phase', choices=('export', 'import'), required=True)
    parser.add_argument('--export-revision'); parser.add_argument('--export-receipt-base64')
    for name in ('archive', 'manifest', 'export-receipt'): parser.add_argument('--'+name+'-bytes', type=int); parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args(argv)
    if args.phase == 'export': rt.require(all(v is None for k, v in vars(args).items() if k != 'phase'), 'Export has no input overrides')
    else:
        rt.require(type(args.export_revision) is str and re.fullmatch('[0-9a-f]{40}', args.export_revision)
            and type(args.export_receipt_base64) is str and 0 < len(args.export_receipt_base64) <= 384 << 10, 'Bounded independent export controls required')
        for name, maximum in (('archive', MAXIMUM), ('manifest', MAX_CONTROL), ('export_receipt', MAX_CONTROL)):
            setattr(args, name+'_pin', fixed_pin(dict(bytes=getattr(args, name+'_bytes'), sha256=getattr(args, name+'_sha256')), maximum))
    return args


if __name__ == '__main__':
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.environ.get('WR_ROOT') == str(ROOT), 'Azure root stdlib-only transfer')
    os.umask(0o077)
    result = run(arguments(sys.argv[1:]), Path(os.environ.get('WR_CODE', '/invalid')), os.environ.get('WR_CODE_REVISION', ''))
    print(encode({k: result[k] for k in ('phase', 'status', 'files', 'outputs_sealed')}).decode(), end=''); raise SystemExit(0 if result['status'] == 'pass' else 1)

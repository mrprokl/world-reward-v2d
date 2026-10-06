"""One VM01→VM02 pose-only public49-leaf replica; no inference or RGB."""
import argparse
import base64
import hashlib
import io
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tarfile
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_full_public_replica as shared
import vcoco_full_pose_run as pose

rt, ROOT, encode = shared.rt, shared.ROOT, shared.encode
transport, atomic, publication = shared.transport, shared.atomic, shared.publication
pin, error, check, snapshot = shared.pin, shared.error, shared.check, shared.snapshot
sync, private_directory, tree_state = shared.sync, shared.private_directory, shared.tree_state
ENTRY = 'run_vcoco_full_pose_replica'
SCHEMA = 'world_reward.vcoco_full_pose_replica.v1'
PUBLIC_SCHEMA = 'world_reward.public_person_pose_reference.v1'
ANCHOR = Path('/srv')
PRIVATE_PARENT = ANCHOR/'world-reward-public-replicas'
DEST = PRIVATE_PARENT/'vcoco_full_pose_replica_v1'
POSE_REV = 'e2d8b5afa9eeaa8d0988cd527ff2e47e2d4e31b4'
DECLARATION = dict(producer_revision=POSE_REV, files=376, entries=381,
    closure_sha256='bc549f0a3841891f9990096e98a51561dbaa18fc398a7336add689f807171aea',
    source_XZ_sha256='d0e6277f951c7b3be3ecf8c89737373b95b12474fc39bccc1f74514ebfca39e4')
BUDGET, MAXIMUM, MAX_CONTROL = 300, 128 << 20, 256 << 10
NAMES = frozenset(('public-pose.json', *(f'banks/image_{i:06d}.npz' for i in range(48))))
ROW_KEYS = frozenset(('image_id', 'original_slot', 'acquired_ordinal', 'original_frame_index', 'image_size',
    'file', 'identity', 'arrays', 'endpoint_bank_identity', 'source_person_ids', 'owl_patches', 'person_ids', 'persons'))
HELPERS = tuple(dict.fromkeys(('infra/vcoco_full_pose_replica.py', 'infra/run_vcoco_full_pose_replica.sh',
    *shared.HELPERS, *pose.helpers())))
STAGES = ('source','peer','blob','sender','projection','pack','commit','head','receipt_decode','receipt_validate',
    'incoming_pins','control_write','namespace','download','archive','install','post','publication')
FLAGS = ('models_loaded', 'GPU_used', 'reference_metadata_read', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified', 'adoption')
fixed_pin = shared.fixed_pin


def bootstrap_parent(deadline, *, expected=None, verify_only=False):
    """Create only the fixed0700 parent; never repair/remove any existing path."""
    check(deadline)
    rt.require(PRIVATE_PARENT == ANCHOR/'world-reward-public-replicas'
        and DEST == PRIVATE_PARENT/'vcoco_full_pose_replica_v1' and type(verify_only) is bool, 'Only fixed anchored receiver namespace')
    def identity(s): return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid)
    ancestors = {}
    for path in (ANCHOR,*ANCHOR.parents):
        s = rt.canonical(path).lstat()
        rt.require(stat.S_ISDIR(s.st_mode) and s.st_uid == s.st_gid == 0 and not s.st_mode & 0o022,
            'Canonical root-owned non-group/world-writable ancestors')
        ancestors[path] = identity(s)
    rt.canonical(PRIVATE_PARENT); flags = os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    anchor_fd = os.open(ANCHOR,flags); parent_fd = None
    try:
        rt.require(identity(os.fstat(anchor_fd)) == ancestors[ANCHOR], 'Original anchor inode')
        try: parent_fd = os.open(PRIVATE_PARENT.name,flags,dir_fd=anchor_fd)
        except FileNotFoundError:
            rt.require(expected is None and not verify_only, 'Original private parent disappeared')
            os.mkdir(PRIVATE_PARENT.name,mode=0o700,dir_fd=anchor_fd)
            parent_fd = os.open(PRIVATE_PARENT.name,flags,dir_fd=anchor_fd)
        value = identity(os.fstat(parent_fd))
        rt.require(stat.S_ISDIR(value[2]) and stat.S_IMODE(value[2]) == 0o700 and value[3] == value[4] == 0
            and (expected is None or value == expected), 'Exact original root0700 private parent')
        if not verify_only: os.fsync(parent_fd); os.fsync(anchor_fd)
        check(deadline)
        rt.require(identity(rt.canonical(PRIVATE_PARENT).lstat()) == value
            and all(identity(rt.canonical(path).lstat()) == before for path,before in ancestors.items()),
            'Private parent/ancestor inode or owner/mode race')
        return value
    finally:
        if parent_fd is not None: os.close(parent_fd)
        os.close(anchor_fd)


def source(code, revision):
    binding = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    for module, name in ((sys.modules[__name__], HELPERS[0]), (shared, 'infra/vcoco_full_public_replica.py'),
                         (pose, 'infra/vcoco_full_pose_run.py')):
        rt.require(Path(module.__file__).resolve() == code/name, 'Actual unchanged explicit helper origin')
    for p in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        s = rt.canonical(p).lstat(); rt.require(s.st_uid == s.st_gid == 0, 'Root-owned immutable source required')
    return dict(binding=binding, states=tree_state(code), markers={n: snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def validate_projection(value, files):
    rt.require(type(value) is dict and set(value) == {'schema', 'rows'} and value['schema'] == PUBLIC_SCHEMA
        and type(value['rows']) is list and len(value['rows']) == 48, 'Only complete public48 pose rows')
    ids = set()
    for i, row in enumerate(value['rows']):
        rt.require(type(row) is dict and set(row) == ROW_KEYS and type(row['image_id']) is str
            and re.fullmatch('[0-9a-f]{32}', row['image_id']) and row['image_id'] not in ids
            and all(type(row[k]) is int and row[k] == i for k in ('original_slot', 'acquired_ordinal'))
            and type(row['original_frame_index']) is int and row['original_frame_index'] == 0
            and row['file'] == f'image_{i:06d}.npz' and row['identity'] == files['banks/'+row['file']]
            and type(row['image_size']) is list and len(row['image_size']) == 2
            and all(type(v) is int and v > 0 for v in row['image_size']) and type(row['persons']) is int
            and 0 <= row['persons'] <= 900 and type(row['owl_patches']) is int and row['owl_patches'] == 3600,
            'All original opaque pose slots/counts/grid required')
        ids.add(row['image_id']); n = row['persons']; fixed_pin(row['endpoint_bank_identity'], 16 << 20)
        rt.require(type(row['person_ids']) is list and row['person_ids'] == row['source_person_ids']
            == [f'image:{row["image_id"]}/person/retained:{j:06d}' for j in range(n)], 'All unchanged native person IDs')
        shapes = dict(person_ids=[n], boxes_original_xyxy=[n,4], detector_scores=[n], keypoints_original_xy=[n,133,2],
            raw_scores=[n,133], native_valid=[n,133], in_original_image=[n,133], image_size=[2],
            original_frame_index=[], original_slot=[], acquired_ordinal=[])
        types = dict(boxes_original_xyxy='<f8', detector_scores='<f8', keypoints_original_xy='<f8', raw_scores='<f4',
            native_valid='|b1', in_original_image='|b1', image_size='<i8', original_frame_index='<i8', original_slot='<i8', acquired_ordinal='<i8')
        rt.require(type(row['arrays']) is dict and set(row['arrays']) == set(shapes), 'Exact full11 array metadata')
        for name, shape in shapes.items():
            a = row['arrays'][name]; rt.require(type(a) is dict and set(a) == {'shape','dtype','sha256'} and type(a['shape']) is list
                and all(type(v) is int and v >= 0 for v in a['shape']) and a['shape'] == shape
                and type(a['dtype']) is str and type(a['sha256']) is str and re.fullmatch('[0-9a-f]{64}', a['sha256'])
                and (a['dtype'] == types[name] if name in types else re.fullmatch('<U[1-9][0-9]*', a['dtype'])), 'Original native array ABI')
    rt.require(files['public-pose.json'] == pin(encode(value)), 'Exact public projection byte pin')
    return value


def sender_inputs(code, binding, pins, deadline):
    check(deadline); old = ROOT/'jobs'/POSE_REV/pose.ENTRY/'code'
    h, n, p = (rt.pinned(pose.OUTPUT/name, pins[name], 2 << 20) for name in ('report.json', 'native.json', 'proof.json'))
    original = rt.source(ROOT, old, POSE_REV, pose.ENTRY, pose.helpers())
    rt.require(original == h['source_binding'] and original['entries'] == DECLARATION['entries']
        and original['closure_sha256'] == DECLARATION['closure_sha256'] and sum(q.is_file() for q in old.rglob('*')) == DECLARATION['files']
        and (old.parent/'source-sha256').read_bytes() == (DECLARATION['source_XZ_sha256']+'\n').encode()
        and all(binding['helpers'][name] == original['helpers'][name] for name in pose.helpers()), 'Whole original376-file pose source')
    rt.require(h['schema'] == pose.SCHEMA and h['stage'] == 'public48_person_pose_host' and h['status'] == 'pass'
        and h['phase'] == 'complete' and h['producer_revision'] == POSE_REV and h['native_identity'] == pins['native.json']
        and h['native_exit_status'] == 0 and h['owned_cleanup_verified'] is h['source_inputs_assets_runtime_rehashed_after'] is h['outputs_sealed'] is True
        and all(h[k] is False for k in pose.FLAGS) and p['producer_revision'] == POSE_REV and p['image_id'] == h['image_id'] == pose.IMAGE
        and p['helpers'] == {name: original['helpers'][name] for name in pose.NATIVE_FILES} and p['markers'] == original['markers'], 'Actual sealed full48 pose PASS')
    transfer, _, _ = pose.host_modules(); inputs = transfer.authenticate_receiver(code, h['replica_revision'], h['replica_identity'])
    projection = rt.pinned(inputs['projection_path'], inputs['projection_identity'], 2 << 20)
    rt.require(p['public'] == dict(path=inputs['projection_path'], identity=inputs['projection_identity']), 'Same complete public endpoint lineage')
    pose.validate_native(n, projection, POSE_REV, pins['proof.json']); rt.require(h['images'] == n['images'] and h['persons'] == n['persons'], 'Full native rows/crops')
    model = pose.asset_context(code); image = pose.image_state(deadline)
    rt.require(model['source'] == h['dwpose_source'] and inputs['import_source'] == h['input_import_source'] and p['assets'] == model['assets'], 'Original qualified CPU assets/source')
    cid = pose.OUTPUT/'.container.cid'; raw = cid.read_bytes(); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Original sealed CID')
    command = pose.public.endpoint.original.command
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+raw.decode().strip()], deadline)
        and not command(['docker','ps','-aq','--no-trunc','--filter','name=^/world-reward-public48-pose-'+POSE_REV[:12]+'$'], deadline), 'Original CPU container absent')
    private_directory(pose.OUTPUT, 0o500, {*pins, '.container.cid', *(r['file'] for r in n['images'])})
    files = {str(q): rt.identity(q, 2 << 20) for q in pose.OUTPUT.iterdir()}
    rt.require(all(stat.S_IMODE(Path(q).stat().st_mode) == 0o400 and Path(q).stat().st_uid == 0 for q in files), 'Original400 readonly pose leaves')
    public = dict(schema=PUBLIC_SCHEMA, rows=[dict(r) for r in n['images']])
    paths = {'banks/'+r['file']: pose.OUTPUT/r['file'] for r in n['images']}
    payload = {name: rt.identity(path, 2 << 20) for name, path in paths.items()}; payload['public-pose.json'] = pin(encode(public)); validate_projection(public, payload)
    return public, paths, payload, dict(source=original, source_states=tree_state(old), files=files, states=tree_state(pose.OUTPUT),
        markers={name: snapshot(old.parent/name) for name in ('revision','source-sha256')}, inputs=inputs, models=model, image=image)


def validate_manifest(value, revision):
    rt.require(type(value) is dict and set(value) == {'schema','export_revision','original_source_declaration','pose_receipt_pins','files'}
        and value['schema'] == SCHEMA and re.fullmatch('[0-9a-f]{40}', str(revision)) and value['export_revision'] == revision
        and value['original_source_declaration'] == DECLARATION and type(value['pose_receipt_pins']) is dict
        and set(value['pose_receipt_pins']) == {'report.json','native.json','proof.json'}
        and type(value['files']) is dict and set(value['files']) == NAMES, 'Exact49 public-only pose payload')
    for p in value['pose_receipt_pins'].values(): fixed_pin(p, 2 << 20)
    for name, p in value['files'].items(): fixed_pin(p, MAX_CONTROL if name == 'public-pose.json' else 2 << 20)
    rt.require(sum(p['bytes'] for p in value['files'].values()) < MAXIMUM-(1 << 20), 'Bounded unchanged pose payload')
    return value


def pack(writer, manifest, paths, deadline):
    raw = encode(manifest); rt.require(len(raw) <= MAX_CONTROL and set(paths) == NAMES, 'Exact first metadata/member allowlist')
    with tarfile.open(fileobj=shared.BoundedWriter(writer, deadline), mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        for name in ('manifest.json', *sorted(paths)):
            check(deadline); row = tarfile.TarInfo(name); row.size = len(raw) if name == 'manifest.json' else manifest['files'][name]['bytes']; row.mode = 0o400
            if name == 'manifest.json': archive.addfile(row, io.BytesIO(raw))
            else:
                rt.require(rt.identity(paths[name], 2 << 20) == manifest['files'][name], 'Original pose bytes changed before pack')
                with paths[name].open('rb') as stream: archive.addfile(row, stream)


def verify_archive(path, archive_pin, manifest_pin, revision, deadline):
    rt.require(rt.identity(path, MAXIMUM) == fixed_pin(archive_pin), 'Independent original archive pin'); fixed_pin(manifest_pin, MAX_CONTROL)
    table, manifest = [], None
    with path.open('rb') as stream:
        while True:
            check(deadline); header = stream.read(512); rt.require(len(header) == 512, 'Complete USTAR header')
            if not any(header):
                tail_size = 0
                for tail in iter(lambda: stream.read(1 << 20), b''):
                    check(deadline); rt.require(not any(tail), 'Only zero USTAR tail'); tail_size += len(tail)
                rt.require(tail_size >= 512 and tail_size % 512 == 0, 'Complete USTAR terminator'); break
            rt.require(len(table) <= len(NAMES), 'No surplus archive members'); info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
            rt.require(info.tobuf(format=tarfile.USTAR_FORMAT) == header and info.type == tarfile.REGTYPE and not info.linkname
                and info.mode == 0o400 and info.uid == info.gid == info.mtime == 0 and not info.uname and not info.gname, 'Canonical regular USTAR only')
            name = 'manifest.json' if not table else sorted(NAMES)[len(table)-1]; expected = manifest_pin if not table else manifest['files'][name]
            rt.require(info.name == name and info.size == expected['bytes'], 'Exact ordered pinned member')
            offset, remaining, digest, first = stream.tell(), info.size, hashlib.sha256(), bytearray()
            while remaining:
                check(deadline); data = stream.read(min(1 << 20, remaining)); rt.require(data, 'Complete bounded payload'); remaining -= len(data); digest.update(data)
                if not table: first.extend(data)
            rt.require(dict(bytes=info.size, sha256=digest.hexdigest()) == expected, 'Member hash differs')
            if not table: manifest = validate_manifest(rt.strict(first), revision)
            padding = stream.read((-info.size) % 512); rt.require(len(padding) == (-info.size) % 512 and not any(padding), 'Exact zero padding')
            table.append((info.name, offset, info.size))
    rt.require(len(table) == 50 and rt.identity(path, MAXIMUM) == archive_pin, 'Complete immutable50-member archive')
    return manifest, table


def install(path, stage, manifest, table, deadline):
    private_directory(DEST.parent, 0o700); rt.canonical(stage); rt.canonical(DEST)
    rt.require(stage.parent == DEST.parent and not DEST.exists() and not DEST.is_symlink(), 'Fresh atomic receiver namespace')
    stage.mkdir(mode=0o700); owner = snapshot(stage); owned = {}
    try:
        (stage/'banks').mkdir(mode=0o700); owned['banks'] = snapshot(stage/'banks')
        with path.open('rb') as stream:
            for name, offset, size in table[1:]:
                check(deadline); rt.require(name in NAMES and name not in owned and size == manifest['files'][name]['bytes'], 'Verified exact member table')
                stream.seek(offset); target = stage/name
                with target.open('xb') as output:
                    os.fchmod(output.fileno(), 0o400); owned[name] = snapshot(target); remaining = size
                    while remaining:
                        check(deadline); data = stream.read(min(1 << 20, remaining)); rt.require(data, 'Complete installed bytes'); output.write(data); remaining -= len(data)
                    output.flush(); os.fsync(output.fileno())
                rt.require(rt.identity(target, 2 << 20) == manifest['files'][name], 'Installed bytes equal original')
        rt.require(set(owned) == {'banks', *NAMES} and snapshot(stage)[:3] == owner[:3]
            and {str(p.relative_to(stage)) for p in stage.rglob('*')} == set(owned)
            and all(snapshot(stage/n)[:2] == s[:2] and (stage/n).lstat().st_uid == os.getuid() and (stage/n).lstat().st_gid == os.getgid()
                and stat.S_IMODE((stage/n).lstat().st_mode) == (0o700 if n == 'banks' else 0o400) for n,s in owned.items()), 'Complete originally owned stage')
        validate_projection(rt.pinned(stage/'public-pose.json', manifest['files']['public-pose.json'], MAX_CONTROL), manifest['files'])
        (stage/'banks').chmod(0o500); sync(stage/'banks'); sync(stage); check(deadline); atomic.rename_noreplace(stage, DEST); sync(DEST.parent)
    except BaseException:
        if stage.exists():
            rt.require(snapshot(stage)[:3] == owner[:3] and {str(p.relative_to(stage)) for p in stage.rglob('*')} == set(owned), 'Foreign stage never removed')
            if (stage/'banks').exists(): rt.require(snapshot(stage/'banks')[:2] == owned['banks'][:2], 'Foreign banks'); (stage/'banks').chmod(0o700)
            for name in sorted(set(owned)-{'banks'}):
                p = rt.canonical(stage/name); rt.require(snapshot(p)[:2] == owned[name][:2] and p.lstat().st_nlink == 1, 'Only owned partial leaves'); p.unlink()
            if (stage/'banks').exists(): (stage/'banks').rmdir()
            stage.rmdir()
        raise


def installed(manifest):
    private_directory(DEST, 0o700, {'banks','public-pose.json'}); private_directory(DEST/'banks', 0o500, {Path(n).name for n in NAMES if n.startswith('banks/')})
    files = {str(DEST/name): rt.identity(DEST/name, 2 << 20) for name in NAMES}
    rt.require(files == {str(DEST/name): p for name,p in manifest['files'].items()}
        and all(Path(n).stat().st_uid == Path(n).stat().st_gid == 0 and stat.S_IMODE(Path(n).stat().st_mode) == 0o400 for n in files), 'Complete49 root-owned readonly leaves')
    public = validate_projection(rt.pinned(DEST/'public-pose.json', manifest['files']['public-pose.json'], MAX_CONTROL), manifest['files'])
    return dict(files=files, states=tree_state(DEST), projection=public)


def control_write(out, name, raw, allowed):
    with (out/name).open('xb') as stream:
        allowed.add(name); os.fchmod(stream.fileno(),0o400)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def export_receipt(raw, expected, revision):
    rt.require(pin(raw) == fixed_pin(expected, MAX_CONTROL), 'Independently pinned original export receipt'); value = rt.strict(raw)
    rt.require(value['schema'] == SCHEMA and value['phase'] == 'export' and value['status'] == 'pass'
        and value['producer_revision'] == value['source_binding']['producer_revision'] == revision
        and value['original_source_declaration'] == DECLARATION and value['files'] == 49 and value['budget_seconds'] == BUDGET
        and value['source_inputs_rehashed_after'] is value['outputs_sealed'] is True and all(value[k] is False for k in FLAGS)
        and value['sender_source_live_verified'] is value['sender_runtime_live_verified'] is True
        and re.fullmatch(r'"[0-9A-Za-z-]{1,128}"', value['blob_etag'])
        and not any(k in value for k in ('error_type','post_error_type','cleanup_error_type','publication_failed','failure_stage','post_failure_stage')), 'Complete sealed pose-only export')
    for p in value['pose_receipt_pins'].values(): fixed_pin(p, 2 << 20)
    rt.require(set(value['pose_receipt_pins']) == {'report.json','native.json','proof.json'}, 'All actual independent pose receipt pins')
    fixed_pin(value['archive_identity']); fixed_pin(value['manifest_identity'], MAX_CONTROL); return value


def authenticate_receiver(code, replica_revision, receipt_pin):
    """HOST ONLY; caller compares original pose pins. No sender is read on VM02."""
    parent_before = bootstrap_parent(time.monotonic()+10,verify_only=True)
    old = ROOT/'jobs'/replica_revision/ENTRY/'code'; binding = rt.source(ROOT, old, replica_revision, ENTRY, HELPERS)
    rt.require(all(rt.identity(code/n, 2 << 20, empty=True) == p for n,p in binding['helpers'].items()), 'Original receiver helper bytes')
    out = ROOT/f'results/vcoco-full-pose-replica-import-{replica_revision}'; private_directory(out, 0o500, {'report.json','manifest.json','export-receipt.json'})
    before = tree_state(out); report = rt.pinned(out/'report.json', receipt_pin, MAX_CONTROL)
    rt.require(report['schema'] == SCHEMA and report['phase'] == 'import' and report['status'] == 'pass' and report['producer_revision'] == replica_revision
        and report['source_binding'] == binding and report['files'] == 49 and report['original_source_declaration'] == DECLARATION
        and report['source_inputs_rehashed_after'] is report['outputs_sealed'] is report['blob_cleanup_verified'] is True
        and report['single_etag_DELETE_202'] is True and report['delete_attempts'] == 1 and report['archive_removed'] is True
        and report['replica_directory'] == str(DEST) and report['budget_seconds'] == BUDGET
        and report['sender_source_live_verified'] is report['sender_runtime_live_verified'] is False
        and all(report[k] is False for k in FLAGS) and not any(k in report for k in ('error_type','post_error_type','cleanup_error_type','publication_failed','failure_stage','post_failure_stage')), 'Actual distinct pose import PASS')
    for p in out.iterdir(): rt.require(p.stat().st_uid == p.stat().st_gid == 0 and stat.S_IMODE(p.stat().st_mode) == 0o400, 'Sealed technical leaves')
    ep = report['export_receipt_identity']; rt.pinned(out/'export-receipt.json', ep, MAX_CONTROL); export_raw = (out/'export-receipt.json').read_bytes()
    export = export_receipt(export_raw, ep, report['export_revision']); manifest = validate_manifest(rt.pinned(out/'manifest.json', report['manifest_identity'], MAX_CONTROL), report['export_revision'])
    rt.require(report['archive_identity'] == export['archive_identity'] and report['manifest_identity'] == export['manifest_identity']
        and report['pose_receipt_pins'] == export['pose_receipt_pins'] == manifest['pose_receipt_pins'], 'Original independent export/payload bindings')
    value = installed(manifest); technical = {str(out/'report.json'): receipt_pin, str(out/'manifest.json'): report['manifest_identity'], str(out/'export-receipt.json'): ep}
    rt.require({n: rt.identity(n, MAX_CONTROL) for n in technical} == technical and tree_state(out) == before
        and installed(manifest) == value and rt.source(ROOT, old, replica_revision, ENTRY, HELPERS) == binding
        and bootstrap_parent(time.monotonic()+10,expected=parent_before,verify_only=True) == parent_before,
        'Complete receiver source/files/private parent unchanged')
    return dict(rows=value['projection']['rows'], projection_path=str(DEST/'public-pose.json'), projection_identity=manifest['files']['public-pose.json'],
        files={**value['files'], **technical}, states={**value['states'], **before}, import_source=binding, import_identity=receipt_pin,
        original_pose_revision=POSE_REV, pose_receipt_pins=report['pose_receipt_pins'], original_source_declaration=DECLARATION,
        sender_source_live_verified=False, sender_runtime_live_verified=False)


def run(args, code, revision):
    started = time.monotonic(); deadline = started+BUDGET
    def expired(*_): raise TimeoutError('Inclusive pose byte replica')
    handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; signal.setitimer(signal.ITIMER_REAL, BUDGET)
    out = ROOT/f'results/vcoco-full-pose-replica-{args.phase}-{revision}'; rt.canonical(out); rt.require(not out.exists() and not out.is_symlink(), 'Fresh technical namespace')
    out.mkdir(mode=0o700); s = out.lstat(); owner = (s.st_dev,s.st_ino,s.st_uid)
    before = original = manifest = replica_before = parent_before = None; archive = out/'archive.tar'; archive_owner = []; allowed = set(); stage = 'source'
    report = dict(schema=SCHEMA, phase=args.phase, status='fail', producer_revision=revision, original_source_declaration=DECLARATION,
        pose_receipt_pins=args.pose_pins, files=49, budget_seconds=BUDGET, source_inputs_rehashed_after=False, outputs_sealed=False,
        sender_source_live_verified=False, sender_runtime_live_verified=False, delete_attempts=0, single_etag_DELETE_202=False, blob_cleanup_verified=False, **{k:False for k in FLAGS})
    try:
        before = source(code, revision); report['source_binding'] = before['binding']; stage = 'peer'
        transport.verify_azure_peer('import' if args.phase == 'export' else 'export') # explicit reverse direction, no mutation
        export_revision = revision if args.phase == 'export' else args.export_revision; stage = 'blob'
        blob = transport.Blob('https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+export_revision+'.tar', export_revision, managed_identity=True)
        if args.phase == 'export':
            stage = 'sender'; public, paths, files, original = sender_inputs(code, before['binding'], args.pose_pins, deadline)
            stage = 'projection'; control_write(out,'public-pose.json',encode(public),allowed); paths['public-pose.json'] = out/'public-pose.json'
            manifest = validate_manifest(dict(schema=SCHEMA, export_revision=revision, original_source_declaration=DECLARATION, pose_receipt_pins=args.pose_pins, files=files), revision)
            stage = 'pack'; writer = transport.BlockWriter(blob); pack(writer, manifest, paths, deadline); stage = 'commit'; archive_pin = fixed_pin(writer.finish()); check(deadline)
            stage = 'head'
            with blob.request('HEAD') as response:
                etag = response.headers.get('ETag',''); rt.require(response.status == 200 and response.headers.get('Content-Length') == str(archive_pin['bytes']) and re.fullmatch(r'"[0-9A-Za-z-]{1,128}"', etag), 'Committed private archive HEAD')
            report.update(archive_identity=archive_pin, manifest_identity=pin(encode(manifest)), blob_etag=etag, sender_source_live_verified=True, sender_runtime_live_verified=True)
        else:
            stage = 'receipt_decode'; raw = base64.b64decode(args.export_receipt_base64, validate=True); rt.require(base64.b64encode(raw).decode() == args.export_receipt_base64, 'Canonical exact control encoding')
            stage = 'receipt_validate'
            export = export_receipt(raw, args.export_receipt_pin, export_revision)
            stage = 'incoming_pins'
            rt.require(export['archive_identity'] == args.archive_pin and export['manifest_identity'] == args.manifest_pin and export['pose_receipt_pins'] == args.pose_pins, 'All independent incoming pins')
            stage = 'control_write'; control_write(out,'export-receipt.json',raw,allowed)
            stage = 'namespace'; parent_before = bootstrap_parent(deadline)
            rt.require(not DEST.exists() and not DEST.is_symlink(), 'Never replace receiver data')
            stage = 'download'; transport.download(blob, shared.OwnedDownload(archive, archive_owner), args.archive_pin); archive.chmod(0o400)
            stage = 'archive'; manifest, table = verify_archive(archive, args.archive_pin, args.manifest_pin, export_revision, deadline)
            rt.require(manifest['pose_receipt_pins'] == args.pose_pins, 'Same independently pinned native pose source')
            control_write(out,'manifest.json',encode(manifest),allowed); stage = 'install'
            rt.require(bootstrap_parent(deadline,expected=parent_before) == parent_before, 'Original private parent before install')
            install(archive, DEST.parent/(DEST.name+'.stage-'+revision), manifest, table, deadline); replica_before = installed(manifest)
            report.update(export_revision=export_revision, archive_identity=args.archive_pin, manifest_identity=args.manifest_pin,
                export_receipt_identity=args.export_receipt_pin, blob_etag=export['blob_etag'], replica_directory=str(DEST))
        stage = 'post'; check(deadline); report['status'] = 'pass'
    except BaseException as exc: report.update(error_type=error(exc), failure_stage=stage)
    finally:
        try:
            rt.require(before is not None and source(code, revision) == before, 'Whole current source unchanged')
            if args.phase == 'export':
                again, _, files, proof = sender_inputs(code, before['binding'], args.pose_pins, deadline)
                rt.require(original is not None and proof == original and files == manifest['files'] and pin(encode(again)) == files['public-pose.json']
                    and rt.identity(out/'public-pose.json', MAX_CONTROL) == files['public-pose.json'], 'Whole original pose/public bytes unchanged')
            else:
                rt.require(parent_before is not None and bootstrap_parent(deadline,expected=parent_before) == parent_before,
                    'Original receiver private parent unchanged')
                rt.require(replica_before is not None and installed(manifest) == replica_before, 'All installed pose bytes unchanged')
            report['source_inputs_rehashed_after'] = True; check(deadline)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc), post_failure_stage='post')
        if archive.exists():
            try:
                s = archive.lstat(); rt.require(archive_owner and (s.st_dev,s.st_ino,s.st_uid) == archive_owner[0], 'Foreign archive never removed'); rt.identity(archive, MAXIMUM, readonly=False, empty=True); archive.unlink()
            except BaseException as exc: report.update(status='fail', cleanup_error_type=error(exc)); allowed.add('archive.tar')
        report['archive_removed'] = not archive.exists()
        def finish():
            check(deadline); rt.require(bootstrap_parent(deadline,expected=parent_before) == parent_before
                and installed(manifest) == replica_before and source(code, revision) == before, 'Sealed pose/source beforeDELETE')
            report['delete_attempts'] += 1
            try:
                with blob.request('DELETE', headers={'If-Match': report['blob_etag']}) as response: rt.require(response.status == 202, 'One exact ETag DELETE202')
                report['single_etag_DELETE_202'] = True
            finally:
                rt.require(bootstrap_parent(deadline,expected=parent_before) == parent_before
                    and installed(manifest) == replica_before and source(code, revision) == before, 'Full pose/source afterDELETE even failure'); check(deadline)
        try:
            publication.publish(out, report, deadline, started, owner, allowed, finish if args.phase == 'import' else None,
                encode=encode, identity=rt.identity, snapshot=snapshot, require=rt.require, check=check, sync=sync, error=error, maximum=MAXIMUM, report_maximum=MAX_CONTROL)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
    return report


def arguments(argv):
    flags = ('phase','export-revision','export-receipt-base64','archive-bytes','archive-sha256','manifest-bytes','manifest-sha256',
        'export-receipt-bytes','export-receipt-sha256', *(f'pose-{n}-{k}' for n in ('host','native','proof') for k in ('bytes','sha256')))
    rt.require(all(sum(token.split('=',1)[0] == '--'+f for token in argv) <= 1 for f in flags), 'No duplicate controls')
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--phase', choices=('export','import'), required=True); p.add_argument('--export-revision'); p.add_argument('--export-receipt-base64')
    for name in ('archive','manifest','export-receipt','pose-host','pose-native','pose-proof'): p.add_argument('--'+name+'-bytes',type=int); p.add_argument('--'+name+'-sha256')
    a = p.parse_args(argv); a.pose_pins = {file:fixed_pin(dict(bytes=getattr(a,'pose_'+kind+'_bytes'),sha256=getattr(a,'pose_'+kind+'_sha256')),2 << 20) for kind,file in (('host','report.json'),('native','native.json'),('proof','proof.json'))}
    controls = ('export_revision','export_receipt_base64',*(n+'_'+k for n in ('archive','manifest','export_receipt') for k in ('bytes','sha256')))
    if a.phase == 'export': rt.require(all(getattr(a,n) is None for n in controls), 'No export input overrides')
    else:
        rt.require(re.fullmatch('[0-9a-f]{40}',str(a.export_revision)) and type(a.export_receipt_base64) is str and 0 < len(a.export_receipt_base64) <= 384 << 10, 'Bounded independent receipt control')
        for name, maximum in (('archive',MAXIMUM),('manifest',MAX_CONTROL),('export_receipt',MAX_CONTROL)): setattr(a,name+'_pin',fixed_pin(dict(bytes=getattr(a,name+'_bytes'),sha256=getattr(a,name+'_sha256')),maximum))
    return a


if __name__ == '__main__':
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.environ.get('WR_ROOT') == str(ROOT), 'Azure root-only pose byte transfer')
    os.umask(0o077); result = run(arguments(sys.argv[1:]),Path(os.environ.get('WR_CODE','/invalid')),os.environ.get('WR_CODE_REVISION',''))
    print(encode({k:result[k] for k in ('phase','status','files','outputs_sealed')}).decode(),end=''); raise SystemExit(0 if result['status'] == 'pass' else 1)

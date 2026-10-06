"""Azure-only byte replica of the 16 already-qualified native HOI banks."""
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
import vcoco_hoi_observations as hoi
import vcoco_person_pose_observations as pose
import sealed_callback_publication as publication

public, rt, ROOT = hoi.public, hoi.rt, hoi.ROOT
transport, atomic = public.transport, public.atomic
ENTRY = 'run_vcoco_hoi_saved_replica'
SCHEMA = 'world_reward.vcoco_hoi_saved_replica.v1'
DEST = Path('/srv/world-reward-data/vcoco_hoi_saved_replica_v1')
REV = 'fc3c91bd9d156c501ed42abd9e342d1158883047'
CLOSURE = 'cc35bb23acd75b89e4ce9ac7f6580604464a76f25ba3138795069b456c8b98d7'
XZ = '45b4980b2e897aba769e3eabe9253a7d08caebf44c0a2dfa72ac4e10c8f51747'
PINS = {'host.json': dict(bytes=63341, sha256='c92e7282c7b73b1760881f4f709ac5bc6c060b4f1b343dbfe2c07bd14a911f09'),
        'model.json': dict(bytes=56785, sha256='3580f7beb7c3bb97a12eac1b4e18ec83b382271d874f78824782d373226bdc76'),
        'model_proof.json': dict(bytes=417398, sha256='468747fa1b2d0d8885e0b132f0013fa8b7ec86761f73a0a632576b0cb4a8cbf6')}
COMPLETION = ('21ac084f9bc83a7d7ec10dc726fcb1611ebae00c', dict(bytes=106667,
    sha256='9c8b31d86cf0d5b8a066030758f850de4f6a6bd9323799197dea0aa511a0344f'))
PUBLISHER = dict(bytes=5861, sha256='9fd62223432ac591b22bab23f916dcd47d9c67264ebfa8c5782a922b0f0e23ae')
BUDGET, MAXIMUM, MAX_MANIFEST = 180, 32 << 20, 256 << 10
NAMES = {*PINS, *(f'image_{i:06d}.npz' for i in range(16))}
DECLARATION = dict(producer_revision=REV, files=344, entries=349, closure_sha256=CLOSURE, source_XZ_sha256=XZ)
CURRENT_EXECUTABLES = frozenset(('infra/run_coco_endpoint_prepare.sh',
    'infra/run_keypoint_rgb_dwpose.sh', 'infra/run_rgb_endpoint_bank.sh'))
SENDER_EXECUTABLES = frozenset(('infra/run_coco_endpoint_prepare.sh', 'infra/run_rgb_endpoint_bank.sh'))
HELPERS = tuple(dict.fromkeys(('infra/vcoco_hoi_saved_replica.py', 'infra/run_vcoco_hoi_saved_replica.sh',
    'infra/sealed_callback_publication.py', *hoi.HELPERS, *pose.HELPERS)))
encode, pin, snapshot, error = public.encode, public.pin, public.snapshot, public.bank.error


def check(deadline):
    rt.require(type(deadline) in (int, float) and math.isfinite(deadline) and time.monotonic() < deadline, 'Inclusive180s replica deadline')


def private_directory(path, mode, names=None):
    rt.canonical(path); s = path.lstat()
    rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == mode and s.st_uid == s.st_gid == 0
        and (names is None or {p.name for p in path.iterdir()} == set(names)), 'Exact root-private namespace required')


def source_modes(code, executables):
    """Exact executable leaves from frozen Git archives, never chmod source."""
    rt.canonical(code)
    rt.require(all((code/name).is_file() for name in executables), 'Every original executable source leaf required')
    for p in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        rt.canonical(p); s = p.lstat()
        mode = 0o555 if stat.S_ISDIR(s.st_mode) or p in {code/name for name in executables} else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == mode
            and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1),
            'Exact immutable Git modes, regular leaves and root owner required')


def source(code, revision):
    value = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(Path(__file__).resolve() == code/HELPERS[0] and Path(hoi.__file__).resolve() == code/'infra/vcoco_hoi_observations.py'
        and Path(pose.__file__).resolve() == code/'infra/vcoco_person_pose_observations.py'
        and value['helpers']['infra/sealed_callback_publication.py'] == PUBLISHER, 'Actual unchanged imported helper origins required')
    source_modes(code, CURRENT_EXECUTABLES)
    return dict(binding=value, states=public.bank.source_state(code), markers={n: snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def validate_metadata(host, model, proof, files, endpoints):
    """Copied producer ABI + original endpoint association; never follow sender paths."""
    sb = host['source_binding']
    rt.require(host['schema'] == hoi.SCHEMA and host['stage'] == 'vcoco_hoi_observations_host' and host['status'] == model['status'] == 'pass'
        and host['producer_revision'] == model['producer_revision'] == REV and host['image_id'] == model['image_id'] == hoi.IMAGE
        and sb == proof['source'] and sb['producer_revision'] == REV and sb['entries'] == 349 and sb['closure_sha256'] == CLOSURE
        and all(sb['helpers'][n] == p for n, p in hoi.FROZEN.items()) and model['schema'] == hoi.SCHEMA
        and model['stage'] == 'native_vcoco_hoi_model' and model['phase'] == 'complete' and model['proof_identity'] == PINS['model_proof.json']
        and host['native_identity'] == PINS['model.json'] and host['images'] == model['images'] and host['public_inputs_identity'] == public.PUBLIC
        and host['endpoint_producer_revision'] == public.REV and host['endpoint_receipt_pins'] == public.PINS
        and host['qualified_model'] == hoi.QUALIFIED and host['runtime_report_identity'] == proof['runtime']['report_identity']
        and proof['image_id'] == hoi.IMAGE and proof['images'] == endpoints['images'] and proof['banks'] == endpoints['banks'],
        'Original complete saved source/runtime/endpoint declarations differ')
    for value in (host, model):
        rt.require(value['source_inputs_runtime_assets_rehashed_after'] is True
            and all(value[k] is False for k in ('reference_metadata_read', 'FIT_performed', 'ownership_verified', 'quality_verified', 'adoption'))
            and type(value['elapsed_seconds']) in (int, float) and 0 < value['elapsed_seconds'] <= hoi.BUDGET
            and not any(k in value for k in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed')), 'Original successful sealed receipts required')
    rt.require(host['outputs_sealed'] is host['owned_cleanup_verified'] is True and model['models_loaded'] == 1
        and type(model['native_forward_calls']) is int and model['native_forward_calls'] == len(model['images']) == 16
        and model['AMP_used'] is model['TF32_used'] is False and model['model']['strict_checkpoint']['keys'] == 1796
        and model['model']['strict_checkpoint']['strict'] is model['model']['strict_checkpoint']['weights_only'] is True
        and model['model']['checkpoint_buffer_schema']['ema_swapped'] is False, 'One original FP32 model/all16 native forwards required')
    for i, (row, image, bank) in enumerate(zip(model['images'], endpoints['images'], endpoints['banks'])):
        n, k, t = (row[a] for a in ('native_detections', 'hand_object_pairs', 'object_target_pairs'))
        rt.require(all(type(v) is int and 0 <= v <= 1_000_000 for v in (n, k, t)) and n <= 1000, 'Native finite counts required')
        d = row['arrays']['native_nms_detections']['shape']
        shapes = dict(query_logits=[1500, 3], query_boxes_cxcywh=[1500, 4], query_tokens=[1500, 256], native_nms_detections=d,
            native_nms_keep=[d[0]], retained_nms_positions=[n], query_ids=[n], class_ids=[n], boxes_original_xyxy=[n, 4], raw_scores=[n],
            decayed_scores=[n], hand_object_pairs=[k, 2], hand_object_logits=[k, 2], object_target_pairs=[t, 2], object_target_logits=[t, 2],
            image_size=[2], original_frame_index=[], original_slot=[], acquired_ordinal=[])
        ints = {'native_nms_keep', 'retained_nms_positions', 'query_ids', 'class_ids', 'hand_object_pairs', 'object_target_pairs',
                'image_size', 'original_frame_index', 'original_slot', 'acquired_ordinal'}
        rt.require(len(d) == 2 and all(type(v) is int and v >= 0 for v in d) and d[1] == 5 and d[0] <= 1000
            and all(type(row[a]) is int for a in ('original_slot', 'acquired_ordinal', 'original_frame_index'))
            and type(row['image_size']) is list and all(type(v) is int and v > 0 for v in row['image_size'])
            and row['image_id'] == image['image_id'] and row['original_slot'] == bank['original_slot'] == i and row['acquired_ordinal'] == i
            and row['original_frame_index'] == 0 and row['image_size'] == bank['image_size'] == [image['height'], image['width']]
            and row['endpoint_bank_identity'] == bank['identity'] and row['source_person_ids'] == bank['person_ids'] and row['owl_patches'] == 3600
            and row['file'] == f'image_{i:06d}.npz' and files[row['file']] == row['identity'] and set(row['arrays']) == set(hoi.FIELDS)
            and all(type(row['arrays'][a]) is dict and set(row['arrays'][a]) == {'shape', 'dtype', 'sha256'}
                and row['arrays'][a]['shape'] == shape and row['arrays'][a]['dtype'] == ('<i8' if a in ints else '<f4')
                and re.fullmatch('[0-9a-f]{64}', str(row['arrays'][a]['sha256'])) for a, shape in shapes.items()), 'Full19-field native banks/original slot identity required')
    rt.require(sum(r['hand_object_pairs'] for r in model['images']) == 165, 'Audited original165 raw pairs required')


def sender_inputs(code, binding, deadline):
    old = ROOT/'jobs'/REV/hoi.ENTRY/'code'; sb = rt.source(ROOT, old, REV, hoi.ENTRY, hoi.HELPERS)
    rt.require(sb['entries'] == 349 and sb['closure_sha256'] == CLOSURE and sum(p.is_file() for p in old.rglob('*')) == 344
        and (old.parent/'source-sha256').read_bytes() == (XZ+'\n').encode()
        and all(binding['helpers'][n] == sb['helpers'][n] for n in hoi.HELPERS), 'Whole original344-file source/unchanged helpers required')
    source_modes(old, SENDER_EXECUTABLES)
    _, _, original = public.export_inputs(code, binding)
    images = public.bank.public_inputs(public.PUBLIC, 16)['images']
    banks = rt.pinned(public.bank.OUTPUT/'native.json', public.PINS['native.json'], 256 << 10)['images']
    endpoints = dict(images=images, banks=banks)
    paths = {n: hoi.OUTPUT/('report.json' if n == 'host.json' else n) for n in NAMES}
    values = {n: rt.pinned(paths[n], p, 1 << 20) for n, p in PINS.items()}
    files = {n: rt.identity(p, MAXIMUM) for n, p in paths.items()}
    validate_metadata(*(values[n] for n in PINS), files, endpoints)
    hoi.validate_native(values['model.json'], values['model_proof.json'], REV, 'model', PINS['model_proof.json'])
    private_directory(hoi.OUTPUT, 0o500, {'report.json', 'model.json', 'model_proof.json', 'overlay.json', 'overlay_proof.json',
        'overlay.cid', 'model.cid', 'FairScale.LICENSE', 'terminaltables.LICENSE', *(f'image_{i:06d}.npz' for i in range(16))})
    rt.require(not (hoi.OUTPUT/'.overlay').exists() and not (hoi.OUTPUT/'.scratch').exists(), 'Original overlays must remain absent')
    for phase in ('overlay', 'model'):
        raw = (hoi.OUTPUT/(phase+'.cid')).read_bytes(); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Saved original CID required')
        rt.require(not public.bank.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+raw.decode().strip()], deadline)
            and not public.bank.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/world-reward-vcoco-hoi-'+phase+'-'+REV[:12]+'$'], deadline), 'Original CID/name must remain absent')
    policy = rt.pinned(code/hoi.original.PROTOCOL, hoi.original.PROTOCOL_PIN, 16 << 10)
    runtime = hoi.original.authenticate_runtime(rt, hoi.runtime, policy, deadline)
    rt.require(runtime == values['model_proof.json']['runtime'] and runtime['image']['Id'] == hoi.IMAGE, 'Actual sender runtime metadata differs')
    for p in paths.values():
        s = p.lstat(); rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Original sealed400 bytes required')
    states = {str(p): snapshot(p) for p in (hoi.OUTPUT, *hoi.OUTPUT.iterdir(), old.parent/'revision', old.parent/'source-sha256')}
    outputs = {p.name: rt.identity(p, MAXIMUM) for p in hoi.OUTPUT.iterdir()}
    return paths, files, dict(source=sb, source_states=public.bank.source_state(old), original_endpoint=original,
        runtime=runtime, states=states, complete_original_output_pins=outputs)


def validate_manifest(value, revision):
    rt.require(type(value) is dict and set(value) == {'schema', 'export_revision', 'original_source_declaration', 'files'}
        and value['schema'] == SCHEMA and value['export_revision'] == revision and re.fullmatch('[0-9a-f]{40}', revision)
        and value['original_source_declaration'] == DECLARATION and type(value['files']) is dict and set(value['files']) == NAMES, 'Exact19-file original manifest required')
    for p in value['files'].values(): public.fixed_pin(p, MAXIMUM)
    rt.require(all(value['files'][n] == p for n, p in PINS.items()) and sum(p['bytes'] for p in value['files'].values()) < MAXIMUM-(1 << 20), 'Bounded original receipt/bank payload required')
    return value


class BoundedWriter:
    def __init__(self, writer, deadline): self.writer, self.deadline, self.size = writer, deadline, 0
    def write(self, raw):
        check(self.deadline); rt.require(type(raw) is bytes and self.size+len(raw) <= MAXIMUM, '32MiB complete USTAR bound')
        self.size += len(raw); return self.writer.write(raw)


def pack(writer, manifest, paths, deadline):
    raw = encode(manifest); rt.require(len(raw) <= MAX_MANIFEST, 'Bounded first manifest')
    with tarfile.open(fileobj=BoundedWriter(writer, deadline), mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        for name in ('manifest.json', *sorted(paths)):
            check(deadline); row = tarfile.TarInfo(name); row.size = len(raw) if name == 'manifest.json' else manifest['files'][name]['bytes']
            row.mode = 0o400; row.uid = row.gid = row.mtime = 0
            if name == 'manifest.json': archive.addfile(row, io.BytesIO(raw))
            else:
                rt.require(rt.identity(paths[name], MAXIMUM) == manifest['files'][name], 'Original source leaf changed')
                with paths[name].open('rb') as stream: archive.addfile(row, stream)


def verify_archive(path, archive_pin, manifest_pin, revision, deadline):
    rt.require(rt.identity(path, MAXIMUM) == public.fixed_pin(archive_pin, MAXIMUM), 'Independent whole archive SHA required')
    public.fixed_pin(manifest_pin, MAX_MANIFEST); table = []; manifest = None
    with path.open('rb') as stream:
        while True:
            check(deadline); header = stream.read(512); rt.require(len(header) == 512, 'Truncated USTAR header')
            if header == b'\0'*512:
                count = 0
                for raw in iter(lambda: stream.read(1 << 20), b''):
                    check(deadline); rt.require(not any(raw), 'Trailing nonzero archive bytes'); count += len(raw)
                rt.require(count >= 512 and count % 512 == 0, 'Complete zero TAR tail'); break
            rt.require(header[257:265] == b'ustar\00000' and not any(header[345:500]) and not header[124] & 128, 'No prefix/GNU/PAX/size alias')
            row = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict'); index = len(table)
            rt.require(row.type == tarfile.REGTYPE and not row.linkname and row.mode == 0o400 and row.uid == row.gid == row.mtime == 0
                and not row.uname and not row.gname and index <= 19 and row.tobuf(format=tarfile.USTAR_FORMAT) == header,
                'Unique canonical bounded regular USTAR members only')
            expected = manifest_pin if index == 0 else manifest['files'].get(row.name)
            rt.require(row.name == ('manifest.json' if index == 0 else sorted(manifest['files'])[index-1])
                and expected is not None and row.size == expected['bytes'], 'First manifest/ordered exact member size')
            offset = stream.tell(); remaining = row.size; digest = hashlib.sha256(); first = bytearray()
            while remaining:
                check(deadline); raw = stream.read(min(1 << 20, remaining)); rt.require(raw, 'Truncated member payload')
                remaining -= len(raw); digest.update(raw)
                if index == 0: first.extend(raw)
            rt.require(dict(bytes=row.size, sha256=digest.hexdigest()) == expected, 'Original member SHA differs')
            if index == 0: manifest = validate_manifest(rt.strict(first), revision)
            raw = stream.read((-row.size) % 512); rt.require(len(raw) == (-row.size) % 512 and not any(raw), 'Exact zero payload padding')
            table.append((row.name, offset, row.size))
    rt.require(len(table) == 20 and {r[0] for r in table} == {'manifest.json', *NAMES}
        and rt.identity(path, MAXIMUM) == archive_pin, 'Complete20-member immutable archive required')
    return manifest, table


def installed(manifest):
    private_directory(DEST, 0o700, {'banks'}); private_directory(DEST/'banks', 0o500, NAMES)
    files = {}
    for n in NAMES:
        p = DEST/'banks'/n; s = p.lstat()
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Root-owned immutable replica leaves required')
        files[n] = rt.identity(p, MAXIMUM)
    rt.require(files == manifest['files'], 'All original19 leaf bytes differ')
    return dict(files=files, states={str(p): snapshot(p) for p in (DEST, DEST/'banks', *(DEST/'banks'/n for n in sorted(NAMES)))})


def install(path, stage, manifest, table, deadline):
    rt.canonical(stage); rt.canonical(DEST); private_directory(DEST.parent, 0o700)
    rt.require(stage.parent == DEST.parent and not DEST.exists() and not DEST.is_symlink(), 'No overwrite or parent bootstrap')
    stage.mkdir(mode=0o700); owner = snapshot(stage); owned = {}
    try:
        (stage/'banks').mkdir(mode=0o700); owned['banks'] = snapshot(stage/'banks')
        with path.open('rb') as stream:
            for name, offset, size in table[1:]:
                check(deadline); rt.require(name in NAMES and name not in owned and size == manifest['files'][name]['bytes'], 'Verified exact member table')
                stream.seek(offset); target = stage/'banks'/name
                with target.open('xb') as output:
                    os.fchmod(output.fileno(), 0o400); owned[name] = snapshot(target); remaining = size
                    while remaining:
                        check(deadline); raw = stream.read(min(1 << 20, remaining)); rt.require(raw, 'Truncated installed payload')
                        output.write(raw); remaining -= len(raw)
                    output.flush(); os.fsync(output.fileno())
                rt.require(rt.identity(target, MAXIMUM) == manifest['files'][name], 'Installed original hash differs')
        rt.require(set(owned) == {'banks', *NAMES} and snapshot(stage)[:3] == owner[:3] and {p.name for p in stage.iterdir()} == {'banks'}
            and {p.name for p in (stage/'banks').iterdir()} == NAMES, 'Exclusive complete staging namespace')
        for n in NAMES:
            p = stage/'banks'/n; s = p.lstat()
            rt.require(snapshot(p)[:2] == owned[n][:2] and s.st_uid == os.getuid() and s.st_nlink == 1
                and stat.S_ISREG(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o400, 'Only originally opened400 staging leaves')
        (stage/'banks').chmod(0o500); public.sync(stage/'banks'); public.sync(stage); check(deadline)
        atomic.rename_noreplace(stage, DEST); public.sync(DEST.parent)
    except BaseException:
        if stage.exists():
            rt.require(snapshot(stage)[:3] == owner[:3] and {p.name for p in stage.iterdir()} <= {'banks'}, 'Foreign staging root never removed')
            folder = stage/'banks'
            if folder.exists():
                rt.require(snapshot(folder)[:2] == owned['banks'][:2] and {p.name for p in folder.iterdir()} == set(owned)-{'banks'}, 'Foreign staging leaves never removed')
                folder.chmod(0o700)
                for n in set(owned)-{'banks'}:
                    p = folder/n; rt.canonical(p); rt.require(snapshot(p)[:2] == owned[n][:2] and p.lstat().st_nlink == 1, 'Only owned partial leaves'); p.unlink()
                folder.rmdir()
            stage.rmdir()
        raise


def export_receipt(raw, expected, revision):
    rt.require(pin(raw) == public.fixed_pin(expected, 256 << 10), 'Independent original export receipt required'); r = rt.strict(raw)
    rt.require(r['schema'] == SCHEMA and r['phase'] == 'export' and r['status'] == 'pass' and r['producer_revision'] == revision
        and r['source_binding']['producer_revision'] == revision and r['original_source_declaration'] == DECLARATION
        and r['files'] == 19 and r['source_inputs_rehashed_after'] is r['outputs_sealed'] is True and public.valid_etag(r['blob_etag'])
        and all(r[k] is False for k in ('models_loaded', 'GPU_used', 'reference_metadata_read', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified', 'adoption'))
        and not any(k in r for k in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed')), 'Complete exclusive export declaration required')
    public.fixed_pin(r['archive_identity'], MAXIMUM); public.fixed_pin(r['manifest_identity'], MAX_MANIFEST); return r


def qualified_inputs(code, revision, receipt_pin):
    old = ROOT/'jobs'/revision/ENTRY/'code'; binding = rt.source(ROOT, old, revision, ENTRY, HELPERS)
    rt.require(all(rt.identity(code/n, 2_000_000, empty=True) == p for n, p in binding['helpers'].items()), 'Unchanged receiver helper source required')
    out = ROOT/f'results/vcoco-hoi-saved-replica-import-{revision}'; private_directory(out, 0o500, {'report.json', 'manifest.json', 'export-receipt.json'})
    receipt_states = {str(p): snapshot(p) for p in (out, *out.iterdir())}
    for p in out.iterdir():
        s = p.lstat(); rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Original400 root-owned technical receipts required')
    report = rt.pinned(out/'report.json', receipt_pin, 256 << 10)
    rt.require(report['schema'] == SCHEMA and report['phase'] == 'import' and report['status'] == 'pass' and report['producer_revision'] == revision
        and report['source_binding'] == binding and report['source_inputs_rehashed_after'] is report['outputs_sealed'] is report['blob_cleanup_verified'] is True
        and report['single_etag_DELETE_202'] is True and report['delete_attempts'] == 1 and report['files'] == 19
        and report['replica_directory'] == str(DEST) and report['budget_seconds'] == BUDGET
        and report['original_source_on_receiver_live_verified'] is False and report['original_runtime_on_receiver_live_verified'] is False
        and not any(k in report for k in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed')), 'Separate actual sealed19-file replica PASS required')
    rt.require(rt.identity(out/'export-receipt.json', 256 << 10) == report['export_receipt_identity'], 'Bounded original receipt before read')
    raw = (out/'export-receipt.json').read_bytes(); export = export_receipt(raw, report['export_receipt_identity'], report['export_revision'])
    manifest = rt.pinned(out/'manifest.json', report['manifest_identity'], MAX_MANIFEST); validate_manifest(manifest, report['export_revision'])
    rt.require(report['archive_identity'] == export['archive_identity'] and report['manifest_identity'] == export['manifest_identity'], 'Original exported artifact pins differ')
    endpoint = pose.completed_replica_inputs(code, *COMPLETION); before = installed(manifest)
    values = {n: rt.pinned(DEST/'banks'/n, p, 1 << 20) for n, p in PINS.items()}
    validate_metadata(*(values[n] for n in PINS), before['files'], endpoint)
    rt.require(installed(manifest) == before and rt.source(ROOT, old, revision, ENTRY, HELPERS) == binding
        and rt.identity(out/'report.json', 256 << 10) == receipt_pin and rt.identity(out/'manifest.json', MAX_MANIFEST) == report['manifest_identity']
        and rt.identity(out/'export-receipt.json', 256 << 10) == report['export_receipt_identity']
        and {str(p): snapshot(p) for p in (out, *out.iterdir())} == receipt_states
        and pose.completed_replica_inputs(code, *COMPLETION) == endpoint, 'Receiver source/complete bytes changed')
    return dict(before, hoi_rows=values['model.json']['images'], original_source_declaration=DECLARATION,
        sender_source_live_verified=False, sender_runtime_live_verified=False, import_identity=receipt_pin, import_source=binding)


def authenticate_receiver(code, replica_revision, receipt_pin):
    """Future blind join: authenticated copied HOI rows + genuine original endpoints."""
    value = qualified_inputs(code, replica_revision, receipt_pin)
    endpoint = pose.completed_replica_inputs(code, *COMPLETION)
    out = ROOT/f'results/vcoco-hoi-saved-replica-import-{replica_revision}'
    paths = {str(DEST/'banks'/n): p for n, p in value['files'].items()}
    report = rt.pinned(out/'report.json', receipt_pin, 256 << 10)
    paths.update({str(out/'report.json'): receipt_pin, str(out/'manifest.json'): report['manifest_identity'],
                  str(out/'export-receipt.json'): report['export_receipt_identity']})
    return dict(value, images=endpoint['images'], banks=endpoint['banks'], files=paths,
        states={**value['states'], **{str(p): snapshot(p) for p in (out, *out.iterdir())}})


def run(args, code, revision):
    started = time.monotonic(); deadline = started+BUDGET
    def expired(*_): raise TimeoutError('Inclusive saved-bank replica deadline')
    before = original = manifest = endpoint = replica_before = None; archive_owner = []; allowed = set()
    rt.require(sys.platform == 'linux' and os.geteuid() == 0, 'Azure root stdlib-only caller')
    out = ROOT/f'results/vcoco-hoi-saved-replica-{args.phase}-{revision}'; rt.canonical(out)
    rt.require(not out.exists() and not out.is_symlink(), 'Fresh technical receipt namespace'); out.mkdir(mode=0o700)
    s = out.lstat(); owner = (s.st_dev, s.st_ino, s.st_uid); archive = out/'archive.tar'
    handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}; signal.alarm(BUDGET)
    report = dict(schema=SCHEMA, phase=args.phase, status='fail', producer_revision=revision, original_source_declaration=DECLARATION,
        files=19, budget_seconds=BUDGET, models_loaded=False, GPU_used=False, reference_metadata_read=False, RGB_NPZ_decoded=False,
        quality_verified=False, ownership_verified=False, adoption=False, source_inputs_rehashed_after=False, blob_cleanup_verified=False,
        original_source_on_receiver_live_verified=False, original_runtime_on_receiver_live_verified=False, delete_attempts=0, single_etag_DELETE_202=False)
    try:
        before = source(code, revision); transport.verify_azure_peer(args.phase); check(deadline)
        export_revision = revision if args.phase == 'export' else args.export_revision
        url = 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+export_revision+'.tar'
        blob = transport.Blob(url, export_revision, managed_identity=True)
        if args.phase == 'export':
            paths, files, original = sender_inputs(code, before['binding'], deadline)
            manifest = validate_manifest(dict(schema=SCHEMA, export_revision=revision, original_source_declaration=DECLARATION, files=files), revision)
            writer = transport.BlockWriter(blob); pack(writer, manifest, paths, deadline); archive_pin = writer.finish(); public.fixed_pin(archive_pin, MAXIMUM); check(deadline)
            with blob.request('HEAD') as response:
                etag = response.headers.get('ETag', ''); rt.require(response.status == 200 and response.headers.get('Content-Length') == str(archive_pin['bytes']) and public.valid_etag(etag), 'Exclusive committed archive HEAD required')
            report.update(archive_identity=archive_pin, manifest_identity=pin(encode(manifest)), blob_etag=etag)
        else:
            endpoint = pose.completed_replica_inputs(code, *COMPLETION)
            raw = base64.b64decode(args.export_receipt_base64, validate=True)
            rt.require(base64.b64encode(raw).decode() == args.export_receipt_base64, 'Canonical bounded receipt encoding')
            export = export_receipt(raw, args.export_receipt_pin, export_revision)
            rt.require(export['archive_identity'] == args.archive_pin and export['manifest_identity'] == args.manifest_pin, 'Independent archive/manifest pins required')
            rt.canonical(DEST); private_directory(DEST.parent, 0o700); rt.require(not DEST.exists() and not DEST.is_symlink(), 'Never overwrite installed banks')
            rt.write(out/'export-receipt.json', raw); allowed.add('export-receipt.json')
            transport.download(blob, public.OwnedDownload(archive, archive_owner), args.archive_pin); archive.chmod(0o400); check(deadline)
            manifest, table = verify_archive(archive, args.archive_pin, args.manifest_pin, export_revision, deadline)
            rt.write(out/'manifest.json', encode(manifest)); allowed.add('manifest.json')
            values = {}
            with archive.open('rb') as stream:
                for n, offset, size in table[1:]:
                    if n in PINS: stream.seek(offset); values[n] = rt.strict(stream.read(size))
            validate_metadata(*(values[n] for n in PINS), manifest['files'], endpoint)
            install(archive, DEST.parent/(DEST.name+'.stage-'+revision), manifest, table, deadline)
            rt.require(rt.identity(archive, MAXIMUM) == args.archive_pin, 'Archive changed during installation')
            replica_before = installed(manifest)
            report.update(export_revision=export_revision, archive_identity=args.archive_pin, manifest_identity=args.manifest_pin,
                export_receipt_identity=args.export_receipt_pin, blob_etag=export['blob_etag'], replica_directory=str(DEST))
        check(deadline); report.update(source_binding=before['binding'], status='pass')
    except BaseException as exc: report['error_type'] = error(exc)
    finally:
        try:
            rt.require(before is not None and source(code, revision) == before, 'Full current source changed or preflight incomplete')
            if args.phase == 'export': rt.require(original is not None and sender_inputs(code, before['binding'], deadline)[1:] == (manifest['files'], original), 'Original sender source/runtime/bytes changed')
            else: rt.require(replica_before is not None and installed(manifest) == replica_before and pose.completed_replica_inputs(code, *COMPLETION) == endpoint, 'Installed banks/completion changed')
            report['source_inputs_rehashed_after'] = True
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc))
        if archive.exists():
            try:
                s = archive.lstat()
                rt.require(archive_owner and (s.st_dev, s.st_ino, s.st_uid) == archive_owner[0], 'Only originally opened owned archive cleanup')
                rt.identity(archive, MAXIMUM, readonly=False, empty=True); archive.unlink()
            except BaseException as exc: report.update(status='fail', cleanup_error_type=error(exc)); allowed.add('archive.tar')
        def finish():
            check(deadline); rt.require(installed(manifest) == replica_before and source(code, revision) == before, 'All sealed installed bytes/source beforeDELETE')
            try:
                report['delete_attempts'] += 1
                with blob.request('DELETE', headers={'If-Match': report['blob_etag']}) as response: rt.require(response.status == 202, 'Exact original ETag DELETE202 required')
                report['single_etag_DELETE_202'] = True
            finally:
                rt.require(installed(manifest) == replica_before and source(code, revision) == before
                    and pose.completed_replica_inputs(code, *COMPLETION) == endpoint, 'All receiver inputs/source afterDELETE'); check(deadline)
        try:
            publication.publish(out, report, deadline, started, owner, allowed, finish if args.phase == 'import' else None,
                encode=encode, identity=rt.identity, snapshot=snapshot, require=rt.require, check=check, sync=public.sync, error=error,
                report_maximum=256 << 10)
        finally:
            signal.alarm(0)
            for s, h in handlers.items(): signal.signal(s, h)
    return report


def arguments(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--phase', choices=('export', 'import'), required=True)
    parser.add_argument('--export-revision'); parser.add_argument('--export-receipt-base64')
    for n in ('archive', 'manifest', 'export-receipt'): parser.add_argument('--'+n+'-bytes', type=int); parser.add_argument('--'+n+'-sha256')
    a = parser.parse_args(argv)
    if a.phase == 'export': rt.require(all(v is None for k, v in vars(a).items() if k != 'phase'), 'Export has no input overrides')
    else:
        rt.require(type(a.export_revision) is str and re.fullmatch('[0-9a-f]{40}', a.export_revision)
            and type(a.export_receipt_base64) is str and 0 < len(a.export_receipt_base64) <= 384 << 10, 'Bounded original export control required')
        for n, maximum in (('archive', MAXIMUM), ('manifest', MAX_MANIFEST), ('export_receipt', 256 << 10)):
            setattr(a, n+'_pin', public.fixed_pin(dict(bytes=getattr(a, n+'_bytes'), sha256=getattr(a, n+'_sha256')), maximum))
    return a


if __name__ == '__main__':
    args = arguments(sys.argv[1:]); report = run(args, Path(os.environ.get('WR_CODE', '/invalid')), os.environ.get('WR_CODE_REVISION', ''))
    print(encode({k: report[k] for k in ('phase', 'status', 'files')}).decode(), end=''); raise SystemExit(0 if report['status'] == 'pass' else 1)

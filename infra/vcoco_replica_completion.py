"""Complete cleanup of a verified installed replica; never import or decode it."""
import base64
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_observation_replica as original
import sealed_callback_publication as publication

rt, ROOT = original.rt, original.ROOT
ENTRY = 'run_vcoco_replica_completion'
OUTPUT = ROOT/'results/vcoco-replica-completion-v1'
SCHEMA = 'world_reward.vcoco_replica_completion.v1'
BUDGET = 180
IMPORT = '39774007d9125caf955cba49bf93702dc323777b'
EXPORT = '7c11662f98cf6f3db28909dce147473f1ab3bd0c'
BOOTSTRAP = 'f90b5b1afdf1557dabd375ebedd996413acddeff'
DIAGNOSTIC = 'configs/vcoco_replica_import_v2_diagnostic.json'
DIAGNOSTIC_PIN = dict(bytes=3473, sha256='3110dc383ea38bfdf7d141cd5e69374fb4d9fb7052bf4eeac8b311f3f17b5358')
FAILED_PIN = dict(bytes=5781, sha256='19441e4e8569ae766ece49547fb6137a691fe51e6265e6427de44a9d0de3f7a0')
MANIFEST_PIN = dict(bytes=5166, sha256='7c0786e64de612080c01cf4699a63fafaf3b70f9ac6a9b70fc77e4d8d9b40174')
EXPORT_PIN = dict(bytes=5638, sha256='cc7874d9c9fd21a6137f48066d7edfe8290af962eed5f8db4e9faff6a35591e3')
OLD_FAILED_PIN = dict(bytes=4969, sha256='877d595c1e91ec937270553595c40e6d8cb782269dd5f62b2eeddbf9de775ee1')
BOOTSTRAP_PIN = dict(bytes=10156, sha256='2390636311f1ff4eaa619de6027362d0f301eabc1ffa427cc0716e7067a605d9')
ARCHIVE_PIN = dict(bytes=20756480, sha256='52fec428ad132b85454e5e47a190db700d72fcc4cdad915805af37ee300207ab')
PUBLISHER_PIN = dict(bytes=5861, sha256='9fd62223432ac591b22bab23f916dcd47d9c67264ebfa8c5782a922b0f0e23ae')
SOURCES = {
    IMPORT: (original.ENTRY, 323, 328, '30076cc99801e1ef70feac7a7104b9e8ae4960c7a5c11d76e484f2c98ab9fa08',
             '040a11d39ecb3de5f30f16dce7a5df6fc829dbf8f862f19be5094eb762eefc56'),
    EXPORT: (original.ENTRY, 322, 327, 'd7bfa3c09b7aa69e147b19b3a9e3a9e1e81103c165d5e052ea451030f29536e1',
             'd5dd6eec412311579a51a3b550d745d19afea8a02a843d7f6feedf5a2ef76a5e'),
    BOOTSTRAP: ('run_vcoco_replica_parent_bootstrap', 325, 330,
                'dcaa57ffabef3634f6b2adb743dae1adf4f3fa9ab64d4461e169c7b265c79015',
                'efcabf91c1a05f996b0b91b44fa304052906cc18de42792b72e0857c2a1796e9')}
HELPERS = tuple(dict.fromkeys(('infra/vcoco_replica_completion.py', 'infra/run_vcoco_replica_completion.sh',
    'infra/sealed_callback_publication.py', DIAGNOSTIC, *original.HELPERS)))


def check(deadline):
    rt.require(time.monotonic() < deadline, 'Inclusive completion deadline')


def directory(path, mode, names):
    rt.canonical(path); s = path.lstat()
    rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == mode and s.st_uid == s.st_gid == 0
               and {p.name for p in path.iterdir()} == set(names), 'Exact private immutable namespace required')


def saved_source(revision):
    entry, count, entries, closure, xz = SOURCES[revision]
    code = ROOT/'jobs'/revision/entry/'code'
    helpers = original.HELPERS if revision != BOOTSTRAP else tuple(dict.fromkeys((
        'infra/vcoco_replica_parent_bootstrap.py', 'infra/run_vcoco_replica_parent_bootstrap.sh',
        'configs/vcoco_replica_parent_diagnostic.json', *original.HELPERS)))
    value = rt.source(ROOT, code, revision, entry, helpers)
    rt.require(value['entries'] == entries and value['closure_sha256'] == closure
        and sum(p.is_file() for p in code.rglob('*')) == count
        and (code.parent/'source-sha256').read_bytes() == (xz+'\n').encode(), 'Frozen whole original Git source differs')
    for p in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        s = p.lstat(); executable = p == code/'infra/run_rgb_endpoint_bank.sh'
        expected = 0o555 if p.is_dir() or executable else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == expected,
                   'Original Git modes and dispatch owner required')
    return dict(binding=value, states=original.bank.source_state(code),
                markers={n: original.snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def source(code, revision):
    value = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(Path(__file__).resolve() == code/HELPERS[0]
        and Path(original.__file__).resolve() == code/'infra/vcoco_observation_replica.py'
        and Path(publication.__file__).resolve() == code/'infra/sealed_callback_publication.py'
        and value['helpers']['infra/sealed_callback_publication.py'] == PUBLISHER_PIN,
        'Actual unchanged helper origins required')
    for p in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        s = p.lstat(); expected = 0o555 if p.is_dir() or p == code/'infra/run_rgb_endpoint_bank.sh' else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == expected,
                   'Root-owned exact readonly current source modes required')
    return dict(binding=value, states=original.bank.source_state(code),
                markers={n: original.snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def saved_banks(public, native, host, proof, files):
    """Validate copied receipt ABI without invoking sender-path validators."""
    rt.require(host['schema'] == 'world_reward.vcoco_pilot_endpoint_bank.v1'
        and host['stage'] == 'vcoco_pilot_endpoint_bank_host' and host['status'] == 'pass'
        and host['producer_revision'] == original.REV and host['source_binding'] == proof['source']
        and host['public_inputs_identity'] == original.PUBLIC and host['native_report_identity'] == original.PINS['native.json']
        and host['native_exit_status'] == 0 and host['acquired_images'] == 16 and host['native_images'] == native['images']
        and host['owned_cleanup_verified'] is host['outputs_sealed'] is host['source_inputs_runtime_assets_rehashed_after'] is True
        and all(host[k] is False for k in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read',
                                         'FIT_performed', 'ownership_verified', 'quality_verified', 'adoption')),
        'Saved original label-blind host required')
    rt.require(proof['source']['producer_revision'] == original.REV and proof['source']['entries'] == 321
        and proof['source']['closure_sha256'] == original.CLOSURE and proof['inputs_identity'] == original.PUBLIC
        and proof['images'] == 16 and proof['image_id'] == original.bank.IMAGE
        and proof['native_files'] == {n: proof['source']['helpers'][n] for n in original.bank.NATIVE_FILES}
        and native['runtime_identity'] == proof['owl_runtime'] == host['original_qualification']['owl']['native_runtime'],
        'Saved source/runtime declarations required, not live sender-source verification')
    rt.require(native['schema'] == host['schema'] and native['stage'] == 'native_vcoco_pilot_endpoint_banks'
        and native['status'] == 'pass' and native['phase'] == 'complete' and native['producer_revision'] == original.REV
        and native['proof_identity'] == original.PINS['proof.json'] and native['image_id'] == original.bank.IMAGE
        and native['model_loads'] == 2 and native['all_patches_retained'] is native['source_inputs_runtime_assets_rehashed_after'] is True
        and native['person_query'] == 'person.' and (native['confidence'], native['text_threshold'], native['nms_iou']) == (.3, .25, .7)
        and all(native[k] is False for k in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read',
            'challenge_inputs_used', 'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption'))
        and all(native[k] == 0 for k in ('dwpose_calls', 'sam_calls', 'hoi_calls', 'tracking_calls'))
        and all(native[k] == 16 for k in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'))
        and len(native['images']) == 16, 'Complete original900/3600 receipt required')
    for i, (image, row) in enumerate(zip(public['images'], native['images'])):
        rt.require(original.bank.rgb_inputs.original_slot(image, 16) == i and row['image_id'] == image['image_id']
            and row['bank_index'] == row['original_slot'] == row['acquired_ordinal'] == i and row['original_frame_index'] == 0
            and row['image_size'] == [image['height'], image['width']] and row['input_file'] == image['file']
            and row['input_identity'] == {k: image[k] for k in ('bytes', 'sha256')}
            and row['file'] == f'image_{i:06d}.npz' and row['identity'] == files['banks/'+row['file']]
            and row['owl_patches'] == 3600 and row['person_native_queries'] == 900
            and type(row['person_retained_rows']) is int and type(row['person_postprocessor_rows']) is int
            and 0 <= row['person_retained_rows'] <= row['person_postprocessor_rows'] <= 900
            and row['person_ids'] == [f'image:{image["image_id"]}/person/retained:{j:06d}' for j in range(row['person_retained_rows'])]
            and set(row['arrays']) == original.bank.BANK_KEYS, 'Original16 slot/grid/person identities required')
        shapes = {'person_model_pred_boxes': [1, 900, 4], 'person_model_logits': [1, 900, 256],
            'owl_patch_ids': [3600], 'owl_boxes_padded_normalized_cxcywh': [3600, 4],
            'owl_objectness_logits': [3600], 'owl_boxes_original_xyxy': [3600, 4], 'image_size': [2], 'original_frame_index': []}
        rt.require(all(row['arrays'][n]['shape'] == shape for n, shape in shapes.items())
            and all(set(v) == {'shape', 'dtype', 'sha256'} and type(v['shape']) is list
                and all(type(x) is int and x >= 0 for x in v['shape']) and type(v['dtype']) is str
                and re.fullmatch('[0-9a-f]{64}', str(v['sha256'])) for v in row['arrays'].values()),
            'All17 stored native array metadata required')


def qualified_inputs(code):
    """Read/hash all36 original bytes and receipts; no RGB/NPZ or model decode.

    Returns the actual FAILED import source separately, never relabeled PASS.
    Frozen source closure authority is the independently Git-bound prior audit.
    """
    sources = {v: saved_source(v) for v in SOURCES}
    rt.require(all(rt.identity(code/n, 2_000_000, empty=True) == p
        for n, p in sources[IMPORT]['binding']['helpers'].items()), 'Every original replica helper must remain unchanged')
    diag = rt.pinned(code/DIAGNOSTIC, DIAGNOSTIC_PIN, 16 << 10)
    rt.require(diag['status'] == 'pass' and diag['stage'] == 'complete' and diag['producer_revision'] == IMPORT
        and diag['receipt_identity'] == FAILED_PIN and diag['saved_status'] == 'fail'
        and diag['saved_flags']['publication_failed'] is True and diag['saved_flags']['blob_cleanup_verified'] is False
        and diag['source_and_receipts_prepost'] is diag['replica_prepost_verified'] is diag['replica_complete_original_hash_modes'] is True
        and diag['replica_files'] == 36 and diag['replica_original_bytes'] == 20712157
        and diag['destination']['inode'] == [2065, 265424] and diag['parent']['inode'] == [2065, 265398],
        'Saved independent installed-replica diagnosis required')
    folders = {IMPORT: ROOT/f'results/vcoco-observation-replica-import-{IMPORT}',
        EXPORT: ROOT/f'results/vcoco-observation-replica-import-{EXPORT}',
        BOOTSTRAP: ROOT/'results/vcoco-replica-parent-bootstrap-v1'}
    pins = {IMPORT: {'report.json': FAILED_PIN, 'manifest.json': MANIFEST_PIN, 'export-receipt.json': EXPORT_PIN},
        EXPORT: {'report.json': OLD_FAILED_PIN, 'manifest.json': MANIFEST_PIN, 'export-receipt.json': EXPORT_PIN},
        BOOTSTRAP: {'report.json': BOOTSTRAP_PIN}}
    values, files, states = {}, {}, {}
    for revision, folder in folders.items():
        directory(folder, 0o500, pins[revision]); values[revision] = {}
        for name, pin in pins[revision].items():
            path = folder/name; values[revision][name] = rt.pinned(path, pin, 256 << 10)
            s = path.lstat(); rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Original receipt owner/mode differs')
            files[str(path)] = pin
        states.update({str(p): original.snapshot(p) for p in (folder, *folder.iterdir())})
    failed, old, boot = (values[v]['report.json'] for v in (IMPORT, EXPORT, BOOTSTRAP))
    for revision, report in ((IMPORT, failed), (EXPORT, old)):
        rt.require(report['schema'] == original.SCHEMA and report['phase'] == 'import' and report['status'] == 'fail'
            and report['producer_revision'] == revision and report['source_binding'] == sources[revision]['binding']
            and report['source_stat_identity'] == sources[revision]['states']
            and report['source_inputs_rehashed_after'] is report['outputs_sealed'] is report['archive_removed'] is True
            and report['blob_cleanup_verified'] is False and report['files'] == 36
            and all(report[k] is False for k in ('models_loaded', 'GPU_used', 'reference_metadata_read', 'ownership_verified', 'quality_verified', 'adoption')),
            'Preserve complete original import FAIL, never reinterpret its status')
    rt.require(failed['publication_failed'] is True and failed['elapsed_seconds'] == .9633648640010506
        and not {'error_type', 'post_error_type', 'cleanup_error_type'} & set(failed)
        and old['error_type'] == 'ValueError' and old['elapsed_seconds'] == .5756366139976308,
        'Frozen failure boundary differs')
    rt.require(boot['status'] == 'pass' and boot['producer_revision'] == BOOTSTRAP
        and boot['source_binding'] == sources[BOOTSTRAP]['binding'] and boot['original_source'] == sources[EXPORT]['binding']
        and boot['source_inputs_rehashed_after'] is boot['outputs_sealed'] is True
        and boot['parent']['created'] is True and boot['parent']['state'][:2] == [2065, 265398], 'Frozen parent bootstrap required')
    manifest = original.validate_manifest(values[IMPORT]['manifest.json'], EXPORT)
    rt.require(manifest == values[EXPORT]['manifest.json'], 'Identical original manifest required')
    raw = (folders[IMPORT]/'export-receipt.json').read_bytes()
    _, export = original.receipt_from_base64(base64.b64encode(raw).decode(), EXPORT_PIN, EXPORT)
    rt.require(export == values[EXPORT]['export-receipt.json'] and export['archive_identity'] == ARCHIVE_PIN
        and export['manifest_identity'] == MANIFEST_PIN and failed['archive_identity'] == ARCHIVE_PIN
        and failed['manifest_identity'] == MANIFEST_PIN and failed['export_receipt_identity'] == EXPORT_PIN
        and failed['blob_etag'] == export['blob_etag'] and failed['replica_directory'] == str(original.DEST)
        and failed['original_source_on_receiver_live_verified'] is False and failed['original_audit_declaration'] == original.AUDIT,
        'Original exclusive export and installed-byte lineage differs')
    replica = original.replica_identity(manifest)
    directory(original.DEST.parent, 0o700, {original.DEST.name})
    rt.require(original.snapshot(original.DEST.parent)[:2] == (2065, 265398)
        and original.snapshot(original.DEST)[:2] == (2065, 265424), 'Independently observed original installed inodes required')
    for revision, folder in folders.items():
        if revision == BOOTSTRAP: continue
        stage = original.DEST.parent/(original.DEST.name+'.stage-'+revision)
        rt.canonical(stage); rt.require(not stage.exists() and not stage.is_symlink()
            and not (folder/'archive.tar').exists(), 'Original staging/archive must remain absent')
    files.update({str(original.DEST/n): p for n, p in replica.items()})
    states.update({str(p): original.snapshot(p) for p in (original.DEST.parent, original.DEST, *sorted(original.DEST.rglob('*')))})
    public = original.bank.rgb_inputs.read_inputs(original.DEST/'inputs', original.PUBLIC, 16,
        identity=rt.identity, pinned=rt.pinned, maximum_slots=16)
    native = rt.pinned(original.DEST/'banks/native.json', original.PINS['native.json'], 256 << 10)
    host = rt.pinned(original.DEST/'banks/host.json', original.PINS['host.json'], 256 << 10)
    proof = rt.pinned(original.DEST/'banks/proof.json', original.PINS['proof.json'], 256 << 10)
    original.bank.configuration(code, dict(helpers={n: rt.identity(code/n, 2_000_000, empty=True) for n in original.HELPERS}))
    saved_banks(public, native, host, proof, replica)
    return dict(original_replica_source=sources[IMPORT]['binding'], original_sources=sources,
        files=files, states=states, manifest=manifest, images=public['images'], banks=native['images'],
        replica_snapshot=original.replica_state(), import_failed_pin=FAILED_PIN, old_failed_pin=OLD_FAILED_PIN,
        bootstrap_pin=BOOTSTRAP_PIN, diagnostic_saved_declaration=DIAGNOSTIC_PIN,
        archive_pin=ARCHIVE_PIN, manifest_pin=MANIFEST_PIN, export_pin=EXPORT_PIN, export_revision=EXPORT,
        blob_etag=export['blob_etag'], original_bank_source_on_receiver_live_verified=False)


def authenticate_completion(code, completion_revision, completion_pin):
    """Future consumer: authenticate NEW completion PASS, keeping both old FAILs.

    Does not issue Blob requests or reuse the original import PASS-only gate.
    The consumer still authenticates its own caller/source/runtime separately.
    """
    rt.require(re.fullmatch('[0-9a-f]{40}', completion_revision), 'Exact completion revision required')
    old = ROOT/'jobs'/completion_revision/ENTRY/'code'
    binding = rt.source(ROOT, old, completion_revision, ENTRY, HELPERS)
    rt.require(all(rt.identity(code/n, 2_000_000, empty=True) == p for n, p in binding['helpers'].items()),
               'Consumer must reuse unchanged completion/helper source')
    receipt = rt.pinned(OUTPUT/'report.json', completion_pin, 512 << 10)
    rt.require(receipt['schema'] == SCHEMA and receipt['stage'] == 'installed_replica_completion' and receipt['status'] == 'pass'
        and receipt['producer_revision'] == completion_revision and receipt['source_binding'] == binding
        and receipt['source_inputs_rehashed_after'] is receipt['outputs_sealed'] is receipt['blob_cleanup_verified'] is True
        and receipt['single_etag_DELETE_202'] is True and receipt['delete_attempts'] == 1 and receipt['files'] == 36
        and 'publication_failed' not in receipt and type(receipt['elapsed_seconds']) in (int, float)
        and 0 < receipt['elapsed_seconds'] <= BUDGET
        and all(receipt[k] is False for k in ('RGB_NPZ_decoded', 'model_execution', 'GPU_used', 'reference_metadata_read',
                                            'import_replayed', 'quality_verified', 'ownership_verified', 'adoption')),
        'Actual separate sealed completion PASS required')
    directory(OUTPUT, 0o500, {'report.json'})
    s = (OUTPUT/'report.json').lstat()
    rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Actual root-owned completion receipt required')
    proof = qualified_inputs(code)
    rt.require(receipt['input_proof'] == rt.strict(original.encode(proof)), 'Complete original installed-byte proof changed')
    rt.require(rt.source(ROOT, old, completion_revision, ENTRY, HELPERS) == binding
        and qualified_inputs(code) == proof and rt.identity(OUTPUT/'report.json', 512 << 10) == completion_pin,
        'Completion source/receipt/installed inputs changed during consumer authentication')
    return dict(proof, completion_identity=completion_pin, completion_revision=completion_revision,
                completion_source=binding, completion_original_source_states=original.bank.source_state(old))


def run(code, revision):
    started = time.monotonic(); deadline = started+BUDGET
    before = inputs = None
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
               'Exact root VM01 completion required')
    rt.canonical(OUTPUT); rt.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Fresh completion result namespace')
    OUTPUT.mkdir(mode=0o700); s = OUTPUT.lstat(); owner = (s.st_dev, s.st_ino, s.st_uid)
    report = dict(schema=SCHEMA, stage='installed_replica_completion', status='fail', producer_revision=revision,
        files=36, budget_seconds=BUDGET, delete_attempts=0, single_etag_DELETE_202=False,
        source_inputs_rehashed_after=False, outputs_sealed=False, blob_cleanup_verified=False,
        RGB_NPZ_decoded=False, model_execution=False, GPU_used=False, reference_metadata_read=False,
        import_replayed=False, quality_verified=False, ownership_verified=False, adoption=False)
    def expired(*_): raise TimeoutError('Inclusive completion deadline')
    handlers = {v: signal.signal(v, expired) for v in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(BUDGET)
    try:
        before = source(code, revision); inputs = qualified_inputs(code); check(deadline)
        report.update(source_binding=before['binding'], source_stat_identity=before['states'], input_proof=inputs, status='pass')
    except BaseException as exc: report['error_type'] = original.bank.error(exc)
    finally:
        try:
            rt.require(before is not None and inputs is not None and source(code, revision) == before
                and qualified_inputs(code) == inputs, 'Complete source and installed inputs changed')
            report['source_inputs_rehashed_after'] = True
        except BaseException as exc: report.update(status='fail', post_error_type=original.bank.error(exc))
        def finish():
            check(deadline)
            rt.require(source(code, revision) == before and qualified_inputs(code) == inputs, 'All gates before the single DELETE')
            url = 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+EXPORT+'.tar'
            try:
                blob = original.transport.Blob(url, EXPORT, managed_identity=True)
                report['delete_attempts'] += 1
                with blob.request('DELETE', headers={'If-Match': inputs['blob_etag']}) as response:
                    rt.require(response.status == 202, 'Original exclusive ETag cleanup did not return202')
                report['single_etag_DELETE_202'] = True
            finally:
                rt.require(source(code, revision) == before and qualified_inputs(code) == inputs, 'Source/installed data changed after DELETE attempt')
                check(deadline)
        try:
            publication.publish(OUTPUT, report, deadline, started, owner, set(), finish,
                encode=original.encode, identity=rt.identity, snapshot=original.snapshot,
                require=rt.require, check=check, sync=original.sync, error=original.bank.error,
                report_maximum=512 << 10)
        finally:
            signal.alarm(0)
            for v, handler in handlers.items(): signal.signal(v, handler)
    return report


if __name__ == '__main__':
    rt.require(not sys.argv[1:] and os.environ.get('WR_ROOT') == str(ROOT), 'Fixed completion environment, no input overrides')
    value = run(Path(os.environ.get('WR_CODE', '/invalid')), os.environ.get('WR_CODE_REVISION', ''))
    print(original.encode({k: value[k] for k in ('stage', 'status', 'files', 'blob_cleanup_verified')}).decode(), end='')
    raise SystemExit(0 if value['status'] == 'pass' else 1)

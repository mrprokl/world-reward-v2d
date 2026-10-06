"""Tiny authored receipt/lifecycle controls; no Azure, media or Blob traffic."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import vcoco_replica_completion as c


def saved_fixture():
    image_id = 'a'*32
    public = dict(schema='world_reward.rgb_proposal_inputs.v1', images=[])
    rows, files = [], {}
    for i in range(16):
        image = dict(image_id=f'{i+1:032x}', file=f'image_{i:06d}.jpg', bytes=12,
                     sha256='1'*64, width=12, height=8)
        public['images'].append(image)
        pin = dict(bytes=16, sha256='2'*64)
        arrays = {n: dict(shape=[], dtype='<f4', sha256='3'*64) for n in c.original.bank.BANK_KEYS}
        for n, shape in {'person_model_pred_boxes': [1, 900, 4], 'person_model_logits': [1, 900, 256],
            'owl_patch_ids': [3600], 'owl_boxes_padded_normalized_cxcywh': [3600, 4],
            'owl_objectness_logits': [3600], 'owl_boxes_original_xyxy': [3600, 4],
            'image_size': [2], 'original_frame_index': []}.items(): arrays[n]['shape'] = shape
        rows.append(dict(image_id=image['image_id'], bank_index=i, original_slot=i, acquired_ordinal=i,
            original_frame_index=0, image_size=[8, 12], input_file=image['file'],
            input_identity={k: image[k] for k in ('bytes', 'sha256')}, file=f'image_{i:06d}.npz', identity=pin,
            owl_patches=3600, person_native_queries=900, person_retained_rows=1, person_postprocessor_rows=2,
            person_ids=[f'image:{image["image_id"]}/person/retained:000000'], arrays=arrays))
        files['banks/'+rows[-1]['file']] = pin
    source = dict(producer_revision=c.original.REV, entries=321, closure_sha256=c.original.CLOSURE,
        helpers={n: dict(bytes=7, sha256='4'*64) for n in c.original.bank.NATIVE_FILES})
    runtime = dict(saved='native')
    proof = dict(source=source, inputs_identity=c.original.PUBLIC, images=16, image_id=c.original.bank.IMAGE,
                 native_files=source['helpers'], owl_runtime=runtime)
    native = dict(schema='world_reward.vcoco_pilot_endpoint_bank.v1', stage='native_vcoco_pilot_endpoint_banks',
        status='pass', phase='complete', producer_revision=c.original.REV, proof_identity=c.original.PINS['proof.json'],
        image_id=c.original.bank.IMAGE, model_loads=2, all_patches_retained=True,
        source_inputs_runtime_assets_rehashed_after=True, person_query='person.', confidence=.3,
        text_threshold=.25, nms_iou=.7, images=rows, runtime_identity=runtime)
    for key in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read', 'challenge_inputs_used',
                'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption'): native[key] = False
    for key in ('dwpose_calls', 'sam_calls', 'hoi_calls', 'tracking_calls'): native[key] = 0
    for key in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'): native[key] = 16
    host = dict(schema=native['schema'], stage='vcoco_pilot_endpoint_bank_host', status='pass',
        producer_revision=c.original.REV, source_binding=source, public_inputs_identity=c.original.PUBLIC,
        native_report_identity=c.original.PINS['native.json'], native_exit_status=0, acquired_images=16,
        native_images=rows, owned_cleanup_verified=True, outputs_sealed=True,
        source_inputs_runtime_assets_rehashed_after=True, original_qualification=dict(owl=dict(native_runtime=runtime)))
    for key in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read', 'FIT_performed',
                'ownership_verified', 'quality_verified', 'adoption'): host[key] = False
    return public, native, host, proof, files


def test_saved_bank_real17field_abi_not_sender_path_validator(monkeypatch):
    args = saved_fixture(); before = deepcopy(args)
    monkeypatch.setattr(c.original.bank, 'validate_report', lambda *_: pytest.fail('Sender-path validator must not run'))
    c.saved_banks(*args)
    assert args == before


@pytest.mark.parametrize('fault', ['person_ids', 'original_slot', 'ordinal', 'frame', 'grid', 'raw900', 'patch3600',
    'missing_array', 'shape_bool', 'sha', 'filepin', 'native_status', 'network_oracle', 'calls', 'runtime', 'source'])
def test_saved_bank_scope_and_abi_rejects_selfconsistent_bad_transport(fault):
    public, native, host, proof, files = saved_fixture()
    row = native['images'][0]
    if fault == 'person_ids': row['person_ids'] = ['guessed-owner']
    elif fault == 'original_slot': row['original_slot'] = 1
    elif fault == 'ordinal': row['acquired_ordinal'] = 1
    elif fault == 'frame': row['original_frame_index'] = 1
    elif fault == 'grid': row['image_size'] = [12, 8]
    elif fault == 'raw900': row['arrays']['person_model_logits']['shape'] = [1, 899, 256]
    elif fault == 'patch3600': row['owl_patches'] = 3599
    elif fault == 'missing_array': del row['arrays']['person_raw_boxes']
    elif fault == 'shape_bool': row['arrays']['person_raw_boxes']['shape'] = [True]
    elif fault == 'sha': row['arrays']['person_raw_boxes']['sha256'] = 'unknown'
    elif fault == 'filepin': row['identity'] = dict(bytes=17, sha256='2'*64)
    elif fault == 'native_status': native['status'] = 'fail'
    elif fault == 'network_oracle': native['ground_truth_used'] = True
    elif fault == 'calls': native['person_forward_calls'] = 15
    elif fault == 'runtime': native['runtime_identity'] = {'different': True}
    elif fault == 'source': proof['source']['entries'] = 320
    with pytest.raises(ValueError): c.saved_banks(public, native, host, proof, files)


def test_qualified_inputs_genuine_failed_receipt_binding_and_bootstrap_helper_abi(monkeypatch, tmp_path):
    """Real primitive schemas with mocked I/O only; no semantic role/reference data."""
    code = tmp_path/'code'; code.mkdir()
    public, native, host, endpoint_proof, bankfiles = saved_fixture()
    dest = tmp_path/'data'/'replica'; dest.parent.mkdir(); dest.mkdir()
    monkeypatch.setattr(c.original, 'DEST', dest)
    sources = {v: dict(binding=dict(producer_revision=v, entries=c.SOURCES[v][2],
        closure_sha256=c.SOURCES[v][3], helpers={}), states='source-'+v) for v in c.SOURCES}
    monkeypatch.setattr(c, 'saved_source', lambda v: deepcopy(sources[v]))
    manifests = dict(schema=c.original.SCHEMA, files=bankfiles, export_revision=c.EXPORT)
    export = dict(archive_identity=c.ARCHIVE_PIN, manifest_identity=c.MANIFEST_PIN, blob_etag='"0xABC"')
    flags = dict(source_inputs_rehashed_after=True, outputs_sealed=True, archive_removed=True,
                 blob_cleanup_verified=False, files=36, models_loaded=False, GPU_used=False,
                 reference_metadata_read=False, ownership_verified=False, quality_verified=False, adoption=False)
    failed = dict(flags, schema=c.original.SCHEMA, phase='import', status='fail', producer_revision=c.IMPORT,
        source_binding=sources[c.IMPORT]['binding'], source_stat_identity=sources[c.IMPORT]['states'],
        publication_failed=True, elapsed_seconds=.9633648640010506, archive_identity=c.ARCHIVE_PIN,
        manifest_identity=c.MANIFEST_PIN, export_receipt_identity=c.EXPORT_PIN, blob_etag='"0xABC"',
        replica_directory=str(dest), original_source_on_receiver_live_verified=False,
        original_audit_declaration=c.original.AUDIT)
    old = dict(flags, schema=c.original.SCHEMA, phase='import', status='fail', producer_revision=c.EXPORT,
        source_binding=sources[c.EXPORT]['binding'], source_stat_identity=sources[c.EXPORT]['states'],
        error_type='ValueError', elapsed_seconds=.5756366139976308)
    boot = dict(status='pass', producer_revision=c.BOOTSTRAP, source_binding=sources[c.BOOTSTRAP]['binding'],
        original_source=sources[c.EXPORT]['binding'], source_inputs_rehashed_after=True, outputs_sealed=True,
        parent=dict(created=True, state=[2065, 265398]))
    diag = dict(status='pass', stage='complete', producer_revision=c.IMPORT, receipt_identity=c.FAILED_PIN,
        saved_status='fail', saved_flags=dict(publication_failed=True, blob_cleanup_verified=False),
        source_and_receipts_prepost=True, replica_prepost_verified=True, replica_complete_original_hash_modes=True,
        replica_files=36, replica_original_bytes=20712157, destination=dict(inode=[2065, 265424]),
        parent=dict(inode=[2065, 265398]))
    folders = {c.IMPORT: c.ROOT/f'results/vcoco-observation-replica-import-{c.IMPORT}',
        c.EXPORT: c.ROOT/f'results/vcoco-observation-replica-import-{c.EXPORT}',
        c.BOOTSTRAP: c.ROOT/'results/vcoco-replica-parent-bootstrap-v1'}
    table = {code/c.DIAGNOSTIC: diag}
    for v, report in ((c.IMPORT, failed), (c.EXPORT, old), (c.BOOTSTRAP, boot)):
        table[folders[v]/'report.json'] = report
        if v != c.BOOTSTRAP:
            table[folders[v]/'manifest.json'] = manifests
            table[folders[v]/'export-receipt.json'] = export
    table.update({dest/'banks/native.json': native, dest/'banks/host.json': host,
                  dest/'banks/proof.json': endpoint_proof})
    real_read, real_lstat, real_iterdir = Path.read_bytes, Path.lstat, Path.iterdir
    def read(path): return b'authored exported metadata' if path.name == 'export-receipt.json' else real_read(path)
    def lstat(path):
        if path in table: return SimpleNamespace(st_uid=0, st_gid=0, st_mode=0o100400)
        return real_lstat(path)
    def iterdir(path):
        if path in folders.values(): return iter(p for p in table if p.parent == path)
        return real_iterdir(path)
    monkeypatch.setattr(Path, 'read_bytes', read); monkeypatch.setattr(Path, 'lstat', lstat)
    monkeypatch.setattr(Path, 'iterdir', iterdir)
    monkeypatch.setattr(c.rt, 'pinned', lambda p, *_: deepcopy(table[Path(p)]))
    monkeypatch.setattr(c, 'directory', lambda *_: None)
    def state(path):
        if Path(path) == dest.parent: return (2065, 265398)
        if Path(path) == dest: return (2065, 265424)
        return (1, 2)
    monkeypatch.setattr(c.original, 'snapshot', state)
    monkeypatch.setattr(c.original, 'validate_manifest', lambda m, v: m if v == c.EXPORT else pytest.fail('wrong export'))
    monkeypatch.setattr(c.original, 'receipt_from_base64', lambda *_: (b'authored', deepcopy(export)))
    monkeypatch.setattr(c.original, 'replica_identity', lambda *_: deepcopy(bankfiles))
    monkeypatch.setattr(c.original, 'replica_state', lambda: 'saved-full36')
    monkeypatch.setattr(c.original.bank.rgb_inputs, 'read_inputs', lambda *_args, **_kwargs: deepcopy(public))
    monkeypatch.setattr(c.original.bank, 'configuration', lambda *_: ({}, {}))
    monkeypatch.setattr(c.rt, 'identity', lambda *_args, **_kwargs: dict(bytes=1, sha256='a'*64))
    value = c.qualified_inputs(code)
    assert value['original_replica_source'] == sources[c.IMPORT]['binding']
    assert value['import_failed_pin'] == c.FAILED_PIN and value['images'] == public['images']
    assert value['original_bank_source_on_receiver_live_verified'] is False
    failed['source_binding'] = sources[c.EXPORT]['binding']
    with pytest.raises(ValueError): c.qualified_inputs(code)


def host_fixture(monkeypatch, tmp_path):
    code = tmp_path/'code'; code.mkdir(); (code/'frozen.py').write_text('authored tiny source\n')
    out = tmp_path/'result'; monkeypatch.setattr(c, 'OUTPUT', out)
    monkeypatch.setattr(c.sys, 'platform', 'linux'); monkeypatch.setattr(c.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(c.os, 'uname', lambda: SimpleNamespace(nodename='scenesmith-ncc-h100-01'))
    before = dict(binding=dict(producer_revision='f'*40), states='tiny')
    inputs = dict(files={'opaque': {'bytes': 1, 'sha256': 'a'*64}}, states={'opaque': [1, 2]},
                  blob_etag='"0xABC"', original_replica_source=dict(status='fail'))
    events = []
    def source(*_): events.append('source'); return deepcopy(before)
    def qualified(*_): events.append('inputs'); return deepcopy(inputs)
    monkeypatch.setattr(c, 'source', source); monkeypatch.setattr(c, 'qualified_inputs', qualified)
    class Response:
        status = 202
        def __enter__(self): return self
        def __exit__(self, *_): pass
    class Blob:
        def __init__(self, url, revision, *, managed_identity):
            assert url == 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+c.EXPORT+'.tar'
            assert revision == c.EXPORT and managed_identity is True
            assert out.stat().st_mode & 0o777 == 0o500 and (out/'report.json').stat().st_mode & 0o777 == 0o400
            events.append('blob')
        def request(self, method, *, headers):
            assert method == 'DELETE' and headers == {'If-Match': '"0xABC"'}
            events.append('delete'); return Response()
    monkeypatch.setattr(c.original.transport, 'Blob', Blob)
    return code, out, events, before, inputs, Response


def test_one_delete_after_strict_seal_and_fullposthash_saved_samefd(monkeypatch, tmp_path):
    code, out, events, _, _, _ = host_fixture(monkeypatch, tmp_path)
    report = c.run(code, 'f'*40); saved = json.loads((out/'report.json').read_bytes())
    assert report == saved and report['status'] == 'pass'
    assert report['delete_attempts'] == 1 and report['single_etag_DELETE_202'] is report['blob_cleanup_verified'] is True
    assert events.count('blob') == events.count('delete') == 1
    assert events[:4] == ['source', 'inputs', 'source', 'inputs']
    assert events[-2:] == ['source', 'inputs']
    assert report['source_inputs_rehashed_after'] is report['outputs_sealed'] is True
    assert all(report[k] is False for k in ('import_replayed', 'model_execution', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified'))
    assert {p.name for p in out.iterdir()} == {'report.json'}
    assert out.stat().st_mode & 0o777 == 0o500 and (out/'report.json').stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize('fault', ['source', 'inputs', 'post_inputs', 'expired', 'DELETE', 'DELETE_status', 'after_DELETE'])
def test_failure_is_persisted_no_import_or_secret_or_retry(monkeypatch, tmp_path, fault):
    code, out, events, before, inputs, response = host_fixture(monkeypatch, tmp_path)
    if fault == 'source': monkeypatch.setattr(c, 'source', lambda *_: (_ for _ in ()).throw(ValueError('secret source')))
    elif fault == 'inputs': monkeypatch.setattr(c, 'qualified_inputs', lambda *_: (_ for _ in ()).throw(ValueError('secret GT')))
    elif fault in ('post_inputs', 'after_DELETE'):
        count = [0]
        def changed(*_):
            count[0] += 1
            if count[0] >= (2 if fault == 'post_inputs' else 4): return dict(inputs, bad=True)
            return deepcopy(inputs)
        monkeypatch.setattr(c, 'qualified_inputs', changed)
    elif fault == 'expired': monkeypatch.setattr(c, 'check', lambda *_: (_ for _ in ()).throw(TimeoutError('secret')))
    elif fault == 'DELETE_status': response.status = 412
    elif fault == 'DELETE':
        class Blob:
            def __init__(self, *_args, **_kwargs): pass
            def request(self, *_args, **_kwargs):
                events.append('delete'); raise RuntimeError('TOKEN_SECRET_https://untrusted?sig=secret')
        monkeypatch.setattr(c.original.transport, 'Blob', Blob)
    report = c.run(code, 'f'*40); saved = json.loads((out/'report.json').read_bytes())
    assert report == saved and report['status'] == 'fail' and events.count('delete') <= 1
    assert 'secret' not in (out/'report.json').read_text().lower()
    if fault in ('source', 'inputs', 'post_inputs', 'expired'): assert not events.count('delete')
    if fault in ('DELETE', 'DELETE_status', 'after_DELETE'):
        assert report['publication_failed'] is True and report['publication_failure_stage'] == 'after_seal'
        assert report['delete_attempts'] == 1 and report['blob_cleanup_verified'] is False
    if fault == 'after_DELETE': assert report['single_etag_DELETE_202'] is True


def test_foreign_result_not_changed_no_cleanup_or_callback(monkeypatch, tmp_path):
    code, out, events, *_ = host_fixture(monkeypatch, tmp_path)
    out.mkdir(); foreign = out/'foreign'; foreign.write_text('owned by somebody else')
    before = (foreign.stat().st_ino, foreign.read_bytes(), out.stat().st_mode)
    with pytest.raises(ValueError): c.run(code, 'f'*40)
    assert before == (foreign.stat().st_ino, foreign.read_bytes(), out.stat().st_mode)
    assert 'blob' not in events and {p.name for p in out.iterdir()} == {'foreign'}


def test_completion_consumer_requires_newpass_not_historicalfail(monkeypatch, tmp_path):
    code, out, _, _, inputs, _ = host_fixture(monkeypatch, tmp_path)
    run = c.run(code, 'f'*40)
    run.update(source_binding=dict(helpers={'authored.py': dict(bytes=7, sha256='a'*64)}), input_proof=inputs)
    def pinned(_path, _pin, _maximum): return deepcopy(run)
    monkeypatch.setattr(c.rt, 'source', lambda *_: run['source_binding'])
    monkeypatch.setattr(c.rt, 'identity', lambda *_args, **_kwargs: dict(bytes=7, sha256='a'*64))
    monkeypatch.setattr(c.rt, 'pinned', pinned); monkeypatch.setattr(c, 'directory', lambda *_: None)
    monkeypatch.setattr(c.original.bank, 'source_state', lambda *_: 'sealed')
    real_lstat = Path.lstat
    def lstat(path):
        s = real_lstat(path)
        if path == out/'report.json': return SimpleNamespace(st_uid=0, st_gid=0, st_mode=s.st_mode)
        return s
    monkeypatch.setattr(Path, 'lstat', lstat)
    value = c.authenticate_completion(code, 'f'*40, dict(bytes=7, sha256='a'*64))
    assert value['completion_revision'] == 'f'*40 and value['original_replica_source']['status'] == 'fail'
    for key, bad in (('status', 'fail'), ('single_etag_DELETE_202', False), ('delete_attempts', 2),
                     ('blob_cleanup_verified', False), ('source_inputs_rehashed_after', False), ('import_replayed', True)):
        original = run[key]; run[key] = bad
        with pytest.raises(ValueError): c.authenticate_completion(code, 'f'*40, dict(bytes=7, sha256='a'*64))
        run[key] = original


def test_no_old_publish_or_import_or_unqualified_peer_or_media():
    src = Path(c.__file__).read_text()
    assert 'original.publish(' not in src and 'original.run(' not in src and 'original.install(' not in src
    assert '.request(\'DELETE\'' in src and '.request(\'GET\'' not in src and '.request(\'HEAD\'' not in src
    assert 'np.load' not in src and 'decode_rgb(' not in src and 'verify_azure_peer(' not in src
    assert 'publication.publish(' in src and 'receipt_from_base64(' in src
    assert c.DIAGNOSTIC_PIN == dict(bytes=3473, sha256='3110dc383ea38bfdf7d141cd5e69374fb4d9fb7052bf4eeac8b311f3f17b5358')

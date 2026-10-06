"""Tiny authored opaque bytes only: no photos, actual NPZ, Torch or HTTP."""
import ast
import base64
import copy
import hashlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest
import vcoco_full_public_replica as v

R = '1'*40


@pytest.fixture(autouse=True)
def authored_group_owner(monkeypatch, tmp_path):
    # macOS pytest fixtures inherit basetemp group; production private root
    # and real process both have gid0. Preserve the production equality gate.
    monkeypatch.setattr(v.os, 'getgid', lambda: tmp_path.lstat().st_gid)


def readonly(path, raw):
    path.write_bytes(raw); path.chmod(0o400); return v.pin(raw)


def fixture(monkeypatch):
    images = []; banks = []; payload = {}
    for i in range(48):
        iid = f'{i:032x}'; rgb = f'authored JPEG bytes {i}'.encode(); npz = f'authored opaque NPZ {i}'.encode()
        images.append(dict(image_id=iid, file=f'image_{i:06d}.jpg', width=16, height=12, **v.pin(rgb)))
        payload['inputs/'+images[-1]['file']] = rgb; payload[f'banks/image_{i:06d}.npz'] = npz
        banks.append(dict(image_id=iid, original_frame_index=0, bank_index=i, image_size=[12, 16], input_file=images[-1]['file'],
            input_identity=v.pin(rgb), person_native_queries=900, person_postprocessor_rows=2, person_retained_rows=1,
            person_ids=[f'image:{iid}/person/retained:000000'], owl_patches=3600, original_slot=i, acquired_ordinal=i,
            file=f'image_{i:06d}.npz', identity=v.pin(npz),
            arrays={n: dict(shape=[1], dtype='<f4', sha256=hashlib.sha256(n.encode()).hexdigest()) for n in v.endpoint.seam.KEYS}))
    inputs = dict(schema=v.endpoint.seam.public.SCHEMA, images=images); public = v.pin(v.encode(inputs)); monkeypatch.setattr(v, 'PUBLIC', public)
    projection = dict(schema='world_reward.public_endpoint_bank_reference.v1', inputs=inputs, manifest_identity=public, banks=banks)
    payload['inputs/manifest.json'] = v.encode(inputs); payload['public-reference.json'] = v.encode(projection)
    m = dict(schema=v.SCHEMA, export_revision=R, original_source_declaration=v.DECLARATION, files={n: v.pin(p) for n, p in payload.items()})
    return m, projection, payload


def archive_raw(manifest, payload, members=None, format=tarfile.USTAR_FORMAT):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=format) as archive:
        for name, raw, kind in members or [('manifest.json', v.encode(manifest), tarfile.REGTYPE),
                *[(n, payload[n], tarfile.REGTYPE) for n in sorted(payload)]]:
            row = tarfile.TarInfo(name); row.size = len(raw); row.mode = 0o400; row.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): row.linkname = 'foreign'
            archive.addfile(row, io.BytesIO(raw))
    return stream.getvalue()


def test_complete98_public_leaves_and99_member_offsets(monkeypatch, tmp_path):
    m, projection, payload = fixture(monkeypatch); raw = archive_raw(m, payload); path = tmp_path/'archive'; ap = readonly(path, raw)
    assert len(v.NAMES) == len(m['files']) == 98 and v.validate_manifest(m, R) == m
    assert v.validate_projection(projection, m['files']) == projection
    manifest, table = v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)
    assert manifest == m and len(table) == 99
    with path.open('rb') as stream:
        for name, offset, size in table[1:]: stream.seek(offset); assert stream.read(size) == payload[name]
    assert not set(v.PINS) & set(m['files']) and all(n not in m['files'] for n in ('banks/native.json', 'banks/proof.json', 'banks/report.json'))


@pytest.mark.parametrize('change', ['hole', 'extra', 'alias', 'revision', 'source', 'zero', 'bool', 'hash', 'large', 'public'])
def test_exact_manifest_rejects(monkeypatch, change):
    m, _, _ = fixture(monkeypatch); m = copy.deepcopy(m)
    if change == 'hole': del m['files']['banks/image_000047.npz']
    elif change == 'extra': m['files']['private/report.json'] = v.pin(b'x')
    elif change == 'alias': m['files']['banks/./image_000000.npz'] = m['files'].pop('banks/image_000000.npz')
    elif change == 'revision': m['export_revision'] = '2'*40
    elif change == 'source': m['original_source_declaration'] = dict(v.DECLARATION, files=339)
    elif change == 'zero': m['files']['banks/image_000000.npz']['bytes'] = 0
    elif change == 'bool': m['files']['banks/image_000000.npz']['bytes'] = True
    elif change == 'hash': m['files']['banks/image_000000.npz']['sha256'] = 'bad'
    elif change == 'large': m['files']['banks/image_000000.npz']['bytes'] = 17 << 20
    elif change == 'public': m['files']['inputs/manifest.json'] = v.pin(b'other')
    with pytest.raises(ValueError): v.validate_manifest(m, R)


@pytest.mark.parametrize('change', ['private', 'hole', 'duplicate', 'slot', 'renumber', 'frame', 'boxcount', 'id', 'queries', 'patches', 'shape', 'input', 'pin'])
def test_projection_metadata_rejects(monkeypatch, change):
    m, p, _ = fixture(monkeypatch); p = copy.deepcopy(p)
    if change == 'private': p['source_proof'] = {}
    elif change == 'hole': p['banks'].pop()
    elif change == 'duplicate': p['inputs']['images'][1] = p['inputs']['images'][0]
    elif change == 'slot': p['banks'][0]['original_slot'] = True
    elif change == 'renumber': p['banks'][1]['bank_index'] = 0
    elif change == 'frame': p['banks'][0]['original_frame_index'] = 1
    elif change == 'boxcount': p['banks'][0]['person_retained_rows'] = 3
    elif change == 'id': p['banks'][0]['person_ids'] = [123]
    elif change == 'queries': p['banks'][0]['person_native_queries'] = 899
    elif change == 'patches': p['banks'][0]['owl_patches'] = 100
    elif change == 'shape': next(iter(p['banks'][0]['arrays'].values()))['shape'] = [-1]
    elif change == 'input': p['banks'][0]['input_file'] = '../old.jpg'
    elif change == 'pin': p['banks'][0]['identity'] = v.pin(b'other')
    with pytest.raises(ValueError): v.validate_projection(p, m['files'])


@pytest.mark.parametrize('change', ['symlink', 'hardlink', 'duplicate', 'unknown', 'escape', 'alias', 'prefix', 'pax', 'mode', 'uid', 'size', 'hash', 'tail', 'truncated', 'padding', 'first'])
def test_archive_firewall(monkeypatch, tmp_path, change):
    m, _, payload = fixture(monkeypatch)
    members = [('manifest.json', v.encode(m), tarfile.REGTYPE), *[(n, payload[n], tarfile.REGTYPE) for n in sorted(payload)]]
    if change in ('symlink', 'hardlink'): members[1] = (members[1][0], b'', tarfile.SYMTYPE if change == 'symlink' else tarfile.LNKTYPE)
    elif change == 'duplicate': members.append(members[1])
    elif change == 'unknown': members.append(('private.json', b'x', tarfile.REGTYPE))
    elif change == 'escape': members[1] = ('../other', members[1][1], tarfile.REGTYPE)
    elif change == 'alias': members[1] = ('./'+members[1][0], members[1][1], tarfile.REGTYPE)
    elif change == 'prefix': members[1] = ('x'*101+'/other', b'x', tarfile.REGTYPE)
    elif change == 'pax': members[1] = ('x'*101, b'x', tarfile.REGTYPE)
    elif change == 'hash': members[1] = (members[1][0], b'bad', tarfile.REGTYPE)
    elif change == 'first': members[0], members[1] = members[1], members[0]
    raw = archive_raw(m, payload, members, tarfile.PAX_FORMAT if change == 'pax' else tarfile.USTAR_FORMAT)
    if change in ('mode', 'uid', 'size'):
        header = tarfile.TarInfo.frombuf(raw[:512], 'utf-8', 'strict')
        if change == 'mode': header.mode = 0o444
        elif change == 'uid': header.uid = 1
        else: header.size += 1
        raw = header.tobuf(format=tarfile.USTAR_FORMAT)+raw[512:]
    elif change == 'tail': raw += b'x'+b'\0'*511
    elif change == 'truncated': raw = raw[:700]
    elif change == 'padding':
        pos = 512+len(v.encode(m)); raw = raw[:pos]+b'x'+raw[pos+1:]
    path = tmp_path/'archive'; ap = readonly(path, raw)
    with pytest.raises((ValueError, tarfile.HeaderError)): v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)


def test_independent_pins_deadline_links(monkeypatch, tmp_path):
    m, _, payload = fixture(monkeypatch); path = tmp_path/'archive'; ap = readonly(path, archive_raw(m, payload)); mp = v.pin(v.encode(m))
    with pytest.raises(ValueError): v.verify_archive(path, dict(ap, sha256='0'*64), mp, R, time.monotonic()+5)
    with pytest.raises(ValueError): v.verify_archive(path, ap, dict(mp, bytes=v.MAX_CONTROL+1), R, time.monotonic()+5)
    with pytest.raises(TimeoutError): v.verify_archive(path, ap, mp, R, time.monotonic()-1)
    symlink = tmp_path/'link'; symlink.symlink_to(path)
    with pytest.raises(ValueError): v.rt.identity(symlink)
    hard = tmp_path/'hard'; os.link(path, hard)
    with pytest.raises(ValueError): v.rt.identity(path)


def local_private(path, mode, names=None):
    s = v.rt.canonical(path).lstat()
    v.rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == mode and s.st_uid == os.getuid()
        and (names is None or {p.name for p in path.iterdir()} == set(names)), 'Authored owned test namespace')


def test_install_atomic_all98_no_overwrite_and_owned_partial(monkeypatch, tmp_path):
    tmp_path.chmod(0o700); m, p, payload = fixture(monkeypatch); incoming = tmp_path/'archive'; ap = readonly(incoming, archive_raw(m, payload))
    manifest, table = v.verify_archive(incoming, ap, v.pin(v.encode(m)), R, time.monotonic()+5)
    target = tmp_path/'replica'; monkeypatch.setattr(v, 'DEST', target); monkeypatch.setattr(v, 'private_directory', local_private)
    def rename(stage, dest):
        if dest.exists(): raise FileExistsError()
        stage.rename(dest)
    monkeypatch.setattr(v.atomic, 'rename_noreplace', rename)
    stage = tmp_path/'stage'; v.install(incoming, stage, manifest, table, time.monotonic()+5)
    assert not stage.exists() and stat.S_IMODE(target.stat().st_mode) == 0o700
    assert len(list(target.rglob('*'))) == 100 and stat.S_IMODE((target/'inputs').stat().st_mode) == 0o500
    assert all(v.rt.identity(target/n) == pin for n, pin in m['files'].items())
    assert v.rt.pinned(target/'public-reference.json', m['files']['public-reference.json']) == p
    with pytest.raises(ValueError): v.install(incoming, stage, manifest, table, time.monotonic()+5)
    assert not stage.exists()


def test_failed_install_removes_only_owned_partial(monkeypatch, tmp_path):
    tmp_path.chmod(0o700); m, _, payload = fixture(monkeypatch); path = tmp_path/'archive'; ap = readonly(path, archive_raw(m, payload))
    manifest, table = v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)
    monkeypatch.setattr(v, 'DEST', tmp_path/'replica'); monkeypatch.setattr(v, 'private_directory', local_private)
    checks = 0
    def fail(deadline):
        nonlocal checks
        checks += 1
        if checks == 3: raise TimeoutError()
    monkeypatch.setattr(v, 'check', fail); stage = tmp_path/'stage'
    with pytest.raises(TimeoutError): v.install(path, stage, manifest, table, time.monotonic()+5)
    assert not stage.exists() and not v.DEST.exists()


def test_pack_actual_blockwriter_commit_exclusive(monkeypatch, tmp_path):
    m, _, payload = fixture(monkeypatch); paths = {}
    for ordinal, (name, raw) in enumerate(payload.items()): paths[name] = tmp_path/str(ordinal); readonly(paths[name], raw)
    calls = []
    class Response:
        status = 201
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Blob:
        def request(self, method, data=None, query='', headers=None): calls.append((method, data, query, headers)); return Response()
    writer = v.transport.BlockWriter(Blob()); v.pack(writer, m, paths, time.monotonic()+5); ap = writer.finish()
    blocks = [c[1] for c in calls if c[2].startswith('comp=block&')]; raw = b''.join(blocks)
    assert ap == v.pin(raw) and calls[-1][2] == 'comp=blocklist' and calls[-1][3]['If-None-Match'] == '*'
    p = tmp_path/'packed'; readonly(p, raw); assert v.verify_archive(p, ap, v.pin(v.encode(m)), R, time.monotonic()+5)[0] == m


def receipt(phase='export'):
    return dict(schema=v.SCHEMA, phase=phase, status='pass', producer_revision=R, source_binding=dict(producer_revision=R),
        original_source_declaration=v.DECLARATION, files=98, budget_seconds=300, source_inputs_rehashed_after=True,
        outputs_sealed=True, blob_etag='"unique-etag"', archive_identity=v.pin(b'archive'), manifest_identity=v.pin(b'manifest'),
        **{k: False for k in ('models_loaded', 'GPU_used', 'reference_metadata_read', 'RGB_NPZ_decoded', 'quality_verified', 'ownership_verified', 'adoption')})


@pytest.mark.parametrize('change', ['fail', 'unsealed', 'private', 'phase', 'source', 'etag', 'count', 'error', 'pin'])
def test_independent_export_receipt_rejects(change):
    r = receipt()
    if change == 'fail': r['status'] = 'fail'
    elif change == 'unsealed': r['outputs_sealed'] = False
    elif change == 'private': r['reference_metadata_read'] = True
    elif change == 'phase': r['phase'] = 'import'
    elif change == 'source': r['source_binding']['producer_revision'] = '2'*40
    elif change == 'etag': r['blob_etag'] = 'unsafe value'
    elif change == 'count': r['files'] = 97
    elif change == 'error': r['failure_stage'] = 'pack'
    raw = v.encode(r); expected = v.pin(raw)
    if change == 'pin': expected['sha256'] = '0'*64
    with pytest.raises(ValueError): v.export_receipt(raw, expected, R)


def test_cli_fixed_profile_no_overrides_duplicate_controls():
    assert v.arguments(['--phase', 'export']).phase == 'export'
    with pytest.raises(ValueError): v.arguments(['--phase', 'export', '--archive-bytes', '1'])
    with pytest.raises(ValueError): v.arguments(['--phase', 'import'])
    raw = v.encode(receipt()); a = ['--phase', 'import', '--export-revision', R, '--export-receipt-base64', base64.b64encode(raw).decode()]
    for n, p in [('archive', v.pin(b'archive')), ('manifest', v.pin(b'manifest')), ('export-receipt', v.pin(raw))]: a += ['--'+n+'-bytes', str(p['bytes']), '--'+n+'-sha256', p['sha256']]
    assert v.arguments(a).export_receipt_pin == v.pin(raw)
    with pytest.raises(ValueError): v.arguments(a+['--archive-bytes='+str(len(b'archive'))])
    with pytest.raises(ValueError): v.arguments(['--phase', 'export', '--phase=export'])


def fake_lifecycle(monkeypatch, tmp_path):
    root = tmp_path/'root'; (root/'results').mkdir(parents=True); root.chmod(0o700)
    data = tmp_path/'data'; data.mkdir(mode=0o700); monkeypatch.setattr(v, 'ROOT', root); monkeypatch.setattr(v, 'DEST', data/'replica')
    m, projection, payload = fixture(monkeypatch); source_dir = tmp_path/'source'; source_dir.mkdir()
    paths = {}
    for i, (name, raw) in enumerate(payload.items()):
        if name == 'public-reference.json': continue
        paths[name] = source_dir/str(i); readonly(paths[name], raw)
    original = dict(unchanged=True)
    monkeypatch.setattr(v, 'source', lambda code, rev: dict(binding=dict(producer_revision=rev), states={}))
    monkeypatch.setattr(v, 'sender_inputs', lambda code, binding, deadline: (projection, dict(paths), m['files'], original))
    monkeypatch.setattr(v.transport, 'verify_azure_peer', lambda phase: None)
    monkeypatch.setattr(v, 'private_directory', local_private)
    def local_installed(manifest):
        local_private(v.DEST, 0o700, {'inputs', 'banks', 'public-reference.json'})
        for folder in ('inputs', 'banks'):
            local_private(v.DEST/folder, 0o500, {Path(n).name for n in v.NAMES if n.startswith(folder+'/')})
        pins = {str(v.DEST/n): v.rt.identity(v.DEST/n, 16 << 20) for n in v.NAMES}
        assert pins == {str(v.DEST/n): p for n, p in manifest['files'].items()}
        assert all(stat.S_IMODE((v.DEST/n).stat().st_mode) == 0o400 for n in v.NAMES)
        value = v.validate_projection(v.rt.pinned(v.DEST/'public-reference.json', manifest['files']['public-reference.json']), manifest['files'])
        return dict(files=pins, states=v.tree_state(v.DEST), projection=value)
    monkeypatch.setattr(v, 'installed', local_installed)
    def rename(stage, dest):
        if dest.exists(): raise FileExistsError()
        stage.rename(dest)
    monkeypatch.setattr(v.atomic, 'rename_noreplace', rename)
    calls = []; blocks = {}; stored = {}; delete_status = [202]
    class Response(io.BytesIO):
        def __init__(self, raw=b'', status=201, headers=None): super().__init__(raw); self.status = status; self.headers = headers or {}
    class Blob:
        def __init__(self, url, revision, managed_identity):
            assert managed_identity is True and url.endswith('articulated-runtime-'+revision+'.tar'); self.revision = revision
        def request(self, method, data=None, query='', headers=None):
            calls.append((method, query, headers))
            if query.startswith('comp=block&'): blocks.setdefault(self.revision, []).append(bytes(data)); return Response()
            if query == 'comp=blocklist':
                assert headers['If-None-Match'] == '*'; stored[self.revision] = b''.join(blocks[self.revision]); return Response()
            if method == 'HEAD': return Response(status=200, headers={'Content-Length': str(len(stored[self.revision])), 'ETag': '"unique-etag"'})
            if method == 'GET': return Response(stored[self.revision], 200, {'Content-Length': str(len(stored[self.revision]))})
            if method == 'DELETE': assert headers == {'If-Match': '"unique-etag"'}; return Response(status=delete_status[0])
            raise AssertionError(method)
    monkeypatch.setattr(v.transport, 'Blob', Blob)
    return root, projection, calls, delete_status


def test_actual_export_import_lifecycle_all98_and_one_aftersealed_delete(monkeypatch, tmp_path):
    root, projection, calls, _ = fake_lifecycle(monkeypatch, tmp_path)
    export = v.run(v.arguments(['--phase', 'export']), tmp_path/'code', R)
    assert export['status'] == 'pass' and export['outputs_sealed'] and export['source_inputs_rehashed_after']
    out = root/f'results/vcoco-full-public-replica-export-{R}'; raw = (out/'report.json').read_bytes(); ep = v.pin(raw)
    assert v.export_receipt(raw, ep, R) == export
    args = SimpleNamespace(phase='import', export_revision=R, export_receipt_base64=base64.b64encode(raw).decode(),
        export_receipt_pin=ep, archive_pin=export['archive_identity'], manifest_pin=export['manifest_identity'])
    imported = v.run(args, tmp_path/'code', '2'*40)
    assert imported['status'] == 'pass' and imported['outputs_sealed'] and imported['archive_removed']
    assert imported['delete_attempts'] == 1 and imported['single_etag_DELETE_202'] and imported['blob_cleanup_verified']
    assert v.rt.pinned(v.DEST/'public-reference.json', v.pin(v.encode(projection))) == projection
    technical = root/f'results/vcoco-full-public-replica-import-{"2"*40}'
    assert {p.name for p in technical.iterdir()} == {'report.json', 'manifest.json', 'export-receipt.json'}
    assert not (technical/'archive.tar').exists() and len([c for c in calls if c[0] == 'DELETE']) == 1
    assert stat.S_IMODE(technical.stat().st_mode) == 0o500 and all(stat.S_IMODE(p.stat().st_mode) == 0o400 for p in technical.iterdir())


def test_sender_error_retains_bounded_stage_not_exception_text(monkeypatch, tmp_path):
    root, _, calls, _ = fake_lifecycle(monkeypatch, tmp_path)
    def failed(*args): raise ValueError('SECRET_OR_PRIVATE_PATH_SHOULD_NEVER_APPEAR')
    monkeypatch.setattr(v, 'sender_inputs', failed)
    report = v.run(v.arguments(['--phase', 'export']), tmp_path/'code', R)
    assert report['status'] == 'fail' and report['failure_stage'] == 'sender' and report['error_type'] == 'ValueError'
    assert not report['source_inputs_rehashed_after'] and report['outputs_sealed'] and not calls
    raw = (root/f'results/vcoco-full-public-replica-export-{R}'/'report.json').read_bytes()
    assert b'SECRET' not in raw
    with pytest.raises(ValueError): v.export_receipt(raw, v.pin(raw), R)


def test_failed_delete_demotes_original_report_fd_and_never_retries(monkeypatch, tmp_path):
    root, _, calls, delete_status = fake_lifecycle(monkeypatch, tmp_path)
    exported = v.run(v.arguments(['--phase', 'export']), tmp_path/'code', R)
    raw = (root/f'results/vcoco-full-public-replica-export-{R}'/'report.json').read_bytes(); delete_status[0] = 500
    args = SimpleNamespace(phase='import', export_revision=R, export_receipt_base64=base64.b64encode(raw).decode(),
        export_receipt_pin=v.pin(raw), archive_pin=exported['archive_identity'], manifest_pin=exported['manifest_identity'])
    report = v.run(args, tmp_path/'code', '2'*40)
    assert report['status'] == 'fail' and report['publication_failed'] and report['publication_failure_stage'] == 'after_seal'
    assert report['delete_attempts'] == 1 and not report['blob_cleanup_verified'] and not report['single_etag_DELETE_202']
    assert report['archive_removed'] and v.DEST.exists() and len([c for c in calls if c[0] == 'DELETE']) == 1
    actual = v.rt.pinned(root/f'results/vcoco-full-public-replica-import-{"2"*40}'/'report.json', v.pin(v.encode(report)))
    assert actual == report


def test_owned_download_actual_transport_never_overwrites(monkeypatch, tmp_path):
    raw = b'authored incoming archive'; path = tmp_path/'download'; owners = []
    class Blob:
        def request(self, method):
            assert method == 'GET'; r = io.BytesIO(raw); r.status = 200; r.headers = {'Content-Length': str(len(raw))}; return r
    v.transport.download(Blob(), v.OwnedDownload(path, owners), v.pin(raw))
    assert len(owners) == 1 and path.read_bytes() == raw
    with pytest.raises(ValueError): v.OwnedDownload(path, owners).open('xb')
    with pytest.raises(FileExistsError): v.transport.download(Blob(), v.OwnedDownload(path, []), v.pin(raw))


def test_atomic_race_foreign_destination_untouched(monkeypatch, tmp_path):
    tmp_path.chmod(0o700); m, _, payload = fixture(monkeypatch); path = tmp_path/'archive'; ap = readonly(path, archive_raw(m, payload))
    manifest, table = v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)
    monkeypatch.setattr(v, 'DEST', tmp_path/'replica'); monkeypatch.setattr(v, 'private_directory', local_private)
    def collision(stage, dest): dest.mkdir(); (dest/'foreign').write_bytes(b'untouched'); raise FileExistsError()
    monkeypatch.setattr(v.atomic, 'rename_noreplace', collision); stage = tmp_path/'stage'
    with pytest.raises(FileExistsError): v.install(path, stage, manifest, table, time.monotonic()+5)
    assert not stage.exists() and (v.DEST/'foreign').read_bytes() == b'untouched'


def test_source_only_no_host_numpy_torch_or_profile_mutation():
    path = Path(v.__file__); source = path.read_text(); tree = ast.parse(source)
    imports = [n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names]
    assert not {'numpy', 'torch', 'PIL'} & set(imports)
    assert 'endpoint.validate_native(native, proof, REV' in source and 'endpoint.host_inputs(' not in source
    assert 'endpoint.seam.validate_records(' not in source and 'endpoint.native(' not in source
    assert 'atomic.rename_noreplace' in source and 'managed_identity=True' in source
    assert 'sender_source_live_verified=False' in source and 'sender_runtime_live_verified=False' in source
    assert not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
        and t.value.id in ('endpoint', 'transport', 'acquisition') for t in n.targets) for n in ast.walk(tree))
    script = 'import sys; sys.path[:0]=["infra","src"]; import vcoco_full_public_replica; assert not any(n in sys.modules for n in ("numpy","torch","PIL"))'
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wrapper_bound_own_entry_and_shell_syntax():
    wrapper = Path(v.__file__).with_name('run_vcoco_full_public_replica.sh'); text = wrapper.read_text()
    assert 'ulimit -v 1048576' in text and '315s env -i' in text and 'python3 -I -B' in text
    assert 'run_vcoco_full_public_replica/code' in text and 'docker' not in text and 'gpus' not in text
    assert subprocess.run(['bash', '-n', str(wrapper)], capture_output=True).returncode == 0

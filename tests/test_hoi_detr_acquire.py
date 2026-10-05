"""Tiny manufactured source/opaque weights; no real downloads or model decode."""
import ast
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('wr_hoi_acquire_test', REPO/'infra/hoi_detr_acquire.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Response(io.BytesIO):
    def __init__(self, raw, url, *, status=200, encoding='identity', length=None, mime='application/octet-stream'):
        super().__init__(raw); self.url = url; self.status = status; self.headers = email.message.Message()
        self.headers['Content-Encoding'] = encoding; self.headers['Content-Length'] = str(len(raw) if length is None else length); self.headers['Content-Type'] = mime
    def geturl(self):
        return self.url
    def read(self, n=-1):
        assert 0 < n <= 1 << 20
        return super().read(n)


class Opener:
    def __init__(self, payload):
        self.rows = {u: Response(b, u) for u, b in payload.items()}; self.calls = []
    def open(self, request, timeout):
        assert 0 < timeout <= 30 and dict(request.header_items()) == {'Accept-encoding': 'identity'}
        self.calls.append(request.full_url)
        return self.rows[request.full_url]


def blob(raw, mode='100644'):
    return dict(mode=mode, bytes=len(raw), git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def pack(prefix, files, links, extra=None):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as z:
        d = tarfile.TarInfo(prefix+'/'); d.type = tarfile.DIRTYPE; z.addfile(d)
        for name, raw in files.items():
            n = tarfile.TarInfo(prefix+'/'+name); n.size = len(raw); n.mode = 0o664; z.addfile(n, io.BytesIO(raw))
        for name, target in links.items():
            n = tarfile.TarInfo(prefix+'/'+name); n.type = tarfile.SYMTYPE; n.linkname = target; z.addfile(n)
        if extra is not None:
            n, raw = extra; z.addfile(n, io.BytesIO(raw) if n.isfile() else None)
    return out.getvalue()


def repin(m, code, c):
    p = code/m.PROTOCOL
    if p.exists():
        p.chmod(0o644)
    p.write_text(json.dumps(c)); p.chmod(0o444)
    m.PROTOCOL_PIN = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def setup(m, tmp_path, monkeypatch):
    root = tmp_path/'root'; data = tmp_path/'data'/'hoi_detr_v1'; data.parent.mkdir()
    revision = 'a'*40; code = root/'jobs'/revision/m.JOB/'code'
    for name in m.HELPERS:
        p = code/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((REPO/name).read_bytes())
    # A legitimate complete closure may include more than the minimal helpers.
    (code/'extra-public-source.py').write_bytes(b'# Public immutable closure\n')
    (code.parent/'revision').write_text(revision+'\n'); (code.parent/'source-sha256').write_text('b'*64+'\n'); (root/'results').mkdir()
    c = json.loads((REPO/m.PROTOCOL).read_bytes()); c['data_root'] = str(data)
    source = c['source']; files = {}
    for name in set(source['required_retained_files']) | set(source['required_source_pins']):
        files[name] = b'# Tiny code not executed\n'
    files['LICENSE.txt'] = b'MIT License\nCopyright manufactured fixture\n'
    files.update({'mmdet/empty.py': b'', 'projects/code.py': b'raise RuntimeError("UPSTREAM MUST NOT EXECUTE")\n',
                  'assets/example.jpg': b'OPAQUE NOT AN IMAGE NEVER DECODE', 'assets/vendor/LICENSE': b'Original retained notice\n',
                  'demo/__pycache__/never.pyc': b'CACHE NEVER RETAIN', 'requirements/build.txt': b'tiny\n'})
    links = {n: r['target'] for n, r in source['symlinks'].items()}
    rows = {n: blob(b) for n, b in files.items()} | {n: blob(b.encode(), '120000') for n, b in links.items()}
    source.update(blob_count=len(rows), expanded_bytes=sum(r['bytes'] for r in rows.values()), git_root_tree_sha1=m.git_tree(rows))
    source['required_source_pins'] = {n: dict(bytes=len(files[n]), sha256=hashlib.sha256(files[n]).hexdigest()) for n in source['required_source_pins']}
    payload = {source['url']: pack(source['archive_prefix'], files, links)}
    for row in c['assets']:
        if row['file'] == 'weights/README.md':
            raw = b'---\r\nlicense: mit\r\n---\r\n'
        elif row['file'].startswith('notices/'):
            raw = b'MIT License\n' if 'co-detr' in row['file'] else b'Apache License Version 2.0\n'
        else:
            raw = b'OPAQUE TINY CHECKPOINT NEVER DESERIALIZE'
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()); payload[row['url']] = raw
    monkeypatch.setattr(m, 'DATA', data); repin(m, code, c)
    for p in sorted(code.rglob('*'), reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    for name in ('revision', 'source-sha256'):
        (code.parent/name).chmod(0o444)
    monkeypatch.setattr(m, '__file__', str(code/m.HELPERS[0]))
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(100_000_000_000, 0, 100_000_000_000))
    original = m.helpers
    def portable_helpers(code):
        rt, mp = original(code)
        def publish(a, b):
            os.link(a, b); a.unlink()
        mp.publish = publish
        return rt, mp
    monkeypatch.setattr(m, 'helpers', portable_helpers)
    return root, data, code, revision, c, files, links, Opener(payload)


def report(root):
    return json.loads((root/'results/hoi-detr-acquire-v1.json').read_bytes())


def test_complete_tiny_acquisition_and_original_licenses_before_weights(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    value = gate.acquire(root, code, rev, opener=opener)
    assert value['status'] == 'pass' and value['phase'] == 'complete' and len(opener.calls) == 6
    assert value['first_party_license_declarations_verified']
    assert value['source_archive']['all_git_blobs_authenticated'] and value['source_archive']['root_tree_reconstructed']
    assert value['source_archive']['original_link_texts_verified'] == 3 and not value['source_archive']['links_materialized']
    assert value['source_binding']['entries'] > len(gate.HELPERS)
    assert all(value[k] for k in ('source_rehashed_after', 'artifacts_rehashed_after', 'owned_partials_removed', 'owned_archive_removed'))
    assert (data/'source/hoi-detr/mmdet/empty.py').read_bytes() == b''
    assert (data/'source/hoi-detr/assets/vendor/LICENSE').read_bytes() == files['assets/vendor/LICENSE']
    assert not (data/'source/hoi-detr/assets/example.jpg').exists() and not (data/'source/hoi-detr/mmdet/.mim').exists()
    assert not (data/'source/hoi-detr/demo/__pycache__').exists()
    assert not list(data.rglob('*.part')) and not list((data/'.archives').iterdir())
    assert data.stat().st_mode & 0o777 == 0o555 and (root/c['report']).stat().st_mode & 0o777 == 0o444
    assert all(p.stat().st_mode & 0o777 == (0o555 if p.is_dir() else 0o444) for p in data.rglob('*'))
    inventory = json.loads((data/'source_manifest.json').read_bytes())
    assert len(inventory['complete_original_git_blobs']) == c['source']['blob_count']
    assert inventory['complete_original_git_blobs']['assets/example.jpg'] == blob(files['assets/example.jpg'])
    assert len(json.dumps(value)) < 15000
    assert value['archive_identity']['hash_basis'] == 'measured_archive_sha256_not_independent'
    assert value['scope']['training_overlap_status'].startswith('unknown')
    assert all(v is False for k, v in value['scope'].items() if k != 'training_overlap_status')
    assert all(v is False for k, v in value['future_runtime'].items() if k != 'required_next_gate')
    assert (data/'weights/epoch_5.pth').read_bytes().startswith(b'OPAQUE')


def test_license_preflight_happens_before_weight_open(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = opener.open
    def spy(request, timeout):
        if request.full_url == c['assets'][-1]['url']:
            assert all((data/r['file']).exists() for r in c['assets'][:-1])
            assert (data/'source/hoi-detr/LICENSE.txt').read_bytes() == files['LICENSE.txt']
        return old(request, timeout)
    opener.open = spy
    gate.acquire(root, code, rev, opener=opener)


@pytest.mark.parametrize('fault', ['hash', 'encoding', 'html', 'status', 'overflow', 'truncated', 'pointer', 'redirect', 'length'])
def test_opaque_weight_transport_faults_no_retry_no_secrets(gate, tmp_path, monkeypatch, fault):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    row = c['assets'][-1]; raw = b'OPAQUE TINY CHECKPOINT NEVER DESERIALIZE'; kw = {}; url = row['url']
    if fault == 'hash': raw = b'X'+raw[1:]
    elif fault == 'encoding': kw['encoding'] = 'gzip'
    elif fault == 'html': raw = b'<html>'; kw['length'] = row['bytes']
    elif fault == 'status': kw['status'] = 403
    elif fault == 'overflow': raw += b'X'; kw['length'] = row['bytes']
    elif fault == 'truncated': raw = raw[:-1]; kw['length'] = row['bytes']
    elif fault == 'pointer': raw = b'version https://git-lfs.github.com/spec/v1'; kw['length'] = row['bytes']
    elif fault == 'redirect': url = 'https://evil.invalid/?token=NEVER_LOG'
    else: kw['length'] = row['bytes']+1
    opener.rows[row['url']] = Response(raw, url, **kw)
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert value['status'] == 'fail' and value['source_rehashed_after'] and value['owned_partials_removed'] and value['owned_archive_removed']
    assert len(opener.calls) == 6 and not (data/'weights/epoch_5.pth').exists() and not list(data.rglob('*.part'))
    assert 'NEVER_LOG' not in (root/c['report']).read_text()
    with pytest.raises(ValueError, match='no resume'):
        gate.acquire(root, code, rev, opener=opener)


@pytest.mark.parametrize('fault', ['traversal', 'absolute', 'hardlink', 'symlink', 'link_text', 'missing_link', 'duplicate', 'wrong_blob', 'excluded_blob', 'missing', 'foreign', 'foreign_directory', 'executable', 'special', 'collision'])
def test_whole_tree_archive_faults_abort_before_notices_and_model(gate, tmp_path, monkeypatch, fault):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    source = c['source']; files = dict(files); links = dict(links); extra = None
    if fault == 'wrong_blob': files['demo/helpers.py'] = b'X'+files['demo/helpers.py'][1:]
    elif fault == 'excluded_blob': files['assets/example.jpg'] += b'X'
    elif fault == 'missing': del files['demo/helpers.py']
    elif fault == 'link_text': links['mmdet/.mim/demo'] = '/etc/passwd'
    elif fault == 'missing_link': del links['mmdet/.mim/demo']
    elif fault == 'collision': files['mmdet'] = b'file colliding with directory'
    else:
        name = '../escape' if fault == 'traversal' else '/etc/passwd' if fault == 'absolute' else source['archive_prefix']+('/LICENSE.txt' if fault == 'duplicate' else '/foreign')
        n = tarfile.TarInfo(name); n.size = 1; raw = b'X'
        if fault in ('hardlink', 'symlink'):
            n.type = tarfile.LNKTYPE if fault == 'hardlink' else tarfile.SYMTYPE; n.linkname = '/etc/passwd'; n.size = 0
        elif fault == 'foreign_directory': n.type = tarfile.DIRTYPE; n.size = 0
        elif fault == 'special': n.type = tarfile.CHRTYPE; n.size = 0
        elif fault == 'executable': n.name = source['archive_prefix']+'/unexpected.py'; n.mode = 0o755
        extra = (n, raw)
    opener.rows[source['url']] = Response(pack(source['archive_prefix'], files, links, extra), source['url'])
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert value['status'] == 'fail' and value['source_rehashed_after'] and value['owned_archive_removed'] and len(opener.calls) == 1
    assert not (data/'weights').exists() and not (tmp_path/'escape').exists() and not list(data.rglob('*.part'))


@pytest.mark.parametrize('fault', ['source_license', 'card', 'apache_notice', 'co_detr_notice'])
def test_exact_declaration_failure_stops_before_any_model(gate, tmp_path, monkeypatch, fault):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    if fault == 'source_license':
        files = dict(files); files['LICENSE.txt'] = b'No grant here\n'
        s = c['source']; rows = {n: blob(b) for n, b in files.items()} | {n: blob(b.encode(), '120000') for n, b in links.items()}
        s['git_root_tree_sha1'] = gate.git_tree(rows); s['expanded_bytes'] = sum(r['bytes'] for r in rows.values())
        s['required_source_pins']['LICENSE.txt'] = dict(bytes=len(files['LICENSE.txt']), sha256=hashlib.sha256(files['LICENSE.txt']).hexdigest())
        opener.rows[s['url']] = Response(pack(s['archive_prefix'], files, links), s['url'])
    else:
        index = 3 if fault == 'card' else 0 if fault == 'apache_notice' else 2
        row = c['assets'][index]; raw = b'No grant here\n'; row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        opener.rows[row['url']] = Response(raw, row['url'])
    repin(gate, code, c)
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert not value['first_party_license_declarations_verified'] and len(opener.calls) == 5 and not (data/'weights/epoch_5.pth').exists()


def test_independent_helper_mutation_before_any_import_or_transfer(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    p = code/next(iter(gate.HELPER_PINS)); p.chmod(0o644); p.write_bytes(b'raise RuntimeError("MUST NOT IMPORT")\n'); p.chmod(0o444)
    with pytest.raises(ValueError, match='pinned helper'):
        gate.acquire(root, code, rev, opener=opener)
    assert not opener.calls and not data.exists()


def test_original_entry_and_low_space_preflight_zero_transfer(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    (code.parent/'unexpected').write_bytes(b'X')
    with pytest.raises(ValueError, match='snapshot parent'):
        gate.acquire(root, code, rev, opener=opener)
    (code.parent/'unexpected').unlink()
    monkeypatch.setattr(gate.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(1, 0, 1))
    with pytest.raises(ValueError, match='16GiB'):
        gate.acquire(root, code, rev, opener=opener)
    assert not opener.calls and not data.exists()


def test_late_source_mutation_not_a_pass(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = opener.open
    def mutate(request, timeout):
        result = old(request, timeout)
        if len(opener.calls) == 6:
            p = code/gate.HELPERS[0]; p.chmod(0o644); p.write_bytes(p.read_bytes()+b'\n'); p.chmod(0o444)
        return result
    opener.open = mutate
    with pytest.raises(ValueError):
        gate.acquire(root, code, rev, opener=opener)
    assert report(root)['source_rehashed_after'] is False


def test_owned_partial_write_failure_cleaned_without_removing_source(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = gate.publish_bytes
    def fail(mp, data, target, raw, owned, directories):
        if target.name == 'helpers.py':
            gate.parents(data, target, directories); part = target.with_name(target.name+'.part'); part.write_bytes(b'tiny interrupted'); s = part.lstat(); owned.append((part, (s.st_dev, s.st_ino, s.st_uid)))
            raise TimeoutError('NEVER_PRINT_URL_OR_SECRET')
        return old(mp, data, target, raw, owned, directories)
    monkeypatch.setattr(gate, 'publish_bytes', fail)
    with pytest.raises(ValueError):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert value['owned_partials_removed'] and value['owned_archive_removed'] and value['source_rehashed_after'] and len(opener.calls) == 1
    assert not list(data.rglob('*.part')) and 'NEVER_PRINT' not in (root/c['report']).read_text()


def test_posthash_inclusive_deadline_no_late_pass(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = gate.binding; calls = []
    def slow(*args):
        value = old(*args); calls.append(1)
        if len(calls) == 2:
            monkeypatch.setattr(gate.time, 'monotonic', lambda: 10**12)
        return value
    monkeypatch.setattr(gate, 'binding', slow)
    with pytest.raises(ValueError):
        gate.acquire(root, code, rev, opener=opener)
    assert report(root)['elapsed_seconds'] > 900 and report(root)['status'] == 'fail'


def test_git_tree_matches_git_native_ordering_modes(gate, tmp_path):
    files = {'root.py': b'tiny\n', 'a.txt': b'outside\n', 'a/z.py': b'inside\n', 'a-': b'order\n', 'z/link': b'../root.py'}
    rows = {n: blob(b, '120000' if n == 'z/link' else '100644') for n, b in files.items()}
    # Native git mktree independently checks tree serialization and directory
    # sort placement (a.txt/a-/a/), not merely our own reconstruction twice.
    subprocess.run(['git', 'init', '-q', str(tmp_path/'git')], check=True)
    def tree(entries):
        text = ''.join(f'{mode} {kind} {oid}\t{name}\n' for mode, kind, oid, name in entries)
        p = subprocess.run(['git', '-C', str(tmp_path/'git'), 'mktree', '--missing'], input=text.encode(), stdout=subprocess.PIPE, check=True)
        return p.stdout.decode().strip()
    a = tree([('100644', 'blob', rows['a/z.py']['git_blob_sha1'], 'z.py')])
    z = tree([('120000', 'blob', rows['z/link']['git_blob_sha1'], 'link')])
    entries = [('040000', 'tree', a, 'a'), ('040000', 'tree', z, 'z')]+[('100644', 'blob', rows[n]['git_blob_sha1'], n) for n in ('root.py', 'a.txt', 'a-')]
    assert gate.git_tree(rows) == tree(entries)
    assert gate.git_tree({}) == '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    with pytest.raises(ValueError, match='collision'):
        gate.git_tree({'a': blob(b'file'), 'a/x': blob(b'child')})


@pytest.mark.parametrize('name', ['assets/a.png', 'figures/a.jpg', 'demo/a.mp4', 'demo/a.json', 'projects/a.npy', 'weights/a.pth', 'mmdet/.mim/configs', 'mmdet/__pycache__/a.pyc', 'resources/a.txt', 'tests/a.py'])
def test_non_runtime_media_data_cache_never_retained(gate, name):
    assert not gate.retained(name)


@pytest.mark.parametrize('name', ['assets/thirdparty/LICENSE', 'docs/NOTICE.txt', 'resources/COPYING.md', 'mmdet/a.py', 'projects/a.py', 'configs/a.py', 'demo/predictions_io.py', 'tools/a.py', 'docker/Dockerfile.hopper', 'docker/build.sh', 'requirements/runtime.txt', 'setup.py', 'INSTALL_HOPPER.md'])
def test_runtime_and_original_notices_retained(gate, name):
    assert gate.retained(name)


def test_real_config_exact_pins_and_no_runtime_claim(gate):
    raw = (REPO/gate.PROTOCOL).read_bytes(); c = json.loads(raw)
    assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == gate.PROTOCOL_PIN
    assert c['source']['revision'] == '1b367292f3833afd64a204bd4d9d84519541d035'
    assert c['source']['git_root_tree_sha1'] == 'e93a418345712fc6c011bfcc84bdddc90f496a70'
    assert (c['source']['blob_count'], c['source']['expanded_bytes']) == (2639, 32009009)
    assert c['assets'][-1]['bytes'] == 5855053598 and c['assets'][-1]['sha256'] == '4708fd0ddc5c3d386bad67c31152de58676840b911f6d01219e0092a603277d3'
    assert c['checkpoint']['revision'] == '2d737cbf189a1b61c2e8e86cfd03df5acba56f52'
    assert c['assets'][-2]['sha256'] == 'd8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17'
    assert all(v is False for k, v in c['scope'].items() if k != 'training_overlap_status')
    assert c['future_runtime']['torch_2_5_1_cu124_h100_qualified'] is False
    for name, pin in gate.HELPER_PINS.items():
        b = (REPO/name).read_bytes(); assert dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) == pin
    ast.parse((REPO/gate.HELPERS[0]).read_text())
    source = (REPO/gate.HELPERS[0]).read_text()
    assert 'extractall' not in source and 'torch.load' not in source and 'pickle' not in source
    subprocess.run(['bash', '-n', str(REPO/gate.HELPERS[1])], check=True)
    wrapper = (REPO/gate.HELPERS[1]).read_text()
    assert 'env -i' in wrapper and '960s' in wrapper and 'docker' not in wrapper


@pytest.mark.parametrize('url', ['http://huggingface.co/x', 'https://user:pass@huggingface.co/x', 'https://hf.co.evil/x', 'https://evil.invalid/x', 'https://huggingface.co/x#fragment'])
def test_closed_public_endpoint(gate, url):
    with pytest.raises(ValueError):
        gate.endpoint(url)


def test_redirects_weight_cdn_only(gate):
    original = 'https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/revision/epoch_5.pth'
    gate.transport_destination(original, 'https://cas-bridge.xethub.hf.co/object?signature=NOT_LOGGED')
    for original, final in [('https://raw.githubusercontent.com/x/LICENSE', 'https://raw.githubusercontent.com/y/LICENSE'),
                            ('https://codeload.github.com/x/tar.gz/rev', 'https://codeload.github.com/y/tar.gz/rev'),
                            ('https://huggingface.co/x/raw/rev/README.md', 'https://cdn-lfs.hf.co/card'),
                            ('https://cdn-lfs.hf.co/weight', 'https://evil.invalid/weight')]:
        with pytest.raises(ValueError):
            gate.transport_destination(original, final)


def test_publisher_redirect_chain_keeps_original_weight_scope(gate):
    handler = gate.PublicRedirect()
    original = gate.urllib.request.Request('https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/rev/epoch_5.pth', headers={'Accept-Encoding': 'identity'})
    one = handler.redirect_request(original, None, 302, '', {}, 'https://cdn-lfs.hf.co/weight')
    two = handler.redirect_request(one, None, 302, '', {}, 'https://cas-bridge.xethub.hf.co/object')
    assert two._wr_public_origin == original.full_url
    with pytest.raises(ValueError):
        handler.redirect_request(two, None, 302, '', {}, 'https://evil.invalid/object')
    original.add_header('Authorization', 'SECRET_NEVER_OUTPUT')
    with pytest.raises(ValueError, match='Authenticated'):
        handler.redirect_request(original, None, 302, '', {}, 'https://cdn-lfs.hf.co/weight')


@pytest.mark.parametrize('fault', ['member_cap', 'retained_cap', 'blob_cap', 'tar_cap', 'expanded_cap', 'non_utf8', 'nul'])
def test_resource_or_source_text_fail_fast_before_any_weight(gate, tmp_path, monkeypatch, fault):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    source = c['source']; files = dict(files)
    if fault in ('non_utf8', 'nul'):
        files['projects/code.py'] = b'\xff' if fault == 'non_utf8' else b'\0'
        opener.rows[source['url']] = Response(pack(source['archive_prefix'], files, links), source['url'])
    elif fault == 'member_cap': source['maximum_member_bytes'] = 1
    elif fault == 'retained_cap': source['maximum_retained_file_bytes'] = 1
    elif fault == 'blob_cap': source['maximum_blobs'] = 1
    elif fault == 'tar_cap': source['maximum_tar_entries'] = 1
    else: source['maximum_expanded_bytes'] = 1
    # Invoke the genuine extractor with a manufactured resource cap. Production
    # manifest constants cannot be weakened to reach this test branch.
    rt, mp = gate.helpers(code); data.mkdir(); directories = set(); owned = []; artifacts = []
    archive = data/'tiny.tar.gz'; archive.write_bytes(pack(source['archive_prefix'], files, links))
    with pytest.raises((ValueError, UnicodeDecodeError)):
        gate.archive_inventory(rt, mp, data, archive, source, gate.time.monotonic()+30, owned, directories, artifacts)
    assert not opener.calls and not (data/'weights').exists()


def test_native_visibility_of_corrupted_published_weight_cannot_pass(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = gate.download
    def corrupt(*args, **kwargs):
        result = old(*args, **kwargs)
        if result['file'].endswith('.pth'):
            p = data/result['file']; p.chmod(0o644); raw = p.read_bytes(); p.write_bytes(b'X'+raw[1:]); p.chmod(0o444)
        return result
    monkeypatch.setattr(gate, 'download', corrupt)
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert value['artifacts_rehashed_after'] is False and value['status'] == 'fail'


def test_foreign_namespace_is_not_deleted_or_certified(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    old = opener.open
    def foreign(request, timeout):
        value = old(request, timeout)
        if len(opener.calls) == 6:
            (data/'foreign').write_bytes(b'NOT OUR ARTIFACT')
        return value
    opener.open = foreign
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener)
    value = report(root)
    assert value['status'] == 'fail' and value['artifacts_rehashed_after'] is False
    assert (data/'foreign').read_bytes() == b'NOT OUR ARTIFACT'


def test_byte_mutated_but_same_length_helper_rejected_before_import(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    p = code/next(iter(gate.HELPER_PINS)); p.chmod(0o644); raw = p.read_bytes(); p.write_bytes(b'!'+raw[1:]); p.chmod(0o444)
    with pytest.raises(ValueError, match='Independent helper bytes'):
        gate.acquire(root, code, rev, opener=opener)
    assert not opener.calls and not data.exists()


def test_wrong_lease_does_not_use_existing_unowned_namespace(gate, tmp_path, monkeypatch):
    root, data, code, rev, c, files, links, opener = setup(gate, tmp_path, monkeypatch)
    data.mkdir(); (data/'foreign').write_bytes(b'PRESERVE')
    with pytest.raises(ValueError, match='immutable receipt'):
        gate.acquire(root, code, rev, opener=opener, namespace_lease={})
    assert not opener.calls and (data/'foreign').read_bytes() == b'PRESERVE' and report(root)['status'] == 'fail'


def test_receipt_fsync_crossing_deadline_is_resealed_fail(gate, tmp_path, monkeypatch):
    path = tmp_path/'report.json'; started = gate.time.monotonic(); deadline = started+900
    value = dict(status='pass', phase='complete', elapsed_seconds=.1)
    real = gate.os.fsync; calls = []
    def late(fd):
        real(fd); calls.append(1)
        if len(calls) == 1:
            monkeypatch.setattr(gate.time, 'monotonic', lambda: deadline+1)
    monkeypatch.setattr(gate.os, 'fsync', late)
    result = gate.write_receipt(path, value, started, deadline)
    saved = json.loads(path.read_bytes())
    assert result['status'] == saved['status'] == 'fail' and saved['phase'] == 'receipt_sealing'
    assert saved['elapsed_seconds'] == 901 and len(calls) == 2 and path.stat().st_mode & 0o777 == 0o444


def test_receipt_write_failure_demotes_using_owned_descriptor(gate, tmp_path, monkeypatch):
    path = tmp_path/'report.json'; started = gate.time.monotonic(); real = gate.os.fsync; calls = []
    def one_failure(fd):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('NEVER_OUTPUT_SECRET')
        real(fd)
    monkeypatch.setattr(gate.os, 'fsync', one_failure)
    value = gate.write_receipt(path, dict(status='pass', phase='complete'), started, started+900)
    assert value['status'] == 'fail' and json.loads(path.read_bytes())['status'] == 'fail'
    assert 'NEVER_OUTPUT_SECRET' not in path.read_text() and path.stat().st_mode & 0o777 == 0o444

"""Tiny fake publisher streams only: no Azure, network, media or real archives."""
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'infra'))
    spec = importlib.util.spec_from_file_location('dexycb_download_test', ROOT / 'infra/dexycb_download.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Response(io.BytesIO):
    def __init__(self, data, *, mime='application/gzip', length=None, status=200,
                 url='https://drive.usercontent.google.com/download?id=x', encoding=None):
        super().__init__(data); self.status = status; self.url = url; self.reads = []
        self.headers = email.message.Message(); self.headers['Content-Type'] = mime
        if length is not None: self.headers['Content-Length'] = str(length)
        if encoding: self.headers['Content-Encoding'] = encoding

    def geturl(self): return self.url

    def read(self, count=-1):
        self.reads.append(count); assert 0 <= count <= 1 << 20
        return super().read(count)


class Opener:
    def __init__(self, responses, barrier=None): self.responses = responses; self.calls = []; self.barrier = barrier

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.barrier: self.barrier.wait(timeout=2)
        file_id = request.full_url.split('id=')[1].split('&')[0]
        return self.responses[file_id]


def setup(gate, tmp_path, monkeypatch, *, responses=None):
    monkeypatch.delenv('WR_CODE_REVISION', raising=False)
    root = tmp_path / 'root'; (root / 'results').mkdir(parents=True)
    incoming = tmp_path / 'incoming'; monkeypatch.setattr(gate, 'INCOMING', incoming)
    protocol = copy.deepcopy(gate.dex.EXPECTED_PROTOCOL); data = b'\x1f\x8b' + b'tiny-opaque-original-bytes' * 3
    streams = {}
    for row in protocol['archives'].values():
        row['bytes'] = len(data)
        file_id = row['url'].rsplit('/', 1)[1]
        streams[file_id] = Response(data, length=len(data))
    if responses:
        for index, replacement in responses.items(): streams[list(streams)[index]] = replacement
    monkeypatch.setattr(gate.dex, 'EXPECTED_PROTOCOL', protocol)
    config = root / 'protocol.json'; config.write_text(json.dumps(protocol)); config.chmod(0o444)
    for module in (gate, gate.dex):
        source = tmp_path / (module.__name__ + '.py'); source.write_bytes(Path(module.__file__).read_bytes())
        source.chmod(0o444); monkeypatch.setattr(module, '__file__', str(source))
    return root, incoming, config, protocol, data, Opener(streams)


@pytest.fixture
def atomic(gate, monkeypatch):
    # Linux production uses renameat2; local macOS tiny tests emulate NOREPLACE.
    def publish(part, target):
        os.link(part, target); part.unlink()
    monkeypatch.setattr(gate, 'publish', publish)


def receipt(gate, root): return json.loads((root / gate.RESULT / 'report.json').read_text())


def test_exact_original_two_parallel_streams_and_measured_not_publisher_pins(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, protocol, data, opener = setup(gate, tmp_path, monkeypatch)
    opener.barrier = threading.Barrier(2)
    report = gate.download(root, config, opener=opener)
    assert report['status'] == 'pass' and report['phase'] == 'complete' and len(opener.calls) == 2
    assert report['publisher_checksum_verified'] is False and report['source_rehashed_after'] is True
    assert not report['gpu_used'] and not report['models_loaded'] and not report['annotation_values_parsed']
    for row in report['archives']:
        path = incoming / row['file']; assert path.read_bytes() == data and path.stat().st_mode & 0o777 == 0o444
        assert row['sha256'] == hashlib.sha256(data).hexdigest() and row['bytes'] == len(data)
        assert row['hash_basis'] == 'caller_measured_not_publisher_checksum'
    assert incoming.stat().st_mode & 0o777 == 0o700
    assert (root / gate.RESULT / 'report.json').stat().st_mode & 0o777 == 0o444
    assert set(p.name for p in incoming.iterdir()) == {r['file'] for r in protocol['archives'].values()}
    for request, timeout in opener.calls:
        assert request.full_url.startswith('https://drive.usercontent.google.com/download?id=')
        assert request.full_url.endswith('&export=download&confirm=t') and timeout == 30
        assert dict(request.header_items()) == {'Accept-encoding': 'identity'}
    encoded = json.dumps(report); assert len(encoded) < 4096 and 'cookie' not in encoded and 'http' not in encoded


@pytest.mark.parametrize('fault', ['html', 'text', 'mime', 'status', 'redirect', 'length', 'encoding', 'body', 'overflow', 'truncated'])
def test_invalid_provider_response_fails_sealed_without_retries_or_partials(gate, tmp_path, monkeypatch, atomic, fault):
    data = b'\x1f\x8b' + b'tiny-opaque-original-bytes' * 3
    opts = {}; payload = data
    if fault == 'html': opts['mime'] = 'text/html'
    elif fault == 'text': opts['mime'] = 'text/plain'
    elif fault == 'mime': opts['mime'] = 'image/png'
    elif fault == 'status': opts['status'] = 429
    elif fault == 'redirect': opts['url'] = 'https://accounts.google.com/login?TOKEN=must-not-log'
    elif fault == 'length': opts['length'] = len(data) + 1
    elif fault == 'encoding': opts['encoding'] = 'gzip'
    elif fault == 'body': payload = b'<!DOCTYPE html>no download'
    elif fault == 'overflow': payload += b'EXTRA'
    else: payload = data[:-1]
    root, incoming, config, _, _, opener = setup(gate, tmp_path, monkeypatch,
                                               responses={0: Response(payload, **opts)})
    with pytest.raises(gate.DownloadError, match='sealed receipt'): gate.download(root, config, opener=opener)
    report = receipt(gate, root)
    assert report['status'] == 'fail' and report['source_rehashed_after'] and report['owned_partials_removed']
    assert 1 <= len(opener.calls) <= 2 and not list(incoming.glob('*.part'))
    assert 'TOKEN' not in json.dumps(report) and 'accounts.google' not in json.dumps(report)


@pytest.mark.parametrize('url', ['http://drive.google.com/x', 'https://evil.example/x',
    'https://accounts.google.com/x', 'https://drive.google.com:444/x', 'https://u:p@drive.google.com/x',
    'https://drive.usercontent.google.com/x#fragment'])
def test_redirect_guard_rejects_foreign_or_auth_provider(gate, url):
    with pytest.raises(gate.DownloadError): gate.safe_url(url)
    handler = gate.PublicRedirect(); request = urllib.request.Request('https://drive.google.com/x')
    with pytest.raises(gate.DownloadError): handler.redirect_request(request, None, 302, 'redirect', {}, url)


def test_frozen_protocol_and_file_ids_cannot_be_edited(gate, tmp_path, monkeypatch):
    root, _, config, protocol, _, opener = setup(gate, tmp_path, monkeypatch)
    changed = copy.deepcopy(protocol); changed['archives'][next(iter(changed['archives']))]['url'] = 'https://evil.example/abc'
    config.chmod(0o644); config.write_text(json.dumps(changed)); config.chmod(0o444)
    with pytest.raises(ValueError): gate.download(root, config, opener=opener)
    assert opener.calls == [] and not (root / gate.RESULT).exists()
    with pytest.raises(gate.DownloadError): gate.download_url({'url': 'https://drive.google.com/file/d/id?different=1'})


@pytest.mark.parametrize('occupied', ['incoming', 'result', 'symlink'])
def test_no_resume_overwrite_or_unknown_namespace_cleanup(gate, tmp_path, monkeypatch, occupied):
    root, incoming, config, _, _, opener = setup(gate, tmp_path, monkeypatch)
    if occupied == 'incoming': incoming.mkdir(); (incoming / 'existing').write_bytes(b'KEEP')
    elif occupied == 'result': (root / gate.RESULT).mkdir()
    else: incoming.symlink_to(root, target_is_directory=True)
    with pytest.raises((gate.DownloadError, ValueError)): gate.download(root, config, opener=opener)
    assert opener.calls == []
    if occupied == 'incoming': assert (incoming / 'existing').read_bytes() == b'KEEP'


def test_midread_failure_preserves_completed_archive_and_removes_only_own_partial(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, protocol, data, opener = setup(gate, tmp_path, monkeypatch)
    first = threading.Event(); normal = opener.responses[next(iter(opener.responses))]
    original_publish = gate.publish
    def publish(part, target): original_publish(part, target); first.set()
    monkeypatch.setattr(gate, 'publish', publish)
    class Broken(Response):
        def read(self, count=-1):
            first.wait(timeout=2)
            if self.tell(): raise OSError('SECRET_URL_should_not_be_logged')
            return super().read(2)
    opener.responses[list(opener.responses)[1]] = Broken(data)
    with pytest.raises(gate.DownloadError): gate.download(root, config, opener=opener)
    report = receipt(gate, root)
    assert len(report['archives']) == 1 and normal.closed
    assert (incoming / protocol['archives'][next(iter(protocol['archives']))]['file']).read_bytes() == data
    assert not list(incoming.glob('*.part')) and 'SECRET' not in json.dumps(report)


def test_bounded_budget_before_any_network_and_source_mutation_after_stream(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, _, _, opener = setup(gate, tmp_path, monkeypatch)
    monkeypatch.setattr(gate, 'BUDGET', 0)
    with pytest.raises(gate.DownloadError): gate.download(root, config, opener=opener)
    assert opener.calls == [] and receipt(gate, root)['status'] == 'fail'


def test_source_post_failure_preserves_finished_archives_but_never_pass(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, _, _, opener = setup(gate, tmp_path, monkeypatch)
    publish = gate.publish
    def mutate(part, target):
        publish(part, target); config.chmod(0o644); config.write_text('{}'); config.chmod(0o444)
    monkeypatch.setattr(gate, 'publish', mutate)
    with pytest.raises(gate.DownloadError): gate.download(root, config, opener=opener)
    assert receipt(gate, root)['status'] == 'fail' and not receipt(gate, root)['source_rehashed_after']
    assert len(list(incoming.glob('*.tar.gz'))) == 2


def test_fragmented_gzip_header_is_streamed_byte_exact_and_progress_bounded(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, _, data, opener = setup(gate, tmp_path, monkeypatch)
    class Fragmented(Response):
        def read(self, count=-1): return super().read(min(count, 1))
    for file_id in opener.responses: opener.responses[file_id] = Fragmented(data)
    report = gate.download(root, config, opener=opener)
    assert all(row['bytes_received'] == row['expected_bytes'] == len(data) for row in report['download_progress'])
    assert all(p.read_bytes() == data for p in incoming.iterdir())


def test_publication_collision_keeps_foreign_target_and_seals_failure(gate, tmp_path, monkeypatch, atomic):
    root, incoming, config, _, _, opener = setup(gate, tmp_path, monkeypatch)
    publish = gate.publish
    def occupied(part, target):
        target.write_bytes(b'FOREIGN_KEEP'); publish(part, target)
    monkeypatch.setattr(gate, 'publish', occupied)
    with pytest.raises(gate.DownloadError): gate.download(root, config, opener=opener)
    assert receipt(gate, root)['status'] == 'fail' and not list(incoming.glob('*.part'))
    assert all(p.read_bytes() == b'FOREIGN_KEEP' for p in incoming.iterdir())


@pytest.mark.skipif(os.uname().sysname != 'Linux', reason='Linux renameat2 contract')
def test_real_atomic_publication_does_not_overwrite(gate, tmp_path):
    part, target = tmp_path / 'part', tmp_path / 'target'; part.write_bytes(b'NEW'); target.write_bytes(b'OLD')
    with pytest.raises(OSError): gate.publish(part, target)
    assert target.read_bytes() == b'OLD' and part.read_bytes() == b'NEW'


def test_fixed_runtime_limits_and_public_source_protocol(gate):
    gate.dex.exact(json.loads((ROOT / 'configs/dexycb_identity_protocol.json').read_text()), gate.dex.EXPECTED_PROTOCOL)
    assert gate.BUDGET == 7200 and gate.BLOCK == 1 << 20 and gate.TIMEOUT == 30
    assert str(gate.INCOMING) == '/srv/world-reward-data/dexycb_identity_download_v1'
    assert 'gdown' not in Path(gate.__file__).read_text() and 'CookieJar' not in Path(gate.__file__).read_text()


def dispatch_fixture(gate, tmp_path, monkeypatch):
    root, incoming, config, protocol, data, opener = setup(gate, tmp_path, monkeypatch)
    revision = 'a' * 40; code = root / 'jobs' / revision / 'run_dexycb_download' / 'code'
    for relative, raw in [('infra/dexycb_download.py', Path(gate.__file__).read_bytes()),
                          ('infra/dexycb_acquire.py', Path(gate.dex.__file__).read_bytes()),
                          ('configs/dexycb_identity_protocol.json', config.read_bytes()),
                          ('infra/run_dexycb_download.sh', b'actual immutable wrapper'),
                          ('src/__init__.py', b'')]:
        path = code / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o444)
    for name, raw in [('revision', revision + '\n'), ('source-sha256', 'b' * 64 + '\n')]:
        path = code.parent / name; path.write_text(raw); path.chmod(0o444)
    for path in (code, *code.rglob('*')):
        if path.is_dir(): path.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', root); monkeypatch.setattr(gate, '__file__', str(code / 'infra/dexycb_download.py'))
    monkeypatch.setattr(gate.dex, '__file__', str(code / 'infra/dexycb_acquire.py'))
    monkeypatch.setenv('WR_CODE_REVISION', revision)
    return root, incoming, code / 'configs/dexycb_identity_protocol.json', code, opener


def test_complete_dispatched_readonly_closure_and_markers_bound_prepost(gate, tmp_path, monkeypatch, atomic):
    root, _, config, code, opener = dispatch_fixture(gate, tmp_path, monkeypatch)
    report = gate.download(root, config, opener=opener)
    dispatch = report['source_helpers']['dispatch']
    assert dispatch['producer_revision'] == 'a' * 40 and dispatch['entries'] == 9
    assert set(dispatch['markers']) == {'revision', 'source-sha256'} and len(dispatch['closure_sha256']) == 64
    assert report['source_rehashed_after'] and 'src/__init__.py' not in json.dumps(report)


@pytest.mark.parametrize('fault', ['revision', 'archive_marker', 'namespace', 'writable', 'symlink'])
def test_dispatched_source_preflight_rejects_marker_path_and_closure_faults(gate, tmp_path, monkeypatch, atomic, fault):
    root, _, config, code, opener = dispatch_fixture(gate, tmp_path, monkeypatch)
    if fault in ('revision', 'archive_marker'):
        path = code.parent / ('revision' if fault == 'revision' else 'source-sha256')
        path.chmod(0o644); path.write_bytes(b'WRONG\n'); path.chmod(0o444)
    elif fault == 'namespace': monkeypatch.setenv('WR_CODE_REVISION', 'c' * 40)
    elif fault == 'writable': (code / 'infra/run_dexycb_download.sh').chmod(0o644)
    else:
        (code / 'src').chmod(0o755); (code / 'src/foreign').symlink_to(config); (code / 'src').chmod(0o555)
    with pytest.raises((ValueError, FileNotFoundError)): gate.download(root, config, opener=opener)
    assert opener.calls == [] and not (root / gate.RESULT).exists()


@pytest.mark.parametrize('fault', ['extra_file', 'wrapper', 'marker'])
def test_dispatched_source_post_hash_never_ignores_extra_or_modified_source(gate, tmp_path, monkeypatch, atomic, fault):
    root, incoming, config, code, opener = dispatch_fixture(gate, tmp_path, monkeypatch)
    publish = gate.publish
    def mutate(part, target):
        publish(part, target)
        path = code / 'extra.py' if fault == 'extra_file' else (code / 'infra/run_dexycb_download.sh'
                 if fault == 'wrapper' else code.parent / 'source-sha256')
        path.parent.chmod(0o755)
        if path.exists(): path.chmod(0o644)
        path.write_bytes(b'changed immutable source'); path.chmod(0o444)
        if path.parent == code: path.parent.chmod(0o555)
    monkeypatch.setattr(gate, 'publish', mutate)
    with pytest.raises(gate.DownloadError): gate.download(root, config, opener=opener)
    report = receipt(gate, root)
    assert report['status'] == 'fail' and not report['source_rehashed_after']
    assert len(list(incoming.glob('*.tar.gz'))) == 2

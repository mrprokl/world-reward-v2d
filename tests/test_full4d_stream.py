"""Manufactured loopback HTTP ranges only; no Azure call or video/render files."""
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import full4d_stream as stream


def receipt():
    revision = 'a'*40
    episodes = [3, 12, 28]
    rows = []
    for ep in episodes:
        for extension, mime in [('mp4', 'video/mp4'), ('jpg', 'image/jpeg')]:
            rows.append(dict(name=f'full4d-{revision}/episode_{ep:06d}.{extension}',
                mime=mime, bytes=20, sha256='b'*64, etag='"0xABC123"', episode_index=ep))
    rows.append(dict(name=f'full4d-{revision}/manifest.json', mime='application/json',
        bytes=20, sha256='b'*64, etag='"0xDEF456"'))
    return dict(schema='world_reward.full4d_publish.v1', status='pass', producer_revision=revision,
        endpoint=stream.ENDPOINT, private_container_verified=True, public_access_changed=False,
        account_keys_used=False, sas_tokens_persisted=False, episodes=episodes, files=rows), revision


@pytest.mark.parametrize('header,expected', [(None, None), ('bytes=0-4', (0, 4)),
    ('bytes=10-', (10, 19)), ('bytes=-5', (15, 19)), ('bytes=0-999', (0, 19))])
def test_native_single_byte_ranges(header, expected):
    assert stream.byte_range(header, 20) == expected


@pytest.mark.parametrize('header', ['bytes=0-1,3-4', 'bytes=-0', 'bytes=30-',
    'bytes=8-2', 'bytes=-', 'bytes=abc-2', 'other=0-2', 'bytes= -2'])
def test_bad_ranges_are_rejected_without_any_remote_read(header):
    with pytest.raises(ValueError): stream.byte_range(header, 20)


def test_only_exact_explicit_private_publication_media_allowed():
    value, revision = receipt()
    episodes, allowed = stream.records(value, revision)
    assert episodes == [3, 12, 28]
    assert set(allowed) == {f'/episode-{ep}.{ext}' for ep in episodes for ext in ('mp4', 'jpg')}
    value['files'][0]['name'] += '?sig=secret'
    with pytest.raises(ValueError): stream.records(value, revision)


def test_tiny_receipt_read_does_not_read_video_files(tmp_path):
    value, revision = receipt()
    path = tmp_path/'published.json'; path.write_text(json.dumps(value))
    episodes, allowed, digest = stream.load_receipt(path, revision)
    assert len(allowed) == 6 and len(digest) == 64
    assert list(tmp_path.iterdir()) == [path]


def test_bearer_only_ram_subprocess_no_log_settings(monkeypatch, capsys):
    invoked = []
    def run(command, **kwargs):
        invoked.append((command, kwargs))
        class Result:
            returncode = 0
            stdout = json.dumps(dict(accessToken='manufactured-secret', expires_on=str(int(time.time())+3600))).encode()
        return Result()
    monkeypatch.setattr(stream.subprocess, 'run', run)
    reader = stream.AzureReader()
    assert reader.authorization() == {'Authorization': 'Bearer manufactured-secret'}
    assert reader.authorization() == {'Authorization': 'Bearer manufactured-secret'}
    assert len(invoked) == 1
    command, kwargs = invoked[0]
    assert command[:3] == ['rtk', 'proxy', 'az']
    assert kwargs['capture_output'] is True and kwargs['env']['AZURE_LOGGING_ENABLE_LOG_FILE'] == 'false'
    assert kwargs['env']['AZURE_CORE_COLLECT_TELEMETRY'] == 'false'
    assert capsys.readouterr().out == ''


def test_authentication_errors_never_expose_token_or_az_error_output(monkeypatch):
    class Result:
        returncode = 1
        stdout = b'manufactured-secret'
        stderr = b'private-error-with-credential'
    monkeypatch.setattr(stream.subprocess, 'run', lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError) as error: stream.AzureReader().authorization()
    assert str(error.value) == 'Azure preview authentication unavailable'


def test_local_html_contains_no_azure_endpoint_secret_and_no_autoplay():
    value, revision = receipt(); episodes, _ = stream.records(value, revision)
    html = stream.index_html(episodes).decode()
    assert html.count('<video ') == 3 and html.count('preload="none"') == 3
    assert 'autoplay' not in html and 'storage.azure' not in html and 'blob.core' not in html
    assert 'sig=' not in html and 'Authorization' not in html


class Response(BytesIO):
    def __init__(self, value): super().__init__(value); self.read_sizes = []
    def read(self, n=-1): self.read_sizes.append(n); return super().read(n)


class Reader:
    def __init__(self): self.opens = []; self.heads = []; self.responses = []
    def verify_head(self, row): self.heads.append(row['name'])
    def open(self, row, selected):
        self.verify_head(row)
        self.opens.append((row['name'], selected))
        raw = b'0123456789abcdefghij'
        selected_raw = raw if selected is None else raw[selected[0]:selected[1]+1]
        response = Response(selected_raw); self.responses.append(response)
        return response, len(selected_raw)


def test_loopback_range_stream_no_video_files_and_no_credential_urls(tmp_path):
    value, revision = receipt(); episodes, allowed = stream.records(value, revision)
    reader = Reader(); server = stream.make_server(episodes, allowed, reader=reader)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/') as response:
            assert response.status == 200 and b'<video ' in response.read()
        request = urllib.request.Request(base+'/episode-3.mp4', headers={'Range': 'bytes=4-8'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206 and response.read() == b'45678'
            assert response.headers['Content-Range'] == 'bytes 4-8/20'
            assert response.headers['Cache-Control'] == 'no-store, private'
        assert len(reader.opens) == 1 and reader.responses[0].read_sizes == [5]
        for path in ('/episode-3.mp4?sig=secret', '/episode-99.mp4', '/../data/original.mp4'):
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(base+path)
        assert len(reader.opens) == 1
        assert list(tmp_path.iterdir()) == []
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_foreign_origin_and_post_are_refused_before_remote_read():
    value, revision = receipt(); episodes, allowed = stream.records(value, revision)
    reader = Reader(); server = stream.make_server(episodes, allowed, reader=reader)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for request in (urllib.request.Request(base+'/episode-3.mp4', headers={'Origin': 'https://foreign.example'}),
                urllib.request.Request(base+'/episode-3.mp4', method='POST', data=b'not allowed')):
            with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(request)
        assert reader.opens == [] and reader.heads == []
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)

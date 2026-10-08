"""Secret-free loopback playback of bounded private Azure4D previews.

No video is saved, predownloaded, or cached by this process. Each browser range
is forwarded in64KiB memory chunks to the pinned private blob. The Azure user
bearer stays in memory; the browser sees only a loopback URL, never a SAS/key.
Only an explicit tiny publication receipt and3or4 MP4/JPEG pairs are accepted.
Playback still consumes the compressed preview bytes across the tether.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


ENDPOINT = 'https://stworldrewardresearch26.blob.core.windows.net/qa-previews'
CHUNK, MAX_RECEIPT = 65536, 16384
MAX_VIDEO, MAX_POSTER, MAX_TTL = 2_000_000, 100_000, 3600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate receipt field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite receipt')))


def records(receipt, revision):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Exact publication revision required')
    require(type(receipt) is dict and receipt.get('schema') == 'world_reward.full4d_publish.v1'
        and receipt.get('status') == 'pass' and receipt.get('producer_revision') == revision
        and receipt.get('endpoint') == ENDPOINT and receipt.get('private_container_verified') is True
        and receipt.get('public_access_changed') is False
        and receipt.get('account_keys_used') is False and receipt.get('sas_tokens_persisted') is False,
        'Original private preview publication receipt required')
    episodes = receipt.get('episodes')
    require(type(episodes) is list and 3 <= len(episodes) <= 4 and len(set(episodes)) == len(episodes)
        and all(type(ep) is int and 0 <= ep < 30 for ep in episodes), 'Explicit3or4 original episodes required')
    rows = receipt.get('files')
    require(type(rows) is list and len(rows) == len(episodes)*2+1, 'Exact preview publication inventory required')
    by_name = {}
    for row in rows:
        require(type(row) is dict and type(row.get('name')) is str and row['name'] not in by_name,
            'Unique original publication blob names required')
        require(type(row.get('bytes')) is int and 0 < row['bytes'] <= MAX_VIDEO
            and type(row.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', row['sha256'])
            and type(row.get('etag')) is str and re.fullmatch(r'"[A-Za-z0-9-]{1,100}"', row['etag']),
            'Exact bounded byte/SHA/ETag binding required')
        by_name[row['name']] = row
    allowed = {}
    for episode in episodes:
        for extension, mime, limit in [('mp4', 'video/mp4', MAX_VIDEO), ('jpg', 'image/jpeg', MAX_POSTER)]:
            name = f'full4d-{revision}/episode_{episode:06d}.{extension}'
            require(name in by_name and by_name[name].get('mime') == mime
                and by_name[name]['bytes'] <= limit and by_name[name].get('episode_index') == episode,
                'Only exact bounded published MP4/JPEG pairs accepted')
            allowed[f'/episode-{episode}.{extension}'] = by_name.pop(name)
    manifest_name = f'full4d-{revision}/manifest.json'
    require(set(by_name) == {manifest_name} and by_name[manifest_name]['mime'] == 'application/json'
        and by_name[manifest_name]['bytes'] <= MAX_RECEIPT, 'Only original tiny publication manifest may remain')
    return episodes, allowed


def load_receipt(path, revision):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
        and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical tiny receipt required')
    info = path.lstat()
    require(path.is_file() and info.st_nlink == 1 and 0 < info.st_size <= MAX_RECEIPT,
        'Bounded ordinary publication receipt required; no media input files')
    raw = path.read_bytes()
    require(len(raw) == info.st_size, 'Tiny receipt changed while reading')
    receipt = strict(raw)
    episodes, allowed = records(receipt, revision)
    return episodes, allowed, hashlib.sha256(raw).hexdigest()


def byte_range(value, size):
    """One validated byte range; never multiple/negative/out-of-bounds ranges."""
    require(type(size) is int and size > 0, 'Bounded resource size required')
    if value is None:
        return None
    require(type(value) is str and len(value) <= 80, 'One bounded byte range required')
    match = re.fullmatch(r'bytes=([0-9]*)-([0-9]*)', value)
    require(match is not None and any(match.groups()), 'One ordinary byte range required')
    left, right = match.groups()
    if not left:
        suffix = int(right)
        require(suffix > 0, 'Positive suffix range required')
        start, end = max(0, size-suffix), size-1
    else:
        start, end = int(left), min(int(right), size-1) if right else size-1
        require(start < size and 0 <= start <= end, 'Range must intersect exact published bytes')
    return start, end


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Azure preview redirects forbidden')


class AzureReader:
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token, self.expiry = None, 0
        self.lock = threading.Lock()

    def authorization(self):
        with self.lock:
            if time.time()+300 >= self.expiry:
                env = os.environ.copy()
                env.update(AZURE_LOGGING_ENABLE_LOG_FILE='false', AZURE_CORE_COLLECT_TELEMETRY='false')
                try:
                    # No shell, printed environment, token file or debug output.
                    result = subprocess.run(['rtk', 'proxy', 'az', 'account', 'get-access-token',
                        '--resource', 'https://storage.azure.com/', '-o', 'json', '--only-show-errors'],
                        env=env, capture_output=True, check=False, timeout=45)
                    require(result.returncode == 0 and 0 < len(result.stdout) <= 32768,
                        'Existing Azure user identity required')
                    reply = strict(result.stdout)
                    require(type(reply.get('accessToken')) is str and 0 < len(reply['accessToken']) < 16384,
                        'Bounded Azure user bearer required')
                    self.token = reply['accessToken']
                    self.expiry = int(reply['expires_on'])
                except Exception:
                    raise RuntimeError('Azure preview authentication unavailable') from None
            return {'Authorization': 'Bearer '+self.token}

    def request(self, method, row, headers=None):
        require(method in {'HEAD', 'GET'} and re.fullmatch(
            r'full4d-[0-9a-f]{40}/episode_[0-9]{6}\.(?:mp4|jpg)', row['name']),
            'Only exact private preview reads allowed')
        try:
            request = urllib.request.Request(ENDPOINT+'/'+row['name'], method=method,
                headers={'x-ms-version': '2023-11-03', 'If-Match': row['etag'],
                    **self.authorization(), **(headers or {})})
            return self.opener.open(request, timeout=30)
        except Exception:
            # Never include Azure response bodies, credentials, URLs or subprocess outputs.
            raise RuntimeError('Pinned private preview read unavailable') from None

    def verify_head(self, row):
        with self.request('HEAD', row) as response:
            require(response.status == 200 and response.headers.get('ETag') == row['etag']
                and int(response.headers.get('Content-Length', '-1')) == row['bytes']
                and response.headers.get('x-ms-meta-sha256') == row['sha256'],
                'Published preview identity changed')

    def open(self, row, selected):
        self.verify_head(row)
        header = {} if selected is None else {'Range': f'bytes={selected[0]}-{selected[1]}'}
        response = self.request('GET', row, header)
        try:
            expected_status = 200 if selected is None else 206
            expected_length = row['bytes'] if selected is None else selected[1]-selected[0]+1
            require(response.status == expected_status and response.headers.get('ETag') == row['etag']
                and int(response.headers.get('Content-Length', '-1')) == expected_length,
                'Exact pinned Azure byte range required')
            if selected is not None:
                require(response.headers.get('Content-Range') ==
                    f'bytes {selected[0]}-{selected[1]}/{row["bytes"]}', 'Exact pinned range response required')
            return response, expected_length
        except Exception:
            response.close()
            raise


def index_html(episodes):
    """Only localhost media resources; tokens never enter the browser document."""
    panes = '\n'.join(f'<section><h2>Épisode {ep:02d}</h2>'
        f'<video controls playsinline preload="none" poster="/episode-{ep}.jpg" '
        f'aria-label="Épisode {ep}: vidéo originale à gauche, reconstruction4D à droite">'
        f'<source src="/episode-{ep}.mp4" type="video/mp4"></video></section>' for ep in episodes)
    return ('<!doctype html><html lang="fr"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>World Reward — Reconstructions4D</title>'
        '<style>:root{color-scheme:light dark}body{font:16px system-ui;margin:24px;max-width:1280px;'
        'margin-inline:auto;padding-inline:16px}h1,h2{font-weight:500}h1{font-size:22px}'
        'h2{font-size:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));'
        'gap:24px}video{width:100%;height:auto;display:block}p{line-height:1.5}</style>'
        '<h1>World Reward — Reconstructions4D</h1><p>Original à gauche · reconstruction à droite · '
        'même échelle inférée, non vérifiée · sol virtuel estimé. Lecture à la demande, sans fichier vidéo enregistré.</p>'
        '<main>'+panes+'</main></html>').encode()


class Proxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    def handle_error(self, request, client_address):
        # Base server tracebacks could expose upstream exception/request state.
        pass


def handler_factory(episodes, allowed, reader):
    html = index_html(episodes)
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_): pass

        def valid_local_request(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            require(time.monotonic() <= self.server.deadline
                and self.client_address[0] == '127.0.0.1'
                and self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
                and self.headers.get('Origin', origin) == origin
                and self.headers.get('Sec-Fetch-Site', 'none') in {'none', 'same-origin'}
                and not self.headers.get('Authorization') and not self.headers.get('Transfer-Encoding')
                and self.headers.get('Content-Length', '0') == '0', 'Loopback same-origin read only')
            parsed = urllib.parse.urlsplit(self.path)
            require(not parsed.query and not parsed.fragment and not parsed.scheme and not parsed.netloc
                and self.path == parsed.path, 'No redirects, queries, credentials or arbitrary paths')
            return parsed.path

        def response_headers(self, status, mime, length, selected=None, total=None):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Content-Security-Policy',
                "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; media-src 'self'; frame-ancestors 'none'")
            if total is not None:
                self.send_header('Accept-Ranges', 'bytes')
            if selected is not None:
                self.send_header('Content-Range', f'bytes {selected[0]}-{selected[1]}/{total}')
            self.end_headers()

        def failure(self, status):
            self.response_headers(status, 'text/plain; charset=utf-8', 0)

        def serve(self, head=False):
            response = None
            try:
                path = self.valid_local_request()
                if path == '/':
                    self.response_headers(200, 'text/html; charset=utf-8', len(html))
                    if not head: self.wfile.write(html)
                    return
                if path not in allowed:
                    self.failure(404); return
                row = allowed[path]
                try:
                    selected = byte_range(self.headers.get('Range'), row['bytes'])
                except ValueError:
                    self.failure(416); return
                if head:
                    reader.verify_head(row)
                    self.response_headers(200, row['mime'], row['bytes'], total=row['bytes'])
                    return
                response, length = reader.open(row, selected)
                self.response_headers(200 if selected is None else 206, row['mime'], length,
                    selected=selected, total=row['bytes'])
                remaining = length
                while remaining:
                    block = response.read(min(CHUNK, remaining))
                    require(block and len(block) <= min(CHUNK, remaining), 'Complete bounded Azure range required')
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                if response is None:
                    try: self.failure(502)
                    except Exception: pass
                self.close_connection = True
            finally:
                if response is not None: response.close()

        def do_GET(self): self.serve()
        def do_HEAD(self): self.serve(head=True)
        def do_POST(self): self.failure(405); self.close_connection = True
        def do_PUT(self): self.failure(405); self.close_connection = True
        def do_DELETE(self): self.failure(405); self.close_connection = True
        def do_OPTIONS(self): self.failure(405); self.close_connection = True
    return Handler


def make_server(episodes, allowed, *, reader=None):
    server = Proxy(('127.0.0.1', 0), handler_factory(episodes, allowed, reader or AzureReader()))
    server.deadline = time.monotonic()+MAX_TTL
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--ttl', type=int, default=MAX_TTL)
    args = parser.parse_args()
    require(60 <= args.ttl <= MAX_TTL, 'Bounded60seconds-to1hour viewer lifetime required')
    episodes, allowed, _ = load_receipt(args.receipt, args.revision)
    server = make_server(episodes, allowed)
    server.deadline = time.monotonic()+args.ttl
    timer = threading.Timer(args.ttl, server.shutdown); timer.daemon = True; timer.start()
    # This URL has no credential/query. No request/media/token logging follows.
    print(f'FULL4D_STREAM_URL http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel(); server.server_close()


if __name__ == '__main__':
    main()

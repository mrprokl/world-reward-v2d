"""Azure CPU public-Drive download only; measured hashes are not publisher pins."""
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import re
import threading
import time
import urllib.parse
import urllib.request

import dexycb_acquire as dex

ROOT = dex.ROOT
INCOMING = dex.INCOMING
RESULT = 'results/dexycb-identity-download-v1'
BUDGET, BLOCK, TIMEOUT = 7200, 1 << 20, 30
MIME = {'application/octet-stream', 'application/gzip', 'application/x-gzip', 'application/x-tar'}
HOSTS = {'drive.google.com', 'drive.usercontent.google.com'}


class DownloadError(ValueError):
    pass


def require(value, message):
    if not value: raise DownloadError(message)


def safe_url(url):
    value = urllib.parse.urlsplit(url)
    require(value.scheme == 'https' and value.hostname in HOSTS and value.port in (None, 443)
            and not value.username and not value.password and not value.fragment,
            'Unsupported public Google redirect')
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_url(row):
    match = re.fullmatch(r'https://drive\.google\.com/file/d/([A-Za-z0-9_-]{20,100})', row['url'])
    require(match is not None, 'Exact frozen publisher Drive file URL required')
    return 'https://drive.usercontent.google.com/download?id=' + match[1] + '&export=download&confirm=t'


def publish(part, target):
    """Linux atomic NOREPLACE: never merge, replace or resume an archive."""
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.renameat2
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(part), -100, os.fsencode(target), 1):
        raise OSError(ctypes.get_errno(), 'Archive publication refused')


def source(protocol_path):
    paths = {'driver': Path(__file__), 'acquisition_helper': Path(dex.__file__), 'protocol': protocol_path}
    observed = {name: dex.identity(path, readonly=True) for name, path in paths.items()}
    revision = os.environ.get('WR_CODE_REVISION')
    if revision:
        require(re.fullmatch(r'[0-9a-f]{40}', revision), 'Exact producer revision required')
        code = dex.canonical(protocol_path.parent.parent)
        require(code == ROOT / 'jobs' / revision / 'run_dexycb_download' / 'code'
                and Path(__file__) == code / 'infra/dexycb_download.py'
                and Path(dex.__file__) == code / 'infra/dexycb_acquire.py'
                and protocol_path == code / 'configs/dexycb_identity_protocol.json', 'Actual producer paths differ')
        markers = {name: dex.identity(code.parent / name, readonly=True) for name in ('revision', 'source-sha256')}
        require((code.parent / 'revision').read_bytes() == (revision + '\n').encode()
                and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_bytes()),
                'Original dispatch markers differ')
        closure = {}
        for path in (code, *sorted(code.rglob('*'))):
            dex.canonical(path); require(not path.lstat().st_mode & 0o222, 'Readonly complete source closure required')
            name = str(path.relative_to(code))
            closure[name] = {'directory': True} if path.is_dir() else dex.identity(path, readonly=True, empty=True)
        observed['dispatch'] = {'producer_revision': revision, 'markers': markers, 'entries': len(closure),
                               'closure_sha256': hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()}
    protocol = dex.strict_json(protocol_path.read_bytes()); dex.exact(protocol, dex.EXPECTED_PROTOCOL)
    for subject, row in protocol['archives'].items():
        require(row['file'] == subject + '.tar.gz' and type(row['bytes']) is int and row['bytes'] > 0,
                'Frozen archive name/positive byte count required')
        download_url(row)
    return protocol, paths, observed


def fetch(row, incoming, opener, deadline, stop, owned, progress):
    started = time.monotonic(); part = incoming / (row['file'] + '.part'); target = incoming / row['file']
    require(not stop.is_set() and started < deadline, 'Download budget/cancellation')
    response = opener.open(urllib.request.Request(download_url(row), headers={'Accept-Encoding': 'identity'}),
                           timeout=TIMEOUT)
    with response:
        require(response.status == 200, 'Publisher HTTP status rejected'); safe_url(response.geturl())
        require(response.headers.get_content_type().lower() in MIME, 'Publisher returned HTML/text or unsupported media')
        length = response.headers.get('Content-Length')
        require(length is None or re.fullmatch(r'[0-9]+', length) and int(length) == row['bytes'],
                'Publisher content length differs')
        require(response.headers.get('Content-Encoding', 'identity').lower() == 'identity',
                'Unexpected HTTP content encoding')
        digest = hashlib.sha256(); count = 0; prefix = b''
        with part.open('xb') as output:
            os.fchmod(output.fileno(), 0o600); owned.append((part, dex.state(part)[:2]))
            while True:
                require(not stop.is_set() and time.monotonic() < deadline, 'Download budget/cancellation')
                block = response.read(BLOCK)
                require(len(block) <= BLOCK and count + len(block) <= row['bytes'], 'Archive overflow')
                if not block: break
                prefix = (prefix + block)[:2] if count < 2 else prefix
                if len(prefix) == 2: require(prefix == b'\x1f\x8b', 'Publisher body is not original gzip')
                output.write(block); digest.update(block); count += len(block)
                progress[row['file']]['bytes_received'] = count
            require(prefix == b'\x1f\x8b' and count == row['bytes'], 'Truncated archive')
            output.flush(); os.fsync(output.fileno()); os.fchmod(output.fileno(), 0o444)
        require(not stop.is_set() and time.monotonic() < deadline, 'Download budget/cancellation')
        publish(part, target)
        fd = os.open(incoming, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return {'file': row['file'], 'bytes': count, 'sha256': digest.hexdigest(),
            'elapsed_seconds': time.monotonic() - started, 'hash_basis': 'caller_measured_not_publisher_checksum'}


def download(root, protocol_path, *, opener=None):
    root, protocol_path = dex.canonical(root), dex.canonical(protocol_path)
    protocol, paths, before = source(protocol_path)
    incoming = dex.canonical(INCOMING); out = dex.canonical(root / RESULT)
    require(not incoming.exists() and not out.exists() and incoming.parent.is_dir() and out.parent.is_dir(),
            'Fresh incoming/result namespace required; no resume')
    out.mkdir(mode=0o700); out.chmod(0o700)
    started = time.monotonic(); stop = threading.Event(); owned = []; rows = []; failure = None
    progress = {r['file']: {'file': r['file'], 'expected_bytes': r['bytes'], 'bytes_received': 0}
                for r in protocol['archives'].values()}
    report = {'stage': 'dexycb_identity_public_drive_download', 'status': 'fail', 'phase': 'download',
              'budget_seconds': BUDGET, 'protocol_identity': before['protocol'], 'source_helpers': before,
              'publisher_checksum_verified': False, 'models_loaded': False, 'gpu_used': False,
              'annotation_values_parsed': False, 'training_overlap_verified': False,
              'challenge_overlap_verified': False, 'archives': rows,
              'download_progress': [progress[name] for name in sorted(progress)]}
    try:
        incoming.mkdir(mode=0o700); incoming.chmod(0o700)
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fetch, row, incoming, opener, started + BUDGET, stop, owned, progress)
                       for row in protocol['archives'].values()]
            for future in concurrent.futures.as_completed(futures):
                try: rows.append(future.result())
                except Exception as exc:
                    stop.set(); failure = failure or exc
        if failure: raise failure
        require(time.monotonic() - started <= BUDGET, 'Download budget exceeded')
        require(source(protocol_path)[2] == before,
                'Source/protocol changed during download')
        report.update(status='pass', phase='complete')
    except Exception as exc:
        failure = exc; report.update(phase='failed', error_type=type(exc).__name__)
        if isinstance(exc, DownloadError): report['reason'] = str(exc)
    finally:
        for part, inode in owned:
            if part.exists() and not part.is_symlink() and dex.state(part)[:2] == inode: part.unlink()
        try: report['source_rehashed_after'] = source(protocol_path)[2] == before
        except Exception: report['source_rehashed_after'] = False
        if not report['source_rehashed_after']: report.update(status='fail', phase='failed'); failure = failure or DownloadError('Source mutation')
        rows.sort(key=lambda row: row['file']); report['elapsed_seconds'] = time.monotonic() - started
        report['owned_partials_removed'] = all(not p.exists() for p, _ in owned)
        with (out / 'report.json').open('x') as output:
            os.fchmod(output.fileno(), 0o444); json.dump(report, output, indent=2, allow_nan=False); output.write('\n')
    if failure: raise DownloadError('DexYCB download failed; inspect sealed receipt') from None
    return report


def main():
    require(platform.system() == 'Linux' and os.environ.get('WR_ROOT') == str(ROOT)
            and os.getuid() == pwd.getpwnam('scenesmith').pw_uid, 'Azure CPU scenesmith runtime required')
    code = dex.canonical(Path(os.environ['WR_CODE']))
    report = download(ROOT, code / 'configs/dexycb_identity_protocol.json')
    print(json.dumps({'stage': report['stage'], 'status': report['status'], 'archives': len(report['archives']),
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'stage': 'dexycb_identity_public_drive_download', 'status': 'fail',
                          'error_type': type(exc).__name__}))
        raise SystemExit(1) from None

"""Narrow Azure-only byte replica of the already-qualified Boots runtime.

Host stdlib/CPU only: exact Docker image, seven source files, one checkpoint and
one CPU receipt. No model/data/GPU execution, retagging or qualification replay.
Managed-identity tokens stay in RAM; optional SAS is never recorded.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
IMAGE = 'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
LAYERS_SHA = '0f1bf78024834b90e5841b8de8893fba8eec4176bf36c071516d0c1205d84fdc'
PROTOCOL = 'configs/robotap_boots_protocol.json'
PROTOCOL_PIN = {'bytes': 6339, 'sha256': '830cbf41b88ab7a2c856884172f226eb4cd7773b169d647af33b729d0d7ac1cf'}
RUNTIME = 'results/bootstapir-runtime-verify-f3cfde1992c6cfc9a6504b393d8a1f24735dc94d/report.json'
RUNTIME_PIN = {'bytes': 3051, 'sha256': '794b6e5313e992c2c86610f855f4a86dc8c9a994ede37c22e33093e6ad9fc5e7'}
BASE = 'validation/robotap_boots_v1/assets'
BLOCK = 4*1024*1024
MAX_IMAGE = 32*1024**3
MAX_ARCHIVE = 33*1024**3
BUDGET = 7200
SCHEMA = 'world_reward.articulated_runtime_replica.v1'


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not any(q.is_symlink() for q in (p, *p.parents)),
            'Canonical nonsymlink path required')
    return p


def identity(path):
    p = canonical(path); before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= MAX_IMAGE,
            'Positive unaliased regular file required')
    with p.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(BLOCK), b''): digest.update(chunk)
    after = p.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'File changed during hashing')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: require(False, 'Nonfinite JSON'))


def asset_pins(code):
    require(identity(code/PROTOCOL) == PROTOCOL_PIN, 'Original whole Boots protocol required')
    protocol = strict_json((code/PROTOCOL).read_bytes())
    require(protocol['source']['revision'] == '730cda1c730877cfedbe01bf87fb1cadb78a565d'
            and len(protocol['source']['files']) == 7, 'Exactly seven original TAPNet source files required')
    files = {f'{BASE}/tapnet_source/{name}': {k: row[k] for k in ('bytes', 'sha256')}
             for name, row in protocol['source']['files'].items()}
    files[f'{BASE}/bootstapir_checkpoint_v2.pt'] = {k: protocol['checkpoint'][k] for k in ('bytes', 'sha256')}
    files[RUNTIME] = dict(RUNTIME_PIN)
    require(len(files) == 9 and all(PurePosixPath(p).as_posix() == p and '..' not in PurePosixPath(p).parts
            and not p.startswith('/') for p in files), 'Exact canonical narrow asset paths required')
    return files


def source_proof(code, revision):
    canonical(code)
    require(code == ROOT/'jobs'/revision/'run_articulated_runtime_transfer/code'
            and re.fullmatch('[0-9a-f]{40}', revision), 'Exact committed transfer namespace required')
    digest = hashlib.sha256(); count = 0
    for path in (code, *sorted(code.rglob('*'))):
        canonical(path); s = path.lstat()
        require(not s.st_mode & 0o222 and (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)), 'Entire code closure must be readonly')
        if path.is_file(): digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(identity(path)['sha256'])); count += 1
    for name in ('revision', 'source-sha256'):
        path = canonical(code.parent/name); s = path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= 100, 'Original regular unaliased dispatch marker required')
        raw = path.read_bytes()
        require(raw == (revision+'\n').encode() if name == 'revision' else re.fullmatch(b'[0-9a-f]{64}\n', raw), 'Original dispatch markers required')
        digest.update(name.encode()+b'\0'+raw)
    return {'files': count, 'sha256': digest.hexdigest()}


def inspect_image():
    result = subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True, timeout=30, check=False)
    require(result.returncode == 0 and len(result.stdout) <= 1024**2, 'Original exact image inspection failed')
    rows = strict_json(result.stdout); require(type(rows) is list and len(rows) == 1, 'One exact Docker image required')
    image = rows[0]; layers = image['RootFS']['Layers']
    require(image['Id'] == IMAGE and image['Architecture'] == 'amd64' and image['Os'] == 'linux'
            and len(layers) == 47 and all(re.fullmatch('sha256:[0-9a-f]{64}', p) for p in layers)
            and hashlib.sha256(json.dumps(layers, separators=(',', ':')).encode()).hexdigest() == LAYERS_SHA,
            'Exact qualified 47-layer image required')
    return {'image_id': IMAGE, 'layers': 47, 'ordered_rootfs_sha256': LAYERS_SHA, 'size': image['Size']}


def verify_image_archive(path):
    """Validate Docker's byte archive before load; never decode layer payloads."""
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        require(len(members) <= 200 and len(set(names)) == len(names)
                and all(not m.pax_headers and (m.isreg() or m.isdir()) and not m.name.startswith('/')
                        and '..' not in PurePosixPath(m.name).parts for m in members), 'Safe exact image archive layout required')
        table = {m.name: m for m in members}
        require('manifest.json' in table and 0 < table['manifest.json'].size <= 65536, 'Bounded Docker manifest required')
        rows = strict_json(archive.extractfile(table['manifest.json']).read())
        require(type(rows) is list and len(rows) == 1 and set(rows[0]) == {'Config', 'RepoTags', 'Layers'}
                and rows[0]['RepoTags'] in (None, []) and type(rows[0]['Layers']) is list and len(rows[0]['Layers']) == 47,
                'One untagged original47-layer Docker image required')
        row = rows[0]; config_member = table.get(row['Config'])
        require(config_member and config_member.isreg() and 0 < config_member.size <= 1024**2, 'Bounded image config required')
        raw = archive.extractfile(config_member).read(); config = strict_json(raw)
        layers = config['rootfs']['diff_ids']
        require('sha256:'+hashlib.sha256(raw).hexdigest() == IMAGE and config['architecture'] == 'amd64'
                and config['os'] == 'linux' and config['rootfs']['type'] == 'layers' and len(layers) == 47
                and hashlib.sha256(json.dumps(layers, separators=(',', ':')).encode()).hexdigest() == LAYERS_SHA,
                'Original image config/ordered layer identities differ')
        require(len(set(row['Layers'])) == 47 and all(p in table and table[p].isreg() for p in row['Layers']), 'All exact regular image layers required')
        for name, wanted in zip(row['Layers'], layers):
            digest = hashlib.sha256()
            with archive.extractfile(table[name]) as stream:
                for chunk in iter(lambda: stream.read(BLOCK), b''): digest.update(chunk)
            require('sha256:'+digest.hexdigest() == wanted, 'Original uncompressed image layer bytes differ')
        allowed = {'manifest.json', row['Config'], *row['Layers']}
        for layer in row['Layers']:
            if not layer.startswith('blobs/'):
                for suffix in ('VERSION', 'json'):
                    name = str(PurePosixPath(layer).parent/suffix)
                    if name in table:
                        require(table[name].isreg() and 0 < table[name].size <= 1024**2, 'Bounded classic layer metadata required')
                        allowed.add(name)
        if 'repositories' in table:
            require(table['repositories'].isreg() and 0 < table['repositories'].size <= 65536
                    and strict_json(archive.extractfile(table['repositories']).read()) == {}, 'No image tags may be imported')
            allowed.add('repositories')
        require(all(m.isdir() and any(p.startswith(m.name.rstrip('/')+'/') for p in allowed) or m.name in allowed
                    for m in members), 'Unexpected Docker archive metadata/member rejected')


class Blob:
    """No redirects/proxies/credentials in errors; narrowly scoped Azure identity."""
    def __init__(self, url, export_revision, opener=None, *, managed_identity=False):
        p = urllib.parse.urlsplit(url)
        require(p.scheme == 'https' and re.fullmatch(r'[a-z0-9]{3,24}\.blob\.core\.windows\.net', p.netloc)
                and not p.fragment and re.fullmatch(rf'/[a-z0-9][a-z0-9-]{{1,61}}/articulated-runtime-{export_revision}\.tar', p.path)
                and (any(k == 'sig' and v for k, v in urllib.parse.parse_qsl(p.query))
                     or managed_identity is True and not p.query
                        and p.netloc == 'stworldrewardresearch26.blob.core.windows.net'),
                'Private expected Azure blob credential required')
        self.url = url; self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.managed_identity = managed_identity
        self.token = None; self.token_expiry = 0

    def authorization(self):
        if not self.managed_identity: return {}
        if time.time()+300 >= self.token_expiry:
            try:
                request = urllib.request.Request('http://169.254.169.254/metadata/identity/oauth2/token'
                    '?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F', headers={'Metadata': 'true'})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=10) as response: raw = response.read(32769)
                require(len(raw) <= 32768, 'Bounded managed identity response required')
                value = strict_json(raw)
                require(value['token_type'].lower() == 'bearer' and type(value['access_token']) is str
                    and 0 < len(value['access_token']) < 16384, 'Managed identity credential required')
                self.token = value['access_token']; self.token_expiry = int(value['expires_on'])
            except Exception: raise RuntimeError('Private Azure identity request failed') from None
        return {'Authorization': 'Bearer '+self.token}

    def request(self, method, data=None, query='', headers=None):
        try:
            url = self.url+(('?' if '?' not in self.url else '&')+query if query else '')
            request = urllib.request.Request(url, data=data, method=method,
                headers={'x-ms-version': '2023-11-03', **self.authorization(), **(headers or {})})
            return self.opener.open(request, timeout=90)
        except Exception:
            raise RuntimeError('Private Azure byte transfer request failed') from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): raise RuntimeError('Private transfer redirects forbidden')


class BlockWriter:
    def __init__(self, blob):
        self.blob = blob; self.buffer = bytearray(); self.digest = hashlib.sha256(); self.size = 0
        self.blocks = []; self.prefix = os.urandom(12).hex()

    def _block(self, data):
        token = base64.b64encode(f'{self.prefix}-{len(self.blocks):06d}'.encode()).decode()
        with self.blob.request('PUT', bytes(data), 'comp=block&blockid='+urllib.parse.quote(token, safe='')) as response:
            require(response.status == 201, 'Azure block upload rejected')
        self.blocks.append(token)

    def write(self, data):
        require(type(data) is bytes and self.size+len(data) <= MAX_ARCHIVE, 'Bounded binary archive stream required')
        self.digest.update(data); self.size += len(data); self.buffer.extend(data)
        while len(self.buffer) >= BLOCK:
            self._block(self.buffer[:BLOCK]); del self.buffer[:BLOCK]
        return len(data)

    def finish(self):
        if self.buffer: self._block(self.buffer); self.buffer.clear()
        raw = ('<?xml version="1.0" encoding="utf-8"?><BlockList>'+''.join(f'<Latest>{p}</Latest>' for p in self.blocks)+'</BlockList>').encode()
        with self.blob.request('PUT', raw, 'comp=blocklist', {'Content-Type': 'application/xml', 'If-None-Match': '*'}) as response:
            require(response.status == 201, 'Exclusive new Azure blob commit rejected')
        return {'bytes': self.size, 'sha256': self.digest.hexdigest()}


def pack(writer, root, files, image_path, image_pin, revision, proof):
    manifest = {'schema': SCHEMA, 'export_revision': revision, 'source_proof': proof,
        'image': {'image_id': IMAGE, 'layers': 47, 'ordered_rootfs_sha256': LAYERS_SHA, **image_pin},
        'assets': files, 'replica_only': True, 'qualification_or_model_execution': False}
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    require(len(raw) <= 16384, 'Bounded first manifest required')
    with tarfile.open(fileobj=writer, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo('manifest.json'); info.size = len(raw); info.mode = 0o444; archive.addfile(info, io.BytesIO(raw))
        for name, path, pin in [(p, root/p, v) for p, v in sorted(files.items())]+[('image.tar', image_path, image_pin)]:
            require(identity(path) == pin, 'Original transfer artifact differs')
            info = tarfile.TarInfo(name); info.size = pin['bytes']; info.mode = 0o444
            with path.open('rb') as stream: archive.addfile(info, stream)
    return manifest


def download(blob, path, expected):
    digest = hashlib.sha256(); size = 0
    with blob.request('GET') as response, path.open('xb') as output:
        require(response.status == 200 and int(response.headers.get('Content-Length', '-1')) == expected['bytes'], 'Exact incoming blob length required')
        while size < expected['bytes']:
            data = response.read(min(BLOCK, expected['bytes']-size))
            require(data and len(data) <= min(BLOCK, expected['bytes']-size), 'Truncated blob stream')
            output.write(data); digest.update(data); size += len(data)
        require(not response.read(1) and digest.hexdigest() == expected['sha256'], 'Incoming blob bytes differ')
        output.flush(); os.fsync(output.fileno())
    require(identity(path) == expected, 'Incoming archive posthash differs')


def unpack(path, scratch, files, export_revision):
    with tarfile.open(path, 'r:') as archive:
        member = archive.next()
        require(member and member.name == 'manifest.json' and member.isreg() and not member.pax_headers
                and 0 < member.size <= 16384, 'First bounded regular manifest required')
        manifest = strict_json(archive.extractfile(member).read())
        require(set(manifest) == {'schema', 'export_revision', 'source_proof', 'image', 'assets', 'replica_only', 'qualification_or_model_execution'}
                and manifest['schema'] == SCHEMA and manifest['export_revision'] == export_revision
                and manifest['assets'] == files and manifest['replica_only'] is True
                and manifest['qualification_or_model_execution'] is False, 'Exact independently pinned narrow manifest required')
        proof = manifest['source_proof']
        require(type(proof) is dict and set(proof) == {'files', 'sha256'} and type(proof['files']) is int and proof['files'] > 0
                and re.fullmatch('[0-9a-f]{64}', proof['sha256']), 'Original export source identity required')
        image = manifest['image']
        require(set(image) == {'image_id', 'layers', 'ordered_rootfs_sha256', 'bytes', 'sha256'}
                and image['image_id'] == IMAGE and image['layers'] == 47 and image['ordered_rootfs_sha256'] == LAYERS_SHA
                and type(image['bytes']) is int and 0 < image['bytes'] <= MAX_IMAGE
                and re.fullmatch('[0-9a-f]{64}', image['sha256']), 'Exact image byte manifest required')
        allowed = {**files, 'image.tar': {k: image[k] for k in ('bytes', 'sha256')}}; seen = {'manifest.json'}
        extracted = {}
        while (member := archive.next()) is not None:
            require(member.name in allowed and member.name not in seen and member.isreg() and not member.pax_headers
                    and member.size == allowed[member.name]['bytes'] and member.mode == 0o444,
                    'Only exact unaliased regular allowlisted tar members accepted')
            seen.add(member.name); target = scratch/f'item-{len(extracted):02d}'
            with target.open('xb') as output:
                shutil.copyfileobj(archive.extractfile(member), output, BLOCK)
            require(identity(target) == allowed[member.name], 'Extracted exact file pin differs'); extracted[member.name] = target
        require(seen == {'manifest.json', *allowed}, 'Complete exact transfer inventory required')
    return manifest, extracted


def install(extracted, root, files):
    """No existing file overwritten; rollback only inode-bound newly created leaves."""
    require(all(not (canonical(root/p)).exists() for p in files), 'All nine destination leaves must be absent')
    owned = []
    try:
        for name in sorted(files):
            target = root/name
            missing = []; parent = target.parent
            while not parent.exists(): missing.append(parent); parent = parent.parent
            for folder in reversed(missing): folder.mkdir(mode=0o755); os.chown(folder, 1000, 1000)
            for folder in (target.parent, *target.parent.parents):
                canonical(folder)
                if folder == root: break
                require(folder.is_dir() and stat.S_IMODE(folder.stat().st_mode) == 0o755, 'Original install directories must remain0755')
            with target.open('xb') as output:
                s = os.fstat(output.fileno()); owned.append((target, s.st_dev, s.st_ino))
                os.fchmod(output.fileno(), 0o444); os.fchown(output.fileno(), 1000, 1000)
                with extracted[name].open('rb') as source: shutil.copyfileobj(source, output, BLOCK)
                output.flush(); os.fsync(output.fileno())
            require(identity(target) == files[name], 'Installed replica byte identity differs')
        return {p: identity(root/p) for p in files}
    except BaseException:
        for path, device, inode in reversed(owned):
            s = path.lstat()
            if (s.st_dev, s.st_ino) == (device, inode): path.unlink()
        raise


def verify_azure_peer(command):
    request = urllib.request.Request('http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01',
                                    headers={'Metadata': 'true'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5) as response:
            raw = response.read(65537)
        require(len(raw) <= 65536, 'Bounded Azure peer metadata required'); peer = strict_json(raw)
        expected = ('world-reward-ncc-h100-02', 'world-reward-research') if command == 'export' else ('scenesmith-ncc-h100-01', 'scenesmith-h100')
        require((peer.get('name'), str(peer.get('resourceGroupName', '')).lower()) == expected, 'Exact Azure exporter/receiver peer required')
    except Exception: raise RuntimeError('Owned Azure peer verification failed') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('export', 'import')); parser.add_argument('--export-revision')
    parser.add_argument('--expected-bytes', type=int); parser.add_argument('--expected-sha256'); args = parser.parse_args(argv)
    revision = os.environ.get('WR_CODE_REVISION', ''); code = Path(os.environ.get('WR_CODE', '/invalid'))
    require(platform.system() == 'Linux' and os.geteuid() == 0 and os.environ.get('WR_ROOT') == str(ROOT)
            and Path(__file__) == code/'infra/articulated_runtime_transfer.py', 'Azure-only root stdlib transfer entry required')
    verify_azure_peer(args.command)
    proof = source_proof(code, revision); files = asset_pins(code)
    export_revision = revision if args.command == 'export' else args.export_revision
    require(type(export_revision) is str and re.fullmatch('[0-9a-f]{40}', export_revision), 'Exact original export revision required')
    managed = os.environ.pop('WR_BOOTSTRAP_MANAGED_IDENTITY', '') == '1'
    blob = Blob(os.environ.pop('WR_BOOTSTRAP_BLOB_URL', ''), export_revision, managed_identity=managed)
    expected = {'bytes': args.expected_bytes, 'sha256': args.expected_sha256}
    require(args.command == 'export' and args.export_revision is None and args.expected_bytes is None and args.expected_sha256 is None
            or args.command == 'import' and type(expected['bytes']) is int and 0 < expected['bytes'] <= MAX_ARCHIVE
            and type(expected['sha256']) is str and re.fullmatch('[0-9a-f]{64}', expected['sha256']), 'Export or independently pinned import arguments required')
    os.environ['DOCKER_HOST'] = f'unix://{ROOT}/docker.sock'
    out = canonical(ROOT/f'results/articulated-runtime-transfer-{args.command}-{revision}')
    require(not out.exists() and out.parent.is_dir(), 'Fresh transfer receipt namespace required')
    out.mkdir(mode=0o700); scratch = out/'scratch'; scratch.mkdir(mode=0o700)
    report = {'schema': SCHEMA, 'stage': 'articulated_runtime_'+args.command, 'status': 'fail', 'phase': 'preflight',
        'producer_revision': revision, 'source_proof': proof, 'budget_seconds': BUDGET, 'replica_only': True,
        'model_execution': False, 'gpu_used': False, 'dataset_transferred': False, 'secrets_recorded': False}
    started = time.monotonic()
    def expired(*unused): raise TimeoutError('Inclusive transfer7200s budget exhausted')
    previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}; signal.alarm(BUDGET)
    try:
        if args.command == 'export':
            image = inspect_image(); original = {p: identity(ROOT/p) for p in files}
            require(original == files, 'All original nine asset bytes required')
            require(shutil.disk_usage(out).free >= image['size']+sum(v['bytes'] for v in files.values())+2*1024**3, 'Bounded export disk capacity insufficient')
            image_path = scratch/'image.tar'; report['phase'] = 'image_save'
            result = subprocess.run(['docker', 'image', 'save', '--output', str(image_path), IMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=BUDGET-(time.monotonic()-started))
            require(result.returncode == 0, 'Exact image byte save failed'); image_pin = identity(image_path)
            verify_image_archive(image_path)
            writer = BlockWriter(blob); report['phase'] = 'upload'
            manifest = pack(writer, ROOT, files, image_path, image_pin, revision, proof)
            report['archive'] = writer.finish(); report['image'] = manifest['image']
            report['assets'] = original
            require(original == {p: identity(ROOT/p) for p in files} and image == inspect_image(), 'Original assets/image changed')
        else:
            require(all(not canonical(ROOT/p).exists() for p in files), 'Receiver assets must be absent before download')
            absent = subprocess.run(['docker', 'image', 'inspect', IMAGE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            require(absent.returncode != 0, 'Receiver exact image already exists; no implicit reuse')
            require(shutil.disk_usage(out).free >= 3*expected['bytes']+2*1024**3, 'Bounded import disk capacity insufficient')
            archive_path = scratch/'archive.tar'; report['phase'] = 'download'; download(blob, archive_path, expected)
            manifest, extracted = unpack(archive_path, scratch, files, export_revision); report['phase'] = 'image_load'
            verify_image_archive(extracted['image.tar'])
            result = subprocess.run(['docker', 'image', 'load', '--input', str(extracted['image.tar'])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=BUDGET-(time.monotonic()-started))
            require(result.returncode == 0, 'Original image byte load failed'); inspect_image()
            report['assets'] = install(extracted, ROOT, files); report['archive'] = expected; report['image'] = manifest['image']
            report['original_export_source_proof'] = manifest['source_proof']
        require(proof == source_proof(code, revision) and time.monotonic()-started <= BUDGET, 'Source changed or transfer deadline exhausted')
        report.update(status='pass', phase='complete', asset_files=9)
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error='Narrow runtime byte transport failed closed; no secret diagnostics')
    finally:
        signal.alarm(0)
        for s, handler in previous.items(): signal.signal(s, handler)
        for path in scratch.iterdir():
            require(path.is_file() and not path.is_symlink(), 'Only own scratch regular files may be removed'); path.unlink()
        scratch.rmdir(); report['scratch_removed'] = True; report['elapsed_seconds'] = time.monotonic()-started
        with (out/'report.json').open('x') as stream:
            os.fchmod(stream.fileno(), 0o400); json.dump(report, stream, sort_keys=True, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({k: report[k] for k in ('stage', 'status', 'scratch_removed')}) , flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    try: sys.exit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit): raise
        print(json.dumps({'stage': 'articulated_runtime_transfer', 'status': 'fail', 'error_type': type(exc).__name__,
                          'error': 'Transfer precondition failed; details suppressed'}), flush=True); sys.exit(1)

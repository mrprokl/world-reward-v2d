"""Bounded CPU-only full original MMCV SM90 build on a new author-runtime child.

No checkpoint/model/data/GPU. The existing runtime and 51 non-MMCV packages
remain unchanged. This producer has its own lineage, never an old-runtime PASS.
"""
import argparse
import email
import hashlib
import io
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
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
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_masa_sm90_build'
CONFIG = 'configs/masa_sm90_build_v1.json'
CONFIG_PIN = dict(bytes=20926, sha256='fdc0134a5037bf446f8f7c755c4252749b4ffba492dda93a43932d14674f252b')
BUDGET, COMPILE_BUDGET, GRACE = 2400, 1800, 30
BASE = 'sha256:5b4cda06057d53e3e1fdd5408e5b14aa2b702318df2b50252558d098300a5ec6'
VENV = '/opt/world-reward-masa'
TOOLKIT = '/opt/world-reward-cuda-11.8'
LABEL = 'world_reward_masa_sm90_owner'
HELPER_PINS = {
    'infra/masa_runtime_build.py': dict(bytes=26038, sha256='0ba354792137f808c86a2eb0eb1c8541af2912d2c4a249cdac9b0dad8212cd12'),
    'infra/mediapipe_cpu_runtime_verify.py': dict(bytes=23559, sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
    'infra/mediapipe_hands_acquire.py': dict(bytes=21381, sha256='8293510c02777cd5b844865285736ff1a653631a8803849a9047ee9030196570')}
HELPERS = ('infra/masa_sm90_build.py', 'infra/run_masa_sm90_build.sh', CONFIG,
           'configs/masa_runtime_v1.json', *HELPER_PINS)
ENV = dict(PATH='/usr/bin:/bin', HOME='/nonexistent', LANG='C.UTF-8',
           DOCKER_HOST='unix://' + str(ROOT / 'docker.sock'), DOCKER_BUILDKIT='0')
CLASSES = {x: x.__name__ for x in (ValueError, RuntimeError, OSError, TimeoutError,
           FileNotFoundError, PermissionError, KeyError, TypeError, AssertionError,
           subprocess.TimeoutExpired)}


def require(value, message='SM90 gate failed'):
    if not value:
        raise ValueError(message)


def load_helpers(code):
    result = {}
    for name, pin in HELPER_PINS.items():
        p = code / name
        s = p.lstat()
        require(p.resolve() == p and not any(x.is_symlink() for x in (p, *p.parents))
                and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and not s.st_mode & 0o222
                and s.st_size == pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == pin['sha256'])
        after = p.lstat()
        require(all(getattr(s, k) == getattr(after, k) for k in
                    ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')))
        spec = importlib.util.spec_from_file_location('wr_sm90_' + p.stem, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        require(Path(m.__file__) == p)
        result[name] = m
    return result['infra/mediapipe_cpu_runtime_verify.py'], result['infra/masa_runtime_build.py']


def current_source(rt, code, revision):
    require(Path(__file__).resolve() == code / HELPERS[0]
            and set(x.name for x in code.parent.iterdir()) == {'code', 'revision', 'source-sha256'})
    return rt.source(ROOT, code, revision, ENTRY, HELPERS)


def policy(rt, code):
    c = rt.pinned(code / CONFIG, CONFIG_PIN, 30000)
    require(c['schema'] == 'world_reward.masa_sm90_build.v1'
            and c['prior_runtime']['image_id'] == BASE
            and c['limits']['root_budget_seconds_including_acquisition_build_and_publication'] == BUDGET
            and c['limits']['cpu_compile_budget_seconds_within_root_budget'] == COMPILE_BUDGET
            and c['limits']['cpus'] == 32 and c['limits']['memory_bytes'] == 128 << 30
            and c['compiler']['torch_cuda_arch_list'] == '9.0'
            and c['compiler']['environment']['LD_LIBRARY_PATH'] == ''
            and c['limits']['maximum_download_bytes'] == 500000000)
    header_metadata(c)
    return c


def header_metadata(c):
    """Frozen publisher package records, never resolving an evolving latest index."""
    records = c['header_package_metadata']['records']
    assets = c['header_assets']
    require(len(records) == len(assets) == 3)
    expected = {'libcublas-dev-11-8': '11.11.3.6-1', 'libcusolver-dev-11-8': '11.4.1.48-1',
                'libcusparse-dev-11-8': '11.7.5.86-1'}
    require({r['package'] for r in records} == {r['package'] for r in assets} == set(expected))
    for record in records:
        raw = record['publisher_stanza'].encode()
        require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'])
        fields = {}
        for line in record['publisher_stanza'].splitlines():
            if ': ' in line and not line.startswith(' '):
                k, v = line.split(': ', 1)
                require(k not in fields)
                fields[k] = v
        asset = next(r for r in assets if r['package'] == record['package'])
        require(fields['Package'] == asset['package'] and fields['Version'] == expected[asset['package']]
                and fields['Version'] == asset['version'] and fields['Architecture'] == 'amd64'
                and fields['Filename'] == './' + asset['file'] and int(fields['Size']) == asset['bytes']
                and fields['SHA256'] == asset['sha256']
                and asset['url'] == c['header_package_metadata']['url'].removesuffix('Packages.gz') + asset['file'])
        endpoint(asset['url'])


def authenticate(rt, mb, code, c, args):
    p = c['prior_runtime']
    require(args.image_id == p['image_id'] and args.runtime_revision == p['producer_revision']
            and args.runtime_report_bytes == p['report']['bytes']
            and args.runtime_report_sha256 == p['report']['sha256'])
    path = ROOT / p['report_relative_to_remote_root']
    r = rt.pinned(path, p['report'], 2 << 20)
    facts = dict(schema='world_reward.masa_runtime_build.v1', stage='masa_author_isolated_runtime_build',
                 producer_revision=p['producer_revision'], status='pass', phase='complete',
                 gpu_used=False, operator_qualified=False, model_constructed=False,
                 model_or_dataset_read=False, quality_verified=False, adopted=False,
                 license_eligibility_verified=False, source_rehashed_after=True,
                 artifacts_rehashed_after=True, base_rechecked_after=True,
                 owned_containers_removed=True, owned_partial_cleanup=True, system_site_packages=False)
    require(all(type(r.get(k)) is type(v) and r[k] == v for k, v in facts.items()))
    require(r['child_image']['image_id'] == BASE
            and r['target_tag'] == 'world-reward/masa-author-runtime:' + p['producer_revision'])
    old = ROOT / 'jobs' / p['producer_revision'] / 'run_masa_runtime_build' / 'code'
    proof = r['source_binding']
    require(rt.source(ROOT, old, p['producer_revision'], 'run_masa_runtime_build', tuple(proof['helpers'])) == proof)
    rc = rt.pinned(old / 'configs/masa_runtime_v1.json', r['config_identity'], 100000)
    require(rc == rt.strict((code / 'configs/masa_runtime_v1.json').read_bytes())
            and rt.identity(old / 'infra/masa_runtime_build.py') == HELPER_PINS['infra/masa_runtime_build.py'])
    versions = {x['name']: x['version'] for x in rc['wheels']}
    require(len(versions) == 52 and versions['torch'] == '2.1.2+cu118' and versions['mmcv'] == '2.1.0'
            and r['cpu_import'] == dict(versions=versions, python='3.11', isolated_venv=True,
                cuda_initialized=False, cuda_build='11.8', extension_imported=True,
                operator_executed=False, model_constructed=False))
    frozen = {path: p['report']}
    for row in rc['wheels']:
        pin = {k: r['wheels'][row['name']]['identity'][k] for k in ('bytes', 'sha256')}
        require(pin['bytes'] == row['bytes'] and (row['sha256'] is None or pin['sha256'] == row['sha256']))
        artifact = path.parent / 'wheels' / row['filename']
        require(rt.identity(artifact, 2500000000) == pin)
        frozen[artifact] = pin
    for row in rc['publisher_notices']:
        pin = {k: row[k] for k in ('bytes', 'sha256')}
        artifact = path.parent / 'notices' / row['file']
        require(rt.identity(artifact, 500000) == pin)
        frozen[artifact] = pin
    return dict(versions=versions, frozen=frozen, source=proof, old_code=old,
                image=r['child_image'], report_pin=p['report'])


def endpoint(url):
    p = urllib.parse.urlsplit(url)
    hosts = {'developer.download.nvidia.com', 'files.pythonhosted.org', 'pypi.org',
             'raw.githubusercontent.com', 'docs.nvidia.com', 'nvlabs.github.io', 'api.github.com', 'codeload.github.com'}
    require(p.scheme == 'https' and p.hostname in hosts and p.username is None and p.password is None
            and p.port in (None, 443) and not p.fragment
            and (not p.query or p.hostname == 'api.github.com' and p.query == 'recursive=1'))
    return url


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_):
        raise ValueError('Publisher redirect forbidden')


def fetch(rt, row, path, opener, deadline, partials, *, maximum=None):
    endpoint(row['url'])
    limit = row['bytes'] if maximum is None else maximum
    part = path.with_name(path.name + '.part')
    digest = hashlib.sha256()
    count = 0
    require(time.monotonic() < deadline)
    request = urllib.request.Request(row['url'], headers={'Accept-Encoding': 'identity', 'User-Agent': 'World-Reward-reproducibility'})
    with opener.open(request, timeout=min(30, max(.001, deadline - time.monotonic()))) as response:
        require(response.status == 200 and response.geturl() == row['url']
                and response.headers.get('Content-Encoding', 'identity').lower() == 'identity')
        length = response.headers.get('Content-Length')
        require(length is None or re.fullmatch('[0-9]+', length) and int(length) <= limit)
        with part.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            s = part.lstat()
            partials.append((part, (s.st_dev, s.st_ino, s.st_uid)))
            while True:
                require(time.monotonic() < deadline)
                b = response.read(min(1 << 20, limit - count + 1))
                require(count + len(b) <= limit)
                if not b:
                    break
                f.write(b)
                digest.update(b)
                count += len(b)
            require(count > 0 and (maximum is not None or
                    count == row['bytes'] and digest.hexdigest() == row['sha256']))
            f.flush()
            os.fsync(f.fileno())
            os.fchmod(f.fileno(), 0o444)
        # Same-owned directory, atomic no-overwrite publication.
        os.link(part, path, follow_symlinks=False)
        part.unlink()
    return dict(bytes=count, sha256=digest.hexdigest())


def git_hash(kind, raw):
    return hashlib.sha1(kind.encode() + b' ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def git_root(rows):
    """Reconstruct Git trees (Git directory sort), not a flat file-list hash."""
    tree = {}
    for row in rows:
        parts = PurePosixPath(row['path']).parts
        require(parts and row['path'] == '/'.join(parts) and '..' not in parts and not row['path'].startswith('/'))
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            require(type(node) is dict)
        require(parts[-1] not in node)
        node[parts[-1]] = (row['mode'], row['sha'])
    def emit(node):
        raw = bytearray()
        for name in sorted(node, key=lambda k: k.encode() + (b'/' if type(node[k]) is dict else b'')):
            value = node[name]
            mode, digest = ('40000', emit(value)) if type(value) is dict else value
            raw += mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(digest)
        return git_hash('tree', bytes(raw))
    return emit(tree)


def tree_rows(rt, raw, c):
    j = rt.strict(raw)
    require(j['sha'] == c['source']['root_tree'] and j['truncated'] is False)
    rows = [r for r in j['tree'] if r['type'] == 'blob']
    require(len(rows) == c['source']['blob_count_including_symlink']
            and len({r['path'] for r in rows}) == len(rows)
            and sum(r['size'] for r in rows) == c['source']['expanded_blob_bytes_including_symlink_content']
            and all(r['mode'] in ('100644', '100755', '120000') and re.fullmatch('[0-9a-f]{40}', r['sha']) for r in rows)
            and all(r['type'] in ('tree', 'blob') for r in j['tree'])
            and git_root(rows) == c['source']['root_tree'])
    return rows


def safe_tar(path, destination, *, prefix, maximum, member_limit=20000):
    """Two-phase extraction: regular files before authenticated internal links."""
    destination.mkdir(mode=0o700)
    records = {}
    with tarfile.open(path, 'r:*') as archive:
        members = archive.getmembers()
        require(len(members) <= member_limit)
        total = 0
        for m in members:
            p = PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.name
                    and p.as_posix().rstrip('/') == m.name.rstrip('/') and p.parts[0] == prefix)
            rel = '/'.join(p.parts[1:])
            if not rel:
                require(m.isdir())
                continue
            require(rel not in records and not m.mode & 0o6000 and (m.isdir() or m.isfile() or m.issym()))
            total += m.size if m.isfile() else len(m.linkname.encode()) if m.issym() else 0
            require(total <= maximum)
            if m.issym():
                require(not m.linkname.startswith('/') and '\\' not in m.linkname)
                target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), m.linkname))
                require(target != '..' and not target.startswith('../'))
            records[rel] = m
        links = {r for r, m in records.items() if m.issym()}
        for rel, m in records.items():
            require(not any('/'.join(PurePosixPath(rel).parts[:i]) in links for i in range(1, len(PurePosixPath(rel).parts))))
            p = destination / rel
            p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if m.isdir():
                p.mkdir(mode=0o700, exist_ok=True)
            elif m.isfile():
                with archive.extractfile(m) as source, p.open('xb') as out:
                    shutil.copyfileobj(source, out, 1 << 20)
                require(p.stat().st_size == m.size)
                p.chmod(0o555 if m.mode & 0o111 else 0o444)
        for rel in links:
            (destination / rel).symlink_to(records[rel].linkname)
    return records


def source_inventory(root, rows, c):
    expected = {r['path']: r for r in rows}
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if not p.is_dir() or p.is_symlink()}
    require(set(actual) == set(expected))
    for name, p in actual.items():
        row = expected[name]
        s = p.lstat()
        raw = os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
        require(len(raw) == row['size'] and git_hash('blob', raw) == row['sha'])
        require((p.is_symlink() and row['mode'] == '120000') or
                stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and not s.st_mode & 0o222
                and row['mode'] == ('100755' if s.st_mode & 0o111 else '100644'))
    link = c['source']['symlink_policy']
    require([r['path'] for r in rows if r['mode'] == '120000'] == [link['file']]
            and os.readlink(root / link['file']) == link['target'])
    for row in c['source']['required_file_pins']:
        raw = (root / row['file']).read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'])
    return dict(root_tree=git_root(rows), blobs=len(rows), expanded_bytes=sum(r['size'] for r in rows))


def toolkit_merge(components, destination):
    destination.mkdir(mode=0o700)
    for root in components:
        for top in ('bin', 'include', 'lib', 'lib64', 'nvvm'):
            if not (root / top).exists():
                continue
            for p in [root / top, *sorted((root / top).rglob('*'))]:
                target = destination / p.relative_to(root)
                if p.is_dir() and not p.is_symlink():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                elif p.is_symlink():
                    if target.is_symlink():
                        require(os.readlink(target) == os.readlink(p))
                    else:
                        require(not target.exists())
                        target.symlink_to(os.readlink(p))
                elif target.exists():
                    require(not target.is_symlink() and target.read_bytes() == p.read_bytes())
                else:
                    shutil.copyfile(p, target, follow_symlinks=False)
                    target.chmod(stat.S_IMODE(p.stat().st_mode))
    for p in destination.rglob('*'):
        if p.is_symlink():
            require(p.resolve().is_relative_to(destination) and p.resolve().is_file())


class LimitedReader:
    """Expose exactly one authenticated ar member without unpacking huge data."""
    def __init__(self, stream, start, size):
        self.stream, self.remaining = stream, size
        stream.seek(start)
    def read(self, count=-1):
        count = self.remaining if count < 0 else min(count, self.remaining)
        raw = self.stream.read(count)
        self.remaining -= len(raw)
        return raw


def deb_members(path):
    """Three Debian members plus optional final bounded publisher signature."""
    members = []
    with path.open('rb') as f:
        require(f.read(8) == b'!<arch>\n')
        while f.tell() < path.stat().st_size:
            raw = f.read(60)
            require(len(raw) == 60 and raw[58:60] == b'`\n')
            name = raw[:16].decode('ascii').strip()
            name = name.removesuffix('/')
            require(name and '/' not in name and not name.startswith(('#', '//')))
            require(re.fullmatch(b'[0-9]+ *', raw[48:58]))
            size = int(raw[48:58])
            start = f.tell()
            require(0 <= size <= path.stat().st_size - start)
            members.append(dict(name=name, offset=start, bytes=size))
            f.seek(size, os.SEEK_CUR)
            if size % 2:
                require(f.read(1) == b'\n')
        require(f.tell() == path.stat().st_size and len(members) in (3, 4))
        require(members[0]['name'] == 'debian-binary' and members[0]['bytes'] == 4
                and members[1]['name'] in ('control.tar.xz', 'control.tar.gz')
                and members[2]['name'] in ('data.tar.xz', 'data.tar.gz'))
        if len(members) == 4:
            signature = members[3]
            require(signature['name'] == '_gpgbuilder' and 0 < signature['bytes'] <= 4096)
            f.seek(signature['offset'])
            raw = f.read(signature['bytes'])
            require(len(raw) == signature['bytes'])
            signature['sha256'] = hashlib.sha256(raw).hexdigest()
        f.seek(members[0]['offset'])
        require(f.read(4) == b'2.0\n')
    return members


def deb_headers(path, destination, c):
    """Full hash is checked by acquisition; stream data, retain headers/notices only."""
    members = deb_members(path)
    data = members[2]
    destination.mkdir(mode=0o700)
    header_prefix = c['header_extraction']['include_prefix']
    rows, names, expanded, selected, header_count, notice_count = [], set(), 0, 0, 0, 0
    with path.open('rb') as source:
        reader = LimitedReader(source, data['offset'], data['bytes'])
        with tarfile.open(fileobj=reader, mode='r|*') as t:
            for m in t:
                name = m.name.removeprefix('./').rstrip('/')
                if name in ('', '.'):
                    require(m.isdir() and m.name in ('.', './') and not m.mode & 0o6000)
                    continue
                p = PurePosixPath(name)
                require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
                        and name == p.as_posix() and name not in names and len(names) < 20000
                        and not m.mode & 0o6000 and (m.isdir() or m.isfile() or m.issym()))
                names.add(name)
                expanded += m.size if m.isfile() else 0
                require(expanded <= c['limits']['maximum_DEB_data_expanded_bytes_per_package'])
                keep_header = name.startswith(header_prefix) and m.isfile()
                keep_notice = name.startswith('usr/share/doc/') and m.isfile() and any(
                    k in p.name.upper() for k in ('COPYRIGHT', 'LICENSE', 'NOTICE', 'COPYING'))
                if name.startswith(header_prefix):
                    require(m.isdir() or m.isfile())
                rows.append(dict(file=name, bytes=m.size, mode=m.mode,
                                 type='file' if m.isfile() else 'dir' if m.isdir() else 'symlink',
                                 link=m.linkname if m.issym() else None))
                if keep_header or keep_notice:
                    require(m.isfile() and m.size <= 4000000)
                    require(not any(token in p.name.lower() for token in ('.so', '.dll', '.dylib'))
                            and not p.name.lower().endswith('.a'))
                    selected += m.size
                    require(selected <= c['limits']['maximum_DEB_selected_header_and_notice_bytes_total'])
                    relative = 'include/' + name.removeprefix(header_prefix) if keep_header else 'notices/' + name
                    target = destination / relative
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with t.extractfile(m) as f, target.open('xb') as out:
                        shutil.copyfileobj(f, out, 1 << 20)
                    require(target.stat().st_size == m.size)
                    target.chmod(0o444)
                    header_count += int(keep_header)
                    notice_count += int(keep_notice)
    require(header_count > 0 and notice_count > 0)
    return dict(ar_members=members, data_members=len(rows), data_expanded_bytes=expanded,
                selected_header_and_notice_bytes=selected, headers=header_count, notices=notice_count,
                signature_verification_claimed=False, signature_executed_or_extracted=False,
                data_inventory_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                selected_inventory=inventory(destination), shared_or_static_libraries_installed=False)


def ninja_binary(path, destination):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and len(names) < 1000)
        for m in z.infolist():
            p = PurePosixPath(m.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.filename
                    and not stat.S_ISLNK(m.external_attr >> 16) and m.file_size < 5000000)
        executable = [n for n in names if n.endswith('/bin/ninja')]
        notices = [n for n in names if 'LICENSE' in PurePosixPath(n).name.upper()]
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
        require(len(executable) == len(metadata) == 1 and notices)
        m = email.message_from_bytes(z.read(metadata[0]))
        require(m['Name'] == 'ninja' and m['Version'] == '1.11.1.1')
        destination.mkdir(mode=0o700)
        (destination / 'ninja').write_bytes(z.read(executable[0]))
        (destination / 'ninja').chmod(0o555)
        return {n: dict(bytes=len(z.read(n)), sha256=hashlib.sha256(z.read(n)).hexdigest()) for n in notices}


def notices(root):
    result = {}
    for p in root.rglob('*'):
        if p.is_file() and not p.is_symlink() and any(k in p.name.upper() for k in ('LICENSE', 'NOTICE', 'COPYING', 'COPYRIGHT')):
            require(p.stat().st_size <= 2000000)
            raw = p.read_bytes()
            result[p.relative_to(root).as_posix()] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return result


def inventory(root):
    """Small digest of authenticated expanded material, with actual counts."""
    rows = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            rows[p.relative_to(root).as_posix()] = dict(link=os.readlink(p))
        elif p.is_file():
            raw = p.read_bytes()
            rows[p.relative_to(root).as_posix()] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return dict(members=len(rows), bytes=sum(r.get('bytes', 0) for r in rows.values()),
                sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def owned_remove_tree(path, uid):
    """Delete only single-link owned material under the exclusive result leaf."""
    require(path.is_dir() and not path.is_symlink())
    for p in sorted(path.rglob('*'), key=lambda x: len(x.parts), reverse=True):
        s = p.lstat()
        require(s.st_uid == uid)
        if stat.S_ISDIR(s.st_mode):
            p.chmod(0o700)
            p.rmdir()
        else:
            require(stat.S_ISLNK(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1)
            p.unlink()
    require(path.lstat().st_uid == uid)
    path.chmod(0o700)
    path.rmdir()


def command(args, deadline, *, log=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Inclusive SM90 deadline')
    if log is None:
        r = subprocess.run(args, env=ENV, capture_output=True, timeout=min(30, remaining))
        require(len(r.stdout) + len(r.stderr) < 200000)
        return r
    # CPU build output is disposable; never include arbitrary compiler text in receipts.
    with log.open('xb') as f:
        os.fchmod(f.fileno(), 0o400)
        r = subprocess.run(args, env=ENV, stdout=f, stderr=subprocess.STDOUT, timeout=remaining)
    require(log.stat().st_size <= 16000000)
    return r


def image(rt, name, deadline, *, absent=False, owner=None):
    r = command(['docker', 'image', 'inspect', name, '--format', '{{json .}}'], deadline)
    if absent:
        require(r.returncode == 1 and r.stdout.strip() in (b'', b'[]') and r.stderr.strip() in tuple(
                (p + name).encode() for p in ('Error: No such image: ', 'error: no such image: ',
                                             'Error response from daemon: No such image: ')))
        return None
    require(r.returncode == 0)
    j = rt.strict(r.stdout)
    require(j['Architecture'] == 'amd64' and j['Os'] == 'linux'
            and re.fullmatch('sha256:[0-9a-f]{64}', j['Id']) and j['RootFS']['Type'] == 'layers')
    if owner is not None:
        require((j['Config'].get('Labels') or {}).get(LABEL) == owner)
    return dict(image_id=j['Id'], layers=j['RootFS']['Layers'])


def absent_container(name, deadline):
    r = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/' + name + '$'], deadline)
    require(r.returncode == 0 and not r.stdout.strip() and not r.stderr.strip())


def remove_container(rt, cid, name, revision, deadline):
    if not cid.exists():
        absent_container(name, deadline)
        return
    pin = rt.identity(cid, 100, readonly=False)
    identity = cid.read_text().strip()
    require(re.fullmatch('[0-9a-f]{64}', identity))
    r = command(['docker', 'inspect', '--type', 'container', '--format', '{{json .}}', identity], deadline)
    if r.returncode == 0:
        j = rt.strict(r.stdout)
        require(j['Id'] == identity and j['Name'] == '/' + name and
                (j['Config'].get('Labels') or {}).get(LABEL) == revision)
        require(command(['docker', 'rm', '-f', identity], deadline).returncode == 0)
    else:
        require(r.returncode == 1 and r.stdout.strip() in (b'', b'[]') and r.stderr.strip() in tuple(
                (p + identity).encode() for p in ('Error: No such object: ', 'error: no such object: ',
                'Error: No such container: ', 'error: no such container: ', 'Error response from daemon: No such container: ')))
    require(rt.identity(cid, 100, readonly=False) == pin)
    absent_container(name, deadline)
    cid.chmod(0o444)


DIST_PROBE = r'''import hashlib,importlib.metadata as m,json,os,pathlib,sys,torch
assert sys.prefix=='/opt/world-reward-masa' and sys.version_info[:2]==(3,11)
assert 'include-system-site-packages = false' in open(sys.prefix+'/pyvenv.cfg').read().lower()
assert not any('site-packages' in p and not p.startswith(sys.prefix+'/') for p in sys.path)
expected=EXPECTED_VERSIONS
assert {n:m.version(n) for n in expected}==expected
inventory={}
for name in sorted(expected):
 if name=='mmcv':continue
 d=m.distribution(name); files={}
 for relative in d.files or []:
  p=pathlib.Path(d.locate_file(relative));assert p.resolve().is_relative_to(pathlib.Path(sys.prefix))
  if p.suffix=='.pyc':continue
  assert p.is_file();b=p.read_bytes();files[str(relative)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 assert files;inventory[name]=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
from mmcv.ops import ModulatedDeformConv2d,RoIAlign
import mmcv._ext as ext
extension_sha256=hashlib.sha256(pathlib.Path(ext.__file__).read_bytes()).hexdigest()
assert torch.version.cuda=='11.8' and not torch.cuda.is_initialized()
print(json.dumps({'versions':expected,'non_mmcv_distribution_file_digests':inventory,'mmcv_extension_sha256':extension_sha256,'python':'3.11','isolated_venv':True,'cuda_build':'11.8','cuda_initialized':False,'operator_executed':False,'model_constructed':False}))
'''


def cpu(rt, image_id, out, revision, label, versions, deadline):
    name = 'wr-masa-sm90-' + revision + '-' + label
    cid, log = out / (label + '.cid'), out / (label + '.log')
    absent_container(name, deadline)
    script = DIST_PROBE.replace('EXPECTED_VERSIONS', repr(versions))
    r = command(['docker', 'run', '--runtime', 'runc', '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--rm', '--cidfile', str(cid), '--name', name,
        '--label', LABEL + '=' + revision, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--memory', '8g', '--cpus', '4',
        '--tmpfs', '/tmp:rw,nosuid,size=512m', '--entrypoint', '/usr/bin/env', image_id, '-i',
        'PATH=' + VENV + '/bin:/usr/bin:/bin', 'HOME=/tmp', 'LANG=C.UTF-8', 'LD_LIBRARY_PATH=',
        'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'MPLBACKEND=Agg',
        VENV + '/bin/python', '-I', '-B', '-c', script], deadline, log=log)
    require(r.returncode == 0 and log.stat().st_size < 100000)
    j = rt.strict(log.read_bytes().splitlines()[-1])
    require(j['versions'] == versions and set(j['non_mmcv_distribution_file_digests']) == set(versions) - {'mmcv'}
            and j['python'] == '3.11' and j['isolated_venv'] is True and j['cuda_build'] == '11.8'
            and j['cuda_initialized'] is False and j['operator_executed'] is False and j['model_constructed'] is False)
    return j


COMPILE = r'''import hashlib,json,os,pathlib,re,shutil,subprocess,time,torch
source=pathlib.Path('/source');work=pathlib.Path('/work');toolkit=pathlib.Path('/opt/world-reward-cuda-11.8')
def step(name):
 print(json.dumps({'native_compile_step':name}),flush=True)
def call(args):
 r=subprocess.run(args,capture_output=True);assert r.returncode==0
 return r.stdout.decode()
step('toolchain_probe')
assert call(['/usr/bin/gcc','-dumpfullversion']).strip()=='11.4.0'
assert call(['/usr/bin/g++','-dumpfullversion']).strip()=='11.4.0'
assert pathlib.Path('/opt/conda/include/python3.11/Python.h').is_file()
assert 'V11.8.89' in call([str(toolkit/'bin/nvcc'),'--version'])
assert re.fullmatch(r'1\.11\.1(?:[.][A-Za-z0-9_.-]+)?',call(['/ninja/ninja','--version']).strip())
test=work/'probe.cu';test.write_text('#include <ATen/cuda/CUDAContext.h>\n#include <Python.h>\n__global__ void wr_sm90_probe() {}\n')
step('sm90_probe')
from torch.utils.cpp_extension import include_paths
assert torch.version.cuda=='11.8' and not torch.cuda.is_initialized()
includes=include_paths(cuda=True)+['/opt/conda/include/python3.11']
flags=['-I'+p for p in includes]
flags+=['-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
call([str(toolkit/'bin/nvcc'),'-arch=sm_90','-std=c++17',*flags,'-c',str(test),'-o',str(work/'probe.o')])
shutil.copytree(source,work/'source',symlinks=True)
os.chdir(work/'source')
step('full_original_build')
with open(work/'compiler-output.log','xb') as log:
 r=subprocess.run(['/opt/world-reward-masa/bin/python','-I','-m','pip','--isolated','wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--disable-pip-version-check','--wheel-dir','/work/wheels','.'],stdout=log,stderr=subprocess.STDOUT)
assert r.returncode==0 and (work/'compiler-output.log').stat().st_size<=16000000
wheels=list((work/'wheels').glob('*.whl'));assert len(wheels)==1
step('source_posthash')
for p in source.rglob('*'):
 rel=p.relative_to(source);q=work/'source'/rel
 if p.is_symlink():assert q.is_symlink() and os.readlink(q)==os.readlink(p)
 elif p.is_file():assert q.is_file() and not q.is_symlink() and p.read_bytes()==q.read_bytes()
print(json.dumps({'compiler':'11.8.89','gcc':'11.4.0','gxx':'11.4.0','ninja':'1.11.1','arch':'sm_90','tiny_object_compiled':True,'ATen_CUDAContext_and_Python_headers_compiled':True,'full_original_source_posthash':True,'gpu_used':False,'wheel_file':wheels[0].name}))
'''


def compile_full(rt, out, revision, c, deadline):
    work = out / 'work'
    work.mkdir(mode=0o700)
    name = 'wr-masa-sm90-' + revision + '-compile'
    absent_container(name, deadline)
    env = c['compiler']['environment']
    args = ['docker', 'run', '--runtime', 'runc', '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--rm', '--cidfile', str(out / 'compile.cid'), '--name', name,
            '--label', LABEL + '=' + revision, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--memory', '128g', '--cpus', '32',
            '--tmpfs', '/tmp:rw,exec,nosuid,size=2g', '--mount', 'type=bind,src=' + str(out / 'source') + ',dst=/source,readonly',
            '--mount', 'type=bind,src=' + str(out / 'toolkit') + ',dst=' + TOOLKIT + ',readonly',
            '--mount', 'type=bind,src=' + str(out / 'ninja') + ',dst=/ninja,readonly',
            '--mount', 'type=bind,src=' + str(work) + ',dst=/work', '--entrypoint', '/usr/bin/env', BASE, '-i',
            'PATH=/ninja:' + TOOLKIT + '/bin:' + VENV + '/bin:/usr/bin:/bin', 'HOME=/tmp', 'LANG=C.UTF-8',
            'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1']
    args += [k + '=' + v for k, v in env.items()]
    args += [VENV + '/bin/python', '-I', '-B', '-c', COMPILE]
    r = command(args, min(deadline, time.monotonic() + COMPILE_BUDGET), log=out / 'compile.log')
    require(r.returncode == 0 and (out / 'compile.log').stat().st_size < 100000)
    proof = rt.strict((out / 'compile.log').read_bytes().splitlines()[-1])
    require(all(proof[k] == v for k, v in dict(compiler='11.8.89', gcc='11.4.0', gxx='11.4.0',
            ninja='1.11.1', arch='sm_90', tiny_object_compiled=True, ATen_CUDAContext_and_Python_headers_compiled=True,
            full_original_source_posthash=True, gpu_used=False).items()))
    wheels = list((work / 'wheels').glob('*.whl'))
    require(len(wheels) == 1 and wheels[0].name == proof['wheel_file'])
    destination = out / wheels[0].name
    shutil.copyfile(wheels[0], destination)
    destination.chmod(0o444)
    return proof, destination


def compile_diagnostic(rt, out):
    """Only fixed native steps and private Azure log identity, never raw errors."""
    result = {}
    p = out / 'compile.log'
    if p.exists() and not p.is_symlink() and p.stat().st_size <= 100000:
        for raw in p.read_bytes().splitlines():
            try:
                row = rt.strict(raw)
                if set(row) == {'native_compile_step'} and row['native_compile_step'] in (
                        'toolchain_probe', 'sm90_probe', 'full_original_build', 'source_posthash'):
                    result['native_compile_step'] = row['native_compile_step']
            except (ValueError, TypeError, AttributeError, KeyError):
                pass
    log = out / 'work/compiler-output.log'
    if log.exists():
        s = log.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and not log.is_symlink()
                and s.st_uid == out.stat().st_uid and s.st_size <= 16000000)
        if s.st_size:
            result['private_compiler_output_identity'] = dict(bytes=s.st_size, sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    return result


def compiled_wheel(path):
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        require(len(members) < 20000 and len({m.filename for m in members}) == len(members)
                and sum(m.file_size for m in members) < 2000000000)
        for m in members:
            p = PurePosixPath(m.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.filename
                    and not stat.S_ISLNK(m.external_attr >> 16) and not m.flag_bits & 1)
        metadata = [m for m in members if m.filename.endswith('.dist-info/METADATA')]
        extension = [m for m in members if re.fullmatch(r'mmcv/_ext\.cpython-311-x86_64-linux-gnu\.so', m.filename)]
        require(len(metadata) == len(extension) == 1 and metadata[0].file_size < 300000)
        m = email.message_from_bytes(z.read(metadata[0]))
        require(m['Name'] == 'mmcv' and m['Version'] == '2.1.0')
        return dict(members=len(members), expanded_bytes=sum(x.file_size for x in members),
                    extension_file=extension[0].filename, extension_bytes=extension[0].file_size,
                    extension_sha256=hashlib.sha256(z.read(extension[0])).hexdigest())


def recipe(wheel):
    require(re.fullmatch(r'mmcv-2\.1\.0-cp311-cp311-linux_x86_64\.whl', wheel.name))
    return ('FROM ' + BASE + '\nCOPY ' + wheel.name + ' /opt/wr-mmcv/' + wheel.name + '\n'
            'COPY notices /opt/world-reward-mmcv-sm90-notices\n'
            'ENV NVIDIA_VISIBLE_DEVICES=void\nENV CUDA_VISIBLE_DEVICES=""\n'
            'RUN /usr/bin/env -i PATH=' + VENV + '/bin:/usr/bin:/bin HOME=/tmp LD_LIBRARY_PATH="" '
            + VENV + '/bin/python -I -m pip --isolated install --no-index --no-deps --no-cache-dir '
            '--disable-pip-version-check --force-reinstall /opt/wr-mmcv/' + wheel.name + ' && '
            + VENV + '/bin/python -I -m pip check && rm -rf /opt/wr-mmcv\n'
            'ENV LD_LIBRARY_PATH=""\n').encode()


def seal_report(rt, out, report, start):
    report['elapsed_seconds'] = time.monotonic() - start
    if report['elapsed_seconds'] >= BUDGET:
        report.update(status='fail', failure_gate='root_deadline', failure_class='TimeoutError')
    p = out / 'report.json'
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
        out.chmod(0o555)
        elapsed = time.monotonic() - start
        if elapsed >= BUDGET and report['status'] == 'pass':
            report.update(status='fail', elapsed_seconds=elapsed, failure_gate='root_deadline', failure_class='TimeoutError')
            stream.seek(0)
            stream.truncate()
            stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())


def run(code, args, *, opener=None, started=None):
    start = time.monotonic() if started is None else started
    require(type(start) in (int, float) and math.isfinite(start) and 0 < start <= time.monotonic() < start + BUDGET)
    deadline = start + BUDGET
    rt, mb = load_helpers(code)
    binding = current_source(rt, code, args.revision)
    c = policy(rt, code)
    proof = authenticate(rt, mb, code, c, args)
    out = rt.canonical(ROOT / 'results' / ('masa-sm90-build-' + args.revision))
    require(not out.exists() and out.parent.is_dir())
    out.mkdir(mode=0o700)
    owner = (out.stat().st_dev, out.stat().st_ino, out.stat().st_uid)
    target = 'world-reward/masa-sm90-runtime:' + args.revision
    partials, assets = [], {}
    tree = None
    child = None
    gate = 'preflight'
    report = dict(schema='world_reward.masa_sm90_build_receipt.v1', stage='masa_full_original_mmcv_sm90_cpu_build',
                  status='fail', producer_revision=args.revision, source_binding=binding, config_identity=CONFIG_PIN,
                  prior_runtime_revision=args.runtime_revision, prior_runtime_report=proof['report_pin'], prior_image=proof['image'],
                  gpu_used=False, model_constructed=False, checkpoint_read=False, dataset_read=False,
                  operator_qualified=False, quality_verified=False, adopted=False, license_eligibility_verified=False,
                  budget_seconds=BUDGET, compile_budget_seconds=COMPILE_BUDGET)
    try:
        require(shutil.disk_usage(out).free >= c['limits']['minimum_free_bytes'])
        require(image(rt, BASE, deadline) == proof['image'])
        image(rt, target, deadline, absent=True)
        report['base_cpu_import'] = cpu(rt, BASE, out, args.revision, 'base', proof['versions'], deadline)
        gate = 'acquisition'
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        downloads, legal = out / 'downloads', out / 'notices'
        downloads.mkdir(mode=0o700)
        legal.mkdir(mode=0o700)
        rows = c['publisher_metadata'] + [c['source']['recursive_tree_metadata']]
        rows += [dict(file='CUDA-11.8-EULA.html', **{k: c['license']['cuda_eula'][k] for k in ('url', 'bytes', 'sha256')}),
                 dict(file='NVIDIA-NC-StyleGAN2.html', **{k: c['license']['mmcv']['research_only_stylegan2_notice'][k]
                      for k in ('url', 'bytes', 'sha256')})]
        metadata = {}
        for i, row in enumerate(rows):
            p = legal / row['file'] if row.get('file', '').endswith('.html') else downloads / row.get('file', 'mmcv-tree.json')
            assets[p] = fetch(rt, row, p, opener, deadline, partials)
            metadata[i] = p
        manifest = rt.strict(metadata[0].read_bytes())
        for row in c['assets'] + c['header_assets']:
            if row.get('component', '').startswith('cuda_'):
                entry = manifest[row['component']]['linux-x86_64']
                require(entry['sha256'] == row['sha256'] and int(entry['size']) == row['bytes']
                        and row['url'].endswith('/' + entry['relative_path']))
            p = downloads / row['file']
            assets[p] = fetch(rt, row, p, opener, deadline, partials)
        ninja_row = next(r for r in c['assets'] if r['component'] == 'ninja')
        release = rt.strict(metadata[1].read_bytes())
        candidates = [r for r in release['urls'] if r['filename'] == ninja_row['file']]
        require(len(candidates) == 1 and candidates[0]['digests']['sha256'] == ninja_row['sha256']
                and candidates[0]['size'] == ninja_row['bytes'] and candidates[0]['url'] == ninja_row['url'])
        tree = tree_rows(rt, metadata[2].read_bytes(), c)
        archive = downloads / 'mmcv-full-source.tar.gz'
        assets[archive] = fetch(rt, dict(url=c['source']['archive_url']), archive, opener, deadline, partials, maximum=20000000)
        require(sum(r['bytes'] for r in assets.values()) <= c['limits']['maximum_download_bytes'])
        gate = 'source_inventory'
        safe_tar(archive, out / 'source', prefix='mmcv-' + c['source']['commit'], maximum=c['limits']['maximum_source_expanded_bytes'])
        report['full_source'] = source_inventory(out / 'source', tree, c)
        for name in ('LICENSE', 'LICENSES.md'):
            shutil.copyfile(out / 'source' / name, legal / ('MMCV-' + name))
        gate = 'toolkit_inventory'
        components = out / 'components'
        components.mkdir(mode=0o700)
        roots = []
        report['component_notices'] = {}
        for row in c['assets']:
            if row['component'].startswith('cuda_'):
                root = components / row['component']
                safe_tar(downloads / row['file'], root, prefix=row['file'].removesuffix('.tar.xz'),
                         maximum=c['limits']['maximum_toolkit_expanded_bytes'])
                found = notices(root)
                require(found)
                report['component_notices'][row['component']] = found
                for name in found:
                    destination = legal / row['component'] / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / name, destination)
                roots.append(root)
        report['header_packages'] = {}
        for row in c['header_assets']:
            root = components / row['package']
            record = deb_headers(downloads / row['file'], root, c)
            report['header_packages'][row['package']] = record
            for p in (root / 'notices').rglob('*'):
                if p.is_file():
                    target_notice = legal / row['package'] / p.relative_to(root / 'notices')
                    target_notice.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    shutil.copyfile(p, target_notice)
            roots.append(root)
        require(sum(r['selected_header_and_notice_bytes'] for r in report['header_packages'].values()) <=
                c['limits']['maximum_DEB_selected_header_and_notice_bytes_total'])
        toolkit_merge(roots, out / 'toolkit')
        report['toolkit_inventory'] = inventory(out / 'toolkit')
        require(report['toolkit_inventory']['bytes'] <= c['limits']['maximum_toolkit_expanded_bytes'])
        for name in c['toolkit_inventory']['before_compile_required']:
            require((out / 'toolkit' / name).exists())
        require(any((out / 'toolkit' / lib / 'libcudart.so').is_file() for lib in ('lib', 'lib64')))
        report['ninja_notices'] = ninja_binary(downloads / ninja_row['file'], out / 'ninja')
        report['ninja_binary_inventory'] = inventory(out / 'ninja')
        shutil.copyfile(downloads / ninja_row['file'], legal / ninja_row['file'])
        gate = 'cpu_compile'
        report['compile'], wheel = compile_full(rt, out, args.revision, c, deadline)
        report['compiled_wheel'] = dict(file=wheel.name, identity=rt.identity(wheel, 1000000000), inventory=compiled_wheel(wheel))
        assets[wheel] = report['compiled_wheel']['identity']
        report['public_assets'] = {p.relative_to(out).as_posix(): pin for p, pin in assets.items()}
        report['preserved_notices'] = inventory(legal)
        require(source_inventory(out / 'source', tree, c) == report['full_source'])
        gate = 'isolated_child_build'
        rt.write(out / 'Dockerfile', recipe(wheel))
        rt.write(out / '.dockerignore', ('*\n!Dockerfile\n!' + wheel.name + '\n!notices/\n!notices/**\n').encode())
        image(rt, target, deadline, absent=True)
        r = command(['docker', 'build', '--pull=false', '--no-cache', '--network', 'none', '--memory', '8g', '--cpu-period', '100000',
            '--cpu-quota', '400000', '--label', LABEL + '=' + args.revision, '--tag', target,
            '--file', str(out / 'Dockerfile'), str(out)], deadline, log=out / 'build.log')
        require(r.returncode == 0)
        child = image(rt, target, deadline, owner=args.revision)
        require(child['image_id'] != BASE and child['layers'][:len(proof['image']['layers'])] == proof['image']['layers'])
        gate = 'child_cpu_import'
        report['child_cpu_import'] = cpu(rt, child['image_id'], out, args.revision, 'child', proof['versions'], deadline)
        require({k: v for k, v in report['child_cpu_import'].items() if k != 'mmcv_extension_sha256'} ==
                {k: v for k, v in report['base_cpu_import'].items() if k != 'mmcv_extension_sha256'}
                and report['child_cpu_import']['mmcv_extension_sha256'] == report['compiled_wheel']['inventory']['extension_sha256'])
        report.update(status='pass', child_image=child, target_tag=target)
    except BaseException as exc:
        report.update(failure_gate=gate, failure_class=CLASSES.get(type(exc), 'other'))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_deadline = deadline + GRACE
        try:
            require(out.resolve() == out and (out.stat().st_dev, out.stat().st_ino, out.stat().st_uid) == owner)
            for label in ('base', 'compile', 'child'):
                remove_container(rt, out / (label + '.cid'), 'wr-masa-sm90-' + args.revision + '-' + label, args.revision, cleanup_deadline)
            report['owned_containers_removed'] = True
            require(current_source(rt, code, args.revision) == binding)
            require(rt.source(ROOT, proof['old_code'], args.runtime_revision, 'run_masa_runtime_build',
                              tuple(proof['source']['helpers'])) == proof['source'])
            require(image(rt, BASE, cleanup_deadline) == proof['image'])
            for p, pin in {**proof['frozen'], **assets}.items():
                require(rt.identity(p, 2500000000) == pin)
            report['public_assets'] = {p.relative_to(out).as_posix(): pin for p, pin in assets.items()}
            if tree is not None and 'full_source' in report:
                require(source_inventory(out / 'source', tree, c) == report['full_source'])
            if 'toolkit_inventory' in report:
                require(inventory(out / 'toolkit') == report['toolkit_inventory'])
            if 'ninja_binary_inventory' in report:
                require(inventory(out / 'ninja') == report['ninja_binary_inventory'])
            if 'preserved_notices' in report:
                require(inventory(out / 'notices') == report['preserved_notices'])
            if child is not None:
                require(image(rt, target, cleanup_deadline, owner=args.revision) == child)
            report.update(source_rehashed_after=True, prior_runtime_rehashed_after=True, assets_rehashed_after=True,
                          prior_image_unchanged=True)
        except BaseException as exc:
            report.update(status='fail', failure_gate='postcheck_or_owned_cleanup', failure_class=CLASSES.get(type(exc), 'other'))
        try:
            report.update(compile_diagnostic(rt, out))
            for p, key in partials:
                if p.exists() or p.is_symlink():
                    s = p.lstat()
                    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino, s.st_uid) == key)
                    p.unlink()
            if time.monotonic() >= deadline:
                report.update(status='fail', failure_gate='root_deadline', failure_class='TimeoutError')
            # Only our exclusive lease is disposable; existing runtime is never removed.
            for folder in ('work', 'source', 'components', 'toolkit', 'ninja'):
                p = out / folder
                if p.exists():
                    require(not p.is_symlink() and p.resolve().parent == out)
                    owned_remove_tree(p, owner[2])
            if report['status'] != 'pass':
                try:
                    owned = image(rt, target, cleanup_deadline, owner=args.revision)
                    require(owned['image_id'] != BASE)
                    require(command(['docker', 'image', 'rm', target], cleanup_deadline).returncode == 0)
                    report['owned_failed_image_removed'] = True
                except BaseException:
                    image(rt, target, cleanup_deadline, absent=True)
            report['owned_disposable_cleanup'] = True
            expected = {out / 'downloads', out / 'notices', out / 'Dockerfile', out / '.dockerignore'}
            expected |= set(assets) | {out / (label + suffix) for label in ('base', 'compile', 'child') for suffix in ('.cid', '.log')}
            expected |= {out / 'build.log'}
            expected |= {p for p in (out / 'notices').rglob('*')} if (out / 'notices').exists() else set()
            for p in out.rglob('*'):
                s = p.lstat()
                require(p in expected and not p.is_symlink() and s.st_uid == owner[2] and
                        (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1))
                if p.is_file():
                    p.chmod(0o444)
            for p in sorted((p for p in out.rglob('*') if p.is_dir()), reverse=True):
                p.chmod(0o555)
            report['public_sealed'] = True
        except BaseException as exc:
            report.update(status='fail', failure_gate='disposable_cleanup_or_seal', failure_class=CLASSES.get(type(exc), 'other'))
        seal_report(rt, out, report, start)
    if report['status'] != 'pass':
        raise ValueError('SM90 CPU build failed; bounded immutable receipt retained') from None
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True)
    parser.add_argument('--runtime-revision', required=True)
    parser.add_argument('--runtime-report-bytes', type=int, required=True)
    parser.add_argument('--runtime-report-sha256', required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.environ.get('WR_ROOT') == str(ROOT)
            and args.revision == os.environ['WR_CODE_REVISION'] and re.fullmatch('[0-9a-f]{40}', args.revision))
    start = float(os.environ['WR_BUILD_STARTED'])
    remaining = start + BUDGET - time.monotonic()
    require(math.isfinite(start) and 0 < remaining <= BUDGET)
    def cancel(*_):
        raise TimeoutError('Build cancellation')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGALRM, cancel)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    result = run(Path(os.environ['WR_CODE']), args, started=start)
    print(json.dumps(dict(stage=result['stage'], status=result['status'], elapsed_seconds=result['elapsed_seconds'], gpu_used=False)))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print('MASA SM90 CPU build failed', file=sys.stderr)
        raise SystemExit(1) from None

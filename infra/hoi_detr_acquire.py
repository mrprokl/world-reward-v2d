"""Azure-only HOI-DETR source/opaque checkpoint acquisition; no model execution.

The complete archive authenticates to the independently frozen Git root tree.
Only original code/build text/notices survive; archive media are never decoded.
Publisher MIT declarations do not establish training overlap or stack clearance.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/hoi_detr_v1')
JOB = 'run_hoi_detr_acquire'
PROTOCOL = 'configs/hoi_detr_acquisition_v1.json'
PROTOCOL_PIN = dict(bytes=6229, sha256='c8cb9dae96d6d82282bc80da8561b8235167323103a35766d9cd81c281666387')
HELPER_PINS = {
    'infra/mediapipe_cpu_runtime_verify.py': dict(bytes=23559, sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
    'infra/mediapipe_hands_acquire.py': dict(bytes=21381, sha256='8293510c02777cd5b844865285736ff1a653631a8803849a9047ee9030196570'),
}
HELPERS = ('infra/hoi_detr_acquire.py', 'infra/run_hoi_detr_acquire.sh', PROTOCOL, *HELPER_PINS)
BLOCK = 1 << 20


def require(value, message):
    if not value:
        raise ValueError(message)


def helpers(code):
    """Authenticate immutable shared stdlib helpers before importing them."""
    loaded = []
    for name, pin in HELPER_PINS.items():
        path = Path(code)/name
        require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'Original helper path required')
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o222 and before.st_size == pin['bytes'], 'Readonly pinned helper required')
        raw = path.read_bytes()
        after = path.lstat()
        require(hashlib.sha256(raw).hexdigest() == pin['sha256'] and all(getattr(before, k) == getattr(after, k)
                for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_uid', 'st_gid')), 'Independent helper bytes differ')
        spec = importlib.util.spec_from_file_location('wr_hoi_acquire_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        require(Path(module.__file__) == path, 'Imported helper origin differs')
        loaded.append(module)
    return tuple(loaded)


def binding(rt, root, code, revision):
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual immutable acquisition source required')
    rt.require({p.name for p in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Exact source snapshot parent required')
    return rt.source(root, code, revision, JOB, HELPERS)


def relative(name):
    p = PurePosixPath(name)
    require(type(name) is str and name and not p.is_absolute() and '..' not in p.parts and p.as_posix() == name
            and '\\' not in name and ':' not in name and not any(ord(c) < 32 or ord(c) == 127 for c in name), 'Strict relative source/artifact path required')
    name.encode('utf-8', 'strict')
    return p


def endpoint(url):
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ''
    require(p.scheme == 'https' and not p.username and not p.password and p.port in (None, 443) and not p.fragment
            and (host in ('codeload.github.com', 'raw.githubusercontent.com', 'huggingface.co')
                 or host.endswith(('.hf.co', '.huggingface.co'))), 'Public closed HTTPS endpoint required')
    return p


def transport_destination(original, final):
    a, b = endpoint(original), endpoint(final)
    if original != final:
        require(a.hostname == 'huggingface.co' and '/resolve/' in a.path
                and (b.hostname == 'huggingface.co' or b.hostname.endswith(('.hf.co', '.huggingface.co'))), 'Only explicit publisher weight redirects allowed')


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        original = getattr(req, '_wr_public_origin', req.full_url)
        transport_destination(original, newurl)
        require(not any(k.lower() in ('authorization', 'cookie') for k in req.headers), 'Authenticated transport forbidden')
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None:
            result._wr_public_origin = original
        return result


def manifest(rt, code):
    c = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 32 << 10)
    rt.require(c['schema'] == 'world_reward.hoi_detr_acquisition.v1' and c['stage'] == 'source_model_acquisition_only'
               and c['data_root'] == str(DATA) and c['report'] == 'results/hoi-detr-acquire-v1.json'
               and c['cutoff'] == '2026-09-30' and (c['budget_seconds'], c['cleanup_grace_seconds'], c['outer_seconds']) == (900, 60, 960)
               and c['minimum_free_bytes'] == 16 << 30 and c['maximum_total_bytes'] == 6_200_000_000
               and c['maximum_manifest_bytes'] == 2 << 20, 'Frozen acquisition scope differs')
    s = c['source']
    relative(s['archive_prefix']); endpoint(s['url'])
    rt.require(re.fullmatch('[0-9a-f]{40}', s['revision']) and re.fullmatch('[0-9a-f]{40}', s['git_root_tree_sha1'])
               and s['repository'] == 'AhmadDarKhalil/HOI-DETR' and s['commit_date'][:10] <= c['cutoff']
               and s['url'] == 'https://codeload.github.com/'+s['repository']+'/tar.gz/'+s['revision']
               and s['archive_prefix'] == 'HOI-DETR-'+s['revision'] and s['archive_sha256_basis'] == 'measured_only_not_independent'
               and 0 < s['blob_count'] <= s['maximum_blobs'] <= 5000 and 0 < s['expanded_bytes'] <= s['maximum_expanded_bytes'] == 128 << 20
               and s['maximum_archive_bytes'] == 128 << 20 and s['maximum_member_bytes'] == 16 << 20
               and s['maximum_retained_file_bytes'] == 500000 and s['maximum_tar_entries'] == 10000, 'Frozen complete source tree contract differs')
    for name, row in s['symlinks'].items():
        relative(name)
        rt.require(name.startswith('mmdet/.mim/') and row['target'] == '../../'+PurePosixPath(name).name
                   and len(row['target'].encode()) == row['bytes'] and re.fullmatch('[0-9a-f]{40}', row['git_blob_sha1']), 'Original link text pin differs')
    for name, pin in s['required_source_pins'].items():
        relative(name); rt.require(retained(name) and type(pin['bytes']) is int and 0 < pin['bytes'] <= 500000 and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'Required retained source pin differs')
    rt.require(all(retained(n) for n in s['required_retained_files']), 'Native source closure retention differs')
    names = []
    for row in c['assets']:
        p = relative(row['file']); names.append(row['file']); endpoint(row['url'])
        rt.require(p.parts[0] in ('notices', 'weights') and type(row['bytes']) is int and 0 < row['bytes'] <= 6_000_000_000
                   and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Exact public asset identity required')
    rt.require(len(names) == len(set(names)) == 5 and names[-2:] == ['weights/README.md', 'weights/epoch_5.pth']
               and all(n.startswith('notices/') for n in names[:-2])
               and sum(r['bytes'] for r in c['assets'])+s['maximum_archive_bytes']+s['maximum_expanded_bytes'] < c['maximum_total_bytes'], 'Bounded ordered public acquisition required')
    cp = c['checkpoint']
    rt.require(cp['repository'] == 'ahmaddarkhalil/hoi-detr' and re.fullmatch('[0-9a-f]{40}', cp['revision'])
               and cp['revision_date'] <= c['cutoff'] and cp['model_card_license'] == 'mit' and cp['checkpoint_composition'] == 'not_decoded_unknown'
               and c['assets'][-1]['url'] == 'https://huggingface.co/'+cp['repository']+'/resolve/'+cp['revision']+'/epoch_5.pth'
               and c['assets'][-2]['url'] == 'https://huggingface.co/'+cp['repository']+'/raw/'+cp['revision']+'/README.md', 'Original pre-cutoff checkpoint/card required')
    rt.require(all(v is False for k, v in c['scope'].items() if k != 'training_overlap_status')
               and c['scope']['training_overlap_status'] == 'unknown_author_claim_not_independent_weight_audit'
               and all(v is False for k, v in c['future_runtime'].items() if k != 'required_next_gate'), 'No execution/quality/leakage assertion permitted')
    return c


def retained(name):
    """Closed source-only rule; original licenses/notices survive anywhere."""
    p = relative(name)
    base = p.name.upper()
    notice = base.startswith(('LICENSE', 'NOTICE', 'COPYING')) and p.suffix.lower() in ('', '.txt', '.md', '.rst')
    if notice:
        return True
    if any(x in ('__pycache__', '.mim') for x in p.parts):
        return False
    if p.parts[0] in ('mmdet', 'projects', 'configs', 'demo', 'tools') and p.suffix == '.py':
        return True
    if p.parts[0] == 'requirements' and p.suffix == '.txt':
        return True
    if p.parts[0] == 'docker' and (p.name.startswith('Dockerfile') or p.suffix in ('.sh', '.txt', '.yml', '.yaml', '.cfg')):
        return True
    return name in ('setup.py', 'setup.cfg', 'requirements.txt', 'README.md', 'INSTALL.md', 'INSTALL_HOPPER.md', '.gitattributes')


def check(deadline):
    require(time.monotonic() < deadline, 'Inclusive acquisition deadline')


def parents(data, target, directories):
    require(target.is_relative_to(data) and target != data, 'Owned destination required')
    for path in reversed(target.parent.parents):
        if path.is_relative_to(data) and path != data:
            if path not in directories:
                path.mkdir(mode=0o700); directories.add(path)
    if target.parent != data and target.parent not in directories:
        target.parent.mkdir(mode=0o700); directories.add(target.parent)
    require(not target.exists() and not target.is_symlink(), 'Fresh artifact destination required')


def publish_bytes(mp, data, target, raw, owned, directories):
    parents(data, target, directories)
    part = target.with_name(target.name+'.part')
    with part.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600); s = part.lstat(); owned.append((part, (s.st_dev, s.st_ino, s.st_uid)))
        stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    mp.publish(part, target)


def download(mp, data, row, opener, deadline, owned, directories, *, archive=False):
    check(deadline); target = data/row['file']; parents(data, target, directories)
    part = target.with_name(target.name+'.part'); maximum = row['maximum_archive_bytes'] if archive else row['bytes']
    request = urllib.request.Request(row['url'], headers={'Accept-Encoding': 'identity'})
    with opener.open(request, timeout=min(30, max(.001, deadline-time.monotonic()))) as response:
        transport_destination(row['url'], response.geturl())
        require(response.status == 200 and response.headers.get('Content-Encoding', 'identity').lower() == 'identity'
                and response.headers.get_content_type().lower() != 'text/html', 'Publisher transport differs')
        length = response.headers.get('Content-Length')
        require(length is None or re.fullmatch('[0-9]+', length) and (0 < int(length) <= maximum if archive else int(length) == maximum), 'Publisher content length differs')
        digest, count = hashlib.sha256(), 0
        with part.open('xb') as stream:
            os.fchmod(stream.fileno(), 0o600); s = part.lstat(); owned.append((part, (s.st_dev, s.st_ino, s.st_uid)))
            while True:
                check(deadline); block = response.read(min(BLOCK, maximum-count+1))
                require(len(block) <= BLOCK and count+len(block) <= maximum, 'Publisher byte overflow')
                if not block:
                    break
                if count == 0:
                    require(not block.lstrip().lower().startswith((b'<html', b'<!doctype html')) and not block.startswith(b'version https://git-lfs.github.com/spec/'), 'HTML/LFS pointer rejected')
                stream.write(block); digest.update(block); count += len(block)
            require(count > 0 and (archive or count == row['bytes'] and digest.hexdigest() == row['sha256']), 'Exact publisher bytes/SHA differ')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
        check(deadline); mp.publish(part, target)
    return dict(file=row['file'], bytes=count, sha256=digest.hexdigest(), hash_basis='measured_archive_sha256_not_independent' if archive else row['hash_basis'])


def git_tree(blobs):
    """Reconstruct Git's exact mode/name/NUL/binary-OID tree serialization."""
    trees = {(): {}}
    for name, row in blobs.items():
        p = relative(name); require(row['mode'] in ('100644', '100755', '120000') and re.fullmatch('[0-9a-f]{40}', row['git_blob_sha1']), 'Canonical Git blob mode/OID required')
        parts = p.parts
        for k in range(len(parts)-1):
            parent, key, child = parts[:k], parts[k], parts[:k+1]
            old = trees.setdefault(parent, {}).get(key)
            require(old is None or old == ('40000', child), 'Git file/directory collision')
            trees[parent][key] = ('40000', child); trees.setdefault(child, {})
        parent, key = parts[:-1], parts[-1]
        require(key not in trees.setdefault(parent, {}), 'Duplicate Git tree entry')
        trees[parent][key] = (row['mode'], row['git_blob_sha1'])
    hashes = {}
    for path in sorted(trees, key=len, reverse=True):
        entries = trees[path]
        raw = bytearray()
        for name in sorted(entries, key=lambda n: n.encode('utf-8')+(b'/' if entries[n][0] == '40000' else b'')):
            mode, oid = entries[name]
            raw.extend(mode.encode()+b' '+name.encode('utf-8')+b'\0'+bytes.fromhex(hashes[oid] if mode == '40000' else oid))
        hashes[path] = hashlib.sha1(b'tree '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    return hashes[()]


def archive_inventory(rt, mp, data, path, source, deadline, owned, directories, artifacts):
    prefix = source['archive_prefix']; blobs = {}; explicit_dirs = set(); entries = total = 0; seen_links = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            check(deadline); entries += 1
            require(entries <= source['maximum_tar_entries'], 'Archive entry count cap')
            full = member.name.rstrip('/') if member.isdir() else member.name
            relative(full)
            require(full == prefix or full.startswith(prefix+'/'), 'Original archive prefix required')
            name = full[len(prefix)+1:] if full != prefix else ''
            require(member.isdir() or member.isfile() or member.issym(), 'Source hardlinks/special members rejected')
            if member.isdir():
                require(full not in explicit_dirs and member.size == 0, 'Duplicate/nonempty archive directory')
                explicit_dirs.add(full); continue
            require(name and name not in blobs and 0 <= member.size <= source['maximum_member_bytes'], 'Duplicate/bounded original blob required')
            if member.issym():
                wanted = source['symlinks'].get(name)
                require(wanted is not None and member.linkname == wanted['target'] and member.size == 0, 'Only exact original link text accepted; never followed')
                raw_link = member.linkname.encode('utf-8'); count = len(raw_link)
                oid = hashlib.sha1(b'blob '+str(count).encode()+b'\0'+raw_link).hexdigest()
                require(count == wanted['bytes'] and oid == wanted['git_blob_sha1'], 'Original symlink Git blob differs')
                blobs[name] = dict(mode='120000', bytes=count, git_blob_sha1=oid); seen_links.add(name); total += count
            else:
                require(not member.mode & 0o111, 'Unexpected executable mode in frozen original source')
                count = 0; raw = bytearray(); digest = hashlib.sha1(b'blob '+str(member.size).encode()+b'\0'); sha = hashlib.sha256()
                keep = retained(name)
                require(not keep or member.size <= source['maximum_retained_file_bytes'], 'Retained source text cap')
                with archive.extractfile(member) as stream:
                    while True:
                        check(deadline); block = stream.read(BLOCK)
                        if not block:
                            break
                        count += len(block); require(count <= member.size, 'Expanded source member overflow'); digest.update(block); sha.update(block)
                        if keep:
                            raw.extend(block)
                require(count == member.size, 'Original source member truncated')
                blobs[name] = dict(mode='100644', bytes=count, git_blob_sha1=digest.hexdigest()); total += count
                if keep:
                    raw.decode('utf-8', 'strict')
                    require(b'\0' not in raw, 'Only original UTF-8 source/build/notice text retained')
                    target = data/'source/hoi-detr'/name
                    publish_bytes(mp, data, target, raw, owned, directories)
                    artifacts.append(dict(file=str(target.relative_to(data)), bytes=count, sha256=sha.hexdigest(), git_blob_sha1=digest.hexdigest(), hash_basis='whole_pinned_git_tree_authenticated_blob_and_measured_sha256'))
            require(total <= source['maximum_expanded_bytes'] and len(blobs) <= source['maximum_blobs'], 'Complete source expansion cap')
    allowed_dirs = {prefix}
    for name in blobs:
        allowed_dirs.update(prefix+'/'+str(p) for p in PurePosixPath(name).parents if str(p) != '.')
    require(explicit_dirs <= allowed_dirs and seen_links == set(source['symlinks']), 'Complete original links/no foreign empty directories required')
    require(len(blobs) == source['blob_count'] and total == source['expanded_bytes'] and git_tree(blobs) == source['git_root_tree_sha1'], 'Independent complete original Git root tree differs')
    retained_names = {r['file'].removeprefix('source/hoi-detr/') for r in artifacts}
    require(set(source['required_retained_files']) <= retained_names, 'Required native inference/build helpers missing')
    for name, pin in source['required_source_pins'].items():
        require(rt.identity(data/'source/hoi-detr'/name, 500000) == pin, 'Independently pinned original source differs')
    return dict(repository=source['repository'], revision=source['revision'], git_root_tree_sha1=source['git_root_tree_sha1'], blob_count=len(blobs), expanded_bytes=total,
                all_git_blobs_authenticated=True, root_tree_reconstructed=True, original_link_texts_verified=len(seen_links), links_materialized=False,
                retained_files=len(artifacts), source_manifest_inventory=blobs)


def declarations(rt, data, c):
    require(b'MIT License' in (data/'source/hoi-detr/LICENSE.txt').read_bytes()
            and (data/'weights/README.md').read_bytes() == b'---\r\nlicense: mit\r\n---\r\n'
            and all(b'Apache License' in (data/r['file']).read_bytes() for r in c['assets'][:2])
            and b'MIT License' in (data/c['assets'][2]['file']).read_bytes(), 'Exact source/card/upstream license declarations required before opaque weights')
    for name, pin in c['source']['required_source_pins'].items():
        require(rt.identity(data/'source/hoi-detr'/name, 500000) == pin, 'Original source changed before weights')


def write_receipt(path, report, started, deadline):
    """Own open descriptor permits FAIL demotion after a deadline-crossing fsync."""
    raw = (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(raw) <= 32 << 10, 'Concise acquisition receipt cap')
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        try:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
            if report['status'] == 'pass':
                check(deadline)
        except Exception as exc:
            report.update(status='fail', phase='receipt_sealing', receipt_error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
            stream.seek(0); stream.truncate()
            stream.write((json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    return report


def acquire(root, code, revision, *, opener=None, namespace_lease=None, started=None):
    started = time.monotonic() if started is None else started
    require(type(started) in (float, int) and 0 < started <= time.monotonic(), 'Original monotonic launch time required')
    rt, mp = helpers(code); root = rt.canonical(root); code = rt.canonical(code); before = binding(rt, root, code, revision); c = manifest(rt, code)
    deadline = started+c['budget_seconds']; check(deadline)
    data = rt.canonical(Path(c['data_root'])); out = rt.canonical(root/c['report'])
    require(data.parent.is_dir() and out.parent.is_dir() and not out.exists() and (namespace_lease is not None or not data.exists()), 'Fresh fixed namespaces only; no resume')
    require(shutil.disk_usage(data.parent).free >= c['minimum_free_bytes'], 'At least 16GiB free before transfer')
    artifacts, owned, created, directories = [], [], [], set(); failure = None; archive_pin = None; source = None
    report = dict(schema='world_reward.hoi_detr_source_model_acquisition.v1', stage='hoi_detr_source_model_acquisition', status='fail', phase='preflight', producer_revision=revision,
                  source_binding=before, protocol_identity=PROTOCOL_PIN, budget_seconds=c['budget_seconds'], cleanup_grace_seconds=c['cleanup_grace_seconds'], receipt_publication_outer_seconds=c['outer_seconds'],
                  budget_scope='launch_preflight_downloads_complete_Git_tree_license_checks_sealing_source_and_all_artifact_posthash', scope=c['scope'], future_runtime=c['future_runtime'],
                  first_party_license_declarations_verified=False, source_rehashed_after=False, artifacts_rehashed_after=False, owned_partials_removed=False, owned_archive_removed=False)
    try:
        if namespace_lease is not None:
            created = mp.validate_namespace_lease(namespace_lease, [data], before['closure_sha256'])
        else:
            data.mkdir(mode=0o700); data.chmod(0o700); s = data.lstat(); created = [(data, (s.st_dev, s.st_ino))]
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())
        report['phase'] = 'source_archive'
        row = dict(c['source'], file='.archives/hoi-detr.tar.gz')
        archive_pin = download(mp, data, row, opener, deadline, owned, directories, archive=True)
        report['archive_identity'] = {k: archive_pin[k] for k in ('bytes', 'sha256', 'hash_basis')}
        report['phase'] = 'complete_original_git_tree'
        source = archive_inventory(rt, mp, data, data/archive_pin['file'], c['source'], deadline, owned, directories, artifacts)
        inventory = source.pop('source_manifest_inventory')
        report['source_archive'] = source
        manifest_raw = (json.dumps(dict(schema='world_reward.hoi_detr_retained_source_manifest.v1', source=source, complete_original_git_blobs=inventory, retained_artifacts=artifacts), sort_keys=True, allow_nan=False)+'\n').encode()
        require(len(manifest_raw) <= c['maximum_manifest_bytes'], 'Useful source manifest cap')
        publish_bytes(mp, data, data/'source_manifest.json', manifest_raw, owned, directories)
        manifest_pin = dict(file='source_manifest.json', bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest(), hash_basis='complete_verified_git_tree_inventory_and_retained_source_pins')
        artifacts.append(manifest_pin); report['source_manifest'] = manifest_pin
        require(rt.identity(data/archive_pin['file'], c['source']['maximum_archive_bytes']) == {k: archive_pin[k] for k in ('bytes', 'sha256')}, 'Owned original archive changed')
        (data/archive_pin['file']).unlink(); archive_pin = None
        for row in c['assets']:
            if row['file'] == 'weights/epoch_5.pth':
                report['phase'] = 'license_preflight'; declarations(rt, data, c); report['first_party_license_declarations_verified'] = True
            report['phase'] = 'opaque_checkpoint_download' if row['file'].endswith('.pth') else 'primary_license_text_download'
            artifacts.append(download(mp, data, row, opener, deadline, owned, directories))
        report['public_assets'] = artifacts[-len(c['assets']):]
        report['phase'] = 'complete'; report['status'] = 'pass'
    except Exception as exc:
        failure = exc; report['error_type'] = type(exc).__name__
    finally:
        # Cleanup/posthash remain in the 900s logical budget; the 960s outer
        # watchdog bounds failure cleanup without permitting a late PASS.
        if signal.getsignal(signal.SIGALRM) is cancelled:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            for path, inode in owned:
                if path.exists() or path.is_symlink():
                    s = path.lstat()
                    require(not path.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and (s.st_dev, s.st_ino, s.st_uid) == inode, 'Owned partial replaced')
                    path.unlink()
            report['owned_partials_removed'] = all(not p.exists() and not p.is_symlink() for p, _ in owned)
            if archive_pin is not None:
                p = data/archive_pin['file']
                require(rt.identity(p, c['source']['maximum_archive_bytes']) == {k: archive_pin[k] for k in ('bytes', 'sha256')}, 'Owned archive replacement')
                p.unlink()
            report['owned_archive_removed'] = not tuple((data/'.archives').iterdir()) if (data/'.archives').exists() else True
            for path, inode in created:
                s = path.lstat()
                require(rt.canonical(path) == path and (s.st_dev, s.st_ino) == inode and s.st_uid == os.getuid(), 'Owned namespace replaced')
                expected = {data/r['file'] for r in artifacts}
                entries = sorted(data.rglob('*'), reverse=True)
                for p in entries:
                    rt.canonical(p); s = p.lstat()
                    require(s.st_uid == os.getuid() and (p in directories if stat.S_ISDIR(s.st_mode) else stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and p in expected), 'Foreign namespace entry')
                for p in entries:
                    p.chmod(0o555 if p.is_dir() else 0o444)
                data.chmod(0o555)
            report['artifacts_rehashed_after'] = all(rt.identity(data/r['file'], 6_000_000_000, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')} for r in artifacts)
            report['retained_artifact_count'] = len(artifacts); report['retained_bytes'] = sum(r['bytes'] for r in artifacts)
        except Exception as exc:
            failure = failure or exc; report['post_error_type'] = type(exc).__name__
        try:
            report['source_rehashed_after'] = binding(rt, root, code, revision) == before
        except Exception as exc:
            failure = failure or exc; report['source_post_error_type'] = type(exc).__name__
        report['elapsed_seconds'] = time.monotonic()-started
        if failure or report['elapsed_seconds'] > c['budget_seconds'] or not all(report[k] for k in ('first_party_license_declarations_verified', 'source_rehashed_after', 'artifacts_rehashed_after', 'owned_partials_removed', 'owned_archive_removed')):
            report['status'] = 'fail'
        write_receipt(out, report, started, deadline)
    if report['status'] != 'pass':
        raise ValueError('HOI-DETR acquisition failed; immutable receipt retained') from None
    return report


def cancelled(*_):
    raise TimeoutError('Acquisition cancellation')


def main():
    require(len(sys.argv) == 1 and sys.platform == 'linux' and os.geteuid() == 1000 and os.uname().nodename == 'world-reward-ncc-h100-02'
            and os.environ.get('WR_ROOT') == str(ROOT), 'Exact Azure VM02 unprivileged acquisition required')
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']; started = float(os.environ['WR_STARTED_MONOTONIC'])
    rt, _ = helpers(code); c = manifest(rt, code); remaining = started+c['budget_seconds']-time.monotonic(); require(remaining > 0, 'Inclusive launch deadline')
    signal.signal(signal.SIGALRM, cancelled); signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGINT, cancelled); signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        lease = rt.strict(os.environ['WR_NAMESPACE_LEASE'])
        value = acquire(ROOT, code, revision, namespace_lease=lease, started=started)
        print(json.dumps({k: value[k] for k in ('stage', 'status', 'elapsed_seconds')}))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(stage='hoi_detr_source_model_acquisition', status='fail', error_type=type(exc).__name__)))
        raise SystemExit(1) from None

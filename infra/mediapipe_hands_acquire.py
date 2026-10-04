"""Azure CPU source/model acquisition only: no install, inference, or dataset."""
import ctypes
import email.parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import pwd
import re
import signal
import stat
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
JOB = 'run_mediapipe_hands_acquire'
EVIDENCE = 'vendor/research/mediapipe_hands_v2'
WEIGHTS = 'weights/mediapipe_hand_landmarker_v2'
RESULT = 'results/mediapipe-hands-acquire-v2'
BUDGET, BLOCK, TIMEOUT = 300, 1 << 20, 20
SOURCE_REV = 'cad7f3ab99ebf175947e40c5252c642612aae927'
WHEEL = 'mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl'
WHEEL_URL = ('https://files.pythonhosted.org/packages/ad/83/'
             '56f760fecdc60de84c529d6c05c9dfc7d972617c6632bd254d5e021e5b16/' + WHEEL)
TASK_OBJECT = 'hand_landmarker%2Fhand_landmarker%2Ffloat16%2F1%2Fhand_landmarker.task'
CARD_OBJECT = 'Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf'
ASSETS = (
    dict(name='pypi.json', folder=EVIDENCE, url='https://pypi.org/pypi/mediapipe/0.10.21/json',
         bytes=24127, sha256='854b19f71fe5730332c9166527a66bda9cfa6e0f44ce39dea227b8f4b10d0680', mime=('application/json',)),
    dict(name='task-metadata.json', folder=EVIDENCE,
         url='https://storage.googleapis.com/storage/v1/b/mediapipe-models/o/' + TASK_OBJECT,
         bytes=1035, sha256='d02019c3fb8c3592b1adb7d913df5c784014b1b17136a158594438208abcb12c', mime=('application/json',)),
    dict(name='model-card-metadata.json', folder=EVIDENCE,
         url='https://storage.googleapis.com/storage/v1/b/mediapipe-assets/o/' + CARD_OBJECT,
         bytes=1045, sha256='4afb09c598b6669a9c98def38f08fc4061d1b7881bd0ce9f99503d5012a28dd5', mime=('application/json',)),
    dict(name='LICENSE', folder=EVIDENCE,
         url='https://raw.githubusercontent.com/google-ai-edge/mediapipe/' + SOURCE_REV + '/LICENSE',
         bytes=12331, sha256='8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8', mime=('text/plain',)),
    dict(name='model-card.pdf', folder=EVIDENCE,
         url='https://storage.googleapis.com/download/storage/v1/b/mediapipe-assets/o/' + CARD_OBJECT + '?generation=1683223215027175&alt=media',
         bytes=358044, sha256='43127ff8a92e22717f0d8c30dab3efef4641a29cbf50b15ae3de001be94ab76b', mime=('application/pdf',)),
    dict(name=WHEEL, folder=EVIDENCE, url=WHEEL_URL, bytes=35622638,
         sha256='05dc4a9e593655a79558d05d6227d31018c2537a4bd3362b51e230cf22aecfe3', mime=('application/octet-stream', 'application/zip')),
    dict(name='hand_landmarker.task', folder=WEIGHTS,
         url='https://storage.googleapis.com/download/storage/v1/b/mediapipe-models/o/' + TASK_OBJECT + '?generation=1682480004222387&alt=media',
         bytes=7819105, md5='15318430ea3851670fe9914116a9cfad', mime=('application/octet-stream',)),
)
URLS = frozenset(row['url'] for row in ASSETS)
METADATA_SHA = 'b72de4a6e099e64941a22f0a89e935c101880950f4cdb729d61b289ba6a767d1'


class AcquisitionError(ValueError):
    pass


def require(value, message):
    if not value: raise AcquisitionError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical non-symlink path required')
    return path


def state(path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def identity(path, *, empty=False):
    path = canonical(path); before = state(path)
    require(stat.S_ISREG(before[2]) and before[3] == 1 and not before[2] & 0o222 and
            (0 if empty else 1) <= before[4] <= 256 << 20, 'Bounded readonly regular artifact required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''): digest.update(block)
    require(state(path) == before, 'Artifact changed while hashing')
    return dict(bytes=before[4], sha256=digest.hexdigest())


def strict_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, 'Duplicate JSON keys'); out[key] = value
        return out
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(AcquisitionError('Nonfinite JSON')))


def source_binding(root, code, revision):
    code = canonical(code)
    require(re.fullmatch('[0-9a-f]{40}', revision) and
            code == root/'jobs'/revision/JOB/'code' and
            Path(__file__) == code/'infra/mediapipe_hands_acquire.py', 'Actual source-bound job required')
    markers = {n: identity(code.parent/n) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    closure = {}
    for path in (code, *sorted(code.rglob('*'))):
        canonical(path); before = state(path)
        require(not before[2] & 0o222, 'Readonly complete source closure required')
        closure[str(path.relative_to(code))] = {'directory': True} if stat.S_ISDIR(before[2]) else identity(path, empty=True)
    helpers = ('infra/mediapipe_hands_acquire.py', 'infra/run_mediapipe_hands_acquire.sh')
    require(all(n in closure and 'sha256' in closure[n] for n in helpers), 'Required acquisition source absent')
    return dict(producer_revision=revision, markers=markers, entries=len(closure),
                helpers={n: closure[n] for n in helpers},
                closure_sha256=hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest())


def safe_url(url):
    parsed = urllib.parse.urlsplit(url)
    require(url in URLS and parsed.scheme == 'https' and not parsed.username and not parsed.password and
            parsed.port in (None, 443) and not parsed.fragment, 'Unlisted public HTTPS endpoint')
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def publish(part, target):
    """Linux atomic NOREPLACE; never overwrite, merge, or resume."""
    call = ctypes.CDLL(None, use_errno=True).renameat2
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(part), -100, os.fsencode(target), 1):
        raise OSError(ctypes.get_errno(), 'Artifact publication refused')


def create_namespace_lease(folders, uid, gid, source_closure_sha256):
    """Root bootstraps only fresh leaves; never changes a shared parent."""
    require(os.getuid() == 0 and type(uid) is int and uid > 0 and type(gid) is int and gid >= 0,
            'Root bootstrap and unprivileged target required')
    folders = [canonical(p) for p in folders]
    require(len(set(folders)) == len(folders) and all(not p.exists() and p.parent.is_dir() for p in folders),
            'Only fresh canonical leaf namespaces may be bootstrapped')
    records = []
    for p in folders:
        p.mkdir(mode=0o700); p.chmod(0o700); os.chown(p, uid, gid)
        s = p.lstat()
        records.append(dict(path=str(p), device=s.st_dev, inode=s.st_ino, uid=uid, gid=gid, mode=0o700))
    return dict(schema='world_reward.fresh_namespace_lease.v1',
                source_closure_sha256=source_closure_sha256, directories=records)


def validate_namespace_lease(lease, folders, source_closure_sha256):
    require(type(lease) is dict and set(lease) == {'schema', 'source_closure_sha256', 'directories'} and
            lease['schema'] == 'world_reward.fresh_namespace_lease.v1' and
            lease['source_closure_sha256'] == source_closure_sha256 and
            type(lease['directories']) is list and len(lease['directories']) == len(folders),
            'Exact source-bound fresh namespace lease required')
    owned = []
    for p, row in zip(folders, lease['directories']):
        p = canonical(p); s = p.lstat()
        expected = dict(path=str(p), device=s.st_dev, inode=s.st_ino,
                        uid=os.getuid(), gid=os.getgid(), mode=0o700)
        require(type(row) is dict and set(row) == set(expected) and
                all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items()) and
                stat.S_ISDIR(s.st_mode) and s.st_mode & 0o777 == 0o700 and
                s.st_uid == os.getuid() and s.st_gid == os.getgid() and not any(p.iterdir()),
                'Bootstrap lease changed, occupied, foreign, or not empty')
        owned.append((p, (s.st_dev, s.st_ino)))
    return owned


def fetch(row, root, opener, deadline, owned):
    require(time.monotonic() < deadline, 'Inclusive acquisition budget exceeded')
    target = root/row['folder']/row['name']; part = target.with_name(target.name+'.part')
    response = opener.open(urllib.request.Request(safe_url(row['url']), headers={'Accept-Encoding': 'identity'}), timeout=TIMEOUT)
    with response:
        require(response.status == 200 and response.geturl() == row['url'], 'Publisher response/endpoint differs')
        require(response.headers.get_content_type().lower() in row['mime'] and
                response.headers.get('Content-Encoding', 'identity').lower() == 'identity', 'Publisher MIME/encoding differs')
        size = response.headers.get('Content-Length')
        require(size is None or re.fullmatch('[0-9]+', size) and int(size) == row['bytes'], 'Publisher length differs')
        digest, md5, count = hashlib.sha256(), hashlib.md5(), 0
        with part.open('xb') as output:
            os.fchmod(output.fileno(), 0o600); owned.append((part, state(part)[:2]))
            while True:
                require(time.monotonic() < deadline, 'Inclusive acquisition budget exceeded')
                block = response.read(BLOCK)
                require(len(block) <= BLOCK and count+len(block) <= row['bytes'], 'Publisher body overflow')
                if not block: break
                output.write(block); digest.update(block); md5.update(block); count += len(block)
            require(count == row['bytes'] and (not row.get('sha256') or digest.hexdigest() == row['sha256']) and
                    (not row.get('md5') or md5.hexdigest() == row['md5']), 'Publisher byte/hash pin differs')
            output.flush(); os.fsync(output.fileno()); os.fchmod(output.fileno(), 0o444)
        require(time.monotonic() < deadline, 'Inclusive acquisition budget exceeded')
        publish(part, target)
    return dict(file=row['folder']+'/'+row['name'], bytes=count, sha256=digest.hexdigest(),
                publisher_sha256_verified=row['name'] == WHEEL, publisher_md5_verified=bool(row.get('md5')),
                hash_basis='publisher_sha256' if row.get('sha256') and row['name'] == WHEEL else
                'independent_audited_sha256' if row.get('sha256') else 'caller_measured_sha256_publisher_md5')


def verify_metadata(root):
    directory = root/EVIDENCE; package = strict_json(directory/'pypi.json')
    wheel = next(row for row in ASSETS if row['name'] == WHEEL)
    entries = [r for r in package['urls'] if r.get('filename') == WHEEL]
    require(len(entries) == 1 and entries[0].get('size') == wheel['bytes'] and entries[0].get('url') == wheel['url'] and
            entries[0].get('digests', {}).get('sha256') == wheel['sha256'] and entries[0].get('yanked') is False and
            entries[0].get('core-metadata', {}).get('sha256') == METADATA_SHA, 'Published wheel metadata differs')
    for name, bucket, object_name, generation, size, md5 in (
        ('task-metadata.json', 'mediapipe-models', urllib.parse.unquote(TASK_OBJECT), '1682480004222387', 7819105, 'FTGEMOo4UWcP6ZFBFqnPrQ=='),
        ('model-card-metadata.json', 'mediapipe-assets', urllib.parse.unquote(CARD_OBJECT), '1683223215027175', 358044, 'G6gDPVW6ruzBtatTpakf2w==')):
        value = strict_json(directory/name)
        require(all(value.get(k) == v for k, v in dict(bucket=bucket, name=object_name,
                    generation=generation, size=str(size), md5Hash=md5).items()), 'Official object metadata differs')
    require(b'Apache License' in (directory/'LICENSE').read_bytes() and
            b'Version 2.0' in (directory/'LICENSE').read_bytes(), 'Audited source license differs')


def zip_inventory(path, deadline, *, wheel=False):
    """Bounded opaque byte inventory/CRC, never extraction or model-node decoding."""
    rows, seen, texts, total = [], set(), {}, 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist(); require(0 < len(entries) <= 5000, 'ZIP inventory count invalid')
        for item in entries:
            require(time.monotonic() < deadline, 'Inclusive acquisition budget exceeded')
            name = item.filename; pure = PurePosixPath(name.rstrip('/')); kind = stat.S_IFMT(item.external_attr >> 16)
            require(item.orig_filename == name and name and len(name) <= 4096 and pure.as_posix() == name.rstrip('/') and not pure.is_absolute() and
                    '..' not in pure.parts and '\\' not in name and ':' not in name and
                    not any(ord(c) < 32 or ord(c) == 127 for c in name) and name not in seen and
                    pure.as_posix() not in seen and kind in (0, stat.S_IFREG, stat.S_IFDIR) and
                    not item.flag_bits & 1 and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'Unsafe ZIP member')
            seen.update((name, pure.as_posix())); total += item.file_size
            require(0 <= item.file_size <= 256 << 20 and total <= 512 << 20 and
                    (not item.is_dir() or item.file_size == 0), 'ZIP expanded byte cap exceeded')
            digest, count, chunks = hashlib.sha256(), 0, []
            text = wheel and (name.endswith('.dist-info/METADATA') or '/LICENSE' in name or '/licenses/' in name)
            if text: require(item.file_size <= 128 << 10, 'Wheel license/metadata text cap exceeded')
            with archive.open(item) as stream:
                while True:
                    require(time.monotonic() < deadline, 'Inclusive acquisition budget exceeded')
                    block = stream.read(BLOCK)
                    if not block: break
                    count += len(block); require(count <= item.file_size, 'ZIP member overflow'); digest.update(block)
                    if text: chunks.append(block)
            require(count == item.file_size, 'ZIP member truncated')
            rows.append(dict(name=name, bytes=count, sha256=digest.hexdigest(), crc32=f'{item.CRC:08x}'))
            if text: texts[name] = b''.join(chunks)
    result = dict(entries=len(rows), expanded_bytes=total, inventory=rows)
    if wheel:
        prefix = 'mediapipe-0.10.21.dist-info/'; raw = texts.get(prefix+'METADATA', b'')
        require(hashlib.sha256(raw).hexdigest() == METADATA_SHA, 'Wheel METADATA publisher SHA differs')
        parsed = email.parser.BytesParser().parsebytes(raw)
        require(parsed.get_all('Name') == ['mediapipe'] and parsed.get_all('Version') == ['0.10.21'] and
                parsed.get_all('License') == ['Apache 2.0'] and parsed.get_all('License-File') == ['LICENSE'], 'Wheel metadata/license identity differs')
        declared = prefix+'LICENSE'
        require(declared in texts and b'Apache License' in texts[declared] and b'Version 2.0' in texts[declared], 'Declared wheel license missing')
        result['package_metadata'] = dict(name='mediapipe', version='0.10.21', license='Apache 2.0',
                                         license_files=parsed.get_all('License-File'), requires_dist=parsed.get_all('Requires-Dist'))
        result['inventory_sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        result['license_metadata_files'] = [r for r in rows if r['name'] in texts]
        del result['inventory']  # full wheel bytes already pinned; retain useful license inventory only
    return result


def acquire(root, code, revision, *, opener=None, namespace_lease=None):
    started = time.monotonic(); root = canonical(root); code = canonical(code)
    folders = [canonical(root/n) for n in (EVIDENCE, WEIGHTS, RESULT)]
    require(all(p.parent.is_dir() for p in folders) and not folders[-1].exists() and
            (namespace_lease is not None or all(not p.exists() for p in folders[:-1])),
            'Fresh existing-parent namespaces required; no resume')
    out = folders[-1]; out.mkdir(mode=0o700); out.chmod(0o700)
    owned, artifacts, created, before, failure = [], [], [], None, None
    report = dict(stage='mediapipe_hands_source_model_acquisition', status='fail', budget_seconds=BUDGET,
                  budget_scope='acquisition_checks_public_sealing_posthash', receipt_publication_outer_seconds=320,
                  models_loaded=False, model_nodes_decoded=False, packages_installed=False, gpu_used=False,
                  dataset_read=False, private_values_read=False, quality_claim=False,
                  license_eligibility_verified=False, task_constituent_license_verified=False,
                  training_overlap_verified=False, challenge_overlap_verified=False, artifacts=artifacts)
    try:
        before = source_binding(root, code, revision); report['source_binding'] = before
        if namespace_lease is None:
            for path in folders[:-1]:
                path.mkdir(mode=0o700); created.append((path, state(path)[:2])); path.chmod(0o700)
        else:
            created = validate_namespace_lease(namespace_lease, folders[:-1], before['closure_sha256'])
            report['namespace_lease'] = namespace_lease
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())
        for row in ASSETS:
            artifacts.append(fetch(row, root, opener, started+BUDGET, owned))
            if row['name'] == 'LICENSE': verify_metadata(root)
        report['wheel'] = zip_inventory(root/EVIDENCE/WHEEL, started+BUDGET, wheel=True)
        report['task'] = zip_inventory(root/WEIGHTS/'hand_landmarker.task', started+BUDGET)
        report['model_card_license_basis'] = 'previous_independent_text_audit_Apache_2.0_not_exact_bundle_clearance'
        report['status'] = 'pass'
    except Exception as exc:
        failure = exc; report['error_type'] = type(exc).__name__
        if isinstance(exc, AcquisitionError): report['reason'] = str(exc)
    finally:
        # Our logical budget remains inclusive through final proof. Disable the
        # interrupting watchdog for bounded sealing; outer timeout remains live.
        if signal.getsignal(signal.SIGALRM) is cancel: signal.setitimer(signal.ITIMER_REAL, 0)
        for part, inode in owned:
            if part.exists() and not part.is_symlink() and state(part)[:2] == inode: part.unlink()
        for folder, inode in created:
            if folder.exists() and not folder.is_symlink() and state(folder)[:2] == inode: folder.chmod(0o555)
        try: report['source_rehashed_after'] = before is not None and source_binding(root, code, revision) == before
        except Exception: report['source_rehashed_after'] = False
        try:
            report['artifacts_rehashed_after'] = all(identity(root/r['file']) ==
                {k: r[k] for k in ('bytes', 'sha256')} for r in artifacts)
        except Exception: report['artifacts_rehashed_after'] = False
        report['elapsed_seconds'] = time.monotonic()-started
        report['owned_partials_removed'] = all(not p.exists() for p, _ in owned)
        if not report['source_rehashed_after'] or not report['artifacts_rehashed_after'] or report['elapsed_seconds'] > BUDGET:
            report['status'] = 'fail'; failure = failure or AcquisitionError('Source or inclusive budget failure')
        with (out/'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    if failure: raise AcquisitionError('MediaPipe acquisition failed; inspect sealed receipt') from None
    return report


def cancel(signum, frame):
    raise AcquisitionError('Acquisition cancelled or budget expired')


def main():
    require(platform.system() == 'Linux' and os.environ.get('WR_ROOT') == str(ROOT) and
            os.getuid() == pwd.getpwnam('scenesmith').pw_uid, 'Azure CPU scenesmith runtime required')
    signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGALRM, cancel)
    signal.setitimer(signal.ITIMER_REAL, BUDGET+5)
    try:
        report = acquire(ROOT, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
                         namespace_lease=json.loads(os.environ['WR_NAMESPACE_LEASE']) if 'WR_NAMESPACE_LEASE' in os.environ else None)
        print(json.dumps({k: report[k] for k in ('stage', 'status', 'elapsed_seconds')}))
    finally: signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps(dict(stage='mediapipe_hands_source_model_acquisition', status='fail', error_type=type(exc).__name__)))
        raise SystemExit(1) from None

"""Fresh Azure-only HO-Cap RGB acquisition; all reference payloads quarantined.

Downloads are measured byte receipts, not publisher-checksum authentication.
Only chosen RGB and the top-level numeric num_frames metadata field are decoded.
No labels, camera calibration, mesh, pose, MANO or checkpoint values are loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import sys
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
DESTINATION = Path('/srv/world-reward-data/hocap_v2')
ENTRY = 'run_hocap_acquire'
PROTOCOL = 'configs/hocap_acquisition_protocol_v1.json'
PROTOCOL_PIN = dict(bytes=3150, sha256='2773671430d4a838179cd5121246c38f93a19a3265a21e5fac31b9b0187d58de')
HELPERS = ('infra/hocap_acquire.py', 'infra/run_hocap_acquire.sh',
           'infra/mediapipe_cpu_runtime_verify.py', PROTOCOL)
BLOCK = 1 << 20


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not any(q.is_symlink() for q in (p, *p.parents)), 'Canonical nonsymlink path required')
    return p


def state(path):
    s = canonical(path).lstat()
    return tuple(getattr(s, n) for n in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink', 'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns'))


def identity(path, maximum):
    p = canonical(path); before = state(p); s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and 0 < s.st_size <= maximum, 'Bounded single-link regular artifact required')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda: stream.read(BLOCK), b''): h.update(data)
    require(state(p) == before, 'Artifact changed while hashing')
    return dict(bytes=s.st_size, sha256=h.hexdigest())


def runtime(code):
    sys.path.insert(0, str(Path(code)/'infra'))
    import mediapipe_cpu_runtime_verify as rt
    require(Path(rt.__file__).resolve() == Path(code)/'infra/mediapipe_cpu_runtime_verify.py', 'Actual helper origin required')
    return rt


def source_binding(rt, code, revision):
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(Path(__file__).resolve() == code/'infra/hocap_acquire.py', 'Actual caller source required')
    protocol = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 16 << 10)
    return protocol, own


def check(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Inclusive acquisition deadline exhausted')


def public_url(url):
    p = urllib.parse.urlsplit(url); host = p.hostname or ''
    require(p.scheme == 'https' and p.port in (None, 443) and not p.username and not p.password and not p.fragment
        and (host in ('utdallas.box.com', 'utdallas.app.box.com') or host == 'boxcloud.com' or host.endswith('.boxcloud.com')
             or host in ('irvlutd.github.io', 'raw.githubusercontent.com')), 'Only public publisher HTTPS destinations allowed')
    return url


class PublisherRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        public_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(row, destination, opener, deadline, *, mode=0o444):
    """One streaming GET, exact final length; no retry/resume, credentials or logs."""
    public_url(row['url']); require(type(row['bytes']) is int and row['bytes'] > 0, 'Exact source size required')
    target = canonical(destination); part = target.with_name(target.name+'.part')
    require(not target.exists() and not part.exists(), 'Fresh download only')
    inode = None
    try:
        with opener.open(urllib.request.Request(row['url'], headers={'User-Agent': 'WorldReward-HOCap-source-acquisition'}),
                timeout=min(30, max(1, deadline-time.monotonic()))) as response:
            public_url(response.geturl()); require(response.status == 200, 'Publisher response must be 200')
            length = response.headers.get('Content-Length')
            require(length is None or length == str(row['bytes']), 'Publisher Content-Length differs')
            h = hashlib.sha256(); size = 0
            with part.open('xb') as stream:
                inode = state(part)[:2]
                while True:
                    check(deadline); data = response.read(min(BLOCK, row['bytes']-size+1))
                    if not data: break
                    size += len(data); require(size <= row['bytes'], 'Download byte cap exceeded')
                    stream.write(data); h.update(data)
                require(size == row['bytes'], 'Truncated publisher download')
                stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), mode)
        check(deadline); require(not target.exists(), 'Download target collision')
        os.link(part, target); part.unlink()
        return dict(bytes=size, sha256=h.hexdigest())
    finally:
        if part.exists():
            require(state(part)[:2] == inode and part.stat().st_uid == os.getuid(), 'Refuse foreign partial cleanup')
            part.unlink()


def inventory(path, protocol, deadline):
    """Central-directory safety only; no inactive member decode or CRC claim."""
    entries = {}; expanded = 0
    with zipfile.ZipFile(path) as saved:
        require(len(saved.infolist()) <= protocol['maximum_zip_members'], 'ZIP member cap exceeded')
        for m in saved.infolist():
            check(deadline); n = m.filename; pure = PurePosixPath(n.rstrip('/') if m.is_dir() else n)
            mode = m.external_attr >> 16; kind = stat.S_IFMT(mode)
            require(n == m.orig_filename and 0 < len(n) <= 512 and str(pure) != '.' and '\\' not in n and not pure.is_absolute()
                and pure.as_posix() == (n.rstrip('/') if m.is_dir() else n)
                and not any(p in ('.', '..') for p in pure.parts) and not any(ord(c) < 32 or ord(c) == 127 for c in n)
                and str(pure) not in entries and not m.flag_bits & (1 | 64)
                and m.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                and kind in (0, stat.S_IFREG, stat.S_IFDIR)
                and (kind != stat.S_IFDIR or m.is_dir()) and (kind != stat.S_IFREG or not m.is_dir())
                and 0 <= m.file_size <= protocol['maximum_member_bytes']
                and (not m.is_dir() or m.file_size == 0), 'Unsafe ZIP name/type/compression/size')
            entries[str(pure)] = m; expanded += m.file_size
            require(expanded <= protocol['maximum_expanded_bytes'], 'Expanded archive cap exceeded')
        for n in entries:
            require(all(str(p) not in entries or entries[str(p)].is_dir() for p in PurePosixPath(n).parents if str(p) != '.'), 'ZIP ancestor file collision')
    rows = [[n, m.file_size, m.compress_size, m.CRC, m.flag_bits, m.external_attr, m.compress_type] for n, m in sorted(entries.items())]
    return entries, dict(members=len(entries), expanded_bytes=expanded,
        inventory_sha256=hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest(), inactive_payloads_decoded=False)


def numeric_frames(raw, maximum):
    """Do not construct YAML or interpret task/object/hand/calibration fields."""
    text = raw.decode('utf-8'); matches = re.findall(r'^num_frames:[ \t]*([1-9][0-9]*)[ \t]*(?:#.*)?$', text, re.MULTILINE)
    keys = re.findall(r'^num_frames[^:\r\n]*:', text, re.MULTILINE)
    require(not re.search(r'^[ \t]*["\']num_frames["\'][ \t]*:', text, re.MULTILINE), 'Quoted frame-count key unsupported')
    require(len(matches) == len(keys) == 1 and int(matches[0]) <= maximum, 'One plain top-level positive num_frames required')
    return int(matches[0])


def extract_public(archive, entries, protocol, public, private, deadline, *, metadata=None):
    """Frozen direct publisher layout; unknown wrappers abstain, never guess."""
    images = []; clips = []; retained = 0; owned = []; metadata = {} if metadata is None else metadata
    try:
        with zipfile.ZipFile(archive) as saved:
            for clip in protocol['clips']:
                check(deadline); prefix = f"{protocol['subject']}/{clip}/"; meta = prefix+'meta.yaml'
                require(meta in entries and 0 < entries[meta].file_size <= protocol['maximum_metadata_bytes'], 'Frozen direct clip metadata absent')
                raw = saved.read(entries[meta]); count = numeric_frames(raw, protocol['maximum_frames_per_clip'])
                name = clip+'-meta.yaml'; path = private/name
                with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o400)
                metadata[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                colors = {int(n.rsplit('_', 1)[1][:-4]): m for n, m in entries.items()
                    if re.fullmatch(re.escape(prefix+protocol['camera']+'/')+r'color_[0-9]{6}\.jpg', n) and not m.is_dir()}
                require(sorted(colors) == list(range(count)), 'All original contiguous RGB frames must match num_frames')
                folder = public/clip; folder.mkdir(mode=0o755); owned.append(folder)
                for i in range(count):
                    check(deadline); m = colors[i]; retained += m.file_size
                    require(0 < m.file_size <= protocol['maximum_member_bytes'] and retained <= protocol['maximum_public_bytes'], 'Public RGB retained-byte cap')
                    path = folder/f'color_{i:06d}.jpg'; h = hashlib.sha256(); size = 0
                    with saved.open(m) as src, path.open('xb') as dst:
                        owned.append(path)
                        while True:
                            check(deadline); data = src.read(BLOCK)
                            if not data: break
                            size += len(data); require(size <= m.file_size, 'Selected member size exceeded'); h.update(data); dst.write(data)
                        require(size == m.file_size, 'Selected RGB member truncated'); dst.flush(); os.fsync(dst.fileno()); os.fchmod(dst.fileno(), 0o444)
                    images.append(dict(clip=clip, camera=protocol['camera'], frame_position=i, source_frame_id=i,
                        file=str(path.relative_to(public)), bytes=size, sha256=h.hexdigest()))
                clips.append(dict(clip=clip, camera=protocol['camera'], num_frames=count))
        return dict(schema='world_reward.hocap_rgb_inventory.v1', subject=protocol['subject'], clips=clips,
            images=images, raw_metadata_public=False, annotations_public=False, calibration_public=False,
            timestamps_verified=False, image_decoder_qualified=False, reference_continuity_qualified=False), metadata
    except BaseException:
        for path in reversed(owned):
            require(path.stat().st_uid == os.getuid() and not path.is_symlink(), 'Refuse foreign public cleanup')
            path.rmdir() if path.is_dir() else path.unlink()
        raise


def write_report(path, report, mode=0o400, *, deadline=None):
    with path.open('xb') as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), mode)
        if deadline is not None and report.get('status') == 'pass' and time.monotonic() >= deadline:
            report.update(status='fail', post_error_type='InclusiveDeadline')
            stream.seek(0); stream.truncate()
            stream.write((json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno())


def acquire(rt, code, revision, target, *, opener=None, watchdog=False, reservation=None):
    started = time.monotonic(); protocol, before = source_binding(rt, code, revision); deadline = started+protocol['budget_seconds']
    target = canonical(target)
    if reservation is None: require(not target.exists(), 'Fresh acquisition namespace required')
    else:
        s = target.lstat()
        require(reservation == dict(device=s.st_dev, inode=s.st_ino, source_sha256=before['closure_sha256'])
            and stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o700
            and not tuple(target.iterdir()), 'Exact fresh owned namespace reservation required')
    require(shutil.disk_usage(target.parent).free >= protocol['minimum_free_bytes'], 'At least25GiB available disk required')
    if reservation is None: target.mkdir(mode=0o755)
    private = target/'quarantine'; private.mkdir(mode=0o700)
    public = target/'inputs'; public.mkdir(mode=0o755); primary = target/'primary'; primary.mkdir(mode=0o755)
    report = dict(schema='world_reward.hocap_acquisition.v1', stage='hocap_rgb_private_reference_acquisition',
        status='fail', phase='primary_sources', producer_revision=revision, source_before=before,
        protocol_identity=PROTOCOL_PIN, archives={}, primary_sources={}, archive_inventories={},
        download_complete=False, public_inventory_qualified=False, label_values_parsed=False,
        calibration_values_parsed=False, model_loaded=False, gpu_used=False, challenge_inputs_used=False,
        training_overlap_verified=False, challenge_overlap_verified=False, full_body_reference=False,
        contact_truth_verified=False, adoption=False, archive_hash_basis='measured_not_publisher_checksum')
    failure = None; opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), PublisherRedirect())
    if watchdog: signal.alarm(protocol['budget_seconds'])
    try:
        for name, row in protocol['primary_sources'].items():
            require(fetch(row, primary/name, opener, deadline) == {k: row[k] for k in ('bytes', 'sha256')}, 'Primary source bytes differ')
            report['primary_sources'][name] = identity(primary/name, protocol['maximum_metadata_bytes'])
        text = (primary/'publisher.html').read_text()
        require('Creative Commons Attribution 4.0 International License (CC BY 4.0)' in text and 'creativecommons.org/licenses/by/4.0/' in text, 'Explicit primary dataset grant required')
        report['phase'] = 'archive_download'
        for row in protocol['archives']:
            report['active_archive'] = row['file']; report['archives'][row['file']] = fetch(row, private/row['file'], opener, deadline, mode=0o400)
        report['download_complete'] = True; report['phase'] = 'safe_inventory'; selected = None
        for row in protocol['archives']:
            entries, audit = inventory(private/row['file'], protocol, deadline); report['archive_inventories'][row['file']] = audit
            if row['file'] == protocol['subject']+'.zip': selected = entries
        report['phase'] = 'public_extract'
        report['opaque_metadata'] = {}
        manifest, _ = extract_public(private/(protocol['subject']+'.zip'), selected, protocol, public, private, deadline,
            metadata=report['opaque_metadata'])
        check(deadline); write_report(public/'manifest.json', manifest, 0o444)
        report.update(public_inventory_qualified=True, frames=len(manifest['images']), clips=manifest['clips'], public_manifest=identity(public/'manifest.json', 16<<20))
    except BaseException as error:
        failure = error; report.update(error_type=type(error).__name__, error_context=str(error)[:180] if isinstance(error, (ValueError, TimeoutError)) else 'Acquisition failed without secret diagnostics')
    finally:
        if watchdog: signal.alarm(protocol['cleanup_grace_seconds'])
        try:
            report['phase_before_posthash'] = report['phase']
            require(source_binding(rt, code, revision) == (protocol, before), 'Frozen source/protocol changed')
            for row in protocol['archives']:
                if row['file'] in report['archives']: require(identity(private/row['file'], row['bytes']) == report['archives'][row['file']], 'Downloaded archive changed')
            for n, pin in report['primary_sources'].items(): require(identity(primary/n, protocol['maximum_metadata_bytes']) == pin, 'Primary source changed')
            for n, pin in report.get('opaque_metadata', {}).items():
                require(identity(private/n, protocol['maximum_metadata_bytes']) == pin, 'Opaque metadata changed')
            if report['public_inventory_qualified']:
                for row in manifest['images']: require(identity(public/row['file'], row['bytes']) == {k: row[k] for k in ('bytes', 'sha256')}, 'Public RGB changed')
                require(identity(public/'manifest.json', 16<<20) == report['public_manifest'], 'Public manifest changed')
            report['source_archive_public_rehashed_after'] = True
            check(deadline)
        except BaseException as error:
            failure = failure or error; report['post_error_type'] = type(error).__name__
        report.update(status='pass' if failure is None else 'fail', phase='complete' if failure is None else report['phase_before_posthash'], elapsed_seconds=time.monotonic()-started)
        primary.chmod(0o555)
        for folder in public.iterdir():
            if folder.is_dir(): folder.chmod(0o555)
        public.chmod(0o555); target.chmod(0o755)
        write_report(target/'report.json', report, deadline=deadline)
        if watchdog: signal.alarm(0)
    return report


def main():
    require(len(sys.argv) == 1 and sys.platform == 'linux' and os.geteuid() == 1000, 'Azure scenesmith CPU no-argument caller only')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    require(Path(os.environ['WR_ROOT']) == ROOT and re.fullmatch('[0-9a-f]{40}', revision), 'Frozen canonical root/revision required')
    def timeout(*_): raise TimeoutError('Inclusive acquisition budget exhausted')
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, timeout)
    rt = runtime(code); reservation = rt.strict(os.environ['WR_HOCAP_NAMESPACE_LEASE'])
    report = acquire(rt, code, revision, DESTINATION, watchdog=True, reservation=reservation)
    print(json.dumps({k: report[k] for k in ('stage', 'status', 'download_complete', 'public_inventory_qualified')}), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(stage='hocap_acquisition', status='fail', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1) from None

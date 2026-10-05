"""One Azure-only acquisition of three publisher metadata ZIPs; no JSON rows/RGB."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/visual_genome_metadata_v1')
ENTRY = 'run_visual_genome_metadata_acquire'
CONFIG = 'configs/visual_genome_metadata_acquire_v1.json'
HELPERS = ('infra/visual_genome_metadata_acquire.py', 'infra/run_visual_genome_metadata_acquire.sh',
           CONFIG, 'infra/mediapipe_cpu_runtime_verify.py')
PREFIX = 'https://homes.cs.washington.edu/~ranjay/visualgenome/'
ASSETS = tuple(dict(file=n+'.json.zip', member=n+'.json', url=PREFIX+'data/dataset/'+n+'.json.zip',
                    observed_bytes=size, version=v) for n, size, v in
               (('image_data', 1780854, '1.2'), ('objects', 55323929, '1.4'), ('relationships', 77904473, '1.4')))
TEXTS = (
    dict(file='publisher_about.html', url=PREFIX+'about.html', bytes=8984,
         sha256='29ec54c0554a4e12c4f695afd7f6bbe7a77db411090ee3031268950ea38a3a7b'),
    dict(file='publisher_api.html', url=PREFIX+'api.html', bytes=8197,
         sha256='43a80892dbc5f5ca6215ad75f7bf3d57c510ddd7444d2d9025165f52e80f7774'))


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check(deadline):
    rt.require(time.monotonic() < deadline, 'Inclusive metadata deadline reached')


def source(code, revision):
    proof = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    modes = {str(p.relative_to(code)): [stat.S_IMODE(p.lstat().st_mode), p.lstat().st_uid, p.lstat().st_gid]
             for p in (code, *sorted(code.rglob('*')))}
    return dict(binding=proof, modes_identity=pin(json.dumps(modes, sort_keys=True).encode()))


def config(code, proof):
    cfg = rt.pinned(code/CONFIG, proof['binding']['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.visual_genome_metadata_acquire.v1', entry=ENTRY, output=str(DATA),
        budget_seconds=600, outer_seconds=615, workers=2, request_timeout_seconds=15,
        max_total_compressed_bytes=140 << 20, max_expanded_asset_bytes=1 << 30,
        max_total_expanded_bytes=2 << 30, assets=list(ASSETS), publisher_texts=list(TEXTS),
        source_helper_pin=dict(bytes=23559, sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
        retry_count=0, no_redirects=True, no_json_values_consulted=True, no_rgb=True)
    rt.require(type(cfg) is dict and json.dumps(cfg, sort_keys=True) == json.dumps(fixed, sort_keys=True),
               'Exact frozen metadata-only protocol required')
    rt.require(proof['binding']['helpers'][HELPERS[-1]] == cfg['source_helper_pin']
        and Path(rt.__file__).resolve() == code/HELPERS[-1], 'Actual unchanged source helper required')
    return cfg


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Publisher redirect rejected without fallback')


class Transfer:
    def __init__(self, cfg, deadline, opener=None):
        self.cfg, self.deadline, self.received, self.expanded_reserved = cfg, deadline, 0, 0
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def check(self):
        check(self.deadline); rt.require(not self.cancelled.is_set(), 'Failed peer acquisition cancelled')

    def reserve_expansion(self, size):
        self.check()
        with self.lock:
            rt.require(self.expanded_reserved+size <= self.cfg['max_total_expanded_bytes'], 'Total expansion budget before member read')
            self.expanded_reserved += size

    def download(self, row, output, *, text=False):
        """Never overwrite; failed writes remove only this opened regular inode."""
        self.check()
        wanted = row['bytes'] if text else row['observed_bytes']
        request = urllib.request.Request(row['url'], headers={'Accept-Encoding': 'identity'})
        with self.opener.open(request, timeout=min(self.cfg['request_timeout_seconds'],
                                                  self.deadline-time.monotonic())) as response:
            h = response.headers
            rt.require(response.status == 200 and response.geturl() == row['url']
                and h.get('Content-Encoding', 'identity').lower() == 'identity'
                and h.get('Content-Length') == str(wanted)
                and ('html' in h.get('Content-Type', '') if text else h.get_content_type() == 'application/zip'),
                'Exact public response headers required before body read')
            path = rt.canonical(output/row['file']); digest = hashlib.sha256(); size = 0
            with path.open('xb') as stream:
                owned = os.fstat(stream.fileno())
                try:
                    os.fchmod(stream.fileno(), 0o600)
                    while True:
                        self.check(); block = response.read(min(1 << 20, wanted-size+1))
                        if not block: break
                        with self.lock:
                            self.received += len(block)
                            rt.require(self.received <= self.cfg['max_total_compressed_bytes'], 'Total public-byte budget')
                        size += len(block); rt.require(size <= wanted, 'Declared original body size exceeded')
                        digest.update(block); stream.write(block)
                    identity = dict(bytes=size, sha256=digest.hexdigest())
                    rt.require(size == wanted and (not text or identity == {k:row[k] for k in ('bytes','sha256')}),
                               'Original response bytes/text SHA differ')
                    stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o400)
                    sync(output); self.check()
                except BaseException:
                    now = path.lstat()
                    rt.require(stat.S_ISREG(now.st_mode) and now.st_nlink == 1
                        and (now.st_dev,now.st_ino) == (owned.st_dev,owned.st_ino), 'Foreign failed publication retained')
                    path.unlink(); raise
        rt.require(rt.identity(path, 140 << 20) == identity, 'Saved original download changed')
        return identity


def qualify_zip(path, row, cfg, deadline, *, reserve=None, checkpoint=None):
    """Stream original CRC/decompressed bytes only. No JSON parse/extract/test selection."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        rt.require(len(entries) == 1, 'Exactly one published metadata member required')
        info = entries[0]; mode = info.external_attr >> 16
        rt.require(info.filename == info.orig_filename == row['member'] and not info.is_dir()
            and not info.flag_bits & (1 | 0x40) and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            and (stat.S_IFMT(mode) in (0, stat.S_IFREG)) and not info.external_attr & 0x10
            and 0 < info.file_size <= cfg['max_expanded_asset_bytes'] and 0 < info.compress_size <= row['observed_bytes'],
            'Unsafe name/type/encryption/compression/expanded size rejected')
        if reserve is not None: reserve(info.file_size)
        checkpoint = checkpoint or (lambda:check(deadline))
        size = 0; digest = hashlib.sha256()
        with archive.open(info, 'r') as stream:
            while True:
                checkpoint(); raw = stream.read(1 << 20)
                if not raw: break
                size += len(raw); rt.require(size <= info.file_size, 'Expanded member size exceeded')
                digest.update(raw)
        checkpoint(); rt.require(size == info.file_size, 'Complete CRC-checked original member required')
        return dict(member=info.filename, expanded_bytes=size, expanded_sha256=digest.hexdigest(),
                    compressed_member_bytes=info.compress_size, crc32=f'{info.CRC:08x}', crc_stream_verified=True,
                    compression=info.compress_type, json_values_consulted=False, expanded_file_written=False)


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish_report(out, report, deadline, *, started=None):
    """Open once; late seal/fsync failure demotes via this owned FD, never replaces."""
    with (out/'report.json').open('x+b') as stream:
        os.fchmod(stream.fileno(), 0o400)
        try:
            out.chmod(0o500); sync(out); sync(out.parent)
            report['outputs_sealed'] = True
            if started is not None: report['elapsed_seconds'] = time.monotonic()-started
            stream.write((json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno()); sync(out); sync(out.parent); check(deadline)
        except BaseException:
            report.update(status='fail', decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY',
                          publication_failed=True)
            report['outputs_sealed'] = stat.S_IMODE(out.lstat().st_mode) == 0o500
            if started is not None: report['elapsed_seconds'] = time.monotonic()-started
            stream.seek(0); stream.truncate()
            stream.write((json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno())
            raise


def run():
    started = time.monotonic()
    rt.require(os.geteuid() == 0 and sys.platform == 'linux' and os.uname().nodename == 'world-reward-ncc-h100-02',
               'Exact Azure VM02 CPU host required')
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual immutable driver origin required')
    before = source(code, revision); cfg = config(code, before); deadline = started+cfg['budget_seconds']
    out = rt.canonical(DATA); rt.require(not out.exists() and out.parent.is_dir(), 'Fresh absent metadata namespace required')
    check(deadline); out.mkdir(mode=0o700); out.chmod(0o700); sync(out.parent)
    transfer = Transfer(cfg, deadline); saved = {}; failure = None
    report = dict(schema=cfg['schema'], stage='publisher_texts', status='fail', producer_revision=revision,
        source_binding=before, configuration_identity=before['binding']['helpers'][CONFIG], outputs=saved,
        publisher_archive_sha256_verified=False, archive_identity_basis='first_read_official_url_and_observed_length',
        publisher_metadata_grant='CC-BY-4.0', underlying_image_rights_verified=False,
        annotation_values_consulted=False, historical_references_read=False, RGB_read=False, GPU_used=False,
        models_loaded=False, cohort_selected=False, adopted=False, source_rehashed_after=False, outputs_sealed=False)
    handlers = {s:signal.signal(s, lambda *_: (_ for _ in ()).throw(TimeoutError('Metadata acquisition interrupted')))
                for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    try:
        for row in TEXTS: saved[row['file']] = transfer.download(row, out, text=True)
        report['stage'] = 'metadata_archives'
        def worker(row):
            identity = transfer.download(row, out)
            return row['file'], dict(identity=identity, safety=qualify_zip(out/row['file'], row, cfg, deadline,
                                    reserve=transfer.reserve_expansion, checkpoint=transfer.check),
                                    url=row['url'], publisher_version=row['version'])
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            futures = [pool.submit(worker, row) for row in ASSETS]
            try:
                for future in as_completed(futures):
                    name, value = future.result(); saved[name] = value
            except BaseException:
                transfer.cancelled.set()
                for future in futures: future.cancel()
                raise
        rt.require(sum(saved[r['file']]['safety']['expanded_bytes'] for r in ASSETS) <= cfg['max_total_expanded_bytes'],
                   'Total metadata expansion budget exceeded')
        report.update(status='pass', stage='complete_metadata_acquisition', decision='METADATA_QUALIFIED_PENDING_SEPARATE_CENSUS')
    except BaseException as exc:
        failure = exc; report.update(status='fail', error_type=type(exc).__name__ if type(exc).__name__ in
            ('ValueError','TimeoutError','HTTPError','URLError','OSError','BadZipFile','RuntimeError') else 'RuntimeError',
            decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY')
    finally:
        try:
            rt.require(source(code, revision) == before, 'Original source/modes changed after acquisition')
            report['source_rehashed_after'] = True
            # Completed original ZIPs can remain when CRC fails; inventory every owned leaf honestly.
            actual = {}
            for p in out.iterdir():
                info = p.lstat()
                rt.require(stat.S_IMODE(info.st_mode) == 0o400 and info.st_uid == os.geteuid(),
                           'Owned400 metadata artifacts required')
                actual[p.name] = rt.identity(p, 140 << 20)
            expected = {r['file'] for r in (*TEXTS,*ASSETS)}
            rt.require(set(actual) <= expected, 'Foreign metadata output retained; no cleanup permitted')
            for name, row in saved.items():
                rt.require(actual[name] == row.get('identity', row), 'Saved metadata changed after acquisition')
            report['artifact_identities'] = actual; check(deadline)
        except BaseException:
            report.update(status='fail', decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY', postcheck_failed=True)
            failure = failure or ValueError('Metadata source/output postcheck failed')
        report.update(elapsed_seconds=time.monotonic()-started, public_bytes_received=transfer.received)
        try: publish_report(out, report, deadline, started=started)
        except BaseException: failure = failure or ValueError('Metadata publication failed')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
        print(json.dumps({k:report[k] for k in ('status','stage','decision','elapsed_seconds','public_bytes_received')}, sort_keys=True))
    if failure is not None: raise ValueError('Closed metadata acquisition; see sealed report') from None
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'Only the frozen single metadata attempt is permitted')
    run()

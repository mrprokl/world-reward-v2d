"""Frozen Azure metadata acquisition; annotation/split bytes remain opaque."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_genome_metadata_acquire as vg

rt = vg.rt
ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/vcoco_metadata_v1')
ENTRY = 'run_vcoco_metadata_acquire'
HELPERS = ('infra/vcoco_metadata_acquire.py', 'infra/run_vcoco_metadata_acquire.sh',
           'infra/visual_genome_metadata_acquire.py', 'infra/mediapipe_cpu_runtime_verify.py')
HELPER_PINS = {
    HELPERS[2]: dict(bytes=15022, sha256='e6f36ee12240b379ddcb52ade7b9e9d413c9309995a73ca8f7df0e3c34695ada'),
    HELPERS[3]: dict(bytes=23559, sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2')}
COMMIT = '489cc4db74f2f10ab4b134f67da3874afbf245ab'
TREE = '517cee7d005cf57fa4e6e5f5bc1eaeb3b3118ac7'
RAW = 'https://raw.githubusercontent.com/s-gupta/v-coco/'+COMMIT+'/'


def git_asset(path, size, blob, sha256=None):
    return dict(file=path.replace('/', '__'), url=RAW+path, bytes=size, git_blob_sha1=blob,
                sha256=sha256, mime='text/plain')


NOTICES = tuple(git_asset(*row) for row in (
    ('README.md', 4031, 'b0eb626353f539fe038b7d24d7b060dd5752fbee', '08894a6d5304a49d0586c9ffeef31a87021d264d2a40d7dbe663c585e72ef5a0'),
    ('LICENSE', 1133, '3b60fbdec51e758b598bd7a4a4e4c80f5b67f9ce', 'f9eef0d2092d17798a6e11738be62c2ecd210bc52c9854d0d9ad5a38a2815c14'),
    ('LICENSE_fast_rcnn', 1107, 'fde68786da5359ce20e221a88cf8e4ee5455e05e', 'dbfe88113cb11c3ae7ec02f7d344e2f2745574eb96a18afcdc4580ede3c8dd3f'),
    ('vsrl_utils.py', 5985, 'ebf1a1284e420c62d1f59f96bce7de56ddd61164', '5b8ae544d79dde56cdd8ee55c47271eed01ba1f0161c665977d5e7a8c32955ac'),
    ('vsrl_eval.py', 17878, '6a179214b65d4db992c0cd31ddd28aa94ce12090', 'eb6e765503bcc27fb73628476b641328e6f8bb26e4839f3f4645dc13afe72567'),
    ('script_pick_annotations.py', 2823, '993a663f33a94cb5b89cf66378f14c5795e5a5e3', 'adf1d6506a187a55ce08496e8d245dd351e784597c8aa2815a54ba2e22f31850')))
OPAQUE = tuple(git_asset(*row) for row in (
    ('data/vcoco/vcoco_train.json', 3135892, 'fa808965b680c2baee3bb5f9041335515ef0af5e'),
    ('data/vcoco/vcoco_val.json', 3590505, 'fbd0af34173838b506a5054e88d559e3dac217d0'),
    ('data/vcoco/vcoco_test.json', 6193877, 'c633fc3aeca04ed7b14c8b5bf7ff5b68bb9552f8'),
    ('data/vcoco/vcoco_trainval.json', 6722166, '017b349e9097ad3a194a3591e549380942f94876'),
    ('data/splits/vcoco_all.ids', 70521, 'f8dad90366d67d0df64f1e3c0c6f0b39885b1590'),
    ('data/splits/vcoco_test.ids', 33736, 'd21ff6d2a17844d62fc6532145347be8c92b6617'),
    ('data/splits/vcoco_train.ids', 17251, 'ad772f302bd5f6b15a083f7a5cd0a002e62550db'),
    ('data/splits/vcoco_trainval.ids', 36785, '98a7e9505db8e3e508757d14497c57a96599b6a5'),
    ('data/splits/vcoco_val.ids', 19534, '7ee718f17d34b5c88816b607f044e16393a37286')))
COCO_RAW = 'https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/'
COCO_TEXTS = (
    dict(file='publisher_coco_download.html', url=COCO_RAW+'download.htm', bytes=11453,
         sha256='fbc44eef7f0fd9a62786b15f9ea66b58754c6f3455a18aa3a8562639e425d5cc', git_blob_sha1=None, mime='text/plain'),
    dict(file='publisher_coco_terms.html', url=COCO_RAW+'termsofuse.htm', bytes=2370,
         sha256='bd019f88ee44c29b2f19c5b99888cf5bc2e7c16f57b6e52af8ed3a60462e8bdd', git_blob_sha1=None, mime='text/plain'))
ARCHIVE = dict(file='annotations_trainval2014.zip',
    url='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',
    bytes=252872794, sha256=None, git_blob_sha1=None, mime='application/zip')
# Publisher central catalogue only: one bounded 206 range, no JSON/decompression.
CATALOGUE = tuple(dict(member='annotations/'+name+'.json', compressed_bytes=compressed,
    expanded_bytes=expanded, crc32=crc, external_attr=attr, compression=8, flags=0)
    for name, compressed, expanded, crc, attr in (
        ('instances_train2014', 107297879, 332556225, '3cd50ca9', 2176057344),
        ('instances_val2014', 51955093, 160682675, '264f83bf', 2176057344),
        ('person_keypoints_train2014', 50041387, 170733465, '23466a04', 2175008768),
        ('person_keypoints_val2014', 23951481, 81637509, '6976f7d7', 2175008768),
        ('captions_train2014', 13124596, 66782097, '35a1fcdb', 2176057344),
        ('captions_val2014', 6501124, 32421077, '8ef135f3', 2176057344)))
TAIL = dict(bytes=4096, sha256='38c634460cccfad846182e4b24d721978afcfcec8c3675c34857a9417849139f')
PROTOCOL = dict(schema='world_reward.vcoco_metadata_acquire.v1', entry=ENTRY, output=str(DATA),
    budget_seconds=600, outer_seconds=615, workers=1, request_timeout_seconds=15,
    max_total_compressed_bytes=300 << 20, max_expanded_asset_bytes=512 << 20,
    max_total_expanded_bytes=1 << 30, max_host_memory_bytes=8 << 30, retry_count=0,
    source_commit=COMMIT, source_tree=TREE, helper_pins=HELPER_PINS,
    notices=list(NOTICES+COCO_TEXTS), opaque_git_assets=list(OPAQUE), archive=ARCHIVE,
    catalogue=list(CATALOGUE), catalogue_tail_identity=TAIL,
    no_redirects=True, no_json_values_consulted=True, no_rgb=True)


def source(code, revision):
    proof = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(all(proof['helpers'][k] == v for k, v in HELPER_PINS.items())
        and Path(vg.__file__).resolve() == code/HELPERS[2]
        and Path(rt.__file__).resolve() == code/HELPERS[3], 'Actual unchanged helper closure required')
    modes = {str(p.relative_to(code)): [stat.S_IMODE(p.lstat().st_mode), p.lstat().st_uid, p.lstat().st_gid]
             for p in (code, *sorted(code.rglob('*')))}
    return dict(binding=proof, modes_identity=vg.pin(json.dumps(modes, sort_keys=True).encode()))


def check_directory(out, owned):
    current = out.lstat()
    rt.require(stat.S_ISDIR(current.st_mode) and stat.S_IMODE(current.st_mode) == 0o700
        and (current.st_dev, current.st_ino, current.st_uid, current.st_gid) ==
        (owned.st_dev, owned.st_ino, owned.st_uid, owned.st_gid), 'Owned original output directory required')


class Transfer(vg.Transfer):
    def download(self, row, output):
        """VG streaming lifecycle, with exact raw-Git MIME/blob qualification."""
        self.check(); wanted = row['bytes']
        rt.require(type(wanted) is int and 0 < wanted <= self.cfg['max_total_compressed_bytes'],
                   'Positive bounded original asset length required')
        request = urllib.request.Request(row['url'], headers={'Accept-Encoding': 'identity'})
        with self.opener.open(request, timeout=min(self.cfg['request_timeout_seconds'],
                                                  self.deadline-time.monotonic())) as response:
            h = response.headers
            rt.require(response.status == 200 and response.geturl() == row['url']
                and h.get('Content-Encoding', 'identity').lower() == 'identity'
                and h.get('Content-Length') == str(wanted) and h.get_content_type() == row['mime'],
                'Exact public headers required before opaque bytes')
            path = rt.canonical(output/row['file']); sha = hashlib.sha256()
            blob = hashlib.sha1(b'blob '+str(wanted).encode()+b'\x00'); size = 0
            with path.open('xb') as stream:
                owned = os.fstat(stream.fileno())
                try:
                    os.fchmod(stream.fileno(), 0o600)
                    while True:
                        self.check(); block = response.read(min(1 << 20, wanted-size+1))
                        if not block: break
                        self.received += len(block); size += len(block)
                        rt.require(size <= wanted and self.received <= self.cfg['max_total_compressed_bytes'],
                                   'Original and total opaque byte bounds')
                        sha.update(block); blob.update(block); stream.write(block)
                    identity = dict(bytes=size, sha256=sha.hexdigest())
                    rt.require(size == wanted and (row['sha256'] is None or sha.hexdigest() == row['sha256'])
                        and (row['git_blob_sha1'] is None or blob.hexdigest() == row['git_blob_sha1']),
                        'Complete original bytes/Git blob or publisher text differ')
                    stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o400)
                    vg.sync(output); self.check()
                except BaseException:
                    now = path.lstat()
                    rt.require(stat.S_ISREG(now.st_mode) and now.st_nlink == 1
                        and (now.st_dev, now.st_ino) == (owned.st_dev, owned.st_ino),
                        'Foreign failed publication retained')
                    path.unlink(); raise
        rt.require(rt.identity(path, self.cfg['max_total_compressed_bytes']) == identity, 'Saved opaque bytes changed')
        self.check(); return identity


def qualify_archive(path, transfer):
    """Exact full catalogue; reserve expansion before any CRC-only member read."""
    transfer.check()
    with path.open('rb') as stream:
        stream.seek(-TAIL['bytes'], os.SEEK_END)
        rt.require(vg.pin(stream.read(TAIL['bytes'])) == TAIL, 'Original bounded catalogue tail differs')
    results = []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist(); catalogue = []
        for info in entries:
            mode = info.external_attr >> 16
            rt.require(info.orig_filename == info.filename and not info.is_dir()
                and stat.S_IFMT(mode) == stat.S_IFREG and not info.flag_bits & (1 | 0x40)
                and not info.external_attr & 0x10, 'Original regular nonencrypted member required')
            catalogue.append(dict(member=info.filename, compressed_bytes=info.compress_size,
                expanded_bytes=info.file_size, crc32=f'{info.CRC:08x}', external_attr=info.external_attr,
                compression=info.compress_type, flags=info.flag_bits))
        rt.require(catalogue == list(CATALOGUE), 'Exact original 2014 member catalogue required')
        for info in entries:
            rt.require(0 < info.file_size <= transfer.cfg['max_expanded_asset_bytes'], 'Expanded asset bound')
        transfer.reserve_expansion(sum(i.file_size for i in entries))
        for info in entries:
            sha = hashlib.sha256(); size = 0
            with archive.open(info) as stream:
                while True:
                    transfer.check(); raw = stream.read(1 << 20)
                    if not raw: break
                    size += len(raw); rt.require(size <= info.file_size, 'Original expanded size bound')
                    sha.update(raw)
            transfer.check(); rt.require(size == info.file_size, 'Complete original CRC stream required')
            results.append(dict(member=info.filename, expanded_bytes=size, expanded_sha256=sha.hexdigest(),
                crc32=f'{info.CRC:08x}', crc_stream_verified=True, json_values_consulted=False,
                expanded_file_written=False))
    return results


def run():
    started = time.monotonic()
    rt.require(os.geteuid() == 0 and sys.platform == 'linux' and os.uname().nodename == 'world-reward-ncc-h100-02',
               'Exact Azure VM02 CPU host required')
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual immutable driver origin required')
    before = source(code, revision); deadline = started+PROTOCOL['budget_seconds']
    out = rt.canonical(DATA); rt.require(not out.exists() and out.parent.is_dir(), 'Fresh metadata namespace required')
    vg.check(deadline); out.mkdir(mode=0o700); out.chmod(0o700); vg.sync(out.parent)
    owned_out = out.lstat()
    transfer = Transfer(PROTOCOL, deadline); saved = {}; failure = None
    report = dict(schema=PROTOCOL['schema'], producer_revision=revision, status='fail', stage='publisher_notices',
        source_binding=before, protocol_identity=vg.pin(json.dumps(PROTOCOL, sort_keys=True).encode()),
        outputs=saved, publisher_archive_sha256_verified=False, archive_identity_basis='first_read_publisher_S3_observed_size',
        source_commit=COMMIT, source_tree=TREE, annotation_license_interpretation='repository_wide_MIT_private_research_only',
        legal_certainty_claimed=False, notices_retained=False, underlying_image_rights_verified=False,
        training_overlap_verified=False, challenge_overlap_verified=False, annotation_values_consulted=False,
        historical_references_read=False, RGB_read=False, GPU_used=False, models_loaded=False, cohort_selected=False,
        adopted=False, source_rehashed_after=False, output_directory_rechecked=False, outputs_sealed=False)
    handlers = {s:signal.signal(s, lambda *_: (_ for _ in ()).throw(TimeoutError('Metadata acquisition interrupted')))
                for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    try:
        for row in NOTICES+COCO_TEXTS: saved[row['file']] = transfer.download(row, out)
        report.update(notices_retained=True, stage='opaque_role_and_split_files')
        for row in OPAQUE:
            saved[row['file']] = transfer.download(row, out)
        report.update(git_blob_sha1_verified=True, stage='original_coco2014_archive')
        saved[ARCHIVE['file']] = transfer.download(ARCHIVE, out)
        report['archive_members'] = qualify_archive(out/ARCHIVE['file'], transfer)
        report.update(status='pass', stage='complete_metadata_acquisition',
                      decision='METADATA_QUALIFIED_PENDING_SEPARATE_CENSUS')
    except BaseException as exc:
        failure = exc
        report.update(status='fail', error_type=type(exc).__name__ if type(exc).__name__ in
            ('ValueError','TimeoutError','HTTPError','URLError','OSError','BadZipFile','RuntimeError') else 'RuntimeError',
            decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY')
    finally:
        try:
            rt.require(source(code, revision) == before, 'Original source/modes changed after acquisition')
            report['source_rehashed_after'] = True
            check_directory(out, owned_out)
            report['output_directory_rechecked'] = True
            actual = {}
            for p in out.iterdir():
                info = p.lstat()
                rt.require(stat.S_IMODE(info.st_mode) == 0o400 and info.st_uid == os.geteuid(), 'Owned400 metadata artifacts required')
                actual[p.name] = rt.identity(p, PROTOCOL['max_total_compressed_bytes'])
            rt.require(set(actual) <= {r['file'] for r in NOTICES+COCO_TEXTS+OPAQUE+(ARCHIVE,)}, 'No foreign output cleanup allowed')
            rt.require(all(actual[name] == identity for name, identity in saved.items()), 'Saved metadata changed after acquisition')
            report['artifact_identities'] = actual; vg.check(deadline)
        except BaseException:
            report.update(status='fail', decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY', postcheck_failed=True)
            failure = failure or ValueError('Metadata source/output postcheck failed')
        report.update(elapsed_seconds=time.monotonic()-started, public_bytes_received=transfer.received)
        try:
            check_directory(out, owned_out)
            vg.publish_report(out, report, deadline, started=started)
        except BaseException:
            report.update(status='fail', decision='CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY')
            failure = failure or ValueError('Metadata publication failed')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
        print(json.dumps({k:report[k] for k in ('status','stage','decision','elapsed_seconds','public_bytes_received')}, sort_keys=True))
    if failure is not None: raise ValueError('Closed metadata acquisition; see sealed report') from None
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'Only the frozen single metadata attempt is permitted')
    run()

"""Frozen, metadata-first external RGB acquisition. Not prediction/quality QA.

The census is authenticated privately; the public cohort exposes only publisher
metadata. Failed slots stay failed. Original JPEG bytes are never transformed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_openimages_joint_pair_acquire'
CONFIG = 'configs/openimages_joint_pair_acquire_v1.json'
HELPERS = ('infra/openimages_joint_pair_acquire.py', 'infra/run_openimages_joint_pair_acquire.sh',
           'infra/mediapipe_cpu_runtime_verify.py', CONFIG)
CENSUS_ENTRY = 'run_openimages_joint_pair_census'
CENSUS_CONFIG = 'configs/openimages_joint_pair_census_v1.json'
CENSUS_HELPERS = ('infra/openimages_joint_pair_census.py', 'infra/run_openimages_joint_pair_census.sh',
                  'infra/mediapipe_cpu_runtime_verify.py', CENSUS_CONFIG)
CENSUS_REV = 'bf47f065c6e076d9a73e5084fe02d9b91cb45d9c'
CENSUS_PIN = dict(bytes=842858, sha256='106fbe042343f5ce3462c89245c2e02eaf4dfaabad145782cc4b06caf4df30d7')
HASH_NAMESPACE = 'world_reward.oi_joint_pair_v1/'
METADATA_KEYS = ('ImageID', 'OriginalLandingURL', 'OriginalURL', 'OriginalSize', 'OriginalMD5',
                 'Rotation', 'Author', 'AuthorProfileURL', 'Title')
LICENSE_URL = 'https://creativecommons.org/licenses/by/2.0/'


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, value):
    raw = encode(value); rt.write(path, raw, 0o444); sync_directory(path.parent)
    return pin(raw)


def profile_identity(url):
    parsed = urllib.parse.urlsplit(url)
    rt.require(parsed.scheme in ('http', 'https') and parsed.netloc == 'www.flickr.com'
        and not parsed.query and not parsed.fragment
        and re.fullmatch(r'/(?:photos|people)/[^/?#]+/?', parsed.path), 'Canonical publisher Flickr author required')
    # Flickr photos/people routes and a trailing slash name the same account.
    return parsed.netloc+'/'+parsed.path.rstrip('/').rsplit('/', 1)[1]


def photo_identity(url):
    match = re.fullmatch(r'https://www\.flickr\.com/photos/[^/?#]+/([0-9]+)/?', url)
    rt.require(match is not None, 'Exact publisher Flickr photo landing URL required')
    return match.group(1)


def excluded_photo_identity(url):
    """Metadata identity only: Flickr route variants never create a new photo.

    Accept publisher /in/set-... and direct /account/photoID routes for
    disjointness only, never for a creator request or acquisition URL. Rights
    require the unchanged exact HTTPS landing page and no redirects.
    """
    match = re.fullmatch(r'https://www\.flickr\.com/(?:photos/)?[^/?#]+/([0-9]+)(?:/in/set-[0-9]+)?/?', url)
    rt.require(match is not None, 'Exact publisher Flickr exclusion identity required')
    return match.group(1)


def md5_identity(value):
    raw = base64.b64decode(value, validate=True)
    rt.require(len(raw) == 16 and base64.b64encode(raw).decode() == value, 'Canonical original MD5 required')
    return value


def old_exclusions(groups):
    """Old rights metadata only; never old reference boxes/predictions/metrics."""
    excluded = {k: set() for k in ('ids', 'md5', 'url', 'photo')}
    for expected, value in groups:
        ids, rows = value['selected_ids'], value['public_metadata']
        rt.require(len(ids) == len(set(ids)) == len(rows) == expected
            and {r['ImageID'] for r in rows} == set(ids)
            and not excluded['ids'].intersection(ids), 'Exact old disjoint metadata cohort required')
        excluded['ids'].update(ids)
        for row in rows:
            if row['OriginalMD5']:
                excluded['md5'].add(md5_identity(row['OriginalMD5']))
            excluded['url'].add(row['OriginalURL'])
            excluded['photo'].add(excluded_photo_identity(row['OriginalLandingURL']))
    return excluded


def select_cohort(records, excluded, *, slots=64, dev=32):
    """Rank once, before accessibility/rights requests; no replacements later."""
    rt.require(type(records) is list and 0 < dev < slots, 'Fixed split and census records required')
    seen = set(); eligible = []
    for record in records:
        iid = record['image_id']; row = record['publisher_metadata']
        rt.require(re.fullmatch('[0-9a-f]{16}', iid) and iid not in seen and row['ImageID'] == iid
            and type(record['eligible']) is bool, 'Unique original census image identity required')
        seen.add(iid)
        if record['eligible']:
            rt.require(row['Rotation'] == '0.0', 'Eligible publisher rotation must be zero')
            eligible.append(row)
    eligible.sort(key=lambda r: (hashlib.sha256((HASH_NAMESPACE+r['ImageID']).encode()).hexdigest(), r['ImageID']))
    authors = set(); hashes = set(); photos = set(); urls = set(); selected = []
    for row in eligible:
        if not row['OriginalMD5'] or not row['AuthorProfileURL']:
            continue
        digest = md5_identity(row['OriginalMD5']); author = profile_identity(row['AuthorProfileURL'])
        photo = excluded_photo_identity(row['OriginalLandingURL']); url = row['OriginalURL']
        if (row['ImageID'] in excluded['ids'] or digest in excluded['md5'] or url in excluded['url']
                or photo in excluded['photo'] or author in authors or digest in hashes or photo in photos or url in urls):
            continue
        metadata = {k: row[k] for k in METADATA_KEYS}
        rt.require(all(type(v) is str for v in metadata.values()), 'Original textual publisher metadata required')
        selected.append(dict(slot=len(selected), split='DEV' if len(selected) < dev else 'TEST', publisher_metadata=metadata))
        authors.add(author); hashes.add(digest); photos.add(photo); urls.add(url)
        if len(selected) == slots:
            break
    rt.require(len(selected) == slots, 'Insufficient untouched author/photo-disjoint slots; no requests allowed')
    return selected


def configuration(code):
    path = code/CONFIG; value = rt.strict(path.read_bytes())
    expected = dict(schema='world_reward.openimages_joint_pair_acquire.v1',
        census=dict(path='results/openimages-joint-pair-census-v1/report.json', **CENSUS_PIN, producer_revision=CENSUS_REV),
        output='/srv/world-reward-data/openimages_joint_pair_acquisition_v1',
        selection_namespace='results/openimages-joint-pair-selection-v1', hash_namespace=HASH_NAMESPACE,
        slots=64, dev=32, test=32, workers=4, request_timeout=15, overall=300,
        max_image_bytes=16 << 20, max_creator_page_bytes=2 << 20, max_total_image_bytes=1 << 30,
        max_decoded_pixels=16 << 20, no_replacements=True, challenge_inputs_used=False,
        training_overlap_verified=False, challenge_overlap_verified=False, quality_verified=False, adoption=False)
    rt.require(value == expected and all(type(value[k]) is type(v) for k, v in expected.items()),
               'Exact predeclared acquisition configuration required')
    return value, rt.identity(path, 16 << 10)


def authenticate(root, code, revision):
    source = rt.source(root, code, revision, ENTRY, HELPERS); cfg, cfg_pin = configuration(code)
    census_path = root/cfg['census']['path']; census = rt.pinned(census_path, CENSUS_PIN)
    old_code = root/'jobs'/CENSUS_REV/CENSUS_ENTRY/'code'
    old_source = rt.source(root, old_code, CENSUS_REV, CENSUS_ENTRY, CENSUS_HELPERS)
    old_config = rt.pinned(old_code/CENSUS_CONFIG, old_source['helpers'][CENSUS_CONFIG])
    expected = dict(schema='world_reward.openimages_joint_pair_census.v1', status='pass', producer_revision=CENSUS_REV,
        source_binding=old_source, frozen_inputs=old_config['files'], selection_performed=False,
        challenge_inputs_used=False, rgb_read=False, models_loaded=False, gpu_used=False, network_used=False,
        source_and_inputs_rehashed_after=True, stage='new_joint_pair_metadata_feasibility')
    rt.require(all(type(census.get(k)) is type(v) and census[k] == v for k, v in expected.items())
        and census['counts']['old_ids_excluded'] == 144, 'Actual original census/source PASS required')
    frozen = {code/CONFIG: cfg_pin, census_path: CENSUS_PIN}; groups = []
    for suffix, count in (('openimages_holds_census_v1/census_v2.json', 16),
                           ('openimages_holds_census_fresh128_v1/census.json', 128)):
        names = [n for n in old_config['files'] if n.endswith('/'+suffix)]
        rt.require(len(names) == 1, 'Exact old cohort rights metadata path required')
        path = Path(names[0]); expected_pin = old_config['files'][names[0]]
        rt.require(rt.identity(path, 1 << 20, readonly=False) == expected_pin, 'Old cohort metadata bytes differ')
        value = rt.strict(path.read_bytes()); groups.append((count, value)); frozen[path] = expected_pin
    exclusions = old_exclusions(groups)
    rt.require(len(exclusions['ids']) == 144, 'All old 144 identities must be excluded')
    return cfg, census['records'], exclusions, source, old_source, frozen


class ExactRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('Redirect not allowed')


def fetch(url, maximum, deadline, request_timeout=15):
    remaining = deadline-time.monotonic(); rt.require(remaining > 0, 'Acquisition deadline reached')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), ExactRedirect())
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 WorldRewardRightsAudit', 'Accept-Encoding': 'identity'})
    with opener.open(request, timeout=min(request_timeout, remaining)) as response:
        rt.require(response.status == 200 and response.url == url, 'Exact public HTTPS response required')
        blocks = []; size = 0
        while True:
            rt.require(time.monotonic() < deadline, 'Acquisition deadline reached')
            block = response.read(min(65536, maximum+1-size))
            if not block:
                break
            size += len(block); rt.require(size <= maximum, 'Publisher response exceeds byte bound'); blocks.append(block)
        return b''.join(blocks)


def creator_rights(raw, row):
    landing = row['OriginalLandingURL']; photo_identity(landing)
    blocks = re.findall(r'<script[^>]+type=["\x27]application/ld\+json["\x27][^>]*>(.*?)</script>',
                        raw.decode('utf-8'), re.S | re.I)
    nodes = []
    for block in blocks:
        value = rt.strict(block)
        for node in value if isinstance(value, list) else [value]:
            if isinstance(node, dict):
                nodes.append(node); nodes.extend(node.get('@graph', []))
    matched = [n for n in nodes if isinstance(n, dict) and n.get('@type') == 'ImageObject' and n.get('acquireLicensePage') == landing]
    rt.require(len(matched) == 1, 'Exactly one matching creator ImageObject declaration required')
    obj = matched[0]
    rt.require(obj.get('license') == LICENSE_URL and type(obj.get('author')) is dict
        and obj['author'].get('name') == row['Author'] and bool(row['Author']), 'Exact creator name and CC-BY-2.0 grant required')
    return dict(schema='world_reward.external_image_rights.v4', image_id=row['ImageID'],
        creator_page=dict(url=landing, **pin(raw)), creator_declaration=dict(acquireLicensePage=landing,
        license=LICENSE_URL, author_name=row['Author']), license='CC-BY-2.0', annotations_license='CC-BY-4.0',
        individual_creator_declaration_verified=True, attribution=row['Author']+' — '+row['Title']+' — '+landing)


def jpeg_header(raw, maximum_pixels):
    """Header qualification only; not a decoded-content or orientation proof."""
    rt.require(raw[:2] == b'\xff\xd8', 'Original JPEG SOI required'); offset = 2
    for _ in range(4096):
        rt.require(offset < len(raw) and raw[offset] == 255, 'Valid JPEG marker required')
        while offset < len(raw) and raw[offset] == 255:
            offset += 1
        rt.require(offset < len(raw), 'Truncated JPEG marker'); marker = raw[offset]; offset += 1
        rt.require(marker not in (0, 0xd9, 0xda), 'JPEG frame header required before scan')
        if marker == 1 or 0xd0 <= marker <= 0xd7:
            continue
        rt.require(offset+2 <= len(raw), 'Truncated JPEG segment')
        size = int.from_bytes(raw[offset:offset+2], 'big')
        rt.require(size >= 2 and offset+size <= len(raw), 'Bounded JPEG segment required')
        if marker in (0xc0, 0xc1, 0xc2):
            rt.require(size == 17 and raw[offset+2] == 8 and raw[offset+7] == 3, '8-bit three-channel JPEG header required')
            height = int.from_bytes(raw[offset+3:offset+5], 'big'); width = int.from_bytes(raw[offset+5:offset+7], 'big')
            rt.require(width > 0 and height > 0 and width*height <= maximum_pixels, 'Original JPEG dimensions exceed bound')
            return dict(width=width, height=height, channels=3, header_only=True, decoded_content_verified=False)
        offset += size
    raise ValueError('JPEG marker count exceeds bound')


def acquire_slot(selected, output, cfg, deadline, *, request=fetch):
    row = selected['publisher_metadata']; folder = output/row['ImageID']; folder.mkdir(mode=0o755)
    result = dict(slot=selected['slot'], split=selected['split'], image_id=row['ImageID'], status='unavailable',
        publisher_metadata=row, creator_grant_verified=False, publisher_md5_matched=False,
        reference_geometry_read=False, training_overlap_verified=False, challenge_overlap_verified=False)
    phase = 'publisher_metadata'
    try:
        rt.require(time.monotonic() < deadline and row['Rotation'] == '0.0', 'Frozen deadline/zero rotation required')
        photo_identity(row['OriginalLandingURL']); profile_identity(row['AuthorProfileURL'])
        phase = 'creator_rights'
        rights = creator_rights(request(row['OriginalLandingURL'], cfg['max_creator_page_bytes'], deadline, cfg['request_timeout']), row)
        result.update(rights_pin=publish(folder/'rights.json', rights), creator_grant_verified=True)
        phase = 'original_rgb'
        rt.require(re.fullmatch(r'https://(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com/[^?#]+\.jpg', row['OriginalURL']),
                   'Exact original static Flickr JPEG URL required')
        rt.require(re.fullmatch('[0-9]+', row['OriginalSize']), 'Original integer byte size required')
        size = int(row['OriginalSize']); rt.require(0 < size <= cfg['max_image_bytes'], 'Original image exceeds byte bound')
        raw = request(row['OriginalURL'], size, deadline, cfg['request_timeout'])
        rt.require(len(raw) == size and base64.b64encode(hashlib.md5(raw).digest()).decode() == md5_identity(row['OriginalMD5']),
                   'Original size/MD5 differs; no transformed substitute allowed')
        header = jpeg_header(raw, cfg['max_decoded_pixels'])
        rt.require(time.monotonic() < deadline, 'Acquisition deadline reached before original publication')
        rt.write(folder/'rgb.jpg', raw, 0o444); sync_directory(folder)
        result.update(status='acquired', image_pin=pin(raw), publisher_md5_matched=True,
                      image_transformation='none_original_file', rotation='0.0', jpeg_header=header)
    except Exception as exc:
        result.update(failed_phase=phase, error_type=type(exc).__name__,
                      reason='fixed_slot_not_qualified_no_retry_or_replacement')
    finally:
        result['record_pin'] = publish(folder/'record.json', result); folder.chmod(0o555); sync_directory(output)
    return result


def acquire_cohort(selected, output, cfg, deadline, cohort_path, cohort_pin, *, request=fetch):
    rt.require(rt.identity(cohort_path, 1 << 20) == cohort_pin and not cohort_path.parent.stat().st_mode & 0o222,
               'Sealed cohort must exist before any rights/RGB request')
    rt.require(len(selected) == cfg['slots'] and [r['slot'] for r in selected] == list(range(cfg['slots'])),
               'Every ordered frozen slot required before requests')
    rt.require(cfg['slots']*cfg['max_image_bytes'] <= cfg['max_total_image_bytes'], 'Worst-case original-byte budget exceeded')
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        records = list(pool.map(lambda row: acquire_slot(row, output, cfg, deadline, request=request), selected))
    rt.require(len(records) == cfg['slots'] and sum(r.get('image_pin', {}).get('bytes', 0) for r in records)
               <= cfg['max_total_image_bytes'], 'Every frozen slot and aggregate byte budget required')
    return records


def run(mode):
    started = time.monotonic()
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02',
               'Exact Azure CPU acquisition host required')
    revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual immutable acquisition driver required')
    cfg, records, exclusions, source, census_source, frozen = authenticate(ROOT, code, revision)
    selected = select_cohort(records, exclusions, slots=cfg['slots'], dev=cfg['dev'])
    selection_dir = rt.canonical(ROOT/cfg['selection_namespace']); output = rt.canonical(Path(cfg['output']))
    rt.require(not selection_dir.exists() and not output.exists() and selection_dir.parent.is_dir()
               and output.parent.is_dir(), 'Fresh selection and acquisition namespaces required')
    selection_dir.mkdir(mode=0o755)
    cohort = dict(schema='world_reward.openimages_joint_pair_cohort.v1', producer_revision=revision,
        source_binding=source, configuration_identity=frozen[code/CONFIG], census_identity=CENSUS_PIN,
        hash_namespace=HASH_NAMESPACE, records=selected, no_replacements=True, rights_requests_before_freeze=0,
        reference_geometry_exposed=False, old_identity_and_photo_exclusion_verified=True,
        challenge_inputs_used=False, training_overlap_verified=False, challenge_overlap_verified=False,
        ownership_verified=False, task_target_verified=False, quality_verified=False, adoption=False)
    cohort_path = selection_dir/'cohort.json'; cohort_pin = publish(cohort_path, cohort)
    selection_dir.chmod(0o555); sync_directory(selection_dir.parent)
    report = dict(schema=cfg['schema'], stage='frozen_external_rgb_acquisition', status='fail', mode=mode,
        producer_revision=revision, source_binding=source, census_source_binding=census_source,
        census_identity=CENSUS_PIN, configuration_identity=frozen[code/CONFIG], cohort_identity=cohort_pin,
        cohort_path=str(cohort_path), records=[], no_replacements=True, reference_geometry_read=False,
        individual_image_license_verified=False, training_overlap_verified=False, challenge_overlap_verified=False,
        challenge_inputs_used=False, ownership_verified=False, task_target_verified=False, quality_verified=False,
        adoption=False, gpu_used=False, models_loaded=False, local_heavy_transfer=False,
        network_used=mode == 'acquire', retry_count=0, replacement_count=0,
        creator_name_and_landing_verified_only=True, creator_account_identity_independently_verified=False,
        full_image_decoding_performed=False,
        source_and_inputs_rehashed_after=False, artifacts_rehashed_after=False)
    destination = selection_dir if mode == 'selection-only' else output
    if mode == 'acquire':
        output.mkdir(mode=0o755)
    def interrupted(*_args):
        raise TimeoutError('Acquisition interrupted; frozen slots are not replaced')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        if mode == 'acquire':
            # Reserve one socket timeout for draining outstanding reads/posthash;
            # this is an engineering bound, not an accessibility-based selection.
            report['records'] = acquire_cohort(selected, output, cfg, started+cfg['overall']-cfg['request_timeout'], cohort_path, cohort_pin)
        for row in report['records']:
            folder = output/row['image_id']; expected_files = {'record.json'}
            stored = rt.pinned(folder/'record.json', row['record_pin'])
            rt.require(stored == {k: v for k, v in row.items() if k != 'record_pin'}, 'Frozen per-slot record differs')
            for field, name in (('rights_pin', 'rights.json'), ('image_pin', 'rgb.jpg')):
                if field in row:
                    rt.require(rt.identity(folder/name, cfg['max_image_bytes']) == row[field], 'Original acquired artifact differs')
                    expected_files.add(name)
            rt.require({p.name for p in folder.iterdir()} == expected_files and not folder.stat().st_mode & 0o222,
                       'Exclusive sealed per-slot artifacts required')
        cfg2, _, _, source2, census2, frozen2 = authenticate(ROOT, code, revision)
        rt.require((cfg2, source2, census2, frozen2) == (cfg, source, census_source, frozen)
                   and rt.identity(cohort_path, 1 << 20) == cohort_pin, 'Original source/cohort/control bytes changed after acquisition')
        report.update(source_and_inputs_rehashed_after=True, artifacts_rehashed_after=True)
        rt.require(time.monotonic()-started <= cfg['overall'], 'Inclusive acquisition budget exceeded')
        report.update(status='pass', counts=dict(slots=cfg['slots'], dev=cfg['dev'], test=cfg['test'],
            acquired=sum(r['status'] == 'acquired' for r in report['records']),
            unavailable=sum(r['status'] != 'acquired' for r in report['records']),
            original_rgb_bytes=sum(r.get('image_pin', {}).get('bytes', 0) for r in report['records'])),
            decision='FROZEN_ACQUISITION_ONLY_NO_QUALITY_OR_ELIGIBILITY_CLAIM')
    except BaseException as exc:
        report.update(status='fail', error_type=type(exc).__name__)
        raise
    finally:
        try:
            report['elapsed_seconds'] = time.monotonic()-started
            manifest_pin = publish(destination/'manifest.json', report); destination.chmod(0o555); sync_directory(destination.parent)
            print(json.dumps(dict(status=report['status'], counts=report.get('counts'), cohort=cohort_pin,
                                  manifest=manifest_pin, elapsed_seconds=report['elapsed_seconds']), sort_keys=True))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--acquire', action='store_true'); modes.add_argument('--selection-only', action='store_true')
    args = parser.parse_args(); run('acquire' if args.acquire else 'selection-only')

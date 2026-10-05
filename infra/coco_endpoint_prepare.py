"""Fresh COCO endpoint census/acquisition; no model or historical reference read.

Reuse only immutable original metadata/JPEG/identity helpers. The new cohort is
not a rescue of closed proposal experiments or an unseen-pretraining benchmark.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coco_proposal_prepare as old

rt, acq = old.rt, old.acq
ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/coco_endpoint_v2')
ENTRY = 'run_coco_endpoint_prepare'
CONFIG = 'configs/coco_endpoint_v2.json'
HELPERS = ('infra/coco_endpoint_prepare.py', 'infra/run_coco_endpoint_prepare.sh', CONFIG,
           *old.HELPERS)
NAMESPACE = 'world_reward.coco_endpoint_v2/'
CACHE_REV = '4657c8b45f733a1043c5a8af2f5b9d6ac027196a'
encode, pin, write, check = old.encode, old.pin, old.write, old.check


def configuration(code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.coco_endpoint_prepare.v2', output=str(DATA),
        slots=64, dev=32, reserved=32, workers=6, overall=300, request_timeout=15,
        max_image_bytes=16 << 20, max_decoded_pixels=16 << 20,
        max_total_image_bytes=1 << 30, min_acquired_dev=24, retry_count=0,
        no_replacements=True, hash_namespace=NAMESPACE, archive_url=old.BUCKET+'annotations/'+old.ARCHIVE,
        archive_bytes=252907541, archive_md5='f4bbac642086de4f52a3fdda2de5fa2c', member=old.MEMBER)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k,v in fixed.items()), 'Frozen endpoint recipe differs')
    rt.require(set(cfg['reused_helper_pins']) == set(old.HELPERS), 'Complete unchanged original helper pins required')
    for name in old.HELPERS:
        rt.require(source['helpers'][name] == cfg['reused_helper_pins'][name], 'Original helper/config bytes differ')
    for module,name in ((old,old.HELPERS[0]),(rt,old.HELPERS[3]),(acq,old.HELPERS[4])):
        rt.require(Path(module.__file__).resolve() == code/name, 'Imported original helper origin differs')
    rt.require(cfg['annotation_cache']['producer_revision'] == CACHE_REV, 'Original cache producer differs')
    return cfg


def cached_archive(cfg):
    cache = cfg['annotation_cache']; code = ROOT/'jobs'/CACHE_REV/old.ENTRY/'code'
    source = rt.source(ROOT, code, CACHE_REV, old.ENTRY, old.HELPERS)
    original_cfg = rt.pinned(code/old.CONFIG, source['helpers'][old.CONFIG], 16 << 10)
    expected_root = Path('/srv/world-reward-data/coco_proposal_v1/metadata')
    rt.require(cache['archive_path'] == str(expected_root/old.ARCHIVE)
        and cache['report']['path'] == str(expected_root/'report.json')
        and cfg['closed_coco_cohort']['path'] == str(expected_root/'cohort.json'), 'Exact original cache paths required')
    rt.require(all(source['helpers'][n] == cfg['reused_helper_pins'][n] for n in old.HELPERS)
        and original_cfg['historical'] == cfg['historical'], 'Independently bound original census source/config differs')
    rp = cache['report']; path = rt.canonical(rp['path']); receipt = rt.pinned(path, rp['pin'], 1 << 20)
    expected = dict(schema='world_reward.coco_proposal_prepare.v1', phase='census', status='pass',
        producer_revision=CACHE_REV, source_binding=source, configuration_identity=source['helpers'][old.CONFIG],
        archive_url=cfg['archive_url'], archive_publisher_md5=cfg['archive_md5'],
        archive_identity=cache['archive_pin'], archive_path=cache['archive_path'],
        cohort_identity=cfg['closed_coco_cohort']['pin'], freeze_before_rgb=True, rgb_read=False,
        source_and_inputs_rehashed_after=True, outputs_sealed=True)
    rt.require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in expected.items()), 'Actual original census PASS required')
    archive = rt.canonical(cache['archive_path'])
    rt.require(rt.identity(archive, cfg['archive_bytes']) == cache['archive_pin'], 'Independent whole archive SHA/bytes differ')
    return receipt, source, {str(path):rp['pin'], str(archive):cache['archive_pin']}


def historical(cfg):
    values, frozen = [], {}
    rt.require(len(cfg['historical']) == 4, 'All four historical OI cohorts required')
    for row in cfg['historical']:
        path = rt.canonical(row['path'])
        rt.require(rt.identity(path, 1 << 20, readonly=row['readonly']) == row['pin'], 'Historical OI metadata differs')
        values.append(rt.strict(path.read_bytes())); frozen[str(path)] = row['pin']
    excluded = old.exclusions(values); wanted = cfg['closed_coco_cohort']; path = rt.canonical(wanted['path'])
    cohort = rt.pinned(path, wanted['pin'], 1 << 20); frozen[str(path)] = wanted['pin']
    rt.require(cohort['schema'] == 'world_reward.coco_proposal_cohort.v1'
        and cohort['producer_revision'] == CACHE_REV and cohort['freeze_before_rgb'] is True
        and cohort['no_replacements'] is True and cohort['hash_namespace'] == old.NAMESPACE,
        'Closed original COCO metadata freeze required')
    rows = cohort['records']
    rt.require(len(rows) == 32 and [r['slot'] for r in rows] == list(range(32))
        and [r['split'] for r in rows] == ['DEV']*16+['RESERVED']*16
        and len({r['image_id'] for r in rows}) == len({r['photo_id'] for r in rows}) == 32,
        'All closed COCO slots including reserved/missing required')
    excluded['coco_ids'] = set()
    for row in rows:
        rt.require(set(row) == {'slot','split','image_id','image','photo_id','reference_file','reference_identity'}
            and old.photo_identity(row['image']['flickr_url']) == row['photo_id']
            and row['image']['id'] == row['image_id']
            and row['reference_file'] == f"reference_{row['slot']:06d}.json", 'Exact old metadata slots required')
        reference = path.parent/row['reference_file']
        rt.require(rt.identity(reference, 4 << 20) == row['reference_identity'], 'Closed reference hash differs; values never parsed')
        frozen[str(reference)] = row['reference_identity']
        excluded['photos'].add(row['photo_id']); excluded['coco_ids'].add(row['image_id'])
    return excluded, frozen, cohort


def select(value, excluded):
    """Filter historical photos before consulting their annotation contents."""
    licenses, categories = value['licenses'], value['categories']
    rt.require(len({r['id'] for r in licenses}) == len(licenses)
        and len({r['id'] for r in categories}) == len(categories), 'Unique native catalogs required')
    grants = [r for r in licenses if r['id'] == 4]; persons = [r['id'] for r in categories if r['name'] == 'person']
    rt.require(len(grants) == len(persons) == 1 and grants[0]['url'] == old.LICENSE_URL, 'Original CC-BY2/person catalog required')
    images, bank, photos, seen = {}, {}, {}, set()
    for image in value['images']:
        iid = image['id']; rt.require(type(iid) is int and 0 <= iid < 10**12 and iid not in seen, 'Unique native image IDs required'); seen.add(iid)
        if image['license'] != 4 or iid in excluded['coco_ids']: continue
        photo = old.photo_identity(image['flickr_url'])
        if photo in excluded['photos']: continue
        images[iid], bank[iid], photos[iid] = image, [], photo
    categories = {r['id'] for r in categories}; ann_ids = set()
    for a in value['annotations']:
        if a['image_id'] not in bank: continue  # No bbox/role values of closed photos.
        rt.require(type(a['id']) is int and a['id'] not in ann_ids and a['category_id'] in categories, 'Unique bound native annotations required'); ann_ids.add(a['id'])
        if old.countable(a): bank[a['image_id']].append(a)
    eligible = []
    for iid,image in images.items():
        refs = bank[iid]; people = [a for a in refs if a['category_id'] == persons[0]]; objects = [a for a in refs if a['category_id'] != persons[0]]
        if len(people) < 2 or len(objects) < 2 or not old.separated(people) or not old.separated(objects): continue
        rt.require(type(image['width']) is int and type(image['height']) is int and image['width'] > 0 and image['height'] > 0
            and image['width']*image['height'] <= 16 << 20 and image['file_name'] == f'{iid:012d}.jpg'
            and image['coco_url'] == f'http://images.cocodataset.org/val2017/{iid:012d}.jpg', 'Original val grid/filename required')
        eligible.append(dict(image_id=iid,image=image,photo_id=photos[iid],annotations=refs))
    eligible.sort(key=lambda r:(hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest(),r['image_id']))
    rows, used = [], set()
    for r in eligible:
        if r['photo_id'] in used: continue
        rows.append(dict(slot=len(rows), split='DEV' if len(rows) < 32 else 'RESERVED', **r)); used.add(r['photo_id'])
        if len(rows) == 64: break
    return rows, dict(eligible=len(eligible), selected=len(rows), historical_slots=272,
        excluded_OI_ids=len(excluded['ids']), excluded_COCO_ids=len(excluded['coco_ids']), excluded_unique_photo_keys=len(excluded['photos']))


def census(cfg, source, revision, metadata, deadline):
    receipt, cache_source, cache_files = cached_archive(cfg); excluded, frozen, closed = historical(cfg)
    rt.require(closed['source_binding'] == cache_source
        and closed['configuration_identity'] == receipt['configuration_identity'], 'Closed cohort source differs')
    raw = Path(cfg['annotation_cache']['archive_path']).read_bytes(); check(deadline)
    rt.require(hashlib.md5(raw).hexdigest() == cfg['archive_md5'] and pin(raw) == cfg['annotation_cache']['archive_pin'], 'Original archive MD5/SHA differs')
    value, member_pin = old.val_metadata(raw); rows, counts = select(value, excluded); check(deadline)
    if len(rows) != 64:
        rt.require(cached_archive(cfg) == (receipt,cache_source,cache_files)
            and historical(cfg) == (excluded,frozen,closed), 'Inputs changed during insufficient-capacity census')
        return dict(counts=counts, original_census_source=cache_source,
            frozen_inputs={**cache_files,**frozen}, archive_identity=cfg['annotation_cache']['archive_pin'],
            member_identity=member_pin, rgb_read=False, network_used=False, freeze_before_rgb=False,
            decision='INCONCLUSIVE_CLOSED_INSUFFICIENT_METADATA')
    for row in rows:
        row['reference_file'] = f"reference_{row['slot']:06d}.json"
        row['reference_identity'] = write(metadata/row['reference_file'], dict(annotations=row.pop('annotations')))
    cohort = dict(schema='world_reward.coco_endpoint_cohort.v2', producer_revision=revision, source_binding=source,
        configuration_identity=source['helpers'][CONFIG], records=rows, categories=value['categories'], licenses=value['licenses'],
        hash_namespace=NAMESPACE, freeze_before_rgb=True, no_replacements=True, author_identity='UNKNOWN', author_disjointness_verified=False)
    cp = write(metadata/'cohort.json', cohort)
    rt.require(cached_archive(cfg) == (receipt,cache_source,cache_files) and historical(cfg) == (excluded,frozen,closed), 'Original inputs changed after freeze')
    for row in rows: rt.require(rt.identity(metadata/row['reference_file'], 4 << 20) == row['reference_identity'], 'New frozen reference bytes differ')
    return dict(cohort_identity=cp, counts=counts, archive_identity=cfg['annotation_cache']['archive_pin'], member_identity=member_pin,
        original_census_report_identity=cfg['annotation_cache']['report']['pin'], original_census_source=cache_source,
        frozen_inputs={**cache_files,**frozen}, freeze_before_rgb=True, rgb_read=False, network_used=False,
        decision='FROZEN64_PENDING_SEPARATE_ACQUISITION')


def frozen_cohort(cfg, source, metadata, report_pin, cohort_pin):
    report = rt.pinned(metadata/'report.json', report_pin, 2 << 20); cohort = rt.pinned(metadata/'cohort.json', cohort_pin, 2 << 20)
    receipt, old_source, cache_files = cached_archive(cfg); excluded, frozen, closed = historical(cfg)
    rt.require(report['schema'] == cfg['schema'] and report['phase'] == 'census' and report['status'] == 'pass'
        and report['source_binding'] == source and report['configuration_identity'] == source['helpers'][CONFIG]
        and report['cohort_identity'] == cohort_pin and report['source_and_inputs_rehashed_after'] is True
        and report['outputs_sealed'] is True and report['freeze_before_rgb'] is True and report['rgb_read'] is False
        and report['network_used'] is False and report['frozen_inputs'] == {**cache_files,**frozen}
        and report['original_census_source'] == old_source and closed['source_binding'] == old_source
        and report['original_census_report_identity'] == cfg['annotation_cache']['report']['pin']
        and report['archive_identity'] == cfg['annotation_cache']['archive_pin'], 'Authentic independent census freeze required')
    rt.require(cohort['schema'] == 'world_reward.coco_endpoint_cohort.v2' and cohort['source_binding'] == source
        and cohort['configuration_identity'] == source['helpers'][CONFIG] and cohort['hash_namespace'] == NAMESPACE
        and cohort['producer_revision'] == source['producer_revision']
        and report['producer_revision'] == source['producer_revision']
        and cohort['freeze_before_rgb'] is True and cohort['no_replacements'] is True, 'New cohort provenance differs')
    rows = cohort['records']
    rt.require(len(rows) == 64 and [r['slot'] for r in rows] == list(range(64))
        and [r['split'] for r in rows] == ['DEV']*32+['RESERVED']*32
        and len({r['image_id'] for r in rows}) == len({r['photo_id'] for r in rows}) == 64, 'Complete untouched64 slots required')
    for r in rows:
        rt.require(set(r) == {'slot','split','image_id','image','photo_id','reference_file','reference_identity'}
            and r['image']['id'] == r['image_id'] and r['image']['license'] == 4
            and r['image']['file_name'] == f"{r['image_id']:012d}.jpg"
            and r['image']['coco_url'] == f"http://images.cocodataset.org/val2017/{r['image_id']:012d}.jpg"
            and old.photo_identity(r['image']['flickr_url']) == r['photo_id'] and r['photo_id'] not in excluded['photos']
            and r['image_id'] not in excluded['coco_ids'] and r['reference_file'] == f"reference_{r['slot']:06d}.json"
            and rt.identity(metadata/r['reference_file'], 4 << 20) == r['reference_identity'], 'Exact fresh metadata/reference hashes required; no reference decoding')
    return rows, excluded, report


def acquire(rows, excluded, cfg, public, deadline, *, request=acq.fetch):
    """One official original request per frozen slot, no retries/aliases/caps."""
    rt.require(len(rows) == cfg['slots'] == 64, 'Exactly64 frozen slots required')
    def slot(r):
        result = dict(slot=r['slot'], split=r['split'], image_id=r['image_id'], status='unavailable'); phase = 'http'; created = False
        path = public/f"image_{r['slot']:06d}.jpg"
        try:
            check(deadline); image = r['image']; name = f"{r['image_id']:012d}.jpg"
            rt.require(image['file_name'] == name and image['license'] == 4, 'Original filename/grant required')
            raw = request(old.BUCKET+'val2017/'+name, cfg['max_image_bytes'], deadline, cfg['request_timeout']); phase = 'jpeg_grid'
            rt.require(0 < len(raw) <= cfg['max_image_bytes'], 'Bounded original JPEG required')
            digest = base64.b64encode(hashlib.md5(raw).digest()).decode()
            rt.require(digest not in excluded['md5'], 'Historical original bytes forbidden')
            header = acq.jpeg_header(raw, cfg['max_decoded_pixels'])
            rt.require((header['width'],header['height']) == (image['width'],image['height']), 'Original JPEG/metadata grid differs'); check(deadline)
            image_pin = pin(raw); phase = 'publish'; rt.write(path,raw); created = True
            rt.require(rt.identity(path,cfg['max_image_bytes']) == image_pin, 'Original publication bytes differ')
            result.update(status='acquired',image_pin=image_pin,jpeg_header=header,original_md5=digest)
        except Exception as exc:
            if created:
                rt.canonical(path); rt.require(path.is_file() and path.stat().st_nlink == 1, 'Cannot remove foreign output'); path.unlink()
            result.update(reason=phase,error_type=type(exc).__name__ if type(exc) in (ValueError,OSError,TimeoutError) else 'other')
        return result
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: records = list(pool.map(slot,rows))
    check(deadline); seen = {}; images, mappings = [], []
    for r in records:
        if r['status'] == 'acquired': seen[r['original_md5']] = seen.get(r['original_md5'],0)+1
    for r in records:
        if r['status'] != 'acquired': continue
        file = f"image_{r['slot']:06d}.jpg"
        if seen[r['original_md5']] > 1:
            path = public/file; rt.require(rt.identity(path,cfg['max_image_bytes']) == r['image_pin'], 'Cannot remove foreign duplicate'); path.unlink()
            r.update(status='unavailable',reason='duplicate_cohort_bytes')
            for k in ('image_pin','jpeg_header','original_md5'): del r[k]
            continue
        opaque = hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()[:32]
        images.append(dict(image_id=opaque,file=file,**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
        mappings.append(dict(slot=r['slot'],public_image_id=opaque,public_file=file,image_pin=r['image_pin']))
    rt.require(sum(r['bytes'] for r in images) <= cfg['max_total_image_bytes'], 'Total original RGB byte budget exceeded')
    pp = write(public/'manifest.json', dict(schema='world_reward.rgb_proposal_inputs.v1',images=images))
    counts = dict(slots=64,acquired=len(images),missing=64-len(images),acquired_DEV=sum(r['status']=='acquired' and r['split']=='DEV' for r in records),
        acquired_RESERVED=sum(r['status']=='acquired' and r['split']=='RESERVED' for r in records))
    return dict(records=records,public_mappings=mappings,public_inputs_identity=pp,counts=counts,
        decision='READY_PENDING_SEPARATE_NATIVE_ENDPOINTS' if counts['acquired_DEV'] >= cfg['min_acquired_dev'] else 'INCONCLUSIVE_CLOSED_NO_MODEL')


def verify_public(public, result, cfg):
    value = rt.pinned(public/'manifest.json', result['public_inputs_identity'], 1 << 20)
    acquired = [r for r in result['records'] if r['status'] == 'acquired']; expected = []
    rt.require(len(result['records']) == 64 and [r['slot'] for r in result['records']] == list(range(64))
        and len(acquired) == len(result['public_mappings']) and len({m['public_image_id'] for m in result['public_mappings']}) == len(acquired), 'Whole64 acquisition ledger required')
    for r,m in zip(acquired,result['public_mappings']):
        rt.require(m == dict(slot=r['slot'],public_file=f"image_{r['slot']:06d}.jpg",image_pin=r['image_pin'],
            public_image_id=hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()[:32])
            and rt.identity(public/m['public_file'], cfg['max_image_bytes']) == r['image_pin'], 'Exact opaque original slot mapping required')
        expected.append(dict(image_id=m['public_image_id'],file=m['public_file'],**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
    rt.require(value == dict(schema='world_reward.rgb_proposal_inputs.v1',images=expected)
        and all(set(r) == old.PUBLIC_KEYS for r in expected) and len(expected) == result['counts']['acquired']
        and {p.name for p in public.iterdir()} == {'manifest.json'}|{r['file'] for r in expected}
        and all(p.lstat().st_uid == os.geteuid() and stat.S_IMODE(p.lstat().st_mode) == 0o400 for p in public.iterdir()), 'No fake/private/foreign public nodes allowed')


def run(phase, pins=()):
    started = time.monotonic(); deadline = started+300
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure CPU root host required')
    revision,code = os.environ['WR_CODE_REVISION'],Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Immutable endpoint driver required')
    source = rt.source(ROOT,code,revision,ENTRY,HELPERS); cfg = configuration(code,source); metadata = DATA/'metadata'
    if phase == 'census':
        rt.require(not rt.canonical(DATA).exists() and DATA.parent.is_dir(), 'Fresh census namespace required'); DATA.mkdir(mode=0o700); metadata.mkdir(mode=0o700); output = metadata
    else:
        report_pin,cohort_pin = pins; rows,excluded,census_report = frozen_cohort(cfg,source,metadata,report_pin,cohort_pin)
        rt.require(not (DATA/'inputs').exists() and not (DATA/'eval_private').exists(), 'Fresh acquisition namespace required')
        DATA.chmod(0o700); (DATA/'inputs').mkdir(mode=0o700); (DATA/'eval_private').mkdir(mode=0o700); output = DATA/'eval_private'
    report = dict(schema=cfg['schema'],phase=phase,status='fail',producer_revision=revision,source_binding=source,
        configuration_identity=source['helpers'][CONFIG],source_and_inputs_rehashed_after=False,outputs_sealed=False,
        models_loaded=False,gpu_used=False,predictions_read=False,challenge_inputs_used=False,
        network_used=phase=='acquire',retry_count=0,replacement_count=0,author_identity='UNKNOWN',
        author_disjointness_verified=False,creator_account_identity_verified=False,training_overlap_verified=False,
        image_license='publisher_recorded_CC-BY-2.0',annotations_license='CC-BY-4.0',raw_redistribution_allowed_by_this_job=False,
        quality_verified=False,adopted=False,reference_values_decoded=phase=='census',closed_individual_reference_values_read=False,
        reference_values_read_for_acquisition=False,rgb_requests_before_freeze=0,original_image_transformations=0)
    report.update(whole_val_json_decoded=phase=='census',closed_annotation_rows_examined=False)
    def interrupted(*_): raise TimeoutError('Inclusive phase interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        if phase == 'census': report.update(census(cfg,source,revision,metadata,deadline))
        else:
            report.update(selected_slots=rows,census_report_identity=report_pin,cohort_identity=cohort_pin,freeze_before_rgb=True)
            report.update(acquire(rows,excluded,cfg,DATA/'inputs',deadline-15)); verify_public(DATA/'inputs',report,cfg)
            rt.require(frozen_cohort(cfg,source,metadata,report_pin,cohort_pin) == (rows,excluded,census_report), 'Original freeze changed during acquisition')
        rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS) == source and configuration(code,source) == cfg, 'Full source/config changed')
        check(deadline); report.update(source_and_inputs_rehashed_after=True)
        if report.get('decision') not in ('INCONCLUSIVE_CLOSED_INSUFFICIENT_METADATA','INCONCLUSIVE_CLOSED_NO_MODEL'):
            report['status'] = 'pass'
    except BaseException as exc:
        report['error_type'] = type(exc).__name__ if type(exc) in (ValueError,OSError,TimeoutError,KeyError,old.zipfile.BadZipFile) else 'other'
    finally:
        signal.alarm(0)
        with (output/('report.json' if phase=='census' else 'manifest.json')).open('xb') as stream:
            os.fchmod(stream.fileno(),0o400)
            for p in (*output.rglob('*'),*((DATA/'inputs').iterdir() if phase=='acquire' else ())):
                rt.canonical(p); rt.require(p.is_file() and p.lstat().st_nlink == 1, 'No foreign output node allowed'); p.chmod(0o400)
            for p in ((DATA/'inputs',output,DATA) if phase=='acquire' else (output,DATA)): p.chmod(0o500)
            acq.sync_directory(output); acq.sync_directory(DATA)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started)
            if time.monotonic() >= deadline: report.update(status='fail',error_type='TimeoutError')
            raw = encode(report); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if time.monotonic() >= deadline and report['status'] == 'pass':
                report.update(status='fail',error_type='TimeoutError',elapsed_seconds=time.monotonic()-started)
                raw=encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        for s,h in handlers.items(): signal.signal(s,h)
        print(encode(dict(status=report['status'],phase=phase,report_identity=pin(raw),cohort_identity=report.get('cohort_identity'),
            counts=report.get('counts'),decision=report.get('decision'))).decode(),end='')
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) in (2,6) and sys.argv[1] in ('--census','--acquire'), 'Only fixed phases and independent freeze pins allowed')
    phase = sys.argv[1][2:]; rt.require((phase=='census') == (len(sys.argv)==2), 'Acquire requires independent census report/cohort pins')
    pins = () if phase=='census' else (dict(bytes=int(sys.argv[2]),sha256=sys.argv[3]),dict(bytes=int(sys.argv[4]),sha256=sys.argv[5]))
    report = run(phase,pins); sys.exit(0 if report['status']=='pass' else 1)

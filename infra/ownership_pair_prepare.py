"""Fresh96 metadata freeze then creator-checked originals; no model/old references."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ownership_pair_census as old
import coco_endpoint_prepare as endpoint

rt, acq, encode, pin = old.rt, old.identities, old.encode, old.pin
ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/ownership_pair_v1')
ENTRY = 'run_ownership_pair_prepare'
CONFIG = 'configs/ownership_pair_prepare_v1.json'
NAMESPACE = 'world_reward.ownership_pair_v1/'
OLD_REV = '8d7b438d0ec2d27868756e20721fb70f15e5d147'
OLD_ENTRY = 'run_ownership_pair_census_v2'
OLD_HELPERS = (old.V1_HELPERS[0], 'infra/run_ownership_pair_census_v2.sh',
               'configs/ownership_pair_census_v2.json', *old.V1_HELPERS[3:], old.V1_CONFIG, old.V1_HELPERS[1])
REUSED = tuple(dict.fromkeys((old.V1_HELPERS[0], *old.V1_HELPERS[3:], *endpoint.HELPERS,
                            old.V1_CONFIG, 'configs/ownership_pair_census_v2.json', old.V1_HELPERS[1],
                            'infra/run_ownership_pair_census_v2.sh')))
HELPERS = ('infra/ownership_pair_prepare.py', 'infra/run_ownership_pair_prepare.sh', CONFIG, *REUSED)
SPLITS = ['FIT']*32+['CAL']*16+['RESERVED']*48
PUBLIC_KEYS = {'image_id', 'file', 'bytes', 'sha256', 'width', 'height'}


def check(deadline):
    rt.require(time.monotonic() < deadline, 'Inclusive ownership phase deadline reached')


def state(path):
    return list(old._state(path))


def source_proof(code, revision, entry, helpers):
    binding = rt.source(ROOT, code, revision, entry, helpers)
    modes = {str(p.relative_to(code)): stat.S_IMODE(p.lstat().st_mode) for p in (code, *sorted(code.rglob('*')))}
    return dict(binding=binding, modes_identity=pin(encode(modes)))


def configuration(code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 32 << 10)
    fixed = dict(schema='world_reward.ownership_pair_prepare.v1', output=str(DATA), hash_namespace=NAMESPACE,
        slots=96, fit=32, cal=16, reserved=48, select_seconds=180, acquire_seconds=300, workers=4,
        request_timeout=15, max_image_bytes=16 << 20, max_decoded_pixels=16 << 20,
        max_creator_page_bytes=2 << 20, max_total_image_bytes=96*(16 << 20),
        minimum_acquired=dict(FIT=24, CAL=12, RESERVED=36), retry_count=0, no_replacements=True,
        identity_policy=old.V2_POLICY)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in fixed.items()), 'Frozen96 phase recipe differs')
    rt.require(set(cfg['reused_helper_pins']) == set(REUSED)
        and all(source['helpers'][n] == cfg['reused_helper_pins'][n] for n in REUSED), 'Complete unchanged helper pins required')
    for module, name in ((old, old.V1_HELPERS[0]), (rt, old.V1_HELPERS[3]), (acq, old.V1_HELPERS[5]),
                         (old.census, old.V1_HELPERS[4]), (old.coco, old.V1_HELPERS[6]), (endpoint, endpoint.HELPERS[0])):
        rt.require(Path(module.__file__).resolve() == code/name, 'Imported immutable helper origin differs')
    row = cfg['original_census']
    rt.require(row['producer_revision'] == OLD_REV and row['path'] == str(ROOT/'results/ownership-pair-census-v2/report.json')
        and row['pin'] == dict(bytes=8286, sha256='dbcdd0e99b93805746317d6f626b110f500d2f2fc9162a7aa20ba870e87049ad')
        and row['configuration_identity'] == source['helpers'][OLD_HELPERS[2]], 'Exact original V2 PASS required')
    original_cfg = rt.pinned(code/OLD_HELPERS[2], row['configuration_identity'], 16 << 10)
    rt.require(all(cfg[k] == original_cfg[k] for k in ('files', 'historical', 'human_classes', 'body_part_classes', 'identity_policy')),
               'Original identity/classes/geometry inputs changed')
    return cfg, original_cfg


def coco_md5(ledger, cohort, count):
    """Acquisition metadata only; reference files/values and RGB never opened."""
    rows = ledger['records']; expected = cohort['records']
    rt.require(len(rows) == len(expected) == count and [r['slot'] for r in rows] == list(range(count))
        and [r['split'] for r in rows] == ['DEV']*(count//2)+['RESERVED']*(count//2), 'All historical COCO slots required')
    digests = set()
    for r, c in zip(rows, expected):
        rt.require(set(r) == {'image_id','image_pin','jpeg_header','original_md5','slot','split','status'}
            and r['status'] == 'acquired' and r['image_id'] == c['image_id'] and r['split'] == c['split']
            and r['slot'] == c['slot'] and set(r['image_pin']) == {'bytes','sha256'}
            and type(r['image_pin']['bytes']) is int and 0 < r['image_pin']['bytes'] <= 16 << 20
            and re.fullmatch('[0-9a-f]{64}', r['image_pin']['sha256'])
            and r['jpeg_header']['width'] == c['image']['width'] and r['jpeg_header']['height'] == c['image']['height'],
            'Original acquired COCO metadata differs')
        digest = acq.md5_identity(r['original_md5'])
        rt.require(digest not in digests, 'Unique original COCO MD5 required'); digests.add(digest)
    return digests


def authenticate(cfg, original_cfg):
    row = cfg['original_census']; code = ROOT/'jobs'/OLD_REV/OLD_ENTRY/'code'
    proof = source_proof(code, OLD_REV, OLD_ENTRY, OLD_HELPERS); source = proof['binding']
    receipt = rt.pinned(Path(row['path']), row['pin'], 16 << 10)
    rt.require(source['closure_sha256'] == row['source_closure_sha256']
        and (code.parent/'source-sha256').read_bytes() == (row['source_archive_sha256']+'\n').encode()
        and sum(p.is_file() for p in code.rglob('*')) == row['source_files'] == 291
        and source['helpers'][OLD_HELPERS[0]] == row['driver_identity']
        and source['helpers'][OLD_HELPERS[2]] == row['configuration_identity']
        and receipt['source_binding'] == source and receipt['producer_revision'] == OLD_REV
        and receipt['schema'] == 'world_reward.ownership_pair_census.v2' and receipt['status'] == 'pass'
        and receipt['counts'] == row['counts'] and receipt['capacity_gate_passed'] is True
        and receipt['configuration_identity'] == row['configuration_identity']
        and receipt['source_and_inputs_rehashed_after'] is True and receipt['outputs_sealed'] is True
        and receipt['invalid_identities_normalized'] is False and receipt['invalid_identities_rejected_not_repaired'] is True,
        'Actual unchanged metadata V2 receipt/source differs')
    prior = old._prior_failure(original_cfg)
    rt.require(receipt['original_closed_failure'] == prior, 'Original closed FAIL differs')
    files = [*cfg['files'].values(), *cfg['historical']]; paths = [rt.canonical(r['path']) for r in files]
    rt.require(len(set(paths)) == len(paths) == 10 and receipt['frozen_inputs'] == files, 'Original ten metadata inputs required')
    inputs = {}
    for p, r in zip(paths, files):
        rt.require(rt.identity(p, 100 << 20, readonly=r['readonly']) == r['pin'], 'Original metadata bytes differ')
        inputs[str(p)] = dict(pin=r['pin'], state=state(p))
    values = [rt.strict(p.read_bytes()) for p in paths[4:]]
    excluded, counts = old._historical_exclusions(values); additional = set(); coco_proofs = []
    for i, spec in enumerate(cfg['coco_acquisitions']):
        module = old.coco if i == 0 else endpoint; count = (32,64)[i]
        rev = ('4657c8b45f733a1043c5a8af2f5b9d6ac027196a','7b557290140dc97c839590c31155fbaf50e442a8')[i]
        expected_path = Path(cfg['historical'][4+i]['path']).parents[1]/'eval_private/manifest.json'
        rt.require(spec['producer_revision'] == rev and spec['path'] == str(expected_path)
            and spec['pin'] == ((dict(bytes=37509,sha256='6f3346be5a0ccb065084226ad6cdaa2423dd0d460427d43823b9635f57141ae6')),
                                (dict(bytes=72806,sha256='53de338a0c23409e12afa2a24b39538ff9873d40fc18c2d142b512790bf5ae10')))[i], 'Exact historical acquisition path/revision required')
        historic_code = ROOT/'jobs'/rev/module.ENTRY/'code'; cp = source_proof(historic_code, rev, module.ENTRY, module.HELPERS)
        rt.require(cp['binding']['closure_sha256'] == spec['source_closure_sha256'] and cp['binding']['entries'] == spec['source_entries']
            and all(cp['binding']['helpers'][n] == cfg['reused_helper_pins'][n] for n in module.HELPERS), 'Historical acquisition source differs')
        path = rt.canonical(spec['path']); ledger = rt.pinned(path, spec['pin'], 1 << 20); cohort = values[4+i]
        rt.require(ledger['schema'] == ('world_reward.coco_proposal_prepare.v1','world_reward.coco_endpoint_prepare.v2')[i]
            and ledger['phase'] == 'acquire' and ledger['status'] == 'pass' and ledger['producer_revision'] == rev
            and ledger['source_binding'] == cp['binding'] and ledger['configuration_identity'] == cp['binding']['helpers'][module.CONFIG]
            and ledger['cohort_identity'] == cfg['historical'][4+i]['pin'] and cohort['source_binding'] == cp['binding']
            and cohort['producer_revision'] == rev and ledger['source_and_inputs_rehashed_after'] is True
            and ledger['outputs_sealed'] is True and not path.parent.stat().st_mode & 0o222,
            'Actual complete COCO acquisition provenance differs')
        additional.update(coco_md5(ledger, cohort, count)); inputs[str(path)] = dict(pin=spec['pin'], state=state(path))
        coco_proofs.append(dict(source=cp, report_identity=spec['pin']))
    rt.require(len(cfg['coco_acquisitions']) == 2 and len(additional) == 96, 'All96 historical original MD5 required')
    return excluded, additional, dict(original_census_source=proof, original_closed_failure=prior,
        original_census_identity=row['pin'], original_census_state=state(Path(row['path'])),
        frozen_inputs=inputs, coco_acquisition_sources=coco_proofs, original_counts=counts)


def select_cohort(records, excluded):
    """New SHA-order gate; previous lexical132 does not imply96 in this order."""
    seen = set(); available = []
    for r in records:
        iid = r['image_id']; m = r['publisher_metadata']
        rt.require(type(r['eligible']) is bool and iid == m['ImageID'] and iid not in seen
            and re.fullmatch('[0-9a-f]{16}', iid), 'Unique original census record required'); seen.add(iid)
        if not r['eligible']: continue
        rt.require(set(acq.METADATA_KEYS) <= set(m) and all(type(m[k]) is str for k in acq.METADATA_KEYS)
            and m['Rotation'] == '0.0' and bool(m['OriginalMD5']) and bool(m['AuthorProfileURL']),
            'Original metadata/zero orientation/author/MD5 required')
        identity = old._metadata_identity(m)
        if iid in excluded['ids'] or any(v in excluded[k] for v,k in zip(identity,('authors','photos','md5','urls'))): continue
        available.append(m)
    available.sort(key=lambda m:(hashlib.sha256((NAMESPACE+m['ImageID']).encode()).hexdigest(),m['ImageID']))
    unique = [set() for _ in range(4)]; selected = []
    for m in available:
        identity = old._metadata_identity(m)
        if any(v in group for v, group in zip(identity,unique)): continue
        for v, group in zip(identity,unique): group.add(v)
        selected.append(dict(slot=len(selected),split=SPLITS[len(selected)],publisher_metadata={k:m[k] for k in acq.METADATA_KEYS}))
        if len(selected) == 96: break
    return selected, dict(new_sha_order_independent_slots=len(selected), eligible_after_extra_md5=len(available))


def freeze_selection(cfg, source, revision, original_cfg, metadata, deadline):
    excluded, extra, proof = authenticate(cfg, original_cfg); records = []
    counts = old.collect({k:Path(v['path']) for k,v in cfg['files'].items()}, cfg, excluded, lambda:check(deadline), records_out=records)
    combined = dict(proof['original_counts'], **counts)
    rt.require(combined == cfg['original_census']['counts'], 'Original139/132 counts/inventory changed')
    excluded['md5'].update(extra); post = old._capacity(records, excluded)
    if post['independent_slots_lower_bound'] >= 96:
        selected, ranked = select_cohort(records, excluded)
    else:
        selected, ranked = [], dict(new_sha_order_independent_slots=0,eligible_after_extra_md5=post['eligible_after_all_exclusions'])
    check(deadline)
    result = dict(original_counts=combined, post_coco_md5_counts=post, selection_counts=ranked,
        input_proof=proof, coco_original_md5_exclusions=96, freeze_before_rgb=False,
        capacity_gate_passed=len(selected) == 96, decision='CLOSED_CAPACITY_INCONCLUSIVE')
    if len(selected) == 96:
        cohort = dict(schema='world_reward.ownership_pair_cohort.v1',producer_revision=revision,source_binding=source,
            configuration_identity=source['helpers'][CONFIG],hash_namespace=NAMESPACE,records=selected,
            freeze_before_rgb=True,no_replacements=True,reference_values_exposed=False)
        result.update(cohort_identity=acq.publish(metadata/'cohort.json',cohort), freeze_before_rgb=True,
                      decision='FROZEN96_PENDING_INDIVIDUAL_RIGHTS')
    rt.require(authenticate(cfg, original_cfg)[2] == proof, 'Original sources/input modes changed after selection')
    return result


def frozen_cohort(cfg, source, original_cfg, report_pin, cohort_pin):
    report = rt.pinned(DATA/'metadata/report.json',report_pin,64 << 10)
    cohort = rt.pinned(DATA/'metadata/cohort.json',cohort_pin,128 << 10)
    excluded, extra, proof = authenticate(cfg,original_cfg); excluded['md5'].update(extra)
    rt.require(report['schema'] == cfg['schema'] and report['phase'] == 'select' and report['status'] == 'pass'
        and report['producer_revision'] == source['producer_revision'] and report['source_binding'] == source
        and report['configuration_identity'] == source['helpers'][CONFIG] and report['input_proof'] == proof
        and report['cohort_identity'] == cohort_pin and report['freeze_before_rgb'] is True
        and report['capacity_gate_passed'] is True and report['outputs_sealed'] is True
        and report['source_and_inputs_rehashed_after'] is True and report['network_used'] is False
        and cohort['schema'] == 'world_reward.ownership_pair_cohort.v1' and cohort['producer_revision'] == source['producer_revision']
        and cohort['source_binding'] == source and cohort['configuration_identity'] == source['helpers'][CONFIG]
        and cohort['hash_namespace'] == NAMESPACE and cohort['freeze_before_rgb'] is True and cohort['no_replacements'] is True
        and cohort['reference_values_exposed'] is False and not (DATA/'metadata').stat().st_mode & 0o222,
        'Independent sealed96 freeze differs')
    rows = cohort['records']; rt.require(len(rows) == 96 and [r['slot'] for r in rows] == list(range(96))
        and [r['split'] for r in rows] == SPLITS and all(set(r) == {'slot','split','publisher_metadata'} for r in rows), 'Exact96 metadata-only slots required')
    fake = [dict(image_id=r['publisher_metadata']['ImageID'],publisher_metadata=r['publisher_metadata'],eligible=True) for r in rows]
    rt.require(select_cohort(fake,excluded)[0] == rows, 'Frozen rank/order/identities differ')
    return rows, proof



def publish_owned(path, raw):
    """Exclusive bytes; on a write failure remove only the FD-owned inode."""
    rt.canonical(path)
    with path.open('xb') as stream:
        owned = os.fstat(stream.fileno())
        try:
            os.fchmod(stream.fileno(),0o444)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            rt.require(rt.identity(path,16 << 20) == pin(raw), 'Publication bytes differ')
            acq.sync_directory(path.parent)
        except BaseException:
            now = path.lstat()
            rt.require(stat.S_ISREG(now.st_mode) and now.st_nlink == 1
                and (now.st_dev,now.st_ino) == (owned.st_dev,owned.st_ino), 'Cannot remove foreign publication')
            path.unlink()
            try: acq.sync_directory(path.parent)
            except OSError: pass
            raise
    return pin(raw)

def acquire(rows, public, private, cfg, deadline, *, request=acq.fetch):
    rt.require(len(rows) == 96 and [r['slot'] for r in rows] == list(range(96)) and [r['split'] for r in rows] == SPLITS
        and 96*cfg['max_image_bytes'] <= cfg['max_total_image_bytes'], 'Complete fixed slots/byte budget required')
    def slot(r):
        m = r['publisher_metadata']; phase = 'publisher_metadata'; file = f"image_{r['slot']:06d}.jpg"
        result = dict(slot=r['slot'],split=r['split'],image_id=m['ImageID'],status='unavailable',creator_grant_verified=False)
        try:
            check(deadline); acq.photo_identity(m['OriginalLandingURL']); acq.profile_identity(m['AuthorProfileURL'])
            rt.require(m['Rotation'] == '0.0', 'Original zero rotation required'); phase = 'creator_rights'
            rights = acq.creator_rights(request(m['OriginalLandingURL'],cfg['max_creator_page_bytes'],deadline,cfg['request_timeout']),m)
            result.update(rights_file=f"rights_{r['slot']:06d}.json",rights_identity=publish_owned(private/f"rights_{r['slot']:06d}.json",encode(rights)),creator_grant_verified=True)
            phase = 'original_rgb'
            rt.require(re.fullmatch(r'https://(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com/[^?#]+\.jpg',m['OriginalURL'])
                and re.fullmatch('[0-9]+',m['OriginalSize']), 'Exact original static Flickr JPEG required')
            size = int(m['OriginalSize']); rt.require(0 < size <= cfg['max_image_bytes'], 'Original JPEG byte bound exceeded')
            raw = request(m['OriginalURL'],size,deadline,cfg['request_timeout']); digest = base64.b64encode(hashlib.md5(raw).digest()).decode()
            rt.require(len(raw) == size and digest == acq.md5_identity(m['OriginalMD5']), 'Original MD5/size differs')
            header = acq.jpeg_header(raw,cfg['max_decoded_pixels']); check(deadline)
            publish_owned(public/file,raw)
            result.update(status='acquired',image_pin=pin(raw),jpeg_header=header,original_md5=digest,public_file=file)
        except Exception as exc:
            result.update(failed_phase=phase,error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','OSError','TimeoutError','HTTPError','URLError','FileExistsError') else 'other',
                          reason='fixed_slot_no_retry_no_replacement')
        return result
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: records = list(pool.map(slot,rows))
    images = []
    for r in records:
        if r['status'] != 'acquired': continue
        opaque = hashlib.sha256((NAMESPACE+r['image_id']).encode()).hexdigest()[:32]
        images.append(dict(image_id=opaque,file=r['public_file'],**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
    rt.require(len({r['original_md5'] for r in records if r['status'] == 'acquired'}) == len(images)
        and sum(r['bytes'] for r in images) <= cfg['max_total_image_bytes'], 'Unique original bytes/aggregate budget required')
    counts = dict(slots=96,acquired=len(images),missing=96-len(images),**{'acquired_'+s:sum(r['status'] == 'acquired' and r['split'] == s for r in records) for s in ('FIT','CAL','RESERVED')})
    enough = all(counts['acquired_'+s] >= n for s,n in cfg['minimum_acquired'].items())
    return dict(records=records,counts=counts,public_inputs_identity=acq.publish(public/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=images)),
        acquisition_capacity_gate_passed=enough, decision='READY_PENDING_SEPARATE_NATIVE_BANK' if enough else 'CLOSED_ACQUISITION_CAPACITY_INCONCLUSIVE')


def verify_outputs(public, private, result, cfg):
    images = []
    expected_private = set()
    for r in result['records']:
        if r['creator_grant_verified']:
            path = private/r['rights_file']; rt.require(rt.identity(path,2 << 20) == r['rights_identity'], 'Creator rights bytes changed')
            expected_private.add(r['rights_file'])
        if r['status'] != 'acquired': continue
        rt.require(rt.identity(public/r['public_file'],cfg['max_image_bytes']) == r['image_pin'], 'Original public JPEG changed')
        images.append(dict(image_id=hashlib.sha256((NAMESPACE+r['image_id']).encode()).hexdigest()[:32],file=r['public_file'],
            **r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
    rt.require({p.name for p in private.iterdir()} == expected_private and {p.name for p in public.iterdir()} == {'manifest.json',*(r['file'] for r in images)}, 'Unexpected output path')
    value = rt.pinned(public/'manifest.json',result['public_inputs_identity'],128 << 10)
    rt.require(value == dict(schema='world_reward.rgb_proposal_inputs.v1',images=images)
        and all(set(r) == PUBLIC_KEYS for r in images), 'Opaque RGB-only public projection differs')


def run(phase, pins=()):
    started = time.monotonic(); budget = 180 if phase == 'select' else 300; deadline = started+budget
    rt.require(phase in ('select','acquire') and sys.platform == 'linux' and os.geteuid() == 0
        and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure metadata/acquisition host required')
    revision, code = os.environ['WR_CODE_REVISION'], Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Actual immutable driver required')
    source = rt.source(ROOT,code,revision,ENTRY,HELPERS); source_before = source_proof(code,revision,ENTRY,HELPERS)
    cfg, original_cfg = configuration(code,source)
    initial_proof = authenticate(cfg,original_cfg)[2]; check(deadline)
    if phase == 'select':
        rt.require(not rt.canonical(DATA).exists() and DATA.parent.is_dir(), 'Fresh96 namespace required')
        DATA.mkdir(mode=0o700); output = DATA/'metadata'; output.mkdir(mode=0o700)
    else:
        rt.require(len(pins) == 2, 'Independent select report/cohort pins required')
        rows, proof = frozen_cohort(cfg,source,original_cfg,*pins)
        rt.require(not (DATA/'inputs').exists() and not (DATA/'eval_private').exists(), 'Acquisition cannot overwrite existing data')
        DATA.chmod(0o700); (DATA/'inputs').mkdir(mode=0o700); output = DATA/'eval_private'; output.mkdir(mode=0o700)
    report = dict(schema=cfg['schema'],phase=phase,status='fail',producer_revision=revision,source_binding=source,
        configuration_identity=source['helpers'][CONFIG],source_and_inputs_rehashed_after=False,outputs_sealed=False,
        network_used=phase == 'acquire',metadata_geometry_consulted_for_selection=phase == 'select',
        historical_reference_values_read=False,historical_reference_files_opened=False,predictor_references_exposed=False,
        models_loaded=False,gpu_used=False,predictions_read=False,challenge_inputs_used=False,old_studies_reopened=False,
        retry_count=0,replacement_count=0,training_overlap_verified=False,ownership_verified=False,quality_verified=False,adopted=False,
        coco_author_disjointness_verified=False,creator_account_identity_verified=False,limitations=cfg['limitations'])
    def interrupted(*_): raise TimeoutError('Fixed phase interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        if phase == 'select': report.update(freeze_selection(cfg,source,revision,original_cfg,output,deadline))
        else:
            report.update(input_proof=proof,select_report_identity=pins[0],cohort_identity=pins[1],freeze_before_rgb=True)
            report.update(acquire(rows,DATA/'inputs',output,cfg,deadline-15)); verify_outputs(DATA/'inputs',output,report,cfg)
            rt.require(frozen_cohort(cfg,source,original_cfg,*pins) == (rows,proof), 'Sealed freeze/input proof changed')
        rt.require(source_proof(code,revision,ENTRY,HELPERS) == source_before and configuration(code,source) == (cfg,original_cfg), 'Source/config bytes or modes changed')
        check(deadline); report['source_and_inputs_rehashed_after'] = True
        report['status'] = 'pass' if report['decision'].startswith(('FROZEN96_','READY_')) else 'fail'
    except BaseException as exc:
        report['error_type'] = type(exc).__name__ if type(exc).__name__ in ('ValueError','OSError','TimeoutError','KeyError','FileExistsError') else 'other'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            rt.require(authenticate(cfg,original_cfg)[2] == initial_proof
                and source_proof(code,revision,ENTRY,HELPERS) == source_before, 'Source or input state changed after phase')
            report['source_and_inputs_rehashed_after'] = True
        except BaseException as exc:
            report.update(status='fail',source_and_inputs_rehashed_after=False,
                          error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','OSError','TimeoutError','KeyError') else 'other')
        path = output/('report.json' if phase == 'select' else 'manifest.json')
        with path.open('xb') as stream:
            os.fchmod(stream.fileno(),0o400)
            for p in output.rglob('*'):
                rt.canonical(p); rt.require(p.is_file() and p.lstat().st_nlink == 1, 'Foreign output node'); p.chmod(0o400)
            if phase == 'acquire':
                for p in (DATA/'inputs').iterdir(): rt.canonical(p); rt.require(p.is_file() and p.lstat().st_nlink == 1, 'Foreign public node'); p.chmod(0o400)
                (DATA/'inputs').chmod(0o500)
            output.chmod(0o500); DATA.chmod(0o500); acq.sync_directory(output); acq.sync_directory(DATA)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started)
            if time.monotonic() >= deadline: report.update(status='fail',error_type='TimeoutError')
            raw = encode(report); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if time.monotonic() >= deadline and report['status'] == 'pass':
                report.update(status='fail',error_type='TimeoutError',elapsed_seconds=time.monotonic()-started)
                raw = encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        for s,h in handlers.items(): signal.signal(s,h)
        print(encode(dict(status=report['status'],phase=phase,report_identity=pin(raw),cohort_identity=report.get('cohort_identity'),
                         decision=report.get('decision'),counts=report.get('counts'),selection_counts=report.get('selection_counts'))).decode(),end='')
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) in (2,6) and sys.argv[1] in ('--select','--acquire'), 'Fixed phases only')
    phase = sys.argv[1][2:]; rt.require((phase == 'select') == (len(sys.argv) == 2), 'Acquire requires independent select pins')
    pins = () if phase == 'select' else (dict(bytes=int(sys.argv[2]),sha256=sys.argv[3]),dict(bytes=int(sys.argv[4]),sha256=sys.argv[5]))
    result = run(phase,pins); sys.exit(0 if result['status'] == 'pass' else 1)

"""Fresh metadata-only VG capacity census; no selection, model or RGB access.

Three pinned primary texts are acquired once before semantic VG rows. Existing
metadata ZIPs are authenticated, streamed/CRC checked, never expanded to files.
Only new rights-eligible, nonhistorical images have object/edge values decoded.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metadata_json_stream as js
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_acquire as identities
import coco_proposal_prepare as coco

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/visual_genome_ownership_census_v1')
ENTRY = 'run_visual_genome_ownership_census'
CONFIG = 'configs/visual_genome_ownership_census_v1.json'
HELPERS = ('infra/visual_genome_ownership_census.py', 'infra/run_visual_genome_ownership_census.sh', CONFIG,
    'infra/metadata_json_stream.py', 'infra/mediapipe_cpu_runtime_verify.py',
    'infra/openimages_joint_pair_acquire.py', 'infra/coco_proposal_prepare.py', 'configs/ownership_pair_prepare_v1.json')
PERSON = frozenset('person persons man mans woman women womans womens boy boys girl girls human humans'.split())
PREDICATES = frozenset(('holding', 'holds'))
GRANT = 'http://creativecommons.org/licenses/by/2.0/'


def encode(value): return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
def pin(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
def check(deadline): rt.require(time.monotonic() < deadline, 'Inclusive census deadline reached')
def state(path):
    s = path.lstat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_nlink,s.st_uid,s.st_gid,s.st_mtime_ns,s.st_ctime_ns]


def source(code, revision, entry, helpers):
    binding = rt.source(ROOT, code, revision, entry, helpers)
    modes = {str(p.relative_to(code)): state(p) for p in (code,*sorted(code.rglob('*')))}
    legacy = {name:[value[3]&0o777,value[5],value[6]] for name,value in modes.items()}
    return dict(binding=binding,stat_identity=pin(encode(modes)),modes_identity=pin(json.dumps(legacy,sort_keys=True).encode()))


def authenticate(cfg, code, revision):
    current = source(code, revision, ENTRY, HELPERS); historical = {}
    for name, spec in cfg['sources'].items():
        old_code = ROOT/'jobs'/spec['revision']/spec['entry']/'code'
        value = source(old_code,spec['revision'],spec['entry'],tuple(spec['helpers']))
        rt.require(value['binding']['closure_sha256'] == spec['closure_sha256']
            and (spec['entries'] is None or value['binding']['entries'] == spec['entries'])
            and sum(p.is_file() for p in old_code.rglob('*')) == spec['files']
            and (old_code.parent/'source-sha256').read_bytes() == (spec['archive_xz_sha256']+'\n').encode(), 'Original source closure differs')
        historical[name] = value
    frozen = {}
    for name, row in cfg['inputs'].items():
        path = rt.canonical(row['path'])
        rt.require(rt.identity(path,row['maximum_bytes'],readonly=row['readonly']) == row['pin'], 'Frozen metadata bytes differ')
        frozen[name] = dict(pin=row['pin'],state=state(path))
    digest = hashlib.md5()
    with Path(cfg['inputs']['coco_archive']['path']).open('rb') as handle:
        for block in iter(lambda:handle.read(1 << 20),b''): digest.update(block)
    rt.require(digest.hexdigest() == 'f4bbac642086de4f52a3fdda2de5fa2c', 'Original publisher COCO MD5 differs')
    def value(name): return rt.strict(Path(cfg['inputs'][name]['path']).read_bytes())
    vg, selected, cohort, acquired, cache = (value(n) for n in ('vg_report','ownership_report','ownership_cohort','ownership_acquired','coco_report'))
    rt.require(vg['status'] == 'pass' and vg['source_binding'] == {k:historical['vg'][k] for k in ('binding','modes_identity')}
        and vg['artifact_identities'] == {n:cfg['inputs'][n]['pin'] for n in cfg['vg_files']}
        and vg['source_rehashed_after'] is True and vg['outputs_sealed'] is True, 'Actual original VG metadata receipt differs')
    own = historical['ownership']['binding']; owner = cfg['sources']['ownership']['revision']
    rt.require(selected['status'] == 'pass' and selected['phase'] == 'select' and selected['source_binding'] == own
        and selected['producer_revision'] == owner and selected['cohort_identity'] == cfg['inputs']['ownership_cohort']['pin']
        and selected['source_and_inputs_rehashed_after'] is True and selected['outputs_sealed'] is True
        and cohort['source_binding'] == own and cohort['producer_revision'] == owner and cohort['reference_values_exposed'] is False
        and acquired['source_binding'] == own and acquired['producer_revision'] == owner and acquired['phase'] == 'acquire'
        and acquired['status'] == 'fail' and acquired['decision'] == 'CLOSED_ACQUISITION_CAPACITY_INCONCLUSIVE'
        and acquired['source_and_inputs_rehashed_after'] is True and acquired['outputs_sealed'] is True, 'Closed96 lineage differs')
    rt.require(cache['status'] == 'pass' and cache['source_binding'] == historical['coco']['binding']
        and cache['producer_revision'] == cfg['sources']['coco']['revision'] and cache['phase'] == 'census'
        and cache['archive_identity'] == cfg['inputs']['coco_archive']['pin']
        and cache['member_identity'] == dict(bytes=cfg['streams']['coco']['expanded_bytes'],sha256=cfg['streams']['coco']['expanded_sha256'])
        and cache['source_and_inputs_rehashed_after'] is True and cache['outputs_sealed'] is True, 'Original COCO cache receipt differs')
    # Verify historical publisher-only inputs against the original selection proof.
    for name in (*cfg['history_names'],'coco32_acquired','coco64_acquired'):
        row = cfg['inputs'][name]; proof = selected['input_proof']['frozen_inputs'][row['path']]
        rt.require(proof['pin'] == row['pin'], 'Historical input not bound by original selection')
    rt.require(selected['input_proof']['original_closed_failure']['report_identity'] == cfg['inputs']['original_closed_failure']['pin'], 'Original closed failure not bound')
    return dict(current=current,historical=historical,inputs=frozen)


def exclusions(groups, cohort, acquired, coco_ledgers):
    """Identity metadata only: never read embedded reference paths or values."""
    photos, authors, digests, ids = set(),set(),set(),set(); slots = 0
    for index, (value, count) in enumerate(zip(groups,(16,128,64,32,32,64))):
        if index < 2:
            rows = value['public_metadata']
            rt.require(len(rows) == len(set(value['selected_ids'])) == count
                and {r['ImageID'] for r in rows} == set(value['selected_ids']), 'All historical slots required')
        else:
            rows = value['records']; rt.require(len(rows) == count and [r['slot'] for r in rows] == list(range(count)), 'Complete historical cohort required')
            rows = [r['publisher_metadata'] if index < 4 else r['image'] for r in rows]
        for row in rows:
            slots += 1
            if index < 4:
                iid = row['ImageID']; rt.require(iid not in ids and re.fullmatch('[0-9a-f]{16}',iid), 'Unique original OI identities required'); ids.add(iid)
                photos.add(identities.excluded_photo_identity(row['OriginalLandingURL']))
                if row['AuthorProfileURL']: authors.add(identities.profile_identity(row['AuthorProfileURL']))
                if row['OriginalMD5']: digests.add(identities.md5_identity(row['OriginalMD5']))
            else: photos.add(coco.photo_identity(row['flickr_url']))
    rows = cohort['records']; rt.require(len(rows) == 96 and [r['slot'] for r in rows] == list(range(96))
        and [r['split'] for r in rows] == ['FIT']*32+['CAL']*16+['RESERVED']*48, 'All closed96 slots required')
    for r in rows:
        rt.require(set(r) == {'slot','split','publisher_metadata'}, 'Closed cohort must be metadata only')
        m = r['publisher_metadata']; photos.add(identities.excluded_photo_identity(m['OriginalLandingURL']))
        authors.add(identities.profile_identity(m['AuthorProfileURL'])); digests.add(identities.md5_identity(m['OriginalMD5']))
    for ledger,count in zip(coco_ledgers,(32,64)):
        rt.require(len(ledger['records']) == count and [r['slot'] for r in ledger['records']] == list(range(count)), 'Complete historical byte ledger required')
        for r in ledger['records']:
            rt.require(r['status'] == 'acquired', 'Original COCO acquisition metadata required'); digests.add(identities.md5_identity(r['original_md5']))
    rt.require(len(acquired['records']) == 96 and [r['slot'] for r in acquired['records']] == list(range(96)), 'Closed acquisition must retain every slot')
    for r in acquired['records']:
        if r['status'] == 'acquired': digests.add(identities.md5_identity(r['original_md5']))
    rt.require(slots == 336 and len(photos) == 432, 'All432 historical photo identities required')
    return dict(photos=photos,authors=authors,md5=digests)


def coco_catalog(stream, excluded, checkpoint=lambda:None):
    catalogs = {k:[] for k in ('images','licenses','categories')}
    for key, raw in js.iter_object_arrays(stream,catalogs,check=checkpoint,max_bytes=64 << 20): catalogs[key].append(js.strict_decode(raw))
    for rows in catalogs.values(): rt.require(len({r['id'] for r in rows}) == len(rows), 'Unique COCO metadata catalogs required')
    grant = [r for r in catalogs['licenses'] if r['id'] == 4]
    names = {r['name'] for r in catalogs['categories']}
    rt.require(len(grant) == 1 and grant[0]['url'] == GRANT and len(names) == 80 and 'person' in names, 'Exact original CC-BY2/80-category catalog required')
    result = {}
    for r in catalogs['images']:
        iid = r['id']; rt.require(type(iid) is int and 0 < iid < 10**12, 'Native COCO image ID required')
        if r['license'] != 4: continue
        rt.require(type(r['width']) is int and type(r['height']) is int and r['width'] > 0 and r['height'] > 0
            and r['file_name'] == f'{iid:012d}.jpg' and r['coco_url'] == f'http://images.cocodataset.org/val2017/{iid:012d}.jpg', 'Original COCO grid/links required')
        photo = coco.photo_identity(r['flickr_url'])
        if photo not in excluded: result[iid] = dict(coco_id=iid,photo_id=photo,width=r['width'],height=r['height'],license_url=GRANT,
            publisher_image_url=r['coco_url'],creator_identity='UNKNOWN',author_disjointness_verified=False)
    return result, names-{'person'}


def image_index(stream, catalog, checkpoint=lambda:None):
    bank = {}; seen = set(); counts = Counter()
    for iid, raw in js.iter_array(stream,id_key='image_id',check=checkpoint):
        rt.require(type(iid) is int and iid > 0 and iid not in seen, 'Unique structural VG image_id required'); seen.add(iid); counts['vg_images'] += 1
        r = js.strict_decode(raw); cid = r.get('coco_id')
        if type(cid) is not int or cid not in catalog: counts['rights_unknown_or_historical'] += 1; continue
        rights = catalog[cid]; fid = r.get('flickr_id')
        if not ((type(fid) is int and fid > 0) or (type(fid) is str and re.fullmatch('[1-9][0-9]*',fid))): counts['crosslink_conflict'] += 1; continue
        if str(fid) != rights['photo_id'] or type(r.get('width')) is not int or type(r.get('height')) is not int \
            or (r['width'],r['height']) != (rights['width'],rights['height']): counts['crosslink_conflict'] += 1; continue
        bank[iid] = dict(image_id=iid,**rights)
    crosslinks = Counter((r['coco_id'],r['photo_id']) for r in bank.values())
    duplicates = [iid for iid,r in bank.items() if crosslinks[(r['coco_id'],r['photo_id'])] > 1]
    for iid in duplicates: del bank[iid]
    counts['duplicate_image_crosslinks'] = len(duplicates)
    counts['rights_eligible_images'] = len(bank)
    return bank, counts


def node(value, width, height):
    rt.require(type(value) is dict and type(value.get('object_id')) is int and value['object_id'] > 0, 'Native object ID required')
    rt.require(('name' in value) != ('names' in value), 'Exclusive source-backed name/names variant required')
    names = [value['name']] if 'name' in value else value['names']
    rt.require(type(names) is list and names and all(type(n) is str and bool(n) for n in names)
        and len(set(names)) == len(names), 'Native names must be nonempty unique strings')
    synsets = value.get('synsets'); rt.require(type(synsets) is list and all(type(n) is str for n in synsets)
        and len(set(synsets)) == len(synsets), 'Native synsets required')
    box = tuple(value[k] for k in ('x','y','w','h'))
    rt.require(all(type(v) in (int,float) and math.isfinite(v) for v in box)
        and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
        and box[0]+box[2] <= width and box[1]+box[3] <= height, 'Original finite in-grid XYWH required')
    return value['object_id'], (box,frozenset(names),frozenset(synsets))


def semantic_census(objects, relations, bank, targets, checkpoint=lambda:None):
    nodes = {}; seen = set(); rejected = {}; counts = Counter()
    for iid, raw in js.iter_array(objects,id_key='image_id',check=checkpoint):
        rt.require(type(iid) is int and iid > 0 and iid not in seen, 'Unique structural object-row image_id required'); seen.add(iid)
        if iid not in bank: continue
        try:
            value = js.strict_decode(raw); rows = value['objects']; rt.require(type(rows) is list, 'Native object list required'); mapping = {}
            for r in rows:
                ident, description = node(r,bank[iid]['width'],bank[iid]['height'])
                rt.require(ident not in mapping and not (description[1]&PERSON and description[1]&targets), 'Duplicate/ambiguous native object identity')
                mapping[ident] = description
            nodes[iid] = mapping
        except (ValueError,KeyError,TypeError,UnicodeError): rejected[iid] = 'object_schema_or_geometry'
    seen_rel = set(); eligible = []
    for iid, raw in js.iter_array(relations,id_key='image_id',check=checkpoint):
        rt.require(type(iid) is int and iid > 0 and iid not in seen_rel, 'Unique structural relation-row image_id required'); seen_rel.add(iid)
        if iid not in bank or iid in rejected: continue
        try:
            rt.require(iid in nodes, 'Missing global object row'); mapping = nodes[iid]
            value = js.strict_decode(raw); rows = value['relationships']; rt.require(type(rows) is list, 'Native relationship list required')
            ids = set(); positives = 0
            for r in rows:
                ident = r['relationship_id']; rt.require(type(ident) is int and ident > 0 and ident not in ids and type(r['predicate']) is str, 'Unique native relation ID/predicate required'); ids.add(ident)
                sid, s = node(r['subject'],bank[iid]['width'],bank[iid]['height']); oid, o = node(r['object'],bank[iid]['width'],bank[iid]['height'])
                rt.require(sid in mapping and oid in mapping and s == mapping[sid] and o == mapping[oid], 'Endpoint/global object conflict')
                positives += int(r['predicate'] in PREDICATES and bool(s[1]&PERSON) and bool(o[1]&targets))
            people = sum(bool(n[1]&PERSON) for n in mapping.values()); target_count = sum(bool(n[1]&targets) for n in mapping.values())
            counts['consulted_images'] += 1; counts['person_ids'] += people; counts['target_ids'] += target_count; counts['published_direct_hold_edges'] += positives
            if people >= 2 and target_count >= 2 and positives: eligible.append(bank[iid])
            else: counts['insufficient_positive_crowd'] += 1
        except (ValueError,KeyError,TypeError,UnicodeError): rejected[iid] = 'relation_schema_or_endpoint_conflict'
    for iid in bank:
        if iid not in seen or iid not in seen_rel: rejected[iid] = 'missing_object_or_relation_row'
    for reason in rejected.values(): counts[reason] += 1
    counts.update(eligible_images=len(eligible),distinct_eligible_photos=len({r['photo_id'] for r in eligible}),rejected_images=len(rejected))
    return sorted(eligible,key=lambda r:r['image_id']),counts


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None


def primary_text(row, deadline):
    parsed = urllib.parse.urlsplit(row['url'])
    rt.require(parsed.scheme == 'https' and parsed.netloc == 'homes.cs.washington.edu' and not parsed.query and not parsed.fragment, 'Exact public primary HTTPS required')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with opener.open(urllib.request.Request(row['url'],headers={'Accept-Encoding':'identity'}),timeout=min(15,deadline-time.monotonic())) as response:
        rt.require(response.status == 200 and response.geturl() == row['url'] and response.headers.get('Content-Encoding','identity') == 'identity', 'Original unredirected primary text required')
        length = response.headers.get('Content-Length')
        rt.require(length is None or length == str(row['pin']['bytes']), 'Original declared primary text length differs')
        raw = response.read(row['pin']['bytes']+1)
    check(deadline); rt.require(pin(raw) == row['pin'], 'Primary text bytes differ'); raw.decode('utf-8'); return raw


def zip_stream(spec, checkpoint):
    archive = zipfile.ZipFile(spec['path'])
    class Reader:
        def __init__(self, handle): self.handle,self.digest,self.size = handle,hashlib.sha256(),0
        def read(self,count):
            checkpoint(); raw = self.handle.read(count); self.digest.update(raw); self.size += len(raw); return raw
    try:
        infos = archive.infolist(); matches = [r for r in infos if r.filename == spec['member']]
        rt.require(len(matches) == 1 and len({r.filename for r in infos}) == len(infos)
            and (len(infos) == 1 or spec['kind'] == 'coco'), 'Exact original ZIP member required')
        item = matches[0]
        rt.require(not item.flag_bits&1 and item.compress_type == 8 and stat.S_IFMT(item.external_attr>>16) in (0,stat.S_IFREG)
            and item.file_size == spec['expanded_bytes'] and (spec['kind'] == 'coco' or f'{item.CRC:08x}' == spec['crc32']), 'Original bounded CRC member differs')
        return archive, Reader(archive.open(item))
    except BaseException: archive.close(); raise


def configuration(cfg):
    rt.require(set(cfg) == {'schema','output','budget_seconds','outer_seconds','max_row_bytes','max_expanded_bytes',
        'minimum_distinct_photos','selection_performed','person_names','predicates','target_names','primary_texts',
        'inputs','streams','sources','history_names','vg_files','helper_pins','limitations'}, 'Exact complete census configuration required')
    fixed = dict(schema='world_reward.visual_genome_ownership_census.v1',output=str(DATA),budget_seconds=600,
        outer_seconds=615,max_row_bytes=16 << 20,max_expanded_bytes=2 << 30,minimum_distinct_photos=96,selection_performed=False)
    rt.require(all(type(cfg[k]) is type(v) and cfg[k] == v for k,v in fixed.items())
        and cfg['person_names'] == sorted(PERSON) and cfg['predicates'] == ['holding','holds']
        and len(cfg['target_names']) == len(set(cfg['target_names'])) == 79 and not set(cfg['target_names'])&PERSON,
        'Fixed census scope differs')
    base = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/'
    expected = [('readme_v1_4.txt',746,'788334c8c396b15869cea8cf3097e837a8f0752fefececb9c10ea3d28089be0e'),
        ('relationship_alias.txt',122102,'15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c'),
        ('object_alias.txt',60166,'0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b')]
    rt.require(cfg['primary_texts'] == [dict(file=n,url=base+n,pin=dict(bytes=s,sha256=h)) for n,s,h in expected], 'Exact audited primary texts required')
    rt.require(set(cfg['helper_pins']) == set(HELPERS[3:7]) and set(cfg['sources']) == {'vg','coco','ownership'}
        and cfg['history_names'] == ['oi16','oi128','oi64','oi32','coco32','coco64'], 'Complete independent provenance closure required')
    rt.require(set(cfg['streams']) == {'coco','images','objects','relations'} and len(cfg['inputs']) == 20, 'Complete bounded input inventory required')
    for name, key in (('coco','coco_archive'),('images','image_data.json.zip'),('objects','objects.json.zip'),('relations','relationships.json.zip')):
        spec = cfg['streams'][name]; rt.require(spec['path'] == cfg['inputs'][key]['path']
            and 0 < spec['expanded_bytes'] <= 1 << 30, 'Authenticated bounded stream path required')
    rt.require(sum(s['expanded_bytes'] for s in cfg['streams'].values()) <= 2 << 30, 'Total expansion bound exceeded')
    return cfg


def publish_report(out,report,deadline,started):
    with (out/'report.json').open('x+b') as handle:
        os.fchmod(handle.fileno(),0o400)
        try:
            out.chmod(0o500); identities.sync_directory(out); identities.sync_directory(out.parent)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started)
            handle.write(encode(report)); handle.flush(); os.fsync(handle.fileno()); identities.sync_directory(out); check(deadline)
        except BaseException:
            report.update(status='fail',decision='CLOSED_CENSUS_PUBLICATION',publication_failed=True,elapsed_seconds=time.monotonic()-started,
                outputs_sealed=stat.S_IMODE(out.lstat().st_mode) == 0o500)
            handle.seek(0); handle.truncate(); handle.write(encode(report)); handle.flush(); os.fsync(handle.fileno())


def census(cfg, excluded, checkpoint):
    streams = []; expanded = {}
    try:
        for name in ('coco','images','objects','relations'):
            archive, reader = zip_stream(cfg['streams'][name],checkpoint); streams.append((name,archive,reader))
        handles = {name:reader for name,_,reader in streams}
        catalog,targets = coco_catalog(handles['coco'],excluded['photos'],checkpoint)
        rt.require(targets == set(cfg['target_names']), 'Fixed79 target catalog differs')
        bank,images = image_index(handles['images'],catalog,checkpoint)
        ledger,counts = semantic_census(handles['objects'],handles['relations'],bank,targets,checkpoint)
        for name,_,reader in streams:
            rt.require(reader.read(1) == b'' and reader.size == cfg['streams'][name]['expanded_bytes'], 'Complete original JSON/CRC stream required')
            expected = cfg['streams'][name].get('expanded_sha256')
            rt.require(expected is None or reader.digest.hexdigest() == expected, 'Original expansion SHA differs')
            expanded[name] = dict(bytes=reader.size,sha256=reader.digest.hexdigest(),crc_stream_verified=True)
        rt.require(sum(r['bytes'] for r in expanded.values()) <= 2 << 30, 'Total expanded metadata limit exceeded')
        return ledger,dict(images,**counts),expanded
    finally:
        for _,archive,reader in streams: reader.handle.close(); archive.close()


def run():
    started = time.monotonic(); deadline = started+600
    rt.require(os.geteuid() == 0 and sys.platform == 'linux' and os.uname().nodename == 'world-reward-ncc-h100-02', 'Exact Azure metadata CPU host required')
    code,revision = Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Immutable driver origin required')
    current = rt.source(ROOT,code,revision,ENTRY,HELPERS); cfg = configuration(rt.pinned(code/CONFIG,current['helpers'][CONFIG],32 << 10))
    for module,path in ((js,HELPERS[3]),(rt,HELPERS[4]),(identities,HELPERS[5]),(coco,HELPERS[6])):
        rt.require(Path(module.__file__).resolve() == code/path and current['helpers'][path] == cfg['helper_pins'][path], 'Imported source byte origin differs')
    before = authenticate(cfg,code,revision); rt.require(not rt.canonical(DATA).exists(), 'Fresh namespace only; no retry')
    DATA.mkdir(mode=0o700); out = DATA
    report = dict(schema=cfg['schema'],status='fail',stage='primary_texts',producer_revision=revision,source_binding=before,
        configuration_identity=current['helpers'][CONFIG],selection_performed=False,RGB_read=False,models_loaded=False,GPU_used=False,
        historical_reference_values_read=False,author_disjointness_verified=False,pretraining_overlap_verified=False,
        creator_identity_verified=False,adopted=False,source_and_inputs_rehashed_after=False,outputs_sealed=False)
    def interrupted(*_): raise TimeoutError('Fixed census interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        texts = {}
        for row in cfg['primary_texts']:
            raw = primary_text(row,deadline); rt.write(out/row['file'],raw); texts[row['file']] = row['pin']
        report['primary_texts'] = texts; report['stage'] = 'metadata_census'
        values = {n:rt.strict(Path(r['path']).read_bytes()) for n,r in cfg['inputs'].items() if n in (*cfg['history_names'],'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
        excluded = exclusions([values[n] for n in cfg['history_names']],values['ownership_cohort'],values['ownership_acquired'],[values['coco32_acquired'],values['coco64_acquired']])
        ledger,counts,expanded = census(cfg,excluded,lambda:check(deadline))
        raw = encode(dict(schema='world_reward.visual_genome_rights_capacity_ledger.v1',images=ledger,reference_values_exposed=False))
        rt.write(out/'eligible_metadata.json',raw); report.update(ledger_identity=pin(raw),counts=counts,expanded=expanded,
            historical_slots=432,historical_known_md5=len(excluded['md5']),historical_known_authors=len(excluded['authors']),
            capacity_gate_passed=counts['distinct_eligible_photos'] >= 96,stage='complete')
        report.update(status='pass' if report['capacity_gate_passed'] else 'fail',decision='CAPACITY_METADATA_ONLY' if report['capacity_gate_passed'] else 'CLOSED_INSUFFICIENT_CAPACITY_NO_RGB')
    except BaseException as exc:
        report.update(status='fail',decision='CLOSED_CENSUS_NO_RGB_OR_RETRY',error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','TimeoutError','OSError','HTTPError','URLError','BadZipFile') else 'other')
    finally:
        try:
            rt.require(authenticate(cfg,code,revision) == before, 'Source/input bytes or modes changed after census')
            report['source_and_inputs_rehashed_after'] = True
            for p in out.iterdir(): rt.require(p.name in {r['file'] for r in cfg['primary_texts']}|{'eligible_metadata.json'} and p.lstat().st_uid == 0 and stat.S_IMODE(p.lstat().st_mode) == 0o400, 'Foreign output node')
            report['artifact_identities'] = {p.name:rt.identity(p,16 << 20) for p in out.iterdir()}; check(deadline)
        except BaseException: report.update(status='fail',decision='CLOSED_CENSUS_POSTCHECK',postcheck_failed=True)
        publish_report(out,report,deadline,started)
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items(): signal.signal(s,h)
        print(encode(dict(status=report['status'],decision=report['decision'],report_identity=pin(encode(report)),counts=report.get('counts'))).decode(),end='')
    return report


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1, 'Single frozen census only')
    sys.exit(0 if run()['status'] == 'pass' else 1)

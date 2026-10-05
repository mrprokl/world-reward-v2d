"""Fresh V-COCO metadata capacity only; no cohort, RGB, model or fitting.

Historical photo IDs are excluded before instance/action semantics are decoded.
Only original COCO2014 identities and publisher photo grants are consulted.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import metadata_json_stream as js
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_acquire as identities
import coco_proposal_prepare as coco
import vcoco_role_stream as projection
from world_reward.vcoco_role_reference import parse_vcoco_role_reference

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/vcoco_role_census_v1')
ENTRY = 'run_vcoco_role_census'
CONFIG = 'configs/vcoco_role_census_v1.json'
HELPERS = ('infra/vcoco_role_census.py','infra/run_vcoco_role_census.sh',CONFIG,
    'infra/metadata_json_stream.py','infra/mediapipe_cpu_runtime_verify.py',
    'infra/openimages_joint_pair_acquire.py','infra/coco_proposal_prepare.py',
    'infra/vcoco_role_stream.py','src/world_reward/vcoco_role_reference.py')
GRANT = 'http://creativecommons.org/licenses/by/2.0/'


def encode(value): return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def pin(raw): return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def check(deadline): rt.require(time.monotonic() < deadline,'Inclusive census deadline reached')
def state(path):
    s = path.lstat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_nlink,s.st_uid,s.st_gid,s.st_mtime_ns,s.st_ctime_ns]


def source(code,revision,entry,helpers):
    binding = rt.source(ROOT,code,revision,entry,helpers)
    states = {str(p.relative_to(code)):state(p) for p in (code,*sorted(code.rglob('*')))}
    modes = {n:[s[3]&0o777,s[5],s[6]] for n,s in states.items()}
    return dict(binding=binding,stat_identity=pin(encode(states)),
        modes_identity=pin(json.dumps(modes,sort_keys=True).encode()))


def configuration(cfg):
    fields = {'FIT_allowed','RGB_allowed','archive','budget_seconds','cpu_threads','history_names','inputs',
        'license_id','license_url','max_expanded_bytes','max_field_bytes','max_host_memory_bytes','max_role_bytes',
        'max_row_bytes','metadata_protocol','metadata_protocol_sha256','minimum_distinct_photos',
        'minimum_localized_positive_pairs','minimum_noncrowd_people','minimum_nonperson_objects','network_allowed',
        'outer_seconds','output','photo_rights_scope','positive_agents_ge2_is_diagnostic_only','roles','schema',
        'selection_performed','source_authenticated_provenance_not_leakage_certainty','sources','splits','helper_pins'}
    rt.require(type(cfg) is dict and set(cfg) == fields,'Complete frozen census configuration required')
    fixed = dict(schema='world_reward.vcoco_role_census_config.v1',output=str(DATA),budget_seconds=1200,outer_seconds=1215,
        cpu_threads=4,max_host_memory_bytes=8 << 30,max_expanded_bytes=512 << 20,max_row_bytes=1 << 20,
        max_field_bytes=2 << 20,max_role_bytes=16 << 20,license_id=4,license_url=GRANT,
        minimum_distinct_photos=dict(val=8,test=8),minimum_noncrowd_people=2,minimum_nonperson_objects=2,
        minimum_localized_positive_pairs=1,positive_agents_ge2_is_diagnostic_only=True,selection_performed=False,
        FIT_allowed=False,RGB_allowed=False,network_allowed=False,source_authenticated_provenance_not_leakage_certainty=True,
        photo_rights_scope='publisher_CC_BY_2_metadata_only_creator_unknown',history_names=['oi16','oi128','oi64','oi32','coco32','coco64'])
    rt.require(all(type(cfg[k]) is type(v) and cfg[k] == v for k,v in fixed.items()),'Predeclared scope and limits differ')
    protocol = cfg['metadata_protocol']; digest = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    rt.require(cfg['metadata_protocol_sha256'] == digest and protocol['source_commit'] == '489cc4db74f2f10ab4b134f67da3874afbf245ab'
        and protocol['archive']['file'] == 'annotations_trainval2014.zip','Original acquisition protocol required')
    rt.require(cfg['archive']['member_catalogue'] == protocol['catalogue'] and cfg['archive']['catalogue_tail_identity'] == protocol['catalogue_tail_identity']
        and cfg['archive']['input'] == 'coco_archive' and cfg['archive']['instances_members'] ==
        {p:'annotations/instances_'+p+'.json' for p in ('train2014','val2014')},'Original archive catalogue required')
    rt.require(cfg['roles'] == {p:'data__vcoco__vcoco_'+p+'.json' for p in ('val','test')}
        and cfg['splits'] == {p:'data__splits__vcoco_'+p+'.ids' for p in ('all','train','val','trainval','test')},'Native role and split paths required')
    expected = set(cfg['history_names'])|{'ownership_report','ownership_cohort','ownership_acquired','original_closed_failure',
        'coco32_acquired','coco64_acquired','metadata_report','coco_archive'}|{r['file'] for r in protocol['opaque_git_assets']}
    rt.require(set(cfg['inputs']) == expected and set(cfg['sources']) == {'metadata','ownership','coco'}
        and set(cfg['helper_pins']) == set(HELPERS[3:]),'Complete input/import closure required')
    for row in cfg['inputs'].values():
        rt.require(set(row) == {'path','pin','maximum_bytes','readonly'} and type(row['readonly']) is bool
            and type(row['maximum_bytes']) is int and 0 < row['maximum_bytes'] <= 300 << 20
            and type(row['pin']['bytes']) is int and 0 < row['pin']['bytes'] <= row['maximum_bytes']
            and re.fullmatch('[0-9a-f]{64}',str(row['pin']['sha256'])),'Bounded exact input identity required')
        rt.canonical(row['path'])
    return cfg


def authenticate(cfg,code,revision):
    current = source(code,revision,ENTRY,HELPERS); old = {}; frozen = {}
    for label,spec in cfg['sources'].items():
        path = ROOT/'jobs'/spec['revision']/spec['entry']/'code'
        proof = source(path,spec['revision'],spec['entry'],tuple(spec['helpers']))
        rt.require(proof['binding']['closure_sha256'] == spec['closure_sha256']
            and (spec['entries'] is None or proof['binding']['entries'] == spec['entries'])
            and sum(p.is_file() for p in path.rglob('*')) == spec['files']
            and (path.parent/'source-sha256').read_bytes() == (spec['archive_xz_sha256']+'\n').encode(),'Original full source differs')
        old[label] = proof
    for label,row in cfg['inputs'].items():
        path = Path(row['path']); rt.require(rt.identity(path,row['maximum_bytes'],readonly=row['readonly']) == row['pin'],'Input bytes differ')
        frozen[label] = dict(pin=row['pin'],state=state(path))
    def load(label): return rt.strict(Path(cfg['inputs'][label]['path']).read_bytes())
    report = load('metadata_report'); protocol = cfg['metadata_protocol']; base = Path(cfg['inputs']['metadata_report']['path']).parent
    rt.require(report['status'] == 'pass' and report['source_binding'] == {k:old['metadata'][k] for k in ('binding','modes_identity')}
        and report['producer_revision'] == cfg['sources']['metadata']['revision'] and report['protocol_identity'] ==
        pin(json.dumps(protocol,sort_keys=True).encode()) and report['source_rehashed_after'] is True
        and report['outputs_sealed'] is True and report['output_directory_rechecked'] is True
        and report['annotation_values_consulted'] is False and 0 < report['elapsed_seconds'] <= 600
        and report['decision'] == 'METADATA_QUALIFIED_PENDING_SEPARATE_CENSUS','Original opaque metadata receipt differs')
    rows = protocol['notices']+protocol['opaque_git_assets']+[protocol['archive']]
    rt.require(set(report['outputs']) == set(report['artifact_identities']) == {r['file'] for r in rows},'All18 original assets required')
    rt.require(stat.S_IMODE(base.lstat().st_mode) == 0o500 and base.lstat().st_uid == 0
        and {p.name for p in base.iterdir()} == {r['file'] for r in rows}|{'report.json'},'Original exclusive sealed metadata namespace required')
    for row in rows:
        path = base/row['file']; wanted = report['outputs'][row['file']]
        rt.require(wanted == report['artifact_identities'][row['file']] and wanted['bytes'] == row['bytes']
            and (row['sha256'] is None or wanted['sha256'] == row['sha256']) and rt.identity(path,300 << 20) == wanted
            and stat.S_IMODE(path.lstat().st_mode) == 0o400 and path.lstat().st_uid == 0,'Original sealed asset differs')
        if row['git_blob_sha1'] is not None:
            raw = path.read_bytes(); rt.require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob_sha1'],'Original Git blob differs')
        frozen[str(path)] = dict(pin=wanted,state=state(path))
        if row['file'] in cfg['inputs']: rt.require(cfg['inputs'][row['file']]['pin'] == wanted,'Input not tied to original asset')
    rt.require(cfg['inputs']['coco_archive']['pin'] == report['outputs'][protocol['archive']['file']]
        and report['source_commit'] == protocol['source_commit'] and report['source_tree'] == protocol['source_tree'],'Original archive/source identity differs')
    expanded = report['archive_members']; catalogue = protocol['catalogue']
    rt.require(len(expanded) == len(catalogue) and [r['member'] for r in expanded] == [r['member'] for r in catalogue]
        and all(r['expanded_bytes'] == c['expanded_bytes'] and r['crc32'] == c['crc32']
        and r['crc_stream_verified'] is True and r['json_values_consulted'] is False and r['expanded_file_written'] is False
        and re.fullmatch('[0-9a-f]{64}',r['expanded_sha256']) for r,c in zip(expanded,catalogue)), 'Original complete expansion proof required')
    selected,cohort,acquired = (load(n) for n in ('ownership_report','ownership_cohort','ownership_acquired'))
    binding = old['ownership']['binding']; owner = cfg['sources']['ownership']['revision']
    rt.require(selected['status'] == 'pass' and selected['phase'] == 'select' and selected['source_binding'] == binding
        and selected['producer_revision'] == owner and selected['cohort_identity'] == cfg['inputs']['ownership_cohort']['pin']
        and selected['source_and_inputs_rehashed_after'] is True and selected['outputs_sealed'] is True
        and cohort['source_binding'] == binding and cohort['producer_revision'] == owner and cohort['reference_values_exposed'] is False
        and acquired['source_binding'] == binding and acquired['producer_revision'] == owner and acquired['phase'] == 'acquire'
        and acquired['status'] == 'fail' and acquired['decision'] == 'CLOSED_ACQUISITION_CAPACITY_INCONCLUSIVE'
        and acquired['source_and_inputs_rehashed_after'] is True and acquired['outputs_sealed'] is True,'Closed historical96 lineage differs')
    for name in (*cfg['history_names'],'coco32_acquired','coco64_acquired'):
        row = cfg['inputs'][name]; rt.require(selected['input_proof']['frozen_inputs'][row['path']]['pin'] == row['pin'],'Historical metadata not bound')
    rt.require(selected['input_proof']['original_closed_failure']['report_identity'] == cfg['inputs']['original_closed_failure']['pin'],'Original closed failure differs')
    return dict(current=current,historical=old,inputs=frozen)


def exclusions(groups,cohort,acquired,coco_ledgers):
    """Historical publisher identities only; embedded reference paths never opened."""
    photos,authors,digests,ids = set(),set(),set(),set(); slots = 0
    for index,(value,count) in enumerate(zip(groups,(16,128,64,32,32,64))):
        if index < 2:
            rows = value['public_metadata']; rt.require(len(rows) == len(set(value['selected_ids'])) == count
                and {r['ImageID'] for r in rows} == set(value['selected_ids']),'All historical slots required')
        else:
            rows = value['records']; rt.require(len(rows) == count and [r['slot'] for r in rows] == list(range(count)),'Complete historical cohort required')
            rows = [r['publisher_metadata'] if index < 4 else r['image'] for r in rows]
        for row in rows:
            slots += 1
            if index < 4:
                iid = row['ImageID']; rt.require(iid not in ids and re.fullmatch('[0-9a-f]{16}',iid),'Unique historical OI ID required'); ids.add(iid)
                photos.add(identities.excluded_photo_identity(row['OriginalLandingURL']))
                if row['AuthorProfileURL']: authors.add(identities.profile_identity(row['AuthorProfileURL']))
                if row['OriginalMD5']: digests.add(identities.md5_identity(row['OriginalMD5']))
            else: photos.add(coco.photo_identity(row['flickr_url']))
    rows = cohort['records']; rt.require(len(rows) == 96 and [r['slot'] for r in rows] == list(range(96))
        and [r['split'] for r in rows] == ['FIT']*32+['CAL']*16+['RESERVED']*48,'All closed96 slots required')
    for row in rows:
        rt.require(set(row) == {'slot','split','publisher_metadata'},'Closed cohort is metadata only')
        m = row['publisher_metadata']; photos.add(identities.excluded_photo_identity(m['OriginalLandingURL']))
        authors.add(identities.profile_identity(m['AuthorProfileURL'])); digests.add(identities.md5_identity(m['OriginalMD5']))
    for ledger,count in zip(coco_ledgers,(32,64)):
        rt.require(len(ledger['records']) == count and [r['slot'] for r in ledger['records']] == list(range(count)),'All historical byte slots required')
        for row in ledger['records']:
            rt.require(row['status'] == 'acquired','Original COCO byte metadata required'); digests.add(identities.md5_identity(row['original_md5']))
    rt.require(len(acquired['records']) == 96 and [r['slot'] for r in acquired['records']] == list(range(96)),'All closed acquisition slots retained')
    for row in acquired['records']:
        if row['status'] == 'acquired': digests.add(identities.md5_identity(row['original_md5']))
    rt.require(slots == 336 and len(photos) == 432,'All432 historical photos excluded')
    return dict(photos=photos,authors=authors,md5=digests)


def split_ids(raw):
    text = raw.decode('ascii'); rows = text.splitlines()
    rt.require(rows and all(re.fullmatch('[1-9][0-9]*',r) for r in rows),'Original positive split ID lines required')
    result = [int(r) for r in rows]; rt.require(len(result) == len(set(result)),'Unique split IDs required'); return set(result)


def catalog(stream,partition,excluded,checkpoint=lambda:None,max_bytes=512 << 20):
    catalogs = {k:[] for k in ('images','licenses','categories')}
    for key,raw in js.iter_object_arrays(stream,catalogs,check=checkpoint,max_bytes=max_bytes,row_bytes=1 << 20):
        catalogs[key].append(js.strict_decode(raw))
    for rows in catalogs.values(): rt.require(len({r['id'] for r in rows}) == len(rows),'Unique original catalog IDs required')
    grant = [r for r in catalogs['licenses'] if r['id'] == 4]; categories = catalogs['categories']
    rt.require(len(grant) == 1 and grant[0]['url'] == GRANT and len(categories) == 80
        and [r['id'] for r in categories if r['name'] == 'person'] == [1],'Original license/person taxonomy required')
    bank = {}; counts = Counter(catalog_images=len(catalogs['images']))
    for row in catalogs['images']:
        iid = row['id']; rt.require(type(iid) is int and 0 < iid < 10**12,'Native COCO image ID required')
        if row['license'] != 4: counts['publisher_rights_unknown_images'] += 1; continue
        try: photo = coco.photo_identity(row['flickr_url'])
        except (ValueError,KeyError,TypeError): counts['publisher_photo_identity_unknown_images'] += 1; continue
        if photo in excluded: counts['historical_photo_images'] += 1; continue
        filename = f'COCO_{partition}_{iid:012d}.jpg'
        rt.require(type(row['width']) is int and type(row['height']) is int and row['width'] > 0 and row['height'] > 0
            and row['file_name'] == filename and row['coco_url'] == f'http://images.cocodataset.org/{partition}/{filename}',
            'Original2014 image grid/name/URL required')
        bank[iid] = dict(id=iid,width=row['width'],height=row['height'],photo_id=photo,
            publisher_image_url=row['coco_url'],license_url=GRANT,creator_identity='UNKNOWN')
    return bank,{r['id']:r['name'] for r in categories},counts,{r['id'] for r in catalogs['images']}


@contextmanager
def member_stream(path,member,spec,checkpoint):
    class Reader:
        def __init__(self,handle): self.handle,self.size,self.digest = handle,0,hashlib.sha256()
        def read(self,count):
            checkpoint(); raw = self.handle.read(count); self.size += len(raw); self.digest.update(raw); return raw
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        rt.require(info.file_size == spec['expanded_bytes'] and f'{info.CRC:08x}' == spec['crc32'] and info.compress_type == 8,'Original bounded member required')
        with archive.open(info) as handle:
            reader = Reader(handle); yield reader
            rt.require(reader.read(1) == b'' and reader.size == spec['expanded_bytes'],'Complete lexical/CRC stream required')


def census(cfg,excluded,checkpoint=lambda:None):
    path = cfg['inputs']['coco_archive']['path']; specs = {r['member']:r for r in cfg['archive']['member_catalogue']}
    with Path(path).open('rb') as handle:
        tail = cfg['archive']['catalogue_tail_identity']; handle.seek(-tail['bytes'],os.SEEK_END)
        rt.require(pin(handle.read(tail['bytes'])) == tail,'Original catalogue tail changed')
    with zipfile.ZipFile(path) as archive:
        actual = [dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits) for i in archive.infolist()]
        rt.require(actual == cfg['archive']['member_catalogue'],'Complete original member catalogue changed')
    split = {name:split_ids(Path(cfg['inputs'][key]['path']).read_bytes()) for name,key in cfg['splits'].items()}
    rt.require(not split['train']&split['val'] and not split['trainval']&split['test']
        and split['train']|split['val'] == split['trainval'] and split['trainval']|split['test'] == split['all'],'Official split partition differs')
    native_report = rt.pinned(cfg['inputs']['metadata_report']['path'],cfg['inputs']['metadata_report']['pin'])
    expansion_pins = {r['member']:dict(bytes=r['expanded_bytes'],sha256=r['expanded_sha256']) for r in native_report['archive_members']}
    banks, taxonomy, counts, expanded, all_ids, partition_ids = {},None,Counter(),{},set(),{}
    for partition,member in cfg['archive']['instances_members'].items():
        with member_stream(path,member,specs[member],checkpoint) as handle:
            bank,categories,count,ids = catalog(handle,partition,excluded['photos'],checkpoint,cfg['max_expanded_bytes'])
        rt.require(taxonomy is None or taxonomy == categories,'Original partition taxonomies differ'); taxonomy = categories
        rt.require(not all_ids&ids,'Original image ID partitions conflict'); all_ids.update(ids); banks.update(bank); counts.update(count)
        partition_ids[partition] = ids
        expanded[partition] = dict(bytes=handle.size,sha256=handle.digest.hexdigest())
        rt.require(expanded[partition] == expansion_pins[member],'Original complete member SHA differs')
    rt.require(split['all'] <= all_ids,'Original split references absent catalog IDs')
    duplicate_photos = Counter(r['photo_id'] for r in banks.values())
    banks = {iid:r for iid,r in banks.items() if duplicate_photos[r['photo_id']] == 1}
    eligible_ids = (split['val']|split['test'])&set(banks); instances = []; seen = set()
    for partition,member in cfg['archive']['instances_members'].items():
        with member_stream(path,member,specs[member],checkpoint) as handle:
            for iid,row in projection.iter_filtered_coco_annotations(handle,eligible_ids,check=checkpoint,
                max_bytes=cfg['max_expanded_bytes'],row_bytes=cfg['max_row_bytes']):
                rt.require(type(row['id']) is int and row['id'] > 0 and row['id'] not in seen
                    and row['category_id'] in taxonomy and iid in partition_ids[partition],'Original unique annotation/category/partition ID required')
                seen.add(row['id']); instances.append(row)
        rt.require(dict(bytes=handle.size,sha256=handle.digest.hexdigest()) == expanded[partition],'Complete member changed between catalog/annotation passes')
    people,objects = Counter(),Counter()
    for row in instances:
        valid = coco.countable(row); counts['crowd_instances'] += row['iscrowd'] == 1
        counts['invalid_or_nonpositive_geometry_instances'] += row['area'] <= 0 or row['bbox'][2] <= 0 or row['bbox'][3] <= 0
        if valid: (people if row['category_id'] == 1 else objects)[row['image_id']] += 1
    # The pure parser owns all finite geometry/agent/positive endpoint validation.
    outputs = {}; inventory = []; images = [{k:r[k] for k in ('id','width','height')} for iid,r in banks.items() if iid in eligible_ids]
    for name,key in cfg['roles'].items():
        selected_ids = eligible_ids&split[name]
        def open_role():
            row = cfg['inputs'][key]; rt.require(rt.identity(row['path'],cfg['max_role_bytes']) == row['pin'],'Role source differs before each open')
            return Path(row['path']).open('rb')
        actions,proof = projection.project_vcoco_actions(open_role,selected_ids,
            check=checkpoint,max_bytes=cfg['max_role_bytes'],field_bytes=cfg['max_field_bytes'],expected_image_ids=split[name])
        rt.require(proof['source_identity'] == cfg['inputs'][key]['pin'],'Original role bytes changed during projection')
        reference = parse_vcoco_role_reference(actions,instances,images); checkpoint()
        pairs,agents = Counter(),{}
        for pair in reference.localized_positive_pairs:
            pairs[pair.image_id] += 1; agents.setdefault(pair.image_id,set()).add(pair.agent_annotation_id)
        local = Counter(rights_eligible_images=len(selected_ids),projected_rows=len(reference.rows),localized_positive_pairs=len(reference.localized_positive_pairs))
        for row in reference.rows:
            local['positive_action_rows'] += row.label
            local['missing_positive_role_ids'] += sum(row.label == 1 and i > 0 and rid == 0 for i,rid in enumerate(row.role_object_ids))
            local['unscorable_positive_roles'] += sum(row.label == 1 and i > 0 and rid != 0 and not row.nonperson_pair_eligible[i] for i,rid in enumerate(row.role_object_ids))
        photos = set()
        for iid in sorted(selected_ids):
            if people[iid] >= 2 and objects[iid] >= 2 and pairs[iid] >= 1:
                photos.add(banks[iid]['photo_id']); inventory.append(dict(split=name,image_id=iid,photo_id=banks[iid]['photo_id']))
                local['eligible_images'] += 1; local['eligible_images_two_positive_agents'] += len(agents.get(iid,set())) >= 2
        local['distinct_eligible_photos'] = len(photos); outputs[name] = dict(local,projection=proof)
    counts['duplicate_fresh_photo_images_rejected'] = sum(v for v in duplicate_photos.values() if v > 1)
    return dict(catalog_counts=dict(counts),splits=outputs,eligibility_inventory_identity=pin(encode(inventory)),
        eligibility_inventory_rows=len(inventory),expanded_instance_members=expanded,
        capacity_gate_passed=all(outputs[n]['distinct_eligible_photos'] >= cfg['minimum_distinct_photos'][n] for n in ('val','test')))


def publish_report(out,report,deadline,started,owned):
    def original(mode):
        now = out.lstat(); rt.require(stat.S_ISDIR(now.st_mode) and stat.S_IMODE(now.st_mode) == mode
            and (now.st_dev,now.st_ino,now.st_uid,now.st_gid) == (owned.st_dev,owned.st_ino,owned.st_uid,owned.st_gid)
            and now.st_uid == 0,'Original owned census directory required')
    original(0o700); rt.require(not tuple(out.iterdir()),'No foreign census output allowed')
    with (out/'report.json').open('x+b') as handle:
        os.fchmod(handle.fileno(),0o400)
        owned_file = os.fstat(handle.fileno())
        try:
            original(0o700); out.chmod(0o500); identities.sync_directory(out); identities.sync_directory(out.parent)
            original(0o500)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started)
            handle.write(encode(report)); handle.flush(); os.fsync(handle.fileno()); identities.sync_directory(out)
            now = (out/'report.json').lstat(); original(0o500)
            rt.require(stat.S_ISREG(now.st_mode) and stat.S_IMODE(now.st_mode) == 0o400 and now.st_nlink == 1
                and (now.st_dev,now.st_ino,now.st_uid,now.st_gid) ==
                (owned_file.st_dev,owned_file.st_ino,owned_file.st_uid,owned_file.st_gid),'Owned report inode changed')
            check(deadline)
        except BaseException:
            report.update(status='fail',decision='CLOSED_CENSUS_PUBLICATION',publication_failed=True,elapsed_seconds=time.monotonic()-started)
            handle.seek(0); handle.truncate(); handle.write(encode(report)); handle.flush(); os.fsync(handle.fileno())


def run(started,deadline):
    rt.require(os.geteuid() == 0 and sys.platform == 'linux' and os.uname().nodename == 'world-reward-ncc-h100-02','Exact Azure CPU host required')
    code,revision = Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0],'Immutable driver origin required')
    current = rt.source(ROOT,code,revision,ENTRY,HELPERS); cfg = configuration(rt.pinned(code/CONFIG,current['helpers'][CONFIG],64 << 10))
    for module,path in ((js,HELPERS[3]),(rt,HELPERS[4]),(identities,HELPERS[5]),(coco,HELPERS[6]),(projection,HELPERS[7])):
        rt.require(Path(module.__file__).resolve() == code/path and current['helpers'][path] == cfg['helper_pins'][path],'Actual unchanged import source required')
    rt.require(Path(sys.modules[parse_vcoco_role_reference.__module__].__file__).resolve() == code/HELPERS[8]
        and current['helpers'][HELPERS[8]] == cfg['helper_pins'][HELPERS[8]],'Actual pure reference source required')
    before = authenticate(cfg,code,revision); check(deadline)
    rt.require(not rt.canonical(DATA).exists() and DATA.parent.is_dir(),'Fresh census namespace; no retry')
    DATA.mkdir(mode=0o700); DATA.chmod(0o700); identities.sync_directory(DATA.parent); owned = DATA.lstat()
    report = dict(schema='world_reward.vcoco_role_census.v1',status='fail',stage='historical_exclusions',producer_revision=revision,
        source_binding=before,configuration_identity=current['helpers'][CONFIG],RGB_read=False,GPU_used=False,models_loaded=False,
        network_used=False,FIT_performed=False,selection_performed=False,historical_reference_values_read=False,
        fresh_reference_geometry_consulted=False,author_disjointness_verified=False,creator_identity_verified=False,
        training_overlap_verified=False,challenge_overlap_verified=False,official_AP_computed=False,adopted=False,
        source_and_inputs_rehashed_after=False,outputs_sealed=False)
    try:
        values = {n:rt.strict(Path(cfg['inputs'][n]['path']).read_bytes()) for n in (*cfg['history_names'],
            'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
        excluded = exclusions([values[n] for n in cfg['history_names']],values['ownership_cohort'],values['ownership_acquired'],
            [values['coco32_acquired'],values['coco64_acquired']])
        report.update(stage='fresh_metadata_census',historical_slots=432,historical_known_md5=len(excluded['md5']),historical_known_authors=len(excluded['authors']))
        report['fresh_reference_geometry_consulted'] = True
        result = census(cfg,excluded,lambda:check(deadline)); report.update(result,stage='complete')
        report.update(status='pass' if result['capacity_gate_passed'] else 'fail',decision='CAPACITY_METADATA_ONLY_NO_RGB' if result['capacity_gate_passed'] else 'CLOSED_INSUFFICIENT_CAPACITY_NO_RGB')
    except BaseException as exc:
        report.update(status='fail',decision='CLOSED_CENSUS_NO_RGB_OR_RETRY',error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','TimeoutError','OSError','BadZipFile') else 'other')
    finally:
        try:
            rt.require(authenticate(cfg,code,revision) == before,'Original source/input bytes or modes changed')
            report['source_and_inputs_rehashed_after'] = True
            rt.require(not tuple(DATA.iterdir()),'No reference/foreign outputs permitted')
            report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            rt.require(report['peak_rss_bytes'] <= cfg['max_host_memory_bytes'],'Host memory budget exceeded'); check(deadline)
        except BaseException: report.update(status='fail',decision='CLOSED_CENSUS_POSTCHECK',postcheck_failed=True)
        publish_report(DATA,report,deadline,started,owned)
        print(encode(dict(status=report['status'],decision=report['decision'],counts=report.get('splits'),report_identity=pin(encode(report)))).decode(),end='')
    return report


def main():
    started = time.monotonic(); deadline = started+1200
    def interrupted(*_): raise TimeoutError('Fixed census interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,1200)
    try: return run(started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items(): signal.signal(s,h)


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1,'Single frozen metadata census only')
    sys.exit(0 if main()['status'] == 'pass' else 1)

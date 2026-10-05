"""New frozen16 role pilot: private references, then separate original RGB only."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import json
import hashlib
import os
from pathlib import Path
import signal
import stat
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import vcoco_role_census as c

rt, js, acq, projection = c.rt, c.js, c.identities, c.projection
ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/vcoco_role_pilot_v1')
ENTRY = 'run_vcoco_role_prepare'
CONFIG = 'configs/vcoco_role_pilot_v1.json'
HELPERS = ('infra/vcoco_role_prepare.py','infra/run_vcoco_role_prepare.sh',CONFIG,
    'infra/vcoco_role_census.py',
    'infra/metadata_json_stream.py','infra/mediapipe_cpu_runtime_verify.py',
    'infra/openimages_joint_pair_acquire.py','infra/coco_proposal_prepare.py',
    'infra/vcoco_role_stream.py','src/world_reward/vcoco_role_reference.py')
NAMESPACE = 'world_reward.vcoco_role_pilot_v1/'
BUCKET = 'https://s3.amazonaws.com/images.cocodataset.org/'
PUBLIC_KEYS = frozenset(('image_id','file','bytes','sha256','width','height'))
SELECT = dict(budget_seconds=1200,outer_seconds=1215,max_host_memory_bytes=8 << 30,
    hash_namespace=NAMESPACE,slots=16,dev=8,reserved=8,dev_native_split='val',reserved_native_split='test',
    minimum_noncrowd_people=2,minimum_nonperson_objects=2,minimum_localized_positive_pairs=1,
    expected_distinct_photos=dict(val=199,test=362),expected_inventory=dict(bytes=36117,
        sha256='7c374f0ff34d340983d93a117e51a4fcf09d4d4425e4d56392a327d84dd6c733'),
    no_replacements=True,network_allowed=False,RGB_allowed=False,FIT_allowed=False)
ACQUIRE = dict(budget_seconds=300,outer_seconds=315,max_host_memory_bytes=8 << 30,workers=4,request_timeout=15,
    max_image_bytes=16 << 20,max_decoded_pixels=16 << 20,max_total_image_bytes=256 << 20,
    min_acquired_dev=6,min_acquired_reserved=6,retry_count=0,no_replacements=True,original_bucket=BUCKET,
    image_license_id=4,image_license_url=c.GRANT,
    grant_authority='publisher_recorded_individual_image_license_not_independent_creator_authentication',
    creator_identity='UNKNOWN',rights_decision='accept_publisher_CC_BY_2_record_for_private_unfitted_pilot_only',
    pretraining_overlap='COCO_family_declared_exact_photo_and_challenge_overlap_unknown',raw_redistribution=False,
    FIT_allowed=False,submission_eligibility_verified=False)


def raw_write(path,raw,ledger):
    """One owned inode; failed writes never leave an untracked partial JPEG."""
    path=rt.canonical(path);owned=None
    try:
        with path.open('xb')as stream:
            owned=os.fstat(stream.fileno());os.fchmod(stream.fileno(),0o400)
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        wanted=c.pin(raw);rt.require(rt.identity(path,max(1,len(raw)))==wanted,'Owned original bytes differ')
        acq.sync_directory(path.parent);ledger[path.name]=wanted;return wanted
    except BaseException:
        if owned is not None:
            now=path.lstat();rt.require((now.st_dev,now.st_ino)==(owned.st_dev,owned.st_ino)
                and stat.S_ISREG(now.st_mode)and now.st_nlink==1,'Cannot remove foreign partial inode');path.unlink()
        raise


def write(path,value,ledger):return raw_write(path,c.encode(value),ledger)


def inventory(out,owned,expected,mode,maximum):
    s=rt.canonical(out).lstat();rt.require(stat.S_ISDIR(s.st_mode)and stat.S_IMODE(s.st_mode)==mode
        and (s.st_dev,s.st_ino,s.st_uid,s.st_gid)==(owned.st_dev,owned.st_ino,owned.st_uid,owned.st_gid)
        and {p.name for p in out.iterdir()}==set(expected),'Owned exact pilot namespace required')
    for name,wanted in expected.items():
        path=out/name;s=path.lstat();rt.require(stat.S_IMODE(s.st_mode)==0o400
            and (s.st_uid,s.st_gid)==(owned.st_uid,owned.st_gid)and rt.identity(path,maximum)==wanted,
            'Owned readonly pilot artifact differs')
    return {name:c.state(out/name)for name in expected}


def namespace_identity(path):
    s=rt.canonical(path).lstat();rt.require(stat.S_ISDIR(s.st_mode)and stat.S_IMODE(s.st_mode)==0o700
        and s.st_uid==s.st_gid==0,'Root-private original pilot namespace required')
    return [s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid]


def configuration(code,source):
    cfg=rt.pinned(code/CONFIG,source['helpers'][CONFIG],65536)
    rt.require(type(cfg)is dict and set(cfg)=={'schema','output','select','acquire','census','helper_pins'},'Exact pilot recipe fields required')
    rt.require(cfg['schema']=='world_reward.vcoco_role_prepare_config.v1' and cfg['output']==str(DATA)
        and c.encode(cfg['select'])==c.encode(SELECT) and c.encode(cfg['acquire'])==c.encode(ACQUIRE),
        'Frozen16 selection/acquisition scope required')
    rt.require(set(cfg['helper_pins'])==set(HELPERS[3:]) and all(source['helpers'][n]==p for n,p in cfg['helper_pins'].items()),
        'Every unchanged reusable source helper required')
    for module,name in ((c,HELPERS[3]),(js,HELPERS[4]),(rt,HELPERS[5]),
                        (acq,HELPERS[6]),(c.coco,HELPERS[7]),(projection,HELPERS[8])):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual unchanged imported helper origin required')
    rt.require(Path(sys.modules[c.parse_vcoco_role_reference.__module__].__file__).resolve()==code/HELPERS[9],
        'Actual unchanged pure reference source required')
    old=cfg['census'];rt.require(set(old)=={'producer_revision','files','entries','closure_sha256','archive_xz_sha256','report','configuration'}
        and old['producer_revision']=='3f445f2d9ee4b2e9849ff15f5dd2cd31cbed964c',
        'Exact original qualified census required')
    return cfg


def publish(out,report,deadline,started,owned,expected):
    """Same fresh FD survives sealing; foreign leaves are never removed/chmodded."""
    before=inventory(out,owned,expected,0o700,4 << 20)
    with (out/'report.json').open('x+b')as stream:
        os.fchmod(stream.fileno(),0o400);file_stat=os.fstat(stream.fileno())
        def update():
            stream.seek(0);stream.write(c.encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            rt.require({p.name for p in out.iterdir()}==set(expected)|{'report.json'},'Foreign pilot leaf introduced')
            out.chmod(0o500);acq.sync_directory(out);acq.sync_directory(out.parent)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started);update()
            rt.require(rt.identity(out/'report.json',1 << 20)==c.pin(c.encode(report)),'Sealed report bytes differ')
            after=inventory(out,owned,{**expected,'report.json':c.pin(c.encode(report))},0o500,4 << 20)
            rt.require(all(after[n]==before[n]for n in expected),'Pilot artifacts changed during sealing')
            now=(out/'report.json').lstat();rt.require((now.st_dev,now.st_ino)==(file_stat.st_dev,file_stat.st_ino)
                and now.st_nlink==1,'Owned report inode changed');c.check(deadline)
        except BaseException:
            report.update(status='fail',decision='CLOSED_PILOT_PUBLICATION',publication_failed=True,elapsed_seconds=time.monotonic()-started);update()


def authenticate(code,rev):
    source=rt.source(ROOT,code,rev,ENTRY,HELPERS);cfg=configuration(code,source);old=cfg['census']
    source_states=c.pin(c.encode({str(p.relative_to(code)):c.state(p)for p in (code,*sorted(code.rglob('*')))}))
    old_code=ROOT/'jobs'/old['producer_revision']/c.ENTRY/'code'
    original=rt.source(ROOT,old_code,old['producer_revision'],c.ENTRY,c.HELPERS)
    rt.require(original['entries']==old['entries'] and sum(p.is_file()for p in old_code.rglob('*'))==old['files']
        and original['closure_sha256']==old['closure_sha256']
        and (old_code.parent/'source-sha256').read_bytes()==(old['archive_xz_sha256']+'\n').encode(),'Original census full source differs')
    old_cfg=c.configuration(rt.pinned(Path(old['configuration']['path']),old['configuration']['pin'],65536))
    rt.require(Path(old['configuration']['path'])==old_code/c.CONFIG
        and old['configuration']['pin']==original['helpers'][c.CONFIG],'Original census configuration binding required')
    before=c.authenticate(old_cfg,old_code,old['producer_revision'])
    report_path=Path(old['report']['path']);report=rt.pinned(report_path,old['report']['pin'],300000)
    rt.require(stat.S_IMODE(report_path.lstat().st_mode)==0o400 and report_path.lstat().st_uid==0
        and stat.S_IMODE(c.DATA.lstat().st_mode)==0o500 and c.DATA.lstat().st_uid==0
        and {p.name for p in c.DATA.iterdir()}=={'report.json'},'Original sealed census namespace required')
    rt.require(old['report']['pin']==dict(bytes=20320,sha256='79142e6a49d11e3570ff28a41cb09b86056769052823e30a6b25b29712a37975')
        and Path(old['report']['path'])==c.DATA/'report.json'
        and report['status']=='pass' and report['stage']=='complete' and report['capacity_gate_passed']is True
        and report['source_binding']==before and report['configuration_identity']==old['configuration']['pin']
        and report['source_and_inputs_rehashed_after']is True and report['outputs_sealed']is True
        and report['historical_slots']==432 and report['eligibility_inventory_rows']==561
        and report['splits']['val']['distinct_eligible_photos']==199 and report['splits']['test']['distinct_eligible_photos']==362,
        'Original sealed metadata capacity receipt required')
    values={n:rt.strict(Path(old_cfg['inputs'][n]['path']).read_bytes())for n in (*old_cfg['history_names'],
        'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
    excluded=c.exclusions([values[n]for n in old_cfg['history_names']],values['ownership_cohort'],values['ownership_acquired'],
        [values['coco32_acquired'],values['coco64_acquired']])
    return cfg,old_cfg,report,excluded,dict(source_binding=source,census_binding=before,
        source_stat_identity=source_states,census_report_identity=old['report']['pin'],
        census_report_state=c.state(report_path),census_directory_state=c.state(c.DATA))


def projected_image(actions,iid):
    """Retain every original action row for one image, role-major and original slots."""
    result=[]
    for action in actions:
        indices=[i for i,image in enumerate(action['image_id'])if image==iid];n=len(action['image_id'])
        result.append(dict(action_name=action['action_name'],role_name=list(action['role_name']),
            ann_id=[action['ann_id'][i]for i in indices],image_id=[iid]*len(indices),label=[action['label'][i]for i in indices],
            role_object_id=[action['role_object_id'][role*n+i]for role in range(len(action['role_name']))for i in indices],
            projection_source=dict(action_slot=action['projection_source']['action_slot'],
                row_slots=[action['projection_source']['row_slots'][i]for i in indices],row_count=action['projection_source']['row_count'])))
    return result


def reproduce(cfg,excluded,expected,check):
    """Same census population/predicates; new cohort construction, no old run call."""
    path=cfg['inputs']['coco_archive']['path'];specs={r['member']:r for r in cfg['archive']['member_catalogue']}
    tail=cfg['archive']['catalogue_tail_identity']
    with Path(path).open('rb')as stream:
        stream.seek(-tail['bytes'],2);rt.require(c.pin(stream.read(tail['bytes']))==tail,'Original ZIP tail differs')
    with zipfile.ZipFile(path)as archive:
        actual=[dict(member=i.filename,compressed_bytes=i.compress_size,expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}',external_attr=i.external_attr,compression=i.compress_type,flags=i.flag_bits)for i in archive.infolist()]
        rt.require(actual==cfg['archive']['member_catalogue'],'Original complete ZIP catalogue differs')
    native=rt.pinned(cfg['inputs']['metadata_report']['path'],cfg['inputs']['metadata_report']['pin'])
    expansion={r['member']:dict(bytes=r['expanded_bytes'],sha256=r['expanded_sha256'])for r in native['archive_members']}
    split={n:c.split_ids(Path(cfg['inputs'][key]['path']).read_bytes())for n,key in cfg['splits'].items()}
    banks,original_images,taxonomy,parts,expanded={}, {},None,{},{}
    for partition,member in cfg['archive']['instances_members'].items():
        catalog={k:[]for k in ('images','licenses','categories')}
        with c.member_stream(path,member,specs[member],check)as handle:
            for key,raw in js.iter_object_arrays(handle,catalog,check=check,max_bytes=cfg['max_expanded_bytes'],row_bytes=cfg['max_row_bytes']):
                catalog[key].append(js.strict_decode(raw))
        rt.require(dict(bytes=handle.size,sha256=handle.digest.hexdigest())==expansion[member],'Original complete catalogue member SHA differs')
        bank,categories,_,ids=c.catalog(io.BytesIO(c.encode(catalog)),partition,excluded['photos'],check,cfg['max_expanded_bytes'])
        rt.require(taxonomy is None or taxonomy==categories,'Original taxonomy differs');taxonomy=categories
        rt.require(not any(ids&v for v in parts.values()),'Native image partitions conflict');parts[partition]=ids;banks.update(bank)
        original_images.update({r['id']:dict(r,partition=partition)for r in catalog['images']if r['id']in bank})
        expanded[partition]=expansion[member]
    rt.require(split['all']<=set().union(*parts.values()) and not split['train']&split['val'] and not split['trainval']&split['test']
        and split['train']|split['val']==split['trainval'] and split['trainval']|split['test']==split['all'],'Official split partition differs')
    duplicate=Counter(r['photo_id']for r in banks.values());banks={iid:r for iid,r in banks.items()if duplicate[r['photo_id']]==1}
    eligible=(split['val']|split['test'])&set(banks);instances=defaultdict(list);seen=set();people,objects=Counter(),Counter()
    for partition,member in cfg['archive']['instances_members'].items():
        with c.member_stream(path,member,specs[member],check)as handle:
            for iid,row in projection.iter_filtered_coco_annotations(handle,eligible,check=check,max_bytes=cfg['max_expanded_bytes'],row_bytes=cfg['max_row_bytes']):
                rt.require(type(row['id'])is int and row['id']>0 and row['id']not in seen and row['category_id']in taxonomy and iid in parts[partition],
                    'Original unique annotation/category/partition required')
                seen.add(row['id']);instances[iid].append(row)
                if c.coco.countable(row):(people if row['category_id']==1 else objects)[iid]+=1
        rt.require(dict(bytes=handle.size,sha256=handle.digest.hexdigest())==expanded[partition],'Original annotation member changed')
    inventory=[];actions_by_split={};images=[{k:r[k]for k in ('id','width','height')}for iid,r in banks.items()if iid in eligible]
    all_instances=[r for rows in instances.values()for r in rows]
    for name,key in cfg['roles'].items():
        selected_ids=eligible&split[name]
        def open_role():
            row=cfg['inputs'][key];rt.require(rt.identity(row['path'],cfg['max_role_bytes'])==row['pin'],'Same role source before each pass required')
            return Path(row['path']).open('rb')
        actions,proof=projection.project_vcoco_actions(open_role,selected_ids,check=check,max_bytes=cfg['max_role_bytes'],
            field_bytes=cfg['max_field_bytes'],expected_image_ids=split[name])
        rt.require(proof['source_identity']==cfg['inputs'][key]['pin'],'Original projected role SHA differs')
        reference=c.parse_vcoco_role_reference(actions,all_instances,images);pairs=Counter(p.image_id for p in reference.localized_positive_pairs)
        for iid in sorted(selected_ids):
            if people[iid]>=2 and objects[iid]>=2 and pairs[iid]>=1:inventory.append(dict(split=name,image_id=iid,photo_id=banks[iid]['photo_id']))
        actions_by_split[name]=actions;check()
    rt.require(c.pin(c.encode(inventory))==expected['eligibility_inventory_identity'] and len(inventory)==561
        and Counter(r['split']for r in inventory)=={'val':199,'test':362},'Reproduced original complete eligibility inventory differs')
    chosen=[]
    for official,study in (('val','DEV'),('test','RESERVED')):
        ranked=sorted((r for r in inventory if r['split']==official),key=lambda r:
            (hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest(),r['image_id']))
        chosen.extend(dict(r,study_split=study)for r in ranked[:8])
    rt.require(len(chosen)==16 and len({r['photo_id']for r in chosen})==16,'Fixed16 unique-photo slots required')
    return chosen,banks,original_images,instances,actions_by_split


def select_artifacts(metadata,chosen,banks,original,instances,actions,proof,check,ledger):
    records=[]
    for slot,record in enumerate(chosen):
        check();iid=record['image_id'];image=original[iid];bank=banks[iid]
        rows=projected_image(actions[record['split']],iid);grid={k:image[k]for k in ('id','width','height')}
        reference=c.parse_vcoco_role_reference(rows,instances[iid],[grid])
        name=f'reference_{slot:06d}.json';reference_pin=write(metadata/name,dict(schema='world_reward.vcoco_pilot_reference.v1',
            image=grid,instances=instances[iid],actions=rows,localized_positive_pairs=[dict(image_id=p.image_id,
                agent_annotation_id=p.agent_annotation_id,object_annotation_id=p.object_annotation_id,
                row_role_references=p.row_role_references)for p in reference.localized_positive_pairs]),ledger)
        records.append(dict(slot=slot,split=record['study_split'],official_split=record['split'],image_id=iid,
            image=dict(id=iid,width=grid['width'],height=grid['height'],file_name=image['file_name'],partition=image['partition'],
                publisher_image_url=image['coco_url'],flickr_url=image['flickr_url'],photo_id=bank['photo_id'],
                license=4,license_url=c.GRANT,creator_identity='UNKNOWN'),reference_file=name,reference_identity=reference_pin))
    cohort=dict(schema='world_reward.vcoco_role_pilot_cohort.v1',producer_revision=proof['source_binding']['producer_revision'],
        source_binding=proof['source_binding'],census_report_identity=proof['census_report_identity'],namespace=NAMESPACE,
        retry_count=0,replacement_count=0,records=records)
    cohort_pin=write(metadata/'cohort.json',cohort,ledger);return cohort,cohort_pin


def frozen(metadata,report_pin,cohort_pin,proof,check):
    owned=metadata.lstat()
    report=rt.pinned(metadata/'report.json',report_pin,1 << 20);cohort=rt.pinned(metadata/'cohort.json',cohort_pin,1 << 20)
    rt.require(report['status']=='pass' and report['phase']=='select' and report['source_binding']==proof['source_binding']
        and report['cohort_identity']==cohort_pin and report['source_and_inputs_rehashed_after']is True and report['outputs_sealed']is True,
        'Sealed newly frozen selection required')
    rt.require(cohort['source_binding']==proof['source_binding'] and cohort['producer_revision']==proof['source_binding']['producer_revision']
        and cohort['census_report_identity']==proof['census_report_identity'] and cohort['namespace']==NAMESPACE
        and cohort['retry_count']==cohort['replacement_count']==0 and cohort['schema']=='world_reward.vcoco_role_pilot_cohort.v1',
        'New cohort identity differs')
    rows=cohort['records'];rt.require(len(rows)==16 and [r['slot']for r in rows]==list(range(16))
        and [r['split']for r in rows]==['DEV']*8+['RESERVED']*8 and len({r['image']['photo_id']for r in rows})==16,'Every original frozen slot required')
    expected={'report.json':report_pin,'cohort.json':cohort_pin}
    for r in rows:
        check();rt.require(r['reference_file']==f"reference_{r['slot']:06d}.json"
            and rt.identity(metadata/r['reference_file'],4 << 20)==r['reference_identity'],'Reference bytes changed; no values consulted')
        rt.require(set(r)=={'slot','split','official_split','image_id','image','reference_file','reference_identity'}
            and r['official_split']==('val'if r['slot']<8 else'test'),'Original split/slot metadata required')
        image=r['image'];rt.require(set(image)=={'id','width','height','file_name','partition','publisher_image_url','flickr_url','photo_id',
            'license','license_url','creator_identity'}and type(r['image_id'])is int and 0<r['image_id']<10**12
            and image['id']==r['image_id']and image['partition']in('train2014','val2014')
            and image['file_name']==f"COCO_{image['partition']}_{r['image_id']:012d}.jpg"
            and image['publisher_image_url']==f"http://images.cocodataset.org/{image['partition']}/{image['file_name']}"
            and image['license']==4 and image['license_url']==c.GRANT and image['creator_identity']=='UNKNOWN'
            and c.coco.photo_identity(image['flickr_url'])==image['photo_id']
            and all(type(image[k])is int and image[k]>0 for k in ('width','height'))
            and image['width']*image['height']<=ACQUIRE['max_decoded_pixels'],'Original frozen image/rights grid required')
        expected[r['reference_file']]=r['reference_identity']
    rt.require(report['artifact_identities']=={n:p for n,p in expected.items()if n!='report.json'},'Every16 original reference pin required')
    return rows,{'.':c.state(metadata),**inventory(metadata,owned,expected,0o500,4 << 20)}


def acquire(rows,excluded,public,deadline,ledger,*,request=acq.fetch):
    """Original publisher bytes only; missing slots and all duplicate copies retained."""
    def slot(row):
        result=dict(slot=row['slot'],split=row['split'],image_id=row['image_id'],status='unavailable');phase='original_http';owned=None
        try:
            c.check(deadline);image=row['image'];name=f"COCO_{image['partition']}_{row['image_id']:012d}.jpg"
            rt.require(image['partition']in('train2014','val2014') and image['file_name']==name and image['license']==4
                and image['license_url']==c.GRANT and image['publisher_image_url']==f"http://images.cocodataset.org/{image['partition']}/{name}",
                'Original frozen publisher filename/grant required')
            raw=request(BUCKET+image['partition']+'/'+name,ACQUIRE['max_image_bytes'],deadline,ACQUIRE['request_timeout'])
            phase='original_byte_identity';rt.require(0<len(raw)<=ACQUIRE['max_image_bytes'],'Bounded original JPEG required')
            md5=base64.b64encode(hashlib.md5(raw).digest()).decode();rt.require(md5 not in excluded['md5'],'Historical byte alias rejected')
            phase='original_jpeg_header';header=acq.jpeg_header(raw,ACQUIRE['max_decoded_pixels'])
            rt.require((header['width'],header['height'])==(image['width'],image['height']),'Original JPEG/source grid differs')
            c.check(deadline);path=public/f"image_{row['slot']:06d}.jpg";image_pin=raw_write(path,raw,ledger);owned=path.lstat()
            result.update(status='acquired',image_pin=image_pin,original_md5=md5,jpeg_header=header)
        except Exception as exc:
            if owned is not None:
                current=path.lstat();rt.require((current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino)
                    and stat.S_ISREG(current.st_mode) and current.st_nlink==1,'Cannot remove foreign RGB inode');path.unlink();ledger.pop(path.name)
            result.update(reason=phase,error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','OSError','TimeoutError','HTTPError','URLError') else 'other')
        return result
    with ThreadPoolExecutor(max_workers=ACQUIRE['workers'])as pool:records=list(pool.map(slot,rows))
    c.check(deadline);counts=Counter(r['original_md5']for r in records if r['status']=='acquired')
    for r in records:
        if r['status']=='acquired' and counts[r['original_md5']]>1:
            p=public/f"image_{r['slot']:06d}.jpg";rt.require(rt.identity(p,ACQUIRE['max_image_bytes'])==r['image_pin'],'Conflicting original inode differs');p.unlink();ledger.pop(p.name)
            r.update(status='unavailable',reason='duplicate_cohort_bytes')
            for key in ('image_pin','jpeg_header','original_md5'):del r[key]
    images=[];mapping=[]
    for r in records:
        if r['status']!='acquired':continue
        opaque=hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()[:32];file=f"image_{r['slot']:06d}.jpg"
        images.append(dict(image_id=opaque,file=file,**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
        mapping.append(dict(slot=r['slot'],public_image_id=opaque,public_file=file,image_pin=r['image_pin']))
    rt.require(all(set(r)==PUBLIC_KEYS for r in images) and sum(r['bytes']for r in images)<=ACQUIRE['max_total_image_bytes'],'Public six-key byte boundary required')
    public_pin=write(public/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=images),ledger)
    acquired=Counter(r['split']for r in records if r['status']=='acquired');gate=acquired['DEV']>=6 and acquired['RESERVED']>=6
    return dict(records=records,public_mappings=mapping,public_inputs_identity=public_pin,
        counts=dict(slots=16,acquired=len(images),missing=16-len(images),acquired_DEV=acquired['DEV'],acquired_RESERVED=acquired['RESERVED']),
        availability_gate_passed=gate,decision='READY_PENDING_SEPARATE_RGB_BANK'if gate else'CLOSED_AVAILABILITY_NO_MODEL_OR_RETRY')


def run(phase,pins):
    rt.require(phase in ('select','acquire'),'Explicit prospective phase required')
    started=time.monotonic();deadline=started+(SELECT if phase=='select'else ACQUIRE)['budget_seconds']
    def interrupted(*_):raise TimeoutError('Frozen pilot interrupted')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        rt.require(os.geteuid()==0 and sys.platform=='linux'and os.uname().nodename=='world-reward-ncc-h100-02','Exact Azure VM02 CPU host required')
        code,rev=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'];rt.require(Path(__file__).resolve()==code/HELPERS[0],'Immutable pilot driver required')
        cfg,old_cfg,old_report,excluded,proof=authenticate(code,rev);c.check(deadline)
        metadata=DATA/'metadata'
        if phase=='select':
            rt.require(not pins and not rt.canonical(DATA).exists()and DATA.parent.is_dir(),'Fresh pilot selection required')
            DATA.mkdir(mode=0o700);DATA.chmod(0o700);metadata.mkdir(mode=0o700);out=metadata
        else:
            rt.require(len(pins)==4 and DATA.is_dir()and not(DATA/'inputs').exists()and not(DATA/'eval_private').exists(),'Fresh separately pinned acquisition required')
            namespace=namespace_identity(DATA)
            selection_pin=dict(bytes=int(pins[0]),sha256=pins[1]);cohort_pin=dict(bytes=int(pins[2]),sha256=pins[3])
            rows,frozen_states=frozen(metadata,selection_pin,cohort_pin,proof,lambda:c.check(deadline))
            public=DATA/'inputs';out=DATA/'eval_private';public.mkdir(mode=0o700);out.mkdir(mode=0o700)
        owned=out.lstat();expected={};public_expected={};public_owned=public.lstat()if phase=='acquire'else None
        report=dict(schema='world_reward.vcoco_role_prepare.v1',phase=phase,status='fail',producer_revision=rev,
            source_binding=proof['source_binding'],census_binding=proof['census_binding'],census_report_identity=proof['census_report_identity'],
            input_proof=proof,
            RGB_read=phase=='acquire',RGB_decoded=False,network_used=phase=='acquire',GPU_used=False,
            models_loaded=False,FIT_performed=False,predictions_read=False,selection_performed=phase=='select',
            fresh_reference_values_consulted=phase=='select',underlying_creator_identity_verified=False,training_overlap_verified=False,
            challenge_overlap_verified=False,adopted=False,source_and_inputs_rehashed_after=False,outputs_sealed=False)
        try:
            if phase=='select':
                chosen,banks,original,instances,actions=reproduce(old_cfg,excluded,old_report,lambda:c.check(deadline))
                cohort,cohort_pin=select_artifacts(metadata,chosen,banks,original,instances,actions,proof,lambda:c.check(deadline),expected)
                rt.require(set(expected)=={'cohort.json'}|{f'reference_{i:06d}.json'for i in range(16)},'Every16 fresh reference file required')
                report.update(cohort_identity=cohort_pin,selected_slots=16,reproduced_inventory_identity=old_report['eligibility_inventory_identity'],
                    decision='FROZEN16_PENDING_SEPARATE_ORIGINAL_RGB_ACQUISITION',status='pass')
            else:
                report.update(selected_slots=16,selection_report_identity=selection_pin,cohort_identity=cohort_pin,
                    attribution_records=[dict(slot=r['slot'],split=r['split'],image=r['image'])for r in rows],
                    records=[dict(slot=r['slot'],split=r['split'],image_id=r['image_id'],status='unavailable',reason='incomplete_acquisition')for r in rows])
                report.update(acquire(rows,excluded,public,deadline-20,public_expected));report['status']='pass'if report['availability_gate_passed']else'fail'
        except BaseException as exc:
            report.update(status='fail',decision='CLOSED_PILOT_NO_RGB_RETRY_OR_FIT',error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','TimeoutError','OSError','KeyError') else 'other')
        finally:
            public_states=None
            try:
                if phase=='acquire':
                    public_states=inventory(public,public_owned,public_expected,0o700,16 << 20)
                    public.chmod(0o500);acq.sync_directory(public);report.update(public_artifact_identities=dict(public_expected),public_outputs_sealed=True)
            except BaseException:
                report.update(status='fail',decision='CLOSED_PILOT_PUBLIC_SEAL',public_seal_failed=True)
            try:
                rt.require(authenticate(code,rev)==(cfg,old_cfg,old_report,excluded,proof),'Source/original inputs changed after pilot work')
                if phase=='acquire':
                    rt.require(namespace_identity(DATA)==namespace,'Original root-private namespace changed')
                    rt.require(frozen(metadata,selection_pin,cohort_pin,proof,lambda:c.check(deadline))[1]==frozen_states,
                        'Frozen metadata/ref bytes or modes changed')
                    rt.require(public_states is not None and inventory(public,public_owned,public_expected,0o500,16 << 20)==public_states,
                        'Original public RGB bytes changed')
                report['source_and_inputs_rehashed_after']=True;c.check(deadline)
            except BaseException:
                report.update(status='fail',decision='CLOSED_PILOT_POSTHASH',posthash_failed=True)
            report['artifact_identities']=dict(expected);publish(out,report,deadline,started,owned,expected)
        print(json.dumps(dict(status=report['status'],phase=phase,decision=report['decision'],report_identity=rt.identity(out/'report.json',1 << 20)),sort_keys=True))
        return report
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    rt.require(len(sys.argv)>=2 and sys.argv[1]in('select','acquire'),'Explicit prospective phase required')
    sys.exit(0 if run(sys.argv[1],sys.argv[2:])['status']=='pass'else 1)

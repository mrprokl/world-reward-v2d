"""Two frozen COCO phases: private metadata census, then original RGB only.

No model, challenge data, creator reconfirmation, replacement or image edits.
The image license is the publisher's recorded grant, not creator authentication.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
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
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt
import openimages_joint_pair_acquire as acq

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/coco_proposal_v1')
ENTRY = 'run_coco_proposal_prepare'
CONFIG = 'configs/coco_proposal_v1.json'
HELPERS = ('infra/coco_proposal_prepare.py', 'infra/run_coco_proposal_prepare.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/openimages_joint_pair_acquire.py')
NAMESPACE = 'world_reward.coco_proposal_v1/'
LICENSE_URL = 'http://creativecommons.org/licenses/by/2.0/'
BUCKET = 'https://s3.amazonaws.com/images.cocodataset.org/'
ARCHIVE = 'annotations_trainval2017.zip'
MEMBER = 'annotations/instances_val2017.json'
PUBLIC_KEYS = {'image_id', 'file', 'bytes', 'sha256', 'width', 'height'}


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    raw = encode(value); rt.write(path, raw); return pin(raw)


def check(deadline):
    rt.require(time.monotonic() < deadline, 'Inclusive phase deadline reached')


def photo_identity(url):
    parsed = urllib.parse.urlsplit(url)
    rt.require(parsed.scheme in ('http', 'https') and not parsed.query and not parsed.fragment
        and re.fullmatch(r'(?:farm[0-9]+|live)\.staticflickr\.com', parsed.netloc), 'Native Flickr static URL required')
    match = re.fullmatch(r'/[0-9]+/([0-9]+)_[0-9a-f]+(?:_[a-z])?\.jpg', parsed.path)
    rt.require(match is not None, 'Native Flickr photo identity required'); return match.group(1)


def exclusions(values):
    result = {k:set() for k in ('ids','photos','md5')}
    for value, count in zip(values, (16,128,64,32)):
        if count in (16,128):
            rows = value['public_metadata']; ids = value['selected_ids']
            rt.require(len(ids)==len(set(ids))==len(rows)==count and {r['ImageID'] for r in rows}==set(ids), 'All historical slots required')
        else:
            schema = 'world_reward.openimages_joint_pair_cohort.v1' if count==64 else 'world_reward.proposal_external_cohort.v1'
            slots=value['records']; split=['DEV']*(count//2)+[('TEST' if count==64 else 'RESERVED')]*(count//2)
            rt.require(value['schema']==schema and value['no_replacements'] is True
                and len(slots)==count and [r['slot'] for r in slots]==list(range(count))
                and [r['split'] for r in slots]==split, 'Exact historical frozen cohort required')
            rows=[r['publisher_metadata'] for r in slots]
        for row in rows:
            iid=row['ImageID']; rt.require(re.fullmatch('[0-9a-f]{16}', iid) and iid not in result['ids'], 'Disjoint historical240 IDs required')
            result['ids'].add(iid); result['photos'].add(acq.excluded_photo_identity(row['OriginalLandingURL']))
            if row['OriginalMD5']: result['md5'].add(acq.md5_identity(row['OriginalMD5']))
    rt.require(len(result['ids'])==240, 'All historical240 slots excluded'); return result


def val_metadata(raw):
    """Read one original val member; no train/caption bytes are inflated."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist(); names=[e.filename for e in entries]
        rt.require(0<len(entries)<=32 and len(names)==len(set(names)) and names.count(MEMBER)==1, 'Unique original val ZIP member required')
        for e in entries:
            mode=e.external_attr>>16
            if e.filename=='annotations/':
                rt.require(e.is_dir() and e.file_size==0 and stat.S_IFMT(mode) in (0,stat.S_IFDIR), 'Exact empty annotation directory required'); continue
            rt.require(re.fullmatch(r'annotations/[a-z_0-9]+\.json', e.filename) and not e.flag_bits&1
                and e.compress_type in (0,8) and stat.S_IFMT(mode) in (0,stat.S_IFREG)
                and 0<e.file_size<=512<<20, 'Bounded regular original annotation members required')
        target=archive.getinfo(MEMBER); rt.require(target.file_size<=64<<20, 'Bounded original val metadata required')
        content=archive.read(target)
    return rt.strict(content), pin(content)


def countable(a):
    box=a['bbox']; rt.require(type(a['iscrowd']) is int and a['iscrowd'] in (0,1)
        and type(box) is list and len(box)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in box)
        and type(a['area']) in (int,float) and math.isfinite(a['area']), 'Native finite annotation required')
    return a['iscrowd']==0 and a['area']>0 and box[2]>0 and box[3]>0


def separated(rows):
    for i,a in enumerate(rows):
        x,y,w,h=a['bbox']
        for b in rows[i+1:]:
            u,v,s,t=b['bbox']
            if min(x+w,u+s)<=max(x,u) or min(y+h,v+t)<=max(y,v): return True
    return False


def select(value, excluded):
    licenses=value['licenses']; categories=value['categories']; images=value['images']; annotations=value['annotations']
    rt.require(len({r['id'] for r in licenses})==len(licenses) and len({r['id'] for r in categories})==len(categories), 'Unique native catalogs required')
    grant=[r for r in licenses if r['id']==4]; people=[r['id'] for r in categories if r['name']=='person']
    rt.require(len(grant)==1 and grant[0]['url']==LICENSE_URL and len(people)==1, 'Exact CC-BY2 grant and native person category required')
    category_ids={r['id'] for r in categories}; bank={}; all_ids=set()
    for image in images:
        iid=image['id']; rt.require(type(iid) is int and 0<=iid<10**12 and iid not in bank, 'Unique native image ID required')
        bank[iid]=[]
    for a in annotations:
        rt.require(type(a['id']) is int and a['id'] not in all_ids and a['image_id'] in bank
            and a['category_id'] in category_ids, 'Unique bound native annotation required'); all_ids.add(a['id'])
        if countable(a): bank[a['image_id']].append(a)
    eligible=[]
    for image in images:
        iid=image['id']; refs=bank[iid]; persons=[a for a in refs if a['category_id']==people[0]]; objects=[a for a in refs if a['category_id']!=people[0]]
        if image['license']!=4 or len(persons)<2 or len(objects)<2 or not separated(persons) or not separated(objects): continue
        rt.require(type(image['width']) is int and type(image['height']) is int and image['width']>0 and image['height']>0
            and image['width']*image['height']<=16<<20 and image['file_name']==f'{iid:012d}.jpg'
            and image['coco_url']==f'http://images.cocodataset.org/val2017/{iid:012d}.jpg', 'Original val grid/filename required')
        photo=photo_identity(image['flickr_url'])
        if photo not in excluded['photos']: eligible.append(dict(image_id=iid,image=image,photo_id=photo,annotations=refs))
    eligible.sort(key=lambda r:(hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest(),r['image_id']))
    rows=[]; photos=set()
    for r in eligible:
        if r['photo_id'] in photos: continue
        rows.append(dict(slot=len(rows),split='DEV' if len(rows)<16 else 'RESERVED',**r)); photos.add(r['photo_id'])
        if len(rows)==32: break
    rt.require(len(rows)==32, 'Insufficient frozen32 metadata slots; no RGB allowed')
    return rows, dict(eligible=len(eligible),selected=32,excluded_ids=len(excluded['ids']))


def configuration(code, source):
    cfg=rt.pinned(code/CONFIG,source['helpers'][CONFIG],16<<10)
    fixed=dict(schema='world_reward.coco_proposal_prepare.v1',output=str(DATA),slots=32,dev=16,reserved=16,
        workers=6,overall=300,request_timeout=15,max_image_bytes=16<<20,max_decoded_pixels=16<<20,
        min_acquired_dev=12,retry_count=0,no_replacements=True,hash_namespace=NAMESPACE,
        archive_url=BUCKET+'annotations/'+ARCHIVE,archive_bytes=252907541,archive_md5='f4bbac642086de4f52a3fdda2de5fa2c',member=MEMBER)
    rt.require(all(type(cfg.get(k)) is type(v) and cfg[k]==v for k,v in fixed.items()), 'Frozen COCO recipe differs')
    rt.require(set(cfg['reused_helper_pins'])==set(HELPERS[3:]), 'Complete original helper pins required')
    for module,name in ((rt,HELPERS[3]),(acq,HELPERS[4])):
        rt.require(source['helpers'][name]==cfg['reused_helper_pins'][name] and Path(module.__file__).resolve()==code/name, 'Original helper identity/origin differs')
    return cfg


def historical(cfg):
    values=[]; frozen={}
    rt.require(len(cfg['historical'])==4, 'Four original historical cohorts required')
    for row in cfg['historical']:
        path=rt.canonical(row['path']); rt.require(rt.identity(path,1<<20,readonly=row['readonly'])==row['pin'], 'Historical metadata input differs')
        values.append(rt.strict(path.read_bytes())); frozen[str(path)]=row['pin']
    return exclusions(values),frozen


def census(cfg, source, revision, metadata, deadline, *, request=acq.fetch):
    excluded,frozen=historical(cfg); check(deadline)
    archive_path=metadata/ARCHIVE; cache=cfg['annotation_cache']
    if cache is None:
        raw=request(cfg['archive_url'],cfg['archive_bytes'],deadline,cfg['request_timeout'])
        rt.require(len(raw)==cfg['archive_bytes'] and hashlib.md5(raw).hexdigest()==cfg['archive_md5'], 'Whole publisher annotation bytes/MD5 differ')
        rt.write(archive_path,raw); archive_pin=pin(raw)
    else:
        receipt=rt.pinned(Path(cache['report']['path']),cache['report']['pin'],1<<20)
        archive_path=rt.canonical(cache['archive_path']); archive_pin=receipt['archive_identity']
        rt.require(receipt['archive_url']==cfg['archive_url'] and receipt['archive_publisher_md5']==cfg['archive_md5']
            and receipt['status']=='pass' and archive_pin['bytes']==cfg['archive_bytes']
            and rt.identity(archive_path, cfg['archive_bytes'])==archive_pin, 'Authenticated whole-archive cache required')
        raw=archive_path.read_bytes(); rt.require(hashlib.md5(raw).hexdigest()==cfg['archive_md5'], 'Cached publisher MD5 differs')
    value,member_pin=val_metadata(raw); rows,counts=select(value,excluded); check(deadline)
    cohort=dict(schema='world_reward.coco_proposal_cohort.v1',producer_revision=revision,source_binding=source,
        configuration_identity=source['helpers'][CONFIG],records=rows,categories=value['categories'],licenses=value['licenses'],
        hash_namespace=NAMESPACE,freeze_before_rgb=True,no_replacements=True,author_identity='UNKNOWN',author_disjointness_verified=False)
    for row in rows:
        name=f"reference_{row['slot']:06d}.json"
        row['reference_identity']=write(metadata/name,dict(annotations=row.pop('annotations')))
        row['reference_file']=name
    cohort_pin=write(metadata/'cohort.json',cohort)
    rt.require(rt.identity(archive_path,cfg['archive_bytes'])==archive_pin and historical(cfg)==(excluded,frozen), 'Census inputs changed after freeze')
    for row in rows: rt.require(rt.identity(metadata/row['reference_file'],4<<20)==row['reference_identity'], 'Frozen reference bytes changed')
    return dict(archive_url=cfg['archive_url'],archive_path=str(archive_path),archive_identity=archive_pin,
        archive_publisher_md5=cfg['archive_md5'],member_identity=member_pin,cohort_identity=cohort_pin,
        exclusion_inputs=frozen,counts=counts,freeze_before_rgb=True,rgb_read=False)


def frozen_cohort(cfg, source, metadata, report_pin, cohort_pin):
    report=rt.pinned(metadata/'report.json',report_pin,2<<20); cohort=rt.pinned(metadata/'cohort.json',cohort_pin,16<<20)
    rt.require(report['schema']==cfg['schema'] and report['phase']=='census' and report['status']=='pass'
        and report['source_binding']==source and report['configuration_identity']==source['helpers'][CONFIG]
        and report['cohort_identity']==cohort_pin and report['freeze_before_rgb'] is True and report['rgb_read'] is False
        and report['source_and_inputs_rehashed_after'] is True and report['outputs_sealed'] is True
        and cohort['schema']=='world_reward.coco_proposal_cohort.v1' and cohort['source_binding']==source
        and cohort['configuration_identity']==source['helpers'][CONFIG] and cohort['hash_namespace']==NAMESPACE
        and cohort['freeze_before_rgb'] is True and cohort['no_replacements'] is True, 'Authentic prior metadata freeze required')
    excluded,frozen=historical(cfg); expected_archive=metadata/ARCHIVE if cfg['annotation_cache'] is None else Path(cfg['annotation_cache']['archive_path'])
    rt.require(report['exclusion_inputs']==frozen and Path(report['archive_path'])==expected_archive and report['archive_url']==cfg['archive_url']
        and report['archive_publisher_md5']==cfg['archive_md5'] and report['archive_identity']['bytes']==cfg['archive_bytes']
        and rt.identity(Path(report['archive_path']),cfg['archive_bytes'])==report['archive_identity'], 'Original census input identities differ')
    rows=cohort['records']; rt.require(len(rows)==32 and [r['slot'] for r in rows]==list(range(32))
        and [r['split'] for r in rows]==['DEV']*16+['RESERVED']*16
        and len({r['image_id'] for r in rows})==len({r['photo_id'] for r in rows})==32
        and all(photo_identity(r['image']['flickr_url'])==r['photo_id'] and r['photo_id'] not in excluded['photos'] for r in rows), 'Exact disjoint32 frozen slots required')
    for row in rows:
        rt.require(set(row)=={'slot','split','image_id','image','photo_id','reference_file','reference_identity'}
            and row['reference_file']==f"reference_{row['slot']:06d}.json"
            and rt.identity(metadata/row['reference_file'],4<<20)==row['reference_identity'], 'Exact private reference bytes required; no semantic decoding')
    return rows,excluded,report


def acquire(rows, excluded, cfg, public, deadline, *, request=acq.fetch):
    """Request only official original RGB, retaining every frozen missing slot."""
    def slot(r):
        result=dict(slot=r['slot'],split=r['split'],image_id=r['image_id'],status='unavailable'); phase='http'; created=False
        try:
            check(deadline); image=r['image']; name=f"{r['image_id']:012d}.jpg"
            rt.require(image['file_name']==name and image['license']==4, 'Original frozen official filename/grant required')
            raw=request(BUCKET+'val2017/'+name,cfg['max_image_bytes'],deadline,cfg['request_timeout']); phase='byte_bound'
            rt.require(0<len(raw)<=cfg['max_image_bytes'], 'Bounded original JPEG required'); phase='historical_md5'
            digest=base64.b64encode(hashlib.md5(raw).digest()).decode()
            rt.require(digest not in excluded['md5'], 'Historical original bytes cannot be reused'); phase='jpeg_grid'
            header=acq.jpeg_header(raw,cfg['max_decoded_pixels'])
            rt.require((header['width'],header['height'])==(image['width'],image['height']), 'Original metadata/JPEG grid differs'); check(deadline)
            image_pin=pin(raw); file=f"image_{r['slot']:06d}.jpg"; phase='publish'
            rt.write(public/file,raw); created=True
            rt.require(rt.identity(public/file,cfg['max_image_bytes'])==image_pin, 'Original RGB publication differs')
            result.update(status='acquired',image_pin=image_pin,jpeg_header=header,original_md5=digest)
        except Exception as exc:
            owned=public/f"image_{r['slot']:06d}.jpg"
            if created:
                rt.canonical(owned); rt.require(owned.is_file() and owned.stat().st_nlink==1, 'Cannot clean foreign output'); owned.unlink()
            result.update(reason=phase,error_type=type(exc).__name__ if type(exc) in (ValueError,OSError,TimeoutError) else 'other')
        return result
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool: records=list(pool.map(slot,rows))
    check(deadline); images=[]; mappings=[]
    duplicates={r['original_md5'] for r in records if r['status']=='acquired'
        and sum(s.get('original_md5')==r['original_md5'] and s['status']=='acquired' for s in records)>1}
    for r in records:
        if r.get('original_md5') not in duplicates: continue
        owned=public/f"image_{r['slot']:06d}.jpg"
        rt.require(rt.identity(owned,cfg['max_image_bytes'])==r['image_pin'], 'Cannot remove foreign conflicting RGB'); owned.unlink()
        r.update(status='unavailable',reason='duplicate_cohort_bytes')
        for key in ('image_pin','jpeg_header','original_md5'): del r[key]
    for r in records:
        if r['status']!='acquired': continue
        file=f"image_{r['slot']:06d}.jpg"; opaque=hashlib.sha256((NAMESPACE+f"{r['image_id']:012d}").encode()).hexdigest()[:32]
        images.append(dict(image_id=opaque,file=file,**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
        mappings.append(dict(slot=r['slot'],public_image_id=opaque,public_file=file,image_pin=r['image_pin']))
    rt.require(all(set(r)==PUBLIC_KEYS for r in images), 'Exact public six-key RGB boundary required')
    public_pin=write(public/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=images))
    counts=dict(slots=32,acquired=len(images),missing=32-len(images),acquired_DEV=sum(r['status']=='acquired' and r['split']=='DEV' for r in records),
                acquired_RESERVED=sum(r['status']=='acquired' and r['split']=='RESERVED' for r in records))
    return dict(records=records,public_mappings=mappings,public_inputs_identity=public_pin,counts=counts,
        decision='READY_PENDING_SEPARATE_NATIVE_BANK' if counts['acquired_DEV']>=cfg['min_acquired_dev'] else 'INCONCLUSIVE_CLOSED_NO_MODEL')


def verify_public(public, result, cfg):
    value=rt.pinned(public/'manifest.json',result['public_inputs_identity'],1<<20)
    expected=[]
    acquired=[r for r in result['records'] if r['status']=='acquired']
    rt.require(len(result['records'])==32 and [r['slot'] for r in result['records']]==list(range(32))
        and len(acquired)==len(result['public_mappings']) and len({m['public_image_id'] for m in result['public_mappings']})==len(acquired), 'Complete32 acquisition ledger required')
    for r,m in zip(acquired,result['public_mappings']):
        rt.require(m['public_file']==f"image_{r['slot']:06d}.jpg" and m['image_pin']==r['image_pin'], 'Original slot/pin mapping required')
        rt.require(m['slot']==r['slot'] and rt.identity(public/m['public_file'],cfg['max_image_bytes'])==r['image_pin'], 'Exact acquired original slots required')
        expected.append(dict(image_id=m['public_image_id'],file=m['public_file'],**r['image_pin'],width=r['jpeg_header']['width'],height=r['jpeg_header']['height']))
    rt.require(value==dict(schema='world_reward.rgb_proposal_inputs.v1',images=expected) and all(set(r)==PUBLIC_KEYS for r in expected)
        and len(expected)==len(result['public_mappings'])==result['counts']['acquired']
        and {p.name for p in public.iterdir()}=={'manifest.json'}|{r['file'] for r in expected}, 'No private fields, missing fake RGB or foreign leaves allowed')
    rt.require(all(p.lstat().st_uid==os.geteuid() and stat.S_IMODE(p.lstat().st_mode)==0o400 for p in public.iterdir()), 'Root-only immutable public leaves required')


def run(phase, pins=()):
    started=time.monotonic(); deadline=started+300
    rt.require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02', 'Exact Azure root CPU host required')
    revision,code=os.environ['WR_CODE_REVISION'],Path(os.environ['WR_CODE'])
    rt.require(Path(__file__).resolve()==code/HELPERS[0], 'Immutable original driver required')
    source=rt.source(ROOT,code,revision,ENTRY,HELPERS); cfg=configuration(code,source); metadata=DATA/'metadata'
    if phase=='census':
        rt.require(not rt.canonical(DATA).exists() and DATA.parent.is_dir(), 'Fresh census namespace required'); DATA.mkdir(mode=0o700); metadata.mkdir(mode=0o700)
        output=metadata
    else:
        report_pin,cohort_pin=pins; rows,excluded,census_report=frozen_cohort(cfg,source,metadata,report_pin,cohort_pin)
        rt.require(not (DATA/'inputs').exists() and not (DATA/'eval_private').exists(), 'Fresh acquisition namespace required')
        DATA.chmod(0o700); (DATA/'inputs').mkdir(mode=0o700); (DATA/'eval_private').mkdir(mode=0o700); output=DATA/'eval_private'
    report=dict(schema=cfg['schema'],phase=phase,status='fail',producer_revision=revision,source_binding=source,
        configuration_identity=source['helpers'][CONFIG],source_and_inputs_rehashed_after=False,outputs_sealed=False,
        models_loaded=False,gpu_used=False,predictions_read=False,challenge_inputs_used=False,retry_count=0,replacement_count=0,
        author_identity='UNKNOWN',author_disjointness_verified=False,creator_account_identity_verified=False,
        image_license='publisher_recorded_CC-BY-2.0',annotations_license='CC-BY-4.0',raw_redistribution_allowed_by_this_job=False,
        training_overlap_verified=False,quality_verified=False,adopted=False,reference_values_decoded=phase=='census',
        rgb_requests_before_freeze=0,reference_values_read_for_acquisition=False,network_used=True,original_image_transformations=0)
    def interrupted(*_): raise TimeoutError('Phase interrupted')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        if phase=='census': report.update(census(cfg,source,revision,metadata,deadline))
        else:
            report.update(selected_slots=rows,census_report_identity=report_pin,cohort_identity=cohort_pin,freeze_before_rgb=True)
            report.update(acquire(rows,excluded,cfg,DATA/'inputs',deadline-15)); verify_public(DATA/'inputs',report,cfg)
            rt.require(frozen_cohort(cfg,source,metadata,report_pin,cohort_pin)==(rows,excluded,census_report), 'Original freeze changed during acquisition')
        rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==source and configuration(code,source)==cfg, 'Full source/config changed')
        report.update(source_and_inputs_rehashed_after=True,status='pass'); check(deadline)
    except BaseException as exc:
        report['error_type']=type(exc).__name__ if type(exc) in (ValueError,OSError,TimeoutError,KeyError,zipfile.BadZipFile) else 'other'
    finally:
        signal.alarm(0)
        path=output/('report.json' if phase=='census' else 'manifest.json')
        # Keep the same writable FD for a late deadline downgrade after sealing.
        with path.open('xb') as stream:
            os.fchmod(stream.fileno(),0o400)
            for p in (*output.rglob('*'),*((DATA/'inputs').iterdir() if phase=='acquire' else ())):
                rt.canonical(p); rt.require(p.is_file(), 'No foreign output node allowed'); p.chmod(0o400)
            for p in ((DATA/'inputs',output,DATA) if phase=='acquire' else (output,DATA)): p.chmod(0o500)
            acq.sync_directory(output); acq.sync_directory(DATA)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started)
            if report['elapsed_seconds']>=300: report.update(status='fail',error_type='TimeoutError')
            raw=encode(report); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if time.monotonic()>=deadline and report['status']=='pass':
                report.update(status='fail',error_type='TimeoutError',elapsed_seconds=time.monotonic()-started)
                raw=encode(report); stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        for s,h in handlers.items(): signal.signal(s,h)
        print(json.dumps(dict(status=report['status'],phase=phase,report_identity=pin(raw),cohort_identity=report.get('cohort_identity'),counts=report.get('counts'),decision=report.get('decision')),sort_keys=True))
    return report


if __name__=='__main__':
    rt.require(len(sys.argv) in (2,6) and sys.argv[1] in ('--census','--acquire'), 'Only fixed phases and independent freeze pins allowed')
    phase=sys.argv[1][2:]; rt.require((phase=='census')==(len(sys.argv)==2), 'Acquire requires independent report/cohort pins')
    pins=() if phase=='census' else (dict(bytes=int(sys.argv[2]),sha256=sys.argv[3]),dict(bytes=int(sys.argv[4]),sha256=sys.argv[5]))
    report=run(phase,pins); sys.exit(0 if report['status']=='pass' else 1)

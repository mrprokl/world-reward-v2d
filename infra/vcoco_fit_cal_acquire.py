"""Original48 publisher JPEGs only; private identity proof, no reference semantics."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_fit_cal_context as context

c,rt,prep=context.c,context.rt,context.prep
ROOT,DATA=context.ROOT,context.freeze.DATA
ENTRY='run_vcoco_fit_cal_acquire'
HELPERS=tuple(dict.fromkeys(('infra/vcoco_fit_cal_acquire.py','infra/run_vcoco_fit_cal_acquire.sh',
    'infra/vcoco_fit_cal_context.py',*context.freeze.HELPERS)))
RECIPE=dict(schema='world_reward.vcoco_fit_cal_acquisition.v1',budget_seconds=1200,outer_seconds=1215,
    http_budget_seconds=300,workers=4,request_timeout=15,max_image_bytes=16 << 20,max_decoded_pixels=16 << 20,
    max_total_image_bytes=768 << 20,max_host_memory_bytes=8 << 30,cpu_threads=4,
    selected_slots=48,FIT_slots=32,CAL_slots=16,all_48_acquired_required=True,retry_count=0,replacement_count=0)


def authenticate(code,revision,checkpoint):
    rt.require(Path(context.__file__).resolve()==code/'infra/vcoco_fit_cal_context.py','Current context origin required')
    return context.authenticate(code,revision,ENTRY,HELPERS,checkpoint)


def catalog(cfg,cohort,checkpoint):
    """Two complete image/license metadata streams; annotation arrays skipped."""
    wanted={r['image_id']:r for r in cohort['records']};found={};seen=set();expanded={}
    path=cfg['inputs']['coco_archive']['path'];specs={r['member']:r for r in cfg['archive']['member_catalogue']}
    native=rt.pinned(cfg['inputs']['metadata_report']['path'],cfg['inputs']['metadata_report']['pin'])
    expected={r['member']:dict(bytes=r['expanded_bytes'],sha256=r['expanded_sha256'])for r in native['archive_members']}
    for partition,member in cfg['archive']['instances_members'].items():
        licenses=[];selected=[]
        with c.member_stream(path,member,specs[member],checkpoint)as handle:
            for key,raw in c.js.iter_object_arrays(handle,('images','licenses'),check=checkpoint,
                max_bytes=cfg['max_expanded_bytes'],row_bytes=cfg['max_row_bytes']):
                value=c.js.strict_decode(raw)
                if key=='licenses':licenses.append(value);continue
                iid=value['id'];rt.require(type(iid)is int and 0<iid<10**12 and iid not in seen,'Unique original catalog IDs required');seen.add(iid)
                if iid in wanted:selected.append(value)
        expanded[member]=dict(bytes=handle.size,sha256=handle.digest.hexdigest())
        rt.require(expanded[member]==expected[member],'Complete original catalog SHA/CRC differs')
        grants=[r for r in licenses if r['id']==4]
        rt.require(len({r['id']for r in licenses})==len(licenses)and len(grants)==1 and grants[0]['url']==c.GRANT,'Original publisher CC-BY2 metadata required')
        for image in selected:
            iid=image['id'];row=wanted[iid];name=f'COCO_{partition}_{iid:012d}.jpg';photo=c.coco.photo_identity(image['flickr_url'])
            rt.require(iid not in found and image['license']==4 and photo==row['photo_id']
                and image['file_name']==name and image['coco_url']==f'http://images.cocodataset.org/{partition}/{name}'
                and all(type(image[n])is int and image[n]>0 for n in('width','height'))
                and image['width']*image['height']<=RECIPE['max_decoded_pixels'],'Original frozen image/grant/grid required')
            found[iid]=dict(row,width=image['width'],height=image['height'],partition=partition,file_name=name,
                original_url=prep.BUCKET+partition+'/'+name,license_id=4,license_url=c.GRANT,creator_identity='UNKNOWN')
    rt.require(set(found)==set(wanted)and len(found)==48,'All48 catalog identities required before HTTP')
    return [found[r['image_id']]for r in cohort['records']],expanded


def acquire(rows,excluded,public,deadline,ledger,*,request=prep.acq.fetch):
    """One request per48 frozen slots; no filters/substitutes/model work."""
    def slot(row):
        record=dict(slot=row['slot'],split=row['split'],image_id=row['image_id'],status='unavailable');phase='original_http'
        try:
            c.check(deadline);raw=request(row['original_url'],RECIPE['max_image_bytes'],deadline,RECIPE['request_timeout'])
            phase='original_bytes';rt.require(type(raw)is bytes and 0<len(raw)<=RECIPE['max_image_bytes'],'Bounded original bytes required')
            md5=base64.b64encode(hashlib.md5(raw).digest()).decode();rt.require(md5 not in excluded['md5'],'Historical byte alias rejected')
            phase='original_header';header=prep.acq.jpeg_header(raw,RECIPE['max_decoded_pixels'])
            rt.require((header['width'],header['height'])==(row['width'],row['height']),'Original JPEG metadata grid differs');c.check(deadline)
            pin=prep.raw_write(public/f"image_{row['slot']:06d}.jpg",raw,ledger)
            record.update(status='acquired',image_pin=pin,original_md5=md5,jpeg_header=header)
        except Exception as exc:record.update(failed_phase=phase,error_type=type(exc).__name__ if type(exc).__name__ in('ValueError','TimeoutError','OSError','HTTPError','URLError')else'other')
        return record
    rt.require(len(rows)==48 and [r['slot']for r in rows]==list(range(48))
        and [r['split']for r in rows]==['FIT']*32+['CAL']*16
        and len({r['image_id']for r in rows})==48 and all(type(r['slot'])is int and type(r['image_id'])is int and 0<r['image_id']<10**12
        and r['partition']in('train2014','val2014') and r['file_name']==f"COCO_{r['partition']}_{r['image_id']:012d}.jpg"
        and r['original_url']==prep.BUCKET+r['partition']+'/'+r['file_name'] and r['license_id']==4
        and r['license_url']==c.GRANT and all(type(r[n])is int and r[n]>0 for n in('width','height'))
        and r['width']*r['height']<=RECIPE['max_decoded_pixels']for r in rows),'Fixed48 original publisher slots before requests')
    with ThreadPoolExecutor(max_workers=4)as pool:records=list(pool.map(slot,rows))
    expired=time.monotonic()>=deadline;images=[];mappings=[];md5s=[]
    for row,record in zip(rows,records):
        if record['status']!='acquired':continue
        name=f"image_{row['slot']:06d}.jpg";opaque=hashlib.sha256((context.selection.NAMESPACE+f"{row['image_id']:012d}").encode()).hexdigest()[:32]
        images.append(dict(image_id=opaque,file=name,**record['image_pin'],width=row['width'],height=row['height']))
        mappings.append(dict(slot=row['slot'],public_image_id=opaque,public_file=name,image_pin=record['image_pin']));md5s.append(record['original_md5'])
    aliases=len(md5s)!=len(set(md5s));gate=len(images)==48 and not aliases and not expired
    rt.require(all(set(r)==prep.PUBLIC_KEYS for r in images)and sum(r['bytes']for r in images)<=RECIPE['max_total_image_bytes'],'Opaque six-key byte boundary required')
    pp=prep.write(public/'manifest.json',dict(schema='world_reward.rgb_proposal_inputs.v1',images=images),ledger)
    return dict(records=records,acquisition_records_complete=True,public_mappings=mappings,public_inputs_identity=pp,duplicate_new_bytes=aliases,http_window_expired=expired,
        counts=dict(slots=48,acquired=len(images),missing=48-len(images),acquired_FIT=sum(r['status']=='acquired'and r['split']=='FIT'for r in records),
                    acquired_CAL=sum(r['status']=='acquired'and r['split']=='CAL'for r in records)),
        availability_gate_passed=gate,decision='READY_FULL48_PENDING_BLIND_BANKS' if gate else'CLOSED_FULL48_ACQUISITION_NO_SUBSET_OR_RETRY')


def publish(public,private,report,deadline,started,owned_public,owned_private,expected_public,expected_private,parent):
    before_public=prep.inventory(public,owned_public,expected_public,0o700,16 << 20)
    before_private=prep.inventory(private,owned_private,expected_private,0o700,1 << 20)
    rt.require(prep.namespace_identity(DATA)==parent,'Root-private original parent required')
    with(private/'report.json').open('x+b')as stream:
        os.fchmod(stream.fileno(),0o400);saved=os.fstat(stream.fileno())
        def update():
            st=(private/'report.json').lstat();rt.require((st.st_dev,st.st_ino)==(saved.st_dev,saved.st_ino)and st.st_nlink==1,'Foreign report inode rejected')
            stream.seek(0);stream.write(c.encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            rt.require({p.name for p in private.iterdir()}==set(expected_private)|{'report.json'},'Foreign private leaf')
            public.chmod(0o500);private.chmod(0o500);prep.acq.sync_directory(public);prep.acq.sync_directory(private);prep.acq.sync_directory(DATA)
            report.update(outputs_sealed=True,public_outputs_sealed=True,elapsed_seconds=time.monotonic()-started);update()
            after_public=prep.inventory(public,owned_public,expected_public,0o500,16 << 20)
            after_private=prep.inventory(private,owned_private,{**expected_private,'report.json':c.pin(c.encode(report))},0o500,1 << 20)
            rt.require(before_public==after_public and all(after_private[n]==before_private[n]for n in expected_private)
                and prep.namespace_identity(DATA)==parent,'Owned artifacts/parent changed');c.check(deadline)
        except BaseException:
            report.update(status='fail',decision='CLOSED_ACQUISITION_PUBLICATION',publication_failed=True,elapsed_seconds=time.monotonic()-started);update()


def run(started,deadline):
    rt.require(os.geteuid()==0 and sys.platform=='linux'and os.uname().nodename=='world-reward-ncc-h100-02','Exact Azure CPU host required')
    code,revision=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'];rt.require(Path(__file__).resolve()==code/HELPERS[0],'Immutable caller origin required')
    cp=lambda:c.check(deadline);before=authenticate(code,revision,cp);cp();parent=prep.namespace_identity(DATA)
    public,private=DATA/'inputs',DATA/'acquisition';rt.require(not public.exists()and not private.exists(),'Fresh acquisition only; no retry')
    public.mkdir(mode=0o700);private.mkdir(mode=0o700);owned_public,owned_private=public.lstat(),private.lstat();public_ledger={};private_ledger={}
    report=dict(**RECIPE,status='fail',stage='catalog_metadata',producer_revision=revision,input_proof=before[3],
        freeze_report_identity=context.REPORT_PIN,cohort_identity=context.COHORT_PIN,historical_photos=448,
        source_and_inputs_rehashed_after=False,outputs_sealed=False,public_outputs_sealed=False,
        annotation_values_consulted=False,role_values_consulted=False,pilot_reference_values_read=False,
        RGB_decoded=False,network_used=False,GPU_used=False,models_loaded=False,FIT_performed=False,CAL_evaluated=False,selection_performed=False,
        author_disjointness_verified=False,training_overlap_verified=False,challenge_overlap_verified=False,adopted=False,
        records=[dict(slot=r['slot'],split=r['split'],image_id=r['image_id'],status='unavailable',failed_phase='not_requested')for r in before[2]['records']],
        counts=dict(slots=48,acquired=0,missing=48,acquired_FIT=0,acquired_CAL=0),availability_gate_passed=False,acquisition_records_complete=False)
    try:
        rows,expanded=catalog(before[0],before[2],cp);report['catalog_expanded_identity']=expanded
        report['publisher_metadata']=rows;cp();report['network_used']=True
        result=acquire(rows,before[1],public,min(deadline,time.monotonic()+300),public_ledger)
        report.update(result,stage='complete',status='pass' if result['availability_gate_passed']else'fail')
    except BaseException as exc:report.update(status='fail',decision='CLOSED_ACQUISITION_NO_RETRY',error_type=type(exc).__name__ if type(exc).__name__ in('ValueError','TimeoutError','OSError','KeyError','TypeError','BadZipFile')else'other')
    finally:
        try:
            after=authenticate(code,revision,cp);rt.require(after[:3]==before[:3]and context.normalized(after[3])==context.normalized(before[3]),'Full source/input bytes or modes changed')
            rt.require(prep.namespace_identity(DATA)==parent,'Root-private parent changed');report['source_and_inputs_rehashed_after']=True
            report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rt.require(report['peak_rss_bytes']<=8 << 30,'Host memory bound');cp()
        except BaseException:report.update(status='fail',decision='CLOSED_ACQUISITION_POSTHASH',posthash_failed=True)
        report['public_artifact_identities']=dict(public_ledger);report['artifact_identities']=dict(private_ledger)
        publish(public,private,report,deadline,started,owned_public,owned_private,public_ledger,private_ledger,parent)
    print(c.encode(dict(status=report['status'],decision=report['decision'],report_identity=rt.identity(private/'report.json',1 << 20))).decode(),end='')
    return report


def main():
    started=time.monotonic();deadline=started+1200
    def interrupted(*_):raise TimeoutError('Frozen acquisition interrupted')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,1200);resource.setrlimit(resource.RLIMIT_AS,(8 << 30,8 << 30))
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    try:return run(started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    rt.require(len(sys.argv)==1,'One frozen acquisition only')
    sys.exit(0 if main()['status']=='pass'else 1)

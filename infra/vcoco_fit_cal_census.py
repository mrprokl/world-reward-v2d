"""First explicit TRAIN/VAL count-only caller: private identities, no selection."""
import base64
import hashlib
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_role_prepare as prep
import vcoco_population_census as population

c,rt=prep.c,prep.rt
ROOT=c.ROOT
OUTPUT=Path('/srv/world-reward-data/vcoco_fit_cal_census_v1')
ENTRY='run_vcoco_fit_cal_census'
HELPERS=tuple(dict.fromkeys(('infra/vcoco_fit_cal_census.py','infra/run_vcoco_fit_cal_census.sh',
    'infra/vcoco_population_census.py',*prep.HELPERS,*c.HELPERS)))
PREPARE=dict(revision='a8da2da838dc4ef513e7819be4d25ece99488207',files=303,entries=308,
    closure_sha256='098811fa1eac670cfa081b8d5d10dcdc3b405ed4b9cc67356345e8de42b4ae0a',
    archive_xz_sha256='bf2a89a90781d306253cb27558c2c2d2fac127b442e44984a067d6a03f8acea4')
PINS=dict(selection=dict(bytes=41759,sha256='d2908db8bc0488bcec7836e938b49fea942214ab729afaffde7296f00baab3d6'),
    cohort=dict(bytes=12849,sha256='2b5048e6816abcbf45dffe7d1c42f5aa9fc22adfd411c8703a5c6f917661970c'),
    acquisition=dict(bytes=58443,sha256='264b9e3eec7a1cde816606886391f795e4dd9f63074216e291f2ac70b85e6177'),
    public=dict(bytes=3293,sha256='b938a7d22737f4399fc14cc0683863e920efaecef5ec544f115180119e4f1680'))
RECIPE=dict(schema='world_reward.vcoco_fit_cal_census.v1',output=str(OUTPUT),budget_seconds=1200,outer_seconds=1215,
    max_host_memory_bytes=8 << 30,cpu_threads=4,roles={n:f'data__vcoco__vcoco_{n}.json'for n in ('train','val')},
    minimum_photos=dict(train=32,val=16),population_source=dict(bytes=8177,
        sha256='8f8433e3b45da06af1912d75a7dd05e7d208c31e1436dbe6c9791fd87d857155'))


def normalized(value):return rt.strict(c.encode(value))


def original(code,revision,entry,helpers,spec):
    old=ROOT/'jobs'/revision/entry/'code';proof=c.source(old,revision,entry,helpers)
    rt.require(proof['binding']['entries']==spec['entries']and proof['binding']['closure_sha256']==spec['closure_sha256']
        and sum(p.is_file()for p in old.rglob('*'))==spec['files']
        and (old.parent/'source-sha256').read_bytes()==(spec['archive_xz_sha256']+'\n').encode(),'Whole original source differs')
    for name in helpers:rt.require(rt.identity(code/name,2 << 20,empty=True)==proof['binding']['helpers'][name],
        'Every reused source/config byte must equal original')
    return old,proof


def pilot(excluded,proof,checkpoint):
    """All16 identities/bytes only; reference payloads are never decoded."""
    rt.require(prep.namespace_identity(prep.DATA)[-2:]==[0,0],'Original private pilot parent required')
    rows,reference_states=prep.frozen(prep.DATA/'metadata',PINS['selection'],PINS['cohort'],proof,checkpoint)
    values={n:rt.pinned(prep.DATA/path,PINS[n],1 << 20)for n,path in
        (('selection','metadata/report.json'),('acquisition','eval_private/report.json'),('public','inputs/manifest.json'))}
    selected,acquired,public=(values[n]for n in ('selection','acquisition','public'))
    rt.require(normalized(selected['input_proof'])==normalized(proof)and selected['producer_revision']==PREPARE['revision']
        and normalized(acquired['input_proof'])==normalized(proof)and acquired['source_binding']==proof['source_binding']
        and acquired['producer_revision']==PREPARE['revision']and acquired['phase']=='acquire'and acquired['status']=='pass'
        and acquired['decision']=='READY_PENDING_SEPARATE_RGB_BANK'and acquired['selected_slots']==16
        and acquired['selection_report_identity']==PINS['selection']and acquired['cohort_identity']==PINS['cohort']
        and acquired['public_inputs_identity']==PINS['public']and acquired['availability_gate_passed']is True
        and all(acquired[k]is True for k in ('source_and_inputs_rehashed_after','outputs_sealed','public_outputs_sealed'))
        and all(acquired[k]is False for k in ('RGB_decoded','GPU_used','models_loaded','FIT_performed','predictions_read',
            'selection_performed','fresh_reference_values_consulted')),'Original acquisition lineage/flags differ')
    rt.require(set(public)=={'schema','images'}and public['schema']=='world_reward.rgb_proposal_inputs.v1'
        and len(public['images'])==len(acquired['records'])==len(acquired['public_mappings'])==16,'All16 original acquired slots required')
    photos=set(excluded['photos']);digests=set(excluded['md5']);pilot_digests=set();frozen={}
    for row,status,mapping,image in zip(rows,acquired['records'],acquired['public_mappings'],public['images']):
        checkpoint();slot=row['slot'];photo=c.coco.photo_identity(row['image']['flickr_url'])
        rt.require(photo==row['image']['photo_id']and photo not in photos,'Historical/pilot photo alias rejected');photos.add(photo)
        rt.require(status['slot']==slot and status['split']==row['split']and status['image_id']==row['image_id']
            and status['status']=='acquired','Every original acquisition row mapping required')
        md5=c.identities.md5_identity(status['original_md5']);rt.require(md5 not in digests,'Historical/pilot byte alias rejected');digests.add(md5);pilot_digests.add(md5)
        opaque=hashlib.sha256((prep.NAMESPACE+f"{row['image_id']:012d}").encode()).hexdigest()[:32];name=f'image_{slot:06d}.jpg'
        rt.require(mapping==dict(slot=slot,public_image_id=opaque,public_file=name,image_pin=status['image_pin'])
            and image==dict(image_id=opaque,file=name,**status['image_pin'],width=row['image']['width'],height=row['image']['height'])
            and acquired['public_artifact_identities'][name]==status['image_pin'],'Original opaque six-key mapping differs')
        path=prep.DATA/'inputs'/name;rt.require(rt.identity(path,16 << 20)==status['image_pin'],'Original JPEG SHA differs')
        digest=hashlib.md5()
        with path.open('rb')as stream:
            for block in iter(lambda:stream.read(1 << 20),b''):checkpoint();digest.update(block)
        rt.require(digest.digest()==base64.b64decode(md5),'Original JPEG MD5 differs')
        frozen[str(path)]=dict(pin=status['image_pin'],state=c.state(path))
    rt.require(len(photos)==448 and len(pilot_digests)==16 and acquired['counts']==dict(slots=16,acquired=16,missing=0,acquired_DEV=8,acquired_RESERVED=8),
        'Exact448 photos/all16 original bytes required')
    for directory,names in ((prep.DATA/'metadata',{'report.json','cohort.json'}|{f'reference_{i:06d}.json'for i in range(16)}),
                            (prep.DATA/'inputs',{'manifest.json'}|{f'image_{i:06d}.jpg'for i in range(16)}),
                            (prep.DATA/'eval_private',{'report.json'})):
        s=directory.lstat();rt.require(stat.S_IMODE(s.st_mode)==0o500 and s.st_uid==s.st_gid==0
            and {p.name for p in directory.iterdir()}==names,'Original sealed pilot namespace differs');frozen[str(directory)]=c.state(directory)
        for path in directory.iterdir():
            s=path.lstat();rt.require(stat.S_IMODE(s.st_mode)==0o400 and s.st_uid==s.st_gid==0,'Original root-owned400 pilot leaf required')
            frozen[str(path)]=dict(pin=rt.identity(path,16 << 20),state=c.state(path))
    rt.require(acquired['public_artifact_identities']['manifest.json']==PINS['public'],'Original public artifact ledger differs')
    return dict(photos=photos,authors=set(excluded['authors']),md5=digests),dict(references=reference_states,
        pilot_states=frozen,namespace=prep.namespace_identity(prep.DATA),pins=PINS)


def authenticate(code,revision,checkpoint):
    current=c.source(code,revision,ENTRY,HELPERS);source=current['binding']
    rt.require(source['helpers']['infra/vcoco_population_census.py']==RECIPE['population_source'],'Frozen population helper required')
    for module,name in ((prep,'infra/vcoco_role_prepare.py'),(population,'infra/vcoco_population_census.py'),(c,'infra/vcoco_role_census.py'),
        (rt,'infra/mediapipe_cpu_runtime_verify.py'),(c.js,'infra/metadata_json_stream.py'),(c.projection,'infra/vcoco_role_stream.py'),
        (c.identities,'infra/openimages_joint_pair_acquire.py'),(c.coco,'infra/coco_proposal_prepare.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual imported source origin differs')
    rt.require(Path(sys.modules[c.parse_vcoco_role_reference.__module__].__file__).resolve()==code/'src/world_reward/vcoco_role_reference.py','Actual reference origin differs')
    old,p=original(code,PREPARE['revision'],prep.ENTRY,prep.HELPERS,PREPARE);cfg=prep.configuration(code,p['binding']);spec=cfg['census']
    old_code,_=original(code,spec['producer_revision'],c.ENTRY,c.HELPERS,spec)
    old_cfg=c.configuration(rt.pinned(Path(spec['configuration']['path']),spec['configuration']['pin'],65536))
    rt.require(Path(spec['configuration']['path'])==old_code/c.CONFIG,'Original census configuration namespace differs')
    before=c.authenticate(old_cfg,old_code,spec['producer_revision']);checkpoint()
    report=rt.pinned(Path(spec['report']['path']),spec['report']['pin'],300000)
    s=Path(spec['report']['path']).lstat();rt.require(stat.S_IMODE(s.st_mode)==0o400 and s.st_uid==s.st_gid==0,'Original sealed census report required')
    rt.require(report['status']=='pass'and report['stage']=='complete'and report['capacity_gate_passed']is True
        and normalized(report['source_binding'])==normalized(before)and report['configuration_identity']==spec['configuration']['pin']
        and report['outputs_sealed']is report['source_and_inputs_rehashed_after']is True
        and report['historical_slots']==432 and report['eligibility_inventory_rows']==561,'Original complete census receipt differs')
    rt.require(spec['report']['path']==str(c.DATA/'report.json')and stat.S_IMODE(c.DATA.lstat().st_mode)==0o500
        and c.DATA.lstat().st_uid==c.DATA.lstat().st_gid==0 and {p.name for p in c.DATA.iterdir()}=={'report.json'},'Original census namespace differs')
    proof=dict(source_binding=p['binding'],census_binding=before,source_stat_identity=p['stat_identity'],
        census_report_identity=spec['report']['pin'],census_report_state=c.state(Path(spec['report']['path'])),census_directory_state=c.state(c.DATA))
    values={n:rt.strict(Path(old_cfg['inputs'][n]['path']).read_bytes())for n in (*old_cfg['history_names'],
        'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
    excluded=c.exclusions([values[n]for n in old_cfg['history_names']],values['ownership_cohort'],values['ownership_acquired'],
        [values['coco32_acquired'],values['coco64_acquired']])
    excluded,pilot_proof=pilot(excluded,proof,checkpoint)
    return old_cfg,excluded,dict(current=current,original_prepare=p,original_census=before,prepare_proof=proof,pilot=pilot_proof)


def run(started,deadline):
    rt.require(os.geteuid()==0 and sys.platform=='linux'and os.uname().nodename=='world-reward-ncc-h100-02','Exact Azure CPU host required')
    code,revision=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve()==code/HELPERS[0],'Immutable caller origin required')
    cp=lambda:c.check(deadline);before=authenticate(code,revision,cp);cp()
    rt.require(not rt.canonical(OUTPUT).exists()and OUTPUT.parent.is_dir(),'Fresh count-only namespace; no retry')
    OUTPUT.mkdir(mode=0o700);OUTPUT.chmod(0o700);prep.acq.sync_directory(OUTPUT.parent);owned=OUTPUT.lstat();expected={}
    report=dict(**{k:v for k,v in RECIPE.items()if k not in ('roles','population_source')},status='fail',stage='fresh_metadata_census',
        producer_revision=revision,input_proof=before[2],recipe_identity=c.pin(c.encode(RECIPE)),historical_photos=448,
        historical_reference_values_read=False,pilot_reference_values_read=False,TEST_role_values_read=False,
        fresh_reference_geometry_consulted=False,RGB_read=False,network_used=False,GPU_used=False,models_loaded=False,
        FIT_performed=False,selection_performed=False,author_disjointness_verified=False,challenge_overlap_verified=False,
        training_overlap_verified=False,adopted=False,source_and_inputs_rehashed_after=False,outputs_sealed=False)
    try:
        report['fresh_reference_geometry_consulted']=True
        result,rows=population.census_population(before[0],before[1],roles=RECIPE['roles'],minimum_photos=RECIPE['minimum_photos'],checkpoint=cp)
        rt.require(all(set(r)=={'split','image_id','photo_id'}for r in rows)and c.pin(c.encode(rows))==result['eligibility_inventory_identity'],
            'Identity-only full census inventory required')
        prep.raw_write(OUTPUT/'inventory.json',c.encode(rows),expected)
        report.update(result,stage='complete',status='pass'if result['capacity_gate_passed']else'fail',
            decision='CAPACITY_METADATA_ONLY_NO_SELECTION'if result['capacity_gate_passed']else'CLOSED_INSUFFICIENT_CAPACITY_NO_RGB')
    except BaseException as exc:report.update(status='fail',decision='CLOSED_CENSUS_NO_RETRY',error_type=type(exc).__name__ if type(exc).__name__ in ('ValueError','TimeoutError','OSError','KeyError','BadZipFile')else'other')
    finally:
        try:
            after=authenticate(code,revision,cp)
            rt.require(after[:2]==before[:2]and normalized(after[2])==normalized(before[2]),'Full original source/input bytes or modes changed')
            report['source_and_inputs_rehashed_after']=True;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            rt.require(report['peak_rss_bytes']<=RECIPE['max_host_memory_bytes'],'Host memory budget exceeded');cp()
        except BaseException:report.update(status='fail',decision='CLOSED_CENSUS_POSTHASH',posthash_failed=True)
        report['artifact_identities']=dict(expected);prep.publish(OUTPUT,report,deadline,started,owned,expected)
    print(c.encode(dict(status=report['status'],decision=report['decision'],report_identity=rt.identity(OUTPUT/'report.json',1 << 20))).decode(),end='')
    return report


def main():
    started=time.monotonic();deadline=started+RECIPE['budget_seconds']
    def interrupted(*_):raise TimeoutError('Fixed census interrupted')
    handlers={s:signal.signal(s,interrupted)for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,RECIPE['budget_seconds']);resource.setrlimit(resource.RLIMIT_AS,(8 << 30,8 << 30))
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    try:return run(started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    rt.require(len(sys.argv)==1,'Single frozen count-only census')
    sys.exit(0 if main()['status']=='pass'else 1)

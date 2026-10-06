"""Fresh private FIT32/CAL16 identity publication; no roles/RGB/acquisition."""
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_fit_cal_census as f
import vcoco_fit_cal_selection as selection

c, rt, prep = f.c, f.rt, f.prep
ROOT = f.ROOT
DATA = Path('/srv/world-reward-data/vcoco_fit_cal_v1')
ENTRY = 'run_vcoco_fit_cal_freeze'
HELPERS = tuple(dict.fromkeys(('infra/vcoco_fit_cal_freeze.py', 'infra/run_vcoco_fit_cal_freeze.sh',
    'infra/vcoco_fit_cal_selection.py', *f.HELPERS)))
OLD = dict(revision=selection.CENSUS_REVISION, files=308, entries=313,
    closure_sha256=selection.CENSUS_CLOSURE,
    archive_xz_sha256='12ae45bb1895da7dc0b312f7e2ec9ed226688233b2eac8a44deb1e1a7f770503')
FROZEN = {'infra/vcoco_fit_cal_census.py': dict(bytes=14765,
    sha256='d22dfbdcd034df29c4a6d405510c0ea758d83126ad529324f99ba65e5265886b'),
    'infra/vcoco_fit_cal_selection.py': dict(bytes=7780,
    sha256='a1300cabfa410c2d1b74723cb073a59664f6f0462d68260db890ce91cb55ab1e')}
RECIPE = dict(schema='world_reward.vcoco_fit_cal_freeze.v1', output=str(DATA), budget_seconds=1200,
    outer_seconds=1215, max_host_memory_bytes=8 << 30, cpu_threads=4, selected_slots=48,
    FIT_slots=32, CAL_slots=16, namespace=selection.NAMESPACE, all_48_acquired_required=True,
    retry_count=0, replacement_count=0)


def normalized(value): return rt.strict(c.encode(value))


def authenticate(code, revision, checkpoint):
    """Run current byte-identical helpers on live old artifacts, never old run()."""
    current = c.source(code, revision, ENTRY, HELPERS)
    rt.require(all(current['binding']['helpers'][n] == p for n,p in FROZEN.items()), 'Exact frozen helpers required')
    for module,name in ((f,'infra/vcoco_fit_cal_census.py'), (selection,'infra/vcoco_fit_cal_selection.py'),
        (prep,'infra/vcoco_role_prepare.py'), (c,'infra/vcoco_role_census.py'),
        (rt,'infra/mediapipe_cpu_runtime_verify.py'), (c.js,'infra/metadata_json_stream.py'),
        (c.projection,'infra/vcoco_role_stream.py'), (c.identities,'infra/openimages_joint_pair_acquire.py'),
        (c.coco,'infra/coco_proposal_prepare.py'), (f.population,'infra/vcoco_population_census.py')):
        rt.require(Path(module.__file__).resolve() == code/name, 'Actual current helper origin differs')
    rt.require(Path(sys.modules[c.parse_vcoco_role_reference.__module__].__file__).resolve() ==
               code/'src/world_reward/vcoco_role_reference.py', 'Actual pure parser origin differs')
    old_code,old = f.original(code, OLD['revision'], f.ENTRY, f.HELPERS, OLD)
    prepare_code,prepare = f.original(code, f.PREPARE['revision'], prep.ENTRY, prep.HELPERS, f.PREPARE)
    cfg = prep.configuration(code, prepare['binding']); spec = cfg['census']
    census_code,_ = f.original(code, spec['producer_revision'], c.ENTRY, c.HELPERS, spec)
    old_cfg = c.configuration(rt.pinned(Path(spec['configuration']['path']), spec['configuration']['pin'],65536))
    rt.require(Path(spec['configuration']['path']) == census_code/c.CONFIG, 'Original configuration path differs')
    original_census = c.authenticate(old_cfg,census_code,spec['producer_revision']); checkpoint()
    prior = rt.pinned(Path(spec['report']['path']),spec['report']['pin'],300000)
    rt.require(Path(spec['report']['path']) == c.DATA/'report.json' and prior['status'] == 'pass'
        and prior['stage'] == 'complete' and prior['capacity_gate_passed'] is True
        and normalized(prior['source_binding']) == normalized(original_census)
        and prior['configuration_identity'] == spec['configuration']['pin']
        and prior['outputs_sealed'] is prior['source_and_inputs_rehashed_after'] is True
        and prior['historical_slots'] == 432 and prior['eligibility_inventory_rows'] == 561,
        'Original sealed census lineage differs')
    for directory,names in ((c.DATA,{'report.json'}),(f.OUTPUT,{'report.json','inventory.json'})):
        st=rt.canonical(directory).lstat()
        rt.require(stat.S_IMODE(st.st_mode) == 0o500 and st.st_uid == st.st_gid == 0
                   and {p.name for p in directory.iterdir()} == names,'Original sealed namespace differs')
        for path in directory.iterdir():
            st=path.lstat();rt.require(stat.S_IMODE(st.st_mode) == 0o400 and st.st_uid == st.st_gid == 0,
                                     'Original root-owned400 leaf required')
    proof = dict(source_binding=prepare['binding'],census_binding=original_census,
        source_stat_identity=prepare['stat_identity'],census_report_identity=spec['report']['pin'],
        census_report_state=c.state(Path(spec['report']['path'])),census_directory_state=c.state(c.DATA))
    values = {n:rt.strict(Path(old_cfg['inputs'][n]['path']).read_bytes()) for n in
        (*old_cfg['history_names'],'ownership_cohort','ownership_acquired','coco32_acquired','coco64_acquired')}
    excluded = c.exclusions([values[n] for n in old_cfg['history_names']], values['ownership_cohort'],
        values['ownership_acquired'],[values['coco32_acquired'],values['coco64_acquired']])
    excluded,pilot = f.pilot(excluded,proof,checkpoint)
    original_proof = dict(current=old, original_prepare=prepare,original_census=original_census,
                         prepare_proof=proof,pilot=pilot)
    report = rt.pinned(f.OUTPUT/'report.json',selection.REPORT_PIN,1 << 20)
    rt.require(rt.identity(f.OUTPUT/'inventory.json',1 << 20) == selection.INVENTORY_PIN
               and normalized(original_proof) == report['input_proof'],'Original complete census proof differs')
    # Include marker/parent modes and inode states, not just their byte identity.
    source_codes = [code,old_code,prepare_code,census_code] + [ROOT/'jobs'/v['revision']/v['entry']/'code'
                                                            for v in old_cfg['sources'].values()]
    states = {str(p):c.state(p) for d in source_codes for p in (d.parent,d.parent/'revision',d.parent/'source-sha256')}
    states.update({str(f.OUTPUT):c.state(f.OUTPUT),str(f.OUTPUT/'report.json'):c.state(f.OUTPUT/'report.json'),
                   str(f.OUTPUT/'inventory.json'):c.state(f.OUTPUT/'inventory.json')})
    return excluded,dict(current_source=current,original_input_proof=original_proof,live_states=states,
                         census_report_identity=selection.REPORT_PIN,census_inventory_identity=selection.INVENTORY_PIN)


def publish(out,report,deadline,started,owned,expected,parent):
    """Original inventory checks plus a fixed original report FD and parent."""
    before=prep.inventory(out,owned,expected,0o700,1 << 20)
    rt.require(prep.namespace_identity(DATA) == parent,'Private original parent required')
    with (out/'report.json').open('x+b') as stream:
        os.fchmod(stream.fileno(),0o400);saved=os.fstat(stream.fileno())
        def update():
            now=(out/'report.json').lstat()
            rt.require((now.st_dev,now.st_ino)==(saved.st_dev,saved.st_ino) and now.st_nlink==1,
                       'Foreign report inode rejected')
            stream.seek(0);stream.write(c.encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            rt.require({p.name for p in out.iterdir()} == set(expected)|{'report.json'},'Foreign publication leaf')
            out.chmod(0o500);prep.acq.sync_directory(out);prep.acq.sync_directory(DATA)
            report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started);update()
            after=prep.inventory(out,owned,{**expected,'report.json':c.pin(c.encode(report))},0o500,1 << 20)
            rt.require(all(after[n]==before[n] for n in expected) and prep.namespace_identity(DATA)==parent,
                       'Owned artifacts or private parent changed');c.check(deadline)
        except BaseException:
            report.update(status='fail',decision='CLOSED_IDENTITY_FREEZE_PUBLICATION',publication_failed=True,
                          elapsed_seconds=time.monotonic()-started);update()


def run(started,deadline):
    rt.require(os.geteuid() == 0 and sys.platform == 'linux' and os.uname().nodename == 'world-reward-ncc-h100-02',
               'Exact Azure VM02 CPU host required')
    code,revision = Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION']
    rt.require(Path(__file__).resolve() == code/HELPERS[0], 'Immutable caller origin required')
    checkpoint = lambda:c.check(deadline); before = authenticate(code,revision,checkpoint); checkpoint()
    rt.require(not rt.canonical(DATA).exists() and DATA.parent.is_dir(), 'Fresh metadata namespace; no retry')
    DATA.mkdir(mode=0o700);DATA.chmod(0o700);parent = prep.namespace_identity(DATA)
    out = DATA/'metadata';out.mkdir(mode=0o700);out.chmod(0o700);prep.acq.sync_directory(DATA);owned = out.lstat();expected={}
    report = dict(**RECIPE,status='fail',stage='identity_projection',producer_revision=revision,
        input_proof=before[1],recipe_identity=c.pin(c.encode(RECIPE)),historical_photos=448,
        current_helpers_live_verified=True,historical_execution_replayed=False,historical_reference_values_read=False,
        pilot_reference_values_read=False,TEST_role_values_read=False,reference_geometry_consulted=False,
        RGB_decoded=False,JPEG_bytes_hashed=True,network_used=False,GPU_used=False,models_loaded=False,
        FIT_performed=False,CAL_evaluated=False,selection_performed=False,source_and_inputs_rehashed_after=False,
        outputs_sealed=False,author_disjointness_verified=False,training_overlap_verified=False,
        challenge_overlap_verified=False,adopted=False)
    try:
        rt.require(rt.identity(f.OUTPUT/'report.json',1 << 20)==selection.REPORT_PIN
                   and rt.identity(f.OUTPUT/'inventory.json',1 << 20)==selection.INVENTORY_PIN,'Same bytes before identity decode')
        cohort=selection.freeze_fit_cal_cohort((f.OUTPUT/'report.json').read_bytes(),(f.OUTPUT/'inventory.json').read_bytes(),
            excluded_photo_ids=before[0]['photos'],expected_input_proof=before[1]['original_input_proof'])
        checkpoint();identity=prep.write(out/'cohort.json',cohort,expected)
        report.update(status='pass',stage='complete',decision='FROZEN48_IDENTITIES_PENDING_SEPARATE_ACQUISITION',
                      cohort_identity=identity,selection_performed=True,split_counts=dict(FIT=32,CAL=16))
    except BaseException as exc:
        report.update(status='fail',decision='CLOSED_IDENTITY_FREEZE_NO_RETRY',error_type=type(exc).__name__ if
                      type(exc).__name__ in ('ValueError','TimeoutError','OSError','KeyError','TypeError') else 'other')
    finally:
        try:
            after=authenticate(code,revision,checkpoint)
            rt.require(after[0] == before[0] and normalized(after[1]) == normalized(before[1]),'Full live original source/input states changed')
            rt.require(prep.namespace_identity(DATA) == parent,'Owned root-private parent changed')
            report['source_and_inputs_rehashed_after']=True
            report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            rt.require(report['peak_rss_bytes'] <= RECIPE['max_host_memory_bytes'],'Host memory bound exceeded');checkpoint()
        except BaseException: report.update(status='fail',decision='CLOSED_IDENTITY_FREEZE_POSTHASH',posthash_failed=True)
        report['artifact_identities']=dict(expected);publish(out,report,deadline,started,owned,expected,parent)
    print(c.encode(dict(status=report['status'],decision=report['decision'],report_identity=rt.identity(out/'report.json',1 << 20))).decode(),end='')
    return report


def main():
    started=time.monotonic();deadline=started+RECIPE['budget_seconds']
    def interrupted(*_):raise TimeoutError('Fixed freeze interrupted')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,RECIPE['budget_seconds']);resource.setrlimit(resource.RLIMIT_AS,(8 << 30,8 << 30))
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    try:return run(started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__ == '__main__':
    rt.require(len(sys.argv) == 1,'One frozen metadata phase; no acquisition')
    sys.exit(0 if main()['status'] == 'pass' else 1)

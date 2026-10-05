"""Azure CPU manufactured cost probe only; no real FIT/reference/model/RGB input."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_coherent_pair_cost_probe'
CONFIG = 'configs/coherent_pair_cost_probe_v1.json'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET = 720
HELPERS = ('infra/coherent_pair_cost_probe.py', 'infra/run_coherent_pair_cost_probe.sh', CONFIG,
    'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/__init__.py',
    'src/world_reward/coherent_pair_learning.py', 'src/world_reward/coherent_route_scorer.py',
    'src/world_reward/interaction_candidate_evidence.py', 'src/world_reward/interaction_tuple_evidence.py',
    'src/world_reward/person_pose_observations.py', 'src/world_reward/hoi_detr_observations.py')


def runtime(code):
    spec = importlib.util.spec_from_file_location('wr_cost_runtime', code/HELPERS[3])
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt); return rt


def configuration(rt, code, source):
    cfg = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16384)
    fixed = dict(schema='world_reward.coherent_pair_cost_probe.v1', budget_seconds=720,
        cpu_count=4, memory_bytes=6*1024**3, object_slots=3600,
        fixtures=[dict(persons=2, hands=2, direct_objects=2, native_pairs=4),
                  dict(persons=4, hands=8, direct_objects=8, native_pairs=64)],
        measurements=['zero_geometry', 'dyadic_geometry', 'dyadic_relational'],
        alpha=.25, fit_images_for_cost=32, evaluations_per_fit_image=1026,
        source_tokens=1500, output_prefix='results/coherent-pair-cost-probe-', image_id=IMAGE)
    rt.require(set(cfg) == set(fixed)|{'reused_helper_pins'} and all(type(cfg.get(k)) is type(v) and cfg[k] == v for k,v in fixed.items()), 'Exact fixed procedural cost scope required')
    rt.require(set(cfg['reused_helper_pins']) == set(HELPERS[3:])
        and all(source['helpers'].get(k) == pin for k,pin in cfg['reused_helper_pins'].items()), 'Frozen numerical/helper bytes differ')
    return cfg


def image(argv):
    r = subprocess.run(argv, capture_output=True, timeout=10, check=False)
    if r.returncode or len(r.stdout) > 16384 or len(r.stderr) > 8192: raise ValueError('Exact CPU image metadata unavailable')
    return json.loads(r.stdout)


def proof(rt, code, revision):
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); configuration(rt, code, source)
    rt.require({p.name for p in code.parent.iterdir()} == {'code','revision','source-sha256'}, 'Only current code and source markers allowed')
    rows = image(['docker','image','inspect',IMAGE]); rt.require(len(rows) == 1, 'One immutable CPU image required')
    row = rows[0]; rt.require(row['Id'] == IMAGE and row['Os'] == 'linux' and row['Architecture'] == 'amd64', 'Actual B47 CPU image required')
    return dict(source_binding=source, image={k:row[k] for k in ('Id','Os','Architecture','RootFS')})


def leaves(code):
    return [*(code/name for name in HELPERS), code.parent/'revision', code.parent/'source-sha256']


def fixtures(np, *, object_count=3600, source_records=None):
    """The optional small object count is unit-test-only, never a CLI/runtime choice."""
    from world_reward.person_pose_observations import PersonPoseObservations
    from world_reward.hoi_detr_observations import HOIDetrObservations
    from world_reward.interaction_candidate_evidence import GenericObjectObservations, build_interaction_candidate_evidence
    from world_reward.coherent_pair_learning import pair_route_bank
    from world_reward.coherent_route_scorer import _fingerprint
    if type(object_count) is not int or object_count < 1: raise ValueError('Positive manufactured object count required')
    for persons, nh, nd in ((2,2,2),(4,8,8)):
        xy = np.full((persons,133,2), 50., np.float64); scores = np.full((persons,133), .75, np.float32)
        pb = np.array([[i*200.+20.,30.,i*200.+190.,690.] for i in range(persons)], np.float64)
        for p in range(persons):
            for side,(body,hand) in enumerate(((9,91),(10,112))):
                xy[p,body] = [p*200.+60.+side*80.,370.]; xy[p,hand] = xy[p,body]+[4.,2.]
                if (p*2+side)%3 == 0: scores[p,body] = 0.
                if (p*2+side)%5 == 0: scores[p,hand] = 0.
        person = PersonPoseObservations(0,(720,960),tuple(f'person-{i}' for i in range(persons)),pb,np.full(persons,.875,np.float32),xy,scores)
        index = np.arange(object_count); ob = np.column_stack((index%60*15.+1.,index//60*11.+1.,index%60*15.+12.,index//60*11.+10.))
        objects = GenericObjectObservations(0,(720,960),tuple(f'object-{i}' for i in range(object_count)),ob,np.zeros(object_count,np.float32))
        hb = np.array([[i*31.+41.,360.,i*31.+55.,379.] for i in range(nh)],np.float32)
        db = ob[np.arange(nd)%object_count].astype(np.float32); boxes = np.vstack((hb,db)); count = len(boxes)
        classes = np.r_[np.zeros(nh,np.int64),np.ones(nd,np.int64)]
        pairs = np.column_stack((np.repeat(np.arange(nh),nd),np.tile(np.arange(nd)+nh,nh))).astype(np.int64)
        logits = np.column_stack((np.zeros(len(pairs)),(np.arange(len(pairs))%9-4)/8.)).astype(np.float32)
        score = np.full(count,.75,np.float32); ids = np.arange(count,dtype=np.int64)
        hoi = HOIDetrObservations(0,(720,960),np.zeros((1500,3),np.float32),np.zeros((1500,4),np.float32),np.zeros((1500,256),np.float32),
            np.column_stack((boxes,score)).astype(np.float32),ids,ids,ids,classes,boxes,score,score,pairs,logits,np.empty((0,2),np.int64),np.empty((0,2),np.float32))
        original = _fingerprint((person,objects,hoi))
        if source_records is not None:source_records.append(dict(full_original_observations_sha256=original,native_tokens=len(hoi.query_tokens),source_rehashed_after=False))
        try:yield pair_route_bank(person,build_interaction_candidate_evidence(person,objects,hoi))
        finally:
            if _fingerprint((person,objects,hoi)) !=original:raise ValueError('Original procedural source observations mutated')
            if source_records is not None:source_records[-1]['source_rehashed_after'] =True


def measure(np, check, progress, *, object_count=3600):
    from world_reward import coherent_pair_learning as learner
    from world_reward.coherent_route_scorer import _fingerprint
    sources = [];started = time.monotonic(); iterator = fixtures(np,object_count=object_count,source_records=sources)
    for case in range(2):
        progress.update(case=case,phase='prepare'); check(); before = time.monotonic(); bank = next(iterator)
        preparation = time.monotonic()-before; digest = _fingerprint(bank.evidence); e = bank.evidence
        n,o,k = len(e.source_person_ids),len(e.objects.object_ids),e.scope['hoi_native_pairs']
        check(); progress['fixtures'].append(dict(case=case,persons=n,objects=o,native_pairs=k,
            original_tuple_rows=n*2*k,original_bridge_rows=k*o,original_candidate_rows=n*2*o,
            exact_person_groups=len(bank.person_boxes),exact_object_groups=len(bank.object_boxes),
            evidence_sha256=digest,preparation_seconds=preparation,missing_feature_cells=int((~e.feature_supported).sum()),
            native_tokens=sources[-1]['native_tokens'],full_original_observations_sha256=sources[-1]['full_original_observations_sha256'],
            retained_person_ids=len(e.source_person_ids),retained_object_ids=len(e.objects.object_ids)))
        scale = learner.PairScale(np.ones(12),np.ones(12,bool)); mask = np.zeros((n,o),bool); mask[0,0] = True
        theta = np.array([(-1 if j%2 == 0 else 1)*(j+1)/32. for j in range(17)])
        for name,coeff,alpha in (('zero_geometry',np.zeros(17),None),('dyadic_geometry',theta,None),('dyadic_relational',theta,.25)):
            progress.update(phase=name); check(); t = time.monotonic()
            loss,gradient,counts = learner.loss_gradient(coeff,(bank,),(mask,),scale,alpha=alpha)
            duration = time.monotonic()-t; check()
            progress['measurements'].append(dict(case=case,name=name,seconds=duration,loss=loss,gradient=list(map(float,gradient)),counts=counts,
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)))
        if _fingerprint(e) != digest: raise ValueError('Original complete fixture mutated')
    if next(iterator,None) is not None or not all(r['source_rehashed_after'] for r in sources):raise ValueError('Complete source fixture census differs')
    times = [r['seconds'] for r in progress['measurements']]; estimate = min(times)*32*1026
    progress.update(phase='complete',measurements_completed=6,fixture_sources_rehashed_after=True,
        minimum_observed_image_pass_seconds=min(times),optimistic_min_pass_recipe_seconds=estimate,
        decision='COST_RECIPE_UNQUALIFIED_NO_REAL_FIT' if estimate >720 else 'PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION',
        full_fit_executed=False,measurement_total_seconds=time.monotonic()-started)


def publish(rt,path,value,deadline,*,seal_directory=False):
    """Owned FD survives sealing; crossing the deadline can never publish PASS."""
    raw = lambda: (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o444)
        stream.write(raw());stream.flush();os.fsync(stream.fileno())
        if seal_directory:path.parent.chmod(0o555)
        if time.monotonic() >= deadline: value.update(status='fail',error_type='TimeoutError')
        stream.seek(0);stream.write(raw());stream.truncate();stream.flush();os.fsync(stream.fileno())
        rt.require(rt.identity(path,1<<20) == dict(bytes=len(raw()),sha256=hashlib.sha256(raw()).hexdigest()),'Published cost receipt differs')
        if time.monotonic() >= deadline and value['status'] == 'pass':
            value.update(status='fail',error_type='TimeoutError');stream.seek(0);stream.write(raw());stream.truncate();stream.flush();os.fsync(stream.fileno())


def native(code,revision,out,proof_pin,deadline):
    rt = runtime(code)
    rt.require(Path(rt.__file__).resolve() ==code/HELPERS[3] and not (code/HELPERS[3]).stat().st_mode&0o222,'Original immutable runtime helper required')
    rt.require(rt.canonical(out) == ROOT/'results'/('coherent-pair-cost-probe-'+revision)
        and out.stat().st_uid ==1000 and stat.S_IMODE(out.stat().st_mode) ==0o755,'Exact owned native output required')
    rt.identity(out/'container.cid',65,readonly=False)
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',(out/'container.cid').read_bytes()),'Exact Docker CID required')
    previous = rt.pinned(out/'proof.json',proof_pin,1<<20)
    cfg = configuration(rt,code,previous['source_binding']); before = {str(p):rt.identity(p,2<<20) for p in leaves(code)}
    rt.require(sys.platform == 'linux' and os.geteuid() == 1000 and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
        and os.environ.get('WR_IMAGE_ID') == IMAGE and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}
        and previous['source_binding']['producer_revision'] == revision and code == ROOT/'jobs'/revision/ENTRY/'code'
        and {p.name for p in out.iterdir()} == {'proof.json','container.cid'}, 'Restricted CPU/native namespace required')
    rt.require(all(before[str(code/k)] == v for k,v in previous['source_binding']['helpers'].items()),'Narrow mounted executable source differs')
    rt.require((code.parent/'revision').read_bytes() ==(revision+'\n').encode()
        and all(before[str(code.parent/k)] ==v for k,v in previous['source_binding']['markers'].items()),'Original mounted source markers differ')
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Inclusive720s cost budget exhausted')
    def timeout(*_): raise TimeoutError('Cost probe cancelled or deadline reached')
    signal.signal(signal.SIGTERM,timeout);signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    report = dict(schema=cfg['schema'],stage='coherent_pair_cost_native',status='fail',phase='imports',producer_revision=revision,
        source_binding=previous['source_binding'],image_id=IMAGE,fixtures=[],measurements=[],gpu_used=False,models_loaded=False,
        rgb_read=False,external_references_read=False,challenge_inputs_used=False,quality_verified=False,adoption=False,source_rehashed_after=False)
    try:
        sys.path[:0] = [str(code/'src')];import numpy as np
        from world_reward import coherent_pair_learning,coherent_route_scorer
        for module,name in ((coherent_pair_learning,HELPERS[5]),(coherent_route_scorer,HELPERS[6])):
            rt.require(Path(module.__file__).resolve() ==code/name,'Numerical import origin differs')
        report['numpy_version'] = np.__version__; measure(np,check,report);check();report['status'] = 'pass'
    except BaseException as exc: report.update(status='fail',error_type=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            rt.require(before == {str(p):rt.identity(p,2<<20) for p in leaves(code)} and rt.identity(out/'proof.json',1<<20) == proof_pin,'Native source/proof changed')
            report['source_rehashed_after'] = True
        except Exception as exc: report.update(status='fail',post_error_type=type(exc).__name__)
        publish(rt,out/'native.json',report,deadline)
    return report


def absent(result,cid):
    return result.returncode == 1 and result.stdout.strip() in (b'',b'[]') and result.stderr.strip() in tuple(x.encode() for x in (
        f'Error: No such object: {cid}',f'error: no such object: {cid}',f'Error: No such container: {cid}',f'Error response from daemon: No such container: {cid}'))


def validate_native(rt,value,binding,revision):
    expected = dict(schema='world_reward.coherent_pair_cost_probe.v1',stage='coherent_pair_cost_native',status='pass',phase='complete',
        producer_revision=revision,source_binding=binding,image_id=IMAGE,measurements_completed=6,fixture_sources_rehashed_after=True,
        full_fit_executed=False,source_rehashed_after=True,gpu_used=False,models_loaded=False,rgb_read=False,
        external_references_read=False,challenge_inputs_used=False,quality_verified=False,adoption=False)
    rt.require(all(type(value.get(k)) is type(v) and value[k] ==v for k,v in expected.items()),'Exact complete native cost scope required')
    rt.require(len(value['fixtures']) ==2 and [(r['case'],r['persons'],r['objects'],r['native_pairs']) for r in value['fixtures']] ==[(0,2,3600,4),(1,4,3600,64)],'All fixed native fixtures required')
    for r in value['fixtures']:
        rt.require(r['original_candidate_rows'] ==r['persons']*2*3600 and r['original_tuple_rows'] ==r['persons']*2*r['native_pairs']
            and r['original_bridge_rows'] ==r['native_pairs']*3600 and r['native_tokens'] ==1500
            and r['retained_person_ids'] ==r['persons'] and r['retained_object_ids'] ==3600
            and r['exact_person_groups'] ==r['persons'] and r['exact_object_groups'] ==3600,'No source slots may be truncated')
    rows = value['measurements'];rt.require([(r['case'],r['name']) for r in rows] ==[(i,n) for i in range(2) for n in ('zero_geometry','dyadic_geometry','dyadic_relational')],'Every fixed cost measurement required')
    rt.require(all(type(r['seconds']) in (int,float) and math.isfinite(r['seconds']) and r['seconds'] >0
        and r['counts'] ==dict(fit_records=1,used=1,missing_positive=0,no_alternative=0) for r in rows),'Finite actual full-bank measured passes required')
    estimate = min(r['seconds'] for r in rows)*32*1026
    rt.require(value['optimistic_min_pass_recipe_seconds'] ==estimate and value['decision'] ==('COST_RECIPE_UNQUALIFIED_NO_REAL_FIT' if estimate >720 else 'PENDING_ACTUAL_FULL_FIT_COST_QUALIFICATION'),'Exact descriptive cost decision required')


def host(code,revision):
    started = time.monotonic();deadline = started+BUDGET
    def cancelled(*_): raise TimeoutError('Inclusive cost host budget/cancellation')
    signal.signal(signal.SIGALRM,cancelled);signal.signal(signal.SIGTERM,cancelled);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    rt = runtime(code);before = proof(rt,code,revision)
    out = rt.canonical(ROOT/'results'/('coherent-pair-cost-probe-'+revision));rt.require(not out.exists(),'Fresh cost namespace required')
    out.mkdir(mode=0o755);os.chown(out,1000,1000);out.chmod(0o755);owner = (out.stat().st_dev,out.stat().st_ino)
    raw = (json.dumps(before,sort_keys=True)+'\n').encode();rt.write(out/'proof.json',raw,0o444);proof_pin = rt.identity(out/'proof.json',1<<20)
    cidfile = out/'container.cid';name = 'world-reward-cost-'+revision[:12];removed = False
    report = dict(schema='world_reward.coherent_pair_cost_probe.v1',stage='coherent_pair_cost_host',status='fail',producer_revision=revision,source_binding=before['source_binding'])
    def cleanup():
        if not cidfile.exists(): return False
        rt.identity(cidfile,65,readonly=False);cid = cidfile.read_text().strip();rt.require(re.fullmatch('[0-9a-f]{64}',cid),'Exact owned CID required')
        def inspect(): return subprocess.run(['docker','inspect',cid,'--format','{{.Id}}|{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.cost.owner"}}'],capture_output=True,timeout=5)
        r = inspect()
        if not absent(r,cid):
            rt.require(r.returncode == 0 and r.stdout.decode().strip() == cid+'|'+IMAGE+'|/'+name+'|'+revision,'Never remove a foreign container')
            subprocess.run(['docker','rm','-f',cid],capture_output=True,timeout=10,check=True);rt.require(absent(inspect(),cid),'Owned CPU container survives')
        cidfile.chmod(0o444);return True
    try:
        argv = ['docker','run','--rm','--cidfile',str(cidfile),'--name',name,'--label','world_reward.cost.owner='+revision,
            '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','6g','--cpus','4','--pids-limit','256','--user','1000:1000',
            '--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=128m','--entrypoint','/usr/bin/env']
        for p in leaves(code):argv += ['--mount',f'type=bind,src={p},dst={p},readonly']
        argv += ['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','PYTHONDONTWRITEBYTECODE=1',
            'CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','WR_IMAGE_ID='+IMAGE,
            'python','-I','-B',str(code/HELPERS[0]),'native',str(code),revision,str(out),str(proof_pin['bytes']),proof_pin['sha256'],str(deadline)]
        r = subprocess.run(argv,capture_output=True,timeout=max(.001,deadline-time.monotonic()),check=False)
        report['native_exit_code'] = r.returncode;rt.require(len(r.stdout) <= 16384 and len(r.stderr) <=16384,'Bounded native diagnostic output required')
        removed = cleanup();native_report = rt.strict((out/'native.json').read_bytes())
        rt.require(r.returncode ==0,'Native cost process failed');validate_native(rt,native_report,before['source_binding'],revision)
        report.update(status='pass',native_report_identity=rt.identity(out/'native.json',1<<20),decision=native_report['decision'])
    except BaseException as exc: report.update(status='fail',error_type=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)  # Cleanup/FAIL receipt grace cannot convert a late run to PASS.
        try:removed = cleanup() or removed
        except Exception as exc:report.update(status='fail',cleanup_error_type=type(exc).__name__)
        try:report['source_rehashed_after'] = proof(rt,code,revision) == before
        except Exception as exc:report.update(status='fail',post_error_type=type(exc).__name__)
        rt.require((out.stat().st_dev,out.stat().st_ino) == owner and out.stat().st_uid ==1000,'Owned output replaced')
        rt.require({p.name for p in out.iterdir()} <= {'proof.json','container.cid','native.json'},'Foreign cost output node')
        report.update(owned_container_removed=removed,elapsed_seconds=time.monotonic()-started)
        if not removed or not report.get('source_rehashed_after') or time.monotonic() >=deadline:report['status'] = 'fail'
        publish(rt,out/'report.json',report,deadline,seal_directory=True)
    return report


def main():
    p = argparse.ArgumentParser(allow_abbrev=False);p.add_argument('mode',choices=('host','native'));p.add_argument('code');p.add_argument('revision');p.add_argument('native_args',nargs='*');a = p.parse_args()
    code = Path(a.code)
    if not re.fullmatch('[0-9a-f]{40}',a.revision) or code != ROOT/'jobs'/a.revision/ENTRY/'code' or Path(__file__).resolve() != code/HELPERS[0]:raise ValueError('Exact published cost source required')
    if a.mode == 'host':
        if a.native_args or sys.platform != 'linux' or os.geteuid() !=0 or os.uname().nodename !='scenesmith-ncc-h100-01' or os.environ.get('DOCKER_HOST') != 'unix://'+str(ROOT)+'/docker.sock':raise ValueError('Actual VM01 root CPU host required')
        result = host(code,a.revision)
    else:
        if len(a.native_args) !=4:raise ValueError('Exact native proof/deadline arguments required')
        out,size,digest,end = a.native_args;deadline = float(end)
        if not math.isfinite(deadline) or deadline-time.monotonic() >720:raise ValueError('Shared finite host deadline required')
        result = native(code,a.revision,Path(out),dict(bytes=int(size),sha256=digest),deadline)
    print(json.dumps(dict(status=result['status'],decision=result.get('decision'),error_type=result.get('error_type')),sort_keys=True))
    if result['status'] !='pass':raise SystemExit(1)


if __name__ == '__main__':main()

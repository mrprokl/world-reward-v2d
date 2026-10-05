"""One fresh tiny procedural GPU arithmetic control; no models/data/FIT."""
import argparse
from collections.abc import Mapping
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
ENTRY = 'run_coherent_pair_gpu_probe'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
RESULT = 'results/coherent-pair-gpu-probe-v1'
BUDGET, MEMORY, RTOL, ATOL, STEP, FD_TOL = 1200, 6*1024**3, 1e-12, 1e-12, 1e-6, 1e-7
CASES = ((3,3,2,'complete'), (3,3,3,'aliases'), (2,3,2,'missing'), (2,2,0,'no_hoi'),
         (1,2,0,'unsupported'), (0,2,2,'empty_persons'), (2,0,2,'empty_objects'), (0,0,0,'empty'))
REUSED = {
 'infra/mediapipe_cpu_runtime_verify.py': (23559,'936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
 'src/world_reward/__init__.py': (81,'1ac8c64041277f0c3cf4d8838b3153210d7076dd86a5121d1e6ef494b64b36a1'),
 'src/world_reward/coherent_pair_learning.py': (15335,'1c0f4f352bd09729cb6c671c2987d430961dc15cb841cd17bd99d039df2f5f96'),
 'src/world_reward/coherent_route_scorer.py': (21201,'a91ba827bddad2f4376e1fc5f16cf253a9a9749218c6df4115c47552f9086b9a'),
 'src/world_reward/coherent_pair_cache.py': (14362,'4dc033f4ae89c5e005bf46e3a6a07114cc58cafa0cc4febf10392c9a2a98edc6'),
 'src/world_reward/coherent_pair_packed.py': (9514,'3eeb51a038bb12af1db6c2a624679f237dd21fb6dc2a77afbfd9f101111ac331'),
 'src/world_reward/coherent_pair_packed_score.py': (7242,'4cb08e538847cef3506cb25d5a381e2ad96c7ab3122253f51aea0e942c5c3812'),
 'src/world_reward/coherent_pair_marginal.py': (10189,'7bc6f51aa3be51f880a6e4ab74fededa7403b01158d3743dc6f20816afac5307'),
 'src/world_reward/coherent_pair_packed_torch.py': (12832,'54fc60baa8390e24aaade74f48c9b473b9396a92e4e3f3469caa55552b96a812'),
 'src/world_reward/interaction_candidate_evidence.py': (11997,'5fb3db6b0a97ef396e01fa9c7852857b02c93625ebb65ee07637136432211235'),
 'src/world_reward/interaction_tuple_evidence.py': (8799,'1c4cdb80a45b6e8a3840d1ff87c033e24d40f80cd8d0b1caf005349ae3211930'),
 'src/world_reward/person_pose_observations.py': (7614,'c93fa24f4ce4bd6b5f4c53f4e48943b2c61d0c20a2211b36ef6d0ccb6b3ce375'),
 'src/world_reward/hoi_detr_observations.py': (14388,'047a80cb618fb98da19b233c199fb927c9b9930b1a6573f62abb2e07ac6ebd8f')}
HELPERS = ('infra/coherent_pair_gpu_probe.py','infra/run_coherent_pair_gpu_probe.sh',*REUSED)
FLOAT_FIELDS = ('native_scores_a','native_scores_b','scores_a','scores_b',
    'geometry_derivatives_a','geometry_derivatives_b','alpha_derivatives_b')
BOOL_FIELDS = ('native_supported','native_route_supported','supported')


def runtime(code):
    spec = importlib.util.spec_from_file_location('wr_gpu_probe_runtime',code/'infra/mediapipe_cpu_runtime_verify.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def manifest():
    return dict(schema='world_reward.coherent_pair_gpu_probe.v1',cases=[dict(persons=n,objects=o,hand_copies=k,
        native_pairs=k*k,variant=v) for n,o,k,v in CASES],image_id=IMAGE,budget_seconds=BUDGET,
        torch_memory_bytes=MEMORY,docker_host_memory_bytes=MEMORY,temperature=.8125,alphas=[0.,.3125],
        theta=[(-1 if i%2 else 1)*(i+1)/64. for i in range(17)],scales=[(i+3)/8. for i in range(12)],
        repeats=2,fd_step=STEP,fd_rtol=FD_TOL,fd_atol=FD_TOL,oracle_rtol=RTOL,oracle_atol=ATOL,
        segment_widths=[1,9,6,2,17],fixture_image_size=[48,64],fixture_frame_index=7,
        fixtures_generated_from_new_procedural_recipe=True,quality_verified=False,fullbank_verified=False)


def control(argv, timeout=8):
    r = subprocess.run(argv,capture_output=True,timeout=timeout,check=False)
    if r.returncode or len(r.stdout)>32768 or len(r.stderr)>16384:raise ValueError('Bounded runtime control failed')
    return r.stdout


def image():
    row = json.loads(control(['docker','image','inspect',IMAGE,'--format',
        '{"Id":{{json .Id}},"Os":{{json .Os}},"Architecture":{{json .Architecture}},"RootFS":{{json .RootFS}}}']))
    if row['Id']!=IMAGE or row['Os']!='linux' or row['Architecture']!='amd64' or row['RootFS']['Type']!='layers':
        raise ValueError('Exact B47 Linux AMD64 image required')
    return row


def proof(rt, code, rev):
    source = rt.source(ROOT,code,rev,ENTRY,HELPERS)
    rt.require(all(source['helpers'][n]==dict(bytes=size,sha256=digest) for n,(size,digest) in REUSED.items()),
        'Frozen scorer/oracle/helper bytes differ')
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact dispatch namespace required')
    return dict(source_binding=source,image=image(),manifest=manifest())


def leaves(code):
    return [*(code/n for n in HELPERS),code.parent/'revision',code.parent/'source-sha256']


def fixtures(np):
    """New procedural observations, not old tests/model outputs/reference labels."""
    from world_reward.person_pose_observations import PersonPoseObservations
    from world_reward.hoi_detr_observations import HOIDetrObservations
    from world_reward.interaction_candidate_evidence import GenericObjectObservations,build_interaction_candidate_evidence
    from world_reward.coherent_pair_learning import pair_route_bank
    for n,o,k,variant in CASES:
        pb = np.array([[2.+i*9.,2.,31.+i*9.,45.] for i in range(n)],np.float64).reshape(n,4)
        xy = np.full((n,133,2),11.,np.float64);scores=np.full((n,133),.625,np.float32)
        for i in range(n):
            xy[i,[9,91,10,112]] = [[8.+i*9.,17.],[9.+i*9.,18.],[22.+i*9.,29.],[23.+i*9.,30.]]
        ob = np.array([[13.+i*11.,12.+i*3.,19.+i*11.,23.+i*3.] for i in range(o)],np.float64).reshape(o,4)
        if variant=='aliases':pb[1:]=pb[0];xy[1:]=xy[0];ob[1:]=ob[0]
        if variant=='missing':
            scores[0,[9,91]]=0.;scores[-1,[10,112]]=-.125;xy[-1,9]=[-3.,51.];ob[-1,2:]=ob[-1,:2]
        if variant=='unsupported':scores[:]=0.;ob[:,2:]=ob[:,:2]
        person=PersonPoseObservations(7,(48,64),tuple('new-person-'+str(i) for i in range(n)),pb,
            np.full(n,.875,np.float32),xy,scores)
        objects=GenericObjectObservations(7,(48,64),tuple('new-object-'+str(i) for i in range(o)),ob,np.zeros(o,np.float32))
        hoi=None
        if k:
            boxes=np.tile([[7.,14.,12.,21.],[14.,13.,21.,24.]],(k,1)).astype(np.float32)
            if variant!='aliases':
                boxes[::2]+=np.arange(k,dtype=np.float32)[:,None]*np.array([2.,1.,2.,1.],np.float32)
                boxes[1::2]+=np.arange(k,dtype=np.float32)[:,None]*np.array([1.,2.,1.,2.],np.float32)
            classes=np.tile([0,1],k).astype(np.int64);ids=np.arange(2*k,dtype=np.int64);hs,oslots=ids[classes==0],ids[classes==1]
            pairs=np.column_stack((np.repeat(hs,k),np.tile(oslots,k))).astype(np.int64)
            raw=np.full(2*k,.75,np.float32);logits=np.column_stack((np.zeros(k*k),(np.arange(k*k)%5-2)/4.)).astype(np.float32)
            hoi=HOIDetrObservations(7,(48,64),np.zeros((1500,3),np.float32),np.zeros((1500,4),np.float32),
                np.zeros((1500,256),np.float32),np.column_stack((boxes,raw)),ids,ids,ids,classes,boxes,raw,raw,
                pairs,logits,np.empty((0,2),np.int64),np.empty((0,2),np.float32))
        yield pair_route_bank(person,build_interaction_candidate_evidence(person,objects,hoi))


def device_fingerprint(t, prepared):
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,Mapping):
            for name in sorted(value):digest.update(name.encode()+b'\0');add(value[name])
        else:
            a=value.detach().cpu().numpy();digest.update(str((str(value.device),str(value.dtype),tuple(a.shape))).encode());digest.update(a.tobytes())
    with t.no_grad():
        for name in ('_arrays','_layouts','_pairs'):digest.update(name.encode());add(getattr(prepared,name))
    return digest.hexdigest()


def arrays(result):
    return {n:getattr(result,n).detach().cpu().numpy() for n in (*FLOAT_FIELDS,*BOOL_FIELDS)}


def bits(a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise ValueError('Repeat dtype/shape/raw bytes differ')


def segment_controls(np,t,scorer,check):
    rows=[]
    for width in (1,9,6,2,17):
        check();x=np.arange(5*width,dtype=np.float64).reshape(5,width)/16.;offsets=np.array([0,2,2,5],np.int64)
        for empty in (False,True):
            values=x[:0] if empty else x;off=np.zeros(4,np.int64) if empty else offsets
            dx=t.tensor(values,dtype=t.float64,device='cuda:0');do=t.tensor(off,device='cuda:0')
            for op in ('sum','max'):
                first=scorer._segment(t,dx,op,do).cpu().numpy();second=scorer._segment(t,dx,op,do).cpu().numpy();bits(first,second)
                expected=np.full((3,width),0. if op=='sum' else -np.inf)
                for j in range(3):
                    z=values[off[j]:off[j+1]]
                    if len(z):expected[j]=z.sum(axis=0) if op=='sum' else z.max(axis=0)
                bits(first,expected);rows.append(dict(width=width,empty=empty,operation=op,bytes_sha256=hashlib.sha256(first.tobytes()).hexdigest()))
    # Known overflow boundary must reject, never clip or invent support.
    bad=t.tensor([-1e308,1e308],dtype=t.float64,device='cuda:0')
    try:scorer._mix(t,bad,t.tensor([0,2],device='cuda:0'),t.tensor([0,0],device='cuda:0'),t.tensor([2],device='cuda:0'),.8125)
    except ValueError:pass
    else:raise ValueError('Extreme mixture overflow was not rejected')
    return rows


def arithmetic(np,t,scorer,check,report,snapshots):
    from world_reward import coherent_route_scorer as core,coherent_pair_learning as old,coherent_pair_marginal as oracle
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    theta=np.asarray(manifest()['theta'],np.float64);scale=old.PairScale(np.asarray(manifest()['scales']),np.ones(12,bool))
    report['segment_controls']=segment_controls(np,t,scorer,check);report['fixtures']=[]
    for case,bank in enumerate(fixtures(np)):
        check();cache=prepare_pair_cache(bank,scale);packed=prepare_marginal_packed(cache);reference=oracle.MarginalPairReference(cache)
        host_before=core._fingerprint((packed.arrays,packed.identity));prepared=scorer.prepare_marginal_packed_torch(packed,device='cuda:0')
        device_before=device_fingerprint(t,prepared);snapshots.append((packed,prepared,host_before,device_before))
        row=dict(case=case,variant=CASES[case][3],counts=list(cache.counts),
            distribution_fingerprint=host_before,device_tables_before=device_before,results=[]);report['fixtures'].append(row)
        for alpha in (0.,.3125):
            check();actual=scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=alpha)
            first=arrays(actual);second=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=alpha))
            expected=oracle.score_pair_marginal(reference,theta,temperature=.8125,alpha=alpha)
            cpu=score_pair_marginal_packed(packed,theta,temperature=.8125,alpha=alpha)
            for name in (*FLOAT_FIELDS,*BOOL_FIELDS):bits(first[name],second[name])
            for name in FLOAT_FIELDS:
                np.testing.assert_allclose(first[name],getattr(expected,name),rtol=RTOL,atol=ATOL,equal_nan=True)
                np.testing.assert_allclose(first[name],getattr(cpu,name),rtol=RTOL,atol=ATOL,equal_nan=True)
            for name in BOOL_FIELDS:bits(first[name],getattr(expected,name))
            if core._fingerprint(actual.identity)!=core._fingerprint(expected.identity) or actual.parameter_fingerprint!=expected.parameter_fingerprint:
                raise ValueError('Original IDs/provenance/parameters differ')
            if alpha==0.:
                for a,b in (('native_scores_a','native_scores_b'),('scores_a','scores_b'),('geometry_derivatives_a','geometry_derivatives_b')):
                    bits(first[a],first[b])
                if actual.native_scores_a.data_ptr()!=actual.native_scores_b.data_ptr() or actual.scores_a.data_ptr()!=actual.scores_b.data_ptr() or actual.geometry_derivatives_a.data_ptr()!=actual.geometry_derivatives_b.data_ptr():
                    raise ValueError('Alpha-zero shared buffers required')
            fd_calls=0;active=first['supported']
            if active.any():
                for j in range(17):
                    check();delta=np.zeros(17);delta[j]=STEP
                    plus=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta+delta,temperature=.8125,alpha=alpha))
                    minus=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta-delta,temperature=.8125,alpha=alpha));fd_calls+=2
                    for name,derivative in (('scores_a','geometry_derivatives_a'),('scores_b','geometry_derivatives_b')):
                        numerical=(plus[name][active]-minus[name][active])/(2*STEP)
                        np.testing.assert_allclose(numerical,first[derivative][active,j],rtol=FD_TOL,atol=FD_TOL)
                if alpha==0.:
                    p=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=STEP))
                    q=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=2*STEP))
                    numerical=(-3*first['scores_b'][active]+4*p['scores_b'][active]-q['scores_b'][active])/(2*STEP)
                else:
                    p=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=alpha+STEP))
                    q=arrays(scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=.8125,alpha=alpha-STEP))
                    numerical=(p['scores_b'][active]-q['scores_b'][active])/(2*STEP)
                fd_calls+=2;np.testing.assert_allclose(numerical,first['alpha_derivatives_b'][active],rtol=FD_TOL,atol=FD_TOL)
            row['results'].append(dict(alpha=alpha,repeat_bit_identical=True,oracle_passed=True,fd17_and_alpha_passed=bool(active.any()),
                fd_skipped_no_supported_target=not bool(active.any()),fd_calls=fd_calls,
                result_sha256=core._fingerprint(first),parameter_fingerprint=actual.parameter_fingerprint))
        check();row['device_tables_after']=device_fingerprint(t,prepared)
        row['host_tables_after']=core._fingerprint((packed.arrays,packed.identity))
        if row['device_tables_after']!=device_before or row['host_tables_after']!=host_before:raise ValueError('Prepared snapshots mutated')
    report.update(fixtures_completed=8,measurement_scope='tiny_arithmetic_only',phase='complete')


def publish(rt,path,value,deadline,*,seal=False):
    """Same owned FD demotes late PASS after hashing/fsync/directory sealing."""
    def raw():return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o400)
        def update():stream.seek(0);stream.write(raw());stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            update()
            if seal:path.parent.chmod(0o500)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            rt.require(rt.identity(path,1<<20)==dict(bytes=len(raw()),sha256=hashlib.sha256(raw()).hexdigest()),'Receipt bytes differ')
            if time.monotonic()>=deadline:raise TimeoutError('Final publication crossed inclusive budget')
        except BaseException as exc:
            value.update(status='fail',error_type=error_family(exc),decision='CLOSED_TINY_GPU_CONTROL');update()


def native(code,rev,out,pin,deadline):
    rt=runtime(code);previous=rt.pinned(out/'proof.json',pin,1<<20)
    before={str(p):rt.identity(p,2<<20) for p in leaves(code)}
    rt.require(out==ROOT/RESULT and stat.S_IMODE(out.stat().st_mode)==0o755 and out.stat().st_uid==1000
        and os.geteuid()==1000 and sys.platform=='linux' and sys.version_info[:2]==(3,11)
        and os.environ.get('WR_IMAGE_ID')==IMAGE and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and {p.name for p in out.iterdir()}=={'proof.json','container.cid'},'Exact isolated GPU/native namespace required')
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',(out/'container.cid').read_bytes()),'Owned CID missing')
    rt.identity(out/'container.cid',65,readonly=False)
    rt.require(previous['manifest']==manifest() and previous['source_binding']['producer_revision']==rev
        and all(before[str(code/k)]==v for k,v in previous['source_binding']['helpers'].items())
        and all(before[str(code.parent/k)]==v for k,v in previous['source_binding']['markers'].items())
        and (code.parent/'revision').read_bytes()==(rev+'\n').encode(),'Manifest/source differs before imports')
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Inclusive1200s tiny probe deadline')
    def cancelled(*_):raise TimeoutError('Tiny GPU probe cancelled')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    report=dict(schema=manifest()['schema'],stage='coherent_pair_gpu_native',status='fail',phase='imports',producer_revision=rev,
        source_binding=previous['source_binding'],manifest=manifest(),image_id=IMAGE,source_rehashed_after=False,
        models_loaded=False,rgb_read=False,external_references_read=False,challenge_inputs_used=False,fit_executed=False,
        quality_verified=False,adoption=False,fullbank_verified=False,decision='CLOSED_TINY_GPU_CONTROL')
    snapshots=[];t=None
    try:
        check();sys.path.insert(0,str(code/'src'));import numpy as np
        from world_reward import coherent_pair_packed_torch as scorer
        check();import torch as t
        rt.require(t.__version__=='2.5.1+cu124' and t.version.cuda=='12.4' and t.cuda.is_available()
            and t.cuda.device_count()==1 and t.cuda.get_device_capability(0)==(9,0)
            and 'H100' in t.cuda.get_device_name(0),'Actual frozen Torch/CUDA/H100 required')
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        t.use_deterministic_algorithms(True);t.set_num_threads(4)
        props=t.cuda.get_device_properties(0);t.cuda.set_per_process_memory_fraction(MEMORY/props.total_memory,0)
        t.cuda.reset_peak_memory_stats(0)
        report['runtime']=dict(python=sys.version.split()[0],numpy=np.__version__,torch=t.__version__,cuda=t.version.cuda,
            gpu_name=props.name,capability=[props.major,props.minor],gpu_total_bytes=props.total_memory,
            deterministic_algorithms=True,tf32=False,cublas_workspace_config=':4096:8')
        arithmetic(np,t,scorer,check,report,snapshots);t.cuda.synchronize();check()
        report.update(peak_torch_allocated=t.cuda.max_memory_allocated(),peak_torch_reserved=t.cuda.max_memory_reserved(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        rt.require(report['peak_torch_allocated']<=MEMORY and report['peak_torch_reserved']<=MEMORY,'Fixed6GiB Torch allocator gate exceeded')
        report.update(status='pass',decision='TINY_GPU_ARITHMETIC_PASS_NOT_FULLBANK_QUALIFICATION')
    except BaseException as exc:report.update(status='fail',error_type=error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if snapshots:
                from world_reward import coherent_route_scorer as core
                rt.require(all(device_fingerprint(t,prepared)==device_before
                    and core._fingerprint((packed.arrays,packed.identity))==host_before
                    for packed,prepared,host_before,device_before in snapshots),'Device/host snapshots changed after checks')
                report['all_prepared_snapshots_rehashed_after']=True
            rt.require(before=={str(p):rt.identity(p,2<<20) for p in leaves(code)}
                and rt.identity(out/'proof.json',1<<20)==pin,'Native source/proof changed')
            report['source_rehashed_after']=True
        except Exception as exc:report.update(status='fail',post_error_type=error_family(exc))
        publish(rt,out/'native.json',report,deadline)
    return report


def error_family(exc):
    return type(exc).__name__ if type(exc).__name__ in ('ValueError','AssertionError','TimeoutError','RuntimeError','OSError','MemoryError') else 'OtherError'


def cleanup(rt,cidfile,name,rev):
    current=control(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).decode().split()
    if not current:
        if cidfile.exists():
            rt.identity(cidfile,65,readonly=False)
            rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',cidfile.read_bytes()),'Owned CID malformed')
            cid=cidfile.read_text().strip()
            rt.require(not control(['docker','ps','-aq','--no-trunc','--filter','id='+cid]).strip(),
                'Saved CID survives without owned name; never remove foreign renamed container')
            cidfile.chmod(0o400)
        return True
    rt.require(len(current)==1 and cidfile.exists(),'Cannot clean unowned container')
    rt.identity(cidfile,65,readonly=False);cid=cidfile.read_text().strip()
    rt.require(re.fullmatch('[0-9a-f]{64}',cid) and current==[cid],'Owned CID differs')
    metadata=control(['docker','inspect',cid,'--format','{{.Id}}|{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.tiny_gpu.owner"}}']).decode().strip()
    rt.require(metadata==cid+'|'+IMAGE+'|/'+name+'|'+rev,'Cannot remove foreign container')
    for command in (['docker','stop','--time','2',cid],['docker','kill',cid],['docker','rm','--force',cid]):
        try:subprocess.run(command,capture_output=True,timeout=5,check=False)
        except subprocess.TimeoutExpired:pass
    rt.require(not control(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).strip(),'Owned GPU container survives')
    rt.require(not control(['docker','ps','-aq','--no-trunc','--filter','id='+cid]).strip(),'Saved owned CID survives')
    cidfile.chmod(0o400);return True


def validate_native(rt,value,binding,rev):
    expected=dict(schema=manifest()['schema'],stage='coherent_pair_gpu_native',status='pass',phase='complete',
        producer_revision=rev,source_binding=binding,manifest=manifest(),image_id=IMAGE,
        source_rehashed_after=True,all_prepared_snapshots_rehashed_after=True,fixtures_completed=8,
        fullbank_verified=False,quality_verified=False,fit_executed=False,models_loaded=False,rgb_read=False,
        external_references_read=False,challenge_inputs_used=False,adoption=False,
        decision='TINY_GPU_ARITHMETIC_PASS_NOT_FULLBANK_QUALIFICATION')
    rt.require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in expected.items()),'Complete tiny native PASS required')
    rt.require(len(value['fixtures'])==8 and len(value['segment_controls'])==20,'Every fixed control required')
    actual=value.get('runtime',{})
    fixed_runtime=dict(torch='2.5.1+cu124',cuda='12.4',capability=[9,0],deterministic_algorithms=True,
        tf32=False,cublas_workspace_config=':4096:8')
    rt.require(all(type(actual.get(k)) is type(v) and actual[k]==v for k,v in fixed_runtime.items())
        and type(actual.get('gpu_name')) is str and 'H100' in actual['gpu_name']
        and type(actual.get('gpu_total_bytes')) is int and actual['gpu_total_bytes']>MEMORY
        and type(actual.get('python')) is str and re.fullmatch(r'3\.11\.[0-9]+',actual['python'])
        and type(actual.get('numpy')) is str and actual['numpy'],'Actual FP64 GPU runtime evidence required')
    controls=[(width,empty,op) for width in (1,9,6,2,17) for empty in (False,True) for op in ('sum','max')]
    for row,(width,empty,op) in zip(value['segment_controls'],controls):
        rt.require(type(row) is dict and set(row)=={'width','empty','operation','bytes_sha256'}
            and type(row['width']) is int and row['width']==width and type(row['empty']) is bool and row['empty']==empty
            and row['operation']==op and re.fullmatch('[0-9a-f]{64}',str(row['bytes_sha256'])),
            'Exact ordered20 segment controls required')
    for case,(row,(n,o,k,variant)) in enumerate(zip(value['fixtures'],CASES)):
        fd_expected=variant in ('complete','aliases','missing','no_hoi')
        rt.require(type(row['case']) is int and row['case']==case and row['variant']==variant and row['counts']==[n,o,k*k]
            and row['device_tables_before']==row['device_tables_after']
            and row['distribution_fingerprint']==row['host_tables_after']
            and all(re.fullmatch('[0-9a-f]{64}',str(row[key])) for key in
                ('device_tables_before','device_tables_after','distribution_fingerprint','host_tables_after'))
            and [r['alpha'] for r in row['results']]==[0.,.3125]
            and all(r['repeat_bit_identical'] is True and r['oracle_passed'] is True
                and r['fd17_and_alpha_passed'] is fd_expected and r['fd_skipped_no_supported_target'] is (not fd_expected)
                and type(r['fd_calls']) is int and r['fd_calls']==(36 if fd_expected else 0)
                and all(re.fullmatch('[0-9a-f]{64}',str(r[key])) for key in ('result_sha256','parameter_fingerprint')) for r in row['results']),
            'Fixed complete source/FD/repeat fixture required')
    rt.require(all(type(value[k]) is int and 0<=value[k]<=MEMORY for k in ('peak_torch_allocated','peak_torch_reserved')),
        'Fixed GPU allocator gate required')


def host(code,rev):
    started=time.monotonic();deadline=started+BUDGET;rt=runtime(code)
    def cancelled(*_):raise TimeoutError('Tiny GPU host cancelled/deadline')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,BUDGET)
    before=proof(rt,code,rev);out=rt.canonical(ROOT/RESULT)
    rt.require(not out.exists(),'Fresh tiny GPU namespace required');out.mkdir(mode=0o755);os.chown(out,1000,1000);out.chmod(0o755)
    inode=(out.stat().st_dev,out.stat().st_ino);raw=(json.dumps(before,sort_keys=True)+'\n').encode()
    rt.write(out/'proof.json',raw,0o444);pin=rt.identity(out/'proof.json',1<<20)
    name='world-reward-tiny-gpu-'+rev[:12];cidfile=out/'container.cid';owned=False;removed=False
    report=dict(schema=manifest()['schema'],stage='coherent_pair_gpu_host',status='fail',producer_revision=rev,
        source_binding=before['source_binding'],image=before['image'],manifest=manifest(),decision='CLOSED_TINY_GPU_CONTROL')
    try:
        rt.require(not control(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).strip(),'Owned name already exists')
        owned=True
        argv=['docker','run','--rm','--cidfile',str(cidfile),'--name',name,'--label','world_reward.tiny_gpu.owner='+rev,
            '--gpus','all','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--memory','6g','--cpus','4','--pids-limit','256','--user','1000:1000','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=128m','--entrypoint','/usr/bin/env']
        for path in leaves(code):argv+=['--mount',f'type=bind,src={path},dst={path},readonly']
        argv+=['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
            'PYTHONDONTWRITEBYTECODE=1','WR_IMAGE_ID='+IMAGE,'CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'python','-I','-B',str(code/HELPERS[0]),'native',str(code),rev,str(out),str(pin['bytes']),pin['sha256'],str(deadline)]
        r=subprocess.run(argv,capture_output=True,timeout=max(.001,deadline-time.monotonic()),check=False)
        report.update(native_exit_code=r.returncode,diagnostics={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in (('stdout',r.stdout),('stderr',r.stderr))})
        rt.require(len(r.stdout)<=32768 and len(r.stderr)<=32768,'Bounded native diagnostic required')
        removed=cleanup(rt,cidfile,name,rev);value=rt.strict((out/'native.json').read_bytes())
        rt.require(r.returncode==0,'Native process failed');validate_native(rt,value,before['source_binding'],rev)
        report.update(status='pass',native_report_identity=rt.identity(out/'native.json',1<<20),decision=value['decision'])
    except BaseException as exc:report.update(status='fail',error_type=error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:removed=cleanup(rt,cidfile,name,rev) if owned else False
        except Exception as exc:report.update(status='fail',cleanup_error_type=error_family(exc))
        try:report['source_rehashed_after']=proof(rt,code,rev)==before
        except Exception as exc:report.update(status='fail',post_error_type=error_family(exc))
        rt.require((out.stat().st_dev,out.stat().st_ino)==inode and out.stat().st_uid==1000,'Owned output replaced')
        rt.require({p.name for p in out.iterdir()}<={'proof.json','container.cid','native.json'},'Foreign output artifact')
        report.update(owned_container_removed=removed,elapsed_seconds=time.monotonic()-started)
        if not removed or not report.get('source_rehashed_after') or time.monotonic()>=deadline:report['status']='fail'
        if report['status']!='pass':report['decision']='CLOSED_TINY_GPU_CONTROL'
        publish(rt,out/'report.json',report,deadline,seal=True)
    return report


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('mode',choices=('host','native'));p.add_argument('code');p.add_argument('revision');p.add_argument('native_args',nargs='*');a=p.parse_args();code=Path(a.code)
    if not re.fullmatch('[0-9a-f]{40}',a.revision) or code!=ROOT/'jobs'/a.revision/ENTRY/'code' or Path(__file__).resolve()!=code/HELPERS[0]:raise ValueError('Exact immutable probe source required')
    if a.mode=='host':
        if a.native_args or sys.platform!='linux' or os.geteuid()!=0 or os.uname().nodename!='world-reward-ncc-h100-02' or os.environ.get('DOCKER_HOST')!='unix://'+str(ROOT)+'/docker.sock':raise ValueError('Actual VM02 root GPU host required')
        lock=ROOT/'jobs/.world-reward-h100.lock';fd=os.fstat(9);node=lock.lstat()
        if lock.is_symlink() or not stat.S_ISREG(node.st_mode) or (fd.st_dev,fd.st_ino)!=(node.st_dev,node.st_ino):raise ValueError('Actual FD9 GPU lease required')
        result=host(code,a.revision)
    else:
        if len(a.native_args)!=4:raise ValueError('Exact native proof/deadline required')
        out,size,digest,end=a.native_args;deadline=float(end)
        if not math.isfinite(deadline) or not 0<deadline-time.monotonic()<=BUDGET:raise ValueError('Shared host deadline required')
        result=native(code,a.revision,Path(out),dict(bytes=int(size),sha256=digest),deadline)
    print(json.dumps(dict(status=result['status'],decision=result['decision'],error_type=result.get('error_type')),sort_keys=True))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()

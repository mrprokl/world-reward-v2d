"""One new complete-bank marginal score/VJP control, not FIT or ownership."""
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
ENTRY = 'run_coherent_pair_gpu_fullbank_cost'
RESULT = 'results/coherent-pair-gpu-fullbank-cost-v1'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SCHEMA = 'world_reward.coherent_pair_gpu_fullbank_cost.v1'
BUDGET, MEMORY = 1200, 6*1024**3
DECISION = 'FULLBANK_SCORE_GRADIENT_CONTROL_PASS_PENDING_OBJECTIVE'
PRIOR_REV = 'd3506722efb77d034ee329d04609c621188d6c0b'
PRIOR_PINS = {
 'proof.json': dict(bytes=7622,sha256='dc1be997eabb0407483af794561115db147172a1dfa0f8bc6df7b9bb12b6af5a'),
 'native.json': dict(bytes=16807,sha256='d95d66d1e4a5b0d3ab38028c9c25bd40f5ea4df749e12da23329ca337b23f6ab'),
 'report.json': dict(bytes=8330,sha256='e24940c7fbd27a969fdb221694b07e7e2ef6d5a3e5f4701e09724185be49420c')}
REUSED = {
 'infra/coherent_pair_gpu_probe.py': (32097,'989b531802797e0fc5eb90a1ef6a4a19e4c7c68d797e8b0451934f68e5a1571b'),
 'infra/coherent_pair_cost_probe.py': (21927,'5757eeaf4de82cbeca268a2df48b0d5612f08ee63b30958aa8c0f1f905ed0e2a'),
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
HELPERS = ('infra/coherent_pair_gpu_fullbank_cost.py','infra/run_coherent_pair_gpu_fullbank_cost.sh',*REUSED)


def module(code, name, path):
    spec = importlib.util.spec_from_file_location(name,code/path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def helpers(code):
    generic = module(code,'fullbank_generic','infra/coherent_pair_gpu_probe.py')
    return generic,generic.runtime(code),module(code,'original_fullbank_fixture','infra/coherent_pair_cost_probe.py')


def manifest():
    return dict(schema=SCHEMA,persons=4,objects=3600,native_pairs=64,native_tokens=1500,
        source_fixture_index=1,source_fixture_count=2,original_frame_index=0,image_size=[720,960],
        theta=[(-1 if i%2 else 1)*(i+1)/64. for i in range(17)],scales=[(i+3)/8. for i in range(12)],
        alphas=[0.,.3125],temperature=.8125,repeats=2,oracle_rtol=1e-12,oracle_atol=1e-12,
        cpu_reference='packed_analytic_all_rows_not_enumerative',projection_evaluations=[1,100,1000],
        projection_is_descriptive_not_bound=True,projection_threshold=None,budget_seconds=BUDGET,
        memory_bytes=MEMORY,image_id=IMAGE,fit_executed=False,optimizer_executed=False)


def tiny_evidence(rt, code):
    old = ROOT/'jobs'/PRIOR_REV/'run_coherent_pair_gpu_probe_v2'/'code'
    out = ROOT/'results/coherent-pair-gpu-probe-v2'
    values = {n:rt.pinned(out/n,p,1<<20) for n,p in PRIOR_PINS.items()}
    proof,native,host = (values[n] for n in ('proof.json','native.json','report.json'))
    binding = proof['source_binding']
    rt.require(host['status']=='pass' and native['status']=='pass' and native['phase']=='complete'
        and host['source_binding']==native['source_binding']==binding and binding['producer_revision']==PRIOR_REV
        and host['native_report_identity']==PRIOR_PINS['native.json'] and host['native_exit_code']==0
        and host['source_rehashed_after'] is True and host['owned_container_removed'] is True
        and native['fixtures_completed']==8 and len(native['segment_controls'])==20
        and native['source_rehashed_after'] is True and native['all_prepared_snapshots_rehashed_after'] is True
        and native['image_id']==IMAGE and host['image']==proof['image']
        and host['decision']==native['decision']=='TINY_GPU_ARITHMETIC_PASS_NOT_FULLBANK_QUALIFICATION',
        'Exact independently sealed tiny GPU qualification required')
    names=('infra/coherent_pair_gpu_probe.py','infra/run_coherent_pair_gpu_probe_v2.sh',
        *(n for n in REUSED if n not in ('infra/coherent_pair_gpu_probe.py','infra/coherent_pair_cost_probe.py')))
    rt.require(rt.source(ROOT,old,PRIOR_REV,'run_coherent_pair_gpu_probe_v2',names)==binding,
        'Original complete tiny producer source differs')
    return dict(producer_revision=PRIOR_REV,receipts=PRIOR_PINS,source_binding=binding,
        historical_unit_collected=True,independent_unit_exit_available=False)


def proof(g, rt, code, rev):
    source=rt.source(ROOT,code,rev,ENTRY,HELPERS)
    rt.require(all(source['helpers'][n]==dict(bytes=b,sha256=s) for n,(b,s) in REUSED.items()),'Frozen original math/fixture/helpers differ')
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact current source namespace required')
    return dict(source_binding=source,manifest=manifest(),image=g.image(),tiny_qualification=tiny_evidence(rt,code))


def leaves(code):
    return [*(code/n for n in HELPERS),code.parent/'revision',code.parent/'source-sha256']


def timed(t, check, function):
    check();t.cuda.synchronize();start=time.monotonic();value=function();t.cuda.synchronize();check()
    return value,time.monotonic()-start


def compare(np, g, actual, repeat, cpu):
    for name in (*g.FLOAT_FIELDS,*g.BOOL_FIELDS):g.bits(actual[name],repeat[name])
    for name in g.FLOAT_FIELDS:
        reference=getattr(cpu,name)
        if actual[name].dtype!=reference.dtype or actual[name].shape!=reference.shape:raise ValueError('Complete FP64 oracle shape/dtype required')
        np.testing.assert_allclose(actual[name],reference,rtol=1e-12,atol=1e-12,equal_nan=True)
    for name in g.BOOL_FIELDS:g.bits(actual[name],getattr(cpu,name))
    if actual['native_scores_a'].shape != cpu.native_scores_a.shape:raise ValueError('All native rows required')
    return dict(repeat_bits_exact=True,all_cpu_packed_arrays_passed=True,
        supported_pairs=int(actual['supported'].sum()),native_supported=int(actual['native_supported'].sum()))


def projection(preparation, upload, passes):
    worst=max(passes)
    return [dict(evaluations=n,projected_seconds=preparation+upload+n*worst) for n in (1,100,1000)]


def recheck_snapshots(g,t,snapshots):
    from world_reward import coherent_route_scorer as core
    for s in snapshots:
        if core._fingerprint(s['bank'])!=s['bank_sha256']:raise ValueError('Original full bank mutated')
        if 'cache' in s and core._fingerprint((s['cache'].factors,s['cache'].scales,
            s['cache'].person_members,s['cache'].object_members))!=s['cache'].factor_fingerprint:
            raise ValueError('Original cached factors mutated')
        if 'packed' in s and core._fingerprint((s['packed'].arrays,s['packed'].identity))!=s['packed_sha256']:
            raise ValueError('Original packed tables mutated')
        if 'device' in s and g.device_fingerprint(t,s['device'])!=s['device_sha256']:raise ValueError('Uploaded tables mutated')


def measure(np, t, g, fixture, check, report, snapshots, *, object_count=3600):
    """Small count is unit-test-only. No runtime selector or full enumerative score."""
    from world_reward import coherent_route_scorer as core,coherent_pair_learning as learner
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    from world_reward import coherent_pair_packed_torch as scorer
    sources=[];iterator=fixture.fixtures(np,object_count=object_count,source_records=sources)
    try:
        report['phase']='fixture_prepare';check();start=time.monotonic();first=next(iterator)
        del first;bank=next(iterator);report['fixture_prepare_seconds']=time.monotonic()-start
        bank_before=core._fingerprint(bank);snapshot=dict(bank=bank,bank_sha256=bank_before);snapshots.append(snapshot)
        n,o,k=learner._bank(bank)
        if (n,o,k)!=(4,object_count,64):raise ValueError('All fixed full-bank slots required')
        report.update(counts=[n,o,k],native_tokens=sources[-1]['native_tokens'],candidate_rows=n*2*o,
            tuple_rows=n*2*k,bridge_rows=k*o,person_groups=len(bank.person_boxes),object_groups=len(bank.object_boxes),
            original_frame_index=bank.evidence.original_frame_index,image_size=list(bank.evidence.image_size),
            bank_sha256=bank_before,full_original_observations_sha256=sources[-1]['full_original_observations_sha256'])
        report['phase']='cache_and_packed_prepare';check();start=time.monotonic()
        scale=learner.PairScale(np.asarray(manifest()['scales']),np.ones(12,bool))
        cache=prepare_pair_cache(bank,scale);snapshot['cache']=cache;packed=prepare_marginal_packed(cache)
        report['packed_prepare_seconds']=time.monotonic()-start;host_before=core._fingerprint((packed.arrays,packed.identity))
        snapshot.update(packed=packed,packed_sha256=host_before)
        expected_refs=int((cache.factors['good']*np.maximum(cache.factors['usable'].sum(axis=2),1)[...,None]).sum())
        if len(packed.arrays['route_refs'])!=expected_refs:raise ValueError('All supported routes/base-only references required')
        report.update(packed_sha256=host_before,route_references=len(packed.arrays['route_refs']),
            complete_geometry_count=len(packed.arrays['geometry_components']),
            native_geometry_entries=len(packed.arrays['native_geometry_ids']),group_geometry_entries=len(packed.arrays['group_geometry_ids']),
            native_margin_entries=len(packed.arrays['native_margin_ids']),group_margin_entries=len(packed.arrays['group_margin_ids']),
            missing_feature_cells=int((~bank.evidence.feature_supported).sum()),
            native_supported_cells=int(packed.arrays['native_supported'].sum()),
            native_route_supported_cells=int(packed.arrays['native_route_supported'].sum()),
            group_supported_cells=int(packed.arrays['group_supported'].sum()),
            identity_sha256=core._fingerprint(packed.identity))
        report['phase']='device_upload';prepared,upload=timed(t,check,lambda:scorer.prepare_marginal_packed_torch(packed,device='cuda:0'))
        device_before=g.device_fingerprint(t,prepared);snapshot.update(device=prepared,device_sha256=device_before)
        report.update(upload_seconds=upload,device_tables_before=device_before,rows=[])
        theta=np.asarray(manifest()['theta']);tau=manifest()['temperature']
        for alpha in manifest()['alphas']:
            report.update(phase='gpu_full_score',current_alpha=alpha)
            a,seconds=timed(t,check,lambda:scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=tau,alpha=alpha))
            b,repeat_seconds=timed(t,check,lambda:scorer.score_pair_marginal_packed_torch(prepared,theta,temperature=tau,alpha=alpha))
            actual,repeat=g.arrays(a),g.arrays(b);report['phase']='cpu_packed_all_rows';check();start=time.monotonic()
            cpu=score_pair_marginal_packed(packed,theta,temperature=tau,alpha=alpha);cpu_seconds=time.monotonic()-start;check()
            flags=compare(np,g,actual,repeat,cpu)
            for result in (a,b):
                if (core._fingerprint(result.identity)!=core._fingerprint(cpu.identity)
                    or result.distribution_fingerprint!=cpu.distribution_fingerprint
                    or result.parameter_fingerprint!=cpu.parameter_fingerprint
                    or result.temperature!=cpu.temperature or result.alpha!=cpu.alpha or result.device!='cuda:0'):
                    raise ValueError('Original IDs/distribution/parameters/device differ')
            if alpha==0.:
                for x,y in (('native_scores_a','native_scores_b'),('scores_a','scores_b'),('geometry_derivatives_a','geometry_derivatives_b')):
                    g.bits(actual[x],actual[y])
                    if any(getattr(result,x).data_ptr()!=getattr(result,y).data_ptr() for result in (a,b)):
                        raise ValueError('Alpha-zero shared buffers required')
            report['rows'].append(dict(alpha=alpha,gpu_seconds=seconds,repeat_gpu_seconds=repeat_seconds,
                cpu_packed_seconds=cpu_seconds,result_sha256=core._fingerprint(actual),parameter_fingerprint=a.parameter_fingerprint,
                identity_sha256=core._fingerprint(a.identity),distribution_fingerprint=a.distribution_fingerprint,
                native_shape=list(actual['native_scores_a'].shape),group_shape=list(actual['scores_a'].shape),
                geometry_vjp_shape=list(actual['geometry_derivatives_a'].shape),alpha_vjp_shape=list(actual['alpha_derivatives_b'].shape),**flags))
        check();report['host_tables_after']=core._fingerprint((packed.arrays,packed.identity));report['device_tables_after']=g.device_fingerprint(t,prepared)
        if host_before!=report['host_tables_after'] or device_before!=report['device_tables_after'] or bank_before!=core._fingerprint(bank):
            raise ValueError('Complete source/device tables mutated')
        if next(iterator,None) is not None or len(sources)!=2 or not all(x['source_rehashed_after'] for x in sources):
            raise ValueError('Full original fixture sources not rehashed')
        prep=report['fixture_prepare_seconds']+report['packed_prepare_seconds']
        report.update(source_fixtures_rehashed_after=True,bank_rehashed_after=True,snapshots_rehashed_after=True,
            descriptive_projection=projection(prep,upload,[x[k] for x in report['rows'] for k in ('gpu_seconds','repeat_gpu_seconds')]),
            phase='complete',rows_completed=2)
    finally:
        iterator.close()
        report['source_fixtures_rehashed_after']=len(sources)==2 and all(x['source_rehashed_after'] for x in sources)
        recheck_snapshots(g,t,snapshots)
        report['bank_rehashed_after']=bool(snapshots)
        report['snapshots_rehashed_after']=bool(snapshots)


def native(code,rev,out,pin,deadline):
    g,rt,fixture=helpers(code);prior=rt.pinned(out/'proof.json',pin,1<<20);before={str(p):rt.identity(p,2<<20) for p in leaves(code)}
    rt.require(out==ROOT/RESULT and stat.S_IMODE(out.stat().st_mode)==0o755 and out.stat().st_uid==1000
        and os.geteuid()==1000 and sys.platform=='linux' and sys.version_info[:2]==(3,11)
        and os.environ.get('WR_IMAGE_ID')==IMAGE and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and {p.name for p in out.iterdir()}=={'proof.json','container.cid'},'Exact fresh native namespace required')
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',(out/'container.cid').read_bytes()),'Owned CID required');rt.identity(out/'container.cid',65,readonly=False)
    rt.require(prior['manifest']==manifest() and prior['source_binding']['producer_revision']==rev
        and all(before[str(code/n)]==v for n,v in prior['source_binding']['helpers'].items())
        and all(before[str(code.parent/n)]==v for n,v in prior['source_binding']['markers'].items())
        and (code.parent/'revision').read_bytes()==(rev+'\n').encode()
        and prior['tiny_qualification']['producer_revision']==PRIOR_REV
        and prior['tiny_qualification']['receipts']==PRIOR_PINS,'Current/prospective source/proof differs')
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Inclusive1200s complete-bank budget')
    def cancelled(*_):raise TimeoutError('Complete-bank control cancelled')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    report=dict(schema=SCHEMA,stage='coherent_pair_gpu_fullbank_native',status='fail',phase='imports',producer_revision=rev,
        source_binding=prior['source_binding'],manifest=manifest(),image_id=IMAGE,tiny_qualification=prior['tiny_qualification'],
        models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,fit_executed=False,optimizer_executed=False,
        quality_verified=False,adoption=False,full_fit_cost_qualified=False,source_rehashed_after=False,decision='CLOSED_FULLBANK_CONTROL')
    snapshots=[];t=None
    try:
        check();sys.path.insert(0,str(code/'src'));import numpy as np;import torch as t
        rt.require(t.__version__=='2.5.1+cu124' and t.version.cuda=='12.4' and t.cuda.is_available()
            and t.cuda.device_count()==1 and t.cuda.get_device_capability(0)==(9,0) and 'H100' in t.cuda.get_device_name(0),
            'Actual original B47 H100 runtime required')
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.use_deterministic_algorithms(True);t.set_num_threads(4)
        props=t.cuda.get_device_properties(0);t.cuda.set_per_process_memory_fraction(MEMORY/props.total_memory,0);t.cuda.reset_peak_memory_stats()
        report['runtime']=dict(python=sys.version.split()[0],numpy=np.__version__,torch=t.__version__,cuda=t.version.cuda,
            gpu_name=props.name,capability=[props.major,props.minor],gpu_total_bytes=props.total_memory,
            deterministic_algorithms=True,tf32=False,cublas_workspace_config=':4096:8')
        measure(np,t,g,fixture,check,report,snapshots);t.cuda.synchronize();check()
        report.update(peak_torch_allocated=t.cuda.max_memory_allocated(),peak_torch_reserved=t.cuda.max_memory_reserved(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        rt.require(report['peak_torch_allocated']<=MEMORY and report['peak_torch_reserved']<=MEMORY,'6GiB allocator gate')
        report.update(status='pass',decision=DECISION)
    except BaseException as exc:report.update(status='fail',error_type=g.error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if snapshots:
                recheck_snapshots(g,t,snapshots)
            rt.require(before=={str(p):rt.identity(p,2<<20) for p in leaves(code)} and rt.identity(out/'proof.json',1<<20)==pin,'Native source/proof changed')
            report['source_rehashed_after']=True
        except Exception as exc:report.update(status='fail',post_error_type=g.error_family(exc))
        publish(g,rt,out/'native.json',report,deadline)
    return report


def validate_native(rt,value,binding,rev,tiny):
    fixed=dict(schema=SCHEMA,stage='coherent_pair_gpu_fullbank_native',status='pass',phase='complete',producer_revision=rev,
        source_binding=binding,manifest=manifest(),image_id=IMAGE,tiny_qualification=tiny,counts=[4,3600,64],native_tokens=1500,
        candidate_rows=28800,tuple_rows=512,bridge_rows=230400,person_groups=4,object_groups=3600,
        original_frame_index=0,image_size=[720,960],rows_completed=2,source_fixtures_rehashed_after=True,
        bank_rehashed_after=True,snapshots_rehashed_after=True,source_rehashed_after=True,
        models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,fit_executed=False,optimizer_executed=False,
        quality_verified=False,adoption=False,full_fit_cost_qualified=False,decision=DECISION)
    rt.require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in fixed.items()),'Complete fixed full-bank PASS required')
    rt.require(value['packed_sha256']==value['host_tables_after'] and value['device_tables_before']==value['device_tables_after']
        and all(re.fullmatch('[0-9a-f]{64}',str(value[k])) for k in ('packed_sha256','device_tables_before','bank_sha256','full_original_observations_sha256','identity_sha256')),'Exact full-bank source hashes required')
    runtime=value.get('runtime',{});expected_runtime=dict(torch='2.5.1+cu124',cuda='12.4',capability=[9,0],
        deterministic_algorithms=True,tf32=False,cublas_workspace_config=':4096:8')
    rt.require(all(type(runtime.get(k)) is type(v) and runtime[k]==v for k,v in expected_runtime.items())
        and type(runtime.get('gpu_name')) is str and 'H100' in runtime['gpu_name']
        and type(runtime.get('gpu_total_bytes')) is int and runtime['gpu_total_bytes']>MEMORY
        and type(runtime.get('python')) is str and re.fullmatch(r'3\.11\.[0-9]+',runtime['python'])
        and type(runtime.get('numpy')) is str and runtime['numpy'],'Actual original FP64 GPU runtime required')
    rt.require(all(type(value.get(k)) is int and 0<value[k]<=28800*64 for k in ('route_references','complete_geometry_count',
        'native_geometry_entries','group_geometry_entries','native_margin_entries','group_margin_entries'))
        and type(value.get('missing_feature_cells')) is int and 0<value['missing_feature_cells']<=28800*10
        and type(value.get('max_rss_bytes')) is int and value['max_rss_bytes']>0
        and all(type(value.get(k)) is int and 0<value[k]<=limit for k,limit in
            (('native_supported_cells',28800),('native_route_supported_cells',28800),('group_supported_cells',14400)))
        and value['native_route_supported_cells']<=value['native_supported_cells'],'Complete source/incidence/support census required')
    rt.require([r['alpha'] for r in value['rows']]==[0.,.3125],'Both fixed alpha rows required')
    for row in value['rows']:
        rt.require(row['repeat_bits_exact'] is True and row['all_cpu_packed_arrays_passed'] is True
            and row['native_shape']==[4,2,3600] and row['group_shape']==[4,3600]
            and row['geometry_vjp_shape']==[4,3600,17] and row['alpha_vjp_shape']==[4,3600]
            and row['identity_sha256']==value['identity_sha256'] and row['distribution_fingerprint']==value['packed_sha256']
            and row['supported_pairs']==value['group_supported_cells'] and row['native_supported']==value['native_supported_cells']
            and all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0 for k in ('gpu_seconds','repeat_gpu_seconds','cpu_packed_seconds'))
            and all(re.fullmatch('[0-9a-f]{64}',str(row[k])) for k in ('result_sha256','parameter_fingerprint')),'All native score/VJP rows and timings required')
    rt.require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k]>0 for k in ('fixture_prepare_seconds','packed_prepare_seconds','upload_seconds')),'Separate full preparation/upload costs required')
    expected=projection(value['fixture_prepare_seconds']+value['packed_prepare_seconds'],value['upload_seconds'],
        [r[k] for r in value['rows'] for k in ('gpu_seconds','repeat_gpu_seconds')])
    rt.require(value['descriptive_projection']==expected and all(type(value[k]) is int and 0<=value[k]<=MEMORY for k in ('peak_torch_allocated','peak_torch_reserved')),'Projection/memory scope differs')


def publish(g,rt,path,value,deadline,*,seal=False):
    """Pinned generic same-FD recipe with this caller's failure decision only."""
    if value['status']!='pass':value['decision']='CLOSED_FULLBANK_CONTROL'
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
            value.update(status='fail',error_type=g.error_family(exc),decision='CLOSED_FULLBANK_CONTROL');update()


def host(code,rev):
    started=time.monotonic();deadline=started+BUDGET;g,rt,_=helpers(code)
    def cancelled(*_):raise TimeoutError('Complete-bank host deadline')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,BUDGET);before=proof(g,rt,code,rev);out=rt.canonical(ROOT/RESULT)
    rt.require(not out.exists(),'Fresh full-bank namespace required');out.mkdir(mode=0o755);os.chown(out,1000,1000);out.chmod(0o755)
    inode=(out.stat().st_dev,out.stat().st_ino);rt.write(out/'proof.json',(json.dumps(before,sort_keys=True)+'\n').encode(),0o444)
    pin=rt.identity(out/'proof.json',1<<20);cid=out/'container.cid';name='world-reward-fullbank-'+rev[:12];owned=False;removed=False
    report=dict(schema=SCHEMA,stage='coherent_pair_gpu_fullbank_host',status='fail',producer_revision=rev,
        source_binding=before['source_binding'],manifest=manifest(),image=before['image'],tiny_qualification=before['tiny_qualification'],decision='CLOSED_FULLBANK_CONTROL')
    try:
        rt.require(not g.control(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).strip(),'Owned name exists');owned=True
        argv=['docker','run','--rm','--cidfile',str(cid),'--name',name,'--label','world_reward.tiny_gpu.owner='+rev,
            '--gpus','all','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--memory','6g','--cpus','4','--pids-limit','256','--user','1000:1000','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=128m','--entrypoint','/usr/bin/env']
        for p in leaves(code):argv+=['--mount',f'type=bind,src={p},dst={p},readonly']
        argv+=['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
            'PYTHONDONTWRITEBYTECODE=1','WR_IMAGE_ID='+IMAGE,'CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'python','-I','-B',str(code/HELPERS[0]),'native',str(code),rev,str(out),str(pin['bytes']),pin['sha256'],str(deadline)]
        r=subprocess.run(argv,capture_output=True,timeout=max(.001,deadline-time.monotonic()),check=False)
        report.update(native_exit_code=r.returncode,diagnostics={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in (('stdout',r.stdout),('stderr',r.stderr))})
        rt.require(len(r.stdout)<=32768 and len(r.stderr)<=32768,'Bounded native diagnostics required')
        removed=g.cleanup(rt,cid,name,rev);value=rt.strict((out/'native.json').read_bytes());rt.require(r.returncode==0,'Full-bank process failed')
        validate_native(rt,value,before['source_binding'],rev,before['tiny_qualification']);report.update(status='pass',native_report_identity=rt.identity(out/'native.json',1<<20),decision=value['decision'])
    except BaseException as exc:report.update(status='fail',error_type=g.error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:removed=g.cleanup(rt,cid,name,rev) if owned else False
        except Exception as exc:report.update(status='fail',cleanup_error_type=g.error_family(exc))
        try:report['source_rehashed_after']=proof(g,rt,code,rev)==before
        except Exception as exc:report.update(status='fail',post_error_type=g.error_family(exc))
        rt.require((out.stat().st_dev,out.stat().st_ino)==inode and out.stat().st_uid==1000,'Owned output replaced')
        rt.require({p.name for p in out.iterdir()}<={'proof.json','container.cid','native.json'},'Foreign output artifact')
        report.update(owned_container_removed=removed,elapsed_seconds=time.monotonic()-started)
        if not removed or not report.get('source_rehashed_after') or time.monotonic()>=deadline:report['status']='fail'
        if report['status']!='pass':report['decision']='CLOSED_FULLBANK_CONTROL'
        publish(g,rt,out/'report.json',report,deadline,seal=True)
    return report


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('mode',choices=('host','native'));p.add_argument('code');p.add_argument('revision');p.add_argument('native_args',nargs='*');a=p.parse_args();code=Path(a.code)
    if not re.fullmatch('[0-9a-f]{40}',a.revision) or code!=ROOT/'jobs'/a.revision/ENTRY/'code' or Path(__file__).resolve()!=code/HELPERS[0]:raise ValueError('Exact new immutable source required')
    if a.mode=='host':
        if a.native_args or os.geteuid()!=0 or sys.platform!='linux' or os.uname().nodename!='scenesmith-ncc-h100-01' or os.environ.get('DOCKER_HOST')!='unix://'+str(ROOT/'docker.sock'):raise ValueError('Actual VM01 root GPU host required')
        node=ROOT/'jobs/.world-reward-h100.lock';s=node.lstat();fd=os.fstat(9)
        if node.is_symlink() or not stat.S_ISREG(s.st_mode) or (s.st_dev,s.st_ino)!=(fd.st_dev,fd.st_ino):raise ValueError('Actual FD9 lease required')
        value=host(code,a.revision)
    else:
        if len(a.native_args)!=4:raise ValueError('Exact shared proof/deadline arguments required')
        out,size,digest,end=a.native_args;deadline=float(end)
        if not math.isfinite(deadline) or not 0<deadline-time.monotonic()<=BUDGET:raise ValueError('Inclusive finite shared deadline required')
        value=native(code,a.revision,Path(out),dict(bytes=int(size),sha256=digest),deadline)
    print(json.dumps(dict(status=value['status'],decision=value['decision'],error_type=value.get('error_type')),sort_keys=True))
    if value['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()

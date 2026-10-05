"""One complete procedural objective/VJP/short-solver control, never real FIT."""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
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

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_coherent_pair_gpu_objective_probe'
RESULT='results/coherent-pair-gpu-objective-probe-v3'
SCHEMA='world_reward.coherent_pair_gpu_objective_probe.v1'
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET,MEMORY=1200,6*1024**3
DECISION='FULLBANK_OBJECTIVE_OPTIMIZER_CONTROL_PASS_NOT_FIT_QUALITY'
PRIOR_REV='504539cd25eb7a0630e79966e89bb542d5feb88d'
PRIOR_CLOSURE='6951381662d652ca87878275b20be3c0bf4f3155a69178c21d06634458650e07'
PRIOR_PINS={
 'proof.json':dict(bytes=10321,sha256='5ae8e18e0a21a21312038a2b46e51da0880f7edc483acb3be4b3882168a18547'),
 'native.json':dict(bytes=11011,sha256='ef6f9ff654347e351e5f83d92048938761822eb6428829b68be31d40f0aa2074'),
 'report.json':dict(bytes=11048,sha256='461388f5fc79855f18df47f607951156ff987f6e22396549ce0b635d366721f9')}
FULLBANK_PIN=(30617,'2a4abd0e871646de32ee00ea847f2663b159f1e497279e538a21d5e8d4521220')
OBJECTIVE_PIN=(13862,'2542534221708a9a3146aa5ff25306b3957a0a101bdb98a181bac05ab10855c1')
FAILED_REV='7281db2a3f7c7298a0d2a22e6ad7e7036fbe4e18'
FAILED_CLOSURE='b31a3d3b97215c33b3f2db5d34b15ae0aded11774409bd328159e0a2059fa6a0'
FAILED_PINS={
 'proof.json':dict(bytes=11121,sha256='f93cdc4db34a4a220e7424aa489007f076a7b99eb66ca0f0bc06f5b357180b92'),
 'native.json':dict(bytes=9351,sha256='ce79dae97060d8044cb9065470088bfa53fe9271d390d6e5523f9338d984fcf6'),
 'report.json':dict(bytes=11725,sha256='2b6de394c25e0ed93167a540fc7a7be21b806555349f81ec8c613108695dcdd5')}
METADATA_FAILED_REV='8db0500a09589dd288ecd8494f00ad33f61983ec'
METADATA_FAILED_CLOSURE='cfb613f47e946704d1dbec0d93552bee63ec802bc378fd1811b209c50f2128ac'
METADATA_FAILED_PINS={
 'proof.json':dict(bytes=15057,sha256='662824ad5edbbfd967ecff70d9ac2b31a0722b44c856ef3609fd43029e53c73b'),
 'native.json':dict(bytes=14463,sha256='217666a8bd5b168389d6d16657a3acd48d6685007edbf74e934f036d5f52af95'),
 'report.json':dict(bytes=15663,sha256='04de0a2f833f408d9ec40d881f206dddae0f30278ba06d58dd729bfcb6faaeb3')}
METADATA_AUDIT=dict(
 reference='results/audits/coherent_pair_scipy_native_inventory_v3_actual.json',
 identity=dict(bytes=4601,sha256='581aa1e6f80ac3e1acfb6a57fac22abadf83e18ee3370994c9747013eb2d6629'),
 declaration_only=True,host_or_child_live_read=False,solver_qualification=False,image_id=IMAGE,
 rootfs_sha256='5ae610524943c42f72e2fa1ac06a9dbe8fa14bf2d572efd8a57cded6ead3503d',
 record=dict(bytes=196853,sha256='c34263ef30380bff9837a2311899667249d27ea73c6a6f226209e762582f38a5'),
 observed_version='1.16.3',examined_rows=2381,claimed_empty_files=41,unclaimed_nonempty_pyc_files=961,
 requested_empty_hash_size_verified=True,all_populated_claims_verified=True)
HELPERS=('infra/coherent_pair_gpu_objective_probe.py','infra/run_coherent_pair_gpu_objective_probe.sh',
 'infra/coherent_pair_gpu_fullbank_cost.py','infra/run_coherent_pair_gpu_fullbank_cost.sh',
 'infra/coherent_pair_gpu_probe.py','infra/coherent_pair_cost_probe.py','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/__init__.py','src/world_reward/coherent_pair_learning.py','src/world_reward/coherent_route_scorer.py',
 'src/world_reward/coherent_pair_cache.py','src/world_reward/coherent_pair_packed.py','src/world_reward/coherent_pair_packed_score.py',
 'src/world_reward/coherent_pair_marginal.py','src/world_reward/coherent_pair_packed_torch.py',
 'src/world_reward/coherent_pair_marginal_objective.py','src/world_reward/interaction_candidate_evidence.py',
 'src/world_reward/interaction_tuple_evidence.py','src/world_reward/person_pose_observations.py','src/world_reward/hoi_detr_observations.py')


def helpers(code):
    spec=importlib.util.spec_from_file_location('objective_fullbank',code/HELPERS[2])
    full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
    g,rt,fixture=full.helpers(code);return full,g,rt,fixture


def manifest():
    return dict(schema=SCHEMA,persons=4,objects=3600,native_pairs=64,native_tokens=1500,temperature=.8125,
        theta=[(-1 if i%2 else 1)*(i+1)/64. for i in range(17)],scales=[(i+3)/8. for i in range(12)],
        alphas=[0.,.3125],regularization=1.,positive_group_rule='first_person_first_object_and_last_person_last_object',
        oracle_rtol=1e-12,oracle_atol=1e-12,fd_step=1e-6,fd_rtol=1e-7,fd_atol=1e-7,fd_calls=72,
        scipy_version='1.16.3',solver='L-BFGS-B',solver_options=dict(maxiter=20,maxfun=100,maxls=20,maxcor=10,ftol=1e-12,gtol=1e-6),
        projected_stationarity=1e-6,budget_seconds=BUDGET,memory_bytes=MEMORY,image_id=IMAGE,
        real_fit_executed=False,quality_verified=False,adoption=False,global_optimum_claimed=False)


def prior_evidence(full,rt):
    out=ROOT/'results/coherent-pair-gpu-fullbank-cost-v1'
    proof,native,host=(rt.pinned(out/n,PRIOR_PINS[n],1<<20) for n in ('proof.json','native.json','report.json'))
    binding=proof['source_binding'];old=ROOT/'jobs'/PRIOR_REV/full.ENTRY/'code'
    rt.require(binding['producer_revision']==PRIOR_REV and binding['entries']==312 and binding['closure_sha256']==PRIOR_CLOSURE
        and rt.source(ROOT,old,PRIOR_REV,full.ENTRY,full.HELPERS)==binding,'Complete original full-bank source required')
    full.validate_native(rt,native,binding,PRIOR_REV,proof['tiny_qualification'])
    rt.require(host['status']=='pass' and host['source_binding']==binding and host['manifest']==full.manifest()
        and host['image']==proof['image'] and host['image']['Id']==IMAGE and host['native_report_identity']==PRIOR_PINS['native.json']
        and host['native_exit_code']==0 and host['source_rehashed_after'] is True and host['owned_container_removed'] is True,
        'Original sealed full-bank PASS required')
    return dict(producer_revision=PRIOR_REV,receipts=PRIOR_PINS,source_binding=binding,host_only_authentication=True)


def technical_resumption(rt):
    """Host-only failed-source authentication, never a prior numerical PASS."""
    old=ROOT/'jobs'/FAILED_REV/ENTRY/'code';out=ROOT/'results/coherent-pair-gpu-objective-probe-v1'
    binding=rt.source(ROOT,old,FAILED_REV,ENTRY,HELPERS)
    rt.require(binding['entries']==316 and binding['closure_sha256']==FAILED_CLOSURE
        and (old.parent/'source-sha256').read_bytes()==b'82b790318f5ecd4ccb0f7a528dd8fe1686c83de3152a9fd4aaf29bc7a1524f63\n',
        'Immutable failed producer source required')
    proof,native,host=(rt.pinned(out/n,FAILED_PINS[n],1<<20) for n in ('proof.json','native.json','report.json'))
    rt.require(proof['source_binding']==native['source_binding']==host['source_binding']==binding
        and proof['manifest']==native['manifest']==host['manifest']==manifest()
        and native['status']==host['status']=='fail' and native['phase']=='objective_oracle'
        and native['error_type']=='OtherError' and native['fd_calls']==0 and native['controls']==[]
        and host['native_exit_code']==1 and host['source_rehashed_after'] is True
        and host['owned_container_removed'] is True and host['decision']==native['decision']=='CLOSED_OBJECTIVE_CONTROL',
        'Original closed failure required, not a prior qualification')
    return dict(producer_revision=FAILED_REV,receipts=FAILED_PINS,source_binding=binding,
        host_only_authentication=True,previous_status='fail',previous_phase='objective_oracle',
        correction='scalar_gradient_boolean_selection_before_reshape',scientific_recipe_unchanged=True,
        previous_failure_not_converted_to_pass=True)


def metadata_resumption(rt):
    """Live host authentication of V2 FAIL; saved diagnostic summary is declarative."""
    old=ROOT/'jobs'/METADATA_FAILED_REV/ENTRY/'code';out=ROOT/'results/coherent-pair-gpu-objective-probe-v2'
    binding=rt.source(ROOT,old,METADATA_FAILED_REV,ENTRY,HELPERS)
    rt.require(binding['entries']==317 and binding['closure_sha256']==METADATA_FAILED_CLOSURE
        and (old.parent/'source-sha256').read_bytes()==b'6eb181f08240eaebcf851fe70425095043250a4d5a813477dd0dece83d21df37\n',
        'Immutable V2 failed producer source required')
    proof,native,host=(rt.pinned(out/n,METADATA_FAILED_PINS[n],1<<20) for n in ('proof.json','native.json','report.json'))
    rt.require(proof['source_binding']==native['source_binding']==host['source_binding']==binding
        and proof['manifest']==native['manifest']==host['manifest']==manifest()
        and native['status']==host['status']=='fail' and native['phase']=='scipy_native_inventory'
        and native['error_type']=='ValueError' and native['fd_calls']==72 and len(native['controls'])==2
        and native.get('solvers',[])==[] and host['native_exit_code']==1 and host['source_rehashed_after'] is True
        and host['owned_container_removed'] is True and host['decision']==native['decision']=='CLOSED_OBJECTIVE_CONTROL',
        'Original V2 metadata failure required, not a solver qualification')
    return dict(producer_revision=METADATA_FAILED_REV,receipts=METADATA_FAILED_PINS,source_binding=binding,
        host_only_authentication=True,previous_status='fail',previous_phase='scipy_native_inventory',
        previous_failure_not_converted_to_pass=True,scientific_recipe_unchanged=True,
        correction='accept_zero_bytes_only_with_explicit_record_sha256_empty_and_size0',
        independent_saved_audit_declaration=METADATA_AUDIT)


def proof(full,g,rt,code,rev):
    source=rt.source(ROOT,code,rev,ENTRY,HELPERS)
    frozen=dict(full.REUSED);frozen.update({HELPERS[2]:FULLBANK_PIN,'src/world_reward/coherent_pair_marginal_objective.py':OBJECTIVE_PIN})
    rt.require(all(source['helpers'][n]==dict(bytes=b,sha256=s) for n,(b,s) in frozen.items()),'Frozen math/objective/runtime source required')
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact current code namespace required')
    return dict(source_binding=source,manifest=manifest(),image=g.image(),fullbank_qualification=prior_evidence(full,rt),
        technical_resumption=technical_resumption(rt),metadata_resumption=metadata_resumption(rt))


def leaves(code):return [*(code/n for n in HELPERS),code.parent/'revision',code.parent/'source-sha256']


def scipy_evidence(rt):
    """First native image-anchored RECORD census, not a previously certified SHA."""
    d=importlib.metadata.distribution('scipy');rt.require(d.version=='1.16.3','Fixed native SciPy1.16.3 required')
    files=list(d.files or ());records=[p for p in files if str(p).endswith('.dist-info/RECORD')]
    rt.require(len(records)==1,'One original SciPy RECORD required');record=rt.canonical(d.locate_file(records[0]))
    root=record.parent.parent;rp=rt.identity(record,2<<20,readonly=False);rows=list(csv.reader(io.StringIO(record.read_text())));ledger={}
    rt.require(0<len(rows)<=20000,'Bounded native distribution census required')
    empty_claim='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b'').digest()).decode().rstrip('=')
    for row in rows:
        rt.require(len(row)==3 and row[0] and row[0] not in ledger,'Unique native RECORD row required')
        path=rt.canonical(d.locate_file(row[0]));rt.require(path.is_relative_to(root),'Contained native distribution file required')
        pin=rt.identity(path,200<<20,readonly=False,empty=row[1:]==[empty_claim,'0'])
        if path==record:rt.require(row[1:]==['',''],'RECORD self-reference required')
        elif row[1:]==['','']:
            rt.require(path.parent.name=='__pycache__' and path.suffix=='.pyc','Only image-anchored native bytecode caches may omit RECORD claims')
        else:
            rt.require(re.fullmatch(r'sha256=[A-Za-z0-9_-]{43}',row[1]) and row[2].isdigit()
                and int(row[2])==pin['bytes'] and base64.urlsafe_b64encode(bytes.fromhex(pin['sha256'])).decode().rstrip('=')==row[1][7:],
                'Original installed SciPy RECORD hash/size differs')
        ledger[row[0]]=pin
    notices=[n for n in ledger if n.startswith(str(records[0].parent)+'/') and re.search(r'(^|/)(LICENSE|LICENSE.txt|LICENSES_bundled.txt)$',n)]
    rt.require(notices and any(b'Redistribution and use' in d.locate_file(n).read_bytes() for n in notices),'Retained original BSD license notice required')
    rt.require(rt.identity(record,2<<20,readonly=False)==rp,'Native RECORD changed')
    return dict(version=d.version,record=rp,entries=len(ledger),license_files={n:ledger[n] for n in notices},
        source_fingerprint=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest(),first_native_record_census=True,
        populated_record_claims_verified=True,unclaimed_cache_files=sum(r[1:]==['',''] and not r[0].endswith('/RECORD') for r in rows),
        caches_image_anchored_not_record_certified=True,
        claimed_empty_files=sum(p['bytes']==0 for p in ledger.values()),
        empty_file_policy='explicit_original_record_sha256_empty_and_exact_size0_only')


def solve(np,evaluate,minimize,theta,check,report):
    """One fixed zero-start A then bounded scalar B; analytic callbacks only."""
    rows=[];frozen=theta.copy()
    for arm in ('A','B'):
        start=np.zeros(17 if arm=='A' else 1);calls=0
        def function(x):
            nonlocal calls
            check();calls+=1
            if calls>manifest()['solver_options']['maxfun']:raise RuntimeError('Fixed objective evaluation cap')
            value=evaluate(x if arm=='A' else frozen,0. if arm=='A' else float(x[0]),arm)
            return value
        initial,initial_gradient=function(start);begin=time.monotonic()
        result=minimize(function,start,method='L-BFGS-B',jac=True,bounds=None if arm=='A' else [(0.,None)],
            options=manifest()['solver_options'])
        check();value,gradient=function(result.x)
        projected=gradient.copy()
        if arm=='B' and result.x[0]==0.:projected[0]=min(projected[0],0.)
        norm=float(np.abs(projected).max());ok=bool(result.success) and norm<=1e-6 and value<=initial
        if not np.isfinite(result.x).all() or not math.isfinite(value) or not np.isfinite(gradient).all():raise ValueError('Finite short solver result required')
        row=dict(arm=arm,success=bool(result.success),status=int(result.status),iterations=int(result.nit),
            evaluations=calls,initial_loss=initial,final_loss=value,projected_gradient_inf=norm,seconds=time.monotonic()-begin,
            parameters=list(map(float,result.x)),control_passed=ok)
        rows.append(row);report['solvers']=rows
        if not ok:raise RuntimeError('Fixed short solver did not qualify; no restart')
        if arm=='A':frozen=np.array(result.x,copy=True)
    return rows


def measure(np,t,full,g,rt,fixture,check,report,snapshots,*,object_count=3600):
    from world_reward import coherent_route_scorer as core,coherent_pair_packed_torch as scorer
    from world_reward.coherent_pair_learning import PairScale
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    from world_reward.coherent_pair_marginal_objective import marginal_objective,marginal_objective_torch
    sources=[];iterator=fixture.fixtures(np,object_count=object_count,source_records=sources)
    try:
        report['phase']='fullbank_prepare';check();begin=time.monotonic();discard=next(iterator);del discard;bank=next(iterator)
        snapshot=dict(bank=bank,bank_sha256=core._fingerprint(bank));snapshots.append(snapshot)
        scale=PairScale(np.asarray(manifest()['scales']),np.ones(12,bool));cache=prepare_pair_cache(bank,scale);snapshot['cache']=cache
        packed=prepare_marginal_packed(cache);snapshot.update(packed=packed,packed_sha256=core._fingerprint((packed.arrays,packed.identity)))
        if cache.counts!=(4,object_count,64):raise ValueError('Fixed all-slot bank required')
        expected=int((cache.factors['good']*np.maximum(cache.factors['usable'].sum(axis=2),1)[...,None]).sum())
        if len(packed.arrays['route_refs'])!=expected:raise ValueError('Complete supported route/base-only incidence required')
        report.update(counts=list(cache.counts),candidate_rows=4*2*object_count,tuple_rows=512,bridge_rows=64*object_count,
            native_tokens=sources[-1]['native_tokens'],route_references=len(packed.arrays['route_refs']),
            group_shape=list(packed.arrays['group_supported'].shape),source_sha256=snapshot['bank_sha256'],packed_sha256=snapshot['packed_sha256'],
            identity_sha256=core._fingerprint(packed.identity),native_supported_cells=int(packed.arrays['native_supported'].sum()),
            group_supported_cells=int(packed.arrays['group_supported'].sum()),
            full_original_observations_sha256=sources[-1]['full_original_observations_sha256'],prepare_seconds=time.monotonic()-begin)
        device,seconds=full.timed(t,check,lambda:scorer.prepare_marginal_packed_torch(packed,device='cuda:0'))
        snapshot.update(device=device,device_sha256=g.device_fingerprint(t,device));report['upload_seconds']=seconds
        mask=np.zeros(packed.arrays['group_supported'].shape,bool);mask[0,0]=mask[-1,-1]=True
        theta=np.asarray(manifest()['theta']);tau=manifest()['temperature'];report['controls']=[];report['fd_calls']=0
        mask_before=core._fingerprint(mask);report['positive_groups']=int(mask.sum())
        def evaluate(coeff,alpha,arm):
            score=scorer.score_pair_marginal_packed_torch(device,coeff,temperature=tau,alpha=alpha)
            value=marginal_objective_torch((score,),(mask,),coeff,alpha=alpha,regularization=1.,arm=arm)
            if value.status!='complete' or value.counts['used']!=1:raise ValueError('Manufactured complete informative objective required')
            return value,score
        def numeric(coeff,alpha,arm):
            check();value,_=evaluate(coeff,alpha,arm);t.cuda.synchronize();check()
            return float(value.loss.cpu().item()),value.gradient.cpu().numpy().copy()
        for alpha in (0.,.3125):
            report.update(phase='objective_oracle',current_alpha=alpha);begin=time.monotonic()
            result,score=evaluate(theta,alpha,'A');repeat,repeat_score=evaluate(theta,alpha,'A')
            b=marginal_objective_torch((score,),(mask,),theta,alpha=alpha,regularization=1.,arm='B')
            repeat_b=marginal_objective_torch((repeat_score,),(mask,),theta,alpha=alpha,regularization=1.,arm='B')
            for a,z in ((result,repeat),(b,repeat_b)):
                for x,y in ((a.loss,z.loss),(a.gradient,z.gradient),(a.record_losses,z.record_losses)):
                    g.bits(x.cpu().numpy(),y.cpu().numpy())
            cpu_score=score_pair_marginal_packed(packed,theta,temperature=tau,alpha=alpha)
            ca=marginal_objective((cpu_score,),(mask,),theta,alpha=alpha,regularization=1.,arm='A')
            cb=marginal_objective((cpu_score,),(mask,),theta,alpha=alpha,regularization=1.,arm='B')
            for gpu,cpu in ((result,ca),(b,cb)):
                np.testing.assert_allclose(gpu.loss.cpu().numpy(),cpu.loss,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(gpu.gradient.cpu().numpy(),cpu.gradient,rtol=1e-12,atol=1e-12)
                if gpu.counts!=cpu.counts or gpu.record_statuses!=cpu.record_statuses:raise ValueError('Complete objective coverage differs')
            for current in (score,repeat_score):
                if core._fingerprint(current.identity)!=core._fingerprint(cpu_score.identity) or current.distribution_fingerprint!=packed.packed_fingerprint:
                    raise ValueError('Full original score identity differs')
                for name in g.BOOL_FIELDS:g.bits(getattr(current,name).cpu().numpy(),getattr(cpu_score,name))
            gradient=result.gradient.cpu().numpy();scalar=float(b.gradient.cpu().numpy()[0]);step=1e-6
            report['phase']='objective_fd'
            for j in range(17):
                delta=np.zeros(17);delta[j]=step
                hi,_=numeric(theta+delta,alpha,'A');lo,_=numeric(theta-delta,alpha,'A');report['fd_calls']+=2
                np.testing.assert_allclose((hi-lo)/(2*step),gradient[j],rtol=1e-7,atol=1e-7)
            if alpha==0.:
                hi,_=numeric(theta,step,'B');hi2,_=numeric(theta,2*step,'B');base=float(b.loss.cpu().numpy())
                fd=(-3*base+4*hi-hi2)/(2*step)
            else:
                hi,_=numeric(theta,alpha+step,'B');lo,_=numeric(theta,alpha-step,'B');fd=(hi-lo)/(2*step)
            report['fd_calls']+=2;np.testing.assert_allclose(fd,scalar,rtol=1e-7,atol=1e-7)
            report['controls'].append(dict(alpha=alpha,repeat_bits_exact=True,cpu_packed_objective_passed=True,fd17_and_alpha_passed=True,
                loss_a=float(result.loss.cpu().numpy()),loss_b=float(b.loss.cpu().numpy()),seconds=time.monotonic()-begin,
                gradient_shape=list(gradient.shape),scalar_gradient_shape=[1],parameter_fingerprint=score.parameter_fingerprint,
                identity_sha256=core._fingerprint(score.identity),distribution_fingerprint=score.distribution_fingerprint,
                group_shape=list(score.supported.shape),native_shape=list(score.native_supported.shape)))
        report['phase']='scipy_native_inventory';report['scipy_before']=scipy_evidence(rt)
        from scipy.optimize import minimize
        report['phase']='short_solver';solve(np,numeric,minimize,theta,check,report)
        if core._fingerprint(mask)!=mask_before:raise ValueError('Manufactured fixed positive mask changed')
        if next(iterator,None) is not None:raise ValueError('Exactly two original source fixtures required')
        report.update(phase='complete',controls_completed=2,solvers_completed=2)
    finally:
        iterator.close();report['source_fixtures_rehashed_after']=len(sources)==2 and all(x['source_rehashed_after'] for x in sources)
        if 'mask_before' in locals() and core._fingerprint(mask)!=mask_before:raise ValueError('Fixed positive mask changed on failure')
        full.recheck_snapshots(g,t,snapshots);report['snapshots_rehashed_after']=bool(snapshots)


def native(code,rev,out,pin,deadline):
    full,g,rt,fixture=helpers(code);before={str(p):rt.identity(p,2<<20) for p in leaves(code)};prior=rt.pinned(out/'proof.json',pin,1<<20)
    rt.require(out==ROOT/RESULT and stat.S_IMODE(out.stat().st_mode)==0o755 and out.stat().st_uid==1000 and os.geteuid()==1000
        and sys.platform=='linux' and sys.version_info[:2]==(3,11) and os.environ.get('WR_IMAGE_ID')==IMAGE
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and {p.name for p in out.iterdir()}=={'proof.json','container.cid'},'Exact owned native namespace required')
    rt.identity(out/'container.cid',65,readonly=False);rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',(out/'container.cid').read_bytes()),'Exact owned CID required')
    rt.require(prior['manifest']==manifest() and prior['source_binding']['producer_revision']==rev
        and all(before[str(code/n)]==v for n,v in prior['source_binding']['helpers'].items())
        and all(before[str(code.parent/n)]==v for n,v in prior['source_binding']['markers'].items())
        and prior['fullbank_qualification']['producer_revision']==PRIOR_REV and prior['fullbank_qualification']['receipts']==PRIOR_PINS,
        'Exact current source and sealed prior declarations required')
    rt.require(prior['technical_resumption']['producer_revision']==FAILED_REV
        and prior['technical_resumption']['receipts']==FAILED_PINS
        and prior['technical_resumption']['previous_failure_not_converted_to_pass'] is True,
        'Explicit immutable closed-failure resumption declarations required')
    resumed=prior['metadata_resumption']
    rt.require(resumed['producer_revision']==METADATA_FAILED_REV and resumed['receipts']==METADATA_FAILED_PINS
        and resumed['previous_status']=='fail' and resumed['previous_phase']=='scipy_native_inventory'
        and resumed['previous_failure_not_converted_to_pass'] is True and resumed['host_only_authentication'] is True
        and resumed['independent_saved_audit_declaration']==METADATA_AUDIT,
        'Explicit host-authenticated V2 failure and declarative metadata evidence required')
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Inclusive objective1200s budget')
    def cancelled(*_):raise TimeoutError('Objective control cancelled')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    report=dict(schema=SCHEMA,stage='coherent_pair_gpu_objective_native',status='fail',phase='imports',producer_revision=rev,
        source_binding=prior['source_binding'],manifest=manifest(),image_id=IMAGE,fullbank_qualification=prior['fullbank_qualification'],
        real_fit_executed=False,models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,quality_verified=False,
        adoption=False,global_optimum_claimed=False,technical_resumption=prior['technical_resumption'],
        metadata_resumption=prior['metadata_resumption'],
        decision='CLOSED_OBJECTIVE_CONTROL');snapshots=[];t=None
    try:
        check();sys.path.insert(0,str(code/'src'));import numpy as np;import torch as t
        rt.require(np.__version__=='1.26.3' and t.__version__=='2.5.1+cu124' and t.version.cuda=='12.4' and t.cuda.is_available() and t.cuda.device_count()==1
            and t.cuda.get_device_capability(0)==(9,0) and 'H100' in t.cuda.get_device_name(0),'Actual original H100/B47 required')
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.use_deterministic_algorithms(True);t.set_num_threads(4)
        props=t.cuda.get_device_properties(0);t.cuda.set_per_process_memory_fraction(MEMORY/props.total_memory,0);t.cuda.reset_peak_memory_stats()
        report['runtime']=dict(torch=t.__version__,cuda=t.version.cuda,python=sys.version.split()[0],numpy=np.__version__,
            gpu_name=props.name,gpu_total_bytes=props.total_memory,capability=[props.major,props.minor],tf32=False,deterministic_algorithms=True,
            cublas_workspace_config=':4096:8')
        measure(np,t,full,g,rt,fixture,check,report,snapshots);t.cuda.synchronize();check()
        report.update(peak_torch_allocated=t.cuda.max_memory_allocated(),peak_torch_reserved=t.cuda.max_memory_reserved(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        rt.require(report['peak_torch_allocated']<=MEMORY and report['peak_torch_reserved']<=MEMORY
            and 0<report['max_rss_bytes']<=MEMORY,'Fixed6GiB allocator/host RSS required')
        report.update(status='pass',decision=DECISION)
    except BaseException as exc:report.update(status='fail',error_type=g.error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            full.recheck_snapshots(g,t,snapshots)
            if 'scipy_before' in report:
                report['scipy_after']=scipy_evidence(rt);rt.require(report['scipy_before']==report['scipy_after'],'Native SciPy changed')
            rt.require(before=={str(p):rt.identity(p,2<<20) for p in leaves(code)} and rt.identity(out/'proof.json',1<<20)==pin,'Native source/proof changed')
            report['source_rehashed_after']=True
        except Exception as exc:report.update(status='fail',post_error_type=g.error_family(exc))
        publish(g,rt,out/'native.json',report,deadline)
    return report


def publish(g,rt,path,report,deadline,*,seal=False):
    """Existing same-FD lifecycle, scoped decision without global mutation."""
    if report['status']!='pass':report['decision']='CLOSED_OBJECTIVE_CONTROL'
    def raw():return (json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        os.fchmod(f.fileno(),0o400)
        def update():f.seek(0);f.write(raw());f.truncate();f.flush();os.fsync(f.fileno())
        try:
            update()
            if seal:path.parent.chmod(0o500)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            rt.require(rt.identity(path,1<<20)==dict(bytes=len(raw()),sha256=hashlib.sha256(raw()).hexdigest()),'Receipt differs')
            if time.monotonic()>=deadline:raise TimeoutError('Inclusive publication budget')
        except BaseException as exc:report.update(status='fail',error_type=g.error_family(exc),decision='CLOSED_OBJECTIVE_CONTROL');update()


def validate_native(rt,value,binding,rev,qualification):
    fixed=dict(schema=SCHEMA,stage='coherent_pair_gpu_objective_native',status='pass',phase='complete',producer_revision=rev,
        source_binding=binding,manifest=manifest(),image_id=IMAGE,fullbank_qualification=qualification,counts=[4,3600,64],
        candidate_rows=28800,tuple_rows=512,bridge_rows=230400,native_tokens=1500,group_shape=[4,3600],fd_calls=72,
        route_references=1843200,native_supported_cells=28800,group_supported_cells=14400,
        controls_completed=2,solvers_completed=2,source_fixtures_rehashed_after=True,snapshots_rehashed_after=True,source_rehashed_after=True,
        positive_groups=2,
        real_fit_executed=False,models_loaded=False,rgb_read=False,references_read=False,challenge_inputs_used=False,
        quality_verified=False,adoption=False,global_optimum_claimed=False,decision=DECISION)
    rt.require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in fixed.items()),'Complete fixed objective control required')
    rt.require(value['scipy_before']==value['scipy_after'] and value['scipy_before']['version']=='1.16.3'
        and value['scipy_before']['first_native_record_census'] is True and value['scipy_before']['populated_record_claims_verified'] is True
        and value['scipy_before']['caches_image_anchored_not_record_certified'] is True
        and type(value['scipy_before']['entries']) is int and value['scipy_before']['entries']>0
        and value['scipy_before']['license_files'],'Native fixed SciPy before/after required')
    rt.require(all(re.fullmatch('[0-9a-f]{64}',str(value[k])) for k in ('source_sha256','packed_sha256','identity_sha256',
        'full_original_observations_sha256')),'Complete original bank/identity fingerprints required')
    runtime=value.get('runtime',{});expected_runtime=dict(numpy='1.26.3',torch='2.5.1+cu124',cuda='12.4',capability=[9,0],tf32=False,
        deterministic_algorithms=True,cublas_workspace_config=':4096:8')
    rt.require(all(type(runtime.get(k)) is type(v) and runtime[k]==v for k,v in expected_runtime.items())
        and type(runtime.get('gpu_name')) is str and 'H100' in runtime['gpu_name']
        and type(runtime.get('gpu_total_bytes')) is int and runtime['gpu_total_bytes']>MEMORY
        and type(runtime.get('python')) is str and re.fullmatch(r'3\.11\.[0-9]+',runtime['python']), 'Actual fixed GPU runtime required')
    for row,alpha in zip(value['controls'],(0.,.3125)):
        rt.require(row['alpha']==alpha and row['repeat_bits_exact'] is True and row['cpu_packed_objective_passed'] is True
            and row['fd17_and_alpha_passed'] is True and row['gradient_shape']==[17] and row['scalar_gradient_shape']==[1],
            'Both complete objective/FD controls required')
        rt.require(row['group_shape']==[4,3600] and row['native_shape']==[4,2,3600]
            and row['identity_sha256']==value['identity_sha256'] and row['distribution_fingerprint']==value['packed_sha256']
            and re.fullmatch('[0-9a-f]{64}',str(row['parameter_fingerprint']))
            and all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('loss_a','loss_b','seconds'))
            and row['seconds']>0,'Complete objective output grids/finite measurements required')
    rt.require(len(value['controls'])==len(value['solvers'])==2,'Every objective/solver control required')
    for row,arm in zip(value['solvers'],('A','B')):
        rt.require(row['arm']==arm and row['success'] is True and row['control_passed'] is True
            and 0<=row['iterations']<=20 and 0<row['evaluations']<=100 and row['projected_gradient_inf']<=1e-6
            and math.isfinite(row['final_loss']) and row['final_loss']<=row['initial_loss']
            and len(row['parameters'])==(17 if arm=='A' else 1) and all(math.isfinite(x) for x in row['parameters'])
            and 0<=row['projected_gradient_inf']<=1e-6 and row['status']==0
            and type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>0
            and (arm=='A' or row['parameters'][0]>=0),'Fixed short solver/stationarity required')
    rt.require(all(type(value[k]) is int and 0<=value[k]<=MEMORY for k in ('peak_torch_allocated','peak_torch_reserved'))
        and type(value.get('max_rss_bytes')) is int and 0<value['max_rss_bytes']<=MEMORY,'6GiB allocator/host RSS required')


def host(code,rev):
    started=time.monotonic();deadline=started+BUDGET;full,g,rt,_=helpers(code)
    def cancelled(*_):raise TimeoutError('Objective host deadline')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    signal.setitimer(signal.ITIMER_REAL,BUDGET);before=proof(full,g,rt,code,rev);out=rt.canonical(ROOT/RESULT)
    rt.require(not out.exists(),'Fresh objective namespace required');out.mkdir(mode=0o755);os.chown(out,1000,1000);out.chmod(0o755)
    inode=(out.stat().st_dev,out.stat().st_ino);rt.write(out/'proof.json',(json.dumps(before,sort_keys=True)+'\n').encode(),0o444);pin=rt.identity(out/'proof.json',1<<20)
    cid=out/'container.cid';name='world-reward-objective-'+rev[:12];owned=False;removed=False
    report=dict(schema=SCHEMA,stage='coherent_pair_gpu_objective_host',status='fail',producer_revision=rev,source_binding=before['source_binding'],
        manifest=manifest(),image=before['image'],fullbank_qualification=before['fullbank_qualification'],
        technical_resumption=before['technical_resumption'],metadata_resumption=before['metadata_resumption'],decision='CLOSED_OBJECTIVE_CONTROL')
    try:
        rt.require(not g.control(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).strip(),'Owned name exists');owned=True
        argv=['docker','run','--rm','--cidfile',str(cid),'--name',name,'--label','world_reward.tiny_gpu.owner='+rev,'--gpus','all',
            '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','6g','--cpus','4',
            '--pids-limit','256','--user','1000:1000','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=128m','--entrypoint','/usr/bin/env']
        for p in leaves(code):argv+=['--mount',f'type=bind,src={p},dst={p},readonly']
        argv+=['--mount',f'type=bind,src={out},dst={out}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
            'WR_IMAGE_ID='+IMAGE,'PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'python','-I','-B',str(code/HELPERS[0]),'native',str(code),rev,str(out),str(pin['bytes']),pin['sha256'],str(deadline)]
        r=subprocess.run(argv,capture_output=True,timeout=max(.001,deadline-time.monotonic()),check=False)
        report.update(native_exit_code=r.returncode,diagnostics={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in (('stdout',r.stdout),('stderr',r.stderr))})
        rt.require(len(r.stdout)<=32768 and len(r.stderr)<=32768,'Bounded native diagnostics required')
        removed=g.cleanup(rt,cid,name,rev);value=rt.strict((out/'native.json').read_bytes());rt.require(r.returncode==0,'Objective native process failed')
        validate_native(rt,value,before['source_binding'],rev,before['fullbank_qualification'])
        rt.require(value['technical_resumption']==before['technical_resumption'],'Native technical-resumption provenance differs')
        rt.require(value['metadata_resumption']==before['metadata_resumption'],'Native metadata-resumption provenance differs')
        report.update(status='pass',decision=DECISION,native_report_identity=rt.identity(out/'native.json',1<<20))
    except BaseException as exc:report.update(status='fail',error_type=g.error_family(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:removed=g.cleanup(rt,cid,name,rev) if owned else False
        except Exception as exc:report.update(status='fail',cleanup_error_type=g.error_family(exc))
        try:report['source_rehashed_after']=proof(full,g,rt,code,rev)==before
        except Exception as exc:report.update(status='fail',post_error_type=g.error_family(exc))
        rt.require((out.stat().st_dev,out.stat().st_ino)==inode and out.stat().st_uid==1000,'Owned output replaced')
        rt.require({p.name for p in out.iterdir()}<={'proof.json','container.cid','native.json'},'Foreign output inventory')
        report.update(owned_container_removed=removed,elapsed_seconds=time.monotonic()-started)
        if not removed or not report.get('source_rehashed_after') or time.monotonic()>=deadline:report['status']='fail'
        publish(g,rt,out/'report.json',report,deadline,seal=True)
    return report


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('mode',choices=('host','native'));p.add_argument('code');p.add_argument('revision');p.add_argument('args',nargs='*');a=p.parse_args();code=Path(a.code)
    if not re.fullmatch('[0-9a-f]{40}',a.revision) or code!=ROOT/'jobs'/a.revision/ENTRY/'code' or Path(__file__).resolve()!=code/HELPERS[0]:raise ValueError('Exact immutable objective source required')
    if a.mode=='host':
        if a.args or os.geteuid()!=0 or sys.platform!='linux' or os.uname().nodename!='scenesmith-ncc-h100-01' or os.environ.get('DOCKER_HOST')!='unix://'+str(ROOT/'docker.sock'):raise ValueError('Exact VM01 root host required')
        node=ROOT/'jobs/.world-reward-h100.lock';s=node.lstat();fd=os.fstat(9)
        if node.is_symlink() or not stat.S_ISREG(s.st_mode) or (s.st_dev,s.st_ino)!=(fd.st_dev,fd.st_ino):raise ValueError('Actual FD9 lease required')
        value=host(code,a.revision)
    else:
        if len(a.args)!=4:raise ValueError('Exact native proof/deadline arguments required')
        out,size,digest,end=a.args;deadline=float(end)
        if not math.isfinite(deadline) or not 0<deadline-time.monotonic()<=BUDGET:raise ValueError('Shared finite1200s deadline required')
        value=native(code,a.revision,Path(out),dict(bytes=int(size),sha256=digest),deadline)
    print(json.dumps(dict(status=value['status'],decision=value['decision'],error_type=value.get('error_type')),sort_keys=True))
    if value['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()

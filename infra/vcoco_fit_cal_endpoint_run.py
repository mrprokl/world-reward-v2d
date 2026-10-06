"""One complete blinded48 endpoint run; explicit context, no legacy profiles."""
import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_fit_cal_endpoint_observations as seam

original,rt=seam.original,seam.rt
ROOT,IMAGE=rt.ROOT,original.IMAGE
ENTRY='run_vcoco_fit_cal_endpoint_observations'
OUTPUT=ROOT/'results/vcoco-fit-cal-endpoint-observations-v1'
DATA=Path('/srv/world-reward-data/vcoco_fit_cal_v1/inputs')
SCHEMA=seam.SCHEMA
BUDGET=600
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_fit_cal_endpoint_run.py','infra/vcoco_fit_cal_endpoint_observations.py',
    'src/world_reward/rgb_bank_inputs.py',*original.NATIVE_FILES)))
HELPERS=tuple(dict.fromkeys((*NATIVE_FILES,'infra/run_vcoco_fit_cal_endpoint_observations.sh',
    'infra/sealed_callback_publication.py','infra/vcoco_fit_cal_acquisition_inputs.py',*seam.HELPERS)))
CONTEXT=seam.EndpointContext(DATA,OUTPUT)


def host_inputs(code,revision,checkpoint):
    # This import stays host-only and is deliberately absent from NATIVE_FILES.
    import vcoco_fit_cal_acquisition_inputs as acquisition
    records,proof=acquisition.authenticate_actual_acquisition(code,revision,ENTRY,
        tuple(dict.fromkeys((*HELPERS,*acquisition.acquisition.HELPERS))),checkpoint)
    rt.require(Path(acquisition.__file__).resolve()==code/'infra/vcoco_fit_cal_acquisition_inputs.py','Actual host input helper required')
    rows=[{k:r[k]for k in seam.public.KEYS}for r in records]
    inputs=seam.public_inputs(CONTEXT,proof['public_manifest_identity'])
    rt.require(inputs['images']==rows and all(r['path']==str(DATA/r['file'])for r in records), 'Public-only48 native projection differs')
    return inputs,proof


def source(code,revision):
    import vcoco_fit_cal_acquisition_inputs as acquisition
    import sealed_callback_publication as publication
    helpers=tuple(dict.fromkeys((*HELPERS,*acquisition.acquisition.HELPERS)))
    value=rt.source(ROOT,code,revision,ENTRY,helpers);seam.source_identity(code,value)
    rt.require(Path(__file__).resolve()==code/NATIVE_FILES[0]
        and Path(publication.__file__).resolve()==code/'infra/sealed_callback_publication.py','Actual caller/publisher origin required')
    return dict(binding=value,states={str(p.relative_to(code)):snapshot(p)for p in(code,*sorted(code.rglob('*')))})


def snapshot(path):
    s=rt.canonical(path).lstat()
    return tuple(getattr(s,n)for n in('st_dev','st_ino','st_mode','st_size','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns'))


def command(args,deadline):
    seam.check(deadline)
    r=subprocess.run(args,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
        capture_output=True,timeout=min(10,max(.001,deadline-time.monotonic())),check=False)
    rt.require(r.returncode==0 and len(r.stdout)<=1 << 20,'Bounded lifecycle command failed')
    return r.stdout.decode().strip()


def output_identities():
    allowed={'proof.json','native.json','.container.cid'}|{f'image_{i:06d}.npz'for i in range(48)}
    rt.require({p.name for p in OUTPUT.iterdir()}<=allowed,'Foreign native output')
    result={p.name:rt.identity(p,16 << 20,readonly=p.name!='.container.cid')for p in OUTPUT.iterdir()}
    rt.require(sum(p['bytes']for p in result.values())<=768 << 20,'Bounded full output bytes')
    return result


def cleanup(name,revision,deadline):
    path=OUTPUT/'.container.cid'
    if path.exists():
        rt.identity(path,65,readonly=False);raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Owned original CID required');cid=raw.decode().strip()
        present=command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline)
        rt.require(present in('',cid),'Ambiguous owned CID')
        if present:
            actual=command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
            rt.require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+revision,'Foreign container cannot be removed')
            command(['docker','rm','-f',cid],deadline)
        rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'Owned CID survives');path.chmod(0o400)
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Owned name survives')


def native_source(code,revision,proof):
    rt.require(proof['source']['producer_revision']==revision and proof['image_id']==IMAGE
        and set(proof['native_files'])==set(NATIVE_FILES)
        and {str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set(NATIVE_FILES), 'Exact native source whitelist required')
    seam.source_identity(code,proof['source'])
    for name,wanted in proof['source']['markers'].items():
        rt.require(rt.identity(code.parent/name,100)==wanted,'Original dispatch marker differs')
    rt.require((code.parent/'revision').read_bytes()==(revision+'\n').encode(),'Exact original native revision')
    return {n:dict(pin=rt.identity(code/n,2 << 20,empty=True),state=snapshot(code/n))for n in NATIVE_FILES}


def validate_native(report,proof,revision,pin,inputs):
    rt.require(report['schema']==SCHEMA and report['stage']=='native_full48_endpoint_observations'
        and report['status']=='pass'and report['phase']=='complete'and report['producer_revision']==revision
        and report['image_id']==IMAGE and report['proof_identity']==pin and report['profile']==seam.PROFILE
        and type(report['model_loads'])is int and report['model_loads']==2
        and report['models_released']is report['source_inputs_assets_runtime_rehashed_after']is True
        and report['runtime_identity']==proof['owl_runtime'] and len(report['images'])==48
        and all(type(report[k])is int and report[k]==48 for k in('person_forward_calls','image_embed_calls','objectness_calls','box_calls'))
        and all(report[k]is False for k in('reference_metadata_read','split_metadata_read','ground_truth_used','challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','FIT_performed','adoption')),
        'Complete qualified original48 native observations required')
    for i,(r,image)in enumerate(zip(report['images'],inputs['images'])):
        rt.require(r['image_id']==image['image_id']and r['bank_index']==r['original_slot']==r['acquired_ordinal']==i
            and r['original_frame_index']==0 and r['image_size']==[image['height'],image['width']]
            and r['input_file']==image['file']and r['input_identity']=={k:image[k]for k in('bytes','sha256')}
            and r['file']==f'image_{i:06d}.npz'and r['person_native_queries']==900 and r['owl_patches']==3600
            and type(r['person_retained_rows'])is type(r['person_postprocessor_rows'])is int
            and 0<=r['person_retained_rows']<=r['person_postprocessor_rows']<=900
            and len(r['person_ids'])==len(set(r['person_ids']))==r['person_retained_rows']
            and rt.identity(OUTPUT/r['file'],16 << 20)==r['identity'],'Every original48 native bank required')
        a=r['arrays'];rt.require(set(a)==seam.KEYS and all(set(v)=={'shape','dtype','sha256'}
            and type(v['shape'])is list and all(type(n)is int and n>=0 for n in v['shape'])
            and type(v['dtype'])is str and re.fullmatch('[0-9a-f]{64}',str(v['sha256']))for v in a.values()),'Full17 array identities required')
        shapes={'person_model_pred_boxes':[1,900,4],'person_model_logits':[1,900,256],'owl_patch_ids':[3600],
            'owl_boxes_padded_normalized_cxcywh':[3600,4],'owl_objectness_logits':[3600],
            'owl_boxes_original_xyxy':[3600,4],'image_size':[2],'original_frame_index':[]}
        rt.require(all(a[k]['shape']==v for k,v in shapes.items()),'All900/3600 raw shapes required')
    rt.require(sum(p.stat().st_size for p in OUTPUT.iterdir())<=768 << 20,'Bounded full output bytes')


def native(code,revision,pin,deadline):
    proof=rt.pinned(OUTPUT/'proof.json',pin,2 << 20);models=None;before=None;assets={};runtime=None;grounding=None;inputs=None
    directory=OUTPUT.lstat();owner=(directory.st_dev,directory.st_ino,directory.st_uid)
    rt.require(stat.S_ISDIR(directory.st_mode)and stat.S_IMODE(directory.st_mode)==0o700
        and directory.st_uid==os.geteuid(),'Owned native output namespace required')
    report=dict(schema=SCHEMA,stage='native_full48_endpoint_observations',status='fail',phase='authentication',
        producer_revision=revision,image_id=IMAGE,proof_identity=pin,profile=seam.PROFILE,images=[],model_loads=0,models_released=False,
        source_inputs_assets_runtime_rehashed_after=False,reference_metadata_read=False,split_metadata_read=False,
        ground_truth_used=False,challenge_inputs_used=False,actor_selection_performed=False,
        ownership_verified=False,quality_verified=False,FIT_performed=False,adoption=False)
    try:
        rt.require(set(proof)=={'source','native_files','assets','owl_runtime','inputs_identity','images','image_id','profile'}
            and proof['images']==48 and proof['profile']==seam.PROFILE and os.environ.get('WR_IMAGE_ID')==IMAGE
            and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Exact offline blinded native proof required')
        before=native_source(code,revision,proof)
        rt.require({n:r['pin']for n,r in before.items()}==proof['native_files'],'Actual native helper bytes differ')
        original.configuration(code,proof['source']);inputs=seam.public_inputs(CONTEXT,proof['inputs_identity'],native_mounts=True)
        policy=rt.pinned(code/original.owl.CONFIG,proof['native_files'][original.owl.CONFIG],16 << 10)
        expected,card=original.expected_asset_files(policy)
        rt.require(set(proof['assets'])==set(expected)|{card}and all(proof['assets'][n]==p for n,p in expected.items()),'Exact model/config/card leaves required')
        assets={n:dict(pin=rt.identity(Path(n),1 << 30),state=snapshot(n))for n in proof['assets']}
        rt.require({n:r['pin']for n,r in assets.items()}==proof['assets'],'Actual native assets differ')
        runtime=original.owl.installed(policy);grounding=original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])
        rt.require(runtime==proof['owl_runtime'],'Actual qualified runtime differs');report.update(runtime_identity=runtime,grounding_source_identity=grounding)
        report['phase']='models';seam.check(deadline);models=original.load_models(policy);report['model_loads']=2
        report['phase']='all48_original_forward';seam.observe(CONTEXT,inputs,models,deadline,records=report['images'])
        seam.validate_records(CONTEXT,inputs,report['images'],deadline);report.update(status='pass',phase='complete')
    except BaseException as exc:report['error_type']=original.error(exc)
    finally:
        try:
            if models is not None:
                ops=models[2];models=None;ops.tensor_ops.cuda.empty_cache();ops.tensor_ops.cuda.synchronize();del ops
            report['models_released']=True
        except BaseException as exc:report.update(status='fail',release_error_type=original.error(exc))
        for k in('person_forward_calls','image_embed_calls','objectness_calls','box_calls'):report[k]=len(report['images'])
        try:
            rt.require(before is not None and native_source(code,revision,proof)==before
                and seam.public_inputs(CONTEXT,proof['inputs_identity'],native_mounts=True)==inputs,'Native source/public bytes changed')
            rt.require({n:dict(pin=rt.identity(n,1 << 30),state=snapshot(n))for n in assets}==assets,'Native model/config/card changed')
            if runtime is not None:rt.require(original.owl.installed(policy)==runtime and original.installed_grounding(runtime['wheel_RECORD_identities']['transformers'])==grounding,'Installed source/runtime changed')
            for r in report['images']:rt.require(rt.identity(OUTPUT/r['file'],16 << 20)==r['identity'],'Saved partial native bank changed')
            if report['status']=='pass':seam.validate_records(CONTEXT,inputs,report['images'],deadline)
            report['source_inputs_assets_runtime_rehashed_after']=True;seam.check(deadline)
        except BaseException as exc:report.update(status='fail',post_error_type=original.error(exc))
        report['elapsed_seconds']=time.monotonic()-float(os.environ['WR_ENDPOINT_STARTED'])
        directory=OUTPUT.lstat();rt.require(stat.S_ISDIR(directory.st_mode)and stat.S_IMODE(directory.st_mode)==0o700
            and(directory.st_dev,directory.st_ino,directory.st_uid)==owner,'Original native output namespace changed')
        with(OUTPUT/'native.json').open('x+b')as stream:
            os.fchmod(stream.fileno(),0o400);inode=os.fstat(stream.fileno())
            def update():
                s=(OUTPUT/'native.json').lstat();raw=original.encode(report)
                rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==0o400
                    and(s.st_dev,s.st_ino,s.st_uid)==(inode.st_dev,inode.st_ino,inode.st_uid)
                    and len(raw)<=2 << 20,'Owned bounded native report required')
                stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
            try:update();seam.check(deadline)
            except BaseException:report.update(status='fail',publication_failed=True);update()
    return report


def host(code,revision,started,deadline):
    import sealed_callback_publication as publication
    before=source(code,revision);inputs,acq=host_inputs(code,revision,lambda:seam.check(deadline))
    policy=original.configuration(code,before['binding']);prior=original.qualifications(code,policy,live=True)
    owl_policy=rt.pinned(code/original.owl.CONFIG,before['binding']['helpers'][original.owl.CONFIG],16 << 10)
    assets=original.asset_files(prior,owl_policy);asset_states={n:snapshot(n)for n in assets}
    rt.require({n:rt.identity(n,1 << 30)for n in assets}==assets,'All original assets authenticated before models')
    rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh48 endpoint output required')
    lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock');node=lock.lstat();fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW)
    created=False;saved_outputs=None;name='world-reward-vcoco-fit-cal-endpoint-'+revision[:12]
    report=dict(schema=SCHEMA,stage='full48_endpoint_observations_host',status='fail',producer_revision=revision,
        source_binding=before['binding'],source_stat_identity=before['states'],acquisition_public_proof=acq,
        original_qualification=prior,image_id=IMAGE,public_inputs_identity=acq['public_manifest_identity'],acquired_images=48,
        owned_cleanup_verified=False,source_inputs_assets_runtime_rehashed_after=False,outputs_sealed=False,
        reference_metadata_read=False,split_metadata_read=False,FIT_performed=False,ground_truth_used=False,
        challenge_inputs_used=False,actor_selection_performed=False,ownership_verified=False,quality_verified=False,adoption=False)
    try:
        rt.require(stat.S_ISREG(node.st_mode)and node.st_nlink==1 and(os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(node.st_dev,node.st_ino),'Actual original GPU lock inode')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline)
            and not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Other GPU work/occupied name')
        image=command(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],deadline);rt.require(image==IMAGE,'Exact qualified original image')
        OUTPUT.mkdir(mode=0o700);created=True;s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
        proof=dict(source=before['binding'],native_files={n:before['binding']['helpers'][n]for n in NATIVE_FILES},
            assets=assets,owl_runtime=prior['owl']['native_runtime'],inputs_identity=acq['public_manifest_identity'],images=48,image_id=IMAGE,profile=seam.PROFILE)
        pin=original.write(OUTPUT/'proof.json',proof)
        mounts=[code/n for n in NATIVE_FILES]+[code.parent/n for n in('revision','source-sha256')]+[Path(n)for n in assets]+[DATA/'manifest.json']+[DATA/r['file']for r in inputs['images']]
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/'.container.cid'),'--label','world-reward.job='+ENTRY,
            '--label','world-reward.revision='+revision,'--gpus','device=0','--network','none','--user','0:0','--memory','64g','--cpus','4','--pids-limit','256',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in mounts:
            rt.canonical(path);rt.require(path.is_file()and','not in str(path)and'\n'not in str(path),'Individual immutable mount required')
            cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','HF_HUB_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','WR_IMAGE_ID='+IMAGE,
            'WR_ENDPOINT_STARTED='+format(started,'.17g'),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,
            'WR_ENDPOINT_DEADLINE='+format(deadline-20,'.17g'),'/opt/conda/bin/python','-I','-B',str(code/NATIVE_FILES[0]),
            '--native','--proof-bytes',str(pin['bytes']),'--proof-sha256',pin['sha256']]
        r=subprocess.run(cmd,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()-20),check=False)
        saved_outputs=output_identities()
        report['native_exit_status']=r.returncode;report['native_report_identity']=rt.identity(OUTPUT/'native.json',2 << 20)
        saved=rt.pinned(OUTPUT/'native.json',report['native_report_identity'],2 << 20)
        report['native_summary']={k:saved[k]for k in('status','phase','error_type','post_error_type','model_loads','models_released')if k in saved}
        rt.require(r.returncode==0,'Native complete endpoint process failed');validate_native(saved,proof,revision,pin,inputs)
        report.update(native_images=saved['images'],status='pass')
    except BaseException as exc:report['error_type']=original.error(exc)
    finally:
        try:
            if created:cleanup(name,revision,min(deadline+10,time.monotonic()+10));report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',cleanup_error_type=original.error(exc))
        try:
            rt.require(source(code,revision)==before and host_inputs(code,revision,lambda:seam.check(deadline))==(inputs,acq)
                and original.qualifications(code,policy,live=True)==prior and{n:rt.identity(n,1 << 30)for n in assets}==assets
                and{n:snapshot(n)for n in assets}==asset_states and(lock.lstat().st_dev,lock.lstat().st_ino)==(node.st_dev,node.st_ino),'Whole source/input/assets/runtime drift')
            rt.require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline),'GPU process survives')
            if created:rt.require(command(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],deadline)==image,'Original image changed')
            if saved_outputs is not None:rt.require(output_identities()==saved_outputs,'Saved native outputs changed')
            if 'native_images'in report:validate_native(saved,proof,revision,pin,inputs);rt.require(rt.identity(OUTPUT/'proof.json',2 << 20)==pin,'Proof changed')
            report['source_inputs_assets_runtime_rehashed_after']=True;seam.check(deadline)
        except BaseException as exc:report.update(status='fail',post_error_type=original.error(exc))
        try:
            if created:
                report['saved_output_identities']=output_identities()
                publication.publish(OUTPUT,report,deadline,started,owner,{p.name for p in OUTPUT.iterdir()},
                    encode=original.encode,identity=rt.identity,snapshot=snapshot,require=rt.require,check=seam.check,
                    sync=sync,error=original.error,maximum=16 << 20,report_maximum=2 << 20)
        finally:os.close(fd)
    return report


def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def arguments(argv):
    rt.require(all(argv.count(k)<=1 for k in('--native','--proof-bytes','--proof-sha256')),'No duplicated native arguments')
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--native',action='store_true');parser.add_argument('--proof-bytes',type=int);parser.add_argument('--proof-sha256')
    a=parser.parse_args(argv)
    rt.require(not a.native and a.proof_bytes is a.proof_sha256 is None or a.native and type(a.proof_bytes)is int
        and 0<a.proof_bytes<=2 << 20 and re.fullmatch('[0-9a-f]{64}',str(a.proof_sha256)),'Only explicit native byte proof or fixed host mode')
    return a


def main():
    args=arguments(sys.argv[1:]);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(code==ROOT/'jobs'/revision/ENTRY/'code'and re.fullmatch('[0-9a-f]{40}',revision)and Path(__file__).resolve()==code/NATIVE_FILES[0],'Exact committed caller origin')
    started=time.monotonic();deadline=float(os.environ['WR_ENDPOINT_DEADLINE'])if args.native else started+BUDGET
    rt.require(math.isfinite(deadline)and 0<deadline-time.monotonic()<=BUDGET,'Shared bounded deadline')
    def expired(*_):raise TimeoutError('Inclusive full48 deadline')
    handlers={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        if args.native:return native(code,revision,dict(bytes=args.proof_bytes,sha256=args.proof_sha256),deadline)
        rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Actual root VM02 host')
        return host(code,revision,started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    value=main();print(original.encode({k:value[k]for k in('stage','status')}).decode(),end='');raise SystemExit(0 if value['status']=='pass'else 1)

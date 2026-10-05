"""One-load/three-forward image-only OWLv2 qualification on procedural RGB.

Host authenticates original image lineage and actual acquisition receipt. Only
individual current source files, original four assets and a sanitized proof
cross the native boundary. No dataset, text query, install or quality claim.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_owlv2_native_qualify'
CONFIG='configs/owlv2_native_qualification_v1.json'
IMAGE='sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252'
ASSETS=ROOT/'weights/owlv2_objectness_v2'
ACQUIRE_REPORT=ROOT/'results/owlv2-objectness-assets-v2.json'
OUTPUT=ROOT/'results/owlv2-native-qualification-v1'
NATIVE_FILES=('infra/owlv2_native_qualify.py','infra/mediapipe_cpu_runtime_verify.py',CONFIG,
              'src/world_reward/__init__.py','src/world_reward/owlv2_object_observations.py')
HELPERS=(*NATIVE_FILES,'infra/run_owlv2_native_qualify.sh','infra/bridge_frontend_bindings.py',
         'infra/frontend_selected_assets.py','infra/frontend_sam2_kernel_gate.py','configs/frontend_grounding_source_pins.json')
ACQUIRE_HELPERS=('infra/owlv2_acquire.py','infra/run_owlv2_acquire.sh','configs/owlv2_assets_v2.json',
    'infra/mediapipe_cpu_runtime_verify.py','infra/mediapipe_hands_acquire.py','infra/masa_acquire.py')


def encode(v):return (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()


def write(path,v):
    raw=encode(v);rt.write(path,raw);return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Inclusive120s qualification deadline')


def error(exc):return type(exc).__name__ if type(exc) in (ValueError,RuntimeError,TimeoutError,OSError,KeyError,ImportError,ModuleNotFoundError) else 'other'


def configuration(code,source):
    p=rt.pinned(code/CONFIG,source['helpers'][CONFIG],16<<10)
    rt.require(p['schema']=='world_reward.owlv2_native_qualification.v1' and p['image_id']==IMAGE
        and p['budget_seconds']==120 and p['memory_bytes']==64<<30 and p['model_loads']==1 and p['image_embed_calls']==3
        and p['patches_per_frame']==3600 and p['text_queries']==0 and p['all_patches_retained'] is True
        and p['quality_verified'] is p['adopted'] is False and p['output']==str(OUTPUT), 'Exact prospective native qualification recipe required')
    rt.require(source['helpers']['src/world_reward/owlv2_object_observations.py']==p['adapter_identity']
        and source['helpers']['infra/mediapipe_cpu_runtime_verify.py']==p['rt_identity'], 'Released original adapter/helper required')
    return p


def runtime_proof(code,*,live=False):
    import bridge_frontend_bindings as b
    kernel=b.kernel_helper();build=b.pinned(b.BUILD_REPORT,b.BUILD_PIN);gate=b.pinned(b.KERNEL_REPORT,b.KERNEL_PIN)
    old=kernel.closure(b.BUILD_CODE,b.BUILD_REV,'run_frontend_grounding_build',kernel.BUILD_HELPERS)
    oldgate=kernel.closure(b.KERNEL_CODE,b.KERNEL_REV,'run_frontend_sam2_kernel_gate',
        ('infra/frontend_sam2_kernel_gate.py','infra/run_frontend_sam2_kernel_gate.sh',b.CONFIG))
    rt.require(build['source_binding']==old and gate['source_binding']==oldgate
        and old['helpers']['infra/frontend_grounding_build.py']['sha256']==b.BUILD_SHA
        and oldgate['helpers']['infra/frontend_sam2_kernel_gate.py']==b.KERNEL_SOURCE_PIN
        and rt.identity(code/b.CONFIG)==old['helpers'][b.CONFIG]==oldgate['helpers'][b.CONFIG], 'Original full build/kernel source/config differs')
    rt.require(build['schema']=='world_reward.frontend_grounding_build.v6' and build['status']=='pass' and build['phase']=='complete'
        and build['producer_revision']==b.BUILD_REV and build['offline_build_exit_code']==build['child_probe_exit_code']==0
        and build['parent_unchanged_verified'] is True and build['source_rechecked_before_and_after'] is True
        and gate['schema']=='world_reward.frontend_sam2_kernel_gate.v1' and gate['status']=='pass'
        and gate['producer_revision']==b.KERNEL_REV and gate['image_id']==IMAGE
        and gate['CUDA_operator_execution_verified'] is True and gate['models_loaded'] is False
        and gate['build_report_identity']==b.BUILD_PIN and gate['operator_source_identities']['extension']==b.EXTENSION_PIN,
        'Genuine original CPU build and kernel PASS required, not OWLv2 model qualification')
    child,parent=build['child_image'],build['parent_image'];owner=hashlib.sha256((b.BUILD_REV+old['closure_sha256']).encode()).hexdigest()
    rt.require(child['Id']==IMAGE and parent['Id']==kernel.BASE and child['Architecture']=='amd64' and child['Os']=='linux'
        and len(parent['RootFS']['Layers'])==44 and len(child['RootFS']['Layers'])==50
        and child['RootFS']['Layers'][:44]==parent['RootFS']['Layers'] and build['owner']==owner, 'Actual unchanged frontend image lineage required')
    probe=build['private_child_probe_log'];probe_path=ROOT/probe['relative_path']
    rt.require(probe['relative_path']=='results/frontend-grounding-build-v6/child-CPU-probe.log'
        and rt.identity(probe_path,128<<10)=={k:probe[k] for k in ('bytes','sha256')}, 'Original CPU import log identity differs')
    if live:kernel.validate_live_image(dict(child_image=child,parent_image=parent,owner=owner))
    return dict(build_report_identity=b.BUILD_PIN,kernel_report_identity=b.KERNEL_PIN,build_source=old,kernel_source=oldgate,
        child_image=child,parent_image=parent,owner=owner,probe_identity={k:probe[k] for k in ('bytes','sha256')})


def acquisition(revision,pin,p):
    old=ROOT/'jobs'/revision/'run_owlv2_acquire/code';source=rt.source(ROOT,old,revision,'run_owlv2_acquire',ACQUIRE_HELPERS)
    report=rt.pinned(ACQUIRE_REPORT,pin,2<<20);cfg=rt.pinned(old/ACQUIRE_HELPERS[2],source['helpers'][ACQUIRE_HELPERS[2]],16<<10)
    rt.require(report['schema']=='world_reward.owlv2_assets_receipt.v1' and report['status']=='pass'
        and report['producer_revision']==revision and report['source_binding']==source and report['configuration_identity']==source['helpers'][ACQUIRE_HELPERS[2]]
        and report['source_rehashed_after'] is report['artifacts_rehashed_after'] is report['owned_partials_removed'] is report['outputs_sealed'] is True
        and report['checkpoint_decoded'] is report['models_loaded'] is report['dataset_read'] is report['gpu_used'] is False
        and report['publisher_revision']==p['publisher_revision'] and report['primary_model_license_recorded'] is True,
        'Actual complete unchanged OWLv2 asset receipt required')
    wanted=p['assets'];rt.require(report['final_asset_pins']==wanted
        and {r['file']:{k:r[k] for k in ('bytes','sha256')} for r in cfg['assets']}==wanted
        and {x.name for x in ASSETS.iterdir()}==set(wanted), 'Exact four original files required')
    for name,identity in wanted.items():rt.require(rt.identity(ASSETS/name,620000000)==identity, 'Original asset differs')
    return dict(producer_revision=revision,report_identity=pin,source_binding=source,assets=wanted)


def procedural_rgb():
    import numpy as np
    values=[]
    for h,w in ((48,80),(80,48),(64,64)):
        yy,xx=np.indices((h,w),dtype=np.uint16)
        values.append(np.stack(((3*xx+5*yy)%256,(11*xx+yy)%256,((xx//7+yy//9)%2)*255),axis=-1).astype(np.uint8))
    return values


def installed(p):
    """Exact audited native files AND wheel RECORD; no version-only adoption."""
    from importlib import metadata
    rt.require(sys.version.split()[0]==p['python'] and sys.prefix=='/opt/conda', 'Original CP311 conda runtime required')
    versions={n:metadata.version(n) for n in p['versions']};rt.require(versions==p['versions'], 'Original native distributions differ')
    pins={};record_pins={}
    for name,wanted in p['native_sources'].items():
        distribution='scipy' if name.startswith('scipy/') else 'transformers';dist=metadata.distribution(distribution)
        file=Path(dist.locate_file(name));rt.require(file.is_relative_to('/opt/conda/lib/python3.11/site-packages'), 'Foreign native package origin')
        rt.require(rt.identity(file,100000,readonly=False)==wanted, 'Original native implementation bytes differ')
        entries=[r for r in dist.files or () if str(r)==name]
        rt.require(len(entries)==1 and entries[0].size==wanted['bytes'] and entries[0].hash is not None
            and entries[0].hash.mode=='sha256' and entries[0].hash.value==base64.urlsafe_b64encode(bytes.fromhex(wanted['sha256'])).decode().rstrip('='), 'Original wheel RECORD entry differs')
        records=[r for r in dist.files if str(r).endswith('.dist-info/RECORD')]
        rt.require(len(records)==1, 'Exact native wheel RECORD required')
        record_pins[distribution]=rt.identity(Path(dist.locate_file(records[0])),2<<20,readonly=False);pins[name]=wanted
    return dict(versions=versions,sources=pins,wheel_RECORD_identities=record_pins)


def strict_state(model,state,torch):
    """Reject ignored/cast tensors before native strict load; retain integer buffers."""
    expected=model.state_dict()
    rt.require(expected and set(state)==set(expected), 'Complete original state keys required, including unused heads')
    for name,wanted in expected.items():
        actual=state[name]
        rt.require(actual.shape==wanted.shape and actual.dtype==wanted.dtype
            and (not actual.is_floating_point() or actual.dtype==torch.float32), 'Exact original state shape/dtype required')
    result=model.load_state_dict(state,strict=True)
    rt.require(not result.missing_keys and not result.unexpected_keys, 'Full original strict state required')


def native(code,revision,out,p,proof_pin,deadline):
    started=time.monotonic();proof=rt.pinned(out/'proof.json',proof_pin,2<<20)
    report=dict(schema=p['schema'],stage='native_owlv2_procedural',status='fail',phase='authentication',proof_identity=proof_pin,
        producer_revision=revision,image_id=IMAGE,model_loads=0,image_embed_calls=0,objectness_calls=0,box_calls=0,frames=[],
        dataset_read=False,text_queries=0,all_patches_retained=True,quality_verified=False,ownership_verified=False,adopted=False)
    try:
        rt.require(os.environ['WR_IMAGE_ID']==IMAGE and {x.name for x in Path('/sys/class/net').iterdir()}=={'lo'}, 'Exact offline original native image required')
        mounted={str(x.relative_to(code)) for x in code.rglob('*') if x.is_file()}
        rt.require(set(proof)=={'source','native_files','assets','image_id'} and proof['assets']==p['assets'] and proof['image_id']==IMAGE
            and mounted==set(NATIVE_FILES)==set(proof['native_files']) and proof['source']['producer_revision']==revision,
                   'Explicit native-only source mount inventory required')
        for name,wanted in proof['native_files'].items():rt.require(rt.identity(code/name,2<<20)==wanted, 'Native mounted source differs')
        for name,wanted in proof['source']['markers'].items():rt.require(rt.identity(code.parent/name,100)==wanted, 'Original native dispatch marker differs')
        for name,wanted in p['assets'].items():rt.require(rt.identity(ASSETS/name,620000000)==wanted, 'Original mounted asset differs')
        runtime=installed(p);report['runtime_identity']=runtime;report['phase']='imports';check(deadline)
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import Owlv2Config,Owlv2ForObjectDetection,Owlv2ImageProcessor
        from transformers.image_transforms import center_to_corners_format
        from transformers.models.owlv2.image_processing_owlv2 import _scale_boxes
        sys.path.insert(0,str(code/'src'))
        from world_reward.owlv2_object_observations import NativeOwlv2Operations,infer_owlv2_object_frame
        rt.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(), 'One actual H100 required')
        torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rt.require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'), 'No ambient native autocast permitted')
        report['phase']='model_construct';model=Owlv2ForObjectDetection(Owlv2Config.from_json_file(str(ASSETS/'config.json')))
        report['phase']='weights_decode';state=load_file(str(ASSETS/'model.safetensors'),device='cpu')
        report['phase']='strict_state';strict_state(model,state,torch)
        del state;model=model.to('cuda').eval();report['model_loads']=1
        processor=Owlv2ImageProcessor(**rt.strict((ASSETS/'preprocessor_config.json').read_bytes()))
        operations=NativeOwlv2Operations(processor,torch,'cuda',center_to_corners_format,_scale_boxes)
        for i,rgb in enumerate(procedural_rgb()):
            report['phase']='native_embedding';check(deadline)
            observation=infer_owlv2_object_frame(model,rgb,i,operations);torch.cuda.synchronize()
            report['image_embed_calls']+=1;report['objectness_calls']+=1;report['box_calls']+=1
            arrays=dict(patch_ids=observation.patch_ids,boxes_padded_normalized_cxcywh=observation.boxes_padded_normalized_cxcywh,
                objectness_logits=observation.objectness_logits,boxes_original_xyxy=observation.boxes_original_xyxy)
            saved=out/f'procedural_{i:02d}.npz'
            with saved.open('xb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o400)
            report['frames'].append(dict(original_frame_index=i,image_size=list(observation.image_size),patches=3600,
                procedural_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),file=saved.name,identity=rt.identity(saved,1<<20)))
        del model,processor;torch.cuda.empty_cache();torch.cuda.synchronize();report['phase']='posthash'
        rt.require(installed(p)==runtime and len(report['frames'])==3 and report['image_embed_calls']==report['objectness_calls']==report['box_calls']==3, 'All original three forwards/runtime required')
        for name,wanted in proof['native_files'].items():rt.require(rt.identity(code/name,2<<20)==wanted, 'Native mounted source changed')
        for name,wanted in p['assets'].items():rt.require(rt.identity(ASSETS/name,620000000)==wanted, 'Original mounted asset changed')
        for frame in report['frames']:rt.require(rt.identity(out/frame['file'],1<<20)==frame['identity'], 'Complete saved native bank changed')
        check(deadline);report.update(status='pass',phase='complete',source_runtime_assets_rehashed_after=True)
    except BaseException as exc:report['error_type']=error(exc)
    report['elapsed_seconds']=time.monotonic()-started;write(out/'native.json',report)
    return report


def command(args,deadline):
    check(deadline)
    env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
    result=subprocess.run(args,env=env,capture_output=True,timeout=min(10,max(.001,deadline-time.monotonic())),check=False)
    rt.require(result.returncode==0 and len(result.stdout)<=1<<20, 'Bounded native lifecycle control failed')
    return result.stdout.decode().strip()


def cleanup(cidfile,name,revision,deadline):
    rt.require(cidfile.exists(), 'Actual owned container ID receipt required');rt.identity(cidfile,65,readonly=False);raw=cidfile.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw), 'Exact owned container ID required');cid=raw.decode().strip()
    found=command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline);rt.require(found in ('',cid), 'Ambiguous owned CID')
    if found:
        actual=command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
        rt.require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Cannot remove foreign container')
        command(['docker','rm','-f',cid],deadline)
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline), 'Owned container survives')
    cidfile.chmod(0o400)


def dispatch(code,revision,p,acquire_revision,acquire_pin):
    import fcntl
    started=time.monotonic();deadline=started+120;source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    actual=configuration(code,source)
    rt.require(p is None or p==actual, 'Current exact qualification recipe required');p=actual
    runtime=runtime_proof(code,live=True);assets=acquisition(acquire_revision,acquire_pin,p)
    rt.require(not OUTPUT.exists() and OUTPUT.parent.is_dir(), 'Fresh fixed qualification output required')
    lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock');s=lock.lstat()
    rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink==1, 'Original single-link cooperative H100 lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);name='world-reward-owlv2-qualify-'+revision[:12];created=False
    host=dict(schema=p['schema'],stage='owlv2_procedural_qualification_host',status='fail',producer_revision=revision,
        source_binding=source,runtime_binding=runtime,acquisition_binding=assets,image_id=IMAGE,dataset_read=False,
        quality_verified=False,ownership_verified=False,adopted=False,owned_cleanup_verified=False,outputs_sealed=False)
    try:
        rt.require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(s.st_dev,s.st_ino), 'H100 lock inode differs')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline)
            and not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline), 'Other GPU process or occupied owned name')
        OUTPUT.mkdir(mode=0o700);created=True
        safe=dict(source=source,native_files={n:source['helpers'][n] for n in NATIVE_FILES},assets=p['assets'],image_id=IMAGE)
        proof_pin=write(OUTPUT/'proof.json',safe)
        mounts=[code/n for n in NATIVE_FILES]+[code.parent/n for n in ('revision','source-sha256')]+[ASSETS/n for n in p['assets']]
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/'.container.cid'),'--label','world-reward.job='+ENTRY,
            '--label','world-reward.revision='+revision,'--gpus','device=0','--network','none','--user','0:0','--memory','64g',
            '--cpus','4','--pids-limit','256','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in mounts:
            rt.canonical(path);rt.require(path.is_file() and ',' not in str(path) and '\n' not in str(path), 'Individual readonly file mounts only')
            cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
            'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+IMAGE,'WR_OWL_DEADLINE='+format(deadline-15,'.17g'),
            '/opt/conda/bin/python','-I','-B',str(code/NATIVE_FILES[0]),'--native','--proof-bytes',str(proof_pin['bytes']),'--proof-sha256',proof_pin['sha256']]
        result=subprocess.run(cmd,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()-15),check=False)
        host['native_exit_status']=result.returncode
        host['native_report_identity']=rt.identity(OUTPUT/'native.json',1<<20)
        native_report=rt.pinned(OUTPUT/'native.json',host['native_report_identity'],1<<20)
        rt.require(result.returncode==0 and native_report['schema']==p['schema'] and native_report['stage']=='native_owlv2_procedural'
            and native_report['producer_revision']==revision and native_report['image_id']==IMAGE
            and native_report['status']=='pass' and native_report['phase']=='complete' and native_report['source_runtime_assets_rehashed_after'] is True
            and native_report['proof_identity']==proof_pin and native_report['model_loads']==1
            and native_report['image_embed_calls']==native_report['objectness_calls']==native_report['box_calls']==3, 'Actual complete native model qualification required')
        host['native_frames']=native_report['frames']
    except BaseException as exc:host['error_type']=error(exc)
    finally:
        try:
            if created:cleanup(OUTPUT/'.container.cid',name,revision,deadline);host['owned_cleanup_verified']=True
            rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==source and configuration(code,source)==p
                and runtime_proof(code,live=True)==runtime and acquisition(acquire_revision,acquire_pin,p)==assets
                and (lock.lstat().st_dev,lock.lstat().st_ino)==(s.st_dev,s.st_ino), 'Full original source/runtime/assets/lock changed')
            rt.require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline), 'GPU process survives')
            if 'native_frames' in host:
                rt.require({x.name for x in OUTPUT.iterdir()}=={'proof.json','native.json','.container.cid',*[f'procedural_{i:02d}.npz' for i in range(3)]}
                    and len(host['native_frames'])==3, 'Exclusive full three-bank output required')
                for i,frame in enumerate(host['native_frames']):
                    rt.require(frame['file']==f'procedural_{i:02d}.npz' and frame['original_frame_index']==i and frame['patches']==3600
                        and rt.identity(OUTPUT/frame['file'],1<<20)==frame['identity'], 'Original complete native bank changed')
                rt.require(rt.identity(OUTPUT/'native.json',1<<20)==host['native_report_identity']
                    and rt.identity(OUTPUT/'proof.json',2<<20)==proof_pin, 'Native receipt/source proof changed')
            host['source_runtime_assets_rehashed_after']=True
            if 'error_type' not in host and host.get('native_exit_status')==0:host['status']='pass'
        except BaseException as exc:host['post_error_type']=error(exc);host['status']='fail'
        os.close(fd)
        if created:
            with (OUTPUT/'host.json').open('xb') as stream:
                os.fchmod(stream.fileno(),0o400)
                for leaf in OUTPUT.iterdir():rt.canonical(leaf);rt.require(leaf.is_file(), 'Unexpected qualification output');leaf.chmod(0o400)
                OUTPUT.chmod(0o500);host['outputs_sealed']=True;host['elapsed_seconds']=time.monotonic()-started
                if host['elapsed_seconds']>=120:host.update(status='fail',post_error_type='TimeoutError')
                raw=encode(host);stream.write(raw);stream.flush();os.fsync(stream.fileno())
                if time.monotonic()>=deadline and host['status']=='pass':
                    host.update(status='fail',post_error_type='TimeoutError');stream.seek(0);stream.write(encode(host));stream.truncate();stream.flush();os.fsync(stream.fileno())
    return host


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--native',action='store_true')
    parser.add_argument('--acquisition-revision');parser.add_argument('--acquisition-report-bytes',type=int);parser.add_argument('--acquisition-report-sha256')
    parser.add_argument('--proof-bytes',type=int);parser.add_argument('--proof-sha256');a=parser.parse_args()
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(sys.platform=='linux' and os.geteuid()==0 and Path(__file__).resolve()==code/NATIVE_FILES[0], 'Actual Azure original qualification driver required')
    if a.native:
        rt.require(a.acquisition_revision is a.acquisition_report_bytes is a.acquisition_report_sha256 is None, 'Native must not inspect acquisition/full old source')
        proof_pin=dict(bytes=a.proof_bytes,sha256=a.proof_sha256);proof=rt.pinned(OUTPUT/'proof.json',proof_pin,2<<20)
        p=rt.pinned(code/CONFIG,proof['native_files'][CONFIG],16<<10)
        deadline=float(os.environ['WR_OWL_DEADLINE'])
        def expired(*_):raise TimeoutError('Inclusive native procedural budget')
        previous={s:signal.signal(s,expired) for s in (signal.SIGTERM,signal.SIGALRM)}
        signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
        try:result=native(code,revision,OUTPUT,p,proof_pin,deadline)
        finally:
            signal.alarm(0)
            for s,h in previous.items():signal.signal(s,h)
    else:
        rt.require(os.uname().nodename=='world-reward-ncc-h100-02' and a.proof_bytes is a.proof_sha256 is None
            and re.fullmatch('[0-9a-f]{40}',a.acquisition_revision or ''), 'Host actual acquisition revision/pins required')
        result=dispatch(code,revision,None,a.acquisition_revision,dict(bytes=a.acquisition_report_bytes,sha256=a.acquisition_report_sha256))
    print(json.dumps(dict(stage=result['stage'],status=result['status'],quality_verified=False),sort_keys=True))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(status='fail',error_type=error(exc)),sort_keys=True));raise SystemExit(1) from None

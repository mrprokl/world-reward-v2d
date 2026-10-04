"""Frozen full-T hand slots -> paired native SAM2 prompts; no identity/GT claims."""
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

if __name__=='__main__':
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    if not(re.fullmatch('[0-9a-f]{40}',revision)and code==Path('/srv/scenesmith/world-reward/jobs')/revision/'run_hand_mask_infer/code'
        and code.resolve()==code and Path(__file__).resolve()==code/'infra/hand_mask_infer.py'):
        raise ValueError('Actual immutable adapter required before imports')
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
import mediapipe_hand_scan as scan

ROOT=scan.ROOT;ENTRY='run_hand_mask_infer';BUDGET=900
PROTOCOL='configs/hand_mask_protocol_v1.json';SCAN_PINS='configs/mediapipe_hand_scan_v2_pins.json'
HELPERS=('infra/hand_mask_infer.py','infra/run_hand_mask_infer.sh','infra/mediapipe_hand_scan.py',
 'infra/mediapipe_hand_evaluate.py','infra/mediapipe_cpu_runtime_verify.py','infra/dexycb_identity_infer.py',
 'infra/bridge_frontend_bindings.py','src/world_reward/hand_mask_proposals.py',
 'src/world_reward/hand_observations.py','src/world_reward/automatic_candidate_bank.py',PROTOCOL,SCAN_PINS,
 *scan.source_helpers('v2'))
HELPERS=tuple(dict.fromkeys(HELPERS))


def frontend_json(rt,model):
    """Project only the authenticated selected-contract destination into JSON."""
    rt.require(type(model)is dict and set(model)=={'child_image','parent_image','owner','source_files','selected_contract'},
        'Exact original frontend proof schema required')
    contract=model['selected_contract']
    rt.require(type(contract)is dict and set(contract)=={'destination','entries','manifest_identity','extraction_receipt_identity','config_identity'}
        and isinstance(contract['destination'],Path),'Exact selected-contract path schema required')
    destination=rt.canonical(contract['destination'])
    projected=dict(model,selected_contract=dict(contract,destination=str(destination)))
    json.dumps(projected,allow_nan=False)  # No other Path/unknown object is coerced.
    return projected


def protocol(rt,code):
    pin=rt.identity(code/PROTOCOL)
    value=rt.strict((code/PROTOCOL).read_bytes())
    expected=dict(schema='world-reward-hand-mask-protocol-v1',cohort='v2',subject='20200903-subject-04',
        camera='836212060125',sequence_lex_indices=[4,39,74],original_image_size=[480,640],
        point_indices=[0,*range(5,21)],all_original_frames=True,
        sam2=dict(model='sam2.1_hiera_large',multimask_output=False,apply_postprocessing=True,
                  autocast='bfloat16',tf32=False,seed=0))
    rt.require(all(type(value.get(k))is type(v)and value[k]==v for k,v in expected.items())
        and value.get('budgets_seconds',{}).get('sam2_inference')==BUDGET,'Frozen native paired protocol required')
    rt.require(rt.identity(code/PROTOCOL)==pin,'Protocol changed during read')
    return value,pin


def authenticate_scan(rt,root,code):
    evidence=scan.authenticate(rt,root,code,'v2')
    pins,pin=scan.pins(rt,code,SCAN_PINS,'world_reward.mediapipe_hand_scan_pins.v1',('report','outputs'))
    revision=pins['producer_revision'];old=root/'jobs'/revision/scan.ENTRY/'code'
    source=rt.source(root,old,revision,scan.ENTRY,scan.source_helpers('v2'))
    control=scan.control_path(root,revision,'v2');out=control/'predictions'
    host=rt.pinned(control/'report.json',pins['report'],4<<20)
    expected=dict(stage='mediapipe_hand_scan_host_seal',status='pass',producer_revision=revision,
        source_binding=source,acquisition_pins=evidence['acquisition'],runtime_pins=evidence['runtime'],
        source_rehashed_after=True,owned_cleanup_verified=True,gpu_used=False,private_values_read=False,
        accuracy_verified=False,cohort='v2',public_base=scan.profile('v2')['base'],protocol_identity=evidence['acquisition']['protocol'])
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items()),'Genuine original scan host PASS required')
    rt.require(type(host.get('elapsed_seconds'))in(int,float)and 0<host['elapsed_seconds']<=scan.BUDGET,'Original full-scan budget differs')
    names={'report.json',*(f'sequence_{i:03d}'+s for i in(4,39,74)for s in('.npz','.npz.json'))}
    rt.require(set(pins['outputs'])==names and host.get('outputs')==pins['outputs']
        and {p.name for p in out.iterdir()}==names,'Exactly seven frozen scan artifacts required')
    frozen={control/'report.json':pins['report'],code/SCAN_PINS:pin}
    for name,identity in pins['outputs'].items():
        path=out/name;rt.require(rt.identity(path,32<<20)==identity,'Frozen native scan artifact changed');frozen[path]=identity
    native=rt.strict((out/'report.json').read_bytes())
    rt.require(native==host.get('native_report')and native.get('stage')=='external_dexycb_full_t_mediapipe_hand_scan'
        and native.get('status')=='pass'and native.get('phase')=='complete'and native.get('source_binding')==source
        and native.get('producer_revision')==revision and type(native.get('native_calls'))is int
        and native['native_calls']==len(evidence['manifest']['images'])and native.get('native_graphs')==1
        and native.get('cohort')=='v2'and native.get('public_base')==expected['public_base']
        and native.get('protocol_identity')==expected['protocol_identity']
        and all(native.get(k)is False for k in('gpu_used','private_values_read','accuracy_verified',
            'availability_is_visibility','handedness_is_detection_confidence','actor_identity_inferred')),
        'Complete lossless native full-T scan required')
    rt.require([(r['sequence_lex_index'],r['frames'],r['file'])for r in native.get('outputs',[])]==[
        (s['sequence_lex_index'],s['frames'],f"sequence_{s['sequence_lex_index']:03d}.npz")for s in evidence['manifest']['sequences']],
        'All original scan sequence outputs required')
    return evidence,pins,out,frozen


def native_mounts(code,revision,predictions):
    import dexycb_identity_infer as frontend
    paths=[p for p in frontend.host_mounts('masks',code,revision)
        if p!=code and p not in(code.parent/'revision',code.parent/'source-sha256',ROOT/frontend.BASE/'inputs')]
    paths += [code.parent,ROOT/scan.profile('v2')['base']/'inputs',*predictions]
    paths=sorted(set(paths))
    frontend.require(all(p.resolve()==p and ','not in str(p)and 'eval_private'not in p.parts
        and not p.is_relative_to(ROOT/'data')and not p.is_relative_to(ROOT/'vendor')for p in paths),
        'Readonly source/public/frontend leaves only')
    return paths


def save_frame(rt,np,path,result):
    arrays={f.name:getattr(result,f.name)for f in fields(result)if isinstance(getattr(result,f.name),np.ndarray)}
    arrays.update(frame_index=np.array(result.frame_index,np.int64),image_size=np.array(result.image_size,np.int64),
        point_indices=np.array(result.point_indices,np.int64),local_ids=np.array(result.local_ids,dtype='U32'),
        diagnostics=np.array(result.diagnostics,dtype='U64'))
    rt.require(not any(v.dtype.hasobject for v in arrays.values()),'No pickle or dropped proposal field')
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return rt.identity(path,16<<20)


def load_frame(rt,np,path,frame_index):
    """Lossless CPU-only restoration; unavailable raw floats are not repaired."""
    from world_reward.hand_mask_proposals import HandMaskProposals,POINT_INDICES
    from world_reward.hand_observations import HandInstances,LandmarkEvidence
    arrays=('original_xy','original_supported','raw_boxes','clipped_boxes','native_boxes','box_usable',
        'points17','native_points17','points_usable','points_outside','masks_a','masks_b',
        'mask_supported_a','mask_supported_b','raw_scores_a','raw_scores_b','b_reuses_a')
    with np.load(path,allow_pickle=False)as archive:
        rt.require(set(archive.files)==set(arrays)|{'frame_index','image_size','point_indices','local_ids','diagnostics'},'Exact mask evidence keys required')
        a={k:archive[k]for k in archive.files}
    rt.require(all(not x.dtype.hasobject for x in a.values())and a['frame_index'].dtype==np.int64
        and a['frame_index'].shape==()and int(a['frame_index'])==frame_index
        and a['image_size'].dtype==np.int64 and a['image_size'].tolist()==[480,640]
        and a['point_indices'].dtype==np.int64 and a['point_indices'].tolist()==list(POINT_INDICES),'Original mask grid/indices required')
    n=len(a['original_xy']);shapes=dict(original_xy=(n,21,2),original_supported=(n,21),
        raw_boxes=(n,4),clipped_boxes=(n,4),native_boxes=(n,4),box_usable=(n,),points17=(n,17,2),
        native_points17=(n,17,2),points_usable=(n,17),points_outside=(n,17),masks_a=(n,480,640),masks_b=(n,480,640),
        mask_supported_a=(n,),mask_supported_b=(n,),raw_scores_a=(n,),raw_scores_b=(n,),b_reuses_a=(n,))
    floats={'original_xy','raw_boxes','clipped_boxes','native_boxes','points17','native_points17','raw_scores_a','raw_scores_b'}
    for name,shape in shapes.items():
        dtype=np.float32 if name.startswith('native_')else np.float64 if name in floats else np.bool_
        rt.require(a[name].shape==shape and a[name].dtype==dtype,'Original array dtype/shape differs')
    ids=tuple(f'hand:{i:06d}'for i in range(n))
    rt.require(a['local_ids'].dtype==np.dtype('U32')and a['local_ids'].shape==(n,)and tuple(a['local_ids'])==ids
        and a['diagnostics'].dtype==np.dtype('U64')and a['diagnostics'].shape==(n,),'All local slots retained')
    class Replay:
        def set_image(self,rgb):pass
        def predict(self,**kw):
            arm='b'if 'point_coords'in kw else'a';eligible=a['box_usable']&~a['b_reuses_a']if arm=='b'else a['box_usable']
            masks,scores=a['masks_'+arm][eligible],a['raw_scores_'+arm][eligible]
            return (masks,scores,None)if len(masks)==1 else(masks[:,None],scores[:,None],None)
    from world_reward.hand_mask_proposals import propose_hand_masks
    restored=propose_hand_masks(np.zeros((480,640,3),np.uint8),HandInstances(LandmarkEvidence(
        a['original_xy'],a['original_supported'],'original_image','pixel')),Replay(),frame_index=frame_index)
    rt.require(all(getattr(restored,k).tobytes()==a[k].tobytes()for k in arrays)
        and restored.diagnostics==tuple(a['diagnostics']),'Lossless evidence/reuse/score reconstruction differs')
    return restored


def infer_frames(rt,np,manifest,observations,predictor,out,read_rgb,check,outputs=None):
    from world_reward.hand_mask_proposals import propose_hand_masks
    rt.require(len(manifest['sequences'])==len(observations)==3,'Exactly three lossless original scans required')
    rows=[] if outputs is None else outputs
    for sequence,obs in zip(manifest['sequences'],observations):
        frames=sequence['frames'];index=sequence['sequence_lex_index']
        rt.require(obs.image_size==(480,640)and len(obs.frames)==frames
            and np.array_equal(obs.frame_index,np.arange(frames,dtype=np.int64)),'Exact full-T restored slots required')
        original=[r for r in manifest['images']if r['sequence_lex_index']==index]
        rt.require(len(original)==frames,'No omitted original RGB frames')
        for t,(row,hands)in enumerate(zip(original,obs.frames)):
            check();rt.require(row['frame_position']==row['source_frame_id']==t,'Original indices differ')
            result=propose_hand_masks(read_rgb(row),hands,predictor,frame_index=t)
            name=f'sequence_{index:03d}_frame_{t:06d}.npz';pin=save_frame(rt,np,out/name,result)
            rows.append(dict(file=name,sequence_lex_index=index,frame_index=t,rgb_file=row['file'],
                rgb_sha256=row['sha256'],**pin,slots=len(result.local_ids),usable_boxes=int(result.box_usable.sum()),
                b_interventions=int((~result.b_reuses_a).sum()),a_supported=int(result.mask_supported_a.sum()),
                b_supported=int(result.mask_supported_b.sum())))
            check()
    rt.require(len(rows)==len(manifest['images']),'All original frames retained');return rows


def run_native(rt,root,code,out,proof_path,proof_pin):
    import dexycb_identity_infer as frontend
    import mediapipe_hand_evaluate as restoration
    start=time.monotonic();revision=os.environ['WR_CODE_REVISION'];proof=rt.pinned(proof_path,proof_pin,4<<20)
    deadline=start+proof['remaining_seconds'];check=lambda:rt.require(time.monotonic()<=deadline,'Inclusive all3 budget exceeded')
    rt.require(root==ROOT and code==root/'jobs'/revision/ENTRY/'code'and Path(__file__).resolve()==code/HELPERS[0]
        and out==root/'results'/('hand-mask-infer-'+revision)/'predictions','Exact source/native namespace required')
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==frontend.binding.IMAGE
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and int(next(s.split()[1]for s in Path('/proc/self/status').read_text().splitlines()if s.startswith('CapEff:')),16)==0,
        'Actual cap-drop offline native CUDA container required')
    rt.require(not any((root/p).exists()for p in('data','vendor',scan.profile('v2')['base']+'/eval_private',
        scan.profile('v2')['base']+'/report.json'))and not Path('/opt/mediapipe-task').exists(),'No private/task/challenge mounts')
    source=rt.source(root,code,revision,ENTRY,HELPERS);rt.require(source==proof['source_binding'],'Own source changed')
    rt.require(out.is_dir()and out.stat().st_uid==0 and stat.S_IMODE(out.stat().st_mode)==0o700 and not tuple(out.iterdir()),'Fresh native leaf required')
    frozen={Path(p):pin for p,pin in proof['frozen'].items()}
    report=dict(stage='public_full_t_paired_hand_sam2',status='fail',phase='preflight',producer_revision=revision,
        source_binding=source,protocol_identity=proof['protocol_identity'],manifest_identity=proof['manifest'],
        scan_pins=proof['scan_pins'],image_id=frontend.binding.IMAGE,budget_seconds=BUDGET,
        private_values_read=False,quality_verified=False,identity_accepted=False,contacts_inferred=False,
        geometry_inferred=False,all_original_frames=True,network='none',device='cuda',sam2_native_postprocessing=True,
        encoder_attempts=0,encoder_completed=0,a_attempts=0,a_completed=0,b_attempts=0,b_completed=0,outputs=[])
    model=None;installed=None;failure=None
    try:
        rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'ALL scans/RGB frozen before values')
        _,pin=protocol(rt,code);rt.require(pin==proof['protocol_identity'],'Paired protocol changed')
        manifest,_=scan.public_inputs(rt,root,code,proof['acquisition'],'v2')
        model,assets=frontend.frontend_proof(code);rt.require(frontend_json(rt,model)==proof['frontend']and assets==proof['assets'],'Frontend source/model proof differs');check()
        import numpy as np
        import torch
        from PIL import Image
        from importlib import metadata
        rt.require(metadata.version('numpy')=='1.26.3'and metadata.version('torch')=='2.5.1+cu124','Measured native runtime required')
        rt.require(torch.cuda.is_available()and 'H100'in torch.cuda.get_device_name(),'Native H100 required')
        torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        installed=frontend.binding.installed_sam2(model)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(frontend.binding.DEST/'weights/sam2/sam2.1_hiera_large.pt'),device='cuda',mode='eval',apply_postprocessing=True))
        class Native:
            def set_image(self,rgb):
                report['encoder_attempts']+=1;check()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):predictor.set_image(rgb)
                torch.cuda.synchronize();report['encoder_completed']+=1;check()
            def predict(self,**kw):
                arm='b'if 'point_coords'in kw else 'a';report[arm+'_attempts']+=1;check()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):value=predictor.predict(**kw)
                torch.cuda.synchronize();report[arm+'_completed']+=1;check();return value
        observations=[]
        for s in manifest['sequences']:
            path=Path(proof['predictions'])/f"sequence_{s['sequence_lex_index']:03d}.npz"
            observations.append(restoration.reload_observations(rt,np,path,Path(str(path)+'.json'),s['frames'],{'scan':proof['scan_pins']}));check()
        def read_rgb(row):
            path=root/scan.profile('v2')['base']/'inputs'/row['file']
            rt.require(rt.identity(path,16<<20)=={k:row[k]for k in('bytes','sha256')},'JPEG changed before decode')
            with Image.open(path)as image:
                rt.require(image.format=='JPEG'and image.mode=='RGB'and image.size==(640,480),'Original RGB required')
                return np.asarray(image).copy()
        report['phase']='native_paired_masks';infer_frames(rt,np,manifest,observations,Native(),out,read_rgb,check,report['outputs'])
        active=sum(r['usable_boxes']>0 for r in report['outputs']);positive=sum(r['b_interventions']>0 for r in report['outputs'])
        rt.require(report['encoder_attempts']==report['encoder_completed']==report['a_attempts']==report['a_completed']==active
            and report['b_attempts']==report['b_completed']==positive,'Exact shared-encode native paired call counts')
        rt.require({p.name for p in out.iterdir()}=={r['file']for r in report['outputs']},'No omitted/extra output');report.update(status='pass',phase='complete')
    except BaseException as error:failure=error;report['error_type']=type(error).__name__
    finally:
        try:
            rt.require(rt.source(root,code,revision,ENTRY,HELPERS)==source and rt.identity(proof_path,4<<20)==proof_pin
                and all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'Public/source changed after inference')
            if model is not None:rt.require(frontend.frontend_proof(code)==(model,proof['assets']),'Assets changed')
            if installed is not None:rt.require(frontend.binding.installed_sam2(model)==installed,'Installed SAM2 changed')
            rt.require(all(rt.identity(out/r['file'],16<<20)=={k:r[k]for k in('bytes','sha256')}for r in report['outputs']),'Output changed');check()
            report['source_rehashed_after']=True
        except BaseException as error:failure=failure or error;report['post_error_type']=type(error).__name__
        report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-start)
        rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Paired native inference failed; retained own receipt')from None
    return report


def run(root,code,revision):
    import dexycb_identity_infer as frontend
    rt=scan.runtime(code);start=time.monotonic();deadline=start+BUDGET
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'
        and root==ROOT and Path(__file__).resolve()==code/HELPERS[0],'Actual immutable VM02 host required')
    source=rt.source(root,code,revision,ENTRY,HELPERS);_,protocol_pin=protocol(rt,code)
    evidence,pins,predictions,frozen=authenticate_scan(rt,root,code)
    model,assets=frontend.frontend_proof(code,live=True);remaining=deadline-time.monotonic()
    rt.require(remaining>0,'Inclusive source prehash exhausted budget')
    name='world-reward-hand-mask-infer-'+revision[:12]
    rt.require(not rt.control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned container name occupied')
    control=root/'results'/('hand-mask-infer-'+revision);rt.canonical(control);rt.require(not control.exists(),'Fresh owned inference namespace required')
    out=control/'predictions'
    proof=dict(source_binding=source,protocol_identity=protocol_pin,manifest=evidence['acquisition']['manifest'],
        acquisition=evidence['acquisition'],scan_pins=pins,predictions=str(predictions),frontend=frontend_json(rt,model),assets=assets,
        frozen={str(p):pin for p,pin in{**evidence['public'],**frozen}.items()if p!=code/SCAN_PINS
                and p!=predictions.parent/'report.json'},remaining_seconds=remaining)
    raw_proof=(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n').encode()
    control.mkdir(mode=0o700);out.mkdir(mode=0o700)
    rt.write(control/'native-proof.json',raw_proof);proof_pin=rt.identity(control/'native-proof.json')
    report=dict(stage='hand_mask_infer_host_seal',status='fail',
        producer_revision=revision,source_binding=source,protocol_identity=protocol_pin,scan_pins=pins,
        image_id=frontend.binding.IMAGE,private_values_read=False,quality_verified=False,owned_cleanup_verified=False,source_rehashed_after=False)
    lock=root/'jobs/.world-reward-h100.lock';state=rt.canonical(lock).lstat()
    rt.require(stat.S_ISREG(state.st_mode)and state.st_nlink==1,'Existing cooperative lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);os.dup2(fd,9)
    if fd!=9:os.close(fd)
    owned=False;failure=None
    def idle():rt.require(not rt.control(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(),'GPU compute active')
    def lock_check():rt.require((os.fstat(9).st_dev,os.fstat(9).st_ino)==(state.st_dev,state.st_ino)==(lock.lstat().st_dev,lock.lstat().st_ino),'Lock changed')
    try:
        lock_check();idle()
        # flock must inherit this original descriptor; control() deliberately does not.
        subprocess.run(['flock','--nonblock','9'],check=True,pass_fds=(9,),timeout=5,capture_output=True)
        lock_check();idle()
        mounts=native_mounts(code,revision,[predictions/n for n in pins['outputs']])
        args=['docker','run','--rm','--name',name,'--cidfile',str(control/'.container.cid'),
            '--label','world_reward.mediapipe_cpu.owner='+revision,'--gpus','all','--network','none','--read-only',
            '--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','16g',
            '--tmpfs','/tmp:rw,nosuid,size=512m']
        for p in mounts:args+=['--mount',f'type=bind,src={p},dst={p},readonly']
        args+=['--mount',f'type=bind,src={control}/native-proof.json,dst=/opt/hand-mask-proof.json,readonly',
            '--mount',f'type=bind,src={out},dst={out}','--entrypoint','/usr/bin/env',frontend.binding.IMAGE,'-i',
            'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','XDG_CACHE_HOME=/tmp','HF_HUB_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
            'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+frontend.binding.IMAGE,
            '/opt/conda/bin/python','-I','-B',str(code/HELPERS[0]),'--native',str(out),str(proof_pin['bytes']),proof_pin['sha256']]
        owned=True
        with(control/'native.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400);result=subprocess.run(args,stdout=log,stderr=log,timeout=max(.001,deadline-time.monotonic()))
        rt.require(result.returncode==0,'Native paired SAM2 stage failed')
        native=rt.pinned(out/'report.json',rt.identity(out/'report.json'),4<<20)
        rt.require(native['status']=='pass'and native['phase']=='complete'and native['source_binding']==source
            and native.get('source_rehashed_after')is True,'Complete native source-bound PASS required')
        report['native_report']=native;report['outputs']={p.name:rt.identity(p,16<<20)for p in out.iterdir()}
    except BaseException as error:failure=error;report['error_type']=type(error).__name__
    finally:
        try:
            if owned:rt.cleanup(control/'.container.cid',name,model['child_image'],revision);idle()
            report['owned_cleanup_verified']=True;lock_check()
        except BaseException as error:failure=failure or error;report['cleanup_error_type']=type(error).__name__
        finally:os.close(9)
        try:
            rt.require(rt.source(root,code,revision,ENTRY,HELPERS)==source and authenticate_scan(rt,root,code)==(evidence,pins,predictions,frozen)
                and frontend.frontend_proof(code,live=True)==(model,assets),'Posthash original inputs/sources/assets changed')
            if 'outputs'in report:rt.require(all(rt.identity(out/n,16<<20)==p for n,p in report['outputs'].items()),'Outputs changed before host seal')
            rt.require(time.monotonic()<=deadline,'Inclusive all3 posthash budget exceeded');report['source_rehashed_after']=True
        except BaseException as error:failure=failure or error;report['post_error_type']=type(error).__name__
        report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-start)
        rt.write(control/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Owned hand-mask stage failed; inspect sealed receipt')from None
    return report


def main():
    args=sys.argv[1:];rt=scan.runtime(Path(os.environ['WR_CODE']))
    if args[:1]==['--native']and len(args)==4:
        return run_native(rt,ROOT,Path(os.environ['WR_CODE']),Path(args[1]),Path('/opt/hand-mask-proof.json'),dict(bytes=int(args[2]),sha256=args[3]))
    if args:raise ValueError('No arbitrary cohort, prompt or threshold arguments')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'])


if __name__=='__main__':
    try:main()
    except Exception as error:print(json.dumps(dict(stage='hand_mask_infer',status='fail',error_type=type(error).__name__)));sys.exit(1)

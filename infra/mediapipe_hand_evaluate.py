"""Private CPU diagnostic only, after ALL original hand predictions are sealed."""
import importlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import zipfile

ROOT=Path('/srv/scenesmith/world-reward');ENTRY='run_mediapipe_hand_evaluate'
SCAN_PINS='configs/mediapipe_hand_scan_pins.json';BUDGET=300
HELPERS=('infra/mediapipe_hand_evaluate.py','infra/run_mediapipe_hand_evaluate.sh',
 'infra/mediapipe_hand_scan.py','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/hand_evaluation.py','src/world_reward/hand_observations.py',
 'configs/mediapipe_hand_scan_pins.json','configs/mediapipe_cpu_runtime_pins.json',
 'configs/dexycb_hand_acquire_pins.json','configs/dexycb_hand_protocol_v1.json')
# Publisher README 64551b001d360ad83bc383157a559ec248fb9100 L157/161,
# 40820 B SHA e19797f352bb5615b43b5a3ed4a6a193cee4eee2cd1cd1f5859615165081bb7c:
# seg uint8[H,W]; joint_2d float32[1,21,2]. Only axis0[0] is projected.


def modules(code):
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
    scan=importlib.import_module('mediapipe_hand_scan');rt=scan.runtime(code)
    rt.require(Path(scan.__file__).resolve()==code/'infra/mediapipe_hand_scan.py','Actual scan helper origin required')
    return scan,rt


def private_inventory(rt,root,manifest,retained):
    files={};folders=[];base=root/'validation/dexycb_hand_v1/eval_private'
    for sequence in manifest['sequences']:
        relative=f"{sequence['subject']}/{sequence['sequence']}/{sequence['camera']}"
        folder=rt.canonical(base/relative);folders.append(folder)
        rt.require(stat.S_IMODE(folder.stat().st_mode)==0o700 and folder.stat().st_uid>0,'Original private directory owner/mode required')
        names={f'labels_{t:06d}.npz'for t in range(sequence['frames'])}
        rt.require({p.name for p in folder.iterdir()}==names,'Only selected complete original label files allowed')
        for name in sorted(names):
            path=folder/name;pin=retained.get(relative+'/'+name)
            rt.require(pin is not None and stat.S_IMODE(path.stat().st_mode)==0o400 and path.stat().st_uid==folder.stat().st_uid
                and rt.identity(path,16<<20)==pin,'Original opaque private label bytes/permissions differ')
            files[path]=pin
    expected={str(p.relative_to(base))for p in files}
    rt.require({n for n in retained if n.endswith('.npz')}==expected,'No private cohort substitution allowed')
    return files,folders


def authenticate(scan,rt,root,code):
    evidence=scan.authenticate(rt,root,code)  # stdlib hashes only; no models loaded.
    pins,pins_pin=scan.pins(rt,code,SCAN_PINS,'world_reward.mediapipe_hand_scan_pins.v1',('report',))
    revision=pins['producer_revision'];original=root/'jobs'/revision/scan.ENTRY/'code'
    source=rt.source(root,original,revision,scan.ENTRY,scan.HELPERS)
    control=root/'results'/('mediapipe-hand-scan-'+revision)
    host=rt.pinned(control/'report.json',pins['report'],4<<20)
    expected=dict(stage='mediapipe_hand_scan_host_seal',status='pass',producer_revision=revision,source_binding=source,
        acquisition_pins=evidence['acquisition'],runtime_pins=evidence['runtime'],source_rehashed_after=True,
        owned_cleanup_verified=True,gpu_used=False,private_values_read=False,accuracy_verified=False)
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items()),'ALL original scans must have genuine sealed host PASS')
    rt.require(type(host.get('elapsed_seconds'))in(int,float)and 0<host['elapsed_seconds']<=600,'Original all-three scan budget differs')
    outputs=host.get('outputs',{});names={'report.json',*(f'sequence_{i:03d}'+s for i in(4,39,74)for s in('.npz','.npz.json'))}
    out=control/'predictions';rt.require(set(outputs)==names and {p.name for p in out.iterdir()}==names,'All seven prediction artifacts required')
    frozen={control/'report.json':pins['report'],code/SCAN_PINS:pins_pin}
    for name,pin in outputs.items():
        path=out/name;rt.require(rt.identity(path,32<<20)==pin,'Frozen native prediction changed');frozen[path]=pin
    native=rt.strict((out/'report.json').read_bytes())
    rt.require(native==host.get('native_report')and native.get('status')=='pass'and native.get('phase')=='complete'
        and native.get('producer_revision')==revision and native.get('source_binding')==source
        and native.get('native_calls')==len(evidence['manifest']['images'])and native.get('native_graphs')==1
        and all(native.get(k)is False for k in('gpu_used','private_values_read','accuracy_verified','availability_is_visibility',
            'handedness_is_detection_confidence','actor_identity_inferred')),'Original complete native scan report differs')
    declared=native.get('outputs',[])
    rt.require(len(declared)==3 and [(r['sequence_lex_index'],r['frames'],r['file'])for r in declared]==[
        (s['sequence_lex_index'],s['frames'],f"sequence_{s['sequence_lex_index']:03d}.npz")for s in evidence['manifest']['sequences']],
        'Native output original sequence/frame declaration differs')
    acq=rt.pinned(root/scan.BASE/'report.json',evidence['acquisition']['report'],4<<20)
    private,folders=private_inventory(rt,root,evidence['manifest'],acq['retained_files']);frozen.update(private)
    manifest_path=root/scan.BASE/'inputs/manifest.json';frozen[manifest_path]=evidence['acquisition']['manifest']
    return dict(evidence=evidence,scan_pins=pins,predictions=out,frozen=frozen,private_folders=folders,manifest_path=manifest_path)


def reload_observations(rt,np,path,categories_path,frames,refs):
    from world_reward.hand_observations import HandObservations,normalized_hand_instances
    with np.load(path,allow_pickle=False)as archive:
        keys={'frame_index','frame_offsets','capacity_saturation','normalized_xyz','native_world_xyz','pixel_xy','image_z',
              'xy_supported','image_z_supported','world_supported','handedness_scores'}
        rt.require(set(archive.files)==keys,'Exact native numeric artifacts required');a={k:archive[k]for k in keys}
    rt.require(all(not v.dtype.hasobject for v in a.values())and a['frame_index'].dtype==np.int64
        and np.array_equal(a['frame_index'],np.arange(frames,dtype=np.int64))and a['frame_offsets'].dtype==np.int64
        and a['frame_offsets'].shape==(frames+1,)and a['frame_offsets'][0]==0,'Native full-T indices/offsets differ')
    counts=np.diff(a['frame_offsets']);n=int(a['frame_offsets'][-1])
    rt.require(np.all((counts>=0)&(counts<=4))and n>=0,'Native ragged capacity differs')
    shapes={'normalized_xyz':(n,21,3),'native_world_xyz':(n,21,3),'pixel_xy':(n,21,2),'image_z':(n,21,1),
            'xy_supported':(n,21),'image_z_supported':(n,21),'world_supported':(n,21),'handedness_scores':(n,),
            'capacity_saturation':(frames,)}
    for name,shape in shapes.items():
        rt.require(a[name].shape==shape and a[name].dtype==(np.bool_ if name.endswith('supported')or name=='capacity_saturation'else np.float64),
                   'Native array shape/dtype changed')
    rt.require(np.array_equal(a['capacity_saturation'],counts==4),'Native capacity flags differ')
    category=rt.strict(categories_path.read_bytes());rt.require(set(category)=={'native_categories'}and len(category['native_categories'])==frames,'Native category timeline differs')
    bank=[]
    for t in range(frames):
        lo,hi=map(int,a['frame_offsets'][t:t+2]);rows=category['native_categories'][t]
        rt.require(type(rows)is list and len(rows)==hi-lo and all(type(r)is dict and set(r)=={'category_name','display_name','index'}
            and r['category_name']in('Left','Right')and type(r['index'])is int and r['index']in(0,1)
            and type(r['display_name'])is str and len(r['display_name'])<=256 for r in rows),'Native categories invalid')
        f=normalized_hand_instances(a['normalized_xyz'][lo:hi],(480,640),native_world_xyz=a['native_world_xyz'][lo:hi],
            native_handedness_labels=tuple(r['category_name']for r in rows),native_handedness_scores=a['handedness_scores'][lo:hi])
        for name,value in(('pixel_xy',f.original_xy.values),('image_z',f.image_z.values),('xy_supported',f.original_xy.supported),
                           ('image_z_supported',f.image_z.supported),('world_supported',f.hand_centred_world_xyz.supported)):
            rt.require(value.tobytes()==a[name][lo:hi].tobytes(),'Lossless native coordinate/support restoration failed')
        bank.append(f)
    return HandObservations(a['frame_index'],(480,640),tuple(bank),'MediaPipe IMAGE CPU frozen native outputs',refs)


def summary(result):
    from dataclasses import asdict
    return dict(counts=asdict(result.counts),joint_weighted_epe_pixels=result.joint_weighted_epe_pixels,
                frame_mean_epe_pixels=result.frame_mean_epe_pixels,diagnostic_status=result.diagnostic_status)


def private_field(rt,np,path,name):
    """Bound only the two permitted NPY fields before allocating their arrays."""
    rt.require(name in('seg','joint_2d'),'Only permitted private diagnostic fields may be decoded')
    shape,dtype=((480,640),np.dtype('uint8'))if name=='seg'else((1,21,2),np.dtype('float32'))
    with zipfile.ZipFile(path)as archive:
        member=name+'.npy';rt.require(archive.namelist().count(member)==1,'Unique permitted private NPY field required')
        info=archive.getinfo(member);rt.require(info.file_size<=dtype.itemsize*int(np.prod(shape))+4096,'Private NPY expanded-byte cap')
        with archive.open(member)as stream:
            version=np.lib.format.read_magic(stream)
            rt.require(version in((1,0),(2,0)),'Known non-pickle NPY header required')
            header=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
            actual,fortran,kind=header(stream);rt.require(actual==shape and kind==dtype and not kind.hasobject,'Publisher private NPY shape/dtype required')
    with np.load(path,allow_pickle=False)as archive:return archive[name]


def run_native(rt,root,code,out,proof_path,proof_pin):
    revision=os.environ['WR_CODE_REVISION']
    rt.require(root==ROOT and code==root/'jobs'/revision/ENTRY/'code'
        and out==root/'results'/('mediapipe-hand-evaluate-'+revision)/'diagnostics'
        and Path(__file__).resolve()==code/HELPERS[0],'Exact native evaluator source/output namespace required')
    rt.require(sys.platform=='linux'and os.geteuid()==0 and Path(sys.prefix)==rt.VENV and sys.version_info[:2]==(3,11)
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Qualified offline CP311 CPU required')
    caps=int(next(line.split()[1]for line in Path('/proc/self/status').read_text().splitlines()if line.startswith('CapEff:')),16)
    rt.require(caps==4,'Only explicitly necessary DAC_READ_SEARCH capability permitted')
    proof=rt.pinned(proof_path,proof_pin,4<<20);started=time.monotonic();deadline=started+proof['remaining_seconds']
    def check():rt.require(time.monotonic()<=deadline,'Inclusive private evaluation budget exceeded')
    rt.require(rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,HELPERS)==proof['source_binding'],'Native original current source differs')
    rt.require(out.is_dir()and stat.S_IMODE(out.stat().st_mode)==0o700 and out.stat().st_uid==0 and not tuple(out.iterdir()),'Fresh owned private diagnostics required')
    frozen={Path(p):pin for p,pin in proof['frozen'].items()}
    rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'ALL frozen predictions/labels before any value read required');check()
    import numpy as np
    from world_reward.hand_evaluation import evaluate_hand_clip,pool_hand_evaluations
    rt.require(Path(np.__file__).is_relative_to(rt.VENV)and 'mediapipe'not in sys.modules and sys.prefix!=sys.base_prefix
        and 'include-system-site-packages = false'in(rt.VENV/'pyvenv.cfg').read_text().lower(),'No model import or base-site numpy permitted')
    manifest=rt.pinned(Path(proof['manifest_path']),proof['manifest_pin'],16<<20)
    bank=[]
    for s in manifest['sequences']:
        base=Path(proof['predictions'])/f"sequence_{s['sequence_lex_index']:03d}.npz"
        bank.append(reload_observations(rt,np,base,Path(str(base)+'.json'),s['frames'],{'scan':proof['scan_pins']}));check()
    results=[]
    for s,obs in zip(manifest['sequences'],bank):
        folder=Path(proof['private_folders'][str(s['sequence_lex_index'])]);joints=[]
        for t in range(s['frames']):
            check();path=folder/f'labels_{t:06d}.npz';rt.require(rt.identity(path,16<<20)==frozen[path],'Private label bytes changed')
            # Only these two NPY members are decoded. No joint_3d/pose/MANO/K.
            xy=private_field(rt,np,path,'joint_2d')
            rt.require(xy.dtype==np.float32 and xy.shape==(1,21,2),'Publisher joint_2d native shape/dtype required')
            joints.append(xy[0])
        def segments():
            for t in range(s['frames']):
                check();path=folder/f'labels_{t:06d}.npz';rt.require(rt.identity(path,16<<20)==frozen[path],'Private segmentation bytes changed')
                seg=private_field(rt,np,path,'seg')
                rt.require(seg.dtype==np.uint8 and seg.shape==(480,640),'Publisher seg native shape/dtype required');yield seg
        r=evaluate_hand_clip(s['sequence'],obs,obs.frame_index,segments(),np.stack(joints));results.append(r)
        diagnostics=dict(frame_index=r.frame_index.tolist(),positive=r.annotated_positive.tolist(),unlabelled=r.unlabelled.tolist(),
          unique=r.unique_association.tolist(),missed=r.missed.tolist(),ambiguous=r.ambiguous.tolist(),matched_prediction=r.matched_prediction.tolist(),
          prediction_count=r.prediction_count.tolist(),valid_gt_joint_count=r.valid_gt_joints.sum(axis=1).tolist(),
          frame_epe_pixels=r.frame_epe_pixels,joint_error_pixels=r.joint_error_pixels)
        rt.write(out/f"sequence_{s['sequence_lex_index']:03d}.json",(json.dumps(diagnostics,allow_nan=False)+'\n').encode());check()
    pooled=pool_hand_evaluations(tuple(results));report=dict(stage='mediapipe_hand_private_evaluation',status='pass',phase='complete',producer_revision=revision,
      source_binding=proof['source_binding'],per_clip=[dict(clip=r.clip,**summary(r))for r in results],pooled=summary(pooled),
      verified_joint_indices=[0,*range(5,21)],private_fields_decoded=['seg','joint_2d'],private_values_read=True,
      prediction_values_modified=False,models_loaded=False,native_graphs=0,gpu_used=False,accuracy_threshold_calibrated=False,
      adoption=False,quality_claim=False)
    rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items())and rt.identity(proof_path,4<<20)==proof_pin
        and rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,HELPERS)==proof['source_binding'],'Source/predictions/private bytes changed after evaluation')
    check();rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());check();return report


def run(root,code,revision):
    scan,rt=modules(code);rt.require(root==ROOT and sys.platform=='linux'and os.geteuid()==0
        and os.uname().nodename=='world-reward-ncc-h100-02'and Path(__file__).resolve()==code/HELPERS[0],'Actual immutable VM02 adapter required')
    started=time.monotonic();deadline=started+BUDGET;source=rt.source(root,code,revision,ENTRY,HELPERS)
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact readonly snapshot parent inventory required')
    evidence=authenticate(scan,rt,root,code);rt.require(time.monotonic()<deadline,'Inclusive source/ALL prediction prehash deadline exceeded')
    control=root/'results'/('mediapipe-hand-evaluate-'+revision);rt.canonical(control);rt.require(not control.exists(),'Fresh evaluator namespace required')
    control.mkdir(mode=0o700);out=control/'diagnostics';out.mkdir(mode=0o700)
    private={str(s['sequence_lex_index']):str(folder)for s,folder in zip(evidence['evidence']['manifest']['sequences'],evidence['private_folders'])}
    proof=dict(source_binding=source,remaining_seconds=deadline-time.monotonic(),scan_pins=evidence['scan_pins'],
        predictions=str(evidence['predictions']),private_folders=private,manifest_path=str(evidence['manifest_path']),
        manifest_pin=evidence['evidence']['acquisition']['manifest'],frozen={str(p):pin for p,pin in evidence['frozen'].items()})
    rt.write(control/'proof.json',(json.dumps(proof,sort_keys=True)+'\n').encode());proof_pin=rt.identity(control/'proof.json',4<<20)
    image=evidence['evidence']['image'];name='world-reward-mediapipe-hand-evaluate-'+revision[:12];owned=False;failure=None;log_inode=None
    report=dict(stage='mediapipe_hand_evaluation_host_seal',status='fail',producer_revision=revision,source_binding=source,
        scan_pins=evidence['scan_pins'],budget_seconds=BUDGET,budget_scope='source_all_prediction_private_hash_evaluate_serialize_posthash',
        private_values_read=None,private_value_reads_certified=False,models_loaded=False,gpu_used=False,adoption=False,quality_claim=False)
    def interrupted(*_):raise TimeoutError('Private evaluation interrupted; owned cleanup only')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        rt.require(not rt.control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Evaluator container namespace occupied')
        command=['docker','run','--rm','--name',name,'--cidfile',str(control/'.container.cid'),'--label','world_reward.mediapipe_cpu.owner='+revision,
          '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--cap-add','DAC_READ_SEARCH','--security-opt','no-new-privileges',
          '--cpus','4','--memory','8g','--tmpfs','/tmp:rw,nosuid,size=64m']
        mounts=[(code.parent,code.parent),(control/'proof.json',Path('/opt/hand-evaluation-proof.json')),
                (evidence['manifest_path'],evidence['manifest_path']),*[(p,p)for p in evidence['private_folders']],
                *[(p,p)for p in evidence['frozen']if p.parent==evidence['predictions']],
                (root/'results'/('mediapipe-hand-scan-'+evidence['scan_pins']['producer_revision'])/'report.json',
                 root/'results'/('mediapipe-hand-scan-'+evidence['scan_pins']['producer_revision'])/'report.json')]
        for src,dst in mounts:command+=['--mount',f'type=bind,src={src},dst={dst},readonly']
        command+=['--mount',f'type=bind,src={out},dst={out}','--entrypoint','/usr/bin/env',image['Id'],'-i',
          f'PATH={rt.VENV}/bin:/usr/bin:/bin','HOME=/nonexistent','CUDA_VISIBLE_DEVICES=-1','JAX_PLATFORMS=cpu',
          'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',f'WR_CODE_REVISION={revision}',
          f'{rt.VENV}/bin/python','-I','-B',str(code/HELPERS[0]),'--native',str(root),str(code),str(out),str(proof_pin['bytes']),proof_pin['sha256']]
        owned=True;remaining=deadline-time.monotonic();rt.require(remaining>0,'No private inference budget remains')
        with (control/'.native-output').open('xb')as log:
            os.fchmod(log.fileno(),0o400);s=os.fstat(log.fileno());log_inode=(s.st_dev,s.st_ino)
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=False)
        rt.require(result.returncode==0,'Private native evaluator failed');(control/'.native-output').unlink()
        native=rt.strict((out/'report.json').read_bytes())
        expected=dict(stage='mediapipe_hand_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=source,private_fields_decoded=['seg','joint_2d'],private_values_read=True,native_graphs=0,
            prediction_values_modified=False,models_loaded=False,gpu_used=False,accuracy_threshold_calibrated=False,adoption=False,quality_claim=False)
        rt.require(all(type(native.get(k))is type(v)and native[k]==v for k,v in expected.items()),'Private evaluation receipt differs')
        report['evaluation']=native;report['private_values_read']=True;report['private_value_reads_certified']=True
        rt.require({p.name for p in out.iterdir()}=={'report.json',*(f'sequence_{i:03d}.json'for i in(4,39,74))},'Exact private diagnostic outputs required')
        report['diagnostics']={p.name:rt.identity(p,4<<20)for p in out.iterdir()}
    except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        try:
            if owned:rt.cleanup(control/'.container.cid',name,image,revision)
            log=control/'.native-output'
            if log.exists():
                rt.canonical(log);s=log.lstat();rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1
                    and (s.st_dev,s.st_ino)==log_inode,'Refuse foreign native-log cleanup');log.unlink()
            report['owned_cleanup_verified']=True
            rt.require(rt.source(root,code,revision,ENTRY,HELPERS)==source and authenticate(scan,rt,root,code)==evidence,'Original source/scan/private bytes changed')
            rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Snapshot parent changed after evaluation')
            rt.require('diagnostics'in report and all(rt.identity(out/n,4<<20)==pin for n,pin in report['diagnostics'].items()),'Diagnostics changed before seal')
            rt.require(time.monotonic()<=deadline,'Inclusive private evaluation posthash budget exceeded');report['source_rehashed_after']=True
        except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
        report.update(status='fail'if failure else 'pass',elapsed_seconds=time.monotonic()-started)
        rt.write(control/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Private hand diagnostic failed; sealed historical evidence retained')from None
    return report


def main():
    if sys.argv[1:2]==['--native']:
        if len(sys.argv)!=7:raise ValueError('Exact internal arguments required')
        root,code,out=map(Path,sys.argv[2:5]);_,rt=modules(code)
        return run_native(rt,root,code,out,Path('/opt/hand-evaluation-proof.json'),dict(bytes=int(sys.argv[5]),sha256=sys.argv[6]))
    if len(sys.argv)!=1:raise ValueError('No cohort/threshold/model arguments permitted')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps(dict(stage='mediapipe_hand_private_evaluation',status='fail',error_type=type(exc).__name__)));sys.exit(1)

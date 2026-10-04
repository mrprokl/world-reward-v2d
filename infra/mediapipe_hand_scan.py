"""One full-T external RGB scan. Availability is not identity or accuracy."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path('/srv/scenesmith/world-reward');ENTRY='run_mediapipe_hand_scan'
BASE='validation/dexycb_hand_v1';RUNTIME_PINS='configs/mediapipe_cpu_runtime_pins.json'
ACQUIRE_PINS='configs/dexycb_hand_acquire_pins.json';PROTOCOL='configs/dexycb_hand_protocol_v1.json'
PROTOCOL_PIN=dict(bytes=1940,sha256='318ed43325545be3d3a1e87b1c50fd0bf44e13ce03752346f731dd1507be599c')
HELPERS=('infra/mediapipe_hand_scan.py','infra/run_mediapipe_hand_scan.sh','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/hand_scan.py','src/world_reward/hand_observations.py',PROTOCOL,RUNTIME_PINS,ACQUIRE_PINS)
BUDGET=600


PROFILE_SUBJECTS={'v1':'20200820-subject-03','v2':'20200903-subject-04','v3':'20200908-subject-05'}


def profile(cohort='v1'):
    if cohort not in PROFILE_SUBJECTS:raise ValueError('Only explicitly frozen v1/v2/v3 cohorts permitted')
    return dict(base='validation/dexycb_hand_'+cohort,subject=PROFILE_SUBJECTS[cohort],
        protocol='configs/dexycb_hand_protocol_'+cohort+'.json',
        acquire=ACQUIRE_PINS if cohort=='v1'else 'configs/dexycb_hand_acquire_'+cohort+'_pins.json',
        manifest_schema='world-reward-dexycb-hand-rgb-v1')


def source_helpers(cohort='v1'):
    p=profile(cohort)
    return tuple(p['protocol']if n==PROTOCOL else p['acquire']if n==ACQUIRE_PINS else n for n in HELPERS)


def control_path(root,revision,cohort='v1'):
    profile(cohort)
    return root/'results'/('mediapipe-hand-scan-'+(cohort+'-'if cohort!='v1'else '')+revision)


def runtime(code):
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
    module=importlib.import_module('mediapipe_cpu_runtime_verify')
    module.require(Path(module.__file__).resolve()==code/'infra/mediapipe_cpu_runtime_verify.py','Actual reusable helper origin required')
    return module


def pins(rt,code,name,schema,keys):
    path=code/name;identity=rt.identity(path,16<<10);value=rt.strict(path.read_bytes())
    rt.require(type(value)is dict and set(value)==set(keys)|{'schema','producer_revision'}and value['schema']==schema
               and re.fullmatch('[0-9a-f]{40}',str(value['producer_revision'])),'Exact independently frozen pin schema required')
    return value,identity


def public_inputs(rt,root,code,acq,cohort='v1'):
    p=profile(cohort);base=p['base']
    manifest=rt.pinned(root/base/'inputs/manifest.json',acq['manifest'],16<<20)
    rt.require(set(manifest)=={'schema','subject','sequences','images','source_archive','license','timestamps_available',
        'training_overlap_verified','challenge_overlap_verified'}and manifest['schema']==p['manifest_schema']
        and manifest['subject']==p['subject']and manifest['license']=='CC-BY-NC-4.0'
        and all(manifest[k]is False for k in('timestamps_available','training_overlap_verified','challenge_overlap_verified')),
        'Frozen external RGB-only manifest required')
    seqs=manifest['sequences'];rows=manifest['images'];rt.require(type(seqs)is list and len(seqs)==3 and type(rows)is list,'All3 sequences required')
    wanted=[]
    for index,sequence in zip((4,39,74),seqs):
        rt.require(set(sequence)=={'subject','sequence','sequence_lex_index','camera','frames'}and sequence['subject']==manifest['subject']
            and sequence['sequence_lex_index']==index and sequence['camera']=='836212060125'
            and re.fullmatch('[0-9]{8}_[0-9]{6}',str(sequence['sequence']))
            and type(sequence['frames'])is int and 1<=sequence['frames']<=10000,'Original complete sequence declaration required')
        for frame in range(sequence['frames']):
            wanted.append((sequence,frame,f'sequence_{index:03d}_frame_{frame:06d}.jpg'))
    rt.require(len(rows)==len(wanted),'Every original frame required')
    frozen={root/base/'inputs/manifest.json':acq['manifest']}
    for row,(sequence,frame,name)in zip(rows,wanted):
        rt.require(set(row)=={'file','bytes','sha256','width','height','sequence','sequence_lex_index','camera','frame_position','source_frame_id'}
            and row['file']==name and row['sequence']==sequence['sequence']and row['sequence_lex_index']==sequence['sequence_lex_index']
            and row['camera']==sequence['camera']and type(row['source_frame_id'])is int and row['source_frame_id']==frame
            and type(row['frame_position'])is int and row['frame_position']==frame and row['width']==640 and row['height']==480,
            'Original RGB IDs/grid must remain unchanged')
        path=root/base/'inputs'/name;pin={k:row[k]for k in('bytes','sha256')}
        rt.require(rt.identity(path,16<<20)==pin,'Original JPEG bytes differ');frozen[path]=pin
    rt.require({p.name for p in(root/base/'inputs').iterdir()}=={'manifest.json',*(r['file']for r in rows)},'No foreign public input allowed')
    return manifest,frozen


def authenticate(rt,root,code,cohort='v1'):
    p=profile(cohort);base=p['base'];protocol_path=p['protocol']
    acq,acq_pin=pins(rt,code,p['acquire'],'world_reward.dexycb_hand_acquire_pins.v1',
                     ('report','manifest','helper')if cohort=='v1'else('report','manifest','helper','protocol'))
    original=root/'jobs'/acq['producer_revision']/'run_dexycb_hand_acquire'/'code'
    protocol_pin=PROTOCOL_PIN if cohort=='v1'else acq['protocol'];protocol=rt.pinned(code/protocol_path,protocol_pin)
    rt.require(protocol['base']==base and protocol['subject']==p['subject']and protocol['schema']=='world-reward-dexycb-hand-acquisition-'+cohort
        and protocol['sequence_lex_indices']==[4,39,74]and protocol['camera']=='836212060125'
        and protocol['all_original_frames']is True and protocol['original_rgb_size']==[640,480], 'Frozen acquisition profile mismatch')
    rt.require(rt.identity(original/protocol_path)==protocol_pin,'Original acquisition protocol differs')
    helper_names=('infra/dexycb_hand_acquire.py','infra/run_dexycb_hand_acquire.sh',protocol_path,
                  *('infra/'+n for n in protocol['helper_pins']))
    snapshot=rt.source(root,original,acq['producer_revision'],'run_dexycb_hand_acquire',helper_names)
    rt.require(snapshot['helpers']['infra/dexycb_hand_acquire.py']==acq['helper'],'Original acquisition producer differs')
    primary={n:rt.identity(root/'vendor/research/dexycb_identity_v1'/n,32<<10)for n in protocol['primary_sources']}
    rt.require(primary=={n:{k:p[k]for k in('bytes','sha256')}for n,p in protocol['primary_sources'].items()},'Primary license/source identity differs')
    original_binding=dict(producer_revision=acq['producer_revision'],protocol=protocol_pin,markers=snapshot['markers'],
                         helpers=protocol['helper_pins'],primary_sources=primary,closure_sha256=snapshot['closure_sha256'])
    if cohort!='v1':original_binding.update(protocol_file=protocol_path,profile=base)
    rt.require(all(snapshot['helpers']['infra/'+n]==p for n,p in protocol['helper_pins'].items()),'Original reusable acquisition helpers differ')
    receipt=rt.pinned(root/base/'report.json',acq['report'],4<<20)
    expected=dict(stage='external_dexycb_hand_rgb_private_byte_acquisition',status='pass',phase='complete',
      producer_revision=acq['producer_revision'],source_before=original_binding,source_rehashed_after=True,
      annotation_values_parsed=False,inference_performed=False,gpu_used=False,disposable_archive_removed=True,sequences=3)
    if cohort!='v1':expected.update(protocol_file=protocol_path,acquisition_profile=base)
    rt.require(all(type(receipt.get(k))is type(v)and receipt[k]==v for k,v in expected.items())
        and receipt.get('public_manifest')==acq['manifest'],'Genuine unmodified acquisition PASS required')
    manifest,files=public_inputs(rt,root,code,acq,cohort)
    rt.require(receipt['frames']==len(manifest['images'])and receipt['selected_sequences']==manifest['sequences']
        and all(receipt['retained_files'].get(row['file'])=={k:row[k]for k in('bytes','sha256')}for row in manifest['images']),
        'Public/full-timeline acquisition lineage differs')
    qualified,qualified_pin=pins(rt,code,RUNTIME_PINS,'world_reward.mediapipe_cpu_runtime_pins.v1',('report','image_id'))
    old=root/'jobs'/qualified['producer_revision']/rt.ENTRY/'code'
    binding=rt.source(root,old,qualified['producer_revision'],rt.ENTRY,rt.HELPERS)
    report=rt.pinned(root/'results'/('mediapipe-cpu-runtime-verify-'+qualified['producer_revision'])/'report.json',qualified['report'],4<<20)
    rt.require(all(report.get(k)is v for k,v in dict(source_rehashed_after=True,owned_cleanup_verified=True,
        owned_build_context_removed=True,gpu_used=False,dataset_read=False,private_values_read=False,quality_verified=False).items())
        and report.get('stage')=='mediapipe_cpu_runtime_verify'and report.get('status')=='pass'and report.get('phase')=='complete'
        and report.get('producer_revision')==qualified['producer_revision']and report.get('source_binding')==binding,
        'Original sealed CPU runtime qualification required')
    proof=rt.prerequisites(root,code)
    rt.require(report['prior_acquisition']==proof['prior']and report['prior_verified_downloads']==proof['verified_downloads']
        and report['dependency_acquisition_pins']==proof['dependency_pins']and report['manifest_identity']==rt.MANIFEST_PIN,
        'Runtime dependency/task lineage differs')
    image=rt.image(qualified['image_id']);rt.qualify_child(report['base_image'],image,qualified['producer_revision'],binding)
    rt.require(image==report['child_image']and image['Id']==qualified['image_id']and rt.image(rt.BASE)==report['base_image'],
               'Actual qualified image/base projection differs')
    native=report['cpu_smoke'];rt.require(native['native_graph_loaded']is True and native['native_graph_closed']is True
        and type(native['detect_calls'])is int and native['detect_calls']==0 and native['task_identity']==proof['task_pin'],
        'Native graph create/close proof required')
    return dict(manifest=manifest,public=files,task=proof['task'],task_pin=proof['task_pin'],image=image,versions=native['versions'],
                acquisition=acq,runtime=qualified,acquisition_pin=acq_pin,runtime_pin=qualified_pin)


def native_callback(detector,mp,np,NativeHandResult):
    def callback(_,rgb):
        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb));count=len(result.hand_landmarks)
        if count>4 or len(result.hand_world_landmarks)!=count or len(result.handedness)!=count:raise ValueError('Native proposal slots differ')
        xyz=lambda rows:np.array([[(p.x,p.y,p.z)for p in hand]for hand in rows],np.float64).reshape(count,21,3)
        labels=[];scores=[];categories=[]
        for row in result.handedness:
            if len(row)!=1:raise ValueError('One native handedness classification per proposal required')
            category=row[0]
            if category.category_name not in('Left','Right')or type(category.index)is not int or not 0<=category.index<=1 \
                or type(category.display_name)is not str or len(category.display_name)>256 \
                or any(ord(c)<32 or ord(c)==127 for c in category.display_name):
                raise ValueError('Native handedness metadata invalid')
            labels.append(category.category_name);scores.append(category.score)
            categories.append(dict(category_name=category.category_name,display_name=category.display_name,index=category.index))
        callback.categories.append(categories)
        return NativeHandResult(xyz(result.hand_landmarks),xyz(result.hand_world_landmarks),tuple(labels),np.array(scores,np.float64))
    callback.categories=[];return callback


def serialize(result,path,np):
    counts=np.array([len(r.normalized_xyz)for r in result.native_results],np.int64);offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
    join=lambda values,shape:np.concatenate(values)if values else np.empty(shape,np.float64)
    frames=result.observations.frames;raw=result.native_results
    arrays=dict(frame_index=result.observations.frame_index,frame_offsets=offsets,capacity_saturation=result.capacity_saturation,
      normalized_xyz=join([r.normalized_xyz for r in raw],(0,21,3)),native_world_xyz=join([r.native_world_xyz for r in raw],(0,21,3)),
      pixel_xy=join([r.original_xy.values for r in frames],(0,21,2)),image_z=join([r.image_z.values for r in frames],(0,21,1)),
      xy_supported=join([r.original_xy.supported for r in frames],(0,21)),image_z_supported=join([r.image_z.supported for r in frames],(0,21)),
      world_supported=join([r.hand_centred_world_xyz.supported for r in frames],(0,21)),
      handedness_scores=join([r.native_handedness_scores for r in raw],(0,)))
    if any(v.dtype.hasobject for v in arrays.values()):raise ValueError('Object pickle forbidden')
    with path.open('xb')as stream:os.fchmod(stream.fileno(),0o400);np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return dict(frames=len(counts),instances=int(counts.sum()),frames_with_proposals=int((counts>0).sum()),
                capacity_saturated_frames=int(result.capacity_saturation.sum()),numerically_available_xy=int(arrays['xy_supported'].sum()))


def run_native(rt,root,code,out,proof_path,proof_pin,cohort='v1'):
    p=profile(cohort);base=p['base'];helpers=source_helpers(cohort)
    rt.require(root==ROOT and code==root/'jobs'/os.environ['WR_CODE_REVISION']/ENTRY/'code'
               and out==control_path(root,os.environ['WR_CODE_REVISION'],cohort)/'predictions'
               and Path(__file__).resolve()==code/HELPERS[0],'Exact native source namespace required')
    proof=rt.pinned(proof_path,proof_pin,64<<10)
    rt.require((cohort=='v1'and 'cohort'not in proof)or(cohort!='v1'and proof.get('cohort')==cohort
        and proof.get('public_base')==base),'Immutable native proof cohort mismatch')
    if cohort!='v1':
        native_acq,_=pins(rt,code,p['acquire'],'world_reward.dexycb_hand_acquire_pins.v1',('report','manifest','helper','protocol'))
        rt.require(native_acq['protocol']==proof['protocol_identity']and native_acq['manifest']==proof['manifest'],
                   'Native independently frozen input/protocol differs')
        protocol=rt.pinned(code/p['protocol'],proof['protocol_identity'])
        rt.require(protocol['base']==base and protocol['subject']==p['subject']and protocol['all_original_frames']is True,
                   'Frozen native protocol mismatch')
    manifest=rt.pinned(root/base/'inputs/manifest.json',proof['manifest'],16<<20)
    rt.require(manifest['schema']==p['manifest_schema']and manifest['subject']==p['subject'],'Immutable native manifest/profile mismatch')
    rt.require(out.is_dir()and out.resolve()==out and out.stat().st_uid==0 and out.stat().st_mode&0o777==0o700 and not tuple(out.iterdir()),
               'Fresh owned native output required')
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}and Path(sys.prefix)==rt.VENV and sys.prefix!=sys.base_prefix
        and sys.version_info[:2]==(3,11)and os.environ.get('JAX_PLATFORMS')=='cpu'
        and 'include-system-site-packages = false'in(rt.VENV/'pyvenv.cfg').read_text().lower(),'Qualified offline CPU venv required')
    rt.require(rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,helpers)==proof['source_binding'],'Native own source differs')
    started=time.monotonic();deadline=started+proof['remaining_seconds']
    def check():rt.require(time.monotonic()<=deadline,'Global full-scan budget exceeded')
    from importlib import metadata
    rt.require({n:metadata.version(n)for n in proof['versions']}==proof['versions'],'Native installed versions differ')
    import numpy as np
    import mediapipe as mp
    from PIL import Image
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
    from world_reward.hand_scan import scan_hands,NativeHandResult
    rt.require(Path(mp.__file__).is_relative_to(rt.VENV)and Path(np.__file__).is_relative_to(rt.VENV),
               'Native packages must belong to qualified isolated venv')
    rt.require(rt.identity('/opt/mediapipe-task/hand_landmarker.task')==proof['task'],'Native original task differs')
    opts=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='/opt/mediapipe-task/hand_landmarker.task',delegate=BaseOptions.Delegate.CPU),
      running_mode=RunningMode.IMAGE,num_hands=4,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    outputs=[];rows=manifest['images'];call_count=0
    with HandLandmarker.create_from_options(opts)as detector:
        for sequence in manifest['sequences']:
            selected=[r for r in rows if r['sequence_lex_index']==sequence['sequence_lex_index']]
            def images():
                for row in selected:
                    check();path=root/base/'inputs'/row['file'];rt.require(rt.identity(path,16<<20)=={k:row[k]for k in('bytes','sha256')},'JPEG changed before decode')
                    with Image.open(path)as image:
                        rt.require(image.mode=='RGB'and image.size==(640,480),'Native original RGB grid required');rgb=np.array(image,dtype=np.uint8)
                    yield row['frame_position'],rgb
            callback=native_callback(detector,mp,np,NativeHandResult)
            result=scan_hands(images(),callback,total_frames=sequence['frames'],image_size=(480,640),method='MediaPipe HandLandmarker IMAGE CPU',
                source_refs={'manifest':proof['manifest'],'task':proof['task'],'runtime':proof['image_id']},budget_seconds=max(.001,deadline-time.monotonic()))
            name=f"sequence_{sequence['sequence_lex_index']:03d}.npz";stats=serialize(result,out/name,np);call_count+=stats['frames']
            rt.write(out/(name+'.json'),(json.dumps(dict(native_categories=callback.categories),sort_keys=True)+'\n').encode())
            outputs.append(dict(sequence_lex_index=sequence['sequence_lex_index'],file=name,**stats));check()
    for row in rows:rt.require(rt.identity(root/base/'inputs'/row['file'],16<<20)=={k:row[k]for k in('bytes','sha256')},'Original JPEG changed after scan')
    rt.require(rt.identity('/opt/mediapipe-task/hand_landmarker.task')==proof['task']and rt.identity(proof_path,64<<10)==proof_pin
        and rt.identity(root/base/'inputs/manifest.json',16<<20)==proof['manifest']
        and {n:metadata.version(n)for n in proof['versions']}==proof['versions']
        and rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,helpers)==proof['source_binding'],
        'Native task/proof/manifest/versions/source changed')
    if cohort!='v1':rt.require(rt.identity(code/p['protocol'])==proof['protocol_identity'],'Native frozen protocol changed after scan')
    check();report=dict(stage='external_dexycb_full_t_mediapipe_hand_scan',status='pass',phase='complete',outputs=outputs,
      producer_revision=os.environ['WR_CODE_REVISION'],source_binding=proof['source_binding'],
      native_graphs=1,native_calls=call_count,gpu_used=False,private_values_read=False,accuracy_verified=False,
      protocol=dict(running_mode='IMAGE',num_hands=4,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5),
      raw_values_basis='native_Python_landmark_floats_without_rounding',availability_is_visibility=False,
      handedness_is_detection_confidence=False,actor_identity_inferred=False,elapsed_seconds=time.monotonic()-started)
    if cohort!='v1':report.update(cohort=cohort,public_base=base,protocol_identity=proof['protocol_identity'])
    rt.write(out/'report.json',(json.dumps(report,sort_keys=True)+'\n').encode());return report


def run(root,code,revision,cohort='v1'):
    p=profile(cohort);base=p['base'];helpers=source_helpers(cohort)
    rt=runtime(code);rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'
        and root==ROOT and Path(__file__).resolve()==code/HELPERS[0],'Actual immutable VM02 CPU adapter required')
    start=time.monotonic();deadline=start+BUDGET;source=rt.source(root,code,revision,ENTRY,helpers)
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact own source snapshot inventory required')
    evidence=authenticate(rt,root,code,cohort)if cohort!='v1'else authenticate(rt,root,code)
    rt.require(time.monotonic()<deadline,'Inclusive prehash budget exceeded')
    control=control_path(root,revision,cohort);rt.canonical(control);rt.require(not control.exists(),'Fresh scan namespace required')
    control.mkdir(mode=0o700);out=control/'predictions';out.mkdir(mode=0o700)
    proof=dict(source_binding=source,manifest=evidence['acquisition']['manifest'],task=evidence['task_pin'],versions=evidence['versions'],
               image_id=evidence['image']['Id'],remaining_seconds=deadline-time.monotonic())
    if cohort!='v1':proof.update(cohort=cohort,public_base=base,protocol_identity=evidence['acquisition']['protocol'])
    rt.write(control/'native-proof.json',(json.dumps(proof,sort_keys=True)+'\n').encode());proof_pin=rt.identity(control/'native-proof.json')
    name='world-reward-mediapipe-hand-scan-'+(cohort+'-'if cohort!='v1'else '')+revision[:12];failure=None;owned=False;report=dict(stage='mediapipe_hand_scan_host_seal',status='fail',
      producer_revision=revision,source_binding=source,acquisition_pins=evidence['acquisition'],runtime_pins=evidence['runtime'],
      budget_seconds=BUDGET,budget_scope='source_inputs_model_load_all_three_full_scans_serialization_posthash',gpu_used=False,private_values_read=False,
      accuracy_verified=False,source_rehashed_after=False,owned_cleanup_verified=False)
    if cohort!='v1':report.update(cohort=cohort,public_base=base,protocol_identity=evidence['acquisition']['protocol'])
    def interrupted(*_):raise TimeoutError('CPU scan interrupted; owned cleanup only')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        rt.require(not rt.control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned CPU container namespace occupied')
        command=['docker','run','--rm','--name',name,'--cidfile',str(control/'.container.cid'),'--label','world_reward.mediapipe_cpu.owner='+revision,
          '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','8g',
          '--tmpfs','/tmp:rw,nosuid,size=512m','--mount',f'type=bind,src={code.parent},dst={code.parent},readonly',
          '--mount',f'type=bind,src={root/base}/inputs,dst={root/base}/inputs,readonly',
          '--mount',f"type=bind,src={evidence['task']},dst=/opt/mediapipe-task/hand_landmarker.task,readonly",
          '--mount',f'type=bind,src={control}/native-proof.json,dst=/opt/mediapipe-hand-proof.json,readonly',
          '--mount',f'type=bind,src={out},dst={out}','--entrypoint','/usr/bin/env',evidence['image']['Id'],'-i',
          f'PATH={rt.VENV}/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=-1','JAX_PLATFORMS=cpu','MPLBACKEND=Agg','XDG_CACHE_HOME=/tmp',
          'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',f'WR_CODE_REVISION={revision}',str(rt.VENV/'bin/python'),'-I','-B',str(code/HELPERS[0]),
          '--native',str(root),str(code),str(out),str(proof_pin['bytes']),proof_pin['sha256']]
        if cohort!='v1':command+=['--cohort',cohort]
        owned=True
        with(control/'native.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400);result=subprocess.run(command,stdout=log,stderr=log,timeout=max(.001,deadline-time.monotonic()))
        rt.require(result.returncode==0,'Native scan failed; partial outputs must not be reused')
        native=rt.pinned(out/'report.json',rt.identity(out/'report.json'),64<<10)
        rt.require(native['status']=='pass'and native['phase']=='complete'and type(native['native_graphs'])is int and native['native_graphs']==1
            and type(native['native_calls'])is int and native['native_calls']==len(evidence['manifest']['images'])and len(native['outputs'])==3
            and native['source_binding']==source and native['producer_revision']==revision
            and all(native.get(k)is False for k in('gpu_used','private_values_read','accuracy_verified','availability_is_visibility',
                'handedness_is_detection_confidence','actor_identity_inferred')),'Full3 native scan gate required')
        if cohort!='v1':rt.require(native.get('cohort')==cohort and native.get('public_base')==base
            and native.get('protocol_identity')==evidence['acquisition']['protocol'],'Native immutable profile proof differs')
        output_names={'report.json',*(f'sequence_{i:03d}'+suffix for i in(4,39,74)for suffix in('.npz','.npz.json'))}
        rt.require({p.name for p in out.iterdir()}==output_names,'Exact full scan output inventory required')
        report['outputs']={p.name:rt.identity(p,32<<20)for p in out.iterdir()};report['native_report']=native
    except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        try:
            if owned:rt.cleanup(control/'.container.cid',name,evidence['image'],revision)
            report['owned_cleanup_verified']=True
            rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'}
                and rt.source(root,code,revision,ENTRY,helpers)==source
                and (authenticate(rt,root,code,cohort)if cohort!='v1'else authenticate(rt,root,code))==evidence,'Source/public/runtime changed after scan')
            report['source_rehashed_after']=True
            if 'outputs'in report:rt.require(all(rt.identity(out/n,32<<20)==p for n,p in report['outputs'].items()),'Output bytes changed before host seal')
            rt.require(time.monotonic()<=deadline,'Inclusive all3 scan posthash budget exceeded')
        except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
        report.update(status='fail'if failure else 'pass',elapsed_seconds=time.monotonic()-start)
        rt.write(control/'report.json',(json.dumps(report,sort_keys=True)+'\n').encode())
    if failure:raise ValueError('CPU hand scan failed; inspect sealed owned evidence')from None
    print(json.dumps({k:report[k]for k in('stage','status','gpu_used','accuracy_verified')}));return report


def main():
    args=sys.argv[1:];cohort='v1'
    if len(args)>=2 and args[-2]=='--cohort'and args[-1]in('v2','v3'):cohort=args[-1];args=args[:-2]
    if args[:1]==['--native']:
        if len(args)!=6:raise ValueError('Exact internal native arguments required')
        root,code,out=map(Path,args[1:4]);rt=runtime(code)
        return run_native(rt,root,code,out,Path('/opt/mediapipe-hand-proof.json'),dict(bytes=int(args[4]),sha256=args[5]),cohort)
    if args:raise ValueError('No arbitrary cohort/threshold arguments')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'],cohort)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps(dict(stage='external_dexycb_full_t_mediapipe_hand_scan',status='fail',error_type=type(exc).__name__)));sys.exit(1)

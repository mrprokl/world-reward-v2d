"""Private CPU YCBV relative-motion evaluation, only after frozen full predictions.

No model/GPU, no geometry import, no optimized alignment or frame deletion.
The public/firewall proof completes for all3scenes before private annotation
values are decoded. Only scalar/errors decisions are serialized, not GT arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import time
import tudl_holdout_inputs as files

ROOT=Path('/srv/scenesmith/world-reward');BASE='validation/ycbv_point_pose_v1'
JOB='run_ycbv_point_evaluate';OUTPUT='evaluation_v1'
PINS='configs/ycbv_point_evaluation_pins.json'
IMAGE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
STAGE='private_ycbv_contiguous_relative_material_motion';BUDGET=240;SCENES=(48,49,50)
SOURCE=('infra/ycbv_point_evaluate.py','infra/run_ycbv_point_evaluate.sh','infra/tudl_holdout_inputs.py',
 'src/world_reward/__init__.py','src/world_reward/point_bop_evaluation.py','src/world_reward/point_motion_evaluation.py',PINS)


def require(value,message):
    if not value:raise ValueError(message)


def pin(value,producer=False):
    fields={'bytes','sha256'}|({'producer_revision','script_sha256'}if producer else set())
    require(type(value)is dict and set(value)==fields,'Exact independent artifact/producer identity required')
    files._identity_record({k:value[k]for k in ('bytes','sha256')})
    if producer:
        require(all(type(value[k])is str and re.fullmatch('[0-9a-f]{%d}'%n,value[k])for k,n in(('producer_revision',40),('script_sha256',64))),'Full original producer source required')


def names():return {f'scene_{scene:06d}.npz'for scene in SCENES}


def validate_pins(value):
    require(type(value)is dict and set(value)=={'schema','acquisition_report','retention','track_report','masks_report','predictions','initial_masks'}
      and value['schema']=='world-reward-ycbv-point-evaluation-pins-v1','Actual completed independently measured evaluator pins required')
    for name in('acquisition_report','track_report','masks_report'):pin(value[name],True)
    pin(value['retention'])
    require(type(value['predictions'])is dict and set(value['predictions'])==names(),'Exactly3 full original predictions required')
    require(type(value['initial_masks'])is dict and set(value['initial_masks'])=={str(s)for s in SCENES},'Exactly3 original automatic initial masks required')
    for row in (*value['predictions'].values(),*value['initial_masks'].values()):pin(row)
    return value


def checked_json(path,wanted):
    require(files.identity(path)=={k:wanted[k]for k in ('bytes','sha256')},'Original independently pinned bytes required before JSON')
    raw=path.read_bytes();require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}=={k:wanted[k]for k in ('bytes','sha256')},'Pinned receipt changed')
    return files.strict_json(raw)


def source_binding(code,revision,*,host=False):
    files._canonical(code);require(code==ROOT/'jobs'/revision/JOB/'code'and re.fullmatch('[0-9a-f]{40}',revision)
      and Path(__file__).resolve()==code/SOURCE[0]and Path(files.__file__).resolve()==code/SOURCE[2],'Actual immutable evaluator source required')
    rows={}
    for p in(code,*sorted(code.rglob('*'))):
        files._canonical(p);mode=p.lstat().st_mode
        require(not mode&0o222 and(stat.S_ISDIR(mode)or stat.S_ISREG(mode)),'Complete readonly evaluator source required')
        if p.is_file():rows[str(p.relative_to(code))]=files.identity(p)
    require(set(SOURCE)<=set(rows)if host else set(rows)==set(SOURCE),'Complete host/exact small CPU evaluator closure required')
    markers={}
    for name in('revision','source-sha256'):
        p=files._canonical(code.parent/name);s=files._state(p);raw=p.read_bytes()
        require(stat.S_ISREG(s[2])and s[3]==1 and files._state(p)==s,'Original unchanged dispatch marker required')
        require(raw==(revision+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Actual dispatch marker differs')
        markers[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return dict(files=rows,markers=markers)


def _report(path,spec,expected):
    r=checked_json(path,spec)
    require(type(r)is dict and all(type(r.get(k))is type(v)and r[k]==v for k,v in expected.items()),'Actual completed original producer required')
    require(r.get('producer_revision')==spec['producer_revision']and r.get('script_sha256')==spec['script_sha256'],'Actual report producer differs')
    return r


def public_predictions(root,pins,*,decode=True):
    """All branches/fullindices sealed before any private metadata is interpreted."""
    validate_pins(pins);base=root/BASE
    acquisition=_report(base/'report.json',pins['acquisition_report'],dict(stage='external_ycbv_contiguous_rgb_only_acquisition',status='pass',phase='complete',
      dataset_revision='5c2c4aa229800355648cd268040aa814f8dc94f0',license='MIT',device='cpu',gpu_used=False,
      challenge_inputs_used=False,inference_performed=False,source_rehashed_after=True,disposable_archives_removed=True,
      selection_before_private_annotation_values=True,selected_frames=288,all_instances_retained=True,
      private_annotations_exported_as_inference_inputs=False))
    require(acquisition.get('retention_receipt')==pins['retention'],'Original all-instance retention receipt differs')
    track=_report(base/'comparison_v1/report.json',pins['track_report'],dict(stage='public_ycbv_same_native25_pool_boots_point_comparison',status='pass',phase='complete',
      full_original_frame_coverage=True,same_native_valid_pool=True,native_pool_frozen_before_tracking_and_rankings=True,
      all_inputs_models_sources_after_reverified=True,ground_truth_used=False,private_annotations_read=False,sensor_depth_used=False,
      source_camera_calibration_used=False,human_scale_used=False,hand_labeled_test=False,oracle_initial_queries=False,oracle_modes=[],challenge_inputs_used=False))
    masks=_report(base/'automatic_masks_v1/report.json',pins['masks_report'],dict(stage='public_ycbv_point_native_object_masks',status='pass',phase='complete',
      private_annotations_read=False,query='object.',frames_completed=288,all_inputs_sources_assets_outputs_rehashed=True))
    durations=track.get('whole_pilot_GPU_elapsed_seconds');require(type(durations)in(int,float)and 0<durations<=3600,'Complete frozen GPU budget required')
    require(type(track.get('outputs'))is list and len(track['outputs'])==3,'All3 output receipts required')
    require(type(masks.get('masks'))is list and len(masks['masks'])==288,'All original automatic mask receipts required')
    rows=[]
    for scene,row in zip(SCENES,track['outputs']):
        name=f'scene_{scene:06d}.npz';path=base/'comparison_v1'/name;wanted=pins['predictions'][name]
        require(type(row)is dict and row.get('file')==name and type(row.get('scene_id'))is int and row['scene_id']==scene and type(row.get('frames'))is int and row['frames']==96
          and {k:row.get(k)for k in ('bytes','sha256')}==wanted and files.identity(path)==wanted,'Original full prediction bytes/scene/order required')
        mask=base/f'automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png';mp=pins['initial_masks'][str(scene)]
        require(files.identity(mask)==mp,'Original automatic initial mask bytes differ')
        mr=masks['masks'][(scene-48)*96]
        require(mr.get('scene_id')==scene and mr.get('frame_id')==0 and mr.get('file')==f'scene_{scene:06d}/masks/1/000000.png'
          and {k:mr.get(k)for k in ('bytes','sha256')}==mp,'Initial mask must be original automatic framezero')
        rows.append(dict(scene_id=scene,path=path,mask_path=mask,pin=wanted,mask_pin=mp,arrays=row.get('arrays')))
    require({p.name for p in(base/'comparison_v1').iterdir()}=={'.container.cid','report.json',*names()},'No missing/extra prediction files allowed')
    if decode:
        import numpy as np
        from PIL import Image
        for row in rows:
            with np.load(row['path'],allow_pickle=False)as data:
                require(len(data.files)==len(set(data.files))and {'frame_index','baseline_poses','candidate_poses'}<=set(data.files),'Explicit full homogeneous trajectory aliases required')
                arrays={k:data[k]for k in data.files}
            require(type(row['arrays'])is dict and set(row['arrays'])==set(arrays),'Original array identity inventory required')
            for key,array in arrays.items():
                identity=dict(dtype=array.dtype.str,shape=list(array.shape),sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
                require(identity==row['arrays'][key],'Original saved native array bytes differ')
            idx=arrays['frame_index'];require(idx.dtype==np.int64 and np.array_equal(idx,np.arange(96,dtype=np.int64)),'Every original frame0..95 required')
            for key in('baseline_poses','candidate_poses'):
                a=arrays[key];require(a.dtype==np.float64 and a.shape==(96,4,4)and np.isfinite(a).all()and np.all(a[:,3]==[0,0,0,1])
                  and np.allclose(a[:,:3,:3].transpose(0,2,1)@a[:,:3,:3],np.eye(3),atol=1e-6,rtol=0)
                  and np.allclose(np.linalg.det(a[:,:3,:3]),1,atol=1e-6,rtol=0),'Complete proper float64 SE3, no fitted scale required')
            with Image.open(row['mask_path'])as image:
                require(image.mode=='L'and image.size==(640,480),'Original binary mask grid required');auto=np.array(image)
            require(auto.dtype==np.uint8 and np.all((auto==0)|(auto==255))and np.any(auto==255),'Original nonempty automatic mask required')
            row.update(baseline=arrays['baseline_poses'],candidate=arrays['candidate_poses'],automatic_mask=auto==255)
    return rows,dict(acquisition=acquisition,track=track,masks=masks)


def producer_sources(root,pins,reports):
    """Host-only original report/entrypoint ancestry, no private values/imports."""
    roles=(('acquisition','acquisition_report','run_ycbv_point_acquire','infra/ycbv_point_acquire.py'),
      ('track','track_report','run_ycbv_point_track','infra/ycbv_point_track.py'),
      ('masks','masks_report','run_ycbv_point_masks','infra/ycbv_point_masks.py'))
    result={}
    for name,key,job,entry in roles:
        receipt=reports[name];spec=pins[key];code=root/'jobs'/spec['producer_revision']/job/'code'
        helpers=receipt.get('source_helpers',{}).get('files')if name!='masks'else receipt.get('source_binding',{}).get('helpers')
        require(type(helpers)is dict and entry in helpers and helpers[entry].get('sha256')==spec['script_sha256'],
          'Actual original producing source inventory required')
        actual={}
        for relative,wanted in helpers.items():
            require(type(relative)is str and re.fullmatch(r'(?:infra/[a-z0-9_]+\.(?:py|sh)|configs/[a-z0-9_]+\.json|src/world_reward/[a-z0-9_/]+\.py)',relative),
              'Original code-only producer source path required')
            pin(wanted);actual[relative]=files.identity(code/relative)
            require(actual[relative]==wanted,'Actual original producer helper changed')
        markers={}
        for marker in('revision','source-sha256'):
            p=files._canonical(code.parent/marker);state=files._state(p);raw=p.read_bytes()
            require(stat.S_ISREG(state[2])and state[3]==1 and files._state(p)==state,'Original producer dispatch marker required')
            require(raw==(spec['producer_revision']+'\n').encode()if marker=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),
              'Original producer marker values differ')
            markers[marker]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        recorded=receipt.get('source_helpers',{}).get('markers')if name!='masks'else receipt.get('source_binding',{}).get('markers')
        require(recorded==markers,'Historical producer marker identities differ from the frozen receipt')
        result[name]=dict(files=actual,markers=markers)
    return result



HOST_SEAL_STAGE='ycbv_private_evaluator_host_post_seal'


def host_snapshot(root,code,revision):
    """Opaque public/source ancestry and safe image projection; no private reads."""
    require(root==ROOT,'Exact owned evaluator root required')
    full=source_binding(code,revision,host=True)
    pins=validate_pins(files.strict_json((code/PINS).read_bytes()));_,reports=public_predictions(root,pins,decode=False)
    producer=producer_sources(root,pins,reports)
    environment={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','DOCKER_HOST':'unix://'+str(root/'docker.sock')}
    result=subprocess.run(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],
        env=environment,capture_output=True,timeout=10)
    require(result.returncode==0 and len(result.stdout)<=16384,'Bounded safe image metadata query required')
    image=files.strict_json(result.stdout)
    require(type(image)is dict and set(image)=={'Id','Architecture','Os','RootFS'}and image['Id']==IMAGE and image['Architecture']=='amd64'and image['Os']=='linux','Pinned CPU evaluator image/platform required')
    layers=image['RootFS'];require(type(layers)is dict and set(layers)=={'Type','Layers'}and layers['Type']=='layers'and type(layers['Layers'])is list and layers['Layers']and
        all(type(value)is str and re.fullmatch('sha256:[0-9a-f]{64}',value)for value in layers['Layers']),'Full actual ordered image layer identity required')
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(schema='world_reward.ycbv_evaluator_host_snapshot.v1',pins_sha256=digest(pins),public_sha256=digest(reports),producer_sha256=digest(producer),
        source=dict(files={name:full['files'][name]for name in SOURCE},markers=full['markers'],complete_closure_sha256=digest(full)),image=image)


def write_host_seal(root,code,revision,before,report_before,exit_status,cleanup_verified):
    """Exclusive host-post receipt; never rewrite/reclassify the CPU report."""
    require(type(exit_status)is int and 0<=exit_status<=255 and type(cleanup_verified)is bool,'Actual wrapper exit/owned-cleanup state required')
    out=files._canonical(root/BASE/OUTPUT)
    require(out.is_dir()and out.stat().st_mode&0o777==0o700,'Existing owned private evaluation output required')
    seal=dict(stage=HOST_SEAL_STAGE,status='fail',producer_revision=revision,wrapper_exit_status=exit_status,
        owned_container_cleanup_verified=cleanup_verified,private_labels_read=False,model_or_GPU_used=False,quality_decision_exported=False,
        original_container_report_rewritten=False,source_public_producer_image_post_verified=False)
    try:
        after=host_snapshot(root,code,revision)
        require(type(before)is dict and before==after,'Original source/public/producer/image changed after CPU evaluation')
        pin(report_before)
        reportpath=out/'report.json';reportpin=files.identity(reportpath)
        require(reportpin==report_before,'Original CPU report changed during owned cleanup/post-verification')
        require(reportpin['bytes']<=262144,'Bounded aggregate-only original CPU report required')
        report=checked_json(reportpath,reportpin)
        expected=dict(stage=STAGE,status='pass',phase='complete',producer_revision=revision,
            script_sha256=before['source']['files'][SOURCE[0]]['sha256'],source_helpers=dict(files=before['source']['files'],markers=before['source']['markers']),
            host_wrapper_post_verified=False,requires_independent_host_post_receipt=True,public_after_reverified=True,source_after_reverified=True,
            predictions_frozen_before_private_values=True,gpu_used=False,inference_performed=False,private_arrays_exported=False,
            alignment_performed=False,scale_fitted=False,budget_seconds=BUDGET)
        require(type(report)is dict and all(type(report.get(k))is type(v)and report[k]==v for k,v in expected.items()),'Actual complete original CPU evaluation report required')
        require(type(report.get('elapsed_seconds'))in(int,float)and 0<report['elapsed_seconds']<=BUDGET,'Original frozen CPU budget must pass')
        require(exit_status==0 and cleanup_verified,'Wrapper execution and exact owned cleanup must both pass')
        require(files.identity(reportpath)==reportpin,'Original CPU report changed before host seal')
        seal.update(status='pass',container_report=reportpin,source_public_producer_image_post_verified=True,
            original_source=before['source'],pins_sha256=before['pins_sha256'],public_sha256=before['public_sha256'],producer_sha256=before['producer_sha256'],image=before['image'])
    except Exception as error:
        seal.update(error_type=type(error).__name__,error='Host post-verification failed closed; original source/report preserved')
    raw=(json.dumps(seal,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=16384,'Bounded host-only metadata seal required')
    with (out/'host-post.json').open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return seal['status']=='pass'

def private_identities(root,pins):
    """Hash all retained private bytes; do not decode annotations or select GT."""
    private=files._canonical(root/BASE/'eval_private')
    require(private.is_dir()and private.stat().st_mode&0o777==0o700,'Private original evaluator-only parent required')
    retention=checked_json(private/'retention-receipt.json',pins['retention'])
    require(type(retention)is dict and set(retention)=={'files','all_instances_retained'}and retention['all_instances_retained']is True
      and type(retention['files'])is list and retention['files'],'Original all-instance private retention proof required')
    expected={};seen=set()
    for row in retention['files']:
        require(type(row)is dict and set(row)=={'file','bytes','sha256'},'Exact retained private metadata fields required')
        name=row['file'];require(type(name)is str and name.startswith('source/')and str(PurePosixPath(name))==name and not PurePosixPath(name).is_absolute()
          and '\\'not in name and all(x not in('', '.', '..')for x in name.split('/'))and name not in seen,'Original relative unaliased private source required')
        allowed=re.fullmatch(r'source/(?:licenses/(?:hf_readme|publisher_readme|publisher_license|bop_format|bop_params)\.txt|licenses/dataset_info\.md|test/0000(?:48|49|50)/(?:scene_(?:camera|gt|gt_info)\.json|depth/[0-9]{6}\.png|(?:mask|mask_visib)/[0-9]{6}_[0-9]{6}\.png))',name)
        require(allowed is not None,'Only the frozen three-scene original private source/license layout allowed')
        if '/depth/'in name or '/mask/'in name or '/mask_visib/'in name:require(0<=int(Path(name).stem.split('_')[0])<96,'Only full original0..95 private frames allowed')
        seen.add(name);wanted={k:row[k]for k in ('bytes','sha256')};pin(wanted)
        require(files.identity(private/name)==wanted,'Original retained private bytes differ');expected[name]=wanted
    minimum={f'source/licenses/{name}.txt'for name in('hf_readme','publisher_readme','publisher_license','bop_format','bop_params')}|{'source/licenses/dataset_info.md'}
    for scene in SCENES:
        prefix=f'source/test/{scene:06d}/'
        minimum.update(prefix+f'scene_{kind}.json'for kind in('camera','gt','gt_info'))
        minimum.update(prefix+f'depth/{i:06d}.png'for i in range(96))
        for i in range(96):
            for folder in('mask','mask_visib'):require(any(re.fullmatch(re.escape(prefix+folder+'/')+fr'{i:06d}_[0-9]{{6}}\.png',n)for n in seen),'All96 original mask/visible-mask inventories required')
    require(minimum<=seen,'Complete full96 three-scene metadata/depth/licenses required before private interpretation')
    for path in private.rglob('*'):
        files._canonical(path);mode=path.lstat().st_mode
        require(stat.S_ISREG(mode)or stat.S_ISDIR(mode),'Only unaliased regular files/directories in private cohort')
    actual={str(p.relative_to(private))for p in private.rglob('*')if p.is_file()}
    require(actual=={'retention-receipt.json',*seen},'No extra/unpinned private values permitted')
    return expected


def evaluate_private(root,rows,pins):
    import numpy as np
    from PIL import Image
    from world_reward.point_bop_evaluation import evaluate_bop_scene
    from world_reward.point_motion_evaluation import cohort_gate
    private=root/BASE/'eval_private';identities=private_identities(root,pins);results=[]
    for item in rows:
        scene=item['scene_id'];prefix=f'source/test/{scene:06d}/'
        def original(name):return checked_json(private/(prefix+name),identities[prefix+name])
        cameras,gt,info=[original(f'scene_{name}.json')for name in('camera','gt','gt_info')]
        require(type(gt)is dict and type(gt.get('0'))is list and gt['0'],'All original initial instances required')
        for frame in range(96):
            require(type(gt.get(str(frame)))is list and gt[str(frame)],'All96 original instance lists required')
            for folder in('mask','mask_visib'):
                expected={prefix+f'{folder}/{frame:06d}_{i:06d}.png'for i in range(len(gt[str(frame)]))}
                actual={n for n in identities if n.startswith(prefix+f'{folder}/{frame:06d}_')}
                require(actual==expected,'All original native instance masks, no deletion/replacement allowed')
        mask_paths=[private/(prefix+f'mask_visib/000000_{i:06d}.png')for i in range(len(gt['0']))]
        require(all(str(p.relative_to(private))in identities for p in mask_paths),'All initial instance masks retained')
        with Image.open(private/(prefix+'depth/000000.png'))as image:
            require(image.mode in('I;16','I;16L','I')and image.size==(640,480),'Original uint16 depth PNG required')
            raw=np.array(image)
            require(raw.dtype.kind in 'iu'and np.all((raw>=0)&(raw<=65535)),'Native unsigned16 sensor values required')
            depth=raw.astype(np.uint16)
        masks=[]
        for path in mask_paths:
            with Image.open(path)as image:require(image.mode=='L'and image.size==(640,480),'Original instance mask grid required');masks.append(np.array(image))
        results.append(evaluate_bop_scene(item['baseline'],item['candidate'],item['automatic_mask'],cameras,gt,info,depth,np.stack(masks),sequence_id=f'ycbv_scene_{scene:06d}'))
    require(private_identities(root,pins)==identities,'Private source changed after evaluation')
    return results,cohort_gate(results)


def main(argv=None):
    argparse.ArgumentParser(allow_abbrev=False).parse_args(argv)
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION'];root=Path(os.environ['WR_ROOT'])
    require(root==ROOT and os.uname().sysname=='Linux'and os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE
      and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}and re.fullmatch('[0-9a-f]{64}',os.environ.get('WR_YCBV_EVALUATION_HOST_PROOF_SHA256','')),
      'Offline original CPU-only ROOT evaluator required')
    source=source_binding(code,revision);pins=validate_pins(files.strict_json((code/PINS).read_bytes()))
    require(hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()==os.environ['WR_YCBV_EVALUATION_HOST_PROOF_SHA256'],'Independent host/public pin proof required')
    out=files._canonical(root/BASE/OUTPUT);require(out.is_dir()and {p.name for p in out.iterdir()}=={'.container.cid'},'Fresh unique evaluator output required')
    started=time.monotonic();report=dict(stage=STAGE,status='fail',producer_revision=revision,script_sha256=source['files'][SOURCE[0]]['sha256'],source_helpers=source,
      budget_seconds=BUDGET,gpu_used=False,inference_performed=False,challenge_inputs_used=False,training_overlap_verified=False,
      CARI4D_victory_verified=False,absolute_shape_accuracy_verified=False,full_HOI_accuracy_verified=False,private_arrays_exported=False,
      predictions_frozen_before_private_values=False,host_wrapper_post_verified=False,requires_independent_host_post_receipt=True,alignment_performed=False,scale_fitted=False,reacquisition_supported=False)
    def expired(*_):raise TimeoutError('Frozen240s CPU evaluator deadline exceeded')
    previous={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    try:
        rows,_=public_predictions(root,pins);report['predictions_frozen_before_private_values']=True
        results,gate=evaluate_private(root,rows,pins);report.update(phase='complete',status='pass',scenes=results,decision=gate,
          scientific_decision='ACCEPT_RELATIVE_MOTION_DIAGNOSTIC'if gate['status']=='pass'else'REJECT')
    except Exception as error:report.update(status='fail',error_type=type(error).__name__,error='Frozen private evaluator failed closed; private values omitted')
    finally:
        try:
            public_predictions(root,pins,decode=False);report['public_after_reverified']=True
            report['source_after_reverified']=source_binding(code,revision)==source
        except Exception:report.update(public_after_reverified=False,source_after_reverified=False,status='fail')
        if not report.get('source_after_reverified'):report['status']='fail'
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>BUDGET:report['status']='fail'
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
        with (out/'report.json').open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write((json.dumps(report,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()

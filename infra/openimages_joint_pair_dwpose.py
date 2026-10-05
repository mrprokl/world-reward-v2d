"""New external ALL retained person crops through unchanged native DWPose133.

One CPU session; no invented persons/neck, full-image fallback, score clipping,
hand ownership, target selection or reference/rights/split metadata access.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import joint_pair_dwpose_acquire as acq
import mediapipe_cpu_runtime_verify as rt

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_openimages_joint_pair_dwpose'
CONFIG='configs/openimages_joint_pair_dwpose_v1.json'
IMAGE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
GDI=ROOT/'results/openimages-joint-pair-gdi-v2'
OUTPUT=ROOT/'results/openimages-joint-pair-dwpose-v1'
NATIVE=OUTPUT/'native'
HELPERS=('infra/openimages_joint_pair_dwpose.py','infra/run_openimages_joint_pair_dwpose.sh',CONFIG,
 'infra/dwpose_smoke.py','infra/run_dwpose_smoke.sh','infra/dwpose_acquire.py','infra/dwpose_wheel_audit.py',
 'infra/joint_pair_dwpose_acquire.py','infra/run_joint_pair_dwpose_acquire.sh','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/person_pose_observations.py','src/world_reward/prompt_selection.py')


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();rt.write(path,raw,0o400)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def configuration(code):
    value=rt.strict((code/CONFIG).read_bytes());gdi=value['gdi']
    expected=dict(schema='world_reward.openimages_joint_pair_dwpose.v1',gdi=gdi,
        asset_acquisition_revision='37a6d39045db215626b8a0268c1357019835af59',
        asset_acquisition_report=dict(bytes=11455,sha256='3967342c41fb81e16f8d3d08c705a6f39bf7ab4d650fdd90bb5e344ca7ec19cd'),
        image_id=IMAGE,output='results/openimages-joint-pair-dwpose-v1',acquired_images=50,missing_images=14,budget_seconds=600,
        ground_truth_used=False,quality_verified=False,anatomical_ownership_verified=False,adoption=False)
    rt.require(value==expected and all(type(value[k])is type(v)for k,v in expected.items())
        and set(gdi)=={'producer_revision','report','native_report'}
        and re.fullmatch('[0-9a-f]{40}',str(gdi['producer_revision'])),'Actual frozen GDI producer pins required; prospective config cannot run')
    for row in (gdi['report'],gdi['native_report']):
        rt.require(type(row)is dict and set(row)=={'bytes','sha256'}and type(row['bytes'])is int
            and 0<row['bytes']<=1 << 20 and re.fullmatch('[0-9a-f]{64}',str(row['sha256'])),'Independent actual GDI byte pins required')
    return value


def asset_identity(code,cfg):
    report=rt.pinned(ROOT/acq.REPORT,cfg['asset_acquisition_report'],1 << 20)
    old=ROOT/'jobs'/cfg['asset_acquisition_revision']/acq.ENTRY/'code'
    source=rt.source(ROOT,old,cfg['asset_acquisition_revision'],acq.ENTRY,acq.HELPERS)
    current={n:rt.identity(code/n,2_000_000)for n in acq.HELPERS if n not in('infra/joint_pair_dwpose_acquire.py','infra/run_joint_pair_dwpose_acquire.sh')}
    rt.require(all(source['helpers'][n]==pin for n,pin in current.items()),'Unchanged original acquisition/audit/runtime source required')
    expected=dict(schema='world_reward.joint_pair_dwpose_acquisition.v1',status='pass',phase='complete',
        producer_revision=cfg['asset_acquisition_revision'],source_binding=source,source_and_assets_rehashed_after=True,
        gpu_used=False,inference_performed=False,packages_installed=False,ground_truth_used=False,challenge_inputs_used=False,
        historical_outputs_modified=False,training_overlap_verified=False,challenge_overlap_verified=False)
    rt.require(all(type(report.get(k))is type(v)and report[k]==v for k,v in expected.items()),'Actual complete NEW asset acquisition PASS required')
    pins=acq.final_assets(ROOT/acq.BASE,report['files'],report['wheel_audits'])
    rt.require(pins==report['final_asset_pins'],'All original nine assets/notices must match final pins')
    return dict(report_identity=cfg['asset_acquisition_report'],source_binding=source,asset_pins=pins)


def gdi_inputs(cfg):
    pin=cfg['gdi'];host=rt.pinned(GDI/'report.json',pin['report'],1 << 20)
    native=rt.pinned(GDI/'native/native.json',pin['native_report'],1 << 20)
    producer=pin['producer_revision'];old=ROOT/'jobs'/producer/'run_openimages_joint_pair_gdi/code'
    source=rt.source(ROOT,old,producer,'run_openimages_joint_pair_gdi',tuple(host['source_binding']['helpers']))
    expected=dict(schema='world_reward.openimages_joint_pair_gdi.v2',stage='external_all_person_object_gdi_banks_host',
        status='pass',producer_revision=producer,source_binding=source,native_report_identity=pin['native_report'],
        slots=64,acquired=50,missing=14,native_forward_calls=100,native_model_loads=1,
        source_inputs_assets_rehashed_after=True,owned_container_removed=True,quality_verified=False,
        ownership_verified=False,task_target_verified=False,adoption=False)
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items())
        and native['status']=='pass'and native['phase']=='complete'and native['producer_revision']==producer
        and native['source_binding']==source and host['banks']==native['images']
        and native['person_query']=='person.'and native['object_query']=='object.'
        and native['confidence']==.3 and native['text_threshold']==.25 and native['nms_iou']==.7,
        'Actual complete unchanged GDI banks PASS required')
    rows=host['banks'];rt.require(len(rows)==50 and len({r['image_id']for r in rows})==50,'All acquired GDI images required')
    images=[]
    for row in rows:
        iid=row['image_id'];rt.require(re.fullmatch('[0-9a-f]{16}',iid),'Canonical original image ID required')
        image_path=Path('/srv/world-reward-data/openimages_joint_pair_acquisition_v1')/iid/'rgb.jpg'
        bank_path=GDI/'native'/(iid+'.npz')
        rt.require(row['input_file']==str(image_path)and row['prediction_file']==bank_path.name
            and rt.identity(image_path,16 << 20)==row['input_identity']
            and rt.identity(bank_path,32 << 20)==row['prediction_identity'],'Original RGB/GDI bank bytes differ')
        images.append(dict(image_id=iid,file=str(image_path),width=row['width'],height=row['height'],
            **row['input_identity'],decoded_rgb_sha256=row['decoded_rgb_sha256'],person_ids=row['person_ids'],
            gdi_file=str(bank_path),gdi_identity=row['prediction_identity']))
    return dict(schema='world_reward.openimages_joint_pair_person_rgb_inputs.v1',images=images),dict(source_binding=source,report_identity=pin['report'],native_report_identity=pin['native_report'])


def person_bank(path,image):
    """Recheck raw→unchanged NMS→retained alignment; no new candidate filtering."""
    import numpy as np
    from world_reward.prompt_selection import BoxDetection,non_maximum_suppression
    rt.require(rt.identity(path,32 << 20)==image['gdi_identity'],'Original GDI file changed')
    with np.load(path,allow_pickle=False)as data:
        boxes=data['person_boxes_original_xyxy'].copy();scores=data['person_detector_scores'].copy()
        ids=tuple(data['person_retained_ids'].tolist());raw_boxes=data['person_raw_boxes'];raw_scores=data['person_raw_scores']
        retained=non_maximum_suppression(tuple(BoxDetection(tuple(map(float,b)),float(s))for b,s in zip(raw_boxes,raw_scores)),
            image['width'],image['height'],.3,.7)
        expected_boxes=np.asarray([r.box for r in retained],np.float64).reshape(-1,4)
        expected_scores=np.asarray([r.score for r in retained],np.float64)
        rt.require(ids==tuple(image['person_ids'])and len(ids)==len(retained)
            and ids==tuple(f'image:{image["image_id"]}/person/retained:{i:06d}'for i in range(len(ids)))
            and boxes.dtype==scores.dtype==np.float64 and np.array_equal(boxes,expected_boxes)
            and np.array_equal(scores,expected_scores),'Complete original retained person census differs')
    return ids,boxes,scores


def pip_arguments(prefix):
    """Original isolated pip options; ONLY fresh acquired wheel paths change."""
    import dwpose_smoke as dw
    original=dw.pip_argv(prefix,ROOT)
    mapping={str(ROOT/dw.acquisition.BASE/name):str(ROOT/acq.BASE/name)
        for name,_,_,_ in dw.acquisition.ASSETS if name.startswith('wheels/')}
    rt.require(all(p in original for p in mapping),'Exact original wheel install argv required')
    return [mapping.get(item,item)for item in original]


def observe(images,out,native,session,decode,report,persist):
    import numpy as np
    from world_reward.person_pose_observations import infer_person_pose_frame
    for image in images:
        rgb=decode(image);rt.require(type(rgb)is np.ndarray and rgb.dtype==np.uint8
            and rgb.shape==(image['height'],image['width'],3)
            and hashlib.sha256(rgb.tobytes()).hexdigest()==image['decoded_rgb_sha256'],'Same original GDI decoded RGB required')
        ids,boxes,scores=person_bank(Path(image['gdi_file']),image)
        result=infer_person_pose_frame(native,session,rgb,0,ids,boxes,scores)
        name=image['image_id']+'.npz'
        with(out/name).open('xb')as stream:
            os.fchmod(stream.fileno(),0o400)
            np.savez(stream,**{k:getattr(result,k)for k in('boxes_original_xyxy','detector_scores','keypoints_original_xy','raw_scores','native_valid','in_original_image')})
            stream.flush();os.fsync(stream.fileno())
        report['banks'].append(dict(image_id=image['image_id'],width=image['width'],height=image['height'],
            input_file=image['file'],input_identity={k:image[k]for k in('bytes','sha256')},decoded_rgb_sha256=image['decoded_rgb_sha256'],
            person_ids=ids,persons=len(ids),prediction_file=name,prediction_identity=rt.identity(out/name,32 << 20)))
        persist()


def native(code,revision,cfg,input_pin):
    started=time.monotonic();source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    rt.require({p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}and os.environ['WR_IMAGE_ID']==IMAGE,'Offline original CPU runtime required')
    inputs=rt.pinned(OUTPUT/'inputs.json',input_pin,1 << 20)
    rt.require(set(inputs)=={'schema','images'}and inputs['schema']=='world_reward.openimages_joint_pair_person_rgb_inputs.v1'
        and len(inputs['images'])==50,'Complete RGB/person-only input projection required')
    report=dict(schema=cfg['schema'],stage='external_all_person_native_dwpose133',status='fail',phase='provenance',
        producer_revision=revision,source_binding=source,image_id=IMAGE,projected_input_identity=input_pin,banks=[],native_calls=[],
        native_session_count=0,raw_scores_clamped=False,own_feed_cast=False,full_image_fallback=False,wrapper_neck134_used=False,
        ground_truth_used=False,reference_metadata_read=False,split_metadata_read=False,rights_metadata_read=False,
        anatomical_ownership_verified=False,quality_verified=False,adoption=False,gpu_used=False,network='none',
        training_overlap_verified=False,challenge_overlap_verified=False)
    def persist():report['elapsed_seconds']=time.monotonic()-started
    def expired(*_):raise TimeoutError('Frozen CPU DWPose600s budget exceeded')
    handlers={s:signal.signal(s,expired)for s in(signal.SIGTERM,signal.SIGALRM)};signal.alarm(600)
    try:
        import dwpose_smoke as dw
        before=asset_identity(code,cfg);report['asset_binding']=before
        report['source_identity']=dw.source_identity();report['dependency_identity']=dw.dependency_identity()
        with dw.private_prefix(report,persist)as prefix:
            subprocess.run(pip_arguments(prefix),capture_output=True,check=True,timeout=max(.001,600-(time.monotonic()-started)))
            ort,imports=dw.import_runtime(prefix)
            path=ROOT/acq.BASE/'source/onnxpose.py';spec=importlib.util.spec_from_file_location('wr_original_dwpose_external',path)
            original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(str(ROOT/acq.BASE/dw.acquisition.ASSETS[0][0]),sess_options=options,providers=['CPUExecutionProvider'])
            session.disable_fallback();dw.validate_session(session);dw.validate_options(session,ort)
            proxy=dw.SessionProxy(session,report['native_calls'],persist);report.update(native_session_count=1,runtime_imports=imports,phase='native_all_person_crops')
            import cv2
            def decode(row):
                path=Path(row['file']);rt.require(rt.identity(path,16 << 20)=={k:row[k]for k in('bytes','sha256')},'Original RGB changed')
                # Same Pillow original RGB decoder as GDI, not a different
                # OpenCV JPEG implementation that could change exact pixels.
                from PIL import Image
                import numpy as np
                with Image.open(path)as image:
                    rt.require(image.format=='JPEG'and image.mode=='RGB'and image.size==(row['width'],row['height']),'Original JPEG grid differs')
                    return np.array(image,copy=True)
            observe(inputs['images'],NATIVE,original.inference_pose,proxy,decode,report,persist)
            rt.require(len(report['banks'])==50 and len(report['native_calls'])==sum(r['persons']for r in report['banks'])
                and all(r['run_completed']for r in report['native_calls']),'Every original person native call required')
            del proxy,session
            import gc
            gc.collect()
        rt.require(report['private_prefix_removed'] is True and not prefix.exists()
            and source==rt.source(ROOT,code,revision,ENTRY,HELPERS) and before==asset_identity(code,cfg)
            and rt.identity(OUTPUT/'inputs.json')==input_pin,'Original source/assets changed or private runtime survives')
        for row in inputs['images']:
            rt.require(rt.identity(Path(row['file']),16 << 20)=={k:row[k]for k in('bytes','sha256')}
                and rt.identity(Path(row['gdi_file']),32 << 20)==row['gdi_identity'],'Original RGB/GDI inputs changed after inference')
        for row in report['banks']:
            rt.require(rt.identity(NATIVE/row['prediction_file'],32 << 20)==row['prediction_identity'],'Native saved observations changed')
        rt.require(time.monotonic()-started<600,'Inclusive native CPU budget exceeded')
        report.update(status='pass',phase='complete',source_inputs_assets_rehashed_after=True)
    except BaseException as exc:
        report.update(status='fail',error_type=type(exc).__name__);raise
    finally:
        signal.alarm(0);persist();write(NATIVE/'native.json',report);NATIVE.chmod(0o500)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def host_prepare(code,revision):
    cfg=configuration(code);source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    inputs,gdi=gdi_inputs(cfg);assets=asset_identity(code,cfg)
    rt.require(os.uname().nodename=='world-reward-ncc-h100-02'and not rt.canonical(OUTPUT).exists(),'Actual fresh VM02 output required')
    OUTPUT.mkdir(mode=0o700);NATIVE.mkdir(mode=0o700);input_pin=write(OUTPUT/'inputs.json',inputs)
    write(OUTPUT/'host-before.json',dict(source_binding=source,gdi_binding=gdi,asset_binding=assets,projected_input_identity=input_pin))
    rt.require(source==rt.source(ROOT,code,revision,ENTRY,HELPERS)and(inputs,gdi)==gdi_inputs(cfg)and assets==asset_identity(code,cfg),'Original input controls changed after preparation')
    print(input_pin['sha256']+' '+str(input_pin['bytes']))


def host_finalize(code,revision,exit_code):
    cfg=configuration(code);source=rt.source(ROOT,code,revision,ENTRY,HELPERS);before=rt.strict((OUTPUT/'host-before.json').read_bytes())
    inputs,gdi=gdi_inputs(cfg);assets=asset_identity(code,cfg);input_pin=before['projected_input_identity']
    result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s','8s','docker','ps','-aq','--no-trunc',
        '--filter','name=^/world-reward-oi-joint-dwpose-'+revision[:12]+'$'],capture_output=True,timeout=12,
        env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')))
    rt.require(result.returncode==0 and not result.stdout.strip(),'Owned CPU native container must be absent')
    rt.require(before==dict(source_binding=source,gdi_binding=gdi,asset_binding=assets,projected_input_identity=input_pin)
        and rt.pinned(OUTPUT/'inputs.json',input_pin,1 << 20)==inputs,'Host source/inputs/assets changed')
    report=rt.strict((NATIVE/'native.json').read_bytes())
    rt.require(exit_code==0 and report['status']=='pass'and report['phase']=='complete'and report['producer_revision']==revision
        and report['source_binding']==source and report['asset_binding']==assets and report['projected_input_identity']==input_pin
        and report['private_prefix_removed'] is True and report['source_inputs_assets_rehashed_after'] is True,'Complete new CPU native PASS required')
    rows=report['banks'];rt.require([r['image_id']for r in rows]==[r['image_id']for r in inputs['images']],'All original fifty banks required')
    for row,image in zip(rows,inputs['images']):
        rt.require(row['prediction_file']==row['image_id']+'.npz'and row['person_ids']==image['person_ids']
            and rt.identity(NATIVE/row['prediction_file'],32 << 20)==row['prediction_identity'],'Original saved person bank differs')
    rt.require({p.name for p in NATIVE.iterdir()}=={'native.json',*[r['prediction_file']for r in rows]}
        and {p.name for p in OUTPUT.iterdir()}=={'native','inputs.json','host-before.json'},'Exclusive new CPU outputs required')
    summary=dict(schema=cfg['schema'],status='pass',stage='external_all_person_native_dwpose133_host',producer_revision=revision,
        source_binding=source,asset_binding=assets,gdi_binding=gdi,native_report_identity=rt.identity(NATIVE/'native.json',4 << 20),
        banks=rows,acquired_images=50,missing_images=14,native_session_count=1,native_calls=len(report['native_calls']),
        source_inputs_assets_rehashed_after=True,private_prefix_removed=True,owned_container_removed=True,
        quality_verified=False,anatomical_ownership_verified=False,adoption=False,training_overlap_verified=False,challenge_overlap_verified=False)
    rt.require(report['native_session_count']==1 and len(report['native_calls'])==sum(r['persons']for r in rows)
        and all(c['run_completed']for c in report['native_calls']),'Every unchanged native person inference required')
    pin=write(OUTPUT/'report.json',summary);OUTPUT.chmod(0o500)
    print(json.dumps(dict(status='pass',report=pin,images=50,native_calls=summary['native_calls']),sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--finalize',type=int);modes.add_argument('--native',action='store_true')
    parser.add_argument('--inputs-sha256');parser.add_argument('--inputs-bytes',type=int);args=parser.parse_args()
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(sys.platform=='linux'and os.geteuid()==0 and Path(__file__).resolve()==code/HELPERS[0],'Actual source-bound Linux root CPU driver required')
    if args.prepare:host_prepare(code,revision)
    elif args.finalize is not None:host_finalize(code,revision,args.finalize)
    else:native(code,revision,configuration(code),dict(bytes=args.inputs_bytes,sha256=args.inputs_sha256))

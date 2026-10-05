"""New external full person/object detector banks; no selection or SAM calls.

Host validates rights/acquisition provenance and projects RGB-only inputs.
GPU sees neither reference annotations nor split/rights/publisher metadata.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import bridge_frontend_bindings as binding
import mediapipe_cpu_runtime_verify as rt
from world_reward.prompt_selection import BoxDetection, non_maximum_suppression

ROOT = binding.ROOT
ENTRY = 'run_openimages_joint_pair_gdi'
CONFIG = 'configs/openimages_joint_pair_gdi_v1.json'
ACQUISITION = Path('/srv/world-reward-data/openimages_joint_pair_acquisition_v1')
COHORT = ROOT/'results/openimages-joint-pair-selection-v1/cohort.json'
OUTPUT = ROOT/'results/openimages-joint-pair-gdi-v1'
NATIVE = OUTPUT/'native'
HELPERS = ('infra/openimages_joint_pair_gdi.py', 'infra/run_openimages_joint_pair_gdi.sh', CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/prompt_selection.py',
           'infra/bridge_frontend_bindings.py', 'infra/frontend_selected_assets.py',
           'infra/frontend_sam2_kernel_gate.py', 'configs/frontend_grounding_source_pins.json',
           'configs/frontend_asset_archive_pins.json')
ACQUIRE_HELPERS = ('infra/openimages_joint_pair_acquire.py', 'infra/run_openimages_joint_pair_acquire.sh',
                   'infra/mediapipe_cpu_runtime_verify.py', 'configs/openimages_joint_pair_acquire_v1.json')
ASSETS = {
 'grounding_dino/model.safetensors': ('5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21', 933400872),
 'grounding_dino/config.json': ('eda416dae6f49419ff831b1c190ec430a060b19aae688dbaf2425a075b650608', 1737),
 'grounding_dino/preprocessor_config.json': ('8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60', 457),
 'grounding_dino/special_tokens_map.json': ('b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3', 125),
 'grounding_dino/tokenizer.json': ('d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66', 711396),
 'grounding_dino/tokenizer_config.json': ('d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56', 1237),
 'grounding_dino/vocab.txt': ('07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3', 231508)}


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    rt.write(path, raw, 0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def configuration(code):
    value = rt.strict((code/CONFIG).read_bytes())
    expected = dict(schema='world_reward.openimages_joint_pair_gdi.v1',
        acquisition_revision='e2b6019f502e18694a221aa6b1818804ef3d6d83',
        cohort=dict(bytes=31043, sha256='441bdc57e101cb4e8291ca6bbebc554ca397619e2a892469d9e410a16359a456'),
        manifest=dict(bytes=76232, sha256='57135ee4853afaff9d5820f11128fc2c8da9c42b2cb14b40597909c9f0f0eeb4'),
        image_id=binding.IMAGE, output='results/openimages-joint-pair-gdi-v1', slots=64, acquired=50, missing=14,
        budget_seconds=600, person_query='person.', object_query='object.', confidence=.3, text_threshold=.25,
        nms_iou=.7, challenge_inputs_used=False, quality_verified=False, adoption=False)
    rt.require(value == expected and all(type(value[k]) is type(v) for k,v in expected.items()), 'Frozen external GDI config required')
    return value


def project_acquisition(cohort, manifest, cfg):
    """Only original RGB file/pins/dimensions cross this host→GPU boundary."""
    expected = dict(schema='world_reward.openimages_joint_pair_acquire.v1', status='pass', mode='acquire',
        stage='frozen_external_rgb_acquisition', producer_revision=cfg['acquisition_revision'], cohort_identity=cfg['cohort'],
        no_replacements=True, retry_count=0, replacement_count=0, reference_geometry_read=False,
        challenge_inputs_used=False, quality_verified=False, adoption=False, gpu_used=False, models_loaded=False,
        source_and_inputs_rehashed_after=True, artifacts_rehashed_after=True)
    rt.require(all(type(manifest.get(k)) is type(v) and manifest[k] == v for k,v in expected.items())
        and cohort['schema'] == 'world_reward.openimages_joint_pair_cohort.v1'
        and cohort['producer_revision'] == cfg['acquisition_revision']
        and cohort['source_binding'] == manifest['source_binding'], 'Complete frozen acquisition/cohort PASS required')
    rows = manifest['records']; chosen = cohort['records']
    rt.require(len(rows) == len(chosen) == cfg['slots']
        and [r['slot'] for r in rows] == list(range(cfg['slots'])), 'All frozen slots required')
    images = []; missing = []; seen = set()
    for row, selected in zip(rows, chosen):
        metadata = selected['publisher_metadata']; iid = metadata['ImageID']
        rt.require(re.fullmatch('[0-9a-f]{16}', iid) and iid not in seen
            and selected['slot'] == row['slot'] and row['image_id'] == iid
            and row['publisher_metadata'] == metadata and row['split'] == selected['split'], 'Original ordered acquisition slots differ')
        seen.add(iid)
        if row['status'] != 'acquired':
            rt.require(row['status'] == 'unavailable' and 'image_pin' not in row, 'Missing original slot must remain missing')
            missing.append(iid); continue
        header = row['jpeg_header']
        rt.require(row['creator_grant_verified'] is True and row['publisher_md5_matched'] is True
            and row['rotation'] == metadata['Rotation'] == '0.0' and row['image_transformation'] == 'none_original_file'
            and header['channels'] == 3 and header['header_only'] is True
            and all(type(header[k]) is int and header[k] > 0 for k in ('width','height'))
            and header['width']*header['height'] <= 16 << 20, 'Original qualified JPEG acquisition required')
        images.append(dict(image_id=iid, file=str(ACQUISITION/iid/'rgb.jpg'), width=header['width'],
                           height=header['height'], **row['image_pin']))
    rt.require(len(images) == cfg['acquired'] and len(missing) == cfg['missing'], 'Exact frozen acquired/missing counts required')
    return dict(schema='world_reward.openimages_joint_pair_rgb_inputs.v1', images=images), missing


def acquisition_identity(cfg):
    cohort = rt.pinned(COHORT, cfg['cohort']); manifest = rt.pinned(ACQUISITION/'manifest.json', cfg['manifest'])
    old = ROOT/'jobs'/cfg['acquisition_revision']/'run_openimages_joint_pair_acquire/code'
    source = rt.source(ROOT, old, cfg['acquisition_revision'], 'run_openimages_joint_pair_acquire', ACQUIRE_HELPERS)
    rt.require(source == manifest['source_binding'], 'Actual historical acquisition source/markers required')
    projected, missing = project_acquisition(cohort, manifest, cfg)
    for image in projected['images']:
        pin = {k:image[k] for k in ('bytes','sha256')}
        rt.require(rt.identity(Path(image['file']), 16 << 20) == pin, 'Original acquired RGB bytes differ')
    return projected, dict(cohort=cfg['cohort'], manifest=cfg['manifest'], acquisition_source=source, missing_image_ids=missing)


def runtime_identity(code, *, live=False):
    """Reuse measured runtime primitives; never pretend this is an old entry."""
    kernel = binding.kernel_helper()
    build = binding.pinned(binding.BUILD_REPORT, binding.BUILD_PIN)
    receipt = binding.pinned(binding.KERNEL_REPORT, binding.KERNEL_PIN)
    original = kernel.closure(binding.BUILD_CODE, binding.BUILD_REV, 'run_frontend_grounding_build', kernel.BUILD_HELPERS)
    original_kernel = kernel.closure(binding.KERNEL_CODE, binding.KERNEL_REV, 'run_frontend_sam2_kernel_gate',
        ('infra/frontend_sam2_kernel_gate.py','infra/run_frontend_sam2_kernel_gate.sh',binding.CONFIG))
    rt.require(build['source_binding'] == original and receipt['source_binding'] == original_kernel
        and original['helpers']['infra/frontend_grounding_build.py']['sha256'] == binding.BUILD_SHA
        and original_kernel['helpers']['infra/frontend_sam2_kernel_gate.py'] == binding.KERNEL_SOURCE_PIN
        and binding.identity(code/binding.CONFIG) == original['helpers'][binding.CONFIG] == original_kernel['helpers'][binding.CONFIG],
        'Measured original runtime source/config differ')
    rt.require(build['status'] == 'pass' and build['phase'] == 'complete' and build['producer_revision'] == binding.BUILD_REV
        and build['offline_build_exit_code'] == build['child_probe_exit_code'] == 0
        and build['parent_unchanged_verified'] is True and build['source_rechecked_before_and_after'] is True
        and receipt['status'] == 'pass' and receipt['CUDA_operator_execution_verified'] is True
        and receipt['producer_revision'] == binding.KERNEL_REV and receipt['image_id'] == binding.IMAGE,
        'Actual measured runtime gates required')
    child, parent = build['child_image'], build['parent_image']
    owner = hashlib.sha256((binding.BUILD_REV+original['closure_sha256']).encode()).hexdigest()
    rt.require(child['Id'] == binding.IMAGE and parent['Id'] == kernel.BASE and len(parent['RootFS']['Layers']) == 44
        and child['RootFS']['Layers'][:44] == parent['RootFS']['Layers'] and owner == build['owner'], 'Actual child/parent ownership differs')
    probe = build['private_child_probe_log']
    rt.require(probe['relative_path'] == 'results/frontend-grounding-build-v6/child-CPU-probe.log'
        and binding.identity(ROOT/probe['relative_path'], 128 << 10) == {k:probe[k] for k in ('bytes','sha256')}, 'Bound CPU probe differs')
    contract = binding.selected.load_contract(ROOT, binding.MANIFEST)
    assets = binding.selected_assets(contract, ASSETS)
    card_name='weights/grounding_dino/README.md';card=contract['entries'][card_name]
    rt.require(card['type']=='file' and card['role']=='license_card_or_config'
        and binding.identity(binding.DEST/card_name,1 << 20)=={k:card[k] for k in ('bytes','sha256')},'Original model card required')
    if live:
        kernel.validate_live_image(dict(child_image=child, parent_image=parent, owner=owner))
    return dict(build_report_identity=binding.BUILD_PIN, kernel_report_identity=binding.KERNEL_PIN,
                build_source=original, kernel_source=original_kernel, child_image=child, parent_image=parent,
                selected_manifest_identity=contract['manifest_identity'], extraction_identity=contract['extraction_receipt_identity'],
                model_card_identity={k:card[k] for k in ('bytes','sha256')},assets=assets)


def retained_bank(boxes, scores, labels, image_id, kind, width, height):
    """Native postprocessor order/raw dtype retained; existing NMS only."""
    import numpy as np
    rt.require(type(boxes) is np.ndarray and boxes.ndim == 2 and boxes.shape[1] == 4
        and boxes.dtype.kind == 'f' and np.isfinite(boxes).all()
        and type(scores) is np.ndarray and scores.shape == (len(boxes),) and scores.dtype.kind == 'f'
        and np.isfinite(scores).all() and type(labels) is list and len(labels) == len(boxes)
        and all(type(x) is str for x in labels), 'Complete native postprocessor boxes/scores/labels required')
    raw = tuple(BoxDetection(tuple(map(float,b)),float(s)) for b,s in zip(boxes,scores))
    retained = non_maximum_suppression(raw, width, height, .3, .7)
    slots = []
    for item in retained:
        match = [i for i,x in enumerate(raw) if x == item]
        rt.require(bool(match), 'NMS must preserve an original native proposal')
        slots.append(match[0])
    return dict(raw_boxes=boxes.copy(), raw_scores=scores.copy(), raw_labels=np.asarray(labels,dtype=str),
        retained_boxes=np.asarray([x.box for x in retained],dtype=np.float64).reshape(-1,4),
        retained_scores=np.asarray([x.score for x in retained],dtype=np.float64),
        retained_raw_slots=np.asarray(slots,dtype=np.int64),
        retained_ids=np.asarray([f'image:{image_id}/{kind}/retained:{i:06d}' for i in range(len(retained))],dtype=str))


def observe(images, output, detect, decode, report, persist):
    import numpy as np
    for image in images:
        rgb = decode(image)
        rt.require(type(rgb) is np.ndarray and rgb.dtype == np.uint8 and rgb.shape == (image['height'],image['width'],3),
                   'Exact decoded original RGB grid required')
        digest = hashlib.sha256(rgb.tobytes()).hexdigest(); arrays = {}; record = dict(image_id=image['image_id'],
            width=image['width'], height=image['height'], decoded_rgb_sha256=digest,
            input_file=image['file'],input_identity={k:image[k] for k in ('bytes','sha256')},queries=[])
        for kind,query in (('person','person.'),('object','object.')):
            boxes,scores,labels,raw_boxes,raw_logits = detect(rgb,query)
            rt.require(all(type(x) is np.ndarray and x.dtype.kind == 'f' and np.isfinite(x).all() for x in (raw_boxes,raw_logits)),
                       'Native model proposal/logit arrays must remain finite')
            rt.require(raw_boxes.ndim==3 and raw_boxes.shape[0]==1 and raw_boxes.shape[2]==4
                and raw_logits.ndim==3 and raw_logits.shape[:2]==raw_boxes.shape[:2], 'Native shared proposal/logit axes required')
            rt.require(hashlib.sha256(rgb.tobytes()).hexdigest() == digest, 'Detector mutated original RGB')
            bank = retained_bank(boxes,scores,labels,image['image_id'],kind,image['width'],image['height'])
            arrays.update({kind+'_'+k:v for k,v in bank.items()})
            arrays[kind+'_boxes_original_xyxy'] = bank['retained_boxes']
            arrays[kind+'_detector_scores'] = bank['retained_scores']
            record[kind+'_ids'] = bank['retained_ids'].tolist()
            arrays[kind+'_model_pred_boxes'] = raw_boxes; arrays[kind+'_model_logits'] = raw_logits
            report['native_forward_calls'] += 1
            record['queries'].append(dict(kind=kind, query=query, postprocessor_rows=len(boxes), retained_rows=len(bank['retained_ids'])))
        name = image['image_id']+'.npz'
        with (output/name).open('xb') as stream:
            os.fchmod(stream.fileno(),0o400); np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        record.update(prediction_file=name, prediction_identity=rt.identity(output/name, 32 << 20))
        report['images'].append(record); persist()


def native(code, revision, cfg, projected_pin):
    started = time.monotonic(); source = rt.source(ROOT,code,revision,ENTRY,HELPERS)
    inputs_path = OUTPUT/'inputs.json'; projected = rt.pinned(inputs_path,projected_pin)
    rt.require(set(projected) == {'schema','images'} and projected['schema'] == 'world_reward.openimages_joint_pair_rgb_inputs.v1'
        and len(projected['images']) == 50, 'Only complete projected RGB inputs required')
    rt.require({x.name for x in Path('/sys/class/net').iterdir()} == {'lo'} and os.environ['WR_IMAGE_ID'] == binding.IMAGE,
               'Explicit offline native runtime required')
    report = dict(schema=cfg['schema'], stage='external_all_person_object_gdi_banks',status='fail',phase='provenance',
        producer_revision=revision,source_binding=source,image_id=binding.IMAGE,projected_input_identity=projected_pin,
        native_model_loads=0,native_forward_calls=0,images=[],person_query='person.',object_query='object.',
        confidence=.3,text_threshold=.25,nms_iou=.7,ground_truth_used=False,reference_metadata_read=False,
        split_metadata_read=False,rights_metadata_read=False,challenge_inputs_used=False,network='none',
        sam_calls=0,actor_selection_performed=False,ownership_verified=False,task_target_verified=False,
        quality_verified=False,training_overlap_verified=False,challenge_overlap_verified=False,adoption=False)
    def persist():
        report['elapsed_seconds'] = time.monotonic()-started
    def expired(*_):
        raise TimeoutError('Frozen external detector budget exhausted')
    previous = {s:signal.signal(s,expired) for s in (signal.SIGALRM,signal.SIGTERM)}; signal.alarm(cfg['budget_seconds'])
    try:
        before = runtime_identity(code); report['runtime_binding'] = before
        import torch
        from PIL import Image
        import numpy as np
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        rt.require(torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(), 'Actual H100 required')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        processor=AutoProcessor.from_pretrained(binding.DEST/'weights/grounding_dino',local_files_only=True)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(binding.DEST/'weights/grounding_dino',local_files_only=True).to('cuda').eval()
        report.update(native_model_loads=1,phase='native_forward')
        def detect(rgb,query):
            value=processor(images=Image.fromarray(rgb),text=query,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                prediction=model(**value)
                result=processor.post_process_grounded_object_detection(prediction,value.input_ids,threshold=.3,
                    text_threshold=.25,target_sizes=[rgb.shape[:2]])[0]
            return (result['boxes'].cpu().numpy(),result['scores'].cpu().numpy(),result['text_labels'],
                    prediction.pred_boxes.cpu().numpy(),prediction.logits.cpu().numpy())
        def decode(row):
            path=Path(row['file']); wanted={k:row[k] for k in ('bytes','sha256')}
            rt.require(rt.identity(path,16 << 20)==wanted,'Original RGB changed before decode')
            with Image.open(path) as image:
                rt.require(image.format=='JPEG' and image.mode=='RGB' and image.size==(row['width'],row['height']),
                           'Original RGB JPEG decoder disagrees with acquired header')
                return np.array(image,copy=True)
        observe(projected['images'],NATIVE,detect,decode,report,persist)
        del model,processor; torch.cuda.empty_cache()
        rt.require(report['native_forward_calls']==100 and len(report['images'])==50,'All acquired images/two queries required')
        rt.require(source==rt.source(ROOT,code,revision,ENTRY,HELPERS) and before==runtime_identity(code)
            and rt.identity(inputs_path)==projected_pin,'Original source/runtime/inputs changed')
        for row in projected['images']:
            rt.require(rt.identity(Path(row['file']),16 << 20)=={k:row[k] for k in ('bytes','sha256')},'Original RGB changed after inference')
        for row in report['images']:
            rt.require(rt.identity(NATIVE/row['prediction_file'],32 << 20)==row['prediction_identity'],'Saved native bank changed')
        rt.require(time.monotonic()-started<cfg['budget_seconds'],'Inclusive original budget exceeded')
        report.update(status='pass',phase='complete',source_inputs_assets_rehashed_after=True)
    except BaseException as exc:
        report.update(status='fail',error_type=type(exc).__name__)
        raise
    finally:
        signal.alarm(0);persist();write(NATIVE/'native.json',report);NATIVE.chmod(0o500)
        for sig,handler in previous.items():signal.signal(sig,handler)


def host_prepare(code, revision):
    cfg=configuration(code);source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    projected,acquisition=acquisition_identity(cfg);runtime=runtime_identity(code,live=True)
    rt.canonical(OUTPUT);rt.require(os.uname().nodename=='world-reward-ncc-h100-02'
        and not OUTPUT.exists() and OUTPUT.parent.is_dir(),'Actual VM02 fresh external detector output required')
    OUTPUT.mkdir(mode=0o700);NATIVE.mkdir(mode=0o700);projected_pin=write(OUTPUT/'inputs.json',projected)
    control=dict(source_binding=source,acquisition=acquisition,runtime_binding=runtime,projected_input_identity=projected_pin)
    write(OUTPUT/'host-before.json',control)
    rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS)==source and acquisition_identity(cfg)==(projected,acquisition)
        and runtime_identity(code,live=True)==runtime,'Original source/runtime/RGB changed after preparation')
    print(projected_pin['sha256']+' '+str(projected_pin['bytes']))


def host_finalize(code,revision,exit_code):
    cfg=configuration(code);source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    before=rt.strict((OUTPUT/'host-before.json').read_bytes());projected,acquisition=acquisition_identity(cfg)
    runtime=runtime_identity(code,live=True)
    name='world-reward-oi-joint-gdi-'+revision[:12]
    rt.require(not binding.kernel_helper().command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).strip(),
        'Owned native detector container must be absent before PASS publication')
    rt.require(before['source_binding']==source and before['acquisition']==acquisition and before['runtime_binding']==runtime
        and rt.pinned(OUTPUT/'inputs.json',before['projected_input_identity'])==projected,'Host original controls/inputs changed')
    native_report=rt.strict((NATIVE/'native.json').read_bytes())
    rt.require(exit_code==0 and native_report['status']=='pass' and native_report['phase']=='complete'
        and native_report['producer_revision']==revision and native_report['source_binding']==source
        and native_report['projected_input_identity']==before['projected_input_identity'] and native_report['runtime_binding']==runtime,
        'Complete original native detector PASS required')
    rows=native_report['images'];rt.require([r['image_id'] for r in rows]==[r['image_id'] for r in projected['images']], 'All original image banks required')
    for row in rows:
        rt.require(row['prediction_file']==row['image_id']+'.npz'
            and rt.identity(NATIVE/row['prediction_file'],32 << 20)==row['prediction_identity'],'Original saved bank changed')
    rt.require(native_report['native_model_loads']==1 and native_report['native_forward_calls']==100
        and native_report['source_inputs_assets_rehashed_after'] is True,'Complete one-load/all-image native work required')
    rt.require({p.name for p in NATIVE.iterdir()}=={'native.json',*[r['prediction_file'] for r in rows]}
        and {p.name for p in OUTPUT.iterdir()}=={'inputs.json','host-before.json','native'},'Exclusive native detector outputs required')
    report=dict(schema=cfg['schema'],stage='external_all_person_object_gdi_banks_host',status='pass',producer_revision=revision,
        source_binding=source,acquisition=acquisition,native_report_identity=rt.identity(NATIVE/'native.json',1 << 20),
        projected_input_identity=before['projected_input_identity'],slots=64,acquired=50,missing=14,native_forward_calls=100,
        native_model_loads=1,banks=rows,source_inputs_assets_rehashed_after=True,owned_container_removed=True,
        quality_verified=False,ownership_verified=False,task_target_verified=False,training_overlap_verified=False,
        challenge_overlap_verified=False,adoption=False)
    result=write(OUTPUT/'report.json',report);NATIVE.chmod(0o500);OUTPUT.chmod(0o500)
    print(json.dumps(dict(status='pass',report=result,acquired=50,missing=14,native_forward_calls=100),sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--finalize',type=int);mode.add_argument('--native',action='store_true')
    parser.add_argument('--inputs-sha256');parser.add_argument('--inputs-bytes',type=int);args=parser.parse_args()
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt.require(sys.platform=='linux' and os.geteuid()==0 and Path(__file__).resolve()==code/HELPERS[0],'Actual Azure source-bound detector driver required')
    if args.prepare:host_prepare(code,revision)
    elif args.finalize is not None:host_finalize(code,revision,args.finalize)
    else:native(code,revision,configuration(code),dict(bytes=args.inputs_bytes,sha256=args.inputs_sha256))

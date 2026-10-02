"""Pinned automatic person/bottle RGB masks; no evaluation truth is accessible."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import hand_synthetic_masks as masks
from world_reward.prompt_selection import BoxDetection

SCHEMA='world-reward-joint-rgb-v1'
STAGE='public_joint_rgb_automatic_masks'


def validate_inputs(directory):
    directory=Path(directory); path=directory/'manifest.json'; receipt=masks.identity(path)
    manifest=json.loads(path.read_text())
    if (not isinstance(manifest,dict) or set(manifest)!={'schema','images'} or manifest['schema']!=SCHEMA
            or not isinstance(manifest['images'],list) or len(manifest['images'])!=9):
        raise ValueError('Require exact public RGB-only J1 manifest')
    records=[]
    for index,item in enumerate(manifest['images']):
        clip,frame=divmod(index,3); filename=f'clip_{clip:02d}_frame_{frame:03d}.png'
        if (not isinstance(item,dict) or set(item)!={'file','sha256','width','height'} or item['file']!=filename
                or not isinstance(item['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',item['sha256'])
                or type(item['width']) is not int or type(item['height']) is not int
                or (item['width'],item['height'])!=(1024,768)):
            raise ValueError('Require nine ordered original1024x768 RGB identities only')
        p=directory/filename
        if masks.identity(p)['sha256']!=item['sha256']: raise ValueError('Original RGB SHA mismatch')
        records.append({**item,'path':p,'clip_index':clip,'frame_index':frame})
    return records,receipt


def run(root,report,persist):
    import numpy as np
    from PIL import Image
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    if not torch.cuda.is_available(): raise RuntimeError('Require CUDA automatic masks')
    base=root/'validation/joint_rgb_v1'; records,receipt=validate_inputs(base/'inputs')
    report['input_manifest_sha256']=receipt['sha256']; report['model_assets']=masks.validate_assets(root,report['image_id'])
    dist=metadata.distribution('SAM-2'); direct=dist.read_text('direct_url.json')
    report['sam2_source']=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct) if direct else {})
    processor=AutoProcessor.from_pretrained(root/'weights/grounding_dino',local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(root/'weights/grounding_dino',local_files_only=True).cuda().eval()
    try:
        for item in records:
            record={k:item[k] for k in ('file','clip_index','frame_index')}; record['rgb_sha256']=item['sha256']
            report['records'].append(record); persist()
            with Image.open(item['path']) as image:
                if image.format!='PNG' or image.mode!='RGB' or image.size!=(1024,768): raise ValueError('Original RGB decode contract failed')
                for label,query in (('human','person.'),('object','bottle.')):
                    entry=processor(images=image,text=query,return_tensors='pt').to('cuda')
                    with torch.inference_mode(): result=processor.post_process_grounded_object_detection(detector(**entry),entry.input_ids,
                        threshold=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,target_sizes=[(768,1024)])[0]
                    boxes=[BoxDetection(tuple(b),float(s)) for b,s in zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist())]
                    chosen=masks.select_person(boxes,1024,768)
                    record[label+'_box']=list(chosen.box); record[label+'_detector_score']=chosen.score
                    del entry,result; persist()
    finally: del detector,processor; torch.cuda.empty_cache()
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(root/'weights/sam2/sam2.1_hiera_large.pt'),device='cuda',mode='eval'))
    try:
        for item,record in zip(records,report['records']):
            with Image.open(item['path']) as image,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                predictor.set_image(np.asarray(image))
                for label in ('human','object'):
                    predicted,scores,_=predictor.predict(box=np.array(record[label+'_box'],np.float32),multimask_output=False)
                    mask=masks.binary_mask(predicted,scores,1024,768)
                    filename=Path(item['file']).stem+'_'+label+'.png'; target=base/'automatic_masks'/filename
                    with target.open('xb') as stream: Image.fromarray(mask,mode='L').save(stream,format='PNG')
                    record[label+'_mask_file']=filename; record[label+'_mask_sha256']=masks.identity(target)['sha256']
                    record[label+'_mask_pixels']=int(np.count_nonzero(mask)); persist()
                predictor.reset_predictor()
    finally: del predictor; torch.cuda.empty_cache()
    _,after=validate_inputs(base/'inputs')
    if after!=receipt: raise ValueError('Public input manifest changed during masks')


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require remote network-none automatic masks')
    root=Path(os.environ['WR_ROOT']); out=root/'validation/joint_rgb_v1/automatic_masks'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()): raise FileExistsError('Require fresh reserved mask output')
    report={'stage':STAGE,'status':'fail','frames':9,'records':[],'code_revision':os.environ['WR_CODE_REVISION'],
        'image_id':os.environ['WR_IMAGE_ID'],'script_sha256':masks.identity(Path(__file__))['sha256'],
        'shared_mask_helper_sha256':masks.identity(Path(masks.__file__))['sha256'],'budget_seconds':120,
        'private_truth_read':False,'ground_truth_used':False,'challenge_inputs_used':False,'hand_labeled_test':False,'oracle_modes':[],
        'human_query':'person.','object_query':'bottle.','accuracy_verified':False,
        'detector_revision':masks.DETECTOR_REVISION,'sam2_weights_revision':masks.SAM2_REVISION,
        'confidence':masks.CONFIDENCE,'text_threshold':masks.TEXT_THRESHOLD,'nms_iou':masks.NMS_IOU,
        'ambiguity_margin':masks.AMBIGUITY_MARGIN}
    if not re.fullmatch('[0-9a-f]{40}',report['code_revision']) or not re.fullmatch('sha256:[0-9a-f]{64}',report['image_id']):
        raise ValueError('Require immutable mask source/image identities')
    start=time.perf_counter()
    with (out/'report.json').open('x') as stream:
        def persist():
            report['elapsed_seconds']=time.perf_counter()-start; stream.seek(0); json.dump(report,stream,allow_nan=False)
            stream.write('\n'); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError('Frozen120s automatic joint mask deadline')
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired); signal.alarm(120)
        try: persist(); run(root,report,persist); report['status']='pass'
        except Exception as exc: report.update(error_type=type(exc).__name__,error=str(exc)); raise
        finally: signal.alarm(0); persist()


if __name__=='__main__': main()

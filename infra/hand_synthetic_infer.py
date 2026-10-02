"""Body/full SAM inference on six public synthetic RGBs and automatic masks.

Only the public input manifest/images, automatic masks, pinned code and model
assets are accessible. Truth/camera labels/private render outputs are not read
or mounted. Raw 266 logits are never geometry controls; fresh parameter-block
MHR decode supplies controls and global rotations after hand fusion. This is
prediction production, not an accuracy or realistic-domain validation gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import time

import numpy as np

import body_smoke as body
from world_reward.data import sha256

WIDTH, HEIGHT, CASES = 1024, 768, 6
MAX_SECONDS = 180.
BODY_SHA = 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'
BODY_BYTES = 2109129346
STAGE = 'public_synthetic_rgb_sam_body_and_full_predictions'


def _write(path,report):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def public_inputs(root):
    """Exact public whitelist; no private truth/schema fields or traversal."""
    base=root/'validation/hands_rgb_v1'; inputs=base/'inputs'; masks=base/'automatic_masks'
    for folder in (inputs,masks):
        if folder.is_symlink() or not folder.is_dir(): raise ValueError('Require regular public input directories')
        for ancestor in folder.parents:
            if ancestor.is_symlink():raise ValueError('Public path ancestor is a symlink')
            if ancestor==root:break
    manifest_path=inputs/'manifest.json';mask_report_path=masks/'report.json'
    for path in (manifest_path,mask_report_path):
        if path.is_symlink() or not path.is_file(): raise ValueError('Missing regular public manifest/report')
    manifest=json.loads(manifest_path.read_text());report=json.loads(mask_report_path.read_text())
    if not isinstance(manifest,dict) or not isinstance(report,dict):raise ValueError('Public manifests must be JSON objects')
    if (set(manifest)!={'schema','images'} or manifest['schema']!='world-reward-hands-rgb-inputs-v1'
            or not isinstance(manifest['images'],list) or len(manifest['images'])!=CASES):
        raise ValueError('Require exact six-image RGB-only input schema')
    expected={'schema':'world-reward-hands-automatic-masks-v1','stage':'hand_synthetic_automatic_person_masks',
              'status':'pass','frames':6,'ground_truth_used':False,'challenge_inputs_used':False,
              'hand_labeled_test':False,'oracle_modes':[],'person_query':'person.'}
    if (any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in expected.items())
            or report.get('input_manifest',{}).get('sha256')!=sha256(manifest_path)
            or report.get('input_manifest',{}).get('bytes')!=manifest_path.stat().st_size
            or len(report.get('images',[]))!=CASES):
        raise ValueError('Automatic six-mask provenance must pass, never abstain/fallback')
    records=[]
    for index,(image,mask) in enumerate(zip(manifest['images'],report['images'])):
        filename=f'case_{index:03d}.png'
        if not isinstance(image,dict) or not isinstance(mask,dict):raise ValueError('Public image records must be objects')
        if (set(image)!={'file','sha256','width','height'} or image.get('file')!=filename
                or type(image.get('width')) is not int or type(image.get('height')) is not int
                or (image['width'],image['height'])!=(WIDTH,HEIGHT)
                or not isinstance(image.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',image['sha256'])
                or type(mask.get('frame_index')) is not int or mask.get('frame_index')!=index
                or mask.get('file')!=filename or mask.get('mask_file')!=filename
                or mask.get('rgb_sha256')!=image['sha256'] or mask.get('width')!=WIDTH or mask.get('height')!=HEIGHT):
            raise ValueError('Public image/mask fixed-grid identity mismatch')
        image_path,mask_path=inputs/filename,masks/filename
        for path,digest in ((image_path,image['sha256']),(mask_path,mask.get('mask_sha256'))):
            if (path.is_symlink() or not path.is_file() or not isinstance(digest,str)
                    or not re.fullmatch(r'[0-9a-f]{64}',digest) or sha256(path)!=digest):
                raise ValueError('Public RGB/mask artifact SHA mismatch')
        records.append({'frame_index':index,'image_path':image_path,'mask_path':mask_path,
                        'image_sha256':image['sha256'],'mask_sha256':mask['mask_sha256']})
    if {p.name for p in inputs.iterdir()}!={'manifest.json',*[f'case_{i:03d}.png' for i in range(CASES)]}:
        raise ValueError('Input directory must expose RGB files only')
    return records,{'input_manifest':sha256(manifest_path),'mask_report':sha256(mask_report_path)}


def derived_bbox(rgb,mask):
    """Actual binary-mask bbox; never detector-only or whole-image fallback."""
    if (rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8 or mask.shape!=(HEIGHT,WIDTH)
            or mask.dtype!=np.uint8 or not np.isin(mask,[0,255]).all()):
        raise ValueError('Original-grid uint8 RGB and binary automatic mask required')
    ys,xs=np.where(mask>0)
    if len(xs)<20: raise ValueError('Automatic mask absent; no inference fallback')
    box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.float32)
    if np.min(box[2:]-box[:2])<=1: raise ValueError('Invalid automatic-mask bbox')
    return box,(mask>0).astype(np.uint8)[...,None]


def load_model(root,torch):
    """Reuse the verified strict loader/asset proof, no vendor or shared edits."""
    source=body._source_identity(root);directory,assets=body._body_assets(root)
    if assets['model.ckpt']['sha256']!=BODY_SHA or assets['model.ckpt']['bytes']!=BODY_BYTES:
        raise RuntimeError('Require the exact verified Body release checkpoint')
    repo=root/'weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main'
    original_hub,calls=body._install_local_dinov3_loader(torch,repo)
    checkpoint_loading={}
    try:
        import sam_3d_body
        from sam_3d_body import build_models
        if Path(sam_3d_body.__file__).resolve()!=Path('/workspace/v2d_sam3d_body/lib/sam_3d_body/__init__.py'):
            raise RuntimeError('Unexpected installed SAM Body package')
        original_loader=build_models.load_state_dict
        asset_state=torch.jit.load(str(directory/'assets/mhr_model.pt'),map_location='cpu').state_dict()
        def strict_loader(module,state_dict,strict=False,logger=None):
            checkpoint_loading.update(body._load_checkpoint_with_asset_buffers(module,state_dict,original_loader,torch,
                                                                               explicit_asset_state=asset_state))
            prepare=module.backbone.encoder.prepare_tokens_with_masks
            def unmasked(x,masks=None):
                if masks is not None:raise RuntimeError('Retained zero DINO token only permitted for unmasked RGB')
                return prepare(x,masks=None)
            module.backbone.encoder.prepare_tokens_with_masks=unmasked
        build_models.load_state_dict=strict_loader
        try:
            model,config=sam_3d_body.load_sam_3d_body(checkpoint_path=str(directory/'model.ckpt'),device='cuda',
                                                   mhr_path=str(directory/'assets/mhr_model.pt'))
        finally:build_models.load_state_dict=original_loader
    finally:torch.hub.load=original_hub
    if len(calls)!=1 or config.MODEL.BACKBONE.TYPE!=calls[0]:raise RuntimeError('DINO strict local-load ABI changed')
    estimator=sam_3d_body.SAM3DBodyEstimator(model,config)
    faces=np.asarray(estimator.faces)
    if faces.shape!=(36874,3) or faces.dtype.kind not in 'iu' or np.min(faces)<0 or np.max(faces)>=18439:
        raise RuntimeError('Native MHR face topology ABI changed')
    partitions={}
    for side in ('left','right'):
        value=getattr(model.head_pose,'hand_joint_idxs_'+side,None)
        if not torch.is_tensor(value):raise RuntimeError('Actual hand control map missing')
        decoded=value.detach().cpu().numpy()
        if decoded.ndim!=1 or decoded.dtype.kind not in 'iu':raise RuntimeError('Hand control map must be an integer vector')
        partitions[side]=decoded.astype(np.int64)
    flat=np.r_[partitions['left'],partitions['right']]
    if flat.shape!=(54,) or len(np.unique(flat))!=54 or not np.array_equal(np.sort(flat),np.arange(68,122)):
        raise RuntimeError('Head hand maps do not partition native controls68:122')
    names=[]
    character=getattr(model.head_pose.mhr,'character_torch',None)
    transform=getattr(character,'parameter_transform',None)
    actual=getattr(transform,'parameter_names',None)
    if actual is not None:
        names=list(actual)
        if len(names)<204 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise RuntimeError('Actual MHR control-name metadata invalid')
        names=names[:204]
    prompt=config.MODEL.PROMPT_ENCODER
    metadata={'body_revision':body.BODY_REVISION,'body_assets':assets,'inference_source_identity':source,
              'upstream_revision':body.UPSTREAM_REVISION,'dinov3_revision':body.DINOV3_REVISION,
              'backbone':calls[0],'checkpoint_loading':checkpoint_loading,
              'hand_indices':{k:v.tolist() for k,v in partitions.items()},'native_first204_control_names':names,
              'reference_control_names_compared':False,'decoder_mask_config':{'enabled':bool(prompt.ENABLE),
               'mask_embed_type':prompt.MASK_EMBED_TYPE,'mask_prompt':prompt.get('MASK_PROMPT','v1')}}
    return model,estimator,faces.astype(np.int64),metadata


def decode_prediction(torch,model,prediction,mode):
    """Fresh exact native block decode; stale fusion rotation cache excluded."""
    for key,shape in body.PARAMETER_SHAPES.items():
        value=prediction.get(key)
        if not torch.is_tensor(value) or tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise RuntimeError('Native prediction shape/finite ABI failed: '+key)
    if torch.count_nonzero(prediction['expr_params']).item()!=0:raise RuntimeError('Expressions must remain zero')
    if mode=='full' and torch.count_nonzero(prediction['pred_pose_raw']).item()!=0:
        raise RuntimeError('Full fused raw266 must be invalidated, never decoder input')
    with torch.inference_mode():
        native,keypoints,joints,controls,rotations=body._native_forward_from_blocks(model.head_pose,prediction)
        if (native.shape!=(1,18439,3) or joints.shape!=(1,127,3) or controls.shape!=(1,204)
                or rotations.shape!=(1,127,3,3) or keypoints.ndim!=3 or keypoints.shape[0]!=1
                or keypoints.shape[1]<70 or keypoints.shape[2]!=3
                or any(not torch.isfinite(x).all() for x in (native,keypoints,joints,controls,rotations))):
            raise RuntimeError('Fresh native block decoder ABI changed')
        flip=torch.tensor([1.,-1.,-1.],device=native.device,dtype=native.dtype)
        errors={'vertices_m':float(torch.linalg.vector_norm(native[0]*flip-prediction['pred_vertices'],dim=-1).max()),
                'joints_m':float(torch.linalg.vector_norm(joints[0]*flip-prediction['pred_joint_coords'],dim=-1).max()),
                'keypoints_m':float(torch.linalg.vector_norm(keypoints[0,:70]*flip-prediction['pred_keypoints_3d'],dim=-1).max()),
                'controls':float((controls[0]-prediction['mhr_model_params']).abs().max())}
    if any(not np.isfinite(x) or x>1e-5 for x in errors.values()):raise RuntimeError('Fresh native decoder parity failed')
    cpu=lambda x:x.detach().float().cpu().numpy()
    translation=cpu(prediction['pred_cam_t'])
    vertices=cpu(native[0]*flip)+translation
    joint_camera=cpu(joints[0]*flip)+translation
    if translation[2]<=0 or not np.isfinite(vertices).all():raise RuntimeError('Nonpositive camera translation')
    rotation_values=cpu(rotations[0])
    if (not np.allclose(rotation_values@rotation_values.swapaxes(-1,-2),np.eye(3),atol=1e-4,rtol=0)
            or not np.allclose(np.linalg.det(rotation_values),1.,atol=1e-4,rtol=0)):
        raise RuntimeError('Fresh native joint rotations must be proper SO3')
    output={key:cpu(prediction[key]) for key in ('global_rot','body_pose_params','hand_pose_params','scale_params','shape_params','expr_params')}
    output.update(vertices_camera_m=vertices,joints_camera_m=joint_camera,joint_global_rotations=rotation_values,
                  mhr_model_params=cpu(controls[0]),pred_cam_t=translation,focal_length=cpu(prediction['focal_length']))
    return output,errors


def run_inference(root,report,path):
    records,hashes=public_inputs(root)
    if hashes!=report['public_hashes']:raise ValueError('Public inputs changed after parent preflight')
    import torch
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU inference fallback')
    report['phase']='model_load';_write(path,report)
    model,estimator,faces,metadata=load_model(root,torch);report['model']=metadata
    arrays={'body':[],'full':[]}
    for record in records:
        with Image.open(record['image_path']) as image:
            if image.mode!='RGB':raise ValueError('Public image must be explicit RGB')
            rgb=np.asarray(image).copy()
        with Image.open(record['mask_path']) as image:
            if image.mode!='L':raise ValueError('Automatic person mask must be one-channel uint8 PNG')
            mask=np.asarray(image).copy()
        box,modelmask=derived_bbox(rgb,mask)
        rgb_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
        for mode in ('body','full'):
            report.update(phase='inference',current_case=record['frame_index'],current_mode=mode);_write(path,report)
            with torch.inference_mode():
                predictions=estimator.process_one_image(img=rgb,bboxes=box[None],masks=modelmask,cam_int=None,inference_type=mode)
            if not isinstance(predictions,list) or len(predictions)!=1:raise RuntimeError('Exactly one person prediction required')
            values,errors=decode_prediction(torch,model,predictions[0],mode)
            focal=float(values['focal_length'])
            if not np.isclose(focal,np.hypot(WIDTH,HEIGHT),rtol=1e-6,atol=1e-4):raise RuntimeError('RGB-only focal ABI changed')
            arrays[mode].append(values)
            report['calls'].append({'case_index':record['frame_index'],'inference_type':mode,'image_sha256':record['image_sha256'],
                'decoded_rgb_sha256':rgb_hash,'mask_sha256':record['mask_sha256'],'bbox_xyxy':box.tolist(),
                'native_forward_errors':errors,'camera_intrinsics_supplied':False})
            _write(path,report)
    for mode in ('body','full'):
        out={key:np.stack([r[key] for r in arrays[mode]]) for key in arrays[mode][0]}
        out.update(faces=faces,frame_index=np.arange(CASES,dtype=np.int64))
        target=path.parent/f'predictions_{mode}.npz'
        with target.open('xb') as h:np.savez_compressed(h,**out)
        report['prediction_sha256'][mode]=sha256(target)
    if public_inputs(root)[1]!=hashes:raise ValueError('Public inputs changed during inference')
    if any(report['calls'][2*i]['decoded_rgb_sha256']!=report['calls'][2*i+1]['decoded_rgb_sha256'] for i in range(CASES)):
        raise RuntimeError('Body/full did not consume identical decoded RGB')
    report.update(status='pass',phase='complete',actual_network_inference=True,actual_native_parameter_block_forward=True,
                  cases=CASES,calls_completed=len(report['calls']))
    _write(path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require isolated Linux GPU inference with network none')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MOMENTUM_ENABLED='0',WANDB_MODE='disabled')
    root=Path(os.environ['WR_ROOT']);output=root/'validation/hands_rgb_v1/predictions';path=output/'report.json'
    if args.worker:
        report=json.loads(path.read_text());nonce=os.environ.get('WR_HAND_INFER_NONCE','')
        if (report.get('status')!='running' or report.get('stage')!=STAGE or report.get('script_sha256')!=sha256(Path(__file__))
                or report.get('body_helper_sha256')!=sha256(Path(body.__file__))
                or report.get('producer_revision')!=os.environ.get('WR_CODE_REVISION')
                or not nonce or report.get('nonce_sha256')!=hashlib.sha256(nonce.encode()).hexdigest()):
            raise RuntimeError('Require parent-created running report; frozen outputs immutable')
        try:run_inference(root,report,path)
        except Exception as exc:
            report.update(status='fail',error_type=type(exc).__name__,error=str(exc));_write(path,report);raise
        return
    reserved=os.environ.get('WR_HAND_OUTPUT_RESERVED')=='1'
    if output.is_symlink() or (output.exists() and (not reserved or not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError('Frozen synthetic predictions exist')
    if reserved and not output.is_dir():raise ValueError('Reserved output must be a newly created empty mount')
    records,hashes=public_inputs(root)
    image=os.environ.get('WR_IMAGE_ID','');revision=os.environ.get('WR_CODE_REVISION','')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image) or not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Require immutable image and source revision')
    nonce=secrets.token_hex(32)
    report={'stage':STAGE,'status':'running','phase':'prerequisites','public_hashes':hashes,'image_id':image,
            'producer_revision':revision,'script_sha256':sha256(Path(__file__)),'body_helper_sha256':sha256(Path(body.__file__)),
            'nonce_sha256':hashlib.sha256(nonce.encode()).hexdigest(),'private_truth_read':False,
            'challenge_inputs_used':False,'hand_labeled_test':False,'oracle_modes':[],
            'accuracy_verified':False,'adoption_performed':False,'photorealistic_domain_validation':False,
            'prompt_mode':'automatic_binary_person_mask_and_derived_bbox_no_fallback','cam_int':'None_RGB_size_FOV',
            'geometry_units':'metres','geometry_frame':'SAM_camera_x_right_y_down_z_forward',
            'native_control_dimension':204,'raw266_used':False,'rotation_source':'fresh_native_parameter_block_decode',
            'identity_per_image_predicted':True,'shared_identity_fitted':False,'network':'none','device':'cuda',
            'max_worker_seconds':MAX_SECONDS,'calls':[],'prediction_sha256':{},'body_checkpoint_sha256':BODY_SHA}
    if not reserved:output.mkdir(exist_ok=False)
    with path.open('x') as h:h.write(json.dumps(report,indent=2)+'\n')
    started=time.perf_counter()
    try:
        process=subprocess.run([sys.executable,str(Path(__file__)),'--worker'],check=False,timeout=MAX_SECONDS,
                               env=os.environ|{'WR_HAND_INFER_NONCE':nonce})
    except subprocess.TimeoutExpired:
        report=json.loads(path.read_text());report.update(status='fail',error_type='TimeoutError',
            error='Hard180s inference budget exceeded',elapsed_seconds=time.perf_counter()-started);_write(path,report);raise
    report=json.loads(path.read_text());report['elapsed_seconds']=time.perf_counter()-started;_write(path,report)
    if process.returncode or report['status']!='pass':raise RuntimeError(f"Synthetic inference failed: {report.get('error','incomplete')}")
    print(json.dumps({'stage':STAGE,'status':'pass','calls':report['calls_completed'],'accuracy_verified':False}))


if __name__=='__main__':main()

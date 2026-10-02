"""Single versus native three-view reconstruction from public RGB-derived inputs.

Stores actual raw decoder geometry and proves exporter/native-camera parity.
No private synthesis assets, pose fitting, scale correction or quality selection.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import multiview_full_gate as full
import multiview_gauge_gate as gauge
from world_reward.data import sha256

STAGE = 'public_object_rgb_single_and_three_view_proposals'


def require_hash(path, digest):
    if (not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)
            or path.is_symlink() or not path.is_file() or path.resolve()!=path.absolute() or sha256(path)!=digest):
        raise ValueError('Frozen regular public artifact mismatch')


def observation_arrays(values, obj, view):
    rgb, mask, points = (np.asarray(values[k]) for k in ('rgb','mask','pointmap'))
    if (rgb.shape!=(384,512,3) or rgb.dtype!=np.uint8 or mask.shape!=(384,512) or mask.dtype!=np.bool_
            or not mask.any() or points.shape!=(3,384,512) or points.dtype!=np.float32
            or not np.isfinite(points).all() or np.any(points[2]<=0)):
        raise ValueError('Require actual finite OpenCV pointmap and automatic mask at original grid')
    for key,expected in (('object_index',obj),('frame_index',view)):
        a=np.asarray(values[key])
        if a.shape!=() or a.dtype.kind not in 'iu' or int(a)!=expected:raise ValueError('Public object/view identity differs')
    K=np.asarray(values['K'])
    if not np.allclose(K,[[640,0,256],[0,640,192],[0,0,1]],rtol=0,atol=1e-4):
        raise ValueError('Require RGB-size-derived camera, never private calibration')
    return rgb,mask,points


def load_observations(root):
    folder=root/'validation/objects_rgb_v1/observations';path=folder/'report.json';digest=sha256(path)
    require_hash(path,digest);report=json.loads(path.read_text())
    expected={'stage':'public_object_rgb_automatic_observations','status':'pass','object_cases':2,
              'train_views':[0,2,4],'private_truth_read':False,'challenge_inputs_used':False}
    if any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in expected.items()):
        raise ValueError('Require actual public RGB-only observation producer')
    outputs=report.get('outputs',[])
    if len(outputs)!=6:raise ValueError('Require six frozen training observations')
    result={0:[],1:[]}
    for record,(obj,view) in zip(outputs,((o,v) for o in range(2) for v in (0,2,4))):
        filename=f'object_{obj:02d}_view_{view:02d}.npz'
        if record.get('file')!=filename or record.get('object_index')!=obj or record.get('view_index')!=view:
            raise ValueError('Public training observation order differs')
        require_hash(folder/filename,record.get('sha256'))
        with np.load(folder/filename,allow_pickle=False) as data:
            result[obj].append(observation_arrays(data,obj,view))
    return result,report,digest


def export_actual(result, destination, name, torch):
    """Actual raw decoder tensor parity, never inverse-export tautology."""
    import trimesh
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    raw=result['mesh'][0]
    vertices=raw.vertices.float().detach().cpu().numpy()
    faces=raw.faces.detach().cpu().numpy()
    mesh=result['glb']
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu'
            or not np.array_equal(np.asarray(mesh.vertices),vertices@gauge.A)
            or not np.array_equal(np.asarray(mesh.faces),faces)):
        raise ValueError('Actual decoder to GLB parity failed')
    q,t,s=(result[k].detach().float().cpu().reshape(1,-1) for k in ('rotation','translation','scale'))
    if tuple(q.shape)!=(1,4) or tuple(t.shape)!=(1,3) or tuple(s.shape)!=(1,3) or not torch.equal(s,s[:,0:1].expand_as(s)):
        raise ValueError('Require actual isotropic decoded scale, no averaging or permutation repair')
    pose={'rotation':q[0].numpy().tolist(),'translation':t[0].numpy().tolist(),'scale':s[0].numpy().tolist()}
    gauge.pose_arrays(pose)
    with torch.inference_mode():
        Q=quaternion_to_matrix(q)
        transformed=compose_transform(s,Q,t).transform_points(torch.from_numpy(vertices)[None])[0].numpy()@gauge.N
    analytical=gauge.native_order_points(np.asarray(mesh.vertices),pose)
    maximum=float(np.abs(transformed-analytical).max()); tolerance=16*np.finfo(np.float32).eps*max(1.,float(np.abs(transformed).max()))
    if maximum>tolerance:raise ValueError('Actual imported native camera transform differs from source-order bridge')
    serialized=mesh.export(file_type='glb');reload=trimesh.load(io.BytesIO(serialized),file_type='glb',force='mesh',process=False)
    if not np.array_equal(reload.vertices,vertices@gauge.A) or not np.array_equal(reload.faces,faces):
        raise ValueError('Actual GLB serialization changed decoded vertices or faces')
    target=destination/(name+'.npz')
    with target.open('xb') as stream:
        np.savez_compressed(stream,raw_decoder_vertices=vertices,glb_vertices=np.asarray(mesh.vertices),faces=faces,
            vertices_camera_m=transformed,rotation_wxyz=q[0].numpy(),translation=t[0].numpy(),scale=s[0].numpy())
    with (destination/(name+'.glb')).open('xb') as stream:stream.write(serialized)
    return {'file':target.name,'sha256':sha256(target),'glb_sha256':sha256(destination/(name+'.glb')),
            'vertices':len(vertices),'faces':len(faces),'watertight':bool(mesh.is_watertight),
            'winding_consistent':bool(mesh.is_winding_consistent),'signed_volume_camera_m3':float(trimesh.Trimesh(transformed,faces,process=False).volume),
            'actual_raw_decoder_parity_verified':True,'native_camera_transform_parity_verified':True,
            'isotropic_scale_verified':True,'camera_parity_max_error_m':maximum,'camera_parity_tolerance_m':tolerance}


def run(root,report,path):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from body_smoke import _pinned_checkout
    observations,receipt,digest=load_observations(root)
    report['observation_report_sha256']=digest
    source,source_receipt=full.prep.validate_source(root)
    report['source_manifest']=source_receipt
    report['ss_prerequisite']=full.validate_ss(root,source_receipt,report['image_id'])
    texts={}
    for relative,(expected,size) in gauge.SOURCES.items():
        p=source/relative;require_hash(p,expected)
        if p.stat().st_size!=size:raise ValueError('Pinned native gauge source size differs')
        texts[relative]=p.read_text()
    report['source_order']=gauge.source_contract(texts)
    workspace=root/'weights/sam3d/hf-download/checkpoints';report['model_assets']={}
    for suffix,inventory in (('.ckpt',full.CHECKPOINTS),('.yaml',full.YAMLS)):
        for name,(digest_value,size) in inventory.items():
            p=workspace/(name+suffix);require_hash(p,digest_value)
            if p.stat().st_size!=size:raise ValueError('Pinned Objects asset size differs')
            report['model_assets'][name+suffix]={'sha256':digest_value,'bytes':size}
    repository=root/'weights/sam3d/torch_home/hub/facebookresearch_dinov2_main'
    _pinned_checkout(repository,full.ss.DINO_REVISION)
    auxiliary=json.loads((root/'results/auxiliary-assets.json').read_text())
    for record in auxiliary['checkpoints']:
        if '_reg4_' in record['filename']:
            require_hash(root/'weights/sam3d/torch_home/hub/checkpoints'/record['filename'],record['sha256'])
    config=full.constructor_config(OmegaConf.to_container(OmegaConf.load(workspace/'pipeline.yaml'),resolve=True),workspace)
    original=torch.hub.load;requests=[]
    torch.hub.load=full.local_hub_loader(original,repository,requests)
    try:pipe=instantiate(config)
    finally:torch.hub.load=original
    if pipe.compile_model or pipe.depth_model is not None:raise ValueError('Require offline external-pointmap native constructor')
    def no_depth(*args,**kwargs):raise RuntimeError('No internal depth model or fallback permitted')
    pipe.depth_model=no_depth;torch.cuda.reset_peak_memory_stats()
    common=dict(seed=42,stage1_inference_steps=50,stage2_inference_steps=25,decode_formats=['gaussian','mesh'],
                use_stage1_distillation=False,use_stage2_distillation=False,stage1_only=False,
                with_mesh_postprocess=False,with_texture_baking=False,use_vertex_color=True)
    report.update(native_constructor_verified=True,configuration=common,proposals=[])
    for obj in range(2):
        images,masks,maps=zip(*observations[obj])
        for mode in ('single','three_view'):
            report['active_object']=obj;report['active_mode']=mode;persist(path,report)
            full.ss._seed(torch)
            with torch.no_grad():
                if mode=='single':result=pipe.run(images[0],masks[0],pointmap=torch.from_numpy(maps[0]),with_layout_postprocess=False,**common)
                else:result=pipe.run_multi_view(list(images),list(masks),view_pointmaps=list(maps),num_samples=1,mode='multidiffusion',ss_weighting=False,weighting_config=None,**common)
            proof=export_actual(result,path.parent,f'object_{obj:02d}_{mode}',torch)
            report['proposals'].append({'object_index':obj,'mode':mode,**proof});del result
            torch.cuda.synchronize();persist(path,report)
            if torch.cuda.max_memory_allocated()>80*1024**3:raise RuntimeError('Frozen80GiB allocation budget exceeded')
    require_hash(root/'validation/objects_rgb_v1/observations/report.json',digest)
    imported={name:full.ss._identity(Path(module.__file__)) for name,module in sys.modules.copy().items()
              if (name=='sam3d_objects' or name.startswith('sam3d_objects.')) and getattr(module,'__file__',None)}
    if any(not Path(r['path']).resolve().is_relative_to(source.resolve()) for r in imported.values()):
        raise ValueError('Mixed native vendor imports')
    report.update(status='pass',imported_source=imported,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  actual_raw_decoder_parity_verified=True,native_camera_transform_parity_verified=True)


def persist(path,report):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(report,allow_nan=False)+'\n');temporary.replace(path)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require isolated remote GPU generation')
    root=Path(os.environ['WR_ROOT']);folder=root/'validation/objects_rgb_v1/proposals';path=folder/'report.json'
    if folder.is_symlink() or not folder.is_dir() or any(folder.iterdir()) or os.environ.get('WR_OBJECT_OUTPUT_RESERVED')!='1':
        raise FileExistsError('Require exclusive newly reserved public proposal output')
    revision,image=os.environ.get('WR_CODE_REVISION',''),os.environ.get('WR_IMAGE_ID','')
    if not re.fullmatch(r'[0-9a-f]{40}',revision) or not re.fullmatch(r'sha256:[0-9a-f]{64}',image):raise ValueError('Require immutable source/image')
    report={'stage':STAGE,'status':'fail','code_revision':revision,'image_id':image,'script_sha256':sha256(Path(__file__)),
            'private_truth_read':False,'challenge_inputs_used':False,'hand_labeled_test':False,'adoption_performed':False,
            'accuracy_verified':False,'network':'none','budget_seconds':300,'generation_calls':4,'train_views':[0,2,4],
            'metric_scale_accuracy_verified':False,'all_view_pose_tracking_verified':False}
    started=time.perf_counter()
    def expired(*args):raise TimeoutError('Frozen300s generation budget exceeded')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    try:run(root,report,path)
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);report['elapsed_seconds']=time.perf_counter()-started;persist(path,report)
    print(json.dumps({'stage':STAGE,'status':report['status'],'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':main()

"""Fixed predicted meshes: monocular depth ICP versus RGB LK mesh-PnP.

Only frozen public observations/anchor predictions are exposed. All eight original
frames remain; unsupported PnP abstains to the measured ICP baseline, not stasis.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import object_synthetic_generate as native
from camera_render import _opencv_camera, _mesh_inputs, raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.rigid_alignment import align_observed_points
from world_reward.rgb_pose_tracking import track_rgb_pose

STAGE='public_fixed_mesh_rgb_motion_proposals'
K=np.array([[640.,0,256.],[0,640.,192.],[0,0,1.]])


def raster_attachments(vertices,faces,pixels,face,bary):
    """Real raster perspective barycentrics, never depth/GT nearest neighbours."""
    ids=face[pixels[:,1],pixels[:,0]];weights=bary[pixels[:,1],pixels[:,0]]
    if (np.any(ids<0) or np.any(ids>=len(faces)) or not np.isfinite(weights).all()
            or not np.allclose(weights.sum(1),1,atol=1e-5,rtol=0) or np.any(weights < -1e-5)):
        raise ValueError('Actual predicted raster attachment invalid')
    points=np.sum(vertices[faces[ids]]*weights[:,:,None],axis=1)
    projected=points[:,:2]/points[:,2,None]*[640.,640.]+[256.,192.]
    if len(points) and np.max(np.linalg.norm(projected-(pixels+.5),axis=1))>1e-3:
        raise ValueError('Predicted raster/OpenCV pixel-centre attachment parity failed')
    return points


def load_inputs(root):
    base=root/'validation/object_motion_v1';a=base/'anchors/report.json';o=base/'observations/report.json'
    ah,oh=sha256(a),sha256(o);native.require_hash(a,ah);native.require_hash(o,oh)
    ar,observations=json.loads(a.read_text()),json.loads(o.read_text())
    if (ar.get('stage')!='public_object_motion_fixed_anchor_generation' or ar.get('status')!='pass'
            or ar.get('observation_report_sha256')!=oh or ar.get('private_truth_read') is not False
            or ar.get('actual_raw_decoder_parity_verified') is not True
            or ar.get('native_camera_transform_parity_verified') is not True
            or observations.get('status')!='pass' or observations.get('private_truth_read') is not False
            or [(p.get('object_index')) for p in ar.get('proposals',[])]!=[0,1,2]):
        raise ValueError('Require complete source-bound frozen public anchors/observations')
    records=observations.get('outputs',[])
    if [(p.get('object_index'),p.get('frame_index')) for p in records]!=[(o,f) for o in range(3) for f in range(8)]:raise ValueError('Public temporal records incomplete')
    meshes=[];images={}
    for obj,r in enumerate(ar['proposals']):
        path=base/'anchors'/f'object_{obj:02d}_anchor.npz'
        if r.get('file')!=path.name:raise ValueError('Anchor filename differs')
        native.require_hash(path,r['sha256'])
        with np.load(path,allow_pickle=False) as d:
            v,f=d['vertices_camera_m'].copy(),d['faces'].copy()
        _mesh_inputs(v,f,K,512,384,1e-4);meshes.append((v,f))
    for r in records:
        obj,frame=r['object_index'],r['frame_index'];p=base/'observations'/f'object_{obj:02d}_frame_{frame:03d}.npz'
        if r.get('file')!=p.name:raise ValueError('Public frame filename differs')
        native.require_hash(p,r['sha256'])
        with np.load(p,allow_pickle=False) as d:
            for key in d.files:
                x=d[key];identity={'sha256':hashlib.sha256(x.tobytes(order='C')).hexdigest(),'dtype':str(x.dtype),'shape':list(x.shape)}
                if r.get('arrays',{}).get(key)!=identity:raise ValueError('Frozen decoded observations differ')
            images[obj,frame]=native.observation_arrays(d,obj,frame)
    return meshes,images,{'anchor_report_sha256':ah,'observation_report_sha256':oh}


def anchor_features(torch,vertices,faces,rgb,mask):
    """Automatic RGB corners attached to actual predicted-mesh raster triangles."""
    import cv2
    from pytorch3d.renderer import MeshRasterizer,RasterizationSettings
    from pytorch3d.structures import Meshes
    with torch.inference_mode():
        mesh=Meshes(verts=[torch.as_tensor(vertices,device='cuda',dtype=torch.float32)],faces=[torch.as_tensor(faces,device='cuda',dtype=torch.int64)])
        fragment=MeshRasterizer(cameras=_opencv_camera(torch,K,512,384),raster_settings=RasterizationSettings(image_size=(384,512),blur_radius=0.,faces_per_pixel=1,
            perspective_correct=True,clip_barycentric_coords=False,cull_backfaces=False,cull_to_frustum=False,max_faces_per_bin=len(faces)))(mesh)
        face=fragment.pix_to_face[0,:,:,0].cpu().numpy();bary=fragment.bary_coords[0,:,:,0].cpu().numpy()
    supported=mask&(face>=0);corners=cv2.goodFeaturesToTrack(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),maxCorners=512,qualityLevel=.01,minDistance=5,mask=supported.astype(np.uint8)*255)
    if corners is None:return np.empty((0,2),float),np.empty((0,3),float)
    pixels=np.unique(np.floor(corners.reshape(-1,2)+.5).astype(np.int64),axis=0)
    good=supported[pixels[:,1],pixels[:,0]];pixels=pixels[good]
    points=raster_attachments(vertices,faces,pixels,face,bary)
    # Raster samples at (x+.5,y+.5); OpenCV LK corners at integer image indices.
    return pixels.astype(np.float64),points


def run(root,report,path):
    import cv2
    import torch
    import trimesh
    meshes,images,chain=load_inputs(root);report.update(chain,opencv_version=cv2.__version__,frames=[])
    rotations=np.tile(np.eye(3),(2,3,8,1,1));translations=np.zeros((2,3,8,3))
    for obj,(v,f) in enumerate(meshes):
        samples,_=trimesh.sample.sample_surface(trimesh.Trimesh(v,f,process=False),8192,seed=0)
        anchor_rgb,anchor_mask,anchor_map=images[obj,0]
        anchor_points=anchor_map.transpose(1,2,0)[anchor_mask];anchor_center=np.median(anchor_points,axis=0)
        pixels,points=anchor_features(torch,v,f,anchor_rgb,anchor_mask);trackingK=K.copy();trackingK[:2,2]-=.5
        previous_r=np.eye(3)
        for frame in range(8):
            rgb,mask,pointmap=images[obj,frame];observed=pointmap.transpose(1,2,0)[mask]
            if len(observed)<40:raise ValueError('No measured baseline depth support')
            if len(observed)>2048:observed=observed[np.random.default_rng(0).choice(len(observed),2048,replace=False)]
            if frame==0:
                R,t=np.eye(3),np.zeros(3);status='shared_anchor'
                detail={'status':'shared_anchor','reason':'no_temporal_pose_fit'}
            else:
                initial_t=np.median(observed,axis=0)-anchor_center
                fit=align_observed_points(samples,observed,previous_r,initial_t)
                R,t=fit.rotation,fit.translation;status='depth_icp'
                initial_mask,_=raster_camera_mesh(v@previous_r.T+initial_t,f,K,512,384)
                fitted_mask,_=raster_camera_mesh(v@R.T+t,f,K,512,384)
                if silhouette_iou(fitted_mask.cpu().numpy(),mask)<silhouette_iou(initial_mask.cpu().numpy(),mask):R,t=previous_r,initial_t
                proposal=track_rgb_pose(anchor_rgb,rgb,anchor_mask,mask,pixels,points,trackingK)
                detail={k:value for k,value in asdict(proposal).items() if k not in ('rotation','translation')}
                if proposal.status=='proposal':rotations[1,obj,frame]=proposal.rotation;translations[1,obj,frame]=proposal.translation
                else:rotations[1,obj,frame]=R;translations[1,obj,frame]=t
            rotations[0,obj,frame]=R;translations[0,obj,frame]=t;previous_r=R
            report['frames'].append({'object_index':obj,'frame_index':frame,'baseline_method':status,'rgb_pnp':detail,'anchor_feature_count':len(pixels)})
            native.persist(path,report)
    artifact=path.parent/'predictions.npz'
    with artifact.open('xb') as stream:np.savez_compressed(stream,rotations=rotations,translations=translations,frame_index=np.arange(8),object_index=np.arange(3))
    for key,relative in [('anchor_report_sha256','anchors/report.json'),('observation_report_sha256','observations/report.json')]:native.require_hash(root/'validation/object_motion_v1'/relative,chain[key])
    report.update(status='pass',predictions_sha256=sha256(artifact),original_frame_coverage_verified=True,
        fixed_shape_preserved=True,fixed_scale_preserved=True,private_truth_read=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require remote isolated GPU raster')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/object_motion_v1/tracking';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require reserved fresh motion proposals')
    report={'stage':STAGE,'status':'fail','code_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],
            'script_sha256':sha256(Path(__file__)),'private_truth_read':False,'challenge_inputs_used':False,'adoption_performed':False,
            'budget_seconds':300,'oracle_pose_used':False,'temporal_interpolation_performed':False,'baseline':'same fixed mesh, observed depth ICP + silhouette gate',
            'candidate':'direct anchor RGB LK to mesh-PnP, measured baseline on explicit abstention','pose_accuracy_verified':False,
            'K_tracking_principal_point_offset':-.5,'spatial_reserved_tracks_are_not_independent_temporal_validation':True}
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen300s tracking deadline exceeded')));signal.alarm(300)
    try:native.persist(path,report);run(root,report,path)
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-started;native.persist(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds')}))

if __name__=='__main__':main()

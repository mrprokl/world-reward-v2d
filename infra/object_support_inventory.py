"""CPU inventory of original object-depth support; never poses or occlusion truth.

The unchanged tracker requires forty supported pixels. Missing observations are
reported, not filled, pruned, or declared occlusion. All media stay on Azure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import time

from body_smoke import TRACK1_EPISODE_COUNT, _validate_inputs

STAGE='world_reward_object_full_original_support_inventory'
MINIMUM_SUPPORT=40
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SOURCES=('object_support_inventory.py','run_object_support_inventory.sh',
         'object_pose_smoke.py','body_smoke.py','depth_smoke.py')


def identity(path, maximum=None):
    path=Path(path)
    if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):
        raise ValueError('Canonical regular input required')
    before=path.stat()
    if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or before.st_size<=0 or maximum and before.st_size>maximum:
        raise ValueError('Bounded nonempty single-link regular input required')
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    after=path.stat()
    if any(getattr(before,k)!=getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):
        raise ValueError('Input changed during hashing')
    return dict(bytes=before.st_size,sha256=digest.hexdigest())


def strict_json(path):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError('Duplicate report key')
            result[key]=value
        return result
    def invalid(_):raise ValueError('Nonfinite report constant')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def _require(record,expected):
    if type(record)is not dict or any(record.get(k)!=v or type(record.get(k))is not type(v)for k,v in expected.items()):
        raise ValueError('Original same-clip predictor provenance differs')


def _frames(record,count):
    rows=record.get('frames')
    if type(rows)is not list or len(rows)!=count or any(type(r)is not dict or type(r.get('frame_index'))is not int for r in rows):
        raise ValueError('Original predictor frame records malformed')
    if [r['frame_index']for r in rows]!=list(range(count)):raise ValueError('Exact original full frame coverage required')
    return rows


def support_row(index,mask,depth,validity,intrinsic,camera,shared_scale):
    """Execute the original +.5 camera ray/scale chain, without filtering changes."""
    import numpy as np
    height,width=mask.shape
    if mask.dtype!=np.bool_ or depth.shape!=(height,width)or depth.dtype!=np.float32 or validity.shape!=depth.shape or validity.dtype!=np.bool_:
        raise ValueError('Original bool masks and FP32 camera-Z grid required')
    if intrinsic.shape!=(3,3)or intrinsic.dtype.kind!='f'or not np.isfinite(intrinsic).all():
        raise ValueError('Finite original normalized inferred camera required')
    pixel_K=np.diag([width,height,1])@intrinsic
    if not np.allclose(pixel_K,camera,atol=1e-3,rtol=1e-6):raise ValueError('Original fixed object/depth cameras differ')
    yy,xx=np.mgrid[:height,:width]
    with np.errstate(over='ignore',invalid='ignore'):
        points=np.stack(((xx+.5-pixel_K[0,2])*depth/pixel_K[0,0],
                         (yy+.5-pixel_K[1,2])*depth/pixel_K[1,1],depth),axis=-1)
        points*=shared_scale
    points[~validity]=np.nan
    finite_positive=np.isfinite(points).all(-1)&(points[...,2]>0)
    support=int(np.count_nonzero(mask&finite_positive));area=int(mask.sum())
    reason=None if support>=MINIMUM_SUPPORT else('empty_segmentation'if area==0 else'nonempty_invalid_depth'if support==0 else'under_40_supported_pixels')
    return dict(frame_index=index,object_mask_area_pixels=area,depth_validity_pixels=int(validity.sum()),
                finite_positive_camera_points_pixels=int(finite_positive.sum()),supported_object_pixels=support,
                meets_original_minimum_support=support>=MINIMUM_SUPPORT,unknown_observation_reason=reason)


def gap_spans(rows):
    gaps=[]
    for row in rows:
        if row['meets_original_minimum_support']:continue
        index=row['frame_index']
        if not gaps or gaps[-1]['last_frame_index']!=index-1:
            gaps.append(dict(first_frame_index=index,last_frame_index=index,frames=0,reasons={}))
        gap=gaps[-1];gap['last_frame_index']=index;gap['frames']+=1
        reason=row['unknown_observation_reason'];gap['reasons'][reason]=gap['reasons'].get(reason,0)+1
    return gaps


def inventory(root,episode):
    """Return a complete inventory, or fail; does not create or modify outputs."""
    import numpy as np
    from PIL import Image
    inputs=_validate_inputs(root,episode_index=episode);count=inputs['total_frames']
    base=root/f'outputs/episode_{episode:06d}'
    report_paths={name:base/name/'report.json'for name in('automatic_masks','body_full','depth_full','scale_smoke','object_grounded')}
    report_ids={name:identity(path,40_000_000)for name,path in report_paths.items()}
    reports={name:strict_json(path)for name,path in report_paths.items()}
    common=dict(status='pass',episode_index=episode,input_track='track_1',input_sha256=inputs['video_sha256'],
                ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    for name,stage in dict(automatic_masks='automatic_masks',body_full='sam3d_body_full_video_initializer',
        depth_full='monocular_moge2_full_video',scale_smoke='predicted_human_anchored_moge2_pointmaps',
        object_grounded='sam3d_objects_grounded_fixed_frame').items():_require(reports[name],dict(common,stage=stage))
    _require(reports['automatic_masks'],dict(frames=count))
    for name in('body_full','depth_full'):_require(reports[name],dict(total_video_frames=count))
    body,depth=_frames(reports['body_full'],count),_frames(reports['depth_full'],count)
    if reports['body_full'].get('mask_report_sha256')!=report_ids['automatic_masks']['sha256']:
        raise ValueError('Original Body mask-report ancestry differs')
    for a,b in zip(body,depth):
        if not re.fullmatch('[0-9a-f]{64}',a.get('decoded_rgb_sha256',''))or a['decoded_rgb_sha256']!=b.get('decoded_rgb_sha256'):
            raise ValueError('Original Body/MoGe decoded RGB identities differ')
    obj=reports['object_grounded']
    if(obj.get('scale_source')!='already_human_anchored_MoGe2_no_second_scalar'
       or obj.get('pointmap_grounding',{}).get('alignment_report_sha256')!=report_ids['scale_smoke']['sha256']):
        raise ValueError('Original object grounding gauge ancestry differs')
    scale=reports['scale_smoke'].get('depth_alignment',{}).get('shared_scale')
    if type(scale)not in(int,float)or not math.isfinite(scale)or scale<=0:raise ValueError('Original shared positive scalar required')
    camera_path=base/'object_grounded/intrinsics.json';camera_id=identity(camera_path,4096)
    if camera_id['sha256']!=obj.get('intrinsics_sha256'):raise ValueError('Original fixed object camera changed')
    camera_record=strict_json(camera_path)
    width,height=camera_record.get('width'),camera_record.get('height')
    if type(width)is not int or type(height)is not int or min(width,height)<=0 or width*height>4_000_000:
        raise ValueError('Bounded original camera grid required')
    camera=np.array([[camera_record['fx'],0,camera_record['cx']],[0,camera_record['fy'],camera_record['cy']],[0,0,1]])
    if not np.isfinite(camera).all()or camera[0,0]<=0 or camera[1,1]<=0:raise ValueError('Finite positive original fixed camera required')
    masks=base/'automatic_masks/masks/1';depth_dir=base/'depth_full'
    expected=[f'{i:06d}.png'for i in range(count)]
    if sorted(p.name for p in masks.iterdir())!=expected:raise ValueError('Exact original full object-mask inventory required')
    if sorted(p.name for p in depth_dir.glob('*.npz'))!=[f'{i:06d}.npz'for i in range(count)]:
        raise ValueError('Exact original full depth inventory required')
    files=[]
    # Hash the complete selected PNG/NPZ cohort before ANY media decode. Mask
    # hashes are first-observed audit pins, not independent publisher/model pins.
    for i,row in enumerate(depth):
        mask_path=masks/f'{i:06d}.png';depth_path=depth_dir/f'{i:06d}.npz'
        mask_id=identity(mask_path,20_000_000);depth_id=identity(depth_path,100_000_000)
        if depth_id['sha256']!=row.get('output_sha256'):raise ValueError('Original per-frame depth SHA differs')
        files.append((mask_path,depth_path,mask_id,depth_id))
    rows=[]
    for i,(mask_path,depth_path,mask_id,depth_id)in enumerate(files):
        if identity(mask_path)!=mask_id or identity(depth_path)!=depth_id:raise ValueError('Pinned media changed before decode')
        with Image.open(mask_path)as image:
            if image.size!=(width,height)or image.mode not in('1','L','P','I','I;16'):
                raise ValueError('Original single-channel object mask grid required')
            mask=np.asarray(image)>0
        with np.load(depth_path,allow_pickle=False)as arrays:
            if set(arrays.files)!={'depth','mask','intrinsics','frame_index'}:
                raise ValueError('Original Z/K/validity storage ABI differs')
            frame=arrays['frame_index']
            if frame.shape!=()or frame.dtype.kind not in'iu'or int(frame)!=i:raise ValueError('Original stored frame index differs')
            row=support_row(i,mask,arrays['depth'],arrays['mask'],arrays['intrinsics'],camera,scale)
        if identity(mask_path)!=mask_id or identity(depth_path)!=depth_id:raise ValueError('Pinned media changed during decode')
        row.update(object_mask_identity=mask_id,depth_identity=depth_id);rows.append(row)
    if {name:identity(path,40_000_000)for name,path in report_paths.items()}!=report_ids or identity(camera_path)!=camera_id:
        raise ValueError('Original predictor reports/camera changed during inventory')
    for mask_path,depth_path,mask_id,depth_id in files:
        if identity(mask_path)!=mask_id or identity(depth_path)!=depth_id:raise ValueError('Full original cohort changed after inventory')
    return dict(original_frames=count,original_frame_indices=list(range(count)),minimum_supported_pixels=MINIMUM_SUPPORT,
        support_rule='original_mask_and_finite_positive_scaled_camera_XYZ_at_pixel_centres_plus_0.5',
        all_frames_meet_original_minimum_support=all(r['meets_original_minimum_support']for r in rows),
        unsupported_frames=sum(not r['meets_original_minimum_support']for r in rows),gap_spans=gap_spans(rows),frames=rows,
        input_sha256=inputs['video_sha256'],predictor_report_identities=report_ids,original_fixed_camera_identity=camera_id,
        original_fixed_camera=camera_record,original_shared_depth_scale=scale,
        all_selected_media_hashed_before_any_decode=True,mask_hash_assurance='first_observed_before_decode_not_independent_producer_pins',
        depth_hash_assurance='verified_against_original_full_depth_report_before_decode')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--episode',type=int,choices=range(TRACK1_EPISODE_COUNT),required=True)
    args=parser.parse_args(argv)
    if platform.system()!='Linux'or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require Azure CPU container with network none')
    root=Path(os.environ['WR_ROOT']);revision=os.environ['WR_CODE_REVISION']
    if root!=Path('/srv/scenesmith/world-reward')or not re.fullmatch('[0-9a-f]{40}',revision)or os.environ.get('WR_IMAGE_ID')!=IMAGE:
        raise ValueError('Original fixed CPU runtime and immutable producer required')
    out=root/f'outputs/episode_{args.episode:06d}/object_support_inventory_v1'
    if not out.is_dir()or out.is_symlink()or any(out.iterdir())or os.environ.get('WR_OUTPUT_RESERVED')!='1':
        raise ValueError('Fresh exclusive support inventory target required')
    started=time.perf_counter();source=Path(__file__).resolve().parent
    source_ids={name:identity(source/name)for name in SOURCES}
    tracker=(source/'object_pose_smoke.py').read_text()
    if 'if len(observed) < 40:'not in tracker:raise ValueError('Original tracker support threshold source differs')
    report=dict(stage=STAGE,status='fail',episode_index=args.episode,input_track='track_1',producer_revision=revision,
        image_id=IMAGE,source_helpers=source_ids,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
        inference_performed=False,poses_produced=False,occlusion_estimated=False,challenge_performance_verified=False,
        submission_eligible=False,budget_seconds=900)
    def expired(*_):raise TimeoutError('Support inventory budget exceeded')
    old_alarm=signal.signal(signal.SIGALRM,expired);old_term=signal.signal(signal.SIGTERM,expired)
    signal.alarm(900)
    try:
        report.update(inventory(root,args.episode))
        if time.perf_counter()-started>900:raise TimeoutError('Support inventory budget exceeded')
        if {name:identity(source/name)for name in SOURCES}!=source_ids:raise ValueError('Support/source helpers changed')
        report['status']='pass'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:250]);raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
        report['elapsed_seconds']=time.perf_counter()-started
        target=out/'report.json'
        with target.open('x')as stream:
            os.fchmod(stream.fileno(),0o400);json.dump(report,stream,allow_nan=False);stream.write('\n')
        print(json.dumps(dict(stage=STAGE,status=report['status'],episode_index=args.episode,
            frames=report.get('original_frames'),unsupported_frames=report.get('unsupported_frames'),
            gap_count=len(report.get('gap_spans',[])),report_identity=identity(target),poses_produced=False)))


if __name__=='__main__':main()

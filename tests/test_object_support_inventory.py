"""Tiny original-grid support fixtures, without actual videos/models/network."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'infra'))
import object_support_inventory as audit


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    root=tmp_path/'root';base=root/'outputs/episode_000004';count=5;height=6;width=8
    masks=base/'automatic_masks/masks/1';masks.mkdir(parents=True)
    depth_dir=base/'depth_full';depth_dir.mkdir()
    camera=dict(width=width,height=height,fx=10.,fy=10.,cx=4.,cy=3.)
    K=np.array([[10./width,0,4./width],[0,10./height,3./height],[0,0,1]],np.float32)
    depth_rows=[];body_rows=[]
    for i in range(count):
        mask=np.ones((height,width),np.uint8)
        if i==1:mask[:]=0
        if i==3:mask.ravel()[20:]=0
        Image.fromarray(mask*255).save(masks/f'{i:06d}.png')
        valid=np.ones((height,width),bool)
        if i==2:valid[:]=False
        path=depth_dir/f'{i:06d}.npz'
        np.savez_compressed(path,depth=np.ones((height,width),np.float32)*2,mask=valid,
                            intrinsics=K,frame_index=np.array(i))
        rgb=hashlib.sha256(('rgb'+str(i)).encode()).hexdigest()
        depth_rows.append(dict(frame_index=i,decoded_rgb_sha256=rgb,output_sha256=audit.identity(path)['sha256']))
        body_rows.append(dict(frame_index=i,decoded_rgb_sha256=rgb))
    common=dict(status='pass',episode_index=4,input_track='track_1',input_sha256='c'*64,
                ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    mask_report=dict(common,stage='automatic_masks',frames=count)
    write_json(base/'automatic_masks/report.json',mask_report)
    body=dict(common,stage='sam3d_body_full_video_initializer',total_video_frames=count,frames=body_rows,
              mask_report_sha256=audit.identity(base/'automatic_masks/report.json')['sha256'])
    depth=dict(common,stage='monocular_moge2_full_video',total_video_frames=count,frames=depth_rows)
    alignment=dict(common,stage='predicted_human_anchored_moge2_pointmaps',depth_alignment=dict(shared_scale=1.25))
    write_json(base/'body_full/report.json',body);write_json(base/'depth_full/report.json',depth)
    write_json(base/'scale_smoke/report.json',alignment);write_json(base/'object_grounded/intrinsics.json',camera)
    obj=dict(common,stage='sam3d_objects_grounded_fixed_frame',scale_source='already_human_anchored_MoGe2_no_second_scalar',
             pointmap_grounding=dict(alignment_report_sha256=audit.identity(base/'scale_smoke/report.json')['sha256']),
             intrinsics_sha256=audit.identity(base/'object_grounded/intrinsics.json')['sha256'])
    write_json(base/'object_grounded/report.json',obj)
    inputs=dict(total_frames=count,video_sha256='c'*64)
    monkeypatch.setattr(audit,'_validate_inputs',lambda r,episode_index:inputs)
    return dict(root=root,base=base,masks=masks,depth=depth_dir,K=K,camera=camera,inputs=inputs)


def test_complete_support_counts_and_unknown_gap_not_pose_or_occlusion(cohort):
    result=audit.inventory(cohort['root'],4)
    assert result['original_frame_indices']==list(range(5))and result['minimum_supported_pixels']==40
    assert[result['frames'][i]['supported_object_pixels']for i in range(5)]==[48,0,0,20,48]
    assert result['all_frames_meet_original_minimum_support']is False and result['unsupported_frames']==3
    assert result['gap_spans']==[dict(first_frame_index=1,last_frame_index=3,frames=3,
        reasons=dict(empty_segmentation=1,nonempty_invalid_depth=1,under_40_supported_pixels=1))]
    assert result['all_selected_media_hashed_before_any_decode']is True
    assert 'first_observed'in result['mask_hash_assurance']
    assert not(cohort['base']/'object_support_inventory_v1').exists()
    assert not any(k in result for k in('rotation','translation','poses','occlusion'))


def test_exact_original_xyz_chain_and40_boundary():
    shape=(8,8);mask=np.ones(shape,bool);valid=np.ones(shape,bool)
    depth=np.ones(shape,np.float32);depth.ravel()[40:]=np.nan
    K=np.array([[1.,0,.5],[0,1.,.5],[0,0,1]],np.float32);camera=np.diag([8,8,1])@K
    row=audit.support_row(0,mask,depth,valid,K,camera,2.)
    assert row['supported_object_pixels']==40 and row['meets_original_minimum_support']
    depth.ravel()[39]=0
    row=audit.support_row(0,mask,depth,valid,K,camera,2.)
    assert row['supported_object_pixels']==39 and row['unknown_observation_reason']=='under_40_supported_pixels'
    original=np.arange(64,dtype=np.float32).reshape(shape);original[0,0]=-1;original[0,1]=np.inf
    valid[1,1]=False;yy,xx=np.mgrid[:8,:8]
    with np.errstate(invalid='ignore'):
        points=np.stack(((xx+.5-camera[0,2])*original/camera[0,0],
                         (yy+.5-camera[1,2])*original/camera[1,1],original),axis=-1)
    points*=1.25;points[~valid]=np.nan
    expected=int((mask&np.isfinite(points).all(-1)&(points[...,2]>0)).sum())
    assert audit.support_row(0,mask,original,valid,K,camera,1.25)['supported_object_pixels']==expected


@pytest.mark.parametrize('fault',['depth64','validfloat','camera','Knan'])
def test_malformed_arrays_and_changed_camera_fail(fault):
    mask=np.ones((8,8),bool);depth=np.ones((8,8),np.float32);valid=mask.copy()
    K=np.array([[1.,0,.5],[0,1.,.5],[0,0,1]],np.float32);camera=np.diag([8,8,1])@K
    if fault=='depth64':depth=depth.astype(np.float64)
    elif fault=='validfloat':valid=valid.astype(np.float32)
    elif fault=='camera':camera[0,0]+=1
    else:K[0,0]=np.nan
    with pytest.raises(ValueError):audit.support_row(0,mask,depth,valid,K,camera,1.)


@pytest.mark.parametrize('fault',['wrongclip','oracle','rgb','frameorder','count','bodymask','scale','grounding','camera'])
def test_provenance_verified_before_any_media_decode(cohort,monkeypatch,fault):
    base=cohort['base'];name='depth_full';field=None
    if fault=='wrongclip':field=('episode_index',3)
    elif fault=='oracle':field=('oracle_modes',['oracle'])
    elif fault=='rgb':field=('frames',None)
    elif fault=='frameorder':field=('frames',None)
    elif fault=='count':field=('total_video_frames',4)
    elif fault=='bodymask':name='body_full';field=('mask_report_sha256','d'*64)
    elif fault=='scale':name='scale_smoke';field=('depth_alignment',dict(shared_scale=0.))
    elif fault=='grounding':name='object_grounded';field=('pointmap_grounding',{})
    else:name='object_grounded';field=('intrinsics_sha256','d'*64)
    path=base/name/'report.json';record=json.loads(path.read_text())
    if fault=='rgb':record['frames'][0]['decoded_rgb_sha256']='d'*64
    elif fault=='frameorder':record['frames'].reverse()
    else:record[field[0]]=field[1]
    write_json(path,record)
    monkeypatch.setattr(Image,'open',lambda *_:pytest.fail('Media decoded before provenance gate'))
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('Media decoded before provenance gate'))
    with pytest.raises(ValueError):audit.inventory(cohort['root'],4)


def test_all_depth_hashes_checked_before_first_png_decode(cohort,monkeypatch):
    last=cohort['depth']/'000004.npz';last.write_bytes(last.read_bytes()+b'tamper')
    monkeypatch.setattr(Image,'open',lambda *_:pytest.fail('PNG decoded before full cohort hash gate'))
    with pytest.raises(ValueError,match='SHA'):audit.inventory(cohort['root'],4)


@pytest.mark.parametrize('kind',['missingmask','extra','symlink','hardlink'])
def test_exact_regular_mask_inventory(cohort,kind,tmp_path):
    path=cohort['masks']/'000004.png'
    if kind=='missingmask':path.unlink()
    elif kind=='extra':(cohort['masks']/'notes.txt').write_text('foreign')
    elif kind=='symlink':path.unlink();path.symlink_to('000003.png')
    else:os.link(path,tmp_path/'alias')
    with pytest.raises(ValueError):audit.inventory(cohort['root'],4)


def test_during_decode_media_tamper_detected(cohort,monkeypatch):
    original=Image.open
    def changed(path,*a,**k):
        image=original(path,*a,**k);other=cohort['masks']/'000004.png'
        other.write_bytes(other.read_bytes()+b'tamper');return image
    monkeypatch.setattr(Image,'open',changed)
    with pytest.raises(ValueError,match='changed'):audit.inventory(cohort['root'],4)


def test_gaps_include_leading_trailing_and_all_unknown():
    def row(i,ok):return dict(frame_index=i,meets_original_minimum_support=ok,unknown_observation_reason=None if ok else'empty_segmentation')
    rows=[row(i,i in(2,3))for i in range(6)]
    assert[(r['first_frame_index'],r['last_frame_index'])for r in audit.gap_spans(rows)]==[(0,1),(4,5)]
    assert audit.gap_spans([row(i,False)for i in range(4)])[0]['frames']==4
    assert audit.gap_spans([row(i,True)for i in range(4)])==[]


def test_wrapper_syntax_scoped_cpu_mounts_and_complete_closure(monkeypatch):
    wrapper=ROOT/'infra/run_object_support_inventory.sh'
    subprocess.run(['rtk','proxy','bash','-n',str(wrapper)],check=True)
    source=wrapper.read_text()
    assert '--network none --memory 4g --cpus 2'in source and '903s docker run'in source
    assert audit.IMAGE in source
    assert not any(x in source for x in('--gpus','nvidia-smi','/validation','/vendor','/weights','/body_full/predictions','object.glb','transform.json'))
    assert 'src=$BASE/body_full/report.json'in source and 'src=$BASE/object_grounded/intrinsics.json'in source
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and'__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_object_support_inventory.sh'))
    assert{'infra/object_support_inventory.py','infra/object_pose_smoke.py','infra/body_smoke.py','infra/depth_smoke.py'}<=selected


@pytest.mark.parametrize('args',[[],['--episode','04'],['--episode','30'],['--episode','4','--min-support','1'],['--help']])
def test_wrapper_rejects_invalid_arguments_before_runtime(tmp_path,args):
    env=dict(PATH=os.environ['PATH'],HOME=str(tmp_path))
    result=subprocess.run(['rtk','proxy','bash',str(ROOT/'infra/run_object_support_inventory.sh'),*args],
                          env=env,capture_output=True,text=True,timeout=5)
    assert result.returncode==2

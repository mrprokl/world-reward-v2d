import copy
import numpy as np
import pytest
from world_reward.point_bop_evaluation import evaluate_bop_scene, FRAMES, HEIGHT, WIDTH


def fixture():
    K=[200.,0.,320.,0.,200.,240.,0.,0.,1.]
    cameras={str(i):dict(cam_K=K.copy(),depth_scale=2.) for i in range(FRAMES)}
    gt={}; info={}; pred=np.broadcast_to(np.eye(4), (FRAMES,4,4)).copy()
    for i in range(FRAMES):
        rows=[dict(obj_id=4,cam_R_m2c=np.eye(3).reshape(-1).tolist(),cam_t_m2c=[i*10.,0.,2000.]),
              dict(obj_id=7,cam_R_m2c=np.eye(3).reshape(-1).tolist(),cam_t_m2c=[0.,0.,2000.])]
        if i%2:rows.reverse()
        gt[str(i)]=rows;info[str(i)]=[{} for _ in rows]
    masks=np.zeros((2,HEIGHT,WIDTH),np.uint8);masks[0,:3,:4]=255;masks[1,4:7,:4]=255
    auto=masks[0]==255
    return dict(baseline_poses=pred.copy(),candidate_poses=pred.copy(),automatic_initial_mask=auto,
        scene_camera=cameras,scene_gt=gt,scene_gt_info=info,initial_depth_png=np.full((HEIGHT,WIDTH),1000,np.uint16),
        initial_visible_masks=masks,sequence_id='synthetic_bop')


def test_original_ids_permuted_without_error_matching_allframes():
    data=fixture();out=evaluate_bop_scene(**data)
    assert out['matched_gt_mask_index']==0 and out['frame_indices']==list(range(96))
    assert out['baseline']['displacement_m']==pytest.approx(np.arange(96)*.01)
    assert not any(k in out for k in ('GT','poses','camera_K','object_ids','points'))


def test_mm_depth_scale_and_translation_conversion_no_alignment():
    data=fixture();R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for i in range(1,96):
        row=next(r for r in data['scene_gt'][str(i)] if r['obj_id']==4)
        row['cam_R_m2c']=R.reshape(-1).tolist();row['cam_t_m2c']=[0.,0.,2000.]
    out=evaluate_bop_scene(**data)
    yy,xx=np.indices((3,4));radius=np.hypot((xx-320)*2/200,(yy-240)*2/200)
    assert out['baseline']['displacement_m'][1:]==pytest.approx(np.full(95,np.sqrt(2)*radius.mean()))
    assert out['baseline']['rotation_degrees'][1:]==pytest.approx(np.full(95,90.))


@pytest.mark.parametrize('fault',['missingframe','extra','duplicatedid','missingid','newid','floatid','rotation',
 'camera','scale','negative_scale','depthfloat','depthshape','maskgray','maskshape','missingpose','inf','info'])
def test_private_contract_stops_without_repair(fault):
    data=fixture()
    if fault=='missingframe':del data['scene_gt']['95']
    elif fault=='extra':data['scene_camera']['96']=data['scene_camera']['0']
    elif fault=='duplicatedid':data['scene_gt']['1'][0]['obj_id']=4
    elif fault=='missingid':data['scene_gt']['1'].pop()
    elif fault=='newid':data['scene_gt']['1'][0]['obj_id']=8
    elif fault=='floatid':data['scene_gt']['0'][0]['obj_id']=4.
    elif fault=='rotation':data['scene_gt']['1'][0]['cam_R_m2c'][0]=2.
    elif fault=='camera':data['scene_camera']['95']['cam_K'][0]=201.
    elif fault=='scale':data['scene_camera']['95']['depth_scale']=1.
    elif fault=='negative_scale':data['scene_camera']['0']['depth_scale']=-2.
    elif fault=='depthfloat':data['initial_depth_png']=data['initial_depth_png'].astype(np.float64)
    elif fault=='depthshape':data['initial_depth_png']=data['initial_depth_png'][:2]
    elif fault=='maskgray':data['initial_visible_masks'][0,0,0]=1
    elif fault=='maskshape':data['initial_visible_masks']=data['initial_visible_masks'][:1]
    elif fault=='missingpose':del data['scene_gt']['1'][0]['cam_t_m2c']
    elif fault=='inf':data['scene_camera']['1']['depth_scale']=float('inf')
    elif fault=='info':data['scene_gt_info']['1'].pop()
    with pytest.raises((ValueError,KeyError)):evaluate_bop_scene(**data)


def test_annotation_inputs_are_not_mutated():
    data=fixture();before=copy.deepcopy(data);evaluate_bop_scene(**data)
    for key in ('scene_camera','scene_gt','scene_gt_info'):assert data[key]==before[key]
    for key in ('baseline_poses','candidate_poses','initial_depth_png','initial_visible_masks'):
        assert np.array_equal(data[key],before[key])

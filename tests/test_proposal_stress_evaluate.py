"""Manufactured two-pixel all-bank evaluator integration; no real RGB or GT."""
import numpy as np
import pytest
import proposal_stress_evaluate as d


def fixture():
    images=[];rows=[];banks={};truths={};inventories={}
    for i in range(32):
        iid=f'{i:032x}';images.append(dict(image_id=iid));rows.append(dict(image_id=iid,native_masks=2,native_generate_seconds=.01))
        masks=np.eye(2,dtype=bool).reshape(2,1,2)
        banks[iid]=dict(packed_masks=np.packbits(masks.reshape(2,2),axis=1,bitorder='little'),
            image_size=np.array([1,2],np.int64),mask_area=np.ones(2,np.int64),native_mask_indices=np.arange(2,dtype=np.int64))
        truths[iid]=dict(visible_entity_masks=masks,visible_entity_ids=np.array([[0,1]],np.int32),
            visible_face_indices=np.array([[0,1]],np.int64),face_entity_ids=np.array([0,1],np.int32))
        inventories[iid]=dict(image_id=iid,split='DEV'if i<16 else'reserved',entities=[dict(kind='human'),dict(kind='object')])
    return {},dict(images=images),dict(banks=rows),dict(images=images),banks,truths,inventories


def test_all32_banks_fixed_splits_and_native_order():
    args=fixture();report=d.evaluate(*args)
    assert report['all']['recalls']['0.5']['object']['macro_image_recall']==1
    assert report['all']['native_masks']==64 and report['all']['images']==32
    assert all(x['images']==16 for x in report['splits'].values())
    assert report['quality_verified']is False and report['adoption']is False


@pytest.mark.parametrize('fault',['id','split','missing','order','raster'])
def test_inconsistent_saved_banks_or_raster_no_automatic_fix(fault):
    args=fixture();iid=args[1]['images'][0]['image_id']
    if fault=='id':args[2]['banks'][0]['image_id']='other'
    elif fault=='split':args[6][iid]['split']='TEST'
    elif fault=='missing':args[2]['banks'].pop()
    elif fault=='order':args[4][iid]['native_mask_indices']=np.array([1,0],np.int64)
    elif fault=='raster':args[5][iid]['visible_entity_ids'][0,0]=1
    with pytest.raises(ValueError):d.evaluate(*args)

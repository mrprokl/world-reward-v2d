"""Tiny external GDI controls: no media/model requests or accuracy labels."""
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import pytest

import openimages_joint_pair_gdi as gdi


def protocol(slots=4, acquired=3):
    return dict(acquisition_revision='e2b6019f502e18694a221aa6b1818804ef3d6d83',
        cohort=dict(bytes=31043,sha256='441bdc57e101cb4e8291ca6bbebc554ca397619e2a892469d9e410a16359a456'),
        slots=slots,acquired=acquired,missing=slots-acquired)


def acquisition(slots=4, acquired=3):
    cfg=protocol(slots,acquired);source={'proof':'original-source'};rows=[];selected=[]
    for i in range(slots):
        iid=f'{i:016x}';metadata=dict(ImageID=iid,Author='private-author',Rotation='0.0')
        selected.append(dict(slot=i,split='DEV' if i<2 else 'TEST',publisher_metadata=metadata))
        row=dict(slot=i,split=selected[-1]['split'],image_id=iid,publisher_metadata=metadata,status='unavailable')
        if i<acquired:
            row.update(status='acquired',creator_grant_verified=True,publisher_md5_matched=True,
                rotation='0.0',image_transformation='none_original_file',image_pin=dict(bytes=10,sha256='a'*64),
                jpeg_header=dict(width=24,height=16,channels=3,header_only=True))
        rows.append(row)
    cohort=dict(schema='world_reward.openimages_joint_pair_cohort.v1',producer_revision=cfg['acquisition_revision'],source_binding=source,records=selected)
    manifest=dict(schema='world_reward.openimages_joint_pair_acquire.v1',status='pass',mode='acquire',
        stage='frozen_external_rgb_acquisition',producer_revision=cfg['acquisition_revision'],cohort_identity=cfg['cohort'],
        source_binding=source,no_replacements=True,retry_count=0,replacement_count=0,reference_geometry_read=False,
        challenge_inputs_used=False,quality_verified=False,adoption=False,gpu_used=False,models_loaded=False,
        source_and_inputs_rehashed_after=True,artifacts_rehashed_after=True,records=rows)
    return cohort,manifest,cfg


def test_projected_inputs_have_no_reference_split_rights_or_counts():
    cohort,manifest,cfg=acquisition();original=deepcopy((cohort,manifest))
    projected,missing=gdi.project_acquisition(cohort,manifest,cfg)
    assert (cohort,manifest)==original and len(projected['images'])==3 and missing==['0000000000000003']
    assert set(projected)=={'schema','images'}
    assert all(set(r)=={'image_id','file','width','height','bytes','sha256'} for r in projected['images'])
    raw=json.dumps(projected)
    for forbidden in ('DEV','TEST','private-author','publisher','rights','positive_pairs','reference'):
        assert forbidden not in raw


@pytest.mark.parametrize('failure',['source','after','slot','split','metadata','missing_rgb','rotation','count','header'])
def test_acquisition_fail_closed_before_projection(failure):
    cohort,manifest,cfg=acquisition()
    if failure=='source':cohort['source_binding']={}
    elif failure=='after':manifest['source_and_inputs_rehashed_after']=False
    elif failure=='slot':manifest['records'][1]['slot']=0
    elif failure=='split':manifest['records'][0]['split']='TEST'
    elif failure=='metadata':manifest['records'][0]['publisher_metadata']={'ImageID':'b'*16}
    elif failure=='missing_rgb':manifest['records'][-1]['image_pin']={'bytes':1,'sha256':'a'*64}
    elif failure=='rotation':manifest['records'][0]['rotation']='-1.0'
    elif failure=='count':cfg['acquired']=4
    else:manifest['records'][0]['jpeg_header'].update(width=65535,height=65535)
    with pytest.raises(ValueError):gdi.project_acquisition(cohort,manifest,cfg)


def test_native_raw_rows_dtypes_preserved_but_existing_nms_retention_exact():
    boxes=np.array([[1,1,5,5],[1,1,5,5],[10,1,15,7],[-1,1,5,5],[1,1,3,3]],np.float32)
    scores=np.array([.5,.7,.8,.99,-.5],np.float32);before=boxes.copy(),scores.copy()
    result=gdi.retained_bank(boxes,scores,['person']*5,'a'*16,'person',24,16)
    np.testing.assert_array_equal(result['raw_boxes'],before[0]);np.testing.assert_array_equal(result['raw_scores'],before[1])
    np.testing.assert_array_equal(boxes,before[0]);np.testing.assert_array_equal(scores,before[1])
    assert result['raw_boxes'].dtype==np.float32 and result['raw_scores'].dtype==np.float32
    assert result['retained_raw_slots'].tolist()==[2,1] and len(result['retained_ids'])==2
    assert result['retained_ids'].tolist()==[f'image:{"a"*16}/person/retained:000000',f'image:{"a"*16}/person/retained:000001']


def test_duplicate_competing_raw_slots_not_discarded_by_nms_record():
    boxes=np.array([[1,1,5,5]]*3,np.float64);scores=np.array([.8,.8,.4],np.float64)
    result=gdi.retained_bank(boxes,scores,['object']*3,'b'*16,'object',16,16)
    assert len(result['raw_boxes'])==3 and len(result['retained_boxes'])==1 and result['retained_raw_slots'].tolist()==[0]


def test_empty_bank_real_shapes_no_full_image_fallback():
    result=gdi.retained_bank(np.empty((0,4),np.float32),np.empty(0,np.float32),[],'b'*16,'person',16,16)
    assert result['retained_boxes'].shape==(0,4) and result['retained_ids'].shape==(0,)
    assert result['raw_boxes'].shape==(0,4) and result['raw_labels'].dtype.kind=='U'


@pytest.mark.parametrize('failure',['nan_box','nan_score','score_shape','label_count','dtype'])
def test_native_invalid_arrays_not_repaired(failure):
    boxes=np.array([[1,1,4,4]],np.float32);scores=np.array([.8],np.float32);labels=['person']
    if failure=='nan_box':boxes[0,0]=np.nan
    elif failure=='nan_score':scores[0]=np.nan
    elif failure=='score_shape':scores=scores[None]
    elif failure=='label_count':labels=[]
    else:boxes=boxes.astype(np.int32)
    with pytest.raises(ValueError):gdi.retained_bank(boxes,scores,labels,'a'*16,'person',16,16)


def images(count=3):
    return [dict(image_id=f'{i:016x}',file=f'never-open-{i}.jpg',width=24,height=16,bytes=10,sha256='a'*64) for i in range(count)]


def test_real_callbacks_one_bank_per_query_every_image_and_native_arrays(tmp_path):
    records=images();calls=[];report=dict(native_forward_calls=0,images=[])
    def decode(row):return np.zeros((row['height'],row['width'],3),np.uint8)
    def detector(rgb,query):
        calls.append(query)
        boxes=np.array([[1,1,5,8],[12,1,18,10]],np.float32);scores=np.array([.6,.9],np.float32)
        return boxes,scores,[query]*2,np.zeros((1,900,4),np.float32),np.zeros((1,900,5),np.float32)
    gdi.observe(records,tmp_path,detector,decode,report,lambda:None)
    assert calls==['person.','object.']*3 and report['native_forward_calls']==6 and len(report['images'])==3
    for row in report['images']:
        assert [q['retained_rows'] for q in row['queries']]==[2,2]
        assert row['input_file']==records[int(row['image_id'],16)]['file'] and len(row['person_ids'])==2
        with np.load(tmp_path/row['prediction_file'],allow_pickle=False) as data:
            assert data['person_raw_boxes'].dtype==np.float32 and data['person_model_logits'].shape==(1,900,5)
            assert data['person_retained_raw_slots'].tolist()==[1,0]
            assert data['object_retained_ids'].dtype.kind=='U'
            np.testing.assert_array_equal(data['person_boxes_original_xyxy'],data['person_retained_boxes'])
            np.testing.assert_array_equal(data['person_detector_scores'],data['person_retained_scores'])
        assert (tmp_path/row['prediction_file']).stat().st_mode&0o777==0o400


def test_empty_query_still_all_forward_calls_and_genuine_npz_empties(tmp_path):
    report=dict(native_forward_calls=0,images=[])
    def detector(*_):return np.empty((0,4),np.float32),np.empty(0,np.float32),[],np.empty((1,0,4),np.float32),np.empty((1,0,3),np.float32)
    gdi.observe(images(1),tmp_path,detector,lambda _:np.zeros((16,24,3),np.uint8),report,lambda:None)
    assert report['native_forward_calls']==2 and all(q['retained_rows']==0 for q in report['images'][0]['queries'])
    with np.load(tmp_path/'0000000000000000.npz',allow_pickle=False) as data:assert data['person_retained_boxes'].shape==(0,4)


def test_decode_grid_or_rgb_mutation_fail_not_resized(tmp_path):
    with pytest.raises(ValueError,match='RGB grid'):
        gdi.observe(images(1),tmp_path,lambda *_:None,lambda _:np.zeros((8,12,3),np.uint8),dict(native_forward_calls=0,images=[]),lambda:None)
    def detector(rgb,_):
        rgb[0,0,0]=1
        return np.empty((0,4),np.float32),np.empty(0,np.float32),[],np.empty((1,0,4),np.float32),np.empty((1,0,3),np.float32)
    with pytest.raises(ValueError,match='mutated'):
        gdi.observe(images(1),tmp_path,detector,lambda _:np.zeros((16,24,3),np.uint8),dict(native_forward_calls=0,images=[]),lambda:None)


def test_no_overwrite_or_repeated_partial_outputs(tmp_path):
    report=dict(native_forward_calls=0,images=[])
    detector=lambda *_:(np.empty((0,4),np.float32),np.empty(0,np.float32),[],np.empty((1,0,4),np.float32),np.empty((1,0,3),np.float32))
    decode=lambda _:np.zeros((16,24,3),np.uint8)
    gdi.observe(images(1),tmp_path,detector,decode,report,lambda:None)
    with pytest.raises(FileExistsError):gdi.observe(images(1),tmp_path,detector,decode,report,lambda:None)


def test_prospective_config_and_scope_exact():
    code=Path(__file__).resolve().parents[1];cfg=gdi.configuration(code)
    assert cfg['slots']==64 and cfg['acquired']==50 and cfg['missing']==14 and cfg['budget_seconds']==600
    assert cfg['person_query']=='person.' and cfg['object_query']=='object.' and cfg['adoption'] is False


def test_wrapper_tight_projection_native_only_write_mount_and_no_reference():
    source=(Path(__file__).resolve().parents[1]/'infra/run_openimages_joint_pair_gdi.sh').read_text()
    assert '--network none' in source and '--user 0:0' in source and '--cap-drop ALL' in source
    assert '--mount "type=bind,src=$OUT/native,dst=$OUT/native"' in source and 'src=$OUT,dst=$OUT' not in source
    assert 'src=$OUT/inputs.json' not in source  # included in readonly path loop, not generic RW.
    assert 'flock --nonblock 9' in source and 'gpu_idle' in source and '615s' in source
    assert '--prepare' in source and '--finalize "$RESULT"' in source and '--native' in source
    assert 'cohort.json' not in source and 'manifest.json"' not in source.replace('world-reward-frontend-assets-manifest.json"','')
    assert '/weights/sam2' not in source and 'src=$ACQUISITION' not in source
    assert 'docker rm --force "$NAME"' in source and 'docker rm -f' not in source

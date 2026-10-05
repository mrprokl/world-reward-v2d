"""Procedural native crop/source tests; no external/model/media acquisition."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import openimages_joint_pair_dwpose as worker


def make_bank(tmp_path,count=2):
    rgb=np.arange(16*24*3,dtype=np.uint8).reshape(16,24,3)
    boxes=np.array([[1+i*8,1,6+i*8,12]for i in range(count)],np.float32).reshape(-1,4)
    scores=np.array([.9-i*.1 for i in range(count)],np.float32)
    ids=[f'image:{"a"*16}/person/retained:{i:06d}'for i in range(count)]
    path=tmp_path/'gdi.npz'
    np.savez(path,person_boxes_original_xyxy=boxes.astype(np.float64),person_detector_scores=scores.astype(np.float64),
        person_retained_ids=np.asarray(ids,dtype=str),person_raw_boxes=boxes,person_raw_scores=scores)
    path.chmod(0o400)
    image=dict(image_id='a'*16,file='unused.jpg',width=24,height=16,bytes=10,sha256='b'*64,
        decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),person_ids=ids,gdi_file=str(path),gdi_identity=worker.rt.identity(path))
    return image,rgb


def test_person_retained_alignment_from_original_nms(tmp_path):
    image,_=make_bank(tmp_path)
    ids,boxes,scores=worker.person_bank(Path(image['gdi_file']),image)
    assert ids==tuple(image['person_ids'])and boxes.dtype==scores.dtype==np.float64
    assert boxes.tolist()==[[1,1,6,12],[9,1,14,12]]


@pytest.mark.parametrize('failure',['ids','scores','boxes','pin','dtypes'])
def test_bank_mismatch_not_repaired(tmp_path,failure):
    image,_=make_bank(tmp_path);path=Path(image['gdi_file'])
    if failure=='ids':image['person_ids'].reverse()
    elif failure=='pin':image['gdi_identity']['sha256']='f'*64
    else:
        with np.load(path,allow_pickle=False)as bank:arrays={k:bank[k].copy()for k in bank.files}
        if failure=='scores':arrays['person_detector_scores'][0]=.1
        elif failure=='boxes':arrays['person_boxes_original_xyxy'][0,0]=0
        else:arrays['person_boxes_original_xyxy']=arrays['person_boxes_original_xyxy'].astype(np.float32)
        path.chmod(0o600);np.savez(path,**arrays);path.chmod(0o400);image['gdi_identity']=worker.rt.identity(path)
    with pytest.raises(ValueError):worker.person_bank(path,image)


def test_real_adapter_callable_all_people_raw133_no_neck_or_quality(tmp_path):
    image,rgb=make_bank(tmp_path);out=tmp_path/'out';out.mkdir();report=dict(banks=[]);calls=[]
    def native(session,boxes,actual_rgb):
        assert session is SESSION;calls.append(len(boxes))
        np.testing.assert_array_equal(actual_rgb,rgb)
        points=np.broadcast_to(boxes[:,None,:2],(len(boxes),133,2)).copy()
        scores=np.ones((len(boxes),133),np.float32);scores[0,:3]=[-2,0,4];points[0,0]=[-10,-5]
        return points,scores
    SESSION=object()
    worker.observe([image],out,native,SESSION,lambda _:rgb.copy(),report,lambda:None)
    assert calls==[2]and len(report['banks'])==1 and report['banks'][0]['person_ids']==tuple(image['person_ids'])
    with np.load(out/(image['image_id']+'.npz'),allow_pickle=False)as data:
        assert set(data)=={'boxes_original_xyxy','detector_scores','keypoints_original_xy','raw_scores','native_valid','in_original_image'}
        assert data['keypoints_original_xy'].shape==(2,133,2)and data['raw_scores'].dtype==np.float32
        assert data['raw_scores'][0,:3].tolist()==[-2,0,4]and data['native_valid'][0,:3].tolist()==[False,False,True]
        assert data['in_original_image'][0,0]==False
    assert(out/(image['image_id']+'.npz')).stat().st_mode&0o777==0o400


def test_no_person_empty_bank_skips_native_fallback(tmp_path):
    image,rgb=make_bank(tmp_path,0);out=tmp_path/'out';out.mkdir();report=dict(banks=[])
    def forbidden(*_):raise AssertionError('native full-image fallback forbidden')
    worker.observe([image],out,forbidden,object(),lambda _:rgb,report,lambda:None)
    assert report['banks'][0]['persons']==0
    with np.load(out/(image['image_id']+'.npz'),allow_pickle=False)as data:
        assert data['keypoints_original_xy'].shape==(0,133,2)and data['native_valid'].shape==(0,133)


def test_different_decoded_pixels_fail_before_model_call(tmp_path):
    image,rgb=make_bank(tmp_path);out=tmp_path/'out';out.mkdir();rgb[0,0,0]^=1
    with pytest.raises(ValueError,match='decoded RGB'):
        worker.observe([image],out,lambda *_:None,object(),lambda _:rgb,dict(banks=[]),lambda:None)
    assert not tuple(out.iterdir())


def test_native_mutation_fails_before_publication(tmp_path):
    image,rgb=make_bank(tmp_path);out=tmp_path/'out';out.mkdir()
    def wrong(_,boxes,actual):
        boxes.flags.writeable=True;boxes[0,0]=0
        return np.zeros((2,133,2),np.float64),np.ones((2,133),np.float32)
    with pytest.raises(ValueError,match='mutated'):
        worker.observe([image],out,wrong,object(),lambda _:rgb,dict(banks=[]),lambda:None)
    assert not tuple(out.iterdir())


def test_fresh_wheel_paths_only_original_pip_options(tmp_path):
    import dwpose_smoke as dw
    original=dw.pip_argv(tmp_path,worker.ROOT);new=worker.pip_arguments(tmp_path)
    assert len(original)==len(new)and original[:13]==new[:13]
    differences=[(a,b)for a,b in zip(original,new)if a!=b]
    assert len(differences)==2 and all('/weights/dwpose_native_v1/wheels/'in a and '/weights/dwpose_joint_pair_v1/wheels/'in b for a,b in differences)
    assert '--no-index'in new and '--no-deps'in new and '--no-compile'in new


def test_prospective_unpinned_gdi_config_fail_closed():
    code=Path(__file__).resolve().parents[1]
    value=json.loads((code/worker.CONFIG).read_bytes())
    if value['gdi']['producer_revision']is None:
        with pytest.raises(ValueError,match='Actual frozen GDI'):worker.configuration(code)
    else:
        assert worker.configuration(code)['gdi']['producer_revision']==value['gdi']['producer_revision']


def test_config_actual_pin_schema_and_no_numeric_retune(tmp_path):
    code=Path(__file__).resolve().parents[1];value=json.loads((code/worker.CONFIG).read_bytes())
    value['gdi']=dict(producer_revision='a'*40,report=dict(bytes=10,sha256='b'*64),native_report=dict(bytes=20,sha256='c'*64))
    target=tmp_path/worker.CONFIG;target.parent.mkdir();target.write_text(json.dumps(value))
    assert worker.configuration(tmp_path)['budget_seconds']==600
    value['budget_seconds']=700;target.write_text(json.dumps(value))
    with pytest.raises(ValueError):worker.configuration(tmp_path)


def test_new_asset_receipt_not_historical_smoke_and_cpu_only_wrapper():
    code=Path(__file__).resolve().parents[1];source=(code/'infra/openimages_joint_pair_dwpose.py').read_text()
    wrapper=(code/'infra/run_openimages_joint_pair_dwpose.sh').read_text()
    assert 'validate_assets('not in source and 'validate_smoke('not in source
    assert '--gpus'not in wrapper and '--network none'in wrapper and '--user 0:0'in wrapper
    assert '--cpus 4'in wrapper and '--memory 8g'in wrapper and '615s'in wrapper
    assert 'src=$OUT/native,dst=$OUT/native'in wrapper and 'src=$OUT,dst=$OUT'not in wrapper
    assert 'manifest.json'not in wrapper and 'cohort.json'not in wrapper and 'rights.json'not in wrapper
    assert 'CUDA_VISIBLE_DEVICES='in wrapper and 'docker rm --force "$NAME"'in wrapper

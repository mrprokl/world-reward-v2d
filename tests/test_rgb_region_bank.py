"""Tiny generated RGB/masks only; no native models or external image bytes."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
import rgb_region_bank as d


def seal(path,raw):
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(0o400)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def fixture(tmp_path):
    directory = tmp_path/'rgb'; directory.mkdir()
    buf = io.BytesIO(); Image.new('RGB',(19,13),'grey').save(buf,format='PNG')
    row = dict(image_id='f'*32,file='image_000000.png',width=19,height=13,**seal(directory/'image_000000.png',buf.getvalue()))
    value = dict(schema='world_reward.rgb_proposal_inputs.v1',images=[row])
    pin = seal(directory/'manifest.json',json.dumps(value).encode()); directory.chmod(0o500)
    p = d.configuration(REPO)
    return directory,pin,p,value


def record(mask,score=.9):
    h,w = mask.shape
    return dict(segmentation=mask,bbox=[0,0,w-1,h-1],crop_box=[0,0,w,h],area=int(mask.sum()),
        predicted_iou=score,stability_score=.99,point_coords=[[.25,.25]])


def test_defaults_do_not_follow_cohort_or_semantic_query():
    p = d.configuration(REPO)
    assert p['amg']['points_per_side'] == 32 and p['amg']['crop_n_layers'] == 0
    assert p['maximum_masks'] == 3072 and p['all_native_returned_masks']
    assert 'infra/proposal_stress_render.py' not in d.NATIVE_FILES
    assert 'configs/proposal_stress_v1.json' not in d.NATIVE_FILES
    assert not any('openimages_joint_pair' in str(x) for x in d.INPUTS)


def test_dispatch_markers_alongside_individual_source_files(tmp_path):
    job=tmp_path/'job';code=job/'code';code.mkdir(parents=True);revision='a'*40
    seal(job/'revision',(revision+'\n').encode());seal(job/'source-sha256',('b'*64+'\n').encode())
    assert d.dispatch_markers(code,revision)==(job/'revision',job/'source-sha256')
    with pytest.raises(ValueError):d.dispatch_markers(code,'c'*40)


def test_exact_public_rgb_pin_and_native_decode(tmp_path):
    directory,pin,p,value = fixture(tmp_path)
    assert d.public_inputs(directory,pin,p,1) == value
    assert d.decode(directory,value['images'][0]).shape == (13,19,3)


@pytest.mark.parametrize('bad',['private','duplicate','extra','changed','boolwidth','badid','path','count','writable'])
def test_public_rejects_private_or_unbound_payload(tmp_path,bad):
    directory,pin,p,value = fixture(tmp_path); row = value['images'][0]
    directory.chmod(0o700)
    if bad == 'private': row['scene'] = 'PRIVATE'
    elif bad == 'duplicate': value['images'].append(copy.deepcopy(row))
    elif bad == 'extra': seal(directory/'recipe.json',b'PRIVATE')
    elif bad == 'changed': seal(directory/row['file'],b'NOT ORIGINAL')
    elif bad == 'boolwidth': row['width'] = True
    elif bad == 'badid': row['image_id'] = 'role=actor'
    elif bad == 'path': row['file'] = '../image_000000.png'
    count = 2 if bad in ('duplicate','count') else 1
    pin = seal(directory/'manifest.json',json.dumps(value).encode())
    if bad != 'writable': directory.chmod(0o500)
    with pytest.raises(ValueError): d.public_inputs(directory,pin,p,count)


def test_native_duplicate_order_values_and_unbounded_predicted_iou():
    mask = np.ones((7,9),bool); records = [record(mask,1.1),record(mask,-.2)]
    p = d.configuration(REPO); row = dict(image_id='b'*32,height=7,width=9)
    saved = copy.deepcopy(records); arrays = d.region_arrays(records,row,p)
    assert np.array_equal(arrays['predicted_iou'],[1.1,-.2])  # No adapter clipping of native quality.
    assert arrays['mask_sha256'][0] == arrays['mask_sha256'][1]
    assert arrays['region_ids'][0] != arrays['region_ids'][1]
    assert np.array_equal(arrays['native_mask_indices'],[0,1])
    restored = np.unpackbits(arrays['packed_masks'],axis=1,bitorder='little')[:,:63].reshape(2,7,9).astype(bool)
    assert np.array_equal(restored,np.stack([mask,mask]))
    for a,b in zip(saved,records):
        assert all(np.array_equal(a[k],b[k]) for k in a)


def test_empty_bank_and_zero_area_preserved():
    p = d.configuration(REPO); row = dict(image_id='b'*32,height=7,width=9)
    assert d.region_arrays([],row,p)['packed_masks'].shape == (0,8)
    empty = np.zeros((7,9),bool)
    assert d.region_arrays([record(empty)],row,p)['mask_area'].tolist() == [0]


@pytest.mark.parametrize('bad',['extra','maskdtype','area','nan','infinity','crop','bbox','stability','point','overflow'])
def test_invalid_native_values_fail_without_rescue(bad):
    p = d.configuration(REPO); row = dict(image_id='b'*32,height=7,width=9)
    r = record(np.ones((7,9),bool))
    if bad == 'extra': r['owner'] = 'PRIVATE'
    elif bad == 'maskdtype': r['segmentation'] = r['segmentation'].astype(np.uint8)
    elif bad == 'area': r['area'] -= 1
    elif bad == 'nan': r['predicted_iou'] = float('nan')
    elif bad == 'infinity': r['predicted_iou'] = float('inf')
    elif bad == 'crop': r['crop_box'][2] -= 1
    elif bad == 'bbox': r['bbox'][3] = -1
    elif bad == 'stability': r['stability_score'] = 1.1
    elif bad == 'point': r['point_coords'] = [[float('nan'),0]]
    else: p['maximum_masks'] = 0
    with pytest.raises(ValueError): d.region_arrays([r],row,p)


def test_callback_all_rows_and_misses_saved_no_model(tmp_path):
    directory,pin,p,value = fixture(tmp_path); out = tmp_path/'out'; out.mkdir()
    report = dict(rgb_decodes=0,amg_attempts=0,amg_calls=0,banks=[])
    d.observe(value['images'],directory,out,p,lambda rgb:[],report,lambda:None)
    assert report['amg_calls'] == report['amg_attempts'] == report['rgb_decodes'] == 1
    assert report['banks'][0]['native_masks'] == 0
    with np.load(out/report['banks'][0]['file'],allow_pickle=False) as z:
        assert z['native_bbox_xywh'].shape == (0,4)
    assert not (out/report['banks'][0]['file']).stat().st_mode & 0o222


def test_rgb_mutation_and_bank_cap_fail_without_partial_adoption(tmp_path):
    directory,pin,p,value = fixture(tmp_path); out = tmp_path/'out'; out.mkdir()
    report = dict(rgb_decodes=0,amg_attempts=0,amg_calls=0,banks=[])
    def mutate(rgb): rgb[0,0] = 0; return []
    with pytest.raises(ValueError): d.observe(value['images'],directory,out,p,mutate,report,lambda:None)
    assert not list(out.iterdir())
    p['maximum_bank_bytes'] = 1
    with pytest.raises(ValueError): d.observe(value['images'],directory,out,p,lambda rgb:[],report,lambda:None)
    assert not list(out.iterdir())

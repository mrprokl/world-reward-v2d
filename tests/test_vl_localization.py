"""Manufactured boxes/tokens only; no local source media or inference."""
import json
import numpy as np
import pytest
from world_reward.vl_localization import requested_frames, question, observation, sam_handoff, encoded_layout


def answer(index=0, box=(300,400,500,600)):
    return json.dumps({'frames':[{'frame_index':index,'person_bbox':[0,0,1000,1000], 'object_bbox':box}]})


def obs(index=0, text=None):
    return observation(answer(index) if text is None else text,episode=14,frame_index=index,
                       width=1536,height=1152,rgb_sha256='a'*64)


def test_full_first30_and_original_scout_endpoints():
    ids = requested_frames(442)
    assert ids[:30] == tuple(range(30)) and ids[-1] == 441
    assert len(ids) == len(set(ids)) == 38


@pytest.mark.parametrize('total', (True,29,0,32.))
def test_reject_truncated_timeline(total):
    with pytest.raises(ValueError):requested_frames(total)


def test_same_general_prompt_one_requested_image():
    text = question('a black pan.','Pick up the object.',29)
    assert '"frame_indices": [29]' in text
    assert 'Pick up the object.' in text and 'INTEGER normalized full-image' in text


def test_original_resolution_mapping_once_without_offset():
    r = obs()
    assert r['object_bbox'] == tuple(x/1000.*s for x,s in zip((300,400,500,600),(1536,1152,1536,1152)))
    assert r['person_bbox'] == (0.,0.,1536.,1152.)


def test_null_and_invalid_not_repaired_or_copied():
    a=obs(0,answer(0,None));b=obs(1,answer(0))
    assert a['object_bbox'] is None and a['status']=='person_box_only'
    assert b['object_bbox'] is None and b['person_bbox'] is None and b['status']=='invalid_response'
    assert obs(1)['object_bbox'] is not None


def test_literal_ordered_sam_boxes_and_missing_no_interpolation():
    a,b=obs(),obs(1,answer(1,None))
    p=sam_handoff([a,b],episode=14,expected_indices=(0,1))
    assert p['prompts'][1]['box']==dict(zip(('x0','y0','x1','y1'),a['object_bbox']))
    assert p['missing']==[{'frame_index':1,'object_id':1,'reason':'person_box_only'}]
    assert p['sam_executed'] is False and p['interpolation'] is False
    with pytest.raises(ValueError):sam_handoff([b,a],episode=14,expected_indices=(0,1))


def test_layout_variable_lengths_grids_and_pixel_slices():
    ids=np.array([[0,0,8,99,9],[8,99,99,9,10]])
    mask=np.array([[0,0,1,1,1],[1,1,1,1,1]])
    grid=np.array([[1,2,2],[1,2,4]])
    rows=encoded_layout(ids,mask,grid,12,99)
    assert rows[0]['tokens']==[8,99,9] and rows[1]['patch_slice']==(4,12)


@pytest.mark.parametrize('problem', ('rightpad','placeholder','grid','patch_count'))
def test_layout_rejects_transport_errors(problem):
    ids=np.array([[0,8,99,9]]);m=np.array([[0,1,1,1]]);g=np.array([[1,2,2]]);n=4
    if problem=='rightpad':m=np.array([[1,1,1,0]])
    if problem=='placeholder':ids[0,2]=8
    if problem=='grid':g[0,0]=2
    if problem=='patch_count':n=8
    with pytest.raises(ValueError):encoded_layout(ids,m,g,n,99)

import json
import numpy as np
import pytest
from world_reward.hybrid_pair import indices,uncertainty_indices,prompt,decision,screening,mask_box,layout

T=[dict(id='p:7',role='person',visible_frames=[0,5,10],best_frame=5),
   dict(id='o:7',role='object',visible_frames=[5,10],best_frame=10)]

def answer(**changes):
    a=dict(state='accepted',person_id='p:7',object_id='o:7',evidence_frames=[5],uncertain_person_ids=[],uncertain_object_ids=[])
    a.update(changes);return json.dumps(a)

def test_indices_and_uncertain_scope():
    assert len(indices(100))==16 and indices(100)[-1]==99
    assert indices(3)==(0,1,2)
    assert uncertainty_indices(T)==(0,5,10)

def test_joint_prompt_no_coordinate_generation():
    p=prompt('a pan','pick up',T,[0,5,10])
    assert 'JOINTLY' in p and 'NEVER coordinates' in p and 'inert object' in p

def test_existing_role_bound_ids_only():
    assert decision(answer(),T,[0,5,10])['object_id']=='o:7'
    for change in ({'person_id':'o:7'},{'object_id':'o:99'},{'evidence_frames':[4]},
                   {'state':'uncertain'},{'evidence_frames':[]},{'extra':True}):
        with pytest.raises(ValueError):decision(answer(**change),T,[0,5,10])

def test_abstention_kept_not_filled():
    a=answer(state='uncertain',person_id=None,object_id=None,uncertain_object_ids=['o:7'])
    assert decision(a,T,[5])['person_id'] is None

def test_no_candidate_page_tail_drop():
    a={'candidates':[{'id':t['id'],'state':'uncertain'} for t in T]}
    assert len(screening(json.dumps(a),T))==2
    a['candidates'].pop()
    with pytest.raises(ValueError):screening(json.dumps(a),T)

def test_mask_bounds_are_exclusive_and_literal():
    a=np.zeros((5,6),bool);a[2:4,3:6]=True
    assert mask_box(a)==[3,2,6,4]
    assert mask_box(np.zeros((5,6),bool)) is None

def test_multi_image_conversation_binding():
    ids=np.array([[0,0,8,99,7,99,9],[8,99,99,7,99,9,10]])
    m=np.array([[0,0,1,1,1,1,1],[1,1,1,1,1,1,1]])
    grid=np.array([[1,2,2],[1,2,2],[1,2,4],[1,2,2]])
    r=layout(ids,m,grid,20,99,[2,2]);assert r[1]['patch_slice']==(8,20)
    with pytest.raises(ValueError):layout(ids,m,grid,20,99,[1,1])
    ids[1]=[8,99,7,99,99,9,10]
    with pytest.raises(ValueError,match='Ordered individual'):layout(ids,m,grid,20,99,[2,2])

def test_duplicate_keys_and_unknown_states_rejected():
    with pytest.raises(ValueError):decision('{"state":"accepted","state":"uncertain"}',T,[0])
    with pytest.raises(ValueError):screening('{"candidates":[{"id":"p:7","state":"likely"}]}',T)

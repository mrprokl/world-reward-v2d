from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import end2end_preview as p

def test_original_random_denominator_not_success_resampled():
    assert p.COHORT == [9,1,14,7]
    assert 'FAIL / not adopted' in p.labels(9,False)[2]
    assert 'GT pending' in p.labels(14,True)[2]

def probe():
    return dict(streams=[dict(codec_name='h264',width=960,height=280,r_frame_rate='30/1',avg_frame_rate='30/1',nb_read_frames='415')])

def test_full_original_encoded_frames_and_sharedview():
    p.probe_contract(probe(),415)
    for key,value in [('nb_read_frames','414'),('avg_frame_rate','29/1'),('width',640)]:
        r=probe();r['streams'][0][key]=value
        with pytest.raises(ValueError):p.probe_contract(r,415)

def test_revision_no_urls_or_expressions():
    assert p.revision('a'*40)=='a'*40
    for value in ['HEAD','a'*39,'a'*40+'?sig=secret','../x']:
        with pytest.raises(ValueError):p.revision(value)

def test_render_source_has_no_modification_or_pixel_label_prompt():
    import ast,inspect
    tree=ast.parse(inspect.getsource(p))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('refine_sequence','inference_pose') for n in ast.walk(tree))
    text=inspect.getsource(p)
    assert "floor = viewer.fixed_floor_height(ah)" in text
    assert "bh[frame]" in text and "range(total)" in text
    assert "str(video) in r['input_ledger']" in text

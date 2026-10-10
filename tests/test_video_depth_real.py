"""Tiny scoring/filename safety contracts, no local video or model execution."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pytest
from video_depth_real_run import score_depth
from video_depth_real_acquire import safe_member


def test_frozen_contiguous_public_selection_excludes_old_frames():
    p=json.loads((Path(__file__).parents[1]/'configs/video_depth_real_v1.json').read_text())
    assert p['challenge_inputs_allowed'] is False
    for seq in p['sequences']:
        assert [x['rank'] for x in seq['frames']]==list(range(200,296))
        assert sum(x['sensor_association_supported'] for x in seq['frames'])>=94
        assert all((x['depth'] is not None)==x['sensor_association_supported'] for x in seq['frames'])


def test_perfect_sensor_score_and_repeated_sensor_pair_exclusion():
    truth=np.ones((4,3,3),np.float32)*2;truth[1:]+=np.arange(1,4)[:,None,None]*.1
    z=truth.copy();mask=np.ones_like(z,bool)
    result=score_depth(z,mask,truth,np.ones(3),['a','a','b','c'])
    assert result['absrel']==0 and result['temporal_eulerian_error_m_s']==0
    assert result['temporal_pairs']==2 and result['coverage']==1


def test_missing_sensor_frame_is_not_filled_or_scored():
    truth=np.ones((4,2,2),np.float32);truth[1]=0
    z=np.ones_like(truth);mask=np.ones_like(truth,bool)
    r=score_depth(z,mask,truth,np.ones(3),['a',None,'b','c'])
    assert r['scored_frames']==3 and r['temporal_pairs']==1


@pytest.mark.parametrize('name',['/tmp/x','s/../rgb/x','other/rgb/x'])
def test_unsafe_archive_names_rejected(name):
    m=SimpleNamespace(name=name,size=3,isfile=lambda:True,isdir=lambda:False)
    with pytest.raises(ValueError):safe_member(m,'s')


def test_wrong_original_grid_rejected():
    with pytest.raises(ValueError):score_depth(np.ones((3,2,2)),np.ones((3,2,2),bool),np.ones((3,3,3)),np.ones(2),['a','b','c'])

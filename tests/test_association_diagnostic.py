"""Procedural observations only, no challenge labels or local model assets."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import association_diagnostic as diagnostic


def test_empty_observation_is_absent_not_perfect_match():
    empty=np.zeros((10,20),bool)
    value=diagnostic.comparison(empty,empty)
    assert value['iou'] is value['observation_coverage'] is value['prediction_precision'] is None
    assert value['observation']['bbox_xyxy'] is None


def test_disjoint_prediction_is_not_ignored():
    observed=np.zeros((10,20),bool);observed[1:4,2:6]=True
    predicted=np.zeros_like(observed);predicted[5:8,11:15]=True
    value=diagnostic.comparison(predicted,observed)
    assert value['iou']==value['observation_coverage']==value['prediction_precision']==0
    assert value['observation']['bbox_xyxy']==[2,1,6,4]
    assert value['centroid_distance_px']>9


def test_mask_coverage_and_precision_distinguish_inflated_predictions():
    observed=np.zeros((10,20),bool);observed[1:4,2:6]=True
    value=diagnostic.comparison(np.ones_like(observed),observed)
    assert value['observation_coverage']==1
    assert value['prediction_precision']==12/200 and value['iou']==12/200


def test_missing_prediction_counts_as_zero_coverage():
    empty=np.zeros((10,20),bool);observed=empty.copy();observed[1:4,2:6]=True
    value=diagnostic.comparison(empty,observed)
    assert value['iou']==value['observation_coverage']==0
    assert value['prediction_precision'] is None


def test_wrong_semantic_pair_can_still_have_perfect_self_consistency():
    # Deliberately no semantic truth passed: expose this diagnostic's limit.
    wrong_actor=np.zeros((10,20),bool);wrong_actor[5:8,11:15]=True
    assert diagnostic.comparison(wrong_actor,wrong_actor)['iou']==1


@pytest.mark.parametrize('mask',[np.zeros((2,2),np.uint8),np.zeros((2,2,1),bool)])
def test_mask_contract_is_explicit(mask):
    with pytest.raises(ValueError):diagnostic.mask_statistics(mask)

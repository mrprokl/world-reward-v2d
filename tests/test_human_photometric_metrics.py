import numpy as np
import pytest

from world_reward.human_photometric_metrics import geometry_diagnostic,quality_decision,second_difference_diagnostic


def arrays():
    pve=np.ones((8,3),np.float64);pve[:,2]=.9
    iou=np.full((24,3),.8,np.float64)
    hands=np.ones((8,3,2),np.float64)
    return pve,iou,hands


def test_exact_sham_and_full_group_paired_gate_no_adoption():
    values=arrays();before=[v.copy()for v in values];decision=quality_decision(*values)
    assert decision["synthetic_human_photometric_hypothesis_supported"]
    assert decision["median_paired_group_aligned_human_relative_gain"]==pytest.approx(.1)
    assert not decision["adoption_authorized"]and not decision["full_HOI_verified"]
    assert all(np.array_equal(v,b)for v,b in zip(values,before))


@pytest.mark.parametrize("fault",["nochange","group","hand","iou","zerobaseline"])
def test_reject_each_failed_gate_without_omitting_a_case(fault):
    p,i,h=arrays()
    if fault=="nochange":p[:,2]=1
    elif fault=="group":p[7,2]=1.051
    elif fault=="hand":h[7,2,1]=1.051
    elif fault=="iou":i[23,2]=.789
    else:p[7,:2]=0
    decision=quality_decision(p,i,h)
    assert not decision["synthetic_human_photometric_hypothesis_supported"]
    assert len(decision["per_frame_iou_change"])==24
    if fault=="zerobaseline":assert decision["zero_baseline_relative_gain_undefined"]and decision["per_group_aligned_human_relative_gain"]is None


@pytest.mark.parametrize("fault",["pvesham","iousham","handsham","count","nan","negative","badIoU","mask"])
def test_bad_coverage_metric_or_sham_is_integrity_failure(fault):
    p,i,h=arrays()
    if fault=="pvesham":p[0,1]=np.nextafter(1.,2.)
    elif fault=="iousham":i[0,1]=np.nextafter(.8,1.)
    elif fault=="handsham":h[0,1,1]=np.nextafter(1.,2.)
    elif fault=="count":p=p[:7]
    elif fault=="nan":p[0,2]=np.nan
    elif fault=="negative":h[0,2,0]=-1
    elif fault=="badIoU":i[0,2]=1.1
    else:p=np.ma.array(p,mask=False)
    with pytest.raises(ValueError):quality_decision(p,i,h)


def test_all_vertices_geometry_raw_camera_bias_not_alignment():
    q=np.zeros((18439,3),np.float64);q[:,2]=3
    p=q+[.1,0,.2];d=geometry_diagnostic(p,q)
    assert d["pve_cm"]==pytest.approx(np.sqrt(5)*10)
    assert d["centered_pve_cm"]<1e-10
    assert d["centroid_error_xyz_cm"]==pytest.approx([10.,0.,20.])
    p[18438,0]+=.5
    assert geometry_diagnostic(p,q)["centered_pve_cm"]>0


def test_temporal_zero_is_not_quality_success_no_fps_scaling():
    j=np.zeros((8,3,127,3),np.float64);j[:,2,:,0]=.1
    assert np.allclose(second_difference_diagnostic(j),10)
    assert np.array_equal(second_difference_diagnostic(j*0),np.zeros(8))

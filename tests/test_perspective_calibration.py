import json

import numpy as np
import pytest

from world_reward.perspective_calibration import fit_perspective_calibration


def analytic(focal=800.,pitch=.19,roll=-.12,*,ow=832,oh=640,rw=416,rh=320,left=0,top=0,width=416,height=320):
    """Independent own synthetic ray/gravity formula; no predicted model or RGB QA."""
    y,x = np.indices((height,width)); cx=rw/2-left-.5; cy=rh/2-top-.5
    rays = np.stack(((x-cx)/(rw/ow*focal),(y-cy)/(rh/oh*focal),np.ones_like(x)),axis=-1)
    gravity=np.array([-np.sin(roll)*np.cos(pitch),-np.cos(roll)*np.cos(pitch),np.sin(pitch)])
    unit=rays/np.linalg.norm(rays,axis=-1,keepdims=True)
    latitude=np.arcsin(unit@gravity)
    direction=gravity[:2]-gravity[2]*rays[:,:,:2]
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    up=direction.transpose(2,0,1); confidence=np.ones((height,width))
    kwargs=dict(original_width=ow,original_height=oh,resized_width=rw,resized_height=rh,crop_left=left,crop_top=top)
    return [up,latitude,confidence,confidence.copy()],kwargs,gravity


@pytest.mark.parametrize("pitch,roll", [(.19,-.12),(0.,0.),(.35,0.),(-.3,.2)])
def test_exact_independently_generated_fields_recover_focal_and_gravity(pitch,roll):
    inputs,kwargs,gravity=analytic(pitch=pitch,roll=roll)
    original=[value.copy() for value in inputs]
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert result.accepted,result.to_dict()
    assert result.focal_px==pytest.approx(800.,rel=1e-7)
    assert result.gravity==pytest.approx(gravity,abs=1e-8)
    assert np.linalg.norm(result.gravity)==pytest.approx(1)
    assert result.K[0]==pytest.approx((400,0,207.5)) and result.K[1]==pytest.approx((0,400,159.5))
    assert result.diagnostics["jacobian_rank"]==3 and result.diagnostics["half_grid_relative_disagreement"]<1e-7
    assert all(np.array_equal(before,after) for before,after in zip(original,inputs))
    report=result.to_dict(); json.dumps(report,allow_nan=False)
    assert report["accuracy_verified"] is report["ground_truth_used"] is False


def test_anisotropic_resize_and_asymmetric_crop_preserve_original_focal():
    inputs,kwargs,gravity=analytic(950.,ow=1000,oh=700,rw=470,rh=350,left=19,top=15,width=416,height=320)
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert result.accepted,result.to_dict()
    assert result.focal_px==pytest.approx(950.,rel=1e-7)
    assert result.K[0]==pytest.approx((446.5,0,215.5)) and result.K[1]==pytest.approx((0,475,159.5))
    assert result.gravity==pytest.approx(gravity,abs=1e-8)


def test_chw_latitude_and_float32_native_contract():
    inputs,kwargs,_=analytic(); inputs=[v.astype(np.float32) for v in inputs]; inputs[1]=inputs[1][None]
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert result.accepted and result.focal_px==pytest.approx(800.,rel=1e-6)


@pytest.mark.parametrize("confidence", [0.,1e-300,1e-20,1e-8])
def test_no_or_infinitesimal_confidence_abstains_not_renormalized(confidence):
    inputs,kwargs,_=analytic(); inputs[2][:]=confidence; inputs[3][:]=confidence
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted and "support" in result.reason and result.gravity is None
    assert result.focal_px==result.fallback_focal_px==np.hypot(832,640)
    assert result.K[0][0]==pytest.approx(.5*result.focal_px)


def test_confidence_concentrated_in_one_pixel_fails_effective_support():
    inputs,kwargs,_=analytic(); inputs[2][:]=0; inputs[3][:]=0; inputs[2][0,0]=inputs[3][0,0]=1
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted and result.diagnostics["support"]["train"]["up"]["effective_samples"]==1


def test_localized_evidence_fails_spatial_coverage():
    inputs,kwargs,_=analytic(); inputs[2][:]=0; inputs[3][:]=0
    inputs[2][:160,:104]=1; inputs[3][:160,:104]=1
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted
    assert result.diagnostics["support"]["half_0"]["up"]["samples"]>256


def test_constant_fields_abstain_with_fallback_not_fake_camera():
    inputs,kwargs,_=analytic(); inputs[0][0]=0; inputs[0][1]=-1; inputs[1][:]=0
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted and result.gravity is None
    assert result.focal_px==np.hypot(832,640)


def test_unsupported_small_grid_abstains_no_pixels_fabricated():
    inputs,kwargs,_=analytic(width=64,height=64)
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted and "support" in result.reason


def test_confidence_amplitude_cannot_change_equal_modality_fit():
    inputs,kwargs,_=analytic(); inputs[2]*=.8; inputs[3]*=.002
    result=fit_perspective_calibration(*inputs,**kwargs)
    assert result.accepted and result.focal_px==pytest.approx(800.,rel=1e-7)
    assert result.diagnostics["support"]["train"]["latitude"]["mean_confidence"]==pytest.approx(.002)


def test_failed_solver_never_returns_an_accepted_fallback(monkeypatch):
    import world_reward.perspective_calibration as module
    def failed(*args,**kwargs):
        assert kwargs["max_nfev"]==100 and kwargs["loss"]=="soft_l1"
        assert kwargs["f_scale"]==pytest.approx(np.deg2rad(2))
        raise FloatingPointError("own synthetic forced numerical failure")
    monkeypatch.setattr(module,"least_squares",failed)
    inputs,kwargs,_=analytic();result=module.fit_perspective_calibration(*inputs,**kwargs)
    assert not result.accepted and result.reason=="no_finite_converged_fit"
    assert result.focal_px==result.fallback_focal_px
    assert [row["start_factor"] for row in result.diagnostics["starts"]]==[.5,1.,2.]
    json.dumps(result.to_dict(),allow_nan=False)


@pytest.mark.parametrize("bad", ["up_shape","lat_shape","confidence_shape","integer","nan","infinity","up_zero","norm","latbound","confnegative","confhigh","masked"])
def test_invalid_field_inputs_raise_rather_than_silent_abstention(bad):
    inputs,kwargs,_=analytic()
    if bad=="up_shape":inputs[0]=inputs[0].transpose(1,2,0)
    if bad=="lat_shape":inputs[1]=inputs[1][None,None]
    if bad=="confidence_shape":inputs[2]=inputs[2][None]
    if bad=="integer":inputs[2]=inputs[2].astype(int)
    if bad=="nan":inputs[1][1,1]=np.nan
    if bad=="infinity":inputs[3][1,1]=np.inf
    if bad=="up_zero":inputs[0][:,1,1]=0
    if bad=="norm":inputs[0]*=1.01
    if bad=="latbound":inputs[1][1,1]=2
    if bad=="confnegative":inputs[2][1,1]=-.001
    if bad=="confhigh":inputs[3][1,1]=1.001
    if bad=="masked":inputs[0]=np.ma.array(inputs[0],mask=False)
    with pytest.raises(ValueError):fit_perspective_calibration(*inputs,**kwargs)


@pytest.mark.parametrize("name,value", [("original_width",True),("original_height",0),("resized_width",416.),("resized_height",-1),("crop_left",-1),("crop_top",False),("crop_left",1)])
def test_dimensions_and_crop_contract(name,value):
    inputs,kwargs,_=analytic();kwargs[name]=value
    with pytest.raises(ValueError):fit_perspective_calibration(*inputs,**kwargs)

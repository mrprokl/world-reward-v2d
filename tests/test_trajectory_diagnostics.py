"""Tiny synthetic geometry only; no challenge records, fitting or GPU."""
import numpy as np
import pytest

from world_reward.trajectory_diagnostics import (
    body_motion, mask_geometry, object_motion, project_centroids, summarize,
)


def scene(total=5):
    vertices=np.array([[0,0,0],[2,0,0],[0,1,0],[-3,0,1]],dtype=np.float64)
    rotations=np.repeat(np.eye(3)[None],total,axis=0)
    translations=np.repeat([[0.,0.,4.]],total,axis=0)
    return vertices,rotations,translations,np.arange(total,dtype=np.int64)


def test_static_all_original_indices_and_no_input_mutation():
    v,r,t,ids=scene(); copies=[a.copy() for a in (v,r,t,ids)]
    report=object_motion(v,r,t,ids,30)
    for key,values in report['series'].items():
        assert np.array_equal(values,np.zeros_like(values))
        assert report['statistics'][key]['rms']==0
    assert np.array_equal(report['frame_index'],ids)
    assert np.array_equal(report['series_frame_index']['surface_rms_step_m'],ids[1:])
    assert np.array_equal(report['series_frame_index']['surface_rms_acceleration_m_s2'],ids[1:-1])
    assert report['predictions_modified'] is False and report['quality_verified'] is False
    for actual,before in zip((v,r,t,ids),copies): assert np.array_equal(actual,before)


def test_uniform_translation_matches_whole_body_every_vertex():
    v,r,t,ids=scene(); t[:,0]=np.arange(len(t))*.25
    world=np.einsum('tij,vj->tvi',r,v)+t[:,None]
    obj=object_motion(v,r,t,ids,20); body=body_motion(world,ids,20)
    assert np.allclose(obj['series']['translation_step_m'],.25)
    assert np.allclose(obj['series']['centroid_step_m'],.25)
    assert np.allclose(obj['series']['surface_rms_step_m'],.25)
    assert np.allclose(obj['series']['surface_rms_acceleration_m_s2'],0)
    for key in body['series']: assert np.allclose(obj['series'][key],body['series'][key])


def test_rotation_off_origin_has_real_centroid_motion_not_zero_translation_proxy():
    v=np.array([[10.,0,0],[12.,0,0]])
    angles=np.radians([0,90,180,270])
    r=np.array([[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]] for a in angles])
    t=np.zeros((4,3)); ids=np.arange(4)
    report=object_motion(v,r,t,ids,1)
    assert np.allclose(report['series']['rotation_step_deg'],90)
    assert np.array_equal(report['series']['translation_step_m'],[0,0,0])
    assert np.allclose(report['series']['centroid_step_m'],np.sqrt(2)*11)
    assert np.allclose(report['series']['surface_rms_step_m'],np.sqrt(244))
    # Re-express the same rigid geometry about another canonical origin.
    offset=np.array([100.,-12,7]); changed_t=t+np.einsum('tij,j->ti',r,offset)
    reframed=object_motion(v-offset,r,changed_t,ids,1)
    assert np.allclose(reframed['centroid_xyz_m'],report['centroid_xyz_m'])
    for key in ('centroid_step_m','surface_rms_step_m','surface_rms_acceleration_m_s2'):
        assert np.allclose(reframed['series'][key],report['series'][key])


def test_analytic_rms_equals_all_vertices_no_component_or_frame_sampling():
    rng=np.random.default_rng(0); v=rng.normal(size=(13,3))*[1,2,3]+[9,8,7]
    angles=np.array([0,.3,-.8,1.6,np.pi])
    r=np.array([[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]] for a in angles])
    t=rng.normal(size=(5,3)); ids=np.arange(5)
    actual=object_motion(v,r,t,ids,12)
    world=np.einsum('tij,vj->tvi',r,v)+t[:,None]
    explicit=body_motion(world,ids,12)
    for key in explicit['series']: assert np.allclose(actual['series'][key],explicit['series'][key])
    assert np.allclose(actual['series']['rotation_step_deg'][-1],np.degrees(np.pi-1.6))


def test_so3_rotation_steps_include_180_and_small_angles_without_acos_bias():
    v,_,t,ids=scene(total=3); angles=np.array([0,np.pi,np.pi+1e-7])
    r=np.array([[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]] for a in angles])
    report=object_motion(v,r,t,ids,30)
    assert np.isclose(report['series']['rotation_step_deg'][0],180)
    assert np.isclose(report['series']['rotation_step_deg'][1],np.degrees(1e-7),rtol=1e-8,atol=0)


def test_acceleration_units_use_seconds_not_frame_or_metres_per_frame():
    v,r,t,ids=scene(); fps=10; acceleration=2.5
    t[:,0]=.5*acceleration*(ids/fps)**2
    report=object_motion(v,r,t,ids,fps)
    assert np.allclose(report['series']['centroid_acceleration_m_s2'],acceleration)
    assert np.allclose(report['series']['surface_rms_acceleration_m_s2'],acceleration)
    assert np.isclose(report['statistics']['surface_rms_acceleration_m_s2']['quantiles']['p95'],acceleration)


def test_supplied_plane_is_full_signed_gap_not_ground_truth_or_contact_score():
    v,r,t,ids=scene(); t[:,1]=[0,.1,.3,.6,1.]
    object_report=object_motion(v,r,t,ids,30,plane=[0,2,0,-1])
    body_report=body_motion(np.einsum('tij,vj->tvi',r,v)+t[:,None],ids,30,plane=[0,2,0,-1])
    assert np.allclose(object_report['series']['plane_min_signed_gap_m'],t[:,1]-.5)
    assert np.allclose(body_report['series']['plane_min_signed_gap_m'],t[:,1]-.5)
    assert np.array_equal(object_report['series_frame_index']['plane_min_signed_gap_m'],ids)
    assert object_report['ground_truth_used'] is False


@pytest.mark.parametrize('fault',['masked','bool','complex','object','nonfinite','shape','improper','missing_frame','short','fps'])
def test_invalid_object_inputs_are_refused_not_fixed_or_filtered(fault):
    v,r,t,ids=scene(); fps=30
    if fault=='masked': v=np.ma.array(v,mask=False)
    elif fault=='bool': v=v.astype(bool)
    elif fault=='complex': v=v.astype(complex)
    elif fault=='object': v=v.astype(object)
    elif fault=='nonfinite': t[2,0]=np.nan
    elif fault=='shape': v=v[:,:2]
    elif fault=='improper': r[0,0,0]=-1
    elif fault=='missing_frame': ids[2]=5
    elif fault=='short': r,t,ids=r[:2],t[:2],ids[:2]
    else: fps=0
    with pytest.raises(ValueError): object_motion(v,r,t,ids,fps)


@pytest.mark.parametrize('vertices',[np.ones((4,0,3)),np.ones((4,3)),np.ma.array(np.ones((4,3,3))),np.full((4,3,3),np.inf)])
def test_invalid_body_geometry_is_refused(vertices):
    with pytest.raises(ValueError): body_motion(vertices,np.arange(4),30)


def test_mask_all_pixels_centroid_half_open_bbox_and_valid_empty_mask():
    mask=np.zeros((7,11),dtype=np.uint8); mask[2:5,4:8]=255
    assert mask_geometry(mask)==dict(area_pixels=12,area_fraction=12/77,centroid_xy=[5.5,3.],bbox_xyxy=[4,2,8,5])
    assert mask_geometry(mask>0)==mask_geometry(mask)
    assert mask_geometry(mask//255)==mask_geometry(mask)
    assert mask_geometry(mask*0)==dict(area_pixels=0,area_fraction=0.,centroid_xy=None,bbox_xyxy=None)


@pytest.mark.parametrize('mask',[np.zeros((0,5),dtype=bool),np.zeros(3,dtype=bool),np.ones((2,2))*255,
    np.array([[0,2]],dtype=np.uint8),np.array([[0,1,255]],dtype=np.uint8),np.ma.array([[True]])])
def test_invalid_masks_are_refused(mask):
    with pytest.raises(ValueError): mask_geometry(mask)


def test_project_true_centroids_with_fixed_camera_positive_z():
    k=np.array([[100.,2,50],[0,80,40],[0,0,1]])
    c=np.array([[0.,0,2],[1,2,4]])
    assert np.allclose(project_centroids(c,k),[[50,40],[76,80]])


@pytest.mark.parametrize('fault',['behind','zero_z','bad_k','negative_focal','nan','masked'])
def test_projection_invalid_inputs_fail(fault):
    c=np.array([[1.,2,4]]); k=np.array([[100.,0,50],[0,80,40],[0,0,1]])
    if fault=='behind': c[:,2]=-1
    elif fault=='zero_z': c[:,2]=0
    elif fault=='bad_k': k[2,0]=1
    elif fault=='negative_focal': k[0,0]=-1
    elif fault=='nan': c[0,0]=np.nan
    else: c=np.ma.array(c)
    with pytest.raises(ValueError): project_centroids(c,k)


def test_summary_linear_quantiles_mad_and_rms_no_outlier_filtering():
    report=summarize([0,1,2,3,100])
    assert report['count']==5 and report['median']==2 and report['mad']==1
    assert np.isclose(report['rms'],np.sqrt(10014/5))
    assert report['quantiles']['p100']==100 and np.isclose(report['quantiles']['p95'],80.6)


@pytest.mark.parametrize('x',[[],[np.nan],[np.inf],[[1,2]],np.ma.array([1]),[True],['1']])
def test_invalid_summary_is_refused(x):
    with pytest.raises(ValueError): summarize(x)


@pytest.mark.parametrize('p',[[0,0,0,1],[0,1,0],[0,np.nan,0,1]])
def test_invalid_supplied_planes_are_refused(p):
    with pytest.raises(ValueError): object_motion(*scene(),30,plane=p)

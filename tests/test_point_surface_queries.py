"""Independent tiny planes/box/perspective: no assets, images or GT records."""
import numpy as np
import pytest
from world_reward.point_surface_queries import canonical_surface_queries


def fixture():
    v = np.array([[-4., -3.13, 2.], [5., -3.13, 2.], [5., 4.07, 2.], [-4., 4.07, 2.]])
    return dict(vertices=v, faces=np.array([[0, 1, 2], [0, 2, 3]], np.int64), R0=np.eye(3), t0=np.zeros(3),
                K=np.array([[32., 0., 16.], [0., 24., 12.], [0., 0., 1.]]), automatic_mask=np.ones((24, 32), bool),
                inferred_depth_valid=np.ones((24, 32), bool), image_width=32, image_height=24)


def test_plane_centres_canonical_points_and_unchanged_query_order():
    data=fixture(); original={k:v.copy() for k,v in data.items() if isinstance(v,np.ndarray)}
    result=canonical_surface_queries(**data)
    expected=np.array([(y,x) for y in (1,3) for x in range(1,32,2)],np.int64)
    np.testing.assert_array_equal(result.grid_indices,expected)
    np.testing.assert_array_equal(result.query_points,np.column_stack((np.zeros(32),expected+.5)))
    xy=(result.query_points[:,[2,1]]-[16,12])/[32,24]*2
    np.testing.assert_allclose(result.canonical_points,np.column_stack((xy,np.full(32,2))),atol=2e-15)
    np.testing.assert_allclose(result.camera_depth_m,2,atol=1e-15)
    assert all(not x.flags.writeable for x in vars(result).values())
    assert result.canonical_points.dtype==result.query_points.dtype==np.float64
    for k,v in original.items():np.testing.assert_array_equal(data[k],v)


def test_initial_rigid_inverse_keeps_original_canonical_frame():
    data=fixture(); angle=.73
    r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    t=np.array([.2,-.1,.3]); data['vertices']=(data['vertices']-t)@r; data['R0']=r;data['t0']=t
    result=canonical_surface_queries(**data)
    xy=(result.query_points[:,[2,1]]-[16,12])/[32,24]*2
    camera=np.column_stack((xy,np.full(32,2)))
    np.testing.assert_allclose(result.canonical_points,(camera-t)@r,atol=2e-15)


def test_perspective_tilted_plane_matches_independent_plane_equation():
    data=fixture(); data['vertices'][:,2]=2+.13*data['vertices'][:,0]-.07*data['vertices'][:,1]
    result=canonical_surface_queries(**data)
    d=np.column_stack(((result.query_points[:,[2,1]]-[16,12])/[32,24],np.ones(32)))
    expected=2/(1-.13*d[:,0]+.07*d[:,1])
    np.testing.assert_allclose(result.camera_depth_m,expected,atol=2e-15)
    np.testing.assert_allclose(result.canonical_points,d*expected[:,None],atol=3e-15)


def test_nearest_surface_original_face_even_if_rear_is_first():
    data=fixture(); front=data['vertices'].copy(); rear=front.copy();rear[:,2]=4
    data['vertices']=np.vstack((rear,front));data['faces']=np.vstack((data['faces'],data['faces']+4))
    result=canonical_surface_queries(**data)
    np.testing.assert_allclose(result.camera_depth_m,2,atol=1e-15)
    assert np.all(result.face_indices>=2)


def test_closed_box_parallel_side_faces_do_not_invent_singular_hits():
    data=fixture(); front=data['vertices'];rear=front.copy();rear[:,2]=4
    data['vertices']=np.vstack((front,rear));data['faces']=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]],np.int64)
    np.testing.assert_allclose(canonical_surface_queries(**data).camera_depth_m,2,atol=1e-15)


def test_automatic_support_intersection_not_future_quality_selects_queries():
    data=fixture();data['automatic_mask'][1,1]=False;data['inferred_depth_valid'][1,3]=False
    result=canonical_surface_queries(**data)
    assert result.grid_indices[0].tolist()==[1,5] and result.grid_indices[-1].tolist()==[5,3]
    data['automatic_mask'][:]=False
    with pytest.raises(ValueError,match='Fewer than8'):canonical_surface_queries(**data)


def test_no_surface_is_no_query_not_fabricated_pointmap():
    data=fixture();data['vertices'][:,0]+=20
    with pytest.raises(ValueError,match='Fewer than8'):canonical_surface_queries(**data)


@pytest.mark.parametrize('key,value',[
 ('vertices',np.zeros((4,3))),('vertices',np.ones((4,3),np.int64)),('vertices',np.full((4,3),np.nan)),
 ('faces',np.array([[0,1,8]],np.int64)),('faces',np.array([[-1,1,2]],np.int64)),
 ('faces',np.array([[0,0,1]],np.int64)),('faces',np.array([[0,1,2]],np.int32)),
 ('R0',np.diag([1.,1.,-1.])),('R0',np.diag([2.,1.,1.])),('t0',np.ones(4)),
 ('K',np.eye(3,dtype=np.int64)),('K',np.array([[1.,.1,0],[0,1,0],[0,0,1]])),
 ('K',np.array([[-1.,0,0],[0,1,0],[0,0,1]])),('K',np.array([[1.,0,0],[0,1,0],[0,.1,1]])),
 ('automatic_mask',np.ones((24,32),np.uint8)),('inferred_depth_valid',np.ones((23,32),bool)),
 ('image_width',True),('image_height',11),('vertices',np.ma.array(np.ones((4,3)))),
])
def test_geometry_pose_camera_dtype_grid_rejected(key,value):
    data=fixture();data[key]=value
    with pytest.raises(ValueError):canonical_surface_queries(**data)


def test_behind_camera_original_vertex_not_clipped_or_removed():
    data=fixture();data['vertices'][0,2]=0
    with pytest.raises(ValueError,match='in front'):canonical_surface_queries(**data)


def test_duplicate_surface_and_exact_shared_edge_fail_not_best_face_refill():
    data=fixture();data['faces']=np.vstack((data['faces'],data['faces']))
    with pytest.raises(ValueError,match='depth tie'):canonical_surface_queries(**data)
    data=fixture();data['K'][0,2]=1.5;data['K'][1,2]=1.5
    data['vertices']=np.array([[-2.,-2.,2.],[2.,-2.,2.],[2.,2.,2.],[-2.,2.,2.]])
    with pytest.raises(ValueError,match='ambiguous boundary'):canonical_surface_queries(**data)


def test_coplanar_supported_ray_fails_no_uncertain_surface_deletion():
    data=fixture();data['K'][0,2]=1.5
    data['vertices']=np.vstack((data['vertices'],[[0.,-1.,1.],[0.,1.,1.],[0.,0.,3.]]))
    data['faces']=np.vstack((data['faces'],[4,5,6]))
    with pytest.raises(ValueError,match='coplanar'):canonical_surface_queries(**data)


def test_chunk_contract_first_positive_surface_across4096_faces():
    # Own tiny4V/2F fixture, repeat rear triangles outside supported rays;
    # this probes chunking only, not a manufactured dataset or repaired mesh.
    data=fixture();other=np.array([[50.,50.,3.],[51.,50.,3.],[50.,51.,3.]])
    data['vertices']=np.vstack((data['vertices'],other))
    data['faces']=np.vstack((np.tile([4,5,6],(4096,1)),data['faces'])).astype(np.int64)
    result=canonical_surface_queries(**data)
    assert np.all(result.face_indices>=4096)
    np.testing.assert_allclose(result.camera_depth_m,2,atol=1e-15)


def test_relative_depth_tie_and_boundary_margin_fail_without_absolute_metres():
    data=fixture();front=data['vertices'].copy();rear=front.copy();rear[:,2]=2+1e-14
    data['vertices']=np.vstack((front,rear));data['faces']=np.vstack((data['faces'],data['faces']+4))
    with pytest.raises(ValueError,match='depth tie'):canonical_surface_queries(**data)
    data=fixture();data['K'][0,2]=1.5;data['K'][1,2]=1.5
    data['vertices']=np.array([[-2.,-2.+1e-14,2.],[2.,-2.+1e-14,2.],
                               [2.,2.+1e-14,2.],[-2.,2.+1e-14,2.]])
    with pytest.raises(ValueError,match='ambiguous boundary'):canonical_surface_queries(**data)


def test_float32_input_preserves_original_camera_and_metres_scale_not_fitted():
    data=fixture()
    for name in('vertices','R0','t0','K'):data[name]=data[name].astype(np.float32)
    result=canonical_surface_queries(**data)
    np.testing.assert_allclose(result.camera_depth_m,2,atol=2e-15)
    data['vertices']*=1e-8;data['t0']*=1e-8
    scaled=canonical_surface_queries(**data)
    np.testing.assert_allclose(scaled.query_points,result.query_points,atol=0)
    np.testing.assert_allclose(scaled.canonical_points,result.canonical_points*1e-8,rtol=1e-6,atol=0)


def test_positive_z_numeric_overflow_is_not_hidden_as_no_surface():
    data=fixture();data['K'][0,0]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='finite numeric range'):canonical_surface_queries(**data)

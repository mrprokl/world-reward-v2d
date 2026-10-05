"""Tiny procedural QA renderer tests, no challenge records or heavy local data."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import reconstruction_preview as preview


def test_frame_sampling_is_predeclared_not_quality_selected():
    assert preview.frame_indices(399) == [0,199,398]
    assert preview.frame_indices(415) == [0,207,414]
    assert preview.frame_indices(4) == [0,1,3]
    for bad in (0,2,True,3.0):
        with pytest.raises(ValueError): preview.frame_indices(bad)


def test_depth_uses_perspective_and_pixel_centres_without_mutation():
    from camera_render import _smoke_reference
    width,height,K,uv,triangle,_,reference,bary = _smoke_reference()
    original = triangle.copy()
    depth = preview.raster_depth(triangle, np.array([[0,1,2]]), K, width,height)
    inside = (bary > 1e-4).all(axis=-1)
    outside = (bary < -1e-4).any(axis=-1)
    np.testing.assert_allclose(depth[inside],reference[inside],atol=1e-6)
    assert np.isinf(depth[outside]).all()
    np.testing.assert_array_equal(triangle,original)


def test_depth_is_face_order_independent_and_keeps_nearest_surface():
    K=np.array([[20.,0,16.],[0,20.,12.],[0,0,1.]])
    triangle=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]])
    vertices=np.concatenate((triangle,triangle*2))
    faces=np.array([[0,1,2],[3,4,5]])
    one=preview.raster_depth(vertices,faces,K,32,24)
    two=preview.raster_depth(vertices,faces[::-1],K,32,24)
    np.testing.assert_array_equal(one,two)
    np.testing.assert_allclose(one[np.isfinite(one)],2.)


@pytest.mark.parametrize('points',[np.array([[0.,0.],[30.,20.]]),
    np.array([[300.,210.],[380.,300.]]),np.array([[-100.,-100.],[-80.,-80.]]),
    np.array([[-1e4,-1e4],[1e4,1e4]])])
def test_detail_is_automatic_bounded_and_never_distorts_aspect(points):
    x0,y0,x1,y1=preview.detail_box(points,320,240)
    assert 0<=x0<x1<=320 and 0<=y0<y1<=240
    assert (x1-x0)*3==(y1-y0)*4


def test_overlay_resolves_human_object_occlusion_without_drawing_absent_mesh():
    rgb=np.full((1,3,3),100,np.uint8)
    human=np.array([[2.,4.,np.inf]])
    obj=np.array([[3.,3.,np.inf]])
    result=preview.overlay(rgb,human,obj)
    np.testing.assert_array_equal(result[0,0],np.rint(48+.52*np.array(preview.HUMAN_RGB)))
    np.testing.assert_array_equal(result[0,1],np.rint(48+.52*np.array(preview.OBJECT_RGB)))
    np.testing.assert_array_equal(result[0,2],rgb[0,2])

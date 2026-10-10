"""CPU procedural bounds against the actual pinned CUDA overlap equations."""

import numpy as np
import pytest

from world_reward.raster_capacity import conservative_raster_capacity
import world_reward.raster_capacity as capacity


def native_edges(size, other, bin_size, *, fused=False):
    """Literal source operations, plus the permitted fused multiply-add case."""
    f = np.float32
    r = f(f(size * f(2)) / other) if size > other else f(2)
    half, half_pixel = f(r / f(2)), f(f(r / f(2)) / size)
    def pixel(i):
        a = f(float(r) * i + float(half)) if fused else f(f(r * i) + half)
        return f(-half + f(a / size))
    n = (size + bin_size - 1) // bin_size
    return (np.array([f(pixel(i * bin_size) - half_pixel) for i in range(n)]),
            np.array([f(pixel((i + 1) * bin_size - 1) + half_pixel) for i in range(n)]))


def brute_counts(vertices, faces, height, width, bin_size, *, fused=False):
    xmin, xmax = native_edges(width, height, bin_size, fused=fused)
    ymin, ymax = native_edges(height, width, bin_size, fused=fused)
    triangles = vertices[faces]
    lo, hi = triangles[..., :2].min(1), triangles[..., :2].max(1)
    return np.array([[np.count_nonzero((lo[:, 0] <= xr) & (xl < hi[:, 0]) &
                     (lo[:, 1] <= yr) & (yl < hi[:, 1]))
                     for xl, xr in zip(xmin, xmax)] for yl, yr in zip(ymin, ymax)])


@pytest.mark.parametrize("height,width", [(1152, 1536), (96, 128), (151, 97),
    (97, 151), (64, 64), (1, 1), (2048, 1537)])
def test_capacity_bounds_source_and_fused_overlap_all_faces(height, width):
    rng = np.random.default_rng(height + width)
    vertices = rng.uniform(-2, 2, (300, 3)).astype(np.float32)
    vertices[:, 2] = 2
    faces = np.arange(300).reshape(-1, 3)
    before = vertices.copy()
    result = conservative_raster_capacity(vertices, faces, (height, width))
    for fused in (False, True):
        expected = brute_counts(vertices, faces, height, width, result.bin_size, fused=fused)
        assert np.all(result.counts[0] >= expected)
        assert result.max_faces_per_bin >= expected.max()
    np.testing.assert_array_equal(vertices, before)
    assert not result.counts.flags.writeable
    assert result.bin_storage_bytes == result.counts.size * result.max_faces_per_bin * 4


def test_exact_bin_boundaries_and_adjacent_float32_are_not_underallocated():
    xl, xh = native_edges(1536, 1152, 128)
    yl, yh = native_edges(1152, 1536, 128)
    vertices = []
    for x in np.concatenate((xl, xh)):
        for y in np.concatenate((yl, yh)):
            vertices.extend([[np.nextafter(x, np.float32(-np.inf)), y, 1],
                             [np.nextafter(x, np.float32(np.inf)), y, 1],
                             [x, np.nextafter(y, np.float32(np.inf)), 1]])
    vertices = np.asarray(vertices, np.float32)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    result = conservative_raster_capacity(vertices, faces, (1152, 1536))
    for fused in (False, True):
        assert np.all(result.counts[0] >= brute_counts(vertices, faces, 1152, 1536, 128, fused=fused))


def test_batch_capacity_is_maximum_not_sum_and_dense_empty_bins_reduce_storage():
    triangles = np.array([[[-.21, -.21, 1], [-.20, -.21, 1], [-.21, -.20, 1]],
                          [[.80, .80, 1], [.81, .80, 1], [.80, .81, 1]]], np.float32)
    vertices = np.repeat(triangles, 30, axis=0).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    batch = np.stack((vertices, vertices + np.array([0, -.2, 0], np.float32)))
    result = conservative_raster_capacity(batch, faces, (1152, 1536))
    assert result.max_faces_per_bin == 30 < len(faces)
    assert result.counts.shape == (2, 9, 12)
    for i, mesh in enumerate(batch):
        assert np.all(result.counts[i] >= brute_counts(mesh, faces, 1152, 1536, 128))


def test_off_image_faces_are_retained_in_input_and_capacity_is_valid_one():
    vertices = np.array([[9, 9, 1], [9.1, 9, 1], [9, 9.1, 1]], np.float32)
    result = conservative_raster_capacity(vertices, [[0, 1, 2]], (96, 128))
    assert result.max_faces_per_bin == 1 and not np.any(result.counts)
    assert result.faces_per_mesh == 1


def test_every_face_across_entire_grid_keeps_full_face_count():
    vertices = np.array([[-9, -9, 1], [9, -9, 1], [0, 9, 1]], np.float32)
    faces = np.tile([0, 1, 2], (50, 1))
    result = conservative_raster_capacity(vertices, faces, (96, 128))
    assert result.max_faces_per_bin == 50 and np.all(result.counts == 50)


@pytest.mark.parametrize("size,expected", [(1, 8), (64, 8), (65, 16), (256, 16),
    (257, 32), (512, 32), (513, 64), (1024, 64), (1025, 128), (2048, 128)])
def test_matches_exact_native_cuda_bin_size_heuristic(size, expected):
    v = np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1]], np.float32)
    assert conservative_raster_capacity(v, [[0, 1, 2]], (size, size)).bin_size == expected


@pytest.mark.parametrize("vertices,faces,image,kwargs", [
    (np.zeros((3, 3), np.float64), [[0, 1, 2]], (96, 128), {}),
    (np.zeros((3, 3), np.float32), [[0, 1, 2]], (96, 128), {}),
    (np.ones((3, 3), np.float32), [[0, 1, 3]], (96, 128), {}),
    (np.ones((3, 3), np.float32), [[0., 1., 2.]], (96, 128), {}),
    (np.ones((3, 3), np.float32), [[0, 1, 2]], (96, True), {}),
    (np.ones((3, 3), np.float32), [[0, 1, 2]], (96, 128), {'bin_size': 0}),
    (np.ones((3, 3), np.float32), [[0, 1, 2]], (96, 128), {'bin_size': 1}),
])
def test_invalid_or_incompatible_geometry_grid_fails(vertices, faces, image, kwargs):
    with pytest.raises(ValueError):
        conservative_raster_capacity(vertices, faces, image, **kwargs)


def reference_capacity_counts(vertices, faces, image_size, bin_size):
    """Literal pre-optimization triangle gather/reduction, procedural inputs."""
    vertices=np.asarray(vertices)
    if vertices.ndim==2:vertices=vertices[None]
    height,width=image_size
    bh,bw=(height+bin_size-1)//bin_size,(width+bin_size-1)//bin_size
    xl,xh=capacity._bin_edges(width,height,bin_size)
    yl,yh=capacity._bin_edges(height,width,bin_size)
    stride,cells=bw+1,(bh+1)*(bw+1)
    result=[]
    for vertex in vertices:
        difference=np.zeros(cells,np.int64)
        for start in range(0,len(faces),capacity._CHUNK_FACES):
            triangles=vertex[faces[start:start+capacity._CHUNK_FACES],:2]
            low,high=triangles.min(axis=1),triangles.max(axis=1)
            x0=np.searchsorted(xh,low[:,0],side='left')
            x1=np.searchsorted(xl,high[:,0],side='right')
            y0=np.searchsorted(yh,low[:,1],side='left')
            y1=np.searchsorted(yl,high[:,1],side='right')
            hit=(x0<x1)&(y0<y1)
            x0,x1,y0,y1=(a[hit] for a in (x0,x1,y0,y1))
            for corners,sign in ((y0*stride+x0,1),(y0*stride+x1,-1),
                                  (y1*stride+x0,-1),(y1*stride+x1,1)):
                difference+=sign*np.bincount(corners,minlength=cells)
        result.append(difference.reshape(bh+1,bw+1).cumsum(0).cumsum(1)[:-1,:-1])
    return np.asarray(result)


@pytest.mark.parametrize('layout',['C','F','strided'])
def test_xy_extrema_byte_exact_and_do_not_alias_or_mutate_vertices(layout):
    rng=np.random.default_rng(24)
    original=rng.uniform(-4,4,(57,3)).astype(np.float32);original[:,2]=1
    # Include both signs of zero, equal extrema and adjacent finite float32.
    original[:3,:2]=[[0.,-0.],[-0.,0.],[0.,-0.]]
    original[3,:2]=np.nextafter(np.float32(1),np.float32(np.inf))
    original[4,:2]=np.nextafter(np.float32(1),np.float32(-np.inf))
    if layout=='F':vertices=np.asfortranarray(original)
    elif layout=='strided':
        storage=np.empty((len(original)*2,6),np.float32)
        vertices=storage[::2,::2];vertices[:]=original
    else:vertices=original.copy()
    vertices.flags.writeable=False
    faces=np.arange(len(vertices),dtype=np.int64).reshape(-1,3)
    before=vertices.tobytes();triangles=vertices[faces,:2]
    low,high=capacity._triangle_xy_bounds(vertices,faces)
    assert low.dtype==high.dtype==np.float32
    assert low.tobytes()==triangles.min(axis=1).tobytes()
    assert high.tobytes()==triangles.max(axis=1).tobytes()
    assert not np.shares_memory(low,vertices) and not np.shares_memory(high,vertices)
    assert not np.shares_memory(low,high) and vertices.tobytes()==before


@pytest.mark.parametrize('batch_size',[1,4])
@pytest.mark.parametrize('chunk_size',[1,7,65536])
def test_all_capacity_counts_byte_exact_b1_b4_chunk_tails_extreme_coordinates(monkeypatch,batch_size,chunk_size):
    monkeypatch.setattr(capacity,'_CHUNK_FACES',chunk_size)
    rng=np.random.default_rng(46)
    vertices=rng.uniform(-3,3,(batch_size,51,3)).astype(np.float32)
    vertices[...,2]=1
    # Entire-grid and off-image triangles, extreme finite values, zero signs.
    limit=np.finfo(np.float32).max
    vertices[:,0:3,:2]=[[-limit,-limit],[limit,-limit],[0.,limit]]
    vertices[:,3:6,:2]=[[limit,limit],[limit,limit],[limit,limit]]
    vertices[:,6:9,:2]=[[0.,-0.],[-0.,0.],[0.,0.]]
    xl,xh=capacity._bin_edges(1536,1152,128)
    yl,yh=capacity._bin_edges(1152,1536,128)
    vertices[:,9:12,:2]=[[xl[2],yl[2]],[xh[3],yh[3]],[xl[2],yh[3]]]
    faces=np.arange(51,dtype=np.int64).reshape(-1,3)
    vertices.flags.writeable=False;faces.flags.writeable=False
    before=(vertices.tobytes(),faces.tobytes())
    result=capacity.conservative_raster_capacity(vertices,faces,(1152,1536))
    reference=reference_capacity_counts(vertices,faces,(1152,1536),128)
    assert result.counts.tobytes()==reference.tobytes()
    assert result.max_faces_per_bin==max(1,int(reference.max()))
    assert result.faces_per_mesh==len(faces) and result.bin_size==128
    assert (vertices.tobytes(),faces.tobytes())==before and not result.counts.flags.writeable
    if batch_size==1:
        scalar=capacity.conservative_raster_capacity(vertices[0],faces,(1152,1536))
        assert scalar.counts.tobytes()==result.counts.tobytes()


def test_pointwise_bounds_receive_every_original_face_exactly_once_per_mesh(monkeypatch):
    monkeypatch.setattr(capacity,'_CHUNK_FACES',7)
    vertices=np.ones((4,51,3),np.float32)
    vertices[:,:,:2]=np.random.default_rng(4).uniform(-2,2,(4,51,2))
    faces=np.arange(51,dtype=np.int64).reshape(-1,3);seen=[]
    original=capacity._triangle_xy_bounds
    def recorded(vertex,chunk):
        seen.append(chunk.copy());return original(vertex,chunk)
    monkeypatch.setattr(capacity,'_triangle_xy_bounds',recorded)
    result=capacity.conservative_raster_capacity(vertices,faces,(1152,1536))
    assert len(seen)==12 and [len(chunk) for chunk in seen]==[7,7,3]*4
    for batch in range(4):np.testing.assert_array_equal(np.concatenate(seen[batch*3:batch*3+3]),faces)
    assert result.faces_per_mesh==len(faces)

"""CPU procedural bounds against the actual pinned CUDA overlap equations."""

import numpy as np
import pytest

from world_reward.raster_capacity import conservative_raster_capacity


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

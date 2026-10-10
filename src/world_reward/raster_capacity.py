"""Non-overflow bin capacity for the existing, full-resolution CUDA raster.

PyTorch3D 33824be3's coarse kernel inserts every projected triangle AABB into
overlapping bins. Its fine kernel scans *all* M slots at every pixel, even -1
slots, so setting M to the whole face count is needlessly expensive. Overflow
only emits CUDA printf and can silently omit faces; a heuristic is not safe.

Supply the exact float32 NDC vertices returned by MeshRasterizer.transform,
not a separately implemented camera projection. We conservatively count all
triangle AABBs, including off-image triangles when they overlap a padded bin.
Outward float32 intervals cover ordinary rounded/fused arithmetic of the pinned
CUDA bin-edge formula; closed overlap also includes its half-open boundaries.
Only zero blur, positive camera Z and unchanged, unclipped faces are supported.
This bounds storage, not image quality. GPU output parity remains a caller gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
_CHUNK_FACES = 65536  # CPU workspace bound, not an evidence/geometry limit.


@dataclass(frozen=True)
class RasterCapacity:
    bin_size: int
    max_faces_per_bin: int
    counts: np.ndarray  # Conservative [B, bins_y, bins_x]; never a face filter.
    faces_per_mesh: int

    @property
    def bin_storage_bytes(self) -> int:
        return int(self.counts.size) * self.max_faces_per_bin * 4


def _size(value, name):
    # Exact conversion to CUDA float32 for pixel indices/range arithmetic.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 < value <= 2**20:
        raise ValueError(name + " must be an integer in [1, 2**20]")
    return int(value)


def _round_out(low, high):
    """Enclose both separate and fused ordinary float32 operation results."""
    return (np.nextafter(np.asarray(low, dtype=np.float32), np.float32(-np.inf)).astype(np.float64),
            np.nextafter(np.asarray(high, dtype=np.float32), np.float32(np.inf)).astype(np.float64))


def _bin_edges(size, other, bin_size):
    # Mirror rasterization_utils.cuh / RasterizeCoarseCudaKernel, using outward
    # intervals rather than assuming the compiler fuses or does not fuse +/*.
    rlow, rhigh = _round_out(2.0 * max(size, other) / other if size > other else 2.0,
                            2.0 * max(size, other) / other if size > other else 2.0)
    olow, ohigh = _round_out(rlow / 2, rhigh / 2)
    hlow, hhigh = _round_out(olow / size, ohigh / size)
    del hlow

    def pixel(i):
        plow, phigh = _round_out(rlow * i, rhigh * i)
        plow, phigh = _round_out(plow + olow, phigh + ohigh)
        plow, phigh = _round_out(plow / size, phigh / size)
        return _round_out(plow - ohigh, phigh - olow)

    index = np.arange((size + bin_size - 1) // bin_size, dtype=np.float64)
    low, _ = pixel(index * bin_size)
    _, high = pixel((index + 1) * bin_size - 1)
    low, _ = _round_out(low - hhigh, low - hhigh)
    _, high = _round_out(high + hhigh, high + hhigh)
    return low, high


def conservative_raster_capacity(vertices_ndc, faces, image_size, *, bin_size=None) -> RasterCapacity:
    """Return safe RasterizationSettings bin_size/M without modifying inputs.

    ``vertices_ndc`` is native projected float32 [V,3] or [B,V,3], retaining
    view-space Z. ``faces`` is the same unchanged [F,3] integer topology for each
    mesh. ``image_size`` is (H,W). No blur, clipping, culling or projection happens
    here. Pass this capacity only for the same vertices, faces and original grid.
    The default bin size equals pinned PyTorch3D's CUDA heuristic exactly.
    """
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be (H,W)")
    height, width = (_size(x, "image dimension") for x in image_size)
    if bin_size is None:
        largest = max(height, width)
        bin_size = 8 if largest <= 64 else 1 << max((largest - 1).bit_length() - 4, 4)
    bin_size = _size(bin_size, "bin_size")
    bh, bw = (height + bin_size - 1) // bin_size, (width + bin_size - 1) // bin_size
    if max(bh, bw) >= 22:
        raise ValueError("Pinned CUDA kernel requires fewer than 22 bins per axis")
    verts, indices = np.asarray(vertices_ndc), np.asarray(faces)
    if verts.ndim == 2:
        verts = verts[None]
    if (verts.dtype != np.float32 or verts.ndim != 3 or verts.shape[0] < 1
            or verts.shape[1] < 3 or verts.shape[2] != 3 or not np.isfinite(verts).all()
            or np.any(verts[..., 2] <= 1e-8)):
        raise ValueError("Exact finite float32 NDC vertices with positive camera Z required")
    if (indices.ndim != 2 or indices.shape[1] != 3 or not len(indices)
            or indices.dtype.kind not in "iu" or len(indices) > np.iinfo(np.int32).max
            or indices.min() < 0 or indices.max() >= verts.shape[1]):
        raise ValueError("Unchanged nonempty valid integer triangle topology required")
    xlow, xhigh = _bin_edges(width, height, bin_size)
    ylow, yhigh = _bin_edges(height, width, bin_size)
    counts = np.empty((len(verts), bh, bw), dtype=np.int64)
    stride, cells = bw + 1, (bh + 1) * (bw + 1)
    for b, vertex in enumerate(verts):
        difference = np.zeros(cells, dtype=np.int64)
        for start in range(0, len(indices), _CHUNK_FACES):
            triangles = vertex[indices[start:start + _CHUNK_FACES], :2]
            low, high = triangles.min(axis=1), triangles.max(axis=1)
            x0 = np.searchsorted(xhigh, low[:, 0], side="left")
            x1 = np.searchsorted(xlow, high[:, 0], side="right")
            y0 = np.searchsorted(yhigh, low[:, 1], side="left")
            y1 = np.searchsorted(ylow, high[:, 1], side="right")
            hit = (x0 < x1) & (y0 < y1)
            x0, x1, y0, y1 = (a[hit] for a in (x0, x1, y0, y1))
            for corners, sign in ((y0 * stride + x0, 1), (y0 * stride + x1, -1),
                                  (y1 * stride + x0, -1), (y1 * stride + x1, 1)):
                difference += sign * np.bincount(corners, minlength=cells)
        counts[b] = difference.reshape(bh + 1, bw + 1).cumsum(0).cumsum(1)[:-1, :-1]
    if np.any(counts < 0) or np.any(counts > len(indices)):
        raise RuntimeError("Internal AABB counting invariant failed")
    capacity = max(1, int(counts.max()))  # Empty projected support still needs valid M.
    counts.flags.writeable = False
    return RasterCapacity(bin_size, capacity, counts, len(indices))

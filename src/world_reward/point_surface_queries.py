"""Canonical witnesses from original reconstructed triangles, not reference truth.

Only192 fixed pixel-centre rays are tested; every original triangle participates.
No mesh repair, GT, sensor calibration, rasterizer or per-record query adjustment.
The caller binds RGB-derived inputs and one clip-constant metric surface/scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .point_pose_cost import GRID_COLUMNS, GRID_ROWS, _bool, _real, _size, select_query_indices

TRIANGLE_CHUNK = 4096
NUMERIC_MARGIN = 64 * np.finfo(np.float64).eps  # Dimensionless conditioning, never a metre epsilon.


@dataclass(frozen=True)
class SurfaceQueries:
    canonical_points: np.ndarray
    query_points: np.ndarray
    grid_indices: np.ndarray
    face_indices: np.ndarray
    camera_depth_m: np.ndarray


def canonical_surface_queries(vertices, faces, R0, t0, K, automatic_mask,
                              inferred_depth_valid, *, image_width, image_height):
    """First32 (>=8) automatic grid hits, before seeing any future track.

    Float32/F64 vertices[V,3], proper R0[3,3]/t0[3], fixed OpenCV K[3,3];
    faces=int64[F,3], masks=bool[H,W]. Rays use (x+.5,y+.5), z-direction1.
    Misses exclude only absent predicted surfaces, never uncertain intersections.
    Any supported-grid near-tangent/edge hit, coplanar ray or unresolved first-
    depth tie fails the whole call. Ordinary noncoplanar parallel faces are misses.
    No backface culling; no whole-mesh embedding/quality claim. Output points
    use original face barycentrics; queries are (0,y+.5,x+.5) on original pixels.
    """
    w, h = _size(image_width, 'width', GRID_COLUMNS), _size(image_height, 'height', GRID_ROWS)
    mask = _bool(automatic_mask, (h, w), 'automatic_mask')
    valid_depth = _bool(inferred_depth_valid, (h, w), 'inferred_depth_valid')
    v, r, t, camera = (_real(x, n) for x, n in ((vertices, 'vertices'), (R0, 'R0'), (t0, 't0'), (K, 'K')))
    f = np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3 or np.ma.isMaskedArray(faces)
            or f.dtype != np.int64 or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError('Original finite Vx3 and valid nonempty int64 Fx3 triangles required')
    if (r.shape != (3, 3) or t.shape != (3,) or not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError('Proper fixed initial SE(3) required')
    if (camera.shape != (3, 3) or camera[0, 0] <= 0 or camera[1, 1] <= 0
            or camera[0, 1] != 0 or camera[1, 0] != 0 or not np.array_equal(camera[2], [0, 0, 1])):
        raise ValueError('Fixed positive-focal zero-skew OpenCV K required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            cam = v @ r.T + t
            if not np.isfinite(cam).all() or np.any(cam[:, 2] <= 0):
                raise ValueError('Every original vertex must be finite and in front; no clipping')
            yy = ((2 * np.arange(GRID_ROWS) + 1) * h) // (2 * GRID_ROWS)
            xx = ((2 * np.arange(GRID_COLUMNS) + 1) * w) // (2 * GRID_COLUMNS)
            y, x = np.meshgrid(yy, xx, indexing='ij')
            grid = np.column_stack((y.ravel(), x.ravel())).astype(np.int64)
            rays = np.column_stack(((grid[:, 1]+.5-camera[0, 2])/camera[0, 0],
                                    (grid[:, 0]+.5-camera[1, 2])/camera[1, 1], np.ones(len(grid))))
            ray_norm = np.linalg.norm(rays, axis=1)
            if not np.isfinite(ray_norm).all():
                raise ValueError('Original rays exceed finite numeric range')
            supported = mask[grid[:, 0], grid[:, 1]] & valid_depth[grid[:, 0], grid[:, 1]]
            best = np.full(len(grid), np.inf); ids = np.full(len(grid), -1, np.int64)
            bary = np.full((len(grid), 3), np.nan)
            for start in range(0, len(f), TRIANGLE_CHUNK):
                tri = cam[f[start:start+TRIANGLE_CHUNK]]
                scale = np.max(np.abs(tri), axis=(1, 2))
                a, b, c = (tri/scale[:, None, None]).transpose(1, 0, 2)
                e1, e2 = b-a, c-a
                normal = np.cross(e1, e2); normal_norm = np.linalg.norm(normal, axis=1)
                if np.any(normal_norm == 0) or not np.isfinite(normal_norm).all():
                    raise ValueError('Every original triangle must be nondegenerate in FP64')
                cross = np.cross(rays[:, None], e2[None])
                determinant = np.einsum('qfi,fi->qf', cross, e1)
                nonparallel = determinant != 0
                q = np.cross(-a, e1)
                u = np.divide(np.einsum('qfi,fi->qf', cross, -a), determinant,
                              out=np.zeros_like(determinant), where=nonparallel)
                s = np.divide(rays @ q.T, determinant, out=np.zeros_like(determinant), where=nonparallel)
                depth = np.divide(np.sum(e2*q, axis=1)[None], determinant,
                                  out=np.zeros_like(determinant), where=nonparallel)*scale[None]
                weights = np.stack((1-u-s, u, s), axis=-1)
                minimum = np.min(weights, axis=-1)
                angle = np.abs(determinant)/(ray_norm[:, None]*normal_norm[None])
                coplanar = ~nonparallel & (np.abs(np.sum(normal*a, axis=1))[None] <= NUMERIC_MARGIN*normal_norm[None])
                nearhit = nonparallel & (depth > 0) & (minimum >= -NUMERIC_MARGIN)
                ambiguous = coplanar | (nearhit & ((minimum <= NUMERIC_MARGIN) | (angle <= NUMERIC_MARGIN)))
                if np.any(ambiguous[supported]):
                    raise ValueError('Supported fixed ray has ambiguous boundary/tangent/coplanar geometry')
                hit = nonparallel & (depth > 0) & (minimum > NUMERIC_MARGIN)
                distances = np.where(hit, depth, np.inf)
                local = np.argmin(distances, axis=1); rows = np.arange(len(grid)); nearest = distances[rows, local]
                tied = np.sum(hit & (np.abs(depth-nearest[:, None]) <= NUMERIC_MARGIN*np.maximum(depth, nearest[:, None])), axis=1) > 1
                oldtie = np.isfinite(best) & np.isfinite(nearest)
                oldtie &= np.abs(np.where(oldtie, nearest, 0)-np.where(oldtie, best, 0)) <= NUMERIC_MARGIN*np.maximum(np.where(oldtie, nearest, 0), np.where(oldtie, best, 0))
                if np.any((tied | oldtie)[supported]):
                    raise ValueError('Supported fixed ray has unresolved nearest-surface depth tie')
                replace = nearest < best
                best[replace] = nearest[replace]; ids[replace] = start+local[replace]; bary[replace] = weights[rows[replace], local[replace]]
            surface = np.zeros((h, w), bool); pointmap = np.full((h, w, 3), np.nan)
            hit = ids >= 0
            surface[grid[hit, 0], grid[hit, 1]] = True
            pointmap[grid[hit, 0], grid[hit, 1]] = rays[hit]*best[hit, None]
            selected = select_query_indices(mask, valid_depth, surface, pointmap)
            lookup = {tuple(pixel): i for i, pixel in enumerate(grid)}
            chosen = np.array([lookup[tuple(pixel)] for pixel in selected], np.int64)
            points = np.sum(v[f[ids[chosen]]]*bary[chosen, :, None], axis=1)
            if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
                raise ValueError('Distinct finite original canonical witnesses required')
            queries = np.column_stack((np.zeros(len(selected)), selected[:, 0]+.5, selected[:, 1]+.5))
    except FloatingPointError as exc:
        raise ValueError('Original ray/triangle arithmetic exceeds finite numeric range') from exc
    arrays = (points, queries, selected.copy(), ids[chosen].copy(), best[chosen].copy())
    for array in arrays:
        array.flags.writeable = False
    return SurfaceQueries(*arrays)

"""Bind existing first-frame surface attachments to unchanged full-T tracks.

No raycast, tracker, refit, future selection, mesh conversion or model/file I/O.
The caller supplies the ACTUAL native F32 mesh in the selector's canonical
frame, not an approximately aligned/quantized replacement. Numeric binding does
not authenticate source references, camera provenance or physical point identity.
"""
from __future__ import annotations

import numpy as np

from .fixed_shape_point_pose import PointTrackEvidence
from .joint_point_objective import FixedTriangleTracks, FRAME_CONVENTION
from .point_pose_cost import MIN_QUERIES, MAX_QUERIES
from .point_surface_queries import SurfaceQueries


def _same(actual, expected, name):
    a = np.asarray(actual)
    if (np.ma.isMaskedArray(actual) or a.dtype != expected.dtype or a.shape != expected.shape
            or not np.array_equal(a, expected)):
        raise ValueError(f'{name}: exact original dtype/order/values required; never matched or refitted')


def bind_joint_point_evidence(queries: SurfaceQueries, tracks: PointTrackEvidence, *,
                             native_vertices, native_faces, K, image_size,
                             frame_index, source_frame_ids, native_frame_names,
                             source_references) -> FixedTriangleTracks:
    """Copy one frozen attachment/track association into the joint-loss contract.

    Original selector arithmetic is EXACT ``sum(V.astype(F64)[F[face_ids]] *
    original_F64_bary[:, :, None], axis=1)``. It must equal the stored canonical
    points bit for bit. The later differentiable loss explicitly casts the SAME
    barycentrics to its Torch compute dtype; this adapter does not round/refit
    them. A different native mesh/frame must create new pre-tracking evidence,
    never a post-tracking attachment fit. Every selected slot/frame is retained,
    including no-visible evidence; positive-weight support gates remain in loss.
    """
    if type(queries) is not SurfaceQueries or type(tracks) is not PointTrackEvidence:
        raise ValueError('Existing typed original SurfaceQueries and PointTrackEvidence required')
    for record in (queries, tracks):
        if any(not isinstance(a, np.ndarray) or np.ma.isMaskedArray(a) or a.flags.writeable
               for a in vars(record).values()):
            raise ValueError('Original attachment/track records must already be frozen before binding')
    if queries.face_indices.ndim != 1:
        raise ValueError('Original one-dimensional attachment face IDs required')
    q = len(queries.face_indices)
    if not MIN_QUERIES <= q <= MAX_QUERIES:
        raise ValueError('All original [8..32] selected query slots required; no refill or cap')
    if queries.barycentric.dtype != np.float64:
        raise ValueError('Original selector F64 barycentrics required; no guessed casting')
    bound = FixedTriangleTracks(native_vertices, native_faces, queries.face_indices, queries.barycentric,
        K, image_size, frame_index, source_frame_ids, tracks.query_ids, tracks.query_points,
        tracks.tracks_256, tracks.native_visible, native_frame_names, source_references, FRAME_CONVENTION)
    _same(tracks.frame_index, bound.frame_index, 'native full timeline')
    _same(tracks.source_frame_ids, bound.source_frame_ids, 'native source-frame association')
    _same(queries.query_points, bound.query_points, 'original initial query coordinates/order')
    h, w = bound.image_size
    grid = queries.grid_indices
    if (grid.dtype != np.int64 or grid.shape != (q, 2) or np.any(grid < 0)
            or np.any(grid[:, 0] >= h) or np.any(grid[:, 1] >= w)):
        raise ValueError('Original in-grid I64 first-frame pixels required')
    if np.any(np.diff(grid[:, 0]*w+grid[:, 1]) <= 0):
        raise ValueError('Original unique raster-order query slots required; never reordered')
    _same(queries.query_points, np.column_stack((np.zeros(q), grid[:, 0]+.5, grid[:, 1]+.5)),
        'original pixel-centre query association')
    depth = queries.camera_depth_m
    if depth.dtype != np.float64 or depth.shape != (q,) or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('Original finite positive F64 ray depths required')
    with np.errstate(over='raise', invalid='raise'):
        points = np.sum(bound.vertices.astype(np.float64)[bound.faces[bound.face_indices]]
            * bound.barycentric[:, :, None], axis=1)
    _same(queries.canonical_points, points, 'native canonical attachment arithmetic')
    return bound

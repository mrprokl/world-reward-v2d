"""Same fixed surface/native25 pool, with versus without automatic RGB tracks.

This extracts composition, not a scorer, model runner, pose fit or empirical gain.
Callbacks provide the original native raster/ICP/track operators. Array identities
bind arithmetic only: they cannot certify sources, model rights or metre gauge.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from typing import Callable, Iterable, Protocol
import weakref

import numpy as np

from .point_candidate_pool import NativeCandidatePool, SURFACE_SAMPLES
from .point_pose_comparison import PointPoseComparison, compare
from .point_surface_queries import SurfaceQueries, canonical_surface_queries


def _copy(value, name, *, dtype=None, shape=None, finite=True):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or (dtype is not None and array.dtype != dtype)
            or (dtype is None and array.dtype not in (np.dtype('float32'), np.dtype('float64')))
            or (shape is not None and array.shape != shape) or (finite and not np.isfinite(array).all())):
        raise ValueError(f'{name}: exact unmasked dtype/grid/support required')
    owned = np.array(array, copy=True, order='C'); owned.flags.writeable = False
    return owned


def _timeline(value, *, source=False):
    array = _copy(value, 'source frame IDs' if source else 'frame_index', dtype=np.int64)
    if (array.ndim != 1 or len(array) < 3 or (source and (np.any(array < 0) or np.any(array[1:] <= array[:-1])))
            or (not source and not np.array_equal(array, np.arange(len(array), dtype=np.int64)))):
        raise ValueError('Full original arange(T), T>=3, and ordered unique source IDs required')
    return array


def _digest(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(a.dtype.str.encode()+b'\0'+str(a.shape).encode()+b'\0')
        digest.update(a.tobytes(order='C'))
    return digest.hexdigest()


def _arrays(record):
    return tuple(getattr(record, f.name) for f in fields(record) if isinstance(getattr(record, f.name), np.ndarray))


def _unchanged(arrays, digest):
    if any(a.flags.writeable for a in arrays) or _digest(*arrays) != digest:
        raise ValueError('Original fixed geometry/observations/queries/tracks/pool changed')


def _alive_unchanged(observations):
    # Released streamed arrays cannot be mutated; never retain all-T pointmaps.
    for refs in observations:
        for ref, digest in refs:
            array = ref()
            if array is not None: _unchanged((array,), digest)


@dataclass(frozen=True, eq=False)
class FixedObjectGeometry:
    """One already scaled native surface; no scale is applied by this operator.

    ``surface_points`` is the original8192 sample and ``mesh_centroid`` the native
    value, not a recomputed vertex mean. ``gauge_convention`` is caller metadata,
    not a calibration claim. Shared RGB-derived K uses OpenCV pixel XY units.
    """
    vertices: np.ndarray
    faces: np.ndarray
    surface_points: np.ndarray
    mesh_centroid: np.ndarray
    R0: np.ndarray
    t0: np.ndarray
    K: np.ndarray
    image_size: tuple[int, int]  # height, width
    gauge_convention: str

    def __post_init__(self):
        if (type(self.image_size) is not tuple or len(self.image_size) != 2
                or any(type(n) is not int or n <= 0 for n in self.image_size)
                or type(self.gauge_convention) is not str or not self.gauge_convention.strip()):
            raise ValueError('Original image_size and explicit shared gauge convention required')
        shapes = dict(surface_points=(SURFACE_SAMPLES, 3), mesh_centroid=(3,), R0=(3, 3), t0=(3,), K=(3, 3))
        for name in ('vertices', 'faces', *shapes):
            object.__setattr__(self, name, _copy(getattr(self, name), name,
                dtype=np.int64 if name == 'faces' else None, shape=shapes.get(name)))
        if self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,) or self.faces.ndim != 2 or self.faces.shape[1:] != (3,):
            raise ValueError('Original Vx3/Fx3 triangular surface required')


@dataclass(frozen=True, eq=False)
class FrameObservation:
    frame_index: int
    source_frame_id: int
    rgb: np.ndarray
    automatic_mask: np.ndarray
    inferred_pointmap: np.ndarray
    inferred_depth_valid: np.ndarray

    def __post_init__(self):
        if type(self.frame_index) is not int or type(self.source_frame_id) is not int:
            raise ValueError('Explicit original integer frame/index required')
        rgb = _copy(self.rgb, 'RGB', dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2:] != (3,): raise ValueError('Original uint8 RGB grid required')
        object.__setattr__(self, 'rgb', rgb); h, w = rgb.shape[:2]
        for name in ('automatic_mask', 'inferred_depth_valid'):
            object.__setattr__(self, name, _copy(getattr(self, name), name, dtype=np.bool_, shape=(h, w)))
        points = _copy(self.inferred_pointmap, 'pointmap', shape=(h, w, 3), finite=False)
        if not np.isfinite(points[self.inferred_depth_valid]).all() or np.any(points[self.inferred_depth_valid, 2] <= 0):
            raise ValueError('Declared inferred-depth support must be finite positive-Z')
        object.__setattr__(self, 'inferred_pointmap', points)


@dataclass(frozen=True, eq=False)
class PointTrackEvidence:
    """Native XY256 and visibility, not confidence or physical point identity."""
    frame_index: np.ndarray
    source_frame_ids: np.ndarray
    query_ids: np.ndarray
    query_points: np.ndarray  # exact original (position0, pixelY+.5, pixelX+.5)
    tracks_256: np.ndarray   # T,Q,XY, native256 grid
    native_visible: np.ndarray

    def __post_init__(self):
        index, source = _timeline(self.frame_index), _timeline(self.source_frame_ids, source=True)
        ids = _copy(self.query_ids, 'original query IDs', dtype=np.int64)
        if source.shape != index.shape or ids.ndim != 1 or not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
            raise ValueError('Exact full-frame/query binding required')
        for name, value in (('frame_index', index), ('source_frame_ids', source), ('query_ids', ids)):
            object.__setattr__(self, name, value)
        for name, dtype, shape in (('query_points', None, (len(ids), 3)), ('tracks_256', None, (len(index), len(ids), 2)),
                                  ('native_visible', np.bool_, (len(index), len(ids)))):
            object.__setattr__(self, name, _copy(getattr(self, name), name, dtype=dtype, shape=shape))


class TrackCallback(Protocol):
    def __call__(self, rgb_frames: tuple[np.ndarray, ...], query_points: np.ndarray, *,
                 frame_index: np.ndarray, source_frame_ids: np.ndarray, query_ids: np.ndarray) -> PointTrackEvidence: ...


@dataclass(frozen=True, eq=False)
class FixedShapePointPoseResult:
    geometry: FixedObjectGeometry
    queries: SurfaceQueries
    tracks: PointTrackEvidence
    comparison: PointPoseComparison
    observation_sha256: tuple[str, ...]  # original ordered RGB/mask/pointmap/support


def compare_fixed_shape_sequence(geometry: FixedObjectGeometry, frame_index, source_frame_ids,
                                 observations: Iterable[FrameObservation], *, raster: Callable,
                                 track: TrackCallback, align: Callable | None = None) -> FixedShapePointPoseResult:
    """Build original native pool BEFORE one RGB-track call, then unchanged A/B.

    A is native image-cost Viterbi; B adds the existing frozen point unary only.
    All-T mask/finite-positive support<40 still FAILS; no occlusion/static rescue.
    No-visible native tracks retain explicit missing evidence and zero added cost.
    RGB tuple lets the callback use its original preprocessing without this module
    resizing, compressing, fitting, sampling a new surface or loading a model.
    """
    if type(geometry) is not FixedObjectGeometry or not callable(track):
        raise ValueError('Fixed shared geometry and native RGB track callback required')
    index, source_ids = _timeline(frame_index), _timeline(source_frame_ids, source=True)
    if source_ids.shape != index.shape: raise ValueError('Source IDs must bind every original frame')
    g = FixedObjectGeometry(**vars(geometry)); geometry_arrays = _arrays(geometry); geometry_digest = _digest(*geometry_arrays)
    h, w = g.image_size
    builder = NativeCandidatePool(index, g.vertices, g.faces, g.surface_points, g.mesh_centroid, g.R0, g.t0, g.K, w, h)
    builder_arrays = tuple(getattr(builder, n) for n in ('frame_index', 'vertices', 'faces', 'surface_points',
        'mesh_centroid', 'initial_rotation', 'initial_translation', 'camera_K'))
    builder_digest = _digest(*builder_arrays)
    videos, observation_refs, observation_hashes = [], [], []; queries = None
    for position, row in enumerate(observations):
        if (type(row) is not FrameObservation or position >= len(index) or row.frame_index != position
                or row.source_frame_id != int(source_ids[position]) or row.rgb.shape != (h, w, 3)):
            raise ValueError('Exactly once ordered original observations/source IDs/grid required')
        arrays = _arrays(row); before = _digest(*arrays)
        if position == 0:
            queries = canonical_surface_queries(g.vertices, g.faces, g.R0, g.t0, g.K, row.automatic_mask,
                row.inferred_depth_valid, image_width=w, image_height=h)
        builder.add_frame(position, row.inferred_pointmap, row.automatic_mask, raster, align=align)
        _unchanged(builder_arrays, builder_digest); _unchanged(arrays, before)
        videos.append(row.rgb); observation_hashes.append(before)
        observation_refs.append(tuple((weakref.ref(a), _digest(a)) for a in arrays))
        del row, arrays
    if len(videos) != len(index): raise ValueError('Complete original frame coverage required before tracking')
    pool = builder.finalize(); pool_arrays = _arrays(pool); pool_digest = _digest(*pool_arrays)
    query_arrays = _arrays(queries); query_digest = _digest(*query_arrays)
    ids = _copy(np.arange(len(queries.query_points), dtype=np.int64), 'query IDs', dtype=np.int64)
    timeline_digest = _digest(index, source_ids, ids)
    _alive_unchanged(observation_refs)
    evidence = track(tuple(videos), queries.query_points, frame_index=index, source_frame_ids=source_ids, query_ids=ids)
    _alive_unchanged(observation_refs)
    if (type(evidence) is not PointTrackEvidence or _digest(evidence.frame_index, evidence.source_frame_ids, evidence.query_ids) != timeline_digest
            or _digest(evidence.query_points) != _digest(queries.query_points)):
        raise ValueError('Native track output changed original frames/queries; no matching or reorder')
    _unchanged(query_arrays, query_digest); _unchanged((index, source_ids, ids), timeline_digest); _unchanged(pool_arrays, pool_digest)
    track_arrays = _arrays(evidence); track_digest = _digest(*track_arrays)
    result = compare(pool, queries.canonical_points, evidence.tracks_256, evidence.native_visible, g.K, w, h)
    _unchanged(track_arrays, track_digest); _unchanged(query_arrays, query_digest); _unchanged(pool_arrays, pool_digest)
    _unchanged(geometry_arrays, geometry_digest); _unchanged(builder_arrays, builder_digest)
    _alive_unchanged(observation_refs)
    return FixedShapePointPoseResult(FixedObjectGeometry(**vars(g)),
        SurfaceQueries(*(_copy(a, 'surface evidence', dtype=a.dtype) for a in query_arrays)),
        PointTrackEvidence(**vars(evidence)), result, tuple(observation_hashes))

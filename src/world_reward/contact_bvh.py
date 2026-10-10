"""Bounded CPU search over every hand point and every original object face.

This is a numerical/proposal primitive, NOT a qualified native CUDA replacement.
The narrow phase follows the pinned PyTorch3D *forward* formula, including its
5e-3 minimum triangle area, 1e-8 regularisation and segment fallback. CPU FP64
is not CUDA FP32/FMA; ties, gradients and an Adam-step parity gate remain open.

Object geometry is clip-constant. A dual AABB search discards only node pairs
whose conservative lower bound exceeds a freshly evaluated witness upper bound.
No hand/face sampling, scale change, static assumption or cached gap is used.
Exhausting an execution budget raises, rather than returning a partial minimum.

Primary source (BSD-style licence; independently expressed NumPy formula):
https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/pytorch3d/csrc/utils/geometry_utils.cuh
https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/pytorch3d/csrc/point_mesh/point_mesh_cuda.cu
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from numbers import Integral, Real
import time

import numpy as np
from scipy.spatial import cKDTree


NATIVE_MIN_TRIANGLE_AREA = 5e-3
NATIVE_GEOMETRY_EPSILON = 1e-8
NATIVE_SQUARED_DISTANCE_FLOOR = 1e-12


class ContactSearchBudgetExceeded(RuntimeError):
    """The complete minimum was not established; no prediction is returned."""


@dataclass(frozen=True)
class ContactSearchLimits:
    """Execution limits, not contact/trajectory hyperparameters."""

    face_leaf_size: int = 32
    point_leaf_size: int = 8
    max_pair_evaluations: int = 2_000_000
    max_node_pairs: int = 100_000
    max_heap_pairs: int = 16_384
    max_query_seconds: float = 30.0
    max_geometry_bytes: int = 1_073_741_824
    max_query_bytes: int = 67_108_864

    def __post_init__(self):
        for name in ('face_leaf_size', 'point_leaf_size', 'max_pair_evaluations',
                     'max_node_pairs', 'max_heap_pairs', 'max_geometry_bytes', 'max_query_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.face_leaf_size > 64 or self.point_leaf_size > 32:
            raise ValueError('Narrow-phase leaves must remain bounded at 64 faces/32 points')
        _positive(self.max_query_seconds, 'max_query_seconds')


@dataclass(frozen=True)
class ContactMinimum:
    squared_distance: float
    distance: float
    point_index: int
    face_index: int
    tied_witness: bool
    floor_ambiguous: bool
    eligible_points: int
    eligible_faces: int
    pair_evaluations: int
    visited_node_pairs: int
    max_heap_pairs_used: int
    seconds: float
    native_parity_verified: bool = False
    whole_input_hand_minimum: bool = True
    search_complete: bool = True
    provenance: str = 'all_provided_hand_points_all_original_faces_CPU_forward_only'


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(name + ' must be finite and ' + ('nonnegative' if zero else 'positive'))
    return float(value)


def _points(value, name):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype.kind not in 'fiu'
            or array.ndim != 2 or array.shape[1] != 3 or len(array) == 0):
        raise ValueError(name + ' must contain finite nonempty [N,3] points')
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(name + ' contains nonfinite coordinates')
    if np.abs(array).max() > np.sqrt(np.finfo(np.float64).max) / 32:
        raise ValueError(name + ' exceeds safely bounded squared-coordinate range')
    return array


def _triangle_terms(triangles):
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e2, e1)
    normal_norm = np.sqrt(np.sum(cross * cross, axis=-1))
    # AreaOfTriangle uses nested hypot, not the normal-norm expression.
    area = np.hypot(cross[:, 0], np.hypot(cross[:, 1], cross[:, 2])) * .5
    normal = cross / (normal_norm[:, None] + NATIVE_GEOMETRY_EPSILON)
    d00, d01, d11 = (np.sum(e1 * e1, -1), np.sum(e1 * e2, -1), np.sum(e2 * e2, -1))
    determinant = d00 * d11 - d01 * d01
    denominator = determinant + NATIVE_GEOMETRY_EPSILON
    return e1, e2, normal_norm, area, normal, d00, d01, d11, determinant, denominator


def _squared_pairs(points, triangles, min_triangle_area):
    """Small Cartesian leaf only; never called with a full hand/full mesh."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        e1, e2, norm, area, normal, d00, d01, d11, _, denominator = _triangle_terms(triangles)
        p, a = points[:, None, :], triangles[None, :, 0]
        t = np.sum((a - p) * normal[None], axis=-1)
        projected = p + t[..., None] * normal[None]
        delta = projected - a
        d20, d21 = np.sum(delta * e1[None], -1), np.sum(delta * e2[None], -1)
        w1 = (d11[None] * d20 - d01[None] * d21) / denominator[None]
        w2 = (d00[None] * d21 - d01[None] * d20) / denominator[None]
        w0 = 1.0 - w1 - w2
        interior = ((area[None] >= min_triangle_area) & (norm[None] > NATIVE_GEOMETRY_EPSILON)
                    & (w0 >= 0) & (w0 <= 1) & (w1 >= 0) & (w1 <= 1)
                    & (w2 >= 0) & (w2 <= 1))
        edges = []
        for first, last in ((0, 1), (0, 2), (1, 2)):
            start = triangles[None, :, first]
            edge = triangles[None, :, last] - start
            length2 = np.sum(edge * edge, -1)
            numerator = np.sum((p - start) * edge, -1)
            parameter = np.divide(numerator, length2, out=np.zeros_like(numerator),
                                  where=length2 >= NATIVE_GEOMETRY_EPSILON)
            parameter = np.clip(parameter, 0, 1)
            difference = p - (start + parameter[..., None] * edge)
            edges.append(np.sum(difference * difference, -1))
        result = np.where(interior, t * t, np.minimum(np.minimum(edges[0], edges[1]), edges[2]))
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Native forward formula exceeded finite numerical range')
    return result


def native_point_triangle_squared_distance(point, triangle, *, min_triangle_area=NATIVE_MIN_TRIANGLE_AREA):
    """Scalar CPU reference, source semantics; no CUDA bit/gradient parity claim."""
    area = _positive(min_triangle_area, 'min_triangle_area', zero=True)
    p = np.asarray(point)
    tri = np.asarray(triangle)
    if p.shape != (3,) or tri.shape != (3, 3):
        raise ValueError('A [3] point and [3,3] triangle are required')
    p, tri = _points(p[None], 'point'), _points(tri, 'triangle')
    return float(_squared_pairs(p, tri[None], area)[0, 0])


class _AABBTree:
    """Balanced, immutable index tree; original primitive IDs never change."""

    def __init__(self, lower, upper, scales, leaf_size):
        self.order = np.arange(len(lower), dtype=np.int64)
        centers = (lower + upper) * .5
        nodes = []

        def build(start, end):
            ids = self.order[start:end]
            node = len(nodes)
            nodes.append([lower[ids].min(0), upper[ids].max(0), float(scales[ids].min()),
                          start, end, -1, -1])
            if end - start > leaf_size:
                extent = np.ptp(centers[ids], axis=0)
                axis = int(np.argmax(extent))
                middle = (end - start) // 2
                partition = np.argpartition(centers[ids, axis], middle)
                self.order[start:end] = ids[partition]
                left = build(start, start + middle)
                right = build(start + middle, end)
                nodes[node][5:] = [left, right]
            return node

        build(0, len(lower))
        self.lower = np.asarray([n[0] for n in nodes])
        self.upper = np.asarray([n[1] for n in nodes])
        self.scale = np.asarray([n[2] for n in nodes])
        self.start = np.asarray([n[3] for n in nodes], dtype=np.int64)
        self.end = np.asarray([n[4] for n in nodes], dtype=np.int64)
        self.left = np.asarray([n[5] for n in nodes], dtype=np.int64)
        self.right = np.asarray([n[6] for n in nodes], dtype=np.int64)
        for value in vars(self).values():
            value.setflags(write=False)


class NativeContactBVH:
    """One fixed original object; query one automatically supplied hand at a time.

    ``minimum(points, seed=(point_id, face_id))`` searches ALL input points/faces.
    A seed is only an upper bound and is recomputed on the current geometry.
    A caller supplying a sampled hand must not relabel this as a full-hand min.
    The output witness cannot be passed off as a native gradient without a
    separate actual CUDA forward/tie/backward/Adam qualification.
    """

    def __init__(self, vertices, faces, *, min_triangle_area=NATIVE_MIN_TRIANGLE_AREA,
                 limits=None):
        self.limits = ContactSearchLimits() if limits is None else limits
        if type(self.limits) is not ContactSearchLimits:
            raise ValueError('Explicit ContactSearchLimits required')
        self.min_triangle_area = _positive(min_triangle_area, 'min_triangle_area', zero=True)
        self.vertices = _points(vertices, 'vertices')
        f = np.asarray(faces)
        if (np.ma.isMaskedArray(faces) or f.dtype.kind not in 'iu' or f.ndim != 2
                or f.shape[1] != 3 or len(f) == 0 or np.any(f < 0)
                or np.any(f >= len(self.vertices))):
            raise ValueError('Nonempty valid integer [F,3] original faces required')
        # Conservative planned peak, including temporary terms/tree construction.
        # This is an allocation gate, not an assertion about allocator/RSS bytes.
        estimate = 768 * len(f) + 128 * len(self.vertices)
        if estimate > self.limits.max_geometry_bytes:
            raise ContactSearchBudgetExceeded('Geometry allocation bound exceeded')
        self.faces = np.array(f, dtype=np.int64, copy=True)
        self.triangles = self.vertices[self.faces]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            e1, e2, norm, area, _, _, _, _, determinant, denominator = _triangle_terms(self.triangles)
            interior_possible = (area >= self.min_triangle_area) & (norm > NATIVE_GEOMETRY_EPSILON)
            if np.any(interior_possible & (determinant <= 0)):
                raise ValueError('Interior triangle determinant is numerically unresolved')
            # Native barycentric +epsilon enlarges the accepted planar triangle.
            # Native normal +epsilon then scales interior squared distance by alpha².
            expansion = np.ones(len(f))
            expansion[interior_possible] = denominator[interior_possible] / determinant[interior_possible]
            virtual = np.stack((self.triangles[:, 0], self.triangles[:, 0] + e1 * expansion[:, None],
                                self.triangles[:, 0] + e2 * expansion[:, None]), axis=1)
            lower, upper = virtual.min(1), virtual.max(1)
            scales = np.ones(len(f))
            scales[interior_possible] = (norm[interior_possible]
                                        / (norm[interior_possible] + NATIVE_GEOMETRY_EPSILON)) ** 2
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(scales).all():
            raise ValueError('Geometry bounds exceeded finite numerical range')
        self._tree = _AABBTree(lower, upper, scales, self.limits.face_leaf_size)
        incident = np.full(len(self.vertices), len(f), dtype=np.int64)
        np.minimum.at(incident, self.faces.reshape(-1), np.repeat(np.arange(len(f)), 3))
        self._referenced = np.flatnonzero(incident < len(f))
        self._incident = incident[self._referenced]
        self._vertex_tree = cKDTree(self.vertices[self._referenced], copy_data=True)
        for array in (self.vertices, self.faces, self.triangles, self._referenced, self._incident):
            array.setflags(write=False)

    def minimum(self, points, *, seed=None):
        """Return a complete CPU-semantic minimum or fail closed on the budget."""
        started = time.monotonic()
        p = _points(points, 'hand points')
        if 512 * len(p) > self.limits.max_query_bytes:
            raise ContactSearchBudgetExceeded('Query allocation bound exceeded')
        if seed is not None:
            if (not isinstance(seed, (tuple, list)) or len(seed) != 2
                    or any(isinstance(i, bool) or not isinstance(i, Integral) for i in seed)
                    or not 0 <= seed[0] < len(p) or not 0 <= seed[1] < len(self.faces)):
                raise ValueError('Seed must name a current original (point,face) pair')
            seed = tuple(map(int, seed))
        nearest_distances, nearest_ids = self._vertex_tree.query(p, k=1)
        if not np.isfinite(nearest_distances).all():
            raise ValueError('Seed distance exceeded finite numerical range')
        seed_point = int(np.argmin(nearest_distances))
        pairs = [(seed_point, int(self._incident[int(nearest_ids[seed_point])]))]
        if seed is not None and seed != pairs[0]:
            pairs.append(seed)
        if len(pairs) > self.limits.max_pair_evaluations:
            raise ContactSearchBudgetExceeded('Initial witness budget exceeded')
        best, winner, tied, evaluations = np.inf, None, False, 0

        def update(distance, point_id, face_id):
            nonlocal best, winner, tied
            pair = (int(point_id), int(face_id))
            if distance < best:
                best, winner, tied = float(distance), pair, False
            elif distance == best and pair != winner:
                tied = True
                # Deterministic CPU tie rule is NOT the native128-thread rule.
                if pair < winner:
                    winner = pair

        for point_id, face_id in pairs:
            d = _squared_pairs(p[point_id:point_id + 1], self.triangles[face_id:face_id + 1],
                               self.min_triangle_area)[0, 0]
            update(d, point_id, face_id)
            evaluations += 1
        point_tree = _AABBTree(p, p, np.ones(len(p)), self.limits.point_leaf_size)
        face_tree = self._tree
        coordinate_scale = max(1., float(np.abs(p).max()), float(np.abs(face_tree.lower).max()),
                               float(np.abs(face_tree.upper).max()))
        with np.errstate(over='ignore', invalid='ignore'):
            guard = 128. * np.finfo(np.float64).eps * coordinate_scale ** 2
        if not np.isfinite(guard):
            raise ValueError('Conservative arithmetic guard is nonfinite')

        def bound(point_node, face_node):
            difference = np.maximum(0., np.maximum(point_tree.lower[point_node] - face_tree.upper[face_node],
                                                  face_tree.lower[face_node] - point_tree.upper[point_node]))
            squared = float(np.dot(difference, difference)) * face_tree.scale[face_node]
            return max(0., squared - guard)

        heap = [(bound(0, 0), 0, 0)]
        visits, max_heap = 0, 1
        while heap:
            lower_bound, pn, fn = heapq.heappop(heap)
            if lower_bound > best:
                continue
            visits += 1
            if visits > self.limits.max_node_pairs:
                raise ContactSearchBudgetExceeded('Node-pair budget exceeded; minimum unresolved')
            if time.monotonic() - started > self.limits.max_query_seconds:
                raise ContactSearchBudgetExceeded('Wall-time budget exceeded; minimum unresolved')
            point_leaf, face_leaf = point_tree.left[pn] < 0, face_tree.left[fn] < 0
            if point_leaf and face_leaf:
                pi = point_tree.order[point_tree.start[pn]:point_tree.end[pn]]
                fi = face_tree.order[face_tree.start[fn]:face_tree.end[fn]]
                cost = len(pi) * len(fi)
                if evaluations + cost > self.limits.max_pair_evaluations:
                    raise ContactSearchBudgetExceeded('Exact-pair budget exceeded; minimum unresolved')
                distances = _squared_pairs(p[pi], self.triangles[fi], self.min_triangle_area)
                evaluations += cost
                candidate = float(distances.min())
                if candidate <= best:
                    for row, column in np.argwhere(distances == candidate):
                        update(candidate, pi[row], fi[column])
                continue
            # Split the spatially wider nonleaf; no candidate truncation.
            p_width = float(np.sum((point_tree.upper[pn] - point_tree.lower[pn]) ** 2))
            f_width = float(np.sum((face_tree.upper[fn] - face_tree.lower[fn]) ** 2))
            if not point_leaf and (face_leaf or p_width >= f_width):
                children = [(int(point_tree.left[pn]), fn), (int(point_tree.right[pn]), fn)]
            else:
                children = [(pn, int(face_tree.left[fn])), (pn, int(face_tree.right[fn]))]
            for child_p, child_f in children:
                child_bound = bound(child_p, child_f)
                if child_bound <= best:
                    if len(heap) >= self.limits.max_heap_pairs:
                        raise ContactSearchBudgetExceeded('Heap budget exceeded; minimum unresolved')
                    heapq.heappush(heap, (child_bound, child_p, child_f))
            max_heap = max(max_heap, len(heap))
        return ContactMinimum(best, float(np.sqrt(max(best, NATIVE_SQUARED_DISTANCE_FLOOR))),
                              winner[0], winner[1], tied, best <= NATIVE_SQUARED_DISTANCE_FLOOR,
                              len(p), len(self.faces), evaluations, visits, max_heap,
                              time.monotonic() - started)

"""Own experimental continuous point/triangle geometry, not an adopted fitter.

Squared unsigned distances in the supplied metric frame, observed-to-surface
only. No geometry rescaling, epsilon-regularized containment, hidden-surface
attraction or vendor dependency. Float64 arithmetic reduces small-triangle
roundoff; it cannot recover coordinates already lost in input float32.
Closest-face/region ties are nondifferentiable. First-order autograd follows
the selected face; no differentiability of discrete assignments is asserted.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _chunk_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("point_chunk_size must be a positive integer")
    return int(value)


def validate_numpy_geometry(points, triangles, *, point_chunk_size=64):
    """Tiny reference/input validation; returns independent float64 copies."""
    _chunk_size(point_chunk_size)
    arrays = []
    for value, name, rank, suffix in ((points, "points", 2, (3,)), (triangles, "triangles", 3, (3, 3))):
        array = np.asarray(value)
        if (np.ma.isMaskedArray(value) or array.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or array.ndim != rank or array.shape[1:] != suffix or not len(array) or not np.isfinite(array).all()):
            raise ValueError(f"{name} requires finite nonempty float32/64 geometry")
        arrays.append(array.astype(np.float64, copy=True))
    if np.asarray(points).dtype != np.asarray(triangles).dtype:
        raise ValueError("Points and triangles must have the same input dtype")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        edges = arrays[1] - np.roll(arrays[1], 1, axis=1)
        lengths = np.sum(edges * edges, axis=-1)
        normal = np.cross(arrays[1][:, 1] - arrays[1][:, 0], arrays[1][:, 2] - arrays[1][:, 0])
        areas = np.sum(normal * normal, axis=-1)
    if (not np.isfinite(lengths).all() or (lengths <= 0).any()
            or not np.isfinite(areas).all() or (areas <= 0).any()):
        raise ValueError("Every triangle needs nonzero edges/area representable in float64; no repair")
    return tuple(arrays)


def numpy_reference_distance_squared(points, triangles, *, point_chunk_size=64):
    """Small independent NumPy reference, not a production CPU fallback.

    Enumerate the planar projection if contained, and the three closed segment
    projections. Their minimum is the closest point on each closed triangle.
    """
    p, faces = validate_numpy_geometry(points, triangles, point_chunk_size=point_chunk_size)
    output = []
    for point in p:
        best = float("inf")
        for a, b, c in faces:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                normal = np.cross(b - a, c - a)
                norm2 = normal @ normal
                height = (point - a) @ normal
                projection = point - normal * (height / norm2)
                sides = [np.dot(np.cross(end - start, projection - start), normal)
                         for start, end in ((a, b), (b, c), (c, a))]
                if not np.isfinite(sides).all() or not np.isfinite(projection).all():
                    raise ValueError("Triangle containment exceeds finite float64 numeric range")
                values = [height**2 / norm2] if min(sides) >= 0 else []
                for start, end in ((a, b), (b, c), (c, a)):
                    edge = end - start
                    parameter = np.clip((point - start) @ edge / (edge @ edge), 0., 1.)
                    delta = point - (start + parameter * edge)
                    values.append(delta @ delta)
            if not np.isfinite(values).all():
                raise ValueError("Geometry distances exceed finite float64 numeric range")
            best = min(best, *values)
        output.append(best)
    return np.array(output, dtype=np.float64)


def _torch_distances(torch, points, triangles):
    """Broadcast leading dimensions, or paired points/faces after selection."""
    a, b, c = triangles.unbind(dim=-2)
    ab, ac, delta = b - a, c - a, points - a
    normal = torch.cross(ab, ac, dim=-1)
    norm2 = (normal * normal).sum(dim=-1)
    height = (delta * normal).sum(dim=-1)
    # Signed areas use cross products, not a cancellation-prone Gram determinant.
    v = (torch.cross(delta, ac, dim=-1) * normal).sum(dim=-1) / norm2
    w = (torch.cross(ab, delta, dim=-1) * normal).sum(dim=-1) / norm2
    if not torch.isfinite(v).all() or not torch.isfinite(w).all() or not torch.isfinite(height).all():
        raise ValueError("Triangle containment exceeds finite float64 numeric range")
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    segments = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        parameter = ((points - start) * edge).sum(dim=-1) / (edge * edge).sum(dim=-1)
        closest = start + parameter.clamp(0., 1.)[..., None] * edge
        difference = points - closest
        segments.append((difference * difference).sum(dim=-1))
    edges = torch.stack(segments, dim=-1).min(dim=-1).values
    plane = height.square() / norm2
    if not torch.isfinite(plane).all() or not torch.isfinite(edges).all():
        raise ValueError("Triangle projections exceed finite float64 numeric range")
    return torch.where(inside, plane, edges)


def observed_to_triangle_distance_squared(points, triangles, *, point_chunk_size=64):
    """CPU/CUDA [P] float64 distances, same device; original input autograd.

    Inputs must be same-device/dtype float32 or float64 [P,3]/[F,3,3] tensors.
    Search uses no-grad point chunks: O(chunk*F) workspace, O(P) assignments.
    Gather and recompute only the winning faces with autograd. Lowest face
    index wins exact ties. No implicit CUDA/CPU transfer or precision downcast.
    """
    chunk = _chunk_size(point_chunk_size)
    import torch
    for value, rank, suffix in ((points, 2, (3,)), (triangles, 3, (3, 3))):
        if (not torch.is_tensor(value) or value.dtype not in (torch.float32, torch.float64)
                or value.ndim != rank or tuple(value.shape[1:]) != suffix or not len(value)
                or value.device.type not in ("cpu", "cuda") or not torch.isfinite(value).all()):
            raise ValueError("Require finite nonempty CPU/CUDA float32/64 point and triangle tensors")
    if points.device != triangles.device or points.dtype != triangles.dtype:
        raise ValueError("Points and triangles must have identical dtype/device")
    p, faces = points.to(dtype=torch.float64), triangles.to(dtype=torch.float64)
    with torch.no_grad():
        edges = faces - faces.roll(1, dims=1)
        lengths = (edges * edges).sum(dim=-1)
        normal = torch.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0], dim=-1)
        areas = (normal * normal).sum(dim=-1)
        if (not torch.isfinite(lengths).all() or (lengths <= 0).any()
                or not torch.isfinite(areas).all() or (areas <= 0).any()):
            raise ValueError("Every triangle needs nonzero edges/area representable in float64; no repair")
        selected = []
        for start in range(0, len(p), chunk):
            distances = _torch_distances(torch, p[start:start + chunk, None], faces[None])
            if not torch.isfinite(distances).all() or (distances < 0).any():
                raise ValueError("Geometry distances exceed finite float64 numeric range")
            selected.append(distances.argmin(dim=1))
            del distances
        indices = torch.cat(selected)
    distance = _torch_distances(torch, p, faces[indices])
    if not torch.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Selected surface distances exceed finite float64 numeric range")
    return distance

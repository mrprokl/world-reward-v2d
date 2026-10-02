"""Candidate one-sided surface distance; lazy CUDA autograd, no pose/shape fit.

Inputs already share a metric coordinate frame. No camera translation, scale,
nearest-vertex approximation or reverse hidden-face attraction is introduced.
Zero-area triangles fail rather than becoming lines. PyTorch3D's default area
threshold is deliberately overridden to zero for valid small metric triangles.
The analytic CUDA gate must pass before adoption: the pinned kernel also uses
regularized barycentric containment, which can misclassify small triangles.
"""

from __future__ import annotations

import platform

import numpy as np


def validate_numpy_point_triangles(points, triangles):
    """Data-free input guard returning independent float32-safe copies."""
    converted = []
    for value, name, rank, suffix in ((points, "points", 2, (3,)), (triangles, "triangles", 3, (3, 3))):
        array = np.asarray(value)
        if (np.ma.isMaskedArray(value) or array.dtype.kind != "f" or array.ndim != rank
                or array.shape[1:] != suffix or not len(array) or not np.isfinite(array).all()):
            raise ValueError(f"{name} must be finite nonempty floating [{','.join(['N', *map(str, suffix)])}]")
        with np.errstate(over="ignore", invalid="ignore"):
            cast = array.astype(np.float32, copy=True)
        if not np.isfinite(cast).all():
            raise ValueError(f"{name} cannot be represented as finite float32")
        converted.append(cast)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normals = np.cross(converted[1][:, 1] - converted[1][:, 0], converted[1][:, 2] - converted[1][:, 0])
        area_squared = np.sum(normals * normals, axis=1)
    if not np.isfinite(area_squared).all() or (area_squared <= 0).any():
        raise ValueError("Every triangle must retain finite positive area in float32")
    return tuple(converted)


def observed_to_triangle_distance_squared(points, triangles):
    """Return CUDA float32 [P] squared metres while retaining input autograd.

    Direct ``point_face_distance`` is one-sided; do not replace with the
    high-level symmetric point/mesh loss. Points/triangles must be floating
    CUDA tensors on exactly the same device. Only first-order gradients are
    supported by the pinned PyTorch3D operation. No tensor-to-NumPy detachment.
    """
    if platform.system() != "Linux":
        raise RuntimeError("Continuous surface distance requires Azure Linux CUDA")
    import torch
    import pytorch3d
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require CUDA and audited PyTorch3D 0.7.9")
    for value, name, rank, suffix in ((points, "points", 2, (3,)), (triangles, "triangles", 3, (3, 3))):
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or not value.is_cuda
                or value.ndim != rank or tuple(value.shape[1:]) != suffix or len(value) < 1
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite nonempty CUDA float32 tensors")
    if points.device != triangles.device:
        raise ValueError("Points and triangles must be on the same CUDA device")
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    area_squared = torch.sum(normals * normals, dim=-1)
    if not torch.isfinite(area_squared).all() or (area_squared <= 0).any():
        raise ValueError("Every triangle must retain finite positive area in float32")
    first = torch.zeros(1, device=points.device, dtype=torch.int64)
    distance = point_face_distance(points.contiguous(), first, triangles.contiguous(), first, len(points), 0.)
    if distance.shape != (len(points),) or not torch.isfinite(distance).all() or (distance < 0).any():
        raise RuntimeError("Continuous point-to-face output is invalid")
    return distance

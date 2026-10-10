"""CUDA-only hard silhouettes and camera-Z depth, without OpenGL/EGL.

Inputs are already in one OpenCV camera frame: X right, Y down, Z forward,
vertices in metres. Camera extrinsics are identity; do not add body translation
again or flip axes before calling. Intrinsics belong to the original image.
No resize, unit inference, alignment, mesh deletion or challenge labels occur.

PyTorch3D 0.7.9's MeshRasterizer retains view-space Z, despite its Fragments
docstring saying "NDC z". Perspective-correct barycentrics therefore give
linear camera Z in metres, not reciprocal depth or Euclidean ray distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
from pathlib import Path
import platform
import time

import numpy as np


PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"


def _finite_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be real, numeric and finite")
    return array.astype(np.float64, copy=False)


def _intrinsics(camera_matrix) -> np.ndarray:
    matrix = _finite_array(camera_matrix, "camera_matrix")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not np.array_equal(matrix[2], [0, 0, 1])):
        raise ValueError("Require positive focal lengths and zero-skew pinhole intrinsics")
    return matrix


def _points(points_camera_m) -> np.ndarray:
    points = _finite_array(points_camera_m, "points_camera_m")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points_camera_m must have nonempty shape (N, 3)")
    if (points[:, 2] <= 0).any():
        raise ValueError("Every point must have positive camera Z; no points are silently removed")
    return points


def project_camera_points(points_camera_m, camera_matrix) -> np.ndarray:
    """Return continuous OpenCV pixel coordinates (u, v), without any resizing.

    The returned coordinates obey u=fx*X/Z+cx and v=fy*Y/Z+cy. Points behind
    the camera, nonfinite arrays and skew/projective intrinsics fail explicitly.
    Numerical values cannot prove physical units; the caller must establish
    that vertices and any upstream translations use the same metre scale.
    """
    points, matrix = _points(points_camera_m), _intrinsics(camera_matrix)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = points[:, :2] / points[:, 2, None]
        pixels = pixels * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]
    if not np.isfinite(pixels).all():
        raise ValueError("Pinhole projection overflowed")
    return pixels


def silhouette_iou(predicted_mask, observed_mask) -> float:
    """Hard-mask IoU; empty unions are invalid rather than awarded a perfect score.

    This is an image-evidence diagnostic, never a held-out 3D challenge metric.
    Convert automatic PNG masks to bool explicitly before calling.
    """
    predicted, observed = np.asarray(predicted_mask), np.asarray(observed_mask)
    if (predicted.ndim != 2 or min(predicted.shape, default=0) == 0
            or predicted.shape != observed.shape
            or predicted.dtype != np.bool_ or observed.dtype != np.bool_):
        raise ValueError("IoU requires two nonempty, same-shape 2D boolean arrays")
    union = np.count_nonzero(predicted | observed)
    if not union:
        raise ValueError("IoU is undefined for an empty union")
    return float(np.count_nonzero(predicted & observed) / union)


def _mesh_inputs(vertices_camera_m, faces, camera_matrix, width, height, near_clip_m):
    vertices, matrix = _points(vertices_camera_m), _intrinsics(camera_matrix)
    if len(vertices) < 3:
        raise ValueError("A mesh requires at least three vertices")
    for name, size in (("width", width), ("height", height)):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError(f"{name} must be a positive integer; no implicit resize")
    if (isinstance(near_clip_m, (bool, np.bool_)) or not np.isscalar(near_clip_m)
            or not isinstance(near_clip_m, (int, float, np.integer, np.floating))
            or not np.isfinite(near_clip_m) or near_clip_m <= 0):
        raise ValueError("near_clip_m must be a finite positive scalar")
    if (vertices[:, 2] <= near_clip_m).any():
        raise ValueError("Mesh crosses the near plane; fail instead of clipping or deleting faces")
    indices = np.asarray(faces)
    if (indices.ndim != 2 or indices.shape[1] != 3 or not len(indices)
            or indices.dtype.kind not in "iu"):
        raise ValueError("faces must be a nonempty (F, 3) integer array")
    if (indices >= len(vertices)).any() or (indices < 0).any():
        raise ValueError("faces contain invalid vertex indices")
    if np.any(indices[:, 0] == indices[:, 1]) or np.any(indices[:, 0] == indices[:, 2]) or np.any(indices[:, 1] == indices[:, 2]):
        raise ValueError("faces contain repeated vertex indices")
    triangles = vertices[indices]
    with np.errstate(over="ignore", invalid="ignore"):
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.isfinite(normals).all() or np.any(np.all(normals == 0, axis=1)):
        raise ValueError("faces contain zero-area or numerically invalid triangles")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices32, matrix32 = vertices.astype(np.float32), matrix.astype(np.float32)
    if (not np.isfinite(vertices32).all() or not np.isfinite(matrix32).all()
            or (vertices32[:, 2] <= near_clip_m).any()
            or matrix32[0, 0] <= 0 or matrix32[1, 1] <= 0):
        raise ValueError("Inputs cannot be represented safely in the CUDA float32 rasterizer")
    # Detect triangles which would collapse only after float32 conversion.
    triangles32 = vertices32[indices]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normals32 = np.cross(triangles32[:, 1] - triangles32[:, 0], triangles32[:, 2] - triangles32[:, 0])
    if not np.isfinite(normals32).all() or np.any(np.all(normals32 == 0, axis=1)):
        raise ValueError("Triangles collapse or overflow in CUDA float32")
    return vertices32, indices.astype(np.int64, copy=False), matrix32


def _opencv_camera(torch, camera_matrix, width, height, *, batch_size=1):
    from pytorch3d.utils import cameras_from_opencv_projection

    return cameras_from_opencv_projection(
        R=torch.eye(3, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1),
        tvec=torch.zeros((batch_size, 3), device="cuda", dtype=torch.float32),
        camera_matrix=torch.as_tensor(camera_matrix, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1),
        image_size=torch.tensor([[height, width]], device="cuda", dtype=torch.float32).expand(batch_size, -1),
    )



def _checked_pin(path, expected=None, *, limit=None):
    """Tiny sealed receipts or explicit Azure-only geometry, never symlinks."""
    path = Path(path)
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Canonical non-symlink artifact required")
    s = path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_mode & 0o222:
        raise ValueError("Sealed single-link regular artifact required")
    if limit is not None and s.st_size > limit:
        raise ValueError("Receipt exceeds bounded size")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    pin = dict(bytes=s.st_size, sha256=h.hexdigest())
    if expected is not None and pin != expected:
        raise ValueError("Exact artifact pin differs")
    return pin


def _raster_source_pins():
    from world_reward import raster_capacity
    return {"infra/camera_render.py": {"bytes": Path(__file__).stat().st_size,
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "src/world_reward/raster_capacity.py": {
                "bytes": Path(raster_capacity.__file__).stat().st_size,
                "sha256": hashlib.sha256(Path(raster_capacity.__file__).read_bytes()).hexdigest()}}


def _optimized_capacity(full_face_capacity):
    """No silent adoption: absent an approved real gate keep historical M=F.

    False is explicit experimental opt-in (the gate itself); True always forces
    the reference. Default None consumes a sealed, same-source PASS receipt only
    when WR_RASTER_CAPACITY_GATE explicitly names it in the offline container.
    """
    if full_face_capacity is not None:
        if type(full_face_capacity) is not bool:
            raise ValueError("full_face_capacity must be bool or None")
        return not full_face_capacity
    path = os.environ.get("WR_RASTER_CAPACITY_GATE")
    if not path:
        return False
    expected_bytes = os.environ.get("WR_RASTER_CAPACITY_GATE_BYTES", "")
    expected_sha = os.environ.get("WR_RASTER_CAPACITY_GATE_SHA256", "")
    if not expected_bytes.isdecimal() or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("Explicit gate receipt byte/SHA256 pin required")
    _checked_pin(path, dict(bytes=int(expected_bytes), sha256=expected_sha), limit=131072)
    receipt = json.loads(Path(path).read_bytes())
    if (type(receipt) is not dict or receipt.get("schema") != "world_reward.raster_capacity_gate.v1"
            or receipt.get("status") != "pass" or receipt.get("source_pins") != _raster_source_pins()
            or receipt.get("expected_pytorch3d_revision") != PYTORCH3D_REVISION
            or receipt.get("image_id") != os.environ.get("WR_IMAGE_ID")
            or type(receipt.get("gates")) is not dict
            or receipt.get("gates", {}).get("all_exact_masks_and_depth_tolerance") is not True
            or receipt.get("gates", {}).get("external_batch_sizes") != [1, 4]):
        raise ValueError("Same-source/image sealed real raster gate required")
    return True


def _raster_native(mesh, camera, image_size, face_count, *, full_face_capacity,
                   capacity_diagnostics=None):
    """Native projection once; only allocation capacity differs from reference."""
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.renderer.mesh import rasterize_meshes
    options = dict(image_size=image_size, blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, clip_barycentric_coords=False,
        cull_backfaces=False, cull_to_frustum=False, z_clip_value=None)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=RasterizationSettings(
        **options, max_faces_per_bin=face_count))
    optimized = _optimized_capacity(full_face_capacity)
    if not optimized:
        fragments = rasterizer(mesh)
        if capacity_diagnostics is not None:
            capacity_diagnostics.update(mode="full_face_reference", max_faces_per_bin=face_count)
        return fragments.pix_to_face, fragments.zbuf
    from world_reward.raster_capacity import conservative_raster_capacity
    # Do not independently reconstruct NDC from K: native matrix/rounding/Z
    # semantics must be identical to MeshRasterizer.forward's own transform.
    if camera.get_znear() is not None:
        raise ValueError("OpenCV native camera must not activate implicit face clipping")
    projected = rasterizer.transform(mesh)
    capacity = conservative_raster_capacity(
        projected.verts_padded().detach().cpu().numpy(),
        mesh.faces_list()[0].detach().cpu().numpy(), image_size)
    if capacity_diagnostics is not None:
        capacity_diagnostics.update(mode="conservative_AABB_bound", bin_size=capacity.bin_size,
            max_faces_per_bin=capacity.max_faces_per_bin, bin_storage_bytes=capacity.bin_storage_bytes,
            full_face_count=face_count, native_NDC_transform=True)
    pix, zbuf, _, _ = rasterize_meshes(projected, **options, bin_size=capacity.bin_size,
        max_faces_per_bin=capacity.max_faces_per_bin)
    return pix, zbuf

def raster_camera_mesh(vertices_camera_m, faces, camera_matrix, width=1536, height=1152,
                       *, near_clip_m=1e-4, full_face_capacity=None, capacity_diagnostics=None):
    """Return CUDA tensors (hard_mask, depth_m), both original-resolution (H, W).

    hard_mask is bool; depth_m is float32 camera Z with NaN outside the mesh.
    CUDA and Linux are mandatory; imports are lazy so numpy contracts remain
    testable locally. All input triangles are retained. Near-plane crossings
    fail explicitly, while off-image geometry is naturally not rasterized.
    Non-square NDC conversion is delegated to PyTorch3D's OpenCV utility.
    No ad-hoc axis flip, aspect scaling, metric normalization or resizing occurs.
    This inference/QA rasterizer is not a differentiable fitting-loss wrapper.
    Allocation optimization is opt-in until a pinned same-source CUDA gate passes;
    full_face_capacity=True always retains the historical reference allocation.
    """
    vertices, indices, matrix = _mesh_inputs(
        vertices_camera_m, faces, camera_matrix, width, height, near_clip_m,
    )
    if platform.system() != "Linux":
        raise RuntimeError("Rendering is restricted to the Azure Linux CUDA runtime")
    import torch
    import pytorch3d
    from pytorch3d.structures import Meshes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU rendering fallback")
    if pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require audited PyTorch3D 0.7.9 camera/depth contracts")
    with torch.inference_mode():
        camera = _opencv_camera(torch, matrix, width, height)
        mesh = Meshes(
            verts=[torch.as_tensor(vertices, device="cuda")],
            faces=[torch.as_tensor(indices, device="cuda")],
        )
        pix, zbuf = _raster_native(mesh, camera, (int(height), int(width)), len(indices),
            full_face_capacity=full_face_capacity, capacity_diagnostics=capacity_diagnostics)
        mask, depth = pix[0, ..., 0] >= 0, zbuf[0, ..., 0]
        if mask.shape != (height, width) or depth.shape != mask.shape or not mask.is_cuda or not depth.is_cuda:
            raise RuntimeError("Raster output violated original-resolution/CUDA contract")
        if not torch.isfinite(depth[mask]).all() or (depth[mask] <= near_clip_m).any():
            raise RuntimeError("Raster camera depth is nonfinite or behind the near plane")
        depth = torch.where(mask, depth, torch.full_like(depth, float("nan")))
    return mask, depth


def _mesh_batch_inputs(vertices_camera_m, faces, camera_matrix, width, height, near_clip_m):
    vertices = _finite_array(vertices_camera_m, "vertices_camera_m")
    if vertices.ndim != 3 or vertices.shape[0] < 1 or vertices.shape[1] < 3 or vertices.shape[2] != 3:
        raise ValueError("Require a nonempty [B,V,3] batch sharing one fixed mesh topology")
    # Apply exactly the scalar contracts to every supplied candidate. Batched
    # execution is not permission to skip a failed candidate or alter topology.
    checked = [_mesh_inputs(value, faces, camera_matrix, width, height, near_clip_m) for value in vertices]
    return np.stack([record[0] for record in checked]), checked[0][1], checked[0][2]


def raster_camera_mesh_batch(vertices_camera_m, faces, camera_matrix, width=1536, height=1152,
                             *, near_clip_m=1e-4, full_face_capacity=None, capacity_diagnostics=None):
    """Same exact CUDA raster contracts, batching independent fixed-mesh poses.

    Caller controls batch size/memory; no downsampling, evidence filtering,
    clipping or approximate projection is introduced to gain throughput.
    Returns bool mask and camera-Z depth CUDA [B,H,W], NaN off the mesh.
    """
    vertices, indices, matrix = _mesh_batch_inputs(
        vertices_camera_m, faces, camera_matrix, width, height, near_clip_m,
    )
    if platform.system() != "Linux":
        raise RuntimeError("Rendering is restricted to the Azure Linux CUDA runtime")
    import torch
    import pytorch3d
    from pytorch3d.structures import Meshes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU rendering fallback")
    if pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require audited PyTorch3D 0.7.9 camera/depth contracts")
    count = len(vertices)
    with torch.inference_mode():
        camera = _opencv_camera(torch, matrix, width, height, batch_size=count)
        verts_gpu = torch.as_tensor(vertices, device="cuda")
        faces_gpu = torch.as_tensor(indices, device="cuda")
        mesh = Meshes(verts=list(verts_gpu.unbind(0)), faces=[faces_gpu] * count)
        pix, zbuf = _raster_native(mesh, camera, (int(height), int(width)), len(indices),
            full_face_capacity=full_face_capacity, capacity_diagnostics=capacity_diagnostics)
        mask, depth = pix[..., 0] >= 0, zbuf[..., 0]
        if mask.shape != (count, height, width) or depth.shape != mask.shape or not mask.is_cuda or not depth.is_cuda:
            raise RuntimeError("Batched raster output violated original-resolution/CUDA contract")
        if not torch.isfinite(depth[mask]).all() or (depth[mask] <= near_clip_m).any():
            raise RuntimeError("Batched raster camera depth is nonfinite or behind the near plane")
        depth = torch.where(mask, depth, torch.full_like(depth, float("nan")))
    return mask, depth


def _run_batch_parity(torch):
    """Own procedural mesh/poses only, exact parity before a speed claim.

    Benchmark full original image grid, 8 independent poses, median of 3
    synchronized trials. No challenge image, inferred mesh, or hidden GT.
    This tests computation parity, not pose reconstruction accuracy.
    """
    from scipy.spatial.transform import Rotation
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.)
    # Deliberately anisotropic, tilted object to test perspective and ordering.
    vertices = np.asarray(mesh.vertices) * [.4, .23, .31]
    rotations = Rotation.from_rotvec(np.arange(24).reshape(8, 3) * .03).as_matrix()
    translations = np.column_stack((np.linspace(-.9, .9, 8), np.linspace(.2, -.2, 8), np.linspace(3., 4., 8)))
    posed = vertices[None] @ rotations.transpose(0, 2, 1) + translations[:, None]
    K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
    # Scalar and batch use identical topology, no shortcuts to a mask score.
    scalar_masks, scalar_depths = [], []
    for value in posed:
        mask, depth = raster_camera_mesh(value, mesh.faces, K)
        scalar_masks.append(mask)
        scalar_depths.append(depth)
    scalar_masks, scalar_depths = torch.stack(scalar_masks), torch.stack(scalar_depths)
    batch_mask, batch_depth = raster_camera_mesh_batch(posed, mesh.faces, K)
    if not torch.equal(scalar_masks, batch_mask):
        raise RuntimeError("Batched CUDA raster changed exact silhouette pixels")
    error_m = float(torch.abs(scalar_depths[scalar_masks] - batch_depth[scalar_masks]).max())
    if not np.isfinite(error_m) or error_m > 1e-5:
        raise RuntimeError("Batched CUDA raster changed camera-Z beyond float32 tolerance")
    singleton_mask, singleton_depth = raster_camera_mesh_batch(posed[:1], mesh.faces, K)
    if not torch.equal(singleton_mask[0], scalar_masks[0]) or not torch.allclose(
        singleton_depth[0], scalar_depths[0], atol=1e-5, rtol=0, equal_nan=True,
    ):
        raise RuntimeError("Batched singleton differs from exact scalar raster")
    del scalar_masks, scalar_depths, batch_mask, batch_depth, singleton_mask, singleton_depth
    durations = {"scalar": [], "batch": []}
    for trial in range(3):
        for mode in (("scalar", "batch") if trial % 2 == 0 else ("batch", "scalar")):
            torch.cuda.synchronize()
            started = time.perf_counter()
            if mode == "scalar":
                for value in posed:
                    result = raster_camera_mesh(value, mesh.faces, K)
            else:
                result = raster_camera_mesh_batch(posed, mesh.faces, K)
            torch.cuda.synchronize()
            durations[mode].append(time.perf_counter() - started)
            del result
    speedup = float(np.median(durations["scalar"]) / np.median(durations["batch"]))
    return {"own_procedural_reference": True, "challenge_inputs_used": False,
            "batch_size": len(posed), "mesh_vertices": len(vertices), "mesh_faces": len(mesh.faces),
            "exact_mask_parity": True, "singleton_parity": True, "camera_z_max_error_m": error_m,
            "image_size_hw": [1152, 1536], "synchronized_trials_seconds": durations,
            "median_speedup": speedup, "predeclared_minimum_useful_speedup": 1.3,
            "throughput_hypothesis_accepted": bool(speedup >= 1.3),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}


def _smoke_reference():
    """Synthetic CPU arrays only; CUDA raster results must match these references."""
    width, height = 128, 96
    matrix = np.array([[160., 0., 64.], [0., 160., 48.], [0., 0., 1.]])
    uv = np.array([[20., 16.], [108., 20.], [44., 82.]])
    z = np.array([2., 4., 3.])
    triangle = np.column_stack(((uv - matrix[:2, 2]) * z[:, None] / 160., z))
    yy, xx = np.mgrid[:height, :width]
    # PyTorch3D's raster samples pixel cells at (x+.5, y+.5), not integers.
    rays = np.stack(((xx + .5 - 64.) / 160., (yy + .5 - 48.) / 160., np.ones_like(xx)), axis=-1)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    reference_z = np.dot(normal, triangle[0]) / np.einsum("hwc,c->hw", rays, normal)
    homogeneous_uv = np.column_stack((uv, np.ones(3)))
    barycentric = np.stack((xx + .5, yy + .5, np.ones_like(xx)), axis=-1) @ np.linalg.inv(homogeneous_uv)
    return width, height, matrix, uv, triangle, rays, reference_z, barycentric


def _run_smoke(torch):
    width, height, matrix, uv, triangle, rays, reference_z, barycentric = _smoke_reference()
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    matrix_full = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
    probes = np.array([[0., 0., 2.], [.1, .1, 2.], [1., -.25, 4.]])
    expected_full = np.array([[768., 576.], [864., 672.], [1248., 456.]])
    np.testing.assert_allclose(project_camera_points(probes, matrix_full), expected_full, atol=1e-10, rtol=0)
    camera = _opencv_camera(torch, matrix_full, 1536, 1152)
    screen = camera.transform_points_screen(torch.tensor(probes[None], device="cuda", dtype=torch.float32))
    error_pixels = float(np.abs(screen[0, :, :2].cpu().numpy() - expected_full).max())
    if error_pixels > 1e-3:
        raise RuntimeError("OpenCV/PyTorch3D projection mismatch, including image aspect")
    np.testing.assert_allclose(project_camera_points(triangle, matrix), uv, atol=1e-12, rtol=0)
    mask, depth = raster_camera_mesh(triangle, faces, matrix, width, height)
    mask_np, depth_np = mask.cpu().numpy(), depth.cpu().numpy()
    interior, exterior = (barycentric > 1e-4).all(axis=-1), (barycentric < -1e-4).any(axis=-1)
    if not interior.any() or not mask_np[interior].all() or mask_np[exterior].any():
        raise RuntimeError("Triangle silhouette fails the analytic pixel-centre pinhole reference")
    depth_error_m = float(np.abs(depth_np[interior] - reference_z[interior]).max())
    if depth_error_m > 1e-4 or not np.isnan(depth_np[~mask_np]).all():
        raise RuntimeError("Perspective-correct linear camera-Z depth gate failed")
    ray_distance_gap = float(np.max(reference_z[interior] * (np.linalg.norm(rays[interior], axis=-1) - 1)))
    if ray_distance_gap < .01:
        raise RuntimeError("Synthetic tilted triangle does not distinguish camera Z from ray distance")
    # Rear face is first in the face array; the front face must still win.
    two_layers = np.concatenate((triangle / triangle[:, 2, None] * 4.,
                                 triangle / triangle[:, 2, None] * 2.))
    layer_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    layer_mask, layer_depth = raster_camera_mesh(two_layers, layer_faces, matrix, width, height)
    layer_mask_np, layer_depth_np = layer_mask.cpu().numpy(), layer_depth.cpu().numpy()
    if not layer_mask_np[interior].all() or layer_mask_np[exterior].any():
        raise RuntimeError("Overlapping layers changed the analytic projected silhouette")
    nearest_depth_error_m = float(np.abs(layer_depth_np[layer_mask_np] - 2.).max())
    if nearest_depth_error_m > 1e-5:
        raise RuntimeError("Raster selected the rear surface or nonmetric depth")
    # A closed box: its .25-pixel edge placement avoids exact shared-diagonal
    # pixel ties, which are not a portable hard-rasterizer coverage guarantee.
    rectangle = np.array([[36.25, 28.25], [92.25, 28.25], [92.25, 68.25], [36.25, 68.25]])
    front = np.column_stack(((rectangle - matrix[:2, 2]) * 2.5 / 160., np.full(4, 2.5)))
    back = front.copy()
    back[:, 2] = 3.5
    box = np.concatenate((front, back))
    box_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                          [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                          [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], dtype=np.int64)
    box_mask, box_depth = raster_camera_mesh(box, box_faces, matrix, width, height)
    yy, xx = np.mgrid[:height, :width]
    reference_box = ((xx + .5 > 36.25) & (xx + .5 < 92.25)
                     & (yy + .5 > 28.25) & (yy + .5 < 68.25))
    box_mask_np, box_depth_np = box_mask.cpu().numpy(), box_depth.cpu().numpy()
    box_iou = silhouette_iou(box_mask_np, reference_box)
    box_depth_error_m = float(np.abs(box_depth_np[reference_box] - 2.5).max())
    if box_iou != 1. or not np.isfinite(box_depth_error_m) or box_depth_error_m > 1e-5:
        raise RuntimeError("Closed-box silhouette or front-surface camera-Z gate failed")
    torch.cuda.synchronize()
    return {"projection_max_error_pixels": error_pixels,
            "tilted_triangle_camera_z_max_error_m": depth_error_m,
            "tilted_triangle_ray_distance_gap_m": ray_distance_gap,
            "occlusion_nearest_depth_max_error_m": nearest_depth_error_m,
            "closed_box_mask_iou": box_iou, "closed_box_camera_z_max_error_m": box_depth_error_m,
            "synthetic_image_size_hw": [height, width], "synthetic_K": matrix.tolist(),
            "original_image_size_hw": [1152, 1536], "original_K": matrix_full.tolist()}



def _load_external_capacity_geometry(args):
    """Exact already-inferred external public-only object, no image/GT mounts."""
    import trimesh
    from scipy.spatial.transform import Rotation
    root = args.root.resolve()
    producer = "125aab2fbbe3a9b1d0f88bea8cb9f2fe17a719ed"
    sequence = "2026-06-03_17-02-20_beige_bin_ground_desk_03"
    original_base = root / "results" / ("form-hoi-external-predict-" + producer) / sequence
    frozen = {
        "object": dict(bytes=9165, sha256="caf4c261e4370e0534c192aec93c3964929c11954079659098c09cadd4465904"),
        "body": dict(bytes=29676, sha256="0b864574dfec12553cf02d7d3981f3e0b624c8c62f901f259c4f63f6b382442a")}
    loaded = []
    for stage in ("object", "body"):
        path = getattr(args, "external_" + stage + "_report")
        expected = dict(bytes=getattr(args, "external_" + stage + "_report_bytes"),
                        sha256=getattr(args, "external_" + stage + "_report_sha256"))
        if path is None or expected["bytes"] is None or expected["sha256"] is None:
            raise ValueError("Both exact external object/body report pins required")
        if (path != original_base / ("body_depth" if stage == "body" else "object") / "report.json"
                or expected != frozen[stage]
                or {"eval_private", "gt", "track_1"}.intersection(path.parts)):
            raise ValueError("External inference stage only, no reference/challenge assets")
        _checked_pin(path, expected, limit=1048576)
        report = json.loads(path.read_bytes())
        if (report.get("schema") != "world_reward.form_external_prediction_stage.v1"
                or report.get("status") != "complete" or report.get("dataset") != "nvidia/form-hoi"
                or report.get("stage") != ("body_depth" if stage == "body" else "object")
                or any(report.get(k) is not False for k in ("ground_truth_used", "private_truth_read",
                    "hand_labeled_test", "reference_inputs_mounted")) or report.get("oracle_modes") != []
                or report.get("original_frame_indices") != list(range(96))):
            raise ValueError("Sealed public-only native external96 prediction required")
        loaded.append((path, report, expected))
    obj_path, obj, obj_pin = loaded[0]; body_path, body, body_pin = loaded[1]
    for key in ("producer_revision", "sequence_id", "dataset", "input_pin", "video_pin", "source_binding"):
        if obj.get(key) is None or obj.get(key) != body.get(key):
            raise ValueError("Paired exact external inference lineage required")
    if obj["producer_revision"] != producer or obj["sequence_id"] != sequence:
        raise ValueError("Frozen first external producer/sequence required")
    frozen_assets = {
        "gauge.json": dict(bytes=8031, sha256="392b416ce038d85b9349a117a7fccc80c0faa5ab5849244f83321bca85bcda6e"),
        "object.glb": dict(bytes=14500684, sha256="367ddadac00735461ae8d20630a72261772d73b52e0f3876e1162ec40f201555"),
        "transform.json": dict(bytes=357, sha256="d2179fe79c63a6cdd816a47671f958d404acb5a29d67f1d5116767a663af50ca")}
    asset_pins = {}
    for path, record, name in ((obj_path, obj, "object.glb"), (obj_path, obj, "transform.json"),
                               (body_path, body, "gauge.json")):
        if name not in record.get("artifacts", {}):
            raise ValueError("Missing pinned native geometry/gauge")
        if record["artifacts"][name] != frozen_assets[name]:
            raise ValueError("Frozen complete native geometry/gauge pin differs")
        asset_pins[name] = _checked_pin(path.parent / name, record["artifacts"][name])
    mesh = trimesh.load(obj_path.parent / "object.glb", force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) != args.external_expected_faces or len(mesh.faces) != obj["faces"]:
        raise ValueError("Whole original external triangle count differs")
    transform = json.loads((obj_path.parent / "transform.json").read_bytes())
    gauge = json.loads((body_path.parent / "gauge.json").read_bytes())
    scale = np.asarray(transform["scale"], dtype=float)
    quaternion = np.asarray(transform["rotation"], dtype=float)
    if (scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any()
            or quaternion.shape != (4,) or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)):
        raise ValueError("Native fixed object scale/quaternion required")
    vertices = np.asarray(mesh.vertices, float) * scale[None]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    translation = np.asarray(transform["translation"], float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Native complete finite translation required")
    # Four declared rigid probe poses retain every face/vertex and scale. The
    # first is the actual native anchor; others exercise bin crossings, not a fit.
    rotations = Rotation.from_rotvec([[0, 0, 0], [.013, -.017, .009],
                                      [-.019, .007, .011], [.009, .015, -.013]]).as_matrix() @ rotation
    translations = translation + np.array([[0, 0, 0], [.013, -.017, 0],
                                           [-.021, .007, 0], [.009, .019, 0]])
    posed = vertices[None] @ rotations.transpose(0, 2, 1) + translations[:, None]
    provenance = dict(object_report_pin=obj_pin, body_report_pin=body_pin, asset_pins=asset_pins,
        producer_revision=obj["producer_revision"], sequence_id=obj["sequence_id"],
        dataset=obj["dataset"], input_pin=obj["input_pin"], video_pin=obj["video_pin"],
        full_vertices=len(vertices), full_faces=len(mesh.faces),
        whole_native_geometry_retained=True, test_pose_fit=False, GT_or_RGB_used=False)
    return posed, np.asarray(mesh.faces, np.int64), _intrinsics(gauge["K"]), provenance


def _procedural_capacity_cases():
    """Disconnected non-coplanar objects, occlusion and deterministic triangles."""
    import trimesh
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(19623)
    a = trimesh.creation.icosphere(subdivisions=3, radius=1.)
    b = trimesh.creation.icosphere(subdivisions=2, radius=1.)
    av = np.asarray(a.vertices) * [.45, .28, .37] + [-.11, -.05, 3.1]
    bv = np.asarray(b.vertices) * [.25, .31, .21] + [.07, .02, 2.6]
    # Independent tilted triangles include off-image AABBs and broad overlap.
    centers = rng.uniform([-.9, -.6, 2.3], [.9, .6, 4.2], (120, 1, 3))
    tv = centers + rng.normal(size=(120, 3, 3)) * [.05, .04, .03]
    vertices = np.concatenate((av, bv, tv.reshape(-1, 3)))
    faces = np.concatenate((a.faces, b.faces + len(av),
                            np.arange(360).reshape(-1, 3) + len(av) + len(bv)))
    centered = vertices - [0, 0, 3]
    rotations = Rotation.from_rotvec(rng.normal(size=(4, 3)) * .035).as_matrix()
    translations = np.array([[0, 0, 3], [.03, -.01, 3.1], [-.025, .015, 2.95], [.01, .02, 3.05]])
    posed = centered[None] @ rotations.transpose(0, 2, 1) + translations[:, None]
    K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
    return posed, faces, K


def _capacity_pair(torch, name, vertices, faces, K, batch_size):
    """One synchronized original-grid timing per mode; never claim a median."""
    measured = {}; outputs = {}; diagnostics = {}
    for reference in (True, False):
        mode = "full_face_reference" if reference else "conservative_AABB_bound"
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); info = {}
        if batch_size == 1:
            result = raster_camera_mesh(vertices[0], faces, K, full_face_capacity=reference,
                                        capacity_diagnostics=info)
        else:
            result = raster_camera_mesh_batch(vertices[:batch_size], faces, K,
                full_face_capacity=reference, capacity_diagnostics=info)
        torch.cuda.synchronize()
        measured[mode] = dict(seconds=time.perf_counter() - start,
                             peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        outputs[mode] = result; diagnostics[mode] = info
    rm, rz = outputs["full_face_reference"]; cm, cz = outputs["conservative_AABB_bound"]
    mask_equal = torch.equal(rm, cm)
    depth_close = bool(torch.allclose(rz, cz, atol=1e-6, rtol=0., equal_nan=True))
    if not mask_equal or not depth_close:
        raise RuntimeError("Allocation-only raster parity failed: " + name + "/B" + str(batch_size))
    error = float(torch.abs(rz[rm] - cz[rm]).max()) if rm.any() else 0.
    row = dict(case=name, batch_size=batch_size, image_size_hw=[1152, 1536],
        mesh_vertices=vertices.shape[1], mesh_faces=len(faces), exact_mask_parity=mask_equal,
        depth_atol_m=1e-6, depth_rtol=0., depth_allclose=depth_close, max_camera_z_error_m=error,
        synchronized_single_trial=measured, capacity=diagnostics,
        single_trial_speed_ratio=measured["full_face_reference"]["seconds"] / measured["conservative_AABB_bound"]["seconds"])
    del outputs, result, rm, rz, cm, cz; torch.cuda.empty_cache()
    if not mask_equal or not depth_close or not np.isfinite(error):
        raise RuntimeError("Allocation-only raster parity failed: " + name + "/B" + str(batch_size))
    return row


def _run_capacity_gate(torch, args, report):
    if args.external_expected_faces != 743576:
        raise ValueError("Frozen first external whole-object gate requires 743576 faces")
    posed, faces, K, provenance = _load_external_capacity_geometry(args)
    report["external_geometry"] = provenance
    gates = dict(all_exact_masks_and_depth_tolerance=False, external_batch_sizes=[1, 4],
        procedural_non_coplanar_disconnected_occluding_objects=True,
        whole_mesh_original_grid_preserved=True, rows=[])
    report["gates"] = gates
    procedural, proc_faces, proc_K = _procedural_capacity_cases()
    for name, vertices, topology, matrix in (("procedural_random_multiobject_occlusion", procedural, proc_faces, proc_K),
                                             ("actual_first_external_object", posed, faces, K)):
        for batch_size in (1, 4):
            gates["rows"].append(_capacity_pair(torch, name, vertices, topology, matrix, batch_size))
            print(json.dumps(dict(case=name, batch_size=batch_size, parity=True)), flush=True)
    _, _, _, rehashed = _load_external_capacity_geometry(args)
    if rehashed != provenance:
        raise RuntimeError("Frozen external inference artifacts changed during gate")
    gates["external_assets_rehashed_after"] = True
    gates["all_exact_masks_and_depth_tolerance"] = True
    gates["performance_inference"] = "single_trial_diagnostic_not_throughput_distribution_or_HOI_quality"
    return gates

def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Renderer smoke is restricted to Azure Linux")
    if {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require docker --network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--batch-parity", action="store_true")
    parser.add_argument("--capacity-gate", action="store_true")
    parser.add_argument("--external-expected-faces", type=int, default=743576)
    for stage in ("object", "body"):
        parser.add_argument("--external-" + stage + "-report", type=Path)
        parser.add_argument("--external-" + stage + "-report-bytes", type=int)
        parser.add_argument("--external-" + stage + "-report-sha256")
    args = parser.parse_args()
    if args.capacity_gate and args.batch_parity:
        raise ValueError("Choose a single bounded gate")
    report_path = args.report or args.root / "results" / ("raster-capacity-gate.json" if args.capacity_gate else
        "camera-render-batch.json" if args.batch_parity else "camera-render.json")
    if report_path.exists():
        raise FileExistsError("Renderer smoke reports are frozen; do not overwrite")
    import torch
    import pytorch3d

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU rendering fallback")
    report = {"schema": "world_reward.raster_capacity_gate.v1" if args.capacity_gate else "world_reward.camera_render.v1",
              "stage": "conservative_raster_capacity" if args.capacity_gate else
                       "opencv_camera_cuda_batch_parity" if args.batch_parity else "opencv_camera_cuda_raster_smoke", "status": "fail",
              "producer_revision": os.environ.get("WR_CODE_REVISION"), "image_id": os.environ.get("WR_IMAGE_ID"),
              "source_pins": _raster_source_pins() if args.capacity_gate else {},
              "torch": torch.__version__, "pytorch3d": pytorch3d.__version__,
              "expected_pytorch3d_revision": PYTORCH3D_REVISION,
              "gpu": torch.cuda.get_device_name(), "backend": "pytorch3d_cuda_no_egl",
              "coordinate_convention": "opencv_x_right_y_down_z_forward",
              "depth_kind": "linear_camera_z", "depth_units": "metres", "background_depth": "nan",
              "pixel_sample": "x_plus_0.5_y_plus_0.5", "resizing": False,
              "synthetic_analytic_reference": not args.capacity_gate, "challenge_inputs_used": False,
              "challenge_ground_truth_used": False, "challenge_performance_verified": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        if args.capacity_gate:
            def expired(*_): raise TimeoutError("Fixed inclusive1200s capacity gate budget exceeded")
            signal.signal(signal.SIGALRM, expired); signal.alarm(1200)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            report["gates"] = _run_capacity_gate(torch, args, report)
        else:
            report["gates"] = _run_batch_parity(torch) if args.batch_parity else _run_smoke(torch)
        if args.capacity_gate and report["source_pins"] != _raster_source_pins():
            raise RuntimeError("Raster source changed during CUDA gate")
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report_path.chmod(0o444)
        print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

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


def _opencv_camera(torch, camera_matrix, width, height):
    from pytorch3d.utils import cameras_from_opencv_projection

    return cameras_from_opencv_projection(
        R=torch.eye(3, device="cuda", dtype=torch.float32)[None],
        tvec=torch.zeros((1, 3), device="cuda", dtype=torch.float32),
        camera_matrix=torch.as_tensor(camera_matrix, device="cuda", dtype=torch.float32)[None],
        image_size=torch.tensor([[height, width]], device="cuda", dtype=torch.float32),
    )


def raster_camera_mesh(vertices_camera_m, faces, camera_matrix, width=1536, height=1152,
                       *, near_clip_m=1e-4):
    """Return CUDA tensors (hard_mask, depth_m), both original-resolution (H, W).

    hard_mask is bool; depth_m is float32 camera Z with NaN outside the mesh.
    CUDA and Linux are mandatory; imports are lazy so numpy contracts remain
    testable locally. All input triangles are retained. Near-plane crossings
    fail explicitly, while off-image geometry is naturally not rasterized.
    Non-square NDC conversion is delegated to PyTorch3D's OpenCV utility.
    No ad-hoc axis flip, aspect scaling, metric normalization or resizing occurs.
    This inference/QA rasterizer is not a differentiable fitting-loss wrapper.
    """
    vertices, indices, matrix = _mesh_inputs(
        vertices_camera_m, faces, camera_matrix, width, height, near_clip_m,
    )
    if platform.system() != "Linux":
        raise RuntimeError("Rendering is restricted to the Azure Linux CUDA runtime")
    import torch
    import pytorch3d
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
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
        settings = RasterizationSettings(
            image_size=(int(height), int(width)), blur_radius=0., faces_per_pixel=1,
            perspective_correct=True, clip_barycentric_coords=False,
            cull_backfaces=False, cull_to_frustum=False, z_clip_value=None,
            # A fixed small capacity can silently overflow a crowded bin. The
            # full face count bounds every bin without changing mesh topology.
            max_faces_per_bin=len(indices),
        )
        fragments = MeshRasterizer(cameras=camera, raster_settings=settings)(mesh)
        mask, depth = fragments.pix_to_face[0, ..., 0] >= 0, fragments.zbuf[0, ..., 0]
        if mask.shape != (height, width) or depth.shape != mask.shape or not mask.is_cuda or not depth.is_cuda:
            raise RuntimeError("Raster output violated original-resolution/CUDA contract")
        if not torch.isfinite(depth[mask]).all() or (depth[mask] <= near_clip_m).any():
            raise RuntimeError("Raster camera depth is nonfinite or behind the near plane")
        depth = torch.where(mask, depth, torch.full_like(depth, float("nan")))
    return mask, depth


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


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Renderer smoke is restricted to Azure Linux")
    if {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require docker --network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.root / "results/camera-render.json"
    if report_path.exists():
        raise FileExistsError("Renderer smoke reports are frozen; do not overwrite")
    import torch
    import pytorch3d

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU rendering fallback")
    report = {"stage": "opencv_camera_cuda_raster_smoke", "status": "fail",
              "torch": torch.__version__, "pytorch3d": pytorch3d.__version__,
              "expected_pytorch3d_revision": PYTORCH3D_REVISION,
              "gpu": torch.cuda.get_device_name(), "backend": "pytorch3d_cuda_no_egl",
              "coordinate_convention": "opencv_x_right_y_down_z_forward",
              "depth_kind": "linear_camera_z", "depth_units": "metres", "background_depth": "nan",
              "pixel_sample": "x_plus_0.5_y_plus_0.5", "resizing": False,
              "synthetic_analytic_reference": True, "challenge_inputs_used": False,
              "challenge_ground_truth_used": False, "challenge_performance_verified": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        report["gates"] = _run_smoke(torch)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

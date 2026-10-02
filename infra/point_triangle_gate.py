"""Analytic CUDA gate for continuous small-triangle distances, not optimization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from world_reward.point_triangle import observed_to_triangle_distance_squared


PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"


def analytic_fixture():
    """Interior, boundary, outside vertex/edge; area 5e-5 square metres.

    The outside-edge case detects the pinned kernel's regularized barycentric
    containment, not just the configurable minimum-area fallback.
    """
    triangle = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]], dtype=np.float32)
    points = np.array([[.0025, .0025, .02], [.005, .005, .02], [-.005, -.004, .02],
                       [.0075, .0075, .02]], dtype=np.float32)
    closest = np.array([[.0025, .0025, 0.], [.005, .005, 0.], [0., 0., 0.],
                        [.005, .005, 0.]], dtype=np.float32)
    delta = points.astype(np.float64) - closest.astype(np.float64)
    return points, triangle, np.sum(delta * delta, axis=1), 2 * delta


def _run(torch, diagnostics):
    from pytorch3d.loss.point_mesh_distance import point_face_distance
    from scipy.spatial.transform import Rotation
    points_np, triangles_np, distances_np, gradient_np = analytic_fixture()
    points = torch.tensor(points_np, device="cuda", requires_grad=True)
    triangles = torch.tensor(triangles_np, device="cuda", requires_grad=True)
    distance = observed_to_triangle_distance_squared(points, triangles)
    distance_error = float(np.abs(distance.detach().cpu().numpy() - distances_np).max())
    diagnostics.update({"squared_distance_max_error_m2": distance_error,
                        "outside_edge_expected_squared_distance_m2": float(distances_np[-1]),
                        "outside_edge_actual_squared_distance_m2": float(distance[-1].detach())})
    if distance_error > 1e-9:
        raise RuntimeError(f"Continuous small-triangle distance disagrees with analytic reference: {distance_error:.9g} m^2")
    distance.sum().backward()
    point_gradient_error = float(np.abs(points.grad.cpu().numpy() - gradient_np).max())
    if point_gradient_error > 1e-6 or not torch.isfinite(triangles.grad).all():
        raise RuntimeError("Point/triangle autograd failed analytic first-order contract")
    interior = points[:1].detach().clone().requires_grad_()
    tri_interior = triangles.detach().clone().requires_grad_()
    observed_to_triangle_distance_squared(interior, tri_interior).sum().backward()
    expected_triangle_gradient = -np.array([.5, .25, .25])[None, :, None] * gradient_np[0]
    triangle_gradient_error = float(np.abs(tri_interior.grad.cpu().numpy() - expected_triangle_gradient).max())
    if triangle_gradient_error > 1e-6:
        raise RuntimeError("Interior triangle gradient disagrees with barycentric analytic derivative")
    first = torch.zeros(1, device="cuda", dtype=torch.int64)
    default_distance = point_face_distance(interior, first, tri_interior, first, 1, 5e-3)
    default_bias = float(default_distance[0].detach() - distance[0].detach())
    if default_bias < 5e-6:
        raise RuntimeError("Small triangle fixture does not expose the upstream area threshold approximation")
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix().astype(np.float32)
    R, t = torch.tensor(rotation, device="cuda"), torch.tensor([.03, -.04, 1.], device="cuda")
    transformed = observed_to_triangle_distance_squared(points @ R.T + t, triangles @ R.T + t)
    rigid_error = float(torch.abs(transformed - distance).max().detach())
    duplicate_error = float(torch.abs(observed_to_triangle_distance_squared(points, triangles.repeat(2, 1, 1)) - distance).max().detach())
    singleton_error = float(torch.abs(observed_to_triangle_distance_squared(points[:1], triangles)[0] - distance[0]).detach())
    if max(rigid_error, duplicate_error, singleton_error) > 1e-8:
        raise RuntimeError("Continuous metric distance changed under rigid pose, duplication or singleton batching")
    rejected = 0
    for bad_points, bad_triangles in ((points[:0], triangles), (points, triangles[:0]),
                                      (points, torch.zeros_like(triangles))):
        try:
            observed_to_triangle_distance_squared(bad_points, bad_triangles)
        except ValueError:
            rejected += 1
        else:
            raise RuntimeError("Invalid/empty continuous surface input was accepted")
    return {"triangle_area_m2": .00005, "min_triangle_area": 0., "squared_distance_max_error_m2": distance_error,
            "point_gradient_max_error": point_gradient_error, "interior_triangle_gradient_max_error": triangle_gradient_error,
            "upstream_default_area_threshold_distance_bias_m2": default_bias,
            "rigid_equivariance_max_error_m2": rigid_error, "triangle_duplicate_max_error_m2": duplicate_error,
            "singleton_max_error_m2": singleton_error, "invalid_inputs_rejected": rejected,
            "direction": "observed_to_continuous_triangle_only", "reverse_hidden_surface_loss": False}


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable WR_CODE_REVISION")
    output = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward")) / "results/point-triangle-gate.json"
    if output.exists():
        raise FileExistsError("Point/triangle gate reports are frozen")
    import torch
    import pytorch3d
    if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Require CUDA and audited PyTorch3D 0.7.9")
    report = {"stage": "analytic_continuous_point_triangle_cuda", "status": "fail", "code_revision": revision,
              "expected_pytorch3d_revision": PYTORCH3D_REVISION, "torch": torch.__version__,
              "pytorch3d": pytorch3d.__version__, "gpu": torch.cuda.get_device_name(),
              "own_procedural_analytic_reference": True, "challenge_inputs_used": False, "models_used": False,
              "challenge_ground_truth_used": False, "challenge_performance_verified": False,
              "optimizer_implemented": False, "geometry_units": "metres", "output_units": "metres_squared",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        report["gates"] = {"triangle_area_m2": .00005, "min_triangle_area": 0.,
                           "direction": "observed_to_continuous_triangle_only"}
        report["gates"].update(_run(torch, report["gates"]))
        torch.cuda.synchronize()
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

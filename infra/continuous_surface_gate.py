"""Own analytic CUDA/first-order gate; no challenge input or shape fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from point_triangle_gate import analytic_fixture
from world_reward.continuous_surface import observed_to_triangle_distance_squared


DISTANCE_ATOL_M2 = 1e-9
GRADIENT_ATOL = 1e-6
MAX_GATE_SECONDS = 30.
MAX_ITERATION_SECONDS = 3.


def _analytic(torch, report):
    points_np, triangles_np, distances_np, gradients_np = analytic_fixture()
    points = torch.tensor(points_np, device="cuda", requires_grad=True)
    triangles = torch.tensor(triangles_np, device="cuda", requires_grad=True)
    distance = observed_to_triangle_distance_squared(points, triangles)
    if distance.dtype != torch.float64 or distance.device != points.device:
        raise RuntimeError("Require metric float64 output on the original CUDA device")
    error = float(np.abs(distance.detach().cpu().numpy() - distances_np).max())
    report["squared_distance_max_error_m2"] = error
    if error > DISTANCE_ATOL_M2:
        raise RuntimeError("Own continuous distance failed the original frozen small-triangle reference")
    distance.sum().backward()
    gradient_error = float(np.abs(points.grad.cpu().numpy() - gradients_np).max())
    report["point_gradient_max_error"] = gradient_error
    if gradient_error > GRADIENT_ATOL or not torch.isfinite(triangles.grad).all():
        raise RuntimeError("Point gradient or finite triangle gradient contract failed")
    interior = points[:1].detach().clone().requires_grad_()
    tri = triangles.detach().clone().requires_grad_()
    observed_to_triangle_distance_squared(interior, tri).sum().backward()
    expected = -np.array([.5, .25, .25])[None, :, None] * gradients_np[0]
    error = float(np.abs(tri.grad.cpu().numpy() - expected).max())
    report["interior_triangle_gradient_max_error"] = error
    if error > GRADIENT_ATOL:
        raise RuntimeError("Selected triangle autograd failed the analytic barycentric derivative")
    from scipy.spatial.transform import Rotation
    R = torch.tensor(Rotation.from_rotvec([.2, -.3, .1]).as_matrix(), device="cuda")
    t = torch.tensor([.03, -.04, 1.], dtype=torch.float64, device="cuda")
    rigid = observed_to_triangle_distance_squared(points.double() @ R.T + t, triangles.double() @ R.T + t)
    error = float(torch.abs(rigid - distance).max().detach())
    report["rigid_equivariance_max_error_m2"] = error
    if error > 1e-12:
        raise RuntimeError("Own continuous metric changed under a float64 rigid transform")
    for chunk in (1, 64):
        duplicate = observed_to_triangle_distance_squared(points, triangles.repeat(2, 1, 1), point_chunk_size=chunk)
        if not torch.equal(duplicate, distance):
            raise RuntimeError("Duplication/chunking changed exact distances")
    duplicated = triangles.detach().repeat(2, 1, 1).requires_grad_()
    observed_to_triangle_distance_squared(interior.detach(), duplicated).sum().backward()
    if not torch.equal(duplicated.grad[1], torch.zeros_like(duplicated.grad[1])):
        raise RuntimeError("Exact duplicate tie must assign lowest face index")
    thin = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1e-10, 0.]]], dtype=torch.float64, device="cuda", requires_grad=True)
    thin_point = torch.tensor([[.75, .25e-10, .02]], dtype=torch.float64, device="cuda", requires_grad=True)
    thin_distance = observed_to_triangle_distance_squared(thin_point, thin)
    thin_distance.sum().backward()
    if abs(float(thin_distance.detach()[0]) - .0004) > 1e-12 or not torch.isfinite(thin.grad).all():
        raise RuntimeError("Thin nondegenerate triangle failed without an epsilon/Gram regularizer")
    rejected = 0
    for p, f in ((points[:0], triangles), (points, triangles[:0]), (points, torch.zeros_like(triangles))):
        try:
            observed_to_triangle_distance_squared(p, f)
        except ValueError:
            rejected += 1
        else:
            raise RuntimeError("Invalid geometry accepted or silently repaired")
    report.update({"invalid_inputs_rejected": rejected, "duplicate_chunk_exact": True,
                   "exact_tie_lowest_face": True, "thin_triangle_pass": True})


def _feasibility(torch, report):
    """No models: original fixed-size own mesh, finite forward+backward budget."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh.vertices *= [.35, .21, .28]
    # 5120 faces stress a larger mesh than the official4096-row budget.
    if len(mesh.faces) != 5120:
        raise RuntimeError("Frozen own mesh topology changed")
    rng = np.random.default_rng(711)
    ids = rng.integers(0, len(mesh.vertices), 2048)
    observations = np.asarray(mesh.vertices)[ids] + rng.normal(0, .002, (2048, 3))
    points = torch.tensor(observations, dtype=torch.float64, device="cuda", requires_grad=True)
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.float64, device="cuda", requires_grad=True)
    durations = report["forward_backward_seconds"] = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(3):
        points.grad = faces.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        observed_to_triangle_distance_squared(points, faces).mean().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        if (not torch.isfinite(points.grad).all() or not torch.isfinite(faces.grad).all()
                or elapsed > MAX_ITERATION_SECONDS):
            raise RuntimeError("Continuous forward/backward failed finite gradients or fixed3s feasibility budget")
    report.update({"observations": 2048, "triangles": len(mesh.faces), "point_chunk_size": 64,
                   "seed": 711, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "median_forward_backward_seconds": float(np.median(durations))})


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable source revision")
    output = Path(os.environ["WR_ROOT"]) / "results/continuous-surface-gate.json"
    if output.exists():
        raise FileExistsError("Own continuous surface report is frozen")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no local fallback")
    import world_reward.continuous_surface as primitive
    report = {"stage": "own_continuous_surface_analytic_cuda", "status": "fail", "code_revision": revision,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "network": "none",
              "own_procedural_inputs_only": True, "challenge_inputs_used": False, "models_used": False,
              "challenge_ground_truth_used": False, "geometry_units": "metres", "output_units": "metres_squared",
              "fitter_adoption_authorized": False, "challenge_performance_verified": False,
              "distance_atol_m2": DISTANCE_ATOL_M2, "gradient_atol": GRADIENT_ATOL,
              "max_iteration_seconds": MAX_ITERATION_SECONDS, "max_gate_seconds": MAX_GATE_SECONDS,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primitive_sha256": hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest(), "analytic": {}, "feasibility": {}}
    started = time.perf_counter()
    try:
        _analytic(torch, report["analytic"])
        _feasibility(torch, report["feasibility"])
        if time.perf_counter() - started > MAX_GATE_SECONDS:
            raise RuntimeError("Own continuous gate exceeded frozen30s total budget")
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

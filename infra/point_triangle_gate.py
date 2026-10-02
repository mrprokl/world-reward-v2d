"""Analytic CUDA gate for continuous small-triangle distances, not optimization."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

from world_reward.point_triangle import observed_to_triangle_distance_squared


PYTORCH3D_REVISION = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
KAOLIN_REVISION = "06ffb7d955ca26b608c60a9e862327c56b226921"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--backend", choices=("pytorch3d", "kaolin"), default="pytorch3d")
    values = list(sys.argv[1:] if argv is None else argv)
    if sum(value == "--backend" or value.startswith("--backend=") for value in values) > 1:
        parser.error("--backend cannot be repeated")
    return parser.parse_args(values)


def report_path(root, backend):
    name = "point-triangle-gate.json" if backend == "pytorch3d" else "point-triangle-kaolin-gate.json"
    return Path(root) / "results" / name


def _file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _kaolin_identity(kaolin):
    """Record installed bytes, not a claim that wheel matches upstream source."""
    from kaolin import _C
    distribution = metadata.distribution("kaolin")
    if kaolin.__version__ != "0.18.0" or distribution.version != "0.18.0":
        raise RuntimeError("Require installed Kaolin distribution and module 0.18.0")
    inventory = distribution.files or []
    metadatas = [path for path in inventory if str(path).endswith(".dist-info/METADATA")]
    if len(metadatas) != 1:
        raise RuntimeError("Require exactly one Kaolin installed distribution METADATA")
    root = Path(kaolin.__file__).parent
    return {"version": distribution.version, "expected_source_revision": KAOLIN_REVISION,
            "installed_distribution_metadata": _file_identity(distribution.locate_file(metadatas[0])),
            "triangle_metric_source": _file_identity(root / "metrics/trianglemesh.py"),
            "cuda_extension_binary": _file_identity(_C.__file__),
            "root_initializer": _file_identity(kaolin.__file__),
            "upstream_release_binary_identity_verified": False,
            "full_import_closure_commercial_eligibility_verified": False,
            "standard_import_noncommercial_components_present": "kaolin.non_commercial" in sys.modules,
            "license_status": "metric_and_kernel_Apache2; standard_import_noncommercial_closure_unresolved"}


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


def _run(torch, diagnostics, backend):
    from scipy.spatial.transform import Rotation
    points_np, triangles_np, distances_np, gradient_np = analytic_fixture()
    points = torch.tensor(points_np, device="cuda", requires_grad=True)
    triangles = torch.tensor(triangles_np, device="cuda", requires_grad=True)

    def evaluate(p, t):
        return observed_to_triangle_distance_squared(p, t, backend=backend)

    distance = evaluate(points, triangles)
    distance_error = float(np.abs(distance.detach().cpu().numpy() - distances_np).max())
    diagnostics.update({"squared_distance_max_error_m2": distance_error,
                        "outside_edge_expected_squared_distance_m2": float(distances_np[-1]),
                        "outside_edge_actual_squared_distance_m2": float(distance[-1].detach())})
    if distance_error > 1e-9:
        raise RuntimeError(f"Continuous small-triangle distance disagrees with analytic reference: {distance_error:.9g} m^2")
    distance.sum().backward()
    point_gradient_error = float(np.abs(points.grad.cpu().numpy() - gradient_np).max())
    diagnostics["point_gradient_max_error"] = point_gradient_error
    if point_gradient_error > 1e-6 or not torch.isfinite(triangles.grad).all():
        raise RuntimeError("Point/triangle autograd failed analytic first-order contract")
    interior = points[:1].detach().clone().requires_grad_()
    tri_interior = triangles.detach().clone().requires_grad_()
    evaluate(interior, tri_interior).sum().backward()
    expected_triangle_gradient = -np.array([.5, .25, .25])[None, :, None] * gradient_np[0]
    triangle_gradient_error = float(np.abs(tri_interior.grad.cpu().numpy() - expected_triangle_gradient).max())
    diagnostics["interior_triangle_gradient_max_error"] = triangle_gradient_error
    if triangle_gradient_error > 1e-6:
        raise RuntimeError("Interior triangle gradient disagrees with barycentric analytic derivative")
    default_bias = None
    if backend == "pytorch3d":
        from pytorch3d.loss.point_mesh_distance import point_face_distance
        first = torch.zeros(1, device="cuda", dtype=torch.int64)
        default_distance = point_face_distance(interior, first, tri_interior, first, 1, 5e-3)
        default_bias = float(default_distance[0].detach() - distance[0].detach())
        if default_bias < 5e-6:
            raise RuntimeError("Small triangle fixture does not expose the upstream area threshold approximation")
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix().astype(np.float32)
    R, t = torch.tensor(rotation, device="cuda"), torch.tensor([.03, -.04, 1.], device="cuda")
    transformed = evaluate(points @ R.T + t, triangles @ R.T + t)
    rigid_error = float(torch.abs(transformed - distance).max().detach())
    duplicate_error = float(torch.abs(evaluate(points, triangles.repeat(2, 1, 1)) - distance).max().detach())
    singleton_error = float(torch.abs(evaluate(points[:1], triangles)[0] - distance[0]).detach())
    if max(rigid_error, duplicate_error, singleton_error) > 1e-8:
        raise RuntimeError("Continuous metric distance changed under rigid pose, duplication or singleton batching")
    rejected = 0
    for bad_points, bad_triangles in ((points[:0], triangles), (points, triangles[:0]),
                                      (points, torch.zeros_like(triangles))):
        try:
            evaluate(bad_points, bad_triangles)
        except ValueError:
            rejected += 1
        else:
            raise RuntimeError("Invalid/empty continuous surface input was accepted")
    return {"triangle_area_m2": .00005, "min_triangle_area": 0. if backend == "pytorch3d" else None,
            "squared_distance_max_error_m2": distance_error,
            "point_gradient_max_error": point_gradient_error, "interior_triangle_gradient_max_error": triangle_gradient_error,
            "upstream_default_area_threshold_distance_bias_m2": default_bias,
            "rigid_equivariance_max_error_m2": rigid_error, "triangle_duplicate_max_error_m2": duplicate_error,
            "singleton_max_error_m2": singleton_error, "invalid_inputs_rejected": rejected,
            "direction": "observed_to_continuous_triangle_only", "reverse_hidden_surface_loss": False}


def main(argv=None):
    args = parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    revision = os.environ.get("WR_CODE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable WR_CODE_REVISION")
    output = report_path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"), args.backend)
    if output.exists():
        raise FileExistsError("Point/triangle gate reports are frozen")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Require CUDA")
    report = {"stage": "analytic_continuous_point_triangle_cuda", "status": "fail", "code_revision": revision,
              "backend": args.backend, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "own_procedural_analytic_reference": True, "challenge_inputs_used": False, "models_used": False,
              "challenge_ground_truth_used": False, "challenge_performance_verified": False,
              "optimizer_implemented": False, "geometry_units": "metres", "output_units": "metres_squared",
              "fitter_adoption_authorized": False, "full_import_closure_commercial_eligibility_verified": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primitive_source": _file_identity(Path(__file__).resolve().parents[1] / "src/world_reward/point_triangle.py")}
    started = time.perf_counter()
    try:
        if args.backend == "pytorch3d":
            import pytorch3d
            if pytorch3d.__version__ != "0.7.9":
                raise RuntimeError("Require audited PyTorch3D 0.7.9")
            report.update({"expected_pytorch3d_revision": PYTORCH3D_REVISION, "pytorch3d": pytorch3d.__version__})
        else:
            import kaolin
            report["kaolin"] = _kaolin_identity(kaolin)
        report["gates"] = {"triangle_area_m2": .00005, "min_triangle_area": 0. if args.backend == "pytorch3d" else None,
                           "direction": "observed_to_continuous_triangle_only"}
        report["gates"].update(_run(torch, report["gates"], args.backend))
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

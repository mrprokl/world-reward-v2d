"""Independent scalar-only CUDA projection-conditioning probe, not validation.

Four new plane triangles only; no failed-cohort reads, images, models or arrays
saved. Original failure and downstream gates remain closed and unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time
import numpy as np

WIDTH, HEIGHT = 640, 480
K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
LABELS = ("wide", "grazing", "background_0", "background_1")


def identity(path):
    with Path(path).open("rb") as stream: sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": Path(path).stat().st_size, "sha256": sha}


def fixture():
    uv = np.array([[240.25, 120.25], [420.25, 170.25], [280.25, 370.25],
                   [500.4999, 80.25], [500.5001, 80.25], [500.5, 400.25]])
    z = np.array([2., 4., 3., 1.5, 2.5, 3.5])
    v = np.vstack((np.column_stack(((uv-[320, 240])*z[:, None]/800, z)),
                   [[-2.2, -1.7, 4.5], [-2.2, 1.7, 4.5], [2.2, 1.7, 4.5], [2.2, -1.7, 4.5]]))
    return v, np.array([[0, 2, 1], [3, 5, 4], [6, 7, 8], [6, 8, 9]], np.int64)


def recovered_projection(ndc):
    a = np.asarray(ndc)
    if a.dtype != np.float32 or a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("Actual finite FP32 projected coordinates required")
    b = a.astype(np.float64)
    return np.column_stack((-b[:, :2]*(240/800)*b[:, 2, None], b[:, 2]))


def reference_summary(triangle, xy, observed_depth, offset):
    if type(offset) is not float or offset not in (0., .5, 1.): raise ValueError("Frozen ray offset required")
    p, q, s = triangle; e1, e2 = q-p, s-p; normal = np.cross(e1, e2)
    direction = np.column_stack(((xy+offset-[320, 240])/800, np.ones(len(xy))))
    denominator = direction @ normal
    parallel = denominator == 0; usable = ~parallel
    if not usable.any(): return {"pixels": len(xy), "parallel_pixels": int(parallel.sum()), "finite_reference": False}
    d = direction[usable]; den = denominator[usable]; expected = (p @ normal)/den
    delta = d*expected[:, None]-p
    a, b, c = e1 @ e1, e1 @ e2, e2 @ e2; determinant = a*c-b*b
    if determinant <= 0: raise ValueError("Diagnostic reference triangle degeneracy")
    u = (c*(delta @ e1)-b*(delta @ e2))/determinant
    v = (a*(delta @ e2)-b*(delta @ e1))/determinant
    error = np.abs(expected-observed_depth[usable]); finite = np.isfinite(expected).all()
    if not finite: raise ValueError("Diagnostic nonfinite reference")
    return {"pixels": len(xy), "parallel_pixels": int(parallel.sum()), "finite_reference": True,
        "max_camera_z_error_m": float(error.max()), "p99_camera_z_error_m": float(np.quantile(error, .99)),
        "count_error_gt_2e_minus5_m": int(np.sum(error > .00002)),
        "minimum_barycentric": float(np.min(np.column_stack((1-u-v, u, v)))),
        "front_pixels": int(np.sum(den < 0)), "back_or_tangent_pixels": int(np.sum(den >= 0)),
        "minimum_abs_normal_ray_cos": float(np.min(np.abs(den)/(np.linalg.norm(d, axis=1)*np.linalg.norm(normal))))}


def face_geometry(triangle):
    uv = triangle[:, :2]/triangle[:, 2, None]*800+[320, 240]
    e1, e2 = uv[1]-uv[0], uv[2]-uv[0]
    return {"projected_double_area_pixels2": float(abs(e1[0]*e2[1]-e1[1]*e2[0])),
            "minimum_vertex_camera_z_m": float(triangle[:, 2].min()), "maximum_vertex_camera_z_m": float(triangle[:, 2].max())}


def run_cuda(torch):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    vertices, faces = fixture(); camera = cameras_from_opencv_projection(R=torch.eye(3, device="cuda")[None],
        tvec=torch.zeros((1, 3), device="cuda"), camera_matrix=torch.as_tensor(K, device="cuda", dtype=torch.float32)[None],
        image_size=torch.tensor([[HEIGHT, WIDTH]], device="cuda", dtype=torch.float32))
    mesh = Meshes(verts=[torch.as_tensor(vertices, dtype=torch.float32, device="cuda")],
                  faces=[torch.as_tensor(faces, device="cuda")])
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=RasterizationSettings(image_size=(HEIGHT, WIDTH),
        blur_radius=0., faces_per_pixel=1, perspective_correct=True, clip_barycentric_coords=False,
        cull_backfaces=False, cull_to_frustum=False, z_clip_value=None, max_faces_per_bin=4))
    projected = rasterizer.transform(mesh).verts_packed().cpu().numpy()
    fragments = rasterizer(mesh); torch.cuda.synchronize()
    indices = fragments.pix_to_face[0, ..., 0].cpu().numpy(); depth = fragments.zbuf[0, ..., 0].cpu().numpy()
    if (indices.shape != (HEIGHT, WIDTH) or np.any(indices < 0) or np.any(indices >= 4)
            or depth.dtype != np.float32 or not np.isfinite(depth).all() or np.any(depth <= 0)):
        raise RuntimeError("Independent physical background/full-grid CUDA contract failed")
    references = {"original_fp64": vertices, "cast_fp32_to_fp64": vertices.astype(np.float32).astype(np.float64),
                  "actual_projected_fp32_recovered_fp64": recovered_projection(projected)}
    result = {}
    for index, label in enumerate(LABELS):
        yy, xx = np.nonzero(indices == index); xy = np.column_stack((xx, yy)); values = depth[yy, xx]
        result[label] = {"selected_pixels": len(xy), "references": {}}
        for ref, v in references.items():
            triangle = v[faces[index]]
            result[label]["references"][ref] = {"geometry": face_geometry(triangle), "ray_offsets": {
                str(offset): reference_summary(triangle, xy, values, offset) if len(xy) else {"pixels": 0}
                for offset in (0., .5, 1.)}}
    return {"faces": result, "full_grid_pixels": WIDTH*HEIGHT, "all_pixels_finite_positive": True,
        "fixture_identity": {"vertices_shape": list(vertices.shape), "faces_shape": list(faces.shape),
            "vertices_fp64_bytes_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
            "faces_int64_bytes_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "camera_fp64_bytes_sha256": hashlib.sha256(K.tobytes()).hexdigest()},
        "grid": {"width": WIDTH, "height": HEIGHT, "fixed_focal_pixels": 800}}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); rev = os.environ.get("WR_CODE_REVISION", "")
    code = root/"jobs"/rev/"run_raster_precision_probe/code"; out = root/"validation/raster_precision_probe_v1"
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or not re.fullmatch(r"[0-9a-f]{40}", rev) or os.environ.get("WR_IMAGE_ID") != IMAGE
            or Path(__file__).resolve() != code/"infra/raster_precision_probe.py"
            or out.is_symlink() or not out.is_dir() or any(out.iterdir())):
        raise ValueError("Fresh independent diagnostic in actual frozen Linux/network-none runtime required")
    paths = ("infra/raster_precision_probe.py", "infra/run_raster_precision_probe.sh")
    before = {p: identity(code/p) for p in paths}; started = time.perf_counter()
    report = {"schema": "world_reward.raster_precision_probe.v1", "stage": "independent_projection_conditioning",
        "status": "fail", "phase": "runtime", "producer_revision": rev, "image_id": IMAGE,
        "source_helpers": before, "script_sha256": before["infra/raster_precision_probe.py"]["sha256"],
        "budget_seconds": 30, "challenge_inputs_used": False,
        "failed_cohort_read": False, "models_used": False, "validation_performed": False,
        "downstream_gates_changed": False, "arrays_or_media_saved": False}
    receipt = out/"report.json"
    def persist():
        report["elapsed_seconds"] = time.perf_counter()-started
        with receipt.open("w") as stream:
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    def expired(*unused): raise TimeoutError("Frozen30s diagnostic budget exceeded")
    old_alarm, old_term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired)
    signal.alarm(30)
    try:
        persist(); import torch; import pytorch3d
        if (not torch.cuda.is_available() or torch.__version__ != "2.5.1+cu124"
                or pytorch3d.__version__ != "0.7.9" or np.__version__ != "1.26.3"):
            raise RuntimeError("Frozen actual CUDA/Torch/PyTorch3D/NumPy runtime required")
        report.update(torch=torch.__version__, pytorch3d=pytorch3d.__version__, numpy=np.__version__,
                      device=torch.cuda.get_device_name(), phase="measurement")
        with torch.inference_mode(): report.update(run_cuda(torch))
        if before != {p: identity(code/p) for p in paths}: raise ValueError("Frozen diagnostic source changed")
        report.update(status="pass", phase="complete", inference_quality_claim=False)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:300]); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term)
        persist(); receipt.chmod(0o400)


if __name__ == "__main__": main()

"""Bounded FP64 pixel-centre triangle renderer and a NEW independent microgate.

Camera-space triangle soup is supported, not certified solids/contact. Rendering
never fits geometry, camera or scale. The CPU reference has no model/data input;
optional actual CUDA raster measurements are diagnostics, never its authority.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

BUDGET = 100
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
MAX_WIDTH, MAX_HEIGHT, MAX_FACES, MAX_VERTICES = 640, 480, 100_000, 100_000
MAX_CANDIDATES, CHUNK = 20_000_000, 8192
Z_MIN, Z_MAX, REFERENCE_ATOL_M = 1e-4, 10_000., 1e-8
SOURCE_FILES = ("infra/triangle_ray_gate.py", "infra/run_triangle_ray_gate.sh")


def _deadline(value):
    now = time.monotonic()
    if value is None:
        return now + BUDGET
    if type(value) is not float or not np.isfinite(value) or not now < value <= now + BUDGET:
        raise ValueError("An unexpired absolute monotonic deadline within100s required")
    return value


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Bounded triangle rendering deadline exceeded")


def _inputs(vertices, faces, K, width, height):
    if any(np.ma.isMaskedArray(x) for x in (vertices, faces, K)):
        raise ValueError("Explicit unmasked triangles/camera required")
    v, f, k = map(np.asarray, (vertices, faces, K))
    if (type(width) is not int or type(height) is not int or not 2 <= width <= MAX_WIDTH
            or not 2 <= height <= MAX_HEIGHT or v.dtype != np.float64 or v.ndim != 2
            or v.shape[1:] != (3,) or not 3 <= len(v) <= MAX_VERTICES or not np.isfinite(v).all()
            or np.any(v[:, 2] <= Z_MIN) or np.any(v[:, 2] > Z_MAX)
            or f.dtype != np.int64 or f.ndim != 2 or f.shape[1:] != (3,)
            or not 1 <= len(f) <= MAX_FACES or np.any(f < 0) or np.any(f >= len(v))
            or k.dtype != np.float64 or k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.array_equal(k[2], [0., 0., 1.]) or k[0, 1] != 0 or k[1, 0] != 0
            or k[0, 2] != width/2 or k[1, 2] != height/2
            or not max(width, height)/16 <= k[0, 0] <= 16*max(width, height)
            or not max(width, height)/16 <= k[1, 1] <= 16*max(width, height)):
        raise ValueError("Bounded FP64 positive camera triangles, I64 faces and centred pinhole K required")
    triangles = v[f]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    if not np.isfinite(normals).all() or np.any(np.max(np.abs(normals), axis=1) == 0):
        raise ValueError("Every supplied triangle must be nondegenerate; no deletion/repair")
    uv = v[:, :2]/v[:, 2, None]*[k[0, 0], k[1, 1]] + k[:2, 2]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e9:
        raise ValueError("Finite bounded original projection required")
    return v, f, k, uv


def render_triangles(vertices, faces, K, width, height, *, labels=None, vertex_colors=None, deadline=None):
    """Nearest positive camera-Z at (x+.5,y+.5), with no back-face culling.

    Screen edge barycentrics become perspective-correct 3D barycentrics through
    division by vertex Z. Projected bounding boxes and8192-pixel chunks bound
    work/memory; >20M candidate tests or deadline stops, never removes geometry.
    Zero-area projections have no surface-interior coverage. Exact computed Z
    ties choose the first original face. Misses are NaN/-1, not filled evidence.
    Optional colors are interpolated linear RGB, not a photorealistic shader.
    """
    deadline = _deadline(deadline)
    v, f, k, uv = _inputs(vertices, faces, K, width, height)
    lab = None if labels is None else np.asarray(labels)
    color = None if vertex_colors is None else np.asarray(vertex_colors)
    if labels is not None and (np.ma.isMaskedArray(labels) or lab.shape != (len(f),)
            or lab.dtype != np.int64 or np.any(lab < 0) or np.any(lab > 2**31-1)):
        raise ValueError("Original I64 nonnegative per-face labels required")
    if vertex_colors is not None and (np.ma.isMaskedArray(vertex_colors) or color.shape != v.shape
            or color.dtype != np.float64 or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1)):
        raise ValueError("Original finite FP64 linear vertex RGB in0..1 required")
    depth = np.full((height, width), np.nan)
    face_index = np.full((height, width), -1, np.int64)
    bary = np.full((height, width, 3), np.nan)
    tested, edge_on = 0, 0
    for face_id, indices in enumerate(f):
        _check_time(deadline)
        a, b, c = uv[indices]
        ab, ac = b-a, c-a
        area = ab[0]*ac[1]-ab[1]*ac[0]
        if area == 0:
            edge_on += 1
            continue
        low, high = np.min(uv[indices], axis=0), np.max(uv[indices], axis=0)
        x0, y0 = np.maximum(np.ceil(low-.5), [0, 0]).astype(np.int64)
        x1, y1 = np.minimum(np.floor(high-.5), [width-1, height-1]).astype(np.int64)
        if x1 < x0 or y1 < y0:
            continue
        columns = int(x1-x0+1); candidates = columns*int(y1-y0+1)
        tested += candidates
        if tested > MAX_CANDIDATES:
            raise ValueError("Candidate work ceiling exceeded; no selective triangle removal")
        for start in range(0, candidates, CHUNK):
            _check_time(deadline)
            flat = np.arange(start, min(start+CHUNK, candidates))
            x, y = flat % columns+x0, flat // columns+y0
            px, py = x+.5-a[0], y+.5-a[1]
            u = (px*ac[1]-py*ac[0])/area
            w = (ab[0]*py-ab[1]*px)/area
            screen = np.column_stack((1-u-w, u, w))
            inside = (screen >= 0).all(axis=1)
            if not inside.any():
                continue
            x, y, screen = x[inside], y[inside], screen[inside]
            reciprocal = screen/v[indices, 2]
            total = reciprocal.sum(axis=1)
            z = np.full(len(total), v[indices[0], 2]) if np.all(v[indices, 2] == v[indices[0], 2]) else 1/total
            if not np.isfinite(z).all() or np.any(z <= Z_MIN):
                raise ValueError("Nonfinite/nonpositive covered ray intersection")
            keep = np.isnan(depth[y, x]) | (z < depth[y, x])
            depth[y[keep], x[keep]] = z[keep]
            face_index[y[keep], x[keep]] = face_id
            bary[y[keep], x[keep]] = reciprocal[keep]/total[keep, None]
    result = dict(depth=depth, face_index=face_index, barycentric=bary,
                  candidate_tests=tested, zero_area_projected_faces=edge_on)
    hit = face_index >= 0
    if lab is not None:
        image = np.full((height, width), -1, np.int64); image[hit] = lab[face_index[hit]]
        result["labels"] = image
    if color is not None:
        rgb = np.full((height, width, 3), np.nan)
        rgb[hit] = np.sum(color[f[face_index[hit]]]*bary[hit, :, None], axis=1)
        result["rgb"] = rgb
    _check_time(deadline)
    return result


def micro_fixture(width=640, height=480):
    """New wide/skinny and rear-first layered faces, unlike the earlier probe."""
    k = np.array([[1.25*width, 0., width/2], [0., 1.25*width, height/2], [0., 0., 1.]])
    sx, sy = width/640, height/480
    skinny_x = np.floor(.735*width)+.5
    uv = [[73.25*sx, 54.25*sy], [371.25*sx, 81.25*sy], [150.25*sx, 381.25*sy],
          [skinny_x-.0003, 90.125*sy], [skinny_x+.0003, 90.125*sy], [skinny_x, 365.875*sy]]
    z = [2.2, 3.7, 2.8, 1.7, 2.3, 3.1]
    rectangle = np.array([[395.25*sx, 255.25*sy], [586.25*sx, 255.25*sy],
                          [586.25*sx, 420.25*sy], [395.25*sx, 420.25*sy]])
    uv.extend(rectangle.tolist()*2); z.extend([4.25]*4+[2.55]*4)
    uv.extend([[-32.25, -32.25], [width+32.25, -32.25], [width+32.25, height+32.25], [-32.25, height+32.25]])
    z.extend([6.3]*4)
    z = np.array(z); uv = np.array(uv)
    v = np.column_stack(((uv-k[:2, 2])*z[:, None]/[k[0, 0], k[1, 1]], z))
    f = np.array([[0, 2, 1], [3, 5, 4], [6, 7, 8], [6, 8, 9], [10, 11, 12], [10, 12, 13],
                  [14, 15, 16], [14, 16, 17]], np.int64)
    return v, f, k, np.array([11, 12, 13, 13, 14, 14, 15, 15], np.int64)


def independent_reference(vertices, faces, K, width, height, deadline):
    """Full-grid Moller-Trumbore, separate from projected-edge renderer math."""
    yy, xx = np.mgrid[:height, :width]
    rays = np.column_stack(((xx.ravel()+.5-K[0, 2])/K[0, 0],
                            (yy.ravel()+.5-K[1, 2])/K[1, 1], np.ones(width*height)))
    depth = np.full(len(rays), np.nan); ids = np.full(len(rays), -1, np.int64)
    for i, (p, b, c) in enumerate(vertices[faces]):
        _check_time(deadline)
        e1, e2 = b-p, c-p; q = np.cross(-p, e1)
        h = np.cross(rays, e2); det = h @ e1
        good = det != 0
        u = np.divide(h @ -p, det, out=np.full(len(rays), np.nan), where=good)
        w = np.divide(rays @ q, det, out=np.full(len(rays), np.nan), where=good)
        z = np.divide(e2 @ q, det, out=np.full(len(rays), np.nan), where=good)
        hit = good & (u >= 0) & (w >= 0) & (u+w <= 1) & (z > Z_MIN)
        keep = hit & (np.isnan(depth) | (z < depth))
        depth[keep], ids[keep] = z[keep], i
    return depth.reshape(height, width), ids.reshape(height, width)


def decimal_depth(triangle, ray):
    """80-digit plane intersection from exact binary64 inputs, no raster data."""
    with localcontext() as ctx:
        ctx.prec = 80
        p, b, c = [[Decimal.from_float(float(x)) for x in row] for row in triangle]
        d = [Decimal.from_float(float(x)) for x in ray]
        e, f = [b[i]-p[i] for i in range(3)], [c[i]-p[i] for i in range(3)]
        n = [e[1]*f[2]-e[2]*f[1], e[2]*f[0]-e[0]*f[2], e[0]*f[1]-e[1]*f[0]]
        return float(sum(n[i]*p[i] for i in range(3))/sum(n[i]*d[i] for i in range(3)))


def validate_micro_reference(width=640, height=480, *, deadline=None):
    deadline = _deadline(deadline); v, f, k, labels = micro_fixture(width, height)
    result = render_triangles(v, f, k, width, height, labels=labels, deadline=deadline)
    expected, ids = independent_reference(v, f, k, width, height, deadline)
    if not np.isfinite(result["depth"]).all() or not np.isfinite(expected).all():
        raise ValueError("New full-grid physical background failed")
    label_error = int(np.count_nonzero(result["labels"] != labels[ids]))
    error = float(np.max(np.abs(expected-result["depth"])))
    yy, xx = np.mgrid[:height, :width]; triangles = v[f[result["face_index"]]]
    normals = np.cross(triangles[..., 1, :]-triangles[..., 0, :], triangles[..., 2, :]-triangles[..., 0, :])
    rays = np.stack(((xx+.5-k[0, 2])/k[0, 0], (yy+.5-k[1, 2])/k[1, 1], np.ones_like(xx)), axis=-1)
    plane = np.sum(normals*triangles[..., 0, :], axis=-1)/np.sum(normals*rays, axis=-1)
    plane_error = float(np.max(np.abs(plane-result["depth"])))
    decimal_errors = []; counts = {}
    for label in (11, 12, 14, 15):
        indices = np.flatnonzero(result["labels"].ravel() == label); counts[str(label)] = len(indices)
        if len(indices) < 2:
            raise ValueError("Every new microfixture stratum must have support")
        for index in indices[np.linspace(0, len(indices)-1, 5, dtype=int)]:
            y, x = divmod(int(index), width); face = result["face_index"][y, x]
            decimal_errors.append(abs(decimal_depth(v[f[face]], rays[y, x])-result["depth"][y, x]))
    bounds = np.array([[395.25*width/640, 255.25*height/480], [586.25*width/640, 420.25*height/480]])
    layer = (xx+.5 > bounds[0, 0]) & (xx+.5 < bounds[1, 0]) & (yy+.5 > bounds[0, 1]) & (yy+.5 < bounds[1, 1])
    # The nearer skinny face may cross this rectangle: it must remain nearer.
    expected_layer = layer & (result["labels"] != 12)
    layers_ok = bool(np.all(result["labels"][expected_layer] == 14) and not np.any(result["labels"] == 13))
    maximum = float(max(decimal_errors))
    if label_error or not layers_ok or max(error, plane_error, maximum) > REFERENCE_ATOL_M:
        raise ValueError("Independent full-grid/Decimal/nearest-layer reference gate failed")
    signature = hashlib.sha256(v.tobytes()+f.tobytes()+k.tobytes()+labels.tobytes()).hexdigest()
    return result, dict(width=width, height=height, fixture_sha256=signature, full_grid_pixels=width*height,
        independent_label_disagreements=label_error, full_grid_ray_max_Z_error_m=error,
        full_grid_plane_max_Z_error_m=plane_error, Decimal80_samples=len(decimal_errors),
        Decimal80_max_Z_error_m=maximum, rear_first_nearest_layer_verified=layers_ok,
        stratum_pixels=counts, threshold_m=REFERENCE_ATOL_M, candidate_tests=result["candidate_tests"])


def cuda_diagnostics(torch, cpu):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    v, f, k, labels = micro_fixture()
    camera = cameras_from_opencv_projection(R=torch.eye(3, device="cuda")[None],
        tvec=torch.zeros((1, 3), device="cuda"), camera_matrix=torch.tensor(k, device="cuda", dtype=torch.float32)[None],
        image_size=torch.tensor([[480, 640]], device="cuda", dtype=torch.float32))
    mesh = Meshes(verts=[torch.tensor(v, device="cuda", dtype=torch.float32)], faces=[torch.tensor(f, device="cuda")])
    settings = RasterizationSettings(image_size=(480, 640), blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, clip_barycentric_coords=False, cull_backfaces=False,
        cull_to_frustum=False, z_clip_value=None, max_faces_per_bin=len(f))
    with torch.inference_mode():
        fragment = MeshRasterizer(cameras=camera, raster_settings=settings)(mesh)
    torch.cuda.synchronize(); ids = fragment.pix_to_face[0, ..., 0].cpu().numpy(); z = fragment.zbuf[0, ..., 0].cpu().numpy()
    valid = ids >= 0; measured_labels = np.full(ids.shape, -1); measured_labels[valid] = labels[ids[valid]]
    rows = []
    for label in (11, 12, 14, 15):
        region = (cpu["labels"] == label) & valid & np.isfinite(z)
        error = np.abs(cpu["depth"][region]-z[region])
        rows.append(dict(label=label, paired_pixels=int(region.sum()),
                         max_Z_error_m=float(error.max()) if len(error) else None,
                         p99_Z_error_m=float(np.quantile(error, .99)) if len(error) else None))
    return dict(diagnostics_only=True, affects_reference_gate=False, missing_pixels=int((~valid).sum()),
                label_disagreements=int((measured_labels != cpu["labels"]).sum()), strata=rows)


def identity(path):
    p = Path(path); raw = p.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); rev = os.environ.get("WR_CODE_REVISION", "")
    code = root/"jobs"/rev/"run_triangle_ray_gate/code"; out = root/"validation/triangle_ray_gate_v1"
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward")
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.geteuid() != 1000
            or not re.fullmatch(r"[0-9a-f]{40}", rev) or os.environ.get("WR_IMAGE_ID") != IMAGE
            or Path(__file__).resolve() != code/"infra/triangle_ray_gate.py"
            or out.resolve() != out or not out.is_dir() or out.stat().st_mode & 0o777 != 0o700 or any(out.iterdir())):
        raise ValueError("Fresh frozen Azure/network-none private microgate required")
    before = {name: identity(code/name) for name in SOURCE_FILES}
    started = time.perf_counter(); deadline = time.monotonic()+BUDGET
    report = dict(schema="world_reward.triangle_ray_gate.v1", stage="new_independent_CPU_triangle_reference",
        status="fail", phase="CPU_reference", producer_revision=rev, image_id=IMAGE, source_helpers=before,
        script_sha256=before[SOURCE_FILES[0]]["sha256"], budget_seconds=BUDGET,
        models_used=False, challenge_inputs_used=False, failed_cohort_read=False, arrays_or_media_saved=False,
        ground_truth_used_for_inference=False, downstream_gates_changed=False, previous_failures_reinterpreted=False,
        reference_scope="camera_triangle_soup_not_physical_contact_or_embedding", adoption_performed=False,
        inference_quality_verified=False, CPU_reference_gate_passed=False)
    fd = os.open(out/"report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as receipt:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            receipt.seek(0); json.dump(report, receipt, allow_nan=False); receipt.write("\n")
            receipt.truncate(); receipt.flush(); os.fsync(receipt.fileno())
        def expired(*unused): raise TimeoutError("Frozen100s microgate exceeded")
        old_alarm, old_term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        try:
            persist(); cpu, evidence = validate_micro_reference(deadline=deadline)
            report.update(CPU_reference_gate_passed=True, CPU_reference=evidence, phase="CUDA_diagnostics"); persist()
            import torch
            import pytorch3d
            if (np.__version__ != "1.26.3" or torch.__version__ != "2.5.1+cu124"
                    or pytorch3d.__version__ != "0.7.9" or not torch.cuda.is_available()):
                raise RuntimeError("Frozen actual NumPy/Torch/PyTorch3D/CUDA diagnostic runtime required")
            report.update(numpy=np.__version__, torch=torch.__version__, pytorch3d=pytorch3d.__version__,
                          gpu=torch.cuda.get_device_name(), CUDA_diagnostics=cuda_diagnostics(torch, cpu))
            if before != {name: identity(code/name) for name in SOURCE_FILES}:
                raise ValueError("Original microgate source changed")
            _check_time(deadline)
            report.update(status="pass", phase="complete", sources_rehashed_after=True)
        except Exception as exc:
            report.update(error_type=type(exc).__name__)
            raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "CPU_reference_gate_passed")}))


if __name__ == "__main__":
    main()

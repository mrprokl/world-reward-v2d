"""New analytic object RGBD reference; CPU Azure only, never model inference.

Stable ray/quadric hits generate RGB and private camera Z together. Independent
completed-square and Decimal gates precede publication of the whole cohort.
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

PROTOCOL_SHA256 = "4ff1631da8a59c4bdf075e2ba59be7d7196f1608ff43c157c851c40bafd3c7c4"
WIDTH, HEIGHT = 640, 480
K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
SOURCE_FILES = ("infra/analytic_rgbd_render.py", "infra/run_analytic_rgbd_render.sh")


def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path)}


def protocol(path):
    if Path(path).is_symlink() or digest(path) != PROTOCOL_SHA256:
        raise ValueError("Exact entire preregistered protocol required; no rescue or retuning")
    return json.loads(Path(path).read_text())


def pose(record, frame):
    if type(frame) is not int or frame not in range(4): raise ValueError("Original frame0..3 required")
    instant = frame/3
    axis = np.array([.2, 1., .3]); axis /= np.linalg.norm(axis)
    x, y, z = axis; skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = record["start_angle_rad"]+record["angle_extent_rad"]*instant
    rotation = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)
    translation = np.array([-.04+.08*instant, .015*np.sin(2*np.pi*instant), record["depth_m"]+.06*instant])
    return rotation, translation


def geometry_gate(semiaxes, rotation, translation, plane_z):
    a, r, t = map(np.asarray, (semiaxes, rotation, translation))
    if (a.shape != (3,) or r.shape != (3, 3) or t.shape != (3,)
            or any(v.dtype != np.float64 or not np.isfinite(v).all() for v in (a, r, t))
            or np.any(a <= 0) or not np.isfinite(plane_z)
            or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-14)
            or abs(np.linalg.det(r)-1) > 1e-14):
        raise ValueError("Positive finite closed convex ellipsoid and proper rigid pose required")
    radius = float(a.max()); minimum = float(t[2]-radius)
    bounds = (np.abs(t[:2])+radius)/minimum*800
    if minimum <= .01 or t[2]+radius >= plane_z or np.any(bounds >= [312, 232]):
        raise ValueError("Whole solid must be in frustum, in front of plane, outside camera")
    return {"closed_convex_positive_implicit_ellipsoid": True,
            "volume_m3": float(4*np.pi*np.prod(a)/3), "minimum_camera_z_bound_m": minimum}


def rays(width=WIDTH, height=HEIGHT):
    yy, xx = np.mgrid[:height, :width]
    return np.stack(((xx+.5-320)/800, (yy+.5-240)/800, np.ones_like(xx)), axis=-1)


def intersections(semiaxes, rotation, translation, plane_z, direction):
    """Stable FP64 primary roots plus algebraically separate reference checks."""
    a, r, t, d = map(np.asarray, (semiaxes, rotation, translation, direction))
    if d.dtype != np.float64 or d.ndim < 2 or d.shape[-1] != 3 or not np.isfinite(d).all() or np.any(d[..., 2] != 1):
        raise ValueError("Finite FP64 +.5 camera-Z parameterized pixel rays required")
    origin = (-t @ r)/a; local = (d @ r)/a
    aa = np.sum(local*local, axis=-1); bb = 2*np.sum(local*origin, axis=-1); cc = float(origin @ origin-1)
    disc = bb*bb-4*aa*cc; scale = np.maximum(np.maximum(bb*bb, np.abs(4*aa*cc)), 1)
    margin = float(np.min(np.abs(disc)/scale))
    if cc <= 0 or np.any(aa <= 0) or margin <= 1e-12:
        raise ValueError("Outside positive quadric and nonambiguous discriminant required; no dropped pixels")
    sqrt = np.sqrt(np.maximum(disc, 0)); q = -.5*(bb+np.copysign(sqrt, bb))
    if np.any(q == 0): raise ValueError("Degenerate stable quadratic q")
    first, second = q/aa, cc/q
    first, second = np.minimum(first, second), np.maximum(first, second)
    visible = (disc > 0) & (first > .01) & (first < plane_z)
    # Independent closest-point construction: ray foot in normalized solid.
    closest = -np.sum(local*origin, axis=-1)/aa
    foot = origin+closest[..., None]*local
    chord = (1-np.sum(foot*foot, axis=-1))/aa
    reference = closest-np.sqrt(np.maximum(chord, 0))
    expected_visible = (chord > 0) & (reference > .01) & (reference < plane_z)
    root_error = float(np.max(np.abs(first[visible]-reference[visible]))) if visible.any() else 0.
    points = origin+first[..., None]*local
    residual = float(np.max(np.abs(np.sum(points[visible]**2, axis=-1)-1))) if visible.any() else 0.
    if (not np.array_equal(visible, expected_visible) or root_error > 1e-10 or residual > 5e-12
            or np.any(2*aa[visible]*first[visible]+bb[visible] >= 0)
            or np.any(second[visible] <= first[visible])):
        raise ValueError("Independent silhouette, implicit residual or nearest entering root gate failed")
    depth64 = np.where(visible, first, plane_z); depth = depth64.astype(np.float32)
    cast_error = float(np.max(np.abs(depth64-depth)))
    if not np.isfinite(depth).all() or np.any(depth <= .01) or cast_error > 1e-6:
        raise ValueError("Finite positive full-grid FP32 camera depth cast gate failed")
    checked = {"discriminant_min_relative_margin": margin, "independent_root_max_error_m": root_error,
        "independent_implicit_residual_max": residual, "fp32_depth_cast_max_error_m": cast_error,
        "independent_discriminant_silhouette_verified": True, "first_hit_entering_derivative": True,
        "checked_pixels": int(depth.size), "visible_pixels": int(visible.sum())}
    return depth64, depth, visible, checked


def decimal_gate(a, rotation, translation, plane_z, depth64, visible, pixels):
    """Nine preregistered pixels, Decimal70 matrix/quadric arithmetic from FP64 inputs."""
    def dec(value): return Decimal.from_float(float(value))
    maximum = 0.
    with localcontext() as context:
        context.prec = 70
        axes = list(map(dec, a)); rot = [[dec(v) for v in row] for row in rotation]; t = list(map(dec, translation))
        for x, y in pixels:
            d = [(Decimal(x)+Decimal('.5')-320)/800, (Decimal(y)+Decimal('.5')-240)/800, Decimal(1)]
            o = [-sum(t[i]*rot[i][j] for i in range(3))/axes[j] for j in range(3)]
            ray = [sum(d[i]*rot[i][j] for i in range(3))/axes[j] for j in range(3)]
            aa = sum(v*v for v in ray); bb = 2*sum(u*v for u, v in zip(o, ray)); cc = sum(v*v for v in o)-1
            discriminant = bb*bb-4*aa*cc; z = dec(plane_z); hit = False
            if discriminant > 0:
                nearest = (-bb-discriminant.sqrt())/(2*aa)
                hit = Decimal('.01') < nearest < z
                if hit: z = nearest
            error = abs(float(z)-float(depth64[y, x])); maximum = max(maximum, error)
            if hit != bool(visible[y, x]) or error > 1e-10:
                raise ValueError("Predetermined Decimal70 visibility/depth probe gate failed")
    return {"decimal_precision": 70, "decimal_probes_checked": len(pixels), "decimal_root_max_error_m": maximum}


def appearance(record, rotation, translation, depth64, visible, direction):
    points = direction*depth64[..., None]; local = (points-translation) @ rotation
    normals = (local/np.asarray(record["semiaxes_m"])**2) @ rotation.T
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(visible[..., None], normals, [0., 0., -1.])
    frequency = np.array([[23., 11., 7.], [9., 27., -8.], [13., -5., 31.]])
    phases = [np.random.Generator(np.random.PCG64(record["material_seed"]+offset)).uniform(0, 2*np.pi, 3) for offset in (0, 1000)]
    color = .44+.25*np.sin(local @ frequency.T+phases[0])
    background = .58+.19*np.sin(points @ (frequency*.6).T+phases[1])
    color = np.where(visible[..., None], color, background)
    light = np.asarray(record["light_m"])-points; light /= np.linalg.norm(light, axis=-1, keepdims=True)
    lambert = np.maximum(np.sum(normals*light, axis=-1), 0)
    rgb = np.rint(np.clip(color*(.45+.55*lambert[..., None]), 0, 1)*255).astype(np.uint8)
    if rgb.shape != (*depth64.shape, 3): raise ValueError("Exact RGB pixel grid required")
    return rgb


def name(scene, frame): return f"scene_{scene:06d}_frame_{frame:06d}"


def public_manifest(hashes):
    if len(hashes) != 12 or any(type(h) is not str or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("All12 original RGB hashes required before public manifest")
    return {"schema": "world_reward.analytic_rgbd_public.v1", "images": [
        {"scene_id": s, "frame_id": f, "file": name(s, f)+".png", "sha256": hashes[(s-1)*4+f],
         "width": WIDTH, "height": HEIGHT} for s in range(1, 4) for f in range(4)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True, type=Path); args = parser.parse_args(argv); frozen = protocol(args.protocol)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Azure CPU Linux runtime with network none required")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or image != frozen["manufacture"]["runtime_image"]:
        raise ValueError("Immutable actual source/image must match preregistered runtime")
    root = Path(os.environ["WR_ROOT"]); code = root/"jobs"/revision/"run_analytic_rgbd_render"/"code"
    if (Path(__file__).resolve() != code/SOURCE_FILES[0] or args.protocol != code/"configs/analytic_rgbd_protocol.json"
            or any(p.is_symlink() for p in (code, *code.parents, *(code/p for p in SOURCE_FILES)))):
        raise ValueError("Exact immutable producer code/protocol namespace required")
    source_helpers = {p: identity(code/p) for p in SOURCE_FILES}; destination = root/frozen["namespace"]
    if (destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())
            or os.environ.get("WR_RENDER_OUTPUT_RESERVED") != "1"):
        raise FileExistsError("Exclusive new reserved empty analytic cohort required")
    public, private = destination/"inputs", destination/"eval_private"; public.mkdir(); private.mkdir(mode=0o700)
    receipt = private/"render-report.json"; started = time.perf_counter()
    report = {"schema": "world_reward.analytic_rgbd_manufacture.v1", "status": "fail", "phase": "runtime_preflight",
        "stage": "analytic_object_rgbd_manufacture", "source_helpers": source_helpers, "producer_revision": revision,
        "image_id": image, "script_sha256": digest(__file__), "protocol_sha256": PROTOCOL_SHA256,
        "protocol_identity": {"file": "configs/analytic_rgbd_protocol.json", **identity(args.protocol)},
        "budget_seconds": 360, "device": "cpu", "network": "none", "challenge_inputs_used": False,
        "inference_performed": False, "synthetic_transfer_only": True, "cases": [], "private_files": {}}
    def persist():
        report["elapsed_seconds"] = time.perf_counter()-started
        with receipt.open("w") as stream:
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    def expired(*unused): raise TimeoutError("Frozen360s manufacture budget exceeded")
    old_alarm, old_term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(360)
    try:
        persist()
        from PIL import Image
        if np.__version__ != "1.26.3": raise RuntimeError("Frozen actual NumPy runtime required")
        direction = rays(); hashes = []; report.update(numpy=np.__version__, phase="geometry_and_render")
        for record in frozen["objects"]:
            scene = record["scene_id"]; axes = np.asarray(record["semiaxes_m"], np.float64)
            extent = np.array([2., 1.6]); plane_z = record["background_z_m"]
            if np.any(np.abs(direction[..., :2]*plane_z) >= extent): raise ValueError("Finite physical plane must fill full grid")
            geometry_path = private/f"scene_{scene:06d}_geometry.npz"
            with geometry_path.open("xb") as stream:
                np.savez_compressed(stream, semiaxes_m=axes, material_seed=np.array(record["material_seed"], np.int64),
                    plane_z_m=np.array(plane_z, np.float64), plane_extent_xy_m=extent)
            geometry_identity = identity(geometry_path)
            for frame in range(4):
                rotation, translation = pose(record, frame); checked = geometry_gate(axes, rotation, translation, plane_z)
                depth64, depth, visible, ray_checks = intersections(axes, rotation, translation, plane_z, direction)
                checked.update(ray_checks)
                if visible.sum() < 1024: raise ValueError("Minimum whole-object visible pixels required")
                checked.update(decimal_gate(axes, rotation, translation, plane_z, depth64, visible,
                    frozen["manufacture_gates"]["decimal_probe_pixels_xy"]))
                rgb = appearance(record, rotation, translation, depth64, visible, direction); stem = name(scene, frame)
                with (private/(stem+".npz")).open("xb") as stream:
                    np.savez_compressed(stream, depth=depth, visible=visible, K=K, scene_id=np.array(scene, np.int64),
                        frame_id=np.array(frame, np.int64), camera_R=rotation, camera_t=translation)
                with (public/(stem+".png")).open("xb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                if identity(geometry_path) != geometry_identity: raise ValueError("Clip-constant geometry bytes changed")
                hashes.append(digest(public/(stem+".png")))
                report["cases"].append({"scene_id": scene, "frame_id": frame, "status": "pass", **checked}); persist()
                print(json.dumps({"rendered_scene": scene, "frame": frame}), flush=True)
        with (public/"manifest.json").open("x") as stream: json.dump(public_manifest(hashes), stream, indent=2)
        for key, folder in (("private_files", private), ("public_files", public)):
            report[key] = {p.name: identity(p) for p in sorted(folder.iterdir()) if p != receipt}
        if (len(report["private_files"]) != 15 or len(report["public_files"]) != 13 or protocol(args.protocol) != frozen
                or source_helpers != {p: identity(code/p) for p in SOURCE_FILES}): raise ValueError("Frozen full recipe/source/inventory changed")
        for folder, mode in ((public, 0o444), (private, 0o400)):
            for path in folder.iterdir():
                if path != receipt: path.chmod(mode)
        report.update(status="pass", phase="complete", public_manifest_sha256=digest(public/"manifest.json"))
    except Exception as exc:
        for folder in (public, private):
            for path in folder.iterdir():
                if path != receipt: path.unlink()
        report.update(error_type=type(exc).__name__, error=str(exc)[:300], arrays_removed=True, private_files={}, public_files={})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist(); receipt.chmod(0o400)


if __name__ == "__main__": main()

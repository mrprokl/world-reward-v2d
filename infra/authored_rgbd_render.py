"""Frozen new object-only RGBD manufacture; Azure CUDA only, never inference.

Public RGB is published only after every private geometry/pixel-ray gate. This
is synthetic camera-depth transfer evidence, not photorealism, HOI or a win.
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

PROTOCOL_SHA256 = "5d53806c3737aa1b15288628ba2fc364bbeb9dc95731075c2f3821ef5a2c053e"
WIDTH, HEIGHT = 640, 480
K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(path):
    if Path(path).is_symlink() or digest(path) != PROTOCOL_SHA256:
        raise ValueError("Exact preregistered complete protocol required; no rescue or retuning")
    return json.loads(Path(path).read_text())


def ellipsoid_mesh(semiaxes):
    """Closed inscribed convex sphere triangulation under positive affine scale."""
    a = np.asarray(semiaxes, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Three positive constant semiaxes required")
    latitude, longitude = 24, 48
    theta = np.repeat(np.arange(1, latitude)*np.pi/latitude, longitude)
    phi = np.tile(np.arange(longitude)*2*np.pi/longitude, latitude-1)
    directions = np.vstack(([0., 0., 1.], np.column_stack((np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi), np.cos(theta))), [0., 0., -1.]))
    faces = [(0, 1+j, 1+(j+1) % longitude) for j in range(longitude)]
    for ring in range(latitude-2):
        for j in range(longitude):
            p = 1+ring*longitude+j; q = 1+ring*longitude+(j+1) % longitude
            faces.extend(((p, p+longitude, q+longitude), (p, q+longitude, q)))
    last, south = 1+(latitude-2)*longitude, len(directions)-1
    faces.extend((last+j, south, last+(j+1) % longitude) for j in range(longitude))
    return directions*a, np.asarray(faces, dtype=np.int64)


def mesh_gate(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or v.dtype != np.float64 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or f.dtype != np.int64 or not len(f)
            or np.any(f < 0) or np.any(f >= len(v)) or len(np.unique(f)) != len(v)):
        raise ValueError("Finite active geometry and exact triangle arrays required")
    triangles = v[f]; normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    if (np.any(lengths <= 0) or np.any(counts != 2) or np.any(winding != 0)
            or len(v)-len(edges)+len(f) != 2 or not volume > 0):
        raise ValueError("Closed consistent positive-volume nondegenerate manifold gate failed")
    unit = normals/lengths[:, None]; supports = np.einsum("ij,ij->i", unit, triangles[:, 0])
    if np.any(supports <= 0) or np.max(unit @ v.T-supports[:, None]) > 1e-12:
        raise ValueError("Outward convex support-plane gate failed; no repair")
    return {"vertices": len(v), "faces": len(f), "volume_m3": volume,
            "closed_outward_convex_manifold": True}


def pose(record, frame):
    if type(frame) is not int or frame not in range(4):
        raise ValueError("Original frame0..3 required")
    instant = frame/3
    axis = np.array([.3, 1., .2]); axis /= np.linalg.norm(axis)
    x, y, z = axis; skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = record["start_angle_rad"]+record["angle_extent_rad"]*instant
    rotation = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)
    translation = np.array([-.045+.09*instant, .02*np.sin(2*np.pi*instant), record["depth_m"]+.05*instant])
    return rotation, translation


def rays():
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    return np.stack(((xx+.5-320)/800, (yy+.5-240)/800, np.ones_like(xx)), axis=-1)


def silhouette_gate(camera_vertices, visible):
    """Independent monotone-chain projected convex hull, no raster fragments."""
    points = sorted(set(map(tuple, camera_vertices[:, :2]/camera_vertices[:, 2, None]*800+[320, 240])))
    def cross(a, b, c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for ordered in (points, points[::-1]):
        half = []
        for p in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0: half.pop()
            half.append(p)
        halves.append(half[:-1])
    hull = np.asarray(halves[0]+halves[1]); yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    minimum = np.full((HEIGHT, WIDTH), np.inf)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        distance = ((b[0]-a[0])*(yy+.5-a[1])-(b[1]-a[1])*(xx+.5-a[0]))/np.linalg.norm(b-a)
        minimum = np.minimum(minimum, distance)
    interior, exterior = minimum > .0002, minimum < -.0002
    if not interior.any() or np.any(~visible[interior]) or np.any(visible[exterior]):
        raise ValueError("Independent convex-hull silhouette gate failed")
    return {"hull_interior_pixels": int(interior.sum()), "hull_exterior_pixels": int(exterior.sum()),
            "hull_boundary_tolerance_pixels": .0002}


def ray_gate(vertices, faces, face_indices, depth, object_faces):
    """Independent FP64 plane intersection and triangle containment at all pixels.

    Every reported object face must be entering/front-facing. For an outward
    convex solid this is the nearest surface, not an arbitrary back triangle.
    """
    if (depth.shape != (HEIGHT, WIDTH) or depth.dtype != np.float32
            or face_indices.shape != depth.shape or face_indices.dtype != np.int64
            or np.any(face_indices < 0) or np.any(face_indices >= len(faces))
            or not np.isfinite(depth).all() or np.any(depth <= .01)):
        raise ValueError("Full-grid finite positive FP32 camera Z and valid faces required")
    triangle = vertices[faces[face_indices]]; p, e1, e2 = triangle[..., 0, :], triangle[..., 1, :]-triangle[..., 0, :], triangle[..., 2, :]-triangle[..., 0, :]
    normal = np.cross(e1, e2); direction = rays()
    denominator = np.einsum("hwc,hwc->hw", direction, normal)
    if np.any(denominator == 0): raise ValueError("Parallel pixel ray selected a face")
    expected = np.einsum("hwc,hwc->hw", p, normal)/denominator
    delta = direction*expected[..., None]-p
    d00 = np.einsum("hwc,hwc->hw", e1, e1); d01 = np.einsum("hwc,hwc->hw", e1, e2); d11 = np.einsum("hwc,hwc->hw", e2, e2)
    d20 = np.einsum("hwc,hwc->hw", delta, e1); d21 = np.einsum("hwc,hwc->hw", delta, e2)
    determinant = d00*d11-d01*d01
    if np.any(determinant <= 0): raise ValueError("Reference triangle degeneracy")
    u, v = (d11*d20-d01*d21)/determinant, (d00*d21-d01*d20)/determinant
    minimum = float(np.min(np.stack((1-u-v, u, v))))
    error = float(np.max(np.abs(expected-depth)))
    visible = face_indices < object_faces
    if (error > .00002 or minimum < -.0002 or int(visible.sum()) < 1024
            or np.any(denominator[visible] >= 0)):
        raise ValueError("Independent nearest camera-ray/triangle gate failed")
    return {"camera_z_max_error_m": error, "minimum_barycentric": minimum,
            "checked_pixels": WIDTH*HEIGHT, "visible_pixels": int(visible.sum())}


def name(scene, frame):
    return f"scene_{scene:06d}_frame_{frame:06d}"


def public_manifest(hashes):
    if len(hashes) != 12 or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("All12 original RGB hashes required before public manifest")
    return {"schema": "world_reward.authored_rgbd_public.v1", "images": [
        {"scene_id": scene, "frame_id": frame, "file": name(scene, frame)+".png",
         "sha256": hashes[(scene-1)*4+frame], "width": WIDTH, "height": HEIGHT}
        for scene in range(1, 4) for frame in range(4)]}


def render(torch, vertices, faces, object_faces, record, rotation, translation):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    camera = cameras_from_opencv_projection(R=torch.eye(3, device="cuda")[None],
        tvec=torch.zeros((1, 3), device="cuda"), camera_matrix=torch.as_tensor(K, device="cuda", dtype=torch.float32)[None],
        image_size=torch.tensor([[HEIGHT, WIDTH]], device="cuda", dtype=torch.float32))
    v = torch.as_tensor(vertices, dtype=torch.float32, device="cuda"); f = torch.as_tensor(faces, device="cuda")
    mesh = Meshes(verts=[v], faces=[f])
    settings = RasterizationSettings(image_size=(HEIGHT, WIDTH), blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, clip_barycentric_coords=False, cull_backfaces=False,
        cull_to_frustum=False, z_clip_value=None, max_faces_per_bin=len(faces))
    fragments = MeshRasterizer(cameras=camera, raster_settings=settings)(mesh)
    indices = fragments.pix_to_face[0, ..., 0]
    if torch.any(indices < 0): raise ValueError("Physical background must fill every pixel")
    triangle = v[f[indices]]
    points = torch.sum(triangle*fragments.bary_coords[0, ..., 0, :, None], dim=-2)
    normals = torch.nn.functional.normalize(torch.cross(triangle[..., 1, :]-triangle[..., 0, :],
        triangle[..., 2, :]-triangle[..., 0, :], dim=-1), dim=-1)
    rotation = torch.as_tensor(rotation, device="cuda", dtype=torch.float32)
    translation = torch.as_tensor(translation, device="cuda", dtype=torch.float32)
    object_points = (points-translation) @ rotation
    phases = [np.random.Generator(np.random.PCG64(record["material_seed"]+offset)).uniform(0, 2*np.pi, 3) for offset in (0, 1000)]
    frequency = torch.tensor([[23., 11., 7.], [9., 27., -8.], [13., -5., 31.]], device="cuda")
    colors = .44+.25*torch.sin(object_points @ frequency.T+torch.as_tensor(phases[0], device="cuda", dtype=torch.float32))
    backdrop = .58+.19*torch.sin(points @ (frequency*.6).T+torch.as_tensor(phases[1], device="cuda", dtype=torch.float32))
    colors = torch.where((indices < object_faces)[..., None], colors, backdrop)
    light = torch.as_tensor(record["light_m"], device="cuda", dtype=torch.float32)
    lambert = torch.sum(normals*torch.nn.functional.normalize(light-points, dim=-1), dim=-1).clamp(0, 1)
    rgb = torch.round((colors*(.45+.55*lambert[..., None])).clamp(0, 1)*255).to(torch.uint8)
    torch.cuda.synchronize()
    return rgb.cpu().numpy(), indices.cpu().numpy(), fragments.zbuf[0, ..., 0].cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True, type=Path); args = parser.parse_args(argv)
    frozen = protocol(args.protocol)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Azure Linux runtime with network none required")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or image != frozen["manufacture"]["runtime_image"]:
        raise ValueError("Immutable actual source/image must match preregistered runtime")
    root = Path(os.environ["WR_ROOT"]); code = root/"jobs"/revision/"run_authored_rgbd_render"/"code"
    if (Path(__file__).resolve() != code/"infra/authored_rgbd_render.py"
            or args.protocol != code/"configs/authored_rgbd_protocol.json"
            or code.is_symlink()):
        raise ValueError("Exact immutable producer code/protocol namespace required")
    source_files = ("infra/authored_rgbd_render.py", "infra/run_authored_rgbd_render.sh")
    if any((code/p).is_symlink() for p in source_files): raise ValueError("No source symlink allowed")
    source_helpers = {p: {"bytes": (code/p).stat().st_size, "sha256": digest(code/p)} for p in source_files}
    destination = root/frozen["namespace"]
    if (destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())
            or os.environ.get("WR_RENDER_OUTPUT_RESERVED") != "1"):
        raise FileExistsError("Exclusive new reserved empty authored cohort required")
    public, private = destination/"inputs", destination/"eval_private"
    public.mkdir(); private.mkdir(mode=0o700)
    report = {"schema": "world_reward.authored_rgbd_manufacture.v1", "status": "fail",
        "stage": "authored_object_rgbd_manufacture", "source_helpers": source_helpers,
        "phase": "runtime_preflight", "producer_revision": revision, "image_id": image,
        "script_sha256": digest(__file__), "protocol_sha256": PROTOCOL_SHA256,
        "protocol_identity": {"file": "configs/authored_rgbd_protocol.json",
                              "bytes": args.protocol.stat().st_size, "sha256": PROTOCOL_SHA256},
        "budget_seconds": 360, "challenge_inputs_used": False, "inference_performed": False,
        "synthetic_transfer_only": True, "cases": [], "private_files": {}}
    started = time.perf_counter(); receipt = private/"render-report.json"
    def persist():
        report["elapsed_seconds"] = time.perf_counter()-started
        with receipt.open("w") as stream:
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    def expired(*unused): raise TimeoutError("Frozen360s manufacture budget exceeded")
    old_alarm, old_term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired)
    signal.alarm(360)
    try:
        persist()
        import torch
        import pytorch3d
        from PIL import Image
        if (not torch.cuda.is_available() or torch.__version__ != "2.5.1+cu124"
                or pytorch3d.__version__ != "0.7.9" or np.__version__ != "1.26.3"):
            raise RuntimeError("Frozen actual CUDA/Torch/PyTorch3D/NumPy runtime required")
        torch.cuda.synchronize()
        report.update(torch=torch.__version__, pytorch3d=pytorch3d.__version__, numpy=np.__version__,
                      device=torch.cuda.get_device_name(), phase="geometry_and_render")
        hashes = []
        for record in frozen["objects"]:
            scene = record["scene_id"]; vertices, faces = ellipsoid_mesh(record["semiaxes_m"])
            topology = mesh_gate(vertices, faces)
            plane = np.array([[-2.2, -1.7, record["background_z_m"]], [-2.2, 1.7, record["background_z_m"]],
                              [2.2, 1.7, record["background_z_m"]], [2.2, -1.7, record["background_z_m"]]])
            plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            mesh_path = private/f"scene_{scene:06d}_mesh.npz"
            with mesh_path.open("xb") as stream:
                np.savez_compressed(stream, vertices=vertices, faces=faces, semiaxes=np.array(record["semiaxes_m"]),
                    material_seed=np.array(record["material_seed"], np.int64), background_vertices=plane, background_faces=plane_faces)
            report.setdefault("meshes", []).append({"scene_id": scene, **topology})
            for frame in range(4):
                rotation, translation = pose(record, frame); camera = vertices @ rotation.T+translation
                pixels = camera[:, :2]/camera[:, 2, None]*800+[320, 240]
                if np.any(camera[:, 2] <= .01) or np.any(pixels < [8, 8]) or np.any(pixels > [WIDTH-8, HEIGHT-8]):
                    raise ValueError("Original whole-object camera/frustum gate failed")
                all_vertices = np.vstack((camera, plane)); all_faces = np.vstack((faces, plane_faces+len(vertices)))
                with torch.inference_mode():
                    rgb, indices, depth = render(torch, all_vertices, all_faces, len(faces), record, rotation, translation)
                checked = ray_gate(all_vertices, all_faces, indices, depth, len(faces))
                checked.update(silhouette_gate(camera, indices < len(faces)))
                if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8: raise ValueError("Original RGB grid required")
                stem = name(scene, frame)
                with (private/(stem+".npz")).open("xb") as stream:
                    np.savez_compressed(stream, depth=depth, visible=indices < len(faces), K=K,
                        scene_id=np.array(scene, np.int64), frame_id=np.array(frame, np.int64),
                        camera_R=rotation, camera_t=translation, face_indices=indices)
                with (public/(stem+".png")).open("xb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                hashes.append(digest(public/(stem+".png")))
                report["cases"].append({"scene_id": scene, "frame_id": frame, "status": "pass", **checked})
                persist(); print(json.dumps({"rendered_scene": scene, "frame": frame}), flush=True)
        with (public/"manifest.json").open("x") as stream: json.dump(public_manifest(hashes), stream, indent=2)
        for key, folder in (("private_files", private), ("public_files", public)):
            report[key] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                           for p in sorted(folder.iterdir()) if p != receipt}
        if (protocol(args.protocol) != frozen or source_helpers !=
                {p: {"bytes": (code/p).stat().st_size, "sha256": digest(code/p)} for p in source_files}):
            raise ValueError("Frozen recipe/source changed during manufacture")
        for folder, mode in ((public, 0o444), (private, 0o400)):
            for path in folder.iterdir():
                if path != receipt: path.chmod(mode)
        report.update(status="pass", phase="complete", public_manifest_sha256=digest(public/"manifest.json"))
    except Exception as exc:
        for folder in (public, private):
            for path in folder.iterdir():
                if path != receipt: path.unlink()
        report.update(error_type=type(exc).__name__, error=str(exc)[:300], arrays_removed=True,
                      private_files={}, public_files={})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist()
        receipt.chmod(0o400)


if __name__ == "__main__": main()

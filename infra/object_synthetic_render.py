"""Render-only independent object cohort; public inputs contain RGB alone.

Own embedded radial sphere and elliptic torus are not downloaded CAD assets.
All geometry, poses, camera-Z and visibility stay in a separate private folder.
Synthetic diffuse RGB is not photorealism or a reconstruction accuracy claim.
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

from camera_render import _mesh_inputs, _opencv_camera, project_camera_points, PYTORCH3D_REVISION


WIDTH, HEIGHT = 512, 384
K = np.array([[640., 0., 256.], [0., 640., 192.], [0., 0., 1.]])
BACKGROUNDS = ((.76, .79, .84), (.82, .78, .72))
TRAIN_VIEWS, HELDOUT_VIEWS = (0, 2, 4), (1, 3, 5)


def digest(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def radial_mesh():
    """Positive polynomial radius on S², then invertible ellipsoid deformation."""
    latitudes, longitudes = 32, 64
    theta = np.repeat(np.arange(1, latitudes) * np.pi / latitudes, longitudes)
    phi = np.tile(np.arange(longitudes) * 2 * np.pi / longitudes, latitudes - 1)
    directions = np.vstack(([0., 0., 1.], np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))), [0., 0., -1.]))
    x, y, z = directions.T
    radius = 1 + .12*(x**3-3*x*y*y) + .09*x*y + .08*z*x + .04*y**3
    if np.any(radius <= 0): raise ValueError("Radial fixture must be positive and embedded")
    vertices = directions * radius[:, None] * [.28, .21, .24]
    faces = []
    for j in range(longitudes): faces.append((0, 1+j, 1+(j+1) % longitudes))
    for ring in range(latitudes - 2):
        for j in range(longitudes):
            a = 1+ring*longitudes+j; d = 1+ring*longitudes+(j+1) % longitudes
            b, c = a+longitudes, d+longitudes
            faces.extend(((a, b, c), (a, c, d)))
    last, south = 1+(latitudes-2)*longitudes, len(vertices)-1
    for j in range(longitudes): faces.append((last+j, south, last+(j+1) % longitudes))
    return vertices, np.asarray(faces, dtype=np.int64)


def ring_mesh():
    """Embedded elliptic ring: scaled XY polar angle uniquely determines u.

    For each u, positive radial/vertical radii give one nondegenerate elliptical
    tube section; this construction has no overlapping unions or hidden repair.
    """
    nu, nv = 64, 32
    u, v = np.meshgrid(np.arange(nu)*2*np.pi/nu, np.arange(nv)*2*np.pi/nv, indexing="ij")
    radius = .17*(1+.12*np.cos(u)+.08*np.sin(2*u))
    factor = 1+radius*np.cos(v)
    vertices = np.stack((.29*factor*np.cos(u), .18*factor*np.sin(u),
                         .037*(1+.1*np.cos(u))*np.sin(v)+.014*np.sin(2*u)), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = i*nv+j, ((i+1) % nu)*nv+j
            c, d = ((i+1) % nu)*nv+(j+1) % nv, i*nv+(j+1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return vertices, np.asarray(faces, dtype=np.int64)


def mesh_contract(vertices, faces, expected_euler):
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or v.dtype.kind != "f" or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu" or not len(f)
            or np.any(f < 0) or np.any(f >= len(v)) or len(np.unique(f)) != len(v)):
        raise ValueError("Own mesh requires finite active vertices and valid triangles")
    triangles = v[f]; normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) <= 0): raise ValueError("Degenerate fixture triangle")
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2): raise ValueError("Own fixture is not a closed two-face-edge manifold")
    signed = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    _, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    if np.any(np.bincount(inverse, weights=signed) != 0): raise ValueError("Inconsistent fixture face winding")
    euler = len(v)-len(edges)+len(f)
    volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    if euler != expected_euler or not volume > 0: raise ValueError("Fixture genus/oriented volume differs")
    return {"vertices": len(v), "faces": len(f), "euler_characteristic": int(euler), "volume_m3": volume,
            "closed_oriented_edge_manifold": True, "embedding_by_analytic_construction": True,
            "pair_triangle_intersection_test_performed": False}


def camera_pose(view_index):
    from scipy.spatial.transform import Rotation
    if type(view_index) is not int or not 0 <= view_index < 6: raise ValueError("Require original view index0..5")
    pitches = (.08, -.16, .22, -.28, .12, -.10)
    r = Rotation.from_euler("yxz", [.55*view_index, pitches[view_index], .06*np.sin(view_index)]).as_matrix()
    t = np.array([.012*np.sin(view_index), .008*np.cos(view_index), 1.60+.025*np.sin(view_index*.7)])
    return r, t


def material(vertices):
    x, y, z = np.asarray(vertices).T
    return np.column_stack((.18+.12*np.sin(17*x+9*y), .34+.10*np.sin(13*y-7*z), .48+.08*np.sin(11*z+5*x)))


def public_manifest(digests):
    if len(digests) != 12 or any(not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s) for s in digests):
        raise ValueError("Require all12 original RGB digests")
    return {"schema": "world-reward-objects-rgb-inputs-v1", "images": [
        {"file": f"object_{obj:02d}_view_{view:02d}.png", "sha256": digests[obj*6+view], "width": WIDTH, "height": HEIGHT}
        for obj in range(2) for view in range(6)]}


def render_rgb(torch, camera_vertices, faces, colors, background):
    from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, PointLights,
                                   RasterizationSettings, SoftPhongShader, TexturesVertex)
    from pytorch3d.structures import Meshes
    v, f, matrix = _mesh_inputs(camera_vertices, faces, K, WIDTH, HEIGHT, .01)
    camera = _opencv_camera(torch, matrix, WIDTH, HEIGHT)
    mesh = Meshes(verts=[torch.as_tensor(v, device="cuda")], faces=[torch.as_tensor(f, device="cuda")],
                  textures=TexturesVertex(torch.as_tensor(colors, device="cuda", dtype=torch.float32)[None]))
    fragments = MeshRasterizer(cameras=camera, raster_settings=RasterizationSettings(image_size=(HEIGHT, WIDTH),
        blur_radius=0., faces_per_pixel=1, perspective_correct=True, clip_barycentric_coords=False,
        cull_backfaces=False, cull_to_frustum=False, max_faces_per_bin=len(f)))(mesh)
    shader = SoftPhongShader(device="cuda", cameras=camera, lights=PointLights(device="cuda", location=((- .3, -.5, .2),),
        ambient_color=((.45,)*3,), diffuse_color=((.55,)*3,), specular_color=((0.,)*3,)),
        materials=Materials(device="cuda", specular_color=((0.,)*3,)), blend_params=BlendParams(background_color=background))
    rgb = torch.round(shader(fragments, mesh)[0, ..., :3].clamp(0, 1)*255).to(torch.uint8).cpu().numpy()
    face = fragments.pix_to_face[0, ..., 0].cpu().numpy()
    depth = fragments.zbuf[0, ..., 0].cpu().numpy()
    return rgb, face, np.where(face >= 0, depth, np.nan)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Linux CUDA renderer with network none")
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable renderer source/image")
    destination = root / "validation/objects_rgb_v1"
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()) or os.environ.get("WR_RENDER_OUTPUT_RESERVED") != "1":
        raise FileExistsError("Require exclusive new empty reserved object cohort")
    public, private = destination / "inputs", destination / "eval_private"
    public.mkdir(); private.mkdir(mode=0o700)
    report = {"stage": "own_procedural_object_rgb_validation", "status": "fail", "phase": "prerequisites", "cases": [],
              "code_revision": revision, "image_id": image, "script_sha256": digest(Path(__file__)), "budget_seconds": 120,
              "challenge_inputs_used": False, "inference_performed": False, "accuracy_verified": False,
              "photorealism_verified": False, "truth_used_for_rendering_only": True,
              "training_view_indices": list(TRAIN_VIEWS), "heldout_view_indices": list(HELDOUT_VIEWS), "randomness": "none"}
    started = time.perf_counter()
    with (private / "render-report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Frozen120s synthesis budget exceeded")
        old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired); signal.alarm(120)
        try:
            persist()
            import torch
            import pytorch3d
            from PIL import Image
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Require CUDA/PyTorch3D0.7.9")
            report.update(torch=torch.__version__, pytorch3d=pytorch3d.__version__, pytorch3d_revision=PYTORCH3D_REVISION,
                          camera_helper_sha256=digest(Path(__file__).with_name("camera_render.py")), phase="rendering")
            images = []
            for obj, (generator, euler) in enumerate(((radial_mesh, 2), (ring_mesh, 0))):
                vertices, faces = generator(); topology = mesh_contract(vertices, faces, euler)
                colors = material(vertices)
                mesh_path = private / f"object_{obj:02d}_mesh.npz"
                with mesh_path.open("xb") as stream: np.savez_compressed(stream, vertices_m=vertices, faces=faces, material_rgb=colors)
                report.setdefault("meshes", []).append({"object_index": obj, "sha256": digest(mesh_path), **topology})
                for view in range(6):
                    report["active_object"], report["active_view"] = obj, view; persist()
                    r, t = camera_pose(view); camera = vertices @ r.T+t
                    pixels = project_camera_points(camera, K)
                    if np.any(pixels < [8, 8]) or np.any(pixels > [WIDTH-8, HEIGHT-8]): raise ValueError("Own whole-object border preflight failed")
                    with torch.inference_mode(): rgb, face, depth = render_rgb(torch, camera, faces, colors, BACKGROUNDS[obj])
                    torch.cuda.synchronize()
                    visible = face >= 0
                    if (rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8 or not visible.any()
                            or not np.isfinite(depth[visible]).all() or np.any(depth[visible] <= 0)):
                        raise ValueError("Own registered diffuse RGB/depth/visibility invalid")
                    name = f"object_{obj:02d}_view_{view:02d}"
                    truth = private / (name+".npz")
                    with truth.open("xb") as stream: np.savez_compressed(stream, camera_K=K, camera_R=r, camera_t=t,
                        depth_camera_m=depth, visibility=visible, visible_face_indices=face)
                    with (public / (name+".png")).open("xb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                    images.append(digest(public / (name+".png")))
                    report["cases"].append({"object_index": obj, "view_index": view, "truth_sha256": digest(truth),
                        "rgb_sha256": images[-1], "visible_pixels": int(visible.sum())})
                    persist(); print(json.dumps({"rendered_object": obj, "view": view}), flush=True)
            if time.perf_counter()-started > 120: raise TimeoutError("Frozen120s synthesis budget exceeded")
            with (public / "manifest.json").open("x") as stream: json.dump(public_manifest(images), stream, indent=2)
            report.update(status="pass", phase="complete", public_manifest_sha256=digest(public / "manifest.json"))
        except Exception as exc:
            report["error_type"], report["error"] = type(exc).__name__, str(exc)
            raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist()


if __name__ == "__main__": main()

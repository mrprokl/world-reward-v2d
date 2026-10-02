"""Own adjacent RGB cohort for fixed-geometry tracking; render truth is private.

New geometric parameters, trajectories and weak-feature control; no old O2
accuracy gate is rerun. Neither masks nor depth/cameras are inference inputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import object_synthetic_render as render


CASES, FRAMES = 3, 8
BACKGROUNDS = (*render.BACKGROUNDS, (.78, .81, .76))


def geometry(object_index):
    """Reuse only closed topology; all three position fields are newly frozen."""
    if type(object_index) is not int or not 0 <= object_index < CASES: raise ValueError("Require object0..2")
    if object_index in (0, 2):
        old, faces = render.radial_mesh()
        directions = old / [.28, .21, .24]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        x, y, z = directions.T
        if object_index == 0:
            radius = 1+.075*(y**3-3*y*x*x)-.06*x*z+.055*y*z+.025*x**3
            radii = [.255, .225, .205]
        else:
            radius = 1-.045*x*y+.065*z*y+.035*z**3
            radii = [.235, .185, .215]
        vertices = directions*radius[:, None]*radii
        euler = 2
    else:
        _, faces = render.ring_mesh()
        u, v = np.meshgrid(np.arange(64)*2*np.pi/64, np.arange(32)*2*np.pi/32, indexing="ij")
        radius = .205*(1-.08*np.sin(u)+.065*np.cos(2*u))
        factor = 1+radius*np.cos(v)
        vertices = np.stack((.265*factor*np.cos(u), .195*factor*np.sin(u),
            .042*(1-.075*np.sin(u))*np.sin(v)+.011*np.cos(3*u)), axis=-1).reshape(-1, 3)
        euler = 0
    return vertices, faces, render.mesh_contract(vertices, faces, euler)


def motion(frame_index):
    from scipy.spatial.transform import Rotation
    if type(frame_index) is not int or not 0 <= frame_index < FRAMES: raise ValueError("Require original frame0..7")
    fast = float(frame_index >= 5)
    r = Rotation.from_euler("yxz", [.035*frame_index+.16*fast, .06+.014*np.sin(frame_index*.6),
                                  -.03+.012*np.cos(frame_index*.5)]).as_matrix()
    t = np.array([-.03+.008*frame_index+.06*fast, .008*np.sin(frame_index*.7), 1.64+.006*frame_index])
    return r, t


def colors(vertices, object_index):
    if object_index == 2: return np.tile([.27, .36, .45], (len(vertices), 1))
    x, y, z = np.asarray(vertices).T
    return np.column_stack((.20+.10*np.sin(43*x+17*y), .35+.11*np.sin(37*y-23*z), .46+.09*np.sin(41*z+19*x)))


def public_manifest(digests):
    if len(digests) != CASES*FRAMES or any(not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s) for s in digests):
        raise ValueError("Require all24 ordered RGB SHA256 digests")
    return {"schema": "world-reward-object-motion-rgb-v1", "images": [
        {"file": f"object_{obj:02d}_frame_{frame:03d}.png", "sha256": digests[obj*FRAMES+frame],
         "width": render.WIDTH, "height": render.HEIGHT} for obj in range(CASES) for frame in range(FRAMES)]}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Linux CUDA renderer with network none")
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable renderer source/image")
    destination = root / "validation/object_motion_v1"
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()) or os.environ.get("WR_RENDER_OUTPUT_RESERVED") != "1":
        raise FileExistsError("Require exclusive empty reserved motion cohort")
    public, private = destination / "inputs", destination / "eval_private"
    public.mkdir(); private.mkdir(mode=0o700)
    report = {"stage": "own_procedural_object_motion_rgb", "status": "fail", "phase": "prerequisites", "cases": [],
        "code_revision": revision, "image_id": image, "script_sha256": render.digest(Path(__file__)), "budget_seconds": 120,
        "object_cases": CASES, "frames_each": FRAMES, "challenge_inputs_used": False, "inference_performed": False,
        "accuracy_verified": False, "photorealism_verified": False, "truth_used_for_rendering_only": True,
        "relative_tracking_accuracy_verified": False, "absolute_metric_accuracy_verified": False,
        "fast_transition": [4, 5], "weak_feature_control": 2, "randomness": "none",
        "inherited_anchor_bias_and_relative_motion_must_be_evaluated_separately": True,
        "render_helper_sha256": render.digest(Path(render.__file__))}
    started = time.perf_counter()
    with (private / "render-report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Frozen120s motion synthesis budget exceeded")
        old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired); signal.alarm(120)
        try:
            persist()
            import torch
            import pytorch3d
            from PIL import Image
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Require CUDA/PyTorch3D0.7.9")
            report.update(torch=torch.__version__, pytorch3d=pytorch3d.__version__, phase="rendering")
            images = []
            for obj in range(CASES):
                vertices, faces, topology = geometry(obj); material = colors(vertices, obj)
                mesh_path = private / f"object_{obj:02d}_mesh.npz"
                with mesh_path.open("xb") as stream: np.savez_compressed(stream, vertices_m=vertices, faces=faces, material_rgb=material)
                report.setdefault("meshes", []).append({"object_index": obj, "sha256": render.digest(mesh_path), **topology})
                for frame in range(FRAMES):
                    report["active_object"], report["active_frame"] = obj, frame; persist()
                    r, t = motion(frame); camera = vertices @ r.T+t
                    px = render.project_camera_points(camera, render.K)
                    if np.any(px < [8, 8]) or np.any(px > [render.WIDTH-8, render.HEIGHT-8]): raise ValueError("Own border preflight failed")
                    with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, camera, faces, material, BACKGROUNDS[obj])
                    torch.cuda.synchronize(); visible = face >= 0
                    if (rgb.shape != (render.HEIGHT, render.WIDTH, 3) or rgb.dtype != np.uint8 or not visible.any()
                            or not np.isfinite(depth[visible]).all() or np.any(depth[visible] <= 0)):
                        raise ValueError("Own registered RGB/depth/visibility invalid")
                    name = f"object_{obj:02d}_frame_{frame:03d}"
                    truth = private / (name+".npz")
                    with truth.open("xb") as stream: np.savez_compressed(stream, camera_K=render.K, camera_R=r, camera_t=t,
                        depth_camera_m=depth, visibility=visible, visible_face_indices=face)
                    with (public / (name+".png")).open("xb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                    images.append(render.digest(public / (name+".png")))
                    report["cases"].append({"object_index": obj, "frame_index": frame, "truth_sha256": render.digest(truth),
                        "rgb_sha256": images[-1], "visible_pixels": int(visible.sum())})
                    persist(); print(json.dumps({"rendered_object": obj, "frame": frame}), flush=True)
            if time.perf_counter()-started > 120: raise TimeoutError("Frozen120s motion synthesis budget exceeded")
            with (public / "manifest.json").open("x") as stream: json.dump(public_manifest(images), stream, indent=2)
            report.update(status="pass", phase="complete", public_manifest_sha256=render.digest(public / "manifest.json"))
        except Exception as exc:
            report["error_type"], report["error"] = type(exc).__name__, str(exc)
            raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist()


if __name__ == "__main__": main()

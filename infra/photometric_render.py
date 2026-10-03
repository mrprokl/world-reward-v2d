"""One own diffuse RGB fixture; no truth arrays are stored or exposed."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import struct
import sys
import time
import zlib

import numpy as np
import hand_synthetic_render as render
import identity_rgb_render as primitives
from world_reward.data import sha256

BASE, SCHEMA, STAGE, BUDGET = "validation/photometric_native_v1", "world-reward-photometric-native-v1", "own_photometric_rgb_render", 120
WIDTH, HEIGHT = 1024, 768
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
MODEL_BYTES = 696110248
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
K = ((1280., 0., 512.), (0., 1280., 384.), (0., 0., 1.))


def public_identity(path):
    path = Path(path)
    if (not path.is_file() or path.resolve() != path.absolute() or path.stat().st_mode & 0o222
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical immutable public regular file required")
    return dict(sha256=sha256(path), bytes=path.stat().st_size)


def public_manifest(digest):
    if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest): raise ValueError("Exact RGB SHA required")
    return dict(schema=SCHEMA, images=[dict(file="frame_000.png", sha256=digest, width=WIDTH, height=HEIGHT)])


def public_inputs(directory):
    """Audit just one immutable RGB and its manifest, never manufacturer state."""
    directory = Path(directory)
    if (not directory.is_dir() or directory.resolve() != directory.absolute()
            or any(p.is_symlink() for p in (directory, *directory.parents))):
        raise ValueError("Canonical RGB-only directory required")
    manifest = directory / "manifest.json"; receipt = public_identity(manifest)
    data = json.loads(manifest.read_text())
    if (type(data) is not dict or set(data) != {"schema", "images"} or data["schema"] != SCHEMA
            or type(data["images"]) is not list or len(data["images"]) != 1):
        raise ValueError("Exact one-image public schema required")
    row = data["images"][0]
    if (type(row) is not dict or set(row) != {"file", "sha256", "width", "height"}
            or row["file"] != "frame_000.png" or type(row["sha256"]) is not str
            or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
            or type(row["width"]) is not int or type(row["height"]) is not int
            or (row["width"], row["height"]) != (WIDTH, HEIGHT)):
        raise ValueError("One original RGB-only record required")
    image = directory / row["file"]; actual = public_identity(image)
    with image.open("rb") as handle: header = handle.read(33)
    if (actual["sha256"] != row["sha256"] or actual["bytes"] <= 33 or len(header) != 33
            or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
            or struct.unpack(">I", header[29:33])[0] != zlib.crc32(header[12:29]) & 0xffffffff
            or struct.unpack(">IIBBBBB", header[16:29]) != (WIDTH, HEIGHT, 8, 2, 0, 0, 0)):
        raise ValueError("Original hashed RGB PNG grid required")
    if {p.name for p in directory.iterdir()} != {"manifest.json", "frame_000.png"}:
        raise ValueError("Only one public RGB and manifest may be exposed")
    return [row | dict(path=image)], receipt


def helper_hashes():
    return dict(render=sha256(Path(render.__file__)), primitives=sha256(Path(primitives.__file__)),
        joint=sha256(Path(__file__).with_name("joint_rgb_render.py")),
        camera=sha256(Path(__file__).with_name("camera_render.py")),
        semantics=sha256(Path(render.semantics.__file__)))


def named_controls(names, limits):
    bounds = np.asarray(limits)
    if (np.ma.isMaskedArray(limits) or type(names) is not list or len(names) != 249 or len(set(names)) != 249
            or any(type(n) is not str or not n for n in names) or bounds.shape != (249, 2)
            or bounds.dtype.kind != "f" or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Actual unique249 names and native bounds required")
    controls = np.zeros((1, 204), np.float32); identity = np.zeros((1, 45), np.float32)
    identity[0, :2] = [.23, -.16]; controls[0, 136:] = .02
    locked = np.all(bounds[136:204] == 0, axis=1); controls[0, 136:][locked] = 0
    if locked.all(): raise ValueError("Actual free native scale controls required")
    recipe = [("l_uparm_ry", .21), ("l_elbow_bend", .36), ("r_uparm_ry", -.015)]
    recipe += [(f"l_{finger}1_rz", .16) for finger in ("index", "middle", "ring", "pinky")]
    for name, value in recipe:
        if name not in names[:136]: raise ValueError("Fixed native named pose absent")
        controls[0, names.index(name)] = value
    full = np.c_[controls, identity]; neutral = full.copy(); neutral[:, :136] = 0
    if np.any(full < bounds[:, 0]) or np.any(full > bounds[:, 1]) or np.any(neutral < bounds[:, 0]) or np.any(neutral > bounds[:, 1]):
        raise ValueError("Fixed animated/neutral249 recipe exceeds native bounds; never clip")
    return controls, identity


def scene(human, neutral, faces, regions):
    """Transient manufacture only; no parameters/geometry/labels are exported."""
    actor = np.asarray(human, np.float64); base = np.asarray(neutral, np.float64)
    if (np.ma.isMaskedArray(human) or np.ma.isMaskedArray(neutral) or actor.shape != (18439, 3)
            or base.shape != actor.shape or not np.isfinite(actor).all() or not np.isfinite(base).all()):
        raise ValueError("Finite own animated/neutral native meshes required")
    center = (base.min(0) + base.max(0)) / 2; relative = base - center
    distance = max(np.max(1280 * np.abs(relative[:, 0]) / (WIDTH * .29) - relative[:, 2]),
        np.max(1280 * np.abs(relative[:, 1]) / (HEIGHT * .29) - relative[:, 2])) + .4
    if not np.isfinite(distance) or distance <= 0: raise ValueError("Fixed whole-actor camera failed")
    actor = actor - center + [0., 0., distance]; camera = np.asarray(K, np.float64)
    pixels = render.project_camera_points(actor, camera)
    if np.any(pixels < 8) or np.any(pixels >= [WIDTH - 8, HEIGHT - 8]): raise ValueError("Fixed whole-actor8px framing failed")
    hand = actor[regions["l"]["vertex_mask"]]
    if len(hand) < 50: raise ValueError("Native hand-region manufacturing anchor absent")
    bottle, of = primitives.bottle_mesh(); bottle = bottle * 1.3
    bottle += (hand.min(0) + hand.max(0)) / 2 - (bottle.min(0) + bottle.max(0)) / 2 + [.045, .02, -.10]
    hc = np.tile([.71, .69, .66], (len(actor), 1)); low, high = human[:, 1].min(), human[:, 1].max()
    if high <= low: raise ValueError("Own actor height is nonpositive")
    band = (human[:, 1] - low) / (high - low); cloth = (band > .23) & (band < .81)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    stripes = (np.floor((human[cloth, 0] + human[cloth, 1]) * 22).astype(np.int64) % 2).astype(bool)
    hc[cloth] = np.where(stripes[:, None], [.46, .42, .36], [.20, .28, .35])
    bv, bf, bc = primitives.background(distance, float(relative[:, 1].max() + .08)); bv += [0., 0., distance]
    vertices = np.r_[actor, bottle, bv]; topology = np.r_[faces, of + len(actor), bf + len(actor) + len(bottle)]
    colors = np.r_[hc, np.tile([.25, .41, .30], (len(bottle), 1)), bc]
    if not np.isfinite(vertices).all() or vertices[:, 2].min() <= .01: raise ValueError("Own scene crosses near plane")
    return vertices, topology, colors, camera, len(faces), len(of)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Offline Linux CUDA manufacture required")
    root = Path(os.environ["WR_ROOT"]); dest = root / BASE
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or not dest.is_dir() or any(dest.iterdir())
            or dest.resolve() != dest.absolute() or any(p.is_symlink() for p in (dest, *dest.parents))
            or os.geteuid() != 1000 or image != IMAGE or not re.fullmatch("[0-9a-f]{40}", revision)):
        raise ValueError("Fresh canonical reserved image/source-bound output required")
    public = dest / "inputs"; public.mkdir(); receipt = dest / "render-report.json"
    started = time.perf_counter(); helpers = helper_hashes(); source_sha = sha256(Path(__file__))
    report = dict(stage=STAGE, status="fail", phase="prerequisites", frames=1, code_revision=revision,
        image_id=image, script_sha256=source_sha, helper_source_sha256=helpers, network="none",
        model_sha256=render.semantics.MODEL_SHA, budget_seconds=BUDGET, inference_performed=False,
        optimizer_performed=False, private_arrays_created=False, truth_arrays_exported=False,
        challenge_inputs_used=False, ground_truth_used_for_inference=False, accuracy_verified=False,
        photorealism_verified=False, license_clearance_verified=False, training_overlap_excluded=False,
        reference_forward_attempts=0, reference_forward_returns=0, raster_attempts=0, raster_returns=0)
    with receipt.open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; handle.seek(0)
            json.dump(report, handle, allow_nan=False); handle.write("\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Public RGB manufacture exceeded120s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); semantic_path = root / "results/mhr-finger-semantics-v4.json"; semantic_sha = sha256(semantic_path)
            render.semantics.regular_hash(semantic_path, root, semantic_sha)
            semantic = json.loads(semantic_path.read_text()); render.require_semantic_report(semantic)
            if semantic.get("source_image_id") != image: raise ValueError("Reference semantic image differs")
            model_path = root / "weights/mhr/mhr_model.pt"
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, MODEL_BYTES)
            if "torch" in sys.modules: raise RuntimeError("CUBLAS setup must precede Torch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Native pinned CUDA renderer required")
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joints = model.get_parameter_names(), model.get_joint_names(); bounds = model.get_parameter_limits().cpu().numpy()
            controls, identity = named_controls(names, bounds); faces = model.character_torch.mesh.faces.cpu().numpy()
            if (faces.dtype != np.int32 or faces.shape != (36874, 3) or faces.min() < 0 or faces.max() >= 18439
                    or sha256_array(faces) != FACE_SHA or joints != semantic["joint_names"]
                    or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
                raise ValueError("Pinned real native topology/name/identity ABI differs")
            regions = render.lbs_regions(*(v.cpu().numpy() for v in model.get_lbsw()), joints)
            p = torch.tensor(controls, device="cuda"); ids = torch.tensor(identity, device="cuda")
            neutral = controls.copy(); neutral[:, :136] = 0
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                report["reference_forward_attempts"] += 1; persist(); raw, sk = model(ids, p, torch.zeros(1, 72, device="cuda"), True)
                report["reference_forward_returns"] += 1; persist(); report["reference_forward_attempts"] += 1; persist()
                nv, ns = model(ids, torch.tensor(neutral, device="cuda"), torch.zeros(1, 72, device="cuda"), True)
                report["reference_forward_returns"] += 1; persist()
            torch.cuda.synchronize(); raw, sk, nv, ns = (v.cpu().numpy() for v in (raw, sk, nv, ns))
            render.semantics.check_geometry(raw, sk, 127); render.semantics.check_geometry(nv, ns, 127)
            if not np.array_equal(p.cpu().numpy(), controls) or not np.array_equal(ids.cpu().numpy(), identity):
                raise ValueError("Reference modified fixed manufacturing controls")
            sv, sf, colors, camera, nf, no = scene(raw[0] / 100 * [1., -1., -1.], nv[0] / 100 * [1., -1., -1.], faces, regions)
            report.update(phase="rendering", semantic_report_sha256=semantic_sha,
                torch=str(torch.__version__), pytorch3d_revision=render.PYTORCH3D_REVISION)
            report["raster_attempts"] += 1; persist()
            with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, sv, sf, colors, camera)
            torch.cuda.synchronize(); report["raster_returns"] += 1; persist()
            if (rgb.dtype != np.uint8 or rgb.shape != (HEIGHT, WIDTH, 3) or face.shape != (HEIGHT, WIDTH)
                    or np.count_nonzero((face >= 0) & (face < nf)) < 64
                    or np.count_nonzero((face >= nf) & (face < nf + no)) < 64):
                raise ValueError("Both own rendered entities must have real visible RGB support")
            image_path = public / "frame_000.png"
            with image_path.open("xb") as output: Image.fromarray(rgb).save(output, format="PNG")
            image_path.chmod(0o444); digest = sha256(image_path)
            with (public / "manifest.json").open("x") as output: json.dump(public_manifest(digest), output)
            (public / "manifest.json").chmod(0o444); _, public_receipt = public_inputs(public)
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, MODEL_BYTES)
            if helper_hashes() != helpers or sha256(Path(__file__)) != source_sha or sha256(semantic_path) != semantic_sha:
                raise ValueError("Manufacturing assets/source changed")
            if {p.name for p in dest.iterdir()} != {"inputs", "render-report.json"}: raise ValueError("No manufacturing caches may be exported")
            report.update(status="pass", phase="complete", rgb_sha256=digest, public_manifest_sha256=public_receipt["sha256"],
                public_manifest_bytes=public_receipt["bytes"], actual_reference_forward_calls=2, actual_render_calls=1,
                source_assets_rechecked=True, all_inputs_public_rgb_only=True)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o444)


def sha256_array(value):
    import hashlib
    return hashlib.sha256(value.tobytes()).hexdigest()


if __name__ == "__main__": main()

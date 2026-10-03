"""NEW four RGB anchors before any temporal/model-inferred bridge experiment.

Known authored MHR triangle-soup poses manufacture images only; they are never
hand observations. This is not an embedding/contact reference or SAM prediction.
Only RGB names/hashes/indices/dimensions are public; all generating values stay
private and must be excluded from future mask/Body/MoGe inference mounts.
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
import stat
import time

import numpy as np
import triangle_ray_gate as ray

BASE = "validation/bridge_rgb_anchor_v1"
SCHEMA = "world_reward.bridge_rgb_anchor_public.v1"
SCENARIOS = ("carry", "free", "slip", "regrasp")
SEED, WIDTH, HEIGHT, FRAMES, BUDGET = 2026100401, 640, 480, 48, 180
K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
IMAGE = ray.IMAGE
MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
MODEL_BYTES = 696110248
MODEL_SOURCE = "/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
RAY_SOURCES = {"infra/triangle_ray_gate.py": "aaea4ce32b68ad6e2e5aa411278cf4dc90264a79ce68e839f349bd8a6fce7444",
    "infra/run_triangle_ray_gate.sh": "112236fcd73a276f2f7afc93fe8ce08ca19d266a0ee9425cee745a1863f81224"}
SOURCE_FILES = ("infra/bridge_rgb_anchor_render.py", "infra/run_bridge_rgb_anchor_render.sh", *RAY_SOURCES)


def identity(path, *, sha=None, size=None, limit=2_000_000, immutable=False):
    p = Path(path)
    if p.resolve() != p.absolute() or any(a.is_symlink() for a in (p, *p.parents)):
        raise ValueError("Canonical original source required")
    before = p.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 1 <= before.st_size <= limit
            or immutable and before.st_mode & 0o222):
        raise ValueError("Bounded regular immutable source required")
    with p.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = p.lstat()
    if any(getattr(before, key) != getattr(after, key) for key in
           ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")):
        raise ValueError("Original source changed during hashing")
    if sha is not None and digest != sha or size is not None and before.st_size != size:
        raise ValueError("Independently pinned source identity differs")
    return dict(bytes=before.st_size, sha256=digest)


def require_ray_receipt(path, sha, size, producer):
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not re.fullmatch(r"[0-9a-f]{40}", producer) or type(size) is not int:
        raise ValueError("Actual independent gate SHA/bytes/producer required")
    pin = identity(path, sha=sha, size=size, immutable=True)
    data = json.loads(Path(path).read_text())
    expected = dict(schema="world_reward.triangle_ray_gate.v1", stage="new_independent_CPU_triangle_reference",
        status="pass", phase="complete", producer_revision=producer, image_id=IMAGE, budget_seconds=100,
        CPU_reference_gate_passed=True, sources_rehashed_after=True, models_used=False,
        challenge_inputs_used=False, failed_cohort_read=False, downstream_gates_changed=False,
        previous_failures_reinterpreted=False, inference_quality_verified=False)
    if any(type(data.get(k)) is not type(v) or data[k] != v for k, v in expected.items()):
        raise ValueError("Actual completed independent ray gate required before manufacture")
    sources = data.get("source_helpers", {})
    if set(sources) != set(RAY_SOURCES) or any(sources[k].get("sha256") != v for k, v in RAY_SOURCES.items()):
        raise ValueError("Exact original independent renderer source required")
    proof = data.get("CPU_reference", {})
    if (proof.get("width") != WIDTH or proof.get("height") != HEIGHT or proof.get("full_grid_pixels") != WIDTH*HEIGHT
            or proof.get("independent_label_disagreements") != 0 or proof.get("Decimal80_samples") != 20
            or proof.get("rear_first_nearest_layer_verified") is not True or proof.get("threshold_m") != 1e-8
            or any(type(proof.get(k)) not in (float, int) or not np.isfinite(proof[k]) or not 0 <= proof[k] <= 1e-8
                for k in ("full_grid_ray_max_Z_error_m", "full_grid_plane_max_Z_error_m", "Decimal80_max_Z_error_m"))):
        raise ValueError("Full-grid/Decimal/nearest-layer evidence must pass unchanged")
    if identity(path, sha=sha, size=size, immutable=True) != pin:
        raise ValueError("Independent gate changed")
    return pin


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def controls(names, bounds, clip, frame):
    if type(clip) is not int or not 0 <= clip < 4 or type(frame) is not int or not 0 <= frame < FRAMES:
        raise ValueError("Original NEW clip0..3/frame0..47 required")
    bounds = np.asarray(bounds)
    if (type(names) is not list or len(names) != 249 or len(set(names)) != 249
            or bounds.shape != (249, 2) or bounds.dtype.kind != "f" or np.isnan(bounds).any()
            or np.any(bounds[:, 0] > bounds[:, 1]) or np.any(bounds[:204, 0] > 0) or np.any(bounds[:204, 1] < 0)):
        raise ValueError("Original actual249 control names/legal bounds required")
    u = frame/(FRAMES-1); result = np.zeros(204, np.float32)
    recipe = {"l_uparm_ry": .177+.035*np.sin(2.1*u), "l_elbow_bend": .367+.043*u,
              "r_uparm_ry": -.123+.025*np.sin(1.9*u), "r_elbow_bend": .293+.031*u}
    recipe.update({f"l_{finger}1_rz": .209+.021*np.sin(2.3*u) for finger in ("index", "middle", "ring", "pinky")})
    for name, value in recipe.items():
        if name not in names[:136] or not bounds[names.index(name), 0] <= value <= bounds[names.index(name), 1]:
            raise ValueError("Frozen named authored pose unavailable; no clamp or alternative")
        result[names.index(name)] = value
    return result


def hand_frame(joints, names):
    """Named reference joints only, never ordinal/control-column guesses."""
    j = np.asarray(joints)
    required = ("l_wrist", "l_index1", "l_middle1")
    if j.shape != (127, 3) or not np.isfinite(j).all() or len(names) != 127 or len(set(names)) != 127 or any(n not in names for n in required):
        raise ValueError("Actual named127 reference joints required")
    wrist, index, middle = j[[names.index(n) for n in required]]
    x = middle-wrist; y = index-middle
    if np.linalg.norm(x) <= 1e-5: raise ValueError("Degenerate authored wrist/knuckle frame")
    x = x/np.linalg.norm(x); y = y-x*(x@y)
    if np.linalg.norm(y) <= 1e-5: raise ValueError("Collinear authored knuckles; no ideal hand fallback")
    y /= np.linalg.norm(y); z = np.cross(x, y)
    return np.column_stack((x, y, z)), wrist


def object_mesh():
    """New constant asymmetric bottle triangle soup, not a downloaded object."""
    n = 24; angle = np.arange(n)*2*np.pi/n
    levels = ((-.161, .057), (-.137, .066), (.079, .062), (.114, .028), (.173, .025))
    vertices = []
    for height, radius in levels:
        r = radius*(1+.071*np.sin(angle)+.037*np.cos(3*angle))
        vertices.extend(np.column_stack((r*np.cos(angle), np.full(n, height), .91*r*np.sin(angle))))
    vertices.extend([[0., levels[0][0], 0.], [0., levels[-1][0], 0.]])
    faces = []
    for ring in range(len(levels)-1):
        for i in range(n):
            a, b = ring*n+i, ring*n+(i+1) % n
            faces.extend([[a, a+n, b+n], [a, b+n, b]])
    for i in range(n): faces.extend([[120, (i+1) % n, i], [121, 96+i, 96+(i+1) % n]])
    return np.array(vertices, np.float64), np.array(faces, np.int64)


def object_pose(clip, frame, reference_hand_R, reference_hand_t):
    if type(clip) is not int or not 0 <= clip < 4 or type(frame) is not int or not 0 <= frame < FRAMES:
        raise ValueError("Frozen NEW motion indices required")
    u = frame/47.; relative = np.array([.065, -.026, -.044])
    r = reference_hand_R@rotation_y(.137+.12*u*u)
    t = reference_hand_t+reference_hand_R@relative
    if clip == 1:
        r = rotation_y(-.219+.51*u); t = t+[.075+.17*u, -.027*np.sin(4*u), -.12]
    elif clip == 2:
        t = t+reference_hand_R@np.array([.014+.13*np.clip((u-.23)/.42, 0, 1), .008, -.019])
        r = r@rotation_y(.071+.42*u)
    elif clip == 3:
        r = r@rotation_y(-.113+(.47 if frame >= 24 else 0.))
        t = t+[.026, -.036, -.047]+([.086, .041, .016] if frame >= 24 else np.zeros(3))
    return r, t


def scene(human, joints, faces, joint_names, clip, frame):
    rotation = rotation_y((-.117, .083, -.043, .149)[clip]+.061*(frame/47.)**2)
    translation = np.array([-.035+.12*(frame/47.)**2, .88, 4.0])
    hv = np.asarray(human, np.float64)@rotation.T+translation
    hj = np.asarray(joints, np.float64)@rotation.T+translation
    hr, ht = hand_frame(hj, joint_names); r, t = object_pose(clip, frame, hr, ht)
    ov, of = object_mesh(); ov = ov@r.T+t
    # New room/backplane geometry and texture are identical across all clips.
    backdrop = np.array([[-4., -3., 7.], [4., -3., 7.], [4., 3., 7.], [-4., 3., 7.]])
    all_vertices = np.r_[hv, ov, backdrop]
    all_faces = np.r_[faces, of+len(hv), np.array([[0, 1, 2], [0, 2, 3]], np.int64)+len(hv)+len(ov)]
    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, 2*np.pi, 3)
    hc = np.tile([.70, .61, .53], (len(hv), 1)); band = np.asarray(human)[:, 1]
    hc[(band > -.75) & (band < -.30)] = [.22, .31, .43]
    original_object, _ = object_mesh()
    oc = .38+.19*np.sin(original_object@np.array([[17., 9., -3.], [7., 19., 11.], [13., -5., 23.]])+phases)
    color = np.r_[hc, oc, np.array([[.63, .65, .67], [.71, .67, .61], [.62, .65, .70], [.69, .68, .64]])]
    labels = np.r_[np.zeros(len(faces), np.int64), np.ones(len(of), np.int64), np.full(2, 2, np.int64)]
    return all_vertices, all_faces, color, labels, dict(human_vertices_camera_m=hv, human_joints_camera_m=hj,
        human_faces=faces, object_vertices_camera_m=ov, object_faces=of, object_rotation=r, object_translation_m=t, camera_K=K.copy())


def certify_render(vertices, faces, K, result, deadline):
    """Full selected-plane proof plus64 independent rays testing EVERY face."""
    z, ids = result["depth"], result["face_index"]
    if not np.isfinite(z).all() or np.any(ids < 0) or np.any(ids >= len(faces)):
        raise ValueError("Original full-grid camera depth/background required")
    height, width = z.shape; yy, xx = np.mgrid[:height, :width]
    d = np.stack(((xx+.5-K[0, 2])/K[0, 0], (yy+.5-K[1, 2])/K[1, 1], np.ones_like(xx)), axis=-1)
    tri = vertices[faces[ids]]; n = np.cross(tri[..., 1, :]-tri[..., 0, :], tri[..., 2, :]-tri[..., 0, :])
    expected = np.sum(n*tri[..., 0, :], axis=-1)/np.sum(n*d, axis=-1)
    maximum = float(np.max(np.abs(expected-z)))
    xs = np.linspace(0, width-1, 8, dtype=int); ys = np.linspace(0, height-1, 8, dtype=int)
    y, x = np.meshgrid(ys, xs, indexing="ij"); y, x = y.ravel(), x.ravel()
    sample_scope = "64_global_grid"
    if "labels" in result:
        # Preserve32 global probes and require16 human+16 object probes. These
        # are private manufacturing checks, not model-visible labels or tuning.
        selected = [y[:32]*width+x[:32]]
        for label in (0, 1):
            pixels = np.flatnonzero(result["labels"].ravel() == label)
            if len(pixels) < 16: raise ValueError("Both entities need independent ray probes")
            selected.append(pixels[np.linspace(0, len(pixels)-1, 16, dtype=int)])
        y, x = np.divmod(np.concatenate(selected), width)
        sample_scope = "32_global_16_human_16_object"
    rays = d[y, x]
    nearest = np.full(64, np.inf); nearest_label = np.full(64, -1, np.int64)
    for face, (p, b, c) in enumerate(vertices[faces]):
        if face % 128 == 0 and time.monotonic() >= deadline: raise TimeoutError("Bounded independent64-ray certificate")
        e1, e2 = b-p, c-p; h = np.cross(rays, e2); det = h@e1; q = np.cross(-p, e1); valid = det != 0
        u = np.divide(h@-p, det, out=np.full(64, np.nan), where=valid)
        v = np.divide(rays@q, det, out=np.full(64, np.nan), where=valid)
        depth = np.divide(e2@q, det, out=np.full(64, np.nan), where=valid)
        keep = valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (depth > ray.Z_MIN) & (depth < nearest)
        nearest[keep], nearest_label[keep] = depth[keep], face
    sampled = float(np.max(np.abs(nearest-z[y, x])))
    if max(maximum, sampled) > 1e-8 or not np.isfinite(expected).all() or np.any(nearest_label < 0):
        raise ValueError("Independent all-triangle rays/full-plane gate failed; no image publication")
    return dict(full_grid_plane_max_Z_error_m=maximum, independent64_ray_max_Z_error_m=sampled,
        independent_rays=64, sample_scope=sample_scope, all_original_faces_tested_per_ray=len(faces), threshold_m=1e-8)


def public_manifest(rows):
    if len(rows) != 4 or any(type(r) is not dict or set(r) != {"sha256", "bytes"}
                           or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) or type(r["bytes"]) is not int or r["bytes"] <= 0 for r in rows):
        raise ValueError("All four independently hashed RGB anchors required")
    return dict(schema=SCHEMA, images=[dict(clip_id=i, frame_id=0, file=f"clip_{i:06d}_frame_000000.png",
        width=WIDTH, height=HEIGHT, **row) for i, row in enumerate(rows)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gate-sha256", required=True); parser.add_argument("--gate-bytes", required=True, type=int)
    parser.add_argument("--gate-producer", required=True); args = parser.parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); rev = os.environ.get("WR_CODE_REVISION", "")
    code = root/"jobs"/rev/"run_bridge_rgb_anchor_render/code"; out = root/BASE
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 0
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or not re.fullmatch(r"[0-9a-f]{40}", rev) or os.environ.get("WR_IMAGE_ID") != IMAGE
            or Path(__file__).resolve() != code/"infra/bridge_rgb_anchor_render.py" or out.resolve() != out
            or not out.is_dir() or out.stat().st_mode & 0o777 != 0o700 or any(out.iterdir())):
        raise ValueError("Fresh frozen Azure CPU-only authored anchor namespace required")
    source = {name: identity(code/name, immutable=True) for name in SOURCE_FILES}
    if any(source[name]["sha256"] != sha for name, sha in RAY_SOURCES.items()): raise ValueError("Original ray primitive changed")
    gate_path = root/"validation/triangle_ray_gate_v1/report.json"
    gate_pin = require_ray_receipt(gate_path, args.gate_sha256, args.gate_bytes, args.gate_producer)
    model_path = root/"weights/mhr/mhr_model.pt"
    model_pin = identity(model_path, sha=MODEL_SHA, size=MODEL_BYTES, limit=MODEL_BYTES)
    os.umask(0o077)
    private = out/"eval_private"; private.mkdir(mode=0o700)
    staging = private/"staged_rgb"; staging.mkdir(mode=0o700)
    report = dict(schema="world_reward.bridge_rgb_anchor_render.v1", status="fail", phase="reference_CPU",
        producer_revision=rev, image_id=IMAGE, source_helpers=source, script_sha256=source[SOURCE_FILES[0]]["sha256"],
        independent_ray_receipt=gate_pin, independent_ray_producer=args.gate_producer, model_file=model_pin,
        actual_reference_asset_source=MODEL_SOURCE, reference_asset_provenance="selected_SAM_3D_Body_asset_not_relicensed_Apache",
        seed=SEED, scenarios=list(SCENARIOS), frames_each_future=FRAMES, anchors=[0]*4, budget_seconds=BUDGET,
        reference_forward_calls=0, RGB_completed=0, model_prediction_calls=0, challenge_inputs_used=False,
        previous_references_reinterpreted=False, reference_scope="MHR_authored_triangle_soup_pose_only_no_embedding_or_contact_truth",
        hand_observations_produced=False, SAM_prediction_provenance_verified=False, SAM_license_clearance_claim=False,
        hand_validity_verified=False, automatic_mask_Body_MoGe_gate_passed=False, photorealism_verified=False,
        quality_verified=False, adoption_performed=False, cases=[])
    started = time.perf_counter(); deadline = time.monotonic()+BUDGET
    fd = os.open(private/"render-report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as receipt:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; receipt.seek(0)
            json.dump(report, receipt, allow_nan=False); receipt.write("\n"); receipt.truncate(); receipt.flush(); os.fsync(receipt.fileno())
        def expired(*unused): raise TimeoutError("Frozen180s four-anchor gate exceeded")
        alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); import torch
            from PIL import Image
            if np.__version__ != "1.26.3" or torch.__version__ != "2.5.1+cu124" or torch.cuda.is_available():
                raise ValueError("Frozen CPU reference runtime only required")
            torch.set_num_threads(4); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cpu").float().eval()
            names, joints = list(model.get_parameter_names()), list(model.get_joint_names())
            bounds = model.get_parameter_limits().cpu().numpy(); faces = model.character_torch.mesh.faces.cpu().numpy().astype(np.int64)
            if (len(joints) != 127 or len(set(joints)) != 127 or joints != list(model.character_torch.skeleton.joint_names)
                    or names != list(model.character_torch.parameter_transform.parameter_names)
                    or faces.shape != (36874, 3) or np.any(faces < 0) or np.any(faces >= 18439)
                    or model.get_num_identity_blendshapes() != 45 or model.get_num_face_expression_blendshapes() != 72):
                raise ValueError("Actual named MHR reference metadata differs; no new anatomical mapping")
            report.update(joint_names=joints, model_metadata_source="actual_scripted_MHR_getters", phase="four_reference_anchors")
            hashes = []
            for clip in range(4):
                q = controls(names, bounds, clip, 0)
                with torch.inference_mode(), torch.jit.optimized_execution(False):
                    vertices, skeleton = model(torch.zeros(1, 45), torch.from_numpy(q[None]), torch.zeros(1, 72), True)
                report["reference_forward_calls"] += 1
                v, j = vertices[0].numpy(), skeleton[0].numpy()
                if v.shape != (18439, 3) or j.shape != (127, 8) or not np.isfinite(v).all() or not np.isfinite(j).all():
                    raise ValueError("Original reference geometry ABI failed")
                if np.any(j[:, 7] <= 0) or np.max(np.abs(np.linalg.norm(j[:, 3:7], axis=1)-1)) > 1e-5:
                    raise ValueError("Original reference skeleton quaternion/scale ABI failed")
                flip = np.array([1., -1., -1.]); v, j = v.astype(np.float64)*flip/100, j[:, :3].astype(np.float64)*flip/100
                sv, sf, colors, labels, truth = scene(v, j, faces, joints, clip, 0)
                limit = float(min(deadline, time.monotonic()+99.))
                result = ray.render_triangles(sv, sf, K, WIDTH, HEIGHT, labels=labels, vertex_colors=colors, deadline=limit)
                proof = certify_render(sv, sf, K, result, deadline)
                counts = [int(np.count_nonzero(result["labels"] == label)) for label in (0, 1)]
                if counts[0] < 2048 or counts[1] < 64 or not np.isfinite(result["rgb"]).all():
                    raise ValueError("Frozen human/object visible anchor support failed; no reroll")
                rgb = np.round(np.clip(result["rgb"], 0, 1)*255).astype(np.uint8)
                name = f"clip_{clip:06d}_frame_000000"
                fd = os.open(private/(name+".npz"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(fd, "wb") as stream: np.savez_compressed(stream, **truth, body_controls=q,
                    identity_coeffs=np.zeros(45, np.float32), frame_index=np.array(0, np.int64), clip_index=np.array(clip, np.int64))
                fd = os.open(staging/(name+".png"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(fd, "wb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                row = identity(staging/(name+".png")); hashes.append(row)
                report["cases"].append(dict(clip_id=clip, frame_id=0, RGB_identity=row, visible_human_pixels=counts[0],
                    visible_object_pixels=counts[1], ray_certificate=proof, geometry_identity=identity(private/(name+".npz"), limit=10_000_000)))
                report["RGB_completed"] = clip+1; persist()
            require_ray_receipt(gate_path, args.gate_sha256, args.gate_bytes, args.gate_producer)
            if source != {name: identity(code/name, immutable=True) for name in SOURCE_FILES} or model_pin != identity(model_path, sha=MODEL_SHA, size=MODEL_BYTES, limit=MODEL_BYTES):
                raise ValueError("Frozen source/reference/gate changed")
            public = out/"inputs"; public.mkdir(mode=0o755); public.chmod(0o755)
            for clip, row in enumerate(hashes):
                name = f"clip_{clip:06d}_frame_000000.png"; p = staging/name
                if identity(p) != row: raise ValueError("Staged anchor changed before publication")
                p.chmod(0o444); os.rename(p, public/name)
            staging.rmdir()
            fd = os.open(public/"manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, "w") as stream: json.dump(public_manifest(hashes), stream, allow_nan=False); stream.write("\n")
            (public/"manifest.json").chmod(0o444)
            if ({p.name for p in public.iterdir()} != {"manifest.json", *[f"clip_{i:06d}_frame_000000.png" for i in range(4)]}
                    or any(identity(public/f"clip_{i:06d}_frame_000000.png") != row for i, row in enumerate(hashes))):
                raise ValueError("Complete RGB-only public inventory changed")
            if time.monotonic() >= deadline: raise TimeoutError("Frozen publication deadline exceeded")
            require_ray_receipt(gate_path, args.gate_sha256, args.gate_bytes, args.gate_producer)
            if source != {name: identity(code/name, immutable=True) for name in SOURCE_FILES} or model_pin != identity(model_path, sha=MODEL_SHA, size=MODEL_BYTES, limit=MODEL_BYTES):
                raise ValueError("Frozen source/reference changed after publication")
            report.update(status="pass", phase="complete", all_four_ray_gates_before_publication=True,
                public_manifest=identity(public/"manifest.json"), source_and_reference_rehashed_after=True,
                future_inference_mount_policy="individual5public_files_only_no_parent_recipe_private_poses_camera_or_reference")
        except Exception as error:
            report.update(error_type=type(error).__name__); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("status", "RGB_completed", "elapsed_seconds", "automatic_mask_Body_MoGe_gate_passed")}))


if __name__ == "__main__": main()

"""Fresh own human photometric factorial manufacture; all geometry stays private."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import human_photometric_protocol as protocol
import hand_synthetic_render as render
import identity_rgb_render as primitives
from world_reward.data import sha256

COHORT = protocol.COHORT
BASE, STAGE, BUDGET = COHORT.base, "own_human_photometric_rgb_render", 120
WIDTH, HEIGHT, GROUPS, FRAMES = COHORT.width, COHORT.height, COHORT.groups, COHORT.frames
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
SHAPES, SCALES = ((.18, -.11), (-.21, .14)), (.025, -.02)
BOTTLE_SCALE, OCCLUSION_OFFSETS, ELBOW_NAME = 1.35, (-.25, .25), "l_elbow"
MODEL_SHA, MODEL_BYTES = render.semantics.MODEL_SHA, 696110248
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
TRUTH_KEYS = frozenset(("human_vertices_camera_m", "human_faces", "human_joints_camera_m", "camera_K",
    "scene_depth_m", "visible_face_indices", "human_visibility", "group_index", "frame_index"))


def array_sha(value): return hashlib.sha256(value.tobytes()).hexdigest()


def helper_hashes():
    return dict(render=sha256(Path(render.__file__)), primitives=sha256(Path(primitives.__file__)),
        joint=sha256(Path(__file__).with_name("joint_rgb_render.py")), camera=sha256(Path(__file__).with_name("camera_render.py")),
        semantics=sha256(Path(render.semantics.__file__)), protocol=sha256(Path(protocol.__file__)))


def factors(group):
    if type(group) is not int or not 0 <= group < GROUPS: raise ValueError("Original group index required")
    return group // 4, (group // 2) % 2, group % 2


def named_controls(names, limits):
    bounds = np.asarray(limits)
    if (np.ma.isMaskedArray(limits) or type(names) is not list or len(names) != 249 or len(set(names)) != 249
            or any(type(n) is not str or not n for n in names) or bounds.shape != (249, 2) or bounds.dtype.kind != "f"
            or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Actual249 unique native names/bounds required")
    scales = np.broadcast_to(np.asarray(SCALES, np.float32)[:, None], (2, 68)).copy()
    locked = np.all(bounds[136:204] == 0, axis=1); scales[:, locked] = 0
    if locked.all(): raise ValueError("Free native scale controls required")
    controls, shape, changes = np.zeros((6, 204), np.float32), np.zeros((6, 45), np.float32), []
    for morphology in range(2):
        for frame in range(FRAMES):
            row = morphology * FRAMES + frame; controls[row, 136:] = scales[morphology]; shape[row, :2] = SHAPES[morphology]
            recipe = [("l_uparm_ry", .16 + .05 * frame), ("l_elbow_bend", .24 + .06 * frame),
                ("l_wrist_ry", .008 - .012 * frame), ("r_uparm_ry", -.018 * (frame - 1))]
            recipe += [(f"l_{finger}1_rz", .10 + .03 * frame) for finger in ("index", "middle", "ring", "pinky")]
            for name, value in recipe:
                if name not in names[:136]: raise ValueError("Prescribed named native pose absent")
                column = names.index(name); controls[row, column] = value
                changes.append(dict(morphology_index=morphology, frame_index=frame, name=name, column=column, value=value))
    full = np.c_[controls, shape]; neutral = full[::FRAMES].copy(); neutral[:, :136] = 0
    if np.any(full < bounds[:, 0]) or np.any(full > bounds[:, 1]) or np.any(neutral < bounds[:, 0]) or np.any(neutral > bounds[:, 1]):
        raise ValueError("Fixed full249 animated/neutral recipe exceeds native bounds; never clip")
    return controls, shape, changes


def fixed_framing(neutral):
    points = np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or points.ndim != 3 or points.shape[0] != 2 or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError("Two finite neutral identities required")
    points = points.reshape(-1, 3); center = (points.min(0) + points.max(0)) / 2; relative = points - center
    distance = max(np.max(1280 * np.abs(relative[:, 0]) / (WIDTH * .29) - relative[:, 2]),
        np.max(1280 * np.abs(relative[:, 1]) / (HEIGHT * .29) - relative[:, 2])) + .4
    if not np.isfinite(distance) or distance <= 0: raise ValueError("Positive common camera required")
    return center, float(distance), float(relative[:, 1].max() + .08)


def transform_points(points, center, distance, frame):
    from scipy.spatial.transform import Rotation
    if type(frame) is not int or not 0 <= frame < FRAMES: raise ValueError("Fixed motion frame required")
    return (np.asarray(points) - center) @ Rotation.from_rotvec([0., .10 + .03 * (frame - 1), 0.]).as_matrix().T + [.01 * (frame - 1), 0., distance]


def scene(human, faces, joints, elbow_index, regions, center, distance, floor, group, frame):
    _, appearance, occlusion = factors(group)
    h, j = np.asarray(human), np.asarray(joints)
    if (any(np.ma.isMaskedArray(x) for x in (human, joints)) or h.ndim != 2 or h.shape[1] != 3
            or j.shape != (127, 3) or type(elbow_index) is not int or not 0 <= elbow_index < 127
            or not np.isfinite(h).all() or not np.isfinite(j).all()): raise ValueError("Actual finite V/J and named joint required")
    actor = transform_points(h, center, distance, frame); sk = transform_points(j, center, distance, frame)
    camera = np.asarray(COHORT.fixed_K, np.float64); pixels = render.project_camera_points(actor, camera)
    if np.any(pixels < 8) or np.any(pixels >= [WIDTH - 8, HEIGHT - 8]): raise ValueError("Fixed whole-human8px framing failed")
    bottle, of = primitives.bottle_mesh(); bottle = bottle * BOTTLE_SCALE
    bottle += sk[elbow_index] - (bottle.min(0) + bottle.max(0)) / 2 + [0., 0., OCCLUSION_OFFSETS[occlusion]]
    hc = np.tile([.70, .68, .65], (len(actor), 1)); low, high = h[:, 1].min(), h[:, 1].max()
    if high <= low: raise ValueError("Own human extent absent")
    band = (h[:, 1] - low) / (high - low); cloth = (band > .23) & (band < .81)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    if appearance == 0: hc[cloth] = [.34, .35, .37]
    else:
        stripes = (np.floor((h[cloth, 0] + h[cloth, 1]) * 20).astype(np.int64) % 2).astype(bool)
        hc[cloth] = np.where(stripes[:, None], [.45, .40, .35], [.22, .29, .38])
    bv, bf, bc = primitives.background(distance, floor); bv += [0., 0., distance]
    vertices = np.r_[actor, bottle, bv]; topology = np.r_[faces, of + len(actor), bf + len(actor) + len(bottle)]
    if not np.isfinite(vertices).all() or vertices[:, 2].min() <= .01: raise ValueError("Own scene crosses near plane")
    return actor, sk, vertices, topology, np.r_[hc, np.tile([.28, .40, .31], (len(bottle), 1)), bc], camera, of


def visible_truth(face, depth, faces, object_faces):
    f, z = np.asarray(face), np.asarray(depth)
    if (any(np.ma.isMaskedArray(x) for x in (face, depth)) or f.shape != (HEIGHT, WIDTH) or z.shape != f.shape
            or f.dtype.kind not in "iu" or z.dtype.kind != "f"): raise ValueError("Original native raster required")
    human = (f >= 0) & (f < len(faces)); obj = (f >= len(faces)) & (f < len(faces) + len(object_faces))
    if human.sum() < 64 or obj.sum() < 64: raise ValueError("Both manufactured entities require64 visible pixels")
    if not np.isfinite(z[human | obj]).all() or np.any(z[human | obj] <= 0): raise ValueError("Visible foreground depth invalid")
    return np.where(human, f, -1).astype(np.int64), np.where(human, z, np.nan).astype(np.float32), human


def occlusion_evidence(front, back):
    if (front.dtype != np.bool_ or back.dtype != np.bool_ or front.shape != (HEIGHT, WIDTH) or back.shape != front.shape):
        raise ValueError("Both original human visibility masks required")
    difference = int(np.count_nonzero(back & ~front))
    if difference < 64: raise ValueError("Depth-order contrast must expose64 human pixels; never adjust")
    return dict(newly_visible_human_pixels=difference, front_human_pixels=int(front.sum()), back_human_pixels=int(back.sum()))


def parity(v, sk, bv, bs):
    render.semantics.check_geometry(v, sk, 127); render.semantics.check_geometry(bv, bs, 127)
    if v.shape != bv.shape or sk.shape != bs.shape: raise ValueError("Same bundled/reference batches required")
    errors = dict(vertices_m=float(np.linalg.norm((v.astype(np.float64) - bv) / 100, axis=-1).max()),
        joints_m=float(np.linalg.norm((sk[..., :3].astype(np.float64) - bs[..., :3]) / 100, axis=-1).max()))
    if any(not np.isfinite(x) or x > 1e-5 for x in errors.values()): raise ValueError("Bundled/source rig geometry exceeds1e-5m")
    return errors


def rig_sources(root):
    root = Path(root); model = root / "weights/mhr/mhr_model.pt"
    bundled = root / render.semantics.BODY_RELATIVE / "assets/mhr_model.pt"; acquisition = root / "results/weights-acquisition.json"
    render.semantics.regular_hash(model, root, MODEL_SHA, MODEL_BYTES); render.semantics.regular_hash(bundled, root, MODEL_SHA, MODEL_BYTES)
    receipt_sha = sha256(acquisition); render.semantics.regular_hash(acquisition, root, receipt_sha)
    rows = [r for r in json.loads(acquisition.read_text()).get("assets", []) if r.get("repo_id") == "facebook/sam-3d-body-dinov3"]
    if (len(rows) != 1 or rows[0].get("revision") != render.semantics.BODY_REVISION
            or Path(rows[0].get("path", "")).resolve() != (root / render.semantics.BODY_RELATIVE).resolve()):
        raise ValueError("Pinned bundled Body rig acquisition required")
    return model, bundled, dict(model_sha256=MODEL_SHA, model_bytes=MODEL_BYTES, bundled_model_sha256=MODEL_SHA,
        bundled_model_bytes=MODEL_BYTES, body_revision=render.semantics.BODY_REVISION, acquisition_report_sha256=receipt_sha)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Offline Linux CUDA manufacture required")
    root = Path(os.environ["WR_ROOT"]); dest = root / BASE; revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or not dest.is_dir() or any(dest.iterdir()) or os.geteuid() != 1000
            or dest.resolve() != dest.absolute() or any(p.is_symlink() for p in (dest, *dest.parents))
            or image != IMAGE or not re.fullmatch("[0-9a-f]{40}", revision)): raise ValueError("Fresh canonical reserved image/source output required")
    public, private = dest / "inputs", dest / "eval_private"; public.mkdir(); private.mkdir(mode=0o700)
    receipt = private / "render-report.json"; started = time.perf_counter(); helpers = helper_hashes(); source = sha256(Path(__file__))
    report = dict(stage=STAGE, status="fail", phase="prerequisites", frames=24, groups=8, cases=[], truth_keys=sorted(TRUTH_KEYS),
        code_revision=revision, image_id=image, script_sha256=source, helper_source_sha256=helpers, network="none", budget_seconds=BUDGET,
        challenge_inputs_used=False, inference_performed=False, optimizer_performed=False, accuracy_verified=False, photorealism_verified=False,
        synthetic_truth_used_for_rendering_only=True, all_truth_private=True, license_clearance_verified=False, training_overlap_excluded=False,
        reference_forward_attempts=0, reference_forward_returns=0, bundled_forward_attempts=0, bundled_forward_returns=0,
        raster_attempts=0, raster_returns=0, shape_first_two=SHAPES, scale68_scalars=SCALES, bottle_scale=BOTTLE_SCALE,
        nuisance_Z_offsets=OCCLUSION_OFFSETS, native_elbow_name=ELBOW_NAME, factor_order="morphology-major,appearance-middle,occlusion-minor")
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Human photometric manufacture exceeded120s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); semantic_path = root / "results/mhr-finger-semantics-v4.json"; semantic_sha = sha256(semantic_path)
            render.semantics.regular_hash(semantic_path, root, semantic_sha); semantic = json.loads(semantic_path.read_text()); render.require_semantic_report(semantic)
            if semantic.get("source_image_id") != image: raise ValueError("Semantic native source image differs")
            model_path, bundled_path, rig_identity = rig_sources(root)
            if "torch" in sys.modules: raise RuntimeError("CUBLAS setup must precede Torch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Pinned native CUDA renderer required")
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joint_names = model.get_parameter_names(), model.get_joint_names(); bounds = model.get_parameter_limits().cpu().numpy()
            controls, identities, changes = named_controls(names, bounds); faces = model.character_torch.mesh.faces.cpu().numpy()
            if (faces.dtype != np.int32 or faces.shape != (36874, 3) or faces.min() < 0 or faces.max() >= 18439 or array_sha(faces) != FACE_SHA
                    or joint_names != semantic["joint_names"] or ELBOW_NAME not in joint_names
                    or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
                raise ValueError("Actual pinned native topology/names/elbow/identity ABI differs")
            elbow_index = joint_names.index(ELBOW_NAME); regions = render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()), joint_names)
            p, ids = torch.tensor(controls, device="cuda"), torch.tensor(identities, device="cuda")
            neutral = controls[::FRAMES].copy(); neutral[:, :136] = 0
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                report["reference_forward_attempts"] += 1; persist(); raw, sk = model(ids, p, torch.zeros(6, 72, device="cuda"), True)
                report["reference_forward_returns"] += 1; persist(); report["reference_forward_attempts"] += 1; persist()
                nv, ns = model(ids[::FRAMES], torch.tensor(neutral, device="cuda"), torch.zeros(2, 72, device="cuda"), True)
                report["reference_forward_returns"] += 1; persist()
            torch.cuda.synchronize(); raw, sk, nv, ns = (x.cpu().numpy() for x in (raw, sk, nv, ns))
            render.semantics.check_geometry(raw, sk, 127); render.semantics.check_geometry(nv, ns, 127)
            with torch.jit.optimized_execution(False): bundled = torch.jit.load(str(bundled_path), map_location="cuda").float().eval()
            if bundled.get_joint_names() != joint_names or not np.array_equal(bundled.character_torch.mesh.faces.cpu().numpy(), faces):
                raise ValueError("Bundled rig native joint names/topology differ")
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                report["bundled_forward_attempts"] += 1; persist(); bv, bs = bundled(ids, p, torch.zeros(6, 72, device="cuda"), True)
                report["bundled_forward_returns"] += 1; persist()
            torch.cuda.synchronize(); errors = parity(raw, sk, bv.cpu().numpy(), bs.cpu().numpy()); del bundled, bv, bs
            if not np.array_equal(p.cpu().numpy(), controls) or not np.array_equal(ids.cpu().numpy(), identities): raise ValueError("Native model mutated fixed249 parameters")
            flip = np.array([1., -1., -1.]); humans = raw / 100 * flip; joints = sk[..., :3] / 100 * flip
            center, distance, floor = fixed_framing(nv / 100 * flip)
            with (private / "rig.npz").open("xb") as out: np.savez_compressed(out, controls=controls, shape45=identities, parameter_limits=bounds,
                hand_mask_left=regions["l"]["vertex_mask"], hand_mask_right=regions["r"]["vertex_mask"])
            (private / "rig.npz").chmod(0o400)
            report.update(phase="rendering", rig_identity=rig_identity, semantic_report_sha256=semantic_sha, rig_sha256=sha256(private / "rig.npz"),
                human_faces_sha256=FACE_SHA, human_faces_dtype="int32", parameter_names=names, joint_names=joint_names, native_elbow_index=elbow_index,
                named_controls=changes, bundled_reference_parity=errors, bundled_reference_same_joint_names=True, bundled_reference_same_topology=True,
                actual_reference_forward_calls=2, actual_bundled_forward_calls=1, actual_total_native_forward_calls=3,
                torch=str(torch.__version__), pytorch3d_revision=render.PYTORCH3D_REVISION)
            digests, geometry, front_masks, contrasts = [], {}, {}, []
            for group in range(GROUPS):
                morphology, appearance, occlusion = factors(group)
                for frame in range(FRAMES):
                    report.update(active_group=group, active_frame=frame); persist(); row = morphology * FRAMES + frame
                    hv, jv, sv, sf, colors, camera, of = scene(humans[row], faces, joints[row], elbow_index, regions, center, distance, floor, group, frame)
                    geometry_id = dict(vertices_sha256=array_sha(hv), joints_sha256=array_sha(jv)); key = (morphology, frame)
                    if key in geometry and geometry[key] != geometry_id: raise ValueError("Appearance/occlusion changed human geometry")
                    geometry[key] = geometry_id; report["raster_attempts"] += 1; persist()
                    with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, sv, sf, colors, camera)
                    torch.cuda.synchronize(); report["raster_returns"] += 1; persist()
                    ids_h, depth_h, visible = visible_truth(face, depth, faces, of)
                    if rgb.dtype != np.uint8 or rgb.shape != (HEIGHT, WIDTH, 3): raise ValueError("Original RGB grid differs")
                    contrast_key = (morphology, appearance, frame)
                    if occlusion == 0: front_masks[contrast_key] = visible.copy()
                    else: contrasts.append(dict(morphology_index=morphology, appearance_index=appearance, frame_index=frame,
                        **occlusion_evidence(front_masks.pop(contrast_key), visible)))
                    stem = Path(COHORT.frame_name(group * FRAMES + frame)).stem; truth = private / (stem + ".npz"); png = public / (stem + ".png")
                    with truth.open("xb") as out: np.savez_compressed(out, human_vertices_camera_m=hv, human_faces=faces, human_joints_camera_m=jv,
                        camera_K=camera, scene_depth_m=depth_h, visible_face_indices=ids_h, human_visibility=visible,
                        group_index=np.array(group, np.int64), frame_index=np.array(frame, np.int64))
                    truth.chmod(0o400)
                    with png.open("xb") as out: Image.fromarray(rgb).save(out, format="PNG")
                    png.chmod(0o444); digests.append(sha256(png)); report["cases"].append(dict(group_index=group, frame_index=frame,
                        morphology_index=morphology, appearance_index=appearance, occlusion_index=occlusion, file=png.name,
                        rgb_sha256=digests[-1], truth_sha256=sha256(truth), geometry=geometry_id,
                        visible_human_pixels=int(visible.sum()), visible_nuisance_pixels=int(np.count_nonzero((face >= len(faces)) & (face < len(faces) + len(of)))))); persist()
            if len(set(digests)) != 24 or front_masks or len(contrasts) != 12: raise ValueError("All24 unique RGB and12 factorial contrasts required")
            with (public / "manifest.json").open("x") as out: json.dump(protocol.public_manifest(digests), out)
            (public / "manifest.json").chmod(0o444); records, manifest = protocol.public_inputs(public)
            if rig_sources(root)[2] != rig_identity or sha256(semantic_path) != semantic_sha or helper_hashes() != helpers or sha256(Path(__file__)) != source:
                raise ValueError("Manufacturing assets/source changed")
            if sha256(private / "rig.npz") != report["rig_sha256"] or any(sha256(private / (Path(r["file"]).stem + ".npz")) != r["truth_sha256"] for r in report["cases"]):
                raise ValueError("Frozen manufacturing geometry changed")
            if {p.name for p in private.iterdir()} != {"render-report.json", "rig.npz", *[Path(r["file"]).stem + ".npz" for r in records]}:
                raise ValueError("Exact private manufacturing inventory required")
            report.update(status="pass", phase="complete", occlusion_evidence=contrasts, all_factors_geometry_independent=True,
                public_manifest_sha256=manifest["sha256"], public_manifest_bytes=manifest["bytes"], source_assets_rechecked=True)
            report.pop("active_group", None); report.pop("active_frame", None)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o400)


if __name__ == "__main__": main()

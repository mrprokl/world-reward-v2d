"""Public-video consistency trial: two constant identities, only camera XY fit.

On a new episode, frame-zero identity and a neutral-mesh medoid receive exactly
the same30-step fit. Z, camera K, poses, hands, identities and automatic object
masks remain fixed. Reserved pixels are never fitted or used to select a step.
This does not establish human/object accuracy or authorize production adoption.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

import body_smoke as body
import camera_render as camera
import identity_consensus_public as consensus
from soft_silhouette_repro import scoped_backward
from world_reward.data import sha256

STAGE = "public_constant_identity_coupled_xy_silhouette_trial"
BASE = "outputs/episode_000015/identity_xy_fit_v1"
FRAMES, WIDTH, HEIGHT, LOW_WIDTH, LOW_HEIGHT = 501, 1536, 1152, 256, 192
ANCHORS = tuple(np.rint(np.linspace(0, 500, 12)).astype(int).tolist())
MIDPOINTS = tuple(((np.asarray(ANCHORS[:-1])+ANCHORS[1:])//2).tolist())
SELECTED = tuple(sorted(ANCHORS+MIDPOINTS))
VERTICES, FACES, STEPS, BUDGET = 18439, 36874, 30, 600
LR, DELTA_BOUND_M, PRIOR_STD_M, PRIOR_WEIGHT = .005, .05, .02, .001
SIGMA, GAMMA, FPP = 1e-4, 1e-4, 8
BLUR = float(np.log(1./1e-4-1.)*SIGMA)
MEMORY_LIMIT = 32*1024**3
BODY_SHA = "345e2f0faba5e21fbc46b99a8c605732dba02d5f6b2eb2e5f1bc5e4786a93277"
PRED_SHA = "e981e21e42012d04912a9895c25ef6dfad35ca331f7d3cb0e7fb945ffff995e3"
MASK_SHA = "eddf4c80bd3950842a8e597b8427d9ba3734cea21a0f07e6b7b27a3ace3b6800"
VIDEO_SHA = "c368809329e922b75629c1d94c39c1c820c9211464a5e5a77a4cdb55803400f0"
PROMPTS_SHA = "31047126b01c27d02e1e74d4d51fd8d7b29f1136ef35fee2f977ab78a0764ac7"
BODY_SCRIPT_SHA = "fab7f52949c7814272b7ba10647282d83a5d610cb68c778bbdb95b2cad2032aa"
MASK_SCRIPT_SHA = "2f991e4f144231db57f4caab92a13d8af5486d82c9f4e5b8bcc36b5b6f4b6c52"
BODY_BYTES, PRED_BYTES, MASK_BYTES = 244066, 201082259, 10708
BLOCKS = consensus.BLOCKS


def split_mask(height, width, block):
    if any(type(v) is not int or v <= 0 for v in (height, width, block)):
        raise ValueError("Positive integer fixed pixel grids required")
    yy, xx = np.mgrid[:height, :width]
    return ((yy//block+xx//block)%2) == 0


def mask_protocol(human, object_mask):
    if any(not isinstance(v, np.ndarray) or np.ma.isMaskedArray(v) or v.dtype != np.bool_
           or v.shape != (HEIGHT, WIDTH) for v in (human, object_mask)):
        raise ValueError("Full original human/object boolean mask grids required")
    if (WIDTH, HEIGHT) != (LOW_WIDTH*6, LOW_HEIGHT*6): raise ValueError("Exact6x6 majority grid required")
    train = split_mask(HEIGHT, WIDTH, 48); region = ~object_mask
    support = [int(np.count_nonzero(human & region & part)) for part in (train, ~train)]
    if min(support) <= 32: raise ValueError("Every frame needs >32 observed human pixels in BOTH partitions")
    human_blocks = human.reshape(LOW_HEIGHT, 6, LOW_WIDTH, 6)
    object_blocks = object_mask.reshape(LOW_HEIGHT, 6, LOW_WIDTH, 6)
    low_human = human_blocks.sum(axis=(1, 3)) >= 19
    low_region = ~object_blocks.any(axis=(1, 3))
    low_train = split_mask(LOW_HEIGHT, LOW_WIDTH, 8) & low_region
    if not np.any(low_human & low_train): raise ValueError("No observed human training support on conservative low grid")
    return low_human, low_train, support


def bounded_xy(w):
    if (not isinstance(w, np.ndarray) or np.ma.isMaskedArray(w) or w.dtype.kind != "f"
            or w.shape != (len(SELECTED), 2) or not np.isfinite(w).all()):
        raise ValueError("All23 finite physical XY optimization variables required")
    return DELTA_BOUND_M*np.tanh(w/DELTA_BOUND_M)


def training_loss(alpha, observed, train, delta):
    """Pure NumPy mirror of equal-frame full-train MSE plus fixed metric prior."""
    shape = (len(SELECTED), LOW_HEIGHT, LOW_WIDTH)
    if (any(np.ma.isMaskedArray(v) for v in (alpha, observed, train, delta))
            or alpha.shape != shape or observed.shape != shape or train.shape != shape
            or alpha.dtype.kind != "f" or observed.dtype != np.bool_ or train.dtype != np.bool_
            or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any()
            or delta.shape != (len(SELECTED), 2) or not np.isfinite(delta).all() or (np.abs(delta) > DELTA_BOUND_M).any()):
        raise ValueError("Finite full-grid alpha, fixed training support and bounded XY required")
    counts = train.sum(axis=(1, 2))
    if (counts <= 0).any(): raise ValueError("No frame dropping from objective")
    data = float((((alpha-observed)**2)*train).sum(axis=(1, 2)).astype(np.float64).dot(1/counts)/len(SELECTED))
    prior = float(PRIOR_WEIGHT*np.mean((delta/PRIOR_STD_M)**2))
    return data+prior


def hard_evidence(prediction, human, object_mask):
    if any(not isinstance(v, np.ndarray) or v.dtype != np.bool_ or np.ma.isMaskedArray(v)
           or v.shape != (HEIGHT, WIDTH) for v in (prediction, human, object_mask)):
        raise ValueError("Complete hard masks required")
    heldout = ~split_mask(HEIGHT, WIDTH, 48) & ~object_mask
    observed = human & heldout
    if observed.sum() <= 32: raise ValueError("No reserved frame may be dropped")
    return camera.silhouette_iou(prediction & heldout, observed)


def decision(rows, branches):
    if not isinstance(rows, list) or len(rows) != len(SELECTED) or set(branches) != {"M0", "M1"}:
        raise ValueError("Complete paired23-frame trial required")
    for name, branch in branches.items():
        if (branch.get("actual_optimizer_steps") != STEPS or branch.get("actual_backward_calls") != STEPS
                or branch.get("all_backward_flags_restored") is not True
                or any(type(branch.get(k)) is not float or not np.isfinite(branch[k]) or branch[k] < 0
                       for k in ("initial_train_loss", "final_train_loss"))):
            raise ValueError("Both branches require identical complete30-step optimization")
    deltas = []
    for index, row in zip(SELECTED, rows):
        if type(row.get("frame_index")) is not int or row["frame_index"] != index:
            raise ValueError("Original paired selected frame order required")
        for key in ("raw", "M0_before", "M1_before", "M0_after", "M1_after"):
            if type(row.get(key)) is not float or not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("All five complete heldout IoUs required")
        deltas.append(row["M1_after"]-row["M0_after"])
    mean, median, worst = float(np.mean(deltas)), float(np.median(deltas)), float(min(deltas))
    decrease = all(b["final_train_loss"] <= b["initial_train_loss"] for b in branches.values())
    return {"paired_mean_delta_iou": mean, "paired_median_delta_iou": median, "paired_worst_delta_iou": worst,
        "both_training_losses_nonincreasing": decrease,
        "hypothesis_gate_pass": decrease and mean >= .005 and median >= .01 and worst >= -.01,
        "adoption_authorized": False, "accuracy_verified": False}


def validate_receipts(raw, masks):
    common = {"status": "pass", "episode_index": 15, "input_track": "track_1", "input_sha256": VIDEO_SHA,
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    consensus.require_fields(raw, {**common, "stage": "sam3d_body_full_video_initializer", "total_video_frames": FRAMES,
        "frame_indices": list(range(FRAMES)), "script_sha256": BODY_SCRIPT_SHA, "predictions_sha256": PRED_SHA,
        "mask_report_sha256": MASK_SHA, "prompts_sha256": PROMPTS_SHA, "network": "none", "inference_type": "body",
        "body_revision": body.BODY_REVISION, "upstream_revision": body.UPSTREAM_REVISION,
        "camera_intrinsics": "RGB_size_default_FOV", "vertices": VERTICES, "faces": FACES,
        "geometry_units": "metres", "geometry_frame": "SAM3D_camera_x_right_y_down_z_forward",
        "translation": "vertices_camera_m = vertices_root_camera_m + pred_cam_t (exactly once)",
        "mhr_geometry_forward_verified": True, "human_mask_id": 0,
        "prompt_mode": "automatic_mask_and_derived_bbox_no_fallback", "model_mask_range": "uint8_0_1_matches_CARI_prepare_batch"})
    consensus.require_fields(masks, {**common, "stage": "automatic_masks", "frames": FRAMES, "script_sha256": MASK_SCRIPT_SHA})
    for receipt in (raw, masks):
        if "input_dataset_revision" in receipt and (type(receipt["input_dataset_revision"]) is not str or receipt["input_dataset_revision"] != body.DATASET_REVISION):
            raise ValueError("Present dataset revision must match audited manifest")
    rows = raw.get("frames")
    if not isinstance(rows, list) or len(rows) != FRAMES: raise ValueError("Full501 ordered producer records required")
    for index, row in enumerate(rows):
        if (type(row.get("frame_index")) is not int or row["frame_index"] != index
                or any(not re.fullmatch("[0-9a-f]{64}", str(row.get(k, ""))) for k in ("mask_sha256", "decoded_rgb_sha256"))):
            raise ValueError("Original frame/RGB/human-mask provenance differs")
    consensus.require_fields(raw.get("checkpoint_loading", {}),
        {"mode": "strict_network_and_head_state_with_explicit_asset_buffer_retention", "unexpected_keys": [], "parameter_tensors_loaded":1101})
    retained=raw["checkpoint_loading"].get("retained_mhr_asset_buffer_names")
    if not isinstance(retained,list) or len(retained)!=113 or len(set(retained))!=113:
        raise ValueError("Exact113 retained asset buffer identities required")
    consensus.require_fields(raw.get("inference_source_identity",{}),
        {"python_files":46,"sha256":"14f583ce78e8e786cb77739addc2358a1245b23f79ab676c4094231694fe8f3c"})


def load_predictions(path):
    shapes = {k: (FRAMES, *v) for k, v in body.PARAMETER_SHAPES.items() if k != "pred_vertices"}
    shapes.update(vertices_root_camera_m=(FRAMES, VERTICES, 3), vertices_camera_m=(FRAMES, VERTICES, 3),
                  faces=(FACES, 3), frame_index=(FRAMES,))
    kept = {*BLOCKS, "vertices_root_camera_m", "vertices_camera_m", "mhr_model_params", "pred_cam_t", "focal_length"}
    result = {}
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(shapes) or set(data.files) != set(shapes): raise ValueError("Exact original NPZ schema required")
        for key, shape in shapes.items():
            value = data[key]; dtype = np.int64 if key in ("frame_index", "faces") else np.float32
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError("All501 finite original native rows/dtypes required: "+key)
            if key == "frame_index" and not np.array_equal(value, np.arange(FRAMES)): raise ValueError("Original frame indices changed")
            if key == "expr_params" and np.any(value != 0): raise ValueError("Expressions must stay zero")
            if key == "focal_length" and not np.allclose(value, 1920., rtol=1e-6, atol=1e-4): raise ValueError("Original FOV changed")
            if key == "faces":
                if value.min() < 0 or value.max() >= VERTICES or np.any(np.diff(np.sort(value, axis=1), axis=1) == 0):
                    raise ValueError("Full native face topology invalid")
                result[key] = value.copy()
            elif key in kept: result[key] = value[list(SELECTED)].copy()
    if (not np.array_equal(result["vertices_camera_m"], result["vertices_root_camera_m"]+result["pred_cam_t"][:, None])
            or (result["pred_cam_t"][:, 2] <= 0).any()): raise ValueError("Native camera translation must occur exactly once")
    return result


def source_inputs(root):
    frozen = []; base = root/"outputs/episode_000015"; md = base/"automatic_masks"
    raw_path = consensus.freeze_input(base/"body_full/report.json", frozen, digest=BODY_SHA, size=BODY_BYTES)
    mask_path = consensus.freeze_input(md/"report.json", frozen, digest=MASK_SHA, size=MASK_BYTES)
    raw, masks = json.loads(raw_path.read_text()), json.loads(mask_path.read_text()); validate_receipts(raw, masks)
    manifest_path = consensus.freeze_input(root/"results/input-manifest.json", frozen)
    manifest = json.loads(manifest_path.read_text())
    consensus.require_fields(manifest, {"track": "track_1", "repo_id": "nvidia/video_to_data_challenge", "revision": body.DATASET_REVISION})
    meta, _ = body._manifest_file(root, manifest, "track_1/meta/episodes.jsonl"); consensus.freeze_input(meta, frozen)
    episodes = [json.loads(line) for line in meta.read_text().splitlines()]
    selected = [r for r in episodes if type(r.get("episode_index")) is int and r["episode_index"] == 15]
    if len(selected) != 1 or type(selected[0].get("length")) is not int or selected[0]["length"] != FRAMES:
        raise ValueError("Audited current manifest501 coverage required")
    video = "track_1/videos/chunk-000/observation.images.exo_camera/episode_000015.mp4"
    matches = [r for r in manifest["files"] if r.get("path") == video]
    if len(matches) != 1 or matches[0].get("sha256") != VIDEO_SHA: raise ValueError("Public video receipt identity differs")
    prompts_path = consensus.freeze_input(md/"prompts.json", frozen, digest=PROMPTS_SHA)
    prompts = json.loads(prompts_path.read_text()).get("prompts")
    if (not isinstance(prompts, list) or len(prompts) != 2 or {p.get("object_id") for p in prompts} != {0, 1}
            or any(p.get(k) is not None for p in prompts for k in ("points", "point_labels", "mask_path"))):
        raise ValueError("No manual or oracle prompts allowed")
    expected = [f"{i:06d}.png" for i in range(FRAMES)]
    for label in (0, 1):
        directory = md/f"masks/{label}"
        if (directory.resolve() != directory.absolute() or sorted(p.name for p in directory.iterdir()) != expected
                or any(p.is_symlink() or not p.is_file() for p in directory.iterdir())):
            raise ValueError("All501 automatic mask paths required")
    human, obj, records = [], [], []
    from PIL import Image
    for index in SELECTED:
        pair = []; row = {"frame_index": index, "decoded_rgb_sha256": raw["frames"][index]["decoded_rgb_sha256"]}
        for label, name in ((0, "human"), (1, "object")):
            path = consensus.freeze_input(md/f"masks/{label}/{index:06d}.png", frozen,
                digest=raw["frames"][index]["mask_sha256"] if label == 0 else None)
            with Image.open(path) as image:
                value = np.asarray(image)
                if image.mode != "L" or image.format != "PNG" or value.shape != (HEIGHT, WIDTH) or value.dtype != np.uint8 or not np.isin(value, [0,255]).all():
                    raise ValueError("Full-grid automatic binary PNG required")
                pair.append(value > 0)
            row[name+"_mask_sha256"] = frozen[-1][1]
        _, _, support = mask_protocol(*pair); row["train_holdout_observed_pixels"] = support
        human.append(pair[0]); obj.append(pair[1]); records.append(row)
    predictions = consensus.freeze_input(base/"body_full/predictions.npz", frozen, digest=PRED_SHA, size=PRED_BYTES)
    arrays = load_predictions(predictions)
    consensus.freeze_input(root/"results/weights-acquisition.json", frozen)
    assets, asset_ids = body._body_assets(root)
    if asset_ids != raw.get("body_assets") or asset_ids["model.ckpt"] != {"bytes": consensus.CHECKPOINT_BYTES, "sha256": consensus.CHECKPOINT_SHA}:
        raise ValueError("Original pinned head/checkpoint assets differ")
    for name, item in asset_ids.items(): consensus.freeze_input(assets/name, frozen, digest=item["sha256"], size=item["bytes"])
    installed = body._source_identity(root)
    if installed != raw.get("inference_source_identity"): raise ValueError("Original installed source differs")
    return arrays, np.stack(human), np.stack(obj), records, assets, asset_ids, installed, frozen


def freeze_npz(path, **arrays):
    with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
    path.chmod(0o444)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(arrays) or any(data[k].tobytes() != arrays[k].tobytes() for k in arrays):
            raise ValueError("Frozen artifact differs before scoring")
    return path, sha256(path)


def memory_check(torch, report):
    allocated = max(int(torch.cuda.memory_allocated()),int(torch.cuda.max_memory_allocated()))
    reserved = max(int(torch.cuda.memory_reserved()),int(torch.cuda.max_memory_reserved()))
    report["max_gpu_allocated_bytes"] = max(report.get("max_gpu_allocated_bytes", 0), allocated)
    report["max_gpu_reserved_bytes"] = max(report.get("max_gpu_reserved_bytes", 0), reserved)
    if max(allocated, reserved, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > MEMORY_LIMIT:
        raise MemoryError("Predeclared32GiB allocated/reserved GPU envelope exceeded")


def strict_forward(torch):
    if (not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark):
        raise ValueError("Every forward must retain predeclared strict flags")


def run(root, out, report, persist):
    raw, humans, objects, mask_records, assets, asset_ids, installed, frozen = source_inputs(root)
    for name,digest in {Path(__file__).name:report["script_sha256"],**report["helper_sha256"]}.items():
        consensus.freeze_input(Path(__file__).with_name(name),frozen,digest=digest)
    report.update(phase="native_decode", input_mask_records=mask_records, body_assets=asset_ids, inference_source_identity=installed,
        input_sha256=VIDEO_SHA, legacy_dataset_revision_omitted=True, legacy_geometry_forward_basis_omitted=True); persist()
    torch = consensus.strict_torch()
    report.update(torch_version=str(torch.__version__),cuda_version=torch.version.cuda);persist()
    native = root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root/"weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
    from lib_mhr.mhr_layer import MHRLayer
    layer_source = consensus.regular(Path(sys.modules[MHRLayer.__module__].__file__))
    if layer_source != native/"lib_mhr/mhr_layer.py": raise ValueError("Pinned native layer path differs")
    consensus.freeze_input(layer_source, frozen)
    buffer = out/"unverified_buffer_must_not_exist.pt"
    if buffer.exists() or buffer.is_symlink(): raise ValueError("No unverified buffer cache allowed")
    with torch.no_grad(), torch.jit.optimized_execution(False):
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets/"model.ckpt",
            buffer_path=buffer, mhr_model_path=assets/"assets/mhr_model.pt", device="cuda")
        head = layer.backend._ensure_head(torch.device("cuda"))
        head_source = consensus.regular(Path(sys.modules[type(head).__module__].__file__))
        if (head_source != Path("/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py") or head.enable_hand_model
                or tuple(head.scale_mean.shape) != (68,) or tuple(head.scale_comps.shape) != (28,68)
                or not np.array_equal(head.faces.cpu().numpy(), raw["faces"])): raise ValueError("Exact native full-body head/topology required")
        consensus.freeze_input(head_source, frozen)
        anchor_rows = [SELECTED.index(i) for i in ANCHORS]
        neutral = consensus.forward_blocks(head, {k: raw[k][anchor_rows] for k in BLOCKS}, torch, neutral=True)
        report["actual_head_calls"] += 1
        position, distances, costs = consensus.neutral_medoid(neutral.cpu().numpy().copy()); del neutral
        identities = {"M0": SELECTED.index(0), "M1": SELECTED.index(ANCHORS[position])}
        identity_file = freeze_npz(out/"identities.npz", anchor_indices=np.asarray(ANCHORS,np.int64),
            selected_identity_frame_indices=np.asarray([0,ANCHORS[position]],np.int64), neutral_vertex_rms_m=distances,
            shape45=np.stack([raw["shape_params"][r] for r in identities.values()]),
            scale_pca28=np.stack([raw["scale_params"][r] for r in identities.values()]))
        frozen.append(identity_file); report.update(medoid_anchor_index=ANCHORS[position], identity_frozen_before_optimization=True,
            identities_sha256=identity_file[1]); persist()
        root_v, controls = consensus.forward_blocks(head, raw, torch); report["actual_head_calls"] += 1
        rv = root_v.cpu().numpy().copy()*np.array([1,-1,-1],np.float32); raw_controls = controls.cpu().numpy().copy()
        raw_v = rv+raw["pred_cam_t"][:,None]; del root_v, controls
        parity = {"root_m":float(np.linalg.norm(rv.astype(np.float64)-raw["vertices_root_camera_m"],axis=-1).max()),
            "camera_m":float(np.linalg.norm(raw_v.astype(np.float64)-raw["vertices_camera_m"],axis=-1).max()),
            "controls":float(np.abs(raw_controls.astype(np.float64)-raw["mhr_model_params"]).max())}
        report["raw_native_parity"] = parity; persist()
        if not all(np.isfinite(v) and v <= 1e-5 for v in parity.values()): raise ValueError("Raw native parity exceeds1e-5")
        geometries = {}
        for name, row in identities.items():
            candidate = {k: raw[k].copy() for k in BLOCKS}
            for k in ("shape_params","scale_params"): candidate[k] = np.repeat(raw[k][row:row+1],len(SELECTED),axis=0)
            vertices, params = consensus.forward_blocks(head,candidate,torch); report["actual_head_calls"] += 1
            cp = params.cpu().numpy().copy()
            if cp[:,:136].tobytes() != raw_controls[:,:136].tobytes() or any(v.tobytes()!=cp[0,136:].tobytes() for v in cp[:,136:]):
                raise ValueError("Fixed poses/hands or shared68 native scales changed")
            geometries[name] = vertices.cpu().numpy().copy()*np.array([1,-1,-1],np.float32)+raw["pred_cam_t"][:,None]
            del vertices, params
    K = np.array([[1920.,0.,WIDTH/2.],[0.,1920.,HEIGHT/2.],[0.,0.,1.]],np.float64); low_K=np.diag([1/6.,1/6.,1.])@K
    for vertices in (raw_v,*geometries.values()):
        for frame in vertices: camera._mesh_inputs(frame,raw["faces"],K,WIDTH,HEIGHT,1e-4)
    geometry_file = freeze_npz(out/"frozen_inputs.npz",frame_index=np.asarray(SELECTED,np.int64),raw_vertices_camera_m=raw_v,
        M0_vertices_camera_m=geometries["M0"],M1_vertices_camera_m=geometries["M1"],faces=raw["faces"],camera_K=K,
        human_masks=humans,object_masks=objects)
    frozen.append(geometry_file); report.update(geometry_and_masks_frozen_before_optimization=True,frozen_inputs_sha256=geometry_file[1],
        native_layer_sha256=sha256(layer_source),native_head_sha256=sha256(head_source),phase="xy_optimization"); persist()
    del head,layer; torch.cuda.empty_cache()
    from pytorch3d.renderer import BlendParams, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader
    from pytorch3d.structures import Meshes
    import pytorch3d
    if pytorch3d.__version__ != "0.7.9": raise ValueError("Pinned PyTorch3D0.7.9 required")
    renderer_sources={}
    for value in (BlendParams,MeshRasterizer,SoftSilhouetteShader):
        path=consensus.freeze_input(Path(sys.modules[value.__module__].__file__),frozen)
        renderer_sources[value.__name__]={"file":str(path),"sha256":frozen[-1][1]}
    report.update(pytorch3d_version=pytorch3d.__version__,renderer_sources=renderer_sources,
        camera_K_full=K.tolist(),camera_K_low=low_K.tolist());persist()
    protocols = [mask_protocol(h,o) for h,o in zip(humans,objects)]
    observed = torch.tensor(np.stack([p[0] for p in protocols]),device="cuda",dtype=torch.float32)
    train = torch.tensor(np.stack([p[1] for p in protocols]),device="cuda",dtype=torch.float32); count=train.sum(dim=(1,2))
    cameras=camera._opencv_camera(torch,low_K,LOW_WIDTH,LOW_HEIGHT,batch_size=len(SELECTED))
    raster=MeshRasterizer(cameras=cameras,raster_settings=RasterizationSettings(image_size=(LOW_HEIGHT,LOW_WIDTH),blur_radius=BLUR,
        faces_per_pixel=FPP,perspective_correct=True,clip_barycentric_coords=False,cull_backfaces=False,cull_to_frustum=False,
        z_clip_value=None,max_faces_per_bin=FACES))
    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=SIGMA,gamma=GAMMA))
    faces_t=torch.tensor(raw["faces"],device="cuda",dtype=torch.int64)
    final_params = {}
    for name in ("M0","M1"):
        base=torch.tensor(geometries[name],device="cuda",dtype=torch.float32)
        w=torch.zeros((len(SELECTED),2),device="cuda",dtype=torch.float32,requires_grad=True)
        optimizer=torch.optim.Adam([w],lr=LR)
        branch={"actual_optimizer_steps":0,"actual_backward_calls":0,"all_backward_flags_restored":True,"train_history":[]}
        report["branches"][name]=branch
        def objective():
            strict_forward(torch)
            delta=DELTA_BOUND_M*torch.tanh(w/DELTA_BOUND_M)
            offset=torch.cat((delta,torch.zeros((len(SELECTED),1),device="cuda",dtype=torch.float32)),dim=1)
            memory_check(torch,report)
            meshes=Meshes(verts=list(base+offset[:,None]),faces=[faces_t]*len(SELECTED))
            alpha=shader(raster(meshes),meshes)[...,3]; report["actual_soft_raster_calls"]+=1
            strict_forward(torch)
            memory_check(torch,report)
            if not torch.isfinite(alpha).all() or bool((alpha<0).any()) or bool((alpha>1).any()): raise ValueError("Soft alpha invalid")
            return ((((alpha-observed)**2)*train).sum(dim=(1,2))/count).mean()+PRIOR_WEIGHT*((delta/PRIOR_STD_M)**2).mean()
        with torch.no_grad(): branch["initial_train_loss"]=float(objective().cpu())
        persist()
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True); loss=objective()
            if not torch.isfinite(loss): raise ValueError("Nonfinite training objective")
            scope=dict(scoped_backward_attempts=0,scoped_backward_completed=0,deterministic_flags_restored=False,
                original_autograd_backward_restored=False)
            with scoped_backward(torch,scope): loss.backward()
            if (scope["scoped_backward_completed"]!=1 or not scope["deterministic_flags_restored"]
                    or not scope["original_autograd_backward_restored"] or w.grad is None or not torch.isfinite(w.grad).all()):
                raise ValueError("Actual translation gradient/backward restoration failed")
            optimizer.step(); branch["actual_optimizer_steps"]+=1; branch["actual_backward_calls"]+=1
            branch["train_history"].append(float(loss.detach().cpu())); memory_check(torch,report); persist()
        with torch.no_grad(): branch["final_train_loss"]=float(objective().cpu())
        final_w=w.detach().cpu().numpy().copy()
        with torch.no_grad():delta=(DELTA_BOUND_M*torch.tanh(w/DELTA_BOUND_M)).cpu().numpy().copy()
        if not np.allclose(delta,bounded_xy(final_w),rtol=1e-6,atol=1e-8):raise ValueError("FrozenCUDA delta differs from bounded physical variables")
        final_params[name]=(final_w,delta); branch["max_abs_delta_m"]=float(np.abs(delta).max()); persist()
        del base,w,optimizer; torch.cuda.empty_cache()
    final_file=freeze_npz(out/"final_xy.npz",frame_index=np.asarray(SELECTED,np.int64),M0_w=final_params["M0"][0],M0_delta_xy_m=final_params["M0"][1],
        M1_w=final_params["M1"][0],M1_delta_xy_m=final_params["M1"][1])
    frozen.append(final_file);report.update(final_parameters_frozen_before_reserved_scoring=True,final_xy_sha256=final_file[1],phase="hard_holdout_scoring");persist()
    results={name:[] for name in ("raw","M0_before","M1_before","M0_after","M1_after")}
    for name in results:
        source=raw_v if name=="raw" else geometries[name[:2]].copy()
        if name.endswith("after"):
            unchanged=source.copy();delta=final_params[name[:2]][1]
            if not np.isfinite(delta).all() or np.abs(delta).max()>DELTA_BOUND_M:raise ValueError("FinalXY exceeds declared bound")
            source[:,:,:2]+=delta[:,None]
            if (source[:,:,2].tobytes()!=unchanged[:,:,2].tobytes()
                    or not np.allclose(source[:,:,:2]-unchanged[:,:,:2],delta[:,None],rtol=0,atol=5e-7)):
                raise ValueError("Final rigidXY changedZ or applied translation inconsistently")
        for j,frame in enumerate(source):
            memory_check(torch,report);mask,depth=camera.raster_camera_mesh(frame,raw["faces"],K,WIDTH,HEIGHT)
            report["actual_hard_raster_calls"]+=1; memory_check(torch,report)
            results[name].append(hard_evidence(mask.cpu().numpy(),humans[j],objects[j]));del mask,depth
        persist()
    report["records"]=[{"frame_index":index,**{k:results[k][j] for k in results}} for j,index in enumerate(SELECTED)]
    for path,digest in frozen:
        if sha256(consensus.regular(path))!=digest: raise ValueError("Frozen source/prediction/mask changed")
    if body._source_identity(root)!=installed: raise ValueError("Source changed during optimization")
    if (report["actual_head_calls"]!=4 or report["actual_soft_raster_calls"]!=64 or report["actual_hard_raster_calls"]!=115):
        raise ValueError("Exact4head/64soft/115hard executions required")
    report.update(**decision(report["records"],report["branches"]),status="pass",phase="complete",
        source_bindings_runtime_verified=True,frozen_inputs_rehashed_after_run=True,poses_hands_Z_K_identity_object_unchanged=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);rev=os.environ["WR_CODE_REVISION"];image=os.environ["WR_IMAGE_ID"]
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or root.resolve()!=root
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or not re.fullmatch("[0-9a-f]{40}",rev) or image!=consensus.IMAGE):
        raise ValueError("Canonical Azure offline pinned runtime required")
    out=root/BASE
    if out.resolve()!=out.absolute() or not out.is_dir() or any(out.iterdir()): raise FileExistsError("Exclusive empty new trial output required")
    start=time.perf_counter()
    helpers=("body_smoke.py","camera_render.py","identity_consensus_public.py","soft_silhouette_repro.py","soft_silhouette_probe.py")
    report={"stage":STAGE,"status":"fail","phase":"provenance","producer_revision":rev,"image_id":image,"script_sha256":sha256(Path(__file__)),
        "helper_sha256":{n:sha256(Path(__file__).with_name(n)) for n in helpers},"network":"none","input_track":"track_1","episode_index":15,
        "ground_truth_used":False,"hand_labeled_test":False,"oracle_modes":[],"accuracy_verified":False,"adoption_authorized":False,
        "submission_produced":False,"original_frames":FRAMES,"anchor_indices":list(ANCHORS),"midpoint_indices":list(MIDPOINTS),"selected_frame_indices":list(SELECTED),
        "body_report_sha256":BODY_SHA,"predictions_sha256":PRED_SHA,"mask_report_sha256":MASK_SHA,"body_legacy_script_sha256":BODY_SCRIPT_SHA,
        "mask_legacy_script_sha256":MASK_SCRIPT_SHA,"input_video_read":False,"legacy_source_executed_by_this_gate":False,
        "branch_order":["M0","M1"],"identity_selection":"M0_first_protocol_frame_M1_actual_anchor_neutral_vertex_RMS_medoid_no_image_score",
        "XY_bound_m":DELTA_BOUND_M,"learning_rate_m":LR,"requested_steps_per_branch":STEPS,"prior_std_m":PRIOR_STD_M,"prior_weight":PRIOR_WEIGHT,
        "mask_reduction":"human_majority19of36_object_exclude_anyof36","training_split":"even_checkerboard8low48full",
        "holdout_split":"odd_checkerboard8low48full_never_fitted","budget_seconds":BUDGET,"GPU_memory_limit_bytes":MEMORY_LIMIT,
        "sigma":SIGMA,"gamma":GAMMA,"faces_per_pixel":FPP,"blur_radius":BLUR,"branches":{},"records":[],"actual_head_calls":0,
        "actual_soft_raster_calls":0,"actual_hard_raster_calls":0,"numerical_reproducibility_verified":False,"bit_determinism_verified":False,
        "nondeterministic_backward_scope":"D92_single_backward_only_each_step_flags_restored","source_bindings_runtime_verified":False,
        "settings":{"seed":0,"CUBLAS_WORKSPACE_CONFIG":":4096:8","TF32":False,"deterministic_algorithms":True,"CPU_threads":4,"dtype":"float32"}}
    with (out/"report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-start;handle.seek(0);json.dump(report,handle,allow_nan=False);handle.write("\n")
            handle.truncate();handle.flush();os.fsync(handle.fileno())
        def expired(*_):raise TimeoutError("Coupled XY trial exceeded600s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();run(root,out,report,persist)
        except BaseException as error:
            report.update(status="fail",error=type(error).__name__,message=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();(out/"report.json").chmod(0o444)
    print(json.dumps({k:report[k] for k in ("stage","status","hypothesis_gate_pass","elapsed_seconds")}))


if __name__=="__main__":main()

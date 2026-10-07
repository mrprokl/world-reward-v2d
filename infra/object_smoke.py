"""Offline, fixed-frame Track 1 object generation; not metric-scale validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from body_smoke import EPISODE, TRACK1_EPISODE_COUNT, _pinned_checkout, _validate_inputs
from world_reward.data import sha256
from world_reward.artifact_paths import episode_output


DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    parser.add_argument("--aligned-pointmap", action="store_true",
                        help="Use already human-anchored MoGe2 XYZ and identical K, not independent MoGe1")
    return parser


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    args = _argument_parser().parse_args()
    root = args.root.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    inputs = _validate_inputs(root, episode_index=args.episode)
    base = episode_output(root, args.episode)
    acquisition = json.loads((root / "results/weights-acquisition.json").read_text())
    principal = [r for r in acquisition["assets"] if r["repo_id"] == "facebook/sam-3d-objects"]
    if len(principal) != 1 or principal[0]["revision"] != "2e73555018d2741ccd486e56c24fac41155a1dc6":
        raise RuntimeError("Require pinned Objects checkpoint acquisition")
    weights = root / "weights/sam3d"
    hub = weights / "torch_home/hub"
    repository = hub / "facebookresearch_dinov2_main"
    _pinned_checkout(repository, DINOV2_REVISION)
    aux = json.loads((root / "results/auxiliary-assets.json").read_text())
    checkpoints = [r for r in aux["checkpoints"] if "_reg4_" in r["filename"]]
    if {r["filename"] for r in checkpoints} != {"dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth"} or len(checkpoints) != 2:
        raise RuntimeError("Require both explicitly acquired Objects DINO register checkpoints")
    for record in checkpoints:
        if sha256(hub / "checkpoints" / record["filename"]) != record["sha256"]:
            raise RuntimeError("Objects DINO checkpoint changed after acquisition")
    output = base / ("object_grounded" if args.aligned_pointmap else "object_smoke")
    if output.exists():
        raise RuntimeError("Frozen object smoke output exists; do not overwrite")
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no local/CPU fallback")
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(inputs["video"]))
    try:
        ok, bgr = cap.read()
        if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != 1:
            raise RuntimeError("Original frame zero decode failed")
    finally:
        cap.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask_path = base / "automatic_masks/masks/1/000000.png"
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 255]).all() or not (mask > 0).any():
        raise RuntimeError("Fixed frame has no valid original-resolution object mask")
    grounding, grounding_arguments = None, {}
    if args.aligned_pointmap:
        alignment_path = base / "scale_smoke/report.json"
        alignment = json.loads(alignment_path.read_text())
        required = {"stage": "predicted_human_anchored_moge2_pointmaps", "status": "pass",
                    "episode_index": args.episode, "input_track": "track_1", "input_sha256": inputs["video_sha256"],
                    "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
                    "coordinate_frame": "OpenCV_x_right_y_down_z_forward",
                    "pointmap_scale_application": "one_clip_scalar_to_MoGe2_XYZ_already_applied"}
        if (any(alignment.get(key) != value for key, value in required.items())
                or alignment["ground_truth_used"] is not False or alignment["hand_labeled_test"] is not False):
            raise RuntimeError("Require verified same-video human-anchored pointmap provenance")
        records = [record for record in alignment["pointmaps"] if record["frame_index"] == 0]
        evidence = [record for record in alignment["human_evidence"] if record["frame_index"] == 0]
        if len(records) != 1 or len(evidence) != 1:
            raise RuntimeError("Require exactly one original-frame-zero alignment record")
        grounding = dict(records[0], alignment_report_sha256=sha256(alignment_path))
        if (grounding["decoded_rgb_sha256"] != hashlib.sha256(rgb.tobytes()).hexdigest()
                or evidence[0]["object_mask_sha256"] != sha256(mask_path)):
            raise RuntimeError("Pointmap RGB/object mask differs from object-generation frame")
        for field in ("pointmap", "intrinsics"):
            path = Path(grounding[field + "_path"])
            expected_path = alignment_path.parent / ("000000.npy" if field == "pointmap" else "000000_intrinsics.json")
            if path != expected_path or sha256(path) != grounding[field + "_sha256"]:
                raise RuntimeError("Aligned grounding path/hash mismatch")
        points = np.load(grounding["pointmap_path"], allow_pickle=False)
        if points.shape != (*rgb.shape[:2], 3):
            raise RuntimeError("Same-camera pointmap original resolution mismatch")
        valid = np.isfinite(points).all(-1) & (points[..., 2] > 0)
        if not (valid & (mask > 0)).any():
            raise RuntimeError("Same-camera pointmap lacks finite visible object observations")
        grounding_arguments = {"pointmap_path": grounding["pointmap_path"],
                               "pointmap_intrinsics_path": grounding["intrinsics_path"]}
    original_hub_load = torch.hub.load
    calls = []
    def local_load(repo_or_dir, *arguments, **kwargs):
        if repo_or_dir not in ("facebookresearch/dinov2", "facebookresearch/dinov2:main"):
            raise RuntimeError(f"Unexpected Torch Hub request: {repo_or_dir}")
        model = kwargs.get("model", arguments[0] if arguments else None)
        if model not in ("dinov2_vitl14_reg", "dinov2_vitb14_reg"):
            raise RuntimeError(f"Unexpected Objects backbone model: {model}")
        calls.append(model)
        kwargs["source"] = "local"
        return original_hub_load(str(repository), *arguments, **kwargs)
    torch.hub.load = local_load
    try:
        import sam3d_objects
        from v2d.sam3d.lib.image_to_mesh import image_to_mesh
        package = Path(sam3d_objects.__file__).parent
        source_hashes = {str(path.relative_to(package)): sha256(path)
                         for path in sorted(package.rglob("*.py"))}
        output.mkdir(parents=True, exist_ok=False)
        frame = output / "frame_000000.png"
        Image.fromarray(rgb).save(frame)
        image_to_mesh(str(frame), str(mask_path), str(output / "object.glb"),
                      str(output / "transform.json"), str(output / "intrinsics.json"),
                      str(weights), seed=0, with_mesh_postprocess=False,
                      with_texture_baking=False, with_layout_postprocess=False,
                      use_vertex_color=True, **grounding_arguments)
    finally:
        torch.hub.load = original_hub_load
    import trimesh
    mesh = trimesh.load(output / "object.glb", force="mesh")
    if (not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 3 or len(mesh.faces) < 1
            or not np.isfinite(mesh.vertices).all() or (mesh.faces < 0).any()
            or (mesh.faces >= len(mesh.vertices)).any() or not np.isfinite(mesh.area) or mesh.area <= 0):
        raise RuntimeError("Generated object mesh is nonfinite/empty/invalid")
    transform = json.loads((output / "transform.json").read_text())
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    scale = np.asarray(transform["scale"], dtype=float)
    if (rotation.shape != (4,) or translation.shape != (3,) or scale.shape != (3,)
            or not np.isfinite(np.concatenate((rotation, translation, scale))).all()
            or not np.isclose(np.linalg.norm(rotation), 1., atol=1e-5) or (scale <= 0).any()):
        raise RuntimeError("Generated pose must have finite unit quaternion, translation and positive scale")
    if grounding is not None:
        if json.loads((output / "intrinsics.json").read_text()) != json.loads(Path(grounding["intrinsics_path"]).read_text()):
            raise RuntimeError("Object output intrinsics must preserve the explicit grounding camera")
    report = {"stage": "sam3d_objects_grounded_fixed_frame" if args.aligned_pointmap else "sam3d_objects_fixed_frame_smoke",
              "status": "pass", "execution_verified": True, "candidate_accuracy_validated": False, "episode_index": args.episode,
              "frame_index": 0, "seed": 0, "elapsed_seconds": time.perf_counter() - started,
              "vertices": len(mesh.vertices), "faces": len(mesh.faces),
              "bounds_generated_units": mesh.bounds.tolist(), "surface_area_generated_units": float(mesh.area),
              "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
              "transform": transform, "dinov2_revision": DINOV2_REVISION, "hub_calls": calls,
              "transform_sha256": sha256(output / "transform.json"),
              "intrinsics_sha256": sha256(output / "intrinsics.json"),
              "pointmap_grounding": grounding,
              "scale_source": "already_human_anchored_MoGe2_no_second_scalar" if args.aligned_pointmap else "independent_MoGe1",
              "installed_objects_source_hashes": source_hashes,
              "installed_source_commit_verified": False,
              "object_sha256": sha256(output / "object.glb"), "input_sha256": inputs["video_sha256"],
              "mask_sha256": sha256(mask_path), "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
              "oracle_modes": [], "metric_scale_verified": False, "submission_eligible": False,
              "challenge_performance_verified": False, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("stage", "status", "elapsed_seconds", "vertices",
                                                  "faces", "watertight", "metric_scale_verified")}))


if __name__ == "__main__":
    main()

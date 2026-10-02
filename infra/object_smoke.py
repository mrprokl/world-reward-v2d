"""Offline, fixed-frame Track 1 object generation; not metric-scale validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from body_smoke import _pinned_checkout, _validate_inputs, DINOV3_REVISION
from world_reward.data import sha256


DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    inputs = _validate_inputs(root)
    acquisition = json.loads((root / "results/weights-acquisition.json").read_text())
    principal = [r for r in acquisition["assets"] if r["repo_id"] == "facebook/sam-3d-objects"]
    if len(principal) != 1 or principal[0]["revision"] != "2e73555018d2741ccd486e56c24fac41155a1dc6":
        raise RuntimeError("Require pinned Objects checkpoint acquisition")
    weights = root / "weights/sam3d"
    hub = weights / "torch_home/hub"
    repository = hub / "facebookresearch_dinov2_main"
    _pinned_checkout(repository, DINOV2_REVISION)
    aux = json.loads((root / "results/auxiliary-assets.json").read_text())
    for record in aux["checkpoints"]:
        if "_reg4_" in record["filename"] and sha256(hub / "checkpoints" / record["filename"]) != record["sha256"]:
            raise RuntimeError("Objects DINO checkpoint changed after acquisition")
    output = root / "outputs/episode_000015/object_smoke"
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
    mask_path = root / "outputs/episode_000015/automatic_masks/masks/1/000000.png"
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 255]).all() or not (mask > 0).any():
        raise RuntimeError("Fixed frame has no valid original-resolution object mask")
    original_hub_load = torch.hub.load
    calls = []
    def local_load(repo_or_dir, *arguments, **kwargs):
        if repo_or_dir not in ("facebookresearch/dinov2", "facebookresearch/dinov2:main"):
            raise RuntimeError(f"Unexpected Torch Hub request: {repo_or_dir}")
        model = kwargs.get("model", arguments[0] if arguments else None)
        if model not in ("dinov2_vitl14_reg", "dinov2_vitb14_reg", "dinov2_vitl14", "dinov2_vitb14"):
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
                      use_vertex_color=True)
    finally:
        torch.hub.load = original_hub_load
    import trimesh
    mesh = trimesh.load(output / "object.glb", force="mesh")
    if (not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 3 or len(mesh.faces) < 1
            or not np.isfinite(mesh.vertices).all() or (mesh.faces < 0).any()
            or (mesh.faces >= len(mesh.vertices)).any() or not np.isfinite(mesh.area) or mesh.area <= 0):
        raise RuntimeError("Generated object mesh is nonfinite/empty/invalid")
    transform = json.loads((output / "transform.json").read_text())
    report = {"stage": "sam3d_objects_fixed_frame_smoke", "status": "pass", "episode_index": 15,
              "frame_index": 0, "seed": 0, "elapsed_seconds": time.perf_counter() - started,
              "vertices": len(mesh.vertices), "faces": len(mesh.faces),
              "bounds_generated_units": mesh.bounds.tolist(), "surface_area_generated_units": float(mesh.area),
              "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
              "transform": transform, "dinov2_revision": DINOV2_REVISION, "hub_calls": calls,
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

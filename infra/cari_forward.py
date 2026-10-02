"""Offline native CoCoNet forward with strict original-video input provenance."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

from body_smoke import _pinned_checkout, _source_identity, _body_assets
from cari_runner import build_cari_forward_command, build_cari_runtime_environment, CHECKPOINT_SHA256, UPSTREAM_REVISION
from world_reward.contracts import require_rigid_transforms
from world_reward.data import sha256


DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def main():
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-only", action="store_true", help="Gate checkpoint loading only, not actual network forward")
    args = parser.parse_args()
    import torch
    root = Path(os.environ["WR_ROOT"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU/local fallback")
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    source_identity = _source_identity(root)
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    assets = root / "weights/cari4d/sam3d_body"
    body_assets, body_hashes = _body_assets(root)
    body_report = json.loads((root / "outputs/episode_000015/body_full/report.json").read_text())
    if body_hashes != body_report["body_assets"] or (body_assets / "mhr_buffers.pt").exists():
        raise RuntimeError("Native CARI decoder must use original verified Body assets, not an unverified compact-buffer override")
    torch_home = assets / "torch_home"
    repository = torch_home / "hub/facebookresearch_dinov2_main"
    _pinned_checkout(repository, DINOV2_REVISION)
    auxiliary = json.loads((root / "results/auxiliary-assets.json").read_text())
    for filename in ("dinov2_vitb14_pretrain.pth", "dinov2_vits14_pretrain.pth"):
        records = [record for record in auxiliary["checkpoints"] if record["filename"] == filename]
        if len(records) != 1 or sha256(torch_home / "hub/checkpoints" / filename) != records[0]["sha256"]:
            raise RuntimeError("Exact offline CoCoNet DINO checkpoints missing or altered")
    acquisition = json.loads((root / "results/weights-acquisition.json").read_text())
    records = [record for record in acquisition["assets"] if record["repo_id"] == "nvidia/cari4d_commercial"]
    if len(records) != 1 or records[0]["revision"] != "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9":
        raise RuntimeError("Require pinned commercial CoCoNet checkpoint acquisition")
    candidates = list((root / "weights/cari4d").rglob("step200000.pth"))
    if len(candidates) != 1:
        raise RuntimeError("Missing or ambiguous CoCoNet checkpoint path")
    checkpoint = candidates[0]
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("CoCoNet checkpoint bytes differ from audited release")
    base = root / "outputs/episode_000015"
    output = base / ("cari_kernel" if args.kernel_only else "cari_forward")
    if output.exists():
        raise RuntimeError("Frozen native CoCoNet result exists")
    inputs_path = None
    if args.kernel_only:
        environment = build_cari_runtime_environment(str(native), str(assets), str(torch_home),
                                                     "/workspace/v2d_sam3d_body/lib")
    else:
        inputs_path = base / "cari_inputs/report.json"
        inputs = json.loads(inputs_path.read_text())
        expected = {"stage": "world_reward_native_cari_inputs", "status": "pass", "input_track": "track_1",
                    "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
        if (any(inputs.get(key) != value for key, value in expected.items())
                or inputs["ground_truth_used"] is not False or inputs["hand_labeled_test"] is not False):
            raise RuntimeError("Require verified full no-oracle CARI inputs")
        for field in ("depth_h5", "mhr_init", "object_poses"):
            if sha256(Path(inputs[field])) != inputs["file_sha256"][field]:
                raise RuntimeError("Frozen CARI preparation changed")
        plan = build_cari_forward_command(str(native), inputs["export_seq"], inputs["depth_h5"], inputs["mhr_init"],
                                          inputs["object_poses"], str(checkpoint), str(output / "coconet.pth"),
                                          total_frames=inputs["frames"], mhr_assets_root=str(assets), torch_home=str(torch_home),
                                          sam3d_source_root="/workspace/v2d_sam3d_body/lib")
        environment = plan["environment"]
    os.environ.update(environment)
    os.environ["MPLCONFIGDIR"] = "/tmp/world-reward-matplotlib"
    # Insertion order matters: SAM Body also exports a top-level tools package.
    # Use the already-tested native-first child ordering in one operation.
    sys.path[:0] = environment["PYTHONPATH"].split(":")
    # Tensor cache completeness cannot stop Hub querying GitHub. Enforce the
    # pinned local checkout explicitly, and reject any unexpected hub source.
    original_hub = torch.hub.load
    hub_calls = []
    def local_hub(repo_or_dir, *arguments, **kwargs):
        if repo_or_dir not in ("facebookresearch/dinov2", "facebookresearch/dinov2:main"):
            raise RuntimeError(f"Unexpected CoCoNet Hub source: {repo_or_dir}")
        model = kwargs.get("model", arguments[0] if arguments else None)
        if model not in ("dinov2_vitb14", "dinov2_vits14"):
            raise RuntimeError(f"Unexpected CoCoNet DINO model: {model}")
        hub_calls.append(model)
        kwargs["source"] = "local"
        return original_hub(str(repository), *arguments, **kwargs)
    torch.hub.load = local_hub
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        from tools import run_mhr_wild_inference as native_inference
        if Path(native_inference.__file__).resolve() != (native / "tools/run_mhr_wild_inference.py").resolve():
            raise RuntimeError("Native CoCoNet import resolved to an unrelated tools package")
        run_mhr_wild_inference = native_inference.run_mhr_wild_inference
        if args.kernel_only:
            from tools.run_mhr_wild_inference import _load_config, _load_checkpoint_model
            cfg = _load_config(native / "learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml")
            model, step, provenance = _load_checkpoint_model(cfg, checkpoint, torch.device("cuda"), offline_supervision_contract=True)
            if not all(torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
                raise RuntimeError("Native model contains nonfinite checkpoint state")
            bundle_path = None
            metadata = {"checkpoint_step": step, "inference_provenance": provenance,
                        "model_state_finite": True, "actual_network_forward_verified": False}
        else:
            bundle_path = run_mhr_wild_inference(inputs["export_seq"], inputs["depth_h5"], inputs["mhr_init"],
                                                  inputs["object_poses"], native / "learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml",
                                                  checkpoint, output / "coconet.pth", stride=96, render_batch_size=32,
                                                  crop_workers=8, crop_buffer_count=2, use_input_cache=False,
                                                  device_name="cuda", offline_supervision_contract=True)
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            if (bundle["gt"] != {} or bundle["metadata"]["ground_truth_used"] is not False
                    or bundle["frames"] != [f"{index:06d}" for index in range(inputs["frames"])]):
                raise RuntimeError("Native inference changed timeline or introduced GT")
            from lib_mhr.schema import MHR_PARAM_DIMS
            import numpy as np
            for key, dimension in MHR_PARAM_DIMS.items():
                value = np.asarray(bundle["pr"][key])
                if value.shape != (inputs["frames"], dimension) or not np.isfinite(value).all():
                    raise RuntimeError(f"Native CoCoNet produced invalid full {key}")
            pose = np.asarray(bundle["pr"]["pose_abs"])
            require_rigid_transforms(pose, inputs["frames"])
            metadata = {"native_metadata": bundle["metadata"], "actual_network_forward_verified": True,
                        "full_original_frame_coverage_verified": True, "human_shared_identity_verified": False}
    finally:
        torch.hub.load = original_hub
    result = {"stage": "native_cari_checkpoint_load_gate" if args.kernel_only else "world_reward_native_cari_full_forward",
              "status": "pass", "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
              "oracle_modes": [], "metadata": metadata, "hub_calls": hub_calls, "dinov2_revision": DINOV2_REVISION,
              "inference_source_identity": source_identity, "checkpoint_sha256": sha256(checkpoint),
              "inputs_report_sha256": None if inputs_path is None else sha256(inputs_path),
              "episode_inputs_used": not args.kernel_only,
              "bundle_sha256": None if bundle_path is None else sha256(bundle_path),
              "method": "native_checkpoint_state_load_only" if args.kernel_only else "native_CoCoNet_forward_with_own_ICP_Viterbi_object_initializer_no_FoundationPose",
              "submission_eligible": False, "challenge_performance_verified": False,
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "status": "pass", "elapsed_seconds": result["elapsed_seconds"],
                      "challenge_performance_verified": False}))


if __name__ == "__main__":
    main()

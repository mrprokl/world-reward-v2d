"""Remote synthetic MHR forward/converter gate; no challenge records or GT."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time

from world_reward.data import sha256
from world_reward.submission import parameters_from_official_converter


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("MHR model/mesh processing is restricted to Azure Linux")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/srv/scenesmith/world-reward"))
    args = parser.parse_args()
    root = args.root
    import numpy as np
    import torch

    model_path = root / "weights/mhr/mhr_model.pt"
    if sha256(model_path) != "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc":
        raise RuntimeError("Reference model differs from the official Apache MHR")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    spec = importlib.util.spec_from_file_location("official_converter", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
    pose = torch.zeros(3, 136, device="cuda")
    pose[:, 0] = torch.tensor([0.0, 0.1, 0.2], device="cuda")
    pose[:, 3:6] = torch.tensor([[0., 0., 0.], [0., .05, 0.], [0., .10, 0.]], device="cuda")
    identity = torch.zeros(3, 45, device="cuda")
    scales = torch.zeros(3, 68, device="cuda")
    expression = torch.zeros(3, 72, device="cuda")
    with torch.inference_mode():
        native, joints = model(identity, torch.cat([pose, scales], dim=1), expression, True)
        vertices = native * torch.tensor([1., -1., -1.], device="cuda") / 100.
    if tuple(vertices.shape) != (3, 18439, 3) or not torch.isfinite(vertices).all():
        raise RuntimeError("MHR forward topology/finite contract failed")
    del model
    result = module.convert(
        vertices.cpu().numpy(), str(model_path), device="cuda", precision="float32",
        model_batch=256, log=lambda *args, **kwargs: None,
    )
    params = parameters_from_official_converter(result, max_mean_vertex_error_mm=0.01)
    reference = module.MHR(str(model_path), "cuda", chunk=256, precision="float32")
    recovered, _ = reference.run(
        torch.tensor(params.pose, device="cuda", dtype=torch.float64),
        torch.tensor(np.concatenate([params.scales, params.shape])[None], device="cuda", dtype=torch.float64),
    )
    error_mm = torch.linalg.vector_norm(recovered / 1000. - vertices.double(), dim=-1).mean(1) * 1000.
    if not torch.isfinite(error_mm).all() or float(error_mm.max()) > 0.01:
        raise RuntimeError("Independent model-forward geometry residual exceeds 0.01 mm")
    report = {
        "stage": "synthetic_mhr_converter", "status": "pass", "frames": 3,
        "mean_vertex_error_mm": float(error_mm.mean()), "worst_frame_mean_mm": float(error_mm.max()),
        "mhr_geometry_forward_verified": True, "challenge_performance_verified": False,
        "model_sha256": sha256(model_path), "converter_sha256": sha256(tool),
        "script_sha256": sha256(Path(__file__)), "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started, "inputs": "synthetic_native_mhr_only",
    }
    (root / "results/synthetic-mhr-converter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

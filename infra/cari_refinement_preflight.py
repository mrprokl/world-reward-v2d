"""CPU-only native import/config gate before waiting for the expensive GPU run."""

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import platform
import sys

from cari_refine import OPTIMIZER_RELATIVE_PATH, OPTIMIZER_SHA256, UPSTREAM_REVISION, require_asset_receipt
from body_smoke import _pinned_checkout
from world_reward.data import sha256


def main():
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure network-none CPU preflight")
    root = Path(os.environ["WR_ROOT"])
    vendor = root / "vendor/video_to_data"
    _pinned_checkout(vendor, UPSTREAM_REVISION)
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    if sha256(native / OPTIMIZER_RELATIVE_PATH) != OPTIMIZER_SHA256:
        raise ValueError("Native refinement optimizer source changed")
    require_asset_receipt(json.loads((root / "results/cari-refinement-assets.json").read_text()))
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    from learning.training import mhr_opt_refineout as optimizer
    if Path(optimizer.__file__).resolve() != (native / OPTIMIZER_RELATIVE_PATH).resolve():
        raise ValueError("Native refinement module shadowed")
    config = asdict(optimizer.MHRParityPostOptConfig())
    for key, value in {"num_steps": 300, "batch_size": 0, "frame_start": 0, "frame_limit": 0,
                       "freeze_object_rotation": True, "freeze_body_internal_translations": True}.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError("Native default full-clip parity configuration changed")
    modules = ("torch", "pytorch3d.transforms", "pytorch3d.loss.point_mesh_distance", "nvdiffrast.torch",
               "transformers", "kaolin.metrics.trianglemesh", "kaolin.ops.mesh", "Utils")
    for name in modules:
        importlib.import_module(name)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight must not instantiate a CUDA context")
    print(json.dumps({"stage": "native_cari_refinement_cpu_import_preflight", "status": "pass",
                      "optimizer_sha256": OPTIMIZER_SHA256, "modules_imported": list(modules),
                      "default_config": config, "GPU_initialized": False,
                      "actual_refinement_verified": False, "challenge_performance_verified": False}), flush=True)


if __name__ == "__main__":
    main()

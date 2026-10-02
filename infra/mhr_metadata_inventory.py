"""Read-only scripted MHR metadata inventory; never forward or semantic proof."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time
from world_reward.data import sha256

MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def inventory(model):
    modules = list(model.named_modules())
    if len(modules) > 512: raise ValueError("Unexpectedly large scripted module inventory")
    names = []
    for path, module in modules:
        for attribute in ("joint_names", "parameter_names"):
            value = getattr(module, attribute, None)
            if isinstance(value, (list, tuple)) and 0 < len(value) <= 512 and all(isinstance(v, str) for v in value):
                names.append({"module": path, "attribute": attribute, "values": list(value)})
    methods = []
    for name in model._c._method_names():
        schema = str(model._c._get_method(name).schema)
        methods.append({"name": name, "schema": schema[:4096], "schema_truncated": len(schema) > 4096})
    return {"modules": [path for path, _ in modules], "methods": methods,
            "buffers": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in model.named_buffers()],
            "exposed_string_metadata": names, "joint_names_available": any(v["attribute"] == "joint_names" for v in names),
            "parameter_names_available": any(v["attribute"] == "parameter_names" for v in names)}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CPU container with network none")
    root, revision, image = Path(os.environ["WR_ROOT"]), os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source revision and Docker image ID")
    model_path, output = root / "weights/mhr/mhr_model.pt", root / "results/mhr-metadata-inventory.json"
    if output.exists() or output.is_symlink(): raise FileExistsError("Metadata inventory report is frozen")
    if not model_path.is_file() or any(p.is_symlink() for p in (model_path, *model_path.parents) if p.is_relative_to(root)):
        raise ValueError("Require regular pinned MHR model without symlinks")
    def expired(*args): raise TimeoutError("Metadata inventory exceeded120s")
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(120)
    started = time.perf_counter()
    try:
        if sha256(model_path) != MODEL_SHA: raise ValueError("Reference MHR model digest mismatch")
        import torch
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        result = inventory(model)
        try: momentum = importlib.metadata.version("pymomentum")
        except importlib.metadata.PackageNotFoundError: momentum = None
        report = {"stage": "reference_mhr_scripted_metadata_inventory", "status": "pass", **result,
                  "model_sha256": MODEL_SHA, "model_bytes": model_path.stat().st_size, "torch": torch.__version__,
                  "pymomentum_distribution_version": momentum, "source_image_id": image, "code_revision": revision,
                  "script_sha256": sha256(Path(__file__)), "execution_device": "CPU", "network": "none",
                  "model_forward_performed": False, "joint_semantic_mapping_verified": False,
                  "challenge_inputs_used": False, "challenge_performance_verified": False,
                  "next_step": "audit_exposed_names" if result["joint_names_available"] and result["parameter_names_available"] else "requires_pinned_official_metadata_acquisition",
                  "elapsed_seconds": time.perf_counter() - started}
        with output.open("x") as handle: handle.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "status", "joint_names_available", "parameter_names_available", "next_step")}), flush=True)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__": main()

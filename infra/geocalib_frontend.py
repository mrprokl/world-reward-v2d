"""Experimental standalone GeoCalib field frontend, not its camera solver.

Original Apache-2.0 MSCAN/Hamburger attribution and licenses are retained with
the pinned assets. Only verified native class definitions are executed; the
GeoCalib package initializer, PerspectiveFields and LM optimizer are excluded.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time
from typing import Dict

import numpy as np

import geocalib_assets as assets

REPORT = "results/geocalib-frontend-smoke-v1.json"
IMAGE_ID = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"


def validate_assets(root):
    target = assets.safe_path(Path(root)/assets.ASSETS)
    receipt_path = assets.safe_path(Path(root)/assets.REPORT)
    receipt = json.loads(receipt_path.read_text())
    required = {"stage": "pinned_geocalib_frontend_assets_acquisition", "status": "pass", "phase": "complete",
                "source_revision": assets.SOURCE_REV, "segnext_revision": assets.SEGNEXT_REV,
                "source_subset_verified": True, "publisher_digest_available": False, "publisher_digest_verified": False,
                "camera_solver_acquired": False, "challenge_inputs_used": False, "ground_truth_used": False}
    if any(receipt.get(k) != v or (type(v) is bool and type(receipt.get(k)) is not bool) for k, v in required.items()):
        raise ValueError("Require exact successful acquisition receipt")
    expected = {name: (size, sha) for name, _, size, sha in assets.SOURCE_RECORDS}
    expected.update({"classes18_89.py": (assets.CLASS_BYTES, assets.CLASS_SHA),
                     "CREDITS.txt": (len(assets.CREDITS), hashlib.sha256(assets.CREDITS).hexdigest())})
    entries = receipt.get("files", {})
    if set(entries) != set(expected)|{"geocalib-pinhole.tar"}: raise ValueError("Unexpected source/weight inventory")
    weight = entries["geocalib-pinhole.tar"]
    if weight.get("bytes") != assets.WEIGHT_BYTES or weight.get("url") != assets.WEIGHT_URL or not re.fullmatch(r"[0-9a-f]{64}", weight.get("sha256", "")):
        raise ValueError("Require actual downloaded weight identity")
    expected["geocalib-pinhole.tar"] = (assets.WEIGHT_BYTES, weight["sha256"])
    if {p.name for p in target.iterdir()} != set(expected): raise ValueError("Asset directory has extra/missing entries")
    for name, (size, sha) in expected.items():
        path = assets.safe_path(target/name)
        if not path.is_file() or path.stat().st_size != size or assets.digest(path) != sha:
            raise ValueError("Immutable asset bytes changed: "+name)
        if (entries[name].get("bytes"), entries[name].get("sha256")) != (size, sha): raise ValueError("Receipt source identity differs")
    assets.extract_classes((target/"geocalib.audit.py").read_bytes())
    return target, {"receipt_sha256": assets.digest(receipt_path), "files": entries, "source_revision": assets.SOURCE_REV}


def build_frontend(target, torch):
    """Execute exact vendor modules and class slice without importing its package."""
    source = (target/"modules.py").read_bytes()
    record = assets.SOURCE_RECORDS[0]
    if (len(source), hashlib.sha256(source).hexdigest()) != (record[2], record[3]):
        raise ValueError("Standalone vendor module hash differs")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append(node.module or "")
    if set(imports) != {"typing", "torch", "torch.nn", "torch.nn.functional", "torch.nn.modules.utils"}:
        raise ValueError("Unexpected standalone vendor import closure")
    spec = importlib.util.spec_from_file_location("wr_geocalib_modules", target/"modules.py")
    modules = importlib.util.module_from_spec(spec); sys.modules[spec.name] = modules; spec.loader.exec_module(modules)
    classes = (target/"classes18_89.py").read_bytes()
    if hashlib.sha256(classes).hexdigest() != assets.CLASS_SHA: raise ValueError("Class bytes changed")
    namespace = {"__name__": "wr_geocalib_frontend_classes", "torch": torch, "nn": torch.nn,
                 "F": torch.nn.functional, "Dict": Dict, "ConvModule": modules.ConvModule, "LightHamHead": modules.LightHamHead}
    exec(compile(classes, str(target/"classes18_89.py"), "exec"), namespace)

    class Frontend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = modules.MSCAN()
            self.ll_enc = namespace["LowLevelEncoder"]()
            self.perspective_decoder = namespace["PerspectiveDecoder"]()

        def forward(self, data):
            features = {"hl": self.backbone(data)["features"], "ll": self.ll_enc(data)["features"]}
            return self.perspective_decoder({"features": features})

    return Frontend()


def strict_checkpoint(model, checkpoint, torch, report):
    """Full state (parameters AND buffers); native second-component mapping only."""
    if not isinstance(checkpoint, dict): raise ValueError("Checkpoint must be a mapping")
    report["checkpoint_top_level_keys"] = sorted(str(k) for k in checkpoint)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) for k in state):
        raise ValueError("Checkpoint model must be nonempty string-key mapping")
    expected = model.state_dict(); original_keys = sorted(state)
    report.update(checkpoint_state_keys=original_keys, state_key_mapping="none",
                  checkpoint_state_inventory=[{"key": key, "shape": list(state[key].shape), "dtype": str(state[key].dtype)}
                                              if torch.is_tensor(state[key]) else {"key": key, "type": type(state[key]).__name__}
                                              for key in original_keys])
    if not (set(state)&set(expected)):
        mapped = [".".join(k.split(".")[:1]+k.split(".")[2:]) for k in state]
        if any(len(k.split(".")) < 3 for k in state) or len(set(mapped)) != len(mapped):
            raise ValueError("Native key transformation is invalid or colliding")
        state = dict(zip(mapped, state.values())); report["state_key_mapping"] = "remove_second_component_once"
    report.update(missing_keys=sorted(set(expected)-set(state)), unexpected_keys=sorted(set(state)-set(expected)))
    if report["missing_keys"] or report["unexpected_keys"]: raise ValueError("Strict full frontend inventory mismatch")
    inventory = []; count = 0
    for key in sorted(expected):
        value = state[key]; reference = expected[key]
        if not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("Checkpoint shape/dtype differs: "+key)
        if not bool(torch.isfinite(value).all().item()): raise ValueError("Nonfinite checkpoint state: "+key)
        inventory.append({"key": key, "shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()})
        count += value.numel()
    report.update(state_inventory=inventory, state_numel=count, state_all_finite=True,
                  parameter_keys=len(dict(model.named_parameters())), buffer_keys=len(dict(model.named_buffers())))
    model.load_state_dict(state, strict=True)
    report["state_load_strict"] = True


def procedural_rgb():
    """Two new unlabelled own RGB arrays; no calibration/render truth supplied."""
    y, x = np.indices((320, 416)); images = np.empty((2, 320, 416, 3), dtype=np.float32)
    images[0] = np.stack((.15+.65*x/415, .2+.55*y/319, .35+.1*np.sin(x/19)), axis=-1)
    grid = (x % 43 < 3)|(y % 37 < 3); images[0, grid] = [.12, .08, .06]
    images[1] = np.stack((.7-.3*y/319, .5+.2*x/415, .3+.2*y/319), axis=-1)
    chair = (((x>105)&(x<289)&(y>78)&(y<170)) | ((x>98)&(x<298)&(y>180)&(y<204))
             | ((x>110)&(x<124)&(y>=204)&(y<292)) | ((x>272)&(x<286)&(y>=204)&(y<292)))
    images[1, chair] = [.19, .3, .58]
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


def validate_fields(fields, torch):
    shapes = {"up_field": (2,2,320,416), "latitude_field": (2,1,320,416),
              "up_confidence": (2,320,416), "latitude_confidence": (2,320,416)}
    if not isinstance(fields, dict) or set(fields) != set(shapes): raise ValueError("Unexpected native field keys")
    result = {}
    for name, shape in shapes.items():
        value = fields[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("Native field shape/dtype/finite failure: "+name)
        low, high = float(value.min().item()), float(value.max().item())
        if (name.endswith("confidence") and not 0 <= low <= high <= 1) or (name == "latitude_field" and not -np.pi/2 <= low <= high <= np.pi/2):
            raise ValueError("Native field bounds failure: "+name)
        result[name] = {"shape": list(shape), "dtype": str(value.dtype), "min": low, "max": high, "mean": float(value.mean().item())}
    error = float((torch.linalg.vector_norm(fields["up_field"], dim=1)-1).abs().max().item())
    if error > 32*torch.finfo(torch.float32).eps: raise ValueError("Up field is not unit-normalized")
    result["up_unit_max_error"] = error
    return result


def seed_all(torch):
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)


def run(root, report, persist):
    target, binding = validate_assets(root); report["assets"] = binding; persist()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("Require CUDA H100")
    report.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0))
    if "H100" not in report["device_name"]: raise RuntimeError("Require H100 engineering gate")
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    seed_all(torch); report["phase"] = "construct_and_strict_load"; persist()
    model = build_frontend(target, torch)
    checkpoint = torch.load(target/"geocalib-pinhole.tar", map_location="cpu", weights_only=True)
    strict_checkpoint(model, checkpoint, torch, report); del checkpoint
    model = model.eval().cuda(); image = torch.from_numpy(procedural_rgb()).cuda()
    report["input"] = {"shape": list(image.shape), "rgb_range": [0,1], "normalization": "native_internal_RGB_to_BGR_times255", "procedural_labels_present": False}
    torch.cuda.reset_peak_memory_stats(); outputs = []
    with torch.inference_mode():
        for index in range(2):
            seed_all(torch); report["phase"] = "frontend_call:"+str(index); persist()
            fields = model({"image": image}); torch.cuda.synchronize()
            report["fields"] = validate_fields(fields, torch)
            outputs.append({k: v.detach().cpu() for k, v in fields.items()})
            del fields
    report["bit_exact_replay"] = all(torch.equal(outputs[0][k], outputs[1][k]) for k in outputs[0])
    if not report["bit_exact_replay"]: raise RuntimeError("Identical seeded native frontend calls differ")
    _, after = validate_assets(root)
    if after != binding: raise RuntimeError("Assets changed during smoke")
    if any(name == "geocalib" or name.startswith("geocalib.") for name in sys.modules): raise RuntimeError("Forbidden full GeoCalib package imported")
    report.update(peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), actual_frontend_calls=2,
                  assets_rechecked=True, full_geocalib_package_imported=False, status="pass", phase="complete")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = assets.safe_path(Path(os.environ["WR_ROOT"])); path = assets.safe_path(root/REPORT)
    revision = os.environ.get("WR_CODE_REVISION", "")
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or not root.is_dir()
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or os.environ.get("WR_IMAGE_ID") != IMAGE_ID
            or Path("/sys/class/net").is_dir() and set(p.name for p in Path("/sys/class/net").iterdir()) != {"lo"}):
        raise RuntimeError("Require canonical Linux root, committed image/source and network-none")
    report = {"stage": "standalone_geocalib_native_frontend_smoke", "status": "fail", "phase": "asset_audit",
              "code_revision": revision, "script_sha256": assets.digest(Path(__file__)), "image_id": IMAGE_ID,
              "source_revision": assets.SOURCE_REV, "budget_seconds": 180, "ground_truth_used": False,
              "challenge_inputs_used": False, "private_truth_read": False, "oracle_modes": [], "accuracy_verified": False,
              "camera_estimated": False, "adoption_authorized": False, "full_package_license_eligibility_verified": False,
              "publisher_weight_digest_verified": False, "network": "none"}
    started = time.perf_counter()
    reserved = os.environ.get("WR_GEOCALIB_REPORT_RESERVED") == "1"
    if reserved and (not path.is_file() or path.stat().st_size != 0 or path.stat().st_uid != os.geteuid()):
        raise ValueError("Reserved report must be an empty owned regular file")
    with path.open("r+" if reserved else "x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Frontend smoke exceeds frozen180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(180)
        try: persist(); run(root, report, persist)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)[:1000]); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()

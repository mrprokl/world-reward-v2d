"""Frozen failed-fixture replay diagnosis, not new hand-quality validation.

Only strict CUDA and strict CPU within-device raw-bit replay authorize PASS.
Default CUDA and cross-device differences are diagnostic, never tolerances.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time

import numpy as np
from world_reward.data import sha256

MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
GROUPS = (("default_cuda", "cuda", False), ("strict_cuda", "cuda", True), ("strict_cpu", "cpu", True))
TOTAL_SECONDS, MAX_CALLS, SEED = 120, 20, 0


def own_inputs():
    """Exact semantic-v2 216-row fixture; explicit MHRDemo row ABI."""
    controls = np.zeros((216, 204), np.float32)
    for i in range(54): controls[i*4:(i+1)*4, 68+i] = [-.001, .001, -.002, .002]
    return np.zeros((216, 45), np.float32), controls, np.zeros((216, 72), np.float32)


def bit_comparison(first, replay):
    if len(first) != 2 or len(replay) != 2: raise ValueError("Require vertices and global skeleton")
    result = {}
    for label, a, b in zip(("vertices", "skeleton"), first, replay):
        a, b = np.asarray(a), np.asarray(b)
        if (a.shape != b.shape or a.dtype != b.dtype or a.dtype.kind != "f"
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Replay must have matching finite floating arrays")
        aa, bb = np.ascontiguousarray(a), np.ascontiguousarray(b)
        raw_a, raw_b = aa.view(np.uint8).reshape(-1), bb.view(np.uint8).reshape(-1)
        element_a, element_b = aa.view(f"V{a.dtype.itemsize}"), bb.view(f"V{b.dtype.itemsize}")
        result[label] = {"raw_byte_differences": int(np.count_nonzero(raw_a != raw_b)),
                         "raw_element_differences": int(np.count_nonzero(element_a != element_b)),
                         "max_absolute_difference": float(np.max(np.abs(a.astype(np.float64)-b))),
                         "first_sha256": hashlib.sha256(raw_a).hexdigest(),
                         "replay_sha256": hashlib.sha256(raw_b).hexdigest()}
    result["bitexact"] = all(result[key]["raw_byte_differences"] == 0 for key in ("vertices", "skeleton"))
    return result


def script_sources(model):
    """Hash actual exported code; names are observed, not assumed LBS API."""
    sources = []
    for name, module in model.named_modules():
        try: code = getattr(module, "code", None)
        except (AttributeError, RuntimeError): continue
        if not isinstance(code, str): continue
        if name == "" or "lbs" in name.lower() or any(op in code for op in ("index_add", "scatter", "linear_blend")):
            sources.append({"module": name, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                            "code_bytes": len(code.encode()), "methods": list(module._c._method_names()),
                            "index_add_occurrences": code.count("index_add"), "scatter_occurrences": code.count("scatter")})
    if not any(entry["module"] == "" for entry in sources): raise ValueError("Actual scripted forward source unavailable")
    return sources


def outcome(groups):
    """Default instability does not hide a strict failure or missing CPU audit."""
    for required in ("strict_cuda", "strict_cpu"):
        group = next((g for g in groups if g.get("name") == required), None)
        if (group is None or group.get("status") != "complete" or len(group.get("conditions", [])) != 2
                or [c.get("apply_correctives") for c in group["conditions"]] != [False, True]
                or any(len(c.get("replays", [])) != 2 or not all(r.get("bitexact") is True for r in c["replays"])
                       for c in group["conditions"])):
            return False
    return True


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require remote Linux CUDA container with network none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source revision and image ID")
    output = root / "results/mhr-determinism.json"
    if output.is_symlink(): raise FileExistsError("Determinism report is frozen")
    with output.open("x") as handle:
        started = time.perf_counter()
        report = {"stage": "reference_mhr_failed_fixture_determinism_diagnosis", "status": "fail", "phase": "integrity",
                  "script_sha256": sha256(Path(__file__)), "code_revision": revision, "source_image_id": image,
                  "network": "none", "challenge_inputs_used": False, "hand_accuracy_verified": False,
                  "challenge_performance_verified": False, "adoption_performed": False,
                  "known_failed_fixture": True, "hypothesis_validation": False,
                  "bitexact_required_within_strict_device": True, "cross_device_bitexact_required": False,
                  "budgets": {"seconds_total": TOTAL_SECONDS, "forward_calls": MAX_CALLS, "CPU_threads": 4},
                  "batch_rows": {"neutral": 1, "finger_perturbations": 216}, "repeats": 3, "seed": SEED,
                  "groups": [], "forward_calls": 0}
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False)+"\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Frozen determinism diagnosis exceeded120s")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(TOTAL_SECONDS)
        try:
            persist()
            model_path = root / "weights/mhr/mhr_model.pt"
            if (not model_path.is_file() or model_path.stat().st_size != 696110248
                    or any(p.is_symlink() for p in (model_path, *model_path.parents) if p.is_relative_to(root))
                    or sha256(model_path) != MODEL_SHA): raise ValueError("Pinned regular MHR model mismatch")
            if "torch" in sys.modules: raise RuntimeError("CUBLAS workspace must precede torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU is a diagnosis, not fallback")
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            report.update(model_sha256=MODEL_SHA, model_bytes=model_path.stat().st_size, torch=torch.__version__,
                          CUDA=torch.version.cuda, CUBLAS_WORKSPACE_CONFIG=":4096:8", execution_dtype="float32",
                          TF32=False, cudnn_benchmark=False, phase="replay")
            inputs = own_inputs(); first_outputs = {}; gpu = model = None; source_evidence = None
            def forward(model, tensors, correctives):
                if report["forward_calls"] >= MAX_CALLS: raise RuntimeError("Frozen call budget exceeded")
                report["forward_calls"] += 1; persist()
                with torch.inference_mode(): value = model(*tensors, correctives)
                if tensors[0].device.type == "cuda": torch.cuda.synchronize()
                if not isinstance(value, tuple) or len(value) != 2: raise ValueError("Unexpected reference result ABI")
                n = len(tensors[0])
                if tuple(value[0].shape) != (n, 18439, 3) or tuple(value[1].shape) != (n, 127, 8):
                    raise ValueError("Unexpected reference geometry dimensions")
                result = tuple(v.detach().cpu().numpy().copy() for v in value)
                if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in result):
                    raise ValueError("Reference geometry must be finite float32")
                return result
            for name, device, strict in GROUPS:
                group = {"name": name, "device": device, "strict_algorithms": strict, "status": "fail", "conditions": []}
                report["groups"].append(group); report["active_group"] = name; persist()
                try:
                    torch.use_deterministic_algorithms(strict, warn_only=False)
                    torch.backends.cudnn.deterministic = strict
                    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
                    if device == "cuda":
                        if gpu is None: gpu = torch.jit.load(str(model_path), map_location=device).eval()
                        model = gpu
                    else:
                        gpu = model = None; torch.cuda.empty_cache()
                        model = torch.jit.load(str(model_path), map_location=device).eval()
                    if any(v.is_floating_point() and v.dtype != torch.float32 for _, v in (*model.named_parameters(), *model.named_buffers())):
                        raise ValueError("Pinned model requires original float32; no dtype mutation allowed")
                    sources = script_sources(model)
                    if source_evidence is None:
                        source_evidence = sources; report["scripted_source_evidence"] = sources
                        report["forward_schema"] = str(model._c._get_method("forward").schema)
                        report["scripted_forward_source_verified"] = True
                        report["called_function_kernel_implementation_verified"] = False
                        report["nondeterministic_kernel_identity_verified"] = False
                        if (len(model.get_joint_names()), len(model.get_parameter_names()), model.get_num_identity_blendshapes(),
                                model.get_num_face_expression_blendshapes()) != (127, 249, 45, 72):
                            raise ValueError("Frozen MHRDemo metadata ABI differs")
                    elif sources != source_evidence: raise ValueError("CPU/CUDA scripted source differs")
                    tensors = tuple(torch.as_tensor(v.copy(), device=device) for v in inputs)
                    for correctives in (False, True):
                        condition = {"apply_correctives": correctives, "replays": [], "completed_calls": 0}
                        group["conditions"].append(condition); persist()
                        if name == "default_cuda":
                            neutral = forward(model, tuple(torch.zeros_like(t[:1]) for t in tensors), correctives)
                            condition["neutral_output_sha256"] = [hashlib.sha256(v.tobytes()).hexdigest() for v in neutral]
                        first = forward(model, tensors, correctives); condition["completed_calls"] += 1
                        first_outputs[name, correctives] = first; persist()
                        for _ in range(2):
                            replay = forward(model, tensors, correctives); condition["completed_calls"] += 1
                            condition["replays"].append(bit_comparison(first, replay)); persist()
                        if any(not np.array_equal(t.detach().cpu().numpy(), v) for t, v in zip(tensors, inputs)):
                            raise ValueError("Model mutated own fixed input controls")
                    group["status"] = "complete"
                except (RuntimeError, ValueError) as error:
                    group["error_type"] = type(error).__name__; group["error"] = str(error)[:2000]
                persist()
            report["contrasts_descriptive_only"] = []
            for left, right in (("default_cuda", "strict_cuda"), ("strict_cuda", "strict_cpu")):
                for correctives in (False, True):
                    if (left, correctives) in first_outputs and (right, correctives) in first_outputs:
                        report["contrasts_descriptive_only"].append({"left": left, "right": right,
                            "apply_correctives": correctives, **bit_comparison(first_outputs[left, correctives], first_outputs[right, correctives])})
                        persist()
            report["strict_replay_bitexact"] = outcome(report["groups"])
            default = next((g for g in report["groups"] if g["name"] == "default_cuda"), {})
            default_changed = any(r.get("bitexact") is False for c in default.get("conditions", []) for r in c["replays"])
            report["default_cuda_replay_changed"] = default_changed
            report["strict_repair_route_observed"] = default_changed and report["strict_replay_bitexact"]
            report["causal_kernel_mechanism_verified"] = False
            if not report["strict_replay_bitexact"]: raise ValueError("Strict CUDA and CPU bitexact replay not both verified")
            report.update(status="pass", phase="complete"); persist()
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)[:2000]); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps({k: report[k] for k in ("stage", "status", "forward_calls", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()

"""Offline partial native SS conditioner/dynamics gate, never shape generation."""

from __future__ import annotations

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

import multiview_native_gate as prep

SS_SHA256 = "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146"
SS_BYTES = 6690136964
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODALITIES = {"image", "rgb_image", "mask", "rgb_image_mask", "pointmap", "rgb_pointmap"}


def _identity(path):
    with Path(path).open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest(), "bytes": Path(path).stat().st_size}


def validate_preprocess(root, source_identity, image_id):
    path = Path(root) / "results/multiview-native-preprocess-gate-v2.json"
    report = json.loads(path.read_text())
    expected = {"stage": "native_mv_sam3d_import_config_preprocess", "status": "pass",
                "vendor_revision": prep.PIN, "image_id": image_id, "procedural_inputs_only": True,
                "challenge_inputs_used": False, "ground_truth_used": False, "models_loaded": False}
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise RuntimeError("SS requires a passed native preprocessing gate on this exact source/image")
    config = Path(root) / "weights/sam3d/hf-download/checkpoints/pipeline.yaml"
    if (report.get("source_manifest") != source_identity or report.get("pipeline_config") != prep._identity(config)
            or report.get("script", {}).get("sha256") != prep._identity(Path(prep.__file__))["sha256"]):
        raise RuntimeError("Passed preprocessing source/config/script binding differs")
    return _identity(path)


def fusion_reference(predictions, pose_keys):
    """Tiny NumPy algebra contract; native runtime comparisons use Torch directly."""
    if not predictions or any(set(p) != set(predictions[0]) for p in predictions):
        raise ValueError("Fusion requires nonempty predictions with identical keys")
    result = {}
    for key in predictions[0]:
        arrays = [np.asarray(p[key]) for p in predictions]
        if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Fusion predictions must be finite with matching shapes")
        result[key] = arrays[0].copy() if key in pose_keys else np.stack(arrays).mean(0)
    return result


def _seed(torch):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _tensor_hash(tensor):
    # byte view supports BF16 without inventing a NumPy BF16 dtype.
    value = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _check(torch, actual, reference):
    if set(actual) != set(reference):
        raise RuntimeError("Native dynamics changed latent keys")
    errors = {}
    for key, expected in reference.items():
        value = actual[key]
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            raise RuntimeError(f"Native output shape/dtype/finite contract failed: {key}")
        error = float(torch.abs(value - expected).max())
        tolerance = 8 * torch.finfo(value.dtype).eps * max(1., float(torch.abs(expected).max()))
        relative = float((torch.abs(value - expected) / torch.abs(expected).clamp_min(torch.finfo(value.dtype).eps)).max())
        errors[key] = {"max_absolute_error": error, "max_relative_error_eps_denominator": relative, "tolerance": tolerance}
        if error > tolerance:
            raise RuntimeError(f"Native fusion/replay mismatch: {key}, {error} > {tolerance}")
    return errors


def _dynamics(torch, pipe, generator, inputs, report, started):
    from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view, POSE_KEYS
    signal.alarm(max(1, min(120, int(300 - (time.perf_counter() - started)))))
    consumed, handles = [], []
    fuser = pipe.condition_embedders["ss_condition_embedder"]
    names = [key for _, info in fuser.embedder_list for key, _ in info]
    if set(names) != MODALITIES or len(names) != 6 or fuser.training or fuser.force_drop_modalities:
        raise RuntimeError("Require actual six-modality native Fuser in evaluation without forced drops")
    for module, info in fuser.embedder_list:
        keys = [key for key, _ in info]
        count = [0]
        def trace(_module, args, keys=keys, count=count):
            consumed.append({"key": keys[count[0] % len(keys)], "shape": list(args[0].shape), "dtype": str(args[0].dtype)})
            count[0] += 1
        handles.append(module.register_forward_pre_hook(trace))
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.dtype):
            conditions, kwargs = pipe.get_multi_view_condition_input(fuser, inputs, pipe.ss_condition_input_mapping)
    finally:
        for handle in handles:
            handle.remove()
    tokens = conditions[0]
    if kwargs or not torch.is_tensor(tokens) or tokens.ndim != 4 or tokens.shape[:2] != (3, 1) or not torch.isfinite(tokens).all():
        raise RuntimeError("Native multi-view condition batching contract failed")
    if len(consumed) != 18 or any({p["key"] for p in consumed[i:i+6]} != MODALITIES for i in range(0, 18, 6)):
        raise RuntimeError("Native conditioner did not consume six inputs for each view")
    _seed(torch)
    shapes = {key: (1, latent.pos_emb.shape[0], latent.input_layer.in_features)
              for key, latent in generator.reverse_fn.backbone.latent_mapping.items()}
    state = generator._generate_noise(shapes, pipe.device)
    frozen = {key: _tensor_hash(value) for key, value in state.items()}
    tokens_hash = _tensor_hash(tokens)
    times, d = generator._prepare_t_and_d()
    t = float(times[0])
    original = generator._generate_dynamics
    report.update({"condition_tokens": {"shape": list(tokens.shape), "dtype": str(tokens.dtype), "sha256": _tensor_hash(tokens)},
                   "consumed_modalities": consumed, "latent_shapes": {k: list(v) for k, v in shapes.items()},
                   "latent_dtypes": {k: str(v.dtype) for k, v in state.items()}, "latent_sha256": frozen,
                   "native_t": t, "native_d": float(d), "fusion_tolerance_rule": "8*dtype_eps*max(1,max_abs_reference)"})
    def call(condition, views=None):
        _seed(torch)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.dtype):
            if views is None:
                result = original(state, t, d, condition)
            else:
                with inject_generator_multi_view(generator, views, generator.inference_steps, mode="multidiffusion"):
                    result = generator._generate_dynamics(state, t, d, condition)
        if (generator._generate_dynamics != original or _tensor_hash(tokens) != tokens_hash
                or any(_tensor_hash(state[k]) != frozen[k] for k in state)):
            raise RuntimeError("Native callable, conditions or latent inputs were not restored/unchanged")
        torch.cuda.synchronize()
        return result
    predictions = [call(tokens[i]) for i in range(3)]
    report["replay"] = _check(torch, call(tokens[0]), predictions[0])
    report["single_view"] = _check(torch, call(tokens[:1], 1), predictions[0])
    report["duplicate_three"] = _check(torch, call(tokens[:1].repeat(3, 1, 1, 1), 3), predictions[0])
    reference = {key: predictions[0][key] if key in POSE_KEYS else torch.stack([p[key] for p in predictions]).mean(0)
                 for key in predictions[0]}
    report["native_vs_manual_fusion"] = _check(torch, call(tokens, 3), reference)
    report["nonanchor_permutation"] = _check(torch, call(tokens[[0, 2, 1]], 3), reference)
    report["native_conditioner_verified"] = report["native_dynamics_verified"] = True


def run(root, source, report, started):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from body_smoke import _pinned_checkout
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    if not torch.cuda.is_available():
        raise RuntimeError("Require CUDA")
    workspace = root / "weights/sam3d/hf-download/checkpoints"
    config = OmegaConf.to_container(OmegaConf.load(workspace / "pipeline.yaml"), resolve=True)
    if config.get("ss_generator_config_path") != "ss_generator.yaml" or config.get("ss_generator_ckpt_path") != "ss_generator.ckpt":
        raise RuntimeError("Unexpected pinned SS checkpoint/config paths")
    checkpoint = _identity(workspace / "ss_generator.ckpt")
    if checkpoint["sha256"] != SS_SHA256 or checkpoint["bytes"] != SS_BYTES:
        raise RuntimeError("Pinned existing SS checkpoint hash/bytes mismatch")
    report.update({"ss_checkpoint": checkpoint, "ss_config": _identity(workspace / "ss_generator.yaml")})
    hub = root / "weights/sam3d/torch_home/hub"
    repository = hub / "facebookresearch_dinov2_main"
    _pinned_checkout(repository, DINO_REVISION)
    auxiliary = json.loads((root / "results/auxiliary-assets.json").read_text())
    records = [r for r in auxiliary["checkpoints"] if r["filename"] == "dinov2_vitl14_reg4_pretrain.pth"]
    if len(records) != 1 or _identity(hub / "checkpoints" / records[0]["filename"])["sha256"] != records[0]["sha256"]:
        raise RuntimeError("Require acquired DINOv2-L register checkpoint integrity")
    original_hub = torch.hub.load
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    def local_load(repo_or_dir, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if repo_or_dir not in ("facebookresearch/dinov2", "facebookresearch/dinov2:main") or model != "dinov2_vitl14_reg":
            raise RuntimeError("Unexpected SS Torch Hub model request")
        kwargs["source"] = "local"
        return original_hub(str(repository), *args, **kwargs)
    torch.hub.load = local_load
    try:
        pipe = object.__new__(InferencePipelinePointMap)
        pipe.device, pipe.dtype, pipe.workspace_dir = torch.device("cuda"), pipe._get_dtype(config["dtype"]), str(workspace)
        pipe.clip_pointmap_beyond_scale = config.get("clip_pointmap_beyond_scale")
        pipe.depth_model = lambda *_: (_ for _ in ()).throw(RuntimeError("External maps required; no depth model"))
        pipe.ss_preprocessor = instantiate(config["ss_preprocessor"])
        pipe.ss_condition_input_mapping = config["ss_condition_input_mapping"]
        generator = pipe.init_ss_generator("ss_generator.yaml", "ss_generator.ckpt")
        conditioner = pipe.init_ss_condition_embedder("ss_generator.yaml", "ss_generator.ckpt")
        pipe.override_ss_generator_cfg_config(generator, **{key: config[name] for key, name in
            (("cfg_strength", "ss_cfg_strength"), ("inference_steps", "ss_inference_steps"), ("rescale_t", "ss_rescale_t"),
             ("cfg_interval", "ss_cfg_interval"), ("cfg_strength_pm", "ss_cfg_strength_pm")) if name in config})
        report["no_shortcut_before_native_sampling_override"] = bool(generator.no_shortcut)
        report["no_shortcut_override_source"] = "pinned sample_sparse_structure_multi_view non-distilled path line1055"
        generator.no_shortcut = True
        generator.eval().requires_grad_(False)
        conditioner.eval().requires_grad_(False)
        pipe.condition_embedders = {"ss_condition_embedder": conditioner}
        inputs = []
        for view in range(3):
            rgba, points, _, _ = prep.procedural_fixture()
            rgba = np.roll(rgba, 2 * view, axis=1).copy()
            points *= 1 + view * .05  # Same local rays, distinct procedural depth; no GT inference.
            computed = pipe.compute_pointmap(rgba, torch.tensor(points))
            inputs.append(pipe.preprocess_image(rgba, pipe.ss_preprocessor, pointmap=computed["pointmap"]))
        report.update({"models_loaded": True, "dino_revision": DINO_REVISION, "dino_checkpoint": records[0],
                       "cfg_strength": generator.reverse_fn.strength, "no_shortcut": generator.no_shortcut})
        _dynamics(torch, pipe, generator, inputs, report, started)
        imported = {name: prep._identity(Path(module.__file__)) for name, module in sys.modules.copy().items()
                    if (name == "sam3d_objects" or name.startswith("sam3d_objects.")) and getattr(module, "__file__", None)}
        if any(not Path(record["path"]).resolve().is_relative_to(source.resolve()) for record in imported.values()):
            raise RuntimeError("Mixed vendor/installed native import tree")
        report["imported_source"] = imported
    finally:
        torch.hub.load = original_hub
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux network-none container")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require immutable code revision")
    output = root / "results/multiview-ss-one-call-gate.json"
    if output.exists():
        raise FileExistsError("SS gate report is frozen")
    report = {"stage": "native_mv_sam3d_partial_ss_one_call", "status": "fail", "code_revision": revision,
              "vendor_revision": prep.PIN, "image_id": image, "script": _identity(Path(__file__)),
              "procedural_inputs_only": True, "ground_truth_used": False, "challenge_inputs_used": False,
              "models_loaded": False, "native_constructor_verified": False, "native_conditioner_verified": False,
              "native_dynamics_verified": False, "shape_generation_implemented": False, "slat_loaded": False,
              "entropy_fusion_verified": False, "adoption_authorized": False, "submission_eligible": False,
              "license_status": "SAM_custom_competition_eligibility_unresolved"}
    started = time.perf_counter()
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("SS gate frozen total300s/forward120s budget exceeded")))
    signal.alarm(300)
    try:
        source, identity = prep.validate_source(root)
        report["preprocess_gate"] = validate_preprocess(root, identity, image)
        run(root, source, report, started)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report["elapsed_seconds"] = time.perf_counter() - started
        with output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()

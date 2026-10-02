"""Lightweight planning/asset/provenance contracts; no Torch/render execution."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


INFRA = Path(__file__).parents[1] / "infra"
old = {name: sys.modules.get(name) for name in ("multiview_native_gate", "multiview_ss_gate")}
try:
    for name in old:
        spec = importlib.util.spec_from_file_location(name, INFRA / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location("full_gate_tests", INFRA / "multiview_full_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
finally:
    for name, previous in old.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def pipeline_config():
    return {"_target_": gate.prep.TARGET, "depth_model": {"_target_": "forbidden.download"},
            "compile_model": True, "dtype": "float16", "ss_preprocessor": {"nested": [1, 2]},
            "slat_cfg_strength": 1, "slat_rescale_t": 1,
            **{name + "_" + key + "_path": name + extension
               for name in gate.CHECKPOINTS for key, extension in (("config", ".yaml"), ("ckpt", ".ckpt"))}}


def test_config_only_explicit_offline_native_overrides_without_mutation(tmp_path):
    config = pipeline_config()
    saved = copy.deepcopy(config)
    result = gate.constructor_config(config, tmp_path)
    assert config == saved
    assert result["depth_model"] is None
    assert result["compile_model"] is False
    assert result["rendering_engine"] == "pytorch3d"
    assert result["layout_post_optimization_method"] is None
    assert result["workspace_dir"] == str(tmp_path)
    for key in ("dtype", "slat_cfg_strength", "slat_rescale_t", "ss_preprocessor"):
        assert result[key] == saved[key]
    result["ss_preprocessor"]["nested"].append(3)
    assert config == saved


@pytest.mark.parametrize("key,value", [("_target_", "other.Model"), ("ss_generator_ckpt_path", "../model.ckpt"),
                                      ("slat_decoder_gs_config_path", "/tmp/other.yaml"),
                                      ("slat_decoder_mesh_ckpt_path", "mesh.safetensors"),
                                      ("ss_encoder_ckpt_path", "unexpected.ckpt")])
def test_unexpected_constructor_paths_fail_before_heavy_import(tmp_path, key, value):
    config = pipeline_config()
    config[key] = value
    with pytest.raises(RuntimeError):
        gate.constructor_config(config, tmp_path)


def test_independent_inventory_contains_all_six_native_loads_and_config_hashes():
    assert set(gate.CHECKPOINTS) == {"ss_generator", "slat_generator", "ss_decoder", "slat_decoder_gs",
                                      "slat_decoder_gs_4", "slat_decoder_mesh"}
    assert set(gate.YAMLS) == {"pipeline", *gate.CHECKPOINTS}
    for inventory in (gate.CHECKPOINTS, gate.YAMLS):
        for digest, size in inventory.values():
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
            assert type(size) is int and size > 0
    assert gate.CHECKPOINTS["ss_generator"] == (gate.ss.SS_SHA256, gate.ss.SS_BYTES)


def prerequisite(tmp_path, monkeypatch):
    workspace = tmp_path / "weights/sam3d/hf-download/checkpoints"
    workspace.mkdir(parents=True)
    (workspace / "ss_generator.yaml").write_text("tiny native config")
    preprocess = {"path": "preprocess_v2", "sha256": "a" * 64, "bytes": 7}
    source = {"path": "vendor_source", "sha256": "b" * 64}
    image = "sha256:" + "c" * 64
    monkeypatch.setattr(gate.ss, "validate_preprocess", lambda root, identity, digest: preprocess)
    data = {"stage": "native_mv_sam3d_partial_ss_one_call", "status": "pass", "vendor_revision": gate.prep.PIN,
            "image_id": image, "procedural_inputs_only": True, "ground_truth_used": False,
            "challenge_inputs_used": False, "native_conditioner_verified": True, "native_dynamics_verified": True,
            "shape_generation_implemented": False, "entropy_fusion_verified": False,
            "preprocess_gate": copy.deepcopy(preprocess), "script": gate.ss._identity(Path(gate.ss.__file__)),
            "ss_config": gate.ss._identity(workspace / "ss_generator.yaml"),
            "ss_checkpoint": {"path": str(workspace / "ss_generator.ckpt"), "sha256": gate.ss.SS_SHA256,
                              "bytes": gate.ss.SS_BYTES}}
    path = tmp_path / "results/multiview-ss-one-call-gate.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    return path, data, source, image


def test_actual_ss_prerequisite_accepts_exact_bytes_and_config(tmp_path, monkeypatch):
    path, _, source, image = prerequisite(tmp_path, monkeypatch)
    assert gate.validate_ss(tmp_path, source, image) == gate.ss._identity(path)


def test_identical_ss_script_in_different_immutable_snapshot_is_valid(tmp_path, monkeypatch):
    path, data, source, image = prerequisite(tmp_path, monkeypatch)
    data["script"]["path"] = "/srv/scenesmith/world-reward/snapshots/old-producer/infra/multiview_ss_gate.py"
    path.write_text(json.dumps(data))
    assert gate.validate_ss(tmp_path, source, image) == gate.ss._identity(path)


@pytest.mark.parametrize("key,value", [("status", "fail"), ("image_id", "sha256:" + "f" * 64),
                                       ("native_conditioner_verified", 1), ("native_dynamics_verified", False),
                                       ("ground_truth_used", 0), ("challenge_inputs_used", True),
                                       ("entropy_fusion_verified", True)])
def test_failed_or_different_actual_ss_gate_cannot_authorize_full(tmp_path, monkeypatch, key, value):
    path, data, source, image = prerequisite(tmp_path, monkeypatch)
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        gate.validate_ss(tmp_path, source, image)


@pytest.mark.parametrize("key", ["script", "ss_config", "ss_checkpoint", "preprocess_gate"])
def test_actual_ss_provenance_tamper_rejected(tmp_path, monkeypatch, key):
    path, data, source, image = prerequisite(tmp_path, monkeypatch)
    data[key]["sha256"] = "f" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        gate.validate_ss(tmp_path, source, image)


def observation_inputs():
    mask = np.zeros((gate.HEIGHT, gate.WIDTH), dtype=bool)
    mask[70:120, 100:150] = True
    depth = np.full(mask.shape, np.nan)
    depth[mask] = 1.4
    return depth, mask, np.eye(3), np.array([0., 0., 1.4])


def test_synthesis_arrays_registered_positive_finite_plus_half_rays():
    depth, mask, r, t = observation_inputs()
    saved = depth.copy()
    rgb, points = gate.observation_arrays(depth, mask, r, t)
    assert rgb.shape == (*mask.shape, 3) and rgb.dtype == np.uint8
    assert points.shape == (3, *mask.shape) and points.dtype == np.float32
    assert np.isfinite(points).all() and (points[2] > 0).all()
    np.testing.assert_allclose(points[2, mask], 1.4)
    np.testing.assert_allclose(points[2, ~mask], 4.)
    assert points[0, 96, 128] == pytest.approx(.5 / 240 * 1.4, abs=1e-8)
    np.testing.assert_array_equal(depth, saved)
    assert np.any(rgb[mask] != [28, 35, 42])


@pytest.mark.parametrize("kind", ["empty", "shape", "dtype", "nan_visible", "negative", "masked"])
def test_invalid_synthetic_observations_do_not_get_repaired(kind):
    depth, mask, r, t = observation_inputs()
    if kind == "empty": mask[:] = False
    elif kind == "shape": depth = depth[:, :-1]
    elif kind == "dtype": mask = mask.astype(np.uint8)
    elif kind == "nan_visible": depth[80, 110] = np.nan
    elif kind == "negative": depth[80, 110] = -1
    else: depth = np.ma.array(depth, mask=~mask)
    with pytest.raises(ValueError):
        gate.observation_arrays(depth, mask, r, t)


@pytest.mark.parametrize("args", [["--episode", "15"], ["--entropy"], ["--stage1_steps", "2"]])
def test_no_hidden_generation_or_challenge_overrides(args):
    with pytest.raises(SystemExit):
        gate.parse_args(args)
    assert vars(gate.parse_args([])) == {}


def test_wrapper_rejects_arguments_before_any_docker():
    result = subprocess.run(["bash", str(INFRA / "run_multiview_full_gate.sh"), "--unknown"], capture_output=True, text=True)
    assert result.returncode == 2 and "No full native gate arguments" in result.stderr


def test_source_declares_execution_not_quality_and_no_extra_data_mounts():
    source = Path(gate.__file__).read_text()
    wrapper = (INFRA / "run_multiview_full_gate.sh").read_text()
    assert 'decode_formats=["gaussian", "mesh"]' in source
    assert 'with_layout_postprocess=False' in source and 'ss_weighting=False, weighting_config=None' in source
    assert '"accuracy_evaluated": False' in source and '"synthetic_oracle_observations": True' in source
    assert "--network none" in wrapper and "readonly" in wrapper and "--mount" in wrapper
    assert "track_" not in wrapper and "datasets" not in wrapper and "outputs" not in wrapper

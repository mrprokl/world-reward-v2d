"""Tiny offline contract tests; no Torch, native imports, Docker or network."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from world_reward.multiview_representation import validate_camera_pointmap

SPEC = importlib.util.spec_from_file_location("mv_native_gate_test_module", Path(__file__).parents[1] / "infra/multiview_native_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def source_fixture(tmp_path):
    source = tmp_path / "vendor/mv-sam3d" / gate.PIN
    files = ["LICENSE", "sam3d_objects/pipeline/inference_pipeline.py",
             "sam3d_objects/pipeline/inference_pipeline_pointmap.py", "sam3d_objects/data/dataset/tdfy/preprocessor.py"]
    records = []
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tiny procedural source identity fixture\n")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {"stage": "pinned_mv_sam3d_source_acquisition", "status": "pass", "revision": gate.PIN,
              "repository": "https://github.com/devinli123/MV-SAM3D.git", "path": str(source),
              "source_git_clean_verified": True, "source_read_only": True,
              "weights_downloaded": False, "challenge_assets_downloaded": False, "files": records}
    manifest = tmp_path / "results/multiview-source.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(report))
    return source, manifest, report


def test_exact_source_inventory_and_manifest_identity(tmp_path):
    source, manifest, _ = source_fixture(tmp_path)
    result, identity = gate.validate_source(tmp_path)
    assert result == source
    assert identity["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("key,value", [("revision", "0" * 40), ("status", "fail"), ("source_git_clean_verified", 1),
                                       ("source_read_only", False), ("weights_downloaded", 0),
                                       ("challenge_assets_downloaded", True), ("repository", "https://example.org")])
def test_manifest_strict_provenance_and_boolean_types(tmp_path, key, value):
    _, manifest, report = source_fixture(tmp_path)
    report[key] = value
    manifest.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="manifest"):
        gate.validate_source(tmp_path)


@pytest.mark.parametrize("kind", ["hash", "bytes", "duplicate", "missing", "extra", "traversal", "absolute", "symlink"])
def test_tampered_or_ambiguous_source_inventory(tmp_path, kind):
    source, manifest, report = source_fixture(tmp_path)
    if kind == "hash":
        (source / "LICENSE").write_text("tampered")
    elif kind == "bytes":
        report["files"][0]["bytes"] += 1
    elif kind == "duplicate":
        report["files"].append(report["files"][0])
    elif kind == "missing":
        report["files"].pop()
    elif kind == "extra":
        (source / "sam3d_objects/untracked.py").write_text("extra")
    elif kind == "traversal":
        report["files"][0]["path"] = "../LICENSE"
    elif kind == "absolute":
        report["files"][0]["path"] = str(source / "LICENSE")
    else:
        path = source / "LICENSE"
        path.rename(source / "real-license")
        path.symlink_to(source / "real-license")
    manifest.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        gate.validate_source(tmp_path)


def test_fixture_own_rays_and_no_images_exported():
    rgba, points, k, mask = gate.procedural_fixture()
    assert rgba.shape == (48, 64, 4) and rgba.dtype == np.uint8
    assert points.shape == (3, 48, 64) and points.dtype == np.float32
    np.testing.assert_array_equal(rgba[..., 3] > 0, mask)
    camera = validate_camera_pointmap(points, k, np.ones_like(mask))
    assert camera.report["max_reprojection_error_px"] < 1e-4


def test_checkpoint_preprocessors_preserved_without_default_substitution():
    ss, slat = {"_target_": "native_ss"}, {"_target_": "native_slat"}
    loaded = []
    config = {"ss_preprocessor": None, "slat_preprocessor": slat, "ss_generator_config_path": "ss.yaml"}
    def load(name):
        loaded.append(name)
        return {"tdfy": {"val_preprocessor": ss}}
    assert gate._preprocessor_specs(config, load) == (ss, slat)
    assert loaded == ["ss.yaml"]
    config["ss_preprocessor"] = ss
    assert gate._preprocessor_specs(config, lambda _: pytest.fail("Do not reload explicit preprocessor")) == (ss, slat)


def test_missing_slat_preprocessor_is_actionable_failure():
    with pytest.raises(RuntimeError, match="explicit"):
        gate._preprocessor_specs({"ss_preprocessor": {}, "slat_preprocessor": None}, lambda _: None)


@pytest.mark.parametrize("argv", [["--episode", "15"], ["--root", "/tmp"], ["--helped"], ["unexpected"]])
def test_no_gate_arguments_accepted(argv):
    with pytest.raises(SystemExit):
        gate.parse_args(argv)


def test_default_argument_parser_is_empty_and_no_model_constructor_code():
    assert vars(gate.parse_args([])) == {}
    text = Path(SPEC.origin).read_text()
    assert "object.__new__(InferencePipelinePointMap)" in text
    assert '"native_constructor_verified": False' in text
    assert '"native_dynamics_verified": False' in text
    assert "torch.hub.load = original_hub" in text


@pytest.mark.parametrize("script", ["acquire_multiview_source.sh", "run_multiview_native_gate.sh"])
def test_shell_syntax_and_bad_args_rejected_before_side_effects(script):
    path = Path(__file__).parents[1] / "infra" / script
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    result = subprocess.run(["bash", str(path), "--unexpected"], capture_output=True)
    assert result.returncode == 2


def test_wrapper_pins_source_image_digest_and_network_none_no_challenge_data():
    text = (Path(__file__).parents[1] / "infra/run_multiview_native_gate.sh").read_text()
    assert gate.PIN in text and "--network none" in text and '"$IMAGE" python' in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/outputs" not in text
    assert "HF_HUB_OFFLINE=1" in text and "readonly" in text


def test_acquisition_is_pinned_public_source_only_and_no_overwrite():
    text = (Path(__file__).parents[1] / "infra/acquire_multiview_source.sh").read_text()
    assert gate.PIN in text and "--depth=1" in text and "staging.rename(destination)" in text
    assert "manifest.open('x')" in text and "~0o222" in text
    assert "huggingface" not in text and "snapshot_download" not in text and "pip install" not in text

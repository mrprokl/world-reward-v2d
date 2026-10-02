"""Tiny planning/provenance/fusion tests, not Torch/native inference execution."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

INFRA = Path(__file__).parents[1] / "infra"
native_spec = importlib.util.spec_from_file_location("multiview_native_gate", INFRA / "multiview_native_gate.py")
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)
spec = importlib.util.spec_from_file_location("ss_gate_test_module", INFRA / "multiview_ss_gate.py")
gate = importlib.util.module_from_spec(spec)
previous = sys.modules.get("multiview_native_gate")
sys.modules["multiview_native_gate"] = native
try:
    spec.loader.exec_module(gate)
finally:
    if previous is None:
        sys.modules.pop("multiview_native_gate")
    else:
        sys.modules["multiview_native_gate"] = previous


def prep_report(tmp_path):
    config = tmp_path / "weights/sam3d/hf-download/checkpoints/pipeline.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("tiny config")
    source = {"path": str(tmp_path / "results/multiview-source.json"), "sha256": "c" * 64}
    image = "sha256:" + "d" * 64
    report = {"stage": "native_mv_sam3d_import_config_preprocess", "status": "pass", "vendor_revision": native.PIN,
              "image_id": image, "procedural_inputs_only": True, "challenge_inputs_used": False,
              "ground_truth_used": False, "models_loaded": False, "source_manifest": source,
              "pipeline_config": native._identity(config), "script": native._identity(Path(native.__file__))}
    path = tmp_path / "results/multiview-native-preprocess-gate-v2.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    return path, report, source, image


def test_preprocessing_pass_exact_bindings(tmp_path):
    path, _, source, image = prep_report(tmp_path)
    assert gate.validate_preprocess(tmp_path, source, image) == gate._identity(path)


@pytest.mark.parametrize("key,value", [("status", "fail"), ("vendor_revision", "0" * 40), ("image_id", "sha256:" + "e" * 64),
                                       ("models_loaded", 0), ("ground_truth_used", 0), ("challenge_inputs_used", True),
                                       ("procedural_inputs_only", 1)])
def test_no_ss_when_preprocessing_missing_or_changed(tmp_path, key, value):
    path, report, source, image = prep_report(tmp_path)
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="passed"):
        gate.validate_preprocess(tmp_path, source, image)


@pytest.mark.parametrize("kind", ["source", "config", "script", "image_format"])
def test_preprocessing_provenance_tamper_rejected(tmp_path, kind):
    path, report, source, image = prep_report(tmp_path)
    if kind == "source":
        source = dict(source, sha256="f" * 64)
    elif kind == "config":
        Path(report["pipeline_config"]["path"]).write_text("changed")
    elif kind == "script":
        report["script"]["sha256"] = "f" * 64
        path.write_text(json.dumps(report))
    else:
        image = "tag:not_digest"
    with pytest.raises(RuntimeError):
        gate.validate_preprocess(tmp_path, source, image)


def test_original_preprocessing_report_cannot_authorize_v2_ss(tmp_path):
    path, _, source, image = prep_report(tmp_path)
    path.rename(path.with_name("multiview-native-preprocess-gate.json"))
    with pytest.raises(FileNotFoundError):
        gate.validate_preprocess(tmp_path, source, image)


def predictions():
    return [{"shape": np.array([[i, 2 * i]], dtype=float), "translation": np.array([[i, 0., 1.]])} for i in (1, 2, 3)]


def test_mean_shape_anchor_pose_not_mean_pose():
    data = predictions()
    result = gate.fusion_reference(data, {"translation"})
    np.testing.assert_array_equal(result["shape"], [[2., 4.]])
    np.testing.assert_array_equal(result["translation"], data[0]["translation"])
    assert not np.array_equal(result["translation"], np.stack([p["translation"] for p in data]).mean(0))


def test_duplicate_one_vs_three_and_nonanchor_permutation():
    data = predictions()
    single = gate.fusion_reference(data[:1], {"translation"})
    duplicate = gate.fusion_reference(data[:1] * 3, {"translation"})
    for key in single:
        np.testing.assert_array_equal(single[key], duplicate[key])
    original = gate.fusion_reference(data, {"translation"})
    permuted = gate.fusion_reference([data[0], data[2], data[1]], {"translation"})
    for key in original:
        np.testing.assert_array_equal(original[key], permuted[key])


def test_changing_anchor_does_not_preserve_pose():
    data = predictions()
    first = gate.fusion_reference(data, {"translation"})
    second = gate.fusion_reference(data[::-1], {"translation"})
    np.testing.assert_array_equal(first["shape"], second["shape"])
    assert not np.array_equal(first["translation"], second["translation"])


def test_reference_does_not_mutate_inputs_or_launder_anchor():
    data = predictions()
    copies = [{k: v.copy() for k, v in p.items()} for p in data]
    result = gate.fusion_reference(data, {"translation"})
    result["translation"][:] = 999
    for p, saved in zip(data, copies):
        for key in p:
            np.testing.assert_array_equal(p[key], saved[key])


@pytest.mark.parametrize("kind", ["empty", "keys", "shape", "nan"])
def test_fusion_contract_bad_inputs(kind):
    data = predictions()
    if kind == "empty":
        data = []
    elif kind == "keys":
        data[1].pop("shape")
    elif kind == "shape":
        data[1]["shape"] = np.zeros((1, 3))
    else:
        data[1]["shape"][0, 0] = np.nan
    with pytest.raises(ValueError):
        gate.fusion_reference(data, {"translation"})


def test_partial_native_loading_checkpoint_and_rng_contract_source():
    text = (INFRA / "multiview_ss_gate.py").read_text()
    assert "pipe.init_ss_generator(" in text and "pipe.init_ss_condition_embedder(" in text
    assert "pipe.init_slat" not in text and "pipe.sample_sparse_structure(" not in text
    assert "original(state, t, d, condition)" in text
    assert "generator.reverse_fn.backbone.latent_mapping.items()" in text
    assert "random.seed(42)" in text and "np.random.seed(42)" in text and "torch.cuda.manual_seed_all(42)" in text
    assert "torch.hub.load = original_hub" in text and "random.setstate" in text
    assert "signal.alarm(300)" in text and "min(120" in text
    assert gate.SS_SHA256 == "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146"
    assert gate.SS_BYTES == 6690136964


def test_no_shape_or_entropy_adoption_claim():
    text = (INFRA / "multiview_ss_gate.py").read_text()
    for field in ("shape_generation_implemented", "slat_loaded", "entropy_fusion_verified", "adoption_authorized", "submission_eligible"):
        assert f'"{field}": False' in text


def test_wrapper_syntax_bad_arguments_no_docker_network_none_digest():
    path = INFRA / "run_multiview_ss_gate.sh"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    assert subprocess.run(["bash", str(path), "--bad"], capture_output=True).returncode == 2
    text = path.read_text()
    assert "--network none" in text and '"$IMAGE" python' in text and native.PIN in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/outputs" not in text
    assert "HF_HUB_OFFLINE=1" in text and "readonly" in text

"""Offline command planning only: no source/assets/models, GPU or subprocess."""

import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "infra/cari_runner.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    return dict(
        native_root="/srv/vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d",
        export_seq="/srv/outputs/own_export/episode_video",
        depth_h5="/srv/outputs/depth_full/aligned_depth.h5",
        mhr_init="/srv/outputs/body_full/cari_adapter/canonical_initializer.pkl",
        object_poses="/srv/outputs/object_pose_full/own_icp_poses.pkl",
        checkpoint="/srv/weights/cari4d/cari4d/2026-08-25-09-35-57/step200000.pth",
        output="/srv/outputs/cari_forward_full/predictions.pth",
        total_frames=501,
        mhr_assets_root="/srv/weights/cari4d/sam3d_body",
        torch_home="/srv/weights/cari4d/sam3d_body/torch_home",
        sam3d_source_root="/workspace/v2d_sam3d_body/lib",
    )


def test_exact_native_forward_only_command_and_offline_plan(runner, inputs):
    before = inputs.copy()
    plan = runner.build_cari_forward_command(**inputs)
    native = inputs["native_root"]
    assert plan["argv"] == [
        "python", native + "/tools/run_mhr_wild_inference.py", inputs["export_seq"],
        "--depth-h5", inputs["depth_h5"], "--mhr-init", inputs["mhr_init"],
        "--foundationpose-file", inputs["object_poses"],
        "--config", native + "/learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml",
        "--checkpoint", inputs["checkpoint"], "--output", inputs["output"],
        "--stride", "96", "--render-batch-size", "32", "--crop-workers", "8",
        "--crop-buffer-count", "2", "--device", "cuda", "--offline-supervision-contract", "--no-input-cache",
    ]
    assert inputs == before
    assert plan["cwd"] == native
    assert plan["environment"]["MHR_ASSETS_ROOT"] == inputs["mhr_assets_root"]
    assert plan["environment"]["TORCH_HOME"] == inputs["torch_home"]
    assert plan["environment"]["PYTHONPATH"] == native + ":" + inputs["sam3d_source_root"]
    assert plan["environment"]["HF_HUB_OFFLINE"] == "1"
    assert "TOKEN" not in " ".join(plan["environment"])
    metadata = plan["metadata"]
    assert metadata["foundationpose_stage_invoked"] is False
    assert metadata["contact_postoptimization_invoked"] is False
    assert metadata["ground_truth_or_oracle_modes_requested"] is False
    assert metadata["challenge_performance_verified"] is False
    assert metadata["submission_eligible"] is False
    assert metadata["original_frame_count"] == 501
    assert "not_performed" in metadata["input_verification"]
    assert "not_FoundationPose" in metadata["object_pose_method"]
    json.dumps(plan)
    for forbidden in ("--overwrite", "--wandb-run-path", "--no-gt", "--no-refine", "--first-usable-frame-gt-rotation-oracle"):
        assert forbidden not in plan["argv"]


def test_exact_96_frame_window_allowed_without_padding(runner, inputs):
    inputs["total_frames"] = 96
    assert runner.build_cari_forward_command(**inputs)["metadata"]["original_frame_count"] == 96


@pytest.mark.parametrize("count", [0, -1, 95, True, False, 501., "501", None, np.bool_(True)])
def test_minimum_window_and_integer_frame_count_strict(runner, inputs, count):
    inputs["total_frames"] = count
    with pytest.raises(ValueError, match="integer total_frames >= 96"):
        runner.build_cari_forward_command(**inputs)


def test_numpy_integer_supported_and_JSON_serializable(runner, inputs):
    inputs["total_frames"] = np.int64(501)
    plan = runner.build_cari_forward_command(**inputs)
    assert type(plan["metadata"]["original_frame_count"]) is int
    json.dumps(plan)


def test_explicit_persistent_cache_mutually_exclusive_with_disable(runner, inputs):
    plan = runner.build_cari_forward_command(**inputs, input_cache="/srv/cache/coconet-input.h5")
    assert plan["argv"][-2:] == ["--input-cache", "/srv/cache/coconet-input.h5"]
    assert "--no-input-cache" not in plan["argv"]
    assert plan["metadata"]["input_cache"] == "/srv/cache/coconet-input.h5"


@pytest.mark.parametrize("key", ["native_root", "export_seq", "depth_h5", "mhr_init", "object_poses", "checkpoint", "output", "mhr_assets_root", "torch_home", "sam3d_source_root"])
@pytest.mark.parametrize("bad", ["relative/path", "/srv/../secret", "/srv/a\nb", "/srv/a\x00b", "/srv/a\x7fb", "/", "//srv/path", "C:\\some\\path"])
def test_remote_paths_absolute_without_traversal_or_control(runner, inputs, key, bad):
    inputs[key] = bad
    with pytest.raises(ValueError):
        runner.build_cari_forward_command(**inputs)


@pytest.mark.parametrize("key", ["native_root", "mhr_init", "torch_home"])
def test_path_non_string_rejected(runner, inputs, key):
    inputs[key] = 42
    with pytest.raises(TypeError, match="absolute Linux path"):
        runner.build_cari_forward_command(**inputs)


@pytest.mark.parametrize("key, bad", [("depth_h5", "/srv/depth.npz"), ("mhr_init", "/srv/body.npz"), ("object_poses", "/srv/object.npy"), ("checkpoint", "/srv/weights.pkl"), ("output", "/srv/predictions.pkl")])
def test_exact_native_file_ABI_extensions(runner, inputs, key, bad):
    inputs[key] = bad
    with pytest.raises(ValueError, match="must end"):
        runner.build_cari_forward_command(**inputs)


@pytest.mark.parametrize("bad", ["/srv/step200000.pth", "/srv/2026-08-25-09-35-57/step100000.pth", "/srv/2026-08-26-09-35-57/step200000.pth"])
def test_wrong_commercial_release_not_silently_selected(runner, inputs, bad):
    inputs["checkpoint"] = bad
    with pytest.raises(ValueError, match="audited commercial"):
        runner.build_cari_forward_command(**inputs)


def test_absolute_python_executable_and_PurePosixPath_supported(runner, inputs):
    inputs["native_root"] = PurePosixPath(inputs["native_root"])
    plan = runner.build_cari_forward_command(**inputs, python_executable="/opt/python/bin/python")
    assert plan["argv"][0] == "/opt/python/bin/python"
    assert all(isinstance(value, str) for value in plan["argv"])


@pytest.mark.parametrize("bad", ["python --flag", "bash", "./python", "python;evil", 1])
def test_python_is_single_executable_not_shell_expression(runner, inputs, bad):
    with pytest.raises((ValueError, TypeError)):
        runner.build_cari_forward_command(**inputs, python_executable=bad)


def test_paths_with_spaces_and_shell_metacharacters_are_single_argv_values(runner, inputs):
    inputs["object_poses"] = "/srv/own result;$(not-executed) poses.pkl"
    plan = runner.build_cari_forward_command(**inputs)
    index = plan["argv"].index("--foundationpose-file")
    assert plan["argv"][index + 1] == inputs["object_poses"]
    assert shlex.split(shlex.join(plan["argv"])) == plan["argv"]


def test_body_and_object_input_alias_rejected(runner, inputs):
    inputs["object_poses"] = inputs["mhr_init"]
    with pytest.raises(ValueError, match="cannot alias"):
        runner.build_cari_forward_command(**inputs)


def test_output_cannot_replace_checkpoint(runner, inputs):
    inputs["output"] = inputs["checkpoint"]
    with pytest.raises(ValueError, match="replace an input"):
        runner.build_cari_forward_command(**inputs)


@pytest.mark.parametrize("key", ["depth_h5", "output", "mhr_init"])
def test_cache_cannot_replace_other_files(runner, inputs, key):
    cache = inputs[key]
    with pytest.raises(ValueError, match="input_cache"):
        runner.build_cari_forward_command(**inputs, input_cache=cache)


def test_no_filesystem_or_subprocess_needed_for_remote_nonexistent_paths(runner, inputs, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Builder must not read or execute anything")
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    assert runner.build_cari_forward_command(**inputs)["argv"][0] == "python"


@pytest.mark.parametrize("key", ["native_root", "sam3d_source_root"])
def test_source_path_cannot_inject_extra_PYTHONPATH_entry(runner, inputs, key):
    inputs[key] = "/srv/source:/srv/untrusted"
    with pytest.raises(ValueError, match="PYTHONPATH separator"):
        runner.build_cari_forward_command(**inputs)

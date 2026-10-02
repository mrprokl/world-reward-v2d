"""Tiny contracts only; no models, RGB, torch, Docker, GPU or Azure execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def refine(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("test_native_cari_refine", ROOT / "infra/cari_refine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(refine):
    return {"upstream_revision": refine.UPSTREAM_REVISION,
            "assets": [{"filename": name, **record} for name, record in refine.REFINEMENT_ASSETS.items()]}


def test_release_receipt_exact_pins_no_mutation(refine):
    record = receipt(refine); before = copy.deepcopy(record)
    refine.require_asset_receipt(record)
    assert record == before
    assert sum(v["bytes"] for v in refine.REFINEMENT_ASSETS.values()) == 173282


@pytest.mark.parametrize("failure", ["revision", "bytes", "bool_bytes", "sha", "pointer_sha", "missing", "duplicate"])
def test_asset_receipt_fails_closed(refine, failure):
    record = receipt(refine)
    if failure == "revision": record["upstream_revision"] = "main"
    elif failure == "bytes": record["assets"][0]["bytes"] += 1
    elif failure == "bool_bytes": record["assets"][0]["bytes"] = True
    elif failure == "sha": record["assets"][0]["sha256"] = "0" * 64
    elif failure == "pointer_sha": record["assets"][0]["sha256"] = "da3c122d47cd57d16c993e1ffb7fdfe0cd33d42303a13a4ff3f5f3629d5f3613"
    elif failure == "missing": record["assets"].pop()
    else: record["assets"][1] = record["assets"][0]
    with pytest.raises(ValueError): refine.require_asset_receipt(record)


def bundles(refine):
    from cari_converter import PARAMETER_DIMS
    count = 3
    params = {k: np.zeros((count, size), np.float32) for k, size in PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    params["mhr_shape"][:, 0] = [1, 2, 3]
    params["pose_abs"] = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    params["pose_abs"][:, 2, 3] = 2
    metadata = {"ground_truth_used": False, "object_mesh": "/own/output_aligned.glb",
                "object_pose_frame": "centered_axis_aligned",
                "object_pose_frame_revision": "cari4d.object_pose_frame.centered_axis_aligned.v1",
                "object_pose_storage_frame": "output_aligned_mesh_frame",
                "object_pose_storage_to_training_transform": np.eye(4).tolist(),
                "object_mesh_to_training_transform": np.eye(4).tolist()}
    source = {"gt": {}, "metadata": metadata, "frames": [f"{i:06d}" for i in range(count)], "pr": params}
    result = copy.deepcopy(source)
    result["pr"]["mhr_body_pose_cont"][:, 3] = .05
    result["pr"]["pose_abs"][:, 0, 3] = .01
    result["postopt"] = {"mode": "smplh_parity", "frame_indices": list(range(count)), "resolved_batch_size": count,
                         "batch_sampling": "full_clip_v1", "optimized_parameters": refine.OPTIMIZED_PARAMETERS.copy(),
                         "fixed_parameters": refine.FIXED_PARAMETERS.copy(),
                         "config": {"num_steps": 300, "batch_size": 0, "frame_start": 0, "frame_limit": 0,
                                    "freeze_object_rotation": True, "freeze_body_internal_translations": True},
                         "history": [{"iter": 0., "loss_total": 1.}, {"iter": 300., "loss_total": .9}],
                         "final_diagnostics": {"human_pose_prior_raw": .001}}
    return source, result


def test_native_refinement_preserves_all_fixed_blocks_and_inputs(refine):
    source, result = bundles(refine); before = copy.deepcopy(source)
    diagnostic = refine.validate_refined_bundle(source, result, 3)
    assert diagnostic["effective_optimizer_updates"] == 301
    assert diagnostic["batch_size"] == 0
    assert diagnostic["frozen_parameters_bit_identical"] is True
    for key in source["pr"]:
        assert np.array_equal(source["pr"][key], before["pr"][key])
    assert np.array_equal(source["pr"]["mhr_shape"], result["pr"]["mhr_shape"])


@pytest.mark.parametrize("block", ["mhr_global_rot6d", "mhr_trans", "mhr_hand", "mhr_shape", "mhr_scale", "mhr_face"])
def test_frozen_parameter_mutation_rejected(refine, block):
    source, result = bundles(refine)
    result["pr"][block][0, 0] += .001
    with pytest.raises(ValueError): refine.validate_refined_bundle(source, result, 3)


@pytest.mark.parametrize("failure", ["body_translation", "object_rotation", "mesh", "gt", "frames", "mode", "steps",
                                     "partial", "minibatch", "toggle", "nan_history", "nan_diagnostics", "dtype"])
def test_no_partial_or_altered_native_protocol(refine, failure):
    source, result = bundles(refine)
    if failure == "body_translation": result["pr"]["mhr_body_pose_cont"][0, 254] = .001
    elif failure == "object_rotation": result["pr"]["pose_abs"][0, :2, :2] = [[0, -1], [1, 0]]
    elif failure == "mesh": result["metadata"]["object_mesh"] = "/other.glb"
    elif failure == "gt": result["gt"] = {"hidden": 1}
    elif failure == "frames": result["frames"][-1] = "000004"
    elif failure == "mode": result["postopt"]["mode"] = "legacy_staged"
    elif failure == "steps": result["postopt"]["config"]["num_steps"] = 200
    elif failure == "partial": result["postopt"]["history"][-1]["iter"] = 100
    elif failure == "minibatch": result["postopt"]["config"]["batch_size"] = 96
    elif failure == "toggle": result["postopt"]["config"]["freeze_object_rotation"] = False
    elif failure == "nan_history": result["postopt"]["history"][-1]["loss_total"] = float("nan")
    elif failure == "nan_diagnostics": result["postopt"]["final_diagnostics"]["human_pose_prior_raw"] = float("nan")
    else: result["pr"]["mhr_trans"] = result["pr"]["mhr_trans"].astype(np.float64)
    with pytest.raises(ValueError): refine.validate_refined_bundle(source, result, 3)


def passing_report(refine):
    source, result = bundles(refine)
    return {"stage": refine.STAGE, "status": "pass", "input_track": "track_1", "ground_truth_used": False,
            "ground_truth_read": False, "oracle_modes": [], "hand_labeled_test": False,
            "network": "none", "phase": "complete", "producer_revision": "a" * 40, "image_id": "sha256:" + "b" * 64,
            "frames": 3, "episode_index": 15, "checkpoint_sha256": refine.CHECKPOINT_SHA256,
            "native_refinement_verified": True, "optimizer_sha256": refine.OPTIMIZER_SHA256,
            "metadata": refine.validate_refined_bundle(source, result, 3), "refinement_assets": refine.REFINEMENT_ASSETS,
            "inference_source_identity": {"sha256": "a" * 64}, "body_assets": {"model.ckpt": {"sha256": "b" * 64}},
            **{key: "c" * 64 for key in ("forward_report_sha256", "inputs_report_sha256", "source_bundle_sha256", "bundle_sha256")}}


def test_consumer_report_contract(refine):
    report = passing_report(refine)
    refine.require_full_refinement_report(report)
    json.dumps(report)


@pytest.mark.parametrize("field,value", [("status", "fail"), ("ground_truth_read", True), ("network", None), ("phase", "running"),
                                          ("producer_revision", "main"), ("image_id", "tag:latest"), ("ground_truth_used", 0),
                                          ("hand_labeled_test", 0), ("native_refinement_verified", False),
                                          ("frames", True), ("episode_index", 30), ("bundle_sha256", None),
                                          ("source_bundle_sha256", "x" * 64), ("optimizer_sha256", "a" * 64),
                                          ("refinement_assets", {}), ("body_assets", {}), ("inference_source_identity", {})])
def test_consumer_rejects_failed_or_unbound_report(refine, field, value):
    report = passing_report(refine); report[field] = value
    with pytest.raises(ValueError): refine.require_full_refinement_report(report)


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_episode_routing_before_heavy_imports(refine, episode):
    assert refine._argument_parser().parse_args(["--episode", str(episode)]).episode == episode


@pytest.mark.parametrize("episode", [-1, 30, "True", "1.5"])
def test_bad_episode_fails_before_runtime(refine, episode):
    with pytest.raises(SystemExit): refine._argument_parser().parse_args(["--episode", str(episode)])


def test_regular_artifact_hash_and_symlink_gate(refine, tmp_path):
    file = tmp_path / "source"; file.write_text("own")
    digest = refine.sha256(file)
    assert refine._file(file, tmp_path, digest) == file
    with pytest.raises(ValueError): refine._file(file, tmp_path, "0" * 64)
    link = tmp_path / "link"; link.symlink_to(file)
    with pytest.raises(ValueError): refine._file(link, tmp_path)


def fake_shell(tmp_path):
    binary = tmp_path / "bin"; binary.mkdir()
    log = tmp_path / "commands"
    scripts = {
        "id": "printf '123\\n'",
        "chown": 'printf "chown %s\\n" "$*" >> "$LOG"',
        "timeout": 'printf "timeout %s\\n" "$*" >> "$LOG"; shift 3; exec "$@"',
        "docker": "if [[ \"$1\" == image ]]; then printf 'sha256:%064d\\n' 0; else printf '%s\\0' \"$@\" >> \"$ARGS\"; fi",
    }
    for name, text in scripts.items():
        path = binary / name; path.write_text("#!/usr/bin/env bash\nset -eu\n" + text + "\n"); path.chmod(0o755)
    root = tmp_path / "runtime"; base = root / "outputs/episode_000015"
    (base / "cari_forward").mkdir(parents=True)
    (base / "cari_forward/report.json").write_text("{}")
    return dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"], WR_ROOT=str(root), WR_CODE=str(ROOT),
                WR_CODE_REVISION="a" * 40, LOG=str(log), ARGS=str(tmp_path / "args")), base


def test_wrapper_only_new_output_writable_and_scoped_chown(tmp_path):
    environment, base = fake_shell(tmp_path)
    completed = subprocess.run(["bash", str(ROOT / "infra/run_cari_refine.sh"), "--episode", "15", "--no-wait"],
                               env=environment, text=True, capture_output=True, timeout=5)
    assert completed.returncode == 0, completed.stderr
    args = Path(environment["ARGS"]).read_bytes().decode().strip("\0").split("\0")
    mounts = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--mount"]
    writable = [mount for mount in mounts if not mount.endswith(",readonly")]
    assert writable == [f"type=bind,src={base}/cari_refined,dst={base}/cari_refined"]
    assert all("eval_private" not in mount and "/data" not in mount for mount in mounts)
    assert args[-2:] == ["--episode", "15"]
    assert args[args.index("--network") + 1] == "none"
    assert "HOME=/tmp" in args and "XDG_CACHE_HOME=/tmp/world-reward-cache" in args
    log = Path(environment["LOG"]).read_text()
    assert f"chown 123:123 {base}/cari_refined" in log
    assert "7203s" in log and "64g" in args


def test_wrapper_never_overwrites_output(tmp_path):
    environment, base = fake_shell(tmp_path)
    output = base / "cari_refined"; output.mkdir(); (output / "report.json").write_text("frozen")
    completed = subprocess.run(["bash", str(ROOT / "infra/run_cari_refine.sh"), "--no-wait"],
                               env=environment, capture_output=True, timeout=5)
    assert completed.returncode != 0
    assert (output / "report.json").read_text() == "frozen"
    assert not Path(environment["ARGS"]).exists()


def mesh_arrays():
    v = np.array([[0, 0, 0], [.3, 0, 0], [0, .2, 0], [0, 0, .1], [0, 0, 0]], dtype=np.float32)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 0, 0]], dtype=np.int64)
    A = np.eye(4); A[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; A[:3, 3] = [.04, .05, .01]
    return v, f, A


def test_aligned_geometry_oriented_permutations_padding_and_float32(refine):
    v, f, A = mesh_arrays()
    transformed = (v @ A[:3, :3].T + A[:3, 3]).astype(np.float32)
    rotated_faces = np.roll(f[:4][::-1], 1, axis=1)
    before = v.copy()
    report = refine.verify_aligned_geometry(v, f, transformed, rotated_faces, A)
    assert report["oriented_triangles_verified"] == 4
    assert report["max_vertex_roundtrip_error_m"] < 1e-7
    assert report["historical_aligned_mesh_byte_identity_proven"] is False
    assert np.array_equal(v, before)


@pytest.mark.parametrize("failure", ["scale", "flip", "drop", "duplicate", "bad_A"])
def test_aligned_geometry_does_not_repair_or_drop_triangles(refine, failure):
    v, f, A = mesh_arrays()
    transformed = v @ A[:3, :3].T + A[:3, 3]; g = f[:4].copy()
    if failure == "scale": transformed *= 1.01
    elif failure == "flip": g[0] = g[0, ::-1]
    elif failure == "drop": g = g[:-1]
    elif failure == "duplicate": g[1] = g[0]
    else: A[0, 0] = .1
    with pytest.raises(ValueError): refine.verify_aligned_geometry(v, f, transformed, g, A)

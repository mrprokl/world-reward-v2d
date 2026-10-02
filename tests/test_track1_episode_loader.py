"""Tiny frozen producer/NPZ fixtures only; no models, GT, media, Torch or Azure."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def loader(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_episode_loader", infra / "track1_episode_loader.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen(loader, tmp_path):
    episode_index, count, video_hash = 15, 3, "a" * 64
    base = tmp_path / f"outputs/episode_{episode_index:06d}"
    paths = {"conversion": base / "cari_conversion/report.json", "inputs": base / "cari_inputs/report.json",
             "forward": base / "cari_forward/report.json", "body": base / "body_full/report.json",
             "adapter": base / "body_full/cari_adapter/report.json", "object": base / "object_pose_full/report.json"}
    common = {"status": "pass", "episode_index": episode_index, "input_track": "track_1",
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "input_sha256": video_hash}
    source_identity = {"python_files": 100, "sha256": "b" * 64}
    asset_identity = {"model.ckpt": {"bytes": 20, "sha256": "c" * 64}}
    decoder_identity = {"mhr_model_sha256": "d" * 64, "mhr_buffer_sha256": "c" * 64,
                        "mhr_decoder_revision": "cari4d.mhr_layer.vertices.v1"}
    reports = {
        "body": dict(common, stage="sam3d_body_full_video_initializer", total_video_frames=count,
                     frame_indices=list(range(count)), mhr_geometry_forward_verified=True,
                     inference_source_identity=source_identity, body_assets=asset_identity),
        "adapter": dict(common, stage="native_cari_body_adapter_full_video", frames=count,
                        inference_source_identity=source_identity, decoder_identity=decoder_identity),
        "object": dict(common, stage="fixed_scale_full_object_pose_initializer", original_frame_coverage_verified=True,
                       fixed_shape=True),
        "inputs": dict(common, stage="world_reward_native_cari_inputs", frames=count, original_frame_coverage_verified=True),
        "forward": dict(common, stage="world_reward_native_cari_full_forward", episode_inputs_used=True,
                        metadata={"actual_network_forward_verified": True, "full_original_frame_coverage_verified": True},
                        checkpoint_sha256=loader.CHECKPOINT_SHA256, bundle_sha256="e" * 64,
                        inference_source_identity=source_identity),
        "conversion": dict(common, stage="world_reward_native_cari_official_conversion", frames=count,
                           original_frame_coverage_verified=True, human_shared_identity_verified=True,
                           facial_expressions_zero=True, human_parameter_format="mhr_model_params_136_68_45",
                           geometry_frame="camera_x_right_y_down_z_forward", geometry_units="metres", network="none",
                           upstream_revision=loader.UPSTREAM_REVISION, checkpoint_sha256=loader.CHECKPOINT_SHA256,
                           reference_model_sha256=loader.REFERENCE_MODEL_SHA256, official_converter_sha256=loader.CONVERTER_SHA256,
                           body_assets=asset_identity, decoder_identity=decoder_identity, inference_source_identity=source_identity,
                           bundle_sha256="e" * 64),
    }
    # Actual adapter/forward reports omit video SHA; their hash chains bind it.
    del reports["adapter"]["input_sha256"], reports["forward"]["input_sha256"]
    arrays = {"pose": np.arange(count * 136, dtype=np.float32).reshape(count, 136) / 100,
              "scales": np.zeros(68), "shape": np.zeros(45), "expression": np.zeros(72), "frame_index": np.arange(count),
              "object_rotation": np.tile(np.eye(3), (count, 1, 1)), "object_translation": np.tile([0., 0., 3.], (count, 1)),
              "object_scale": np.asarray(1.),
              "object_vertices": np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 0.]], dtype=np.float32),
              "object_faces": np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int32)}
    def write_report(name):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_text(json.dumps(reports[name]))
        return loader.sha256(paths[name])
    def sync_reports():
        body_hash, object_hash = write_report("body"), write_report("object")
        reports["adapter"]["body_report_sha256"] = body_hash
        adapter_hash = write_report("adapter")
        reports["inputs"]["input_report_sha256"] = {"body": body_hash, "adapter": adapter_hash, "object": object_hash}
        reports["forward"]["inputs_report_sha256"] = write_report("inputs")
        write_report("forward")
        reports["conversion"]["input_report_sha256"] = {name: loader.sha256(paths[name]) for name in paths if name != "conversion"}
        write_report("conversion")
    def write_archive():
        path = base / "cari_conversion/episode.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        params = base / "cari_conversion/params.npz"
        # Its linked bytes are integrity-verified, not loaded as predictions.
        params.write_bytes(b"synthetic params checksum only")
        reports["conversion"]["episode_sha256"] = loader.sha256(path)
        reports["conversion"]["params_sha256"] = loader.sha256(params)
        write_report("conversion")
    write_archive()
    sync_reports()
    return tmp_path, base, reports, paths, arrays, sync_reports, write_archive, video_hash


def load(loader, fixture):
    root, _, _, _, _, _, _, video_hash = fixture
    return loader.load_track1_episode(root, 15, 3, video_hash)


def test_roundtrip_native_episode_preserves_arrays_padding_and_manifest(loader, frozen):
    result = load(loader, frozen)
    result.episode.validate()
    original = frozen[4]
    assert result.episode.total_video_frames == 3
    for name in ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale"):
        value = getattr(result.episode.reconstruction, name)
        assert value.dtype == original[name].dtype and value.tobytes() == original[name].tobytes()
    assert result.episode.object_vertices.tobytes() == original["object_vertices"].tobytes()
    assert result.episode.object_faces.dtype == np.int32
    assert result.episode.object_faces.tobytes() == original["object_faces"].tobytes()
    assert result.manifest["integrity_and_schema_verified"] is True
    assert result.manifest["numerical_truth_independently_reverified"] is False
    assert result.manifest["submission_eligibility_verified"] is False
    assert result.manifest["challenge_performance_verified"] is False
    json.dumps(result.manifest)


@pytest.mark.parametrize("key,bad", [
    ("stage", "native_cari_checkpoint_load_gate"), ("status", "fail"), ("episode_index", True),
    ("episode_index", 0), ("frames", True), ("frames", 3.), ("frames", 4),
    ("ground_truth_used", 0), ("hand_labeled_test", True), ("oracle_modes", ["gt_pose"]), ("input_track", "track_2"),
    ("input_sha256", "0" * 64), ("original_frame_coverage_verified", 1), ("human_shared_identity_verified", False),
    ("facial_expressions_zero", False), ("geometry_units", "millimetres"), ("geometry_frame", "world"),
    ("human_parameter_format", "raw_sam_266"), ("upstream_revision", "0" * 40),
    ("checkpoint_sha256", "0" * 64), ("official_converter_sha256", "0" * 64), ("reference_model_sha256", "0" * 64),
])
def test_final_report_contract_fails_before_arrays(loader, frozen, key, bad):
    _, _, reports, _, _, sync, _, _ = frozen
    reports["conversion"][key] = bad
    sync()
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("name", ["forward", "inputs", "body", "adapter", "object"])
@pytest.mark.parametrize("key,bad", [("status", "fail"), ("ground_truth_used", True),
                                    ("hand_labeled_test", None), ("episode_index", 29)])
def test_each_frozen_dependency_status_oracle_and_episode_bound(loader, frozen, name, key, bad):
    frozen[2][name][key] = bad
    frozen[5]()
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("name,key,value", [
    ("forward", "metadata", {"actual_network_forward_verified": False, "full_original_frame_coverage_verified": True}),
    ("forward", "episode_inputs_used", False), ("forward", "checkpoint_sha256", "0" * 64),
    ("body", "frame_indices", [0, 0, 2]), ("body", "frame_indices", [False, 1, 2]),
    ("body", "total_video_frames", 4), ("body", "mhr_geometry_forward_verified", False),
    ("adapter", "frames", True), ("inputs", "frames", 4), ("inputs", "original_frame_coverage_verified", False),
    ("object", "original_frame_coverage_verified", False), ("object", "fixed_shape", False),
    ("body", "input_sha256", "0" * 64), ("inputs", "input_sha256", "0" * 64), ("object", "input_sha256", "0" * 64),
    ("body", "body_assets", {}), ("adapter", "decoder_identity", {}), ("forward", "inference_source_identity", {}),
])
def test_linked_report_semantics_not_just_any_hash(loader, frozen, name, key, value):
    frozen[2][name][key] = value
    frozen[5]()
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("name", ["conversion", "forward", "inputs", "body", "adapter", "object", "episode", "params"])
def test_changed_artifact_hash_is_never_trusted(loader, frozen, name):
    _, base, _, paths, _, _, _, _ = frozen
    path = paths[name] if name in paths else base / f"cari_conversion/{name}.npz"
    if name == "conversion":
        report = json.loads(path.read_text())
        report["episode_sha256"] = "0" * 64
        path.write_text(json.dumps(report))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("name", ["forward", "inputs", "body", "adapter", "object", "episode", "params"])
def test_missing_frozen_dependencies_fail_closed(loader, frozen, name):
    _, base, _, paths, _, _, _, _ = frozen
    path = paths[name] if name in paths else base / f"cari_conversion/{name}.npz"
    path.unlink()
    with pytest.raises(ValueError, match="missing"): load(loader, frozen)


@pytest.mark.parametrize("key", ["pose", "scales", "shape", "expression", "object_rotation", "object_translation", "object_scale", "object_vertices"])
@pytest.mark.parametrize("kind", ["nan", "integer", "float16"])
def test_all_float_arrays_strict_types_and_finite(loader, frozen, key, kind):
    arrays = frozen[4]
    if kind == "nan": arrays[key].flat[0] = np.nan
    else: arrays[key] = arrays[key].astype(np.int64 if kind == "integer" else np.float16)
    frozen[6]()
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("failure", ["extra_key", "missing_key", "reordered", "float_frames", "bool_frames", "short_frames",
                                    "perframe_scales", "perframe_shape", "expression_nonzero", "scale_twice", "reflection",
                                    "float_faces", "uint_faces", "invalid_faces", "overbudget", "zero_mesh"])
def test_archive_no_fallback_frame_identity_mesh_or_scale_repairs(loader, frozen, failure):
    arrays = frozen[4]
    if failure == "extra_key": arrays["unverified_prediction"] = np.zeros(1)
    elif failure == "missing_key": del arrays["expression"]
    elif failure == "reordered": arrays["frame_index"] = np.array([2, 1, 0])
    elif failure == "float_frames": arrays["frame_index"] = np.arange(3, dtype=float)
    elif failure == "bool_frames": arrays["frame_index"] = np.array([False, True, True])
    elif failure == "short_frames": arrays["frame_index"] = np.arange(2)
    elif failure == "perframe_scales": arrays["scales"] = np.zeros((3, 68))
    elif failure == "perframe_shape": arrays["shape"] = np.zeros((3, 45))
    elif failure == "expression_nonzero": arrays["expression"][0] = 1
    elif failure == "scale_twice": arrays["object_scale"] = np.asarray(2.)
    elif failure == "reflection": arrays["object_rotation"][0, 0, 0] = -1
    elif failure == "float_faces": arrays["object_faces"] = arrays["object_faces"].astype(float)
    elif failure == "uint_faces": arrays["object_faces"] = arrays["object_faces"].astype(np.uint64)
    elif failure == "invalid_faces": arrays["object_faces"][0, 0] = 4
    elif failure == "overbudget": arrays["object_faces"] = np.tile(np.array([[0, 1, 2]]), (4097, 1))
    elif failure == "zero_mesh": arrays["object_vertices"][:] = 0
    frozen[6]()
    with pytest.raises(ValueError): load(loader, frozen)


@pytest.mark.parametrize("target", ["report", "episode", "directory"])
def test_external_symlink_or_symlink_directory_rejected(loader, frozen, tmp_path, target):
    _, base, _, paths, _, _, _, _ = frozen
    if target == "directory":
        original = base / "cari_conversion"
        external = tmp_path / "moved"
        original.rename(external)
        original.symlink_to(external, target_is_directory=True)
    else:
        path = paths["conversion"] if target == "report" else base / "cari_conversion/episode.npz"
        external = tmp_path / "external"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"): load(loader, frozen)


@pytest.mark.parametrize("episode", [True, -1, 30, 15., "15"])
def test_invalid_episode_rejected_before_reads(loader, frozen, episode):
    with pytest.raises(ValueError, match="episode integer"):
        loader.load_track1_episode(frozen[0], episode, 3, frozen[7])


@pytest.mark.parametrize("frames", [True, 0, -1, 3., "3"])
def test_invalid_frame_count_rejected_before_reads(loader, frozen, frames):
    with pytest.raises(ValueError, match="frame count"):
        loader.load_track1_episode(frozen[0], 15, frames, frozen[7])


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "A" * 64, "z" * 64])
def test_caller_video_sha_must_be_strict(loader, frozen, digest):
    with pytest.raises(ValueError, match="64-hex"):
        loader.load_track1_episode(frozen[0], 15, 3, digest)


def test_path_traversal_rejected_without_resolving_elsewhere(loader, frozen):
    with pytest.raises(ValueError, match="traversal"):
        loader.load_track1_episode(frozen[0] / ".." / frozen[0].name, 15, 3, frozen[7])


def test_pure_loader_import_does_not_require_torch_or_remote_linux(loader, frozen):
    # Fixture executes on developer laptop; only report+two tiny NPZ paths read.
    result = load(loader, frozen)
    assert result.episode.provenance["ground_truth_used"] is False
    assert not (frozen[1] / "cari_forward/coconet.pth").exists()
    assert not (frozen[0] / "data").exists()


def test_exact_historical_converter_format_preserves_report_bytes(loader, frozen):
    root, _, reports, paths, _, sync, _, _ = frozen
    final = reports["conversion"]
    del final["episode_index"]
    final.update(producer_revision=loader.LEGACY_CONVERSION_REVISION,
                 script_sha256=loader.LEGACY_CONVERSION_SCRIPT_SHA256)
    sync()
    before = paths["conversion"].read_bytes()
    result = load(loader, frozen)
    assert result.manifest["legacy_episode15_conversion_format"] is True
    assert paths["conversion"].read_bytes() == before
    assert "episode_index" not in final


@pytest.mark.parametrize("kind", ["unknown_source", "wrong_revision", "no_identity", "present_null", "present_other"])
def test_missing_episode_never_generically_assumed_from_selected_directory(loader, frozen, kind):
    final = frozen[2]["conversion"]
    del final["episode_index"]
    final.update(producer_revision=loader.LEGACY_CONVERSION_REVISION,
                 script_sha256=loader.LEGACY_CONVERSION_SCRIPT_SHA256)
    if kind == "unknown_source": final["script_sha256"] = "0" * 64
    elif kind == "wrong_revision": final["producer_revision"] = "0" * 40
    elif kind == "no_identity": final.pop("script_sha256")
    elif kind == "present_null": final["episode_index"] = None
    else: final["episode_index"] = 0
    frozen[5]()
    with pytest.raises(ValueError): load(loader, frozen)


def test_legacy_converter_source_is_exact_and_hardwired_episode15(loader):
    import hashlib, subprocess
    repository = Path(loader.__file__).parents[1]
    source = subprocess.check_output(["rtk", "proxy", "git", "show",
        loader.LEGACY_CONVERSION_REVISION + ":infra/cari_converter.py"], cwd=repository)
    assert hashlib.sha256(source).hexdigest() == loader.LEGACY_CONVERSION_SCRIPT_SHA256
    assert b'base = root / "outputs/episode_000015"' in source
    report = {"producer_revision": loader.LEGACY_CONVERSION_REVISION,
              "script_sha256": loader.LEGACY_CONVERSION_SCRIPT_SHA256}
    for bad in (0, 29, True, "15", 15.):
        assert loader.legacy_episode15_conversion(report, bad) is False

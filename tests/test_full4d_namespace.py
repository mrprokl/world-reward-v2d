"""Tiny namespace/provenance checks; no model, media, GPU or Azure calls."""
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from world_reward.artifact_paths import episode_output, episode_relative, pin_path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "experiments/full4d-v1-" + "a" * 40 + "/outputs"
AZURE_ROOT = Path("/srv/scenesmith/world-reward")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    monkeypatch.delenv("WR_PIN_ROOT", raising=False)
    return {name: importlib.import_module(name) for name in (
        "cari_clip_inputs", "cari_shared_prepare", "cari_full_forward",
        "cari_full_refine", "cari_full_export")}


@pytest.mark.parametrize("prefix", ["outputs", PREFIX])
@pytest.mark.parametrize("episode", [0, 7, 29])
def test_public_inputs_and_full_stages_share_one_namespace(modules, monkeypatch, prefix, episode):
    monkeypatch.setenv("WR_OUTPUT_PREFIX", prefix)
    base = f"{prefix}/episode_{episode:06d}"
    public = modules["cari_clip_inputs"]
    spec = public.PublicClipSpec(episode, 97, "front_stereo_camera_left", 1152, 1536)
    assert episode_relative(episode) == base
    assert episode_output(AZURE_ROOT, episode) == AZURE_ROOT / base
    paths = public.relative_paths(spec)
    assert paths["export_seq"] == base + f"/cari_inputs/export/episode_{episode:06d}"
    assert paths["mesh"] == paths["export_seq"] + "/object_mesh/output_aligned.glb"
    for profile in ("default", "solid", "surface"):
        sources = public.source_paths(spec, object_source=profile)
        assert len(sources) == 15
        assert all(path.startswith(base + "/") for path in sources)
        suffix = "" if profile == "default" else "_" + profile
        assert public.dependency_paths(spec, object_source=profile)["object"] == (
            base + "/object_pose_full" + suffix + "/report.json")
    for name, role in (("cari_shared_prepare", "prepare"), ("cari_full_forward", "forward"),
                       ("cari_full_refine", "refined"), ("cari_full_export", "export")):
        assert modules[name].output_relative(episode) == base + "/cari_shared_" + role + "_v1"


@pytest.mark.parametrize("episode", [0, 29])
def test_consumer_pins_are_explicit_and_never_baseline_fallback(monkeypatch, tmp_path, episode):
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    monkeypatch.delenv("WR_PIN_ROOT", raising=False)
    for role in ("input", "shared_prepare", "shared_forward", "shared_refined"):
        legacy = tmp_path / "configs" / f"cari_clip_{episode:06d}_{role}_pins.json"
        assert pin_path(tmp_path, episode, role) == legacy
    monkeypatch.setenv("WR_OUTPUT_PREFIX", PREFIX)
    with pytest.raises(ValueError, match="isolated pins"):
        pin_path(tmp_path, episode, "input")
    pins = AZURE_ROOT / Path(PREFIX).parent / "pins"
    monkeypatch.setenv("WR_PIN_ROOT", str(pins))
    for role in ("input", "shared_prepare", "shared_forward", "shared_refined"):
        assert pin_path(tmp_path, episode, role) == pins / f"cari_clip_{episode:06d}_{role}_pins.json"
    for role in ("surface_mesh", "solid_mesh"):
        assert pin_path(tmp_path, episode, role) == pins / f"{role}_{episode:06d}_pins.json"
    monkeypatch.setenv("WR_PIN_ROOT", str(AZURE_ROOT / "outputs"))
    with pytest.raises(ValueError, match="same explicit experiment"):
        pin_path(tmp_path, episode, "input")


def test_wild_mesh_validation_rejects_a_legacy_absolute_mesh_in_candidate_namespace(modules, monkeypatch):
    public = modules["cari_clip_inputs"]
    spec = public.PublicClipSpec(7, 97, "front_stereo_camera_left", 1152, 1536)
    monkeypatch.setenv("WR_OUTPUT_PREFIX", PREFIX)
    K = public.inferred_camera(spec)
    wild = dict(schema="cari4d.mhr_wild_export.v2", frame_count=97, sequence=spec.sequence,
        camera_id=0, height=1152, width=1536,
        object_mesh_file=str(AZURE_ROOT / public.relative_paths(spec)["mesh"]),
        depth_backend="moge2", object_pose_frame="centered_axis_aligned",
        object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
        object_pose_storage_frame="output_aligned_mesh_frame",
        object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist(), intrinsics=K.tolist())
    edex = [{"cameras": [{"intrinsics": {"focal": [K[0, 0], K[1, 1]],
        "principal": [K[0, 2], K[1, 2]]}, "transform": np.eye(4)[:3].tolist()}]}]
    public.validate_wild(wild, edex, AZURE_ROOT, spec)
    wild["object_mesh_file"] = str(AZURE_ROOT / "outputs/episode_000007/cari_inputs/export/episode_000007/object_mesh/output_aligned.glb")
    with pytest.raises(ValueError, match="single-camera predicted wild export"):
        public.validate_wild(wild, edex, AZURE_ROOT, spec)


def test_namespace_helper_is_bound_in_every_actual_shared_stage_ledger(modules, tmp_path):
    names = set()
    for name in ("cari_shared_prepare", "cari_full_forward", "cari_full_refine", "cari_full_export"):
        tree = ast.parse((ROOT / "infra" / (name + ".py")).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_helpers")
        names.update(node.value for node in ast.walk(function) if isinstance(node, ast.Constant)
                     and isinstance(node.value, str) and node.value.endswith((".py", ".sh")))
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_source": name}))
        path.chmod(0o444)
    for name in ("cari_shared_prepare", "cari_full_forward", "cari_full_refine", "cari_full_export"):
        ledger = modules[name].source_helpers(tmp_path)
        assert "src/world_reward/artifact_paths.py" in ledger
        assert ledger["src/world_reward/artifact_paths.py"]["bytes"] > 0


def test_producer_sources_have_no_remaining_legacy_episode_output_literals():
    for name in ("cari_clip_inputs", "cari_prepare", "cari_body_adapter_smoke", "cari_shared_prepare",
                 "cari_full_forward", "cari_full_refine", "cari_full_export"):
        source = (ROOT / "infra" / (name + ".py")).read_text()
        assert '"outputs/episode_' not in source
        assert "'outputs/episode_" not in source
        assert '"outputs/" + spec.sequence' not in source
        assert "'outputs/'+spec.sequence" not in source

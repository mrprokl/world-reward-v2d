"""Tiny fabricated assets/Git/image metadata only; no Azure, models or Docker."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def gate():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("wr_frontend_inventory", root / "infra/frontend_replica_inventory.py")
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_fixed_constants_equal_genuine_existing_binders(gate, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "infra")); monkeypatch.syspath_prepend(str(root / "src"))
    import body_smoke, object_smoke, hand_synthetic_infer, hand_synthetic_masks, multiview_full_gate, acquire_auxiliary
    assert gate.GROUNDING == hand_synthetic_masks.ASSETS
    assert gate.CHECKPOINTS == multiview_full_gate.CHECKPOINTS and gate.YAMLS == multiview_full_gate.YAMLS
    assert gate.UPSTREAM == body_smoke.UPSTREAM_REVISION and gate.BODY_REV == body_smoke.BODY_REVISION
    assert gate.DINO_REVS == acquire_auxiliary.REPOSITORIES
    assert gate.DINO_REVS["dinov2"] == object_smoke.DINOV2_REVISION
    assert gate.FIXED[gate.BODY + "/model.ckpt"] == (hand_synthetic_infer.BODY_SHA, hand_synthetic_infer.BODY_BYTES)
    for name, (size, digest) in gate.CONSTANT_EVIDENCE.items():
        raw = (root / "infra" / (name + ".py")).read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == (size, digest)


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "a//b", "a/./b", "a\\b", "", "a\x00b"])
def test_name_closed(gate, name):
    with pytest.raises(ValueError): gate.safe_name(name)


def test_real_kit_namespace_not_fabricated_initializer(gate, tmp_path):
    assert gate.KIT_NAMESPACE + "/__init__.py" not in gate.CARDS
    assert gate.KIT_NAMESPACE + "/mesh_budget.py" in gate.FIXED
    kit = tmp_path / gate.KIT_NAMESPACE
    kit.mkdir(parents=True)
    # Namespace is accepted up to the next independently missing prerequisite.
    with pytest.raises(FileNotFoundError): gate.inventory(tmp_path)
    (kit / "__init__.py").write_bytes(b"# invented")
    with pytest.raises(ValueError, match="namespace package"): gate.inventory(tmp_path)


def test_regular_identity_exact_before_after_empty_source_and_no_alias(gate, tmp_path, monkeypatch):
    path = tmp_path / "asset"; path.write_bytes(b"tiny")
    result = gate.identity(path, 4)
    assert result == dict(bytes=4, sha256=hashlib.sha256(b"tiny").hexdigest(), git_blob_sha1=hashlib.sha1(b"blob 4\0tiny").hexdigest())
    with pytest.raises(ValueError): gate.identity(path, 3)
    link = tmp_path / "alias"; link.symlink_to(path)
    with pytest.raises(ValueError): gate.identity(link)
    path.write_bytes(b"")
    with pytest.raises(ValueError): gate.identity(path)
    assert gate.identity(path, allow_empty=True)["bytes"] == 0
    native = gate.state; calls = []
    def changed(path):
        calls.append(1); value = native(path)
        return value if len(calls) == 1 else (*value[:-1], value[-1] + 1)
    monkeypatch.setattr(gate, "state", changed)
    with pytest.raises(ValueError, match="changed"): gate.identity(path, allow_empty=True)


def fake_moge(gate, tmp_path, links=True):
    raw = b"tiny learned model, not a checkpoint"
    gate.MOGE_EXPECTED = (hashlib.sha256(raw).hexdigest(), len(raw))
    snapshot = tmp_path / gate.MOGE_SNAPSHOT; snapshot.mkdir(parents=True)
    model = snapshot / "model.pt"
    if links:
        blob = tmp_path / gate.HF / "blobs/ab" / ("a" * 64); blob.parent.mkdir(parents=True); blob.write_bytes(raw)
        repository_blob = tmp_path / gate.MOGE_REPO / "blobs" / ("b" * 64); repository_blob.parent.mkdir()
        repository_blob.symlink_to("../../blobs/ab/" + "a" * 64)
        model.symlink_to("../../blobs/" + "b" * 64)
    else: model.write_bytes(raw)
    (snapshot / "README.md").write_bytes(b"card")
    return model


@pytest.mark.parametrize("links", [True, False])
def test_moge_exact_snapshot_actual_discovered_binding_not_guessed(gate, tmp_path, links):
    fake_moge(gate, tmp_path, links)
    records, binding = gate.moge_chain(tmp_path, "model.pt")
    assert binding["links"] == (2 if links else 0) and binding["independent_primary_model_identity_verified"]
    assert len(records) == binding["links"] + 1 and binding["XET_path_discovered_not_fabricated"]
    assert binding["sha256"] == gate.MOGE_EXPECTED[0]
    assert set(records) <= {gate.MOGE_SNAPSHOT + "/model.pt", gate.MOGE_REPO + "/blobs/" + "b" * 64, gate.HF + "/blobs/ab/" + "a" * 64}


@pytest.mark.parametrize("fault", ["revision", "absolute", "escape", "cycle", "wrong_sha", "parent_alias"])
def test_moge_failure_never_scans_other_cache_or_uses_other_snapshot(gate, tmp_path, fault):
    model = fake_moge(gate, tmp_path)
    if fault == "revision": (tmp_path / gate.MOGE_REPO / "snapshots" / ("c" * 40)).mkdir()
    elif fault == "wrong_sha": gate.MOGE_EXPECTED = ("0" * 64, gate.MOGE_EXPECTED[1])
    elif fault == "parent_alias":
        parent = tmp_path / gate.MOGE_REPO / "blobs"; replacement = parent.with_name("saved-blobs"); parent.rename(replacement); parent.symlink_to(replacement)
    else:
        model.unlink()
        if fault == "absolute": model.symlink_to("/etc/passwd")
        elif fault == "escape": model.symlink_to("../../../../../.secrets/hf_token")
        else:
            target = tmp_path / gate.MOGE_REPO / "blobs" / ("c" * 64); target.symlink_to("c" * 64); model.symlink_to("../../blobs/" + "c" * 64)
    with pytest.raises(ValueError): gate.moge_chain(tmp_path, "model.pt")


def test_selected_public_source_never_includes_assets_data_truth_or_unneeded_modules(gate):
    assert gate.selected_source(gate.BODY_PACKAGE + "/data/utils/io.py")
    assert gate.selected_source(gate.CARI + "/lib_mhr/contact.py")
    assert gate.selected_source(gate.CARI + "/prep/mhr_sensor_depth_h5.py")
    for name in ("data/track_2/x", "data/track_1/x", "validation/tudl_rgb_v1/eval_private/source/x", "outputs/episode_000004/report.json",
            gate.BODY_PACKAGE + "/assets/body.ply", gate.BODY_PACKAGE + "/data/x.png", gate.CARI + "/data/x.py", gate.CARI + "/model/coconet.py"):
        assert not gate.selected_source(name)
    assert gate.selected_source("dinov2/models/vision_transformer.py", dino="dinov2")
    assert not gate.selected_source("dinov2/data/gt.png", dino="dinov2")


def test_source_git_blob_identity_rejects_missing_or_modified_without_lazyfetch(gate, tmp_path, monkeypatch):
    repository = tmp_path / "vendor/public"; repository.mkdir(parents=True)
    path = repository / "hubconf.py"; path.write_bytes(b"tiny source")
    blob = hashlib.sha1(b"blob 11\0tiny source").hexdigest()
    def git(repo, *args):
        if args[0] == "rev-parse": return ("a" * 40).encode()
        if args[0] == "ls-tree": return b"100644 blob " + blob.encode() + b"\thubconf.py\0"
        return b""
    monkeypatch.setattr(gate, "git", git)
    with pytest.raises(ValueError, match="Complete"): gate.source_inventory(tmp_path, "vendor/public", "a" * 40, "dinov2")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"): gate.source_inventory(tmp_path, "vendor/public", "a" * 40, "dinov2")


def image_fixture():
    return dict(Id="sha256:" + "a" * 64, Architecture="amd64", Os="linux", Size=100,
        RootFS={"Type": "layers", "Layers": ["sha256:" + "b" * 64, "sha256:" + "c" * 64]})


def test_image_only_inspect_bounded_metadata_never_environment_no_fake_config(gate, monkeypatch):
    actual = image_fixture(); seen = []
    def query(args): seen.append(args); return json.dumps(actual).encode()
    monkeypatch.setattr(gate, "command", query)
    result = gate.image_record("world-reward/test:0.1", actual["Id"], actual)
    assert result["platform_config_id"] is None and not result["sealed_export_graph_verified"]
    assert result["RootFS"]["Layers"] == actual["RootFS"]["Layers"]
    assert seen[0][:3] == ["docker", "image", "inspect"] and "Config" not in seen[0][-1] and "Env" not in seen[0][-1]


@pytest.mark.parametrize("fault", ["id", "platform", "rootfs", "receipt", "empty", "extra"])
def test_image_actual_binders_failclosed(gate, monkeypatch, fault):
    actual = image_fixture(); receipt = json.loads(json.dumps(actual)); expected = actual["Id"]
    if fault == "id": actual["Id"] = "sha256:" + "d" * 64
    elif fault == "platform": actual["Architecture"] = "arm64"
    elif fault == "rootfs": actual["RootFS"]["Layers"][0] = "not_sha"
    elif fault == "receipt": receipt["RootFS"]["Layers"].reverse()
    elif fault == "empty": actual["RootFS"]["Layers"] = []
    else: actual["Config"] = {"Env": ["unsafe"]}
    monkeypatch.setattr(gate, "command", lambda _: json.dumps(actual).encode())
    with pytest.raises(ValueError): gate.image_record("tag", expected, receipt)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_receipts_strict_json(gate, raw):
    with pytest.raises(ValueError): gate.strict_json(raw)


@pytest.mark.parametrize("value", [{"hf_token": "no"}, {"x": "Bearer credential"}, {"x": "hf_" + "a" * 25}])
def test_no_secrets_in_reproduction_metadata(gate, value):
    with pytest.raises(ValueError): gate.no_secrets(value)


def test_summary_under4KB_fullmanifest_not_printed(gate):
    report = dict(stage="inventory", status="pass", elapsed_seconds=1., entries={str(i): {"type": "file", "bytes": 1} for i in range(2000)},
        entries_sha256="b" * 64, blockers=["unsealed"], internal_moge1={"model.pt": dict(bytes=1, sha256="a" * 64, links=2, resolved_file=gate.HF + "/blobs/ab/" + "a" * 64)},
        images={"objects": {**image_fixture(), "rootfs_sha256": "c" * 64}})
    raw = gate.small_summary(report)
    assert len(raw.encode()) < 4000 and '"entries"' not in raw and '"Layers"' not in raw
    assert json.loads(raw)["files"] == 2000 and json.loads(raw)["replica_ready"] is False


def test_stdlib_only_closure_no_mutation_or_transfers(gate, monkeypatch):
    root = Path(__file__).resolve().parents[1]; monkeypatch.syspath_prepend(str(root / "infra"))
    import azure_job
    files = {str(p.relative_to(root)): p.read_bytes() for d in ("infra", "src", "configs") for p in (root / d).rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".sh", ".cpp", ".toml")}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, "infra/run_frontend_replica_inventory.sh")
    assert {p for p in selected if p.startswith("infra/")} == {"infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh"}
    source = (root / "infra/frontend_replica_inventory.py").read_text(); wrapper = (root / "infra/run_frontend_replica_inventory.sh").read_text()
    assert all(x not in source for x in ("import torch", "import numpy", "snapshot_download", "urlopen", "extractall", "docker save"))
    assert "--gpus" not in wrapper and "flock" not in wrapper and "docker run" not in wrapper.replace("No Docker run/save/load/build", "")
    assert "603s nice -n 15 ionice -c 3 python3 -I -B" in wrapper and "--kill-after=10s" in wrapper


def test_sanitized_actionable_failure_never_echoes_credentials(gate):
    assert gate.error_text(ValueError("Missing: weights/sam3d/model.pt")) == "Missing: weights/sam3d/model.pt"
    for text in ("hf_" + "a" * 25, "Bearer private", "AccountKey=private", "https://host/?sig=private", "password=private"):
        assert "private" not in gate.error_text(ValueError(text))
    assert len(gate.error_text(ValueError("x" * 500))) == 250


def test_raw_build_receipts_evidence_only_not_transfereligible(gate):
    import inspect
    source = inspect.getsource(gate.inventory)
    assert 'evidence[name] = {**entries.pop(name)' in source
    assert '"raw_build_receipt_must_not_be_exported": True' in source
    assert "credentials_read=False" not in source
    assert "raw_build_receipts_transfer_eligible=False" in source

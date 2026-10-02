"""Tiny exact-asset TAR roundtrips and hostile inventory; no real data/network."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/research_transfer.py"
    spec = importlib.util.spec_from_file_location("wr_research_transfer_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source(gate, tmp_path, monkeypatch, private):
    root = tmp_path/"source"; root.mkdir()
    model = b"tiny pinned model"; monkeypatch.setattr(gate, "MODEL_BYTES", len(model)); monkeypatch.setattr(gate, "MODEL_SHA", hashlib.sha256(model).hexdigest())
    # Storage filename remains frozen, while tiny fixture content is separately pinned.
    files = {gate.BLOB: model}
    images = []
    for name in gate.public_names():
        files[name] = name.encode(); stem = Path(name).stem.split("_")
        images.append({"scene_id": int(stem[1]), "frame_id": int(stem[3]), "file": Path(name).name,
            "sha256": hashlib.sha256(files[name]).hexdigest(), "width": 640, "height": 480})
    public = {"schema": "world-reward-tudl-rgb-v1", "revision": gate.TUDL_REV, "license": "CC-BY-SA-4.0", "selection": "names", "images": images}
    files[gate.BASE+"/inputs/manifest.json"] = json.dumps(public).encode(); monkeypatch.setattr(gate, "PUBLIC_SHA", hashlib.sha256(files[gate.BASE+"/inputs/manifest.json"]).hexdigest())
    files["results/weights-acquisition.json"] = json.dumps({"assets": [{"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": gate.MODEL_REV, "cache_dir": str(root/gate.HF)}]}).encode()
    if private:
        for name in gate.private_names(): files[name] = b"private " + name.encode()
        receipt = {"stage": "external_tudl_rgb_only_validation_acquisition", "status": "pass", "dataset_revision": gate.TUDL_REV,
            "public_manifest_sha256": gate.PUBLIC_SHA, "challenge_inputs_used": False, "inference_performed": False,
            "retained_files": [{"file": name.removeprefix(gate.BASE+"/eval_private/"), "sha256": hashlib.sha256(files[name]).hexdigest(), "bytes": len(files[name])} for name in gate.private_names()]}
        files[gate.BASE+"/eval_private/acquisition-report.json"] = json.dumps(receipt).encode()
        monkeypatch.setattr(gate, "ACQUISITION_SHA", hashlib.sha256(files[gate.BASE+"/eval_private/acquisition-report.json"]).hexdigest())
    for name, data in files.items():
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for name, target in ((gate.REPO_BLOB, gate.BLOB), (gate.SNAPSHOT, gate.REPO_BLOB)):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.symlink_to(os.path.relpath(root/target, path.parent))
    return root


@pytest.mark.parametrize("private", [False, True])
def test_exact_roundtrip_model_once_public_only_or_private42(gate, tmp_path, monkeypatch, private):
    root = source(gate, tmp_path, monkeypatch, private); path = tmp_path/"transfer.tar"
    sha = gate.export(root, path, private); destination = tmp_path/"destination"; destination.mkdir()
    # VM02 retains same WR_ROOT path: the fixture needs to imitate that frozen receipt.
    monkeypatch.setattr(gate, "metadata", lambda r, p, read=None: {})
    manifest = gate.verify(path, sha, destination)
    assert len(manifest["entries"]) == 56 if private else len(manifest["entries"]) == 14
    assert sum(e["path"] == gate.BLOB for e in manifest["entries"]) == 1
    assert manifest["include_private"] is private and manifest["predictions_included"] is False
    gate.extract(destination, path, sha)
    assert (destination/gate.SNAPSHOT).resolve() == destination/gate.BLOB
    assert (destination/gate.BLOB).read_bytes() == b"tiny pinned model"
    assert (destination/gate.BASE/"inputs/manifest.json").stat().st_mode & 0o777 == 0o444
    if private:
        assert (destination/gate.BASE/"eval_private").stat().st_mode & 0o777 == 0o700
        assert (destination/gate.BASE/"eval_private/source/licenses/attribution.json").stat().st_mode & 0o777 == 0o600
    else: assert not (destination/gate.BASE/"eval_private").exists()
    with pytest.raises(FileExistsError): gate.extract(destination, path, sha)


def rewrite(path, target, modify):
    with tarfile.open(path) as archive:
        values = [(m, archive.extractfile(m).read() if m.isfile() else None) for m in archive.getmembers()]
    values = modify(values)
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as archive:
        for member, data in values: archive.addfile(member, io.BytesIO(data) if data is not None else None)


@pytest.mark.parametrize("fault", ["sha", "traversal", "duplicate", "unknown", "hardlink", "link_escape", "model", "mode", "scope", "order"])
def test_hostile_tar_fails_before_any_extract_write(gate, tmp_path, monkeypatch, fault):
    root = source(gate, tmp_path, monkeypatch, False); original = tmp_path/"original.tar"; sha = gate.export(root, original, False)
    bad = tmp_path/"bad.tar"
    def modify(values):
        if fault == "traversal": values[1][0].name = "../outside"
        elif fault == "duplicate": values.append(values[1])
        elif fault == "unknown": values[1][0].name = "weights/token"
        elif fault == "hardlink": values[1][0].type = tarfile.LNKTYPE; values[1][0].linkname = gate.BLOB; values[1][0].size = 0; values[1] = (values[1][0], None)
        elif fault == "link_escape":
            for member, _ in values:
                if member.issym(): member.linkname = "../../../../../outside"; break
        elif fault == "model":
            for i, (member, data) in enumerate(values):
                if member.name == gate.BLOB: values[i] = (member, b"X"*len(data)); break
        elif fault == "mode": values[1][0].mode = 0o777
        elif fault == "scope":
            member, data = values[0]; manifest = json.loads(data); manifest["secrets_included"] = True; data = json.dumps(manifest).encode(); member.size = len(data); values[0] = (member, data)
        elif fault == "order": values[1], values[2] = values[2], values[1]
        return values
    rewrite(original, bad, modify); destination = tmp_path/"destination"; destination.mkdir()
    expected = sha if fault == "sha" else gate.file_hash(bad)
    with pytest.raises(ValueError): gate.extract(destination, bad, expected)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("fault", ["HF_link", "parent_link", "private_missing", "public_changed", "token", "blob_changed"])
def test_source_inventory_no_escape_secrets_missing_or_mutation(gate, tmp_path, monkeypatch, fault):
    root = source(gate, tmp_path, monkeypatch, True)
    if fault == "HF_link":
        path = root/gate.SNAPSHOT; path.unlink(); path.symlink_to("/outside")
    elif fault == "parent_link":
        folder = root/gate.BASE/"inputs"; moved = tmp_path/"moved"; folder.rename(moved); folder.symlink_to(moved)
    elif fault == "private_missing": (root/gate.private_names()[0]).unlink()
    elif fault == "public_changed": (root/gate.public_names()[0]).write_bytes(b"changed")
    elif fault == "token":
        path = root/"results/weights-acquisition.json"; value = json.loads(path.read_text()); value["HF_TOKEN"] = "secret"; path.write_text(json.dumps(value))
    else: (root/gate.BLOB).write_bytes(b"altered")
    with pytest.raises((ValueError, FileNotFoundError)): gate.inventory(root, True)


def test_extracted_metadata_audit_before_any_write(gate, tmp_path, monkeypatch):
    root = source(gate, tmp_path, monkeypatch, False); path = tmp_path/"transfer.tar"; sha = gate.export(root, path, False)
    # Wrong destination root makes the source cache_dir contract fail, never auto-rewritten.
    destination = tmp_path/"other-root"; destination.mkdir()
    with pytest.raises(ValueError, match="acquisition receipt mismatch"): gate.extract(destination, path, sha)
    assert not list(destination.iterdir())


def test_allowlist_exact_and_credential_detection(gate):
    assert len(gate.private_names()) == 41 and len(set(gate.private_names())) == 41
    assert not any(x in p for p in gate.whitelist(True) for x in ("predictions", "vendor", "/refs/", "token", "challenge"))
    for value in ({"password": "x"}, {"api_key": "x"}, ["Bearer secret"], "url?sig=secret", "hf_"+"a"*25):
        with pytest.raises(ValueError): gate.reject_secrets(value)
    gate.reject_secrets({"assets": [{"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": "pinned"}]})


def test_CLI_remote_only_SHA_and_UID_required(gate, tmp_path, monkeypatch):
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError): gate.main(["export", "--archive", str(tmp_path/"out.tar")])
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError): gate.main(["extract", "--archive", str(tmp_path/"out.tar")])
    monkeypatch.setattr(gate.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="UID1000"): gate.main(["extract", "--archive", str(tmp_path/"out.tar"), "--sha256", "a"*64])
    with pytest.raises(SystemExit): gate.main(["export", "--root", "/other", "--archive", "out.tar"])

"""Reuse the existing tiny procedural archive cohort; no remote asset reads."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from test_frontend_asset_archive import contract, cohort, pin, rebuild


@pytest.fixture
def extract():
    path = Path(__file__).resolve().parents[1] / "infra/frontend_asset_extract.py"
    spec = importlib.util.spec_from_file_location("wr_asset_extract", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def transfer(extract, cohort, tmp_path):
    archive = tmp_path / "pinned.tar"; receipt = cohort["build"](archive)
    code = Path(extract.__file__).parent.parent
    sources = {n: {k: pin((code / n).read_bytes())[k] for k in ("bytes", "sha256")} for n in extract.SOURCE_FILES}
    destination = tmp_path / "new-private-assets"
    def run(): return extract.extract_archive(archive, receipt["archive_identity"], receipt["manifest_identity"], cohort["code"], cohort["helpers"], destination, sources)
    return dict(archive=archive, receipt=receipt, destination=destination, run=run, sources=sources)


def test_full_verified_extract_manifest_receipts_and_links(extract, contract, cohort, transfer):
    result = transfer["run"](); assert result["status"] == "pass", result
    out = transfer["destination"]; assert out.stat().st_mode & 0o777 == 0o700
    for name, row in cohort["entries"].items():
        path = out / name
        if row["type"] == "symlink": assert path.is_symlink() and path.readlink().as_posix() == row["link"]
        else:
            assert path.read_bytes() == (cohort["root"] / name).read_bytes() and path.stat().st_mode & 0o777 == 0o400
    assert hashlib.sha256((out / contract.MANIFEST).read_bytes()).hexdigest() == transfer["receipt"]["manifest_identity"]["sha256"]
    assert result["imported_links"] == 3 and result["complete_payload_rehashed"]
    assert result["replica_ready"] is result["license_eligibility_verified"] is result["CUDA_verified"] is False
    assert result["promotion_performed"] is result["models_loaded"] is result["images_imported"] is False
    assert not result["partial_output_retained"]


@pytest.mark.parametrize("fault", ["archive", "manifest", "source", "oldhelper"])
def test_all_pins_verified_before_any_output_write(extract, cohort, transfer, fault):
    if fault == "archive": transfer["receipt"]["archive_identity"]["sha256"] = "0" * 64
    elif fault == "manifest": transfer["receipt"]["manifest_identity"]["sha256"] = "0" * 64
    elif fault == "source": transfer["sources"][extract.SOURCE_FILES[1]]["sha256"] = "0" * 64
    else:
        path = cohort["code"] / "infra/frontend_replica_inventory.py"; path.chmod(0o600); path.write_text("changed"); path.chmod(0o400)
    result = transfer["run"](); assert result["status"] == "fail" and not result["partial_output_retained"]
    assert not transfer["destination"].exists()


@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_destination_never_merged_overwritten_or_removed(transfer, kind):
    out = transfer["destination"]
    if kind == "directory": out.mkdir(); (out / "retained").write_bytes(b"user work")
    elif kind == "file": out.write_bytes(b"user work")
    else: out.symlink_to("absent")
    result = transfer["run"](); assert result["status"] == "fail" and not result["partial_output_retained"]
    assert out.exists() or out.is_symlink()
    if kind == "directory": assert (out / "retained").read_bytes() == b"user work"
    elif kind == "file": assert out.read_bytes() == b"user work"


def test_partial_owned_tree_retained_private_no_links_before_regulars(extract, transfer, monkeypatch):
    native = extract._write; calls = []
    def fail(path, stream, row, gate, directories):
        native(path, stream, row, gate, directories); calls.append(path)
        if len(calls) == 2: raise ValueError("procedural failure")
    monkeypatch.setattr(extract, "_write", fail)
    result = transfer["run"](); out = transfer["destination"]
    assert result["status"] == "fail" and result["partial_output_retained"] and out.is_dir()
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(not p.is_symlink() for p in out.rglob("*"))
    assert all(p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o400) for p in out.rglob("*"))


@pytest.mark.parametrize("fault", ["tamper", "extra", "link"])
def test_complete_postwrite_inventory_rejects_foreign_tamper(extract, cohort, transfer, monkeypatch, fault):
    native = extract._inventory; once = []
    def mutate(destination, entries, raw, gate, directories):
        if not once:
            once.append(True)
            if fault == "extra": (destination / "unknown").write_bytes(b"user work")
            elif fault == "tamper":
                path = destination / cohort["gate"].BODY / "LICENSE"; path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o400)
            else:
                path = destination / (cohort["gate"].MOGE_SNAPSHOT + "/model.pt"); path.unlink(); path.symlink_to("/outside")
        return native(destination, entries, raw, gate, directories)
    monkeypatch.setattr(extract, "_inventory", mutate)
    result = transfer["run"](); assert result["status"] == "fail" and result["partial_output_retained"]
    assert transfer["destination"].exists()


@pytest.mark.parametrize("fault", ["extra", "traversal", "hardlink", "link", "duplicate"])
def test_archive_invalid_graph_never_reserves_destination(contract, transfer, fault):
    def mutate(members):
        member, raw = members[1]
        if fault == "extra": member.name = "results/unknown.json"
        elif fault == "traversal": member.name = "../foreign"
        elif fault == "duplicate": members.insert(1, members[1])
        elif fault == "link": next(m for m, _ in members if m.issym()).linkname = "/outside"
        else:
            import tarfile
            member.type = tarfile.LNKTYPE; member.size = 0; member.linkname = "outside"; members[1] = member, None
    raw = rebuild(contract, transfer["archive"].read_bytes(), mutate)
    transfer["archive"].chmod(0o600); transfer["archive"].write_bytes(raw); transfer["archive"].chmod(0o400)
    transfer["receipt"]["archive_identity"] = {k: pin(raw)[k] for k in ("bytes", "sha256")}
    result = transfer["run"](); assert result["status"] == "fail" and not transfer["destination"].exists()


def test_archive_changed_after_extraction_retains_private_failure(extract, transfer, monkeypatch):
    native = extract._inventory; once = []
    def mutate(*args):
        result = native(*args)
        if not once:
            once.append(True); path = transfer["archive"]; path.chmod(0o600)
            with path.open("r+b") as stream: stream.seek(-1, 2); stream.write(b"X")
            path.chmod(0o400)
        return result
    monkeypatch.setattr(extract, "_inventory", mutate)
    result = transfer["run"](); assert result["status"] == "fail" and result["partial_output_retained"]


def test_failure_metadata_never_echoes_untrusted_secret_pin(extract, transfer):
    transfer["sources"][extract.SOURCE_FILES[1]]["sha256"] = "hf_" + "x" * 30
    result = transfer["run"]()
    assert result["status"] == "fail" and "hf_" not in str(result) and not transfer["destination"].exists()


def test_atomic_private_mode_under_permissive_umask(extract, transfer, monkeypatch):
    import os
    native = extract._write; observed = []
    def check(path, *args):
        result = native(path, *args); observed.append(path.stat().st_mode & 0o777); return result
    monkeypatch.setattr(extract, "_write", check); old = os.umask(0)
    try: result = transfer["run"]()
    finally: os.umask(old)
    assert result["status"] == "pass" and set(observed) == {0o400}


def test_source_has_no_extractall_model_network_or_tree_cleanup(extract):
    source = Path(extract.__file__).read_text()
    assert all(word not in source for word in ("extractall", "rmtree", "import torch", "import numpy", "urllib", "subprocess", "os.environ"))

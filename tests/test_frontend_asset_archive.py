"""Procedural tiny assets/TARs only; no network, Azure, images or model loading."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def contract():
    path = Path(__file__).resolve().parents[1] / "infra/frontend_asset_archive.py"
    spec = importlib.util.spec_from_file_location("wr_asset_archive", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pin(raw, blob=False):
    result = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if blob: result["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return result


@pytest.fixture
def cohort(contract, tmp_path):
    root = tmp_path / "assets"; root.mkdir(); code = tmp_path / "authenticated-old-code"; (code / "infra").mkdir(parents=True)
    genuine = Path(__file__).resolve().parents[1] / contract.INVENTORY_SOURCES[0]
    spec = importlib.util.spec_from_file_location("wr_original_inventory_fixture", genuine)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    fixed = {name: (pin(("asset:" + name).encode())["sha256"], len(("asset:" + name).encode())) for name in gate.FIXED}
    model = b"procedural model bytes, never decoded"
    source = genuine.read_bytes() + ("\nFIXED=" + repr(fixed) + "\nMOGE_EXPECTED=" + repr((pin(model)["sha256"], len(model))) + "\n").encode()
    helpers = {contract.INVENTORY_SOURCES[0]: pin(source, True), contract.INVENTORY_SOURCES[1]: pin(b"# inert procedural source wrapper\n", True)}
    for name, raw in ((contract.INVENTORY_SOURCES[0], source), (contract.INVENTORY_SOURCES[1], b"# inert procedural source wrapper\n")):
        path = code / name; path.write_bytes(raw); path.chmod(0o400)
    gate = contract._source(code, helpers); entries = {}
    def add(name, raw, role):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        entries[name] = dict(type="file", **pin(raw, True), role=role)
    for name in gate.FIXED: add(name, ("asset:" + name).encode(), "asset")
    for name in gate.CARDS: add(name, b"original license/card/config notice", "license_card_or_config")
    for name in contract.REG4: add("weights/sam3d/torch_home/hub/checkpoints/" + name, ("tiny:" + name).encode(), "reg4_first_observed_receipt_bound")
    sources = []
    for repository, revision, dino in contract._repositories(gate):
        if dino: names = {"hubconf.py", "LICENSE.md" if dino == "dinov3" else "LICENSE", "MODEL_CARD.md", dino + "/__init__.py", dino + "/models/test.py"}
        else:
            names = {"LICENSE", gate.BODY_PACKAGE + "/LICENSE", gate.BODY_PACKAGE + "/__init__.py",
                *[gate.CARI + "/lib_mhr/" + n for n in gate.LIB_MHR], *[gate.CARI + "/prep/" + n for n in gate.PREP],
                *[gate.CARI + "/" + n for n in gate.EXTRA_SOURCE], *[gate.BODY_PACKAGE + "/" + n for n in gate.BODY_DATA]}
        rows = {}
        for relative in names:
            name = repository + "/" + relative
            add(name, b"" if name.endswith("/__init__.py") else ("source/license:" + relative).encode(), "public_source"); rows[name] = entries[name]
        sources.append(dict(path=repository, revision=revision, selected_worktree_clean_verified=True,
            complete_checkout_clean_verified=False, source_files=len(rows), source_sha256=gate.digest_json(rows)))
    snapshot = gate.MOGE_SNAPSHOT
    blob = gate.HF + "/blobs/ab/" + "a" * 64; intermediate = gate.MOGE_REPO + "/blobs/" + "b" * 64
    add(blob, model, "internal_moge1")
    readme = gate.MOGE_REPO + "/blobs/" + "c" * 40; add(readme, b"original MoGe license/card", "internal_moge1")
    for name, link, target in ((snapshot + "/model.pt", "../../blobs/" + "b" * 64, intermediate),
            (intermediate, "../../blobs/ab/" + "a" * 64, blob), (snapshot + "/README.md", "../../blobs/" + "c" * 40, readme)):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.symlink_to(link)
        entries[name] = dict(type="symlink", link=link, target=target, role="internal_moge1")
    moge = {}
    for label, final, links in (("model.pt", blob, 2), ("README.md", readme, 1)):
        moge[label] = dict(snapshot=snapshot + "/" + label, resolved_file=final, links=links,
            **{k: entries[final][k] for k in ("bytes", "sha256", "git_blob_sha1")},
            independent_primary_model_identity_verified=label == "model.pt", XET_path_discovered_not_fabricated=True)
    principal = dict(assets=[dict(repo_id=repo, revision=revision, **{field: str(root / folder)}) for repo, revision, field, folder in (
        ("nvidia/cari4d_commercial", "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9", "path", "weights/cari4d/cari4d"),
        ("facebook/sam-3d-body-dinov3", gate.BODY_REV, "path", gate.BODY), ("facebook/sam-3d-objects", gate.OBJECT_REV, "path", gate.OBJECT),
        ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "path", "weights/sam2"),
        ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "path", "weights/grounding_dino"),
        ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340", "cache_dir", "weights/cari4d/hf_home/hub"),
        ("Ruicheng/moge-vitl", gate.MOGE_REV, "cache_dir", gate.HF))],
        cari4d_sha256="78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3", mhr_license="Apache-2.0",
        scope="principal_hf_and_mhr_assets_only", auxiliary_assets_required=["FoundationPose", "DINOv2", "DINOv3_torch_hub"])
    rows = []
    for filename in (*contract.RELEASES, *contract.REG4):
        model_name = filename.removeprefix("dinov2_").split("_")[0]
        row = dict(filename=filename, url="https://dl.fbaipublicfiles.com/dinov2/dinov2_" + model_name + "/" + filename,
            sha256=contract.RELEASES.get(filename, "0" * 64), bytes=1,
            hash_source="official_pinned_downloader" if filename in contract.RELEASES else "first_observed_https_download")
        if filename in contract.REG4:
            row.update({k: entries["weights/sam3d/torch_home/hub/checkpoints/" + filename][k] for k in ("bytes", "sha256")})
        rows.append(row)
    auxiliary = dict(scope="DINO_source_and_checkpoints_only", source_revisions=gate.DINO_REVS, checkpoints=rows, foundationpose_acquired=False)
    for name, value in zip(gate.RECEIPTS[:2], (principal, auxiliary)): add(name, json.dumps(value, indent=2).encode() + b"\n", "source_receipt")
    report = dict(stage="frontend_replica_prerequisite_inventory", status="pass", producer_revision="a" * 40,
        source_helpers=helpers, entries=entries, entries_sha256=gate.digest_json(entries), sources=sources, internal_moge1=moge,
        actual_prerequisites_unchanged=True, evidence_only_files={n: dict(transfer_eligible=False, raw_build_receipt_must_not_be_exported=True) for n in gate.RECEIPTS[2:]})
    for key in ("challenge_data_read", "private_validation_read", "predictions_read", "GPU_used", "models_loaded", "raw_build_receipts_transfer_eligible", "license_eligibility_verified", "training_overlap_verified"): report[key] = False
    path = tmp_path / "pinned-inventory-report.json"
    def freeze_report():
        report["entries_sha256"] = gate.digest_json(entries); path.chmod(0o600) if path.exists() else None
        raw = json.dumps(report, sort_keys=True).encode(); path.write_bytes(raw); path.chmod(0o400)
        return dict(**pin(raw), producer_revision=report["producer_revision"], entries_sha256=report["entries_sha256"],
            files=sum(r["type"] == "file" for r in entries.values()), links=sum(r["type"] == "symlink" for r in entries.values()), total_bytes=sum(r.get("bytes", 0) for r in entries.values()))
    inventory_pin = freeze_report(); exporter = pin(Path(contract.__file__).read_bytes())
    def build(destination): return contract.build_archive(root, path, code, inventory_pin, helpers, destination, exporter)
    return dict(root=root, code=code, gate=gate, helpers=helpers, report=report, path=path, inventory_pin=inventory_pin,
        entries=entries, build=build, freeze_report=freeze_report, exporter=exporter)


def test_authenticated_deterministic_full_archive_and_verify(contract, cohort, tmp_path):
    one, two = tmp_path / "one.tar", tmp_path / "two.tar"
    receipt = cohort["build"](one); again = cohort["build"](two)
    assert one.read_bytes() == two.read_bytes() and receipt == again and one.stat().st_mode & 0o777 == 0o400
    assert len(json.dumps(receipt)) < 4000 and receipt["links"] == 3
    assert receipt["replica_ready"] is receipt["license_eligibility_verified"] is receipt["CUDA_verified"] is False
    with tarfile.open(one) as archive:
        names = archive.getnames(); assert names == [contract.MANIFEST, *sorted(cohort["entries"])]
        assert all(n not in names for n in cohort["gate"].RECEIPTS[2:])
        for name in cohort["gate"].RECEIPTS[:2]: assert archive.extractfile(name).read() == (cohort["root"] / name).read_bytes()
        for name, row in cohort["entries"].items():
            if row["type"] == "symlink": assert archive.getmember(name).linkname == row["link"]
        assert archive.extractfile(cohort["gate"].BODY + "/LICENSE").read() == b"original license/card/config notice"
        assert cohort["gate"].KIT_NAMESPACE + "/__init__.py" not in names
    result = contract.verify_archive(one, receipt["archive_identity"], receipt["manifest_identity"], cohort["code"], cohort["helpers"])
    assert result["status"] == "pass" and result["extraction_performed"] is False


def test_archive_is_private_before_first_byte_even_with_permissive_umask(contract, cohort, tmp_path, monkeypatch):
    destination = tmp_path / "private.tar"
    original = contract.tarfile.open
    observed = []
    def checked_open(*args, **kwargs):
        if kwargs.get("mode") == "w|":
            assert destination.stat().st_size == 0
            observed.append(destination.stat().st_mode & 0o777)
        return original(*args, **kwargs)
    monkeypatch.setattr(contract.tarfile, "open", checked_open)
    before = os.umask(0)
    try:
        cohort["build"](destination)
    finally:
        os.umask(before)
    assert observed == [0o400]


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../b", "a//b", "a/./b", "a\\b", ".git/config", "weights/.secrets/x", "a/__pycache__/x", ""])
def test_paths_fail_closed(contract, name):
    with pytest.raises(ValueError): contract._name(name)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'])
def test_strict_bounded_json(contract, raw):
    with pytest.raises(ValueError): contract._parse(raw)


def test_report_pin_checked_before_parsing_and_source_before_import(contract, cohort, tmp_path, monkeypatch):
    original = contract._parse
    monkeypatch.setattr(contract, "_parse", lambda raw: pytest.fail("untrusted report parsed"))
    cohort["path"].chmod(0o600); cohort["path"].write_bytes(b"not original JSON"); cohort["path"].chmod(0o400)
    with pytest.raises(ValueError, match="identity differs"): cohort["build"](tmp_path / "never.tar")
    monkeypatch.setattr(contract, "_parse", original)
    source = cohort["code"] / contract.INVENTORY_SOURCES[0]; source.chmod(0o600); source.write_text('raise AssertionError("untrusted code executed")'); source.chmod(0o400)
    with pytest.raises(ValueError, match="helper identity"): contract._source(cohort["code"], cohort["helpers"])


@pytest.mark.parametrize("fault", ["file", "link", "alias", "empty", "special", "parent_alias"])
def test_actual_inputs_no_fill_no_alias_no_nonregular(contract, cohort, tmp_path, fault):
    name = cohort["gate"].BODY + "/LICENSE"; path = cohort["root"] / name
    if fault == "file": path.write_bytes(b"tampered")
    elif fault == "empty": path.write_bytes(b"")
    elif fault == "alias": os.link(path, tmp_path / "hardalias")
    elif fault == "special": path.unlink(); os.mkfifo(path)
    elif fault == "parent_alias":
        saved = path.parent.with_name("saved"); path.parent.rename(saved); path.parent.symlink_to(saved)
    else:
        path = cohort["root"] / (cohort["gate"].MOGE_SNAPSHOT + "/model.pt"); path.unlink(); path.symlink_to("/etc/passwd")
    with pytest.raises(ValueError): cohort["build"](tmp_path / "never.tar")
    assert not (tmp_path / "never.tar").exists()


@pytest.mark.parametrize("name", ["results/extra.json", "results/image-cari4d-source.json", "validation/test.py", "data/test.py", "vendor/foreign/x.py", "weights/cache/file", ".env"])
def test_pinned_report_cannot_expand_export_whitelist(contract, cohort, tmp_path, name):
    cohort["entries"][name] = dict(type="file", **pin(b"unknown", True), role="asset")
    cohort["inventory_pin"].update(cohort["freeze_report"]())
    with pytest.raises(ValueError, match="not export eligible"): cohort["build"](tmp_path / "never.tar")


@pytest.mark.parametrize("fault", ["unknown", "secret", "binding", "reg4", "duplicate", "license"])
def test_exact_two_receipts_preserve_semantics_no_secret_env(contract, cohort, tmp_path, fault):
    name = cohort["gate"].RECEIPTS[1 if fault == "reg4" else 0]; path = cohort["root"] / name; value = json.loads(path.read_bytes())
    if fault == "unknown": value["Config"] = {"Env": ["innocent-looking"]}
    elif fault == "secret": value["api_token"] = "never printed"
    elif fault == "binding": value["assets"][0]["path"] += "/other"
    elif fault == "reg4": value["checkpoints"][-1]["sha256"] = "0" * 64
    elif fault == "license": value["mhr_license"] = "claimed sublicense"
    raw = json.dumps(value).encode() if fault != "duplicate" else b'{"assets":[],"assets":[]}'
    path.write_bytes(raw); cohort["entries"][name].update(pin(raw, True)); cohort["inventory_pin"].update(cohort["freeze_report"]())
    with pytest.raises(ValueError): cohort["build"](tmp_path / "never.tar")


@pytest.mark.parametrize("fault", ["cycle", "foreign", "missing", "fourth", "noninit_empty"])
def test_audited_graph_and_empty_source_contract(contract, cohort, tmp_path, fault):
    entries = cohort["entries"]; link = cohort["gate"].MOGE_SNAPSHOT + "/model.pt"
    if fault == "cycle": entries[link]["link"] = "model.pt"; entries[link]["target"] = link
    elif fault == "foreign": entries[link]["link"] = "../../../../../foreign/model"; entries[link]["target"] = "foreign/model"
    elif fault == "missing": del entries[entries[link]["target"]]
    elif fault == "fourth": entries[cohort["gate"].HF + "/blobs/ab/" + "d" * 64] = dict(type="symlink", link="x", target="x", role="internal_moge1")
    else:
        name = "vendor/video_to_data/" + cohort["gate"].CARI + "/lib_mhr/contact.py"; entries[name].update(pin(b"", True))
        source = cohort["report"]["sources"][0]; rows = {n: r for n, r in entries.items() if n.startswith(source["path"] + "/") and r["role"] == "public_source"}; source["source_sha256"] = cohort["gate"].digest_json(rows)
    cohort["inventory_pin"].update(cohort["freeze_report"]())
    with pytest.raises(ValueError): cohort["build"](tmp_path / "never.tar")


def test_owned_failed_stream_removed_and_existing_never_overwritten(contract, cohort, tmp_path, monkeypatch):
    existing = tmp_path / "existing.tar"; existing.write_bytes(b"user work")
    with pytest.raises(FileExistsError): cohort["build"](existing)
    assert existing.read_bytes() == b"user work"
    native = contract._Reader.finish
    def fail(self, row): native(self, row); raise ValueError("procedural streaming failure")
    monkeypatch.setattr(contract._Reader, "finish", fail)
    failed = tmp_path / "failed.tar"
    with pytest.raises(ValueError, match="streaming failure"): cohort["build"](failed)
    assert not failed.exists() and existing.read_bytes() == b"user work"


@pytest.mark.parametrize("fault", ["file", "link", "source", "report"])
def test_final_recheck_catches_stream_time_input_changes(contract, cohort, tmp_path, monkeypatch, fault):
    native = contract._Reader.finish; once = []
    def mutate(self, row):
        native(self, row)
        if once: return
        once.append(True)
        if fault == "file": (cohort["root"] / (cohort["gate"].BODY + "/LICENSE")).write_bytes(b"changed during stream")
        elif fault == "link":
            link = cohort["root"] / (cohort["gate"].MOGE_SNAPSHOT + "/model.pt"); link.unlink(); link.symlink_to("/foreign")
        else:
            path = cohort["code"] / contract.INVENTORY_SOURCES[1] if fault == "source" else cohort["path"]
            path.chmod(0o600); path.write_bytes(b"changed during stream"); path.chmod(0o400)
    monkeypatch.setattr(contract._Reader, "finish", mutate)
    output = tmp_path / "owned-failure.tar"
    with pytest.raises(ValueError): cohort["build"](output)
    assert not output.exists()


def rebuild(contract, original, mutate):
    with tarfile.open(fileobj=io.BytesIO(original)) as archive:
        members = [(m, archive.extractfile(m).read() if m.isreg() else None) for m in archive.getmembers()]
    mutate(members); output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member, raw in members: archive.addfile(member, io.BytesIO(raw) if raw is not None else None)
    return output.getvalue()


@pytest.mark.parametrize("fault", ["file", "duplicate", "traversal", "hardlink", "device", "foreign", "link", "header", "trailer", "padding"])
def test_verify_stream_rejects_tamper_even_with_new_transport_pin(contract, cohort, tmp_path, fault):
    path = tmp_path / "valid.tar"; receipt = cohort["build"](path); original = path.read_bytes()
    def mutate(members):
        index = next(i for i, (m, raw) in enumerate(members) if i and m.isreg() and raw)
        member, raw = members[index]
        if fault == "file": members[index] = member, bytes([raw[0] ^ 1]) + raw[1:]
        elif fault == "duplicate": members.insert(index, members[index])
        elif fault == "traversal": member.name = "../escape"
        elif fault == "hardlink": member.type = tarfile.LNKTYPE; member.size = 0; member.linkname = "outside"; members[index] = member, None
        elif fault == "device": member.type = tarfile.CHRTYPE; member.size = 0; members[index] = member, None
        elif fault == "foreign": member.name = "results/extra.json"
        elif fault == "link": next(m for m, _ in members if m.issym()).linkname = "/etc/passwd"
        elif fault == "header": member.mtime = 1
    raw = rebuild(contract, original, mutate)
    if fault == "trailer": raw = raw[:-1] + b"X"
    elif fault == "padding":
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive: m = next(m for m in archive if m.isreg() and m.size % 512)
        offset = m.offset_data + m.size; raw = raw[:offset] + b"X" + raw[offset + 1:]
    tampered = tmp_path / "tampered.tar"; tampered.write_bytes(raw)
    with pytest.raises(ValueError): contract.verify_archive(tampered, pin(raw), receipt["manifest_identity"], cohort["code"], cohort["helpers"])


def test_verify_original_archive_and_manifest_pins_required(contract, cohort, tmp_path):
    path = tmp_path / "archive.tar"; receipt = cohort["build"](path)
    with pytest.raises(ValueError, match="archive identity"): contract.verify_archive(path, {**receipt["archive_identity"], "sha256": "0" * 64}, receipt["manifest_identity"], cohort["code"], cohort["helpers"])
    with pytest.raises(ValueError, match="manifest identity"): contract.verify_archive(path, receipt["archive_identity"], {**receipt["manifest_identity"], "sha256": "0" * 64}, cohort["code"], cohort["helpers"])


def test_manifest_cannot_self_claim_readiness_or_wrong_inventory_counts(contract, cohort, tmp_path):
    original = tmp_path / "valid.tar"; receipt = cohort["build"](original)
    def mutate(members):
        member, raw = members[0]; manifest = json.loads(raw); manifest["inventory"]["files"] += 1
        raw = contract._json(manifest); member.size = len(raw); members[0] = member, raw
    raw = rebuild(contract, original.read_bytes(), mutate); changed = tmp_path / "wrong-cohort.tar"; changed.write_bytes(raw)
    with tarfile.open(changed) as archive: manifest_pin = pin(archive.extractfile(contract.MANIFEST).read())
    with pytest.raises(ValueError, match="quantities"): contract.verify_archive(changed, pin(raw), manifest_pin, cohort["code"], cohort["helpers"])


def test_no_network_gpu_git_or_image_runtime(contract):
    source = Path(contract.__file__).read_text()
    assert all(word not in source for word in ("import torch", "import numpy", "subprocess", "urllib", "extractall", "docker save", "docker load", "os.environ"))

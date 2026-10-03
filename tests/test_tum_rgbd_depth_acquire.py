"""Artificial tiny TAR/HTTP metadata only; no external media/label downloads."""
from decimal import Decimal
import hashlib
import importlib
import io
import json
from pathlib import Path
import struct
import tarfile
import types

import pytest


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "infra"))
    return importlib.import_module("tum_rgbd_depth_acquire")


def png(kind):
    return b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + struct.pack(">IIBBBBB", 640, 480, 8 if kind == "rgb" else 16, 2 if kind == "rgb" else 0, 0, 0, 0) + b"tiny-header-only-no-pixel-values"


def fixture_sequence(gate, scene=1):
    name = "rgbd_dataset_freiburg" + str(scene) + "_fixture"
    names = [f"{name}/{kind}/1000.{i * 1000:06d}.png" for kind in ("rgb", "depth") for i in range(5)]
    sequence = dict(name=name, sequence_id=scene, rgb_file_count=5, depth_file_count=5,
        archive=dict(bytes=100000), selected=[])
    for rank in (0, 1, 2, 3):
        stamp = f"1000.{rank * 1000:06d}"
        sequence["selected"].append(dict(rgb_zero_based_rank=rank, rgb_timestamp=stamp, depth_timestamp=stamp,
            timestamp_offset_seconds="0.000000", **{kind: dict(path=f"{kind}/{stamp}.png", **gate.bytes_identity(png(kind))) for kind in ("rgb", "depth")}))
    return sequence, names


def make_tar(path, rows):
    with tarfile.open(path, "w:gz") as archive:
        for name, raw, kind in rows:
            member = tarfile.TarInfo(name); member.size = len(raw)
            if kind == "directory": member.type = tarfile.DIRTYPE; member.size = 0
            elif kind == "symlink": member.type = tarfile.SYMTYPE; member.linkname = "other"; member.size = 0
            archive.addfile(member, io.BytesIO(raw) if member.isfile() else None)
    path.chmod(0o400)


def test_frozen_protocol_and_description_source_shapes(gate):
    root = Path(__file__).resolve().parents[1]
    assert gate.identity(root / gate.PROTOCOL) == gate.PROTOCOL_ID
    assert gate.BUDGET == 600 and len(gate.SOURCE_FILES) == 2
    source = (root / gate.SOURCE_FILES[0]).read_text()
    assert all(term not in source for term in ("import numpy", "from PIL", "import torch", "groundtruth.txt", "calibration.txt"))
    wrapper = (root / gate.SOURCE_FILES[1]).read_text()
    assert "603s runuser" in wrapper and "source_identity" in wrapper and "--kill-after=10s" in wrapper
    assert "docker run" not in wrapper and "--gpus" not in wrapper


@pytest.mark.parametrize("fault", [None, "drift", "missing"])
def test_primary_license_exact_section_not_whole_mutable_page(gate, fault):
    section = '<h2 id="license">License</h2>\n<p>CC BY 4.0</p>\n'
    raw = ("unrelated changing prefix" + section + "<h2>Next</h2>changing bibliography").encode()
    expected = gate.bytes_identity(section.encode())
    if fault == "drift": raw = raw.replace(b"CC BY 4.0", b"NC license")
    elif fault == "missing": raw = b"no license heading"
    if fault:
        with pytest.raises(ValueError): gate.license_section(raw, expected)
    else: assert gate.license_section(raw, expected) == section.encode()


@pytest.mark.parametrize("fault", [None, "description", "archive_link"])
def test_sequence_publisher_description_pinned_before_archive_download(gate, monkeypatch, fault):
    sequence, _ = fixture_sequence(gate)
    name = sequence["name"].removeprefix("rgbd_dataset_")
    section = f"<a name='{name}'>Sequence '{name}'</a></b><br>\n<i>original office description</i>"
    sequence["archive"]["url"] = "https://cvg.cit.tum.de/original.tgz"
    monkeypatch.setitem(gate.DESCRIPTION_IDS, sequence["name"], gate.bytes_identity(section.encode()))
    raw = (section + "<a href='/original.tgz'>tgz</a>").encode()
    if fault == "description": raw = raw.replace(b"original office", b"new restrictions")
    elif fault == "archive_link": raw = raw.replace(b"/original.tgz", b"/different.tgz")
    if fault:
        with pytest.raises(ValueError): gate.sequence_description(raw, sequence)
    else: assert gate.sequence_description(raw, sequence) == section.encode()


@pytest.mark.parametrize("fault", [None, "count", "rank", "offset", "tie", "reuse"])
def test_filename_only_exact_nearest_decimal_association(gate, fault):
    sequence, names = fixture_sequence(gate)
    if fault == "count": names.pop()
    elif fault == "rank": sequence["selected"][0]["rgb_zero_based_rank"] = 1
    elif fault == "offset": sequence["selected"][0]["timestamp_offset_seconds"] = ".020001"
    elif fault == "tie":
        names.remove(sequence["name"] + "/depth/1000.000000.png")
        names += [sequence["name"] + "/depth/999.999500.png", sequence["name"] + "/depth/1000.000500.png"]
        sequence["depth_file_count"] += 1
    elif fault == "reuse": sequence["selected"][1] = dict(sequence["selected"][0])
    if fault:
        with pytest.raises(ValueError): gate.select_pairs(names, sequence)
    else:
        pairs, inventories = gate.select_pairs(names, sequence)
        assert len(pairs) == 4 and inventories["rgb"]["files"] == 5
        assert all(PurePath(r).stem == PurePath(d).stem for r, d in pairs)


PurePath = Path


@pytest.mark.parametrize("fault", [None, "sha", "bytes", "symlink", "duplicate", "traversal", "expanded", "png", "terms"])
def test_streaming_tar_selected_bytes_and_metadata_only_no_GT_reads(gate, tmp_path, monkeypatch, fault):
    sequence, names = fixture_sequence(gate); rows = []
    for name in names: rows.append((name, png("rgb" if "/rgb/" in name else "depth"), "file"))
    rows += [(sequence["name"] + "/groundtruth.txt", b"DO_NOT_OPEN_TRAJECTORY_VALUES", "file")]
    if fault == "sha": sequence["selected"][0]["rgb"]["sha256"] = "0" * 64
    elif fault == "bytes": sequence["selected"][0]["rgb"]["bytes"] += 1
    elif fault == "symlink": rows.append((sequence["name"] + "/alias", b"", "symlink"))
    elif fault == "duplicate": rows.append(rows[0])
    elif fault == "traversal": rows.append((sequence["name"] + "/../escape", b"x", "file"))
    elif fault == "expanded": monkeypatch.setattr(gate, "MAX_MEMBERS", 2)
    elif fault == "png":
        name, raw, kind = rows[0]; raw = b"NOT_PNG"; rows[0] = name, raw, kind
        sequence["selected"][0]["rgb"].update(gate.bytes_identity(raw))
    elif fault == "terms": rows.append((sequence["name"] + "/LICENSE", b"Research only non-commercial", "file"))
    archive = tmp_path / "tiny.tgz"; make_tar(archive, rows); staging = tmp_path / "staging"; staging.mkdir()
    native = tarfile.TarFile.extractfile; opened = []
    def extract(instance, member, *args, **kwargs):
        opened.append(member.name)
        assert not member.name.endswith("groundtruth.txt"), "Unknown/private trajectory body must never be selected"
        return native(instance, member, *args, **kwargs)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", extract)
    if fault:
        with pytest.raises(ValueError): gate.scan_archive(archive, sequence, staging)
    else:
        retained, checks, terms = gate.scan_archive(archive, sequence, staging)
        assert len(retained) == len(opened) == 8 and checks["nearest_timestamp_pairs_verified"]
        assert checks["archive_license_files_present"] is False and terms == []
        assert {p.name for p in staging.iterdir()} == {r["file"] for r in retained}


@pytest.mark.parametrize("name", ["/absolute", "seq/../bad", "seq//bad", "seq/./bad", "seq\\bad", "other/file"])
def test_unsafe_tar_names_fail_without_extract(gate, name):
    member = tarfile.TarInfo(name); member.size = 1
    with pytest.raises(ValueError): gate.member_name(member, "seq")


def test_harmless_single_dot_prefix_and_directory_suffix_normalization(gate):
    member = tarfile.TarInfo("./seq/rgb/"); member.type = tarfile.DIRTYPE
    assert gate.member_name(member, "seq") == "seq/rgb"


@pytest.mark.parametrize("terms", [b"CC-BY-4.0 https://creativecommons.org/licenses/by/4.0/", b"BSD-2-Clause"])
def test_compatible_embedded_terms_if_present(gate, terms): gate.archive_terms(terms)


@pytest.mark.parametrize("terms", [b"unknown custom terms", b"CC-BY-NC-4.0", b"research only", b"CC-BY-SA-4.0"])
def test_unknown_or_contrary_embedded_terms_stop_no_waiver(gate, terms):
    with pytest.raises(ValueError): gate.archive_terms(terms)


@pytest.mark.parametrize("fault", [None, "headers", "truncated", "oversize", "redirect"])
def test_original_download_headers_and_first_observed_digest_not_fake_pin(gate, tmp_path, monkeypatch, fault):
    raw = b"tiny original archive bytes"
    record = dict(url="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz", bytes=len(raw), etag='"original"', last_modified="old")
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(raw)), "ETag": '"original"', "Last-Modified": "old"}
        def geturl(self): return record["url"] if fault != "redirect" else "https://other.invalid/"
    response = Response(raw[:-1] if fault == "truncated" else raw + b"x" if fault == "oversize" else raw)
    if fault == "headers": response.headers = {**response.headers, "ETag": "different"}
    monkeypatch.setattr(gate, "urlopen", lambda *_args, **_kwargs: response)
    if fault:
        with pytest.raises(ValueError): gate.download_archive(record, tmp_path / "download.tgz")
    else:
        observed = gate.download_archive(record, tmp_path / "download.tgz")
        assert observed == gate.bytes_identity(raw) and "sha256" not in record


def test_failed_cleanup_reserved_only_receipt_and_terms_survive(gate, tmp_path):
    sequence, _ = fixture_sequence(gate); frozen = dict(sequences=[sequence])
    public, private = tmp_path / "inputs", tmp_path / "eval_private"; public.mkdir(); private.mkdir(); (private / "source").mkdir()
    receipt = private / "acquisition-report.json"; receipt.write_bytes(b"original failure")
    (public / "manifest.json").write_bytes(b"partial manifest")
    name = gate.output_name(1, "rgb", sequence["selected"][0]["rgb_timestamp"]); (public / name).write_bytes(b"partial RGB")
    assert gate.cleanup_failure(tmp_path, frozen) == 2
    assert receipt.read_bytes() == b"original failure" and not any(public.iterdir())
    (public / "unknown.png").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="preserved"): gate.cleanup_failure(tmp_path, frozen)
    assert (public / "unknown.png").exists()


def test_actual_archive_closure_only_two_acquisition_helpers(gate, monkeypatch):
    import azure_job
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): p.read_bytes() for folder in ("infra", "src", "configs") for p in (root / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".json", ".toml", ".cpp")}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, gate.SOURCE_FILES[1])
    assert {name for name in selected if name.startswith("infra/")} == set(gate.SOURCE_FILES)
    assert not any("infer" in name or "render" in name or "evaluate" in name for name in selected if name.startswith("infra/"))


@pytest.mark.parametrize("failure", [None, "byte_pin"])
def test_complete_acquisition_control_chain_public_manifest_last_and_no_sensor_values(gate, tmp_path, monkeypatch, failure):
    sequences = [fixture_sequence(gate, scene)[0] for scene in (1, 2, 3)]
    originals = tmp_path / "originals"; originals.mkdir(); destination = tmp_path / "output"; destination.mkdir()
    frozen = dict(sequences=sequences); public, private = destination / "inputs", destination / "eval_private"
    public.mkdir(mode=0o755); private.mkdir(mode=0o700); (private / "acquisition-report.json").write_bytes(b"mock only")
    def licenses(folder, frozen):
        (folder / "source").mkdir(mode=0o700); gate.save(folder / "source/license.txt", b"CC-BY-4.0 fixture")
        return [dict(file="source/license.txt", **gate.bytes_identity(b"CC-BY-4.0 fixture"))]
    monkeypatch.setattr(gate, "license_evidence", licenses)
    paths = {}
    for sequence in sequences:
        seq, names = fixture_sequence(gate, sequence["sequence_id"])
        path = originals / (sequence["name"] + ".tgz")
        make_tar(path, [(name, png("rgb" if "/rgb/" in name else "depth"), "file") for name in names])
        sequence["archive"].update(url="mockoriginal:" + sequence["name"], bytes=path.stat().st_size)
        paths[sequence["archive"]["url"]] = path
    def download(record, destination):
        assert not (public / "manifest.json").exists()
        gate.save(destination, paths[record["url"]].read_bytes()); return gate.identity(destination)
    monkeypatch.setattr(gate, "download_archive", download)
    if failure: sequences[1]["selected"][0]["depth"]["sha256"] = "0" * 64
    report = dict(archives=[]); seen = []
    def persist():
        if (public / "manifest.json").exists():
            assert report["all24_retained_bytes_rehashed_before_public_manifest"] is True
        seen.append(dict(report))
    if failure:
        with pytest.raises(ValueError, match="LFS pin"): gate.acquire(destination, frozen, report, persist)
        assert not (private / ".staging").exists() and not (public / "manifest.json").exists()
        assert not any(public.iterdir()) and report["disposable_archives_and_staging_removed"] is True
        return
    inventory = gate.acquire(destination, frozen, report, persist)
    assert len(inventory) == 26 and report["depth_values_decoded"] is False
    assert report["all24_original_files_hashed_before_depth_values"] is True
    assert not (private / ".staging").exists() and len(report["selected_records"]) == 24
    manifest = json.loads((public / "manifest.json").read_bytes())
    assert manifest["schema"] == gate.SCHEMA and len(manifest["images"]) == 12
    assert all(set(row) == {"scene_id", "frame_id", "file", "original_rgb_file", "timestamp", "sha256", "width", "height"} for row in manifest["images"])
    assert gate.output_inventory(destination, frozen) == inventory
    assert all(row["sha256_independently_preknown"] is False for row in report["archives"])


@pytest.mark.parametrize("failure", ["timeout", "source_recheck"])
def test_main_original_failure_receipt400_signals_restored_no_partial_media(gate, tmp_path, monkeypatch, failure):
    sequences = [fixture_sequence(gate, scene)[0] for scene in (1, 2, 3)]
    frozen = dict(sequences=sequences); output = tmp_path / gate.BASE; output.mkdir(parents=True)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    for name, value in dict(WR_ROOT=str(tmp_path), WR_CODE=str(tmp_path / "code"), WR_CODE_REVISION="a" * 40, WR_TUM_OUTPUT_RESERVED="1").items():
        monkeypatch.setenv(name, value)
    helpers = {gate.SOURCE_FILES[0]: dict(bytes=1, sha256="a" * 64)}; binds = []
    def binding(*_):
        binds.append(1)
        if failure == "source_recheck" and len(binds) > 1: raise ValueError("source bytes changed")
        return helpers, gate.PROTOCOL_ID, frozen
    monkeypatch.setattr(gate, "bound_source", binding)
    handlers = {gate.signal.SIGALRM: object(), gate.signal.SIGTERM: object()}; before = handlers.copy(); alarms = []
    def signal(number, handler): previous = handlers[number]; handlers[number] = handler; return previous
    monkeypatch.setattr(gate.signal, "signal", signal); monkeypatch.setattr(gate.signal, "alarm", alarms.append)
    def acquire(destination, frozen, report, persist):
        (destination / "eval_private/source").mkdir(mode=0o700)
        gate.save(destination / "inputs" / gate.output_name(1, "rgb", sequences[0]["selected"][0]["rgb_timestamp"]), b"partial data", 0o444)
        if failure == "timeout": handlers[gate.signal.SIGALRM]()
        return {}
    monkeypatch.setattr(gate, "acquire", acquire); monkeypatch.setattr(gate, "output_inventory", lambda *_: {})
    with pytest.raises((ValueError, TimeoutError)): gate.main([])
    assert handlers == before and alarms == [600, 0]
    receipt = output / "eval_private/acquisition-report.json"; report = json.loads(receipt.read_bytes())
    assert receipt.stat().st_mode & 0o777 == 0o400 and report["status"] == "fail"
    assert report["depth_values_decoded"] is False and report["source_camera_or_trajectory_read"] is False
    assert report["failed_partial_media_removed"] == 1 and not any((output / "inputs").iterdir())

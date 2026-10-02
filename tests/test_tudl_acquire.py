"""Tiny fake ZIP/HTTP contracts only; no external images, models or GPU."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
import zipfile
import zlib

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/tudl_acquire.py"
    spec = importlib.util.spec_from_file_location("wr_tudl_acquire_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def rgb_png(width=640, height=480, color=2):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n"+struct.pack(">I", 13)+b"IHDR"+ihdr+struct.pack(">I", zlib.crc32(b"IHDR"+ihdr))


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items(): archive.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize("fault", ["traversal", "absolute", "backslash", "duplicate", "symlink", "special", "nonregular", "directory_type", "large", "aggregate", "files"])
def test_zip_inventory_fail_closed_every_member_even_discarded(gate, fault):
    item = SimpleNamespace(filename="test/000001/rgb/000000.png", external_attr=stat.S_IFREG << 16,
        file_size=20, flag_bits=0, compress_type=zipfile.ZIP_DEFLATED, is_dir=lambda: False)
    items = [item]; budget = [0, 0]
    if fault == "traversal": item.filename = "test/../escape"
    elif fault == "absolute": item.filename = "/escape"
    elif fault == "backslash": item.filename = "test\\escape"
    elif fault == "duplicate": items.append(item)
    elif fault == "symlink": item.external_attr = stat.S_IFLNK << 16
    elif fault == "special": item.external_attr = stat.S_IFIFO << 16
    elif fault == "nonregular": item.external_attr = stat.S_IFDIR << 16
    elif fault == "directory_type": item.filename += "/"; item.is_dir = lambda: True
    elif fault == "large": item.file_size = 128_000_001
    elif fault == "aggregate": budget[0] = gate.MAX_EXPANDED
    elif fault == "files": budget[1] = gate.MAX_FILES
    with pytest.raises(ValueError): gate.zip_inventory(SimpleNamespace(infolist=lambda: items), budget)


def test_zip_safety_budget_shared_across_archives(gate):
    with zipfile.ZipFile(io.BytesIO(zip_bytes({"models/obj_000001.ply": b"abc"}))) as archive:
        budget = [8, 2]; inventory = gate.zip_inventory(archive, budget)
    assert budget == [11, 3] and set(inventory) == {"models/obj_000001.ply"}


def test_pinned_layout_unknown_or_license_member_fails(gate):
    gate.inspect_layout("tudl_base.zip", {x: None for x in ("tudl/camera.json", "tudl/dataset_info.md", "tudl/test_targets_bop19.json")})
    models = {f"{folder}/{file}": None for folder in ("models", "models_eval") for file in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")}
    gate.inspect_layout("tudl_models.zip", models)
    with pytest.raises(ValueError): gate.inspect_layout("tudl_models.zip", models | {"LICENSE-NC": None})
    with pytest.raises(ValueError): gate.inspect_layout("tudl_test_bop19.zip", {"test/000004/rgb/000000.png": None})


def test_filename_selection_no_annotation_values_or_order_dependence(gate):
    names = {}
    for scene, (first, median, last) in gate.EXPECTED_IDS.items():
        ids = [first, *range(first+1, first+100), median, *range(median+1, median+99), last]
        assert len(ids) == 200
        for frame in reversed(ids): names[f"test/{scene:06d}/rgb/{frame:06d}.png"] = "never_read_gt"
    selected = gate.select_rgb_names(names)
    assert [(scene, frame) for scene, frame, _ in selected] == [(s, f) for s, ids in gate.EXPECTED_IDS.items() for f in ids]
    with pytest.raises(ValueError): gate.select_rgb_names({k: v for i, (k, v) in enumerate(names.items()) if i})


@pytest.mark.parametrize("fault", ["size", "rgba", "crc", "header"])
def test_png_public_native_dimensions_no_reencoding(gate, fault):
    data = rgb_png()
    assert gate.png_dimensions(data) == (640, 480)
    if fault == "size": data = rgb_png(width=512)
    elif fault == "rgba": data = rgb_png(color=6)
    elif fault == "crc": data = data[:-1]+bytes([data[-1]^1])
    else: data = b"notpng"
    with pytest.raises(ValueError): gate.png_dimensions(data)


def fixtures(gate):
    base = {"tudl/camera.json": b'{"cam_K":"private"}', "tudl/dataset_info.md": b"Hodan et al. BOP attribution",
        "tudl/test_targets_bop19.json": b'["private labels"]'}
    models = {f"{folder}/{file}": b"private mesh" for folder in ("models", "models_eval") for file in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")}
    files = {}; selected = []
    for scene, frames in gate.EXPECTED_IDS.items():
        for kind in ("camera", "gt", "gt_info"):
            values = {str(f): [{"obj_id": scene}, {"obj_id": scene}] if kind != "camera" else {"cam_K": ["private"], "depth_scale": 1.} for f in [*frames, 99999]}
            files[f"test/{scene:06d}/scene_{kind}.json"] = json.dumps(values).encode()
        for frame in frames:
            prefix = f"test/{scene:06d}/"; rgb = prefix+f"rgb/{frame:06d}.png"
            files[rgb] = rgb_png(); selected.append((scene, frame, rgb))
            files[prefix+f"depth/{frame:06d}.png"] = b"private depth"
            for i in range(2): files[prefix+f"mask_visib/{frame:06d}_{i:06d}.png"] = b"private mask"
    archives = {}
    for name, values in (("tudl_base.zip", base), ("tudl_models.zip", models), ("tudl_test_bop19.zip", files)):
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes(values))); archives[name] = (archive, {name: None for name in values})
    return archives, selected


def test_retained_private_all_instances_and_public_rgb_only(gate, tmp_path):
    private, inputs = tmp_path/"eval_private", tmp_path/"inputs"; private.mkdir(mode=0o700); inputs.mkdir()
    archives, selected = fixtures(gate)
    try: manifest = gate.retain_subset(archives, private, inputs, selected)
    finally:
        for archive, _ in archives.values(): archive.close()
    assert set(manifest) == {"schema", "revision", "license", "selection", "images"}
    assert manifest["selection"] == gate.SELECTION and manifest["license"] == "CC-BY-SA-4.0"
    assert len(manifest["images"]) == 9 and len(list(inputs.iterdir())) == 9
    for record in manifest["images"]:
        assert set(record) == {"scene_id", "frame_id", "file", "sha256", "width", "height"}
        assert (inputs/record["file"]).read_bytes() == rgb_png() and (inputs/record["file"]).stat().st_mode & 0o777 == 0o444
    for scene, frames in gate.EXPECTED_IDS.items():
        folder = private/f"source/test/{scene:06d}"
        gt = json.loads((folder/"scene_gt.json").read_text())
        assert set(gt) == {str(f) for f in frames} and all(len(v) == 2 for v in gt.values())
        assert len(list((folder/"mask_visib").iterdir())) == 6 and len(list((folder/"depth").iterdir())) == 3
        assert (folder/"scene_camera.json").stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError): gate.save_bytes(inputs/manifest["images"][0]["file"], b"no overwrite")


def test_missing_instance_mask_never_ignored(gate, tmp_path):
    private, inputs = tmp_path/"eval_private", tmp_path/"inputs"; private.mkdir(); inputs.mkdir()
    archives, selected = fixtures(gate); del archives["tudl_test_bop19.zip"][1]["test/000001/mask_visib/000000_000001.png"]
    try:
        with pytest.raises(ValueError, match="All selected instance"):
            gate.retain_subset(archives, private, inputs, selected)
    finally:
        for archive, _ in archives.values(): archive.close()


def response(gate, monkeypatch, data, url="https://huggingface.co/pinned"):
    class Response(io.BytesIO):
        def geturl(self): return url
    monkeypatch.setattr(gate.urllib.request, "urlopen", lambda *a, **k: Response(data))


@pytest.mark.parametrize("fault", [None, "sha", "short", "large", "http", "host"])
def test_stream_download_exact_size_hash_allowlisted_https(gate, tmp_path, monkeypatch, fault):
    data = b"tiny archive"; size = len(data); sha = hashlib.sha256(data).hexdigest(); url = "https://huggingface.co/pinned"
    if fault == "sha": sha = "0"*64
    elif fault == "short": data = data[:-1]
    elif fault == "large": data += b"!"
    elif fault == "http": url = "http://huggingface.co/pinned"
    elif fault == "host": url = "https://evil.invalid/archive"
    response(gate, monkeypatch, data, url)
    if fault:
        with pytest.raises(ValueError): gate.download("https://huggingface.co/pinned", tmp_path/"archive.zip", size, sha)
    else:
        gate.download(url, tmp_path/"archive.zip", size, sha)
        assert (tmp_path/"archive.zip").read_bytes() == data and (tmp_path/"archive.zip").stat().st_mode & 0o777 == 0o400


def test_acquisition_error_preserves_evidence_cleans_disposable_no_overwrite(gate, tmp_path, monkeypatch):
    destination = tmp_path/"validation/tudl_rgb_v1"; destination.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_TUDL_OUTPUT_RESERVED", "1")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    def evidence(private):
        gate.save_bytes(private/"source/licenses/evidence.txt", b"verified evidence"); return [{"sha256": "a"*64}]
    def fail(url, path, size, sha): path.write_bytes(b"partial"); raise ValueError("hash mismatch")
    monkeypatch.setattr(gate, "license_evidence", evidence); monkeypatch.setattr(gate, "download", fail)
    with pytest.raises(ValueError, match="hash mismatch"): gate.main([])
    report_path = destination/"eval_private/acquisition-report.json"; frozen = report_path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["disposable_archives_removed"] is True and report["inference_performed"] is False
    assert not (destination/"eval_private/.downloads").exists() and (destination/"eval_private/source/licenses/evidence.txt").read_bytes() == b"verified evidence"
    assert destination.joinpath("eval_private").stat().st_mode & 0o777 == 0o700
    assert destination.joinpath("eval_private/source").stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="exclusively reserved"): gate.main([])
    assert report_path.read_bytes() == frozen


def test_public_license_sources_pin_and_fail_conflict(gate, tmp_path, monkeypatch):
    readme = b"---\r\nlicense: cc-by-sa-4.0\r\n---\r\n"; section = b'<h3 id="TUD-L">CC BY-SA 4.0'
    monkeypatch.setattr(gate, "BOP_SECTION_SHA", hashlib.sha256(section).hexdigest())
    monkeypatch.setattr(gate, "fetch", lambda url, limit: readme if url == gate.README_URL else section+b'<div class="download_links">')
    assert len(gate.license_evidence(tmp_path)) == 2
    monkeypatch.setattr(gate, "fetch", lambda *a: b"cc-by-nc")
    with pytest.raises(ValueError): gate.license_evidence(tmp_path/"new")


def test_full_fake_acquisition_pass_receipts_and_cleanup(gate, tmp_path, monkeypatch):
    source, _ = fixtures(gate); blobs = {}
    for name, (archive, _) in source.items():
        values = {p: archive.read(p) for p in archive.namelist()}; archive.close()
        if name == "tudl_test_bop19.zip":
            for scene, (first, median, last) in gate.EXPECTED_IDS.items():
                ids = [first, *range(first+1, first+100), median, *range(median+1, median+99), last]
                for f in ids: values.setdefault(f"test/{scene:06d}/rgb/{f:06d}.png", rgb_png())
        blobs[name] = zip_bytes(values)
    monkeypatch.setattr(gate, "ARCHIVES", {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in blobs.items()})
    monkeypatch.setattr(gate, "download", lambda url, path, size, sha: path.write_bytes(blobs[path.name]))
    def evidence(private):
        gate.save_bytes(private/"source/licenses/evidence.txt", b"verified evidence"); return [{"sha256": "a"*64}]
    monkeypatch.setattr(gate, "license_evidence", evidence)
    destination = tmp_path/"validation/tudl_rgb_v1"; destination.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_TUDL_OUTPUT_RESERVED", "1")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); gate.main([])
    report = json.loads((destination/"eval_private/acquisition-report.json").read_text())
    assert report["status"] == "pass" and report["selection_before_private_annotation_values"] is True
    assert report["disposable_archives_removed"] is True and len(report["selected_records"]) == 9 and len(report["archives"]) == 3
    assert len(list((destination/"inputs").iterdir())) == 10 and not (destination/"eval_private/.downloads").exists()
    for record in report["retained_files"]:
        assert record["sha256"] == gate.digest(destination/"eval_private"/record["file"])
        assert record["file"].startswith("source/")
    for scene in gate.EXPECTED_IDS:
        assert len(list((destination/f"eval_private/source/test/{scene:06d}/mask_visib").iterdir())) == 6


def test_wrapper_scope_only_375mb_pinned_external_no_GPU_train_or_local(gate):
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_tudl_acquire.sh").read_text()
    assert sum(size for size, _ in gate.ARCHIVES.values()) == 374952356
    assert "603s runuser -u scenesmith" in wrapper and "ulimit -v 16777216" in wrapper and 'WR_TUDL_OUTPUT_RESERVED=1' in wrapper
    assert "--gpus" not in wrapper and "docker run" not in wrapper and "chown -R" not in wrapper
    assert "tudl_train" not in source and "tudl_test_all.zip" not in source
    assert "selection_before_private_annotation_values=True" in source
    with pytest.raises(SystemExit): gate.main(["--scene", "1"])

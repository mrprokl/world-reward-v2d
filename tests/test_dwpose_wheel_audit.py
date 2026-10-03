"""No real wheel/model/GPU/network: exact-case audit logic with tiny ZIP bytes."""
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("dwpose_wheel_audit", Path(__file__).parents[1] / "infra/dwpose_wheel_audit.py")
gate = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(gate)
old = gate.load_acquirer()


def wheel_raw(*, version="25.12.19", package="flatbuffers", requires=(), tags=("py2-none-any", "py3-none-any"), unsafe=None):
    buffer = io.BytesIO(); prefix = "flatbuffers-25.12.19.dist-info/"
    metadata = f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
    for row in requires: metadata += f"Requires-Dist: {row}\n"
    wheel = "Wheel-Version: 1.0\n" + "".join(f"Tag: {tag}\n" for tag in tags)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(prefix + "METADATA", metadata); archive.writestr(prefix + "WHEEL", wheel)
        for i in range(12):
            name, raw, mode = unsafe if unsafe and i == 0 else (f"flatbuffers/file_{i}.py", b"never execute\n", stat.S_IFREG | 0o644)
            info = zipfile.ZipInfo(name); info.external_attr = mode << 16; archive.writestr(info, raw)
    return buffer.getvalue()


def fixture(tmp_path, **kwargs):
    wheel = tmp_path / "flatbuffers-25.12.19-py2.py3-none-any.whl"; wheel.write_bytes(wheel_raw(**kwargs))
    license = tmp_path / "flatbuffers-LICENSE"; license.write_bytes(b"own primary license fixture")
    actual_digest = old.digest
    def digest(path):
        return "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" if Path(path) == license else actual_digest(path)
    # The fixture SHA is injected; production pins and negative guards stay fixed.
    acq = SimpleNamespace(ASSETS=[
        ("wheels/" + wheel.name, wheel.stat().st_size, actual_digest(wheel), "https://files.pythonhosted.org/frozen"),
        ("licenses/flatbuffers-LICENSE", 11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", "https://raw.githubusercontent.com/google/flatbuffers/pin/LICENSE")],
        FLATBUFFERS_REV=old.FLATBUFFERS_REV, digest=digest, safe_relative=old.safe_relative)
    # Pad the own text to the exact declared primary-license byte count (11KB only).
    license.write_bytes(b"own independent primary fixture\n".ljust(11358, b" "))
    return wheel, license, acq


def test_exact_pinned_missing_license_external_primary_retained(tmp_path):
    wheel, license, acq = fixture(tmp_path); before = wheel.read_bytes(), license.read_bytes()
    target = tmp_path / "fresh"
    row = gate.audit_flatbuffers(wheel, license, target, acq)
    assert row["embedded_license_present"] is False and row["external_primary_license_verified"] is True
    assert row["member_count"] == 14 and row["archive_expanded"] is False
    assert len(row["retained_texts"]) == 3 and not (target / "flatbuffers").exists()
    assert (target / "external-primary/flatbuffers-LICENSE").read_bytes() == license.read_bytes()
    assert before == (wheel.read_bytes(), license.read_bytes())
    for row in row["retained_texts"]: assert stat.S_IMODE((target / row["file"]).stat().st_mode) == 0o444


@pytest.mark.parametrize("field", ["sha", "size", "filename", "revision", "primary_sha", "primary_size", "missing_primary"])
def test_missing_license_exception_never_generic(tmp_path, field):
    wheel, license, acq = fixture(tmp_path)
    if field == "sha": acq.ASSETS[0] = (acq.ASSETS[0][0], acq.ASSETS[0][1], "0" * 64, acq.ASSETS[0][3])
    elif field == "size": acq.ASSETS[0] = (acq.ASSETS[0][0], 1, acq.ASSETS[0][2], acq.ASSETS[0][3])
    elif field == "filename": renamed = tmp_path / "other.whl"; wheel.rename(renamed); wheel = renamed
    elif field == "revision": acq.FLATBUFFERS_REV = "0" * 40
    elif field == "primary_sha": acq.digest = old.digest
    elif field == "primary_size": license.write_bytes(b"short")
    elif field == "missing_primary": license.unlink()
    with pytest.raises(ValueError): gate.audit_flatbuffers(wheel, license, tmp_path / "fresh", acq)
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("kwargs", [{"version": "0"}, {"package": "evil"}, {"requires": ["dependency"]}, {"tags": ["py3-cp311-any"]},
                                      {"unsafe": ("../evil", b"bad", stat.S_IFREG | 0o644)},
                                      {"unsafe": ("link", b"target", stat.S_IFLNK | 0o777)},
                                      {"unsafe": ("flatbuffers/LICENSE", b"new license", stat.S_IFREG | 0o644)}])
def test_flatbuffers_metadata_and_unsafe_zip_failclosed(tmp_path, kwargs):
    wheel, license, acq = fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError): gate.audit_flatbuffers(wheel, license, tmp_path / "fresh", acq)
    assert not (tmp_path / "fresh").exists()


def test_fresh_text_output_exclusive_no_old_rewrite(tmp_path):
    wheel, license, acq = fixture(tmp_path); target = tmp_path / "fresh"
    gate.audit_flatbuffers(wheel, license, target, acq)
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError): gate.audit_flatbuffers(wheel, license, target, acq)
    assert before == {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}


def failed_fixture(tmp_path):
    base = tmp_path / old.BASE; base.mkdir(parents=True)
    rows = []
    for name, size, sha, url in old.ASSETS:
        path = base / name; path.parent.mkdir(parents=True, exist_ok=True)
        # Bytes are tiny; identity wrapper below supplies reported sizes/digests.
        path.write_bytes(name.encode()); rows.append({"file": name, "bytes": size, "sha256": sha, "url": url})
    data = {"stage": "pinned_dwpose_native_cpu_assets_acquisition", "status": "fail", "phase": "download", "error_type": "ValueError",
            "producer_revision": gate.FAILED_REVISION, "script_sha256": gate.SOURCE_SHA,
            "source_revision": old.SOURCE_REV, "model_revision": old.MODEL_REV,
            "flatbuffers_source_revision": old.FLATBUFFERS_REV, "ort_source_revision": old.ORT_REV,
            "oracle_modes": [], "files": rows}
    for key in ("gpu_used", "inference_performed", "packages_installed", "source_executed", "image_used", "global_image_modified", "credentials_used", "private_truth_read", "challenge_inputs_used"): data[key] = False
    path = tmp_path / old.REPORT; path.parent.mkdir(parents=True); path.write_text(json.dumps(data))
    sizes = {str(base / r[0]): r[1] for r in old.ASSETS}; hashes = {str(base / r[0]): r[2] for r in old.ASSETS}
    return path, data, sizes, hashes


def test_real_shaped_failed_receipt_validates_all_nine_and_preserves(tmp_path, monkeypatch):
    path, data, sizes, hashes = failed_fixture(tmp_path); original = path.read_bytes()
    original_stat = Path.stat
    def stat_size(self, *args, **kwargs):
        value = original_stat(self, *args, **kwargs)
        return SimpleNamespace(st_mode=value.st_mode, st_size=sizes[str(self)]) if str(self) in sizes else value
    monkeypatch.setattr(Path, "stat", stat_size)
    acq = SimpleNamespace(**{k: getattr(old, k) for k in ("REPORT", "BASE", "ASSETS", "SOURCE_REV", "MODEL_REV", "FLATBUFFERS_REV", "ORT_REV")},
                          digest=lambda p: hashes.get(str(p), old.digest(p)))
    assert gate.validate_failed(tmp_path, acq, expected_sha=old.digest(path)) == data
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["status", "phase", "source", "producer", "inference", "rows", "sha", "bytes", "oracle", "wheel_audits"])
def test_failed_receipt_invalid_provenance_rejects_before_asset_reads(tmp_path, field):
    path, data, _, _ = failed_fixture(tmp_path)
    if field == "status": data["status"] = "pass"
    elif field == "phase": data["phase"] = "complete"
    elif field == "source": data["script_sha256"] = "0" * 64
    elif field == "producer": data["producer_revision"] = "0" * 40
    elif field == "inference": data["inference_performed"] = True
    elif field == "rows": data["files"] = data["files"][:-1]
    elif field == "sha": data["files"][0]["sha256"] = "0" * 64
    elif field == "bytes": data["files"][0]["bytes"] = True
    elif field == "oracle": data["oracle_modes"] = ["ground_truth"]
    elif field == "wheel_audits": data["wheel_audits"] = []  # Actual fixed receipt omits this key; even empty is different.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): gate.validate_failed(tmp_path, old, expected_sha=old.digest(path))


def test_failed_receipt_exact_sha_rejects(tmp_path):
    path, _, _, _ = failed_fixture(tmp_path)
    with pytest.raises(ValueError, match="receipt SHA"): gate.validate_failed(tmp_path, old)


def test_previous_failed_audit_is_hash_bound_not_rewritten(tmp_path):
    path = tmp_path / gate.PREVIOUS_REPORT; path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"stage": "pinned_dwpose_wheels_notice_audit_v2", "status": "fail", "phase": "failed_receipt_validation"}))
    original = path.read_bytes()
    acq = SimpleNamespace(digest=lambda p: gate.PREVIOUS_SHA)
    assert gate.validate_previous(tmp_path, acq) == gate.PREVIOUS_SHA
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="SHA"): gate.validate_previous(tmp_path, old)


def test_source_pinned_import_own_module_only():
    assert old.digest(Path(old.__file__)) == gate.SOURCE_SHA
    assert old.SOURCE_REV == "3dca5db79d9f9ffdd378753ddf6ec66535aace88"
    assert gate.IMAGE == "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"


def test_symlink_input_rejected(tmp_path):
    target = tmp_path / "file"; target.write_bytes(b"fixture")
    link = tmp_path / "link"; link.symlink_to(target)
    with pytest.raises(ValueError): gate.regular(link)


def test_main_not_remote_rejected_no_network(monkeypatch):
    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError): gate.main([])


def test_wrapper_offline_cpu_new_scope_only():
    path = Path(__file__).parents[1] / "infra/run_dwpose_wheel_audit.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert subprocess.run(["bash", str(path), "--unknown"], capture_output=True).returncode == 2
    text = path.read_text()
    assert "--network none --memory 8g --cpus 4" in text and "--gpus" not in text
    assert 'src=$ASSETS,dst=$ASSETS,readonly' in text and 'src=$RECEIPT,dst=$RECEIPT,readonly' in text
    assert 'src=$PREVIOUS,dst=$PREVIOUS,readonly' in text
    assert gate.OUT == "results/dwpose-wheel-audit-v3"
    assert '"$CODE/infra/dwpose_wheel_audit.py"' in text

"""Offline v2 audit of intact D94 assets, retaining an external Flatbuffers license.

The original failed acquisition is evidence, never repaired or overwritten.
This is byte/notice integrity only, not installation, legal clearance or accuracy.
"""
import argparse
from email.parser import BytesParser
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import time
import zipfile

SOURCE_SHA = "9a5c24de16fe6ec9169b836f1cba8e435e3d9cbb3e1ac159ca416b42363246ea"
FAILED_SHA = "9cff44a6fa41d7f9c54dac0c212b74057d4852686276e9cc740298e6b6bc3f82"
FAILED_REVISION = "b067acf9dbb8e463884dc58461d6f3dd03e77975"
OUT = "results/dwpose-wheel-audit-v2"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BUDGET_SECONDS = 120


def load_acquirer():
    path = Path(__file__).with_name("dwpose_acquire.py")
    import hashlib
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Original acquisition source SHA mismatch")
    spec = importlib.util.spec_from_file_location("world_reward_d94_acquisition", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def regular(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Require regular nonsymlink audit input")
    return path


def validate_failed(root, acquisition, *, expected_sha=FAILED_SHA):
    path = regular(root / acquisition.REPORT)
    if acquisition.digest(path) != expected_sha: raise ValueError("Original failed receipt SHA mismatch")
    data = json.loads(path.read_text())
    if (data.get("stage") != "pinned_dwpose_native_cpu_assets_acquisition" or data.get("status") != "fail"
            or data.get("phase") != "download" or data.get("error_type") != "ValueError"
            or data.get("producer_revision") != FAILED_REVISION or data.get("script_sha256") != SOURCE_SHA
            or data.get("source_revision") != acquisition.SOURCE_REV or data.get("model_revision") != acquisition.MODEL_REV
            or data.get("flatbuffers_source_revision") != acquisition.FLATBUFFERS_REV or data.get("ort_source_revision") != acquisition.ORT_REV
            or data.get("wheel_audits") != []):
        raise ValueError("Original failed acquisition provenance/phase mismatch")
    for key in ("gpu_used", "inference_performed", "packages_installed", "source_executed", "image_used",
                "global_image_modified", "credentials_used", "private_truth_read", "challenge_inputs_used"):
        if data.get(key) is not False: raise ValueError("Original acquisition safety flags differ")
    if data.get("oracle_modes") != []: raise ValueError("Original acquisition oracle mode differs")
    rows = data.get("files")
    if not isinstance(rows, list) or len(rows) != len(acquisition.ASSETS) or sum(r.get("bytes", 0) for r in rows) != 158360974:
        raise ValueError("Original acquisition nine-file byte inventory mismatch")
    for row, (name, size, sha, url) in zip(rows, acquisition.ASSETS):
        if set(row) != {"file", "bytes", "sha256", "url"} or type(row["bytes"]) is not int or row != {"file": name, "bytes": size, "sha256": sha, "url": url}:
            raise ValueError("Original acquisition exact asset inventory mismatch")
        path = regular(root / acquisition.BASE / name)
        if path.stat().st_size != size or acquisition.digest(path) != sha:
            raise ValueError("Original asset bytes/SHA changed")
    return data


def audit_flatbuffers(wheel, primary_license, retained, acquisition):
    """One explicit missing-license case: exact pinned 25.12.19 pure-Python wheel."""
    name, size, sha, _ = next(r for r in acquisition.ASSETS if r[0].startswith("wheels/flatbuffers-"))
    wheel = regular(wheel); primary_license = regular(primary_license)
    if (wheel.name != Path(name).name or wheel.stat().st_size != size or acquisition.digest(wheel) != sha
            or acquisition.FLATBUFFERS_REV != "7e163021e59cca4f8e1e35a7c828b5c6b7915953"):
        raise ValueError("Missing embedded license exception is pinned-wheel-only")
    license_record = next(r for r in acquisition.ASSETS if r[0] == "licenses/flatbuffers-LICENSE")
    if (primary_license.stat().st_size, acquisition.digest(primary_license)) != (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30") or license_record[1:3] != (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"):
        raise ValueError("Separate primary Flatbuffers Apache license binding mismatch")
    prefix = "flatbuffers-25.12.19.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist(); names = [r.filename for r in infos]
        if len(infos) != 14 or len(set(names)) != len(names) or sum(r.file_size for r in infos) > 500000000:
            raise ValueError("Pinned Flatbuffers ZIP member inventory mismatch")
        for item in infos:
            acquisition.safe_relative(item.filename); kind = stat.S_IFMT(item.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Unsafe Flatbuffers ZIP type/encryption/compression")
        regular_names = {n.rstrip("/") for n in names if not n.endswith("/")}
        if any(any(str(p) in regular_names for p in PurePosixPath(n).parents if str(p) != ".") for n in names):
            raise ValueError("Flatbuffers ZIP file/directory collision")
        if any(n.endswith(".dist-info/METADATA") and n != prefix + "METADATA" for n in names):
            raise ValueError("Multiple/incorrect Flatbuffers distribution metadata")
        def text(name):
            if archive.getinfo(name).file_size > 2000000: raise ValueError("Flatbuffers text exceeds fixed bound")
            raw = archive.read(name); raw.decode("utf-8"); return raw
        metadata = text(prefix + "METADATA"); wheel_text = text(prefix + "WHEEL")
        msg = BytesParser().parsebytes(metadata); tags = BytesParser().parsebytes(wheel_text).get_all("Tag", [])
        if msg.get("Name") != "flatbuffers" or msg.get("Version") != "25.12.19" or msg.get_all("Requires-Dist", []) != [] or sorted(tags) != ["py2-none-any", "py3-none-any"]:
            raise ValueError("Pinned Flatbuffers metadata/tags/dependencies mismatch")
        selected = [n for n in names if not n.endswith("/") and (n in (prefix + "METADATA", prefix + "WHEEL")
                    or any(k in Path(n).name.upper() for k in ("LICENSE", "LICENCE", "NOTICE", "COPYRIGHT")))]
        if any("LICEN" in Path(n).name.upper() for n in selected):
            raise ValueError("Unexpected embedded Flatbuffers license changes exact known case")
        if sum(archive.getinfo(n).file_size for n in selected) > 10000000: raise ValueError("Retained text bound exceeded")
        records = []
        for member in selected:
            raw = text(member); target = retained / acquisition.safe_relative(member); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream: stream.write(raw)
            target.chmod(0o444); records.append({"member": member, "file": str(target.relative_to(retained)), "bytes": len(raw), "sha256": acquisition.digest(target)})
    target = retained / "external-primary/flatbuffers-LICENSE"; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream: stream.write(primary_license.read_bytes())
    target.chmod(0o444); records.append({"member": None, "file": str(target.relative_to(retained)), "bytes": 11358, "sha256": acquisition.digest(target)})
    return {"package": "flatbuffers", "version": "25.12.19", "member_count": len(names), "tags": tags,
            "requires_dist": [], "safe_member_paths_verified": True, "symlinks_present": False,
            "archive_expanded": False, "selected_text_crc_verified": True, "embedded_license_present": False,
            "external_primary_license_verified": True, "external_primary_license_source_revision": acquisition.FLATBUFFERS_REV,
            "external_primary_license_url": license_record[3], "retained_texts": records}


def perform(root, out, report, persist, acquisition):
    validate_failed(root, acquisition); report["phase"] = "wheel_audit"; persist()
    base = root / acquisition.BASE
    ort = next(r for r in acquisition.ASSETS if r[0].startswith("wheels/onnxruntime-"))
    row = acquisition.audit_wheel(base / ort[0], "onnxruntime", out / "onnxruntime")
    if row["member_count"] != 353: raise ValueError("Pinned ORT member count mismatch")
    row.update(embedded_license_present=True, external_primary_license_verified=False)
    report["wheel_audits"].append(row); persist()
    flat = next(r for r in acquisition.ASSETS if r[0].startswith("wheels/flatbuffers-"))
    report["wheel_audits"].append(audit_flatbuffers(base / flat[0], base / "licenses/flatbuffers-LICENSE", out / "flatbuffers", acquisition)); persist()
    for row in report["wheel_audits"]:
        for item in row["retained_texts"]:
            path = regular(out / row["package"] / item["file"])
            if path.stat().st_size != item["bytes"] or acquisition.digest(path) != item["sha256"]:
                raise ValueError("Final retained text bytes/SHA changed")
    validate_failed(root, acquisition)
    if acquisition.digest(Path(acquisition.__file__)) != SOURCE_SHA: raise ValueError("Final original source SHA changed")
    report.update(status="pass", phase="complete", final_assets_receipt_source_rehashed=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    acquisition = load_acquirer(); root = Path(os.environ.get("WR_ROOT", "")); revision = os.environ.get("WR_CODE_REVISION", "")
    out = root / OUT
    if (platform.system() != "Linux" or str(root) != "/srv/scenesmith/world-reward" or not root.is_dir()
            or root.resolve() != root.absolute() or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or os.geteuid() != 1000 or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or os.environ.get("WR_IMAGE_ID") != IMAGE or os.environ.get("WR_DWPOSE_AUDIT_RESERVED") != "1"
            or not out.is_dir() or any(out.iterdir()) or any(p.is_symlink() for p in (out, *out.parents))):
        raise ValueError("Require fresh reserved offline pinned-image audit directory")
    report = {"stage": "pinned_dwpose_wheels_notice_audit_v2", "status": "fail", "phase": "failed_receipt_validation",
              "producer_revision": revision, "script_sha256": acquisition.digest(Path(__file__)), "image_id": IMAGE,
              "original_receipt_sha256": FAILED_SHA, "original_producer_revision": FAILED_REVISION,
              "original_acquisition_script_sha256": SOURCE_SHA, "wheel_audits": [], "budget_seconds": BUDGET_SECONDS,
              "network": "none", "device": "cpu", "gpu_used": False, "assets_redownloaded": False,
              "original_assets_modified": False, "original_failure_rewritten": False, "packages_installed": False,
              "upstream_source_executed": False, "inference_performed": False, "runtime_verified": False,
              "accuracy_verified": False, "license_clearance_verified": False, "training_data_rights_verified": False,
              "training_overlap_excluded": False, "challenge_eligibility_verified": False, "adoption_authorized": False,
              "private_truth_read": False, "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": []}
    started = time.perf_counter(); path = out / "report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("DWPose wheel audit fixed120s exceeded")
        old = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET_SECONDS)
        try: persist(); perform(root, out, report, persist, acquisition)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error="Pinned offline wheel audit failed; original evidence unchanged")
            raise RuntimeError("DWPose v2 wheel audit failed; inspect fresh immutable receipt") from None
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)
            for item in out.rglob("*"):
                if item.is_file(): item.chmod(0o444)
            for item in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True): item.chmod(0o555)
            out.chmod(0o555)


if __name__ == "__main__": main()

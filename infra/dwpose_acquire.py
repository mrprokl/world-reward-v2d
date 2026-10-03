"""Remote-only pinned DWPose/CPU-ORT acquisition; no imports, installs or inference."""
import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import time
import urllib.parse
import urllib.request
import zipfile

BASE = "weights/dwpose_native_v1"
REPORT = "results/dwpose-acquisition-v1.json"
SOURCE_REV = "3dca5db79d9f9ffdd378753ddf6ec66535aace88"
MODEL_REV = "1a7144101628d69ee7a3768d1ee3a094070dc388"
ORT_REV = "f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7"
FLATBUFFERS_REV = "7e163021e59cca4f8e1e35a7c828b5c6b7915953"
BUDGET_SECONDS = 600
ASSETS = (
    ("dw-ll_ucoco_384.onnx", 134399116, "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843", f"https://huggingface.co/yzd-v/DWPose/resolve/{MODEL_REV}/dw-ll_ucoco_384.onnx"),
    ("publisher/README.md", 28, "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74", f"https://huggingface.co/yzd-v/DWPose/raw/{MODEL_REV}/README.md"),
    ("source/onnxpose.py", 11608, "16fb69ab54f5e1ce8a5ad186e92f357da0162fc2ca2eeec4ccf1db72949291a2", f"https://raw.githubusercontent.com/IDEA-Research/DWPose/{SOURCE_REV}/ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py"),
    ("licenses/DWPose-LICENSE", 11996, "ec5aa54eef312195c4494873a37158b19b0d523e25cec570189e6aa86289cf42", f"https://raw.githubusercontent.com/IDEA-Research/DWPose/{SOURCE_REV}/LICENSE"),
    ("licenses/onnxruntime-LICENSE", 1073, "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c", f"https://raw.githubusercontent.com/microsoft/onnxruntime/{ORT_REV}/LICENSE"),
    ("licenses/onnxruntime-ThirdPartyNotices.txt", 338088, "143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3", f"https://raw.githubusercontent.com/microsoft/onnxruntime/{ORT_REV}/ThirdPartyNotices.txt"),
    ("licenses/flatbuffers-LICENSE", 11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", f"https://raw.githubusercontent.com/google/flatbuffers/{FLATBUFFERS_REV}/LICENSE"),
    ("wheels/onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl", 23561046, "fd54b314ea385bcecac69ab431f020ba503e3878dad4ebb645fec5a24b041242", "https://files.pythonhosted.org/packages/4f/2e/4c96278a99140d0307ccda6fd51e6e9d1ca7c9f6fdd1fa96ff375acfc713/onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl"),
    ("wheels/flatbuffers-25.12.19-py2.py3-none-any.whl", 26661, "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4", "https://files.pythonhosted.org/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl"),
)
WHEELS = {
    "onnxruntime": {"version": "1.30.0", "tags": ["cp311-cp311-manylinux_2_28_x86_64"],
                    "requires": ["flatbuffers", "numpy>=1.21.6", "packaging", "protobuf>=4.25.8", 'sympy; extra == "symbolic"', 'ml_dtypes; extra == "quantization"']},
    "flatbuffers": {"version": "25.12.19", "tags": ["py2-none-any", "py3-none-any"], "requires": []},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\0" in name or ":" in name
            or name.startswith("/") or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))):
        raise ValueError("Unsafe asset/archive member path")
    return PurePosixPath(name)


def validate_https(url):
    p = urllib.parse.urlsplit(url); host = p.hostname or ""
    if (p.scheme != "https" or p.username or p.password or p.port not in (None, 443)
            or not (host in {"huggingface.co", "raw.githubusercontent.com", "pypi.org", "files.pythonhosted.org"}
                    or host.endswith((".hf.co", ".huggingface.co")))):
        raise ValueError("Public allowlisted HTTPS required")


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        validate_https(url)
        return super().redirect_request(request, fp, code, message, headers, url)


def open_public(url):
    validate_https(url)
    return urllib.request.build_opener(HTTPSRedirects()).open(
        urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=180)


def fetch_json(url):
    with open_public(url) as response:
        validate_https(response.geturl()); raw = response.read(500001)
    if len(raw) > 500000: raise ValueError("Primary metadata exceeds bounded size")
    return json.loads(raw)


def primary_metadata(fetch=fetch_json):
    url = f"https://huggingface.co/api/models/yzd-v/DWPose/revision/{MODEL_REV}?blobs=true"
    data = fetch(url)
    if data.get("sha") != MODEL_REV or data.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("Pinned publisher revision/license metadata mismatch")
    rows = {r["rfilename"]: r for r in data["siblings"]}; item = rows.get(ASSETS[0][0], {})
    if (item.get("size"), item.get("lfs", {}).get("size"), item.get("lfs", {}).get("sha256")) != (ASSETS[0][1], ASSETS[0][1], ASSETS[0][2]):
        raise ValueError("Publisher model SHA/bytes mismatch")
    records = [{"url": url, "revision": MODEL_REV, "license_declaration": "apache-2.0"}]
    for package, spec in WHEELS.items():
        url = f"https://pypi.org/pypi/{package}/{spec['version']}/json"; data = fetch(url)
        name, size, sha, wheel_url = next(r for r in ASSETS if r[0].startswith(f"wheels/{package}-"))
        candidates = [r for r in data["urls"] if r["filename"] == Path(name).name]
        if (data["info"].get("version") != spec["version"] or len(candidates) != 1
                or sorted(data["info"].get("requires_dist") or []) != sorted(spec["requires"])):
            raise ValueError("Pinned PyPI version/dependencies mismatch")
        row = candidates[0]
        if (row.get("size"), row.get("digests", {}).get("sha256"), row.get("url"), row.get("yanked")) != (size, sha, wheel_url, False):
            raise ValueError("Pinned PyPI wheel SHA/bytes/URL mismatch")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+Z", row.get("upload_time_iso_8601", "")) or row["upload_time_iso_8601"] >= "2026-10-01T00:00:00Z":
            raise ValueError("Wheel published after literature cutoff or date unknown")
        records.append({"url": url, "version": spec["version"], "upload_time": row["upload_time_iso_8601"]})
    return records


def download(url, destination, size, sha, opener=open_public):
    """One exclusive, exact-size attempt; failed partial bytes remain for audit."""
    if type(size) is not int or not 0 < size <= 200000000 or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid frozen asset byte/SHA contract")
    count = 0; h = hashlib.sha256()
    with opener(url) as response, Path(destination).open("xb") as out:
        validate_https(response.geturl())
        if response.headers.get("Content-Encoding", "identity") != "identity": raise ValueError("Encoded asset rejected")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != size: raise ValueError("Asset Content-Length mismatch")
        while block := response.read(min(1024 * 1024, size - count + 1)):
            if count == 0 and (block.lstrip().lower().startswith((b"<!doctype html", b"<html")) or block.startswith(b"version https://git-lfs.github.com/spec/")):
                raise ValueError("HTML/LFS pointer is not an asset")
            count += len(block)
            if count > size: raise ValueError("Asset exceeds exact byte bound")
            h.update(block); out.write(block)
    if count != size or h.hexdigest() != sha: raise ValueError("Asset byte/SHA mismatch")
    Path(destination).chmod(0o444)


def audit_wheel(path, package, retained):
    """Inspect all paths/types, but extract only bounded text licenses/metadata."""
    spec = WHEELS[package]; prefix = f"{package}-{spec['version']}.dist-info/"
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [r.filename for r in infos]
        if not infos or len(infos) > 10000 or len(set(names)) != len(names) or sum(r.file_size for r in infos) > 500000000:
            raise ValueError("Unsafe wheel inventory/count/expanded-size")
        for item in infos:
            safe_relative(item.filename); mode = item.external_attr >> 16; kind = stat.S_IFMT(mode)
            if (kind not in (0, stat.S_IFREG, stat.S_IFDIR) or item.flag_bits & 1
                    or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("Wheel symlink/special/encrypted member rejected")
        regular = {n.rstrip("/") for n in names if not n.endswith("/")}
        if any(any(str(parent) in regular for parent in PurePosixPath(n).parents if str(parent) != ".") for n in names):
            raise ValueError("Wheel file/directory path collision")
        if any(n.endswith(".dist-info/METADATA") and n != prefix + "METADATA" for n in names):
            raise ValueError("Multiple/incorrect distribution metadata")
        def text(name):
            item = archive.getinfo(name)
            if item.file_size > 2000000: raise ValueError("Wheel text exceeds fixed bound")
            raw = archive.read(item); raw.decode("utf-8"); return raw
        metadata = text(prefix + "METADATA"); wheel = text(prefix + "WHEEL")
        msg = BytesParser().parsebytes(metadata); tags = BytesParser().parsebytes(wheel).get_all("Tag", [])
        if (msg.get("Name") != package or msg.get("Version") != spec["version"]
                or sorted(msg.get_all("Requires-Dist", [])) != sorted(spec["requires"]) or sorted(tags) != sorted(spec["tags"])):
            raise ValueError("Wheel version/tags/dependencies mismatch")
        selected = [n for n in names if not n.endswith("/") and (n in (prefix + "METADATA", prefix + "WHEEL")
                    or any(k in Path(n).name.upper() for k in ("LICENSE", "LICENCE", "NOTICE", "COPYRIGHT")))]
        if not any("LICEN" in Path(n).name.upper() for n in selected): raise ValueError("Wheel embedded license missing")
        if sum(archive.getinfo(n).file_size for n in selected) > 10000000: raise ValueError("Wheel retained texts exceed fixed bound")
        records = []
        for name in selected:
            raw = text(name); target = Path(retained) / safe_relative(name); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out: out.write(raw)
            target.chmod(0o444); records.append({"member": name, "file": str(target.relative_to(retained)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"package": package, "version": spec["version"], "tags": tags, "requires_dist": msg.get_all("Requires-Dist", []),
            "member_count": len(names), "safe_member_paths_verified": True, "symlinks_present": False,
            "archive_expanded": False, "selected_text_crc_verified": True, "retained_texts": records}


def save_json(path, value):
    with Path(path).open("x") as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    Path(path).chmod(0o444)


def acquire(root, report, persist, metadata=primary_metadata, downloader=download):
    base = root / BASE; staging = base / ".downloads"; staging.mkdir(mode=0o755)
    report["primary_metadata"] = metadata(); persist()
    for index, (name, size, sha, url) in enumerate(ASSETS):
        report.update(phase="download", active_file=name); persist()
        part = staging / f"{index:02d}.part"; downloader(url, part, size, sha)
        target = base / safe_relative(name); target.parent.mkdir(parents=True, exist_ok=True)
        os.link(part, target); part.unlink()
        report["files"].append({"file": name, "bytes": size, "sha256": sha, "url": url}); persist()
    imports = set()
    for node in ast.walk(ast.parse((base / "source/onnxpose.py").read_text())):
        if isinstance(node, ast.Import): imports.update(r.name for r in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level: raise ValueError("Standalone source acquired unexpected relative import")
            imports.add(node.module)
    if imports != {"typing", "cv2", "numpy", "onnxruntime"}: raise ValueError("Standalone source import closure changed")
    report["wheel_audits"] = [audit_wheel(base / r[0], package, base / "wheel_audit" / package)
                              for package in WHEELS for r in ASSETS if r[0].startswith(f"wheels/{package}-")]
    for audit in report["wheel_audits"]:
        for row in audit["retained_texts"]:
            path = base / "wheel_audit" / audit["package"] / row["file"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Final retained wheel text rehash differs")
    for row in report["files"]:
        path = base / row["file"]
        if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Final asset rehash differs")
        path.chmod(0o444)
    staging.rmdir()
    report.update(status="pass", phase="complete", source_import_names=sorted(imports), final_assets_rehashed=True,
                  runtime_prefix_plan="Private future pip --no-deps --target; not installed by acquisition")
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ.get("WR_ROOT", "")); revision = os.environ.get("WR_CODE_REVISION", "")
    if (platform.system() != "Linux" or str(root) != "/srv/scenesmith/world-reward" or not root.is_dir()
            or root.resolve() != root.absolute() or os.geteuid() != 1000 or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or os.environ.get("WR_DWPOSE_OUTPUT_RESERVED") != "1"):
        raise ValueError("Require reserved canonical Azure Linux root/UID1000/committed source")
    base = root / BASE; path = root / REPORT
    for target in (base, path):
        if any(p.is_symlink() for p in (target, *target.parents)): raise ValueError("Symlink acquisition path rejected")
    if not base.is_dir() or any(base.iterdir()) or path.exists(): raise FileExistsError("Acquisition targets must be fresh")
    report = {"stage": "pinned_dwpose_native_cpu_assets_acquisition", "status": "fail", "phase": "start", "files": [],
              "producer_revision": revision, "script_sha256": digest(Path(__file__)), "source_revision": SOURCE_REV,
              "model_revision": MODEL_REV, "ort_source_revision": ORT_REV, "flatbuffers_source_revision": FLATBUFFERS_REV,
              "budget_seconds": BUDGET_SECONDS, "device": "cpu", "network": "public_https_acquisition_only",
              "source_license": "Apache-2.0", "publisher_model_license_declaration": "Apache-2.0",
              "ort_license": "MIT", "flatbuffers_license": "Apache-2.0", "license_clearance_verified": False,
              "training_data_rights_verified": False, "teacher_rights_verified": False, "training_overlap_excluded": False,
              "challenge_eligibility_verified": False, "source_executed": False, "onnx_graph_verified": False,
              "runtime_verified": False, "accuracy_verified": False, "adoption_authorized": False,
              "gpu_used": False, "inference_performed": False, "packages_installed": False, "image_used": False,
              "global_image_modified": False, "credentials_used": False, "private_truth_read": False,
              "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [],
              "future_input_channel_order": "RGB per publisher clarification, not a graph audit",
              "input_channel_primary_url": "https://github.com/IDEA-Research/DWPose/issues/25#issuecomment-1696943489",
              "training_sources": ["COCO-WholeBody", "UBody"], "training_sources_rights_cleared": False}
    started = time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("DWPose acquisition fixed budget exceeded")
        old = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET_SECONDS)
        try: persist(); acquire(root, report, persist)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error="Acquisition failed; signed URLs/credentials are not logged")
            raise RuntimeError("Pinned DWPose acquisition failed; inspect immutable receipt") from None
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term)
            report["partial_files"] = [{"file": str(p.relative_to(base)), "bytes": p.stat().st_size, "sha256": digest(p)}
                                       for p in (base / ".downloads").glob("*.part")]
            for p in base.rglob("*"):
                if p.is_file(): p.chmod(0o444)
            for p in sorted((p for p in base.rglob("*") if p.is_dir()), reverse=True): p.chmod(0o555)
            base.chmod(0o555); persist(); path.chmod(0o444)


if __name__ == "__main__": main()

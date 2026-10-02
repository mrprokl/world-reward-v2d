"""Remote-only acquisition of pinned GeoCalib frontend source and public weights.

No upstream imports, installs, camera solver, credentials or dataset downloads.
The publisher gives no weight digest: the receipt records downloaded bytes, not
an independently verified checkpoint identity or competition eligibility.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import time
import urllib.parse
import urllib.request

SOURCE_REV = "97b8968e7798a66bf04fcf791fb535624241bda7"
SEGNEXT_REV = "d46ffa737980ec7a9f5b9465c78f254449163509"
ASSETS = "weights/geocalib-pinhole-v1-frontend"
REPORT = "results/geocalib-assets-v1.json"
WEIGHT_URL = "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar"
WEIGHT_BYTES = 116074121
CLASS_BYTES = 2569
CLASS_SHA = "95b63c6917f9922e0859c6e2ce268bd25915ef965aa7646ede6cfd40d9bb7414"
RAW = f"https://raw.githubusercontent.com/cvg/GeoCalib/{SOURCE_REV}/"
SOURCE_RECORDS = (
    ("modules.py", RAW+"geocalib/modules.py", 18900, "222f28ef570fbda9bd46bbd7c6089ccff8b9d96072177dd54179b6444fd48de6"),
    ("geocalib.audit.py", RAW+"geocalib/geocalib.py", 5474, "4c221b3922911e0372f5837bc3a4e87ac44afe2721049b20a65cdcf6b8a2eab9"),
    ("LICENSE.GeoCalib", RAW+"LICENSE", 10755, "b458637616125d96702f04666c7717d3a1ade0824f6ea6ccf512b4828f35c8d5"),
    ("LICENSE.SegNeXt", f"https://raw.githubusercontent.com/Visual-Attention-Network/SegNeXt/{SEGNEXT_REV}/LICENSE", 11345, "76466caf809e511437c8265f484054aec05be2a70d9a2900246f46a57d158c8b"),
    ("README.GeoCalib.md", RAW+"README.md", 23790, "9568126845a274760210f7a41a3915ea42aab48b6a938be07f92e673a6a5ccfb"),
)
CREDITS = ("GeoCalib frontend extraction, cvg/GeoCalib@"+SOURCE_REV+"\n"
           "Original source is retained unchanged; classes are original lines 18-89.\n"
           "Code: Apache-2.0, see LICENSE.GeoCalib and original attribution notices.\n"
           "MSCAN/Light Hamburger are adapted from SegNeXt; attribution is retained\n"
           "in modules.py and LICENSE.SegNeXt. Weights: publisher README states CC-BY-4.0.\n"
           "No PerspectiveFields, LM optimizer or original package initializer is acquired.\n").encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): value.update(block)
    return value.hexdigest()


def safe_path(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink asset/report path forbidden")
    return path


def validate_https(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.hostname not in ("github.com", "api.github.com", "raw.githubusercontent.com",
                                       "release-assets.githubusercontent.com", "objects.githubusercontent.com")):
        raise ValueError("Require allowlisted public HTTPS")


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        validate_https(url)
        return super().redirect_request(request, fp, code, message, headers, url)


def open_public(url):
    validate_https(url)
    return urllib.request.build_opener(Redirects()).open(
        urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=30)


def download(url, destination, size, expected_sha=None):
    """Exclusive bounded stream; signed redirect URLs are never persisted."""
    if type(size) is not int or not 0 < size <= WEIGHT_BYTES: raise ValueError("Invalid fixed byte bound")
    count = 0; sha = hashlib.sha256()
    with open_public(url) as response, Path(destination).open("xb") as output:
        validate_https(response.geturl())
        if response.headers.get("Content-Encoding", "identity") != "identity": raise ValueError("Encoded download forbidden")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != size: raise ValueError("Declared bytes differ")
        while block := response.read(min(1024*1024, size-count+1)):
            count += len(block)
            if count > size: raise ValueError("Download exceeds exact size")
            sha.update(block); output.write(block)
    if count != size or (expected_sha is not None and sha.hexdigest() != expected_sha):
        raise ValueError("Source byte/hash mismatch")
    Path(destination).chmod(0o444)
    return {"url": url, "bytes": count, "sha256": sha.hexdigest()}


def extract_classes(raw):
    source = b"".join(raw.splitlines(keepends=True)[17:89])
    if len(source) != CLASS_BYTES or hashlib.sha256(source).hexdigest() != CLASS_SHA:
        raise ValueError("Exact native class slice differs")
    tree = ast.parse(source)
    if [n.name for n in tree.body if isinstance(n, ast.ClassDef)] != ["LowLevelEncoder", "UpDecoder", "LatitudeDecoder", "PerspectiveDecoder"] or any(not isinstance(n, ast.ClassDef) for n in tree.body):
        raise ValueError("Native class slice contains unexpected executable nodes")
    return source


def publisher_metadata():
    url = "https://api.github.com/repos/cvg/GeoCalib/releases/tags/v1.0"
    with open_public(url) as response: data = response.read(100001)
    if len(data) > 100000: raise ValueError("Release metadata exceeds bound")
    release = json.loads(data); items = [a for a in release.get("assets", []) if a.get("name") == "geocalib-pinhole.tar"]
    if (release.get("published_at") != "2024-09-08T13:05:40Z" or len(items) != 1
            or items[0].get("size") != WEIGHT_BYTES or items[0].get("browser_download_url") != WEIGHT_URL
            or items[0].get("digest") is not None): raise ValueError("Pinned publisher metadata differs")
    return {"publisher_release_url": url, "publisher_metadata_verified": True,
            "publisher_digest_available": False, "publisher_digest_verified": False,
            "weight_published_at": release["published_at"]}


def acquire(root, report, persist):
    target = root/ASSETS; staging = target/".downloads"; staging.mkdir(mode=0o700)
    try:
        report.update(publisher_metadata()); persist()
        for name, url, size, sha in SOURCE_RECORDS:
            report["phase"] = "source:"+name; persist()
            record = download(url, staging/name, size, sha)
            os.link(staging/name, target/name); (staging/name).unlink()
            report["files"][name] = record; persist()
        classes = extract_classes((target/"geocalib.audit.py").read_bytes())
        for name, content in (("classes18_89.py", classes), ("CREDITS.txt", CREDITS)):
            with (target/name).open("xb") as stream: stream.write(content)
            (target/name).chmod(0o444)
            report["files"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        report["phase"] = "weights"; persist()
        record = download(WEIGHT_URL, staging/"weights", WEIGHT_BYTES)
        os.link(staging/"weights", target/"geocalib-pinhole.tar"); (staging/"weights").unlink()
        report["files"]["geocalib-pinhole.tar"] = record
        report.update(status="pass", phase="complete", source_subset_verified=True)
    finally:
        shutil.rmtree(staging)
        target.chmod(0o555)
        report["disposable_downloads_removed"] = True


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Acquisition stays remote Linux only")
    root = safe_path(Path(os.environ["WR_ROOT"])); revision = os.environ.get("WR_CODE_REVISION", "")
    target = safe_path(root/ASSETS); path = safe_path(root/REPORT)
    if (root != Path("/srv/scenesmith/world-reward") or not root.is_dir() or root.resolve() != root
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or os.geteuid() != 1000
            or os.environ.get("WR_GEOCALIB_OUTPUTS_RESERVED") != "1" or not target.is_dir() or any(target.iterdir())):
        raise ValueError("Require canonical root, committed source and reserved empty UID1000 directory")
    report = {"stage": "pinned_geocalib_frontend_assets_acquisition", "status": "fail", "phase": "metadata", "files": {},
              "code_revision": revision, "script_sha256": digest(Path(__file__)), "source_revision": SOURCE_REV,
              "segnext_revision": SEGNEXT_REV, "code_license": "Apache-2.0", "weight_license_publisher_statement": "CC-BY-4.0",
              "publisher_digest_available": False, "publisher_digest_verified": False, "checkpoint_header_verified": False,
              "full_package_license_eligibility_verified": False, "camera_solver_acquired": False, "gpu_used": False,
              "challenge_inputs_used": False, "ground_truth_used": False, "inference_performed": False, "budget_seconds": 300}
    started = time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Acquisition budget exceeded")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(300)
        try: persist(); acquire(root, report, persist)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error="Bounded public acquisition failed; inspect phase, no signed URLs logged")
            raise RuntimeError("GeoCalib asset acquisition failed") from None
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()

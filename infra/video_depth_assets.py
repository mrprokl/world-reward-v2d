"""Acquire only pinned public VDA Small metric assets on Azure; never infer."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time
import urllib.parse
import urllib.request

ROOT = Path("/srv/scenesmith/world-reward")
SOURCE_DIR = "vendor/research/vda_metric_small_v1"
WEIGHTS_DIR = "weights/research/vda_metric_small_v1"
REPORT = "results/video-depth-assets-v1.json"
CONFIG = "configs/video_depth_assets_v1.json"
CONFIG_SHA256 = "5d672397091ab01bd686fa5c7edbdb00d57a720347bef53edb131706d0ba7820"


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def configuration(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("Frozen VDA acquisition configuration changed")
    return json.loads(raw)


def approved_url(url, metadata=False):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    allowed = {"api.github.com", "raw.githubusercontent.com", "huggingface.co"}
    if not metadata:
        allowed.update({"cdn-lfs.huggingface.co", "cdn-lfs.hf.co", "us.aws.cdn.hf.co"})
    return (parsed.scheme == "https" and not parsed.username and not parsed.password
            and (host in allowed or not metadata and host.endswith(".xethub.hf.co")))


def download(url, path, expected_size):
    """Retry transport only; fixed byte cap, bounded requests, fresh partial file."""
    if not approved_url(url, metadata=True):
        raise ValueError("Non-authoritative asset URL")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "World-Reward-V2D/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if not approved_url(response.geturl()):
                    raise ValueError("Unexpected asset redirect host")
                with Path(path).open("wb") as stream:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > expected_size:
                            raise ValueError("Asset exceeds independent size pin")
                        stream.write(chunk)
            if total != expected_size:
                raise ValueError("Incomplete asset byte count")
            return
        except (OSError, TimeoutError):
            Path(path).unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def publisher_metadata(config):
    """Cross-check current response at immutable commits against committed pins."""
    urls = ["https://api.github.com/repos/" + config["source_repository"] + "/git/trees/"
            + config["source_revision"] + "?recursive=1",
            "https://huggingface.co/api/models/" + config["model_repository"] + "/revision/"
            + config["model_revision"] + "?blobs=true"]
    documents = []
    for url in urls:
        with urllib.request.urlopen(url, timeout=30) as response:
            if not approved_url(response.geturl(), metadata=True):
                raise ValueError("Unexpected metadata redirect")
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Publisher metadata exceeds budget")
            documents.append(json.loads(raw))
    tree, model = documents
    if tree.get("sha") != config["source_revision"] or tree.get("truncated") is not False:
        raise ValueError("Exact complete immutable GitHub tree required")
    entries = {row["path"]: row for row in tree["tree"]}
    for row in config["source_files"]:
        entry = entries.get(row["file"], {})
        if (entry.get("type"), entry.get("size"), entry.get("sha")) != (
                "blob", row["bytes"], row["git_blob_sha1"]):
            raise ValueError("Native source tree differs from independent pins")
    weights = [row for row in model.get("siblings", []) if row.get("rfilename") == config["model_file"]]
    if (model.get("sha") != config["model_revision"] or model.get("cardData", {}).get("license")
            != config["model_license"] or len(weights) != 1 or weights[0].get("size") != config["model_bytes"]
            or weights[0].get("lfs", {}).get("sha256") != config["model_sha256"]):
        raise ValueError("Official checkpoint revision/license/LFS size/hash mismatch")


def acquire(root, code, revision, *, publisher_cdn_v2=False):
    source, weights = root / SOURCE_DIR, root / WEIGHTS_DIR
    receipt=root/("results/video-depth-assets-v2.json" if publisher_cdn_v2 else REPORT)
    config = configuration(code / CONFIG)
    if any(path.exists() or path.is_symlink() for path in (source, weights, receipt)):
        raise ValueError("Acquisition destinations already exist; no overwrite/restart")
    if any(path.resolve() != path for path in (source, weights, receipt)):
        raise ValueError("Asset destination symlink ancestry is forbidden")
    if publisher_cdn_v2:
        failure=root/REPORT
        old=json.loads(failure.read_text())
        if old.get('status')!='fail' or old.get('failed_partial_assets_removed') is not True:
            raise ValueError('Verified prior transport failure required')
    source.mkdir(parents=True); weights.mkdir(parents=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); error = None; records = []
    report = dict(schema="world_reward.video_depth_assets_receipt.v1", status="fail", producer_revision=revision,
        configuration_sha256=CONFIG_SHA256, source_revision=config["source_revision"],
        model_revision=config["model_revision"], license=config["model_license"], GPU_used=False,
        inference_performed=False, challenge_inputs_used=False, challenge_overlap_verified=False,
        exact_training_frames_attested=False, source_modified=False)
    try:
        publisher_metadata(config)
        for row in config["source_files"]:
            path = source / row["file"]; path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://raw.githubusercontent.com/" + config["source_repository"] + "/" + config["source_revision"] + "/" + row["file"]
            download(url, path, row["bytes"]); raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if identity(path) != {key: row[key] for key in ("bytes", "sha256")} or blob != row["git_blob_sha1"]:
                raise ValueError("Native source blob/byte identity mismatch")
            path.chmod(0o444); records.append({"file": str(path.relative_to(root)), **identity(path)})
        path = weights / config["model_file"]
        url = "https://huggingface.co/" + config["model_repository"] + "/resolve/" + config["model_revision"] + "/" + config["model_file"]
        download(url, path, config["model_bytes"])
        if identity(path) != {"bytes": config["model_bytes"], "sha256": config["model_sha256"]}:
            raise ValueError("Checkpoint differs from independently published LFS hash")
        path.chmod(0o444); records.append({"file": str(path.relative_to(root)), **identity(path)})
        for record in records:
            if identity(root / record["file"]) != {key: record[key] for key in ("bytes", "sha256")}:
                raise ValueError("Asset changed after acquisition")
        report.update(status="pass", files=records, assets_rehashed_after=True)
    except Exception as exc:
        error = exc
        report.update(error_type=type(exc).__name__, error="Asset acquisition failed; no secrets/redirect URLs logged")
        for directory in (source, weights):
            for path in sorted(directory.rglob("*"), key=lambda value: len(value.parts), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            directory.rmdir()
        report["failed_partial_assets_removed"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    with receipt.open("x") as stream:
        json.dump(report, stream, allow_nan=False); stream.write("\n")
    receipt.chmod(0o444)
    if error:
        raise RuntimeError("VDA acquisition failed; immutable failure receipt preserved") from None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publisher-cdn-v2',action='store_true');args=parser.parse_args()
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    if (root != ROOT or platform.system() != "Linux" or root.resolve() != root
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / ("run_video_depth_assets_v2/code" if args.publisher_cdn_v2 else "run_video_depth_assets/code") or code.resolve() != code
            or Path(__file__).resolve() != code / "infra/video_depth_assets.py"):
        raise ValueError("Immutable Azure acquisition namespace required")
    def expired(*_):
        raise TimeoutError("VDA acquisition budget expired")
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired); signal.alarm(590)
    try:
        acquire(root, code, revision,publisher_cdn_v2=args.publisher_cdn_v2)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

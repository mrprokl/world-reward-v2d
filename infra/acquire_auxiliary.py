"""Azure-only pinned DINO source/checkpoints for offline Body/Objects smoke tests.

FoundationPose is intentionally not acquired here: its source eligibility is a
separate organizer question. First-seen official reg4 checkpoint hashes are
recorded explicitly, not represented as independently verified release hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request


REPOSITORIES = {
    "dinov3": "6876159a11b4df116f30f667f8c9888617df0751",
    "dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
}
RELEASE_HASHES = {
    "dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkout(destination: Path, repository: str, revision: str) -> None:
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin",
                        f"https://github.com/facebookresearch/{repository}.git"], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True)
    if actual != revision or dirty:
        raise RuntimeError(f"Unpinned/modified {repository} source: {destination}")


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Auxiliary downloads are restricted to Azure Linux")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    weights = root / "weights"
    body_hub = weights / "cari4d/sam3d_body/torch_home/hub"
    object_hub = weights / "sam3d/torch_home/hub"
    for hub in (body_hub, object_hub):
        for repository, revision in REPOSITORIES.items():
            checkout(hub / f"facebookresearch_{repository}_main", repository, revision)
        (hub / "trusted_list").touch(exist_ok=True)
    records = []
    for model, suffix, hub in (
        ("vitb14", "", body_hub), ("vits14", "", body_hub),
        ("vitl14", "_reg4", object_hub), ("vitb14", "_reg4", object_hub),
    ):
        filename = f"dinov2_{model}{suffix}_pretrain.pth"
        url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{model}/{filename}"
        target = hub / "checkpoints" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_suffix(".part")
            print(json.dumps({"acquiring": filename}), flush=True)
            urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Empty checkpoint: {filename}")
            if filename in RELEASE_HASHES and digest(partial) != RELEASE_HASHES[filename]:
                raise RuntimeError(f"Checkpoint integrity mismatch: {filename}")
            partial.replace(target)
        actual = digest(target)
        if filename in RELEASE_HASHES and actual != RELEASE_HASHES[filename]:
            raise RuntimeError(f"Checkpoint integrity mismatch: {filename}")
        records.append({"filename": filename, "url": url, "sha256": actual,
                        "bytes": target.stat().st_size,
                        "hash_source": "official_pinned_downloader" if filename in RELEASE_HASHES else "first_observed_https_download"})
    report = {"scope": "DINO_source_and_checkpoints_only", "source_revisions": REPOSITORIES,
              "checkpoints": records, "foundationpose_acquired": False}
    (root / "results/auxiliary-assets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

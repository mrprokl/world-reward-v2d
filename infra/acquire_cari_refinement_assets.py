"""Acquire two pinned native refinement assets directly on Azure, not laptop."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import urllib.request


REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
ASSETS = {
    "mhr_collision_proxy_4000v.npz": (126142, "a026fe82599609ee82814b4c18eb0335fa70925740dea5e1c1e1c0930d5edcf7"),
    "mhr_hand_surface_spec.npz": (47140, "65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8"),
}
URL_BASE = (f"https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/{REVISION}/"
            "reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/")


def asset_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_asset(path, size, digest):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != size
            or asset_sha(path) != digest):
        raise ValueError("Refinement asset bytes do not match pinned Git LFS object")


def acquire(root):
    directory = root / "weights/cari4d/refinement"
    if directory.is_symlink():
        raise ValueError("Refinement asset directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "results/cari-refinement-assets.json"
    records = []
    for name, (size, digest) in ASSETS.items():
        target = directory / name
        url = URL_BASE + name
        if not target.exists() and not target.is_symlink():
            partial = directory / (name + ".part")
            # Never overwrite an interrupted acquisition silently.
            with partial.open("xb") as handle:
                with urllib.request.urlopen(url, timeout=60) as response:
                    # Bound bytes even if a server sends a wrong or pointer file.
                    payload = response.read(size + 1)
                handle.write(payload)
            try:
                verify_asset(partial, size, digest)
                partial.replace(target)
            except Exception:
                partial.unlink()
                raise
        verify_asset(target, size, digest)
        records.append({"filename": name, "bytes": size, "sha256": digest,
                        "revision": REVISION, "url": url})
    result = {"upstream_revision": REVISION, "assets": records,
              "scope": "native_MHR_refinement_collision_and_hand_surface_assets_only",
              "challenge_data_acquired": False, "vendor_source_modified": False}
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink() or json.loads(receipt_path.read_text()) != result:
            raise ValueError("Preserve existing differing frozen refinement asset receipt")
    else:
        with receipt_path.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    return result


def main():
    root = Path(os.environ["WR_ROOT"])
    if platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward"):
        raise RuntimeError("Refinement downloads require the isolated Azure Linux root")
    print(json.dumps(acquire(root)), flush=True)


if __name__ == "__main__":
    main()

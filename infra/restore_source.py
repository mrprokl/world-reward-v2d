"""Materialize only whitelisted Python source omitted by sparse data exclusion."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Vendor source materialization is Azure-only")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    repository = root / "vendor/video_to_data"
    revision = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={repository}",
                                        "-C", str(repository), *args])
    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise RuntimeError("Unpinned upstream source")
    prefix = "reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/data/"
    paths = git("ls-tree", "-r", "--name-only", revision, prefix).decode().splitlines()
    if len(paths) != 6 or any(not path.endswith(".py") for path in paths):
        raise RuntimeError("Source whitelist changed; re-audit before materialization")
    for relative in paths:
        # Git may fetch a missing source blob here; no datasets are read.
        data = git("show", f"{revision}:{relative}")
        if len(data) > 100_000:
            raise RuntimeError("Unexpectedly large source file")
        path = repository / relative
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError("Preserve modified vendor work; refusing overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print("restored_sam3d_body_source_python_files=6")


if __name__ == "__main__":
    main()

"""Download only the pinned Track 1 inputs, on a remote Linux host.

No credentials are written by this module. The pinned Hub snapshot supplies
content integrity; the result manifest records SHA-256 of actual local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sources.json"


def allowed_path(path: str) -> bool:
    if "\\" in path or ".." in path.split("/"):
        return False
    p = PurePosixPath(path)
    if p.is_absolute() or len(p.parts) < 2 or p.parts[0] != "track_1":
        return False
    return bool(
        path in {"track_1/README.md", "track_1/.gitignore"}
        or re.fullmatch(r"track_1/meta/(info\.json|(?:episodes|episodes_metadata|episodes_stats|tasks)\.jsonl)", path)
        or re.fullmatch(r"track_1/data/chunk-000/episode_\d{6}\.parquet", path)
        or re.fullmatch(r"track_1/videos/chunk-000/observation\.images\.exo_camera/episode_\d{6}\.mp4", path)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frame_indexes(path: Path, episode: int, expected_frames: int) -> None:
    import pyarrow.parquet as pq
    import numpy as np

    table = pq.read_table(path)
    allowed_columns = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"}
    if set(table.column_names) != allowed_columns:
        raise ValueError(f"Unexpected non-structural columns in {path.name}")
    if table.num_rows != expected_frames:
        raise ValueError(f"Frame count mismatch in {path.name}")
    frames = table["frame_index"].to_numpy()
    episodes = table["episode_index"].to_numpy()
    if not np.array_equal(frames, np.arange(expected_frames)) or not (episodes == episode).all():
        raise ValueError(f"Non-contiguous or wrong-episode frame indexes in {path.name}")


def download(config_path: Path, root: Path, manifest_path: Path) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Dataset download is restricted to the remote Linux compute/storage host")
    from huggingface_hub import HfApi, hf_hub_download

    config = json.loads(config_path.read_text())["dataset"]
    if config["repo_id"] != "nvidia/video_to_data_challenge" or config["prefix"] != "track_1/":
        raise ValueError("Only the official Track 1 source is permitted")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Dataset must be pinned to a full revision, never main")
    root = root.resolve()
    api = HfApi()
    entries = list(api.list_repo_tree(config["repo_id"], path_in_repo="track_1", recursive=True,
                                     revision=config["revision"], repo_type="dataset"))
    files = [entry for entry in entries if hasattr(entry, "size")]
    for entry in files:
        if not allowed_path(entry.path):
            raise ValueError(f"Source revision contains a disallowed path: {entry.path}")
    records = []
    for entry in files:
        name = hf_hub_download(config["repo_id"], entry.path, repo_type="dataset",
                               revision=config["revision"], local_dir=root)
        path = Path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Downloaded file escapes the data root")
        if path.stat().st_size != entry.size:
            raise ValueError(f"Size mismatch: {entry.path}")
        records.append({"path":entry.path,"bytes":entry.size,"sha256":sha256(path)})
    episodes = [json.loads(line) for line in (root / "track_1/meta/episodes.jsonl").read_text().splitlines()]
    if len(episodes) != config["episodes"] or sum(e["length"] for e in episodes) != config["frames"]:
        raise ValueError("Unexpected episode/frame counts at the pinned revision")
    for e in episodes:
        verify_frame_indexes(root / f"track_1/data/chunk-000/episode_{e['episode_index']:06d}.parquet",
                             e["episode_index"], e["length"])
    manifest = {"repo_id":config["repo_id"],"revision":config["revision"],"track":"track_1",
                "episodes":len(episodes),"frames":config["frames"],"files":records}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = download(args.config, args.root, args.manifest)
    print(json.dumps({"track":manifest["track"],"episodes":manifest["episodes"],
                      "frames":manifest["frames"],"files":len(manifest["files"])}))

"""Explicit storage namespaces; no change to reconstruction or asset roots."""
from pathlib import Path
import os
import re


def output_prefix() -> str:
    value = os.environ.get("WR_OUTPUT_PREFIX", "outputs")
    if value != "outputs" and re.fullmatch(
        r"experiments/[a-z][a-z0-9-]{0,50}-[0-9a-f]{40}/outputs", value
    ) is None:
        raise ValueError("Require legacy outputs or a revision-bound experiment namespace")
    return value


def episode_output(root: Path, episode: int) -> Path:
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Require an original Track1 episode index")
    path = Path(root) / output_prefix() / f"episode_{episode:06d}"
    if not path.is_absolute() or path.resolve() != path or any(
        parent.is_symlink() for parent in (path, *path.parents)
    ):
        raise ValueError("Canonical output namespace required; no aliases")
    return path

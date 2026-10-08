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


def episode_relative(episode: int) -> str:
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Require an original Track1 episode index")
    return output_prefix() + f"/episode_{episode:06d}"


def episode_output(root: Path, episode: int) -> Path:
    path = Path(root) / episode_relative(episode)
    if not path.is_absolute() or path.resolve() != path or any(
        parent.is_symlink() for parent in (path, *path.parents)
    ):
        raise ValueError("Canonical output namespace required; no aliases")
    return path


def pin_path(code: Path, episode: int, role: str) -> Path:
    episode_relative(episode)
    roles = {'input', 'shared_prepare', 'shared_forward', 'shared_refined',
             'shared_export', 'historical_source', 'surface_mesh', 'solid_mesh'}
    if role not in roles:
        raise ValueError('Explicit known artifact pin role required')
    name = (f'{role}_{episode:06d}_pins.json' if role in {'surface_mesh', 'solid_mesh'}
            else f'cari_clip_{episode:06d}_{role}_pins.json')
    root = os.environ.get('WR_PIN_ROOT')
    if root is None:
        if output_prefix() != 'outputs':
            raise ValueError('An isolated experiment requires isolated pins')
        path = Path(code) / 'configs' / name
    else:
        prefix = output_prefix()
        expected = Path('/srv/scenesmith/world-reward') / Path(prefix).parent / 'pins'
        if prefix == 'outputs' or Path(root) != expected:
            raise ValueError('Pins must belong to the same explicit experiment')
        path = Path(root) / name
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Canonical readonly artifact pin path required')
    return path

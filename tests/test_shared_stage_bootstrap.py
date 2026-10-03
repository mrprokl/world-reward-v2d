"""Execute actual VM path bootstraps without site/numerical dependencies."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "run_cari_shared_prepare.sh", "run_cari_full_forward.sh",
    "run_cari_full_refine.sh", "run_cari_full_export.sh",
)


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("fault", [None, "episode", "source_hash"])
def test_vm_bootstrap_is_stdlib_only_and_preserves_public_pin_gates(tmp_path, wrapper, fault):
    text = (ROOT / "infra" / wrapper).read_text()
    blocks = re.findall(r"<<'PYPATHS'\n(.*?)\nPYPATHS", text, re.S)
    assert len(blocks) == 1
    pins = json.loads((ROOT / "configs/cari_clip_000015_input_pins.json").read_text())
    if fault == "source_hash":
        next(iter(pins["source_files"].values()))["sha256"] = "not-a-sha"
    pin_path = tmp_path / "inputs.json"
    pin_path.write_text(json.dumps(pins))
    episode = "16" if fault == "episode" else "15"
    arguments = [str(pin_path)]
    if wrapper == "run_cari_full_forward.sh":
        arguments.append(str(ROOT / "configs/cari_clip_000015_shared_prepare_pins.json"))
    arguments.append(episode)
    guard = '''import builtins
original = builtins.__import__
def import_stdlib(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'joblib', 'h5py', 'scipy', 'trimesh', 'pyarrow'}:
        raise AssertionError('Numerical/model imports must remain inside the pinned container')
    return original(name, *args, **kwargs)
builtins.__import__ = import_stdlib
'''
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "infra"), str(ROOT / "src"))),
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(["rtk", "proxy", sys.executable, "-S", "-c", guard + blocks[0], *arguments],
                            capture_output=True, text=True, env=environment)
    if fault is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == sorted(pins["source_files"])
        assert len(result.stdout.splitlines()) == 15
    else:
        assert result.returncode != 0
        assert result.stdout == ""
        assert "Numerical/model imports" not in result.stderr

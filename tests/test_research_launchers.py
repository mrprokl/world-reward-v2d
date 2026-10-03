"""Small export/bootstrap control closures, never Docker/network execution."""
import base64
import importlib.util
import lzma
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', ['run_research_export.sh', 'run_research_runtime.sh', 'run_research_import.sh'])
def test_research_wrapper_syntax_and_small_runtime_closure(script):
    subprocess.run(['bash', '-n', str(ROOT/'infra'/script)], check=True)
    spec=importlib.util.spec_from_file_location('research_launcher', ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ('infra','src','configs')
           for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    paths=module.runtime_bundle_paths(files,'infra/'+script)
    assert len(base64.b64encode(lzma.compress(b''.join(files[p] for p in paths)))) <= module.MAX_CODE_CONTROL_BYTES
    assert not any(p.startswith(('data/','weights/','validation/','docs/')) for p in paths)
    dependency='research_runtime_bootstrap.sh' if 'runtime' in script else 'research_transfer.py'
    assert 'infra/'+dependency in paths


def test_export_only_pinned_image_and_allowlisted_fixture_not_live_state():
    source=(ROOT/'infra/run_research_export.sh').read_text()
    assert 'docker image save' in source and 'docker image inspect' in source
    assert 'research_transfer.py' in source and '--include-private' in source
    assert 'b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7' in source
    for unsafe in ('docker export', 'docker stop', 'docker commit', 'tar -h', 'snapshot', 'scp '):
        assert unsafe not in source.removeprefix('#!/usr/bin/env bash\n# Task-only immutable export on VM01. Never snapshot Docker state/user disks.\n')
    assert 'nice -n 10 ionice -c 3' in source


def test_import_source_binding_before_extract_load_and_GPU_smoke():
    source=(ROOT/'infra/run_research_import.sh').read_text()
    assert source.index('Transferred artifact changed')<source.index('research_transfer.py')<source.index('docker image load')<source.index('docker run')
    assert '--export-sha256' in source and '671d10cacb146c1ef3c1edd22af155eb994a4122' in source
    assert 'runuser -u scenesmith' in source and '--network none' in source
    assert '--mount' not in source and 'model_inference_verified' in source
    assert 'docker image ls -q' in source and 'source_unit_restarted' in source
    assert subprocess.run(['bash',str(ROOT/'infra/run_research_import.sh'),'--unknown'],capture_output=True).returncode==2

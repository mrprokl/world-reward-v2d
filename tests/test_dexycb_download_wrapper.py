"""Code-only publication closure; no network or external archive fixtures."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_isolated_driver_and_sibling_are_in_publication_closure():
    spec = importlib.util.spec_from_file_location('dex_download_archive', ROOT / 'infra/azure_job.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    files = {str(path.relative_to(ROOT)): path.read_bytes()
             for folder in ('infra', 'src', 'configs') for path in (ROOT / folder).rglob('*')
             if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.toml', '.cpp')}
    files['pyproject.toml'] = (ROOT / 'pyproject.toml').read_bytes()
    selected = set(module.runtime_bundle_paths(files, 'infra/run_dexycb_download.sh'))
    assert {'infra/run_dexycb_download.sh', 'infra/dexycb_download.py', 'infra/dexycb_acquire.py',
            'configs/dexycb_identity_protocol.json'} <= selected
    wrapper = files['infra/run_dexycb_download.sh'].decode()
    assert "sys.path.insert(0, str(code / 'infra'))" in wrapper
    assert 'python3 -I -B' in wrapper and 'max_workers' not in wrapper
    assert 'infra/dexycb_acquire.py' in module.runtime_bundle_paths(files, 'infra/run_dexycb_acquire.sh')

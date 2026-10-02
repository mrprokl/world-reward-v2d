from pathlib import Path
import importlib.util
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name,stages',[
    ('hand',['render','masks','infer','convert','evaluate']),
    ('object',['render','observations','generate','evaluate'])])
def test_serial_pipeline_includes_exact_static_entrypoints(name,stages):
    source=(ROOT/f'infra/run_{name}_synthetic_pipeline.sh').read_text()
    assert 'set -euo pipefail' in source
    positions=[source.index(f'/infra/run_{name}_synthetic_{stage}.sh') for stage in stages]
    assert positions==sorted(positions)
    assert ' & ' not in source and 'eval_private' not in source
    bad=subprocess.run(['bash',str(ROOT/f'infra/run_{name}_synthetic_pipeline.sh'),'--unknown'],capture_output=True)
    assert bad.returncode==2

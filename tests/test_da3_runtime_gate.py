from pathlib import Path
import subprocess


def test_alias_after_real_imports_no_dependency_install_or_gpu():
    p=Path(__file__).resolve().parents[1]/'infra/run_da3_runtime_gate.sh';s=p.read_text()
    subprocess.run(['bash','-n',str(p)],check=True)
    assert '--gpus' not in s and 'pip install' not in s and '--network none' in s
    assert s.index('importlib.import_module')<s.index('docker image tag')
    assert 'source_manifest_sha256' in s and 'depth_anything_3.api' in s and 'n.startswith("evo.")' in s
    assert 'model_instantiated' in s and 'dependency_install_performed' in s

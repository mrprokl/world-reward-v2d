"""Source/ZIP-path gates only; no network, installed package or model."""
from pathlib import Path
import subprocess


def test_one_actual_pinned_wheel_no_resolver_install_or_source_mutation():
    root=Path(__file__).resolve().parents[1];p=root/'infra/run_da3_dependency_acquire.sh';s=p.read_text()
    subprocess.run(['bash','-n',str(p)],check=True)
    assert 'len(b)==3832' in s and '249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc' in s
    assert 'ca488d33c512d0b226142090af90e89ae266a901a293f89fd642dfec931e22c1' in s
    assert 'Requires-Dist:' in s and 'len(names)==7' in s and 'set(names)' in s
    assert 'pip install' not in s and 'extractall' not in s and "installed_into_image':False" in s
    for file in ['run_da3_runtime_gate.sh','run_da3_metric_infer.sh']:
        p=root/'infra'/file;s=p.read_text();subprocess.run(['bash','-n',str(p)],check=True)
        assert 'src=$WHEEL,dst=$WHEEL,readonly' in s and '249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc' in s

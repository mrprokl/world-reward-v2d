"""Serial research dispatch contract, no GPU or remote job."""
from pathlib import Path
import subprocess


def test_independent_gpu_serialization_and_existing_stage_sequence():
    path=Path(__file__).resolve().parents[1]/'infra/run_joint_rgb_chain.sh'
    source=path.read_text()
    assert 'inactive|failed) break' in source and '43200' in source
    assert 'outputs/' not in source and 'systemctl show "$UNIT"' in source
    for stage in ['render','masks','infer','evaluate']:
        assert f'run_joint_rgb_{stage}.sh"' in source
    subprocess.run(['bash','-n',str(path)],check=True)

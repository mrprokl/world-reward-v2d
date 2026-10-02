from pathlib import Path
import subprocess


def test_chain_actual_import_frozen_baseline_inference_then_private_eval():
    p=Path(__file__).resolve().parents[1]/'infra/run_da3_metric_validation.sh';s=p.read_text()
    subprocess.run(['bash','-n',str(p)],check=True)
    steps=['run_da3_runtime_gate.sh','run_tudl_prediction_import.sh',
           'run_da3_metric_infer.sh','run_da3_metric_evaluate.sh']
    assert [s.index(n) for n in steps]==sorted(s.index(n) for n in steps)
    assert '81e3226c43c6311f98c7a43f704b80a9dc69ce4bb0d233d971ffc6c88f281881' in s
    assert 'run_tudl_infer.sh' not in s and 'set -euo pipefail' in s

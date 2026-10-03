import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("scoped_replay_test",ROOT/"infra/photometric_scoped_replay.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_preregistered_distinct_h100c_mode_and_failure_pin(gate):
    assert gate.OUT=="validation/photometric_native_v1/capability_scoped_v1"and gate.BUDGET==180
    assert gate.FAIL_SHA=="110eebee952eb1fcd2c6db4b359a1756831ff1ad1d4549f8807f4f62c6a0e11c"
    text=Path(gate.__file__).read_text()
    assert "stack.enter_context(execution.strict_native_head"in text and "native_model_loader=load"in text
    assert "deterministic_algorithms=False"in text and "hidden_warmup_calls=0"in text
    assert "exact_SHAM_gate_relaxed=False"in text


def test_pinned_prior_inputs_required_before_cuda(gate,tmp_path,monkeypatch):
    monkeypatch.setattr(gate.prior,"historical_inputs",lambda *_:(_ for _ in()).throw(ValueError("historical mismatch")))
    with pytest.raises(ValueError,match="historical mismatch"):gate.public_inputs(tmp_path,"a"*40)


def test_original_module_core_default_still_strict(gate):
    import inspect
    signature=inspect.signature(gate.core.run_body)
    assert signature.parameters["deterministic_algorithms"].default is True
    assert signature.parameters["native_model_loader"].default is None


def test_scoped_wrapper_mounts_only_failure_receipts_not_arrays(gate):
    p=ROOT/"infra/run_photometric_scoped_replay.sh";s=p.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(p)],check=True)
    assert s.count("docker run")==1 and "183s docker run"in s
    assert 'src=$BASE/capability_replay_v1/report.json,dst=$BASE/capability_replay_v1/report.json,readonly'in s
    assert "eval_private"not in s and "weights/mhr"not in s and 'src=$BASE,dst='not in s
    assert '"$CODE/infra/photometric_scoped_replay.py"'in s
    assert subprocess.run(["rtk","proxy","bash",str(p),"bad"],capture_output=True).returncode==2

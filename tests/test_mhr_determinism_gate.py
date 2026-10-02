"""Pure tiny raw-bit/planning tests; no torch, assets, CUDA or large fixtures."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/mhr_determinism_gate.py"
    spec = importlib.util.spec_from_file_location("wr_test_mhr_determinism", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exact_failed_semantic_fixture_and_explicit_rows(gate):
    identity, controls, expression = gate.own_inputs()
    assert identity.shape == (216, 45) and expression.shape == (216, 72) and controls.shape == (216, 204)
    assert all(v.dtype == np.float32 for v in (identity, controls, expression))
    assert np.count_nonzero(identity) == np.count_nonzero(expression) == 0
    assert np.count_nonzero(controls) == 216
    for i in range(54):
        assert np.array_equal(controls[i*4:(i+1)*4, 68+i], np.array([-.001, .001, -.002, .002], np.float32))
    assert np.count_nonzero(controls[:, :68]) == np.count_nonzero(controls[:, 122:]) == 0


def pair(): return (np.zeros((2, 3), np.float32), np.ones((2, 8), np.float32))


def test_bitexact_replay_zero_tolerance(gate):
    first = pair(); result = gate.bit_comparison(first, tuple(v.copy() for v in first))
    assert result["bitexact"] is True and result["vertices"]["raw_byte_differences"] == 0
    replay = pair(); replay[1][0, 0] = np.nextafter(np.float32(1), np.float32(2))
    result = gate.bit_comparison(first, replay)
    assert result["bitexact"] is False and result["skeleton"]["raw_element_differences"] == 1
    assert result["skeleton"]["max_absolute_difference"] > 0


def test_negative_zero_not_hidden_by_numeric_equality(gate):
    first, replay = pair(), pair(); replay[0][0, 0] = -0.
    result = gate.bit_comparison(first, replay)
    assert result["bitexact"] is False and result["vertices"]["max_absolute_difference"] == 0


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "inf", "missing"])
def test_replay_failure_not_repaired(gate, fault):
    first, replay = pair(), list(pair())
    if fault == "shape": replay[0] = replay[0][:1]
    elif fault == "dtype": replay[0] = replay[0].astype(np.float64)
    elif fault == "nan": replay[0][0, 0] = np.nan
    elif fault == "inf": replay[1][0, 0] = np.inf
    else: replay.pop()
    with pytest.raises(ValueError): gate.bit_comparison(first, replay)


def complete(name):
    return {"name": name, "status": "complete", "conditions": [
        {"apply_correctives": corrective, "replays": [{"bitexact": True}, {"bitexact": True}]} for corrective in (False, True)]}


def test_only_strict_gpu_and_cpu_pass_default_instability_descriptive(gate):
    groups = [dict(complete("default_cuda"), status="fail"), complete("strict_cuda"), complete("strict_cpu")]
    assert gate.outcome(groups) is True
    groups[1]["conditions"][0]["replays"][0]["bitexact"] = False
    assert gate.outcome(groups) is False


@pytest.mark.parametrize("fault", ["missing_gpu", "missing_cpu", "unsupported", "condition", "repeat", "unknown", "duplicate_correctives"])
def test_incomplete_or_unsupported_never_pass(gate, fault):
    groups = [complete("strict_cuda"), complete("strict_cpu")]
    if fault == "missing_gpu": groups.pop(0)
    elif fault == "missing_cpu": groups.pop()
    elif fault == "unsupported": groups[0]["status"] = "fail"
    elif fault == "condition": groups[0]["conditions"].pop()
    elif fault == "repeat": groups[1]["conditions"][0]["replays"].pop()
    elif fault == "duplicate_correctives": groups[0]["conditions"][1]["apply_correctives"] = False
    else: groups[1]["conditions"][0]["replays"][0]["bitexact"] = None
    assert gate.outcome(groups) is False


def test_actual_scripted_source_hashes_not_claimed_generic_lbs(gate):
    class Methods:
        def _method_names(self): return ["forward"]
    class Module:
        def __init__(self, code): self.code = code; self._c = Methods()
    class Model(Module):
        def named_modules(self): return [("", self), ("character_torch.lbs", Module("return torch.index_add(x, 0, i, v)")), ("unused", Module("return x"))]
    evidence = gate.script_sources(Model("def forward(): return self.character_torch()"))
    assert [v["module"] for v in evidence] == ["", "character_torch.lbs"]
    assert evidence[1]["index_add_occurrences"] == 1 and len(evidence[1]["code_sha256"]) == 64
    assert evidence == gate.script_sources(Model("def forward(): return self.character_torch()"))
    assert evidence != gate.script_sources(Model("def forward(): return other()"))


def test_failed_integrity_preserves_exclusive_report_before_torch(gate, tmp_path, monkeypatch):
    (tmp_path/"results").mkdir()
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="model mismatch"): gate.main([])
    output = tmp_path/"results/mhr-determinism.json"; frozen = output.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["forward_calls"] == 0
    assert report["known_failed_fixture"] is True and report["hypothesis_validation"] is False
    assert report["adoption_performed"] is False
    with pytest.raises(FileExistsError): gate.main([])
    assert output.read_bytes() == frozen


def test_no_cli_tuning_offline_remote_and_frozen_numeric_settings(gate):
    with pytest.raises(SystemExit): gate.main(["--tolerance", "1e-5"])
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_mhr_determinism_gate.sh").read_text()
    assert source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"') < source.index("import torch")
    assert "torch.use_deterministic_algorithms(strict, warn_only=False)" in source
    assert "allow_tf32 = False" in source and "cudnn.benchmark = False" in source
    assert "handle.seek(0)" in source and "handle.truncate()" in source and "os.fsync" in source
    assert "123s docker run" in wrapper and "--network none" in wrapper and "--memory 16g" in wrapper
    assert "weights/mhr,readonly" in wrapper and '"$IMAGE" python' in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor", "validation", "weights/cari4d"))
    assert gate.TOTAL_SECONDS == 120 and gate.MAX_CALLS == 20

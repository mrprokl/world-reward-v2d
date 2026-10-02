"""Gate planning/immutability only; actual tensors and timing are Azure-only."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "infra/continuous_surface_gate.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("world_reward_test_continuous_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote(monkeypatch, gate):
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("/sys/class/net/lo")]) if str(p) == "/sys/class/net" else original(p))


def test_keeps_frozen_original_small_triangle_and_tolerances(gate):
    points, triangles, distance, gradient = gate.analytic_fixture()
    assert triangles.dtype == np.float32
    assert triangles[0, 1, 0] == pytest.approx(.01)
    assert distance[-1] == pytest.approx(.0004125)
    assert gradient.shape == points.shape
    assert gate.DISTANCE_ATOL_M2 == 1e-9 and gate.GRADIENT_ATOL == 1e-6
    assert gate.MAX_GATE_SECONDS == 30 and gate.MAX_ITERATION_SECONDS == 3


def test_frozen_report_fails_before_torch_import(gate, monkeypatch, tmp_path):
    remote(monkeypatch, gate)
    output = tmp_path / "results/continuous-surface-gate.json"
    output.parent.mkdir()
    output.write_text("frozen")
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    with pytest.raises(FileExistsError, match="frozen"):
        gate.main([])
    assert output.read_text() == "frozen"


@pytest.mark.parametrize("revision", ["", "main", "a" * 39, "A" * 40, "a" * 41])
def test_invalid_revision_fails_before_tensor_backend(gate, monkeypatch, revision):
    remote(monkeypatch, gate)
    monkeypatch.setenv("WR_CODE_REVISION", revision)
    with pytest.raises(RuntimeError, match="immutable"):
        gate.main([])


@pytest.mark.parametrize("argv", [["--max-seconds", "1000"], ["--tolerance", ".1"], ["--backend", "kaolin"], ["--episode", "15"]])
def test_gate_has_no_tuning_or_challenge_controls(gate, argv):
    with pytest.raises(SystemExit):
        gate.main(argv)


def test_wrapper_no_data_models_cache_or_vendor_mount(gate):
    wrapper = Path(gate.__file__).with_name("run_continuous_surface_gate.sh").read_text()
    assert "--network none" in wrapper and "--gpus all" in wrapper
    assert "src=$CODE,dst=$CODE,readonly" in wrapper and "WR_CODE_REVISION" in wrapper
    for name in ("data", "outputs", "weights", "vendor", "cache"):
        assert f"src=$ROOT/{name}" not in wrapper


def test_analytic_dtype_and_first_order_contract_not_vendor_gate(gate):
    source = inspect.getsource(gate._analytic)
    assert "dtype=torch.float64" in source and "duplicated.grad[1]" in source
    assert "pytorch3d" not in source and "kaolin" not in source
    assert "distance.sum().backward()" in source and "tri.grad" in source

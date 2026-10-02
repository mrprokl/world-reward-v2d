"""Tiny metadata-only fake modules; never load or run actual MHR."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def driver(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "infra/mhr_metadata_inventory.py"
    spec = importlib.util.spec_from_file_location("wr_test_mhr_inventory", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fake(names=None):
    model = SimpleNamespace(joint_names=names, _c=SimpleNamespace(_method_names=lambda: ["forward"],
        _get_method=lambda name: SimpleNamespace(schema="forward(Tensor)->Tensor")))
    model.named_modules = lambda: [("", model)]
    model.named_buffers = lambda: [("buffer", SimpleNamespace(shape=(127, 3), dtype="float32"))]
    return model


def test_inventory_shapes_and_actual_names_only_without_forward(driver):
    model = fake(["root", "l_wrist"]); model.parameter_names = ["root_tx"]
    result = driver.inventory(model)
    assert result["joint_names_available"] and result["parameter_names_available"]
    assert result["buffers"] == [{"name": "buffer", "shape": [127, 3], "dtype": "float32"}]
    assert result["methods"][0]["name"] == "forward"


@pytest.mark.parametrize("names", [None, [], [1], ["x"] * 513])
def test_unexposed_or_invalid_metadata_never_assumed(driver, names):
    result = driver.inventory(fake(names))
    assert result["joint_names_available"] is False and result["parameter_names_available"] is False


def test_schema_truncation_is_explicit(driver):
    model = fake(); model._c._get_method = lambda name: SimpleNamespace(schema="x" * 5000)
    result = driver.inventory(model)
    assert result["methods"][0]["schema_truncated"] and len(result["methods"][0]["schema"]) == 4096


def test_driver_and_wrapper_no_forward_or_gpu_or_heavy_other_inputs(driver):
    source = Path(driver.__file__).read_text()
    assert 'map_location="cpu"' in source and "model(" not in source and ".forward(" not in source
    wrapper = Path(driver.__file__).with_name("run_mhr_metadata_inventory.sh").read_text()
    assert "--network none --memory 4g --cpus 2" in wrapper and "--gpus" not in wrapper
    assert "src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" in wrapper
    assert "120s docker run" in wrapper and '"$IMAGE" python' in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor"))

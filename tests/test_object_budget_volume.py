import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def driver(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('wr_object_volume',infra/'object_budget_volume.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_backend_reuses_exact_export_packing_with_own_prerequisites(driver):
    source=Path(driver.__file__).read_text()
    assert 'check_prerequisites=prerequisites,simplify=volume.simplify' in source
    assert 'source_shell_volume_relative_limit' in source and 'adoption_performed' in source
    assert 'object_budget_guarded as proposal' in source
    with pytest.raises(SystemExit):driver.main(['--episode','1'])

def test_output_only_writable_and_old_failures_readonly(driver):
    wrapper=Path(driver.__file__).with_name('run_object_budget_volume.sh').read_text()
    assert '--gpus' not in wrapper and '903s docker run' in wrapper
    assert 'src=$BASE,dst=$BASE,readonly' in wrapper and 'src=$OUT,dst=$OUT"' in wrapper
    assert 'src=$ROOT/outputs,dst=' not in wrapper
    assert 'mesh_volume_qem.cpp' in wrapper and 'mesh_guarded_qem.cpp' in wrapper

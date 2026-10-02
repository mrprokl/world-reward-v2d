import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def driver(monkeypatch):
 p=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(p))
 s=importlib.util.spec_from_file_location('wr_motion_gen',p/'object_motion_generate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_no_private_inputs_shape_repair_or_multiview_repeat(driver):
 s=Path(driver.__file__).read_text();w=Path(driver.__file__).with_name('run_object_motion_generate.sh').read_text()
 assert 'native.build_pipeline(root,report,torch)' in s and 'native.export_actual' in s
 assert 'run_multi_view' not in s and 'eval_private' not in w and 'src=$BASE/inputs' not in w
 assert 'stage1_inference_steps=50,stage2_inference_steps=25' in s
 assert 'with_layout_postprocess=False' in s and '243s docker run' in w
 assert 'src=$BASE/observations,dst=$BASE/observations,readonly' in w
 assert 'outputs_completed' in s and 'private_truth_read' in s

def test_manifest_missing_fails_before_models(driver,tmp_path):
 with pytest.raises(FileNotFoundError):driver.load_anchors(tmp_path)
 with pytest.raises(SystemExit):driver.main(['--oracle'])

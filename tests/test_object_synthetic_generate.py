import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def generate(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'infra'))
    s=importlib.util.spec_from_file_location('wr_test_object_generate',root/'infra/object_synthetic_generate.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def observation():
    p=np.zeros((3,384,512),np.float32);p[2]=2
    return {'rgb':np.zeros((384,512,3),np.uint8),'mask':np.ones((384,512),bool),'pointmap':p,
            'K':np.array([[640,0,256],[0,640,192],[0,0,1]],float),'object_index':np.asarray(0),'frame_index':np.asarray(2)}

@pytest.mark.parametrize('fault',['nan','shape','grid','dtype','empty','view','K'])
def test_observations_fail_closed(generate,fault):
    x=observation();generate.observation_arrays(x,0,2)
    if fault=='nan':x['pointmap'][0,0,0]=np.nan
    if fault=='shape':x['pointmap']=x['pointmap'].transpose(1,2,0)
    if fault=='grid':x['rgb']=x['rgb'][:-1]
    if fault=='dtype':x['mask']=x['mask'].astype(int)
    if fault=='empty':x['mask'][:]=False
    if fault=='view':x['frame_index']=np.asarray(4)
    if fault=='K':x['K'][0,0]=600
    with pytest.raises(ValueError):generate.observation_arrays(x,0,2)

def test_sha_frozen_paths(generate,tmp_path):
    p=tmp_path/'tiny';p.write_text('x');digest=generate.sha256(p);generate.require_hash(p,digest)
    q=tmp_path/'symlink';q.symlink_to(p)
    with pytest.raises(ValueError):generate.require_hash(q,digest)

def test_exact_generation_configuration_and_actual_raw_parity_no_tautology(generate):
    source=Path(generate.__file__).read_text()
    assert "raw=result['mesh'][0]" in source and 'compose_transform(s,Q,t).transform_points' in source
    assert 'vertices@gauge.A' in source and 'torch.equal(s,s[:,0:1].expand_as(s))' in source
    assert 'stage1_inference_steps=50,stage2_inference_steps=25' in source
    assert 'ss_weighting=False,weighting_config=None' in source
    assert 'with_layout_postprocess=False' in source and 'for mode in (\'single\',\'three_view\')' in source
    wrapper=Path(generate.__file__).with_name('run_object_synthetic_generate.sh').read_text()
    assert 'eval_private' not in wrapper and 'src=$BASE/inputs' not in wrapper
    assert 'src=$BASE/observations,dst=$BASE/observations,readonly' in wrapper and '--network none' in wrapper

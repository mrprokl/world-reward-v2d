import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def driver(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    s=importlib.util.spec_from_file_location('wr_object_guarded',infra/'object_budget_guarded.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def candidate():return np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]]),np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])

def test_export_f32_weld_reorders_birth_exact(driver):
    v,f=candidate();v[0,0]+=1e-8;birth=np.array([9,8,7,6]);original={'I':birth.tolist(),'J':[10,11,12,13]}
    ev,inverse=np.unique(v.astype(np.float32).astype(np.float64),axis=0,return_inverse=True);ef=inverse[f]
    r=driver.export_birth_mapping((v,f),(ev,ef),original)
    assert r['J']==original['J'] and np.array_equal(np.array(r['I'])[inverse],birth)
    assert original['I']==birth.tolist()

@pytest.mark.parametrize('fault',['faceflip','perturb','merge'])
def test_export_hidden_change_or_quantized_vertex_merge_fails(driver,fault):
    v,f=candidate();mapping={'I':[0,1,2,3],'J':[0,1,2,3]}
    if fault=='merge':v=np.r_[v,v[:1]+[1e-8,0,0]];f=np.r_[f,[[4,1,2]]];mapping['I']+=[4];mapping['J']+=[4]
    ev,inverse=np.unique(v.astype(np.float32).astype(np.float64),axis=0,return_inverse=True);ef=inverse[f]
    if fault=='faceflip':ef=ef[:,::-1]
    if fault=='perturb':ev[0,0]+=1e-6
    with pytest.raises(ValueError):driver.export_birth_mapping((v,f),(ev,ef),mapping)

def test_selected_episode_requires_guarded_control_and_no_endpoint_gate_call(driver):
    source=Path(driver.__file__).read_text()
    assert 'endpoint.experiment_gate' not in source and 'endpoint_simplify' not in source
    assert "'new_close_asymmetric_shells','new_disconnected_smooth_asymmetric'" in source
    assert 'guarded.mapped_geometry' in source and 'source_intersections' in source
    assert 'scale_or_pose_fitted' not in source
    with pytest.raises(SystemExit):driver.main([])
    with pytest.raises(SystemExit):driver.main(['--episode','30'])

def test_wrapper_only_output_writable_oldproposal_immutable(driver):
    s=Path(driver.__file__).with_name('run_object_budget_guarded.sh').read_text()
    assert '--gpus' not in s and '903s docker run' in s
    assert 'src=$BASE,dst=$BASE,readonly' in s
    assert 'src=$ROOT/outputs,dst=$ROOT/outputs' not in s
    assert 'src=$OUT,dst=$OUT"' in s and 'chown -R' not in s
    assert 'object_budget_endpoint' not in s and 'mesh_guarded_qem.cpp' in s

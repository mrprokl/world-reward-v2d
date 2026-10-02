from pathlib import Path
import importlib.util
import numpy as np
import pytest

@pytest.fixture
def evaluate():
    p=Path(__file__).resolve().parents[1]/'infra/object_synthetic_evaluate.py'
    s=importlib.util.spec_from_file_location('wr_object_evaltest',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def tetra():return np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]]),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])

def test_samples_deterministic_without_geometry_mutation(evaluate):
    v,f=tetra();a=evaluate.surface_samples(v,f);b=evaluate.surface_samples(v,f)
    assert np.array_equal(a,b) and a.shape==(18439,3)
    assert np.all(a>=0) and np.all(a.sum(1)<=1+1e-12)

def test_camera_chamfer_no_alignment_sum_not_half(evaluate):
    v,f=tetra();v*=.01
    zero=evaluate.metrics(v,f,v,f);assert zero['chamfer_sum_cm']==0
    moved=evaluate.metrics(v+[1.,0,0],f,v,f)
    assert moved['chamfer_sum_cm']>190 and moved['chamfer_sum_cm']==pytest.approx(moved['pred_to_truth_cm']+moved['truth_to_pred_cm'])

def case(before,after):
    def x(value):return {'chamfer_sum_cm':value,'watertight':True,'winding_consistent':True,'signed_volume_camera_m3':1.}
    return {'single':x(before),'three_view':x(after)}

def test_predeclared_gain_per_object_regression_and_geometry(evaluate):
    assert evaluate.decision([case(10,9),case(20,18)])['synthetic_hypothesis_supported']
    assert not evaluate.decision([case(10,11),case(20,16)])['synthetic_hypothesis_supported']
    a=[case(10,9),case(20,18)];a[0]['three_view']['watertight']=False
    assert not evaluate.decision(a)['synthetic_hypothesis_supported']

@pytest.mark.parametrize('fault',['nan','integer','badindex','badshape'])
def test_geometry_invalid_no_repair(evaluate,fault):
    v,f=tetra()
    if fault=='nan':v[0,0]=np.nan
    if fault=='integer':v=v.astype(int)
    if fault=='badindex':f[0,0]=9
    if fault=='badshape':f=f[:,:2]
    with pytest.raises(ValueError):evaluate.mesh_arrays(v,f)

def test_private_eval_no_model_gpu_fit_and_truth_readonly(evaluate):
    wrapper=Path(evaluate.__file__).with_name('run_object_synthetic_evaluate.sh').read_text()
    assert '--gpus' not in wrapper and 'weights' not in wrapper
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper and '--network none' in wrapper

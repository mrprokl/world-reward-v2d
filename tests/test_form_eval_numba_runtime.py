"""Tiny runtime safety/operator contracts; no Docker/GPU/model/dataset work."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import form_eval_numba_runtime as runtime


def test_frozen_cpu_wheels_platform_source_and_new_image_names():
    assert [row['version'] for row in runtime.WHEELS] == ['0.60.0','0.43.0']
    assert sum(row['bytes'] for row in runtime.WHEELS) == 47576799
    assert all('cp311-cp311-manylinux' in row['filename'] for row in runtime.WHEELS)
    assert all(row['url'].startswith('https://files.pythonhosted.org/') for row in runtime.WHEELS)
    assert runtime.target_name('a'*40) == 'world-reward/form-eval-numba:'+'a'*40
    for wrong in ('latest','a'*39,42):
        with pytest.raises(ValueError):runtime.target_name(wrong)


def test_dockerfile_offline_isolated_no_base_numpy_torch_mutation():
    value=runtime.child_dockerfile()
    assert value.startswith('FROM '+runtime.BASE+'\n')
    assert '--no-index --no-deps' in value and '--target '+runtime.SITE in value
    assert 'numpy.__version__' in value and 'torch.__version__' in value
    assert 'pip install numpy' not in value and 'pip install torch' not in value
    assert 'rm -rf /wheels' in value and 'ENV PYTHONPATH='+runtime.SITE in value


def pypi(row):
    return json.dumps(dict(info=dict(name=row['name'],version=row['version'],license='BSD'),
        urls=[dict(filename=row['filename'],url=row['url'],size=row['bytes'],
                   digests=dict(sha256=row['sha256']),yanked=False)])).encode()


def test_primary_metadata_pins_never_choose_dynamic_latest():
    row=runtime.WHEELS[0]
    receipt=runtime.validate_pypi(pypi(row),row)
    assert receipt['license']=='BSD' and len(receipt['sha256'])==64
    for field,value in (('size',1),('yanked',True),('url','https://evil.invalid/wheel')):
        bad=json.loads(pypi(row));bad['urls'][0][field]=value
        with pytest.raises(ValueError):runtime.validate_pypi(json.dumps(bad),row)
    bad=json.loads(pypi(row));bad['info']['license']='unknown'
    with pytest.raises(ValueError):runtime.validate_pypi(json.dumps(bad),row)


def test_packaged_license_notices_required(tmp_path):
    path=tmp_path/'wheel.whl'
    with zipfile.ZipFile(path,'w') as z:z.writestr('foo.dist-info/LICENSE','BSD LICENSE fixture')
    notices=runtime.wheel_notices(path)
    assert len(notices)==1 and notices['foo.dist-info/LICENSE']['bytes']==19
    path=tmp_path/'missing.whl'
    with zipfile.ZipFile(path,'w') as z:z.writestr('foo.dist-info/METADATA','Name: foo')
    with pytest.raises(ValueError,match='license notices'):runtime.wheel_notices(path)


def test_analytic_controls_include_inside_outside_boundaries_reversed_degenerate_parallel():
    cases=runtime.procedural_cases();assert len(cases)==5
    assert cases[0][0]=='cube' and cases[1][0]=='tetra' and len(cases[2][1])==2048
    np.testing.assert_allclose(cases[0][4],[1,.25,.1,0,0,0,0,0],atol=1e-15,rtol=0)
    np.testing.assert_allclose(cases[1][4],[.1,.2,.1/np.sqrt(3),0,0,0])
    np.testing.assert_array_equal(cases[3][3],cases[0][3][:,::-1])
    assert len(cases[4][3])==len(cases[0][3])+1


def test_parity_uses_both_unchanged_paths_restores_kernel_and_checks_analytic_truth():
    kernel=object(); calls=[]
    module=SimpleNamespace(_PENETRATION_KERNELS=kernel)
    def depth(points,vertices,faces):
        calls.append(module._PENETRATION_KERNELS)
        if len(vertices)==4:
            inside=(points>=0).all(axis=1)&(points.sum(axis=1)<=1)
            result=np.maximum(0,np.minimum(points.min(axis=1),(1-points.sum(axis=1))/np.sqrt(3)))
            return np.where(inside,result,0.)
        return np.maximum(0.,1-np.abs(points).max(axis=1))
    module.penetration_depth=depth
    rows=runtime.depth_parity(module)
    assert len(rows)==5 and calls==[kernel,None]*5 and module._PENETRATION_KERNELS is kernel
    assert all(row['byte_exact_numpy'] for row in rows)
    original=depth
    module.penetration_depth=lambda *args:original(*args)+.001
    with pytest.raises(ValueError,match='analytic'):runtime.depth_parity(module)
    assert module._PENETRATION_KERNELS is kernel


def test_parity_missing_compiler_is_not_successful_fallback_qualification():
    with pytest.raises(ValueError,match='must be active'):
        runtime.depth_parity(SimpleNamespace(_PENETRATION_KERNELS=None))


def test_all_declared_runtime_helpers_wrapper_no_gpu_or_data_mounts():
    root=Path(__file__).resolve().parents[1]
    assert all((root/name).is_file() for name in runtime.HELPERS)
    shell=(root/'infra/run_form_eval_numba_runtime.sh').read_text()
    assert 'world-reward-ncc-h100-02' in shell and 'env -i' in shell
    source=(root/'infra/form_eval_numba_runtime.py').read_text()
    assert "'--network=none','--read-only'" in source
    assert '--gpus' not in source and 'CUDA_VISIBLE_DEVICES=-1' in source
    assert 'form_hoi_external_dev_v1' not in source and 'mhr_model.pt' not in source
    assert "'compiled_public_kernels_active'" in source


def test_seals_exclusive_bounded_readonly_json(tmp_path):
    out=tmp_path/'report.json';runtime.seal(out,dict(status='pass'))
    assert not out.stat().st_mode&0o222
    with pytest.raises(ValueError):runtime.seal(out,dict(status='other'))

"""No Docker/services/real binaries; sealed proof and scoped restoration tests."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'infra'))
spec = importlib.util.spec_from_file_location('raster_prefix_activation_test', ROOT/'infra/raster_prefix_activation.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def sealed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)); path.chmod(0o444)
    return module.identity(path)


def fixtures(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    rev = 'a'*40; child = 'sha256:'+'b'*64
    camera = types.ModuleType('camera_render'); camera._raster_source_pins = lambda: {'cam': {'bytes': 1, 'sha256': 'c'*64}}
    monkeypatch.setitem(sys.modules, 'camera_render', camera)
    original_pin = dict(bytes=3, sha256='f'*64); binary_pin = dict(bytes=9, sha256='e'*64)
    rows = [dict(case=name, batch_size=b, exact_mask_parity=True, depth_atol_m=1e-6, depth_rtol=0.,
        camera_z_max_error_m=0., image_size_hw=[1152, 1536], native_capacity=dict(native_NDC_transform=True))
        for name in ('procedural_random_multiobject_occlusion', 'actual_first_external_object') for b in (1, 4)]
    qa = dict(schema='world_reward.raster_prefix_qualification.v1', status='pass',
        decision='INFERENCE_PREFIX_QUALIFIED_NOT_ADOPTED', producer_revision=rev, image_id=child,
        source_revision='33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba', original_binary=original_pin,
        candidate_binary=binary_pin, geometry_deleted=False, grid_resized=False, model_or_RGB_loaded=False,
        truth_read=False, adopted=False, source_pins=camera._raster_source_pins(), rows=rows,
        prefix_invariant=dict(actual_coarse_valid_prefix=True, capacity_nonoverflow=True, original_native_coarse_used=True),
        fixed_bin_kernel_gate=dict(coarse_inputs_shared=True, fine_all_outputs_bit_equal=True,
            one_pixel_cotangent_backward_bit_equal=True, broad_training_validation=False))
    path = tmp_path/'results'/('raster-prefix-runtime-'+rev)/'report.json'
    qpin = sealed(path.parent/'qualification.json', qa)
    rt = dict(schema='world_reward.raster_prefix_runtime.v1', status='pass',
        decision='ISOLATED_PREFIX_RUNTIME_QUALIFIED_NOT_ADOPTED', producer_revision=rev, qualified_image_id=child,
        base_image_id=module.BASE, one_line_only_patch=True, base_modified=False, packages_installed=False,
        models_loaded=False, reference_inputs_read=False, scientific_adoption=False, GPU_build=False,
        source_before={'helpers': camera._raster_source_pins()}, source_after={'helpers': camera._raster_source_pins()},
        qualification_pin=qpin, preflight=dict(original_binary=original_pin))
    return path, rt, qa, sealed(path, rt)


def test_admission_exact_paired_pass_current_base_and_source(monkeypatch, tmp_path):
    path, rt, qa, pin = fixtures(monkeypatch, tmp_path)
    a, b, pins, paths = module.qualification(path, pin, current_image=module.BASE)
    assert a == rt and b == qa and pins['runtime_report'] == pin and len(paths) == 2


@pytest.mark.parametrize('bad', ['failed', 'image', 'source', 'prefix', 'original', 'rows'])
def test_incomplete_or_incompatible_qualification_rejected(monkeypatch, tmp_path, bad):
    path, rt, qa, pin = fixtures(monkeypatch, tmp_path)
    if bad == 'failed': rt['status'] = 'fail'
    if bad == 'image': qa['image_id'] = module.BASE
    if bad == 'source': qa['source_pins'] = {}
    if bad == 'prefix': qa['prefix_invariant']['capacity_nonoverflow'] = False
    if bad == 'original': qa['original_binary'] = {}
    if bad == 'rows': qa['rows'] = qa['rows'][:3]
    qpath = path.parent/'qualification.json'; qpath.chmod(0o600); qpath.unlink()
    rt['qualification_pin'] = sealed(qpath, qa)
    path.chmod(0o600); path.unlink(); pin = sealed(path, rt)
    with pytest.raises(ValueError): module.qualification(path, pin, current_image=module.BASE)


def test_failed_activation_wrong_pin_or_other_base_rejected(monkeypatch, tmp_path):
    path, _, _, pin = fixtures(monkeypatch, tmp_path)
    with pytest.raises(ValueError): module.qualification(path, pin, current_image='different')
    with pytest.raises(ValueError): module.qualification(path, dict(pin, bytes=1), current_image=module.BASE)


def test_proxy_changes_only_forward_every_other_operator_and_backward_original():
    original = types.SimpleNamespace(rasterize_meshes=lambda: 'old', rasterize_meshes_backward=lambda: 'grad',
        _rasterize_meshes_coarse=lambda: 'coarse', other=lambda: 'unrelated')
    candidate = types.SimpleNamespace(rasterize_meshes=lambda: 'fast', rasterize_meshes_backward=lambda: 'wrong')
    proxy = module.RasterOnlyProxy(original, candidate)
    assert proxy.rasterize_meshes() == 'fast'
    assert proxy.rasterize_meshes_backward() == 'grad' and proxy.other() == 'unrelated'
    assert proxy._rasterize_meshes_coarse() == 'coarse'


def test_default_and_full_reference_bypass_never_import_load_or_admit(monkeypatch):
    monkeypatch.setattr(module, 'activation', lambda *_a, **_k: pytest.fail('admission must be bypassed'))
    with module.scoped_raster(current_image='unused') as active: assert active is False
    with module.scoped_raster('irrelevant', {}, current_image='unused', full_face_capacity=True) as active:
        assert active is False


def test_scoped_and_nested_restore_even_on_exception(monkeypatch, tmp_path):
    rev = 'a'*40; original = types.ModuleType('pytorch3d._C'); original.__file__ = str(tmp_path/'original.so')
    target = types.SimpleNamespace(_C=original); pkg = types.ModuleType('pytorch3d'); pkg.__path__ = []
    pkg._C = original
    monkeypatch.setitem(sys.modules, 'pytorch3d', pkg); monkeypatch.setitem(sys.modules, 'pytorch3d._C', original)
    key = 'wr_raster_prefix_'+rev+'._C'; candidate = types.SimpleNamespace(rasterize_meshes=lambda: 'fast')
    monkeypatch.setitem(sys.modules, key, candidate)
    proof = dict(report={'runtime_revision': rev}, qualification={'original_binary': {}}, binary=tmp_path/'candidate.so')
    monkeypatch.setattr(module, 'activation', lambda *_a, **_k: proof)
    monkeypatch.setattr(module, 'identity', lambda *_a, **_k: {})
    monkeypatch.setattr(module.importlib, 'import_module', lambda _: target)
    with pytest.raises(RuntimeError):
        with module.scoped_raster('path', {}, current_image=module.BASE):
            first = target._C
            assert isinstance(first, module.RasterOnlyProxy)
            with module.scoped_raster('path', {}, current_image=module.BASE): assert target._C is not first
            assert target._C is first
            raise RuntimeError('test')
    assert target._C is original


def test_only_single_binary_export_no_start_no_gpu_or_overlay():
    src = (ROOT/'infra/raster_prefix_activation.py').read_text()
    assert "['cp', cid+':'+CHILD_FILE" in src
    assert "['create', '--name'" in src and "['rm', cid]" in src
    assert "'--network=none'" in src
    assert 'docker start' not in src and '--gpus' not in src and 'pip install' not in src

"""Manufactured label-free CPU/strict fake-device controls; no native Torch."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from world_reward import coherent_pair_partition as partition
from world_reward import coherent_route_scorer as core

spec = importlib.util.spec_from_file_location('partition_original_controls',
    Path(__file__).with_name('test_coherent_pair_marginal_objective.py'))
controls = importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)
THETA = controls.THETA


def run(score=None, *, theta=THETA, alpha=0., arm='A', backend='numpy'):
    if score is None:
        score = controls.scores(theta, alpha)
    return partition.marginal_partition(score, theta, alpha=alpha, arm=arm, backend=backend)


def fake_backend(monkeypatch):
    # Extend the existing tiny backend's transport only; never import Torch.
    cls = controls.Tensor
    monkeypatch.setattr(cls, 'detach', lambda self: self, raising=False)
    monkeypatch.setattr(cls, 'cpu', lambda self: self, raising=False)
    monkeypatch.setattr(cls, 'numpy', lambda self: self.array, raising=False)
    monkeypatch.setattr(cls, '__gt__', lambda self, x: cls(self.array > x), raising=False)
    monkeypatch.setitem(sys.modules, 'torch', controls.fake_torch())


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', 0.), ('B', .3125)])
def test_independent_log_partition_and_vjp_not_extra_temperature(arm, alpha):
    score = controls.scores(alpha=alpha); out = run(score, arm=arm, alpha=alpha)
    x = getattr(score, 'scores_'+arm.lower()).ravel().astype(np.longdouble)
    z = np.exp(x); d = controls.X.reshape(-1, 17) if arm == 'A' else controls.M.reshape(-1, 1)
    np.testing.assert_allclose(out.log_partition, float(np.log(z.sum())), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(out.gradient, np.asarray((z/z.sum())@d, float), rtol=1e-13, atol=1e-13)
    assert out.supported_groups == 6 and out.status == 'supported' and out.arm == arm
    assert out.temperature == .8125
    assert out.score_reference == (score.distribution_fingerprint, score.parameter_fingerprint)


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', .3125)])
def test_all_active_gradient_finite_difference(arm, alpha):
    out = run(alpha=alpha, arm=arm); h = 1e-6
    for i in range(17 if arm == 'A' else 1):
        delta = np.zeros(17); delta[i] = h
        hi = run(theta=THETA+delta, arm=arm) if arm == 'A' else run(alpha=alpha+h, arm=arm)
        lo = run(theta=THETA-delta, arm=arm) if arm == 'A' else run(alpha=alpha-h, arm=arm)
        np.testing.assert_allclose((hi.log_partition-lo.log_partition)/(2*h), out.gradient[i], rtol=1e-7, atol=1e-7)


def test_right_alpha_zero_shared_score_and_nonzero_derivative():
    a, b = run(), run(arm='B'); h = 1e-6
    assert a.log_partition == b.log_partition
    f = lambda x: run(alpha=x, arm='B').log_partition
    np.testing.assert_allclose((-3*f(0.)+4*f(h)-f(2*h))/(2*h), b.gradient[0], atol=1e-7)
    assert b.gradient[0] != 0.


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', 0.), ('B', .3125)])
def test_fake_torch_full_abi_parity_owned_cpu_return(monkeypatch, arm, alpha):
    fake_backend(monkeypatch); score = controls.scores(alpha=alpha)
    device = controls.device_scores(score); before = core._fingerprint((score, THETA))
    actual = run(device, alpha=alpha, arm=arm, backend='torch'); expected = run(score, alpha=alpha, arm=arm)
    np.testing.assert_allclose(actual.log_partition, expected.log_partition, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-13, atol=1e-13)
    assert type(actual.gradient) is np.ndarray and actual.gradient.dtype == np.float64
    assert actual.score_reference == expected.score_reference and before == core._fingerprint((score, THETA))
    with pytest.raises(ValueError): actual.gradient.setflags(write=True)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('arm', ['A', 'B'])
def test_shared_extreme_derivative_is_added_back_exactly(monkeypatch, backend, arm):
    theta = np.zeros(17); theta[1] = 1.; score = controls.scores(theta, 0.)
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    d = np.zeros((2, 3, 17)); d[..., 0] = 1e300; d[..., 1] = x
    score = replace(score, scores_a=x, scores_b=x.copy(), native_scores_a=np.repeat(x[:, None], 2, axis=1),
        native_scores_b=np.repeat(x[:, None], 2, axis=1), geometry_derivatives_a=d,
        geometry_derivatives_b=d.copy(), alpha_derivatives_b=np.full((2, 3), 1e300))
    before = core._fingerprint((score, theta))
    if backend == 'torch': fake_backend(monkeypatch); source = controls.device_scores(score)
    else: source = score
    out = run(source, theta=theta, arm=arm, backend=backend)
    assert out.gradient[0] == 1e300   # Centering WITHOUT addback would incorrectly return zero.
    assert before == core._fingerprint((score, theta))
    if arm == 'A':
        z = np.exp(x.astype(np.longdouble)); expected = float((z*x).sum()/z.sum())
        np.testing.assert_allclose(out.gradient[1], expected, rtol=1e-14)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_extreme_constant_score_offset_and_zero_scores(monkeypatch, backend):
    theta = np.zeros(17); score = controls.scores(theta, 0., shift=1e300)
    if backend == 'torch': fake_backend(monkeypatch); score = controls.device_scores(score)
    out = run(score, theta=theta, backend=backend)
    assert out.log_partition == 1e300 and np.isfinite(out.gradient).all()
    normal = controls.scores(theta, 0.)
    if backend == 'torch': normal = controls.device_scores(normal)
    assert run(normal, theta=theta, backend=backend).log_partition == np.log(6.)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_masked_unsupported_scores_and_derivatives_remain_raw(monkeypatch, backend):
    good = np.ones((2, 3), bool); good[0, :2] = False
    score = controls.scores(alpha=0., supported=good); before = core._fingerprint(score)
    source = score
    if backend == 'torch': fake_backend(monkeypatch); source = controls.device_scores(score)
    out = run(source, backend=backend); x = score.scores_a[good]
    assert out.supported_groups == 4 and np.isnan(score.scores_a[~good]).all()
    np.testing.assert_allclose(out.log_partition, np.log(np.exp(x).sum()), rtol=1e-14)
    assert before == core._fingerprint(score)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('kind', ['unsupported', 'no_person', 'no_object', 'empty'])
def test_no_supported_is_undefined_not_valid_zero_loss(monkeypatch, backend, kind):
    score = controls.scores(alpha=0., supported=np.zeros((2, 3), bool))
    if kind != 'unsupported':
        n = 0 if kind in ('no_person', 'empty') else 2
        o = 0 if kind in ('no_object', 'empty') else 3
        identity = dict(score.identity)
        identity.update(source_person_ids=tuple(f'p{i}' for i in range(n)), source_object_ids=tuple(f'o{i}' for i in range(o)),
            person_members=tuple(np.array([i], np.int64) for i in range(n)), object_members=tuple(np.array([i], np.int64) for i in range(o)),
            person_to_group=np.arange(n), object_to_group=np.arange(o))
        score = replace(score, identity=identity, supported=np.empty((n, o), bool), native_supported=np.empty((n, 2, o), bool),
            native_route_supported=np.empty((n, 2, o), bool), native_scores_a=np.empty((n, 2, o)), native_scores_b=np.empty((n, 2, o)),
            scores_a=np.empty((n, o)), scores_b=np.empty((n, o)), geometry_derivatives_a=np.empty((n, o, 17)),
            geometry_derivatives_b=np.empty((n, o, 17)), alpha_derivatives_b=np.empty((n, o)))
    if backend == 'torch': fake_backend(monkeypatch); score = controls.device_scores(score)
    out = run(score, backend=backend)
    assert out.log_partition is None and out.status == 'no_supported' and out.supported_groups == 0
    assert not out.gradient.any() and out.gradient.shape == (17,)


def test_alias_native_duplicates_do_not_add_group_partition_mass():
    score = controls.scores(alpha=0.); identity = dict(score.identity)
    identity.update(source_object_ids=('o0', 'o1', 'o2', 'alias'), object_members=(np.array([0, 3]), np.array([1]), np.array([2])),
                    object_to_group=np.array([0, 1, 2, 0]))
    alias = replace(score, identity=identity, native_supported=score.native_supported[:, :, [0, 1, 2, 0]],
        native_route_supported=score.native_route_supported[:, :, [0, 1, 2, 0]],
        native_scores_a=score.native_scores_a[:, :, [0, 1, 2, 0]], native_scores_b=score.native_scores_b[:, :, [0, 1, 2, 0]])
    a, b = run(score), run(alias)
    assert a.log_partition == b.log_partition and a.gradient.tobytes() == b.gradient.tobytes()


def test_group_permutation_preserves_log_partition_and_vjp():
    score = controls.scores(alpha=0.); order = np.array([2, 0, 1]); identity = dict(score.identity)
    identity.update(source_object_ids=('o2', 'o0', 'o1'))
    changes = dict(identity=identity)
    for name in ('supported', 'scores_a', 'scores_b', 'geometry_derivatives_a',
                 'geometry_derivatives_b', 'alpha_derivatives_b'):
        changes[name] = getattr(score, name)[:, order]
    for name in ('native_supported', 'native_route_supported', 'native_scores_a', 'native_scores_b'):
        changes[name] = getattr(score, name)[:, :, order]
    expected, actual = run(score), run(replace(score, **changes))
    np.testing.assert_allclose(actual.log_partition, expected.log_partition, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('arm', ['A', 'B'])
def test_single_supported_group_partition_is_its_score_and_vjp(monkeypatch, backend, arm):
    good = np.zeros((2, 3), bool); good[1, 2] = True
    score = controls.scores(alpha=0., supported=good); source = score
    if backend == 'torch': fake_backend(monkeypatch); source = controls.device_scores(score)
    out = run(source, arm=arm, backend=backend)
    assert out.supported_groups == 1 and out.log_partition == score.scores_a[1, 2]
    expected = score.geometry_derivatives_a[1, 2] if arm == 'A' else np.array([score.alpha_derivatives_b[1, 2]])
    assert out.gradient.tobytes() == expected.tobytes()


@pytest.mark.parametrize('bad', [dict(theta=np.zeros(16)), dict(theta=np.zeros(17, np.float32)),
    dict(theta=np.full(17, np.nan)), dict(theta=[0.]*17), dict(alpha=-1.), dict(alpha=True), dict(alpha=np.float64(0.)),
    dict(alpha=float('inf')), dict(arm='C'), dict(arm=True), dict(backend='cuda'), dict(backend=True), dict(alpha=.1, arm='A')])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError): run(**bad)


@pytest.mark.parametrize('field,value', [('parameter_fingerprint', 'b'*64), ('distribution_fingerprint', 'a'*63),
    ('temperature', 0.), ('temperature', 1), ('temperature', float('inf')), ('alpha', 1.), ('alpha', 0),
    ('scores_a', np.zeros((2, 3), np.float32)), ('scores_b', np.full((2, 3), np.nan)),
    ('native_scores_b', np.zeros((2, 2, 2))), ('geometry_derivatives_b', np.zeros((2, 3, 16))),
    ('alpha_derivatives_b', np.full((2, 3), np.nan)), ('native_route_supported', np.ones((1, 2, 3), bool))])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_entire_score_abi_validated_even_in_inactive_arm(monkeypatch, field, value, backend):
    score = replace(controls.scores(alpha=0.), **{field: value})
    if backend == 'torch': fake_backend(monkeypatch); score = controls.device_scores(score)
    with pytest.raises(ValueError): run(score, backend=backend)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_forged_type_or_alias_mapping_rejected(monkeypatch, backend):
    if backend == 'torch': fake_backend(monkeypatch)
    with pytest.raises(ValueError): run(object(), backend=backend)
    score = controls.scores(alpha=0.); identity = dict(score.identity); identity['object_to_group'] = np.array([0, 0, 0])
    score = replace(score, identity=identity)
    if backend == 'torch': score = controls.device_scores(score)
    with pytest.raises(ValueError): run(score, backend=backend)


@pytest.mark.parametrize('change', ['device', 'requires_grad', 'dtype', 'unsupported_derivative', 'unsupported_score', 'route'])
def test_device_abi_rejects_mutable_autograd_wrong_device_and_filled_unsupported(monkeypatch, change):
    fake_backend(monkeypatch); good = np.ones((2, 3), bool); good[0, 0] = False
    score = controls.device_scores(controls.scores(alpha=0., supported=good))
    if change == 'device': score.geometry_derivatives_b.device = 'cuda:1'
    elif change == 'requires_grad': score.scores_b.requires_grad = True
    elif change == 'dtype': score.native_supported.array = score.native_supported.array.astype(int)
    elif change == 'unsupported_derivative': score.geometry_derivatives_b.array[0, 0, 0] = 1.
    elif change == 'unsupported_score': score.native_scores_a.array[0, :, 0] = 0.
    else: score.native_route_supported.array[0, :, 0] = True
    with pytest.raises(ValueError): run(score, backend='torch')


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('kind', ['score', 'derivative'])
def test_finite_intermediate_overflow_fails_without_clamp(monkeypatch, backend, kind):
    score = controls.scores(alpha=0.)
    if kind == 'score':
        value = score.scores_a.copy(); value.flat[:2] = [1e308, -1e308]; score = replace(score, scores_a=value)
    else:
        value = score.geometry_derivatives_a.copy(); value[0, 0, 0] = 1e308; value[0, 1, 0] = -1e308
        score = replace(score, geometry_derivatives_a=value)
    if backend == 'torch': fake_backend(monkeypatch); score = controls.device_scores(score)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises((ValueError, FloatingPointError)): run(score, backend=backend)


@pytest.mark.parametrize('target', ['theta', 'score'])
def test_borrowed_cpu_mutation_detected(monkeypatch, target):
    theta = THETA.copy(); score = controls.scores(theta, 0.); original = partition.original._numpy_score
    def bad(value):
        original(value)
        if target == 'theta': theta[0] += 1.
        else: value.geometry_derivatives_a[0, 0, 0] += 1.
    monkeypatch.setattr(partition.original, '_numpy_score', bad)
    with pytest.raises(ValueError, match='Borrowed'): run(score, theta=theta)


@pytest.mark.parametrize('target', ['theta', 'metadata'])
def test_borrowed_device_metadata_and_theta_mutation_detected(monkeypatch, target):
    fake_backend(monkeypatch); theta = THETA.copy(); score = controls.device_scores(controls.scores(theta, 0.))
    original = partition._torch_score
    def bad(value, t):
        original(value, t)
        if target == 'theta': theta[0] += 1.
        else: value.identity['object_to_group'][0] = 1
    monkeypatch.setattr(partition, '_torch_score', bad)
    with pytest.raises(ValueError, match='Borrowed'): run(score, theta=theta, backend='torch')


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_empty_B_uses_scalar_undefined_gradient_and_keeps_full_metadata(monkeypatch, backend):
    score = controls.scores(alpha=0., supported=np.zeros((2, 3), bool))
    source = score
    if backend == 'torch': fake_backend(monkeypatch); source = controls.device_scores(score)
    out = run(source, arm='B', backend=backend)
    assert out.log_partition is None and out.status == 'no_supported'
    assert out.gradient.shape == (1,) and out.gradient[0] == 0.
    assert out.score_reference == (score.distribution_fingerprint, score.parameter_fingerprint)


def test_outputs_sealed_and_function_has_no_reference_mask_argument():
    score = controls.scores(alpha=0.); before = core._fingerprint((score, THETA)); out = run(score)
    assert before == core._fingerprint((score, THETA))
    with pytest.raises(ValueError): out.gradient.setflags(write=True)
    with pytest.raises(FrozenInstanceError): out.status = 'FIT'
    with pytest.raises(TypeError): partition.marginal_partition(score, THETA, positive_mask=np.ones((2, 3), bool))


def test_import_is_lazy_and_no_scipy_or_native_torch_loaded():
    code = '''import sys, importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','scipy'): raise AssertionError('Forbidden heavy import')
sys.meta_path.insert(0,Deny())
sys.path.insert(0,sys.argv[1])
from world_reward.coherent_pair_partition import marginal_partition
assert 'torch' not in sys.modules and 'scipy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(Path(__file__).resolve().parents[1]/'src')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

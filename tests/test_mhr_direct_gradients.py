"""Manufactured algebra and callback wiring only; NO real Torch/MHR gradients."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'infra'))
import mhr_direct_gradients as q
import mhr_direct_bridge as bridge


def metadata():
    columns = (11, 23, 71, 103)
    bounds = np.tile(np.asarray([-1., 1.], np.float32), (249, 1))
    idx = np.zeros((18439, 8), np.int64); idx[:, 0] = 1
    weight = np.zeros(idx.shape, np.float32); weight[:, 0] = 1
    return dict(controls=[dict(name=n, column=c, global_closure=[1]) for n, c in zip(q.PARAMETERS, columns)],
                arrays=dict(bounds=bounds, lbs_indices=idx, lbs_weights=weight), fingerprint='manufactured-only')


def test_exact_full_bank_and_no_borrowed_alias():
    m = metadata(); before = m['arrays']['bounds'].tobytes(); b = q.gradient_batch(m)
    assert b['centre'].shape == (1, 204) and b['perturbations'].shape == (16, 204)
    assert b['identity'].shape == (17, 45) and b['expression'].shape == (17, 72)
    assert not b['identity'].any() and not b['expression'].any()
    assert np.array_equal(b['centre'][0, [11, 23, 71, 103]], [.03125]*4)
    for row in b['rows']:
        difference = b['perturbations'][row['row']] - b['centre'][0]
        assert np.count_nonzero(difference) == 1
        assert difference[m['controls'][row['direction']]['column']] == row['sign']*row['step']
    assert not b['centre'][:, 136:].any() and not b['perturbations'][:, 136:].any()
    for key in ('centre', 'perturbations', 'identity', 'expression'):
        assert b[key].dtype == np.float32 and not b[key].flags.writeable
        with pytest.raises(ValueError): b[key].setflags(write=True)
    assert m['arrays']['bounds'].tobytes() == before


@pytest.mark.parametrize('fault', ['centre', 'last_perturb', 'fixedscale', 'names', 'columns'])
def test_whole_preflight_rejects_before_any_execution(fault):
    m = metadata()
    if fault == 'centre': m['arrays']['bounds'][11, 0] = .04
    elif fault == 'last_perturb': m['arrays']['bounds'][103, 1] = q.CENTRE + q.FD_STEPS[1]/2
    elif fault == 'fixedscale': m['arrays']['bounds'][147] = [.001, .002]
    elif fault == 'names': m['controls'][3]['name'] = 'unknown'
    else: m['controls'][3]['column'] = m['controls'][0]['column']
    with pytest.raises(ValueError): q.gradient_batch(m)


def scalar_values(g):
    return np.asarray([17. + sign*h*g[j] for j in range(4) for h in q.FD_STEPS for sign in (-1, 1)])


def test_fixed_central_differences_against_independent_linear_derivative():
    gradient = np.asarray([.2, -.4, .8, -1.6])
    result = q.directional_comparison(17., scalar_values(gradient), gradient)
    assert len(result) == 8
    assert [r['step'] for r in result] == list(q.FD_STEPS)*4
    assert all(r['absolute_error_m_per_unit'] < 1e-10 for r in result)
    assert all(r['allowed_error_m_per_unit'] >= 1e-5 for r in result)


@pytest.mark.parametrize('fault', ['coarse', 'fine', 'zero', 'nan', 'shape', 'centre'])
def test_no_better_step_selection_zero_floor_or_nonfinite_rescue(fault):
    g = np.ones(4); values = scalar_values(g); centre = 17.
    if fault == 'coarse': values[1] += .1
    elif fault == 'fine': values[3] += .1
    elif fault == 'zero': g[2] = 0
    elif fault == 'nan': values[0] = np.nan
    elif fault == 'shape': values = values[:15]
    else: centre = np.inf
    with pytest.raises(ValueError): q.directional_comparison(centre, values, g)


class Tensor:
    """Tiny NumPy-backed API double; NOT an autograd implementation."""
    def __init__(self, data, *, requires_grad=False):
        self.data = np.asarray(data); self.requires_grad = requires_grad
        self.device = 'cuda'; self.grad = None; self.is_leaf = True; self._version = 0
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    def requires_grad_(self, value): self.requires_grad = value; return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.data
    def __getitem__(self, index): return Tensor(self.data[index], requires_grad=self.requires_grad)
    def to(self, *, dtype): return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)
    def sum(self, *, dim): return Tensor(self.data.sum(axis=dim), requires_grad=self.requires_grad)
    def __mul__(self, other):
        return Tensor(self.data*(other.data if isinstance(other, Tensor) else other), requires_grad=self.requires_grad)
    def __add__(self, other):
        return Tensor(self.data+(other.data if isinstance(other, Tensor) else other), requires_grad=self.requires_grad)
    def __truediv__(self, other): return Tensor(self.data/other, requires_grad=self.requires_grad)
    def __gt__(self, other): return self.data > other


def runtime_case(monkeypatch, fault=None):
    m = metadata(); calls = []; state = {'enabled': False}; columns = [11, 23, 71, 103]
    parameter = Tensor(np.asarray([1.], np.float32)); coefficient = np.asarray([10., 20., 30., 40.], np.float32)
    class Model:
        def named_parameters(self): return [('weight', parameter)]
        def __call__(self, identity, controls, expression, correctives):
            calls.append(('forward', len(controls.data), state['enabled'], correctives))
            n = len(controls.data); v = np.zeros((n, 18439, 3), np.float32)
            s = np.zeros((n, 127, 8), np.float32); s[:, :, 6:8] = 1
            v[:, :4, 0] = controls.data[:, columns]*coefficient
            if fault == 'shape': v = v[:, :-1]
            if fault == 'nonfinite': v[0, 0, 0] = np.nan
            if fault == 'mutation': controls.data[0, 0] = 1
            if fault == 'weight_version': parameter._version += 1
            if fault == 'weight_grad': parameter.grad = object()
            if fault == 'scale' and n == 16: s[0, 0, 7] = 2
            return Tensor(v, requires_grad=state['enabled'] and fault != 'detached'), Tensor(s, requires_grad=state['enabled'])
    @contextmanager
    def context(value):
        before = state['enabled']; state['enabled'] = value
        try: yield
        finally: state['enabled'] = before
    def grad(scalar, controls, **kwargs):
        calls.append(('vjp', kwargs)); gradient = np.zeros_like(controls.data)
        # Analytic manufactured linear probe, NOT evidence about real MHR autograd.
        cotangent = np.asarray([1., -2., 3., -1.]) / 18439 / 100
        gradient[0, columns] = coefficient*cotangent
        if fault == 'wronggradient': gradient[0, columns] *= -1
        if fault == 'zerogradient': gradient[0, columns[0]] = 0
        return (Tensor(gradient),)
    torch = SimpleNamespace(float32=np.dtype('float32'), float64=np.dtype('float64'),
        as_tensor=lambda a, **kw: Tensor(np.asarray(a, dtype=kw['dtype'])),
        is_tensor=lambda a: isinstance(a, Tensor), isfinite=lambda a: np.isfinite(a.data),
        is_inference_mode_enabled=lambda: fault == 'inference_mode', is_grad_enabled=lambda: state['enabled'],
        enable_grad=lambda: context(True), no_grad=lambda: context(False), autograd=SimpleNamespace(grad=grad),
        cuda=SimpleNamespace(synchronize=lambda: calls.append('sync')))
    monkeypatch.setattr(bridge, 'inspect_metadata', lambda model: m)
    return Model(), torch, calls, m, parameter


def test_two_full_forwards_one_vjp_fixed_order_without_backward_accumulation(monkeypatch):
    model, torch, calls, _, parameter = runtime_case(monkeypatch)
    checks = []; result = q.run_control(model, torch, check=lambda: checks.append('check'))
    assert calls == [('forward', 1, True, True),
        ('vjp', dict(create_graph=False, retain_graph=False)), 'sync', ('forward', 16, False, True), 'sync']
    assert len(checks) == 5 and parameter.grad is None
    assert result['native_forward_calls'] == 2 and result['decoded_frames'] == 17 and result['vjp_calls'] == 1
    assert len(result['comparisons']) == 8 and len(result['material_support_motion']) == 4
    assert result['SAM_checkpoint_used'] is result['root_gradients_qualified'] is result['backend_adoption'] is False


def test_last_bound_failure_means_zero_forward_or_vjp(monkeypatch):
    model, torch, calls, m, _ = runtime_case(monkeypatch)
    m['arrays']['bounds'][103, 1] = q.CENTRE + q.FD_STEPS[1]/2
    with pytest.raises(ValueError, match='bounds'): q.run_control(model, torch, check=lambda: None)
    assert calls == []


@pytest.mark.parametrize('fault', ['shape', 'nonfinite', 'mutation', 'detached', 'scale', 'weight_version',
    'weight_grad', 'wronggradient', 'zerogradient', 'inference_mode'])
def test_runtime_failures_never_retry_or_claim_pass(monkeypatch, fault):
    model, torch, calls, _, _ = runtime_case(monkeypatch, fault)
    with pytest.raises(ValueError): q.run_control(model, torch, check=lambda: None)
    assert sum(isinstance(c, tuple) and c[0] == 'forward' for c in calls) <= 2
    assert sum(isinstance(c, tuple) and c[0] == 'vjp' for c in calls) <= 1


def test_dirty_initial_weight_gradient_stops_before_forward(monkeypatch):
    model, torch, calls, _, parameter = runtime_case(monkeypatch)
    parameter.grad = object()
    with pytest.raises(ValueError, match='initially'): q.run_control(model, torch, check=lambda: None)
    assert not calls


def test_zero_vjp_aborts_before_second_forward(monkeypatch):
    model, torch, calls, _, _ = runtime_case(monkeypatch, 'zerogradient')
    with pytest.raises(ValueError, match='nonzero before FD'): q.run_control(model, torch, check=lambda: None)
    assert sum(isinstance(c, tuple) and c[0] == 'forward' for c in calls) == 1


def test_import_has_no_scientific_dependencies_or_native_model_loader():
    path = Path(q.__file__); tree = ast.parse(path.read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(top_imports) == 1 and top_imports[0].module == '__future__'
    source = path.read_text()
    assert 'jit.load' not in source and '.backward(' not in source and 'run_control(model, torch' in source
    assert 'ATOL_M_PER_UNIT = 1e-5' in source and 'RTOL = 1e-2' in source

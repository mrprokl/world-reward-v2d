"""Tiny manufactured analytic callbacks/fake minimizers; no SciPy or real FIT."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward.bounded_analytic_optimizer import (
    AnalyticSolveResult, SolverClosed, SolverLimits, solve_zero_start,
)


LIMITS = SolverLimits(5, 8, 4, 3, 1e-12, 1e-6, 1e-6)


def quadratic(x):
    target = np.arange(1., len(x)+1)
    return float(.5*((x-target)@(x-target))), x-target


def minimizer(target, *, success=True, nit=1, status=0, capture=None):
    def run(function, start, **kwargs):
        if capture is not None:
            capture.update(start=start.copy(), options=kwargs)
        function(start)
        x = np.asarray(target, np.float64).copy()
        value, gradient = function(x)
        return SimpleNamespace(x=x, success=success, nit=nit, status=status,
                               fun=value, jac=gradient)
    return run


def solve(evaluate=quadratic, dimension=2, limits=LIMITS, **kwargs):
    args = dict(checkpoint=lambda: None, minimize=minimizer(np.arange(1., dimension+1)))
    args.update(kwargs)
    return solve_zero_start(evaluate, dimension, limits, **args)


def test_exact_quadratic_one_zero_start_options_and_owned_result():
    capture = {}; visited = []
    out = solve(checkpoint=lambda: visited.append(1),
                minimize=minimizer([1., 2.], capture=capture))
    assert type(out) is AnalyticSolveResult and out.success and out.status == 0
    np.testing.assert_array_equal(capture['start'], np.zeros(2))
    assert capture['options'] == dict(method='L-BFGS-B', jac=True, bounds=None,
        options=dict(maxiter=5, maxfun=6, maxls=4, maxcor=3, ftol=1e-12, gtol=1e-6))
    assert out.initial_loss == 2.5 and out.final_loss == 0.
    assert out.evaluations == 4 and out.iterations == 1 and out.projected_gradient_inf == 0.
    assert len(visited) == 11
    for a in (out.parameters, out.gradient):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        out.success = False


def test_bounded_active_constraint_projects_positive_gradient_not_fake_stationarity():
    capture = {}
    def objective(x):
        return float(.5*((x+1.)@(x+1.))), x+1.
    out = solve(objective, 1, nonnegative=True, minimize=minimizer([0.], capture=capture))
    assert out.parameters[0] == 0. and out.gradient[0] == 1.
    assert out.projected_gradient_inf == 0. and out.initial_loss == out.final_loss == .5
    assert capture['options']['bounds'] == [(0., None)]
    with pytest.raises(SolverClosed, match='nonstationary'):
        solve(lambda x: (float(.5*((x-1.)@(x-1.))), x-1.), 1,
              nonnegative=True, minimize=minimizer([0.]))


def test_all_coordinate_bounds_and_interior_stationarity():
    capture = {}
    out = solve(nonnegative=True, minimize=minimizer([1., 2.], capture=capture))
    assert capture['options']['bounds'] == [(0., None), (0., None)]
    assert out.projected_gradient_inf == 0.
    with pytest.raises(SolverClosed, match='nonstationary'):
        solve(minimize=minimizer([.5, 2.]))


@pytest.mark.parametrize('change', [
    {'maxiter': 0}, {'maxiter': True}, {'max_evaluations': 1}, {'max_evaluations': 2.},
    {'maxls': -1}, {'maxcor': 0}, {'ftol': np.nan}, {'gtol': np.inf},
    {'projected_stationarity': -1.}, {'ftol': True},
])
def test_invalid_explicit_limits(change):
    with pytest.raises(SolverClosed, match='invalid_limits'):
        replace(LIMITS, **change)


@pytest.mark.parametrize('change', [
    {'dimension': 0}, {'dimension': True}, {'dimension': 2.}, {'limits': {}},
    {'evaluate': None}, {'checkpoint': None}, {'minimize': None}, {'nonnegative': 1},
])
def test_invalid_call_fails_before_objective(change):
    args = dict(evaluate=lambda x: pytest.fail('No callback on invalid inputs'), dimension=2,
                limits=LIMITS, checkpoint=lambda: None, minimize=minimizer([1., 2.]))
    args.update(change)
    with pytest.raises(SolverClosed, match='invalid_inputs'):
        solve_zero_start(**args)


@pytest.mark.parametrize('value', [
    (np.inf, np.zeros(2)), (0., np.full(2, np.nan)), (0., np.ones(2, np.float32)),
    (0., np.ones(1)), (np.float32(0.), np.zeros(2)), (False, np.zeros(2)),
    [0., np.zeros(2)], (np.array(0.), np.zeros(2)), (0., [0., 0.]),
])
def test_invalid_objective_outputs_close(value):
    with pytest.raises(SolverClosed, match='invalid_objective') as exc:
        solve(lambda x: value)
    assert exc.value.evaluations == 1


@pytest.mark.parametrize('x', [np.ones(2, np.float32), np.ones(1), np.full(2, np.nan), [1., 2.]])
def test_invalid_minimizer_parameter_input_rejected(x):
    def bad(function, start, **kwargs):
        function(x)
    with pytest.raises(SolverClosed, match='invalid_parameters'):
        solve(minimize=bad)


def test_objective_cannot_mutate_owned_parameter_or_external_solver_parameter():
    seen = []
    def mutate(x):
        seen.append(x); x[0] = 4.
        return quadratic(x)
    with pytest.raises(SolverClosed, match='parameter_mutation'):
        solve(mutate)
    assert seen[0][0] == 4.
    external = np.zeros(2)
    def touch_other(x):
        external[0] = 9.
        return quadratic(x)
    def bad(function, start, **kwargs):
        external[:] = 0.
        function(external)
    with pytest.raises(SolverClosed, match='parameter_mutation'):
        solve(touch_other, minimize=bad)


def test_gradient_return_is_owned_and_final_parameter_not_aliased_to_solver():
    source_gradient = np.zeros(2); final = np.array([1., 2.]); captured = []
    def evaluate(x):
        source_gradient[:] = x-final
        return float(.5*(source_gradient@source_gradient)), source_gradient
    def run(function, start, **kwargs):
        loss, gradient = function(final); captured.append(gradient)
        source_gradient[:] = 99.
        return SimpleNamespace(x=final, success=True, nit=1, status=0)
    out = solve(evaluate, minimize=run)
    final[:] = 44.; source_gradient[:] = 33.
    np.testing.assert_array_equal(out.parameters, [1., 2.])
    np.testing.assert_array_equal(out.gradient, [0., 0.])
    np.testing.assert_array_equal(captured[0], [0., 0.])


def test_post_callback_checkpoint_parameter_drift_rejected():
    borrowed = []
    def evaluate(x):
        borrowed.append(x)
        return quadratic(x)
    def checkpoint():
        if borrowed:
            borrowed[-1][0] += 1.
    with pytest.raises(SolverClosed, match='parameter_mutation'):
        solve(evaluate, checkpoint=checkpoint)


def test_final_result_x_drift_during_verification_is_rejected():
    final = np.array([1., 2.]); calls = 0
    def evaluate(x):
        nonlocal calls
        calls += 1
        answer = quadratic(x)
        if calls == 2:
            final[0] += 1.
        return answer
    def minimize(function, start, **kwargs):
        return SimpleNamespace(x=final, success=True, nit=1, status=0)
    with pytest.raises(SolverClosed, match='parameter_mutation'):
        solve(evaluate, minimize=minimize)


def test_last_checkpoint_drift_of_solver_result_prevents_publication():
    final = np.array([1., 2.]); calls = 0
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 7:
            final[0] += 1.
    def minimize(function, start, **kwargs):
        return SimpleNamespace(x=final, success=True, nit=1, status=0)
    with pytest.raises(SolverClosed, match='parameter_mutation'):
        solve(checkpoint=checkpoint, minimize=minimize)


@pytest.mark.parametrize('cap,reason', [(2, 'evaluation_cap'), (3, 'evaluation_cap'), (4, None)])
def test_cap_includes_initial_minimizer_and_independent_final(cap, reason):
    limits = replace(LIMITS, max_evaluations=cap)
    if reason:
        with pytest.raises(SolverClosed, match=reason) as exc:
            solve(limits=limits)
        assert exc.value.evaluations == cap
    else:
        assert solve(limits=limits).evaluations == cap


def test_actual_callback_cap_overrules_minimizer_options_and_never_restarts():
    starts = []
    def bad(function, x, **kwargs):
        starts.append(1)
        for _ in range(100):
            function(x)
    with pytest.raises(SolverClosed, match='evaluation_cap') as exc:
        solve(minimize=bad)
    assert starts == [1] and exc.value.evaluations == LIMITS.max_evaluations


@pytest.mark.parametrize('kind,reason', [
    ('parameters', 'invalid_parameters'), ('objective', 'objective_failed'),
    ('gradient', 'invalid_objective'), ('mutation', 'parameter_mutation'),
    ('checkpoint', 'checkpoint_failed'),
])
def test_caught_callback_failure_is_latched_not_a_successful_old_point(kind, reason):
    objective_calls = 0; fail_checkpoint = False; swallowed = []; native_calls = []
    def evaluate(x):
        nonlocal objective_calls
        objective_calls += 1
        if objective_calls == 2:
            if kind == 'objective': raise RuntimeError('SECRET_SENTINEL')
            if kind == 'gradient': return 0., np.ones(1)
            if kind == 'mutation': x[0] = 44.
        return quadratic(x)
    def checkpoint():
        nonlocal fail_checkpoint
        if fail_checkpoint:
            fail_checkpoint = False
            raise TimeoutError('SECRET_SENTINEL')
    def native(function, start, **kwargs):
        nonlocal fail_checkpoint
        native_calls.append(1); fail_checkpoint = kind == 'checkpoint'
        try:
            function(np.zeros(3) if kind == 'parameters' else start)
        except SolverClosed as exc:
            swallowed.append(exc.reason)
        return SimpleNamespace(x=np.array([1., 2.]), success=True, nit=1, status=0)
    with pytest.raises(SolverClosed, match=reason) as exc:
        solve(evaluate, checkpoint=checkpoint, minimize=native)
    assert native_calls == [1] and swallowed == [reason]
    assert exc.value.reason == reason and 'SECRET' not in str(exc.value)
    assert objective_calls <= 2  # No final verification or retry after the failure.


def test_after_caught_failure_no_later_objective_or_checkpoint_callback_is_run():
    calls = []; checkpoints = []; failures = []
    def evaluate(x):
        calls.append(1); return quadratic(x)
    def native(function, start, **kwargs):
        for value in (np.zeros(3), np.array([1., 2.]), start):
            try:
                function(value)
            except SolverClosed as exc:
                failures.append(exc)
        return SimpleNamespace(x=np.array([1., 2.]), success=True, nit=1, status=0)
    with pytest.raises(SolverClosed, match='invalid_parameters') as exc:
        solve(evaluate, checkpoint=lambda: checkpoints.append(1), minimize=native)
    assert len(calls) == 1 and len(checkpoints) == 4
    assert len(failures) == 3 and all(x is exc.value for x in failures)
    assert exc.value.evaluations == 2


def test_caught_evaluation_cap_does_not_resume_callbacks_or_final_verification():
    calls = []; failures = []; limits = replace(LIMITS, max_evaluations=3)
    def evaluate(x):
        calls.append(1); return quadratic(x)
    def native(function, start, **kwargs):
        for _ in range(6):
            try:
                function(start)
            except SolverClosed as exc:
                failures.append(exc)
        return SimpleNamespace(x=np.array([1., 2.]), success=True, nit=1, status=0)
    with pytest.raises(SolverClosed, match='evaluation_cap') as exc:
        solve(evaluate, limits=limits, minimize=native)
    assert len(calls) == exc.value.evaluations == 3 and len(failures) == 4
    assert all(x is exc.value for x in failures)


@pytest.mark.parametrize('change,reason', [
    ({'success': False}, 'solver_unsuccessful'), ({'nit': 6}, 'iteration_cap'),
    ({'nit': -1}, 'invalid_result'), ({'nit': True}, 'invalid_result'),
    ({'success': 1}, 'invalid_result'), ({'status': '0'}, 'invalid_result'),
    ({'x': np.full(2, np.inf)}, 'invalid_parameters'), ({'x': np.ones(1)}, 'invalid_parameters'),
])
def test_result_claims_never_replace_independent_final_verification(change, reason):
    def bad(function, x, **kwargs):
        values = dict(x=np.array([1., 2.]), success=True, nit=1, status=0); values.update(change)
        return SimpleNamespace(**values)
    with pytest.raises(SolverClosed, match=reason):
        solve(minimize=bad)


def test_loss_increase_and_out_of_bound_final_fail_closed():
    with pytest.raises(SolverClosed, match='loss_increased'):
        solve(minimize=minimizer([5., 7.]))
    with pytest.raises(SolverClosed, match='invalid_parameters'):
        solve(nonnegative=True, minimize=minimizer([-1., 2.]))


@pytest.mark.parametrize('at', [1, 2, 3, 4, 5])
def test_checkpoint_failure_prevents_success_even_after_solver_or_final(at):
    calls = 0
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == at:
            raise TimeoutError('SECRET_SENTINEL')
    with pytest.raises(SolverClosed, match='checkpoint_failed') as exc:
        solve(checkpoint=checkpoint, minimize=lambda f, x, **kw:
              SimpleNamespace(x=np.array([1., 2.]), success=True, nit=1, status=0))
    assert 'SECRET_SENTINEL' not in str(exc.value)


def test_same_zero_point_drift_not_accepted_as_new_best_result():
    calls = 0
    def drift(x):
        nonlocal calls
        calls += 1
        loss, gradient = quadratic(x)
        return loss/calls, gradient/calls
    with pytest.raises(SolverClosed, match='objective_drift'):
        solve(drift, minimize=lambda f, x, **kw: SimpleNamespace(x=x, success=True, nit=0, status=0))


@pytest.mark.parametrize('which', ['objective', 'minimizer'])
def test_raw_exception_text_is_not_exposed(which):
    def bad(*args, **kwargs):
        raise RuntimeError('SECRET_SENTINEL https://private.invalid/?token=secret')
    with pytest.raises(SolverClosed) as exc:
        solve(evaluate=bad if which == 'objective' else quadratic,
              minimize=bad if which == 'minimizer' else minimizer([1., 2.]))
    assert exc.value.reason == which+'_failed' and 'SECRET' not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_module_import_has_no_scipy_torch_or_autograd_import():
    root = Path(__file__).resolve().parents[1]
    script = f"""import sys
sys.path.insert(0, {str(root/'src')!r})
import world_reward.bounded_analytic_optimizer
assert 'scipy' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, '-I', '-B', '-c', script], check=True, capture_output=True)

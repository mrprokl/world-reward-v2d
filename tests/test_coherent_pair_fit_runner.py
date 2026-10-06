"""Tiny genuine-score orchestration; fake transport plus local CPU SciPy only."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward import coherent_pair_fit_runner as runner
from world_reward.bounded_analytic_optimizer import SolverClosed, SolverLimits
from world_reward.coherent_pair_marginal_objective import marginal_objective


spec = importlib.util.spec_from_file_location('fit_original_objective_controls',
    Path(__file__).with_name('test_coherent_pair_marginal_objective.py'))
controls = importlib.util.module_from_spec(spec); spec.loader.exec_module(controls)
IDS = tuple(f'opaque{i}' for i in range(8)); FOLDS = tuple(i % 4 for i in range(8))
LAMBDAS = (.25, 1., 4.)
LIMITS = SolverLimits(12, 40, 20, 10, 1e-12, 1e-6, 1e-6)


def score(theta, alpha):
    value = controls.scores(theta, alpha)
    a = np.zeros((2, 3)); b = a + alpha*controls.M
    return replace(value, scores_a=a, scores_b=b,
        native_scores_a=np.repeat(a[:, None, :], 2, axis=1),
        native_scores_b=np.repeat(b[:, None, :], 2, axis=1),
        geometry_derivatives_a=np.zeros((2, 3, 17)), geometry_derivatives_b=np.zeros((2, 3, 17)))


class Callbacks:
    def __init__(self):
        self.prepared = []; self.released = []; self.rows = []; self.solves = []
    def prepare(self, ids):
        value = SimpleNamespace(train_ids=ids, sequence=len(self.prepared), disposed=False)
        self.prepared.append(value); return value
    def release(self, context):
        assert not context.disposed
        context.disposed = True; self.released.append(context)
    def score_rows(self, context, ids, theta, *, alpha):
        assert not context.disposed and not theta.flags.writeable
        self.rows.append((context, ids, theta.copy(), alpha))
        return ((identifier, score(theta, alpha), controls.positive()) for identifier in ids)
    def minimize(self, function, start, **options):
        self.solves.append((start.copy(), options)); x = start.copy()
        loss, gradient = function(x); iteration = 0
        while np.abs(gradient).max() > 1e-9 and iteration < 10:
            h = 1e-5; up = x.copy(); up[0] += h
            _, upper = function(up)
            curvature = (upper[0]-gradient[0])/h
            x[0] -= gradient[0]/curvature
            if options['bounds'] is not None:
                x[0] = max(0., x[0])
            loss, gradient = function(x); iteration += 1
        return SimpleNamespace(x=x, success=True, nit=iteration, status=0)


def run(callbacks=None, **changes):
    callbacks = callbacks or Callbacks()
    args = dict(record_ids=IDS, folds=FOLDS, lambdas=LAMBDAS,
        prepare=callbacks.prepare, score_rows=callbacks.score_rows, limits_a=LIMITS,
        limits_b=LIMITS, regularization_b=1., backend='numpy', checkpoint=lambda: None,
        minimize=callbacks.minimize, release=callbacks.release)
    args.update(changes)
    return runner.fit_coherent_pair_ab(**args)


def test_twelve_cv_final_a_and_b_once_five_contexts_reused_then_released():
    callbacks = Callbacks(); result = run(callbacks)
    assert type(result) is runner.CoherentPairFitResult and result.status == 'complete'
    assert len(callbacks.solves) == 14 and len(callbacks.prepared) == len(callbacks.released) == 5
    for fold in range(4):
        assert callbacks.prepared[fold].train_ids == tuple(x for x, f in zip(IDS, FOLDS) if f != fold)
        for _, ids, _, alpha in [x for x in callbacks.rows if x[0] is callbacks.prepared[fold]]:
            assert ids in (callbacks.prepared[fold].train_ids,
                           tuple(x for x, f in zip(IDS, FOLDS) if f == fold)) and alpha == 0.
    assert callbacks.prepared[-1].train_ids == IDS
    assert result.cross_validation.selected_regularization == 4.  # Exact equal held loss.
    assert len(result.cross_validation.ledger) == 12 and result.geometry.parameters.shape == (17,)
    assert result.relation.parameters.shape == (1,) and result.relation.parameters[0] > 0.
    assert result.relation.projected_gradient_inf <= LIMITS.projected_stationarity
    assert not result.context_integrity_verified
    assert all((s[0] == 0).all() for s in callbacks.solves)
    assert all(s[1]['bounds'] is None for s in callbacks.solves[:-1])
    assert callbacks.solves[-1][1]['bounds'] == [(0., None)]
    assert all(c.disposed for c in callbacks.prepared)
    assert result.geometry_counts['records'] == result.relation_counts['records'] == 8


def test_final_b_freezes_a_and_original_objective_ridge_once():
    callbacks = Callbacks(); out = run(callbacks, regularization_b=.25)
    final_context = callbacks.prepared[-1]
    b_rows = [r for r in callbacks.rows if r[0] is final_context and r[3] > 0.]
    assert b_rows and all(np.array_equal(r[2], out.geometry.parameters) for r in b_rows)
    alpha = float(out.relation.parameters[0]); s = score(out.geometry.parameters, alpha)
    expected = marginal_objective((s,)*len(IDS), (controls.positive(),)*len(IDS),
        out.geometry.parameters, alpha=alpha, regularization=.25, arm='B')
    assert out.relation.final_loss == expected.loss
    np.testing.assert_allclose(out.relation.gradient, expected.gradient, atol=1e-12)
    for a in (out.geometry.parameters, out.relation.parameters, out.geometry.gradient):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        out.status = 'changed'


def test_native_local_scipy_nonconstant_geometry_and_relation_integration():
    # Tiny manufactured CPU control: no Azure runtime, role values or accuracy.
    from scipy.optimize import minimize
    callbacks = Callbacks(); native_calls = []
    def rows(context, ids, theta, *, alpha):
        assert not context.disposed and not theta.flags.writeable
        return ((identifier, controls.scores(theta, alpha), controls.positive())
                for identifier in ids)
    def native(*args, **kwargs):
        result = minimize(*args, **kwargs)
        native_calls.append(result.success)
        return result
    limits = SolverLimits(200, 500, 20, 10, 1e-12, 1e-6, 1e-6)
    out = run(callbacks, score_rows=rows, minimize=native,
              limits_a=limits, limits_b=limits)
    assert len(native_calls) == 14 and all(native_calls)
    assert len(callbacks.prepared) == len(callbacks.released) == 5
    assert np.any(out.geometry.parameters != 0.) and out.relation.parameters[0] > 0.
    assert out.geometry.projected_gradient_inf <= limits.projected_stationarity
    assert out.relation.projected_gradient_inf <= limits.projected_stationarity
    theta = out.geometry.parameters; alpha = float(out.relation.parameters[0])
    for solution, arm, a, lam in ((out.geometry, 'A', 0.,
            out.cross_validation.selected_regularization), (out.relation, 'B', alpha, 1.)):
        expected = marginal_objective((controls.scores(theta, a),)*len(IDS),
            (controls.positive(),)*len(IDS), theta, alpha=a, regularization=lam, arm=arm)
        np.testing.assert_allclose(solution.final_loss, expected.loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(solution.gradient, expected.gradient, rtol=1e-12, atol=1e-12)


def test_partial_training_fixed_n_retained_but_held_missing_closes():
    callbacks = Callbacks()
    def rows(context, ids, theta, *, alpha):
        def each():
            for identifier in ids:
                s = score(theta, alpha)
                if identifier == IDS[0]:
                    s = controls.scores(theta, alpha, supported=np.zeros((2, 3), bool))
                yield identifier, s, controls.positive()
        return each()
    with pytest.raises(runner.FixedFitClosed, match='invalid_held_losses') as exc:
        run(callbacks, score_rows=rows)
    assert exc.value.phase == 'cross_validation'
    assert len(callbacks.prepared) == len(callbacks.released) == 1


def test_partial_positive_status_preserved_without_renormalizing_population():
    callbacks = Callbacks(); good = np.ones((2, 3), bool); good[0, 0] = False
    def rows(context, ids, theta, *, alpha):
        value = score(theta, alpha)
        s = replace(value, supported=good, native_supported=np.repeat(good[:, None, :], 2, axis=1),
            native_route_supported=np.repeat(good[:, None, :], 2, axis=1),
            scores_a=np.where(good, value.scores_a, np.nan), scores_b=np.where(good, value.scores_b, np.nan),
            native_scores_a=np.repeat(np.where(good, value.scores_a, np.nan)[:, None, :], 2, axis=1),
            native_scores_b=np.repeat(np.where(good, value.scores_b, np.nan)[:, None, :], 2, axis=1),
            alpha_derivatives_b=np.where(good, value.alpha_derivatives_b, 0.))
        return ((identifier, s, controls.positive()) for identifier in ids)
    out = run(callbacks, score_rows=rows)
    assert out.geometry_record_statuses == out.relation_record_statuses == ('used_partial_positive',)*8
    assert out.geometry_counts['records'] == out.geometry_counts['used'] == 8
    assert out.geometry_counts['partially_missing_positive'] == 8


def test_fake_torch_lazy_backend_reuses_original_stream_no_device_import(monkeypatch):
    fake = controls.fake_torch()
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(controls.Tensor, 'detach', lambda self: self, raising=False)
    monkeypatch.setattr(controls.Tensor, 'cpu', lambda self: self, raising=False)
    monkeypatch.setattr(controls.Tensor, 'numpy', lambda self: self.array, raising=False)
    callbacks = Callbacks()
    def rows(context, ids, theta, *, alpha):
        return ((identifier, controls.device_scores(value), mask)
                for identifier, value, mask in callbacks.score_rows(context, ids, theta, alpha=alpha))
    actual = run(callbacks, backend='torch', score_rows=rows)
    expected = run()
    for left, right in ((actual.geometry, expected.geometry), (actual.relation, expected.relation)):
        np.testing.assert_allclose(left.parameters, right.parameters, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(left.final_loss, right.final_loss, atol=1e-12, rtol=1e-12)
    assert actual.cross_validation.selected_regularization == expected.cross_validation.selected_regularization


@pytest.mark.parametrize('kind', ['no_positive', 'no_support', 'all_positive'])
def test_no_informative_training_closes_not_fake_zero_fit(kind):
    callbacks = Callbacks()
    def rows(context, ids, theta, *, alpha):
        mask = controls.positive(); s = score(theta, alpha)
        if kind == 'no_positive': mask[:] = False
        if kind == 'all_positive': mask[:] = True
        if kind == 'no_support': s = controls.scores(theta, alpha, supported=np.zeros((2, 3), bool))
        return ((identifier, s, mask) for identifier in ids)
    with pytest.raises(runner.FixedFitClosed, match='no_informative') as exc:
        run(callbacks, score_rows=rows)
    assert exc.value.phase == 'objective_A' and exc.value.evaluations == 1
    assert not callbacks.solves and len(callbacks.released) == 1


@pytest.mark.parametrize('change', [
    {'record_ids': ()}, {'folds': (0,)*8}, {'lambdas': ()}, {'limits_a': None},
    {'limits_b': None}, {'regularization_b': -1.}, {'regularization_b': True},
    {'regularization_b': np.inf}, {'regularization_b': 10**400}, {'backend': 'invalid'},
    {'backend': ['numpy']}, {'prepare': None}, {'release': None}, {'minimize': None},
])
def test_invalid_inputs_no_context_preparation(change):
    callbacks = Callbacks()
    with pytest.raises(runner.FixedFitClosed):
        run(callbacks, **change)
    assert not callbacks.prepared and not callbacks.solves


@pytest.mark.parametrize('where,reason', [('prepare', 'prepare_failed'),
    ('rows', 'objective_stream_failed'), ('minimize', 'minimizer_failed'), ('release', 'release_failed')])
def test_callback_errors_safe_and_no_restart(where, reason):
    callbacks = Callbacks(); kwargs = {}
    def fail(*a, **k):
        raise RuntimeError('SECRET_SENTINEL https://private.invalid/token')
    kwargs[{'prepare': 'prepare', 'rows': 'score_rows', 'minimize': 'minimize', 'release': 'release'}[where]] = fail
    with pytest.raises(runner.FixedFitClosed, match=reason) as exc:
        run(callbacks, **kwargs)
    assert 'SECRET' not in str(exc.value) and 'private' not in str(exc.value)
    assert len(callbacks.prepared) <= 1 and len(callbacks.solves) <= 3


def test_missing_context_rejects_not_reprepared():
    count = []
    with pytest.raises(runner.FixedFitClosed, match='missing_context'):
        run(prepare=lambda ids: count.append(ids))
    assert len(count) == 1


def test_failed_next_fold_retains_phase_and_complete_cv_progress_without_retry():
    callbacks = Callbacks(); attempts = []
    def prepare(ids):
        attempts.append(ids)
        if len(attempts) == 2:
            raise RuntimeError('SECRET_SENTINEL')
        return callbacks.prepare(ids)
    with pytest.raises(runner.FixedFitClosed, match='prepare_failed') as exc:
        run(callbacks, prepare=prepare)
    assert exc.value.phase == 'prepare' and exc.value.fold == 1
    assert exc.value.lambda_index == 0 and exc.value.completed == 3
    assert len(attempts) == 2 and len(callbacks.released) == 1
    assert len(callbacks.solves) == 3 and 'SECRET' not in repr(vars(exc.value))


def test_release_failure_after_successful_b_is_not_promoted_to_pass():
    callbacks = Callbacks()
    def release(context):
        callbacks.release(context)
        if context.train_ids == IDS:
            raise RuntimeError('private cleanup')
    with pytest.raises(runner.FixedFitClosed, match='release_failed') as exc:
        run(callbacks, release=release)
    assert exc.value.phase == 'release' and len(callbacks.solves) == 14
    assert len(callbacks.released) == 5


def test_release_after_failed_score_preserves_original_reason_and_cleanup_flag():
    def fail(*a, **k):
        raise ValueError('secret')
    with pytest.raises(runner.FixedFitClosed, match='objective_stream_failed') as exc:
        run(score_rows=fail, release=fail)
    assert exc.value.cleanup_failed


def test_prepared_context_is_registered_before_postprepare_checkpoint_failure():
    callbacks = Callbacks(); ready = False
    def prepare(ids):
        nonlocal ready
        c = callbacks.prepare(ids); ready = True; return c
    def checkpoint():
        if ready:
            raise TimeoutError('private time')
    with pytest.raises(runner.FixedFitClosed, match='checkpoint_failed'):
        run(callbacks, prepare=prepare, checkpoint=checkpoint)
    assert len(callbacks.prepared) == len(callbacks.released) == 1


@pytest.mark.parametrize('where', ['prepare', 'score', 'minimize', 'release', 'checkpoint'])
def test_borrowed_ids_folds_lambdas_mutation_guards_before_after_callbacks(where):
    callbacks = Callbacks(); ids = list(IDS); folds = list(FOLDS); lambdas = list(LAMBDAS)
    def mutate(): ids[0] = 'changed'
    def prepare(train):
        out = callbacks.prepare(train)
        if where == 'prepare': mutate()
        return out
    def rows(*a, **k):
        if where == 'score': mutate()
        return callbacks.score_rows(*a, **k)
    def minimize(*a, **k):
        if where == 'minimize': mutate()
        return callbacks.minimize(*a, **k)
    def release(context):
        if where == 'release': mutate()
        callbacks.release(context)
    calls = 0
    def checkpoint():
        nonlocal calls
        calls += 1
        if where == 'checkpoint' and calls == 2: mutate()
    with pytest.raises(runner.FixedFitClosed, match='input_mutation'):
        run(callbacks, record_ids=ids, folds=folds, lambdas=lambdas, prepare=prepare,
            score_rows=rows, minimize=minimize, release=release, checkpoint=checkpoint)
    assert len(callbacks.prepared) == len(callbacks.released)


@pytest.mark.parametrize('mode', ['extra', 'missing', 'reversed', 'wrong_fingerprint'])
def test_full_score_stream_unchanged_order_exhaustion_and_genuine_parameter_checks(mode):
    callbacks = Callbacks()
    def rows(context, ids, theta, *, alpha):
        result = list(callbacks.score_rows(context, ids, theta, alpha=alpha))
        if mode == 'extra': result += result[:1]
        if mode == 'missing': result = result[:-1]
        if mode == 'reversed': result.reverse()
        if mode == 'wrong_fingerprint':
            result[0] = (result[0][0], replace(result[0][1], parameter_fingerprint='f'*64), result[0][2])
        return iter(result)
    with pytest.raises(runner.FixedFitClosed, match='objective_stream_failed'):
        run(callbacks, score_rows=rows)
    assert len(callbacks.released) == 1


def test_b_scalar_failure_after_all_a_fits_releases_final_context_without_rescue():
    callbacks = Callbacks()
    def minimize(function, start, **kwargs):
        if len(start) == 1:
            return SimpleNamespace(x=start, success=False, nit=0, status=1)
        return callbacks.minimize(function, start, **kwargs)
    with pytest.raises(runner.FixedFitClosed, match='solver_unsuccessful') as exc:
        run(callbacks, minimize=minimize)
    assert exc.value.phase == 'solve_B' and len(callbacks.solves) == 13
    assert len(callbacks.prepared) == len(callbacks.released) == 5


def test_b_no_informative_preserves_phase_not_generic_objective_error():
    callbacks = Callbacks(); b_started = False
    def minimize(function, start, **kwargs):
        nonlocal b_started
        if len(start) == 1: b_started = True
        result = callbacks.minimize(function, start, **kwargs)
        return result
    def rows(context, ids, theta, *, alpha):
        if b_started:
            return ((identifier, score(theta, alpha), np.zeros((2, 3), bool)) for identifier in ids)
        return callbacks.score_rows(context, ids, theta, alpha=alpha)
    with pytest.raises(runner.FixedFitClosed, match='no_informative') as exc:
        run(callbacks, score_rows=rows, minimize=minimize)
    assert exc.value.phase == 'objective_B' and exc.value.evaluations == 2
    assert len(callbacks.released) == 5


@pytest.mark.parametrize('counter,expected', [
    ('SECRET_SENTINEL', None), (True, None), (-1, None), (41, None),
    (None, None), (0, 0), (40, 40),
])
def test_forged_solver_counter_cannot_escape_fixed_safe_diagnostics(counter, expected):
    callbacks = Callbacks()
    def minimize(*a, **kwargs):
        raise SolverClosed('minimizer_failed', evaluations=counter)
    with pytest.raises(runner.FixedFitClosed, match='minimizer_failed') as exc:
        run(callbacks, minimize=minimize)
    assert exc.value.evaluations == expected
    assert 'SECRET' not in str(exc.value) and 'SECRET' not in repr(vars(exc.value))
    assert len(callbacks.prepared) == len(callbacks.released) == 1


def test_minimizer_cannot_swallow_once_failed_rows_and_return_a_success():
    callbacks = Callbacks(); row_calls = 0; swallowed = []; solver_calls = 0
    def rows(*a, **kwargs):
        nonlocal row_calls
        row_calls += 1
        if row_calls == 2:
            raise ValueError('SECRET_SENTINEL')
        return callbacks.score_rows(*a, **kwargs)
    def minimize(function, start, **kwargs):
        nonlocal solver_calls
        solver_calls += 1
        try:
            function(start)
        except SolverClosed as exc:
            swallowed.append(exc.reason)
        return SimpleNamespace(x=start, success=True, nit=0, status=0)
    with pytest.raises(runner.FixedFitClosed, match='objective_stream_failed') as exc:
        run(callbacks, score_rows=rows, minimize=minimize)
    assert exc.value.phase == 'objective_A'
    assert swallowed == ['objective_failed'] and solver_calls == 1 and row_calls == 2
    assert len(callbacks.released) == 1


def test_minimizer_cannot_swallow_once_failed_checkpoint_then_return_success():
    callbacks = Callbacks(); fail_next = False; swallowed = []; solver_calls = 0
    def checkpoint():
        nonlocal fail_next
        if fail_next:
            fail_next = False
            raise TimeoutError('SECRET_SENTINEL')
    def minimize(function, start, **kwargs):
        nonlocal fail_next, solver_calls
        solver_calls += 1; fail_next = True
        try:
            function(start)
        except SolverClosed as exc:
            swallowed.append(exc.reason)
        return SimpleNamespace(x=start, success=True, nit=0, status=0)
    with pytest.raises(runner.FixedFitClosed, match='checkpoint_failed') as exc:
        run(callbacks, checkpoint=checkpoint, minimize=minimize)
    assert exc.value.phase == 'checkpoint'
    assert swallowed == ['checkpoint_failed'] and solver_calls == 1
    assert len(callbacks.released) == 1


def test_runtime_model_data_imports_remain_absent():
    source = Path(__file__).resolve().parents[1]/'src'
    code = f"import sys; sys.path.insert(0,{str(source)!r}); import world_reward.coherent_pair_fit_runner; assert not any(x in sys.modules for x in ('torch','scipy','scipy.optimize'))"
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

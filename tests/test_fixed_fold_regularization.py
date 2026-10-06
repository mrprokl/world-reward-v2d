"""Manufactured callback-only CV; no labels, solvers, devices or files."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from world_reward.fixed_fold_regularization import (
    FixedFoldClosed, FixedFoldSelection, FoldLoss, select_fixed_fold_regularization,
)


IDS = tuple(f'opaque{i:02}' for i in range(8))
FOLDS = tuple(i % 4 for i in range(8))
LAMBDAS = (.25, 1., 4.)


def select(ids=IDS, folds=FOLDS, lambdas=LAMBDAS, *, fit=None, evaluate=None,
           checkpoint=lambda: None):
    fit = fit or (lambda train, lam, fold: (train, lam, fold))
    evaluate = evaluate or (lambda artifact, held, *, regularization:
                           np.full(len(held), artifact[1], np.float64))
    return select_fixed_fold_regularization(ids, folds, lambdas, fit, evaluate, checkpoint)


def test_all_twelve_fits_then_one_final_once_complete_order_and_unpenalized_validation():
    fits = []; evaluations = []; checkpoints = []
    def fit(train, lam, fold):
        fits.append((train, lam, fold)); return fits[-1]
    def evaluate(artifact, held, *, regularization):
        evaluations.append((artifact, held, regularization))
        return np.full(len(held), (artifact[1]-1.)**2, np.float64)
    out = select(fit=fit, evaluate=evaluate, checkpoint=lambda: checkpoints.append(1))
    assert type(out) is FixedFoldSelection and out.status == 'complete'
    assert not out.artifact_integrity_verified
    assert len(fits) == 13 and len(evaluations) == 12 and len(out.ledger) == 12
    for i, (fold, lam) in enumerate((f, l) for f in range(4) for l in LAMBDAS):
        train = tuple(x for x, f in zip(IDS, FOLDS) if f != fold)
        held = tuple(x for x, f in zip(IDS, FOLDS) if f == fold)
        assert fits[i] == (train, lam, fold)
        assert evaluations[i] == (fits[i], held, 0.)
        row = out.ledger[i]
        assert type(row) is FoldLoss and (row.fold, row.regularization) == (fold, lam)
        assert (row.train_ids, row.held_ids) == (train, held)
    assert fits[-1] == (IDS, 1., None) and out.final_artifact is fits[-1]
    assert out.selected_index == 1 and out.selected_regularization == 1.
    np.testing.assert_array_equal(out.mean_held_losses, [0.5625, 0., 9.])
    assert len(checkpoints) == 51


def test_unequal_folds_macro_weights_each_fixed_record_not_each_fold():
    ids = ('a', 'b', 'c', 'd'); folds = (0, 1, 1, 1)
    def evaluate(artifact, held, *, regularization):
        lam = artifact[1]
        values = {'a': 6. if lam == 1 else 0., 'b': 0. if lam == 1 else 3.,
                  'c': 0. if lam == 1 else 3., 'd': 0. if lam == 1 else 3.}
        return np.array([values[x] for x in held], np.float64)
    out = select(ids, folds, (1., 2.), evaluate=evaluate)
    np.testing.assert_array_equal(out.mean_held_losses, [1.5, 2.25])
    assert out.selected_regularization == 1.  # Equal-fold averaging would choose 2.


def test_exact_tie_stronger_lambda_and_caller_order_no_epsilon():
    evaluate = lambda artifact, held, *, regularization: np.ones(len(held), np.float64)
    out = select(lambdas=(4., .25, 1.), evaluate=evaluate)
    assert out.lambdas == (4., .25, 1.) and out.selected_index == 0
    def near(artifact, held, *, regularization):
        return np.full(len(held), 1. if artifact[1] == .25 else np.nextafter(1., 2.), np.float64)
    assert select(evaluate=near).selected_regularization == .25


def test_fold_numbers_need_not_contiguous_and_one_lambda_is_valid():
    out = select(('b', 'a'), (9, 2), (2.,))
    assert [r.fold for r in out.ledger] == [2, 9]
    assert out.ledger[0].held_ids == ('a',) and out.ledger[1].held_ids == ('b',)


def test_metadata_arrays_owned_sealed_and_artifact_is_explicitly_opaque():
    borrowed = []; artifact = {'caller_owned': []}
    def fit(train, lam, fold):
        return artifact
    def evaluate(a, held, *, regularization):
        value = np.arange(len(held), dtype=np.float64); borrowed.append(value); return value
    out = select(fit=fit, evaluate=evaluate)
    assert out.final_artifact is artifact
    artifact['caller_owned'].append('not certified')
    for a in borrowed:
        a[:] = 999.
    for row in out.ledger:
        np.testing.assert_array_equal(row.record_losses, [0., 1.])
    for a in (*[r.record_losses for r in out.ledger], out.mean_held_losses):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        out.selected_index = 2
    with pytest.raises(FrozenInstanceError):
        out.ledger[0].fold = 3


@pytest.mark.parametrize('change', [
    {'ids': ()}, {'ids': ('a', 'a')}, {'ids': ('a', True)}, {'ids': ('a', '')},
    {'ids': ('a', 'bad\x00')}, {'ids': ('a', 'b'*257)}, {'ids': np.array(IDS)},
    {'folds': (0,)*8}, {'folds': (0, 1)}, {'folds': (0, 1, 2, 3, True, 1, 2, 3)},
    {'folds': (-1, 1, 2, 3, 0, 1, 2, 3)}, {'folds': (0., 1., 2., 3., 0., 1., 2., 3.)},
    {'lambdas': ()}, {'lambdas': (0.,)}, {'lambdas': (-1.,)}, {'lambdas': (True,)},
    {'lambdas': (1, 1.)}, {'lambdas': (np.nan,)}, {'lambdas': (np.inf,)},
    {'lambdas': (np.float64(1.),)}, {'lambdas': (10**400,)}, {'lambdas': iter((1.,))},
    {'lambdas': (2**53, 2**53+1)},
])
def test_invalid_inputs_fail_before_callbacks(change):
    called = []
    with pytest.raises(FixedFoldClosed, match='invalid_inputs') as exc:
        select(**change, fit=lambda *a: called.append(a), checkpoint=lambda: called.append(0))
    assert not called and exc.value.completed == 0 and exc.value.status == 'INCONCLUSIVE'


@pytest.mark.parametrize('position', ['fit', 'evaluate', 'checkpoint'])
def test_noncallable_rejected(position):
    args = dict(record_ids=IDS, fold_assignments=FOLDS, lambdas=LAMBDAS,
                fit=lambda *a: object(), evaluate=lambda *a, **k: None, checkpoint=lambda: None)
    args[position] = None
    with pytest.raises(FixedFoldClosed, match='invalid_inputs'):
        select_fixed_fold_regularization(**args)


@pytest.mark.parametrize('value', [None, np.array([0., np.nan]), np.array([np.inf, 0.]),
    np.array([1., 2.], np.float32), np.ones(1), np.ones((2, 1)), [1., 2.],
    (1., 2.), np.array(0., np.float64), np.array([True, False])])
def test_missing_or_invalid_held_losses_close_all_fixed_records(value):
    calls = []
    def fit(*a):
        calls.append(a); return object()
    with pytest.raises(FixedFoldClosed, match='invalid_held_losses') as exc:
        select(fit=fit, evaluate=lambda *a, **k: value)
    assert len(calls) == 1 and exc.value.completed == 0
    assert (exc.value.fold, exc.value.lambda_index) == (0, 0)


def test_overflow_of_finite_loss_sum_closes_no_final_fit():
    fits = []
    def fit(*a):
        fits.append(a); return object()
    with pytest.raises(FixedFoldClosed, match='loss_overflow') as exc:
        select(fit=fit, evaluate=lambda a, h, **k: np.full(len(h), 1e308))
    assert len(fits) == exc.value.completed == 12


def test_stable_cancelling_sum_retains_fixed_record_weight():
    losses = {IDS[i]: x for i, x in enumerate((1e300, 1., -1e300, 1., 1., 1., 1., 1.))}
    out = select(evaluate=lambda a, h, **k: np.array([losses[x] for x in h], np.float64))
    assert (out.mean_held_losses == .75).all()


@pytest.mark.parametrize('where,reason', [('fit', 'fit_failed'), ('evaluate', 'evaluation_failed'),
                                         ('final', 'final_fit_failed'), ('checkpoint', 'checkpoint_failed')])
def test_callback_errors_sanitized_no_retry_or_secret(where, reason):
    fits = []; evaluations = []
    def fit(*a):
        fits.append(a)
        if where == 'fit' or (where == 'final' and a[-1] is None):
            raise RuntimeError('SECRET_SENTINEL https://private.invalid/?token=x')
        return object()
    def evaluate(a, held, **kwargs):
        evaluations.append(held)
        if where == 'evaluate':
            raise RuntimeError('SECRET_SENTINEL')
        return np.zeros(len(held), np.float64)
    def checkpoint():
        if where == 'checkpoint':
            raise ValueError('SECRET_SENTINEL')
    with pytest.raises(FixedFoldClosed, match=reason) as exc:
        select(fit=fit, evaluate=evaluate, checkpoint=checkpoint)
    assert 'SECRET' not in str(exc.value) and 'private' not in str(exc.value)
    assert len(fits) == (13 if where == 'final' else 0 if where == 'checkpoint' else 1)
    assert len(evaluations) == (12 if where == 'final' else 1 if where == 'evaluate' else 0)


@pytest.mark.parametrize('final', [False, True])
def test_none_artifact_is_missing_not_a_success(final):
    fits = []
    def fit(*a):
        fits.append(a); return None if not final or a[-1] is None else object()
    with pytest.raises(FixedFoldClosed, match='missing_artifact') as exc:
        select(fit=fit, evaluate=lambda a, h, **k: np.zeros(len(h), np.float64))
    assert len(fits) == (13 if final else 1) and exc.value.completed == (12 if final else 0)


@pytest.mark.parametrize('which', [0, 1, 2])
@pytest.mark.parametrize('where', ['fit', 'evaluate', 'checkpoint', 'final'])
def test_borrowed_input_mutation_caught_around_every_callback(which, where):
    inputs = [list(IDS), list(FOLDS), list(LAMBDAS)]; checks = 0
    def mutate():
        inputs[which][0] = ('changed', 8, 2.)[which]
    def fit(*a):
        if where == 'fit' or (where == 'final' and a[-1] is None):
            mutate()
        return (a[0], a[1], a[2])
    def evaluate(a, h, **k):
        if where == 'evaluate':
            mutate()
        return np.ones(len(h), np.float64)
    def checkpoint():
        nonlocal checks
        checks += 1
        if where == 'checkpoint' and checks == 2:
            mutate()
    with pytest.raises(FixedFoldClosed, match='input_mutation'):
        select(*inputs, fit=fit, evaluate=evaluate, checkpoint=checkpoint)


def test_borrowed_list_type_change_equal_numeric_value_is_mutation():
    folds = list(FOLDS)
    def fit(*a):
        folds[0] = False; return object()
    with pytest.raises(FixedFoldClosed, match='input_mutation'):
        select(folds=folds, fit=fit)


def test_loss_vector_changed_by_postcallback_checkpoint_closes_before_averaging():
    borrowed = np.zeros(2, np.float64); calls = 0
    def evaluate(*a, **k):
        return borrowed
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 4:
            borrowed[0] = 100.
    with pytest.raises(FixedFoldClosed, match='held_losses_mutation'):
        select(evaluate=evaluate, checkpoint=checkpoint)


def test_bounded_checkpoint_caps_are_not_caught_as_fit_success_or_restarted():
    fits = []; calls = 0
    def fit(*a):
        fits.append(a); return a
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 11:
            raise TimeoutError('private deadline')
    with pytest.raises(FixedFoldClosed, match='checkpoint_failed') as exc:
        select(fit=fit, checkpoint=checkpoint)
    assert len(fits) == 3 and exc.value.completed == 2


def test_callback_cannot_smuggle_untrusted_reason_through_our_exception_type():
    def fit(*a):
        raise FixedFoldClosed('SECRET_SENTINEL')
    with pytest.raises(FixedFoldClosed, match='fit_failed') as exc:
        select(fit=fit)
    assert 'SECRET' not in str(exc.value)


def test_module_has_no_solver_device_or_data_imports():
    source = Path(__file__).resolve().parents[1]/'src'
    code = f"""import sys
sys.path.insert(0, {str(source)!r})
import world_reward.fixed_fold_regularization
assert not any(x in sys.modules for x in ('torch', 'scipy', 'scipy.optimize'))
"""
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

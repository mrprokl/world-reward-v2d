"""Caller-context A-CV/refit then frozen-geometry B; no data/device/recipe.

Context provenance, train-only scales, scorer snapshots and alpha0 parity remain
caller obligations. This composes qualified numerical utilities, not a new loss.
"""
from dataclasses import dataclass
import math

import numpy as np

from .bounded_analytic_optimizer import SolverClosed, SolverLimits, solve_zero_start
from .coherent_pair_stream_objective import ordered_stream_marginal_objective
from .fixed_fold_regularization import FixedFoldClosed, select_fixed_fold_regularization


class FixedFitClosed(ValueError):
    def __init__(self, reason, phase, *, evaluations=None, fold=None, lambda_index=None):
        self.reason, self.phase, self.evaluations = reason, phase, evaluations
        self.fold, self.lambda_index = fold, lambda_index
        self.status, self.cleanup_failed = 'INCONCLUSIVE', False
        super().__init__(reason)
@dataclass(frozen=True, eq=False)
class _GeometryFit:
    solution: object
    objective: object
@dataclass(frozen=True, eq=False)
class CoherentPairFitResult:
    cross_validation: object
    geometry: object
    relation: object
    geometry_record_statuses: tuple
    relation_record_statuses: tuple
    geometry_counts: object
    relation_counts: object
    status: str = 'complete'
    context_integrity_verified: bool = False
_SOLVER_REASONS = frozenset(('invalid_limits', 'invalid_inputs', 'checkpoint_failed',
    'evaluation_cap', 'invalid_parameters', 'objective_failed', 'invalid_objective',
    'parameter_mutation', 'minimizer_failed', 'invalid_result', 'objective_drift',
    'iteration_cap', 'solver_unsuccessful', 'loss_increased', 'nonstationary'))


def fit_coherent_pair_ab(record_ids, folds, lambdas, prepare, score_rows, *,
                        limits_a, limits_b, regularization_b, backend,
                        checkpoint, minimize, release):
    """Prepare each fold once across lambdas, refit A once, then solve B once.

    prepare(train_ids)->opaque context. score_rows(context, ids, theta,
    alpha=...)->one-pass (id,genuine_score,full_mask) rows using that context's
    TRAIN-ONLY scales even for held records. release(context) is mandatory,
    including failure cleanup; at most one context survives a fold transition.
    A starts geometry17 at zero; B starts scalar alpha0 with frozen final A.
    All limits/regularization/backend are explicit; no restart or best-fit rescue.
    Partial training loss retains fixed N; no-informative training closes.
    Held missing loss remains NaN and closes CV, not a smaller validation mean.
    """
    try:
        valid_lam = type(regularization_b) in (int, float) and math.isfinite(regularization_b) and regularization_b >= 0
    except (OverflowError, ValueError):
        valid_lam = False
    if (not all(type(x) in (tuple, list) for x in (record_ids, folds, lambdas))
            or type(limits_a) is not SolverLimits or type(limits_b) is not SolverLimits
            or not valid_lam or type(backend) is not str or backend not in ('numpy', 'torch')
            or not all(map(callable, (prepare, score_rows, checkpoint, minimize, release)))):
        raise FixedFitClosed('invalid_inputs', 'inputs')
    supplied = (record_ids, folds, lambdas)
    snapshots = tuple(tuple((type(x), x) for x in a) for a in supplied)
    ids = tuple(record_ids); context = None; context_key = None
    pending = None; failure = None; result = None
    def check():
        nonlocal pending
        for a, saved in zip(supplied, snapshots):
            if len(a) != len(saved) or any(type(x) is not kind or x != value
                    for x, (kind, value) in zip(a, saved)):
                pending = FixedFitClosed('input_mutation', 'checkpoint'); raise pending
        try:
            checkpoint()
        except Exception:
            pending = FixedFitClosed('checkpoint_failed', 'checkpoint'); raise pending from None
        for a, saved in zip(supplied, snapshots):
            if len(a) != len(saved) or any(type(x) is not kind or x != value
                    for x, (kind, value) in zip(a, saved)):
                pending = FixedFitClosed('input_mutation', 'checkpoint'); raise pending
    def call(function, reason, phase, *args, cleanup=False, **kwargs):
        error = None
        try:
            check()
        except FixedFitClosed as exc:
            error = exc
        if error is None or cleanup:
            try:
                value = function(*args, **kwargs)
            except Exception:
                error = error or FixedFitClosed(reason, phase)
        try:
            check()
        except FixedFitClosed as exc:
            error = error or exc
        if error is not None:
            raise error
        return value
    def dispose():
        nonlocal context, context_key
        if context is not None:
            owned = context; context = context_key = None
            call(release, 'release_failed', 'release', owned, cleanup=True)
    def objective(which_ids, theta, alpha, lam, arm):
        scoring_theta = np.frombuffer(theta.tobytes(), dtype=np.float64)
        value = call(lambda: ordered_stream_marginal_objective(
            score_rows(context, which_ids, scoring_theta, alpha=alpha), which_ids, theta,
            alpha=alpha, regularization=lam, arm=arm, backend=backend),
            'objective_stream_failed', 'objective_'+arm)
        return value
    def solve(evaluate, dimension, limits, nonnegative):
        error = None
        def guarded(parameters):
            nonlocal error
            try:
                return evaluate(parameters)
            except FixedFitClosed as exc:
                error = exc; raise
        try:
            result = solve_zero_start(guarded, dimension, limits, nonnegative=nonnegative,
                                      checkpoint=check, minimize=minimize)
        except SolverClosed as exc:
            reason = exc.reason if exc.reason in _SOLVER_REASONS else 'solver_closed'
            error = error or pending or FixedFitClosed(reason, 'solve_'+('B' if nonnegative else 'A'))
            error.evaluations = (exc.evaluations if type(exc.evaluations) is int
                and 0 <= exc.evaluations <= limits.max_evaluations else None)
            raise error from None
        if error is not None or pending is not None:
            raise error or pending
        return result
    def fit_a(train_ids, lam, fold):
        nonlocal context, context_key, pending
        try:
            key = fold, train_ids
            if key != context_key:
                dispose()
                def build():
                    nonlocal context
                    context = prepare(train_ids); return context
                call(build, 'prepare_failed', 'prepare')
                if context is None:
                    raise FixedFitClosed('missing_context', 'prepare')
                context_key = key
            last = None
            def evaluate(theta):
                nonlocal last
                last = objective(train_ids, theta, 0., lam, 'A')
                if last.status == 'no_informative':
                    raise FixedFitClosed('no_informative', 'objective_A')
                return last.loss, last.gradient
            solution = solve(evaluate, 17, limits_a, False)
            return _GeometryFit(solution, last)
        except FixedFitClosed as exc:
            pending = exc
            raise
    def held(artifact, held_ids, *, regularization):
        nonlocal pending
        try:
            if regularization != 0.:
                raise FixedFitClosed('held_regularization', 'held')
            return objective(held_ids, artifact.solution.parameters, 0., 0., 'A').record_losses
        except FixedFitClosed as exc:
            pending = exc
            raise
    try:
        cv = select_fixed_fold_regularization(record_ids, folds, lambdas, fit_a, held, check)
        final_a = cv.final_artifact; last_b = None
        def evaluate_b(alpha):
            nonlocal last_b
            last_b = objective(ids, final_a.solution.parameters, float(alpha[0]), regularization_b, 'B')
            if last_b.status == 'no_informative':
                raise FixedFitClosed('no_informative', 'objective_B')
            return last_b.loss, last_b.gradient
        relation = solve(evaluate_b, 1, limits_b, True)
        result = CoherentPairFitResult(cv, final_a.solution, relation,
            final_a.objective.record_statuses, last_b.record_statuses,
            final_a.objective.counts, last_b.counts)
    except FixedFoldClosed as exc:
        failure = pending or FixedFitClosed(exc.reason, 'cross_validation',
            fold=exc.fold, lambda_index=exc.lambda_index)
    except FixedFitClosed as exc:
        failure = exc
    except Exception:
        failure = FixedFitClosed('fit_pipeline_failed', 'pipeline')
    finally:
        try:
            dispose()
        except FixedFitClosed as exc:
            if failure is None:
                failure = exc
            else:
                failure.cleanup_failed = True
    if failure is not None:
        raise failure from None
    return result

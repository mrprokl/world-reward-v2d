"""Caller-fixed complete-fold regularization selection; no solver, data or recipe.

Opaque artifacts are NOT copied, frozen or authenticated here. The caller owns
their integrity/provenance; only CV metadata and owned FP64 loss arrays are sealed.
Missing held-record losses close INCONCLUSIVE, never change the denominator.
"""
from dataclasses import dataclass
import math

import numpy as np


class FixedFoldClosed(ValueError):
    def __init__(self, reason, fold=None, lambda_index=None, completed=0):
        self.reason, self.fold, self.lambda_index = reason, fold, lambda_index
        self.completed, self.status = completed, 'INCONCLUSIVE'
        super().__init__(reason)


@dataclass(frozen=True, eq=False)
class FoldLoss:
    fold: int
    lambda_index: int
    regularization: float
    train_ids: tuple
    held_ids: tuple
    record_losses: np.ndarray


@dataclass(frozen=True, eq=False)
class FixedFoldSelection:
    record_ids: tuple
    fold_assignments: tuple
    lambdas: tuple
    ledger: tuple
    mean_held_losses: np.ndarray
    selected_index: int
    selected_regularization: float
    final_artifact: object
    status: str = 'complete'
    artifact_integrity_verified: bool = False


def _sealed(a):
    return np.frombuffer(a.tobytes(order='C'), dtype=np.float64).reshape(a.shape)


def select_fixed_fold_regularization(record_ids, fold_assignments, lambdas,
                                     fit, evaluate, checkpoint):
    """Every ascending fold × caller-ordered lambda, then one final full fit.

    ``fit(train_ids, lambda, fold)`` returns a non-None opaque artifact;
    ``evaluate(artifact, held_ids, regularization=0.)`` returns a plain finite
    FP64 per-record vector in exact held-ID order. The caller validates statuses
    and aligns losses to IDs; NaN/None is never a usable held-record loss here.
    Lambda means weight every fixed image equally, even with unequal folds.
    Exact minimum ties choose stronger regularization, without an epsilon.
    Callback failures/caps close; this function performs no retry or fallback.
    """
    fold = index = None; completed = 0
    def close(reason):
        raise FixedFoldClosed(reason, fold, index, completed) from None
    supplied = (record_ids, fold_assignments, lambdas)
    if (not all(type(x) in (tuple, list) for x in supplied)
            or not all(map(callable, (fit, evaluate, checkpoint)))):
        close('invalid_inputs')
    ids, assignments, candidates = map(tuple, supplied)
    try:
        valid = (bool(ids) and all(type(x) is str and 0 < len(x) <= 256
            and x.isprintable() for x in ids) and len(set(ids)) == len(ids)
            and len(assignments) == len(ids)
            and all(type(x) is int and x >= 0 for x in assignments)
            and len(set(assignments)) >= 2 and bool(candidates)
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                    for x in candidates) and len(set(candidates)) == len(candidates))
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        close('invalid_inputs')
    snapshots = tuple(tuple((type(x), x) for x in a) for a in supplied)
    def unchanged():
        for borrowed, saved in zip(supplied, snapshots):
            if len(borrowed) != len(saved) or any(type(x) is not kind or x != value
                    for x, (kind, value) in zip(borrowed, saved)):
                close('input_mutation')
    def check():
        unchanged()
        try:
            checkpoint()
        except Exception:
            close('checkpoint_failed')
        unchanged()
    def call(function, reason, *args, capture_losses=False, **kwargs):
        check()
        try:
            try:
                result = function(*args, **kwargs)
            except Exception:
                close(reason)
            if capture_losses:
                if (type(result) is not np.ndarray or result.dtype != np.float64
                        or result.shape != (len(args[1]),) or not np.isfinite(result).all()):
                    close('invalid_held_losses')
                owned = _sealed(result.copy())
        finally:
            check()
        if capture_losses:
            if (result.dtype != np.float64 or result.shape != owned.shape
                    or result.tobytes(order='C') != owned.tobytes(order='C')):
                close('held_losses_mutation')
            return owned
        return result
    candidates = tuple(float(x) for x in candidates)
    if len(set(candidates)) != len(candidates):
        close('invalid_inputs')
    ledger = []; by_lambda = [[] for _ in candidates]
    for fold in sorted(set(assignments)):
        train = tuple(x for x, f in zip(ids, assignments) if f != fold)
        held = tuple(x for x, f in zip(ids, assignments) if f == fold)
        for index, regularization in enumerate(candidates):
            artifact = call(fit, 'fit_failed', train, regularization, fold)
            if artifact is None:
                close('missing_artifact')
            losses = call(evaluate, 'evaluation_failed', artifact, held,
                          capture_losses=True, regularization=0.)
            ledger.append(FoldLoss(fold, index, regularization, train, held, losses))
            by_lambda[index].extend(float(x) for x in losses)
            completed += 1
            del artifact
    fold = index = None
    try:
        means = np.array([math.fsum(x)/len(ids) for x in by_lambda], np.float64)
    except (OverflowError, ValueError):
        close('loss_overflow')
    if not np.isfinite(means).all():
        close('loss_overflow')
    best = min(range(len(candidates)), key=lambda i: (means[i], -candidates[i]))
    index = best
    final = call(fit, 'final_fit_failed', ids, candidates[best], None)
    if final is None:
        close('missing_artifact')
    check()
    return FixedFoldSelection(ids, assignments, candidates, tuple(ledger), _sealed(means),
                              best, candidates[best], final)

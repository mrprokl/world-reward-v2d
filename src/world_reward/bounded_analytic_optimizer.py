"""One caller-bounded zero-start analytic solve; no recipe, SciPy import or FIT."""
from dataclasses import dataclass
import math

import numpy as np


class SolverClosed(ValueError):
    """Fixed diagnostic only; never include upstream exception text or parameters."""
    def __init__(self, reason, evaluations=0, iterations=None):
        self.reason, self.evaluations, self.iterations = reason, evaluations, iterations
        super().__init__(reason)


@dataclass(frozen=True)
class SolverLimits:
    maxiter: int
    max_evaluations: int
    maxls: int
    maxcor: int
    ftol: float
    gtol: float
    projected_stationarity: float

    def __post_init__(self):
        integers = (self.maxiter, self.max_evaluations, self.maxls, self.maxcor)
        if not all(type(x) is int and x > 0 for x in integers) or self.max_evaluations < 2:
            raise SolverClosed('invalid_limits')
        if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                   for x in (self.ftol, self.gtol, self.projected_stationarity)):
            raise SolverClosed('invalid_limits')


def _sealed(value):
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, eq=False)
class AnalyticSolveResult:
    parameters: np.ndarray
    initial_loss: float
    final_loss: float
    gradient: np.ndarray
    projected_gradient_inf: float
    evaluations: int
    iterations: int
    status: int
    success: bool


def solve_zero_start(evaluate, dimension, limits, *, nonnegative=False, checkpoint, minimize):
    """One L-BFGS-B call with independently evaluated initial/final points.

    Limits include ALL callbacks: initial, minimizer and final verification.
    Minimize receives maxfun=max_evaluations-2, reserving those two checks;
    the callback hard cap remains authoritative. Bounds are none or [0,+inf]
    for every coordinate. All checkpoints/metadata/objective failures close;
    no best iterate, restart, fallback, numerical tolerance or budget is chosen.
    """
    calls = 0; iterations = None
    def close(reason):
        raise SolverClosed(reason, calls, iterations) from None
    if (type(limits) is not SolverLimits or type(dimension) is not int or dimension <= 0
            or type(nonnegative) is not bool or not all(map(callable, (evaluate, checkpoint, minimize)))):
        close('invalid_inputs')
    def check():
        try:
            checkpoint()
        except Exception:
            close('checkpoint_failed')
    def parameters(x):
        if (type(x) is not np.ndarray or x.dtype != np.float64 or x.shape != (dimension,)
                or not np.isfinite(x).all() or (nonnegative and (x < 0).any())):
            close('invalid_parameters')
        return np.array(x, copy=True)
    def function(x):
        nonlocal calls
        check()
        if calls >= limits.max_evaluations:
            close('evaluation_cap')
        calls += 1
        try:
            owned = parameters(x); before = owned.tobytes()
            try:
                value = evaluate(owned)
            except Exception:
                close('objective_failed')
            if type(value) is not tuple or len(value) != 2:
                close('invalid_objective')
            loss, gradient = value
            if (type(loss) not in (float, np.float64) or not math.isfinite(loss)
                    or type(gradient) is not np.ndarray or gradient.dtype != np.float64
                    or gradient.shape != (dimension,) or not np.isfinite(gradient).all()):
                close('invalid_objective')
            answer = float(loss), np.array(gradient, copy=True)
        finally:
            check()
        if (owned.shape != (dimension,) or owned.dtype != np.float64 or owned.tobytes() != before
                or x.shape != (dimension,) or x.dtype != np.float64 or x.tobytes() != before):
            close('parameter_mutation')
        return answer
    start = np.zeros(dimension, np.float64)
    initial_loss, initial_gradient = function(start)
    options = dict(maxiter=limits.maxiter, maxfun=limits.max_evaluations-2,
        maxls=limits.maxls, maxcor=limits.maxcor, ftol=float(limits.ftol), gtol=float(limits.gtol))
    check()
    try:
        result = minimize(function, start.copy(), method='L-BFGS-B', jac=True,
            bounds=[(0., None)]*dimension if nonnegative else None, options=options)
    except SolverClosed:
        raise
    except Exception:
        close('minimizer_failed')
    finally:
        check()
    try:
        success, iterations, status = result.success, result.nit, result.status
        final = parameters(result.x); final_bytes = final.tobytes()
    except SolverClosed:
        raise
    except Exception:
        close('invalid_result')
    if type(success) is not bool or type(iterations) is not int or iterations < 0 or type(status) is not int:
        close('invalid_result')
    final_loss, gradient = function(final)
    if parameters(result.x).tobytes() != final_bytes:
        close('parameter_mutation')
    if np.array_equal(final, start) and (final_loss != initial_loss
            or gradient.tobytes() != initial_gradient.tobytes()):
        close('objective_drift')
    projected = gradient.copy()
    if nonnegative:
        projected[final == 0.] = np.minimum(projected[final == 0.], 0.)
    norm = float(np.abs(projected).max())
    if iterations > limits.maxiter:
        close('iteration_cap')
    if not success:
        close('solver_unsuccessful')
    if final_loss > initial_loss:
        close('loss_increased')
    if norm > limits.projected_stationarity:
        close('nonstationary')
    check()
    if parameters(result.x).tobytes() != final_bytes or final.tobytes() != final_bytes:
        close('parameter_mutation')
    return AnalyticSolveResult(_sealed(final), initial_loss, final_loss, _sealed(gradient),
        norm, calls, iterations, status, success)

"""Manufactured CPU and fake-device controls; no Torch, FIT or file data."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from world_reward import coherent_pair_marginal_objective as objective
from world_reward import coherent_pair_stream_objective as stream
from world_reward import coherent_route_scorer as core

spec = importlib.util.spec_from_file_location('stream_original_objective_controls',
    Path(__file__).with_name('test_coherent_pair_marginal_objective.py'))
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
THETA = controls.THETA


def run(scores, masks, *, theta=THETA, alpha=0., regularization=.25, arm='A',
        backend='numpy', ids=None):
    if ids is None:
        ids = tuple(f'image:{i}' for i in range(len(scores)))
    rows = ((identifier, score, mask) for identifier, score, mask in zip(ids, scores, masks))
    return stream.ordered_stream_marginal_objective(rows, ids, theta, alpha=alpha,
        regularization=regularization, arm=arm, backend=backend)


def parity(actual, expected):
    assert type(actual) is objective.MarginalObjective
    np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.record_losses, expected.record_losses,
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    for field in ('counts', 'record_statuses', 'status', 'score_references', 'arm',
                  'regularization', 'fixed_population_average', 'unlisted_alternatives_verified_negative'):
        assert getattr(actual, field) == getattr(expected, field)


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', 0.), ('B', .3125)])
def test_full_population_tuple_parity_missing_allpositive_and_partial(arm, alpha):
    good = np.ones((2, 3), bool); good[0, 0] = False
    scores = (controls.scores(alpha=alpha),
              controls.scores(alpha=alpha, supported=np.zeros((2, 3), bool)),
              controls.scores(alpha=alpha), controls.scores(alpha=alpha, supported=good))
    masks = (controls.positive(), controls.positive(), np.ones((2, 3), bool), controls.positive())
    before = core._fingerprint((scores, masks, THETA))
    actual = run(scores, masks, alpha=alpha, arm=arm)
    expected = objective.marginal_objective(scores, masks, THETA, alpha=alpha,
                                           regularization=.25, arm=arm)
    parity(actual, expected)
    assert actual.status == 'partial' and actual.counts['records'] == 4
    assert actual.counts['missing_positive'] == actual.counts['no_alternative'] == 1
    assert actual.counts['partially_missing_positive'] == 1
    assert before == core._fingerprint((scores, masks, THETA))
    for value in (actual.gradient, actual.record_losses):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        actual.counts['used'] = 9


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', .3125)])
def test_fixed_denominator_ridge_once_and_gradient_fd(arm, alpha):
    mask = controls.positive(); empty = np.zeros((2, 3), bool)
    def evaluate(theta, a):
        return run((controls.scores(theta, a), controls.scores(theta, a, supported=empty)),
                   (mask, mask), theta=theta, alpha=a, arm=arm, regularization=.125)
    actual = evaluate(THETA, alpha)
    singleton = objective.marginal_objective((controls.scores(alpha=alpha),), (mask,),
                                            THETA, alpha=alpha, arm=arm)
    parameter = THETA if arm == 'A' else np.array([alpha])
    np.testing.assert_allclose(actual.loss, singleton.loss/2 + .0625*(parameter@parameter))
    for j in range(17 if arm == 'A' else 1):
        delta = np.zeros(17); delta[j] = 1e-6
        hi = evaluate(THETA+delta, alpha) if arm == 'A' else evaluate(THETA, alpha+1e-6)
        lo = evaluate(THETA-delta, alpha) if arm == 'A' else evaluate(THETA, alpha-1e-6)
        np.testing.assert_allclose((hi.loss-lo.loss)/2e-6, actual.gradient[j], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('kind', ['no_positive', 'missing_positive', 'no_alternative', 'zero_person'])
def test_no_informative_keeps_original_status_and_nan(kind):
    score = controls.scores(alpha=0.); mask = controls.positive()
    if kind == 'no_positive':
        mask[:] = False
    elif kind == 'missing_positive':
        score = controls.scores(alpha=0., supported=np.zeros((2, 3), bool))
    elif kind == 'no_alternative':
        mask[:] = True
    else:
        identity = dict(score.identity)
        identity.update(source_person_ids=(), person_members=(), person_to_group=np.empty(0, np.int64))
        score = replace(score, identity=identity, supported=np.empty((0, 3), bool),
            native_supported=np.empty((0, 2, 3), bool), native_route_supported=np.empty((0, 2, 3), bool),
            native_scores_a=np.empty((0, 2, 3)), native_scores_b=np.empty((0, 2, 3)),
            scores_a=np.empty((0, 3)), scores_b=np.empty((0, 3)),
            geometry_derivatives_a=np.empty((0, 3, 17)), geometry_derivatives_b=np.empty((0, 3, 17)),
            alpha_derivatives_b=np.empty((0, 3)))
        mask = np.empty((0, 3), bool)
    actual = run((score,), (mask,), regularization=0.)
    expected = objective.marginal_objective((score,), (mask,), THETA)
    parity(actual, expected)
    assert actual.status == 'no_informative' and actual.loss == 0. and not actual.gradient.any()
    assert np.isnan(actual.record_losses[0]) if kind != 'no_alternative' else actual.record_losses[0] == 0.


@pytest.mark.parametrize('rows,ids', [
    ([], ('a',)), ([('b', None, None)], ('a',)),
    ([('a', None, None), ('a', None, None)], ('a', 'b')),
    ([('b', None, None), ('a', None, None)], ('a', 'b')),
    ([('a', None)], ('a',)), ([['a', None, None]], ('a',)),
])
def test_incomplete_reordered_or_malformed_stream_fails(rows, ids):
    with pytest.raises(ValueError):
        stream.ordered_stream_marginal_objective(iter(rows), ids, THETA)


def test_extra_and_missing_after_valid_row_are_not_partial_success():
    row = ('a', controls.scores(alpha=0.), controls.positive())
    for rows, ids in (([row, row], ('a',)), ([row], ('a', 'b'))):
        with pytest.raises(ValueError):
            stream.ordered_stream_marginal_objective(iter(rows), ids, THETA)


def test_iterator_read_once_including_exact_exhaustion_and_exception_propagation():
    row = ('a', controls.scores(alpha=0.), controls.positive()); visited = []
    class Rows:
        def __iter__(self):
            return self
        def __next__(self):
            visited.append(len(visited))
            if len(visited) == 1:
                return row
            raise StopIteration
    stream.ordered_stream_marginal_objective(Rows(), ('a',), THETA)
    assert visited == [0, 1]
    def broken():
        yield row
        raise RuntimeError('Authored interrupted stream')
    with pytest.raises(RuntimeError, match='Authored interrupted stream'):
        stream.ordered_stream_marginal_objective(broken(), ('a',), THETA)


@pytest.mark.parametrize('ids', [(), ('a', 'a'), ['a'], (1,), ('',), ('a\n',)])
def test_bad_fixed_ids_rejected_before_consumption(ids):
    def rows():
        pytest.fail('Invalid fixed IDs must fail before consuming any row')
        yield
    with pytest.raises(ValueError):
        stream.ordered_stream_marginal_objective(rows(), ids, THETA)


@pytest.mark.parametrize('change', [
    {'theta': np.zeros(17, np.float32)}, {'theta': np.zeros(16)}, {'theta': np.full(17, np.nan)},
    {'alpha': True}, {'alpha': -1.}, {'alpha': .25}, {'alpha': np.inf},
    {'regularization': True}, {'regularization': -1.}, {'regularization': np.nan},
    {'arm': 'C'}, {'backend': 'jax'}, {'rows': []}, {'rows': None},
])
def test_invalid_call_parameters(change):
    args = dict(rows=iter([('a', controls.scores(alpha=0.), controls.positive())]),
                record_ids=('a',), theta=THETA)
    args.update(change)
    with pytest.raises(ValueError):
        stream.ordered_stream_marginal_objective(**args)


@pytest.mark.parametrize('bad', [
    lambda s: replace(s, parameter_fingerprint='b'*64),
    lambda s: replace(s, distribution_fingerprint='not-a-hash'),
    lambda s: replace(s, scores_a=np.zeros((2, 3), np.float32)),
    lambda s: object(),
])
def test_original_score_validation_is_not_bypassed(bad):
    with pytest.raises(ValueError):
        run((bad(controls.scores(alpha=0.)),), (controls.positive(),))


def test_uniform_temperature_with_valid_individual_fingerprints_required():
    score = controls.scores(alpha=0.)
    other = replace(score, temperature=1., parameter_fingerprint=core._fingerprint((THETA, 1., 0.)))
    with pytest.raises(ValueError, match='uniform stream temperature'):
        run((score, other), (controls.positive(), controls.positive()))


@pytest.mark.parametrize('mask', [np.ones((2, 3), np.int64), np.ones((2, 2), bool)])
def test_full_original_mask_validator_required(mask):
    with pytest.raises(ValueError):
        run((controls.scores(alpha=0.),), (mask,))


def test_extreme_score_offset_stability_and_nonfinite_derivative_rejection():
    theta = np.zeros(17); score = controls.scores(theta, 0., shift=1e300)
    out = run((score,), (controls.positive(),), theta=theta, regularization=0.)
    np.testing.assert_allclose(out.loss, np.log(3.), rtol=1e-15)
    broken = replace(score, geometry_derivatives_a=np.full((2, 3, 17), np.inf))
    with pytest.raises(ValueError):
        run((broken,), (controls.positive(),), theta=theta)


def test_finite_input_regularizer_overflow_fails_not_clipped():
    theta = np.full(17, 1e308)
    score = replace(controls.scores(np.zeros(17), 0.),
                    parameter_fingerprint=core._fingerprint((theta, .8125, 0.)))
    before = core._fingerprint((score, theta))
    with pytest.raises((ValueError, FloatingPointError)):
        run((score,), (np.zeros((2, 3), bool),), theta=theta, regularization=1.)
    assert core._fingerprint((score, theta)) == before


def test_mutation_during_iterator_protocol_is_not_hidden():
    theta = THETA.copy(); score = controls.scores(theta, 0.)
    class Rows:
        def __iter__(self):
            theta[0] += 1.
            return self
        def __next__(self):
            if hasattr(self, 'done'):
                raise StopIteration
            self.done = True
            return 'a', score, controls.positive()
    with pytest.raises(ValueError, match='Borrowed geometry changed'):
        stream.ordered_stream_marginal_objective(Rows(), ('a',), theta)


def test_mutating_borrowed_theta_from_iterator_fails_without_result():
    theta = THETA.copy(); score = controls.scores(theta, 0.)
    def rows():
        yield 'a', score, controls.positive()
        theta[0] += 1.
    with pytest.raises(ValueError, match='Borrowed geometry changed'):
        stream.ordered_stream_marginal_objective(rows(), ('a',), theta)


@pytest.mark.parametrize('arm,alpha', [('A', 0.), ('B', 0.), ('B', .3125)])
def test_lazy_fake_torch_owned_cpu_output_parity(monkeypatch, arm, alpha):
    monkeypatch.setitem(sys.modules, 'torch', controls.fake_torch())
    for name, function in (('detach', lambda self: self), ('cpu', lambda self: self),
                           ('numpy', lambda self: self.array)):
        monkeypatch.setattr(controls.Tensor, name, function, raising=False)
    scores = (controls.scores(alpha=alpha), controls.scores(alpha=alpha, supported=np.zeros((2, 3), bool)))
    masks = (controls.positive(), controls.positive())
    expected = objective.marginal_objective(scores, masks, THETA, alpha=alpha, arm=arm, regularization=.25)
    parity(run(tuple(map(controls.device_scores, scores)), masks, alpha=alpha, arm=arm, backend='torch'), expected)


def test_lazy_torch_stream_rejects_different_individually_valid_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', controls.fake_torch())
    for name, function in (('detach', lambda self: self), ('cpu', lambda self: self),
                           ('numpy', lambda self: self.array)):
        monkeypatch.setattr(controls.Tensor, name, function, raising=False)
    first = controls.device_scores(controls.scores(alpha=0.))
    second = controls.device_scores(controls.scores(alpha=0.))
    for field in vars(second).values():
        if type(field) is controls.Tensor:
            field.device = 'cpu'
    second = replace(second, device='cpu')
    with pytest.raises(ValueError, match='stream scorer device'):
        run((first, second), (controls.positive(), controls.positive()), backend='torch')


def test_real_packed_cpu_score_consumed_without_rebuilding_or_pruning():
    spec = importlib.util.spec_from_file_location('stream_original_marginal_controls',
        Path(__file__).with_name('test_coherent_pair_marginal.py'))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    from world_reward.coherent_pair_learning import PairScale
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    bank = fixture.bank(n=1, o=2, copies=1, aliases=True, missing=True)
    packed = prepare_marginal_packed(prepare_pair_cache(bank, PairScale(np.ones(12), np.ones(12, bool))))
    before = core._fingerprint(packed)
    score = score_pair_marginal_packed(packed, THETA, temperature=1., alpha=0.)
    mask = np.ones(score.supported.shape, bool)
    parity(run((score,), (mask,)), objective.marginal_objective((score,), (mask,), THETA, regularization=.25))
    assert before == core._fingerprint(packed) and score.native_supported.shape == (1, 2, 2)


def test_module_import_does_not_import_torch_or_optimizer():
    root = Path(__file__).resolve().parents[1]
    script = f"""import sys
sys.path.insert(0, {str(root/'src')!r})
import world_reward.coherent_pair_stream_objective
assert 'torch' not in sys.modules and 'scipy.optimize' not in sys.modules
"""
    subprocess.run([sys.executable, '-I', '-B', '-c', script], check=True, capture_output=True)

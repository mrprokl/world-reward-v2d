"""AST-only prototype checks: no Torch import, execution or GPU qualification."""
import ast
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1]/'src/world_reward/coherent_pair_packed_torch.py'
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {node.name:node for node in TREE.body if isinstance(node, ast.FunctionDef)}
CLASSES = {node.name:node for node in TREE.body if isinstance(node, ast.ClassDef)}


def calls(node):
    return [ast.unparse(x.func) for x in ast.walk(node) if isinstance(x, ast.Call)]


def function_source(name):
    return ast.get_source_segment(SOURCE, FUNCTIONS[name])


def test_source_parse_and_budget():
    assert len(SOURCE.splitlines()) <= 300
    assert set(FUNCTIONS) == {'_csr', '_indices', '_inverse', 'prepare_marginal_packed_torch',
        '_finite', '_segment', '_mix', '_geometry_vjp', '_targets', '_pairs', 'score_pair_marginal_packed_torch'}


def test_torch_is_imported_only_inside_preparation():
    imports = [x for x in ast.walk(TREE) if isinstance(x, ast.Import)
        and any(y.name == 'torch' for y in x.names)]
    assert len(imports) == 1
    assert imports[0] in list(ast.walk(FUNCTIONS['prepare_marginal_packed_torch']))
    assert not any(isinstance(x, ast.ImportFrom) and x.module == 'torch' for x in ast.walk(TREE))
    assert 'TorchMarginalPairScores' in CLASSES and 'TorchMarginalPairPacked' in CLASSES


def test_factory_disallows_caller_tables():
    klass = CLASSES['TorchMarginalPairPacked']
    constructor = next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
    assert len(constructor.args.args) == 1 and constructor.args.vararg is None
    assert isinstance(constructor.body[0], ast.Raise)
    src = function_source('prepare_marginal_packed_torch')
    assert src.index('before == packed.packed_fingerprint') < src.index('import torch')
    assert src.index("== c.factor_fingerprint") < src.index('import torch')
    assert src.index('for prefix in') < src.index('import torch')
    assert src.index('Host changed during device preparation') > src.index('_arrays=upload(selected)')
    assert 'torch.tensor(np.array(value, copy=True), device=device)' in src


def test_explicit_device_and_mutable_snapshot_caveat():
    assert "r'cpu|cuda:[0-9]+'" in function_source('prepare_marginal_packed_torch')
    assert 'mutable Torch buffers, NOT immutable evidence' in ast.get_docstring(TREE)
    assert 'CPU/GPU 1e-12 agreement is proposed' in ast.get_docstring(TREE)
    assert 'device' in [x.target.id for x in CLASSES['TorchMarginalPairScores'].body if isinstance(x, ast.AnnAssign)]


@pytest.mark.parametrize('name', ['_csr', '_indices'])
def test_fixed_layout_checks_are_before_upload(name):
    assert '_require' in calls(FUNCTIONS[name])
    assert 'np.int64' in function_source(name)
    assert 'import torch' not in function_source(name)
    assert '_csr' in calls(FUNCTIONS['prepare_marginal_packed_torch'])


def test_csr_and_ids_checks_include_complete_bounds():
    src = function_source('_csr')
    for part in ('offsets[0] == 0', 'offsets[-1] == entries', '(targets+1,)', 'np.diff(offsets) >= 0'):
        assert part in src
    assert '(values >= 0) & (values < count)' in function_source('_indices')


def test_no_autograd_scatter_or_host_score_array_transfer():
    banned = {'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'index_add', 'index_add_',
        'backward', 'autograd', 'cpu', 'numpy', 'detach', 'tolist'}
    assert not any(isinstance(x, ast.Attribute) and x.attr in banned for x in ast.walk(TREE))
    score = FUNCTIONS['score_pair_marginal_packed_torch']
    contexts = [x for x in score.body if isinstance(x, ast.With)]
    assert len(contexts) == 1
    assert ast.unparse(contexts[0].items[0].context_expr) == 't.no_grad()'


def test_segment_api_is_two_dimensional_with_explicit_axis():
    node = FUNCTIONS['_segment']; call = next(x for x in ast.walk(node)
        if isinstance(x, ast.Call) and ast.unparse(x.func) == 't.segment_reduce')
    assert [ast.unparse(x) for x in call.args] == ['data', 'op']
    assert {x.arg:ast.unparse(x.value) for x in call.keywords} == {'offsets':'offsets', 'axis':'0'}
    src = function_source('_segment')
    assert "data = x.reshape(-1, 1) if flat else x" in src
    assert src.index('if data.shape[0] == 0:') < src.index('t.segment_reduce')


def test_stable_lme_preserves_uniform_count_and_empty_support():
    src = function_source('_mix')
    assert "if not x.numel():" in src
    assert "float('nan')" in src and 'lengths > 0' in src
    assert "_segment(t, x, 'max', offsets)" in src
    assert 'x-maximum[ids]' in src and 't.log(safe_z)' in src
    assert 't.log(lengths.clamp_min(1).to(x.dtype))' in src
    assert 'posterior = weights/total[ids]' in src


def test_alpha_zero_shared_and_right_derivative_kept():
    src = function_source('_targets')
    assert 'b, wb = a, wa' in src
    assert "_segment(t, margins, 'sum', layout['moff'])/layout['mlengths'].to(g.dtype)" in src
    assert 'db = da if alpha == 0.' in src
    assert "dm = _segment(t, wb*conditional, 'sum', layout['offsets'])" in src
    assert 'if alpha == 0.: b, wb, db = a, wa, da' in function_source('_pairs')


def test_whole_route_factors_and_feature_column_order():
    src = function_source('score_pair_marginal_packed_torch')
    assert 'base[comp[:, 0]]+local[comp[:, 1]]+bridge[comp[:, 2]]' in src
    assert 'coef[[0, 1, 2, 3, 4, 5, 12, 13, 14]]' in src
    assert 'coef[[6, 7, 8, 9, 15, 16]]' in src
    assert "t.cat((base[:, :6], local[:, :4], bridge, base[:, 6:], local[:, 4:]), dim=1)" in function_source('_geometry_vjp')


def test_inverse_component_csr_and_no_per_object_loop():
    src = function_source('_inverse')
    assert 'np.lexsort((np.arange(len(targets)), components, targets))' in src
    assert 'component_offsets' in src and 'target_offsets' in src
    vjp = function_source('_geometry_vjp')
    assert "posterior[inv['order']]" in vjp and "inv['component_offsets']" in vjp
    assert "inv['target_offsets']" in vjp
    runtime = ('_mix', '_segment', '_targets', '_pairs', '_geometry_vjp', 'score_pair_marginal_packed_torch')
    loops = [x for name in runtime for x in ast.walk(FUNCTIONS[name]) if isinstance(x, (ast.For, ast.While))]
    assert len(loops) == 1 and ast.unparse(loops[0].iter) == "('base', 'local', 'bridge')"


def test_support_priors_and_native_slots_not_pruned():
    src = function_source('prepare_marginal_packed_torch')
    assert 'np.array_equal(prior, expected)' in src
    assert "a['native_supported'].reshape(-1), layouts['native']['lengths'] > 0" in src
    fields = [x.target.id for x in CLASSES['TorchMarginalPairScores'].body if isinstance(x, ast.AnnAssign)]
    assert fields[5:16] == ['native_scores_a', 'native_scores_b', 'native_supported', 'native_route_supported',
        'scores_a', 'scores_b', 'supported', 'geometry_derivatives_a', 'geometry_derivatives_b', 'alpha_derivatives_b', 'device']
    assert 'loss' not in FUNCTIONS and 'fit' not in FUNCTIONS


def test_finite_checks_and_small_host_parameter_fingerprint():
    assert 't.isfinite(x).all()' in function_source('_finite')
    assert '_finite' in calls(FUNCTIONS['_mix'])
    assert '_finite' in calls(FUNCTIONS['_geometry_vjp'])
    score = function_source('score_pair_marginal_packed_torch')
    assert 'theta.shape == (17,)' in score and 'core._fingerprint((theta, tau, alpha))' in score
    assert 'core._fingerprint((packed.arrays' not in score

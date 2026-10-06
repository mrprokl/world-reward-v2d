"""Authored transport/native-shape fixtures only; no data/models/quality."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import vcoco_full_interaction_core as p
import vcoco_interaction_observations as old
from test_vcoco_interaction_observations import fixture, identities


def at_slot(args, slot, ordinal=None):
    args = [deepcopy(a) for a in args]
    ordinal = slot if ordinal is None else ordinal
    for i in (0, 2, 4):
        args[i].update(original_slot=slot, acquired_ordinal=ordinal, file=f'image_{slot:06d}.npz')
    for i in (3, 5):
        args[i]['original_slot'] = np.asarray(slot, np.int64)
        args[i]['acquired_ordinal'] = np.asarray(ordinal, np.int64)
        args[i-1]['arrays'] = identities(args[i])
    return args


def compare(actual, expected):
    assert (actual.image_id, actual.original_slot, actual.acquired_ordinal) == (expected.image_id, expected.original_slot, expected.acquired_ordinal)
    for left, right in ((actual.endpoint_arrays, expected.endpoint_arrays), (actual.pose_arrays, expected.pose_arrays), (actual.hoi_arrays, expected.hoi_arrays)):
        assert identities(left) == identities(right)
    for name in ('features', 'feature_supported', 'route_features', 'route_supported'):
        a, b = getattr(actual.evidence, name), getattr(expected.evidence, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert actual.evidence.source_person_ids == expected.evidence.source_person_ids
    assert actual.evidence.source_observation_references == expected.evidence.source_observation_references


@pytest.mark.parametrize('n,k', [(2, True), (0, True), (2, False), (0, False)])
def test_default16_is_byte_equal_and_keeps_historical_slot_holes(n, k):
    args, _ = fixture(n, k)
    compare(p.reconstruct_interaction(*args), old.reconstruct_interaction(*args))
    assert p.reconstruct_interaction(*args).acquired_ordinal == 1


@pytest.mark.parametrize('slot', [0, 15, 16, 31, 47])
@pytest.mark.parametrize('n,k', [(2, True), (0, False)])
def test_explicit48_preserves_original_slots_complete_raw_and_empty_banks(slot, n, k):
    args, _ = fixture(n, k)
    full = at_slot(args, slot)
    actual = p.reconstruct_interaction(*full, population=48)
    base = p.reconstruct_interaction(*at_slot(args, 0))
    assert actual.original_slot == actual.acquired_ordinal == slot
    assert actual.evidence.features.shape == (n*2*3600, 10)
    assert actual.hoi.query_logits.shape == (1500, 3)
    assert actual.owl.patch_ids.shape == (3600,)
    assert actual.evidence.features.tobytes() == base.evidence.features.tobytes()
    assert actual.evidence.route_features.tobytes() == base.evidence.route_features.tobytes()
    assert identities(actual.pose_arrays) == identities(full[3])
    assert not actual.evidence.scope['scoring_performed']
    if slot >= 16:
        with pytest.raises(ValueError): old.reconstruct_interaction(*full)


@pytest.mark.parametrize('population', [True, False, 0, 15, 17, 64, 48.0, '48', None])
def test_population_not_guessed(population):
    args, _ = fixture(0, False)
    with pytest.raises(ValueError): p.reconstruct_interaction(*args, population=population)


@pytest.mark.parametrize('slot,ordinal', [(-1, -1), (48, 48), (16, 15), (47, 0)])
def test_full48_no_compression_relabel_or_out_of_bound(slot, ordinal):
    args, _ = fixture(0, False)
    with pytest.raises(ValueError): p.reconstruct_interaction(*at_slot(args, slot, ordinal), population=48)


@pytest.mark.parametrize('fault', ['dtype', 'flags', 'foreign_person', 'missing', 'cross_grid', 'pair_drop', 'identity'])
def test_general48_transport_guards_remain_fail_closed(fault):
    args, _ = fixture()
    args = at_slot(args, 47)
    if fault == 'dtype':
        args[3]['raw_scores'] = args[3]['raw_scores'].astype(np.float64); args[2]['arrays'] = identities(args[3])
    elif fault == 'flags':
        args[3]['native_valid'] = ~args[3]['native_valid']; args[2]['arrays'] = identities(args[3])
    elif fault == 'foreign_person': args[4]['source_person_ids'] = ['foreign']
    elif fault == 'missing': args[3] = None
    elif fault == 'cross_grid': args[2]['image_size'] = [12, 8]
    elif fault == 'pair_drop':
        args[5]['hand_object_pairs'] = args[5]['hand_object_pairs'][:-1]; args[4]['arrays'] = identities(args[5])
    else: args[4]['endpoint_bank_identity']['sha256'] = 'c'*64
    with pytest.raises(ValueError): p.reconstruct_interaction(*args, population=48)


def test_frozen_original_unchanged_and_numerical_ast_identical():
    root = Path(__file__).resolve().parents[1]
    original = root/'infra/vcoco_interaction_observations.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == 'e8333e3180fe009f08bb14201b20919a6bec823164d19c7f21d2124eae56a2a1'
    a = ast.parse(original.read_text()); b = ast.parse((root/'infra/vcoco_full_interaction_core.py').read_text())
    functions = lambda tree: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    x, y = functions(a), functions(b)
    for name in ('_require', '_identity', '_snapshot'):
        assert ast.dump(x[name], include_attributes=False) == ast.dump(y[name], include_attributes=False)
    old_body, new_body = x['reconstruct_interaction'].body, y['reconstruct_interaction'].body
    # Explicit population policy is the only new prelude; native numerical body
    # begins at n-person census m, after the three population-aware grid calls.
    start = lambda body: next(i for i,n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'm' for t in n.targets))
    assert ast.dump(ast.Module(body=old_body[start(old_body):], type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=new_body[start(new_body):], type_ignores=[]), include_attributes=False)


def test_imports_are_public_and_lazy_no_old_producer():
    root = Path(__file__).resolve().parents[1]
    script = """import importlib.abc,runpy,sys
sys.path[:0]=[sys.argv[1]+'/infra',sys.argv[1]+'/src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in{'numpy','torch','transformers','PIL','vcoco_interaction_observations','vcoco_person_pose_observations','vcoco_role_reference','vcoco_fit_cal_context'}:raise ImportError('private/eager import')
sys.meta_path.insert(0,Deny())
runpy.run_path(sys.argv[1]+'/infra/vcoco_full_interaction_core.py')
"""
    assert subprocess.run([sys.executable, '-I', '-B', '-c', script, str(root)], capture_output=True).returncode == 0

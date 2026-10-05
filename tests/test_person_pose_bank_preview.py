"""Tiny procedural display/source tests; no local challenge media or models."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import person_pose_bank_preview as preview


def test_display_is_separate_from_native_validity_and_never_mutates():
    points = np.zeros((2, 133, 2), np.float64)
    valid = np.ones((2, 133), bool); inside = valid.copy()
    valid[0, 9] = False; inside[1, 112] = False
    before = points.copy(), valid.copy(), inside.copy()
    mask = preview.display_points(points, valid, inside)
    assert mask.sum() == 264 and not mask[0, 9] and not mask[1, 112]
    for a, b in zip((points, valid, inside), before): np.testing.assert_array_equal(a, b)


def test_empty_points_are_not_fabricated():
    assert preview.display_points(np.empty((0, 133, 2)), np.empty((0, 133), bool),
                                  np.empty((0, 133), bool)).shape == (0, 133)
    with pytest.raises(ValueError):
        preview.display_points(np.zeros((1, 21, 2)), np.ones((1, 21), bool), np.ones((1, 21), bool))


def test_scope_and_source_do_not_select_or_infer():
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root/preview.CONFIG).read_bytes())
    assert c['maximum_preview_bytes'] == 180000 and not c['model_execution'] and not c['ownership_verified']
    assert sum(n for _, _, n in preview.EXPECTED_BANKS) == 14 and len(preview.EXPECTED_BANKS) == 6
    tree = ast.parse((root/'infra/person_pose_bank_preview.py').read_text())
    render = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render'))
    assert 'allow_pickle=False' in render and 'decoded_RGB_sha256' in render
    assert 'range(bank[\'persons\'])' in render and 'argmax' not in render
    calls = [n for n in ast.walk(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render'))
             if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Name) and n.func.id.startswith('infer') for n in calls)
    main = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
    assert "'--gpus'" not in main and "'--network', 'none'" in main

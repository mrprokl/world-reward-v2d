"""Tiny source/protocol-only QA; actual rendering stays on Azure."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_preview_source_and_scope():
    p = ROOT/'infra/hoi_observation_preview.py'
    s = importlib.util.spec_from_file_location('wr_hoi_preview_test', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    raw = (ROOT/m.CONFIG).read_bytes(); c = json.loads(raw)
    assert m.CONFIG_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert c['all_native_proposals_and_pairs'] and not c['manual_annotations'] and not c['reference_geometry_used']
    assert c['maximum_preview_bytes'] == 95000 and c['maximum_width'] == 640
    tree = ast.parse(p.read_bytes())
    render = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render'))
    assert 'allow_pickle=False' in render and 'hand_object_logits' in render and 'object_target_logits' in render
    assert 'for (i, j), raw in zip(pairs, logits)' in render
    assert 'softmax' not in render and 'argmax' not in render
    run = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    assert "'--gpus'" not in run and 'reference_geometry' not in c['observations']
    assert "'--network', 'none'" in run and "'--user', '1000:1000'" in run

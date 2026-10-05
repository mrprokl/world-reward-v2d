"""Compile an authenticated original verifier with only its replay replaced.

The native QSlim/qualification source remains untouched. The original source,
input, full I/J/quotient/position/component/Euler/boundary gates execute exactly;
only local face incidence and an exact filtered orientation predicate differ.
Not a simplifier, new surface acceptance domain or reconstruction-quality claim.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

from world_reward.exact_normal_dot import positive_normal_dot
from world_reward.surface_mapping_replay import replay_surface_mapping

SOURCE_SHA256 = '59bf5bbc9ae4eae77ed0757fb3a8c9f9475458ea14b8c74c0cdcfca6996b2a74'
FUNCTION_SHA256 = '74c3b447a6cd37ba1d890e6baf3ce7dfc3e2d0465716f6da88ba88786fb629bd'


def require_orientation(before, after):
    decisions, _ = positive_normal_dot(before, after)
    if not decisions.all():
        raise ValueError('Exact triangle orientation reversed/collapsed')


def verifier(q):
    """Bind the exact original outside-replay AST; no caller-supplied source."""
    path = Path(q.__file__).resolve()
    if path.name != 'surface_qslim_qualify.py':
        raise ValueError('Actual original qualified verifier module required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unmodified original qualified verifier bytes required')
    text = raw.decode()
    node = next(n for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == 'verify_mapping')
    if hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest() != FUNCTION_SHA256:
        raise ValueError('Exact original verifier AST required')
    if len(node.body) != 30 or not isinstance(node.body[16], ast.For) or not isinstance(node.body[17], ast.FunctionDef):
        raise ValueError('Original replay region differs')
    original = copy.deepcopy(node)
    replacement = ast.parse("""
state = replay_surface_mapping(v, f, mapping['ledger'], boundary_vertices=boundary_vertices,
    orientation=orientation, orientation_batch=require_orientation)
parent, live, positions, removed = state.parent, state.live, state.positions, state.removed_faces
def root(n):
    return int(state.quotient[n])
""").body
    node.body = node.body[:12] + replacement + node.body[18:]
    # All original complete gates before/after the replay must remain verbatim.
    if (ast.dump(ast.Module(body=node.body[:12], type_ignores=[])) !=
            ast.dump(ast.Module(body=original.body[:12], type_ignores=[])) or
        ast.dump(ast.Module(body=node.body[15:], type_ignores=[])) !=
            ast.dump(ast.Module(body=original.body[18:], type_ignores=[]))):
        raise ValueError('Original outside-replay gates changed')
    node.name = 'verify_mapping_local_incidence'
    scope = dict(q.__dict__, replay_surface_mapping=replay_surface_mapping,
        require_orientation=require_orientation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(path)+'::qualified-local-replay', 'exec'), scope)
    return scope[node.name]

"""Exact original DWPose metadata validators, without NumPy/runtime imports.

This is a host-only source seam, not a runtime or model qualification. The
caller authenticates whole producer closures and calls the returned validators.
Only pinned original metadata definitions and literal constants are evaluated;
no package install, source main, array decode or session code is selected.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mediapipe_cpu_runtime_verify as rt

PINS = {
    'infra/dwpose_smoke.py': dict(bytes=28479,sha256='ea0beb43dd698261a5b2accdbd0432dcc8de82b56958b0e93cec8062089ce72c'),
    'infra/run_dwpose_smoke.sh': dict(bytes=3558,sha256='c4142525731c619d2e2c313920c886b7fab2e7fb5ea33b46ccceaa42b8fa6604'),
    'infra/keypoint_rgb_dwpose.py': dict(bytes=16781,sha256='f6fbf3e578e2a3c603a9644f341401576cc6360694ec52cf909907ba00771121'),
    'infra/dwpose_acquire.py': dict(bytes=17584,sha256='9a5c24de16fe6ec9169b836f1cba8e435e3d9cbb3e1ac159ca416b42363246ea'),
    'infra/dwpose_wheel_audit.py': dict(bytes=13898,sha256='4f43263edcb366c06a310a51cb2300057fe3ee1357fc02c8c36013dc29464e97')}
HELPERS = ('infra/dwpose_metadata_context.py','infra/mediapipe_cpu_runtime_verify.py',*PINS)
SMOKE_NAMES = ('identity','source_identity','validate_previous_smoke','validate_replay')
SMOKE_CONSTANTS = ('OUT','STAGE','PREVIOUS_SMOKE','PREVIOUS_SMOKE_SHA','PREVIOUS_SMOKE_REVISION','PREVIOUS_SMOKE_SOURCE_SHA','AUDIT_SOURCE_SHA')
CAPABILITY_NAMES = ('require_fields','validate_smoke')
CAPABILITY_CONSTANTS = ('SMOKE_SHA','SMOKE_REVISION','SMOKE_SOURCE_SHA')


def _definitions(path, raw, names, constants, dependencies):
    tree = ast.parse(raw); namespace = dict(Path=Path,json=json,__file__=str(path),**dependencies)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    rt.require(tuple(n.name for n in nodes)==names,'Exact original metadata definition order')
    values = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in constants:
            name = node.targets[0].id; rt.require(name not in values,'Duplicate metadata constant'); values[name]=ast.literal_eval(node.value)
    rt.require(set(values)==set(constants),'Exact original literal metadata constants'); namespace.update(values)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**namespace)


def metadata_delegate(code):
    """Return (smoke, capability); current source must own every imported helper."""
    code = rt.canonical(code); rt.require(Path(__file__).resolve()==code/'infra/dwpose_metadata_context.py'
        and Path(rt.__file__).resolve()==code/'infra/mediapipe_cpu_runtime_verify.py','Actual metadata seam/dependency origins')
    for name,pin in PINS.items(): rt.require(rt.identity(code/name,2 << 20)==pin,'Original metadata/helper bytes changed')
    acquisition=importlib.import_module('dwpose_acquire'); audit=importlib.import_module('dwpose_wheel_audit')
    for module,name in ((acquisition,'infra/dwpose_acquire.py'),(audit,'infra/dwpose_wheel_audit.py')):
        rt.require(Path(module.__file__).resolve()==code/name and rt.identity(code/name,2 << 20)==PINS[name],'Actual pinned stdlib helper origin')
    smoke_path=code/'infra/dwpose_smoke.py'; capability_path=code/'infra/keypoint_rgb_dwpose.py'
    raws={path:path.read_bytes() for path in (smoke_path,capability_path)}
    rt.require(all(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==PINS['infra/'+path.name]
        for path,raw in raws.items()),'Exact metadata bytes before any definition execution')
    smoke=_definitions(smoke_path,raws[smoke_path],SMOKE_NAMES,SMOKE_CONSTANTS,dict(acquisition=acquisition,audit=audit))
    capability=_definitions(capability_path,raws[capability_path],CAPABILITY_NAMES,CAPABILITY_CONSTANTS,dict(smoke=smoke))
    for name,pin in PINS.items(): rt.require(rt.identity(code/name,2 << 20)==pin,'Metadata helper changed during extraction')
    return smoke,capability

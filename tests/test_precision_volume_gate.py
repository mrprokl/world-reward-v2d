"""Tiny data-free precision/backend contracts, no native job or local assets."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    import exact_mesh_geometry as geometry
    import precision_volume_gate as gate
    import guarded_mesh_gate as guarded
    import object_budget_endpoint as endpoint
    import mesh_link_gate as legacy
    import mesh_endpoint_gate as containment
    return geometry, gate, guarded, endpoint, legacy, containment


def tetra():
    return (np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
            np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]))


def _function(source, name):
    return next(n for n in ast.parse(Path(source).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _strip_docstring(node):
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body = node.body[1:]
    return node


def _normalize_topology_calls(node):
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id == "exact_mesh_topology":
            item.id = "mesh_topology"
    return _strip_docstring(node)


def test_static_helper_math_parity_no_runtime_ast_or_functiontype(modules):
    geometry, _, guarded, endpoint, legacy, containment = modules
    for original, name in ((guarded, "write_obj"), (guarded, "read_obj"),
                           (guarded, "mapped_geometry"), (endpoint, "verify_pack_fidelity"),
                           (containment, "true_hollow_containment")):
        old = _strip_docstring(_function(original.__file__, name))
        new = _normalize_topology_calls(_function(geometry.__file__, name))
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    old = _strip_docstring(_function(legacy.__file__, "mesh_topology"))
    new = _strip_docstring(_function(geometry.__file__, "exact_mesh_topology"))
    new.name = old.name
    start = next(i for i,n in enumerate(old.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "double_area")
    del old.body[start:start+2]
    start = next(i for i,n in enumerate(new.body) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "extent")
    del new.body[start:start+2]
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    text = Path(geometry.__file__).read_text()
    assert not any(token in text for token in ("FunctionType", "__globals__", "compile(", "exec(", "ast."))
    assert geometry.validate_legacy_sources() == geometry.LEGACY_HASHES


def test_graded_closed_corner_falsifies_only_global_area_gate(modules):
    geometry, gate, _, _, legacy, _ = modules
    source = gate.graded_corner(*tetra())
    before = [a.copy() for a in source]
    with pytest.raises(ValueError, match="zero-area"):
        legacy.mesh_topology(*source)
    top = geometry.exact_mesh_topology(*source)
    assert (top["vertices"], top["faces"]) == (7, 10)
    assert top["components"][0]["signed_volume"] == pytest.approx(1/6)
    gate.native_readiness(source)
    assert all(a.tobytes() == b.tobytes() for a,b in zip(source, before))
    assert geometry.mapped_geometry(source, source, gate.identity_mapping(source))["net_volume_relative_error"] == 0


def test_strict_collinear_faces_fail_whole_geometry_no_legacy_retry(modules):
    geometry, gate, *_ = modules
    v, f = gate.graded_corner(*tetra()); v[f[0,1]] = v[f[0,0]]
    before = v.copy(), f.copy()
    with pytest.raises(ValueError, match="Exactly collinear"):
        geometry.exact_mesh_topology(v, f)
    assert v.tobytes() == before[0].tobytes() and f.tobytes() == before[1].tobytes()


def test_float32_predicate_view_changes_no_original_dtype_or_geometry(modules):
    geometry, gate, *_ = modules
    v, f = gate.graded_corner(*tetra()); v = v.astype(np.float32)
    before = v.copy(), f.copy()
    top = geometry.exact_mesh_topology(v, f)
    assert top["faces"] == 10 and v.dtype == np.float32
    assert v.tobytes() == before[0].tobytes() and f.tobytes() == before[1].tobytes()


def test_exact_positive_underflow_cannot_bypass_native_readiness(modules):
    _, gate, *_ = modules
    v, f = tetra()
    with pytest.raises(ValueError, match="native arithmetic"):
        gate.native_readiness((v * 1e-200, f))


def test_native_readiness_uses_double_parser_arithmetic_not_storage_dtype(modules):
    _, gate, *_ = modules
    v,f = tetra();stored = (v * 2.**-30).astype(np.float32)
    before=stored.copy();gate.native_readiness((stored,f))
    assert stored.dtype==np.float32 and stored.tobytes()==before.tobytes()


def test_original_global_volume_guard_not_relaxed_for_tiny_disconnected_shell(modules):
    geometry, *_ = modules
    v, f = tetra(); tiny = v * 2.**-30 + [2.,0.,0.]
    with pytest.raises(ValueError, match="nonzero signed volume"):
        geometry.exact_mesh_topology(np.r_[v,tiny], np.r_[f,f+4])


def test_exact_obj_roundtrip_preserves_graded_original_and_exclusive_output(modules, tmp_path):
    geometry, gate, *_ = modules
    source = gate.graded_corner(*tetra()); p = tmp_path / "source.obj"
    geometry.write_obj(p, *source); read = geometry.read_obj(p)
    assert all(np.array_equal(a,b) for a,b in zip(source,read))
    with pytest.raises(FileExistsError): geometry.write_obj(p,*source)


def test_pack_and_birth_fidelity_does_not_drop_or_reverse_faces(modules):
    geometry, gate, *_ = modules
    v, f = gate.graded_corner(*tetra())
    v,f,_ = gate.endpoint.exact_weld(v,f)
    pv = np.r_[v,np.repeat(v[:1],4096-len(v),axis=0)]
    pf = np.r_[f,np.zeros((4096-len(f),3),dtype=int)]
    compact, fidelity = geometry.verify_pack_fidelity((v,f),pv,pf)
    assert fidelity["active_faces"] == 10
    pmap = gate.packed_mapping((v,f),compact,gate.identity_mapping((v,f)))
    assert geometry.mapped_geometry((v,f),compact,pmap)["net_volume_relative_error"] == 0
    wrong = pf.copy(); wrong[0] = wrong[0,::-1]
    with pytest.raises(ValueError, match="triangle geometry"):
        geometry.verify_pack_fidelity((v,f),pv,wrong)


@pytest.mark.parametrize("fault", ["I", "J", "scale"])
def test_original_birth_mapping_and_five_percent_bands_still_fail(modules, fault):
    geometry, gate, *_ = modules
    v, f = tetra(); source = np.r_[v,v*.8+[3.,0.,0.]], np.r_[f,f+4]
    mapping = gate.identity_mapping(source)
    candidate = tuple(a.copy() for a in source)
    if fault == "I": mapping["I"] = list(range(4))*2
    if fault == "J": mapping["J"] = list(range(4))*2
    if fault == "scale": candidate = candidate[0]*.9,candidate[1]
    with pytest.raises(ValueError): geometry.mapped_geometry(source,candidate,mapping)


def test_containment_body_remains_independent_of_full_mesh_parity(modules, monkeypatch):
    geometry, *_ = modules
    v,f = tetra(); mesh=np.r_[v,v*.1+[.1,.1,.1]],np.r_[f,f[:,::-1]+4]
    # Tiny test-only math substitute; never a real embedding/containment claim.
    monkeypatch.setattr(geometry,"solid_winding_and_distances",lambda p,t:(np.ones(len(p)),np.ones(len(p))))
    result=geometry.true_hollow_containment(*mesh,self_intersecting_faces=0)
    assert result["true_containment_verified"] is True
    with pytest.raises(ValueError,match="zero pair"):
        geometry.true_hollow_containment(*mesh,self_intersecting_faces=1)


def test_large_fixture_declared_native_only_and_closed_when_trimesh_available(modules):
    pytest.importorskip("trimesh")
    geometry, gate, *_ = modules
    fixtures=gate.fixtures(); assert [r[0] for r in fixtures] == gate.FIXTURES
    assert [r[2] for r in fixtures] == [False,True]
    large=fixtures[1][1];top=geometry.exact_mesh_topology(*large)
    assert len(large[1])>4096 and len(top["components"])==2
    assert sorted(r["volume_sign"] for r in top["components"])==[-1,1]


def test_original_build_pins_and_nonexecutable_missing_binary_fail_fast(modules, monkeypatch, tmp_path):
    _, gate, *_ = modules
    (tmp_path/"results").mkdir();p=tmp_path/"results/image-volume-qem.json";p.write_text('{}')
    with pytest.raises(ValueError,match="receipt"):
        gate.validate_build(tmp_path)
    monkeypatch.setattr(gate,"BUILD_BYTES",p.stat().st_size)
    monkeypatch.setattr(gate,"BUILD_SHA",hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(gate,"BINARY_SHA",None);monkeypatch.setenv("WR_IMAGE_ID",gate.IMAGE)
    with pytest.raises(ValueError,match="nonexecutable"):
        gate.validate_build(tmp_path)


def test_network_none_wrapper_cpu_only_no_episode_data_or_gt_mount(modules):
    _, gate, *_ = modules
    p=ROOT/"infra/run_precision_volume_gate.sh";text=p.read_text()
    assert "--network none" in text and "--cpus 4" in text and "--memory 16g" in text
    assert "3603s docker run" in text and "--gpus" not in text
    assert not any(f"src=$ROOT/{folder}" in text for folder in ("outputs","data","weights","validation/volume_qem_v1"))
    assert "--mount \"type=bind,src=$HELPER,dst=$HELPER,readonly\"" in text
    assert "--read-only" in text and "--cap-drop ALL" in text and "no-new-privileges" in text
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m" in text
    assert "--cidfile" in text and "docker stop --time 5" in text and "docker kill" in text
    assert "world-reward.job" in text and "world-reward.revision" in text
    assert 'helper.stat().st_size==2031' in text
    assert 'p.chmod(0o444)' in text
    assert "BINARY_SHA" in text and text.index("BINARY_SHA")<text.index('mkdir -m 700 "$OUT"')
    source=Path(gate.__file__).read_text()
    assert "NATIVE_BUDGET = 3600, 900" in source
    assert 'native_cost_and_placement_unchanged' in source
    assert not any(token in source for token in ("FunctionType","__globals__","compile(","exec("))
    with pytest.raises(SystemExit): gate.main(["--episode","9"])


def test_entrypoint_runtime_source_closure_includes_pinned_math_and_predicates(modules):
    import sys
    sys.path.insert(0,str(ROOT/"infra"))
    from azure_job import runtime_bundle_paths
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ("infra","src","configs")
           for p in (ROOT/folder).rglob('*') if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    paths=runtime_bundle_paths(files,"infra/run_precision_volume_gate.sh")
    assert all(p in paths for p in ("infra/exact_mesh_geometry.py","infra/mesh_volume_qem.cpp",
                                  "infra/mesh_guarded_qem.cpp","src/world_reward/exact_triangle_predicates.py"))


def test_runtime_identity_requires_original_dispatch_before_geometry(modules):
    _,gate,*_=modules
    with pytest.raises(ValueError,match="dispatched runtime"):
        gate.runtime_identity(Path('/tmp/fake'),Path('/tmp/fake/code'),'a'*40)
    node=_function(gate.__file__,'main');calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Name)and n.func.id=='runtime_identity'for n in calls)==2
    assert sum(isinstance(n.func,ast.Attribute)and n.func.attr=='validate_legacy_sources'for n in calls)==1
    assert sum(isinstance(n.func,ast.Name)and n.func.id=='validate_build'for n in calls)==1
    assert 'finally' in Path(gate.__file__).read_text()
    assert 'path.chmod(0o444)' in Path(gate.__file__).read_text()

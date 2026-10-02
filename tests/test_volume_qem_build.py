"""Tiny source/ABI contracts only; no libigl compiler/backend claim locally."""
import ast
from pathlib import Path
import subprocess

import numpy as np
import pytest


INFRA = Path(__file__).resolve().parents[1] / "infra"


def code(name):
    return (INFRA / name).read_text()


def test_base_parser_is_included_not_copied_or_changed():
    source = code("mesh_volume_qem.cpp")
    assert '#define main wr_guarded_base_main' in source
    assert '#include "/opt/world-reward/guarded-qem/mesh_guarded_qem.cpp"' in source
    assert "void read_obj(" not in source and "read_obj(argv[1],V,F)" in source
    assert "require_absent(argv[2]); require_absent(argv[3]);" in source
    assert "WR_BASE_SOURCE_SHA256" in source and "WR_VOLUME_SOURCE_SHA256" in source


def test_single_mechanism_native_cost_global_callbacks_and_full_birth_maps():
    source = code("mesh_volume_qem.cpp")
    assert "igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post)" in source
    assert "igl::intersection_blocking_collapse_edge_callbacks(pre,post,tree,pre,post)" in source
    assert "igl::decimate(V,F,cost,igl::max_faces_stopping_condition(m,F.rows(),TARGET)" in source
    assert '"cost_normalization\\\":false' in source
    assert "vector_json(stream,J,true)" in source and "vector_json(stream,I,true)" in source
    assert "return reached && U.rows()<=TARGET && G.rows()<=TARGET ? 0 : 3" in source
    assert "igl::qslim(V," not in source


def test_signed_cumulative_limit_not_local_or_net_volume_only():
    source = code("mesh_volume_qem.cpp")
    assert "RELATIVE_VOLUME_LIMIT = .05L" in source
    assert "shells[k].current.sum + delta.sum" in source
    assert "proposed * shells[k].source <= 0" in source
    assert "std::abs(proposed - shells[k].source) / std::abs(shells[k].source)" in source
    assert "igl::circulation(e, true, EMAP, EF, EI)" in source
    assert "igl::circulation(e, false, EMAP, EF, EI)" in source
    assert "(F.row(f).array() == IGL_COLLAPSE_EDGE_NULL).all()" in source
    assert "pending_shell = -1" in source
    start = source.index("post = [&state,guarded_post]")
    end = source.index("int m = F.rows();", start)
    post = source[start:end]
    assert post.index("if (collapsed)") < post.index("shell.current.add(state.pending_delta)")
    assert "guarded_post(V,F,E,EMAP,EF,EI,Q,EQ,C,e,e1,e2,f1,f2,collapsed)" in post


def test_final_recompute_and_birth_binding_are_independent_of_queue_state():
    source = code("mesh_volume_qem.cpp")
    assert "const int k = face_shell[J(f)]" in source
    assert "vertex_shell[I(G(f,j))] != k" in source
    assert "final[k].add(term)" in source
    assert "shell.current.sum - shell.final_volume" in source
    assert "std::abs(shell.final_volume - shell.source) / std::abs(shell.source)" in source
    assert "state.verify_final(U,G,J,I)" in source
    assert "std::numeric_limits<long double>::epsilon()" in source


def test_inherited_headers_only_offline_bound_base_image():
    build, wrapper, docker = map(code, ("build_volume_qem.sh", "run_volume_qem_build.sh", "Dockerfile.volume_qem"))
    assert "600s python3" in build and "660s docker build" in wrapper
    assert "--network none" in wrapper and "--pull=false" in wrapper
    assert "--gpus" not in wrapper and "urllib" not in build and "acquire(" not in build
    assert "source_inventory_sha256" in build and "inventory_sha!=inherited.get('source_inventory_sha256')" in build
    assert "sha(base/'mesh_guarded_qem.cpp')!=expected_base" in build
    assert "sha(base/'mesh_guarded_qem')!=inherited.get('binary_sha256')" in build
    assert "world-reward/guarded-qem:0.1" in wrapper
    assert "world-reward/volume-qem:0.1" in wrapper
    assert "image-volume-qem.json" in wrapper
    assert "FROM ${BASE_IMAGE}" in docker and "COPY mesh_guarded_qem.cpp" not in docker
    assert "WR_BASE_CPP_SHA256" in build and "BASE_CPP_SHA256" in docker


def test_frozen_abi_names_agree_with_root_consumer():
    source, build = code("mesh_volume_qem.cpp"), code("build_volume_qem.sh")
    for key in ("volume_relative_limit", "native_cost_and_placement_unchanged", "cost_normalization",
                "target_faces", "block_intersections", "base_source_sha256", "source_sha256"):
        assert key in source and key in build
    assert "world_reward_volume_qem_binary_build" in build
    assert "world_reward_volume_qem_build" in code("run_volume_qem_build.sh")
    assert "source_cpp_sha256" in build and "base_source_cpp_sha256" in build


@pytest.mark.parametrize("name", ["build_volume_qem.sh", "run_volume_qem_build.sh"])
def test_shell_and_embedded_python_syntax(name):
    path = INFRA / name
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    text = path.read_text()
    for marker in ("PYBUILD", "PYBASE", "PYREPORT"):
        if f"<<'{marker}'" in text:
            snippet = text.split(f"<<'{marker}'", 1)[1].split("\n", 1)[1].split(f"\n{marker}", 1)[0]
            ast.parse(snippet)


def tetra():
    return np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]]), np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])


def volume(v, f, origin):
    p = v[f] - origin
    return np.einsum("ij,ij->i", p[:,0], np.cross(p[:,1], p[:,2])).sum() / 6


def test_signed_local_delta_matches_full_volume_and_translation():
    # NumPy mathematical reference only, not execution of native callbacks.
    v, f = tetra()
    for faces in (f, f[:, ::-1]):
        for shift in (np.zeros(3), np.array([4., -3., 7.])):
            vertices = v + shift
            origin = vertices.mean(0)
            proposed = vertices.copy()
            proposed[[0, 1]] = vertices[[0, 1]].mean(0)
            source = volume(vertices, faces, origin)
            delta = volume(proposed, faces, origin) - source
            assert delta == pytest.approx(-source)
            assert volume(proposed, faces, origin) == pytest.approx(source + delta)
            assert abs(delta) / abs(source) > .05


def test_inward_shell_cannot_be_hidden_by_global_volume():
    source = np.array([1., -1e-6])
    proposed = np.array([1., -.9e-6])
    assert abs(proposed.sum()-source.sum())/source.sum() < .05
    assert abs(proposed[1]-source[1])/abs(source[1]) > .05

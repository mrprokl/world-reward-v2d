"""Source/independent arithmetic contracts only; no local native compilation."""
from fractions import Fraction
from pathlib import Path
import struct

import numpy as np
import pytest

from world_reward.mesh_serialization import serialization_preflight

INFRA = Path(__file__).resolve().parents[1] / "infra"


def source():
    return (INFRA / "mesh_serialization_qem.cpp").read_text()


def section(start, end):
    code = source()
    return code[code.index(start):code.index(end, code.index(start))]


def test_authenticated_original_core_is_included_without_main_macro_rewrite():
    code = source()
    assert '#include "wr_volume_core.hpp"' in code
    assert '#define main' not in code
    assert 'void read_obj(' not in code and 'struct Volumes' not in code
    assert code.count('int main(') == 1
    for macro in ("WR_SERIALIZATION_SOURCE_SHA256", "WR_VOLUME_CORE_PREFIX_SHA256",
                  "WR_VOLUME_SOURCE_SHA256", "WR_BASE_SOURCE_SHA256"):
        assert macro in code
    original = (INFRA / "mesh_volume_qem.cpp").read_text()
    marker = "\nint main(int argc, char** argv) {"
    assert original.count(marker) == 1
    prefix = original.split(marker)[0]
    assert "namespace volume_qem" in prefix and "bool simplify(" in prefix
    assert "read_obj(" not in prefix  # parser remains in the immutable nested base include


def test_same_native_qslim_callbacks_and_volume_math_are_delegated():
    code = source()
    assert "igl::qslim_optimal_collapse_edge_callbacks(E,quadrics,v1,v2,cost,pre,post)" in code
    assert "igl::intersection_blocking_collapse_edge_callbacks(pre,post,tree,pre,post)" in code
    assert "igl::per_vertex_point_to_plane_quadrics(V,F,EMAP,EF,EI,quadrics)" in code
    assert "native_pre(V,F,E,EMAP,EF,EI,Q,EQ,C,e)" in code
    assert "volumes.allowed(e,V,F,E,EMAP,EF,EI,C)" in code
    assert "volumes.verify_final(U,G,J,I)" in code
    assert "volume_qem::mapping_json(mapping.get(),V,F,U,G,J,I,volumes,reached,true)" in code
    assert "RELATIVE_VOLUME_LIMIT =" not in code
    assert "cost =" not in code and "p = -b" not in code
    assert "fast_simplification" not in code


def test_pending_transaction_never_mutates_native_arrays_or_committed_indices_in_pre():
    pre = section("  bool allowed(", "  void finish(")
    assert "const Eigen::MatrixXd& V,const Eigen::MatrixXi& F" in pre
    assert "Transaction t;" in pre and "t.approved=true; pending=std::move(t)" in pre
    assert "std::min(E(e,0),E(e,1))" in pre and "std::max(E(e,0),E(e,1))" in pre
    assert "t.buckets" in pre and "t.references" in pre
    assert pre.index("t.references.emplace(t.s,references[t.s])") < pre.index("for(int f:faces)")
    assert "t.references.emplace(after[j],references[after[j]])" in pre
    assert "buckets.erase(" not in pre and "bad.erase(" not in pre
    assert "V.row(" not in pre and "F.row(" not in pre
    assert "references[v]=" not in pre and "positions[v]=" not in pre


def test_rejected_native_link_and_pre_veto_discard_every_pending_change():
    post = section("  void finish(", "bool simplify(")
    assert "if(!collapsed) { pending=Transaction{}; return; }" in post
    assert post.index("if(!collapsed)") < post.index("references[item.first]=item.second")
    assert "t.deleted!=std::set<int>{f1,f2}" in post
    assert "triangle(F,item.first)!=item.second" in post
    assert "position(V,t.s)!=t.p || position(V,t.d)!=t.p" in post
    assert "buckets.erase(item.first)" in post
    assert "bad.erase(pair)" in post and "bad.insert(t.after_bad.begin()" in post
    callbacks = section("    post=[", "    const igl::decimate_stopping_condition_callback")
    assert callbacks.index("native_post(") < callbacks.index("if(collapsed)")
    assert "serialization.finish(V,F,collapsed,f1,f2)" in callbacks


def test_complete_null_row_and_referenced_only_state_not_zero_index_filter():
    code = source()
    assert "(F.row(f).array()==IGL_COLLAPSE_EDGE_NULL).all()" in code
    assert "if(references[v])" in code
    assert "if(t.references[t.d]!=0 || t.references[t.s]<=0)" in code
    assert "live_vertices+=(item.second>0)-(references[item.first]>0)" in code
    assert "if(item.second.empty()) buckets.erase(item.first)" in code
    assert "Invalid live face/reference/position binding" in code
    assert "Unrelated face in native one-ring" in code
    initialization = section("  Serialization(", "  bool safe()")
    assert initialization.index("if(!source_active) throw") < initialization.index("bad_pairs(item.second)")


def test_collision_pair_identity_gate_is_global_not_count_only_or_coincident_position_snap():
    code = source()
    assert "using Pair = std::pair<int,int>" in code
    assert "std::map<Key,Bucket> buckets" in code
    assert "if(pa!=pb) pairs.emplace(*a,*b)" in code
    assert "for(const auto& pair:pairs) if(!bad.count(pair))" in code
    assert "t.after_bad.size()>t.before_bad.size()" in code
    assert "current==buckets.end()?Bucket{}:current->second" in code
    assert "snap(" not in code and "positions[v]=t.p" in code


def test_exact_units_normals_orientation_and_binary_arithmetic_guards():
    code = source()
    assert "boost::multiprecision::cpp_int" in code and "result<<=(exponent-1)" in code
    assert "fraction|0x800000" in code and "return (bits>>31) ? -result : result" in code
    assert "before[0]*after[0]+before[1]*after[1]+before[2]*after[2]>0" in code
    assert "if(!orientation(normal(a),normal(b)))" in code
    assert "FE_TONEAREST" in code and "__FAST_MATH__" in code
    assert "preserved!=smallest" in code and "denorm_min()" in code
    assert "volatile float stored" in code and "volatile double product" in code
    assert "rounded < -0x1p63 || rounded >= 0x1p63" in code
    assert "fraction==.5 && std::fmod(lower,2.)!=0." in code
    assert "std::round(" not in code and "epsilon()" not in code


def test_budget_is_conjunction_and_initial_safe_identity_needs_no_collapse():
    code = source()
    assert "safe() && live_faces<=TARGET && live_vertices<=TARGET" in code
    simplify = section("bool simplify(", "void summary(")
    assert simplify.index("if(serialization.budget_safe())") < simplify.index("igl::edge_flaps")
    assert "igl::remove_unreferenced(V,F,U,G,old_to_new,I)" in simplify
    assert "J=Eigen::VectorXi::LinSpaced" in simplify
    assert "return serialization.budget_safe();" in simplify
    assert "igl::max_faces_stopping_condition(" not in simplify
    assert "igl::decimate(V,F,cost,stop,pre,post,U,G,J,I)" in simplify
    assert "Native queue exhausted before budget AND serialization safety" in code
    assert "Final serialization recomputation differs" in code


def test_cli_and_fail_before_geometry_output_are_preparatory_only():
    code = source()
    assert '"--preflight"' in code and '"--build-info"' in code and '"--triangle-predicates"' in code
    assert "argc==20" in code and "eighteen finite coordinates" in code
    assert "before_float32_exactly_active" in code and "exact_normal_dot_positive" in code
    main = code[code.index("int main("):]
    assert main.index("volumes.verify_final(U,G,J,I)") < main.index("auto out=exclusive_file")
    assert main.index("!independent.budget_safe()") < main.index("auto out=exclusive_file")
    assert "vector_json(stream,J,true)" in (INFRA / "mesh_volume_qem.cpp").read_text()
    for field in ("source_float32_exactly_active", "nonexact_collision_pairs", "nonexact_key_collision_pairs",
                  "float32_collapsed_distinct_position_pairs", "active_vertices", "active_faces",
                  "keys_in_int64_range", "serialization_safe", "adopted"):
        assert field in code
    assert '\\"adopted\\":false' in code and '\\"geometry_snapped\\":false' in code


def units(value):
    """Independent integer decoding of the documented C++ dyadic units."""
    bits = struct.unpack("<I", np.float32(value).tobytes())[0]
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    assert exponent != 255
    result = ((fraction | 0x800000) << (exponent - 1)) if exponent else fraction
    return -result if bits >> 31 else result


@pytest.mark.parametrize("value", [0., -0., 1., -3., 2.**-126, 2.**-149, np.finfo(np.float32).max])
def test_integer_units_equal_exact_stored_float32_fraction(value):
    q = np.float32(value)
    assert Fraction(units(q), 2**149) == Fraction.from_float(float(q))


def integer_normal(triangle):
    p = [[units(x) for x in row] for row in np.asarray(triangle, np.float32)]
    a, b = [[p[j][i] - p[0][i] for i in range(3)] for j in (1, 2)]
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def test_exact_normal_dot_rejects_flip_and_collinearity_without_area_tolerance():
    before = np.array([[0., 0., 0.], [2.**-149, 0., 0.], [0., 2.**-149, 0.]])
    n = integer_normal(before)
    assert sum(x*x for x in n) > 0
    assert sum(a*b for a,b in zip(n, integer_normal(before[[0, 2, 1]]))) < 0
    assert integer_normal([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]) == (0, 0, 0)


def pairs(vertices, faces):
    used = sorted(set(np.asarray(faces).ravel().tolist()))
    stored = np.asarray(vertices).astype(np.float32).astype(np.float64)
    keys = np.round(stored[used] * 1e8).astype(np.int64)
    return {(a,b) for i,a in enumerate(used) for j,b in enumerate(used) if i < j
            and np.array_equal(keys[i],keys[j]) and not np.array_equal(vertices[a],vertices[b])}


def test_pair_count_alone_cannot_exchange_old_cross_shell_collision_for_new_pair():
    v = np.array([[0., 0., 0.], [1e-9, 0., 0.], [1., 0., 0.], [1.+1e-9, 0., 0.]])
    # This tests pair algebra only, not a mesh or native legal collapse.
    before = {(0,1)}
    after = {(2,3)}
    assert len(after) <= len(before) and not after.issubset(before)
    assert set().issubset(before)  # disappearance is permitted
    assert pairs(v, [[0,1,2],[1,2,3]]) == {(0,1),(2,3)}


def test_first_representative_signed_zero_and_orphan_agree_with_preflight_contract():
    v = np.array([[1e300,0.,0.], [-0.,0.,0.], [0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    f = np.array([[2,3,4],[1,4,3]])
    before = v.tobytes(), f.tobytes()
    report = serialization_preflight(v,f)
    assert report["ignored_orphan_vertices"] == 1 and report["position_weld_admissible"]
    assert report["source_exact_seam_duplicates"] == 1
    assert not report["serialized_triangles_byte_preserved_by_weld"]
    assert pairs(v[1:],f-1) == set()
    assert (v.tobytes(),f.tobytes()) == before


def test_128_seed8401_preflight_fixtures_remain_scalar_only_and_unmodified():
    rng = np.random.default_rng(8401)
    tetra = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    for index in range(128):
        v = tetra * 2.**int(rng.integers(-12,13)) + rng.integers(-8,9,3)
        if index % 4 == 0:
            v = np.r_[v, v[[0]]]
            f = np.r_[faces, [[4,2,1]]]
        elif index % 4 == 1:
            v = np.r_[v, v[[0]] + [[1e-9,0.,0.]]]
            f = np.r_[faces, [[4,2,1]]]
        else:
            f = faces.copy()
        before = v.tobytes(),f.tobytes()
        report = serialization_preflight(v,f)
        assert report["float32_triangles_exactly_active"]
        assert all(type(x) in (str,int,bool) for x in report.values())
        assert (v.tobytes(),f.tobytes()) == before

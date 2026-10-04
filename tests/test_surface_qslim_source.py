"""Prepared source/API and manufactured topology checks; NO native compilation/QEM."""
from collections import Counter
from fractions import Fraction
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from world_reward.surface_identity import prepare_surface_identity

CPP = Path(__file__).resolve().parents[1] / 'infra' / 'surface_qslim.cpp'


def source():
    return CPP.read_text()


def block(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def patch(n, a, b, translate=0):
    """Literal authored definition from protocol; no existing shape is repaired."""
    cells = [(i, j) for i in range(n) for j in range(n)
             if not (a <= i < b and a <= j < b)]
    indices = sorted({p for i, j in cells for p in
                      ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))})
    lookup = {p: k for k, p in enumerate(indices)}
    vertices = []
    for i, j in indices:
        x, y = (i - n / 2) / 32, (j - n / 2) / 32
        vertices.append((x + translate, y, (x*x + y*y) / 16))
    faces = [(lookup[p], lookup[q], lookup[r]) for i, j in cells for p, q, r in (
        ((i, j), (i + 1, j), (i + 1, j + 1)),
        ((i, j), (i + 1, j + 1), (i, j + 1)))]
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)


def boundary(faces):
    counts = Counter(tuple(sorted((int(row[k]), int(row[(k + 1) % 3]))))
                     for row in faces for k in range(3))
    return {e for e, count in counts.items() if count == 1}


@pytest.mark.parametrize('n,a,b,vcount,fcount,bcount', [
    (64, 24, 40, 4000, 7680, 320), (48, 20, 28, 2352, 4480, 224)])
def test_frozen_curved_source_definition_counts_activity_boundary(n, a, b, vcount, fcount, bcount):
    v, f = patch(n, a, b)
    before = (v.tobytes(), f.tobytes())
    assert v.shape == (vcount, 3) and f.shape == (fcount, 3)
    assert len(boundary(f)) == bcount
    assert np.unique(f).size == vcount
    assert np.array_equal(v.astype(np.float64).astype(np.float32), v)
    xy = v[f, :2].astype(np.float64)
    cross = ((xy[:, 1, 0]-xy[:, 0, 0])*(xy[:, 2, 1]-xy[:, 0, 1]) -
             (xy[:, 1, 1]-xy[:, 0, 1])*(xy[:, 2, 0]-xy[:, 0, 0]))
    assert np.all(cross == 1 / 1024)  # Exact dyadic positive triangle predicate.
    assert before == (v.tobytes(), f.tobytes())


def test_frozen_disjoint_family_and_small_native_identity_domains():
    a, af = patch(48, 20, 28)
    b, bf = patch(48, 20, 28, translate=4)
    v, f = np.vstack((a, b)), np.vstack((af, bf + len(a)))
    assert v.shape == (4704, 3) and f.shape == (8960, 3)
    assert len(boundary(f)) == 448
    assert len({int(x) for e in boundary(f) for x in e}) == 448
    # A local tiny equivalent verifies the open-manifold component domain.
    av, aff = patch(4, 1, 3)
    bv, bff = patch(4, 1, 3, translate=4)
    result = prepare_surface_identity(np.vstack((av, bv)), np.vstack((aff, bff + len(av))))
    assert len(result.component_keys) == 2 and len(result.boundary_loops) == 4
    assert not result.report()['solid_geometry_certified']


def test_explicit_boundary_excess_is_inapplicable_not_a_new_identity_rejection():
    s = np.arange(512) / 256
    rim = np.concatenate((np.column_stack((-1+s, -np.ones(512))),
                          np.column_stack((np.ones(512), -1+s)),
                          np.column_stack((1-s, np.ones(512))),
                          np.column_stack((-np.ones(512), 1-s))))
    v = np.vstack((np.column_stack((rim, np.zeros(2048))),
                   np.column_stack((rim, np.ones(2048)))))
    f = np.array([(i, (i+1) % 2048, (i+1) % 2048 + 2048) for i in range(2048)] +
                 [(i, (i+1) % 2048 + 2048, i + 2048) for i in range(2048)], np.int64)
    assert v.shape == (4096, 3) and f.shape == (4096, 3)
    assert len({x for e in boundary(f) for x in e}) == 4096
    code = source()
    assert 'if(source.boundary_vertices>=TARGET) throw Inapplicable' in code
    assert 'return 4;' in code


def test_exact_native_sources_and_build_identity_are_explicit_not_qualified():
    code = source()
    assert 'WR_SURFACE_QSLIM_SOURCE_SHA256' in code
    assert '40e7900ccbd767f1f360e0eb10f0f1a6432e0993' in code
    assert '3147391d946bb4b6c68edd901f2add6ac1f31f8c' in code
    for item in ('surface_qslim_build_info_v1', 'surface-qslim-mapping-v1',
                 'prepared_only\\\":true', 'adoption\\\":false', '--preflight'):
        assert item in code
    assert not re.search(r'#include.*(?:volume|condition|certified|cgal)', code, re.I)
    assert '#ifdef __FAST_MATH__' in code and 'FE_TONEAREST' in code


def test_native_qslim_math_and_real_upstream_intersection_callback_not_custom_cost():
    code = source()
    for call in ('igl::connect_boundary_to_infinity', 'igl::edge_flaps',
                 'igl::per_vertex_point_to_plane_quadrics',
                 'igl::qslim_optimal_collapse_edge_callbacks',
                 'igl::intersection_blocking_collapse_edge_callbacks', 'igl::decimate('):
        assert call in code
    assert code.index('tree->init(V,F)') < code.index('igl::connect_boundary_to_infinity')
    cost = block(code, 'cost=[&]', 'pre=[&]')
    assert cost.index('std::numeric_limits<double>::infinity()') < cost.index('native_cost(')
    assert 'state.eligible' in cost and '++' not in cost  # Native initial-cost parallelism.
    assert 'initial_embedding_certified\\\":false' in code


def test_failed_callback_never_reads_undefined_face_ids_or_updates_ledger():
    code = source()
    post = block(code, 'post=[&]', 'igl::decimate_stopping_condition_callback')
    failed = block(post, 'if(!collapsed)', 'intersection_post(')
    assert 'f1' not in failed and 'f2' not in failed
    assert 'return;' in failed and 'state.pending=Transaction{}' in failed
    assert 'state.commit' not in failed and 'ledger.push_back' not in failed
    assert post.index('if(!collapsed)') < post.index('intersection_post(') < post.index('state.commit(')
    assert source().count('ledger.push_back') == 1


def test_committed_quotient_is_not_survivor_coordinate_equality():
    code = source()
    assert 'quotient.parent[pending.d]=pending.s' in code
    assert 'state.quotient.find(v)' in code
    assert 'Oriented face birth/quotient replay mismatch' in code
    assert 'I[retained[v]]' in code
    # Manufactured replay: removal of 3 into 0 changes source vertices 0/3 positions.
    original = np.array([[0, 1, 4], [3, 4, 2]], np.int64)
    quotient = np.array([0, 1, 2, 0, 3])
    candidate = quotient[original]
    assert candidate.tolist() == [[0, 1, 3], [0, 3, 2]]
    births = np.array([0, 1, 2, 4])
    assert not np.array_equal(births, quotient[:4])


def test_strip_only_synthetic_faces_and_keep_real_face_containing_zero():
    code = source()
    final = block(code, 'Result finalize(', 'Result simplify(')
    assert 'if(J[f]<originalF.rows())keep.push_back(f)' in final
    assert 'if(J[f]>=originalF.rows())' not in final
    J = np.array([0, 5, 1, 6])
    F = np.array([[0, 1, 2], [0, 2, 9], [0, 2, 3], [9, 2, 3]])
    assert F[J < 2].tolist() == [[0, 1, 2], [0, 2, 3]]
    assert 'Preserve original unused vertices verbatim' in final
    assert 'Fixed boundary vertex bytes changed' in final
    assert 'Original unused vertex bytes changed' in final


def test_transactions_no_inplace_native_mutation_prior_to_commit():
    code = source()
    pre = block(code, 'bool prepare(', 'void commit(')
    assert 'const VMat& V,const FMat& F' in pre
    assert 'igl::edge_collapse_is_valid' in pre
    assert 'same_orientation(normal(V,before,f32),normal(V,after,f32' in pre
    assert 'for(bool f32:{false,true})' in pre
    assert 'component_faces[source.vertex_component[s]]<=2' in pre
    assert 'pending=std::move(next)' in pre
    assert 'ledger.push_back' not in pre and 'quotient.parent' not in pre
    assert not re.search(r'(?<!next\.)\b[ VF]\([^\n]+\)\s*=', pre)


def test_exact_dyadic_predicate_covers_collapse_reverse_and_roundoff_without_epsilon():
    def normal(points):
        p = [[Fraction(float(x)) for x in row] for row in points]
        a, b = ([p[1][k]-p[0][k] for k in range(3)], [p[2][k]-p[0][k] for k in range(3)])
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    triangle = np.array([[1, 0, 0], [1 + 2**-25, 0, 0], [1, 2**-25, 0]], np.float64)
    assert normal(triangle)[2] > 0 and normal(triangle.astype(np.float32))[2] == 0
    native = normal(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]))
    reverse = normal(np.array([[0., 0, 0], [0, 1, 0], [1, 0, 0]]))
    assert sum(a*b for a, b in zip(native, reverse)) < 0
    code = source()
    assert 'boost::multiprecision::cpp_int' in code and 'std::memcpy' in code
    assert 'a[0]*b[0]+a[1]*b[1]+a[2]*b[2]>0' in code
    assert 'epsilon' not in code.lower()


def test_both_real_budgets_and_full_component_boundary_checks_before_fresh_export():
    code = source()
    assert 'state.real_faces<=TARGET && state.real_vertices<=TARGET' in code
    for predicate in ('final.components==source.components', 'actual_boundary==source.boundary_edges',
                      'final.euler[c]==source.euler[sc]', 'reference_count[pending.d]==0'):
        assert predicate in code
    assert 'O_EXCL|O_NOFOLLOW' in code and 'fsync' in code
    assert code.index('Result result=simplify') < code.index('write(argv[2]')
    assert code.index('absent(argv[2])') < code.index('read_obj(argv[')
    assert 'No source component/vertex cleanup' not in code  # Comments do not serve as proof.


def test_source_file_is_small_standalone_and_original_sources_untouched():
    assert CPP.stat().st_size < 32_000
    assert len(source().splitlines()) < 400
    assert hashlib.sha256(CPP.read_bytes()).hexdigest()
    assert '#include "' not in source()  # No inherited closed/volume main or generated solver.

"""Independent tiny IEEE triangle diagnostics; no model/data/media or NumPy."""
import ast
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from world_reward import exact_triangle_witness as exact


A = [[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]]
B = [[0., -.5, -1.], [0., -.5, 1.], [0., .5, 0.]]


def rational(value):
    assert set(value) == {"numerator", "denominator"}
    assert type(value["numerator"]) is str and type(value["denominator"]) is str
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    assert str(result.numerator) == value["numerator"]
    assert str(result.denominator) == value["denominator"]
    return result


def require_witness(a, b, report):
    assert report["strict_transverse_witness"] is True
    assert report["status"] == "strict_transverse_witness"
    assert report["reason"] == "strict_transverse_interior_intersection"
    witness = report["witness"]
    point = list(map(rational, witness["point"]))
    for triangle, key in ((a, "triangle_a_barycentric"), (b, "triangle_b_barycentric")):
        weights = list(map(rational, witness[key]))
        assert all(weight > 0 for weight in weights) and sum(weights) == 1
        for coordinate in range(3):
            value = sum(weight * Fraction.from_float(float(vertex[coordinate]))
                for weight, vertex in zip(weights, triangle))
            assert value == point[coordinate]
    assert all(rational(value) == 0 for value in witness["plane_residuals"])
    assert rational(witness["normal_cross_squared"]) > 0
    assert rational(witness["interval_overlap_projection"]) > 0
    assert report["full_separation_proven"] is report["adjacency_evaluated"] is False
    assert report["replacement_certificate"] is report["touching_certified"] is report["force_closure_verified"] is False
    assert report["arithmetic_operations"] < 10000 and report["peak_fraction_bits"] <= 4096
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_strict_transverse_exact_interior_point_and_rational_json():
    report = exact.exact_triangle_witness(A, B)
    require_witness(A, B, report)
    assert list(map(rational, report["witness"]["point"])) == [0, 0, 0]


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
def test_orientation_vertex_order_and_triangle_swap(permutation):
    a, b = [A[i] for i in permutation], [B[i] for i in permutation[::-1]]
    require_witness(a, b, exact.exact_triangle_witness(a, b))
    require_witness(b, a, exact.exact_triangle_witness(b, a))


@pytest.mark.parametrize("power", [-20, -5, 0, 10, 40])
def test_exact_scale_translation_and_axis_permutation(power):
    scale = 2. ** power
    shift = [3. * scale, -2. * scale, scale]
    transform = lambda triangle: [[row[2]*scale+shift[0], row[0]*scale+shift[1], row[1]*scale+shift[2]] for row in triangle]
    a, b = transform(A), transform(B)
    require_witness(a, b, exact.exact_triangle_witness(a, b))


def test_independent_random_integer_triangles_share_strict_centroid():
    rng = random.Random(91824)
    checked = 0
    while checked < 32:
        first = [[rng.randrange(-20, 21) for _ in range(3)] for _ in range(2)]
        last = [[rng.randrange(-20, 21) for _ in range(3)] for _ in range(2)]
        a = first + [[-first[0][i]-first[1][i] for i in range(3)]]
        b = last + [[-last[0][i]-last[1][i] for i in range(3)]]
        try: report = exact.exact_triangle_witness(a, b)
        except ValueError: continue  # Exactly degenerate draw, not a failed candidate retry.
        if report["reason"] == "coplanar_planes": continue
        require_witness(a, b, report)
        checked += 1


@pytest.mark.parametrize("slope", [2.**-20, 2.**-60, 2.**-300])
def test_quasiparallel_crossing_below_any_float_tolerance_is_exact(slope):
    b = [[x, y, slope*x] for x, y, _ in A]
    require_witness(A, b, exact.exact_triangle_witness(A, b))


@pytest.mark.parametrize("b,reason", [
    (A, "coplanar_planes"),
    ([[x, y, 1.] for x, y, _ in A], "parallel_distinct_planes"),
    ([[x, y, 2.**-300] for x, y, _ in A], "parallel_distinct_planes"),
    ([[0., 1., -1.], [0., 1., 1.], [0., 3., 0.]], "plane_cut_intervals_do_not_overlap_strictly"),
    ([[0., 2., -1.], [0., 2., 1.], [0., 4., 0.]], "plane_cut_intervals_do_not_overlap_strictly"),
    ([[0., -.5, 0.], [0., .5, 0.], [0., 0., 1.]], "no_strict_two_sided_plane_cuts"),
])
def test_no_strict_witness_is_not_separation_contact_or_full_certificate(b, reason):
    report = exact.exact_triangle_witness(A, b)
    assert report["status"] == "no_strict_witness" and report["reason"] == reason
    assert report["strict_transverse_witness"] is False and report["witness"] is None
    assert report["full_separation_proven"] is report["touching_certified"] is False


@pytest.mark.parametrize("a,b", [
    ([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], [[0.,0.,0.],[-1.,0.,0.],[0.,0.,1.]]),
    ([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], [[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]]),
    ([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], [[0.,0.,0.],[1.,0.,0.],[.5,-1.,0.]]),
])
def test_shared_vertex_or_edge_never_invents_strict_interior_witness(a, b):
    result = exact.exact_triangle_witness(a, b)
    assert result["strict_transverse_witness"] is False and result["adjacency_evaluated"] is False


def test_shared_vertex_can_still_have_actual_transverse_interior_intersection():
    a = [[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]
    b = [[0.,0.,0.],[1.,.5,-1.],[1.,.5,1.]]
    require_witness(a, b, exact.exact_triangle_witness(a, b))


def test_ieee_signzero_is_hashbound_but_not_changed_geometry():
    first = exact.exact_triangle_witness(A, B)
    a = [row.copy() for row in A]; a[0][2] = -0.
    second = exact.exact_triangle_witness(a, B)
    assert first["coordinate_sha256"][0] != second["coordinate_sha256"][0]
    assert first["witness"] == second["witness"]


@pytest.mark.parametrize("bad", [None, "triangle", [[0.,0.]]*3, [[0.,0.,0.]]*2,
    [[True,0.,0.],[1.,0.,0.],[0.,1.,0.]], [[float("nan"),0.,0.],[1.,0.,0.],[0.,1.,0.]],
    [[float("inf"),0.,0.],[1.,0.,0.],[0.,1.,0.]], [[2**53+1,0.,0.],[1.,0.,0.],[0.,1.,0.]],
    [[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]])
def test_invalid_or_lossy_ieee_inputs_raise_no_witness(bad):
    with pytest.raises(ValueError): exact.exact_triangle_witness(bad, B)


@pytest.mark.parametrize("kwargs", [{"budget_seconds":0}, {"budget_seconds":6}, {"budget_seconds":True},
    {"budget_seconds":float("nan")}, {"max_fraction_bits":0}, {"max_fraction_bits":4097},
    {"max_fraction_bits":True}, {"max_operations":0}, {"max_operations":10001}])
def test_budget_increases_or_invalid_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError): exact.exact_triangle_witness(A, B, **kwargs)


def test_exact_arithmetic_bit_and_operation_caps_raise_failclosed():
    with pytest.raises(exact.WitnessBudgetError, match="bit budget"):
        exact.exact_triangle_witness(A, B, max_fraction_bits=1)
    with pytest.raises(exact.WitnessBudgetError, match="operation budget"):
        exact.exact_triangle_witness(A, B, max_operations=1)
    tiny = float.fromhex("0x0.0000000000001p-1022")
    b = [[x, y, tiny*x] for x, y, _ in A]
    with pytest.raises(exact.WitnessBudgetError, match="bit budget"):
        exact.exact_triangle_witness(A, b, max_fraction_bits=2048)


def test_deadline_checked_during_arithmetic_and_before_witness_return(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(exact.time, "monotonic", lambda: next(ticks)*.01)
    with pytest.raises(exact.WitnessBudgetError, match="deadline"):
        exact.exact_triangle_witness(A, B, budget_seconds=.03)


def test_expired_deadline_before_positive_finish_never_returns_witness(monkeypatch):
    calls = 0
    original = exact._rational
    def expired_after_serialization(value, arithmetic):
        nonlocal calls
        result = original(value, arithmetic)
        calls += 1
        if calls == 13: arithmetic.deadline = float("-inf")
        return result
    monkeypatch.setattr(exact, "_rational", expired_after_serialization)
    with pytest.raises(exact.WitnessBudgetError, match="deadline"):
        exact.exact_triangle_witness(A, B)


def test_module_import_and_real_probe_are_stdlib_only_in_fresh_process():
    source = Path(exact.__file__).resolve()
    command = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('w',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "r=m.exact_triangle_witness([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]],"
        "[[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]]); assert r['strict_transverse_witness']; "
        "assert not any(n.split('.')[0] in {'numpy','scipy','torch','joblib','trimesh'} for n in sys.modules)")
    result = subprocess.run(["rtk", "proxy", sys.executable, "-I", "-B", "-c", command, str(source)],
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    tree = ast.parse(source.read_text())
    names = [node.module if isinstance(node, ast.ImportFrom) else item.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for item in (node.names if isinstance(node, ast.Import) else [None])]
    assert set(names) <= {"__future__", "fractions", "hashlib", "math", "numbers", "struct", "time"}


def test_indexable_array_protocol_does_not_require_numpy():
    class Array:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]
    require_witness(A, B, exact.exact_triangle_witness(Array(A), Array(B)))

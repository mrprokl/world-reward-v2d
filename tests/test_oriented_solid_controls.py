"""Tiny geometric logic plus deterministic authored arrays; no native QEM/CGAL."""
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
import exact_mesh_geometry
SPEC = importlib.util.spec_from_file_location("wr_oriented_solid_controls", ROOT / "infra/oriented_solid_controls.py")
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


def topology(v, f):
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert np.all(counts == 2)
    assert np.all(np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)) == 0)
    assert len(v) - len(edges) + len(f) == 2
    for vertex in range(len(v)):
        link = {}
        for row in f[np.any(f == vertex, axis=1)]:
            a, b = row[row != vertex]
            link.setdefault(int(a), []).append(int(b)); link.setdefault(int(b), []).append(int(a))
        assert all(len(neighbors) == 2 for neighbors in link.values())
        visited, pending = set(), [next(iter(link))]
        while pending:
            current = pending.pop()
            if current not in visited:
                visited.add(current); pending.extend(link[current])
        assert len(visited) == len(link)
    triangle = v[f] - v[0]
    return float(np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])).sum() / 6)


@pytest.mark.parametrize("sign", [1, -1])
def test_tiny_uv_surface_unique_seam_poles_closed_vertex_links_and_euler(sign):
    v, f = controls.ellipsoid(8, 6, (1., .875, .75), (0., 0., 0.), sign)
    assert v.shape == (42, 3) and f.shape == (80, 3)
    assert len(np.unique(v, axis=0)) == len(v)
    assert np.count_nonzero(np.all(v[:, :2] == 0, axis=1)) == 2
    assert np.sign(topology(v, f)) == sign
    assert v.dtype == np.float64 and f.dtype == np.int64
    assert not v.flags.writeable and not f.flags.writeable
    if sign == 1:
        actual = exact_mesh_geometry.exact_mesh_topology(v, f)
        assert actual["components"][0]["euler"] == 2
        assert actual["components"][0]["volume_sign"] == 1


@pytest.mark.parametrize("meridians,latitudes", [(True, 4), (6, 4), (8, 5), (4, 4), (8, 2)])
def test_bad_sampling_is_not_rescued(meridians, latitudes):
    with pytest.raises(ValueError):
        controls.ellipsoid(meridians, latitudes)


def test_fresh_forest_specs_have_analytic_containment_and_disjoint_bounds_not_certificates():
    outer = np.array(controls.OUTER_AXES)
    for axes, center in zip(controls.MULTICAVITY_AXES, controls.MULTICAVITY_CENTERS):
        # Triangle points lie in convex axis-aligned box, itself inside smooth
        # outer ellipsoid by this conservative quadratic corner bound.
        assert np.sum(((np.abs(center) + axes) / outer) ** 2) < 1
    centers = controls.MULTICAVITY_CENTERS
    assert centers[0][0] + controls.MULTICAVITY_AXES[0][0] < centers[1][0] - controls.MULTICAVITY_AXES[1][0]
    assert np.all(np.array(controls.VOID_AXES) < outer)
    assert np.all(np.array(controls.ISLAND_AXES) < np.array(controls.VOID_AXES))
    assert controls.INDEPENDENT_CENTER[0] - controls.ROOT_AXES[0] > outer[0]
    # Mesh UV chord deficit is far smaller than these deliberately thick
    # component clearances; still leave stored-triangle certification native.
    assert np.all(np.array(controls.VOID_AXES) / outer == .5)
    # This test makes no CGAL/predicate claim for represented triangle surfaces.


def test_all_overbudget_sources_have_fixed_ranges_original_keys_and_exact_similitudes():
    fixtures = controls.fixtures()
    assert tuple(n for n, _, _ in fixtures) == controls.FIXTURE_NAMES
    expected = ((2950, 5888, (1, -1, -1), (-1, 0, 0)),
                (3432, 6848, (1, -1, 1, 1), (-1, 0, 1, -1)))
    for family, (nv, nf, signs, parents) in zip(range(2), expected):
        _, (base_v, base_f), base = fixtures[family * 2]
        assert base_v.shape == (nv, 3) and base_f.shape == (nf, 3) and nf > 4096
        np.testing.assert_array_equal(base["expected_signs"], signs)
        np.testing.assert_array_equal(base["expected_parents"], parents)
        assert len(base["component_keys"]) == len(signs)
        assert len(np.unique(base_v, axis=0)) == len(base_v)
        _, (small_v, small_f), small = fixtures[family * 2 + 1]
        np.testing.assert_array_equal(small_v, base_v * 2.**-16 + np.array(controls.TRANSLATIONS[1]))
        np.testing.assert_array_equal(small_f, base_f)
        assert small["component_keys"] == base["component_keys"]
        for mesh, meta in (((base_v, base_f), base), ((small_v, small_f), small)):
            assert isinstance(meta, MappingProxyType) and not meta["geometry_certified"]
            assert meta["metric_scale"] == .375
            assert meta["source_array_sha256"] == controls.array_fingerprints(mesh)
            for i, (v0, v1, f0, f1) in enumerate(meta["component_ranges"]):
                np.testing.assert_array_equal(meta["face_components"][f0:f1], i)
                assert np.all((mesh[1][f0:f1] >= v0) & (mesh[1][f0:f1] < v1))
            for array in (*mesh, meta["face_components"], meta["expected_parents"], meta["expected_signs"]):
                with pytest.raises(ValueError):
                    array.flags.writeable = True


def test_fixed_sources_repeated_generation_has_deterministic_typed_fingerprints():
    first, second = controls.fixtures(), controls.fixtures()
    for a, b in zip(first, second):
        assert a[0] == b[0] and a[2]["source_array_sha256"] == b[2]["source_array_sha256"]
        assert a[1][0].tobytes() == b[1][0].tobytes() and a[1][1].tobytes() == b[1][1].tobytes()
    source = (ROOT / "infra/oriented_solid_controls.py").read_text()
    assert "import trimesh" not in source and "subprocess" not in source
    assert "mesh_conditioned_geometry" not in source


def test_frozen_cohort_rejects_v1_midpoint_chart_before_any_qem():
    """Regression for an observed scientific rejection, not solver qualification."""
    from world_reward.mesh_conditioning import prepare_conditioning
    outcomes = []
    for name, mesh, _ in controls.fixtures():
        chart = prepare_conditioning(*mesh)
        outcomes.append(chart.diagnostics["roundtrip_numerically_exact"])
        if name == "newforest_island_independentroot_identity":
            decoded = chart.decode(chart.encode(mesh[0]))
            assert np.count_nonzero(decoded != mesh[0]) == 1477
            assert float(np.max(np.abs(decoded - mesh[0]))) == 2.**-53
            np.testing.assert_array_equal(chart.origin, [.625, 0., 0.])
            assert chart.scale == 4.
    assert outcomes == [True, True, False, True]
    # Compiler predeclares an all-source gate. Never drop the rejected source,
    # change its generation, loosen exactness or invoke QEM to rescue this cohort.
    assert not all(outcomes)

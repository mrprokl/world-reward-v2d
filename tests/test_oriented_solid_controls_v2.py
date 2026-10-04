"""Tiny deterministic source/metadata tests, not native geometry qualification."""
from decimal import getcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "infra/oriented_solid_controls_v2.py"
SPEC = importlib.util.spec_from_file_location("wr_solid_controls_v2", PATH)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


@pytest.fixture(scope="module")
def cohort():
    return controls.fixtures()


def signed_volume_and_edges(v, f):
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert np.all(counts == 2)
    assert np.all(np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)) == 0)
    assert len(v)-len(edges)+len(f) == 2
    t = v[f]-v.mean(axis=0)
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()/6)


@pytest.mark.parametrize("sign", (1, -1))
def test_tiny_harmonic_surface_closed_unique_poles_links_and_orientation(sign):
    v, f = controls.radial_surface(8, 6, (1.125, .9375, .8125), (0., 0., 0.), (1/32, 1/64), sign)
    assert v.shape == (42, 3) and f.shape == (80, 3)
    assert len(np.unique(v, axis=0)) == len(v)
    assert np.count_nonzero(np.all(v[:, :2] == 0, axis=1)) == 2
    assert np.sign(signed_volume_and_edges(v, f)) == sign
    for vertex in range(len(v)):
        link = {}
        for row in f[np.any(f == vertex, axis=1)]:
            a, b = row[row != vertex]
            link.setdefault(int(a), []).append(int(b)); link.setdefault(int(b), []).append(int(a))
        assert all(len(neighbors) == 2 for neighbors in link.values())
        seen, pending = set(), [next(iter(link))]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current); pending.extend(link[current])
        assert len(seen) == len(link)


@pytest.mark.parametrize("args", [
    (True, 6, (1, 1, 1), (0, 0, 0), (0, 0), 1),
    (6, 6, (1, 1, 1), (0, 0, 0), (0, 0), 1),
    (8, 5, (1, 1, 1), (0, 0, 0), (0, 0), 1),
    (8, 6, (1, 0, 1), (0, 0, 0), (0, 0), 1),
    (8, 6, (1, 1, 1), (np.nan, 0, 0), (0, 0), 1),
    (8, 6, (1, 1, 1), (0, 0, 0), (1, 0), 1),
    (8, 6, (1, 1, 1), (0, 0, 0), (0, 0), True),
])
def test_malformed_parameters_fail_without_resampling(args):
    with pytest.raises(ValueError):
        controls.radial_surface(*args)


def test_specification_and_all_original_array_hashes_frozen(cohort):
    assert controls.SPEC_SHA256 == "b166277680c12b2a493474670cdddedf3510240ff557b7d4b40019e956684fbd"
    assert tuple(row[0] for row in cohort) == controls.FIXTURE_NAMES
    assert set(controls.SOURCE_ARRAY_SHA256) == set(controls.FIXTURE_NAMES)
    for name, mesh, metadata in cohort:
        assert metadata["source_array_sha256"] == controls.SOURCE_ARRAY_SHA256[name]
        assert controls.array_fingerprints(mesh) == controls.SOURCE_ARRAY_SHA256[name]
        assert mesh[0].dtype == np.float64 and mesh[1].dtype == np.int64
        assert metadata["specification_sha256"] == controls.SPEC_SHA256
    with pytest.raises(TypeError):
        controls.SOURCE_ARRAY_SHA256[controls.FIXTURE_NAMES[0]] = ("0"*64, "0"*64)


def test_full_overbudget_ranges_forest_expectations_and_orphans_not_dropped(cohort):
    for i, (_, (v, f), meta) in enumerate(cohort):
        nv, nf = (2358, 4704) if i < 2 else (2724, 5432)
        assert v.shape == (nv, 3) and f.shape == (nf, 3) and nf > 4096
        np.testing.assert_array_equal(np.unique(f), np.arange(nv))
        assert len(np.unique(v, axis=0)) == nv
        np.testing.assert_array_equal(meta["expected_parents"], controls.PARENTS[i//2])
        assert len(meta["component_keys"]) == len(meta["expected_parents"])
        for component, (v0, v1, f0, f1) in enumerate(meta["component_ranges"]):
            assert np.all((f[f0:f1] >= v0) & (f[f0:f1] < v1))
            np.testing.assert_array_equal(meta["face_components"][f0:f1], component)
            assert np.sign(signed_volume_and_edges(v[v0:v1], f[f0:f1]-v0)) == meta["expected_signs"][component]


def test_exact_declared_similitudes_and_no_geometry_quality_claim(cohort):
    for family in range(2):
        _, base, meta = cohort[2*family]
        _, transformed, other = cohort[2*family+1]
        np.testing.assert_array_equal(transformed[0], base[0]*2.**-11 + np.array((8., -12., 16.))*2.**-11)
        np.testing.assert_array_equal(transformed[1], base[1])
        assert other["similarity_translation"] == (2.**-8, -3*2.**-9, 2.**-7)
        assert meta["component_keys"] == other["component_keys"]
        for metadata in (meta, other):
            assert metadata["metric_scale"] == .375
            assert metadata["geometry_certified"] is metadata["chart_qualified"] is metadata["adopted"] is False
            assert metadata["native_calls"] == 0 and metadata["historical_failure_replayed"] is False


def test_zero_and_sterbenz_domains_independent_exact_fraction_conditions(cohort):
    for i, (_, (v, _), meta) in enumerate(cohort):
        domains = (False,)*3 if i % 2 == 0 else (True,)*3
        assert meta["midpoint_sterbenz_conditions"] == domains
        assert meta["intended_chart_axes"] == (("zero",)*3 if i % 2 == 0 else ("sterbenz",)*3)
        origin = v.min(axis=0)+(v.max(axis=0)-v.min(axis=0))*.5
        for axis in range(3):
            exact_o = Fraction(abs(float(origin[axis])))
            valid = bool(exact_o and all(np.signbit(x) == np.signbit(origin[axis]) and
                exact_o/2 <= Fraction(abs(float(x))) <= 2*exact_o for x in v[:, axis]))
            assert valid == domains[axis]
    # Only conditions are tested: no new chart or simplification is executed.


def test_analytic_thick_clearances_are_expectations_not_stored_embedding_certificate():
    for specs in controls.FAMILY_SPECS:
        outer = specs[0]
        inner = specs[1]
        a = np.array(outer[2])*(1-sum(map(abs, outer[4])))
        for shell in specs[1:3] if len(specs) == 3 else (inner,):
            upper = np.abs(shell[3])+np.array(shell[2])*(1+sum(map(abs, shell[4])))
            assert np.linalg.norm(upper/a) < 1
        if len(specs) == 3:
            _, _, axes_a, center_a, harm_a, _ = specs[1]
            _, _, axes_b, center_b, harm_b, _ = specs[2]
            assert center_a[0]+axes_a[0]*(1+sum(map(abs, harm_a))) < center_b[0]-axes_b[0]*(1+sum(map(abs, harm_b)))
        else:
            island, root = specs[2:]
            inside_bound = np.abs(np.array(island[3])-inner[3])+np.array(island[2])*(1+sum(map(abs, island[4])))
            assert np.linalg.norm(inside_bound/(np.array(inner[2])*(1-sum(map(abs, inner[4]))))) < 1
            assert root[3][0]-root[2][0]*(1+sum(map(abs, root[4]))) > outer[2][0]*(1+sum(map(abs, outer[4])))


def test_owned_arrays_metadata_and_repeated_generation_have_no_mutable_alias(cohort):
    repeated = controls.fixtures()
    for row, again in zip(cohort, repeated):
        assert isinstance(row[2], MappingProxyType)
        assert row[2]["source_array_sha256"] == again[2]["source_array_sha256"]
        for array in (*row[1], row[2]["face_components"], row[2]["expected_signs"], row[2]["expected_parents"]):
            with pytest.raises(ValueError):
                array.flags.writeable = True
        assert not np.shares_memory(row[1][0], again[1][0])


def test_decimal_context_and_frozen_source_mutation_fail_without_new_cohort(monkeypatch):
    precision = getcontext().prec
    controls._sincos.cache_clear()
    controls._sincos(1, 30)
    assert getcontext().prec == precision
    original = controls.radial_surface
    def altered(*args):
        v, f = original(*args)
        v = v.copy(); v[0, 0] = np.nextafter(v[0, 0], np.inf)
        return v, f
    monkeypatch.setattr(controls, "radial_surface", altered)
    with pytest.raises(ValueError, match="never resample"):
        controls.fixtures()


def test_no_historical_geometry_or_native_model_dependencies():
    source = PATH.read_text()
    assert all(word not in source for word in ("import trimesh", "subprocess", "oriented_solid_controls import", "mesh_conditioned_geometry"))
    assert controls.SPEC["resample"] is False
    assert controls.SPEC["target_faces"] == controls.SPEC["target_vertices"] == 4096
    assert controls.SPEC["chamfer_diagonal_limit"] == .01 and controls.SPEC["each_shell_and_net_volume_limit"] == .05
